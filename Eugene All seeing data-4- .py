#!/usr/bin/env python3
# unified_telemetry_daemon.py
# Cross-platform, single-file telemetry daemon with:
# - Auto-loader
# - FS + network + resource tracking
# - Full connection history (1-year retention)
# - IP ledger (IP -> hostname -> processes/directions/ports/bytes)
# - Aggressive risk scoring for IPs (defensive, local-only)
# - Per-process network behavior profiles
# - Call-home beacon
# - Console HUD

import sys
import os
import json
import time
import threading
import subprocess
import platform
import queue
import socket
from datetime import datetime, timedelta

# -----------------------------
# 0. Auto-loader for libraries
# -----------------------------

REQUIRED_LIBS = [
    "psutil",       # process, net, disk, memory
    "watchdog",     # filesystem events
    "requests",     # HTTPS call-home
]

def ensure_dependencies():
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)

    if not missing:
        return

    print(f"[AUTOLOADER] Missing libraries detected: {missing}")
    print("[AUTOLOADER] Attempting to install via pip...")

    py_exe = sys.executable or "python"
    cmd = [py_exe, "-m", "pip", "install"] + missing
    try:
        subprocess.check_call(cmd)
        print("[AUTOLOADER] Installation complete. Continuing...")
    except Exception as e:
        print(f"[AUTOLOADER] Failed to install dependencies: {e}")
        print("[AUTOLOADER] Please install manually and re-run.")
        sys.exit(1)

ensure_dependencies()

import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests

# -----------------------------
# 1. Config & paths
# -----------------------------

APP_NAME = "unified_telemetry_daemon"
STATE_DIR = os.path.join(os.path.expanduser("~"), f".{APP_NAME}")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
LOG_FILE = os.path.join(STATE_DIR, "events.log")

os.makedirs(STATE_DIR, exist_ok=True)

# Call-home configuration (defensive telemetry to your own collector)
HOME_ENABLED = True          # set False to disable remote reporting
HOME_ENDPOINT = "https://your-telemetry-endpoint.example.com/api/events"
HOME_API_KEY = "CHANGE_ME_SECRET"  # replace with your key/token

HOME_TIMEOUT = 5.0           # seconds
HOME_MAX_QUEUE = 2000        # max events queued
HOME_BATCH_SIZE = 50         # events per POST
HOME_INTERVAL = 2.0          # seconds between send attempts

# Connection history retention
CONN_RETENTION_DAYS = 365

# -----------------------------
# 2. Utility helpers
# -----------------------------

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def parse_iso(ts):
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)

def safe_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def append_log(event):
    event = dict(event)
    event.setdefault("ts", now_iso())
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")

def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

# -----------------------------
# 3. Call-home beacon client
# -----------------------------

class BeaconClient(threading.Thread):
    def __init__(self, enabled=True):
        super().__init__(daemon=True)
        self.enabled = enabled and bool(HOME_ENDPOINT)
        self.q = queue.Queue(maxsize=HOME_MAX_QUEUE)
        self.session = requests.Session()

    def enqueue(self, event):
        if not self.enabled:
            return
        try:
            self.q.put_nowait(event)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
                self.q.put_nowait(event)
            except queue.Empty:
                pass

    def run(self):
        if not self.enabled:
            append_log({"type": "beacon_disabled"})
            return

        append_log({"type": "beacon_started", "endpoint": HOME_ENDPOINT})
        while True:
            try:
                self.flush_batch()
            except Exception as e:
                append_log({"type": "error", "source": "BeaconClient.flush_batch", "error": str(e)})
            time.sleep(HOME_INTERVAL)

    def flush_batch(self):
        batch = []
        while len(batch) < HOME_BATCH_SIZE:
            try:
                ev = self.q.get_nowait()
                batch.append(ev)
            except queue.Empty:
                break

        if not batch:
            return

        payload = {
            "host": platform.node(),
            "os": platform.platform(),
            "app": APP_NAME,
            "ts": now_iso(),
            "events": batch,
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": HOME_API_KEY,
        }

        try:
            resp = self.session.post(
                HOME_ENDPOINT,
                data=json.dumps(payload),
                headers=headers,
                timeout=HOME_TIMEOUT,
            )
            append_log({
                "type": "beacon_sent",
                "count": len(batch),
                "status_code": resp.status_code,
            })
        except Exception as e:
            append_log({
                "type": "beacon_failed",
                "count": len(batch),
                "error": str(e),
            })

# -----------------------------
# 4. Persistent state model
# -----------------------------

class PersistentState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {
            "meta": {
                "app_name": APP_NAME,
                "created": now_iso(),
                "last_update": now_iso(),
                "host": platform.node(),
                "os": platform.platform(),
            },
            "counters": {
                "inbound_bytes": 0,
                "outbound_bytes": 0,
                "files_created": 0,
                "files_modified": 0,
                "files_deleted": 0,
            },
            "processes": {},   # pid -> {name, inbound_bytes, outbound_bytes, open_files, last_seen, ips, ports}
            "files": {},       # path -> {size, last_event, last_ts}
            "resources": {
                "memory": {},
                "disk": {},
                "net": {},
            },
            "ip_ledger": {},   # ip -> {...}
            "connections": [], # full connection history (pruned by retention)
        }
        self.load()

    def load(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self.lock:
                self.state.update(data)
        except Exception as e:
            append_log({"type": "error", "source": "PersistentState.load", "error": str(e)})

    def save(self):
        with self.lock:
            self.state["meta"]["last_update"] = now_iso()
            safe_write_json(STATE_FILE, self.state)

    def update_counter(self, key, delta):
        with self.lock:
            self.state["counters"][key] = self.state["counters"].get(key, 0) + delta

    def update_process_flow(self, pid, name, inbound_delta=0, outbound_delta=0):
        with self.lock:
            proc = self.state["processes"].setdefault(str(pid), {
                "name": name,
                "inbound_bytes": 0,
                "outbound_bytes": 0,
                "open_files": 0,
                "last_seen": now_iso(),
                "ips": {},   # ip -> count
                "ports": {}, # port -> count
            })
            proc["inbound_bytes"] += inbound_delta
            proc["outbound_bytes"] += outbound_delta
            proc["last_seen"] = now_iso()

    def update_process_net_profile(self, pid, ip, port):
        with self.lock:
            proc = self.state["processes"].get(str(pid))
            if not proc:
                return
            if ip:
                proc["ips"][ip] = proc["ips"].get(ip, 0) + 1
            if port:
                proc["ports"][str(port)] = proc["ports"].get(str(port), 0) + 1

    def update_file_event(self, path, event_type, size=None):
        with self.lock:
            f = self.state["files"].setdefault(path, {
                "size": size or 0,
                "last_event": event_type,
                "last_ts": now_iso(),
            })
            if size is not None:
                f["size"] = size
            f["last_event"] = event_type
            f["last_ts"] = now_iso()

    def update_resources(self, memory_info, disk_info, net_info):
        with self.lock:
            self.state["resources"]["memory"] = memory_info
            self.state["resources"]["disk"] = disk_info
            self.state["resources"]["net"] = net_info

    def _ensure_ip_entry(self, ip, ts):
        entry = self.state["ip_ledger"].get(ip)
        if entry is None:
            hostname = None
            try:
                hostname = socket.gethostbyaddr(ip)[0]
            except Exception:
                hostname = None
            entry = {
                "hostname": hostname,
                "first_seen": ts,
                "last_seen": ts,
                "directions": [],
                "processes": {},  # pid -> {name, last_seen, count}
                "ports": {},      # port -> count
                "bytes_in": 0,
                "bytes_out": 0,
                "bytes_total": 0,
                "risk_score": 0,
                "risk_reasons": [],
                "first_flagged": None,
                "last_flagged": None,
            }
            self.state["ip_ledger"][ip] = entry
        return entry

    def update_ip_ledger(self, ip, direction, pid, proc_name, port=None, ts=None):
        if not ip:
            return
        ts = ts or now_iso()
        with self.lock:
            entry = self._ensure_ip_entry(ip, ts)
            entry["last_seen"] = ts
            if direction and direction not in entry["directions"]:
                entry["directions"].append(direction)

            if port is not None:
                entry["ports"][str(port)] = entry["ports"].get(str(port), 0) + 1

            p = entry["processes"].get(str(pid))
            if p is None:
                p = {
                    "name": proc_name,
                    "last_seen": ts,
                    "count": 0,
                }
                entry["processes"][str(pid)] = p
            p["last_seen"] = ts
            p["count"] += 1

            self._evaluate_ip_risk(ip, entry)

    def add_ip_bytes(self, ip, in_bytes, out_bytes, ts=None):
        if not ip:
            return
        ts = ts or now_iso()
        with self.lock:
            entry = self._ensure_ip_entry(ip, ts)
            entry["bytes_in"] += max(0, in_bytes)
            entry["bytes_out"] += max(0, out_bytes)
            entry["bytes_total"] = entry["bytes_in"] + entry["bytes_out"]
            entry["last_seen"] = ts
            self._evaluate_ip_risk(ip, entry)

    def _evaluate_ip_risk(self, ip, entry):
        """
        Aggressive, heuristic, defensive-only risk scoring.
        """
        score = 0
        reasons = []

        proc_count = len(entry.get("processes", {}))
        port_count = len(entry.get("ports", {}))
        dirs = entry.get("directions", [])
        bytes_total = entry.get("bytes_total", 0)
        hostname = entry.get("hostname")

        # Many processes talking to same IP
        if proc_count >= 5:
            score += 2
            reasons.append(f"talked_to_by_{proc_count}_processes")

        # Many ports
        if port_count >= 10:
            score += 2
            reasons.append(f"many_ports_{port_count}")

        # Both inbound-like and outbound
        if "outbound" in dirs and "listen" in dirs:
            score += 2
            reasons.append("both_outbound_and_listen")

        # No reverse DNS
        if hostname is None:
            score += 1
            reasons.append("no_reverse_dns")

        # High total bytes (heuristic)
        if bytes_total > 500 * 1024 * 1024:  # > 500MB
            score += 3
            reasons.append("high_volume_bytes")

        # Aggressive: any combination of above pushes it up
        if proc_count >= 10 or port_count >= 20:
            score += 3
            reasons.append("extreme_fanout")

        # Update entry
        prev_score = entry.get("risk_score", 0)
        entry["risk_score"] = score
        entry["risk_reasons"] = reasons

        now_ts = now_iso()
        if score > 0:
            if not entry.get("first_flagged"):
                entry["first_flagged"] = now_ts
            entry["last_flagged"] = now_ts

    def add_connection_event(self, ev):
        cutoff = datetime.utcnow() - timedelta(days=CONN_RETENTION_DAYS)
        with self.lock:
            self.state["connections"].append(ev)
            pruned = []
            for e in self.state["connections"]:
                try:
                    ts = parse_iso(e.get("ts", now_iso()))
                except Exception:
                    ts = datetime.utcnow()
                if ts >= cutoff:
                    pruned.append(e)
            self.state["connections"] = pruned

# -----------------------------
# 5. Filesystem watcher
# -----------------------------

class TelemetryFSHandler(FileSystemEventHandler):
    def __init__(self, state: PersistentState, beacon: BeaconClient, root_label="fs"):
        super().__init__()
        self.state = state
        self.beacon = beacon
        self.root_label = root_label

    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        size = os.path.getsize(path) if os.path.exists(path) else 0
        self.state.update_file_event(path, "created", size)
        self.state.update_counter("files_created", 1)
        ev = {
            "type": "fs_created",
            "path": path,
            "size": size,
            "ts": now_iso(),
        }
        append_log(ev)
        self.beacon.enqueue(ev)

    def on_modified(self, event):
        if event.is_directory:
            return
        path = event.src_path
        size = os.path.getsize(path) if os.path.exists(path) else 0
        self.state.update_file_event(path, "modified", size)
        self.state.update_counter("files_modified", 1)
        ev = {
            "type": "fs_modified",
            "path": path,
            "size": size,
            "ts": now_iso(),
        }
        append_log(ev)
        self.beacon.enqueue(ev)

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = event.src_path
        self.state.update_file_event(path, "deleted", 0)
        self.state.update_counter("files_deleted", 1)
        ev = {
            "type": "fs_deleted",
            "path": path,
            "ts": now_iso(),
        }
        append_log(ev)
        self.beacon.enqueue(ev)

class FileSystemMonitor(threading.Thread):
    def __init__(self, state: PersistentState, beacon: BeaconClient, paths=None):
        super().__init__(daemon=True)
        self.state = state
        self.beacon = beacon
        self.paths = paths or [os.path.expanduser("~")]
        self.observer = Observer()

    def run(self):
        try:
            handler = TelemetryFSHandler(self.state, self.beacon)
            for p in self.paths:
                if os.path.exists(p):
                    self.observer.schedule(handler, p, recursive=True)
            self.observer.start()
            ev = {"type": "fs_monitor_started", "paths": self.paths, "ts": now_iso()}
            append_log(ev)
            self.beacon.enqueue(ev)
            while True:
                time.sleep(1)
        except Exception as e:
            append_log({"type": "error", "source": "FileSystemMonitor.run", "error": str(e), "ts": now_iso()})
        finally:
            self.observer.stop()
            self.observer.join()

# -----------------------------
# 6. Network & process monitor
# -----------------------------

class NetProcessMonitor(threading.Thread):
    """
    Uses psutil to:
    - Track global net I/O
    - Enumerate all inet connections
    - Maintain IP ledger (ports, processes, bytes, risk)
    - Store full connection history (1-year retention)
    - Update per-process network profiles
    """
    def __init__(self, state: PersistentState, beacon: BeaconClient, interval=1.0):
        super().__init__(daemon=True)
        self.state = state
        self.beacon = beacon
        self.interval = interval
        self.prev_net_io = psutil.net_io_counters() if psutil.net_io_counters() else None

    def run(self):
        ev = {"type": "net_monitor_started", "ts": now_iso()}
        append_log(ev)
        self.beacon.enqueue(ev)
        while True:
            try:
                self.sample()
            except Exception as e:
                append_log({"type": "error", "source": "NetProcessMonitor.sample", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)

    def sample(self):
        # Global net I/O
        net_io = psutil.net_io_counters()
        in_delta = 0
        out_delta = 0
        if net_io and self.prev_net_io:
            in_delta = net_io.bytes_recv - self.prev_net_io.bytes_recv
            out_delta = net_io.bytes_sent - self.prev_net_io.bytes_sent
            if in_delta < 0: in_delta = 0
            if out_delta < 0: out_delta = 0
            self.state.update_counter("inbound_bytes", in_delta)
            self.state.update_counter("outbound_bytes", out_delta)
            ev = {
                "type": "net_global",
                "inbound_delta": in_delta,
                "outbound_delta": out_delta,
                "ts": now_iso(),
            }
            append_log(ev)
            self.beacon.enqueue(ev)
        self.prev_net_io = net_io

        # Collect all remote IPs this tick for approximate byte distribution
        remote_ips = set()
        proc_conns = []

        for proc in psutil.process_iter(attrs=["pid", "name"]):
            pid = proc.info["pid"]
            name = proc.info.get("name") or f"pid_{pid}"
            try:
                conns = proc.connections(kind="inet")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            try:
                open_files = proc.open_files()
                open_files_count = len(open_files)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                open_files_count = 0

            self.state.update_process_flow(pid, name, 0, 0)
            with self.state.lock:
                self.state.state["processes"][str(pid)]["open_files"] = open_files_count

            for c in conns:
                ts = now_iso()
                laddr_ip = c.laddr.ip if c.laddr else None
                laddr_port = c.laddr.port if c.laddr else None
                raddr_ip = c.raddr.ip if c.raddr else None
                raddr_port = c.raddr.port if c.raddr else None

                # Direction heuristic
                if c.status == psutil.CONN_LISTEN:
                    direction = "listen"
                    remote_ip = None
                    remote_port = None
                else:
                    if raddr_ip:
                        direction = "outbound"
                        remote_ip = raddr_ip
                        remote_port = raddr_port
                    else:
                        direction = "unknown"
                        remote_ip = None
                        remote_port = None

                if remote_ip:
                    remote_ips.add(remote_ip)
                    self.state.update_ip_ledger(remote_ip, direction, pid, name, port=remote_port, ts=ts)
                    self.state.update_process_net_profile(pid, remote_ip, remote_port)

                ev = {
                    "type": "net_conn",
                    "pid": pid,
                    "proc_name": name,
                    "status": c.status,
                    "direction": direction,
                    "laddr": f"{laddr_ip}:{laddr_port}" if laddr_ip else None,
                    "raddr": f"{raddr_ip}:{raddr_port}" if raddr_ip else None,
                    "ts": ts,
                }
                append_log(ev)
                self.beacon.enqueue(ev)
                self.state.add_connection_event(ev)

                proc_conns.append((pid, name, remote_ip, remote_port))

        # Approximate per-IP bytes: distribute global deltas across active remote IPs
        if remote_ips and (in_delta > 0 or out_delta > 0):
            in_share = in_delta / max(1, len(remote_ips))
            out_share = out_delta / max(1, len(remote_ips))
            for ip in remote_ips:
                self.state.add_ip_bytes(ip, in_share, out_share, ts=now_iso())

# -----------------------------
# 7. Resource monitor
# -----------------------------

class ResourceMonitor(threading.Thread):
    def __init__(self, state: PersistentState, beacon: BeaconClient, interval=2.0):
        super().__init__(daemon=True)
        self.state = state
        self.beacon = beacon
        self.interval = interval

    def run(self):
        ev = {"type": "resource_monitor_started", "ts": now_iso()}
        append_log(ev)
        self.beacon.enqueue(ev)
        while True:
            try:
                self.sample()
            except Exception as e:
                append_log({"type": "error", "source": "ResourceMonitor.sample", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)

    def sample(self):
        vm = psutil.virtual_memory()
        disks = {}
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                disks[part.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                }
            except PermissionError:
                continue

        net = {}
        net_io = psutil.net_io_counters(pernic=True)
        for nic, stats in net_io.items():
            net[nic] = {
                "bytes_sent": stats.bytes_sent,
                "bytes_recv": stats.bytes_recv,
                "packets_sent": stats.packets_sent,
                "packets_recv": stats.packets_recv,
            }

        mem_info = {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        }

        self.state.update_resources(mem_info, disks, net)
        ev = {
            "type": "resource_sample",
            "memory": mem_info,
            "disk_mounts": list(disks.keys()),
            "nics": list(net.keys()),
            "ts": now_iso(),
        }
        append_log(ev)
        self.beacon.enqueue(ev)

# -----------------------------
# 8. State flusher
# -----------------------------

class StateFlusher(threading.Thread):
    def __init__(self, state: PersistentState, interval=5.0):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval

    def run(self):
        ev = {"type": "state_flusher_started", "ts": now_iso()}
        append_log(ev)
        while True:
            try:
                self.state.save()
            except Exception as e:
                append_log({"type": "error", "source": "StateFlusher.run", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)

# -----------------------------
# 9. Console HUD
# -----------------------------

class ConsoleHUD(threading.Thread):
    def __init__(self, state: PersistentState, interval=2.0):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval

    def run(self):
        while True:
            self.render()
            time.sleep(self.interval)

    def render(self):
        os_name = self.state.state["meta"]["os"]
        host = self.state.state["meta"]["host"]
        with self.state.lock:
            counters = dict(self.state.state["counters"])
            resources = dict(self.state.state["resources"])
            processes = dict(self.state.state["processes"])
            ip_ledger = dict(self.state.state.get("ip_ledger", {}))

        inbound = counters.get("inbound_bytes", 0)
        outbound = counters.get("outbound_bytes", 0)
        files_created = counters.get("files_created", 0)
        files_modified = counters.get("files_modified", 0)
        files_deleted = counters.get("files_deleted", 0)

        mem = resources.get("memory", {})
        mem_used = mem.get("used", 0)
        mem_total = mem.get("total", 0)
        mem_pct = mem.get("percent", 0)

        top_procs = sorted(
            processes.items(),
            key=lambda kv: kv[1].get("outbound_bytes", 0),
            reverse=True
        )[:5]

        # Top IPs by risk score
        ip_summary = []
        for ip, entry in ip_ledger.items():
            risk = entry.get("risk_score", 0)
            ip_summary.append((ip, entry, risk))
        ip_summary = sorted(ip_summary, key=lambda x: x[2], reverse=True)[:5]

        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")

        print(f"[{APP_NAME}] Host: {host} | OS: {os_name} | {now_iso()}")
        print("-" * 110)
        print(f" INBOUND:  {human_bytes(inbound):>10}   OUTBOUND: {human_bytes(outbound):>10}")
        print(f" FILES:    +{files_created} created, +{files_modified} modified, +{files_deleted} deleted")
        print(f" MEMORY:   {human_bytes(mem_used)} / {human_bytes(mem_total)} ({mem_pct:.1f}%)")
        print("-" * 110)
        print(" Top processes by outbound bytes:")
        for pid, info in top_procs:
            name = info.get("name", "?")
            outb = info.get("outbound_bytes", 0)
            ofc = info.get("open_files", 0)
            print(f"  PID {pid:>6} | {name:<25} | OUT: {human_bytes(outb):>10} | open_files: {ofc}")
        print("-" * 110)
        print(" Top IPs by risk score:")
        for ip, entry, risk in ip_summary:
            hostname = entry.get("hostname") or "-"
            dirs = ",".join(entry.get("directions", []))
            bytes_total = entry.get("bytes_total", 0)
            reasons = ";".join(entry.get("risk_reasons", [])) or "-"
            print(f"  {ip:<15} | risk:{risk:<3} | {hostname:<30} | dirs:{dirs:<15} | bytes:{human_bytes(bytes_total):>10}")
            print(f"      reasons: {reasons}")
        print("-" * 110)
        print(f" State file: {STATE_FILE}")
        print(f" Log file:   {LOG_FILE}")
        print(f" Call-home:  {'ENABLED' if HOME_ENABLED else 'DISABLED'} -> {HOME_ENDPOINT if HOME_ENABLED else '-'}")
        print(" (Press Ctrl+C to exit)")

# -----------------------------
# 10. Main bootstrap
# -----------------------------

def main():
    state = PersistentState()
    beacon = BeaconClient(enabled=HOME_ENABLED)

    fs_monitor = FileSystemMonitor(state, beacon)
    net_monitor = NetProcessMonitor(state, beacon)
    res_monitor = ResourceMonitor(state, beacon)
    flusher = StateFlusher(state)
    hud = ConsoleHUD(state)

    beacon.start()
    fs_monitor.start()
    net_monitor.start()
    res_monitor.start()
    flusher.start()
    hud.start()

    append_log({"type": "daemon_started", "ts": now_iso()})

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        append_log({"type": "daemon_stopped", "reason": "KeyboardInterrupt", "ts": now_iso()})
        print("\nShutting down...")

if __name__ == "__main__":
    main()