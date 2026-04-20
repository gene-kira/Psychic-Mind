#!/usr/bin/env python3
# unified_telemetry_daemon.py
# Cross-platform, single-file defensive telemetry daemon with:
# - Auto-loader
# - FS + network + resource tracking
# - Full connection history (1-year retention)
# - IP ledger (IP -> hostname -> processes/directions/ports/bytes/risk)
# - Per-process network behavior profiles
# - Local network scanner + rogue host detection
# - Auto-blocker via local firewall (per-OS, toggleable)
# - Anomaly engine (entropy, rare events, bursts, time-of-day deviation)
# - Altered states: NORMAL / ALERT / HUNT / LOCKDOWN
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
import math
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

# Local network scan
LAN_SCAN_INTERVAL = 60.0     # seconds

# Auto-blocker toggle (local firewall only)
AUTO_BLOCK_ENABLED = True

# Anomaly engine config
ANOMALY_WINDOW_SECONDS = 300.0   # rolling window for bursts
ANOMALY_PORT_ENTROPY_THRESHOLD = 2.5
ANOMALY_IP_ENTROPY_THRESHOLD = 2.5
ANOMALY_RARE_EVENT_WEIGHT = 2.0
ANOMALY_BURST_WEIGHT = 2.0
ANOMALY_TIME_DEVIATION_WEIGHT = 1.5

# Altered states thresholds
STATE_NORMAL = "NORMAL"
STATE_ALERT = "ALERT"
STATE_HUNT = "HUNT"
STATE_LOCKDOWN = "LOCKDOWN"

STATE_THRESHOLDS = {
    STATE_NORMAL: 0.0,
    STATE_ALERT: 5.0,
    STATE_HUNT: 10.0,
    STATE_LOCKDOWN: 20.0,
}

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

def run_cmd(cmd):
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def hour_of_day(dt=None):
    if dt is None:
        dt = datetime.utcnow()
    return dt.hour + dt.minute / 60.0

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
            "processes": {},   # pid -> {name, inbound_bytes, outbound_bytes, open_files, last_seen, ips, ports, anomalies}
            "files": {},       # path -> {size, last_event, last_ts}
            "resources": {
                "memory": {},
                "disk": {},
                "net": {},
            },
            "ip_ledger": {},   # ip -> {...}
            "connections": [], # full connection history (pruned by retention)
            "lan_hosts": {},   # ip -> {mac, hostname, first_seen, last_seen, vendor_guess, risk_score, reasons, blocked}
            "anomaly": {
                "events": [],          # recent anomaly events
                "pressure": 0.0,       # rolling anomaly pressure
                "state": STATE_NORMAL, # NORMAL/ALERT/HUNT/LOCKDOWN
            },
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

    def _ensure_process(self, pid, name):
        with self.lock:
            proc = self.state["processes"].setdefault(str(pid), {
                "name": name,
                "inbound_bytes": 0,
                "outbound_bytes": 0,
                "open_files": 0,
                "last_seen": now_iso(),
                "ips": {},   # ip -> count
                "ports": {}, # port -> count
                "anomalies": [],  # recent anomaly tags
                "hours_seen": [], # hours of day seen
            })
            return proc

    def update_process_flow(self, pid, name, inbound_delta=0, outbound_delta=0):
        proc = self._ensure_process(pid, name)
        with self.lock:
            proc["inbound_bytes"] += inbound_delta
            proc["outbound_bytes"] += outbound_delta
            proc["last_seen"] = now_iso()

    def update_process_net_profile(self, pid, ip, port):
        proc = self._ensure_process(pid, f"pid_{pid}")
        with self.lock:
            if ip:
                proc["ips"][ip] = proc["ips"].get(ip, 0) + 1
            if port:
                proc["ports"][str(port)] = proc["ports"].get(str(port), 0) + 1
            h = hour_of_day()
            proc["hours_seen"].append(h)
            if len(proc["hours_seen"]) > 500:
                proc["hours_seen"] = proc["hours_seen"][-500:]

    def add_process_anomaly(self, pid, tag, score):
        with self.lock:
            proc = self.state["processes"].get(str(pid))
            if not proc:
                return
            proc["anomalies"].append({"tag": tag, "score": score, "ts": now_iso()})
            if len(proc["anomalies"]) > 100:
                proc["anomalies"] = proc["anomalies"][-100:]

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
        score = 0
        reasons = []

        proc_count = len(entry.get("processes", {}))
        port_count = len(entry.get("ports", {}))
        dirs = entry.get("directions", [])
        bytes_total = entry.get("bytes_total", 0)
        hostname = entry.get("hostname")

        if proc_count >= 5:
            score += 2
            reasons.append(f"talked_to_by_{proc_count}_processes")

        if port_count >= 10:
            score += 2
            reasons.append(f"many_ports_{port_count}")

        if "outbound" in dirs and "listen" in dirs:
            score += 2
            reasons.append("both_outbound_and_listen")

        if hostname is None:
            score += 1
            reasons.append("no_reverse_dns")

        if bytes_total > 500 * 1024 * 1024:
            score += 3
            reasons.append("high_volume_bytes")

        if proc_count >= 10 or port_count >= 20:
            score += 3
            reasons.append("extreme_fanout")

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

    def update_lan_host(self, ip, mac=None, hostname=None):
        ts = now_iso()
        with self.lock:
            host = self.state["lan_hosts"].get(ip)
            if host is None:
                host = {
                    "ip": ip,
                    "mac": mac,
                    "hostname": hostname,
                    "first_seen": ts,
                    "last_seen": ts,
                    "vendor_guess": None,
                    "risk_score": 0,
                    "reasons": [],
                    "blocked": False,
                }
                self.state["lan_hosts"][ip] = host
            else:
                host["last_seen"] = ts
                if mac and not host.get("mac"):
                    host["mac"] = mac
                if hostname and not host.get("hostname"):
                    host["hostname"] = hostname

            self._evaluate_lan_host_risk(host)

    def _evaluate_lan_host_risk(self, host):
        score = 0
        reasons = []

        ip = host.get("ip")
        mac = host.get("mac")
        hostname = host.get("hostname")

        if hostname is None:
            score += 1
            reasons.append("no_hostname")

        if mac is None:
            score += 1
            reasons.append("no_mac")

        try:
            first_seen = parse_iso(host.get("first_seen", now_iso()))
            age = datetime.utcnow() - first_seen
            if age < timedelta(minutes=10):
                score += 2
                reasons.append("new_host_recent")
        except Exception:
            pass

        ip_ledger_entry = self.state["ip_ledger"].get(ip, {})
        ip_risk = ip_ledger_entry.get("risk_score", 0)
        if ip_risk >= 3:
            score += 3
            reasons.append(f"ip_ledger_risk_{ip_risk}")

        host["risk_score"] = score
        host["reasons"] = reasons

    def add_anomaly_event(self, ev, score):
        with self.lock:
            a = self.state["anomaly"]
            ev = dict(ev)
            ev["score"] = score
            ev["ts"] = now_iso()
            a["events"].append(ev)
            cutoff = datetime.utcnow() - timedelta(seconds=ANOMALY_WINDOW_SECONDS)
            filtered = []
            for e in a["events"]:
                try:
                    ts = parse_iso(e["ts"])
                except Exception:
                    ts = datetime.utcnow()
                if ts >= cutoff:
                    filtered.append(e)
            a["events"] = filtered
            pressure = sum(e.get("score", 0.0) for e in a["events"])
            a["pressure"] = pressure
            a["state"] = self._derive_state_from_pressure(pressure)

    def _derive_state_from_pressure(self, pressure):
        if pressure >= STATE_THRESHOLDS[STATE_LOCKDOWN]:
            return STATE_LOCKDOWN
        if pressure >= STATE_THRESHOLDS[STATE_HUNT]:
            return STATE_HUNT
        if pressure >= STATE_THRESHOLDS[STATE_ALERT]:
            return STATE_ALERT
        return STATE_NORMAL

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
# 6. Anomaly engine helpers
# -----------------------------

def entropy_from_counts(counts_dict):
    total = sum(counts_dict.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts_dict.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def time_deviation_score(hours_list):
    if not hours_list:
        return 0.0
    mean = sum(hours_list) / len(hours_list)
    now_h = hour_of_day()
    diff = abs(now_h - mean)
    diff = min(diff, 24 - diff)
    return diff / 12.0

# -----------------------------
# 7. Network & process monitor
# -----------------------------

class NetProcessMonitor(threading.Thread):
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

        remote_ips = set()

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

            self._evaluate_process_anomalies(pid)

        if remote_ips and (in_delta > 0 or out_delta > 0):
            in_share = in_delta / max(1, len(remote_ips))
            out_share = out_delta / max(1, len(remote_ips))
            for ip in remote_ips:
                self.state.add_ip_bytes(ip, in_share, out_share, ts=now_iso())

    def _evaluate_process_anomalies(self, pid):
        with self.state.lock:
            proc = self.state.state["processes"].get(str(pid))
            if not proc:
                return
            ips = dict(proc.get("ips", {}))
            ports = dict(proc.get("ports", {}))
            hours_seen = list(proc.get("hours_seen", []))

        port_entropy = entropy_from_counts(ports)
        ip_entropy = entropy_from_counts(ips)
        time_dev = time_deviation_score(hours_seen)

        score = 0.0
        tags = []

        if port_entropy > ANOMALY_PORT_ENTROPY_THRESHOLD:
            score += ANOMALY_RARE_EVENT_WEIGHT
            tags.append("high_port_entropy")

        if ip_entropy > ANOMALY_IP_ENTROPY_THRESHOLD:
            score += ANOMALY_RARE_EVENT_WEIGHT
            tags.append("high_ip_entropy")

        if time_dev > 0.5:
            score += ANOMALY_TIME_DEVIATION_WEIGHT
            tags.append("time_of_day_deviation")

        if len(ips) > 20 or len(ports) > 20:
            score += ANOMALY_BURST_WEIGHT
            tags.append("fanout_burst")

        if score > 0:
            for t in tags:
                self.state.add_process_anomaly(pid, t, score)
            ev = {
                "type": "proc_anomaly",
                "pid": pid,
                "proc_name": proc.get("name", "?"),
                "tags": tags,
                "port_entropy": port_entropy,
                "ip_entropy": ip_entropy,
                "time_dev": time_dev,
            }
            append_log(ev)
            self.beacon.enqueue(ev)
            self.state.add_anomaly_event(ev, score)

# -----------------------------
# 8. Resource monitor
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
# 9. Local network scanner + auto-blocker
# -----------------------------

class LocalNetworkScanner(threading.Thread):
    def __init__(self, state: PersistentState, beacon: BeaconClient, interval=LAN_SCAN_INTERVAL):
        super().__init__(daemon=True)
        self.state = state
        self.beacon = beacon
        self.interval = interval

    def run(self):
        ev = {"type": "lan_scanner_started", "ts": now_iso()}
        append_log(ev)
        self.beacon.enqueue(ev)
        while True:
            try:
                self.scan_once()
            except Exception as e:
                append_log({"type": "error", "source": "LocalNetworkScanner.scan_once", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)

    def scan_once(self):
        os_name = platform.system().lower()
        hosts = {}

        if os_name == "windows":
            cmd = ["arp", "-a"]
        else:
            cmd = ["arp", "-a"]

        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            out = ""

        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if "(" in line and ")" in line:
                try:
                    ip_part = line.split("(")[1].split(")")[0].strip()
                    if " " in ip_part:
                        continue
                    ip = ip_part
                    mac = None
                    if " at " in line:
                        mac = line.split(" at ")[1].split(" ")[0].strip()
                    hosts[ip] = {"ip": ip, "mac": mac}
                except Exception:
                    continue

        for ip, info in hosts.items():
            mac = info.get("mac")
            hostname = None
            try:
                hostname = socket.gethostbyaddr(ip)[0]
            except Exception:
                hostname = None
            self.state.update_lan_host(ip, mac=mac, hostname=hostname)
            ev = {
                "type": "lan_host_seen",
                "ip": ip,
                "mac": mac,
                "hostname": hostname,
                "ts": now_iso(),
            }
            append_log(ev)
            self.beacon.enqueue(ev)

        self._auto_block_rogues()

    def _auto_block_rogues(self):
        if not AUTO_BLOCK_ENABLED:
            return

        os_name = platform.system().lower()
        with self.state.lock:
            lan_hosts = self.state.state.get("lan_hosts", {})

        for ip, host in lan_hosts.items():
            risk = host.get("risk_score", 0)
            blocked = host.get("blocked", False)

            if risk >= 4 and not blocked:
                success = self._block_ip_local(ip, os_name)
                host["blocked"] = success
                reason = ";".join(host.get("reasons", [])) or "unknown"
                ev = {
                    "type": "lan_host_blocked" if success else "lan_host_block_failed",
                    "ip": ip,
                    "risk_score": risk,
                    "reasons": reason,
                    "ts": now_iso(),
                }
                append_log(ev)
                self.beacon.enqueue(ev)

    def _block_ip_local(self, ip, os_name):
        if os_name == "windows":
            cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                "name=UTD_Block_" + ip,
                "dir=out", "action=block", "remoteip=" + ip
            ]
            ok1 = run_cmd(cmd)
            cmd2 = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                "name=UTD_Block_" + ip + "_in",
                "dir=in", "action=block", "remoteip=" + ip
            ]
            ok2 = run_cmd(cmd2)
            return ok1 and ok2
        elif os_name == "linux":
            cmd1 = ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]
            cmd2 = ["iptables", "-A", "OUTPUT", "-d", ip, "-j", "DROP"]
            ok1 = run_cmd(cmd1)
            ok2 = run_cmd(cmd2)
            return ok1 and ok2
        elif os_name == "darwin":
            append_log({
                "type": "pf_block_intent",
                "ip": ip,
                "note": "Add to pf.conf block rules manually for persistent block",
                "ts": now_iso(),
            })
            return False
        else:
            return False

# -----------------------------
# 10. State flusher
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
# 11. Console HUD
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
            lan_hosts = dict(self.state.state.get("lan_hosts", {}))
            anomaly = dict(self.state.state.get("anomaly", {}))

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

        ip_summary = []
        for ip, entry in ip_ledger.items():
            risk = entry.get("risk_score", 0)
            ip_summary.append((ip, entry, risk))
        ip_summary = sorted(ip_summary, key=lambda x: x[2], reverse=True)[:5]

        lan_summary = []
        for ip, host in lan_hosts.items():
            risk = host.get("risk_score", 0)
            lan_summary.append((ip, host, risk))
        lan_summary = sorted(lan_summary, key=lambda x: x[2], reverse=True)[:5]

        anomaly_state = anomaly.get("state", STATE_NORMAL)
        anomaly_pressure = anomaly.get("pressure", 0.0)

        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")

        print(f"[{APP_NAME}] Host: {host} | OS: {os_name} | {now_iso()}")
        print(f" STATE: {anomaly_state} | PRESSURE: {anomaly_pressure:.2f}")
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
        print(" Top LAN hosts by risk score:")
        for ip, host, risk in lan_summary:
            mac = host.get("mac") or "-"
            hostname = host.get("hostname") or "-"
            blocked = host.get("blocked", False)
            reasons = ";".join(host.get("reasons", [])) or "-"
            print(f"  {ip:<15} | risk:{risk:<3} | mac:{mac:<17} | host:{hostname:<25} | blocked:{blocked}")
            print(f"      reasons: {reasons}")
        print("-" * 110)
        print(f" State file: {STATE_FILE}")
        print(f" Log file:   {LOG_FILE}")
        print(f" Call-home:  {'ENABLED' if HOME_ENABLED else 'DISABLED'} -> {HOME_ENDPOINT if HOME_ENABLED else '-'}")
        print(f" Auto-block: {'ENABLED' if AUTO_BLOCK_ENABLED else 'DISABLED'}")
        print(" (Press Ctrl+C to exit)")

# -----------------------------
# 12. Main bootstrap
# -----------------------------

def main():
    state = PersistentState()
    beacon = BeaconClient(enabled=HOME_ENABLED)

    fs_monitor = FileSystemMonitor(state, beacon)
    net_monitor = NetProcessMonitor(state, beacon)
    res_monitor = ResourceMonitor(state, beacon)
    lan_scanner = LocalNetworkScanner(state, beacon)
    flusher = StateFlusher(state)
    hud = ConsoleHUD(state)

    beacon.start()
    fs_monitor.start()
    net_monitor.start()
    res_monitor.start()
    lan_scanner.start()
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