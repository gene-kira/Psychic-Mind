#!/usr/bin/env python3
# unified_telemetry_daemon_conscious_swarm_ml_predictive_threaded_gui.py

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
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# -----------------------------
# 0. Auto-loader for libraries
# -----------------------------

REQUIRED_LIBS = [
    "psutil",
    "watchdog",
    "requests",
    "PySide6",
    "numpy",
]

OPTIONAL_LIBS = [
    "cupy",
    "bcc",
    "pywin32",
    "scapy",
    "sklearn",
]

def ensure_dependencies():
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)

    if missing:
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

    opt_missing = []
    for lib in OPTIONAL_LIBS:
        try:
            __import__(lib)
        except ImportError:
            opt_missing.append(lib)
    if opt_missing:
        print(f"[AUTOLOADER] Optional libraries not present (features degraded): {opt_missing}")

ensure_dependencies()

import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

try:
    from bcc import BPF
    EBPF_AVAILABLE = True
except Exception:
    EBPF_AVAILABLE = False

try:
    import win32evtlog
    import win32con
    ETW_AVAILABLE = True
except Exception:
    ETW_AVAILABLE = False

try:
    from scapy.all import sniff
    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from PySide6 import QtCore, QtGui, QtWidgets

# -----------------------------
# 1. Config & paths
# -----------------------------

APP_NAME = "unified_telemetry_daemon"
STATE_DIR = os.path.join(os.path.expanduser("~"), f".{APP_NAME}")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
LOG_FILE = os.path.join(STATE_DIR, "events.log")

os.makedirs(STATE_DIR, exist_ok=True)

HOME_ENABLED = True
HOME_ENDPOINT = "https://your-telemetry-endpoint.example.com/api/events"
HOME_API_KEY = "CHANGE_ME_SECRET"

HOME_TIMEOUT = 5.0
HOME_MAX_QUEUE = 2000
HOME_BATCH_SIZE = 50
HOME_INTERVAL = 2.0

CONN_RETENTION_DAYS = 365
LAN_SCAN_INTERVAL = 60.0
AUTO_BLOCK_ENABLED = True

ANOMALY_WINDOW_SECONDS = 300.0
ANOMALY_PORT_ENTROPY_THRESHOLD = 2.5
ANOMALY_IP_ENTROPY_THRESHOLD = 2.5
ANOMALY_RARE_EVENT_WEIGHT = 2.0
ANOMALY_BURST_WEIGHT = 2.0
ANOMALY_TIME_DEVIATION_WEIGHT = 1.5

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

CONSC_BASELINE = "BASELINE"
CONSC_FOCUSED = "FOCUSED"
CONSC_HYPERVIGILANT = "HYPERVIGILANT"
CONSC_DREAM = "DREAM"
CONSC_DISSOCIATION = "DISSOCIATION"
CONSC_EVENT_HORIZON = "EVENT_HORIZON"

REFRESH_INTERVAL_MS = 2000

SUBSYSTEM_FLAGS = {
    "beacon": True,
    "fs_monitor": True,
    "net_monitor": True,
    "res_monitor": True,
    "lan_scanner": True,
    "state_flusher": True,
    "console_hud": False,
    "ebpf_monitor": True,
    "etw_monitor": True,
    "remote_api": True,
    "swarm_sync": True,
    "packet_capture": True,
    "ml_anomaly": True,
    "consciousness": True,
    "missing_detail": True,
}

REMOTE_API_PORT = 8787

SWARM_ENABLED = True
SWARM_NODE_ID = platform.node()
SWARM_PEERS = []
SWARM_INTERVAL = 10.0
SWARM_TLS_VERIFY = False
SWARM_SHARED_SECRET = "CHANGE_ME_SWARM_SECRET"

GLOBAL_STATE = None

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

def get_state_snapshot():
    global GLOBAL_STATE
    if GLOBAL_STATE is None:
        return None
    with GLOBAL_STATE.lock:
        return json.loads(json.dumps(GLOBAL_STATE.state))

# -----------------------------
# 3. Prediction helpers
# -----------------------------

def ewma(series, alpha=0.3):
    if not series:
        return 0.0
    v = float(series[0])
    for x in series[1:]:
        v = alpha * float(x) + (1.0 - alpha) * v
    return v

def linear_trend(series):
    n = len(series)
    if n < 2:
        return 0.0
    xs = np.arange(n, dtype=np.float32)
    ys = np.array(series, dtype=np.float32)
    x_mean = xs.mean()
    y_mean = ys.mean()
    num = ((xs - x_mean) * (ys - y_mean)).sum()
    den = ((xs - x_mean) ** 2).sum()
    if den == 0:
        return 0.0
    return float(num / den)

def regime_baseline(series, window=50):
    if not series:
        return 0.0, 0.0
    arr = np.array(series[-window:], dtype=np.float32)
    return float(arr.mean()), float(arr.std())

# -----------------------------
# 4. Call-home beacon client
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
        while SUBSYSTEM_FLAGS.get("beacon", True):
            try:
                self.flush_batch()
            except Exception as e:
                append_log({"type": "error", "source": "BeaconClient.flush_batch", "error": str(e)})
            time.sleep(HOME_INTERVAL)
        append_log({"type": "beacon_stopped", "ts": now_iso()})

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
# 5. Persistent state model
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
                "gpu_available": GPU_AVAILABLE,
                "ebpf_available": EBPF_AVAILABLE,
                "etw_available": ETW_AVAILABLE,
                "scapy_available": SCAPY_AVAILABLE,
                "ml_available": SKLEARN_AVAILABLE,
                "consciousness_state": CONSC_BASELINE,
            },
            "counters": {
                "inbound_bytes": 0,
                "outbound_bytes": 0,
                "files_created": 0,
                "files_modified": 0,
                "files_deleted": 0,
            },
            "processes": {},
            "files": {},
            "resources": {
                "memory": {},
                "disk": {},
                "net": {},
            },
            "ip_ledger": {},
            "connections": [],
            "lan_hosts": {},
            "anomaly": {
                "events": [],
                "pressure": 0.0,
                "state": STATE_NORMAL,
                "history": [],
            },
            "timeline": [],
            "swarm": {
                "node_id": SWARM_NODE_ID,
                "last_sync": None,
                "peers": SWARM_PEERS,
            },
            "ml": {
                "model_trained": False,
                "last_train": None,
            },
            "missing_baseline": {
                "processes": None,
                "nics": None,
                "lan_hosts": None,
                "files": None,
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
                "ips": {},
                "ports": {},
                "anomalies": [],
                "hours_seen": [],
                "embedding": None,
                "outbound_history": [],
                "outbound_ewma": 0.0,
                "outbound_trend": 0.0,
                "outbound_baseline_mean": 0.0,
                "outbound_baseline_std": 0.0,
                "outbound_delta": 0.0,
            })
            return proc

    def update_process_flow(self, pid, name, inbound_delta=0, outbound_delta=0):
        proc = self._ensure_process(pid, name)
        with self.lock:
            proc["inbound_bytes"] += inbound_delta
            proc["outbound_bytes"] += outbound_delta
            proc["last_seen"] = now_iso()
            if outbound_delta > 0:
                hist = proc.get("outbound_history", [])
                hist.append(float(outbound_delta))
                if len(hist) > 300:
                    hist = hist[-300:]
                proc["outbound_history"] = hist
                ew = ewma(hist, alpha=0.3)
                tr = linear_trend(hist[-50:]) if len(hist) > 2 else 0.0
                bm, bs = regime_baseline(hist, window=100)
                proc["outbound_ewma"] = ew
                proc["outbound_trend"] = tr
                proc["outbound_baseline_mean"] = bm
                proc["outbound_baseline_std"] = bs
                proc["outbound_delta"] = float(outbound_delta - ew)

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
                "processes": {},
                "ports": {},
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
            self.add_timeline_event({
                "kind": "ip_risk",
                "ip": ip,
                "risk": score,
                "reasons": reasons,
            })

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

        if score > 0:
            self.add_timeline_event({
                "kind": "lan_risk",
                "ip": ip,
                "risk": score,
                "reasons": reasons,
            })

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
            scores = np.array([e.get("score", 0.0) for e in a["events"]], dtype=np.float32)
            if GPU_AVAILABLE and scores.size > 0:
                try:
                    gpu_scores = cp.asarray(scores)
                    pressure = float(cp.sum(gpu_scores).get())
                except Exception:
                    pressure = float(scores.sum())
            else:
                pressure = float(scores.sum())
            a["pressure"] = pressure
            a["state"] = self._derive_state_from_pressure(pressure)

            hist = a.get("history", [])
            hist.append(pressure)
            if len(hist) > 300:
                hist = hist[-300:]
            a["history"] = hist

            self.add_timeline_event({
                "kind": "anomaly",
                "tags": ev.get("tags", []),
                "score": score,
                "pid": ev.get("pid"),
                "proc_name": ev.get("proc_name"),
            })

    def _derive_state_from_pressure(self, pressure):
        if pressure >= STATE_THRESHOLDS[STATE_LOCKDOWN]:
            return STATE_LOCKDOWN
        if pressure >= STATE_THRESHOLDS[STATE_HUNT]:
            return STATE_HUNT
        if pressure >= STATE_THRESHOLDS[STATE_ALERT]:
            return STATE_ALERT
        return STATE_NORMAL

    def add_timeline_event(self, ev):
        ev = dict(ev)
        ev.setdefault("ts", now_iso())
        with self.lock:
            self.state["timeline"].append(ev)
            if len(self.state["timeline"]) > 2000:
                self.state["timeline"] = self.state["timeline"][-2000:]

    def update_swarm_sync(self):
        with self.lock:
            self.state["swarm"]["last_sync"] = now_iso()

    def mark_ml_trained(self):
        with self.lock:
            self.state["ml"]["model_trained"] = True
            self.state["ml"]["last_train"] = now_iso()

    def set_consciousness_state(self, new_state, reason=None):
        with self.lock:
            old = self.state["meta"].get("consciousness_state", CONSC_BASELINE)
            if old != new_state:
                self.state["meta"]["consciousness_state"] = new_state
                self.add_timeline_event({
                    "kind": "consciousness_shift",
                    "from": old,
                    "to": new_state,
                    "reason": reason or "",
                })

    def update_missing_baseline(self, key, value):
        with self.lock:
            self.state["missing_baseline"][key] = value

# -----------------------------
# 6. Filesystem watcher
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
            while SUBSYSTEM_FLAGS.get("fs_monitor", True):
                time.sleep(1)
        except Exception as e:
            append_log({"type": "error", "source": "FileSystemMonitor.run", "error": str(e), "ts": now_iso()})
        finally:
            self.observer.stop()
            self.observer.join()
            append_log({"type": "fs_monitor_stopped", "ts": now_iso()})

# -----------------------------
# 7. Anomaly helpers
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
# 8. Network & process monitor
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
        while SUBSYSTEM_FLAGS.get("net_monitor", True):
            try:
                self.sample()
            except Exception as e:
                append_log({"type": "error", "source": "NetProcessMonitor.sample", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)
        append_log({"type": "net_monitor_stopped", "ts": now_iso()})

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
            outbound_delta = proc.get("outbound_delta", 0.0)
            outbound_ewma_val = proc.get("outbound_ewma", 0.0)
            outbound_baseline_mean = proc.get("outbound_baseline_mean", 0.0)
            outbound_baseline_std = proc.get("outbound_baseline_std", 0.0)

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

        if outbound_baseline_std > 0:
            z = (outbound_delta) / outbound_baseline_std
            if z > 3.0:
                score += 2.5
                tags.append("outbound_spike_vs_regime")

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
                "outbound_delta": outbound_delta,
                "outbound_ewma": outbound_ewma_val,
                "outbound_baseline_mean": outbound_baseline_mean,
                "outbound_baseline_std": outbound_baseline_std,
            }
            append_log(ev)
            self.beacon.enqueue(ev)
            self.state.add_anomaly_event(ev, score)

# -----------------------------
# 9. Resource monitor
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
        while SUBSYSTEM_FLAGS.get("res_monitor", True):
            try:
                self.sample()
            except Exception as e:
                append_log({"type": "error", "source": "ResourceMonitor.sample", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)
        append_log({"type": "resource_monitor_stopped", "ts": now_iso()})

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
# 10. Local network scanner + auto-blocker
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
        while SUBSYSTEM_FLAGS.get("lan_scanner", True):
            try:
                self.scan_once()
            except Exception as e:
                append_log({"type": "error", "source": "LocalNetworkScanner.scan_once", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)
        append_log({"type": "lan_scanner_stopped", "ts": now_iso()})

    def scan_once(self):
        os_name = platform.system().lower()
        hosts = {}

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
                self.state.add_timeline_event({
                    "kind": "lan_block",
                    "ip": ip,
                    "risk": risk,
                    "success": success,
                })

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
# 11. State flusher
# -----------------------------

class StateFlusher(threading.Thread):
    def __init__(self, state: PersistentState, interval=5.0):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval

    def run(self):
        ev = {"type": "state_flusher_started", "ts": now_iso()}
        append_log(ev)
        while SUBSYSTEM_FLAGS.get("state_flusher", True):
            try:
                self.state.save()
            except Exception as e:
                append_log({"type": "error", "source": "StateFlusher.run", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)
        append_log({"type": "state_flusher_stopped", "ts": now_iso()})

# -----------------------------
# 12. Console HUD
# -----------------------------

class ConsoleHUD(threading.Thread):
    def __init__(self, state: PersistentState, interval=2.0):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval

    def run(self):
        while SUBSYSTEM_FLAGS.get("console_hud", False):
            self.render()
            time.sleep(self.interval)
        append_log({"type": "console_hud_stopped", "ts": now_iso()})

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
            consciousness_state = self.state.state["meta"].get("consciousness_state", CONSC_BASELINE)

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
        print(f" STATE: {anomaly_state} | PRESSURE: {anomaly_pressure:.2f} | CONSC: {consciousness_state}")
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
# 13. eBPF monitor (stub)
# -----------------------------

class EBPFMonitor(threading.Thread):
    def __init__(self, state: PersistentState, beacon: BeaconClient):
        super().__init__(daemon=True)
        self.state = state
        self.beacon = beacon
        self.os_name = platform.system().lower()

    def run(self):
        if self.os_name != "linux" or not EBPF_AVAILABLE:
            append_log({"type": "ebpf_unavailable", "ts": now_iso()})
            return
        append_log({"type": "ebpf_monitor_started", "ts": now_iso()})
        while SUBSYSTEM_FLAGS.get("ebpf_monitor", True):
            time.sleep(5.0)
        append_log({"type": "ebpf_monitor_stopped", "ts": now_iso()})

# -----------------------------
# 14. ETW monitor (stub)
# -----------------------------

class ETWMonitor(threading.Thread):
    def __init__(self, state: PersistentState, beacon: BeaconClient):
        super().__init__(daemon=True)
        self.state = state
        self.beacon = beacon
        self.os_name = platform.system().lower()

    def run(self):
        if self.os_name != "windows" or not ETW_AVAILABLE:
            append_log({"type": "etw_unavailable", "ts": now_iso()})
            return
        append_log({"type": "etw_monitor_started", "ts": now_iso()})
        while SUBSYSTEM_FLAGS.get("etw_monitor", True):
            time.sleep(5.0)
        append_log({"type": "etw_monitor_stopped", "ts": now_iso()})

# -----------------------------
# 15. Kernel-level packet capture (scapy stub)
# -----------------------------

class PacketCaptureMonitor(threading.Thread):
    def __init__(self, state: PersistentState, beacon: BeaconClient):
        super().__init__(daemon=True)
        self.state = state
        self.beacon = beacon

    def run(self):
        if not SCAPY_AVAILABLE:
            append_log({"type": "packet_capture_unavailable", "ts": now_iso()})
            return
        append_log({"type": "packet_capture_started", "ts": now_iso()})
        try:
            sniff(prn=self._handle_packet, store=False, stop_filter=self._stop_filter)
        except Exception as e:
            append_log({"type": "error", "source": "PacketCaptureMonitor.sniff", "error": str(e), "ts": now_iso()})
        append_log({"type": "packet_capture_stopped", "ts": now_iso()})

    def _stop_filter(self, pkt):
        return not SUBSYSTEM_FLAGS.get("packet_capture", True)

    def _handle_packet(self, pkt):
        try:
            summary = pkt.summary()
        except Exception:
            summary = "packet"
        ev = {
            "type": "packet_observed",
            "summary": summary,
            "ts": now_iso(),
        }
        append_log(ev)
        self.beacon.enqueue(ev)

# -----------------------------
# 16. ML-based anomaly model (stub)
# -----------------------------

class MLAnomalyModel(threading.Thread):
    def __init__(self, state: PersistentState, beacon: BeaconClient, interval=60.0):
        super().__init__(daemon=True)
        self.state = state
        self.beacon = beacon
        self.interval = interval
        self.model = None

    def run(self):
        if not SKLEARN_AVAILABLE:
            append_log({"type": "ml_unavailable", "ts": now_iso()})
            return
        append_log({"type": "ml_anomaly_started", "ts": now_iso()})
        while SUBSYSTEM_FLAGS.get("ml_anomaly", True):
            try:
                self.train_and_score()
            except Exception as e:
                append_log({"type": "error", "source": "MLAnomalyModel.train_and_score", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)
        append_log({"type": "ml_anomaly_stopped", "ts": now_iso()})

    def _extract_features(self, processes):
        feats = []
        pids = []
        for pid, info in processes.items():
            inb = info.get("inbound_bytes", 0)
            outb = info.get("outbound_bytes", 0)
            ips = len(info.get("ips", {}))
            ports = len(info.get("ports", {}))
            ofc = info.get("open_files", 0)
            feats.append([inb, outb, ips, ports, ofc])
            pids.append(pid)
        if not feats:
            return None, None
        X = np.array(feats, dtype=np.float32)
        return X, pids

    def train_and_score(self):
        snap = get_state_snapshot()
        if not snap:
            return
        procs = snap.get("processes", {})
        X, pids = self._extract_features(procs)
        if X is None:
            return

        if self.model is None or X.shape[0] < 5:
            self.model = IsolationForest(contamination=0.05, random_state=42)
            self.model.fit(X)
            GLOBAL_STATE.mark_ml_trained()
            append_log({"type": "ml_model_trained", "n_samples": int(X.shape[0]), "ts": now_iso()})
            return

        preds = self.model.predict(X)
        scores = self.model.decision_function(X)
        for pid, pred, score in zip(pids, preds, scores):
            if pred == -1:
                ev = {
                    "type": "ml_proc_anomaly",
                    "pid": pid,
                    "score": float(-score),
                }
                append_log(ev)
                self.beacon.enqueue(ev)
                GLOBAL_STATE.add_anomaly_event(ev, float(-score))

# -----------------------------
# 17. Consciousness Engine
# -----------------------------

class ConsciousnessEngine(threading.Thread):
    def __init__(self, state: PersistentState, interval=5.0):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval

    def run(self):
        append_log({"type": "consciousness_engine_started", "ts": now_iso()})
        while SUBSYSTEM_FLAGS.get("consciousness", True):
            try:
                self.evaluate()
            except Exception as e:
                append_log({"type": "error", "source": "ConsciousnessEngine.evaluate", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)
        append_log({"type": "consciousness_engine_stopped", "ts": now_iso()})

    def evaluate(self):
        snap = get_state_snapshot()
        if not snap:
            return
        anomaly = snap.get("anomaly", {})
        pressure = anomaly.get("pressure", 0.0)
        state_label = anomaly.get("state", STATE_NORMAL)

        history = anomaly.get("history", [])
        ewma_val = ewma(history, alpha=0.3) if history else pressure
        trend_val = linear_trend(history[-50:]) if history else 0.0
        base_mean, base_std = regime_baseline(history, window=100) if history else (pressure, 0.0)

        projected = ewma_val + trend_val * 10.0

        timeline = snap.get("timeline", [])
        recent = [e for e in timeline if e.get("kind") in ("lan_block", "ip_risk", "ml_proc_anomaly", "anomaly")]
        recent_count = len(recent)

        new_state = CONSC_BASELINE
        reason = None

        eff_pressure = max(pressure, projected)

        if eff_pressure < 3.0 and recent_count < 5:
            new_state = CONSC_BASELINE
            reason = "low_pressure"
        elif 3.0 <= eff_pressure < 8.0:
            new_state = CONSC_FOCUSED
            reason = "moderate_pressure"
        elif 8.0 <= eff_pressure < 15.0:
            new_state = CONSC_HYPERVIGILANT
            reason = "high_pressure"
        elif eff_pressure < 3.0 and recent_count > 20:
            new_state = CONSC_DREAM
            reason = "post_spike_reconstruction"
        elif eff_pressure >= 15.0 and state_label in (STATE_HUNT, STATE_LOCKDOWN):
            new_state = CONSC_EVENT_HORIZON
            reason = "critical_pressure"
        else:
            new_state = CONSC_FOCUSED
            reason = "default_focus"

        self.state.add_timeline_event({
            "kind": "pressure_prediction",
            "pressure": pressure,
            "ewma": ewma_val,
            "trend": trend_val,
            "projected": projected,
            "baseline_mean": base_mean,
            "baseline_std": base_std,
        })

        self.state.set_consciousness_state(new_state, reason=reason)

# -----------------------------
# 18. Missing Detail Detector
# -----------------------------

class MissingDetailDetector(threading.Thread):
    def __init__(self, state: PersistentState, interval=30.0):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval

    def run(self):
        append_log({"type": "missing_detail_detector_started", "ts": now_iso()})
        while SUBSYSTEM_FLAGS.get("missing_detail", True):
            try:
                self.check_missing()
            except Exception as e:
                append_log({"type": "error", "source": "MissingDetailDetector.check_missing", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)
        append_log({"type": "missing_detail_detector_stopped", "ts": now_iso()})

    def check_missing(self):
        snap = get_state_snapshot()
        if not snap:
            return

        procs = snap.get("processes", {})
        resources = snap.get("resources", {})
        lan_hosts = snap.get("lan_hosts", {})
        files = snap.get("files", {})

        current_proc_names = set(info.get("name", "?") for info in procs.values())
        current_nics = set(resources.get("net", {}).keys())
        current_lan_ips = set(lan_hosts.keys())
        current_files = set(files.keys())

        with self.state.lock:
            baseline = self.state.state["missing_baseline"]

        if baseline["processes"] is None:
            self.state.update_missing_baseline("processes", list(current_proc_names))
            self.state.update_missing_baseline("nics", list(current_nics))
            self.state.update_missing_baseline("lan_hosts", list(current_lan_ips))
            self.state.update_missing_baseline("files", list(current_files))
            append_log({"type": "missing_baseline_initialized", "ts": now_iso()})
            return

        missing_procs = set(baseline["processes"]) - current_proc_names
        missing_nics = set(baseline["nics"]) - current_nics
        missing_lan = set(baseline["lan_hosts"]) - current_lan_ips

        missing_files = set()
        for f in baseline["files"]:
            if f not in current_files:
                missing_files.add(f)

        for name in missing_procs:
            ev = {
                "type": "missing_process",
                "name": name,
            }
            append_log(ev)
            self.state.add_anomaly_event(ev, score=1.0)

        for nic in missing_nics:
            ev = {
                "type": "missing_nic",
                "nic": nic,
            }
            append_log(ev)
            self.state.add_anomaly_event(ev, score=1.0)

        for ip in missing_lan:
            ev = {
                "type": "missing_lan_host",
                "ip": ip,
            }
            append_log(ev)
            self.state.add_anomaly_event(ev, score=1.5)

        for f in missing_files:
            ev = {
                "type": "missing_file",
                "path": f,
            }
            append_log(ev)
            self.state.add_anomaly_event(ev, score=0.5)

# -----------------------------
# 19. Remote command/control API
# -----------------------------

class RemoteAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/state":
            snap = get_state_snapshot() or {}
            self._send_json(200, snap)
        elif parsed.path == "/subsystems":
            self._send_json(200, SUBSYSTEM_FLAGS)
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8") if length > 0 else ""
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if parsed.path == "/toggle":
            name = data.get("name")
            enabled = bool(data.get("enabled", True))
            if name in SUBSYSTEM_FLAGS:
                SUBSYSTEM_FLAGS[name] = enabled
                append_log({"type": "remote_toggle", "subsystem": name, "enabled": enabled, "ts": now_iso()})
                self._send_json(200, {"ok": True, "subsystem": name, "enabled": enabled})
            else:
                self._send_json(400, {"error": "unknown_subsystem"})
        elif parsed.path == "/swarm":
            secret = data.get("secret")
            if secret != SWARM_SHARED_SECRET:
                self._send_json(403, {"error": "forbidden"})
                return
            append_log({"type": "swarm_update_received", "from": data.get("node_id"), "payload": data, "ts": now_iso()})
            GLOBAL_STATE.add_timeline_event({
                "kind": "swarm_update",
                "node_id": data.get("node_id"),
                "anomaly_state": data.get("anomaly_state"),
                "anomaly_pressure": data.get("anomaly_pressure"),
            })
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "not_found"})

class RemoteAPIServer(threading.Thread):
    def __init__(self, port=REMOTE_API_PORT):
        super().__init__(daemon=True)
        self.port = port
        self.httpd = None

    def run(self):
        try:
            self.httpd = HTTPServer(("0.0.0.0", self.port), RemoteAPIHandler)
            append_log({"type": "remote_api_started", "port": self.port, "ts": now_iso()})
            self.httpd.serve_forever()
        except Exception as e:
            append_log({"type": "error", "source": "RemoteAPIServer.run", "error": str(e), "ts": now_iso()})
        finally:
            if self.httpd:
                self.httpd.server_close()
            append_log({"type": "remote_api_stopped", "ts": now_iso()})

# -----------------------------
# 20. Swarm sync (encrypted)
# -----------------------------

class SwarmSync(threading.Thread):
    def __init__(self, state: PersistentState, interval=SWARM_INTERVAL):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval
        self.session = requests.Session()
        if not SWARM_TLS_VERIFY:
            self.session.verify = False

    def run(self):
        append_log({"type": "swarm_sync_started", "ts": now_iso()})
        while SUBSYSTEM_FLAGS.get("swarm_sync", True) and SWARM_ENABLED:
            try:
                self.sync_once()
            except Exception as e:
                append_log({"type": "error", "source": "SwarmSync.sync_once", "error": str(e), "ts": now_iso()})
            time.sleep(self.interval)
        append_log({"type": "swarm_sync_stopped", "ts": now_iso()})

    def sync_once(self):
        snap = get_state_snapshot()
        if not snap:
            return
        payload = {
            "node_id": SWARM_NODE_ID,
            "ts": now_iso(),
            "anomaly_state": snap.get("anomaly", {}).get("state"),
            "anomaly_pressure": snap.get("anomaly", {}).get("pressure"),
            "secret": SWARM_SHARED_SECRET,
        }
        for peer in SWARM_PEERS:
            try:
                url = peer.rstrip("/") + "/swarm"
                resp = self.session.post(url, data=json.dumps(payload), timeout=3.0, headers={"Content-Type": "application/json"})
                append_log({"type": "swarm_sync_sent", "peer": peer, "status": resp.status_code, "ts": now_iso()})
            except Exception as e:
                append_log({"type": "swarm_sync_failed", "peer": peer, "error": str(e), "ts": now_iso()})
        self.state.update_swarm_sync()

# -----------------------------
# 21. Snapshot helpers for GUI
# -----------------------------

def extract_dashboard(state):
    counters = state.get("counters", {})
    resources = state.get("resources", {})
    anomaly = state.get("anomaly", {})
    meta = state.get("meta", {})

    inbound = counters.get("inbound_bytes", 0)
    outbound = counters.get("outbound_bytes", 0)
    files_created = counters.get("files_created", 0)
    files_modified = counters.get("files_modified", 0)
    files_deleted = counters.get("files_deleted", 0)

    mem = resources.get("memory", {})
    mem_used = mem.get("used", 0)
    mem_total = mem.get("total", 0)
    mem_pct = mem.get("percent", 0.0)

    anomaly_state = anomaly.get("state", "NORMAL")
    anomaly_pressure = anomaly.get("pressure", 0.0)
    consciousness_state = meta.get("consciousness_state", CONSC_BASELINE)

    history = anomaly.get("history", [])
    ewma_val = ewma(history, alpha=0.3)
    trend_val = linear_trend(history[-50:]) if history else 0.0
    base_mean, base_std = regime_baseline(history, window=100)

    return {
        "inbound": inbound,
        "outbound": outbound,
        "files_created": files_created,
        "files_modified": files_modified,
        "files_deleted": files_deleted,
        "mem_used": mem_used,
        "mem_total": mem_total,
        "mem_pct": mem_pct,
        "anomaly_state": anomaly_state,
        "anomaly_pressure": anomaly_pressure,
        "consciousness_state": consciousness_state,
        "pressure_ewma": ewma_val,
        "pressure_trend": trend_val,
        "pressure_baseline_mean": base_mean,
        "pressure_baseline_std": base_std,
        "pressure_history": history,
    }

def extract_top_ips(state, limit=10):
    ip_ledger = state.get("ip_ledger", {})
    items = []
    for ip, entry in ip_ledger.items():
        risk = entry.get("risk_score", 0)
        hostname = entry.get("hostname") or "-"
        bytes_total = entry.get("bytes_total", 0)
        reasons = ";".join(entry.get("risk_reasons", [])) or "-"
        items.append((risk, ip, hostname, bytes_total, reasons))
    items.sort(reverse=True, key=lambda x: x[0])
    return items[:limit]

def extract_top_lan_hosts(state, limit=10):
    lan_hosts = state.get("lan_hosts", {})
    items = []
    for ip, host in lan_hosts.items():
        risk = host.get("risk_score", 0)
        mac = host.get("mac") or "-"
        hostname = host.get("hostname") or "-"
        blocked = host.get("blocked", False)
        reasons = ";".join(host.get("reasons", [])) or "-"
        items.append((risk, ip, mac, hostname, blocked, reasons))
    items.sort(reverse=True, key=lambda x: x[0])
    return items[:limit]

def extract_top_processes(state, limit=10):
    procs = state.get("processes", {})
    items = []
    for pid, info in procs.items():
        name = info.get("name", "?")
        outb = info.get("outbound_bytes", 0)
        inb = info.get("inbound_bytes", 0)
        ofc = info.get("open_files", 0)
        ew = info.get("outbound_ewma", 0.0)
        tr = info.get("outbound_trend", 0.0)
        delta = info.get("outbound_delta", 0.0)
        items.append((outb, pid, name, inb, ofc, ew, tr, delta))
    items.sort(reverse=True, key=lambda x: x[0])
    return items[:limit]

def extract_recent_anomalies(state, minutes=10, limit=50):
    anomaly = state.get("anomaly", {})
    events = anomaly.get("events", [])
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    out = []
    for e in events:
        ts_str = e.get("ts")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            continue
        if ts >= cutoff:
            out.append(e)
    out.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return out[:limit]

def extract_timeline(state, limit=200):
    timeline = state.get("timeline", [])
    timeline = sorted(timeline, key=lambda x: x.get("ts", ""), reverse=True)
    return timeline[:limit]

# -----------------------------
# 22. GUI data worker (background thread)
# -----------------------------

class GUIDataWorker(QtCore.QThread):
    data_ready = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def run(self):
        while not self._stop:
            snap = get_state_snapshot()
            if snap:
                try:
                    dashboard = extract_dashboard(snap)
                    ips = extract_top_ips(snap, limit=10)
                    lans = extract_top_lan_hosts(snap, limit=10)
                    procs = extract_top_processes(snap, limit=10)
                    anomalies = extract_recent_anomalies(snap, minutes=10, limit=50)
                    timeline = extract_timeline(snap, limit=200)

                    payload = {
                        "dashboard": dashboard,
                        "ips": ips,
                        "lans": lans,
                        "procs": procs,
                        "anomalies": anomalies,
                        "timeline": timeline,
                    }
                    self.data_ready.emit(payload)
                except Exception as e:
                    append_log({"type": "error", "source": "GUIDataWorker.run", "error": str(e), "ts": now_iso()})
            self.msleep(REFRESH_INTERVAL_MS)

    def stop(self):
        self._stop = True

# -----------------------------
# 23. PySide6 Operator Dashboard
# -----------------------------

class PressureGraph(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.projected = None

    def set_data(self, history, projected=None):
        self.history = list(history or [])
        self.projected = projected
        self.update()

    def paintEvent(self, event):
        if not self.history:
            return
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(5, 5, -5, -5)

        vals = self.history[-100:]
        n = len(vals)
        if n < 2:
            return
        vmin = min(vals)
        vmax = max(vals + ([self.projected] if self.projected is not None else []))
        if vmax == vmin:
            vmax = vmin + 1.0

        def map_point(i, v):
            x = rect.left() + (rect.width() * i / (n - 1))
            y = rect.bottom() - (rect.height() * (v - vmin) / (vmax - vmin))
            return QtCore.QPointF(x, y)

        path = QtGui.QPainterPath()
        path.moveTo(map_point(0, vals[0]))
        for i, v in enumerate(vals[1:], start=1):
            path.lineTo(map_point(i, v))

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 255), 2))
        painter.drawPath(path)

        if self.projected is not None:
            proj_x = rect.right()
            proj_y = rect.bottom() - (rect.height() * (self.projected - vmin) / (vmax - vmin))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 128, 0), 2, QtCore.Qt.DashLine))
            painter.drawLine(QtCore.QPointF(rect.right() - 10, proj_y), QtCore.QPointF(proj_x, proj_y))

class OperatorDashboard(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unified Telemetry Daemon — Operator Dashboard")
        self.resize(1600, 900)
        self.worker = None
        self._build_ui()
        self._start_worker()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.status_label = QtWidgets.QLabel("State: - | Pressure: - | Consciousness: -")
        font = self.status_label.font()
        font.setPointSize(11)
        font.setBold(True)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

        sys_group = QtWidgets.QGroupBox("System Overview")
        sys_layout = QtWidgets.QGridLayout(sys_group)

        self.inbound_label = QtWidgets.QLabel("Inbound: 0B")
        self.outbound_label = QtWidgets.QLabel("Outbound: 0B")
        self.files_label = QtWidgets.QLabel("Files: +0 created, +0 modified, +0 deleted")
        self.mem_label = QtWidgets.QLabel("Memory: 0 / 0 (0.0%)")
        self.mem_bar = QtWidgets.QProgressBar()
        self.mem_bar.setRange(0, 100)

        sys_layout.addWidget(self.inbound_label, 0, 0)
        sys_layout.addWidget(self.outbound_label, 0, 1)
        sys_layout.addWidget(self.files_label, 1, 0, 1, 2)
        sys_layout.addWidget(self.mem_label, 2, 0)
        sys_layout.addWidget(self.mem_bar, 2, 1)

        self.pressure_graph = PressureGraph()
        self.pressure_graph.setMinimumHeight(80)
        sys_layout.addWidget(self.pressure_graph, 3, 0, 1, 2)

        toggles_group = QtWidgets.QGroupBox("Subsystem Toggles")
        toggles_layout = QtWidgets.QGridLayout(toggles_group)

        self.check_beacon = QtWidgets.QCheckBox("Beacon")
        self.check_fs = QtWidgets.QCheckBox("FS Monitor")
        self.check_net = QtWidgets.QCheckBox("Net Monitor")
        self.check_res = QtWidgets.QCheckBox("Resource Monitor")
        self.check_lan = QtWidgets.QCheckBox("LAN Scanner")
        self.check_flush = QtWidgets.QCheckBox("State Flusher")
        self.check_console = QtWidgets.QCheckBox("Console HUD")
        self.check_ebpf = QtWidgets.QCheckBox("eBPF (Linux)")
        self.check_etw = QtWidgets.QCheckBox("ETW (Windows)")
        self.check_api = QtWidgets.QCheckBox("Remote API")
        self.check_swarm = QtWidgets.QCheckBox("Swarm Sync")
        self.check_pkt = QtWidgets.QCheckBox("Packet Capture")
        self.check_ml = QtWidgets.QCheckBox("ML Anomaly")
        self.check_consc = QtWidgets.QCheckBox("Consciousness Engine")
        self.check_missing = QtWidgets.QCheckBox("Missing Detail Detector")

        self.check_beacon.setChecked(SUBSYSTEM_FLAGS["beacon"])
        self.check_fs.setChecked(SUBSYSTEM_FLAGS["fs_monitor"])
        self.check_net.setChecked(SUBSYSTEM_FLAGS["net_monitor"])
        self.check_res.setChecked(SUBSYSTEM_FLAGS["res_monitor"])
        self.check_lan.setChecked(SUBSYSTEM_FLAGS["lan_scanner"])
        self.check_flush.setChecked(SUBSYSTEM_FLAGS["state_flusher"])
        self.check_console.setChecked(SUBSYSTEM_FLAGS["console_hud"])
        self.check_ebpf.setChecked(SUBSYSTEM_FLAGS["ebpf_monitor"])
        self.check_etw.setChecked(SUBSYSTEM_FLAGS["etw_monitor"])
        self.check_api.setChecked(SUBSYSTEM_FLAGS["remote_api"])
        self.check_swarm.setChecked(SUBSYSTEM_FLAGS["swarm_sync"])
        self.check_pkt.setChecked(SUBSYSTEM_FLAGS["packet_capture"])
        self.check_ml.setChecked(SUBSYSTEM_FLAGS["ml_anomaly"])
        self.check_consc.setChecked(SUBSYSTEM_FLAGS["consciousness"])
        self.check_missing.setChecked(SUBSYSTEM_FLAGS["missing_detail"])

        self.check_beacon.stateChanged.connect(lambda v: self.toggle_subsystem("beacon", v))
        self.check_fs.stateChanged.connect(lambda v: self.toggle_subsystem("fs_monitor", v))
        self.check_net.stateChanged.connect(lambda v: self.toggle_subsystem("net_monitor", v))
        self.check_res.stateChanged.connect(lambda v: self.toggle_subsystem("res_monitor", v))
        self.check_lan.stateChanged.connect(lambda v: self.toggle_subsystem("lan_scanner", v))
        self.check_flush.stateChanged.connect(lambda v: self.toggle_subsystem("state_flusher", v))
        self.check_console.stateChanged.connect(lambda v: self.toggle_subsystem("console_hud", v))
        self.check_ebpf.stateChanged.connect(lambda v: self.toggle_subsystem("ebpf_monitor", v))
        self.check_etw.stateChanged.connect(lambda v: self.toggle_subsystem("etw_monitor", v))
        self.check_api.stateChanged.connect(lambda v: self.toggle_subsystem("remote_api", v))
        self.check_swarm.stateChanged.connect(lambda v: self.toggle_subsystem("swarm_sync", v))
        self.check_pkt.stateChanged.connect(lambda v: self.toggle_subsystem("packet_capture", v))
        self.check_ml.stateChanged.connect(lambda v: self.toggle_subsystem("ml_anomaly", v))
        self.check_consc.stateChanged.connect(lambda v: self.toggle_subsystem("consciousness", v))
        self.check_missing.stateChanged.connect(lambda v: self.toggle_subsystem("missing_detail", v))

        toggles_layout.addWidget(self.check_beacon, 0, 0)
        toggles_layout.addWidget(self.check_fs, 0, 1)
        toggles_layout.addWidget(self.check_net, 1, 0)
        toggles_layout.addWidget(self.check_res, 1, 1)
        toggles_layout.addWidget(self.check_lan, 2, 0)
        toggles_layout.addWidget(self.check_flush, 2, 1)
        toggles_layout.addWidget(self.check_console, 3, 0)
        toggles_layout.addWidget(self.check_ebpf, 3, 1)
        toggles_layout.addWidget(self.check_etw, 4, 0)
        toggles_layout.addWidget(self.check_api, 4, 1)
        toggles_layout.addWidget(self.check_swarm, 5, 0)
        toggles_layout.addWidget(self.check_pkt, 5, 1)
        toggles_layout.addWidget(self.check_ml, 6, 0)
        toggles_layout.addWidget(self.check_consc, 6, 1)
        toggles_layout.addWidget(self.check_missing, 7, 0)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(sys_group)
        top_row.addWidget(toggles_group)
        layout.addLayout(top_row)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        tab_ips_lan = QtWidgets.QWidget()
        t1_layout = QtWidgets.QHBoxLayout(tab_ips_lan)

        self.table_ips = QtWidgets.QTableWidget(0, 5)
        self.table_ips.setHorizontalHeaderLabels(["IP", "Hostname", "Risk", "Bytes", "Reasons"])
        self.table_ips.horizontalHeader().setStretchLastSection(True)
        self.table_ips.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        self.table_lan = QtWidgets.QTableWidget(0, 6)
        self.table_lan.setHorizontalHeaderLabels(["IP", "MAC", "Hostname", "Risk", "Blocked", "Reasons"])
        self.table_lan.horizontalHeader().setStretchLastSection(True)
        self.table_lan.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        t1_layout.addWidget(self._wrap_group("Top Risky IPs", self.table_ips))
        t1_layout.addWidget(self._wrap_group("LAN Hosts", self.table_lan))
        tabs.addTab(tab_ips_lan, "Network")

        tab_proc_anom = QtWidgets.QWidget()
        t2_layout = QtWidgets.QHBoxLayout(tab_proc_anom)

        self.table_procs = QtWidgets.QTableWidget(0, 8)
        self.table_procs.setHorizontalHeaderLabels([
            "PID", "Name", "Inbound", "Outbound",
            "Open Files", "EWMA Out", "Trend", "Δ vs EWMA"
        ])
        self.table_procs.horizontalHeader().setStretchLastSection(True)
        self.table_procs.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        self.table_anom = QtWidgets.QTableWidget(0, 10)
        self.table_anom.setHorizontalHeaderLabels([
            "TS", "Type", "PID", "Process", "Tags", "Score",
            "OutΔ", "OutEWMA", "BaseMean", "BaseStd"
        ])
        self.table_anom.horizontalHeader().setStretchLastSection(True)
        self.table_anom.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        t2_layout.addWidget(self._wrap_group("Top Processes (Predictive Outbound)", self.table_procs))
        t2_layout.addWidget(self._wrap_group("Recent Anomalies (last 10 minutes)", self.table_anom))
        tabs.addTab(tab_proc_anom, "Processes & Anomalies")

        tab_timeline = QtWidgets.QWidget()
        t3_layout = QtWidgets.QVBoxLayout(tab_timeline)

        self.table_timeline = QtWidgets.QTableWidget(0, 6)
        self.table_timeline.setHorizontalHeaderLabels(["TS", "Kind", "Details1", "Details2", "Details3", "Details4"])
        self.table_timeline.horizontalHeader().setStretchLastSection(True)
        self.table_timeline.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        t3_layout.addWidget(self._wrap_group("Threat Timeline", self.table_timeline))
        tabs.addTab(tab_timeline, "Timeline")

    def _wrap_group(self, title, widget):
        group = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(group)
        v.addWidget(widget)
        return group

    def _start_worker(self):
        self.worker = GUIDataWorker()
        self.worker.data_ready.connect(self.apply_payload)
        self.worker.start()

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1000)
        super().closeEvent(event)

    def toggle_subsystem(self, name, value):
        enabled = (value == QtCore.Qt.Checked)
        SUBSYSTEM_FLAGS[name] = enabled
        append_log({"type": "subsystem_toggle", "subsystem": name, "enabled": enabled, "ts": now_iso()})

    @QtCore.Slot(dict)
    def apply_payload(self, payload):
        dash = payload.get("dashboard", {})
        if not dash:
            self.status_label.setText("State: (initializing...)")
            return

        self.status_label.setText(
            f"State: {dash['anomaly_state']} | Pressure: {dash['anomaly_pressure']:.2f} | Consciousness: {dash['consciousness_state']}"
        )
        self.inbound_label.setText(f"Inbound:  {human_bytes(dash['inbound'])}")
        self.outbound_label.setText(f"Outbound: {human_bytes(dash['outbound'])}")
        self.files_label.setText(
            f"Files: +{dash['files_created']} created, +{dash['files_modified']} modified, +{dash['files_deleted']} deleted"
        )
        mem_used = dash["mem_used"]
        mem_total = dash["mem_total"]
        mem_pct = dash["mem_pct"]
        self.mem_label.setText(
            f"Memory: {human_bytes(mem_used)} / {human_bytes(mem_total)} ({mem_pct:.1f}%)"
        )
        self.mem_bar.setValue(int(mem_pct))

        history = dash.get("pressure_history", [])
        ewma_val = dash.get("pressure_ewma", dash["anomaly_pressure"])
        trend_val = dash.get("pressure_trend", 0.0)
        projected = ewma_val + trend_val * 10.0
        self.pressure_graph.set_data(history, projected=projected)

        self._fill_table_ips(payload.get("ips", []))
        self._fill_table_lan(payload.get("lans", []))
        self._fill_table_procs(payload.get("procs", []))
        self._fill_table_anom(payload.get("anomalies", []))
        self._fill_table_timeline(payload.get("timeline", []))

    def _fill_table_ips(self, ips):
        self.table_ips.setRowCount(len(ips))
        for row, (risk, ip, hostname, bytes_total, reasons) in enumerate(ips):
            self.table_ips.setItem(row, 0, QtWidgets.QTableWidgetItem(ip))
            self.table_ips.setItem(row, 1, QtWidgets.QTableWidgetItem(hostname))
            self.table_ips.setItem(row, 2, QtWidgets.QTableWidgetItem(str(risk)))
            self.table_ips.setItem(row, 3, QtWidgets.QTableWidgetItem(human_bytes(bytes_total)))
            self.table_ips.setItem(row, 4, QtWidgets.QTableWidgetItem(reasons))

    def _fill_table_lan(self, lans):
        self.table_lan.setRowCount(len(lans))
        for row, (risk, ip, mac, hostname, blocked, reasons) in enumerate(lans):
            self.table_lan.setItem(row, 0, QtWidgets.QTableWidgetItem(ip))
            self.table_lan.setItem(row, 1, QtWidgets.QTableWidgetItem(mac))
            self.table_lan.setItem(row, 2, QtWidgets.QTableWidgetItem(hostname))
            self.table_lan.setItem(row, 3, QtWidgets.QTableWidgetItem(str(risk)))
            self.table_lan.setItem(row, 4, QtWidgets.QTableWidgetItem(str(blocked)))
            self.table_lan.setItem(row, 5, QtWidgets.QTableWidgetItem(reasons))

    def _fill_table_procs(self, procs):
        self.table_procs.setRowCount(len(procs))
        for row, (outb, pid, name, inb, ofc, ew, tr, delta) in enumerate(procs):
            self.table_procs.setItem(row, 0, QtWidgets.QTableWidgetItem(str(pid)))
            self.table_procs.setItem(row, 1, QtWidgets.QTableWidgetItem(name))
            self.table_procs.setItem(row, 2, QtWidgets.QTableWidgetItem(human_bytes(inb)))
            self.table_procs.setItem(row, 3, QtWidgets.QTableWidgetItem(human_bytes(outb)))
            self.table_procs.setItem(row, 4, QtWidgets.QTableWidgetItem(str(ofc)))
            self.table_procs.setItem(row, 5, QtWidgets.QTableWidgetItem(human_bytes(ew)))
            self.table_procs.setItem(row, 6, QtWidgets.QTableWidgetItem(f"{tr:.3f}"))
            self.table_procs.setItem(row, 7, QtWidgets.QTableWidgetItem(human_bytes(delta)))

    def _fill_table_anom(self, anomalies):
        self.table_anom.setRowCount(len(anomalies))
        for row, e in enumerate(anomalies):
            ts = e.get("ts", "")
            etype = e.get("type", "")
            pid = str(e.get("pid", ""))
            pname = e.get("proc_name", "")
            tags = ",".join(e.get("tags", [])) if e.get("tags") else ""
            score = str(e.get("score", ""))
            out_delta = e.get("outbound_delta", "")
            out_ewma = e.get("outbound_ewma", "")
            base_mean = e.get("outbound_baseline_mean", "")
            base_std = e.get("outbound_baseline_std", "")
            self.table_anom.setItem(row, 0, QtWidgets.QTableWidgetItem(ts))
            self.table_anom.setItem(row, 1, QtWidgets.QTableWidgetItem(etype))
            self.table_anom.setItem(row, 2, QtWidgets.QTableWidgetItem(pid))
            self.table_anom.setItem(row, 3, QtWidgets.QTableWidgetItem(pname))
            self.table_anom.setItem(row, 4, QtWidgets.QTableWidgetItem(tags))
            self.table_anom.setItem(row, 5, QtWidgets.QTableWidgetItem(score))
            self.table_anom.setItem(row, 6, QtWidgets.QTableWidgetItem(str(out_delta)))
            self.table_anom.setItem(row, 7, QtWidgets.QTableWidgetItem(str(out_ewma)))
            self.table_anom.setItem(row, 8, QtWidgets.QTableWidgetItem(str(base_mean)))
            self.table_anom.setItem(row, 9, QtWidgets.QTableWidgetItem(str(base_std)))

    def _fill_table_timeline(self, timeline):
        self.table_timeline.setRowCount(len(timeline))
        for row, e in enumerate(timeline):
            ts = e.get("ts", "")
            kind = e.get("kind", "")
            d1 = e.get("ip", "") or e.get("pid", "") or e.get("node_id", "") or e.get("from", "")
            d2 = ",".join(e.get("tags", [])) if e.get("tags") else e.get("proc_name", "") or e.get("to", "") or ""
            d3 = ";".join(e.get("reasons", [])) if e.get("reasons") else e.get("reason", "") or ""
            d4 = str(e.get("risk", "")) or str(e.get("score", ""))
            self.table_timeline.setItem(row, 0, QtWidgets.QTableWidgetItem(ts))
            self.table_timeline.setItem(row, 1, QtWidgets.QTableWidgetItem(kind))
            self.table_timeline.setItem(row, 2, QtWidgets.QTableWidgetItem(str(d1)))
            self.table_timeline.setItem(row, 3, QtWidgets.QTableWidgetItem(str(d2)))
            self.table_timeline.setItem(row, 4, QtWidgets.QTableWidgetItem(str(d3)))
            self.table_timeline.setItem(row, 5, QtWidgets.QTableWidgetItem(str(d4)))

# -----------------------------
# 24. Watchdog (auto-restart)
# -----------------------------

class Watchdog(threading.Thread):
    def __init__(self, factories):
        super().__init__(daemon=True)
        self.factories = factories
        self.threads = {name: None for name in factories}

    def run(self):
        append_log({"type": "watchdog_started", "ts": now_iso()})
        while True:
            for name, factory in self.factories.items():
                enabled = SUBSYSTEM_FLAGS.get(name, True)
                t = self.threads.get(name)
                if not enabled:
                    continue
                if t is None or not t.is_alive():
                    try:
                        new_t = factory()
                        new_t.start()
                        self.threads[name] = new_t
                        append_log({"type": "watchdog_restart", "subsystem": name, "ts": now_iso()})
                    except Exception as e:
                        append_log({"type": "error", "source": "Watchdog.restart", "subsystem": name, "error": str(e), "ts": now_iso()})
            time.sleep(5.0)

# -----------------------------
# 25. Main bootstrap
# -----------------------------

def main():
    global GLOBAL_STATE

    state = PersistentState()
    GLOBAL_STATE = state

    beacon = BeaconClient(enabled=HOME_ENABLED)

    factories = {
        "beacon": lambda: BeaconClient(enabled=HOME_ENABLED),
        "fs_monitor": lambda: FileSystemMonitor(state, beacon),
        "net_monitor": lambda: NetProcessMonitor(state, beacon),
        "res_monitor": lambda: ResourceMonitor(state, beacon),
        "lan_scanner": lambda: LocalNetworkScanner(state, beacon),
        "state_flusher": lambda: StateFlusher(state),
        "console_hud": lambda: ConsoleHUD(state),
        "ebpf_monitor": lambda: EBPFMonitor(state, beacon),
        "etw_monitor": lambda: ETWMonitor(state, beacon),
        "remote_api": lambda: RemoteAPIServer(port=REMOTE_API_PORT),
        "swarm_sync": lambda: SwarmSync(state),
        "packet_capture": lambda: PacketCaptureMonitor(state, beacon),
        "ml_anomaly": lambda: MLAnomalyModel(state, beacon),
        "consciousness": lambda: ConsciousnessEngine(state),
        "missing_detail": lambda: MissingDetailDetector(state),
    }

    watchdog = Watchdog(factories)
    watchdog.start()

    append_log({"type": "daemon_started", "ts": now_iso()})

    app = QtWidgets.QApplication(sys.argv)
    win = OperatorDashboard()
    win.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        append_log({"type": "daemon_stopped", "reason": "KeyboardInterrupt", "ts": now_iso()})
        print("\nShutting down...")

if __name__ == "__main__":
    main()
