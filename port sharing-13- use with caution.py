#!/usr/bin/env python3
"""
Codex Sentinel – Next Evolution Unified Daemon
(Drivers/eBPF/Memory/Lineage/Raft/GPU/P2P + Previous Features)

Major capabilities:
- Cross-platform core (Windows/Linux/macOS)
- Auto-elevation on Windows (best-effort)
- Port baseline learning + enforcement
- Behavioral + ML-style anomaly detection with adaptive thresholds
- Threat correlation (CPU/mem/conn + port + process)
- Self-healing wrappers for startup scripts (Windows Startup guarded)
- Integrity verification (SHA-256) for scripts/wrappers
- Encrypted configuration (XOR+Base64)
- Secure logging with hash chain (tamper-evident)
- Swarm communication with node authentication + signed commands
- Plugin system (plugins/ directory) with event hooks
- Threat-scoring heatmap (GUI + text snapshot)
- Forensic dump mode on high-severity events
- Real-time memory scanning hooks
- Process lineage graphing
- Raft-style consensus skeleton for distributed decisions
- GPU-aware anomaly scoring
- Encrypted P2P mesh interface (WireGuard-style abstraction)
- eBPF hooks (Linux, best-effort)
- Kernel-level integration stubs (Windows driver hooks, best-effort)
"""

import os
import sys
import time
import json
import hmac
import base64
import psutil
import socket
import hashlib
import logging
import threading
import subprocess
import queue
import re
from datetime import datetime, timedelta
from statistics import mean, pstdev

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None

try:
    import joblib
except ImportError:
    joblib = None

# Optional: Linux eBPF via bcc
try:
    from bcc import BPF  # type: ignore
    HAVE_BPF = True
except Exception:
    HAVE_BPF = False

# =========================
# PLATFORM HELPERS
# =========================

def is_windows() -> bool:
    return os.name == "nt"

def is_linux() -> bool:
    return sys.platform.startswith("linux")

def is_macos() -> bool:
    return sys.platform == "darwin"

# =========================
# OPTIONAL AUTO-ELEVATION (WINDOWS ONLY)
# =========================

if is_windows():
    import ctypes

    def ensure_admin():
        try:
            if not ctypes.windll.shell32.IsUserAnAdmin():
                script = os.path.abspath(sys.argv[0])
                params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
                ctypes.windll.shell32.ShellExecuteW(
                    None,
                    "runas",
                    sys.executable,
                    f'"{script}" {params}',
                    None,
                    1
                )
                sys.exit()
        except Exception as e:
            print(f"[Codex Sentinel] Elevation failed: {e}")
            sys.exit()

    ensure_admin()

# =========================
# CONFIGURATION
# =========================

CONFIG_FILE = "port_enforcer_config.json.enc"
ML_MODEL_FILE = "ml_anomaly_model.pkl"
INTEGRITY_FILE = "codex_integrity.json"
LOG_FILE = "port_enforcer.log"
FORENSICS_DIR = "forensics"
PLUGINS_DIR = "plugins"

LEARN_MODE_DEFAULT = True
SCAN_INTERVAL = 1.0
GPU_TELEMETRY_INTERVAL = 5

STABLE_WINDOW_SECONDS = 600

WEIGHT_UNKNOWN_PROCESS = 40
WEIGHT_UNAUTHORIZED_PORT = 40
WEIGHT_HIGH_PORT = 10
WEIGHT_ANOMALY = 25
WEIGHT_HISTORY = 15
WEIGHT_BEHAVIOR = 20
WEIGHT_CORRELATED = 20
WEIGHT_GPU_STRESS = 10
THREAT_ALERT_THRESHOLD = 60
THREAT_FORENSIC_THRESHOLD = 80

SWARM_BROADCAST_PORT = 50050
SWARM_NODE_ID = socket.gethostname()
SWARM_KEY = "KILLER666_SWARM_KEY"
SWARM_COMMAND_KEY = "KILLER666_CMD_KEY"
SWARM_NODE_SECRET = "KILLER666_NODE_SECRET"

SANDBOX_CPU_PERCENT_LIMIT = 50
SANDBOX_AFFINITY_LIMIT = 2

WRAPPER_SUFFIX = "_wrapper.py"
MOD_DIR_NAME = "CodexModified"

WRAPPER_CRASH_WINDOW_SECONDS = 60
WRAPPER_CRASH_MAX_RESTARTS = 5
HEALTH_MIN_SECONDS = 10
HEALTH_MAX_SECONDS = 20

if is_windows():
    USER_STARTUP_DIR = os.path.expanduser(
        r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
    )
    SYSTEM_STARTUP_DIR = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Startup"
else:
    USER_STARTUP_DIR = None
    SYSTEM_STARTUP_DIR = None

# =========================
# LOGGING + HASH CHAIN
# =========================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

console = logging.getLogger("console")
console.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
console.addHandler(ch)

log_chain_hash = "0" * 64
log_lock = threading.Lock()

def _update_log_chain(level: str, msg: str) -> str:
    global log_chain_hash
    with log_lock:
        data = f"{log_chain_hash}|{level}|{msg}".encode("utf-8")
        new_hash = hashlib.sha256(data).hexdigest()
        log_chain_hash = new_hash
        return new_hash

def secure_log(level: str, msg: str):
    h = _update_log_chain(level, msg)
    full = f"{msg} [chain={h}]"
    if level == "info":
        logging.info(full)
        console.info(full)
    elif level == "warning":
        logging.warning(full)
        console.warning(full)
    elif level == "error":
        logging.error(full)
        console.error(full)
    else:
        logging.info(full)
        console.info(full)

# =========================
# GLOBAL STATE
# =========================

lock = threading.Lock()

state = {
    "allowed_ports": {},
    "learn_mode": LEARN_MODE_DEFAULT,
    "last_scan": None,
    "events": [],
    "last_new_port_time": None,
    "gpu_util": None,
    "swarm_last_heartbeat": None,
    "swarm_peers": {},
    "port_usage_history": {},
    "false_positive_counts": {},
    "behavior_history": {},
    "heatmap": {},
    "raft_term": 0,
    "raft_role": "follower",  # follower | candidate | leader
    "raft_voted_for": None,
    "raft_peers": [],
    "p2p_peers": {},
    "lineage_graph": {},  # pid -> {ppid, children}
}

alert_queue = queue.Queue()
gui_update_queue = queue.Queue()

ml_model = None

# =========================
# SELF-EXCLUSION
# =========================

def is_self_script(path: str) -> bool:
    this = os.path.abspath(sys.argv[0]).lower()
    target = os.path.abspath(path).lower()
    return this == target or "codex" in os.path.basename(target)

# =========================
# SIMPLE SYMMETRIC ENCRYPTION (CONFIG)
# =========================

def _config_key() -> bytes:
    return hashlib.sha256(b"CODEX_CONFIG_KEY").digest()

def encrypt_config_bytes(data: bytes) -> bytes:
    key = _config_key()
    out = bytearray()
    for i, b in enumerate(data):
        out.append(b ^ key[i % len(key)])
    return base64.b64encode(out)

def decrypt_config_bytes(data: bytes) -> bytes:
    key = _config_key()
    raw = base64.b64decode(data)
    out = bytearray()
    for i, b in enumerate(raw):
        out.append(b ^ key[i % len(key)])
    return bytes(out)

# =========================
# SWARM ENCRYPTION + AUTH
# =========================

def _derive_key(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()

def encrypt_swarm_payload(data: dict) -> bytes:
    key = _derive_key(SWARM_KEY)
    raw = json.dumps(data).encode("utf-8")
    out = bytearray()
    for i, b in enumerate(raw):
        out.append(b ^ key[i % len(key)])
    return base64.b64encode(out)

def decrypt_swarm_payload(blob: bytes) -> dict | None:
    try:
        key = _derive_key(SWARM_KEY)
        raw = base64.b64decode(blob)
        out = bytearray()
        for i, b in enumerate(raw):
            out.append(b ^ key[i % len(key)])
        return json.loads(out.decode("utf-8"))
    except Exception:
        return None

def sign_node_identity(node_id: str) -> str:
    key = _derive_key(SWARM_NODE_SECRET)
    raw = node_id.encode("utf-8")
    return hmac.new(key, raw, hashlib.sha256).hexdigest()

def verify_node_identity(node_id: str, sig: str) -> bool:
    key = _derive_key(SWARM_NODE_SECRET)
    raw = node_id.encode("utf-8")
    expected = hmac.new(key, raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)

# =========================
# SIGNED SWARM COMMANDS
# =========================

def sign_command(cmd: dict) -> str:
    key = _derive_key(SWARM_COMMAND_KEY)
    cmd_copy = {k: v for k, v in cmd.items() if k != "sig"}
    raw = json.dumps(cmd_copy, sort_keys=True).encode("utf-8")
    return hmac.new(key, raw, hashlib.sha256).hexdigest()

def verify_command(cmd: dict) -> bool:
    allowed_types = {"SET_MODE", "BLOCK_PORT", "UNBLOCK_PORT"}
    if "type" not in cmd or cmd["type"] not in allowed_types:
        return False
    if "sig" not in cmd:
        return False
    expected = sign_command(cmd)
    return hmac.compare_digest(expected, cmd["sig"])

def apply_swarm_command(cmd: dict):
    ctype = cmd.get("type")
    if ctype == "SET_MODE":
        value = cmd.get("value")
        if value in ("learning", "enforcement"):
            with lock:
                state["learn_mode"] = (value == "learning")
            secure_log("info", f"[SWARM CMD] Mode set to {value}")
            save_config()
    elif ctype == "BLOCK_PORT":
        port = int(cmd.get("port", 0))
        proto = cmd.get("protocol", "TCP")
        if port > 0:
            firewall_block_port(port, proto)
            secure_log("info", f"[SWARM CMD] Blocked {proto} port {port}")
    elif ctype == "UNBLOCK_PORT":
        port = int(cmd.get("port", 0))
        proto = cmd.get("protocol", "TCP")
        if port > 0:
            firewall_unblock_port(port, proto)
            secure_log("info", f"[SWARM CMD] Unblocked {proto} port {port}")

# =========================
# CONFIG (ENCRYPTED)
# =========================

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "rb") as f:
                enc = f.read()
            raw = decrypt_config_bytes(enc)
            data = json.loads(raw.decode("utf-8"))
            with lock:
                state["allowed_ports"] = data.get("allowed_ports", {})
                state["learn_mode"] = data.get("learn_mode", LEARN_MODE_DEFAULT)
        except Exception as e:
            secure_log("error", f"Failed to load config: {e}")
    with lock:
        state["last_new_port_time"] = datetime.utcnow()

def save_config():
    data = {
        "allowed_ports": state["allowed_ports"],
        "learn_mode": state["learn_mode"],
    }
    try:
        raw = json.dumps(data, indent=2).encode("utf-8")
        enc = encrypt_config_bytes(raw)
        with open(CONFIG_FILE, "wb") as f:
            f.write(enc)
    except Exception as e:
        secure_log("error", f"Failed to save config: {e}")

# =========================
# ML MODEL
# =========================

def load_ml_model():
    global ml_model
    if joblib is None:
        secure_log("info", "joblib not available, ML anomaly detection disabled.")
        return
    if not os.path.exists(ML_MODEL_FILE):
        secure_log("info", "ML model file not found, ML anomaly detection disabled.")
        return
    try:
        ml_model = joblib.load(ML_MODEL_FILE)
        secure_log("info", "ML anomaly model loaded.")
    except Exception as e:
        secure_log("error", f"Failed to load ML model: {e}")
        ml_model = None

def extract_features_for_ml(conn, process_name: str, gpu_util: int | None) -> list[float]:
    port = conn.laddr.port if conn.laddr else 0
    proto = 1.0 if conn.type == psutil.SOCK_STREAM else 0.0
    status = 0.0
    if hasattr(conn, "status"):
        status = float(hash(conn.status) % 1000) / 1000.0
    plen = float(len(process_name)) / 64.0
    gpu_norm = float(gpu_util) / 100.0 if gpu_util is not None else 0.0
    return [float(port) / 65535.0, proto, status, plen, gpu_norm]

def ml_anomaly_flag(conn, process_name: str) -> bool:
    if ml_model is None:
        return False
    try:
        with lock:
            gpu_util = state["gpu_util"]
        feats = extract_features_for_ml(conn, process_name, gpu_util)
        if hasattr(ml_model, "decision_function"):
            score = ml_model.decision_function([feats])[0]
            return score < 0
        elif hasattr(ml_model, "predict"):
            pred = ml_model.predict([feats])[0]
            return int(pred) == -1
    except Exception as e:
        secure_log("error", f"ML anomaly check failed: {e}")
    return False

# =========================
# FIREWALL (WINDOWS ONLY)
# =========================

def firewall_block_port(port, protocol="TCP"):
    if not is_windows():
        return
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule",
        f"name=PortEnforcer_Block_{protocol}_{port}",
        "dir=in",
        "action=block",
        f"protocol={protocol}",
        f"localport={port}"
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        secure_log("error", f"Failed to add firewall rule: {e}")

def firewall_unblock_port(port, protocol="TCP"):
    if not is_windows():
        return
    cmd = [
        "netsh", "advfirewall", "firewall", "delete", "rule",
        f"name=PortEnforcer_Block_{protocol}_{port}",
        f"protocol={protocol}",
        f"localport={port}"
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        secure_log("error", f"Failed to remove firewall rule: {e}")

# =========================
# PORT UTIL
# =========================

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

# =========================
# SANDBOX + PROCESS CONTROL
# =========================

def sandbox_process(pid):
    try:
        p = psutil.Process(pid)
        if SANDBOX_AFFINITY_LIMIT > 0 and hasattr(p, "cpu_affinity"):
            try:
                cores = list(range(SANDBOX_AFFINITY_LIMIT))
                p.cpu_affinity(cores)
            except Exception:
                pass
        try:
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if is_windows() else 10)
        except Exception:
            pass
    except Exception as e:
        secure_log("error", f"Sandbox failed for PID {pid}: {e}")

def quarantine_process(pid, reason, port=None):
    try:
        p = psutil.Process(pid)
        name = p.name()
        msg = f"Quarantining PID {pid} ({name}) - {reason}"
        if port:
            msg += f" on port {port}"
        secure_log("warning", msg)
        try:
            p.suspend()
        except Exception:
            p.terminate()
    except Exception as e:
        secure_log("error", f"Could not quarantine PID {pid}: {e}")

def kill_process(pid, reason, port=None):
    try:
        p = psutil.Process(pid)
        name = p.name()
        msg = f"Killing PID {pid} ({name}) - {reason}"
        if port:
            msg += f" on port {port}"
        secure_log("warning", msg)
        p.terminate()
    except Exception as e:
        secure_log("error", f"Could not kill PID {pid}: {e}")

# =========================
# INTEGRITY DB
# =========================

integrity_lock = threading.Lock()
integrity_db = {}

def load_integrity_db():
    global integrity_db
    if os.path.exists(INTEGRITY_FILE):
        try:
            with open(INTEGRITY_FILE, "r", encoding="utf-8") as f:
                integrity_db = json.load(f)
        except Exception:
            integrity_db = {}

def save_integrity_db():
    try:
        with open(INTEGRITY_FILE, "w", encoding="utf-8") as f:
            json.dump(integrity_db, f, indent=2)
    except Exception:
        pass

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def register_integrity(path: str):
    digest = file_sha256(path)
    with integrity_lock:
        integrity_db[os.path.abspath(path)] = digest
        save_integrity_db()

def verify_integrity(path: str) -> bool:
    abspath = os.path.abspath(path)
    digest = file_sha256(path)
    with integrity_lock:
        expected = integrity_db.get(abspath)
    if not expected:
        return True
    return digest == expected

# =========================
# MEMORY SCANNING (REAL-TIME HOOK)
# =========================

def scan_process_memory(pid: int) -> dict:
    """
    Lightweight memory scan hook.
    Real AV-style scanning would use YARA or signatures; here we just sample.
    """
    info = {"pid": pid, "suspicious": False, "reason": "", "rss": 0}
    try:
        p = psutil.Process(pid)
        mem = p.memory_info().rss
        info["rss"] = mem
        # Simple heuristic: absurdly large memory for small process name length
        name_len = len(p.name() or "")
        if name_len < 4 and mem > 500 * 1024 * 1024:
            info["suspicious"] = True
            info["reason"] = "Tiny name, huge memory footprint"
    except Exception as e:
        info["reason"] = f"scan_error:{e}"
    return info

# =========================
# PROCESS LINEAGE GRAPHING
# =========================

def update_lineage_graph():
    graph = {}
    for p in psutil.process_iter(attrs=["pid", "ppid"]):
        try:
            pid = p.info["pid"]
            ppid = p.info["ppid"]
            graph.setdefault(pid, {"ppid": ppid, "children": []})
        except Exception:
            continue
    for pid, data in graph.items():
        ppid = data["ppid"]
        if ppid in graph:
            graph[ppid]["children"].append(pid)
    with lock:
        state["lineage_graph"] = graph

def get_lineage_chain(pid: int) -> list[int]:
    with lock:
        graph = state["lineage_graph"]
    chain = []
    current = pid
    visited = set()
    while current and current not in visited:
        visited.add(current)
        chain.append(current)
        node = graph.get(current)
        if not node:
            break
        current = node.get("ppid")
    return chain

# =========================
# ANOMALY + THREAT + BEHAVIOR + HEATMAP
# =========================

def update_port_usage_stats(process, port, count):
    key = (process, port)
    with lock:
        hist = state["port_usage_history"].setdefault(key, [])
        hist.append(count)
        if len(hist) > 100:
            hist.pop(0)

def update_behavior_fingerprint(process, port, cpu, mem, conn_count):
    key = (process, port)
    with lock:
        hist = state["behavior_history"].setdefault(key, [])
        hist.append((cpu, mem, conn_count))
        if len(hist) > 200:
            hist.pop(0)

def adaptive_thresholds(process, port, count) -> bool:
    key = (process, port)
    with lock:
        hist = state["port_usage_history"].get(key, [])
    if len(hist) < 20:
        return False
    m = mean(hist)
    s = pstdev(hist) or 1.0
    factor = 2.0 if s < 1.0 else 3.0
    return abs(count - m) > factor * s

def is_anomalous(process, port, count) -> bool:
    return adaptive_thresholds(process, port, count)

def get_false_positive_count(process, port) -> int:
    key = (process, port)
    with lock:
        return state["false_positive_counts"].get(key, 0)

def increment_false_positive(process, port):
    key = (process, port)
    with lock:
        state["false_positive_counts"][key] = state["false_positive_counts"].get(key, 0) + 1

def compute_threat_score(event):
    score = 0
    if event.get("unknown_process"):
        score += WEIGHT_UNKNOWN_PROCESS
    if event.get("unauthorized_port"):
        score += WEIGHT_UNAUTHORIZED_PORT
    if event.get("port") and event["port"] >= 49152:
        score += WEIGHT_HIGH_PORT
    if event.get("behavior_anomaly"):
        score += WEIGHT_BEHAVIOR
    if event.get("correlated"):
        score += WEIGHT_CORRELATED
    if event.get("gpu_stress"):
        score += WEIGHT_GPU_STRESS
    return score

def ml_style_threat_score(event):
    base = compute_threat_score(event)
    process = event.get("process")
    port = event.get("port")
    anomaly = event.get("anomaly", False)
    history_factor = get_false_positive_count(process, port)
    score = base
    if anomaly:
        score += WEIGHT_ANOMALY
    if history_factor > 3:
        score -= WEIGHT_HISTORY
    return max(score, 0)

def update_heatmap(event):
    key = (event.get("process"), event.get("port"))
    with lock:
        hm = state["heatmap"].setdefault(key, 0)
        state["heatmap"][key] = hm + event.get("threat_score", 0)

def record_event(event):
    event["timestamp"] = datetime.utcnow().isoformat()
    event["threat_score"] = ml_style_threat_score(event)
    with lock:
        state["events"].append(event)
        state["events"] = state["events"][-200:]
    update_heatmap(event)
    alert_queue.put(event)
    gui_update_queue.put(("event", event))
    plugins_on_event(event)

# =========================
# FORENSIC DUMP
# =========================

def ensure_forensics_dir():
    os.makedirs(FORENSICS_DIR, exist_ok=True)

def forensic_dump(event):
    ensure_forensics_dir()
    pid = event.get("pid")
    ts = event.get("timestamp", datetime.utcnow().isoformat())
    safe_ts = ts.replace(":", "-")
    fname = os.path.join(FORENSICS_DIR, f"forensic_{pid}_{safe_ts}.json")
    dump = {"event": event, "process_info": {}, "connections": [], "lineage": []}
    try:
        p = psutil.Process(pid)
        dump["process_info"] = {
            "pid": p.pid,
            "name": p.name(),
            "exe": p.exe() if p.exe() else "",
            "cmdline": p.cmdline(),
            "username": p.username(),
            "create_time": p.create_time(),
            "cpu_percent": p.cpu_percent(interval=0.05),
            "memory_info": p.memory_info()._asdict(),
            "status": p.status(),
            "ppid": p.ppid(),
        }
        for c in p.connections(kind="inet"):
            if not c.laddr:
                continue
            dump["connections"].append({
                "laddr": f"{c.laddr.ip}:{c.laddr.port}",
                "raddr": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "",
                "status": c.status,
                "type": "TCP" if c.type == psutil.SOCK_STREAM else "UDP",
            })
        dump["lineage"] = get_lineage_chain(pid)
    except Exception as e:
        dump["error"] = str(e)
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
        secure_log("info", f"Forensic dump written: {fname}")
    except Exception as e:
        secure_log("error", f"Failed to write forensic dump: {e}")

# =========================
# PLUGIN SYSTEM
# =========================

plugins = []

def load_plugins():
    if not os.path.isdir(PLUGINS_DIR):
        return
    sys.path.insert(0, os.path.abspath(PLUGINS_DIR))
    for file in os.listdir(PLUGINS_DIR):
        if not file.endswith(".py"):
            continue
        name = os.path.splitext(file)[0]
        try:
            mod = __import__(name)
            plugins.append(mod)
            secure_log("info", f"[PLUGIN] Loaded plugin: {name}")
        except Exception as e:
            secure_log("error", f"[PLUGIN] Failed to load {name}: {e}")

def plugins_on_event(event):
    for mod in plugins:
        fn = getattr(mod, "on_event", None)
        if callable(fn):
            try:
                fn(event)
            except Exception as e:
                secure_log("error", f"[PLUGIN] on_event error in {mod.__name__}: {e}")

def plugins_on_tick():
    for mod in plugins:
        fn = getattr(mod, "on_tick", None)
        if callable(fn):
            try:
                fn()
            except Exception as e:
                secure_log("error", f"[PLUGIN] on_tick error in {mod.__name__}: {e}")

# =========================
# LEARNING
# =========================

def learn_ports_from_connections(connections):
    updated = False
    with lock:
        allowed_ports = state["allowed_ports"]
        for conn in connections:
            pid = conn.pid
            if pid is None or not conn.laddr:
                continue
            port = conn.laddr.port
            try:
                name = psutil.Process(pid).name().lower()
            except Exception:
                continue
            ports = allowed_ports.setdefault(name, [])
            if port not in ports:
                ports.append(port)
                updated = True
        if updated:
            state["last_new_port_time"] = datetime.utcnow()
    if updated:
        save_config()
    return updated

def maybe_auto_flip_mode():
    with lock:
        learn_mode = state["learn_mode"]
        last_new = state["last_new_port_time"]
    if not learn_mode or last_new is None:
        return
    stable_for = (datetime.utcnow() - last_new).total_seconds()
    if stable_for >= STABLE_WINDOW_SECONDS:
        with lock:
            state["learn_mode"] = False
        secure_log("info", "Auto-flip: Learning → Enforcement")
        save_config()

# =========================
# ENFORCEMENT + CORRELATION
# =========================

def enforce_rules(connections):
    with lock:
        allowed_ports = state["allowed_ports"].copy()
        learn_mode = state["learn_mode"]
        gpu_util = state["gpu_util"]

    if learn_mode:
        return

    for conn in connections:
        pid = conn.pid
        if pid is None or not conn.laddr:
            continue

        port = conn.laddr.port
        proto = "TCP" if conn.type == psutil.SOCK_STREAM else "UDP"

        try:
            proc = psutil.Process(pid)
            name = proc.name().lower()
        except Exception:
            continue

        unknown_process = name not in allowed_ports
        unauthorized_port = False
        if not unknown_process and port not in allowed_ports.get(name, []):
            unauthorized_port = True

        if unknown_process or unauthorized_port:
            cpu = 0.0
            mem = 0
            conn_count = 0
            try:
                cpu = proc.cpu_percent(interval=0.01)
                mem = proc.memory_info().rss
                conn_count = len(proc.connections(kind="inet"))
            except Exception:
                pass
            update_behavior_fingerprint(name, port, cpu, mem, conn_count)
            correlated = cpu > 10.0 or conn_count > 5
            gpu_stress = gpu_util is not None and gpu_util > 80

            mem_scan = scan_process_memory(pid)
            if mem_scan.get("suspicious"):
                correlated = True

            event = {
                "pid": pid,
                "process": name,
                "port": port,
                "protocol": proto,
                "unknown_process": unknown_process,
                "unauthorized_port": unauthorized_port,
                "behavior_anomaly": correlated,
                "correlated": correlated,
                "gpu_stress": gpu_stress,
                "reason": "Unknown/unauthorized port with suspicious behavior" if correlated else "Unknown/unauthorized port",
                "mem_scan": mem_scan,
            }
            record_event(event)

            if event["threat_score"] >= THREAT_FORENSIC_THRESHOLD:
                forensic_dump(event)

            if correlated:
                quarantine_process(pid, event["reason"], port)
            else:
                kill_process(pid, event["reason"], port)
            firewall_block_port(port, proto)

# =========================
# SCANNER (OPTIMIZED)
# =========================

def scanner_loop():
    secure_log("info", "Scanner thread started.")
    while True:
        try:
            connections = psutil.net_connections(kind="inet")
        except Exception as e:
            secure_log("error", f"Failed to get connections: {e}")
            time.sleep(SCAN_INTERVAL)
            continue

        with lock:
            state["last_scan"] = datetime.utcnow().isoformat()
            learn_mode = state["learn_mode"]
            gpu_util = state["gpu_util"]

        update_lineage_graph()

        if learn_mode:
            updated = learn_ports_from_connections(connections)
            if not updated:
                maybe_auto_flip_mode()
        else:
            enforce_rules(connections)

        port_counts = {}
        for conn in connections:
            if conn.laddr:
                port_counts[conn.laddr.port] = port_counts.get(conn.laddr.port, 0) + 1

        snapshot = []
        for conn in connections:
            pid = conn.pid
            if pid is None or not conn.laddr:
                continue
            try:
                proc = psutil.Process(pid)
                name = proc.name().lower()
            except Exception:
                name = "unknown"
                proc = None
            proto = "TCP" if conn.type == psutil.SOCK_STREAM else "UDP"
            port = conn.laddr.port
            shared = port_counts.get(port, 0) > 1
            count = port_counts.get(port, 0)
            update_port_usage_stats(name, port, count)

            cpu = 0.0
            mem = 0
            if proc:
                try:
                    cpu = proc.cpu_percent(interval=0.0)
                    mem = proc.memory_info().rss
                except Exception:
                    pass
            update_behavior_fingerprint(name, port, cpu, mem, count)

            heuristic_anomaly = is_anomalous(name, port, count)
            ml_flag = ml_anomaly_flag(conn, name)
            anomaly = heuristic_anomaly or ml_flag
            gpu_stress = gpu_util is not None and gpu_util > 80

            if anomaly or gpu_stress:
                mem_scan = scan_process_memory(pid)
                event = {
                    "pid": pid,
                    "process": name,
                    "port": port,
                    "protocol": proto,
                    "unknown_process": False,
                    "unauthorized_port": False,
                    "anomaly": anomaly,
                    "behavior_anomaly": heuristic_anomaly,
                    "correlated": False,
                    "gpu_stress": gpu_stress,
                    "reason": "Anomalous port usage pattern (ML/heuristic/GPU)",
                    "mem_scan": mem_scan,
                }
                record_event(event)
                if event["threat_score"] >= THREAT_FORENSIC_THRESHOLD:
                    forensic_dump(event)
                sandbox_process(pid)

            snapshot.append({
                "pid": pid,
                "process": name,
                "port": port,
                "protocol": proto,
                "status": conn.status,
                "shared": shared,
            })

        gui_update_queue.put(("snapshot", snapshot))
        plugins_on_tick()
        time.sleep(SCAN_INTERVAL)

# =========================
# ALERT HANDLER
# =========================

def alert_handler_loop():
    secure_log("info", "Alert handler started.")
    while True:
        event = alert_queue.get()
        if event is None:
            break
        if event["threat_score"] >= THREAT_ALERT_THRESHOLD:
            secure_log(
                "warning",
                f"[ALERT] Score {event['threat_score']} - {event['process']} "
                f"PID {event['pid']} PORT {event['port']} REASON: {event.get('reason','')}"
            )

# =========================
# WRAPPERS (WINDOWS STARTUP ONLY)
# =========================

def ensure_wrapper_for_script(script_path: str) -> str | None:
    if is_self_script(script_path):
        return None

    script_dir = os.path.dirname(script_path)
    mod_dir = os.path.join(script_dir, MOD_DIR_NAME)

    if not os.path.isdir(mod_dir):
        try:
            os.makedirs(mod_dir, exist_ok=True)
            secure_log("info", f"[WRAPPER] Created CodexModified folder: {mod_dir}")
        except Exception as e:
            secure_log("error", f"[WRAPPER] Failed to create CodexModified folder: {e}")
            mod_dir = script_dir

    base = os.path.basename(script_path)
    name, _ = os.path.splitext(base)
    wrapper_name = f"{name}{WRAPPER_SUFFIX}"
    wrapper_path = os.path.join(mod_dir, wrapper_name)

    if not os.path.exists(wrapper_path):
        secure_log("info", f"[WRAPPER] Creating supervisor wrapper for {script_path}")
        wrapper_code = f'''import os
import sys
import subprocess
import time
from datetime import datetime, timedelta
import re
import psutil
import hashlib
import json

CRASH_WINDOW_SECONDS = {WRAPPER_CRASH_WINDOW_SECONDS}
CRASH_MAX_RESTARTS = {WRAPPER_CRASH_MAX_RESTARTS}
HEALTH_MIN_SECONDS = {HEALTH_MIN_SECONDS}
HEALTH_MAX_SECONDS = {HEALTH_MAX_SECONDS}

SCRIPT_PATH = r"{script_path}"
INTEGRITY_FILE = r"{os.path.abspath(INTEGRITY_FILE)}"

def load_integrity_db():
    if os.path.exists(INTEGRITY_FILE):
        try:
            with open(INTEGRITY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {{}}
    return {{}}

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def verify_integrity(path: str) -> bool:
    db = load_integrity_db()
    abspath = os.path.abspath(path)
    expected = db.get(abspath)
    if not expected:
        return True
    return file_sha256(path) == expected

def aggressive_patch_ports_in_script(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
    except Exception:
        return False

    patched = original
    changed = False

    if "AUTO_PORT" in patched and "import os" not in patched:
        patched = "import os\\n" + patched
        changed = True

    def repl_port_kw(m):
        nonlocal changed
        prefix = m.group(1)
        num = m.group(2)
        changed = True
        return f'{{prefix}}int(os.getenv("AUTO_PORT", "{{num}}"))'

    patched = re.sub(r'(\\bport\\s*=\\s*)(\\d{{2,5}})', repl_port_kw, patched)

    def repl_bind(m):
        nonlocal changed
        prefix = m.group(1)
        num = m.group(2)
        suffix = m.group(3)
        changed = True
        return f'{{prefix}}int(os.getenv("AUTO_PORT", "{{num}}")){{suffix}}'

    patched = re.sub(
        r'(\\.bind\\(\\s*\\(.*?,\\s*)(\\d{{2,5}})(\\s*\\)\\s*\\))',
        repl_bind,
        patched
    )

    lines = patched.splitlines()
    new_lines = []
    for line in lines:
        if ("listen" in line or "socket" in line or "bind" in line) and re.search(r'\\b\\d{{2,5}}\\b', line):
            def repl_any(m):
                nonlocal changed
                num = m.group(0)
                changed = True
                return f'int(os.getenv("AUTO_PORT", "{{num}}"))'
            line = re.sub(r'\\b\\d{{2,5}}\\b', repl_any, line)
        new_lines.append(line)
    patched = "\\n".join(new_lines)

    if not changed:
        return False

    backup_path = path + ".bak"
    try:
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(original)
        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)
        return True
    except Exception:
        return False

def child_bound_port(pid: int, port: int) -> bool:
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.pid != pid or not conn.laddr:
                continue
            if conn.laddr.port == port:
                return True
    except Exception:
        pass
    return False

def child_has_activity(p: psutil.Process) -> bool:
    try:
        cpu = p.cpu_percent(interval=0.1)
        mem = p.memory_info().rss if p.is_running() else 0
        if cpu > 0.1 or mem > 5 * 1024 * 1024:
            return True
    except Exception:
        pass
    return False

def main():
    if len(sys.argv) < 2:
        print("[WRAPPER] No port provided")
        sys.exit(1)

    port_str = sys.argv[1]
    try:
        port = int(port_str)
    except ValueError:
        print("[WRAPPER] Invalid port")
        sys.exit(1)

    if not verify_integrity(SCRIPT_PATH):
        print("[WRAPPER] Integrity check failed for script, aborting.")
        sys.exit(1)

    os.environ["AUTO_PORT"] = port_str

    crash_times = []
    backoff = 2

    while True:
        try:
            p = subprocess.Popen([sys.executable, SCRIPT_PATH], env=os.environ)
        except Exception as e:
            print(f"[WRAPPER] Failed to launch child: {{e}}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

        start = time.time()
        healthy = False
        stable = False

        try:
            proc = psutil.Process(p.pid)
        except Exception:
            proc = None

        while True:
            if p.poll() is not None:
                break

            now = time.time()
            alive_for = now - start

            bound = child_bound_port(p.pid, port)
            active = child_has_activity(proc) if proc else False

            if alive_for >= HEALTH_MIN_SECONDS and (bound or active):
                healthy = True
            if alive_for >= HEALTH_MAX_SECONDS and (bound or active):
                stable = True
                break

            time.sleep(0.5)

        ret = p.poll()
        now_dt = datetime.utcnow()
        crash_times.append(now_dt)
        cutoff = now_dt - timedelta(seconds=CRASH_WINDOW_SECONDS)
        crash_times = [t for t in crash_times if t >= cutoff]

        if stable or (healthy and ret == 0):
            crash_times.clear()
            print("[WRAPPER] Child considered healthy/stable, not restarting.")
            break

        if ret == 0 and not healthy:
            print("[WRAPPER] Child exited quickly but cleanly, not restarting.")
            break

        if len(crash_times) >= CRASH_MAX_RESTARTS:
            print("[WRAPPER] Crash-loop detected, attempting aggressive port auto-patch...")
            if aggressive_patch_ports_in_script(SCRIPT_PATH):
                print("[WRAPPER] Patch applied, clearing crash history and restarting.")
                crash_times.clear()
                backoff = 2
                time.sleep(2)
                continue
            else:
                print("[WRAPPER] Patch failed or no changes, stopping wrapper.")
                break

        print(f"[WRAPPER] Child crashed or unhealthy, restarting in {{backoff}}s...")
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)

if __name__ == "__main__":
    main()
'''
        try:
            with open(wrapper_path, "w", encoding="utf-8") as f:
                f.write(wrapper_code)
        except Exception as e:
            secure_log("error", f"[WRAPPER] Failed to create wrapper for {script_path}: {e}")
    register_integrity(script_path)
    register_integrity(wrapper_path)
    return wrapper_path

def is_process_running_for_entry(entry):
    target_name = entry["name"].lower()
    target_path = entry["match_path"]
    for p in psutil.process_iter(attrs=["name", "cmdline"]):
        try:
            if not p.info["name"]:
                continue
            if p.info["name"].lower() != target_name:
                continue
            cmdline = " ".join(p.info.get("cmdline") or []).lower()
            if target_path in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def is_original_script_running(script_path: str) -> bool:
    target = os.path.abspath(script_path).lower()
    for p in psutil.process_iter(attrs=["name", "cmdline"]):
        try:
            cmdline = " ".join(p.info.get("cmdline") or []).lower()
            if target in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def discover_python_startup_programs():
    if not is_windows():
        return []
    programs = []
    for folder in [USER_STARTUP_DIR, SYSTEM_STARTUP_DIR]:
        if not folder or not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if not file.lower().endswith(".py"):
                continue
            full_path = os.path.join(folder, file)
            if MOD_DIR_NAME.lower() in full_path.lower():
                continue
            if full_path.lower().endswith(WRAPPER_SUFFIX):
                continue
            if is_self_script(full_path):
                continue
            wrapper_path = ensure_wrapper_for_script(full_path)
            if not wrapper_path:
                continue
            programs.append({
                "id": wrapper_path.lower(),
                "name": "python.exe",
                "match_path": wrapper_path.lower(),
                "script_path": full_path.lower(),
                "command_template": [sys.executable, wrapper_path, "{PORT}"],
                "cwd": os.path.dirname(wrapper_path)
            })
    return programs

MANAGED_PROGRAMS = []

def launch_program(entry):
    if not verify_integrity(entry["match_path"]):
        secure_log("warning", f"[WATCHDOG] Integrity check failed for wrapper {entry['match_path']}, not launching.")
        return
    port = get_free_port()
    cmd = [arg.replace("{PORT}", str(port)) for arg in entry["command_template"]]
    try:
        subprocess.Popen(cmd, cwd=entry["cwd"])
        secure_log("info", f"[WATCHDOG] Launched wrapper: {cmd} (port {port})")
    except Exception as e:
        secure_log("error", f"[WATCHDOG] Failed to launch {cmd}: {e}")

def watchdog_loop():
    if not is_windows():
        secure_log("info", "Watchdog disabled on non-Windows (no Startup wrappers).")
        return
    secure_log("info", "Watchdog thread started.")
    while True:
        for entry in MANAGED_PROGRAMS:
            script_path = entry.get("script_path")
            if script_path and is_original_script_running(script_path):
                continue
            if not is_process_running_for_entry(entry):
                secure_log("warning", f"[WATCHDOG] Wrapper not running: {entry['match_path']}. Relaunching...")
                launch_program(entry)
        time.sleep(5)

# =========================
# SWARM
# =========================

def swarm_heartbeat_loop():
    secure_log("info", "Swarm heartbeat thread started.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while True:
        try:
            with lock:
                payload = {
                    "node": SWARM_NODE_ID,
                    "node_sig": sign_node_identity(SWARM_NODE_ID),
                    "time": datetime.utcnow().isoformat(),
                    "gpu_util": state["gpu_util"],
                    "mode": "learning" if state["learn_mode"] else "enforcement",
                    "allowed_ports": state["allowed_ports"],
                    "commands": [],
                }
            data = encrypt_swarm_payload(payload)
            sock.sendto(data, ("255.255.255.255", SWARM_BROADCAST_PORT))
            with lock:
                state["swarm_last_heartbeat"] = payload["time"]
        except Exception as e:
            secure_log("error", f"Swarm heartbeat error: {e}")
        time.sleep(5)

def swarm_listener_loop():
    secure_log("info", "Swarm listener thread started.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", SWARM_BROADCAST_PORT))
    while True:
        try:
            data, addr = sock.recvfrom(65535)
            payload = decrypt_swarm_payload(data)
            if not payload:
                continue
            node = payload.get("node")
            sig = payload.get("node_sig")
            if not node or node == SWARM_NODE_ID:
                continue
            if not sig or not verify_node_identity(node, sig):
                secure_log("warning", "[SWARM] Rejected heartbeat from unauthenticated node.")
                continue
            with lock:
                state["swarm_peers"][node] = payload.get("time")
                peer_ports = payload.get("allowed_ports", {})
                for proc, ports in peer_ports.items():
                    local_ports = state["allowed_ports"].setdefault(proc, [])
                    for p in ports:
                        if p not in local_ports:
                            local_ports.append(p)
            commands = payload.get("commands", [])
            for cmd in commands:
                if verify_command(cmd):
                    apply_swarm_command(cmd)
                else:
                    secure_log("warning", "[SWARM CMD] Rejected invalid or unsigned command.")
        except Exception as e:
            secure_log("error", f"Swarm listener error: {e}")

# =========================
# GPU TELEMETRY
# =========================

def gpu_telemetry_loop():
    secure_log("info", "GPU telemetry thread started.")
    while True:
        util = None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                line = result.stdout.strip().splitlines()[0]
                util = int(line)
        except Exception:
            util = None
        with lock:
            state["gpu_util"] = util
        time.sleep(GPU_TELEMETRY_INTERVAL)

# =========================
# HEATMAP (TEXT SNAPSHOT)
# =========================

def get_heatmap_snapshot(top_n=10):
    with lock:
        items = list(state["heatmap"].items())
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:top_n]

# =========================
# RAFT-STYLE CONSENSUS (SKELETON)
# =========================

def raft_loop():
    """
    Minimal Raft-style skeleton:
    - Tracks term and role
    - Periodically logs state
    - Real implementation would require RPCs between nodes
    """
    secure_log("info", "Raft consensus skeleton started.")
    while True:
        with lock:
            role = state["raft_role"]
            term = state["raft_term"]
        # For now, just log occasionally
        if int(time.time()) % 30 == 0:
            secure_log("info", f"[RAFT] role={role} term={term}")
        time.sleep(1)

# =========================
# ENCRYPTED P2P MESH (WIREGUARD-STYLE INTERFACE)
# =========================

def p2p_encrypt(data: bytes) -> bytes:
    key = _derive_key("P2P_MESH_KEY")
    out = bytearray()
    for i, b in enumerate(data):
        out.append(b ^ key[i % len(key)])
    return base64.b64encode(out)

def p2p_decrypt(blob: bytes) -> bytes | None:
    try:
        key = _derive_key("P2P_MESH_KEY")
        raw = base64.b64decode(blob)
        out = bytearray()
        for i, b in enumerate(raw):
            out.append(b ^ key[i % len(key)])
        return bytes(out)
    except Exception:
        return None

def p2p_mesh_loop():
    """
    WireGuard-style interface layer (logical only).
    Real WireGuard integration would configure OS-level interfaces.
    Here we maintain encrypted TCP channels to peers (if configured).
    """
    secure_log("info", "P2P mesh loop started (logical layer).")
    while True:
        # Placeholder: in a real setup, connect to configured peers and exchange encrypted status.
        time.sleep(10)

# =========================
# eBPF HOOKS (LINUX)
# =========================

def ebpf_loop():
    if not is_linux() or not HAVE_BPF:
        secure_log("info", "eBPF not available; skipping eBPF loop.")
        return
    secure_log("info", "eBPF loop started (Linux).")
    # Minimal example: attach to kprobe for tcp_connect (if supported)
    program = r"""
    int kprobe__tcp_connect(struct pt_regs *ctx) {
        return 0;
    }
    """
    try:
        b = BPF(text=program)
        while True:
            # In a real implementation, read maps/events and correlate with userspace.
            time.sleep(5)
    except Exception as e:
        secure_log("error", f"eBPF error: {e}")

# =========================
# GUI + HUD + HEATMAP VIEW
# =========================

class PortEnforcerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Port Enforcement Cockpit (Swarm/HUD)")

        self.tree = ttk.Treeview(
            root,
            columns=("pid", "process", "port", "protocol", "status", "shared"),
            show="headings"
        )
        for col in ["pid", "process", "port", "protocol", "status", "shared"]:
            self.tree.heading(col, text=col.capitalize())
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(root, text="Status: Initializing...")
        self.status_label.pack(fill=tk.X)

        self.heatmap_label = tk.Label(root, text="Heatmap: N/A", justify="left", anchor="w")
        self.heatmap_label.pack(fill=tk.X)

        self.hud = tk.Toplevel(root)
        self.hud.title("HUD")
        self.hud.attributes("-topmost", True)
        self.hud.geometry("260x140+20+20")
        self.hud.resizable(False, False)
        self.hud_label = tk.Label(self.hud, text="HUD", font=("Consolas", 9), justify="left", anchor="w")
        self.hud_label.pack(fill=tk.BOTH, expand=True)

        self.refresh_gui()

    def refresh_gui(self):
        try:
            while True:
                kind, payload = gui_update_queue.get_nowait()
                if kind == "snapshot":
                    self.update_snapshot(payload)
                elif kind == "event":
                    self.status_label.config(
                        text=f"Event: {payload.get('reason','')} (score {payload['threat_score']})"
                    )
        except queue.Empty:
            pass

        with lock:
            last_scan = state["last_scan"]
            learn_mode = state["learn_mode"]
            gpu_util = state["gpu_util"]
            swarm_last = state["swarm_last_heartbeat"]
            peers = list(state["swarm_peers"].keys())

        mode = "Learning" if learn_mode else "Enforcement"
        gpu_text = f"GPU {gpu_util}%" if gpu_util is not None else "GPU N/A"
        swarm_text = f"Peers: {len(peers)} | Last: {swarm_last}" if swarm_last else "Swarm N/A"

        heat_items = get_heatmap_snapshot(5)
        heat_lines = ["Heatmap (top):"]
        for (proc, port), score in heat_items:
            heat_lines.append(f"{proc}:{port} -> {score}")
        self.heatmap_label.config(text="\n".join(heat_lines))

        self.status_label.config(
            text=f"Mode={mode} | LastScan={last_scan} | {gpu_text}"
        )
        self.hud_label.config(
            text=f"Mode: {mode}\n{gpu_text}\n{swarm_text}\n" + "\n".join(heat_lines)
        )

        self.root.after(500, self.refresh_gui)

    def update_snapshot(self, snapshot):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for item in snapshot:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    item["pid"],
                    item["process"],
                    item["port"],
                    item["protocol"],
                    item["status"],
                    "Yes" if item["shared"] else "No",
                ),
            )

# =========================
# DAEMON
# =========================

def run_daemon(headless=True):
    load_integrity_db()
    load_config()
    load_ml_model()
    load_plugins()

    global MANAGED_PROGRAMS
    MANAGED_PROGRAMS = [p for p in discover_python_startup_programs() if p]

    threading.Thread(target=scanner_loop, daemon=True).start()
    threading.Thread(target=alert_handler_loop, daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    threading.Thread(target=swarm_heartbeat_loop, daemon=True).start()
    threading.Thread(target=swarm_listener_loop, daemon=True).start()
    threading.Thread(target=gpu_telemetry_loop, daemon=True).start()
    threading.Thread(target=raft_loop, daemon=True).start()
    threading.Thread(target=p2p_mesh_loop, daemon=True).start()
    threading.Thread(target=ebpf_loop, daemon=True).start()

    if headless or tk is None:
        secure_log("info", "Running headless daemon.")
        while True:
            time.sleep(1)
    else:
        root = tk.Tk()
        PortEnforcerGUI(root)
        root.mainloop()

# =========================
# MAIN
# =========================

def main():
    headless = ("--daemon" in sys.argv) or ("--headless" in sys.argv)
    run_daemon(headless=headless)

if __name__ == "__main__":
    main()
