import subprocess
import sys

# 🔄 Auto-loader for non-stdlib packages
def autoload(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        if pip_name is None:
            pip_name = package
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        __import__(package)

for pkg in ["uuid", "socket", "platform", "psutil", "threading", "time", "re", "datetime", "os", "json"]:
    autoload(pkg)

# Optional GPU / NumPy stack with safe fallback
GPU_ENABLED = False
try:
    try:
        autoload("cupy")
        import cupy as xp
        GPU_ENABLED = True
    except Exception:
        raise ImportError("CuPy unavailable or broken")
except ImportError:
    autoload("numpy")
    import numpy as xp
    GPU_ENABLED = False

import tkinter as tk
from tkinter import ttk
import uuid
import socket
import platform
import psutil
import threading
import time
import re
from datetime import datetime, timedelta
import os
import json
import ctypes

# === AUTO-ELEVATION CHECK ===
def ensure_admin():
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}" {params}', None, 1
            )
            sys.exit()
    except Exception as e:
        print(f"[Codex Sentinel] Elevation failed: {e}")
        sys.exit()

ensure_admin()

# =========================
# 🔐 Privacy / Chameleon Layer
# =========================

GLYPH_ALPHABET = "⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁⟁"

def glyph_encode(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    h = s.encode("utf-8").hex()
    mirrored = h[::-1]
    out = []
    for i, ch in enumerate(mirrored):
        idx = (ord(ch) + i) % len(GLYPH_ALPHABET)
        out.append(GLYPH_ALPHABET[idx])
    return "".join(out)

def mirror_reverse(s: str) -> str:
    return s[::-1]

def shred_string(s: str) -> str:
    s = "\x00" * len(s)
    return ""

def chameleon_wrap(label: str, value: str) -> str:
    encoded = glyph_encode(value)
    mirrored = mirror_reverse(encoded)
    shredded = shred_string(value)
    return f"{label}:{mirrored}"

# =========================
# 🧠 Persistent Brain
# =========================

BRAIN_FILE = "magicbox_brain.json"
brain_lock = threading.Lock()
brain_state = {
    "trust_config": {},
    "mutation_log": [],
    "swarm_id": None,
    "phantom_history": [],
    "node_id": None,
    "last_anomaly_score": 0.0,
    "remote_nodes": {},
    "meta_state": "Sentinel",
    "regime": "stable",
    "integrity_score": 1.0,
    "health_score": 1.0,
    "prediction_confidence": 0.5,
    "reasoning_heatmap": {},
    "pattern_memory": {},
    "stance_thresholds": {
        "low": 25.0,
        "medium": 50.0,
        "high": 75.0
    },
    "appetite": {
        "ingest": 1.0,
        "thread_expansion": 1.0,
        "deep_ram": 1.0
    },
    "horizons": {
        "micro": 5.0,
        "mid": 30.0,
        "macro": 120.0
    },
    "dampening": {
        "micro": 0.5,
        "macro": 0.7
    },
    "last_calibration": None
}

def load_brain():
    global brain_state, trust_config
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with brain_lock:
                brain_state.update(data)
            if "trust_config" in brain_state and isinstance(brain_state["trust_config"], dict):
                trust_config.update(brain_state["trust_config"])
        except Exception:
            pass

def save_brain():
    with brain_lock:
        brain_state["trust_config"] = trust_config
        brain_state["mutation_log"] = mutation_log[-500:]
    try:
        with open(BRAIN_FILE, "w", encoding="utf-8") as f:
            json.dump(brain_state, f, indent=2, default=str)
    except Exception:
        pass

def periodic_brain_save():
    save_brain()
    root.after(15000, periodic_brain_save)

# =========================
# 🔧 Config-Driven Trust Rules
# SwarmID / Telemetry / Phantom TTL=10
# =========================

trust_config = {
    "MAC": {"action": "destroy", "ttl": 86400},
    "IP": {"action": "cloak", "ttl": 86400},
    "Telemetry": {"action": "destroy", "ttl": 10},
    "Phantom": {"action": "destroy", "ttl": 10},
    "SwarmID": {"action": "preserve", "ttl": 10}
}

# =========================
# 🧠 Symbolic Memory Routing / Tier-6 Mutation Engine
# =========================

def symbolic_route(data):
    sigil = uuid.uuid4().hex[:8]
    return f"{sigil}:{data}"

mutation_log = []
destruction_queue = []
mutation_lock = threading.Lock()

event_history = []
event_history_lock = threading.Lock()
MAX_HISTORY = 1024

def record_event(kind, payload, severity="info", tags=None):
    if tags is None:
        tags = []
    timestamp = datetime.now().isoformat()
    entry = {
        "ts": timestamp,
        "kind": kind,
        "payload": payload,
        "severity": severity,
        "tags": tags
    }
    line = symbolic_route(f"[{severity.upper()}][{kind}] {payload}")
    with mutation_lock:
        mutation_log.append(line)
    with event_history_lock:
        event_history.append(entry)
        if len(event_history) > MAX_HISTORY:
            event_history.pop(0)
    update_pattern_memory(entry)
    return entry

def update_pattern_memory(entry):
    with brain_lock:
        pm = brain_state.setdefault("pattern_memory", {})
        key = f"{entry['kind']}|{entry['severity']}"
        stats = pm.get(key, {"count": 0, "preceded_overload": 0, "preceded_stable": 0, "preceded_beast": 0})
        stats["count"] += 1
        pm[key] = stats

def mark_outcome(outcome):
    with event_history_lock:
        recent = list(event_history[-50:])
    with brain_lock:
        pm = brain_state.setdefault("pattern_memory", {})
        for e in recent:
            key = f"{e['kind']}|{e['severity']}"
            stats = pm.get(key, {"count": 0, "preceded_overload": 0, "preceded_stable": 0, "preceded_beast": 0})
            if outcome == "overload":
                stats["preceded_overload"] += 1
            elif outcome == "stable":
                stats["preceded_stable"] += 1
            elif outcome == "beast":
                stats["preceded_beast"] += 1
            pm[key] = stats

# =========================
# 🧾 Mutation Trail Logger
# =========================

def update_log():
    log_text.delete(1.0, tk.END)
    with mutation_lock:
        for entry in mutation_log[-10:]:
            log_text.insert(tk.END, f"{entry}\n")

# =========================
# ⏳ Self-Destruct Logic
# =========================

def schedule_destruction(tag, ttl_seconds):
    if ttl_seconds:
        expiry = datetime.now() + timedelta(seconds=ttl_seconds)
        destruction_queue.append((tag, expiry))

def check_destruction():
    now = datetime.now()
    for tag, expiry in destruction_queue[:]:
        if now >= expiry:
            record_event("self_destruct", f"Self-destructed: {tag}", severity="info")
            destruction_queue.remove((tag, expiry))
    update_log()
    watchdog_touch("destruction")
    root.after(5000, check_destruction)

# =========================
# 🧩 Trust Engine Handler
# =========================

def handle_data(tag, value):
    rule = trust_config.get(tag, {})
    action = rule.get("action")
    ttl = rule.get("ttl")

    safe_value = chameleon_wrap(tag, str(value))

    if action == "destroy":
        record_event("trust", f"{safe_value}", severity="info", tags=["destroy"])
        schedule_destruction(tag, ttl)
    elif action == "cloak":
        record_event("trust", f"{tag}: [CLOAKED]", severity="info", tags=["cloak"])
        schedule_destruction(tag, ttl)
    elif action == "preserve":
        record_event("trust", f"{safe_value}", severity="info", tags=["preserve"])

# =========================
# 🦎 Real-Time Detection (Identifiers)
# =========================

def get_real_mac():
    try:
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if getattr(psutil, "AF_LINK", None) is not None:
                    if addr.family == psutil.AF_LINK:
                        return addr.address
                else:
                    if getattr(addr.family, "name", "") == "AF_LINK":
                        return addr.address
    except Exception:
        pass
    return "MAC not found"

def get_real_ip():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        try:
            public_ip = socket.gethostbyname_ex(hostname)[2][-1]
        except Exception:
            public_ip = local_ip
        return local_ip, public_ip
    except Exception as e:
        return "IP error", str(e)

def get_telemetry():
    os_info = platform.platform()
    browser_fingerprint = platform.system() + "-" + platform.machine()
    return os_info, browser_fingerprint

def get_swarm_id():
    with brain_lock:
        if brain_state.get("swarm_id"):
            return brain_state["swarm_id"]
    sid = str(uuid.getnode())
    with brain_lock:
        brain_state["swarm_id"] = sid
    return sid

def synthesize_phantom():
    entropy = uuid.uuid4().hex + str(time.time_ns())
    phantom = f"phantom://{entropy[:12]}"
    with brain_lock:
        brain_state.setdefault("phantom_history", []).append(phantom)
        brain_state["phantom_history"] = brain_state["phantom_history"][-50:]
    return phantom

# =========================
# 🌐 Network / Remote Control Detection
# =========================

PRIVATE_NETS = [
    ("10.",),
    ("172.", range(16, 32)),
    ("192.168.",)
]

def is_private_ip(ip: str) -> bool:
    if ip.startswith("10."):
        return True
    if ip.startswith("192.168."):
        return True
    if ip.startswith("172."):
        parts = ip.split(".")
        if len(parts) >= 2:
            try:
                second = int(parts[1])
                return 16 <= second <= 31
            except ValueError:
                return False
    return False

REMOTE_TOOL_PATTERNS = re.compile(
    r"(teamviewer|anydesk|vnc|remote|rdp|shadow|splashtop|ultraviewer|ammyy|logmein)",
    re.IGNORECASE
)

FIREWALL_CMD_PATTERNS = re.compile(
    r"(netsh\s+advfirewall|Set-NetFirewall|New-NetFirewall|ufw\s+enable|ufw\s+disable)",
    re.IGNORECASE
)

SETTINGS_CMD_PATTERNS = re.compile(
    r"(reg\s+add|reg\s+delete|powershell\s+Set-ItemProperty|gpedit\.msc|secpol\.msc)",
    re.IGNORECASE
)

SHELL_NAMES = re.compile(
    r"(cmd\.exe|powershell\.exe|pwsh\.exe|bash\.exe|wsl\.exe)",
    re.IGNORECASE
)

def monitor_network():
    try:
        conns = psutil.net_connections(kind="inet")
    except Exception as e:
        record_event("net", f"Failed to read connections: {e}", "warn")
        root.after(8000, monitor_network)
        return

    for c in conns:
        try:
            raddr = c.raddr
            if not raddr:
                continue
            ip = raddr.ip
            port = raddr.port
            if not is_private_ip(ip):
                record_event("net", f"Foreign connection: {ip}:{port}", "warn", tags=["foreign"])
        except Exception:
            continue

    root.after(8000, monitor_network)

# =========================
# 🛡️ Process / Policy Engine
# =========================

def terminate_proc(proc, reason):
    try:
        name = proc.name()
    except Exception:
        name = "unknown"
    try:
        pid = proc.pid
    except Exception:
        pid = -1
    try:
        proc.terminate()
        record_event("policy", f"Terminated PID {pid} ({name}) - {reason}", "crit", tags=["kill"])
    except Exception as e:
        record_event("policy", f"Failed to terminate PID {pid} ({name}) - {reason} - {e}", "warn")

def is_windows_system_process(proc):
    try:
        exe = proc.exe() or ""
    except Exception:
        return False
    exe_lower = exe.lower()
    return ("\\windows\\system32\\" in exe_lower) or ("\\windows defender\\" in exe_lower)

# =========================
# 🧠 Local Operator Activity
# =========================

STATE_FILE = "guardian_state.json"
state_lock = threading.Lock()
state = {
    "log": [],
    "max_log": 500,
    "last_input_ts": None,
    "policy": {
        "block_remote_control": True,
        "block_firewall_changes": True,
        "block_settings_changes": True
    }
}

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with state_lock:
                state.update(data)
        except Exception:
            pass

def save_state_guardian():
    with state_lock:
        data = dict(state)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass

def periodic_save_guardian():
    save_state_guardian()
    root.after(15000, periodic_save_guardian)

def mark_local_input(event=None):
    with state_lock:
        state["last_input_ts"] = datetime.now().isoformat(timespec="seconds")

def is_local_operator_active(window=10):
    with state_lock:
        ts = state.get("last_input_ts")
    if not ts:
        return False
    try:
        last = datetime.fromisoformat(ts)
    except Exception:
        return False
    delta = (datetime.now() - last).total_seconds()
    return delta <= window

# =========================
# 🧾 Guardian Log (GUI)
# =========================

guardian_log_lock = threading.Lock()
GUARDIAN_LOG_WIDGET = None

def guardian_log_event(kind, msg, severity="info", tags=None):
    if tags is None:
        tags = []
    ts = datetime.now().isoformat(timespec="seconds")
    entry = {
        "ts": ts,
        "kind": kind,
        "msg": msg,
        "severity": severity,
        "tags": tags
    }
    with state_lock:
        state["log"].append(entry)
        if len(state["log"]) > state["max_log"]:
            state["log"].pop(0)
    with guardian_log_lock:
        if GUARDIAN_LOG_WIDGET is not None:
            GUARDIAN_LOG_WIDGET.insert(tk.END, f"[{ts}][{severity.upper()}][{kind}] {msg}\n")
            GUARDIAN_LOG_WIDGET.see(tk.END)
    return entry

# =========================
# 🧬 Hybrid Policy Monitor
# =========================

def monitor_processes():
    local_active = is_local_operator_active()
    with state_lock:
        policy = dict(state["policy"])

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = " ".join(proc.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

        if policy.get("block_remote_control", True):
            if REMOTE_TOOL_PATTERNS.search(name) or REMOTE_TOOL_PATTERNS.search(cmdline):
                if not local_active:
                    terminate_proc(proc, "remote control tool detected without local operator")

        if SHELL_NAMES.search(name):
            if FIREWALL_CMD_PATTERNS.search(cmdline) and policy.get("block_firewall_changes", True):
                if not is_windows_system_process(proc) or not local_active:
                    terminate_proc(proc, "firewall change attempt blocked")
                    continue
            if SETTINGS_CMD_PATTERNS.search(cmdline) and policy.get("block_settings_changes", True):
                if not is_windows_system_process(proc) or not local_active:
                    terminate_proc(proc, "settings change attempt blocked")
                    continue

    root.after(7000, monitor_processes)

# =========================
# ⚔️ Threat Detection + Response (all listening ports)
# =========================

def threat_scan_and_respond():
    try:
        try:
            conns = psutil.net_connections(kind='inet')
        except Exception as e:
            record_event("threat_scan", f"net_connections failed: {e}", severity="warn")
            conns = []

        seen = set()
        for conn in conns:
            try:
                if conn.status == 'LISTEN':
                    ip = getattr(conn.laddr, "ip", "unknown")
                    port = conn.laddr.port
                    key = (ip, port)
                    if key in seen:
                        continue
                    seen.add(key)
                    record_event(
                        "port_listen",
                        f"LISTEN {ip}:{port}",
                        severity="info",
                        tags=["port"]
                    )
            except Exception:
                continue

        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info.get('name') or ""
                pid = proc.info.get('pid')
                if name and re.search(r"(keylogger|sniffer|injector|bot|miner)", name, re.IGNORECASE):
                    try:
                        proc.terminate()
                        record_event("threat_neutralized", f"{name} (PID {pid})", severity="crit", tags=["proc", "kill"])
                    except Exception as e:
                        record_event("threat_failure", f"Failed to terminate {name} (PID {pid}) - {e}", severity="warn")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        record_event("self_rewrite", f"threat_scan_and_respond() failed - {e}", severity="crit", tags=["self_rewrite"])

    update_log()
    watchdog_touch("threat_scan")
    root.after(10000, threat_scan_and_respond)

# =========================
# 🧬 Self-Rewriting Engine
# =========================

def self_check():
    try:
        assert callable(threat_scan_and_respond)
        assert callable(update_log)
        assert isinstance(trust_config, dict)
    except Exception as e:
        record_event("self_rewrite", f"Integrity check failed - {e}", severity="crit", tags=["self_rewrite"])
    watchdog_touch("self_check")
    root.after(15000, self_check)

# =========================
# 🧮 Metrics / Regime / Prediction
# =========================

def compute_event_metrics(window_sec=300):
    with event_history_lock:
        if not event_history:
            return {
                "rate": 0.0,
                "variance": 0.0,
                "turbulence": 0.0,
                "trend": 0.0
            }
        now = datetime.now()
        times = []
        sev_vals = []
        sev_map = {"info": 1.0, "warn": 2.0, "crit": 4.0}
        for e in event_history:
            try:
                ts = datetime.fromisoformat(e["ts"])
            except Exception:
                continue
            dt = (now - ts).total_seconds()
            if dt <= window_sec:
                times.append(dt)
                sev_vals.append(sev_map.get(e["severity"], 1.0))
        if len(times) < 2:
            return {
                "rate": float(len(times)) / max(1.0, window_sec),
                "variance": 0.0,
                "turbulence": 0.0,
                "trend": 0.0
            }
        t_arr = xp.asarray(times, dtype=xp.float32)
        s_arr = xp.asarray(sev_vals, dtype=xp.float32)
        rate = float(len(times)) / max(1.0, window_sec)
        variance = float(s_arr.var())
        t_sorted = xp.sort(t_arr)
        diffs = t_sorted[:-1] - t_sorted[1:]
        turbulence = float((diffs ** 2).mean())
        x = t_arr
        y = s_arr
        x_mean = x.mean()
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum() + 1e-6
        trend = float(num / den)
        return {
            "rate": rate,
            "variance": variance,
            "turbulence": turbulence,
            "trend": trend
        }

def detect_regime(metrics):
    var_ = metrics["variance"]
    turb = metrics["turbulence"]
    trend = metrics["trend"]
    if var_ < 0.2 and turb < 0.01:
        return "stable"
    if var_ > 1.5 or turb > 0.5:
        return "chaotic"
    if trend > 0.01:
        return "rising"
    if trend < -0.01:
        return "cooling"
    return "stable"

def multi_horizon_prediction(metrics, anomaly_score):
    with brain_lock:
        horizons = brain_state.get("horizons", {"micro": 5.0, "mid": 30.0, "macro": 120.0})
    rate = metrics["rate"]
    trend = metrics["trend"]
    variance = metrics["variance"]
    turbulence = metrics["turbulence"]

    base = anomaly_score
    preds = {}
    for name, horizon in horizons.items():
        projected = base + trend * horizon * 10.0 + turbulence * 50.0
        projected += variance * 10.0
        projected = max(0.0, min(100.0, projected))
        preds[name] = projected
    return preds

def fuse_predictions(preds, regime, anomaly_score, metrics):
    turb = metrics["turbulence"]
    variance = metrics["variance"]
    if regime == "stable":
        model_conf = 0.8
    elif regime in ("rising", "cooling"):
        model_conf = 0.6
    else:
        model_conf = 0.4
    model_conf *= max(0.2, 1.0 - min(1.0, turb))
    ewma_conf = 1.0 - model_conf

    if regime == "chaotic":
        micro_w, mid_w, macro_w = 0.6, 0.3, 0.1
    elif regime == "rising":
        micro_w, mid_w, macro_w = 0.4, 0.4, 0.2
    elif regime == "cooling":
        micro_w, mid_w, macro_w = 0.3, 0.4, 0.3
    else:
        micro_w, mid_w, macro_w = 0.2, 0.3, 0.5

    blended_model = (
        preds.get("micro", anomaly_score) * micro_w +
        preds.get("mid", anomaly_score) * mid_w +
        preds.get("macro", anomaly_score) * macro_w
    )
    ewma_baseline = anomaly_score
    fused = blended_model * model_conf + ewma_baseline * ewma_conf

    heatmap = {
        "regime": regime,
        "model_confidence": model_conf,
        "ewma_confidence": ewma_conf,
        "variance": variance,
        "turbulence": turb,
        "trend": metrics["trend"],
        "rate": metrics["rate"],
        "micro_pred": preds.get("micro", anomaly_score),
        "mid_pred": preds.get("mid", anomaly_score),
        "macro_pred": preds.get("macro", anomaly_score)
    }

    return fused, model_conf, heatmap

# =========================
# 🧮 GPU-Accelerated Anomaly Core
# =========================

anomaly_lock = threading.Lock()
anomaly_score = 0.0

def compute_anomaly_score():
    with event_history_lock:
        if len(event_history) < 10:
            return 0.0
        sev_map = {"info": 1.0, "warn": 2.0, "crit": 4.0}
        now = datetime.now()
        times = []
        weights = []
        for e in event_history[-200:]:
            try:
                ts = datetime.fromisoformat(e["ts"])
            except Exception:
                continue
            dt = (now - ts).total_seconds()
            if dt > 600:
                continue
            times.append(max(1.0, 600.0 - dt))
            weights.append(sev_map.get(e["severity"], 1.0))

    if not times:
        return 0.0

    try:
        t_arr = xp.asarray(times, dtype=xp.float32)
        w_arr = xp.asarray(weights, dtype=xp.float32)
        activity = w_arr * t_arr
        mean = activity.mean()
        std = activity.std()
        if float(std) == 0.0:
            score = float(mean)
        else:
            score = float((activity[-1] - mean) / (std + 1e-6))
        score = max(0.0, min(100.0, (score + 3.0) * (100.0 / 6.0)))
    except Exception:
        score = 0.0

    return score

# =========================
# 🧠 Meta-State Machine / Self-Integrity / Micro-Recovery / Auto-Tuning / Auto-Calibration
# =========================

meta_state_lock = threading.Lock()
meta_state_momentum = {
    "Hyper-Flow": 0.0,
    "Deep-Dream": 0.0,
    "Sentinel": 1.0,
    "Recovery-Flow": 0.0
}

def update_meta_state(fused_risk, regime):
    with meta_state_lock, brain_lock:
        current = brain_state.get("meta_state", "Sentinel")
        for k in meta_state_momentum:
            meta_state_momentum[k] *= 0.9

        if fused_risk > 70 or regime == "chaotic":
            meta_state_momentum["Hyper-Flow"] += 0.3
            meta_state_momentum["Sentinel"] += 0.1
        elif fused_risk < 30 and regime in ("stable", "cooling"):
            meta_state_momentum["Deep-Dream"] += 0.2
            meta_state_momentum["Recovery-Flow"] += 0.1
        else:
            meta_state_momentum["Sentinel"] += 0.2

        total = sum(meta_state_momentum.values()) + 1e-6
        for k in meta_state_momentum:
            meta_state_momentum[k] /= total

        new_state = max(meta_state_momentum.items(), key=lambda x: x[1])[0]

        if current == "Hyper-Flow" and fused_risk < 50:
            if meta_state_momentum["Recovery-Flow"] < 0.3:
                new_state = "Hyper-Flow"
        if current == "Sentinel" and fused_risk > 60:
            if regime != "stable":
                new_state = "Sentinel"

        brain_state["meta_state"] = new_state
        return new_state

def self_integrity_organ():
    now = datetime.now()
    with watchdog_lock:
        stale_count = 0
        for k, last in watchdog_state.items():
            delta = (now - last).total_seconds()
            if delta > WATCHDOG_THRESHOLDS.get(k, 60):
                stale_count += 1
    with brain_lock:
        integrity = 1.0
        integrity -= min(0.7, stale_count * 0.15)
        a = brain_state.get("last_anomaly_score", 0.0)
        if a < 0 or a > 100:
            integrity -= 0.3
        integrity = max(0.0, min(1.0, integrity))
        brain_state["integrity_score"] = integrity
        if integrity < 0.4:
            brain_state["meta_state"] = "Sentinel"
            brain_state["horizons"]["macro"] = 60.0
            brain_state["dampening"]["macro"] = 0.9

    root.after(20000, self_integrity_organ)

def micro_recovery_loops():
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
    except Exception:
        cpu = ram = 0.0

    with brain_lock:
        appetite = brain_state["appetite"]
        damp = brain_state["dampening"]

        if cpu > 80:
            appetite["ingest"] *= 0.9
            appetite["thread_expansion"] *= 0.9
        else:
            appetite["ingest"] = min(1.5, appetite["ingest"] * 1.01)

        if ram > 80:
            appetite["deep_ram"] *= 0.9
        else:
            appetite["deep_ram"] = min(1.5, appetite["deep_ram"] * 1.01)

        if cpu > 90 or ram > 90:
            damp["macro"] = min(0.95, damp["macro"] + 0.02)
        else:
            damp["macro"] = max(0.6, damp["macro"] - 0.01)

    root.after(5000, micro_recovery_loops)

def auto_tuning():
    with brain_lock:
        health = brain_state.get("health_score", 1.0)
        stance = brain_state["stance_thresholds"]
        horizons = brain_state["horizons"]
        damp = brain_state["dampening"]

        if health > 0.8:
            stance["low"] = max(15.0, stance["low"] - 1.0)
            stance["high"] = min(85.0, stance["high"] + 1.0)
            horizons["macro"] = min(240.0, horizons["macro"] + 5.0)
        elif health < 0.5:
            stance["low"] = min(35.0, stance["low"] + 1.0)
            stance["high"] = max(65.0, stance["high"] - 1.0)
            horizons["macro"] = max(60.0, horizons["macro"] - 5.0)
            damp["macro"] = min(0.95, damp["macro"] + 0.02)

    root.after(60000, auto_tuning)

def auto_calibration():
    with event_history_lock:
        hist = list(event_history)
    if not hist:
        root.after(3600000, auto_calibration)
        return
    sev_map = {"info": 1.0, "warn": 2.0, "crit": 4.0}
    vals = [sev_map.get(e["severity"], 1.0) for e in hist]
    arr = xp.asarray(vals, dtype=xp.float32)
    mean = float(arr.mean())
    std = float(arr.std())

    with brain_lock:
        stance = brain_state["stance_thresholds"]
        stance["low"] = max(5.0, min(40.0, mean * 10.0))
        stance["medium"] = max(20.0, min(60.0, (mean + std) * 10.0))
        stance["high"] = max(40.0, min(90.0, (mean + 2 * std) * 10.0))
        brain_state["last_calibration"] = datetime.now().isoformat()

    record_event("calibration", "Auto-calibration completed", severity="info")
    root.after(3600000, auto_calibration)

# =========================
# 🌐 Swarm Sync Mesh
# =========================

SWARM_PORT = 40444
swarm_sock = None

def init_swarm_socket():
    global swarm_sock
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception:
            pass
        sock.bind(("", SWARM_PORT))
        swarm_sock = sock
        record_event("swarm", f"Swarm socket bound on port {SWARM_PORT}", severity="info")
    except Exception as e:
        record_event("swarm", f"Failed to init swarm socket: {e}", severity="warn")

def swarm_heartbeat():
    if swarm_sock is None:
        return
    with brain_lock:
        node_id = brain_state.get("node_id")
        if not node_id:
            node_id = uuid.uuid4().hex[:12]
            brain_state["node_id"] = node_id
        score = brain_state.get("last_anomaly_score", 0.0)
        health = brain_state.get("health_score", 1.0)
        meta = brain_state.get("meta_state", "Sentinel")
    payload = json.dumps({
        "node_id": node_id,
        "swarm_id": brain_state.get("swarm_id"),
        "score": score,
        "health": health,
        "meta_state": meta,
        "ts": datetime.now().isoformat()
    }).encode("utf-8")
    try:
        swarm_sock.sendto(payload, ("255.255.255.255", SWARM_PORT))
    except Exception:
        pass
    watchdog_touch("swarm_heartbeat")
    root.after(7000, swarm_heartbeat)

def swarm_listener():
    if swarm_sock is None:
        return
    while True:
        try:
            data, addr = swarm_sock.recvfrom(4096)
            msg = json.loads(data.decode("utf-8"))
            nid = msg.get("node_id")
            if not nid:
                continue
            with brain_lock:
                if nid == brain_state.get("node_id"):
                    continue
                brain_state.setdefault("remote_nodes", {})[nid] = {
                    "addr": addr[0],
                    "score": msg.get("score", 0.0),
                    "health": msg.get("health", 1.0),
                    "meta_state": msg.get("meta_state"),
                    "swarm_id": msg.get("swarm_id"),
                    "ts": msg.get("ts")
                }
        except Exception:
            time.sleep(1.0)

def start_swarm_listener_thread():
    t = threading.Thread(target=swarm_listener, daemon=True)
    t.start()

# =========================
# 🛡️ Self-Healing Watchdog
# =========================

watchdog_lock = threading.Lock()
watchdog_state = {
    "threat_scan": datetime.now(),
    "destruction": datetime.now(),
    "self_check": datetime.now(),
    "anomaly": datetime.now(),
    "swarm_heartbeat": datetime.now()
}
WATCHDOG_THRESHOLDS = {
    "threat_scan": 30,
    "destruction": 30,
    "self_check": 45,
    "anomaly": 40,
    "swarm_heartbeat": 40
}

def watchdog_touch(name):
    with watchdog_lock:
        watchdog_state[name] = datetime.now()

def watchdog_loop():
    now = datetime.now()
    with watchdog_lock:
        stale = []
        for k, last in watchdog_state.items():
            delta = (now - last).total_seconds()
            if delta > WATCHDOG_THRESHOLDS.get(k, 60):
                stale.append((k, delta))
    for name, delta in stale:
        record_event("watchdog", f"{name} stale ({delta:.1f}s) - scheduling recovery", severity="warn", tags=["watchdog"])
        if name == "threat_scan":
            root.after(1000, threat_scan_and_respond)
        elif name == "destruction":
            root.after(1000, check_destruction)
        elif name == "self_check":
            root.after(1000, self_check)
        elif name == "anomaly":
            root.after(1000, hybrid_brain_loop)
        elif name == "swarm_heartbeat":
            root.after(1000, swarm_heartbeat)
        watchdog_touch(name)
    root.after(10000, watchdog_loop)

# =========================
# 🧙‍♂️ GUI Setup
# =========================

root = tk.Tk()
root.title("MagicBox Guardian: Chameleon ASI Sentinel")
root.geometry("1200x900")
root.configure(bg="#1e1e2f")

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=("Consolas", 11), background="#1e1e2f", foreground="#00ffcc")
style.configure("TButton", font=("Segoe UI", 10), padding=5)

root.bind_all("<Key>", mark_local_input)
root.bind_all("<Button>", mark_local_input)

mac_var = tk.StringVar()
ip_var = tk.StringVar()
telemetry_var = tk.StringVar()
hallucination_var = tk.StringVar()
swarm_var = tk.StringVar()
gpu_var = tk.StringVar(value=f"GPU: {'ON (CuPy)' if GPU_ENABLED else 'OFF (NumPy)'}")
anomaly_var = tk.StringVar(value="Anomaly: 0.00")
meta_state_var = tk.StringVar(value="Meta-State: Sentinel")
regime_var = tk.StringVar(value="Regime: stable")
health_var = tk.StringVar(value="Health: 1.00")
integrity_var = tk.StringVar(value="Integrity: 1.00")
confidence_var = tk.StringVar(value="Confidence: 0.50")
operator_label = tk.StringVar(value="Operator: Inactive / Remote-only")
id_label = tk.StringVar(value="Identifiers: [protected]")

top_frame = ttk.Frame(root)
top_frame.pack(pady=5, fill="x")

ttk.Label(top_frame, textvariable=mac_var).grid(row=0, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=ip_var).grid(row=1, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=telemetry_var).grid(row=2, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=hallucination_var).grid(row=3, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=swarm_var).grid(row=4, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=gpu_var).grid(row=5, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=anomaly_var).grid(row=6, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=meta_state_var).grid(row=7, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=regime_var).grid(row=8, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=health_var).grid(row=9, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=integrity_var).grid(row=10, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=confidence_var).grid(row=11, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=operator_label).grid(row=12, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=id_label).grid(row=13, column=0, padx=5, sticky="w")

ttk.Label(root, text="🧾 Mutation Trail Log (Last 10)").pack(pady=5)
log_text = tk.Text(root, height=10, width=140, bg="#2e2e3f", fg="#00ffcc", font=("Consolas", 10))
log_text.pack()
GUARDIAN_LOG_WIDGET = log_text

ttk.Label(root, text="🛠️ Live Config Panel").pack(pady=5)
config_frame = ttk.Frame(root)
config_frame.pack()

def update_config(tag, action_var, ttl_var):
    action = action_var.get()
    try:
        ttl = int(ttl_var.get()) if ttl_var.get() else None
    except:
        ttl = None
    trust_config[tag] = {"action": action, "ttl": ttl}
    record_event("config", f"Config Updated: {tag} → {action}, TTL={ttl}", severity="info")
    update_log()
    save_brain()

def make_update_callback(t, a_var, ttl_var):
    return lambda: update_config(t, a_var, ttl_var)

def build_config_panel():
    for widget in config_frame.winfo_children():
        widget.destroy()
    for i, tag in enumerate(trust_config.keys()):
        ttk.Label(config_frame, text=tag).grid(row=i, column=0, padx=5, pady=2)

        action_var = tk.StringVar(value=trust_config[tag]["action"])
        action_menu = ttk.Combobox(config_frame, textvariable=action_var, values=["destroy", "cloak", "preserve"], width=10)
        action_menu.grid(row=i, column=1)

        ttl_value = str(trust_config[tag]["ttl"]) if trust_config[tag]["ttl"] is not None else ""
        ttl_var = tk.StringVar(value=ttl_value)
        ttl_entry = ttk.Entry(config_frame, textvariable=ttl_var, width=10)
        ttl_entry.grid(row=i, column=2)

        update_button = ttk.Button(config_frame, text="Update", command=make_update_callback(tag, action_var, ttl_var))
        update_button.grid(row=i, column=3)

policy_frame = ttk.LabelFrame(root, text="Guardian Policy")
policy_frame.pack(pady=5, fill="x")

block_remote_var = tk.BooleanVar(value=True)
block_fw_var = tk.BooleanVar(value=True)
block_settings_var = tk.BooleanVar(value=True)

def update_policy():
    with state_lock:
        state["policy"]["block_remote_control"] = block_remote_var.get()
        state["policy"]["block_firewall_changes"] = block_fw_var.get()
        state["policy"]["block_settings_changes"] = block_settings_var.get()
    guardian_log_event("policy", f"Policy updated: remote={block_remote_var.get()}, fw={block_fw_var.get()}, settings={block_settings_var.get()}", "info")

ttk.Checkbutton(policy_frame, text="Block remote control tools", variable=block_remote_var, command=update_policy).pack(anchor="w", padx=5)
ttk.Checkbutton(policy_frame, text="Block firewall changes", variable=block_fw_var, command=update_policy).pack(anchor="w", padx=5)
ttk.Checkbutton(policy_frame, text="Block system settings changes", variable=block_settings_var, command=update_policy).pack(anchor="w", padx=5)

overlay_frame = ttk.Frame(root)
overlay_frame.pack(pady=10)

canvas = tk.Canvas(overlay_frame, width=260, height=260, bg="#151525", highlightthickness=0)
canvas.pack()

overlay_circle = canvas.create_oval(30, 30, 230, 230, outline="#00ffcc", width=3)
overlay_pulse = canvas.create_oval(80, 80, 180, 180, outline="#00ffcc", width=2)
overlay_text = canvas.create_text(130, 130, text="0.00", fill="#00ffcc", font=("Consolas", 18, "bold"))

overlay_phase = 0.0

def update_overlay_from_anomaly(score):
    global overlay_phase
    overlay_phase += 0.3
    r = int(min(255, max(0, (score / 100.0) * 255)))
    g = int(min(255, max(0, (1.0 - score / 100.0) * 255)))
    color = f"#{r:02x}{g:02x}00"
    base = 80
    amp = 10 + (score / 10.0)
    try:
        osc = float(xp.cos(overlay_phase))
    except Exception:
        osc = 0.0
    offset = amp * (0.5 + 0.5 * (1 + osc) / 2)
    x0, y0 = 130 - (base + offset), 130 - (base + offset)
    x1, y1 = 130 + (base + offset), 130 + (base + offset)
    canvas.coords(overlay_circle, x0, y0, x1, y1)
    canvas.itemconfig(overlay_circle, outline=color)
    canvas.itemconfig(overlay_pulse, outline=color)
    canvas.itemconfig(overlay_text, text=f"{score:.2f}", fill=color)
    anomaly_var.set(f"Anomaly: {score:.2f}")

def overlay_anim_loop():
    with anomaly_lock:
        score = anomaly_score
    update_overlay_from_anomaly(score)
    root.after(200, overlay_anim_loop)

# =========================
# 🧠 Hybrid Brain Loop
# =========================

def hybrid_brain_loop():
    global anomaly_score
    base_score = compute_anomaly_score()
    with anomaly_lock:
        anomaly_score = base_score

    metrics = compute_event_metrics()
    regime = detect_regime(metrics)
    preds = multi_horizon_prediction(metrics, base_score)
    fused_risk, model_conf, heatmap = fuse_predictions(preds, regime, base_score, metrics)
    meta = update_meta_state(fused_risk, regime)

    with brain_lock:
        integrity = brain_state.get("integrity_score", 1.0)
        health = max(0.0, min(1.0, (1.0 - fused_risk / 100.0) * 0.7 + integrity * 0.3))
        brain_state["last_anomaly_score"] = base_score
        brain_state["health_score"] = health
        brain_state["prediction_confidence"] = model_conf
        brain_state["reasoning_heatmap"] = heatmap
        brain_state["regime"] = regime

    meta_state_var.set(f"Meta-State: {meta}")
    regime_var.set(f"Regime: {regime}")
    health_var.set(f"Health: {health:.2f}")
    integrity_var.set(f"Integrity: {integrity:.2f}")
    confidence_var.set(f"Confidence: {model_conf:.2f}")

    record_event("hybrid_brain", f"Risk={fused_risk:.2f}, meta={meta}, regime={regime}", severity="info")

    watchdog_touch("anomaly")
    root.after(8000, hybrid_brain_loop)

# =========================
# 🧬 Identifier Refresh / Operator Status
# =========================

def refresh_identifiers():
    mac = get_real_mac()
    local_ip, public_ip = get_real_ip()
    encoded_mac = chameleon_wrap("MAC", mac)
    encoded_ip = chameleon_wrap("IP", local_ip)
    encoded_host = chameleon_wrap("HOST", socket.gethostname())
    mac_var.set(f"🦎 MAC: {encoded_mac}")
    ip_var.set(f"🌐 IP: {encoded_ip} | {chameleon_wrap('PUB', public_ip)}")
    id_label.set(f"Identifiers: MAC={encoded_mac}, IP={encoded_ip}, HOST={encoded_host}")
    root.after(60000, refresh_identifiers)

def refresh_operator_status():
    active = is_local_operator_active()
    operator_label.set(f"Operator: {'Active (local)' if active else 'Inactive / Remote-only'}")
    root.after(2000, refresh_operator_status)

# =========================
# 🚀 Autonomous Startup
# =========================

def autonomous_start():
    load_brain()
    load_state()

    mac = get_real_mac()
    mac_var.set(f"🦎 MAC: {chameleon_wrap('MAC', mac)}")
    handle_data("MAC", mac)

    local_ip, public_ip = get_real_ip()
    ip_var.set(f"🌐 IP: {chameleon_wrap('IP', local_ip)} | {chameleon_wrap('PUB', public_ip)}")
    handle_data("IP", f"{local_ip} | {public_ip}")

    os_info, browser_fp = get_telemetry()
    telemetry_var.set(f"🧢 Telemetry: {chameleon_wrap('OS', os_info)} | {chameleon_wrap('FP', browser_fp)}")
    handle_data("Telemetry", f"{os_info} | {browser_fp}")

    swarm_id = get_swarm_id()
    swarm_var.set(f"🔗 Swarm ID: {chameleon_wrap('SWARM', swarm_id)}")
    handle_data("SwarmID", swarm_id)

    phantom = synthesize_phantom()
    hallucination_var.set(f"👻 Phantom: {chameleon_wrap('PHANTOM', phantom)}")
    handle_data("Phantom", phantom)

    build_config_panel()
    update_log()
    save_brain()
    guardian_log_event("system", "Guardian policy engine online", "info")

# =========================
# 🔌 Initialize Swarm Mesh
# =========================

init_swarm_socket()
start_swarm_listener_thread()

# =========================
# 🧠 Trigger autonomous startup and defense cycles
# =========================

root.after(100, autonomous_start)
root.after(5000, check_destruction)
root.after(6000, monitor_network)
root.after(7000, monitor_processes)
root.after(8000, threat_scan_and_respond)
root.after(9000, hybrid_brain_loop)
root.after(10000, swarm_heartbeat)
root.after(11000, periodic_brain_save)
root.after(12000, periodic_save_guardian)
root.after(13000, self_check)
root.after(14000, watchdog_loop)
root.after(15000, self_integrity_organ)
root.after(16000, micro_recovery_loops)
root.after(17000, auto_tuning)
root.after(18000, auto_calibration)
root.after(19000, refresh_identifiers)
root.after(20000, refresh_operator_status)
root.after(200, overlay_anim_loop)

# =========================
# 🌀 Start GUI loop
# =========================

root.mainloop()
