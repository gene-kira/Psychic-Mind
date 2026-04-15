#!/usr/bin/env python3
"""
Universal Guardian Cockpit v8 – Coordinator-Driven Swarm Organism (Monolithic)

Includes:
- Persistent brain (agent weights, long-term stats, enforcement count, mutation log)
- Config system (thresholds & policies)
- Monitoring daemon (processes, drives, network)
- Multi-OS Wi-Fi trust engine
- Threat scoring & anomaly detection
- AI mode engine (LEARNING / RELAXED / STRICT)
- Enforcement (defensive only):
  - Kill suspicious processes
  - Quarantine executables
  - Block/suggest firewall rules
- GUI cockpit (Tkinter) for live operator view
- Swarm layer (Coordinator-Driven Hybrid):
  - Node identity
  - Local HTTP swarm API
  - Heartbeats to coordinator
  - Alert broadcast to coordinator
  - Coordinator aggregates alerts and can push policy updates
"""

import sys
import os
import time
import json
import threading
import subprocess
import shutil
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request

# ============================================================
# =============== PERSISTENT BRAIN (INLINE) ==================
# ============================================================

BRAIN_FILE = "guardian_brain.json"

_default_brain = {
    "agent_weights": [0.6, -0.8, -0.3],
    "long_term_conn_avg": None,
    "enforcement_count": 0,
    "mutation_log": [],
    "created_at": None,
    "updated_at": None,
}

_brain_state: Dict[str, Any] = {}


def _brain_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def load_brain() -> Dict[str, Any]:
    global _brain_state
    if not os.path.exists(BRAIN_FILE):
        state = dict(_default_brain)
        ts = _brain_now_iso()
        state["created_at"] = ts
        state["updated_at"] = ts
        _brain_state = state
        save_brain()
        return _brain_state

    try:
        with open(BRAIN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    state = dict(_default_brain)
    for k, v in data.items():
        if k in state:
            state[k] = v

    if state.get("created_at") is None:
        state["created_at"] = _brain_now_iso()
    if state.get("updated_at") is None:
        state["updated_at"] = _brain_now_iso()

    _brain_state = state
    return _brain_state


def save_brain() -> None:
    global _brain_state
    if not _brain_state:
        return
    _brain_state["updated_at"] = _brain_now_iso()
    try:
        with open(BRAIN_FILE, "w", encoding="utf-8") as f:
            json.dump(_brain_state, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def get_brain_state() -> Dict[str, Any]:
    if not _brain_state:
        load_brain()
    return _brain_state


def get_agent_weights() -> List[float]:
    state = get_brain_state()
    w = state.get("agent_weights")
    if not isinstance(w, list) or len(w) != 3:
        w = [0.6, -0.8, -0.3]
        state["agent_weights"] = w
        save_brain()
    return list(w)


def set_agent_weights(weights: List[float]) -> None:
    state = get_brain_state()
    state["agent_weights"] = list(weights)
    save_brain()


def mutate_agent_weights(reason: str, delta: List[float]) -> None:
    state = get_brain_state()
    old = get_agent_weights()
    new = [w + d for w, d in zip(old, delta)]
    state["agent_weights"] = new

    log = state.get("mutation_log")
    if not isinstance(log, list):
        log = []
    log.append({
        "ts": _brain_now_iso(),
        "reason": reason,
        "old": old,
        "new": new,
    })
    state["mutation_log"] = log
    save_brain()


def record_long_term_conn(current_conn_count: int) -> Dict[str, Any]:
    state = get_brain_state()
    lt = state.get("long_term_conn_avg")
    if lt is None:
        lt = float(current_conn_count)
    else:
        lt = (float(lt) * 0.99) + (float(current_conn_count) * 0.01)
    state["long_term_conn_avg"] = lt
    save_brain()
    return {
        "long_term_avg": lt,
        "current": current_conn_count,
    }


def increment_enforcement_count() -> int:
    state = get_brain_state()
    count = int(state.get("enforcement_count", 0)) + 1
    state["enforcement_count"] = count
    save_brain()
    return count


# ============================================================
# =============== CONFIG / POLICY (INLINE) ===================
# ============================================================

CONFIG_FILE = "guardian_config.json"

_default_config: Dict[str, Any] = {
    "network": {
        "high_conn_base": 150,
        "high_conn_min": 50,
        "network_spike_hard_limit": 300
    },
    "threat": {
        "score_enforce_strict": 5,
        "score_sandbox_candidate": 8
    },
    "ai": {
        "allow_enforcement_in_relaxed": False,
        "allow_enforcement_in_learning": False
    },
    "swarm": {
        "enabled": True,
        "coordinator": False,
        "port": 8787,
        "peers": []  # list of "http://ip:8787"
    }
}

_config_state: Dict[str, Any] = {}


def load_config() -> Dict[str, Any]:
    global _config_state
    if not os.path.exists(CONFIG_FILE):
        _config_state = dict(_default_config)
        return _config_state

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    merged = dict(_default_config)

    def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                dst[k] = deep_merge(dst[k], v)
            else:
                dst[k] = v
        return dst

    merged = deep_merge(merged, data)
    _config_state = merged
    return _config_state


def get_config() -> Dict[str, Any]:
    if not _config_state:
        load_config()
    return _config_state


def _cfg_get_nested(path: str, default: Any) -> Any:
    parts = path.split(".")
    cur: Any = get_config()
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def get_threshold(path: str, default: float) -> float:
    val = _cfg_get_nested(path, default)
    try:
        return float(val)
    except Exception:
        return float(default)


def get_flag(path: str, default: bool) -> bool:
    val = _cfg_get_nested(path, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(val, (int, float)):
        return bool(val)
    return bool(default)


def get_list(path: str, default: List[Any]) -> List[Any]:
    val = _cfg_get_nested(path, default)
    if isinstance(val, list):
        return val
    return list(default)


# ============================================================
# =============== MAIN ORGANISM STARTS HERE ==================
# ============================================================

# Swarm config (resolved after load_config)
NODE_ID = str(uuid.uuid4())
SWARM_ENABLED = False
SWARM_COORDINATOR = False
SWARM_PORT = 8787
SWARM_PEERS: List[str] = []

# ---------- Optional Windows auto-elevation ----------
if os.name == "nt":
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
                    1,
                )
                sys.exit()
        except Exception as e:
            print(f"[Guardian] Elevation failed: {e}")
            sys.exit()

    ensure_admin()

# ---------- Dependency check ----------
REQUIRED_MODULES = ["psutil"]

def ensure_dependencies():
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print("\n[!] Missing modules:", ", ".join(missing))
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)

ensure_dependencies()
import psutil

# ---------- GUI imports ----------
try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import scrolledtext
except ImportError:
    tk = None
    ttk = None
    scrolledtext = None
    print("[!] Tkinter not available. GUI disabled.")

# ---------- Core config ----------
LOG_FILE = "guardian_daemon.log"
TRUST_DB_FILE = "guardian_trust.json"
QUARANTINE_DIR = "guardian_quarantine"

WIFI_CHECK_INTERVAL = 15
MONITOR_INTERVAL = 5
NETWORK_SAMPLE_LIMIT = 80

# ---------- Logging ----------
_log_buffer: List[str] = []
_log_lock = threading.Lock()
_LOG_MAX = 500

def log_event(level: str, message: str, data: Optional[Dict[str, Any]] = None):
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level.upper(),
        "message": message,
        "data": data or {},
    }
    line = json.dumps(entry, ensure_ascii=False)
    print(line)

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

    with _log_lock:
        _log_buffer.append(line)
        if len(_log_buffer) > _LOG_MAX:
            _log_buffer[:] = _log_buffer[-_LOG_MAX:]


def get_log_buffer() -> List[str]:
    with _log_lock:
        return list(_log_buffer)

# ---------- Trust DB ----------
def load_trust_db():
    if not os.path.exists(TRUST_DB_FILE):
        return {"wifi": {}}
    try:
        with open(TRUST_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"wifi": {}}


def save_trust_db(db):
    try:
        with open(TRUST_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)
    except Exception:
        pass

# ---------- Enforcement primitives ----------
def ensure_quarantine_dir():
    q = os.path.abspath(QUARANTINE_DIR)
    os.makedirs(q, exist_ok=True)
    return q


def enforce_kill(pid: int, reason: str):
    try:
        p = psutil.Process(pid)
    except Exception as e:
        log_event("ERROR", "Process lookup failed for kill", {"pid": pid, "error": str(e)})
        return
    try:
        log_event("WARN", "Killing process", {"pid": pid, "reason": reason, "name": p.name()})
        p.terminate()
        try:
            p.wait(2)
        except Exception:
            p.kill()
    except Exception as e:
        log_event("ERROR", "Kill failed", {"pid": pid, "error": str(e)})


def enforce_quarantine(path: str, reason: str):
    if not path or not os.path.exists(path):
        return None
    qdir = ensure_quarantine_dir()
    base = os.path.basename(path)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    qpath = os.path.join(qdir, f"{ts}_{base}")
    try:
        shutil.move(path, qpath)
        log_event("WARN", "File quarantined", {"src": path, "dst": qpath, "reason": reason})
        return qpath
    except Exception as e:
        log_event("ERROR", "Quarantine failed", {"path": path, "error": str(e)})
        return None


def enforce_firewall(ip: str, reason: str):
    if not ip:
        return
    if os.name == "nt":
        cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name=GuardianBlock_{ip}",
            "dir=out",
            "action=block",
            f"remoteip={ip}",
        ]
        try:
            subprocess.run(cmd, capture_output=True)
            log_event("WARN", "Firewall block applied", {"ip": ip, "reason": reason})
        except Exception as e:
            log_event("ERROR", "Firewall block failed", {"ip": ip, "error": str(e)})
    else:
        log_event("WARN", "Firewall block suggestion", {
            "ip": ip,
            "cmd": f"iptables -A OUTPUT -d {ip} -j DROP",
            "reason": reason
        })

# ---------- Snapshots ----------
class ProcessSnapshot:
    def __init__(self):
        self.by_pid: Dict[int, Dict[str, Any]] = {}

    def capture(self):
        snap: Dict[int, Dict[str, Any]] = {}
        for p in psutil.process_iter(attrs=["pid", "name", "exe", "username", "ppid", "create_time"]):
            info = p.info
            snap[info["pid"]] = {
                "pid": info["pid"],
                "name": info.get("name") or "",
                "exe": info.get("exe") or "",
                "user": info.get("username") or "",
                "ppid": info.get("ppid"),
                "create_time": info.get("create_time"),
            }
        self.by_pid = snap

    def diff(self, old: "ProcessSnapshot"):
        added: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        old_pids = set(old.by_pid.keys())
        new_pids = set(self.by_pid.keys())
        for pid in new_pids - old_pids:
            added.append(self.by_pid[pid])
        for pid in old_pids - new_pids:
            removed.append(old.by_pid[pid])
        return added, removed


class DriveSnapshot:
    def __init__(self):
        self.by_mount: Dict[str, Dict[str, Any]] = {}

    def capture(self):
        snap: Dict[str, Dict[str, Any]] = {}
        for part in psutil.disk_partitions(all=False):
            m = part.mountpoint
            try:
                u = psutil.disk_usage(m)
            except OSError as e:
                log_event("ERROR", "Skipping incompatible drive", {"mount": m, "error": str(e)})
                continue
            except Exception as e:
                log_event("ERROR", "Skipping drive due to unexpected error", {"mount": m, "error": str(e)})
                continue
            snap[m] = {
                "mount": m,
                "fstype": part.fstype,
                "total": u.total,
                "used": u.used,
                "free": u.free,
                "percent": u.percent,
            }
        self.by_mount = snap

    def diff(self, old: "DriveSnapshot"):
        added: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        old_m = set(old.by_mount.keys())
        new_m = set(self.by_mount.keys())
        for m in new_m - old_m:
            added.append(self.by_mount[m])
        for m in old_m - new_m:
            removed.append(old.by_mount[m])
        return added, removed


class NetworkSnapshot:
    def __init__(self):
        self.connections: List[Dict[str, Any]] = []

    def capture(self):
        safe: List[Dict[str, Any]] = []
        try:
            raw = psutil.net_connections(kind="inet")
        except Exception as e:
            log_event("ERROR", "net_connections failed", {"error": str(e)})
            self.connections = []
            return
        for c in raw:
            try:
                l = ""
                r = ""
                if c.laddr:
                    l = f"{c.laddr.ip}:{c.laddr.port}"
                if c.raddr:
                    r = f"{c.raddr.ip}:{c.raddr.port}"
                safe.append({
                    "pid": c.pid,
                    "laddr": l,
                    "raddr": r,
                    "status": str(c.status),
                })
            except Exception:
                continue
        self.connections = safe

# ---------- Wi-Fi scanners ----------
def wifi_windows():
    try:
        out = subprocess.check_output(
            ["netsh", "wlan", "show", "interfaces"],
            text=True, errors="ignore"
        )
    except Exception as e:
        log_event("ERROR", "netsh wlan show interfaces failed", {"error": str(e)})
        return None
    ssid = None
    bssid = None
    for line in out.splitlines():
        line = line.strip()
        if line.lower().startswith("ssid") and "bssid" not in line.lower():
            parts = line.split(":", 1)
            if len(parts) == 2:
                ssid = parts[1].strip()
        if line.lower().startswith("bssid"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                bssid = parts[1].strip().lower()
    if ssid and bssid:
        return {"ssid": ssid, "bssid": bssid}
    return None


def wifi_linux():
    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "active,ssid,bssid", "dev", "wifi"],
            text=True, errors="ignore"
        )
        for line in out.splitlines():
            p = line.split(":")
            if len(p) >= 3 and p[0] == "yes":
                return {"ssid": p[1], "bssid": p[2].lower()}
    except Exception:
        pass
    return None


def wifi_macos():
    airport = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
    if not os.path.exists(airport):
        return None
    try:
        out = subprocess.check_output([airport, "-I"], text=True, errors="ignore")
    except Exception:
        return None
    ssid = None
    bssid = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("SSID:"):
            ssid = line.split(":", 1)[1].strip()
        if line.startswith("BSSID:"):
            bssid = line.split(":", 1)[1].strip().lower()
    if ssid and bssid:
        return {"ssid": ssid, "bssid": bssid}
    return None


def get_wifi():
    if os.name == "nt":
        return wifi_windows()
    if sys.platform.startswith("linux"):
        return wifi_linux()
    if sys.platform == "darwin":
        return wifi_macos()
    return None

# ---------- Wi-Fi evaluation + AI mode ----------
def evaluate_wifi_trust(wifi_info: Optional[Dict[str, Any]], trust_db: Dict[str, Any]) -> Tuple[str, str, bool, bool]:
    if wifi_info is None:
        return ("No Wi-Fi info", "INFO", False, False)

    ssid = wifi_info["ssid"]
    bssid = wifi_info["bssid"]
    wifi_db = trust_db.get("wifi", {})
    known = wifi_db.get(ssid, [])

    if ssid in wifi_db:
        if bssid in known:
            return (f"Trusted Wi-Fi: {ssid} ({bssid})", "OK", True, False)
        else:
            return (f"Known SSID but new BSSID {bssid}", "WARN", False, True)
    else:
        return (f"Unknown Wi-Fi SSID: {ssid} ({bssid})", "WARN", False, True)


def ai_select_mode(
    wifi_eval: Tuple[str, str, bool, bool],
    connection_count: int
) -> Tuple[str, str, bool]:
    msg, level, is_trusted, should_learn = wifi_eval
    weights = get_agent_weights()
    wifi_weight, conn_weight, _ = weights

    high_conn_base = get_threshold("network.high_conn_base", 150.0)
    high_conn_min = get_threshold("network.high_conn_min", 50.0)
    high_conn_threshold = int(high_conn_base + conn_weight * -50)
    high_conn_threshold = max(high_conn_min, high_conn_threshold)
    high_conn = connection_count > high_conn_threshold

    if level == "WARN":
        if should_learn:
            return ("LEARNING", f"Learning new Wi-Fi: {msg}", True)
        return ("STRICT", f"Wi-Fi warning: {msg}", False)

    if level == "OK":
        if high_conn:
            return ("STRICT", f"High connection count: {connection_count}", False)
        return ("RELAXED", "Trusted Wi-Fi, normal activity", False)

    if high_conn:
        return ("STRICT", "No Wi-Fi trust + high connections", False)

    return ("RELAXED", "Normal conditions", False)

# ---------- Threat scoring & anomaly detection ----------
def threat_score_process(proc: Dict[str, Any], conns: List[Dict[str, Any]]) -> int:
    score = 0
    name = (proc.get("name") or "").lower()
    exe = (proc.get("exe") or "").lower()
    ppid = proc.get("ppid")
    create_time = proc.get("create_time")

    if "temp" in exe or "appdata" in exe:
        score += 2
    if name in ("powershell.exe", "cmd.exe", "wscript.exe", "cscript.exe"):
        score += 3
    if name.endswith(".exe") and "download" in exe:
        score += 2

    pid = proc.get("pid")
    outbound = []
    if pid is not None:
        outbound = [c for c in conns if c.get("pid") == pid and c.get("raddr")]
        if len(outbound) > 5:
            score += 3

    unique_ips = set()
    for c in outbound:
        raddr = c.get("raddr") or ""
        ip = raddr.split(":")[0] if ":" in raddr else raddr
        if ip:
            unique_ips.add(ip)
    if len(unique_ips) > 5:
        score += 2

    if create_time is not None:
        lifetime = time.time() - create_time
        if lifetime < 10:
            score += 1

    if ppid in (0, 1, None):
        score += 1

    return score


class AnomalyTracker:
    def __init__(self):
        self.conn_history: List[int] = []
        self.max_len = 100

    def update(self, conn_count: int) -> Dict[str, Any]:
        self.conn_history.append(conn_count)
        if len(self.conn_history) > self.max_len:
            self.conn_history = self.conn_history[-self.max_len:]

        avg = sum(self.conn_history) / len(self.conn_history)
        dev = conn_count - avg

        lt_info = record_long_term_conn(conn_count)

        anomaly = dev > max(30, avg * 0.8)
        return {
            "avg": avg,
            "current": conn_count,
            "delta": dev,
            "long_term_avg": lt_info["long_term_avg"],
            "anomaly": anomaly,
        }

# ---------- Swarm broadcast helpers ----------
def broadcast_swarm_alert(alert: Dict[str, Any]):
    if not SWARM_ENABLED:
        return
    payload = json.dumps({
        "node_id": NODE_ID,
        "alert": alert,
    }).encode("utf-8")
    for peer in SWARM_PEERS:
        try:
            req = urllib.request.Request(
                peer + "/swarm/alert",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            continue


def broadcast_swarm_policy(policy: Dict[str, Any]):
    if not SWARM_ENABLED or not SWARM_COORDINATOR:
        return
    payload = json.dumps({
        "node_id": NODE_ID,
        "policy": policy,
    }).encode("utf-8")
    for peer in SWARM_PEERS:
        try:
            req = urllib.request.Request(
                peer + "/swarm/policy",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            continue

# ---------- Rule engine + enforcement ----------
class RuleEngine:
    def __init__(self, state_ref: "GuardianState"):
        self.alerts: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.state_ref = state_ref

    def add_alert(self, kind: str, msg: str, data: Dict[str, Any]) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "kind": kind,
            "msg": msg,
            "data": data,
        }
        with self.lock:
            self.alerts.append(entry)
            if len(self.alerts) > 200:
                self.alerts = self.alerts[-200:]
        log_event("WARN", f"ALERT: {kind} - {msg}", data)
        broadcast_swarm_alert(entry)

    def get_alerts(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.alerts)

    def evaluate(self, state: "GuardianState") -> None:
        status = state.get_status()
        conns = state.list_connections()
        procs = state.list_processes()
        wifi_eval = state.get_wifi_eval()
        ai_mode = status.get("ai_mode", "RELAXED")

        if wifi_eval:
            msg, level, _, _ = wifi_eval
            if level == "WARN":
                self.add_alert("WIFI", msg, {"wifi_eval": wifi_eval})

        hard_limit = int(get_threshold("network.network_spike_hard_limit", 300.0))
        if status["connection_count"] > hard_limit:
            self.add_alert(
                "NETWORK_SPIKE",
                f"Very high connection count: {status['connection_count']}",
                {"status": status},
            )

        score_enforce = int(get_threshold("threat.score_enforce_strict", 5.0))
        score_sandbox = int(get_threshold("threat.score_sandbox_candidate", 8.0))
        allow_relaxed = get_flag("ai.allow_enforcement_in_relaxed", False)
        allow_learning = get_flag("ai.allow_enforcement_in_learning", False)

        for p in procs:
            score = threat_score_process(p, conns)
            if score >= score_enforce:
                sandbox_flag = score >= score_sandbox
                self.add_alert(
                    "PROCESS_SUSPICIOUS",
                    f"Process {p['name']} (PID {p['pid']}) score={score}",
                    {"proc": p, "score": score, "sandbox_candidate": sandbox_flag},
                )

                enforce = False
                if ai_mode == "STRICT":
                    enforce = True
                elif ai_mode == "RELAXED" and allow_relaxed:
                    enforce = True
                elif ai_mode == "LEARNING" and allow_learning:
                    enforce = True

                if enforce:
                    self.enforce_on_process(p, conns, score, sandbox_flag, ai_mode)
                else:
                    log_event("INFO", "Enforcement skipped due to AI mode policy", {
                        "proc": p,
                        "score": score,
                        "ai_mode": ai_mode,
                    })

    def enforce_on_process(
        self,
        proc: Dict[str, Any],
        conns: List[Dict[str, Any]],
        score: int,
        sandbox_flag: bool,
        ai_mode: str
    ) -> None:
        pid = proc.get("pid")
        exe = proc.get("exe") or ""
        name = proc.get("name") or ""

        reason = f"Threat score={score} for process {name} (PID {pid}) in mode={ai_mode}"
        if sandbox_flag:
            reason += " [sandbox_candidate]"

        if pid is not None:
            enforce_kill(pid, reason)

        if exe and os.path.exists(exe):
            qpath = enforce_quarantine(exe, reason)
            if qpath:
                log_event("INFO", "Executable quarantined after kill", {"original": exe, "quarantine": qpath})

        outbound = [c for c in conns if c.get("pid") == pid and c.get("raddr")]
        seen_ips = set()
        for c in outbound:
            raddr = c.get("raddr") or ""
            ip = raddr.split(":")[0] if ":" in raddr else raddr
            if ip and ip not in seen_ips:
                seen_ips.add(ip)
                enforce_firewall(ip, f"Process {name} (PID {pid}) flagged as suspicious")

        increment_enforcement_count()
        mutate_agent_weights("enforcement_event", [0.02, -0.02, 0.01])

# ---------- Swarm HTTP server ----------
class SwarmHandler(BaseHTTPRequestHandler):
    def _json_response(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            data = {}

        if self.path == "/swarm/heartbeat":
            self.server.guardian_state.handle_swarm_heartbeat(data)
            self._json_response(200, {"ok": True})
        elif self.path == "/swarm/alert":
            self.server.guardian_state.handle_swarm_alert(data)
            self._json_response(200, {"ok": True})
        elif self.path == "/swarm/policy":
            self.server.guardian_state.handle_swarm_policy(data)
            self._json_response(200, {"ok": True})
        else:
            self._json_response(404, {"error": "unknown endpoint"})

    def log_message(self, format, *args):
        return  # silence default HTTP logging


class SwarmServer(HTTPServer):
    def __init__(self, addr, handler, guardian_state):
        super().__init__(addr, handler)
        self.guardian_state = guardian_state


def start_swarm_server(state):
    if not SWARM_ENABLED:
        return None
    server = SwarmServer(("", SWARM_PORT), SwarmHandler, state)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log_event("INFO", "Swarm HTTP server started", {"port": SWARM_PORT})
    return server

# ---------- Swarm heartbeat thread ----------
class SwarmThread(threading.Thread):
    def __init__(self, state: "GuardianState", interval_seconds: int = 10):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval_seconds
        self._stop_flag = threading.Event()

    def run(self) -> None:
        if not SWARM_ENABLED:
            return
        log_event("INFO", "Swarm thread started", {"interval": self.interval})
        while not self._stop_flag.is_set():
            try:
                self.send_heartbeat()
            except Exception as e:
                log_event("ERROR", "Swarm heartbeat error", {"error": str(e)})
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_flag.set()

    def send_heartbeat(self) -> None:
        status = self.state.get_status()
        payload = json.dumps({
            "node_id": NODE_ID,
            "status": status,
        }).encode("utf-8")
        for peer in SWARM_PEERS:
            try:
                req = urllib.request.Request(
                    peer + "/swarm/heartbeat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
            except Exception:
                continue

# ---------- Guardian state ----------
class GuardianState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_proc_snapshot = ProcessSnapshot()
        self.last_drive_snapshot = DriveSnapshot()
        self.last_net_snapshot = NetworkSnapshot()

        self.last_proc_snapshot.capture()
        self.last_drive_snapshot.capture()
        self.last_net_snapshot.capture()

        self.start_time = time.time()
        self.trust_db = load_trust_db()

        self.last_wifi_info: Optional[Dict[str, Any]] = None
        self.last_wifi_eval: Optional[Tuple[str, str, bool, bool]] = None

        self.ai_mode = "RELAXED"
        self.ai_reason = "Initial state."

        self.anomaly_tracker = AnomalyTracker()
        self.rule_engine = RuleEngine(self)

        self.node_id = NODE_ID
        self.known_nodes: Dict[str, Dict[str, Any]] = {}
        self.swarm_alerts: List[Dict[str, Any]] = []
        self.swarm_policies: Dict[str, Any] = {}

    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "uptime_seconds": self.uptime_seconds(),
                "process_count": len(self.last_proc_snapshot.by_pid),
                "drive_count": len(self.last_drive_snapshot.by_mount),
                "connection_count": len(self.last_net_snapshot.connections),
                "ai_mode": self.ai_mode,
                "ai_reason": self.ai_reason,
                "node_id": self.node_id,
                "known_nodes": list(self.known_nodes.keys()),
            }

    def list_processes(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.last_proc_snapshot.by_pid.values())

    def list_drives(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.last_drive_snapshot.by_mount.values())

    def list_connections(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.last_net_snapshot.connections)

    def get_wifi_eval(self) -> Optional[Tuple[str, str, bool, bool]]:
        with self.lock:
            return self.last_wifi_eval

    def get_alerts(self) -> List[Dict[str, Any]]:
        local_alerts = self.rule_engine.get_alerts()
        with self.lock:
            return local_alerts + list(self.swarm_alerts)

    def handle_swarm_heartbeat(self, data: Dict[str, Any]) -> None:
        nid = data.get("node_id")
        if not nid:
            return
        status = data.get("status", {})
        with self.lock:
            self.known_nodes[nid] = {
                "last_heartbeat": time.time(),
                "status": status,
            }

    def handle_swarm_alert(self, data: Dict[str, Any]) -> None:
        with self.lock:
            self.swarm_alerts.append({
                "ts": datetime.utcnow().isoformat() + "Z",
                "from": data.get("node_id"),
                "alert": data.get("alert"),
            })
            if len(self.swarm_alerts) > 200:
                self.swarm_alerts = self.swarm_alerts[-200:]

    def handle_swarm_policy(self, data: Dict[str, Any]) -> None:
        policy = data.get("policy", {})
        with self.lock:
            self.swarm_policies.update(policy)
        log_event("INFO", "Swarm policy update received", {"policy": self.swarm_policies})

    def update_processes(self) -> None:
        with self.lock:
            new = ProcessSnapshot()
            new.capture()
            added, removed = new.diff(self.last_proc_snapshot)
            self.last_proc_snapshot = new

        for p in added:
            log_event("INFO", "Process started", p)
        for p in removed:
            log_event("INFO", "Process ended", p)

    def update_drives(self) -> None:
        with self.lock:
            new = DriveSnapshot()
            new.capture()
            added, removed = new.diff(self.last_drive_snapshot)
            self.last_drive_snapshot = new

        for d in added:
            log_event("WARN", "Drive mounted", d)
        for d in removed:
            log_event("WARN", "Drive unmounted", d)

    def update_network(self) -> None:
        with self.lock:
            new = NetworkSnapshot()
            new.capture()
            self.last_net_snapshot = new

        anomaly = self.anomaly_tracker.update(len(self.last_net_snapshot.connections))
        if anomaly["anomaly"]:
            log_event("WARN", "Network anomaly detected", anomaly)
            mutate_agent_weights("network_anomaly", [0.05, -0.1, 0.0])

    def update_wifi_and_ai(self) -> None:
        wifi_info = get_wifi()
        wifi_eval = evaluate_wifi_trust(wifi_info, self.trust_db)

        msg, level, is_trusted, should_learn = wifi_eval

        if level == "WARN":
            log_event("WARN", "Wi-Fi trust warning", {"wifi": wifi_info, "msg": msg})
        else:
            log_event("INFO", "Wi-Fi status", {"wifi": wifi_info, "msg": msg})

        connection_count = len(self.last_net_snapshot.connections)
        mode, reason, learn_now = ai_select_mode(wifi_eval, connection_count)

        if learn_now and wifi_info:
            ssid = wifi_info["ssid"]
            bssid = wifi_info["bssid"]
            wifi_db = self.trust_db.setdefault("wifi", {})
            bssids = wifi_db.setdefault(ssid, [])
            if bssid not in bssids:
                bssids.append(bssid)
                save_trust_db(self.trust_db)
                log_event("INFO", "Wi-Fi fingerprint learned", {"ssid": ssid, "bssid": bssid})
                mutate_agent_weights("wifi_learn", [0.05, 0.0, 0.0])

        with self.lock:
            self.last_wifi_info = wifi_info
            self.last_wifi_eval = wifi_eval
            self.ai_mode = mode
            self.ai_reason = reason

        self.rule_engine.evaluate(self)

# ---------- Monitor threads ----------
class MonitorThread(threading.Thread):
    def __init__(self, state: GuardianState, interval_seconds: int = MONITOR_INTERVAL):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval_seconds
        self._stop_flag = threading.Event()

    def run(self) -> None:
        log_event("INFO", "Monitor thread started", {"interval": self.interval})
        while not self._stop_flag.is_set():
            try:
                self.state.update_processes()
                self.state.update_drives()
                self.state.update_network()
            except Exception as e:
                log_event("ERROR", "Monitor loop error", {"error": str(e)})
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_flag.set()


class WifiThread(threading.Thread):
    def __init__(self, state: GuardianState, interval_seconds: int = WIFI_CHECK_INTERVAL):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval_seconds
        self._stop_flag = threading.Event()

    def run(self) -> None:
        log_event("INFO", "Wi-Fi/AI monitor thread started", {"interval": self.interval})
        while not self._stop_flag.is_set():
            try:
                self.state.update_wifi_and_ai()
            except Exception as e:
                log_event("ERROR", "Wi-Fi/AI loop error", {"error": str(e)})
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_flag.set()

# ---------- GUI ----------
class GuardianGUI:
    def __init__(self, root: tk.Tk, state: GuardianState):
        self.root = root
        self.state = state

        root.title("Universal Guardian Cockpit v8 (Swarm)")
        root.geometry("1300x750")

        self.build_layout()
        self.refresh_ui()

    def build_layout(self) -> None:
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(self.main_frame, text="Status:", font=("Segoe UI", 11, "bold"))
        self.status_label.pack(anchor="w")

        self.ai_label = ttk.Label(self.main_frame, text="AI Mode:", font=("Segoe UI", 10, "bold"))
        self.ai_label.pack(anchor="w", pady=(3, 3))

        self.wifi_label = ttk.Label(self.main_frame, text="Wi-Fi:", font=("Segoe UI", 10))
        self.wifi_label.pack(anchor="w", pady=(3, 3))

        self.swarm_label = ttk.Label(self.main_frame, text="Swarm:", font=("Segoe UI", 10))
        self.swarm_label.pack(anchor="w", pady=(3, 10))

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Processes tab
        self.proc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.proc_frame, text="Processes")

        self.proc_tree = ttk.Treeview(
            self.proc_frame,
            columns=("pid", "name", "user", "exe"),
            show="headings"
        )
        for col, w in [("pid", 60), ("name", 200), ("user", 200), ("exe", 700)]:
            self.proc_tree.heading(col, text=col.upper())
            self.proc_tree.column(col, width=w, anchor="w")
        self.proc_tree.pack(fill=tk.BOTH, expand=True)

        # Drives tab
        self.drive_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.drive_frame, text="Drives")

        self.drive_tree = ttk.Treeview(
            self.drive_frame,
            columns=("mount", "fstype", "total", "used", "free", "percent"),
            show="headings"
        )
        for col, w in [
            ("mount", 160),
            ("fstype", 80),
            ("total", 160),
            ("used", 160),
            ("free", 160),
            ("percent", 80),
        ]:
            self.drive_tree.heading(col, text=col.upper())
            self.drive_tree.column(col, width=w, anchor="w")
        self.drive_tree.pack(fill=tk.BOTH, expand=True)

        # Network tab
        self.net_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.net_frame, text="Network")

        self.net_tree = ttk.Treeview(
            self.net_frame,
            columns=("pid", "laddr", "raddr", "status"),
            show="headings"
        )
        for col, w in [
            ("pid", 60),
            ("laddr", 320),
            ("raddr", 320),
            ("status", 120),
        ]:
            self.net_tree.heading(col, text=col.upper())
            self.net_tree.column(col, width=w, anchor="w")
        self.net_tree.pack(fill=tk.BOTH, expand=True)

        # Alerts tab
        self.alert_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.alert_frame, text="Alerts (Local + Swarm)")

        self.alert_tree = ttk.Treeview(
            self.alert_frame,
            columns=("ts", "kind", "msg"),
            show="headings"
        )
        for col, w in [
            ("ts", 200),
            ("kind", 120),
            ("msg", 700),
        ]:
            self.alert_tree.heading(col, text=col.upper())
            self.alert_tree.column(col, width=w, anchor="w")
        self.alert_tree.pack(fill=tk.BOTH, expand=True)

        # Logs tab
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Logs")

        if scrolledtext is not None:
            self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.NONE, height=10)
            self.log_text.pack(fill=tk.BOTH, expand=True)
            self.log_text.configure(state=tk.DISABLED)
        else:
            self.log_text = None

    def refresh_ui(self) -> None:
        try:
            status = self.state.get_status()
            processes = self.state.list_processes()
            drives = self.state.list_drives()
            conns = self.state.list_connections()
            alerts = self.state.get_alerts()
            logs = get_log_buffer()
        except Exception as e:
            log_event("ERROR", "GUI data fetch failed", {"error": str(e)})
            self.root.after(2000, self.refresh_ui)
            return

        self.status_label.config(
            text=(
                f"Status: node={status.get('node_id')} | uptime={int(status['uptime_seconds'])}s | "
                f"processes={status['process_count']} | drives={status['drive_count']} | "
                f"connections={status['connection_count']}"
            )
        )

        mode = status.get("ai_mode", "UNKNOWN")
        reason = status.get("ai_reason", "")
        color = {"STRICT": "red", "RELAXED": "green", "LEARNING": "orange"}.get(mode, "black")
        self.ai_label.config(text=f"AI Mode: {mode} — {reason}", foreground=color)

        wifi_eval = self.state.get_wifi_eval()
        if wifi_eval:
            msg, level, _, _ = wifi_eval
            wifi_color = {"OK": "green", "WARN": "red", "INFO": "black"}.get(level, "black")
            self.wifi_label.config(text=f"Wi-Fi: {msg}", foreground=wifi_color)
        else:
            self.wifi_label.config(text="Wi-Fi: (no data yet)", foreground="black")

        known_nodes = status.get("known_nodes", [])
        swarm_text = f"Swarm: {'ENABLED' if SWARM_ENABLED else 'DISABLED'} | role={'COORDINATOR' if SWARM_COORDINATOR else 'WORKER'} | known_nodes={len(known_nodes)}"
        self.swarm_label.config(text=swarm_text)

        for i in self.proc_tree.get_children():
            self.proc_tree.delete(i)
        for proc in processes:
            self.proc_tree.insert(
                "",
                tk.END,
                values=(proc["pid"], proc["name"], proc["user"], proc["exe"])
            )

        for i in self.drive_tree.get_children():
            self.drive_tree.delete(i)
        for d in drives:
            self.drive_tree.insert(
                "",
                tk.END,
                values=(
                    d["mount"],
                    d["fstype"],
                    d["total"],
                    d["used"],
                    d["free"],
                    f"{d['percent']}%",
                ),
            )

        for i in self.net_tree.get_children():
            self.net_tree.delete(i)
        for c in conns[:NETWORK_SAMPLE_LIMIT]:
            self.net_tree.insert(
                "",
                tk.END,
                values=(c["pid"], c["laddr"], c["raddr"], c["status"])
            )

        for i in self.alert_tree.get_children():
            self.alert_tree.delete(i)
        for a in alerts:
            if "alert" in a:
                ts = a.get("ts")
                kind = a["alert"].get("kind")
                msg = a["alert"].get("msg")
            else:
                ts = a.get("ts")
                kind = a.get("kind")
                msg = a.get("msg")
            self.alert_tree.insert(
                "",
                tk.END,
                values=(ts, kind, msg)
            )

        if self.log_text is not None:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert(tk.END, "\n".join(logs))
            self.log_text.configure(state=tk.DISABLED)

        self.root.after(2000, self.refresh_ui)

# ---------- Main ----------
def main() -> None:
    load_brain()
    load_config()

    global SWARM_ENABLED, SWARM_COORDINATOR, SWARM_PORT, SWARM_PEERS
    cfg = get_config()
    SWARM_ENABLED = bool(cfg.get("swarm", {}).get("enabled", True))
    SWARM_COORDINATOR = bool(cfg.get("swarm", {}).get("coordinator", False))
    SWARM_PORT = int(cfg.get("swarm", {}).get("port", 8787))
    SWARM_PEERS = get_list("swarm.peers", [])

    log_event("INFO", "Universal Guardian Cockpit v8 starting", {
        "brain": get_brain_state(),
        "config": cfg,
        "node_id": NODE_ID,
        "swarm_enabled": SWARM_ENABLED,
        "swarm_coordinator": SWARM_COORDINATOR,
        "swarm_port": SWARM_PORT,
        "swarm_peers": SWARM_PEERS,
    })

    state = GuardianState()

    swarm_server = start_swarm_server(state)
    swarm_thread = SwarmThread(state)
    monitor = MonitorThread(state)
    wifi_mon = WifiThread(state)

    monitor.start()
    wifi_mon.start()
    swarm_thread.start()

    if tk is not None and ttk is not None:
        root = tk.Tk()
        gui = GuardianGUI(root, state)
        try:
            root.mainloop()
        finally:
            log_event("INFO", "GUI closed by operator", {})
    else:
        print("[!] Running without GUI. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    monitor.stop()
    wifi_mon.stop()
    swarm_thread.stop()
    time.sleep(1)
    log_event("INFO", "Universal Guardian Cockpit v8 exited", {})


if __name__ == "__main__":
    main()