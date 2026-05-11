import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import importlib
import subprocess
import sys
import math
from collections import deque
import random
import hashlib
import hmac
import ctypes
import inspect

# =========================
# CONFIG FLAGS
# =========================

SILENT_ELEVATION = False          # if True, no messageboxes on elevation failure
FORCE_ELEVATE_FROZEN = True       # try to elevate even if running as frozen EXE
INTEGRITY_HASH_FILE = "sentinel_integrity.sha256"

# =========================
# AUTOLOAD LIBS
# =========================

REQUIRED_LIBS = ["pywin32", "psutil", "torch", "flask", "requests"]

def autoload_libraries():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except Exception:
                pass

autoload_libraries()

# Named pipe
try:
    import win32pipe
    import win32file
    import pywintypes
    PIPE_AVAILABLE = True
except ImportError:
    PIPE_AVAILABLE = False

# System introspection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional GPU/NPU via PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Web cockpit (Flask)
try:
    from flask import Flask, jsonify, render_template_string, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# HTTP for swarm gossip
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

import winreg  # stdlib, Windows only

# =========================
# AUTO-ELEVATION CHECK
# =========================

def _log_elevation(msg):
    ts = datetime.utcnow().isoformat() + "Z"
    line = f"{ts} :: ELEVATION :: {msg}"
    print(line)
    try:
        with open("sentinel_elevation.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def ensure_admin():
    try:
        is_admin = False
        try:
            is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            # Fallback: try opening a privileged handle
            try:
                import win32security
                import win32con
                token = win32security.OpenProcessToken(
                    win32api.GetCurrentProcess(),
                    win32con.TOKEN_QUERY
                )
                is_admin = True if token else False
            except Exception:
                is_admin = False

        if is_admin:
            _log_elevation("Already running as admin.")
            return

        # Determine script/executable path
        if getattr(sys, "frozen", False) and FORCE_ELEVATE_FROZEN:
            script = sys.executable
        else:
            script = os.path.abspath(sys.argv[0])

        params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
        _log_elevation(f"Attempting elevation for: {script} {params}")
        rc = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, f'"{script}" {params}', None, 1
        )
        if rc <= 32:
            _log_elevation(f"ShellExecuteW failed with code {rc}")
            if not SILENT_ELEVATION:
                print("[Codex Sentinel] Elevation failed.")
            sys.exit(1)
        else:
            _log_elevation("Elevation request sent, exiting parent.")
            sys.exit(0)
    except Exception as e:
        _log_elevation(f"Elevation exception: {e}")
        if not SILENT_ELEVATION:
            print(f"[Codex Sentinel] Elevation failed: {e}")
        sys.exit(1)

ensure_admin()

# =========================
# CONFIG / FILE PATHS
# =========================

STATE_FILE       = "sentinel_state.json"
TIMELINE_FILE    = "sentinel_timeline.json"
THREATS_FILE     = "sentinel_threats.json"
NODES_FILE       = "sentinel_nodes.json"
LOGS_FILE        = "sentinel_logs.json"
SWARM_FILE       = "sentinel_swarm.json"
PERSONA_FILE     = "sentinel_persona.json"
NARRATIVE_FILE   = "sentinel_narrative.json"
MODEL_FILE       = "sentinel_anomaly_model.pt"
TELEMETRY_FILE   = "sentinel_telemetry.jsonl"
RL_PERSONA_FILE  = "sentinel_persona_rl.json"
SWARM_CONFIG     = "sentinel_swarm_peers.json"  # list of peer URLs + ids
EVENT_LAKE_FILE  = "sentinel_telemetry_lake.jsonl"

PIPE_NAME = r"\\.\pipe\sentinel_bus"

LOCAL_NODE_ID   = os.environ.get("SENTINEL_NODE_ID", "LocalNode")
LOCAL_HTTP_PORT = int(os.environ.get("SENTINEL_HTTP_PORT", "5000"))
SWARM_SHARED_KEY = os.environ.get("SENTINEL_SWARM_KEY", "changeme_shared_key")

# =========================
# BASIC PERSISTENCE
# =========================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"lockdown": False, "autonomous": True, "last_changed": None}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"lockdown": False, "autonomous": True, "last_changed": None}

def save_state(state):
    state["last_changed"] = datetime.utcnow().isoformat() + "Z"
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)

def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def load_json_list(path, default=None):
    if default is None:
        default = []
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else default
    except Exception:
        return default

def load_json_dict(path, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else default
    except Exception:
        return default

# =========================
# SAFE INTROSPECTION
# =========================

def scan_processes():
    results = []
    if not PSUTIL_AVAILABLE:
        return results
    for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "exe", "memory_percent"]):
        info = p.info
        pid = info.get("pid")
        name = info.get("name") or "?"
        cpu = info.get("cpu_percent") or 0.0
        exe = info.get("exe") or ""
        mem = info.get("memory_percent") or 0.0
        suspicious = False
        if cpu > 50.0 and "Windows" not in exe:
            suspicious = True
        if mem > 10.0 and "defender" not in name.lower():
            suspicious = True
        results.append({
            "pid": pid,
            "name": name,
            "cpu": cpu,
            "exe": exe,
            "mem": mem,
            "suspicious": suspicious
        })
    return results

def probe_network():
    results = []
    if not PSUTIL_AVAILABLE:
        return results
    try:
        conns = psutil.net_connections(kind="inet")
    except Exception:
        return results
    for c in conns:
        laddr = f"{getattr(c.laddr, 'ip', '')}:{getattr(c.laddr, 'port', '')}" if c.laddr else ""
        raddr = f"{getattr(c.raddr, 'ip', '')}:{getattr(c.raddr, 'port', '')}" if c.raddr else ""
        results.append({
            "laddr": laddr,
            "raddr": raddr,
            "status": c.status,
            "pid": c.pid
        })
    return results

def check_registry():
    results = []
    keys = [
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
    ]
    for root, subkey in keys:
        try:
            h = winreg.OpenKey(root, subkey)
        except OSError:
            continue
        i = 0
        while True:
            try:
                name, data, _ = winreg.EnumValue(h, i)
                results.append({
                    "key": subkey,
                    "value_name": name,
                    "data": str(data)
                })
                i += 1
            except OSError:
                break
        winreg.CloseKey(h)
    return results

# =========================
# GROUP POLICY / DIAGNOSTIC SUPPRESSION (SAFE HARDENING)
# =========================

def harden_diagnostics():
    try:
        # Telemetry level = 0 (Security)
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key = winreg.CreateKey(root, r"SOFTWARE\Policies\Microsoft\Windows\DataCollection")
                winreg.SetValueEx(key, "AllowTelemetry", 0, winreg.REG_DWORD, 0)
                winreg.CloseKey(key)
            except Exception:
                pass

        # Disable Feedback Hub / error reporting / some scheduled tasks (best-effort, non-fatal)
        try:
            key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"SOFTWARE\Policies\Microsoft\Windows\Windows Error Reporting")
            winreg.SetValueEx(key, "Disabled", 0, winreg.REG_DWORD, 1)
            winreg.CloseKey(key)
        except Exception:
            pass

        # Scheduled tasks that phone home (best-effort)
        try:
            tasks = [
                r"\Microsoft\Windows\Customer Experience Improvement Program\Consolidator",
                r"\Microsoft\Windows\Customer Experience Improvement Program\KernelCeipTask",
                r"\Microsoft\Windows\Application Experience\ProgramDataUpdater",
            ]
            for t in tasks:
                try:
                    subprocess.run(["schtasks", "/Change", "/TN", t, "/Disable"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

harden_diagnostics()

# =========================
# ETW INGESTION (REAL-READY / STUB)
# =========================

class ETWIngestor:
    """
    Pluggable ETW ingestion interface.
    In a real deployment, wire krabsetw/etwpy here and push events into self.buffer.
    """
    def __init__(self):
        self.enabled = False
        self.buffer = deque(maxlen=4096)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def ingest_event(self, evt):
        if not self.enabled:
            return
        self.buffer.append(evt)

    def poll(self):
        if not self.enabled:
            return []
        evts = list(self.buffer)
        self.buffer.clear()
        return evts

etw_ingestor = ETWIngestor()

def listen_etw():
    # In real use, ETW callbacks feed etw_ingestor.ingest_event(...)
    # Hook krabsetw / syscall tracing / context switch deltas here.
    return etw_ingestor.poll()

def kernel_sensors_stub():
    return {
        "context_switches": 0,
        "syscalls": 0,
        "driver_events": 0
    }

# =========================
# TELEMETRY RECORDER (FOR REAL TRAINING)
# =========================

class TelemetryRecorder:
    """
    Records feature vectors + labels to JSONL for offline training.
    Label is currently anomaly_score (self-supervised).
    """
    def __init__(self, path=TELEMETRY_FILE, max_lines=100000):
        self.path = path
        self.max_lines = max_lines
        self.count = 0

    def record(self, features, anomaly_score):
        try:
            line = json.dumps({"features": features, "score": anomaly_score})
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self.count += 1
            if self.count > self.max_lines:
                self._truncate()
        except Exception:
            pass

    def _truncate(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-self.max_lines//2:]
            with open(self.path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            self.count = len(lines)
        except Exception:
            pass

telemetry_recorder = TelemetryRecorder()

# =========================
# DEEP LEARNING ANOMALY MODEL
# =========================

class DeepAnomalyModel(torch.nn.Module):
    def __init__(self, input_dim=7, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2)  # mean, log_var
        )

    def forward(self, x):
        return self.net(x)

class AnomalyModelWrapper:
    def __init__(self):
        self.model = None
        self.device = "cpu"

    def _make_synthetic_baseline(self, n=600):
        X = []
        y = []
        for _ in range(n):
            sp = torch.randint(0, 4, (1,)).item()
            outb = torch.randint(0, 40, (1,)).item()
            run = torch.randint(5, 80, (1,)).item()
            etw_c = torch.randint(0, 5, (1,)).item()
            drv = torch.randint(0, 3, (1,)).item()
            mem_hot = torch.randint(0, 4, (1,)).item()
            proc_total = torch.randint(40, 200, (1,)).item()

            base = sp * 12 + outb * 1.8 + run * 0.4 + etw_c * 4 + drv * 9 + mem_hot * 6
            base += max(0, proc_total - 120) * 0.3

            if sp > 2 or drv > 1 or outb > 30:
                base += 25

            X.append([sp, outb, run, etw_c, drv, mem_hot, proc_total])
            y.append(base)
        return X, y

    def _load_real_telemetry(self):
        """
        Optional: load real telemetry from TELEMETRY_FILE.
        If present, it overrides synthetic baseline.
        """
        if not os.path.exists(TELEMETRY_FILE):
            return None, None
        X, y = [], []
        try:
            with open(TELEMETRY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    feats = obj.get("features")
                    score = obj.get("score")
                    if isinstance(feats, list) and isinstance(score, (int, float)):
                        if len(feats) == 7:
                            X.append(feats)
                            y.append(score)
        except Exception:
            return None, None
        if not X:
            return None, None
        return X, y

    def train_baseline(self, epochs=120):
        if not TORCH_AVAILABLE:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        X_real, y_real = self._load_real_telemetry()
        if X_real is not None:
            X, y = X_real, y_real
        else:
            X, y = self._make_synthetic_baseline()

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.model = DeepAnomalyModel(input_dim=7, hidden=64).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.003)
        loss_fn = torch.nn.MSELoss()

        for _ in range(epochs):
            opt.zero_grad()
            out = self.model(X)
            mean = out[:, 0:1]
            loss = loss_fn(mean, y)
            loss.backward()
            opt.step()

        try:
            torch.save(self.model.state_dict(), MODEL_FILE)
        except Exception:
            pass

    def load_or_init(self):
        if not TORCH_AVAILABLE:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DeepAnomalyModel(input_dim=7, hidden=64).to(self.device)
        if os.path.exists(MODEL_FILE):
            try:
                self.model.load_state_dict(torch.load(MODEL_FILE, map_location=self.device))
            except Exception:
                self.train_baseline()
        else:
            self.train_baseline()

    def score(self, features, lockdown, autonomous):
        if not TORCH_AVAILABLE or self.model is None:
            return None, None, None
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        mean = out[0, 0]
        log_var = out[0, 1]
        var = torch.exp(log_var)
        std = torch.sqrt(var + 1e-6)

        score = mean
        if lockdown:
            score = score * 0.7
        if not autonomous:
            score = score * 0.9

        score_val = float(score.item())
        std_val = float(std.item())
        risk = max(0.0, min(1.0, score_val / 150.0))

        return int(score_val), std_val, risk

anomaly_model = AnomalyModelWrapper()
if TORCH_AVAILABLE:
    anomaly_model.load_or_init()

def compute_anomaly_features(proc_list, net_list, reg_list, etw_events, kernel_data):
    suspicious_procs = sum(1 for p in proc_list if p.get("suspicious"))
    outbound = sum(1 for n in net_list if n.get("raddr"))
    run_entries = len(reg_list)
    etw_count = len(etw_events)
    drv_events = kernel_data.get("driver_events", 0)
    mem_hot = sum(1 for p in proc_list if p.get("mem", 0) > 10.0)
    proc_total = len(proc_list)
    return [suspicious_procs, outbound, run_entries, etw_count, drv_events, mem_hot, proc_total]

def compute_anomaly_score_cpu(features, lockdown, autonomous):
    sp, outb, run, etw_c, drv, mem_hot, proc_total = features
    score = 0
    score += sp * 15
    score += min(outb, 60) * 2
    score += min(run, 100) * 0.8
    score += etw_c * 5
    score += drv * 10
    score += mem_hot * 8
    score += max(0, proc_total - 120) * 0.4
    score += int(math.sqrt(outb) * 5)
    if lockdown:
        score *= 0.7
    if not autonomous:
        score *= 0.9
    return int(score)

def compute_anomaly_score_gpu(features, lockdown, autonomous):
    if not TORCH_AVAILABLE:
        return compute_anomaly_score_cpu(features, lockdown, autonomous), None, None
    s, std, risk = anomaly_model.score(features, lockdown, autonomous)
    if s is not None:
        return s, std, risk
    return compute_anomaly_score_cpu(features, lockdown, autonomous), None, None

def compute_anomaly_score(proc_list, net_list, reg_list, etw_events, kernel_data, lockdown, autonomous):
    features = compute_anomaly_features(proc_list, net_list, reg_list, etw_events, kernel_data)
    score, std, risk = compute_anomaly_score_gpu(features, lockdown, autonomous)
    telemetry_recorder.record(features, score)
    return score, std, risk, features

# =========================
# ATTACK CHAIN ENGINE + EVENT BUS + SEC EVENT
# =========================

class SecEvent:
    def __init__(self, etype, entity, meta=None):
        self.ts = time.time()
        self.type = etype
        self.entity = entity
        self.meta = meta or {}

class AttackChainEngine:
    def __init__(self):
        self.events = deque()
        self.window = 120

    def ingest(self, evt: SecEvent):
        self.events.append(evt)
        self._cleanup()

    def _cleanup(self):
        now = time.time()
        while self.events and now - self.events[0].ts > self.window:
            self.events.popleft()

    def detect_chains(self):
        seq = [e.type for e in self.events]
        chains = []
        if "proc_start" in seq and "net_conn" in seq and "file_mod" in seq:
            chains.append(("FULL_ATTACK_CHAIN", 0.95))
        if seq.count("proc_start") > 10:
            chains.append(("SPAWN_STORM", 0.8))
        if "net_conn" in seq and any("powershell" in (e.meta.get("cmd", "") or "").lower() for e in self.events):
            chains.append(("LOLBIN_BEACON", 0.9))
        return chains

class AttackChainEngineAlt:
    def __init__(self):
        self.events = deque()
        self.window = 120

    def add_event(self, event_type, data):
        now = time.time()
        self.events.append((now, event_type, data))
        self._cleanup(now)

    def _cleanup(self, now):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def detect(self):
        types = [e[1] for e in self.events]
        chains = []
        if all(x in types for x in ["proc_spawn", "powershell", "net_connect"]):
            chains.append(("LOLBIN_ATTACK", 0.9))
        if types.count("proc_spawn") > 5 and "net_connect" in types:
            chains.append(("PROCESS_STORM", 0.8))
        if "file_mod" in types and "net_connect" in types:
            chains.append(("PERSISTENCE_EXFIL", 0.85))
        return chains

class EventBus:
    def __init__(self):
        self.subscribers = []
        self.queue = deque()
        self.running = True

    def publish(self, event: SecEvent):
        self.queue.append(event)

    def subscribe(self, fn):
        self.subscribers.append(fn)

    def run(self):
        while self.running:
            if self.queue:
                evt = self.queue.popleft()
                for fn in self.subscribers:
                    try:
                        fn(evt)
                    except Exception:
                        pass
            time.sleep(0.01)

# =========================
# ALTERED STATES ENGINE
# =========================

class AlteredStatesEngine:
    def __init__(self, window=60):
        self.window = window
        self.history = deque()  # (ts, anomaly_score, proc_count, net_count, reg_count, risk)

    def ingest_snapshot(self, anomaly_score, proc_count, net_count, reg_count, risk):
        now = time.time()
        self.history.append((now, anomaly_score, proc_count, net_count, reg_count, risk))
        self._cleanup(now)

    def _cleanup(self, now):
        cutoff = now - self.window
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def analyze_state(self):
        if len(self.history) < 3:
            return {"state": "baseline", "confidence": 0.3, "missing_signals": []}
        scores = [h[1] for h in self.history]
        procs  = [h[2] for h in self.history]
        nets   = [h[3] for h in self.history]
        regs   = [h[4] for h in self.history]
        risks  = [h[5] for h in self.history]
        delta_score = scores[-1] - scores[0]
        avg_risk = sum(risks) / len(risks)

        state = "baseline"
        confidence = 0.5
        missing = []

        if delta_score > 40 or avg_risk > 0.6:
            state = "escalating"
            confidence = 0.85
        elif delta_score < -30 and avg_risk < 0.3:
            state = "cooling"
            confidence = 0.75

        if nets[-1] == 0 and scores[-1] > 60:
            missing.append("network_signals")
        if procs[-1] < 3 and scores[-1] > 50:
            missing.append("process_signals")

        if missing and state == "baseline":
            state = "hidden_activity"
            confidence = 0.8

        return {"state": state, "confidence": confidence, "missing_signals": missing}

# =========================
# ENCRYPTED SWARM + NEGOTIATION
# =========================

def _swarm_sign(payload: dict) -> str:
    msg = json.dumps(payload, sort_keys=True).encode()
    key = SWARM_SHARED_KEY.encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

def _swarm_verify(payload: dict, sig: str) -> bool:
    expected = _swarm_sign(payload)
    return hmac.compare_digest(expected, sig)

class SwarmManager:
    """
    Multi-node encrypted swarm via HTTPS + shared key:
      - per-node trust
      - score decay
      - jittered polling
      - weighted consensus
      - behavioral negotiation (lockdown/autonomous proposals)
    """
    def __init__(self):
        self.remote_nodes = {}  # node_id -> {"score": int, "ts": float, "trust": float, "lockdown": bool, "autonomous": bool}
        self.lock = threading.Lock()
        self.last_gossip = 0.0
        self.min_interval = 1.0
        self.max_interval = 3.0

    def load_peers(self):
        peers = load_json_list(SWARM_CONFIG, [])
        return peers

    def update_local(self, score, lockdown, autonomous):
        now = time.time()
        with self.lock:
            self.remote_nodes[LOCAL_NODE_ID] = {
                "score": score,
                "ts": now,
                "trust": 1.0,
                "lockdown": lockdown,
                "autonomous": autonomous
            }

    def _decay_scores(self):
        now = time.time()
        with self.lock:
            for nid, info in list(self.remote_nodes.items()):
                age = now - info["ts"]
                if age > 60:
                    info["score"] = int(info["score"] * 0.5)
                    info["trust"] = max(0.1, info["trust"] * 0.8)

    def gossip(self):
        if not REQUESTS_AVAILABLE:
            return
        now = time.time()
        interval = random.uniform(self.min_interval, self.max_interval)
        if now - self.last_gossip < interval:
            return
        self.last_gossip = now

        peers = self.load_peers()
        for p in peers:
            url = p.get("url")
            nid = p.get("id")
            if not url or not nid:
                continue
            try:
                r = requests.get(url + "/api/swarm/state", timeout=0.7, verify=False)
                if r.status_code == 200:
                    data = r.json()
                    payload = data.get("payload", {})
                    sig = data.get("sig", "")
                    if not _swarm_verify(payload, sig):
                        continue
                    score = int(payload.get("score", 0))
                    lockdown = bool(payload.get("lockdown", False))
                    autonomous = bool(payload.get("autonomous", True))
                    with self.lock:
                        info = self.remote_nodes.get(nid, {"score": 0, "ts": 0.0, "trust": 0.5,
                                                           "lockdown": False, "autonomous": True})
                        info["score"] = score
                        info["ts"] = time.time()
                        info["trust"] = min(1.0, info["trust"] + 0.05)
                        info["lockdown"] = lockdown
                        info["autonomous"] = autonomous
                        self.remote_nodes[nid] = info
            except Exception:
                with self.lock:
                    info = self.remote_nodes.get(nid, {"score": 0, "ts": 0.0, "trust": 0.5,
                                                       "lockdown": False, "autonomous": True})
                    info["trust"] = max(0.1, info["trust"] * 0.9)
                    self.remote_nodes[nid] = info

        self._decay_scores()

    def get_nodes_scores(self):
        self.gossip()
        with self.lock:
            out = []
            for nid, info in self.remote_nodes.items():
                out.append({
                    "node": nid,
                    "score": info["score"],
                    "trust": round(info["trust"], 2),
                    "lockdown": info.get("lockdown", False),
                    "autonomous": info.get("autonomous", True)
                })
            return out

swarm_manager = SwarmManager()

def swarm_consensus(nodes_scores):
    if not nodes_scores:
        return {"consensus_score": 0, "max_score": 0, "trust_weighted": 0,
                "lockdown_vote": False, "autonomous_vote": True}
    scores = [n["score"] for n in nodes_scores]
    trusts = [n.get("trust", 1.0) for n in nodes_scores]
    mx = max(scores)
    total_w = sum(trusts)
    if total_w <= 0:
        avg = sum(scores) / len(scores)
        tw = avg
    else:
        tw = sum(s * w for s, w in zip(scores, trusts)) / total_w
    avg = sum(scores) / len(scores)

    lockdown_votes = []
    auto_votes = []
    for n in nodes_scores:
        w = n.get("trust", 1.0)
        lockdown_votes.append(w if n.get("lockdown") else -w)
        auto_votes.append(w if n.get("autonomous") else -w)
    lockdown_vote = sum(lockdown_votes) > 0
    autonomous_vote = sum(auto_votes) > 0

    return {
        "consensus_score": int(avg),
        "max_score": int(mx),
        "trust_weighted": int(tw),
        "lockdown_vote": lockdown_vote,
        "autonomous_vote": autonomous_vote
    }

# =========================
# THREAT NARRATIVE
# =========================

def build_threat_narrative(timeline, anomaly_score, chain_findings, altered_state, std_dev, risk):
    if not timeline:
        return {"summary": "No events", "steps": []}
    steps = []
    for ev in timeline[-5:]:
        steps.append(f"{ev.get('time', '?')} :: {ev.get('event', '')}")
    chain_summary = ", ".join([c[0] for c in chain_findings]) if chain_findings else "no chains"
    altered_desc = f"{altered_state.get('state')} (conf={altered_state.get('confidence'):.2f})"
    risk_str = f"risk={risk:.2f}" if risk is not None else "risk=unknown"
    std_str = f"±{std_dev:.1f}" if std_dev is not None else "±?"
    if anomaly_score > 80 or chain_findings:
        summary = f"High-risk sequence; chains: {chain_summary}; state: {altered_desc}; score={anomaly_score}{std_str}, {risk_str}"
    elif anomaly_score > 40:
        summary = f"Elevated activity; chains: {chain_summary}; state: {altered_desc}; score={anomaly_score}{std_str}, {risk_str}"
    else:
        summary = f"Normal background; chains: {chain_summary}; state: {altered_desc}; score={anomaly_score}{std_str}, {risk_str}"
    return {"summary": summary, "steps": steps}

# =========================
# ACTIVE DEFENSE (ADVISORY)
# =========================

class ActiveDefenseLayer:
    def __init__(self):
        self.recommendations = deque(maxlen=100)

    def evaluate(self, anomaly_score, chain_findings, risk):
        recs = []
        if anomaly_score > 80 or (risk is not None and risk > 0.7):
            recs.append("Consider isolating host from network (manual action).")
        for cname, score in chain_findings:
            if score > 0.85:
                recs.append(f"Investigate chain: {cname}; consider suspending related processes.")
        if not recs:
            recs.append("No active defense actions recommended.")
        ts = datetime.utcnow().isoformat() + "Z"
        for r in recs:
            self.recommendations.append(f"{ts} :: {r}")
        return recs

active_defense = ActiveDefenseLayer()

# =========================
# RL PERSONA ENGINE
# =========================

class PersonaRL:
    """
    Simple Q-style persona mode learner.
    States: (risk_band, altered_state)
    Actions: persona modes: Observer, Watcher, Guardian, Hunter
    Reward: derived from anomaly_score + chains (offline heuristic)
    """
    MODES = ["Observer", "Watcher", "Guardian", "Hunter"]

    def __init__(self, path=RL_PERSONA_FILE):
        self.path = path
        self.q = self._load_q()

    def _load_q(self):
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_q(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.q, f, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def _state_key(self, risk, altered_state):
        band = "low"
        if risk is not None:
            if risk > 0.7:
                band = "high"
            elif risk > 0.4:
                band = "mid"
        return f"{band}:{altered_state.get('state', 'baseline')}"

    def choose_mode(self, risk, altered_state):
        s = self._state_key(risk, altered_state)
        if s not in self.q:
            self.q[s] = {m: 0.0 for m in self.MODES}
        if random.random() < 0.1:
            return random.choice(self.MODES)
        return max(self.q[s], key=self.q[s].get)

    def update(self, risk, altered_state, anomaly_score, chain_findings, chosen_mode):
        s = self._state_key(risk, altered_state)
        if s not in self.q:
            self.q[s] = {m: 0.0 for m in self.MODES}
        base = anomaly_score / 100.0
        if chain_findings:
            base += 0.5
        if chosen_mode in ["Hunter", "Guardian"]:
            reward = base
        elif chosen_mode == "Watcher":
            reward = base * 0.7
        else:
            reward = base * 0.3
        old = self.q[s].get(chosen_mode, 0.0)
        self.q[s][chosen_mode] = old + 0.1 * (reward - old)
        self._save_q()

persona_rl = PersonaRL()

# =========================
# MINI-SENTINEL HOOKS
# =========================

def hook_collect_timeline(lockdown, autonomous,
                          proc_list, net_list, reg_list, anomaly_score, std_dev, risk):
    now = datetime.utcnow().isoformat() + "Z"
    events = [
        {"time": now, "event": "Heartbeat", "severity": "info"},
        {"time": now, "event": f"Processes scanned: {len(proc_list)}", "severity": "info"},
        {"time": now, "event": f"Network conns: {len(net_list)}", "severity": "info"},
        {"time": now, "event": f"Run entries: {len(reg_list)}", "severity": "info"},
        {"time": now, "event": f"Anomaly score: {anomaly_score} (std={std_dev if std_dev is not None else '?'}, risk={risk if risk is not None else '?'})",
         "severity": "warn" if anomaly_score > 50 else "info"},
    ]
    if lockdown:
        events.append({"time": now, "event": "Lockdown active", "severity": "warn"})
    if not autonomous:
        events.append({"time": now, "event": "Manual override", "severity": "info"})
    return events

def hook_collect_threat_matrix(lockdown, autonomous, anomaly_score):
    swarm_manager.update_local(anomaly_score, lockdown, autonomous)
    nodes_scores = swarm_manager.get_nodes_scores()
    consensus = swarm_consensus(nodes_scores)
    nodes = []
    for n in nodes_scores:
        nodes.append({"node": n["node"], "score": n["score"]})
    nodes.append({"node": "Consensus(avg)", "score": consensus["consensus_score"]})
    nodes.append({"node": "Consensus(weighted)", "score": consensus["trust_weighted"]})
    return nodes

def hook_collect_node_sync(lockdown, autonomous):
    status = "in-sync"
    if not autonomous:
        status = "manual"
    if lockdown:
        status = "lockdown"
    return [{"node": LOCAL_NODE_ID, "status": status}]

def hook_collect_logs(lockdown, autonomous, anomaly_score, std_dev, risk,
                      chain_findings, altered_state, defense_recs, consensus):
    now = datetime.utcnow().isoformat() + "Z"
    lines = [f"{now} :: heartbeat :: anomaly_score={anomaly_score} std={std_dev} risk={risk} :: state={altered_state.get('state')} :: consensus={consensus}"]
    if anomaly_score > 80:
        lines.append(f"{now} :: ALERT :: high anomaly")
    if risk is not None and risk > 0.7:
        lines.append(f"{now} :: ALERT :: high risk band")
    if chain_findings:
        for cname, score in chain_findings:
            lines.append(f"{now} :: CHAIN :: {cname} :: score={score}")
    for r in defense_recs:
        lines.append(f"{now} :: DEFENSE :: {r}")
    if lockdown:
        lines.append(f"{now} :: lockdown mode: aggressive filters enabled")
    if not autonomous:
        lines.append(f"{now} :: manual override: operator in control")
    return lines

def hook_collect_swarm(lockdown, autonomous, anomaly_score):
    nodes_scores = swarm_manager.get_nodes_scores()
    consensus = swarm_consensus(nodes_scores)
    return {
        "nodes": len(nodes_scores),
        "cohesion": 0.9 if autonomous else 0.7,
        "drift": min(anomaly_score / 100.0, 1.0),
        "consensus": consensus
    }

def hook_collect_persona(lockdown, autonomous, anomaly_score, altered_state, risk):
    mode = persona_rl.choose_mode(risk, altered_state)
    return {"mode": mode, "autonomous": autonomous}

def hook_emit_events(lockdown, autonomous, anomaly_score, chain_findings, altered_state, risk):
    now = datetime.utcnow().isoformat() + "Z"
    events = [f"{now} :: EVENT :: heartbeat :: anomaly_score={anomaly_score} :: state={altered_state.get('state')} :: risk={risk}"]
    if anomaly_score > 80 or (risk is not None and risk > 0.7):
        events.append(f"{now} :: EVENT :: high_anomaly_or_risk")
    for cname, score in chain_findings:
        if score > 0.8:
            events.append(f"{now} :: EVENT :: attack_chain::{cname}::{score}")
    if lockdown:
        events.append(f"{now} :: EVENT :: lockdown_enforced")
    if not autonomous:
        events.append(f"{now} :: EVENT :: manual_override")
    return events

def hook_behavior_on_lockdown(lockdown, autonomous):
    return

# =========================
# THREAT MATRIX GLYPH SYSTEM (ABSTRACT)
# =========================

def glyphs_for_state(anomaly_score, altered_state, chain_findings, persona_mode, lockdown):
    glyphs = []
    if anomaly_score > 90 and altered_state.get("state") == "escalating":
        glyphs.append("RESURRECTION")
    if altered_state.get("state") in ("escalating", "hidden_activity"):
        glyphs.append("DRIFT")
    if chain_findings:
        glyphs.append("CHAIN")
    if persona_mode in ("Guardian", "Hunter"):
        glyphs.append("PERSONA")
    if lockdown:
        glyphs.append("LOCKDOWN")
    return glyphs

# =========================
# ORGANISM SIDE
# =========================

class EventBusServer(threading.Thread):
    def __init__(self, pipe_name=PIPE_NAME):
        super().__init__(daemon=True)
        self.pipe_name = pipe_name
        self.running = True

    def run(self):
        if not PIPE_AVAILABLE:
            return
        while self.running:
            try:
                handle = win32pipe.CreateNamedPipe(
                    self.pipe_name,
                    win32pipe.PIPE_ACCESS_OUTBOUND,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                    1, 65536, 65536, 0, None
                )
                win32pipe.ConnectNamedPipe(handle, None)
                while self.running:
                    state = load_state()
                    autonomous = state.get("autonomous", True)
                    time.sleep(1 if autonomous else 2)
                win32file.CloseHandle(handle)
            except pywintypes.error:
                time.sleep(1)

class OrganismLoop(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.chain_engine = AttackChainEngine()
        self.chain_engine_alt = AttackChainEngineAlt()
        self.internal_bus = EventBus()
        self.altered = AlteredStatesEngine()
        self.bus_thread = threading.Thread(target=self.internal_bus.run, daemon=True)
        self.bus_thread.start()
        self.internal_bus.subscribe(self._chain_ingest_callback)

    def _chain_ingest_callback(self, evt: SecEvent):
        self.chain_engine.ingest(evt)
        mapped_type = evt.type
        if evt.type == "proc_start":
            mapped_type = "proc_spawn"
        elif evt.type == "net_conn":
            mapped_type = "net_connect"
        self.chain_engine_alt.add_event(mapped_type, evt.meta)

    def run(self):
        while self.running:
            state = load_state()
            lockdown   = state.get("lockdown", False)
            autonomous = state.get("autonomous", True)

            hook_behavior_on_lockdown(lockdown, autonomous)

            proc_list = scan_processes()
            net_list  = probe_network()
            reg_list  = check_registry()
            etw_events = listen_etw()
            kernel_data = kernel_sensors_stub()
            anomaly_score, std_dev, risk, features = compute_anomaly_score(
                proc_list, net_list, reg_list,
                etw_events, kernel_data,
                lockdown, autonomous
            )

            self.altered.ingest_snapshot(anomaly_score, len(proc_list), len(net_list), len(reg_list), risk if risk is not None else 0.0)
            altered_state = self.altered.analyze_state()

            for p in proc_list[:20]:
                evt = SecEvent("proc_start", p["pid"], {"name": p["name"], "exe": p["exe"]})
                self.internal_bus.publish(evt)
            for n in net_list[:20]:
                evt = SecEvent("net_conn", n["raddr"], {"laddr": n["laddr"], "status": n["status"]})
                self.internal_bus.publish(evt)

            chains1 = self.chain_engine.detect_chains()
            chains2 = self.chain_engine_alt.detect()
            chain_findings = list({c[0]: c for c in chains1 + chains2}.values())

            defense_recs = active_defense.evaluate(anomaly_score, chain_findings, risk)

            persona_mode = persona_rl.choose_mode(risk, altered_state)
            persona_rl.update(risk, altered_state, anomaly_score, chain_findings, persona_mode)

            swarm_manager.update_local(anomaly_score, lockdown, autonomous)
            nodes_scores = swarm_manager.get_nodes_scores()
            consensus = swarm_consensus(nodes_scores)

            timeline = hook_collect_timeline(lockdown, autonomous,
                                             proc_list, net_list, reg_list,
                                             anomaly_score, std_dev, risk)
            threats  = hook_collect_threat_matrix(lockdown, autonomous, anomaly_score)
            nodes    = hook_collect_node_sync(lockdown, autonomous)
            logs     = hook_collect_logs(lockdown, autonomous, anomaly_score, std_dev, risk,
                                         chain_findings, altered_state, defense_recs, consensus)
            swarm    = hook_collect_swarm(lockdown, autonomous, anomaly_score)
            persona  = {"mode": persona_mode, "autonomous": autonomous}
            narrative= build_threat_narrative(timeline, anomaly_score, chain_findings, altered_state, std_dev, risk)

            save_json(TIMELINE_FILE, timeline)
            save_json(THREATS_FILE, threats)
            save_json(NODES_FILE,   nodes)
            save_json(LOGS_FILE,    logs)
            save_json(SWARM_FILE,   swarm)
            save_json(PERSONA_FILE, persona)
            save_json(NARRATIVE_FILE, narrative)

            # Cloud-backed telemetry lake (local file stub)
            try:
                with open(EVENT_LAKE_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "anomaly_score": anomaly_score,
                        "risk": risk,
                        "altered_state": altered_state,
                        "chains": chain_findings
                    }) + "\n")
            except Exception:
                pass

            if PIPE_AVAILABLE:
                try:
                    handle = win32file.CreateFile(
                        PIPE_NAME,
                        win32file.GENERIC_WRITE,
                        0,
                        None,
                        win32file.OPEN_EXISTING,
                        0,
                        None
                    )
                    ev_lines = hook_emit_events(lockdown, autonomous, anomaly_score,
                                                chain_findings, altered_state, risk)
                    for ev in ev_lines:
                        try:
                            win32file.WriteFile(handle, (ev + "\n").encode())
                        except pywintypes.error:
                            break
                    win32file.CloseHandle(handle)
                except pywintypes.error:
                    pass

            time.sleep(1 if autonomous else 2)

# =========================
# COCKPIT STATE
# =========================

class SentinelState:
    def __init__(self):
        self.state = load_state()
        self.timeline = []
        self.threat_matrix = []
        self.node_sync = []
        self.logs = []
        self.swarm_sim = {}
        self.persona = {}
        self.narrative = {}
        self.event_bus = []

    def refresh_from_files(self):
        self.state        = load_state()
        self.timeline     = load_json_list(TIMELINE_FILE, [])
        self.threat_matrix= load_json_list(THREATS_FILE, [])
        self.node_sync    = load_json_list(NODES_FILE, [])
        self.logs         = load_json_list(LOGS_FILE, [])
        self.swarm_sim    = load_json_dict(SWARM_FILE, {"nodes": 0})
        self.persona      = load_json_dict(PERSONA_FILE, {"mode": "Unknown", "autonomous": True})
        self.narrative    = load_json_dict(NARRATIVE_FILE, {"summary": "No data", "steps": []})

# =========================
# EVENT BUS LISTENER (GUI SIDE)
# =========================

class EventBusListener(threading.Thread):
    def __init__(self, sentinel_state: SentinelState, pipe_name=PIPE_NAME):
        super().__init__(daemon=True)
        self.sentinel_state = sentinel_state
        self.pipe_name = pipe_name
        self.running = True

    def run(self):
        if not PIPE_AVAILABLE:
            return
        while self.running:
            try:
                handle = win32file.CreateFile(
                    self.pipe_name,
                    win32file.GENERIC_READ,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None
                )
                while self.running:
                    try:
                        hr, data = win32file.ReadFile(handle, 4096)
                        if hr == 0 and data:
                            line = data.decode(errors="ignore").strip()
                            if line:
                                self.sentinel_state.event_bus.append(line)
                                self.sentinel_state.event_bus = self.sentinel_state.event_bus[-500:]
                    except pywintypes.error:
                        break
                win32file.CloseHandle(handle)
            except pywintypes.error:
                time.sleep(1)

# =========================
# TKINTER OPERATOR COCKPIT (ANIMATED)
# =========================

class OperatorCockpit:
    REFRESH_MS = 1000
    OVERLAY_MS = 80

    def __init__(self, root):
        self.root = root
        self.root.title(f"Sentinel Operator Cockpit - {LOCAL_NODE_ID}")

        self.sentinel = SentinelState()

        self.event_listener = EventBusListener(self.sentinel)
        self.event_listener.start()

        self._build_header()
        self._build_tabs()

        self._overlay_phase = 0.0
        self._swarm_phase = 0.0

        self._refresh_all()
        self._animate_overlays()

    def _build_header(self):
        frame = tk.Frame(self.root)
        frame.pack(fill="x", pady=5)

        self.lockdown_label = tk.Label(
            frame, text=self._lockdown_text(),
            font=("Consolas", 12, "bold"),
            fg=self._lockdown_color()
        )
        self.lockdown_label.pack(side="left", padx=10)

        self.autonomous_label = tk.Label(
            frame, text=self._autonomous_text(),
            font=("Consolas", 10),
            fg=self._autonomous_color()
        )
        self.autonomous_label.pack(side="left", padx=10)

        self.lockdown_button = tk.Button(
            frame, text=self._lockdown_button_text(),
            command=self.toggle_lockdown
        )
        self.lockdown_button.pack(side="right", padx=10)

        self.autonomous_button = tk.Button(
            frame, text=self._autonomous_button_text(),
            command=self.toggle_autonomous
        )
        self.autonomous_button.pack(side="right", padx=10)

    def _lockdown_text(self):
        return "LOCKDOWN: ON" if self.sentinel.state.get("lockdown") else "LOCKDOWN: OFF"

    def _lockdown_color(self):
        return "#FF3333" if self.sentinel.state.get("lockdown") else "#33AA33"

    def _lockdown_button_text(self):
        return "Disable Lockdown" if self.sentinel.state.get("lockdown") else "Enable Lockdown"

    def _autonomous_text(self):
        return "AUTONOMOUS: ON" if self.sentinel.state.get("autonomous") else "AUTONOMOUS: OFF"

    def _autonomous_color(self):
        return "#3399FF" if self.sentinel.state.get("autonomous") else "#AAAAAA"

    def _autonomous_button_text(self):
        return "Manual Override" if self.sentinel.state.get("autonomous") else "Return to Autonomous"

    def toggle_lockdown(self):
        st = load_state()
        st["lockdown"] = not st.get("lockdown", False)
        save_state(st)
        self.sentinel.state = st
        self.lockdown_label.config(text=self._lockdown_text(), fg=self._lockdown_color())
        self.lockdown_button.config(text=self._lockdown_button_text())
        messagebox.showinfo("Lockdown", f"Lockdown is now {'ON' if st['lockdown'] else 'OFF'}.")

    def toggle_autonomous(self):
        st = load_state()
        st["autonomous"] = not st.get("autonomous", True)
        save_state(st)
        self.sentinel.state = st
        self.autonomous_label.config(text=self._autonomous_text(), fg=self._autonomous_color())
        self.autonomous_button.config(text=self._autonomous_button_text())
        messagebox.showinfo("Autonomous", f"Autonomous is now {'ON' if st['autonomous'] else 'OFF'}.")

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.timeline_frame   = tk.Frame(self.notebook)
        self.threat_frame     = tk.Frame(self.notebook)
        self.node_sync_frame  = tk.Frame(self.notebook)
        self.logs_frame       = tk.Frame(self.notebook)
        self.swarm_frame      = tk.Frame(self.notebook)
        self.persona_frame    = tk.Frame(self.notebook)
        self.narrative_frame  = tk.Frame(self.notebook)
        self.event_bus_frame  = tk.Frame(self.notebook)

        self.notebook.add(self.timeline_frame,  text="Timeline")
        self.notebook.add(self.threat_frame,    text="Threat Matrix")
        self.notebook.add(self.node_sync_frame, text="Node Sync")
        self.notebook.add(self.logs_frame,      text="Logs")
        self.notebook.add(self.swarm_frame,     text="Swarm")
        self.notebook.add(self.persona_frame,   text="Persona")
        self.notebook.add(self.narrative_frame, text="Narrative")
        self.notebook.add(self.event_bus_frame, text="Event Bus")

        # Timeline heatmap + list
        self.timeline_canvas = tk.Canvas(self.timeline_frame, height=60, bg="#111")
        self.timeline_canvas.pack(fill="x", side="top")
        self.timeline_list = tk.Listbox(self.timeline_frame, font=("Consolas", 9))
        self.timeline_list.pack(fill="both", expand=True)

        self.threat_list = tk.Listbox(self.threat_frame, font=("Consolas", 9))
        self.threat_list.pack(fill="both", expand=True)

        self.node_sync_list = tk.Listbox(self.node_sync_frame, font=("Consolas", 9))
        self.node_sync_list.pack(fill="both", expand=True)

        self.logs_text = tk.Text(self.logs_frame, font=("Consolas", 9))
        self.logs_text.pack(fill="both", expand=True)

        # Swarm node orbit visualization
        self.swarm_canvas = tk.Canvas(self.swarm_frame, bg="black")
        self.swarm_canvas.pack(fill="both", expand=True)

        # Persona morphing indicator
        self.persona_label = tk.Label(self.persona_frame, font=("Consolas", 11))
        self.persona_label.pack(pady=10)
        self.persona_canvas = tk.Canvas(self.persona_frame, height=80, bg="#111")
        self.persona_canvas.pack(fill="x")

        self.narrative_text = tk.Text(self.narrative_frame, font=("Consolas", 9))
        self.narrative_text.pack(fill="both", expand=True)

        self.event_bus_list = tk.Listbox(self.event_bus_frame, font=("Consolas", 9))
        self.event_bus_list.pack(fill="both", expand=True)

    def _refresh_all(self):
        self.sentinel.refresh_from_files()
        self._update_timeline()
        self._update_threat_matrix()
        self._update_node_sync()
        self._update_logs()
        self._update_swarm()
        self._update_persona()
        self._update_narrative()
        self._update_event_bus()
        self.root.after(self.REFRESH_MS, self._refresh_all)

    def _update_timeline(self):
        self.timeline_list.delete(0, tk.END)
        for ev in self.sentinel.timeline:
            t = ev.get("time", "?")
            s = ev.get("severity", "info").upper()
            msg = ev.get("event", "")
            self.timeline_list.insert(tk.END, f"{t} :: {s} :: {msg}")
        # Heatmap
        self.timeline_canvas.delete("all")
        w = self.timeline_canvas.winfo_width()
        h = self.timeline_canvas.winfo_height()
        n = len(self.sentinel.timeline)
        if n == 0:
            return
        max_idx = max(1, n)
        for i, ev in enumerate(self.sentinel.timeline[-50:]):
            sev = ev.get("severity", "info")
            if sev == "warn":
                color = "#FF9933"
            elif sev == "error":
                color = "#FF3333"
            else:
                color = "#33AAFF"
            x0 = i * (w / 50.0)
            x1 = (i + 1) * (w / 50.0)
            self.timeline_canvas.create_rectangle(x0, 0, x1, h, fill=color, outline="")

    def _update_threat_matrix(self):
        self.threat_list.delete(0, tk.END)
        for row in self.sentinel.threat_matrix:
            node = row.get("node", "?")
            score = row.get("score", 0)
            self.threat_list.insert(tk.END, f"{node} :: score={score}")

    def _update_node_sync(self):
        self.node_sync_list.delete(0, tk.END)
        for row in self.sentinel.node_sync:
            node = row.get("node", "?")
            status = row.get("status", "unknown")
            self.node_sync_list.insert(tk.END, f"{node} :: {status}")

    def _update_logs(self):
        self.logs_text.delete("1.0", tk.END)
        for line in self.sentinel.logs:
            self.logs_text.insert(tk.END, line + "\n")

    def _update_swarm(self):
        self.swarm_canvas.delete("all")
        w = self.swarm_canvas.winfo_width()
        h = self.swarm_canvas.winfo_height()
        nodes = self.sentinel.swarm_sim.get("nodes", 0)
        if nodes <= 0:
            return
        drift = self.sentinel.swarm_sim.get("drift", 0.0)
        radius = min(w, h) / 3
        cx, cy = w / 2, h / 2
        for i in range(nodes):
            angle = (2 * math.pi * i / nodes) + self._swarm_phase
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle) + (drift * 40.0)
            self.swarm_canvas.create_oval(
                x - 10, y - 10, x + 10, y + 10,
                fill="#33AAFF", outline="#FFFFFF"
            )
        # central consensus glyph
        self.swarm_canvas.create_oval(
            cx - 14, cy - 14, cx + 14, cy + 14,
            outline="#FFCC33"
        )

    def _update_persona(self):
        mode = self.sentinel.persona.get("mode", "Unknown")
        auto = self.sentinel.persona.get("autonomous", True)
        self.persona_label.config(text=f"Persona Mode: {mode}\nAutonomous: {'ON' if auto else 'OFF'}")
        self.persona_canvas.delete("all")
        w = self.persona_canvas.winfo_width()
        h = self.persona_canvas.winfo_height()
        # persona morphing indicator: color + pulsing radius
        base_r = 15
        pulse = 5 * math.sin(self._overlay_phase)
        if mode == "Hunter":
            color = "#FF3333"
        elif mode == "Guardian":
            color = "#FF9933"
        elif mode == "Watcher":
            color = "#33AAFF"
        else:
            color = "#777777"
        cx, cy = w / 2, h / 2
        r = base_r + pulse
        self.persona_canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=color, outline="#FFFFFF"
        )

    def _update_narrative(self):
        self.narrative_text.delete("1.0", tk.END)
        summary = self.sentinel.narrative.get("summary", "No data")
        steps = self.sentinel.narrative.get("steps", [])
        self.narrative_text.insert(tk.END, "Summary:\n" + summary + "\n\nSteps:\n")
        for s in steps:
            self.narrative_text.insert(tk.END, "- " + s + "\n")

    def _update_event_bus(self):
        self.event_bus_list.delete(0, tk.END)
        for ev in self.sentinel.event_bus:
            self.event_bus_list.insert(tk.END, ev)

    def _animate_overlays(self):
        self._overlay_phase += 0.2
        self._swarm_phase += 0.05
        self._update_swarm()
        self._update_persona()
        self.root.after(self.OVERLAY_MS, self._animate_overlays)

# =========================
# WEB COCKPIT + SWARM + EXTERNAL SYNC APIS
# =========================

WEB_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Sentinel Web Cockpit - {{ node_id }}</title>
  <style>
    body { font-family: Consolas, monospace; background: #111; color: #eee; }
    .panel { border: 1px solid #444; padding: 10px; margin: 10px; }
    h2 { margin-top: 0; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Sentinel Web Cockpit - {{ node_id }}</h1>
  <div class="panel">
    <h2>State</h2>
    <pre>{{ state }}</pre>
  </div>
  <div class="panel">
    <h2>Timeline (last)</h2>
    <pre>{{ timeline }}</pre>
  </div>
  <div class="panel">
    <h2>Threat Matrix</h2>
    <pre>{{ threats }}</pre>
  </div>
  <div class="panel">
    <h2>Persona</h2>
    <pre>{{ persona }}</pre>
  </div>
  <div class="panel">
    <h2>Narrative</h2>
    <pre>{{ narrative }}</pre>
  </div>
</body>
</html>
"""

def start_web_cockpit():
    if not FLASK_AVAILABLE:
        return
    app = Flask(__name__)

    @app.route("/")
    def index():
        st = load_state()
        tl = load_json_list(TIMELINE_FILE, [])
        th = load_json_list(THREATS_FILE, [])
        pe = load_json_dict(PERSONA_FILE, {})
        na = load_json_dict(NARRATIVE_FILE, {})
        return render_template_string(
            WEB_TEMPLATE,
            node_id=LOCAL_NODE_ID,
            state=json.dumps(st, indent=2),
            timeline=json.dumps(tl[-5:], indent=2),
            threats=json.dumps(th, indent=2),
            persona=json.dumps(pe, indent=2),
            narrative=json.dumps(na, indent=2),
        )

    @app.route("/api/state")
    def api_state():
        return jsonify(load_state())

    @app.route("/api/timeline")
    def api_timeline():
        return jsonify(load_json_list(TIMELINE_FILE, []))

    @app.route("/api/threats")
    def api_threats():
        return jsonify(load_json_list(THREATS_FILE, []))

    @app.route("/api/persona")
    def api_persona():
        return jsonify(load_json_dict(PERSONA_FILE, {}))

    @app.route("/api/swarm/state")
    def api_swarm_state():
        logs = load_json_list(LOGS_FILE, [])
        last_score = 0
        for line in reversed(logs):
            if "anomaly_score=" in line:
                try:
                    part = line.split("anomaly_score=")[1]
                    last_score = int(part.split()[0])
                    break
                except Exception:
                    pass
        st = load_state()
        payload = {
            "node": LOCAL_NODE_ID,
            "score": last_score,
            "lockdown": st.get("lockdown", False),
            "autonomous": st.get("autonomous", True)
        }
        sig = _swarm_sign(payload)
        return jsonify({"payload": payload, "sig": sig})

    # Persona fusion: REST push/pull
    @app.route("/api/persona/fuse", methods=["POST"])
    def api_persona_fuse():
        data = request.json or {}
        remote_mode = data.get("mode")
        remote_weight = float(data.get("weight", 1.0))
        local = load_json_dict(PERSONA_FILE, {"mode": "Observer", "autonomous": True})
        local_mode = local.get("mode", "Observer")
        # simple fusion: if remote weight high and mode more aggressive, adopt
        order = ["Observer", "Watcher", "Guardian", "Hunter"]
        try:
            if remote_mode in order and local_mode in order:
                if order.index(remote_mode) > order.index(local_mode) and remote_weight > 0.5:
                    local["mode"] = remote_mode
                    save_json(PERSONA_FILE, local)
        except Exception:
            pass
        return jsonify({"status": "ok", "local_mode": local.get("mode")})

    # Remote operator commands
    @app.route("/api/commands", methods=["POST"])
    def api_commands():
        data = request.json or {}
        cmd = data.get("command")
        st = load_state()
        if cmd == "lockdown_on":
            st["lockdown"] = True
        elif cmd == "lockdown_off":
            st["lockdown"] = False
        elif cmd == "autonomous_on":
            st["autonomous"] = True
        elif cmd == "autonomous_off":
            st["autonomous"] = False
        save_state(st)
        return jsonify({"status": "ok", "state": st})

    # Distributed event bus (simple append)
    @app.route("/api/events", methods=["POST"])
    def api_events():
        data = request.json or {}
        try:
            with open("sentinel_remote_events.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": datetime.utcnow().isoformat() + "Z", "event": data}) + "\n")
        except Exception:
            pass
        return jsonify({"status": "ok"})

    # Cloud-backed telemetry lake (local file stub)
    @app.route("/api/telemetry", methods=["POST"])
    def api_telemetry():
        data = request.json or {}
        try:
            with open(EVENT_LAKE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": datetime.utcnow().isoformat() + "Z", "payload": data}) + "\n")
        except Exception:
            pass
        return jsonify({"status": "ok"})

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=LOCAL_HTTP_PORT, debug=False, use_reloader=False),
        daemon=True
    ).start()

# =========================
# INTEGRITY HASHING + WATCHDOG
# =========================

def compute_self_hash():
    try:
        if getattr(sys, "frozen", False):
            path = sys.executable
        else:
            path = os.path.abspath(sys.argv[0])
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def save_integrity_hash():
    h = compute_self_hash()
    if not h:
        return
    try:
        with open(INTEGRITY_HASH_FILE, "w", encoding="utf-8") as f:
            f.write(h)
    except Exception:
        pass

def verify_integrity_hash():
    h = compute_self_hash()
    if not h:
        return False
    if not os.path.exists(INTEGRITY_HASH_FILE):
        return False
    try:
        with open(INTEGRITY_HASH_FILE, "r", encoding="utf-8") as f:
            stored = f.read().strip()
        return stored == h
    except Exception:
        return False

def run_child():
    organism = OrganismLoop()
    organism.start()

    pipe_server = EventBusServer()
    pipe_server.start()

    start_web_cockpit()

    root = tk.Tk()
    cockpit = OperatorCockpit(root)
    root.geometry("1100x700")
    root.mainloop()

def run_watchdog():
    save_integrity_hash()
    while True:
        try:
            args = [sys.executable, os.path.abspath(sys.argv[0]), "--child"]
            proc = subprocess.Popen(args)
            rc = proc.wait()
            # Auto-restart on crash / non-zero exit
            time.sleep(2)
        except Exception:
            time.sleep(5)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    if "--child" in sys.argv:
        # State resurrection is implicit via persisted JSON files
        run_child()
    else:
        run_watchdog()
