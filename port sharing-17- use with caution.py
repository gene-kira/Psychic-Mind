#!/usr/bin/env python3
"""
Codex Sentinel – Unified Daemon + Mesh + Wrappers + Policy Engine + Raft Consensus + Local LLM
+ Policy Compiler + LLM Feedback Loop + Secure Raft Gossip
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
import uuid
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

# eBPF (Linux)
try:
    from bcc import BPF  # type: ignore
    HAVE_BPF = True
except Exception:
    HAVE_BPF = False

# ----------------- PLATFORM -----------------

def is_windows() -> bool:
    return os.name == "nt"

def is_linux() -> bool:
    return sys.platform.startswith("linux")

def is_macos() -> bool:
    return sys.platform == "darwin"

if is_windows():
    import ctypes
    def ensure_admin():
        try:
            if not ctypes.windll.shell32.IsUserAnAdmin():
                script = os.path.abspath(sys.argv[0])
                params = " ".join([f'"{a}"' for a in sys.argv[1:]])
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable,
                    f'"{script}" {params}', None, 1
                )
                sys.exit()
        except Exception as e:
            print(f"[Codex Sentinel] Elevation failed: {e}")
            sys.exit()
    ensure_admin()

# ----------------- CONFIG -----------------

CONFIG_FILE = "port_enforcer_config.json.enc"
INTEGRITY_FILE = "codex_integrity.json"
LOG_FILE = "port_enforcer.log"
FORENSICS_DIR = "forensics"
PLUGINS_DIR = "plugins"
POLICY_FILE = "codex_policies.json"
POLICY_DSL_FILE = "codex_policies.dsl"   # NEW: human-readable policy DSL
LLM_FEEDBACK_FILE = "llm_feedback.jsonl" # NEW: LLM feedback dataset

ML_MODEL_FILE = "ml_anomaly_model.pkl"

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
THREAT_CONFIRMED_THRESHOLD = 90  # NEW: for LLM feedback

SWARM_BROADCAST_PORT = 50050
MESH_TCP_PORT = 50051
SWARM_NODE_ID = socket.gethostname()
SWARM_KEY = "KILLER666_SWARM_KEY"
SWARM_COMMAND_KEY = "KILLER666_CMD_KEY"
SWARM_NODE_SECRET = "KILLER666_NODE_SECRET"
P2P_MESH_KEY = "P2P_MESH_KEY"
RAFT_GOSSIP_KEY = "KILLER666_RAFT_GOSSIP"  # NEW: secure Raft gossip key

CLUSTER_ID = "CODEX_CLUSTER_ALPHA"

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

# ----------------- LOGGING + HASH CHAIN -----------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
console = logging.getLogger("console")
console.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s] %(message)s"))
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
        logging.info(full); console.info(full)
    elif level == "warning":
        logging.warning(full); console.warning(full)
    elif level == "error":
        logging.error(full); console.error(full)
    else:
        logging.info(full); console.info(full)

# ----------------- GLOBAL STATE -----------------

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
    "lineage_graph": {},
    "raft_term": 0,
    "raft_role": "follower",
    "raft_voted_for": None,
    "raft_log": [],
    "raft_commit_index": -1,
    "raft_last_applied": -1,
    "raft_leader_id": None,
    "raft_peers": set(),
    "mesh_peers": {},
    "code_fingerprint": None,
    "cluster_blocklist": {},
    "cluster_baselines": {},
    "consensus_proposals": {},
}

alert_queue = queue.Queue()
gui_update_queue = queue.Queue()
ml_model = None
plugins = []
policy_rules = []

# ----------------- SELF-EXCLUSION -----------------

def is_self_script(path: str) -> bool:
    this = os.path.abspath(sys.argv[0]).lower()
    target = os.path.abspath(path).lower()
    return this == target or "codex" in os.path.basename(target)

# ----------------- SIMPLE CONFIG ENCRYPTION -----------------

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

# ----------------- SWARM ENCRYPTION + AUTH -----------------

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

# ----------------- SIGNED COMMANDS -----------------

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

# ----------------- CONFIG -----------------

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

# ----------------- CODE FINGERPRINT -----------------

def compute_code_fingerprint() -> str:
    try:
        with open(os.path.abspath(sys.argv[0]), "rb") as f:
            code = f.read()
    except Exception:
        code = b""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "rb") as f:
                cfg = f.read()
        else:
            cfg = b""
    except Exception:
        cfg = b""
    h = hashlib.sha256()
    h.update(code)
    h.update(cfg)
    return h.hexdigest()

# ----------------- ML MODEL -----------------

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

# ----------------- FIREWALL (WINDOWS) -----------------

def firewall_block_port(port, protocol="TCP"):
    if not is_windows():
        return
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule",
        f"name=PortEnforcer_Block_{protocol}_{port}",
        "dir=in", "action=block",
        f"protocol={protocol}", f"localport={port}"
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
        f"protocol={protocol}", f"localport={port}"
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        secure_log("error", f"Failed to remove firewall rule: {e}")

# ----------------- UTIL -----------------

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def sandbox_process(pid):
    try:
        p = psutil.Process(pid)
        if SANDBOX_AFFINITY_LIMIT > 0 and hasattr(p, "cpu_affinity"):
            try:
                p.cpu_affinity(list(range(SANDBOX_AFFINITY_LIMIT)))
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

# ----------------- INTEGRITY -----------------

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

# ----------------- MEMORY SCAN -----------------

def scan_process_memory(pid: int) -> dict:
    info = {"pid": pid, "suspicious": False, "reason": "", "rss": 0}
    try:
        p = psutil.Process(pid)
        mem = p.memory_info().rss
        info["rss"] = mem
        name_len = len(p.name() or "")
        if name_len < 4 and mem > 500 * 1024 * 1024:
            info["suspicious"] = True
            info["reason"] = "Tiny name, huge memory footprint"
    except Exception as e:
        info["reason"] = f"scan_error:{e}"
    return info

# ----------------- LINEAGE -----------------

def update_lineage_graph():
    graph = {}
    for p in psutil.process_iter(attrs=["pid", "ppid"]):
        try:
            pid = p.info["pid"]; ppid = p.info["ppid"]
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
    chain = []; current = pid; visited = set()
    while current and current not in visited:
        visited.add(current); chain.append(current)
        node = graph.get(current)
        if not node: break
        current = node.get("ppid")
    return chain

# ----------------- THREAT / HEATMAP -----------------

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
    m = mean(hist); s = pstdev(hist) or 1.0
    factor = 2.0 if s < 1.0 else 3.0
    return abs(count - m) > factor * s

def is_anomalous(process, port, count) -> bool:
    return adaptive_thresholds(process, port, count)

def get_false_positive_count(process, port) -> int:
    key = (process, port)
    with lock:
        return state["false_positive_counts"].get(key, 0)

def compute_threat_score(event):
    score = 0
    if event.get("unknown_process"): score += WEIGHT_UNKNOWN_PROCESS
    if event.get("unauthorized_port"): score += WEIGHT_UNAUTHORIZED_PORT
    if event.get("port") and event["port"] >= 49152: score += WEIGHT_HIGH_PORT
    if event.get("behavior_anomaly"): score += WEIGHT_BEHAVIOR
    if event.get("correlated"): score += WEIGHT_CORRELATED
    if event.get("gpu_stress"): score += WEIGHT_GPU_STRESS
    return score

def ml_style_threat_score(event):
    base = compute_threat_score(event)
    process = event.get("process"); port = event.get("port")
    anomaly = event.get("anomaly", False)
    history_factor = get_false_positive_count(process, port)
    score = base
    if anomaly: score += WEIGHT_ANOMALY
    if history_factor > 3: score -= WEIGHT_HISTORY
    return max(score, 0)

def update_heatmap(event):
    key = (event.get("process"), event.get("port"))
    with lock:
        hm = state["heatmap"].setdefault(key, 0)
        state["heatmap"][key] = hm + event.get("threat_score", 0)

# ----------------- POLICY COMPILER (DSL → JSON) -----------------

def parse_policy_dsl_line(line: str) -> dict | None:
    """
    Simple DSL:
      rule NAME:
        when process=foo port=80 min_score=50 has_tag=unknown_process,behavior_anomaly
        then raise_score=20 quarantine
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.lower().startswith("rule "):
        # handled in multi-line parser
        return None
    return None  # single-line rules not used; we parse blocks below

def compile_policy_dsl():
    if not os.path.exists(POLICY_DSL_FILE):
        return
    try:
        with open(POLICY_DSL_FILE, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception as e:
        secure_log("error", f"[POLICY] Failed to read DSL: {e}")
        return
    rules = []
    current = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("rule "):
            if current:
                rules.append(current)
            name = line[5:].strip().rstrip(":")
            current = {"name": name, "condition": {}, "actions": []}
        elif line.lower().startswith("when ") and current:
            cond_part = line[5:].strip()
            for token in cond_part.split():
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                k = k.strip(); v = v.strip()
                if k == "min_score":
                    current["condition"][k] = int(v)
                elif k == "port":
                    current["condition"][k] = int(v)
                elif k == "has_tag":
                    current["condition"][k] = [t.strip() for t in v.split(",") if t.strip()]
                else:
                    current["condition"][k] = v
        elif line.lower().startswith("then ") and current:
            act_part = line[5:].strip()
            for token in act_part.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    if k == "raise_score":
                        current["actions"].append({"type": "raise_score", "delta": int(v)})
                else:
                    if token == "quarantine":
                        current["actions"].append({"type": "quarantine"})
                    elif token == "kill":
                        current["actions"].append({"type": "kill"})
                    elif token == "block_port":
                        current["actions"].append({"type": "block_port"})
                    elif token == "forensic_dump":
                        current["actions"].append({"type": "forensic_dump"})
                    elif token == "alert_only":
                        current["actions"].append({"type": "alert_only"})
    if current:
        rules.append(current)
    try:
        with open(POLICY_FILE, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2)
        secure_log("info", f"[POLICY] Compiled DSL → {len(rules)} rules.")
    except Exception as e:
        secure_log("error", f"[POLICY] Failed to write compiled policies: {e}")

# ----------------- POLICY ENGINE -----------------

def load_policies():
    global policy_rules
    # NEW: compile DSL first if present
    compile_policy_dsl()
    if not os.path.exists(POLICY_FILE):
        policy_rules = []
        return
    try:
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            policy_rules = json.load(f)
        secure_log("info", f"[POLICY] Loaded {len(policy_rules)} rules.")
    except Exception as e:
        secure_log("error", f"[POLICY] Failed to load policies: {e}")
        policy_rules = []

def match_policy(event: dict) -> list[dict]:
    matches = []
    for rule in policy_rules:
        cond = rule.get("condition", {})
        ok = True
        if "process" in cond and cond["process"] != event.get("process"):
            ok = False
        if "port" in cond and cond["port"] != event.get("port"):
            ok = False
        if "min_score" in cond and event.get("threat_score", 0) < cond["min_score"]:
            ok = False
        if "has_tag" in cond:
            tags = cond["has_tag"]
            ev_tags = [k for k, v in event.items() if isinstance(v, bool) and v]
            if not any(t in ev_tags for t in tags):
                ok = False
        if not ok:
            continue
        matches.append(rule)
    return matches

def apply_policy_actions(event: dict):
    matches = match_policy(event)
    if not matches:
        return
    event["policy_matched"] = True
    for rule in matches:
        actions = rule.get("actions", [])
        for act in actions:
            t = act.get("type")
            if t == "raise_score":
                delta = int(act.get("delta", 0))
                event["threat_score"] = max(0, event.get("threat_score", 0) + delta)
            elif t == "quarantine":
                quarantine_process(event["pid"], f"Policy: {rule.get('name','')}", event.get("port"))
            elif t == "kill":
                kill_process(event["pid"], f"Policy: {rule.get('name','')}", event.get("port"))
            elif t == "block_port":
                port = event.get("port")
                proto = event.get("protocol", "TCP")
                if port:
                    firewall_block_port(port, proto)
            elif t == "forensic_dump":
                forensic_dump(event)
            elif t == "alert_only":
                secure_log("warning", f"[POLICY] Alert-only rule matched: {rule.get('name','')}")

# ----------------- LOCAL LLM CLASSIFIER + FEEDBACK -----------------

LLM_ENDPOINT = "http://127.0.0.1:11434/api/generate"  # example

def llm_classify_event(event: dict) -> dict | None:
    try:
        import requests
    except ImportError:
        return None
    try:
        prompt = (
            "You are a security analyst. Classify this event as benign, suspicious, or malicious.\n"
            f"Event JSON:\n{json.dumps(event, indent=2)}\n"
            "Respond in JSON: {\"label\": \"...\", \"confidence\": 0-1, \"explanation\": \"...\"}"
        )
        payload = {
            "model": "security-analyst",
            "prompt": prompt,
            "stream": False,
        }
        r = requests.post(LLM_ENDPOINT, json=payload, timeout=5)
        if r.status_code != 200:
            return None
        text = r.json().get("response", "").strip()
        start = text.find("{"); end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        obj = json.loads(text[start:end+1])
        return obj
    except Exception as e:
        secure_log("error", f"[LLM] classify error: {e}")
        return None

def apply_llm_to_event(event: dict):
    res = llm_classify_event(event)
    if not res:
        return
    label = res.get("label", "").lower()
    conf = float(res.get("confidence", 0.0))
    event["llm_label"] = label
    event["llm_confidence"] = conf
    event["llm_explanation"] = res.get("explanation", "")
    if label == "malicious":
        event["threat_score"] += int(30 * conf)
    elif label == "suspicious":
        event["threat_score"] += int(10 * conf)

def llm_feedback_on_event(event: dict):
    """
    NEW: log confirmed incidents for future retraining.
    """
    try:
        label = "malicious" if event["threat_score"] >= THREAT_CONFIRMED_THRESHOLD else "suspicious"
        record = {
            "timestamp": event.get("timestamp"),
            "event": event,
            "label": label,
        }
        with open(LLM_FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        secure_log("error", f"[LLM] feedback log error: {e}")

# ----------------- EVENTS / FORENSICS -----------------

def record_event(event):
    event["timestamp"] = datetime.utcnow().isoformat()
    event["threat_score"] = ml_style_threat_score(event)

    apply_policy_actions(event)
    apply_llm_to_event(event)

    with lock:
        state["events"].append(event)
        state["events"] = state["events"][-200:]
    update_heatmap(event)
    alert_queue.put(event)
    gui_update_queue.put(("event", event))
    plugins_on_event(event)
    raft_append_entry({"type": "THREAT_EVENT", "event": event})
    send_threat_event_intel(event)

    # NEW: LLM feedback loop on confirmed incidents
    if event["threat_score"] >= THREAT_CONFIRMED_THRESHOLD or event.get("policy_matched"):
        llm_feedback_on_event(event)

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
            "exe": p.exe() or "",
            "cmdline": p.cmdline(),
            "username": p.username(),
            "create_time": p.create_time(),
            "cpu_percent": p.cpu_percent(interval=0.05),
            "memory_info": p.memory_info()._asdict(),
            "status": p.status(),
            "ppid": p.ppid(),
        }
        for c in p.connections(kind="inet"):
            if not c.laddr: continue
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

# ----------------- PLUGINS -----------------

def load_plugins():
    if not os.path.isdir(PLUGINS_DIR):
        return
    sys.path.insert(0, os.path.abspath(PLUGINS_DIR))
    for file in os.listdir(PLUGINS_DIR):
        if not file.endswith(".py"): continue
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
            try: fn(event)
            except Exception as e:
                secure_log("error", f"[PLUGIN] on_event error in {mod.__name__}: {e}")

def plugins_on_tick():
    for mod in plugins:
        fn = getattr(mod, "on_tick", None)
        if callable(fn):
            try: fn()
            except Exception as e:
                secure_log("error", f"[PLUGIN] on_tick error in {mod.__name__}: {e}")

# ----------------- LEARNING -----------------

def learn_ports_from_connections(connections):
    updated = False
    with lock:
        allowed_ports = state["allowed_ports"]
        for conn in connections:
            pid = conn.pid
            if pid is None or not conn.laddr: continue
            port = conn.laddr.port
            try:
                name = psutil.Process(pid).name().lower()
            except Exception:
                continue
            ports = allowed_ports.setdefault(name, [])
            if port not in ports:
                ports.append(port); updated = True
        if updated:
            state["last_new_port_time"] = datetime.utcnow()
    if updated: save_config()
    return updated

def maybe_auto_flip_mode():
    with lock:
        learn_mode = state["learn_mode"]
        last_new = state["last_new_port_time"]
    if not learn_mode or last_new is None: return
    stable_for = (datetime.utcnow() - last_new).total_seconds()
    if stable_for >= STABLE_WINDOW_SECONDS:
        with lock:
            state["learn_mode"] = False
        secure_log("info", "Auto-flip: Learning → Enforcement")
        save_config()

# ----------------- ENFORCEMENT -----------------

def enforce_rules(connections):
    with lock:
        allowed_ports = state["allowed_ports"].copy()
        learn_mode = state["learn_mode"]
        gpu_util = state["gpu_util"]
        cluster_blocklist = state["cluster_blocklist"].copy()
    if learn_mode: return

    for conn in connections:
        pid = conn.pid
        if pid is None or not conn.laddr: continue
        port = conn.laddr.port
        proto = "TCP" if conn.type == psutil.SOCK_STREAM else "UDP"
        try:
            proc = psutil.Process(pid)
            name = proc.name().lower()
        except Exception:
            name = "unknown"
            proc = None

        unknown_process = name not in allowed_ports
        unauthorized_port = False
        if not unknown_process and port not in allowed_ports.get(name, []):
            unauthorized_port = True

        bl_key_port = ("port", str(port))
        bl_key_proc = ("process", name)
        blocked_by_cluster = bl_key_port in cluster_blocklist or bl_key_proc in cluster_blocklist

        if unknown_process or unauthorized_port or blocked_by_cluster:
            cpu = mem = conn_count = 0
            if proc:
                try:
                    cpu = proc.cpu_percent(interval=0.01)
                    mem = proc.memory_info().rss
                    conn_count = len(proc.connections(kind="inet"))
                except Exception:
                    pass
            update_behavior_fingerprint(name, port, cpu, mem, conn_count)
            correlated = cpu > 10.0 or conn_count > 5 or blocked_by_cluster
            gpu_stress = gpu_util is not None and gpu_util > 80
            mem_scan = scan_process_memory(pid)
            if mem_scan.get("suspicious"): correlated = True

            reason = "Unknown/unauthorized port"
            if blocked_by_cluster:
                reason = "Cluster blocklist"
            if correlated and not blocked_by_cluster:
                reason = "Unknown/unauthorized port with suspicious behavior"

            event = {
                "pid": pid, "process": name, "port": port, "protocol": proto,
                "unknown_process": unknown_process,
                "unauthorized_port": unauthorized_port,
                "behavior_anomaly": correlated,
                "correlated": correlated,
                "gpu_stress": gpu_stress,
                "reason": reason,
                "mem_scan": mem_scan,
            }
            record_event(event)
            if event["threat_score"] >= THREAT_FORENSIC_THRESHOLD:
                forensic_dump(event)
                if event.get("port"):
                    propose_block_port_cluster(event["port"], event.get("reason",""), event["threat_score"])
            if correlated: quarantine_process(pid, event["reason"], port)
            else: kill_process(pid, event["reason"], port)
            firewall_block_port(port, proto)

# ----------------- SCANNER -----------------

def scanner_loop():
    secure_log("info", "Scanner thread started.")
    while True:
        try:
            connections = psutil.net_connections(kind="inet")
        except Exception as e:
            secure_log("error", f"Failed to get connections: {e}")
            time.sleep(SCAN_INTERVAL); continue

        with lock:
            state["last_scan"] = datetime.utcnow().isoformat()
            learn_mode = state["learn_mode"]
            gpu_util = state["gpu_util"]

        update_lineage_graph()

        if learn_mode:
            updated = learn_ports_from_connections(connections)
            if not updated: maybe_auto_flip_mode()
        else:
            enforce_rules(connections)

        port_counts = {}
        for c in connections:
            if c.laddr:
                port_counts[c.laddr.port] = port_counts.get(c.laddr.port, 0) + 1

        snapshot = []
        for c in connections:
            pid = c.pid
            if pid is None or not c.laddr: continue
            try:
                proc = psutil.Process(pid)
                name = proc.name().lower()
            except Exception:
                name = "unknown"; proc = None
            proto = "TCP" if c.type == psutil.SOCK_STREAM else "UDP"
            port = c.laddr.port
            shared = port_counts.get(port, 0) > 1
            count = port_counts.get(port, 0)
            update_port_usage_stats(name, port, count)

            cpu = mem = 0
            if proc:
                try:
                    cpu = proc.cpu_percent(interval=0.0)
                    mem = proc.memory_info().rss
                except Exception:
                    pass
            update_behavior_fingerprint(name, port, cpu, mem, count)

            heuristic_anomaly = is_anomalous(name, port, count)
            ml_flag = ml_anomaly_flag(c, name)
            anomaly = heuristic_anomaly or ml_flag
            gpu_stress = gpu_util is not None and gpu_util > 80

            if anomaly or gpu_stress:
                mem_scan = scan_process_memory(pid)
                event = {
                    "pid": pid, "process": name, "port": port, "protocol": proto,
                    "unknown_process": False, "unauthorized_port": False,
                    "anomaly": anomaly, "behavior_anomaly": heuristic_anomaly,
                    "correlated": False, "gpu_stress": gpu_stress,
                    "reason": "Anomalous port usage pattern (ML/heuristic/GPU)",
                    "mem_scan": mem_scan,
                }
                record_event(event)
                if event["threat_score"] >= THREAT_FORENSIC_THRESHOLD:
                    forensic_dump(event)
                    if event.get("port"):
                        propose_block_port_cluster(event["port"], event.get("reason",""), event["threat_score"])
                sandbox_process(pid)

            snapshot.append({
                "pid": pid, "process": name, "port": port,
                "protocol": proto, "status": c.status, "shared": shared,
            })

        gui_update_queue.put(("snapshot", snapshot))
        plugins_on_tick()
        time.sleep(SCAN_INTERVAL)

# ----------------- ALERT HANDLER -----------------

def alert_handler_loop():
    secure_log("info", "Alert handler started.")
    while True:
        event = alert_queue.get()
        if event is None: break
        if event["threat_score"] >= THREAT_ALERT_THRESHOLD:
            secure_log(
                "warning",
                f"[ALERT] Score {event['threat_score']} - {event['process']} "
                f"PID {event['pid']} PORT {event['port']} REASON: {event.get('reason','')}"
            )

# ----------------- SUPERVISOR WRAPPERS (CodexModified) -----------------

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
    name, ext = os.path.splitext(base)
    wrapper_name = f"{name}{WRAPPER_SUFFIX}"
    wrapper_path = os.path.join(mod_dir, wrapper_name)

    if not os.path.exists(wrapper_path):
        secure_log("info", f"[WRAPPER] Creating supervisor wrapper for {script_path}")
        wrapper_code = f'''import os
import sys
import time
import json
import socket
import psutil
import subprocess
from datetime import datetime, timedelta

WRAPPER_STATE_FILE = os.path.join(os.path.dirname(__file__), "{name}_wrapper_state.json")
TARGET_SCRIPT = r"{script_path.replace("\\", "\\\\")}"
HEALTH_MIN_SECONDS = {HEALTH_MIN_SECONDS}
HEALTH_MAX_SECONDS = {HEALTH_MAX_SECONDS}
CRASH_WINDOW_SECONDS = {WRAPPER_CRASH_WINDOW_SECONDS}
CRASH_MAX_RESTARTS = {WRAPPER_CRASH_MAX_RESTARTS}

def load_state():
    if not os.path.exists(WRAPPER_STATE_FILE):
        return {{"crashes": []}}
    try:
        with open(WRAPPER_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {{"crashes": []}}

def save_state(state):
    try:
        with open(WRAPPER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def record_crash():
    st = load_state()
    now = datetime.utcnow().isoformat()
    st.setdefault("crashes", []).append(now)
    cutoff = datetime.utcnow() - timedelta(seconds=CRASH_WINDOW_SECONDS)
    st["crashes"] = [c for c in st["crashes"] if c >= cutoff.isoformat()]
    save_state(st)

def crash_loop_active():
    st = load_state()
    crashes = st.get("crashes", [])
    cutoff = datetime.utcnow() - timedelta(seconds=CRASH_WINDOW_SECONDS)
    recent = [c for c in crashes if c >= cutoff.isoformat()]
    return len(recent) >= CRASH_MAX_RESTARTS

def wait_for_port(port, timeout):
    end = time.time() + timeout
    while time.time() < end:
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.laddr.port == port:
                return True
        time.sleep(0.5)
    return False

def main():
    if len(sys.argv) < 2:
        print("[WRAPPER] Missing port argument")
        sys.exit(1)
    port = int(sys.argv[1])
    if crash_loop_active():
        print("[WRAPPER] Crash loop detected, backing off.")
        time.sleep(HEALTH_MAX_SECONDS)
        sys.exit(1)

    os.environ["AUTO_PORT"] = str(port)

    cmd = [sys.executable, TARGET_SCRIPT]
    print(f"[WRAPPER] Launching {{TARGET_SCRIPT}} on AUTO_PORT={{port}}")
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(TARGET_SCRIPT))

    start = time.time()
    healthy = wait_for_port(port, HEALTH_MAX_SECONDS)
    if not healthy:
        print("[WRAPPER] Port did not bind in time, marking as crash.")
        record_crash()
        try:
            proc.terminate()
        except Exception:
            pass
        sys.exit(1)

    while True:
        ret = proc.poll()
        if ret is not None:
            runtime = time.time() - start
            if runtime < HEALTH_MIN_SECONDS:
                print("[WRAPPER] Process died too quickly, recording crash.")
                record_crash()
            else:
                print("[WRAPPER] Process exited cleanly.")
            sys.exit(ret)
        time.sleep(1)

if __name__ == "__main__":
    main()
'''
        try:
            with open(wrapper_path, "w", encoding="utf-8") as f:
                f.write(wrapper_code)
        except Exception as e:
            secure_log("error", f"[WRAPPER] Failed to create wrapper: {e}")

    register_integrity(script_path)
    register_integrity(wrapper_path)
    return wrapper_path

def discover_python_startup_programs():
    if not is_windows():
        return []
    programs = []
    for folder in [USER_STARTUP_DIR, SYSTEM_STARTUP_DIR]:
        if not folder or not os.path.isdir(folder): continue
        for file in os.listdir(folder):
            if not file.lower().endswith(".py"): continue
            full_path = os.path.join(folder, file)
            if MOD_DIR_NAME.lower() in full_path.lower(): continue
            if full_path.lower().endswith(WRAPPER_SUFFIX): continue
            if is_self_script(full_path): continue
            wrapper_path = ensure_wrapper_for_script(full_path)
            if not wrapper_path: continue
            programs.append({
                "id": wrapper_path.lower(),
                "name": "python.exe",
                "match_path": wrapper_path.lower(),
                "script_path": full_path.lower(),
                "command_template": [sys.executable, wrapper_path, "{PORT}"],
                "cwd": os.path.dirname(wrapper_path)
            })
    return programs

def is_process_running_for_entry(entry):
    target_name = entry["name"].lower()
    target_path = entry["match_path"]
    for p in psutil.process_iter(attrs=["name", "cmdline"]):
        try:
            if not p.info["name"]: continue
            if p.info["name"].lower() != target_name: continue
            cmdline = " ".join(p.info.get("cmdline") or []).lower()
            if target_path in cmdline: return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def is_original_script_running(script_path: str) -> bool:
    target = os.path.abspath(script_path).lower()
    for p in psutil.process_iter(attrs=["name", "cmdline"]):
        try:
            cmdline = " ".join(p.info.get("cmdline") or []).lower()
            if target in cmdline: return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

MANAGED_PROGRAMS = []

def launch_program(entry):
    if not verify_integrity(entry["match_path"]):
        secure_log("warning", f"[WATCHDOG] Integrity failed for wrapper {entry['match_path']}, not launching.")
        return
    port = get_free_port()
    cmd = [a.replace("{PORT}", str(port)) for a in entry["command_template"]]
    try:
        subprocess.Popen(cmd, cwd=entry["cwd"])
        secure_log("info", f"[WATCHDOG] Launched wrapper: {cmd} (port {port})")
    except Exception as e:
        secure_log("error", f"[WATCHDOG] Failed to launch {cmd}: {e}")

def watchdog_loop():
    if not is_windows():
        secure_log("info", "Watchdog disabled on non-Windows.")
        return
    secure_log("info", "Watchdog thread started.")
    while True:
        for entry in MANAGED_PROGRAMS:
            script_path = entry.get("script_path")
            if script_path and is_original_script_running(script_path): continue
            if not is_process_running_for_entry(entry):
                secure_log("warning", f"[WATCHDOG] Wrapper not running: {entry['match_path']}. Relaunching...")
                launch_program(entry)
        time.sleep(5)

# ----------------- SWARM (UDP DISCOVERY) -----------------

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
                    "cluster_id": CLUSTER_ID,
                    "code_fingerprint": state["code_fingerprint"],
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
            if not payload: continue
            node = payload.get("node"); sig = payload.get("node_sig")
            if not node or node == SWARM_NODE_ID: continue
            if not sig or not verify_node_identity(node, sig):
                secure_log("warning", "[SWARM] Rejected unauthenticated heartbeat.")
                continue
            with lock:
                state["swarm_peers"][node] = payload.get("time")
                state["raft_peers"].add(node)
            mesh_connect_to_peer(addr[0], node, payload)
        except Exception as e:
            secure_log("error", f"Swarm listener error: {e}")

# ----------------- GPU TELEMETRY -----------------

def gpu_telemetry_loop():
    secure_log("info", "GPU telemetry thread started.")
    while True:
        util = None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                line = result.stdout.strip().splitlines()[0]
                util = int(line)
        except Exception:
            util = None
        with lock:
            state["gpu_util"] = util
        time.sleep(GPU_TELEMETRY_INTERVAL)

# ----------------- HEATMAP SNAPSHOT -----------------

def get_heatmap_snapshot(top_n=10):
    with lock:
        items = list(state["heatmap"].items())
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:top_n]

# ----------------- SECURE RAFT GOSSIP (HMAC) -----------------

def sign_raft_message(msg: dict) -> str:
    key = _derive_key(RAFT_GOSSIP_KEY)
    body = {k: v for k, v in msg.items() if k != "raft_sig"}
    raw = json.dumps(body, sort_keys=True).encode("utf-8")
    return hmac.new(key, raw, hashlib.sha256).hexdigest()

def verify_raft_message(msg: dict) -> bool:
    sig = msg.get("raft_sig")
    if not sig:
        return False
    key = _derive_key(RAFT_GOSSIP_KEY)
    body = {k: v for k, v in msg.items() if k != "raft_sig"}
    raw = json.dumps(body, sort_keys=True).encode("utf-8")
    expected = hmac.new(key, raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)

# ----------------- RAFT (MINIMAL) -----------------

def raft_append_entry(command: dict):
    with lock:
        entry = {"term": state["raft_term"], "command": command}
        state["raft_log"].append(entry)
    raft_broadcast_append()

def raft_broadcast_append():
    with lock:
        leader = state["raft_leader_id"]
        peers = list(state["mesh_peers"].keys())
        log_copy = list(state["raft_log"])
    if leader and leader != SWARM_NODE_ID:
        return
    msg = {"type": "RAFT_APPEND", "from": SWARM_NODE_ID, "log": log_copy}
    msg["raft_sig"] = sign_raft_message(msg)  # NEW
    mesh_broadcast(msg)

def raft_apply_committed():
    with lock:
        while state["raft_last_applied"] < state["raft_commit_index"]:
            state["raft_last_applied"] += 1
            entry = state["raft_log"][state["raft_last_applied"]]
            cmd = entry["command"]
            # apply replicated commands if needed

def raft_loop():
    secure_log("info", "Raft loop started.")
    election_timeout = lambda: time.time() + (3 + 2 * (hash(SWARM_NODE_ID) % 3))
    next_election = election_timeout()
    heartbeat_interval = 1.0
    last_heartbeat = time.time()
    while True:
        time.sleep(0.2)
        with lock:
            role = state["raft_role"]; term = state["raft_term"]
        now = time.time()
        if role == "leader":
            if now - last_heartbeat >= heartbeat_interval:
                msg = {"type": "RAFT_HEARTBEAT", "from": SWARM_NODE_ID, "term": term}
                msg["raft_sig"] = sign_raft_message(msg)  # NEW
                mesh_broadcast(msg)
                last_heartbeat = now
        else:
            if now >= next_election:
                with lock:
                    state["raft_role"] = "candidate"
                    state["raft_term"] += 1
                    state["raft_voted_for"] = SWARM_NODE_ID
                    term = state["raft_term"]
                secure_log("info", f"[RAFT] {SWARM_NODE_ID} starting election term {term}")
                msg = {"type": "RAFT_REQUEST_VOTE", "from": SWARM_NODE_ID, "term": term}
                msg["raft_sig"] = sign_raft_message(msg)  # NEW
                mesh_broadcast(msg)
                next_election = election_timeout()
        raft_apply_committed()

def raft_handle_message(msg: dict):
    if not verify_raft_message(msg):
        secure_log("warning", "[RAFT] Dropped unauthenticated Raft message.")
        return
    mtype = msg.get("type")
    if mtype == "RAFT_HEARTBEAT":
        with lock:
            if msg["term"] >= state["raft_term"]:
                state["raft_term"] = msg["term"]
                state["raft_role"] = "follower"
                state["raft_leader_id"] = msg["from"]
    elif mtype == "RAFT_REQUEST_VOTE":
        with lock:
            if msg["term"] > state["raft_term"]:
                state["raft_term"] = msg["term"]
                state["raft_voted_for"] = None
                state["raft_role"] = "follower"
            if state["raft_voted_for"] in (None, msg["from"]) and msg["term"] >= state["raft_term"]:
                state["raft_voted_for"] = msg["from"]
                reply = {"type": "RAFT_VOTE", "from": SWARM_NODE_ID, "term": state["raft_term"]}
                reply["raft_sig"] = sign_raft_message(reply)  # NEW
                mesh_send_to(msg["from"], reply)
    elif mtype == "RAFT_VOTE":
        with lock:
            if state["raft_role"] == "candidate" and msg["term"] == state["raft_term"]:
                state["raft_role"] = "leader"
                state["raft_leader_id"] = SWARM_NODE_ID
                secure_log("info", f"[RAFT] {SWARM_NODE_ID} became leader term {state['raft_term']}")
    elif mtype == "RAFT_APPEND":
        with lock:
            if msg["from"] != SWARM_NODE_ID:
                state["raft_log"] = msg["log"]
                state["raft_leader_id"] = msg["from"]
                state["raft_role"] = "follower"
                state["raft_commit_index"] = len(state["raft_log"]) - 1

# ----------------- P2P MESH (TCP, ENCRYPTED) -----------------

def p2p_encrypt(data: bytes) -> bytes:
    key = _derive_key(P2P_MESH_KEY)
    out = bytearray()
    for i, b in enumerate(data):
        out.append(b ^ key[i % len(key)])
    return base64.b64encode(out)

def p2p_decrypt(blob: bytes) -> bytes | None:
    try:
        key = _derive_key(P2P_MESH_KEY)
        raw = base64.b64decode(blob)
        out = bytearray()
        for i, b in enumerate(raw):
            out.append(b ^ key[i % len(key)])
        return bytes(out)
    except Exception:
        return None

def mesh_send(sock: socket.socket, msg: dict):
    try:
        raw = json.dumps(msg).encode("utf-8")
        enc = p2p_encrypt(raw)
        length = len(enc).to_bytes(4, "big")
        sock.sendall(length + enc)
    except Exception as e:
        secure_log("error", f"[MESH] send error: {e}")

def mesh_recv(sock: socket.socket) -> dict | None:
    try:
        hdr = sock.recv(4)
        if not hdr: return None
        length = int.from_bytes(hdr, "big")
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk: return None
            data += chunk
        dec = p2p_decrypt(data)
        if not dec: return None
        return json.loads(dec.decode("utf-8"))
    except Exception:
        return None

def mesh_broadcast(msg: dict):
    with lock:
        peers = list(state["mesh_peers"].items())
    for node_id, (sock, _, _) in peers:
        mesh_send(sock, msg)

def mesh_send_to(node_id: str, msg: dict):
    with lock:
        peer = state["mesh_peers"].get(node_id)
    if not peer: return
    sock, _, _ = peer
    mesh_send(sock, msg)

def mesh_peer_loop(sock: socket.socket, node_id: str):
    secure_log("info", f"[MESH] Peer loop started for {node_id}")
    while True:
        msg = mesh_recv(sock)
        if msg is None:
            secure_log("warning", f"[MESH] Peer {node_id} disconnected.")
            with lock:
                state["mesh_peers"].pop(node_id, None)
            break
        mtype = msg.get("type")
        if mtype and mtype.startswith("RAFT_"):
            raft_handle_message(msg)
        elif mtype == "THREAT_INTEL":
            handle_threat_intel(msg)
        elif mtype == "CONSENSUS_PROPOSAL":
            handle_consensus_proposal(msg)
        elif mtype == "CONSENSUS_VOTE":
            handle_consensus_vote(msg)

def mesh_accept_loop():
    secure_log("info", "MESH TCP accept loop started.")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", MESH_TCP_PORT))
    srv.listen(5)
    while True:
        conn, addr = srv.accept()
        try:
            hello = mesh_recv(conn)
            if not hello or hello.get("type") != "HELLO":
                conn.close(); continue
            node = hello.get("node_id")
            sig = hello.get("sig")
            cfp = hello.get("code_fingerprint")
            cluster = hello.get("cluster_id")
            if not node or not sig or not verify_node_identity(node, sig):
                conn.close(); continue
            with lock:
                local_cfp = state["code_fingerprint"]
                is_twin = (cfp == local_cfp)
                meta = {
                    "is_twin": is_twin,
                    "cluster_id": cluster,
                    "code_fingerprint": cfp,
                }
                state["mesh_peers"][node] = (conn, time.time(), meta)
            threading.Thread(target=mesh_peer_loop, args=(conn, node), daemon=True).start()
            secure_log("info", f"[MESH] Accepted peer {node} twin={meta['is_twin']}")
        except Exception:
            conn.close()

def mesh_connect_to_peer(ip: str, node_id: str, heartbeat_payload: dict | None = None):
    with lock:
        if node_id in state["mesh_peers"]:
            return
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((ip, MESH_TCP_PORT))
        with lock:
            hello = {
                "type": "HELLO",
                "node_id": SWARM_NODE_ID,
                "code_fingerprint": state["code_fingerprint"],
                "cluster_id": CLUSTER_ID,
                "sig": sign_node_identity(SWARM_NODE_ID),
            }
        mesh_send(sock, hello)
        with lock:
            remote_cfp = heartbeat_payload.get("code_fingerprint") if heartbeat_payload else None
            is_twin = (remote_cfp == state["code_fingerprint"])
            meta = {
                "is_twin": is_twin,
                "cluster_id": heartbeat_payload.get("cluster_id") if heartbeat_payload else None,
                "code_fingerprint": remote_cfp,
            }
            state["mesh_peers"][node_id] = (sock, time.time(), meta)
        threading.Thread(target=mesh_peer_loop, args=(sock, node_id), daemon=True).start()
        secure_log("info", f"[MESH] Connected to peer {node_id}@{ip} twin={meta['is_twin']}")
    except Exception:
        pass

# ----------------- eBPF (REAL HOOK) -----------------

def ebpf_loop():
    if not is_linux() or not HAVE_BPF:
        secure_log("info", "eBPF not available; skipping.")
        return
    secure_log("info", "eBPF loop started.")
    program = r"""
    #include <uapi/linux/ptrace.h>
    #include <net/sock.h>
    #include <net/inet_sock.h>

    struct event_t {
        u32 pid;
        u16 dport;
    };

    BPF_PERF_OUTPUT(events);

    int kprobe__tcp_connect(struct pt_regs *ctx, struct sock *sk) {
        struct event_t ev = {};
        u16 dport = 0;
        bpf_probe_read_kernel(&dport, sizeof(dport), &inet_sk(sk)->inet_dport);
        ev.pid = bpf_get_current_pid_tgid() >> 32;
        ev.dport = ntohs(dport);
        events.perf_submit(ctx, &ev, sizeof(ev));
        return 0;
    }
    """
    try:
        b = BPF(text=program)
        def handle_event(cpu, data, size):
            ev = b["events"].event(data)
            pid = int(ev.pid); port = int(ev.dport)
            try:
                p = psutil.Process(pid)
                name = p.name().lower()
            except Exception:
                name = "unknown"
            event = {
                "pid": pid, "process": name, "port": port,
                "protocol": "TCP", "unknown_process": False,
                "unauthorized_port": False, "behavior_anomaly": False,
                "correlated": False, "gpu_stress": False,
                "reason": "eBPF tcp_connect trace",
            }
            record_event(event)
        b["events"].open_perf_buffer(handle_event)
        while True:
            b.perf_buffer_poll()
    except Exception as e:
        secure_log("error", f"eBPF error: {e}")

# ----------------- THREAT INTEL PROTOCOL -----------------

def is_peer_twin(node_id: str) -> bool:
    with lock:
        peer = state["mesh_peers"].get(node_id)
    if not peer: return False
    _, _, meta = peer
    return bool(meta.get("is_twin"))

def intel_envelope(payload: dict) -> dict:
    with lock:
        cfp = state["code_fingerprint"]
    return {
        "type": "THREAT_INTEL",
        "node_id": SWARM_NODE_ID,
        "code_fingerprint": cfp,
        "cluster_id": CLUSTER_ID,
        "ts": datetime.utcnow().isoformat(),
        "payload": payload,
    }

def send_blocklist_update(entries: list[dict], version: int):
    payload = {
        "subtype": "BLOCKLIST_UPDATE",
        "version": version,
        "entries": entries,
    }
    msg = intel_envelope(payload)
    mesh_broadcast(msg)

def send_baseline_update(process: str, port: int, features: dict, samples: int):
    payload = {
        "subtype": "BASELINE_UPDATE",
        "process": process,
        "port": port,
        "features": features,
        "samples": samples,
    }
    msg = intel_envelope(payload)
    mesh_broadcast(msg)

def send_threat_event_intel(event: dict):
    payload = {
        "subtype": "THREAT_EVENT",
        "event_id": str(uuid.uuid4()),
        "process": event.get("process"),
        "port": event.get("port"),
        "score": event.get("threat_score", 0),
        "tags": [
            k for k, v in event.items()
            if isinstance(v, bool) and v
        ],
        "reason": event.get("reason", ""),
    }
    msg = intel_envelope(payload)
    mesh_broadcast(msg)

def merge_blocklist_entry(entry: dict):
    kind = entry.get("kind")
    value = str(entry.get("value"))
    if not kind or value is None:
        return
    key = (kind, value)
    with lock:
        bl = state["cluster_blocklist"].setdefault(key, {
            "kind": kind,
            "value": value,
            "reason": entry.get("reason", ""),
            "source_nodes": [],
            "score": 0,
        })
        bl["score"] = max(bl.get("score", 0), entry.get("score", 0))
        src = entry.get("source_nodes", [])
        for n in src:
            if n not in bl["source_nodes"]:
                bl["source_nodes"].append(n)

def merge_baseline(process: str, port: int, features: dict, samples: int):
    key = (process, port)
    with lock:
        existing = state["cluster_baselines"].get(key)
        if not existing:
            state["cluster_baselines"][key] = {
                "avg_cpu": features.get("avg_cpu", 0.0),
                "avg_mem": features.get("avg_mem", 0.0),
                "avg_conn": features.get("avg_conn_count", 0.0),
                "samples": samples,
            }
            return
        total_samples = existing["samples"] + samples
        if total_samples <= 0:
            return
        def wavg(old, old_n, new, new_n):
            return (old * old_n + new * new_n) / total_samples
        state["cluster_baselines"][key] = {
            "avg_cpu": wavg(existing["avg_cpu"], existing["samples"], features.get("avg_cpu", 0.0), samples),
            "avg_mem": wavg(existing["avg_mem"], existing["samples"], features.get("avg_mem", 0.0), samples),
            "avg_conn": wavg(existing["avg_conn"], existing["samples"], features.get("avg_conn_count", 0.0), samples),
            "samples": total_samples,
        }

def handle_threat_intel(msg: dict):
    payload = msg.get("payload") or {}
    subtype = payload.get("subtype")
    sender = msg.get("node_id")
    cfp_remote = msg.get("code_fingerprint")
    cluster = msg.get("cluster_id")
    with lock:
        local_cfp = state["code_fingerprint"]
    is_twin = (cfp_remote == local_cfp and cluster == CLUSTER_ID)

    if subtype == "BLOCKLIST_UPDATE":
        entries = payload.get("entries", [])
        for e in entries:
            merge_blocklist_entry(e)
        secure_log("info", f"[INTEL] BLOCKLIST_UPDATE from {sender}, entries={len(entries)} twin={is_twin}")
    elif subtype == "BASELINE_UPDATE":
        process = payload.get("process")
        port = payload.get("port")
        features = payload.get("features", {})
        samples = payload.get("samples", 0)
        if process and port is not None:
            merge_baseline(process, port, features, samples)
            secure_log("info", f"[INTEL] BASELINE_UPDATE from {sender} {process}:{port} twin={is_twin}")
    elif subtype == "THREAT_EVENT":
        secure_log("info", f"[INTEL] THREAT_EVENT from {sender}: {payload.get('process')}:{payload.get('port')} score={payload.get('score')} twin={is_twin}")

# ----------------- CONSENSUS (REAL VOTE) -----------------

def propose_block_port_cluster(port: int, reason: str, score: int):
    proposal_id = str(uuid.uuid4())
    action = {
        "kind": "BLOCK_PORT",
        "port": port,
        "reason": reason,
        "score": score,
    }
    with lock:
        state["consensus_proposals"][proposal_id] = {
            "action": action,
            "votes_yes": 1,
            "votes_no": 0,
            "voters": {SWARM_NODE_ID},
        }
    msg = {
        "type": "CONSENSUS_PROPOSAL",
        "proposal_id": proposal_id,
        "from": SWARM_NODE_ID,
        "action": action,
    }
    mesh_broadcast(msg)

def handle_consensus_proposal(msg: dict):
    proposal_id = msg.get("proposal_id")
    action = msg.get("action") or {}
    with lock:
        proposals = state["consensus_proposals"]
        if proposal_id not in proposals:
            proposals[proposal_id] = {
                "action": action,
                "votes_yes": 0,
                "votes_no": 0,
                "voters": set(),
            }
        proposals[proposal_id]["votes_yes"] += 1
        proposals[proposal_id]["voters"].add(SWARM_NODE_ID)
    vote_msg = {
        "type": "CONSENSUS_VOTE",
        "proposal_id": proposal_id,
        "from": SWARM_NODE_ID,
        "vote": "yes",
    }
    mesh_send_to(msg.get("from"), vote_msg)

def handle_consensus_vote(msg: dict):
    proposal_id = msg.get("proposal_id")
    vote = msg.get("vote")
    voter = msg.get("from")
    with lock:
        proposals = state["consensus_proposals"]
        peers = state["mesh_peers"]
        if proposal_id not in proposals:
            return
        p = proposals[proposal_id]
        if voter in p["voters"]:
            return
        p["voters"].add(voter)
        if vote == "yes":
            p["votes_yes"] += 1
        else:
            p["votes_no"] += 1
        total_nodes = len(peers) + 1
        majority = total_nodes // 2 + 1
        if p["votes_yes"] >= majority:
            action = p["action"]
            if action.get("kind") == "BLOCK_PORT":
                port = action.get("port")
                reason = action.get("reason", "cluster consensus")
                score = action.get("score", 0)
                entry = {
                    "kind": "port",
                    "value": str(port),
                    "reason": reason,
                    "score": score,
                    "source_nodes": list(p["voters"]),
                }
                merge_blocklist_entry(entry)
                send_blocklist_update([entry], version=int(time.time()))
                secure_log("info", f"[CONSENSUS] Blocklist accepted for port {port}")
            del proposals[proposal_id]

# ----------------- GUI -----------------

class PortEnforcerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Codex Sentinel Cockpit")
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
        heat_lines = ["Heatmap (top):"] + [f"{p}:{port} -> {score}" for (p, port), score in heat_items]
        self.heatmap_label.config(text="\n".join(heat_lines))
        self.status_label.config(text=f"Mode={mode} | LastScan={last_scan} | {gpu_text}")
        self.hud_label.config(text=f"Mode: {mode}\n{gpu_text}\n{swarm_text}\n" + "\n".join(heat_lines))
        self.root.after(500, self.refresh_gui)

    def update_snapshot(self, snapshot):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for item in snapshot:
            self.tree.insert(
                "", tk.END,
                values=(item["pid"], item["process"], item["port"],
                        item["protocol"], item["status"],
                        "Yes" if item["shared"] else "No")
            )

# ----------------- DAEMON -----------------

def run_daemon(headless=True):
    load_integrity_db()
    load_config()
    with lock:
        state["code_fingerprint"] = compute_code_fingerprint()
    load_ml_model()
    load_plugins()
    load_policies()

    global MANAGED_PROGRAMS
    MANAGED_PROGRAMS = [p for p in discover_python_startup_programs() if p]

    threading.Thread(target=scanner_loop, daemon=True).start()
    threading.Thread(target=alert_handler_loop, daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    threading.Thread(target=swarm_heartbeat_loop, daemon=True).start()
    threading.Thread(target=swarm_listener_loop, daemon=True).start()
    threading.Thread(target=gpu_telemetry_loop, daemon=True).start()
    threading.Thread(target=raft_loop, daemon=True).start()
    threading.Thread(target=mesh_accept_loop, daemon=True).start()
    threading.Thread(target=ebpf_loop, daemon=True).start()

    if headless or tk is None:
        secure_log("info", "Running headless daemon.")
        while True:
            time.sleep(1)
    else:
        root = tk.Tk()
        PortEnforcerGUI(root)
        root.mainloop()

def main():
    headless = ("--daemon" in sys.argv) or ("--headless" in sys.argv)
    run_daemon(headless=headless)

if __name__ == "__main__":
    main()
