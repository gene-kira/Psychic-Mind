# === AUTO-ELEVATION CHECK ===
import ctypes
import os
import sys

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

#!/usr/bin/env python3
"""
Codex Sentinel – Unified Autonomous Swarm Port Guardian (CodexModified + self-exclusion)

- Auto-elevated (Admin)
- Port baseline learning + enforcement
- Auto-flip from learning to enforcement
- Firewall blocking of unauthorized ports
- Supervisor wrappers stored in CodexModified/ per script directory
- Wrapper-based crash-loop restart + aggressive source auto-patching
- Smart health-check (10–20s + port binding + behavioral signals)
- Process-specific matching (name + path)
- Self-exclusion (daemon never wraps itself or Codex files)
- Multi-node swarm sync with encrypted payloads
- Signed swarm commands (strict allow-list)
- GPU telemetry (NVIDIA)
- Heuristic + ML-style anomaly detection (optional model file)
- ML-style threat scoring
- Basic sandboxing (priority + affinity)
- GUI cockpit + HUD overlay
- Daemon/headless mode
"""

import psutil
import time
import json
import threading
import queue
import subprocess
import logging
import socket
import base64
import hashlib
import hmac
from datetime import datetime, timedelta
from statistics import mean, pstdev
import re

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None

try:
    import joblib
except ImportError:
    joblib = None

# =========================
# CONFIGURATION
# =========================

CONFIG_FILE = "port_enforcer_config.json"
ML_MODEL_FILE = "ml_anomaly_model.pkl"
LEARN_MODE = True
SCAN_INTERVAL = 1.0
FIREWALL_BLOCK_ENABLED = True
LOG_FILE = "port_enforcer.log"

STABLE_WINDOW_SECONDS = 600

WEIGHT_UNKNOWN_PROCESS = 40
WEIGHT_UNAUTHORIZED_PORT = 40
WEIGHT_HIGH_PORT = 10
WEIGHT_ANOMALY = 25
WEIGHT_HISTORY = 15
THREAT_ALERT_THRESHOLD = 60

SWARM_BROADCAST_PORT = 50050
SWARM_NODE_ID = socket.gethostname()
SWARM_KEY = "KILLER666_SWARM_KEY"
SWARM_COMMAND_KEY = "KILLER666_CMD_KEY"

GPU_TELEMETRY_INTERVAL = 5

USER_STARTUP_DIR = os.path.expanduser(
    r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
)
SYSTEM_STARTUP_DIR = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Startup"

SANDBOX_CPU_PERCENT_LIMIT = 50
SANDBOX_AFFINITY_LIMIT = 2

WRAPPER_SUFFIX = "_wrapper.py"
MOD_DIR_NAME = "CodexModified"

WRAPPER_CRASH_WINDOW_SECONDS = 60
WRAPPER_CRASH_MAX_RESTARTS = 5

HEALTH_MIN_SECONDS = 10
HEALTH_MAX_SECONDS = 20

# =========================
# LOGGING
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

# =========================
# STATE
# =========================

lock = threading.Lock()

state = {
    "allowed_ports": {},
    "learn_mode": LEARN_MODE,
    "last_scan": None,
    "events": [],
    "last_new_port_time": None,
    "gpu_util": None,
    "swarm_last_heartbeat": None,
    "swarm_peers": {},
    "port_usage_history": {},
    "false_positive_counts": {},
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
    # Skip the daemon itself and any codex-related files
    return this == target or "codex" in os.path.basename(target)

# =========================
# SWARM ENCRYPTION
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
            console.info(f"[SWARM CMD] Mode set to {value}")
            save_config()
    elif ctype == "BLOCK_PORT":
        port = int(cmd.get("port", 0))
        proto = cmd.get("protocol", "TCP")
        if port > 0:
            firewall_block_port(port, proto)
            console.info(f"[SWARM CMD] Blocked {proto} port {port}")
    elif ctype == "UNBLOCK_PORT":
        port = int(cmd.get("port", 0))
        proto = cmd.get("protocol", "TCP")
        if port > 0:
            firewall_unblock_port(port, proto)
            console.info(f"[SWARM CMD] Unblocked {proto} port {port}")

# =========================
# CONFIG
# =========================

def load_config():
    global state
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            with lock:
                state["allowed_ports"] = data.get("allowed_ports", {})
                state["learn_mode"] = data.get("learn_mode", LEARN_MODE)
            console.info("Loaded config.")
        except Exception as e:
            console.error("Failed to load config: %s", e)
    with lock:
        state["last_new_port_time"] = datetime.utcnow()

def save_config():
    data = {
        "allowed_ports": state["allowed_ports"],
        "learn_mode": state["learn_mode"],
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
        console.info("Config saved.")
    except Exception as e:
        console.error("Failed to save config: %s", e)

# =========================
# ML MODEL
# =========================

def load_ml_model():
    global ml_model
    if joblib is None:
        console.info("joblib not available, ML anomaly detection disabled.")
        return
    if not os.path.exists(ML_MODEL_FILE):
        console.info("ML model file not found, ML anomaly detection disabled.")
        return
    try:
        ml_model = joblib.load(ML_MODEL_FILE)
        console.info("ML anomaly model loaded.")
    except Exception as e:
        console.error(f"Failed to load ML model: {e}")
        ml_model = None

def extract_features_for_ml(conn, process_name: str) -> list[float]:
    port = conn.laddr.port if conn.laddr else 0
    proto = 1.0 if conn.type == psutil.SOCK_STREAM else 0.0
    status = 0.0
    if hasattr(conn, "status"):
        status = float(hash(conn.status) % 1000) / 1000.0
    plen = float(len(process_name)) / 64.0
    return [float(port) / 65535.0, proto, status, plen]

def ml_anomaly_flag(conn, process_name: str) -> bool:
    if ml_model is None:
        return False
    try:
        feats = extract_features_for_ml(conn, process_name)
        if hasattr(ml_model, "decision_function"):
            score = ml_model.decision_function([feats])[0]
            return score < 0
        elif hasattr(ml_model, "predict"):
            pred = ml_model.predict([feats])[0]
            return int(pred) == -1
    except Exception as e:
        logging.error(f"ML anomaly check failed: {e}")
    return False

# =========================
# FIREWALL
# =========================

def is_windows():
    return os.name == "nt"

def firewall_block_port(port, protocol="TCP"):
    if not FIREWALL_BLOCK_ENABLED or not is_windows():
        return
    rule_name = f"PortEnforcer_Block_{protocol}_{port}"
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule",
        f"name={rule_name}",
        "dir=in",
        "action=block",
        f"protocol={protocol}",
        f"localport={port}"
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        logging.info(f"Firewall rule added for {protocol} port {port}")
    except Exception as e:
        logging.error(f"Failed to add firewall rule: {e}")

def firewall_unblock_port(port, protocol="TCP"):
    if not FIREWALL_BLOCK_ENABLED or not is_windows():
        return
    rule_name = f"PortEnforcer_Block_{protocol}_{port}"
    cmd = [
        "netsh", "advfirewall", "firewall", "delete", "rule",
        f"name={rule_name}",
        f"protocol={protocol}",
        f"localport={port}"
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        logging.info(f"Firewall rule removed for {protocol} port {port}")
    except Exception as e:
        logging.error(f"Failed to remove firewall rule: {e}")

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
        if SANDBOX_AFFINITY_LIMIT > 0:
            try:
                cores = list(range(SANDBOX_AFFINITY_LIMIT))
                p.cpu_affinity(cores)
            except Exception:
                pass
        if is_windows():
            try:
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            except Exception:
                pass
    except Exception as e:
        logging.error(f"Sandbox failed for PID {pid}: {e}")

def kill_process(pid, reason, port=None):
    try:
        p = psutil.Process(pid)
        name = p.name()
        msg = f"Killing PID {pid} ({name}) - {reason}"
        if port:
            msg += f" on port {port}"
        logging.warning(msg)
        console.warning(msg)
        p.terminate()
    except Exception as e:
        logging.error(f"Could not kill PID {pid}: {e}")

# =========================
# ANOMALY + THREAT
# =========================

def update_port_usage_stats(process, port, count):
    key = (process, port)
    with lock:
        hist = state["port_usage_history"].setdefault(key, [])
        hist.append(count)
        if len(hist) > 100:
            hist.pop(0)

def is_anomalous(process, port, count) -> bool:
    key = (process, port)
    with lock:
        hist = state["port_usage_history"].get(key, [])
    if len(hist) < 10:
        return False
    m = mean(hist)
    s = pstdev(hist) or 1.0
    return abs(count - m) > 3 * s

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

def record_event(event):
    event["timestamp"] = datetime.utcnow().isoformat()
    event["threat_score"] = ml_style_threat_score(event)
    with lock:
        state["events"].append(event)
        state["events"] = state["events"][-200:]
    alert_queue.put(event)
    gui_update_queue.put(("event", event))

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
            console.info("Learning: updated allowed ports.")
    if updated:
        with lock:
            state["last_new_port_time"] = datetime.utcnow()
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
        console.info("Auto-flip: Learning → Enforcement")
        save_config()

# =========================
# ENFORCEMENT
# =========================

def enforce_rules(connections):
    with lock:
        allowed_ports = state["allowed_ports"].copy()
        learn_mode = state["learn_mode"]

    for conn in connections:
        pid = conn.pid
        if pid is None or not conn.laddr:
            continue

        port = conn.laddr.port
        proto = "TCP" if conn.type == psutil.SOCK_STREAM else "UDP"

        try:
            name = psutil.Process(pid).name().lower()
        except Exception:
            continue

        unknown_process = name not in allowed_ports

        if not learn_mode:
            if unknown_process:
                reason = "Unknown process using port"
                event = {
                    "pid": pid,
                    "process": name,
                    "port": port,
                    "protocol": proto,
                    "unknown_process": True,
                    "unauthorized_port": False,
                }
                record_event(event)
                kill_process(pid, reason, port)
                firewall_block_port(port, proto)
                continue

            if port not in allowed_ports.get(name, []):
                reason = "Unauthorized port"
                event = {
                    "pid": pid,
                    "process": name,
                    "port": port,
                    "protocol": proto,
                    "unknown_process": False,
                    "unauthorized_port": True,
                }
                record_event(event)
                kill_process(pid, reason, port)
                firewall_block_port(port, proto)
                continue

# =========================
# SCANNER
# =========================

def scanner_loop():
    console.info("Scanner thread started.")
    while True:
        try:
            connections = psutil.net_connections(kind="inet")
        except Exception as e:
            logging.error(f"Failed to get connections: {e}")
            time.sleep(SCAN_INTERVAL)
            continue

        with lock:
            state["last_scan"] = datetime.utcnow().isoformat()
            learn_mode = state["learn_mode"]

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
                name = psutil.Process(pid).name().lower()
            except Exception:
                name = "unknown"
            proto = "TCP" if conn.type == psutil.SOCK_STREAM else "UDP"
            port = conn.laddr.port
            shared = port_counts.get(port, 0) > 1
            count = port_counts.get(port, 0)
            update_port_usage_stats(name, port, count)

            heuristic_anomaly = is_anomalous(name, port, count)
            ml_flag = ml_anomaly_flag(conn, name)
            anomaly = heuristic_anomaly or ml_flag

            if anomaly:
                event = {
                    "pid": pid,
                    "process": name,
                    "port": port,
                    "protocol": proto,
                    "unknown_process": False,
                    "unauthorized_port": False,
                    "anomaly": True,
                    "reason": "Anomalous port usage pattern (ML/heuristic)",
                }
                record_event(event)
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
        time.sleep(SCAN_INTERVAL)

# =========================
# ALERT HANDLER
# =========================

def alert_handler_loop():
    console.info("Alert handler started.")
    while True:
        event = alert_queue.get()
        if event is None:
            break
        if event["threat_score"] >= THREAT_ALERT_THRESHOLD:
            console.warning(
                f"[ALERT] Score {event['threat_score']} - {event['process']} "
                f"PID {event['pid']} PORT {event['port']} REASON: {event.get('reason','')}"
            )

# =========================
# SUPERVISOR WRAPPERS (CodexModified)
# =========================

def ensure_wrapper_for_script(script_path: str) -> str | None:
    """
    Create a supervisor wrapper in CodexModified/ that:
    - Receives a port from the main daemon
    - Sets AUTO_PORT env
    - Launches the real script as a child
    - Performs smart health-check (10–20s + port binding + behavior)
    - Detects crash-loops
    - Aggressively auto-patches ports in the original script if needed
    """
    # Never wrap the daemon or any codex-related file
    if is_self_script(script_path):
        return None

    script_dir = os.path.dirname(script_path)
    mod_dir = os.path.join(script_dir, MOD_DIR_NAME)

    if not os.path.isdir(mod_dir):
        try:
            os.makedirs(mod_dir, exist_ok=True)
            console.info(f"[WRAPPER] Created CodexModified folder: {mod_dir}")
        except Exception as e:
            console.error(f"[WRAPPER] Failed to create CodexModified folder: {e}")
            mod_dir = script_dir  # fallback

    base = os.path.basename(script_path)
    name, ext = os.path.splitext(base)
    wrapper_name = f"{name}{WRAPPER_SUFFIX}"
    wrapper_path = os.path.join(mod_dir, wrapper_name)

    if not os.path.exists(wrapper_path):
        console.info(f"[WRAPPER] Creating supervisor wrapper for {script_path}")
        wrapper_code = f'''import os
import sys
import subprocess
import time
from datetime import datetime, timedelta
import re
import psutil

CRASH_WINDOW_SECONDS = {WRAPPER_CRASH_WINDOW_SECONDS}
CRASH_MAX_RESTARTS = {WRAPPER_CRASH_MAX_RESTARTS}
HEALTH_MIN_SECONDS = {HEALTH_MIN_SECONDS}
HEALTH_MAX_SECONDS = {HEALTH_MAX_SECONDS}

SCRIPT_PATH = r"{script_path}"

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

    os.environ["AUTO_PORT"] = port_str

    crash_times = []

    while True:
        try:
            p = subprocess.Popen([sys.executable, SCRIPT_PATH], env=os.environ)
        except Exception as e:
            print(f"[WRAPPER] Failed to launch child: {{e}}")
            time.sleep(5)
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
                time.sleep(2)
                continue
            else:
                print("[WRAPPER] Patch failed or no changes, stopping wrapper.")
                break

        print("[WRAPPER] Child crashed or unhealthy, restarting in 2s...")
        time.sleep(2)

if __name__ == "__main__":
    main()
'''
        try:
            with open(wrapper_path, "w", encoding="utf-8") as f:
                f.write(wrapper_code)
        except Exception as e:
            console.error(f"[WRAPPER] Failed to create wrapper for {script_path}: {e}")
    return wrapper_path

# =========================
# WATCHDOG
# =========================

def discover_python_startup_programs():
    programs = []
    for folder in [USER_STARTUP_DIR, SYSTEM_STARTUP_DIR]:
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if not file.lower().endswith(".py"):
                continue

            full_path = os.path.join(folder, file)

            # Skip CodexModified folder
            if MOD_DIR_NAME.lower() in full_path.lower():
                continue

            # Skip wrappers
            if full_path.lower().endswith(WRAPPER_SUFFIX):
                continue

            # Skip daemon itself and any codex-related files
            if is_self_script(full_path):
                continue

            wrapper_path = ensure_wrapper_for_script(full_path)
            if not wrapper_path:
                continue

            programs.append({
                "id": wrapper_path.lower(),
                "name": "python.exe",
                "match_path": wrapper_path.lower(),
                "command_template": [sys.executable, wrapper_path, "{PORT}"],
                "cwd": os.path.dirname(wrapper_path)
            })
    return programs

MANAGED_PROGRAMS = [p for p in discover_python_startup_programs() if p]

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

def launch_program(entry):
    port = get_free_port()
    cmd = [arg.replace("{PORT}", str(port)) for arg in entry["command_template"]]
    try:
        subprocess.Popen(cmd, cwd=entry["cwd"])
        console.info(f"[WATCHDOG] Launched wrapper: {cmd} (port {port})")
    except Exception as e:
        console.error(f"[WATCHDOG] Failed to launch {cmd}: {e}")

def watchdog_loop():
    console.info("Watchdog thread started.")
    while True:
        for entry in MANAGED_PROGRAMS:
            if not is_process_running_for_entry(entry):
                console.warning(f"[WATCHDOG] Wrapper not running: {entry['match_path']}. Relaunching...")
                launch_program(entry)
        time.sleep(5)

# =========================
# SWARM
# =========================

def swarm_heartbeat_loop():
    console.info("Swarm heartbeat thread started.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while True:
        try:
            with lock:
                payload = {
                    "node": SWARM_NODE_ID,
                    "time": datetime.utcnow().isoformat(),
                    "gpu_util": state["gpu_util"],
                    "mode": "learning" if state["learn_mode"] else "enforcement",
                    "allowed_ports": state["allowed_ports"],
                    "commands": []
                }
            data = encrypt_swarm_payload(payload)
            sock.sendto(data, ("255.255.255.255", SWARM_BROADCAST_PORT))
            with lock:
                state["swarm_last_heartbeat"] = payload["time"]
        except Exception as e:
            logging.error(f"Swarm heartbeat error: {e}")
        time.sleep(5)

def swarm_listener_loop():
    console.info("Swarm listener thread started.")
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
            if not node or node == SWARM_NODE_ID:
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
                    console.warning("[SWARM CMD] Rejected invalid or unsigned command.")
        except Exception as e:
            logging.error(f"Swarm listener error: {e}")

# =========================
# GPU TELEMETRY
# =========================

def gpu_telemetry_loop():
    console.info("GPU telemetry thread started.")
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
# GUI + HUD
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

        self.hud = tk.Toplevel(root)
        self.hud.title("HUD")
        self.hud.attributes("-topmost", True)
        self.hud.geometry("260x100+20+20")
        self.hud.resizable(False, False)
        self.hud_label = tk.Label(self.hud, text="HUD", font=("Consolas", 9))
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

        self.status_label.config(
            text=f"Mode={mode} | LastScan={last_scan} | {gpu_text}"
        )
        self.hud_label.config(
            text=f"Mode: {mode}\n{gpu_text}\n{swarm_text}"
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
    load_config()
    load_ml_model()

    threading.Thread(target=scanner_loop, daemon=True).start()
    threading.Thread(target=alert_handler_loop, daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    threading.Thread(target=swarm_heartbeat_loop, daemon=True).start()
    threading.Thread(target=swarm_listener_loop, daemon=True).start()
    threading.Thread(target=gpu_telemetry_loop, daemon=True).start()

    if headless or tk is None:
        console.info("Running headless daemon.")
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
