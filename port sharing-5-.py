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
Unified Port Enforcement + Swarm + HUD System

Features:
- Auto-elevated (Admin)
- Port sharing allowed (no killing just for sharing)
- Auto-learning allowed ports per program
- Auto-flip from learning to enforcement after 10 minutes of no new ports
- Enforcement (kill + firewall block) for unknown/unauthorized ports
- Auto-port reassignment for managed programs using {PORT} placeholder
- Crash-loop detection for managed programs (rate-limited restarts)
- Process-specific matching (name + command line path)
- Multi-node swarm sync (UDP heartbeat broadcast)
- GPU-accelerated telemetry (via nvidia-smi polling)
- GUI cockpit (Tkinter) with "Shared" column
- HUD overlay (always-on-top compact status window)
- Daemon-style loop + optional startup registration (Windows)
- Universal watchdog: auto-restart ALL Python scripts in BOTH Startup folders
"""

import psutil
import time
import json
import threading
import queue
import subprocess
import logging
import socket
from datetime import datetime, timedelta

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None

# =========================
# CONFIGURATION
# =========================

CONFIG_FILE = "port_enforcer_config.json"
LEARN_MODE = True
SCAN_INTERVAL = 1.0
FIREWALL_BLOCK_ENABLED = True
LOG_FILE = "port_enforcer.log"
STARTUP_TASK_NAME = "PortEnforcerDaemon"

STABLE_WINDOW_SECONDS = 600

WEIGHT_UNKNOWN_PROCESS = 40
WEIGHT_UNAUTHORIZED_PORT = 40
WEIGHT_HIGH_PORT = 10
THREAT_ALERT_THRESHOLD = 50

CRASH_WINDOW_SECONDS = 60
CRASH_MAX_RESTARTS = 5

SWARM_BROADCAST_PORT = 50050
SWARM_NODE_ID = socket.gethostname()

GPU_TELEMETRY_INTERVAL = 5

USER_STARTUP_DIR = os.path.expanduser(
    r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
)
SYSTEM_STARTUP_DIR = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Startup"

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
}

alert_queue = queue.Queue()
gui_update_queue = queue.Queue()

crash_history = {}  # key: entry_id, value: list[datetime]

# =========================
# CONFIG FILE HANDLING
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
# FIREWALL CONTROL
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

# =========================
# PORT UTILITIES
# =========================

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

# =========================
# PROCESS CONTROL
# =========================

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
# THREAT SCORING
# =========================

def compute_threat_score(event):
    score = 0
    if event.get("unknown_process"):
        score += WEIGHT_UNKNOWN_PROCESS
    if event.get("unauthorized_port"):
        score += WEIGHT_UNAUTHORIZED_PORT
    if event.get("port") and event["port"] >= 49152:
        score += WEIGHT_HIGH_PORT
    return score

def record_event(event):
    event["timestamp"] = datetime.utcnow().isoformat()
    event["threat_score"] = compute_threat_score(event)
    with lock:
        state["events"].append(event)
        state["events"] = state["events"][-200:]
    alert_queue.put(event)
    gui_update_queue.put(("event", event))

# =========================
# AUTO-LEARNING
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
                record_event({
                    "pid": pid,
                    "process": name,
                    "port": port,
                    "protocol": proto,
                    "unknown_process": True,
                    "unauthorized_port": False,
                    "reason": reason,
                })
                kill_process(pid, reason, port)
                firewall_block_port(port, proto)
                continue

            if port not in allowed_ports.get(name, []):
                reason = "Unauthorized port"
                record_event({
                    "pid": pid,
                    "process": name,
                    "port": port,
                    "protocol": proto,
                    "unknown_process": False,
                    "unauthorized_port": True,
                    "reason": reason,
                })
                kill_process(pid, reason, port)
                firewall_block_port(port, proto)
                continue

# =========================
# SCANNER THREAD
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
# ALERT HANDLER THREAD
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
                f"PID {event['pid']} PORT {event['port']} REASON: {event['reason']}"
            )

# =========================
# MANAGED PROGRAM DISCOVERY (WATCHDOG)
# =========================

def discover_python_startup_programs():
    programs = []
    for folder in [USER_STARTUP_DIR, SYSTEM_STARTUP_DIR]:
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(".py"):
                full_path = os.path.join(folder, file)
                programs.append({
                    "id": full_path.lower(),
                    "name": "python.exe",
                    "match_path": full_path.lower(),
                    "command_template": [sys.executable, full_path, "{PORT}"],
                    "cwd": folder
                })
    return programs

MANAGED_PROGRAMS = discover_python_startup_programs()

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

def record_crash(entry_id):
    now = datetime.utcnow()
    history = crash_history.setdefault(entry_id, [])
    history.append(now)
    cutoff = now - timedelta(seconds=CRASH_WINDOW_SECONDS)
    crash_history[entry_id] = [t for t in history if t >= cutoff]

def can_restart(entry_id):
    history = crash_history.get(entry_id, [])
    return len(history) < CRASH_MAX_RESTARTS

def launch_program(entry):
    entry_id = entry["id"]
    if not can_restart(entry_id):
        console.error(f"[WATCHDOG] Crash-loop detected for {entry_id}, suppressing restart.")
        return
    port = get_free_port()
    cmd = [arg.replace("{PORT}", str(port)) for arg in entry["command_template"]]
    try:
        subprocess.Popen(cmd, cwd=entry["cwd"])
        record_crash(entry_id)
        console.info(f"[WATCHDOG] Launched: {cmd} (port {port})")
    except Exception as e:
        console.error(f"[WATCHDOG] Failed to launch {cmd}: {e}")

def watchdog_loop():
    console.info("Watchdog thread started.")
    while True:
        for entry in MANAGED_PROGRAMS:
            if not is_process_running_for_entry(entry):
                console.warning(f"[WATCHDOG] Target not running: {entry['match_path']}. Relaunching...")
                launch_program(entry)
        time.sleep(5)

# =========================
# SWARM SYNC (UDP HEARTBEAT)
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
                }
            data = json.dumps(payload).encode("utf-8")
            sock.sendto(data, ("255.255.255.255", SWARM_BROADCAST_PORT))
            with lock:
                state["swarm_last_heartbeat"] = payload["time"]
        except Exception as e:
            logging.error(f"Swarm heartbeat error: {e}")
        time.sleep(5)

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
        self.hud.geometry("260x90+20+20")
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
                        text=f"Event: {payload['reason']} (score {payload['threat_score']})"
                    )
        except queue.Empty:
            pass

        with lock:
            last_scan = state["last_scan"]
            learn_mode = state["learn_mode"]
            gpu_util = state["gpu_util"]
            swarm_last = state["swarm_last_heartbeat"]

        mode = "Learning" if learn_mode else "Enforcement"
        gpu_text = f"GPU {gpu_util}%" if gpu_util is not None else "GPU N/A"
        swarm_text = f"Swarm {swarm_last}" if swarm_last else "Swarm N/A"

        self.status_label.config(
            text=f"Mode={mode} | LastScan={last_scan} | {gpu_text}"
        )
        self.hud_label.config(
            text=f"Mode: {mode}\nGPU: {gpu_text}\nSwarm: {swarm_text}"
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
# DAEMON MODE
# =========================

def run_daemon(headless=True):
    load_config()

    threading.Thread(target=scanner_loop, daemon=True).start()
    threading.Thread(target=alert_handler_loop, daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    threading.Thread(target=swarm_heartbeat_loop, daemon=True).start()
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
