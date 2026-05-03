import subprocess
import sys

# 🔄 Auto-loader (no tkinter here; it's stdlib)
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

# Optional GPU / NumPy stack
try:
    autoload("cupy")
    import cupy as xp
    GPU_ENABLED = True
except Exception:
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

# 🧠 Persistent Brain
BRAIN_FILE = "magicbox_brain.json"
brain_lock = threading.Lock()
brain_state = {
    "trust_config": {},
    "mutation_log": [],
    "swarm_id": None,
    "phantom_history": [],
    "node_id": None,
    "last_anomaly_score": 0.0,
    "remote_nodes": {}
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

# 🔧 Config-Driven Trust Rules (TTL=10 for Telemetry, Phantom, SwarmID)
trust_config = {
    "MAC": {"action": "destroy", "ttl": 86400},
    "IP": {"action": "cloak", "ttl": 86400},
    "Telemetry": {"action": "destroy", "ttl": 10},
    "Phantom": {"action": "destroy", "ttl": 10},
    "SwarmID": {"action": "preserve", "ttl": 10}
}

# 🧠 Symbolic Memory Routing
def symbolic_route(data):
    sigil = uuid.uuid4().hex[:8]
    return f"{sigil}:{data}"

# 🧬 Tier-6 Mutation Engine
mutation_log = []
destruction_queue = []
mutation_lock = threading.Lock()

event_history = []
event_history_lock = threading.Lock()
MAX_HISTORY = 512

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
    return entry

# 🧾 Mutation Trail Logger
def update_log():
    log_text.delete(1.0, tk.END)
    with mutation_lock:
        for entry in mutation_log[-10:]:
            log_text.insert(tk.END, f"{entry}\n")

# ⏳ Self-Destruct Logic
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

# 🧩 Trust Engine Handler
def handle_data(tag, value):
    rule = trust_config.get(tag, {})
    action = rule.get("action")
    ttl = rule.get("ttl")

    if action == "destroy":
        record_event("trust", f"{tag}: {value}", severity="info", tags=["destroy"])
        schedule_destruction(tag, ttl)
    elif action == "cloak":
        record_event("trust", f"{tag}: [CLOAKED]", severity="info", tags=["cloak"])
        schedule_destruction(tag, ttl)
    elif action == "preserve":
        record_event("trust", f"{tag}: {value}", severity="info", tags=["preserve"])

# 🦎 Real-Time Detection
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

# ⚔️ Threat Detection + Response (full port sweep)
def threat_scan_and_respond():
    try:
        interesting_ports = {22, 23, 80, 443, 1337, 31337, 6666, 9001}
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
                    if port in interesting_ports:
                        record_event(
                            "port_listen",
                            f"LISTEN {ip}:{port} (interesting)",
                            severity="warn",
                            tags=["port", "interesting"]
                        )
                    else:
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

# 🧬 Self-Rewriting Engine
def self_check():
    try:
        assert callable(threat_scan_and_respond)
        assert callable(update_log)
        assert isinstance(trust_config, dict)
    except Exception as e:
        record_event("self_rewrite", f"Integrity check failed - {e}", severity="crit", tags=["self_rewrite"])
    watchdog_touch("self_check")
    root.after(15000, self_check)

# 🧮 GPU-Accelerated Anomaly Core (placeholder for future Bernoulli/entropy tightening)
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

def anomaly_loop():
    global anomaly_score
    score = compute_anomaly_score()
    with anomaly_lock:
        anomaly_score = score
    with brain_lock:
        brain_state["last_anomaly_score"] = score
    record_event("anomaly", f"Anomaly score updated: {score:.2f}", severity="info")
    update_overlay_from_anomaly(score)
    watchdog_touch("anomaly")
    root.after(8000, anomaly_loop)

# 🌐 Swarm Sync Mesh (UDP Gossip)
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
    payload = json.dumps({
        "node_id": node_id,
        "swarm_id": brain_state.get("swarm_id"),
        "score": score,
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
                    "swarm_id": msg.get("swarm_id"),
                    "ts": msg.get("ts")
                }
        except Exception:
            time.sleep(1.0)

def start_swarm_listener_thread():
    t = threading.Thread(target=swarm_listener, daemon=True)
    t.start()

# 🛡️ Self-Healing Watchdog
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
            root.after(1000, anomaly_loop)
        elif name == "swarm_heartbeat":
            root.after(1000, swarm_heartbeat)
        watchdog_touch(name)
    root.after(10000, watchdog_loop)

# 🧙‍♂️ GUI Setup
root = tk.Tk()
root.title("MagicBox: Chameleon ASI (Tier-6)")
root.geometry("900x800")
root.configure(bg="#1e1e2f")

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=("Consolas", 11), background="#1e1e2f", foreground="#00ffcc")
style.configure("TButton", font=("Segoe UI", 10), padding=5)

mac_var = tk.StringVar()
ip_var = tk.StringVar()
telemetry_var = tk.StringVar()
hallucination_var = tk.StringVar()
swarm_var = tk.StringVar()
gpu_var = tk.StringVar(value=f"GPU: {'ON (CuPy)' if GPU_ENABLED else 'OFF (NumPy)'}")
anomaly_var = tk.StringVar(value="Anomaly: 0.00")

top_frame = ttk.Frame(root)
top_frame.pack(pady=5)

ttk.Label(top_frame, textvariable=mac_var).grid(row=0, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=ip_var).grid(row=1, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=telemetry_var).grid(row=2, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=hallucination_var).grid(row=3, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=swarm_var).grid(row=4, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=gpu_var).grid(row=5, column=0, padx=5, sticky="w")
ttk.Label(top_frame, textvariable=anomaly_var).grid(row=6, column=0, padx=5, sticky="w")

ttk.Label(root, text="🧾 Mutation Trail Log (Last 10)").pack(pady=10)
log_text = tk.Text(root, height=10, width=100, bg="#2e2e3f", fg="#00ffcc", font=("Consolas", 10))
log_text.pack()

ttk.Label(root, text="🛠️ Live Config Panel").pack(pady=10)
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

# 🎛️ Tactical Overlay Canvas (Animated)
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

# 🚀 Autonomous Startup
def autonomous_start():
    load_brain()

    mac = get_real_mac()
    mac_var.set(f"🦎 MAC: {mac}")
    handle_data("MAC", mac)

    local_ip, public_ip = get_real_ip()
    ip_var.set(f"🌐 IP: {local_ip} | {public_ip}")
    handle_data("IP", f"{local_ip} | {public_ip}")

    os_info, browser_fp = get_telemetry()
    telemetry_var.set(f"🧢 Telemetry: {os_info} | {browser_fp}")
    handle_data("Telemetry", f"{os_info} | {browser_fp}")

    swarm_id = get_swarm_id()
    swarm_var.set(f"🔗 Swarm ID: {swarm_id}")
    handle_data("SwarmID", swarm_id)

    phantom = synthesize_phantom()
    hallucination_var.set(f"👻 Phantom: {phantom}")
    handle_data("Phantom", phantom)

    build_config_panel()
    update_log()
    save_brain()

# 🔌 Initialize Swarm Mesh
init_swarm_socket()
start_swarm_listener_thread()

# 🧠 Trigger autonomous startup and defense cycles
root.after(100, autonomous_start)
root.after(5000, check_destruction)
root.after(10000, threat_scan_and_respond)
root.after(12000, anomaly_loop)
root.after(13000, swarm_heartbeat)
root.after(14000, periodic_brain_save)
root.after(15000, self_check)
root.after(16000, watchdog_loop)
root.after(200, overlay_anim_loop)

# 🌀 Start GUI loop
root.mainloop()
