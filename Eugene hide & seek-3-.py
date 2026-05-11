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

# =========================
# AUTOLOAD LIBS
# =========================

REQUIRED_LIBS = ["pywin32", "psutil"]

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

try:
    import win32pipe
    import win32file
    import pywintypes
    PIPE_AVAILABLE = True
except ImportError:
    PIPE_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import winreg  # stdlib, Windows only

# =========================
# FILE PATHS
# =========================

STATE_FILE   = "sentinel_state.json"
TIMELINE_FILE = "sentinel_timeline.json"
THREATS_FILE  = "sentinel_threats.json"
NODES_FILE    = "sentinel_nodes.json"
LOGS_FILE     = "sentinel_logs.json"
SWARM_FILE    = "sentinel_swarm.json"
PERSONA_FILE  = "sentinel_persona.json"

PIPE_NAME = r"\\.\pipe\sentinel_bus"

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
# LOW-LEVEL HOOKS (REAL INTROSPECTION)
# =========================

def scan_processes():
    """
    Real process scan using psutil.
    Returns list of dicts: {pid, name, cpu, suspicious}
    """
    results = []
    if not PSUTIL_AVAILABLE:
        return results
    for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "exe"]):
        info = p.info
        pid = info.get("pid")
        name = info.get("name") or "?"
        cpu = info.get("cpu_percent") or 0.0
        exe = info.get("exe") or ""
        suspicious = False
        # simple heuristic: high CPU and unknown path
        if cpu > 50.0 and "Windows" not in exe:
            suspicious = True
        results.append({
            "pid": pid,
            "name": name,
            "cpu": cpu,
            "exe": exe,
            "suspicious": suspicious
        })
    return results

def probe_network():
    """
    Real local network probing using psutil.
    Returns list of dicts: {laddr, raddr, status, pid}
    """
    results = []
    if not PSUTIL_AVAILABLE:
        return results
    try:
        conns = psutil.net_connections(kind="inet")
    except Exception:
        return results
    for c in conns:
        laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else ""
        raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else ""
        results.append({
            "laddr": laddr,
            "raddr": raddr,
            "status": c.status,
            "pid": c.pid
        })
    return results

def check_registry():
    """
    Safe registry checks: reads a few Run keys.
    Returns list of dicts: {key, value_name, data}
    """
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

def listen_etw_stub():
    """
    ETW listener placeholder.
    Real ETW requires specialized libs (e.g., krabsetw, etwpy).
    Here we just return an empty list.
    """
    return []

def compute_anomaly_score(proc_list, net_list, reg_list, etw_events, lockdown, autonomous):
    """
    Simple anomaly scoring:
    - suspicious processes
    - many outbound connections
    - many Run entries
    """
    score = 0
    suspicious_procs = sum(1 for p in proc_list if p.get("suspicious"))
    outbound = sum(1 for n in net_list if n.get("raddr"))
    run_entries = len(reg_list)

    score += suspicious_procs * 10
    score += min(outbound, 20) * 2
    score += min(run_entries, 50) * 1

    if lockdown:
        score *= 0.7  # we assume lockdown reduces effective risk
    if not autonomous:
        score *= 0.9  # operator watching

    return int(score)

# =========================
# MINI-SENTINEL HOOKS (USE REAL DATA ABOVE)
# =========================

def hook_collect_timeline(lockdown: bool, autonomous: bool,
                          proc_list, net_list, reg_list, etw_events, anomaly_score):
    now = datetime.utcnow().isoformat() + "Z"
    events = [
        {"time": now, "event": "Heartbeat", "severity": "info"},
        {"time": now, "event": f"Processes scanned: {len(proc_list)}", "severity": "info"},
        {"time": now, "event": f"Network conns: {len(net_list)}", "severity": "info"},
        {"time": now, "event": f"Run entries: {len(reg_list)}", "severity": "info"},
        {"time": now, "event": f"Anomaly score: {anomaly_score}", "severity": "warn" if anomaly_score > 50 else "info"},
    ]
    if lockdown:
        events.append({"time": now, "event": "Lockdown active", "severity": "warn"})
    if not autonomous:
        events.append({"time": now, "event": "Manual override", "severity": "info"})
    return events

def hook_collect_threat_matrix(lockdown: bool, autonomous: bool, anomaly_score: int):
    # Single-node example; you can expand to multi-node later
    return [
        {"node": "LocalNode", "score": anomaly_score}
    ]

def hook_collect_node_sync(lockdown: bool, autonomous: bool):
    # Placeholder: single node, always in-sync
    return [
        {"node": "LocalNode", "status": "in-sync" if autonomous else "manual"}
    ]

def hook_collect_logs(lockdown: bool, autonomous: bool, anomaly_score: int):
    now = datetime.utcnow().isoformat() + "Z"
    lines = [f"{now} :: mini-sentinel heartbeat :: anomaly_score={anomaly_score}"]
    if lockdown:
        lines.append(f"{now} :: lockdown mode: aggressive filters enabled")
    if not autonomous:
        lines.append(f"{now} :: manual override: operator in control")
    return lines

def hook_collect_swarm(lockdown: bool, autonomous: bool, anomaly_score: int):
    return {
        "nodes": 1,
        "cohesion": 0.9 if autonomous else 0.7,
        "drift": min(anomaly_score / 100.0, 1.0),
    }

def hook_collect_persona(lockdown: bool, autonomous: bool, anomaly_score: int):
    mode = "Guardian" if lockdown or anomaly_score > 60 else "Observer"
    return {
        "mode": mode,
        "autonomous": autonomous,
    }

def hook_emit_events(lockdown: bool, autonomous: bool, anomaly_score: int):
    now = datetime.utcnow().isoformat() + "Z"
    events = [f"{now} :: EVENT :: heartbeat :: anomaly_score={anomaly_score}"]
    if anomaly_score > 60:
        events.append(f"{now} :: EVENT :: high_anomaly")
    if lockdown:
        events.append(f"{now} :: EVENT :: lockdown_enforced")
    if not autonomous:
        events.append(f"{now} :: EVENT :: manual_override")
    return events

def hook_behavior_on_lockdown(lockdown: bool, autonomous: bool):
    """
    Here you’d actually change behavior:
    - skip risky operations when lockdown is ON
    - slow down scans when manual override is ON
    For now, we just no-op.
    """
    return

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
                    lockdown   = state.get("lockdown", False)
                    autonomous = state.get("autonomous", True)

                    # local introspection
                    proc_list = scan_processes()
                    net_list  = probe_network()
                    reg_list  = check_registry()
                    etw_events = listen_etw_stub()
                    anomaly_score = compute_anomaly_score(proc_list, net_list, reg_list, etw_events,
                                                          lockdown, autonomous)

                    events = hook_emit_events(lockdown, autonomous, anomaly_score)
                    for ev in events:
                        data = (ev + "\n").encode()
                        try:
                            win32file.WriteFile(handle, data)
                        except pywintypes.error:
                            break
                    time.sleep(1 if autonomous else 2)
                win32file.CloseHandle(handle)
            except pywintypes.error:
                time.sleep(1)

class OrganismLoop(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        while self.running:
            state = load_state()
            lockdown   = state.get("lockdown", False)
            autonomous = state.get("autonomous", True)

            hook_behavior_on_lockdown(lockdown, autonomous)

            proc_list = scan_processes()
            net_list  = probe_network()
            reg_list  = check_registry()
            etw_events = listen_etw_stub()
            anomaly_score = compute_anomaly_score(proc_list, net_list, reg_list, etw_events,
                                                  lockdown, autonomous)

            timeline = hook_collect_timeline(lockdown, autonomous,
                                             proc_list, net_list, reg_list, etw_events, anomaly_score)
            threats  = hook_collect_threat_matrix(lockdown, autonomous, anomaly_score)
            nodes    = hook_collect_node_sync(lockdown, autonomous)
            logs     = hook_collect_logs(lockdown, autonomous, anomaly_score)
            swarm    = hook_collect_swarm(lockdown, autonomous, anomaly_score)
            persona  = hook_collect_persona(lockdown, autonomous, anomaly_score)

            save_json(TIMELINE_FILE, timeline)
            save_json(THREATS_FILE, threats)
            save_json(NODES_FILE,   nodes)
            save_json(LOGS_FILE,    logs)
            save_json(SWARM_FILE,   swarm)
            save_json(PERSONA_FILE, persona)

            time.sleep(1 if autonomous else 2)

# =========================
# COCKPIT SIDE
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
        self.event_bus = []

    def refresh_from_files(self):
        self.state        = load_state()
        self.timeline     = load_json_list(TIMELINE_FILE, [])
        self.threat_matrix= load_json_list(THREATS_FILE, [])
        self.node_sync    = load_json_list(NODES_FILE, [])
        self.logs         = load_json_list(LOGS_FILE, [])
        self.swarm_sim    = load_json_dict(SWARM_FILE, {"nodes": 0})
        self.persona      = load_json_dict(PERSONA_FILE, {"mode": "Unknown", "autonomous": True})

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

class OperatorCockpit:
    REFRESH_MS = 1000

    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel Operator Cockpit")

        self.sentinel = SentinelState()

        self.event_listener = EventBusListener(self.sentinel)
        self.event_listener.start()

        self._build_header()
        self._build_tabs()
        self._refresh_all()

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
        self.event_bus_frame  = tk.Frame(self.notebook)

        self.notebook.add(self.timeline_frame,  text="Timeline")
        self.notebook.add(self.threat_frame,    text="Threat Matrix")
        self.notebook.add(self.node_sync_frame, text="Node Sync")
        self.notebook.add(self.logs_frame,      text="Logs")
        self.notebook.add(self.swarm_frame,     text="Swarm")
        self.notebook.add(self.persona_frame,   text="Persona")
        self.notebook.add(self.event_bus_frame, text="Event Bus")

        self.timeline_list = tk.Listbox(self.timeline_frame, font=("Consolas", 9))
        self.timeline_list.pack(fill="both", expand=True)

        self.threat_list = tk.Listbox(self.threat_frame, font=("Consolas", 9))
        self.threat_list.pack(fill="both", expand=True)

        self.node_sync_list = tk.Listbox(self.node_sync_frame, font=("Consolas", 9))
        self.node_sync_list.pack(fill="both", expand=True)

        self.logs_text = tk.Text(self.logs_frame, font=("Consolas", 9))
        self.logs_text.pack(fill="both", expand=True)

        self.swarm_canvas = tk.Canvas(self.swarm_frame, bg="black")
        self.swarm_canvas.pack(fill="both", expand=True)

        self.persona_label = tk.Label(self.persona_frame, font=("Consolas", 11))
        self.persona_label.pack(pady=10)

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
        self._update_event_bus()
        self.root.after(self.REFRESH_MS, self._refresh_all)

    def _update_timeline(self):
        self.timeline_list.delete(0, tk.END)
        for ev in self.sentinel.timeline:
            t = ev.get("time", "?")
            s = ev.get("severity", "info").upper()
            msg = ev.get("event", "")
            self.timeline_list.insert(tk.END, f"{t} :: {s} :: {msg}")

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
        for i in range(nodes):
            x = (i + 1) * w / (nodes + 1)
            y = h / 2
            self.swarm_canvas.create_oval(
                x - 10, y - 10, x + 10, y + 10,
                fill="#33AAFF", outline="#FFFFFF"
            )

    def _update_persona(self):
        mode = self.sentinel.persona.get("mode", "Unknown")
        auto = self.sentinel.persona.get("autonomous", True)
        self.persona_label.config(text=f"Persona Mode: {mode}\nAutonomous: {'ON' if auto else 'OFF'}")

    def _update_event_bus(self):
        self.event_bus_list.delete(0, tk.END)
        for ev in self.sentinel.event_bus:
            self.event_bus_list.insert(tk.END, ev)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    organism = OrganismLoop()
    organism.start()

    pipe_server = EventBusServer()
    pipe_server.start()

    root = tk.Tk()
    cockpit = OperatorCockpit(root)
    root.geometry("1000x650")
    root.mainloop()
