import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import random
import importlib
import subprocess
import sys

# =========================
# AUTOLOADER FOR OPTIONAL LIBS (pywin32 for named pipes)
# =========================

REQUIRED_LIBS = ["pywin32"]  # for win32pipe/win32file

def autoload_libraries():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except Exception:
                pass  # if install fails, we'll just run without pipe support

autoload_libraries()

try:
    import win32file
    import win32pipe
    import pywintypes
    PIPE_AVAILABLE = True
except ImportError:
    PIPE_AVAILABLE = False

# =========================
# STATE FILE (lockdown + autonomous)
# =========================

STATE_FILE = "sentinel_state.json"

def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "lockdown": False,
            "autonomous": True,
            "last_changed": None
        }
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "lockdown": False,
            "autonomous": True,
            "last_changed": None
        }

def save_state(state):
    state["last_changed"] = datetime.utcnow().isoformat() + "Z"
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# =========================
# FILE-BASED TELEMETRY PATHS
# (your organism should write these)
# =========================

TIMELINE_FILE = "sentinel_timeline.json"
THREATS_FILE = "sentinel_threats.json"
NODES_FILE = "sentinel_nodes.json"
LOGS_FILE = "sentinel_logs.json"
SWARM_FILE = "sentinel_swarm.json"
PERSONA_FILE = "sentinel_persona.json"

# =========================
# SENTINEL STATE WRAPPER
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

    # ---- File-based fetchers: your organism writes these JSON files ----

    def _load_json_list(self, path, default=None):
        if default is None:
            default = []
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return default
        except Exception:
            return default

    def _load_json_dict(self, path, default=None):
        if default is None:
            default = {}
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return default
        except Exception:
            return default

    def fetch_timeline_events(self):
        self.timeline = self._load_json_list(TIMELINE_FILE, default=[])

    def fetch_threat_matrix(self):
        self.threat_matrix = self._load_json_list(THREATS_FILE, default=[])

    def fetch_node_sync_state(self):
        self.node_sync = self._load_json_list(NODES_FILE, default=[])

    def fetch_logs(self):
        self.logs = self._load_json_list(LOGS_FILE, default=[])

    def fetch_swarm_sim_state(self):
        self.swarm_sim = self._load_json_dict(SWARM_FILE, default={"nodes": 3})

    def fetch_persona_state(self):
        self.persona = self._load_json_dict(
            PERSONA_FILE,
            default={
                "mode": "Observer",
                "autonomous": self.state.get("autonomous", True),
            },
        )

    # event_bus is filled by the named pipe listener


# =========================
# NAMED PIPE EVENT BUS LISTENER
# =========================

class EventBusListener(threading.Thread):
    """
    Listens to a Windows named pipe: \\.\pipe\sentinel_bus
    Your organism should act as the SERVER and write one line per event.
    This cockpit acts as the CLIENT and reads events.
    """
    def __init__(self, sentinel_state: SentinelState, pipe_name=r"\\.\pipe\sentinel_bus"):
        super().__init__(daemon=True)
        self.sentinel_state = sentinel_state
        self.pipe_name = pipe_name
        self.running = True

    def run(self):
        if not PIPE_AVAILABLE:
            return

        while self.running:
            try:
                # Try to connect to existing pipe (organism must create it)
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
                                # keep only last N events
                                self.sentinel_state.event_bus = self.sentinel_state.event_bus[-500:]
                    except pywintypes.error:
                        break  # pipe broken or closed

                win32file.CloseHandle(handle)

            except pywintypes.error:
                # Pipe not available yet; wait and retry
                time.sleep(1)


# =========================
# OPERATOR COCKPIT GUI
# =========================

class OperatorCockpit:
    REFRESH_MS = 1000

    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel Operator Cockpit")

        self.sentinel = SentinelState()

        # Start event bus listener (named pipe)
        self.event_listener = EventBusListener(self.sentinel)
        self.event_listener.start()

        self._build_header()
        self._build_tabs()

        self._refresh_all()

    # ---------- HEADER (lockdown + autonomous) ----------

    def _build_header(self):
        frame = tk.Frame(self.root)
        frame.pack(fill="x", pady=5)

        self.lockdown_label = tk.Label(
            frame,
            text=self._lockdown_text(),
            font=("Consolas", 12, "bold"),
            fg=self._lockdown_color()
        )
        self.lockdown_label.pack(side="left", padx=10)

        self.autonomous_label = tk.Label(
            frame,
            text=self._autonomous_text(),
            font=("Consolas", 10),
            fg=self._autonomous_color()
        )
        self.autonomous_label.pack(side="left", padx=10)

        self.lockdown_button = tk.Button(
            frame,
            text=self._lockdown_button_text(),
            command=self.toggle_lockdown
        )
        self.lockdown_button.pack(side="right", padx=10)

        self.autonomous_button = tk.Button(
            frame,
            text=self._autonomous_button_text(),
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
        new_state = not self.sentinel.state.get("lockdown")
        self.sentinel.state["lockdown"] = new_state
        save_state(self.sentinel.state)
        self.lockdown_label.config(text=self._lockdown_text(), fg=self._lockdown_color())
        self.lockdown_button.config(text=self._lockdown_button_text())
        messagebox.showinfo(
            "Lockdown Toggled",
            f"Lockdown is now {'ON' if new_state else 'OFF'}.\n"
            f"Your organism should enforce this."
        )

    def toggle_autonomous(self):
        new_state = not self.sentinel.state.get("autonomous")
        self.sentinel.state["autonomous"] = new_state
        save_state(self.sentinel.state)
        self.autonomous_label.config(text=self._autonomous_text(), fg=self._autonomous_color())
        self.autonomous_button.config(text=self._autonomous_button_text())
        messagebox.showinfo(
            "Autonomous Mode",
            f"Autonomous mode is now {'ON' if new_state else 'OFF'}.\n"
            f"Manual override applies when OFF."
        )

    # ---------- TABS ----------

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.timeline_frame = tk.Frame(self.notebook)
        self.threat_frame = tk.Frame(self.notebook)
        self.node_sync_frame = tk.Frame(self.notebook)
        self.logs_frame = tk.Frame(self.notebook)
        self.swarm_frame = tk.Frame(self.notebook)
        self.persona_frame = tk.Frame(self.notebook)
        self.event_bus_frame = tk.Frame(self.notebook)

        self.notebook.add(self.timeline_frame, text="Timeline")
        self.notebook.add(self.threat_frame, text="Threat Matrix")
        self.notebook.add(self.node_sync_frame, text="Node Sync")
        self.notebook.add(self.logs_frame, text="Logs")
        self.notebook.add(self.swarm_frame, text="Swarm Sim")
        self.notebook.add(self.persona_frame, text="Persona")
        self.notebook.add(self.event_bus_frame, text="Event Bus")

        # Timeline
        self.timeline_list = tk.Listbox(self.timeline_frame, font=("Consolas", 9))
        self.timeline_list.pack(fill="both", expand=True)

        # Threat matrix
        self.threat_list = tk.Listbox(self.threat_frame, font=("Consolas", 9))
        self.threat_list.pack(fill="both", expand=True)

        # Node sync
        self.node_sync_list = tk.Listbox(self.node_sync_frame, font=("Consolas", 9))
        self.node_sync_list.pack(fill="both", expand=True)

        # Logs
        self.logs_text = tk.Text(self.logs_frame, font=("Consolas", 9))
        self.logs_text.pack(fill="both", expand=True)

        # Swarm sim (animated overlay via canvas)
        self.swarm_canvas = tk.Canvas(self.swarm_frame, bg="black")
        self.swarm_canvas.pack(fill="both", expand=True)

        # Persona
        self.persona_label = tk.Label(self.persona_frame, font=("Consolas", 11))
        self.persona_label.pack(pady=10)

        # Event bus
        self.event_bus_list = tk.Listbox(self.event_bus_frame, font=("Consolas", 9))
        self.event_bus_list.pack(fill="both", expand=True)

    # ---------- REFRESH LOOP ----------

    def _refresh_all(self):
        self.sentinel.fetch_timeline_events()
        self.sentinel.fetch_threat_matrix()
        self.sentinel.fetch_node_sync_state()
        self.sentinel.fetch_logs()
        self.sentinel.fetch_swarm_sim_state()
        self.sentinel.fetch_persona_state()
        # event_bus is updated by the pipe listener

        self._update_timeline()
        self._update_threat_matrix()
        self._update_node_sync()
        self._update_logs()
        self._update_swarm_canvas()
        self._update_persona()
        self._update_event_bus()

        self.root.after(self.REFRESH_MS, self._refresh_all)

    def _update_timeline(self):
        self.timeline_list.delete(0, tk.END)
        for ev in self.sentinel.timeline:
            time_str = ev.get("time", "?")
            sev = ev.get("severity", "info").upper()
            msg = ev.get("event", "")
            line = f"{time_str} :: {sev} :: {msg}"
            self.timeline_list.insert(tk.END, line)

    def _update_threat_matrix(self):
        self.threat_list.delete(0, tk.END)
        for row in self.sentinel.threat_matrix:
            node = row.get("node", "?")
            score = row.get("score", 0)
            line = f"{node} :: score={score}"
            self.threat_list.insert(tk.END, line)

    def _update_node_sync(self):
        self.node_sync_list.delete(0, tk.END)
        for row in self.sentinel.node_sync:
            node = row.get("node", "?")
            status = row.get("status", "unknown")
            line = f"{node} :: {status}"
            self.node_sync_list.insert(tk.END, line)

    def _update_logs(self):
        self.logs_text.delete("1.0", tk.END)
        for line in self.sentinel.logs:
            self.logs_text.insert(tk.END, line + "\n")

    def _update_swarm_canvas(self):
        self.swarm_canvas.delete("all")
        w = self.swarm_canvas.winfo_width()
        h = self.swarm_canvas.winfo_height()
        nodes = self.sentinel.swarm_sim.get("nodes", 3)
        for i in range(nodes):
            x = (i + 1) * w / (nodes + 1)
            y = h / 2 + random.randint(-20, 20)
            self.swarm_canvas.create_oval(
                x - 10, y - 10, x + 10, y + 10,
                fill="#33AAFF", outline="#FFFFFF"
            )

    def _update_persona(self):
        mode = self.sentinel.persona.get("mode", "Unknown")
        auto = self.sentinel.persona.get("autonomous", self.sentinel.state.get("autonomous", True))
        text = f"Persona Mode: {mode}\nAutonomous: {'ON' if auto else 'OFF'}"
        self.persona_label.config(text=text)

    def _update_event_bus(self):
        self.event_bus_list.delete(0, tk.END)
        for ev in self.sentinel.event_bus:
            self.event_bus_list.insert(tk.END, ev)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    root = tk.Tk()
    cockpit = OperatorCockpit(root)
    root.geometry("1000x650")
    root.mainloop()
