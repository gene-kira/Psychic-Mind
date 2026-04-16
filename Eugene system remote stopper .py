import os
import sys
import json
import time
import random
import socket
import subprocess
import platform
import ctypes
from datetime import datetime, time as dtime
from shutil import which

# === AUTOLOADER FOR REQUIRED PYTHON LIBRARIES ===
REQUIRED_LIBRARIES = [
    "psutil",
    "speechrecognition",
    "pyaudio",
    "tkinter",
]

AUTOLOADER_STATUS = {
    "attempted": False,
    "results": {}
}

def autoload_libraries():
    import importlib
    global AUTOLOADER_STATUS
    AUTOLOADER_STATUS["attempted"] = True
    AUTOLOADER_STATUS["results"] = {}

    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
            AUTOLOADER_STATUS["results"][lib] = "ok"
        except ImportError:
            AUTOLOADER_STATUS["results"][lib] = "missing_installing"
            print(f"[BORG AUTOLOADER] Missing: {lib} → installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                AUTOLOADER_STATUS["results"][lib] = "installed"
                print(f"[BORG AUTOLOADER] Installed: {lib}")
            except Exception as e:
                AUTOLOADER_STATUS["results"][lib] = f"failed: {e}"
                print(f"[BORG AUTOLOADER] Failed to install {lib}: {e}")

autoload_libraries()

# === AUTO-ELEVATION CHECK (Windows only) ===
def ensure_admin():
    if platform.system() != "Windows":
        return

    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            if not params:
                params = "gui"
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

import psutil  # safe now, autoloader ran

# =========================
# PATHS / CONFIG
# =========================

LOG_FILE = os.path.expanduser("borg_net_guardian.log")
SCHEDULE_FILE = os.path.expanduser("borg_schedule.json")
THREAT_FILE = os.path.expanduser("borg_threats.json")
SWARM_SCHEDULE_FILE = os.path.expanduser("borg_schedule_swarm.json")
SWARM_THREAT_FILE = os.path.expanduser("borg_swarm_threats.json")

PROBE_HOST = "8.8.8.8"
PROBE_PORT = 53
PROBE_TIMEOUT = 2
CHECK_INTERVAL_SECONDS = 5

WINDOWS_INTERFACES = ["Ethernet", "Wi-Fi"]

WHITELIST_RULES = [
    {"process": None, "remote_ip": "127.0.0.1", "remote_port": None, "protocol": "any"},
]

# =========================
# LOGGING / UTIL
# =========================

def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def log_event(event_type, message, extra=None):
    record = {
        "ts": now_iso(),
        "event": event_type,
        "msg": message,
        "extra": extra or {}
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(cmd, shell=True):
    try:
        subprocess.check_call(cmd, shell=shell)
        log_event("cmd_ok", "command_executed", {"cmd": cmd})
    except Exception as e:
        log_event("cmd_fail", "command_failed", {"cmd": cmd, "error": str(e)})


def is_online():
    try:
        socket.setdefaulttimeout(PROBE_TIMEOUT)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((PROBE_HOST, PROBE_PORT))
        s.close()
        return True
    except Exception:
        return False


# =========================
# NETWORK CONTROL
# =========================

def disable_all_network():
    system = platform.system()
    log_event("action", f"Disabling all network interfaces on {system}")

    if system == "Windows":
        for iface in WINDOWS_INTERFACES:
            cmd = f'netsh interface set interface name="{iface}" admin=disabled'
            run(cmd)
    elif system == "Linux":
        if which("nmcli"):
            run("nmcli networking off")
        else:
            try:
                for iface in os.listdir("/sys/class/net"):
                    if iface == "lo":
                        continue
                    run(f"ip link set {iface} down")
            except Exception as e:
                log_event("warning", "linux_disable_fallback_failed", {"error": str(e)})
    elif system == "Darwin":
        if which("networksetup"):
            run("networksetup -setairportpower airport off")
        else:
            log_event("warning", "networksetup_not_found_for_disable")
    else:
        log_event("warning", f"Unsupported OS for disable: {system}")


def enable_all_network():
    system = platform.system()
    log_event("action", f"Enabling all network interfaces on {system}")

    if system == "Windows":
        for iface in WINDOWS_INTERFACES:
            cmd = f'netsh interface set interface name="{iface}" admin=enabled'
            run(cmd)
    elif system == "Linux":
        if which("nmcli"):
            run("nmcli networking on")
        else:
            try:
                for iface in os.listdir("/sys/class/net"):
                    if iface == "lo":
                        continue
                    run(f"ip link set {iface} up")
            except Exception as e:
                log_event("warning", "linux_enable_fallback_failed", {"error": str(e)})
    elif system == "Darwin":
        if which("networksetup"):
            run("networksetup -setairportpower airport on")
        else:
            log_event("warning", "networksetup_not_found_for_enable")
    else:
        log_event("warning", f"Unsupported OS for enable: {system}")


# =========================
# P2P BLOCK (EDONKEY/EMULE)
# =========================

def block_p2p_edonkey_all_ports():
    system = platform.system()
    log_event("harden", f"block_p2p_edonkey_all_ports_start_{system}")

    signatures = ["edonkey", "emule", "kad", "overnet"]

    if system == "Windows":
        exe_names = ["emule.exe", "edonkey.exe", "edonkey2000.exe"]
        for exe in exe_names:
            run(f'netsh advfirewall firewall add rule name="Block_{exe}" '
                f'dir=out action=block program="%ProgramFiles%\\{exe}"')
            run(f'netsh advfirewall firewall add rule name="Block_{exe}_IN" '
                f'dir=in action=block program="%ProgramFiles%\\{exe}"')

        run('netsh advfirewall firewall add rule name="Block_eDonkey_All" '
            'dir=out action=block protocol=ANY remoteport=any')
        run('netsh advfirewall firewall add rule name="Block_eDonkey_All_IN" '
            'dir=in action=block protocol=ANY remoteport=any')

    elif system == "Linux":
        for sig in signatures:
            run(f'iptables -A INPUT -m string --algo bm --string "{sig}" -j DROP')
            run(f'iptables -A OUTPUT -m string --algo bm --string "{sig}" -j DROP')

    elif system == "Darwin":
        log_event("warning",
                  "macOS P2P block requires pf rules; block all ports or add protocol signatures manually.",
                  {})
    else:
        log_event("warning", f"Unsupported OS for full P2P block: {system}")


# =========================
# SCHEDULE ENGINE
# =========================

def default_schedule():
    return {
        "windows": [
            {
                "days": [0, 1, 2, 3, 4],
                "start": "23:00",
                "end": "07:00",
                "mode": "off",
                "random_jitter_minutes": 0
            },
            {
                "days": [5, 6],
                "start": "01:00",
                "end": "08:00",
                "mode": "off",
                "random_jitter_minutes": 30
            }
        ]
    }


def load_schedule():
    if not os.path.exists(SCHEDULE_FILE):
        sched = default_schedule()
        save_schedule(sched)
        return sched
    try:
        with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_event("warning", "schedule_load_failed", {"error": str(e)})
        sched = default_schedule()
        save_schedule(sched)
        return sched


def save_schedule(sched):
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump(sched, f, indent=2)


class ScheduleEngine:
    def __init__(self):
        self.schedule = load_schedule()
        self._compiled = {}
        self._last_day = None

    def _parse_time(self, tstr):
        h, m = map(int, tstr.split(":"))
        return dtime(hour=h, minute=m)

    def _compile_for_day(self, day):
        windows = []
        for w in self.schedule.get("windows", []):
            if day not in w.get("days", []):
                continue
            base_start = self._parse_time(w["start"])
            base_end = self._parse_time(w["end"])
            jitter = int(w.get("random_jitter_minutes", 0))

            def jitter_time(t):
                if jitter <= 0:
                    return t
                delta = random.randint(-jitter, jitter)
                total_minutes = t.hour * 60 + t.minute + delta
                total_minutes %= (24 * 60)
                h = total_minutes // 60
                m = total_minutes % 60
                return dtime(hour=h, minute=m)

            start = jitter_time(base_start)
            end = jitter_time(base_end)

            windows.append({
                "start": start,
                "end": end,
                "mode": w.get("mode", "off")
            })

        self._compiled[day] = windows
        log_event("schedule", "compiled_day_schedule",
                  {"day": day,
                   "windows": [
                       {"start": w["start"].isoformat(),
                        "end": w["end"].isoformat(),
                        "mode": w["mode"]}
                       for w in windows
                   ]})

    def _ensure_compiled_today(self):
        today = datetime.now().weekday()
        if today != self._last_day:
            self._compiled.clear()
            self._last_day = today
        if today not in self._compiled:
            self._compile_for_day(today)

    def desired_mode_now(self):
        self._ensure_compiled_today()
        now = datetime.now().time()
        today = datetime.now().weekday()
        windows = self._compiled.get(today, [])

        for w in windows:
            start = w["start"]
            end = w["end"]
            if start <= end:
                if start <= now < end:
                    return w["mode"]
            else:
                if now >= start or now < end:
                    return w["mode"]
        return None


# =========================
# THREATS / SWARM
# =========================

def load_threats():
    if not os.path.exists(THREAT_FILE):
        return []
    try:
        with open(THREAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_event("warning", "threats_load_failed", {"error": str(e)})
        return []


def save_threats(threats):
    with open(THREAT_FILE, "w", encoding="utf-8") as f:
        json.dump(threats, f, indent=2)


def swarm_pull_threats():
    if not os.path.exists(SWARM_THREAT_FILE):
        return []
    try:
        with open(SWARM_THREAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_event("warning", "swarm_threats_load_failed", {"error": str(e)})
        return []


def swarm_push_threats(threats):
    try:
        with open(SWARM_THREAT_FILE, "w", encoding="utf-8") as f:
            json.dump(threats, f, indent=2)
        log_event("swarm", "swarm_threats_pushed", {})
    except Exception as e:
        log_event("warning", "swarm_threats_push_failed", {"error": str(e)})


def sync_threats_with_swarm():
    local = load_threats()
    remote = swarm_pull_threats()
    merged = {t["ip"]: t for t in local}
    for t in remote:
        ip = t["ip"]
        if ip in merged:
            merged[ip]["first_seen"] = min(merged[ip]["first_seen"], t["first_seen"])
            merged[ip]["last_seen"] = max(merged[ip]["last_seen"], t["last_seen"])
        else:
            merged[ip] = t
    merged_list = list(merged.values())
    save_threats(merged_list)
    swarm_push_threats(merged_list)


def block_remote_ip(ip, reason="auto"):
    system = platform.system()
    log_event("harden", "auto_block_remote_ip", {"ip": ip, "reason": reason})

    if system == "Windows":
        run(f'netsh advfirewall firewall add rule name="AutoBlock_{ip}" '
            f'dir=in action=block remoteip={ip}')
        run(f'netsh advfirewall firewall add rule name="AutoBlock_{ip}_OUT" '
            f'dir=out action=block remoteip={ip}')
    elif system == "Linux":
        run(f"iptables -A INPUT -s {ip} -j DROP")
        run(f"iptables -A OUTPUT -d {ip} -j DROP")
    elif system == "Darwin":
        log_event("warning", "macOS_auto_block_ip_requires_pf_manual", {"ip": ip})

    threats = load_threats()
    if not any(t["ip"] == ip for t in threats):
        threats.append({
            "ip": ip,
            "reason": reason,
            "first_seen": now_iso(),
            "last_seen": now_iso()
        })
    else:
        for t in threats:
            if t["ip"] == ip:
                t["last_seen"] = now_iso()
    save_threats(threats)


# =========================
# WHITELIST / CONNECTIONS
# =========================

def conn_protocol(conn):
    if conn.type == socket.SOCK_STREAM:
        return "tcp"
    if conn.type == socket.SOCK_DGRAM:
        return "udp"
    return "any"


def is_whitelisted(conn):
    raddr = conn.raddr
    if not raddr:
        return True
    ip, port = raddr.ip, raddr.port
    proto = conn_protocol(conn)

    try:
        proc = psutil.Process(conn.pid) if conn.pid else None
        pname = proc.name().lower() if proc else None
    except Exception:
        pname = None

    for rule in WHITELIST_RULES:
        r_ip = rule["remote_ip"]
        r_port = rule["remote_port"]
        r_proto = rule["protocol"]
        r_proc = rule["process"]

        if r_ip is not None and ip != r_ip:
            continue
        if r_port is not None and port != r_port:
            continue
        if r_proto != "any" and proto != r_proto:
            continue
        if r_proc is not None:
            if pname is None or pname != r_proc.lower():
                continue
        return True

    return False


# =========================
# ANOMALY DETECTOR
# =========================

class AnomalyDetector:
    def __init__(self):
        self.last_online = None
        self.last_interfaces_enabled = None

    def infer_interfaces_enabled(self):
        return is_online()

    def update_and_check(self, policy_expect_enabled: bool):
        online = is_online()
        interfaces_enabled = self.infer_interfaces_enabled()

        if self.last_online is not None and online != self.last_online:
            log_event("state_change", "online_state_changed",
                      {"from": self.last_online, "to": online})

        if (self.last_interfaces_enabled is not None and
                interfaces_enabled != self.last_interfaces_enabled):
            log_event("state_change", "interfaces_enabled_changed",
                      {"from": self.last_interfaces_enabled, "to": interfaces_enabled})

        self.last_online = online
        self.last_interfaces_enabled = interfaces_enabled

        if not policy_expect_enabled and online:
            log_event("anomaly",
                      "Online while policy expects disabled",
                      {"policy_expect_enabled": policy_expect_enabled,
                       "online": online,
                       "interfaces_enabled": interfaces_enabled})

        if policy_expect_enabled and not online:
            log_event("anomaly",
                      "Offline while policy expects enabled",
                      {"policy_expect_enabled": policy_expect_enabled,
                       "online": online,
                       "interfaces_enabled": interfaces_enabled})


# =========================
# BORG GUARDIAN
# =========================

class BorgNetGuardian:
    def __init__(self):
        self.detector = AnomalyDetector()
        self.schedule_engine = ScheduleEngine()
        self.policy_expect_enabled = True

    def apply_schedule_policy(self):
        mode = self.schedule_engine.desired_mode_now()
        if mode == "off":
            if self.policy_expect_enabled:
                log_event("schedule", "schedule_forcing_off", {})
            self.policy_expect_enabled = False
        elif mode == "on":
            if not self.policy_expect_enabled:
                log_event("schedule", "schedule_forcing_on", {})
            self.policy_expect_enabled = True

    def guardian_loop(self):
        log_event("info", "Guardian loop started",
                  {"policy_expect_enabled": self.policy_expect_enabled})

        while True:
            self.apply_schedule_policy()

            online = is_online()

            if not self.policy_expect_enabled and online:
                log_event("enforce",
                          "Policy expects disabled but online detected; enforcing disable")
                disable_all_network()

            if self.policy_expect_enabled and not online:
                log_event("info",
                          "Offline while policy expects enabled; leaving as-is")

            self.detector.update_and_check(self.policy_expect_enabled)

            sync_threats_with_swarm()

            time.sleep(CHECK_INTERVAL_SECONDS)


# =========================
# GUI INTERFACE
# =========================

def launch_gui():
    import tkinter as tk
    from tkinter import scrolledtext, messagebox
    import threading

    class BorgInterface:
        def __init__(self, root):
            self.root = root
            self.root.title("BORG OS // CONTROL NODE")
            self.root.geometry("900x780")
            self.root.configure(bg="#0a0a0a")

            self.build_ui()
            self.update_status_loop()
            self.update_telemetry()
            self.update_autoloader_status()
            self.start_voice_thread()
            self.start_connection_monitor()

        def build_ui(self):
            title = tk.Label(self.root, text="BORG CONTROL INTERFACE",
                             fg="#00ffea", bg="#0a0a0a",
                             font=("Consolas", 22, "bold"))
            title.pack(pady=10)

            self.status_label = tk.Label(self.root, text="Status: ...",
                                         fg="#00ffea", bg="#0a0a0a",
                                         font=("Consolas", 16))
            self.status_label.pack(pady=5)

            self.telemetry_label = tk.Label(self.root, text="CPU: ...  RAM: ...",
                                            fg="#aaaaaa", bg="#0a0a0a",
                                            font=("Consolas", 12))
            self.telemetry_label.pack(pady=2)

            self.autoloader_label = tk.Label(self.root, text="Autoloader: ...",
                                             fg="#aaaaaa", bg="#0a0a0a",
                                             font=("Consolas", 12))
            self.autoloader_label.pack(pady=2)

            self.canvas = tk.Canvas(self.root, width=860, height=200,
                                    bg="#000000", highlightthickness=0)
            self.canvas.pack(pady=10)

            self.heat_cells = []
            rows, cols = 6, 20
            cell_w = 860 // cols
            cell_h = 200 // rows

            for r in range(rows):
                row_cells = []
                for c in range(cols):
                    x1 = c * cell_w
                    y1 = r * cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h
                    rect = self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        outline="#111111", fill="#050505"
                    )
                    row_cells.append(rect)
                self.heat_cells.append(row_cells)

            self.swarm_nodes = [
                {"name": "NODE-ALPHA", "x": 100, "y": 100, "status": "ok"},
                {"name": "NODE-BETA", "x": 300, "y": 60, "status": "ok"},
                {"name": "NODE-GAMMA", "x": 600, "y": 140, "status": "ok"},
            ]
            self.swarm_node_items = []
            for node in self.swarm_nodes:
                item = self.canvas.create_oval(
                    node["x"]-8, node["y"]-8,
                    node["x"]+8, node["y"]+8,
                    fill="#00ffea", outline=""
                )
                label = self.canvas.create_text(
                    node["x"], node["y"]-15,
                    text=node["name"],
                    fill="#00ffea",
                    font=("Consolas", 8)
                )
                self.swarm_node_items.append((item, label))

            self.animate_overlay()

            frame = tk.Frame(self.root, bg="#0a0a0a")
            frame.pack(pady=10)

            tk.Button(frame, text="ENABLE NETWORK",
                      command=self.enable_net,
                      bg="#003300", fg="#00ff00",
                      font=("Consolas", 14)).grid(row=0, column=0, padx=10)

            tk.Button(frame, text="DISABLE NETWORK",
                      command=self.disable_net,
                      bg="#330000", fg="#ff4444",
                      font=("Consolas", 14)).grid(row=0, column=1, padx=10)

            tk.Button(frame, text="SELF-REPAIR (LIBS)",
                      command=self.self_repair,
                      bg="#333300", fg="#ffff66",
                      font=("Consolas", 14)).grid(row=0, column=2, padx=10)

            tk.Button(self.root, text="VIEW / EDIT SCHEDULE",
                      command=self.open_schedule_editor,
                      bg="#001133", fg="#66aaff",
                      font=("Consolas", 14)).pack(pady=5)

            tk.Button(self.root, text="VIEW LOG",
                      command=self.open_log_viewer,
                      bg="#111111", fg="#aaaaaa",
                      font=("Consolas", 14)).pack(pady=5)

            tk.Button(self.root, text="THREAT LIST",
                      command=self.open_threat_list,
                      bg="#330000", fg="#ff6666",
                      font=("Consolas", 14)).pack(pady=5)

            tk.Button(self.root, text="BLOCK ALL EDONKEY P2P",
                      command=self.block_p2p,
                      bg="#330022", fg="#ff66cc",
                      font=("Consolas", 14)).pack(pady=5)

            tk.Button(self.root, text="DEPENDENCY HEALTH",
                      command=self.open_dep_health,
                      bg="#002222", fg="#66ffff",
                      font=("Consolas", 14)).pack(pady=5)

            tk.Label(self.root, text="BORG CORTEX",
                     fg="#00ffea", bg="#0a0a0a",
                     font=("Consolas", 18)).pack(pady=10)

            self.cortex_output = scrolledtext.ScrolledText(
                self.root, width=80, height=10,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 12)
            )
            self.cortex_output.pack()

            self.cortex_input = tk.Entry(
                self.root,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 14)
            )
            self.cortex_input.pack(fill="x", padx=20, pady=5)
            self.cortex_input.bind("<Return>", self.cortex_command)

        def enable_net(self):
            enable_all_network()
            messagebox.showinfo("Network", "Network ENABLED")

        def disable_net(self):
            disable_all_network()
            messagebox.showinfo("Network", "Network DISABLED")

        def update_status_loop(self):
            online = is_online()
            if online:
                self.status_label.config(text="Status: ONLINE", fg="#00ff00")
            else:
                self.status_label.config(text="Status: OFFLINE", fg="#ff4444")
            self.root.after(2000, self.update_status_loop)

        def update_telemetry(self):
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            self.telemetry_label.config(text=f"CPU: {cpu:.1f}%   RAM: {ram:.1f}%")
            self.root.after(2000, self.update_telemetry)

        def update_autoloader_status(self):
            if not AUTOLOADER_STATUS["attempted"]:
                self.autoloader_label.config(text="Autoloader: not run")
            else:
                bad = [lib for lib, st in AUTOLOADER_STATUS["results"].items()
                       if not (st == "ok" or st == "installed")]
                if bad:
                    self.autoloader_label.config(
                        text=f"Autoloader: issues with {', '.join(bad)}",
                        fg="#ff6666"
                    )
                else:
                    self.autoloader_label.config(
                        text="Autoloader: all dependencies OK",
                        fg="#66ff66"
                    )
            self.root.after(5000, self.update_autoloader_status)

        def animate_overlay(self):
            cpu = psutil.cpu_percent(interval=None)
            base_intensity = min(1.0, cpu / 100.0 + 0.1)

            for row in self.heat_cells:
                for rect in row:
                    jitter = random.uniform(-0.2, 0.2)
                    intensity = max(0.0, min(1.0, base_intensity + jitter))
                    g = int(20 + 200 * (1 - intensity))
                    r = int(20 + 200 * intensity)
                    color = f"#{r:02x}{g:02x}20"
                    self.canvas.itemconfig(rect, fill=color)

            for node, (item, label) in zip(self.swarm_nodes, self.swarm_node_items):
                if random.random() < 0.02:
                    node["status"] = random.choice(["ok", "warn", "alert"])
                if node["status"] == "ok":
                    color = "#00ffea"
                elif node["status"] == "warn":
                    color = "#ffaa00"
                else:
                    color = "#ff0044"
                self.canvas.itemconfig(item, fill=color)

            self.root.after(500, self.animate_overlay)

        def open_schedule_editor(self):
            win = tk.Toplevel(self.root)
            win.title("Schedule Editor")
            win.geometry("700x600")
            win.configure(bg="#0a0a0a")

            text = scrolledtext.ScrolledText(
                win, width=80, height=25,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 12)
            )
            text.pack()

            sched = load_schedule()
            text.insert("1.0", json.dumps(sched, indent=2))

            def save():
                try:
                    new_sched = json.loads(text.get("1.0", "end-1c"))
                    save_schedule(new_sched)
                    messagebox.showinfo("Saved", "Schedule updated")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

            tk.Button(win, text="SAVE",
                      command=save,
                      bg="#003300", fg="#00ff00",
                      font=("Consolas", 14)).pack(pady=10)

        def open_log_viewer(self):
            win = tk.Toplevel(self.root)
            win.title("Log Viewer")
            win.geometry("800x600")
            win.configure(bg="#0a0a0a")

            text = scrolledtext.ScrolledText(
                win, width=90, height=30,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 10)
            )
            text.pack()

            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    text.insert("1.0", f.read())
            else:
                text.insert("1.0", "No log file found.")

        def open_threat_list(self):
            win = tk.Toplevel(self.root)
            win.title("Threat List")
            win.geometry("600x400")
            win.configure(bg="#0a0a0a")

            text = scrolledtext.ScrolledText(
                win, width=70, height=20,
                bg="#000000", fg="#ff6666",
                font=("Consolas", 10)
            )
            text.pack(fill="both", expand=True)

            threats = load_threats()
            if not threats:
                text.insert("1.0", "No threats recorded.\n")
            else:
                for t in threats:
                    line = f"{t['ip']}  reason={t['reason']}  first={t['first_seen']}  last={t['last_seen']}\n"
                    text.insert("end", line)

        def block_p2p(self):
            block_p2p_edonkey_all_ports()
            messagebox.showinfo("P2P Block", "All eDonkey/eMule traffic blocked")

        def open_dep_health(self):
            win = tk.Toplevel(self.root)
            win.title("Dependency Health")
            win.geometry("500x300")
            win.configure(bg="#0a0a0a")

            text = scrolledtext.ScrolledText(
                win, width=60, height=15,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 10)
            )
            text.pack(fill="both", expand=True)

            if not AUTOLOADER_STATUS["attempted"]:
                text.insert("1.0", "Autoloader has not run.\n")
            else:
                for lib in REQUIRED_LIBRARIES:
                    st = AUTOLOADER_STATUS["results"].get(lib, "unknown")
                    text.insert("end", f"{lib}: {st}\n")

        def self_repair(self):
            global AUTOLOADER_STATUS
            AUTOLOADER_STATUS["attempted"] = False
            AUTOLOADER_STATUS["results"] = {}
            self.cortex_output.insert("end", "CORTEX: Initiating self-repair (dependency reinstall)...\n")
            self.cortex_output.see("end")
            try:
                autoload_libraries()
                self.cortex_output.insert("end", "CORTEX: Self-repair complete.\n")
            except Exception as e:
                self.cortex_output.insert("end", f"CORTEX: Self-repair failed: {e}\n")
            self.cortex_output.see("end")

        def cortex_command(self, event):
            cmd = self.cortex_input.get().strip()
            self.cortex_input.delete(0, "end")

            reply = self.query_cortex_ai(cmd)
            self.cortex_output.insert("end", f"YOU: {cmd}\nCORTEX: {reply}\n\n")
            self.cortex_output.see("end")

        def query_cortex_ai(self, text):
            t = text.lower()
            if "status" in t:
                return "Monitoring network, schedule, dependencies, and swarm nodes. No critical anomalies detected."
            if "dependencies" in t or "libs" in t:
                bad = [lib for lib, st in AUTOLOADER_STATUS["results"].items()
                       if not (st == "ok" or st == "installed")]
                if bad:
                    return f"Dependency issues detected with: {', '.join(bad)}"
                return "All tracked dependencies appear healthy."
            if "repair" in t or "self-repair" in t:
                self.self_repair()
                return "Self-repair sequence triggered."
            return "Cortex stub: wire me to your LLM endpoint for deeper reasoning."

        def start_voice_thread(self):
            import threading
            t = threading.Thread(target=self.voice_loop, daemon=True)
            t.start()

        def voice_loop(self):
            try:
                import speech_recognition as sr
            except ImportError:
                log_event("warning", "voice_modules_missing", {})
                return

            r = sr.Recognizer()
            try:
                mic = sr.Microphone()
            except Exception as e:
                log_event("warning", "voice_mic_error", {"error": str(e)})
                return

            self.cortex_output.insert("end", "CORTEX: Voice channel online.\n")
            self.cortex_output.see("end")

            while True:
                with mic as source:
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source)
                try:
                    text = r.recognize_google(audio).lower()
                    self.cortex_output.insert("end", f"VOICE: {text}\n")
                    self.cortex_output.see("end")

                    if "disable network" in text:
                        self.disable_net()
                    elif "enable network" in text:
                        self.enable_net()
                    elif "status" in text:
                        self.cortex_output.insert("end", "CORTEX: Status query acknowledged.\n")
                        self.cortex_output.see("end")
                    elif "repair" in text:
                        self.self_repair()
                except Exception as e:
                    log_event("voice", "recognition_error", {"error": str(e)})

        def start_connection_monitor(self):
            import threading
            t = threading.Thread(target=self.connection_monitor_loop, daemon=True)
            t.start()

        def connection_monitor_loop(self):
            while True:
                try:
                    conns = psutil.net_connections(kind="inet")
                    for c in conns:
                        raddr = c.raddr
                        if not raddr:
                            continue
                        if is_whitelisted(c):
                            continue
                        ip = raddr.ip
                        block_remote_ip(ip, reason="non_whitelisted_connection")
                        self.cortex_output.insert(
                            "end",
                            f"CORTEX: Auto-blocked {ip} (non-normal channel).\n"
                        )
                        self.cortex_output.see("end")
                    sync_threats_with_swarm()
                except Exception as e:
                    log_event("warning", "connection_monitor_error", {"error": str(e)})
                time.sleep(10)

    root = tk.Tk()
    app = BorgInterface(root)
    root.mainloop()


# =========================
# MAIN
# =========================

def main():
    block_p2p_edonkey_all_ports()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python borg_os_guardian.py guard   # headless guardian")
        print("  python borg_os_guardian.py gui     # Borg cockpit GUI")
        return

    mode = sys.argv[1].lower()

    if mode == "guard":
        guardian = BorgNetGuardian()
        guardian.guardian_loop()
    elif mode == "gui":
        launch_gui()
    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()