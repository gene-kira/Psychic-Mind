import os
import sys
import json
import time
import random
import socket
import subprocess
import platform
import ctypes
from datetime import datetime
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
THREAT_FILE = os.path.expanduser("borg_threats.json")
SWARM_THREAT_FILE = os.path.expanduser("borg_swarm_threats.json")
TRUSTED_FOLDERS_FILE = os.path.expanduser("borg_trusted_folders.json")

PROBE_HOST = "8.8.8.8"
PROBE_PORT = 53
PROBE_TIMEOUT = 2
CHECK_INTERVAL_SECONDS = 5

WINDOWS_INTERFACES = ["Ethernet", "Wi-Fi"]

WHITELIST_RULES = [
    {"process": None, "remote_ip": "127.0.0.1", "remote_port": None, "protocol": "any"},
]

BAD_PORTS = {23, 2323, 3389}
REMOTE_ADMIN_PORTS = {22, 23, 3389}

SYSTEM_DIRS = []
if platform.system() == "Windows":
    SYSTEM_DIRS = [
        os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32"),
        os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "SysWOW64"),
    ]
else:
    SYSTEM_DIRS = ["/bin", "/sbin", "/usr/bin", "/usr/sbin"]

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
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


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
# GPU TELEMETRY
# =========================

def get_gpu_load():
    """
    Best-effort GPU load (0-100).
    NVIDIA: nvidia-smi
    AMD:    rocm-smi
    Intel:  intel_gpu_top (approx)
    Returns 0 if unknown/unavailable.
    """
    # NVIDIA
    try:
        if which("nvidia-smi"):
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            )
            vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
            if vals:
                return max(0, min(100, sum(vals) // len(vals)))
    except Exception:
        pass

    # AMD
    try:
        if which("rocm-smi"):
            out = subprocess.check_output(
                ["rocm-smi", "--showuse"],
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            )
            # crude parse: look for '%' tokens
            nums = []
            for tok in out.replace("%", " ").split():
                if tok.isdigit():
                    nums.append(int(tok))
            if nums:
                return max(0, min(100, sum(nums) // len(nums)))
    except Exception:
        pass

    # Intel (very rough; intel_gpu_top is interactive, so we just return 0)
    # Could be extended with a wrapper or sysfs parsing.
    return 0


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
    try:
        with open(THREAT_FILE, "w", encoding="utf-8") as f:
            json.dump(threats, f, indent=2)
    except Exception as e:
        log_event("warning", "threats_save_failed", {"error": str(e)})


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
# MANUAL TRUSTED FOLDERS
# =========================

def load_manual_trusted_folders():
    if not os.path.exists(TRUSTED_FOLDERS_FILE):
        return []
    try:
        with open(TRUSTED_FOLDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_event("warning", "trusted_folders_load_failed", {"error": str(e)})
        return []


def save_manual_trusted_folders(folders):
    try:
        with open(TRUSTED_FOLDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(folders, f, indent=2)
    except Exception as e:
        log_event("warning", "trusted_folders_save_failed", {"error": str(e)})


# =========================
# TRUSTED GAMING ZONE DETECTION (IMPROVED)
# =========================

def detect_steam_libraries():
    paths = []
    if platform.system() != "Windows":
        return paths
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\\Valve\\Steam")
        steam_path, _ = winreg.QueryValueEx(key, "SteamPath")
        winreg.CloseKey(key)
        steam_path = os.path.expandvars(steam_path)
        common = os.path.join(steam_path, "steamapps", "common")
        if os.path.isdir(common):
            paths.append(common)
    except Exception as e:
        log_event("warning", "steam_detect_failed", {"error": str(e)})
    return paths


def detect_epic_libraries():
    paths = []
    try:
        if platform.system() == "Windows":
            base = os.path.join(os.environ.get("ProgramData", "C:\\ProgramData"),
                                "Epic", "EpicGamesLauncher", "Data", "Manifests")
            if os.path.isdir(base):
                for fn in os.listdir(base):
                    if not fn.lower().endswith(".item"):
                        continue
                    full = os.path.join(base, fn)
                    try:
                        with open(full, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        install_loc = data.get("InstallLocation")
                        if install_loc and os.path.isdir(install_loc):
                            paths.append(install_loc)
                    except Exception:
                        continue
        else:
            default = os.path.expanduser("~/EpicGames")
            if os.path.isdir(default):
                paths.append(default)
    except Exception as e:
        log_event("warning", "epic_detect_failed", {"error": str(e)})
    return paths


def gather_trusted_game_roots():
    roots = set()
    for p in detect_steam_libraries():
        roots.add(os.path.abspath(p))
    for p in detect_epic_libraries():
        roots.add(os.path.abspath(p))
    for p in load_manual_trusted_folders():
        roots.add(os.path.abspath(p))
    return list(roots)


def is_under_any(path, roots):
    path = os.path.abspath(path)
    for r in roots:
        try:
            r = os.path.abspath(r)
            if os.path.commonpath([path, r]) == r:
                return True
        except Exception:
            continue
    return False


def is_trusted_game_exe(exe_path, roots=None):
    if not exe_path:
        return False
    if roots is None:
        roots = gather_trusted_game_roots()
    return is_under_any(exe_path, roots)


# =========================
# PROCESS CLASSIFICATION
# =========================

CLASS_SYSTEM = "system_core"
CLASS_TRUSTED_GAME = "trusted_gaming"
CLASS_USER_APP = "user_app"
CLASS_UNKNOWN = "unknown"

def classify_process(proc, trusted_roots=None):
    try:
        exe = proc.exe()
    except Exception:
        exe = None

    try:
        name = proc.name().lower()
    except Exception:
        name = ""

    try:
        ppid = proc.ppid()
        parent = psutil.Process(ppid) if ppid else None
        parent_name = parent.name().lower() if parent else ""
    except Exception:
        parent = None
        parent_name = ""

    if platform.system() == "Windows":
        if exe and is_under_any(exe, SYSTEM_DIRS):
            return CLASS_SYSTEM
        if name in ("system", "smss.exe", "csrss.exe", "wininit.exe",
                    "services.exe", "lsass.exe", "winlogon.exe"):
            return CLASS_SYSTEM

    if is_trusted_game_exe(exe, trusted_roots):
        return CLASS_TRUSTED_GAME

    if name in ("steam.exe", "epicgameslauncher.exe"):
        return CLASS_TRUSTED_GAME

    user_app_names = [
        "chrome.exe", "msedge.exe", "firefox.exe", "notepad.exe",
        "code.exe", "explorer.exe", "discord.exe"
    ]
    if name in user_app_names:
        return CLASS_USER_APP

    return CLASS_UNKNOWN


# =========================
# INSANE BEHAVIOR DETECTION
# =========================

def check_insane_behavior(proc):
    alerts = []
    try:
        exe = proc.exe()
    except Exception:
        exe = None

    try:
        name = proc.name()
    except Exception:
        name = "<unknown>"

    try:
        conns = proc.connections(kind="inet")
    except Exception:
        conns = []

    for c in conns:
        try:
            laddr = c.laddr
            raddr = c.raddr
            status = c.status
        except Exception:
            continue

        if status == psutil.CONN_LISTEN and laddr:
            if laddr.port in REMOTE_ADMIN_PORTS:
                alerts.append({
                    "type": "insane_listener_remote_admin",
                    "proc": name,
                    "pid": proc.pid,
                    "port": laddr.port
                })

        if raddr:
            if raddr.port in REMOTE_ADMIN_PORTS:
                alerts.append({
                    "type": "insane_outbound_remote_admin",
                    "proc": name,
                    "pid": proc.pid,
                    "remote_ip": raddr.ip,
                    "remote_port": raddr.port
                })

    for a in alerts:
        log_event("insane", "insane_behavior_detected", a)

    return alerts


# =========================
# ANOMALY DETECTOR
# =========================

class AnomalyDetector:
    def __init__(self):
        self.last_online = None
        self.last_interfaces_enabled = None

    def infer_interfaces_enabled(self):
        return is_online()

    def update_and_check(self, expect_enabled: bool):
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

        if not expect_enabled and online:
            log_event("anomaly",
                      "Online while policy expects disabled",
                      {"policy_expect_enabled": expect_enabled,
                       "online": online,
                       "interfaces_enabled": interfaces_enabled})

        if expect_enabled and not online:
            log_event("anomaly",
                      "Offline while policy expects enabled",
                      {"policy_expect_enabled": expect_enabled,
                       "online": online,
                       "interfaces_enabled": interfaces_enabled})


# =========================
# INTRUSION DETECTOR (WITH TRUSTED GAME EXCEPTIONS)
# =========================

class IntrusionDetector:
    def __init__(self):
        self.ip_outbound_counts_prev = {}
        self.last_check_time = time.time()

    def analyze(self):
        now = time.time()
        self.last_check_time = now

        try:
            conns = psutil.net_connections(kind="inet")
        except Exception:
            return []

        alerts = []
        ip_outbound_counts = {}
        trusted_roots = gather_trusted_game_roots()

        for c in conns:
            try:
                pid = c.pid
                status = c.status
                laddr = c.laddr
                raddr = c.raddr
            except Exception:
                continue

            proc = None
            pclass = CLASS_UNKNOWN
            try:
                if pid:
                    proc = psutil.Process(pid)
                    pclass = classify_process(proc, trusted_roots)
            except Exception:
                proc = None

            trusted_game = (pclass == CLASS_TRUSTED_GAME)

            if status == psutil.CONN_LISTEN and laddr:
                lip = laddr.ip
                lport = laddr.port
                if lip not in ("127.0.0.1", "::1") and lport in BAD_PORTS:
                    if not trusted_game:
                        alerts.append({
                            "type": "listener_bad_port",
                            "ip": lip,
                            "port": lport,
                            "pid": pid,
                            "class": pclass
                        })
                        log_event("intrusion", "listener_bad_port",
                                  {"ip": lip, "port": lport, "pid": pid, "class": pclass})

            if raddr:
                rip = raddr.ip
                rport = raddr.port
                ip_outbound_counts[rip] = ip_outbound_counts.get(rip, 0) + 1

                if rport in BAD_PORTS and not trusted_game:
                    alerts.append({
                        "type": "outbound_bad_port",
                        "remote_ip": rip,
                        "remote_port": rport,
                        "pid": pid,
                        "class": pclass
                    })
                    log_event("intrusion", "outbound_bad_port",
                              {"remote_ip": rip, "remote_port": rport,
                               "pid": pid, "class": pclass})

        for ip, count in ip_outbound_counts.items():
            prev = self.ip_outbound_counts_prev.get(ip, 0)
            if count > prev * 5 and count > 20:
                alerts.append({
                    "type": "outbound_spike",
                    "remote_ip": ip,
                    "count": count,
                    "prev": prev
                })
                log_event("intrusion", "outbound_spike",
                          {"remote_ip": ip, "count": count, "prev": prev})
            self.ip_outbound_counts_prev[ip] = count

        return alerts


# =========================
# BORG GUARDIAN (NO SCHEDULE, AUTO-DISABLE ON ANOMALIES)
# =========================

class BorgNetGuardian:
    def __init__(self):
        self.detector = AnomalyDetector()
        self.policy_expect_enabled = True  # always expects enabled now
        self.intrusion_detector = IntrusionDetector()

    def _run_intrusion_detection(self):
        alerts = self.intrusion_detector.analyze()
        for a in alerts:
            ip = a.get("remote_ip") or a.get("ip")
            if ip:
                block_remote_ip(ip, reason=a["type"])
        return alerts

    def _scan_insane_behavior(self):
        insane_alerts = []
        try:
            for proc in psutil.process_iter(["pid", "name", "exe"]):
                insane_alerts.extend(check_insane_behavior(proc))
        except Exception:
            pass
        return insane_alerts

    def guardian_loop(self):
        log_event("info", "Guardian loop started",
                  {"policy_expect_enabled": self.policy_expect_enabled})

        while True:
            online = is_online()

            intrusion_alerts = self._run_intrusion_detection()
            insane_alerts = self._scan_insane_behavior()

            if intrusion_alerts or insane_alerts:
                log_event("enforce",
                          "Anomalies detected; auto-disabling network",
                          {"intrusion_count": len(intrusion_alerts),
                           "insane_count": len(insane_alerts)})
                disable_all_network()
                self.policy_expect_enabled = False
            else:
                pass

            self.detector.update_and_check(self.policy_expect_enabled)

            sync_threats_with_swarm()

            time.sleep(CHECK_INTERVAL_SECONDS)


# =========================
# CORTEX MEMORY HELPERS
# =========================

def load_recent_anomalies(limit=20):
    """
    Parse log and return last N anomaly/insane/intrusion/enforce events.
    """
    if not os.path.exists(LOG_FILE):
        return []
    events = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") in ("anomaly", "insane", "intrusion", "enforce"):
                    events.append(rec)
    except Exception:
        return []
    return events[-limit:]


def load_cortex_memory(limit=50):
    """
    Extract 'why blocked' style explanations from log.
    """
    if not os.path.exists(LOG_FILE):
        return []
    mem = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ev = rec.get("event")
                msg = rec.get("msg", "")
                extra = rec.get("extra", {})
                if ev in ("harden", "intrusion", "insane", "enforce"):
                    mem.append({
                        "ts": rec.get("ts"),
                        "event": ev,
                        "msg": msg,
                        "extra": extra
                    })
    except Exception:
        return []
    return mem[-limit:]

# =========================
# GUI INTERFACE
# =========================

def launch_gui():
    import tkinter as tk
    from tkinter import scrolledtext, messagebox, filedialog
    import threading

    class BorgInterface:
        def __init__(self, root):
            self.root = root
            self.root.title("BORG OS // CONTROL NODE")
            self.root.geometry("1200x900")
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

            self.telemetry_label = tk.Label(self.root, text="CPU: ...  RAM: ...  GPU: ...",
                                            fg="#aaaaaa", bg="#0a0a0a",
                                            font=("Consolas", 12))
            self.telemetry_label.pack(pady=2)

            self.autoloader_label = tk.Label(self.root, text="Autoloader: ...",
                                             fg="#aaaaaa", bg="#0a0a0a",
                                             font=("Consolas", 12))
            self.autoloader_label.pack(pady=2)

            self.canvas = tk.Canvas(self.root, width=1100, height=220,
                                    bg="#000000", highlightthickness=0)
            self.canvas.pack(pady=10)

            self.heat_cells = []
            rows, cols = 6, 28
            cell_w = 1100 // cols
            cell_h = 220 // rows

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
                {"name": "NODE-ALPHA", "x": 160, "y": 110, "status": "ok"},
                {"name": "NODE-BETA", "x": 480, "y": 70, "status": "ok"},
                {"name": "NODE-GAMMA", "x": 880, "y": 150, "status": "ok"},
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

            tk.Button(frame, text="TRUSTED GAMING ZONE",
                      command=self.open_trusted_gaming_zone,
                      bg="#002233", fg="#66ffff",
                      font=("Consolas", 14)).grid(row=0, column=3, padx=10)

            tk.Button(frame, text="THREAT REPLAY",
                      command=self.open_threat_replay,
                      bg="#221100", fg="#ffbb66",
                      font=("Consolas", 14)).grid(row=0, column=4, padx=10)

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

            tk.Button(self.root, text="NETWORK MAP",
                      command=self.open_network_map,
                      bg="#001122", fg="#66aaff",
                      font=("Consolas", 14)).pack(pady=5)

            tk.Button(self.root, text="CORTEX MEMORY",
                      command=self.open_cortex_memory,
                      bg="#112200", fg="#ccff66",
                      font=("Consolas", 14)).pack(pady=5)

            tk.Button(self.root, text="SWARM SYNC MONITOR",
                      command=self.open_swarm_monitor,
                      bg="#220011", fg="#ff99cc",
                      font=("Consolas", 14)).pack(pady=5)

            tk.Label(self.root, text="BORG CORTEX",
                     fg="#00ffea", bg="#0a0a0a",
                     font=("Consolas", 18)).pack(pady=10)

            self.cortex_output = scrolledtext.ScrolledText(
                self.root, width=110, height=10,
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
            log_event("operator", "network_enabled_gui", {})

        def disable_net(self):
            disable_all_network()
            messagebox.showinfo("Network", "Network DISABLED")
            log_event("operator", "network_disabled_gui", {})

        def self_repair(self):
            autoload_libraries()
            messagebox.showinfo("Self-Repair", "Autoloader re-run. Check Dependency Health for status.")

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
            gpu = get_gpu_load()
            self.telemetry_label.config(
                text=f"CPU: {cpu:.1f}%   RAM: {ram:.1f}%   GPU: {gpu:.1f}%"
            )
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
            gpu = get_gpu_load()
            # Blend CPU and GPU: 50/50
            load = (cpu + gpu) / 2.0
            base_intensity = min(1.0, load / 100.0 + 0.1)

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

        # ====== LOG VIEWER ======

        def open_log_viewer(self):
            win = tk.Toplevel(self.root)
            win.title("Log Viewer")
            win.geometry("900x600")
            win.configure(bg="#0a0a0a")

            text = scrolledtext.ScrolledText(
                win, width=100, height=32,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 10)
            )
            text.pack()

            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    text.insert("1.0", f.read())
            else:
                text.insert("1.0", "No log file found.")

        # ====== THREAT LIST ======

        def open_threat_list(self):
            win = tk.Toplevel(self.root)
            win.title("Threat List")
            win.geometry("700x400")
            win.configure(bg="#0a0a0a")

            text = scrolledtext.ScrolledText(
                win, width=80, height=20,
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

        # ====== P2P BLOCK ======

        def block_p2p(self):
            block_p2p_edonkey_all_ports()
            messagebox.showinfo("P2P Block", "All eDonkey/eMule traffic blocked")

        # ====== DEP HEALTH ======

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
                text.insert("1.0", "Autoloader has not run yet.\n")
            else:
                for lib, st in AUTOLOADER_STATUS["results"].items():
                    text.insert("end", f"{lib}: {st}\n")

        # ====== TRUSTED GAMING ZONE PANEL ======

        def open_trusted_gaming_zone(self):
            win = tk.Toplevel(self.root)
            win.title("Trusted Gaming Zone")
            win.geometry("900x600")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Trusted Gaming Zone",
                              fg="#00ffea", bg="#0a0a0a",
                              font=("Consolas", 16))
            header.pack(pady=5)

            status_label = tk.Label(win, text="Gaming zone: clear",
                                    fg="#66ff66", bg="#0a0a0a",
                                    font=("Consolas", 12))
            status_label.pack(pady=2)

            top_frame = tk.Frame(win, bg="#0a0a0a")
            top_frame.pack(fill="both", expand=True, padx=10, pady=5)

            libs_frame = tk.LabelFrame(top_frame, text="Detected libraries",
                                       fg="#66aaff", bg="#0a0a0a",
                                       font=("Consolas", 11))
            libs_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

            libs_text = scrolledtext.ScrolledText(
                libs_frame, width=40, height=15,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 10)
            )
            libs_text.pack(fill="both", expand=True, padx=5, pady=5)

            proc_frame = tk.LabelFrame(top_frame, text="Active trusted processes",
                                       fg="#66aaff", bg="#0a0a0a",
                                       font=("Consolas", 11))
            proc_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

            proc_text = scrolledtext.ScrolledText(
                proc_frame, width=40, height=15,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 10)
            )
            proc_text.pack(fill="both", expand=True, padx=5, pady=5)

            bottom_frame = tk.LabelFrame(win, text="Manual trusted folders",
                                         fg="#66aaff", bg="#0a0a0a",
                                         font=("Consolas", 11))
            bottom_frame.pack(fill="both", expand=True, padx=10, pady=5)

            listbox = tk.Listbox(bottom_frame, width=80, height=6,
                                 bg="#000000", fg="#00ffea",
                                 font=("Consolas", 10))
            listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)

            scrollbar = tk.Scrollbar(bottom_frame, orient="vertical", command=listbox.yview)
            scrollbar.pack(side="right", fill="y")
            listbox.config(yscrollcommand=scrollbar.set)

            def refresh_manual_folders():
                listbox.delete(0, "end")
                folders = load_manual_trusted_folders()
                if not folders:
                    listbox.insert("end", "(No manual trusted folders configured)")
                else:
                    for p in folders:
                        listbox.insert("end", p)

            def add_folder():
                path = filedialog.askdirectory()
                if not path:
                    return
                folders = load_manual_trusted_folders()
                if path not in folders:
                    folders.append(path)
                    save_manual_trusted_folders(folders)
                refresh_manual_folders()

            def remove_folder():
                folders = load_manual_trusted_folders()
                if not folders:
                    messagebox.showinfo("Manual folders", "No manual folders to remove.")
                    return
                sel = listbox.curselection()
                if not sel:
                    messagebox.showinfo("Manual folders", "Select a folder to remove.")
                    return
                path = listbox.get(sel[0])
                if path in folders:
                    folders.remove(path)
                    save_manual_trusted_folders(folders)
                refresh_manual_folders()

            btn_frame = tk.Frame(win, bg="#0a0a0a")
            btn_frame.pack(pady=5)

            tk.Button(btn_frame, text="Add trusted folder",
                      command=add_folder,
                      bg="#003300", fg="#00ff00",
                      font=("Consolas", 11)).pack(side="left", padx=5)

            tk.Button(btn_frame, text="Remove selected folder",
                      command=remove_folder,
                      bg="#330000", fg="#ff6666",
                      font=("Consolas", 11)).pack(side="left", padx=5)

            def refresh_libraries_and_processes():
                libs_text.delete("1.0", "end")
                proc_text.delete("1.0", "end")

                steam_paths = detect_steam_libraries()
                epic_paths = detect_epic_libraries()
                manual_paths = load_manual_trusted_folders()

                libs_text.insert("end", "Steam paths:\n")
                if steam_paths:
                    for p in steam_paths:
                        libs_text.insert("end", f"  {p}\n")
                else:
                    libs_text.insert("end", "  (none detected)\n")

                libs_text.insert("end", "\nEpic paths:\n")
                if epic_paths:
                    for p in epic_paths:
                        libs_text.insert("end", f"  {p}\n")
                else:
                    libs_text.insert("end", "  (none detected)\n")

                libs_text.insert("end", "\nManual folders:\n")
                if manual_paths:
                    for p in manual_paths:
                        libs_text.insert("end", f"  {p}\n")
                else:
                    libs_text.insert("end", "  (none configured)\n")

                roots = gather_trusted_game_roots()
                insane_found = False

                try:
                    for proc in psutil.process_iter(["pid", "name", "exe", "cpu_percent"]):
                        try:
                            exe = proc.info.get("exe") or proc.exe()
                        except Exception:
                            exe = None
                        if not exe:
                            continue
                        if not is_trusted_game_exe(exe, roots):
                            continue

                        name = proc.info.get("name") or proc.name()
                        cpu = proc.info.get("cpu_percent")
                        if cpu is None:
                            cpu = proc.cpu_percent(interval=0.0)

                        proc_text.insert(
                            "end",
                            f"{name} (PID {proc.pid})\n"
                            f"  EXE: {exe}\n"
                            f"  CPU: {cpu:.1f}%\n\n"
                        )

                        insane_alerts = check_insane_behavior(proc)
                        if insane_alerts:
                            insane_found = True

                except Exception as e:
                    proc_text.insert("end", f"Error scanning processes: {e}\n")

                if insane_found:
                    status_label.config(
                        text="Gaming zone: anomaly detected in trusted process",
                        fg="#ff6666"
                    )
                else:
                    status_label.config(
                        text="Gaming zone: clear",
                        fg="#66ff66"
                    )

            refresh_manual_folders()
            refresh_libraries_and_processes()

        # ====== THREAT REPLAY PANEL ======

        def open_threat_replay(self):
            win = tk.Toplevel(self.root)
            win.title("Threat Replay")
            win.geometry("900x500")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Threat Replay (last 20 anomalies)",
                              fg="#ffbb66", bg="#0a0a0a",
                              font=("Consolas", 14))
            header.pack(pady=5)

            canvas = tk.Canvas(win, width=850, height=200,
                               bg="#000000", highlightthickness=0)
            canvas.pack(pady=10)

            text = scrolledtext.ScrolledText(
                win, width=100, height=10,
                bg="#000000", fg="#ffbb66",
                font=("Consolas", 10)
            )
            text.pack(fill="both", expand=True, padx=10, pady=5)

            events = load_recent_anomalies(limit=20)
            if not events:
                text.insert("1.0", "No recent anomalies.\n")
                return

            # Timeline: left to right
            margin = 40
            width = 850 - 2 * margin
            y = 100
            n = len(events)
            if n == 1:
                xs = [margin + width // 2]
            else:
                xs = [margin + int(i * width / (n - 1)) for i in range(n)]

            # Draw baseline
            canvas.create_line(margin, y, 850 - margin, y, fill="#444444")

            nodes = []
            for i, (ev, x) in enumerate(zip(events, xs)):
                etype = ev.get("event")
                if etype == "intrusion":
                    color = "#ff4444"
                elif etype == "insane":
                    color = "#ff00ff"
                elif etype == "enforce":
                    color = "#ffaa00"
                else:
                    color = "#66aaff"
                node = canvas.create_oval(
                    x-8, y-8, x+8, y+8,
                    fill=color, outline=""
                )
                canvas.create_text(
                    x, y-18,
                    text=str(i+1),
                    fill="#ffffff",
                    font=("Consolas", 8)
                )
                nodes.append(node)

            # Animation: highlight each node in sequence and show details
            idx = {"i": 0}

            def step():
                i = idx["i"]
                if i >= len(nodes):
                    return
                for j, node in enumerate(nodes):
                    canvas.itemconfig(node, outline="", width=1)
                canvas.itemconfig(nodes[i], outline="#ffffff", width=2)

                ev = events[i]
                text.insert(
                    "end",
                    f"[{i+1}] {ev.get('ts')}  event={ev.get('event')}  msg={ev.get('msg')}  extra={ev.get('extra')}\n"
                )
                text.see("end")

                idx["i"] += 1
                if idx["i"] < len(nodes):
                    win.after(1200, step)

            step()

        # ====== NETWORK MAP ======

        def open_network_map(self):
            win = tk.Toplevel(self.root)
            win.title("Network Map")
            win.geometry("1000x700")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Network Map (processes & connections)",
                              fg="#66aaff", bg="#0a0a0a",
                              font=("Consolas", 14))
            header.pack(pady=5)

            canvas = tk.Canvas(win, width=960, height=600,
                               bg="#000000", highlightthickness=0)
            canvas.pack(pady=10)

            # Build node set: processes with inet connections
            try:
                conns = psutil.net_connections(kind="inet")
            except Exception:
                conns = []

            trusted_roots = gather_trusted_game_roots()
            nodes = {}
            edges = []

            for c in conns:
                pid = c.pid
                if not pid:
                    continue
                try:
                    proc = psutil.Process(pid)
                    name = proc.name()
                    pclass = classify_process(proc, trusted_roots)
                except Exception:
                    continue

                if pid not in nodes:
                    nodes[pid] = {
                        "name": name,
                        "class": pclass,
                        "degree": 0
                    }

                nodes[pid]["degree"] += 1

                raddr = c.raddr
                if raddr:
                    edges.append((pid, raddr.ip, raddr.port))

            if not nodes:
                canvas.create_text(
                    480, 300,
                    text="No active inet connections.",
                    fill="#888888",
                    font=("Consolas", 14)
                )
                return

            # Layout nodes in a circle
            center_x, center_y = 480, 320
            radius = 250
            pids = list(nodes.keys())
            n = len(pids)
            node_items = {}

            for i, pid in enumerate(pids):
                angle = 2 * 3.14159 * i / max(1, n)
                x = center_x + radius * 0.9 * (1.0 * (i % 2) - 0.5) if n < 6 else center_x + radius * 0.8 * (i / n - 0.5)
                # simpler: just use angle
                x = center_x + radius * 0.8 * (i / max(1, n) - 0.5)
                y = center_y + radius * 0.6 * (0.5 - i / max(1, n))

                pinfo = nodes[pid]
                pclass = pinfo["class"]
                if pclass == CLASS_TRUSTED_GAME:
                    color = "#66ff66"
                elif pclass == CLASS_SYSTEM:
                    color = "#66aaff"
                elif pclass == CLASS_USER_APP:
                    color = "#ffff66"
                else:
                    color = "#ff4444"

                node = canvas.create_oval(
                    x-10, y-10, x+10, y+10,
                    fill=color, outline=""
                )
                label = canvas.create_text(
                    x, y-18,
                    text=f"{pinfo['name']} ({pid})",
                    fill="#ffffff",
                    font=("Consolas", 8)
                )
                node_items[pid] = (node, label, x, y)

            # Draw edges (to remote IPs as small nodes on outer ring)
            remote_nodes = {}
            for pid, rip, rport in edges:
                if pid not in node_items:
                    continue
                if rip not in remote_nodes:
                    angle = random.uniform(0, 2 * 3.14159)
                    rx = center_x + radius * 1.1 * random.uniform(0.8, 1.2) * (1 if random.random() < 0.5 else -1)
                    ry = center_y + radius * 1.1 * random.uniform(0.8, 1.2) * (1 if random.random() < 0.5 else -1)
                    rnode = canvas.create_oval(
                        rx-6, ry-6, rx+6, ry+6,
                        fill="#888888", outline=""
                    )
                    rlabel = canvas.create_text(
                        rx, ry-12,
                        text=rip,
                        fill="#aaaaaa",
                        font=("Consolas", 7)
                    )
                    remote_nodes[rip] = (rnode, rlabel, rx, ry)

                node, label, x, y = node_items[pid]
                rnode, rlabel, rx, ry = remote_nodes[rip]

                # threat score color: if IP in threat list, edge red
                threats = load_threats()
                is_threat = any(t["ip"] == rip for t in threats)
                ecolor = "#ff4444" if is_threat else "#4444ff"

                canvas.create_line(x, y, rx, ry, fill=ecolor)

        # ====== CORTEX MEMORY PANEL ======

        def open_cortex_memory(self):
            win = tk.Toplevel(self.root)
            win.title("Cortex Memory")
            win.geometry("900x600")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Cortex Memory — Why things were blocked",
                              fg="#ccff66", bg="#0a0a0a",
                              font=("Consolas", 14))
            header.pack(pady=5)

            text = scrolledtext.ScrolledText(
                win, width=100, height=30,
                bg="#000000", fg="#ccff66",
                font=("Consolas", 10)
            )
            text.pack(fill="both", expand=True, padx=10, pady=5)

            mem = load_cortex_memory(limit=50)
            if not mem:
                text.insert("1.0", "No cortex memory entries yet.\n")
                return

            for m in mem:
                ts = m["ts"]
                ev = m["event"]
                msg = m["msg"]
                extra = m["extra"]
                text.insert(
                    "end",
                    f"[{ts}] event={ev}  msg={msg}\n  extra={extra}\n\n"
                )

        # ====== SWARM SYNC MONITOR ======

        def open_swarm_monitor(self):
            win = tk.Toplevel(self.root)
            win.title("Swarm Sync Monitor")
            win.geometry("900x600")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Swarm Sync Monitor",
                              fg="#ff99cc", bg="#0a0a0a",
                              font=("Consolas", 14))
            header.pack(pady=5)

            canvas = tk.Canvas(win, width=860, height=260,
                               bg="#000000", highlightthickness=0)
            canvas.pack(pady=10)

            text = scrolledtext.ScrolledText(
                win, width=100, height=15,
                bg="#000000", fg="#ff99cc",
                font=("Consolas", 10)
            )
            text.pack(fill="both", expand=True, padx=10, pady=5)

            # Visualize local vs swarm threats as nodes
            local = load_threats()
            remote = swarm_pull_threats()

            all_ips = set(t["ip"] for t in local) | set(t["ip"] for t in remote)
            if not all_ips:
                canvas.create_text(
                    430, 130,
                    text="No swarm threat data yet.",
                    fill="#888888",
                    font=("Consolas", 14)
                )
                text.insert("1.0", "No swarm threat data yet.\n")
                return

            center_x, center_y = 430, 130
            radius = 100
            ips = list(all_ips)
            n = len(ips)

            for i, ip in enumerate(ips):
                angle = 2 * 3.14159 * i / max(1, n)
                x = center_x + radius * 1.2 * (i / max(1, n) - 0.5)
                y = center_y + radius * 0.8 * (0.5 - i / max(1, n))

                in_local = any(t["ip"] == ip for t in local)
                in_remote = any(t["ip"] == ip for t in remote)

                if in_local and in_remote:
                    color = "#ff66ff"
                elif in_local:
                    color = "#ff4444"
                else:
                    color = "#66aaff"

                canvas.create_oval(
                    x-8, y-8, x+8, y+8,
                    fill=color, outline=""
                )
                canvas.create_text(
                    x, y-14,
                    text=ip,
                    fill="#ffffff",
                    font=("Consolas", 7)
                )

            text.insert("end", "Local threats:\n")
            for t in local:
                text.insert("end", f"  {t['ip']} reason={t['reason']} first={t['first_seen']} last={t['last_seen']}\n")

            text.insert("end", "\nSwarm threats:\n")
            for t in remote:
                text.insert("end", f"  {t['ip']} reason={t['reason']} first={t['first_seen']} last={t['last_seen']}\n")

        # ====== CORTEX ======

        def cortex_command(self, event=None):
            cmd = self.cortex_input.get().strip()
            if not cmd:
                return
            self.cortex_output.insert("end", f"> {cmd}\n")
            self.cortex_input.delete(0, "end")

            if cmd.lower() in ("help", "?"):
                self.cortex_output.insert(
                    "end",
                    "Commands:\n"
                    "  status   - show online/offline\n"
                    "  threats  - show threat count\n"
                    "  gpu      - show GPU load\n"
                )
            elif cmd.lower() == "status":
                self.cortex_output.insert(
                    "end",
                    f"Online: {is_online()}\n"
                )
            elif cmd.lower() == "threats":
                threats = load_threats()
                self.cortex_output.insert(
                    "end",
                    f"Threat entries: {len(threats)}\n"
                )
            elif cmd.lower() == "gpu":
                self.cortex_output.insert(
                    "end",
                    f"GPU load: {get_gpu_load():.1f}%\n"
                )
            else:
                self.cortex_output.insert("end", "Unknown command.\n")

        # ====== STUBS FOR VOICE / CONNECTION MONITOR ======

        def start_voice_thread(self):
            # Safe stub
            pass

        def start_connection_monitor(self):
            # Safe stub
            pass

    root = tk.Tk()
    app = BorgInterface(root)
    root.mainloop()
# =========================
# MAIN ENTRY
# =========================

def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() == "guardian":
        guardian = BorgNetGuardian()
        guardian.guardian_loop()
    else:
        launch_gui()


if __name__ == "__main__":
    main()