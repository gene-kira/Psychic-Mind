import os
import sys
import json
import time
import random
import socket
import subprocess
import platform
import ctypes
import hashlib
from datetime import datetime
from shutil import which

# === AUTOLOADER FOR REQUIRED PYTHON LIBRARIES ===
REQUIRED_LIBRARIES = [
    "psutil",
    "tkinter",
    "scapy",
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
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                AUTOLOADER_STATUS["results"][lib] = "installed"
            except Exception as e:
                AUTOLOADER_STATUS["results"][lib] = f"failed: {e}"

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
    except Exception:
        sys.exit()

ensure_admin()

import psutil

# =========================
# PATHS / CONFIG
# =========================

LOG_FILE = os.path.expanduser("borg_net_guardian.log")
THREAT_FILE = os.path.expanduser("borg_threats.json")
SWARM_THREAT_FILE = os.path.expanduser("borg_swarm_threats.json")
TRUSTED_FOLDERS_FILE = os.path.expanduser("borg_trusted_folders.json")
NODE_DISCOVERY_FILE = os.path.expanduser("borg_nodes.json")
POLICY_FILE = os.path.expanduser("borg_policies.json")

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

NODE_DISCOVERY_PORT = 44555
NODE_DISCOVERY_MAGIC = b"BORG_NODE_DISCOVERY_V1"

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

def sha256_of_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

# =========================
# GPU TELEMETRY
# =========================

def get_gpu_load():
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

    try:
        if which("rocm-smi"):
            out = subprocess.check_output(
                ["rocm-smi", "--showuse"],
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            )
            nums = []
            for tok in out.replace("%", " ").split():
                if tok.isdigit():
                    nums.append(int(tok))
            if nums:
                return max(0, min(100, sum(nums) // len(nums)))
    except Exception:
        pass

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

# =========================
# P2P BLOCK (EDONKEY/EMULE) — AUTO ON STARTUP
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
# NODE DISCOVERY STORAGE
# =========================

def load_nodes():
    if not os.path.exists(NODE_DISCOVERY_FILE):
        return []
    try:
        with open(NODE_DISCOVERY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_nodes(nodes):
    try:
        with open(NODE_DISCOVERY_FILE, "w", encoding="utf-8") as f:
            json.dump(nodes, f, indent=2)
    except Exception:
        pass
# =========================
# WHITELIST / CONNECTIONS
# =========================

def conn_protocol(conn):
    if conn.type == socket.SOCK_STREAM:
        return "tcp"
    if conn.type == socket.SOCK_DGRAM:
        return "udp"
    return "any"

# =========================
# TRUSTED FOLDERS
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
# TRUSTED GAMING ZONE DETECTION
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

        if raddr and raddr.port in REMOTE_ADMIN_PORTS:
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
# RAW PACKET SNIFFER (PS-B)
# =========================

try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False

SUSPICIOUS_PORTS = {23, 2323, 3389, 4444, 6667, 31337}
SUSPICIOUS_KEYWORDS = [b"botnet", b"shell", b"backdoor", b"c2", b"malware"]

class RawPacketSniffer:
    def __init__(self):
        self.running = False

    def _packet_handler(self, pkt):
        try:
            if IP not in pkt:
                return
            src = pkt[IP].src
            dst = pkt[IP].dst
            proto = None
            sport = None
            dport = None

            if TCP in pkt:
                proto = "TCP"
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
            elif UDP in pkt:
                proto = "UDP"
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport
            else:
                proto = "OTHER"

            suspicious = False
            reason = []

            if sport in SUSPICIOUS_PORTS or dport in SUSPICIOUS_PORTS:
                suspicious = True
                reason.append("suspicious_port")

            raw_bytes = bytes(pkt)
            for kw in SUSPICIOUS_KEYWORDS:
                if kw in raw_bytes:
                    suspicious = True
                    reason.append("keyword:" + kw.decode(errors="ignore"))
                    break

            if suspicious:
                info = {
                    "src": src,
                    "dst": dst,
                    "proto": proto,
                    "sport": sport,
                    "dport": dport,
                    "reason": reason
                }
                log_event("intrusion", "raw_packet_suspicious", info)
        except Exception:
            pass

    def start(self):
        if not SCAPY_AVAILABLE:
            log_event("warning", "scapy_not_available_for_raw_sniffer", {})
            return
        if self.running:
            return
        self.running = True

        import threading
        def _run():
            try:
                sniff(prn=self._packet_handler, store=False)
            except Exception as e:
                log_event("warning", "raw_sniffer_failed", {"error": str(e)})
            finally:
                self.running = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()

# =========================
# ADAPTIVE FIREWALL ORGAN
# =========================

class AdaptiveFirewall:
    def __init__(self):
        self.baseline = {}
        self.alpha = 0.1
        self.threshold_factor = 5.0

    def observe(self):
        try:
            conns = psutil.net_connections(kind="inet")
        except Exception:
            return []

        trusted_roots = gather_trusted_game_roots()
        alerts = []
        counts = {}

        for c in conns:
            pid = c.pid
            if not pid:
                continue
            try:
                proc = psutil.Process(pid)
                name = proc.name().lower()
                pclass = classify_process(proc, trusted_roots)
            except Exception:
                continue

            if pclass == CLASS_TRUSTED_GAME:
                continue

            raddr = c.raddr
            if not raddr:
                continue
            key = (name, raddr.ip)
            counts[key] = counts.get(key, 0) + 1

        for key, count in counts.items():
            base = self.baseline.get(key, None)
            if base is None:
                self.baseline[key] = float(count)
                continue

            if count > base * self.threshold_factor and count > 10:
                name, ip = key
                alert = {
                    "type": "adaptive_spike",
                    "process": name,
                    "remote_ip": ip,
                    "count": count,
                    "baseline": base
                }
                alerts.append(alert)
                log_event("intrusion", "adaptive_firewall_spike", alert)

            new_base = (1 - self.alpha) * base + self.alpha * count
            self.baseline[key] = new_base

        return alerts

# =========================
# DISTRIBUTED NODE DISCOVERY
# =========================

class NodeDiscovery:
    def __init__(self):
        self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        import threading
        t1 = threading.Thread(target=self._listener, daemon=True)
        t2 = threading.Thread(target=self._broadcaster, daemon=True)
        t1.start()
        t2.start()

    def _listener(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", NODE_DISCOVERY_PORT))
        except Exception as e:
            log_event("warning", "node_discovery_listener_failed", {"error": str(e)})
            return

        while self.running:
            try:
                data, addr = s.recvfrom(1024)
                if data.startswith(NODE_DISCOVERY_MAGIC):
                    node_ip = addr[0]
                    nodes = load_nodes()
                    if node_ip not in nodes:
                        nodes.append(node_ip)
                        save_nodes(nodes)
                        log_event("swarm", "node_discovered", {"ip": node_ip})
            except Exception:
                time.sleep(1)

    def _broadcaster(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception as e:
            log_event("warning", "node_discovery_broadcaster_failed", {"error": str(e)})
            return

        while self.running:
            try:
                s.sendto(NODE_DISCOVERY_MAGIC, ("<broadcast>", NODE_DISCOVERY_PORT))
            except Exception:
                pass
            time.sleep(10)

# =========================
# ZERO TRUST POLICY ENGINE + ADMIN OVERRIDE
# =========================

DEFAULT_POLICIES = [
    {
        "id": "trusted_gaming_anywhere",
        "match": {
            "class": ["trusted_gaming"]
        },
        "allow": {
            "remote_ip": "any",
            "remote_port": "any"
        },
        "enabled": True
    },
    {
        "id": "system_core_https_only",
        "match": {
            "class": ["system_core"]
        },
        "allow": {
            "remote_port": [80, 443],
            "remote_ip": "any"
        },
        "enabled": True
    },
    {
        "id": "default_deny",
        "match": {},
        "allow": None,
        "enabled": True
    }
]

def load_policies():
    if not os.path.exists(POLICY_FILE):
        save_policies(DEFAULT_POLICIES)
        return DEFAULT_POLICIES
    try:
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_event("warning", "policies_load_failed", {"error": str(e)})
        return DEFAULT_POLICIES

def save_policies(policies):
    try:
        with open(POLICY_FILE, "w", encoding="utf-8") as f:
            json.dump(policies, f, indent=2)
    except Exception as e:
        log_event("warning", "policies_save_failed", {"error": str(e)})

POLICIES = load_policies()

ADMIN_OVERRIDE = {
    "active": False,
    "until": None
}

def enable_admin_override(duration_sec=600):
    ADMIN_OVERRIDE["active"] = True
    ADMIN_OVERRIDE["until"] = time.time() + duration_sec
    log_event("override", "admin_override_enabled", {"duration": duration_sec})

def is_admin_override_active():
    if not ADMIN_OVERRIDE["active"]:
        return False
    if time.time() > ADMIN_OVERRIDE["until"]:
        ADMIN_OVERRIDE["active"] = False
        log_event("override", "admin_override_expired", {})
        return False
    return True

def ip_in_threats(ip):
    threats = load_threats()
    return any(t["ip"] == ip for t in threats)

def match_policy(policy, context):
    if not policy.get("enabled", True):
        return False
    match = policy.get("match", {})
    for key, val in match.items():
        if key == "class":
            if context.get("class") not in val:
                return False
        elif key == "proc_hash":
            if context.get("hash") not in val:
                return False
        elif key == "proc_name":
            if context.get("proc_name") not in val:
                return False
    return True

def allow_block_from_policy(policy, remote_ip, remote_port):
    allow = policy.get("allow")
    if allow is None:
        return False
    ip_rule = allow.get("remote_ip", "any")
    port_rule = allow.get("remote_port", "any")

    if ip_rule != "any":
        if remote_ip not in ip_rule:
            return False
    if port_rule != "any":
        if remote_port not in port_rule:
            return False
    return True

def is_connection_allowed(proc, remote_ip, remote_port, proto, context):
    # Admin override with risk-aware behavior
    if is_admin_override_active():
        if not ip_in_threats(remote_ip) and context.get("hash") is not None:
            log_event("policy_decision", "admin_override_allow", {
                "proc": context.get("proc_name"),
                "hash": context.get("hash"),
                "remote_ip": remote_ip,
                "remote_port": remote_port
            })
            return True, "admin_override_safe", "admin_override"

    for policy in POLICIES:
        if match_policy(policy, context):
            allowed = allow_block_from_policy(policy, remote_ip, remote_port)
            log_event("policy_decision", "connection_checked", {
                "proc": context.get("proc_name"),
                "hash": context.get("hash"),
                "remote_ip": remote_ip,
                "remote_port": remote_port,
                "proto": proto,
                "allowed": allowed,
                "policy_id": policy["id"]
            })
            return allowed, ("policy_allow" if allowed else "policy_deny"), policy["id"]

    log_event("policy_decision", "connection_checked", {
        "proc": context.get("proc_name"),
        "hash": context.get("hash"),
        "remote_ip": remote_ip,
        "remote_port": remote_port,
        "proto": proto,
        "allowed": False,
        "policy_id": "none"
    })
    return False, "no_policy_match", None

def evaluate_connection(conn):
    pid = conn.pid
    if not pid:
        return True, "no_pid", None
    try:
        proc = psutil.Process(pid)
    except Exception:
        return False, "proc_missing", None

    try:
        exe = proc.exe()
    except Exception:
        exe = None
    proc_hash = sha256_of_file(exe) if exe else None
    proc_name = proc.name()
    pclass = classify_process(proc, gather_trusted_game_roots())
    raddr = conn.raddr
    if not raddr:
        return True, "no_remote", None

    allowed, reason, policy_id = is_connection_allowed(
        proc=proc,
        remote_ip=raddr.ip,
        remote_port=raddr.port,
        proto=conn_protocol(conn),
        context={
            "class": pclass,
            "hash": proc_hash,
            "proc_name": proc_name
        }
    )
    if not allowed:
        block_remote_ip(raddr.ip, reason=f"policy:{policy_id or reason}")
    return allowed, reason, policy_id
# =========================
# INTRUSION DETECTOR (WIRED TO POLICY ENGINE)
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

            if raddr:
                allowed, reason, policy_id = evaluate_connection(c)
                if not allowed:
                    alerts.append({
                        "type": "zero_trust_block",
                        "remote_ip": raddr.ip,
                        "remote_port": raddr.port,
                        "pid": pid,
                        "reason": reason,
                        "policy_id": policy_id
                    })
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
# BORG GUARDIAN
# =========================

class BorgNetGuardian:
    def __init__(self):
        self.detector = AnomalyDetector()
        self.policy_expect_enabled = True
        self.intrusion_detector = IntrusionDetector()
        self.adaptive_firewall = AdaptiveFirewall()
        self.raw_sniffer = RawPacketSniffer()
        self.node_discovery = NodeDiscovery()

    def _run_intrusion_detection(self):
        alerts = self.intrusion_detector.analyze()
        for a in alerts:
            ip = a.get("remote_ip") or a.get("ip")
            if ip and a["type"] != "zero_trust_block":
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
        self.raw_sniffer.start()
        self.node_discovery.start()

        while True:
            online = is_online()
            intrusion_alerts = self._run_intrusion_detection()
            insane_alerts = self._scan_insane_behavior()
            adaptive_alerts = self.adaptive_firewall.observe()

            if intrusion_alerts or insane_alerts or adaptive_alerts:
                log_event("enforce",
                          "Anomalies detected; auto-disabling network",
                          {"intrusion_count": len(intrusion_alerts),
                           "insane_count": len(insane_alerts),
                           "adaptive_count": len(adaptive_alerts)})
                disable_all_network()
                self.policy_expect_enabled = False

            self.detector.update_and_check(self.policy_expect_enabled)
            sync_threats_with_swarm()
            time.sleep(CHECK_INTERVAL_SECONDS)

# =========================
# CORTEX MEMORY HELPERS
# =========================

def load_recent_anomalies(limit=20):
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
                if rec.get("event") in ("anomaly", "insane", "intrusion", "enforce", "policy_decision"):
                    events.append(rec)
    except Exception:
        return []
    return events[-limit:]

def load_cortex_memory(limit=50):
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
                if ev in ("harden", "intrusion", "insane", "enforce", "policy_decision", "override"):
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
# GUI INTERFACE (COCKPIT + POLICY EDITOR + ADMIN OVERRIDE)
# =========================

def launch_gui():
    import tkinter as tk
    from tkinter import scrolledtext, messagebox, filedialog
    import threading

    class BorgInterface:
        def __init__(self, root):
            self.root = root
            self.root.title("BORG OS // ZERO TRUST CONTROL NODE")
            self.root.geometry("900x675")
            self.root.configure(bg="#0a0a0a")

            self.build_ui()
            self.update_status_loop()
            self.update_telemetry()
            self.update_autoloader_status()
            self.start_voice_thread()
            self.start_connection_monitor()

        def build_ui(self):
            title = tk.Label(self.root, text="BORG ZERO TRUST CONTROL",
                             fg="#00ffea", bg="#0a0a0a",
                             font=("Consolas", 18, "bold"))
            title.pack(pady=8)

            self.status_label = tk.Label(self.root, text="Status: ...",
                                         fg="#00ffea", bg="#0a0a0a",
                                         font=("Consolas", 14))
            self.status_label.pack(pady=4)

            self.telemetry_label = tk.Label(self.root, text="CPU: ...  RAM: ...  GPU: ...",
                                            fg="#aaaaaa", bg="#0a0a0a",
                                            font=("Consolas", 11))
            self.telemetry_label.pack(pady=2)

            self.autoloader_label = tk.Label(self.root, text="Autoloader: ...",
                                             fg="#aaaaaa", bg="#0a0a0a",
                                             font=("Consolas", 11))
            self.autoloader_label.pack(pady=2)

            self.canvas = tk.Canvas(self.root, width=825, height=165,
                                    bg="#000000", highlightthickness=0)
            self.canvas.pack(pady=8)

            self.heat_cells = []
            rows, cols = 6, 28
            cell_w = 825 // cols
            cell_h = 165 // rows
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
                {"name": "NODE-ALPHA", "x": 140, "y": 80, "status": "ok"},
                {"name": "NODE-BETA", "x": 410, "y": 60, "status": "ok"},
                {"name": "NODE-GAMMA", "x": 700, "y": 120, "status": "ok"},
            ]
            self.swarm_node_items = []
            for node in self.swarm_nodes:
                item = self.canvas.create_oval(
                    node["x"]-7, node["y"]-7,
                    node["x"]+7, node["y"]+7,
                    fill="#00ffea", outline=""
                )
                label = self.canvas.create_text(
                    node["x"], node["y"]-14,
                    text=node["name"],
                    fill="#00ffea",
                    font=("Consolas", 7)
                )
                self.swarm_node_items.append((item, label))

            self.animate_overlay()

            frame = tk.Frame(self.root, bg="#0a0a0a")
            frame.pack(pady=6)

            tk.Button(frame, text="ENABLE NET",
                      command=self.enable_net,
                      bg="#003300", fg="#00ff00",
                      font=("Consolas", 12)).grid(row=0, column=0, padx=6)

            tk.Button(frame, text="DISABLE NET",
                      command=self.disable_net,
                      bg="#330000", fg="#ff4444",
                      font=("Consolas", 12)).grid(row=0, column=1, padx=6)

            tk.Button(frame, text="SELF-REPAIR",
                      command=self.self_repair,
                      bg="#333300", fg="#ffff66",
                      font=("Consolas", 12)).grid(row=0, column=2, padx=6)

            tk.Button(frame, text="TRUSTED GAMING",
                      command=self.open_trusted_gaming_zone,
                      bg="#002233", fg="#66ffff",
                      font=("Consolas", 12)).grid(row=0, column=3, padx=6)

            tk.Button(frame, text="THREAT REPLAY",
                      command=self.open_threat_replay,
                      bg="#221100", fg="#ffbb66",
                      font=("Consolas", 12)).grid(row=0, column=4, padx=6)

            tk.Button(frame, text="POLICY EDITOR",
                      command=self.open_policy_editor,
                      bg="#112233", fg="#99ddff",
                      font=("Consolas", 12)).grid(row=0, column=5, padx=6)

            tk.Button(self.root, text="VIEW LOG",
                      command=self.open_log_viewer,
                      bg="#111111", fg="#aaaaaa",
                      font=("Consolas", 12)).pack(pady=3)

            tk.Button(self.root, text="THREAT LIST",
                      command=self.open_threat_list,
                      bg="#330000", fg="#ff6666",
                      font=("Consolas", 12)).pack(pady=3)

            tk.Button(self.root, text="BLOCK EDONKEY",
                      command=self.block_p2p,
                      bg="#330022", fg="#ff66cc",
                      font=("Consolas", 12)).pack(pady=3)

            tk.Button(self.root, text="DEP HEALTH",
                      command=self.open_dep_health,
                      bg="#002222", fg="#66ffff",
                      font=("Consolas", 12)).pack(pady=3)

            tk.Button(self.root, text="NETWORK MAP",
                      command=self.open_network_map,
                      bg="#001122", fg="#66aaff",
                      font=("Consolas", 12)).pack(pady=3)

            tk.Button(self.root, text="CORTEX MEMORY",
                      command=self.open_cortex_memory,
                      bg="#112200", fg="#ccff66",
                      font=("Consolas", 12)).pack(pady=3)

            tk.Button(self.root, text="SWARM MONITOR",
                      command=self.open_swarm_monitor,
                      bg="#220011", fg="#ff99cc",
                      font=("Consolas", 12)).pack(pady=3)

            tk.Button(self.root, text="ADMIN OVERRIDE (10m)",
                      command=self.admin_override_button,
                      bg="#442200", fg="#ffdd88",
                      font=("Consolas", 12)).pack(pady=3)

            tk.Label(self.root, text="BORG CORTEX",
                     fg="#00ffea", bg="#0a0a0a",
                     font=("Consolas", 16)).pack(pady=6)

            self.cortex_output = scrolledtext.ScrolledText(
                self.root, width=90, height=7,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 11)
            )
            self.cortex_output.pack()

            self.cortex_input = tk.Entry(
                self.root,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 12)
            )
            self.cortex_input.pack(fill="x", padx=16, pady=4)
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

        def admin_override_button(self):
            enable_admin_override(600)
            messagebox.showinfo("Admin Override", "Admin override enabled for 10 minutes (risk-aware).")

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
            import tkinter as tk
            from tkinter import scrolledtext
            win = tk.Toplevel(self.root)
            win.title("Log Viewer")
            win.geometry("750x500")
            win.configure(bg="#0a0a0a")
            text = scrolledtext.ScrolledText(
                win, width=80, height=26,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 10)
            )
            text.pack()
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    text.insert("1.0", f.read())
            else:
                text.insert("1.0", "No log file found.\n")

        # ====== THREAT LIST ======
        def open_threat_list(self):
            import tkinter as tk
            from tkinter import scrolledtext
            win = tk.Toplevel(self.root)
            win.title("Threat List")
            win.geometry("650x400")
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

        # ====== P2P BLOCK ======
        def block_p2p(self):
            block_p2p_edonkey_all_ports()
            from tkinter import messagebox
            messagebox.showinfo("P2P Block", "All eDonkey/eMule traffic blocked")

        # ====== DEP HEALTH ======
        def open_dep_health(self):
            import tkinter as tk
            from tkinter import scrolledtext
            win = tk.Toplevel(self.root)
            win.title("Dependency Health")
            win.geometry("450x280")
            win.configure(bg="#0a0a0a")
            text = scrolledtext.ScrolledText(
                win, width=55, height=14,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 10)
            )
            text.pack(fill="both", expand=True)
            if not AUTOLOADER_STATUS["attempted"]:
                text.insert("1.0", "Autoloader has not run yet.\n")
            else:
                for lib, st in AUTOLOADER_STATUS["results"].items():
                    text.insert("end", f"{lib}: {st}\n")

        # ====== TRUSTED GAMING ZONE ======
        def open_trusted_gaming_zone(self):
            import tkinter as tk
            from tkinter import scrolledtext, filedialog, messagebox
            win = tk.Toplevel(self.root)
            win.title("Trusted Gaming Zone")
            win.geometry("800x550")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Trusted Gaming Zone",
                              fg="#00ffea", bg="#0a0a0a",
                              font=("Consolas", 14))
            header.pack(pady=4)

            status_label = tk.Label(win, text="Gaming zone: clear",
                                    fg="#66ff66", bg="#0a0a0a",
                                    font=("Consolas", 11))
            status_label.pack(pady=2)

            top_frame = tk.Frame(win, bg="#0a0a0a")
            top_frame.pack(fill="both", expand=True, padx=8, pady=4)

            libs_frame = tk.LabelFrame(top_frame, text="Detected libraries",
                                       fg="#66aaff", bg="#0a0a0a",
                                       font=("Consolas", 10))
            libs_frame.pack(side="left", fill="both", expand=True, padx=4, pady=4)

            libs_text = scrolledtext.ScrolledText(
                libs_frame, width=35, height=14,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 9)
            )
            libs_text.pack(fill="both", expand=True, padx=4, pady=4)

            proc_frame = tk.LabelFrame(top_frame, text="Active trusted processes",
                                       fg="#66aaff", bg="#0a0a0a",
                                       font=("Consolas", 10))
            proc_frame.pack(side="left", fill="both", expand=True, padx=4, pady=4)

            proc_text = scrolledtext.ScrolledText(
                proc_frame, width=35, height=14,
                bg="#000000", fg="#00ffea",
                font=("Consolas", 9)
            )
            proc_text.pack(fill="both", expand=True, padx=4, pady=4)

            bottom_frame = tk.LabelFrame(win, text="Manual trusted folders",
                                         fg="#66aaff", bg="#0a0a0a",
                                         font=("Consolas", 10))
            bottom_frame.pack(fill="both", expand=True, padx=8, pady=4)

            listbox = tk.Listbox(bottom_frame, width=70, height=5,
                                 bg="#000000", fg="#00ffea",
                                 font=("Consolas", 9))
            listbox.pack(side="left", fill="both", expand=True, padx=4, pady=4)

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
            btn_frame.pack(pady=4)

            tk.Button(btn_frame, text="Add trusted folder",
                      command=add_folder,
                      bg="#003300", fg="#00ff00",
                      font=("Consolas", 10)).pack(side="left", padx=4)

            tk.Button(btn_frame, text="Remove selected folder",
                      command=remove_folder,
                      bg="#330000", fg="#ff6666",
                      font=("Consolas", 10)).pack(side="left", padx=4)

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

        # ====== THREAT REPLAY ======
        def open_threat_replay(self):
            import tkinter as tk
            from tkinter import scrolledtext
            win = tk.Toplevel(self.root)
            win.title("Threat Replay")
            win.geometry("800x450")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Threat Replay (last 20 anomalies)",
                              fg="#ffbb66", bg="#0a0a0a",
                              font=("Consolas", 12))
            header.pack(pady=4)

            canvas = tk.Canvas(win, width=760, height=180,
                               bg="#000000", highlightthickness=0)
            canvas.pack(pady=6)

            text = scrolledtext.ScrolledText(
                win, width=90, height=9,
                bg="#000000", fg="#ffbb66",
                font=("Consolas", 9)
            )
            text.pack(fill="both", expand=True, padx=8, pady=4)

            events = load_recent_anomalies(limit=20)
            if not events:
                text.insert("1.0", "No recent anomalies.\n")
                return

            margin = 40
            width = 760 - 2 * margin
            y = 90
            n = len(events)
            xs = [margin + int(i * width / max(1, n - 1)) for i in range(n)] if n > 1 else [margin + width // 2]

            canvas.create_line(margin, y, 760 - margin, y, fill="#444444")

            nodes = []
            for i, (ev, x) in enumerate(zip(events, xs)):
                etype = ev.get("event")
                if etype == "intrusion":
                    color = "#ff4444"
                elif etype == "insane":
                    color = "#ff00ff"
                elif etype == "enforce":
                    color = "#ffaa00"
                elif etype == "policy_decision":
                    color = "#66ffff"
                else:
                    color = "#66aaff"
                node = canvas.create_oval(
                    x-7, y-7, x+7, y+7,
                    fill=color, outline=""
                )
                canvas.create_text(
                    x, y-16,
                    text=str(i+1),
                    fill="#ffffff",
                    font=("Consolas", 7)
                )
                nodes.append(node)

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
            import tkinter as tk
            win = tk.Toplevel(self.root)
            win.title("Network Map")
            win.geometry("900x600")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Network Map (processes & connections)",
                              fg="#66aaff", bg="#0a0a0a",
                              font=("Consolas", 12))
            header.pack(pady=4)

            canvas = tk.Canvas(win, width=860, height=520,
                               bg="#000000", highlightthickness=0)
            canvas.pack(pady=6)

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
                    430, 260,
                    text="No active inet connections.",
                    fill="#888888",
                    font=("Consolas", 12)
                )
                return

            center_x, center_y = 430, 260
            radius = 220
            pids = list(nodes.keys())
            n = len(pids)
            node_items = {}

            for i, pid in enumerate(pids):
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
                    x-9, y-9, x+9, y+9,
                    fill=color, outline=""
                )
                label = canvas.create_text(
                    x, y-16,
                    text=f"{pinfo['name']} ({pid})",
                    fill="#ffffff",
                    font=("Consolas", 7)
                )
                node_items[pid] = (node, label, x, y)

            remote_nodes = {}
            threats = load_threats()

            for pid, rip, rport in edges:
                if pid not in node_items:
                    continue
                if rip not in remote_nodes:
                    rx = center_x + radius * 1.1 * (random.random() - 0.5) * 2
                    ry = center_y + radius * 1.1 * (random.random() - 0.5) * 2
                    rnode = canvas.create_oval(
                        rx-5, ry-5, rx+5, ry+5,
                        fill="#888888", outline=""
                    )
                    rlabel = canvas.create_text(
                        rx, ry-11,
                        text=rip,
                        fill="#aaaaaa",
                        font=("Consolas", 7)
                    )
                    remote_nodes[rip] = (rnode, rlabel, rx, ry)

                node, label, x, y = node_items[pid]
                rnode, rlabel, rx, ry = remote_nodes[rip]
                is_threat = any(t["ip"] == rip for t in threats)
                ecolor = "#ff4444" if is_threat else "#4444ff"
                canvas.create_line(x, y, rx, ry, fill=ecolor)

        # ====== CORTEX MEMORY ======
        def open_cortex_memory(self):
            import tkinter as tk
            from tkinter import scrolledtext
            win = tk.Toplevel(self.root)
            win.title("Cortex Memory")
            win.geometry("800x550")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Cortex Memory — Why things were blocked",
                              fg="#ccff66", bg="#0a0a0a",
                              font=("Consolas", 12))
            header.pack(pady=4)

            text = scrolledtext.ScrolledText(
                win, width=90, height=26,
                bg="#000000", fg="#ccff66",
                font=("Consolas", 9)
            )
            text.pack(fill="both", expand=True, padx=8, pady=4)

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

        # ====== SWARM MONITOR ======
        def open_swarm_monitor(self):
            import tkinter as tk
            from tkinter import scrolledtext
            win = tk.Toplevel(self.root)
            win.title("Swarm Sync Monitor")
            win.geometry("800x550")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Swarm Sync Monitor",
                              fg="#ff99cc", bg="#0a0a0a",
                              font=("Consolas", 12))
            header.pack(pady=4)

            canvas = tk.Canvas(win, width=760, height=220,
                               bg="#000000", highlightthickness=0)
            canvas.pack(pady=6)

            text = scrolledtext.ScrolledText(
                win, width=90, height=18,
                bg="#000000", fg="#ff99cc",
                font=("Consolas", 9)
            )
            text.pack(fill="both", expand=True, padx=8, pady=4)

            local = load_threats()
            remote = swarm_pull_threats()
            nodes = load_nodes()

            all_ips = set(t["ip"] for t in local) | set(t["ip"] for t in remote)
            if not all_ips:
                canvas.create_text(
                    380, 110,
                    text="No swarm threat data yet.",
                    fill="#888888",
                    font=("Consolas", 12)
                )
            else:
                center_x, center_y = 380, 110
                radius = 90
                ips = list(all_ips)
                n = len(ips)
                for i, ip in enumerate(ips):
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
                        x-7, y-7, x+7, y+7,
                        fill=color, outline=""
                    )
                    canvas.create_text(
                        x, y-13,
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

            text.insert("end", "\nDiscovered nodes:\n")
            if not nodes:
                text.insert("end", "  (none discovered yet)\n")
            else:
                for ip in nodes:
                    text.insert("end", f"  {ip}\n")

        # ====== POLICY EDITOR ======
        def open_policy_editor(self):
            import tkinter as tk
            from tkinter import scrolledtext, messagebox
            win = tk.Toplevel(self.root)
            win.title("Zero Trust Policy Editor")
            win.geometry("800x550")
            win.configure(bg="#0a0a0a")

            header = tk.Label(win, text="Zero Trust Policy Editor",
                              fg="#99ddff", bg="#0a0a0a",
                              font=("Consolas", 12))
            header.pack(pady=4)

            text = scrolledtext.ScrolledText(
                win, width=90, height=26,
                bg="#000000", fg="#99ddff",
                font=("Consolas", 9)
            )
            text.pack(fill="both", expand=True, padx=8, pady=4)

            # load current policies
            global POLICIES
            POLICIES = load_policies()
            text.insert("1.0", json.dumps(POLICIES, indent=2))

            def save_from_editor():
                global POLICIES
                raw = text.get("1.0", "end").strip()
                try:
                    data = json.loads(raw)
                    if not isinstance(data, list):
                        raise ValueError("Policies must be a list.")
                    POLICIES = data
                    save_policies(POLICIES)
                    messagebox.showinfo("Policy Editor", "Policies saved and applied.")
                except Exception as e:
                    messagebox.showerror("Policy Editor", f"Error parsing policies: {e}")

            btn = tk.Button(win, text="Save Policies",
                            command=save_from_editor,
                            bg="#003355", fg="#99ddff",
                            font=("Consolas", 11))
            btn.pack(pady=4)

        # ====== CORTEX COMMANDS ======
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
                    "  override - show admin override status\n"
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
            elif cmd.lower() == "override":
                self.cortex_output.insert(
                    "end",
                    f"Admin override active: {is_admin_override_active()}\n"
                )
            else:
                self.cortex_output.insert("end", "Unknown command.\n")

        # ====== STUBS ======
        def start_voice_thread(self):
            pass

        def start_connection_monitor(self):
            pass

    root = tk.Tk()
    app = BorgInterface(root)
    root.mainloop()

# =========================
# MAIN ENTRY
# =========================

def main():
    block_p2p_edonkey_all_ports()
    if len(sys.argv) > 1 and sys.argv[1].lower() == "guardian":
        guardian = BorgNetGuardian()
        guardian.guardian_loop()
    else:
        launch_gui()

if __name__ == "__main__":
    main()