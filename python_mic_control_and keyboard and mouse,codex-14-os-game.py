#!/usr/bin/env python3
# ============================================================
# Codex Input Firewall — OBLIVION BUILD (REFINED v2)
# - Learning mode (observe-only)
# - Tuned thresholds (less noisy)
# - Richer logging (per-process feature dump)
# - Behavior profile export (offline JSON)
# - Adaptive thresholds (per-process)
# - Per-process trust scoring
# - Simple anomaly clustering (behavior buckets)
# ============================================================

import os
import sys
import time
import math
import json
import shutil
import string
import random
import socket
import hashlib
import traceback
import threading
import platform

IS_WINDOWS = (os.name == "nt")
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = (sys.platform == "darwin")

try:
    import psutil
except ImportError:
    print("psutil is required. Install with: pip install psutil")
    sys.exit(1)

if IS_WINDOWS:
    try:
        import winreg
    except ImportError:
        winreg = None

HAS_TK = False
try:
    import tkinter as tk
    from tkinter import scrolledtext, ttk
    HAS_TK = True
except Exception:
    HAS_TK = False

# ============================================================
# 0. AUTO-ELEVATION (Windows only)
# ============================================================
if IS_WINDOWS:
    import ctypes

    def ensure_admin():
        try:
            if not ctypes.windll.shell32.IsUserAnAdmin():
                script = os.path.abspath(sys.argv[0])
                params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable, f'"{script}" {params}', None, 1
                )
                sys.exit()
        except Exception as e:
            print("Elevation failed:", e)
            input("Press ENTER to exit…")
            sys.exit()

    ensure_admin()

# ============================================================
# 1. CONFIG & CONSTANTS
# ============================================================
if IS_WINDOWS:
    BASE_DIR = r"C:\ProgramData\CodexInputFirewall"
else:
    BASE_DIR = os.path.expanduser("~/.codex_input_firewall")

os.makedirs(BASE_DIR, exist_ok=True)

LOG_FILE = os.path.join(BASE_DIR, "firewall_log.txt")
CONFIG_FILE = os.path.join(BASE_DIR, "firewall_config.json")
QUARANTINE_DIR = os.path.join(BASE_DIR, "Quarantine")
PROFILE_EXPORT_FILE = os.path.join(BASE_DIR, "behavior_profiles.json")

os.makedirs(QUARANTINE_DIR, exist_ok=True)

MONITORED_EXECUTABLES = [
    "python", "python3", "python.exe", "pythonw.exe",
    "node", "node.exe",
    "java", "java.exe",
    "ruby", "ruby.exe",
    "perl", "perl.exe",
    "powershell.exe", "pwsh.exe",
    "wscript.exe", "cscript.exe"
]

SUSPICIOUS_MARKERS = [
    "pynput",
    "pyautogui",
    "keyboard",
    "mouse",
    "pyaudio",
    "sounddevice",
    "speech_recognition",
    "pyhook",
    "pywin32",
    "keylogger",
    "rat_client",
    "remote_input",
    "input_hook",
]

BLACKLISTED_HASHES = [
    # "deadbeef..."  # add real hashes here
]

BLOCKED_IPS = {
    # "1.2.3.4",
}
BLOCKED_PORTS = {
    4444, 1337, 5555, 6666, 8081
}

REG_BASE_MIC = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone\\NonPackaged"
RUN_KEY_PATH = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"
RUN_VALUE_NAME = "CodexInputFirewall"

DEFAULT_GAME_WHITELIST = [
    "steam.exe", "steam",
    "epicgameslauncher.exe", "epicgameslauncher",
    "battle.net.exe", "battle.net",
    "riotclientservices.exe", "riotclientservices",
    "origin.exe", "origin",
    "ea app.exe", "eaapp.exe", "ealauncher.exe",
    "gog galaxy.exe", "goggalaxy.exe",
    "uplay.exe", "ubisoftconnect.exe",
    "rockstar games launcher.exe", "rockstargameslauncher.exe",
]

STATE = {
    "firewall_enabled": False,
    "stealth_mode": False,
    "protect_all_runtimes": True,
    "kernel_driver_enabled": False,
    "whitelist": [],
    "autostart_enabled": False,
    "learning_mode": True,  # observe-only by default
}

STATE_LOCK = threading.Lock()

BEHAVIOR_DB = {}          # pid -> stats dict
BASELINE_HASH_DB = {}     # exe path -> baseline hash

# ============================================================
# 2. LOGGING & CONFIG
# ============================================================
def write_log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    return line.strip()

def load_log():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def save_config():
    try:
        with STATE_LOCK:
            data = {
                "firewall_enabled": STATE["firewall_enabled"],
                "stealth_mode": STATE["stealth_mode"],
                "protect_all_runtimes": STATE["protect_all_runtimes"],
                "kernel_driver_enabled": STATE["kernel_driver_enabled"],
                "whitelist": STATE["whitelist"],
                "autostart_enabled": STATE["autostart_enabled"],
                "learning_mode": STATE["learning_mode"],
            }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        write_log(f"[CONFIG] Save failed: {e}")

def load_config():
    if not os.path.exists(CONFIG_FILE):
        save_config()
        return
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        with STATE_LOCK:
            STATE["firewall_enabled"] = data.get("firewall_enabled", False)
            STATE["stealth_mode"] = data.get("stealth_mode", False)
            STATE["protect_all_runtimes"] = data.get("protect_all_runtimes", True)
            STATE["kernel_driver_enabled"] = data.get("kernel_driver_enabled", False)
            STATE["whitelist"] = data.get("whitelist", [])
            STATE["autostart_enabled"] = data.get("autostart_enabled", False)
            STATE["learning_mode"] = data.get("learning_mode", True)
    except Exception as e:
        write_log(f"[CONFIG] Load failed: {e}")

load_config()

def export_behavior_profiles():
    """Dump BEHAVIOR_DB to JSON for offline analysis."""
    try:
        snapshot = {}
        for pid, stats in BEHAVIOR_DB.items():
            snapshot[str(pid)] = {
                "last_check": stats.get("last_check"),
                "cpu_samples": stats.get("cpu_samples", []),
                "conn_samples": stats.get("conn_samples", []),
                "ai_score": stats.get("ai_score", 0.0),
                "suspicious": stats.get("suspicious", False),
                "features": stats.get("features", {}),
                "trust_score": stats.get("trust_score", 0.0),
                "dynamic_threshold": stats.get("dynamic_threshold", 20.0),
                "cluster_id": stats.get("cluster_id", "unknown"),
            }
        with open(PROFILE_EXPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        write_log(f"[EXPORT] Behavior profiles exported to {PROFILE_EXPORT_FILE}")
    except Exception as e:
        write_log(f"[EXPORT] Failed to export profiles: {e}")

# ============================================================
# 3. MIC FIREWALL (Windows only)
# ============================================================
def ensure_mic_parent_keys():
    if not (IS_WINDOWS and winreg):
        return
    try:
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore")
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone")
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, REG_BASE_MIC)
    except Exception:
        pass

def block_mic_global():
    if not (IS_WINDOWS and winreg):
        write_log("Mic block requested, but OS is not Windows or winreg unavailable.")
        return
    ensure_mic_parent_keys()
    try:
        key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE_MIC}\\Codex_Global_Block")
        winreg.SetValueEx(key, "Value", 0, winreg.REG_SZ, "Deny")
        write_log("Mic access DENIED for nonpackaged apps (Codex global block).")
    except Exception as e:
        write_log(f"Mic registry block failed: {e}")

def allow_mic_global():
    if not (IS_WINDOWS and winreg):
        return
    try:
        winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE_MIC}\\Codex_Global_Block")
        write_log("Mic access ALLOWED (Codex global block removed).")
    except Exception:
        pass

# ============================================================
# 4. AUTO-STARTUP SERVICE MODE (Windows only)
# ============================================================
def install_autostart():
    if not (IS_WINDOWS and winreg):
        write_log("[AUTOSTART] Not supported on this OS.")
        return
    try:
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE)
        except FileNotFoundError:
            key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, RUN_KEY_PATH)
        script = os.path.abspath(sys.argv[0])
        cmd = f'"{sys.executable}" "{script}" --stealth'
        winreg.SetValueEx(key, RUN_VALUE_NAME, 0, winreg.REG_SZ, cmd)
        write_log("[AUTOSTART] Installed Run entry for stealth mode.")
    except Exception as e:
        write_log(f"[AUTOSTART] Failed: {e}")

def remove_autostart():
    if not (IS_WINDOWS and winreg):
        return
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE)
        winreg.DeleteValue(key, RUN_VALUE_NAME)
        write_log("[AUTOSTART] Removed Run entry.")
    except FileNotFoundError:
        pass
    except OSError:
        pass
    except Exception as e:
        write_log(f"[AUTOSTART] Remove failed: {e}")

# ============================================================
# 5. WHITELIST SYSTEM
# ============================================================
def is_whitelisted(proc: psutil.Process) -> bool:
    try:
        with STATE_LOCK:
            wl_user = STATE["whitelist"][:]
        wl_default = DEFAULT_GAME_WHITELIST
        wl_all = [w.lower() for w in (wl_user + wl_default)]

        name = (proc.info.get("name") or "").lower()
        exe = (proc.info.get("exe") or "").lower()
        if name in wl_all or exe in wl_all:
            return True
    except Exception:
        pass
    return False

def add_to_whitelist(entry: str):
    entry = entry.strip()
    if not entry:
        return
    with STATE_LOCK:
        if entry not in STATE["whitelist"]:
            STATE["whitelist"].append(entry)
    save_config()
    write_log(f"[WHITELIST] Added: {entry}")

def remove_from_whitelist(entry: str):
    entry = entry.strip()
    with STATE_LOCK:
        STATE["whitelist"] = [e for e in STATE["whitelist"] if e.lower() != entry.lower()]
    save_config()
    write_log(f"[WHITELIST] Removed: {entry}")

# ============================================================
# 6. HASHING & SIGNATURES
# ============================================================
def hash_executable(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        write_log(f"[HASH] Failed to hash {path}: {e}")
        return None

def is_signature_bad(proc: psutil.Process) -> bool:
    try:
        exe = proc.info.get("exe") or ""
        exe_lower = exe.lower()
        if not exe_lower:
            return False
        h = hash_executable(exe_lower)
        if h and h in BLACKLISTED_HASHES:
            write_log(f"[SIG] Blacklisted hash match for {exe_lower}")
            return True
    except Exception:
        pass
    return False

def get_baseline_hash(exe_path: str):
    exe_path = exe_path.lower()
    if exe_path in BASELINE_HASH_DB:
        return BASELINE_HASH_DB[exe_path]
    h = hash_executable(exe_path)
    if h:
        BASELINE_HASH_DB[exe_path] = h
    return h

def inline_code_comparison(proc: psutil.Process) -> bool:
    try:
        exe = (proc.info.get("exe") or "").lower()
        if not exe or not os.path.exists(exe):
            return False
        current_hash = hash_executable(exe)
        baseline_hash = get_baseline_hash(exe)
        if baseline_hash and current_hash and baseline_hash != current_hash:
            write_log(f"[INLINE] Hash mismatch for {exe} (baseline vs current).")
            return True
    except Exception:
        pass
    return False

# ============================================================
# 7. ENTROPY / OBFUSCATION
# ============================================================
def string_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    ent = 0.0
    for c in freq.values():
        p = c / length
        ent -= p * math.log2(p)
    return ent

def looks_obfuscated_path(path: str) -> bool:
    if not path:
        return False
    base = os.path.basename(path)
    ent = string_entropy(base)
    weird_chars = sum(1 for c in base if c not in (string.ascii_letters + string.digits + "._-"))
    if ent > 4.0 and weird_chars > 2:
        return True
    return False

def looks_obfuscated_cmdline(cmd: str) -> bool:
    if not cmd:
        return False
    ent = string_entropy(cmd)
    if ent > 4.5 and len(cmd) > 80:
        return True
    return False

# ============================================================
# 8. MEMORY / DLL / HOLLOWING
# ============================================================
SUSPICIOUS_DLL_MARKERS = [
    "inject",
    "hook",
    "keylog",
    "rat",
    "remote",
]

def scan_memory_maps_for_injection(proc: psutil.Process) -> bool:
    try:
        for m in proc.memory_maps():
            path = (m.path or "").lower()
            if not path:
                continue
            if path.endswith((".dll", ".so", ".dylib")):
                if any(marker in path for marker in SUSPICIOUS_DLL_MARKERS):
                    return True
                if any(p in path for p in ["\\temp\\", "/tmp/", "\\appdata\\local\\temp\\", "\\users\\", "/home/"]):
                    return True
    except Exception:
        return False
    return False

def scan_memory_for_suspicious_strings(proc: psutil.Process) -> bool:
    try:
        for m in proc.memory_maps():
            path = (m.path or "").lower()
            for marker in SUSPICIOUS_MARKERS:
                if marker.lower() in path:
                    return True
    except Exception:
        return False
    return False

def detect_process_hollowing(proc: psutil.Process) -> bool:
    try:
        exe = (proc.info.get("exe") or "").lower()
        if not exe:
            return False
        suspicious_images = 0
        for m in proc.memory_maps():
            path = (m.path or "").lower()
            if not path:
                continue
            if (path.endswith(".exe") or path.endswith(".bin")) and path != exe:
                if any(p in path for p in ["\\temp\\", "/tmp/", "\\appdata\\local\\temp\\", "\\users\\", "/home/"]):
                    suspicious_images += 1
        return suspicious_images > 0
    except Exception:
        return False

def memory_page_hashing_stub(proc: psutil.Process) -> float:
    try:
        paths = []
        for m in proc.memory_maps():
            p = (m.path or "").lower()
            if p:
                paths.append(p)
        if not paths:
            return 0.0
        concat = "|".join(sorted(set(paths)))
        h = hashlib.sha256(concat.encode("utf-8")).hexdigest()
        ent = string_entropy(h)
        distinct = len(set(paths))
        if ent > 3.5 and distinct > 50:
            return 2.0
    except Exception:
        return 0.0
    return 0.0

def generate_memory_dump_stub(proc: psutil.Process) -> str:
    try:
        dump_id = f"dump_{proc.pid}_{int(time.time())}"
        write_log(f"[MEMDUMP] (DESIGN) Would generate memory dump for PID {proc.pid} as {dump_id}")
        return dump_id
    except Exception:
        return ""

def analyze_memory_dump_stub(dump_id: str) -> dict:
    if not dump_id:
        return {}
    score = random.uniform(0, 1)
    suspicious = score > 0.7
    write_log(f"[MEMDUMP] (DESIGN) Analyzed dump {dump_id}, suspicious={suspicious}")
    return {"dump_id": dump_id, "suspicious": suspicious, "score": score}

# ============================================================
# 9. NETWORK PROFILE
# ============================================================
def process_network_profile(proc: psutil.Process) -> dict:
    profile = {
        "connections": 0,
        "remote_ips": set(),
        "blocked_hit": False,
        "suspicious": False,
    }
    try:
        conns = proc.connections(kind="inet")
        profile["connections"] = len(conns)
        for c in conns:
            if c.raddr:
                ip = c.raddr.ip
                port = c.raddr.port
                profile["remote_ips"].add(ip)
                if ip in BLOCKED_IPS or port in BLOCKED_PORTS:
                    profile["blocked_hit"] = True
    except Exception:
        return profile

    non_local = [ip for ip in profile["remote_ips"] if not ip.startswith(("127.", "10.", "192.168.", "172.16."))]
    if len(non_local) > 0 and profile["connections"] > 5:
        profile["suspicious"] = True
    if profile["blocked_hit"]:
        profile["suspicious"] = True
    return profile

# ============================================================
# 10. ANTI-DEBUG / ANTI-VM / INLINE PATCH
# ============================================================
def detect_debugging(proc: psutil.Process) -> bool:
    debugger_names = [
        "ollydbg.exe", "x64dbg.exe", "ida.exe", "ida64.exe",
        "windbg.exe", "gdb", "gdb.exe", "immunitydebugger.exe"
    ]
    try:
        parent = proc.parent()
        if parent and (parent.name() or "").lower() in debugger_names:
            return True
    except Exception:
        pass

    try:
        for p in psutil.process_iter(["name"]):
            if (p.info.get("name") or "").lower() in debugger_names:
                return True
    except Exception:
        pass

    return False

def detect_vm_environment() -> bool:
    try:
        sysinfo = (platform.uname().release + " " +
                   platform.uname().version + " " +
                   platform.uname().machine).lower()
        vm_markers = [
            "virtualbox", "vmware", "qemu", "hyper-v", "xen"
        ]
        return any(m in sysinfo for m in vm_markers)
    except Exception:
        return False

def detect_inline_patches_stub(proc: psutil.Process) -> bool:
    try:
        exe = (proc.info.get("exe") or "").lower()
        if not exe:
            return False
        anon_exec = 0
        for m in proc.memory_maps():
            path = (m.path or "").lower()
            perms = getattr(m, "perms", "")
            if not path and ("x" in perms or "r-x" in perms):
                anon_exec += 1
        return anon_exec > 3
    except Exception:
        return False

# ============================================================
# 11. eBPF / EndpointSecurity / GPU STUBS
# ============================================================
class EBPFControllerStub:
    def __init__(self):
        self.active = False

    def attach(self):
        if IS_LINUX:
            self.active = True
            write_log("[eBPF] (DESIGN) Would attach syscall hooks via eBPF here.")
        else:
            write_log("[eBPF] Not Linux; skipping.")

    def detach(self):
        if self.active:
            write_log("[eBPF] (DESIGN) Would detach eBPF hooks here.")
        self.active = False

    def syscall_score(self, proc: psutil.Process) -> float:
        try:
            cpu = proc.cpu_percent(interval=0.0)
            if cpu > 70.0:
                return 1.5
        except Exception:
            pass
        return 0.0

class EndpointSecurityStub:
    def __init__(self):
        self.active = False

    def start(self):
        if IS_MAC:
            self.active = True
            write_log("[ES] (DESIGN) Would start macOS EndpointSecurity subscription here.")
        else:
            write_log("[ES] Not macOS; skipping.")

    def stop(self):
        if self.active:
            write_log("[ES] (DESIGN) Would stop EndpointSecurity subscription here.")
        self.active = False

    def es_score(self, proc: psutil.Process) -> float:
        return 0.0

class GPUAnomalyBackendStub:
    def __init__(self):
        self.available = False
        self._init_backend()

    def _init_backend(self):
        write_log("[GPU] (DESIGN) GPU anomaly backend stub initialized (no real GPU model).")
        self.available = False

    def score(self, features: dict) -> float:
        base = (
            features.get("cpu", 0) * 0.05 +
            features.get("conns", 0) * 0.3 +
            (4.0 if features.get("net_suspicious") else 0.0) +
            (4.0 if features.get("dll_injection") else 0.0) +
            (3.0 if features.get("hollowing") else 0.0) +
            (2.0 if features.get("inline_patch") else 0.0)
        )
        return base

EBPF_CONTROLLER = EBPFControllerStub()
ES_CONTROLLER = EndpointSecurityStub()
GPU_BACKEND = GPUAnomalyBackendStub()

# ============================================================
# 12. ADAPTIVE THRESHOLDS / TRUST / CLUSTERING
# ============================================================
def update_trust_score(stats: dict, flagged: bool):
    """
    Simple trust model:
    - start at 0
    - if not flagged, trust += 0.2 (up to +5)
    - if flagged, trust -= 1.0 (down to -5)
    """
    trust = stats.get("trust_score", 0.0)
    if flagged:
        trust -= 1.0
    else:
        trust += 0.2
    trust = max(-5.0, min(5.0, trust))
    stats["trust_score"] = trust

def compute_dynamic_threshold(stats: dict) -> float:
    """
    Adaptive threshold:
    - base = 20
    - if trust high (>2), threshold up to 26
    - if trust low (<-2), threshold down to 14
    """
    base = 20.0
    trust = stats.get("trust_score", 0.0)
    if trust > 2.0:
        base += min(6.0, (trust - 2.0) * 1.5)
    elif trust < -2.0:
        base -= min(6.0, (-trust - 2.0) * 1.5)
    base = max(10.0, min(30.0, base))
    stats["dynamic_threshold"] = base
    return base

def assign_cluster(features: dict) -> str:
    """
    Very simple anomaly clustering:
    - cluster_0: low CPU, low conn, no net_suspicious
    - cluster_1: high CPU or high conn, no net_suspicious
    - cluster_2: net_suspicious or dll/hollowing/inline
    """
    cpu = features.get("avg_cpu", 0.0)
    conn = features.get("avg_conn", 0.0)
    net_susp = features.get("net_suspicious", False)
    dll = features.get("dll_injection", False)
    hollow = features.get("hollowing", False)
    inline = features.get("inline_patch", False) or features.get("inline_code_mismatch", False)

    if net_susp or dll or hollow or inline:
        return "cluster_2_high_risk"
    if cpu > 40.0 or conn > 10.0:
        return "cluster_1_heavy_activity"
    return "cluster_0_normal"

# ============================================================
# 13. PROCESS MONITORING & AI-STYLE MODEL
# ============================================================
def process_is_monitored(proc: psutil.Process) -> bool:
    try:
        name = (proc.info.get("name") or "").lower()
        if not name:
            return False
        if STATE["protect_all_runtimes"]:
            return any(name == n.lower() for n in MONITORED_EXECUTABLES)
        else:
            return name in ["python", "python3", "python.exe", "pythonw.exe"]
    except Exception:
        return False

def process_looks_suspicious_by_markers(proc: psutil.Process) -> bool:
    if is_whitelisted(proc):
        return False
    try:
        exe = (proc.info.get("exe") or "").lower()
        cmd = " ".join(proc.cmdline()).lower()
    except Exception:
        exe = ""
        cmd = ""
    for marker in SUSPICIOUS_MARKERS:
        m = marker.lower()
        if m in cmd or m in exe:
            return True
    try:
        for m in proc.memory_maps():
            path = (m.path or "").lower()
            for marker in SUSPICIOUS_MARKERS:
                if marker.lower() in path:
                    return True
    except Exception:
        pass
    try:
        for f in proc.open_files():
            path = (f.path or "").lower()
            for marker in SUSPICIOUS_MARKERS:
                if marker.lower() in path:
                    return True
    except Exception:
        pass
    return False

def process_behavior_profile(proc: psutil.Process) -> dict:
    pid = proc.pid
    now = time.time()
    stats = BEHAVIOR_DB.get(pid, {
        "last_check": now,
        "cpu_samples": [],
        "conn_samples": [],
        "ai_score": 0.0,
        "suspicious": False,
        "features": {},
        "trust_score": 0.0,
        "dynamic_threshold": 20.0,
        "cluster_id": "unknown",
    })

    try:
        cpu = proc.cpu_percent(interval=0.0)
    except Exception:
        cpu = 0.0

    net = process_network_profile(proc)
    stats["cpu_samples"].append(cpu)
    stats["conn_samples"].append(net["connections"])
    stats["last_check"] = now

    stats["cpu_samples"] = stats["cpu_samples"][-30:]
    stats["conn_samples"] = stats["conn_samples"][-30:]

    avg_cpu = sum(stats["cpu_samples"]) / max(1, len(stats["cpu_samples"]))
    avg_conn = sum(stats["conn_samples"]) / max(1, len(stats["conn_samples"]))

    score = 0.0
    score += min(avg_cpu / 15.0, 4.0)
    score += min(avg_conn / 3.0, 4.0)
    if net["suspicious"]:
        score += 4.0
    if net["blocked_hit"]:
        score += 8.0

    try:
        exe = proc.info.get("exe") or ""
        cmd = " ".join(proc.cmdline())
    except Exception:
        exe = ""
        cmd = ""
    obf_path = looks_obfuscated_path(exe)
    obf_cmd = looks_obfuscated_cmdline(cmd)
    if obf_path:
        score += 3.0
    if obf_cmd:
        score += 3.0

    dll_inj = scan_memory_maps_for_injection(proc)
    mem_str = scan_memory_for_suspicious_strings(proc)
    hollow = detect_process_hollowing(proc)
    dbg = detect_debugging(proc)
    vm_env = detect_vm_environment()
    inline_patch = detect_inline_patches_stub(proc)
    inline_code_mismatch = inline_code_comparison(proc)
    mem_page_bump = memory_page_hashing_stub(proc)
    syscall_bump = EBPF_CONTROLLER.syscall_score(proc)
    es_bump = ES_CONTROLLER.es_score(proc)

    if dll_inj:
        score += 7.0
    if mem_str:
        score += 3.0
    if hollow:
        score += 7.0
    if dbg:
        score += 2.0
    if vm_env:
        score += 1.0
    if inline_patch:
        score += 4.0
    if inline_code_mismatch:
        score += 5.0
    score += mem_page_bump
    score += syscall_bump
    score += es_bump

    gpu_score = GPU_BACKEND.score({
        "cpu": avg_cpu,
        "conns": avg_conn,
        "net_suspicious": net["suspicious"],
        "dll_injection": dll_inj,
        "hollowing": hollow,
        "inline_patch": inline_patch or inline_code_mismatch,
    })
    score += gpu_score * 0.25

    if score >= 22.0:
        dump_id = generate_memory_dump_stub(proc)
        dump_result = analyze_memory_dump_stub(dump_id)
        if dump_result.get("suspicious"):
            score += 4.0

    features = {
        "avg_cpu": avg_cpu,
        "avg_conn": avg_conn,
        "net_suspicious": net["suspicious"],
        "net_blocked_hit": net["blocked_hit"],
        "dll_injection": dll_inj,
        "mem_strings": mem_str,
        "hollowing": hollow,
        "debugging": dbg,
        "vm_env": vm_env,
        "inline_patch": inline_patch,
        "inline_code_mismatch": inline_code_mismatch,
        "mem_page_bump": mem_page_bump,
        "syscall_bump": syscall_bump,
        "es_bump": es_bump,
        "gpu_score": gpu_score,
        "obf_path": obf_path,
        "obf_cmd": obf_cmd,
    }

    stats["ai_score"] = score
    stats["features"] = features

    # Adaptive threshold + trust
    dynamic_threshold = compute_dynamic_threshold(stats)
    stats["suspicious"] = score >= dynamic_threshold

    # Simple anomaly clustering
    cluster_id = assign_cluster(features)
    stats["cluster_id"] = cluster_id

    BEHAVIOR_DB[pid] = stats

    write_log(
        f"[PROFILE] PID {pid} score={score:.2f} thr={dynamic_threshold:.2f} "
        f"cpu={avg_cpu:.1f} conn={avg_conn:.1f} dll={dll_inj} hol={hollow} net_susp={net['suspicious']} "
        f"cluster={cluster_id} trust={stats.get('trust_score', 0.0):.2f}"
    )

    return stats

def process_is_behaviorally_suspicious(proc: psutil.Process) -> bool:
    stats = process_behavior_profile(proc)
    return stats.get("suspicious", False)

# ============================================================
# 14. QUARANTINE
# ============================================================
def quarantine_executable(proc: psutil.Process):
    try:
        exe = proc.info.get("exe") or ""
        if not exe or not os.path.exists(exe):
            return
        base = os.path.basename(exe)
        dest = os.path.join(QUARANTINE_DIR, f"{base}.{int(time.time())}.quarantined")
        shutil.move(exe, dest)
        write_log(f"[QUARANTINE] Moved {exe} -> {dest}")
    except Exception as e:
        write_log(f"[QUARANTINE] Failed to quarantine: {e}")

# ============================================================
# 15. KERNEL DRIVER STUB
# ============================================================
class KernelDriverInterface:
    def __init__(self):
        self.loaded = False

    def load_driver(self):
        self.loaded = True
        write_log("[KERNEL] (DESIGN) Kernel driver would be loaded here.")

    def unload_driver(self):
        self.loaded = False
        write_log("[KERNEL] (DESIGN) Kernel driver would be unloaded here.")

    def set_policy(self, policy: dict):
        write_log(f"[KERNEL] (DESIGN) Policy update: {policy}")

KERNEL_DRIVER = KernelDriverInterface()

def sync_kernel_driver_state():
    with STATE_LOCK:
        enabled = STATE["kernel_driver_enabled"]
    if enabled and not KERNEL_DRIVER.loaded:
        KERNEL_DRIVER.load_driver()
    elif not enabled and KERNEL_DRIVER.loaded:
        KERNEL_DRIVER.unload_driver()
    policy = {
        "firewall_enabled": STATE["firewall_enabled"],
        "protect_all_runtimes": STATE["protect_all_runtimes"],
        "learning_mode": STATE["learning_mode"],
    }
    KERNEL_DRIVER.set_policy(policy)

# ============================================================
# 16. HARD FIREWALL WATCHER
# ============================================================
class HardFirewallWatcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        write_log("[WATCHER] HardFirewallWatcher started.")
        EBPF_CONTROLLER.attach()
        ES_CONTROLLER.start()
        while self.running:
            try:
                with STATE_LOCK:
                    fw_on = STATE["firewall_enabled"]
                    learning = STATE["learning_mode"]
                if fw_on:
                    for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
                        try:
                            if not process_is_monitored(proc):
                                continue
                            if is_whitelisted(proc):
                                continue

                            bad_sig = is_signature_bad(proc)
                            bad_markers = process_looks_suspicious_by_markers(proc)
                            behavior_stats = process_behavior_profile(proc)
                            bad_behavior = behavior_stats.get("suspicious", False)
                            ai_score = behavior_stats.get("ai_score", 0.0)
                            dyn_thr = behavior_stats.get("dynamic_threshold", 20.0)

                            # Decision: flagged if above dynamic threshold or strong markers/signature
                            flagged = (
                                bad_sig or
                                (bad_markers and ai_score >= dyn_thr * 0.8) or
                                (bad_behavior and ai_score >= dyn_thr)
                            )

                            # Update trust score based on this cycle
                            update_trust_score(behavior_stats, flagged)

                            if flagged:
                                if learning:
                                    write_log(
                                        f"[LEARNING] Would kill PID {proc.pid} ({proc.info.get('name')}) "
                                        f"(sig={bad_sig}, markers={bad_markers}, behavior={bad_behavior}, "
                                        f"ai_score={ai_score:.2f}, thr={dyn_thr:.2f})"
                                    )
                                else:
                                    write_log(
                                        f"[FIREWALL] Killing suspicious process PID {proc.pid} ({proc.info.get('name')}) "
                                        f"(sig={bad_sig}, markers={bad_markers}, behavior={bad_behavior}, "
                                        f"ai_score={ai_score:.2f}, thr={dyn_thr:.2f})"
                                    )
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                                    quarantine_executable(proc)
                        except psutil.NoSuchProcess:
                            continue
                        except Exception as e:
                            write_log(f"[WATCHER] Error inspecting process: {e}")
                time.sleep(1.0)
            except Exception as e:
                write_log(f"[WATCHER] Loop error: {e}\n{traceback.format_exc()}")
                time.sleep(2.0)
        EBPF_CONTROLLER.detach()
        ES_CONTROLLER.stop()
        write_log("[WATCHER] HardFirewallWatcher stopped.")

WATCHER = HardFirewallWatcher()
WATCHER.start()

# ============================================================
# 17. GUI COCKPIT
# ============================================================
class CodexCockpitGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Codex Input Firewall — OBLIVION Cockpit (Refined v2)")
        self.root.geometry("980x720")
        self.root.configure(bg="#101010")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", background="#202020", foreground="#EEEEEE")
        style.configure("TCheckbutton", background="#101010", foreground="#EEEEEE")
        style.configure("TLabel", background="#101010", foreground="#EEEEEE")

        self.build_layout()
        self.refresh_ui()
        self.schedule_log_refresh()

    def build_layout(self):
        self.status_label = tk.Label(
            self.root,
            text="Firewall: UNKNOWN",
            font=("Consolas", 18, "bold"),
            bg="#101010",
            fg="#FF5555"
        )
        self.status_label.pack(pady=10, fill="x")

        controls = tk.Frame(self.root, bg="#101010")
        controls.pack(pady=5, fill="x")

        self.btn_enable = ttk.Button(controls, text="Enable Firewall", command=self.enable_firewall)
        self.btn_enable.pack(side="left", padx=5)

        self.btn_disable = ttk.Button(controls, text="Disable Firewall", command=self.disable_firewall)
        self.btn_disable.pack(side="left", padx=5)

        self.btn_stealth = ttk.Button(controls, text="Toggle Stealth", command=self.toggle_stealth)
        self.btn_stealth.pack(side="left", padx=5)

        self.btn_refresh = ttk.Button(controls, text="Refresh Log", command=self.refresh_log_box)
        self.btn_refresh.pack(side="left", padx=5)

        self.btn_export = ttk.Button(controls, text="Export Profiles", command=self.export_profiles)
        self.btn_export.pack(side="left", padx=5)

        checks = tk.Frame(self.root, bg="#101010")
        checks.pack(pady=5, fill="x")

        self.var_protect_all = tk.BooleanVar(value=STATE["protect_all_runtimes"])
        self.chk_protect_all = ttk.Checkbutton(
            checks,
            text="Protect all runtimes (Python, Node, Java, etc.)",
            variable=self.var_protect_all,
            command=self.toggle_protect_all
        )
        self.chk_protect_all.pack(side="left", padx=5)

        self.var_kernel = tk.BooleanVar(value=STATE["kernel_driver_enabled"])
        self.chk_kernel = ttk.Checkbutton(
            checks,
            text="Kernel driver (design stub)",
            variable=self.var_kernel,
            command=self.toggle_kernel_driver
        )
        self.chk_kernel.pack(side="left", padx=5)

        self.var_autostart = tk.BooleanVar(value=STATE["autostart_enabled"])
        self.chk_autostart = ttk.Checkbutton(
            checks,
            text="Auto-start at boot (stealth mode, Windows only)",
            variable=self.var_autostart,
            command=self.toggle_autostart
        )
        self.chk_autostart.pack(side="left", padx=5)

        self.var_learning = tk.BooleanVar(value=STATE["learning_mode"])
        self.chk_learning = ttk.Checkbutton(
            checks,
            text="Learning mode (observe-only, no kills)",
            variable=self.var_learning,
            command=self.toggle_learning_mode
        )
        self.chk_learning.pack(side="left", padx=5)

        wl_frame = tk.LabelFrame(self.root, text="Whitelist (names or full paths)", bg="#101010", fg="#EEEEEE")
        wl_frame.pack(padx=10, pady=5, fill="x")

        self.wl_entry = tk.Entry(wl_frame, bg="#202020", fg="#EEEEEE", insertbackground="#EEEEEE")
        self.wl_entry.pack(side="left", padx=5, fill="x", expand=True)

        self.btn_wl_add = ttk.Button(wl_frame, text="Add", command=self.add_whitelist_entry)
        self.btn_wl_add.pack(side="left", padx=5)

        self.btn_wl_remove = ttk.Button(wl_frame, text="Remove", command=self.remove_whitelist_entry)
        self.btn_wl_remove.pack(side="left", padx=5)

        self.wl_listbox = tk.Listbox(wl_frame, bg="#181818", fg="#EEEEEE", height=8)
        self.wl_listbox.pack(padx=5, pady=5, fill="x")
        self.refresh_whitelist_listbox()

        self.log_box = scrolledtext.ScrolledText(
            self.root,
            wrap="word",
            bg="#181818",
            fg="#DDDDDD",
            insertbackground="#DDDDDD"
        )
        self.log_box.pack(expand=True, fill="both", padx=10, pady=10)

    def refresh_ui(self):
        with STATE_LOCK:
            fw_on = STATE["firewall_enabled"]
            stealth = STATE["stealth_mode"]
            learning = STATE["learning_mode"]
        mode_str = "LEARNING" if learning else "ENFORCING"
        if fw_on:
            self.status_label.config(
                text=f"Firewall: ENABLED ({mode_str}, {'STEALTH' if stealth else 'VISIBLE'})",
                fg="#FF5555"
            )
        else:
            self.status_label.config(
                text="Firewall: DISABLED",
                fg="#55FF55"
            )

        self.var_protect_all.set(STATE["protect_all_runtimes"])
        self.var_kernel.set(STATE["kernel_driver_enabled"])
        self.var_autostart.set(STATE["autostart_enabled"])
        self.var_learning.set(STATE["learning_mode"])

    def refresh_log_box(self):
        content = load_log()
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", content)

    def schedule_log_refresh(self):
        self.refresh_log_box()
        self.refresh_ui()
        self.root.after(2000, self.schedule_log_refresh)

    def refresh_whitelist_listbox(self):
        self.wl_listbox.delete(0, "end")
        with STATE_LOCK:
            for entry in STATE["whitelist"]:
                self.wl_listbox.insert("end", f"[USER] {entry}")
        for entry in DEFAULT_GAME_WHITELIST:
            self.wl_listbox.insert("end", f"[GAME] {entry}")

    # ---------- Actions ----------
    def enable_firewall(self):
        with STATE_LOCK:
            STATE["firewall_enabled"] = True
        block_mic_global()
        save_config()
        sync_kernel_driver_state()
        write_log("[COCKPIT] Firewall ENABLED from GUI.")
        self.refresh_ui()

    def disable_firewall(self):
        with STATE_LOCK:
            STATE["firewall_enabled"] = False
        allow_mic_global()
        save_config()
        sync_kernel_driver_state()
        write_log("[COCKPIT] Firewall DISABLED from GUI.")
        self.refresh_ui()

    def toggle_stealth(self):
        with STATE_LOCK:
            STATE["stealth_mode"] = not STATE["stealth_mode"]
        save_config()
        write_log(f"[COCKPIT] Stealth mode set to {STATE['stealth_mode']}.")
        self.refresh_ui()

    def toggle_protect_all(self):
        with STATE_LOCK:
            STATE["protect_all_runtimes"] = self.var_protect_all.get()
        save_config()
        sync_kernel_driver_state()
        write_log(f"[COCKPIT] Protect all runtimes set to {STATE['protect_all_runtimes']}.")
        self.refresh_ui()

    def toggle_kernel_driver(self):
        with STATE_LOCK:
            STATE["kernel_driver_enabled"] = self.var_kernel.get()
        save_config()
        sync_kernel_driver_state()
        write_log(f"[COCKPIT] Kernel driver flag set to {STATE['kernel_driver_enabled']} (design stub).")
        self.refresh_ui()

    def toggle_autostart(self):
        with STATE_LOCK:
            STATE["autostart_enabled"] = self.var_autostart.get()
        if STATE["autostart_enabled"]:
            install_autostart()
        else:
            remove_autostart()
        save_config()
        self.refresh_ui()

    def toggle_learning_mode(self):
        with STATE_LOCK:
            STATE["learning_mode"] = self.var_learning.get()
        save_config()
        write_log(f"[COCKPIT] Learning mode set to {STATE['learning_mode']}.")
        self.refresh_ui()

    def add_whitelist_entry(self):
        entry = self.wl_entry.get().strip()
        if not entry:
            return
        add_to_whitelist(entry)
        self.refresh_whitelist_listbox()
        self.wl_entry.delete(0, "end")

    def remove_whitelist_entry(self):
        sel = self.wl_listbox.curselection()
        if sel:
            raw = self.wl_listbox.get(sel[0])
            entry = raw.replace("[USER]", "").replace("[GAME]", "").strip()
            remove_from_whitelist(entry)
        else:
            entry = self.wl_entry.get().strip()
            if entry:
                remove_from_whitelist(entry)
        self.refresh_whitelist_listbox()

    def export_profiles(self):
        export_behavior_profiles()
        self.refresh_log_box()

# ============================================================
# 18. STEALTH MODE ENTRY
# ============================================================
def run_stealth_daemon():
    write_log("[STEALTH] Codex Input Firewall running in stealth mode (no GUI).")
    sync_kernel_driver_state()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        pass

# ============================================================
# 19. MAIN
# ============================================================
def main():
    if "--stealth" in sys.argv:
        with STATE_LOCK:
            STATE["stealth_mode"] = True
            STATE["firewall_enabled"] = True
        save_config()
        block_mic_global()
        sync_kernel_driver_state()
        run_stealth_daemon()
        return

    if HAS_TK:
        root = tk.Tk()
        gui = CodexCockpitGUI(root)
        sync_kernel_driver_state()
        root.mainloop()
    else:
        write_log("[MAIN] No GUI available; running in console/stealth-like mode.")
        with STATE_LOCK:
            STATE["firewall_enabled"] = True
        save_config()
        sync_kernel_driver_state()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_log(f"[FATAL] {e}\n{traceback.format_exc()}")
        print("Codex Input Firewall crashed. See log for details.")
        input("Press ENTER to exit…")
