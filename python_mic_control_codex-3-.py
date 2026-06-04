# === AUTO-ELEVATION CHECK (MUST BE FIRST) ===
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
        sys.exit(1)

ensure_admin()
print("[Codex] Mic Control (Tkinter) starting...")

# ============================================================
# Tier‑4 Codex Autoloader (no GUI deps)
# ============================================================

import time
import subprocess
import importlib
import json
import threading

REQUIRED_LIBS = {
    "psutil": {
        "pip": "psutil",
        "threat_level": "low"
    }
    # winreg, tkinter are built-in
}

CODEX_NODE_SYNC_FILE = os.path.join(
    os.path.expanduser("~"),
    ".codex_purge_shell",
    "python_mic_node_sync.json"
)

GLYPHS = ["◐", "◓", "◑", "◒"]


def codex_log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[CODEX {ts}] {msg}"
    print(line)
    return line


def ensure_node_sync_dir():
    d = os.path.dirname(CODEX_NODE_SYNC_FILE)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def write_node_sync(status):
    ensure_node_sync_dir()
    try:
        data = {
            "node": "python_mic_control_tk",
            "status": status,
            "timestamp": time.time()
        }
        with open(CODEX_NODE_SYNC_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        codex_log(f"Node-sync write failed: {e}")


def threat_aware_validate(lib_name):
    meta = REQUIRED_LIBS.get(lib_name, {})
    threat = meta.get("threat_level", "unknown")
    if threat not in ("low", "medium"):
        codex_log(f"THREAT ALERT: {lib_name} threat level {threat}")
    else:
        codex_log(f"Threat check OK for {lib_name} ({threat})")


def install_lib(pip_name):
    try:
        codex_log(f"Installing dependency via pip: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
    except Exception as e:
        codex_log(f"Dependency install failed for {pip_name}: {e}")


def tier4_autoload():
    write_node_sync("autoload_start")
    glyph_index = 0
    for lib, meta in REQUIRED_LIBS.items():
        threat_aware_validate(lib)
        try:
            importlib.import_module(lib)
            codex_log(f"{GLYPHS[glyph_index % len(GLYPHS)]} Resurrection check: {lib} already present")
        except ImportError:
            codex_log(f"{GLYPHS[glyph_index % len(GLYPHS)]} Resurrection needed: {lib} missing")
            install_lib(meta["pip"])
            try:
                importlib.invalidate_caches()
                importlib.import_module(lib)
                codex_log(f"{GLYPHS[glyph_index % len(GLYPHS)]} Resurrection complete: {lib} loaded")
            except ImportError as e:
                codex_log(f"Failed to load {lib} after install: {e}")
        glyph_index += 1
    write_node_sync("autoload_complete")


try:
    tier4_autoload()
except Exception as e:
    codex_log(f"Autoloader error (non-fatal): {e}")

# ============================================================
# Heavy deps
# ============================================================

import winreg
import psutil
import tkinter as tk
from tkinter import scrolledtext

# ============================================================
# CODEX SYSTEM SCANNER — AUTO-DETECT MIC BLOCKING METHOD
# ============================================================

def codex_system_scan():
    result = {
        "win32_privacy_supported": False,
        "nonpackaged_exists": False,
        "uwp_privacy_model": False,
        "python_uses_lowlevel_audio": False,
        "registry_block_allowed": False,
        "recommended_mode": "unknown"
    }

    try:
        winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
        )
        result["win32_privacy_supported"] = True
    except:
        result["win32_privacy_supported"] = False

    try:
        winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone\NonPackaged"
        )
        result["nonpackaged_exists"] = True
    except:
        result["nonpackaged_exists"] = False

    try:
        out = subprocess.check_output("tasklist /m python.exe", shell=True, text=True)
        lowlevel = ["audioses.dll", "mmdevapi.dll", "dsound.dll", "wdmaud.drv"]
        if any(dll.lower() in out.lower() for dll in lowlevel):
            result["python_uses_lowlevel_audio"] = True
    except:
        result["python_uses_lowlevel_audio"] = False

    if result["win32_privacy_supported"] and result["nonpackaged_exists"]:
        result["registry_block_allowed"] = True

    if not result["win32_privacy_supported"] and not result["nonpackaged_exists"]:
        result["uwp_privacy_model"] = True

    if result["registry_block_allowed"]:
        result["recommended_mode"] = "registry"
    elif result["uwp_privacy_model"]:
        result["recommended_mode"] = "uwp_broker"
    elif result["python_uses_lowlevel_audio"]:
        result["recommended_mode"] = "device_interceptor"
    else:
        result["recommended_mode"] = "watchdog_only"

    codex_log(f"System Scan Result: {result}")
    return result


SYSTEM_STATUS = codex_system_scan()

# ============================================================
# CONFIG
# ============================================================

PYTHON_PATHS = [
    r"C:\Users\<you>\AppData\Local\Programs\Python\Python3x\python.exe",
    r"C:\Users\<you>\anaconda3\python.exe",
    r"C:\Python3x\python.exe"
]

REG_BASE = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone\NonPackaged"
LOG_FILE = r"C:\ProgramData\PythonMicControl\python_mic_log.txt"

log_dir = os.path.dirname(LOG_FILE)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

def write_log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[Codex Log Error] {e}")
    return line.strip()


# ============================================================
# REGISTRY HELPERS
# ============================================================

def ensure_parent_keys():
    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore"
    mic = base + r"\microphone"
    nonpack = mic + r"\NonPackaged"
    try:
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, base)
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, mic)
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, nonpack)
    except Exception as e:
        write_log(f"Parent key creation error: {e}")


def block_python_mic():
    ensure_parent_keys()
    for path in PYTHON_PATHS:
        if os.path.exists(path):
            safe = path.replace("\\", "_")
            try:
                key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\{safe}")
                winreg.SetValueEx(key, "Value", 0, winreg.REG_SZ, "Deny")
                winreg.SetValueEx(key, "LastUsedTimeStart", 0, winreg.REG_QWORD, 0)
                winreg.SetValueEx(key, "LastUsedTimeStop", 0, winreg.REG_QWORD, 0)
                write_log(f"Mic access DENIED for {path}")
            except PermissionError:
                write_log("ERROR: Run as Administrator")


def allow_python_mic():
    for path in PYTHON_PATHS:
        safe = path.replace("\\", "_")
        try:
            winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\{safe}")
            write_log(f"Mic access ALLOWED for {path}")
        except FileNotFoundError:
            pass
        except PermissionError:
            write_log("ERROR: Run as Administrator")


def get_status():
    for path in PYTHON_PATHS:
        safe = path.replace("\\", "_")
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\{safe}")
            val, _ = winreg.QueryValueEx(key, "Value")
            if val == "Deny":
                return "Blocked"
        except FileNotFoundError:
            continue
    return "Allowed"


# ============================================================
# WATCHDOG THREAD
# ============================================================

class Watchdog(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.enabled = True

    def run(self):
        while True:
            try:
                if self.enabled and get_status() == "Blocked":
                    for proc in psutil.process_iter(['pid', 'name', 'exe']):
                        try:
                            if proc.info['name'] and proc.info['name'].lower() == "python.exe":
                                write_log(f"[WATCHDOG] Killing python PID {proc.pid} at {proc.info.get('exe')}")
                                proc.kill()
                        except Exception:
                            pass
                time.sleep(2)
            except Exception as e:
                write_log(f"[WATCHDOG ERROR] {e}")
                time.sleep(2)


# ============================================================
# DEVICE-LEVEL MIC INTERCEPTOR
# ============================================================

class DeviceLevelInterceptor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.enabled = False

    def run(self):
        while True:
            try:
                if self.enabled:
                    for proc in psutil.process_iter(['pid', 'name', 'exe']):
                        try:
                            if proc.info['name'] and proc.info['name'].lower() == "python.exe":
                                write_log(f"[DEVICE-INTERCEPTOR] Killing python PID {proc.pid} at {proc.info.get('exe')}")
                                proc.kill()
                        except Exception:
                            pass
                time.sleep(1.0)
            except Exception as e:
                write_log(f"[DEVICE-INTERCEPTOR ERROR] {e}")
                time.sleep(1.0)


# ============================================================
# TKINTER GUI
# ============================================================

class MicControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Microphone Control — Codex Panel (Tkinter)")
        self.root.geometry("700x500")

        self.status_label = tk.Label(
            root,
            text="",
            font=("Consolas", 18, "bold"),
            anchor="center"
        )
        self.status_label.pack(pady=10, fill="x")

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.btn_block = tk.Button(
            btn_frame,
            text="Disable Python Mic",
            width=18,
            command=self.disable_mic
        )
        self.btn_block.pack(side="left", padx=5)

        self.btn_allow = tk.Button(
            btn_frame,
            text="Enable Python Mic",
            width=18,
            command=self.enable_mic
        )
        self.btn_allow.pack(side="left", padx=5)

        self.btn_watchdog = tk.Button(
            btn_frame,
            text="Toggle Watchdog",
            width=18,
            command=self.toggle_watchdog
        )
        self.btn_watchdog.pack(side="left", padx=5)

        self.log_box = scrolledtext.ScrolledText(root, wrap="word", state="normal")
        self.log_box.pack(expand=True, fill="both", padx=10, pady=10)

        # Load existing log
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    self.log_box.insert("1.0", f.read())
            except Exception as e:
                self.log_box.insert("end", f"[Log Load Error] {e}\n")

        # Threads
        self.watchdog = Watchdog()
        self.watchdog.start()

        self.device_interceptor = DeviceLevelInterceptor()
        self.device_interceptor.start()

        self.update_status()
        self.poll_log()

    def poll_log(self):
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                self.log_box.delete("1.0", "end")
                self.log_box.insert("1.0", content)
        except Exception as e:
            # Don't spam logs here; just show once in UI
            pass
        self.root.after(1000, self.poll_log)

    def update_status(self):
        status = get_status()
        if status == "Blocked":
            self.status_label.config(text="Python Mic: BLOCKED", fg="red")
        else:
            self.status_label.config(text="Python Mic: ALLOWED", fg="lime")

    def disable_mic(self):
        mode = SYSTEM_STATUS.get("recommended_mode", "watchdog_only")

        if mode == "registry":
            block_python_mic()
            write_log("Registry mic block applied.")
        elif mode == "uwp_broker":
            write_log("UWP privacy model detected — registry mic block ignored. Using watchdog-only mode.")
        elif mode == "device_interceptor":
            write_log("Device-level interceptor mode: killing any python.exe while blocked.")
            self.device_interceptor.enabled = True
        else:
            write_log("Fallback: watchdog-only mode active.")

        self.update_status()

    def enable_mic(self):
        allow_python_mic()
        self.device_interceptor.enabled = False
        write_log("Device-level interceptor disabled (Python allowed).")
        self.update_status()

    def toggle_watchdog(self):
        self.watchdog.enabled = not self.watchdog.enabled
        state = "ENABLED" if self.watchdog.enabled else "DISABLED"
        write_log(f"Watchdog {state}")


# ============================================================
# MAIN ENTRY
# ============================================================

def main():
    root = tk.Tk()
    app = MicControlGUI(root)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[Codex FATAL ERROR]")
        print(e)
        input("Press ENTER to close…")
