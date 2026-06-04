# =========================
# 0. AUTO-ELEVATION
# =========================
import ctypes, os, sys

def ensure_admin():
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}"', None, 1
            )
            sys.exit()
    except Exception as e:
        print("Elevation failed:", e)
        input("Press ENTER to exit…")
        sys.exit()

ensure_admin()

# =========================
# 1. GUI FIRST
# =========================
import tkinter as tk
from tkinter import scrolledtext

root = tk.Tk()
root.title("Codex Input Firewall — HARD Mode")
root.geometry("780x540")

status_label = tk.Label(root, text="Firewall: UNKNOWN", font=("Consolas", 18, "bold"))
status_label.pack(pady=10, fill="x")

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

btn_enable = tk.Button(btn_frame, text="Enable Firewall", width=18)
btn_enable.pack(side="left", padx=5)

btn_disable = tk.Button(btn_frame, text="Disable Firewall", width=18)
btn_disable.pack(side="left", padx=5)

btn_refresh = tk.Button(btn_frame, text="Refresh Log", width=18)
btn_refresh.pack(side="left", padx=5)

log_box = scrolledtext.ScrolledText(root, wrap="word")
log_box.pack(expand=True, fill="both", padx=10, pady=10)

# =========================
# 2. BACKGROUND SYSTEMS
# =========================
import time, threading, psutil, winreg

LOG_FILE = r"C:\ProgramData\CodexInputFirewall\firewall_log.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

PYTHON_NAMES = ["python.exe", "pythonw.exe"]

# Libraries that usually mean input/mic control
SUSPICIOUS_MARKERS = [
    "pynput",
    "pyautogui",
    "keyboard",
    "mouse",
    "pyaudio",
    "sounddevice",
    "speech_recognition",
]

REG_BASE = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone\\NonPackaged"

FIREWALL_STATE = {"enabled": False}

def write_log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except:
        pass
    return line.strip()

def load_log_into_box():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        log_box.delete("1.0", "end")
        log_box.insert("1.0", content)
    except:
        pass

def ensure_parent_keys():
    try:
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore")
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone")
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, REG_BASE)
    except:
        pass

def block_python_mic():
    ensure_parent_keys()
    try:
        key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\Python_Global_Block")
        winreg.SetValueEx(key, "Value", 0, winreg.REG_SZ, "Deny")
        write_log("Mic access DENIED for Python (global nonpackaged block).")
    except:
        write_log("Mic registry block failed.")

def allow_python_mic():
    try:
        winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\Python_Global_Block")
        write_log("Mic access ALLOWED for Python (global block removed).")
    except:
        pass

def update_status_label():
    if FIREWALL_STATE["enabled"]:
        status_label.config(
            text="Firewall: ENABLED (Input-capable Python will be killed)",
            fg="red"
        )
    else:
        status_label.config(
            text="Firewall: DISABLED (Python allowed, no input filtering)",
            fg="lime"
        )

class HardFirewallWatcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def process_looks_suspicious(self, proc: psutil.Process) -> bool:
        # Check name first
        name = proc.info.get("name") or ""
        if not name.lower() in PYTHON_NAMES:
            return False

        # Check command line for suspicious markers
        try:
            cmd = " ".join(proc.cmdline()).lower()
            for marker in SUSPICIOUS_MARKERS:
                if marker.lower() in cmd:
                    return True
        except:
            pass

        # Check loaded files / memory maps for suspicious libs
        try:
            for m in proc.memory_maps():
                path = (m.path or "").lower()
                for marker in SUSPICIOUS_MARKERS:
                    if marker.lower() in path:
                        return True
        except:
            pass

        return False

    def run(self):
        write_log("HardFirewallWatcher thread started.")
        while self.running:
            if FIREWALL_STATE["enabled"]:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if self.process_looks_suspicious(proc):
                            write_log(f"[FIREWALL] Killing suspicious Python PID {proc.pid}")
                            proc.kill()
                    except:
                        pass
            time.sleep(1.0)
        write_log("HardFirewallWatcher thread stopped.")

watcher = HardFirewallWatcher()
watcher.start()

# =========================
# 3. GUI LOGIC
# =========================
def enable_firewall():
    FIREWALL_STATE["enabled"] = True
    block_python_mic()
    write_log("Firewall ENABLED: suspicious Python processes will be terminated.")
    update_status_label()
    load_log_into_box()

def disable_firewall():
    FIREWALL_STATE["enabled"] = False
    allow_python_mic()
    write_log("Firewall DISABLED: Python processes allowed (no input filtering).")
    update_status_label()
    load_log_into_box()

def refresh_log():
    load_log_into_box()

btn_enable.config(command=enable_firewall)
btn_disable.config(command=disable_firewall)
btn_refresh.config(command=refresh_log)

update_status_label()
load_log_into_box()

# =========================
# 4. MAIN LOOP
# =========================
try:
    root.mainloop()
finally:
    watcher.running = False
    time.sleep(0.5)
