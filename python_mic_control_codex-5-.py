# ============================================================
# 0. AUTO-ELEVATION (kept, but GUI loads even if elevation fails)
# ============================================================
import ctypes, os, sys

def ensure_admin():
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}" {params}', None, 1
            )
            sys.exit()
    except:
        pass

ensure_admin()

# ============================================================
# 1. GUI FIRST — NOTHING BEFORE THIS POINT CAN BLOCK
# ============================================================
import tkinter as tk
from tkinter import scrolledtext

root = tk.Tk()
root.title("Python Microphone Control — Codex Panel (Tkinter)")
root.geometry("700x500")

status_label = tk.Label(root, text="Loading…", font=("Consolas", 18, "bold"))
status_label.pack(pady=10, fill="x")

btn_frame = tk.Frame(root)
btn_frame.pack()

log_box = scrolledtext.ScrolledText(root, wrap="word")
log_box.pack(expand=True, fill="both", padx=10, pady=10)

# Buttons (wired later)
btn_block = tk.Button(btn_frame, text="Disable Python Mic", width=18)
btn_block.pack(side="left", padx=5)

btn_allow = tk.Button(btn_frame, text="Enable Python Mic", width=18)
btn_allow.pack(side="left", padx=5)

btn_watchdog = tk.Button(btn_frame, text="Toggle Watchdog", width=18)
btn_watchdog.pack(side="left", padx=5)

# GUI is now visible — NOTHING can stop it from showing.


# ============================================================
# 2. BACKGROUND SYSTEMS (loaded AFTER GUI appears)
# ============================================================
import threading, time, psutil, winreg

LOG_FILE = r"C:\ProgramData\PythonMicControl\python_mic_log.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

PYTHON_PATHS = [
    r"C:\Users\<you>\AppData\Local\Programs\Python\Python3x\python.exe",
    r"C:\Users\<you>\anaconda3\python.exe",
    r"C:\Python3x\python.exe"
]

REG_BASE = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone\\NonPackaged"

SYSTEM_STATUS = {"recommended_mode": "registry"}

def write_log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except:
        pass
    return line.strip()

def refresh_log():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        log_box.delete("1.0", "end")
        log_box.insert("1.0", content)
    except:
        pass
    root.after(1000, refresh_log)

def safe_system_scan():
    global SYSTEM_STATUS
    result = {"recommended_mode": "registry"}
    try:
        winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone")
        result["recommended_mode"] = "registry"
    except:
        result["recommended_mode"] = "watchdog_only"
    SYSTEM_STATUS = result
    write_log(f"System Scan: {result}")

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
    for path in PYTHON_PATHS:
        if os.path.exists(path):
            safe = path.replace("\\", "_")
            try:
                key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\{safe}")
                winreg.SetValueEx(key, "Value", 0, winreg.REG_SZ, "Deny")
                write_log(f"Mic DENIED for {path}")
            except:
                write_log("Registry write failed")

def allow_python_mic():
    for path in PYTHON_PATHS:
        safe = path.replace("\\", "_")
        try:
            winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\{safe}")
            write_log(f"Mic ALLOWED for {path}")
        except:
            pass

def get_status():
    for path in PYTHON_PATHS:
        safe = path.replace("\\", "_")
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\{safe}")
            val, _ = winreg.QueryValueEx(key, "Value")
            if val == "Deny":
                return "Blocked"
        except:
            pass
    return "Allowed"

class Watchdog(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.enabled = True

    def run(self):
        while True:
            if self.enabled and get_status() == "Blocked":
                for proc in psutil.process_iter(['pid','name']):
                    if proc.info['name'] and proc.info['name'].lower() == "python.exe":
                        write_log(f"[WATCHDOG] Killing python PID {proc.pid}")
                        proc.kill()
            time.sleep(2)

class DeviceInterceptor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.enabled = False

    def run(self):
        while True:
            if self.enabled:
                for proc in psutil.process_iter(['pid','name']):
                    if proc.info['name'] and proc.info['name'].lower() == "python.exe":
                        write_log(f"[INTERCEPTOR] Killing python PID {proc.pid}")
                        proc.kill()
            time.sleep(1)

watchdog = Watchdog()
interceptor = DeviceInterceptor()

# ============================================================
# 3. BUTTON LOGIC (wired AFTER background systems exist)
# ============================================================

def update_status():
    status = get_status()
    if status == "Blocked":
        status_label.config(text="Python Mic: BLOCKED", fg="red")
    else:
        status_label.config(text="Python Mic: ALLOWED", fg="lime")

def disable_mic():
    mode = SYSTEM_STATUS["recommended_mode"]
    if mode == "registry":
        block_python_mic()
    else:
        interceptor.enabled = True
        write_log("Device interceptor ENABLED")
    update_status()

def enable_mic():
    allow_python_mic()
    interceptor.enabled = False
    write_log("Device interceptor DISABLED")
    update_status()

def toggle_watchdog():
    watchdog.enabled = not watchdog.enabled
    write_log(f"Watchdog {'ENABLED' if watchdog.enabled else 'DISABLED'}")

btn_block.config(command=disable_mic)
btn_allow.config(command=enable_mic)
btn_watchdog.config(command=toggle_watchdog)

# ============================================================
# 4. START BACKGROUND THREADS AFTER GUI IS VISIBLE
# ============================================================

watchdog.start()
interceptor.start()
threading.Thread(target=safe_system_scan, daemon=True).start()
refresh_log()
update_status()

# ============================================================
# 5. START GUI LOOP
# ============================================================

root.mainloop()
