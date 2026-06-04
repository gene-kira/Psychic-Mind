# ============================================================
# 0. AUTO-ELEVATION
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
    except Exception as e:
        print("Elevation failed:", e)
        input("Press ENTER to exit…")
        sys.exit()

ensure_admin()

# ============================================================
# 1. IMPORTS
# ============================================================
import time
import threading
import psutil
import winreg
import json
import traceback

import tkinter as tk
from tkinter import scrolledtext, ttk

# ============================================================
# 2. CONFIG & CONSTANTS
# ============================================================
BASE_DIR = r"C:\ProgramData\CodexInputFirewall"
LOG_FILE = os.path.join(BASE_DIR, "firewall_log.txt")
CONFIG_FILE = os.path.join(BASE_DIR, "firewall_config.json")

os.makedirs(BASE_DIR, exist_ok=True)

# Runtimes we care about (protect beyond Python)
MONITORED_EXECUTABLES = [
    "python.exe", "pythonw.exe",
    "node.exe", "java.exe",
    "ruby.exe", "perl.exe",
    "powershell.exe", "pwsh.exe",
    "wscript.exe", "cscript.exe"
]

# Libraries / markers that strongly suggest input/mic control
SUSPICIOUS_MARKERS = [
    # Python / scripting input libs
    "pynput",
    "pyautogui",
    "keyboard",
    "mouse",
    "pyaudio",
    "sounddevice",
    "speech_recognition",
    "pyhook",
    "pywin32",
    # Generic keylogger / RAT hints
    "keylogger",
    "rat_client",
    "remote_input",
    "input_hook",
]

# Registry base for mic privacy
REG_BASE_MIC = r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone\\NonPackaged"

# Global state
STATE = {
    "firewall_enabled": False,
    "stealth_mode": False,
    "protect_all_runtimes": True,
    "kernel_driver_enabled": False,  # design stub
    "whitelist": [],  # list of exe paths or process names
}

STATE_LOCK = threading.Lock()

# ============================================================
# 3. LOGGING & CONFIG
# ============================================================
def write_log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except:
        pass
    return line.strip()

def load_log():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except:
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
    except Exception as e:
        write_log(f"[CONFIG] Load failed: {e}")

load_config()

# ============================================================
# 4. MIC FIREWALL (REGISTRY)
# ============================================================
def ensure_mic_parent_keys():
    try:
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore")
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone")
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, REG_BASE_MIC)
    except:
        pass

def block_mic_global():
    ensure_mic_parent_keys()
    try:
        key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE_MIC}\\Codex_Global_Block")
        winreg.SetValueEx(key, "Value", 0, winreg.REG_SZ, "Deny")
        write_log("Mic access DENIED for nonpackaged apps (Codex global block).")
    except Exception as e:
        write_log(f"Mic registry block failed: {e}")

def allow_mic_global():
    try:
        winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE_MIC}\\Codex_Global_Block")
        write_log("Mic access ALLOWED (Codex global block removed).")
    except:
        pass

# ============================================================
# 5. WHITELIST SYSTEM
# ============================================================
def is_whitelisted(proc: psutil.Process) -> bool:
    try:
        with STATE_LOCK:
            wl = STATE["whitelist"][:]
        name = (proc.info.get("name") or "").lower()
        exe = (proc.info.get("exe") or "").lower()
        for entry in wl:
            e = entry.lower()
            if e == name or e == exe:
                return True
    except:
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
# 6. DEEPER DETECTION ENGINE
# ============================================================
def process_is_monitored(proc: psutil.Process) -> bool:
    try:
        name = (proc.info.get("name") or "").lower()
        if not name:
            return False
        if STATE["protect_all_runtimes"]:
            # Monitor all known script runtimes
            return name in [n.lower() for n in MONITORED_EXECUTABLES]
        else:
            # Only Python
            return name in ["python.exe", "pythonw.exe"]
    except:
        return False

def process_looks_suspicious(proc: psutil.Process) -> bool:
    # Already whitelisted?
    if is_whitelisted(proc):
        return False

    try:
        name = (proc.info.get("name") or "").lower()
        exe = (proc.info.get("exe") or "").lower()
        cmd = " ".join(proc.cmdline()).lower()
    except:
        name = ""
        exe = ""
        cmd = ""

    # Quick name / cmdline scan
    for marker in SUSPICIOUS_MARKERS:
        m = marker.lower()
        if m in cmd or m in exe:
            return True

    # Deeper: memory maps / loaded files
    try:
        for m in proc.memory_maps():
            path = (m.path or "").lower()
            for marker in SUSPICIOUS_MARKERS:
                if marker.lower() in path:
                    return True
    except:
        pass

    # Optional: open files scan
    try:
        for f in proc.open_files():
            path = (f.path or "").lower()
            for marker in SUSPICIOUS_MARKERS:
                if marker.lower() in path:
                    return True
    except:
        pass

    return False

# ============================================================
# 7. KERNEL DRIVER DESIGN STUB
# ============================================================
class KernelDriverInterface:
    """
    DESIGN STUB ONLY.

    In a real implementation, this would:
    - Load a signed kernel driver
    - Intercept low-level input APIs (IRPs, HID, KBD, MOU)
    - Enforce per-process input/mic policies
    - Expose IOCTLs for user-mode control

    Here we just log intended actions.
    """
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
    # Push current policy
    policy = {
        "firewall_enabled": STATE["firewall_enabled"],
        "protect_all_runtimes": STATE["protect_all_runtimes"],
    }
    KERNEL_DRIVER.set_policy(policy)

# ============================================================
# 8. HARD FIREWALL WATCHER (STEALTH-CAPABLE)
# ============================================================
class HardFirewallWatcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        write_log("[WATCHER] HardFirewallWatcher started.")
        while self.running:
            try:
                with STATE_LOCK:
                    fw_on = STATE["firewall_enabled"]
                if fw_on:
                    for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
                        try:
                            if not process_is_monitored(proc):
                                continue
                            if process_looks_suspicious(proc):
                                write_log(f"[FIREWALL] Killing suspicious process PID {proc.pid} ({proc.info.get('name')})")
                                proc.kill()
                        except psutil.NoSuchProcess:
                            continue
                        except Exception as e:
                            write_log(f"[WATCHER] Error inspecting process: {e}")
                time.sleep(1.0)
            except Exception as e:
                write_log(f"[WATCHER] Loop error: {e}\n{traceback.format_exc()}")
                time.sleep(2.0)
        write_log("[WATCHER] HardFirewallWatcher stopped.")

WATCHER = HardFirewallWatcher()
WATCHER.start()

# ============================================================
# 9. CODEX COCKPIT GUI (DARK THEME)
# ============================================================
class CodexCockpitGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Codex Input Firewall — Cockpit")
        self.root.geometry("900x600")
        self.root.configure(bg="#101010")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except:
            pass
        style.configure("TButton", background="#202020", foreground="#EEEEEE")
        style.configure("TCheckbutton", background="#101010", foreground="#EEEEEE")
        style.configure("TLabel", background="#101010", foreground="#EEEEEE")

        self.build_layout()
        self.refresh_ui()
        self.schedule_log_refresh()

    def build_layout(self):
        # Top status bar
        self.status_label = tk.Label(
            self.root,
            text="Firewall: UNKNOWN",
            font=("Consolas", 18, "bold"),
            bg="#101010",
            fg="#FF5555"
        )
        self.status_label.pack(pady=10, fill="x")

        # Controls frame
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

        # Checkboxes
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

        # Whitelist panel
        wl_frame = tk.LabelFrame(self.root, text="Whitelist (names or full paths)", bg="#101010", fg="#EEEEEE")
        wl_frame.pack(padx=10, pady=5, fill="x")

        self.wl_entry = tk.Entry(wl_frame, bg="#202020", fg="#EEEEEE", insertbackground="#EEEEEE")
        self.wl_entry.pack(side="left", padx=5, fill="x", expand=True)

        self.btn_wl_add = ttk.Button(wl_frame, text="Add", command=self.add_whitelist_entry)
        self.btn_wl_add.pack(side="left", padx=5)

        self.btn_wl_remove = ttk.Button(wl_frame, text="Remove", command=self.remove_whitelist_entry)
        self.btn_wl_remove.pack(side="left", padx=5)

        self.wl_listbox = tk.Listbox(wl_frame, bg="#181818", fg="#EEEEEE", height=4)
        self.wl_listbox.pack(padx=5, pady=5, fill="x")
        self.refresh_whitelist_listbox()

        # Log box
        self.log_box = scrolledtext.ScrolledText(
            self.root,
            wrap="word",
            bg="#181818",
            fg="#DDDDDD",
            insertbackground="#DDDDDD"
        )
        self.log_box.pack(expand=True, fill="both", padx=10, pady=10)

    # ---------- UI helpers ----------
    def refresh_ui(self):
        with STATE_LOCK:
            fw_on = STATE["firewall_enabled"]
            stealth = STATE["stealth_mode"]
        if fw_on:
            self.status_label.config(
                text=f"Firewall: ENABLED ({'STEALTH' if stealth else 'VISIBLE'})",
                fg="#FF5555"
            )
        else:
            self.status_label.config(
                text="Firewall: DISABLED",
                fg="#55FF55"
            )

        self.var_protect_all.set(STATE["protect_all_runtimes"])
        self.var_kernel.set(STATE["kernel_driver_enabled"])

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
                self.wl_listbox.insert("end", entry)

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
        # NOTE: Stealth mode here is conceptual: in a real build,
        # you'd hide the window / run as background-only.

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

    def add_whitelist_entry(self):
        entry = self.wl_entry.get().strip()
        if not entry:
            return
        add_to_whitelist(entry)
        self.refresh_whitelist_listbox()
        self.wl_entry.delete(0, "end")

    def remove_whitelist_entry(self):
        sel = self.wl_listbox.curselection()
        if not sel:
            entry = self.wl_entry.get().strip()
            if entry:
                remove_from_whitelist(entry)
        else:
            entry = self.wl_listbox.get(sel[0])
            remove_from_whitelist(entry)
        self.refresh_whitelist_listbox()

# ============================================================
# 10. STEALTH MODE ENTRY (NO GUI)
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
# 11. MAIN
# ============================================================
def main():
    # If launched with --stealth, force stealth mode
    if "--stealth" in sys.argv:
        with STATE_LOCK:
            STATE["stealth_mode"] = True
            STATE["firewall_enabled"] = True
        save_config()
        block_mic_global()
        sync_kernel_driver_state()
        run_stealth_daemon()
        return

    # Normal cockpit mode
    root = tk.Tk()
    gui = CodexCockpitGUI(root)
    sync_kernel_driver_state()
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_log(f"[FATAL] {e}\n{traceback.format_exc()}")
        print("Codex Input Firewall crashed. See log for details.")
        input("Press ENTER to exit…")
