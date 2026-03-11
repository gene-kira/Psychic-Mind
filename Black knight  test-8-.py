#!/usr/bin/env python3
# Black Knight – Borg Hot Zone Guardian (Minimal Cockpit, Manual Zones)
# Local only. No telemetry. No biometrics. No network. No OS hooks. No system manipulation.

import os
import sys
import json
import random
import string
import threading
import time
from pathlib import Path
import stat
import shutil

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from cryptography.fernet import Fernet
except ImportError:
    raise SystemExit(
        "Missing dependency: cryptography\n"
        "Install with: pip install cryptography\n"
        "This program does not auto-install anything to avoid network traffic."
    )

# === AUTO-ELEVATION CHECK (Windows only) ===
def ensure_admin():
    import ctypes
    try:
        if os.name == "nt" and not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}" {params}', None, 1
            )
            sys.exit()
    except Exception as e:
        print(f"[Black Knight] Elevation failed: {e}")
        sys.exit()

if os.name == "nt":
    ensure_admin()

# -----------------------------
#  STORAGE LOCATION (emptiest drive)
# -----------------------------

def list_windows_drives():
    drives = []
    if os.name == "nt":
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{letter}:\\"
            if os.path.exists(root):
                drives.append(root)
    else:
        drives.append(str(Path("/").resolve()))
    return drives

def pick_emptiest_drive():
    drives = list_windows_drives()
    best = None
    best_free = -1
    for d in drives:
        try:
            total, used, free = shutil.disk_usage(d)
            if free > best_free:
                best_free = free
                best = d
        except Exception:
            continue
    if best is None:
        return Path.home()
    return Path(best)

BASE_ROOT = pick_emptiest_drive()
BASE_DIR = Path(BASE_ROOT) / ".black_knight"
BASE_DIR.mkdir(exist_ok=True)

KEY_FILE = BASE_DIR / "key.bin"
INDEX_FILE = BASE_DIR / "index.json"
SETTINGS_FILE = BASE_DIR / "settings.json"

WATCH_INTERVAL_SECONDS = 10

DEFAULT_SETTINGS = {
    "skip_system_folders": True,
    "skip_readonly_files": True,
    "skip_smb_readonly_shares": True,
    "log_once_per_folder": True,
    "borg_enabled": True,
}

def load_or_create_key():
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    return key

FERNET_KEY = load_or_create_key()
FERNET = Fernet(FERNET_KEY)

def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

SETTINGS = load_json(SETTINGS_FILE, DEFAULT_SETTINGS.copy())
for k, v in DEFAULT_SETTINGS.items():
    SETTINGS.setdefault(k, v)
save_json(SETTINGS_FILE, SETTINGS)

INDEX = load_json(INDEX_FILE, {"hot_zones": [], "files": {}, "borg": {"zones": {}}})
# hot_zones: list of paths (manual only, no auto-drive hot zones)
# files: { original_path: { "obfuscated": str, "size": int, "mtime": float, "status": "encrypted"/"plain" } }
# borg.zones: { zone_path: { "heat": float, "last_activity": float } }

WATCHER_STOP = False
LOGGED_ERROR_DIRS = set()

# -----------------------------
#  UTILITY
# -----------------------------

def random_name(length=32):
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))

def is_system_path(path: Path) -> bool:
    if os.name != "nt":
        return False
    p = str(path.resolve()).lower()
    system_roots = [
        r"c:\windows",
        r"c:\program files",
        r"c:\program files (x86)",
    ]
    user = os.environ.get("USERNAME", "")
    if user:
        system_roots.append(fr"c:\users\{user}\appdata")
    return any(p.startswith(root) for root in system_roots)

def has_write_access(path: Path) -> bool:
    try:
        st = path.stat()
        if not stat.S_IWRITE & st.st_mode:
            return False
        return os.access(str(path), os.W_OK)
    except Exception:
        return False

def is_smb_path(path: Path) -> bool:
    s = str(path)
    return s.startswith("\\\\") or s.startswith("//")

def log_error_once(log, folder: Path, msg: str):
    if not SETTINGS.get("log_once_per_folder", True):
        log(msg)
        return
    key = str(folder.resolve()) if folder.exists() else str(folder)
    if key in LOGGED_ERROR_DIRS:
        return
    LOGGED_ERROR_DIRS.add(key)
    log(msg)

def file_signature(path: Path):
    try:
        st = path.stat()
        return st.st_size, st.st_mtime
    except Exception:
        return None, None

# -----------------------------
#  BORG CORTEX
# -----------------------------

class BorgCortex:
    def __init__(self, index_ref):
        self.index = index_ref

    def ensure_zone(self, zone: str):
        borg = self.index.setdefault("borg", {}).setdefault("zones", {})
        if zone not in borg:
            borg[zone] = {"heat": 0.0, "last_activity": 0.0}
        return borg[zone]

    def record_activity(self, zone: Path, intensity: float):
        z = str(zone.resolve()) if zone.exists() else str(zone)
        zone_state = self.ensure_zone(z)
        zone_state["heat"] = min(zone_state["heat"] + intensity, 1.0)
        zone_state["last_activity"] = time.time()

    def cool_zones(self):
        borg = self.index.setdefault("borg", {}).setdefault("zones", {})
        now = time.time()
        for z, state in borg.items():
            dt = now - state.get("last_activity", now)
            decay = dt / 600.0
            state["heat"] = max(0.0, state["heat"] - decay * 0.1)

    def get_sorted_zones(self):
        borg = self.index.setdefault("borg", {}).setdefault("zones", {})
        zones = []
        for z in INDEX.get("hot_zones", []):
            state = borg.get(z, {"heat": 0.0})
            zones.append((z, state.get("heat", 0.0)))
        zones.sort(key=lambda x: x[1], reverse=True)
        return zones

BORG = BorgCortex(INDEX)

# -----------------------------
#  CHAMELEON + MIXER
# -----------------------------

def encrypt_and_obfuscate(path: Path, log=None):
    if not path.is_file():
        return

    if SETTINGS.get("skip_system_folders", True) and is_system_path(path.parent):
        if log:
            log_error_once(log, path.parent, f"[PROTECT] Skipping system folder: {path.parent}")
        return

    if SETTINGS.get("skip_readonly_files", True) and not has_write_access(path):
        if log:
            log_error_once(log, path.parent, f"[PROTECT] Skipping read-only or no-write file: {path}")
        return

    size, mtime = file_signature(path)
    if size is None:
        return

    rec = INDEX["files"].get(str(path))
    if rec and rec.get("status") == "encrypted":
        return

    try:
        data = path.read_bytes()
        enc = FERNET.encrypt(data)
        path.write_bytes(enc)
    except Exception as e:
        if log:
            log_error_once(log, path.parent, f"[MIXER][ERROR] {path}: {e}")
        return

    obf_name = random_name() + path.suffix
    obf_path = path.with_name(obf_name)
    try:
        path.rename(obf_path)
    except Exception as e:
        if log:
            log_error_once(log, path.parent, f"[CHAMELEON][ERROR] {path}: {e}")
        return

    INDEX["files"][str(path)] = {
        "obfuscated": str(obf_path),
        "size": size,
        "mtime": mtime,
        "status": "encrypted",
    }
    save_json(INDEX_FILE, INDEX)
    if log:
        log(f"[PROTECT] {path} -> {obf_path} (encrypted + obfuscated)")

def decrypt_and_restore(original: str, log=None):
    rec = INDEX["files"].get(original)
    if not rec:
        if log:
            log(f"[RESTORE] No index record for {original}")
        return

    obf_path = Path(rec["obfuscated"])
    orig_path = Path(original)

    if not obf_path.exists():
        if log:
            log(f"[RESTORE] Obfuscated file missing: {obf_path}")
        return

    try:
        data = obf_path.read_bytes()
        dec = FERNET.decrypt(data)
        obf_path.write_bytes(dec)
    except Exception as e:
        if log:
            log_error_once(log, obf_path.parent, f"[MIXER][ERROR] decrypt {obf_path}: {e}")
        return

    try:
        obf_path.rename(orig_path)
    except Exception as e:
        if log:
            log_error_once(log, obf_path.parent, f"[CHAMELEON][ERROR] restore {obf_path}: {e}")
        return

    INDEX["files"][original]["status"] = "plain"
    save_json(INDEX_FILE, INDEX)
    if log:
        log(f"[RESTORE] {obf_path} -> {orig_path} (decrypted + restored)")

# -----------------------------
#  HOT ZONES + INCREMENTAL SCAN (NO DRIVE-WIDE DEFAULTS)
# -----------------------------

def scan_hot_zone(zone: Path, log=None):
    if not zone.exists() or not zone.is_dir():
        if log:
            log_error_once(log, zone, f"[SCAN] Hot zone missing or not a directory: {zone}")
        return

    if SETTINGS.get("skip_smb_readonly_shares", True) and is_smb_path(zone) and not has_write_access(zone):
        if log:
            log_error_once(log, zone, f"[SCAN] Skipping SMB read-only share: {zone}")
        return

    zone_activity = 0

    for current_root, dirs, files in os.walk(zone):
        current_root_path = Path(current_root)

        if SETTINGS.get("skip_system_folders", True) and is_system_path(current_root_path):
            if log:
                log_error_once(log, current_root_path, f"[SCAN] Skipping system folder: {current_root_path}")
            continue

        for name in files:
            fpath = current_root_path / name

            if any(rec.get("obfuscated") == str(fpath) for rec in INDEX["files"].values()):
                continue

            size, mtime = file_signature(fpath)
            if size is None:
                continue

            rec = INDEX["files"].get(str(fpath))
            if rec is None:
                encrypt_and_obfuscate(fpath, log)
                zone_activity += 1
            else:
                if rec.get("size") != size or rec.get("mtime") != mtime or rec.get("status") != "encrypted":
                    encrypt_and_obfuscate(fpath, log)
                    zone_activity += 1

    to_delete = []
    for orig, rec in INDEX["files"].items():
        orig_path = Path(orig)
        if str(orig_path).startswith(str(zone.resolve())):
            if not orig_path.exists() and not Path(rec["obfuscated"]).exists():
                to_delete.append(orig)
    for orig in to_delete:
        if log:
            log(f"[INDEX] Removing record for missing file: {orig}")
        del INDEX["files"][orig]
    if to_delete:
        save_json(INDEX_FILE, INDEX)

    if SETTINGS.get("borg_enabled", True) and zone_activity > 0:
        BORG.record_activity(zone, min(0.1 * zone_activity, 0.5))

def restore_hot_zone(zone: Path, log=None):
    for orig, rec in list(INDEX["files"].items()):
        orig_path = Path(orig)
        if str(orig_path).startswith(str(zone.resolve())):
            if rec.get("status") == "encrypted":
                decrypt_and_restore(orig, log)

# -----------------------------
#  WATCHER
# -----------------------------

def watcher_loop(gui_ref):
    global WATCHER_STOP
    while not WATCHER_STOP:
        if SETTINGS.get("borg_enabled", True):
            BORG.cool_zones()
            zones = BORG.get_sorted_zones()
            ordered_zones = [z for z, h in zones]
        else:
            ordered_zones = list(INDEX.get("hot_zones", []))

        for z in ordered_zones:
            zone_path = Path(z)
            if zone_path.exists() and zone_path.is_dir():
                if gui_ref:
                    gui_ref.log(f"[WATCHER] Scanning hot zone: {z}")
                scan_hot_zone(zone_path, gui_ref.log if gui_ref else None)
        time.sleep(WATCH_INTERVAL_SECONDS)

# -----------------------------
#  GUI – MINIMAL COCKPIT
# -----------------------------

class BlackKnightGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Black Knight – Borg Hot Zone Guardian (Minimal Cockpit)")
        self.root.geometry("1000x620")

        self.skip_system_var = tk.BooleanVar(value=SETTINGS.get("skip_system_folders", True))
        self.skip_readonly_var = tk.BooleanVar(value=SETTINGS.get("skip_readonly_files", True))
        self.skip_smb_ro_var = tk.BooleanVar(value=SETTINGS.get("skip_smb_readonly_shares", True))
        self.log_once_var = tk.BooleanVar(value=SETTINGS.get("log_once_per_folder", True))
        self.borg_enabled_var = tk.BooleanVar(value=SETTINGS.get("borg_enabled", True))

        self.create_widgets()
        self.refresh_hot_zones()
        self.refresh_borg_view()

        self.watcher_thread = threading.Thread(target=watcher_loop, args=(self,), daemon=True)
        self.watcher_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_frame, text="Hot Zones (Manual – Folders / SMB Paths)").pack(anchor="w")

        self.zone_list = tk.Listbox(left_frame, height=18)
        self.zone_list.pack(fill=tk.Y, expand=True)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Add Hot Zone", command=self.add_zone).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected_zone).pack(side=tk.LEFT, padx=2)

        mid_frame = ttk.Frame(main_frame)
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        ttk.Label(mid_frame, text="Protection Filters").pack(anchor="w", pady=(5, 0))

        self.skip_system_check = ttk.Checkbutton(
            mid_frame,
            text="Skip system folders (Windows / Program Files / AppData)",
            variable=self.skip_system_var,
            command=self.on_filter_toggle
        )
        self.skip_system_check.pack(anchor="w")

        self.skip_readonly_check = ttk.Checkbutton(
            mid_frame,
            text="Skip read-only / no-write files",
            variable=self.skip_readonly_var,
            command=self.on_filter_toggle
        )
        self.skip_readonly_check.pack(anchor="w")

        self.skip_smb_ro_check = ttk.Checkbutton(
            mid_frame,
            text="Skip SMB read-only shares",
            variable=self.skip_smb_ro_var,
            command=self.on_filter_toggle
        )
        self.skip_smb_ro_check.pack(anchor="w")

        self.log_once_check = ttk.Checkbutton(
            mid_frame,
            text="Reduce repeated error logs (log once per folder)",
            variable=self.log_once_var,
            command=self.on_filter_toggle
        )
        self.log_once_check.pack(anchor="w")

        ttk.Separator(mid_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        self.borg_check = ttk.Checkbutton(
            mid_frame,
            text="Enable Borg Cortex (heat-based prioritization)",
            variable=self.borg_enabled_var,
            command=self.on_filter_toggle
        )
        self.borg_check.pack(anchor="w")

        ttk.Separator(mid_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(mid_frame, text="Watcher Status").pack(anchor="w")
        self.watcher_label = ttk.Label(mid_frame, text=f"Running (interval: {WATCH_INTERVAL_SECONDS}s, event-driven)")
        self.watcher_label.pack(anchor="w")

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right_frame, text="Borg Cortex – Zone Heat Map").pack(anchor="w")

        self.borg_list = tk.Listbox(right_frame, height=18)
        self.borg_list.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.Frame(self.root)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        ttk.Label(log_frame, text="Event Log").pack(anchor="w")

        self.log_box = tk.Text(log_frame, height=10)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def refresh_hot_zones(self):
        self.zone_list.delete(0, tk.END)
        for z in INDEX.get("hot_zones", []):
            self.zone_list.insert(tk.END, z)

    def refresh_borg_view(self):
        self.borg_list.delete(0, tk.END)
        if not SETTINGS.get("borg_enabled", True):
            self.borg_list.insert(tk.END, "Borg Cortex disabled.")
            return
        zones = BORG.get_sorted_zones()
        if not zones:
            self.borg_list.insert(tk.END, "No zone activity yet.")
            return
        for z, h in zones:
            self.borg_list.insert(tk.END, f"{z}  |  heat={h:.2f}")

    def add_zone(self):
        d = filedialog.askdirectory()
        if not d:
            return
        if d not in INDEX["hot_zones"]:
            INDEX["hot_zones"].append(d)
            save_json(INDEX_FILE, INDEX)
            self.refresh_hot_zones()
            self.log(f"[CONFIG] Added hot zone: {d}")

    def remove_selected_zone(self):
        sel = self.zone_list.curselection()
        if not sel:
            return
        idx = sel[0]
        z = INDEX["hot_zones"][idx]
        del INDEX["hot_zones"][idx]
        save_json(INDEX_FILE, INDEX)
        self.refresh_hot_zones()
        self.log(f"[CONFIG] Removed hot zone: {z}")

    def on_filter_toggle(self):
        SETTINGS["skip_system_folders"] = self.skip_system_var.get()
        SETTINGS["skip_readonly_files"] = self.skip_readonly_var.get()
        SETTINGS["skip_smb_readonly_shares"] = self.skip_smb_ro_var.get()
        SETTINGS["log_once_per_folder"] = self.log_once_var.get()
        SETTINGS["borg_enabled"] = self.borg_enabled_var.get()
        save_json(SETTINGS_FILE, SETTINGS)
        self.refresh_borg_view()
        self.log("[CONFIG] Updated filters / Borg settings.")

    def on_close(self):
        global WATCHER_STOP
        WATCHER_STOP = True
        self.log("[SYSTEM] Stopping watcher and exiting...")
        self.root.after(300, self.root.destroy)

# -----------------------------
#  ENTRY POINT
# -----------------------------

def main():
    root = tk.Tk()
    app = BlackKnightGUI(root)
    app.log(f"[SYSTEM] Black Knight started. Metadata stored on: {BASE_DIR}")
    app.log("[SYSTEM] Local only. No telemetry. No biometrics. No network. No system manipulation.")
    app.log("[SYSTEM] Pure event-driven mode. Only manually configured hot zones are watched.")
    root.after(5000, app.refresh_borg_view)
    root.mainloop()

if __name__ == "__main__":
    main()

