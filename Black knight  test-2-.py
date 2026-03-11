#!/usr/bin/env python3
# Black Knight – System-wide Local Privacy Guardian
# Local only. No telemetry. No biometrics. No network. No OS hooks. No system manipulation.

import os
import json
import random
import string
import threading
import time
from pathlib import Path
import sys
import stat

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

# -----------------------------
#  GLOBAL CONFIG / STATE
# -----------------------------

BASE_DIR = Path.home() / ".black_knight"
BASE_DIR.mkdir(exist_ok=True)

KEY_FILE = BASE_DIR / "key.bin"
MAP_FILE = BASE_DIR / "chameleon_map.json"
PROTECTED_ROOTS_FILE = BASE_DIR / "protected_roots.json"
SETTINGS_FILE = BASE_DIR / "settings.json"

WATCH_INTERVAL_SECONDS = 10  # watcher scan interval

DEFAULT_SETTINGS = {
    "auto_protect_all_drives": True,
    "skip_system_folders": True,
    "skip_readonly_files": True,
    "skip_smb_readonly_shares": True,
    "log_once_per_folder": True,
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

CHAMELEON_MAP = load_json(MAP_FILE, {})                 # original_path -> obfuscated_path
PROTECTED_ROOTS = load_json(PROTECTED_ROOTS_FILE, [])   # list of root paths (drives or dirs)
SETTINGS = load_json(SETTINGS_FILE, DEFAULT_SETTINGS.copy())

# ensure all settings keys exist
for k, v in DEFAULT_SETTINGS.items():
    SETTINGS.setdefault(k, v)
save_json(SETTINGS_FILE, SETTINGS)

WATCHER_STOP = False

# in-memory set for "log once per folder"
LOGGED_ERROR_DIRS = set()

# -----------------------------
#  UTILITY
# -----------------------------

def random_name(length=32):
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))

def is_under_root(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except AttributeError:
        return str(path.resolve()).startswith(str(root.resolve()))

def list_windows_drives():
    """
    Returns all drive roots on Windows, including mapped SMB/network drives.
    Example: ['C:\\', 'D:\\', 'Z:\\']
    """
    drives = []
    if os.name == "nt":
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{letter}:\\"
            if os.path.exists(root):
                drives.append(root)
    else:
        # Non-Windows: treat root as single "drive"
        drives.append(str(Path("/").resolve()))
    return drives

def is_system_path(path: Path) -> bool:
    if os.name != "nt":
        return False
    p = str(path.resolve()).lower()
    system_roots = [
        r"c:\windows",
        r"c:\program files",
        r"c:\program files (x86)",
    ]
    # user appdata
    user = os.environ.get("USERNAME", "")
    if user:
        system_roots.append(fr"c:\users\{user}\appdata")
    return any(p.startswith(root) for root in system_roots)

def has_write_access(path: Path) -> bool:
    try:
        # check read-only attribute
        st = path.stat()
        if not stat.S_IWRITE & st.st_mode:
            return False
        # os.access as extra check
        return os.access(str(path), os.W_OK)
    except Exception:
        return False

def can_write_to_root(root: Path) -> bool:
    """
    Best-effort check if we can write to a root (for SMB/read-only shares).
    """
    try:
        test_file = root / f".bk_test_{random_name(8)}"
        with open(test_file, "wb") as f:
            f.write(b"test")
        test_file.unlink()
        return True
    except Exception:
        return False

def log_error_once(log, folder: Path, msg: str):
    if not SETTINGS.get("log_once_per_folder", True):
        log(msg)
        return
    key = str(folder.resolve())
    if key in LOGGED_ERROR_DIRS:
        return
    LOGGED_ERROR_DIRS.add(key)
    log(msg)

# -----------------------------
#  CHAMELEON + MERIT
# -----------------------------

def chameleon_obfuscate(path: Path, log=None):
    if not path.is_file():
        return
    if str(path) in CHAMELEON_MAP:
        return  # already mapped

    obf_name = random_name() + path.suffix
    obf_path = path.with_name(obf_name)

    try:
        path.rename(obf_path)
        CHAMELEON_MAP[str(path)] = str(obf_path)
        save_json(MAP_FILE, CHAMELEON_MAP)
        if log:
            log(f"[CHAMELEON] {path} -> {obf_path}")
    except Exception as e:
        if log:
            folder = path.parent
            log_error_once(log, folder, f"[CHAMELEON][ERROR] {path}: {e}")

def chameleon_restore_any(path: Path, log=None):
    original = None
    obfuscated = None

    if str(path) in CHAMELEON_MAP:
        original = Path(str(path))
        obfuscated = Path(CHAMELEON_MAP[str(path)])
    else:
        for orig, obf in CHAMELEON_MAP.items():
            if obf == str(path):
                original = Path(orig)
                obfuscated = Path(obf)
                break

    if not original or not obfuscated:
        if log:
            log(f"[CHAMELEON] No mapping for {path}")
        return

    try:
        obfuscated.rename(original)
        if log:
            log(f"[CHAMELEON] Restored {obfuscated} -> {original}")
        del CHAMELEON_MAP[str(original)]
        save_json(MAP_FILE, CHAMELEON_MAP)
    except Exception as e:
        if log:
            folder = obfuscated.parent
            log_error_once(log, folder, f"[CHAMELEON][ERROR] restore {path}: {e}")

# -----------------------------
#  MIXER (encrypt/decrypt)
# -----------------------------

def mixer_encrypt(path: Path, log=None):
    if not path.is_file():
        return
    try:
        data = path.read_bytes()
        enc = FERNET.encrypt(data)
        path.write_bytes(enc)
        if log:
            log(f"[MIXER] Encrypted {path}")
    except Exception as e:
        if log:
            folder = path.parent
            log_error_once(log, folder, f"[MIXER][ERROR] {path}: {e}")

def mixer_decrypt(path: Path, log=None):
    if not path.is_file():
        return
    try:
        data = path.read_bytes()
        dec = FERNET.decrypt(data)
        path.write_bytes(dec)
        if log:
            log(f"[MIXER] Decrypted {path}")
    except Exception as e:
        if log:
            folder = path.parent
            log_error_once(log, folder, f"[MIXER][ERROR] {path}: {e}")

# -----------------------------
#  SHREDDER (manual)
# -----------------------------

def shredder_delete(path: Path, passes: int, log=None):
    if not path.is_file():
        return
    try:
        size = path.stat().st_size
        with open(path, "ba+", buffering=0) as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(size))
                f.flush()
                os.fsync(f.fileno())
        path.unlink()
        if log:
            log(f"[SHREDDER] Shredded {path} with {passes} passes")
    except Exception as e:
        if log:
            folder = path.parent
            log_error_once(log, folder, f"[SHREDDER][ERROR] {path}: {e}")

# -----------------------------
#  PROTECTION PIPELINE
# -----------------------------

def protect_root(root_path: Path, log=None):
    # SMB read-only share skip
    if SETTINGS.get("skip_smb_readonly_shares", True):
        if not can_write_to_root(root_path):
            if log:
                log_error_once(log, root_path, f"[PROTECT] Skipping read-only or inaccessible root: {root_path}")
            return

    if not root_path.exists() or not root_path.is_dir():
        if log:
            log_error_once(log, root_path, f"[PROTECT][ERROR] {root_path} is not a directory")
        return

    for current_root, dirs, files in os.walk(root_path):
        current_root_path = Path(current_root)

        # Skip system folders
        if SETTINGS.get("skip_system_folders", True) and is_system_path(current_root_path):
            if log:
                log_error_once(log, current_root_path, f"[PROTECT] Skipping system folder: {current_root_path}")
            continue

        for name in files:
            fpath = current_root_path / name

            # Skip if already mapped (original) or already obfuscated (value)
            if str(fpath) in CHAMELEON_MAP or str(fpath) in CHAMELEON_MAP.values():
                continue

            # Skip read-only files if enabled
            if SETTINGS.get("skip_readonly_files", True) and not has_write_access(fpath):
                if log:
                    log_error_once(log, current_root_path, f"[PROTECT] Skipping read-only or no-write file: {fpath}")
                continue

            mixer_encrypt(fpath, log)
            chameleon_obfuscate(fpath, log)

def restore_root(root_path: Path, log=None):
    if not root_path.exists() or not root_path.is_dir():
        if log:
            log_error_once(log, root_path, f"[RESTORE][ERROR] {root_path} is not a directory")
        return

    # Restore names for files under this root
    for orig, obf in list(CHAMELEON_MAP.items()):
        orig_p = Path(orig)
        obf_p = Path(obf)
        if is_under_root(orig_p, root_path):
            chameleon_restore_any(obf_p, log)

    # Decrypt all files under root
    for current_root, dirs, files in os.walk(root_path):
        current_root_path = Path(current_root)

        # Skip system folders on restore too (optional, but safer)
        if SETTINGS.get("skip_system_folders", True) and is_system_path(current_root_path):
            if log:
                log_error_once(log, current_root_path, f"[RESTORE] Skipping system folder: {current_root_path}")
            continue

        for name in files:
            fpath = current_root_path / name
            # Skip read-only files if enabled
            if SETTINGS.get("skip_readonly_files", True) and not has_write_access(fpath):
                if log:
                    log_error_once(log, current_root_path, f"[RESTORE] Skipping read-only or no-write file: {fpath}")
                continue
            mixer_decrypt(fpath, log)

# -----------------------------
#  WATCHER (always-on, local)
# -----------------------------

def watcher_loop(gui_ref):
    global WATCHER_STOP
    while not WATCHER_STOP:
        roots_snapshot = list(PROTECTED_ROOTS)
        for r in roots_snapshot:
            root_path = Path(r)
            if root_path.exists() and root_path.is_dir():
                if gui_ref:
                    gui_ref.log(f"[WATCHER] Scanning {r}")
                protect_root(root_path, gui_ref.log if gui_ref else None)
        time.sleep(WATCH_INTERVAL_SECONDS)

# -----------------------------
#  GUI COCKPIT
# -----------------------------

class BlackKnightGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Black Knight – System-wide Local Privacy Guardian")
        self.root.geometry("1000x620")

        self.auto_protect_var = tk.BooleanVar(value=SETTINGS.get("auto_protect_all_drives", True))
        self.skip_system_var = tk.BooleanVar(value=SETTINGS.get("skip_system_folders", True))
        self.skip_readonly_var = tk.BooleanVar(value=SETTINGS.get("skip_readonly_files", True))
        self.skip_smb_ro_var = tk.BooleanVar(value=SETTINGS.get("skip_smb_readonly_shares", True))
        self.log_once_var = tk.BooleanVar(value=SETTINGS.get("log_once_per_folder", True))

        self.create_widgets()
        self.apply_auto_protect_setting(initial=True)
        self.refresh_protected_roots()

        self.watcher_thread = threading.Thread(target=watcher_loop, args=(self,), daemon=True)
        self.watcher_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: protected roots
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_frame, text="Protected Roots (Drives / Directories)").pack(anchor="w")

        self.root_list = tk.Listbox(left_frame, height=18)
        self.root_list.pack(fill=tk.Y, expand=True)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Add Directory", command=self.add_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Drive", command=self.add_drive).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected_root).pack(side=tk.LEFT, padx=2)

        # Middle: actions + settings
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        ttk.Label(action_frame, text="Actions").pack(anchor="w")

        ttk.Button(action_frame, text="Protect All Now", command=self.protect_all_now).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Restore All Now", command=self.restore_all_now).pack(fill=tk.X, pady=2)

        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Button(action_frame, text="Shred File...", command=self.shred_file_dialog).pack(fill=tk.X, pady=2)

        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Drive protection mode
        ttk.Label(action_frame, text="Drive Protection Mode").pack(anchor="w", pady=(5, 0))
        self.auto_protect_check = ttk.Checkbutton(
            action_frame,
            text="Auto-protect all detected drives (including SMB/mapped drives)",
            variable=self.auto_protect_var,
            command=self.on_auto_protect_toggle
        )
        self.auto_protect_check.pack(anchor="w")
        ttk.Label(action_frame, text="If off: drives are detected, but you choose which to protect.").pack(anchor="w")

        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Protection filters
        ttk.Label(action_frame, text="Protection Filters").pack(anchor="w", pady=(5, 0))

        self.skip_system_check = ttk.Checkbutton(
            action_frame,
            text="Skip system folders (Windows / Program Files / AppData)",
            variable=self.skip_system_var,
            command=self.on_filter_toggle
        )
        self.skip_system_check.pack(anchor="w")

        self.skip_readonly_check = ttk.Checkbutton(
            action_frame,
            text="Skip read-only / no-write files",
            variable=self.skip_readonly_var,
            command=self.on_filter_toggle
        )
        self.skip_readonly_check.pack(anchor="w")

        self.skip_smb_ro_check = ttk.Checkbutton(
            action_frame,
            text="Skip read-only SMB/network shares",
            variable=self.skip_smb_ro_var,
            command=self.on_filter_toggle
        )
        self.skip_smb_ro_check.pack(anchor="w")

        self.log_once_check = ttk.Checkbutton(
            action_frame,
            text="Reduce repeated error logs (log once per folder)",
            variable=self.log_once_var,
            command=self.on_filter_toggle
        )
        self.log_once_check.pack(anchor="w")

        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(action_frame, text="Watcher Status").pack(anchor="w")
        self.watcher_label = ttk.Label(action_frame, text=f"Running (interval: {WATCH_INTERVAL_SECONDS}s)")
        self.watcher_label.pack(anchor="w")

        # Bottom: log
        log_frame = ttk.Frame(self.root)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        ttk.Label(log_frame, text="Event Log").pack(anchor="w")

        self.log_box = tk.Text(log_frame, height=12)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def refresh_protected_roots(self):
        self.root_list.delete(0, tk.END)
        for r in PROTECTED_ROOTS:
            self.root_list.insert(tk.END, r)

    def apply_auto_protect_setting(self, initial=False):
        drives = list_windows_drives()
        if self.auto_protect_var.get():
            # Add all detected drives to protected roots
            added = False
            for d in drives:
                if d not in PROTECTED_ROOTS:
                    PROTECTED_ROOTS.append(d)
                    added = True
                    if not initial:
                        self.log(f"[CONFIG] Auto-protect added drive: {d}")
            if added:
                save_json(PROTECTED_ROOTS_FILE, PROTECTED_ROOTS)
                self.refresh_protected_roots()
        else:
            if not initial:
                self.log("[CONFIG] Auto-protect disabled. Drives will not be auto-added.")

        SETTINGS["auto_protect_all_drives"] = self.auto_protect_var.get()
        save_json(SETTINGS_FILE, SETTINGS)

    def on_auto_protect_toggle(self):
        self.apply_auto_protect_setting(initial=False)

    def on_filter_toggle(self):
        SETTINGS["skip_system_folders"] = self.skip_system_var.get()
        SETTINGS["skip_readonly_files"] = self.skip_readonly_var.get()
        SETTINGS["skip_smb_readonly_shares"] = self.skip_smb_ro_var.get()
        SETTINGS["log_once_per_folder"] = self.log_once_var.get()
        save_json(SETTINGS_FILE, SETTINGS)
        self.log("[CONFIG] Updated protection filters.")

    def add_dir(self):
        d = filedialog.askdirectory()
        if not d:
            return
        if d not in PROTECTED_ROOTS:
            PROTECTED_ROOTS.append(d)
            save_json(PROTECTED_ROOTS_FILE, PROTECTED_ROOTS)
            self.refresh_protected_roots()
            self.log(f"[CONFIG] Added protected directory: {d}")

    def add_drive(self):
        drives = list_windows_drives()
        if not drives:
            messagebox.showinfo("Black Knight", "No drives detected. Use 'Add Directory' instead.")
            return

        win = tk.Toplevel(self.root)
        win.title("Select Drive")
        ttk.Label(win, text="Select a drive to protect:").pack(anchor="w", padx=10, pady=5)

        lb = tk.Listbox(win, height=len(drives))
        for d in drives:
            lb.insert(tk.END, d)
        lb.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def confirm():
            sel = lb.curselection()
            if not sel:
                win.destroy()
                return
            drive = lb.get(sel[0])
            if drive not in PROTECTED_ROOTS:
                PROTECTED_ROOTS.append(drive)
                save_json(PROTECTED_ROOTS_FILE, PROTECTED_ROOTS)
                self.refresh_protected_roots()
                self.log(f"[CONFIG] Added protected drive: {drive}")
            win.destroy()

        ttk.Button(win, text="OK", command=confirm).pack(pady=5)

    def remove_selected_root(self):
        sel = self.root_list.curselection()
        if not sel:
            return
        idx = sel[0]
        r = PROTECTED_ROOTS[idx]
        del PROTECTED_ROOTS[idx]
        save_json(PROTECTED_ROOTS_FILE, PROTECTED_ROOTS)
        self.refresh_protected_roots()
        self.log(f"[CONFIG] Removed protected root: {r}")

    def protect_all_now(self):
        if not PROTECTED_ROOTS:
            messagebox.showwarning("Black Knight", "No protected roots configured.")
            return
        for r in PROTECTED_ROOTS:
            self.log(f"[PROTECT] Manual protect: {r}")
            protect_root(Path(r), self.log)
        self.log("[PROTECT] Manual protect completed.")

    def restore_all_now(self):
        if not PROTECTED_ROOTS:
            messagebox.showwarning("Black Knight", "No protected roots configured.")
            return
        for r in PROTECTED_ROOTS:
            self.log(f"[RESTORE] Manual restore: {r}")
            restore_root(Path(r), self.log)
        self.log("[RESTORE] Manual restore completed.")

    def shred_file_dialog(self):
        f = filedialog.askopenfilename()
        if not f:
            return
        if not messagebox.askyesno(
            "Confirm Shred",
            f"Shred this file?\n\n{f}\n\nThis is irreversible."
        ):
            return
        shredder_delete(Path(f), passes=3, log=self.log)

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
    app.log("[SYSTEM] Black Knight started. Local only. No telemetry. No biometrics. No network.")
    app.log("[SYSTEM] Watcher is running and will auto-protect all configured roots.")
    if SETTINGS.get("auto_protect_all_drives", True):
        app.log("[SYSTEM] Auto-protect mode: ALL detected drives (including SMB/mapped) are protected by default.")
    else:
        app.log("[SYSTEM] Auto-protect mode: OFF. You choose which drives/directories to protect.")
    root.mainloop()

if __name__ == "__main__":
    main()

