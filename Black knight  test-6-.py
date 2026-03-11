#!/usr/bin/env python3
# Black Knight – Event-Driven Privacy Guardian (All Drives, No Scanning)
# Local only. No telemetry. No biometrics. No network. No OS hooks. No system manipulation.

import os
import json
import random
import string
import threading
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

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    raise SystemExit(
        "Missing dependency: watchdog\n"
        "Install with: pip install watchdog\n"
        "This program does not auto-install anything to avoid network traffic."
    )

# -----------------------------
#  DRIVE ENUM + STORAGE LOCATION
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

DEFAULT_SETTINGS = {
    "skip_system_folders": True,
    "skip_readonly_files": True,
    "log_once_per_folder": True,
}

LOGGED_ERROR_DIRS = set()

# -----------------------------
#  CORE STORAGE + CRYPTO
# -----------------------------

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

INDEX = load_json(INDEX_FILE, {"files": {}})
# files: { original_path: { "obfuscated": str, "status": "encrypted"/"plain" } }

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

def log_error_once(log, folder: Path, msg: str):
    if not SETTINGS.get("log_once_per_folder", True):
        log(msg)
        return
    key = str(folder.resolve())
    if key in LOGGED_ERROR_DIRS:
        return
    LOGGED_ERROR_DIRS.add(key)
    log(msg)

def is_interesting_file(path: Path) -> bool:
    if not path.is_file():
        return False

    name = path.name.lower()
    if name.startswith("~$") or name.endswith(".tmp"):
        return False

    exts = {
        ".txt", ".log", ".cfg", ".ini", ".json", ".xml",
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".pdf", ".rtf",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
        ".zip", ".7z", ".rar",
        ".db", ".sqlite", ".sqlite3",
    }
    if path.suffix.lower() in exts:
        return True

    try:
        st = path.stat()
        if st.st_size < 1024:
            return True
        if st.st_size < 50 * 1024 * 1024:
            return True
    except Exception:
        return False

    return False

# -----------------------------
#  PROTECTION PIPELINE
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
#  FILE SYSTEM EVENT HANDLER
# -----------------------------

class BKEventHandler(FileSystemEventHandler):
    def __init__(self, gui_ref=None):
        super().__init__()
        self.gui_ref = gui_ref

    def log(self, msg):
        if self.gui_ref:
            self.gui_ref.log(msg)
        else:
            print(msg)

    def _handle_path(self, path_str: str):
        p = Path(path_str)
        if not p.exists():
            return
        if not is_interesting_file(p):
            return
        encrypt_and_obfuscate(p, self.log)

    def on_created(self, event):
        if event.is_directory:
            return
        self.log(f"[EVENT] Created: {event.src_path}")
        self._handle_path(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        self.log(f"[EVENT] Modified: {event.src_path}")
        self._handle_path(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        self.log(f"[EVENT] Moved: {event.src_path} -> {event.dest_path}")
        self._handle_path(event.dest_path)

# -----------------------------
#  WATCHER (ALL DRIVES, NO SCAN)
# -----------------------------

class BKWatcher:
    def __init__(self, gui_ref=None):
        self.gui_ref = gui_ref
        self.observer = Observer()
        self.handler = BKEventHandler(gui_ref)

    def start(self):
        drives = list_windows_drives()
        for d in drives:
            path = d
            try:
                self.observer.schedule(self.handler, path, recursive=True)
                if self.gui_ref:
                    self.gui_ref.log(f"[WATCHER] Watching: {path}")
                else:
                    print(f"[WATCHER] Watching: {path}")
            except Exception as e:
                if self.gui_ref:
                    self.gui_ref.log(f"[WATCHER][ERROR] Cannot watch {path}: {e}")
                else:
                    print(f"[WATCHER][ERROR] Cannot watch {path}: {e}")
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()

# -----------------------------
#  GUI
# -----------------------------

class BlackKnightGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Black Knight – Event-Driven Privacy Guardian")
        self.root.geometry("950x600")

        self.skip_system_var = tk.BooleanVar(value=SETTINGS.get("skip_system_folders", True))
        self.skip_readonly_var = tk.BooleanVar(value=SETTINGS.get("skip_readonly_files", True))
        self.log_once_var = tk.BooleanVar(value=SETTINGS.get("log_once_per_folder", True))

        self.create_widgets()

        self.watcher = BKWatcher(self)
        self.watcher_thread = threading.Thread(target=self._start_watcher, daemon=True)
        self.watcher_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _start_watcher(self):
        self.watcher.start()

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_frame, text="Watched Roots (All Drives)").pack(anchor="w")

        self.roots_list = tk.Listbox(left_frame, height=18)
        self.roots_list.pack(fill=tk.Y, expand=True)

        for d in list_windows_drives():
            self.roots_list.insert(tk.END, d)

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        ttk.Label(action_frame, text="Actions").pack(anchor="w")

        ttk.Button(action_frame, text="Restore All Known Files", command=self.restore_all_known).pack(fill=tk.X, pady=2)

        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Button(action_frame, text="Shred File...", command=self.shred_file_dialog).pack(fill=tk.X, pady=2)

        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

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

        self.log_once_check = ttk.Checkbutton(
            action_frame,
            text="Reduce repeated error logs (log once per folder)",
            variable=self.log_once_var,
            command=self.on_filter_toggle
        )
        self.log_once_check.pack(anchor="w")

        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(action_frame, text="Watcher Status").pack(anchor="w")
        self.watcher_label = ttk.Label(action_frame, text="Running (event-driven, all drives)")
        self.watcher_label.pack(anchor="w")

        log_frame = ttk.Frame(self.root)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        ttk.Label(log_frame, text="Event Log").pack(anchor="w")

        self.log_box = tk.Text(log_frame, height=12)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def restore_all_known(self):
        if not INDEX["files"]:
            messagebox.showinfo("Black Knight", "No indexed files to restore.")
            return
        for orig, rec in list(INDEX["files"].items()):
            if rec.get("status") == "encrypted":
                decrypt_and_restore(orig, self.log)
        self.log("[RESTORE] Restore attempt completed for all indexed files.")

    def shred_file_dialog(self):
        f = filedialog.askopenfilename()
        if not f:
            return
        if not messagebox.askyesno(
            "Confirm Shred",
            f"Shred this file?\n\n{f}\n\nThis is irreversible."
        ):
            return
        path = Path(f)
        from os import urandom, fsync
        try:
            size = path.stat().st_size
            with open(path, "ba+", buffering=0) as fh:
                for _ in range(3):
                    fh.seek(0)
                    fh.write(urandom(size))
                    fh.flush()
                    fsync(fh.fileno())
            path.unlink()
            self.log(f"[SHREDDER] Shredded {path} with 3 passes")
        except Exception as e:
            self.log(f"[SHREDDER][ERROR] {path}: {e}")

    def on_filter_toggle(self):
        SETTINGS["skip_system_folders"] = self.skip_system_var.get()
        SETTINGS["skip_readonly_files"] = self.skip_readonly_var.get()
        SETTINGS["log_once_per_folder"] = self.log_once_var.get()
        save_json(SETTINGS_FILE, SETTINGS)
        self.log("[CONFIG] Updated protection filters.")

    def on_close(self):
        self.log("[SYSTEM] Stopping watcher and exiting...")
        try:
            self.watcher.stop()
        except Exception:
            pass
        self.root.after(300, self.root.destroy)

# -----------------------------
#  ENTRY POINT
# -----------------------------

def main():
    root = tk.Tk()
    app = BlackKnightGUI(root)
    app.log(f"[SYSTEM] Black Knight started. Metadata stored on: {BASE_DIR}")
    app.log("[SYSTEM] Local only. No telemetry. No biometrics. No network. No system manipulation.")
    app.log("[SYSTEM] All detected drives are watched. Files are protected on creation/modification only (no scanning).")
    root.mainloop()

if __name__ == "__main__":
    main()

