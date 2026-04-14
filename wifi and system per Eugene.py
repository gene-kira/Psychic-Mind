#!/usr/bin/env python3
"""
Universal Guardian Cockpit v3 (Hybrid AI Mode, Hardened)
- Cross-platform (Windows, Linux, macOS)
- Auto-checks dependencies (psutil, tkinter)
- Process, drive, network monitoring
- Multi-OS Wi-Fi scanner
- AI mode engine: Learning / Relaxed / Strict (Hybrid)
- Wi-Fi trust DB (learning mode)
- Minimal Tkinter GUI cockpit
- Defensive, local-only, no remote control
- Hardened: skips incompatible drives, bad net entries, Wi-Fi errors
"""

import sys
import os
import time
import json
import threading
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# ---------- Auto-loader / dependency checker ----------

REQUIRED_MODULES = ["psutil"]

def ensure_dependencies():
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print("\n[!] Missing required modules:", ", ".join(missing))
        print("    Install them with:")
        print(f"    pip install {' '.join(missing)}\n")
        sys.exit(1)

ensure_dependencies()
import psutil  # type: ignore

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None
    print("[!] Tkinter not available. GUI cockpit disabled.")

# ---------- Config ----------

LOG_FILE = "guardian_daemon.log"
TRUST_DB_FILE = "guardian_trust.json"

WIFI_CHECK_INTERVAL = 15
MONITOR_INTERVAL = 5
NETWORK_SAMPLE_LIMIT = 80

# ---------- Trust DB ----------

def load_trust_db() -> Dict[str, Any]:
    if not os.path.exists(TRUST_DB_FILE):
        return {"wifi": {}}
    try:
        with open(TRUST_DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "wifi" not in data or not isinstance(data["wifi"], dict):
            data["wifi"] = {}
        return data
    except Exception:
        return {"wifi": {}}

def save_trust_db(db: Dict[str, Any]) -> None:
    try:
        with open(TRUST_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# ---------- Logging ----------

def log_event(level: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level.upper(),
        "message": message,
        "data": data or {},
    }
    line = json.dumps(entry, ensure_ascii=False)
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---------- Snapshots ----------

class ProcessSnapshot:
    def __init__(self):
        self.by_pid: Dict[int, Dict[str, Any]] = {}

    def capture(self) -> None:
        snapshot: Dict[int, Dict[str, Any]] = {}
        for p in psutil.process_iter(attrs=["pid", "name", "username", "exe"]):
            info = p.info
            snapshot[info["pid"]] = {
                "pid": info["pid"],
                "name": info.get("name") or "",
                "user": info.get("username") or "",
                "exe": info.get("exe") or "",
            }
        self.by_pid = snapshot

    def diff(self, other: "ProcessSnapshot") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        added: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []

        old_pids = set(other.by_pid.keys())
        new_pids = set(self.by_pid.keys())

        for pid in new_pids - old_pids:
            added.append(self.by_pid[pid])
        for pid in old_pids - new_pids:
            removed.append(other.by_pid[pid])

        return added, removed


class DriveSnapshot:
    def __init__(self):
        self.by_mount: Dict[str, Dict[str, Any]] = {}

    def capture(self) -> None:
        snapshot: Dict[str, Dict[str, Any]] = {}

        for part in psutil.disk_partitions(all=False):
            mount = part.mountpoint
            try:
                usage = psutil.disk_usage(mount)
            except OSError as e:
                # Skip ANY incompatible / ghost / broken drive
                log_event("ERROR", "Skipping incompatible drive", {
                    "mount": mount,
                    "error": str(e),
                })
                continue
            except Exception as e:
                log_event("ERROR", "Skipping drive due to unexpected error", {
                    "mount": mount,
                    "error": str(e),
                })
                continue

            snapshot[mount] = {
                "mount": mount,
                "fstype": part.fstype,
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": usage.percent,
            }

        self.by_mount = snapshot

    def diff(self, other: "DriveSnapshot") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        added: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []

        old_mounts = set(other.by_mount.keys())
        new_mounts = set(self.by_mount.keys())

        for m in new_mounts - old_mounts:
            added.append(self.by_mount[m])
        for m in old_mounts - new_mounts:
            removed.append(other.by_mount[m])

        return added, removed


class NetworkSnapshot:
    def __init__(self):
        self.connections: List[Dict[str, Any]] = []

    def capture(self) -> None:
        safe_list: List[Dict[str, Any]] = []

        try:
            raw = psutil.net_connections(kind="inet")
        except Exception as e:
            log_event("ERROR", "net_connections failed", {"error": str(e)})
            self.connections = []
            return

        for c in raw:
            try:
                laddr = ""
                raddr = ""

                if c.laddr:
                    laddr = f"{getattr(c.laddr, 'ip', '')}:{getattr(c.laddr, 'port', '')}"
                if c.raddr:
                    raddr = f"{getattr(c.raddr, 'ip', '')}:{getattr(c.raddr, 'port', '')}"

                safe_list.append({
                    "pid": c.pid if isinstance(c.pid, int) else None,
                    "laddr": laddr,
                    "raddr": raddr,
                    "status": str(c.status),
                })
            except Exception as e:
                log_event("ERROR", "Bad connection entry skipped", {"error": str(e)})
                continue

        self.connections = safe_list

# ---------- Multi-OS Wi-Fi scanner ----------

def get_wifi_status_windows() -> Optional[Dict[str, Any]]:
    try:
        out = subprocess.check_output(
            ["netsh", "wlan", "show", "interfaces"],
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
        )
    except Exception as e:
        log_event("ERROR", "netsh failed", {"error": str(e)})
        return None

    ssid = None
    bssid = None
    state = None

    for line in out.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, val = line.split(":", 1)
        key = key.lower().strip()
        val = val.strip()

        if key == "state":
            state = val
        elif key == "ssid" and "bssid" not in key:
            ssid = val
        elif key == "bssid":
            bssid = val.lower()

    if not ssid or not bssid:
        return None

    return {"ssid": ssid, "bssid": bssid, "state": state}


def get_wifi_status_linux() -> Optional[Dict[str, Any]]:
    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "active,ssid,bssid", "dev", "wifi"],
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
        )
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0] == "yes":
                return {"ssid": parts[1], "bssid": parts[2].lower(), "state": "connected"}
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["iwconfig"],
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
        )
    except Exception:
        return None

    ssid = None
    bssid = None

    for line in out.splitlines():
        line = line.strip()
        if "ESSID" in line and "off/any" not in line:
            ssid = line.split("ESSID:")[1].strip().strip('"')
        if "Access Point:" in line:
            bssid_raw = line.split("Access Point:")[1].strip()
            if bssid_raw != "Not-Associated":
                bssid = bssid_raw.lower()

    if ssid and bssid:
        return {"ssid": ssid, "bssid": bssid, "state": "connected"}
    return None


def get_wifi_status_macos() -> Optional[Dict[str, Any]]:
    airport = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
    if not os.path.exists(airport):
        return None

    try:
        out = subprocess.check_output(
            [airport, "-I"],
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
        )
    except Exception:
        return None

    ssid = None
    bssid = None
    state = "unknown"

    for line in out.splitlines():
        line = line.strip()
        if line.startswith("SSID:"):
            ssid = line.split(":", 1)[-1].strip()
        elif line.startswith("BSSID:"):
            bssid = line.split(":", 1)[-1].strip().lower()
        elif line.startswith("state:"):
            state = line.split(":", 1)[-1].strip()

    if ssid and bssid:
        return {"ssid": ssid, "bssid": bssid, "state": state}
    return None


def get_wifi_status_any() -> Optional[Dict[str, Any]]:
    if os.name == "nt":
        return get_wifi_status_windows()
    if sys.platform.startswith("linux"):
        return get_wifi_status_linux()
    if sys.platform == "darwin":
        return get_wifi_status_macos()
    return None

# ---------- Wi-Fi evaluation + AI mode ----------

def evaluate_wifi_trust(
    wifi_info: Optional[Dict[str, Any]],
    trust_db: Dict[str, Any]
) -> Tuple[str, str, bool, bool]:
    if wifi_info is None:
        return ("No Wi-Fi info", "INFO", False, False)

    ssid = wifi_info["ssid"]
    bssid = wifi_info["bssid"]

    wifi_db = trust_db.get("wifi", {})
    known = wifi_db.get(ssid, [])

    if ssid in wifi_db:
        if bssid in known:
            return (f"Trusted Wi-Fi: {ssid} ({bssid})", "OK", True, False)
        else:
            return (f"Known SSID but new BSSID {bssid}", "WARN", False, True)
    else:
        return (f"Unknown Wi-Fi SSID: {ssid} ({bssid})", "WARN", False, True)


def ai_select_mode(
    wifi_eval: Tuple[str, str, bool, bool],
    connection_count: int
) -> Tuple[str, str, bool]:
    msg, level, is_trusted, should_learn = wifi_eval

    high_conn = connection_count > 150

    if level == "WARN":
        if should_learn:
            return ("LEARNING", f"Learning new Wi-Fi: {msg}", True)
        return ("STRICT", f"Wi-Fi warning: {msg}", False)

    if level == "OK":
        if high_conn:
            return ("STRICT", f"High connection count: {connection_count}", False)
        return ("RELAXED", "Trusted Wi-Fi, normal activity", False)

    if high_conn:
        return ("STRICT", "No Wi-Fi trust + high connections", False)

    return ("RELAXED", "Normal conditions", False)

# ---------- Guardian state ----------

class GuardianState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_proc_snapshot = ProcessSnapshot()
        self.last_drive_snapshot = DriveSnapshot()
        self.last_net_snapshot = NetworkSnapshot()

        self.last_proc_snapshot.capture()
        self.last_drive_snapshot.capture()
        self.last_net_snapshot.capture()

        self.start_time = time.time()
        self.trust_db = load_trust_db()

        self.last_wifi_info: Optional[Dict[str, Any]] = None
        self.last_wifi_eval: Optional[Tuple[str, str, bool, bool]] = None

        self.ai_mode = "RELAXED"
        self.ai_reason = "Initial state."

    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "uptime_seconds": self.uptime_seconds(),
                "process_count": len(self.last_proc_snapshot.by_pid),
                "drive_count": len(self.last_drive_snapshot.by_mount),
                "connection_count": len(self.last_net_snapshot.connections),
                "ai_mode": self.ai_mode,
                "ai_reason": self.ai_reason,
            }

    def list_processes(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.last_proc_snapshot.by_pid.values())

    def list_drives(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.last_drive_snapshot.by_mount.values())

    def list_connections(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.last_net_snapshot.connections)

    def get_wifi_eval(self) -> Optional[Tuple[str, str, bool, bool]]:
        with self.lock:
            return self.last_wifi_eval

    def update_processes(self) -> None:
        with self.lock:
            new = ProcessSnapshot()
            new.capture()
            added, removed = new.diff(self.last_proc_snapshot)
            self.last_proc_snapshot = new

        for p in added:
            log_event("INFO", "Process started", p)
        for p in removed:
            log_event("INFO", "Process ended", p)

    def update_drives(self) -> None:
        with self.lock:
            new = DriveSnapshot()
            new.capture()
            added, removed = new.diff(self.last_drive_snapshot)
            self.last_drive_snapshot = new

        for d in added:
            log_event("WARN", "Drive mounted", d)
        for d in removed:
            log_event("WARN", "Drive unmounted", d)

    def update_network(self) -> None:
        with self.lock:
            new = NetworkSnapshot()
            new.capture()
            self.last_net_snapshot = new

    def update_wifi_and_ai(self) -> None:
        wifi_info = get_wifi_status_any()
        wifi_eval = evaluate_wifi_trust(wifi_info, self.trust_db)

        msg, level, is_trusted, should_learn = wifi_eval

        if level == "WARN":
            log_event("WARN", "Wi-Fi trust warning", {"wifi": wifi_info, "msg": msg})
        else:
            log_event("INFO", "Wi-Fi status", {"wifi": wifi_info, "msg": msg})

        connection_count = len(self.last_net_snapshot.connections)
        mode, reason, learn_now = ai_select_mode(wifi_eval, connection_count)

        if learn_now and wifi_info:
            ssid = wifi_info["ssid"]
            bssid = wifi_info["bssid"]
            wifi_db = self.trust_db.setdefault("wifi", {})
            bssids = wifi_db.setdefault(ssid, [])
            if bssid not in bssids:
                bssids.append(bssid)
                save_trust_db(self.trust_db)
                log_event("INFO", "Wi-Fi fingerprint learned", {"ssid": ssid, "bssid": bssid})

        with self.lock:
            self.last_wifi_info = wifi_info
            self.last_wifi_eval = wifi_eval
            self.ai_mode = mode
            self.ai_reason = reason

# ---------- Monitor threads ----------

class MonitorThread(threading.Thread):
    def __init__(self, state: GuardianState, interval_seconds: int = MONITOR_INTERVAL):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval_seconds
        self._stop_flag = threading.Event()

    def run(self) -> None:
        log_event("INFO", "Monitor thread started", {"interval": self.interval})
        while not self._stop_flag.is_set():
            try:
                self.state.update_processes()
                self.state.update_drives()
                self.state.update_network()
            except Exception as e:
                log_event("ERROR", "Monitor loop error", {"error": str(e)})
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_flag.set()


class WifiThread(threading.Thread):
    def __init__(self, state: GuardianState, interval_seconds: int = WIFI_CHECK_INTERVAL):
        super().__init__(daemon=True)
        self.state = state
        self.interval = interval_seconds
        self._stop_flag = threading.Event()

    def run(self) -> None:
        log_event("INFO", "Wi-Fi/AI monitor thread started", {"interval": self.interval})
        while not self._stop_flag.is_set():
            try:
                self.state.update_wifi_and_ai()
            except Exception as e:
                log_event("ERROR", "Wi-Fi/AI loop error", {"error": str(e)})
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_flag.set()

# ---------- GUI ----------

class GuardianGUI:
    def __init__(self, root: tk.Tk, state: GuardianState):
        self.root = root
        self.state = state

        root.title("Universal Guardian Cockpit v3")
        root.geometry("1000x650")

        self.build_layout()
        self.refresh_ui()

    def build_layout(self) -> None:
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(self.main_frame, text="Status:", font=("Segoe UI", 11, "bold"))
        self.status_label.pack(anchor="w")

        self.ai_label = ttk.Label(self.main_frame, text="AI Mode:", font=("Segoe UI", 10, "bold"))
        self.ai_label.pack(anchor="w", pady=(3, 3))

        self.wifi_label = ttk.Label(self.main_frame, text="Wi-Fi:", font=("Segoe UI", 10))
        self.wifi_label.pack(anchor="w", pady=(3, 10))

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Processes tab
        self.proc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.proc_frame, text="Processes")

        self.proc_tree = ttk.Treeview(
            self.proc_frame,
            columns=("pid", "name", "user", "exe"),
            show="headings"
        )
        for col, w in [("pid", 60), ("name", 180), ("user", 180), ("exe", 500)]:
            self.proc_tree.heading(col, text=col.upper())
            self.proc_tree.column(col, width=w, anchor="w")
        self.proc_tree.pack(fill=tk.BOTH, expand=True)

        # Drives tab
        self.drive_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.drive_frame, text="Drives")

        self.drive_tree = ttk.Treeview(
            self.drive_frame,
            columns=("mount", "fstype", "total", "used", "free", "percent"),
            show="headings"
        )
        for col, w in [
            ("mount", 140),
            ("fstype", 80),
            ("total", 140),
            ("used", 140),
            ("free", 140),
            ("percent", 80),
        ]:
            self.drive_tree.heading(col, text=col.upper())
            self.drive_tree.column(col, width=w, anchor="w")
        self.drive_tree.pack(fill=tk.BOTH, expand=True)

        # Network tab
        self.net_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.net_frame, text="Network")

        self.net_tree = ttk.Treeview(
            self.net_frame,
            columns=("pid", "laddr", "raddr", "status"),
            show="headings"
        )
        for col, w in [
            ("pid", 60),
            ("laddr", 260),
            ("raddr", 260),
            ("status", 120),
        ]:
            self.net_tree.heading(col, text=col.upper())
            self.net_tree.column(col, width=w, anchor="w")
        self.net_tree.pack(fill=tk.BOTH, expand=True)

    def refresh_ui(self) -> None:
        try:
            status = self.state.get_status()
            processes = self.state.list_processes()
            drives = self.state.list_drives()
            conns = self.state.list_connections()
        except Exception as e:
            log_event("ERROR", "GUI data fetch failed", {"error": str(e)})
            self.root.after(2000, self.refresh_ui)
            return

        self.status_label.config(
            text=(
                f"Status: uptime={int(status['uptime_seconds'])}s | "
                f"processes={status['process_count']} | drives={status['drive_count']} | "
                f"connections={status['connection_count']}"
            )
        )

        mode = status.get("ai_mode", "UNKNOWN")
        reason = status.get("ai_reason", "")
        color = {"STRICT": "red", "RELAXED": "green", "LEARNING": "orange"}.get(mode, "black")
        self.ai_label.config(text=f"AI Mode: {mode} — {reason}", foreground=color)

        wifi_eval = self.state.get_wifi_eval()
        if wifi_eval:
            msg, level, _, _ = wifi_eval
            wifi_color = {"OK": "green", "WARN": "red", "INFO": "black"}.get(level, "black")
            self.wifi_label.config(text=f"Wi-Fi: {msg}", foreground=wifi_color)
        else:
            self.wifi_label.config(text="Wi-Fi: (no data yet)", foreground="black")

        for i in self.proc_tree.get_children():
            self.proc_tree.delete(i)
        for proc in processes:
            self.proc_tree.insert(
                "",
                tk.END,
                values=(proc["pid"], proc["name"], proc["user"], proc["exe"])
            )

        for i in self.drive_tree.get_children():
            self.drive_tree.delete(i)
        for d in drives:
            self.drive_tree.insert(
                "",
                tk.END,
                values=(
                    d["mount"],
                    d["fstype"],
                    d["total"],
                    d["used"],
                    d["free"],
                    f"{d['percent']}%",
                ),
            )

        for i in self.net_tree.get_children():
            self.net_tree.delete(i)
        for c in conns[:NETWORK_SAMPLE_LIMIT]:
            self.net_tree.insert(
                "",
                tk.END,
                values=(c["pid"], c["laddr"], c["raddr"], c["status"])
            )

        self.root.after(2000, self.refresh_ui)

# ---------- Main ----------

def main() -> None:
    log_event("INFO", "Universal Guardian Cockpit v3 starting", {})
    state = GuardianState()

    monitor = MonitorThread(state)
    wifi_mon = WifiThread(state)

    monitor.start()
    wifi_mon.start()

    if tk is not None and ttk is not None:
        root = tk.Tk()
        gui = GuardianGUI(root, state)
        try:
            root.mainloop()
        finally:
            log_event("INFO", "GUI closed by operator", {})
    else:
        print("[!] Running without GUI. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    monitor.stop()
    wifi_mon.stop()
    time.sleep(1)
    log_event("INFO", "Universal Guardian Cockpit v3 exited", {})

if __name__ == "__main__":
    main()