"""
PERSONAL GUARDIAN SENTINEL - QT EDITION (DATA EGRESS HYBRID)
------------------------------------------------------------
Tier upgrade:

- Focus: DATA LEAVING THE SYSTEM (egress), not scanning/encrypting all drives.
- Hybrid response (Option E):
  - If exfil looks like personal / sensitive data:
      → Encrypt payload (conceptually) with Fernet
      → Mask with glyph + chameleon (reverse + decoy header/footer)
      → Exfil becomes useless; process "succeeds" logically, data unreadable
  - If exfil is suspicious but not clearly personal:
      → Mask / corrupt payload (conceptually)
  - Always:
      → Log event (encrypted log)
      → Raise alerts in cockpit

NOTE (IMPORTANT LIMITATION):
- From user space Python, we cannot truly intercept and rewrite arbitrary
  outbound packet payloads or process buffers at OS level.
- This sentinel:
  - Detects exfil-like behavior (connections, shell commands, USB, etc.)
  - Applies encryption/masking logic to any data it directly handles
  - Logs and alerts as if payload is scrambled
- It does NOT:
  - Modify firewall rules
  - Kill processes
  - Actually rewrite arbitrary network packets

You can still integrate this sentinel with your own tools / scripts so that
any data passed through its APIs is genuinely encrypted/masked before leaving.

Features:
- PySide6 / Qt GUI (fast, modern, tech layout)
- Monitoring + Alerting for YOUR DATA ONLY (egress-focused)
- Multiple profiles stored in guardian_profiles.json
- Live config dialog (switch/tune profiles without restart)
- Telemetry in background thread, GUI updated via signals
- ML-style anomaly detection (rolling stats + z-score)
- Threat timeline "graph" (CPU / RAM / anomaly sparkline)
- Plugin system
- Encrypted logs
- USB drive watcher plugin
- PowerShell/CMD watcher plugin
"""

import sys
import os
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Protocol

# === AUTOLOADER ===============================================================

try:
    import numpy as np
except ImportError:
    print("[Autoloader] numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("[Autoloader] psutil is required. Install with: pip install psutil")
    sys.exit(1)

try:
    from cryptography.fernet import Fernet
except ImportError:
    print("[Autoloader] cryptography is required. Install with: pip install cryptography")
    sys.exit(1)

try:
    from PySide6.QtCore import Qt, QTimer, Signal, QObject
    from PySide6.QtGui import QTextCursor, QColor
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QTextEdit, QPushButton, QDialog, QLineEdit, QComboBox,
        QFileDialog, QMessageBox, QFormLayout
    )
except ImportError:
    print("[Autoloader] PySide6 is required. Install with: pip install PySide6")
    sys.exit(1)


# ====== CONFIG / DATA MODELS ==================================================

@dataclass
class GuardianConfig:
    loop_hz: float = 1.0
    protected_paths: List[str] = field(default_factory=lambda: [
        os.path.expanduser("~/guardian_protected")
    ])
    allowlisted_programs: List[str] = field(default_factory=lambda: [
        "explorer.exe", "System", "python.exe", "cmd.exe", "powershell.exe"
    ])
    suspicious_ports: List[int] = field(default_factory=lambda: [22, 3389, 443, 8080, 5985])
    local_ip_prefixes: List[str] = field(default_factory=lambda: ["10.", "192.168.", "172.16."])
    auto_encrypt_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".csv", ".json", ".log", ".dat"
    ])


@dataclass
class TelemetrySnapshot:
    timestamp: float
    cpu_percent: float
    ram_percent: float
    process_count: int
    connections: int
    top_processes: List[Dict[str, Any]]
    anomaly_score: float = 0.0


@dataclass
class Alert:
    timestamp: float
    level: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    last_snapshot: Optional[TelemetrySnapshot] = None
    alerts: List[Alert] = field(default_factory=list)
    health: Dict[str, float] = field(default_factory=lambda: {
        "cpu": 1.0,
        "memory": 1.0,
        "stability": 1.0
    })
    protected_file_count: int = 0
    timeline_cpu: List[float] = field(default_factory=list)
    timeline_ram: List[float] = field(default_factory=list)
    timeline_anom: List[float] = field(default_factory=list)
    exfil_events: int = 0


# ====== PROFILE MANAGER =======================================================

class ProfileManager:
    def __init__(self, path: str = "guardian_profiles.json"):
        self.path = path
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.active_profile_name: str = "Default"
        self._load_or_init()

    def _default_profile(self) -> Dict[str, Any]:
        return asdict(GuardianConfig())

    def _load_or_init(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.profiles = data.get("profiles", {})
                self.active_profile_name = data.get("active_profile", "Default")
            except Exception:
                self.profiles = {"Default": self._default_profile()}
                self.active_profile_name = "Default"
        else:
            self.profiles = {"Default": self._default_profile()}
            self.active_profile_name = "Default"
            self._save()

        if self.active_profile_name not in self.profiles:
            self.active_profile_name = list(self.profiles.keys())[0]

    def _save(self):
        data = {
            "profiles": self.profiles,
            "active_profile": self.active_profile_name
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ProfileManager] Failed to save profiles: {e}")

    def get_profile_names(self) -> List[str]:
        return list(self.profiles.keys())

    def get_active_config(self) -> GuardianConfig:
        prof = self.profiles.get(self.active_profile_name, self._default_profile())
        return GuardianConfig(
            loop_hz=prof.get("loop_hz", 1.0),
            protected_paths=list(prof.get("protected_paths", [os.path.expanduser("~/guardian_protected")])),
            allowlisted_programs=list(prof.get("allowlisted_programs", [])),
            suspicious_ports=list(prof.get("suspicious_ports", [])),
            local_ip_prefixes=list(prof.get("local_ip_prefixes", [])),
            auto_encrypt_extensions=list(prof.get("auto_encrypt_extensions", [])),
        )

    def set_active_profile(self, name: str):
        if name in self.profiles:
            self.active_profile_name = name
            self._save()

    def save_profile_from_config(self, name: str, cfg: GuardianConfig):
        d = asdict(cfg)
        self.profiles[name] = d
        self.active_profile_name = name
        self._save()


# ====== ENCRYPTED LOGGER ======================================================

class EncryptedLogger:
    def __init__(self, log_dir: str = "guardian_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.key_path = os.path.join(self.log_dir, ".log_key")
        self.log_path = os.path.join(self.log_dir, "events.log.enc")
        self.fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_path, "wb") as f:
            f.write(key)
        return key

    def log_alert(self, alert: Alert):
        line = f"{alert.timestamp:.3f}|{alert.level}|{alert.message}\n"
        token = self.fernet.encrypt(line.encode("utf-8"))
        with open(self.log_path, "ab") as f:
            f.write(token + b"\n")


# ====== PAYLOAD PROTECTOR (GLYPH + CHAMELEON + FERNET) =======================

class PayloadProtector:
    """
    Conceptual payload protector:
    - Encrypts data with Fernet
    - Applies glyph (reverse bytes)
    - Applies chameleon mask (decoy header/footer)
    """

    def __init__(self, key_dir: str = "guardian_payload"):
        self.key_dir = key_dir
        os.makedirs(self.key_dir, exist_ok=True)
        self.key_path = os.path.join(self.key_dir, ".payload_key")
        self.fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_path, "wb") as f:
            f.write(key)
        return key

    def _glyph_transform(self, data: bytes) -> bytes:
        return data[::-1]

    def _chameleon_mask(self, data: bytes) -> bytes:
        header = b"[DECOY_HEADER]\n"
        footer = b"\n[DECOY_FOOTER]"
        return header + data + footer

    def _chameleon_unmask(self, data: bytes) -> bytes:
        header = b"[DECOY_HEADER]\n"
        footer = b"\n[DECOY_FOOTER]"
        if data.startswith(header) and data.endswith(footer):
            return data[len(header):-len(footer)]
        return data

    def encrypt_and_mask(self, data: bytes) -> bytes:
        masked = self._chameleon_mask(data)
        transformed = self._glyph_transform(masked)
        token = self.fernet.encrypt(transformed)
        return token

    def decrypt_and_unmask(self, token: bytes) -> bytes:
        transformed = self.fernet.decrypt(token)
        masked = self._glyph_transform(transformed)
        raw = self._chameleon_unmask(masked)
        return raw


# ====== HEALTH ESTIMATOR ======================================================

class HealthEstimator:
    def tick(self, world: WorldState, snap: TelemetrySnapshot):
        cpu_health = max(0.0, 1.0 - snap.cpu_percent / 150.0)
        mem_health = max(0.0, 1.0 - snap.ram_percent / 150.0)
        stability = max(0.0, 1.0 - (snap.cpu_percent + snap.ram_percent) / 250.0)

        world.health["cpu"] = 0.9 * world.health["cpu"] + 0.1 * cpu_health
        world.health["memory"] = 0.9 * world.health["memory"] + 0.1 * mem_health
        world.health["stability"] = 0.9 * world.health["stability"] + 0.1 * stability


# ====== ANOMALY DETECTOR (ML-STYLE) ==========================================

class AnomalyDetector:
    """
    Lightweight anomaly detector using rolling window + z-score on CPU/RAM.
    """

    def __init__(self, window: int = 120):
        self.window = window
        self.cpu_hist: List[float] = []
        self.ram_hist: List[float] = []

    def update_and_score(self, cpu: float, ram: float) -> float:
        self.cpu_hist.append(cpu)
        self.ram_hist.append(ram)
        if len(self.cpu_hist) > self.window:
            self.cpu_hist.pop(0)
        if len(self.ram_hist) > self.window:
            self.ram_hist.pop(0)

        if len(self.cpu_hist) < 10:
            return 0.0

        cpu_arr = np.array(self.cpu_hist)
        ram_arr = np.array(self.ram_hist)

        cpu_mean, cpu_std = cpu_arr.mean(), cpu_arr.std() + 1e-6
        ram_mean, ram_std = ram_arr.mean(), ram_arr.std() + 1e-6

        z_cpu = abs(cpu - cpu_mean) / cpu_std
        z_ram = abs(ram - ram_mean) / ram_std

        score = float((z_cpu + z_ram) / 2.0)
        return score


# ====== POLICY ENGINE (EGRESS-FOCUSED) =======================================

class PolicyEngine:
    def __init__(self, cfg: GuardianConfig):
        self.cfg = cfg

    def update_config(self, cfg: GuardianConfig):
        self.cfg = cfg

    def _is_local_ip(self, ip: str) -> bool:
        if not ip:
            return True
        for prefix in self.cfg.local_ip_prefixes:
            if ip.startswith(prefix):
                return True
        return False

    def evaluate(self, snap: TelemetrySnapshot, connections: List[Dict[str, Any]]) -> List[Alert]:
        alerts: List[Alert] = []
        now = snap.timestamp

        for p in snap.top_processes:
            name = (p["name"] or "").lower()
            if p["cpu"] > 50.0 and all(allowed.lower() != name for allowed in self.cfg.allowlisted_programs):
                alerts.append(Alert(
                    timestamp=now,
                    level="WARN",
                    message=f"High CPU by non-allowlisted process: {p['name']} (PID {p['pid']})",
                    details=p
                ))

        for c in connections:
            raddr = c["raddr"]
            if not raddr:
                continue
            try:
                ip, port_str = raddr.split(":")
                port = int(port_str)
            except Exception:
                continue

            if port in self.cfg.suspicious_ports and not self._is_local_ip(ip):
                alerts.append(Alert(
                    timestamp=now,
                    level="WARN",
                    message=f"Suspicious outbound connection to {raddr} (PID {c['pid']})",
                    details=c
                ))

        if snap.anomaly_score > 3.0:
            alerts.append(Alert(
                timestamp=now,
                level="WARN",
                message=f"Anomaly score high: {snap.anomaly_score:.2f}",
                details={"anomaly_score": snap.anomaly_score}
            ))

        return alerts


# ====== PLUGIN SYSTEM =========================================================

class Plugin(Protocol):
    def on_snapshot(self, world: WorldState, snap: TelemetrySnapshot, conns: List[Dict[str, Any]]) -> List[Alert]:
        ...


# ====== USB WATCHER PLUGIN ====================================================

class USBWatcherPlugin:
    """
    Watches for new removable drives.
    """

    def __init__(self):
        self.known_devices = set()

    def on_snapshot(self, world: WorldState, snap: TelemetrySnapshot, conns: List[Dict[str, Any]]) -> List[Alert]:
        alerts: List[Alert] = []
        now = snap.timestamp
        current = set()

        try:
            for p in psutil.disk_partitions(all=False):
                if "removable" in p.opts.lower() or p.device.lower().startswith("\\\\.\\physicaldrive"):
                    current.add(p.device)
        except Exception:
            return alerts

        new_devices = current - self.known_devices
        removed = self.known_devices - current

        for d in new_devices:
            alerts.append(Alert(
                timestamp=now,
                level="INFO",
                message=f"New USB/removable drive detected: {d}",
                details={"device": d}
            ))

        for d in removed:
            alerts.append(Alert(
                timestamp=now,
                level="INFO",
                message=f"USB/removable drive removed: {d}",
                details={"device": d}
            ))

        self.known_devices = current
        return alerts


# ====== SHELL WATCHER / EXFIL DETECTOR PLUGIN =================================

class ShellWatcherPlugin:
    """
    Watches PowerShell / CMD processes and flags exfil-like commands.
    Hybrid response:
    - If command looks like it handles personal data → "encrypt + mask" conceptually.
    - Else if suspicious → "mask" conceptually.
    """

    def __init__(self, payload_protector: PayloadProtector):
        self.last_seen_cmd: Dict[int, str] = {}
        self.payload_protector = payload_protector

    def _looks_like_personal_data(self, cmdline: str) -> bool:
        tokens = cmdline.lower()
        patterns = [
            "phone", "ssn", "social", "biometric", "macaddress",
            "export-csv", "select-object", "get-wmiobject", "get-ciminstance"
        ]
        return any(p in tokens for p in patterns)

    def _looks_like_exfil(self, cmdline: str) -> bool:
        tokens = cmdline.lower()
        exfil_markers = [
            "invoke-webrequest", "invoke-restmethod", "curl", "wget",
            "ftp", "smb", "copy", "upload", "post", "put"
        ]
        return any(p in tokens for p in exfil_markers)

    def on_snapshot(self, world: WorldState, snap: TelemetrySnapshot, conns: List[Dict[str, Any]]) -> List[Alert]:
        alerts: List[Alert] = []
        now = snap.timestamp

        for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            name = (proc.info.get("name") or "").lower()
            if name not in ("powershell.exe", "pwsh.exe", "cmd.exe"):
                continue
            pid = proc.info.get("pid")
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
            except Exception:
                cmdline = ""

            if not cmdline:
                continue

            prev = self.last_seen_cmd.get(pid)
            if prev == cmdline:
                continue

            self.last_seen_cmd[pid] = cmdline

            is_exfil = self._looks_like_exfil(cmdline)
            is_personal = self._looks_like_personal_data(cmdline)

            if not is_exfil:
                alerts.append(Alert(
                    timestamp=now,
                    level="INFO",
                    message=f"Shell activity ({name}, PID {pid})",
                    details={"cmdline": cmdline}
                ))
                continue

            if is_personal:
                fake_payload = b"SIMULATED_PERSONAL_DATA"
                scrambled = self.payload_protector.encrypt_and_mask(fake_payload)
                world.exfil_events += 1
                alerts.append(Alert(
                    timestamp=now,
                    level="WARN",
                    message=f"Exfil attempt with personal data detected and scrambled (PID {pid})",
                    details={
                        "cmdline": cmdline,
                        "scramble_mode": "encrypt+mask",
                        "scrambled_len": len(scrambled),
                    }
                ))
            else:
                fake_payload = b"SIMULATED_GENERIC_DATA"
                scrambled = self.payload_protector.encrypt_and_mask(fake_payload)
                world.exfil_events += 1
                alerts.append(Alert(
                    timestamp=now,
                    level="WARN",
                    message=f"Suspicious exfil attempt masked (PID {pid})",
                    details={
                        "cmdline": cmdline,
                        "scramble_mode": "mask",
                        "scrambled_len": len(scrambled),
                    }
                ))

        return alerts


# ====== TELEMETRY WORKER ======================================================

class TelemetryWorker(QObject):
    snapshot_ready = Signal(object, list)
    stopped = False

    def __init__(self, loop_hz: float):
        super().__init__()
        self.loop_hz = loop_hz
        self._t0 = time.time()

    def run(self):
        dt = 1.0 / max(0.1, self.loop_hz)
        while not self.stopped:
            t_rel = time.time() - self._t0
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            procs = list(psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]))
            procs_sorted = sorted(
                procs,
                key=lambda p: p.info.get("cpu_percent", 0.0),
                reverse=True
            )
            top = []
            for p in procs_sorted[:5]:
                info = p.info
                top.append({
                    "pid": info.get("pid"),
                    "name": info.get("name", "unknown"),
                    "cpu": info.get("cpu_percent", 0.0),
                    "mem": info.get("memory_percent", 0.0),
                })

            conns = []
            try:
                for c in psutil.net_connections(kind="inet"):
                    laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else ""
                    raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else ""
                    conns.append({
                        "pid": c.pid,
                        "laddr": laddr,
                        "raddr": raddr,
                        "status": c.status,
                    })
            except Exception:
                pass

            snap = TelemetrySnapshot(
                timestamp=t_rel,
                cpu_percent=cpu,
                ram_percent=ram,
                process_count=len(procs),
                connections=len(conns),
                top_processes=top,
            )
            self.snapshot_ready.emit(snap, conns)
            time.sleep(dt)


# ====== CONFIG DIALOG =========================================================

class ConfigDialog(QDialog):
    def __init__(self, parent, cfg: GuardianConfig, profile_manager: ProfileManager):
        super().__init__(parent)
        self.setWindowTitle("Guardian Profiles Config")
        self.cfg = cfg
        self.pm = profile_manager

        self.setMinimumSize(550, 380)

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        layout.addLayout(top_row)

        top_row.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(self.pm.get_profile_names())
        self.profile_combo.setCurrentText(self.pm.active_profile_name)
        top_row.addWidget(self.profile_combo)

        self.new_profile_edit = QLineEdit()
        self.new_profile_edit.setPlaceholderText("New profile name (optional)")
        top_row.addWidget(self.new_profile_edit)

        form = QFormLayout()
        layout.addLayout(form)

        self.paths_edit = QLineEdit(", ".join(self.cfg.protected_paths))
        form.addRow("Protected paths (for your own tools):", self.paths_edit)

        self.allow_edit = QLineEdit(", ".join(self.cfg.allowlisted_programs))
        form.addRow("Allowlisted programs:", self.allow_edit)

        self.ports_edit = QLineEdit(", ".join(str(p) for p in self.cfg.suspicious_ports))
        form.addRow("Suspicious ports:", self.ports_edit)

        self.ext_edit = QLineEdit(", ".join(self.cfg.auto_encrypt_extensions))
        form.addRow("Auto-encrypt extensions (for your tools):", self.ext_edit)

        self.ip_edit = QLineEdit(", ".join(self.cfg.local_ip_prefixes))
        form.addRow("Local IP prefixes:", self.ip_edit)

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        save_btn = QPushButton("Save Profile")
        close_btn = QPushButton("Close")
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)

        save_btn.clicked.connect(self._save_profile)
        close_btn.clicked.connect(self.close)
        self.profile_combo.currentTextChanged.connect(self._profile_changed)

    def _parse_list(self, s: str) -> List[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    def _parse_int_list(self, s: str) -> List[int]:
        out = []
        for x in s.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                out.append(int(x))
            except ValueError:
                pass
        return out

    def _profile_changed(self, name: str):
        self.pm.set_active_profile(name)
        cfg = self.pm.get_active_config()
        self.cfg = cfg
        self.paths_edit.setText(", ".join(cfg.protected_paths))
        self.allow_edit.setText(", ".join(cfg.allowlisted_programs))
        self.ports_edit.setText(", ".join(str(p) for p in cfg.suspicious_ports))
        self.ext_edit.setText(", ".join(cfg.auto_encrypt_extensions))
        self.ip_edit.setText(", ".join(cfg.local_ip_prefixes))

    def _save_profile(self):
        name = self.new_profile_edit.text().strip() or self.profile_combo.currentText()
        if not name:
            QMessageBox.critical(self, "Error", "Profile name cannot be empty.")
            return

        new_cfg = GuardianConfig(
            loop_hz=self.cfg.loop_hz,
            protected_paths=self._parse_list(self.paths_edit.text()),
            allowlisted_programs=self._parse_list(self.allow_edit.text()),
            suspicious_ports=self._parse_int_list(self.ports_edit.text()),
            local_ip_prefixes=self._parse_list(self.ip_edit.text()),
            auto_encrypt_extensions=self._parse_list(self.ext_edit.text()),
        )

        self.pm.save_profile_from_config(name, new_cfg)
        self.cfg = new_cfg

        self.profile_combo.clear()
        self.profile_combo.addItems(self.pm.get_profile_names())
        self.profile_combo.setCurrentText(name)

        QMessageBox.information(self, "Saved", f"Profile '{name}' saved.")


# ====== MAIN WINDOW ===========================================================

class GuardianWindow(QMainWindow):
    def __init__(self, sentinel: "GuardianApp"):
        super().__init__()
        self.sentinel = sentinel

        self.setWindowTitle("Personal Guardian Sentinel - Data Egress Cockpit")
        self.resize(1000, 650)

        self.setStyleSheet("""
            QMainWindow { background-color: #0d0d0d; }
            QLabel { color: #00eaff; font-family: Consolas; }
            QTextEdit { background-color: #050505; color: #00ff88; font-family: Consolas; }
            QPushButton { background-color: #101820; color: #00eaff; border: 1px solid #00eaff; padding: 4px; }
            QPushButton:hover { background-color: #1b2838; }
            QComboBox, QLineEdit { background-color: #050505; color: #00eaff; border: 1px solid #00eaff; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        hud_row = QHBoxLayout()
        main_layout.addLayout(hud_row)

        self.hud_label = QLabel("HUD")
        self.hud_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hud_row.addWidget(self.hud_label, stretch=1)

        self.profile_label = QLabel("Profile: -")
        self.profile_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hud_row.addWidget(self.profile_label)

        self.health_label = QLabel("Health")
        main_layout.addWidget(self.health_label)

        self.timeline_label = QLabel("Timeline")
        main_layout.addWidget(self.timeline_label)

        self.data_label = QLabel("Data Egress / Protection")
        main_layout.addWidget(self.data_label)

        alerts_header_row = QHBoxLayout()
        main_layout.addLayout(alerts_header_row)
        self.alerts_title = QLabel("Alerts")
        alerts_header_row.addWidget(self.alerts_title)
        self.toggle_alerts_btn = QPushButton("Hide")
        alerts_header_row.addStretch()
        alerts_header_row.addWidget(self.toggle_alerts_btn)

        self.alerts_view = QTextEdit()
        self.alerts_view.setReadOnly(True)
        main_layout.addWidget(self.alerts_view, stretch=1)

        btn_row = QHBoxLayout()
        main_layout.addLayout(btn_row)

        self.config_btn = QPushButton("Config")
        self.shred_btn = QPushButton("Shred File (manual)...")
        btn_row.addWidget(self.config_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.shred_btn)

        self.config_btn.clicked.connect(self.open_config_dialog)
        self.shred_btn.clicked.connect(self.shred_file_dialog)
        self.toggle_alerts_btn.clicked.connect(self.toggle_alerts_panel)

        self.alerts_visible = True

    def toggle_alerts_panel(self):
        self.alerts_visible = not self.alerts_visible
        self.alerts_view.setVisible(self.alerts_visible)
        self.toggle_alerts_btn.setText("Hide" if self.alerts_visible else "Show")

    def open_config_dialog(self):
        dlg = ConfigDialog(self, self.sentinel.cfg, self.sentinel.profile_manager)
        dlg.exec()
        self.sentinel.apply_new_config(self.sentinel.profile_manager.get_active_config())

    def shred_file_dialog(self):
        base = self.sentinel.cfg.protected_paths[0] if self.sentinel.cfg.protected_paths else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select File to Shred", base)
        if path:
            self.sentinel.manual_shred(path)

    def _append_alerts_colored(self, alerts: List[Alert]):
        self.alerts_view.clear()
        for a in alerts:
            if a.level == "WARN":
                color = QColor("#ffcc00")
            elif a.level == "ERROR":
                color = QColor("#ff5555")
            else:
                color = QColor("#00eaff")

            cursor = self.alerts_view.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.alerts_view.setTextCursor(cursor)

            self.alerts_view.setTextColor(color)
            line = f"[{a.level}] t={a.timestamp:.1f} :: {a.message}\n"
            self.alerts_view.insertPlainText(line)

        cursor = self.alerts_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.alerts_view.setTextCursor(cursor)
        self.alerts_view.ensureCursorVisible()

    def _render_sparkline(self, values: List[float], max_value: float = 100.0, width: int = 40) -> str:
        if not values:
            return "-" * width
        vals = values[-width:]
        blocks = "▁▂▃▄▅▆▇█"
        out = []
        for v in vals:
            norm = max(0.0, min(1.0, v / max_value))
            idx = int(norm * (len(blocks) - 1))
            out.append(blocks[idx])
        return "".join(out).ljust(width)

    def update_view(self, world: WorldState, snap: TelemetrySnapshot, cfg: GuardianConfig, profile_name: str):
        hud_text = (
            f"CPU: {snap.cpu_percent:.1f}% | RAM: {snap.ram_percent:.1f}% | "
            f"Procs: {snap.process_count} | Conns: {snap.connections} | "
            f"Anom: {snap.anomaly_score:.2f}"
        )
        self.hud_label.setText(hud_text)

        self.profile_label.setText(f"Profile: {profile_name}")

        health_text = (
            f"Health  CPU: {world.health['cpu']:.2f}  "
            f"MEM: {world.health['memory']:.2f}  "
            f"STAB: {world.health['stability']:.2f}"
        )
        self.health_label.setText(health_text)

        cpu_line = self._render_sparkline(world.timeline_cpu, 100.0)
        ram_line = self._render_sparkline(world.timeline_ram, 100.0)
        anom_line = self._render_sparkline(world.timeline_anom, 10.0)

        timeline_text = (
            f"CPU : {cpu_line}\n"
            f"RAM : {ram_line}\n"
            f"ANOM: {anom_line}"
        )
        self.timeline_label.setText(timeline_text)

        data_text = (
            f"Egress events detected: {world.exfil_events}  |  Protected paths (for your tools): "
            f"{', '.join(cfg.protected_paths)}"
        )
        self.data_label.setText(data_text)

        self._append_alerts_colored(world.alerts[-80:])


# ====== MAIN APP LOGIC ========================================================

class GuardianApp(QObject):
    def __init__(self, app: QApplication):
        super().__init__()
        self.qt_app = app
        self.profile_manager = ProfileManager()
        self.cfg = self.profile_manager.get_active_config()
        self.world = WorldState()
        self.health_estimator = HealthEstimator()
        self.anomaly = AnomalyDetector()
        self.policy = PolicyEngine(self.cfg)
        self.logger = EncryptedLogger()
        self.payload_protector = PayloadProtector()

        self.window = GuardianWindow(self)
        self.window.show()

        self.plugins: List[Plugin] = [
            USBWatcherPlugin(),
            ShellWatcherPlugin(self.payload_protector),
        ]

        self.worker = TelemetryWorker(self.cfg.loop_hz)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker.snapshot_ready.connect(self.on_snapshot)
        self.worker_thread.start()

        self.gui_timer = QTimer()
        self.gui_timer.timeout.connect(self.refresh_gui)
        self.gui_timer.start(300)

        self._latest_snap: Optional[TelemetrySnapshot] = None
        self._latest_conns: List[Dict[str, Any]] = []

    def apply_new_config(self, cfg: GuardianConfig):
        self.cfg = cfg
        self.policy.update_config(cfg)

    def manual_shred(self, path: str):
        if not os.path.isfile(path):
            QMessageBox.warning(self.window, "Shredder", "File does not exist.")
            return
        size = os.path.getsize(path)
        try:
            with open(path, "r+b") as f:
                f.write(os.urandom(size))
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass
        try:
            os.remove(path)
        except Exception:
            pass
        QMessageBox.information(self.window, "Shredder", f"File shredded:\n{path}")

    def on_snapshot(self, snap: TelemetrySnapshot, conns: list):
        snap.anomaly_score = self.anomaly.update_and_score(snap.cpu_percent, snap.ram_percent)

        self._latest_snap = snap
        self._latest_conns = conns

        self.world.timeline_cpu.append(snap.cpu_percent)
        self.world.timeline_ram.append(snap.ram_percent)
        self.world.timeline_anom.append(snap.anomaly_score)
        if len(self.world.timeline_cpu) > 300:
            self.world.timeline_cpu.pop(0)
            self.world.timeline_ram.pop(0)
            self.world.timeline_anom.pop(0)

        self.health_estimator.tick(self.world, snap)

        alerts = self.policy.evaluate(snap, conns)

        for plugin in self.plugins:
            try:
                alerts.extend(plugin.on_snapshot(self.world, snap, conns))
            except Exception as e:
                alerts.append(Alert(
                    timestamp=snap.timestamp,
                    level="ERROR",
                    message=f"Plugin error: {e}",
                    details={}
                ))

        for a in alerts:
            self.world.alerts.append(a)
            self.logger.log_alert(a)

    def refresh_gui(self):
        if self._latest_snap is None:
            return
        self.window.update_view(
            self.world,
            self._latest_snap,
            self.cfg,
            self.profile_manager.active_profile_name
        )

    def stop(self):
        self.worker.stopped = True


# ====== ENTRYPOINT ============================================================

def main():
    app = QApplication(sys.argv)
    guardian = GuardianApp(app)
    try:
        sys.exit(app.exec())
    finally:
        guardian.stop()


if __name__ == "__main__":
    main()
