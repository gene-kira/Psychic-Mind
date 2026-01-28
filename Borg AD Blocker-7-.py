import sys
import time
import threading
import queue
import re
import os
import json
import string
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, List, Set, Optional

import psutil
from PyQt5 import QtWidgets, QtCore, QtGui


# ========== Reboot memory manager ==========

class RebootMemoryManager:
    def __init__(self, filename="reboot_memory.json"):
        self.filename = filename
        self.memory_path = self._select_memory_path()
        self.full_path = self.memory_path / self.filename
        self.memory_path.mkdir(parents=True, exist_ok=True)

    def _select_memory_path(self) -> Path:
        # Scan D:..Z: for a writable drive, fall back to C:
        for letter in string.ascii_uppercase[3:]:  # D..Z
            drive = Path(f"{letter}:/")
            if drive.is_dir():
                test_path = drive / "QueenGuardMemory"
                try:
                    test_path.mkdir(parents=True, exist_ok=True)
                    test_file = test_path / "test.tmp"
                    with open(test_file, "w") as f:
                        f.write("ok")
                    test_file.unlink()
                    return test_path
                except Exception:
                    continue
        fallback = Path("C:/QueenGuardMemory")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def save(self, state: dict):
        try:
            with open(self.full_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"[RebootMemory] ERROR saving memory: {e}")

    def load(self) -> dict:
        if not self.full_path.exists():
            return {}
        try:
            with open(self.full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_storage_location(self) -> str:
        return str(self.full_path)


# ========== Swarm sync manager (multi-node) ==========

class SwarmSyncManager:
    """
    Very simple multi-node sync:
    - Uses a shared JSON file (e.g., on SMB or local folder).
    - Merges policies and history from other nodes.
    - Writes local snapshot back.
    """
    def __init__(self, path: Optional[str] = None, node_id: Optional[str] = None):
        if path is None:
            # Default to same folder as reboot memory
            base = Path("C:/QueenGuardSwarm")
            base.mkdir(parents=True, exist_ok=True)
            path = base / "swarm_state.json"
        self.path = Path(path)
        self.node_id = node_id or os.environ.get("COMPUTERNAME", "UNKNOWN_NODE")

        self.lock = threading.Lock()
        self.last_sync = 0

    def load_swarm(self) -> dict:
        with self.lock:
            if not self.path.exists():
                return {}
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

    def save_swarm(self, swarm_state: dict):
        with self.lock:
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(swarm_state, f, indent=4)
            except Exception as e:
                print(f"[Swarm] ERROR saving swarm: {e}")

    def merge(self, local_state: dict) -> dict:
        """
        Merge local policies/history into swarm and pull others back.
        """
        now = time.time()
        if now - self.last_sync < 10:
            return local_state  # avoid thrashing
        self.last_sync = now

        swarm = self.load_swarm()
        nodes = swarm.get("nodes", {})
        nodes[self.node_id] = {
            "timestamp": now,
            "policies": local_state.get("policies", {}),
            "history": local_state.get("history", {}),
        }
        swarm["nodes"] = nodes

        # Merge all nodes into a combined view
        merged_policies: Dict[str, dict] = {}
        merged_history: Dict[str, dict] = {}

        for nid, data in nodes.items():
            pol = data.get("policies", {})
            hist = data.get("history", {})
            for k, v in pol.items():
                existing = merged_policies.get(k)
                if not existing:
                    merged_policies[k] = v
                else:
                    # Simple rule: if any node blocks, keep block; else if any allows, keep allow
                    if v.get("block_always"):
                        existing["block_always"] = True
                        existing["allow_always"] = False
                    if v.get("allow_always") and not existing.get("block_always"):
                        existing["allow_always"] = True
                    if v.get("radioactive"):
                        existing["radioactive"] = True
            for k, h in hist.items():
                existing = merged_history.get(k)
                if not existing:
                    merged_history[k] = h
                else:
                    # Merge counts and timestamps
                    existing["count"] = existing.get("count", 0) + h.get("count", 0)
                    existing["first_seen"] = min(existing.get("first_seen", h.get("first_seen", 0)),
                                                 h.get("first_seen", 0))
                    existing["last_seen"] = max(existing.get("last_seen", h.get("last_seen", 0)),
                                                h.get("last_seen", 0))

        swarm["merged_policies"] = merged_policies
        swarm["merged_history"] = merged_history
        self.save_swarm(swarm)

        return {
            "policies": merged_policies,
            "history": merged_history,
        }


# ========== Data models ==========

@dataclass
class SensitiveTokenConfig:
    personal_tokens: List[str] = field(default_factory=list)
    system_tokens: List[str] = field(default_factory=list)


@dataclass
class ConnectionEvent:
    timestamp: float
    direction: str
    process_name: str
    pid: int
    origin: str
    local_addr: str
    remote_addr: str
    remote_port: int
    country: str
    region: str
    city: str
    resolution: int
    confidence: str
    overseas: bool
    pii_detected: bool
    pii_types: List[str]
    safe_startup: bool
    heavy_watch: bool = False
    risk_score: int = 0
    pending_approval: bool = False
    deviation_score: int = 0


@dataclass
class PolicyDecision:
    allow_always: bool
    block_always: bool
    radioactive: bool = False


# ========== PII detector (simulated, pluggable) ==========

class PIIDetector:
    def __init__(self, config: SensitiveTokenConfig):
        self.config = config
        self.ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        self.phone_pattern = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
        self.email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    def scan_payload(self, payload: str) -> Tuple[bool, List[str]]:
        types = []
        if self.ssn_pattern.search(payload):
            types.append("SSN")
        if self.phone_pattern.search(payload):
            types.append("PHONE")
        if self.email_pattern.search(payload):
            types.append("EMAIL")
        for token in self.config.personal_tokens + self.config.system_tokens:
            if token and token in payload:
                types.append("TOKEN")
        return (len(types) > 0, list(set(types)))


# ========== Geo resolver ==========

class GeoResolver:
    def __init__(self, home_country: str = "US", db_path: str = "GeoLite2-City.mmdb"):
        self.home_country = home_country
        self.db_path = db_path
        self._geo_db_loaded = False
        self._geo_reader = None

    def _lazy_load_geo_db(self):
        if self._geo_db_loaded:
            return
        self._geo_db_loaded = True
        try:
            import geoip2.database
            self._geo_reader = geoip2.database.Reader(self.db_path)
        except Exception:
            self._geo_reader = None

    def scan_ip(self, ip: str) -> Tuple[str, str, str, int, str, bool]:
        if ip == "127.0.0.1" or ip.startswith(("10.", "192.168.", "172.")):
            return self.home_country, "Local", "Local", 3, "High", False
        self._lazy_load_geo_db()
        if not self._geo_reader:
            return "??", "Unknown", "Unknown", 0, "Low", True
        try:
            resp = self._geo_reader.city(ip)
            country = resp.country.iso_code or "??"
            region = (resp.subdivisions.most_specific.name or "Unknown") if resp.subdivisions else "Unknown"
            city = resp.city.name or "Unknown"
            resolution = 0
            if country != "??":
                resolution = 1
            if region != "Unknown":
                resolution = 2
            if city != "Unknown":
                resolution = 3
            confidence = "High" if resolution == 3 else "Medium" if resolution == 2 else "Low"
            overseas = (country != self.home_country)
            return country, region, city, resolution, confidence, overseas
        except Exception:
            return "??", "Unknown", "Unknown", 0, "Low", True


# ========== Startup helpers ==========

def get_startup_dirs() -> List[Path]:
    dirs = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / r"Microsoft\Windows\Start Menu\Programs\Startup")
    programdata = os.environ.get("PROGRAMDATA")
    if programdata:
        dirs.append(Path(programdata) / r"Microsoft\Windows\Start Menu\Programs\Startup")
    return dirs


def is_safe_startup_py(proc: psutil.Process, startup_dirs: List[Path]) -> bool:
    try:
        cmdline = proc.cmdline()
    except Exception:
        cmdline = []
    for arg in cmdline[1:]:
        p = Path(arg)
        if p.suffix.lower() == ".py":
            for sdir in startup_dirs:
                try:
                    if sdir.exists() and sdir in p.parents:
                        return True
                except Exception:
                    continue
    return False


# ========== Origin classification ==========

def classify_origin(proc: psutil.Process) -> str:
    try:
        name = proc.name().lower()
    except Exception:
        name = "unknown"
    try:
        exe = proc.exe().lower()
    except Exception:
        exe = ""

    if "back4blood" in exe or "back4blood" in name:
        return "Game.Back4Blood"
    if "steam.exe" in exe or "steam" in name:
        return "Steam.Client"
    if "epicgameslauncher" in exe or "epic" in name:
        return "Epic.Launcher"
    if "svchost" in name:
        return "Windows.ServiceHost"
    if "explorer" in name:
        return "Windows.Shell"
    if "system" == name:
        return "Windows.Core"
    if "nvidia" in exe or "nvcontainer" in name:
        return "NVIDIA.Telemetry"
    if "chrome" in name:
        return "Browser.Chrome"
    if "msedge" in name or "edge" in name:
        return "Browser.Edge"
    if "firefox" in name:
        return "Browser.Firefox"
    if "python" in name:
        return "Python.Script"
    if name == "unknown":
        return "Unknown.Process"
    return f"App.{name}"


def is_browser_origin(origin: str) -> bool:
    return origin.startswith("Browser.")


# ========== Predictive risk engine with deviation ==========

class RiskEngine:
    def __init__(self, history: Dict[Tuple[str, str], dict]):
        self.history = history

    def compute_risk(self, event: ConnectionEvent) -> int:
        base = self._compute_base_risk(event)
        deviation = self._compute_deviation(event)
        event.deviation_score = deviation
        risk = base + deviation
        if self.is_anomalous(event):
            risk += 10
        return max(0, min(100, risk))

    def _compute_base_risk(self, event: ConnectionEvent) -> int:
        key = (event.process_name, event.remote_addr)
        h = self.history.get(key, {})
        risk = 0

        if not event.safe_startup:
            risk += 10
        if event.overseas:
            risk += 25
        if not h:
            risk += 20
        else:
            if h.get("count", 1) < 3:
                risk += 10
        if is_browser_origin(event.origin):
            risk += 15
        if event.pii_detected:
            risk += 30
        if event.heavy_watch:
            risk += 15

        proc_key = (event.process_name, "__ALL__")
        ph = self.history.get(proc_key, {})
        distinct_remotes = ph.get("distinct_remotes", 1)
        if distinct_remotes > 10:
            risk += 15
        elif distinct_remotes > 5:
            risk += 8
        countries_seen: Set[str] = set(ph.get("countries", []))
        if event.country not in countries_seen and event.country not in ("??", ""):
            risk += 10
        return risk

    def _compute_deviation(self, event: ConnectionEvent) -> int:
        proc_key = (event.process_name, "__ALL__")
        ph = self.history.get(proc_key, {})
        last_ts = ph.get("last_seen_ts", 0)
        burst_count = ph.get("burst_count", 0)
        now = event.timestamp
        deviation = 0

        if last_ts > 0 and (now - last_ts) < 10:
            burst_count += 1
        else:
            burst_count = 1
        ph["burst_count"] = burst_count
        ph["last_seen_ts"] = now
        self.history[proc_key] = ph

        if burst_count >= 5:
            deviation += 15
        elif burst_count >= 3:
            deviation += 8

        countries = set(ph.get("countries", []))
        if event.country not in ("", "??") and event.country not in countries and len(countries) >= 2:
            deviation += 10

        return deviation

    def is_anomalous(self, event: ConnectionEvent) -> bool:
        proc_key = (event.process_name, "__ALL__")
        ph = self.history.get(proc_key, {})
        countries = set(ph.get("countries", []))
        remotes = set(ph.get("remotes", []))
        new_country = event.country not in ("", "??") and event.country not in countries
        new_remote = event.remote_addr not in remotes
        return new_country or new_remote


# ========== BoardGuard monitor ==========

class BoardGuardMonitor(threading.Thread):
    def __init__(self,
                 event_queue: queue.Queue,
                 pii_detector: PIIDetector,
                 geo_resolver: GeoResolver,
                 poll_interval: float = 2.0):
        super().__init__(daemon=True)
        self.event_queue = event_queue
        self.pii_detector = pii_detector
        self.geo_resolver = geo_resolver
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._seen_connections = set()
        self.startup_dirs = get_startup_dirs()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self.scan_connections()
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def scan_connections(self):
        for conn in psutil.net_connections(kind='inet'):
            if not conn.raddr:
                continue
            pid = conn.pid or -1
            laddr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
            raddr = conn.raddr.ip
            rport = conn.raddr.port
            key = (pid, laddr, raddr, rport)
            if key in self._seen_connections:
                continue
            self._seen_connections.add(key)

            try:
                proc = psutil.Process(pid) if pid > 0 else None
                pname = proc.name() if proc else "UNKNOWN"
            except Exception:
                proc = None
                pname = "UNKNOWN"

            safe_startup = False
            origin = "Unknown.Process"
            if proc is not None:
                safe_startup = is_safe_startup_py(proc, self.startup_dirs)
                origin = classify_origin(proc)

            country, region, city, resolution, confidence, overseas = self.geo_resolver.scan_ip(raddr)

            if raddr == "127.0.0.1":
                pii_detected = False
                pii_types = []
            else:
                if safe_startup:
                    pii_detected = False
                    pii_types = []
                else:
                    simulated_payload = f"{pname} talking to {raddr}"
                    pii_detected, pii_types = self.pii_detector.scan_payload(simulated_payload)

            heavy_watch = True

            event = ConnectionEvent(
                timestamp=time.time(),
                direction="OUTBOUND",
                process_name=pname,
                pid=pid,
                origin=origin,
                local_addr=laddr,
                remote_addr=raddr,
                remote_port=rport,
                country=country,
                region=region,
                city=city,
                resolution=resolution,
                confidence=confidence,
                overseas=overseas,
                pii_detected=pii_detected,
                pii_types=pii_types,
                safe_startup=safe_startup,
                heavy_watch=heavy_watch
            )
            self.event_queue.put({"type": "connection", "payload": event})


# ========== Event Horizon window (predictive cockpit) ==========

class EventHorizonWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EVENT HORIZON — BORG SYSTEM CORE")
        self.resize(900, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        header_layout = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel("EVENT HORIZON — OPTIMIZED SYSTEM HEALTH")
        self.title_label.setStyleSheet("color:#22c55e; font-size:18px; font-weight:bold;")
        header_layout.addWidget(self.title_label)

        self.heartbeat_label = QtWidgets.QLabel("HEARTBEAT: ●")
        self.heartbeat_label.setStyleSheet("color:#22c55e; font-weight:bold;")
        header_layout.addWidget(self.heartbeat_label)

        self.state_label = QtWidgets.QLabel("AI STATE: ACTIVE")
        self.state_label.setStyleSheet("color:#38bdf8; font-weight:bold;")
        header_layout.addWidget(self.state_label)

        header_layout.addStretch()
        root.addLayout(header_layout)

        health_group = QtWidgets.QGroupBox("System Health")
        health_group.setStyleSheet(
            "QGroupBox { border: 1px solid #22c55e; margin-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }"
        )
        health_layout = QtWidgets.QGridLayout(health_group)
        self.cpu_label = QtWidgets.QLabel("CPU: 0%")
        self.ram_label = QtWidgets.QLabel("RAM: 0%")
        self.disk_label = QtWidgets.QLabel("Disk: 0%")
        self.net_label = QtWidgets.QLabel("Net: 0 KB/s")
        for lbl in (self.cpu_label, self.ram_label, self.disk_label, self.net_label):
            lbl.setStyleSheet("color:#e5e7eb;")
        health_layout.addWidget(self.cpu_label, 0, 0)
        health_layout.addWidget(self.ram_label, 0, 1)
        health_layout.addWidget(self.disk_label, 1, 0)
        health_layout.addWidget(self.net_label, 1, 1)
        root.addWidget(health_group)

        cog_group = QtWidgets.QGroupBox("Cognitive Stream")
        cog_group.setStyleSheet(
            "QGroupBox { border: 1px solid #22c55e; margin-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }"
        )
        cog_layout = QtWidgets.QVBoxLayout(cog_group)
        self.cog_view = QtWidgets.QTextEdit()
        self.cog_view.setReadOnly(True)
        self.cog_view.setStyleSheet("background-color:#020617; color:#a5b4fc;")
        cog_layout.addWidget(self.cog_view)
        root.addWidget(cog_group, 2)

        panels_layout = QtWidgets.QHBoxLayout()
        self.allow_list = self._make_list_panel("ALLOW", "#22c55e")
        self.block_list = self._make_list_panel("BLOCK", "#ef4444")
        self.radio_list = self._make_list_panel("RADIOACTIVE", "#a855f7")
        self.pending_list = self._make_list_panel("PENDING / UNKNOWN", "#f59e0b")
        panels_layout.addWidget(self.allow_list["group"])
        panels_layout.addWidget(self.block_list["group"])
        panels_layout.addWidget(self.radio_list["group"])
        panels_layout.addWidget(self.pending_list["group"])
        root.addLayout(panels_layout, 3)

        self.last_net = psutil.net_io_counters()
        self.health_timer = QtCore.QTimer(self)
        self.health_timer.timeout.connect(self.update_health)
        self.health_timer.start(1000)

        self.cog_timer = QtCore.QTimer(self)
        self.cog_timer.timeout.connect(self.update_cognitive_stream)
        self.cog_timer.start(3000)

        self.heartbeat_state = True
        self.heartbeat_timer = QtCore.QTimer(self)
        self.heartbeat_timer.timeout.connect(self.toggle_heartbeat)
        self.heartbeat_timer.start(1000)

        for panel in (self.allow_list, self.block_list, self.radio_list, self.pending_list):
            panel["list"].setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            panel["list"].customContextMenuRequested.connect(self.show_item_menu)

        self.recent_risks: List[int] = []
        self.recent_deviations: List[int] = []

    def _make_list_panel(self, title: str, color: str):
        group = QtWidgets.QGroupBox(title)
        group.setStyleSheet(
            "QGroupBox { border: 1px solid " + color + "; margin-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color:" + color + "; }"
        )
        layout = QtWidgets.QVBoxLayout(group)
        lst = QtWidgets.QListWidget()
        lst.setStyleSheet("background-color:#020617; color:#e5e7eb;")
        layout.addWidget(lst)
        return {"group": group, "list": lst, "color": color}

    def add_threat_item(self, proc: str, remote: str, context: str, risk: int,
                        deviation: int, bucket: str):
        text = f"{proc} → {remote} | {context} | Risk {risk} | Dev {deviation}"
        item = QtWidgets.QListWidgetItem(text)
        if bucket == "allow":
            self.allow_list["list"].addItem(item)
        elif bucket == "block":
            self.block_list["list"].addItem(item)
        elif bucket == "radio":
            self.radio_list["list"].addItem(item)
        else:
            self.pending_list["list"].addItem(item)

    def reflect_event(self, event: ConnectionEvent, action: str):
        context_bits = []
        if event.overseas:
            context_bits.append("Overseas")
        if event.pii_detected:
            context_bits.append("PII")
        if event.deviation_score >= 10:
            context_bits.append("Deviation Spike")
        if not context_bits:
            context_bits.append("Baseline")
        context = ", ".join(context_bits)

        if action in ("auto_block", "manual_block", "quarantine_block"):
            bucket = "block"
        elif action in ("manual_allow", "pending_trust", "auto_promote_allow"):
            bucket = "allow"
        elif action in ("auto_radioactive", "manual_radioactive", "auto_demote_radioactive"):
            bucket = "radio"
        else:
            bucket = "pending"

        self.add_threat_item(
            event.process_name,
            event.remote_addr,
            context,
            event.risk_score,
            event.deviation_score,
            bucket
        )

        self.recent_risks.append(event.risk_score)
        self.recent_deviations.append(event.deviation_score)
        if len(self.recent_risks) > 50:
            self.recent_risks = self.recent_risks[-50:]
        if len(self.recent_deviations) > 50:
            self.recent_deviations = self.recent_deviations[-50:]

    def update_health(self):
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        now_net = psutil.net_io_counters()
        sent_diff = now_net.bytes_sent - self.last_net.bytes_sent
        recv_diff = now_net.bytes_recv - self.last_net.bytes_recv
        self.last_net = now_net
        kbps = (sent_diff + recv_diff) / 1024.0
        self.cpu_label.setText(f"CPU: {cpu:.0f}%")
        self.ram_label.setText(f"RAM: {ram:.0f}%")
        self.disk_label.setText(f"Disk: {disk:.0f}%")
        self.net_label.setText(f"Net: {kbps:.1f} KB/s")

    def update_cognitive_stream(self):
        if self.recent_risks:
            avg_risk = sum(self.recent_risks) / len(self.recent_risks)
        else:
            avg_risk = 0.0
        if self.recent_deviations:
            avg_dev = sum(self.recent_deviations) / len(self.recent_deviations)
        else:
            avg_dev = 0.0

        if avg_risk >= 60 or avg_dev >= 10:
            forecast = "[FORECAST] High deviation regime — expect new overseas or PII spikes."
            self.state_label.setText("AI STATE: ALERT")
            self.state_label.setStyleSheet("color:#f97316; font-weight:bold;")
        elif avg_risk >= 30:
            forecast = "[FORECAST] Mixed baseline — watching for emerging anomalies."
            self.state_label.setText("AI STATE: WATCHING")
            self.state_label.setStyleSheet("color:#eab308; font-weight:bold;")
        else:
            forecast = "[FORECAST] Stable baseline — no major deviations."
            self.state_label.setText("AI STATE: ACTIVE")
            self.state_label.setStyleSheet("color:#38bdf8; font-weight:bold;")

        lines = [
            f"[METRIC] Avg Risk: {avg_risk:.1f} | Avg Deviation: {avg_dev:.1f}",
            forecast,
            "[TRACK] Monitoring high-risk overseas flows and burst patterns...",
            "[PREDICT] Evaluating next likely deviation window..."
        ]
        self.cog_view.append("\n".join(lines))
        self.cog_view.append("-" * 40)
        self.cog_view.verticalScrollBar().setValue(self.cog_view.verticalScrollBar().maximum())

    def toggle_heartbeat(self):
        self.heartbeat_state = not self.heartbeat_state
        if self.heartbeat_state:
            self.heartbeat_label.setText("HEARTBEAT: ●")
            self.heartbeat_label.setStyleSheet("color:#22c55e; font-weight:bold;")
        else:
            self.heartbeat_label.setText("HEARTBEAT: ○")
            self.heartbeat_label.setStyleSheet("color:#16a34a; font-weight:bold;")

    def show_item_menu(self, pos: QtCore.QPoint):
        sender_list: QtWidgets.QListWidget = self.sender()
        item = sender_list.itemAt(pos)
        if not item:
            return
        menu = QtWidgets.QMenu(self)
        allow_action = menu.addAction("Move to ALLOW")
        block_action = menu.addAction("Move to BLOCK")
        radio_action = menu.addAction("Move to RADIOACTIVE")
        pending_action = menu.addAction("Move to PENDING")
        action = menu.exec_(sender_list.mapToGlobal(pos))
        if not action:
            return
        text = item.text()
        sender_list.takeItem(sender_list.row(item))
        if action == allow_action:
            self.allow_list["list"].addItem(text)
        elif action == block_action:
            self.block_list["list"].addItem(text)
        elif action == radio_action:
            self.radio_list["list"].addItem(text)
        elif action == pending_action:
            self.pending_list["list"].addItem(text)
        # Visual only; QueenGuard owns real policy.


# ========== QueenGuard window (predictive + policy evolution + clustering + self-heal) ==========

class QueenWindow(QtWidgets.QMainWindow):
    MODE_AUTOMATED = "Automated Sentinel"

    # Stability thresholds
    STABILITY_DAYS_REQUIRED = 3
    STABILITY_MIN_COUNT = 5
    STABILITY_MAX_RISK = 25
    STABILITY_MAX_DEVIATION = 5

    def __init__(self, event_queue: queue.Queue, memory: RebootMemoryManager,
                 swarm: SwarmSyncManager,
                 horizon: Optional[EventHorizonWindow] = None):
        super().__init__()
        self.event_queue = event_queue
        self.memory = memory
        self.swarm = swarm
        self.horizon = horizon
        self.policy: Dict[Tuple[str, str], PolicyDecision] = {}
        self.history: Dict[Tuple[str, str], dict] = {}
        self.mode = self.MODE_AUTOMATED
        self.quarantined_procs: Set[str] = set()
        self.event_log: List[dict] = []
        self.timeline: List[dict] = []
        self.threat_clusters: Dict[str, dict] = {}
        self.process_fingerprints: Dict[str, dict] = {}

        saved = self.memory.load()
        if "policies" in saved:
            for key_str, val in saved["policies"].items():
                try:
                    proc_name, remote_addr = key_str.split("||", 1)
                except ValueError:
                    continue
                self.policy[(proc_name, remote_addr)] = PolicyDecision(
                    allow_always=val.get("allow_always", False),
                    block_always=val.get("block_always", False),
                    radioactive=val.get("radioactive", False)
                )
        if "history" in saved:
            for key_str, h in saved["history"].items():
                try:
                    proc_name, remote_addr = key_str.split("||", 1)
                except ValueError:
                    continue
                self.history[(proc_name, remote_addr)] = h

        self.risk_engine = RiskEngine(self.history)

        self._build_gui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.drain_events)
        self.timer.start(500)

        self.save_timer = QtCore.QTimer(self)
        self.save_timer.timeout.connect(self.save_state)
        self.save_timer.start(5000)

        self.swarm_timer = QtCore.QTimer(self)
        self.swarm_timer.timeout.connect(self.swarm_sync)
        self.swarm_timer.start(15000)

    def _build_gui(self):
        self.setWindowTitle("Queen Guard - Personal Data Sentinel (Automated)")
        self.resize(900, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        top_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Status: Automated Monitoring (Predictive, Self-Healing, Swarm-Synced)")
        self.status_label.setStyleSheet("color: #22c55e; font-weight: bold;")
        top_layout.addWidget(self.status_label)

        self.mem_label = QtWidgets.QLabel(f"Memory: {self.memory.get_storage_location()}")
        self.mem_label.setStyleSheet("color: gray; font-size: 10px;")
        top_layout.addWidget(self.mem_label)
        main_layout.addLayout(top_layout)

        legend_layout = QtWidgets.QHBoxLayout()
        legend_label = QtWidgets.QLabel(
            "Legend:  "
            "<span style='background-color:#14532d;color:#e5e7eb'>&nbsp;&nbsp;&nbsp;</span> Always Allow   "
            "<span style='background-color:#7f1d1d;color:#e5e7eb'>&nbsp;&nbsp;&nbsp;</span> Always Block   "
            "<span style='background-color:#4c1d95;color:#e5e7eb'>&nbsp;&nbsp;&nbsp;</span> Radioactive   "
            "<span style='background-color:#854d0e;color:#e5e7eb'>&nbsp;&nbsp;&nbsp;</span> Pending Admin Approval"
        )
        legend_label.setStyleSheet("font-size: 12px;")
        legend_layout.addWidget(legend_label)
        main_layout.addLayout(legend_layout)

        body_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(body_layout)

        left_panel = QtWidgets.QVBoxLayout()
        self.btn_allow = QtWidgets.QPushButton("Always Allow")
        self.btn_allow.clicked.connect(self.manual_always_allow)
        left_panel.addWidget(self.btn_allow)

        self.btn_block = QtWidgets.QPushButton("Always Block")
        self.btn_block.clicked.connect(self.manual_always_block)
        left_panel.addWidget(self.btn_block)

        self.btn_radio = QtWidgets.QPushButton("Keep Radioactive")
        self.btn_radio.clicked.connect(self.manual_keep_radioactive)
        left_panel.addWidget(self.btn_radio)

        self.btn_approve = QtWidgets.QPushButton("Approve Trust")
        self.btn_approve.clicked.connect(self.manual_approve_trust)
        left_panel.addWidget(self.btn_approve)

        self.btn_quarantine = QtWidgets.QPushButton("Quarantine Process")
        self.btn_quarantine.clicked.connect(self.quarantine_selected_process)
        left_panel.addWidget(self.btn_quarantine)

        self.btn_timeline = QtWidgets.QPushButton("Threat Timeline")
        self.btn_timeline.clicked.connect(self.show_timeline)
        left_panel.addWidget(self.btn_timeline)

        self.btn_proc_dash = QtWidgets.QPushButton("Process Dashboard")
        self.btn_proc_dash.clicked.connect(self.show_process_dashboard)
        left_panel.addWidget(self.btn_proc_dash)

        self.btn_clusters = QtWidgets.QPushButton("Threat Clusters")
        self.btn_clusters.clicked.connect(self.show_clusters)
        left_panel.addWidget(self.btn_clusters)

        self.btn_export = QtWidgets.QPushButton("Export JSON")
        self.btn_export.clicked.connect(self.export_json)
        left_panel.addWidget(self.btn_export)

        left_panel.addStretch()
        body_layout.addLayout(left_panel, 1)

        center_layout = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget(0, 19)
        self.table.setHorizontalHeaderLabels([
            "Time", "Dir", "Process", "PID", "Origin",
            "Local", "Remote",
            "Country", "Region", "City", "Res", "Conf",
            "Overseas", "PII", "Risk", "Deviation", "Watch", "Radioactive", "Pending"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        center_layout.addWidget(self.table)
        body_layout.addLayout(center_layout, 4)

        right_panel = QtWidgets.QVBoxLayout()
        self.save_local_btn = QtWidgets.QPushButton("Save Memory (Local)")
        self.save_local_btn.clicked.connect(self.manual_save_local)
        right_panel.addWidget(self.save_local_btn)

        self.save_smb_btn = QtWidgets.QPushButton("Save Memory (SMB)")
        self.save_smb_btn.clicked.connect(self.manual_save_smb)
        right_panel.addWidget(self.save_smb_btn)

        self.kill_btn = QtWidgets.QPushButton("Kill Selected Process")
        self.kill_btn.clicked.connect(self.kill_selected_process)
        right_panel.addWidget(self.kill_btn)

        right_panel.addStretch()
        body_layout.addLayout(right_panel, 1)

        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_table_context_menu)

    # ---------- swarm sync ----------

    def swarm_sync(self):
        local_state = self._serialize_state()
        merged = self.swarm.merge(local_state)
        # Pull merged policies/history back in
        for key_str, val in merged.get("policies", {}).items():
            try:
                proc_name, remote_addr = key_str.split("||", 1)
            except ValueError:
                continue
            self.policy[(proc_name, remote_addr)] = PolicyDecision(
                allow_always=val.get("allow_always", False),
                block_always=val.get("block_always", False),
                radioactive=val.get("radioactive", False)
            )
        for key_str, h in merged.get("history", {}).items():
            try:
                proc_name, remote_addr = key_str.split("||", 1)
            except ValueError:
                continue
            self.history[(proc_name, remote_addr)] = h

    # ---------- event draining ----------

    def drain_events(self):
        while True:
            try:
                msg = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if msg["type"] == "connection":
                event: ConnectionEvent = msg["payload"]
                self.handle_event(event)

    # ---------- history + fingerprints + clusters ----------

    def update_history(self, event: ConnectionEvent):
        key = (event.process_name, event.remote_addr)
        h = self.history.get(key, {})

        h["count"] = h.get("count", 0) + 1
        h["first_seen"] = h.get("first_seen", event.timestamp)
        h["last_seen"] = event.timestamp

        if "stable_days" not in h:
            h["stable_days"] = 0
        if "last_stable_ts" not in h:
            h["last_stable_ts"] = event.timestamp
        if "ever_unstable" not in h:
            h["ever_unstable"] = False

        h["overseas"] = event.overseas
        h["country"] = event.country
        h["region"] = event.region
        h["port"] = event.remote_port
        h["origin"] = event.origin

        self.history[key] = h

        proc_key = (event.process_name, "__ALL__")
        ph = self.history.get(proc_key, {})
        remotes = set(ph.get("remotes", []))
        remotes.add(event.remote_addr)
        ph["remotes"] = list(remotes)
        ph["distinct_remotes"] = len(remotes)

        countries = set(ph.get("countries", []))
        if event.country not in ("", "??"):
            countries.add(event.country)
        ph["countries"] = list(countries)

        self.history[proc_key] = ph

        # Per-process fingerprint
        fp = self.process_fingerprints.get(event.process_name, {
            "countries": set(),
            "ports": set(),
            "avg_risk": 0.0,
            "avg_dev": 0.0,
            "count": 0,
        })
        fp["countries"].add(event.country)
        fp["ports"].add(event.remote_port)
        c = fp["count"]
        fp["avg_risk"] = (fp["avg_risk"] * c + event.risk_score) / (c + 1)
        fp["avg_dev"] = (fp["avg_dev"] * c + event.deviation_score) / (c + 1)
        fp["count"] = c + 1
        self.process_fingerprints[event.process_name] = fp

        # Threat clustering
        cluster_key = f"{event.origin}|{event.country}|{event.remote_port}"
        cl = self.threat_clusters.get(cluster_key, {
            "origin": event.origin,
            "country": event.country,
            "port": event.remote_port,
            "count": 0,
            "max_risk": 0,
            "processes": set(),
        })
        cl["count"] += 1
        cl["max_risk"] = max(cl["max_risk"], event.risk_score)
        cl["processes"].add(event.process_name)
        self.threat_clusters[cluster_key] = cl

    # ---------- stability evaluation + self-healing ----------

    def evaluate_stability(self, event: ConnectionEvent, h: dict) -> str:
        if event.overseas or event.pii_detected:
            h["ever_unstable"] = True
            return "unstable"
        if event.risk_score > self.STABILITY_MAX_RISK or event.deviation_score > self.STABILITY_MAX_DEVIATION:
            h["ever_unstable"] = True
            return "unstable"

        days_seen = (event.timestamp - h.get("first_seen", event.timestamp)) / 86400
        if days_seen >= self.STABILITY_DAYS_REQUIRED and h.get("count", 0) >= self.STABILITY_MIN_COUNT:
            # self-healing: if stable for long, clear ever_unstable
            if h.get("ever_unstable", False):
                h["ever_unstable"] = False
            return "stable"

        return "neutral"

    def evolve_policy(self, event: ConnectionEvent, stability: str):
        key = (event.process_name, event.remote_addr)
        decision = self.policy.get(key, PolicyDecision(False, False, False))

        if stability == "unstable":
            if decision.allow_always:
                decision.allow_always = False
                decision.radioactive = True
                self.policy[key] = decision
                self.log_event(event, "auto_demote_radioactive")
            return

        if stability == "stable":
            if not decision.allow_always and not decision.block_always:
                decision.allow_always = True
                decision.radioactive = False
                self.policy[key] = decision
                self.log_event(event, "auto_promote_allow")
            return

    # ---------- main event handler ----------

    def _auto_trusted(self, event: ConnectionEvent) -> bool:
        if event.process_name == "UNKNOWN":
            return False
        if event.origin.startswith("Unknown"):
            return False
        if event.overseas:
            return False
        if event.risk_score >= 30:
            return False
        if event.resolution >= 2 and event.confidence in ("Medium", "High"):
            return True
        return False

    def handle_event(self, event: ConnectionEvent):
        event.heavy_watch = True
        event.pending_approval = False

        self.update_history(event)
        h = self.history[(event.process_name, event.remote_addr)]

        self.risk_engine.history = self.history
        event.risk_score = self.risk_engine.compute_risk(event)

        stability = self.evaluate_stability(event, h)
        self.evolve_policy(event, stability)

        key = (event.process_name, event.remote_addr)
        decision = self.policy.get(key)

        if event.process_name in self.quarantined_procs:
            row = self.add_event_row(event, forced_block=True)
            self._recolor_row_policy(row, "block")
            self.log_event(event, "quarantine_block")
            return

        if event.risk_score >= 80 or (event.overseas and event.pii_detected):
            row = self.add_event_row(event, forced_block=True)
            self._recolor_row_policy(row, "block")
            self.log_event(event, "auto_block")
            return

        if decision:
            if decision.block_always:
                row = self.add_event_row(event, forced_block=True)
                self._recolor_row_policy(row, "block")
                self.log_event(event, "manual_block")
                return
            if decision.allow_always:
                event.heavy_watch = False
                row = self.add_event_row(event, forced_allow=True)
                self._recolor_row_policy(row, "allow")
                self.log_event(event, "manual_allow")
                return
            if decision.radioactive:
                row = self.add_event_row(event, forced_allow=True)
                self._recolor_row_policy(row, "radioactive")
                self.log_event(event, "manual_radioactive")
                return

        if self._auto_trusted(event):
            event.pending_approval = True
            row = self.add_event_row(event, forced_allow=True, pending=True)
            self._recolor_row_policy(row, "pending")
            self.log_event(event, "pending_trust")
            return

        row = self.add_event_row(event, forced_allow=True)
        self._recolor_row_policy(row, "radioactive")
        self.log_event(event, "auto_radioactive")

    # ---------- table + UI helpers ----------

    def add_event_row(self, event: ConnectionEvent,
                      forced_allow: bool = False,
                      forced_block: bool = False,
                      pending: bool = False) -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)

        t_str = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        pii_str = "YES" if event.pii_detected else "NO"
        overseas_str = "YES" if event.overseas else "NO"
        risk_str = str(event.risk_score)
        dev_str = str(event.deviation_score)
        watch_str = "HEAVY" if event.heavy_watch else ""
        radioactive_str = "YES" if event.heavy_watch else "NO"
        pending_str = "YES" if pending else "NO"
        res_str = f"{event.resolution}/3"

        values = [
            t_str,
            event.direction,
            event.process_name,
            str(event.pid),
            event.origin,
            event.local_addr,
            f"{event.remote_addr}:{event.remote_port}",
            event.country,
            event.region,
            event.city,
            res_str,
            event.confidence,
            overseas_str,
            pii_str,
            risk_str,
            dev_str,
            watch_str,
            radioactive_str,
            pending_str
        ]

        for col, val in enumerate(values):
            item = QtWidgets.QTableWidgetItem(val)
            if forced_block:
                item.setBackground(QtGui.QColor("#7f1d1d"))
            elif pending:
                item.setBackground(QtGui.QColor("#854d0e"))
            elif event.heavy_watch:
                item.setBackground(QtGui.QColor("#4c1d95"))
            else:
                if event.risk_score >= 70:
                    item.setBackground(QtGui.QColor("#7f1d1d"))
                elif event.risk_score >= 40:
                    item.setBackground(QtGui.QColor("#854d0e"))
            self.table.setItem(row, col, item)

        self.table.scrollToBottom()
        return row

    def _get_selected_row(self) -> int:
        return self.table.currentRow()

    def _get_selected_key_from_row(self, row: int) -> Optional[Tuple[str, str]]:
        proc_item = self.table.item(row, 2)
        remote_item = self.table.item(row, 6)
        if not proc_item or not remote_item:
            return None
        process_name = proc_item.text()
        remote_addr = remote_item.text().split(":", 1)[0]
        return process_name, remote_addr

    def _get_selected_key(self):
        row = self._get_selected_row()
        if row < 0:
            return None
        key = self._get_selected_key_from_row(row)
        if not key:
            return None
        process_name, remote_addr = key
        return process_name, remote_addr, row

    def manual_always_allow(self):
        key = self._get_selected_key()
        if not key:
            return
        process_name, remote_addr, row = key
        dec = PolicyDecision(True, False, False)
        self.policy[(process_name, remote_addr)] = dec
        self._recolor_row_policy(row, "allow")
        pending_item = self.table.item(row, 18)
        if pending_item:
            pending_item.setText("NO")

    def manual_always_block(self):
        key = self._get_selected_key()
        if not key:
            return
        process_name, remote_addr, row = key
        dec = PolicyDecision(False, True, False)
        self.policy[(process_name, remote_addr)] = dec
        self._recolor_row_policy(row, "block")

    def manual_keep_radioactive(self):
        key = self._get_selected_key()
        if not key:
            return
        process_name, remote_addr, row = key
        dec = PolicyDecision(False, False, True)
        self.policy[(process_name, remote_addr)] = dec
        self._recolor_row_policy(row, "radioactive")

    def manual_approve_trust(self):
        key = self._get_selected_key()
        if not key:
            return
        process_name, remote_addr, row = key
        dec = PolicyDecision(True, False, False)
        self.policy[(process_name, remote_addr)] = dec
        self._recolor_row_policy(row, "allow")
        pending_item = self.table.item(row, 18)
        if pending_item:
            pending_item.setText("NO")

    def _recolor_row_policy(self, row: int, mode: str):
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if not item:
                continue
            if mode == "allow":
                item.setBackground(QtGui.QColor("#14532d"))
            elif mode == "block":
                item.setBackground(QtGui.QColor("#7f1d1d"))
            elif mode == "radioactive":
                item.setBackground(QtGui.QColor("#4c1d95"))
            elif mode == "pending":
                item.setBackground(QtGui.QColor("#854d0e"))

    def show_table_context_menu(self, pos: QtCore.QPoint):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        menu = QtWidgets.QMenu(self)
        always_allow_action = menu.addAction("Always Allow")
        always_block_action = menu.addAction("Always Block")
        radioactive_action = menu.addAction("Keep Radioactive")
        approve_action = menu.addAction("Approve Trust")
        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if not action:
            return
        key = self._get_selected_key_from_row(row)
        if not key:
            return
        process_name, remote_addr = key
        decision = self.policy.get((process_name, remote_addr), PolicyDecision(False, False, False))
        if action == always_allow_action:
            decision.allow_always = True
            decision.block_always = False
            decision.radioactive = False
            self.policy[(process_name, remote_addr)] = decision
            self._recolor_row_policy(row, "allow")
            pending_item = self.table.item(row, 18)
            if pending_item:
                pending_item.setText("NO")
        elif action == always_block_action:
            decision.allow_always = False
            decision.block_always = True
            decision.radioactive = False
            self.policy[(process_name, remote_addr)] = decision
            self._recolor_row_policy(row, "block")
        elif action == radioactive_action:
            decision.allow_always = False
            decision.block_always = False
            decision.radioactive = True
            self.policy[(process_name, remote_addr)] = decision
            self._recolor_row_policy(row, "radioactive")
        elif action == approve_action:
            decision.allow_always = True
            decision.block_always = False
            decision.radioactive = False
            self.policy[(process_name, remote_addr)] = decision
            self._recolor_row_policy(row, "allow")
            pending_item = self.table.item(row, 18)
            if pending_item:
                pending_item.setText("NO")

    def quarantine_selected_process(self):
        row = self._get_selected_row()
        if row < 0:
            return
        proc_item = self.table.item(row, 2)
        if not proc_item:
            return
        process_name = proc_item.text()
        self.quarantined_procs.add(process_name)

    def show_timeline(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Threat Timeline")
        dlg.resize(800, 400)
        layout = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["Time", "Process", "Remote", "Action/Risk"])
        layout.addWidget(table)
        for entry in sorted(self.timeline, key=lambda x: x["timestamp"]):
            row = table.rowCount()
            table.insertRow(row)
            t_str = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(t_str))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(entry["process"]))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(entry["remote"]))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{entry['action']} (risk={entry['risk']})"))
        dlg.exec_()

    def show_process_dashboard(self):
        row = self._get_selected_row()
        if row < 0:
            return
        proc_item = self.table.item(row, 2)
        if not proc_item:
            return
        process_name = proc_item.text()
        per_remote = {
            remote: h for (p, remote), h in self.history.items()
            if p == process_name and remote != "__ALL__"
        }
        proc_key = (process_name, "__ALL__")
        aggregate = self.history.get(proc_key, {})
        fp = self.process_fingerprints.get(process_name, {
            "countries": set(),
            "ports": set(),
            "avg_risk": 0.0,
            "avg_dev": 0.0,
            "count": 0,
        })

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Process Dashboard - {process_name}")
        dlg.resize(900, 500)
        layout = QtWidgets.QVBoxLayout(dlg)
        agg_label = QtWidgets.QLabel(
            f"Distinct remotes: {aggregate.get('distinct_remotes', 0)} | "
            f"Countries: {', '.join(aggregate.get('countries', []))}"
        )
        layout.addWidget(agg_label)
        fp_label = QtWidgets.QLabel(
            f"Fingerprint: Countries={', '.join(fp.get('countries', []))} | "
            f"Ports={', '.join(str(p) for p in fp.get('ports', []))} | "
            f"AvgRisk={fp.get('avg_risk', 0):.1f} | AvgDev={fp.get('avg_dev', 0):.1f}"
        )
        layout.addWidget(fp_label)

        table = QtWidgets.QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["Remote", "Count", "Country", "Region", "Port", "Last Seen"])
        layout.addWidget(table)
        for remote, h in per_remote.items():
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(remote))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(h.get("count", 0))))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(h.get("country", "")))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(h.get("region", "")))
            table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(h.get("port", ""))))
            ts = h.get("last_seen", 0)
            t_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""
            table.setItem(row, 5, QtWidgets.QTableWidgetItem(t_str))
        dlg.exec_()

    def show_clusters(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Threat Clusters")
        dlg.resize(900, 500)
        layout = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(["Origin", "Country", "Port", "Count", "Max Risk / Processes"])
        layout.addWidget(table)
        for key, cl in self.threat_clusters.items():
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(cl["origin"]))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(cl["country"]))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(cl["port"])))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(cl["count"])))
            procs = ", ".join(sorted(cl["processes"]))
            table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{cl['max_risk']} / {procs}"))
        dlg.exec_()

    def export_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export JSON", "", "JSON Files (*.json)"
        )
        if not path:
            return
        state = self._serialize_state()
        data = {
            "events": self.event_log,
            "policies": state["policies"],
            "history": state["history"],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export JSON:\n{e}")

    def log_event(self, event: ConnectionEvent, action: str):
        d = asdict(event)
        d["action"] = action
        self.event_log.append(d)
        if event.risk_score >= 60 or action in ("auto_block", "quarantine_block", "auto_demote_radioactive"):
            self.timeline.append({
                "timestamp": event.timestamp,
                "process": event.process_name,
                "remote": event.remote_addr,
                "risk": event.risk_score,
                "action": action,
            })
        if self.horizon is not None:
            self.horizon.reflect_event(event, action)

    def _serialize_state(self) -> dict:
        policies_serialized = {}
        for (proc_name, remote_addr), dec in self.policy.items():
            key_str = f"{proc_name}||{remote_addr}"
            policies_serialized[key_str] = {
                "allow_always": dec.allow_always,
                "block_always": dec.block_always,
                "radioactive": dec.radioactive,
            }
        history_serialized = {}
        for (proc_name, remote_addr), h in self.history.items():
            key_str = f"{proc_name}||{remote_addr}"
            history_serialized[key_str] = h
        return {"policies": policies_serialized, "history": history_serialized}

    def save_state(self):
        state = self._serialize_state()
        self.memory.save(state)

    def _save_to_path(self, path: Path):
        state = self._serialize_state()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save memory:\n{e}")

    def manual_save_local(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Local Folder")
        if not folder:
            return
        path = Path(folder) / "queen_memory.json"
        self._save_to_path(path)
        QtWidgets.QMessageBox.information(self, "Saved", f"Memory saved to:\n{path}")

    def manual_save_smb(self):
        smb_path, ok = QtWidgets.QInputDialog.getText(
            self,
            "SMB Path",
            "Enter SMB path (e.g. \\\\SERVER\\Share):"
        )
        if not ok or not smb_path:
            return
        path = Path(smb_path) / "queen_memory.json"
        self._save_to_path(path)
        QtWidgets.QMessageBox.information(self, "Saved", f"Memory saved to:\n{path}")

    def kill_selected_process(self):
        row = self.table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Select a row first.")
            return
        pid_item = self.table.item(row, 3)
        if not pid_item:
            QtWidgets.QMessageBox.warning(self, "No PID", "No PID found for selected row.")
            return
        try:
            pid = int(pid_item.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid PID", "PID is not a valid number.")
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Kill Process",
            f"Are you sure you want to kill PID {pid}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        try:
            p = psutil.Process(pid)
            p.terminate()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to kill process:\n{e}")


# ========== MAIN ==========

def main():
    sensitive_config = SensitiveTokenConfig(personal_tokens=[], system_tokens=[])
    pii_detector = PIIDetector(sensitive_config)
    geo_resolver = GeoResolver(home_country="US", db_path="GeoLite2-City.mmdb")
    memory = RebootMemoryManager()
    swarm = SwarmSyncManager()
    event_q = queue.Queue()

    monitor = BoardGuardMonitor(event_q, pii_detector, geo_resolver, poll_interval=3.0)
    monitor.start()

    app = QtWidgets.QApplication(sys.argv)

    dark_qss = """
    QWidget { background-color: #0b1120; color: #e5e7eb; }
    QTableWidget { gridline-color: #1f2937; background-color: #020617; }
    QHeaderView::section { background-color: #111827; color: #e5e7eb; }
    QPushButton {
        background-color: #1f2933; color: #e5e7eb;
        border: 1px solid #374151; padding: 4px 8px;
    }
    QPushButton:hover { background-color: #111827; }
    QLineEdit, QPlainTextEdit, QTextEdit {
        background-color: #020617; color: #e5e7eb;
        border: 1px solid #374151;
    }
    QDialog { background-color: #020617; }
    """
    app.setStyleSheet(dark_qss)

    borg_win = EventHorizonWindow()
    queen_win = QueenWindow(event_q, memory, swarm=swarm, horizon=borg_win)

    queen_win.resize(900, 800)
    borg_win.resize(900, 800)

    queen_win.move(100, 100)
    borg_win.move(1100, 100)

    queen_win.show()
    borg_win.show()

    ret = app.exec_()
    monitor.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()

