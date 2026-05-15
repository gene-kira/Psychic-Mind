#!/usr/bin/env python3
"""
AI Borg Secure Connection Organism (Full Unified File, Autonomous)

Features:
- Autoloads required libraries
- PySide6 cockpit UI
- Four Queens:
    * QueenGuardian: Global TLS Guardian (all connections go through here)
    * QueenScanner: Background Scanner + Enforcement (always running)
    * QueenLockdown: Full Lockdown Mode (zero-trust perimeter + manual override)
    * QueenResolver: Dual-Resolver DNS + IP verification (system DNS + 1.1.1.1)
- Unlimited workers vibe via dynamic threads
- IP verification against DNS and historical logs
- Autonomous behavior:
    * Scanner auto-reacts to threats (can auto-enable lockdown)
    * All queens run continuously; operator only overrides when desired
"""

import sys
import importlib
import traceback
import time
import struct
import random

# ---------------------------------------------------------------------------
# Autoloader for necessary libraries
# ---------------------------------------------------------------------------

REQUIRED_LIBS = [
    "ssl",
    "socket",
    "hashlib",
    "json",
    "threading",
    "queue",
]

def autoload_libraries():
    missing = []
    for name in REQUIRED_LIBS:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing required stdlib modules (unexpected): {missing}")

autoload_libraries()

import ssl
import socket
import hashlib
import json
import threading
import queue

# PySide6 must be installed via pip:
#   pip install PySide6
from PySide6.QtCore import Qt, QObject, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QFrame,
    QGridLayout,
    QSizePolicy,
    QCheckBox,
)


# ---------------------------------------------------------------------------
# Connection registry (for scanner, resolver, lockdown)
# ---------------------------------------------------------------------------

class ConnectionRegistry:
    """
    Central registry of all connection attempts and states.
    Stores host, port, IPs, security status, reasons, fingerprints.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._connections = []  # list of dicts

    def add_connection(self, info: dict) -> int:
        with self._lock:
            self._connections.append(info)
            return len(self._connections) - 1

    def update_connection(self, idx: int, **kwargs):
        with self._lock:
            if 0 <= idx < len(self._connections):
                self._connections[idx].update(kwargs)

    def snapshot(self):
        with self._lock:
            return list(self._connections)

    def get_previous_ips_for_host(self, host: str):
        with self._lock:
            ips = set()
            for c in self._connections:
                if c.get("host") == host:
                    for ip in c.get("ips", []):
                        ips.add(ip)
            return ips


# ---------------------------------------------------------------------------
# Lockdown policy engine (QueenLockdown brain)
# ---------------------------------------------------------------------------

class LockdownPolicy:
    """
    Full Lockdown Mode:
    - When enabled, only allow connections that match allowlists.
    - Manual override can temporarily bypass lockdown decisions.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = False
        self._manual_override = False

        # Simple allowlists (extend as needed)
        self.allowed_hosts = set()
        self.allowed_ports = set()
        self.allowed_hosts.add("example.com")
        self.allowed_ports.add(443)

    def set_enabled(self, enabled: bool):
        with self._lock:
            self._enabled = enabled

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_manual_override(self, enabled: bool):
        with self._lock:
            self._manual_override = enabled

    def is_manual_override(self) -> bool:
        with self._lock:
            return self._manual_override

    def decide(self, host: str, port: int) -> bool:
        """
        Return True if connection is allowed under current policy.
        If lockdown disabled → allow.
        If manual override enabled → allow.
        Else enforce allowlists.
        """
        with self._lock:
            if not self._enabled:
                return True
            if self._manual_override:
                return True

            host_ok = (host in self.allowed_hosts)
            port_ok = (port in self.allowed_ports)
            return host_ok and port_ok


# ---------------------------------------------------------------------------
# QueenResolver – Dual-Resolver DNS + IP verification
# ---------------------------------------------------------------------------

class QueenResolver(QObject):
    log_message = Signal(str)

    def __init__(self, registry: ConnectionRegistry, policy: LockdownPolicy, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._policy = policy

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _system_resolve(self, host: str):
        ips = set()
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            for family, _, _, _, sockaddr in infos:
                if family == socket.AF_INET:
                    ips.add(sockaddr[0])
        except Exception as e:
            self._log(f"[Resolver] System DNS resolve failed for {host}: {e}")
        return ips

    def _encode_dns_name(self, host: str) -> bytes:
        parts = host.strip(".").split(".")
        out = b""
        for p in parts:
            out += struct.pack("!B", len(p)) + p.encode("ascii", errors="ignore")
        return out + b"\x00"

    def _query_dns_server(self, server_ip: str, host: str, timeout: float = 2.0):
        """
        Minimal DNS A-record query over UDP to a specific server (e.g., 1.1.1.1).
        Returns a set of IPv4 strings.
        """
        ips = set()
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)

            tid = random.randint(0, 0xFFFF)
            flags = 0x0100  # standard query, recursion desired
            qdcount = 1
            ancount = 0
            nscount = 0
            arcount = 0
            header = struct.pack("!HHHHHH", tid, flags, qdcount, ancount, nscount, arcount)

            qname = self._encode_dns_name(host)
            qtype = 1   # A
            qclass = 1  # IN
            question = qname + struct.pack("!HH", qtype, qclass)

            packet = header + question
            sock.sendto(packet, (server_ip, 53))
            data, _ = sock.recvfrom(2048)

            if len(data) < 12:
                return ips

            # Parse header
            _tid, _flags, qdcount, ancount, _, _ = struct.unpack("!HHHHHH", data[:12])
            offset = 12

            # Skip questions
            for _ in range(qdcount):
                while True:
                    if offset >= len(data):
                        return ips
                    length = data[offset]
                    offset += 1
                    if length == 0:
                        break
                    offset += length
                offset += 4  # qtype + qclass

            # Parse answers
            for _ in range(ancount):
                if offset + 12 > len(data):
                    break

                # Name (could be pointer)
                first_byte = data[offset]
                if (first_byte & 0xC0) == 0xC0:
                    offset += 2
                else:
                    while True:
                        if offset >= len(data):
                            break
                        length = data[offset]
                        offset += 1
                        if length == 0:
                            break
                        offset += length

                if offset + 10 > len(data):
                    break

                rtype, rclass, ttl, rdlength = struct.unpack("!HHIH", data[offset:offset+10])
                offset += 10

                if offset + rdlength > len(data):
                    break

                rdata = data[offset:offset+rdlength]
                offset += rdlength

                if rtype == 1 and rclass == 1 and rdlength == 4:  # A IN
                    ip = ".".join(str(b) for b in rdata)
                    ips.add(ip)

        except Exception as e:
            self._log(f"[Resolver] External DNS query to {server_ip} failed for {host}: {e}")
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

        return ips

    def analyze(self, host: str, port: int):
        """
        Dual-resolver verification + IP log correlation.

        Returns dict:
        {
            "allow": bool,
            "reason": str,
            "system_ips": set,
            "external_ips": set,
        }
        """
        system_ips = self._system_resolve(host)
        external_ips = self._query_dns_server("1.1.1.1", host)

        self._log(f"[Resolver] System DNS IPs for {host}: {sorted(system_ips) or '[]'}")
        self._log(f"[Resolver] External DNS (1.1.1.1) IPs for {host}: {sorted(external_ips) or '[]'}")

        intersection = system_ips & external_ips
        previous_ips = self._registry.get_previous_ips_for_host(host)

        allow = True
        reason = "trusted"

        if system_ips and external_ips:
            if not intersection:
                allow = False
                reason = "dns_mismatch"
        elif not system_ips and not external_ips:
            allow = False
            reason = "dns_failure"
        else:
            reason = "partial_dns"

        # Historical IP correlation under lockdown
        if allow and self._policy.is_enabled() and not self._policy.is_manual_override():
            new_ips = (system_ips | external_ips) - previous_ips
            if previous_ips and new_ips:
                allow = False
                reason = "ip_change_under_lockdown"

        self._log(f"[Resolver] Decision for {host}:{port} → allow={allow}, reason={reason}")
        return {
            "allow": allow,
            "reason": reason,
            "system_ips": system_ips,
            "external_ips": external_ips,
        }


# ---------------------------------------------------------------------------
# Secure connection manager (TLS + fingerprint pinning) – QueenGuardian engine
# ---------------------------------------------------------------------------

class SecureConnectionManager(QObject):
    connection_state_changed = Signal(str, bool)  # message, is_secure
    log_message = Signal(str)

    def __init__(self, registry: ConnectionRegistry, policy: LockdownPolicy,
                 resolver: QueenResolver, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._policy = policy
        self._resolver = resolver
        self._task_queue = queue.Queue()
        self._workers = []
        self._spawn_worker()  # start at least one worker

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _set_state(self, msg: str, secure: bool):
        self.connection_state_changed.emit(msg, secure)

    def _spawn_worker(self):
        """
        Spawn a new worker thread that processes connection tasks.
        Unlimited workers vibe: we can spawn more if queue grows.
        """
        t = threading.Thread(target=self._worker_loop, daemon=True)
        self._workers.append(t)
        t.start()

    def _worker_loop(self):
        while True:
            task = self._task_queue.get()
            if task is None:
                break
            try:
                self._handle_task(**task)
            except Exception as e:
                self._log("Worker exception:\n" + "".join(traceback.format_exception(e)))
            finally:
                self._task_queue.task_done()

    def connect_secure(self, host: str, port: int, expected_fingerprint: str):
        """
        Public entry point: QueenGuardian uses this for all connections.
        Any subsystem that wants to talk out should call this.
        """
        if self._task_queue.qsize() > 5 and len(self._workers) < 32:
            self._spawn_worker()

        self._task_queue.put({
            "host": host,
            "port": port,
            "expected_fingerprint": expected_fingerprint,
        })

    def _handle_task(self, host: str, port: int, expected_fingerprint: str):
        expected_fingerprint = expected_fingerprint.replace(":", "").lower().strip()

        # QueenResolver: DNS + IP verification
        resolver_result = self._resolver.analyze(host, port)
        system_ips = resolver_result["system_ips"]
        external_ips = resolver_result["external_ips"]
        combined_ips = sorted(system_ips | external_ips)

        if not resolver_result["allow"]:
            msg = f"Resolver blocked {host}:{port} ({resolver_result['reason']})"
            self._log(f"[Guardian] {msg}")
            self._set_state(msg, False)
            self._registry.add_connection({
                "host": host,
                "port": port,
                "secure": False,
                "reason": resolver_result["reason"],
                "fingerprint": None,
                "ips": combined_ips,
            })
            return

        # Lockdown decision (QueenLockdown)
        if not self._policy.decide(host, port):
            msg = f"Lockdown blocked connection to {host}:{port}"
            self._log(f"[Guardian] {msg}")
            self._set_state(msg, False)
            self._registry.add_connection({
                "host": host,
                "port": port,
                "secure": False,
                "reason": "lockdown_block",
                "fingerprint": None,
                "ips": combined_ips,
            })
            return

        idx = self._registry.add_connection({
            "host": host,
            "port": port,
            "secure": False,
            "reason": "pending",
            "fingerprint": None,
            "ips": combined_ips,
        })

        self._set_state(f"Connecting to {host}:{port}...", False)
        self._log(f"[Guardian] Connecting to {host}:{port} with pinned fingerprint {expected_fingerprint or '(none)'}")

        try:
            context = ssl.create_default_context()
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED

            with socket.create_connection((host, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    der_cert = ssock.getpeercert(binary_form=True)
                    sha256 = hashlib.sha256(der_cert).hexdigest().lower()
                    self._log(f"[Guardian] Server cert SHA256: {sha256}")

                    if expected_fingerprint:
                        if sha256 != expected_fingerprint:
                            msg = "Fingerprint mismatch – NOT SECURE"
                            self._set_state(msg, False)
                            self._log("[Guardian] Pinned fingerprint does not match. Dropping connection.")
                            self._registry.update_connection(idx,
                                                             secure=False,
                                                             reason="fingerprint_mismatch",
                                                             fingerprint=sha256)
                            return
                        else:
                            self._log("[Guardian] Pinned fingerprint matches.")

                    msg = "Secure connection established"
                    self._set_state(msg, True)
                    self._log("[Guardian] TLS handshake complete and verified.")
                    self._registry.update_connection(idx,
                                                     secure=True,
                                                     reason="ok",
                                                     fingerprint=sha256)

        except Exception as e:
            msg = f"Connection failed: {e}"
            self._set_state(msg, False)
            self._log("[Guardian] Exception during connection:\n" + "".join(traceback.format_exception(e)))
            self._registry.update_connection(idx,
                                             secure=False,
                                             reason="exception",
                                             fingerprint=None)


# ---------------------------------------------------------------------------
# QueenScanner – background scanner + enforcement (always running)
# ---------------------------------------------------------------------------

class QueenScanner(QObject):
    log_message = Signal(str)
    threat_detected = Signal(str)

    def __init__(self, registry: ConnectionRegistry, policy: LockdownPolicy, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._policy = policy
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            snapshot = self._registry.snapshot()
            for conn in snapshot:
                # Autonomous enforcement logic:
                # - If lockdown is enabled and a non-secure connection exists → flag
                if self._policy.is_enabled() and not conn.get("secure", False):
                    msg = (f"[Scanner] Non-secure connection under lockdown: "
                           f"{conn.get('host')}:{conn.get('port')} ({conn.get('reason')})")
                    self._log(msg)
                    self.threat_detected.emit(msg)
            time.sleep(3.0)


# ---------------------------------------------------------------------------
# QueenLockdown – orchestrates lockdown policy
# ---------------------------------------------------------------------------

class QueenLockdown(QObject):
    lockdown_state_changed = Signal(bool)
    manual_override_changed = Signal(bool)
    log_message = Signal(str)

    def __init__(self, policy: LockdownPolicy, parent=None):
        super().__init__(parent)
        self._policy = policy

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def set_lockdown(self, enabled: bool):
        self._policy.set_enabled(enabled)
        self.lockdown_state_changed.emit(enabled)
        self._log(f"[Lockdown] Lockdown {'ENABLED' if enabled else 'DISABLED'}")

    def set_manual_override(self, enabled: bool):
        self._policy.set_manual_override(enabled)
        self.manual_override_changed.emit(enabled)
        self._log(f"[Lockdown] Manual override {'ENABLED' if enabled else 'DISABLED'}")


# ---------------------------------------------------------------------------
# PySide6 cockpit UI – AI Borg control panel
# ---------------------------------------------------------------------------

class SecureCockpit(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Borg Secure Connection Cockpit")
        self.setMinimumSize(900, 550)

        # Core shared components
        self.registry = ConnectionRegistry()
        self.policy = LockdownPolicy()

        # Queens
        self.resolver = QueenResolver(self.registry, self.policy)
        self.guardian = SecureConnectionManager(self.registry, self.policy, self.resolver)
        self.scanner = QueenScanner(self.registry, self.policy)
        self.lockdown = QueenLockdown(self.policy)

        # Wire signals
        self.guardian.connection_state_changed.connect(self.on_connection_state_changed)
        self.guardian.log_message.connect(self.append_log)
        self.scanner.log_message.connect(self.append_log)
        self.scanner.threat_detected.connect(self.on_threat_detected)
        self.lockdown.lockdown_state_changed.connect(self.on_lockdown_state_changed)
        self.lockdown.manual_override_changed.connect(self.on_manual_override_changed)
        self.lockdown.log_message.connect(self.append_log)
        self.resolver.log_message.connect(self.append_log)

        self._build_ui()

        # Periodic UI refresh for registry snapshot
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh_status_summary)
        self._refresh_timer.start(4000)

        # Autonomous default: scanner already running; you can choose whether
        # to start in lockdown or learning mode. Here we start unlocked.
        # self.lockdown.set_lockdown(True)

    # ---------------- UI construction ----------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # Top cockpit bar
        cockpit_frame = QFrame()
        cockpit_frame.setFrameShape(QFrame.StyledPanel)
        cockpit_frame.setObjectName("cockpitFrame")
        cockpit_layout = QHBoxLayout(cockpit_frame)
        cockpit_layout.setContentsMargins(10, 10, 10, 10)
        cockpit_layout.setSpacing(15)

        title_label = QLabel("AI BORG – SECURE LINK CONTROL")
        title_label.setObjectName("titleLabel")
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.secure_indicator = QLabel("NOT SECURE")
        self.secure_indicator.setAlignment(Qt.AlignCenter)
        self.secure_indicator.setFixedWidth(140)
        self.secure_indicator.setObjectName("secureIndicator")

        self.lockdown_indicator = QLabel("LOCKDOWN: OFF")
        self.lockdown_indicator.setAlignment(Qt.AlignCenter)
        self.lockdown_indicator.setFixedWidth(160)
        self.lockdown_indicator.setObjectName("lockdownIndicator")

        self.override_indicator = QLabel("OVERRIDE: OFF")
        self.override_indicator.setAlignment(Qt.AlignCenter)
        self.override_indicator.setFixedWidth(160)
        self.override_indicator.setObjectName("overrideIndicator")

        cockpit_layout.addWidget(title_label)
        cockpit_layout.addWidget(self.secure_indicator)
        cockpit_layout.addWidget(self.lockdown_indicator)
        cockpit_layout.addWidget(self.override_indicator)

        # Connection panel
        conn_frame = QFrame()
        conn_frame.setFrameShape(QFrame.StyledPanel)
        conn_layout = QGridLayout(conn_frame)
        conn_layout.setContentsMargins(10, 10, 10, 10)
        conn_layout.setHorizontalSpacing(10)
        conn_layout.setVerticalSpacing(8)

        host_label = QLabel("Host:")
        port_label = QLabel("Port:")
        fp_label = QLabel("Pinned SHA256 fingerprint (optional):")

        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("example.com")
        self.port_edit = QLineEdit()
        self.port_edit.setPlaceholderText("443")
        self.port_edit.setText("443")

        self.fp_edit = QLineEdit()
        self.fp_edit.setPlaceholderText("e.g. 9f2c... (no colons)")

        self.connect_button = QPushButton("Connect via Borg")
        self.connect_button.clicked.connect(self.on_connect_clicked)

        self.lockdown_checkbox = QCheckBox("Enable Lockdown Mode (Queen C)")
        self.lockdown_checkbox.stateChanged.connect(self.on_lockdown_checkbox_changed)

        self.override_checkbox = QCheckBox("Manual Override (allow under lockdown)")
        self.override_checkbox.stateChanged.connect(self.on_override_checkbox_changed)

        conn_layout.addWidget(host_label, 0, 0)
        conn_layout.addWidget(self.host_edit, 0, 1)
        conn_layout.addWidget(port_label, 0, 2)
        conn_layout.addWidget(self.port_edit, 0, 3)

        conn_layout.addWidget(fp_label, 1, 0, 1, 1)
        conn_layout.addWidget(self.fp_edit, 1, 1, 1, 3)

        conn_layout.addWidget(self.connect_button, 2, 0, 1, 4)
        conn_layout.addWidget(self.lockdown_checkbox, 3, 0, 1, 4)
        conn_layout.addWidget(self.override_checkbox, 4, 0, 1, 4)

        # Log panel
        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(10, 10, 10, 10)

        log_label = QLabel("Event Log:")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.status_summary = QLabel("Connections: 0 | Secure: 0 | Insecure: 0")
        self.status_summary.setObjectName("statusSummary")

        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_view)
        log_layout.addWidget(self.status_summary)

        # Assemble root
        root_layout.addWidget(cockpit_frame)
        root_layout.addWidget(conn_frame)
        root_layout.addWidget(log_frame)

        self._apply_styles()
        self._set_secure_indicator(False, "NOT SECURE")
        self._set_lockdown_indicator(False)
        self._set_override_indicator(False)

    def _apply_styles(self):
        self.setStyleSheet("""
        #cockpitFrame {
            background-color: #111827;
            border-radius: 6px;
        }
        #titleLabel {
            color: #E5E7EB;
            font-size: 18px;
            font-weight: 600;
        }
        #secureIndicator, #lockdownIndicator, #overrideIndicator {
            border-radius: 14px;
            padding: 6px 10px;
            color: #FFFFFF;
            font-weight: 600;
        }
        #statusSummary {
            color: #9CA3AF;
            font-size: 11px;
        }
        QFrame {
            background-color: #020617;
            border: 1px solid #1F2937;
            border-radius: 6px;
        }
        QLabel {
            color: #E5E7EB;
        }
        QLineEdit {
            background-color: #020617;
            color: #E5E7EB;
            border: 1px solid #374151;
            border-radius: 4px;
            padding: 4px 6px;
        }
        QTextEdit {
            background-color: #020617;
            color: #E5E7EB;
            border: 1px solid #374151;
            border-radius: 4px;
        }
        QPushButton {
            background-color: #2563EB;
            color: #F9FAFB;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #1D4ED8;
        }
        QPushButton:pressed {
            background-color: #1E40AF;
        }
        QCheckBox {
            color: #E5E7EB;
        }
        """)

    # ---------------- Slots / logic ----------------

    @Slot()
    def on_connect_clicked(self):
        host = self.host_edit.text().strip()
        port_text = self.port_edit.text().strip()
        fp = self.fp_edit.text().strip()

        if not host:
            self.append_log("Host is required.")
            return

        try:
            port = int(port_text)
        except ValueError:
            self.append_log("Port must be an integer.")
            return

        self.append_log(f"[Operator] Requesting secure connection to {host}:{port} via AI Borg...")
        self.guardian.connect_secure(host, port, fp)

    @Slot(str, bool)
    def on_connection_state_changed(self, message: str, is_secure: bool):
        self._set_secure_indicator(is_secure, "SECURE" if is_secure else "NOT SECURE")
        self.append_log(message)

    @Slot(str)
    def on_threat_detected(self, msg: str):
        # Autonomous reaction: if not manually overridden, escalate to lockdown
        self.append_log(msg)
        if not self.policy.is_manual_override() and not self.policy.is_enabled():
            self.append_log("[Scanner] Threat detected – auto-enabling lockdown.")
            self.lockdown.set_lockdown(True)
            self.lockdown_checkbox.blockSignals(True)
            self.lockdown_checkbox.setChecked(True)
            self.lockdown_checkbox.blockSignals(False)
        self._set_secure_indicator(False, "THREAT")

    @Slot(int)
    def on_lockdown_checkbox_changed(self, state: int):
        enabled = (state == Qt.Checked)
        self.lockdown.set_lockdown(enabled)

    @Slot(int)
    def on_override_checkbox_changed(self, state: int):
        enabled = (state == Qt.Checked)
        self.lockdown.set_manual_override(enabled)

    @Slot(bool)
    def on_lockdown_state_changed(self, enabled: bool):
        self._set_lockdown_indicator(enabled)

    @Slot(bool)
    def on_manual_override_changed(self, enabled: bool):
        self._set_override_indicator(enabled)

    def _set_secure_indicator(self, secure: bool, text: str):
        self.secure_indicator.setText(text)
        if secure:
            self.secure_indicator.setStyleSheet("""
                border-radius: 14px;
                padding: 6px 10px;
                color: #FFFFFF;
                font-weight: 600;
                background-color: #16A34A; /* green */
            """)
        else:
            self.secure_indicator.setStyleSheet("""
                border-radius: 14px;
                padding: 6px 10px;
                color: #FFFFFF;
                font-weight: 600;
                background-color: #B91C1C; /* red */
            """)

    def _set_lockdown_indicator(self, enabled: bool):
        self.lockdown_indicator.setText(f"LOCKDOWN: {'ON' if enabled else 'OFF'}")
        self.lockdown_indicator.setStyleSheet(f"""
            border-radius: 14px;
            padding: 6px 10px;
            color: #FFFFFF;
            font-weight: 600;
            background-color: {'#7C3AED' if enabled else '#4B5563'};
        """)

    def _set_override_indicator(self, enabled: bool):
        self.override_indicator.setText(f"OVERRIDE: {'ON' if enabled else 'OFF'}")
        self.override_indicator.setStyleSheet(f"""
            border-radius: 14px;
            padding: 6px 10px;
            color: #FFFFFF;
            font-weight: 600;
            background-color: {'#F59E0B' if enabled else '#4B5563'};
        """)

    @Slot(str)
    def append_log(self, msg: str):
        self.log_view.append(msg)

    @Slot()
    def refresh_status_summary(self):
        snapshot = self.registry.snapshot()
        total = len(snapshot)
        secure = sum(1 for c in snapshot if c.get("secure"))
        insecure = total - secure
        self.status_summary.setText(f"Connections: {total} | Secure: {secure} | Insecure: {insecure}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = SecureCockpit()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
