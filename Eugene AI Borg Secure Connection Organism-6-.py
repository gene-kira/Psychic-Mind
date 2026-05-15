#!/usr/bin/env python3
"""
Borg Brain v3 – Sentinel Organism

Features:
- Wireshark-style sniffer (Scapy)
- TLS Guardian with fingerprint pinning
- Dual-resolver DNS verification
- Lockdown policy engine
- ML anomaly detection (IsolationForest if available, heuristic fallback)
- Threat scoring with hybrid blocking:
  - High threat → OS firewall + Borg block
  - Medium threat → Borg-only block
- GeoIP/ASN-style intelligence (lightweight)
- Threat reputation feed stub
- Process lineage tracking
- ETW-style correlation hook (via bridge/driver)
- Swarm mesh (UDP gossip) for distributed threat intel
- Persistent allow/block lists (JSON)
- Common system IPs, Block list, Pending list
- Admin cockpit with:
  - Single panel
  - Collapsible sections: Allowed / Blocked / Pending
  - Timeline panel
  - Threat heatmap panel
- Bridge v2 JSON protocol for kernel/driver integration
"""

import sys
import os
import json
import time
import random
import struct
import socket
import ssl
import hashlib
import threading
import queue
import platform
import subprocess
import traceback
from datetime import datetime
from collections import defaultdict

# Optional dependencies
try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional ML
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    QScrollArea,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QTabWidget,
)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

BORG_BRIDGE_HOST = "127.0.0.1"
BORG_BRIDGE_PORT = 55555

SWARM_UDP_PORT = 55556
SWARM_BROADCAST_INTERVAL = 10.0  # seconds

LISTS_FILE = "borg_lists.json"
REPUTATION_FILE = "borg_reputation.json"

# Static common system IPs (seed)
COMMON_SYSTEM_IPS_SEED = {
    "8.8.8.8", "8.8.4.4",
    "1.1.1.1", "1.0.0.1",
    "9.9.9.9",
    "208.67.222.222", "208.67.220.220",
    "13.107.4.50", "13.107.5.88",
    "129.6.15.28", "132.163.96.1",
}

# Example reputation feed seed (IP → score)
REPUTATION_SEED = {
    "45.155.205.10": 8.5,
    "185.234.217.42": 9.1,
    "185.220.100.5": 9.5,
    "104.244.72.33": 8.0,
}

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def now_ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------------
# List manager – persistent allow/block + runtime pending
# ---------------------------------------------------------------------------

class ListManager:
    def __init__(self, logger):
        self._log = logger
        self.allowed_ips = set(COMMON_SYSTEM_IPS_SEED)
        self.blocked_ips = set()
        self.pending_ips = set()
        self._load()

    def _load(self):
        if not os.path.exists(LISTS_FILE):
            self._log(f"[{now_ts()}] [Lists] No existing {LISTS_FILE}, starting fresh.")
            return
        try:
            with open(LISTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.allowed_ips |= set(data.get("allowed_ips", []))
            self.blocked_ips |= set(data.get("blocked_ips", []))
            self._log(f"[{now_ts()}] [Lists] Loaded {len(self.allowed_ips)} allowed, {len(self.blocked_ips)} blocked from disk.")
        except Exception as e:
            self._log(f"[{now_ts()}] [Lists] Failed to load {LISTS_FILE}: {e}")

    def _save(self):
        try:
            data = {
                "allowed_ips": sorted(self.allowed_ips),
                "blocked_ips": sorted(self.blocked_ips),
            }
            with open(LISTS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._log(f"[{now_ts()}] [Lists] Saved allow/block lists to {LISTS_FILE}.")
        except Exception as e:
            self._log(f"[{now_ts()}] [Lists] Failed to save {LISTS_FILE}: {e}")

    def is_allowed(self, ip: str) -> bool:
        return ip in self.allowed_ips

    def is_blocked(self, ip: str) -> bool:
        return ip in self.blocked_ips

    def add_allowed(self, ip: str, persist=True):
        if ip and ip not in self.allowed_ips:
            self.allowed_ips.add(ip)
            self._log(f"[{now_ts()}] [Lists] Added {ip} to allowed list.")
            if ip in self.pending_ips:
                self.pending_ips.discard(ip)
            if persist:
                self._save()

    def add_blocked(self, ip: str, persist=True):
        if ip and ip not in self.blocked_ips:
            self.blocked_ips.add(ip)
            self._log(f"[{now_ts()}] [Lists] Added {ip} to blocked list.")
            if ip in self.pending_ips:
                self.pending_ips.discard(ip)
            if persist:
                self._save()

    def add_pending(self, ip: str):
        if ip and ip not in self.allowed_ips and ip not in self.blocked_ips:
            if ip not in self.pending_ips:
                self.pending_ips.add(ip)
                self._log(f"[{now_ts()}] [Lists] Added {ip} to pending list.")

    def remove_pending(self, ip: str):
        if ip in self.pending_ips:
            self.pending_ips.discard(ip)
            self._log(f"[{now_ts()}] [Lists] Removed {ip} from pending list.")

# ---------------------------------------------------------------------------
# Reputation manager
# ---------------------------------------------------------------------------

class ReputationManager:
    def __init__(self, logger):
        self._log = logger
        self.reputation = dict(REPUTATION_SEED)
        self._load()

    def _load(self):
        if not os.path.exists(REPUTATION_FILE):
            self._log(f"[{now_ts()}] [Reputation] No existing {REPUTATION_FILE}, using seed.")
            return
        try:
            with open(REPUTATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.reputation.update(data)
            self._log(f"[{now_ts()}] [Reputation] Loaded {len(self.reputation)} entries from disk.")
        except Exception as e:
            self._log(f"[{now_ts()}] [Reputation] Failed to load {REPUTATION_FILE}: {e}")

    def _save(self):
        try:
            with open(REPUTATION_FILE, "w", encoding="utf-8") as f:
                json.dump(self.reputation, f, indent=2)
            self._log(f"[{now_ts()}] [Reputation] Saved reputation DB to {REPUTATION_FILE}.")
        except Exception as e:
            self._log(f"[{now_ts()}] [Reputation] Failed to save {REPUTATION_FILE}: {e}")

    def get_score(self, ip: str) -> float:
        return float(self.reputation.get(ip, 0.0))

    def update_from_swarm(self, ip: str, score: float):
        old = self.reputation.get(ip, 0.0)
        if score > old:
            self.reputation[ip] = score
            self._log(f"[{now_ts()}] [Reputation] Updated {ip} score from swarm: {old:.1f} → {score:.1f}")
            self._save()

# ---------------------------------------------------------------------------
# GeoIP / ASN (lightweight heuristic)
# ---------------------------------------------------------------------------

def geoip_asn_lookup(ip: str):
    # No external DB: classify by RFC ranges and simple heuristics
    try:
        parts = [int(x) for x in ip.split(".")]
        if len(parts) != 4:
            return {"country": "UNK", "asn": "AS-UNK", "scope": "unknown"}
        if parts[0] == 10 or (parts[0] == 192 and parts[1] == 168) or (parts[0] == 172 and 16 <= parts[1] <= 31):
            return {"country": "LAN", "asn": "AS-LAN", "scope": "private"}
        if parts[0] == 127:
            return {"country": "LOOP", "asn": "AS-LOOP", "scope": "loopback"}
        if parts[0] in (8, 9, 13):
            return {"country": "US", "asn": "AS-GLOBAL", "scope": "public"}
        if parts[0] in (1, 4, 5):
            return {"country": "GLOBAL", "asn": "AS-CDN", "scope": "public"}
        return {"country": "GLOBAL", "asn": "AS-UNKNOWN", "scope": "public"}
    except Exception:
        return {"country": "UNK", "asn": "AS-UNK", "scope": "unknown"}

# ---------------------------------------------------------------------------
# Connection registry
# ---------------------------------------------------------------------------

class ConnectionRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._connections = []

    def add_connection(self, info: dict) -> int:
        with self._lock:
            info.setdefault("tags", [])
            info.setdefault("anomaly_score", 0.0)
            info.setdefault("threat_score", 0.0)
            info.setdefault("process_name", None)
            info.setdefault("pid", None)
            info.setdefault("lineage", None)
            info.setdefault("geo", None)
            info.setdefault("reputation", 0.0)
            info.setdefault("timeline_label", None)
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

    def add_tag(self, idx: int, tag: str):
        with self._lock:
            if 0 <= idx < len(self._connections):
                tags = self._connections[idx].setdefault("tags", [])
                if tag not in tags:
                    tags.append(tag)

    def set_anomaly_score(self, idx: int, score: float):
        with self._lock:
            if 0 <= idx < len(self._connections):
                self._connections[idx]["anomaly_score"] = score

    def set_threat_score(self, idx: int, score: float):
        with self._lock:
            if 0 <= idx < len(self._connections):
                self._connections[idx]["threat_score"] = score

# ---------------------------------------------------------------------------
# Lockdown policy
# ---------------------------------------------------------------------------

class LockdownPolicy:
    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = False
        self._manual_override = False
        self.allowed_ports = {443, 80, 53}

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
        with self._lock:
            if not self._enabled:
                return True
            if self._manual_override:
                return True
            return port in self.allowed_ports

# ---------------------------------------------------------------------------
# Firewall blocker (hybrid)
# ---------------------------------------------------------------------------

class FirewallBlocker:
    def __init__(self, logger, list_manager: ListManager):
        self._log = logger
        self._is_windows = platform.system().lower().startswith("win")
        self.list_manager = list_manager

    def _log_msg(self, msg: str):
        self._log(msg)

    def block_ip_borg_only(self, ip: str, reason: str):
        if not ip:
            return
        if self.list_manager.is_blocked(ip):
            return
        self.list_manager.add_blocked(ip, persist=True)
        self._log_msg(f"[{now_ts()}] [Firewall] Borg-only block for {ip} (reason={reason})")

    def block_ip_firewall(self, ip: str, reason: str):
        if not ip:
            return
        if self.list_manager.is_blocked(ip):
            return
        self.list_manager.add_blocked(ip, persist=True)
        self._log_msg(f"[{now_ts()}] [Firewall] OS firewall block for {ip} (reason={reason})")
        if self._is_windows:
            try:
                rule_name = f"BorgBlock_{ip}"
                cmd = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_name}",
                    "dir=out",
                    "action=block",
                    f"remoteip={ip}",
                ]
                subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            except Exception as e:
                self._log_msg(f"[{now_ts()}] [Firewall] Failed to add firewall rule for {ip}: {e}")
        else:
            self._log_msg(f"[{now_ts()}] [Firewall] Non-Windows OS – logging only, no firewall rule applied.")

# ---------------------------------------------------------------------------
# Resolver
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
            self._log(f"[{now_ts()}] [Resolver] System DNS resolve failed for {host}: {e}")
        return ips

    def _encode_dns_name(self, host: str) -> bytes:
        parts = host.strip(".").split(".")
        out = b""
        for p in parts:
            out += struct.pack("!B", len(p)) + p.encode("ascii", errors="ignore")
        return out + b"\x00"

    def _query_dns_server(self, server_ip: str, host: str, timeout: float = 2.0):
        ips = set()
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            tid = random.randint(0, 0xFFFF)
            flags = 0x0100
            qdcount = 1
            header = struct.pack("!HHHHHH", tid, flags, qdcount, 0, 0, 0)
            qname = self._encode_dns_name(host)
            qtype = 1
            qclass = 1
            question = qname + struct.pack("!HH", qtype, qclass)
            packet = header + question
            sock.sendto(packet, (server_ip, 53))
            data, _ = sock.recvfrom(2048)
            if len(data) < 12:
                return ips
            _tid, _flags, qdcount, ancount, _, _ = struct.unpack("!HHHHHH", data[:12])
            offset = 12
            for _ in range(qdcount):
                while True:
                    if offset >= len(data):
                        return ips
                    length = data[offset]
                    offset += 1
                    if length == 0:
                        break
                    offset += length
                offset += 4
            for _ in range(ancount):
                if offset + 12 > len(data):
                    break
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
                if rtype == 1 and rclass == 1 and rdlength == 4:
                    ip = ".".join(str(b) for b in rdata)
                    ips.add(ip)
        except Exception as e:
            self._log(f"[{now_ts()}] [Resolver] External DNS query to {server_ip} failed for {host}: {e}")
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        return ips

    def analyze(self, host: str, port: int):
        system_ips = self._system_resolve(host)
        external_ips = self._query_dns_server("1.1.1.1", host)
        self._log(f"[{now_ts()}] [Resolver] System DNS IPs for {host}: {sorted(system_ips) or '[]'}")
        self._log(f"[{now_ts()}] [Resolver] External DNS IPs for {host}: {sorted(external_ips) or '[]'}")
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
        if allow and self._policy.is_enabled() and not self._policy.is_manual_override():
            new_ips = (system_ips | external_ips) - previous_ips
            if previous_ips and new_ips:
                allow = False
                reason = "ip_change_under_lockdown"
        self._log(f"[{now_ts()}] [Resolver] Decision for {host}:{port} → allow={allow}, reason={reason}")
        return {
            "allow": allow,
            "reason": reason,
            "system_ips": system_ips,
            "external_ips": external_ips,
        }

# ---------------------------------------------------------------------------
# Process attribution + lineage
# ---------------------------------------------------------------------------

def find_process_for_connection(remote_ip, remote_port):
    if not PSUTIL_AVAILABLE:
        return None, None
    try:
        for proc in psutil.process_iter(["pid", "name", "connections"]):
            pid = proc.info["pid"]
            name = proc.info["name"]
            for c in proc.connections(kind="inet"):
                raddr = c.raddr
                if not raddr:
                    continue
                if raddr.ip == remote_ip and raddr.port == remote_port:
                    return name, pid
    except Exception:
        pass
    return None, None

def get_process_lineage(pid):
    if not PSUTIL_AVAILABLE or pid is None:
        return None
    try:
        chain = []
        p = psutil.Process(pid)
        while p:
            chain.append(f"{p.name()}({p.pid})")
            p = p.parent()
        return " -> ".join(chain)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Guardian
# ---------------------------------------------------------------------------

class QueenGuardian(QObject):
    connection_state_changed = Signal(str, bool)
    log_message = Signal(str)
    timeline_event = Signal(str)

    def __init__(self, registry: ConnectionRegistry, policy: LockdownPolicy,
                 resolver: QueenResolver, firewall: FirewallBlocker,
                 lists: ListManager, reputation: ReputationManager, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._policy = policy
        self._resolver = resolver
        self._firewall = firewall
        self._lists = lists
        self._reputation = reputation
        self._task_queue = queue.Queue()
        self._workers = []
        self._spawn_worker()

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _timeline(self, msg: str):
        self.timeline_event.emit(msg)

    def _set_state(self, msg: str, secure: bool):
        self.connection_state_changed.emit(msg, secure)

    def _spawn_worker(self):
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
        if self._task_queue.qsize() > 5 and len(self._workers) < 32:
            self._spawn_worker()
        self._task_queue.put({
            "host": host,
            "port": port,
            "expected_fingerprint": expected_fingerprint,
        })

    def _handle_task(self, host: str, port: int, expected_fingerprint: str):
        expected_fingerprint = expected_fingerprint.replace(":", "").lower().strip()
        resolver_result = self._resolver.analyze(host, port)
        system_ips = resolver_result["system_ips"]
        external_ips = resolver_result["external_ips"]
        combined_ips = sorted(system_ips | external_ips)

        # Geo + reputation
        geo_info = geoip_asn_lookup(combined_ips[0]) if combined_ips else None
        rep_score = max(self._reputation.get_score(ip) for ip in combined_ips) if combined_ips else 0.0

        for ip in combined_ips:
            if self._lists.is_blocked(ip):
                msg = f"Blocked by list: {ip}"
                self._log(f"[{now_ts()}] [Guardian] {msg}")
                self._set_state(msg, False)
                idx = self._registry.add_connection({
                    "kind": "active",
                    "host": host,
                    "port": port,
                    "secure": False,
                    "reason": "list_block",
                    "fingerprint": None,
                    "ips": combined_ips,
                    "geo": geo_info,
                    "reputation": rep_score,
                    "timeline_label": f"LIST BLOCK {host}:{port}",
                })
                self._timeline(f"[{now_ts()}] LIST BLOCK {host}:{port} ({ip})")
                return

        if not resolver_result["allow"]:
            msg = f"Resolver blocked {host}:{port} ({resolver_result['reason']})"
            self._log(f"[{now_ts()}] [Guardian] {msg}")
            self._set_state(msg, False)
            idx = self._registry.add_connection({
                "kind": "active",
                "host": host,
                "port": port,
                "secure": False,
                "reason": resolver_result["reason"],
                "fingerprint": None,
                "ips": combined_ips,
                "geo": geo_info,
                "reputation": rep_score,
                "timeline_label": f"DNS BLOCK {host}:{port}",
            })
            for ip in combined_ips:
                self._lists.add_pending(ip)
            self._timeline(f"[{now_ts()}] DNS BLOCK {host}:{port}")
            return

        if not self._policy.decide(host, port):
            msg = f"Lockdown blocked connection to {host}:{port}"
            self._log(f"[{now_ts()}] [Guardian] {msg}")
            self._set_state(msg, False)
            idx = self._registry.add_connection({
                "kind": "active",
                "host": host,
                "port": port,
                "secure": False,
                "reason": "lockdown_block",
                "fingerprint": None,
                "ips": combined_ips,
                "geo": geo_info,
                "reputation": rep_score,
                "timeline_label": f"LOCKDOWN BLOCK {host}:{port}",
            })
            for ip in combined_ips:
                self._lists.add_pending(ip)
            self._timeline(f"[{now_ts()}] LOCKDOWN BLOCK {host}:{port}")
            return

        idx = self._registry.add_connection({
            "kind": "active",
            "host": host,
            "port": port,
            "secure": False,
            "reason": "pending",
            "fingerprint": None,
            "ips": combined_ips,
            "geo": geo_info,
            "reputation": rep_score,
            "timeline_label": f"CONNECT {host}:{port}",
        })
        self._set_state(f"Connecting to {host}:{port}...", False)
        self._log(f"[{now_ts()}] [Guardian] Connecting to {host}:{port} with pinned fingerprint {expected_fingerprint or '(none)'}")
        self._timeline(f"[{now_ts()}] CONNECT {host}:{port}")

        try:
            context = ssl.create_default_context()
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            with socket.create_connection((host, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    der_cert = ssock.getpeercert(binary_form=True)
                    sha256 = hashlib.sha256(der_cert).hexdigest().lower()
                    self._log(f"[{now_ts()}] [Guardian] Server cert SHA256: {sha256}")
                    if expected_fingerprint:
                        if sha256 != expected_fingerprint:
                            msg = "Fingerprint mismatch – NOT SECURE"
                            self._set_state(msg, False)
                            self._log(f"[{now_ts()}] [Guardian] Pinned fingerprint mismatch.")
                            self._registry.update_connection(idx,
                                                             secure=False,
                                                             reason="fingerprint_mismatch",
                                                             fingerprint=sha256)
                            for ip in combined_ips:
                                self._lists.add_pending(ip)
                            self._timeline(f"[{now_ts()}] FINGERPRINT MISMATCH {host}:{port}")
                            return
                        else:
                            self._log(f"[{now_ts()}] [Guardian] Pinned fingerprint matches.")
                    msg = "Secure connection established"
                    self._set_state(msg, True)
                    self._log(f"[{now_ts()}] [Guardian] TLS handshake complete and verified.")
                    self._registry.update_connection(idx,
                                                     secure=True,
                                                     reason="ok",
                                                     fingerprint=sha256)
                    self._timeline(f"[{now_ts()}] SECURE {host}:{port}")
        except Exception as e:
            msg = f"Connection failed: {e}"
            self._set_state(msg, False)
            self._log(f"[{now_ts()}] [Guardian] Exception during connection:\n" + "".join(traceback.format_exception(e)))
            self._registry.update_connection(idx,
                                             secure=False,
                                             reason="exception",
                                             fingerprint=None)
            self._timeline(f"[{now_ts()}] CONNECT FAIL {host}:{port}")

# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class QueenScanner(QObject):
    log_message = Signal(str)
    threat_detected = Signal(str)
    timeline_event = Signal(str)

    def __init__(self, registry: ConnectionRegistry, policy: LockdownPolicy,
                 firewall: FirewallBlocker, lists: ListManager, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._policy = policy
        self._firewall = firewall
        self._lists = lists
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _timeline(self, msg: str):
        self.timeline_event.emit(msg)

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            snapshot = self._registry.snapshot()
            for idx, conn in enumerate(snapshot):
                if conn.get("kind") == "active":
                    if self._policy.is_enabled() and not conn.get("secure", False):
                        msg = (f"[{now_ts()}] [Scanner] Non-secure active connection under lockdown: "
                               f"{conn.get('host')}:{conn.get('port')} ({conn.get('reason')})")
                        self._log(msg)
                        self.threat_detected.emit(msg)
                        self._timeline(f"[{now_ts()}] LOCKDOWN NON-SECURE {conn.get('host')}:{conn.get('port')}")
                        for ip in conn.get("ips", []):
                            self._lists.add_pending(ip)
            time.sleep(3.0)

# ---------------------------------------------------------------------------
# Lockdown
# ---------------------------------------------------------------------------

class QueenLockdown(QObject):
    lockdown_state_changed = Signal(bool)
    manual_override_changed = Signal(bool)
    log_message = Signal(str)
    timeline_event = Signal(str)

    def __init__(self, policy: LockdownPolicy, parent=None):
        super().__init__(parent)
        self._policy = policy

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _timeline(self, msg: str):
        self.timeline_event.emit(msg)

    def set_lockdown(self, enabled: bool):
        self._policy.set_enabled(enabled)
        self.lockdown_state_changed.emit(enabled)
        self._log(f"[{now_ts()}] [Lockdown] Lockdown {'ENABLED' if enabled else 'DISABLED'}")
        self._timeline(f"[{now_ts()}] LOCKDOWN {'ON' if enabled else 'OFF'}")

    def set_manual_override(self, enabled: bool):
        self._policy.set_manual_override(enabled)
        self.manual_override_changed.emit(enabled)
        self._log(f"[{now_ts()}] [Lockdown] Manual override {'ENABLED' if enabled else 'DISABLED'}")
        self._timeline(f"[{now_ts()}] OVERRIDE {'ON' if enabled else 'OFF'}")

# ---------------------------------------------------------------------------
# Anomaly (ML + heuristic)
# ---------------------------------------------------------------------------

class QueenAnomaly(QObject):
    log_message = Signal(str)
    timeline_event = Signal(str)

    def __init__(self, registry: ConnectionRegistry, lists: ListManager,
                 reputation: ReputationManager, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._lists = lists
        self._reputation = reputation
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _timeline(self, msg: str):
        self.timeline_event.emit(msg)

    def stop(self):
        self._stop_event.set()

    def _heuristic_score(self, conn: dict) -> float:
        score = 0.0
        if conn.get("kind") == "active":
            if conn.get("secure") is False:
                score += 3.0
            if conn.get("reason") in ("dns_mismatch", "ip_change_under_lockdown", "fingerprint_mismatch"):
                score += 4.0
        if conn.get("kind") == "sniff":
            port = conn.get("port") or 0
            if port not in (80, 443, 53):
                score += 1.0
            if port == 0:
                score += 0.5
        host = conn.get("host")
        if host and self._lists.is_blocked(host):
            score += 5.0
        if host and host in COMMON_SYSTEM_IPS_SEED:
            score -= 1.0
        rep = conn.get("reputation", 0.0)
        score += rep / 3.0
        return max(score, 0.0)

    def _ml_scores(self, snapshot):
        if not SKLEARN_AVAILABLE or len(snapshot) < 10:
            return None
        features = []
        for c in snapshot:
            port = c.get("port") or 0
            secure = 1 if c.get("secure") else 0
            rep = c.get("reputation", 0.0)
            kind = {"active": 1, "sniff": 2, "bridge": 3}.get(c.get("kind"), 0)
            features.append([port, secure, rep, kind])
        try:
            model = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
            model.fit(features)
            scores = model.decision_function(features)
            return scores
        except Exception:
            return None

    def _loop(self):
        while not self._stop_event.is_set():
            snapshot = self._registry.snapshot()
            ml_scores = self._ml_scores(snapshot)
            for idx, conn in enumerate(snapshot):
                base = self._heuristic_score(conn)
                if ml_scores is not None:
                    ml_component = -ml_scores[idx] * 5.0
                    score = max(base + ml_component, 0.0)
                else:
                    score = base
                self._registry.set_anomaly_score(idx, score)
                if score >= 4.0:
                    self._registry.add_tag(idx, "high_anomaly")
                    self._log(f"[{now_ts()}] [Anomaly] High anomaly score {score:.1f} for {conn.get('kind')} "
                              f"{conn.get('host')}:{conn.get('port')} (reason={conn.get('reason')})")
                    self._timeline(f"[{now_ts()}] HIGH ANOMALY {conn.get('host')}:{conn.get('port')} ({score:.1f})")
                    host = conn.get("host")
                    if host and not self._lists.is_allowed(host) and not self._lists.is_blocked(host):
                        self._lists.add_pending(host)
            time.sleep(5.0)

# ---------------------------------------------------------------------------
# Threat
# ---------------------------------------------------------------------------

class QueenThreat(QObject):
    log_message = Signal(str)
    timeline_event = Signal(str)

    def __init__(self, registry: ConnectionRegistry, firewall: FirewallBlocker,
                 lists: ListManager, reputation: ReputationManager, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._firewall = firewall
        self._lists = lists
        self._reputation = reputation
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _timeline(self, msg: str):
        self.timeline_event.emit(msg)

    def stop(self):
        self._stop_event.set()

    def _compute_threat_score(self, conn: dict) -> float:
        base = conn.get("anomaly_score", 0.0)
        if "high_anomaly" in conn.get("tags", []):
            base += 1.0
        if conn.get("reason") in ("dns_mismatch", "fingerprint_mismatch", "ip_change_under_lockdown"):
            base += 1.0
        host = conn.get("host")
        if host and self._lists.is_blocked(host):
            base += 3.0
        rep = conn.get("reputation", 0.0)
        base += rep / 2.0
        return base

    def _loop(self):
        while not self._stop_event.is_set():
            snapshot = self._registry.snapshot()
            for idx, conn in enumerate(snapshot):
                score = self._compute_threat_score(conn)
                self._registry.set_threat_score(idx, score)
                host = conn.get("host")
                if not host:
                    continue
                if score >= 8.0:
                    if not self._lists.is_blocked(host):
                        self._log(f"[{now_ts()}] [Threat] High threat {score:.1f} for {host}:{conn.get('port')} – OS firewall block.")
                        self._firewall.block_ip_firewall(host, reason="high_threat")
                        self._timeline(f"[{now_ts()}] HIGH THREAT BLOCK {host}:{conn.get('port')} ({score:.1f})")
                elif score >= 5.0:
                    if not self._lists.is_blocked(host):
                        self._log(f"[{now_ts()}] [Threat] Medium threat {score:.1f} for {host}:{conn.get('port')} – Borg-only block.")
                        self._firewall.block_ip_borg_only(host, reason="medium_threat")
                        self._timeline(f"[{now_ts()}] MEDIUM THREAT BLOCK {host}:{conn.get('port')} ({score:.1f})")
            time.sleep(5.0)

# ---------------------------------------------------------------------------
# Swarm mesh (UDP gossip)
# ---------------------------------------------------------------------------

class QueenSwarm(QObject):
    log_message = Signal(str)

    def __init__(self, registry: ConnectionRegistry, reputation: ReputationManager, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._reputation = reputation
        self._stop_event = threading.Event()
        self._peers = set()  # you can manually add peers if desired
        self._thread_rx = threading.Thread(target=self._loop_rx, daemon=True)
        self._thread_tx = threading.Thread(target=self._loop_tx, daemon=True)
        self._thread_rx.start()
        self._thread_tx.start()
        self._log(f"[{now_ts()}] [Swarm] Swarm subsystem initialized (UDP gossip on port {SWARM_UDP_PORT}).")

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def stop(self):
        self._stop_event.set()

    def add_peer(self, ip: str):
        self._peers.add(ip)

    def _loop_rx(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("0.0.0.0", SWARM_UDP_PORT))
        except Exception as e:
            self._log(f"[{now_ts()}] [Swarm] Failed to bind UDP port {SWARM_UDP_PORT}: {e}")
            return
        sock.settimeout(1.0)
        while not self._stop_event.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception as e:
                self._log(f"[{now_ts()}] [Swarm] RX error: {e}")
                break
            try:
                msg = json.loads(data.decode("utf-8", errors="ignore"))
                if msg.get("type") == "threat_update":
                    ip = msg.get("ip")
                    score = float(msg.get("score", 0.0))
                    self._reputation.update_from_swarm(ip, score)
            except Exception:
                continue
        try:
            sock.close()
        except Exception:
            pass

    def _loop_tx(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        while not self._stop_event.is_set():
            snapshot = self._registry.snapshot()
            # broadcast top N high-threat IPs
            top = sorted(
                [c for c in snapshot if c.get("host")],
                key=lambda c: c.get("threat_score", 0.0),
                reverse=True
            )[:5]
            for c in top:
                ip = c.get("host")
                score = c.get("threat_score", 0.0)
                if score < 5.0:
                    continue
                msg = json.dumps({"type": "threat_update", "ip": ip, "score": score}).encode("utf-8")
                for peer in self._peers:
                    try:
                        sock.sendto(msg, (peer, SWARM_UDP_PORT))
                    except Exception:
                        pass
            time.sleep(SWARM_BROADCAST_INTERVAL)
        try:
            sock.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Sniffer
# ---------------------------------------------------------------------------

class QueenSniffer(QObject):
    log_message = Signal(str)
    timeline_event = Signal(str)

    def __init__(self, registry: ConnectionRegistry, firewall: FirewallBlocker,
                 lists: ListManager, reputation: ReputationManager, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._firewall = firewall
        self._lists = lists
        self._reputation = reputation
        self._stop_event = threading.Event()
        self._thread = None
        if SCAPY_AVAILABLE:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        else:
            self.log_message.emit(f"[{now_ts()}] [Sniffer] Scapy not available – install with 'pip install scapy' to enable packet capture.")

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _timeline(self, msg: str):
        self.timeline_event.emit(msg)

    def stop(self):
        self._stop_event.set()

    def _handle_packet(self, pkt):
        if IP not in pkt:
            return
        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto
        proto_name = {6: "TCP", 17: "UDP"}.get(proto, str(proto))
        sport = dport = None
        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        line = f"[{now_ts()}] [Sniffer] {src}:{sport or '-'} -> {dst}:{dport or '-'} [{proto_name}]"
        self._log(line)

        proc_name, pid = (None, None)
        if src.startswith(("192.", "10.", "172.")):
            proc_name, pid = find_process_for_connection(dst, dport or 0)
        lineage = get_process_lineage(pid) if pid else None
        geo_info = geoip_asn_lookup(dst)
        rep_score = self._reputation.get_score(dst)

        idx = self._registry.add_connection({
            "kind": "sniff",
            "host": dst,
            "port": dport or 0,
            "secure": None,
            "reason": "observed_packet",
            "fingerprint": None,
            "ips": [dst],
            "process_name": proc_name,
            "pid": pid,
            "lineage": lineage,
            "geo": geo_info,
            "reputation": rep_score,
            "timeline_label": f"SNIFF {src}->{dst}:{dport or '-'}",
        })
        self._timeline(f"[{now_ts()}] SNIFF {src}->{dst}:{dport or '-'} [{proto_name}]")

        if self._lists.is_blocked(dst):
            self._log(f"[{now_ts()}] [Sniffer] Destination {dst} is in blocked list.")
        elif not self._lists.is_allowed(dst):
            self._lists.add_pending(dst)

    def _loop(self):
        self._log(f"[{now_ts()}] [Sniffer] Starting packet capture.")
        try:
            sniff(prn=self._handle_packet, store=False, stop_filter=lambda _: self._stop_event.is_set())
        except Exception as e:
            self._log(f"[{now_ts()}] [Sniffer] Error during sniffing: {e}")

# ---------------------------------------------------------------------------
# Bridge v2 (driver / ETW / kernel integration)
# ---------------------------------------------------------------------------

class QueenBridge(QObject):
    log_message = Signal(str)
    timeline_event = Signal(str)

    def __init__(self, registry: ConnectionRegistry, policy: LockdownPolicy,
                 firewall: FirewallBlocker, lists: ListManager,
                 reputation: ReputationManager, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._policy = policy
        self._firewall = firewall
        self._lists = lists
        self._reputation = reputation
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _timeline(self, msg: str):
        self.timeline_event.emit(msg)

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        host = BORG_BRIDGE_HOST
        port = BORG_BRIDGE_PORT
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(5)
            self._log(f"[{now_ts()}] [Bridge] Listening on {host}:{port} for driver/service events.")
        except Exception as e:
            self._log(f"[{now_ts()}] [Bridge] Failed to bind bridge socket: {e}")
            return
        while not self._stop_event.is_set():
            try:
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
            except Exception as e:
                self._log(f"[{now_ts()}] [Bridge] Error in accept loop: {e}")
                break
        try:
            srv.close()
        except Exception:
            pass

    def _handle_client(self, conn, addr):
        try:
            data = conn.recv(4096)
            if not data:
                return
            try:
                msg = json.loads(data.decode("utf-8", errors="ignore"))
            except Exception:
                self._log(f"[{now_ts()}] [Bridge] Invalid JSON from client.")
                return

            if msg.get("event") == "new_connection":
                pid = msg.get("pid")
                proc = msg.get("process")
                rip = msg.get("remote_ip")
                rport = int(msg.get("remote_port") or 0)
                etw_tag = msg.get("source", "driver")
                self._log(f"[{now_ts()}] [Bridge] New connection from {etw_tag}: {proc}({pid}) -> {rip}:{rport}")
                lineage = get_process_lineage(pid)
                geo_info = geoip_asn_lookup(rip)
                rep_score = self._reputation.get_score(rip)

                idx = self._registry.add_connection({
                    "kind": "bridge",
                    "host": rip,
                    "port": rport,
                    "secure": None,
                    "reason": "bridge_observed",
                    "fingerprint": None,
                    "ips": [rip],
                    "process_name": proc,
                    "pid": pid,
                    "lineage": lineage,
                    "geo": geo_info,
                    "reputation": rep_score,
                    "timeline_label": f"BRIDGE {proc}({pid})->{rip}:{rport}",
                })
                self._timeline(f"[{now_ts()}] BRIDGE {proc}({pid})->{rip}:{rport}")

                decision = "allow"
                reason = "ok"
                if self._lists.is_blocked(rip):
                    decision = "block"
                    reason = "list_block"
                elif self._policy.is_enabled() and rport not in self._policy.allowed_ports:
                    decision = "block"
                    reason = "lockdown_port_block"
                    self._registry.add_tag(idx, "lockdown_block")
                    self._lists.add_pending(rip)

                resp = {"decision": decision, "reason": reason}
                conn.sendall(json.dumps(resp).encode("utf-8"))
            else:
                self._log(f"[{now_ts()}] [Bridge] Unknown event type.")
        except Exception as e:
            self._log(f"[{now_ts()}] [Bridge] Error handling client {addr}: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Cockpit UI
# ---------------------------------------------------------------------------

class SecureCockpit(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Borg Brain v3 – Sentinel Cockpit")
        self.setMinimumSize(1200, 720)

        self._early_logs = []

        self.registry = ConnectionRegistry()
        self.policy = LockdownPolicy()
        self.list_manager = ListManager(self._lists_log)
        self.reputation = ReputationManager(self._lists_log)

        self.firewall = FirewallBlocker(self._firewall_log, self.list_manager)
        self.resolver = QueenResolver(self.registry, self.policy)
        self.guardian = QueenGuardian(self.registry, self.policy, self.resolver,
                                      self.firewall, self.list_manager, self.reputation)
        self.scanner = QueenScanner(self.registry, self.policy, self.firewall, self.list_manager)
        self.lockdown = QueenLockdown(self.policy)
        self.anomaly = QueenAnomaly(self.registry, self.list_manager, self.reputation)
        self.threat = QueenThreat(self.registry, self.firewall, self.list_manager, self.reputation)
        self.swarm = QueenSwarm(self.registry, self.reputation)
        self.sniffer = QueenSniffer(self.registry, self.firewall, self.list_manager, self.reputation)
        self.bridge = QueenBridge(self.registry, self.policy, self.firewall, self.list_manager, self.reputation)

        self.guardian.connection_state_changed.connect(self.on_connection_state_changed)
        self.guardian.log_message.connect(self.append_log)
        self.guardian.timeline_event.connect(self.append_timeline)

        self.scanner.log_message.connect(self.append_log)
        self.scanner.threat_detected.connect(self.on_threat_detected)
        self.scanner.timeline_event.connect(self.append_timeline)

        self.lockdown.lockdown_state_changed.connect(self.on_lockdown_state_changed)
        self.lockdown.manual_override_changed.connect(self.on_manual_override_changed)
        self.lockdown.log_message.connect(self.append_log)
        self.lockdown.timeline_event.connect(self.append_timeline)

        self.resolver.log_message.connect(self.append_log)

        self.sniffer.log_message.connect(self.append_log)
        self.sniffer.timeline_event.connect(self.append_timeline)

        self.anomaly.log_message.connect(self.append_log)
        self.anomaly.timeline_event.connect(self.append_timeline)

        self.threat.log_message.connect(self.append_log)
        self.threat.timeline_event.connect(self.append_timeline)

        self.swarm.log_message.connect(self.append_log)

        self.bridge.log_message.connect(self.append_log)
        self.bridge.timeline_event.connect(self.append_timeline)

        self._build_ui()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh_status_summary)
        self._refresh_timer.start(4000)

        self._lists_timer = QTimer(self)
        self._lists_timer.timeout.connect(self.refresh_ip_lists)
        self._lists_timer.start(5000)

        self._heatmap_timer = QTimer(self)
        self._heatmap_timer.timeout.connect(self.refresh_heatmap)
        self._heatmap_timer.start(6000)

    # Logging helpers
    def _firewall_log(self, msg: str):
        self.append_log(msg)

    def _lists_log(self, msg: str):
        self.append_log(msg)

    # UI build
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

        title_label = QLabel("BORG BRAIN v3 – SENTINEL COCKPIT")
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

        # Connection controls
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

        self.lockdown_checkbox = QCheckBox("Enable Lockdown Mode")
        self.lockdown_checkbox.stateChanged.connect(self.on_lockdown_checkbox_changed)

        self.override_checkbox = QCheckBox("Manual Override (allow under lockdown)")
        self.override_checkbox.stateChanged.connect(self.on_override_checkbox_changed)

        conn_layout.addWidget(host_label, 0, 0)
        conn_layout.addWidget(self.host_edit, 0, 1)
        conn_layout.addWidget(port_label, 0, 2)
        conn_layout.addWidget(self.port_edit, 0, 3)
        conn_layout.addWidget(fp_label, 1, 0)
        conn_layout.addWidget(self.fp_edit, 1, 1, 1, 3)
        conn_layout.addWidget(self.connect_button, 2, 0, 1, 4)
        conn_layout.addWidget(self.lockdown_checkbox, 3, 0, 1, 4)
        conn_layout.addWidget(self.override_checkbox, 4, 0, 1, 4)

        # Middle: tabs (IP lists, Timeline, Heatmap)
        tabs = QTabWidget()

        # IP lists tab
        lists_tab = QWidget()
        lists_layout = QVBoxLayout(lists_tab)
        lists_layout.setContentsMargins(10, 10, 10, 10)
        lists_layout.setSpacing(8)

        lists_title = QLabel("IP Intelligence – Allowed / Blocked / Pending")
        lists_layout.addWidget(lists_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        lists_layout.addWidget(scroll)

        lists_container = QWidget()
        scroll.setWidget(lists_container)
        lists_container_layout = QVBoxLayout(lists_container)
        lists_container_layout.setContentsMargins(0, 0, 0, 0)
        lists_container_layout.setSpacing(8)

        self.allowed_group = QGroupBox("Allowed IPs (Common + Admin Approved)")
        self.allowed_group.setCheckable(True)
        self.allowed_group.setChecked(True)
        allowed_layout = QVBoxLayout(self.allowed_group)
        self.allowed_list_widget = QListWidget()
        allowed_layout.addWidget(self.allowed_list_widget)
        self.allowed_group.toggled.connect(lambda checked: self.allowed_list_widget.setVisible(checked))

        self.blocked_group = QGroupBox("Blocked IPs (Borg + Firewall)")
        self.blocked_group.setCheckable(True)
        self.blocked_group.setChecked(True)
        blocked_layout = QVBoxLayout(self.blocked_group)
        self.blocked_list_widget = QListWidget()
        blocked_layout.addWidget(self.blocked_list_widget)
        self.blocked_group.toggled.connect(lambda checked: self.blocked_list_widget.setVisible(checked))

        self.pending_group = QGroupBox("Pending IPs (Admin Decision)")
        self.pending_group.setCheckable(True)
        self.pending_group.setChecked(True)
        pending_layout = QVBoxLayout(self.pending_group)
        self.pending_list_widget = QListWidget()
        pending_layout.addWidget(self.pending_list_widget)

        btn_row = QHBoxLayout()
        self.btn_allow_ip = QPushButton("Allow Selected")
        self.btn_block_ip = QPushButton("Block Selected (Borg-only)")
        self.btn_block_firewall_ip = QPushButton("Block Selected (Firewall)")
        self.btn_ignore_ip = QPushButton("Ignore Selected")
        btn_row.addWidget(self.btn_allow_ip)
        btn_row.addWidget(self.btn_block_ip)
        btn_row.addWidget(self.btn_block_firewall_ip)
        btn_row.addWidget(self.btn_ignore_ip)
        pending_layout.addLayout(btn_row)

        self.pending_group.toggled.connect(lambda checked: self.pending_list_widget.setVisible(checked))

        self.btn_allow_ip.clicked.connect(self.on_allow_selected_ip)
        self.btn_block_ip.clicked.connect(self.on_block_selected_ip_borg)
        self.btn_block_firewall_ip.clicked.connect(self.on_block_selected_ip_firewall)
        self.btn_ignore_ip.clicked.connect(self.on_ignore_selected_ip)

        lists_container_layout.addWidget(self.allowed_group)
        lists_container_layout.addWidget(self.blocked_group)
        lists_container_layout.addWidget(self.pending_group)
        lists_container_layout.addStretch(1)

        tabs.addTab(lists_tab, "IP Lists")

        # Timeline tab
        timeline_tab = QWidget()
        timeline_layout = QVBoxLayout(timeline_tab)
        timeline_layout.setContentsMargins(10, 10, 10, 10)
        timeline_layout.setSpacing(8)
        timeline_label = QLabel("Timeline – Event Stream")
        self.timeline_view = QTextEdit()
        self.timeline_view.setReadOnly(True)
        timeline_layout.addWidget(timeline_label)
        timeline_layout.addWidget(self.timeline_view)
        tabs.addTab(timeline_tab, "Timeline")

        # Heatmap tab
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_tab)
        heatmap_layout.setContentsMargins(10, 10, 10, 10)
        heatmap_layout.setSpacing(8)
        heatmap_label = QLabel("Threat Heatmap – IP vs Threat Score")
        self.heatmap_view = QTextEdit()
        self.heatmap_view.setReadOnly(True)
        heatmap_layout.addWidget(heatmap_label)
        heatmap_layout.addWidget(self.heatmap_view)
        tabs.addTab(heatmap_tab, "Threat Heatmap")

        # Log frame
        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(10, 10, 10, 10)

        log_label = QLabel("Event Log:")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.status_summary = QLabel("Connections: 0 | Secure: 0 | Insecure: 0 | Observed: 0 | High Anomaly: 0 | Allowed IPs: 0 | Blocked IPs: 0 | Pending IPs: 0")
        self.status_summary.setObjectName("statusSummary")

        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_view)
        log_layout.addWidget(self.status_summary)

        root_layout.addWidget(cockpit_frame)
        root_layout.addWidget(conn_frame)
        root_layout.addWidget(tabs)
        root_layout.addWidget(log_frame)

        self._apply_styles()
        self._set_secure_indicator(False, "NOT SECURE")
        self._set_lockdown_indicator(False)
        self._set_override_indicator(False)
        self.refresh_ip_lists()

        if hasattr(self, "_early_logs") and self._early_logs:
            for m in self._early_logs:
                self.log_view.append(m)
            self._early_logs = []

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
        QGroupBox {
            color: #E5E7EB;
            font-weight: 600;
        }
        QListWidget {
            background-color: #020617;
            color: #E5E7EB;
            border: 1px solid #374151;
            border-radius: 4px;
        }
        """)

    def _set_secure_indicator(self, secure: bool, text: str):
        self.secure_indicator.setText(text)
        if secure:
            self.secure_indicator.setStyleSheet("""
                border-radius: 14px;
                padding: 6px 10px;
                color: #FFFFFF;
                font-weight: 600;
                background-color: #16A34A;
            """)
        else:
            self.secure_indicator.setStyleSheet("""
                border-radius: 14px;
                padding: 6px 10px;
                color: #FFFFFF;
                font-weight: 600;
                background-color: #B91C1C;
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

    # Slots
    @Slot()
    def on_connect_clicked(self):
        host = self.host_edit.text().strip()
        port_text = self.port_edit.text().strip()
        fp = self.fp_edit.text().strip()
        if not host:
            self.append_log(f"[{now_ts()}] Host is required.")
            return
        try:
            port = int(port_text)
        except ValueError:
            self.append_log(f"[{now_ts()}] Port must be an integer.")
            return
        self.append_log(f"[{now_ts()}] [Operator] Requesting secure connection to {host}:{port} via Borg...")
        self.guardian.connect_secure(host, port, fp)

    @Slot(str, bool)
    def on_connection_state_changed(self, message: str, is_secure: bool):
        self._set_secure_indicator(is_secure, "SECURE" if is_secure else "NOT SECURE")
        self.append_log(message)

    @Slot(str)
    def on_threat_detected(self, msg: str):
        self.append_log(msg)
        if not self.policy.is_manual_override() and not self.policy.is_enabled():
            self.append_log(f"[{now_ts()}] [Scanner] Threat detected – auto-enabling lockdown.")
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

    @Slot(str)
    def append_log(self, msg: str):
        if not hasattr(self, "log_view") or self.log_view is None:
            if not hasattr(self, "_early_logs"):
                self._early_logs = []
            self._early_logs.append(msg)
            return
        self.log_view.append(msg)

    @Slot(str)
    def append_timeline(self, msg: str):
        if hasattr(self, "timeline_view") and self.timeline_view is not None:
            self.timeline_view.append(msg)

    @Slot()
    def refresh_status_summary(self):
        snapshot = self.registry.snapshot()
        total = len(snapshot)
        secure = sum(1 for c in snapshot if c.get("secure") is True)
        insecure = sum(1 for c in snapshot if c.get("secure") is False and c.get("kind") == "active")
        observed = sum(1 for c in snapshot if c.get("kind") in ("sniff", "bridge"))
        high_anom = sum(1 for c in snapshot if c.get("anomaly_score", 0.0) >= 4.0)
        allowed_count = len(self.list_manager.allowed_ips)
        blocked_count = len(self.list_manager.blocked_ips)
        pending_count = len(self.list_manager.pending_ips)
        self.status_summary.setText(
            f"Connections: {total} | Secure: {secure} | Insecure: {insecure} | "
            f"Observed: {observed} | High Anomaly: {high_anom} | "
            f"Allowed IPs: {allowed_count} | Blocked IPs: {blocked_count} | Pending IPs: {pending_count}"
        )

    @Slot()
    def refresh_ip_lists(self):
        self.allowed_list_widget.clear()
        for ip in sorted(self.list_manager.allowed_ips):
            self.allowed_list_widget.addItem(QListWidgetItem(ip))
        self.blocked_list_widget.clear()
        for ip in sorted(self.list_manager.blocked_ips):
            self.blocked_list_widget.addItem(QListWidgetItem(ip))
        self.pending_list_widget.clear()
        for ip in sorted(self.list_manager.pending_ips):
            self.pending_list_widget.addItem(QListWidgetItem(ip))

    @Slot()
    def refresh_heatmap(self):
        snapshot = self.registry.snapshot()
        scores_by_ip = defaultdict(float)
        for c in snapshot:
            ip = c.get("host")
            if not ip:
                continue
            scores_by_ip[ip] = max(scores_by_ip[ip], c.get("threat_score", 0.0))
        lines = ["IP                | ThreatScore", "------------------+------------"]
        for ip, score in sorted(scores_by_ip.items(), key=lambda kv: kv[1], reverse=True)[:30]:
            lines.append(f"{ip:<18} | {score:>10.2f}")
        self.heatmap_view.setPlainText("\n".join(lines))

    # Pending IP actions
    def _get_selected_pending_ip(self):
        item = self.pending_list_widget.currentItem()
        if not item:
            return None
        return item.text().strip()

    @Slot()
    def on_allow_selected_ip(self):
        ip = self._get_selected_pending_ip()
        if not ip:
            self.append_log(f"[{now_ts()}] [Lists] No pending IP selected to allow.")
            return
        self.list_manager.add_allowed(ip, persist=True)
        self.refresh_ip_lists()

    @Slot()
    def on_block_selected_ip_borg(self):
        ip = self._get_selected_pending_ip()
        if not ip:
            self.append_log(f"[{now_ts()}] [Lists] No pending IP selected to block (Borg-only).")
            return
        self.firewall.block_ip_borg_only(ip, reason="admin_borg_block")
        self.refresh_ip_lists()

    @Slot()
    def on_block_selected_ip_firewall(self):
        ip = self._get_selected_pending_ip()
        if not ip:
            self.append_log(f"[{now_ts()}] [Lists] No pending IP selected to block (Firewall).")
            return
        self.firewall.block_ip_firewall(ip, reason="admin_firewall_block")
        self.refresh_ip_lists()

    @Slot()
    def on_ignore_selected_ip(self):
        ip = self._get_selected_pending_ip()
        if not ip:
            self.append_log(f"[{now_ts()}] [Lists] No pending IP selected to ignore.")
            return
        self.list_manager.remove_pending(ip)
        self.refresh_ip_lists()

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
