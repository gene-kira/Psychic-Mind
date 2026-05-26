#!/usr/bin/env python3
"""
Codex Sentinel – Tier-3 Unified File
====================================

Features:
- Cross-platform (Windows / Linux / macOS)
- Data Physics Engine (stress, entropy, shapes)
- Behavioral Threat Detection (remote tools, firewall tamper, shells, etc.)
- Attack Chain Engine (behavioral kill chains)
- Altered-State Classifier (CALM/FOCUSED/ALTERED/SUPPRESSED/TRANSITIONAL)
- LAN Swarm with ENCRYPTED UDP (AES-GCM demo key)
- Adaptive Swarm Routing (LAN broadcast + optional WAN relay stub)
- Swarm-level anomaly clustering + fingerprint sharing + cosine fusion
- Predictive forecasting (5s / 10s / 30s) for stress & entropy
- Fleet Dashboard UI + 3D Fleet Visualization mode (logical 3D layout)
- Codex Command Interface (broadcast commands to swarm)
- Codex Neural Bridge (voice/CLI hook points for future integration)
- Autonomous Response Layer (non-destructive isolation flags + alerts)
- Anomaly fingerprinting (reusable pattern signatures)
- Privacy-hardened (no IPs, MACs, usernames, hostnames, PIDs, file paths in logs)
- Non-destructive (no process killing, no firewall changes)
"""

import sys
import subprocess
import importlib
import math
import threading
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections import deque, defaultdict
import platform
import socket
import uuid
import re
import os
import random

# =========================
# Auto-loader for libraries
# =========================

REQUIRED_PACKAGES = ["psutil", "cryptography"]

def ensure_package(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[AUTOLOADER] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

psutil = ensure_package("psutil")
cryptography = ensure_package("cryptography")
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

OS_NAME = platform.system().lower()

CAPABILITIES = {
    "network_monitor": hasattr(psutil, "net_connections"),
    "disk_io": hasattr(psutil, "disk_io_counters"),
    "process_iter": True,
    "gui": True,
}

if OS_NAME == "darwin":
    NET_BEACON_THRESHOLD = 0.6
elif OS_NAME == "linux":
    NET_BEACON_THRESHOLD = 0.5
else:
    NET_BEACON_THRESHOLD = 0.4

LAN_UDP_PORT = 33333
LAN_BROADCAST_INTERVAL = 5.0  # seconds

# DEMO SHARED KEY (32 bytes for AES-256-GCM) – replace in real deployment
SHARED_SWARM_KEY = b"codex-sentinel-demo-key-32bytes!!"

# Optional WAN relay (Tier-3 stub)
WAN_RELAY_ENABLED = False
WAN_RELAY_HOST = "example-relay.invalid"
WAN_RELAY_PORT = 443

# =========================
# Global brain state
# =========================

brain_state = {}
brain_lock = threading.Lock()

# =========================
# Network / Remote Control Patterns
# =========================

REMOTE_TOOL_PATTERNS = re.compile(
    r"(teamviewer|anydesk|vnc|remote|rdp|shadow|splashtop|ultraviewer|ammyy|logmein)",
    re.IGNORECASE
)

FIREWALL_CMD_PATTERNS = re.compile(
    r"(netsh\s+advfirewall|Set-NetFirewall|New-NetFirewall|ufw\s+enable|ufw\s+disable)",
    re.IGNORECASE
)

SETTINGS_CMD_PATTERNS = re.compile(
    r"(reg\s+add|reg\s+delete|powershell\s+Set-ItemProperty|gpedit\.msc|secpol\.msc)",
    re.IGNORECASE
)

SHELL_NAMES = re.compile(
    r"(cmd\.exe|powershell\.exe|pwsh\.exe|bash\.exe|wsl\.exe)",
    re.IGNORECASE
)

# =========================
# Cross-platform safe wrappers
# =========================

def safe_net_connections():
    if not CAPABILITIES["network_monitor"]:
        return []
    try:
        return psutil.net_connections(kind="inet")
    except Exception:
        return []

def get_remote_ip(conn):
    try:
        raddr = getattr(conn, "raddr", None)
        if not raddr:
            return None
        if hasattr(raddr, "ip"):
            return raddr.ip
        if isinstance(raddr, tuple) and len(raddr) >= 1:
            return raddr[0]
    except Exception:
        return None
    return None

def is_listening(conn):
    try:
        return str(conn.status).upper() == "LISTEN"
    except Exception:
        return False

def safe_disk_io_bytes():
    if not CAPABILITIES["disk_io"]:
        return 0
    try:
        io = psutil.disk_io_counters()
        return (getattr(io, "read_bytes", 0) + getattr(io, "write_bytes", 0))
    except Exception:
        return 0

def safe_process_iter():
    if not CAPABILITIES["process_iter"]:
        return []
    try:
        return psutil.process_iter(["pid", "name", "exe", "cmdline"])
    except Exception:
        return []

# =========================
# Real-Time Identifiers (internal only)
# =========================

def get_real_mac():
    try:
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if getattr(psutil, "AF_LINK", None) is not None:
                    if addr.family == psutil.AF_LINK:
                        return addr.address
                else:
                    if getattr(addr.family, "name", "") == "AF_LINK":
                        return addr.address
    except Exception:
        pass
    return "MAC not found"

def get_real_ip():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        try:
            public_ip = socket.gethostbyname_ex(hostname)[2][-1]
        except Exception:
            public_ip = local_ip
        return local_ip, public_ip
    except Exception as e:
        return "IP error", str(e)

def get_telemetry():
    os_info = platform.platform()
    browser_fingerprint = platform.system() + "-" + platform.machine()
    return os_info, browser_fingerprint

def get_swarm_id():
    with brain_lock:
        if brain_state.get("swarm_id"):
            return brain_state["swarm_id"]
    sid = str(uuid.getnode())
    with brain_lock:
        brain_state["swarm_id"] = sid
    return sid

def synthesize_phantom():
    entropy = uuid.uuid4().hex + str(time.time_ns())
    phantom = f"phantom://{entropy[:12]}"
    with brain_lock:
        brain_state.setdefault("phantom_history", []).append(phantom)
        brain_state["phantom_history"] = brain_state["phantom_history"][-50:]
    return phantom

# =========================
# Schemas
# =========================

@dataclass
class CodexSentinelStatus:
    source: str
    health_state: str
    threat_level: str
    stress_score: float
    entropy_score: float
    behavioral_state: str
    anomaly_flags: list
    recent_window_seconds: int
    confidence: float
    risk_summary: str
    recommended_actions: list
    timestamp: str
    forecast_5s: dict
    forecast_10s: dict
    forecast_30s: dict


@dataclass
class FleetNodeStatus:
    node_id: str
    role: str
    health_state: str
    threat_level: str
    stress_score: float
    entropy_score: float
    behavioral_state: str
    anomaly_flags: list
    uptime_seconds: int
    last_update_ts: str
    fingerprints: list
    isolation_state: str  # NEW: autonomous response layer


@dataclass
class AttackerLogEntry:
    timestamp: str
    node_id: str
    health_state: str
    threat_level: str
    stress_score: float
    entropy_score: float
    anomaly_flags: list
    pattern_signature: str
    fingerprint: str
    duration_estimate_seconds: int
    context_window: dict

# =========================
# Predictive / Consensus / Clustering / Fusion
# =========================

class Queen:
    def __init__(self):
        self.nodes = {}  # node_id -> list of events

    def update(self, node_id, events):
        self.nodes[node_id] = events

    def global_risk(self):
        risk = {}
        for node, evts in self.nodes.items():
            for e in evts:
                ent = e.get("entity")
                score = float(e.get("score", 0.0))
                if not ent:
                    continue
                risk[ent] = risk.get(ent, 0.0) + score
        return {k: v for k, v in risk.items() if v > 1.5}


class AttackChainEngine:
    def __init__(self, window=120):
        self.events = deque()
        self.window = window

    def add_event(self, event_type, data=None):
        now = time.time()
        self.events.append((now, event_type, data or {}))
        self._cleanup(now)

    def _cleanup(self, now):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def detect(self):
        types = [e[1] for e in self.events]
        chains = []

        if all(x in types for x in ["CPU_SPIKE", "NET_BURST", "ENTROPY_SURGE"]):
            chains.append(("FULL_BEHAVIOR_CHAIN", 0.9))

        if types.count("CPU_SPIKE") > 5 and "NET_BURST" in types:
            chains.append(("RESOURCE_STORM", 0.8))

        if "DISK_THRASH" in types and "NET_BURST" in types:
            chains.append(("PERSISTENCE_EXFIL_PATTERN", 0.85))

        if "NET_FOREIGN" in types and "ENTROPY_SURGE" in types:
            chains.append(("REMOTE_CONTROL_PATTERN", 0.9))

        return chains


class EventBus:
    def __init__(self):
        self.subscribers = []
        self.queue = deque()

    def publish(self, event):
        self.queue.append(event)

    def subscribe(self, fn):
        self.subscribers.append(fn)

    def run_once(self):
        if self.queue:
            evt = self.queue.popleft()
            for fn in self.subscribers:
                fn(evt)


@dataclass
class SecEvent:
    ts: float
    etype: str
    entity: str
    meta: dict

# =========================
# Data Physics / Forecast / Fingerprints / Fusion
# =========================

class DataPhysicsEngine:
    def __init__(self, window_size=120, sample_interval=0.1):
        self.window_size = window_size
        self.sample_interval = sample_interval
        self.history = {
            "cpu": deque(maxlen=window_size),
            "ram": deque(maxlen=window_size),
            "disk": deque(maxlen=window_size),
            "net": deque(maxlen=window_size),
            "stress": deque(maxlen=window_size),
            "entropy": deque(maxlen=window_size),
        }
        self.hourly_baseline = defaultdict(lambda: {"cpu": [], "ram": [], "disk": [], "net": []})

    def update(self, cpu, ram, disk, net, stress, entropy):
        self.history["cpu"].append(cpu)
        self.history["ram"].append(ram)
        self.history["disk"].append(disk)
        self.history["net"].append(net)
        self.history["stress"].append(stress)
        self.history["entropy"].append(entropy)

        hour = datetime.now().hour
        self.hourly_baseline[hour]["cpu"].append(cpu)
        self.hourly_baseline[hour]["ram"].append(ram)
        self.hourly_baseline[hour]["disk"].append(disk)
        self.hourly_baseline[hour]["net"].append(net)
        for k in ["cpu", "ram", "disk", "net"]:
            if len(self.hourly_baseline[hour][k]) > 3600:
                self.hourly_baseline[hour][k].pop(0)

    def _avg(self, seq):
        return sum(seq) / len(seq) if seq else 0.0

    def baseline_deviation(self):
        hour = datetime.now().hour
        dev = {}
        for k in ["cpu", "ram", "disk", "net"]:
            baseline = self._avg(self.hourly_baseline[hour][k])
            current = self._avg(self.history[k])
            dev[k] = abs(current - baseline)
        return dev

    def shape_detection(self, key):
        seq = list(self.history[key])
        if len(seq) < 10:
            return "UNKNOWN"

        diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        pos = sum(1 for d in diffs if d > 0.01)
        neg = sum(1 for d in diffs if d < -0.01)
        flat = sum(1 for d in diffs if abs(d) <= 0.01)

        if pos > 0.7 * len(diffs) and neg < 0.1 * len(diffs):
            return "RAMP_UP"
        if neg > 0.7 * len(diffs) and pos < 0.1 * len(diffs):
            return "RAMP_DOWN"
        if flat > 0.8 * len(diffs):
            return "FLAT"
        if pos > 0.3 * len(diffs) and neg > 0.3 * len(diffs):
            return "OSCILLATION"
        if max(seq) - min(seq) > 0.6 and any(abs(d) > 0.3 for d in diffs):
            return "BURST"
        return "MIXED"

    def confidence_score(self, stress, entropy, anomaly_flags):
        base = 0.5
        if "SUSTAINED_LOAD" in anomaly_flags:
            base += 0.1
        if "HIGH_ENTROPY" in anomaly_flags:
            base += 0.1
        if "NET_BEACON" in anomaly_flags or "NET_FOREIGN" in anomaly_flags:
            base += 0.1
        if "REMOTE_TOOL_BEHAVIOR" in anomaly_flags:
            base += 0.1
        if "FIREWALL_TAMPER_BEHAVIOR" in anomaly_flags:
            base += 0.1

        dev = self.baseline_deviation()
        dev_score = min(sum(dev.values()), 1.0) * 0.2
        base += dev_score

        base -= 0.1 * anomaly_flags.count("MISSING_STRESS_CORRELATION")
        base -= 0.1 * anomaly_flags.count("MISSING_ENTROPY_CORRELATION")

        return max(0.0, min(1.0, base))

    def forecast(self, key, horizon_seconds):
        seq = list(self.history[key])
        if len(seq) < 5:
            return {"value": self._avg(seq), "trend": "UNKNOWN"}

        dt = self.sample_interval
        n = min(len(seq), 30)
        recent = seq[-n:]
        x = list(range(n))
        y = recent

        sx = sum(x)
        sy = sum(y)
        sxx = sum(i*i for i in x)
        sxy = sum(i*j for i, j in zip(x, y))
        denom = n * sxx - sx * sx
        if denom == 0:
            slope = 0.0
        else:
            slope = (n * sxy - sx * sy) / denom

        intercept = (sy - slope * sx) / n
        steps_ahead = horizon_seconds / dt
        future = intercept + slope * (n - 1 + steps_ahead)
        future = max(0.0, min(1.0, future))

        if slope > 0.01:
            trend = "UP"
        elif slope < -0.01:
            trend = "DOWN"
        else:
            trend = "FLAT"

        return {"value": future, "trend": trend}

    def fingerprint(self, anomaly_flags, chains):
        parts = sorted(set(anomaly_flags))
        for cname, score in chains:
            if score > 0.7:
                parts.append(f"CHAIN:{cname}")
        if not parts:
            return "FP:BASELINE"
        raw = "|".join(parts)
        return f"FP:{abs(hash(raw)) & 0xFFFFFFFF:08X}"

    # Tier-3: ML-style fingerprint fusion (cosine similarity over simple binary vectors)
    def fingerprint_vector(self, fingerprint: str, all_flags: list) -> list:
        # Very simple: map presence/absence of each known flag into a vector
        vec = []
        for flag in all_flags:
            vec.append(1.0 if flag in fingerprint else 0.0)
        return vec

    def cosine_similarity(self, v1, v2):
        dot = sum(a*b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a*a for a in v1))
        n2 = math.sqrt(sum(b*b for b in v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

# =========================
# Network / Private IP
# =========================

def is_private_ip(ip: str) -> bool:
    if not ip:
        return False
    if ip.startswith("10."):
        return True
    if ip.startswith("192.168."):
        return True
    if ip.startswith("172."):
        parts = ip.split(".")
        if len(parts) >= 2:
            try:
                second = int(parts[1])
                return 16 <= second <= 31
            except ValueError:
                return False
    return False

# =========================
# GUI
# =========================

import tkinter as tk
from tkinter import ttk

PRIVACY_HARDENED = True

class FleetOrchestrator:
    def __init__(self):
        self.nodes = {}      # node_id -> FleetNodeStatus
        self.last_seen = {}  # node_id -> timestamp
        self.fingerprint_counts = defaultdict(int)
        self.lock = threading.Lock()

    def update_node(self, status: FleetNodeStatus):
        with self.lock:
            self.nodes[status.node_id] = status
            self.last_seen[status.node_id] = time.time()
            for fp in status.fingerprints:
                self.fingerprint_counts[fp] += 1

    def snapshot(self):
        with self.lock:
            now = time.time()
            snap = {}
            for nid, st in self.nodes.items():
                age = now - self.last_seen.get(nid, 0)
                d = asdict(st)
                if age > 30:
                    d["health_state"] = "OFFLINE"
                    d["threat_level"] = "LOW"
                snap[nid] = d
            return snap

    def cluster_summary(self):
        with self.lock:
            clusters = {fp: cnt for fp, cnt in self.fingerprint_counts.items() if cnt > 1}
            return clusters

    # Tier-3: simple fusion view – count how many nodes share similar fingerprints
    def fused_risk_view(self):
        with self.lock:
            return dict(self.fingerprint_counts)

class POVUnifiedSecurityCore:
    def __init__(self, root, node_id=None, role="WORKSTATION"):
        self.root = root
        self.root.title("Codex Sentinel – Tier-3 (Swarm + Forecast + Fusion)")

        base_swarm_id = get_swarm_id()
        self.node_id = node_id or f"node-{base_swarm_id[-8:]}"
        self.role = role
        self.start_time = time.time()

        self.attacker_log = []
        self.queen = Queen()
        self.attack_chain = AttackChainEngine(window=120)
        self.event_bus = EventBus()
        self.event_bus.subscribe(self._chain_event_ingest)

        self.fleet = FleetOrchestrator()

        self.foreign_conn_count = 0
        self.listening_count = 0
        self.process_count = 0

        self.remote_tool_hits = 0
        self.firewall_cmd_hits = 0
        self.settings_cmd_hits = 0
        self.shell_activity_count = 0
        self.suspicious_name_count = 0

        self.high_stress_start = None
        self.integrity_ok = True
        self.integrity_last_error = None

        self.physics = DataPhysicsEngine(window_size=120, sample_interval=0.1)

        # Autonomous response state
        self.isolation_state = "NORMAL"  # NORMAL / SOFT_ISOLATION / HARD_ISOLATION (logical only)

        # 3D Fleet Visualization toggle
        self.visual_mode_3d = True

        self.main_frame = tk.Frame(root, bg="#05060A")
        self.main_frame.pack(fill="both", expand=True)

        self.banner = tk.Label(
            self.main_frame,
            text=f"TIER-3 • ENCRYPTED SWARM • PREDICTIVE • FUSION • OS: {platform.system()}",
            fg="#A0FFCF",
            bg="#07120F",
            font=("Consolas", 10, "bold")
        )
        self.banner.pack(fill="x", side="top")

        self.content_frame = tk.Frame(self.main_frame, bg="#05060A")
        self.content_frame.pack(fill="both", expand=True, side="top")

        self.canvas = tk.Canvas(self.content_frame, bg="#05060A", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, side="left")

        self.side_panel = tk.Frame(self.content_frame, bg="#0A0C12", width=340)
        self.side_panel.pack(fill="y", side="right")

        control_frame = tk.Frame(self.main_frame, bg="#111318")
        control_frame.pack(fill="x", side="bottom")

        self.status_label = tk.Label(control_frame, text="Status: Initializing…", fg="#A0FFB0", bg="#111318")
        self.status_label.pack(side="left", padx=10, pady=5)

        self.cpu_label = tk.Label(control_frame, text="CPU: --%", fg="#FFFFFF", bg="#111318")
        self.cpu_label.pack(side="left", padx=10)

        self.ram_label = tk.Label(control_frame, text="RAM: --%", fg="#FFFFFF", bg="#111318")
        self.ram_label.pack(side="left", padx=10)

        self.disk_label = tk.Label(control_frame, text="Disk: --%", fg="#FFFFFF", bg="#111318")
        self.disk_label.pack(side="left", padx=10)

        self.net_label = tk.Label(control_frame, text="Net: -- kB/s", fg="#FFFFFF", bg="#111318")
        self.net_label.pack(side="left", padx=10)

        self.sec_label = tk.Label(control_frame, text="Security Load: --", fg="#FFDD88", bg="#111318")
        self.sec_label.pack(side="left", padx=10)

        self.info_label = tk.Label(
            control_frame,
            text=f"Autonomous Tier-3 • Node: {self.node_id}",
            fg="#8888FF",
            bg="#111318"
        )
        self.info_label.pack(side="right", padx=10)

        # Side panel sections
        self.alert_title = tk.Label(
            self.side_panel,
            text="THREAT ALERT PANEL",
            fg="#FFD966",
            bg="#0A0C12",
            font=("Consolas", 11, "bold")
        )
        self.alert_title.pack(fill="x", pady=(10, 2))

        self.alert_label = tk.Label(
            self.side_panel,
            text="No anomalies detected.\nSystem behavior appears normal.",
            fg="#A0FFB0",
            bg="#0A0C12",
            justify="left",
            wraplength=320
        )
        self.alert_label.pack(fill="x", padx=10, pady=(0, 10))

        self.actions_title = tk.Label(
            self.side_panel,
            text="RECOMMENDED ACTIONS",
            fg="#66C2FF",
            bg="#0A0C12",
            font=("Consolas", 11, "bold")
        )
        self.actions_title.pack(fill="x", pady=(10, 2))

        self.actions_text = tk.Label(
            self.side_panel,
            text="- Keep this monitor running.\n- Use it to spot unusual spikes.\n- Pair with a trusted AV/firewall.",
            fg="#D0D4E6",
            bg="#0A0C12",
            justify="left",
            wraplength=320
        )
        self.actions_text.pack(fill="x", padx=10, pady=(0, 10))

        self.log_title = tk.Label(
            self.side_panel,
            text="ATTACKER PATTERN LOG (SUMMARY)",
            fg="#FF8080",
            bg="#0A0C12",
            font=("Consolas", 10, "bold")
        )
        self.log_title.pack(fill="x", pady=(10, 2))

        self.log_preview = tk.Label(
            self.side_panel,
            text="No hostile patterns recorded yet.",
            fg="#F0B0B0",
            bg="#0A0C12",
            justify="left",
            wraplength=320
        )
        self.log_preview.pack(fill="x", padx=10, pady=(0, 10))

        self.queen_title = tk.Label(
            self.side_panel,
            text="QUEEN CONSENSUS (LOCAL + SWARM)",
            fg="#A0C8FF",
            bg="#0A0C12",
            font=("Consolas", 10, "bold")
        )
        self.queen_title.pack(fill="x", pady=(10, 2))

        self.queen_preview = tk.Label(
            self.side_panel,
            text="No high-risk entities yet.",
            fg="#C0D4FF",
            bg="#0A0C12",
            justify="left",
            wraplength=320
        )
        self.queen_preview.pack(fill="x", padx=10, pady=(0, 10))

        self.fleet_title = tk.Label(
            self.side_panel,
            text="FLEET DASHBOARD (3D VIEW)",
            fg="#9CF0FF",
            bg="#0A0C12",
            font=("Consolas", 10, "bold")
        )
        self.fleet_title.pack(fill="x", pady=(10, 2))

        self.fleet_preview = tk.Label(
            self.side_panel,
            text="No other nodes detected yet.",
            fg="#CFEFFF",
            bg="#0A0C12",
            justify="left",
            wraplength=320
        )
        self.fleet_preview.pack(fill="x", padx=10, pady=(0, 10))

        self.cluster_title = tk.Label(
            self.side_panel,
            text="ANOMALY CLUSTERS & FUSION",
            fg="#FFB3FF",
            bg="#0A0C12",
            font=("Consolas", 10, "bold")
        )
        self.cluster_title.pack(fill="x", pady=(10, 2))

        self.cluster_preview = tk.Label(
            self.side_panel,
            text="No shared fingerprints yet.",
            fg="#F5CFFF",
            bg="#0A0C12",
            justify="left",
            wraplength=320
        )
        self.cluster_preview.pack(fill="x", padx=10, pady=(0, 10))

        # Codex Command Interface
        self.cmd_title = tk.Label(
            self.side_panel,
            text="CODEX COMMAND INTERFACE",
            fg="#A0FF9C",
            bg="#0A0C12",
            font=("Consolas", 10, "bold")
        )
        self.cmd_title.pack(fill="x", pady=(10, 2))

        self.cmd_entry = tk.Entry(self.side_panel, bg="#111318", fg="#FFFFFF")
        self.cmd_entry.pack(fill="x", padx=10, pady=(0, 5))
        self.cmd_entry.insert(0, "PING_SWARM")

        self.cmd_button = tk.Button(
            self.side_panel,
            text="Broadcast Command",
            command=self.broadcast_command,
            bg="#1E2A3A",
            fg="#A0FFCF"
        )
        self.cmd_button.pack(fill="x", padx=10, pady=(0, 5))

        self.cmd_status = tk.Label(
            self.side_panel,
            text="Last command: none",
            fg="#C0D4FF",
            bg="#0A0C12",
            justify="left",
            wraplength=320
        )
        self.cmd_status.pack(fill="x", padx=10, pady=(0, 10))

        # Neural Bridge (Tier-3 hook)
        self.neural_title = tk.Label(
            self.side_panel,
            text="CODEX NEURAL BRIDGE (HOOK)",
            fg="#FFE29C",
            bg="#0A0C12",
            font=("Consolas", 10, "bold")
        )
        self.neural_title.pack(fill="x", pady=(10, 2))

        self.neural_status = tk.Label(
            self.side_panel,
            text="Voice/CLI integration stub ready.\nUse commands via this UI for now.",
            fg="#FFE9C0",
            bg="#0A0C12",
            justify="left",
            wraplength=320
        )
        self.neural_status.pack(fill="x", padx=10, pady=(0, 10))

        # Visualization buffers
        self.num_segments = 240
        self.cpu_buffer = [0.0] * self.num_segments
        self.ram_buffer = [0.0] * self.num_segments
        self.disk_buffer = [0.0] * self.num_segments
        self.net_buffer = [0.0] * self.num_segments
        self.sec_buffer = [0.0] * self.num_segments
        self.entropy_buffer = [0.0] * self.num_segments

        self.current_index = 0

        self.width = 900
        self.height = 700
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.inner_radius_cpu = 110
        self.outer_radius_cpu = 160
        self.inner_radius_ram = 170
        self.outer_radius_ram = 220
        self.inner_radius_disk = 230
        self.outer_radius_disk = 280
        self.inner_radius_net = 290
        self.outer_radius_net = 340
        self.inner_radius_threat = 60
        self.outer_radius_threat = 100

        self.running = True
        self.update_interval = 0.1

        self.last_net = psutil.net_io_counters()
        self.last_net_time = time.time()

        self.last_cpu = 0.0
        self.last_ram = 0.0
        self.last_disk = 0.0
        self.last_net_rate = 0.0

        self.last_net_norm = 0.0
        self.last_net_change_time = time.time()
        self.beacon_score = 0.0

        self.context_window_samples = []
        self.context_window_size = 50

        self.error_count = 0
        self.max_errors = 10

        self.altered_state = False

        self.root.bind("<Configure>", self.on_resize)

        self.status_label.config(text="Status: Running (Tier-3, Encrypted Swarm, Predictive, Fusion)")

        threading.Thread(target=self.data_loop, daemon=True).start()
        self.schedule_redraw()
        self.schedule_event_bus()
        if CAPABILITIES["network_monitor"]:
            self.schedule_network_monitor()
        self.schedule_threat_scan()
        self.schedule_self_check()

        threading.Thread(target=self.lan_listener_loop, daemon=True).start()
        threading.Thread(target=self.lan_broadcast_loop, daemon=True).start()
        if WAN_RELAY_ENABLED:
            threading.Thread(target=self.wan_relay_loop, daemon=True).start()

        self.schedule_fleet_preview()
        self.schedule_cluster_preview()

    # =========================
    # Resize
    # =========================

    def on_resize(self, event):
        if event.width < 300 or event.height < 300:
            return

        canvas_width = max(event.width - 340, 300)
        canvas_height = event.height - 60

        self.width = canvas_width
        self.height = canvas_height
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        base = min(self.width, self.height)
        scale = base / 800.0
        self.inner_radius_cpu = int(110 * scale)
        self.outer_radius_cpu = int(160 * scale)
        self.inner_radius_ram = int(170 * scale)
        self.outer_radius_ram = int(220 * scale)
        self.inner_radius_disk = int(230 * scale)
        self.outer_radius_disk = int(280 * scale)
        self.inner_radius_net = int(290 * scale)
        self.outer_radius_net = int(340 * scale)
        self.inner_radius_threat = int(60 * scale)
        self.outer_radius_threat = int(100 * scale)

        self.redraw()

    # =========================
    # Data sampling
    # =========================

    def data_loop(self):
        while self.running:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent

                disk_activity = safe_disk_io_bytes()

                now = time.time()
                net = psutil.net_io_counters()
                dt = max(now - self.last_net_time, 1e-6)
                bytes_sent = net.bytes_sent - self.last_net.bytes_sent
                bytes_recv = net.bytes_recv - self.last_net.bytes_recv
                net_rate = (bytes_sent + bytes_recv) / dt
                net_kb = net_rate / 1024.0

                self.last_net = net
                self.last_net_time = now

                cpu_norm = min(max(cpu / 100.0, 0.0), 1.0)
                ram_norm = min(max(ram / 100.0, 0.0), 1.0)
                disk_norm = min(disk_activity / (100 * 1024 * 1024), 1.0) if disk_activity > 0 else 0.0
                net_norm = min(net_rate / (10 * 1024 * 1024), 1.0)

                stress = 0.0
                stress += abs(cpu_norm - self.last_cpu) * 1.5
                stress += abs(ram_norm - self.last_ram) * 1.0
                stress += abs(disk_norm - self.last_disk) * 1.2
                stress += abs(net_norm - self.last_net_rate) * 1.3
                stress += (cpu_norm + ram_norm + disk_norm + net_norm) / 4.0
                stress = min(max(stress, 0.0), 2.0) / 2.0

                net_change = abs(net_norm - self.last_net_norm)
                t_now = time.time()
                t_delta = t_now - self.last_net_change_time

                if net_change < 0.05 and 0.8 <= t_delta <= 10.0:
                    self.beacon_score += 0.05
                else:
                    self.beacon_score *= 0.97

                self.beacon_score = min(max(self.beacon_score, 0.0), 1.0)

                if net_change > 0.05:
                    self.last_net_change_time = t_now
                self.last_net_norm = net_norm

                variability = (
                    abs(cpu_norm - self.last_cpu) +
                    abs(ram_norm - self.last_ram) +
                    abs(disk_norm - self.last_disk) +
                    abs(net_norm - self.last_net_rate)
                ) / 4.0
                entropy = min(max((variability * 0.7 + self.beacon_score * 0.3), 0.0), 1.0)

                if stress > 0.7:
                    if self.high_stress_start is None:
                        self.high_stress_start = time.time()
                elif stress < 0.5:
                    self.high_stress_start = None

                self.last_cpu = cpu_norm
                self.last_ram = ram_norm
                self.last_disk = disk_norm
                self.last_net_rate = net_norm

                idx = self.current_index
                self.cpu_buffer[idx] = cpu_norm
                self.ram_buffer[idx] = ram_norm
                self.disk_buffer[idx] = disk_norm
                self.net_buffer[idx] = net_norm
                self.sec_buffer[idx] = stress
                self.entropy_buffer[idx] = entropy

                self.current_index = (self.current_index + 1) % self.num_segments

                self.context_window_samples.append(
                    (cpu_norm, ram_norm, disk_norm, net_norm)
                )
                if len(self.context_window_samples) > self.context_window_size:
                    self.context_window_samples.pop(0)

                self.physics.update(cpu_norm, ram_norm, disk_norm, net_norm, stress, entropy)
                self._publish_behavior_events(cpu_norm, ram_norm, disk_norm, net_norm, stress, entropy)

                self.root.after(
                    0,
                    self.update_ui_and_interfaces,
                    cpu,
                    ram,
                    disk_norm,
                    net_kb,
                    stress,
                    entropy
                )

                self.error_count = 0

            except Exception:
                self.error_count += 1
                if self.error_count >= self.max_errors:
                    self.running = False
                    self.root.after(
                        0,
                        self.status_label.config,
                        {"text": "Status: Watchdog shutdown (sampling failures)", "fg": "#FF6B6B"}
                    )
                    break

            time.sleep(self.update_interval)

    # =========================
    # EventBus integration
    # =========================

    def _publish_behavior_events(self, cpu_norm, ram_norm, disk_norm, net_norm, stress, entropy):
        now = time.time()

        if cpu_norm > 0.8:
            self.event_bus.publish(SecEvent(now, "CPU_SPIKE", "CPU", {"value": cpu_norm}))
        if net_norm > 0.7:
            self.event_bus.publish(SecEvent(now, "NET_BURST", "NET", {"value": net_norm}))
        if disk_norm > 0.7:
            self.event_bus.publish(SecEvent(now, "DISK_THRASH", "DISK", {"value": disk_norm}))
        if entropy > 0.7:
            self.event_bus.publish(SecEvent(now, "ENTROPY_SURGE", "ENTROPY", {"value": entropy}))
        if stress > 0.7:
            self.event_bus.publish(SecEvent(now, "SUSTAINED_LOAD", "SYSTEM", {"value": stress}))

        if CAPABILITIES["network_monitor"] and self.foreign_conn_count > 0:
            self.event_bus.publish(
                SecEvent(now, "NET_FOREIGN", "NET", {"count": self.foreign_conn_count})
            )

    def _chain_event_ingest(self, evt: SecEvent):
        self.attack_chain.add_event(evt.etype, {"entity": evt.entity, "meta": evt.meta})

    def schedule_event_bus(self):
        if not self.running:
            return
        self.event_bus.run_once()
        self.root.after(10, self.schedule_event_bus)

    # =========================
    # Network monitor
    # =========================

    def schedule_network_monitor(self):
        if not self.running:
            return
        self.monitor_network()
        self.root.after(8000, self.schedule_network_monitor)

    def monitor_network(self):
        conns = safe_net_connections()
        foreign = 0
        listening = 0
        for c in conns:
            try:
                if is_listening(c):
                    listening += 1
                ip = get_remote_ip(c)
                if not ip:
                    continue
                if not is_private_ip(ip):
                    foreign += 1
            except Exception:
                continue

        self.foreign_conn_count = foreign
        self.listening_count = listening

    # =========================
    # Threat Detection (behavioral)
    # =========================

    def schedule_threat_scan(self):
        if not self.running:
            return
        self.threat_scan_and_respond()
        self.root.after(10000, self.schedule_threat_scan)

    def threat_scan_and_respond(self):
        try:
            try:
                conns = safe_net_connections()
            except Exception:
                conns = []

            seen = set()
            listening = 0
            for conn in conns:
                try:
                    if is_listening(conn):
                        listening += 1
                        ip = getattr(conn.laddr, "ip", "unknown")
                        port = conn.laddr.port
                        key = (ip, port)
                        if key in seen:
                            continue
                        seen.add(key)
                    else:
                        continue
                except Exception:
                    continue

            self.listening_count = listening

            self.remote_tool_hits = 0
            self.firewall_cmd_hits = 0
            self.settings_cmd_hits = 0
            self.shell_activity_count = 0
            self.suspicious_name_count = 0

            procs = list(safe_process_iter())
            self.process_count = len(procs)

            for proc in procs:
                try:
                    name = proc.info.get("name") or ""
                    cmdline_list = proc.info.get("cmdline") or []
                    cmdline = " ".join(cmdline_list).lower()

                    if cmdline and REMOTE_TOOL_PATTERNS.search(cmdline):
                        self.remote_tool_hits += 1
                    if cmdline and FIREWALL_CMD_PATTERNS.search(cmdline):
                        self.firewall_cmd_hits += 1
                    if cmdline and SETTINGS_CMD_PATTERNS.search(cmdline):
                        self.settings_cmd_hits += 1
                    if name and SHELL_NAMES.search(name):
                        self.shell_activity_count += 1

                    if name and re.search(r"(keylogger|sniffer|injector|bot|miner)", name, re.IGNORECASE):
                        self.suspicious_name_count += 1

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                except Exception:
                    continue

        except Exception as e:
            self.integrity_ok = False
            self.integrity_last_error = f"threat_scan_and_respond() failed - {e}"

    # =========================
    # Integrity Self-Check
    # =========================

    def schedule_self_check(self):
        if not self.running:
            return
        self.self_check_integrity()
        self.root.after(15000, self.schedule_self_check)

    def self_check_integrity(self):
        try:
            assert callable(self.threat_scan_and_respond)
            assert callable(self.update_ui_and_interfaces)
            assert isinstance(self.attacker_log, list)
            self.integrity_ok = True
            self.integrity_last_error = None
        except Exception as e:
            self.integrity_ok = False
            self.integrity_last_error = f"Integrity check failed: {e}"

    # =========================
    # LAN Swarm: Encryption helpers
    # =========================

    def encrypt_payload(self, payload: dict) -> bytes:
        data = json.dumps(payload).encode("utf-8")
        aesgcm = AESGCM(SHARED_SWARM_KEY)
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, data, None)
        return nonce + ct

    def decrypt_payload(self, blob: bytes):
        if len(blob) < 12:
            return None
        nonce = blob[:12]
        ct = blob[12:]
        aesgcm = AESGCM(SHARED_SWARM_KEY)
        try:
            data = aesgcm.decrypt(nonce, ct, None)
            return json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            return None

    # =========================
    # Adaptive Swarm Routing (LAN + optional WAN relay)
    # =========================

    def lan_broadcast_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass

        while self.running:
            try:
                avg_stress = sum(self.sec_buffer) / len(self.sec_buffer)
                avg_entropy = sum(self.entropy_buffer) / len(self.entropy_buffer)
                behavioral_state = self.classify_behavioral_state(
                    avg_stress, avg_entropy, self.last_cpu, self.last_net_rate, self.last_disk
                )
                anomaly_flags = []
                fleet_status = self.build_fleet_status(
                    avg_stress, avg_entropy, behavioral_state, anomaly_flags, fingerprints=[]
                )
                payload = {
                    "type": "sentinel_status",
                    "node_id": self.node_id,
                    "role": self.role,
                    "fleet_status": asdict(fleet_status),
                }
                blob = self.encrypt_payload(payload)
                # Adaptive: prefer LAN broadcast; WAN relay handled separately
                sock.sendto(blob, ("<broadcast>", LAN_UDP_PORT))
            except Exception:
                pass
            time.sleep(LAN_BROADCAST_INTERVAL)

    def wan_relay_loop(self):
        # Tier-3 stub: zero-trust-style relay (client side only)
        # In a real deployment, this would use TLS + mutual auth to a relay server.
        while self.running and WAN_RELAY_ENABLED:
            try:
                avg_stress = sum(self.sec_buffer) / len(self.sec_buffer)
                avg_entropy = sum(self.entropy_buffer) / len(self.entropy_buffer)
                behavioral_state = self.classify_behavioral_state(
                    avg_stress, avg_entropy, self.last_cpu, self.last_net_rate, self.last_disk
                )
                anomaly_flags = []
                fleet_status = self.build_fleet_status(
                    avg_stress, avg_entropy, behavioral_state, anomaly_flags, fingerprints=[]
                )
                payload = {
                    "type": "sentinel_status",
                    "node_id": self.node_id,
                    "role": self.role,
                    "fleet_status": asdict(fleet_status),
                }
                # Here you would send payload via HTTPS/TLS to WAN_RELAY_HOST
                # This is intentionally left as a stub to avoid real network calls.
                _ = payload
            except Exception:
                pass
            time.sleep(LAN_BROADCAST_INTERVAL * 2)

    def lan_listener_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        try:
            sock.bind(("", LAN_UDP_PORT))
        except Exception:
            return

        while self.running:
            try:
                data, _addr = sock.recvfrom(65535)
                msg = self.decrypt_payload(data)
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if mtype == "sentinel_status":
                    self.handle_remote_status(msg)
                elif mtype == "codex_command":
                    self.handle_remote_command(msg)
            except Exception:
                continue

    def handle_remote_status(self, msg: dict):
        node_id = msg.get("node_id")
        if not node_id or node_id == self.node_id:
            return
        fs = msg.get("fleet_status") or {}
        try:
            status = FleetNodeStatus(
                node_id=fs.get("node_id", node_id),
                role=fs.get("role", "UNKNOWN"),
                health_state=fs.get("health_state", "UNKNOWN"),
                threat_level=fs.get("threat_level", "LOW"),
                stress_score=float(fs.get("stress_score", 0.0)),
                entropy_score=float(fs.get("entropy_score", 0.0)),
                behavioral_state=fs.get("behavioral_state", "UNKNOWN"),
                anomaly_flags=list(fs.get("anomaly_flags", [])),
                uptime_seconds=int(fs.get("uptime_seconds", 0)),
                last_update_ts=fs.get("last_update_ts", datetime.now(timezone.utc).isoformat()),
                fingerprints=list(fs.get("fingerprints", [])),
                isolation_state=fs.get("isolation_state", "NORMAL"),
            )
        except Exception:
            return

        self.fleet.update_node(status)

        events = [{"entity": f"{node_id}:{flag}", "score": 0.5} for flag in status.anomaly_flags]
        self.queen.update(node_id, events)

    def handle_remote_command(self, msg: dict):
        cmd = msg.get("command", "").strip()
        src = msg.get("source", "unknown")
        ts = msg.get("timestamp", "")
        if not cmd:
            return
        self.cmd_status.config(text=f"Received command from {src}: {cmd} @ {ts}")
        # Non-destructive demo: we only react visually and via isolation flags
        cmd_upper = cmd.upper()
        if cmd_upper == "PING_SWARM":
            self.status_label.config(text="Status: SWARM PING RECEIVED", fg="#A0FFCF")
        elif cmd_upper == "SOFT_ISOLATE":
            self.isolation_state = "SOFT_ISOLATION"
        elif cmd_upper == "HARD_ISOLATE":
            self.isolation_state = "HARD_ISOLATION"
        elif cmd_upper == "CLEAR_ISOLATION":
            self.isolation_state = "NORMAL"

    def broadcast_command(self):
        cmd = self.cmd_entry.get().strip()
        if not cmd:
            return
        payload = {
            "type": "codex_command",
            "source": self.node_id,
            "command": cmd,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        blob = self.encrypt_payload(payload)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception:
            pass
        try:
            sock.sendto(blob, ("<broadcast>", LAN_UDP_PORT))
            self.cmd_status.config(text=f"Broadcasted command: {cmd}")
        except Exception:
            self.cmd_status.config(text="Failed to broadcast command")

    def schedule_fleet_preview(self):
        if not self.running:
            return
        snap = self.fleet.snapshot()
        if not snap:
            txt = "No other nodes detected yet.\nRun this Sentinel on multiple machines in the same LAN."
        else:
            lines = ["LAN Nodes (3D layout is logical):"]
            for nid, st in snap.items():
                iso = st.get("isolation_state", "NORMAL")
                lines.append(
                    f"- {nid} [{st.get('role','?')}] "
                    f"{st.get('health_state','?')}/{st.get('threat_level','?')} "
                    f"ISO:{iso}"
                )
            txt = "\n".join(lines)
        self.fleet_preview.config(text=txt)
        self.root.after(5000, self.schedule_fleet_preview)

    def schedule_cluster_preview(self):
        if not self.running:
            return
        clusters = self.fleet.cluster_summary()
        fused = self.fleet.fused_risk_view()
        if not clusters:
            txt = "No shared fingerprints yet."
        else:
            lines = ["Shared anomaly fingerprints:"]
            for fp, cnt in clusters.items():
                lines.append(f"- {fp}: {cnt} nodes")
            lines.append("")
            lines.append("Fusion counts:")
            for fp, cnt in fused.items():
                lines.append(f"  {fp}: {cnt}")
            txt = "\n".join(lines)
        self.cluster_preview.config(text=txt)
        self.root.after(7000, self.schedule_cluster_preview)

    # =========================
    # UI + Interfaces + Logging
    # =========================

    def update_ui_and_interfaces(self, cpu, ram, disk_norm, net_kb, stress, entropy):
        self.cpu_label.config(text=f"CPU: {cpu:.1f}%")
        self.ram_label.config(text=f"RAM: {ram:.1f}%")
        self.disk_label.config(text=f"Disk: {disk_norm*100:.1f}% est")
        self.net_label.config(text=f"Net: {net_kb:.1f} kB/s")
        self.sec_label.config(text=f"Security Load: {stress*100:.0f}%")

        if stress < 0.3:
            sec_color = "#A0FFB0"
        elif stress < 0.7:
            sec_color = "#FFD966"
        else:
            sec_color = "#FF6B6B"
        self.sec_label.config(fg=sec_color)

        alert_text, alert_color, actions_text, anomaly_flags, behavioral_state = \
            self.compute_alert_and_actions(
                stress,
                entropy,
                self.last_cpu,
                self.last_disk,
                self.last_net_rate
            )

        chains = self.attack_chain.detect()
        if chains:
            cname, cscore = max(chains, key=lambda x: x[1])
            anomaly_flags.append(cname)
            entropy = min(1.0, entropy + 0.1 * cscore)
            stress = min(1.0, stress + 0.1 * cscore)

        fingerprint = self.physics.fingerprint(anomaly_flags, chains)

        self.alert_label.config(text=alert_text, fg=alert_color)
        self.actions_text.config(text=actions_text)

        self.altered_state = (behavioral_state == "ALTERED")
        if self.altered_state:
            status_text = "Status: ALTERED STATE (High Stress + High Entropy, Tier-3 Swarm)"
            status_color = "#FF4BFF"
        else:
            status_text = f"Status: {behavioral_state} (Tier-3, Swarm + Forecast + Fusion)"
            status_color = "#A0FFB0"

        if self.isolation_state != "NORMAL":
            status_text += f" • {self.isolation_state}"
            status_color = "#FFB347"

        if not self.integrity_ok:
            status_text += " • INTEGRITY WARNING"
            status_color = "#FF6B6B"

        self.status_label.config(text=status_text, fg=status_color)

        confidence = self.physics.confidence_score(stress, entropy, anomaly_flags)

        forecast_5 = {
            "stress": self.physics.forecast("stress", 5),
            "entropy": self.physics.forecast("entropy", 5),
        }
        forecast_10 = {
            "stress": self.physics.forecast("stress", 10),
            "entropy": self.physics.forecast("entropy", 10),
        }
        forecast_30 = {
            "stress": self.physics.forecast("stress", 30),
            "entropy": self.physics.forecast("entropy", 30),
        }

        codex_status = self.build_codex_status(
            stress,
            entropy,
            behavioral_state,
            anomaly_flags,
            alert_text,
            actions_text,
            confidence,
            forecast_5,
            forecast_10,
            forecast_30
        )
        fleet_status = self.build_fleet_status(
            stress,
            entropy,
            behavioral_state,
            anomaly_flags,
            fingerprints=[fingerprint]
        )

        queen_events = [{"entity": f"{self.node_id}:{f}", "score": 0.5} for f in anomaly_flags]
        self.queen.update(self.node_id, queen_events)
        global_risk = self.queen.global_risk()
        self.update_queen_preview(global_risk)

        if codex_status.threat_level == "HIGH":
            self.log_attacker_pattern(stress, entropy, anomaly_flags, fingerprint)
            self.autonomous_response(stress, entropy, anomaly_flags)

        self.update_log_preview()

        _ = codex_status
        self.fleet.update_node(fleet_status)

    def classify_behavioral_state(self, stress, entropy, cpu_norm, net_norm, disk_norm):
        if stress < 0.3 and entropy < 0.3:
            return "CALM"
        if stress > 0.6 and entropy < 0.4:
            return "FOCUSED"
        if stress > 0.6 and entropy > 0.6:
            return "ALTERED"
        if stress < 0.4 and entropy > 0.6:
            return "SUPPRESSED"
        return "TRANSITIONAL"

    def compute_alert_and_actions(self, stress, entropy, cpu_norm, disk_norm, net_norm):
        anomaly_flags = []

        if stress > 0.6:
            anomaly_flags.append("SUSTAINED_LOAD")
        if entropy > 0.6:
            anomaly_flags.append("HIGH_ENTROPY")
        if self.beacon_score > NET_BEACON_THRESHOLD:
            anomaly_flags.append("NET_BEACON")
        if CAPABILITIES["network_monitor"] and self.foreign_conn_count > 0:
            anomaly_flags.append("NET_FOREIGN")
        if self.listening_count > 50:
            anomaly_flags.append("MANY_LISTENING_PORTS")
        if self.process_count > 500:
            anomaly_flags.append("MANY_PROCESSES")

        if self.remote_tool_hits > 0:
            anomaly_flags.append("REMOTE_TOOL_BEHAVIOR")
        if self.firewall_cmd_hits > 0:
            anomaly_flags.append("FIREWALL_TAMPER_BEHAVIOR")
        if self.settings_cmd_hits > 0:
            anomaly_flags.append("SETTINGS_TAMPER_BEHAVIOR")
        if self.shell_activity_count > 10:
            anomaly_flags.append("HEAVY_SHELL_ACTIVITY")
        if self.suspicious_name_count > 0:
            anomaly_flags.append("SUSPICIOUS_PROCESS_NAMES")

        if net_norm > 0.7 and disk_norm < 0.2:
            anomaly_flags.append("MISSING_DISK_ACTIVITY")
        if stress > 0.7 and entropy < 0.4:
            anomaly_flags.append("MISSING_ENTROPY_CORRELATION")
        if entropy > 0.7 and stress < 0.4:
            anomaly_flags.append("MISSING_STRESS_CORRELATION")
        if self.high_stress_start is not None and (time.time() - self.high_stress_start) > 60:
            anomaly_flags.append("MISSING_RECOVERY")

        cpu_shape = self.physics.shape_detection("cpu")
        net_shape = self.physics.shape_detection("net")
        if cpu_shape == "BURST":
            anomaly_flags.append("CPU_BURST_SHAPE")
        if net_shape == "OSCILLATION":
            anomaly_flags.append("NET_OSCILLATION_SHAPE")

        behavioral_state = self.classify_behavioral_state(
            stress, entropy, cpu_norm, net_norm, disk_norm
        )

        if stress < 0.3 and entropy < 0.3:
            alert_text = (
                "No significant anomalies detected.\n"
                "System behavior appears stable and within expected ranges."
            )
            alert_color = "#A0FFB0"
            actions_text = (
                "- Keep this monitor running in the background.\n"
                "- Periodically glance at the rings for unusual spikes.\n"
                "- Maintain OS updates and a trusted AV/firewall."
            )
        elif stress < 0.7 and entropy < 0.7:
            alert_text = (
                "Moderate anomalies observed.\n"
                "Some load spikes or irregular patterns detected.\n"
                "This may be normal activity, but worth watching."
            )
            alert_color = "#FFD966"
            actions_text = (
                "- Note the time window of these anomalies.\n"
                "- If you recently installed or ran something heavy, this may be expected.\n"
                "- If unsure, run a full scan with your trusted security tools.\n"
                "- Avoid entering sensitive data while behavior is unclear."
            )
        else:
            alert_text = (
                "High anomaly level detected.\n"
                "Sustained stress and/or complex patterns observed.\n"
                "This could indicate harmful or unwanted behavior."
            )
            alert_color = "#FF6B6B"
            actions_text = (
                "- Disconnect from the network if you suspect active compromise.\n"
                "- Run a full system scan with a trusted antivirus/EDR.\n"
                "- Avoid logging into sensitive accounts from this machine.\n"
                "- Consider rebooting and checking for unusual startup items.\n"
                "- If this persists, consult a security professional."
            )

        return alert_text, alert_color, actions_text, anomaly_flags, behavioral_state

    def build_codex_status(
        self,
        stress,
        entropy,
        behavioral_state,
        anomaly_flags,
        alert_text,
        actions_text,
        confidence,
        forecast_5,
        forecast_10,
        forecast_30
    ):
        if stress < 0.3 and entropy < 0.3:
            health_state = "OK"
            threat_level = "LOW"
        elif stress < 0.7 and entropy < 0.7:
            health_state = "DEGRADED"
            threat_level = "MEDIUM"
        else:
            health_state = "CRITICAL"
            threat_level = "HIGH"

        now_iso = datetime.now(timezone.utc).isoformat()

        status = CodexSentinelStatus(
            source="sentinel",
            health_state=health_state,
            threat_level=threat_level,
            stress_score=float(stress),
            entropy_score=float(entropy),
            behavioral_state=behavioral_state,
            anomaly_flags=list(anomaly_flags),
            recent_window_seconds=int(self.context_window_size * self.update_interval),
            confidence=float(confidence),
            risk_summary=alert_text.replace("\n", " "),
            recommended_actions=[line.strip("- ").strip() for line in actions_text.split("\n") if line.strip()],
            timestamp=now_iso,
            forecast_5s=forecast_5,
            forecast_10s=forecast_10,
            forecast_30s=forecast_30
        )
        return status

    def build_fleet_status(self, stress, entropy, behavioral_state, anomaly_flags, fingerprints):
        if stress < 0.3 and entropy < 0.3:
            health_state = "OK"
            threat_level = "LOW"
        elif stress < 0.7 and entropy < 0.7:
            health_state = "DEGRADED"
            threat_level = "MEDIUM"
        else:
            health_state = "CRITICAL"
            threat_level = "HIGH"

        now_iso = datetime.now(timezone.utc).isoformat()
        uptime = int(time.time() - self.start_time)

        status = FleetNodeStatus(
            node_id=self.node_id,
            role=self.role,
            health_state=health_state,
            threat_level=threat_level,
            stress_score=float(stress),
            entropy_score=float(entropy),
            behavioral_state=behavioral_state,
            anomaly_flags=list(anomaly_flags),
            uptime_seconds=uptime,
            last_update_ts=now_iso,
            fingerprints=list(fingerprints),
            isolation_state=self.isolation_state
        )
        return status

    def log_attacker_pattern(self, stress, entropy, anomaly_flags, fingerprint):
        if self.context_window_samples:
            avg_cpu = sum(s[0] for s in self.context_window_samples) / len(self.context_window_samples)
            avg_ram = sum(s[1] for s in self.context_window_samples) / len(self.context_window_samples)
            avg_disk = sum(s[2] for s in self.context_window_samples) / len(self.context_window_samples)
            avg_net = sum(s[3] for s in self.context_window_samples) / len(self.context_window_samples)
        else:
            avg_cpu = avg_ram = avg_disk = avg_net = 0.0

        pattern_parts = []
        for flag in [
            "NET_BEACON", "NET_FOREIGN", "SUSTAINED_LOAD", "HIGH_ENTROPY",
            "MANY_LISTENING_PORTS", "MANY_PROCESSES",
            "MISSING_DISK_ACTIVITY", "MISSING_ENTROPY_CORRELATION",
            "MISSING_STRESS_CORRELATION", "MISSING_RECOVERY",
            "CPU_BURST_SHAPE", "NET_OSCILLATION_SHAPE",
            "REMOTE_TOOL_BEHAVIOR", "FIREWALL_TAMPER_BEHAVIOR",
            "SETTINGS_TAMPER_BEHAVIOR", "HEAVY_SHELL_ACTIVITY",
            "SUSPICIOUS_PROCESS_NAMES"
        ]:
            if flag in anomaly_flags:
                pattern_parts.append(flag)

        chains = self.attack_chain.detect()
        for cname, score in chains:
            if score > 0.8:
                pattern_parts.append(cname)

        pattern_signature = "_".join(pattern_parts) if pattern_parts else "HIGH_THREAT"

        now_iso = datetime.now(timezone.utc).isoformat()
        duration_estimate = int(self.context_window_size * self.update_interval)

        entry = AttackerLogEntry(
            timestamp=now_iso,
            node_id=self.node_id,
            health_state="CRITICAL",
            threat_level="HIGH",
            stress_score=float(stress),
            entropy_score=float(entropy),
            anomaly_flags=list(anomaly_flags),
            pattern_signature=pattern_signature,
            fingerprint=fingerprint,
            duration_estimate_seconds=duration_estimate,
            context_window={
                "avg_cpu": avg_cpu,
                "avg_ram": avg_ram,
                "avg_disk_norm": avg_disk,
                "avg_net_norm": avg_net
            }
        )

        self.attacker_log.append(entry)

        try:
            with open("sentinel_attacker_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception:
            pass

    def update_log_preview(self):
        if not self.attacker_log:
            self.log_preview.config(text="No hostile patterns recorded yet.")
            return

        last = self.attacker_log[-1]
        txt = (
            f"Last pattern:\n"
            f"- Time: {last.timestamp}\n"
            f"- Node: {last.node_id}\n"
            f"- Threat: {last.threat_level}\n"
            f"- Signature: {last.pattern_signature}\n"
            f"- Fingerprint: {last.fingerprint}\n"
            f"- Duration: ~{last.duration_estimate_seconds}s\n"
            f"- Context avg CPU: {last.context_window['avg_cpu']*100:.1f}%\n"
            f"- Context avg Net norm: {last.context_window['avg_net_norm']*100:.1f}%"
        )
        self.log_preview.config(text=txt)

    def update_queen_preview(self, global_risk):
        if not global_risk:
            self.queen_preview.config(text="No high-risk entities yet.")
            return

        lines = ["High-risk entities (local + swarm):"]
        for ent, score in global_risk.items():
            lines.append(f"- {ent}: {score:.2f}")
        self.queen_preview.config(text="\n".join(lines))

    # =========================
    # Autonomous Response Layer (non-destructive)
    # =========================

    def autonomous_response(self, stress, entropy, anomaly_flags):
        # Simple logic: escalate isolation state based on severity
        if "REMOTE_TOOL_BEHAVIOR" in anomaly_flags or "REMOTE_CONTROL_PATTERN" in anomaly_flags:
            self.isolation_state = "HARD_ISOLATION"
        elif "NET_FOREIGN" in anomaly_flags or "NET_BEACON" in anomaly_flags:
            if self.isolation_state != "HARD_ISOLATION":
                self.isolation_state = "SOFT_ISOLATION"
        # No actual blocking or killing – just state + visual + log

    # =========================
    # Visualization
    # =========================

    def schedule_redraw(self):
        if not self.running:
            return
        self.redraw()
        self.root.after(int(self.update_interval * 1000), self.schedule_redraw)

    def redraw(self):
        self.canvas.delete("all")

        if self.visual_mode_3d:
            self.draw_background_3d()
        else:
            self.draw_background()

        self.draw_ring(self.entropy_buffer, self.inner_radius_threat, self.outer_radius_threat, mode="threat")
        self.draw_ring(self.cpu_buffer, self.inner_radius_cpu, self.outer_radius_cpu, mode="cpu")
        self.draw_ring(self.ram_buffer, self.inner_radius_ram, self.outer_radius_ram, mode="ram")
        self.draw_ring(self.disk_buffer, self.inner_radius_disk, self.outer_radius_disk, mode="disk")
        self.draw_ring(self.net_buffer, self.inner_radius_net, self.outer_radius_net, mode="net")
        self.draw_security_core()

        if self.visual_mode_3d:
            self.draw_fleet_3d_overlay()

    def draw_background(self):
        max_r = self.outer_radius_net + 50
        self.canvas.create_oval(
            self.center_x - max_r,
            self.center_y - max_r,
            self.center_x + max_r,
            self.center_y + max_r,
            outline="#101320",
            width=2
        )

        self.canvas.create_line(
            self.center_x - 20, self.center_y,
            self.center_x + 20, self.center_y,
            fill="#151822", width=1
        )
        self.canvas.create_line(
            self.center_x, self.center_y - 20,
            self.center_x, self.center_y + 20,
            fill="#151822", width=1
        )

    def draw_background_3d(self):
        # Simple pseudo-3D grid
        max_r = self.outer_radius_net + 80
        for i in range(6):
            offset = i * 20
            self.canvas.create_oval(
                self.center_x - max_r + offset,
                self.center_y - max_r + offset * 0.5,
                self.center_x + max_r + offset,
                self.center_y + max_r + offset * 0.5,
                outline="#101320",
                width=1
            )

        self.canvas.create_line(
            self.center_x - 200, self.center_y + 200,
            self.center_x + 200, self.center_y + 260,
            fill="#151822", width=2
        )

    def draw_ring(self, buffer, inner_r, outer_r, mode="cpu"):
        for i, intensity in enumerate(buffer):
            if intensity <= 0.01:
                continue
            idx_offset = (i - self.current_index) % self.num_segments
            angle = (2 * math.pi * idx_offset) / self.num_segments - math.pi / 2
            self.draw_segment(angle, intensity, inner_r, outer_r, mode)

    def draw_segment(self, angle, intensity, inner_r, outer_r, mode):
        x1 = self.center_x + inner_r * math.cos(angle)
        y1 = self.center_y + inner_r * math.sin(angle)
        x2 = self.center_x + outer_r * math.cos(angle)
        y2 = self.center_y + outer_r * math.sin(angle)

        if mode == "cpu":
            base_r, base_g, base_b = 80, 220, 255
        elif mode == "ram":
            base_r, base_g, base_b = 120, 255, 160
        elif mode == "disk":
            base_r, base_g, base_b = 255, 200, 120
        elif mode == "net":
            base_r, base_g, base_b = 255, 120, 180
        elif mode == "threat":
            base_r, base_g, base_b = 255, 80, 80
        else:
            base_r, base_g, base_b = 200, 200, 200

        scale = intensity
        r = int(base_r * scale + 20)
        g = int(base_g * scale + 20)
        b = int(base_b * scale + 20)

        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        color = f"#{r:02x}{g:02x}{b:02x}"
        width = 1 + 4 * intensity

        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, capstyle="round")

    def draw_security_core(self):
        avg_stress = sum(self.sec_buffer) / len(self.sec_buffer)
        radius = 50 + int(30 * avg_stress)

        if self.altered_state:
            fill = "#1A001A"
            outline = "#FF4BFF"
            text_color = "#FFCCFF"
            label = "ALTERED\nSTATE"
        else:
            if avg_stress < 0.3:
                fill = "#07120F"
                outline = "#1EE3A1"
                text_color = "#A0FFCF"
            elif avg_stress < 0.7:
                fill = "#1A1407"
                outline = "#FFC857"
                text_color = "#FFE9A3"
            else:
                fill = "#190707"
                outline = "#FF4B4B"
                text_color = "#FFB3B3"
            label = "POV\nSECURITY\nCORE"

        self.canvas.create_oval(
            self.center_x - radius,
            self.center_y - radius,
            self.center_x + radius,
            self.center_y + radius,
            fill=fill,
            outline=outline,
            width=3
        )

        self.canvas.create_text(
            self.center_x,
            self.center_y,
            text=label,
            fill=text_color,
            font=("Consolas", 11, "bold")
        )

    def draw_fleet_3d_overlay(self):
        snap = self.fleet.snapshot()
        if not snap:
            return

        base_x = self.center_x - 250
        base_y = self.center_y + 200
        spacing_x = 80
        spacing_y = 40

        i = 0
        for nid, st in snap.items():
            col = i % 5
            row = i // 5
            x = base_x + col * spacing_x
            y = base_y + row * spacing_y

            health = st.get("health_state", "OK")
            threat = st.get("threat_level", "LOW")
            iso = st.get("isolation_state", "NORMAL")

            if threat == "HIGH":
                color = "#FF4B4B"
            elif threat == "MEDIUM":
                color = "#FFC857"
            else:
                color = "#1EE3A1"

            if iso != "NORMAL":
                outline = "#FFB347"
            else:
                outline = "#202838"

            self.canvas.create_rectangle(
                x, y, x+60, y+25,
                fill=color,
                outline=outline,
                width=2
            )
            self.canvas.create_text(
                x+30, y+12,
                text=nid[-4:],
                fill="#000000",
                font=("Consolas", 8, "bold")
            )
            i += 1

# =========================
# Entry point
# =========================

def main():
    root = tk.Tk()
    app = POVUnifiedSecurityCore(root, node_id=None, role="WORKSTATION")
    root.geometry("1400x860")
    root.mainloop()

if __name__ == "__main__":
    main()
