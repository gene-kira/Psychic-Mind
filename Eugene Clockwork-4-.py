#!/usr/bin/env python3
"""
POV Unified Autonomous Privacy & Security Monitor
+ Predictive Consensus Engine (Queen)
+ Behavioral Attack Chain Engine (abstract, no IPs/PIDs)
+ Privacy-respecting Attacker Log
+ Codex–Sentinel + Fleet Interface Schemas

SAFE DESIGN:
- No IP addresses, MAC addresses, usernames, hostnames, process names, PIDs, file paths.
- No content inspection, no keystrokes, no biometrics.
- No destructive actions (no killing processes, no blocking traffic, no firewall changes).
- Optional JSONL log is behavior-only (patterns, scores, anomaly types).

CONCEPTUAL ADD-ONS:
- Queen: global consensus engine over anomaly types.
- AttackChainEngine: detects chains of abstract behavioral events.
- Bernoulli-style risk: repeated anomalies increase probability of hostile behavior.
- "Altered States": special mode when system enters rare, high-entropy, high-stress regimes.
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
from collections import deque

# =========================
# Auto-loader for libraries
# =========================

REQUIRED_PACKAGES = ["psutil"]

def ensure_package(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[AUTOLOADER] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

psutil = ensure_package("psutil")

# =========================
# Codex–Sentinel / Fleet / Log Schemas
# =========================

@dataclass
class CodexSentinelStatus:
    source: str
    health_state: str          # "OK" | "DEGRADED" | "CRITICAL"
    threat_level: str          # "LOW" | "MEDIUM" | "HIGH"
    stress_score: float        # 0.0–1.0
    entropy_score: float       # 0.0–1.0
    anomaly_flags: list        # e.g. ["NET_BEACON", "SUSTAINED_LOAD"]
    recent_window_seconds: int
    confidence: float          # 0.0–1.0
    risk_summary: str
    recommended_actions: list
    timestamp: str             # ISO 8601 UTC


@dataclass
class FleetNodeStatus:
    node_id: str
    role: str                  # "DRONE" | "CAR" | "SERVER" | ...
    health_state: str
    threat_level: str
    stress_score: float
    entropy_score: float
    anomaly_flags: list
    uptime_seconds: int
    last_update_ts: str        # ISO 8601 UTC


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
    duration_estimate_seconds: int
    context_window: dict       # {avg_cpu, avg_ram, avg_disk_norm, avg_net_norm}


# =========================
# Predictive / Consensus Engines
# =========================

class Queen:
    """
    REAL-TIME QUEEN (CONSENSUS ENGINE)
    Aggregates anomaly events (abstract entities like "NET_BEACON", "SUSTAINED_LOAD")
    across nodes and computes a global risk weight per entity.
    """
    def __init__(self):
        self.nodes = {}  # node_id -> list of events

    def update(self, node_id, events):
        # events: list of {"entity": str, "score": float}
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

        # Simple threshold: entities with cumulative risk > 1.5
        return {k: v for k, v in risk.items() if v > 1.5}


class AttackChainEngine:
    """
    Behavioral Attack Chain Engine (abstract):
    Works on high-level event types like:
    - "CPU_SPIKE"
    - "NET_BURST"
    - "DISK_THRASH"
    - "ENTROPY_SURGE"
    - "SUSTAINED_LOAD"
    No processes, IPs, or file paths.
    """
    def __init__(self, window=120):
        self.events = deque()
        self.window = window  # seconds

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

        # Example abstract chains:
        if all(x in types for x in ["CPU_SPIKE", "NET_BURST", "ENTROPY_SURGE"]):
            chains.append(("FULL_BEHAVIOR_CHAIN", 0.9))

        if types.count("CPU_SPIKE") > 5 and "NET_BURST" in types:
            chains.append(("RESOURCE_STORM", 0.8))

        if "DISK_THRASH" in types and "NET_BURST" in types:
            chains.append(("PERSISTENCE_EXFIL_PATTERN", 0.85))

        return chains


class EventBus:
    """
    Simple in-process event bus for behavioral events.
    """
    def __init__(self):
        self.subscribers = []
        self.queue = deque()

    def publish(self, event):
        self.queue.append(event)

    def subscribe(self, fn):
        self.subscribers.append(fn)

    def run_once(self):
        # Non-blocking single-step run for integration into GUI loop
        if self.queue:
            evt = self.queue.popleft()
            for fn in self.subscribers:
                fn(evt)


@dataclass
class SecEvent:
    ts: float
    etype: str      # e.g. "CPU_SPIKE", "NET_BURST"
    entity: str     # abstract entity, e.g. "CPU", "NET", "ENTROPY"
    meta: dict


# =========================
# GUI + POV visualizer
# =========================

import tkinter as tk

PRIVACY_HARDENED = True  # explicit flag


class POVUnifiedSecurityCore:
    def __init__(self, root, node_id="NODE_LOCAL", role="WORKSTATION"):
        self.root = root
        self.root.title("POV Unified Autonomous Privacy & Security Monitor")

        self.node_id = node_id
        self.role = role
        self.start_time = time.time()

        # In-memory attacker log (privacy-respecting)
        self.attacker_log = []  # list[AttackerLogEntry]

        # Predictive / consensus components
        self.queen = Queen()
        self.attack_chain = AttackChainEngine(window=120)
        self.event_bus = EventBus()

        # Subscribe chain detector to event bus
        self.event_bus.subscribe(self._chain_event_ingest)

        # --- Layout root ---
        self.main_frame = tk.Frame(root, bg="#05060A")
        self.main_frame.pack(fill="both", expand=True)

        # Top privacy banner
        self.banner = tk.Label(
            self.main_frame,
            text="AUTONOMOUS • PRIVACY HARDENED • LOCAL ONLY • NO IDENTIFIERS COLLECTED",
            fg="#A0FFCF",
            bg="#07120F",
            font=("Consolas", 10, "bold")
        )
        self.banner.pack(fill="x", side="top")

        # Split: left = canvas, right = alerts/actions/log
        self.content_frame = tk.Frame(self.main_frame, bg="#05060A")
        self.content_frame.pack(fill="both", expand=True, side="top")

        self.canvas = tk.Canvas(self.content_frame, bg="#05060A", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, side="left")

        self.side_panel = tk.Frame(self.content_frame, bg="#0A0C12", width=300)
        self.side_panel.pack(fill="y", side="right")

        control_frame = tk.Frame(self.main_frame, bg="#111318")
        control_frame.pack(fill="x", side="bottom")

        # --- Status labels ---
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
            text="Autonomous mode: monitoring only (no destructive actions)",
            fg="#8888FF",
            bg="#111318"
        )
        self.info_label.pack(side="right", padx=10)

        # --- Side panel: Threat Alert + Recommended Actions + Log preview + Queen ---
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
            wraplength=280
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
            wraplength=280
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
            wraplength=280
        )
        self.log_preview.pack(fill="x", padx=10, pady=(0, 10))

        self.queen_title = tk.Label(
            self.side_panel,
            text="QUEEN CONSENSUS (LOCAL VIEW)",
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
            wraplength=280
        )
        self.queen_preview.pack(fill="x", padx=10, pady=(0, 10))

        # --- POV buffers ---
        self.num_segments = 240
        self.cpu_buffer = [0.0] * self.num_segments
        self.ram_buffer = [0.0] * self.num_segments
        self.disk_buffer = [0.0] * self.num_segments
        self.net_buffer = [0.0] * self.num_segments
        self.sec_buffer = [0.0] * self.num_segments  # security stress
        self.entropy_buffer = [0.0] * self.num_segments  # threat matrix (entropy / beacon-like)

        self.current_index = 0

        # --- Visualization parameters ---
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

        # Threat matrix ring (inner)
        self.inner_radius_threat = 60
        self.outer_radius_threat = 100

        self.running = True
        self.update_interval = 0.1  # seconds

        # For network speed calculation
        self.last_net = psutil.net_io_counters()
        self.last_net_time = time.time()

        # For security stress heuristic
        self.last_cpu = 0.0
        self.last_ram = 0.0
        self.last_disk = 0.0
        self.last_net_rate = 0.0

        # For simple beacon-like pattern detection (timing + amplitude)
        self.last_net_norm = 0.0
        self.last_net_change_time = time.time()
        self.beacon_score = 0.0

        # For context window averages
        self.context_window_samples = []
        self.context_window_size = 50  # last N samples

        # Watchdog
        self.error_count = 0
        self.max_errors = 10

        # Altered state flag (rare, high-entropy/high-stress regime)
        self.altered_state = False

        # Resize handling
        self.root.bind("<Configure>", self.on_resize)

        # Start autonomous loops
        self.status_label.config(text="Status: Running (Autonomous, Monitoring Only, Privacy Hardened)")
        threading.Thread(target=self.data_loop, daemon=True).start()
        self.schedule_redraw()
        self.schedule_event_bus()

    # =========================
    # Control / Resize
    # =========================

    def on_resize(self, event):
        if event.width < 300 or event.height < 300:
            return

        canvas_width = max(event.width - 300, 300)
        canvas_height = event.height - 60  # banner + control bar

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
    # Data sampling (anonymous)
    # =========================

    def data_loop(self):
        while self.running:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent

                disk_io = psutil.disk_io_counters()
                disk_activity = (disk_io.read_bytes + disk_io.write_bytes)

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
                disk_norm = min(disk_activity / (100 * 1024 * 1024), 1.0)
                net_norm = min(net_rate / (10 * 1024 * 1024), 1.0)

                # Security stress heuristic
                stress = 0.0
                stress += abs(cpu_norm - self.last_cpu) * 1.5
                stress += abs(ram_norm - self.last_ram) * 1.0
                stress += abs(disk_norm - self.last_disk) * 1.2
                stress += abs(net_norm - self.last_net_rate) * 1.3
                stress += (cpu_norm + ram_norm + disk_norm + net_norm) / 4.0
                stress = min(max(stress, 0.0), 2.0) / 2.0

                # Beacon-like score
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

                # Context window
                self.context_window_samples.append(
                    (cpu_norm, ram_norm, disk_norm, net_norm)
                )
                if len(self.context_window_samples) > self.context_window_size:
                    self.context_window_samples.pop(0)

                # Publish abstract events to EventBus (predictive / chain engine)
                self._publish_behavior_events(cpu_norm, ram_norm, disk_norm, net_norm, stress, entropy)

                # UI + interfaces
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

        # Simple thresholds for abstract events
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

    def _chain_event_ingest(self, evt: SecEvent):
        # Feed into attack chain engine
        self.attack_chain.add_event(evt.etype, {"entity": evt.entity, "meta": evt.meta})

    def schedule_event_bus(self):
        if not self.running:
            return
        self.event_bus.run_once()
        self.root.after(10, self.schedule_event_bus)

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

        alert_text, alert_color, actions_text, anomaly_flags = self.compute_alert_and_actions(stress, entropy)
        self.alert_label.config(text=alert_text, fg=alert_color)
        self.actions_text.config(text=actions_text)

        # Attack chain detection (predictive)
        chains = self.attack_chain.detect()
        if chains:
            # Take the highest scoring chain
            cname, cscore = max(chains, key=lambda x: x[1])
            anomaly_flags.append(cname)
            # Bernoulli-style: repeated detection increases belief
            # (here we just treat cscore as an extra risk weight)
            entropy = min(1.0, entropy + 0.1 * cscore)
            stress = min(1.0, stress + 0.1 * cscore)

        # Altered state: rare, high-stress + high-entropy regime
        self.altered_state = (stress > 0.8 and entropy > 0.8)
        if self.altered_state:
            self.status_label.config(
                text="Status: ALTERED STATE (High Stress + High Entropy)",
                fg="#FF4BFF"
            )

        codex_status = self.build_codex_status(stress, entropy, anomaly_flags, alert_text, actions_text)
        fleet_status = self.build_fleet_status(stress, entropy, anomaly_flags)

        # Queen consensus: local node contributes its anomaly entities
        queen_events = [{"entity": f, "score": 0.5} for f in anomaly_flags]
        self.queen.update(self.node_id, queen_events)
        global_risk = self.queen.global_risk()
        self.update_queen_preview(global_risk)

        if codex_status.threat_level == "HIGH":
            self.log_attacker_pattern(stress, entropy, anomaly_flags)

        self.update_log_preview()

        _ = codex_status
        _ = fleet_status

    def compute_alert_and_actions(self, stress, entropy):
        anomaly_flags = []

        if stress > 0.6:
            anomaly_flags.append("SUSTAINED_LOAD")
        if entropy > 0.6:
            anomaly_flags.append("HIGH_ENTROPY")
        if self.beacon_score > 0.4:
            anomaly_flags.append("NET_BEACON")

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
                "Sustained stress and/or beacon-like patterns observed.\n"
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

        return alert_text, alert_color, actions_text, anomaly_flags

    def build_codex_status(self, stress, entropy, anomaly_flags, alert_text, actions_text):
        if stress < 0.3 and entropy < 0.3:
            health_state = "OK"
            threat_level = "LOW"
            confidence = 0.9
        elif stress < 0.7 and entropy < 0.7:
            health_state = "DEGRADED"
            threat_level = "MEDIUM"
            confidence = 0.8
        else:
            health_state = "CRITICAL"
            threat_level = "HIGH"
            confidence = 0.85

        now_iso = datetime.now(timezone.utc).isoformat()

        status = CodexSentinelStatus(
            source="sentinel",
            health_state=health_state,
            threat_level=threat_level,
            stress_score=float(stress),
            entropy_score=float(entropy),
            anomaly_flags=list(anomaly_flags),
            recent_window_seconds=int(self.context_window_size * self.update_interval),
            confidence=float(confidence),
            risk_summary=alert_text.replace("\n", " "),
            recommended_actions=[line.strip("- ").strip() for line in actions_text.split("\n") if line.strip()],
            timestamp=now_iso
        )
        return status

    def build_fleet_status(self, stress, entropy, anomaly_flags):
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
            anomaly_flags=list(anomaly_flags),
            uptime_seconds=uptime,
            last_update_ts=now_iso
        )
        return status

    def log_attacker_pattern(self, stress, entropy, anomaly_flags):
        if self.context_window_samples:
            avg_cpu = sum(s[0] for s in self.context_window_samples) / len(self.context_window_samples)
            avg_ram = sum(s[1] for s in self.context_window_samples) / len(self.context_window_samples)
            avg_disk = sum(s[2] for s in self.context_window_samples) / len(self.context_window_samples)
            avg_net = sum(s[3] for s in self.context_window_samples) / len(self.context_window_samples)
        else:
            avg_cpu = avg_ram = avg_disk = avg_net = 0.0

        pattern_parts = []
        if "NET_BEACON" in anomaly_flags:
            pattern_parts.append("NET_BEACON")
        if "SUSTAINED_LOAD" in anomaly_flags:
            pattern_parts.append("SUSTAINED_LOAD")
        if "HIGH_ENTROPY" in anomaly_flags:
            pattern_parts.append("HIGH_ENTROPY")

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
            f"- Duration: ~{last.duration_estimate_seconds}s\n"
            f"- Context avg CPU: {last.context_window['avg_cpu']*100:.1f}%\n"
            f"- Context avg Net norm: {last.context_window['avg_net_norm']*100:.1f}%"
        )
        self.log_preview.config(text=txt)

    def update_queen_preview(self, global_risk):
        if not global_risk:
            self.queen_preview.config(text="No high-risk entities yet.")
            return

        lines = ["High-risk entities (local view):"]
        for ent, score in global_risk.items():
            lines.append(f"- {ent}: {score:.2f}")
        self.queen_preview.config(text="\n".join(lines))

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

        self.draw_background()

        self.draw_ring(self.entropy_buffer, self.inner_radius_threat, self.outer_radius_threat, mode="threat")
        self.draw_ring(self.cpu_buffer, self.inner_radius_cpu, self.outer_radius_cpu, mode="cpu")
        self.draw_ring(self.ram_buffer, self.inner_radius_ram, self.outer_radius_ram, mode="ram")
        self.draw_ring(self.disk_buffer, self.inner_radius_disk, self.outer_radius_disk, mode="disk")
        self.draw_ring(self.net_buffer, self.inner_radius_net, self.outer_radius_net, mode="net")

        self.draw_security_core()

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


# =========================
# Entry point
# =========================

def main():
    root = tk.Tk()
    app = POVUnifiedSecurityCore(root, node_id="NODE_LOCAL", role="WORKSTATION")
    root.geometry("1250x800")
    root.mainloop()

if __name__ == "__main__":
    main()
