#!/usr/bin/env python3
"""
Sentinel Tier‑8+ – unified, fully autonomous, cockpit‑capable organism

Includes:
- Capability autoloader
- Intelligent Water Data Physics Engine (flow / pressure / phase)
- Growing Data Tree (rings / branches / leaves)
- Borg Queens + Borg Collective (swarm growth & risk)
- Real‑Time Queen (consensus over entities)
- Attack Chain Engine + Event Bus (kill‑chain awareness)
- 🧠 Level 4 Awareness Engine (meta‑states: Hyper‑Flow / Deep‑Dream / Sentinel / Recovery‑Flow)
- Self‑Check Engine (integrity watchdog)
- PySide6 cockpit (if available)
"""

import sys
import time
import logging
import importlib
import threading
from collections import deque
from types import SimpleNamespace

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_MODE = "DEFEND"   # OBSERVE / DEFEND / AGGRESSIVE
LOOP_BASE_INTERVAL = 0.5  # starting loop interval

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sentinel.tier8")

# ---------------------------------------------------------------------
# Capability loader (autoloader → capability negotiation)
# ---------------------------------------------------------------------

MODULES = {
    "numpy":   {"required": True,  "role": "core"},
    "psutil":  {"required": True,  "role": "core"},
    "torch":   {"required": False, "role": "dl"},
    "cupy":    {"required": False, "role": "gpu"},
    "PySide6": {"required": False, "role": "cockpit"},
}

def load_capabilities():
    caps = SimpleNamespace()
    caps.modules = {}
    caps.flags = {}

    for name, meta in MODULES.items():
        try:
            m = importlib.import_module(name)
            caps.modules[name] = m
            caps.flags[name] = True
            log.info(f"[CAP] {name} OK ({meta.get('role','core')})")
        except Exception as e:
            caps.modules[name] = None
            caps.flags[name] = False
            if meta.get("required", False):
                log.error(f"[CAP] Required module missing: {name}")
                raise RuntimeError(f"Required module missing: {name}") from e
            log.warning(f"[CAP] {name} unavailable: {e}")

    caps.has_gpu     = caps.flags.get("cupy", False)
    caps.has_dl      = caps.flags.get("torch", False)
    caps.has_cockpit = caps.flags.get("PySide6", False)
    return caps

CAPS = load_capabilities()

# ---------------------------------------------------------------------
# Telemetry bus (stub – replace with your real sensors)
# ---------------------------------------------------------------------

class TelemetryBus:
    def __init__(self):
        self._tick = 0

    def collect(self):
        self._tick += 1
        now = time.time()
        return {
            "tick": self._tick,
            "time": now,
            "cpu_load": (self._tick % 100) / 100.0,
            "io_rate": (self._tick % 50) / 10.0,
            "net_rate": (self._tick % 70) / 10.0,
            "proc_churn": (self._tick % 30),
        }

# ---------------------------------------------------------------------
# Anomaly engine (stub – replace with your real models)
# ---------------------------------------------------------------------

class AnomalyEngine:
    def __init__(self):
        self._counter = 0

    def evaluate(self, signals):
        self._counter += 1
        if self._counter % 20 == 0:
            return [{
                "label": "synthetic_anomaly",
                "score": 2.5,
                "pid": 1234,
            }]
        return []

# ---------------------------------------------------------------------
# Swarm engine – trust‑decay + risk aggregation
# ---------------------------------------------------------------------

class SwarmEngine:
    def __init__(self):
        self.nodes = {}
        now = time.time()
        self.nodes["node-1"] = {"id": "node-1", "trust": 0.95, "last_seen": now, "state": "ok"}
        self.nodes["node-2"] = {"id": "node-2", "trust": 0.90, "last_seen": now, "state": "ok"}

    def update(self, signals, anomalies):
        now = time.time()
        for nid, n in list(self.nodes.items()):
            dt = now - n["last_seen"]
            n["last_seen"] = now
            n["trust"] *= 0.99 ** (dt / 5.0)
            if n["trust"] < 0.3:
                n["state"] = "suspect"

        risk = sum(1.0 - n["trust"] for n in self.nodes.values())
        return {
            "nodes": list(self.nodes.values()),
            "risk": risk,
        }

# ---------------------------------------------------------------------
# Intelligent Water Data Physics Engine
# ---------------------------------------------------------------------

class IntelligentWaterEngine:
    def __init__(self, history_len=64):
        self.history = deque(maxlen=history_len)

    def update(self, signals, tension):
        self.history.append({
            "time": signals["time"],
            "cpu_load": signals["cpu_load"],
            "io_rate": signals["io_rate"],
            "net_rate": signals["net_rate"],
            "proc_churn": signals["proc_churn"],
        })

        flow = self._compute_flow()
        pressure = self._compute_pressure(tension)
        phase = self._compute_phase(pressure)

        return {
            "flow": flow,
            "pressure": pressure,
            "phase": phase,
        }

    def _compute_flow(self):
        if len(self.history) < 2:
            return 0.0
        a = self.history[-2]
        b = self.history[-1]
        dt = max(b["time"] - a["time"], 1e-6)
        dcpu = abs(b["cpu_load"] - a["cpu_load"])
        dio = abs(b["io_rate"] - a["io_rate"])
        dnet = abs(b["net_rate"] - a["net_rate"])
        dproc = abs(b["proc_churn"] - a["proc_churn"]) / 10.0
        return (dcpu + dio + dnet + dproc) / dt

    def _compute_pressure(self, tension):
        if not self.history:
            return tension
        h = self.history[-1]
        load = h["cpu_load"] + (h["io_rate"] / 10.0) + (h["net_rate"] / 10.0)
        return 0.6 * tension + 0.4 * load * 5.0

    def _compute_phase(self, pressure):
        if pressure < 1.0:
            return "LIQUID"
        elif pressure < 4.0:
            return "VAPOR"
        else:
            return "PLASMA"

# ---------------------------------------------------------------------
# Data Tree – growing memory (rings / branches / leaves)
# ---------------------------------------------------------------------

class DataTree:
    def __init__(self):
        self.root = {
            "type": "root",
            "children": [],
            "rings": [],
        }

    def add_ring(self, tension, flow, pressure, phase, anomalies, swarm, borg_risk, chain_risk):
        ring = {
            "tension": tension,
            "flow": flow,
            "pressure": pressure,
            "phase": phase,
            "anomaly_count": len(anomalies),
            "swarm_risk": swarm.get("risk", 0.0),
            "borg_risk": borg_risk,
            "chain_risk": chain_risk,
            "timestamp": time.time(),
        }
        self.root["rings"].append(ring)

    def add_leaf(self, signals):
        leaf = {
            "type": "leaf",
            "signals": signals,
            "timestamp": signals.get("time", time.time()),
        }
        self._attach_leaf(leaf)

    def _attach_leaf(self, leaf):
        key = (
            round(leaf["signals"]["cpu_load"], 1),
            round(leaf["signals"]["io_rate"], 1),
            round(leaf["signals"]["net_rate"], 1),
        )
        branch = None
        for child in self.root["children"]:
            if child["key"] == key:
                branch = child
                break
        if branch is None:
            branch = {
                "type": "branch",
                "key": key,
                "children": [],
            }
            self.root["children"].append(branch)
        branch["children"].append(leaf)

    def stats(self):
        branch_count = len(self.root["children"])
        leaf_count = sum(len(b["children"]) for b in self.root["children"])
        ring_count = len(self.root["rings"])
        return {
            "branches": branch_count,
            "leaves": leaf_count,
            "rings": ring_count,
        }

# ---------------------------------------------------------------------
# Borg Queens & Borg Collective
# ---------------------------------------------------------------------

class BorgQueen:
    def __init__(self, queen_id, origin="root"):
        self.id = queen_id
        self.origin = origin
        self.trust = 1.0
        self.nodes = {}
        self.created_at = time.time()
        self.last_update = self.created_at

    def attach_node(self, node_id, initial_trust=0.9):
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "trust": initial_trust,
                "state": "ok",
                "last_seen": time.time(),
            }

    def update_node(self, node_id, delta_trust=-0.01):
        n = self.nodes.get(node_id)
        if not n:
            return
        n["trust"] = max(0.0, min(1.0, n["trust"] + delta_trust))
        n["last_seen"] = time.time()
        if n["trust"] < 0.3:
            n["state"] = "suspect"
        self.last_update = time.time()

    def aggregate_risk(self):
        if not self.nodes:
            return 0.0
        return sum(1.0 - n["trust"] for n in self.nodes.values())

class BorgCollective:
    def __init__(self):
        self.queens = {}

    def ensure_queen(self, queen_id, origin="root"):
        if queen_id not in self.queens:
            self.queens[queen_id] = BorgQueen(queen_id, origin)
        return self.queens[queen_id]

    def grow_node(self, queen_id, node_id):
        q = self.ensure_queen(queen_id)
        q.attach_node(node_id)

    def update_from_swarm(self, swarm_state):
        q = self.ensure_queen("queen-alpha", origin="swarm")
        for n in swarm_state.get("nodes", []):
            q.attach_node(n["id"], initial_trust=n["trust"])
            if n["state"] == "suspect":
                q.update_node(n["id"], delta_trust=-0.05)

    def total_risk(self):
        return sum(q.aggregate_risk() for q in self.queens.values())

    def snapshot(self):
        return {
            "queens": {
                qid: {
                    "id": q.id,
                    "origin": q.origin,
                    "trust": q.trust,
                    "nodes": list(q.nodes.values()),
                }
                for qid, q in self.queens.items()
            }
        }

# ---------------------------------------------------------------------
# Real‑Time Queen (entity consensus over events)
# ---------------------------------------------------------------------

class RealTimeQueen:
    def __init__(self):
        self.nodes = {}  # node_id -> list of events

    def update(self, node_id, events):
        self.nodes[node_id] = events

    def global_risk(self):
        risk = {}
        for node, evts in self.nodes.items():
            for e in evts:
                ent = e.get("entity")
                score = e.get("score", 0.0)
                if ent is None:
                    continue
                risk[ent] = risk.get(ent, 0.0) + score
        return {k: v for k, v in risk.items() if v > 1.5}

# ---------------------------------------------------------------------
# Attack Chain Engine + Event Bus
# ---------------------------------------------------------------------

class SecEvent:
    def __init__(self, etype, entity, meta=None):
        self.ts = time.time()
        self.type = etype
        self.entity = entity
        self.meta = meta or {}

class AttackChainEngine:
    def __init__(self, window=120):
        self.events = deque()
        self.window = window

    def add_event(self, event_type, data):
        now = time.time()
        self.events.append((now, event_type, data))
        self._cleanup(now)

    def _cleanup(self, now):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def detect(self):
        types = [e[1] for e in self.events]
        chains = []

        if all(x in types for x in ["proc_spawn", "powershell", "net_connect"]):
            chains.append(("LOLBIN_ATTACK", 0.9))

        if types.count("proc_spawn") > 5 and "net_connect" in types:
            chains.append(("PROCESS_STORM", 0.8))

        if "file_mod" in types and "net_connect" in types:
            chains.append(("PERSISTENCE_EXFIL", 0.85))

        return chains

class EventBus:
    def __init__(self):
        self.subscribers = []
        self.queue = deque()
        self._running = False

    def publish(self, event):
        self.queue.append(event)

    def subscribe(self, fn):
        self.subscribers.append(fn)

    def run(self):
        self._running = True
        while self._running:
            if self.queue:
                evt = self.queue.popleft()
                for fn in self.subscribers:
                    fn(evt)
            time.sleep(0.01)

    def stop(self):
        self._running = False

# ---------------------------------------------------------------------
# Awareness Engine – 🧠 Level 4 Meta‑States
# ---------------------------------------------------------------------

class AwarenessEngine:
    """
    Meta‑states:
      - HYPER_FLOW
      - DEEP_DREAM
      - SENTINEL
      - RECOVERY_FLOW
    """

    def __init__(self):
        self.state = "SENTINEL"
        self.params = self._params_for(self.state)

    def _params_for(self, state):
        if state == "HYPER_FLOW":
            return {
                "prediction_horizon": 5,
                "aggressiveness": 1.2,
                "ram_appetite": 1.3,
                "thread_expansion": 1.5,
                "cache_factor": 1.2,
            }
        if state == "DEEP_DREAM":
            return {
                "prediction_horizon": 20,
                "aggressiveness": 0.7,
                "ram_appetite": 1.5,
                "thread_expansion": 1.2,
                "cache_factor": 1.5,
            }
        if state == "RECOVERY_FLOW":
            return {
                "prediction_horizon": 3,
                "aggressiveness": 0.5,
                "ram_appetite": 0.8,
                "thread_expansion": 0.7,
                "cache_factor": 0.9,
            }
        # SENTINEL default
        return {
            "prediction_horizon": 10,
            "aggressiveness": 1.0,
            "ram_appetite": 1.0,
            "thread_expansion": 1.0,
            "cache_factor": 1.0,
        }

    def update_state(self, tension, risk, chain_risk):
        if risk > 6 or chain_risk > 1.0:
            new_state = "HYPER_FLOW"
        elif tension > 4 and risk > 3:
            new_state = "SENTINEL"
        elif tension < 1 and risk < 1:
            new_state = "RECOVERY_FLOW"
        else:
            new_state = "DEEP_DREAM"

        if new_state != self.state:
            log.info(f"[AWARENESS] state change {self.state} → {new_state}")
            self.state = new_state
            self.params = self._params_for(self.state)

    def snapshot(self):
        return {
            "state": self.state,
            "params": self.params,
        }

# ---------------------------------------------------------------------
# Policy engine – mode‑, phase‑, Borg‑ & awareness‑aware
# ---------------------------------------------------------------------

class PolicyEngine:
    def __init__(self, mode=DEFAULT_MODE, awareness_engine=None):
        self.mode = mode
        self.awareness = awareness_engine or AwarenessEngine()

    def decide(self, signals, anomalies, swarm, tension, water_state, borg_risk, chain_risk):
        phase = water_state["phase"]
        flow = water_state["flow"]
        pressure = water_state["pressure"]

        max_score = max((a["score"] for a in anomalies), default=0.0)
        swarm_risk = swarm.get("risk", 0.0)

        phase_factor = {
            "LIQUID": 0.8,
            "VAPOR": 1.0,
            "PLASMA": 1.3,
        }.get(phase, 1.0)

        risk = (max_score + swarm_risk + borg_risk + chain_risk + tension * 0.2 + flow * 0.3) * phase_factor

        self.awareness.update_state(tension, risk, chain_risk)
        aware = self.awareness.snapshot()
        aggr = aware["params"]["aggressiveness"]

        if self.mode == "OBSERVE":
            level = "log"
        elif self.mode == "DEFEND":
            level = "micro" if risk * aggr < 3 else "macro"
        else:  # AGGRESSIVE
            level = "macro" if risk * aggr > 1 else "micro"

        decision = {
            "risk": risk,
            "level": level,
            "severity": 1.0 if level == "log" else 3.0 if level == "micro" else 6.0,
            "actions": self._plan_actions(level, anomalies),
            "phase": phase,
            "flow": flow,
            "pressure": pressure,
            "borg_risk": borg_risk,
            "chain_risk": chain_risk,
            "awareness": aware,
        }
        return decision

    def _plan_actions(self, level, anomalies):
        if level == "log":
            return [{"type": "log"}]
        if level == "micro":
            return [{"type": "throttle", "targets": [a["pid"] for a in anomalies[:3] if "pid" in a]}]
        if level == "macro":
            return [{"type": "isolate", "targets": [a["pid"] for a in anomalies[:5] if "pid" in a]}]

    def execute(self, decision):
        log.info(
            f"[POLICY] phase={decision['phase']} level={decision['level']} "
            f"risk={decision['risk']:.2f} borg_risk={decision['borg_risk']:.2f} "
            f"chain_risk={decision['chain_risk']:.2f} "
            f"awareness={decision['awareness']['state']} "
            f"actions={decision['actions']}"
        )

# ---------------------------------------------------------------------
# Self‑Check Engine – integrity watchdog
# ---------------------------------------------------------------------

class SelfCheckEngine:
    def __init__(self, core_ref):
        self.core = core_ref
        self.interval = 15.0
        self._running = False

    def _check_once(self):
        try:
            assert isinstance(self.core.policy, PolicyEngine)
            assert callable(self.core.step)
            assert hasattr(self.core, "swarm")
            assert hasattr(self.core, "borg")
        except Exception as e:
            log.error(f"[SELF_CHECK] Integrity check failed: {e}")
        else:
            log.debug("[SELF_CHECK] OK")

    def run(self):
        self._running = True
        while self._running:
            self._check_once()
            time.sleep(self.interval)

    def stop(self):
        self._running = False

# ---------------------------------------------------------------------
# Tier‑8 core engine – loop + water + tree + Borg + chains + awareness
# ---------------------------------------------------------------------

class Tier8Core:
    def __init__(self, capability_map, policy_engine, telemetry_bus,
                 anomaly_engine, swarm_engine, water_engine,
                 data_tree, borg_collective,
                 rt_queen, chain_engine, event_bus, awareness_engine):
        self.cap = capability_map
        self.policy = policy_engine
        self.telemetry = telemetry_bus
        self.anomaly = anomaly_engine
        self.swarm = swarm_engine
        self.water = water_engine
        self.tree = data_tree
        self.borg = borg_collective
        self.rt_queen = rt_queen
        self.chain_engine = chain_engine
        self.event_bus = event_bus
        self.awareness = awareness_engine

        self.loop_interval = LOOP_BASE_INTERVAL
        self.tension = 0.0
        self.history_tension = deque(maxlen=256)
        self.timeline = deque(maxlen=1024)

    def ingest_sec_event(self, etype, entity, meta=None):
        evt = SecEvent(etype, entity, meta)
        self.chain_engine.add_event(etype, {"entity": entity, "meta": meta or {}})

    def step(self):
        signals = self.telemetry.collect()
        anomalies = self.anomaly.evaluate(signals)
        swarm_state = self.swarm.update(signals, anomalies)

        self.borg.update_from_swarm(swarm_state)
        borg_state = self.borg.snapshot()
        borg_risk = self.borg.total_risk()

        water_state = self.water.update(signals, self.tension)

        chains = self.chain_engine.detect()
        chain_risk = sum(score for _, score in chains)

        decision = self.policy.decide(
            signals=signals,
            anomalies=anomalies,
            swarm=swarm_state,
            tension=self.tension,
            water_state=water_state,
            borg_risk=borg_risk,
            chain_risk=chain_risk,
        )

        self.policy.execute(decision)

        self._update_tension(anomalies, decision, water_state, borg_risk, chain_risk)
        self._adapt_loop_interval()

        self.tree.add_leaf(signals)
        self.tree.add_ring(
            tension=self.tension,
            flow=water_state["flow"],
            pressure=water_state["pressure"],
            phase=water_state["phase"],
            anomalies=anomalies,
            swarm=swarm_state,
            borg_risk=borg_risk,
            chain_risk=chain_risk,
        )

        self._append_timeline(signals, anomalies, decision, water_state, borg_risk, chain_risk)

        return {
            "signals": signals,
            "anomalies": anomalies,
            "swarm": swarm_state,
            "decision": decision,
            "tension": self.tension,
            "loop_interval": self.loop_interval,
            "timeline": list(self.timeline),
            "water": water_state,
            "borg": borg_state,
            "tree_stats": self.tree.stats(),
        }

    def _update_tension(self, anomalies, decision, water_state, borg_risk, chain_risk):
        base = sum(a["score"] for a in anomalies[-10:]) if anomalies else 0.0
        sev = decision.get("severity", 0.0)
        pressure = water_state["pressure"]
        self.tension = 0.7 * self.tension + 0.3 * (
            base + sev + pressure * 0.5 + borg_risk * 0.3 + chain_risk * 0.5
        )
        self.history_tension.append(self.tension)

    def _adapt_loop_interval(self):
        aware = self.awareness.snapshot()
        exp_factor = aware["params"]["thread_expansion"]
        if self.tension > 5:
            target = 0.1 / exp_factor
        elif self.tension > 1:
            target = 0.25 / exp_factor
        else:
            target = 0.75 / exp_factor
        self.loop_interval = max(0.05, 0.8 * self.loop_interval + 0.2 * target)

    def _append_timeline(self, signals, anomalies, decision, water_state, borg_risk, chain_risk):
        tick = signals.get("tick")
        risk = decision.get("risk", 0.0)
        level = decision.get("level", "log")
        phase = decision.get("phase", "LIQUID")
        aware_state = decision["awareness"]["state"]
        line = (
            f"[{tick}] phase={phase} aware={aware_state} risk={risk:.2f} "
            f"flow={water_state['flow']:.2f} pressure={water_state['pressure']:.2f} "
            f"borg_risk={borg_risk:.2f} chain_risk={chain_risk:.2f} "
            f"level={level} anomalies={len(anomalies)}"
        )
        self.timeline.append(line)

# ---------------------------------------------------------------------
# PySide6 cockpit – awareness, Borg & tree‑aware
# ---------------------------------------------------------------------

if CAPS.has_cockpit:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QTabWidget, QTextEdit, QListWidget
    )
    from PySide6.QtCore import Qt, QTimer

    class SentinelCockpit(QMainWindow):
        def __init__(self, core_state_callable, policy_engine, parent=None):
            super().__init__(parent)
            self.core_state = core_state_callable
            self.policy = policy_engine
            self.setWindowTitle("Sentinel Cockpit – Tier‑8+ (Water / Tree / Borg / Awareness)")
            self.resize(1500, 850)
            self._build_ui()
            self._wire_timers()

        def _build_ui(self):
            central = QWidget()
            root_layout = QVBoxLayout(central)

            status_bar = QHBoxLayout()
            self.lbl_mode = QLabel(f"MODE: {self.policy.mode}")
            self.lbl_health = QLabel("HEALTH: Unknown")
            self.lbl_swarm = QLabel("SWARM: 0 nodes")
            self.lbl_phase = QLabel("PHASE: LIQUID")
            self.lbl_tree = QLabel("TREE: 0b/0l/0r")
            self.lbl_awareness = QLabel("AWARE: SENTINEL")
            for w in (self.lbl_mode, self.lbl_health, self.lbl_swarm, self.lbl_phase, self.lbl_tree, self.lbl_awareness):
                w.setStyleSheet("font-weight: bold;")
                status_bar.addWidget(w)
            status_bar.addStretch()
            root_layout.addLayout(status_bar)

            tabs = QTabWidget()
            tabs.addTab(self._build_timeline_tab(), "Timeline")
            tabs.addTab(self._build_anomaly_tab(), "Anomalies")
            tabs.addTab(self._build_swarm_tab(), "Swarm / Borg")
            tabs.addTab(self._build_controls_tab(), "Controls")
            root_layout.addWidget(tabs)

            self.setCentralWidget(central)

        def _build_timeline_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            self.timeline_view = QTextEdit()
            self.timeline_view.setReadOnly(True)
            self.timeline_view.setPlaceholderText("Event timeline, attack narratives, state transitions...")
            layout.addWidget(self.timeline_view)
            return w

        def _build_anomaly_tab(self):
            w = QWidget()
            layout = QHBoxLayout(w)
            self.lst_anomalies = QListWidget()
            self.lst_anomalies.setMinimumWidth(350)
            self.anomaly_details = QTextEdit()
            self.anomaly_details.setReadOnly(True)
            self.anomaly_details.setPlaceholderText("Selected anomaly details, scores, features, narrative...")
            layout.addWidget(self.lst_anomalies)
            layout.addWidget(self.anomaly_details)
            return w

        def _build_swarm_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            self.swarm_view = QTextEdit()
            self.swarm_view.setReadOnly(True)
            self.swarm_view.setPlaceholderText("Swarm nodes, Borg queens, trust levels, consensus state...")
            layout.addWidget(self.swarm_view)
            return w

        def _build_controls_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)

            btn_row = QHBoxLayout()
            self.btn_mode_observe = QPushButton("Observe")
            self.btn_mode_defend = QPushButton("Defend")
            self.btn_mode_aggressive = QPushButton("Aggressive")
            for b in (self.btn_mode_observe, self.btn_mode_defend, self.btn_mode_aggressive):
                btn_row.addWidget(b)
            btn_row.addStretch()
            layout.addLayout(btn_row)

            self.control_log = QTextEdit()
            self.control_log.setReadOnly(True)
            self.control_log.setPlaceholderText("Control actions, overrides, and system decisions...")
            layout.addWidget(self.control_log)

            self.btn_mode_observe.clicked.connect(lambda: self._set_mode("OBSERVE"))
            self.btn_mode_defend.clicked.connect(lambda: self._set_mode("DEFEND"))
            self.btn_mode_aggressive.clicked.connect(lambda: self._set_mode("AGGRESSIVE"))

            return w

        def _wire_timers(self):
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._refresh_from_core)
            self.timer.start(500)

        def _refresh_from_core(self):
            state = self.core_state()
            tension = state.get("tension", 0.0)
            decision = state.get("decision", {})
            risk = decision.get("risk", 0.0)
            phase = decision.get("phase", "LIQUID")
            borg_risk = decision.get("borg_risk", 0.0)
            chain_risk = decision.get("chain_risk", 0.0)
            aware = decision.get("awareness", {"state": "SENTINEL"})
            aware_state = aware.get("state", "SENTINEL")
            water = state.get("water", {})
            flow = water.get("flow", 0.0)
            pressure = water.get("pressure", 0.0)
            swarm = state.get("swarm", {})
            nodes = swarm.get("nodes", [])
            borg = state.get("borg", {})
            queens = borg.get("queens", {})
            tree_stats = state.get("tree_stats", {"branches": 0, "leaves": 0, "rings": 0})

            self.lbl_health.setText(
                f"HEALTH: Tension={tension:.2f} Risk={risk:.2f} Borg={borg_risk:.2f} Chain={chain_risk:.2f}"
            )
            self.lbl_swarm.setText(f"SWARM: {len(nodes)} nodes / {len(queens)} queens")
            self.lbl_phase.setText(f"PHASE: {phase}")
            self.lbl_tree.setText(
                f"TREE: {tree_stats['branches']}b/{tree_stats['leaves']}l/{tree_stats['rings']}r"
            )
            self.lbl_awareness.setText(f"AWARE: {aware_state}")

            if tension < 1:
                color = "#4caf50"
            elif tension < 4:
                color = "#ff9800"
            else:
                color = "#f44336"
            self.lbl_health.setStyleSheet(f"font-weight: bold; color: {color};")

            timeline = state.get("timeline", [])
            self.timeline_view.setPlainText("\n".join(timeline[-500:]))

            anomalies = state.get("anomalies", [])
            self.lst_anomalies.clear()
            for a in anomalies[-100:]:
                score = a.get("score", 0.0)
                label = a.get("label", "unknown")
                self.lst_anomalies.addItem(f"[{score:.3f}] {label}")

            swarm_lines = []
            for n in nodes:
                swarm_lines.append(f"{n['id']} | trust={n['trust']:.2f} | state={n['state']}")
            swarm_lines.append(f"\nflow={flow:.2f} pressure={pressure:.2f}")
            swarm_lines.append(f"borg_risk={borg_risk:.2f} chain_risk={chain_risk:.2f}")
            for qid, q in queens.items():
                swarm_lines.append(f"[QUEEN] {qid} nodes={len(q['nodes'])} trust={q['trust']:.2f}")
            self.swarm_view.setPlainText("\n".join(swarm_lines))

        def _set_mode(self, mode: str):
            self.policy.mode = mode
            self.lbl_mode.setText(f"MODE: {mode}")
            self.control_log.append(f"Mode set to: {mode}")

    def run_cockpit(core_state_callable, policy_engine):
        app = QApplication(sys.argv)
        win = SentinelCockpit(core_state_callable, policy_engine)
        win.show()
        sys.exit(app.exec())

else:
    def run_cockpit(core_state_callable, policy_engine):
        log.warning("PySide6 not available – cockpit disabled.")
        while True:
            time.sleep(10)

# ---------------------------------------------------------------------
# Wiring – fully autonomous Tier‑8+ fusion
# ---------------------------------------------------------------------

telemetry_bus = TelemetryBus()
anomaly_engine = AnomalyEngine()
swarm_engine = SwarmEngine()
water_engine = IntelligentWaterEngine()
data_tree = DataTree()
borg_collective = BorgCollective()
rt_queen = RealTimeQueen()
chain_engine = AttackChainEngine()
event_bus = EventBus()
awareness_engine = AwarenessEngine()
policy_engine = PolicyEngine(mode=DEFAULT_MODE, awareness_engine=awareness_engine)

core = Tier8Core(
    capability_map=CAPS,
    policy_engine=policy_engine,
    telemetry_bus=telemetry_bus,
    anomaly_engine=anomaly_engine,
    swarm_engine=swarm_engine,
    water_engine=water_engine,
    data_tree=data_tree,
    borg_collective=borg_collective,
    rt_queen=rt_queen,
    chain_engine=chain_engine,
    event_bus=event_bus,
    awareness_engine=awareness_engine,
)

_state_snapshot = {
    "tension": 0.0,
    "anomalies": [],
    "swarm": {"nodes": []},
    "decision": {},
    "timeline": [],
    "water": {"flow": 0.0, "pressure": 0.0, "phase": "LIQUID"},
    "borg": {"queens": {}},
    "tree_stats": {"branches": 0, "leaves": 0, "rings": 0},
}

def core_thread():
    global _state_snapshot
    while True:
        snap = core.step()
        _state_snapshot = {
            "tension": snap["tension"],
            "anomalies": snap["anomalies"],
            "swarm": snap["swarm"],
            "decision": snap["decision"],
            "timeline": snap["timeline"],
            "water": snap["water"],
            "borg": snap["borg"],
            "tree_stats": snap["tree_stats"],
        }
        time.sleep(core.loop_interval)

def get_state():
    return dict(_state_snapshot)

def self_check_thread(core_ref):
    checker = SelfCheckEngine(core_ref)
    checker.run()

def event_bus_thread():
    event_bus.run()

def main():
    t_core = threading.Thread(target=core_thread, daemon=True)
    t_core.start()

    t_self = threading.Thread(target=self_check_thread, args=(core,), daemon=True)
    t_self.start()

    t_evt = threading.Thread(target=event_bus_thread, daemon=True)
    t_evt.start()

    run_cockpit(get_state, policy_engine)

if __name__ == "__main__":
    main()
