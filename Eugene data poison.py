#!/usr/bin/env python3
"""
Sentinel Tier‑8 – unified, fully autonomous, cockpit‑capable organism.
"""

import sys
import time
import math
import logging
import importlib
import threading
from collections import deque
from types import SimpleNamespace

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
        # Replace with real ETW, process, FS, net, GPU, etc.
        self._tick += 1
        return {
            "tick": self._tick,
            "time": time.time(),
        }

# ---------------------------------------------------------------------
# Anomaly engine (stub – replace with your real models)
# ---------------------------------------------------------------------

class AnomalyEngine:
    def __init__(self):
        self._counter = 0

    def evaluate(self, signals):
        # Replace with IsolationForest / DL / entropy models
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
        # Example static nodes; replace with real swarm mesh
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
# Policy engine – autonomous, mode‑aware
# ---------------------------------------------------------------------

class PolicyEngine:
    def __init__(self, mode="DEFEND"):
        self.mode = mode  # OBSERVE / DEFEND / AGGRESSIVE

    def decide(self, signals, anomalies, swarm, tension):
        max_score = max((a["score"] for a in anomalies), default=0.0)
        swarm_risk = swarm.get("risk", 0.0)
        risk = max_score + swarm_risk + tension * 0.2

        if self.mode == "OBSERVE":
            level = "log"
        elif self.mode == "DEFEND":
            level = "micro" if risk < 3 else "macro"
        else:  # AGGRESSIVE
            level = "macro" if risk > 1 else "micro"

        decision = {
            "risk": risk,
            "level": level,
            "severity": 1.0 if level == "log" else 3.0 if level == "micro" else 6.0,
            "actions": self._plan_actions(level, anomalies),
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
        # Wire this into your real action layer (process isolation, firewall, etc.)
        log.info(f"[POLICY] level={decision['level']} risk={decision['risk']:.2f} actions={decision['actions']}")

# ---------------------------------------------------------------------
# Tier‑8 core engine – self‑optimizing loop
# ---------------------------------------------------------------------

class Tier8Core:
    def __init__(self, capability_map, policy_engine, telemetry_bus, anomaly_engine, swarm_engine):
        self.cap = capability_map
        self.policy = policy_engine
        self.telemetry = telemetry_bus
        self.anomaly = anomaly_engine
        self.swarm = swarm_engine

        self.loop_interval = 0.5
        self.tension = 0.0
        self.history_tension = deque(maxlen=256)
        self.timeline = deque(maxlen=1024)

    def step(self):
        # 1) Sense
        signals = self.telemetry.collect()

        # 2) Interpret
        anomalies = self.anomaly.evaluate(signals)
        swarm_state = self.swarm.update(signals, anomalies)

        # 3) Decide
        decision = self.policy.decide(
            signals=signals,
            anomalies=anomalies,
            swarm=swarm_state,
            tension=self.tension,
        )

        # 4) Act
        self.policy.execute(decision)

        # 5) Learn / adjust
        self._update_tension(anomalies, decision)
        self._adapt_loop_interval()

        # 6) Timeline
        self._append_timeline(signals, anomalies, decision)

        # 7) Export snapshot
        return {
            "signals": signals,
            "anomalies": anomalies,
            "swarm": swarm_state,
            "decision": decision,
            "tension": self.tension,
            "loop_interval": self.loop_interval,
            "timeline": list(self.timeline),
        }

    def _update_tension(self, anomalies, decision):
        base = sum(a["score"] for a in anomalies[-10:]) if anomalies else 0.0
        sev = decision.get("severity", 0.0)
        self.tension = 0.8 * self.tension + 0.2 * (base + sev)
        self.history_tension.append(self.tension)

    def _adapt_loop_interval(self):
        if self.tension > 5:
            target = 0.1
        elif self.tension > 1:
            target = 0.25
        else:
            target = 0.75
        self.loop_interval = 0.8 * self.loop_interval + 0.2 * target

    def _append_timeline(self, signals, anomalies, decision):
        tick = signals.get("tick")
        risk = decision.get("risk", 0.0)
        level = decision.get("level", "log")
        line = f"[{tick}] risk={risk:.2f} level={level} anomalies={len(anomalies)}"
        self.timeline.append(line)

# ---------------------------------------------------------------------
# PySide6 cockpit – tension‑aware operator bridge
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
            self.setWindowTitle("Sentinel Cockpit – Tier‑8")
            self.resize(1400, 800)
            self._build_ui()
            self._wire_timers()

        def _build_ui(self):
            central = QWidget()
            root_layout = QVBoxLayout(central)

            status_bar = QHBoxLayout()
            self.lbl_mode = QLabel(f"MODE: {self.policy.mode}")
            self.lbl_health = QLabel("HEALTH: Unknown")
            self.lbl_swarm = QLabel("SWARM: 0 nodes")
            for w in (self.lbl_mode, self.lbl_health, self.lbl_swarm):
                w.setStyleSheet("font-weight: bold;")
                status_bar.addWidget(w)
            status_bar.addStretch()
            root_layout.addLayout(status_bar)

            tabs = QTabWidget()
            tabs.addTab(self._build_timeline_tab(), "Timeline")
            tabs.addTab(self._build_anomaly_tab(), "Anomalies")
            tabs.addTab(self._build_swarm_tab(), "Swarm")
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
            self.swarm_view.setPlaceholderText("Swarm nodes, trust levels, consensus state...")
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
            swarm = state.get("swarm", {})
            nodes = swarm.get("nodes", [])

            self.lbl_health.setText(f"HEALTH: Tension={tension:.2f} Risk={risk:.2f}")
            self.lbl_swarm.setText(f"SWARM: {len(nodes)} nodes")

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
# Wiring – fully autonomous Tier‑8 fusion
# ---------------------------------------------------------------------

telemetry_bus = TelemetryBus()
anomaly_engine = AnomalyEngine()
swarm_engine = SwarmEngine()
policy_engine = PolicyEngine(mode="DEFEND")
core = Tier8Core(
    capability_map=CAPS,
    policy_engine=policy_engine,
    telemetry_bus=telemetry_bus,
    anomaly_engine=anomaly_engine,
    swarm_engine=swarm_engine,
)

_state_snapshot = {
    "tension": 0.0,
    "anomalies": [],
    "swarm": {"nodes": []},
    "decision": {},
    "timeline": [],
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
        }
        time.sleep(core.loop_interval)

def get_state():
    return dict(_state_snapshot)

def main():
    t = threading.Thread(target=core_thread, daemon=True)
    t.start()
    run_cockpit(get_state, policy_engine)

if __name__ == "__main__":
    main()
