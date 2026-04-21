#!/usr/bin/env python3
# apex_sentinel_unified_v2.py
#
# APEX‑Sentinel OS / ChronoMind Nexus / Parallax Swarm Engine
# HybridBrain + ReplicaNPU + Reinforcement + WaterPhysics + Swarm + Queen + ETW stub + UIAutomation stub
# PySide6 cockpit with Altered States panel, Best‑Guess line, Reasoning heatmap, Reboot memory

import sys
import os
import json
import time
import math
import random
import subprocess
import importlib
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

# ============================================================
# Autoloader
# ============================================================

REQUIRED_LIBS = ["PySide6"]

def ensure_deps():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

ensure_deps()

from PySide6 import QtWidgets, QtCore, QtGui

try:
    import onnxruntime as ort
except Exception:
    ort = None

# ============================================================
# Helpers
# ============================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def rolling_avg(old: float, new: float, weight: float = 0.1) -> float:
    if old == 0.0:
        return new
    return (1.0 - weight) * old + weight * new

# ============================================================
# PatternMemory (Bernoulli) + WaterPhysics + RL
# ============================================================

class PatternMemory:
    def __init__(self, max_patterns=5000):
        self.stats: Dict[Any, Dict[str, int]] = {}
        self.max_patterns = max_patterns

    def _key(self, x, stance=None, meta_state=None):
        return (
            stance,
            meta_state,
            round(x[0], 1) if len(x) > 0 else 0.0,
            round(x[1], 1) if len(x) > 1 else 0.0,
            round(x[2], 1) if len(x) > 2 else 0.0,
        )

    def record(self, x, outcome, stance=None, meta_state=None):
        k = self._key(x, stance, meta_state)
        entry = self.stats.setdefault(k, {"overload": 0, "stable": 0, "wins": 0})
        entry[outcome] = entry.get(outcome, 0) + 1
        if len(self.stats) > self.max_patterns:
            self.stats.pop(next(iter(self.stats)))

    def bernoulli_risk(self, x, stance=None, meta_state=None):
        k = self._key(x, stance, meta_state)
        entry = self.stats.get(k)
        if not entry:
            return 0.5, 0.1
        overload = entry.get("overload", 0)
        stable = entry.get("stable", 0)
        total = overload + stable
        if total == 0:
            return 0.5, 0.1
        p = overload / total
        conf = clamp(total / 50.0, 0.0, 1.0)
        return p, conf


class WaterPhysicsEngine:
    def pressure_risk(self, x):
        load = x[0] if len(x) > 0 else 0.5
        trend = x[1] if len(x) > 1 else 0.0
        turb = x[2] if len(x) > 2 else 0.0
        p = clamp(0.5 * load + 0.3 * abs(trend) + 0.2 * turb, 0.0, 1.0)
        conf = clamp(1.0 - abs(turb - 0.5), 0.0, 1.0)
        return p, conf

    def turbulence_profile(self, history: List[float]) -> Dict[str, float]:
        if len(history) < 2:
            return {"turbulence": 0.0, "smoothness": 1.0}
        mean = sum(history) / len(history)
        var = sum((v - mean) ** 2 for v in history) / len(history)
        turb = clamp(var, 0.0, 1.0)
        return {"turbulence": turb, "smoothness": 1.0 - turb}


class ReinforcementEngine:
    def __init__(self):
        self.q_meta: Dict[str, float] = {
            "Sentinel": 0.5,
            "Hyper-Flow": 0.5,
            "Deep-Dream": 0.5,
            "Recovery-Flow": 0.5,
        }
        self.alpha = 0.1
        self.gamma = 0.9

    def reward_meta_state(self, meta_state: str, reward: float):
        old = self.q_meta.get(meta_state, 0.5)
        self.q_meta[meta_state] = old + self.alpha * (reward - old)

    def best_meta_state(self) -> str:
        return max(self.q_meta.items(), key=lambda kv: kv[1])[0]

# ============================================================
# ReplicaNPU
# ============================================================

class ReplicaNPU:
    def __init__(self, cores=16, frequency_ghz=1.5):
        self.cores = cores
        self.frequency_ghz = frequency_ghz
        self.heads: Dict[str, Dict[str, Any]] = {}
        self.plasticity = 1.0
        self.energy = 0.0
        self.cycles = 0
        self.memory = deque(maxlen=256)
        self.symbolic_bias: Dict[str, float] = {}
        self.model_integrity = 1.0
        self.integrity_threshold = 0.4
        self.frozen = False

        self.pattern_memory = PatternMemory()
        self.water_engine = WaterPhysicsEngine()
        self.rl = ReinforcementEngine()

        self.meta_state = "Sentinel"
        self.meta_momentum = 0.0  # inertia for transitions

    def mac(self, a, b):
        return a * b

    def add_head(self, name, input_dim, lr=0.01, risk=1.0, organ=None):
        self.heads[name] = {
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "organ": organ,
            "history": deque(maxlen=32),
        }

    def _symbolic_modulation(self, name):
        return self.symbolic_bias.get(name, 0.0)

    def _predict_head(self, head, x, name):
        y = 0.0
        for i in range(len(x)):
            y += self.mac(x[i], head["w"][i])
        y += head["b"]
        y += self._symbolic_modulation(name)
        head["history"].append(y)
        self.memory.append(y)
        return y

    def predict(self, x):
        preds = {}
        for name, head in self.heads.items():
            preds[name] = self._predict_head(head, x, name)
        return preds

    def best_guess(self, x, stance=None):
        preds = self.predict(x)
        engines = []

        for name, y in preds.items():
            conf = self.confidence(name)
            engines.append((f"head:{name}", y, conf))

        bern_p, bern_conf = self.pattern_memory.bernoulli_risk(
            x, stance=stance, meta_state=self.meta_state
        )
        engines.append(("bernoulli", bern_p, bern_conf))

        water_p, water_conf = self.water_engine.pressure_risk(x)
        engines.append(("water", water_p, water_conf))

        num = 0.0
        den = 0.0
        reasoning: Dict[str, Dict[str, float]] = {}
        for name, p, c in engines:
            w = max(0.01, c)
            num += p * w
            den += w
            reasoning[name] = {"p": p, "conf": c, "weight": w}
        best = num / den if den > 0 else 0.5
        meta_conf = self._meta_confidence(engines)
        self._update_meta_state(best, meta_conf)
        return best, meta_conf, reasoning

    def _meta_confidence(self, engines):
        ps = [p for _, p, _ in engines]
        if len(ps) < 2:
            base = 0.5
        else:
            m = sum(ps) / len(ps)
            var = sum((v - m) ** 2 for v in ps) / len(ps)
            base = clamp(1.0 - var, 0.0, 1.0)
        return clamp(0.5 * base + 0.5 * self.model_integrity, 0.0, 1.0)

    def _update_meta_state(self, best, meta_conf):
        # reward: low risk + high confidence
        reward = clamp(1.0 - best, 0.0, 1.0) * meta_conf
        self.rl.reward_meta_state(self.meta_state, reward)

        target = self.rl.best_meta_state()

        # momentum: avoid instant flips
        if target != self.meta_state:
            self.meta_momentum += 0.1
            if self.meta_momentum > 0.7:
                self.meta_state = target
                self.meta_momentum = 0.0
        else:
            self.meta_momentum = max(0.0, self.meta_momentum - 0.05)

        # integrity override
        if self.model_integrity < 0.5:
            self.meta_state = "Recovery-Flow"

    def learn(self, x, targets, outcome=None, stance=None):
        if self.frozen:
            return {}
        errors = {}
        for name, target in targets.items():
            head = self.heads[name]
            pred = self._predict_head(head, x, name)
            error = target - pred
            weighted_error = (
                error * head["risk"] * self.plasticity * self.model_integrity
            )
            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted_error * x[i]
                self.cycles += 1
            head["b"] += head["lr"] * weighted_error
            self.energy += 0.005
            errors[name] = error
        if outcome in ("overload", "stable", "wins"):
            self.pattern_memory.record(x, outcome, stance=stance, meta_state=self.meta_state)
        return errors

    def confidence(self, name):
        h = self.heads[name]["history"]
        if len(h) < 2:
            return 0.5
        mean = sum(h) / len(h)
        var = sum((v - mean) ** 2 for v in h) / len(h)
        return clamp(1.0 - var, 0.0, 1.0)

    def check_integrity(self, external_integrity=1.0):
        self.model_integrity = external_integrity
        self.frozen = self.model_integrity < self.integrity_threshold

    def micro_recovery(self, rate=0.01):
        self.plasticity = min(1.0, self.plasticity + rate)

    def set_symbolic_bias(self, name, value):
        self.symbolic_bias[name] = value

    def save_state(self, path):
        state = {
            "heads": self.heads,
            "plasticity": self.plasticity,
            "energy": self.energy,
            "cycles": self.cycles,
            "pattern_memory": self.pattern_memory.stats,
            "meta_state": self.meta_state,
            "q_meta": self.rl.q_meta,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load_state(self, path):
        with open(path, "r") as f:
            state = json.load(f)
        self.heads = state["heads"]
        self.plasticity = state["plasticity"]
        self.energy = state["energy"]
        self.cycles = state["cycles"]
        self.pattern_memory.stats = state.get("pattern_memory", {})
        self.meta_state = state.get("meta_state", "Sentinel")
        self.rl.q_meta = state.get("q_meta", self.rl.q_meta)

    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9)
        return {
            "cores": self.cores,
            "cycles": self.cycles,
            "estimated_time_sec": time_sec,
            "energy_units": round(self.energy, 6),
            "plasticity": round(self.plasticity, 3),
            "integrity": round(self.model_integrity, 3),
            "frozen": self.frozen,
            "meta_state": self.meta_state,
            "q_meta": self.rl.q_meta,
            "confidence": {k: round(self.confidence(k), 3) for k in self.heads},
        }

# ============================================================
# Movidius / ONNX stub
# ============================================================

class MovidiusInferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session = None
        if model_path and ort is not None:
            self.session = ort.InferenceSession(model_path)

    def predict_risk(self, features: Dict[str, float]):
        if self.session is None:
            return 0.5, 0.1
        import numpy as np
        keys = ["load_norm", "trend", "variance", "turbulence"]
        x = np.array([[features.get(k, 0.0) for k in keys]], dtype=np.float32)
        out = self.session.run(None, {"input": x})[0]
        p = float(out[0])
        return clamp(p, 0.0, 1.0), 0.7

    def train_offline(self, dataset_path: str, output_model_path: str):
        # offline training hook
        pass

# ============================================================
# SelfIntegrityOrgan
# ============================================================

class SelfIntegrityOrgan:
    def __init__(self, expected_sensors: List[str]):
        self.expected_sensors = set(expected_sensors)
        self.last_seen: Dict[str, float] = {}
        self.status = {
            "missing": set(),
            "stale": set(),
            "integrity_score": 1.0,
        }

    def mark_seen(self, sensor_name: str):
        self.last_seen[sensor_name] = time.time()

    def evaluate(self):
        now = time.time()
        missing = set()
        stale = set()
        for s in self.expected_sensors:
            if s not in self.last_seen:
                missing.add(s)
            else:
                if now - self.last_seen[s] > 10.0:
                    stale.add(s)
        total = len(self.expected_sensors)
        bad = len(missing) + len(stale)
        score = clamp(1.0 - bad / max(1, total), 0.0, 1.0)
        self.status["missing"] = missing
        self.status["stale"] = stale
        self.status["integrity_score"] = score
        return self.status

# ============================================================
# UIAutomationOrgan (stub)
# ============================================================

class UIAutomationOrgan:
    def __init__(self, brain, poll_interval=1.0):
        self.brain = brain
        self.poll_interval = poll_interval
        self.last_snapshot = None
        self.last_tick = 0.0

    def snapshot_ui(self):
        # stub – replace with real UIAutomation calls
        return {
            "active_window_title": "demo",
            "active_process": "apex_sentinel.exe",
            "has_error_dialog": False,
        }

    def tick(self):
        now = time.time()
        if now - self.last_tick < self.poll_interval:
            return
        self.last_tick = now
        snap = self.snapshot_ui()
        delta = {}
        if self.last_snapshot is not None:
            for k in snap:
                if self.last_snapshot.get(k) != snap.get(k):
                    delta[k] = (self.last_snapshot.get(k), snap.get(k))
        self.last_snapshot = snap
        self.brain.feed_ui_signal(snap, delta)

# ============================================================
# ETW / Kernel hooks (stub)
# ============================================================

class ETWKernelSensor:
    def __init__(self, brain):
        self.brain = brain

    def poll(self):
        # stub – here you'd read ETW events and feed into brain/swarm
        pass

# ============================================================
# HybridBrain
# ============================================================

class HybridBrain:
    def __init__(self):
        self.npu = ReplicaNPU()
        self.npu.add_head("short", 3, lr=0.05, risk=1.5, organ="brain")
        self.npu.add_head("medium", 3, lr=0.03, risk=1.0, organ="cortex")
        self.npu.add_head("long", 3, lr=0.02, risk=0.7, organ="planner")

        self.integrity_organ = SelfIntegrityOrgan(
            expected_sensors=["cpu", "mem", "disk", "net"]
        )
        self.ui_organ = UIAutomationOrgan(self)
        self.etw_sensor = ETWKernelSensor(self)

        self.stance = "Neutral"
        self.last_best_guess = 0.5
        self.last_meta_conf = 0.5
        self.last_reasoning: Dict[str, Any] = {}
        self.last_features: List[float] = [0.0, 0.0, 0.0]

    def feed_ui_signal(self, snapshot, delta):
        # could influence stance/meta_state later
        pass

    def tick(self, sensors: Dict[str, float]):
        x = [
            sensors.get("load_norm", 0.5),
            sensors.get("trend", 0.0),
            sensors.get("turbulence", 0.0),
        ]
        self.last_features = x

        for s in ["cpu", "mem", "disk", "net"]:
            if s in sensors:
                self.integrity_organ.mark_seen(s)
        integ = self.integrity_organ.evaluate()
        self.npu.check_integrity(external_integrity=integ["integrity_score"])

        best, meta_conf, reasoning = self.npu.best_guess(x, stance=self.stance)
        self.last_best_guess = best
        self.last_meta_conf = meta_conf
        self.last_reasoning = reasoning

        return best, meta_conf, reasoning, integ

    def stats(self):
        s = self.npu.stats()
        s["integrity_score"] = self.integrity_organ.status["integrity_score"]
        return s

    def to_dict(self):
        return {
            "npu": {
                "state": {
                    "heads": self.npu.heads,
                    "pattern_memory": self.npu.pattern_memory.stats,
                    "meta_state": self.npu.meta_state,
                    "q_meta": self.npu.rl.q_meta,
                }
            },
            "stance": self.stance,
        }

# ============================================================
# Swarm / Queen (with simple distance metric)
# ============================================================

@dataclass
class Cluster:
    cluster_id: str
    type: str
    version: int
    fingerprint_ids: List[str] = field(default_factory=list)
    aggregate_features: Dict[str, Any] = field(default_factory=dict)
    supporting_nodes: Dict[str, int] = field(default_factory=dict)
    swarm_confidence: float = 0.0
    last_updated: str = field(default_factory=now_utc_iso)

@dataclass
class NodeReputation:
    node_id: str
    reputation_score: float = 0.5

@dataclass
class SwarmState:
    clusters: Dict[str, Cluster] = field(default_factory=dict)
    reputations: Dict[str, NodeReputation] = field(default_factory=dict)

def fingerprint_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    # simple numeric + categorical distance
    keys = set(a.keys()) | set(b.keys())
    dist = 0.0
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            dist += abs(va - vb)
        else:
            if va != vb:
                dist += 1.0
    return dist

def derive_orders_for_cluster(cluster: Cluster) -> Dict[str, Any]:
    sc = cluster.swarm_confidence
    if sc >= 0.9:
        severity = "critical"
        actions = ["isolate_entity", "block_ip", "kill_process", "escalate_human"]
    elif sc >= 0.7:
        severity = "high"
        actions = ["block_ip", "kill_process", "escalate_human"]
    elif sc >= 0.4:
        severity = "medium"
        actions = ["increase_logging", "temporary_quarantine"]
    elif sc > 0.0:
        severity = "low"
        actions = ["monitor", "tag_suspicious"]
    else:
        severity = "unknown"
        actions = []
    return {
        "cluster_id": cluster.cluster_id,
        "severity": severity,
        "swarm_confidence": sc,
        "aggregate_features": cluster.aggregate_features,
        "actions": actions,
        "supporting_nodes": list(cluster.supporting_nodes.keys()),
        "last_updated": cluster.last_updated,
    }

@dataclass
class AttackNarrative:
    narrative_id: str
    entities: List[str]
    clusters: List[str]
    chains: List[str]
    severity: str
    global_score: float
    created_at: str
    last_updated: str

class QueenConsensus:
    def __init__(self, window: int = 180):
        self.window = window
        self.global_events: deque[Tuple[float, str, Dict[str, Any]]] = deque()
        self.narratives: Dict[str, AttackNarrative] = {}

    def update_from_cluster_orders(self, orders: List[Dict[str, Any]]):
        now = time.time()
        for order in orders:
            ent = order["aggregate_features"].get("dominant_process_name") or \
                  order["aggregate_features"].get("dominant_ip") or "unknown"
            evt = {
                "entity": ent,
                "cluster_id": order["cluster_id"],
                "severity": order["severity"],
                "swarm_confidence": order["swarm_confidence"],
            }
            self.global_events.append((now, "cluster_order", evt))
        self._cleanup(now)
        self.reconstruct_narratives()

    def update_from_best_guess(self, node_id: str, best: float, meta_conf: float, reasoning: Dict[str, Any]):
        now = time.time()
        evt = {
            "node": node_id,
            "best": best,
            "meta_conf": meta_conf,
            "reasoning": reasoning,
        }
        self.global_events.append((now, "best_guess", evt))
        self._cleanup(now)
        self.reconstruct_narratives()

    def _cleanup(self, now: float):
        cutoff = now - self.window
        while self.global_events and self.global_events[0][0] < cutoff:
            self.global_events.popleft()

    def reconstruct_narratives(self):
        by_entity: Dict[str, Dict[str, Any]] = {}
        for ts, etype, evt in self.global_events:
            if etype == "cluster_order":
                ent = evt["entity"]
                bucket = by_entity.setdefault(ent, {
                    "clusters": set(),
                    "best": [],
                    "meta_conf": [],
                    "severity": "unknown",
                    "score": 0.0,
                })
                bucket["clusters"].add(evt["cluster_id"])
                sev = evt["severity"]
                sev_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
                if sev_rank.get(sev, 0) > sev_rank.get(bucket["severity"], 0):
                    bucket["severity"] = sev
                bucket["score"] = max(bucket["score"], evt["swarm_confidence"])
            elif etype == "best_guess":
                ent = evt["node"]
                bucket = by_entity.setdefault(ent, {
                    "clusters": set(),
                    "best": [],
                    "meta_conf": [],
                    "severity": "unknown",
                    "score": 0.0,
                })
                bucket["best"].append(evt["best"])
                bucket["meta_conf"].append(evt["meta_conf"])
        self.narratives.clear()
        for ent, data in by_entity.items():
            nid = "nar-" + ent
            avg_best = sum(data["best"]) / len(data["best"]) if data["best"] else 0.0
            avg_meta = sum(data["meta_conf"]) / len(data["meta_conf"]) if data["meta_conf"] else 0.0
            nar = AttackNarrative(
                narrative_id=nid,
                entities=[ent],
                clusters=list(data["clusters"]),
                chains=[],
                severity=data["severity"],
                global_score=max(data["score"], avg_best),
                created_at=now_utc_iso(),
                last_updated=now_utc_iso(),
            )
            self.narratives[nid] = nar

    def get_active_narratives(self) -> List[AttackNarrative]:
        return list(self.narratives.values())

# ============================================================
# RebootMemoryManager
# ============================================================

class RebootMemoryManager:
    def __init__(self, brain: HybridBrain, organs: Dict[str, Any]):
        self.brain = brain
        self.organs = organs

    def snapshot_state(self) -> dict:
        return {
            "brain": self.brain.to_dict(),
            "organs": {name: getattr(org, "to_dict", lambda: {} )() for name, org in self.organs.items()},
        }

    def save_to_path(self, base_path: str) -> str:
        state = self.snapshot_state()
        path = Path(base_path)
        path.mkdir(parents=True, exist_ok=True)
        json_path = path / "organism_state.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        return str(json_path)

    def load_from_path(self, base_path: str) -> bool:
        json_path = Path(base_path) / "organism_state.json"
        if not json_path.exists():
            return False
        with json_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        # rehydrate brain/organs as needed (stub)
        return True

# ============================================================
# GUI: PredictionChart + Altered States panel
# ============================================================

class PredictionChart(QtWidgets.QWidget):
    def __init__(self, brain: HybridBrain, parent=None):
        super().__init__(parent)
        self.brain = brain
        self.setMinimumHeight(140)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        rect = self.rect()
        w = rect.width()
        h = rect.height()
        p.fillRect(rect, QtGui.QColor("#101010"))

        baseline = 0.5
        y_base = h - baseline * h
        p.setPen(QtGui.QPen(QtGui.QColor("#444444"), 1))
        p.drawLine(0, int(y_base), w, int(y_base))

        best = self.brain.last_best_guess
        y_best = h - best * h
        p.setPen(QtGui.QPen(QtGui.QColor("#ff00ff"), 2))
        p.drawLine(0, int(y_best), w, int(y_best))

        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        p.drawText(5, 15, f"Best-Guess: {best:.3f}")
        p.drawText(5, 30, f"Meta-Conf: {self.brain.last_meta_conf:.3f}")
        p.drawText(5, 45, f"Meta-State: {self.brain.npu.meta_state}")

class AlteredStatesPanel(QtWidgets.QWidget):
    def __init__(self, brain: HybridBrain, parent=None):
        super().__init__(parent)
        self.brain = brain
        layout = QtWidgets.QFormLayout(self)

        self.lbl_meta_state = QtWidgets.QLabel()
        self.lbl_q_meta = QtWidgets.QLabel()
        self.lbl_integrity = QtWidgets.QLabel()
        self.lbl_plasticity = QtWidgets.QLabel()

        layout.addRow("Meta-State:", self.lbl_meta_state)
        layout.addRow("Meta Q-Values:", self.lbl_q_meta)
        layout.addRow("Integrity Score:", self.lbl_integrity)
        layout.addRow("Plasticity:", self.lbl_plasticity)

    def refresh(self):
        s = self.brain.stats()
        self.lbl_meta_state.setText(s["meta_state"])
        self.lbl_q_meta.setText(json.dumps(s["q_meta"], indent=2))
        self.lbl_integrity.setText(f"{s['integrity_score']:.3f}")
        self.lbl_plasticity.setText(f"{s['plasticity']:.3f}")

# ============================================================
# MainWindow
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, brain: HybridBrain, swarm: SwarmState, queen: QueenConsensus):
        super().__init__()
        self.brain = brain
        self.swarm = swarm
        self.queen = queen

        self.setWindowTitle("APEX‑Sentinel OS / ChronoMind Nexus / Parallax Swarm Engine")
        self.resize(1200, 750)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Tab 1: Brain Cortex
        tab_brain = QtWidgets.QWidget()
        tabs.addTab(tab_brain, "Brain Cortex")

        v1 = QtWidgets.QVBoxLayout(tab_brain)
        self.chart = PredictionChart(brain)
        v1.addWidget(self.chart)

        self.txt_stats = QtWidgets.QTextEdit()
        self.txt_stats.setReadOnly(True)
        v1.addWidget(self.txt_stats)

        self.txt_reason = QtWidgets.QTextEdit()
        self.txt_reason.setReadOnly(True)
        v1.addWidget(self.txt_reason)

        # Tab 2: Swarm / Queen
        tab_swarm = QtWidgets.QWidget()
        tabs.addTab(tab_swarm, "Swarm / Queen")

        v2 = QtWidgets.QVBoxLayout(tab_swarm)
        self.lst_orders = QtWidgets.QTextEdit()
        self.lst_orders.setReadOnly(True)
        v2.addWidget(self.lst_orders)

        self.lst_narratives = QtWidgets.QTextEdit()
        self.lst_narratives.setReadOnly(True)
        v2.addWidget(self.lst_narratives)

        # Tab 3: Reboot Memory
        tab_reboot = QtWidgets.QWidget()
        tabs.addTab(tab_reboot, "Reboot Memory")

        v3 = QtWidgets.QVBoxLayout(tab_reboot)
        form = QtWidgets.QFormLayout()
        self.entry_reboot_path = QtWidgets.QLineEdit()
        form.addRow("SMB / UNC Path:", self.entry_reboot_path)
        v3.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_save_reboot = QtWidgets.QPushButton("Save Memory for Reboot")
        self.btn_load_reboot = QtWidgets.QPushButton("Load Memory from SMB")
        btn_row.addWidget(self.btn_save_reboot)
        btn_row.addWidget(self.btn_load_reboot)
        v3.addLayout(btn_row)

        self.chk_reboot_autoload = QtWidgets.QCheckBox("Load memory from SMB on startup")
        v3.addWidget(self.chk_reboot_autoload)

        self.lbl_reboot_status = QtWidgets.QLabel("Status: Ready")
        v3.addWidget(self.lbl_reboot_status)

        self.reboot_mgr = RebootMemoryManager(brain, {})

        self.btn_save_reboot.clicked.connect(self.cmd_save_reboot_memory)
        self.btn_load_reboot.clicked.connect(self.cmd_load_reboot_memory)

        # Tab 4: Altered States
        tab_states = QtWidgets.QWidget()
        tabs.addTab(tab_states, "Altered States")

        v4 = QtWidgets.QVBoxLayout(tab_states)
        self.states_panel = AlteredStatesPanel(brain)
        v4.addWidget(self.states_panel)

        # timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    def cmd_save_reboot_memory(self):
        path = self.entry_reboot_path.text().strip()
        if not path:
            self.lbl_reboot_status.setText("Status: No path set")
            return
        try:
            p = self.reboot_mgr.save_to_path(path)
            self.lbl_reboot_status.setText(f"Status: Saved to {p}")
        except Exception as e:
            self.lbl_reboot_status.setText(f"Status: Error: {e}")

    def cmd_load_reboot_memory(self):
        path = self.entry_reboot_path.text().strip()
        if not path:
            self.lbl_reboot_status.setText("Status: No path set")
            return
        ok = self.reboot_mgr.load_from_path(path)
        self.lbl_reboot_status.setText("Status: Loaded" if ok else "Status: No state found")

    def tick(self):
        # demo sensors
        t = time.time()
        sensors = {
            "cpu": 0.5 + 0.3 * math.sin(t / 10.0),
            "mem": 0.4,
            "disk": 0.3,
            "net": 0.2,
            "load_norm": 0.5 + 0.3 * math.sin(t / 15.0),
            "trend": 0.1 * math.cos(t / 20.0),
            "turbulence": 0.5 + 0.4 * math.sin(t / 7.0),
        }
        best, meta_conf, reasoning, integ = self.brain.tick(sensors)
        self.brain.ui_organ.tick()
        self.brain.etw_sensor.poll()

        # swarm/queen demo
        if "cluster-demo" not in self.swarm.clusters:
            self.swarm.clusters["cluster-demo"] = Cluster(
                cluster_id="cluster-demo",
                type="process",
                version=1,
                aggregate_features={"dominant_process_name": "demo.exe"},
                swarm_confidence=0.0,
            )
        c = self.swarm.clusters["cluster-demo"]
        c.swarm_confidence = best
        orders = [derive_orders_for_cluster(c)]
        self.queen.update_from_cluster_orders(orders)
        self.queen.update_from_best_guess("node-1", best, meta_conf, reasoning)

        self.refresh_brain_view()
        self.refresh_swarm_view()
        self.states_panel.refresh()
        self.chart.update()

    def refresh_brain_view(self):
        s = self.brain.stats()
        self.txt_stats.setPlainText(json.dumps(s, indent=2))

        lines = ["Reasoning Heatmap:"]
        for name, info in self.brain.last_reasoning.items():
            lines.append(
                f"{name:<12} p={info['p']:.3f} conf={info['conf']:.3f} w={info['weight']:.3f}"
            )
        self.txt_reason.setPlainText("\n".join(lines))

    def refresh_swarm_view(self):
        orders_text = []
        for cluster in self.swarm.clusters.values():
            o = derive_orders_for_cluster(cluster)
            orders_text.append(json.dumps(o, indent=2))
        self.lst_orders.setPlainText("\n\n".join(orders_text))

        nar_text = []
        for nar in self.queen.get_active_narratives():
            nar_text.append(json.dumps(asdict(nar), indent=2))
        self.lst_narratives.setPlainText("\n\n".join(nar_text))

# ============================================================
# Main
# ============================================================

def main():
    brain = HybridBrain()
    swarm = SwarmState()
    queen = QueenConsensus()

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(brain, swarm, queen)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
