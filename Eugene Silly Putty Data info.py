#!/usr/bin/env python3
# Full Borg Organism – Unified File with All Upgrades

import sys
import os
import platform
import importlib
import subprocess
import threading
import time
import hashlib
import zlib
import random
import json
from collections import deque
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# =========================
# Autoloader
# =========================

class BorgAutoloader:
    def __init__(self):
        self.required_libs = self._build_required_libs()
        self.loaded: Dict[str, object] = {}
        self.failed: Dict[str, str] = {}
        self.lock = threading.Lock()
        self.swarm_state: Dict[str, Dict] = {}

    def _build_required_libs(self) -> List[str]:
        base = [
            "numpy",
            "psutil",
            "pyyaml",
            "requests",
            "rich",
            "PySide6",
        ]
        system = platform.system().lower()
        if "windows" in system:
            base += ["wmi", "uiautomation"]
        elif "linux" in system:
            base += ["pyroute2"]
        elif "darwin" in system:
            base += []
        base += ["cupy"]  # best-effort GPU
        return base

    def install(self, package: str) -> bool:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception:
            return False

    def load(self, module_name: str):
        with self.lock:
            if module_name in self.loaded:
                return self.loaded[module_name]
        try:
            module = importlib.import_module(module_name)
            with self.lock:
                self.loaded[module_name] = module
            return module
        except ImportError:
            if self.install(module_name):
                try:
                    module = importlib.import_module(module_name)
                    with self.lock:
                        self.loaded[module_name] = module
                    return module
                except Exception:
                    with self.lock:
                        self.failed[module_name] = "import_failed_after_install"
            else:
                with self.lock:
                    self.failed[module_name] = "install_failed"
        return None

    def autoload(self):
        for lib in self.required_libs:
            self.load(lib)

    def status(self) -> Dict:
        with self.lock:
            return {
                "loaded": list(self.loaded.keys()),
                "failed": dict(self.failed),
                "swarm_state": dict(self.swarm_state),
            }

    def export_state(self) -> Dict:
        with self.lock:
            return {
                "loaded": list(self.loaded.keys()),
                "failed": dict(self.failed),
            }

    def import_state_from_swarm(self, node_id: str, state: Dict):
        with self.lock:
            self.swarm_state[node_id] = state


class AutoloaderWatchdog(threading.Thread):
    def __init__(self, autoloader: BorgAutoloader, interval: float = 30.0):
        super().__init__(daemon=True)
        self.autoloader = autoloader
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            self.autoloader.autoload()
            time.sleep(self.interval)

    def stop(self):
        self.running = False


# =========================
# Silly Putty Organism
# =========================

class SillyPuttyDataGatherer:
    META_STATES = ["hyper_flow", "sentinel", "recovery_flow", "deep_dream"]

    def __init__(self):
        self.mass = 1.0
        self.fingerprints: List[str] = []
        self.efficiency_gain = 0.0
        self.flow_shapes: Dict[str, str] = {}
        self.byte_hist: Dict[int, int] = {}
        self.total_bytes = 0

        self.pressure = 0.0
        self.flow_rate = 0.0
        self.turbulence = 0.0

        self.state = "baseline"
        self.meta_state = "sentinel"
        self.meta_target = "sentinel"
        self.meta_momentum = 0.0
        self.last_update_ts = time.time()

        # micro-patterns
        self.cpu_osc = 0.0
        self.mem_creep = 0.0
        self.disk_burst = 0.0
        self.net_pulse = 0.0

    def set_state(self, state: str):
        if state in ("baseline", "hypervigilant", "dreamlike"):
            self.state = state

    def set_meta_target(self, target: str):
        if target in self.META_STATES:
            self.meta_target = target

    def _classify_shape(self, data: bytes) -> str:
        length = len(data)
        entropy_est = len(set(data)) / max(length, 1)
        if entropy_est < 0.2:
            return "blob"
        elif entropy_est < 0.5:
            return "strand"
        elif entropy_est < 0.8:
            return "ripple"
        else:
            return "echo"

    def _update_histogram(self, data: bytes):
        for b in data:
            self.byte_hist[b] = self.byte_hist.get(b, 0) + 1
            self.total_bytes += 1

    def _bernoulli_rarity(self, data: bytes) -> float:
        if self.total_bytes == 0:
            return 0.5
        rarity = 1.0
        for b in set(data):
            p = self.byte_hist.get(b, 1) / self.total_bytes
            rarity *= max(0.01, 1.0 - p)
        return max(0.0, min(rarity, 1.0))

    def _infer_missing_details(self, data: bytes) -> Dict[str, Any]:
        text = ""
        try:
            text = data.decode(errors="ignore")
        except Exception:
            pass
        hints = []
        if "GET " in text or "POST " in text:
            hints.append("http_like")
        if b"\x00\x00" in data:
            hints.append("binary_padding")
        if len(text.strip()) == 0:
            hints.append("non_textual")
        return {
            "guessed_protocol": "http" if "http_like" in hints else "unknown",
            "structure_hints": hints,
        }

    def _update_water_physics(self, data: bytes):
        size = len(data)
        prev_pressure = self.pressure
        self.pressure = 0.8 * self.pressure + 0.2 * (size / 1024.0)
        self.flow_rate = 0.8 * self.flow_rate + 0.2 * 1.0
        self.turbulence = 0.7 * self.turbulence + 0.3 * abs(self.pressure - prev_pressure)

        # micro-patterns (synthetic)
        self.cpu_osc = 0.8 * self.cpu_osc + 0.2 * random.random()
        self.mem_creep = 0.8 * self.mem_creep + 0.2 * random.random()
        self.disk_burst = 0.8 * self.disk_burst + 0.2 * random.random()
        self.net_pulse = 0.8 * self.net_pulse + 0.2 * random.random()

    def _state_sensitivity_factor(self) -> float:
        if self.state == "baseline":
            return 1.0
        if self.state == "hypervigilant":
            return 1.5
        if self.state == "dreamlike":
            return 0.7
        return 1.0

    def _update_meta_state(self):
        now = time.time()
        self.last_update_ts = now

        desired = 0.0
        if self.meta_target == "hyper_flow":
            desired = 1.0
        elif self.meta_target == "sentinel":
            desired = 0.0
        elif self.meta_target == "recovery_flow":
            desired = -0.5
        elif self.meta_target == "deep_dream":
            desired = -1.0

        alpha = 0.1
        self.meta_momentum += alpha * (desired - self.meta_momentum)
        self.meta_momentum = max(-1.0, min(1.0, self.meta_momentum))

        if self.meta_state == "hyper_flow" and self.turbulence > 0.5:
            self.meta_momentum -= 0.05
        if self.meta_state == "sentinel" and self.flow_rate < 0.2:
            self.meta_momentum -= 0.05
        if self.meta_state == "recovery_flow" and self.turbulence < 0.2:
            self.meta_momentum -= 0.05

        m = self.meta_momentum
        if m > 0.5:
            self.meta_state = "hyper_flow"
        elif 0.1 < m <= 0.5:
            self.meta_state = "sentinel"
        elif -0.4 < m <= 0.1:
            self.meta_state = "recovery_flow"
        else:
            self.meta_state = "deep_dream"

    def absorb(self, data: bytes):
        original_size = len(data)
        compressed = zlib.compress(data)
        compressed_size = len(compressed)

        gain = (original_size - compressed_size) / max(original_size, 1)
        self.efficiency_gain += gain

        fp = hashlib.sha256(data).hexdigest()[:16]
        self.fingerprints.append(fp)

        shape = self._classify_shape(data)
        self.flow_shapes[fp] = shape

        self._update_histogram(data)
        rarity = self._bernoulli_rarity(data)
        inferred = self._infer_missing_details(data)
        self._update_water_physics(data)
        self._update_meta_state()

        sensitivity = self._state_sensitivity_factor()
        rarity_boost = rarity * sensitivity

        self.mass += max(gain * 2 + rarity_boost * 0.5, 0.01)

        return {
            "fingerprint": fp,
            "shape": shape,
            "rarity": rarity,
            "inferred": inferred,
            "pressure": self.pressure,
            "flow_rate": self.flow_rate,
            "turbulence": self.turbulence,
            "state": self.state,
            "meta_state": self.meta_state,
        }

    def anomaly_density(self) -> float:
        unique_shapes = len(set(self.flow_shapes.values()))
        total_flows = max(len(self.fingerprints), 1)
        return unique_shapes / total_flows

    def report(self) -> Dict:
        return {
            "mass": round(self.mass, 3),
            "efficiency_gain": round(self.efficiency_gain, 3),
            "unique_flows": len(self.fingerprints),
            "flow_fingerprints": self.fingerprints[-5:],
            "shapes": {fp: self.flow_shapes[fp] for fp in self.fingerprints[-5:]},
            "anomaly_density": round(self.anomaly_density(), 3),
            "pressure": round(self.pressure, 3),
            "flow_rate": round(self.flow_rate, 3),
            "turbulence": round(self.turbulence, 3),
            "state": self.state,
            "meta_state": self.meta_state,
            "cpu_osc": round(self.cpu_osc, 3),
            "mem_creep": round(self.mem_creep, 3),
            "disk_burst": round(self.disk_burst, 3),
            "net_pulse": round(self.net_pulse, 3),
        }


# =========================
# Organs
# =========================

class BaseOrgan:
    def __init__(self, name: str):
        self.name = name
        self.health = 1.0
        self.risk = 0.0
        self.integrity = 1.0
        self.last_update = time.time()

    def update(self):
        self.health = max(0.0, min(1.0, self.health + random.uniform(-0.02, 0.02)))
        self.risk = max(0.0, min(1.0, self.risk + random.uniform(-0.02, 0.02)))
        self.integrity = max(0.0, min(1.0, self.integrity + random.uniform(-0.01, 0.01)))
        self.last_update = time.time()

    def micro_recovery(self):
        self.health = min(1.0, self.health + 0.02)
        self.risk = max(0.0, self.risk - 0.02)

    def snapshot(self) -> Dict:
        return {
            "name": self.name,
            "health": round(self.health, 3),
            "risk": round(self.risk, 3),
            "integrity": round(self.integrity, 3),
        }


class DeepRamOrgan(BaseOrgan):
    def micro_recovery(self):
        self.health = min(1.0, self.health + 0.03)
        self.risk = max(0.0, self.risk - 0.03)


class BackupEngineOrgan(BaseOrgan):
    def micro_recovery(self):
        self.integrity = min(1.0, self.integrity + 0.04)


class NetworkWatcherOrgan(BaseOrgan):
    def micro_recovery(self):
        self.risk = max(0.0, self.risk - 0.03)


class GPUCacheOrgan(BaseOrgan):
    pass


class ThermalOrgan(BaseOrgan):
    def micro_recovery(self):
        self.risk = max(0.0, self.risk - 0.04)


class DiskOrgan(BaseOrgan):
    def micro_recovery(self):
        self.risk = max(0.0, self.risk - 0.03)


class VRAMOrgan(BaseOrgan):
    pass


class AICoachOrgan(BaseOrgan):
    pass


class SwarmNodeOrgan(BaseOrgan):
    pass


class Back4BloodAnalyzer(BaseOrgan):
    pass


class SelfIntegrityOrgan(BaseOrgan):
    def update(self, organs: List[BaseOrgan]):
        super().update()
        inconsistencies = 0
        now = time.time()
        for o in organs:
            if now - o.last_update > 10:
                inconsistencies += 1
            if abs(o.health - o.integrity) > 0.5:
                inconsistencies += 1
        drop = 0.05 * inconsistencies
        self.integrity = max(0.0, self.integrity - drop)


# =========================
# Queens & Chains
# =========================

class RealTimeQueen:
    def __init__(self):
        self.nodes: Dict[str, List[Dict]] = {}
        self.lock = threading.Lock()

    def update(self, node: str, events: List[Dict]):
        with self.lock:
            self.nodes[node] = events

    def global_risk(self) -> Dict[str, float]:
        risk: Dict[str, float] = {}
        with self.lock:
            for node, evts in self.nodes.items():
                for e in evts:
                    ent = e.get("entity")
                    score = e.get("score", 0.0)
                    if ent is None:
                        continue
                    risk[ent] = risk.get(ent, 0.0) + score
        return {k: v for k, v in risk.items() if v > 1.5}


class AttackChainEngine:
    def __init__(self, window: int = 120):
        self.events = deque()
        self.window = window

    def add_event(self, event_type: str, data: Dict):
        now = time.time()
        self.events.append((now, event_type, data))
        self._cleanup(now)

    def _cleanup(self, now: float):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def detect(self) -> List[tuple]:
        types = [e[1] for e in self.events]
        chains = []
        if all(x in types for x in ["proc_spawn", "powershell", "net_connect"]):
            chains.append(("LOLBIN_ATTACK", 0.9))
        if types.count("proc_spawn") > 5 and "net_connect" in types:
            chains.append(("PROCESS_STORM", 0.8))
        if "file_mod" in types and "net_connect" in types:
            chains.append(("PERSISTENCE_EXFIL", 0.85))
        if "proc_start" in types and "net_conn" in types and "file_mod" in types:
            chains.append(("FULL_ATTACK_CHAIN", 0.95))
        if types.count("proc_start") > 10:
            chains.append(("SPAWN_STORM", 0.8))
        if "net_conn" in types and "powershell" in str(types):
            chains.append(("LOLBIN_BEACON", 0.9))
        return chains


class BorgQueen:
    def __init__(self):
        self.total_efficiency = 0.0
        self.node_reports: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    def assimilate(self, node_id: str, report: Dict):
        with self.lock:
            self.total_efficiency += report.get("efficiency_gain", 0.0)
            self.node_reports[node_id] = report

    def status(self) -> Dict:
        with self.lock:
            return {
                "efficiency": round(self.total_efficiency, 3),
                "nodes": len(self.node_reports),
                "state": "optimizing",
            }

    def export_swarm_view(self) -> Dict:
        with self.lock:
            return {
                "queen_efficiency": self.total_efficiency,
                "node_reports": dict(self.node_reports),
            }


# =========================
# Event Bus
# =========================

class SecEvent:
    def __init__(self, etype: str, entity: str, meta: Optional[Dict] = None):
        self.ts = time.time()
        self.type = etype
        self.entity = entity
        self.meta = meta or {}


class EventBus:
    def __init__(self):
        self.subscribers = []
        self.queue = deque()
        self.running = False

    def publish(self, event: SecEvent):
        self.queue.append(event)

    def subscribe(self, fn):
        self.subscribers.append(fn)

    def run(self):
        self.running = True
        while self.running:
            if self.queue:
                evt = self.queue.popleft()
                for fn in self.subscribers:
                    fn(evt)
            time.sleep(0.01)

    def stop(self):
        self.running = False


# =========================
# Adaptive Codex Mutation
# =========================

class AdaptiveCodexMutation:
    def __init__(self):
        self.retention_seconds = 3600.0
        self.ghost_sensitive = True
        self.phantom_node_enabled = False
        self.mutation_log: List[Dict[str, Any]] = []
        self.last_ghost_detected: Optional[float] = None

    def detect_ghost_sync(self, risk: float, health: float, turbulence: float) -> bool:
        ghost = False
        if turbulence > 0.6 and health < 0.5 and 0.3 < risk < 0.7:
            ghost = True
        if ghost:
            self.last_ghost_detected = time.time()
        return ghost

    def mutate_rules(self, reason: str):
        before = {
            "retention_seconds": self.retention_seconds,
            "ghost_sensitive": self.ghost_sensitive,
            "phantom_node_enabled": self.phantom_node_enabled,
        }
        if self.last_ghost_detected and time.time() - self.last_ghost_detected < 120:
            self.retention_seconds = max(300.0, self.retention_seconds * 0.7)
            self.phantom_node_enabled = True
        else:
            self.retention_seconds = min(7200.0, self.retention_seconds * 1.05)
        self.ghost_sensitive = self.retention_seconds < 2000.0
        after = {
            "retention_seconds": self.retention_seconds,
            "ghost_sensitive": self.ghost_sensitive,
            "phantom_node_enabled": self.phantom_node_enabled,
        }
        self.mutation_log.append({
            "ts": time.time(),
            "reason": reason,
            "before": before,
            "after": after,
        })

    def snapshot(self) -> Dict[str, Any]:
        return {
            "retention_seconds": round(self.retention_seconds, 1),
            "ghost_sensitive": self.ghost_sensitive,
            "phantom_node_enabled": self.phantom_node_enabled,
            "mutations": len(self.mutation_log),
        }

    def export_codex(self) -> Dict[str, Any]:
        return {
            "retention_seconds": self.retention_seconds,
            "ghost_sensitive": self.ghost_sensitive,
            "phantom_node_enabled": self.phantom_node_enabled,
        }


# =========================
# Self-Rewriting Agent
# =========================

def _load_array_backend():
    try:
        import cupy as cp
        return cp, True
    except Exception:
        import numpy as cp
        return cp, False


class SelfRewritingAgent:
    def __init__(self):
        self.cp, self.gpu = _load_array_backend()
        self.agent_weights = self.cp.array([0.6, -0.8, -0.3], dtype=float)
        self.mutation_log: List[Dict[str, Any]] = []

    def step(self, risk: float, health: float, ghost_sync: bool):
        scale = 0.01 + 0.04 * risk
        if ghost_sync:
            scale *= 2.0
        noise = self.cp.random.uniform(-scale, scale, size=self.agent_weights.shape)
        self.agent_weights = self.agent_weights + noise
        self.mutation_log.append({
            "ts": time.time(),
            "risk": risk,
            "health": health,
            "ghost_sync": ghost_sync,
            "delta": [float(x) for x in noise.tolist()],
        })

    def snapshot(self) -> Dict[str, Any]:
        return {
            "weights": [float(x) for x in self.agent_weights.tolist()],
            "mutations": len(self.mutation_log),
            "backend": "CuPy" if self.gpu else "NumPy",
        }


# =========================
# Autonomous Cipher Engine
# =========================

class AutonomousCipherEngine:
    POSTURES = ["RELAXED", "GUARDED", "LOCKDOWN"]

    def __init__(self):
        self.posture = "RELAXED"
        self.rotation_interval = 600.0
        self.last_rotation = time.time()
        self.current_key_id = 0

    def _rotate_key(self):
        self.current_key_id += 1
        self.last_rotation = time.time()

    def tick(self, risk: float, integrity: float):
        if risk > 0.7 or integrity < 0.4:
            self.posture = "LOCKDOWN"
            self.rotation_interval = 60.0
        elif risk > 0.4 or integrity < 0.7:
            self.posture = "GUARDED"
            self.rotation_interval = 180.0
        else:
            self.posture = "RELAXED"
            self.rotation_interval = 600.0
        if time.time() - self.last_rotation >= self.rotation_interval:
            self._rotate_key()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "posture": self.posture,
            "rotation_interval": int(self.rotation_interval),
            "current_key_id": self.current_key_id,
            "seconds_since_rotation": int(time.time() - self.last_rotation),
        }


# =========================
# Brain Autonomy (Appetite / Thresholds / Horizon / Dampening)
# =========================

class BrainAutonomy:
    def __init__(self, chain_engine: AttackChainEngine):
        self.appetite = 1.0
        self.threshold_risk = 1.5
        self.horizon = 120
        self.dampening = 0.8
        self.cache_aggressiveness = 0.5
        self.thread_expansion = 1.0
        self.success_score = 0.0
        self.last_calibration = datetime.utcnow()
        self.chain_engine = chain_engine

    def record_outcome(self, true_positive: bool, false_positive: bool):
        if true_positive:
            self.success_score += 1.0
        if false_positive:
            self.success_score -= 0.5

    def auto_tune(self):
        s = self.success_score
        self.appetite = max(0.5, min(2.0, 1.0 + 0.1 * s))
        self.threshold_risk = max(1.0, min(3.0, 1.5 - 0.05 * s))
        self.horizon = int(max(60, min(600, 120 + 5 * s)))
        self.dampening = max(0.5, min(0.95, 0.8 + 0.01 * s))
        self.cache_aggressiveness = max(0.2, min(0.9, 0.5 + 0.05 * s))
        self.thread_expansion = max(0.5, min(3.0, 1.0 + 0.1 * s))
        self.chain_engine.window = self.horizon

    def auto_calibrate_if_due(self, anomaly_density: float, pressure: float):
        now = datetime.utcnow()
        if now - self.last_calibration >= timedelta(hours=24):
            self.threshold_risk = max(1.0, 1.5 + anomaly_density - pressure * 0.2)
            self.success_score *= 0.5
            self.appetite = max(0.5, min(2.0, 1.0 + anomaly_density))
            self.last_calibration = now

    def summary(self) -> Dict:
        return {
            "appetite": round(self.appetite, 3),
            "threshold_risk": round(self.threshold_risk, 3),
            "horizon": self.horizon,
            "dampening": round(self.dampening, 3),
            "cache_aggressiveness": round(self.cache_aggressiveness, 3),
            "thread_expansion": round(self.thread_expansion, 3),
            "success_score": round(self.success_score, 3),
        }


# =========================
# BrainCore: All Intelligence Layers
# =========================

class BrainCore:
    MISSIONS = ["AUTO", "PROTECT", "STABILITY", "LEARN", "OPTIMIZE"]
    ENVIRONMENTS = ["CALM", "TENSE", "DANGER"]
    HORIZONS = ["SHORT", "MEDIUM", "LONG"]

    REGIMES = ["STABLE", "CHAOTIC", "RISING", "COOLING"]

    META_STATES = ["Hyper-Flow", "Deep-Dream", "Sentinel", "Recovery-Flow"]

    def __init__(self, chain_engine: AttackChainEngine):
        # situational awareness
        self.mission = "AUTO"
        self.environment = "CALM"
        self.opportunity_score = 0.3
        self.risk_score = 0.2
        self.anticipation = "Stable window"

        self.risk_tolerance = 0.5
        self.opportunity_bias = 0.5

        # predictive intelligence
        self.prediction_horizon = "MEDIUM"
        self.anomaly_risk = 0.2
        self.drive_risk = 0.2
        self.hive_risk = 0.2
        self.collective_health = 0.8
        self.health_trend = 0.0

        self.anomaly_sensitivity = 0.5
        self.collective_weighting = 0.5

        # multi-horizon predictions
        self.predictions = {
            "1s": 0.2,
            "5s": 0.2,
            "30s": 0.2,
            "120s": 0.2,
        }

        # regime + meta-confidence
        self.regime = "STABLE"
        self.meta_confidence = 0.7

        # pattern memory / behavioral fingerprinting
        self.pattern_memory: List[Dict[str, Any]] = []
        self.mode_memory: Dict[str, Dict[str, float]] = {}

        # meta-state (awareness upgrade)
        self.meta_state = "Sentinel"

        # predictive dampening
        self.short_damp = 0.0
        self.long_damp = 0.0

        # reasoning heatmap
        self.reasoning_heatmap: Dict[str, float] = {}

        # subsystems
        self.codex = AdaptiveCodexMutation()
        self.agent = SelfRewritingAgent()
        self.cipher = AutonomousCipherEngine()
        self.autonomy = BrainAutonomy(chain_engine)

        self._tick_count = 0
        self._risk_history = deque(maxlen=60)

    # ---- Controls ----

    def set_mission(self, mission: str):
        if mission in self.MISSIONS:
            self.mission = mission

    def set_environment(self, env: str):
        if env in self.ENVIRONMENTS:
            self.environment = env

    def adjust_risk_tolerance(self, delta: float):
        self.risk_tolerance = max(0.0, min(1.0, self.risk_tolerance + delta))

    def adjust_opportunity_bias(self, delta: float):
        self.opportunity_bias = max(0.0, min(1.0, self.opportunity_bias + delta))

    def set_horizon(self, horizon: str):
        if horizon in self.HORIZONS:
            self.prediction_horizon = horizon

    def adjust_anomaly_sensitivity(self, delta: float):
        self.anomaly_sensitivity = max(0.0, min(1.0, self.anomaly_sensitivity + delta))

    def adjust_collective_weighting(self, delta: float):
        self.collective_weighting = max(0.0, min(1.0, self.collective_weighting + delta))

    # ---- Internal helpers ----

    def _update_regime(self):
        if len(self._risk_history) < 5:
            self.regime = "STABLE"
            return
        vals = list(self._risk_history)
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        trend = vals[-1] - vals[0]

        if var < 0.01 and abs(trend) < 0.05:
            self.regime = "STABLE"
        elif var > 0.05 and abs(trend) < 0.1:
            self.regime = "CHAOTIC"
        elif trend > 0.1:
            self.regime = "RISING"
        elif trend < -0.1:
            self.regime = "COOLING"
        else:
            self.regime = "STABLE"

    def _multi_horizon_forecast(self, base_risk: float, turbulence: float):
        # simple regime-based scaling
        if self.regime == "STABLE":
            factor = 0.8
        elif self.regime == "CHAOTIC":
            factor = 1.2
        elif self.regime == "RISING":
            factor = 1.4
        else:  # COOLING
            factor = 0.7

        # horizon-specific
        self.predictions["1s"] = max(0.0, min(1.0, base_risk * factor * (1.0 + 0.2 * turbulence)))
        self.predictions["5s"] = max(0.0, min(1.0, base_risk * factor * (1.1 + 0.3 * turbulence)))
        self.predictions["30s"] = max(0.0, min(1.0, base_risk * factor * (1.2 + 0.4 * turbulence)))
        self.predictions["120s"] = max(0.0, min(1.0, base_risk * factor * (1.3 + 0.5 * turbulence)))

    def _meta_confidence_fusion(self, variance: float, trend: float, turbulence: float, reinforcement: float) -> float:
        # lower variance + stable trend + moderate turbulence + good reinforcement => high confidence
        v_term = max(0.0, 1.0 - min(variance * 10, 1.0))
        t_term = max(0.0, 1.0 - min(abs(trend) * 5, 1.0))
        turb_term = max(0.0, 1.0 - min(turbulence, 1.0))
        r_term = max(0.0, min(1.0, 0.5 + reinforcement * 0.1))
        conf = 0.25 * (v_term + t_term + turb_term + r_term)
        return max(0.0, min(1.0, conf))

    def _multi_engine_vote(self, base_risk: float, variance: float, trend: float, turbulence: float, baseline_dev: float) -> float:
        engines = {}

        # EWMA (here base_risk)
        engines["ewma"] = base_risk

        # trend engine
        engines["trend"] = max(0.0, min(1.0, 0.5 + trend))

        # variance engine
        engines["variance"] = max(0.0, min(1.0, variance * 5))

        # turbulence engine
        engines["turbulence"] = max(0.0, min(1.0, turbulence))

        # baseline deviation
        engines["baseline_dev"] = max(0.0, min(1.0, baseline_dev))

        # reinforcement memory (from autonomy success_score)
        engines["reinforcement"] = max(0.0, min(1.0, 0.5 + self.autonomy.success_score * 0.05))

        # weights (could be learned; here static)
        weights = {
            "ewma": 0.25,
            "trend": 0.15,
            "variance": 0.15,
            "turbulence": 0.15,
            "baseline_dev": 0.15,
            "reinforcement": 0.15,
        }

        num = sum(engines[k] * weights[k] for k in engines)
        den = sum(weights.values())
        best_guess = num / den if den > 0 else base_risk

        # store reasoning heatmap
        self.reasoning_heatmap = {k: round(engines[k] * weights[k], 3) for k in engines}
        return max(0.0, min(1.0, best_guess))

    def _update_meta_state_from_performance(self):
        # meta-state evolution based on success_score and regime
        s = self.autonomy.success_score
        if self.meta_state == "Hyper-Flow" and s < -2:
            self.meta_state = "Sentinel"
        elif self.meta_state == "Sentinel" and s > 3:
            self.meta_state = "Hyper-Flow"
        elif self.meta_state == "Recovery-Flow" and self.regime == "COOLING":
            self.meta_state = "Deep-Dream"
        elif self.meta_state == "Deep-Dream" and self.regime == "RISING":
            self.meta_state = "Sentinel"

    def _apply_meta_state_effects(self):
        # change appetite / threads / cache behavior
        if self.meta_state == "Hyper-Flow":
            self.autonomy.appetite = min(2.0, self.autonomy.appetite + 0.1)
            self.autonomy.thread_expansion = min(3.0, self.autonomy.thread_expansion + 0.2)
            self.autonomy.cache_aggressiveness = min(0.9, self.autonomy.cache_aggressiveness + 0.1)
        elif self.meta_state == "Deep-Dream":
            self.autonomy.appetite = max(0.5, self.autonomy.appetite - 0.1)
            self.autonomy.thread_expansion = max(0.5, self.autonomy.thread_expansion - 0.2)
        elif self.meta_state == "Sentinel":
            # balanced
            pass
        elif self.meta_state == "Recovery-Flow":
            self.autonomy.dampening = min(0.95, self.autonomy.dampening + 0.05)

    def _predictive_dampening(self):
        # short-term (5s) and long-term (120s) dampening
        short = self.predictions["5s"]
        long = self.predictions["120s"]
        self.short_damp = max(0.0, min(1.0, short))
        self.long_damp = max(0.0, min(1.0, long))

    def _update_pattern_memory(self, putty_report: Dict[str, Any]):
        pattern = {
            "risk": self.risk_score,
            "health": self.collective_health,
            "meta_state": self.meta_state,
            "regime": self.regime,
            "timestamp": time.time(),
        }
        self.pattern_memory.append(pattern)
        if len(self.pattern_memory) > 200:
            self.pattern_memory.pop(0)

    # ---- Tick ----

    def tick(self, putty: SillyPuttyDataGatherer):
        self._tick_count += 1
        p = putty.report()

        base_risk = 0.3 * p["anomaly_density"] + 0.2 * p["pressure"] + 0.2 * p["turbulence"]
        env_factor = {"CALM": 0.7, "TENSE": 1.0, "DANGER": 1.3}[self.environment]
        mission_factor = {
            "AUTO": 1.0,
            "PROTECT": 0.8,
            "STABILITY": 0.9,
            "LEARN": 1.1,
            "OPTIMIZE": 1.0,
        }[self.mission]

        self.risk_score = max(0.0, min(1.0, base_risk * env_factor * mission_factor))
        self._risk_history.append(self.risk_score)
        self._update_regime()

        self.opportunity_score = max(0.0, min(1.0, self.opportunity_bias * (1.0 - self.risk_score)))

        if self.risk_score > 0.7:
            self.anticipation = "Incoming turbulence / spike"
        elif self.opportunity_score > 0.6:
            self.anticipation = "Good learning/optimization window"
        else:
            self.anticipation = "Neutral / stable drift"

        # multi-horizon forecasting
        self._multi_horizon_forecast(base_risk, p["turbulence"])

        # variance/trend for meta-confidence + voting
        vals = list(self._risk_history)
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / len(vals)
            trend = vals[-1] - vals[0]
        else:
            variance = 0.0
            trend = 0.0

        baseline_dev = abs(self.risk_score - 0.3)
        self.meta_confidence = self._meta_confidence_fusion(variance, trend, p["turbulence"], self.autonomy.success_score)

        # multi-engine voting for best-guess risk
        best_guess = self._multi_engine_vote(base_risk, variance, trend, p["turbulence"], baseline_dev)

        # anomaly/local/hive risk from best-guess
        self.anomaly_risk = max(0.0, min(1.0, best_guess * (0.5 + 0.5 * self.anomaly_sensitivity)))
        self.drive_risk = max(0.0, min(1.0, best_guess * (0.5 + 0.5 * (1 - self.collective_weighting))))
        self.hive_risk = max(0.0, min(1.0, best_guess * (0.5 + 0.5 * self.collective_weighting)))

        self.collective_health = max(0.0, min(1.0, 1.0 - best_guess * 0.7))
        self.health_trend = max(-1.0, min(1.0, (self.collective_health - 0.5) * 2))

        # predictive dampening
        self._predictive_dampening()

        # pattern memory
        self._update_pattern_memory(p)

        # meta-state evolution + effects
        self._update_meta_state_from_performance()
        self._apply_meta_state_effects()

        # codex / agent / cipher
        ghost = self.codex.detect_ghost_sync(
            risk=self.risk_score,
            health=self.collective_health,
            turbulence=p["turbulence"],
        )

        if ghost or self._tick_count % 20 == 0:
            self.codex.mutate_rules(
                reason="ghost_sync" if ghost else "periodic_recalibration"
            )

        self.agent.step(
            risk=self.risk_score,
            health=self.collective_health,
            ghost_sync=ghost,
        )

        self.cipher.tick(
            risk=self.risk_score,
            integrity=self.collective_health,
        )

        # autonomy hooks
        self.autonomy.record_outcome(true_positive=self.risk_score > 0.8, false_positive=False)
        self.autonomy.auto_tune()
        self.autonomy.auto_calibrate_if_due(
            anomaly_density=p["anomaly_density"],
            pressure=p["pressure"],
        )

    # ---- Snapshots ----

    def situational_snapshot(self) -> Dict[str, Any]:
        return {
            "mission": self.mission,
            "environment": self.environment,
            "opportunity_score": round(self.opportunity_score, 3),
            "risk_score": round(self.risk_score, 3),
            "anticipation": self.anticipation,
            "risk_tolerance": round(self.risk_tolerance, 3),
            "opportunity_bias": round(self.opportunity_bias, 3),
            "meta_state": self.meta_state,
            "regime": self.regime,
        }

    def predictive_snapshot(self) -> Dict[str, Any]:
        return {
            "horizon": self.prediction_horizon,
            "anomaly_risk": round(self.anomaly_risk, 3),
            "drive_risk": round(self.drive_risk, 3),
            "hive_risk": round(self.hive_risk, 3),
            "collective_health": round(self.collective_health, 3),
            "health_trend": round(self.health_trend, 3),
            "anomaly_sensitivity": round(self.anomaly_sensitivity, 3),
            "collective_weighting": round(self.collective_weighting, 3),
            "pred_1s": round(self.predictions["1s"], 3),
            "pred_5s": round(self.predictions["5s"], 3),
            "pred_30s": round(self.predictions["30s"], 3),
            "pred_120s": round(self.predictions["120s"], 3),
            "meta_confidence": round(self.meta_confidence, 3),
            "short_damp": round(self.short_damp, 3),
            "long_damp": round(self.long_damp, 3),
        }

    def codex_snapshot(self) -> Dict[str, Any]:
        return self.codex.snapshot()

    def agent_snapshot(self) -> Dict[str, Any]:
        return self.agent.snapshot()

    def cipher_snapshot(self) -> Dict[str, Any]:
        return self.cipher.snapshot()

    def autonomy_snapshot(self) -> Dict[str, Any]:
        return self.autonomy.summary()

    def reasoning_snapshot(self) -> Dict[str, Any]:
        return self.reasoning_heatmap


# =========================
# GUI: Cockpit
# =========================

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QListWidget
)
from PySide6.QtCore import Qt, QTimer


class BorgCockpit(QWidget):
    def __init__(self,
                 autoloader: BorgAutoloader,
                 putty: SillyPuttyDataGatherer,
                 organs: List[BaseOrgan],
                 self_integrity: SelfIntegrityOrgan,
                 queen: BorgQueen,
                 rt_queen: RealTimeQueen,
                 chain_engine: AttackChainEngine,
                 brain: BrainCore):
        super().__init__()
        self.autoloader = autoloader
        self.putty = putty
        self.organs = organs
        self.self_integrity = self_integrity
        self.queen = queen
        self.rt_queen = rt_queen
        self.chain_engine = chain_engine
        self.brain = brain

        self.setWindowTitle("Borg Organism Cockpit – Full Brain")
        self.resize(1300, 900)

        root = QVBoxLayout()

        # --- Core Telemetry ---
        top_group = QGroupBox("Core Telemetry")
        top_layout = QVBoxLayout()

        self.label_putty = QLabel()
        self.label_queen = QLabel()
        self.label_rtqueen = QLabel()
        self.label_chains = QLabel()
        self.label_autonomy = QLabel()

        top_layout.addWidget(self.label_putty)
        top_layout.addWidget(self.label_queen)
        top_layout.addWidget(self.label_rtqueen)
        top_layout.addWidget(self.label_chains)
        top_layout.addWidget(self.label_autonomy)
        top_group.setLayout(top_layout)

        # --- Organs ---
        organ_group = QGroupBox("Organs")
        organ_layout = QVBoxLayout()
        self.label_organs = QLabel()
        self.label_self_integrity = QLabel()
        organ_layout.addWidget(self.label_organs)
        organ_layout.addWidget(self.label_self_integrity)
        organ_group.setLayout(organ_layout)

        # --- Situational Cortex ---
        sa_group = QGroupBox("Situational Awareness Cortex")
        sa_layout = QVBoxLayout()
        self.sa_label_state = QLabel()
        self.sa_label_scores = QLabel()
        self.sa_label_anticipation = QLabel()

        mission_row = QHBoxLayout()
        for m in ["AUTO", "PROTECT", "STABILITY", "LEARN", "OPTIMIZE"]:
            btn = QPushButton(m)
            btn.clicked.connect(lambda _, mm=m: self.brain.set_mission(mm))
            mission_row.addWidget(btn)

        risk_row = QHBoxLayout()
        risk_row.addWidget(QLabel("Risk tolerance"))
        self.slider_risk_tol = QSlider(Qt.Horizontal)
        self.slider_risk_tol.setMinimum(0)
        self.slider_risk_tol.setMaximum(100)
        self.slider_risk_tol.setValue(int(self.brain.risk_tolerance * 100))
        self.slider_risk_tol.valueChanged.connect(
            lambda v: setattr(self.brain, "risk_tolerance", v / 100.0)
        )
        risk_row.addWidget(self.slider_risk_tol)

        opp_row = QHBoxLayout()
        opp_row.addWidget(QLabel("Opportunity bias (learning windows)"))
        self.slider_opp = QSlider(Qt.Horizontal)
        self.slider_opp.setMinimum(0)
        self.slider_opp.setMaximum(100)
        self.slider_opp.setValue(int(self.brain.opportunity_bias * 100))
        self.slider_opp.valueChanged.connect(
            lambda v: setattr(self.brain, "opportunity_bias", v / 100.0)
        )
        opp_row.addWidget(self.slider_opp)

        sa_layout.addWidget(self.sa_label_state)
        sa_layout.addWidget(self.sa_label_scores)
        sa_layout.addWidget(self.sa_label_anticipation)
        sa_layout.addLayout(mission_row)
        sa_layout.addLayout(risk_row)
        sa_layout.addLayout(opp_row)
        sa_group.setLayout(sa_layout)

        # --- Predictive Panel + Codex/Agent/Cipher + Reasoning Heatmap ---
        pi_group = QGroupBox("Predictive Intelligence + Codex / Agent / Cipher / Reasoning")
        pi_layout = QVBoxLayout()

        self.pi_label_risks = QLabel()
        self.pi_label_health = QLabel()
        self.pi_label_horizons = QLabel()
        self.pi_label_conf = QLabel()
        self.label_codex = QLabel()
        self.label_agent = QLabel()
        self.label_cipher = QLabel()
        self.label_reasoning = QLabel()

        horizon_row = QHBoxLayout()
        for h in ["SHORT", "MEDIUM", "LONG"]:
            btn = QPushButton(h)
            btn.clicked.connect(lambda _, hh=h: self.brain.set_horizon(hh))
            horizon_row.addWidget(btn)

        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Anomaly sensitivity"))
        self.slider_sens = QSlider(Qt.Horizontal)
        self.slider_sens.setMinimum(0)
        self.slider_sens.setMaximum(100)
        self.slider_sens.setValue(int(self.brain.anomaly_sensitivity * 100))
        self.slider_sens.valueChanged.connect(
            lambda v: setattr(self.brain, "anomaly_sensitivity", v / 100.0)
        )
        sens_row.addWidget(self.slider_sens)

        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel("Collective weighting (0=local, 1=hive)"))
        self.slider_weight = QSlider(Qt.Horizontal)
        self.slider_weight.setMinimum(0)
        self.slider_weight.setMaximum(100)
        self.slider_weight.setValue(int(self.brain.collective_weighting * 100))
        self.slider_weight.valueChanged.connect(
            lambda v: setattr(self.brain, "collective_weighting", v / 100.0)
        )
        weight_row.addWidget(self.slider_weight)

        pi_layout.addWidget(self.pi_label_risks)
        pi_layout.addWidget(self.pi_label_health)
        pi_layout.addWidget(self.pi_label_horizons)
        pi_layout.addWidget(self.pi_label_conf)
        pi_layout.addLayout(horizon_row)
        pi_layout.addLayout(sens_row)
        pi_layout.addLayout(weight_row)
        pi_layout.addWidget(self.label_codex)
        pi_layout.addWidget(self.label_agent)
        pi_layout.addWidget(self.label_cipher)
        pi_layout.addWidget(self.label_reasoning)
        pi_group.setLayout(pi_layout)

        # --- Autoloader status ---
        auto_group = QGroupBox("Autoloader")
        auto_layout = QVBoxLayout()
        self.list_loaded = QListWidget()
        self.list_failed = QListWidget()
        auto_layout.addWidget(QLabel("Loaded Libraries"))
        auto_layout.addWidget(self.list_loaded)
        auto_layout.addWidget(QLabel("Failed Libraries"))
        auto_layout.addWidget(self.list_failed)
        auto_group.setLayout(auto_layout)

        root.addWidget(top_group)
        root.addWidget(organ_group)
        root.addWidget(sa_group)
        root.addWidget(pi_group)
        root.addWidget(auto_group)
        self.setLayout(root)

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(500)
        self._tick()

    def _tick(self):
        # update organs
        for o in self.organs:
            o.update()
            o.micro_recovery()
        self.self_integrity.update(self.organs)

        # putty + brain
        self.putty.absorb(os.urandom(64))
        self.brain.tick(self.putty)

        # queens / chains synthetic
        report = self.putty.report()
        self.queen.assimilate("node-001", report)
        events_for_rt = [{"entity": f"node-001_flow_{i}", "score": report["anomaly_density"] + i * 0.01} for i in range(3)]
        self.rt_queen.update("node-001", events_for_rt)
        chains = self.chain_engine.detect()

        # autoloader
        auto_status = self.autoloader.status()

        # snapshots
        putty_report = self.putty.report()
        queen_status = self.queen.status()
        rt_risk = self.rt_queen.global_risk()
        autonomy = self.brain.autonomy_snapshot()
        sa = self.brain.situational_snapshot()
        pi = self.brain.predictive_snapshot()
        codex = self.brain.codex_snapshot()
        agent = self.brain.agent_snapshot()
        cipher = self.brain.cipher_snapshot()
        reasoning = self.brain.reasoning_snapshot()

        self.label_putty.setText(
            f"Putty - Mass: {putty_report['mass']} | EffGain: {putty_report['efficiency_gain']} | Flows: {putty_report['unique_flows']} | State: {putty_report['state']} | Meta: {putty_report['meta_state']}"
        )
        self.label_queen.setText(
            f"Queen - Efficiency: {queen_status['efficiency']} | Nodes: {queen_status['nodes']} | State: {queen_status['state']}"
        )
        self.label_rtqueen.setText(
            f"RT Queen Risk Entities: {len(rt_risk)}"
        )
        self.label_chains.setText(
            "Chains: " + ", ".join([f"{c[0]}({c[1]})" for c in chains]) if chains else "Chains: none"
        )
        self.label_autonomy.setText(
            f"Autonomy - appetite={autonomy['appetite']} thr={autonomy['threshold_risk']} hor={autonomy['horizon']} damp={autonomy['dampening']} cache={autonomy['cache_aggressiveness']} threads={autonomy['thread_expansion']}"
        )

        organ_str = " | ".join([f"{o.name}: H={o.health:.2f} R={o.risk:.2f} I={o.integrity:.2f}" for o in self.organs])
        self.label_organs.setText(f"Organs: {organ_str}")
        self.label_self_integrity.setText(
            f"Self-Integrity - H={self.self_integrity.health:.2f} R={self.self_integrity.risk:.2f} I={self.self_integrity.integrity:.2f}"
        )

        self.sa_label_state.setText(
            f"Mission: {sa['mission']} | Env: {sa['environment']} | MetaState: {sa['meta_state']} | Regime: {sa['regime']} | RiskTol: {sa['risk_tolerance']} | OppBias: {sa['opportunity_bias']}"
        )
        self.sa_label_scores.setText(
            f"Risk: {sa['risk_score']} | Opportunity: {sa['opportunity_score']}"
        )
        self.sa_label_anticipation.setText(
            f"Anticipation: {sa['anticipation']}"
        )

        self.pi_label_risks.setText(
            f"Horizon: {pi['horizon']} | AnomRisk: {pi['anomaly_risk']} | DriveRisk: {pi['drive_risk']} | HiveRisk: {pi['hive_risk']}"
        )
        self.pi_label_health.setText(
            f"CollectiveHealth: {pi['collective_health']} | Trend: {pi['health_trend']} | Sens: {pi['anomaly_sensitivity']} | Weight: {pi['collective_weighting']}"
        )
        self.pi_label_horizons.setText(
            f"Pred(1s/5s/30s/120s): {pi['pred_1s']}, {pi['pred_5s']}, {pi['pred_30s']}, {pi['pred_120s']} | ShortDamp: {pi['short_damp']} | LongDamp: {pi['long_damp']}"
        )
        self.pi_label_conf.setText(
            f"MetaConfidence: {pi['meta_confidence']}"
        )

        self.label_codex.setText(
            f"Codex - Retention: {codex['retention_seconds']}s | GhostSensitive: {codex['ghost_sensitive']} | PhantomNode: {codex['phantom_node_enabled']} | Mutations: {codex['mutations']}"
        )
        self.label_agent.setText(
            f"Agent - Weights: {agent['weights']} | Mutations: {agent['mutations']} | Backend: {agent['backend']}"
        )
        self.label_cipher.setText(
            f"Cipher - Posture: {cipher['posture']} | KeyID: {cipher['current_key_id']} | RotInt: {cipher['rotation_interval']}s | SinceRot: {cipher['seconds_since_rotation']}s"
        )

        # reasoning heatmap
        if reasoning:
            parts = [f"{k}:{v:.3f}" for k, v in reasoning.items()]
            self.label_reasoning.setText("Reasoning Heatmap: " + " | ".join(parts))
        else:
            self.label_reasoning.setText("Reasoning Heatmap: (warming up)")

        self.list_loaded.clear()
        for lib in auto_status["loaded"]:
            self.list_loaded.addItem(lib)
        self.list_failed.clear()
        for lib, reason in auto_status["failed"].items():
            self.list_failed.addItem(f"{lib}: {reason}")


# =========================
# Main
# =========================

def main():
    autoloader = BorgAutoloader()
    autoloader.autoload()
    watchdog = AutoloaderWatchdog(autoloader, interval=60.0)
    watchdog.start()

    putty = SillyPuttyDataGatherer()
    putty.set_state("hypervigilant")
    putty.set_meta_target("hyper_flow")

    queen = BorgQueen()
    rt_queen = RealTimeQueen()
    chain_engine = AttackChainEngine()

    organs: List[BaseOrgan] = [
        DeepRamOrgan("DeepRAM"),
        BackupEngineOrgan("Backup"),
        NetworkWatcherOrgan("Network"),
        GPUCacheOrgan("GPUCache"),
        ThermalOrgan("Thermal"),
        DiskOrgan("Disk"),
        VRAMOrgan("VRAM"),
        AICoachOrgan("AICoach"),
        SwarmNodeOrgan("SwarmNode"),
        Back4BloodAnalyzer("Back4Blood"),
    ]
    self_integrity = SelfIntegrityOrgan("SelfIntegrity")

    brain = BrainCore(chain_engine)

    app = QApplication(sys.argv)
    cockpit = BorgCockpit(
        autoloader=autoloader,
        putty=putty,
        organs=organs,
        self_integrity=self_integrity,
        queen=queen,
        rt_queen=rt_queen,
        chain_engine=chain_engine,
        brain=brain,
    )
    cockpit.show()
    try:
        sys.exit(app.exec())
    finally:
        watchdog.stop()


if __name__ == "__main__":
    main()
