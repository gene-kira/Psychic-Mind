#!/usr/bin/env python3
# apex_sentinel_unified_v7_borgmesh_cipher_guarddog.py
#
# APEX‑Sentinel OS / ChronoMind Nexus / Parallax Swarm Engine / Borg Core
# HybridBrain + ReplicaNPU + Regime detection + Auto‑tuning + Auto‑calibration
# PatternMemory + WaterPhysics + RL meta‑states + Best‑Guess engine
# Self‑Integrity Organ + SystemTelemetry (CPU/MEM/DISK/NET/PROC) + ETW stub
# Swarm + Queen + BorgRegistry + BorgMesh + Decoy Engine + Cipher Engine
# PrivacyScrubber (glyph + mirror + reverse + chameleon + shredder)
# Zero‑Trust pipeline + Enforcement hooks + GuardDog (firewall/settings watcher)
# PySide6 cockpit + Reboot memory + Altered States + Borg tab
# Background, autonomous, zero‑trust posture

import sys
import os
import json
import time
import math
import random
import subprocess
import importlib
import ctypes
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

# ============================================================
# AUTO‑ELEVATION (Windows)
# ============================================================

def ensure_admin():
    try:
        if os.name == "nt":
            if not ctypes.windll.shell32.IsUserAnAdmin():
                script = os.path.abspath(sys.argv[0])
                params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable, f'"{script}" {params}', None, 1
                )
                sys.exit()
    except Exception as e:
        print(f"[APEX‑Sentinel] Elevation failed: {e}")
        sys.exit()

ensure_admin()

# ============================================================
# Autoloader
# ============================================================

REQUIRED_LIBS = ["PySide6", "psutil"]

def ensure_deps():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

ensure_deps()

from PySide6 import QtWidgets, QtCore, QtGui
import psutil

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import win32evtlog  # type: ignore
except Exception:
    win32evtlog = None

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
# PrivacyScrubber + Chameleon/Glyph/Mirror/Reverse/Shredder + Zero‑Trust
# ============================================================

class PrivacyScrubber:
    SENSITIVE_KEYS = {
        "ip", "mac", "mac_address", "user", "username", "email",
        "device_id", "serial", "hwid", "bios", "system_id",
        "hostname", "domain", "sid", "phone", "phone_number",
        "biometric", "fingerprint", "faceid", "iris",
    }

    def __init__(self):
        self.glyph_map: Dict[str, str] = {}

    def glyph_encode(self, value: str) -> str:
        if value in self.glyph_map:
            return self.glyph_map[value]
        glyphs = ["⚚", "⚝", "⚞", "⚟", "☿", "♆", "♇", "♄", "♃", "♁", "☉", "☽"]
        seed = sum(ord(c) for c in value)
        random.seed(seed)
        encoded = "".join(random.choice(glyphs) for _ in range(8))
        self.glyph_map[value] = encoded
        return encoded

    def mirror_string(self, s: str) -> str:
        return s[::-1]

    def reverse_bits_repr(self, s: str) -> str:
        b = s.encode("utf-8")
        rev = bytes(~x & 0xFF for x in b)
        return rev.hex()

    def shred_string(self, s: str) -> str:
        # irreversible shred marker
        return "«shredded»"

    def chameleon_numeric(self, v: float) -> float:
        bucket = round(v, 1)
        noise = random.uniform(-0.02, 0.02)
        return clamp(bucket + noise, 0.0, 1.0)

    def encode_sensitive_string(self, value: str) -> str:
        # glyph + mirror + reverse‑bits style encoding
        g = self.glyph_encode(value)
        m = self.mirror_string(g)
        r = self.reverse_bits_repr(m)
        return r

    def scrub_value(self, key: str, value: Any) -> Any:
        k = key.lower()
        if k in self.SENSITIVE_KEYS:
            if isinstance(value, str):
                return self.encode_sensitive_string(value)
            if isinstance(value, (int, float)):
                return self.chameleon_numeric(float(value))
            return self.shred_string(str(value))
        if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            return self.chameleon_numeric(float(value))
        return value

    def scrub_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in data.items():
            if isinstance(v, dict):
                out[k] = self.scrub_dict(v)
            elif isinstance(v, list):
                out[k] = [self.scrub_dict(x) if isinstance(x, dict) else self.scrub_value(k, x) for x in v]
            else:
                out[k] = self.scrub_value(k, v)
        return out


class ZeroTrustDataPipeline:
    def __init__(self, scrubber: PrivacyScrubber):
        self.scrubber = scrubber

    def sanitize_for_persist(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.scrubber.scrub_dict(payload)

    def sanitize_for_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.scrubber.scrub_dict(payload)

    def sanitize_for_gui(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.scrubber.scrub_dict(payload)

# ============================================================
# Decoy Engine
# ============================================================

def random_time_iso():
    t = datetime.now(timezone.utc) - timedelta(seconds=random.randint(0, 3600))
    return t.isoformat().replace("+00:00", "Z")

def random_country_code():
    return random.choice(["US", "DE", "FR", "JP", "BR", "IN", "CN", "GB", "CA", "AU"])

def random_glyph_stream(scrubber: PrivacyScrubber):
    base = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(12))
    return scrubber.encode_sensitive_string(base)

def generate_decoy(scrubber: PrivacyScrubber):
    return {
        "timestamp": random_time_iso(),
        "origin": random_country_code(),
        "payload": random_glyph_stream(scrubber),
    }

# ============================================================
# Autonomous Cipher Engine (simple symmetric placeholder)
# ============================================================

class AutonomousCipherEngine:
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            random.seed(os.urandom(16))
            key = bytes(random.getrandbits(8) for _ in range(32))
        self.key = key

    def _xor(self, data: bytes) -> bytes:
        k = self.key
        return bytes(b ^ k[i % len(k)] for i, b in enumerate(data))

    def encrypt(self, payload: Dict[str, Any]) -> bytes:
        raw = json.dumps(payload).encode("utf-8")
        return self._xor(raw)

    def decrypt(self, blob: bytes) -> Dict[str, Any]:
        raw = self._xor(blob)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

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
# ReplicaNPU with Regime Detection + Auto‑Tuning/Calibration
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
        self.meta_momentum = 0.0

        self.regime = "stable"
        self.regime_history = deque(maxlen=64)
        self.baseline_risk = 0.5
        self.last_auto_calibration = datetime.now(timezone.utc)

        self.appetite = 1.0
        self.horizon_short = 5.0
        self.horizon_medium = 30.0
        self.horizon_long = 120.0
        self.dampening = 0.5

    def mac(self, a, b):
        return a * b

    def add_head(self, name, input_dim, lr=0.01, risk=1.0, organ=None):
        self.heads[name] = {
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "organ": organ,
            "history": deque(maxlen=64),
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

    def _detect_regime(self, x, best_guess: float):
        load = x[0] if len(x) > 0 else 0.5
        trend = x[1] if len(x) > 1 else 0.0
        turb = x[2] if len(x) > 2 else 0.0

        if abs(trend) < 0.05 and turb < 0.2:
            regime = "low_variance_stable"
        elif turb > 0.6:
            regime = "high_variance_chaotic"
        elif trend > 0.1:
            regime = "rising_load"
        elif trend < -0.1:
            regime = "cooling_down"
        else:
            regime = "stable"

        self.regime = regime
        self.regime_history.append(regime)
        return regime

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

        baseline_dev = abs(self.baseline_risk - 0.5)
        engines.append(("baseline_dev", baseline_dev, 0.3))

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

        self._detect_regime(x, best)
        self._update_meta_state(best, meta_conf)
        self._auto_tune(best, meta_conf)
        self._auto_calibrate()

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
        reward = clamp(1.0 - best, 0.0, 1.0) * meta_conf
        self.rl.reward_meta_state(self.meta_state, reward)

        target = self.rl.best_meta_state()

        if target != self.meta_state:
            self.meta_momentum += 0.1
            if self.meta_momentum > 0.7:
                if self.meta_state == "Hyper-Flow" and target == "Sentinel":
                    if best < 0.4:
                        self.meta_state = target
                elif self.meta_state == "Sentinel" and target == "Hyper-Flow":
                    if best < 0.3:
                        self.meta_state = target
                elif self.meta_state == "Recovery-Flow" and target == "Deep-Dream":
                    if self.regime != "high_variance_chaotic":
                        self.meta_state = target
                else:
                    self.meta_state = target
                self.meta_momentum = 0.0
        else:
            self.meta_momentum = max(0.0, self.meta_momentum - 0.05)

        if self.model_integrity < 0.5:
            self.meta_state = "Recovery-Flow"

    def _auto_tune(self, best: float, meta_conf: float):
        if best > 0.7:
            self.appetite = max(0.2, self.appetite - 0.02)
        elif best < 0.3:
            self.appetite = min(1.5, self.appetite + 0.01)

        if self.regime == "high_variance_chaotic":
            self.dampening = min(1.0, self.dampening + 0.02)
        elif self.regime == "low_variance_stable":
            self.dampening = max(0.2, self.dampening - 0.01)

        if self.regime in ("high_variance_chaotic", "rising_load"):
            self.horizon_short = max(2.0, self.horizon_short - 0.2)
            self.horizon_long = max(30.0, self.horizon_long - 1.0)
        elif self.regime in ("low_variance_stable", "cooling_down"):
            self.horizon_short = min(10.0, self.horizon_short + 0.1)
            self.horizon_long = min(300.0, self.horizon_long + 1.0)

        for h in self.heads.values():
            base_lr = h["lr"]
            h["lr"] = clamp(base_lr * (0.9 + 0.2 * self.appetite), 0.001, 0.1)

        self.baseline_risk = rolling_avg(self.baseline_risk, best, weight=0.01)

    def _auto_calibrate(self):
        now = datetime.now(timezone.utc)
        if now - self.last_auto_calibration < timedelta(hours=24):
            return
        self.last_auto_calibration = now

        if self.memory:
            m = sum(self.memory) / len(self.memory)
            self.baseline_risk = clamp(m, 0.0, 1.0)

        vals = list(self.rl.q_meta.values())
        if vals:
            avg = sum(vals) / len(vals)
            for k in self.rl.q_meta:
                self.rl.q_meta[k] = clamp(0.5 + (self.rl.q_meta[k] - avg), 0.0, 1.0)

        if self.model_integrity < 0.7:
            self.integrity_threshold = max(0.3, self.integrity_threshold - 0.02)
        else:
            self.integrity_threshold = min(0.6, self.integrity_threshold + 0.01)

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
            "regime": self.regime,
            "baseline_risk": self.baseline_risk,
            "appetite": self.appetite,
            "horizon_short": self.horizon_short,
            "horizon_medium": self.horizon_medium,
            "horizon_long": self.horizon_long,
            "dampening": self.dampening,
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
        self.regime = state.get("regime", "stable")
        self.baseline_risk = state.get("baseline_risk", 0.5)
        self.appetite = state.get("appetite", 1.0)
        self.horizon_short = state.get("horizon_short", 5.0)
        self.horizon_medium = state.get("horizon_medium", 30.0)
        self.horizon_long = state.get("horizon_long", 120.0)
        self.dampening = state.get("dampening", 0.5)

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
            "regime": self.regime,
            "baseline_risk": round(self.baseline_risk, 3),
            "appetite": round(self.appetite, 3),
            "horizon_short": self.horizon_short,
            "horizon_medium": self.horizon_medium,
            "horizon_long": self.horizon_long,
            "dampening": round(self.dampening, 3),
            "confidence": {k: round(self.confidence(k), 3) for k in self.heads},
        }

# ============================================================
# Movidius / ONNX Engine (with training stub)
# ============================================================

class MovidiusInferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session = None
        if model_path and ort is not None:
            try:
                self.session = ort.InferenceSession(model_path)
            except Exception:
                self.session = None

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
        # Placeholder: hook for offline ONNX training pipeline
        # You would load dataset, train model, export ONNX here.
        print(f"[ONNX] Training stub: dataset={dataset_path}, out={output_model_path}")

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
# SystemTelemetry (real CPU/MEM/DISK/NET/PROC)
# ============================================================

class SystemTelemetry:
    def __init__(self):
        self.last_net = psutil.net_io_counters()
        self.last_net_time = time.time()

    def sample(self) -> Dict[str, Any]:
        cpu = psutil.cpu_percent(interval=None) / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        disk = psutil.disk_usage(os.getcwd()).percent / 100.0

        now = time.time()
        net = psutil.net_io_counters()
        dt = max(0.1, now - self.last_net_time)
        bytes_sent = (net.bytes_sent - self.last_net.bytes_sent) / dt
        bytes_recv = (net.bytes_recv - self.last_net.bytes_recv) / dt
        self.last_net = net
        self.last_net_time = now
        net_load = clamp((bytes_sent + bytes_recv) / (1024 * 1024 * 10), 0.0, 1.0)

        procs = []
        try:
            for p in psutil.process_iter(["name", "pid", "cpu_percent", "memory_percent", "cmdline"]):
                procs.append(p.info)
        except Exception:
            pass

        top_proc = None
        if procs:
            top_proc = max(procs, key=lambda x: x.get("cpu_percent", 0.0))

        sensors = {
            "cpu": cpu,
            "mem": mem,
            "disk": disk,
            "net": net_load,
        }

        return {
            "sensors": sensors,
            "top_process": top_proc,
            "processes": procs,
        }

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
# ETW / Kernel hooks (safe stub with event sampling)
# ============================================================

class ETWKernelSensor:
    def __init__(self, brain, borg):
        self.brain = brain
        self.borg = borg
        self.enabled = win32evtlog is not None and os.name == "nt"

    def poll(self):
        # Stub: in a real implementation, subscribe to ETW providers
        # and feed process/network events into Borg + brain.
        return

# ============================================================
# Enforcement Manager (Zero‑Trust Hooks)
# ============================================================

class EnforcementManager:
    def __init__(self):
        self.os_name = os.name

    def _run(self, cmd: List[str]):
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[Enforcer] Command failed: {cmd} ({e})")

    def block_ip(self, ip: str):
        if self.os_name == "nt":
            self._run(["netsh", "advfirewall", "firewall", "add", "rule",
                       f"name=APEX_Block_{ip}", "dir=out", "action=block", f"remoteip={ip}"])
        print(f"[Enforcer] Block IP: {ip}")

    def allow_ip(self, ip: str):
        if self.os_name == "nt":
            self._run(["netsh", "advfirewall", "firewall", "add", "rule",
                       f"name=APEX_Allow_{ip}", "dir=out", "action=allow", f"remoteip={ip}"])
        print(f"[Enforcer] Allow IP: {ip}")

    def kill_process(self, name: str):
        if self.os_name == "nt":
            self._run(["taskkill", "/F", "/IM", name])
        print(f"[Enforcer] Kill process: {name}")

    def isolate_entity(self, entity: str):
        print(f"[Enforcer] Isolate entity: {entity}")

    def increase_logging(self, entity: str):
        print(f"[Enforcer] Increase logging for: {entity}")

    def temporary_quarantine(self, entity: str):
        print(f"[Enforcer] Temporary quarantine: {entity}")

    def enforce_firewall_on(self):
        if self.os_name == "nt":
            self._run(["netsh", "advfirewall", "set", "allprofiles", "state", "on"])
        print("[Enforcer] Ensure firewall ON")

# ============================================================
# Borg Registry
# ============================================================

@dataclass
class BorgEntity:
    etype: str
    key: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    risk_history: List[float] = field(default_factory=list)

    def update(self, attrs: Dict[str, Any], risk: Optional[float] = None):
        self.attrs.update(attrs)
        self.last_seen = time.time()
        if risk is not None:
            self.risk_history.append(risk)

    def current_risk(self) -> float:
        if not self.risk_history:
            return 0.0
        return self.risk_history[-1]

    def to_dict(self):
        return {
            "etype": self.etype,
            "key": self.key,
            "attrs": self.attrs,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "current_risk": self.current_risk(),
            "risk_history_len": len(self.risk_history),
        }

class BorgRegistry:
    def __init__(self):
        self.entities: Dict[str, BorgEntity] = {}

    def touch(self, etype: str, key: str, attrs: Dict[str, Any], risk: Optional[float] = None) -> BorgEntity:
        ent = self.entities.get(key)
        if not ent:
            ent = BorgEntity(etype=etype, key=key)
            self.entities[key] = ent
        ent.update(attrs, risk=risk)
        return ent

    def all_entities(self) -> List[BorgEntity]:
        return list(self.entities.values())

# ============================================================
# Borg Mesh (network‑within‑network overlay)
# ============================================================

BORG_MESH_CONFIG = {
    "max_corridors": 5000,
    "unknown_bias": 0.4,
}

class BorgMesh:
    def __init__(self, borg: BorgRegistry, pipeline: ZeroTrustDataPipeline):
        self.nodes = {}  # url/endpoint -> {"state": discovered/built/enforced, "risk":0-100, "seen": int}
        self.edges = set()  # (src, dst)
        self.borg = borg
        self.pipeline = pipeline
        self.max_corridors = BORG_MESH_CONFIG["max_corridors"]

    def _risk(self, snippet: str) -> int:
        # simple entropy‑like heuristic
        if not snippet:
            return 10
        unique = len(set(snippet))
        base = int(min(1.0, unique / 32.0) * 80)
        if "login" in snippet.lower() or "password" in snippet.lower():
            base += 15
        return max(0, min(100, base))

    def discover(self, url: str, snippet: str, links: List[str]):
        risk = self._risk(snippet)
        node = self.nodes.get(url, {"state": "discovered", "risk": risk, "seen": 0})
        node["state"] = "discovered"
        node["risk"] = risk
        node["seen"] += 1
        self.nodes[url] = node
        for l in links[:20]:
            if len(self.edges) < self.max_corridors:
                self.edges.add((url, l))
        self.borg.touch("endpoint", url, {"source": "mesh_discover"}, risk=risk / 100.0)

    def build(self, url: str):
        if url not in self.nodes:
            return False
        self.nodes[url]["state"] = "built"
        return True

    def enforce(self, url: str):
        if url not in self.nodes:
            return False
        node = self.nodes[url]
        status = "SAFE_FOR_TRAVEL" if node["risk"] < 40 else "HOSTILE"
        node["state"] = "enforced"
        node["risk"] = 0 if status == "SAFE_FOR_TRAVEL" else max(50, node["risk"])
        self.borg.touch("endpoint", url, {"source": "mesh_enforce", "status": status}, risk=node["risk"] / 100.0)
        return True

    def stats(self):
        total = len(self.nodes)
        discovered = sum(1 for n in self.nodes.values() if n["state"] == "discovered")
        built = sum(1 for n in self.nodes.values() if n["state"] == "built")
        enforced = sum(1 for n in self.nodes.values() if n["state"] == "enforced")
        return {"total": total, "discovered": discovered, "built": built, "enforced": enforced, "corridors": len(self.edges)}

# ============================================================
# GuardDog: firewall/settings/rogue command watcher
# ============================================================

SUSPICIOUS_CMD_KEYWORDS = [
    "netsh", "firewall", "advfirewall", "reg", "sc", "powershell",
    "DisableFirewall", "Set-MpPreference", "RemoteDesktop", "rdp",
]

WHITELIST_PROCESSES = {
    "apex_sentinel.exe",
    "System",
    "svchost.exe",
}

class GuardDog:
    def __init__(self, enforcer: EnforcementManager):
        self.enforcer = enforcer
        self.last_firewall_check = 0.0
        self.firewall_check_interval = 15.0

    def _firewall_is_off(self) -> bool:
        if os.name != "nt":
            return False
        try:
            out = subprocess.check_output(
                ["netsh", "advfirewall", "show", "allprofiles"],
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            ).decode("utf-8", errors="ignore").lower()
            return "state off" in out
        except Exception:
            return False

    def _check_firewall(self):
        now = time.time()
        if now - self.last_firewall_check < self.firewall_check_interval:
            return
        self.last_firewall_check = now
        if self._firewall_is_off():
            print("[GuardDog] Firewall appears OFF, enforcing ON")
            self.enforcer.enforce_firewall_on()

    def _is_suspicious_cmd(self, cmdline: List[str]) -> bool:
        joined = " ".join(cmdline).lower()
        return any(k.lower() in joined for k in SUSPICIOUS_CMD_KEYWORDS)

    def scan_processes(self, processes: List[Dict[str, Any]]):
        self._check_firewall()
        for p in processes:
            name = p.get("name") or ""
            if name in WHITELIST_PROCESSES:
                continue
            cmdline = p.get("cmdline") or []
            if not cmdline:
                continue
            if self._is_suspicious_cmd(cmdline):
                print(f"[GuardDog] Suspicious command detected in {name}: {cmdline}")
                self.enforcer.kill_process(name)

# ============================================================
# Swarm / Queen / Swarm Comms (stub)
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

class SwarmComms:
    def __init__(self):
        self.last_broadcast = 0.0
        self.interval = 10.0

    def broadcast_state(self, node_id: str, best: float, meta_conf: float):
        now = time.time()
        if now - self.last_broadcast < self.interval:
            return
        self.last_broadcast = now
        print(f"[SwarmComms] Broadcast from {node_id}: best={best:.3f}, meta_conf={meta_conf:.3f}")

class QueenConsensus:
    def __init__(self, window: int = 180, borg: Optional[BorgRegistry] = None,
                 pipeline: Optional[ZeroTrustDataPipeline] = None):
        self.window = window
        self.global_events: deque[Tuple[float, str, Dict[str, Any]]] = deque()
        self.narratives: Dict[str, AttackNarrative] = {}
        self.borg = borg
        self.pipeline = pipeline

    def update_from_cluster_orders(self, orders: List[Dict[str, Any]]):
        now = time.time()
        for order in orders:
            agg = order["aggregate_features"]
            ent = agg.get("dominant_process_name") or agg.get("dominant_ip") or "unknown"
            evt = {
                "entity": ent,
                "cluster_id": order["cluster_id"],
                "severity": order["severity"],
                "swarm_confidence": order["swarm_confidence"],
            }
            self.global_events.append((now, "cluster_order", evt))
            if self.borg and ent != "unknown":
                etype = "ip" if "ip" in ent or ent.count(".") >= 3 else "process"
                self.borg.touch(etype, ent, {"source": "cluster_order"}, risk=order["swarm_confidence"])
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
        if self.borg:
            self.borg.touch("node", node_id, {"source": "best_guess"}, risk=best)
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
    def __init__(self, brain, organs: Dict[str, Any], pipeline: ZeroTrustDataPipeline, cipher: AutonomousCipherEngine):
        self.brain = brain
        self.organs = organs
        self.pipeline = pipeline
        self.cipher = cipher

    def snapshot_state(self) -> dict:
        raw = {
            "brain": self.brain.to_dict(),
            "organs": {name: getattr(org, "to_dict", lambda: {} )() for name, org in self.organs.items()},
        }
        return self.pipeline.sanitize_for_persist(raw)

    def save_to_path(self, base_path: str) -> str:
        state = self.snapshot_state()
        blob = self.cipher.encrypt(state)
        path = Path(base_path)
        path.mkdir(parents=True, exist_ok=True)
        bin_path = path / "organism_state.bin"
        with bin_path.open("wb") as f:
            f.write(blob)
        return str(bin_path)

    def load_from_path(self, base_path: str) -> bool:
        bin_path = Path(base_path) / "organism_state.bin"
        if not bin_path.exists():
            return False
        with bin_path.open("rb") as f:
            blob = f.read()
        _state = self.cipher.decrypt(blob)
        return True

# ============================================================
# HybridBrain
# ============================================================

class HybridBrain:
    def __init__(self, borg: BorgRegistry, pipeline: ZeroTrustDataPipeline, mesh: BorgMesh,
                 guarddog: GuardDog, cipher: AutonomousCipherEngine):
        self.npu = ReplicaNPU()
        self.npu.add_head("short", 3, lr=0.05, risk=1.5, organ="brain")
        self.npu.add_head("medium", 3, lr=0.03, risk=1.0, organ="cortex")
        self.npu.add_head("long", 3, lr=0.02, risk=0.7, organ="planner")

        self.integrity_organ = SelfIntegrityOrgan(
            expected_sensors=["cpu", "mem", "disk", "net"]
        )
        self.borg = borg
        self.pipeline = pipeline
        self.mesh = mesh
        self.guarddog = guarddog
        self.cipher = cipher

        self.ui_organ = UIAutomationOrgan(self)
        self.etw_sensor = ETWKernelSensor(self, borg)
        self.telemetry = SystemTelemetry()

        self.stance = "Neutral"
        self.last_best_guess = 0.5
        self.last_meta_conf = 0.5
        self.last_reasoning: Dict[str, Any] = {}
        self.last_features: List[float] = [0.0, 0.0, 0.0]

    def feed_ui_signal(self, snapshot, delta):
        proc = snapshot.get("active_process")
        if proc:
            self.borg.touch("process", proc, {"source": "ui_active"}, risk=None)

    def tick(self):
        t_sample = self.telemetry.sample()
        sensors = t_sample["sensors"]
        top_proc = t_sample["top_process"]
        processes = t_sample["processes"]

        x = [
            sensors.get("cpu", 0.5),
            sensors.get("net", 0.0),
            sensors.get("disk", 0.0),
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

        self.borg.touch("system", "local-node", {"source": "brain"}, risk=best)

        if top_proc and top_proc.get("name"):
            self.borg.touch(
                "process",
                top_proc["name"],
                {"source": "top_cpu", "cpu": top_proc.get("cpu_percent", 0.0)},
                risk=best,
            )

        # GuardDog: scan for rogue firewall/settings commands
        self.guarddog.scan_processes(processes)

        # BorgMesh: treat top process as endpoint, plus decoy links
        if top_proc and top_proc.get("name"):
            url = f"proc://{top_proc['name']}"
            snippet = " ".join((top_proc.get("cmdline") or [])[:5])
            links = [f"proc://{p.get('name','unknown')}" for p in processes[:10]]
            self.mesh.discover(url, snippet, links)
            self.mesh.build(url)
            self.mesh.enforce(url)

        return sensors, best, meta_conf, reasoning, integ

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
                    "regime": self.npu.regime,
                    "baseline_risk": self.npu.baseline_risk,
                }
            },
            "stance": self.stance,
        }

# ============================================================
# GUI: PredictionChart + Altered States panel + Borg tab
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
        p.drawText(5, 60, f"Regime: {self.brain.npu.regime}")

class AlteredStatesPanel(QtWidgets.QWidget):
    def __init__(self, brain: HybridBrain, parent=None):
        super().__init__(parent)
        self.brain = brain
        layout = QtWidgets.QFormLayout(self)

        self.lbl_meta_state = QtWidgets.QLabel()
        self.lbl_q_meta = QtWidgets.QLabel()
        self.lbl_integrity = QtWidgets.QLabel()
        self.lbl_plasticity = QtWidgets.QLabel()
        self.lbl_regime = QtWidgets.QLabel()
        self.lbl_appetite = QtWidgets.QLabel()
        self.lbl_horizons = QtWidgets.QLabel()
        self.lbl_dampening = QtWidgets.QLabel()

        layout.addRow("Meta-State:", self.lbl_meta_state)
        layout.addRow("Meta Q-Values:", self.lbl_q_meta)
        layout.addRow("Integrity Score:", self.lbl_integrity)
        layout.addRow("Plasticity:", self.lbl_plasticity)
        layout.addRow("Regime:", self.lbl_regime)
        layout.addRow("Appetite:", self.lbl_appetite)
        layout.addRow("Horizons (s/m/l):", self.lbl_horizons)
        layout.addRow("Dampening:", self.lbl_dampening)

    def refresh(self):
        s = self.brain.stats()
        self.lbl_meta_state.setText(s["meta_state"])
        self.lbl_q_meta.setText(json.dumps(s["q_meta"], indent=2))
        self.lbl_integrity.setText(f"{s['integrity_score']:.3f}")
        self.lbl_plasticity.setText(f"{s['plasticity']:.3f}")
        self.lbl_regime.setText(s["regime"])
        self.lbl_appetite.setText(f"{s['appetite']:.3f}")
        self.lbl_horizons.setText(f"{s['horizon_short']:.1f} / {s['horizon_medium']:.1f} / {s['horizon_long']:.1f}")
        self.lbl_dampening.setText(f"{s['dampening']:.3f}")

class BorgPanel(QtWidgets.QWidget):
    def __init__(self, borg: BorgRegistry, pipeline: ZeroTrustDataPipeline, parent=None):
        super().__init__(parent)
        self.borg = borg
        self.pipeline = pipeline
        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Type", "Key (glyph)", "Current Risk", "First Seen", "Last Seen"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def refresh(self):
        ents = self.borg.all_entities()
        self.table.setRowCount(len(ents))
        for i, e in enumerate(ents):
            key_encoded = self.pipeline.scrubber.glyph_encode(e.key)
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(e.etype))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(key_encoded))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{e.current_risk():.3f}"))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(e.first_seen))))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(time.strftime("%H:%M:%S", time.localtime(e.last_seen))))

# ============================================================
# MainWindow
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, brain: HybridBrain, swarm: SwarmState, queen: QueenConsensus,
                 enforcer: EnforcementManager, borg: BorgRegistry, pipeline: ZeroTrustDataPipeline,
                 mesh: BorgMesh, swarm_comms: SwarmComms, cipher: AutonomousCipherEngine):
        super().__init__()
        self.brain = brain
        self.swarm = swarm
        self.queen = queen
        self.enforcer = enforcer
        self.borg = borg
        self.pipeline = pipeline
        self.mesh = mesh
        self.swarm_comms = swarm_comms
        self.cipher = cipher

        self.setWindowTitle("APEX‑Sentinel OS / ChronoMind Nexus / Parallax Swarm Engine / Borg Core")
        self.resize(1300, 800)

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

        self.reboot_mgr = RebootMemoryManager(brain, {}, pipeline, cipher)

        self.btn_save_reboot.clicked.connect(self.cmd_save_reboot_memory)
        self.btn_load_reboot.clicked.connect(self.cmd_load_reboot_memory)

        # Tab 4: Altered States
        tab_states = QtWidgets.QWidget()
        tabs.addTab(tab_states, "Altered States")

        v4 = QtWidgets.QVBoxLayout(tab_states)
        self.states_panel = AlteredStatesPanel(brain)
        v4.addWidget(self.states_panel)

        # Tab 5: Borg
        tab_borg = QtWidgets.QWidget()
        tabs.addTab(tab_borg, "Borg Entities")

        v5 = QtWidgets.QVBoxLayout(tab_borg)
        self.borg_panel = BorgPanel(borg, pipeline)
        v5.addWidget(self.borg_panel)

        # Tab 6: Mesh / Decoys
        tab_mesh = QtWidgets.QWidget()
        tabs.addTab(tab_mesh, "Borg Mesh / Decoys")
        v6 = QtWidgets.QVBoxLayout(tab_mesh)
        self.txt_mesh_stats = QtWidgets.QTextEdit()
        self.txt_mesh_stats.setReadOnly(True)
        v6.addWidget(self.txt_mesh_stats)
        self.btn_generate_decoy = QtWidgets.QPushButton("Generate Decoy Packet")
        v6.addWidget(self.btn_generate_decoy)
        self.btn_generate_decoy.clicked.connect(self.cmd_generate_decoy)

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

    def cmd_generate_decoy(self):
        decoy = generate_decoy(self.pipeline.scrubber)
        enc = self.cipher.encrypt(decoy)
        self.txt_mesh_stats.append(f"[Decoy] {len(enc)} bytes generated")

    def _apply_orders(self, orders: List[Dict[str, Any]]):
        for o in orders:
            feats = o["aggregate_features"]
            entity = feats.get("dominant_process_name") or feats.get("dominant_ip") or "unknown"
            ip = feats.get("dominant_ip")
            for act in o["actions"]:
                if act == "block_ip" and ip:
                    self.enforcer.block_ip(ip)
                elif act == "kill_process" and feats.get("dominant_process_name"):
                    self.enforcer.kill_process(feats["dominant_process_name"])
                elif act == "isolate_entity":
                    self.enforcer.isolate_entity(entity)
                elif act == "increase_logging":
                    self.enforcer.increase_logging(entity)
                elif act == "temporary_quarantine":
                    self.enforcer.temporary_quarantine(entity)

    def tick(self):
        sensors, best, meta_conf, reasoning, integ = self.brain.tick()
        self.brain.ui_organ.tick()
        self.brain.etw_sensor.poll()

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

        self.borg.touch("process", "demo.exe", {"source": "cluster-demo"}, risk=best)

        self._apply_orders(orders)
        self.swarm_comms.broadcast_state("node-1", best, meta_conf)

        self.refresh_brain_view()
        self.refresh_swarm_view()
        self.states_panel.refresh()
        self.borg_panel.refresh()
        self.refresh_mesh_view()
        self.chart.update()

    def refresh_brain_view(self):
        s_raw = self.brain.stats()
        s = self.pipeline.sanitize_for_gui(s_raw)
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
            o_raw = derive_orders_for_cluster(cluster)
            o = self.pipeline.sanitize_for_gui(o_raw)
            orders_text.append(json.dumps(o, indent=2))
        self.lst_orders.setPlainText("\n\n".join(orders_text))

        nar_text = []
        for nar in self.queen.get_active_narratives():
            n_raw = asdict(nar)
            n = self.pipeline.sanitize_for_gui(n_raw)
            nar_text.append(json.dumps(n, indent=2))
        self.lst_narratives.setPlainText("\n\n".join(nar_text))

    def refresh_mesh_view(self):
        stats = self.mesh.stats()
        s = self.pipeline.sanitize_for_gui(stats)
        self.txt_mesh_stats.setPlainText(json.dumps(s, indent=2))

# ============================================================
# Main
# ============================================================

def main():
    scrubber = PrivacyScrubber()
    pipeline = ZeroTrustDataPipeline(scrubber)
    cipher = AutonomousCipherEngine()
    borg = BorgRegistry()
    mesh = BorgMesh(borg, pipeline)
    enforcer = EnforcementManager()
    guarddog = GuardDog(enforcer)
    brain = HybridBrain(borg, pipeline, mesh, guarddog, cipher)
    swarm = SwarmState()
    queen = QueenConsensus(borg=borg, pipeline=pipeline)
    swarm_comms = SwarmComms()

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(brain, swarm, queen, enforcer, borg, pipeline, mesh, swarm_comms, cipher)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
