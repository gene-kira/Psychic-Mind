#!/usr/bin/env python3
# Full Borg Organism – Tier-5 Autonomous ASI Core
# Upgraded: Prediction, Learning, Defense, Autonomy, GUI, Swarm Mesh, Narrative Reconstruction
# Telemetry: psutil + optional ETW + optional pcap + kernel-sensor stubs
# ML: IsolationForest + Deep Model Scaffold
# Swarm: HTTP + Mesh Protocol Scaffold

import sys
import os
import platform
import importlib
import subprocess
import threading
import asyncio
import time
import hashlib
import zlib
import random
import json
import socket
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set
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
            "scikit-learn",
        ]
        system = platform.system().lower()
        if "windows" in system:
            base += ["wmi", "uiautomation"]
        elif "linux" in system:
            base += ["pyroute2"]
        base += ["cupy"]
        # deep model scaffolds
        base += ["torch"]
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
# Reboot Memory Manager (Hybrid SMB + Local)
# =========================

class RebootMemoryManager:
    def __init__(self,
                 primary_smb: str = r"\\192.168.1.50\borg",
                 fallback_local: str = r"C:\BorgMemory"):
        self.primary_smb = primary_smb.rstrip("\\/")
        self.fallback_local = fallback_local.rstrip("\\/")
        self.last_save_time: Optional[float] = None
        self.last_load_time: Optional[float] = None
        self.last_status: str = "INIT"
        self.last_integrity_score: float = 1.0
        self.max_snapshots = 5
        self._ensure_dirs()

    def _ensure_dirs(self):
        try:
            if self.fallback_local:
                os.makedirs(self.fallback_local, exist_ok=True)
        except Exception:
            pass
        try:
            if self.primary_smb and not self.primary_smb.startswith("http"):
                os.makedirs(self.primary_smb, exist_ok=True)
        except Exception:
            pass

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _write_snapshot(self, base_dir: str, state: Dict[str, Any]) -> bool:
        try:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fname = f"memory_{ts}.json"
            fpath = os.path.join(base_dir, fname)
            raw = json.dumps(state, sort_keys=True).encode("utf-8")
            h = self._hash_bytes(raw)
            with open(fpath, "wb") as f:
                f.write(raw)
            with open(fpath + ".sha256", "w", encoding="utf-8") as f:
                f.write(h)
            snaps = sorted(
                [x for x in os.listdir(base_dir) if x.startswith("memory_") and x.endswith(".json")]
            )
            if len(snaps) > self.max_snapshots:
                for old in snaps[:-self.max_snapshots]:
                    try:
                        os.remove(os.path.join(base_dir, old))
                        if os.path.exists(os.path.join(base_dir, old + ".sha256")):
                            os.remove(os.path.join(base_dir, old + ".sha256"))
                    except Exception:
                        pass
            return True
        except Exception:
            return False

    def _read_latest_valid(self, base_dir: str) -> Optional[Dict[str, Any]]:
        try:
            snaps = sorted(
                [x for x in os.listdir(base_dir) if x.startswith("memory_") and x.endswith(".json")],
                reverse=True,
            )
            for fname in snaps:
                fpath = os.path.join(base_dir, fname)
                hpath = fpath + ".sha256"
                try:
                    with open(fpath, "rb") as f:
                        raw = f.read()
                    if os.path.exists(hpath):
                        with open(hpath, "r", encoding="utf-8") as f:
                            expected = f.read().strip()
                        actual = self._hash_bytes(raw)
                        if expected != actual:
                            continue
                    state = json.loads(raw.decode("utf-8"))
                    return state
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def save_state(self, state: Dict[str, Any]):
        ok_primary = False
        ok_fallback = False
        if self.primary_smb:
            ok_primary = self._write_snapshot(self.primary_smb, state)
        if self.fallback_local:
            ok_fallback = self._write_snapshot(self.fallback_local, state)
        self.last_save_time = time.time()
        if ok_primary and ok_fallback:
            self.last_status = "SAVE_OK_PRIMARY_AND_FALLBACK"
        elif ok_primary:
            self.last_status = "SAVE_OK_PRIMARY_ONLY"
        elif ok_fallback:
            self.last_status = "SAVE_OK_FALLBACK_ONLY"
        else:
            self.last_status = "SAVE_FAILED"

    def load_state(self) -> Optional[Dict[str, Any]]:
        state = None
        if self.primary_smb:
            state = self._read_latest_valid(self.primary_smb)
        if state is None and self.fallback_local:
            state = self._read_latest_valid(self.fallback_local)
        if state is not None:
            self.last_load_time = time.time()
            self.last_status = "LOAD_OK"
        else:
            self.last_status = "LOAD_FAILED"
        return state

    def snapshot(self) -> Dict[str, Any]:
        return {
            "primary_smb": self.primary_smb,
            "fallback_local": self.fallback_local,
            "last_save_time": self.last_save_time,
            "last_load_time": self.last_load_time,
            "last_status": self.last_status,
            "integrity_score": self.last_integrity_score,
        }


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

    def preemptive_hardening(self, global_risk: float):
        if global_risk > 0.7:
            self.risk = max(0.0, self.risk - 0.05)
            self.health = min(1.0, self.health + 0.02)

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
# Queens & Swarm Consensus
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


class SwarmConsensusEngine:
    def __init__(self):
        self.node_votes: Dict[str, Dict[str, float]] = {}
        self.last_consensus: Dict[str, float] = {}
        self.lock = threading.Lock()

    def submit_votes(self, node_id: str, votes: Dict[str, float]):
        with self.lock:
            self.node_votes[node_id] = votes

    def compute_consensus(self) -> Dict[str, float]:
        with self.lock:
            agg: Dict[str, List[float]] = {}
            for node, votes in self.node_votes.items():
                for k, v in votes.items():
                    agg.setdefault(k, []).append(v)
            consensus = {}
            for k, vals in agg.items():
                consensus[k] = sum(vals) / len(vals)
            self.last_consensus = consensus
            return consensus

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "nodes": len(self.node_votes),
                "last_consensus": self.last_consensus,
            }


# =========================
# Attack Chain Engine – temporal, probabilistic, multi-node
# =========================

class AttackChainEngine:
    def __init__(self, window: int = 120, max_gap: int = 60):
        self.window = window
        self.max_gap = max_gap
        self.events: deque = deque()  # {"ts","node","type","data","p"}

        self.patterns = [
            {
                "name": "LOLBIN_ATTACK",
                "sequence": ["proc_spawn", "powershell", "net_connect"],
                "base_score": 0.9,
            },
            {
                "name": "PROCESS_STORM",
                "sequence": ["proc_spawn", "proc_spawn", "proc_spawn", "net_connect"],
                "base_score": 0.8,
            },
            {
                "name": "PERSISTENCE_EXFIL",
                "sequence": ["file_mod", "net_connect"],
                "base_score": 0.85,
            },
            {
                "name": "FULL_ATTACK_CHAIN",
                "sequence": ["proc_start", "net_conn", "file_mod"],
                "base_score": 0.95,
            },
            {
                "name": "SPAWN_STORM",
                "sequence": ["proc_start"] * 10,
                "base_score": 0.8,
            },
            {
                "name": "LOLBIN_BEACON",
                "sequence": ["net_conn", "powershell"],
                "base_score": 0.9,
            },
        ]

    def _cleanup(self, now: float):
        cutoff = now - self.window
        while self.events and self.events[0]["ts"] < cutoff:
            self.events.popleft()

    def add_event(self, node_id: str, event_type: str, data: Dict, probability: float = 1.0):
        now = time.time()
        self.events.append({
            "ts": now,
            "node": node_id,
            "type": event_type,
            "data": data,
            "p": max(0.0, min(1.0, probability)),
        })
        self._cleanup(now)

    def _score_pattern_for_node(
        self,
        node_id: str,
        pattern: Dict[str, Any],
    ) -> Tuple[float, Optional[Tuple[float, float]]]:
        seq = pattern["sequence"]
        base = pattern["base_score"]

        node_events = [e for e in self.events if e["node"] == node_id]
        if not node_events:
            return 0.0, None

        best_score = 0.0
        best_window = None

        for i in range(len(node_events)):
            if node_events[i]["type"] != seq[0]:
                continue
            idx = 1
            current_prob = node_events[i]["p"]
            start_ts = node_events[i]["ts"]
            last_ts = start_ts
            for j in range(i + 1, len(node_events)):
                if node_events[j]["ts"] - last_ts > self.max_gap:
                    break
                if node_events[j]["type"] == seq[idx]:
                    current_prob *= node_events[j]["p"]
                    last_ts = node_events[j]["ts"]
                    idx += 1
                    if idx == len(seq):
                        total_span = last_ts - start_ts
                        time_factor = max(0.3, min(1.0, 1.0 - (total_span / max(self.window, 1.0))))
                        score = base * current_prob * time_factor
                        if score > best_score:
                            best_score = score
                            best_window = (start_ts, last_ts)
                        break

        return best_score, best_window

    def detect(self) -> List[Tuple[str, float, Set[str]]]:
        now = time.time()
        self._cleanup(now)
        if not self.events:
            return []

        nodes = {e["node"] for e in self.events}
        chains: List[Tuple[str, float, Set[str]]] = []

        for pattern in self.patterns:
            node_scores: Dict[str, float] = {}
            for n in nodes:
                s, _ = self._score_pattern_for_node(n, pattern)
                if s > 0.0:
                    node_scores[n] = s

            if not node_scores:
                continue

            involved_nodes = set(node_scores.keys())
            avg_score = sum(node_scores.values()) / len(node_scores)
            multi_node_factor = 1.0 + 0.15 * (len(involved_nodes) - 1)
            final_score = max(0.0, min(1.0, avg_score * multi_node_factor))

            if final_score > 0.5:
                chains.append((pattern["name"], round(final_score, 3), involved_nodes))

        return chains


# =========================
# Attack Narrative Reconstruction
# =========================

class AttackNarrativeReconstructor:
    """
    Builds human-readable narratives from attack chains + events.
    """

    def __init__(self, chain_engine: AttackChainEngine):
        self.chain_engine = chain_engine
        self.narratives: List[Dict[str, Any]] = []

    def reconstruct(self) -> List[Dict[str, Any]]:
        chains = self.chain_engine.detect()
        narratives = []
        for name, score, nodes in chains:
            story = {
                "pattern": name,
                "score": score,
                "nodes": list(nodes),
                "summary": self._build_summary(name, score, nodes),
            }
            narratives.append(story)
        self.narratives = narratives[-50:]
        return narratives

    def _build_summary(self, name: str, score: float, nodes: Set[str]) -> str:
        base = f"Detected pattern {name} with confidence {score:.2f} across nodes {', '.join(nodes)}."
        if "LOLBIN" in name:
            base += " Likely abuse of built-in binaries for execution or lateral movement."
        if "PERSISTENCE" in name:
            base += " Indicates possible persistence followed by data exfiltration."
        if "STORM" in name:
            base += " High process churn suggests automated tooling or malware."
        return base

    def snapshot(self) -> List[Dict[str, Any]]:
        return self.narratives


# =========================
# Borg Queen
# =========================

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
# Optional Escalation Stubs
# =========================

class AgvraMbseModule:
    def __init__(self):
        self.enabled = True
        self.last_sync = None

    def tick(self):
        self.last_sync = time.time()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "last_sync": self.last_sync,
        }


class ScionAdapter:
    def __init__(self):
        self.enabled = True
        self.ras_mode = "NORMAL"

    def tick(self, risk: float):
        if risk > 0.7:
            self.ras_mode = "HARDENED"
        elif risk > 0.4:
            self.ras_mode = "ELEVATED"
        else:
            self.ras_mode = "NORMAL"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ras_mode": self.ras_mode,
        }


class AsiArchCompliance:
    def __init__(self):
        self.enabled = True
        self.control_level = "NORMAL"
        self.last_audit = None

    def tick(self, risk: float):
        self.last_audit = time.time()
        if risk > 0.7:
            self.control_level = "STRICT"
        elif risk > 0.4:
            self.control_level = "TIGHT"
        else:
            self.control_level = "NORMAL"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "control_level": self.control_level,
            "last_audit": self.last_audit,
        }


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
        self.ghost_sensitive = self.ghost_sensitive
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

    def load_from_state(self, state: Dict[str, Any]):
        rs = state.get("retention_seconds")
        if isinstance(rs, (int, float)):
            self.retention_seconds = float(rs)
        gs = state.get("ghost_sensitive")
        if isinstance(gs, bool):
            self.ghost_sensitive = gs
        pn = state.get("phantom_node_enabled")
        if isinstance(pn, bool):
            self.phantom_node_enabled = pn


# =========================
# Self-Rewriting Agent
# =========================

class SelfRewritingAgent:
    def __init__(self, learning_rate: float = 0.05):
        self.weights: Dict[str, float] = {
            "risk": 0.5,
            "health": 0.5,
            "turbulence": 0.5,
            "swarm_risk": 0.5,
        }
        self.learning_rate = learning_rate
        self.mutations: int = 0
        self.backend = "local"

    def mutate(self, signal: Dict[str, float]):
        for k, v in signal.items():
            if k not in self.weights:
                continue
            delta = (v - 0.5) * self.learning_rate
            self.weights[k] = max(0.0, min(1.0, self.weights[k] + delta))
        self.mutations += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
            "mutations": self.mutations,
            "backend": self.backend,
            "learning_rate": round(self.learning_rate, 3),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "learning_rate": self.learning_rate,
            "mutations": self.mutations,
        }

    def load_from_state(self, state: Dict[str, Any]):
        w = state.get("weights")
        if isinstance(w, dict):
            self.weights = {k: float(v) for k, v in w.items() if isinstance(v, (int, float))}
        lr = state.get("learning_rate")
        if isinstance(lr, (int, float)):
            self.learning_rate = float(lr)
        m = state.get("mutations")
        if isinstance(m, int):
            self.mutations = m


# =========================
# Autonomous Cipher Engine
# =========================

class AutonomousCipherEngine:
    def __init__(self):
        self.posture = "NORMAL"
        self.current_key_id = "key-0001"
        self.rotation_interval = 600.0
        self.last_rotation = time.time()

    def tick(self, risk: float):
        now = time.time()
        if risk > 0.7:
            self.posture = "HARDENED"
            self.rotation_interval = 300.0
        elif risk > 0.4:
            self.posture = "ELEVATED"
            self.rotation_interval = 450.0
        else:
            self.posture = "NORMAL"
            self.rotation_interval = 600.0

        if now - self.last_rotation >= self.rotation_interval:
            self.current_key_id = f"key-{int(now)}"
            self.last_rotation = now

    def snapshot(self) -> Dict[str, Any]:
        return {
            "posture": self.posture,
            "current_key_id": self.current_key_id,
            "rotation_interval": int(self.rotation_interval),
            "seconds_since_rotation": int(time.time() - self.last_rotation),
        }


# =========================
# ML Anomaly Engine (IsolationForest + Deep Model Scaffold)
# =========================

class MLAnomalyEngine:
    def __init__(self, autoloader: BorgAutoloader):
        self.autoloader = autoloader
        self.isolation_forest = None
        self.deep_model = None
        self.last_train = None
        self.buffer: List[List[float]] = []
        self.max_buffer = 1000
        self._init_models()

    def _init_models(self):
        skl = self.autoloader.load("sklearn.ensemble")
        torch = self.autoloader.load("torch")
        if skl is not None:
            try:
                IsolationForest = getattr(skl, "IsolationForest")
                self.isolation_forest = IsolationForest(
                    n_estimators=100,
                    contamination=0.05,
                    random_state=42,
                )
            except Exception:
                self.isolation_forest = None
        if torch is not None:
            try:
                import torch.nn as nn
                class TinyNet(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc1 = nn.Linear(8, 16)
                        self.fc2 = nn.Linear(16, 8)
                        self.fc3 = nn.Linear(8, 1)

                    def forward(self, x):
                        x = torch.relu(self.fc1(x))
                        x = torch.relu(self.fc2(x))
                        x = torch.sigmoid(self.fc3(x))
                        return x
                self.deep_model = TinyNet()
            except Exception:
                self.deep_model = None

    def add_sample(self, features: List[float]):
        self.buffer.append(features)
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer:]

    def train_if_needed(self):
        if self.isolation_forest is None:
            return
        if len(self.buffer) < 100:
            return
        now = time.time()
        if self.last_train and now - self.last_train < 300:
            return
        import numpy as np
        X = np.array(self.buffer, dtype=float)
        try:
            self.isolation_forest.fit(X)
            self.last_train = now
        except Exception:
            pass

    def score(self, features: List[float]) -> float:
        import numpy as np
        x = np.array(features, dtype=float).reshape(1, -1)
        iso_score = 0.5
        if self.isolation_forest is not None:
            try:
                s = self.isolation_forest.decision_function(x)[0]
                iso_score = float(1.0 - (s + 1.0) / 2.0)
            except Exception:
                iso_score = 0.5
        deep_score = 0.5
        if self.deep_model is not None:
            try:
                import torch
                with torch.no_grad():
                    t = torch.tensor(x, dtype=torch.float32)
                    out = self.deep_model(t)
                    deep_score = float(out.item())
            except Exception:
                deep_score = 0.5
        return max(0.0, min(1.0, 0.6 * iso_score + 0.4 * deep_score))


# =========================
# Brain Autonomy
# =========================

class BrainAutonomy:
    def __init__(self, chain_engine: AttackChainEngine, learning_rate: float = 0.05):
        self.appetite = 1.0
        self.threshold_risk = 1.5
        self.horizon = 120
        self.dampening = 0.8
        self.cache_aggressiveness = 0.5
        self.thread_expansion = 1.0
        self.success_score = 0.0
        self.last_calibration = datetime.utcnow()
        self.chain_engine = chain_engine
        self.learning_rate = learning_rate

    def record_outcome(self, true_positive: bool, false_positive: bool):
        if true_positive:
            self.success_score += 1.0 * self.learning_rate
        if false_positive:
            self.success_score -= 0.5 * self.learning_rate

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
            "learning_rate": round(self.learning_rate, 3),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "success_score": self.success_score,
            "learning_rate": self.learning_rate,
        }

    def load_from_state(self, state: Dict[str, Any]):
        ss = state.get("success_score")
        if isinstance(ss, (int, float)):
            self.success_score = float(ss)
        lr = state.get("learning_rate")
        if isinstance(lr, (int, float)):
            self.learning_rate = float(lr)


# =========================
# Telemetry: psutil + optional ETW + pcap + kernel sensor stubs
# =========================

try:
    import psutil
except Exception:
    psutil = None

class WindowsETWListener:
    def __init__(self):
        self.available = False
        if platform.system().lower().startswith("win"):
            try:
                self.available = True
            except Exception:
                self.available = False

    def poll_events(self) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        events = []
        now = time.time()
        events.append({"ts": now, "type": "proc_start", "proc": "powershell.exe", "p": 0.8})
        return events


class PcapListener:
    def __init__(self):
        self.available = False
        try:
            import scapy  # noqa
            self.available = True
        except Exception:
            self.available = False

    def poll_packets(self) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        # Placeholder synthetic packets
        return [{"ts": time.time(), "type": "net_conn", "dst": "8.8.8.8", "p": 0.6}]


class KernelSensorStub:
    """
    Stub for kernel-level sensors: callbacks, driver hooks, etc.
    """
    def __init__(self):
        self.available = False
        # Real implementation would hook kernel callbacks / drivers

    def poll(self) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        return []


class TelemetryCollector:
    def __init__(self, node_id: str = "node-001"):
        self.node_id = node_id
        self.etw = WindowsETWListener()
        self.pcap = PcapListener()
        self.kernel = KernelSensorStub()
        self.last_net_scan = 0.0
        self.rogue_hosts: Set[str] = set()
        self.blocked_ips: Set[str] = set()

    def collect_psutil_metrics(self) -> Dict[str, float]:
        if psutil is None:
            return {"cpu": 0.0, "mem": 0.0, "disk": 0.0, "net": 0.0}
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0
        net = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv if psutil.net_io_counters() else 0
        return {
            "cpu": cpu / 100.0,
            "mem": mem / 100.0,
            "disk": min(1.0, disk / (1024 * 1024 * 1024)),
            "net": min(1.0, net / (1024 * 1024 * 1024)),
        }

    def scan_rogue_hosts(self):
        if psutil is None:
            return
        now = time.time()
        if now - self.last_net_scan < 10.0:
            return
        self.last_net_scan = now
        try:
            conns = psutil.net_connections(kind="inet")
            suspicious = set()
            for c in conns:
                raddr = c.raddr
                if not raddr:
                    continue
                ip = raddr.ip
                if ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("172.16."):
                    continue
                suspicious.add(ip)
            self.rogue_hosts = suspicious
        except Exception:
            pass

    def auto_block_ip(self, ip: str):
        if ip in self.blocked_ips:
            return
        if platform.system().lower().startswith("win"):
            try:
                subprocess.run(
                    ["netsh", "advfirewall", "firewall", "add", "rule",
                     f"name=BorgBlock_{ip}", "dir=out", "action=block", f"remoteip={ip}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                self.blocked_ips.add(ip)
            except Exception:
                pass

    def feed_attack_chain(self, chain_engine: AttackChainEngine):
        if psutil is not None:
            try:
                for p in psutil.process_iter(attrs=["name"]):
                    name = (p.info.get("name") or "").lower()
                    if "powershell" in name:
                        chain_engine.add_event(self.node_id, "powershell", {"proc": name}, probability=0.8)
                    elif name:
                        chain_engine.add_event(self.node_id, "proc_start", {"proc": name}, probability=0.3)
            except Exception:
                pass
        for e in self.etw.poll_events():
            chain_engine.add_event(self.node_id, e["type"], e, probability=e.get("p", 0.5))
        for p in self.pcap.poll_packets():
            chain_engine.add_event(self.node_id, p["type"], p, probability=p.get("p", 0.5))
        for k in self.kernel.poll():
            chain_engine.add_event(self.node_id, k["type"], k, probability=k.get("p", 0.5))


# =========================
# Swarm Sync over HTTP + Mesh Protocol Scaffold
# =========================

class SwarmSyncClient:
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.enabled = bool(peers)
        try:
            import requests  # noqa
            self.requests = importlib.import_module("requests")
        except Exception:
            self.requests = None
            self.enabled = False

    def push_state(self, state: Dict[str, Any]):
        if not self.enabled or self.requests is None:
            return
        for peer in self.peers:
            url = f"http://{peer}/borg_swarm/{self.node_id}"
            try:
                self.requests.post(url, json=state, timeout=0.5)
            except Exception:
                continue

    def pull_state(self) -> Dict[str, Any]:
        if not self.enabled or self.requests is None:
            return {}
        merged = {}
        for peer in self.peers:
            url = f"http://{peer}/borg_swarm_snapshot"
            try:
                r = self.requests.get(url, timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    merged[peer] = data
            except Exception:
                continue
        return merged


class SwarmMeshProtocol:
    """
    Tier-4/5 mesh protocol scaffold.
    Real implementation would use UDP, gossip, CRDTs, signatures, etc.
    """

    def __init__(self, node_id: str, listen_port: int = 9999):
        self.node_id = node_id
        self.listen_port = listen_port
        self.running = False
        self.peers: Set[Tuple[str, int]] = set()
        self.sock = None

    def start(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("", self.listen_port))
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()
        except Exception:
            self.running = False

    def _loop(self):
        while self.running and self.sock:
            try:
                data, addr = self.sock.recvfrom(65535)
                # placeholder: decode + merge
            except Exception:
                continue

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

    def broadcast_state(self, state: Dict[str, Any]):
        if not self.sock:
            return
        payload = json.dumps({"node": self.node_id, "state": state}).encode("utf-8")
        for (ip, port) in self.peers:
            try:
                self.sock.sendto(payload, (ip, port))
            except Exception:
                continue


# =========================
# BrainCore – Upgraded
# =========================

class BrainCore:
    MISSIONS = ["AUTO", "PROTECT", "STABILITY", "LEARN", "OPTIMIZE"]
    ENVIRONMENTS = ["CALM", "TENSE", "DANGER"]
    HORIZONS = ["SHORT", "MEDIUM", "LONG"]
    REGIMES = ["STABLE", "CHAOTIC", "RISING", "COOLING"]
    META_STATES = ["Hyper-Flow", "Deep-Dream", "Sentinel", "Recovery-Flow"]
    EMOTIONS = ["NEUTRAL", "ALERT", "ANXIOUS", "CONFIDENT", "OVERWHELMED"]

    def __init__(self,
                 chain_engine: AttackChainEngine,
                 ml_engine: MLAnomalyEngine,
                 learning_rate: float = 0.05,
                 node_id: str = "node-001",
                 swarm_peers: Optional[List[str]] = None):
        self.mission = "AUTO"
        self.environment = "CALM"
        self.opportunity_score = 0.3
        self.risk_score = 0.2
        self.anticipation = "Stable window"

        self.risk_tolerance = 0.5
        self.opportunity_bias = 0.5

        self.prediction_horizon = "MEDIUM"
        self.anomaly_risk = 0.2
        self.drive_risk = 0.2
        self.hive_risk = 0.2
        self.collective_health = 0.8
        self.health_trend = 0.0

        self.anomaly_sensitivity = 0.5
        self.collective_weighting = 0.5

        self.predictions = {
            "1s": 0.2,
            "5s": 0.2,
            "30s": 0.2,
            "120s": 0.2,
        }

        self.regime = "STABLE"
        self.meta_confidence = 0.7

        self.pattern_memory: List[Dict[str, Any]] = []
        self.episodic_memory: List[Dict[str, Any]] = []
        self.mode_memory: Dict[str, Dict[str, float]] = {}

        self.meta_state = "Sentinel"
        self.meta_momentum = 0.0
        self.meta_inertia = 0.5

        self.short_damp = 0.0
        self.long_damp = 0.0

        self.reasoning_heatmap: Dict[str, float] = {}
        self.internal_dialogue: deque = deque(maxlen=20)

        self.codex = AdaptiveCodexMutation()
        self.agent = SelfRewritingAgent(learning_rate=learning_rate)
        self.cipher = AutonomousCipherEngine()
        self.autonomy = BrainAutonomy(chain_engine, learning_rate=learning_rate)

        self.agvra = AgvraMbseModule()
        self.scion = ScionAdapter()
        self.asi_arch = AsiArchCompliance()

        self._tick_count = 0
        self._risk_history = deque(maxlen=120)

        self.swarm_consensus = SwarmConsensusEngine()
        self.node_id = node_id
        self.swarm_sync = SwarmSyncClient(node_id=node_id, peers=swarm_peers or [])
        self.mesh = SwarmMeshProtocol(node_id=node_id)

        self.emotional_state = "NEUTRAL"
        self.ml_engine = ml_engine
        self.tier = 5  # Tier-5 autonomous

    def _log_internal(self, msg: str):
        ts = datetime.utcnow().strftime("%H:%M:%S")
        self.internal_dialogue.append(f"[{ts}] {msg}")

    def set_mission(self, mission: str):
        if mission in self.MISSIONS:
            self.mission = mission
            self._log_internal(f"Mission set to {mission}")

    def set_environment(self, env: str):
        if env in self.ENVIRONMENTS:
            self.environment = env

    def set_horizon(self, horizon: str):
        if horizon in self.HORIZONS:
            self.prediction_horizon = horizon

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
        if self.regime == "STABLE":
            factor = 0.8
        elif self.regime == "CHAOTIC":
            factor = 1.2
        elif self.regime == "RISING":
            factor = 1.4
        else:
            factor = 0.7

        self.predictions["1s"] = max(0.0, min(1.0, base_risk * factor * (1.0 + 0.2 * turbulence)))
        self.predictions["5s"] = max(0.0, min(1.0, base_risk * factor * (1.1 + 0.3 * turbulence)))
        self.predictions["30s"] = max(0.0, min(1.0, base_risk * factor * (1.2 + 0.4 * turbulence)))
        self.predictions["120s"] = max(0.0, min(1.0, base_risk * factor * (1.3 + 0.5 * turbulence)))

    def _meta_confidence_fusion(self, variance: float, trend: float, turbulence: float, reinforcement: float) -> float:
        v_term = max(0.0, 1.0 - min(variance * 10, 1.0))
        t_term = max(0.0, 1.0 - min(abs(trend) * 5, 1.0))
        turb_term = max(0.0, 1.0 - min(turbulence, 1.0))
        r_term = max(0.0, min(1.0, 0.5 + reinforcement * 0.1))
        conf = 0.25 * (v_term + t_term + turb_term + r_term)
        return max(0.0, min(1.0, conf))

    def _multi_engine_vote(self, base_risk: float, variance: float, trend: float, turbulence: float, baseline_dev: float, organ_vector: Dict[str, float], ml_score: float) -> float:
        engines = {}
        engines["ewma"] = base_risk
        engines["trend"] = max(0.0, min(1.0, 0.5 + trend))
        engines["variance"] = max(0.0, min(1.0, variance * 5))
        engines["turbulence"] = max(0.0, min(1.0, turbulence))
        engines["baseline_dev"] = max(0.0, min(1.0, baseline_dev))
        engines["reinforcement"] = max(0.0, min(1.0, 0.5 + self.autonomy.success_score * 0.05))
        engines["ml"] = max(0.0, min(1.0, ml_score))
        if organ_vector:
            organ_avg = sum(organ_vector.values()) / len(organ_vector)
        else:
            organ_avg = 0.0
        engines["organs"] = max(0.0, min(1.0, organ_avg))

        weights = {
            "ewma": 0.15,
            "trend": 0.1,
            "variance": 0.1,
            "turbulence": 0.1,
            "baseline_dev": 0.1,
            "reinforcement": 0.1,
            "ml": 0.2,
            "organs": 0.15,
        }

        num = sum(engines[k] * weights[k] for k in engines)
        den = sum(weights.values())
        best_guess = num / den if den > 0 else base_risk

        self.reasoning_heatmap = {k: round(engines[k] * weights[k], 3) for k in engines}
        return max(0.0, min(1.0, best_guess))

    def _update_meta_state_from_performance(self):
        s = self.autonomy.success_score
        inertia = self.meta_inertia
        if self.meta_state == "Hyper-Flow" and s < -2 * inertia:
            self.meta_state = "Sentinel"
            self._log_internal("Dropping from Hyper-Flow to Sentinel due to poor outcomes.")
        elif self.meta_state == "Sentinel" and s > 3 * inertia:
            self.meta_state = "Hyper-Flow"
            self._log_internal("Escalating to Hyper-Flow; Sentinel has been consistently successful.")
        elif self.meta_state == "Recovery-Flow" and self.regime == "COOLING":
            self.meta_state = "Deep-Dream"
            self._log_internal("Sliding into Deep-Dream; recovery is stabilizing.")
        elif self.meta_state == "Deep-Dream" and self.regime == "RISING":
            self.meta_state = "Sentinel"
            self._log_internal("Waking from Deep-Dream into Sentinel; load is rising.")

    def _update_emotional_state(self):
        r = self.risk_score
        h = self.collective_health
        if r < 0.3 and h > 0.7:
            self.emotional_state = "CONFIDENT"
        elif r > 0.7 and h < 0.5:
            self.emotional_state = "OVERWHELMED"
        elif r > 0.6:
            self.emotional_state = "ANXIOUS"
        elif r > 0.4:
            self.emotional_state = "ALERT"
        else:
            self.emotional_state = "NEUTRAL"

    def tick(self,
             putty: SillyPuttyDataGatherer,
             organs: List[BaseOrgan],
             swarm_view: Dict[str, Any],
             metrics: Dict[str, float]):
        self._tick_count += 1

        report = putty.report()
        anomaly_density = report["anomaly_density"]
        turbulence = report["turbulence"]
        pressure = report["pressure"]

        organ_vector = {o.name: o.risk for o in organs}
        organ_health_avg = sum(o.health for o in organs) / max(len(organs), 1)
        organ_risk_avg = sum(o.risk for o in organs) / max(len(organs), 1)

        swarm_risk = swarm_view.get("swarm_risk", 0.0)
        baseline_dev = abs(anomaly_density - self.anomaly_sensitivity)

        base_risk = 0.4 * anomaly_density + 0.3 * organ_risk_avg + 0.3 * swarm_risk
        self._risk_history.append(base_risk)
        self._update_regime()

        variance = 0.0
        trend = 0.0
        if len(self._risk_history) > 1:
            vals = list(self._risk_history)
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / len(vals)
            trend = vals[-1] - vals[0]

        features = [
            anomaly_density,
            organ_risk_avg,
            swarm_risk,
            pressure,
            turbulence,
            metrics.get("cpu", 0.0),
            metrics.get("mem", 0.0),
            metrics.get("net", 0.0),
        ]
        self.ml_engine.add_sample(features)
        self.ml_engine.train_if_needed()
        ml_score = self.ml_engine.score(features)

        best_guess = self._multi_engine_vote(base_risk, variance, trend, turbulence, baseline_dev, organ_vector, ml_score)
        self.risk_score = best_guess
        self.anomaly_risk = anomaly_density
        self.drive_risk = organ_risk_avg
        self.hive_risk = swarm_risk
        self.collective_health = organ_health_avg
        self.health_trend = trend

        self._multi_horizon_forecast(self.risk_score, turbulence)
        self.meta_confidence = self._meta_confidence_fusion(variance, trend, turbulence, self.autonomy.success_score)

        self.autonomy.auto_tune()
        self.autonomy.auto_calibrate_if_due(anomaly_density, pressure)

        ghost = self.codex.detect_ghost_sync(self.risk_score, self.collective_health, turbulence)
        if ghost:
            self.codex.mutate_rules("ghost_sync_detected")
            self._log_internal("Ghost sync detected; codex mutated.")
        elif self._tick_count % 100 == 0:
            self.codex.mutate_rules("periodic_adjustment")

        self.agent.mutate({
            "risk": self.risk_score,
            "health": self.collective_health,
            "turbulence": turbulence,
            "swarm_risk": swarm_risk,
        })

        self.cipher.tick(self.risk_score)
        self.agvra.tick()
        self.scion.tick(self.risk_score)
        self.asi_arch.tick(self.risk_score)

        self._update_meta_state_from_performance()
        self._update_emotional_state()

        episode = {
            "ts": time.time(),
            "risk": self.risk_score,
            "anomaly_density": anomaly_density,
            "turbulence": turbulence,
            "swarm_risk": swarm_risk,
            "meta_state": self.meta_state,
            "emotion": self.emotional_state,
        }
        self.episodic_memory.append(episode)
        if len(self.episodic_memory) > 500:
            self.episodic_memory = self.episodic_memory[-500:]

        self.mode_memory[self.meta_state] = {
            "avg_risk": self.risk_score,
            "avg_anomaly": anomaly_density,
            "avg_swarm": swarm_risk,
        }

        if self.mission == "PROTECT" and self.risk_score > self.autonomy.threshold_risk:
            self._log_internal("Mission PROTECT: risk above threshold; would escalate defenses.")

        if self.swarm_sync.enabled and self._tick_count % 20 == 0:
            state = {
                "node_id": self.node_id,
                "risk": self.risk_score,
                "health": self.collective_health,
                "meta_state": self.meta_state,
                "emotion": self.emotional_state,
            }
            self.swarm_sync.push_state(state)

        if self.mesh.running and self._tick_count % 30 == 0:
            mesh_state = {
                "risk": self.risk_score,
                "meta_state": self.meta_state,
                "emotion": self.emotional_state,
            }
            self.mesh.broadcast_state(mesh_state)

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
            "emotion": self.emotional_state,
            "tier": self.tier,
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
            "short_damp": round(self.short_damp, 3),
            "long_damp": round(self.long_damp, 3),
            "meta_confidence": round(self.meta_confidence, 3),
        }

    def codex_snapshot(self) -> Dict[str, Any]:
        return self.codex.snapshot()

    def agent_snapshot(self) -> Dict[str, Any]:
        return self.agent.snapshot()

    def cipher_snapshot(self) -> Dict[str, Any]:
        return self.cipher.snapshot()

    def autonomy_snapshot(self) -> Dict[str, Any]:
        return self.autonomy.summary()

    def escalation_snapshot(self) -> Dict[str, Any]:
        return {
            "agvra": self.agvra.snapshot(),
            "scion": self.scion.snapshot(),
            "asi_arch": self.asi_arch.snapshot(),
        }

    def swarm_snapshot(self) -> Dict[str, Any]:
        return self.swarm_consensus.snapshot()

    def reasoning_snapshot(self) -> Dict[str, float]:
        return self.reasoning_heatmap

    def internal_dialogue_snapshot(self) -> List[str]:
        return list(self.internal_dialogue)

    def export_learning_state(self) -> Dict[str, Any]:
        return {
            "pattern_memory": self.pattern_memory[-500:],
            "episodic_memory": self.episodic_memory[-200:],
            "autonomy": self.autonomy.export_state(),
            "agent": self.agent.export_state(),
            "codex": {
                "retention_seconds": self.codex.retention_seconds,
                "ghost_sensitive": self.codex.ghost_sensitive,
                "phantom_node_enabled": self.codex.phantom_node_enabled,
            },
        }

    def load_learning_state(self, state: Dict[str, Any]):
        pm = state.get("pattern_memory")
        if isinstance(pm, list):
            self.pattern_memory = pm[-500:]
        em = state.get("episodic_memory")
        if isinstance(em, list):
            self.episodic_memory = em[-200:]
        auto_state = state.get("autonomy")
        if isinstance(auto_state, dict):
            self.autonomy.load_from_state(auto_state)
        agent_state = state.get("agent")
        if isinstance(agent_state, dict):
            self.agent.load_from_state(agent_state)
        codex_state = state.get("codex")
        if isinstance(codex_state, dict):
            self.codex.load_from_state(codex_state)


# =========================
# GUI Cockpit
# =========================

GUI_AVAILABLE = False
try:
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QGroupBox, QListWidget, QTextEdit
    )
    from PySide6.QtCore import Qt, QTimer
    GUI_AVAILABLE = True
except Exception:
    GUI_AVAILABLE = False


if GUI_AVAILABLE:
    class BorgCockpit(QWidget):
        def __init__(self,
                     autoloader: BorgAutoloader,
                     putty: SillyPuttyDataGatherer,
                     organs: List[BaseOrgan],
                     self_integrity: SelfIntegrityOrgan,
                     queen: BorgQueen,
                     rt_queen: RealTimeQueen,
                     swarm_consensus: SwarmConsensusEngine,
                     chain_engine: AttackChainEngine,
                     brain: BrainCore,
                     reboot_mgr: RebootMemoryManager,
                     narrative: AttackNarrativeReconstructor):
            super().__init__()
            self.autoloader = autoloader
            self.putty = putty
            self.organs = organs
            self.self_integrity = self_integrity
            self.queen = queen
            self.rt_queen = rt_queen
            self.swarm_consensus = swarm_consensus
            self.chain_engine = chain_engine
            self.brain = brain
            self.reboot_mgr = reboot_mgr
            self.narrative = narrative

            self.setWindowTitle("Borg Organism Cockpit – Full Brain v5")
            self.resize(1400, 1000)

            root = QVBoxLayout()

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

            organ_group = QGroupBox("Organs")
            organ_layout = QVBoxLayout()
            self.label_organs = QLabel()
            self.label_self_integrity = QLabel()
            organ_layout.addWidget(self.label_organs)
            organ_layout.addWidget(self.label_self_integrity)
            organ_group.setLayout(organ_layout)

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
            opp_row.addWidget(QLabel("Opportunity bias"))
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

            pi_group = QGroupBox("Predictive Intelligence / Reasoning / Escalations")
            pi_layout = QVBoxLayout()
            self.pi_label_risks = QLabel()
            self.pi_label_health = QLabel()
            self.pi_label_horizons = QLabel()
            self.pi_label_conf = QLabel()
            self.label_codex = QLabel()
            self.label_agent = QLabel()
            self.label_cipher = QLabel()
            self.label_reasoning = QLabel()
            self.label_escalations = QLabel()

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
            weight_row.addWidget(QLabel("Collective weighting"))
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
            pi_layout.addWidget(self.label_escalations)
            pi_group.setLayout(pi_layout)

            auto_group = QGroupBox("Autoloader / Swarm")
            auto_layout = QVBoxLayout()
            self.list_loaded = QListWidget()
            self.list_failed = QListWidget()
            self.label_swarm = QLabel()
            auto_layout.addWidget(QLabel("Loaded Libraries"))
            auto_layout.addWidget(self.list_loaded)
            auto_layout.addWidget(QLabel("Failed Libraries"))
            auto_layout.addWidget(self.list_failed)
            auto_layout.addWidget(self.label_swarm)
            auto_group.setLayout(auto_layout)

            reboot_group = QGroupBox("Reboot Memory Panel")
            reboot_layout = QVBoxLayout()
            self.label_reboot_paths = QLabel()
            self.label_reboot_status = QLabel()
            self.btn_reboot_save = QPushButton("Save Memory Now")
            self.btn_reboot_load = QPushButton("Force Load Memory")
            self.btn_reboot_save.clicked.connect(self._force_save_memory)
            self.btn_reboot_load.clicked.connect(self._force_load_memory)
            reboot_layout.addWidget(self.label_reboot_paths)
            reboot_layout.addWidget(self.label_reboot_status)
            reboot_layout.addWidget(self.btn_reboot_save)
            reboot_layout.addWidget(self.btn_reboot_load)
            reboot_group.setLayout(reboot_layout)

            dialogue_group = QGroupBox("Internal Dialogue / Meta-State Timeline / Narratives")
            dialogue_layout = QVBoxLayout()
            self.text_dialogue = QTextEdit()
            self.text_dialogue.setReadOnly(True)
            dialogue_layout.addWidget(self.text_dialogue)
            dialogue_group.setLayout(dialogue_layout)

            root.addWidget(top_group)
            root.addWidget(organ_group)
            root.addWidget(sa_group)
            root.addWidget(pi_group)
            root.addWidget(auto_group)
            root.addWidget(reboot_group)
            root.addWidget(dialogue_group)
            self.setLayout(root)

            self.timer = QTimer()
            self.timer.timeout.connect(self._tick)
            self.timer.start(500)
            self._tick()

        def _force_save_memory(self):
            state = {
                "ts": time.time(),
                "brain_learning": self.brain.export_learning_state(),
            }
            self.reboot_mgr.save_state(state)

        def _force_load_memory(self):
            state = self.reboot_mgr.load_state()
            if isinstance(state, dict):
                bl = state.get("brain_learning", {})
                if isinstance(bl, dict):
                    self.brain.load_learning_state(bl)

        def _tick(self):
            report = self.putty.report()
            queen_status = self.queen.status()
            rt_risk = self.rt_queen.global_risk()
            chains = self.chain_engine.detect()
            auto_status = self.autoloader.status()

            autonomy = self.brain.autonomy_snapshot()
            sa = self.brain.situational_snapshot()
            pi = self.brain.predictive_snapshot()
            codex = self.brain.codex_snapshot()
            agent = self.brain.agent_snapshot()
            cipher = self.brain.cipher_snapshot()
            reasoning = self.brain.reasoning_snapshot()
            escalations = self.brain.escalation_snapshot()
            reboot = self.reboot_mgr.snapshot()
            swarm = self.brain.swarm_snapshot()
            dialogue = self.brain.internal_dialogue_snapshot()
            narratives = self.narrative.snapshot()

            self.label_putty.setText(
                f"Putty - Mass: {report['mass']} | EffGain: {report['efficiency_gain']} | Flows: {report['unique_flows']} | State: {report['state']} | Meta: {report['meta_state']}"
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
                f"Autonomy - appetite={autonomy['appetite']} thr={autonomy['threshold_risk']} hor={autonomy['horizon']} damp={autonomy['dampening']} cache={autonomy['cache_aggressiveness']} threads={autonomy['thread_expansion']} lr={autonomy['learning_rate']}"
            )

            organ_str = " | ".join([f"{o.name}: H={o.health:.2f} R={o.risk:.2f} I={o.integrity:.2f}" for o in self.organs])
            self.label_organs.setText(f"Organs: {organ_str}")
            self.label_self_integrity.setText(
                f"Self-Integrity - H={self.self_integrity.health:.2f} R={self.self_integrity.risk:.2f} I={self.self_integrity.integrity:.2f}"
            )

            self.sa_label_state.setText(
                f"Mission: {sa['mission']} | Env: {sa['environment']} | MetaState: {sa['meta_state']} | Regime: {sa['regime']} | Emotion: {sa['emotion']} | Tier: {sa['tier']} | RiskTol: {sa['risk_tolerance']} | OppBias: {sa['opportunity_bias']}"
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
                f"Agent - Weights: {agent['weights']} | Mutations: {agent['mutations']} | Backend: {agent['backend']} | LR: {agent['learning_rate']}"
            )
            self.label_cipher.setText(
                f"Cipher - Posture: {cipher['posture']} | KeyID: {cipher['current_key_id']} | RotInt: {cipher['rotation_interval']}s | SinceRot: {cipher['seconds_since_rotation']}s"
            )

            if reasoning:
                parts = [f"{k}:{v:.3f}" for k, v in reasoning.items()]
                self.label_reasoning.setText("Reasoning Heatmap: " + " | ".join(parts))
            else:
                self.label_reasoning.setText("Reasoning Heatmap: (warming up)")

            self.label_escalations.setText(
                f"AGVRA: {escalations['agvra']} | SCION: {escalations['scion']} | ASI-Arch: {escalations['asi_arch']}"
            )

            self.list_loaded.clear()
            for lib in auto_status["loaded"]:
                self.list_loaded.addItem(lib)
            self.list_failed.clear()
            for lib, reason in auto_status["failed"].items():
                self.list_failed.addItem(f"{lib}: {reason}")

            self.label_swarm.setText(
                f"Swarm - Nodes: {swarm['nodes']} | Consensus: {swarm['last_consensus']}"
            )

            self.label_reboot_paths.setText(
                f"Reboot Memory - SMB: {reboot['primary_smb']} | Local: {reboot['fallback_local']}"
            )
            self.label_reboot_status.setText(
                f"Status: {reboot['last_status']} | LastSave: {reboot['last_save_time']} | LastLoad: {reboot['last_load_time']} | IntegrityScore: {reboot['integrity_score']}"
            )

            self.text_dialogue.clear()
            if dialogue:
                self.text_dialogue.append("\n".join(dialogue))
            else:
                self.text_dialogue.append("(No internal dialogue yet)")

            if narratives:
                self.text_dialogue.append("\n--- Attack Narratives ---")
                for n in narratives:
                    self.text_dialogue.append(n["summary"])


# =========================
# Autonomous Daemon Loop
# =========================

def autonomous_loop(autoloader: BorgAutoloader,
                    putty: SillyPuttyDataGatherer,
                    organs: List[BaseOrgan],
                    self_integrity: SelfIntegrityOrgan,
                    queen: BorgQueen,
                    rt_queen: RealTimeQueen,
                    swarm_consensus: SwarmConsensusEngine,
                    chain_engine: AttackChainEngine,
                    brain: BrainCore,
                    reboot_mgr: RebootMemoryManager,
                    telemetry: TelemetryCollector,
                    narrative: AttackNarrativeReconstructor,
                    snapshot_path: str = "brain_snapshot.json",
                    interval: float = 0.5):
    last_snapshot = time.time()
    snapshot_interval = 10.0
    last_learned_save = time.time()
    learned_interval = 30.0

    while True:
        rand_bytes = os.urandom(64)
        putty.absorb(rand_bytes)

        metrics = telemetry.collect_psutil_metrics()

        telemetry.scan_rogue_hosts()
        for ip in list(telemetry.rogue_hosts):
            telemetry.auto_block_ip(ip)

        telemetry.feed_attack_chain(chain_engine)

        for o in organs:
            o.update()
            o.micro_recovery()
        self_integrity.update(organs)

        report = putty.report()
        queen.assimilate(telemetry.node_id, report)

        events_for_rt = [{"entity": f"{telemetry.node_id}_flow_{i}", "score": report["anomaly_density"] + i * 0.01} for i in range(3)]
        rt_queen.update(telemetry.node_id, events_for_rt)
        chains = chain_engine.detect()

        node_votes = {
            "risk": brain.risk_score,
            "anomaly_risk": brain.anomaly_risk,
            "collective_health": brain.collective_health,
        }
        swarm_consensus.submit_votes(telemetry.node_id, node_votes)
        consensus = swarm_consensus.compute_consensus()
        swarm_view = {
            "swarm_risk": consensus.get("risk", 0.0),
        }

        if chains:
            brain.set_environment("DANGER")
        else:
            brain.set_environment("CALM")

        for o in organs:
            o.preemptive_hardening(brain.risk_score)

        brain.tick(putty, organs, swarm_view, metrics)
        narrative.reconstruct()

        now = time.time()
        if now - last_snapshot >= snapshot_interval:
            snapshot = {
                "ts": now,
                "putty": report,
                "queen": queen.status(),
                "rt_queen_risk": rt_queen.global_risk(),
                "chains": chains,
                "situational": brain.situational_snapshot(),
                "predictive": brain.predictive_snapshot(),
                "codex": brain.codex_snapshot(),
                "agent": brain.agent_snapshot(),
                "cipher": brain.cipher_snapshot(),
                "autonomy": brain.autonomy_snapshot(),
                "escalations": brain.escalation_snapshot(),
                "organs": [o.snapshot() for o in organs],
                "self_integrity": self_integrity.snapshot(),
                "reasoning": brain.reasoning_snapshot(),
                "autoloader": autoloader.status(),
                "swarm": brain.swarm_snapshot(),
                "dialogue": brain.internal_dialogue_snapshot(),
                "telemetry_metrics": metrics,
                "rogue_hosts": list(telemetry.rogue_hosts),
                "blocked_ips": list(telemetry.blocked_ips),
                "narratives": narrative.snapshot(),
            }
            try:
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=2)
            except Exception:
                pass
            last_snapshot = now

        if now - last_learned_save >= learned_interval:
            state = {
                "ts": now,
                "brain_learning": brain.export_learning_state(),
            }
            reboot_mgr.save_state(state)
            last_learned_save = now

        time.sleep(interval)


# =========================
# Entry Point
# =========================

def main():
    autoloader = BorgAutoloader()
    autoloader.autoload()

    putty = SillyPuttyDataGatherer()

    organs: List[BaseOrgan] = [
        DeepRamOrgan("DeepRAM"),
        BackupEngineOrgan("BackupEngine"),
        NetworkWatcherOrgan("NetWatcher"),
        GPUCacheOrgan("GPUCache"),
        ThermalOrgan("Thermal"),
        DiskOrgan("Disk"),
        VRAMOrgan("VRAM"),
        AICoachOrgan("AICoach"),
        SwarmNodeOrgan("SwarmNode"),
        Back4BloodAnalyzer("Back4Blood"),
    ]
    self_integrity = SelfIntegrityOrgan("SelfIntegrity")

    queen = BorgQueen()
    rt_queen = RealTimeQueen()
    swarm_consensus = SwarmConsensusEngine()
    chain_engine = AttackChainEngine()

    reboot_mgr = RebootMemoryManager()

    node_id = "node-001"
    swarm_peers: List[str] = []
    ml_engine = MLAnomalyEngine(autoloader)
    brain = BrainCore(chain_engine, ml_engine, learning_rate=0.05, node_id=node_id, swarm_peers=swarm_peers)

    telemetry = TelemetryCollector(node_id=node_id)
    narrative = AttackNarrativeReconstructor(chain_engine)

    watchdog = AutoloaderWatchdog(autoloader, interval=60.0)
    watchdog.start()

    if GUI_AVAILABLE:
        app = QApplication(sys.argv)
        cockpit = BorgCockpit(
            autoloader=autoloader,
            putty=putty,
            organs=organs,
            self_integrity=self_integrity,
            queen=queen,
            rt_queen=rt_queen,
            swarm_consensus=swarm_consensus,
            chain_engine=chain_engine,
            brain=brain,
            reboot_mgr=reboot_mgr,
            narrative=narrative,
        )
        cockpit.show()

        loop_thread = threading.Thread(
            target=autonomous_loop,
            args=(autoloader, putty, organs, self_integrity, queen, rt_queen,
                  swarm_consensus, chain_engine, brain, reboot_mgr, telemetry, narrative),
            daemon=True,
        )
        loop_thread.start()

        sys.exit(app.exec())
    else:
        autonomous_loop(
            autoloader, putty, organs, self_integrity, queen, rt_queen,
            swarm_consensus, chain_engine, brain, reboot_mgr, telemetry, narrative
        )


if __name__ == "__main__":
    main()
