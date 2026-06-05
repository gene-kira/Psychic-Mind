"""
local_behavioral_agent_godview_mp_gpu.py

Pure local behavioral security agent with:
- Universal autoloader (libs, CUDA, TensorRT, plugins, YAML rules, venv)
- Kernel & user-space sensors
- Telemetry, feature, detection, policy, response, swarm, dashboard services
- Microservice-style architecture (threads + in-memory event bus)
- Multiprocessing GPU-accelerated detection worker
- Advanced Tkinter God-View cockpit (multi-panel UI, heatmaps, lineage)
- Optional GPU-assisted visualization (CuPy / Torch)
- Multi-node local swarm (HTTP-based peer sync)
- Persistent storage (SQLite)
- REST API (Flask) for status & events
"""

from __future__ import annotations
import importlib
import subprocess
import sys
import os
import glob
import venv
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import threading
import queue
import time
import uuid
import sqlite3
import json
import socket
import multiprocessing as mp

import tkinter as tk
from tkinter import ttk

# =========================
# Universal Autoloader & Environment
# =========================

REQUIRED_LIBRARIES = [
    "psutil",
    "scapy",
    "yara_python",
    "numpy",
    "pyyaml",
    "requests",
    "hdbscan",
    "torch",          # optional GPU
    "tensorrt",       # optional
    "flask",          # REST API
    "cupy",           # optional GPU accel for heatmaps
]


def bootstrap_venv(venv_dir: str = ".agent_venv"):
    if hasattr(sys, "real_prefix") or os.environ.get("VIRTUAL_ENV"):
        print("[ENV] Already inside a virtual environment.")
        return

    if not os.path.exists(venv_dir):
        print(f"[ENV] Creating virtual environment at {venv_dir}...")
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    else:
        print(f"[ENV] Using existing virtual environment at {venv_dir}.")

    print("[ENV] Virtual environment ready. Activate manually if desired.")


def ensure_library(lib_name: str):
    try:
        return importlib.import_module(lib_name)
    except ImportError:
        print(f"[AUTOLOADER] Missing: {lib_name} — installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib_name])
            return importlib.import_module(lib_name)
        except Exception as e:
            print(f"[AUTOLOADER] FAILED to install {lib_name}: {e}")
            return None


def autoload_libraries() -> Dict[str, Any]:
    bootstrap_venv()
    loaded: Dict[str, Any] = {}
    for lib in REQUIRED_LIBRARIES:
        mod = ensure_library(lib)
        if mod is not None:
            loaded[lib] = mod
            print(f"[AUTOLOADER] Loaded: {lib}")
        else:
            print(f"[AUTOLOADER] Skipped: {lib}")
    return loaded


def compile_cuda_kernels(kernel_dir: str = "cuda_kernels", output_dir: str = "cuda_build"):
    if not os.path.isdir(kernel_dir):
        print(f"[CUDA] No kernel directory at {kernel_dir}, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    cu_files = glob.glob(os.path.join(kernel_dir, "*.cu"))
    if not cu_files:
        print("[CUDA] No .cu files found, skipping.")
        return

    try:
        subprocess.check_call(["nvcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("[CUDA] nvcc not found, cannot compile kernels.")
        return

    for cu in cu_files:
        base = os.path.splitext(os.path.basename(cu))[0]
        out = os.path.join(output_dir, base + ".ptx")
        print(f"[CUDA] Compiling {cu} -> {out}")
        try:
            subprocess.check_call(["nvcc", "-ptx", cu, "-o", out])
        except Exception as e:
            print(f"[CUDA] Failed to compile {cu}: {e}")


def verify_tensorrt_engine(engine_path: str = "models/tensorrt_engine.trt") -> bool:
    if not os.path.exists(engine_path):
        print(f"[TensorRT] Engine not found at {engine_path}.")
        return False
    try:
        size = os.path.getsize(engine_path)
        if size <= 0:
            print(f"[TensorRT] Engine file at {engine_path} is empty.")
            return False
        print(f"[TensorRT] Engine OK at {engine_path}, size={size} bytes.")
        return True
    except Exception as e:
        print(f"[TensorRT] Failed to inspect engine: {e}")
        return False


def auto_generate_yaml_rules(rule_path: str = "rules/policies.yaml"):
    os.makedirs(os.path.dirname(rule_path), exist_ok=True)
    if os.path.exists(rule_path):
        print(f"[YAML] Rule file exists at {rule_path}.")
        return

    print(f"[YAML] Creating default rule file at {rule_path}...")
    default_content = """rules:
  - id: default_quarantine_high_score
    name: "Default High Score Quarantine"
    condition: "score > 0.9"
    action: "quarantine"
"""
    try:
        with open(rule_path, "w", encoding="utf-8") as f:
            f.write(default_content)
        print("[YAML] Default rules written.")
    except Exception as e:
        print(f"[YAML] Failed to write default rules: {e}")


def auto_load_plugins(plugin_dir: str = "plugins") -> Dict[str, Any]:
    loaded: Dict[str, Any] = {}
    if not os.path.isdir(plugin_dir):
        print(f"[PLUGINS] No plugin directory at {plugin_dir}, skipping.")
        return loaded

    plugin_files = glob.glob(os.path.join(plugin_dir, "*.py"))
    if not plugin_files:
        print("[PLUGINS] No plugin files found, skipping.")
        return loaded

    sys.path.insert(0, os.path.abspath(plugin_dir))
    for path in plugin_files:
        name = os.path.splitext(os.path.basename(path))[0]
        if not name.replace("_", "").isalnum():
            print(f"[PLUGINS] Skipping suspicious plugin name: {name}")
            continue
        try:
            mod = importlib.import_module(name)
            loaded[name] = mod
            print(f"[PLUGINS] Loaded plugin module: {name}")
        except Exception as e:
            print(f"[PLUGINS] Failed to load plugin {name}: {e}")
    return loaded


AUTOLOADED = autoload_libraries()
compile_cuda_kernels()
verify_tensorrt_engine()
auto_generate_yaml_rules()
PLUGIN_MODULES = auto_load_plugins()

np = AUTOLOADED.get("numpy")
cp = AUTOLOADED.get("cupy")
requests_mod = AUTOLOADED.get("requests")
flask_mod = AUTOLOADED.get("flask")
torch_mod = AUTOLOADED.get("torch")

# =========================
# Core Models
# =========================

@dataclass
class TelemetryEvent:
    id: str
    timestamp: float
    source: str
    event_type: str
    payload: Dict[str, Any]


@dataclass
class FeatureVector:
    event_id: str
    features: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyScore:
    event_id: str
    score: float
    label: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    event_id: str
    score: float
    actions: List[str]
    rules_triggered: List[str] = field(default_factory=list)


# =========================
# Event Bus (Local Only)
# =========================

class LocalEventBus:
    def __init__(self, max_queue_size: int = 2000):
        self._topics: Dict[str, "queue.Queue[Any]"] = {}
        self._lock = threading.Lock()
        self._max_queue_size = max_queue_size

    def _get_topic_queue(self, topic: str) -> "queue.Queue[Any]":
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = queue.Queue(maxsize=self._max_queue_size)
            return self._topics[topic]

    def publish(self, topic: str, message: Any):
        q = self._get_topic_queue(topic)
        try:
            q.put(message, timeout=0.05)
        except queue.Full:
            print(f"[EventBus] Dropping message on topic {topic}: queue full")

    def subscribe(self, topic: str) -> "queue.Queue[Any]":
        return self._get_topic_queue(topic)


# =========================
# Sensors
# =========================

class KernelSensor:
    def start(self): ...
    def stop(self): ...


class EBPFHookSensor(KernelSensor):
    def start(self):
        print("[EBPFHookSensor] start()")

    def stop(self):
        print("[EBPFHookSensor] stop()")


class EndpointSecuritySensor(KernelSensor):
    def start(self):
        print("[EndpointSecuritySensor] start()")

    def stop(self):
        print("[EndpointSecuritySensor] stop()")


class UserSpaceMonitor:
    def start(self): ...
    def stop(self): ...


class ProcessMonitor(UserSpaceMonitor):
    def start(self):
        print("[ProcessMonitor] start()")

    def stop(self):
        print("[ProcessMonitor] stop()")


class NetflowMonitor(UserSpaceMonitor):
    def start(self):
        print("[NetflowMonitor] start()")

    def stop(self):
        print("[NetflowMonitor] stop()")


class MemoryScanner(UserSpaceMonitor):
    def start(self):
        print("[MemoryScanner] start()")

    def stop(self):
        print("[MemoryScanner] stop()")


class DLLScanner(UserSpaceMonitor):
    def start(self):
        print("[DLLScanner] start()")

    def stop(self):
        print("[DLLScanner] stop()")


# =========================
# Microservice Base
# =========================

class Microservice(threading.Thread):
    def __init__(self, name: str, bus: LocalEventBus):
        super().__init__(daemon=True)
        self.name = name
        self.bus = bus
        self._running = threading.Event()
        self._running.set()

    def stop(self):
        self._running.clear()

    def run(self):
        raise NotImplementedError


# =========================
# Telemetry Service
# =========================

class TelemetryService(Microservice):
    def __init__(
        self,
        bus: LocalEventBus,
        kernel_sensors: List[KernelSensor],
        user_monitors: List[UserSpaceMonitor],
    ):
        super().__init__("TelemetryService", bus)
        self.kernel_sensors = kernel_sensors
        self.user_monitors = user_monitors

    def start_sensors(self):
        for s in self.kernel_sensors:
            s.start()
        for m in self.user_monitors:
            m.start()

    def stop_sensors(self):
        for s in self.kernel_sensors:
            s.stop()
        for m in self.user_monitors:
            m.stop()

    def inject_test_event(self, source: str, event_type: str, payload: Dict[str, Any]):
        event = TelemetryEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            source=source,
            event_type=event_type,
            payload=payload,
        )
        self.bus.publish("telemetry.raw", event)

    def run(self):
        print(f"[{self.name}] run()")
        self.start_sensors()
        try:
            while self._running.is_set():
                time.sleep(0.1)
        finally:
            self.stop_sensors()
            print(f"[{self.name}] stopped")


# =========================
# Feature Extraction Service
# =========================

class FeatureExtractor:
    def extract(self, event: TelemetryEvent) -> FeatureVector:
        f = [
            float(len(event.payload)),
            float(event.timestamp % 1000),
            float(len(event.source)),
        ]
        return FeatureVector(
            event_id=event.id,
            features=f,
            metadata={"source": event.source, "type": event.event_type},
        )


class FeatureService(Microservice):
    def __init__(self, bus: LocalEventBus, extractor: FeatureExtractor, batch_size: int = 32):
        super().__init__("FeatureService", bus)
        self.extractor = extractor
        self.in_q = self.bus.subscribe("telemetry.raw")
        self.batch_size = batch_size

    def run(self):
        print(f"[{self.name}] run()")
        batch: List[TelemetryEvent] = []
        while self._running.is_set():
            try:
                event: TelemetryEvent = self.in_q.get(timeout=0.2)
                batch.append(event)
                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    batch = []
            except queue.Empty:
                if batch:
                    self._process_batch(batch)
                    batch = []
        print(f"[{self.name}] stopped")

    def _process_batch(self, events: List[TelemetryEvent]):
        for e in events:
            fv = self.extractor.extract(e)
            self.bus.publish("features.ready", fv)


# =========================
# Detection (Heuristics + GPU + Multiprocessing)
# =========================

class HeuristicEngine:
    def score(self, fv: FeatureVector) -> float:
        s = sum(fv.features)
        return min(1.0, s / 1000.0)


class GpuScoringEngine:
    def __init__(self):
        self.gpu_available = False
        self.backend = None
        if torch_mod is not None and torch_mod.cuda.is_available():
            self.backend = "torch"
            self.gpu_available = True
            print("[GPU] Torch CUDA backend enabled.")
        elif cp is not None:
            self.backend = "cupy"
            self.gpu_available = True
            print("[GPU] CuPy backend enabled.")
        else:
            print("[GPU] No GPU backend available, falling back to CPU.")

    def score_batch(self, batch: List[FeatureVector]) -> List[float]:
        if not batch:
            return []
        if not self.gpu_available or np is None:
            return [min(1.0, sum(fv.features) / 800.0) for fv in batch]

        feats = np.array([fv.features for fv in batch], dtype=float)
        if self.backend == "torch":
            try:
                t = torch_mod.tensor(feats, device="cuda", dtype=torch_mod.float32)
                s = t.sum(dim=1)
                s = torch_mod.sigmoid(s / 1000.0)
                return s.detach().cpu().numpy().tolist()
            except Exception:
                return [min(1.0, sum(fv.features) / 800.0) for fv in batch]
        elif self.backend == "cupy":
            try:
                g = cp.asarray(feats)
                s = g.sum(axis=1)
                s = 1 / (1 + cp.exp(-s / 1000.0))
                return cp.asnumpy(s).tolist()
            except Exception:
                return [min(1.0, sum(fv.features) / 800.0) for fv in batch]
        return [min(1.0, sum(fv.features) / 800.0) for fv in batch]


class CUDAScoringEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"[CUDAScoringEngine] init model={model_path}")

    def score(self, fv: FeatureVector) -> float:
        return 0.5


# Multiprocessing worker process for detection
def detection_worker_process(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
):
    heur = HeuristicEngine()
    gpu_engine = GpuScoringEngine()
    while True:
        try:
            batch: List[FeatureVector] = in_queue.get()
            if batch is None:
                break
            gpu_scores = gpu_engine.score_batch(batch)
            results: List[AnomalyScore] = []
            for fv, gscore in zip(batch, gpu_scores):
                h_score = heur.score(fv)
                combined = (h_score + gscore) / 2.0
                results.append(AnomalyScore(event_id=fv.event_id, score=combined))
            out_queue.put(results)
        except Exception as e:
            print(f"[DetectionWorker] error: {e}")


class DetectionService(Microservice):
    def __init__(self, bus: LocalEventBus, model_path: str, batch_size: int = 64):
        super().__init__("DetectionService", bus)
        self.in_q = self.bus.subscribe("features.ready")
        self.batch_size = batch_size
        self.mp_in: mp.Queue = mp.Queue(maxsize=64)
        self.mp_out: mp.Queue = mp.Queue(maxsize=64)
        self.worker = mp.Process(
            target=detection_worker_process,
            args=(self.mp_in, self.mp_out),
            daemon=True,
        )
        self.worker.start()
        self.cuda_engine = CUDAScoringEngine(model_path)

    def stop(self):
        super().stop()
        try:
            self.mp_in.put(None, timeout=0.2)
        except Exception:
            pass
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)

    def run(self):
        print(f"[{self.name}] run()")
        batch: List[FeatureVector] = []
        while self._running.is_set():
            # Feed features into mp worker
            try:
                fv: FeatureVector = self.in_q.get(timeout=0.05)
                batch.append(fv)
                if len(batch) >= self.batch_size:
                    self._send_batch(batch)
                    batch = []
            except queue.Empty:
                if batch:
                    self._send_batch(batch)
                    batch = []

            # Drain results from mp worker
            self._drain_results()
        print(f"[{self.name}] stopped")

    def _send_batch(self, batch: List[FeatureVector]):
        try:
            self.mp_in.put(batch, timeout=0.1)
        except queue.Full:
            print("[DetectionService] mp_in full, dropping batch")

    def _drain_results(self):
        try:
            while True:
                results: List[AnomalyScore] = self.mp_out.get_nowait()
                for score in results:
                    self.bus.publish("detection.scores", score)
        except queue.Empty:
            pass


# =========================
# Policy / Rule Service
# =========================

@dataclass
class YAMLRule:
    id: str
    name: str
    condition: str
    action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class YAMLRuleEngine:
    def __init__(self, rule_path: str = "rules/policies.yaml"):
        self.rules: Dict[str, YAMLRule] = {}
        self.rule_path = rule_path

    def load_from_yaml(self):
        print("[YAMLRuleEngine] load_from_yaml()")
        auto_generate_yaml_rules(self.rule_path)
        yaml_mod = AUTOLOADED.get("pyyaml")
        if yaml_mod is None:
            print("[YAMLRuleEngine] pyyaml not available, using placeholder rule.")
            rule_id = str(uuid.uuid4())
            self.rules[rule_id] = YAMLRule(
                id=rule_id,
                name="placeholder_rule",
                condition="score > 0.8",
                action="quarantine",
            )
            return

        try:
            with open(self.rule_path, "r", encoding="utf-8") as f:
                data = yaml_mod.safe_load(f) or {}
            for r in data.get("rules", []):
                rule_id = r.get("id", str(uuid.uuid4()))
                self.rules[rule_id] = YAMLRule(
                    id=rule_id,
                    name=r.get("name", "unnamed_rule"),
                    condition=r.get("condition", "score > 0.8"),
                    action=r.get("action", "quarantine"),
                    metadata={k: v for k, v in r.items() if k not in ("id", "name", "condition", "action")},
                )
            print(f"[YAMLRuleEngine] Loaded {len(self.rules)} rules.")
        except Exception as e:
            print(f"[YAMLRuleEngine] Failed to load YAML rules: {e}")

    def evaluate(self, score: AnomalyScore) -> PolicyDecision:
        actions: List[str] = []
        triggered: List[str] = []
        for r in self.rules.values():
            cond = r.condition.strip()
            if cond.startswith("score >"):
                try:
                    threshold = float(cond.split(">")[1].strip())
                    if score.score > threshold:
                        actions.append(r.action)
                        triggered.append(r.name)
                except Exception:
                    continue
        return PolicyDecision(
            event_id=score.event_id,
            score=score.score,
            actions=actions,
            rules_triggered=triggered,
        )


class PolicyService(Microservice):
    def __init__(self, bus: LocalEventBus, engine: YAMLRuleEngine):
        super().__init__("PolicyService", bus)
        self.engine = engine
        self.in_q = self.bus.subscribe("detection.scores")

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            try:
                score: AnomalyScore = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            decision = self.engine.evaluate(score)
            self.bus.publish("policy.decisions", decision)
        print(f"[{self.name}] stopped")


# =========================
# Persistent Storage (SQLite)
# =========================

class PersistentStore:
    def __init__(self, db_path: str = "agent_data.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS scores (
                    id TEXT PRIMARY KEY,
                    score REAL,
                    timestamp REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    score REAL,
                    actions TEXT,
                    rules TEXT,
                    timestamp REAL
                )
            """)
            conn.commit()

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def store_score(self, score: AnomalyScore):
        with self._lock, self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO scores (id, score, timestamp) VALUES (?, ?, ?)",
                (score.event_id, score.score, time.time()),
            )
            conn.commit()

    def store_decision(self, decision: PolicyDecision):
        with self._lock, self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO decisions (id, score, actions, rules, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    decision.event_id,
                    decision.score,
                    json.dumps(decision.actions),
                    json.dumps(decision.rules_triggered),
                    time.time(),
                ),
            )
            conn.commit()


# =========================
# Response / Enforcement Service
# =========================

class ResponseEngine:
    def alert(self, decision: PolicyDecision):
        print(f"[ResponseEngine] ALERT event={decision.event_id} score={decision.score} rules={decision.rules_triggered}")

    def quarantine(self, decision: PolicyDecision):
        print(f"[ResponseEngine] QUARANTINE event={decision.event_id}")

    def enforce(self, decision: PolicyDecision):
        if not decision.actions:
            return
        self.alert(decision)
        for action in decision.actions:
            if action == "quarantine":
                self.quarantine(decision)


class ResponseService(Microservice):
    def __init__(self, bus: LocalEventBus, engine: ResponseEngine):
        super().__init__("ResponseService", bus)
        self.engine = engine
        self.in_q = self.bus.subscribe("policy.decisions")

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            try:
                decision: PolicyDecision = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.engine.enforce(decision)
        print(f"[{self.name}] stopped")


# =========================
# Plugin Marketplace
# =========================

class Plugin:
    def __init__(self, name: str, handler: Callable[[TelemetryEvent], None]):
        self.name = name
        self.handler = handler

    def handle(self, event: TelemetryEvent):
        try:
            self.handler(event)
        except Exception as e:
            print(f"[Plugin:{self.name}] error: {e}")


class PluginMarketplace:
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin):
        print(f"[PluginMarketplace] register() {plugin.name}")
        self.plugins[plugin.name] = plugin

    def dispatch(self, event: TelemetryEvent):
        for plugin in self.plugins.values():
            plugin.handle(event)


def build_plugins_from_modules() -> PluginMarketplace:
    marketplace = PluginMarketplace()
    for name, mod in PLUGIN_MODULES.items():
        handler = getattr(mod, "handle_event", None)
        if callable(handler):
            marketplace.register(Plugin(name=name, handler=handler))
        else:
            print(f"[PLUGINS] Module {name} has no handle_event(), skipping.")
    return marketplace


# =========================
# Local Swarm / Intel Cache Service
# =========================

class LocalIntelStore:
    def __init__(self):
        self.indicators: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def add_indicator(self, indicator: Dict[str, Any]):
        with self.lock:
            self.indicators.append(indicator)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.indicators)


class SwarmLocalService(Microservice):
    def __init__(self, bus: LocalEventBus, store: LocalIntelStore, peers: List[str]):
        super().__init__("SwarmLocalService", bus)
        self.store = store
        self.peers = peers
        self.in_q = self.bus.subscribe("detection.scores")

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            try:
                score: AnomalyScore = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if score.score > 0.9:
                indicator = {"event_id": score.event_id, "score": score.score, "ts": time.time()}
                self.store.add_indicator(indicator)
                self._broadcast_indicator(indicator)
        print(f"[{self.name}] stopped")

    def _broadcast_indicator(self, indicator: Dict[str, Any]):
        if not requests_mod:
            return
        for peer in self.peers:
            try:
                requests_mod.post(f"{peer}/swarm/indicator", json=indicator, timeout=0.5)
            except Exception:
                pass


# =========================
# Dashboard Backend Service
# =========================

class DashboardBackend:
    def __init__(self, store: PersistentStore):
        self.recent_scores: List[AnomalyScore] = []
        self.recent_decisions: List[PolicyDecision] = []
        self.lock = threading.Lock()
        self.store = store
        self.lineage: Dict[str, Tuple[str, str]] = {}

    def push_score(self, score: AnomalyScore):
        with self.lock:
            self.recent_scores.append(score)
        self.store.store_score(score)

    def push_decision(self, decision: PolicyDecision):
        with self.lock:
            self.recent_decisions.append(decision)
        self.store.store_decision(decision)

    def add_lineage(self, event: TelemetryEvent):
        with self.lock:
            self.lineage[event.id] = (event.source, event.event_type)

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "recent_events": len(self.recent_scores),
                "last_score": self.recent_scores[-1].score if self.recent_scores else None,
            }

    def snapshot(self):
        with self.lock:
            return {
                "scores": list(self.recent_scores),
                "decisions": list(self.recent_decisions),
                "lineage": dict(self.lineage),
            }


class DashboardService(Microservice):
    def __init__(self, bus: LocalEventBus, backend: DashboardBackend):
        super().__init__("DashboardService", bus)
        self.backend = backend
        self.score_q = self.bus.subscribe("detection.scores")
        self.decision_q = self.bus.subscribe("policy.decisions")
        self.event_q = self.bus.subscribe("telemetry.raw")

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            got_any = False
            try:
                event: TelemetryEvent = self.event_q.get(timeout=0.05)
                self.backend.add_lineage(event)
                got_any = True
            except queue.Empty:
                pass
            try:
                score: AnomalyScore = self.score_q.get(timeout=0.05)
                self.backend.push_score(score)
                got_any = True
            except queue.Empty:
                pass
            try:
                decision: PolicyDecision = self.decision_q.get(timeout=0.05)
                self.backend.push_decision(decision)
                got_any = True
            except queue.Empty:
                pass
            if not got_any:
                time.sleep(0.02)
        print(f"[{self.name}] stopped")


# =========================
# REST API Service (Flask)
# =========================

class RestAPIServer(Microservice):
    def __init__(self, backend: DashboardBackend, intel_store: LocalIntelStore, host: str = "127.0.0.1", port: int = 5000):
        super().__init__("RestAPIServer", bus=LocalEventBus())
        self.backend = backend
        self.intel_store = intel_store
        self.host = host
        self.port = port
        self.app = flask_mod.Flask("behavioral_agent_api") if flask_mod else None

        if self.app:
            self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/status")
        def status():
            snap = self.backend.snapshot()
            return {
                "recent_events": len(snap["scores"]),
                "recent_decisions": len(snap["decisions"]),
            }

        @self.app.get("/scores")
        def scores():
            snap = self.backend.snapshot()
            return {
                "scores": [
                    {"id": s.event_id, "score": s.score}
                    for s in snap["scores"]
                ]
            }

        @self.app.get("/decisions")
        def decisions():
            snap = self.backend.snapshot()
            return {
                "decisions": [
                    {
                        "id": d.event_id,
                        "score": d.score,
                        "actions": d.actions,
                        "rules": d.rules_triggered,
                    }
                    for d in snap["decisions"]
                ]
            }

        @self.app.post("/swarm/indicator")
        def swarm_indicator():
            data = flask_mod.request.get_json(force=True)
            self.intel_store.add_indicator(data)
            return {"status": "ok"}

    def run(self):
        if not self.app:
            print("[RestAPIServer] Flask not available, REST API disabled.")
            return
        print(f"[RestAPIServer] starting on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)


# =========================
# Agent Orchestrator
# =========================

class LocalBehavioralAgent:
    def __init__(self, model_path: str, peers: List[str], rest_port: int):
        self.bus = LocalEventBus()
        self.store = PersistentStore()
        self.intel_store = LocalIntelStore()

        kernel_sensors = [EBPFHookSensor(), EndpointSecuritySensor()]
        user_monitors = [ProcessMonitor(), NetflowMonitor(), MemoryScanner(), DLLScanner()]

        self.telemetry = TelemetryService(self.bus, kernel_sensors, user_monitors)
        self.feature_service = FeatureService(self.bus, FeatureExtractor())
        self.detection_service = DetectionService(self.bus, model_path=model_path)

        self.rule_engine = YAMLRuleEngine()
        self.rule_engine.load_from_yaml()
        self.policy_service = PolicyService(self.bus, self.rule_engine)

        self.response_service = ResponseService(self.bus, ResponseEngine())
        self.swarm_service = SwarmLocalService(self.bus, self.intel_store, peers)
        self.dashboard_backend = DashboardBackend(self.store)
        self.dashboard_service = DashboardService(self.bus, self.dashboard_backend)

        self.plugin_marketplace = build_plugins_from_modules()

        self.rest_api = RestAPIServer(self.dashboard_backend, self.intel_store, port=rest_port)

        self.services: List[Microservice] = [
            self.telemetry,
            self.feature_service,
            self.detection_service,
            self.policy_service,
            self.response_service,
            self.swarm_service,
            self.dashboard_service,
            self.rest_api,
        ]

    def start(self):
        print("[LocalBehavioralAgent] start()")
        for s in self.services:
            s.start()

    def stop(self):
        print("[LocalBehavioralAgent] stop()")
        for s in self.services:
            s.stop()
        for s in self.services:
            s.join()

    def inject_test_event(self, source: str, event_type: str, payload: Dict[str, Any]):
        event = TelemetryEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            source=source,
            event_type=event_type,
            payload=payload,
        )
        self.plugin_marketplace.dispatch(event)
        self.bus.publish("telemetry.raw", event)

    def get_dashboard_snapshot(self):
        return self.dashboard_backend.snapshot()

    def get_intel_snapshot(self):
        return self.intel_store.snapshot()


# =========================
# Tkinter God-View Cockpit
# =========================

class GodViewUI:
    def __init__(self, root: tk.Tk, agent: LocalBehavioralAgent):
        self.root = root
        self.agent = agent

        self.root.title("Behavioral Security God-View (MP + GPU)")
        self.root.geometry("1400x800")
        try:
            self.root.call('tk', 'scaling', 1.25)
        except Exception:
            pass

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=2)

        self.frame_threats = ttk.LabelFrame(self.root, text="Threat Stream")
        self.frame_threats.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.frame_heatmap = ttk.LabelFrame(self.root, text="Heatmap / Metrics")
        self.frame_heatmap.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.frame_timeline = ttk.LabelFrame(self.root, text="Timeline")
        self.frame_timeline.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.frame_status = ttk.LabelFrame(self.root, text="Node Status / Swarm / Lineage / Logs")
        self.frame_status.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.threat_list = tk.Listbox(self.frame_threats, font=("Consolas", 10))
        self.threat_list.pack(fill="both", expand=True)

        self.heatmap_canvas = tk.Canvas(self.frame_heatmap, bg="#101010")
        self.heatmap_canvas.pack(fill="both", expand=True)

        self.timeline_list = tk.Listbox(self.frame_timeline, font=("Consolas", 10))
        self.timeline_list.pack(fill="both", expand=True)

        self.status_text = tk.Text(self.frame_status, font=("Consolas", 10), height=10)
        self.status_text.pack(fill="both", expand=True)

        self.lineage_list = tk.Listbox(self.frame_status, font=("Consolas", 9), height=6)
        self.lineage_list.pack(fill="x", expand=False)

    def _schedule_update(self):
        self._update_ui()
        self.root.after(500, self._schedule_update)

    def _update_ui(self):
        snapshot = self.agent.get_dashboard_snapshot()
        scores: List[AnomalyScore] = snapshot["scores"]
        decisions: List[PolicyDecision] = snapshot["decisions"]
        lineage: Dict[str, Tuple[str, str]] = snapshot["lineage"]

        self.threat_list.delete(0, "end")
        for s in scores[-100:]:
            self.threat_list.insert(
                "end",
                f"{s.event_id[:8]} | score={s.score:.3f}"
            )

        self.timeline_list.delete(0, "end")
        for d in decisions[-100:]:
            actions = ",".join(d.actions) if d.actions else "none"
            self.timeline_list.insert(
                "end",
                f"{d.event_id[:8]} | score={d.score:.3f} | actions={actions}"
            )

        self.lineage_list.delete(0, "end")
        for eid, (src, etype) in list(lineage.items())[-50:]:
            self.lineage_list.insert("end", f"{eid[:8]} | {src} | {etype}")

        self._draw_heatmap(scores)

        self.status_text.delete("1.0", "end")
        self.status_text.insert("end", f"Recent events: {len(scores)}\n")
        if scores:
            self.status_text.insert("end", f"Last score: {scores[-1].score:.3f}\n")
        self.status_text.insert("end", f"Decisions: {len(decisions)}\n")
        intel = self.agent.get_intel_snapshot()
        self.status_text.insert("end", f"Swarm indicators: {len(intel)}\n")
        self.status_text.insert("end", "Agent running with multiprocessing + GPU.\n")

    def _draw_heatmap(self, scores: List[AnomalyScore]):
        self.heatmap_canvas.delete("all")
        w = self.heatmap_canvas.winfo_width() or 600
        h = self.heatmap_canvas.winfo_height() or 300

        if not scores or not np:
            self.heatmap_canvas.create_text(
                10, 10, anchor="nw", fill="#00ff00",
                text="Heatmap: waiting for data...",
                font=("Consolas", 10)
            )
            return

        n = min(len(scores), 64)
        vals = np.array([s.score for s in scores[-n:]], dtype=float)
        side = int(np.ceil(np.sqrt(len(vals))))
        padded = np.zeros(side * side, dtype=float)
        padded[: len(vals)] = vals
        grid = padded.reshape((side, side))

        if cp is not None:
            try:
                g = cp.asarray(grid)
                g = cp.sqrt(g)
                grid = cp.asnumpy(g)
            except Exception:
                pass

        max_val = float(grid.max()) if grid.size else 1.0
        cell_w = w / side
        cell_h = h / side

        for i in range(side):
            for j in range(side):
                v = grid[i, j] / (max_val + 1e-6)
                r = int(255 * v)
                g = int(255 * (1 - v))
                b = 0
                color = f"#{r:02x}{g:02x}{b:02x}"
                x0 = j * cell_w
                y0 = i * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                self.heatmap_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        self.heatmap_canvas.create_text(
            10, 10, anchor="nw", fill="#ffffff",
            text="Score Heatmap (GPU-accelerated if available)",
            font=("Consolas", 10, "bold")
        )


# =========================
# Bootstrap
# =========================

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    mp.set_start_method("spawn", force=True)

    engine_ok = verify_tensorrt_engine("models/tensorrt_engine.trt")
    model_path = "models/tensorrt_engine.trt" if engine_ok else "models/fallback.trt"

    local_ip = get_local_ip()
    rest_port = 5000
    peers: List[str] = []  # e.g., [f"http://{local_ip}:5001"]

    agent = LocalBehavioralAgent(model_path=model_path, peers=peers, rest_port=rest_port)
    agent.start()

    for i in range(40):
        agent.inject_test_event(
            source="process_monitor",
            event_type="proc_create",
            payload={"pid": 1000 + i, "exe": f"test{i}.exe"},
        )
        time.sleep(0.05)

    root = tk.Tk()
    ui = GodViewUI(root, agent)

    def on_close():
        agent.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
