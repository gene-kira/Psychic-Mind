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
import math

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
    "torch",          # optional GPU / training
    "tensorrt",       # optional
    "flask",          # REST API
    "cupy",           # optional GPU accel for heatmaps
    "onnxruntime",    # optional ONNX inference
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
  - id: resurrection_alert
    name: "Resurrection Alert"
    condition: "resurrected == True and score > 0.5"
    action: "alert"
  - id: autopilot_block_high
    name: "Autopilot Block High"
    condition: "score > 0.85 and autopilot_mode is not None"
    action: "quarantine"
"""
    try:
        with open(rule_path, "w", encoding="utf-8") as f:
            f.write(default_content)
        print("[YAML] Default rules written.")
    except Exception as e:
        print(f"[YAML] Failed to write default rules: {e}")


def auto_generate_rego_policies(path: str = "rules/policies.rego"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"[Rego] Policy file exists at {path}.")
        return
    print(f"[Rego] Creating default mini‑Rego policy at {path}...")
    content = """package agent.policy

default allow = true

deny["high_score_block"] {
  input.score > 0.95
}

deny["autopilot_extreme"] {
  input.autopilot_mode != null
  input.score > 0.9
}
"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[Rego] Default mini‑Rego policies written.")
    except Exception as e:
        print(f"[Rego] Failed to write default policies: {e}")


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
auto_generate_rego_policies()
PLUGIN_MODULES = auto_load_plugins()

np = AUTOLOADED.get("numpy")
cp = AUTOLOADED.get("cupy")
requests_mod = AUTOLOADED.get("requests")
flask_mod = AUTOLOADED.get("flask")
torch_mod = AUTOLOADED.get("torch")
onnxruntime_mod = AUTOLOADED.get("onnxruntime")
tensorrt_mod = AUTOLOADED.get("tensorrt")
psutil_mod = AUTOLOADED.get("psutil")
yara_mod = AUTOLOADED.get("yara_python")

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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutopilotCommandAssessment:
    command_id: str
    mode: Optional[str]
    risk_score: float
    allowed: bool
    reasons: List[str] = field(default_factory=list)
    raw_decision: Optional[PolicyDecision] = None


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
        print("[EBPFHookSensor] start() (placeholder for real eBPF hooks)")

    def stop(self):
        print("[EBPFHookSensor] stop()")


class EndpointSecuritySensor(KernelSensor):
    def start(self):
        print("[EndpointSecuritySensor] start() (placeholder for real endpoint hooks)")

    def stop(self):
        print("[EndpointSecuritySensor] stop()")


class UserSpaceMonitor:
    def start(self): ...
    def stop(self): ...


class ProcessMonitor(UserSpaceMonitor):
    def __init__(self, bus: LocalEventBus, interval: float = 1.0):
        self.bus = bus
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._seen_pids: Dict[int, str] = {}

    def _loop(self):
        print("[ProcessMonitor] loop()")
        while self._running:
            try:
                if not psutil_mod:
                    time.sleep(self.interval)
                    continue
                for p in psutil_mod.process_iter(["pid", "name", "exe", "username"]):
                    pid = p.info.get("pid")
                    exe = p.info.get("exe") or p.info.get("name") or "unknown"
                    if pid not in self._seen_pids:
                        self._seen_pids[pid] = exe
                        event = TelemetryEvent(
                            id=str(uuid.uuid4()),
                            timestamp=time.time(),
                            source="process_monitor",
                            event_type="proc_create",
                            payload={
                                "pid": pid,
                                "exe": exe,
                                "user": p.info.get("username"),
                            },
                        )
                        self.bus.publish("telemetry.raw", event)
                time.sleep(self.interval)
            except Exception as e:
                print(f"[ProcessMonitor] error: {e}")
                time.sleep(self.interval)

    def start(self):
        print("[ProcessMonitor] start()")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        print("[ProcessMonitor] stop()")
        self._running = False


class NetflowMonitor(UserSpaceMonitor):
    def __init__(self, bus: LocalEventBus, interval: float = 2.0):
        self.bus = bus
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _loop(self):
        print("[NetflowMonitor] loop()")
        while self._running:
            try:
                if not psutil_mod:
                    time.sleep(self.interval)
                    continue
                conns = psutil_mod.net_connections(kind="inet")
                for c in conns[:200]:
                    laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "?"
                    raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "?"
                    event = TelemetryEvent(
                        id=str(uuid.uuid4()),
                        timestamp=time.time(),
                        source="netflow_monitor",
                        event_type="net_conn",
                        payload={
                            "pid": c.pid,
                            "laddr": laddr,
                            "raddr": raddr,
                            "status": c.status,
                        },
                    )
                    self.bus.publish("telemetry.raw", event)
                time.sleep(self.interval)
            except Exception as e:
                print(f"[NetflowMonitor] error: {e}")
                time.sleep(self.interval)

    def start(self):
        print("[NetflowMonitor] start()")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        print("[NetflowMonitor] stop()")
        self._running = False


class MemoryScanner(UserSpaceMonitor):
    def __init__(self, bus: LocalEventBus, interval: float = 10.0, rule_file: Optional[str] = None):
        self.bus = bus
        self.interval = interval
        self.rule_file = rule_file
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._rules = None

    def _load_rules(self):
        if not yara_mod or not self.rule_file or not os.path.exists(self.rule_file):
            return
        try:
            self._rules = yara_mod.compile(self.rule_file)
            print("[MemoryScanner] YARA rules loaded.")
        except Exception as e:
            print(f"[MemoryScanner] failed to load YARA rules: {e}")

    def _loop(self):
        print("[MemoryScanner] loop()")
        self._load_rules()
        while self._running:
            try:
                if not self._rules:
                    time.sleep(self.interval)
                    continue
                for root, dirs, files in os.walk("."):
                    for fname in files:
                        path = os.path.join(root, fname)
                        try:
                            with open(path, "rb") as f:
                                data = f.read(1024 * 1024)
                            matches = self._rules.match(data=data)
                            if matches:
                                event = TelemetryEvent(
                                    id=str(uuid.uuid4()),
                                    timestamp=time.time(),
                                    source="memory_scanner",
                                    event_type="yara_match",
                                    payload={
                                        "file": path,
                                        "matches": [m.rule for m in matches],
                                    },
                                )
                                self.bus.publish("telemetry.raw", event)
                        except Exception:
                            continue
                time.sleep(self.interval)
            except Exception as e:
                print(f"[MemoryScanner] error: {e}")
                time.sleep(self.interval)

    def start(self):
        print("[MemoryScanner] start()")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        print("[MemoryScanner] stop()")
        self._running = False


class DLLScanner(UserSpaceMonitor):
    def __init__(self, bus: LocalEventBus, interval: float = 5.0):
        self.bus = bus
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _loop(self):
        print("[DLLScanner] loop()")
        while self._running:
            try:
                if not psutil_mod:
                    time.sleep(self.interval)
                    continue
                for p in psutil_mod.process_iter(["pid", "name"]):
                    try:
                        maps = p.memory_maps()
                        suspicious = [m.path for m in maps if "temp" in m.path.lower() or "appdata" in m.path.lower()]
                        if suspicious:
                            event = TelemetryEvent(
                                id=str(uuid.uuid4()),
                                timestamp=time.time(),
                                source="dll_scanner",
                                event_type="dll_loaded",
                                payload={
                                    "pid": p.pid,
                                    "exe": p.info.get("name"),
                                    "modules": suspicious[:20],
                                },
                            )
                            self.bus.publish("telemetry.raw", event)
                    except Exception:
                        continue
                time.sleep(self.interval)
            except Exception as e:
                print(f"[DLLScanner] error: {e}")
                time.sleep(self.interval)

    def start(self):
        print("[DLLScanner] start()")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        print("[DLLScanner] stop()")
        self._running = False


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
        base = [
            float(len(event.payload)),
            float(event.timestamp % 1000),
            float(len(event.source)),
        ]
        extra = []
        if event.event_type == "proc_create":
            exe = event.payload.get("exe", "")
            extra.append(float(len(exe)))
        elif event.event_type == "net_conn":
            status = event.payload.get("status", "")
            extra.append(float(len(status)))
        elif event.event_type == "yara_match":
            extra.append(float(len(event.payload.get("matches", []))))
        elif event.event_type == "dll_loaded":
            extra.append(float(len(event.payload.get("modules", []))))
        elif event.event_type == "autopilot_cmd":
            mode = event.payload.get("mode", "")
            extra.append(float(len(mode)))
        features = base + extra
        return FeatureVector(
            event_id=event.id,
            features=features,
            metadata={
                "source": event.source,
                "type": event.event_type,
                **event.payload,
            },
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
# Detection (Heuristics + GPU + Real Model Hooks + Multiprocessing)
# =========================

class HeuristicEngine:
    def score(self, fv: FeatureVector) -> float:
        s = sum(fv.features)
        return min(1.0, s / 1000.0)


class RealModelEngine:
    def __init__(self, model_dir: str = "models"):
        self.backend = None
        self.model = None
        self.session = None
        self.trt_engine = None
        self.model_dir = model_dir
        self._init_backends()

    def _init_backends(self):
        if torch_mod is not None:
            pt_path = os.path.join(self.model_dir, "model.pt")
            if os.path.exists(pt_path):
                try:
                    self.model = torch_mod.jit.load(
                        pt_path,
                        map_location="cuda" if torch_mod.cuda.is_available() else "cpu",
                    )
                    self.model.eval()
                    self.backend = "torch"
                    print("[RealModelEngine] Torch model loaded.")
                    return
                except Exception as e:
                    print(f"[RealModelEngine] Torch load failed: {e}")

        if onnxruntime_mod is not None:
            onnx_path = os.path.join(self.model_dir, "model.onnx")
            if os.path.exists(onnx_path):
                try:
                    self.session = onnxruntime_mod.InferenceSession(
                        onnx_path,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    self.backend = "onnx"
                    print("[RealModelEngine] ONNX model loaded.")
                    return
                except Exception as e:
                    print(f"[RealModelEngine] ONNX load failed: {e}")

        if tensorrt_mod is not None:
            trt_path = os.path.join(self.model_dir, "tensorrt_engine.trt")
            if os.path.exists(trt_path):
                print("[RealModelEngine] TensorRT engine detected (placeholder integration).")
                self.backend = "tensorrt"
                self.trt_engine = trt_path
                return

        print("[RealModelEngine] No real model backend available, using heuristic-only mode.")

    def score_batch(self, batch: List[FeatureVector]) -> List[float]:
        if not batch:
            return []
        if self.backend is None or np is None:
            return [0.0 for _ in batch]

        feats = np.array([fv.features for fv in batch], dtype=float)

        if self.backend == "torch" and self.model is not None:
            try:
                device = "cuda" if torch_mod.cuda.is_available() else "cpu"
                t = torch_mod.tensor(feats, device=device, dtype=torch_mod.float32)
                with torch_mod.no_grad():
                    out = self.model(t)
                if out.ndim > 1:
                    out = out.squeeze(-1)
                scores = torch_mod.sigmoid(out).detach().cpu().numpy().tolist()
                return [float(min(1.0, max(0.0, s))) for s in scores]
            except Exception as e:
                print(f"[RealModelEngine] Torch inference error: {e}")
                return [0.0 for _ in batch]

        if self.backend == "onnx" and self.session is not None:
            try:
                inp_name = self.session.get_inputs()[0].name
                out_name = self.session.get_outputs()[0].name
                out = self.session.run([out_name], {inp_name: feats.astype("float32")})[0]
                out = out.reshape(-1)
                scores = 1.0 / (1.0 + np.exp(-out))
                return [float(min(1.0, max(0.0, s))) for s in scores]
            except Exception as e:
                print(f"[RealModelEngine] ONNX inference error: {e}")
                return [0.0 for _ in batch]

        if self.backend == "tensorrt":
            return [0.5 for _ in batch]

        return [0.0 for _ in batch]


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


def detection_worker_process(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
):
    heur = HeuristicEngine()
    gpu_engine = GpuScoringEngine()
    real_model = RealModelEngine()
    while True:
        try:
            batch: List[FeatureVector] = in_queue.get()
            if batch is None:
                break
            gpu_scores = gpu_engine.score_batch(batch)
            model_scores = real_model.score_batch(batch)
            results: List[AnomalyScore] = []
            for idx, fv in enumerate(batch):
                gscore = gpu_scores[idx] if idx < len(gpu_scores) else 0.0
                mscore = model_scores[idx] if idx < len(model_scores) else 0.0
                h_score = heur.score(fv)
                combined = (h_score + gscore + mscore) / 3.0
                results.append(
                    AnomalyScore(
                        event_id=fv.event_id,
                        score=float(min(1.0, max(0.0, combined))),
                        details=fv.metadata,
                    )
                )
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
            c.execute("""
                CREATE TABLE IF NOT EXISTS quarantined (
                    exe TEXT,
                    pid INTEGER,
                    timestamp REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id TEXT PRIMARY KEY,
                    features TEXT,
                    label REAL,
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

    def store_quarantine(self, exe: str, pid: int):
        with self._lock, self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO quarantined (exe, pid, timestamp) VALUES (?, ?, ?)",
                (exe, pid, time.time()),
            )
            conn.commit()

    def get_quarantined_executables(self) -> List[str]:
        with self._lock, self._conn() as conn:
            c = conn.cursor()
            c.execute("SELECT DISTINCT exe FROM quarantined")
            rows = c.fetchall()
            return [r[0] for r in rows]

    def store_feature_for_training(self, fv: FeatureVector, label: float):
        with self._lock, self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO features (id, features, label, timestamp) VALUES (?, ?, ?, ?)",
                (fv.event_id, json.dumps(fv.features), float(label), time.time()),
            )
            conn.commit()

    def load_training_data(self, limit: int = 5000) -> Tuple[List[List[float]], List[float]]:
        with self._lock, self._conn() as conn:
            c = conn.cursor()
            c.execute("SELECT features, label FROM features ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = c.fetchall()
            X, y = [], []
            for f_str, label in rows:
                try:
                    feats = json.loads(f_str)
                    X.append(feats)
                    y.append(float(label))
                except Exception:
                    continue
            return X, y


# =========================
# Resurrection Tracker
# =========================

class ResurrectionTracker:
    def __init__(self, store: PersistentStore):
        self.store = store

    def is_resurrected(self, score: AnomalyScore) -> bool:
        exe = score.details.get("exe")
        if not exe:
            return False
        quarantined = self.store.get_quarantined_executables()
        return exe in quarantined


# =========================
# Group Policy Manager (Stub) + Telemetry Suppression
# =========================

class GroupPolicyManager:
    def __init__(self):
        self.telemetry_suppressed = False

    def enforce_minimal_telemetry(self):
        print("[GroupPolicyManager] enforce_minimal_telemetry() (stub: would apply GPO/registry changes)")
        self.telemetry_suppressed = True

    def restore_default_telemetry(self):
        print("[GroupPolicyManager] restore_default_telemetry() (stub)")
        self.telemetry_suppressed = False

    def is_telemetry_suppressed(self) -> bool:
        return self.telemetry_suppressed


# =========================
# Codex Purge Shell (Integration Layer, Non-Destructive)
# =========================

class CodexPurgeShell:
    def plan_purge(self, decision: PolicyDecision) -> Dict[str, Any]:
        plan = {
            "event_id": decision.event_id,
            "score": decision.score,
            "actions": decision.actions,
            "rules": decision.rules_triggered,
            "timestamp": time.time(),
        }
        print(f"[CodexPurgeShell] Plan: {plan}")
        return plan

    def execute_purge(self, plan: Dict[str, Any]):
        print(f"[CodexPurgeShell] Execute (simulated): {plan}")


# =========================
# Mini‑Rego Policy Engine
# =========================

class MiniRegoEngine:
    """
    Very small, safe-ish Rego-like evaluator:
    - Reads a .rego-like file
    - Supports deny["reason"] { <boolean expression> }
    - Context is 'input' dict
    """

    def __init__(self, path: str = "rules/policies.rego"):
        self.path = path
        self.deny_rules: List[Tuple[str, str]] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            print(f"[MiniRego] No policy file at {self.path}, skipping.")
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[MiniRego] Failed to read {self.path}: {e}")
            return

        current_reason = None
        current_expr_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("deny["):
                try:
                    reason = stripped.split("[", 1)[1].split("]", 1)[0].strip('"\' ')
                    current_reason = reason
                    current_expr_lines = []
                except Exception:
                    current_reason = None
                    current_expr_lines = []
            elif stripped == "}":
                if current_reason and current_expr_lines:
                    expr = " and ".join(current_expr_lines)
                    self.deny_rules.append((current_reason, expr))
                current_reason = None
                current_expr_lines = []
            elif current_reason and stripped and not stripped.startswith("#"):
                if stripped.endswith("{"):
                    stripped = stripped[:-1].strip()
                current_expr_lines.append(stripped)
        print(f"[MiniRego] Loaded {len(self.deny_rules)} deny rules.")

    def _safe_eval(self, expr: str, ctx: Dict[str, Any]) -> bool:
        try:
            allowed = {"True": True, "False": False, "None": None}
            return bool(eval(expr, {"__builtins__": {}}, {"input": ctx, **allowed}))
        except Exception:
            return False

    def evaluate(self, ctx: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        for reason, expr in self.deny_rules:
            if self._safe_eval(expr, ctx):
                reasons.append(reason)
        return reasons


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

    def _safe_eval_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        try:
            allowed_names = {
                "True": True,
                "False": False,
                "None": None,
            }
            return bool(eval(condition, {"__builtins__": {}}, {**allowed_names, **context}))
        except Exception:
            return False

    def evaluate(self, score: AnomalyScore, extra_ctx: Dict[str, Any]) -> PolicyDecision:
        actions: List[str] = []
        triggered: List[str] = []
        ctx = {
            "score": score.score,
            "label": score.label,
            "source": score.details.get("source"),
            "event_type": score.details.get("type"),
            "autopilot_mode": score.details.get("mode"),
            **extra_ctx,
        }
        for r in self.rules.values():
            cond = r.condition.strip()
            if self._safe_eval_condition(cond, ctx):
                actions.append(r.action)
                triggered.append(r.name)
        return PolicyDecision(
            event_id=score.event_id,
            score=score.score,
            actions=actions,
            rules_triggered=triggered,
            metadata=ctx,
        )


class PolicyService(Microservice):
    def __init__(self, bus: LocalEventBus, engine: YAMLRuleEngine, resurrection_tracker: "ResurrectionTracker", rego_engine: MiniRegoEngine):
        super().__init__("PolicyService", bus)
        self.engine = engine
        self.in_q = self.bus.subscribe("detection.scores")
        self.resurrection_tracker = resurrection_tracker
        self.rego_engine = rego_engine

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            try:
                score: AnomalyScore = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            resurrected = self.resurrection_tracker.is_resurrected(score)
            base_decision = self.engine.evaluate(score, {"resurrected": resurrected})
            ctx = {
                "score": score.score,
                "autopilot_mode": base_decision.metadata.get("autopilot_mode"),
            }
            rego_denies = self.rego_engine.evaluate(ctx)
            if rego_denies:
                if "quarantine" not in base_decision.actions:
                    base_decision.actions.append("quarantine")
                base_decision.rules_triggered.extend([f"rego:{r}" for r in rego_denies])
            base_decision.metadata["resurrected"] = resurrected
            self.bus.publish("policy.decisions", base_decision)
        print(f"[{self.name}] stopped")


# =========================
# Training Pipeline for Real Models
# =========================

class TrainingPipeline:
    """
    Simple training pipeline:
    - Pulls labeled features from SQLite
    - Trains a small Torch model
    - Saves model.pt and optionally model.onnx
    """

    def __init__(self, store: PersistentStore, model_dir: str = "models"):
        self.store = store
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, max_samples: int = 5000, epochs: int = 5, lr: float = 1e-3):
        if torch_mod is None or np is None:
            print("[TrainingPipeline] Torch or numpy not available, skipping training.")
            return
        X, y = self.store.load_training_data(limit=max_samples)
        if not X or not y:
            print("[TrainingPipeline] No training data available.")
            return
        X_arr = np.array(X, dtype="float32")
        y_arr = np.array(y, dtype="float32").reshape(-1, 1)
        in_dim = X_arr.shape[1]

        class SmallNet(torch_mod.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.fc1 = torch_mod.nn.Linear(d, 32)
                self.fc2 = torch_mod.nn.Linear(32, 1)

            def forward(self, x):
                x = torch_mod.relu(self.fc1(x))
                return self.fc2(x)

        device = "cuda" if torch_mod.cuda.is_available() else "cpu"
        model = SmallNet(in_dim).to(device)
        opt = torch_mod.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch_mod.nn.BCEWithLogitsLoss()

        X_t = torch_mod.tensor(X_arr, device=device)
        y_t = torch_mod.tensor(y_arr, device=device)

        print(f"[TrainingPipeline] Training on {len(X)} samples, in_dim={in_dim}")
        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            out = model(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            opt.step()
            print(f"[TrainingPipeline] epoch={ep+1}/{epochs} loss={loss.item():.4f}")

        pt_path = os.path.join(self.model_dir, "model.pt")
        traced = torch_mod.jit.trace(model, X_t[:1])
        traced.save(pt_path)
        print(f"[TrainingPipeline] Saved Torch model to {pt_path}")

        try:
            import torch.onnx as torch_onnx
            onnx_path = os.path.join(self.model_dir, "model.onnx")
            torch_onnx.export(
                model,
                X_t[:1],
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=11,
            )
            print(f"[TrainingPipeline] Exported ONNX model to {onnx_path}")
        except Exception as e:
            print(f"[TrainingPipeline] ONNX export failed: {e}")


# =========================
# Response / Enforcement Service
# =========================

class ResponseEngine:
    def __init__(self, store: PersistentStore, gpo: GroupPolicyManager, purge_shell: CodexPurgeShell):
        self.store = store
        self.gpo = gpo
        self.purge_shell = purge_shell

    def alert(self, decision: PolicyDecision):
        print(f"[ResponseEngine] ALERT event={decision.event_id} score={decision.score} rules={decision.rules_triggered}")

    def quarantine(self, decision: PolicyDecision):
        exe = decision.metadata.get("exe")
        pid = decision.metadata.get("pid")
        print(f"[ResponseEngine] QUARANTINE event={decision.event_id} exe={exe} pid={pid}")
        if psutil_mod and pid:
            try:
                p = psutil_mod.Process(pid)
                p.terminate()
                print(f"[ResponseEngine] Terminated PID {pid}")
            except Exception as e:
                print(f"[ResponseEngine] Failed to terminate PID {pid}: {e}")
        if exe:
            self.store.store_quarantine(exe, pid or -1)

    def enforce_group_policy(self, decision: PolicyDecision):
        if "quarantine" in decision.actions and decision.score > 0.9:
            self.gpo.enforce_minimal_telemetry()

    def enforce(self, decision: PolicyDecision):
        if not decision.actions:
            return
        self.alert(decision)
        plan = self.purge_shell.plan_purge(decision)
        for action in decision.actions:
            if action == "quarantine":
                self.quarantine(decision)
            elif action == "alert":
                self.alert(decision)
        self.enforce_group_policy(decision)
        self.purge_shell.execute_purge(plan)


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
                indicator = {
                    "event_id": score.event_id,
                    "score": score.score,
                    "ts": time.time(),
                    "exe": score.details.get("exe"),
                    "source": score.details.get("source"),
                }
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
        self.swarm_indicators: List[Dict[str, Any]] = []
        self.autopilot_assessments: List[AutopilotCommandAssessment] = []

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

    def add_swarm_indicator(self, indicator: Dict[str, Any]):
        with self.lock:
            self.swarm_indicators.append(indicator)

    def push_autopilot_assessment(self, assessment: AutopilotCommandAssessment):
        with self.lock:
            self.autopilot_assessments.append(assessment)

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
                "swarm": list(self.swarm_indicators),
                "autopilot": list(self.autopilot_assessments),
            }


class DashboardService(Microservice):
    def __init__(self, bus: LocalEventBus, backend: DashboardBackend, intel_store: LocalIntelStore):
        super().__init__("DashboardService", bus)
        self.backend = backend
        self.score_q = self.bus.subscribe("detection.scores")
        self.decision_q = self.bus.subscribe("policy.decisions")
        self.event_q = self.bus.subscribe("telemetry.raw")
        self.intel_store = intel_store

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
            swarm_snap = self.intel_store.snapshot()
            for ind in swarm_snap:
                self.backend.add_swarm_indicator(ind)
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

        @self.app.get("/swarm")
        def swarm():
            snap = self.backend.snapshot()
            return {"indicators": snap["swarm"]}

        @self.app.get("/autopilot")
        def autopilot():
            snap = self.backend.snapshot()
            return {
                "assessments": [
                    {
                        "command_id": a.command_id,
                        "mode": a.mode,
                        "risk_score": a.risk_score,
                        "allowed": a.allowed,
                        "reasons": a.reasons,
                    }
                    for a in snap["autopilot"]
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
# Watchdog Service
# =========================

class WatchdogService(Microservice):
    def __init__(self, services: List[Microservice]):
        super().__init__("WatchdogService", bus=LocalEventBus())
        self.services = services

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            for s in self.services:
                if not s.is_alive():
                    print(f"[Watchdog] Service {s.name} is not alive (would restart in a full implementation).")
            time.sleep(2.0)
        print(f"[{self.name}] stopped")


# =========================
# Agent Orchestrator
# =========================

class LocalBehavioralAgent:
    def __init__(self, model_path: str, peers: List[str], rest_port: int, daemon_mode: bool = False):
        self.bus = LocalEventBus()
        self.store = PersistentStore()
        self.intel_store = LocalIntelStore()
        self.gpo = GroupPolicyManager()
        self.purge_shell = CodexPurgeShell()
        self.daemon_mode = daemon_mode

        kernel_sensors = [EBPFHookSensor(), EndpointSecuritySensor()]
        user_monitors = [
            ProcessMonitor(self.bus),
            NetflowMonitor(self.bus),
            MemoryScanner(self.bus, rule_file=None),
            DLLScanner(self.bus),
        ]

        self.telemetry = TelemetryService(self.bus, kernel_sensors, user_monitors)
        self.feature_service = FeatureService(self.bus, FeatureExtractor())
        self.detection_service = DetectionService(self.bus, model_path=model_path)

        self.rule_engine = YAMLRuleEngine()
        self.rule_engine.load_from_yaml()
        self.resurrection_tracker = ResurrectionTracker(self.store)
        self.rego_engine = MiniRegoEngine()
        self.policy_service = PolicyService(self.bus, self.rule_engine, self.resurrection_tracker, self.rego_engine)

        self.response_service = ResponseService(self.bus, ResponseEngine(self.store, self.gpo, self.purge_shell))
        self.swarm_service = SwarmLocalService(self.bus, self.intel_store, peers)
        self.dashboard_backend = DashboardBackend(self.store)
        self.dashboard_service = DashboardService(self.bus, self.dashboard_backend, self.intel_store)

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

        self.watchdog = WatchdogService(self.services)
        self.training_pipeline = TrainingPipeline(self.store)

    def start(self):
        print("[LocalBehavioralAgent] start()")
        for s in self.services:
            s.start()
        self.watchdog.start()

    def stop(self):
        print("[LocalBehavioralAgent] stop()")
        for s in self.services:
            s.stop()
        self.watchdog.stop()
        for s in self.services:
            s.join()
        self.watchdog.join()

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

    def is_telemetry_suppressed(self) -> bool:
        return self.gpo.is_telemetry_suppressed()

    def train_models(self, max_samples: int = 5000, epochs: int = 5):
        self.training_pipeline.train(max_samples=max_samples, epochs=epochs)


# =========================
# Autopilot Security Governor
# =========================

class AutopilotSecurityGovernor:
    """
    Integration layer for your universal autopilot control hub.
    The hub calls assess_command() before executing any motion/route/drive/fly command.
    """

    def __init__(self, agent: LocalBehavioralAgent):
        self.agent = agent

    def assess_command(self, mode: str, command: Dict[str, Any], timeout: float = 1.0) -> AutopilotCommandAssessment:
        cmd_id = str(uuid.uuid4())
        event = TelemetryEvent(
            id=cmd_id,
            timestamp=time.time(),
            source="autopilot_hub",
            event_type="autopilot_cmd",
            payload={
                "mode": mode,
                "command": command,
            },
        )
        self.agent.bus.publish("telemetry.raw", event)

        start = time.time()
        decision_match: Optional[PolicyDecision] = None
        while time.time() - start < timeout:
            snap = self.agent.get_dashboard_snapshot()
            for d in snap["decisions"]:
                if d.event_id == cmd_id:
                    decision_match = d
                    break
            if decision_match:
                break
            time.sleep(0.05)

        if decision_match is None:
            assessment = AutopilotCommandAssessment(
                command_id=cmd_id,
                mode=mode,
                risk_score=0.0,
                allowed=True,
                reasons=["no_decision_timeout"],
                raw_decision=None,
            )
            self.agent.dashboard_backend.push_autopilot_assessment(assessment)
            return assessment

        allowed = "quarantine" not in decision_match.actions
        reasons = decision_match.rules_triggered or []
        assessment = AutopilotCommandAssessment(
            command_id=cmd_id,
            mode=mode,
            risk_score=decision_match.score,
            allowed=allowed,
            reasons=reasons,
            raw_decision=decision_match,
        )
        self.agent.dashboard_backend.push_autopilot_assessment(assessment)
        return assessment


# =========================
# Tkinter God-View Cockpit (Enhanced)
# =========================

class GodViewUI:
    def __init__(self, root: tk.Tk, agent: LocalBehavioralAgent, peers: List[str]):
        self.root = root
        self.agent = agent
        self.peers = peers

        self.root.title("Behavioral Security God-View (MP + GPU + Swarm + Rego + Autopilot Governor)")
        self.root.geometry("1700x950")
        try:
            self.root.call('tk', 'scaling', 1.25)
        except Exception:
            pass

        self._pulse_phase = 0.0

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=2)

        self.frame_threats = ttk.LabelFrame(self.root, text="Threat Stream")
        self.frame_threats.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.frame_heatmap = ttk.LabelFrame(self.root, text="Heatmap / Shaders / Swarm Map")
        self.frame_heatmap.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.frame_timeline = ttk.LabelFrame(self.root, text="Timeline / Threat-Matrix / Autopilot Risk")
        self.frame_timeline.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.frame_status = ttk.LabelFrame(self.root, text="Node Status / Swarm / Lineage / Logs")
        self.frame_status.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.threat_list = tk.Listbox(self.frame_threats, font=("Consolas", 10))
        self.threat_list.pack(fill="both", expand=True)

        self.heatmap_canvas = tk.Canvas(self.frame_heatmap, bg="#101010", height=300)
        self.heatmap_canvas.pack(fill="x", expand=False)

        self.swarm_canvas = tk.Canvas(self.frame_heatmap, bg="#050505", height=200)
        self.swarm_canvas.pack(fill="both", expand=True)

        self.timeline_list = tk.Listbox(self.frame_timeline, font=("Consolas", 10), height=15)
        self.timeline_list.pack(fill="x", expand=False)

        self.autopilot_list = tk.Listbox(self.frame_timeline, font=("Consolas", 10), height=10)
        self.autopilot_list.pack(fill="both", expand=True)

        self.status_text = tk.Text(self.frame_status, font=("Consolas", 10), height=10)
        self.status_text.pack(fill="both", expand=True)

        self.lineage_list = tk.Listbox(self.frame_status, font=("Consolas", 9), height=6)
        self.lineage_list.pack(fill="x", expand=False)

        self.swarm_list = tk.Listbox(self.frame_status, font=("Consolas", 9), height=6)
        self.swarm_list.pack(fill="x", expand=False)

    def _schedule_update(self):
        self._update_ui()
        self.root.after(500, self._schedule_update)

    def _update_ui(self):
        snapshot = self.agent.get_dashboard_snapshot()
        scores: List[AnomalyScore] = snapshot["scores"]
        decisions: List[PolicyDecision] = snapshot["decisions"]
        lineage: Dict[str, Tuple[str, str]] = snapshot["lineage"]
        swarm: List[Dict[str, Any]] = snapshot["swarm"]
        autopilot: List[AutopilotCommandAssessment] = snapshot["autopilot"]

        self.threat_list.delete(0, "end")
        for s in scores[-100:]:
            glyph = "⚠" if s.score > 0.7 else "·"
            self.threat_list.insert(
                "end",
                f"{glyph} {s.event_id[:8]} | score={s.score:.3f}"
            )

        self.timeline_list.delete(0, "end")
        for d in decisions[-100:]:
            actions = ",".join(d.actions) if d.actions else "none"
            resurrected = d.metadata.get("resurrected", False)
            tag = "RES" if resurrected else "   "
            autopilot_mode = d.metadata.get("autopilot_mode")
            ap_tag = f" AP:{autopilot_mode}" if autopilot_mode else ""
            self.timeline_list.insert(
                "end",
                f"{tag}{ap_tag} {d.event_id[:8]} | score={d.score:.3f} | actions={actions}"
            )

        self.autopilot_list.delete(0, "end")
        for a in autopilot[-50:]:
            flag = "ALLOW" if a.allowed else "BLOCK"
            self.autopilot_list.insert(
                "end",
                f"{flag} {a.command_id[:8]} | mode={a.mode} | risk={a.risk_score:.3f} | reasons={','.join(a.reasons)}"
            )

        self.lineage_list.delete(0, "end")
        for eid, (src, etype) in list(lineage.items())[-50:]:
            self.lineage_list.insert("end", f"{eid[:8]} | {src} | {etype}")

        self.swarm_list.delete(0, "end")
        for ind in swarm[-50:]:
            self.swarm_list.insert(
                "end",
                f"{ind.get('event_id','')[:8]} | {ind.get('score',0):.3f} | {ind.get('exe','?')}"
            )

        self._draw_heatmap(scores)
        self._draw_swarm_map(swarm)

        self.status_text.delete("1.0", "end")
        self.status_text.insert("end", f"Recent events: {len(scores)}\n")
        if scores:
            self.status_text.insert("end", f"Last score: {scores[-1].score:.3f}\n")
        self.status_text.insert("end", f"Decisions: {len(decisions)}\n")
        intel = self.agent.get_intel_snapshot()
        self.status_text.insert("end", f"Swarm indicators: {len(intel)}\n")
        self.status_text.insert("end", f"Telemetry suppressed: {self.agent.is_telemetry_suppressed()}\n")
        self.status_text.insert("end", "Agent running with multiprocessing + GPU + swarm + Rego + autopilot governor + training.\n")

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
                g = cp.sqrt(g) + 0.15 * cp.sin(g * 12.0) + 0.05 * cp.cos(g * 20.0)
                grid = cp.asnumpy(g)
            except Exception:
                grid = np.sqrt(grid) + 0.15 * np.sin(grid * 12.0) + 0.05 * np.cos(grid * 20.0)
        else:
            grid = np.sqrt(grid) + 0.15 * np.sin(grid * 12.0) + 0.05 * np.cos(grid * 20.0)

        max_val = float(grid.max()) if grid.size else 1.0
        cell_w = w / side
        cell_h = h / side

        self._pulse_phase += 0.2
        pulse = (math.sin(self._pulse_phase) + 1.0) / 2.0

        for i in range(side):
            for j in range(side):
                v = grid[i, j] / (max_val + 1e-6)
                r = int(255 * v)
                g = int(255 * (1 - v) * (0.5 + 0.5 * pulse))
                b = int(120 * pulse * v)
                color = f"#{r:02x}{g:02x}{b:02x}"
                x0 = j * cell_w
                y0 = i * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                self.heatmap_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        self.heatmap_canvas.create_text(
            10, 10, anchor="nw", fill="#ffffff",
            text="Score Heatmap (GPU-accelerated shaders if available)",
            font=("Consolas", 10, "bold")
        )
        self.heatmap_canvas.create_text(
            w - 10, h - 10, anchor="se", fill="#ff00ff",
            text="⟡ Codex Matrix / Autopilot Governor",
            font=("Consolas", 12, "bold")
        )

    def _draw_swarm_map(self, swarm: List[Dict[str, Any]]):
        self.swarm_canvas.delete("all")
        w = self.swarm_canvas.winfo_width() or 600
        h = self.swarm_canvas.winfo_height() or 200

        nodes = ["local"] + [f"peer_{i}" for i in range(len(self.peers))]
        num_nodes = len(nodes)
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 20

        positions: Dict[str, Tuple[float, float]] = {}
        for idx, node in enumerate(nodes):
            angle = 2 * math.pi * idx / max(1, num_nodes)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            positions[node] = (x, y)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                n1, n2 = nodes[i], nodes[j]
                x1, y1 = positions[n1]
                x2, y2 = positions[n2]
                self.swarm_canvas.create_line(x1, y1, x2, y2, fill="#202020")

        for node, (x, y) in positions.items():
            fill = "#00ff00" if node == "local" else "#0088ff"
            self.swarm_canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill=fill, outline="#ffffff")
            self.swarm_canvas.create_text(x, y - 15, text=node, fill="#ffffff", font=("Consolas", 9))

        if swarm:
            self.swarm_canvas.create_text(
                10, 10, anchor="nw", fill="#ffaa00",
                text=f"Swarm indicators: {len(swarm)}",
                font=("Consolas", 10)
            )
        else:
            self.swarm_canvas.create_text(
                10, 10, anchor="nw", fill="#555555",
                text="Swarm map: no high‑risk indicators yet.",
                font=("Consolas", 10)
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


def main(daemon_mode: bool = False):
    mp.set_start_method("spawn", force=True)

    engine_ok = verify_tensorrt_engine("models/tensorrt_engine.trt")
    model_path = "models/tensorrt_engine.trt" if engine_ok else "models/fallback.trt"

    local_ip = get_local_ip()
    rest_port = 5000
    peers: List[str] = []  # e.g., [f"http://{local_ip}:5001"]

    agent = LocalBehavioralAgent(model_path=model_path, peers=peers, rest_port=rest_port, daemon_mode=daemon_mode)
    governor = AutopilotSecurityGovernor(agent)
    agent.start()

    assessment = governor.assess_command(
        mode="road_autopilot",
        command={"target_speed": 80, "lane_change": "left"},
        timeout=1.0,
    )
    print(f"[AutopilotGovernor] assessment={assessment}")

    if daemon_mode:
        print("[main] Running in daemon mode (no UI). Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            agent.stop()
        return

    root = tk.Tk()
    ui = GodViewUI(root, agent, peers)

    def on_close():
        agent.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main(daemon_mode=False)
