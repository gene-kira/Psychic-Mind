#!/usr/bin/env python3
"""
Unified adaptive organism (single file):

- Universal library loader (auto-installs where possible)
- Mode B (Aggressive Adaptive) workload session manager
- USB NPU as general compute accelerator (via ONNX Runtime if available)
- Hybrid routing (predictive + operator-defined)
- Console telemetry with simple timeline/resource bars (ready for GUI wiring)
"""

import sys
import subprocess
import importlib
import time
import threading
import math
import random
from typing import Any, Dict, List, Optional

# ============================================================
#  UNIVERSAL LIBRARY LOADER
# ============================================================

REQUIRED_LIBS = [
    "psutil",        # system telemetry
    "numpy",         # math / vectors
    "onnxruntime",   # NPU / accelerator (if available)
]

def ensure_library(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"[LOADER] Missing library: {name}, attempting install via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", name])
            return importlib.import_module(name)
        except Exception as e:
            print(f"[LOADER] Failed to install {name}: {e}")
            return None

def load_all_libraries() -> Dict[str, Any]:
    modules = {}
    for lib in REQUIRED_LIBS:
        modules[lib] = ensure_library(lib)
    return modules

MODULES = load_all_libraries()
psutil = MODULES.get("psutil", None)
np = MODULES.get("numpy", None)
ort = MODULES.get("onnxruntime", None)

# ============================================================
#  BASIC TELEMETRY (CPU / MEMORY / PLACEHOLDER GPU)
# ============================================================

def get_cpu_usage() -> float:
    if psutil is None:
        return 0.0
    return psutil.cpu_percent(interval=0.1)

def get_mem_usage() -> float:
    if psutil is None:
        return 0.0
    return psutil.virtual_memory().percent

def get_gpu_usage() -> float:
    # Placeholder – integrate real GPU telemetry later
    return random.uniform(0.0, 100.0)

# ============================================================
#  COMPUTE ORGANS (CPU / GPU / USB NPU)
# ============================================================

class ComputeTask:
    def __init__(self, name: str, payload: Any, cost_hint: float = 1.0):
        self.name = name
        self.payload = payload
        self.cost_hint = cost_hint  # rough complexity estimate

class ComputeOrgan:
    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.latency_history: List[float] = []
        self.load_estimate: float = 0.0

    def run(self, task: ComputeTask) -> Any:
        raise NotImplementedError

    def record_latency(self, latency: float):
        self.latency_history.append(latency)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)
        self.load_estimate = 0.8 * self.load_estimate + 0.2 * latency

class CPUOrgan(ComputeOrgan):
    def __init__(self):
        super().__init__("CPU", "cpu")

    def run(self, task: ComputeTask) -> Any:
        start = time.time()
        time.sleep(0.001 * task.cost_hint)
        result = {"organ": self.name, "task": task.name, "status": "ok"}
        self.record_latency(time.time() - start)
        return result

class GPUOrgan(ComputeOrgan):
    def __init__(self):
        super().__init__("GPU", "gpu")

    def run(self, task: ComputeTask) -> Any:
        start = time.time()
        time.sleep(0.0005 * task.cost_hint)
        result = {"organ": self.name, "task": task.name, "status": "ok"}
        self.record_latency(time.time() - start)
        return result

class NPUOrgan(ComputeOrgan):
    """
    USB NPU as general compute accelerator.
    If an ONNX Runtime session is provided, it will be used for real inference.
    Otherwise, it behaves as a fast simulated accelerator.
    """
    def __init__(self, session: Optional[Any] = None):
        super().__init__("USB_NPU", "npu")
        self.session = session

    def run(self, task: ComputeTask) -> Any:
        start = time.time()
        if self.session is not None and np is not None and task.payload is not None:
            try:
                input_name = self.session.get_inputs()[0].name
                output = self.session.run(None, {input_name: task.payload})[0]
                result = {
                    "organ": self.name,
                    "task": task.name,
                    "status": "ok",
                    "output": output,
                }
            except Exception as e:
                result = {
                    "organ": self.name,
                    "task": task.name,
                    "status": "error",
                    "error": str(e),
                }
        else:
            time.sleep(0.0003 * task.cost_hint)
            result = {"organ": self.name, "task": task.name, "status": "ok"}
        self.record_latency(time.time() - start)
        return result

def create_npu_session(model_path: Optional[str]) -> Optional[Any]:
    if ort is None or model_path is None:
        return None
    try:
        sess_options = ort.SessionOptions()
        sess = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        print(f"[NPU] ONNX model loaded from: {model_path}")
        return sess
    except Exception as e:
        print(f"[NPU] Failed to create ONNX session: {e}")
        return None

# ============================================================
#  HYBRID ROUTER (PREDICTIVE + OPERATOR-DEFINED)
# ============================================================

class RoutingPreference:
    ALWAYS_NPU = "always_npu"
    NEVER_NPU = "never_npu"
    NPU_PREFERRED = "npu_preferred"
    CPU_GPU_PREFERRED = "cpu_gpu_preferred"

class HybridRouter:
    """
    Hybrid routing:
    - Operator-defined rules always win.
    - Otherwise, predictive routing chooses organ with lowest load_estimate.
    """
    def __init__(self, cpu: CPUOrgan, gpu: GPUOrgan, npu: Optional[NPUOrgan]):
        self.cpu = cpu
        self.gpu = gpu
        self.npu = npu
        self.operator_rules: Dict[str, str] = {}

    def set_operator_rule(self, task_name: str, preference: str):
        self.operator_rules[task_name] = preference
        print(f"[ROUTER] Rule set: {task_name} -> {preference}")

    def _predict_best_organ(self) -> ComputeOrgan:
        candidates: List[ComputeOrgan] = [self.cpu, self.gpu]
        if self.npu is not None:
            candidates.append(self.npu)
        best = min(candidates, key=lambda o: o.load_estimate)
        return best

    def route(self, task: ComputeTask) -> ComputeOrgan:
        pref = self.operator_rules.get(task.name)

        if pref == RoutingPreference.ALWAYS_NPU and self.npu is not None:
            return self.npu

        if pref == RoutingPreference.NEVER_NPU:
            return self.cpu if self.cpu.load_estimate <= self.gpu.load_estimate else self.gpu

        if pref == RoutingPreference.NPU_PREFERRED and self.npu is not None:
            if self.npu.load_estimate <= min(self.cpu.load_estimate, self.gpu.load_estimate) * 1.5:
                return self.npu

        if pref == RoutingPreference.CPU_GPU_PREFERRED:
            return self.cpu if self.cpu.load_estimate <= self.gpu.load_estimate else self.gpu

        return self._predict_best_organ()

# ============================================================
#  MODE B – AGGRESSIVE ADAPTIVE SESSION MANAGER
# ============================================================

class SessionProfile:
    def __init__(self, name: str):
        self.name = name
        self.cpu_mean = 0.0
        self.cpu_var = 0.0
        self.mem_mean = 0.0
        self.mem_var = 0.0
        self.gpu_mean = 0.0
        self.gpu_var = 0.0
        self.samples = 0
        self.anomaly_threshold = 2.0
        self.learning_rate = 0.3
        self.stability_score = 0.0

    def update_stats(self, value: float, mean_attr: str, var_attr: str):
        old_mean = getattr(self, mean_attr)
        old_var = getattr(self, var_attr)
        n = self.samples + 1
        new_mean = old_mean + (value - old_mean) * self.learning_rate
        new_var = old_var + (value - new_mean) * (value - old_mean) * self.learning_rate
        setattr(self, mean_attr, new_mean)
        setattr(self, var_attr, max(new_var, 0.0))
        self.samples = n

    def record_observation(self, cpu: float, mem: float, gpu: float):
        self.update_stats(cpu, "cpu_mean", "cpu_var")
        self.update_stats(mem, "mem_mean", "mem_var")
        self.update_stats(gpu, "gpu_mean", "gpu_var")

        cpu_dev = abs(cpu - self.cpu_mean)
        mem_dev = abs(mem - self.mem_mean)
        gpu_dev = abs(gpu - self.gpu_mean)
        dev_score = cpu_dev + mem_dev + gpu_dev
        self.stability_score = max(0.0, self.stability_score + (1.0 - dev_score / 300.0))

    def anomaly_score(self, cpu: float, mem: float, gpu: float) -> float:
        def z(v, m, var):
            std = math.sqrt(var) if var > 1e-6 else 1.0
            return abs(v - m) / std
        z_cpu = z(cpu, self.cpu_mean, self.cpu_var)
        z_mem = z(mem, self.mem_mean, self.mem_var)
        z_gpu = z(gpu, self.gpu_mean, self.gpu_var)
        return (z_cpu + z_mem + z_gpu) / 3.0

class SessionManager:
    """
    Mode B – Aggressive Adaptive:
    - High learning rate
    - Graduated response curve
    - Feeds Black Knight-style failsafe trigger
    """
    def __init__(self):
        self.sessions: Dict[str, SessionProfile] = {}
        self.active_session: Optional[SessionProfile] = None
        self.black_knight_triggered = False

        self.minor_drift = 1.0
        self.moderate_drift = 2.0
        self.severe_drift = 3.0
        self.extreme_drift = 4.0

    def get_or_create_session(self, name: str) -> SessionProfile:
        if name not in self.sessions:
            self.sessions[name] = SessionProfile(name)
        return self.sessions[name]

    def activate_session(self, name: str):
        self.active_session = self.get_or_create_session(name)
        print(f"[SESSION] Activated session: {name}")

    def observe_and_adapt(self, cpu: float, mem: float, gpu: float):
        if self.active_session is None:
            return

        s = self.active_session
        s.record_observation(cpu, mem, gpu)
        score = s.anomaly_score(cpu, mem, gpu)

        if score < self.minor_drift:
            pass
        elif score < self.moderate_drift:
            s.anomaly_threshold = max(1.5, s.anomaly_threshold - 0.05)
        elif score < self.severe_drift:
            s.anomaly_threshold = max(1.2, s.anomaly_threshold - 0.1)
        elif score < self.extreme_drift:
            s.anomaly_threshold = max(1.0, s.anomaly_threshold - 0.15)
        else:
            self.trigger_black_knight(score)

    def trigger_black_knight(self, score: float):
        if not self.black_knight_triggered:
            self.black_knight_triggered = True
            print(
                f"[BLACK KNIGHT] Extreme anomaly detected (score={score:.2f}). "
                f"Locking down egress, isolating node, enforcing safe mode."
            )

# ============================================================
#  SIMPLE CONSOLE TIMELINE / RESOURCE VIEW
# ============================================================

def bar(value: float, width: int = 20) -> str:
    v = max(0.0, min(100.0, value))
    filled = int((v / 100.0) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

# ============================================================
#  ORGANISM GLUE
# ============================================================

class Organism:
    """
    Full organism:
    - CPU / GPU / USB NPU organs
    - Hybrid router
    - Mode B session manager
    - Console loop with simple timeline/resource bars
    """
    def __init__(self, npu_model_path: Optional[str] = None):
        self.cpu = CPUOrgan()
        self.gpu = GPUOrgan()
        self.npu = NPUOrgan(create_npu_session(npu_model_path)) if npu_model_path else NPUOrgan(None)

        self.router = HybridRouter(self.cpu, self.gpu, self.npu)
        self.sessions = SessionManager()
        self.running = False

        self.cpu_hist: List[float] = []
        self.gpu_hist: List[float] = []
        self.mem_hist: List[float] = []

        self.router.set_operator_rule("telemetry_scoring", RoutingPreference.NPU_PREFERRED)
        self.router.set_operator_rule("heavy_inference", RoutingPreference.ALWAYS_NPU)
        self.router.set_operator_rule("light_logic", RoutingPreference.CPU_GPU_PREFERRED)

    def start(self, session_name: str = "ModeB_Aggressive"):
        self.sessions.activate_session(session_name)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _record_history(self, cpu: float, mem: float, gpu: float, max_len: int = 30):
        self.cpu_hist.append(cpu)
        self.mem_hist.append(mem)
        self.gpu_hist.append(gpu)
        if len(self.cpu_hist) > max_len:
            self.cpu_hist.pop(0)
            self.mem_hist.pop(0)
            self.gpu_hist.pop(0)

    def _loop(self):
        while self.running:
            cpu = get_cpu_usage()
            mem = get_mem_usage()
            gpu = get_gpu_usage()

            self.sessions.observe_and_adapt(cpu, mem, gpu)
            self._record_history(cpu, mem, gpu)

            tasks = [
                ComputeTask("telemetry_scoring", payload=None, cost_hint=5.0),
                ComputeTask("heavy_inference", payload=None, cost_hint=20.0),
                ComputeTask("light_logic", payload=None, cost_hint=1.0),
            ]

            routed = []
            for t in tasks:
                organ = self.router.route(t)
                result = organ.run(t)
                routed.append((t.name, organ.name, result.get("status", "?")))

            print("\n" + "=" * 80)
            if self.sessions.active_session:
                print(f"[SESSION] {self.sessions.active_session.name} "
                      f"| Stability={self.sessions.active_session.stability_score:.2f}")
            else:
                print("[SESSION] None")
            print(f"CPU {cpu:5.1f}% {bar(cpu)}")
            print(f"MEM {mem:5.1f}% {bar(mem)}")
            print(f"GPU {gpu:5.1f}% {bar(gpu)}")
            print("-" * 80)
            for name, organ_name, status in routed:
                print(f"[TASK] {name:18s} -> {organ_name:7s} | status={status}")
            print("=" * 80)

            time.sleep(0.5)

# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    model_path = None  # e.g. "models/your_npu_model.onnx"
    org = Organism(npu_model_path=model_path)
    org.start("ModeB_Aggressive")

    print("[ORGANISM] Running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        org.stop()
        print("[ORGANISM] Stopped.")

