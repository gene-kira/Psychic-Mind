#!/usr/bin/env python3
"""
Unified adaptive organism (single file, extended):

- Universal library loader (auto-installs where possible)
- Mode B (Aggressive Adaptive) workload session manager
- USB NPU as general compute accelerator (via ONNX Runtime if available)
- Hybrid routing (predictive + operator-defined + UI context + ONNX predictor)
- Dual-stack UI context organ (uiautomation + pywinauto fallback)
- Persistence (sessions + UI patterns to JSON)
- Tiny ONNX predictor hook for routing (optional)
- Real GPU telemetry when GPUtil is available (fallback to random)
- Task scheduler (dynamic task set)
- Simple distributed node abstraction (multi-node ready)
- GUI cockpit (Tkinter) with symbolic overlays for organs and status
"""

import sys
import subprocess
import importlib
import time
import threading
import math
import random
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None

# ============================================================
#  UNIVERSAL LIBRARY LOADER
# ============================================================

REQUIRED_LIBS = [
    "psutil",        # system telemetry
    "numpy",         # math / vectors
    "onnxruntime",   # NPU / accelerator (if available)
    "uiautomation",  # UI automation (primary)
    "pywinauto",     # UI automation (fallback)
    "GPUtil",        # GPU telemetry (if available)
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
uia = MODULES.get("uiautomation", None)
pywinauto = MODULES.get("pywinauto", None)
GPUtil = MODULES.get("GPUtil", None)

# ============================================================
#  BASIC TELEMETRY (CPU / MEMORY / GPU)
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
    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return float(gpus[0].load * 100.0)
        except Exception:
            pass
    return random.uniform(0.0, 100.0)

# ============================================================
#  UI CONTEXT ORGAN (DUAL-STACK: UIAUTOMATION + PYWINAUTO)
# ============================================================

class UIContext:
    def __init__(self, app_name: str, window_title: str):
        self.app_name = app_name
        self.window_title = window_title

    def key(self) -> str:
        return f"{self.app_name}|{self.window_title}"

class UIContextOrgan:
    def __init__(self):
        self.last_context: Optional[UIContext] = None

    def _get_with_uiautomation(self) -> Optional[UIContext]:
        if uia is None:
            return None
        try:
            win = uia.GetForegroundControl()
            name = win.Name or ""
            pid = win.ProcessId
            app_name = f"pid:{pid}"
            return UIContext(app_name=app_name, window_title=name)
        except Exception:
            return None

    def _get_with_pywinauto(self) -> Optional[UIContext]:
        if pywinauto is None:
            return None
        try:
            from pywinauto import Desktop
            desk = Desktop(backend="uia")
            handle = desk.get_active()[0].handle
            win = desk.window(handle=handle)
            title = win.window_text() or ""
            app_name = win.process_id()
            return UIContext(app_name=f"pid:{app_name}", window_title=title)
        except Exception:
            return None

    def get_context(self) -> Optional[UIContext]:
        ctx = self._get_with_uiautomation()
        if ctx is None:
            ctx = self._get_with_pywinauto()
        self.last_context = ctx
        return ctx

# ============================================================
#  COMPUTE ORGANS (CPU / GPU / USB NPU)
# ============================================================

class ComputeTask:
    def __init__(self, name: str, payload: Any, cost_hint: float = 1.0, ui_context: Optional[UIContext] = None):
        self.name = name
        self.payload = payload
        self.cost_hint = cost_hint
        self.ui_context = ui_context

class ComputeOrgan:
    def __init__(self, name: str, kind: str, symbol: str):
        self.name = name
        self.kind = kind
        self.symbol = symbol
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
        super().__init__("CPU", "cpu", "C")

    def run(self, task: ComputeTask) -> Any:
        start = time.time()
        time.sleep(0.001 * task.cost_hint)
        result = {"organ": self.name, "task": task.name, "status": "ok"}
        self.record_latency(time.time() - start)
        return result

class GPUOrgan(ComputeOrgan):
    def __init__(self):
        super().__init__("GPU", "gpu", "G")

    def run(self, task: ComputeTask) -> Any:
        start = time.time()
        time.sleep(0.0005 * task.cost_hint)
        result = {"organ": self.name, "task": task.name, "status": "ok"}
        self.record_latency(time.time() - start)
        return result

class NPUOrgan(ComputeOrgan):
    def __init__(self, session: Optional[Any] = None):
        super().__init__("USB_NPU", "npu", "N")
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

def create_onnx_session(model_path: Optional[str]) -> Optional[Any]:
    if ort is None or model_path is None:
        return None
    try:
        sess_options = ort.SessionOptions()
        sess = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        print(f"[ONNX] Model loaded from: {model_path}")
        return sess
    except Exception as e:
        print(f"[ONNX] Failed to create session: {e}")
        return None

# ============================================================
#  HYBRID ROUTER (PREDICTIVE + OPERATOR + UI CONTEXT + ONNX)
# ============================================================

class RoutingPreference:
    ALWAYS_NPU = "always_npu"
    NEVER_NPU = "never_npu"
    NPU_PREFERRED = "npu_preferred"
    CPU_GPU_PREFERRED = "cpu_gpu_preferred"

class HybridRouter:
    def __init__(self, cpu: CPUOrgan, gpu: GPUOrgan, npu: Optional[NPUOrgan], predictor_session: Optional[Any] = None):
        self.cpu = cpu
        self.gpu = gpu
        self.npu = npu
        self.operator_rules: Dict[str, str] = {}
        self.ui_context_prefs: Dict[str, str] = {}
        self.predictor_session = predictor_session

    def set_operator_rule(self, task_name: str, preference: str):
        self.operator_rules[task_name] = preference
        print(f"[ROUTER] Rule set: {task_name} -> {preference}")

    def learn_ui_preference(self, ctx: Optional[UIContext], organ: ComputeOrgan):
        if ctx is None:
            return
        key = ctx.key()
        prev = self.ui_context_prefs.get(key)
        if prev is None:
            self.ui_context_prefs[key] = organ.kind
        else:
            if prev != organ.kind and random.random() < 0.1:
                self.ui_context_prefs[key] = organ.kind

    def _predict_best_organ_base(self) -> ComputeOrgan:
        candidates: List[ComputeOrgan] = [self.cpu, self.gpu]
        if self.npu is not None:
            candidates.append(self.npu)
        best = min(candidates, key=lambda o: o.load_estimate)
        return best

    def _apply_ui_context_bias(self, ctx: Optional[UIContext], base: ComputeOrgan) -> ComputeOrgan:
        if ctx is None:
            return base
        key = ctx.key()
        pref_kind = self.ui_context_prefs.get(key)
        if pref_kind is None:
            return base
        candidates: Dict[str, ComputeOrgan] = {
            "cpu": self.cpu,
            "gpu": self.gpu,
            "npu": self.npu if self.npu is not None else self.cpu,
        }
        preferred = candidates.get(pref_kind, base)
        if preferred.load_estimate <= base.load_estimate * 1.5:
            return preferred
        return base

    def _apply_onnx_predictor_bias(
        self,
        cpu: float,
        mem: float,
        gpu: float,
        base: ComputeOrgan
    ) -> ComputeOrgan:
        if self.predictor_session is None or np is None:
            return base
        try:
            x = np.array([[cpu, mem, gpu]], dtype=np.float32)
            input_name = self.predictor_session.get_inputs()[0].name
            out = self.predictor_session.run(None, {input_name: x})[0]
            scores = out[0]
            mapping: List[Tuple[str, ComputeOrgan]] = [
                ("cpu", self.cpu),
                ("gpu", self.gpu),
                ("npu", self.npu if self.npu is not None else self.cpu),
            ]
            best_idx = int(np.argmax(scores))
            candidate = mapping[best_idx][1]
            if candidate.load_estimate <= base.load_estimate * 1.5:
                return candidate
            return base
        except Exception:
            return base

    def route(self, task: ComputeTask, cpu: float, mem: float, gpu: float) -> ComputeOrgan:
        pref = self.operator_rules.get(task.name)

        if pref == RoutingPreference.ALWAYS_NPU and self.npu is not None:
            organ = self.npu
        elif pref == RoutingPreference.NEVER_NPU:
            organ = self.cpu if self.cpu.load_estimate <= self.gpu.load_estimate else self.gpu
        elif pref == RoutingPreference.NPU_PREFERRED and self.npu is not None:
            if self.npu.load_estimate <= min(self.cpu.load_estimate, self.gpu.load_estimate) * 1.5:
                organ = self.npu
            else:
                organ = self._predict_best_organ_base()
        elif pref == RoutingPreference.CPU_GPU_PREFERRED:
            organ = self.cpu if self.cpu.load_estimate <= self.gpu.load_estimate else self.gpu
        else:
            base = self._predict_best_organ_base()
            biased = self._apply_ui_context_bias(task.ui_context, base)
            organ = self._apply_onnx_predictor_bias(cpu, mem, gpu, biased)

        self.learn_ui_preference(task.ui_context, organ)
        return organ

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
        self.ui_context_counts: Dict[str, int] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cpu_mean": self.cpu_mean,
            "cpu_var": self.cpu_var,
            "mem_mean": self.mem_mean,
            "mem_var": self.mem_var,
            "gpu_mean": self.gpu_mean,
            "gpu_var": self.gpu_var,
            "samples": self.samples,
            "anomaly_threshold": self.anomaly_threshold,
            "learning_rate": self.learning_rate,
            "stability_score": self.stability_score,
            "ui_context_counts": self.ui_context_counts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionProfile":
        s = cls(data.get("name", "unknown"))
        s.cpu_mean = data.get("cpu_mean", 0.0)
        s.cpu_var = data.get("cpu_var", 0.0)
        s.mem_mean = data.get("mem_mean", 0.0)
        s.mem_var = data.get("mem_var", 0.0)
        s.gpu_mean = data.get("gpu_mean", 0.0)
        s.gpu_var = data.get("gpu_var", 0.0)
        s.samples = data.get("samples", 0)
        s.anomaly_threshold = data.get("anomaly_threshold", 2.0)
        s.learning_rate = data.get("learning_rate", 0.3)
        s.stability_score = data.get("stability_score", 0.0)
        s.ui_context_counts = data.get("ui_context_counts", {})
        return s

    def update_stats(self, value: float, mean_attr: str, var_attr: str):
        old_mean = getattr(self, mean_attr)
        old_var = getattr(self, var_attr)
        n = self.samples + 1
        new_mean = old_mean + (value - old_mean) * self.learning_rate
        new_var = old_var + (value - new_mean) * (value - old_mean) * self.learning_rate
        setattr(self, mean_attr, new_mean)
        setattr(self, var_attr, max(new_var, 0.0))
        self.samples = n

    def record_observation(self, cpu: float, mem: float, gpu: float, ctx: Optional[UIContext]):
        self.update_stats(cpu, "cpu_mean", "cpu_var")
        self.update_stats(mem, "mem_mean", "mem_var")
        self.update_stats(gpu, "gpu_mean", "gpu_var")

        cpu_dev = abs(cpu - self.cpu_mean)
        mem_dev = abs(mem - self.mem_mean)
        gpu_dev = abs(gpu - self.gpu_mean)
        dev_score = cpu_dev + mem_dev + gpu_dev
        self.stability_score = max(0.0, self.stability_score + (1.0 - dev_score / 300.0))

        if ctx is not None:
            key = ctx.key()
            self.ui_context_counts[key] = self.ui_context_counts.get(key, 0) + 1

    def anomaly_score(self, cpu: float, mem: float, gpu: float) -> float:
        def z(v, m, var):
            std = math.sqrt(var) if var > 1e-6 else 1.0
            return abs(v - m) / std
        z_cpu = z(cpu, self.cpu_mean, self.cpu_var)
        z_mem = z(mem, self.mem_mean, self.mem_var)
        z_gpu = z(gpu, self.gpu_mean, self.gpu_var)
        return (z_cpu + z_mem + z_gpu) / 3.0

class SessionManager:
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

    def observe_and_adapt(self, cpu: float, mem: float, gpu: float, ctx: Optional[UIContext]):
        if self.active_session is None:
            return

        s = self.active_session
        s.record_observation(cpu, mem, gpu, ctx)
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessions": {name: s.to_dict() for name, s in self.sessions.items()},
            "active_session": self.active_session.name if self.active_session else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionManager":
        sm = cls()
        sessions_data = data.get("sessions", {})
        for name, sdata in sessions_data.items():
            sm.sessions[name] = SessionProfile.from_dict(sdata)
        active_name = data.get("active_session")
        if active_name is not None and active_name in sm.sessions:
            sm.active_session = sm.sessions[active_name]
        return sm

# ============================================================
#  TASK SCHEDULER
# ============================================================

class TaskScheduler:
    """
    Simple scheduler:
    - Maintains a dynamic list of tasks
    - Can adjust tasks based on UI context or load
    """
    def __init__(self):
        self.base_tasks: List[Tuple[str, float]] = [
            ("telemetry_scoring", 5.0),
            ("heavy_inference", 20.0),
            ("light_logic", 1.0),
        ]

    def get_tasks(self, ctx: Optional[UIContext], cpu: float, mem: float, gpu: float) -> List[ComputeTask]:
        tasks: List[ComputeTask] = []
        for name, cost in self.base_tasks:
            # Example: if GPU is high, reduce heavy_inference frequency
            if name == "heavy_inference" and gpu > 80.0:
                if random.random() < 0.5:
                    continue
            tasks.append(ComputeTask(name, payload=None, cost_hint=cost, ui_context=ctx))
        return tasks

# ============================================================
#  SIMPLE DISTRIBUTED NODE ABSTRACTION
# ============================================================

class Node:
    """
    Simple node abstraction:
    - For now, just a label and weight
    - Future: real networking / RPC
    """
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.last_load: float = 0.0

class NodeManager:
    """
    Manages multiple nodes (including local):
    - Can choose a node based on load/weight
    - Currently local-only, but ready for extension
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {"local": Node("local", weight=1.0)}

    def update_local_load(self, load: float):
        self.nodes["local"].last_load = load

    def choose_node(self) -> Node:
        # For now, always local; extend later
        return self.nodes["local"]

# ============================================================
#  SIMPLE CONSOLE BAR
# ============================================================

def bar(value: float, width: int = 20) -> str:
    v = max(0.0, min(100.0, value))
    filled = int((v / 100.0) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

# ============================================================
#  GUI COCKPIT (TKINTER)
# ============================================================

class CockpitGUI:
    """
    Simple Tkinter cockpit:
    - Shows CPU/MEM/GPU
    - Shows session stability
    - Shows UI context
    - Shows last routed tasks with organ symbols
    """
    def __init__(self, organism: "Organism"):
        self.organism = organism
        if tk is None:
            self.root = None
            return
        self.root = tk.Tk()
        self.root.title("Adaptive Organism Cockpit")

        self.cpu_var = tk.StringVar()
        self.mem_var = tk.StringVar()
        self.gpu_var = tk.StringVar()
        self.session_var = tk.StringVar()
        self.ui_var = tk.StringVar()
        self.tasks_var = tk.StringVar()

        ttk.Label(self.root, text="Session:").grid(row=0, column=0, sticky="w")
        ttk.Label(self.root, textvariable=self.session_var).grid(row=0, column=1, sticky="w")

        ttk.Label(self.root, text="UI Context:").grid(row=1, column=0, sticky="w")
        ttk.Label(self.root, textvariable=self.ui_var, wraplength=400).grid(row=1, column=1, sticky="w")

        ttk.Label(self.root, text="CPU:").grid(row=2, column=0, sticky="w")
        ttk.Label(self.root, textvariable=self.cpu_var).grid(row=2, column=1, sticky="w")

        ttk.Label(self.root, text="MEM:").grid(row=3, column=0, sticky="w")
        ttk.Label(self.root, textvariable=self.mem_var).grid(row=3, column=1, sticky="w")

        ttk.Label(self.root, text="GPU:").grid(row=4, column=0, sticky="w")
        ttk.Label(self.root, textvariable=self.gpu_var).grid(row=4, column=1, sticky="w")

        ttk.Label(self.root, text="Tasks:").grid(row=5, column=0, sticky="nw")
        ttk.Label(self.root, textvariable=self.tasks_var, justify="left", wraplength=400).grid(row=5, column=1, sticky="w")

        self.root.after(500, self._update)

    def _update(self):
        if self.root is None:
            return
        cpu = self.organism.last_cpu
        mem = self.organism.last_mem
        gpu = self.organism.last_gpu
        sess = self.organism.sessions.active_session
        ctx = self.organism.last_ctx
        routed = self.organism.last_routed

        self.cpu_var.set(f"{cpu:5.1f}% {bar(cpu)}")
        self.mem_var.set(f"{mem:5.1f}% {bar(mem)}")
        self.gpu_var.set(f"{gpu:5.1f}% {bar(gpu)}")

        if sess is not None:
            self.session_var.set(f"{sess.name} | Stability={sess.stability_score:.2f}")
        else:
            self.session_var.set("None")

        if ctx is not None:
            self.ui_var.set(f"{ctx.app_name} | '{ctx.window_title}'")
        else:
            self.ui_var.set("No context")

        lines = []
        for name, organ_name, symbol, status in routed:
            lines.append(f"{symbol} {organ_name}: {name} [{status}]")
        self.tasks_var.set("\n".join(lines))

        self.root.after(500, self._update)

    def run(self):
        if self.root is not None:
            self.root.mainloop()

# ============================================================
#  ORGANISM GLUE + PERSISTENCE + DISTRIBUTED + GUI
# ============================================================

STATE_FILE = "organism_state.json"

class Organism:
    def __init__(
        self,
        npu_model_path: Optional[str] = None,
        routing_predictor_model_path: Optional[str] = None,
    ):
        self.cpu = CPUOrgan()
        self.gpu = GPUOrgan()
        self.npu = NPUOrgan(create_onnx_session(npu_model_path)) if npu_model_path else NPUOrgan(None)

        predictor_session = create_onnx_session(routing_predictor_model_path)
        self.ui_organ = UIContextOrgan()
        self.router = HybridRouter(self.cpu, self.gpu, self.npu, predictor_session=predictor_session)

        self.sessions = SessionManager()
        self._load_state()

        self.scheduler = TaskScheduler()
        self.node_manager = NodeManager()

        self.running = False
        self.last_cpu = 0.0
        self.last_mem = 0.0
        self.last_gpu = 0.0
        self.last_ctx: Optional[UIContext] = None
        self.last_routed: List[Tuple[str, str, str, str]] = []

        self.router.set_operator_rule("telemetry_scoring", RoutingPreference.NPU_PREFERRED)
        self.router.set_operator_rule("heavy_inference", RoutingPreference.ALWAYS_NPU)
        self.router.set_operator_rule("light_logic", RoutingPreference.CPU_GPU_PREFERRED)

    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.sessions = SessionManager.from_dict(data.get("sessions_state", {}))
            self.router.ui_context_prefs = data.get("ui_context_prefs", {})
            print("[STATE] Loaded organism state from disk.")
        except Exception as e:
            print(f"[STATE] Failed to load state: {e}")

    def _save_state(self):
        try:
            data = {
                "sessions_state": self.sessions.to_dict(),
                "ui_context_prefs": self.router.ui_context_prefs,
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print("[STATE] Saved organism state to disk.")
        except Exception as e:
            print(f"[STATE] Failed to save state: {e}")

    def start(self, session_name: str = "ModeB_Aggressive"):
        self.sessions.activate_session(session_name)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        self._save_state()

    def _loop(self):
        while self.running:
            ctx = self.ui_organ.get_context()
            cpu = get_cpu_usage()
            mem = get_mem_usage()
            gpu = get_gpu_usage()

            self.last_cpu = cpu
            self.last_mem = mem
            self.last_gpu = gpu
            self.last_ctx = ctx

            self.node_manager.update_local_load(cpu)
            self.sessions.observe_and_adapt(cpu, mem, gpu, ctx)

            tasks = self.scheduler.get_tasks(ctx, cpu, mem, gpu)

            routed: List[Tuple[str, str, str, str]] = []
            for t in tasks:
                node = self.node_manager.choose_node()
                # For now, only local node is implemented
                organ = self.router.route(t, cpu=cpu, mem=mem, gpu=gpu)
                result = organ.run(t)
                routed.append((t.name, organ.name, organ.symbol, result.get("status", "?")))

            self.last_routed = routed

            print("\n" + "=" * 80)
            if self.sessions.active_session:
                print(f"[SESSION] {self.sessions.active_session.name} "
                      f"| Stability={self.sessions.active_session.stability_score:.2f}")
            else:
                print("[SESSION] None")

            if ctx is not None:
                print(f"[UI] App={ctx.app_name} | Window='{ctx.window_title}'")
            else:
                print("[UI] No context")

            print(f"CPU {cpu:5.1f}% {bar(cpu)}")
            print(f"MEM {mem:5.1f}% {bar(mem)}")
            print(f"GPU {gpu:5.1f}% {bar(gpu)}")
            print("-" * 80)
            for name, organ_name, symbol, status in routed:
                print(f"[TASK] {symbol} {organ_name:7s} <- {name:18s} | status={status}")
            print("=" * 80)

            time.sleep(0.5)

# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    npu_model_path = None          # e.g. "models/npu_model.onnx"
    routing_model_path = None     # e.g. "models/routing_predictor.onnx"

    org = Organism(
        npu_model_path=npu_model_path,
        routing_predictor_model_path=routing_model_path,
    )
    org.start("ModeB_Aggressive")

    gui = CockpitGUI(org)
    print("[ORGANISM] Running. Close GUI or press Ctrl+C in console to stop.")

    try:
        if gui.root is not None:
            gui.run()
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        org.stop()
        print("[ORGANISM] Stopped.")

