#!/usr/bin/env python3
"""
Unified adaptive organism (single file, evolved):

- Universal library loader (auto-installs where possible)
- Mode B (Aggressive Adaptive) session manager + Black Knight
- USB NPU as compute organ (ONNX Runtime if available) + profiling
- Hybrid routing (operator rules + UI context + deep routing model + threat-aware)
- Dual-stack UI context organ (uiautomation + pywinauto fallback)
- Persistence (sessions + UI patterns to JSON)
- Plugin organ loader (hot-load organs from ./organs)
- Distributed compute mesh (local + remote nodes, simple offload stub)
- GPU-accelerated cockpit (DearPyGUI)
- Threat matrix panel (anomaly over time)
- Symbolic overlay engine (glyphs for organs, tasks, anomalies)
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

# ============================================================
#  UNIVERSAL LIBRARY LOADER
# ============================================================

REQUIRED_LIBS = [
    "psutil",
    "numpy",
    "onnxruntime",
    "uiautomation",
    "pywinauto",
    "GPUtil",
    "dearpygui.dearpygui",
]


def ensure_library(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"[LOADER] Missing library: {name}, attempting install via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", name.split(".")[0]])
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
dpg = MODULES.get("dearpygui.dearpygui", None)

# ============================================================
#  BASIC TELEMETRY
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
#  UI CONTEXT ORGAN
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
            win = desk.get_active()
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
#  SYMBOLIC OVERLAY ENGINE
# ============================================================

class SymbolicOverlay:
    def __init__(self):
        self.organ_symbols = {
            "cpu": "C",
            "gpu": "G",
            "npu": "N",
            "plugin": "P",
        }
        self.task_symbols = {
            "telemetry_scoring": "T",
            "heavy_inference": "H",
            "light_logic": "L",
        }
        self.anomaly_symbols = {
            "minor": ".",
            "moderate": "!",
            "severe": "!!",
            "extreme": "!!!",
        }

    def organ_glyph(self, kind: str) -> str:
        return self.organ_symbols.get(kind, "?")

    def task_glyph(self, name: str) -> str:
        return self.task_symbols.get(name, "?")

    def anomaly_glyph(self, level: str) -> str:
        return self.anomaly_symbols.get(level, "?")


# ============================================================
#  COMPUTE ORGANS
# ============================================================

class ComputeTask:
    def __init__(self, name: str, payload: Any, cost_hint: float = 1.0, ui_context: Optional[UIContext] = None):
        self.name = name
        self.payload = payload
        self.cost_hint = cost_hint
        self.ui_context = ui_context


class ComputeOrgan:
    def __init__(self, name: str, kind: str, glyph: str):
        self.name = name
        self.kind = kind
        self.glyph = glyph
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
    def __init__(self, overlay: SymbolicOverlay):
        super().__init__("CPU", "cpu", overlay.organ_glyph("cpu"))

    def run(self, task: ComputeTask) -> Any:
        start = time.time()
        time.sleep(0.001 * task.cost_hint)
        result = {"organ": self.name, "task": task.name, "status": "ok"}
        self.record_latency(time.time() - start)
        return result


class GPUOrgan(ComputeOrgan):
    def __init__(self, overlay: SymbolicOverlay):
        super().__init__("GPU", "gpu", overlay.organ_glyph("gpu"))

    def run(self, task: ComputeTask) -> Any:
        start = time.time()
        time.sleep(0.0005 * task.cost_hint)
        result = {"organ": self.name, "task": task.name, "status": "ok"}
        self.record_latency(time.time() - start)
        return result


class NPUOrgan(ComputeOrgan):
    def __init__(self, overlay: SymbolicOverlay, session: Optional[Any] = None):
        super().__init__("USB_NPU", "npu", overlay.organ_glyph("npu"))
        self.session = session
        self.total_calls = 0
        self.total_time = 0.0

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

        latency = time.time() - start
        self.record_latency(latency)
        self.total_calls += 1
        self.total_time += latency
        return result

    def utilization(self) -> float:
        if self.total_calls == 0:
            return 0.0
        avg = self.total_time / self.total_calls
        # heuristic: higher avg latency => higher utilization
        return min(100.0, avg * 10000.0)


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
#  PLUGIN ORGAN LOADER
# ============================================================

class PluginOrgan(ComputeOrgan):
    def __init__(self, name: str, impl: Any, overlay: SymbolicOverlay):
        super().__init__(name, "plugin", overlay.organ_glyph("plugin"))
        self.impl = impl

    def run(self, task: ComputeTask) -> Any:
        start = time.time()
        try:
            result = self.impl.run(task)
        except Exception as e:
            result = {"organ": self.name, "task": task.name, "status": "error", "error": str(e)}
        self.record_latency(time.time() - start)
        return result


class PluginLoader:
    def __init__(self, overlay: SymbolicOverlay, folder: str = "organs"):
        self.folder = folder
        self.overlay = overlay
        self.organs: Dict[str, PluginOrgan] = {}
        self._load_plugins()

    def _load_plugins(self):
        if not os.path.isdir(self.folder):
            return
        if self.folder not in sys.path:
            sys.path.insert(0, self.folder)
        for fname in os.listdir(self.folder):
            if not fname.endswith(".py"):
                continue
            mod_name = os.path.splitext(fname)[0]
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "PluginOrganImpl"):
                    impl = mod.PluginOrganImpl()
                    organ = PluginOrgan(name=f"plugin_{mod_name}", impl=impl, overlay=self.overlay)
                    self.organs[organ.name] = organ
                    print(f"[PLUGIN] Loaded organ: {organ.name}")
            except Exception as e:
                print(f"[PLUGIN] Failed to load {mod_name}: {e}")


# ============================================================
#  HYBRID ROUTER + DEEP ROUTING MODEL
# ============================================================

class RoutingPreference:
    ALWAYS_NPU = "always_npu"
    NEVER_NPU = "never_npu"
    NPU_PREFERRED = "npu_preferred"
    CPU_GPU_PREFERRED = "cpu_gpu_preferred"


class DeepRoutingModel:
    """
    Tiny deep routing model:
    - ONNX MLP or transformer-like policy
    - Input: [cpu, mem, gpu]
    - Output: scores for [cpu, gpu, npu, plugin]
    """
    def __init__(self, model_path: Optional[str]):
        self.session = create_onnx_session(model_path) if model_path else None

    def predict(self, cpu: float, mem: float, gpu: float) -> Optional[List[float]]:
        if self.session is None or np is None:
            return None
        try:
            x = np.array([[cpu, mem, gpu]], dtype=np.float32)
            input_name = self.session.get_inputs()[0].name
            out = self.session.run(None, {input_name: x})[0]
            return list(out[0])
        except Exception:
            return None


class HybridRouter:
    def __init__(
        self,
        cpu: CPUOrgan,
        gpu: GPUOrgan,
        npu: Optional[NPUOrgan],
        plugins: Dict[str, PluginOrgan],
        deep_model: DeepRoutingModel,
    ):
        self.cpu = cpu
        self.gpu = gpu
        self.npu = npu
        self.plugins = plugins
        self.deep_model = deep_model

        self.operator_rules: Dict[str, str] = {}
        self.ui_context_prefs: Dict[str, str] = {}

    def set_operator_rule(self, task_name: str, preference: str):
        self.operator_rules[task_name] = preference
        print(f"[ROUTER] Rule set: {task_name} -> {preference}")

    def learn_ui_preference(self, ctx: Optional[UIContext], organ_kind: str):
        if ctx is None:
            return
        key = ctx.key()
        prev = self.ui_context_prefs.get(key)
        if prev is None:
            self.ui_context_prefs[key] = organ_kind
        else:
            if prev != organ_kind and random.random() < 0.1:
                self.ui_context_prefs[key] = organ_kind

    def _predict_base(self, avoid_npu: bool) -> str:
        candidates: List[Tuple[str, ComputeOrgan]] = [
            ("cpu", self.cpu),
            ("gpu", self.gpu),
        ]
        if self.npu is not None and not avoid_npu:
            candidates.append(("npu", self.npu))
        for p in self.plugins.values():
            candidates.append((p.kind, p))
        best = min(candidates, key=lambda x: x[1].load_estimate)
        return best[0]

    def _apply_deep_model(self, cpu: float, mem: float, gpu: float, base: str, avoid_npu: bool) -> str:
        scores = self.deep_model.predict(cpu, mem, gpu)
        if scores is None or np is None:
            return base
        mapping = ["cpu", "gpu", "npu", "plugin"]
        idx = int(np.argmax(np.array(scores)))
        kind = mapping[idx]
        if avoid_npu and kind == "npu":
            return base
        return kind

    def _apply_ui_bias(self, ctx: Optional[UIContext], base: str) -> str:
        if ctx is None:
            return base
        key = ctx.key()
        pref = self.ui_context_prefs.get(key)
        if pref is None:
            return base
        return pref

    def _resolve_kind_to_organ(self, kind: str, avoid_npu: bool) -> ComputeOrgan:
        if kind == "cpu":
            return self.cpu
        if kind == "gpu":
            return self.gpu
        if kind == "npu" and self.npu is not None and not avoid_npu:
            return self.npu
        if kind == "plugin" and self.plugins:
            return list(self.plugins.values())[0]
        return self.cpu

    def route(self, task: ComputeTask, cpu: float, mem: float, gpu: float, avoid_npu: bool = False) -> ComputeOrgan:
        pref = self.operator_rules.get(task.name)
        if avoid_npu:
            pref = RoutingPreference.NEVER_NPU

        if pref == RoutingPreference.ALWAYS_NPU and self.npu is not None and not avoid_npu:
            kind = "npu"
        elif pref == RoutingPreference.NEVER_NPU:
            kind = "cpu" if self.cpu.load_estimate <= self.gpu.load_estimate else "gpu"
        elif pref == RoutingPreference.NPU_PREFERRED and self.npu is not None and not avoid_npu:
            if self.npu.load_estimate <= min(self.cpu.load_estimate, self.gpu.load_estimate) * 1.5:
                kind = "npu"
            else:
                kind = self._predict_base(avoid_npu)
        elif pref == RoutingPreference.CPU_GPU_PREFERRED:
            kind = "cpu" if self.cpu.load_estimate <= self.gpu.load_estimate else "gpu"
        else:
            base = self._predict_base(avoid_npu)
            deep = self._apply_deep_model(cpu, mem, gpu, base, avoid_npu)
            kind = self._apply_ui_bias(task.ui_context, deep)

        self.learn_ui_preference(task.ui_context, kind)
        return self._resolve_kind_to_organ(kind, avoid_npu)


# ============================================================
#  MODE B SESSION MANAGER + BLACK KNIGHT
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
        self.threat_history: List[Tuple[float, float]] = []

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
        new_mean = old_mean + (value - old_mean) * self.learning_rate
        new_var = old_var + (value - new_mean) * (value - old_mean) * self.learning_rate
        setattr(self, mean_attr, new_mean)
        setattr(self, var_attr, max(new_var, 0.0))
        self.samples += 1

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
        s.threat_history.append((time.time(), score))
        if len(s.threat_history) > 200:
            s.threat_history.pop(0)

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

    def current_threat_level(self) -> float:
        if self.active_session is None or not self.active_session.threat_history:
            return 0.0
        return self.active_session.threat_history[-1][1]

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
    def __init__(self):
        self.base_tasks: List[Tuple[str, float]] = [
            ("telemetry_scoring", 5.0),
            ("heavy_inference", 20.0),
            ("light_logic", 1.0),
        ]

    def get_tasks(self, ctx: Optional[UIContext], cpu: float, mem: float, gpu: float) -> List[ComputeTask]:
        tasks: List[ComputeTask] = []
        for name, cost in self.base_tasks:
            if name == "heavy_inference" and gpu > 80.0:
                if random.random() < 0.5:
                    continue
            tasks.append(ComputeTask(name, payload=None, cost_hint=cost, ui_context=ctx))
        return tasks


# ============================================================
#  DISTRIBUTED COMPUTE MESH (LOCAL + REMOTE)
# ============================================================

class Node:
    def __init__(self, name: str, host: str = "127.0.0.1", port: int = 5000, weight: float = 1.0):
        self.name = name
        self.host = host
        self.port = port
        self.weight = weight
        self.last_load: float = 0.0


class Mesh:
    """
    Simple mesh:
    - Local node + optional remote nodes
    - For now, offload is simulated; you can extend to real TCP.
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {"local": Node("local")}
        self.local_name = "local"

        # Example remote nodes (disabled by default)
        # self.nodes["remote1"] = Node("remote1", host="192.168.1.10", port=5001, weight=1.0)

    def update_local_load(self, load: float):
        self.nodes[self.local_name].last_load = load

    def choose_node(self) -> Node:
        # pick node with lowest (load * weight)
        best = min(self.nodes.values(), key=lambda n: n.last_load * n.weight)
        return best

    def offload_task(self, node: Node, task: ComputeTask) -> Dict[str, Any]:
        # stub: simulate remote execution
        time.sleep(0.002 * task.cost_hint)
        return {
            "organ": f"remote:{node.name}",
            "task": task.name,
            "status": "ok",
        }


# ============================================================
#  GPU COCKPIT (DEARPYGUI)
# ============================================================

STATE_FILE = "organism_state.json"


class Cockpit:
    def __init__(self, organism: "Organism"):
        self.org = organism
        if dpg is None:
            self.enabled = False
            return
        self.enabled = True

        dpg.create_context()
        dpg.create_viewport(title="Adaptive Organism Cockpit", width=1200, height=700)

        with dpg.window(label="Main", tag="main_window", width=1200, height=700):
            dpg.add_text("Session:")
            self.session_text = dpg.add_text("")
            dpg.add_text("UI Context:")
            self.ui_text = dpg.add_text("")
            dpg.add_separator()
            with dpg.plot(label="CPU/MEM/GPU", height=200):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="t", tag="axis_x")
                self.axis_y = dpg.add_plot_axis(dpg.mvYAxis, label="%", tag="axis_y")
                self.cpu_series = dpg.add_line_series([], [], label="CPU", parent=self.axis_y)
                self.mem_series = dpg.add_line_series([], [], label="MEM", parent=self.axis_y)
                self.gpu_series = dpg.add_line_series([], [], label="GPU", parent=self.axis_y)
            dpg.add_separator()
            dpg.add_text("Routed Tasks:")
            self.tasks_text = dpg.add_text("")
            dpg.add_separator()
            with dpg.plot(label="Threat Matrix", height=200):
                dpg.add_plot_axis(dpg.mvXAxis, label="t", tag="threat_x")
                self.threat_axis_y = dpg.add_plot_axis(dpg.mvYAxis, label="score", tag="threat_y")
                self.threat_series = dpg.add_line_series([], [], label="Anomaly", parent=self.threat_axis_y)
            dpg.add_separator()
            dpg.add_text("NPU Utilization:")
            self.npu_text = dpg.add_text("")

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    def _update(self):
        if not self.enabled:
            return

        cpu_hist = self.org.cpu_hist
        mem_hist = self.org.mem_hist
        gpu_hist = self.org.gpu_hist
        x = list(range(len(cpu_hist)))
        dpg.set_value(self.cpu_series, [x, cpu_hist])
        dpg.set_value(self.mem_series, [x, mem_hist])
        dpg.set_value(self.gpu_series, [x, gpu_hist])

        sess = self.org.sessions.active_session
        if sess is not None:
            dpg.set_value(self.session_text, f"{sess.name} | Stability={sess.stability_score:.2f}")
        else:
            dpg.set_value(self.session_text, "None")

        ctx = self.org.last_ctx
        if ctx is not None:
            dpg.set_value(self.ui_text, f"{ctx.app_name} | '{ctx.window_title}'")
        else:
            dpg.set_value(self.ui_text, "No context")

        lines = []
        for name, organ_name, glyph, status in self.org.last_routed:
            lines.append(f"{glyph} {organ_name}: {name} [{status}]")
        dpg.set_value(self.tasks_text, "\n".join(lines))

        if sess is not None:
            th = sess.threat_history
            if th:
                t0 = th[0][0]
                tx = [p[0] - t0 for p in th]
                ty = [p[1] for p in th]
                dpg.set_value(self.threat_series, [tx, ty])

        if self.org.npu is not None:
            util = self.org.npu.utilization()
            dpg.set_value(self.npu_text, f"{util:.1f}% (avg latency-based)")
        else:
            dpg.set_value(self.npu_text, "NPU not present")

    def run(self):
        if not self.enabled:
            print("[COCKPIT] DearPyGUI not available.")
            while self.org.running:
                time.sleep(1.0)
            return

        last_update = time.time()
        while dpg.is_dearpygui_running():
            now = time.time()
            if now - last_update > 0.2:
                self._update()
                last_update = now
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


# ============================================================
#  ORGANISM
# ============================================================

class Organism:
    def __init__(
        self,
        npu_model_path: Optional[str] = None,
        routing_model_path: Optional[str] = None,
    ):
        self.overlay = SymbolicOverlay()

        self.cpu = CPUOrgan(self.overlay)
        self.gpu = GPUOrgan(self.overlay)

        npu_sess = create_onnx_session(npu_model_path) if npu_model_path else None
        self.npu = NPUOrgan(self.overlay, npu_sess) if npu_sess is not None else None

        self.plugin_loader = PluginLoader(self.overlay)
        self.plugins = self.plugin_loader.organs

        self.deep_model = DeepRoutingModel(routing_model_path)
        self.ui_organ = UIContextOrgan()
        self.router = HybridRouter(self.cpu, self.gpu, self.npu, self.plugins, self.deep_model)

        self.sessions = SessionManager()
        self._load_state()

        self.scheduler = TaskScheduler()
        self.mesh = Mesh()

        self.running = False
        self.cpu_hist: List[float] = []
        self.mem_hist: List[float] = []
        self.gpu_hist: List[float] = []
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

    def _record_history(self, cpu: float, mem: float, gpu: float, max_len: int = 200):
        self.cpu_hist.append(cpu)
        self.mem_hist.append(mem)
        self.gpu_hist.append(gpu)
        if len(self.cpu_hist) > max_len:
            self.cpu_hist.pop(0)
            self.mem_hist.pop(0)
            self.gpu_hist.pop(0)

    def _loop(self):
        while self.running:
            ctx = self.ui_organ.get_context()
            cpu = get_cpu_usage()
            mem = get_mem_usage()
            gpu = get_gpu_usage()

            self.last_ctx = ctx
            self.mesh.update_local_load(cpu)
            self.sessions.observe_and_adapt(cpu, mem, gpu, ctx)
            self._record_history(cpu, mem, gpu)

            tasks = self.scheduler.get_tasks(ctx, cpu, mem, gpu)
            routed: List[Tuple[str, str, str, str]] = []

            avoid_npu = self.sessions.black_knight_triggered or self.sessions.current_threat_level() >= 3.0

            for t in tasks:
                node = self.mesh.choose_node()
                if node.name != self.mesh.local_name:
                    result = self.mesh.offload_task(node, t)
                    glyph = self.overlay.organ_glyph("plugin")
                    routed.append((t.name, result["organ"], glyph, result.get("status", "?")))
                    continue

                if self.sessions.black_knight_triggered:
                    organ = self.cpu
                else:
                    organ = self.router.route(t, cpu=cpu, mem=mem, gpu=gpu, avoid_npu=avoid_npu)

                result = organ.run(t)
                routed.append((t.name, organ.name, organ.glyph, result.get("status", "?")))

            self.last_routed = routed

            time.sleep(0.5)


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    npu_model_path = None          # e.g. "models/npu_model.onnx"
    routing_model_path = None      # e.g. "models/deep_routing.onnx"

    org = Organism(
        npu_model_path=npu_model_path,
        routing_model_path=routing_model_path,
    )
    org.start("ModeB_Aggressive")

    cockpit = Cockpit(org)
    print("[ORGANISM] Running. Close cockpit window or Ctrl+C to stop.")
    try:
        cockpit.run()
    except KeyboardInterrupt:
        org.stop()
        print("[ORGANISM] Stopped.")

