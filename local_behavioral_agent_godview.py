"""
local_behavioral_agent_godview.py

Pure local behavioral security agent with:
- Universal autoloader (libs, CUDA, TensorRT, plugins, YAML rules, venv)
- Kernel & user-space sensors
- Telemetry, feature, detection, policy, response, swarm, dashboard services
- Microservice-style architecture (threads + in-memory event bus)
- Advanced Tkinter God-View cockpit (multi-panel UI)
"""

from __future__ import annotations
import importlib
import subprocess
import sys
import os
import glob
import venv
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import threading
import queue
import time
import uuid

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
    "torch",          # optional
    "tensorrt",       # optional
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
    def __init__(self):
        self._topics: Dict[str, "queue.Queue[Any]"] = {}
        self._lock = threading.Lock()

    def _get_topic_queue(self, topic: str) -> "queue.Queue[Any]":
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = queue.Queue()
            return self._topics[topic]

    def publish(self, topic: str, message: Any):
        q = self._get_topic_queue(topic)
        q.put(message)

    def subscribe(self, topic: str) -> "queue.Queue[Any]":
        return self._get_topic_queue(topic)


# =========================
# Sensors (Kernel & User-Space)
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
        print(f"[FeatureExtractor] extract() event={event.id}")
        return FeatureVector(
            event_id=event.id,
            features=[0.0, 1.0, 2.0],
            metadata={"source": event.source, "type": event.event_type},
        )


class FeatureService(Microservice):
    def __init__(self, bus: LocalEventBus, extractor: FeatureExtractor):
        super().__init__("FeatureService", bus)
        self.extractor = extractor
        self.in_q = self.bus.subscribe("telemetry.raw")

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            try:
                event: TelemetryEvent = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            fv = self.extractor.extract(event)
            self.bus.publish("features.ready", fv)
        print(f"[{self.name}] stopped")


# =========================
# Detection Service (Heuristics + CUDA/TensorRT)
# =========================

class HeuristicEngine:
    def score(self, fv: FeatureVector) -> float:
        print(f"[HeuristicEngine] score() event={fv.event_id}")
        return 0.1


class CUDAScoringEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"[CUDAScoringEngine] init model={model_path}")

    def score(self, fv: FeatureVector) -> float:
        print(f"[CUDAScoringEngine] score() event={fv.event_id}")
        return 0.5


class DetectionService(Microservice):
    def __init__(
        self,
        bus: LocalEventBus,
        heuristics: HeuristicEngine,
        cuda_engine: CUDAScoringEngine,
    ):
        super().__init__("DetectionService", bus)
        self.heuristics = heuristics
        self.cuda_engine = cuda_engine
        self.in_q = self.bus.subscribe("features.ready")

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            try:
                fv: FeatureVector = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue

            h_score = self.heuristics.score(fv)
            ml_score = self.cuda_engine.score(fv)
            combined = (h_score + ml_score) / 2.0
            score = AnomalyScore(event_id=fv.event_id, score=combined)
            print(f"[{self.name}] combined score={combined} event={fv.event_id}")
            self.bus.publish("detection.scores", score)
        print(f"[{self.name}] stopped")


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
        print(f"[YAMLRuleEngine] evaluate() event={score.event_id}")
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
# Plugin Marketplace (Auto-loaded)
# =========================

class Plugin:
    def __init__(self, name: str, handler: Callable[[TelemetryEvent], None]):
        self.name = name
        self.handler = handler

    def handle(self, event: TelemetryEvent):
        print(f"[Plugin:{self.name}] handling event {event.id}")
        self.handler(event)


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

    def add_indicator(self, indicator: Dict[str, Any]):
        print(f"[LocalIntelStore] add_indicator {indicator}")
        self.indicators.append(indicator)


class SwarmLocalService(Microservice):
    def __init__(self, bus: LocalEventBus, store: LocalIntelStore):
        super().__init__("SwarmLocalService", bus)
        self.store = store
        self.in_q = self.bus.subscribe("detection.scores")

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            try:
                score: AnomalyScore = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if score.score > 0.9:
                self.store.add_indicator(
                    {"event_id": score.event_id, "score": score.score}
                )
        print(f"[{self.name}] stopped")


# =========================
# Dashboard Backend Service
# =========================

class DashboardBackend:
    def __init__(self):
        self.recent_scores: List[AnomalyScore] = []
        self.recent_decisions: List[PolicyDecision] = []
        self.lock = threading.Lock()

    def push_score(self, score: AnomalyScore):
        with self.lock:
            self.recent_scores.append(score)

    def push_decision(self, decision: PolicyDecision):
        with self.lock:
            self.recent_decisions.append(decision)

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
            }


class DashboardService(Microservice):
    def __init__(self, bus: LocalEventBus, backend: DashboardBackend):
        super().__init__("DashboardService", bus)
        self.backend = backend
        self.score_q = self.bus.subscribe("detection.scores")
        self.decision_q = self.bus.subscribe("policy.decisions")

    def run(self):
        print(f"[{self.name}] run()")
        while self._running.is_set():
            got_any = False
            try:
                score: AnomalyScore = self.score_q.get(timeout=0.1)
                self.backend.push_score(score)
                got_any = True
            except queue.Empty:
                pass
            try:
                decision: PolicyDecision = self.decision_q.get(timeout=0.1)
                self.backend.push_decision(decision)
                got_any = True
            except queue.Empty:
                pass
            if not got_any:
                time.sleep(0.05)
        print(f"[{self.name}] stopped")


# =========================
# Agent Orchestrator
# =========================

class LocalBehavioralAgent:
    def __init__(self, model_path: str):
        self.bus = LocalEventBus()

        kernel_sensors = [EBPFHookSensor(), EndpointSecuritySensor()]
        user_monitors = [ProcessMonitor(), NetflowMonitor(), MemoryScanner(), DLLScanner()]

        self.telemetry = TelemetryService(self.bus, kernel_sensors, user_monitors)
        self.feature_service = FeatureService(self.bus, FeatureExtractor())
        self.detection_service = DetectionService(
            self.bus,
            HeuristicEngine(),
            CUDAScoringEngine(model_path),
        )

        self.rule_engine = YAMLRuleEngine()
        self.rule_engine.load_from_yaml()
        self.policy_service = PolicyService(self.bus, self.rule_engine)

        self.response_service = ResponseService(self.bus, ResponseEngine())
        self.swarm_service = SwarmLocalService(self.bus, LocalIntelStore())
        self.dashboard_backend = DashboardBackend()
        self.dashboard_service = DashboardService(self.bus, self.dashboard_backend)

        self.plugin_marketplace = build_plugins_from_modules()

        self.services: List[Microservice] = [
            self.telemetry,
            self.feature_service,
            self.detection_service,
            self.policy_service,
            self.response_service,
            self.swarm_service,
            self.dashboard_service,
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


# =========================
# Tkinter God-View Cockpit
# =========================

class GodViewUI:
    def __init__(self, root: tk.Tk, agent: LocalBehavioralAgent):
        self.root = root
        self.agent = agent

        self.root.title("Behavioral Security God-View")
        self.root.geometry("1200x700")
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

        self.frame_status = ttk.LabelFrame(self.root, text="Node Status / Logs")
        self.frame_status.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # Threats list
        self.threat_list = tk.Listbox(self.frame_threats, font=("Consolas", 10))
        self.threat_list.pack(fill="both", expand=True)

        # Heatmap placeholder
        self.heatmap_canvas = tk.Canvas(self.frame_heatmap, bg="#101010")
        self.heatmap_canvas.pack(fill="both", expand=True)
        self.heatmap_canvas.create_text(
            10, 10, anchor="nw", fill="#00ff00",
            text="Heatmap placeholder\n(aggregate scores, CPU/network/memory anomalies)",
            font=("Consolas", 10)
        )

        # Timeline
        self.timeline_list = tk.Listbox(self.frame_timeline, font=("Consolas", 10))
        self.timeline_list.pack(fill="both", expand=True)

        # Status / logs
        self.status_text = tk.Text(self.frame_status, font=("Consolas", 10), height=10)
        self.status_text.pack(fill="both", expand=True)
        self.status_text.insert("end", "God-View initialized.\n")

    def _schedule_update(self):
        self._update_ui()
        self.root.after(500, self._schedule_update)

    def _update_ui(self):
        snapshot = self.agent.get_dashboard_snapshot()
        scores: List[AnomalyScore] = snapshot["scores"]
        decisions: List[PolicyDecision] = snapshot["decisions"]

        self.threat_list.delete(0, "end")
        for s in scores[-50:]:
            self.threat_list.insert(
                "end",
                f"{s.event_id[:8]} | score={s.score:.3f}"
            )

        self.timeline_list.delete(0, "end")
        for d in decisions[-50:]:
            actions = ",".join(d.actions) if d.actions else "none"
            self.timeline_list.insert(
                "end",
                f"{d.event_id[:8]} | score={d.score:.3f} | actions={actions}"
            )

        self.heatmap_canvas.delete("all")
        self.heatmap_canvas.create_rectangle(0, 0, 1200, 700, fill="#101010", outline="")
        self.heatmap_canvas.create_text(
            10, 10, anchor="nw", fill="#00ff00",
            text="Heatmap placeholder",
            font=("Consolas", 12, "bold")
        )
        if scores:
            max_score = max(s.score for s in scores)
        else:
            max_score = 0.0
        bar_width = 15
        margin = 5
        for idx, s in enumerate(scores[-40:]):
            x0 = margin + idx * (bar_width + 2)
            y0 = 200
            height = int(150 * (s.score / (max_score + 1e-6)))
            y1 = y0 - height
            color = "#ff0000" if s.score > 0.9 else "#ffaa00" if s.score > 0.7 else "#00ff00"
            self.heatmap_canvas.create_rectangle(x0, y0, x0 + bar_width, y1, fill=color, outline="")

        self.status_text.delete("1.0", "end")
        self.status_text.insert("end", f"Recent events: {len(scores)}\n")
        if scores:
            self.status_text.insert("end", f"Last score: {scores[-1].score:.3f}\n")
        self.status_text.insert("end", f"Decisions: {len(decisions)}\n")
        self.status_text.insert("end", "Agent running.\n")


# =========================
# Bootstrap
# =========================

def main():
    engine_ok = verify_tensorrt_engine("models/tensorrt_engine.trt")
    model_path = "models/tensorrt_engine.trt" if engine_ok else "models/fallback.trt"

    agent = LocalBehavioralAgent(model_path=model_path)
    agent.start()

    for i in range(5):
        agent.inject_test_event(
            source="process_monitor",
            event_type="proc_create",
            payload={"pid": 1000 + i, "exe": f"test{i}.exe"},
        )
        time.sleep(0.2)

    root = tk.Tk()
    ui = GodViewUI(root, agent)

    def on_close():
        agent.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
