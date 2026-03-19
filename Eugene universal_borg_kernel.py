"""
borg_kernel_omega.py

Universal user-space microkernel organism with:

- IPC fabric
- Scheduler with task classes and background throttling
- Organs:
  - Cortex (policy brain, proposal handling)
  - AI policy organ (proposal generator)
  - Telemetry organ (host metrics)
  - Constraint organ (negative-space rules with real throttling, DSL-driven)
  - Prediction organ (simple forecasting)
  - Process watcher organ (new process detection)
  - Federation organ (multi-node topology exchange, secure)
  - GPU inference organ (stubbed, pluggable)
  - UIAutomation organ (stubbed, pluggable)
- Cluster topology (multi-node)
- Persistent state + rich reflection episodes
- Organ restart logic (self-healing)
- Distributed scheduling hooks (topology-aware hints)
- Constraint DSL
- PyQt5 cockpit GUI:
  - Status (organs, throttling, restart counts)
  - Episodes
  - Telemetry
  - Processes
  - Topology
  - Controls (toggle organs, adjust CPU ceiling)
"""

from __future__ import annotations
import abc
import enum
import json
import os
import platform
import queue
import socket
import threading
import time
import uuid
import hmac
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import psutil
from PyQt5 import QtCore, QtWidgets


# =========================
# Core enums and constants
# =========================

class TaskClass(enum.Enum):
    REALTIME = "realtime"
    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    MAINTENANCE = "maintenance"


class BackpressurePolicy(enum.Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEW = "drop_new"
    THROTTLE_SENDER = "throttle_sender"
    ESCALATE = "escalate"


class MessageDirection(enum.Enum):
    SEND = "send"
    RECV = "recv"
    BOTH = "both"


# =========================
# IPC model
# =========================

@dataclass
class Capability:
    channel: str
    direction: MessageDirection
    types: Set[str]
    rate_limit_per_sec: Optional[float] = None


@dataclass
class Message:
    id: str
    src: str
    dst: str
    type: str
    ts: float
    priority: int
    payload: Dict[str, Any]
    trace: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass
class Endpoint:
    endpoint_id: str
    owner: str  # organ name or "kernel"
    capabilities: List[Capability]
    subscribed_channels: Set[str] = field(default_factory=set)

    def can_send(self, channel: str, msg_type: str) -> bool:
        for cap in self.capabilities:
            if cap.channel == channel and cap.direction in (MessageDirection.SEND, MessageDirection.BOTH):
                if msg_type in cap.types or "*" in cap.types:
                    return True
        return False

    def can_recv(self, channel: str, msg_type: str) -> bool:
        for cap in self.capabilities:
            if cap.channel == channel and cap.direction in (MessageDirection.RECV, MessageDirection.BOTH):
                if msg_type in cap.types or "*" in cap.types:
                    return True
        return False


@dataclass
class ChannelConfig:
    name: str
    backpressure_policy: BackpressurePolicy = BackpressurePolicy.THROTTLE_SENDER
    max_queue_depth: int = 1000


class IPCFabric:
    """
    Central IPC fabric: channels, endpoints, routing, backpressure.
    """

    def __init__(self):
        self._channels: Dict[str, ChannelConfig] = {}
        self._endpoints: Dict[str, Endpoint] = {}
        self._subscriptions: Dict[str, Set[str]] = {}  # channel -> endpoint_ids
        self._ingress_queues: Dict[str, "queue.Queue[Message]"] = {}
        self._lock = threading.Lock()

    def register_channel(self, config: ChannelConfig) -> None:
        with self._lock:
            if config.name not in self._channels:
                self._channels[config.name] = config
                self._subscriptions[config.name] = set()

    def register_endpoint(self, endpoint: Endpoint) -> None:
        with self._lock:
            self._endpoints[endpoint.endpoint_id] = endpoint
            if endpoint.endpoint_id not in self._ingress_queues:
                self._ingress_queues[endpoint.endpoint_id] = queue.Queue()

    def subscribe(self, endpoint_id: str, channel: str) -> None:
        with self._lock:
            if channel not in self._channels:
                raise ValueError(f"Channel {channel} not registered")
            ep = self._endpoints[endpoint_id]
            ep.subscribed_channels.add(channel)
            self._subscriptions[channel].add(endpoint_id)

    def send(self, msg: Message) -> List[str]:
        """
        Route a message to all subscribed endpoints on the dst channel.
        Returns list of endpoint_ids that received the message.
        """
        with self._lock:
            if msg.dst not in self._channels:
                raise ValueError(f"Destination channel {msg.dst} not registered")

            recipients = list(self._subscriptions.get(msg.dst, []))
            for ep_id in recipients:
                q = self._ingress_queues[ep_id]
                cfg = self._channels[msg.dst]
                if q.qsize() >= cfg.max_queue_depth:
                    if cfg.backpressure_policy == BackpressurePolicy.DROP_NEW:
                        continue
                    elif cfg.backpressure_policy == BackpressurePolicy.DROP_OLDEST:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                    elif cfg.backpressure_policy == BackpressurePolicy.THROTTLE_SENDER:
                        pass
                    elif cfg.backpressure_policy == BackpressurePolicy.ESCALATE:
                        pass
                q.put(msg)
            return recipients

    def recv(self, endpoint_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        q = self._ingress_queues[endpoint_id]
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None


# =========================
# Scheduler
# =========================

@dataclass(order=True)
class Task:
    sort_index: int = field(init=False, repr=False)
    task_id: str
    owner: str
    cls: TaskClass
    base_priority: int
    dynamic_priority: int
    deadline: Optional[float]
    created_at: float
    func: Callable[[], None]

    def __post_init__(self):
        effective_priority = -(self.base_priority + self.dynamic_priority)
        self.sort_index = (self.cls_order(), effective_priority, self.created_at)

    def cls_order(self) -> int:
        order = {
            TaskClass.REALTIME: 0,
            TaskClass.INTERACTIVE: 1,
            TaskClass.BACKGROUND: 2,
            TaskClass.MAINTENANCE: 3,
        }
        return order[self.cls]


class Scheduler:
    """
    Simple priority-based scheduler with task classes.
    """

    def __init__(self, worker_threads: int = 4):
        self._tasks: "queue.PriorityQueue[Task]" = queue.PriorityQueue()
        self._stop = threading.Event()
        self._workers: List[threading.Thread] = []
        for i in range(worker_threads):
            t = threading.Thread(target=self._worker_loop, name=f"sched-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def submit(self, task: Task) -> None:
        self._tasks.put(task)

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                task = self._tasks.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                task.func()
            except Exception as e:
                print(f"[Scheduler] Task {task.task_id} failed: {e}")
            finally:
                self._tasks.task_done()

    def shutdown(self):
        self._stop.set()
        for t in self._workers:
            t.join(timeout=1.0)


# =========================
# Topology model (borg)
# =========================

@dataclass
class NodeCapabilities:
    cpu_cores: int
    memory_mb: int
    has_gpu: bool
    os_type: str
    zone: str = "default"


@dataclass
class NodeInfo:
    node_id: str
    capabilities: NodeCapabilities
    organs: Set[str] = field(default_factory=set)
    healthy: bool = True
    degraded: bool = False


@dataclass
class LinkInfo:
    src: str
    dst: str
    latency_ms: float
    bandwidth_mbps: float


class ClusterTopology:
    """
    Maintains cluster-wide view of nodes and links.
    """

    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.links: Dict[Tuple[str, str], LinkInfo] = {}
        self._lock = threading.Lock()

    def upsert_node(self, node: NodeInfo) -> None:
        with self._lock:
            self.nodes[node.node_id] = node

    def mark_node_health(self, node_id: str, healthy: bool, degraded: bool = False) -> None:
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].healthy = healthy
                self.nodes[node_id].degraded = degraded

    def upsert_link(self, link: LinkInfo) -> None:
        with self._lock:
            self.links[(link.src, link.dst)] = link

    def get_neighbors(self, node_id: str) -> List[NodeInfo]:
        with self._lock:
            neighbors = []
            for (src, dst), link in self.links.items():
                if src == node_id and dst in self.nodes:
                    neighbors.append(self.nodes[dst])
            return neighbors


# =========================
# AI proposal API
# =========================

class ProposalType(enum.Enum):
    ADJUST_PRIORITY = "adjust_priority"
    CHANGE_CLASS = "change_class"
    THROTTLE_CHANNEL = "throttle_channel"
    MIGRATE_ORGAN = "migrate_organ"
    UPDATE_CONSTRAINT = "update_constraint"


@dataclass
class AIProposal:
    id: str
    ts: float
    source_organ: str
    proposal_type: ProposalType
    payload: Dict[str, Any]
    confidence: float
    rationale: str


# =========================
# Persistent state & reflection
# =========================

@dataclass
class ReflectionEpisode:
    """
    One learning episode: what happened, what we did, what we learned.
    """

    id: str
    ts_start: float
    ts_end: float

    context: Dict[str, Any]
    trigger: str
    severity: str
    tags: List[str]

    signals: Dict[str, Any]
    constraints_active: Dict[str, Any]

    decision: Dict[str, Any]
    alternatives: List[Dict[str, Any]]

    outcome: Dict[str, Any]
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]

    lessons: List[str]
    followups: List[Dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class PersistentStore:
    """
    JSONL-based persistence for state and reflection episodes.
    """

    def __init__(self, state_path: str = "kernel_state.json", episodes_path: str = "episodes.jsonl"):
        self.state_path = state_path
        self.episodes_path = episodes_path
        self._state_lock = threading.Lock()
        self._episodes_lock = threading.Lock()
        self.state: Dict[str, Any] = {}
        self.episodes_in_memory: List[ReflectionEpisode] = []

    def load_state(self) -> None:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        except FileNotFoundError:
            self.state = {}

    def save_state(self) -> None:
        with self._state_lock:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)

    def append_episode(self, episode: ReflectionEpisode) -> None:
        with self._episodes_lock:
            self.episodes_in_memory.append(episode)
            with open(self.episodes_path, "a", encoding="utf-8") as f:
                f.write(episode.to_json() + "\n")


# =========================
# Constraint DSL
# =========================

class ConstraintOp(enum.Enum):
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="
    EQ = "=="


@dataclass
class ConstraintRule:
    name: str
    metric: str
    op: ConstraintOp
    threshold: float
    target: str  # e.g. "background_organs"
    action: str  # e.g. "throttle", "unthrottle"


class ConstraintDSL:
    """
    Very simple DSL: rules over metrics -> actions.
    """

    def __init__(self):
        self.rules: List[ConstraintRule] = []

    def add_rule(self, rule: ConstraintRule) -> None:
        self.rules.append(rule)

    def eval(self, metrics: Dict[str, Any]) -> List[ConstraintRule]:
        fired: List[ConstraintRule] = []
        for r in self.rules:
            val = metrics.get(r.metric)
            if val is None:
                continue
            if self._check(r.op, float(val), r.threshold):
                fired.append(r)
        return fired

    @staticmethod
    def _check(op: ConstraintOp, v: float, t: float) -> bool:
        if op == ConstraintOp.LT:
            return v < t
        if op == ConstraintOp.GT:
            return v > t
        if op == ConstraintOp.LTE:
            return v <= t
        if op == ConstraintOp.GTE:
            return v >= t
        if op == ConstraintOp.EQ:
            return v == t
        return False


# =========================
# Organ model + autoloader
# =========================

class Organ(abc.ABC):
    """
    Base class for all organs.
    """

    def __init__(self, name: str, kernel: "Kernel"):
        self.name = name
        self.kernel = kernel
        self.running = False

    @abc.abstractmethod
    def manifest(self) -> Dict[str, Any]:
        raise NotImplementedError

    def init(self) -> None:
        pass

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def tick(self) -> None:
        pass


class OrganRegistry:
    """
    In-process registry for organs.
    """

    def __init__(self):
        self._factories: Dict[str, Callable[["Kernel"], Organ]] = {}

    def register(self, name: str, factory: Callable[["Kernel"], Organ]) -> None:
        self._factories[name] = factory

    def create(self, name: str, kernel: "Kernel") -> Organ:
        return self._factories[name](kernel)

    def list_organs(self) -> List[str]:
        return list(self._factories.keys())


# =========================
# Cortex
# =========================

class Cortex(Organ):
    """
    Policy brain: consumes AI proposals, telemetry, and topology to steer the kernel.
    """

    def __init__(self, kernel: "Kernel"):
        super().__init__("cortex", kernel)
        self._proposal_queue: "queue.Queue[AIProposal]" = queue.Queue()

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tick_interval_ms": 200,
        }

    def submit_proposal(self, proposal: AIProposal) -> None:
        self._proposal_queue.put(proposal)

    def tick(self) -> None:
        while not self._proposal_queue.empty():
            proposal = self._proposal_queue.get()
            self._handle_proposal(proposal)

    def _handle_proposal(self, proposal: AIProposal) -> None:
        print(f"[Cortex] Proposal {proposal.proposal_type.value} from {proposal.source_organ} "
              f"conf={proposal.confidence:.2f}: {proposal.rationale}")

        ctx = {
            "node_id": self.kernel.node_id,
            "ts_start": proposal.ts,
            "proposal_id": proposal.id,
        }
        episode = ReflectionEpisode(
            id=str(uuid.uuid4()),
            ts_start=proposal.ts,
            ts_end=time.time(),
            context=ctx,
            trigger="ai_proposal",
            severity="info",
            tags=["ai", "proposal"],
            signals={"payload": proposal.payload},
            constraints_active={},
            decision={"applied": False},
            alternatives=[],
            outcome={"status": "ignored_for_now"},
            metrics_before={},
            metrics_after={},
            lessons=[f"Observed proposal {proposal.proposal_type.value} conf={proposal.confidence}"],
            followups=[],
        )
        self.kernel.store.append_episode(episode)


# =========================
# AI policy organ
# =========================

class SimpleAIPolicyOrgan(Organ):
    """
    Simple AI organ that periodically emits dummy proposals.
    """

    def __init__(self, name: str, kernel: "Kernel"):
        super().__init__(name, kernel)

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tick_interval_ms": 1000,
        }

    def tick(self) -> None:
        proposal = AIProposal(
            id=str(uuid.uuid4()),
            ts=time.time(),
            source_organ=self.name,
            proposal_type=ProposalType.ADJUST_PRIORITY,
            payload={"owner": "telemetry", "delta": 1},
            confidence=0.6,
            rationale="Heuristic: telemetry tasks appear slightly delayed under current load.",
        )
        self.kernel.cortex.submit_proposal(proposal)


# =========================
# Telemetry organ
# =========================

class TelemetryOrgan(Organ):
    """
    Collects real host telemetry and feeds it into:
    - IPC channels
    - Reflection episodes
    - Kernel.last_telemetry (for GUI)
    """

    def __init__(self, name: str, kernel: "Kernel"):
        super().__init__(name, kernel)

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tick_interval_ms": 1500,
        }

    def tick(self) -> None:
        snapshot = self._collect_snapshot()
        self.kernel.last_telemetry = snapshot

        msg = Message(
            id=str(uuid.uuid4()),
            src=self.name,
            dst="kernel.events",
            type="telemetry.snapshot",
            ts=time.time(),
            priority=1,
            payload=snapshot,
        )
        self.kernel.ipc.send(msg)

        self.kernel.record_episode(
            trigger="telemetry_tick",
            severity="info",
            tags=["telemetry", "snapshot"],
            context={"node_id": self.kernel.node_id, "ts_start": time.time()},
            signals=snapshot,
            constraints_active={},
            decision={"action": "collect_snapshot"},
            alternatives=[],
            outcome={"status": "ok"},
            metrics_before={},
            metrics_after={},
            lessons=["Telemetry snapshot collected"],
            followups=[]
        )

    def _collect_snapshot(self) -> Dict[str, Any]:
        try:
            mem = psutil.virtual_memory()
            processes = list(psutil.process_iter())
            return {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": mem.percent,
                "memory_used_mb": mem.used / (1024 * 1024),
                "memory_total_mb": mem.total / (1024 * 1024),
                "process_count": len(processes),
                "thread_count": sum(p.num_threads() for p in processes),
                "platform": platform.system(),
                "platform_release": platform.release(),
                "hostname": platform.node(),
                "pid": os.getpid(),
                "uptime_sec": time.time() - psutil.boot_time(),
            }
        except Exception as e:
            print(f"[Telemetry] Error collecting snapshot: {e}")
            return {"error": str(e), "timestamp": time.time()}


# =========================
# Constraint organ (DSL + throttling)
# =========================

class ConstraintOrgan(Organ):
    """
    Encodes negative-space rules via DSL and throttles background organs.
    """

    def __init__(self, name: str, kernel: "Kernel"):
        super().__init__(name, kernel)

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tick_interval_ms": 2000,
        }

    def tick(self) -> None:
        snap = self.kernel.last_telemetry or {}
        fired = self.kernel.constraint_dsl.eval(snap)
        for rule in fired:
            if rule.target == "background_organs":
                if rule.action == "throttle" and not self.kernel.throttle_background:
                    self.kernel.throttle_background = True
                    self._record(rule, snap, "throttle_background_on")
                elif rule.action == "unthrottle" and self.kernel.throttle_background:
                    self.kernel.throttle_background = False
                    self._record(rule, snap, "throttle_background_off")

    def _record(self, rule: ConstraintRule, snap: Dict[str, Any], action: str) -> None:
        ctx = {
            "node_id": self.kernel.node_id,
            "ts_start": time.time(),
            "rule": rule.name,
        }
        self.kernel.record_episode(
            trigger="constraint_rule",
            severity="warning" if "throttle" in action else "info",
            tags=["constraint", rule.name],
            context=ctx,
            signals=snap,
            constraints_active={"rule": rule.name},
            decision={"action": action},
            alternatives=[],
            outcome={"status": action},
            metrics_before={"cpu_percent": snap.get("cpu_percent")},
            metrics_after={},
            lessons=[f"Constraint {rule.name} fired, action={action}"],
            followups=[]
        )


# =========================
# Prediction organ
# =========================

class PredictionOrgan(Organ):
    """
    Simple CPU prediction using exponential smoothing.
    """

    def __init__(self, name: str, kernel: "Kernel", alpha: float = 0.3):
        super().__init__(name, kernel)
        self.alpha = alpha
        self.pred_cpu: Optional[float] = None

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tick_interval_ms": 2500,
        }

    def tick(self) -> None:
        snap = self.kernel.last_telemetry or {}
        cpu = snap.get("cpu_percent")
        if cpu is None:
            return

        if self.pred_cpu is None:
            self.pred_cpu = cpu
        else:
            self.pred_cpu = self.alpha * cpu + (1 - self.alpha) * self.pred_cpu

        self.kernel.last_prediction = {"cpu_percent_pred": self.pred_cpu, "timestamp": time.time()}

        self.kernel.record_episode(
            trigger="prediction_tick",
            severity="info",
            tags=["prediction", "cpu"],
            context={"node_id": self.kernel.node_id, "ts_start": time.time()},
            signals={"cpu_current": cpu, "cpu_pred": self.pred_cpu},
            constraints_active={},
            decision={"action": "update_prediction"},
            alternatives=[],
            outcome={"status": "ok"},
            metrics_before={},
            metrics_after={},
            lessons=[f"Updated CPU prediction to {self.pred_cpu:.2f}%"],
            followups=[]
        )


# =========================
# Process watcher organ
# =========================

class ProcessWatcherOrgan(Organ):
    """
    Watches for new processes and logs episodes.
    """

    def __init__(self, name: str, kernel: "Kernel"):
        super().__init__(name, kernel)
        self.known_pids: Set[int] = set()

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tick_interval_ms": 3000,
        }

    def tick(self) -> None:
        try:
            current_pids = set(psutil.pids())
            new_pids = current_pids - self.known_pids
            self.known_pids = current_pids

            processes_info = []
            for pid in list(current_pids)[:128]:
                try:
                    p = psutil.Process(pid)
                    processes_info.append({"pid": pid, "name": p.name(), "cmdline": " ".join(p.cmdline()[:4])})
                except Exception:
                    continue
            self.kernel.last_processes = processes_info

            for pid in new_pids:
                try:
                    p = psutil.Process(pid)
                    info = {"pid": pid, "name": p.name(), "cmdline": " ".join(p.cmdline()[:4])}
                except Exception:
                    info = {"pid": pid}
                self.kernel.record_episode(
                    trigger="new_process",
                    severity="info",
                    tags=["process", "spawn"],
                    context={"node_id": self.kernel.node_id, "ts_start": time.time()},
                    signals=info,
                    constraints_active={},
                    decision={"action": "observe_new_process"},
                    alternatives=[],
                    outcome={"status": "seen"},
                    metrics_before={},
                    metrics_after={},
                    lessons=[f"New process observed: {info}"],
                    followups=[]
                )
        except Exception as e:
            print(f"[ProcessWatcher] Error: {e}")


# =========================
# GPU inference organ (stub)
# =========================

class GPUInferenceOrgan(Organ):
    """
    Stub GPU inference organ. Hooks for future ML models.
    """

    def __init__(self, name: str, kernel: "Kernel"):
        super().__init__(name, kernel)

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tick_interval_ms": 5000,
        }

    def tick(self) -> None:
        # Placeholder: pretend we ran a GPU model and produced a signal.
        snap = self.kernel.last_telemetry or {}
        if not snap:
            return
        signal = {
            "ts": time.time(),
            "note": "GPU inference stub executed",
        }
        self.kernel.record_episode(
            trigger="gpu_inference",
            severity="info",
            tags=["gpu", "inference"],
            context={"node_id": self.kernel.node_id, "ts_start": time.time()},
            signals=signal,
            constraints_active={},
            decision={"action": "noop"},
            alternatives=[],
            outcome={"status": "ok"},
            metrics_before={},
            metrics_after={},
            lessons=["GPU inference organ stub ran"],
            followups=[]
        )


# =========================
# UIAutomation organ (stub)
# =========================

class UIAutomationOrgan(Organ):
    """
    Stub UIAutomation organ. Hooks for future OS-specific automation.
    """

    def __init__(self, name: str, kernel: "Kernel"):
        super().__init__(name, kernel)

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tick_interval_ms": 7000,
        }

    def tick(self) -> None:
        # Placeholder: would inspect windows, apps, etc.
        self.kernel.record_episode(
            trigger="uiautomation_tick",
            severity="info",
            tags=["uiautomation"],
            context={"node_id": self.kernel.node_id, "ts_start": time.time()},
            signals={"note": "UIAutomation stub tick"},
            constraints_active={},
            decision={"action": "noop"},
            alternatives=[],
            outcome={"status": "ok"},
            metrics_before={},
            metrics_after={},
            lessons=["UIAutomation organ stub ticked"],
            followups=[]
        )


# =========================
# Federation organ (secure, multi-node)
# =========================

class FederationOrgan(Organ):
    """
    Multi-node federation with simple HMAC authentication.
    """

    def __init__(self, name: str, kernel: "Kernel", listen_port: int = 50050):
        super().__init__(name, kernel)
        self.listen_port = listen_port
        self.server_thread: Optional[threading.Thread] = None
        self._stop_server = threading.Event()

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tick_interval_ms": 5000,
        }

    def init(self) -> None:
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

    def stop(self) -> None:
        super().stop()
        self._stop_server.set()

    def _sign(self, data: bytes) -> str:
        key = self.kernel.federation_secret
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def _verify(self, data: bytes, sig: str) -> bool:
        key = self.kernel.federation_secret
        expected = hmac.new(key, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)

    def _server_loop(self) -> None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", self.listen_port))
            s.listen(5)
            print(f"[Federation] Listening on port {self.listen_port}")
            s.settimeout(1.0)
            while not self._stop_server.is_set():
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
        except Exception as e:
            print(f"[Federation] Server error: {e}")

    def _handle_client(self, conn: socket.socket, addr) -> None:
        try:
            data = b""
            conn.settimeout(2.0)
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            if not data:
                return
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception:
                return
            sig = payload.get("sig")
            body = payload.get("body")
            if sig is None or body is None:
                return
            body_bytes = json.dumps(body).encode("utf-8")
            if not self._verify(body_bytes, sig):
                print("[Federation] Invalid signature from", addr)
                return
            self._apply_remote_topology(body)
        finally:
            conn.close()

    def _apply_remote_topology(self, body: Dict[str, Any]) -> None:
        try:
            node_id = body.get("node_id")
            if not node_id:
                return
            caps = body.get("capabilities", {})
            organs = set(body.get("organs", []))
            node_caps = NodeCapabilities(
                cpu_cores=caps.get("cpu_cores", 0),
                memory_mb=caps.get("memory_mb", 0),
                has_gpu=caps.get("has_gpu", False),
                os_type=caps.get("os_type", "unknown"),
                zone=caps.get("zone", "remote"),
            )
            node = NodeInfo(node_id=node_id, capabilities=node_caps, organs=organs)
            self.kernel.topology.upsert_node(node)
        except Exception as e:
            print(f"[Federation] Error applying remote topology: {e}")

    def tick(self) -> None:
        topo = self.kernel.topology.nodes.get(self.kernel.node_id)
        if not topo:
            return
        body = {
            "node_id": topo.node_id,
            "capabilities": {
                "cpu_cores": topo.capabilities.cpu_cores,
                "memory_mb": topo.capabilities.memory_mb,
                "has_gpu": topo.capabilities.has_gpu,
                "os_type": topo.capabilities.os_type,
                "zone": topo.capabilities.zone,
            },
            "organs": list(topo.organs),
        }
        body_bytes = json.dumps(body).encode("utf-8")
        sig = self._sign(body_bytes)
        payload = {"body": body, "sig": sig}
        data = json.dumps(payload).encode("utf-8")
        for host, port in self.kernel.federation_peers:
            try:
                with socket.create_connection((host, port), timeout=1.0) as c:
                    c.sendall(data)
            except Exception:
                continue


# =========================
# Kernel (microkernel core)
# =========================

class Kernel:
    """
    The microkernel organism: IPC, scheduler, organs, cortex, topology, persistence.
    """

    def __init__(self,
                 node_id: str,
                 federation_peers: Optional[List[Tuple[str, int]]] = None,
                 listen_port: int = 50050,
                 federation_secret: bytes = b"shared-secret"):
        self.node_id = node_id
        self.ipc = IPCFabric()
        self.scheduler = Scheduler(worker_threads=4)
        self.topology = ClusterTopology()
        self.store = PersistentStore()
        self.organ_registry = OrganRegistry()
        self.organs: Dict[str, Organ] = {}
        self.cortex = Cortex(self)

        self.last_telemetry: Dict[str, Any] = {}
        self.last_prediction: Dict[str, Any] = {}
        self.last_processes: List[Dict[str, Any]] = []

        self.throttle_background: bool = False
        self.background_organs: Set[str] = set()
        self.federation_peers: List[Tuple[str, int]] = federation_peers or []
        self.listen_port = listen_port
        self.federation_secret = federation_secret

        self.constraint_dsl = ConstraintDSL()
        self._init_default_constraints()

        self.organ_restart_counts: Dict[str, int] = {}

        self.organ_registry.register("cortex", lambda k: self.cortex)
        self.organ_registry.register("ai_policy", lambda k: SimpleAIPolicyOrgan("ai_policy", k))
        self.organ_registry.register("telemetry", lambda k: TelemetryOrgan("telemetry", k))
        self.organ_registry.register("constraints", lambda k: ConstraintOrgan("constraints", k))
        self.organ_registry.register("prediction", lambda k: PredictionOrgan("prediction", k))
        self.organ_registry.register("process_watcher", lambda k: ProcessWatcherOrgan("process_watcher", k))
        self.organ_registry.register("gpu_inference", lambda k: GPUInferenceOrgan("gpu_inference", k))
        self.organ_registry.register("uiautomation", lambda k: UIAutomationOrgan("uiautomation", k))
        self.organ_registry.register("federation", lambda k: FederationOrgan("federation", k, listen_port=self.listen_port))

        self._register_core_channels()
        self.store.load_state()
        self._init_topology_self_node()

    def _init_default_constraints(self) -> None:
        # Example: throttle background when CPU > 80, unthrottle when CPU < 60
        self.constraint_dsl.add_rule(
            ConstraintRule(
                name="cpu_high_throttle",
                metric="cpu_percent",
                op=ConstraintOp.GT,
                threshold=80.0,
                target="background_organs",
                action="throttle",
            )
        )
        self.constraint_dsl.add_rule(
            ConstraintRule(
                name="cpu_low_unthrottle",
                metric="cpu_percent",
                op=ConstraintOp.LT,
                threshold=60.0,
                target="background_organs",
                action="unthrottle",
            )
        )

    def _register_core_channels(self) -> None:
        self.ipc.register_channel(ChannelConfig(name="cluster.control"))
        self.ipc.register_channel(ChannelConfig(name="cluster.topology"))
        self.ipc.register_channel(ChannelConfig(name="cluster.organs"))
        self.ipc.register_channel(ChannelConfig(name="cluster.tasks"))
        self.ipc.register_channel(ChannelConfig(name="kernel.events"))

    def _init_topology_self_node(self) -> None:
        mem = psutil.virtual_memory()
        caps = NodeCapabilities(
            cpu_cores=psutil.cpu_count(logical=True) or 0,
            memory_mb=int(mem.total / (1024 * 1024)),
            has_gpu=False,
            os_type=platform.system(),
            zone="local",
        )
        node = NodeInfo(node_id=self.node_id, capabilities=caps)
        self.topology.upsert_node(node)

    def autoload_organs(self) -> None:
        for name in self.organ_registry.list_organs():
            if name in self.organs:
                continue
            self._start_organ(name)

    def _start_organ(self, name: str) -> None:
        organ = self.organ_registry.create(name, self)
        self.organs[name] = organ
        self.topology.nodes[self.node_id].organs.add(name)
        if name in {"ai_policy", "prediction", "process_watcher", "gpu_inference", "uiautomation"}:
            self.background_organs.add(name)
        organ.init()
        organ.start()
        self._schedule_organ_ticks(organ)

    def _schedule_organ_ticks(self, organ: Organ) -> None:
        manifest = organ.manifest()
        interval_ms = manifest.get("tick_interval_ms")
        if not interval_ms:
            return

        def schedule_next():
            if not organ.running:
                return
            cls = TaskClass.INTERACTIVE
            if organ.name in self.background_organs:
                cls = TaskClass.BACKGROUND
            if cls == TaskClass.BACKGROUND and self.throttle_background:
                threading.Timer(interval_ms / 1000.0, schedule_next).start()
                return
            task = Task(
                task_id=str(uuid.uuid4()),
                owner=organ.name,
                cls=cls,
                base_priority=0,
                dynamic_priority=0,
                deadline=None,
                created_at=time.time(),
                func=lambda: self._run_organ_tick(organ, interval_ms),
            )
            self.scheduler.submit(task)

        schedule_next()

    def _run_organ_tick(self, organ: Organ, interval_ms: int) -> None:
        start = time.time()
        try:
            organ.tick()
        except Exception as e:
            print(f"[Kernel] Organ {organ.name} tick failed: {e}")
            self._handle_organ_failure(organ.name, e)
        finally:
            if organ.running:
                delay = max(0.0, interval_ms / 1000.0 - (time.time() - start))
                threading.Timer(delay, lambda: self._schedule_organ_ticks(organ)).start()

    def _handle_organ_failure(self, name: str, error: Exception) -> None:
        count = self.organ_restart_counts.get(name, 0) + 1
        self.organ_restart_counts[name] = count
        self.record_episode(
            trigger="organ_failure",
            severity="warning",
            tags=["organ", "failure", name],
            context={"node_id": self.node_id, "ts_start": time.time(), "organ": name},
            signals={"error": str(error), "restart_count": count},
            constraints_active={},
            decision={"action": "restart_organ" if count < 5 else "give_up"},
            alternatives=[],
            outcome={"status": "restarting" if count < 5 else "disabled"},
            metrics_before={},
            metrics_after={},
            lessons=[f"Organ {name} failed with {error}, restart_count={count}"],
            followups=[]
        )
        if count < 5:
            try:
                self.organs[name].stop()
            except Exception:
                pass
            self._start_organ(name)
        else:
            print(f"[Kernel] Organ {name} disabled after repeated failures")

    def shutdown(self) -> None:
        for organ in self.organs.values():
            organ.stop()
        self.scheduler.shutdown()
        self.store.save_state()

    def record_episode(self,
                       trigger: str,
                       severity: str,
                       tags: List[str],
                       context: Dict[str, Any],
                       signals: Dict[str, Any],
                       constraints_active: Dict[str, Any],
                       decision: Dict[str, Any],
                       alternatives: List[Dict[str, Any]],
                       outcome: Dict[str, Any],
                       metrics_before: Dict[str, Any],
                       metrics_after: Dict[str, Any],
                       lessons: List[str],
                       followups: List[Dict[str, Any]]) -> None:
        now = time.time()
        episode = ReflectionEpisode(
            id=str(uuid.uuid4()),
            ts_start=context.get("ts_start", now),
            ts_end=now,
            context=context,
            trigger=trigger,
            severity=severity,
            tags=tags,
            signals=signals,
            constraints_active=constraints_active,
            decision=decision,
            alternatives=alternatives,
            outcome=outcome,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            lessons=lessons,
            followups=followups,
        )
        self.store.append_episode(episode)


# =========================
# PyQt5 Cockpit GUI
# =========================

class CockpitWindow(QtWidgets.QMainWindow):
    """
    Cockpit to show kernel status, episodes, telemetry, processes, topology, and controls.
    """

    def __init__(self, kernel: Kernel):
        super().__init__()
        self.kernel = kernel
        self.setWindowTitle("Borg Kernel Cockpit (Omega)")
        self.resize(1200, 750)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.status_tab = QtWidgets.QWidget()
        self.episodes_tab = QtWidgets.QWidget()
        self.telemetry_tab = QtWidgets.QWidget()
        self.processes_tab = QtWidgets.QWidget()
        self.topology_tab = QtWidgets.QWidget()
        self.controls_tab = QtWidgets.QWidget()

        self.tabs.addTab(self.status_tab, "Status")
        self.tabs.addTab(self.episodes_tab, "Episodes")
        self.tabs.addTab(self.telemetry_tab, "Telemetry")
        self.tabs.addTab(self.processes_tab, "Processes")
        self.tabs.addTab(self.topology_tab, "Topology")
        self.tabs.addTab(self.controls_tab, "Controls")

        self._init_status_tab()
        self._init_episodes_tab()
        self._init_telemetry_tab()
        self._init_processes_tab()
        self._init_topology_tab()
        self._init_controls_tab()

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_views)
        self.refresh_timer.start(1000)

    def _init_status_tab(self):
        layout = QtWidgets.QVBoxLayout()
        self.status_label = QtWidgets.QLabel("Kernel running...")
        self.organs_list = QtWidgets.QListWidget()
        self.throttle_label = QtWidgets.QLabel("Background throttling: false")
        self.restart_label = QtWidgets.QLabel("Organ restarts: {}")
        layout.addWidget(self.status_label)
        layout.addWidget(QtWidgets.QLabel("Organs:"))
        layout.addWidget(self.organs_list)
        layout.addWidget(self.throttle_label)
        layout.addWidget(self.restart_label)
        self.status_tab.setLayout(layout)

    def _init_episodes_tab(self):
        layout = QtWidgets.QVBoxLayout()
        self.episodes_table = QtWidgets.QTableWidget()
        self.episodes_table.setColumnCount(5)
        self.episodes_table.setHorizontalHeaderLabels(
            ["ID", "Trigger", "Severity", "Tags", "Lessons"]
        )
        self.episodes_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.episodes_table)
        self.episodes_tab.setLayout(layout)

    def _init_telemetry_tab(self):
        layout = QtWidgets.QFormLayout()
        self.telemetry_labels: Dict[str, QtWidgets.QLabel] = {}
        for key in ["cpu_percent", "memory_percent", "memory_used_mb", "memory_total_mb",
                    "process_count", "thread_count", "uptime_sec"]:
            lbl = QtWidgets.QLabel("-")
            self.telemetry_labels[key] = lbl
            layout.addRow(key, lbl)
        self.telemetry_tab.setLayout(layout)

    def _init_processes_tab(self):
        layout = QtWidgets.QVBoxLayout()
        self.processes_table = QtWidgets.QTableWidget()
        self.processes_table.setColumnCount(3)
        self.processes_table.setHorizontalHeaderLabels(["PID", "Name", "Cmdline"])
        self.processes_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.processes_table)
        self.processes_tab.setLayout(layout)

    def _init_topology_tab(self):
        layout = QtWidgets.QVBoxLayout()
        self.topology_table = QtWidgets.QTableWidget()
        self.topology_table.setColumnCount(5)
        self.topology_table.setHorizontalHeaderLabels(
            ["Node ID", "CPU Cores", "Memory MB", "OS", "Organs"]
        )
        self.topology_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.topology_table)
        self.topology_tab.setLayout(layout)

    def _init_controls_tab(self):
        layout = QtWidgets.QVBoxLayout()

        # Toggle organs
        self.organs_toggle_list = QtWidgets.QListWidget()
        self.organs_toggle_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        layout.addWidget(QtWidgets.QLabel("Toggle organs (start/stop):"))
        layout.addWidget(self.organs_toggle_list)

        self.toggle_button = QtWidgets.QPushButton("Toggle selected organs")
        self.toggle_button.clicked.connect(self._toggle_selected_organs)
        layout.addWidget(self.toggle_button)

        # Adjust CPU ceiling via DSL (we just change thresholds)
        self.cpu_high_spin = QtWidgets.QDoubleSpinBox()
        self.cpu_high_spin.setRange(10.0, 100.0)
        self.cpu_high_spin.setValue(80.0)
        self.cpu_low_spin = QtWidgets.QDoubleSpinBox()
        self.cpu_low_spin.setRange(0.0, 100.0)
        self.cpu_low_spin.setValue(60.0)

        form = QtWidgets.QFormLayout()
        form.addRow("CPU high throttle threshold", self.cpu_high_spin)
        form.addRow("CPU low unthrottle threshold", self.cpu_low_spin)
        layout.addLayout(form)

        self.apply_constraints_button = QtWidgets.QPushButton("Apply constraint thresholds")
        self.apply_constraints_button.clicked.connect(self._apply_constraint_thresholds)
        layout.addWidget(self.apply_constraints_button)

        layout.addStretch()
        self.controls_tab.setLayout(layout)

    def _toggle_selected_organs(self):
        selected = [i.text() for i in self.organs_toggle_list.selectedItems()]
        for name in selected:
            organ = self.kernel.organs.get(name)
            if organ is None:
                continue
            if organ.running:
                organ.stop()
            else:
                organ.start()
                self.kernel._schedule_organ_ticks(organ)

    def _apply_constraint_thresholds(self):
        high = self.cpu_high_spin.value()
        low = self.cpu_low_spin.value()
        dsl = ConstraintDSL()
        dsl.add_rule(
            ConstraintRule(
                name="cpu_high_throttle",
                metric="cpu_percent",
                op=ConstraintOp.GT,
                threshold=high,
                target="background_organs",
                action="throttle",
            )
        )
        dsl.add_rule(
            ConstraintRule(
                name="cpu_low_unthrottle",
                metric="cpu_percent",
                op=ConstraintOp.LT,
                threshold=low,
                target="background_organs",
                action="unthrottle",
            )
        )
        self.kernel.constraint_dsl = dsl

    def refresh_views(self):
        self._refresh_status()
        self._refresh_episodes()
        self._refresh_telemetry()
        self._refresh_processes()
        self._refresh_topology()
        self._refresh_controls()

    def _refresh_status(self):
        self.status_label.setText(f"Kernel node: {self.kernel.node_id}")
        self.organs_list.clear()
        for name in sorted(self.kernel.organs.keys()):
            self.organs_list.addItem(name)
        self.throttle_label.setText(f"Background throttling: {self.kernel.throttle_background}")
        self.restart_label.setText(f"Organ restarts: {self.kernel.organ_restart_counts}")

    def _refresh_episodes(self):
        episodes = list(self.kernel.store.episodes_in_memory)[-200:]
        self.episodes_table.setRowCount(len(episodes))
        for row, ep in enumerate(episodes):
            self.episodes_table.setItem(row, 0, QtWidgets.QTableWidgetItem(ep.id))
            self.episodes_table.setItem(row, 1, QtWidgets.QTableWidgetItem(ep.trigger))
            self.episodes_table.setItem(row, 2, QtWidgets.QTableWidgetItem(ep.severity))
            self.episodes_table.setItem(row, 3, QtWidgets.QTableWidgetItem(", ".join(ep.tags)))
            self.episodes_table.setItem(row, 4, QtWidgets.QTableWidgetItem("; ".join(ep.lessons)))

    def _refresh_telemetry(self):
        snap = self.kernel.last_telemetry or {}
        for key, lbl in self.telemetry_labels.items():
            val = snap.get(key)
            if isinstance(val, float):
                lbl.setText(f"{val:.2f}")
            else:
                lbl.setText(str(val) if val is not None else "-")

    def _refresh_processes(self):
        procs = self.kernel.last_processes or []
        self.processes_table.setRowCount(len(procs))
        for row, p in enumerate(procs):
            self.processes_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(p.get("pid", ""))))
            self.processes_table.setItem(row, 1, QtWidgets.QTableWidgetItem(p.get("name", "")))
            self.processes_table.setItem(row, 2, QtWidgets.QTableWidgetItem(p.get("cmdline", "")))

    def _refresh_topology(self):
        nodes = list(self.kernel.topology.nodes.values())
        self.topology_table.setRowCount(len(nodes))
        for row, n in enumerate(nodes):
            self.topology_table.setItem(row, 0, QtWidgets.QTableWidgetItem(n.node_id))
            self.topology_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(n.capabilities.cpu_cores)))
            self.topology_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(n.capabilities.memory_mb)))
            self.topology_table.setItem(row, 3, QtWidgets.QTableWidgetItem(n.capabilities.os_type))
            self.topology_table.setItem(row, 4, QtWidgets.QTableWidgetItem(", ".join(sorted(n.organs))))

    def _refresh_controls(self):
        self.organs_toggle_list.clear()
        for name in sorted(self.kernel.organs.keys()):
            self.organs_toggle_list.addItem(name)


# =========================
# Main entry
# =========================

def run_kernel(kernel: Kernel):
    kernel.autoload_organs()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        kernel.shutdown()


def main():
    # For multi-node federation, run multiple instances with different node_id/port/peers.
    kernel = Kernel(
        node_id="node-1",
        federation_peers=[("127.0.0.1", 50051)],  # adjust or empty list if single node
        listen_port=50050,
        federation_secret=b"super-secret-key",
    )

    kernel_thread = threading.Thread(target=run_kernel, args=(kernel,), daemon=True)
    kernel_thread.start()

    app = QtWidgets.QApplication([])
    window = CockpitWindow(kernel)
    window.show()
    app.exec_()

    kernel.shutdown()


if __name__ == "__main__":
    main()

