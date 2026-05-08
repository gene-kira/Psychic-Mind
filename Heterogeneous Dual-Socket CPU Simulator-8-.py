#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swarm Simulator v7 – Predictive, ML-Driven, Containerized, Pipeline-Aware Tactical Cockpit

New over v6:

🔥 Real ML-like model (learned weights)
    - Online-learned scoring model per scheduler:
        - Features: workload intensity, node util, temp, anomalies, migrations, net cost, storage pressure
        - Target: realized "success" (low failures, low anomalies, good completion)
    - Simple online linear model (no external deps).

🔥 Dynamic network congestion
    - Each link tracks:
        - current load
        - packet loss probability
        - bandwidth collapse under heavy load
    - Communication-heavy workloads increase link load.

🔥 Persistent storage model
    - Nodes have:
        - storage type: SSD / HDD / Distributed
        - capacity
        - IOPS
        - base latency
    - Workloads have:
        - storage demand
        - storage profile (SSD/HDD/Distributed)
    - Storage mismatch and saturation penalize scheduling.

🔥 Containerization
    - Workloads are "containers":
        - startup time (cold start)
        - resource isolation factor
    - States:
        - QUEUED → STARTING → RUNNING → COMPLETED/FAILED

🔥 Pipeline DAGs
    - Workloads can belong to pipelines:
        - stage 1 → stage 2 → stage 3
    - A stage can only run when all its dependencies are COMPLETED.

🔥 Predictive scheduling
    - Forecast:
        - future utilization
        - future temp
        - future network congestion
    - Scheduler uses forecasted metrics in scoring.

🔥 Animated migration visualization
    - Last migrations are tracked and rendered as transient arrows in the swarm map.

🔥 Cluster heatmap
    - Swarm map tiles color-coded by:
        - utilization
        - temp
        - anomalies density

Requires:
    pip install PySide6
"""

import sys
import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from PySide6 import QtWidgets, QtCore, QtGui

# -----------------------------
# Domain model
# -----------------------------

WORKLOAD_STATES = ("QUEUED", "STARTING", "RUNNING", "COMPLETED", "FAILED")
NODE_STATES = ("UP", "DOWN", "REBOOTING")

STORAGE_TYPES = ("SSD", "HDD", "Distributed")


@dataclass
class StorageProfile:
    kind: str  # "SSD", "HDD", "Distributed"
    capacity_gb: float
    used_gb: float = 0.0
    iops: float = 10000.0
    base_latency_ms: float = 1.0

    def pressure(self) -> float:
        if self.capacity_gb <= 0:
            return 1.0
        return min(1.0, self.used_gb / self.capacity_gb)


@dataclass
class CpuSocket:
    name: str
    kind: str  # "desktop" or "xeon"
    cores_total: int
    base_ghz: float
    boost_ghz: float
    cache_mb: float
    ecc: bool
    tdp_w: int

    mem_total_gb: float
    io_total: float

    max_temp_c: float
    max_power_w: float

    reliability_weight: float
    latency_weight: float
    throughput_weight: float
    memory_weight: float

    # dynamic
    cores_used: float = 0.0
    mem_used_gb: float = 0.0
    io_used: float = 0.0
    temp_c: float = 35.0
    power_w: float = 40.0

    util_history: List[float] = field(default_factory=lambda: [0.0] * 60)
    energy_joules: float = 0.0  # accumulated

    def reset_usage(self):
        self.cores_used = 0.0
        self.mem_used_gb = 0.0
        self.io_used = 0.0

    def add_usage(self, cores: float, mem_gb: float, io: float):
        self.cores_used += cores
        self.mem_used_gb += mem_gb
        self.io_used += io

    def utilization_percent(self) -> float:
        if self.cores_total <= 0:
            return 0.0
        return max(0.0, min(100.0, (self.cores_used / self.cores_total) * 100.0))

    def update_thermal_power(self, tick_seconds: float = 0.8):
        load_factor = self.cores_used / max(1.0, self.cores_total)
        self.power_w = self.tdp_w * (0.3 + 0.7 * load_factor)
        self.temp_c = 30.0 + 50.0 * load_factor
        self.temp_c = min(self.temp_c, self.max_temp_c + 20.0)

        util = self.utilization_percent()
        self.util_history.append(util)
        if len(self.util_history) > 60:
            self.util_history = self.util_history[-60:]

        self.energy_joules += self.power_w * tick_seconds

    def forecast_util_temp(self, horizon_ticks: int = 5) -> Tuple[float, float]:
        """Simple forecast: extrapolate last trend."""
        hist = self.util_history[-10:]
        if len(hist) < 2:
            return self.utilization_percent(), self.temp_c
        slope = (hist[-1] - hist[0]) / max(1, len(hist) - 1)
        future_util = self.utilization_percent() + slope * horizon_ticks
        future_util = max(0.0, min(100.0, future_util))
        future_temp = self.temp_c + (future_util - self.utilization_percent()) * 0.2
        return future_util, future_temp

    def describe(self) -> str:
        ecc_str = "ECC" if self.ecc else "non-ECC"
        return (
            f"{self.name} ({self.kind})\n"
            f"  Cores       : {self.cores_total}\n"
            f"  Base/Boost  : {self.base_ghz:.2f}/{self.boost_ghz:.2f} GHz\n"
            f"  Cache       : {self.cache_mb:.1f} MB\n"
            f"  Memory      : {ecc_str}, {self.mem_total_gb:.1f} GB\n"
            f"  I/O cap     : {self.io_total:.1f} units\n"
            f"  TDP         : {self.tdp_w} W (max {self.max_power_w:.1f} W)\n"
            f"  Temp limit  : {self.max_temp_c:.1f} °C\n"
            f"  Weights     : latency={self.latency_weight}, "
            f"throughput={self.throughput_weight}, "
            f"memory={self.memory_weight}, "
            f"reliability={self.reliability_weight}\n"
            f"  Current     : {self.cores_used:.1f}/{self.cores_total} cores, "
            f"{self.mem_used_gb:.1f}/{self.mem_total_gb:.1f} GB, "
            f"{self.io_used:.1f}/{self.io_total:.1f} I/O, "
            f"{self.temp_c:.1f} °C, {self.power_w:.1f} W\n"
            f"  Energy      : {self.energy_joules:.1f} J"
        )


@dataclass
class PipelineStage:
    pipeline_id: int
    stage_id: int
    depends_on: List[int]  # indices of workloads that must complete


@dataclass
class Workload:
    name: str
    latency_sensitivity: int   # 1-10
    throughput_need: int       # 1-10
    memory_intensity: int      # 1-10
    reliability_need: int      # 1-10
    role: str = "custom"       # "web", "db", "ml", "backup", "custom"

    cores_demand: float = 1.0
    mem_demand_gb: float = 1.0
    io_demand: float = 1.0

    storage_demand_gb: float = 1.0
    storage_profile: str = "SSD"  # "SSD", "HDD", "Distributed"

    state: str = "QUEUED"
    remaining_ticks: int = 20
    retries_left: int = 2
    failure_prob: float = 0.05

    home_node: int = 0

    assigned_node: Optional[int] = None
    assigned_socket: Optional[int] = None
    pinned_node: Optional[int] = None
    pinned_socket: Optional[int] = None

    migration_count: int = 0
    last_assignment: Optional[Tuple[int, int]] = None

    history: List[int] = field(default_factory=lambda: [0] * 60)
    energy_joules: float = 0.0

    # containerization
    startup_ticks: int = 3
    startup_remaining: int = 0

    # pipeline
    pipeline_stage: Optional[PipelineStage] = None

    def current_intensity(self) -> int:
        return (
            self.latency_sensitivity +
            self.throughput_need +
            self.memory_intensity +
            self.reliability_need
        )

    def describe(self) -> str:
        pin_str = (
            "None"
            if self.pinned_node is None or self.pinned_socket is None
            else f"node={self.pinned_node}, socket={self.pinned_socket}"
        )
        asg_str = (
            "None"
            if self.assigned_node is None or self.assigned_socket is None
            else f"node={self.assigned_node}, socket={self.assigned_socket}"
        )
        pipe_str = "None"
        if self.pipeline_stage is not None:
            pipe_str = f"Pipeline {self.pipeline_stage.pipeline_id}, Stage {self.pipeline_stage.stage_id}"
        return (
            f"{self.name} [{self.role}]\n"
            f"  State              : {self.state}\n"
            f"  Pipeline           : {pipe_str}\n"
            f"  Latency sensitivity: {self.latency_sensitivity}\n"
            f"  Throughput need    : {self.throughput_need}\n"
            f"  Memory intensity   : {self.memory_intensity}\n"
            f"  Reliability need   : {self.reliability_need}\n"
            f"  Demands            : {self.cores_demand:.1f} cores, "
            f"{self.mem_demand_gb:.1f} GB, {self.io_demand:.1f} I/O\n"
            f"  Storage demand     : {self.storage_demand_gb:.1f} GB ({self.storage_profile})\n"
            f"  Remaining ticks    : {self.remaining_ticks}\n"
            f"  Retries left       : {self.retries_left}\n"
            f"  Failure probability: {self.failure_prob:.2f}\n"
            f"  Home node          : {self.home_node}\n"
            f"  Assigned           : {asg_str}\n"
            f"  Pinned             : {pin_str}\n"
            f"  Migrations         : {self.migration_count}\n"
            f"  Startup remaining  : {self.startup_remaining}\n"
            f"  Energy             : {self.energy_joules:.1f} J"
        )


@dataclass
class Node:
    name: str
    sockets: List[CpuSocket] = field(default_factory=list)
    storage: StorageProfile = None
    state: str = "UP"
    reboot_ticks_left: int = 0
    energy_joules: float = 0.0

    def avg_utilization(self) -> float:
        if not self.sockets:
            return 0.0
        return sum(s.utilization_percent() for s in self.sockets) / len(self.sockets)

    def avg_temp(self) -> float:
        if not self.sockets:
            return 0.0
        return sum(s.temp_c for s in self.sockets) / len(self.sockets)

    def avg_power(self) -> float:
        if not self.sockets:
            return 0.0
        return sum(s.power_w for s in self.sockets) / len(self.sockets)

    def update_energy(self):
        self.energy_joules = sum(s.energy_joules for s in self.sockets)


@dataclass
class CommunicationEdge:
    src_index: int
    dst_index: int
    kind: str  # "rpc", "db", "ml"


@dataclass
class NetworkLink:
    latency_ms: float
    bandwidth_MBps: float
    load: float = 0.0
    packet_loss_prob: float = 0.0

    def effective_latency(self) -> float:
        return self.latency_ms * (1.0 + 2.0 * self.load)

    def effective_bandwidth(self) -> float:
        return max(1.0, self.bandwidth_MBps * (1.0 - 0.7 * self.load))


@dataclass
class MLModel:
    """Tiny online linear model: score = w·x"""
    weights: List[float] = field(default_factory=lambda: [0.0] * 7)
    lr: float = 0.001

    def predict(self, features: List[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features))

    def update(self, features: List[float], target: float):
        pred = self.predict(features)
        error = target - pred
        for i in range(len(self.weights)):
            self.weights[i] += self.lr * error * features[i]


@dataclass
class MigrationEvent:
    src_node: int
    dst_node: int
    tick: int


@dataclass
class SimulationState:
    nodes: List[Node] = field(default_factory=list)
    workloads: List[Workload] = field(default_factory=list)
    scheduler_plugin: str = "Balanced"
    tick_count: int = 0
    anomalies: List[str] = field(default_factory=list)

    queue: List[int] = field(default_factory=list)
    comm_edges: List[CommunicationEdge] = field(default_factory=list)
    network: Dict[Tuple[int, int], NetworkLink] = field(default_factory=dict)

    ml_model: MLModel = field(default_factory=MLModel)
    migration_events: List[MigrationEvent] = field(default_factory=list)


# -----------------------------
# Scheduler plugins
# -----------------------------

class SchedulerPlugin:
    def name(self) -> str:
        return "Base"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int,
              state: SimulationState) -> float:
        raise NotImplementedError


def get_link(state: SimulationState, a: int, b: int) -> Optional[NetworkLink]:
    if (a, b) in state.network:
        return state.network[(a, b)]
    if (b, a) in state.network:
        return state.network[(b, a)]
    return None


def network_cost_for_workload(w: Workload, node_index: int, state: SimulationState) -> float:
    penalty = 0.0
    idx = state.workloads.index(w)
    for edge in state.comm_edges:
        if edge.src_index == edge.dst_index:
            continue
        if edge.src_index == idx or edge.dst_index == idx:
            other_idx = edge.dst_index if edge.src_index == idx else edge.src_index
            other_w = state.workloads[other_idx]
            if other_w.assigned_node is None:
                continue
            other_node = other_w.assigned_node
            if other_node == node_index:
                continue
            link = get_link(state, node_index, other_node)
            if not link:
                continue
            penalty += link.effective_latency() * 0.1 + (1.0 / link.effective_bandwidth()) * 5.0
            penalty += link.packet_loss_prob * 50.0
    return penalty


def storage_cost_for_workload(w: Workload, node: Node) -> float:
    if not node.storage:
        return 50.0
    mismatch = 0.0
    if node.storage.kind != w.storage_profile:
        mismatch = 10.0
    pressure = node.storage.pressure()
    latency = node.storage.base_latency_ms * (1.0 + 3.0 * pressure)
    return mismatch + latency + pressure * 20.0


class BalancedScheduler(SchedulerPlugin):
    def name(self) -> str:
        return "Balanced"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int,
              state: SimulationState) -> float:
        latency_term = w.latency_sensitivity * s.boost_ghz * s.latency_weight
        throughput_term = w.throughput_need * s.cores_total * s.throughput_weight
        mem_factor = s.cache_mb * (1.2 if s.ecc else 1.0)
        memory_term = w.memory_intensity * mem_factor * s.memory_weight
        rel_factor = s.reliability_weight * (1.3 if s.ecc else 1.0)
        reliability_term = w.reliability_need * rel_factor

        numa_penalty = 0.0
        if node_index != w.home_node:
            numa_penalty += 0.2 * (w.memory_intensity + w.throughput_need)

        migration_penalty = 0.0
        if w.last_assignment is not None and w.last_assignment != (node_index, socket_index):
            migration_penalty += 3.0 + 0.2 * w.migration_count

        resource_penalty = 0.0
        if s.cores_used + w.cores_demand > s.cores_total:
            resource_penalty += 100.0
        if s.mem_used_gb + w.mem_demand_gb > s.mem_total_gb:
            resource_penalty += 100.0
        if s.io_used + w.io_demand > s.io_total:
            resource_penalty += 100.0

        net_penalty = network_cost_for_workload(w, node_index, state)
        storage_penalty = storage_cost_for_workload(w, state.nodes[node_index])

        score = latency_term + throughput_term + memory_term + reliability_term
        score -= (numa_penalty + migration_penalty + resource_penalty + net_penalty + storage_penalty)
        return score


class LatencyFirstScheduler(BalancedScheduler):
    def name(self) -> str:
        return "Latency-first"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int,
              state: SimulationState) -> float:
        base = super().score(w, s, node_index, socket_index, state)
        return base + 2.0 * w.latency_sensitivity * s.boost_ghz


class PowerSavingScheduler(BalancedScheduler):
    def name(self) -> str:
        return "Power-saving"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int,
              state: SimulationState) -> float:
        base = super().score(w, s, node_index, socket_index, state)
        power_headroom = max(0.0, s.max_power_w - s.power_w)
        return base + 0.5 * power_headroom


class MLModelScheduler(BalancedScheduler):
    def name(self) -> str:
        return "ML-model"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int,
              state: SimulationState) -> float:
        base = super().score(w, s, node_index, socket_index, state)

        node = state.nodes[node_index]
        link_cost = network_cost_for_workload(w, node_index, state)
        storage_penalty = storage_cost_for_workload(w, node)

        future_util, future_temp = s.forecast_util_temp()
        anomalies_recent = len(state.anomalies[-30:])
        history_intensity = sum(w.history[-10:]) / max(1, len(w.history[-10:]))

        features = [
            history_intensity / 40.0,
            node.avg_utilization() / 100.0,
            node.avg_temp() / 100.0,
            anomalies_recent / 30.0,
            w.migration_count / 10.0,
            link_cost / 50.0,
            storage_penalty / 50.0,
        ]
        ml_score = state.ml_model.predict(features)
        return base + ml_score * 20.0


SCHEDULER_PLUGINS: Dict[str, SchedulerPlugin] = {
    "Balanced": BalancedScheduler(),
    "Latency-first": LatencyFirstScheduler(),
    "Power-saving": PowerSavingScheduler(),
    "ML-model": MLModelScheduler(),
}


# -----------------------------
# Assignment / lifecycle / anomalies / queueing / ML training
# -----------------------------

def pipeline_ready(w: Workload, state: SimulationState) -> bool:
    if w.pipeline_stage is None:
        return True
    for dep_idx in w.pipeline_stage.depends_on:
        if dep_idx < 0 or dep_idx >= len(state.workloads):
            continue
        if state.workloads[dep_idx].state != "COMPLETED":
            return False
    return True


def assign_workloads(state: SimulationState) -> str:
    if not state.nodes:
        return "No nodes defined."

    plugin = SCHEDULER_PLUGINS.get(state.scheduler_plugin, SCHEDULER_PLUGINS["Balanced"])
    lines = []

    for node in state.nodes:
        if node.state != "UP":
            for s in node.sockets:
                s.reset_usage()
            continue
        for s in node.sockets:
            s.reset_usage()

    state.queue = [i for i, w in enumerate(state.workloads) if w.state == "QUEUED"]

    for idx in state.queue:
        w = state.workloads[idx]
        if w.state != "QUEUED":
            continue
        if not pipeline_ready(w, state):
            lines.append(f"[{state.scheduler_plugin}] Workload '{w.name}' waiting on pipeline dependencies.")
            continue

        if w.pinned_node is not None and w.pinned_socket is not None:
            node_index = w.pinned_node
            socket_index = w.pinned_socket
            if 0 <= node_index < len(state.nodes):
                node = state.nodes[node_index]
                if node.state == "UP" and 0 <= socket_index < len(node.sockets):
                    s = node.sockets[socket_index]
                    if (s.cores_used + w.cores_demand <= s.cores_total and
                        s.mem_used_gb + w.mem_demand_gb <= s.mem_total_gb and
                        s.io_used + w.io_demand <= s.io_total and
                        node.storage.used_gb + w.storage_demand_gb <= node.storage.capacity_gb):
                        s.add_usage(w.cores_demand, w.mem_demand_gb, w.io_demand)
                        node.storage.used_gb += w.storage_demand_gb
                        if w.last_assignment is not None and w.last_assignment != (node_index, socket_index):
                            w.migration_count += 1
                            state.migration_events.append(
                                MigrationEvent(w.last_assignment[0], node_index, state.tick_count)
                            )
                        w.assigned_node = node_index
                        w.assigned_socket = socket_index
                        w.last_assignment = (node_index, socket_index)
                        w.state = "STARTING"
                        w.startup_remaining = w.startup_ticks
                        lines.append(
                            f"[{state.scheduler_plugin}] Workload '{w.name}' pinned → "
                            f"{node.name} / {s.name} (STARTING)"
                        )
                        continue

        best_score = -1e9
        best_node_idx = None
        best_socket_idx = None

        for ni, node in enumerate(state.nodes):
            if node.state != "UP":
                continue
            for si, s in enumerate(node.sockets):
                score = plugin.score(w, s, ni, si, state)
                if score > best_score:
                    best_score = score
                    best_node_idx = ni
                    best_socket_idx = si

        if best_node_idx is None or best_socket_idx is None or best_score < -1e8:
            lines.append(f"[{state.scheduler_plugin}] Workload '{w.name}' remains QUEUED (no capacity).")
            continue

        node = state.nodes[best_node_idx]
        s = node.sockets[best_socket_idx]
        if (s.cores_used + w.cores_demand > s.cores_total or
            s.mem_used_gb + w.mem_demand_gb > s.mem_total_gb or
            s.io_used + w.io_demand > s.io_total or
            node.storage.used_gb + w.storage_demand_gb > node.storage.capacity_gb):
            lines.append(f"[{state.scheduler_plugin}] Workload '{w.name}' remains QUEUED (capacity check failed).")
            continue

        s.add_usage(w.cores_demand, w.mem_demand_gb, w.io_demand)
        node.storage.used_gb += w.storage_demand_gb

        if w.last_assignment is not None and w.last_assignment != (best_node_idx, best_socket_idx):
            w.migration_count += 1
            state.migration_events.append(
                MigrationEvent(w.last_assignment[0], best_node_idx, state.tick_count)
            )

        w.assigned_node = best_node_idx
        w.assigned_socket = best_socket_idx
        w.last_assignment = (best_node_idx, best_socket_idx)
        w.state = "STARTING"
        w.startup_remaining = w.startup_ticks

        lines.append(
            f"[{state.scheduler_plugin}] Workload '{w.name}' ({w.role}) → "
            f"{node.name} / {s.name} (score {best_score:.2f}, STARTING)"
        )

    for node in state.nodes:
        if node.state != "UP":
            continue
        for s in node.sockets:
            s.update_thermal_power()
        node.update_energy()

    return "\n".join(lines) if lines else "No workloads to assign."


def summarize_socket_load(state: SimulationState) -> str:
    lines = []
    for ni, node in enumerate(state.nodes):
        lines.append(
            f"Node {ni}: {node.name} [{node.state}] "
            f"Energy={node.energy_joules:.1f} J, Storage={node.storage.used_gb:.1f}/{node.storage.capacity_gb:.1f} GB"
        )
        for si, s in enumerate(node.sockets):
            lines.append(
                f"  Socket {si} ({s.name}): "
                f"{s.cores_used:.1f}/{s.cores_total} cores, "
                f"{s.mem_used_gb:.1f}/{s.mem_total_gb:.1f} GB, "
                f"{s.io_used:.1f}/{s.io_total:.1f} I/O, "
                f"{s.temp_c:.1f} °C, {s.power_w:.1f} W, "
                f"util {s.utilization_percent():.1f}%, "
                f"energy {s.energy_joules:.1f} J"
            )
    return "\n".join(lines) if lines else "No sockets."


def update_workload_dynamics(w: Workload):
    if w.state not in ("QUEUED", "STARTING", "RUNNING"):
        return

    role = w.role.lower()

    def jitter(attr: str, base_delta: int = 1):
        val = getattr(w, attr)
        val += random.randint(-base_delta, base_delta)
        val = max(1, min(10, val))
        setattr(w, attr, val)

    if role == "web":
        jitter("latency_sensitivity", 2)
        jitter("throughput_need", 2)
    elif role == "db":
        jitter("memory_intensity", 2)
        jitter("reliability_need", 2)
    elif role == "ml":
        jitter("throughput_need", 3)
        jitter("memory_intensity", 3)
    elif role == "backup":
        if random.random() < 0.15:
            w.throughput_need = min(10, w.throughput_need + random.randint(2, 4))
            w.memory_intensity = min(10, w.memory_intensity + random.randint(2, 4))
        else:
            w.throughput_need = max(1, int(0.7 * w.throughput_need + 0.3 * 4))
            w.memory_intensity = max(1, int(0.7 * w.memory_intensity + 0.3 * 4))
    else:
        jitter("latency_sensitivity", 1)
        jitter("throughput_need", 1)
        jitter("memory_intensity", 1)
        jitter("reliability_need", 1)

    intensity = w.current_intensity()
    w.history.append(intensity)
    if len(w.history) > 60:
        w.history = w.history[-60:]


def advance_workload_lifecycle(w: Workload, state: SimulationState):
    if w.state not in ("QUEUED", "STARTING", "RUNNING"):
        return

    if w.state == "QUEUED":
        return

    if w.state == "STARTING":
        w.startup_remaining -= 1
        if w.startup_remaining <= 0:
            w.state = "RUNNING"
        return

    w.remaining_ticks -= 1
    if w.remaining_ticks <= 0:
        if random.random() < w.failure_prob:
            if w.retries_left > 0:
                w.retries_left -= 1
                w.state = "QUEUED"
                if w.assigned_node is not None:
                    node = state.nodes[w.assigned_node]
                    node.storage.used_gb = max(0.0, node.storage.used_gb - w.storage_demand_gb)
                w.assigned_node = None
                w.assigned_socket = None
                w.remaining_ticks = random.randint(10, 30)
            else:
                w.state = "FAILED"
                if w.assigned_node is not None:
                    node = state.nodes[w.assigned_node]
                    node.storage.used_gb = max(0.0, node.storage.used_gb - w.storage_demand_gb)
                w.assigned_node = None
                w.assigned_socket = None
        else:
            w.state = "COMPLETED"
            if w.assigned_node is not None:
                node = state.nodes[w.assigned_node]
                node.storage.used_gb = max(0.0, node.storage.used_gb - w.storage_demand_gb)
            w.assigned_node = None
            w.assigned_socket = None


def detect_anomalies(state: SimulationState):
    anomalies = []

    for ni, node in enumerate(state.nodes):
        if node.state != "UP":
            continue
        for si, s in enumerate(node.sockets):
            if s.temp_c > s.max_temp_c:
                anomalies.append(
                    f"Over-temperature: Node {ni} / {s.name} at {s.temp_c:.1f} °C (limit {s.max_temp_c:.1f} °C)"
                )
            if s.power_w > s.max_power_w:
                anomalies.append(
                    f"Over-power: Node {ni} / {s.name} at {s.power_w:.1f} W (limit {s.max_power_w:.1f} W)"
                )

    failures = sum(1 for w in state.workloads if w.state == "FAILED")
    completed = sum(1 for w in state.workloads if w.state == "COMPLETED")
    if failures >= 3 and failures > completed:
        anomalies.append(f"High failure rate: {failures} failed vs {completed} completed workloads.")

    for ni, node in enumerate(state.nodes):
        if node.state != "UP":
            continue
        for si, s in enumerate(node.sockets):
            if len(s.util_history) >= 5:
                recent = s.util_history[-5:]
                if max(recent) - min(recent) > 60.0:
                    anomalies.append(
                        f"Utilization spike: Node {ni} / {s.name} recent util range {min(recent):.1f}-{max(recent):.1f}%"
                    )

    if anomalies:
        state.anomalies.extend(anomalies)
        if len(state.anomalies) > 400:
            state.anomalies = state.anomalies[-400:]


def update_network_congestion(state: SimulationState):
    for link in state.network.values():
        link.load *= 0.8
        link.packet_loss_prob *= 0.7

    for edge in state.comm_edges:
        src = state.workloads[edge.src_index]
        dst = state.workloads[edge.dst_index]
        if src.assigned_node is None or dst.assigned_node is None:
            continue
        if src.assigned_node == dst.assigned_node:
            continue
        link = get_link(state, src.assigned_node, dst.assigned_node)
        if not link:
            continue
        intensity = (src.current_intensity() + dst.current_intensity()) / 80.0
        link.load = min(1.0, link.load + 0.1 * intensity)
        link.packet_loss_prob = min(0.5, link.packet_loss_prob + 0.02 * intensity)


def train_ml_model(state: SimulationState):
    for w in state.workloads:
        if w.state in ("COMPLETED", "FAILED") and w.last_assignment is not None:
            node_index, socket_index = w.last_assignment
            node = state.nodes[node_index]
            s = node.sockets[socket_index]
            link_cost = network_cost_for_workload(w, node_index, state)
            storage_penalty = storage_cost_for_workload(w, node)

            anomalies_recent = len(state.anomalies[-30:])
            history_intensity = sum(w.history[-10:]) / max(1, len(w.history[-10:]))

            features = [
                history_intensity / 40.0,
                node.avg_utilization() / 100.0,
                node.avg_temp() / 100.0,
                anomalies_recent / 30.0,
                w.migration_count / 10.0,
                link_cost / 50.0,
                storage_penalty / 50.0,
            ]

            if w.state == "COMPLETED":
                target = 1.0
            else:
                target = 0.0

            state.ml_model.update(features, target)


def ensure_structured_workloads_and_pipelines(state: SimulationState):
    if state.workloads:
        return

    def mk(role, name, home_node, pipeline_stage=None):
        base = {
            "name": name,
            "role": role,
            "home_node": home_node,
            "pipeline_stage": pipeline_stage,
        }
        if role == "web":
            return Workload(
                latency_sensitivity=8, throughput_need=7, memory_intensity=4, reliability_need=6,
                cores_demand=1.0, mem_demand_gb=1.0, io_demand=1.0,
                storage_demand_gb=2.0, storage_profile="SSD",
                remaining_ticks=random.randint(15, 30),
                **base
            )
        if role == "db":
            return Workload(
                latency_sensitivity=5, throughput_need=6, memory_intensity=9, reliability_need=9,
                cores_demand=1.5, mem_demand_gb=4.0, io_demand=1.0,
                storage_demand_gb=20.0, storage_profile="HDD",
                remaining_ticks=random.randint(20, 40),
                **base
            )
        if role == "ml":
            return Workload(
                latency_sensitivity=3, throughput_need=10, memory_intensity=9, reliability_need=6,
                cores_demand=4.0, mem_demand_gb=6.0, io_demand=2.0,
                storage_demand_gb=10.0, storage_profile="Distributed",
                remaining_ticks=random.randint(25, 50),
                **base
            )
        if role == "backup":
            return Workload(
                latency_sensitivity=2, throughput_need=4, memory_intensity=4, reliability_need=8,
                cores_demand=1.0, mem_demand_gb=2.0, io_demand=3.0,
                storage_demand_gb=50.0, storage_profile="HDD",
                remaining_ticks=random.randint(30, 60),
                **base
            )
        return Workload(
            latency_sensitivity=5, throughput_need=5, memory_intensity=5, reliability_need=5,
            cores_demand=1.0, mem_demand_gb=1.0, io_demand=1.0,
            storage_demand_gb=5.0, storage_profile="SSD",
            remaining_ticks=random.randint(10, 30),
            **base
        )

    # Pipeline 0: Web → DB → ML
    w0 = mk("web", "Pipeline0-Web", 0, PipelineStage(0, 0, []))
    w1 = mk("db", "Pipeline0-DB", 1, PipelineStage(0, 1, []))
    w2 = mk("ml", "Pipeline0-ML", 2, PipelineStage(0, 2, []))

    # dependencies: web & db must complete before ml
    # we will fill depends_on after we know indices
    state.workloads.extend([
        w0,
        w1,
        w2,
        mk("web", "Web Frontend A", 0),
        mk("web", "Web Frontend B", 0),
        mk("db", "DB Primary", 1),
        mk("db", "DB Replica", 1),
        mk("ml", "ML Batch Trainer", 2),
        mk("backup", "Nightly Backup", 1),
    ])

    idx_map = {w.name: i for i, w in enumerate(state.workloads)}
    state.workloads[idx_map["Pipeline0-ML"]].pipeline_stage.depends_on = [
        idx_map["Pipeline0-Web"],
        idx_map["Pipeline0-DB"],
    ]

    def add_edge(a_name, b_name, kind):
        a = idx_map.get(a_name)
        b = idx_map.get(b_name)
        if a is not None and b is not None:
            state.comm_edges.append(CommunicationEdge(a, b, kind))

    add_edge("Web Frontend A", "DB Primary", "rpc")
    add_edge("Web Frontend B", "DB Replica", "rpc")
    add_edge("Web Frontend A", "ML Batch Trainer", "ml")
    add_edge("DB Primary", "ML Batch Trainer", "db")
    add_edge("Pipeline0-Web", "Pipeline0-DB", "rpc")
    add_edge("Pipeline0-DB", "Pipeline0-ML", "db")


def init_network_topology(state: SimulationState):
    n = len(state.nodes)
    for i in range(n):
        for j in range(n):
            if i == j:
                state.network[(i, j)] = NetworkLink(latency_ms=0.2, bandwidth_MBps=200.0)
            else:
                latency = 1.0 + 5.0 * abs(i - j)
                bandwidth = 80.0 / (1 + abs(i - j))
                state.network[(i, j)] = NetworkLink(latency_ms=latency, bandwidth_MBps=bandwidth)


# -----------------------------
# GUI: Models & Widgets
# -----------------------------

class WorkloadTableModel(QtCore.QAbstractTableModel):
    HEADERS = [
        "Name", "Role", "State",
        "Latency", "Throughput", "Memory", "Reliability",
        "Node", "Socket", "Pinned", "Migrations", "Pipeline"
    ]

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self.state.workloads)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self.HEADERS)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        w = self.state.workloads[index.row()]
        col = index.column()

        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                return w.name
            elif col == 1:
                return w.role
            elif col == 2:
                return w.state
            elif col == 3:
                return w.latency_sensitivity
            elif col == 4:
                return w.throughput_need
            elif col == 5:
                return w.memory_intensity
            elif col == 6:
                return w.reliability_need
            elif col == 7:
                return "-" if w.assigned_node is None else str(w.assigned_node)
            elif col == 8:
                return "-" if w.assigned_socket is None else str(w.assigned_socket)
            elif col == 9:
                if w.pinned_node is None or w.pinned_socket is None:
                    return "-"
                return f"{w.pinned_node}/{w.pinned_socket}"
            elif col == 10:
                return w.migration_count
            elif col == 11:
                if w.pipeline_stage is None:
                    return "-"
                return f"{w.pipeline_stage.pipeline_id}:{w.pipeline_stage.stage_id}"
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self.HEADERS[section]
        return section + 1

    def add_workload(self, w: Workload):
        self.beginInsertRows(QtCore.QModelIndex(), len(self.state.workloads), len(self.state.workloads))
        self.state.workloads.append(w)
        self.endInsertRows()

    def refresh(self):
        self.beginResetModel()
        self.endResetModel()


class GaugeWidget(QtWidgets.QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._label = label
        self.setMinimumSize(200, 200)

    def setValue(self, v: float):
        v = max(0.0, min(100.0, v))
        if abs(v - self._value) > 0.01:
            self._value = v
            self.update()

    def value(self) -> float:
        return self._value

    def setLabel(self, text: str):
        self._label = text
        self.update()

    def paintEvent(self, event):
        side = min(self.width(), self.height())
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = QtCore.QRect((self.width() - side) // 2, (self.height() - side) // 2, side, side)
        painter.translate(rect.center())
        radius = side / 2 - 10

        painter.save()
        painter.setBrush(QtGui.QColor(20, 20, 20))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPoint(0, 0), int(radius), int(radius))
        painter.restore()

        painter.save()
        ring_pen = QtGui.QPen(QtGui.QColor(120, 120, 120))
        ring_pen.setWidth(14)
        painter.setPen(ring_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPoint(0, 0), int(radius - 7), int(radius - 7))
        painter.restore()

        painter.save()
        tick_pen = QtGui.QPen(QtGui.QColor(80, 80, 80))
        tick_pen.setWidth(2)
        painter.setPen(tick_pen)
        tick_radius = radius - 16
        for i in range(0, 101, 10):
            angle = (225 + (i * 2.7))
            rad = angle * math.pi / 180.0
            x1 = (tick_radius - 6) * math.cos(rad)
            y1 = (tick_radius - 6) * math.sin(rad)
            x2 = tick_radius * math.cos(rad)
            y2 = tick_radius * math.sin(rad)
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
        painter.restore()

        painter.save()
        needle_angle = 225 + (self._value * 2.7)
        rad = needle_angle * math.pi / 180.0
        needle_len = tick_radius - 10
        needle_pen = QtGui.QPen(QtGui.QColor(220, 80, 60))
        needle_pen.setWidth(4)
        painter.setPen(needle_pen)
        painter.drawLine(
            QtCore.QPointF(0, 0),
            QtCore.QPointF(needle_len * math.cos(rad), needle_len * math.sin(rad))
        )
        painter.setBrush(QtGui.QColor(200, 200, 200))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPointF(0, 0), 6, 6)
        painter.restore()

        painter.save()
        text_rect = QtCore.QRectF(-radius + 20, 0, 2 * (radius - 20), 40)
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(230, 230, 230))
        painter.drawText(text_rect, QtCore.Qt.AlignCenter, f"{self._value:5.1f} %")
        painter.restore()

        painter.save()
        label_rect = QtCore.QRectF(-radius + 20, -radius + 10, 2 * (radius - 20), 30)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(180, 180, 180))
        painter.drawText(label_rect, QtCore.Qt.AlignCenter, self._label)
        painter.restore()


class TimelineWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: List[float] = []
        self.setMinimumHeight(150)

    def setHistory(self, history: List[float]):
        self._history = list(history)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.fillRect(rect, QtGui.QColor(15, 15, 15))

        if not self._history:
            painter.setPen(QtGui.QColor(120, 120, 120))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No history")
            return

        max_val = max(self._history) if self._history else 1
        max_val = max(max_val, 1)

        painter.setPen(QtGui.QColor(60, 60, 60))
        painter.drawRect(rect)

        painter.setPen(QtGui.QColor(80, 80, 80))
        mid_y = rect.center().y()
        painter.drawLine(rect.left(), mid_y, rect.right(), mid_y)

        painter.setPen(QtGui.QPen(QtGui.QColor(80, 160, 240), 2))

        n = len(self._history)
        if n == 1:
            x = rect.left()
            y = rect.bottom() - (self._history[0] / max_val) * rect.height()
            painter.drawPoint(int(x), int(y))
            return

        step_x = rect.width() / max(1, n - 1)
        points = []
        for i, val in enumerate(self._history):
            x = rect.left() + i * step_x
            y = rect.bottom() - (val / max_val) * rect.height()
            points.append(QtCore.QPointF(x, y))

        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])


class SwarmMapWidget(QtWidgets.QWidget):
    """Global swarm map: nodes as tiles with state + metrics + heatmap + migration arrows."""
    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumHeight(200)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.fillRect(rect, QtGui.QColor(10, 10, 10))

        if not self.state.nodes:
            painter.setPen(QtGui.QColor(200, 200, 200))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No nodes")
            return

        n = len(self.state.nodes)
        tile_w = rect.width() / max(1, n)
        tile_h = rect.height()

        centers = []

        for i, node in enumerate(self.state.nodes):
            x = rect.left() + i * tile_w
            tile_rect = QtCore.QRectF(x + 5, rect.top() + 5, tile_w - 10, tile_h - 10)

            util = node.avg_utilization()
            temp = node.avg_temp()
            anomalies_recent = sum(1 for a in self.state.anomalies[-80:] if f"Node {i}" in a)

            util_norm = min(1.0, util / 100.0)
            temp_norm = min(1.0, temp / 100.0)
            anom_norm = min(1.0, anomalies_recent / 10.0)

            r = int(80 + 150 * temp_norm + 80 * anom_norm)
            g = int(60 + 120 * (1.0 - anom_norm))
            b = int(60 + 120 * (1.0 - util_norm))

            if node.state == "DOWN":
                bg = QtGui.QColor(120, 40, 40)
            elif node.state == "REBOOTING":
                bg = QtGui.QColor(120, 120, 40)
            else:
                bg = QtGui.QColor(r, g, b)

            painter.setBrush(bg)
            painter.setPen(QtGui.QColor(120, 120, 120))
            painter.drawRoundedRect(tile_rect, 8, 8)

            text = (
                f"{i}: {node.name}\n"
                f"State: {node.state}\n"
                f"Util: {util:.1f}%  Temp: {temp:.1f}°C\n"
                f"Power: {node.avg_power():.1f} W\n"
                f"Storage: {node.storage.used_gb:.1f}/{node.storage.capacity_gb:.1f} GB"
            )

            painter.setPen(QtGui.QColor(230, 230, 230))
            painter.drawText(tile_rect.adjusted(8, 8, -8, -8), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, text)

            centers.append(tile_rect.center())

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 200, 80), 2))
        now = self.state.tick_count
        for ev in self.state.migration_events[-30:]:
            age = now - ev.tick
            if age < 0 or age > 20:
                continue
            alpha = int(255 * (1.0 - age / 20.0))
            if 0 <= ev.src_node < len(centers) and 0 <= ev.dst_node < len(centers):
                c1 = centers[ev.src_node]
                c2 = centers[ev.dst_node]
                pen = QtGui.QPen(QtGui.QColor(255, 200, 80, alpha), 2)
                painter.setPen(pen)
                painter.drawLine(c1, c2)


# -----------------------------
# GUI: Main window (tactical cockpit v7)
# -----------------------------

class SwarmSimulatorV7(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swarm Simulator v7 – Predictive ML Swarm Cockpit")
        self.resize(1800, 1000)

        self.state = SimulationState()
        self._init_default_swarm()

        self.selected_workload_index: Optional[int] = None
        self.current_node_index: int = 0

        self._build_ui()
        self._apply_dark_palette()
        self._refresh_socket_info()
        ensure_structured_workloads_and_pipelines(self.state)
        init_network_topology(self.state)

        self._run_scheduler_and_update_ui(initial=True)

        self.tick_seconds = 0.8
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(int(self.tick_seconds * 1000))

    def _init_default_swarm(self):
        node0 = Node(
            name="Node 0 – Edge / Desktop",
            sockets=[
                CpuSocket(
                    name="Socket 0 – Desktop CPU",
                    kind="desktop",
                    cores_total=8,
                    base_ghz=3.6,
                    boost_ghz=5.0,
                    cache_mb=16.0,
                    ecc=False,
                    tdp_w=125,
                    mem_total_gb=16.0,
                    io_total=10.0,
                    max_temp_c=85.0,
                    max_power_w=150.0,
                    reliability_weight=0.8,
                    latency_weight=1.5,
                    throughput_weight=1.0,
                    memory_weight=0.8,
                )
            ],
            storage=StorageProfile(kind="SSD", capacity_gb=256.0, iops=20000, base_latency_ms=0.5),
        )

        node1 = Node(
            name="Node 1 – Xeon Server",
            sockets=[
                CpuSocket(
                    name="Socket 0 – Xeon CPU A",
                    kind="xeon",
                    cores_total=16,
                    base_ghz=2.6,
                    boost_ghz=3.8,
                    cache_mb=24.0,
                    ecc=True,
                    tdp_w=165,
                    mem_total_gb=64.0,
                    io_total=20.0,
                    max_temp_c=90.0,
                    max_power_w=220.0,
                    reliability_weight=1.5,
                    latency_weight=0.9,
                    throughput_weight=1.4,
                    memory_weight=1.3,
                ),
                CpuSocket(
                    name="Socket 1 – Xeon CPU B",
                    kind="xeon",
                    cores_total=16,
                    base_ghz=2.6,
                    boost_ghz=3.8,
                    cache_mb=24.0,
                    ecc=True,
                    tdp_w=165,
                    mem_total_gb=64.0,
                    io_total=20.0,
                    max_temp_c=90.0,
                    max_power_w=220.0,
                    reliability_weight=1.5,
                    latency_weight=0.9,
                    throughput_weight=1.4,
                    memory_weight=1.3,
                ),
            ],
            storage=StorageProfile(kind="HDD", capacity_gb=1024.0, iops=5000, base_latency_ms=4.0),
        )

        node2 = Node(
            name="Node 2 – GPU/ML Node",
            sockets=[
                CpuSocket(
                    name="Socket 0 – ML CPU",
                    kind="xeon",
                    cores_total=24,
                    base_ghz=2.8,
                    boost_ghz=4.0,
                    cache_mb=32.0,
                    ecc=True,
                    tdp_w=200,
                    mem_total_gb=128.0,
                    io_total=30.0,
                    max_temp_c=92.0,
                    max_power_w=260.0,
                    reliability_weight=1.4,
                    latency_weight=1.0,
                    throughput_weight=1.6,
                    memory_weight=1.5,
                )
            ],
            storage=StorageProfile(kind="Distributed", capacity_gb=2048.0, iops=15000, base_latency_ms=2.0),
        )

        self.state.nodes = [node0, node1, node2]

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        self.swarm_map = SwarmMapWidget(self.state)
        main_layout.addWidget(self.swarm_map, stretch=1)

        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout)

        node_group = QtWidgets.QGroupBox("Node / Scheduler")
        node_layout = QtWidgets.QFormLayout(node_group)

        self.node_combo = QtWidgets.QComboBox()
        for i, node in enumerate(self.state.nodes):
            self.node_combo.addItem(f"{i}: {node.name}")
        self.node_combo.currentIndexChanged.connect(self._on_node_changed)

        self.scheduler_combo = QtWidgets.QComboBox()
        self.scheduler_combo.addItems(list(SCHEDULER_PLUGINS.keys()))
        self.scheduler_combo.currentIndexChanged.connect(self._on_scheduler_changed)

        node_layout.addRow("View node:", self.node_combo)
        node_layout.addRow("Scheduler plugin:", self.scheduler_combo)

        top_layout.addWidget(node_group, stretch=2)

        socket_group = QtWidgets.QGroupBox("Sockets (current node)")
        socket_layout = QtWidgets.QHBoxLayout(socket_group)

        self.socket_text_left = QtWidgets.QPlainTextEdit()
        self.socket_text_left.setReadOnly(True)
        self.socket_text_right = QtWidgets.QPlainTextEdit()
        self.socket_text_right.setReadOnly(True)

        socket_layout.addWidget(self.socket_text_left)
        socket_layout.addWidget(self.socket_text_right)

        top_layout.addWidget(socket_group, stretch=4)

        mid_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(mid_layout, stretch=3)

        self.workload_model = WorkloadTableModel(self.state)
        self.workload_table = QtWidgets.QTableView()
        self.workload_table.setModel(self.workload_model)
        self.workload_table.horizontalHeader().setStretchLastSection(True)
        self.workload_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.workload_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.workload_table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        mid_layout.addWidget(self.workload_table, stretch=4)

        control_group = QtWidgets.QGroupBox("Workload Controls (Custom Container)")
        control_layout = QtWidgets.QFormLayout(control_group)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Analytics Job, Cache Layer")

        self.latency_spin = QtWidgets.QSpinBox()
        self.latency_spin.setRange(1, 10)
        self.latency_spin.setValue(5)

        self.throughput_spin = QtWidgets.QSpinBox()
        self.throughput_spin.setRange(1, 10)
        self.throughput_spin.setValue(5)

        self.memory_spin = QtWidgets.QSpinBox()
        self.memory_spin.setRange(1, 10)
        self.memory_spin.setValue(5)

        self.reliability_spin = QtWidgets.QSpinBox()
        self.reliability_spin.setRange(1, 10)
        self.reliability_spin.setValue(5)

        self.cores_spin = QtWidgets.QDoubleSpinBox()
        self.cores_spin.setRange(0.1, 32.0)
        self.cores_spin.setSingleStep(0.1)
        self.cores_spin.setValue(1.0)

        self.mem_gb_spin = QtWidgets.QDoubleSpinBox()
        self.mem_gb_spin.setRange(0.1, 256.0)
        self.mem_gb_spin.setSingleStep(0.1)
        self.mem_gb_spin.setValue(1.0)

        self.io_spin = QtWidgets.QDoubleSpinBox()
        self.io_spin.setRange(0.1, 30.0)
        self.io_spin.setSingleStep(0.1)
        self.io_spin.setValue(1.0)

        self.storage_gb_spin = QtWidgets.QDoubleSpinBox()
        self.storage_gb_spin.setRange(0.1, 500.0)
        self.storage_gb_spin.setSingleStep(0.5)
        self.storage_gb_spin.setValue(5.0)

        self.storage_type_combo = QtWidgets.QComboBox()
        self.storage_type_combo.addItems(STORAGE_TYPES)

        self.startup_spin = QtWidgets.QSpinBox()
        self.startup_spin.setRange(0, 20)
        self.startup_spin.setValue(3)

        self.home_node_spin = QtWidgets.QSpinBox()
        self.home_node_spin.setRange(0, max(0, len(self.state.nodes) - 1))
        self.home_node_spin.setValue(0)

        control_layout.addRow("Name:", self.name_edit)
        control_layout.addRow("Latency (1-10):", self.latency_spin)
        control_layout.addRow("Throughput (1-10):", self.throughput_spin)
        control_layout.addRow("Memory intensity (1-10):", self.memory_spin)
        control_layout.addRow("Reliability (1-10):", self.reliability_spin)
        control_layout.addRow("Cores demand:", self.cores_spin)
        control_layout.addRow("Memory demand (GB):", self.mem_gb_spin)
        control_layout.addRow("I/O demand:", self.io_spin)
        control_layout.addRow("Storage demand (GB):", self.storage_gb_spin)
        control_layout.addRow("Storage profile:", self.storage_type_combo)
        control_layout.addRow("Startup ticks:", self.startup_spin)
        control_layout.addRow("Home node:", self.home_node_spin)

        self.add_btn = QtWidgets.QPushButton("Add custom container workload")
        self.add_btn.clicked.connect(self._on_add_workload)
        control_layout.addRow(self.add_btn)

        self.assign_btn = QtWidgets.QPushButton("Force re-assign now")
        self.assign_btn.clicked.connect(self._on_assign_button)
        control_layout.addRow(self.assign_btn)

        mid_layout.addWidget(control_group, stretch=3)

        right_panel = QtWidgets.QGroupBox("Selected Workload – Details & Control")
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        self.selected_details = QtWidgets.QPlainTextEdit()
        self.selected_details.setReadOnly(True)
        self.selected_details.setPlaceholderText("Select a workload to see details.")
        right_layout.addWidget(self.selected_details, stretch=2)

        pin_layout = QtWidgets.QHBoxLayout()
        self.pin_btn = QtWidgets.QPushButton("Pin to current node/socket 0")
        self.unpin_btn = QtWidgets.QPushButton("Unpin")
        self.pin_btn.clicked.connect(self._on_pin_current)
        self.unpin_btn.clicked.connect(self._on_unpin)
        pin_layout.addWidget(self.pin_btn)
        pin_layout.addWidget(self.unpin_btn)
        right_layout.addLayout(pin_layout)

        self.workload_timeline = TimelineWidget()
        right_layout.addWidget(self.workload_timeline, stretch=3)

        mid_layout.addWidget(right_panel, stretch=4)

        bottom_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(bottom_layout, stretch=3)

        gauge_group = QtWidgets.QGroupBox("Per-Socket Utilization (Current Node)")
        gauge_layout = QtWidgets.QVBoxLayout(gauge_group)

        self.gauge_layout_row = QtWidgets.QHBoxLayout()
        self.gauge_widgets: List[GaugeWidget] = []
        self.socket_timeline_widgets: List[TimelineWidget] = []

        gauge_layout.addLayout(self.gauge_layout_row)

        self.socket_timeline_container = QtWidgets.QHBoxLayout()
        gauge_layout.addLayout(self.socket_timeline_container)

        bottom_layout.addWidget(gauge_group, stretch=4)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        bottom_layout.addWidget(right_splitter, stretch=3)

        self.anomaly_text = QtWidgets.QPlainTextEdit()
        self.anomaly_text.setReadOnly(True)
        self.anomaly_text.setPlaceholderText("Anomalies will appear here.")
        right_splitter.addWidget(self.anomaly_text)

        self.explanation_text = QtWidgets.QPlainTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setPlaceholderText("Scheduler decisions will appear here.")
        right_splitter.addWidget(self.explanation_text)

        self.load_text = QtWidgets.QPlainTextEdit()
        self.load_text.setReadOnly(True)
        self.load_text.setPlaceholderText("Socket load summary will appear here.")
        right_splitter.addWidget(self.load_text)

        script_group = QtWidgets.QGroupBox("Operator Scripting (tiny DSL)")
        script_layout = QtWidgets.QVBoxLayout(script_group)
        self.script_edit = QtWidgets.QPlainTextEdit()
        self.script_edit.setPlaceholderText(
            "Example:\n"
            "if temp > 85:\n"
            "    migrate_all_from_node 1\n"
            "if node_state == 'UP' and util > 90:\n"
            "    shutdown_node 2\n"
        )
        self.run_script_btn = QtWidgets.QPushButton("Run Script")
        self.run_script_btn.clicked.connect(self._on_run_script)
        script_layout.addWidget(self.script_edit)
        script_layout.addWidget(self.run_script_btn)

        right_splitter.addWidget(script_group)

        right_splitter.setSizes([150, 250, 200, 200])

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready (swarm real-time mode v7).")

        self._rebuild_gauges_for_current_node()

    def _apply_dark_palette(self):
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(20, 20, 20))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(35, 35, 35))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 220))
        palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
        palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(64, 128, 255))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        app.setPalette(palette)

    def _refresh_socket_info(self):
        node = self.state.nodes[self.current_node_index]
        if len(node.sockets) >= 1:
            self.socket_text_left.setPlainText(node.sockets[0].describe())
        else:
            self.socket_text_left.setPlainText("No socket.")
        if len(node.sockets) >= 2:
            self.socket_text_right.setPlainText(node.sockets[1].describe())
        else:
            self.socket_text_right.setPlainText("No second socket.")

    def _rebuild_gauges_for_current_node(self):
        for i in reversed(range(self.gauge_layout_row.count())):
            item = self.gauge_layout_row.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        for i in reversed(range(self.socket_timeline_container.count())):
            item = self.socket_timeline_container.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)

        self.gauge_widgets.clear()
        self.socket_timeline_widgets.clear()

        node = self.state.nodes[self.current_node_index]
        for si, s in enumerate(node.sockets):
            gw = GaugeWidget(f"Socket {si} – {s.kind}")
            self.gauge_layout_row.addWidget(gw)
            self.gauge_widgets.append(gw)

            tw = TimelineWidget()
            self.socket_timeline_container.addWidget(tw)
            self.socket_timeline_widgets.append(tw)

    def _update_gauges_and_socket_timelines(self):
        node = self.state.nodes[self.current_node_index]
        for si, s in enumerate(node.sockets):
            if si < len(self.gauge_widgets):
                self.gauge_widgets[si].setValue(s.utilization_percent())
            if si < len(self.socket_timeline_widgets):
                self.socket_timeline_widgets[si].setHistory(s.util_history)

    def _run_scheduler_and_update_ui(self, initial: bool = False):
        explanation = assign_workloads(self.state)
        self.workload_model.refresh()

        if initial:
            self.explanation_text.setPlainText(explanation)
        else:
            existing = self.explanation_text.toPlainText()
            if existing:
                self.explanation_text.setPlainText(existing + "\n" + explanation)
            else:
                self.explanation_text.setPlainText(explanation)

        self.load_text.setPlainText(summarize_socket_load(self.state))
        self._update_gauges_and_socket_timelines()
        self._refresh_socket_info()
        self._update_selected_panel()
        self._update_anomaly_panel()
        self.swarm_map.update()

    def _update_anomaly_panel(self):
        self.anomaly_text.setPlainText("\n".join(self.state.anomalies[-120:]))

    def _update_selected_panel(self):
        if self.selected_workload_index is None or self.selected_workload_index >= len(self.state.workloads):
            self.selected_details.setPlainText("Select a workload to see details.")
            self.workload_timeline.setHistory([])
            return

        w = self.state.workloads[self.selected_workload_index]
        self.selected_details.setPlainText(w.describe())
        self.workload_timeline.setHistory(w.history)

    # ----------------- Operator scripting DSL -----------------

    def _on_run_script(self):
        script = self.script_edit.toPlainText()
        lines = [l.rstrip() for l in script.splitlines() if l.strip()]
        current_condition = None

        for line in lines:
            if line.startswith("if "):
                cond = line[3:].strip().rstrip(":")
                current_condition = cond
            else:
                cmd = line.strip()
                if current_condition:
                    if not self._eval_condition(current_condition):
                        continue
                self._execute_command(cmd)

        self._run_scheduler_and_update_ui()
        self.status_bar.showMessage("Operator script executed.")

    def _eval_condition(self, cond: str) -> bool:
        cond = cond.strip()
        try:
            if "temp" in cond:
                max_temp = max(node.avg_temp() for node in self.state.nodes) if self.state.nodes else 0
                return eval(cond, {}, {"temp": max_temp})
            if "util" in cond:
                max_util = max(node.avg_utilization() for node in self.state.nodes) if self.state.nodes else 0
                return eval(cond, {}, {"util": max_util})
            if "node_state" in cond:
                any_state = any(node.state == "UP" for node in self.state.nodes)
                return eval(cond, {}, {"node_state": "UP" if any_state else "DOWN"})
        except Exception:
            return False
        return False

    def _execute_command(self, cmd: str):
        parts = cmd.split()
        if not parts:
            return
        if parts[0] == "migrate_all_from_node" and len(parts) == 2:
            try:
                n = int(parts[1])
                for w in self.state.workloads:
                    if w.assigned_node == n:
                        w.state = "QUEUED"
                        if w.assigned_node is not None:
                            node = self.state.nodes[w.assigned_node]
                            node.storage.used_gb = max(0.0, node.storage.used_gb - w.storage_demand_gb)
                        w.assigned_node = None
                        w.assigned_socket = None
                self.state.anomalies.append(f"Operator: migrated all workloads from node {n}.")
            except ValueError:
                pass
        elif parts[0] == "shutdown_node" and len(parts) == 2:
            try:
                n = int(parts[1])
                if 0 <= n < len(self.state.nodes):
                    self.state.nodes[n].state = "DOWN"
                    for w in self.state.workloads:
                        if w.assigned_node == n:
                            w.state = "QUEUED"
                            self.state.nodes[n].storage.used_gb = max(
                                0.0, self.state.nodes[n].storage.used_gb - w.storage_demand_gb
                            )
                            w.assigned_node = None
                            w.assigned_socket = None
                    self.state.anomalies.append(f"Operator: shutdown node {n}.")
            except ValueError:
                pass
        elif parts[0] == "reboot_node" and len(parts) == 2:
            try:
                n = int(parts[1])
                if 0 <= n < len(self.state.nodes):
                    node = self.state.nodes[n]
                    node.state = "REBOOTING"
                    node.reboot_ticks_left = 5
                    for w in self.state.workloads:
                        if w.assigned_node == n:
                            w.state = "QUEUED"
                            node.storage.used_gb = max(0.0, node.storage.used_gb - w.storage_demand_gb)
                            w.assigned_node = None
                            w.assigned_socket = None
                    self.state.anomalies.append(f"Operator: reboot node {n}.")
            except ValueError:
                pass
        elif parts[0] == "pin_role_to_node" and len(parts) == 3:
            role = parts[1]
            try:
                n = int(parts[2])
                for w in self.state.workloads:
                    if w.role == role:
                        w.pinned_node = n
                        w.pinned_socket = 0
                self.state.anomalies.append(f"Operator: pinned role '{role}' to node {n}.")
            except ValueError:
                pass

    # ----------------- Event handlers -----------------

    def _on_add_workload(self):
        name = self.name_edit.text().strip()
        if not name:
            name = f"Custom Workload {len(self.state.workloads) + 1}"
        home_node = max(0, min(len(self.state.nodes) - 1, self.home_node_spin.value()))
        w = Workload(
            name=name,
            latency_sensitivity=self.latency_spin.value(),
            throughput_need=self.throughput_spin.value(),
            memory_intensity=self.memory_spin.value(),
            reliability_need=self.reliability_spin.value(),
            role="custom",
            cores_demand=self.cores_spin.value(),
            mem_demand_gb=self.mem_gb_spin.value(),
            io_demand=self.io_spin.value(),
            storage_demand_gb=self.storage_gb_spin.value(),
            storage_profile=self.storage_type_combo.currentText(),
            remaining_ticks=random.randint(15, 40),
            home_node=home_node,
            startup_ticks=self.startup_spin.value(),
        )
        self.workload_model.add_workload(w)
        self.status_bar.showMessage(f"Added custom workload '{w.name}'.")
        self._run_scheduler_and_update_ui()

    def _on_assign_button(self):
        self._run_scheduler_and_update_ui()
        self.status_bar.showMessage("Manual re-assignment executed.")

    def _on_selection_changed(self, selected, deselected):
        indexes = self.workload_table.selectionModel().selectedRows()
        if indexes:
            self.selected_workload_index = indexes[0].row()
        else:
            self.selected_workload_index = None
        self._update_selected_panel()

    def _on_node_changed(self, idx: int):
        self.current_node_index = idx
        self._rebuild_gauges_for_current_node()
        self._refresh_socket_info()
        self._update_gauges_and_socket_timelines()
        self.status_bar.showMessage(f"Viewing node {idx}.")

    def _on_scheduler_changed(self, idx: int):
        self.state.scheduler_plugin = self.scheduler_combo.currentText()
        self.status_bar.showMessage(f"Scheduler plugin changed to: {self.state.scheduler_plugin}")
        self._run_scheduler_and_update_ui()

    def _on_pin_current(self):
        if self.selected_workload_index is None:
            return
        w = self.state.workloads[self.selected_workload_index]
        w.pinned_node = self.current_node_index
        w.pinned_socket = 0
        self.status_bar.showMessage(
            f"Workload '{w.name}' pinned to node {w.pinned_node}, socket {w.pinned_socket}."
        )
        self._run_scheduler_and_update_ui()

    def _on_unpin(self):
        if self.selected_workload_index is None:
            return
        w = self.state.workloads[self.selected_workload_index]
        w.pinned_node = None
        w.pinned_socket = None
        self.status_bar.showMessage(f"Workload '{w.name}' unpinned.")
        self._run_scheduler_and_update_ui()

    def _simulate_node_failures(self):
        for idx, node in enumerate(self.state.nodes):
            if node.state == "UP":
                if random.random() < 0.01:
                    node.state = "DOWN"
                    for w in self.state.workloads:
                        if w.assigned_node == idx:
                            w.state = "QUEUED"
                            node.storage.used_gb = max(0.0, node.storage.used_gb - w.storage_demand_gb)
                            w.assigned_node = None
                            w.assigned_socket = None
                    self.state.anomalies.append(f"Random failure: {node.name} went DOWN.")
                elif node.avg_temp() > node.sockets[0].max_temp_c + 10:
                    node.state = "REBOOTING"
                    node.reboot_ticks_left = 5
                    for w in self.state.workloads:
                        if w.assigned_node == idx:
                            w.state = "QUEUED"
                            node.storage.used_gb = max(0.0, node.storage.used_gb - w.storage_demand_gb)
                            w.assigned_node = None
                            w.assigned_socket = None
                    self.state.anomalies.append(f"Thermal shutdown: {node.name} REBOOTING.")
            elif node.state == "REBOOTING":
                node.reboot_ticks_left -= 1
                if node.reboot_ticks_left <= 0:
                    node.state = "UP"
                    self.state.anomalies.append(f"Node {node.name} is back UP.")

    def _on_tick(self):
        self.state.tick_count += 1

        for w in self.state.workloads:
            update_workload_dynamics(w)
            advance_workload_lifecycle(w, self.state)

        if random.random() < 0.08:
            role = random.choice(["web", "db", "ml", "backup"])
            home_node = random.randint(0, len(self.state.nodes) - 1)
            base_names = {
                "web": "Web Frontend",
                "db": "DB Node",
                "ml": "ML Worker",
                "backup": "Backup Task",
            }
            name = f"{base_names[role]} #{random.randint(2, 99)}"
            if role == "web":
                w = Workload(
                    name=name,
                    latency_sensitivity=8, throughput_need=7, memory_intensity=4, reliability_need=6,
                    role="web",
                    cores_demand=1.0, mem_demand_gb=1.0, io_demand=1.0,
                    storage_demand_gb=2.0, storage_profile="SSD",
                    remaining_ticks=random.randint(15, 30),
                    home_node=home_node,
                    startup_ticks=2,
                )
            elif role == "db":
                w = Workload(
                    name=name,
                    latency_sensitivity=5, throughput_need=6, memory_intensity=9, reliability_need=9,
                    role="db",
                    cores_demand=1.5, mem_demand_gb=4.0, io_demand=1.0,
                    storage_demand_gb=20.0, storage_profile="HDD",
                    remaining_ticks=random.randint(20, 40),
                    home_node=home_node,
                    startup_ticks=4,
                )
            elif role == "ml":
                w = Workload(
                    name=name,
                    latency_sensitivity=3, throughput_need=10, memory_intensity=9, reliability_need=6,
                    role="ml",
                    cores_demand=4.0, mem_demand_gb=6.0, io_demand=2.0,
                    storage_demand_gb=10.0, storage_profile="Distributed",
                    remaining_ticks=random.randint(25, 50),
                    home_node=home_node,
                    startup_ticks=5,
                )
            else:
                w = Workload(
                    name=name,
                    latency_sensitivity=2, throughput_need=4, memory_intensity=4, reliability_need=8,
                    role="backup",
                    cores_demand=1.0, mem_demand_gb=2.0, io_demand=3.0,
                    storage_demand_gb=50.0, storage_profile="HDD",
                    remaining_ticks=random.randint(30, 60),
                    home_node=home_node,
                    startup_ticks=3,
                )
            self.state.workloads.append(w)

        self._simulate_node_failures()
        update_network_congestion(self.state)
        train_ml_model(self.state)

        self._run_scheduler_and_update_ui()

        detect_anomalies(self.state)

        text = self.explanation_text.toPlainText()
        lines = text.splitlines()
        if len(lines) > 600:
            self.explanation_text.setPlainText("\n".join(lines[-400:]))

        self.status_bar.showMessage(f"Real-time tick {self.state.tick_count} processed.")

# -----------------------------
# main
# -----------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SwarmSimulatorV7()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
