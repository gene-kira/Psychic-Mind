#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Socket / Swarm Simulator v5 – Tactical Cockpit

Adds on top of v4:

1) Resource models (per socket)
   - CPU cores (total / used)
   - Memory (GB total / used)
   - I/O capacity (units total / used)
   - Thermal limits (temp, max_temp)
   - Power envelopes (power, max_power)

2) Workload lifecycles
   - States: SPAWNED, RUNNING, COMPLETED, FAILED
   - Remaining time
   - Retry count
   - Failure probability
   - Automatic respawn of new workloads

3) Migration cost
   - Moving a workload between sockets adds penalty
   - Tracks migration count
   - Temporary “migration cooldown” penalty in scoring

4) NUMA simulation
   - Nodes with multiple sockets
   - Local vs remote memory
   - Cross-socket bandwidth / latency penalty
   - Each workload has a “home node” for memory locality

5) Anomaly detection (simple but extensible)
   - Detects:
       - Sudden utilization spikes
       - Over-temperature
       - Over-power
       - High failure rate
   - Flags anomalies in a side panel

6) Swarm mode
   - Multiple nodes (each with sockets)
   - Scheduler chooses node + socket
   - GUI lets you select which node to “view”

7) Socket utilization timelines
   - Per-socket history graph (like per-workload timeline)
   - Shows utilization over time

8) Scheduler plugin system
   - Pluggable strategies:
       - Balanced
       - Latency-first
       - Power-saving
       - ML-like heuristic (simple scoring tweak placeholder)
   - Easy to extend with real ML later

Requires:
    pip install PySide6
"""

import sys
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from PySide6 import QtWidgets, QtCore, QtGui

# -----------------------------
# Domain model
# -----------------------------

WORKLOAD_STATES = ("SPAWNED", "RUNNING", "COMPLETED", "FAILED")


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

    def update_thermal_power(self):
        load_factor = self.cores_used / max(1.0, self.cores_total)
        self.power_w = self.tdp_w * (0.3 + 0.7 * load_factor)
        self.temp_c = 30.0 + 50.0 * load_factor
        self.temp_c = min(self.temp_c, self.max_temp_c + 10.0)

        util = self.utilization_percent()
        self.util_history.append(util)
        if len(self.util_history) > 60:
            self.util_history = self.util_history[-60:]

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
            f"{self.temp_c:.1f} °C, {self.power_w:.1f} W"
        )


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

    state: str = "SPAWNED"
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
        return (
            f"{self.name} [{self.role}]\n"
            f"  State              : {self.state}\n"
            f"  Latency sensitivity: {self.latency_sensitivity}\n"
            f"  Throughput need    : {self.throughput_need}\n"
            f"  Memory intensity   : {self.memory_intensity}\n"
            f"  Reliability need   : {self.reliability_need}\n"
            f"  Demands            : {self.cores_demand:.1f} cores, "
            f"{self.mem_demand_gb:.1f} GB, {self.io_demand:.1f} I/O\n"
            f"  Remaining ticks    : {self.remaining_ticks}\n"
            f"  Retries left       : {self.retries_left}\n"
            f"  Failure probability: {self.failure_prob:.2f}\n"
            f"  Home node          : {self.home_node}\n"
            f"  Assigned           : {asg_str}\n"
            f"  Pinned             : {pin_str}\n"
            f"  Migrations         : {self.migration_count}"
        )


@dataclass
class Node:
    name: str
    sockets: List[CpuSocket] = field(default_factory=list)


@dataclass
class SimulationState:
    nodes: List[Node] = field(default_factory=list)
    workloads: List[Workload] = field(default_factory=list)
    policy: str = "Balanced"
    scheduler_plugin: str = "Balanced"
    tick_count: int = 0
    anomalies: List[str] = field(default_factory=list)


# -----------------------------
# Scheduler plugins
# -----------------------------

class SchedulerPlugin:
    def name(self) -> str:
        return "Base"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int, state: SimulationState) -> float:
        raise NotImplementedError


class BalancedScheduler(SchedulerPlugin):
    def name(self) -> str:
        return "Balanced"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int, state: SimulationState) -> float:
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
            migration_penalty += 3.0

        resource_penalty = 0.0
        if s.cores_used + w.cores_demand > s.cores_total:
            resource_penalty += 100.0
        if s.mem_used_gb + w.mem_demand_gb > s.mem_total_gb:
            resource_penalty += 100.0
        if s.io_used + w.io_demand > s.io_total:
            resource_penalty += 100.0

        score = latency_term + throughput_term + memory_term + reliability_term
        score -= (numa_penalty + migration_penalty + resource_penalty)
        return score


class LatencyFirstScheduler(BalancedScheduler):
    def name(self) -> str:
        return "Latency-first"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int, state: SimulationState) -> float:
        base = super().score(w, s, node_index, socket_index, state)
        return base + 2.0 * w.latency_sensitivity * s.boost_ghz


class PowerSavingScheduler(BalancedScheduler):
    def name(self) -> str:
        return "Power-saving"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int, state: SimulationState) -> float:
        base = super().score(w, s, node_index, socket_index, state)
        power_headroom = max(0.0, s.max_power_w - s.power_w)
        return base + 0.5 * power_headroom


class MLHeuristicScheduler(BalancedScheduler):
    def name(self) -> str:
        return "ML-like heuristic"

    def score(self, w: Workload, s: CpuSocket, node_index: int, socket_index: int, state: SimulationState) -> float:
        base = super().score(w, s, node_index, socket_index, state)
        recent_anomalies = len(state.anomalies[-10:])
        penalty = 0.5 * recent_anomalies
        return base - penalty


SCHEDULER_PLUGINS: Dict[str, SchedulerPlugin] = {
    "Balanced": BalancedScheduler(),
    "Latency-first": LatencyFirstScheduler(),
    "Power-saving": PowerSavingScheduler(),
    "ML-like heuristic": MLHeuristicScheduler(),
}


# -----------------------------
# Assignment / lifecycle / anomalies
# -----------------------------

def assign_workloads(state: SimulationState) -> str:
    if not state.nodes:
        return "No nodes defined."

    plugin = SCHEDULER_PLUGINS.get(state.scheduler_plugin, SCHEDULER_PLUGINS["Balanced"])
    lines = []

    for node in state.nodes:
        for s in node.sockets:
            s.reset_usage()

    for w in state.workloads:
        if w.state not in ("SPAWNED", "RUNNING"):
            continue

        if w.pinned_node is not None and w.pinned_socket is not None:
            node_index = w.pinned_node
            socket_index = w.pinned_socket
            if 0 <= node_index < len(state.nodes) and 0 <= socket_index < len(state.nodes[node_index].sockets):
                s = state.nodes[node_index].sockets[socket_index]
                s.add_usage(w.cores_demand, w.mem_demand_gb, w.io_demand)
                if w.last_assignment is not None and w.last_assignment != (node_index, socket_index):
                    w.migration_count += 1
                w.assigned_node = node_index
                w.assigned_socket = socket_index
                w.last_assignment = (node_index, socket_index)
                lines.append(
                    f"[{state.scheduler_plugin}] Workload '{w.name}' pinned → "
                    f"{state.nodes[node_index].name} / {s.name}"
                )
            continue

        best_score = -1e9
        best_node_idx = None
        best_socket_idx = None

        for ni, node in enumerate(state.nodes):
            for si, s in enumerate(node.sockets):
                score = plugin.score(w, s, ni, si, state)
                if score > best_score:
                    best_score = score
                    best_node_idx = ni
                    best_socket_idx = si

        if best_node_idx is None or best_socket_idx is None:
            lines.append(f"[{state.scheduler_plugin}] Workload '{w.name}' could not be assigned (no capacity).")
            continue

        node = state.nodes[best_node_idx]
        s = node.sockets[best_socket_idx]
        s.add_usage(w.cores_demand, w.mem_demand_gb, w.io_demand)

        if w.last_assignment is not None and w.last_assignment != (best_node_idx, best_socket_idx):
            w.migration_count += 1

        w.assigned_node = best_node_idx
        w.assigned_socket = best_socket_idx
        w.last_assignment = (best_node_idx, best_socket_idx)
        w.state = "RUNNING"

        lines.append(
            f"[{state.scheduler_plugin}] Workload '{w.name}' ({w.role}) → "
            f"{node.name} / {s.name} (score {best_score:.2f})"
        )

    for node in state.nodes:
        for s in node.sockets:
            s.update_thermal_power()

    return "\n".join(lines) if lines else "No workloads to assign."


def summarize_socket_load(state: SimulationState) -> str:
    lines = []
    for ni, node in enumerate(state.nodes):
        lines.append(f"Node {ni}: {node.name}")
        for si, s in enumerate(node.sockets):
            lines.append(
                f"  Socket {si} ({s.name}): "
                f"{s.cores_used:.1f}/{s.cores_total} cores, "
                f"{s.mem_used_gb:.1f}/{s.mem_total_gb:.1f} GB, "
                f"{s.io_used:.1f}/{s.io_total:.1f} I/O, "
                f"{s.temp_c:.1f} °C, {s.power_w:.1f} W, "
                f"util {s.utilization_percent():.1f}%"
            )
    return "\n".join(lines) if lines else "No sockets."


def update_workload_dynamics(w: Workload):
    if w.state not in ("SPAWNED", "RUNNING"):
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
    if w.state not in ("SPAWNED", "RUNNING"):
        return

    w.remaining_ticks -= 1
    if w.remaining_ticks <= 0:
        if random.random() < w.failure_prob:
            w.state = "FAILED"
            if w.retries_left > 0:
                w.retries_left -= 1
                w.state = "SPAWNED"
                w.remaining_ticks = random.randint(10, 30)
            else:
                w.state = "FAILED"
        else:
            w.state = "COMPLETED"


def detect_anomalies(state: SimulationState):
    anomalies = []

    for ni, node in enumerate(state.nodes):
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
        for si, s in enumerate(node.sockets):
            if len(s.util_history) >= 5:
                recent = s.util_history[-5:]
                if max(recent) - min(recent) > 60.0:
                    anomalies.append(
                        f"Utilization spike: Node {ni} / {s.name} recent util range {min(recent):.1f}-{max(recent):.1f}%"
                    )

    if anomalies:
        state.anomalies.extend(anomalies)
        if len(state.anomalies) > 200:
            state.anomalies = state.anomalies[-200:]


def ensure_structured_workloads(state: SimulationState):
    if state.workloads:
        return

    def mk(role, name, home_node):
        if role == "web":
            return Workload(name, 8, 7, 4, 6, role="web",
                            cores_demand=1.0, mem_demand_gb=1.0, io_demand=1.0,
                            remaining_ticks=random.randint(15, 30),
                            home_node=home_node)
        if role == "db":
            return Workload(name, 5, 6, 9, 9, role="db",
                            cores_demand=1.5, mem_demand_gb=4.0, io_demand=1.0,
                            remaining_ticks=random.randint(20, 40),
                            home_node=home_node)
        if role == "ml":
            return Workload(name, 3, 10, 9, 6, role="ml",
                            cores_demand=4.0, mem_demand_gb=6.0, io_demand=2.0,
                            remaining_ticks=random.randint(25, 50),
                            home_node=home_node)
        if role == "backup":
            return Workload(name, 2, 4, 4, 8, role="backup",
                            cores_demand=1.0, mem_demand_gb=2.0, io_demand=3.0,
                            remaining_ticks=random.randint(30, 60),
                            home_node=home_node)
        return Workload(name, 5, 5, 5, 5, role="custom",
                        cores_demand=1.0, mem_demand_gb=1.0, io_demand=1.0,
                        remaining_ticks=random.randint(10, 30),
                        home_node=home_node)

    state.workloads.extend([
        mk("web", "Web Frontend A", 0),
        mk("web", "Web Frontend B", 0),
        mk("db", "DB Primary", 0),
        mk("db", "DB Replica", 1),
        mk("ml", "ML Batch Trainer", 1),
        mk("backup", "Nightly Backup", 1),
    ])


# -----------------------------
# GUI: Models & Widgets
# -----------------------------

class WorkloadTableModel(QtCore.QAbstractTableModel):
    HEADERS = [
        "Name", "Role", "State",
        "Latency", "Throughput", "Memory", "Reliability",
        "Node", "Socket", "Pinned", "Migrations"
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
            rad = angle * 3.14159 / 180.0
            x1 = (tick_radius - 6) * QtCore.qCos(rad)
            y1 = (tick_radius - 6) * QtCore.qSin(rad)
            x2 = tick_radius * QtCore.qCos(rad)
            y2 = tick_radius * QtCore.qSin(rad)
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
        painter.restore()

        painter.save()
        needle_angle = 225 + (self._value * 2.7)
        rad = needle_angle * 3.14159 / 180.0
        needle_len = tick_radius - 10
        needle_pen = QtGui.QPen(QtGui.QColor(220, 80, 60))
        needle_pen.setWidth(4)
        painter.setPen(needle_pen)
        painter.drawLine(
            QtCore.QPointF(0, 0),
            QtCore.QPointF(needle_len * QtCore.qCos(rad), needle_len * QtCore.qSin(rad))
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


# -----------------------------
# GUI: Main window (tactical cockpit)
# -----------------------------

class DualSocketSwarmSimulator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swarm Dual-Socket Simulator v5 – Tactical Cockpit")
        self.resize(1600, 900)

        self.state = SimulationState()
        self._init_default_swarm()

        self.selected_workload_index: Optional[int] = None
        self.current_node_index: int = 0

        self._build_ui()
        self._apply_dark_palette()
        self._refresh_socket_info()
        ensure_structured_workloads(self.state)

        self._run_scheduler_and_update_ui(initial=True)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(800)

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
            ]
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
            ]
        )

        self.state.nodes = [node0, node1]

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Top: node selector + scheduler plugin + policy
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

        # Middle: workloads table + controls + right panel
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

        control_group = QtWidgets.QGroupBox("Workload Controls (Custom)")
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
        self.cores_spin.setRange(0.1, 16.0)
        self.cores_spin.setSingleStep(0.1)
        self.cores_spin.setValue(1.0)

        self.mem_gb_spin = QtWidgets.QDoubleSpinBox()
        self.mem_gb_spin.setRange(0.1, 64.0)
        self.mem_gb_spin.setSingleStep(0.1)
        self.mem_gb_spin.setValue(1.0)

        self.io_spin = QtWidgets.QDoubleSpinBox()
        self.io_spin.setRange(0.1, 10.0)
        self.io_spin.setSingleStep(0.1)
        self.io_spin.setValue(1.0)

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
        control_layout.addRow("Home node:", self.home_node_spin)

        self.add_btn = QtWidgets.QPushButton("Add custom workload")
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
        self.pin_btn = QtWidgets.QPushButton("Pin to current node/socket")
        self.unpin_btn = QtWidgets.QPushButton("Unpin")
        self.pin_btn.clicked.connect(self._on_pin_current)
        self.unpin_btn.clicked.connect(self._on_unpin)
        pin_layout.addWidget(self.pin_btn)
        pin_layout.addWidget(self.unpin_btn)
        right_layout.addLayout(pin_layout)

        self.workload_timeline = TimelineWidget()
        right_layout.addWidget(self.workload_timeline, stretch=3)

        mid_layout.addWidget(right_panel, stretch=4)

        # Bottom: gauges + socket timelines + anomalies + logs
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

        right_splitter.setSizes([150, 300, 200])

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready (swarm real-time mode).")

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

    def _update_anomaly_panel(self):
        self.anomaly_text.setPlainText("\n".join(self.state.anomalies[-50:]))

    def _update_selected_panel(self):
        if self.selected_workload_index is None or self.selected_workload_index >= len(self.state.workloads):
            self.selected_details.setPlainText("Select a workload to see details.")
            self.workload_timeline.setHistory([])
            return

        w = self.state.workloads[self.selected_workload_index]
        self.selected_details.setPlainText(w.describe())
        self.workload_timeline.setHistory(w.history)

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
            remaining_ticks=random.randint(15, 40),
            home_node=home_node,
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
                w = Workload(name, 8, 7, 4, 6, role="web",
                             cores_demand=1.0, mem_demand_gb=1.0, io_demand=1.0,
                             remaining_ticks=random.randint(15, 30),
                             home_node=home_node)
            elif role == "db":
                w = Workload(name, 5, 6, 9, 9, role="db",
                             cores_demand=1.5, mem_demand_gb=4.0, io_demand=1.0,
                             remaining_ticks=random.randint(20, 40),
                             home_node=home_node)
            elif role == "ml":
                w = Workload(name, 3, 10, 9, 6, role="ml",
                             cores_demand=4.0, mem_demand_gb=6.0, io_demand=2.0,
                             remaining_ticks=random.randint(25, 50),
                             home_node=home_node)
            else:
                w = Workload(name, 2, 4, 4, 8, role="backup",
                             cores_demand=1.0, mem_demand_gb=2.0, io_demand=3.0,
                             remaining_ticks=random.randint(30, 60),
                             home_node=home_node)
            self.state.workloads.append(w)

        self._run_scheduler_and_update_ui()

        detect_anomalies(self.state)

        text = self.explanation_text.toPlainText()
        lines = text.splitlines()
        if len(lines) > 400:
            self.explanation_text.setPlainText("\n".join(lines[-250:]))

        self.status_bar.showMessage(f"Real-time tick {self.state.tick_count} processed.")

# -----------------------------
# main
# -----------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = DualSocketSwarmSimulator()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
