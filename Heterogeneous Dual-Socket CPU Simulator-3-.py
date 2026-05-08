#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heterogeneous Dual-Socket CPU Simulator v3 (Real-Time, Structured Enterprise Mode)

- One "Desktop" CPU socket
- One "Xeon" CPU socket
- Workloads with traits (latency, throughput, memory, reliability)
- Structured enterprise workload types: Web, DB, ML, Backup (+ custom)
- Policy-based scheduler (Balanced / Latency / Throughput / Memory / Reliability)
- Real-time updates via QTimer (workloads fluctuate, scheduler re-runs)
- Industrial-style dual gauges (one per socket) with needle + digital readout
- PySide6 GUI cockpit
"""

import sys
import random
from dataclasses import dataclass, field
from typing import List, Optional

from PySide6 import QtWidgets, QtCore, QtGui

# -----------------------------
# Domain model
# -----------------------------

@dataclass
class CpuSocket:
    name: str
    kind: str  # "desktop" or "xeon"
    cores: int
    base_ghz: float
    boost_ghz: float
    cache_mb: float
    ecc: bool
    tdp_w: int
    reliability_weight: float
    latency_weight: float
    throughput_weight: float
    memory_weight: float

    def describe(self) -> str:
        ecc_str = "ECC" if self.ecc else "non-ECC"
        return (
            f"{self.name} ({self.kind})\n"
            f"  Cores       : {self.cores}\n"
            f"  Base/Boost  : {self.base_ghz:.2f}/{self.boost_ghz:.2f} GHz\n"
            f"  Cache       : {self.cache_mb:.1f} MB\n"
            f"  Memory      : {ecc_str}\n"
            f"  TDP         : {self.tdp_w} W\n"
            f"  Weights     : latency={self.latency_weight}, "
            f"throughput={self.throughput_weight}, "
            f"memory={self.memory_weight}, "
            f"reliability={self.reliability_weight}"
        )


@dataclass
class Workload:
    name: str
    latency_sensitivity: int   # 1-10
    throughput_need: int       # 1-10
    memory_intensity: int      # 1-10
    reliability_need: int      # 1-10
    role: str = "custom"       # "web", "db", "ml", "backup", "custom"
    assigned_socket: Optional[int] = None  # 0 or 1

    def describe(self) -> str:
        return (
            f"{self.name} [{self.role}]\n"
            f"  Latency sensitivity : {self.latency_sensitivity}\n"
            f"  Throughput need     : {self.throughput_need}\n"
            f"  Memory intensity    : {self.memory_intensity}\n"
            f"  Reliability need    : {self.reliability_need}\n"
            f"  Assigned socket     : {self.assigned_socket}"
        )


@dataclass
class SimulationState:
    sockets: List[CpuSocket] = field(default_factory=list)
    workloads: List[Workload] = field(default_factory=list)
    policy: str = "Balanced"  # default


# -----------------------------
# Scheduler logic
# -----------------------------

def score_workload_on_socket(w: Workload, s: CpuSocket, policy: str) -> float:
    """
    Scoring model with policy bias:
    - Latency-sensitive workloads like high boost clocks and latency_weight
    - Throughput workloads like cores and throughput_weight
    - Memory-intensive workloads like cache and memory_weight (and ECC)
    - Reliability workloads like reliability_weight and ECC
    Policy modifies emphasis.
    """
    # Base contributions
    latency_term = w.latency_sensitivity * s.boost_ghz * s.latency_weight
    throughput_term = w.throughput_need * s.cores * s.throughput_weight
    mem_factor = s.cache_mb * (1.2 if s.ecc else 1.0)
    memory_term = w.memory_intensity * mem_factor * s.memory_weight
    rel_factor = s.reliability_weight * (1.3 if s.ecc else 1.0)
    reliability_term = w.reliability_need * rel_factor

    # Policy multipliers
    pl = policy.lower()
    latency_mul = 1.0
    throughput_mul = 1.0
    memory_mul = 1.0
    reliability_mul = 1.0

    if "latency" in pl:
        latency_mul = 1.8
        throughput_mul = 0.9
        memory_mul = 0.9
        reliability_mul = 0.9
    elif "throughput" in pl:
        throughput_mul = 1.8
        latency_mul = 0.9
        memory_mul = 1.0
        reliability_mul = 1.0
    elif "memory" in pl:
        memory_mul = 1.8
        latency_mul = 1.0
        throughput_mul = 1.0
        reliability_mul = 1.0
    elif "reliability" in pl:
        reliability_mul = 1.8
        latency_mul = 0.9
        throughput_mul = 0.9
        memory_mul = 1.0
    # Balanced: all 1.0

    score = (
        latency_term * latency_mul +
        throughput_term * throughput_mul +
        memory_term * memory_mul +
        reliability_term * reliability_mul
    )
    return score


def assign_workloads(state: SimulationState) -> str:
    """
    Assign each workload to the best socket based on score.
    Returns a textual explanation.
    """
    if len(state.sockets) < 2:
        return "Need at least two sockets to simulate."

    s0, s1 = state.sockets[0], state.sockets[1]
    lines = []
    for w in state.workloads:
        score0 = score_workload_on_socket(w, s0, state.policy)
        score1 = score_workload_on_socket(w, s1, state.policy)
        if score0 >= score1:
            w.assigned_socket = 0
            chosen = s0
            other = s1
            chosen_score = score0
            other_score = score1
        else:
            w.assigned_socket = 1
            chosen = s1
            other = s0
            chosen_score = score1
            other_score = score0

        lines.append(
            f"[{state.policy}] Workload '{w.name}' ({w.role}) → {chosen.name} "
            f"(score {chosen_score:.2f} vs {other.name} {other_score:.2f})"
        )
    return "\n".join(lines) if lines else "No workloads to assign."


def summarize_socket_load(state: SimulationState) -> str:
    if len(state.sockets) < 2:
        return "No sockets defined."

    s0, s1 = state.sockets[0], state.sockets[1]
    count0 = sum(1 for w in state.workloads if w.assigned_socket == 0)
    count1 = sum(1 for w in state.workloads if w.assigned_socket == 1)

    lines = []
    lines.append(f"{s0.name}: {count0} workloads assigned.")
    lines.append(f"{s1.name}: {count1} workloads assigned.")
    return "\n".join(lines)


def compute_socket_utilization(state: SimulationState, socket_index: int) -> float:
    """
    Simple utilization metric per socket:
    - Sum of workload "weight" on that socket / total possible weight.
    - Returns 0..100 (%).
    """
    if not state.workloads:
        return 0.0

    total_weight = 0.0
    socket_weight = 0.0
    for w in state.workloads:
        w_weight = (
            w.latency_sensitivity +
            w.throughput_need +
            w.memory_intensity +
            w.reliability_need
        )  # max 40
        total_weight += w_weight
        if w.assigned_socket == socket_index:
            socket_weight += w_weight

    if total_weight <= 0:
        return 0.0
    util = (socket_weight / total_weight) * 100.0
    if util > 100.0:
        util = 100.0
    return util


# -----------------------------
# Structured enterprise workload dynamics
# -----------------------------

def _clamp(v: int, lo: int = 1, hi: int = 10) -> int:
    return max(lo, min(hi, v))


def update_workload_dynamics(w: Workload):
    """
    Structured enterprise mode:
    - web: latency + throughput fluctuate
    - db: memory + reliability dominate
    - ml: throughput + memory heavy, bursty
    - backup: mostly idle, occasional heavy reliability/throughput spikes
    - custom: small random jitter
    """
    role = w.role.lower()
    # Small random jitter helper
    def jitter(attr: str, base_delta: int = 1):
        val = getattr(w, attr)
        val += random.randint(-base_delta, base_delta)
        setattr(w, attr, _clamp(val))

    if role == "web":
        # Web: latency & throughput fluctuate, memory moderate, reliability medium
        jitter("latency_sensitivity", 2)
        jitter("throughput_need", 2)
        jitter("memory_intensity", 1)
        jitter("reliability_need", 1)
        # Bias back toward typical web profile
        w.latency_sensitivity = _clamp(int(0.7 * w.latency_sensitivity + 0.3 * 8))
        w.throughput_need = _clamp(int(0.7 * w.throughput_need + 0.3 * 7))
        w.memory_intensity = _clamp(int(0.7 * w.memory_intensity + 0.3 * 4))
        w.reliability_need = _clamp(int(0.7 * w.reliability_need + 0.3 * 6))

    elif role == "db":
        # DB: memory + reliability heavy, latency moderate, throughput medium
        jitter("memory_intensity", 2)
        jitter("reliability_need", 2)
        jitter("latency_sensitivity", 1)
        jitter("throughput_need", 1)
        w.latency_sensitivity = _clamp(int(0.7 * w.latency_sensitivity + 0.3 * 5))
        w.throughput_need = _clamp(int(0.7 * w.throughput_need + 0.3 * 6))
        w.memory_intensity = _clamp(int(0.7 * w.memory_intensity + 0.3 * 9))
        w.reliability_need = _clamp(int(0.7 * w.reliability_need + 0.3 * 9))

    elif role == "ml":
        # ML: throughput + memory very high, latency low, reliability medium
        jitter("throughput_need", 3)
        jitter("memory_intensity", 3)
        jitter("latency_sensitivity", 1)
        jitter("reliability_need", 1)
        w.latency_sensitivity = _clamp(int(0.7 * w.latency_sensitivity + 0.3 * 3))
        w.throughput_need = _clamp(int(0.7 * w.throughput_need + 0.3 * 10))
        w.memory_intensity = _clamp(int(0.7 * w.memory_intensity + 0.3 * 9))
        w.reliability_need = _clamp(int(0.7 * w.reliability_need + 0.3 * 6))

    elif role == "backup":
        # Backup: mostly low, occasional spikes
        if random.random() < 0.15:
            # Spike event
            w.throughput_need = _clamp(w.throughput_need + random.randint(3, 5))
            w.memory_intensity = _clamp(w.memory_intensity + random.randint(2, 4))
            w.reliability_need = _clamp(w.reliability_need + random.randint(2, 4))
        else:
            # Decay toward low baseline
            w.latency_sensitivity = _clamp(int(0.7 * w.latency_sensitivity + 0.3 * 2))
            w.throughput_need = _clamp(int(0.7 * w.throughput_need + 0.3 * 4))
            w.memory_intensity = _clamp(int(0.7 * w.memory_intensity + 0.3 * 4))
            w.reliability_need = _clamp(int(0.7 * w.reliability_need + 0.3 * 8))

    else:
        # custom: gentle jitter only
        jitter("latency_sensitivity", 1)
        jitter("throughput_need", 1)
        jitter("memory_intensity", 1)
        jitter("reliability_need", 1)


def ensure_structured_workloads(state: SimulationState):
    """
    Ensure we have a baseline set of structured enterprise workloads.
    Only called when there are no workloads yet.
    """
    if state.workloads:
        return

    state.workloads.extend([
        Workload("Web Frontend A", 8, 7, 4, 6, role="web"),
        Workload("Web Frontend B", 7, 7, 4, 6, role="web"),
        Workload("DB Primary", 5, 6, 9, 9, role="db"),
        Workload("DB Replica", 4, 5, 8, 8, role="db"),
        Workload("ML Batch Trainer", 3, 10, 9, 6, role="ml"),
        Workload("Nightly Backup", 2, 4, 4, 8, role="backup"),
    ])


# -----------------------------
# GUI: Workload table model
# -----------------------------

class WorkloadTableModel(QtCore.QAbstractTableModel):
    HEADERS = ["Name", "Role", "Latency", "Throughput", "Memory", "Reliability", "Assigned Socket"]

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
                return w.latency_sensitivity
            elif col == 3:
                return w.throughput_need
            elif col == 4:
                return w.memory_intensity
            elif col == 5:
                return w.reliability_need
            elif col == 6:
                if w.assigned_socket is None:
                    return "-"
                return str(w.assigned_socket)
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


# -----------------------------
# GUI: Industrial Gauge Widget
# -----------------------------

class GaugeWidget(QtWidgets.QWidget):
    """
    Industrial-style circular gauge:
    - Thick metallic ring
    - Needle
    - Digital readout
    """
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._value = 0.0  # 0..100
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

        # Centered square
        rect = QtCore.QRect((self.width() - side) // 2, (self.height() - side) // 2, side, side)
        painter.translate(rect.center())
        radius = side / 2 - 10

        # Background
        painter.save()
        painter.setBrush(QtGui.QColor(20, 20, 20))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPoint(0, 0), int(radius), int(radius))
        painter.restore()

        # Outer metallic ring
        painter.save()
        ring_pen = QtGui.QPen(QtGui.QColor(120, 120, 120))
        ring_pen.setWidth(14)
        painter.setPen(ring_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPoint(0, 0), int(radius - 7), int(radius - 7))
        painter.restore()

        # Tick marks
        painter.save()
        tick_pen = QtGui.QPen(QtGui.QColor(80, 80, 80))
        tick_pen.setWidth(2)
        painter.setPen(tick_pen)
        tick_radius = radius - 16
        for i in range(0, 101, 10):
            angle = (225 + (i * 2.7))  # 225° to ~495° (sweep)
            rad = angle * 3.14159 / 180.0
            x1 = (tick_radius - 6) * QtCore.qCos(rad)
            y1 = (tick_radius - 6) * QtCore.qSin(rad)
            x2 = tick_radius * QtCore.qCos(rad)
            y2 = tick_radius * QtCore.qSin(rad)
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
        painter.restore()

        # Needle
        painter.save()
        needle_angle = 225 + (self._value * 2.7)  # 0..100 mapped to 225..495
        rad = needle_angle * 3.14159 / 180.0
        needle_len = tick_radius - 10
        needle_pen = QtGui.QPen(QtGui.QColor(220, 80, 60))
        needle_pen.setWidth(4)
        painter.setPen(needle_pen)
        painter.drawLine(
            QtCore.QPointF(0, 0),
            QtCore.QPointF(needle_len * QtCore.qCos(rad), needle_len * QtCore.qSin(rad))
        )
        # Needle hub
        painter.setBrush(QtGui.QColor(200, 200, 200))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPointF(0, 0), 6, 6)
        painter.restore()

        # Digital readout
        painter.save()
        text_rect = QtCore.QRectF(-radius + 20, 0, 2 * (radius - 20), 40)
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(230, 230, 230))
        painter.drawText(text_rect, QtCore.Qt.AlignCenter, f"{self._value:5.1f} %")
        painter.restore()

        # Label
        painter.save()
        label_rect = QtCore.QRectF(-radius + 20, -radius + 10, 2 * (radius - 20), 30)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(180, 180, 180))
        painter.drawText(label_rect, QtCore.Qt.AlignCenter, self._label)
        painter.restore()


# -----------------------------
# GUI: Main window
# -----------------------------

class DualSocketSimulator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heterogeneous Dual-Socket CPU Simulator – Real-Time Enterprise Cockpit")
        self.resize(1200, 800)

        self.state = SimulationState()
        self._init_default_sockets()

        self._build_ui()
        self._apply_dark_palette()
        self._refresh_socket_info()

        # Ensure baseline structured workloads
        ensure_structured_workloads(self.state)
        self._run_scheduler_and_update_ui(initial=True)

        # Real-time timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(700)  # ms

    def _init_default_sockets(self):
        desktop = CpuSocket(
            name="Socket 0 – Desktop CPU",
            kind="desktop",
            cores=8,
            base_ghz=3.6,
            boost_ghz=5.0,
            cache_mb=16.0,
            ecc=False,
            tdp_w=125,
            reliability_weight=0.8,
            latency_weight=1.5,
            throughput_weight=1.0,
            memory_weight=0.8,
        )
        xeon = CpuSocket(
            name="Socket 1 – Xeon CPU",
            kind="xeon",
            cores=16,
            base_ghz=2.6,
            boost_ghz=3.8,
            cache_mb=24.0,
            ecc=True,
            tdp_w=165,
            reliability_weight=1.5,
            latency_weight=0.9,
            throughput_weight=1.4,
            memory_weight=1.3,
        )
        self.state.sockets = [desktop, xeon]

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Top: socket descriptions + policy
        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout)

        socket_group = QtWidgets.QGroupBox("Sockets")
        socket_layout = QtWidgets.QHBoxLayout(socket_group)

        self.socket0_text = QtWidgets.QPlainTextEdit()
        self.socket0_text.setReadOnly(True)
        self.socket1_text = QtWidgets.QPlainTextEdit()
        self.socket1_text.setReadOnly(True)

        socket_layout.addWidget(self.socket0_text)
        socket_layout.addWidget(self.socket1_text)

        top_layout.addWidget(socket_group, stretch=4)

        policy_group = QtWidgets.QGroupBox("Scheduler Policy")
        policy_layout = QtWidgets.QVBoxLayout(policy_group)

        self.policy_combo = QtWidgets.QComboBox()
        self.policy_combo.addItems([
            "Balanced",
            "Latency-first",
            "Throughput-first",
            "Memory-first",
            "Reliability-first",
        ])
        self.policy_combo.currentIndexChanged.connect(self._on_policy_changed)
        policy_layout.addWidget(QtWidgets.QLabel("Policy mode:"))
        policy_layout.addWidget(self.policy_combo)

        top_layout.addWidget(policy_group, stretch=1)

        # Middle: workloads table + controls
        mid_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(mid_layout, stretch=3)

        # Workload table
        self.workload_model = WorkloadTableModel(self.state)
        self.workload_table = QtWidgets.QTableView()
        self.workload_table.setModel(self.workload_model)
        self.workload_table.horizontalHeader().setStretchLastSection(True)
        self.workload_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.workload_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        mid_layout.addWidget(self.workload_table, stretch=3)

        # Controls
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

        control_layout.addRow("Name:", self.name_edit)
        control_layout.addRow("Latency sensitivity (1-10):", self.latency_spin)
        control_layout.addRow("Throughput need (1-10):", self.throughput_spin)
        control_layout.addRow("Memory intensity (1-10):", self.memory_spin)
        control_layout.addRow("Reliability need (1-10):", self.reliability_spin)

        self.add_btn = QtWidgets.QPushButton("Add custom workload")
        self.add_btn.clicked.connect(self._on_add_workload)
        control_layout.addRow(self.add_btn)

        self.assign_btn = QtWidgets.QPushButton("Force re-assign now")
        self.assign_btn.clicked.connect(self._on_assign_button)
        control_layout.addRow(self.assign_btn)

        mid_layout.addWidget(control_group, stretch=2)

        # Bottom: gauges + explanation + load summary
        bottom_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(bottom_layout, stretch=3)

        # Gauges
        gauge_group = QtWidgets.QGroupBox("Per-Socket Utilization (Real-Time)")
        gauge_layout = QtWidgets.QHBoxLayout(gauge_group)

        self.gauge0 = GaugeWidget("Socket 0 – Desktop")
        self.gauge1 = GaugeWidget("Socket 1 – Xeon")

        gauge_layout.addWidget(self.gauge0)
        gauge_layout.addWidget(self.gauge1)

        bottom_layout.addWidget(gauge_group, stretch=3)

        # Explanation + load summary
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        bottom_layout.addWidget(right_splitter, stretch=2)

        self.explanation_text = QtWidgets.QPlainTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setPlaceholderText("Scheduler decisions will appear here.")
        right_splitter.addWidget(self.explanation_text)

        self.load_text = QtWidgets.QPlainTextEdit()
        self.load_text.setReadOnly(True)
        self.load_text.setPlaceholderText("Socket load summary will appear here.")
        right_splitter.addWidget(self.load_text)

        right_splitter.setSizes([300, 200])

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready (real-time mode).")

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
        if len(self.state.sockets) >= 1:
            self.socket0_text.setPlainText(self.state.sockets[0].describe())
        if len(self.state.sockets) >= 2:
            self.socket1_text.setPlainText(self.state.sockets[1].describe())

    def _update_gauges(self):
        util0 = compute_socket_utilization(self.state, 0)
        util1 = compute_socket_utilization(self.state, 1)
        self.gauge0.setValue(util0)
        self.gauge1.setValue(util1)

    def _run_scheduler_and_update_ui(self, initial: bool = False):
        explanation = assign_workloads(self.state)
        self.workload_model.refresh()
        if initial:
            self.explanation_text.setPlainText(explanation)
        else:
            # Append to keep a running log
            existing = self.explanation_text.toPlainText()
            if existing:
                self.explanation_text.setPlainText(existing + "\n" + explanation)
            else:
                self.explanation_text.setPlainText(explanation)
        self.load_text.setPlainText(summarize_socket_load(self.state))
        self._update_gauges()

    # ----------------- Event handlers -----------------

    def _on_add_workload(self):
        name = self.name_edit.text().strip()
        if not name:
            name = f"Custom Workload {len(self.state.workloads) + 1}"
        w = Workload(
            name=name,
            latency_sensitivity=self.latency_spin.value(),
            throughput_need=self.throughput_spin.value(),
            memory_intensity=self.memory_spin.value(),
            reliability_need=self.reliability_spin.value(),
            role="custom",
        )
        self.workload_model.add_workload(w)
        self.status_bar.showMessage(f"Added custom workload '{w.name}'.")
        self._run_scheduler_and_update_ui()

    def _on_assign_button(self):
        self._run_scheduler_and_update_ui()
        self.status_bar.showMessage("Manual re-assignment executed.")

    def _on_policy_changed(self, idx: int):
        self.state.policy = self.policy_combo.currentText()
        self.status_bar.showMessage(f"Policy changed to: {self.state.policy}")
        self._run_scheduler_and_update_ui()

    def _on_tick(self):
        # Real-time tick: update workload dynamics, then re-run scheduler
        for w in self.state.workloads:
            update_workload_dynamics(w)

        # Occasionally spawn a new structured workload to keep things interesting
        if random.random() < 0.05:
            new_role = random.choice(["web", "db", "ml", "backup"])
            base_names = {
                "web": "Web Frontend",
                "db": "DB Node",
                "ml": "ML Worker",
                "backup": "Backup Task",
            }
            name = f"{base_names[new_role]} #{random.randint(2, 99)}"
            # Rough archetype seeds
            if new_role == "web":
                w = Workload(name, 8, 7, 4, 6, role="web")
            elif new_role == "db":
                w = Workload(name, 5, 6, 9, 9, role="db")
            elif new_role == "ml":
                w = Workload(name, 3, 10, 9, 6, role="ml")
            else:
                w = Workload(name, 2, 4, 4, 8, role="backup")
            self.state.workloads.append(w)

        self._run_scheduler_and_update_ui()

        # Keep log from growing unbounded: trim if too long
        text = self.explanation_text.toPlainText()
        lines = text.splitlines()
        if len(lines) > 300:
            self.explanation_text.setPlainText("\n".join(lines[-200:]))

        self.status_bar.showMessage("Real-time tick processed.")

# -----------------------------
# main
# -----------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = DualSocketSimulator()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
