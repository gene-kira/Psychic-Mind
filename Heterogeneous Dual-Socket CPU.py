#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heterogeneous Dual-Socket CPU Simulator
- One "Desktop" CPU socket
- One "Xeon" CPU socket
- Workloads with traits (latency, throughput, memory, reliability)
- Simple scoring-based scheduler
- PySide6 GUI cockpit
"""

import sys
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
    assigned_socket: Optional[int] = None  # 0 or 1

    def describe(self) -> str:
        return (
            f"{self.name}\n"
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


# -----------------------------
# Scheduler logic
# -----------------------------

def score_workload_on_socket(w: Workload, s: CpuSocket) -> float:
    """
    Simple scoring model:
    - Latency-sensitive workloads like high boost clocks and latency_weight
    - Throughput workloads like cores and throughput_weight
    - Memory-intensive workloads like cache and memory_weight (and ECC)
    - Reliability workloads like reliability_weight and ECC
    """
    score = 0.0

    # Latency: prefer higher boost and latency_weight
    score += w.latency_sensitivity * s.boost_ghz * s.latency_weight

    # Throughput: prefer more cores and throughput_weight
    score += w.throughput_need * s.cores * s.throughput_weight

    # Memory: prefer more cache and ECC if memory_weight is high
    mem_factor = s.cache_mb
    if s.ecc:
        mem_factor *= 1.2
    score += w.memory_intensity * mem_factor * s.memory_weight

    # Reliability: prefer ECC and reliability_weight
    rel_factor = s.reliability_weight
    if s.ecc:
        rel_factor *= 1.3
    score += w.reliability_need * rel_factor

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
        score0 = score_workload_on_socket(w, s0)
        score1 = score_workload_on_socket(w, s1)
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
            f"Workload '{w.name}' → {chosen.name} "
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


# -----------------------------
# GUI
# -----------------------------

class WorkloadTableModel(QtCore.QAbstractTableModel):
    HEADERS = ["Name", "Latency", "Throughput", "Memory", "Reliability", "Assigned Socket"]

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
                return w.latency_sensitivity
            elif col == 2:
                return w.throughput_need
            elif col == 3:
                return w.memory_intensity
            elif col == 4:
                return w.reliability_need
            elif col == 5:
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


class DualSocketSimulator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heterogeneous Dual-Socket CPU Simulator")
        self.resize(1100, 700)

        self.state = SimulationState()
        self._init_default_sockets()

        self._build_ui()
        self._apply_dark_palette()
        self._refresh_socket_info()

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

        # Top: socket descriptions
        socket_group = QtWidgets.QGroupBox("Sockets")
        socket_layout = QtWidgets.QHBoxLayout(socket_group)

        self.socket0_text = QtWidgets.QPlainTextEdit()
        self.socket0_text.setReadOnly(True)
        self.socket1_text = QtWidgets.QPlainTextEdit()
        self.socket1_text.setReadOnly(True)

        socket_layout.addWidget(self.socket0_text)
        socket_layout.addWidget(self.socket1_text)

        main_layout.addWidget(socket_group)

        # Middle: workloads table + controls
        mid_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(mid_layout)

        # Workload table
        self.workload_model = WorkloadTableModel(self.state)
        self.workload_table = QtWidgets.QTableView()
        self.workload_table.setModel(self.workload_model)
        self.workload_table.horizontalHeader().setStretchLastSection(True)
        self.workload_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.workload_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        mid_layout.addWidget(self.workload_table, stretch=3)

        # Controls
        control_group = QtWidgets.QGroupBox("Workload Controls")
        control_layout = QtWidgets.QFormLayout(control_group)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Web API, DB, ML job")

        self.latency_spin = QtWidgets.QSpinBox()
        self.latency_spin.setRange(1, 10)
        self.latency_spin.setValue(7)

        self.throughput_spin = QtWidgets.QSpinBox()
        self.throughput_spin.setRange(1, 10)
        self.throughput_spin.setValue(7)

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

        self.add_btn = QtWidgets.QPushButton("Add workload")
        self.add_btn.clicked.connect(self._on_add_workload)
        control_layout.addRow(self.add_btn)

        self.assign_btn = QtWidgets.QPushButton("Auto-assign workloads")
        self.assign_btn.clicked.connect(self._on_assign)
        control_layout.addRow(self.assign_btn)

        mid_layout.addWidget(control_group, stretch=2)

        # Bottom: explanation + load summary
        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(bottom_splitter, stretch=2)

        self.explanation_text = QtWidgets.QPlainTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setPlaceholderText("Scheduler decisions will appear here.")
        bottom_splitter.addWidget(self.explanation_text)

        self.load_text = QtWidgets.QPlainTextEdit()
        self.load_text.setReadOnly(True)
        self.load_text.setPlaceholderText("Socket load summary will appear here.")
        bottom_splitter.addWidget(self.load_text)

        bottom_splitter.setSizes([700, 400])

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready.")

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

    def _on_add_workload(self):
        name = self.name_edit.text().strip()
        if not name:
            name = f"Workload {len(self.state.workloads) + 1}"
        w = Workload(
            name=name,
            latency_sensitivity=self.latency_spin.value(),
            throughput_need=self.throughput_spin.value(),
            memory_intensity=self.memory_spin.value(),
            reliability_need=self.reliability_spin.value(),
        )
        self.workload_model.add_workload(w)
        self.status_bar.showMessage(f"Added workload '{w.name}'.")

    def _on_assign(self):
        explanation = assign_workloads(self.state)
        self.workload_model.refresh()
        self.explanation_text.setPlainText(explanation)
        self.load_text.setPlainText(summarize_socket_load(self.state))
        self.status_bar.showMessage("Workloads assigned.")

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
