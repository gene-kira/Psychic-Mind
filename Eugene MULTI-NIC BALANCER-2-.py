#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network_organism_hud.py

Unified network organism with:
- Cross-platform HAL (Windows/Linux/macOS)
- NIC health sensors (latency/jitter/loss)
- Throughput meters (bytes/sec per NIC)
- Threat heatmap + GPU/NPU anomaly bridge
- AI cortex (heuristic + optional PyTorch/CuPy)
- Swarm mesh (queen/worker)
- NetworkBrain (global / in-out / priority split)
- Gaming QoS hooks (placeholders)
- PySide6 tactical HUD cockpit (dark, neon, cyberpunk)
"""

import sys
import subprocess
import time
import threading
import socket
import json
import platform
import statistics
import random
from enum import Enum

# =========================
#  AUTOLOADER
# =========================

REQUIRED_LIBS = ["psutil"]
def ensure_libs():
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except Exception:
            pass

ensure_libs()
import psutil  # type: ignore

# Optional imports
try:
    from PySide6 import QtWidgets, QtCore, QtGui
    HAVE_QT = True
except Exception:
    HAVE_QT = False

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    HAVE_CUPY = False


# =========================
#  CONFIG
# =========================

SWARM_PORT = 49231
SWARM_BROADCAST_INTERVAL = 3.0
NODE_TTL = 15.0

DEFAULT_PING_TARGET = "8.8.8.8"
PROBE_WINDOW = 20
PROBE_INTERVAL = 1.0
THROUGHPUT_INTERVAL = 1.0

ROLE_QUEEN = "queen"
ROLE_WORKER = "worker"

class RoutingMode(Enum):
    GLOBAL_BEST = "global_best"
    IN_OUT_SPLIT = "in_out_split"
    PRIORITY_SPLIT = "priority_split"


# =========================
#  HAL (CROSS-PLATFORM)
# =========================

class NetworkHAL:
    def __init__(self):
        self.os = platform.system()

    def set_default_route(self, nic):
        if not nic:
            return
        try:
            if self.os == "Windows":
                subprocess.run(
                    ["netsh", "interface", "ipv4", "set", "interface", nic, "metric=5"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif self.os == "Linux":
                subprocess.run(
                    ["ip", "route", "replace", "default", "dev", nic],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif self.os == "Darwin":
                subprocess.run(
                    ["route", "change", "default", nic],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    def set_interface_metric(self, nic, metric):
        if not nic or self.os != "Windows":
            return
        try:
            subprocess.run(
                ["netsh", "interface", "ipv4", "set", "interface", nic, f"metric={metric}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def apply_global_best(self, nic):
        self.set_default_route(nic)

    def apply_in_out_split(self, in_nic, out_nic):
        if self.os == "Windows":
            if in_nic:
                self.set_interface_metric(in_nic, 5)
            if out_nic and out_nic != in_nic:
                self.set_interface_metric(out_nic, 15)
        else:
            if in_nic:
                self.set_default_route(in_nic)

    def apply_priority_split(self, high_prio_nic, low_prio_nic):
        if self.os == "Windows":
            if high_prio_nic:
                self.set_interface_metric(high_prio_nic, 5)
            if low_prio_nic and low_prio_nic != high_prio_nic:
                self.set_interface_metric(low_prio_nic, 25)
        else:
            if high_prio_nic:
                self.set_default_route(high_prio_nic)

    def apply_gaming_qos(self, game_process_names, high_prio_nic):
        # Placeholder for DSCP/WFP/tc/PF integration
        pass


# =========================
#  NIC SENSORS (LATENCY/JITTER/LOSS)
# =========================

class NICProbe:
    def __init__(self, name, addr, ping_target=DEFAULT_PING_TARGET):
        self.name = name
        self.addr = addr
        self.ping_target = ping_target
        self.lat_samples = []
        self.loss_samples = []
        self.jitter_samples = []
        self.lock = threading.Lock()

    def _ping_once(self):
        try:
            if platform.system() == "Windows":
                cmd = ["ping", "-n", "1", "-w", "500", self.ping_target]
            else:
                cmd = ["ping", "-c", "1", "-W", "1", self.ping_target]
            start = time.time()
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            end = time.time()
            if out.returncode != 0:
                return None
            return (end - start) * 1000.0
        except Exception:
            return None

    def run(self, stop_event: threading.Event):
        while not stop_event.is_set():
            lat = self._ping_once()
            with self.lock:
                if lat is None:
                    self.loss_samples.append(1)
                else:
                    self.lat_samples.append(lat)
                    self.loss_samples.append(0)
                if len(self.lat_samples) > PROBE_WINDOW:
                    self.lat_samples.pop(0)
                if len(self.loss_samples) > PROBE_WINDOW:
                    self.loss_samples.pop(0)
                if len(self.lat_samples) > 2:
                    diffs = [
                        abs(self.lat_samples[i] - self.lat_samples[i - 1])
                        for i in range(1, len(self.lat_samples))
                    ]
                    self.jitter_samples = diffs[-10:]
            time.sleep(PROBE_INTERVAL)

    def base_score(self):
        with self.lock:
            if not self.lat_samples:
                return 9999.0
            avg_lat = statistics.mean(self.lat_samples)
            loss = sum(self.loss_samples) / len(self.loss_samples)
            jitter = statistics.mean(self.jitter_samples) if self.jitter_samples else 0.0
            return avg_lat + (loss * 500.0) + jitter


class NICSensorCortex:
    def __init__(self):
        self.nics = self._discover_nics()
        self.probes = {}
        self.threads = []
        self.stop_event = threading.Event()
        self._start_probes()

    def _discover_nics(self):
        nics = {}
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family == psutil.AF_LINK:
                    continue
                if any(tag in name.lower() for tag in ["eth", "en", "ethernet"]):
                    nics[name] = a.address
        return nics

    def _start_probes(self):
        for name, addr in self.nics.items():
            probe = NICProbe(name, addr)
            self.probes[name] = probe
            t = threading.Thread(target=probe.run, args=(self.stop_event,), daemon=True)
            t.start()
            self.threads.append(t)

    def get_scores(self):
        scores = {}
        for name, probe in self.probes.items():
            scores[name] = probe.base_score()
        return scores

    def stop(self):
        self.stop_event.set()


# =========================
#  THROUGHPUT METER
# =========================

class DataFlowMeter:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_counters = psutil.net_io_counters(pernic=True)
        self.throughput = {}  # nic -> {"tx_bps":..., "rx_bps":...}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            time.sleep(THROUGHPUT_INTERVAL)
            now = psutil.net_io_counters(pernic=True)
            with self.lock:
                for nic, counters in now.items():
                    if nic not in self.last_counters:
                        continue
                    prev = self.last_counters[nic]
                    dt = THROUGHPUT_INTERVAL
                    tx_bytes = counters.bytes_sent - prev.bytes_sent
                    rx_bytes = counters.bytes_recv - prev.bytes_recv
                    tx_bps = tx_bytes * 8 / dt
                    rx_bps = rx_bytes * 8 / dt
                    self.throughput[nic] = {
                        "tx_bps": max(tx_bps, 0),
                        "rx_bps": max(rx_bps, 0),
                    }
                self.last_counters = now

    def get_throughput(self):
        with self.lock:
            return dict(self.throughput)

    def stop(self):
        self.stop_event.set()


# =========================
#  THREAT HEATMAP + GPU/NPU BRIDGE
# =========================

class ThreatHeatmap:
    def __init__(self):
        self.nic_threat = {}
        self.lock = threading.Lock()

    def update_threat(self, nic_name: str, score: float):
        with self.lock:
            self.nic_threat[nic_name] = score

    def get_threat(self, nic_name: str) -> float:
        with self.lock:
            return self.nic_threat.get(nic_name, 0.0)

    def apply(self, nic_scores: dict):
        out = {}
        for nic, base in nic_scores.items():
            threat = self.get_threat(nic)
            out[nic] = base + threat * 1000.0
        return out


class GPUAnomalyBridge:
    def __init__(self, heatmap: ThreatHeatmap):
        self.heatmap = heatmap

    def push_scores(self, nic_scores: dict):
        for nic, score in nic_scores.items():
            self.heatmap.update_threat(nic, float(score))


# =========================
#  AI CORTEX (PLUGGABLE)
# =========================

class AICortex:
    def __init__(self):
        self.backend = "heuristic"
        if HAVE_TORCH:
            self.backend = "torch"
        elif HAVE_CUPY:
            self.backend = "cupy"

    def score(self, nic_features: dict):
        if self.backend == "torch":
            return self._score_torch(nic_features)
        elif self.backend == "cupy":
            return self._score_cupy(nic_features)
        else:
            return self._score_heuristic(nic_features)

    def _score_heuristic(self, nic_features: dict):
        out = {}
        for nic, feats in nic_features.items():
            out[nic] = sum(feats)
        return out

    def _score_torch(self, nic_features: dict):
        out = {}
        for nic, feats in nic_features.items():
            x = torch.tensor(feats, dtype=torch.float32)
            out[nic] = float(x.mean().item())
        return out

    def _score_cupy(self, nic_features: dict):
        out = {}
        for nic, feats in nic_features.items():
            x = cp.asarray(feats, dtype=cp.float32)
            out[nic] = float(cp.mean(x).get())
        return out


# =========================
#  SWARM MESH
# =========================

class SwarmMesh:
    def __init__(self, node_id: str, role: str):
        self.node_id = node_id
        self.role = role
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("", SWARM_PORT))
        self.peers = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def broadcast_loop(self, get_local_scores):
        while not self.stop_event.is_set():
            payload = {
                "node_id": self.node_id,
                "role": self.role,
                "ts": time.time(),
                "nic_scores": get_local_scores(),
            }
            data = json.dumps(payload).encode("utf-8")
            try:
                self.sock.sendto(data, ("<broadcast>", SWARM_PORT))
            except Exception:
                pass
            time.sleep(SWARM_BROADCAST_INTERVAL)

    def listen_loop(self):
        self.sock.settimeout(1.0)
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                self._reap_peers()
                continue
            except Exception:
                continue
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            node_id = payload.get("node_id")
            if not node_id or node_id == self.node_id:
                continue
            with self.lock:
                self.peers[node_id] = {
                    "role": payload.get("role", ROLE_WORKER),
                    "last_seen": payload.get("ts", time.time()),
                    "nic_scores": payload.get("nic_scores", {}),
                }

    def _reap_peers(self):
        now = time.time()
        with self.lock:
            dead = [nid for nid, info in self.peers.items()
                    if now - info["last_seen"] > NODE_TTL]
            for nid in dead:
                del self.peers[nid]

    def get_peer_snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.peers))

    def merge_scores(self, local_scores: dict):
        peers = self.get_peer_snapshot()
        agg = {}
        count = {}
        for nic, score in local_scores.items():
            agg[nic] = agg.get(nic, 0.0) + score
            count[nic] = count.get(nic, 0) + 1
        for nid, info in peers.items():
            for nic, score in info.get("nic_scores", {}).items():
                agg[nic] = agg.get(nic, 0.0) + score
                count[nic] = count.get(nic, 0) + 1
        if not agg:
            return local_scores
        return {nic: agg[nic] / count[nic] for nic in agg}

    def infer_queen_choice(self):
        peers = self.get_peer_snapshot()
        queen_scores = None
        for nid, info in peers.items():
            if info.get("role") == ROLE_QUEEN:
                queen_scores = info.get("nic_scores", {})
                break
        if not queen_scores:
            return None
        return min(queen_scores, key=queen_scores.get)

    def stop(self):
        self.stop_event.set()
        try:
            self.sock.close()
        except Exception:
            pass


# =========================
#  NETWORK BRAIN
# =========================

class NetworkBrain:
    def __init__(self, node_id=None, role=ROLE_WORKER,
                 routing_mode: RoutingMode = RoutingMode.GLOBAL_BEST):
        self.node_id = node_id or f"node-{random.randint(1000, 9999)}"
        self.role = role
        self.routing_mode = routing_mode

        self.sensors = NICSensorCortex()
        self.flow = DataFlowMeter()
        self.heatmap = ThreatHeatmap()
        self.gpu_bridge = GPUAnomalyBridge(self.heatmap)
        self.ai_cortex = AICortex()
        self.swarm = SwarmMesh(self.node_id, self.role)
        self.hal = NetworkHAL()

        self.stop_event = threading.Event()
        self.threads = []
        self.brain_state = {}
        self.brain_state_lock = threading.Lock()

    def snapshot_state(self):
        nic_health = self.sensors.get_scores()
        nic_threat = {n: self.heatmap.get_threat(n) for n in nic_health}
        merged = self.heatmap.apply(nic_health)
        throughput = self.flow.get_throughput()
        peers = self.swarm.get_peer_snapshot()
        best, second = self._decide(merged, throughput, peers)
        state = {
            "node_id": self.node_id,
            "role": self.role,
            "routing_mode": self.routing_mode.value,
            "nic_health": nic_health,
            "nic_threat": nic_threat,
            "merged_scores": merged,
            "throughput": throughput,
            "peers": peers,
            "best_nic": best,
            "second_nic": second,
        }
        with self.brain_state_lock:
            self.brain_state = state
        return state

    def get_state(self):
        with self.brain_state_lock:
            return dict(self.brain_state)

    def push_gpu_anomaly_scores(self, nic_scores: dict):
        self.gpu_bridge.push_scores(nic_scores)

    def _decide(self, merged_scores: dict, throughput: dict, peers_snapshot: dict):
        if not merged_scores:
            return None, None
        nic_features = {}
        for nic, score in merged_scores.items():
            tx = throughput.get(nic, {}).get("tx_bps", 0.0)
            rx = throughput.get(nic, {}).get("rx_bps", 0.0)
            nic_features[nic] = [score, tx / 1e6, rx / 1e6]
        ai_scores = self.ai_cortex.score(nic_features)
        combined = {nic: merged_scores[nic] + ai_scores.get(nic, 0.0)
                    for nic in merged_scores}
        consensus = self.swarm.merge_scores(combined)
        sorted_nics = sorted(consensus.items(), key=lambda x: x[1])
        best = sorted_nics[0][0]
        second = sorted_nics[1][0] if len(sorted_nics) > 1 else best
        if self.role == ROLE_WORKER:
            queen_choice = self.swarm.infer_queen_choice()
            if queen_choice and queen_choice in consensus:
                best = queen_choice
        return best, second

    def _routing_control_thread(self):
        while not self.stop_event.is_set():
            state = self.snapshot_state()
            best = state["best_nic"]
            second = state["second_nic"]
            if self.routing_mode == RoutingMode.GLOBAL_BEST:
                self.hal.apply_global_best(best)
            elif self.routing_mode == RoutingMode.IN_OUT_SPLIT:
                self.hal.apply_in_out_split(best, second)
            elif self.routing_mode == RoutingMode.PRIORITY_SPLIT:
                self.hal.apply_priority_split(best, second)
            time.sleep(5.0)

    def _swarm_broadcast_thread(self):
        self.swarm.broadcast_loop(self.sensors.get_scores)

    def _swarm_listen_thread(self):
        self.swarm.listen_loop()

    def run(self):
        t1 = threading.Thread(target=self._swarm_broadcast_thread, daemon=True)
        t2 = threading.Thread(target=self._swarm_listen_thread, daemon=True)
        t3 = threading.Thread(target=self._routing_control_thread, daemon=True)
        self.threads.extend([t1, t2, t3])
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_event.set()
        self.sensors.stop()
        self.flow.stop()
        self.swarm.stop()


# =========================
#  PYSIDE6 TACTICAL HUD COCKPIT
# =========================

if HAVE_QT:
    class NetworkCockpit(QtWidgets.QWidget):
        def __init__(self, brain: NetworkBrain):
            super().__init__()
            self.brain = brain
            self.setWindowTitle("Network Brain Tactical HUD")
            self.resize(1100, 650)
            self._apply_hud_style()
            self._build_ui()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(1000)

        def _apply_hud_style(self):
            palette = self.palette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor(10, 10, 20))
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0, 255, 180))
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor(5, 5, 15))
            palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(15, 15, 30))
            palette.setColor(QtGui.QPalette.Text, QtGui.QColor(0, 255, 180))
            palette.setColor(QtGui.QPalette.Button, QtGui.QColor(20, 20, 40))
            palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(0, 255, 180))
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 200, 255))
            palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
            self.setPalette(palette)
            self.setAutoFillBackground(True)
            self.setStyleSheet("""
                QWidget {
                    font-family: Consolas, "Fira Code", monospace;
                    color: #00FFB4;
                    background-color: #050510;
                }
                QTableWidget {
                    gridline-color: #00FFB4;
                    selection-background-color: #00C8FF;
                    selection-color: #000000;
                }
                QHeaderView::section {
                    background-color: #101020;
                    color: #00FFB4;
                    border: 1px solid #00FFB4;
                }
                QTextEdit {
                    border: 1px solid #00FFB4;
                }
                QLabel {
                    font-size: 12px;
                }
            """)

        def _build_ui(self):
            layout = QtWidgets.QVBoxLayout(self)

            self.info_label = QtWidgets.QLabel()
            layout.addWidget(self.info_label)

            self.mode_label = QtWidgets.QLabel()
            layout.addWidget(self.mode_label)

            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(7)
            self.table.setHorizontalHeaderLabels(
                ["NIC", "Health", "Threat", "Merged", "TX Mbps", "RX Mbps", "Role"]
            )
            self.table.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(self.table)

            self.peers_view = QtWidgets.QTextEdit()
            self.peers_view.setReadOnly(True)
            layout.addWidget(self.peers_view)

        def refresh(self):
            state = self.brain.get_state()
            nic_health = state.get("nic_health", {})
            nic_threat = state.get("nic_threat", {})
            merged = state.get("merged_scores", {})
            throughput = state.get("throughput", {})
            best = state.get("best_nic")
            second = state.get("second_nic")
            role = state.get("role")
            mode = state.get("routing_mode")

            self.info_label.setText(
                f"NODE {state.get('node_id')} :: ROLE {role} :: BEST {best} :: SECOND {second}"
            )
            self.mode_label.setText(
                f"MODE {mode} :: AI BACKEND {self.brain.ai_cortex.backend.upper()}"
            )

            self.table.setRowCount(len(nic_health))
            for row, nic in enumerate(sorted(nic_health.keys())):
                tx_mbps = throughput.get(nic, {}).get("tx_bps", 0.0) / 1e6
                rx_mbps = throughput.get(nic, {}).get("rx_bps", 0.0) / 1e6
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(nic))
                self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{nic_health[nic]:.1f}"))
                self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{nic_threat.get(nic, 0.0):.2f}"))
                self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{merged.get(nic, 0.0):.1f}"))
                self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{tx_mbps:.2f}"))
                self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{rx_mbps:.2f}"))
                self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(role))

            peers = state.get("peers", {})
            self.peers_view.setPlainText(json.dumps(peers, indent=2))


# =========================
#  MAIN
# =========================

def main():
    role = ROLE_QUEEN  # or ROLE_WORKER
    mode = RoutingMode.PRIORITY_SPLIT  # gaming-friendly default

    brain = NetworkBrain(role=role, routing_mode=mode)
    brain.run()

    if HAVE_QT:
        app = QtWidgets.QApplication(sys.argv)
        cockpit = NetworkCockpit(brain)
        cockpit.show()
        app.exec()
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    brain.stop()

if __name__ == "__main__":
    main()
