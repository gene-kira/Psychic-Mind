#!/usr/bin/env python3
import sys
import subprocess
import time
import threading
import queue
import json
import os
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
from collections import deque

# =========================
# 0. Auto-loader for libs
# =========================

REQUIRED_LIBS = [
    ("psutil", "psutil"),
    ("PySide6", "PySide6"),
    ("requests", "requests"),
]

def ensure_libs():
    for mod_name, pip_name in REQUIRED_LIBS:
        try:
            __import__(mod_name)
        except ImportError:
            print(f"[BORG] Missing {mod_name}, installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            __import__(mod_name)

ensure_libs()

import psutil
from PySide6 import QtWidgets, QtCore
import requests

# Optional GPU libs
GPU_LIBS = []
for name in ["torch", "cupy"]:
    try:
        __import__(name)
        GPU_LIBS.append(name)
    except ImportError:
        pass

# =========================
# 1. Message bus & events
# =========================

@dataclass
class BorgEvent:
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

class MessageBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[BorgEvent], None]]] = {}
        self.queue: "queue.Queue[BorgEvent]" = queue.Queue()
        self._running = False

    def publish(self, event: BorgEvent):
        self.queue.put(event)

    def subscribe(self, event_type: str, handler: Callable[[BorgEvent], None]):
        self.subscribers.setdefault(event_type, []).append(handler)

    def start(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self._running:
            try:
                ev = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            for h in self.subscribers.get(ev.type, []):
                try:
                    h(ev)
                except Exception as e:
                    print(f"[BORG][BUS] Handler error for {ev.type}: {e}")

# =========================
# 2. Config
# =========================

@dataclass
class BorgConfig:
    node_id: str = field(default_factory=lambda: f"node-{os.name}-{os.getpid()}")
    swarm_endpoint: Optional[str] = None  # e.g. "http://localhost:5000/swarm"
    cache_enabled: bool = True
    anomaly_enabled: bool = True
    telemetry_interval_sec: float = 2.0
    ai_decision_interval_sec: float = 5.0
    history_len: int = 5  # sequence memory length

# =========================
# 3. Learned policy with sequence memory (reboot memory)
# =========================

class LearnedPolicy:
    ACTIONS = ["NO_OP", "PREFETCH_COMMON_FILES", "THROTTLE_HEAVY_PROCS"]
    MODEL_PATH = Path.home() / ".borg_queen_policy_seq.json"

    def __init__(self, history_len: int = 5, lr: float = 0.01):
        self.lr = lr
        self.history_len = history_len
        self.dim = history_len * 2 + 3
        self.weights = {a: [0.0] * self.dim for a in self.ACTIONS}
        self._load()

    def _features(self, cpu_seq: List[float], mem_seq: List[float], hour: int) -> list:
        cpu_seq = (cpu_seq + [cpu_seq[-1]] * self.history_len)[:self.history_len] if cpu_seq else [0.0] * self.history_len
        mem_seq = (mem_seq + [mem_seq[-1]] * self.history_len)[:self.history_len] if mem_seq else [0.0] * self.history_len
        cpu_norm = [c / 100.0 for c in cpu_seq]
        mem_norm = [m / 100.0 for m in mem_seq]
        hour_norm = hour / 23.0 if hour > 0 else 0.0
        is_rush = 1.0 if hour in (8, 9, 17, 18) else 0.0
        is_off = 1.0 if 0 <= hour <= 5 else 0.0
        return cpu_norm + mem_norm + [hour_norm, is_rush, is_off]

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        max_s = max(scores.values())
        exps = {a: math.exp(s - max_s) for a, s in scores.items()}
        z = sum(exps.values())
        return {a: exps[a] / z for a in exps}

    def choose_action(self, cpu_seq: List[float], mem_seq: List[float], hour: int) -> (str, list, Dict[str, float]):
        x = self._features(cpu_seq, mem_seq, hour)
        scores = {}
        for a in self.ACTIONS:
            w = self.weights[a]
            scores[a] = sum(w[i] * x[i] for i in range(self.dim))
        probs = self._softmax(scores)
        if random.random() < 0.1:
            action = random.choice(self.ACTIONS)
        else:
            action = max(probs.items(), key=lambda kv: kv[1])[0]
        return action, x, probs

    def update(self, action: str, x: list, reward: float, probs: Dict[str, float]):
        for i in range(self.dim):
            self.weights[action][i] += self.lr * reward * x[i]
        for b in self.ACTIONS:
            if b == action:
                continue
            for i in range(self.dim):
                self.weights[b][i] -= self.lr * reward * probs[b] * x[i]
        self._save()

    def merge_remote(self, remote_weights: Dict[str, List[float]], alpha: float = 0.3):
        try:
            for a in self.ACTIONS:
                if a in remote_weights and len(remote_weights[a]) == self.dim:
                    for i in range(self.dim):
                        self.weights[a][i] = (1 - alpha) * self.weights[a][i] + alpha * remote_weights[a][i]
            self._save()
        except Exception as e:
            print(f"[BORG][POLICY] Merge error: {e}")

    def _save(self):
        try:
            data = {
                "weights": self.weights,
                "lr": self.lr,
                "dim": self.dim,
                "history_len": self.history_len,
            }
            self.MODEL_PATH.write_text(json.dumps(data))
        except Exception as e:
            print(f"[BORG][POLICY] Save error: {e}")

    def _load(self):
        try:
            if self.MODEL_PATH.exists():
                data = json.loads(self.MODEL_PATH.read_text())
                self.weights = data.get("weights", self.weights)
                self.lr = data.get("lr", self.lr)
                self.dim = data.get("dim", self.dim)
                self.history_len = data.get("history_len", self.history_len)
        except Exception as e:
            print(f"[BORG][POLICY] Load error: {e}")

# =========================
# 4. Queen core (AI brain with sequence memory)
# =========================

class QueenCore:
    def __init__(self, bus: MessageBus, config: BorgConfig):
        self.bus = bus
        self.config = config
        self.policy = LearnedPolicy(history_len=config.history_len, lr=0.02)

        self.state: Dict[str, Any] = {
            "health": "OK",
            "last_anomalies": [],
            "cache_stats": {},
            "telemetry_snapshot": {},
            "last_thought": None,
            "before_after": None,
            "thought_log": [],
        }
        self._lock = threading.Lock()
        self._running = False

        self.cpu_history = deque(maxlen=config.history_len)
        self.mem_history = deque(maxlen=config.history_len)

        self.bus.subscribe("TELEMETRY_SNAPSHOT", self._on_telemetry)
        self.bus.subscribe("ANOMALY_DETECTED", self._on_anomaly)
        self.bus.subscribe("CACHE_UPDATE", self._on_cache_update)

    def _on_telemetry(self, ev: BorgEvent):
        snap = ev.payload
        cpu = snap.get("cpu_percent", 0.0)
        mem = snap.get("mem_percent", 0.0)
        with self._lock:
            self.state["telemetry_snapshot"] = snap
            self.cpu_history.append(cpu)
            self.mem_history.append(mem)

    def _on_anomaly(self, ev: BorgEvent):
        with self._lock:
            self.state["last_anomalies"].append(ev.payload)
            self.state["health"] = "WARN"

    def _on_cache_update(self, ev: BorgEvent):
        with self._lock:
            self.state["cache_stats"] = ev.payload

    def start_ai_loop(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._ai_loop, daemon=True)
        t.start()

    def _ai_loop(self):
        while self._running:
            time.sleep(self.config.ai_decision_interval_sec)

            with self._lock:
                snap = dict(self.state.get("telemetry_snapshot") or {})
                cpu_seq = list(self.cpu_history)
                mem_seq = list(self.mem_history)
            if not snap:
                continue

            cpu = snap.get("cpu_percent", 0.0)
            mem = snap.get("mem_percent", 0.0)
            hour = time.localtime().tm_hour

            action, feats, probs = self.policy.choose_action(cpu_seq, mem_seq, hour)

            trend = "stable"
            if len(cpu_seq) >= 2 and cpu_seq[-1] > cpu_seq[-2] + 5:
                trend = "rising"
            elif len(cpu_seq) >= 2 and cpu_seq[-1] < cpu_seq[-2] - 5:
                trend = "falling"

            if action == "THROTTLE_HEAVY_PROCS":
                thought = (
                    f"Sequence shows {trend} CPU trend with last={cpu:.1f}%, mem={mem:.1f}% at hour={hour} → "
                    f"learned that throttling heavy processes tends to stabilize load."
                )
            elif action == "PREFETCH_COMMON_FILES":
                thought = (
                    f"Sequence shows {trend} CPU trend with last={cpu:.1f}%, mem={mem:.1f}% at hour={hour} → "
                    f"learned that prefetching common files reduces future spikes."
                )
            else:
                thought = (
                    f"Sequence shows {trend} CPU trend with last={cpu:.1f}%, mem={mem:.1f}% at hour={hour} → "
                    f"learned that doing nothing is often optimal here."
                )

            before = {"cpu": cpu, "mem": mem, "hour": hour}

            self.bus.publish(BorgEvent(
                type="QUEEN_ACTION",
                payload={
                    "action_type": action,
                    "context": {
                        "cpu": cpu,
                        "mem": mem,
                        "hour": hour,
                        "probs": probs,
                        "trend": trend,
                    }
                }
            ))

            time.sleep(1.0)
            with self._lock:
                snap_after = dict(self.state.get("telemetry_snapshot") or {})
            after = {
                "cpu": snap_after.get("cpu_percent", 0.0),
                "mem": snap_after.get("mem_percent", 0.0),
            }

            reward = 0.0
            if after["cpu"] <= cpu and after["mem"] <= mem:
                reward = 1.0
            elif after["cpu"] > cpu + 5 or after["mem"] > mem + 5:
                reward = -1.0

            self.policy.update(action, feats, reward, probs)

            record = {
                "ts": time.time(),
                "thought": thought,
                "action": action,
                "before": before,
                "after": after,
                "reward": reward,
            }
            with self._lock:
                self.state["last_thought"] = record
                self.state["before_after"] = {
                    "cpu_before": before["cpu"],
                    "cpu_after": after["cpu"],
                    "mem_before": before["mem"],
                    "mem_after": after["mem"],
                    "action": action,
                    "reward": reward,
                }
                self.state["thought_log"].append(record)
                if len(self.state["thought_log"]) > 300:
                    self.state["thought_log"].pop(0)

# =========================
# 5. Telemetry spine
# =========================

class TelemetrySpine:
    def __init__(self, bus: MessageBus, config: BorgConfig):
        self.bus = bus
        self.config = config
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self._running:
            snapshot = self._collect_snapshot()
            self.bus.publish(BorgEvent(type="TELEMETRY_SNAPSHOT", payload=snapshot))
            time.sleep(self.config.telemetry_interval_sec)

    def _collect_snapshot(self) -> Dict[str, Any]:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        disk = psutil.disk_io_counters()
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
            info = p.info
            procs.append({
                "pid": info.get("pid"),
                "name": info.get("name"),
                "cpu": info.get("cpu_percent"),
                "rss": getattr(info.get("memory_info"), "rss", None)
            })
        procs = sorted(procs, key=lambda x: x["cpu"] or 0, reverse=True)[:10]
        return {
            "cpu_percent": cpu,
            "mem_percent": mem.percent,
            "disk_read": getattr(disk, "read_bytes", 0),
            "disk_write": getattr(disk, "write_bytes", 0),
            "top_procs": procs,
            "ts": time.time(),
        }

# =========================
# 6. GPU/VRAM staging with simple acceleration
# =========================

class GPUStager:
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.enabled = bool(GPU_LIBS)

    def stage_buffer(self, size_bytes: int = 8 * 1024 * 1024):
        if not self.enabled:
            return
        payload = {"size": size_bytes, "libs": GPU_LIBS, "ts": time.time()}
        try:
            if "torch" in GPU_LIBS:
                import torch
                buf = torch.ones(size_bytes // 4, device="cuda" if torch.cuda.is_available() else "cpu")
                _ = buf.sum().item()
                del buf
                payload["engine"] = "torch"
            elif "cupy" in GPU_LIBS:
                import cupy as cp
                buf = cp.ones(size_bytes // 4, dtype=cp.float32)
                _ = cp.sum(buf).item()
                del buf
                payload["engine"] = "cupy"
        except Exception as e:
            payload["error"] = str(e)
        self.bus.publish(BorgEvent(type="GPU_STAGE", payload=payload))

# =========================
# 7. Fast cache manager (deeper caching logic)
# =========================

class FastCacheManager:
    def __init__(self, bus: MessageBus, config: BorgConfig, gpu_stager: GPUStager):
        self.bus = bus
        self.config = config
        self.gpu_stager = gpu_stager
        self.hot_files: Dict[str, int] = {}
        self.bus.subscribe("TELEMETRY_SNAPSHOT", self._on_telemetry)
        self.bus.subscribe("QUEEN_ACTION", self._on_action)

    def _on_telemetry(self, ev: BorgEvent):
        stats = {
            "hot_files_count": len(self.hot_files),
            "total_hits": sum(self.hot_files.values()) if self.hot_files else 0,
            "top_hot_files": sorted(self.hot_files.items(), key=lambda kv: kv[1], reverse=True)[:5],
        }
        self.bus.publish(BorgEvent(type="CACHE_UPDATE", payload=stats))

    def _on_action(self, ev: BorgEvent):
        act = ev.payload.get("action_type")
        if act == "PREFETCH_COMMON_FILES":
            self._prefetch_common()
            self.gpu_stager.stage_buffer()
        elif act == "THROTTLE_HEAVY_PROCS":
            self._throttle_heavy_procs()

    def _prefetch_common(self):
        candidates = [
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Desktop"),
        ]
        for base in candidates:
            if os.path.isdir(base):
                try:
                    entries = sorted(
                        (os.path.join(base, n) for n in os.listdir(base)),
                        key=lambda p: os.path.getsize(p) if os.path.isfile(p) else 0,
                        reverse=True
                    )
                    for path in entries[:10]:
                        if os.path.isfile(path):
                            self._prefetch_file(path)
                except Exception:
                    pass

    def _prefetch_file(self, path: str):
        try:
            with open(path, "rb") as f:
                _ = f.read(1024 * 1024)
            self.hot_files[path] = self.hot_files.get(path, 0) + 1
        except Exception as e:
            print(f"[BORG][CACHE] Prefetch error for {path}: {e}")

    def _throttle_heavy_procs(self):
        try:
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent"]):
                info = p.info
                procs.append((p, info.get("cpu_percent") or 0.0))
            procs.sort(key=lambda x: x[1], reverse=True)
            for p, cpu in procs[:3]:
                try:
                    if os.name == "nt":
                        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    else:
                        cur = p.nice()
                        p.nice(min(cur + 5, 19))
                except Exception:
                    continue
        except Exception as e:
            print(f"[BORG][THROTTLE] Error: {e}")

# =========================
# 8. Anomaly engine
# =========================

class AnomalyEngine:
    def __init__(self, bus: MessageBus, config: BorgConfig):
        self.bus = bus
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.bus.subscribe("TELEMETRY_SNAPSHOT", self._on_telemetry)

    def _on_telemetry(self, ev: BorgEvent):
        snap = ev.payload
        self.history.append(snap)
        if len(self.history) > 200:
            self.history.pop(0)
        self._check_anomalies(snap)

    def _check_anomalies(self, snap: Dict[str, Any]):
        cpu = snap.get("cpu_percent", 0)
        mem = snap.get("mem_percent", 0)
        if cpu > 95 or mem > 95:
            self.bus.publish(BorgEvent(
                type="ANOMALY_DETECTED",
                payload={"reason": "HIGH_RESOURCE", "cpu": cpu, "mem": mem, "ts": snap.get("ts")}
            ))

# =========================
# 9. Swarm mesh (policy sharing)
# =========================

class SwarmMesh:
    def __init__(self, bus: MessageBus, config: BorgConfig, queen: QueenCore):
        self.bus = bus
        self.config = config
        self.queen = queen
        self._running = False

    def start(self):
        if not self.config.swarm_endpoint:
            return
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self._running:
            try:
                payload = {
                    "node_id": self.config.node_id,
                    "ts": time.time(),
                    "weights": self.queen.policy.weights,
                }
                resp = requests.post(self.config.swarm_endpoint, json=payload, timeout=3)
                if resp.ok:
                    data = resp.json()
                    remote_weights = data.get("weights")
                    if remote_weights:
                        self.queen.policy.merge_remote(remote_weights, alpha=0.3)
            except Exception:
                pass
            time.sleep(30)

# =========================
# 10. Stress & test harness manager (visual controls)
# =========================

class StressManager:
    def __init__(self, bus: MessageBus, config: BorgConfig, gpu_stager: GPUStager, queen: QueenCore):
        self.bus = bus
        self.config = config
        self.gpu_stager = gpu_stager
        self.queen = queen

        self._cpu_disk_running = False
        self._gpu_stress_running = False
        self._net_stress_running = False
        self._swarm_sim_running = False

    # ---- CPU + Disk load ----
    def start_cpu_disk_stress(self):
        if self._cpu_disk_running:
            return
        self._cpu_disk_running = True
        t = threading.Thread(target=self._cpu_disk_loop, daemon=True)
        t.start()

    def stop_cpu_disk_stress(self):
        self._cpu_disk_running = False

    def _cpu_disk_loop(self):
        path = os.path.join(os.path.expanduser("~"), "borg_stress.bin")
        while self._cpu_disk_running:
            # CPU burn
            start = time.time()
            while time.time() - start < 0.7:
                x = 0
                for i in range(10000):
                    x += i * i
            # Disk I/O
            try:
                block = os.urandom(256 * 1024)
                with open(path, "wb") as f:
                    f.write(block)
                with open(path, "rb") as f:
                    _ = f.read()
            except Exception:
                pass
            time.sleep(0.3)

    # ---- GPU-only stress ----
    def start_gpu_stress(self):
        if self._gpu_stress_running:
            return
        self._gpu_stress_running = True
        t = threading.Thread(target=self._gpu_loop, daemon=True)
        t.start()

    def stop_gpu_stress(self):
        self._gpu_stress_running = False

    def _gpu_loop(self):
        while self._gpu_stress_running:
            self.gpu_stager.stage_buffer(size_bytes=16 * 1024 * 1024)
            time.sleep(1.0)

    # ---- Network I/O stress ----
    def start_net_stress(self):
        if self._net_stress_running:
            return
        self._net_stress_running = True
        t = threading.Thread(target=self._net_loop, daemon=True)
        t.start()

    def stop_net_stress(self):
        self._net_stress_running = False

    def _net_loop(self):
        while self._net_stress_running:
            try:
                url = self.config.swarm_endpoint or "https://example.com"
                requests.get(url, timeout=2)
            except Exception:
                pass
            time.sleep(0.5)

    # ---- Synthetic anomaly injector ----
    def inject_anomaly(self):
        ev = BorgEvent(
            type="ANOMALY_DETECTED",
            payload={
                "reason": "SYNTHETIC_INJECT",
                "cpu": 99.0,
                "mem": 99.0,
                "ts": time.time(),
            }
        )
        self.bus.publish(ev)

    # ---- Multi-node swarm test simulator ----
    def start_swarm_sim(self):
        if self._swarm_sim_running:
            return
        self._swarm_sim_running = True
        t = threading.Thread(target=self._swarm_sim_loop, daemon=True)
        t.start()

    def stop_swarm_sim(self):
        self._swarm_sim_running = False

    def _swarm_sim_loop(self):
        while self._swarm_sim_running:
            try:
                # create synthetic remote weights with small noise
                remote = {}
                for a, w in self.queen.policy.weights.items():
                    remote[a] = [wi + random.uniform(-0.05, 0.05) for wi in w]
                self.queen.policy.merge_remote(remote, alpha=0.2)
            except Exception:
                pass
            time.sleep(10)

# =========================
# 11. PySide6 cockpit (read-only mindstream + visual load controls)
# =========================

class CockpitWindow(QtWidgets.QMainWindow):
    def __init__(self, queen: QueenCore, stress: StressManager):
        super().__init__()
        self.queen = queen
        self.stress = stress

        self.setWindowTitle("Borg Tier-8 Cockpit – Queen Mindstream")
        self.resize(1500, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.health_label = QtWidgets.QLabel("Health: UNKNOWN")
        self.health_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.health_label)

        self.thought_label = QtWidgets.QLabel("Last thought: (waiting...)")
        self.thought_label.setWordWrap(True)
        self.thought_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.thought_label)

        self.before_after_label = QtWidgets.QLabel("Before/After: (waiting...)")
        self.before_after_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.before_after_label)

        # Visual load generator controls
        controls_box = QtWidgets.QGroupBox("Test Harness Controls (Environment Only)")
        controls_layout = QtWidgets.QHBoxLayout(controls_box)

        self.btn_cpu_disk = QtWidgets.QPushButton("CPU+Disk Load: OFF")
        self.btn_cpu_disk.setCheckable(True)
        self.btn_cpu_disk.clicked.connect(self.toggle_cpu_disk)

        self.btn_gpu = QtWidgets.QPushButton("GPU Stress: OFF")
        self.btn_gpu.setCheckable(True)
        self.btn_gpu.clicked.connect(self.toggle_gpu)

        self.btn_net = QtWidgets.QPushButton("Network I/O: OFF")
        self.btn_net.setCheckable(True)
        self.btn_net.clicked.connect(self.toggle_net)

        self.btn_swarm = QtWidgets.QPushButton("Swarm Simulator: OFF")
        self.btn_swarm.setCheckable(True)
        self.btn_swarm.clicked.connect(self.toggle_swarm)

        self.btn_anomaly = QtWidgets.QPushButton("Inject Synthetic Anomaly")
        self.btn_anomaly.clicked.connect(self.inject_anomaly)

        controls_layout.addWidget(self.btn_cpu_disk)
        controls_layout.addWidget(self.btn_gpu)
        controls_layout.addWidget(self.btn_net)
        controls_layout.addWidget(self.btn_swarm)
        controls_layout.addWidget(self.btn_anomaly)

        layout.addWidget(controls_box)

        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter, 1)

        self.telemetry_view = QtWidgets.QTextEdit()
        self.telemetry_view.setReadOnly(True)
        self.telemetry_view.setPlaceholderText("Telemetry snapshot")

        self.cache_view = QtWidgets.QTextEdit()
        self.cache_view.setReadOnly(True)
        self.cache_view.setPlaceholderText("Cache stats")

        self.thought_log_view = QtWidgets.QTextEdit()
        self.thought_log_view.setReadOnly(True)
        self.thought_log_view.setPlaceholderText("AI thought log")

        splitter.addWidget(self.telemetry_view)
        splitter.addWidget(self.cache_view)
        splitter.addWidget(self.thought_log_view)

        self.anomaly_view = QtWidgets.QTextEdit()
        self.anomaly_view.setReadOnly(True)
        self.anomaly_view.setPlaceholderText("Anomalies")
        layout.addWidget(self.anomaly_view, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_from_state)
        self.timer.start(1000)

    # ---- control handlers ----
    def toggle_cpu_disk(self, checked: bool):
        if checked:
            self.stress.start_cpu_disk_stress()
            self.btn_cpu_disk.setText("CPU+Disk Load: ON")
        else:
            self.stress.stop_cpu_disk_stress()
            self.btn_cpu_disk.setText("CPU+Disk Load: OFF")

    def toggle_gpu(self, checked: bool):
        if checked:
            self.stress.start_gpu_stress()
            self.btn_gpu.setText("GPU Stress: ON")
        else:
            self.stress.stop_gpu_stress()
            self.btn_gpu.setText("GPU Stress: OFF")

    def toggle_net(self, checked: bool):
        if checked:
            self.stress.start_net_stress()
            self.btn_net.setText("Network I/O: ON")
        else:
            self.stress.stop_net_stress()
            self.btn_net.setText("Network I/O: OFF")

    def toggle_swarm(self, checked: bool):
        if checked:
            self.stress.start_swarm_sim()
            self.btn_swarm.setText("Swarm Simulator: ON")
        else:
            self.stress.stop_swarm_sim()
            self.btn_swarm.setText("Swarm Simulator: OFF")

    def inject_anomaly(self):
        self.stress.inject_anomaly()

    # ---- state refresh ----
    def refresh_from_state(self):
        st = self.queen.state
        health = st.get("health", "UNKNOWN")
        self.health_label.setText(f"Health: {health}")

        tele = st.get("telemetry_snapshot", {})
        cache = st.get("cache_stats", {})
        anomalies = st.get("last_anomalies", [])
        ba = st.get("before_after")
        last_thought = st.get("last_thought")
        log = st.get("thought_log", [])

        if last_thought:
            self.thought_label.setText(f"Last thought: {last_thought['thought']}")
        else:
            self.thought_label.setText("Last thought: (waiting for first decision...)")

        if ba:
            txt = (
                f"CPU: {ba['cpu_before']:.1f}% → {ba['cpu_after']:.1f}%   "
                f"MEM: {ba['mem_before']:.1f}% → {ba['mem_after']:.1f}%   "
                f"Action: {ba['action']}   "
                f"Reward: {ba['reward']:+.1f}"
            )
        else:
            txt = "Before/After: (waiting for first decision...)"
        self.before_after_label.setText(txt)

        self.telemetry_view.setPlainText(json.dumps(tele, indent=2))
        self.cache_view.setPlainText(json.dumps(cache, indent=2))
        self.anomaly_view.setPlainText(json.dumps(anomalies[-10:], indent=2))
        self.thought_log_view.setPlainText(json.dumps(log[-40:], indent=2))

# =========================
# 12. Wiring it all together
# =========================

def main():
    bus = MessageBus()
    config = BorgConfig()
    queen = QueenCore(bus, config)
    telemetry = TelemetrySpine(bus, config)
    gpu_stager = GPUStager(bus)
    cache = FastCacheManager(bus, config, gpu_stager)
    anomaly = AnomalyEngine(bus, config)
    swarm = SwarmMesh(bus, config, queen)
    stress = StressManager(bus, config, gpu_stager, queen)

    bus.start()
    telemetry.start()
    swarm.start()
    queen.start_ai_loop()

    app = QtWidgets.QApplication(sys.argv)
    win = CockpitWindow(queen, stress)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
