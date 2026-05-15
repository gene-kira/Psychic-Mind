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
import ctypes
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
from PySide6 import QtWidgets, QtCore, QtGui
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
# 0.1 Auto-elevation (Windows)
# =========================

def ensure_admin():
    if os.name != "nt":
        return
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}" {params}', None, 1
            )
            sys.exit()
    except Exception as e:
        print(f"[Codex Sentinel] Elevation failed: {e}")
        sys.exit()

ensure_admin()

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
    swarm_endpoint: Optional[str] = None
    cache_enabled: bool = True
    anomaly_enabled: bool = True
    telemetry_interval_sec: float = 2.0
    ai_decision_interval_sec: float = 5.0
    history_len: int = 8
    dream_start_hour: int = 2
    dream_end_hour: int = 4

# =========================
# 3. Tiny latent encoder (autoencoder-style)
# =========================

class LatentEncoder:
    def __init__(self, input_dim: int, latent_dim: int = 4, lr: float = 0.001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        rnd = random.Random(42)
        self.W_enc = [[(rnd.random() - 0.5) * 0.2 for _ in range(input_dim)] for _ in range(latent_dim)]
        self.W_dec = [[(rnd.random() - 0.5) * 0.2 for _ in range(latent_dim)] for _ in range(input_dim)]

    def encode(self, x: List[float]) -> List[float]:
        z = []
        for j in range(self.latent_dim):
            z.append(sum(self.W_enc[j][i] * x[i] for i in range(self.input_dim)))
        return z

    def decode(self, z: List[float]) -> List[float]:
        x_hat = []
        for i in range(self.input_dim):
            x_hat.append(sum(self.W_dec[i][j] * z[j] for j in range(self.latent_dim)))
        return x_hat

    def reconstruct(self, x: List[float]) -> (List[float], float):
        z = self.encode(x)
        x_hat = self.decode(z)
        err = sum((xi - xhi) ** 2 for xi, xhi in zip(x, x_hat)) / len(x)
        return x_hat, err

    def train_step(self, x: List[float]):
        x_hat, _ = self.reconstruct(x)
        grad_dec = [[0.0] * self.latent_dim for _ in range(self.input_dim)]
        grad_enc = [[0.0] * self.input_dim for _ in range(self.latent_dim)]

        for i in range(self.input_dim):
            diff = x_hat[i] - x[i]
            for j in range(self.latent_dim):
                grad_dec[i][j] += 2 * diff * self.encode(x)[j]

        for j in range(self.latent_dim):
            for i in range(self.input_dim):
                diff = x_hat[i] - x[i]
                grad_enc[j][i] += 2 * diff * self.W_dec[i][j] * x[i]

        for i in range(self.input_dim):
            for j in range(self.latent_dim):
                self.W_dec[i][j] -= self.lr * grad_dec[i][j]
        for j in range(self.latent_dim):
            for i in range(self.input_dim):
                self.W_enc[j][i] -= self.lr * grad_enc[j][i]

# =========================
# 4. Learned policy + multi-head prediction + curiosity
# =========================

class LearnedPolicy:
    ACTIONS = ["NO_OP", "PREFETCH_COMMON_FILES", "THROTTLE_HEAVY_PROCS", "PROBE_PROCS"]
    MODEL_PATH = Path.home() / ".borg_queen_policy_tier11.json"

    def __init__(self, history_len: int = 8, lr: float = 0.01):
        self.lr = lr
        self.history_len = history_len

        # features:
        # cpu_seq (h), mem_seq (h),
        # disk_read_seq (h), disk_write_seq (h),
        # hour_norm, is_rush, is_off,
        # prediction_error_cpu, prediction_error_mem, prediction_error_dread, prediction_error_dwrite,
        # telemetry_gap_norm, missing_flag,
        # mode_one_hot (4),
        # water_flow (4): cpu_flow, mem_flow, disk_r_flow, disk_w_flow
        self.dim = history_len * 4 + 3 + 4 + 2 + 4 + 4
        self.weights = {a: [0.0] * self.dim for a in self.ACTIONS}

        self.pred_w_cpu = [0.0] * self.dim
        self.pred_w_mem = [0.0] * self.dim
        self.pred_w_dread = [0.0] * self.dim
        self.pred_w_dwrite = [0.0] * self.dim

        self.last_pred_err_cpu = 0.0
        self.last_pred_err_mem = 0.0
        self.last_pred_err_dread = 0.0
        self.last_pred_err_dwrite = 0.0

        self._load()

    def _mode_one_hot(self, mode: str) -> List[float]:
        modes = ["BASELINE", "HYPERVIGILANT", "DREAMING", "FRAGMENTED"]
        return [1.0 if mode == m else 0.0 for m in modes]

    def _features(
        self,
        cpu_seq: List[float],
        mem_seq: List[float],
        dread_seq: List[float],
        dwrite_seq: List[float],
        hour: int,
        mode: str,
        telemetry_gap_sec: float,
        missing_flag: float,
    ) -> list:
        def pad(seq):
            if not seq:
                return [0.0] * self.history_len
            seq = (seq + [seq[-1]] * self.history_len)[:self.history_len]
            return seq

        cpu_seq = pad(cpu_seq)
        mem_seq = pad(mem_seq)
        dread_seq = pad(dread_seq)
        dwrite_seq = pad(dwrite_seq)

        cpu_norm = [c / 100.0 for c in cpu_seq]
        mem_norm = [m / 100.0 for m in mem_seq]

        max_io = max(max(dread_seq + dwrite_seq) or 1.0, 1.0)
        dread_norm = [d / max_io for d in dread_seq]
        dwrite_norm = [d / max_io for d in dwrite_seq]

        hour_norm = hour / 23.0 if hour > 0 else 0.0
        is_rush = 1.0 if hour in (8, 9, 17, 18) else 0.0
        is_off = 1.0 if 0 <= hour <= 5 else 0.0

        gap_norm = min(telemetry_gap_sec / 10.0, 1.0)

        mode_vec = self._mode_one_hot(mode)

        def flow(seq):
            if len(seq) >= 2:
                return (seq[-1] - seq[-2]) / (max(seq[-2], 1.0))
            return 0.0

        cpu_flow = flow(cpu_seq)
        mem_flow = flow(mem_seq)
        dread_flow = flow(dread_seq)
        dwrite_flow = flow(dwrite_seq)

        return (
            cpu_norm
            + mem_norm
            + dread_norm
            + dwrite_norm
            + [hour_norm, is_rush, is_off]
            + [
                self.last_pred_err_cpu,
                self.last_pred_err_mem,
                self.last_pred_err_dread,
                self.last_pred_err_dwrite,
            ]
            + [gap_norm, missing_flag]
            + mode_vec
            + [cpu_flow, mem_flow, dread_flow, dwrite_flow]
        )

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        max_s = max(scores.values())
        exps = {a: math.exp(s - max_s) for a, s in scores.items()}
        z = sum(exps.values())
        return {a: exps[a] / z for a in exps}

    def _predict_next(self, x: list) -> (float, float, float, float):
        cpu_hat = sum(w * xi for w, xi in zip(self.pred_w_cpu, x))
        mem_hat = sum(w * xi for w, xi in zip(self.pred_w_mem, x))
        dread_hat = sum(w * xi for w, xi in zip(self.pred_w_dread, x))
        dwrite_hat = sum(w * xi for w, xi in zip(self.pred_w_dwrite, x))

        cpu_hat = max(0.0, min(100.0, cpu_hat))
        mem_hat = max(0.0, min(100.0, mem_hat))
        dread_hat = max(0.0, dread_hat)
        dwrite_hat = max(0.0, dwrite_hat)
        return cpu_hat, mem_hat, dread_hat, dwrite_hat

    def update_prediction_head(
        self,
        x: list,
        cpu_next: float,
        mem_next: float,
        dread_next: float,
        dwrite_next: float,
        pred_lr: float = 0.003,
    ):
        cpu_hat, mem_hat, dread_hat, dwrite_hat = self._predict_next(x)
        err_cpu = cpu_next - cpu_hat
        err_mem = mem_next - mem_hat
        err_dread = dread_next - dread_hat
        err_dwrite = dwrite_next - dwrite_hat

        for i in range(self.dim):
            self.pred_w_cpu[i] += pred_lr * err_cpu * x[i]
            self.pred_w_mem[i] += pred_lr * err_mem * x[i]
            self.pred_w_dread[i] += pred_lr * err_dread * x[i]
            self.pred_w_dwrite[i] += pred_lr * err_dwrite * x[i]

        self.last_pred_err_cpu = abs(err_cpu) / 100.0
        self.last_pred_err_mem = abs(err_mem) / 100.0
        self.last_pred_err_dread = abs(err_dread) / (max(dread_next, 1.0))
        self.last_pred_err_dwrite = abs(err_dwrite) / (max(dwrite_next, 1.0))

    def choose_action(
        self,
        cpu_seq: List[float],
        mem_seq: List[float],
        dread_seq: List[float],
        dwrite_seq: List[float],
        hour: int,
        mode: str,
        telemetry_gap_sec: float,
        missing_flag: float,
        curiosity_level: float,
    ) -> (str, list, Dict[str, float]):
        x = self._features(cpu_seq, mem_seq, dread_seq, dwrite_seq, hour, mode, telemetry_gap_sec, missing_flag)
        scores = {}
        for a in self.ACTIONS:
            w = self.weights[a]
            scores[a] = sum(w[i] * x[i] for i in range(self.dim))
        probs = self._softmax(scores)

        base_eps = 0.1
        if mode == "DREAMING":
            base_eps = 0.3
        elif mode == "HYPERVIGILANT":
            base_eps = 0.05

        eps = min(0.5, base_eps + curiosity_level * 0.3)

        if random.random() < eps:
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
                "pred_w_cpu": self.pred_w_cpu,
                "pred_w_mem": self.pred_w_mem,
                "pred_w_dread": self.pred_w_dread,
                "pred_w_dwrite": self.pred_w_dwrite,
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
                self.pred_w_cpu = data.get("pred_w_cpu", self.pred_w_cpu)
                self.pred_w_mem = data.get("pred_w_mem", self.pred_w_mem)
                self.pred_w_dread = data.get("pred_w_dread", self.pred_w_dread)
                self.pred_w_dwrite = data.get("pred_w_dwrite", self.pred_w_dwrite)
        except Exception as e:
            print(f"[BORG][POLICY] Load error: {e}")

# =========================
# 5. Queen core (modes, curiosity, dream cycle, threat chain)
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
            "mode": "BASELINE",
            "threat_chain": [],
            "timeline": [],
        }
        self._lock = threading.Lock()
        self._running = False

        self.cpu_history = deque(maxlen=config.history_len)
        self.mem_history = deque(maxlen=config.history_len)
        self.dread_history = deque(maxlen=config.history_len)
        self.dwrite_history = deque(maxlen=config.history_len)
        self.last_snapshot_ts = None
        self.last_action_features = None

        self.latent_encoder = LatentEncoder(input_dim=self.policy.dim, latent_dim=6, lr=0.0008)
        self.latent_history = deque(maxlen=512)

        self.bus.subscribe("TELEMETRY_SNAPSHOT", self._on_telemetry)
        self.bus.subscribe("ANOMALY_DETECTED", self._on_anomaly)
        self.bus.subscribe("CACHE_UPDATE", self._on_cache_update)
        self.bus.subscribe("QUEEN_ACTION", self._on_action_event)

    def _on_telemetry(self, ev: BorgEvent):
        snap = ev.payload
        cpu = snap.get("cpu_percent", 0.0)
        mem = snap.get("mem_percent", 0.0)
        dread = snap.get("disk_read", 0.0)
        dwrite = snap.get("disk_write", 0.0)
        ts = snap.get("ts", time.time())

        with self._lock:
            if self.last_action_features is not None:
                self.policy.update_prediction_head(
                    self.last_action_features, cpu, mem, dread, dwrite
                )

            self.state["telemetry_snapshot"] = snap
            self.cpu_history.append(cpu)
            self.mem_history.append(mem)
            self.dread_history.append(dread)
            self.dwrite_history.append(dwrite)
            self.last_snapshot_ts = ts

            self._update_mode_locked()

    def _update_mode_locked(self):
        anomalies = self.state.get("last_anomalies", [])
        recent_anoms = [a for a in anomalies if time.time() - a.get("ts", 0) < 60]
        anom_rate = len(recent_anoms)

        err = max(
            self.policy.last_pred_err_cpu,
            self.policy.last_pred_err_mem,
            self.policy.last_pred_err_dread,
            self.policy.last_pred_err_dwrite,
        )
        hour = time.localtime().tm_hour

        if anom_rate > 3 or err > 0.5:
            mode = "HYPERVIGILANT"
        elif self.config.dream_start_hour <= hour <= self.config.dream_end_hour:
            mode = "DREAMING"
        elif err > 0.3:
            mode = "FRAGMENTED"
        else:
            mode = "BASELINE"

        self.state["mode"] = mode

    def _on_anomaly(self, ev: BorgEvent):
        with self._lock:
            self.state["last_anomalies"].append(ev.payload)
            self.state["health"] = "WARN"
            self._update_mode_locked()
            self._append_timeline("ANOMALY", ev.payload)
            self._update_threat_chain(ev.payload)

    def _on_cache_update(self, ev: BorgEvent):
        with self._lock:
            self.state["cache_stats"] = ev.payload

    def _on_action_event(self, ev: BorgEvent):
        with self._lock:
            self._append_timeline("ACTION", ev.payload)

    def _append_timeline(self, kind: str, payload: Dict[str, Any]):
        entry = {
            "ts": time.time(),
            "kind": kind,
            "payload": payload,
        }
        self.state["timeline"].append(entry)
        if len(self.state["timeline"]) > 300:
            self.state["timeline"].pop(0)

    def _update_threat_chain(self, anomaly_payload: Dict[str, Any]):
        chain = self.state.get("threat_chain", [])
        chain.append({
            "ts": anomaly_payload.get("ts", time.time()),
            "reason": anomaly_payload.get("reason", "UNKNOWN"),
            "cpu": anomaly_payload.get("cpu"),
            "mem": anomaly_payload.get("mem"),
        })
        if len(chain) > 50:
            chain.pop(0)
        self.state["threat_chain"] = chain

    def start_ai_loop(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._ai_loop, daemon=True)
        t.start()
        d = threading.Thread(target=self._dream_loop, daemon=True)
        d.start()

    def _compute_missingness(self, snap: Dict[str, Any]) -> (float, float, float):
        now = time.time()
        if self.last_snapshot_ts is None:
            gap = 0.0
        else:
            gap = now - self.last_snapshot_ts

        top_procs = snap.get("top_procs", [])
        disk_read = snap.get("disk_read", None)
        disk_write = snap.get("disk_write", None)

        missing_flag = 0.0
        if top_procs is None or not isinstance(top_procs, list):
            missing_flag = 1.0
        if disk_read is None or disk_write is None:
            missing_flag = 1.0

        return gap, missing_flag, now

    def _curiosity_level(self) -> float:
        err = max(
            self.policy.last_pred_err_cpu,
            self.policy.last_pred_err_mem,
            self.policy.last_pred_err_dread,
            self.policy.last_pred_err_dwrite,
        )
        anomalies = self.state.get("last_anomalies", [])
        recent_anoms = [a for a in anomalies if time.time() - a.get("ts", 0) < 60]
        anom_rate = len(recent_anoms)
        base = err
        if anom_rate == 0:
            base += 0.1
        return max(0.0, min(1.0, base))

    def _ai_loop(self):
        while self._running:
            time.sleep(self.config.ai_decision_interval_sec)

            with self._lock:
                snap = dict(self.state.get("telemetry_snapshot") or {})
                cpu_seq = list(self.cpu_history)
                mem_seq = list(self.mem_history)
                dread_seq = list(self.dread_history)
                dwrite_seq = list(self.dwrite_history)
                mode = self.state.get("mode", "BASELINE")
            if not snap:
                continue

            cpu = snap.get("cpu_percent", 0.0)
            mem = snap.get("mem_percent", 0.0)
            dread = snap.get("disk_read", 0.0)
            dwrite = snap.get("disk_write", 0.0)
            hour = time.localtime().tm_hour

            gap, missing_flag, _ = self._compute_missingness(snap)

            curiosity = self._curiosity_level()

            action, feats, probs = self.policy.choose_action(
                cpu_seq, mem_seq, dread_seq, dwrite_seq, hour, mode, gap, missing_flag, curiosity
            )

            trend = "stable"
            if len(cpu_seq) >= 2 and cpu_seq[-1] > cpu_seq[-2] + 5:
                trend = "rising"
            elif len(cpu_seq) >= 2 and cpu_seq[-1] < cpu_seq[-2] - 5:
                trend = "falling"

            water_pressure = (cpu / 100.0) + (mem / 100.0) + min(dread + dwrite, 1e9) / 1e9

            if action == "THROTTLE_HEAVY_PROCS":
                thought = (
                    f"[{mode}] CPU trend={trend}, last={cpu:.1f}%, mem={mem:.1f}%, water={water_pressure:.2f}, "
                    f"gap={gap:.1f}s, missing={missing_flag:.1f}, curiosity={curiosity:.2f} → "
                    f"THROTTLE_HEAVY_PROCS to relieve pressure."
                )
            elif action == "PREFETCH_COMMON_FILES":
                thought = (
                    f"[{mode}] CPU trend={trend}, last={cpu:.1f}%, mem={mem:.1f}%, water={water_pressure:.2f}, "
                    f"gap={gap:.1f}s, missing={missing_flag:.1f}, curiosity={curiosity:.2f} → "
                    f"PREFETCH_COMMON_FILES to smooth future flow."
                )
            elif action == "PROBE_PROCS":
                thought = (
                    f"[{mode}] High uncertainty (pred_err={max(self.policy.last_pred_err_cpu, self.policy.last_pred_err_mem):.2f}), "
                    f"water={water_pressure:.2f}, gap={gap:.1f}s, missing={missing_flag:.1f}, curiosity={curiosity:.2f} → "
                    f"PROBE_PROCS to reduce blind spots."
                )
            else:
                thought = (
                    f"[{mode}] CPU trend={trend}, last={cpu:.1f}%, mem={mem:.1f}%, water={water_pressure:.2f}, "
                    f"gap={gap:.1f}s, missing={missing_flag:.1f}, curiosity={curiosity:.2f} → "
                    f"NO_OP as safest action under current uncertainty."
                )

            before = {"cpu": cpu, "mem": mem, "dread": dread, "dwrite": dwrite, "hour": hour}

            self.bus.publish(BorgEvent(
                type="QUEEN_ACTION",
                payload={
                    "action_type": action,
                    "context": {
                        "cpu": cpu,
                        "mem": mem,
                        "dread": dread,
                        "dwrite": dwrite,
                        "hour": hour,
                        "probs": probs,
                        "trend": trend,
                        "mode": mode,
                        "gap": gap,
                        "missing": missing_flag,
                        "curiosity": curiosity,
                        "water_pressure": water_pressure,
                    }
                }
            ))

            with self._lock:
                self.last_action_features = feats
                self.latent_history.append(feats)

            time.sleep(1.0)
            with self._lock:
                snap_after = dict(self.state.get("telemetry_snapshot") or {})
            after = {
                "cpu": snap_after.get("cpu_percent", 0.0),
                "mem": snap_after.get("mem_percent", 0.0),
                "dread": snap_after.get("disk_read", 0.0),
                "dwrite": snap_after.get("disk_write", 0.0),
            }

            reward = 0.0
            if after["cpu"] <= cpu and after["mem"] <= mem:
                reward += 1.0
            elif after["cpu"] > cpu + 5 or after["mem"] > mem + 5:
                reward -= 1.0

            reward -= max(
                self.policy.last_pred_err_cpu,
                self.policy.last_pred_err_mem,
                self.policy.last_pred_err_dread,
                self.policy.last_pred_err_dwrite,
            )

            self.policy.update(action, feats, reward, probs)

            record = {
                "ts": time.time(),
                "thought": thought,
                "action": action,
                "before": before,
                "after": after,
                "reward": reward,
                "mode": mode,
                "pred_err_cpu": self.policy.last_pred_err_cpu,
                "pred_err_mem": self.policy.last_pred_err_mem,
                "pred_err_dread": self.policy.last_pred_err_dread,
                "pred_err_dwrite": self.policy.last_pred_err_dwrite,
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
                if len(self.state["thought_log"]) > 600:
                    self.state["thought_log"].pop(0)

    def _dream_loop(self):
        while self._running:
            time.sleep(10)
            hour = time.localtime().tm_hour
            if not (self.config.dream_start_hour <= hour <= self.config.dream_end_hour):
                continue
            with self._lock:
                if not self.latent_history:
                    continue
                batch = random.sample(list(self.latent_history), min(16, len(self.latent_history)))
            for feats in batch:
                self.latent_encoder.train_step(feats)
                _, err = self.latent_encoder.reconstruct(feats)
                if err > 0.5:
                    self.bus.publish(BorgEvent(
                        type="ANOMALY_DETECTED",
                        payload={
                            "reason": "LATENT_ANOMALY",
                            "score": err,
                            "ts": time.time(),
                        }
                    ))

# =========================
# 6. Telemetry spine
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
# 7. GPU/VRAM staging
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
# 8. Fast cache manager
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
        elif act == "PROBE_PROCS":
            self._probe_procs()

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

    def _probe_procs(self):
        try:
            procs = []
            for p in psutil.process_iter(["pid", "name", "memory_info"]):
                info = p.info
                rss = getattr(info.get("memory_info"), "rss", 0) or 0
                procs.append({"pid": info.get("pid"), "name": info.get("name"), "rss": rss})
            procs = sorted(procs, key=lambda x: x["rss"], reverse=True)[:5]
            self.bus.publish(BorgEvent(
                type="PROBE_RESULT",
                payload={"top_rss": procs, "ts": time.time()}
            ))
        except Exception as e:
            print(f"[BORG][PROBE] Error: {e}")

# =========================
# 9. Anomaly engine
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
        if len(self.history) > 400:
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
# 10. Swarm mesh
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
# 11. Stress & test harness
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
            start = time.time()
            while time.time() - start < 0.7:
                x = 0
                for i in range(10000):
                    x += i * i
            try:
                block = os.urandom(256 * 1024)
                with open(path, "wb") as f:
                    f.write(block)
                with open(path, "rb") as f:
                    _ = f.read()
            except Exception:
                pass
            time.sleep(0.3)

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
                remote = {}
                for a, w in self.queen.policy.weights.items():
                    remote[a] = [wi + random.uniform(-0.05, 0.05) for wi in w]
                self.queen.policy.merge_remote(remote, alpha=0.2)
            except Exception:
                pass
            time.sleep(10)

# =========================
# 12. DualPersonalityBot (guardian / rogue)
# =========================

def adaptive_mutation(tag: str):
    return f"mut-{tag}-{int(time.time())}"

def generate_decoy():
    return {"decoy": hex(int(time.time()))}

def compliance_auditor(items):
    return {"items": len(items), "status": "ok"}

def reverse_mirror_encrypt(s: str) -> str:
    return s[::-1] + "|" + "".join(chr((ord(c) + 3) % 126) for c in s[::-1])

def camouflage(s: str, style: str) -> str:
    return f"{style}:{s}"

def random_glyph_stream(n: int = 16) -> str:
    glyphs = "⟁⟡⚚⚝⚚⚝✶✷✸✹✺✻✼✽✾✿"
    return "".join(random.choice(glyphs) for _ in range(n))

class DualPersonalityBot:
    def __init__(self, cb: Callable[[str], None]):
        self.cb = cb
        self.run = True
        self.mode = "guardian"
        self.rogue_weights = [0.2, -0.4, 0.7]
        self.rogue_log = []

    def switch_mode(self):
        self.mode = "rogue" if self.mode == "guardian" else "guardian"
        self.cb(f"🔺 Personality switched to {self.mode.upper()}")

    def guardian_behavior(self):
        tag = adaptive_mutation("ghost sync")
        decoy = generate_decoy()
        self.cb(f"🕊️ Guardian audit tag: {tag}")
        self.cb(f"🕊️ Guardian decoy: {decoy}")
        self.cb(f"🔱 Compliance: {compliance_auditor([decoy])}")

    def rogue_behavior(self):
        entropy = int(time.time()) % 2048
        scrambled = reverse_mirror_encrypt(str(entropy))
        camo = camouflage(str(entropy), "alien")
        glyph_stream = random_glyph_stream()
        unusual_pattern = f"{scrambled[:16]}-{camo}-{glyph_stream[:8]}"

        self.rogue_weights = [
            w + (entropy % 5 - 2) * 0.01 for w in self.rogue_weights
        ]
        self.rogue_log.append(self.rogue_weights)

        score = sum(self.rogue_weights) / len(self.rogue_weights)

        self.cb("💀⚔️ Rogue escalation initiated")
        self.cb(f"🜏 Rogue pattern: {unusual_pattern}")
        self.cb(f"📊 Rogue weights: {self.rogue_weights} | Trust {score:.3f}")

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.run:
            if self.mode == "guardian":
                self.guardian_behavior()
            else:
                self.rogue_behavior()
            time.sleep(10)

# =========================
# 13. Prediction error overlay
# =========================

class PredictionErrorOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None, max_points=120):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.max_points = max_points
        self.errors = deque(maxlen=max_points)
        self.setAutoFillBackground(False)

    def add_error(self, err):
        self.errors.append(err)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QtGui.QColor(10, 10, 20))

        if not self.errors:
            return

        max_err = max(self.errors) if self.errors else 1.0
        max_err = max(max_err, 0.01)

        pen = QtGui.QPen(QtGui.QColor(0, 255, 180), 2)
        painter.setPen(pen)

        step = w / max(1, len(self.errors) - 1)
        path = QtGui.QPainterPath()

        for i, e in enumerate(self.errors):
            x = i * step
            y = h - (e / max_err) * h
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        painter.drawPath(path)

        last_err = self.errors[-1]
        confidence = max(0.0, 1.0 - last_err)
        bar_color = QtGui.QColor(
            int((1 - confidence) * 255),
            int(confidence * 255),
            80
        )
        painter.fillRect(0, h - 10, int(w * confidence), 10, bar_color)

# =========================
# 14. Cockpit (mindstream + timeline + persona)
# =========================

class CockpitWindow(QtWidgets.QMainWindow):
    persona_signal = QtCore.Signal(str)

    def __init__(self, queen: QueenCore, stress: StressManager, persona_bot: DualPersonalityBot):
        super().__init__()
        self.queen = queen
        self.stress = stress
        self.persona_bot = persona_bot

        self.setWindowTitle("Borg Tier-11.1 Cockpit – Queen Mindstream")
        self.resize(1700, 950)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.setStyleSheet("""
            QWidget {
                background-color: #101010;
                color: #E0E0E0;
                font-family: Consolas, monospace;
            }
            QTextEdit {
                background-color: #0D0D0D;
                border: 1px solid #333;
                color: #E0E0E0;
            }
            QLabel {
                color: #E0E0E0;
            }
            QGroupBox {
                border: 1px solid #333;
                margin-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px 0 3px;
            }
        """)

        top_bar = QtWidgets.QHBoxLayout()
        layout.addLayout(top_bar)

        self.health_label = QtWidgets.QLabel("Health: UNKNOWN")
        self.health_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        top_bar.addWidget(self.health_label)

        self.mode_label = QtWidgets.QLabel("Mode: BASELINE")
        self.mode_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        top_bar.addWidget(self.mode_label)

        self.curiosity_label = QtWidgets.QLabel("Curiosity: 0.00")
        self.curiosity_label.setStyleSheet("font-size: 14px;")
        top_bar.addWidget(self.curiosity_label)

        top_bar.addStretch(1)

        self.persona_button = QtWidgets.QPushButton("Switch Persona (Guardian/Rogue)")
        self.persona_button.clicked.connect(self.persona_bot.switch_mode)
        top_bar.addWidget(self.persona_button)

        self.mode_bar = QtWidgets.QFrame()
        self.mode_bar.setFixedHeight(6)
        self.mode_bar.setStyleSheet("background-color: #00FF88;")
        layout.addWidget(self.mode_bar)

        self.thought_label = QtWidgets.QLabel("Last thought: (waiting...)")
        self.thought_label.setWordWrap(True)
        self.thought_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.thought_label)

        self.before_after_label = QtWidgets.QLabel("Before/After: (waiting...)")
        self.before_after_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.before_after_label)

        self.pred_overlay = PredictionErrorOverlay()
        layout.addWidget(self.pred_overlay)

        controls_box = QtWidgets.QGroupBox("Test Harness Controls")
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

        splitter_main = QtWidgets.QSplitter()
        splitter_main.setOrientation(QtCore.Qt.Horizontal)
        layout.addWidget(splitter_main, 1)

        left_panel = QtWidgets.QSplitter()
        left_panel.setOrientation(QtCore.Qt.Vertical)
        splitter_main.addWidget(left_panel)

        self.telemetry_view = QtWidgets.QTextEdit()
        self.telemetry_view.setReadOnly(True)
        self.telemetry_view.setPlaceholderText("Telemetry snapshot")

        self.cache_view = QtWidgets.QTextEdit()
        self.cache_view.setReadOnly(True)
        self.cache_view.setPlaceholderText("Cache stats")

        left_panel.addWidget(self.telemetry_view)
        left_panel.addWidget(self.cache_view)

        center_panel = QtWidgets.QSplitter()
        center_panel.setOrientation(QtCore.Qt.Vertical)
        splitter_main.addWidget(center_panel)

        self.thought_log_view = QtWidgets.QTextEdit()
        self.thought_log_view.setReadOnly(True)
        self.thought_log_view.setPlaceholderText("AI thought log")

        self.timeline_view = QtWidgets.QTextEdit()
        self.timeline_view.setReadOnly(True)
        self.timeline_view.setPlaceholderText("Timeline (actions + anomalies)")

        center_panel.addWidget(self.thought_log_view)
        center_panel.addWidget(self.timeline_view)

        right_panel = QtWidgets.QSplitter()
        right_panel.setOrientation(QtCore.Qt.Vertical)
        splitter_main.addWidget(right_panel)

        self.anomaly_view = QtWidgets.QTextEdit()
        self.anomaly_view.setReadOnly(True)
        self.anomaly_view.setPlaceholderText("Anomalies / Threat chain")

        self.persona_view = QtWidgets.QTextEdit()
        self.persona_view.setReadOnly(True)
        self.persona_view.setPlaceholderText("DualPersonalityBot stream")

        right_panel.addWidget(self.anomaly_view)
        right_panel.addWidget(self.persona_view)

        self.persona_signal.connect(self.persona_log)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_from_state)
        self.timer.start(1000)

    def persona_log(self, msg: str):
        cur = self.persona_view.toPlainText()
        new = (cur + "\n" + msg).strip()
        lines = new.splitlines()
        if len(lines) > 200:
            lines = lines[-200:]
        self.persona_view.setPlainText("\n".join(lines))

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

    def refresh_from_state(self):
        st = self.queen.state
        health = st.get("health", "UNKNOWN")
        mode = st.get("mode", "BASELINE")
        self.health_label.setText(f"Health: {health}")
        self.mode_label.setText(f"Mode: {mode}")

        if mode == "DREAMING":
            self.mode_bar.setStyleSheet("background-color: #3A6DFF;")
        elif mode == "HYPERVIGILANT":
            self.mode_bar.setStyleSheet("background-color: #FF3A3A;")
        elif mode == "FRAGMENTED":
            self.mode_bar.setStyleSheet("background-color: #A03AFF;")
        else:
            self.mode_bar.setStyleSheet("background-color: #00FF88;")

        last_thought = st.get("last_thought")
        ba = st.get("before_after")
        tele = st.get("telemetry_snapshot", {})
        cache = st.get("cache_stats", {})
        anomalies = st.get("last_anomalies", [])
        log = st.get("thought_log", [])
        timeline = st.get("timeline", [])

        if last_thought:
            self.thought_label.setText(f"Last thought: {last_thought['thought']}")
        else:
            self.thought_label.setText("Last thought: (waiting for first decision...)")

        if ba:
            txt = (
                f"CPU: {ba['cpu_before']:.1f}% → {ba['cpu_after']:.1f}%   "
                f"MEM: {ba['mem_before']:.1f}% → {ba['mem_after']:.1f}%   "
                f"Action: {ba['action']}   "
                f"Reward: {ba['reward']:+.2f}"
            )
        else:
            txt = "Before/After: (waiting for first decision...)"
        self.before_after_label.setText(txt)

        err_cpu = last_thought.get("pred_err_cpu", 0.0) if last_thought else 0.0
        err_mem = last_thought.get("pred_err_mem", 0.0) if last_thought else 0.0
        err_dread = last_thought.get("pred_err_dread", 0.0) if last_thought else 0.0
        err_dwrite = last_thought.get("pred_err_dwrite", 0.0) if last_thought else 0.0
        err = (err_cpu + err_mem + err_dread + err_dwrite) / 4.0
        self.pred_overlay.add_error(err)

        self.telemetry_view.setPlainText(json.dumps(tele, indent=2))
        self.cache_view.setPlainText(json.dumps(cache, indent=2))
        self.anomaly_view.setPlainText(json.dumps({
            "recent_anomalies": anomalies[-10:],
            "threat_chain": st.get("threat_chain", [])[-20:],
        }, indent=2))
        self.thought_log_view.setPlainText(json.dumps(log[-80:], indent=2))
        self.timeline_view.setPlainText(json.dumps(timeline[-80:], indent=2))

        curiosity = self.queen._curiosity_level()
        self.curiosity_label.setText(f"Curiosity: {curiosity:.2f}")

# =========================
# 15. Wiring it all together
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

    persona_bot = DualPersonalityBot(lambda msg: None)

    win = CockpitWindow(queen, stress, persona_bot)
    persona_bot.cb = win.persona_signal.emit
    persona_bot.start()

    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
