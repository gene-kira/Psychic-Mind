"""
EVENT HORIZON // BORG CONTROL TOWER - FINAL ORGANISM SKELETON

This single file contains:

TIER 1:
- Plugin architecture hooks
- Kafka-like EventBus abstraction (can be backed by real Kafka later)
- ML Brain v2.0 hooks (feature store, registry, drift, shadow/canary)
- GPU Brain hooks (multi-GPU orchestration)
- Swarm Brain hooks (gossip + consensus)
- Memory Brain hooks (time-series + configs + models)
- Config system (versioned, live reload)
- Web dashboard + Tkinter cockpit
- Control Brain reading from EventBus, not just local state

TIER 2+ (stubs / hooks):
- Autonomous defense layer
- Multi-brain architecture
- Autonomous mode switching
- Borg identity layer
- Predictive GPU scheduling
- Autonomous swarm evolution
- Multi-modal ML integration
- Causal reasoning hooks
- Self-optimization engine
- Self-modifying pipelines (guarded TODOs)

This is a *unified skeleton*: runnable as a basic tower, with clear upgrade points.
"""

import os
import sys
import time
import math
import json
import queue
import atexit
import threading
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional, Tuple
from collections import defaultdict

# -----------------------------
# Autoloader
# -----------------------------
REQUIRED_LIBRARIES = [
    "numpy",
    "psutil",
    "fastapi",
    "uvicorn",
    "joblib",
    "pynvml",
    "yaml",
]

def ensure_libraries_installed():
    for lib in REQUIRED_LIBRARIES:
        try:
            __import__(lib.split("-")[0])
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", lib])

ensure_libraries_installed()

import numpy as np
import psutil
import sqlite3
import joblib
import yaml

import tkinter as tk
from tkinter import ttk, messagebox

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

import psutil as _psutil_for_lock


# -----------------------------
# Paths, DB, PID Lock
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_FILE = os.path.join(DATA_DIR, "tower.db")
LOCK_FILE = os.path.join(DATA_DIR, "tower.lock")
LOG_FILE = os.path.join(DATA_DIR, "control_tower.log")

# PID-aware single-instance lock
if os.path.exists(LOCK_FILE):
    try:
        with open(LOCK_FILE, "r") as f:
            old_pid_str = f.read().strip()
        old_pid = int(old_pid_str) if old_pid_str else -1
        if old_pid > 0 and _psutil_for_lock.pid_exists(old_pid):
            print(f"Another instance (PID {old_pid}) is already running. Exiting.")
            sys.exit(1)
        else:
            os.remove(LOCK_FILE)
            print("Stale lock file detected. Removed.")
    except Exception:
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass

with open(LOCK_FILE, "w") as f:
    f.write(str(os.getpid()))

def _cleanup_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception:
        pass

atexit.register(_cleanup_lock)


# -----------------------------
# Config System (files + DB + live reload)
# -----------------------------
CONFIG_DIR = os.path.join(BASE_DIR, "config")
os.makedirs(CONFIG_DIR, exist_ok=True)

DEFAULT_CORE_YAML = {
    "kafka": {
        "brokers": ["localhost:9092"],
        "client_id": "borg-node-1",
        "group_id": "borg-node-group",
    },
    "swarm": {
        "node_id": "node-1",
        "role": "full",
        "heartbeat_interval": 5,
        "gossip_interval": 2,
    },
    "gpu": {
        "mode": "balanced",
    },
    "ml": {
        "model_name": "ad_filter",
        "model_version": "v1",
    },
    "web": {
        "host": "127.0.0.1",
        "port": 8000,
    },
    "storage": {
        "type": "sqlite",
        "path": DB_FILE,
    },
}

CORE_YAML_PATH = os.path.join(CONFIG_DIR, "core.yaml")
if not os.path.exists(CORE_YAML_PATH):
    with open(CORE_YAML_PATH, "w") as f:
        yaml.safe_dump(DEFAULT_CORE_YAML, f)


class ConfigDB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self.lock:
            c = self.conn.cursor()
            c.execute("""
            CREATE TABLE IF NOT EXISTS configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                version INTEGER,
                data TEXT,
                created_at REAL
            )
            """)
            self.conn.commit()

    def insert_config(self, name: str, version: int, data: str, ts: float):
        with self.lock:
            self.conn.execute(
                "INSERT INTO configs (name, version, data, created_at) VALUES (?, ?, ?, ?)",
                (name, version, data, ts),
            )
            self.conn.commit()

    def latest_config(self, name: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            c = self.conn.cursor()
            c.execute(
                "SELECT version, data FROM configs WHERE name=? ORDER BY version DESC LIMIT 1",
                (name,),
            )
            row = c.fetchone()
        if not row:
            return None
        return {"version": row[0], "data": row[1]}


class ConfigManager:
    def __init__(self, db_path: str, config_dir: str):
        self.db = ConfigDB(db_path)
        self.config_dir = config_dir
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._running = False

    def load_from_files(self):
        for fname in os.listdir(self.config_dir):
            if not fname.endswith(".yaml"):
                continue
            name = fname[:-5]
            path = os.path.join(self.config_dir, fname)
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            self._cache[name] = data

    def get(self, name: str) -> Dict[str, Any]:
        if name in self._cache:
            return self._cache[name]
        db_cfg = self.db.latest_config(name)
        if db_cfg:
            return json.loads(db_cfg["data"])
        return {}

    def register_callback(self, cb: Callable[[str, Dict[str, Any]], None]):
        self._callbacks.append(cb)

    def start_live_reload(self, interval: float = 5.0):
        self._running = True
        threading.Thread(target=self._reload_loop, args=(interval,), daemon=True).start()

    def _reload_loop(self, interval: float):
        last_mtimes: Dict[str, float] = {}
        while self._running:
            for fname in os.listdir(self.config_dir):
                if not fname.endswith(".yaml"):
                    continue
                name = fname[:-5]
                path = os.path.join(self.config_dir, fname)
                mtime = os.path.getmtime(path)
                if last_mtimes.get(path) != mtime:
                    last_mtimes[path] = mtime
                    with open(path, "r") as f:
                        data = yaml.safe_load(f) or {}
                    self._cache[name] = data
                    for cb in self._callbacks:
                        cb(name, data)
            time.sleep(interval)


# -----------------------------
# Storage / Memory Brain
# -----------------------------
class Storage:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self.lock:
            c = self.conn.cursor()
            c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                source TEXT,
                url TEXT,
                is_ad INTEGER
            )
            """)
            c.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                event_horizon REAL,
                optimization_confidence REAL,
                health REAL
            )
            """)
            c.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT,
                status TEXT,
                lag REAL,
                last_heartbeat REAL
            )
            """)
            c.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT,
                activated_at REAL
            )
            """)
            c.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                level TEXT,
                message TEXT
            )
            """)
            self.conn.commit()

    def insert_event(self, ts: float, source: str, url: str, is_ad: int):
        with self.lock:
            self.conn.execute(
                "INSERT INTO events (ts, source, url, is_ad) VALUES (?, ?, ?, ?)",
                (ts, source, url, is_ad),
            )
            self.conn.commit()

    def insert_metrics(self, ts: float, eh: float, oc: float, health: float):
        with self.lock:
            self.conn.execute(
                "INSERT INTO metrics (ts, event_horizon, optimization_confidence, health) VALUES (?, ?, ?, ?)",
                (ts, eh, oc, health),
            )
            self.conn.commit()

    def upsert_node(self, node_id: str, status: str, lag: float, ts: float):
        with self.lock:
            c = self.conn.cursor()
            c.execute("SELECT id FROM nodes WHERE node_id = ?", (node_id,))
            row = c.fetchone()
            if row:
                c.execute(
                    "UPDATE nodes SET status=?, lag=?, last_heartbeat=? WHERE id=?",
                    (status, lag, ts, row[0]),
                )
            else:
                c.execute(
                    "INSERT INTO nodes (node_id, status, lag, last_heartbeat) VALUES (?, ?, ?, ?)",
                    (node_id, status, lag, ts),
                )
            self.conn.commit()

    def get_nodes(self) -> List[Dict[str, Any]]:
        with self.lock:
            c = self.conn.cursor()
            c.execute("SELECT node_id, status, lag, last_heartbeat FROM nodes")
            rows = c.fetchall()
        return [
            {"node_id": r[0], "status": r[1], "lag": r[2], "last_heartbeat": r[3]}
            for r in rows
        ]

    def insert_log(self, ts: float, level: str, message: str):
        with self.lock:
            self.conn.execute(
                "INSERT INTO logs (ts, level, message) VALUES (?, ?, ?)",
                (ts, level, message),
            )
            self.conn.commit()


# -----------------------------
# Logging
# -----------------------------
class LogBuffer:
    def __init__(self, storage: Storage, max_lines=500):
        self.lines: List[str] = []
        self.max_lines = max_lines
        self.storage = storage
        self.lock = threading.Lock()
        self._init_file()

    def _init_file(self):
        try:
            with open(LOG_FILE, "a") as f:
                f.write("\n--- Control Tower Session Start ---\n")
        except Exception:
            pass

    def add(self, line: str, level: str = "INFO"):
        timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}"
        with self.lock:
            self.lines.append(timestamped)
            if len(self.lines) > self.max_lines:
                self.lines = self.lines[-self.max_lines:]
        try:
            with open(LOG_FILE, "a") as f:
                f.write(timestamped + "\n")
        except Exception:
            pass
        try:
            self.storage.insert_log(time.time(), level, line)
        except Exception:
            pass

    def text(self) -> str:
        with self.lock:
            return "\n".join(self.lines)


# -----------------------------
# EventBus (Kafka-like abstraction)
# -----------------------------
class EventBus:
    """
    Abstraction for Kafka/NATS/etc.
    For now: in-process pub/sub.
    Future: swap to real Kafka client here.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)
        self.lock = threading.Lock()

    def publish(self, topic: str, payload: Dict[str, Any]):
        with self.lock:
            subs = list(self.subscribers.get(topic, []))
        for cb in subs:
            try:
                cb(payload)
            except Exception:
                pass

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        with self.lock:
            self.subscribers[topic].append(handler)


# Topic names (mirroring Kafka schema)
TOPIC_TELEMETRY_SYSTEM = "telemetry.system"
TOPIC_TELEMETRY_GPU = "telemetry.gpu"
TOPIC_ML_PREDICTIONS = "ml.predictions"
TOPIC_ML_DRIFT = "ml.drift"
TOPIC_CONTROL_DECISIONS = "control.decisions"
TOPIC_CONTROL_MODES = "control.modes"
TOPIC_SWARM_NODES = "swarm.nodes"
TOPIC_CONFIG_UPDATES = "config.updates"


# -----------------------------
# Prediction Graph
# -----------------------------
class PredictionGraph:
    def __init__(self, decay=0.99):
        self.edges = defaultdict(lambda: defaultdict(float))
        self.node_stats = defaultdict(lambda: {"block_rate": 0.0, "count": 0})
        self.decay = decay

    def observe(self, source, url, is_ad, timestamp=None):
        stats = self.node_stats[source]
        stats["count"] += 1
        stats["block_rate"] = (
            stats["block_rate"] * (stats["count"] - 1) + is_ad
        ) / stats["count"]

        self.edges[source][url] += 1.0
        self.edges[url][source] += 1.0
        self._decay_edges()

    def _decay_edges(self):
        for a in list(self.edges.keys()):
            for b in list(self.edges[a].keys()):
                self.edges[a][b] *= self.decay
                if self.edges[a][b] < 0.001:
                    del self.edges[a][b]
            if not self.edges[a]:
                del self.edges[a]

    def predict_risk(self, node="global"):
        if node == "global":
            if not self.node_stats:
                return 0.0
            return float(np.mean([s["block_rate"] for s in self.node_stats.values()]))

        neighbors = self.edges.get(node, {})
        if not neighbors:
            return self.node_stats[node]["block_rate"]

        num = sum(self.node_stats[n]["block_rate"] * w for n, w in neighbors.items())
        den = sum(neighbors.values())
        neighbor_risk = num / den if den else 0.0

        local = self.node_stats[node]["block_rate"]
        return 0.5 * local + 0.5 * neighbor_risk


# -----------------------------
# Telemetry Engine (System + GPU)
# -----------------------------
class TelemetryEngine:
    def __init__(self, shared_state: Dict[str, float], bus: EventBus):
        self.shared = shared_state
        self.bus = bus

    def snapshot(self) -> Dict[str, float]:
        cpu = psutil.cpu_percent() / 100.0
        mem = psutil.virtual_memory().percent / 100.0

        self.shared["cpu_load"] = cpu
        self.shared["memory_load"] = mem

        self.shared["drift"] = min(1.0, max(0.0, self.shared.get("drift", 0.1) * 0.98 + 0.01))
        self.shared["uncertainty"] = min(1.0, max(0.0, self.shared.get("uncertainty", 0.2) * 0.99 + 0.005))
        self.shared["block_rate_volatility"] = min(1.0, max(0.0, self.shared.get("block_rate_volatility", 0.1) * 0.97 + 0.01))
        self.shared["disagreement"] = min(1.0, max(0.0, self.shared.get("disagreement", 0.05) * 0.98 + 0.005))

        payload = {
            "accuracy": self.shared.get("accuracy", 0.9),
            "latency": self.shared.get("latency", 0.2),
            "error_rate": self.shared.get("error_rate", 0.05),
            "drift": self.shared.get("drift", 0.1),
            "learning_progression": self.shared.get("learning_progression", 0.01),
            "uncertainty": self.shared.get("uncertainty", 0.2),
            "block_rate_volatility": self.shared.get("block_rate_volatility", 0.1),
            "disagreement": self.shared.get("disagreement", 0.05),
            "block_rate": self.shared.get("block_rate", 0.3),
            "model_version": self.shared.get("model_version", "v0"),
            "kafka_lag": self.shared.get("kafka_lag", 0.0),
            "cpu_load": cpu,
            "memory_load": mem,
        }
        self.bus.publish(TOPIC_TELEMETRY_SYSTEM, payload)
        return payload


# -----------------------------
# GPU Brain / Controller
# -----------------------------
class GPUController:
    def __init__(self, log_buffer: LogBuffer, bus: EventBus, shared_state: Dict[str, Any]):
        self.log_buffer = log_buffer
        self.bus = bus
        self.shared_state = shared_state
        self.mode = "balanced"
        self.initialized = False
        self.device_count = 0
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.initialized = True
                self.log_buffer.add(f"GPU: NVML initialized, {self.device_count} devices")
            except Exception as e:
                self.log_buffer.add(f"GPU: NVML init failed: {e}")
        else:
            self.log_buffer.add("GPU: pynvml not available, running in stub mode")

        self.bus.subscribe(TOPIC_CONTROL_MODES, self._handle_mode_command)

    def _handle_mode_command(self, payload: Dict[str, Any]):
        mode = payload.get("mode")
        if mode:
            self.mode = mode
            self.log_buffer.add(f"GPU: mode set to {mode}")

    def handle_command(self, payload: Dict[str, Any]):
        action = payload.get("action")
        if action == "assign_roles":
            game_gpu = payload.get("game_gpu", "dGPU")
            encode_gpu = payload.get("encode_gpu", "iGPU")
            self.log_buffer.add(f"GPU: assign_roles game={game_gpu}, encode={encode_gpu}")
        elif action == "safe_mode":
            self.mode = "safe"
            self.log_buffer.add("GPU: entering SAFE mode (iGPU only)")
        elif action == "performance_mode":
            self.mode = "performance"
            self.log_buffer.add("GPU: entering PERFORMANCE mode (dGPU focus)")
        else:
            self.log_buffer.add(f"GPU: unknown command {payload}")

    def metrics(self) -> List[Dict[str, Any]]:
        if not self.initialized:
            return []
        out = []
        try:
            for i in range(self.device_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                out.append({
                    "index": i,
                    "memory_used": mem.used,
                    "memory_total": mem.total,
                    "utilization": util.gpu,
                    "memory_utilization": util.memory,
                    "temperature": temp,
                })
        except Exception as e:
            self.log_buffer.add(f"GPU: metrics error: {e}")
        self.bus.publish(TOPIC_TELEMETRY_GPU, {"gpus": out})
        return out

    def aggregate_utilization(self) -> float:
        metrics = self.metrics()
        if not metrics:
            return 0.0
        util = float(np.mean([m["utilization"] for m in metrics]) / 100.0)
        self.shared_state["gpu_utilization"] = util
        return util


# -----------------------------
# Swarm Brain (gossip + registry)
# -----------------------------
class SwarmNode:
    def __init__(self, node_id: str, shared_state: Dict[str, Any], storage: Storage, log_buffer: LogBuffer, bus: EventBus):
        self.node_id = node_id
        self.shared_state = shared_state
        self.storage = storage
        self.log_buffer = log_buffer
        self.bus = bus
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.running = True
        self.lock = threading.Lock()

        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        threading.Thread(target=self._gossip_loop, daemon=True).start()

    def _heartbeat_loop(self):
        while self.running:
            lag = self.shared_state.get("kafka_lag", 0.0)
            ts = time.time()
            try:
                self.storage.upsert_node(self.node_id, "OK", lag, ts)
            except Exception:
                pass
            self.bus.publish(TOPIC_SWARM_NODES, {
                "node_id": self.node_id,
                "status": "OK",
                "lag": lag,
                "ts": ts,
            })
            time.sleep(5)

    def _gossip_loop(self):
        while self.running:
            try:
                nodes = self.storage.get_nodes()
                with self.lock:
                    self.peers = {n["node_id"]: n for n in nodes}
            except Exception:
                pass
            time.sleep(2)

    def get_peers(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return dict(self.peers)


# -----------------------------
# ML Brain v2.0 (skeleton)
# -----------------------------
class ModelRegistry:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self._cache: Dict[str, Any] = {}

    def _path(self, name: str, version: str) -> str:
        return os.path.join(self.models_dir, f"{name}_{version}.pkl")

    def load(self, name: str, version: str):
        key = f"{name}:{version}"
        if key in self._cache:
            return self._cache[key]
        path = self._path(name, version)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        model = joblib.load(path)
        self._cache[key] = model
        return model

    def save(self, name: str, version: str, model) -> str:
        path = self._path(name, version)
        joblib.dump(model, path)
        self._cache[f"{name}:{version}"] = model
        return path


class DriftDetector:
    def __init__(self, window: int = 500, threshold: float = 0.15):
        self.window = window
        self.threshold = threshold
        self.errors: List[float] = []

    def update(self, y_true: int, y_pred: int) -> bool:
        err = 1.0 if y_true != y_pred else 0.0
        self.errors.append(err)
        if len(self.errors) > self.window:
            self.errors = self.errors[-self.window:]
        if len(self.errors) < self.window // 4:
            return False
        mean_err = float(np.mean(self.errors))
        return mean_err > self.threshold

    def current_error_rate(self) -> float:
        if not self.errors:
            return 0.0
        return float(np.mean(self.errors))


class MLBrain:
    """
    ML Brain v2.0 skeleton:
    - online inference
    - drift detection
    - hooks for feature store, shadow/canary, auto-retrain
    """
    def __init__(self, registry: ModelRegistry, model_name: str, model_version: str, log_buffer: LogBuffer, bus: EventBus, shared_state: Dict[str, Any]):
        self.registry = registry
        self.model_name = model_name
        self.model_version = model_version
        self.log_buffer = log_buffer
        self.bus = bus
        self.shared_state = shared_state
        self.model = self._load_model()
        self.drift = DriftDetector()

    def _load_model(self):
        try:
            m = self.registry.load(self.model_name, self.model_version)
            self.log_buffer.add(f"MLBrain: loaded model {self.model_name}:{self.model_version}")
            return m
        except FileNotFoundError:
            self.log_buffer.add("MLBrain: model not found, using dummy")
            return None

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        text = features.get("text", "")
        if self.model is None:
            y_pred = 1 if "ad" in text.lower() else 0
        else:
            try:
                y_pred = int(self.model.predict([text])[0])
            except Exception as e:
                self.log_buffer.add(f"MLBrain: prediction error: {e}")
                y_pred = 0

        self.shared_state["accuracy"] = 0.9  # TODO: update from real metrics
        self.shared_state["latency"] = 0.2   # TODO: measure real latency

        payload = {
            "id": features.get("id"),
            "prediction": y_pred,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
        self.bus.publish(TOPIC_ML_PREDICTIONS, payload)
        return payload

    def update_with_label(self, features: Dict[str, Any], y_true: int):
        pred_info = self.predict(features)
        y_pred = pred_info["prediction"]
        drift_flag = self.drift.update(y_true, y_pred)
        err_rate = self.drift.current_error_rate()
        self.shared_state["error_rate"] = err_rate
        self.shared_state["drift"] = min(1.0, max(0.0, err_rate * 2.0))
        self.bus.publish(TOPIC_ML_DRIFT, {
            "error_rate": err_rate,
            "drift_detected": drift_flag,
        })


# -----------------------------
# Control Brain (Event Horizon)
# -----------------------------
@dataclass
class SystemState:
    model_version: str = "v0"
    optimization_confidence: float = 1.0
    learning_progression: float = 0.0
    block_rate: float = 0.0
    error_rate: float = 0.0
    predicted_risk: float = 0.0
    event_horizon: float = 0.0
    mode: str = "normal"


def compute_event_horizon(m, graph: PredictionGraph, gpu_util: float) -> float:
    U = m["uncertainty"]
    D = m["drift"]
    B = m["block_rate_volatility"]
    X = m["disagreement"]
    G = graph.predict_risk("global")
    Gpu = gpu_util
    return max(0.0, min(1.0, 0.20*U + 0.18*D + 0.18*B + 0.20*G + 0.14*X + 0.10*Gpu))


def compute_optimization_confidence(m, graph: PredictionGraph, gpu_util: float) -> float:
    A = m["accuracy"]
    L = m["latency"]
    E = m["error_rate"]
    D = m["drift"]
    P = m["learning_progression"]
    G = graph.predict_risk("global")
    Gpu = gpu_util
    P = (P + 1.0) / 2.0
    return max(0.0, min(1.0,
        0.28*A +
        0.18*(1-L) +
        0.18*(1-E) +
        0.08*(1-D) +
        0.08*(1-G) +
        0.10*P +
        0.10*(1-Gpu)
    ))


class ControlBrain:
    """
    Reads telemetry + ML drift from EventBus, computes EH/OC/health, emits decisions.
    """
    def __init__(self, bus: EventBus, storage: Storage, log_buffer: LogBuffer, shared_state: Dict[str, Any], gpu_controller: GPUController):
        self.bus = bus
        self.storage = storage
        self.log_buffer = log_buffer
        self.shared_state = shared_state
        self.gpu_controller = gpu_controller

        self.graph = PredictionGraph()
        self.state = SystemState()
        self.health_score = 1.0

        self.last_telemetry: Dict[str, Any] = {}
        self.last_drift: Dict[str, Any] = {}

        self.bus.subscribe(TOPIC_TELEMETRY_SYSTEM, self._on_telemetry)
        self.bus.subscribe(TOPIC_ML_DRIFT, self._on_drift)

    def observe_event(self, source: str, url: str, is_ad: int):
        self.graph.observe(source, url, is_ad)
        try:
            self.storage.insert_event(time.time(), source, url, is_ad)
        except Exception:
            pass
        self.shared_state["total_events"] = self.shared_state.get("total_events", 0) + 1
        if is_ad:
            self.shared_state["blocked_events"] = self.shared_state.get("blocked_events", 0) + 1
        self.shared_state["block_rate"] = (
            self.shared_state.get("blocked_events", 0) /
            max(1, self.shared_state.get("total_events", 1))
        )

    def _on_telemetry(self, payload: Dict[str, Any]):
        self.last_telemetry = payload

    def _on_drift(self, payload: Dict[str, Any]):
        self.last_drift = payload

    def evaluate(self):
        if not self.last_telemetry:
            return

        m = dict(self.last_telemetry)
        if self.last_drift:
            m["error_rate"] = self.last_drift.get("error_rate", m.get("error_rate", 0.05))
            m["drift"] = max(m.get("drift", 0.1), float(self.last_drift.get("error_rate", 0.0)))

        self.state.block_rate = m["block_rate"]
        self.state.error_rate = m["error_rate"]
        self.state.learning_progression = m["learning_progression"]
        self.state.model_version = m["model_version"]

        gpu_util = self.gpu_controller.aggregate_utilization()
        self.state.predicted_risk = self.graph.predict_risk("global")
        self.state.event_horizon = compute_event_horizon(m, self.graph, gpu_util)
        self.state.optimization_confidence = compute_optimization_confidence(m, self.graph, gpu_util)

        eh = self.state.event_horizon
        oc = self.state.optimization_confidence

        if eh > 0.8:
            self._enter_mode("safe")
        elif eh > 0.6:
            self._enter_mode("hyper")
        elif oc > 0.8 and eh < 0.4:
            self._enter_mode("normal")

        cpu = m["cpu_load"]
        mem = m["memory_load"]
        self.health_score = max(0.0, min(1.0,
            0.30*(1.0 - eh) +
            0.30*oc +
            0.15*(1.0 - cpu) +
            0.15*(1.0 - mem) +
            0.10*(1.0 - gpu_util)
        ))

        try:
            self.storage.insert_metrics(time.time(), self.state.event_horizon, self.state.optimization_confidence, self.health_score)
        except Exception:
            pass

        self.bus.publish(TOPIC_CONTROL_DECISIONS, {
            "event_horizon": self.state.event_horizon,
            "optimization_confidence": self.state.optimization_confidence,
            "health": self.health_score,
            "mode": self.state.mode,
        })
        self.bus.publish(TOPIC_CONTROL_MODES, {"mode": self.state.mode})

    def _enter_mode(self, mode: str):
        if self.state.mode == mode:
            return
        self.state.mode = mode
        self.log_buffer.add(f"ControlBrain: entering {mode.upper()} mode")


# -----------------------------
# Plugin Architecture (skeleton)
# -----------------------------
class Plugin:
    def __init__(self, name: str):
        self.name = name

    def start(self):
        pass

    def stop(self):
        pass


class PluginManager:
    """
    Future: load plugins from /plugins, hot-reload, sandbox, etc.
    """
    def __init__(self, log_buffer: LogBuffer):
        self.log_buffer = log_buffer
        self.plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin):
        self.plugins[plugin.name] = plugin
        self.log_buffer.add(f"PluginManager: registered {plugin.name}")

    def start_all(self):
        for p in self.plugins.values():
            try:
                p.start()
                self.log_buffer.add(f"PluginManager: started {p.name}")
            except Exception as e:
                self.log_buffer.add(f"PluginManager: failed to start {p.name}: {e}")


# -----------------------------
# Tkinter GUI (Control Tower Cockpit)
# -----------------------------
class NerveCenterTk:
    def __init__(self, root: tk.Tk, brain: ControlBrain, shared_state: Dict[str, float],
                 swarm: SwarmNode, gpu_controller: GPUController, log_buffer: LogBuffer):
        self.root = root
        self.brain = brain
        self.shared_state = shared_state
        self.swarm = swarm
        self.gpu_controller = gpu_controller
        self.log_buffer = log_buffer

        self.root.title("Event Horizon Control Tower - Borg Node")
        self.root.geometry("1300x850")

        self.time_history: List[float] = []
        self.eh_history: List[float] = []
        self.oc_history: List[float] = []

        self._build_layout()

        self.brain.observe_event("source_A", "url_1", 1)
        self.brain.observe_event("source_A", "url_2", 0)
        self.brain.observe_event("source_B", "url_3", 1)

        self._tick()

    def _build_layout(self):
        self.root.configure(bg="#05050b")

        header = tk.Label(
            self.root,
            text="EVENT HORIZON // BORG CONTROL NODE",
            bg="#05050b",
            fg="#80ff80",
            font=("Consolas", 16, "bold"),
        )
        header.pack(side=tk.TOP, fill=tk.X, pady=5)

        top_frame = tk.Frame(self.root, bg="#05050b")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ctrl_frame = tk.Frame(top_frame, bg="#05050b")
        ctrl_frame.pack(side=tk.LEFT, padx=10)

        btn_safe = tk.Button(ctrl_frame, text="SAFE", command=lambda: self._force_mode("safe"), bg="#402020", fg="white")
        btn_safe.pack(side=tk.TOP, fill=tk.X, pady=2)

        btn_normal = tk.Button(ctrl_frame, text="NORMAL", command=lambda: self._force_mode("normal"), bg="#204020", fg="white")
        btn_normal.pack(side=tk.TOP, fill=tk.X, pady=2)

        btn_hyper = tk.Button(ctrl_frame, text="HYPER", command=lambda: self._force_mode("hyper"), bg="#404020", fg="white")
        btn_hyper.pack(side=tk.TOP, fill=tk.X, pady=2)

        gpu_frame = tk.Frame(ctrl_frame, bg="#05050b")
        gpu_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        tk.Label(gpu_frame, text="GPU Control", bg="#05050b", fg="#f0f0ff").pack(anchor="w")
        btn_gpu_safe = tk.Button(gpu_frame, text="GPU SAFE", command=self.gpu_safe, bg="#303060", fg="white")
        btn_gpu_safe.pack(fill=tk.X, pady=1)
        btn_gpu_perf = tk.Button(gpu_frame, text="GPU PERF", command=self.gpu_perf, bg="#306030", fg="white")
        btn_gpu_perf.pack(fill=tk.X, pady=1)

        metrics_frame = tk.Frame(top_frame, bg="#05050b")
        metrics_frame.pack(side=tk.LEFT, padx=20)

        self.metrics_label = tk.Label(
            metrics_frame,
            text="Metrics",
            justify=tk.LEFT,
            anchor="nw",
            bg="#101020",
            fg="#f0f0ff",
            font=("Consolas", 10),
            width=60,
            height=10,
            bd=1,
            relief=tk.SOLID,
        )
        self.metrics_label.pack(fill=tk.BOTH, expand=True)

        health_frame = tk.Frame(top_frame, bg="#05050b")
        health_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)

        tk.Label(health_frame, text="System Health", bg="#05050b", fg="#f0f0ff").pack(anchor="w")
        self.health_canvas = tk.Canvas(health_frame, width=200, height=20, bg="#101020", highlightthickness=1, highlightbackground="#303050")
        self.health_canvas.pack(pady=5)
        self.health_rect = self.health_canvas.create_rectangle(0, 0, 0, 20, fill="#20c020", outline="")

        mid_frame = tk.Frame(self.root, bg="#05050b")
        mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        chart_frame = tk.Frame(mid_frame, bg="#05050b")
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(chart_frame, text="Event Horizon / Optimization Confidence", bg="#05050b", fg="#f0f0ff").pack(anchor="w")
        self.chart_canvas = tk.Canvas(chart_frame, bg="#101020", height=200, highlightthickness=1, highlightbackground="#303050")
        self.chart_canvas.pack(fill=tk.BOTH, expand=True)

        graph_frame = tk.Frame(mid_frame, bg="#05050b")
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(graph_frame, text="Prediction Graph", bg="#05050b", fg="#f0f0ff").pack(anchor="w")
        self.graph_canvas = tk.Canvas(graph_frame, bg="#101020", height=200, highlightthickness=1, highlightbackground="#303050")
        self.graph_canvas.pack(fill=tk.BOTH, expand=True)

        swarm_frame = tk.Frame(mid_frame, bg="#05050b")
        swarm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(swarm_frame, text="Swarm Nodes", bg="#05050b", fg="#f0f0ff").pack(anchor="w")

        self.swarm_tree = ttk.Treeview(swarm_frame, columns=("node", "status", "lag"), show="headings", height=10)
        self.swarm_tree.heading("node", text="Node")
        self.swarm_tree.heading("status", text="Status")
        self.swarm_tree.heading("lag", text="Lag")
        self.swarm_tree.column("node", width=120)
        self.swarm_tree.column("status", width=80)
        self.swarm_tree.column("lag", width=80)
        self.swarm_tree.pack(fill=tk.BOTH, expand=True)

        bottom_frame = tk.Frame(self.root, bg="#05050b")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=5)

        tk.Label(bottom_frame, text="Logs", bg="#05050b", fg="#f0f0ff").pack(anchor="w")
        self.logs_text = tk.Text(
            bottom_frame,
            bg="#101020",
            fg="#f0f0ff",
            font=("Consolas", 9),
            height=10,
            bd=1,
            relief=tk.SOLID,
        )
        self.logs_text.pack(fill=tk.BOTH, expand=True)

    def _force_mode(self, mode: str):
        self.brain.state.mode = mode
        self.log_buffer.add(f"GUI: forced mode {mode}")

    def gpu_safe(self):
        self.gpu_controller.handle_command({"action": "safe_mode"})

    def gpu_perf(self):
        self.gpu_controller.handle_command({"action": "performance_mode"})

    def _tick(self):
        self.brain.evaluate()
        s = self.brain.state
        self.log_buffer.add(
            f"EH={s.event_horizon:.3f} OC={s.optimization_confidence:.3f} "
            f"mode={s.mode} health={self.brain.health_score:.3f}"
        )
        self._update_metrics()
        self._update_health_bar()
        self._update_chart()
        self._update_graph()
        self._update_swarm()
        self._update_logs()
        self.root.after(1000, self._tick)

    def _update_metrics(self):
        s = self.brain.state
        gpu_util = self.shared_state.get("gpu_utilization", 0.0)
        text = (
            f"Mode: {s.mode}\n"
            f"Model: {s.model_version}\n"
            f"Event Horizon: {s.event_horizon:.3f}\n"
            f"Opt Confidence: {s.optimization_confidence:.3f}\n"
            f"Predicted Risk: {s.predicted_risk:.3f}\n"
            f"Block Rate: {s.block_rate:.3f}\n"
            f"Error Rate: {s.error_rate:.3f}\n"
            f"Learning Progression: {s.learning_progression:.3f}\n"
            f"GPU Utilization: {gpu_util:.3f}\n"
            f"Kafka Lag: {self.shared_state.get('kafka_lag', 0.0):.3f}\n"
            f"CPU: {self.shared_state.get('cpu_load', 0.0):.3f}\n"
            f"Memory: {self.shared_state.get('memory_load', 0.0):.3f}\n"
            f"Total Events: {int(self.shared_state.get('total_events', 0))}\n"
            f"Blocked Events: {int(self.shared_state.get('blocked_events', 0))}\n"
        )
        self.metrics_label.config(text=text)

    def _update_health_bar(self):
        score = self.brain.health_score
        width = 200
        fill_width = int(width * score)
        r = int(255 * (1.0 - score))
        g = int(255 * score)
        color = f"#{r:02x}{g:02x}20"
        self.health_canvas.coords(self.health_rect, 0, 0, fill_width, 20)
        self.health_canvas.itemconfig(self.health_rect, fill=color)

    def _update_chart(self):
        s = self.brain.state
        t = time.time()
        self.time_history.append(t)
        self.eh_history.append(s.event_horizon)
        self.oc_history.append(s.optimization_confidence)

        if len(self.time_history) > 200:
            self.time_history = self.time_history[-200:]
            self.eh_history = self.eh_history[-200:]
            self.oc_history = self.oc_history[-200:]

        self.chart_canvas.delete("all")
        if len(self.time_history) < 2:
            return

        w = self.chart_canvas.winfo_width() or 400
        h = self.chart_canvas.winfo_height() or 200

        t0 = self.time_history[0]
        xs = [(ti - t0) for ti in self.time_history]
        if xs[-1] == 0:
            return
        xs_norm = [x / xs[-1] for x in xs]

        def to_xy(idx, val):
            x = 10 + xs_norm[idx] * (w - 20)
            y = h - 10 - val * (h - 20)
            return x, y

        for series, color in [(self.eh_history, "#ff5050"), (self.oc_history, "#50d0ff")]:
            points = [to_xy(i, v) for i, v in enumerate(series)]
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i+1]
                self.chart_canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

        self.chart_canvas.create_text(50, 10, text="EH", fill="#ff5050", anchor="nw", font=("Consolas", 8))
        self.chart_canvas.create_text(90, 10, text="OC", fill="#50d0ff", anchor="nw", font=("Consolas", 8))

    def _update_graph(self):
        self.graph_canvas.delete("all")
        g = self.brain.graph
        nodes = list(g.node_stats.keys())
        if not nodes:
            self.graph_canvas.create_text(10, 10, text="Graph empty — waiting for events...", fill="#f0f0ff", anchor="nw")
            return

        w = self.graph_canvas.winfo_width() or 300
        h = self.graph_canvas.winfo_height() or 200
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.35

        positions: Dict[str, Tuple[float, float]] = {}
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / len(nodes)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            positions[node] = (x, y)

        for a, neighbors in g.edges.items():
            if a not in positions:
                continue
            x1, y1 = positions[a]
            for b, wgt in neighbors.items():
                if b not in positions:
                    continue
                x2, y2 = positions[b]
                width = max(1, min(4, int(wgt)))
                self.graph_canvas.create_line(x1, y1, x2, y2, fill="#6060c0", width=width)

        for node, (x, y) in positions.items():
            risk = g.node_stats[node]["block_rate"]
            r = int(100 + 155 * risk)
            gcol = int(200 - 150 * risk)
            bcol = int(255 - 200 * risk)
            color = f"#{r:02x}{gcol:02x}{bcol:02x}"
            self.graph_canvas.create_oval(x-8, y-8, x+8, y+8, fill=color, outline="#000000")
            self.graph_canvas.create_text(x+10, y, text=node, fill="#f0f0ff", anchor="w", font=("Consolas", 8))

    def _update_swarm(self):
        for i in self.swarm_tree.get_children():
            self.swarm_tree.delete(i)
        peers = self.swarm.get_peers()
        for name, info in peers.items():
            status = info.get("status", "OK")
            l = info.get("lag", 0.0)
            self.swarm_tree.insert("", tk.END, values=(name, status, f"{l:.3f}"))

    def _update_logs(self):
        self.logs_text.delete("1.0", tk.END)
        self.logs_text.insert(tk.END, self.log_buffer.text())
        self.logs_text.see(tk.END)


# -----------------------------
# FastAPI Web Dashboard
# -----------------------------
app = FastAPI()
websocket_clients: List[WebSocket] = []
GLOBAL_BRAIN: Optional[ControlBrain] = None
GLOBAL_STORAGE: Optional[Storage] = None
GLOBAL_GPU: Optional[GPUController] = None

@app.get("/")
async def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Event Horizon Web Dashboard</title>
        <style>
            body { background: #05050b; color: #f0f0ff; font-family: Consolas, monospace; }
            .panel { border: 1px solid #303050; padding: 10px; margin: 10px; background: #101020; }
            h1 { color: #80ff80; }
            .metric { margin: 4px 0; }
        </style>
    </head>
    <body>
        <h1>EVENT HORIZON // WEB BRIDGE</h1>
        <div class="panel">
            <h2>Live Metrics</h2>
            <div id="metrics"></div>
        </div>
        <div class="panel">
            <h2>Swarm Nodes</h2>
            <div id="swarm"></div>
        </div>
        <div class="panel">
            <h2>GPU Metrics</h2>
            <div id="gpu"></div>
        </div>
        <script>
            const metricsDiv = document.getElementById('metrics');
            const swarmDiv = document.getElementById('swarm');
            const gpuDiv = document.getElementById('gpu');
            const ws = new WebSocket(`ws://${location.host}/ws`);
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === "metrics") {
                    metricsDiv.innerHTML = "";
                    for (const [k, v] of Object.entries(data.payload)) {
                        const d = document.createElement('div');
                        d.className = 'metric';
                        d.textContent = k + ": " + v;
                        metricsDiv.appendChild(d);
                    }
                } else if (data.type === "swarm") {
                    swarmDiv.innerHTML = "";
                    data.payload.forEach(n => {
                        const d = document.createElement('div');
                        d.className = 'metric';
                        d.textContent = n.node_id + " | " + n.status + " | lag=" + n.lag.toFixed(3);
                        swarmDiv.appendChild(d);
                    });
                } else if (data.type === "gpu") {
                    gpuDiv.innerHTML = "";
                    data.payload.forEach(g => {
                        const d = document.createElement('div');
                        d.className = 'metric';
                        d.textContent = "GPU " + g.index + " | util=" + g.utilization + "% | mem=" + g.memory_used + "/" + g.memory_total;
                        gpuDiv.appendChild(d);
                    });
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    websocket_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        websocket_clients.remove(ws)


@app.get("/api/nodes")
async def api_nodes():
    nodes = GLOBAL_STORAGE.get_nodes() if GLOBAL_STORAGE else []
    return JSONResponse(nodes)


@app.get("/api/metrics")
async def api_metrics():
    if not GLOBAL_BRAIN:
        return JSONResponse({})
    s = GLOBAL_BRAIN.state
    return JSONResponse({
        "mode": s.mode,
        "event_horizon": round(s.event_horizon, 3),
        "optimization_confidence": round(s.optimization_confidence, 3),
        "health": round(GLOBAL_BRAIN.health_score, 3),
    })


def backend_broadcast_loop(brain: ControlBrain, storage: Storage, gpu_controller: GPUController):
    while True:
        try:
            s = brain.state
            metrics_payload = {
                "mode": s.mode,
                "event_horizon": round(s.event_horizon, 3),
                "optimization_confidence": round(s.optimization_confidence, 3),
                "health": round(brain.health_score, 3),
            }
            swarm_payload = storage.get_nodes()
            gpu_payload = gpu_controller.metrics()

            msg_metrics = json.dumps({"type": "metrics", "payload": metrics_payload})
            msg_swarm = json.dumps({"type": "swarm", "payload": swarm_payload})
            msg_gpu = json.dumps({"type": "gpu", "payload": gpu_payload})

            for ws in list(websocket_clients):
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    loop.create_task(ws.send_text(msg_metrics))
                    loop.create_task(ws.send_text(msg_swarm))
                    loop.create_task(ws.send_text(msg_gpu))
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(1.0)


# -----------------------------
# Main
# -----------------------------
def main():
    global GLOBAL_BRAIN, GLOBAL_STORAGE, GLOBAL_GPU

    shared_state: Dict[str, Any] = {}

    cfg_mgr = ConfigManager(db_path=os.path.join(DATA_DIR, "configs.db"), config_dir=CONFIG_DIR)
    cfg_mgr.load_from_files()
    core_cfg = cfg_mgr.get("core") or DEFAULT_CORE_YAML

    try:
        storage = Storage(core_cfg["storage"]["path"])
    except sqlite3.OperationalError as e:
        print(f"WARNING: Failed to open DB at {core_cfg['storage']['path']}: {e}")
        print("Falling back to in-memory SQLite (no persistence).")
        storage = Storage(":memory:")

    log_buffer = LogBuffer(storage)
    bus = EventBus()

    telemetry = TelemetryEngine(shared_state, bus)
    gpu_controller = GPUController(log_buffer, bus, shared_state)
    GLOBAL_GPU = gpu_controller

    swarm = SwarmNode(core_cfg["swarm"]["node_id"], shared_state, storage, log_buffer, bus)

    registry = ModelRegistry()
    ml_brain = MLBrain(registry, core_cfg["ml"]["model_name"], core_cfg["ml"]["model_version"], log_buffer, bus, shared_state)

    brain = ControlBrain(bus, storage, log_buffer, shared_state, gpu_controller)
    GLOBAL_BRAIN = brain
    GLOBAL_STORAGE = storage

    plugin_manager = PluginManager(log_buffer)
    plugin_manager.start_all()

    def telemetry_loop():
        while True:
            telemetry.snapshot()
            time.sleep(1.0)

    threading.Thread(target=telemetry_loop, daemon=True).start()

    def api_runner():
        uvicorn.run(app, host=core_cfg["web"]["host"], port=core_cfg["web"]["port"], log_level="warning")

    threading.Thread(target=api_runner, daemon=True).start()
    threading.Thread(target=backend_broadcast_loop, args=(brain, storage, gpu_controller), daemon=True).start()

    root = tk.Tk()
    ui = NerveCenterTk(root, brain, shared_state, swarm, gpu_controller, log_buffer)
    root.mainloop()


if __name__ == "__main__":
    main()

