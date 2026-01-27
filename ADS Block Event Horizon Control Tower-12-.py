"""
EVENT HORIZON // BORG CONTROL TOWER
DISTRIBUTED SWARM // ULTRA-FUTURISTIC COCKPIT + BORG SOVEREIGN SHELL
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
import hashlib
import socket
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Optional, Tuple
from collections import defaultdict

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

REQUIRED_LIBRARIES = [
    "numpy",
    "psutil",
    "fastapi",
    "uvicorn",
    "joblib",
    "pynvml",
    "yaml",
    "aiokafka",
    "scikit-learn",
    "requests",
    "selenium",
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import asyncio

import requests
from selenium import webdriver

# -----------------------------
# Paths, DB, PID Lock
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_FILE = os.path.join(DATA_DIR, "tower.db")
LOCK_FILE = os.path.join(DATA_DIR, "tower.lock")
LOG_FILE = os.path.join(DATA_DIR, "control_tower.log")

import psutil as _psutil_for_lock

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
# Config System
# -----------------------------
CONFIG_DIR = os.path.join(BASE_DIR, "config")
os.makedirs(CONFIG_DIR, exist_ok=True)

DEFAULT_CORE_YAML = {
    "kafka": {
        "brokers": ["localhost:9092"],
        "client_id": "borg-node-1",
        "group_id": "borg-node-group",
        "enabled": False,
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
    "selfopt": {
        "eh_safe_threshold": 0.80,
        "eh_hyper_threshold": 0.60,
        "oc_normal_threshold": 0.80,
        "aggressiveness": 0.50,
    },
    "_version": 1,
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

class ConfigWriter:
    def __init__(self, config_path: str, log_buffer: "LogBuffer", min_interval_sec: int = 300):
        self.config_path = config_path
        self.log_buffer = log_buffer
        self.min_interval = min_interval_sec
        self.last_write = 0
        self.last_good_version = None
        self.last_good_snapshot: Dict[str, Any] = {}

    def write(self, new_data: Dict[str, Any]) -> int:
        now = time.time()
        if now - self.last_write < self.min_interval:
            return -1
        try:
            with open(self.config_path, "r") as f:
                current = yaml.safe_load(f) or {}
        except Exception:
            current = {}
        merged = {**current, **new_data}
        version = merged.get("_version", 0) + 1
        merged["_version"] = version
        tmp_path = self.config_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                yaml.safe_dump(merged, f)
            os.replace(tmp_path, self.config_path)
            self.last_write = now
            self.log_buffer.add(f"ConfigWriter: wrote version {version} to core.yaml")
            return version
        except Exception as e:
            self.log_buffer.add(f"ConfigWriter: write failed: {e}")
            return -1

    def mark_good(self, snapshot: Dict[str, Any], version: int):
        self.last_good_snapshot = dict(snapshot)
        self.last_good_version = version
        self.log_buffer.add(f"ConfigWriter: marked version {version} as GOOD")

    def rollback(self) -> Optional[int]:
        if self.last_good_version is None:
            self.log_buffer.add("ConfigWriter: rollback requested but no GOOD version known")
            return None
        tmp_path = self.config_path + ".tmp"
        try:
            snap = dict(self.last_good_snapshot)
            snap["_version"] = self.last_good_version + 1
            with open(tmp_path, "w") as f:
                yaml.safe_dump(snap, f)
            os.replace(tmp_path, self.config_path)
            self.log_buffer.add(f"ConfigWriter: rolled back to snapshot (new version {self.last_good_version + 1})")
            return self.last_good_version + 1
        except Exception as e:
            self.log_buffer.add(f"ConfigWriter: rollback failed: {e}")
            return None

# -----------------------------
# Storage
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
                node_id TEXT,
                model_name TEXT,
                model_version TEXT,
                ts REAL
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
            c.execute("""
            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                text TEXT,
                label INTEGER
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

    def insert_model_version(self, node_id: str, model_name: str, model_version: str):
        with self.lock:
            self.conn.execute(
                "INSERT INTO model_versions (node_id, model_name, model_version, ts) VALUES (?, ?, ?, ?)",
                (node_id, model_name, model_version, time.time()),
            )
            self.conn.commit()

    def get_model_versions(self) -> List[Dict[str, Any]]:
        with self.lock:
            c = self.conn.cursor()
            c.execute("SELECT node_id, model_name, model_version, ts FROM model_versions")
            rows = c.fetchall()
        return [
            {"node_id": r[0], "model_name": r[1], "model_version": r[2], "ts": r[3]}
            for r in rows
        ]

    def insert_log(self, ts: float, level: str, message: str):
        with self.lock:
            self.conn.execute(
                "INSERT INTO logs (ts, level, message) VALUES (?, ?, ?)",
                (ts, level, message),
            )
            self.conn.commit()

    def insert_training_sample(self, text: str, label: int):
        with self.lock:
            self.conn.execute(
                "INSERT INTO training_samples (ts, text, label) VALUES (?, ?, ?)",
                (time.time(), text, label),
            )
            self.conn.commit()

    def get_training_samples(self, limit: int = 1000) -> List[Tuple[str, int]]:
        with self.lock:
            c = self.conn.cursor()
            c.execute("SELECT text, label FROM training_samples ORDER BY id DESC LIMIT ?", (limit,))
            rows = c.fetchall()
        return [(r[0], r[1]) for r in rows]

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
# EventBus (InMemory + Kafka)
# -----------------------------
TOPIC_TELEMETRY_SYSTEM = "telemetry.system"
TOPIC_TELEMETRY_GPU = "telemetry.gpu"
TOPIC_ML_PREDICTIONS = "ml.predictions"
TOPIC_ML_DRIFT = "ml.drift"
TOPIC_CONTROL_DECISIONS = "control.decisions"
TOPIC_CONTROL_MODES = "control.modes"
TOPIC_SWARM_NODES = "swarm.nodes"
TOPIC_CONFIG_UPDATES = "config.updates"
TOPIC_SWARM_POLICY = "swarm.policy"
TOPIC_SWARM_MODEL = "swarm.model"
TOPIC_SWARM_HEARTBEAT = "swarm.heartbeat"
TOPIC_TRAINING_SAMPLES = "ml.training.samples"

class EventBusBase:
    def publish(self, topic: str, payload: Dict[str, Any]): ...
    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]): ...

class InMemoryEventBus(EventBusBase):
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

class KafkaEventBus(EventBusBase):
    def __init__(self, brokers: List[str], group_id: str, log_buffer: LogBuffer):
        self.brokers = brokers
        self.group_id = group_id
        self.log_buffer = log_buffer
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)
        self.loop = asyncio.new_event_loop()
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumer: Optional[AIOKafkaConsumer] = None
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_main())

    async def _async_main(self):
        try:
            self.producer = AIOKafkaProducer(
                loop=self.loop,
                bootstrap_servers=self.brokers,
                client_id="event-horizon-producer",
            )
            await self.producer.start()
            self.consumer = AIOKafkaConsumer(
                TOPIC_TELEMETRY_SYSTEM,
                TOPIC_TELEMETRY_GPU,
                TOPIC_ML_PREDICTIONS,
                TOPIC_ML_DRIFT,
                TOPIC_CONTROL_DECISIONS,
                TOPIC_CONTROL_MODES,
                TOPIC_SWARM_POLICY,
                TOPIC_SWARM_MODEL,
                TOPIC_SWARM_HEARTBEAT,
                TOPIC_TRAINING_SAMPLES,
                loop=self.loop,
                bootstrap_servers=self.brokers,
                group_id=self.group_id,
                enable_auto_commit=True,
            )
            await self.consumer.start()
            self.log_buffer.add("KafkaEventBus: started producer and consumer")
            async for msg in self.consumer:
                topic = msg.topic
                try:
                    payload = json.loads(msg.value.decode("utf-8"))
                except Exception:
                    continue
                handlers = list(self.subscribers.get(topic, []))
                for h in handlers:
                    try:
                        h(payload)
                    except Exception:
                        pass
        except Exception as e:
            self.log_buffer.add(f"KafkaEventBus: error {e}")
        finally:
            if self.consumer:
                await self.consumer.stop()
            if self.producer:
                await self.producer.stop()

    def publish(self, topic: str, payload: Dict[str, Any]):
        if not self.producer:
            return
        data = json.dumps(payload).encode("utf-8")
        async def _send():
            try:
                await self.producer.send_and_wait(topic, data)
            except Exception as e:
                self.log_buffer.add(f"KafkaEventBus: publish error {e}")
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(_send()))

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]):
        self.subscribers[topic].append(handler)

# -----------------------------
# Prediction Graph + Telemetry
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

class TelemetryEngine:
    def __init__(self, shared_state: Dict[str, float], bus: EventBusBase):
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
# GPU Controller + Steering
# -----------------------------
class GPUController:
    def __init__(self, log_buffer: LogBuffer, bus: EventBusBase, shared_state: Dict[str, Any]):
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
            self.log_buffer.add("GPU: pynvml not available, stub mode")
        self.bus.subscribe(TOPIC_CONTROL_MODES, self._handle_mode_command)

    def _handle_mode_command(self, payload: Dict[str, Any]):
        mode = payload.get("mode")
        if mode:
            self.mode = mode
            self.log_buffer.add(f"GPU: mode set to {mode}")

    def handle_command(self, payload: Dict[str, Any]):
        action = payload.get("action")
        if action == "safe_mode":
            self.mode = "safe"
            self.log_buffer.add("GPU: entering SAFE mode")
        elif action == "performance_mode":
            self.mode = "performance"
            self.log_buffer.add("GPU: entering PERFORMANCE mode")
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
            util = 0.0
        else:
            util = float(np.mean([m["utilization"] for m in metrics]) / 100.0)
        self.shared_state["gpu_utilization"] = util
        return util

    def inspect_processes(self) -> List[Dict[str, Any]]:
        procs = []
        try:
            for p in psutil.process_iter(["pid", "name", "cpu_percent"]):
                procs.append({
                    "pid": p.info["pid"],
                    "name": p.info["name"],
                    "cpu": p.info["cpu_percent"],
                })
        except Exception:
            pass
        return procs

    def apply_steering_policy(self):
        procs = self.inspect_processes()
        for p in procs:
            name = (p["name"] or "").lower()
            if "obs" in name or "encoder" in name:
                self.log_buffer.add(f"GPU: would steer encoder-like process {p['pid']} to iGPU")
            if "game" in name or "steam" in name:
                self.log_buffer.add(f"GPU: would steer game-like process {p['pid']} to dGPU")

# -----------------------------
# Swarm Node
# -----------------------------
class SwarmNode:
    def __init__(self, node_id: str, shared_state: Dict[str, Any],
                 storage: Storage, log_buffer: LogBuffer, bus: EventBusBase):
        self.node_id = node_id
        self.shared_state = shared_state
        self.storage = storage
        self.log_buffer = log_buffer
        self.bus = bus
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.running = True
        self.lock = threading.Lock()
        self.policy_version = 0
        self.policy_hash = ""
        self.policy_votes: Dict[str, Tuple[int, str]] = {}
        self.model_name = ""
        self.model_version = ""
        self.model_votes: Dict[str, Tuple[str, str]] = {}
        self.bus.subscribe(TOPIC_SWARM_HEARTBEAT, self._on_heartbeat)
        self.bus.subscribe(TOPIC_SWARM_POLICY, self._on_policy_announce)
        self.bus.subscribe(TOPIC_SWARM_MODEL, self._on_model_announce)
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _heartbeat_loop(self):
        while self.running:
            lag = self.shared_state.get("kafka_lag", 0.0)
            ts = time.time()
            try:
                self.storage.upsert_node(self.node_id, "OK", lag, ts)
            except Exception:
                pass
            payload = {
                "node_id": self.node_id,
                "status": "OK",
                "lag": lag,
                "ts": ts,
            }
            self.bus.publish(TOPIC_SWARM_HEARTBEAT, payload)
            time.sleep(5)

    def _on_heartbeat(self, payload: Dict[str, Any]):
        node = payload.get("node_id")
        if not node:
            return
        with self.lock:
            self.peers[node] = payload

    def get_peers(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return dict(self.peers)

    def announce_policy(self, version: int, policy: Dict[str, Any]):
        payload_str = json.dumps(policy, sort_keys=True)
        h = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        self.policy_version = version
        self.policy_hash = h
        self.bus.publish(TOPIC_SWARM_POLICY, {
            "node_id": self.node_id,
            "version": version,
            "hash": h,
        })

    def _on_policy_announce(self, payload: Dict[str, Any]):
        node = payload.get("node_id")
        version = payload.get("version")
        h = payload.get("hash")
        if not node or version is None or not h:
            return
        self.policy_votes[node] = (version, h)
        self._check_policy_consensus()

    def _check_policy_consensus(self):
        if not self.policy_votes:
            return
        counts = defaultdict(int)
        for (v, h) in self.policy_votes.values():
            counts[(v, h)] += 1
        (best_v, best_h), cnt = max(counts.items(), key=lambda kv: kv[1])
        total = len(self.policy_votes)
        if cnt >= max(1, total // 2 + 1):
            if best_v == self.policy_version and best_h == self.policy_hash:
                self.shared_state["selfopt_swarm_approved"] = True
                self.shared_state["selfopt_swarm_version"] = best_v
                self.log_buffer.add(f"SwarmNode: policy v{best_v} consensus ({cnt}/{total})")
            else:
                self.shared_state["selfopt_swarm_approved"] = False

    def announce_model(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.bus.publish(TOPIC_SWARM_MODEL, {
            "node_id": self.node_id,
            "model_name": model_name,
            "model_version": model_version,
        })
        try:
            self.storage.insert_model_version(self.node_id, model_name, model_version)
        except Exception:
            pass

    def _on_model_announce(self, payload: Dict[str, Any]):
        node = payload.get("node_id")
        name = payload.get("model_name")
        version = payload.get("model_version")
        if not node or not name or not version:
            return
        self.model_votes[node] = (name, version)
        self._check_model_consensus()

    def _check_model_consensus(self):
        if not self.model_votes:
            return
        counts = defaultdict(int)
        for (name, ver) in self.model_votes.values():
            counts[(name, ver)] += 1
        (best_name, best_ver), cnt = max(counts.items(), key=lambda kv: kv[1])
        total = len(self.model_votes)
        if cnt >= max(1, total // 2 + 1):
            self.shared_state["swarm_model_name"] = best_name
            self.shared_state["swarm_model_version"] = best_ver
            self.shared_state["swarm_model_consensus"] = True
            self.log_buffer.add(
                f"SwarmNode: model consensus {best_name}:{best_ver} ({cnt}/{total})"
            )

# -----------------------------
# Model Registry (safe dir + integrity)
# -----------------------------
class ModelRegistry:
    def __init__(self, models_dir: str = None, log_buffer: Optional[LogBuffer] = None):
        if models_dir is None:
            base = os.environ.get("LOCALAPPDATA")
            if base:
                models_dir = os.path.join(base, "EventHorizonModels")
            else:
                models_dir = os.path.join(os.path.expanduser("~"), "EventHorizonModels")
        try:
            os.makedirs(models_dir, exist_ok=True)
        except PermissionError:
            import tempfile
            models_dir = os.path.join(tempfile.gettempdir(), "EventHorizonModels")
            os.makedirs(models_dir, exist_ok=True)
        self.models_dir = models_dir
        self._cache: Dict[str, Any] = {}
        self.log_buffer = log_buffer
        try:
            print(f"[ModelRegistry] Using model directory: {self.models_dir}")
        except Exception:
            pass

    def _path(self, name: str, version: str) -> str:
        return os.path.abspath(os.path.join(self.models_dir, f"{name}_{version}.pkl"))

    def _hash_path(self, name: str, version: str) -> str:
        return os.path.abspath(os.path.join(self.models_dir, f"{name}_{version}.sha256"))

    def _compute_hash(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_hash(self, name: str, version: str, digest: str):
        with open(self._hash_path(name, version), "w") as f:
            f.write(digest)

    def _read_hash(self, name: str, version: str) -> Optional[str]:
        hp = self._hash_path(name, version)
        if not os.path.exists(hp):
            return None
        with open(hp, "r") as f:
            return f.read().strip()

    def load(self, name: str, version: str):
        key = f"{name}:{version}"
        if key in self._cache:
            return self._cache[key]
        path = self._path(name, version)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        expected = self._read_hash(name, version)
        if expected:
            actual = self._compute_hash(path)
            if actual != expected:
                if self.log_buffer:
                    self.log_buffer.add(f"ModelRegistry: integrity FAILED for {name}:{version}")
                raise RuntimeError("Model integrity check failed")
        model = joblib.load(path)
        self._cache[key] = model
        return model

    def save(self, name: str, version: str, model) -> str:
        path = self._path(name, version)
        joblib.dump(model, path)
        digest = self._compute_hash(path)
        self._write_hash(name, version, digest)
        self._cache[f"{name}:{version}"] = model
        if self.log_buffer:
            self.log_buffer.add(f"ModelRegistry: saved {name}:{version} hash {digest[:8]}...")
        return path

# -----------------------------
# Drift Detector + MLBrain
# -----------------------------
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
    def __init__(self, registry: ModelRegistry, model_name: str, model_version: str,
                 log_buffer: LogBuffer, bus: EventBusBase, shared_state: Dict[str, Any],
                 swarm: Optional[SwarmNode]):
        self.registry = registry
        self.model_name = model_name
        self.model_version = model_version
        self.log_buffer = log_buffer
        self.bus = bus
        self.shared_state = shared_state
        self.swarm = swarm
        self.model = self._load_model()
        self.drift = DriftDetector()
        self.drift_window_sec = 600
        self.drift_threshold = 0.20
        self.last_retrain = 0
        self.min_retrain_interval = 900
        self.drift_history: List[Tuple[float, float]] = []
        self.bus.subscribe(TOPIC_SWARM_MODEL, self._on_swarm_model)

    def _load_model(self):
        try:
            m = self.registry.load(self.model_name, self.model_version)
            self.log_buffer.add(f"MLBrain: loaded model {self.model_name}:{self.model_version}")
            return m
        except FileNotFoundError:
            self.log_buffer.add("MLBrain: model not found, using dummy")
            return None
        except RuntimeError as e:
            self.log_buffer.add(f"MLBrain: integrity error: {e}")
            return None

    def _on_swarm_model(self, payload: Dict[str, Any]):
        name = payload.get("model_name")
        version = payload.get("model_version")
        if not name or not version:
            return
        if name == self.model_name and version == self.model_version:
            return
        try:
            self.log_buffer.add(f"MLBrain: adopting swarm model {name}:{version}")
            self.model_name = name
            self.model_version = version
            self.model = self._load_model()
            self.shared_state["model_version"] = self.model_version
        except Exception as e:
            self.log_buffer.add(f"MLBrain: failed to adopt swarm model {name}:{version}: {e}")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        text = features.get("text", "")
        if self.model is None:
            y_pred = 1 if "ad" in text.lower() else 0
        else:
            try:
                vec, clf = self.model
                X = vec.transform([text])
                y_pred = int(clf.predict(X)[0])
            except Exception as e:
                self.log_buffer.add(f"MLBrain: prediction error: {e}")
                y_pred = 0
        self.shared_state["accuracy"] = 0.9
        self.shared_state["latency"] = 0.2
        payload = {
            "id": features.get("id"),
            "prediction": y_pred,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
        self.bus.publish(TOPIC_ML_PREDICTIONS, payload)
        return payload

    def update_with_label(self, features: Dict[str, Any], y_true: int, storage: Storage):
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
        storage.insert_training_sample(features.get("text", ""), y_true)
        now = time.time()
        self.drift_history.append((now, err_rate))
        self._maybe_retrain(now, storage)

    def _maybe_retrain(self, now: float, storage: Storage):
        self.drift_history = [(t, e) for (t, e) in self.drift_history if now - t <= self.drift_window_sec]
        if len(self.drift_history) < 10:
            return
        avg_err = float(np.mean([e for (_, e) in self.drift_history]))
        self.shared_state["drift_avg_error"] = avg_err
        if avg_err < self.drift_threshold:
            return
        if now - self.last_retrain < self.min_retrain_interval:
            return
        self.log_buffer.add(f"MLBrain: drift high (avg_err={avg_err:.3f}), auto-retrain")
        self._auto_retrain(storage)
        self.last_retrain = now

    def _auto_retrain(self, storage: Storage):
        try:
            samples = storage.get_training_samples(limit=1000)
            if len(samples) < 20:
                self.log_buffer.add("MLBrain: not enough training samples")
                return
            texts, labels = zip(*samples)
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.linear_model import LogisticRegression
            vec = CountVectorizer()
            X = vec.fit_transform(texts)
            y = np.array(labels)
            clf = LogisticRegression(max_iter=200)
            clf.fit(X, y)
            model = (vec, clf)
            old_version = self.model_version
            new_version = f"{old_version}_auto_{int(time.time())}"
            self.registry.save(self.model_name, new_version, model)
            self.model_version = new_version
            self.model = model
            self.shared_state["model_version"] = self.model_version
            self.log_buffer.add(f"MLBrain: auto-retrained model {self.model_version}")
            if self.swarm:
                self.swarm.announce_model(self.model_name, self.model_version)
        except Exception as e:
            self.log_buffer.add(f"MLBrain: auto-retrain failed: {e}")

# -----------------------------
# Control Brain + Self-Optimization
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

class SelfOptimizationEngine:
    def __init__(self, shared_state: Dict[str, Any], log_buffer: LogBuffer,
                 config_writer: Optional[ConfigWriter] = None,
                 swarm: Optional[SwarmNode] = None):
        self.shared_state = shared_state
        self.log_buffer = log_buffer
        self.config_writer = config_writer
        self.swarm = swarm
        self.eh_safe_threshold = shared_state.get("selfopt_eh_safe", 0.80)
        self.eh_hyper_threshold = shared_state.get("selfopt_eh_hyper", 0.60)
        self.oc_normal_threshold = shared_state.get("selfopt_oc_normal", 0.80)
        self.hysteresis = 0.05
        self.aggressiveness = shared_state.get("selfopt_aggressiveness", 0.50)
        self.window = 120
        self.eh_hist: List[float] = []
        self.oc_hist: List[float] = []
        self.health_hist: List[float] = []
        self.shadow_policy: Optional[Dict[str, float]] = None
        self.shadow_start_time: Optional[float] = None
        self.shadow_eval_window = 300
        self.shadow_baseline_health = None
        self.last_commit_version: Optional[int] = None
        self.last_commit_health: Optional[float] = None

    def observe(self, eh: float, oc: float, health: float):
        self.eh_hist.append(eh)
        self.oc_hist.append(oc)
        self.health_hist.append(health)
        if len(self.eh_hist) > self.window:
            self.eh_hist = self.eh_hist[-self.window:]
            self.oc_hist = self.oc_hist[-self.window:]
            self.health_hist = self.health_hist[-self.window:]
        if len(self.eh_hist) < self.window // 3:
            return
        self._maybe_finalize_shadow()
        self._maybe_spawn_shadow()

    def _maybe_spawn_shadow(self):
        if self.shadow_policy is not None:
            return
        avg_eh = float(np.mean(self.eh_hist))
        avg_health = float(np.mean(self.health_hist))
        if avg_health > 0.85 and avg_eh < 0.4:
            shadow = {
                "eh_safe_threshold": max(0.6, self.eh_safe_threshold - 0.02),
                "eh_hyper_threshold": max(0.4, self.eh_hyper_threshold - 0.02),
                "oc_normal_threshold": min(0.9, self.oc_normal_threshold + 0.02),
                "aggressiveness": max(0.0, self.aggressiveness - 0.05),
            }
        elif avg_health < 0.6 or avg_eh > 0.6:
            shadow = {
                "eh_safe_threshold": min(0.95, self.eh_safe_threshold + 0.02),
                "eh_hyper_threshold": min(0.85, self.eh_hyper_threshold + 0.02),
                "oc_normal_threshold": max(0.7, self.oc_normal_threshold - 0.02),
                "aggressiveness": min(1.0, self.aggressiveness + 0.05),
            }
        else:
            return
        self.shadow_policy = shadow
        self.shadow_start_time = time.time()
        self.shadow_baseline_health = float(np.mean(self.health_hist))
        self.log_buffer.add(f"SelfOpt: spawned SHADOW {shadow}, baseline={self.shadow_baseline_health:.3f}")

    def _maybe_finalize_shadow(self):
        if self.shadow_policy is None or self.shadow_start_time is None:
            return
        if time.time() - self.shadow_start_time < self.shadow_eval_window:
            return
        avg_health = float(np.mean(self.health_hist))
        if avg_health >= (self.shadow_baseline_health or avg_health):
            self._commit_policy(self.shadow_policy, avg_health)
        else:
            self.log_buffer.add(
                f"SelfOpt: shadow rejected (health {avg_health:.3f} < baseline {self.shadow_baseline_health:.3f})"
            )
        self.shadow_policy = None
        self.shadow_start_time = None
        self.shadow_baseline_health = None

    def _commit_policy(self, policy: Dict[str, float], avg_health: float):
        self.eh_safe_threshold = policy["eh_safe_threshold"]
        self.eh_hyper_threshold = policy["eh_hyper_threshold"]
        self.oc_normal_threshold = policy["oc_normal_threshold"]
        self.aggressiveness = policy["aggressiveness"]
        self.shared_state["selfopt_eh_safe"] = self.eh_safe_threshold
        self.shared_state["selfopt_eh_hyper"] = self.eh_hyper_threshold
        self.shared_state["selfopt_oc_normal"] = self.oc_normal_threshold
        self.shared_state["selfopt_aggressiveness"] = self.aggressiveness
        self.log_buffer.add(f"SelfOpt: COMMIT {policy}, avg_health={avg_health:.3f}")
        if self.config_writer:
            version = self.config_writer.write({
                "selfopt": {
                    "eh_safe_threshold": self.eh_safe_threshold,
                    "eh_hyper_threshold": self.eh_hyper_threshold,
                    "oc_normal_threshold": self.oc_normal_threshold,
                    "aggressiveness": self.aggressiveness,
                }
            })
            if version > 0:
                self.last_commit_version = version
                self.last_commit_health = avg_health
                self.config_writer.mark_good({
                    "selfopt": {
                        "eh_safe_threshold": self.eh_safe_threshold,
                        "eh_hyper_threshold": self.eh_hyper_threshold,
                        "oc_normal_threshold": self.oc_normal_threshold,
                        "aggressiveness": self.aggressiveness,
                    }
                }, version)
        if self.swarm:
            self.swarm.announce_policy(
                self.last_commit_version or 0,
                {
                    "eh_safe_threshold": self.eh_safe_threshold,
                    "eh_hyper_threshold": self.eh_hyper_threshold,
                    "oc_normal_threshold": self.oc_normal_threshold,
                    "aggressiveness": self.aggressiveness,
                }
            )

    def maybe_rollback(self, current_health: float):
        if self.last_commit_health is None or not self.config_writer:
            return
        if current_health >= self.last_commit_health - 0.15:
            return
        self.log_buffer.add(
            f"SelfOpt: health drop {self.last_commit_health:.3f} -> {current_health:.3f}, rollback"
        )
        self.config_writer.rollback()

    def choose_mode(self, current_mode: str, eh: float, oc: float) -> str:
        safe_th = self.shadow_policy["eh_safe_threshold"] if self.shadow_policy else self.eh_safe_threshold
        hyper_th = self.shadow_policy["eh_hyper_threshold"] if self.shadow_policy else self.eh_hyper_threshold
        normal_th = self.shadow_policy["oc_normal_threshold"] if self.shadow_policy else self.oc_normal_threshold
        safe_back = safe_th - self.hysteresis
        hyper_back = hyper_th - self.hysteresis
        normal_back = normal_th + self.hysteresis
        eh_adj = eh + 0.1 * self.aggressiveness
        oc_adj = oc - 0.1 * self.aggressiveness
        if eh_adj >= safe_th:
            return "safe"
        if eh_adj >= hyper_th:
            return "hyper"
        if oc_adj >= normal_th:
            return "normal"
        if current_mode == "safe" and eh_adj <= safe_back:
            return "hyper"
        if current_mode == "hyper" and eh_adj <= hyper_back and oc_adj >= normal_back:
            return "normal"
        return current_mode

class ControlBrain:
    def __init__(self, bus: EventBusBase, storage: Storage, log_buffer: LogBuffer,
                 shared_state: Dict[str, Any], gpu_controller: GPUController,
                 config_writer: Optional[ConfigWriter], swarm: Optional[SwarmNode]):
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
        self.selfopt = SelfOptimizationEngine(shared_state, log_buffer, config_writer, swarm)
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
        cpu = m["cpu_load"]
        mem = m["memory_load"]
        self.health_score = max(0.0, min(1.0,
            0.30*(1.0 - eh) +
            0.30*oc +
            0.15*(1.0 - cpu) +
            0.15*(1.0 - mem) +
            0.10*(1.0 - gpu_util)
        ))
        self.selfopt.observe(eh, oc, self.health_score)
        self.selfopt.maybe_rollback(self.health_score)
        new_mode = self.selfopt.choose_mode(self.state.mode, eh, oc)
        if new_mode != self.state.mode:
            self._enter_mode(new_mode)
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
# Plugin Architecture
# -----------------------------
class Plugin:
    def __init__(self, name: str):
        self.name = name
    def start(self): ...
    def stop(self): ...

class PluginManager:
    def __init__(self, log_buffer: LogBuffer, bus: EventBusBase, shared_state: Dict[str, Any]):
        self.log_buffer = log_buffer
        self.bus = bus
        self.shared_state = shared_state
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

    def discover_and_load(self, plugins_dir: str):
        if not os.path.isdir(plugins_dir):
            return
        for fname in os.listdir(plugins_dir):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            mod_name = fname[:-3]
            full_mod = f"plugins.{mod_name}"
            try:
                mod = __import__(full_mod, fromlist=["*"])
                if hasattr(mod, "register"):
                    mod.register(self)
                    self.log_buffer.add(f"PluginManager: loaded plugin module {full_mod}")
            except Exception as e:
                self.log_buffer.add(f"PluginManager: failed to load plugin {full_mod}: {e}")

# -----------------------------
# Borg Sovereign Shell (v14) as Toplevel
# -----------------------------
NODE_ID = "killer666@JunctionCity"
MESH_PATH = r"\\MESH\shared\asi_memory_mesh.json"
MEMORY: List[Dict[str, Any]] = []
CLOAK_MEMORY = {
    "cloaked": {"success": 0, "fail": 0},
    "mimic": {"success": 0, "fail": 0},
    "nullified": {"success": 0, "fail": 0},
    "telemetry": {"success": 0, "fail": 0},
    "observed": {"success": 0, "fail": 0},
}
TELEMETRY = ["telemetry.", "analytics.", "doubleclick.", "ads-api."]

POPUP_JS = """(function(){const N='killer666@JunctionCity',L=[];window.open=()=>0;window.alert=()=>0;window.confirm=()=>0;window.prompt=()=>0;
const S=['iframe[src*="ads"]','.popup','.ad-banner','[id*="ad"]','[class*="ad"]'],log=(t,d)=>{L.push({t:new Date().toISOString(),n:N,y:t,d});localStorage.setItem('borg_popup_log',JSON.stringify(L))};
const purge=()=>S.forEach(s=>document.querySelectorAll(s).forEach(e=>{e.remove();log('purged',s)}));
new MutationObserver(m=>m.forEach(x=>x.addedNodes.forEach(n=>{if(n.nodeType===1&&/ad|popup/i.test(n.outerHTML)){n.remove();log('mutated',n.outerHTML.slice(0,100))}})))
.observe(document.body,{childList:1,subtree:1});setInterval(purge,3e3);console.log('🧠 Borg Popup Interceptor initialized')})();"""

def borg_load_mem():
    global MEMORY, CLOAK_MEMORY
    if os.path.exists("asi_memory.json"):
        try:
            MEMORY = json.load(open("asi_memory.json", "r"))
        except Exception:
            MEMORY = []
    else:
        MEMORY = []
    if os.path.exists("cloak_memory.json"):
        try:
            CLOAK_MEMORY = json.load(open("cloak_memory.json", "r"))
        except Exception:
            pass

def borg_save_mem():
    try:
        json.dump(MEMORY, open("asi_memory.json", "w"))
        json.dump(CLOAK_MEMORY, open("cloak_memory.json", "w"))
    except Exception:
        pass

def borg_best_cloak():
    return max(CLOAK_MEMORY, key=lambda k: CLOAK_MEMORY[k]["success"] - CLOAK_MEMORY[k]["fail"])

def borg_log_ad(output_widget, domain, port, payload, cloak="cloaked", direction="in", success=True):
    MEMORY.append(dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        domain=domain,
        port=port,
        payload=payload,
        cloak=cloak,
        direction=direction,
        node=NODE_ID,
    ))
    if cloak in CLOAK_MEMORY:
        CLOAK_MEMORY[cloak]["success" if success else "fail"] += 1
    borg_save_mem()
    arrow = "⬇️" if direction == "in" else "⬆️"
    lineage = f"{cloak}→mimic" if not success else cloak
    output_widget.insert(tk.END, f"{arrow} [{lineage}] {domain}:{port} → {payload}\n")
    output_widget.see(tk.END)

def borg_detect_telemetry(output_widget, domain, port, payload):
    for t in TELEMETRY:
        if t in domain:
            borg_log_ad(output_widget, domain, port, payload, "telemetry", "out")

def borg_dissect(output_widget):
    def run():
        output_widget.insert(tk.END, "🔍 Dissecting YouTube manifest...\n")
        output_widget.see(tk.END)
        strategy = borg_best_cloak()
        try:
            url = "https://youtube.com/fake_manifest.mpd"
            manifest = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
            segments = re.findall(r"https?://[^\s\"']+\.(m4s|ts|mp4)", manifest)
            flagged = [s for s in segments if any(k in s.lower() for k in ["ad", "promo", "sponsor", "preroll", "midroll", "postroll"])]
            for seg in flagged:
                mutated = seg.replace("ad", "intro").replace("promo", "chapter1").replace("sponsor", "lecture")
                borg_log_ad(output_widget, "youtube.com", 443, mutated, cloak=strategy, success=True)
            output_widget.insert(tk.END, f"✅ Dissection complete: {len(flagged)} segments mutated\n")
        except Exception as e:
            output_widget.insert(tk.END, f"⚠️ Dissection failed: {e}\n")
        output_widget.see(tk.END)
    threading.Thread(target=run, daemon=True).start()

def borg_scan_ports(output_widget):
    def run():
        try:
            ip_lines = subprocess.check_output("ipconfig", shell=True).decode().splitlines()
            ip = [l.split(":")[-1].strip() for l in ip_lines if "Default Gateway" in l]
        except Exception:
            ip = []
        if not ip:
            output_widget.insert(tk.END, "⚠️ No router IP\n")
            output_widget.see(tk.END)
            return
        router = ip[0]
        output_widget.insert(tk.END, f"🛡️ Scanning router: {router}\n")
        output_widget.see(tk.END)
        for p in range(1, 1025):
            try:
                s = socket.socket()
                s.settimeout(0.05)
                if s.connect_ex((router, p)) == 0:
                    borg_log_ad(output_widget, router, p, "unknown", "observed")
                    borg_detect_telemetry(output_widget, router, p, "unknown")
                s.close()
            except Exception:
                pass
        output_widget.insert(tk.END, "✅ Scan complete\n")
        output_widget.see(tk.END)
    threading.Thread(target=run, daemon=True).start()

def borg_inject_popup_interceptor(output_widget):
    try:
        driver = webdriver.Chrome()
        driver.get("https://youtube.com")
        driver.execute_script(POPUP_JS)
        output_widget.insert(tk.END, "🧠 Popup interceptor injected\n")
    except Exception as e:
        output_widget.insert(tk.END, f"⚠️ Selenium injection failed: {e}\n")
    output_widget.see(tk.END)

def borg_replay(root):
    win = tk.Toplevel(root)
    win.title("Replay")
    out = scrolledtext.ScrolledText(win, width=120, height=30)
    out.pack()
    f = tk.Frame(win)
    f.pack()
    vars_ = [tk.StringVar() for _ in range(4)]
    labels = ["Domain", "Cloak", "Payload", "Direction"]
    for i, l in enumerate(labels):
        tk.Label(f, text=l).grid(row=0, column=i)
        tk.Entry(f, textvariable=vars_[i], width=15).grid(row=1, column=i)
    def apply():
        out.delete(1.0, tk.END)
        seen, deduped = set(), []
        mesh = []
        if os.path.exists(MESH_PATH):
            try:
                mesh = json.load(open(MESH_PATH, "r"))
            except Exception:
                mesh = []
        for e in MEMORY + mesh:
            key = (e["timestamp"], e["domain"], e["payload"])
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        for e in deduped:
            if any(v.get() and v.get().lower() not in e[k].lower()
                   for v, k in zip(vars_, ["domain", "cloak", "payload", "direction"])):
                continue
            arrow = "⬇️" if e["direction"] == "in" else "⬆️"
            out.insert(tk.END, f"[{e['timestamp']}] {arrow} [{e['cloak']}] {e['domain']}:{e['port']} → {e['payload']} ({e['node']})\n")
        out.see(tk.END)
    tk.Button(f, text="Filter", command=apply).grid(row=1, column=4)

def borg_timeline(root):
    win = tk.Toplevel(root)
    win.title("Timeline")
    canvas = tk.Canvas(win, width=1400, height=600, bg="white")
    canvas.pack()
    grouped = defaultdict(lambda: defaultdict(list))
    mesh = []
    if os.path.exists(MESH_PATH):
        try:
            mesh = json.load(open(MESH_PATH, "r"))
        except Exception:
            mesh = []
    for e in MEMORY + mesh:
        d, h = e["timestamp"].split()[0], e["timestamp"].split()[1][:2]
        grouped[d][h].append(e)
    y = 20
    for date in sorted(grouped):
        canvas.create_text(20, y, anchor="nw", text=f"📅 {date}", font=("Arial", 12, "bold"))
        y += 20
        for hour in sorted(grouped[date]):
            x = int(hour) * 50 + 100
            canvas.create_text(x, y, anchor="nw", text=f"{hour}:00", font=("Arial", 10))
            for i, e in enumerate(grouped[date][hour]):
                color = {
                    "cloaked": "red",
                    "telemetry": "yellow",
                    "observed": "blue",
                    "mimic": "orange",
                    "nullified": "gray",
                }.get(e["cloak"], "black")
                canvas.create_rectangle(x, y + 20 + i * 20, x + 40, y + 35 + i * 20, fill=color)
                canvas.create_text(x + 45, y + 20 + i * 20, anchor="nw",
                                   text=f"{e['domain']} ({e['node']})", font=("Arial", 8))
        y += 100

def open_borg_shell(root):
    borg_load_mem()
    win = tk.Toplevel(root)
    win.title("Borg v14.0 — Sovereign Mutation Shell")
    frame = tk.Frame(win)
    frame.pack()
    output = scrolledtext.ScrolledText(win, width=120, height=30)
    output.pack()

    def log(msg):
        output.insert(tk.END, msg + "\n")
        output.see(tk.END)

    buttons = [
        ("Dissect YouTube", lambda: borg_dissect(output)),
        ("Scan Router", lambda: borg_scan_ports(output)),
        ("Replay", lambda: borg_replay(win)),
        ("Timeline", lambda: borg_timeline(win)),
        ("Inject Popup Cloak", lambda: borg_inject_popup_interceptor(output)),
    ]
    for i, (label, cmd) in enumerate(buttons):
        tk.Button(frame, text=label, command=cmd).grid(row=0, column=i, padx=5)

    log("🧠 Sovereign shell initialized")
    log(f"🧿 Node online: {NODE_ID}")

# -----------------------------
# Ultra-Futuristic Tkinter Cockpit
# -----------------------------
class NerveCenterTk:
    def __init__(self, root: tk.Tk, brain: ControlBrain, shared_state: Dict[str, Any],
                 swarm: SwarmNode, gpu_controller: GPUController, log_buffer: LogBuffer,
                 storage: Storage):
        self.root = root
        self.brain = brain
        self.shared_state = shared_state
        self.swarm = swarm
        self.gpu_controller = gpu_controller
        self.log_buffer = log_buffer
        self.storage = storage

        self.root.title("EVENT HORIZON // BORG CONTROL TOWER // SWARM NODE")
        self.root.configure(bg="#050510")
        self.root.geometry("1400x800")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#050510")
        style.configure("Dark.TLabel", background="#050510", foreground="#e0e0ff", font=("Consolas", 10))
        style.configure("Title.TLabel", background="#050510", foreground="#80ff80", font=("Consolas", 14, "bold"))
        style.configure("Gauge.TLabel", background="#050510", foreground="#ffdf80", font=("Consolas", 11, "bold"))
        style.configure("Mode.TLabel", background="#050510", foreground="#ff80c0", font=("Consolas", 12, "bold"))
        style.configure("Treeview", background="#101020", foreground="#e0e0ff", fieldbackground="#101020")
        style.map("Treeview", background=[("selected", "#303060")])

        self._build_layout()
        self._tick()

    def _build_layout(self):
        self.top_frame = ttk.Frame(self.root, style="Dark.TFrame")
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.left_frame = ttk.Frame(self.root, style="Dark.TFrame")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = ttk.Frame(self.root, style="Dark.TFrame")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        title = ttk.Label(self.top_frame, text="EVENT HORIZON // ULTRA-FUTURISTIC SWARM COCKPIT", style="Title.TLabel")
        title.pack(side=tk.LEFT, padx=10, pady=5)

        self.mode_label = ttk.Label(self.top_frame, text="MODE: ???", style="Mode.TLabel")
        self.mode_label.pack(side=tk.RIGHT, padx=10)

        # LEFT: Gauges + Swarm + GPU
        self.gauge_frame = ttk.LabelFrame(self.left_frame, text="SYSTEM GAUGES", style="Dark.TFrame")
        self.gauge_frame.pack(fill=tk.X, padx=8, pady=8)

        self.canvas_eh = tk.Canvas(self.gauge_frame, width=180, height=120, bg="#050510", highlightthickness=0)
        self.canvas_oc = tk.Canvas(self.gauge_frame, width=180, height=120, bg="#050510", highlightthickness=0)
        self.canvas_health = tk.Canvas(self.gauge_frame, width=180, height=120, bg="#050510", highlightthickness=0)
        self.canvas_eh.grid(row=0, column=0, padx=5, pady=5)
        self.canvas_oc.grid(row=0, column=1, padx=5, pady=5)
        self.canvas_health.grid(row=0, column=2, padx=5, pady=5)

        self.label_eh = ttk.Label(self.gauge_frame, text="EVENT HORIZON", style="Gauge.TLabel")
        self.label_oc = ttk.Label(self.gauge_frame, text="OPTIMIZATION CONF", style="Gauge.TLabel")
        self.label_health = ttk.Label(self.gauge_frame, text="SYSTEM HEALTH", style="Gauge.TLabel")
        self.label_eh.grid(row=1, column=0)
        self.label_oc.grid(row=1, column=1)
        self.label_health.grid(row=1, column=2)

        self.swarm_frame = ttk.LabelFrame(self.left_frame, text="SWARM NODES", style="Dark.TFrame")
        self.swarm_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.swarm_tree = ttk.Treeview(self.swarm_frame, columns=("status", "lag", "last"), show="headings", height=6)
        self.swarm_tree.heading("status", text="Status")
        self.swarm_tree.heading("lag", text="Lag")
        self.swarm_tree.heading("last", text="Last Heartbeat")
        self.swarm_tree.column("status", width=80)
        self.swarm_tree.column("lag", width=80)
        self.swarm_tree.column("last", width=160)
        self.swarm_tree.pack(fill=tk.BOTH, expand=True)

        self.gpu_frame = ttk.LabelFrame(self.left_frame, text="GPU TELEMETRY", style="Dark.TFrame")
        self.gpu_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.gpu_tree = ttk.Treeview(self.gpu_frame, columns=("util", "mem", "temp"), show="headings", height=4)
        self.gpu_tree.heading("util", text="Util %")
        self.gpu_tree.heading("mem", text="Mem Used/Total")
        self.gpu_tree.heading("temp", text="Temp C")
        self.gpu_tree.column("util", width=80)
        self.gpu_tree.column("mem", width=160)
        self.gpu_tree.column("temp", width=80)
        self.gpu_tree.pack(fill=tk.BOTH, expand=True)

        # RIGHT: Event Horizon graph + logs + controls
        self.graph_frame = ttk.LabelFrame(self.right_frame, text="EVENT HORIZON VECTOR FIELD", style="Dark.TFrame")
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.graph_canvas = tk.Canvas(self.graph_frame, bg="#050510", highlightthickness=0)
        self.graph_canvas.pack(fill=tk.BOTH, expand=True)

        self.log_frame = ttk.LabelFrame(self.right_frame, text="SWARM LOG STREAM", style="Dark.TFrame")
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.log_text = tk.Text(self.log_frame, bg="#050510", fg="#a0ffb0", insertbackground="#a0ffb0",
                                font=("Consolas", 9), height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.control_frame = ttk.LabelFrame(self.right_frame, text="MODE & POLICY CONTROLS", style="Dark.TFrame")
        self.control_frame.pack(fill=tk.X, padx=8, pady=4)

        self.btn_safe = ttk.Button(self.control_frame, text="FORCE SAFE", command=lambda: self._force_mode("safe"))
        self.btn_normal = ttk.Button(self.control_frame, text="FORCE NORMAL", command=lambda: self._force_mode("normal"))
        self.btn_hyper = ttk.Button(self.control_frame, text="FORCE HYPER", command=lambda: self._force_mode("hyper"))
        self.btn_borg = ttk.Button(self.control_frame, text="OPEN BORG SHELL", command=lambda: open_borg_shell(self.root))
        self.btn_safe.grid(row=0, column=0, padx=4, pady=4)
        self.btn_normal.grid(row=0, column=1, padx=4, pady=4)
        self.btn_hyper.grid(row=0, column=2, padx=4, pady=4)
        self.btn_borg.grid(row=0, column=3, padx=4, pady=4)

    def _force_mode(self, mode: str):
        self.brain.state.mode = mode
        self.log_buffer.add(f"GUI: manual override -> {mode.upper()}")

    def _draw_gauge(self, canvas: tk.Canvas, value: float, color: str):
        canvas.delete("all")
        w = int(canvas["width"])
        h = int(canvas["height"])
        cx, cy = w // 2, h * 0.8
        r = min(w, h) * 0.6
        canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=180, extent=180,
                          outline="#303050", style=tk.ARC, width=3)
        angle = 180 + 180 * value
        rad = math.radians(angle)
        x = cx + r * math.cos(rad)
        y = cy + r * math.sin(rad)
        canvas.create_line(cx, cy, x, y, fill=color, width=4)
        canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=color, outline=color)

    def _update_swarm_table(self):
        for i in self.swarm_tree.get_children():
            self.swarm_tree.delete(i)
        nodes = self.storage.get_nodes()
        for n in nodes:
            ts = time.strftime("%H:%M:%S", time.localtime(n["last_heartbeat"]))
            self.swarm_tree.insert("", tk.END, values=(n["status"], f"{n['lag']:.3f}", ts))

    def _update_gpu_table(self):
        for i in self.gpu_tree.get_children():
            self.gpu_tree.delete(i)
        metrics = self.gpu_controller.metrics()
        for g in metrics:
            mem_str = f"{g['memory_used']//(1024**2)} / {g['memory_total']//(1024**2)} MB"
            self.gpu_tree.insert("", tk.END, values=(g["utilization"], mem_str, g["temperature"]))

    def _update_log_view(self):
        text = self.log_buffer.text()
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _update_graph_canvas(self):
        self.graph_canvas.delete("all")
        w = self.graph_canvas.winfo_width()
        h = self.graph_canvas.winfo_height()
        if w < 10 or h < 10:
            return
        s = self.brain.state
        eh = s.event_horizon
        oc = s.optimization_confidence
        health = self.brain.health_score
        cx, cy = w // 2, h // 2
        radius = min(w, h) * 0.35
        self.graph_canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                      outline="#202040", width=2)
        angle_eh = math.pi * eh
        x_eh = cx + radius * math.cos(angle_eh)
        y_eh = cy - radius * math.sin(angle_eh)
        self.graph_canvas.create_line(cx, cy, x_eh, y_eh, fill="#ff8080", width=3)
        self.graph_canvas.create_text(x_eh, y_eh - 10, text=f"EH {eh:.2f}", fill="#ff8080", font=("Consolas", 9))
        angle_oc = math.pi * oc + math.pi / 4
        x_oc = cx + radius * 0.8 * math.cos(angle_oc)
        y_oc = cy - radius * 0.8 * math.sin(angle_oc)
        self.graph_canvas.create_line(cx, cy, x_oc, y_oc, fill="#80ff80", width=3)
        self.graph_canvas.create_text(x_oc, y_oc - 10, text=f"OC {oc:.2f}", fill="#80ff80", font=("Consolas", 9))
        aura_r = radius * health
        self.graph_canvas.create_oval(cx - aura_r, cy - aura_r, cx + aura_r, cy + aura_r,
                                      outline="#80c0ff", width=2)
        self.graph_canvas.create_text(cx, cy + radius + 15,
                                      text=f"HEALTH {health:.2f}",
                                      fill="#80c0ff", font=("Consolas", 10))

    def _tick(self):
        self.brain.evaluate()
        s = self.brain.state
        self.mode_label.config(text=f"MODE: {s.mode.upper()}")
        self._draw_gauge(self.canvas_eh, s.event_horizon, "#ff8080")
        self._draw_gauge(self.canvas_oc, s.optimization_confidence, "#80ff80")
        self._draw_gauge(self.canvas_health, self.brain.health_score, "#80c0ff")
        self._update_swarm_table()
        self._update_gpu_table()
        self._update_log_view()
        self._update_graph_canvas()
        self.root.after(1000, self._tick)

# -----------------------------
# FastAPI Web Dashboard + REST
# -----------------------------
app = FastAPI()
websocket_clients: List[WebSocket] = []
GLOBAL_BRAIN: Optional[ControlBrain] = None
GLOBAL_STORAGE: Optional[Storage] = None
GLOBAL_GPU: Optional[GPUController] = None
GLOBAL_SWARM: Optional[SwarmNode] = None
GLOBAL_SHARED_STATE: Dict[str, Any] = {}

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

@app.post("/api/mode")
async def api_set_mode(mode: str = Body(..., embed=True)):
    if not GLOBAL_BRAIN:
        return JSONResponse({"error": "no brain"}, status_code=500)
    GLOBAL_BRAIN.state.mode = mode
    return {"status": "ok", "mode": mode}

@app.post("/api/gpu/mode")
async def api_set_gpu_mode(mode: str = Body(..., embed=True)):
    if not GLOBAL_GPU:
        return JSONResponse({"error": "no gpu"}, status_code=500)
    if mode == "safe":
        GLOBAL_GPU.handle_command({"action": "safe_mode"})
    elif mode == "performance":
        GLOBAL_GPU.handle_command({"action": "performance_mode"})
    else:
        GLOBAL_GPU.mode = mode
    return {"status": "ok", "mode": GLOBAL_GPU.mode}

@app.post("/api/selfopt/tune")
async def api_selfopt_tune(
    eh_safe: float = Body(None),
    eh_hyper: float = Body(None),
    oc_normal: float = Body(None),
    aggressiveness: float = Body(None),
):
    if not GLOBAL_SHARED_STATE:
        return JSONResponse({"error": "no shared state"}, status_code=500)
    if eh_safe is not None:
        GLOBAL_SHARED_STATE["selfopt_eh_safe"] = eh_safe
    if eh_hyper is not None:
        GLOBAL_SHARED_STATE["selfopt_eh_hyper"] = eh_hyper
    if oc_normal is not None:
        GLOBAL_SHARED_STATE["selfopt_oc_normal"] = oc_normal
    if aggressiveness is not None:
        GLOBAL_SHARED_STATE["selfopt_aggressiveness"] = aggressiveness
    return {"status": "ok", "selfopt": {
        "eh_safe": GLOBAL_SHARED_STATE.get("selfopt_eh_safe"),
        "eh_hyper": GLOBAL_SHARED_STATE.get("selfopt_eh_hyper"),
        "oc_normal": GLOBAL_SHARED_STATE.get("selfopt_oc_normal"),
        "aggressiveness": GLOBAL_SHARED_STATE.get("selfopt_aggressiveness"),
    }}

@app.post("/api/swarm/policy")
async def api_swarm_policy(policy: Dict[str, Any]):
    if not GLOBAL_SWARM:
        return JSONResponse({"error": "no swarm"}, status_code=500)
    version = int(time.time())
    GLOBAL_SWARM.announce_policy(version, policy)
    return {"status": "ok", "version": version}

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
    global GLOBAL_BRAIN, GLOBAL_STORAGE, GLOBAL_GPU, GLOBAL_SWARM, GLOBAL_SHARED_STATE

    shared_state: Dict[str, Any] = {}
    GLOBAL_SHARED_STATE = shared_state

    cfg_mgr = ConfigManager(db_path=os.path.join(DATA_DIR, "configs.db"), config_dir=CONFIG_DIR)
    cfg_mgr.load_from_files()
    core_cfg = cfg_mgr.get("core") or DEFAULT_CORE_YAML

    selfopt_cfg = core_cfg.get("selfopt", {})
    shared_state["selfopt_eh_safe"] = selfopt_cfg.get("eh_safe_threshold", 0.80)
    shared_state["selfopt_eh_hyper"] = selfopt_cfg.get("eh_hyper_threshold", 0.60)
    shared_state["selfopt_oc_normal"] = selfopt_cfg.get("oc_normal_threshold", 0.80)
    shared_state["selfopt_aggressiveness"] = selfopt_cfg.get("aggressiveness", 0.50)

    try:
        storage = Storage(core_cfg["storage"]["path"])
    except sqlite3.OperationalError as e:
        print(f"WARNING: Failed to open DB at {core_cfg['storage']['path']}: {e}")
        print("Falling back to in-memory SQLite.")
        storage = Storage(":memory:")

    log_buffer = LogBuffer(storage)

    if core_cfg["kafka"].get("enabled", False):
        bus: EventBusBase = KafkaEventBus(
            brokers=core_cfg["kafka"]["brokers"],
            group_id=core_cfg["kafka"]["group_id"],
            log_buffer=log_buffer,
        )
        log_buffer.add("Main: using KafkaEventBus")
    else:
        bus = InMemoryEventBus()
        log_buffer.add("Main: using InMemoryEventBus")

    telemetry = TelemetryEngine(shared_state, bus)
    gpu_controller = GPUController(log_buffer, bus, shared_state)
    GLOBAL_GPU = gpu_controller

    config_writer = ConfigWriter(CORE_YAML_PATH, log_buffer)
    swarm = SwarmNode(core_cfg["swarm"]["node_id"], shared_state, storage, log_buffer, bus)
    GLOBAL_SWARM = swarm

    registry = ModelRegistry(log_buffer=log_buffer)
    log_buffer.add(f"ModelRegistry active directory: {registry.models_dir}")
    ml_brain = MLBrain(registry, core_cfg["ml"]["model_name"], core_cfg["ml"]["model_version"],
                       log_buffer, bus, shared_state, swarm)

    brain = ControlBrain(bus, storage, log_buffer, shared_state, gpu_controller, config_writer, swarm)
    GLOBAL_BRAIN = brain
    GLOBAL_STORAGE = storage

    plugin_manager = PluginManager(log_buffer, bus, shared_state)
    plugin_manager.discover_and_load(os.path.join(BASE_DIR, "plugins"))
    plugin_manager.start_all()

    def telemetry_loop():
        while True:
            telemetry.snapshot()
            gpu_controller.apply_steering_policy()
            time.sleep(1.0)

    threading.Thread(target=telemetry_loop, daemon=True).start()

    def api_runner():
        uvicorn.run(app, host=core_cfg["web"]["host"], port=core_cfg["web"]["port"], log_level="warning")

    threading.Thread(target=api_runner, daemon=True).start()
    threading.Thread(target=backend_broadcast_loop, args=(brain, storage, gpu_controller), daemon=True).start()

    root = tk.Tk()
    ui = NerveCenterTk(root, brain, shared_state, swarm, gpu_controller, log_buffer, storage)
    root.mainloop()

if __name__ == "__main__":
    main()

