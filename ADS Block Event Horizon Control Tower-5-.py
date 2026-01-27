"""
Event Horizon Control Tower - Full Stack Organism (Single File)
- Config (inline, can be externalized)
- FastAPI backend (REST + WebSocket)
- GPU controller (pynvml if available)
- Swarm gossip protocol (in-process)
- ML pipeline (joblib model + fallback)
- SQLite persistence (with safe path + fallback)
- Tkinter cockpit
- Web dashboard frontend (served by FastAPI)
"""

import sys
import time
import subprocess
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
import math
import threading
import json
import os
import sqlite3
import queue
import atexit

# -------- Autoloader --------
REQUIRED_LIBRARIES = [
    "numpy",
    "psutil",
    "fastapi",
    "uvicorn",
    "joblib",
    "pynvml",
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
import tkinter as tk
from tkinter import ttk, messagebox

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import joblib

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


# =========================
# Paths, DB, Lock
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_FILE = os.path.join(DATA_DIR, "tower.db")
LOG_FILE = os.path.join(DATA_DIR, "control_tower.log")
LOCK_FILE = os.path.join(DATA_DIR, "tower.lock")

# simple single-instance lock
if os.path.exists(LOCK_FILE):
    print("Another instance of Event Horizon Control Tower appears to be running. Exiting.")
    sys.exit(1)
with open(LOCK_FILE, "w") as f:
    f.write(str(os.getpid()))

def _cleanup_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception:
        pass

atexit.register(_cleanup_lock)


# =========================
# Config (inline)
# =========================

CONFIG: Dict[str, Any] = {
    "kafka": {
        "enabled": False,
        "brokers": ["localhost:9092"],
    },
    "gpu": {
        "mode": "balanced",
    },
    "swarm": {
        "node_id": "node-1",
        "heartbeat_interval": 5,
        "gossip_interval": 2,
    },
    "storage": {
        "type": "sqlite",
        "path": DB_FILE,
    },
    "web": {
        "host": "127.0.0.1",
        "port": 8000,
    },
}


# =========================
# Persistence (SQLite)
# =========================

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


# =========================
# Logging
# =========================

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


# =========================
# Prediction Graph
# =========================

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


# =========================
# Telemetry Engine
# =========================

class TelemetryEngine:
    def __init__(self, shared_state: Dict[str, float]):
        self.shared = shared_state

    def snapshot(self) -> Dict[str, float]:
        cpu = psutil.cpu_percent() / 100.0
        mem = psutil.virtual_memory().percent / 100.0

        self.shared["cpu_load"] = cpu
        self.shared["memory_load"] = mem

        self.shared["drift"] = min(1.0, max(0.0, self.shared.get("drift", 0.1) * 0.98 + 0.01))
        self.shared["uncertainty"] = min(1.0, max(0.0, self.shared.get("uncertainty", 0.2) * 0.99 + 0.005))
        self.shared["block_rate_volatility"] = min(1.0, max(0.0, self.shared.get("block_rate_volatility", 0.1) * 0.97 + 0.01))
        self.shared["disagreement"] = min(1.0, max(0.0, self.shared.get("disagreement", 0.05) * 0.98 + 0.005))

        return {
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


# =========================
# Command Bus
# =========================

class CommandBus:
    def __init__(self):
        self.local_handlers: Dict[str, Any] = {}

    def register_local(self, target: str, handler):
        self.local_handlers[target] = handler

    def send(self, target: str, payload: Dict[str, Any]):
        if target in self.local_handlers:
            self.local_handlers[target](payload)


# =========================
# Metrics + Control Tower
# =========================

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


def compute_event_horizon(m, graph: PredictionGraph) -> float:
    U = m["uncertainty"]
    D = m["drift"]
    B = m["block_rate_volatility"]
    X = m["disagreement"]
    G = graph.predict_risk("global")
    return max(0.0, min(1.0, 0.25*U + 0.20*D + 0.20*B + 0.25*G + 0.10*X))


def compute_optimization_confidence(m, graph: PredictionGraph) -> float:
    A = m["accuracy"]
    L = m["latency"]
    E = m["error_rate"]
    D = m["drift"]
    P = m["learning_progression"]
    G = graph.predict_risk("global")

    P = (P + 1.0) / 2.0
    return max(0.0, min(1.0,
        0.30*A +
        0.20*(1-L) +
        0.20*(1-E) +
        0.10*(1-D) +
        0.10*(1-G) +
        0.10*P
    ))


class ControlTower:
    def __init__(self, telemetry: TelemetryEngine, config: Dict[str, Any], bus: CommandBus, log_buffer: LogBuffer, storage: Storage):
        self.telemetry = telemetry
        self.config = config
        self.bus = bus
        self.graph = PredictionGraph()
        self.state = SystemState()
        self.health_score = 1.0
        self.log_buffer = log_buffer
        self.storage = storage

    def observe_event(self, source: str, url: str, is_ad: int):
        self.graph.observe(source, url, is_ad)
        try:
            self.storage.insert_event(time.time(), source, url, is_ad)
        except Exception:
            pass

    def enter_safe_mode(self):
        self.state.mode = "safe"
        self.bus.send("filter_engine", {"mode": "rules_first"})
        self.log_buffer.add("ControlTower: entering SAFE mode")

    def enter_hypervigilant_mode(self):
        self.state.mode = "hypervigilant"
        self.bus.send("filter_engine", {"mode": "hybrid_strict"})
        self.log_buffer.add("ControlTower: entering HYPERVIGILANT mode")

    def enter_normal_mode(self):
        self.state.mode = "normal"
        self.bus.send("filter_engine", {"mode": "hybrid"})
        self.log_buffer.add("ControlTower: entering NORMAL mode")

    def reload_model(self):
        self.state.model_version = f"v{int(time.time())}"
        self.log_buffer.add(f"ControlTower: model reloaded -> {self.state.model_version}")
        try:
            self.storage.conn.execute(
                "INSERT INTO model_versions (version, activated_at) VALUES (?, ?)",
                (self.state.model_version, time.time()),
            )
            self.storage.conn.commit()
        except Exception:
            pass

    def quarantine_source(self, source: str):
        self.log_buffer.add(f"ControlTower: quarantine requested for source={source}")

    def evaluate(self):
        m = self.telemetry.snapshot()

        self.state.block_rate = m["block_rate"]
        self.state.error_rate = m["error_rate"]
        self.state.learning_progression = m["learning_progression"]
        self.state.model_version = m["model_version"]

        self.state.predicted_risk = self.graph.predict_risk("global")
        self.state.event_horizon = compute_event_horizon(m, self.graph)
        self.state.optimization_confidence = compute_optimization_confidence(m, self.graph)

        eh = self.state.event_horizon
        oc = self.state.optimization_confidence

        if eh > 0.8:
            self.enter_safe_mode()
        elif eh > 0.6:
            self.enter_hypervigilant_mode()
        elif oc > 0.8 and eh < 0.4:
            self.enter_normal_mode()

        cpu = m["cpu_load"]
        mem = m["memory_load"]
        self.health_score = max(0.0, min(1.0,
            0.35*(1.0 - eh) +
            0.35*oc +
            0.15*(1.0 - cpu) +
            0.15*(1.0 - mem)
        ))

        try:
            self.storage.insert_metrics(time.time(), self.state.event_horizon, self.state.optimization_confidence, self.health_score)
        except Exception:
            pass


# =========================
# ML Pipeline
# =========================

class AdFilterModel:
    def __init__(self, log_buffer: LogBuffer):
        self.pipeline = None
        self.log_buffer = log_buffer
        self._load_model()

    def _load_model(self):
        if os.path.exists("ad_filter_model.pkl"):
            try:
                self.pipeline = joblib.load("ad_filter_model.pkl")
                self.log_buffer.add("ML: Loaded ad_filter_model.pkl")
            except Exception as e:
                self.log_buffer.add(f"ML: Failed to load model: {e}")
                self.pipeline = None
        else:
            self.log_buffer.add("ML: ad_filter_model.pkl not found, using dummy model")
            self.pipeline = None

    def predict_is_ad(self, text: str) -> int:
        if self.pipeline is None:
            return 1 if "ad" in text.lower() else 0
        try:
            pred = self.pipeline.predict([text])[0]
            return int(pred)
        except Exception as e:
            self.log_buffer.add(f"ML: prediction error: {e}")
            return 0


# =========================
# GPU Controller
# =========================

class GPUController:
    def __init__(self, log_buffer: LogBuffer):
        self.log_buffer = log_buffer
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
        return out


# =========================
# Swarm Gossip Protocol (in-process)
# =========================

class SwarmNode:
    def __init__(self, node_id: str, shared_state: Dict[str, Any], storage: Storage, log_buffer: LogBuffer):
        self.node_id = node_id
        self.shared_state = shared_state
        self.storage = storage
        self.log_buffer = log_buffer
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
            time.sleep(CONFIG["swarm"]["heartbeat_interval"])

    def _gossip_loop(self):
        while self.running:
            try:
                nodes = self.storage.get_nodes()
                with self.lock:
                    self.peers = {n["node_id"]: n for n in nodes}
            except Exception:
                pass
            time.sleep(CONFIG["swarm"]["gossip_interval"])

    def get_peers(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return dict(self.peers)


# =========================
# Tkinter GUI
# =========================

class NerveCenterTk:
    def __init__(self, root: tk.Tk, tower: ControlTower, shared_state: Dict[str, float], bus: CommandBus, swarm: SwarmNode):
        self.root = root
        self.tower = tower
        self.shared_state = shared_state
        self.bus = bus
        self.log_buffer = tower.log_buffer
        self.swarm = swarm

        self.root.title("Event Horizon Control Tower (Tk) - Borg Node")
        self.root.geometry("1300x850")

        self.time_history: List[float] = []
        self.eh_history: List[float] = []
        self.oc_history: List[float] = []

        self._build_layout()

        self.tower.observe_event("source_A", "url_1", 1)
        self.tower.observe_event("source_A", "url_2", 0)
        self.tower.observe_event("source_B", "url_3", 1)

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

        btn_safe = tk.Button(ctrl_frame, text="SAFE", command=self.force_safe, bg="#402020", fg="white")
        btn_safe.pack(side=tk.TOP, fill=tk.X, pady=2)

        btn_normal = tk.Button(ctrl_frame, text="NORMAL", command=self.force_normal, bg="#204020", fg="white")
        btn_normal.pack(side=tk.TOP, fill=tk.X, pady=2)

        btn_hyper = tk.Button(ctrl_frame, text="HYPER", command=self.force_hyper, bg="#404020", fg="white")
        btn_hyper.pack(side=tk.TOP, fill=tk.X, pady=2)

        btn_reload = tk.Button(ctrl_frame, text="Reload Model", command=self.reload_model, bg="#203040", fg="white")
        btn_reload.pack(side=tk.TOP, fill=tk.X, pady=4)

        gpu_frame = tk.Frame(ctrl_frame, bg="#05050b")
        gpu_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        tk.Label(gpu_frame, text="GPU Control", bg="#05050b", fg="#f0f0ff").pack(anchor="w")
        btn_gpu_safe = tk.Button(gpu_frame, text="GPU SAFE", command=self.gpu_safe, bg="#303060", fg="white")
        btn_gpu_safe.pack(fill=tk.X, pady=1)
        btn_gpu_perf = tk.Button(gpu_frame, text="GPU PERF", command=self.gpu_perf, bg="#306030", fg="white")
        btn_gpu_perf.pack(fill=tk.X, pady=1)

        quarantine_frame = tk.Frame(ctrl_frame, bg="#05050b")
        quarantine_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        tk.Label(quarantine_frame, text="Quarantine Source:", bg="#05050b", fg="#f0f0ff").pack(anchor="w")
        self.quarantine_entry = tk.Entry(quarantine_frame, bg="#101020", fg="#f0f0ff")
        self.quarantine_entry.pack(fill=tk.X)
        btn_quarantine = tk.Button(quarantine_frame, text="Quarantine", command=self.quarantine_source, bg="#603030", fg="white")
        btn_quarantine.pack(fill=tk.X, pady=2)

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
            height=8,
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

    def force_safe(self):
        self.tower.enter_safe_mode()
        self.log_buffer.add("GUI: Forced SAFE mode")

    def force_normal(self):
        self.tower.enter_normal_mode()
        self.log_buffer.add("GUI: Forced NORMAL mode")

    def force_hyper(self):
        self.tower.enter_hypervigilant_mode()
        self.log_buffer.add("GUI: Forced HYPER mode")

    def reload_model(self):
        self.tower.reload_model()

    def quarantine_source(self):
        src = self.quarantine_entry.get().strip()
        if not src:
            messagebox.showinfo("Quarantine", "Enter a source ID.")
            return
        self.tower.quarantine_source(src)
        self.quarantine_entry.delete(0, tk.END)

    def gpu_safe(self):
        self.bus.send("gpu_controller", {"action": "safe_mode"})

    def gpu_perf(self):
        self.bus.send("gpu_controller", {"action": "performance_mode"})

    def _tick(self):
        self.tower.evaluate()
        s = self.tower.state

        self.log_buffer.add(
            f"EH={s.event_horizon:.3f} OC={s.optimization_confidence:.3f} "
            f"mode={s.mode} health={self.tower.health_score:.3f}"
        )

        self._update_metrics()
        self._update_health_bar()
        self._update_chart()
        self._update_graph()
        self._update_swarm()
        self._update_logs()

        self.root.after(1000, self._tick)

    def _update_metrics(self):
        s = self.tower.state
        text = (
            f"Mode: {s.mode}\n"
            f"Model: {s.model_version}\n"
            f"Event Horizon: {s.event_horizon:.3f}\n"
            f"Opt Confidence: {s.optimization_confidence:.3f}\n"
            f"Predicted Risk: {s.predicted_risk:.3f}\n"
            f"Block Rate: {s.block_rate:.3f}\n"
            f"Error Rate: {s.error_rate:.3f}\n"
            f"Learning Progression: {s.learning_progression:.3f}\n"
            f"Kafka Lag: {self.shared_state.get('kafka_lag', 0.0):.3f}\n"
            f"CPU: {self.shared_state.get('cpu_load', 0.0):.3f}\n"
            f"Memory: {self.shared_state.get('memory_load', 0.0):.3f}\n"
            f"Total Events: {int(self.shared_state.get('total_events', 0))}\n"
            f"Blocked Events: {int(self.shared_state.get('blocked_events', 0))}\n"
        )
        self.metrics_label.config(text=text)

    def _update_health_bar(self):
        score = self.tower.health_score
        width = 200
        fill_width = int(width * score)
        r = int(255 * (1.0 - score))
        g = int(255 * score)
        color = f"#{r:02x}{g:02x}20"
        self.health_canvas.coords(self.health_rect, 0, 0, fill_width, 20)
        self.health_canvas.itemconfig(self.health_rect, fill=color)

    def _update_chart(self):
        s = self.tower.state
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
        g = self.tower.graph
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


# =========================
# FastAPI Backend + Web Dashboard
# =========================

app = FastAPI()
websocket_clients: List[WebSocket] = []
backend_state_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()


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
    nodes = GLOBAL_STORAGE.get_nodes()
    return JSONResponse(nodes)


@app.get("/api/metrics")
async def api_metrics():
    s = GLOBAL_TOWER.state
    return JSONResponse({
        "mode": s.mode,
        "event_horizon": s.event_horizon,
        "optimization_confidence": s.optimization_confidence,
        "health": GLOBAL_TOWER.health_score,
    })


@app.post("/api/mode/{mode}")
async def api_mode(mode: str):
    if mode == "safe":
        GLOBAL_TOWER.enter_safe_mode()
    elif mode == "normal":
        GLOBAL_TOWER.enter_normal_mode()
    elif mode == "hyper":
        GLOBAL_TOWER.enter_hypervigilant_mode()
    else:
        return JSONResponse({"error": "invalid mode"}, status_code=400)
    return JSONResponse({"status": "ok"})


def backend_broadcast_loop(tower: ControlTower, swarm: SwarmNode, gpu_controller: GPUController):
    while True:
        try:
            s = tower.state
            metrics_payload = {
                "mode": s.mode,
                "event_horizon": round(s.event_horizon, 3),
                "optimization_confidence": round(s.optimization_confidence, 3),
                "health": round(tower.health_score, 3),
            }
            swarm_payload = GLOBAL_STORAGE.get_nodes()
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


# =========================
# Globals wired in main()
# =========================

GLOBAL_STORAGE: Storage
GLOBAL_LOG_BUFFER: LogBuffer
GLOBAL_TOWER: ControlTower


def main():
    global GLOBAL_STORAGE, GLOBAL_LOG_BUFFER, GLOBAL_TOWER

    shared_state: Dict[str, Any] = {}

    # robust storage init with fallback
    try:
        storage = Storage(CONFIG["storage"]["path"])
    except sqlite3.OperationalError as e:
        print(f"WARNING: Failed to open DB at {CONFIG['storage']['path']}: {e}")
        print("Falling back to in-memory SQLite (no persistence).")
        storage = Storage(":memory:")

    log_buffer = LogBuffer(storage)
    telemetry = TelemetryEngine(shared_state)
    bus = CommandBus()
    config = CONFIG

    tower = ControlTower(telemetry, config, bus, log_buffer, storage)
    GLOBAL_STORAGE = storage
    GLOBAL_LOG_BUFFER = log_buffer
    GLOBAL_TOWER = tower

    gpu_controller = GPUController(log_buffer)
    bus.register_local("gpu_controller", gpu_controller.handle_command)

    def filter_engine_handler(payload: Dict[str, Any]):
        log_buffer.add(f"FilterEngine: mode set to {payload.get('mode')}")
    bus.register_local("filter_engine", filter_engine_handler)

    model = AdFilterModel(log_buffer)

    swarm = SwarmNode(CONFIG["swarm"]["node_id"], shared_state, storage, log_buffer)

    threading.Thread(
        target=backend_broadcast_loop,
        args=(tower, swarm, gpu_controller),
        daemon=True,
    ).start()

    def run_api():
        uvicorn.run(app, host=CONFIG["web"]["host"], port=CONFIG["web"]["port"], log_level="warning")

    threading.Thread(target=run_api, daemon=True).start()

    root = tk.Tk()
    app_tk = NerveCenterTk(root, tower, shared_state, bus, swarm)
    root.mainloop()


if __name__ == "__main__":
    main()

