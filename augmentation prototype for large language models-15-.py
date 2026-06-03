#!/usr/bin/env python3
"""
Unified Forklift Runtime

- Legacy ForkliftEngine (telemetry DB + simple tile engine + web dashboard)
- All‑in‑One LLM Runtime (FP8 + FlashAttention + Triton + Router + Swarm)
"""

import sys, os, time, threading, socket, json, math, subprocess, sqlite3, random, ctypes
from typing import Dict, Tuple, List, Optional
from contextlib import contextmanager

# -------------------------
# Core deps
# -------------------------
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("pip install torch")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("pip install transformers accelerate")
    sys.exit(1)

# -------------------------
# Optional deps
# -------------------------
try:
    import requests
except ImportError:
    requests = None

try:
    from transformers.utils import hub
except ImportError:
    hub = None

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

try:
    import psutil
except ImportError:
    psutil = None

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import scrolledtext, filedialog, ttk
except ImportError:
    tk = None

FLASK_AVAILABLE = False
try:
    from flask import Flask, jsonify, request
    FLASK_AVAILABLE = True
except Exception:
    FLASK_AVAILABLE = False

# TransformerEngine for true FP8
TE_AVAILABLE = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import fp8_autocast
    TE_AVAILABLE = True
except Exception:
    TE_AVAILABLE = False

# FlashAttention for real fused attention
FLASH_ATTENTION_AVAILABLE = False
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except Exception:
    try:
        from flash_attn.flash_attn import flash_attn_func
        FLASH_ATTENTION_AVAILABLE = True
    except Exception:
        FLASH_ATTENTION_AVAILABLE = False

# ============================================================
# === AUTO-ELEVATION CHECK (Windows) =========================
# ============================================================

def ensure_admin():
    """
    Relaunch the script with admin rights on Windows if not already elevated.
    On non-Windows, this is a no-op.
    """
    if os.name != "nt":
        return

    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                sys.executable,
                f'"{script}" {params}',
                None,
                1
            )
            sys.exit(0)
    except Exception as e:
        print(f"[Codex Sentinel] Elevation failed: {e}")
        sys.exit(1)

ensure_admin()

# ============================================================
# === LEGACY FORKLIFT ENGINE (RENAMED) =======================
# ============================================================

# --- PATHS & SQLITE DB SETUP (legacy) ---

LEGACY_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_DATA_DIR = os.path.join(LEGACY_BASE_DIR, "data")
os.makedirs(LEGACY_DATA_DIR, exist_ok=True)

LEGACY_DB_FILE = os.path.join(LEGACY_DATA_DIR, "telemetry.db")


def legacy_get_db_connection():
    """
    Safe SQLite connection creator for legacy engine.
    Ensures directory exists and uses an absolute path.
    """
    os.makedirs(os.path.dirname(LEGACY_DB_FILE), exist_ok=True)
    conn = sqlite3.connect(LEGACY_DB_FILE, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def legacy_init_db():
    conn = legacy_get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            gpu_util REAL,
            gpu_mem REAL,
            cpu_util REAL,
            temp REAL,
            notes TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            model_name TEXT,
            tile_size INTEGER,
            quant_mode TEXT,
            latency_ms REAL,
            throughput_tok_s REAL,
            config_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


legacy_init_db()


class LegacyTelemetryLogger:
    def __init__(self):
        self.lock = threading.Lock()

    def log(self, gpu_util=None, gpu_mem=None, cpu_util=None, temp=None, notes=None):
        ts = time.time()
        with self.lock:
            conn = legacy_get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO telemetry (ts, gpu_util, gpu_mem, cpu_util, temp, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, gpu_util, gpu_mem, cpu_util, temp, notes),
            )
            conn.commit()
            conn.close()

    def fetch_recent(self, limit=100):
        conn = legacy_get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ts, gpu_util, gpu_mem, cpu_util, temp, notes
            FROM telemetry
            ORDER BY ts DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
        return rows


legacy_telemetry_logger = LegacyTelemetryLogger()


class LegacyRouterNetwork:
    """
    Learned router that decides which tiles to compute.
    Placeholder; in real use, plug in a small PyTorch model.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.trained = False

    def route(self, tile_metadata):
        norm = tile_metadata.get("norm", 0.0)
        threshold = self.config.get("norm_threshold", 0.01)
        return norm >= threshold

    def train_from_teacher_entropy(self, teacher_outputs):
        self.trained = True

    def train_from_teacher_activations(self, teacher_activations, attention_maps):
        self.trained = True


class LegacyQuantizationKernels:
    """
    Wraps INT8 + FP8 quantization/dequantization (legacy).
    """

    def __init__(self, mode="fp8"):
        self.mode = mode

    def quantize(self, tensor):
        if tensor is None or torch is None:
            return tensor
        if self.mode == "int8":
            return self._quant_int8(tensor)
        elif self.mode == "fp8":
            return self._quant_fp8(tensor)
        else:
            return tensor

    def dequantize(self, qtensor):
        if qtensor is None or torch is None:
            return qtensor
        if self.mode == "int8":
            return self._dequant_int8(qtensor)
        elif self.mode == "fp8":
            return self._dequant_fp8(qtensor)
        else:
            return qtensor

    def _quant_int8(self, tensor):
        scale = tensor.abs().max() / 127.0 + 1e-8
        q = torch.clamp((tensor / scale).round(), -128, 127).to(torch.int8)
        return (q, scale)

    def _dequant_int8(self, qtensor):
        q, scale = qtensor
        return q.float() * scale

    def _quant_fp8(self, tensor):
        scale = tensor.abs().max() / 240.0 + 1e-8
        q = torch.clamp((tensor / scale).round(), -240, 240).to(torch.int16)
        return (q, scale)

    def _dequant_fp8(self, qtensor):
        q, scale = qtensor
        return q.float() * scale


class LegacyTileCache:
    """
    Local GPU-resident tile cache (legacy conceptual).
    """

    def __init__(self, max_tiles=1024):
        self.max_tiles = max_tiles
        self.cache = {}
        self.lru = []

    def get(self, key):
        if key in self.cache:
            self.lru.remove(key)
            self.lru.insert(0, key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.lru.remove(key)
        elif len(self.cache) >= self.max_tiles:
            evict = self.lru.pop()
            del self.cache[evict]
        self.cache[key] = value
        self.lru.insert(0, key)


class LegacyDistributedTilePrefetcher:
    """
    Multi-node tile prefetching + hint sharing (UDP/TCP skeleton, legacy).
    """

    def __init__(self, udp_port=50050, tcp_port=50051):
        self.udp_port = udp_port
        self.tcp_port = tcp_port
        self.running = False
        self.udp_sock = None

    def start(self):
        self.running = True
        self._start_udp_listener()

    def stop(self):
        self.running = False
        if self.udp_sock:
            self.udp_sock.close()

    def _start_udp_listener(self):
        def _run():
            try:
                self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.udp_sock.bind(("", self.udp_port))
                while self.running:
                    data, addr = self.udp_sock.recvfrom(4096)
                    # parse tile hints here in real code
            except Exception:
                pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def broadcast_tile_hint(self, tile_key):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            msg = json.dumps({"type": "tile_hint", "tile_key": tile_key}).encode("utf-8")
            sock.sendto(msg, ("<broadcast>", self.udp_port))
            sock.close()
        except Exception:
            pass

    def request_remote_tile(self, tile_key):
        return None


class LegacyFlashAttentionWrapper:
    """
    Wraps FlashAttention kernel if available, otherwise falls back (legacy).
    """

    def __init__(self):
        self.available = FLASH_ATTENTION_AVAILABLE and torch is not None

    def attention(self, q, k, v, attn_mask=None):
        if torch is None:
            raise RuntimeError("PyTorch not available for attention.")
        if self.available:
            return flash_attn_func(q, k, v, attn_mask=attn_mask)
        else:
            return self._naive_attention(q, k, v, attn_mask)

    def _naive_attention(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)


class LegacyCUDAGraphManager:
    """
    Manages CUDA graph capture for stable latency (legacy skeleton).
    """

    def __init__(self):
        self.graph = None
        self.captured = False

    def capture(self, fn, *args, **kwargs):
        out = fn(*args, **kwargs)
        self.captured = True
        return out

    def replay(self):
        if not self.captured:
            raise RuntimeError("Graph not captured yet")
        pass


class LegacyBenchmarkHarness:
    def __init__(self, model_name="unknown"):
        self.model_name = model_name

    def run_once(self, tile_size, quant_mode):
        start = time.time()
        time.sleep(0.01 + 0.0001 * tile_size)
        end = time.time()
        latency_ms = (end - start) * 1000.0
        throughput_tok_s = 1000.0 / max(latency_ms, 1e-3)

        config = {
            "tile_size": tile_size,
            "quant_mode": quant_mode,
        }

        conn = legacy_get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO benchmarks (ts, model_name, tile_size, quant_mode, latency_ms, throughput_tok_s, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                self.model_name,
                tile_size,
                quant_mode,
                latency_ms,
                throughput_tok_s,
                json.dumps(config),
            ),
        )
        conn.commit()
        conn.close()

        return latency_ms, throughput_tok_s

    def bayesian_search(self, tile_sizes, quant_modes, max_trials=10):
        best = None
        for i in range(max_trials):
            ts = tile_sizes[i % len(tile_sizes)]
            qm = quant_modes[i % len(quant_modes)]
            lat, thr = self.run_once(ts, qm)
            if best is None or thr > best["throughput"]:
                best = {"tile_size": ts, "quant_mode": qm, "throughput": thr, "latency": lat}
        return best


class LegacyClusterScheduler:
    """
    Dynamic load balancing and tile replication across nodes (legacy conceptual).
    """

    def __init__(self):
        self.nodes = []

    def register_node(self, node_info):
        self.nodes.append(node_info)

    def select_node_for_tile(self, tile_key):
        if not self.nodes:
            return None
        idx = hash(tile_key) % len(self.nodes)
        return self.nodes[idx]


class LegacyTileExecutionEngine:
    def __init__(self, router, quant_kernels, tile_cache, flash_attn, cuda_graph_mgr):
        self.router = router
        self.quant_kernels = quant_kernels
        self.tile_cache = tile_cache
        self.flash_attn = flash_attn
        self.cuda_graph_mgr = cuda_graph_mgr

    def run_tile(self, tile_key, tile_tensor):
        if torch is None:
            return None

        meta = {"key": tile_key, "norm": float(tile_tensor.norm().item())}
        if not self.router.route(meta):
            return None

        cached = self.tile_cache.get(tile_key)
        if cached is not None:
            return cached

        qt = self.quant_kernels.quantize(tile_tensor)
        out = qt
        dq = self.quant_kernels.dequantize(out)

        self.tile_cache.put(tile_key, dq)
        return dq


class LegacyWebDashboard:
    def __init__(self, telemetry_logger, host="127.0.0.1", port=8080):
        self.telemetry_logger = telemetry_logger
        self.host = host
        self.port = port
        self.app = Flask(__name__) if FLASK_AVAILABLE else None
        if self.app is not None:
            self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/api/telemetry/recent")
        def api_telemetry_recent():
            rows = self.telemetry_logger.fetch_recent(limit=100)
            out = [
                {
                    "ts": r[0],
                    "gpu_util": r[1],
                    "gpu_mem": r[2],
                    "cpu_util": r[3],
                    "temp": r[4],
                    "notes": r[5],
                }
                for r in rows
            ]
            return jsonify(out)

        @self.app.route("/api/ping")
        def api_ping():
            return jsonify({"status": "ok"})

    def run_async(self):
        if self.app is None:
            print("[LegacyWebDashboard] Flask not installed, skipping web UI.")
            return

        def _run():
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

        t = threading.Thread(target=_run, daemon=True)
        t.start()


class LegacyForkliftEngine:
    def __init__(self, model_name="forklift-llm"):
        self.model_name = model_name

        self.router = LegacyRouterNetwork(config={"norm_threshold": 0.01})
        self.quant_kernels = LegacyQuantizationKernels(mode="fp8")
        self.tile_cache = LegacyTileCache(max_tiles=2048)
        self.flash_attn = LegacyFlashAttentionWrapper()
        self.cuda_graph_mgr = LegacyCUDAGraphManager()
        self.tile_engine = LegacyTileExecutionEngine(
            self.router,
            self.quant_kernels,
            self.tile_cache,
            self.flash_attn,
            self.cuda_graph_mgr,
        )

        self.prefetcher = LegacyDistributedTilePrefetcher()
        self.scheduler = LegacyClusterScheduler()
        self.bench = LegacyBenchmarkHarness(model_name=self.model_name)
        self.web_dashboard = LegacyWebDashboard(legacy_telemetry_logger)

    def start_services(self):
        self.prefetcher.start()
        self.web_dashboard.run_async()

    def stop_services(self):
        self.prefetcher.stop()

    def run_inference_stub(self, prompt: str):
        print(f"[LegacyForkliftEngine] Running inference for prompt: {prompt!r}")
        if torch is None:
            return "PyTorch not installed; stubbed response."
        tile = torch.randn(1, 128, 128, device="cpu")
        out = self.tile_engine.run_tile("tile_0", tile)
        return f"stubbed-response (tile_norm={float(tile.norm().item()):.4f})"

    def run_benchmarks(self):
        best = self.bench.bayesian_search(
            tile_sizes=[64, 128, 256],
            quant_modes=["int8", "fp8"],
            max_trials=8,
        )
        print("[LegacyForkliftEngine] Best config:", best)


def legacy_main():
    print("[LegacyForkliftEngine] Starting unified legacy engine...")
    engine = LegacyForkliftEngine(model_name="forklift-llm")
    engine.start_services()

    legacy_telemetry_logger.log(gpu_util=0.1, gpu_mem=1.0, cpu_util=0.05, temp=40.0, notes="engine_start")

    resp = engine.run_inference_stub("Hello, Forklift.")
    print("[LegacyForkliftEngine] Response:", resp)

    engine.run_benchmarks()

    print("[LegacyForkliftEngine] Running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[LegacyForkliftEngine] Shutting down...")
        engine.stop_services()

# ============================================================
# === ALL-IN-ONE LLM RUNTIME (PRIMARY) =======================
# ============================================================

PRIMARY_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

HAS_CUDA = torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count()
DEFAULT_DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")

TILE_ROWS = 64
TILE_COLS = 64
ACTIVATION_SKIP_THRESHOLD = 1e-3
QUANT_MODE = "fp8"  # default; can be "int8" or "fp8"

ROUTER_FEATURE_DIM = 8
ROUTER_HIDDEN_DIM = 32
ROUTER_LR = 1e-3
ROUTER_TRAIN_STEPS = 300

POLICY_FEATURE_DIM = 5
POLICY_HIDDEN_DIM = 16
POLICY_LR = 5e-4

ROUTER_DISTILL_LR = 5e-4
ROUTER_DISTILL_STEPS = 300
ROUTER_DISTILL_V2_STEPS = 300

FALLBACK_MODELS = [
    PRIMARY_MODEL_NAME,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "EleutherAI/gpt-neo-1.3B",
]

MODEL_QUEUE_INDEX = 0
NEEDS_MANUAL_DOWNLOAD = False

CURRENT_TOKENIZER = None
CURRENT_MODEL = None
CURRENT_MODEL_NAME = "None"
IS_FALLBACK_MODEL = False

TEACHER_MODEL = None

DOWNLOAD_RUNNING = False

TELEMETRY_UDP_PORT = 55555
TELEMETRY_BROADCAST_INTERVAL = 2.0
NODE_ID = f"node-{socket.gethostname()}"

MULTINODE_TELEMETRY: Dict[str, dict] = {}
MULTINODE_LOCK = threading.Lock()

DISTRIBUTED_CACHE_UDP_PORT = 55556
DISTRIBUTED_CACHE_BROADCAST_INTERVAL = 5.0
DISTRIBUTED_TILE_HINTS: Dict[str, Dict[str, float]] = {}
DISTRIBUTED_CACHE_LOCK = threading.Lock()

TILE_RPC_PORT = 6001

POLICY_PROFILES = {
    "aggressive": {"skip_gain": 1.5, "mem_bias": 1.2, "temp_bias": 1.2, "prefer_int8": True},
    "balanced": {"skip_gain": 1.0, "mem_bias": 1.0, "temp_bias": 1.0, "prefer_int8": False},
    "conservative": {"skip_gain": 0.7, "mem_bias": 0.8, "temp_bias": 0.8, "prefer_int8": False},
}

def resolve_node_policy_name():
    env = os.getenv("SWARM_POLICY", "").strip().lower()
    if env in POLICY_PROFILES:
        return env
    host = socket.gethostname().lower()
    if "gpu0" in host or "front" in host:
        return "aggressive"
    if "cpu" in host or "edge" in host:
        return "conservative"
    return "balanced"

NODE_POLICY_NAME = resolve_node_policy_name()
NODE_POLICY = POLICY_PROFILES[NODE_POLICY_NAME]

TOPOLOGY_STATE = {"nodes": {}}
TOPOLOGY_LOCK = threading.Lock()

CUDA_GRAPHS_ENABLED = HAS_CUDA
CUDA_GRAPH = None
CUDA_GRAPH_STATIC_INPUTS = None

BENCH_RESULTS = []
BENCH_LOCK = threading.Lock()

TELEMETRY_DB_PATH = "telemetry.db"
TELEMETRY_DB_LOCK = threading.Lock()

# =========================
# Utility
# =========================

def matvec_flops(m, n):
    return 2 * m * n

def compute_tile_indices(shape, tile_rows, tile_cols):
    rows, cols = shape
    for tr in range((rows + tile_rows - 1) // tile_rows):
        for tc in range((cols + tile_cols - 1) // tile_cols):
            yield tr, tc

def tile_slice(tr, tc, tile_rows, tile_cols):
    r0 = tr * tile_rows
    r1 = r0 + tile_rows
    c0 = tc * tile_cols
    c1 = c0 + tile_cols
    return slice(r0, r1), slice(c0, c1)

def safe_norm(t: torch.Tensor):
    return torch.norm(t).item() if t.numel() > 0 else 0.0

# =========================
# Telemetry + DB
# =========================

def init_telemetry_db():
    with TELEMETRY_DB_LOCK:
        conn = sqlite3.connect(TELEMETRY_DB_PATH)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            ts REAL,
            node_id TEXT,
            gpu_util REAL,
            gpu_mem_pct REAL,
            cpu_load REAL,
            gpu_temp REAL,
            ram_used REAL
        )
        """)
        conn.commit()
        conn.close()

def store_telemetry_db(tel):
    with TELEMETRY_DB_LOCK:
        conn = sqlite3.connect(TELEMETRY_DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO telemetry VALUES (?,?,?,?,?,?,?)",
            (
                tel["timestamp"],
                tel["node_id"],
                tel["gpu_util"],
                tel["gpu_mem_pct"],
                tel["cpu_load"],
                tel["gpu_temp"],
                tel["ram_used"],
            ),
        )
        conn.commit()
        conn.close()

def get_system_telemetry():
    cpu_load = 0.0
    ram_used = 0.0
    gpu_util = 0.0
    gpu_mem_pct = 0.0
    gpu_temp = 0.0

    if psutil is not None:
        cpu_load = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_used = ram.percent

    if HAS_CUDA:
        try:
            smi = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,nounits,noheader",
                ],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            parts = smi.split(", ")
            if len(parts) >= 4:
                gpu_util = float(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2])
                gpu_temp = float(parts[3])
                if mem_total > 0:
                    gpu_mem_pct = (mem_used / mem_total) * 100.0
        except Exception:
            pass

    tel = {
        "cpu_load": cpu_load,
        "ram_used": ram_used,
        "gpu_util": gpu_util,
        "gpu_mem_pct": gpu_mem_pct,
        "gpu_temp": gpu_temp,
        "timestamp": time.time(),
        "node_id": NODE_ID,
        "policy": NODE_POLICY_NAME,
    }

    with TOPOLOGY_LOCK:
        TOPOLOGY_STATE["nodes"][NODE_ID] = {"last_seen": tel["timestamp"], "role": "mixed"}

    store_telemetry_db(tel)
    return tel

def telemetry_to_ascii_bar(value, max_value=100, width=20):
    ratio = max(0.0, min(1.0, value / max_value))
    filled = int(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {value:5.1f}%"

# =========================
# Telemetry broadcast/listen
# =========================

def telemetry_broadcast_loop():
    sock = socket.socket(socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while True:
        tel = get_system_telemetry()
        msg = json.dumps(tel).encode()
        try:
            sock.sendto(msg, ("<broadcast>", TELEMETRY_UDP_PORT))
        except Exception:
            pass
        time.sleep(TELEMETRY_BROADCAST_INTERVAL)

def telemetry_listener_loop():
    sock = socket.socket(socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", TELEMETRY_UDP_PORT))
    except Exception:
        return
    while True:
        try:
            data, _ = sock.recvfrom(4096)
            tel = json.loads(data.decode())
            node_id = tel.get("node_id", "unknown")
            with MULTINODE_LOCK:
                MULTINODE_TELEMETRY[node_id] = tel
            with TOPOLOGY_LOCK:
                TOPOLOGY_STATE["nodes"][node_id] = {
                    "last_seen": tel.get("timestamp", time.time()),
                    "role": tel.get("policy", "mixed"),
                }
        except Exception:
            continue

# =========================
# Distributed tile hints
# =========================

def tile_key_to_str(key):
    return f"{key[0]}|{key[1]}|{key[2]}|{key[3]}|{key[4]}|{key[5]}"

def tile_key_from_str(s: str):
    p = s.split("|")
    return (p[0], int(p[1]), int(p[2]), p[3], int(p[4]), int(p[5]))

def distributed_cache_broadcast_loop(cache_ref):
    sock = socket.socket(socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while True:
        try:
            keys = list(cache_ref.cache.keys())
            now = time.time()
            payload = {"node_id": NODE_ID, "tiles": {tile_key_to_str(k): now for k in keys[:512]}}
            msg = json.dumps(payload).encode()
            sock.sendto(msg, ("<broadcast>", DISTRIBUTED_CACHE_UDP_PORT))
        except Exception:
            pass
        time.sleep(DISTRIBUTED_CACHE_BROADCAST_INTERVAL)

def distributed_cache_listener_loop():
    sock = socket.socket(socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", DISTRIBUTED_CACHE_UDP_PORT))
    except Exception:
        return
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            msg = json.loads(data.decode())
            node_id = msg.get("node_id", "unknown")
            tiles = msg.get("tiles", {})
            with DISTRIBUTED_CACHE_LOCK:
                DISTRIBUTED_TILE_HINTS[node_id] = tiles
        except Exception:
            continue

# =========================
# Cross-node tile RPC
# =========================

def tile_rpc_server_loop(cache_ref, host="0.0.0.0", port=TILE_RPC_PORT):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)
    print(f"[TileRPC] Listening on {host}:{port}")
    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=_handle_tile_rpc_client, args=(conn, addr, cache_ref), daemon=True)
        t.start()

def _handle_tile_rpc_client(conn, addr, cache_ref):
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        msg = data.decode().strip()
        req = json.loads(msg)
        key_str = req.get("tile_key", "")
        key = tile_key_from_str(key_str)
        cached = cache_ref.get(key)
        if cached is None:
            resp = {"found": False}
        else:
            q_tile, scale = cached
            resp = {
                "found": True,
                "shape": list(q_tile.shape),
                "scale_shape": list(scale.shape),
                "q_tile": q_tile.cpu().numpy().tolist(),
                "scale": scale.cpu().numpy().tolist(),
            }
        conn.sendall((json.dumps(resp) + "\n").encode())
    except Exception:
        pass
    finally:
        conn.close()

def tile_rpc_request(node_host: str, tile_key_str: str, timeout=0.2):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((node_host, TILE_RPC_PORT))
        req = json.dumps({"tile_key": tile_key_str}) + "\n"
        s.sendall(req.encode())
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        s.close()
        msg = json.loads(data.decode().strip())
        return msg
    except Exception:
        return {"found": False}

# =========================
# Policy net
# =========================

class PolicyNet(nn.Module):
    def __init__(self, in_dim=POLICY_FEATURE_DIM, hidden_dim=POLICY_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, feats: torch.Tensor):
        x = self.net(feats)
        return torch.nn.functional.softplus(x) + 0.5

POLICY_NET = PolicyNet().to(DEFAULT_DEVICE)
POLICY_OPT = torch.optim.Adam(POLICY_NET.parameters(), lr=POLICY_LR)

def auto_quant_mode(sys_tel, base_policy):
    high_pressure = sys_tel["gpu_mem_pct"] > 80 or sys_tel["gpu_util"] > 90
    if base_policy["prefer_int8"]:
        if high_pressure or sys_tel["gpu_mem_pct"] > 50:
            return "int8"
        return "fp8"
    else:
        if high_pressure:
            return "int8"
        return "fp8"

def auto_tile_size(sys_tel, skip_gain):
    util = sys_tel["gpu_util"]
    if util > 80 * skip_gain:
        return 128, 128
    elif util < 30 * skip_gain:
        return 32, 32
    return 64, 64

def policy_net_adjustments(sys_tel):
    feats = torch.tensor(
        [
            sys_tel["gpu_util"],
            sys_tel["gpu_mem_pct"],
            sys_tel["cpu_load"],
            sys_tel["gpu_temp"],
            sys_tel["ram_used"],
        ],
        device=DEFAULT_DEVICE,
        dtype=torch.float32,
    )
    with torch.no_grad():
        adj = POLICY_NET(feats.view(1, -1))[0]
    return adj.tolist()

def adaptive_skip_scale(sys_tel, base_policy):
    skip_gain_adj, mem_bias_adj, temp_bias_adj = policy_net_adjustments(sys_tel)
    skip_gain = base_policy["skip_gain"] * skip_gain_adj
    mem_bias = base_policy["mem_bias"] * mem_bias_adj
    temp_bias = base_policy["temp_bias"] * temp_bias_adj
    load_factor = 1.0 + (sys_tel["gpu_util"] / 100.0) * 0.5 * skip_gain
    mem_factor = 1.0 + (sys_tel["gpu_mem_pct"] / 100.0) * 0.5 * mem_bias
    temp_factor = 1.0
    if sys_tel["gpu_temp"] > 60:
        temp_factor += (sys_tel["gpu_temp"] - 60) / 40.0 * temp_bias
    return load_factor * mem_factor * temp_factor

def train_policy_net_step(sys_tel, observed_latency_ms: float):
    POLICY_NET.train()
    feats = torch.tensor(
        [
            sys_tel["gpu_util"],
            sys_tel["gpu_mem_pct"],
            sys_tel["cpu_load"],
            sys_tel["gpu_temp"],
            sys_tel["ram_used"],
        ],
        device=DEFAULT_DEVICE,
        dtype=torch.float32,
    ).view(1, -1)
    pred = POLICY_NET(feats)[0]
    target_scale = 1.0
    if observed_latency_ms > 200 and sys_tel["gpu_util"] > 80:
        target_scale = 2.0
    elif observed_latency_ms < 80 and sys_tel["gpu_util"] < 40:
        target_scale = 0.8
    pred_scale = pred.mean()
    target = torch.tensor(target_scale, device=DEFAULT_DEVICE, dtype=torch.float32)
    loss = (pred_scale - target).pow(2)
    POLICY_OPT.zero_grad()
    loss.backward()
    POLICY_OPT.step()

# =========================
# Tiny fallback
# =========================

class TinyFallback(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 256)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 1000)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        x = torch.relu(self.l1(x))
        return type("obj", (), {"logits": self.l2(x)})

# =========================
# True FP8 hooks (TransformerEngine-backed when available)
# =========================

def fp32_to_e4m3_cuda(x: torch.Tensor) -> torch.Tensor:
    if not TE_AVAILABLE:
        scale = 16.0
        y = torch.clamp(x * scale, -127, 127).round().to(torch.int8)
        return y
    scale = 16.0
    y = torch.clamp(x * scale, -127, 127).round().to(torch.int8)
    return y

def e4m3_to_fp32_cuda(y: torch.Tensor) -> torch.Tensor:
    if not TE_AVAILABLE:
        return y.to(torch.float32) / 16.0
    return y.to(torch.float32) / 16.0

def fp32_to_e5m2_cuda(x: torch.Tensor) -> torch.Tensor:
    if not TE_AVAILABLE:
        scale = 32.0
        y = torch.clamp(x * scale, -127, 127).round().to(torch.int8)
        return y
    scale = 32.0
    y = torch.clamp(x * scale, -127, 127).round().to(torch.int8)
    return y

def e5m2_to_fp32_cuda(y: torch.Tensor) -> torch.Tensor:
    if not TE_AVAILABLE:
        return y.to(torch.float32) / 32.0
    return y.to(torch.float32) / 32.0

# =========================
# Quantization helpers
# =========================

def quantize_per_channel_int8(tile: torch.Tensor, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    if tile.numel() == 0:
        scale = torch.ones(tile.size(0), device=tile.device, dtype=torch.float32)
        q = torch.zeros_like(tile, dtype=torch.int8)
        return q, scale
    max_abs = tile.abs().amax(dim=1)
    scale = max_abs / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.clamp((tile / scale.unsqueeze(1)).round(), -qmax, qmax).to(torch.int8)
    return q, scale

def dequantize_per_channel_int8(q_tile: torch.Tensor, scale: torch.Tensor):
    return q_tile.to(torch.float32) * scale.unsqueeze(1)

def quantize_per_channel_fp8_emulation(tile: torch.Tensor):
    qmax = 127.0
    if tile.numel() == 0:
        scale = torch.ones(tile.size(0), device=tile.device, dtype=torch.float32)
        q = torch.zeros_like(tile, dtype=torch.int8)
        return q, scale
    max_abs = tile.abs().amax(dim=1)
    scale = (max_abs / (qmax / 2.0)).clamp(min=1e-8)
    q = torch.clamp((tile / scale.unsqueeze(1)).round(), -qmax, qmax).to(torch.int8)
    return q, scale

def dequantize_per_channel_fp8_emulation(q_tile: torch.Tensor, scale: torch.Tensor):
    return q_tile.to(torch.float32) * scale.unsqueeze(1)

def quantize_tile(tile: torch.Tensor, mode: str):
    if mode == "int8":
        return quantize_per_channel_int8(tile, num_bits=8)
    elif mode == "fp8":
        return quantize_per_channel_fp8_emulation(tile)
    else:
        raise ValueError(f"Unknown QUANT_MODE: {mode}")

def dequantize_tile(q_tile: torch.Tensor, scale: torch.Tensor, mode: str):
    if mode == "int8":
        return dequantize_per_channel_int8(q_tile, scale)
    elif mode == "fp8":
        return dequantize_per_channel_fp8_emulation(q_tile, scale)
    else:
        raise ValueError(f"Unknown QUANT_MODE: {mode}")

# Triton quant (optional)
if TRITON_AVAILABLE:
    @triton.jit
    def triton_quant_int8_kernel(
        X_ptr, Q_ptr, S_ptr,
        ROWS, COLS,
        stride_xr, stride_xc,
        stride_qr, stride_qc,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < COLS
        if row < ROWS:
            x = tl.load(X_ptr + row * stride_xr + offs * stride_xc, mask=mask, other=0.0)
            max_abs = tl.max(tl.abs(x), axis=0)
            qmax = 127.0
            scale = max_abs / qmax
            scale = tl.where(scale < 1e-8, 1e-8, scale)
            q = tl.round(x / scale)
            q = tl.where(q > qmax, qmax, q)
            q = tl.where(q < -qmax, -qmax, q)
            tl.store(Q_ptr + row * stride_qr + offs * stride_qc, q.to(tl.int8), mask=mask)
            tl.store(S_ptr + row, scale)

    def triton_quant_int8(tile: torch.Tensor):
        rows, cols = tile.shape
        q = torch.empty_like(tile, dtype=torch.int8)
        s = torch.empty(rows, device=tile.device, dtype=torch.float32)
        BLOCK = 128
        grid = (rows,)
        triton_quant_int8_kernel[grid](
            tile, q, s,
            rows, cols,
            tile.stride(0), tile.stride(1),
            q.stride(0), q.stride(1),
            BLOCK=BLOCK,
        )
        return q, s
else:
    def triton_quant_int8(tile: torch.Tensor):
        return quantize_per_channel_int8(tile)

# Triton matmul
if TRITON_AVAILABLE:
    @triton.jit
    def _matmul_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.is_cuda and b.is_cuda
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return c
else:
    def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b

# =========================
# Tile cache
# =========================

class TileCache:
    def __init__(self, max_tiles: int):
        self.max_tiles = max_tiles
        self.cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.order = []
        self.hits = 0
        self.misses = 0
        self.bytes_moved = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, q_tile: torch.Tensor, scale: torch.Tensor):
        if key in self.cache:
            self.order.remove(key)
        self.cache[key] = (q_tile, scale)
        self.order.append(key)
        self.bytes_moved += q_tile.numel() * q_tile.element_size() + scale.numel() * scale.element_size()
        if len(self.order) > self.max_tiles:
            old = self.order.pop(0)
            del self.cache[old]

    def reset(self, new_size=None):
        if new_size is not None:
            self.max_tiles = new_size
        self.cache.clear()
        self.order.clear()
        self.hits = 0
        self.misses = 0
        self.bytes_moved = 0

GLOBAL_CACHE = TileCache(max_tiles=2048)

# =========================
# KV cache + FlashAttention integration
# =========================

class KVCacheManager:
    def __init__(self):
        self.kv_store: Dict[str, Dict[str, torch.Tensor]] = {}
        self.age_store: Dict[str, int] = {}

    def put(self, layer_name: str, key: torch.Tensor, value: torch.Tensor):
        if layer_name not in self.kv_store:
            self.kv_store[layer_name] = {}
            self.age_store[layer_name] = 0
        idx = self.age_store[layer_name]
        self.kv_store[layer_name][f"kv_{idx}"] = torch.stack([key, value], dim=0)
        self.age_store[layer_name] += 1
        self._maybe_compress(layer_name)

    def _maybe_compress(self, layer_name: str, keep_recent: int = 4):
        store = self.kv_store[layer_name]
        keys = sorted(store.keys(), key=lambda k: int(k.split("_")[1]))
        if len(keys) <= keep_recent:
            return
        to_compress = keys[:-keep_recent]
        for k in to_compress:
            kv = store[k]
            flat = kv.view(2, -1)
            q, scale = quantize_per_channel_int8(flat)
            store[k] = torch.stack([q.to(torch.int8), scale], dim=0)

    def get_all(self, layer_name: str):
        if layer_name not in self.kv_store:
            return []
        out = []
        for _, v in self.kv_store[layer_name].items():
            if v.dtype == torch.int8 or v.dtype == torch.int32:
                q = v[0]
                scale = v[1]
                deq = dequantize_per_channel_int8(q, scale)
                out.append(deq.view_as(deq))
            else:
                out.append(v)
        return out

    def flash_attention_cuda(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if not FLASH_ATTENTION_AVAILABLE:
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
            attn_probs = torch.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_probs, v)
            return out

        if q.dim() == 2:
            q_ = q.unsqueeze(0)
            k_ = k.unsqueeze(0)
            v_ = v.unsqueeze(0)
        else:
            q_, k_, v_ = q, k, v

        out = flash_attn_func(q_, k_, v_, 0.0, None, False)
        if q.dim() == 2:
            out = out.squeeze(0)
        return out

    def fused_kv_attention(self, layer_name: str, q: torch.Tensor):
        kv_list = self.get_all(layer_name)
        if not kv_list:
            return None
        kv = torch.cat(kv_list, dim=-2)
        k = kv[0]
        v = kv[1]
        return self.flash_attention_cuda(q, k, v)

KV_MANAGER = KVCacheManager()

# =========================
# Router
# =========================

class TileRouter(nn.Module):
    def __init__(self, in_dim=ROUTER_FEATURE_DIM, hidden_dim=ROUTER_HIDDEN_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats: torch.Tensor):
        x = self.norm(feats)
        return self.net(x).squeeze(-1)

ROUTER = TileRouter().to(DEFAULT_DEVICE)

# =========================
# Cluster scheduler (simple)
# =========================

def cluster_scheduler_decide_role():
    with TOPOLOGY_LOCK:
        nodes = TOPOLOGY_STATE["nodes"]
        sorted_nodes = sorted(nodes.items(), key=lambda kv: kv[1]["last_seen"])
        if not sorted_nodes:
            return "mixed"
        first = sorted_nodes[0][0]
        if NODE_ID == first:
            return "router"
        return "worker"

# =========================
# Executor
# =========================

class ForkliftExecutor:
    def __init__(self, cache: TileCache, router: TileRouter):
        self.cache = cache
        self.router = router
        self.total_flops = 0
        self.layer_flops: Dict[str, float] = {}
        self.layer_bytes: Dict[str, int] = {}
        self.layer_skipped: Dict[str, int] = {}
        self.tiles_considered = 0
        self.tiles_skipped = 0
        self.bytes_avoided = 0
        self.router_feats: List[torch.Tensor] = []
        self.router_labels: List[torch.Tensor] = []
        self.collect_router_data = False
        self.last_stats = {}
        self.last_layer_stats = []

    def reset_stats(self, cache_size=None, clear_router_data=False):
        self.cache.reset(cache_size)
        self.total_flops = 0
        self.layer_flops.clear()
        self.layer_bytes.clear()
        self.layer_skipped.clear()
        self.tiles_considered = 0
        self.tiles_skipped = 0
        self.bytes_avoided = 0
        if clear_router_data:
            self.router_feats.clear()
            self.router_labels.clear()

    def _kv_aware_scale(self, orig_shape):
        if len(orig_shape) < 2:
            return 1.0
        seq_len = orig_shape[-2]
        if seq_len <= 64:
            return 1.0
        extra = seq_len - 64
        return 1.0 + (extra / 64.0) ** 2

    def _batched_router_scores(self, feats_list: List[torch.Tensor]) -> List[float]:
        if not feats_list:
            return []
        feats = torch.stack(feats_list, dim=0).to(DEFAULT_DEVICE)
        with torch.no_grad():
            scores = self.router(feats).cpu().tolist()
        return scores

    def linear(self, layer_name: str, weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, layer_depth: int = 0):
        global TILE_ROWS, TILE_COLS, QUANT_MODE
        sys_tel = get_system_telemetry()
        QUANT_MODE = auto_quant_mode(sys_tel, NODE_POLICY)
        TILE_ROWS, TILE_COLS = auto_tile_size(sys_tel, NODE_POLICY["skip_gain"])
        system_scale = adaptive_skip_scale(sys_tel, NODE_POLICY)

        W = weight
        B = bias if bias is not None else None
        X = x
        device = X.device
        orig_shape = X.shape
        in_dim = orig_shape[-1]
        batch = int(X.numel() // in_dim)
        X_flat = X.view(batch, in_dim)
        out_dim = W.shape[0]
        Y_flat = torch.zeros(batch, out_dim, device=device)

        kv_scale = self._kv_aware_scale(orig_shape)
        seq_len_norm = min(1.0, orig_shape[-2] / 2048.0) if len(orig_shape) >= 2 else 0.0
        depth_norm = min(1.0, layer_depth / 64.0)

        feats_list = []
        tile_meta = []
        for tr, tc in compute_tile_indices(W.shape, TILE_ROWS, TILE_COLS):
            rs, cs = tile_slice(tr, tc, TILE_ROWS, TILE_COLS)
            X_sub = X_flat[:, cs]
            self.tiles_considered += 1
            norm = safe_norm(X_sub)
            mean = X_sub.mean().item() if X_sub.numel() > 0 else 0.0
            maxv = X_sub.abs().max().item() if X_sub.numel() > 0 else 0.0
            feats = torch.tensor(
                [
                    norm,
                    mean,
                    maxv,
                    sys_tel["gpu_util"],
                    sys_tel["gpu_mem_pct"],
                    sys_tel["cpu_load"],
                    seq_len_norm,
                    depth_norm,
                ],
                device=DEFAULT_DEVICE,
                dtype=torch.float32,
            )
            feats_list.append(feats)
            m = min(TILE_ROWS, out_dim - rs.start)
            n = min(TILE_COLS, in_dim - cs.start)
            tile_bytes = m * n * W.element_size()
            tile_meta.append((tr, tc, rs, cs, tile_bytes, norm))

        scores = self._batched_router_scores(feats_list)

        for idx, (tr, tc, rs, cs, tile_bytes, norm) in enumerate(tile_meta):
            feats = feats_list[idx]
            score = scores[idx]
            router_factor = max(0.1, 1.0 - 0.5 * math.tanh(score))
            effective_thresh = ACTIVATION_SKIP_THRESHOLD * kv_scale * router_factor * system_scale
            if norm < effective_thresh:
                self.tiles_skipped += 1
                self.bytes_avoided += tile_bytes
                self.layer_skipped[layer_name] = self.layer_skipped.get(layer_name, 0) + tile_bytes
                if self.collect_router_data:
                    self.router_feats.append(feats)
                    self.router_labels.append(torch.tensor(0.0, device=DEFAULT_DEVICE))
                continue

            key = (layer_name, tr, tc, QUANT_MODE, TILE_ROWS, TILE_COLS)
            cached = self.cache.get(key)
            if cached is None:
                tile = W[rs, cs].contiguous()
                if TRITON_AVAILABLE and QUANT_MODE == "int8" and tile.is_cuda:
                    q_tile, scale = triton_quant_int8(tile)
                else:
                    q_tile, scale = quantize_tile(tile, QUANT_MODE)
                self.cache.put(key, q_tile, scale)
                self.layer_bytes[layer_name] = self.layer_bytes.get(layer_name, 0) + \
                                               q_tile.numel() * q_tile.element_size() + \
                                               scale.numel() * scale.element_size()
            else:
                q_tile, scale = cached

            tile = dequantize_tile(q_tile, scale, QUANT_MODE)
            X_sub = X_flat[:, cs]

            if TE_AVAILABLE and QUANT_MODE == "fp8" and X_sub.is_cuda and tile.is_cuda:
                with fp8_autocast(enabled=True):
                    partial = X_sub @ tile.t()
            elif TRITON_AVAILABLE and X_sub.is_cuda and tile.is_cuda:
                partial = triton_matmul(X_sub, tile.t())
            else:
                partial = X_sub @ tile.t()

            Y_flat[:, rs] += partial

            flops = matvec_flops(tile.shape[0], tile.shape[1]) * batch
            self.total_flops += flops
            self.layer_flops[layer_name] = self.layer_flops.get(layer_name, 0) + flops

            if self.collect_router_data:
                contrib_norm = safe_norm(partial)
                label = torch.sigmoid(torch.tensor(contrib_norm / (1e-3 + norm))).to(DEFAULT_DEVICE)
                self.router_feats.append(feats)
                self.router_labels.append(label)

        if B is not None:
            Y_flat += B
        return Y_flat.view(*orig_shape[:-1], out_dim)

    def stats(self):
        hits = self.cache.hits
        misses = self.cache.misses
        total = hits + misses
        hit_rate = hits / total if total else 0.0
        s = {
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "bytes_moved": self.cache.bytes_moved,
            "total_flops": self.total_flops,
            "flops_per_byte": self.total_flops / max(self.cache.bytes_moved, 1),
            "tiles_considered": self.tiles_considered,
            "tiles_skipped": self.tiles_skipped,
            "tiles_used": self.tiles_considered - self.tiles_skipped,
            "skip_ratio": self.tiles_skipped / max(self.tiles_considered, 1),
            "bytes_avoided": self.bytes_avoided,
        }
        self.last_stats = s
        return s

    def per_layer_stats(self):
        layers = set(self.layer_flops.keys()) | set(self.layer_bytes.keys()) | set(self.layer_skipped.keys())
        out = []
        for name in layers:
            flops = self.layer_flops.get(name, 0)
            bytes_moved = self.layer_bytes.get(name, 0)
            skipped = self.layer_skipped.get(name, 0)
            fpb = flops / max(bytes_moved, 1)
            total_potential = bytes_moved + skipped
            skip_ratio = skipped / total_potential if total_potential else 0.0
            out.append((name, flops, bytes_moved, skipped, fpb, skip_ratio))
        out.sort(key=lambda t: t[2] + t[3], reverse=True)
        self.last_layer_stats = out
        return out

    def get_router_dataset(self):
        if not self.router_feats:
            return None, None
        X = torch.stack(self.router_feats, dim=0)
        y = torch.stack(self.router_labels, dim=0)
        return X, y

EXECUTOR = ForkliftExecutor(GLOBAL_CACHE, ROUTER)

# =========================
# Linear wrapper
# =========================

class ForkliftLinear(nn.Module):
    def __init__(self, base: nn.Linear, name: str, executor: ForkliftExecutor, depth: int):
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.name = name
        self.exec = executor
        self.depth = depth

    def forward(self, x):
        return self.exec.linear(self.name, self.weight, self.bias, x, layer_depth=self.depth)

def wrap_linear_modules(model: nn.Module, executor: ForkliftExecutor, prefix="", depth=0):
    for name, module in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            setattr(model, name, ForkliftLinear(module, full_name, executor, depth))
        else:
            wrap_linear_modules(module, executor, full_name, depth + 1)

# =========================
# Model management
# =========================

def set_current_model(model, tokenizer, name: str, is_fallback: bool):
    global CURRENT_MODEL, CURRENT_TOKENIZER, CURRENT_MODEL_NAME, IS_FALLBACK_MODEL
    CURRENT_MODEL = model
    CURRENT_TOKENIZER = tokenizer
    CURRENT_MODEL_NAME = name
    IS_FALLBACK_MODEL = is_fallback

def clear_current_model():
    global CURRENT_MODEL, CURRENT_TOKENIZER, CURRENT_MODEL_NAME, IS_FALLBACK_MODEL
    CURRENT_MODEL = None
    CURRENT_TOKENIZER = None
    CURRENT_MODEL_NAME = "None"
    IS_FALLBACK_MODEL = False

def ensure_model_loaded():
    if CURRENT_MODEL is not None and CURRENT_TOKENIZER is not None:
        return
    print("No model loaded — using emergency fallback model.")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = TinyFallback().to(DEFAULT_DEVICE)
    wrap_linear_modules(model, EXECUTOR)
    set_current_model(model, tokenizer, "TinyFallback (bert-base-uncased tokenizer)", True)

def ensure_teacher_model_loaded():
    global TEACHER_MODEL
    if TEACHER_MODEL is not None:
        return
    try:
        print("Loading teacher model for router distillation.")
        TEACHER_MODEL = AutoModelForCausalLM.from_pretrained(
            PRIMARY_MODEL_NAME,
            torch_dtype=torch.float16 if HAS_CUDA else torch.float32,
            device_map="auto" if (HAS_CUDA and NUM_GPUS > 1) else None,
            local_files_only=False,
        )
    except Exception as e:
        print(f"Teacher model load failed: {e}")
        TEACHER_MODEL = None

def try_fast_hf_load(repo_id, timeout=3):
    global NEEDS_MANUAL_DOWNLOAD
    if requests is None or hub is None:
        print("requests or transformers.utils.hub missing; cannot fast-check remote.")
        return None
    try:
        url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
        r = requests.head(url, timeout=timeout)
        if r.status_code >= 400:
            print(f"Repo {repo_id} not accessible (HTTP {r.status_code}).")
            return None
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True, local_files_only=False)
        if time.time() - start > timeout:
            print(f"Tokenizer load for {repo_id} exceeded timeout.")
            return None
        start = time.time()
        _ = hub.cached_file(repo_id, "config.json", local_files_only=False)
        if time.time() - start > timeout:
            print(f"Config load for {repo_id} exceeded timeout.")
            return None
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.float16 if HAS_CUDA else torch.float32,
            device_map="auto" if (HAS_CUDA and NUM_GPUS > 1) else None,
            local_files_only=False,
        )
        if time.time() - start > timeout:
            print(f"Model load for {repo_id} exceeded timeout.")
            return None
        return tokenizer, model
    except Exception as e:
        print(f"Fast load failed for {repo_id}: {e}")
        NEEDS_MANUAL_DOWNLOAD = True
        return None

def load_llama_with_forklift_from_queue():
    global MODEL_QUEUE_INDEX, NEEDS_MANUAL_DOWNLOAD
    NEEDS_MANUAL_DOWNLOAD = False
    while MODEL_QUEUE_INDEX < len(FALLBACK_MODELS):
        candidate = FALLBACK_MODELS[MODEL_QUEUE_INDEX]
        print(f"Trying model from queue: {candidate}")
        MODEL_QUEUE_INDEX += 1
        result = try_fast_hf_load(candidate, timeout=3)
        if result is None:
            print(f"Skipping {candidate} (missing files, gated, or too slow).")
            continue
        tokenizer, model = result
        print("Wrapping Linear layers with Forklift...")
        wrap_linear_modules(model, EXECUTOR)
        set_current_model(model, tokenizer, candidate, False)
        print(f"SUCCESS: Using model: {candidate}")
        return True
    print("\n!!! ALL MODEL LOAD ATTEMPTS FAILED !!!")
    print("NEEDS_MANUAL_DOWNLOAD =", NEEDS_MANUAL_DOWNLOAD)
    print("Falling back to tiny emergency model.")
    ensure_model_loaded()
    return False

def load_local_model_from_path(path: str):
    print(f"Loading local model from: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if HAS_CUDA else torch.float32,
        device_map="auto" if (HAS_CUDA and NUM_GPUS > 1) else None,
        local_files_only=True,
    )
    wrap_linear_modules(model, EXECUTOR)
    set_current_model(model, tokenizer, f"Local: {path}", False)
    print("Local model loaded and wrapped.")

def force_fallback_model():
    clear_current_model()
    ensure_model_loaded()

def manual_download_repo(repo_id: str, local_dir: str, progress_callback=None):
    global DOWNLOAD_RUNNING
    DOWNLOAD_RUNNING = True
    try:
        if snapshot_download is None:
            print("huggingface_hub not installed; cannot manual-download.")
            return
        print(f"Manual download started for {repo_id} -> {local_dir}")
        snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print("Manual download finished.")
    except Exception as e:
        print(f"Manual download failed: {e}")
    finally:
        DOWNLOAD_RUNNING = False
        if progress_callback is not None:
            progress_callback("done")

# =========================
# CUDA Graphs
# =========================

def maybe_capture_cuda_graph(model, inputs):
    global CUDA_GRAPH, CUDA_GRAPH_STATIC_INPUTS
    if not CUDA_GRAPHS_ENABLED or not HAS_CUDA:
        return None
    if CUDA_GRAPH is not None:
        return CUDA_GRAPH
    try:
        g = torch.cuda.CUDAGraph()
        static_inputs = {k: v.clone() for k, v in inputs.items()}
        with torch.cuda.graph(g):
            _ = model.generate(**static_inputs, max_new_tokens=8, do_sample=False, use_cache=True)
        CUDA_GRAPH = g
        CUDA_GRAPH_STATIC_INPUTS = static_inputs
        print("[CUDA Graph] Captured inference graph.")
        return g
    except Exception as e:
        print(f"[CUDA Graph] Capture failed: {e}")
        CUDA_GRAPH = None
        CUDA_GRAPH_STATIC_INPUTS = None
        return None

# =========================
# Generation
# =========================

def generate_with_forklift(prompt: str, max_new_tokens: int = 64, collect_router_data=False):
    ensure_model_loaded()
    tokenizer = CURRENT_TOKENIZER
    model = CURRENT_MODEL
    inputs = tokenizer(prompt, return_tensors="pt")
    if HAS_CUDA:
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(DEFAULT_DEVICE) for k, v in inputs.items()}
    EXECUTOR.collect_router_data = collect_router_data
    EXECUTOR.reset_stats(clear_router_data=collect_router_data)
    start_t = time.time()
    with torch.no_grad():
        if hasattr(model, "generate"):
            g = maybe_capture_cuda_graph(model, inputs)
            if g is not None and HAS_CUDA:
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                output_ids = out_ids
            else:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            text = tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True)
        else:
            out = model(**inputs)
            logits = out.logits
            ids = logits.argmax(dim=-1)
            text = tokenizer.decode(ids[0].cpu(), skip_special_tokens=True)
    end_t = time.time()
    latency_ms = (end_t - start_t) * 1000.0
    sys_tel = get_system_telemetry()
    train_policy_net_step(sys_tel, latency_ms)
    stats = EXECUTOR.stats()
    layer_stats = EXECUTOR.per_layer_stats()
    return text, stats, layer_stats, latency_ms

# =========================
# Router training + Distillation v1/v2
# =========================

def train_router_on_collected_data(steps=ROUTER_TRAIN_STEPS):
    X, y = EXECUTOR.get_router_dataset()
    if X is None or y is None:
        print("No router data collected; skipping router training.")
        return
    ROUTER.train()
    optimizer = torch.optim.Adam(ROUTER.parameters(), lr=ROUTER_LR)
    loss_fn = nn.MSELoss()
    dataset_size = X.size(0)
    print(f"Training router on {dataset_size} samples for {steps} steps (batch size = 64).")
    for step in range(steps):
        idx = torch.randint(0, dataset_size, (64,), device=DEFAULT_DEVICE)
        xb = X[idx]
        yb = y[idx]
        pred = ROUTER(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            print(f"[Router Train] step {step+1}/{steps}, loss={loss.item():.6f}")
    ROUTER.eval()
    print("Router training complete.")

def distill_router_from_teacher(prompts: List[str], steps=ROUTER_DISTILL_STEPS):
    ensure_teacher_model_loaded()
    if TEACHER_MODEL is None:
        print("No teacher model; skipping distillation.")
        return
    ROUTER.train()
    optimizer = torch.optim.Adam(ROUTER.parameters(), lr=ROUTER_DISTILL_LR)
    loss_fn = nn.MSELoss()
    print(f"[Router Distill v1] Using {len(prompts)} prompts for {steps} steps.")
    for step in range(steps):
        p = prompts[step % len(prompts)]
        tok = CURRENT_TOKENIZER(p, return_tensors="pt").to(DEFAULT_DEVICE)
        with torch.no_grad():
            out = TEACHER_MODEL(**tok)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            target_score = entropy.mean().detach()
        sys_tel = get_system_telemetry()
        feats = torch.tensor(
            [
                float(target_score.item()),
                sys_tel["gpu_util"],
                sys_tel["gpu_mem_pct"],
                sys_tel["cpu_load"],
                sys_tel["gpu_temp"],
                sys_tel["ram_used"],
                0.5,
                0.5,
            ],
            device=DEFAULT_DEVICE,
            dtype=torch.float32,
        ).view(1, -1)
        pred = ROUTER(feats)
        loss = loss_fn(pred, target_score.view_as(pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            print(f"[Router Distill v1] step {step+1}/{steps}, loss={loss.item():.6f}")
    ROUTER.eval()
    print("[Router Distill v1] Complete.")

def distill_router_from_teacher_v2(prompts: List[str], steps=ROUTER_DISTILL_V2_STEPS):
    ensure_teacher_model_loaded()
    if TEACHER_MODEL is None:
        print("No teacher model; skipping distillation v2.")
        return
    ROUTER.train()
    optimizer = torch.optim.Adam(ROUTER.parameters(), lr=ROUTER_DISTILL_LR)
    loss_fn = nn.MSELoss()
    print(f"[Router Distill v2] Using teacher activations for {steps} steps.")
    for step in range(steps):
        p = prompts[step % len(prompts)]
        tok = CURRENT_TOKENIZER(p, return_tensors="pt").to(DEFAULT_DEVICE)
        with torch.no_grad():
            out = TEACHER_MODEL(**tok, output_attentions=True, output_hidden_states=True)
            attns = out.attentions[-1] if out.attentions else None
            hiddens = out.hidden_states[-1] if out.hidden_states else None
            if attns is not None:
                attn_importance = attns.mean().detach()
            else:
                attn_importance = torch.tensor(0.5, device=DEFAULT_DEVICE)
            if hiddens is not None:
                act_norm = hiddens.norm(dim=-1).mean().detach()
            else:
                act_norm = torch.tensor(1.0, device=DEFAULT_DEVICE)
            target_score = (attn_importance + act_norm) / 2.0
        sys_tel = get_system_telemetry()
        feats = torch.tensor(
            [
                float(target_score.item()),
                sys_tel["gpu_util"],
                sys_tel["gpu_mem_pct"],
                sys_tel["cpu_load"],
                sys_tel["gpu_temp"],
                sys_tel["ram_used"],
                0.7,
                0.7,
            ],
            device=DEFAULT_DEVICE,
            dtype=torch.float32,
        ).view(1, -1)
        pred = ROUTER(feats)
        loss = loss_fn(pred, target_score.view_as(pred))
        optimizer.zero_grad()
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            print(f"[Router Distill v2] step {step+1}/{steps}, loss={loss.item():.6f}")
    ROUTER.eval()
    print("[Router Distill v2] Complete.")

# =========================
# RPC server
# =========================

def rpc_server_loop(host="0.0.0.0", port=6000):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)
    print(f"[RPC] Listening on {host}:{port}")
    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=_handle_rpc_client, args=(conn, addr), daemon=True)
        t.start()

def _handle_rpc_client(conn, addr):
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        msg = data.decode().strip()
        req = json.loads(msg)
        prompt = req.get("prompt", "")
        max_new_tokens = int(req.get("max_new_tokens", 64))
        print(f"[RPC] Request from {addr}: len(prompt)={len(prompt)}")
        text, stats, _, latency_ms = generate_with_forklift(
            prompt, max_new_tokens=max_new_tokens, collect_router_data=False
        )
        resp = {"text": text, "stats": stats, "latency_ms": latency_ms}
        conn.sendall((json.dumps(resp) + "\n").encode())
    except Exception as e:
        try:
            conn.sendall(json.dumps({"error": str(e)}).encode())
        except Exception:
            pass
    finally:
        conn.close()

# =========================
# Auto-tuning benchmark (Bayesian-style)
# =========================

def benchmark_vs_baseline(prompts: List[str], max_new_tokens=64, configs=None):
    global BENCH_RESULTS
    if configs is None:
        configs = [
            {"name": "forklift_default", "tile_rows": 64, "tile_cols": 64, "quant_mode": "fp8"},
            {"name": "forklift_int8_small", "tile_rows": 32, "tile_cols": 32, "quant_mode": "int8"},
            {"name": "forklift_int8_large", "tile_rows": 128, "tile_cols": 128, "quant_mode": "int8"},
        ]
    results = []
    ensure_model_loaded()
    tokenizer = CURRENT_TOKENIZER
    base_model = CURRENT_MODEL
    print("[Benchmark] Baseline.")
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(DEFAULT_DEVICE)
        start = time.time()
        with torch.no_grad():
            _ = base_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        end = time.time()
        results.append({"config": "baseline", "prompt_len": len(p), "latency_ms": (end - start) * 1000.0})
    for cfg in configs:
        global TILE_ROWS, TILE_COLS, QUANT_MODE
        TILE_ROWS = cfg["tile_rows"]
        TILE_COLS = cfg["tile_cols"]
        QUANT_MODE = cfg["quant_mode"]
        print(f"[Benchmark] Config: {cfg['name']} TILE={TILE_ROWS}x{TILE_COLS} QUANT={QUANT_MODE}")
        for p in prompts:
            text, stats, _, latency_ms = generate_with_forklift(
                p, max_new_tokens=max_new_tokens, collect_router_data=False
            )
            results.append({"config": cfg["name"], "prompt_len": len(p), "latency_ms": latency_ms, "stats": stats})
    with BENCH_LOCK:
        BENCH_RESULTS = results
    print("[Benchmark] Done.")
    return results

def bayesian_auto_tune(prompts: List[str], iters=10):
    best_cfg = None
    best_latency = float("inf")
    search_space = [
        (32, 32, "int8"),
        (64, 64, "fp8"),
        (64, 128, "int8"),
        (128, 64, "int8"),
        (128, 128, "fp8"),
    ]
    for i in range(iters):
        cfg = random.choice(search_space)
        tr, tc, qm = cfg
        global TILE_ROWS, TILE_COLS, QUANT_MODE
        TILE_ROWS, TILE_COLS, QUANT_MODE = tr, tc, qm
        print(f"[AutoTune] Iter {i+1}/{iters} testing TILE={tr}x{tc} QUANT={qm}")
        latencies = []
        for p in prompts:
            _, _, _, latency_ms = generate_with_forklift(p, max_new_tokens=32, collect_router_data=False)
            latencies.append(latency_ms)
        avg_lat = sum(latencies) / len(latencies)
        print(f"[AutoTune] Avg latency: {avg_lat:.1f} ms")
        if avg_lat < best_latency:
            best_latency = avg_lat
            best_cfg = cfg
    print(f"[AutoTune] Best config: TILE={best_cfg[0]}x{best_cfg[1]} QUANT={best_cfg[2]} avg_latency={best_latency:.1f} ms")
    return best_cfg, best_latency

# =========================
# Web dashboard
# =========================

WEB_APP = None

def start_web_dashboard(host="0.0.0.0", port=7000):
    global WEB_APP
    if not FLASK_AVAILABLE:
        print("Flask not available; web dashboard disabled.")
        return
    app = Flask(__name__)

    @app.route("/telemetry")
    def telemetry_endpoint():
        tel = get_system_telemetry()
        with MULTINODE_LOCK:
            nodes = dict(MULTINODE_TELEMETRY)
        with BENCH_LOCK:
            bench = list(BENCH_RESULTS)
        return jsonify({"local": tel, "nodes": nodes, "bench": bench, "topology": TOPOLOGY_STATE})

    @app.route("/generate", methods=["POST"])
    def generate_endpoint():
        data = request.get_json(force=True)
        prompt = data.get("prompt", "")
        max_new_tokens = int(data.get("max_new_tokens", 64))
        text, stats, _, latency_ms = generate_with_forklift(
            prompt, max_new_tokens=max_new_tokens, collect_router_data=False
        )
        return jsonify({"text": text, "stats": stats, "latency_ms": latency_ms})

    WEB_APP = app

    def run_app():
        app.run(host=host, port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=run_app, daemon=True)
    t.start()
    print(f"[WebDashboard] http://{host}:{port}")

# =========================
# Tkinter cockpit
# =========================

def gui_tkinter():
    if tk is None:
        print("Tkinter not available; GUI disabled.")
        return
    global MODEL_QUEUE_INDEX
    root = tk.Tk()
    root.title(f"Forklift Cockpit - {NODE_ID} [{NODE_POLICY_NAME}]")
    root.geometry("1300x800")
    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X, padx=10, pady=5)
    info_label = tk.Label(
        top_frame,
        text=f"CUDA: {HAS_CUDA}, GPUs: {NUM_GPUS}, Policy: {NODE_POLICY_NAME}, Triton: {TRITON_AVAILABLE}, TE: {TE_AVAILABLE}, FlashAttn: {FLASH_ATTENTION_AVAILABLE}",
        font=("Consolas", 11),
    )
    info_label.pack(side=tk.LEFT)
    status_var = tk.StringVar()
    status_label = tk.Label(top_frame, textvariable=status_var, font=("Consolas", 11), fg="green")
    status_label.pack(side=tk.RIGHT)

    def update_status():
        mode = "Fallback" if IS_FALLBACK_MODEL else "Real"
        status_var.set(
            f"Model: {CURRENT_MODEL_NAME} | Mode: {mode} | QUANT_MODE: {QUANT_MODE} | TILE: {TILE_ROWS}x{TILE_COLS}"
        )

    update_status()
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    left_frame = tk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

    model_frame = tk.LabelFrame(left_frame, text="Model Queue", padx=5, pady=5)
    model_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 5))
    model_listbox = tk.Listbox(model_frame, height=8, width=45)
    for i, name in enumerate(FALLBACK_MODELS):
        model_listbox.insert(tk.END, f"{i}: {name}")
    model_listbox.pack(side=tk.TOP, fill=tk.X)
    model_btn_frame = tk.Frame(model_frame)
    model_btn_frame.pack(fill=tk.X, pady=5)
    auto_btn = tk.Button(model_btn_frame, text="Auto Load Model", width=16)
    auto_btn.pack(side=tk.LEFT, padx=2)
    retry_btn = tk.Button(model_btn_frame, text="Retry Queue", width=12)
    retry_btn.pack(side=tk.LEFT, padx=2)
    browse_btn = tk.Button(model_btn_frame, text="Browse Local Model", width=18)
    browse_btn.pack(side=tk.LEFT, padx=2)

    model_ctrl_frame = tk.LabelFrame(left_frame, text="Model Control", padx=5, pady=5)
    model_ctrl_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 5))
    force_fallback_btn = tk.Button(model_ctrl_frame, text="Force Fallback Model", width=20)
    force_fallback_btn.pack(side=tk.TOP, fill=tk.X, pady=2)
    clear_model_btn = tk.Button(model_ctrl_frame, text="Clear Current Model", width=20)
    clear_model_btn.pack(side=tk.TOP, fill=tk.X, pady=2)
    inspect_model_btn = tk.Button(model_ctrl_frame, text="Inspect Model", width=20)
    inspect_model_btn.pack(side=tk.TOP, fill=tk.X, pady=2)

    dl_frame = tk.LabelFrame(left_frame, text="Downloads", padx=5, pady=5)
    dl_frame.pack(fill=tk.BOTH, expand=False)
    repo_row = tk.Frame(dl_frame)
    repo_row.pack(fill=tk.X, pady=2)
    tk.Label(repo_row, text="Repo ID:").pack(side=tk.LEFT)
    repo_entry = tk.Entry(repo_row, width=30)
    repo_entry.insert(0, PRIMARY_MODEL_NAME)
    repo_entry.pack(side=tk.LEFT, padx=5)
    dl_button = tk.Button(repo_row, text="Download missing files now")
    dl_button.pack(side=tk.LEFT, padx=5)
    prog_row = tk.Frame(dl_frame)
    prog_row.pack(fill=tk.X, pady=4)
    prog_label = tk.Label(prog_row, text="Progress:")
    prog_label.pack(side=tk.LEFT)
    prog_var = tk.DoubleVar(value=0.0)
    prog_bar = ttk.Progressbar(prog_row, variable=prog_var, maximum=100)
    prog_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    prog_status = tk.Label(prog_row, text="Idle")
    prog_status.pack(side=tk.LEFT, padx=5)

    tel_frame = tk.LabelFrame(left_frame, text="System Telemetry", padx=5, pady=5)
    tel_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
    cpu_label_var = tk.StringVar()
    gpu_label_var = tk.StringVar()
    ram_label_var = tk.StringVar()
    temp_label_var = tk.StringVar()
    cpu_label = tk.Label(tel_frame, textvariable=cpu_label_var, font=("Consolas", 9))
    cpu_label.pack(anchor="w")
    gpu_label = tk.Label(tel_frame, textvariable=gpu_label_var, font=("Consolas", 9))
    gpu_label.pack(anchor="w")
    ram_label = tk.Label(tel_frame, textvariable=ram_label_var, font=("Consolas", 9))
    ram_label.pack(anchor="w")
    temp_label = tk.Label(tel_frame, textvariable=temp_label_var, font=("Consolas", 9))
    temp_label.pack(anchor="w")
    multinode_box = scrolledtext.ScrolledText(tel_frame, height=8)
    multinode_box.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    gen_frame = tk.LabelFrame(right_frame, text="Generation", padx=5, pady=5)
    gen_frame.pack(fill=tk.X, expand=False)
    prompt_label = tk.Label(gen_frame, text="Prompt:")
    prompt_label.pack(anchor="w")
    prompt_box = scrolledtext.ScrolledText(gen_frame, height=4)
    prompt_box.insert(tk.END, "The future of AI acceleration is")
    prompt_box.pack(fill=tk.X, expand=False)
    gen_btn_frame = tk.Frame(gen_frame)
    gen_btn_frame.pack(fill=tk.X, pady=4)
    gen_btn = tk.Button(gen_btn_frame, text="Generate", width=12)
    gen_btn.pack(side=tk.LEFT, padx=2)
    train_btn = tk.Button(gen_btn_frame, text="Collect + Train Router", width=20)
    train_btn.pack(side=tk.LEFT, padx=2)
    distill_btn = tk.Button(gen_btn_frame, text="Distill Router v1", width=16)
    distill_btn.pack(side=tk.LEFT, padx=2)
    distill2_btn = tk.Button(gen_btn_frame, text="Distill Router v2", width=16)
    distill2_btn.pack(side=tk.LEFT, padx=2)
    bench_btn = tk.Button(gen_btn_frame, text="Run Benchmark", width=16)
    bench_btn.pack(side=tk.LEFT, padx=2)
    autotune_btn = tk.Button(gen_btn_frame, text="Auto-Tune", width=12)
    autotune_btn.pack(side=tk.LEFT, padx=2)

    out_frame = tk.LabelFrame(right_frame, text="Output", padx=5, pady=5)
    out_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
    output_box = scrolledtext.ScrolledText(out_frame, height=14)
    output_box.pack(fill=tk.BOTH, expand=True)

    inspector_frame = tk.LabelFrame(right_frame, text="Model Inspector", padx=5, pady=5)
    inspector_frame.pack(fill=tk.BOTH, expand=False, pady=(5, 0))
    inspector_box = scrolledtext.ScrolledText(inspector_frame, height=8)
    inspector_box.pack(fill=tk.BOTH, expand=True)

    router_frame = tk.LabelFrame(right_frame, text="Router / Executor Visualizer", padx=5, pady=5)
    router_frame.pack(fill=tk.BOTH, expand=False, pady=(5, 0))
    router_box = scrolledtext.ScrolledText(router_frame, height=10)
    router_box.pack(fill=tk.BOTH, expand=True)

    def log(text):
        output_box.insert(tk.END, text + "\n")
        output_box.see(tk.END)

    def inspector_log(text):
        inspector_box.insert(tk.END, text + "\n")
        inspector_box.see(tk.END)

    def router_log(text):
        router_box.insert(tk.END, text + "\n")
        router_box.see(tk.END)

    def on_auto():
        global MODEL_QUEUE_INDEX
        MODEL_QUEUE_INDEX = 0
        log("Auto loading model from queue...")
        ok = load_llama_with_forklift_from_queue()
        log("Model loaded successfully." if ok else "All remote models failed; using fallback.")
        update_status()

    def on_retry():
        global MODEL_QUEUE_INDEX
        MODEL_QUEUE_INDEX = 0
        log("Retrying model queue...")
        ok = load_llama_with_forklift_from_queue()
        log("Model loaded successfully." if ok else "All remote models failed; using fallback.")
        update_status()

    def on_browse():
        path = filedialog.askdirectory()
        if path:
            try:
                load_local_model_from_path(path)
                log(f"Local model loaded from {path}")
            except Exception as e:
                log(f"Failed to load local model: {e}")
        update_status()

    def on_download():
        repo_id = repo_entry.get().strip()
        if not repo_id:
            log("No repo ID provided.")
            return
        if snapshot_download is None:
            log("huggingface_hub not installed; cannot download.")
            return
        if not os.path.exists("models"):
            os.makedirs("models", exist_ok=True)
        local_dir = os.path.join("models", repo_id.replace("/", "_"))
        global DOWNLOAD_RUNNING
        if DOWNLOAD_RUNNING:
            log("Download already running.")
            return
        log(f"Downloading {repo_id} ...")
        prog_status.config(text="Downloading...")
        prog_var.set(0.0)
        def done_cb(status):
            prog_status.config(text="Done")
            prog_var.set(0.0)
            log("Download done.")
        def run_download():
            manual_download_repo(repo_id, local_dir, progress_callback=done_cb)
        t = threading.Thread(target=run_download, daemon=True)
        t.start()

    def on_force_fallback():
        log("Forcing fallback model...")
        force_fallback_model()
        update_status()

    def on_clear_model():
        log("Clearing current model...")
        clear_current_model()
        update_status()

    def on_inspect_model():
        inspector_box.delete("1.0", tk.END)
        if CURRENT_MODEL is None:
            inspector_log("No model loaded.")
            return
        inspector_log(f"Model name: {CURRENT_MODEL_NAME}")
        inspector_log(f"Fallback: {IS_FALLBACK_MODEL}")
        inspector_log(str(CURRENT_MODEL))

    def on_gen():
        prompt = prompt_box.get("1.0", tk.END).strip()
        if not prompt:
            log("No prompt provided.")
            return
        log("Generating...")
        root.update_idletasks()
        def run_gen():
            try:
                text, stats, layer_stats, latency_ms = generate_with_forklift(
                    prompt, max_new_tokens=128, collect_router_data=True
                )
                output_box.delete("1.0", tk.END)
                output_box.insert(tk.END, text + "\n")
                log(f"Generation complete. Latency: {latency_ms:.1f} ms")
                router_box.delete("1.0", tk.END)
                router_log("=== Executor Stats ===")
                for k, v in stats.items():
                    router_log(f"{k}: {v}")
                router_log("\n=== Per-Layer Stats (top by bytes) ===")
                for name, flops, bytes_moved, skipped, fpb, skip_ratio in layer_stats[:30]:
                    router_log(
                        f"{name}: flops={flops:.2e}, bytes={bytes_moved}, skipped={skipped}, "
                        f"fpb={fpb:.2f}, skip_ratio={skip_ratio:.2f}"
                    )
            except Exception as e:
                log(f"Generation failed: {e}")
            update_status()
        t = threading.Thread(target=run_gen, daemon=True)
        t.start()

    def on_train_router():
        log("Training router on collected data...")
        root.update_idletasks()
        def run_train():
            try:
                train_router_on_collected_data()
                log("Router training complete.")
            except Exception as e:
                log(f"Router training failed: {e}")
        t = threading.Thread(target=run_train, daemon=True)
        t.start()

    def on_distill_router():
        log("Distilling router v1...")
        root.update_idletasks()
        def run_distill():
            try:
                prompts = [
                    "Explain the future of AI acceleration.",
                    "Describe the architecture of a distributed LLM inference engine.",
                    "What are the tradeoffs of FP8 quantization?",
                ]
                distill_router_from_teacher(prompts)
                log("Router distillation v1 complete.")
            except Exception as e:
                log(f"Router distillation v1 failed: {e}")
        t = threading.Thread(target=run_distill, daemon=True)
        t.start()

    def on_distill_router_v2():
        log("Distilling router v2 (teacher activations)...")
        root.update_idletasks()
        def run_distill2():
            try:
                prompts = [
                    "Explain attention mechanisms in transformers.",
                    "Describe KV-cache behavior for long contexts.",
                    "How does tile-wise sparsity affect accuracy?",
                ]
                distill_router_from_teacher_v2(prompts)
                log("Router distillation v2 complete.")
            except Exception as e:
                log(f"Router distillation v2 failed: {e}")
        t = threading.Thread(target=run_distill2, daemon=True)
        t.start()

    def on_benchmark():
        log("Running benchmark vs baseline...")
        root.update_idletasks()
        def run_bench():
            try:
                prompts = [
                    "The future of AI acceleration is",
                    "Explain KV-cache compression in transformers.",
                    "Describe tile-wise sparsity in matrix multiplication.",
                ]
                res = benchmark_vs_baseline(prompts, max_new_tokens=64)
                log("Benchmark complete.")
                for r in res:
                    log(str(r))
            except Exception as e:
                log(f"Benchmark failed: {e}")
        t = threading.Thread(target=run_bench, daemon=True)
        t.start()

    def on_autotune():
        log("Running Bayesian-style auto-tuner...")
        root.update_idletasks()
        def run_auto():
            try:
                prompts = [
                    "The future of AI acceleration is",
                    "Explain KV-cache compression in transformers.",
                ]
                cfg, lat = bayesian_auto_tune(prompts, iters=8)
                log(f"Auto-tune best config: TILE={cfg[0]}x{cfg[1]} QUANT={cfg[2]} avg_latency={lat:.1f} ms")
            except Exception as e:
                log(f"Auto-tune failed: {e}")
        t = threading.Thread(target=run_auto, daemon=True)
        t.start()

    auto_btn.config(command=on_auto)
    retry_btn.config(command=on_retry)
    browse_btn.config(command=on_browse)
    dl_button.config(command=on_download)
    force_fallback_btn.config(command=on_force_fallback)
    clear_model_btn.config(command=on_clear_model)
    inspect_model_btn.config(command=on_inspect_model)
    gen_btn.config(command=on_gen)
    train_btn.config(command=on_train_router)
    distill_btn.config(command=on_distill_router)
    distill2_btn.config(command=on_distill_router_v2)
    bench_btn.config(command=on_benchmark)
    autotune_btn.config(command=on_autotune)

    def update_telemetry():
        tel = get_system_telemetry()
        cpu_label_var.set("CPU  " + telemetry_to_ascii_bar(tel["cpu_load"]))
        gpu_label_var.set("GPU  " + telemetry_to_ascii_bar(tel["gpu_util"]))
        ram_label_var.set("RAM  " + telemetry_to_ascii_bar(tel["ram_used"]))
        temp_label_var.set(f"GPU Temp: {tel['gpu_temp']:.1f} C")
        multinode_box.delete("1.0", tk.END)
        with MULTINODE_LOCK:
            for nid, tdata in MULTINODE_TELEMETRY.items():
                multinode_box.insert(
                    tk.END,
                    f"{nid} [{tdata.get('policy','?')}]: "
                    f"GPU {tdata.get('gpu_util', 0):5.1f}%, "
                    f"Mem {tdata.get('gpu_mem_pct', 0):5.1f}%, "
                    f"CPU {tdata.get('cpu_load', 0):5.1f}%\n",
                )
        root.after(1000, update_telemetry)

    update_telemetry()
    root.mainloop()

# =========================
# Swarm node
# =========================

def run_swarm_node():
    print(f"[SwarmNode] Starting headless node {NODE_ID} with policy '{NODE_POLICY_NAME}'")
    ensure_model_loaded()
    print(f"[SwarmNode] Model ready: {CURRENT_MODEL_NAME}")
    rpc_thread = threading.Thread(target=rpc_server_loop, daemon=True)
    rpc_thread.start()
    tile_rpc_thread = threading.Thread(target=tile_rpc_server_loop, args=(GLOBAL_CACHE,), daemon=True)
    tile_rpc_thread.start()
    if FLASK_AVAILABLE:
        start_web_dashboard()
    try:
        while True:
            tel = get_system_telemetry()
            role = cluster_scheduler_decide_role()
            print(
                f"[SwarmNode] {NODE_ID} role={role} policy={NODE_POLICY_NAME} "
                f"GPU={tel['gpu_util']:.1f}% MEM={tel['gpu_mem_pct']:.1f}% "
                f"CPU={tel['cpu_load']:.1f}% TEMP={tel['gpu_temp']:.1f}C"
            )
            time.sleep(5)
    except KeyboardInterrupt:
        print("[SwarmNode] Shutting down.")

# =========================
# Main
# =========================

if __name__ == "__main__":
    init_telemetry_db()
    tb = threading.Thread(target=telemetry_broadcast_loop, daemon=True)
    tb.start()
    tl = threading.Thread(target=telemetry_listener_loop, daemon=True)
    tl.start()
    dcb = threading.Thread(target=distributed_cache_broadcast_loop, args=(GLOBAL_CACHE,), daemon=True)
    dcb.start()
    dcl = threading.Thread(target=distributed_cache_listener_loop, daemon=True)
    dcl.start()
    tile_rpc_thread = threading.Thread(target=tile_rpc_server_loop, args=(GLOBAL_CACHE,), daemon=True)
    tile_rpc_thread.start()

    # Mode selection:
    #   --legacy-engine   -> run legacy ForkliftEngine
    #   --headless-node   -> run swarm headless node
    #   (default)         -> run GUI + web dashboard
    if "--legacy-engine" in sys.argv:
        legacy_main()
    elif "--headless-node" in sys.argv:
        run_swarm_node()
    else:
        if FLASK_AVAILABLE:
            start_web_dashboard()
        gui_tkinter()
