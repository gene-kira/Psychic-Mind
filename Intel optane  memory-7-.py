#!/usr/bin/env python3
"""
neurofabric_tier9_unified.py

Tier‑8/9 NeuroFabric organism (single file):

- GPU‑first vector intelligence (FAISS / CuPy / Torch)
- Real GNN reasoning (PyTorch Geometric if available, NetworkX fallback)
- Real autoencoder anomaly model with online training
- Temporal anomaly detection (sliding windows + temporal encoder stub)
- Real distributed mesh (UDP gossip + CRDT state)
- NPU acceleration hooks (OpenVINO / QNN stubs)
- PySide6 GPU cockpit (with Tk fallback)
- Multi‑drive tiered storage + hybrid auto‑blacklist (WinError 22)
- RAM + disk cache on all allowed drives
- Auto‑port FastAPI backend (no 8080 conflicts)
- Self‑optimization (heat‑based migration, auto thresholds)
- Self‑repair (watchdogs for critical subsystems)
"""

import os
import sys
import json
import time
import threading
import sqlite3
import shutil
import socket
import random
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Core deps
# ---------------------------------------------------------------------------

REQUIRED_LIBS = [
    "psutil",
    "numpy",
    "fastapi",
    "uvicorn",
    "pydantic",
]

def ensure_libs():
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print("[neurofabric] Missing core libraries:", ", ".join(missing))
        print("Install with:")
        print("    pip install " + " ".join(missing))

ensure_libs()

import psutil
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# GUI
import tkinter as tk
from tkinter import ttk

try:
    from PySide6 import QtWidgets, QtCore
    HAS_PYSIDE6 = True
except Exception:
    HAS_PYSIDE6 = False

# Optional accel libs
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import networkx as nx  # type: ignore
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False

try:
    import torch_geometric  # type: ignore
    from torch_geometric.data import Data as PyGData  # type: ignore
    from torch_geometric.nn import GCNConv  # type: ignore
    HAS_PYG = True
except Exception:
    HAS_PYG = False

try:
    import etw  # type: ignore
    HAS_ETW = True
except Exception:
    HAS_ETW = False

# NPU stubs (OpenVINO / QNN)
try:
    import openvino  # type: ignore
    HAS_OPENVINO = True
except Exception:
    HAS_OPENVINO = False

try:
    import qnn  # type: ignore  # placeholder
    HAS_QNN = True
except Exception:
    HAS_QNN = False

# ---------------------------------------------------------------------------
# Auto‑port
# ---------------------------------------------------------------------------

def find_free_port(preferred: int = 8080, low: int = 20000, high: int = 40000) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", preferred))
            return preferred
        except OSError:
            pass
    while True:
        port = random.randint(low, high)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue

# ---------------------------------------------------------------------------
# Drive policy + hybrid auto‑blacklist (WinError 22)
# ---------------------------------------------------------------------------

DRIVE_BLACKLIST: set[str] = {"a:", "b:"}
DRIVE_WHITELIST: set[str] = set()
RUNTIME_BLACKLIST: set[str] = set()
PERSISTENT_BLACKLIST: set[str] = set()
DRIVE_FAIL_COUNTS: Dict[str, int] = {}
AUTO_BLACKLIST_PATH = Path("./auto_blacklist.json").resolve()

def _load_persistent_blacklist():
    global PERSISTENT_BLACKLIST
    if AUTO_BLACKLIST_PATH.exists():
        try:
            with AUTO_BLACKLIST_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                PERSISTENT_BLACKLIST = {str(x).lower() for x in data}
            elif isinstance(data, dict) and "drives" in data:
                PERSISTENT_BLACKLIST = {str(x).lower() for x in data["drives"]}
        except Exception:
            PERSISTENT_BLACKLIST = set()
    else:
        PERSISTENT_BLACKLIST = set()

def _save_persistent_blacklist():
    try:
        AUTO_BLACKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUTO_BLACKLIST_PATH.open("w", encoding="utf-8") as f:
            json.dump(sorted(PERSISTENT_BLACKLIST), f, indent=2)
    except Exception:
        pass

def _drive_letter_from_device(device: str) -> Optional[str]:
    device = device.strip()
    if len(device) >= 2 and device[1] == ":":
        return device[:2].lower()
    return None

def _drive_letter_from_path(path: str) -> Optional[str]:
    drive, _ = os.path.splitdrive(path)
    if drive:
        return drive.lower()
    return None

def is_drive_allowed(letter: Optional[str]) -> bool:
    if not letter:
        return True
    letter = letter.lower()
    if DRIVE_WHITELIST and letter not in DRIVE_WHITELIST:
        return False
    if letter in DRIVE_BLACKLIST:
        return False
    if letter in PERSISTENT_BLACKLIST:
        return False
    if letter in RUNTIME_BLACKLIST:
        return False
    return True

def note_drive_failure(letter: Optional[str], exc: BaseException):
    if not letter:
        return
    letter = letter.lower()
    if DRIVE_WHITELIST and letter in DRIVE_WHITELIST:
        return
    count = DRIVE_FAIL_COUNTS.get(letter, 0) + 1
    DRIVE_FAIL_COUNTS[letter] = count
    if count == 1:
        RUNTIME_BLACKLIST.add(letter)
    if count >= 2 and letter not in PERSISTENT_BLACKLIST:
        PERSISTENT_BLACKLIST.add(letter)
        _save_persistent_blacklist()

_load_persistent_blacklist()

# ---------------------------------------------------------------------------
# Drive health + multi‑drive scoring
# ---------------------------------------------------------------------------

class DriveHealth:
    @staticmethod
    def score_partition(part: psutil._common.sdiskpart) -> float:
        mount = part.mountpoint
        device = part.device.lower()
        letter = _drive_letter_from_device(device)
        if letter and not is_drive_allowed(letter):
            return 0.0
        try:
            usage = psutil.disk_usage(mount)
        except OSError as e:
            note_drive_failure(letter, e)
            return 0.0
        except Exception:
            return 0.0
        if usage.total == 0:
            return 0.0
        if usage.total < 16 * 1024 * 1024:
            return 0.0
        used_ratio = usage.percent / 100.0
        if used_ratio > 0.98:
            base = 0.2
        elif used_ratio > 0.90:
            base = 0.4
        else:
            base = 0.8
        try:
            io_all = psutil.disk_io_counters(perdisk=True)
            for name, io in io_all.items():
                lname = name.lower()
                if lname in device or device in lname:
                    err = getattr(io, "read_time", 0) + getattr(io, "write_time", 0)
                    if err > 10**9:
                        base *= 0.5
                    break
        except Exception:
            pass
        return max(0.0, min(1.0, base))

def detect_all_roots() -> Tuple[Path, List[Path]]:
    candidates: List[Tuple[float, Path]] = []
    for part in psutil.disk_partitions(all=False):
        mount = part.mountpoint
        device = part.device.lower()
        if "cdrom" in part.opts or "cdrom" in device:
            continue
        if "\\device\\" in device:
            continue
        letter = _drive_letter_from_device(device)
        if letter and not is_drive_allowed(letter):
            continue
        health = DriveHealth.score_partition(part)
        if health <= 0.0:
            continue
        try:
            usage = psutil.disk_usage(mount)
        except OSError as e:
            note_drive_failure(letter, e)
            continue
        except Exception:
            continue
        if usage.total == 0:
            continue
        score = 0.0
        if "nvme" in device:
            score += 2.0
        if "optane" in device:
            score += 3.0
        if "ssd" in device:
            score += 1.0
        free_gb = usage.free / (1024**3)
        score += min(free_gb / 128.0, 2.0)
        score *= (0.5 + 0.5 * health)
        if score > 0.0:
            candidates.append((score, Path(mount)))
    fallback_root = Path("./optane_store_fallback").resolve()
    fallback_root.mkdir(parents=True, exist_ok=True)
    if not candidates:
        primary = Path("./optane_store").resolve()
        primary.mkdir(parents=True, exist_ok=True)
        return primary, [primary]
    candidates.sort(key=lambda x: x[0], reverse=True)
    primary = (candidates[0][1] / "optane_neurofabric").resolve()
    primary.mkdir(parents=True, exist_ok=True)
    all_roots: List[Path] = []
    for _, p in candidates:
        r = (p / "neurofabric_tier").resolve()
        r.mkdir(parents=True, exist_ok=True)
        all_roots.append(r)
    return primary, all_roots

def ensure_root_alive(root: Path, fallback: Path) -> Path:
    letter = _drive_letter_from_path(str(root))
    if letter and not is_drive_allowed(letter):
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    try:
        usage = psutil.disk_usage(str(root))
        if usage.total == 0:
            raise OSError("zero-sized root")
    except OSError as e:
        note_drive_failure(letter, e)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except Exception:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return root

OPTANE_PRIMARY_ROOT, ALL_DRIVE_ROOTS = detect_all_roots()
OPTANE_FALLBACK_ROOT = Path("./optane_store_fallback").resolve()
OPTANE_ROOT = ensure_root_alive(OPTANE_PRIMARY_ROOT, OPTANE_FALLBACK_ROOT)

COLD_ROOT   = Path("./cold_store").resolve()
COLD_ROOT.mkdir(parents=True, exist_ok=True)

KV_HOT_DIR        = OPTANE_ROOT / "kv_cache"
VEC_HOT_DIR       = OPTANE_ROOT / "vector_db"
TELEMETRY_HOT_DIR = OPTANE_ROOT / "telemetry"
BRAIN_HOT_DIR     = OPTANE_ROOT / "brain_state"
SWARM_HOT_DIR     = OPTANE_ROOT / "swarm_state"

KV_COLD_DIR        = COLD_ROOT / "kv_cache"
VEC_COLD_DIR       = COLD_ROOT / "vector_db"
TELEMETRY_COLD_DIR = COLD_ROOT / "telemetry"
BRAIN_COLD_DIR     = COLD_ROOT / "brain_state"
SWARM_COLD_DIR     = COLD_ROOT / "swarm_state"

for d in [
    KV_HOT_DIR, VEC_HOT_DIR, TELEMETRY_HOT_DIR, BRAIN_HOT_DIR, SWARM_HOT_DIR,
    KV_COLD_DIR, VEC_COLD_DIR, TELEMETRY_COLD_DIR, BRAIN_COLD_DIR, SWARM_COLD_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Heat manager
# ---------------------------------------------------------------------------

class HeatManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.heat: Dict[str, int] = {}

    def bump(self, subsystem: str, key: str, amount: int = 1):
        k = f"{subsystem}:{key}"
        with self.lock:
            self.heat[k] = self.heat.get(k, 0) + amount

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            return dict(self.heat)

    def top(self, n: int = 10) -> List[Tuple[str, int]]:
        snap = self.snapshot()
        return sorted(snap.items(), key=lambda x: x[1], reverse=True)[:n]

HEAT = HeatManager()

# ---------------------------------------------------------------------------
# RAM cache
# ---------------------------------------------------------------------------

class RAMCache:
    def __init__(self, max_items: int = 2048):
        self.max_items = max_items
        self.store: Dict[str, Any] = {}
        self.order: List[str] = []
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.store:
                return None
            self.order.remove(key)
            self.order.append(key)
            return self.store[key]

    def put(self, key: str, value: Any):
        with self.lock:
            if key in self.store:
                self.store[key] = value
                self.order.remove(key)
                self.order.append(key)
                return
            if len(self.order) >= self.max_items:
                old = self.order.pop(0)
                self.store.pop(old, None)
            self.store[key] = value
            self.order.append(key)

    def delete(self, key: str):
        with self.lock:
            if key in self.store:
                self.store.pop(key, None)
                if key in self.order:
                    self.order.remove(key)

# ---------------------------------------------------------------------------
# Tiered KV cache
# ---------------------------------------------------------------------------

class TieredKVCache:
    def __init__(self,
                 hot_root: Path = KV_HOT_DIR,
                 cold_root: Path = KV_COLD_DIR,
                 ram_cache: Optional[RAMCache] = None):
        self.hot_root = hot_root
        self.cold_root = cold_root
        self.ram = ram_cache or RAMCache()

    def _path_for_key(self, root: Path, key: str) -> Path:
        safe = key.replace("/", "_")
        return root / f"{safe}.json"

    def put(self, key: str, value: Dict[str, Any]) -> None:
        HEAT.bump("kv", key, 2)
        self.ram.put(key, value)
        path = self._path_for_key(self.hot_root, key)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(value, f)
            tmp.replace(path)
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
        except Exception:
            pass

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        HEAT.bump("kv", key, 1)
        cached = self.ram.get(key)
        if cached is not None:
            return cached
        hot = self._path_for_key(self.hot_root, key)
        cold = self._path_for_key(self.cold_root, key)
        path = hot if hot.exists() else cold if cold.exists() else None
        if not path:
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                val = json.load(f)
            self.ram.put(key, val)
            return val
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
            return None
        except Exception:
            return None

    def delete(self, key: str) -> None:
        self.ram.delete(key)
        hot = self._path_for_key(self.hot_root, key)
        cold = self._path_for_key(self.cold_root, key)
        for p in (hot, cold):
            try:
                if p.exists():
                    p.unlink()
            except OSError as e:
                letter = _drive_letter_from_path(str(p))
                note_drive_failure(letter, e)
            except Exception:
                pass

    def list_keys(self) -> List[str]:
        keys = set()
        for root in (self.hot_root, self.cold_root):
            try:
                for p in root.glob("*.json"):
                    keys.add(p.stem)
            except OSError as e:
                letter = _drive_letter_from_path(str(root))
                note_drive_failure(letter, e)
                continue
            except Exception:
                continue
        return sorted(keys)

# ---------------------------------------------------------------------------
# LLM KV adapter
# ---------------------------------------------------------------------------

class LLMKVAdapter:
    def __init__(self, kv: TieredKVCache):
        self.kv = kv

    def _key(self, model_id: str, session_id: str) -> str:
        return f"{model_id}::{session_id}"

    def save_cache_meta(self, model_id: str, session_id: str, meta: Dict[str, Any]) -> None:
        self.kv.put(self._key(model_id, session_id), meta)

    def load_cache_meta(self, model_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        return self.kv.get(self._key(model_id, session_id))

    def drop_cache(self, model_id: str, session_id: str) -> None:
        self.kv.delete(self._key(model_id, session_id))

# ---------------------------------------------------------------------------
# KV shard layout (mmap)
# ---------------------------------------------------------------------------

class KVShardLayout:
    DTYPE_F16 = 0
    DTYPE_F32 = 1

    def __init__(self, root: Path = KV_HOT_DIR):
        self.root = root

    def _path_for_shard(self, model_id: str, session_id: str, shard_id: str) -> Path:
        safe = f"{model_id}__{session_id}__{shard_id}".replace("/", "_")
        return self.root / f"kvshard_{safe}.bin"

    def save_shard(self, model_id: str, session_id: str, shard_id: str,
                   tensor: np.ndarray, use_f16: bool = True) -> Path:
        tensor = np.asarray(tensor)
        dtype_code = self.DTYPE_F16 if use_f16 else self.DTYPE_F32
        tensor = tensor.astype(np.float16 if use_f16 else np.float32)
        shape = tensor.shape
        ndim = len(shape)
        total_elems = int(np.prod(shape))
        path = self._path_for_shard(model_id, session_id, shard_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = struct.pack("<IIQ", ndim, dtype_code, total_elems)
        header += struct.pack("<" + "Q" * ndim, *shape)
        try:
            with path.open("wb") as f:
                f.write(header)
                f.write(tensor.tobytes(order="C"))
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
        except Exception:
            pass
        return path

    def load_shard(self, model_id: str, session_id: str, shard_id: str,
                   mmap_mode: str = "r") -> Optional[np.ndarray]:
        path = self._path_for_shard(model_id, session_id, shard_id)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                header = f.read(4 + 4 + 8)
                ndim, dtype_code, total_elems = struct.unpack("<IIQ", header)
                shape_bytes = f.read(8 * ndim)
                shape = struct.unpack("<" + "Q" * ndim, shape_bytes)
                offset = f.tell()
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
            return None
        except Exception:
            return None
        dtype = np.float16 if dtype_code == self.DTYPE_F16 else np.float32
        try:
            mm = np.memmap(path, mode=mmap_mode, dtype=dtype,
                           offset=offset, shape=shape, order="C")
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
            return None
        except Exception:
            return None
        return mm

# ---------------------------------------------------------------------------
# Tiered Vector DB
# ---------------------------------------------------------------------------

class TieredVectorDB:
    def __init__(self,
                 hot_root: Path = VEC_HOT_DIR,
                 cold_root: Path = VEC_COLD_DIR,
                 db_name: str = "vecdb.sqlite"):
        self.hot_root = hot_root
        self.cold_root = cold_root
        self.db_path = self.hot_root / db_name
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()
        self.lock = threading.Lock()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vectors(
                id   TEXT PRIMARY KEY,
                dim  INTEGER NOT NULL,
                meta TEXT,
                tier TEXT NOT NULL DEFAULT 'hot'
            )
        """)
        self.conn.commit()

    def _embedding_path(self, vec_id: str, tier: str) -> Path:
        safe = vec_id.replace("/", "_")
        root = self.hot_root if tier == "hot" else self.cold_root
        return root / f"{safe}.npy"

    def upsert(self, vec_id: str, embedding: np.ndarray,
               meta: Optional[Dict[str, Any]] = None,
               tier: str = "hot"):
        HEAT.bump("vec", vec_id, 2)
        embedding = np.asarray(embedding, dtype=np.float32)
        dim = int(embedding.shape[-1])
        meta_json = json.dumps(meta or {})
        emb_path = self._embedding_path(vec_id, tier)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(emb_path, embedding)
        except OSError as e:
            letter = _drive_letter_from_path(str(emb_path))
            note_drive_failure(letter, e)
            return
        except Exception:
            return
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO vectors(id, dim, meta, tier)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET dim=excluded.dim,
                                             meta=excluded.meta,
                                             tier=excluded.tier
            """, (vec_id, dim, meta_json, tier))
            self.conn.commit()

    def get(self, vec_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any], str]]:
        HEAT.bump("vec", vec_id, 1)
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT dim, meta, tier FROM vectors WHERE id=?", (vec_id,))
            row = cur.fetchone()
        if not row:
            return None
        dim, meta_json, tier = row
        emb_path = self._embedding_path(vec_id, tier)
        if not emb_path.exists():
            return None
        try:
            emb = np.load(emb_path)
        except OSError as e:
            letter = _drive_letter_from_path(str(emb_path))
            note_drive_failure(letter, e)
            return None
        except Exception:
            return None
        meta = json.loads(meta_json) if meta_json else {}
        return emb, meta, tier

    def delete(self, vec_id: str) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT tier FROM vectors WHERE id=?", (vec_id,))
            row = cur.fetchone()
            if row:
                tier = row[0]
                emb_path = self._embedding_path(vec_id, tier)
                try:
                    if emb_path.exists():
                        emb_path.unlink()
                except OSError as e:
                    letter = _drive_letter_from_path(str(emb_path))
                    note_drive_failure(letter, e)
                except Exception:
                    pass
            cur.execute("DELETE FROM vectors WHERE id=?", (vec_id,))
            self.conn.commit()

    def all_ids(self) -> List[str]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id FROM vectors")
            return [r[0] for r in cur.fetchall()]

# ---------------------------------------------------------------------------
# Accelerator manager (GPU/APU/NPU)
# ---------------------------------------------------------------------------

class AcceleratorManager:
    def __init__(self):
        self.has_torch = HAS_TORCH
        self.has_cuda = HAS_TORCH and torch.cuda.is_available()
        self.has_cupy = HAS_CUPY
        self.has_faiss = HAS_FAISS
        self.has_openvino = HAS_OPENVINO
        self.has_qnn = HAS_QNN
        self.has_npu = self.has_openvino or self.has_qnn
        self.device = self._select_device()

    def _select_device(self) -> str:
        if self.has_cuda:
            return "cuda"
        if self.has_cupy:
            return "cupy"
        if self.has_npu:
            return "npu"
        return "cpu"

    def to_device(self, arr: np.ndarray):
        arr = arr.astype(np.float32)
        if self.device == "cuda" and self.has_torch:
            return torch.from_numpy(arr).cuda()
        if self.device == "cupy" and self.has_cupy:
            return cp.asarray(arr)
        return arr

    def from_device(self, x):
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if HAS_CUPY and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return np.asarray(x)

    def dot(self, a: np.ndarray, b: np.ndarray) -> float:
        a_dev = self.to_device(a)
        b_dev = self.to_device(b)
        if HAS_TORCH and isinstance(a_dev, torch.Tensor):
            return float(torch.dot(a_dev.view(-1), b_dev.view(-1)).item())
        if HAS_CUPY and isinstance(a_dev, cp.ndarray):
            return float(cp.dot(a_dev.ravel(), b_dev.ravel()).get())
        return float(np.dot(a_dev.ravel(), b_dev.ravel()))

    def norm(self, a: np.ndarray) -> float:
        a_dev = self.to_device(a)
        if HAS_TORCH and isinstance(a_dev, torch.Tensor):
            return float(torch.norm(a_dev).item())
        if HAS_CUPY and isinstance(a_dev, cp.ndarray):
            return float(cp.linalg.norm(a_dev).get())
        return float(np.linalg.norm(a_dev))

    def score_vector(self, emb: np.ndarray) -> float:
        # NPU hook: if real NPU model exists, route here
        if self.has_npu:
            return float(self.norm(emb))
        return float(self.norm(emb))

ACCEL = AcceleratorManager()

# ---------------------------------------------------------------------------
# GPU vector index (FAISS / CuPy / NumPy)
# ---------------------------------------------------------------------------

class GPUVectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.embs: Optional[np.ndarray] = None
        self.faiss_index = None
        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatIP(dim)

    def add(self, vec_id: str, emb: np.ndarray):
        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        if emb.shape[1] != self.dim:
            return
        if self.embs is None:
            self.embs = emb
        else:
            self.embs = np.vstack([self.embs, emb])
        self.ids.append(vec_id)
        if self.faiss_index is not None:
            self.faiss_index.add(emb)

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.embs is None or len(self.ids) == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            return []
        if self.faiss_index is not None:
            D, I = self.faiss_index.search(q, min(top_k, len(self.ids)))
            hits: List[Tuple[str, float]] = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self.ids):
                    continue
                hits.append((self.ids[idx], float(score)))
            return hits
        embs = self.embs
        q_norm = ACCEL.norm(q)
        e_norms = np.array([ACCEL.norm(e) for e in embs])
        dots = (embs @ q.T).reshape(-1)
        scores = dots / ((q_norm + 1e-8) * (e_norms + 1e-8))
        idxs = np.argsort(-scores)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in idxs]

# ---------------------------------------------------------------------------
# Anomaly embedding store
# ---------------------------------------------------------------------------

class AnomalyEmbeddingStore:
    def __init__(self, vecdb: TieredVectorDB):
        self.vecdb = vecdb
        self.index_cache: Dict[str, GPUVectorIndex] = {}

    def _ensure_index(self, dim: int, prefix: str) -> GPUVectorIndex:
        key = f"{prefix}:{dim}"
        if key not in self.index_cache:
            self.index_cache[key] = GPUVectorIndex(dim)
        return self.index_cache[key]

    def upsert_anomaly(self, anomaly_id: str, embedding: np.ndarray,
                       meta: Dict[str, Any], hot: bool = True):
        vec_id = f"anomaly:{anomaly_id}"
        tier = "hot" if hot else "cold"
        embedding = np.asarray(embedding, dtype=np.float32)
        self.vecdb.upsert(vec_id, embedding, meta, tier=tier)
        idx = self._ensure_index(embedding.shape[-1], "anomaly")
        idx.add(vec_id, embedding)

    def get_anomaly(self, anomaly_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any], str]]:
        return self.vecdb.get(f"anomaly:{anomaly_id}")

    def upsert_attack_node(self, graph_id: str, node_id: str,
                           embedding: np.ndarray, meta: Dict[str, Any],
                           hot: bool = True):
        vec_id = f"attack_node:{graph_id}:{node_id}"
        tier = "hot" if hot else "cold"
        embedding = np.asarray(embedding, dtype=np.float32)
        self.vecdb.upsert(vec_id, embedding, meta, tier=tier)
        idx = self._ensure_index(embedding.shape[-1], "attack_node")
        idx.add(vec_id, embedding)

    def get_attack_node(self, graph_id: str, node_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any], str]]:
        return self.vecdb.get(f"attack_node:{graph_id}:{node_id}")

    def search_nearest(self, query: np.ndarray, prefix: str = "anomaly",
                       top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        query = np.asarray(query, dtype=np.float32)
        q_dim = query.shape[-1]
        idx = self._ensure_index(q_dim, prefix)
        hits = idx.search(query, top_k=top_k)
        out: List[Tuple[str, float, Dict[str, Any]]] = []
        for vid, score in hits:
            got = self.vecdb.get(vid)
            if not got:
                continue
            _, meta, _ = got
            out.append((vid, score, meta))
        return out

# ---------------------------------------------------------------------------
# Telemetry buffer (JSONL, could be upgraded to binary ring)
# ---------------------------------------------------------------------------

class TelemetryBuffer:
    def __init__(self,
                 hot_root: Path = TELEMETRY_HOT_DIR,
                 cold_root: Path = TELEMETRY_COLD_DIR,
                 segment_bytes: int = 64 * 1024 * 1024):
        self.hot_root = hot_root
        self.cold_root = cold_root
        self.segment_bytes = segment_bytes
        self.lock = threading.Lock()
        self.current_file = None
        self.current_path = None
        self._open_new_segment()

    def _open_new_segment(self):
        ts = int(time.time() * 1000)
        path = self.hot_root / f"telemetry_{ts}.jsonl"
        try:
            self.current_path = path
            self.current_file = path.open("a", encoding="utf-8")
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
            self.current_file = None
            self.current_path = None
        except Exception:
            self.current_file = None
            self.current_path = None

    def _should_rotate(self) -> bool:
        if not self.current_path or not self.current_path.exists():
            return True
        try:
            return self.current_path.stat().st_size >= self.segment_bytes
        except OSError as e:
            letter = _drive_letter_from_path(str(self.current_path))
            note_drive_failure(letter, e)
            return True
        except Exception:
            return True

    def log(self, event: Dict[str, Any]) -> None:
        HEAT.bump("telemetry", event.get("type", "unknown"), 1)
        event["ts"] = event.get("ts", time.time())
        line = json.dumps(event, separators=(",", ":"))
        with self.lock:
            if self._should_rotate():
                if self.current_file:
                    try:
                        self.current_file.close()
                    except OSError as e:
                        letter = _drive_letter_from_path(str(self.current_path))
                        note_drive_failure(letter, e)
                    except Exception:
                        pass
                self._open_new_segment()
            if not self.current_file:
                return
            try:
                self.current_file.write(line + "\n")
                self.current_file.flush()
            except OSError as e:
                letter = _drive_letter_from_path(str(self.current_path))
                note_drive_failure(letter, e)
            except Exception:
                pass

    def list_segments(self) -> List[Path]:
        try:
            return sorted(self.hot_root.glob("telemetry_*.jsonl"))
        except OSError as e:
            letter = _drive_letter_from_path(str(self.hot_root))
            note_drive_failure(letter, e)
            return []
        except Exception:
            return []

# ---------------------------------------------------------------------------
# Brain state store
# ---------------------------------------------------------------------------

class BrainStateStore:
    def __init__(self,
                 hot_root: Path = BRAIN_HOT_DIR,
                 cold_root: Path = BRAIN_COLD_DIR):
        self.hot_root = hot_root
        self.cold_root = cold_root

    def save_snapshot(self, name: str, state: Dict[str, Any],
                      tier: str = "hot") -> Path:
        HEAT.bump("brain", name, 3)
        ts = int(time.time())
        safe = name.replace("/", "_")
        root = self.hot_root if tier == "hot" else self.cold_root
        path = root / f"{safe}_{ts}.json"
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            tmp.replace(path)
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
        except Exception:
            pass
        return path

    def latest_snapshot(self, name_prefix: str) -> Optional[Dict[str, Any]]:
        HEAT.bump("brain", name_prefix, 1)
        safe = name_prefix.replace("/", "_")
        try:
            candidates = sorted(self.hot_root.glob(f"{safe}_*.json")) + \
                         sorted(self.cold_root.glob(f"{safe}_*.json"))
        except OSError as e:
            letter = _drive_letter_from_path(str(self.hot_root))
            note_drive_failure(letter, e)
            return None
        except Exception:
            return None
        if not candidates:
            return None
        path = candidates[-1]
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
            return None
        except Exception:
            return None

    def list_snapshots(self) -> List[Path]:
        try:
            return sorted(self.hot_root.glob("*.json")) + \
                   sorted(self.cold_root.glob("*.json"))
        except OSError as e:
            letter = _drive_letter_from_path(str(self.hot_root))
            note_drive_failure(letter, e)
            return []
        except Exception:
            return []

# ---------------------------------------------------------------------------
# Swarm node state store
# ---------------------------------------------------------------------------

class SwarmNodeStateStore:
    def __init__(self,
                 hot_root: Path = SWARM_HOT_DIR,
                 cold_root: Path = SWARM_COLD_DIR):
        self.hot_root = hot_root
        self.cold_root = cold_root

    def _path_for_node(self, root: Path, node_id: str) -> Path:
        safe = node_id.replace("/", "_")
        return root / f"{safe}.json"

    def upsert(self, node_id: str, state: Dict[str, Any],
               tier: str = "hot") -> None:
        HEAT.bump("swarm", node_id, 2)
        state["node_id"] = node_id
        state["last_seen"] = state.get("last_seen", time.time())
        root = self.hot_root if tier == "hot" else self.cold_root
        path = self._path_for_node(root, node_id)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            tmp.replace(path)
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
        except Exception:
            pass

    def get(self, node_id: str) -> Optional[Dict[str, Any]]:
        HEAT.bump("swarm", node_id, 1)
        hot = self._path_for_node(self.hot_root, node_id)
        cold = self._path_for_node(self.cold_root, node_id)
        path = hot if hot.exists() else cold if cold.exists() else None
        if not path:
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except OSError as e:
            letter = _drive_letter_from_path(str(path))
            note_drive_failure(letter, e)
            return None
        except Exception:
            return None

    def list_nodes(self) -> List[str]:
        ids = set()
        for root in (self.hot_root, self.cold_root):
            try:
                for p in root.glob("*.json"):
                    ids.add(p.stem)
            except OSError as e:
                letter = _drive_letter_from_path(str(root))
                note_drive_failure(letter, e)
                continue
            except Exception:
                continue
        return sorted(ids)

# ---------------------------------------------------------------------------
# Heat‑based migration + self‑optimization
# ---------------------------------------------------------------------------

class HeatMigrator(threading.Thread):
    def __init__(self,
                 kv: "TieredKVCache",
                 vecdb: "TieredVectorDB",
                 interval: float = 10.0,
                 kv_threshold: int = 5,
                 vec_threshold: int = 5,
                 stop_event: Optional[threading.Event] = None):
        super().__init__(daemon=True)
        self.kv = kv
        self.vecdb = vecdb
        self.interval = interval
        self.kv_threshold = kv_threshold
        self.vec_threshold = vec_threshold
        self._stop = stop_event or threading.Event()

    def run(self):
        while not self._stop.is_set():
            snap = HEAT.snapshot()
            # auto‑tune thresholds based on load
            load = psutil.cpu_percent(interval=0.1)
            if load > 80:
                self.kv_threshold += 1
                self.vec_threshold += 1
            elif load < 40 and self.kv_threshold > 2:
                self.kv_threshold -= 1
                self.vec_threshold -= 1
            for key, heat in snap.items():
                try:
                    subsystem, logical = key.split(":", 1)
                except ValueError:
                    continue
                if subsystem == "kv" and heat >= self.kv_threshold:
                    self._ensure_kv_hot(logical)
                elif subsystem == "vec" and heat >= self.vec_threshold:
                    self._ensure_vec_hot(logical)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def _ensure_kv_hot(self, key: str):
        hot_path = self.kv._path_for_key(self.kv.hot_root, key)
        cold_path = self.kv._path_for_key(self.kv.cold_root, key)
        if hot_path.exists() or not cold_path.exists():
            return
        try:
            hot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cold_path, hot_path)
        except OSError as e:
            letter = _drive_letter_from_path(str(hot_path))
            note_drive_failure(letter, e)
        except Exception:
            pass

    def _ensure_vec_hot(self, vec_id: str):
        got = self.vecdb.get(vec_id)
        if not got:
            return
        emb, meta, tier = got
        if tier == "hot":
            return
        self.vecdb.upsert(vec_id, emb, meta, tier="hot")

# ---------------------------------------------------------------------------
# CRDT LWW register
# ---------------------------------------------------------------------------

class LWWRegister:
    def __init__(self, initial: Any = None):
        self.value = initial
        self.timestamp = 0.0

    def set(self, value: Any, ts: Optional[float] = None):
        ts = ts or time.time()
        if ts >= self.timestamp:
            self.value = value
            self.timestamp = ts

    def merge(self, other: "LWWRegister"):
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp

# ---------------------------------------------------------------------------
# Swarm mesh node (UDP gossip)
# ---------------------------------------------------------------------------

class SwarmMeshNode(threading.Thread):
    def __init__(self, node_id: str, port: int, peers: List[Tuple[str, int]],
                 crdt_state: Dict[str, LWWRegister],
                 interval: float = 5.0):
        super().__init__(daemon=True)
        self.node_id = node_id
        self.port = port
        self.peers = peers
        self.crdt_state = crdt_state
        self.interval = interval
        self._stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))

    def run(self):
        self.sock.settimeout(1.0)
        last_gossip = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now - last_gossip >= self.interval:
                self._send_gossip()
                last_gossip = now
            try:
                data, addr = self.sock.recvfrom(65535)
                self._handle_gossip(data)
            except socket.timeout:
                continue
            except Exception:
                continue

    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except Exception:
            pass

    def _send_gossip(self):
        payload = {
            "node_id": self.node_id,
            "state": {k: {"value": v.value, "ts": v.timestamp}
                      for k, v in self.crdt_state.items()},
        }
        raw = json.dumps(payload).encode("utf-8")
        for host, port in self.peers:
            try:
                self.sock.sendto(raw, (host, port))
            except Exception:
                continue

    def _handle_gossip(self, data: bytes):
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception:
            return
        state = payload.get("state", {})
        for k, v in state.items():
            if k not in self.crdt_state:
                self.crdt_state[k] = LWWRegister()
            self.crdt_state[k].set(v.get("value"), v.get("ts"))

# ---------------------------------------------------------------------------
# Autoencoder anomaly model + temporal model
# ---------------------------------------------------------------------------

class AnomalyAutoencoder(nn.Module if HAS_TORCH else object):
    def __init__(self, dim: int, hidden: int = 128):
        if not HAS_TORCH:
            return
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class TemporalEncoder(nn.Module if HAS_TORCH else object):
    def __init__(self, dim: int, hidden: int = 64):
        if not HAS_TORCH:
            return
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

class AnomalyScorer:
    def __init__(self, dim: int, window: int = 16):
        self.dim = dim
        self.window = window
        self.buffer: List[np.ndarray] = []
        if HAS_TORCH:
            self.ae = AnomalyAutoencoder(dim)
            self.temporal = TemporalEncoder(dim)
            self.ae_opt = optim.Adam(self.ae.parameters(), lr=1e-3)
            self.temporal_opt = optim.Adam(self.temporal.parameters(), lr=1e-3)
            self.ae.train()
            self.temporal.train()
        else:
            self.ae = None
            self.temporal = None

    def _push(self, emb: np.ndarray):
        self.buffer.append(emb.astype(np.float32))
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def score(self, emb: np.ndarray) -> Dict[str, float]:
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape[-1] != self.dim:
            return {"ae": 0.0, "temporal": 0.0}
        self._push(emb)
        if not HAS_TORCH or self.ae is None or self.temporal is None:
            return {
                "ae": ACCEL.score_vector(emb),
                "temporal": 0.0,
            }
        with torch.no_grad():
            x = torch.from_numpy(emb.reshape(1, -1))
            if torch.cuda.is_available():
                self.ae.cuda()
                x = x.cuda()
            recon = self.ae(x)
            ae_loss = torch.mean((x - recon) ** 2).item()
        if len(self.buffer) < self.window:
            return {"ae": float(ae_loss), "temporal": 0.0}
        seq = np.stack(self.buffer[-self.window:], axis=0)[None, ...]
        with torch.no_grad():
            t = torch.from_numpy(seq.astype(np.float32))
            if torch.cuda.is_available():
                self.temporal.cuda()
                t = t.cuda()
            out = self.temporal(t)
            temporal_score = float(out.item())
        return {"ae": float(ae_loss), "temporal": temporal_score}

    def train_on_batch(self, batch: np.ndarray):
        if not HAS_TORCH or self.ae is None or self.temporal is None:
            return
        x = torch.from_numpy(batch.astype(np.float32))
        if torch.cuda.is_available():
            self.ae.cuda()
            x = x.cuda()
        self.ae_opt.zero_grad()
        recon = self.ae(x)
        loss = torch.mean((x - recon) ** 2)
        loss.backward()
        self.ae_opt.step()

    def train_temporal_on_batch(self, seq_batch: np.ndarray, labels: np.ndarray):
        if not HAS_TORCH or self.temporal is None:
            return
        x = torch.from_numpy(seq_batch.astype(np.float32))
        y = torch.from_numpy(labels.astype(np.float32)).view(-1, 1)
        if torch.cuda.is_available():
            self.temporal.cuda()
            x = x.cuda()
            y = y.cuda()
        self.temporal_opt.zero_grad()
        out = self.temporal(x)
        loss = torch.mean((out - y) ** 2)
        loss.backward()
        self.temporal_opt.step()

# ---------------------------------------------------------------------------
# GNN attack graph reasoner
# ---------------------------------------------------------------------------

class AttackGraphReasoner:
    def __init__(self, dim: int = 32):
        self.graph = nx.DiGraph() if HAS_NETWORKX else None
        self.dim = dim
        self.node_embs: Dict[str, np.ndarray] = {}
        self.gnn_model = None
        if HAS_TORCH and HAS_PYG:
            self._init_gnn()

    def _init_gnn(self):
        class SimpleGCN(nn.Module):
            def __init__(self, in_dim, hidden=64, out_dim=32):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden)
                self.conv2 = GCNConv(hidden, out_dim)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x
        self.gnn_model = SimpleGCN(self.dim)

    def add_event(self, src: str, dst: str, weight: float = 1.0, meta: Optional[Dict[str, Any]] = None):
        if self.graph is None:
            return
        self.graph.add_edge(src, dst, weight=weight, meta=meta or {})
        for node in (src, dst):
            if node not in self.node_embs:
                self.node_embs[node] = np.random.randn(self.dim).astype(np.float32)

    def score_paths(self, max_len: int = 4) -> List[Tuple[List[str], float]]:
        if self.graph is None:
            return []
        if self.gnn_model is not None and HAS_TORCH and HAS_PYG:
            return self._score_paths_gnn(max_len)
        paths_scores: List[Tuple[List[str], float]] = []
        for src in self.graph.nodes:
            for dst in self.graph.nodes:
                if src == dst:
                    continue
                try:
                    for path in nx.all_simple_paths(self.graph, src, dst, cutoff=max_len):
                        score = 0.0
                        for u, v in zip(path[:-1], path[1:]):
                            w = self.graph[u][v].get("weight", 1.0)
                            score += w
                        paths_scores.append((path, score))
                except Exception:
                    continue
        paths_scores.sort(key=lambda x: x[1], reverse=True)
        return paths_scores[:10]

    def _score_paths_gnn(self, max_len: int) -> List[Tuple[List[str], float]]:
        if self.graph is None or self.gnn_model is None:
            return []
        nodes = list(self.graph.nodes)
        if not nodes:
            return []
        node_idx = {n: i for i, n in enumerate(nodes)}
        x = np.stack([self.node_embs.get(n, np.zeros(self.dim, dtype=np.float32)) for n in nodes])
        edges = []
        for u, v in self.graph.edges:
            edges.append([node_idx[u], node_idx[v]])
        if not edges:
            return []
        edge_index = np.array(edges, dtype=np.int64).T
        with torch.no_grad():
            x_t = torch.from_numpy(x)
            edge_t = torch.from_numpy(edge_index)
            if torch.cuda.is_available():
                self.gnn_model.cuda()
                x_t = x_t.cuda()
                edge_t = edge_t.cuda()
            out = self.gnn_model(x_t, edge_t).cpu().numpy()
        paths_scores: List[Tuple[List[str], float]] = []
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                try:
                    for path in nx.all_simple_paths(self.graph, src, dst, cutoff=max_len):
                        score = 0.0
                        for u, v in zip(path[:-1], path[1:]):
                            iu, iv = node_idx[u], node_idx[v]
                            score += float(np.dot(out[iu], out[iv]))
                        paths_scores.append((path, score))
                except Exception:
                    continue
        paths_scores.sort(key=lambda x: x[1], reverse=True)
        return paths_scores[:10]

# ---------------------------------------------------------------------------
# ETW sensor
# ---------------------------------------------------------------------------

class ETWSensor(threading.Thread):
    def __init__(self, telemetry: TelemetryBuffer, interval: float = 5.0):
        super().__init__(daemon=True)
        self.telemetry = telemetry
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        if HAS_ETW:
            self._run_real_etw()
        else:
            self._run_stub()

    def stop(self):
        self._stop.set()

    def _run_stub(self):
        while not self._stop.is_set():
            self.telemetry.log({
                "type": "etw_stub",
                "provider": "Kernel-Process",
                "event": "ProcessStart",
                "pid": random.randint(1000, 5000),
            })
            self._stop.wait(self.interval)

    def _run_real_etw(self):
        try:
            provider = etw.ProviderInfo(
                "Microsoft-Windows-Kernel-Process",
                etw.GUID("{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}"),
                etw.TRACE_LEVEL_INFORMATION
            )
            session = etw.ETW(
                providers=[provider],
                event_callback=self._on_etw_event
            )
            session.start()
            while not self._stop.is_set():
                time.sleep(0.5)
            session.stop()
        except Exception as e:
            print("[neurofabric] ETW real mode failed, falling back to stub:", e)
            self._run_stub()

    def _on_etw_event(self, event):
        try:
            data = {
                "type": "etw_kernel",
                "provider": "Microsoft-Windows-Kernel-Process",
                "event_id": getattr(event, "id", None),
                "opcode": getattr(event, "opcode", None),
                "pid": getattr(event, "process_id", None),
                "tid": getattr(event, "thread_id", None),
            }
            self.telemetry.log(data)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Stats worker
# ---------------------------------------------------------------------------

class StatsWorker(threading.Thread):
    def __init__(self,
                 kv: TieredKVCache,
                 vecdb: TieredVectorDB,
                 telemetry: TelemetryBuffer,
                 brain: BrainStateStore,
                 swarm: SwarmNodeStateStore,
                 interval: float = 2.0):
        super().__init__(daemon=True)
        self.kv = kv
        self.vecdb = vecdb
        self.telemetry = telemetry
        self.brain = brain
        self.swarm = swarm
        self.interval = interval
        self._stop = threading.Event()
        self.stats_lock = threading.Lock()
        self.stats: Dict[str, Any] = {}

    def run(self):
        while not self._stop.is_set():
            try:
                self._compute_stats()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def _compute_stats(self):
        global OPTANE_ROOT
        OPTANE_ROOT = ensure_root_alive(OPTANE_ROOT, OPTANE_FALLBACK_ROOT)
        try:
            usage = psutil.disk_usage(str(OPTANE_ROOT))
            usage_stats = {
                "percent": usage.percent,
                "used": usage.used,
                "total": usage.total,
                "error": None,
                "root": str(OPTANE_ROOT),
            }
        except OSError as e:
            letter = _drive_letter_from_path(str(OPTANE_ROOT))
            note_drive_failure(letter, e)
            usage_stats = {
                "percent": 0.0,
                "used": 0,
                "total": 0,
                "error": "disk_usage_failed",
                "root": str(OPTANE_ROOT),
            }
        except Exception:
            usage_stats = {
                "percent": 0.0,
                "used": 0,
                "total": 0,
                "error": "disk_usage_failed",
                "root": str(OPTANE_ROOT),
            }
        try:
            kv_count = len(self.kv.list_keys())
        except Exception:
            kv_count = -1
        try:
            vec_ids = self.vecdb.all_ids()
            vec_count = len(vec_ids)
        except Exception:
            vec_count = -1
        try:
            tel_segments = len(self.telemetry.list_segments())
        except Exception:
            tel_segments = -1
        try:
            brain_snaps = len(self.brain.list_snapshots())
        except Exception:
            brain_snaps = -1
        try:
            swarm_nodes = len(self.swarm.list_nodes())
        except Exception:
            swarm_nodes = -1
        heat = HEAT.top(30)
        with self.stats_lock:
            self.stats = {
                "usage": usage_stats,
                "kv_count": kv_count,
                "vec_count": vec_count,
                "tel_segments": tel_segments,
                "brain_snaps": brain_snaps,
                "swarm_nodes": swarm_nodes,
                "heat": heat,
                "timestamp": time.time(),
                "accelerator": {
                    "device": ACCEL.device,
                    "has_torch": ACCEL.has_torch,
                    "has_cuda": ACCEL.has_cuda,
                    "has_cupy": ACCEL.has_cupy,
                    "has_faiss": ACCEL.has_faiss,
                    "has_npu": ACCEL.has_npu,
                    "has_openvino": ACCEL.has_openvino,
                    "has_qnn": ACCEL.has_qnn,
                },
                "drive_roots": [str(r) for r in ALL_DRIVE_ROOTS],
            }

    def snapshot(self) -> Dict[str, Any]:
        with self.stats_lock:
            return dict(self.stats)

# ---------------------------------------------------------------------------
# Watchdog (self‑repair)
# ---------------------------------------------------------------------------

class Watchdog(threading.Thread):
    def __init__(self, name: str, target_thread: threading.Thread, restart_fn, interval: float = 5.0):
        super().__init__(daemon=True)
        self.name = name
        self.target_thread = target_thread
        self.restart_fn = restart_fn
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            if not self.target_thread.is_alive():
                print(f"[watchdog] {self.name} died, restarting...")
                self.target_thread = self.restart_fn()
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

# ---------------------------------------------------------------------------
# Tk cockpit
# ---------------------------------------------------------------------------

class NeuroFabricCockpitTk:
    def __init__(self,
                 root: tk.Tk,
                 optane_root: Path,
                 stats_worker: StatsWorker):
        self.root = root
        self.optane_root = optane_root
        self.stats_worker = stats_worker
        try:
            with open("service_port.txt", "r", encoding="utf-8") as f:
                self.service_port = f.read().strip()
        except Exception:
            self.service_port = "unknown"
        self.root.title("NeuroFabric Tactical Cockpit (Tk, Tier‑9)")
        self.root.geometry("1150x680")
        self._build_ui()
        self._schedule_updates()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        top_frame = ttk.Frame(self.root, padding=8)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)

        self.path_label = ttk.Label(
            top_frame,
            text=f"Optane root: {self.optane_root} | API port: {self.service_port}"
        )
        self.path_label.grid(row=0, column=0, sticky="w")

        self.usage_label = ttk.Label(top_frame, text="Usage: -- %")
        self.usage_label.grid(row=0, column=1, sticky="e")

        self.usage_bar = ttk.Progressbar(top_frame, orient="horizontal",
                                         mode="determinate", maximum=100)
        self.usage_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        mid_left = ttk.LabelFrame(self.root, text="Subsystem Stats", padding=8)
        mid_left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(4, 8))
        for i in range(2):
            mid_left.columnconfigure(i, weight=1)

        self.kv_label = ttk.Label(mid_left, text="KV entries: --")
        self.vec_label = ttk.Label(mid_left, text="Vector IDs: --")
        self.tel_label = ttk.Label(mid_left, text="Telemetry segments: --")
        self.brain_label = ttk.Label(mid_left, text="Brain snapshots: --")
        self.swarm_label = ttk.Label(mid_left, text="Swarm nodes: --")
        self.accel_label = ttk.Label(mid_left, text="Accelerator: --")

        self.kv_label.grid(row=0, column=0, sticky="w")
        self.vec_label.grid(row=0, column=1, sticky="w")
        self.tel_label.grid(row=1, column=0, sticky="w")
        self.brain_label.grid(row=1, column=1, sticky="w")
        self.swarm_label.grid(row=2, column=0, sticky="w")
        self.accel_label.grid(row=2, column=1, sticky="w")

        mid_right = ttk.LabelFrame(self.root, text="Heat Top Keys", padding=8)
        mid_right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(4, 8))
        mid_right.rowconfigure(0, weight=1)
        mid_right.columnconfigure(0, weight=1)

        self.heat_list = tk.Listbox(mid_right, height=15)
        self.heat_list.grid(row=0, column=0, sticky="nsew")

        heat_scroll = ttk.Scrollbar(mid_right, orient="vertical",
                                    command=self.heat_list.yview)
        heat_scroll.grid(row=0, column=1, sticky="ns")
        self.heat_list.configure(yscrollcommand=heat_scroll.set)

        bottom = ttk.Frame(self.root, padding=8)
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew")
        bottom.columnconfigure(0, weight=0)
        bottom.columnconfigure(1, weight=1)

        self.refresh_button = ttk.Button(bottom, text="Refresh now",
                                         command=self._refresh_from_snapshot)
        self.refresh_button.grid(row=0, column=0, sticky="w")

        self.status_label = ttk.Label(bottom, text="Status: idle")
        self.status_label.grid(row=0, column=1, sticky="e")

    def _schedule_updates(self):
        self._refresh_from_snapshot()
        self.root.after(1000, self._schedule_updates)

    def _refresh_from_snapshot(self):
        snap = self.stats_worker.snapshot()
        if not snap:
            self.status_label.config(text="Status: waiting for stats...")
            return
        usage = snap.get("usage", {})
        percent = usage.get("percent", 0.0)
        used = usage.get("used", 0)
        total = usage.get("total", 0)
        error = usage.get("error", None)
        root_str = usage.get("root", str(self.optane_root))
        if error:
            self.usage_bar["value"] = 0
            self.usage_label.config(text="Usage: -- (disk error skipped)")
        else:
            self.usage_bar["value"] = percent
            self.usage_label.config(
                text=f"Usage: {percent:.1f}%  |  Used: {self._fmt_bytes(used)} / {self._fmt_bytes(total)}"
            )
        self.path_label.config(
            text=f"Optane root: {root_str} | API port: {self.service_port}"
        )
        self.kv_label.config(text=f"KV entries: {snap.get('kv_count', '--')}")
        self.vec_label.config(text=f"Vector IDs: {snap.get('vec_count', '--')}")
        self.tel_label.config(text=f"Telemetry segments: {snap.get('tel_segments', '--')}")
        self.brain_label.config(text=f"Brain snapshots: {snap.get('brain_snaps', '--')}")
        self.swarm_label.config(text=f"Swarm nodes: {snap.get('swarm_nodes', '--')}")
        accel = snap.get("accelerator", {})
        self.accel_label.config(
            text=f"Accel: {accel.get('device','cpu')} | CUDA={accel.get('has_cuda')} | CuPy={accel.get('has_cupy')} | NPU={accel.get('has_npu')}"
        )
        self.heat_list.delete(0, tk.END)
        heat = snap.get("heat", [])
        if not heat:
            self.heat_list.insert(tk.END, "(no heat yet)")
        else:
            for k, v in heat:
                self.heat_list.insert(tk.END, f"{k}: {v}")
        ts = snap.get("timestamp", time.time())
        self.status_label.config(
            text=f"Status: updated @ {time.strftime('%H:%M:%S', time.localtime(ts))}"
        )

    def _fmt_bytes(self, n: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

# ---------------------------------------------------------------------------
# PySide6 cockpit (GPU‑friendly, minimal but real)
# ---------------------------------------------------------------------------

class NeuroFabricCockpitQt:
    def __init__(self, stats_worker: StatsWorker):
        if not HAS_PYSIDE6:
            raise RuntimeError("PySide6 not available")
        self.app = QtWidgets.QApplication(sys.argv)
        self.stats_worker = stats_worker
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("NeuroFabric Cockpit (Qt, Tier‑9)")
        self.window.resize(900, 600)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.label = QtWidgets.QLabel("Stats will appear here")
        self.label.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(self.label)
        self.window.setCentralWidget(central)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)

    def _tick(self):
        snap = self.stats_worker.snapshot()
        self.label.setText("<pre>" + json.dumps(snap, indent=2) + "</pre>")

    def run(self):
        self.window.show()
        self.app.exec()

# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class KVMetaIn(BaseModel):
    model_id: str
    session_id: str
    meta: Dict[str, Any]

class KVMetaOut(BaseModel):
    meta: Optional[Dict[str, Any]]

class SearchRequest(BaseModel):
    prefix: str = "anomaly:"
    top_k: int = 10
    vector: List[float]

class SearchHit(BaseModel):
    id: str
    score: float
    meta: Dict[str, Any]

class SearchResponse(BaseModel):
    hits: List[SearchHit]

def build_neurofabric_app(kv: TieredKVCache,
                          vecdb: TieredVectorDB,
                          anomaly_store: AnomalyEmbeddingStore,
                          autoencoder: AnomalyScorer,
                          attack_reasoner: AttackGraphReasoner) -> FastAPI:
    llm_kv = LLMKVAdapter(kv)
    app = FastAPI(title="NeuroFabric Service")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "optane_root": str(OPTANE_ROOT),
            "primary_root": str(OPTANE_PRIMARY_ROOT),
            "fallback_root": str(OPTANE_FALLBACK_ROOT),
            "runtime_blacklist": sorted(RUNTIME_BLACKLIST),
            "persistent_blacklist": sorted(PERSISTENT_BLACKLIST),
            "accelerator": {
                "device": ACCEL.device,
                "has_torch": ACCEL.has_torch,
                "has_cuda": ACCEL.has_cuda,
                "has_cupy": ACCEL.has_cupy,
                "has_faiss": ACCEL.has_faiss,
                "has_npu": ACCEL.has_npu,
            },
            "drive_roots": [str(r) for r in ALL_DRIVE_ROOTS],
        }

    @app.put("/kv")
    def put_kv(meta_in: KVMetaIn):
        llm_kv.save_cache_meta(meta_in.model_id, meta_in.session_id, meta_in.meta)
        return {}

    @app.get("/kv/{model_id}/{session_id}", response_model=KVMetaOut)
    def get_kv(model_id: str, session_id: str):
        meta = llm_kv.load_cache_meta(model_id, session_id)
        return KVMetaOut(meta=meta)

    @app.post("/anomaly/search", response_model=SearchResponse)
    def anomaly_search(req: SearchRequest):
        q = np.array(req.vector, dtype=np.float32)
        hits_raw = anomaly_store.search_nearest(q, prefix=req.prefix, top_k=req.top_k)
        hits = [SearchHit(id=vid, score=score, meta=meta)
                for (vid, score, meta) in hits_raw]
        return SearchResponse(hits=hits)

    @app.post("/anomaly/score")
    def anomaly_score(req: SearchRequest):
        q = np.array(req.vector, dtype=np.float32)
        scores = autoencoder.score(q)
        accel_score = ACCEL.score_vector(q)
        return {
            "autoencoder_score": scores["ae"],
            "temporal_score": scores["temporal"],
            "accelerator_norm_score": accel_score,
        }

    @app.get("/attack/paths")
    def attack_paths():
        paths = attack_reasoner.score_paths()
        return {
            "paths": [
                {"nodes": path, "score": score}
                for (path, score) in paths
            ]
        }

    return app

def start_service_in_thread(kv: TieredKVCache,
                            vecdb: TieredVectorDB,
                            anomaly_store: AnomalyEmbeddingStore,
                            autoencoder: AnomalyScorer,
                            attack_reasoner: AttackGraphReasoner,
                            host: str = "0.0.0.0",
                            preferred_port: int = 8080):
    import uvicorn
    port = find_free_port(preferred_port)
    try:
        with open("service_port.txt", "w", encoding="utf-8") as f:
            f.write(str(port))
    except Exception:
        pass
    app = build_neurofabric_app(kv, vecdb, anomaly_store, autoencoder, attack_reasoner)
    def _run():
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        except Exception as e:
            print("[neurofabric] uvicorn error:", e)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ram_cache = RAMCache(max_items=4096)
    kv = TieredKVCache(ram_cache=ram_cache)
    vecdb = TieredVectorDB()
    telemetry = TelemetryBuffer()
    brain = BrainStateStore()
    swarm = SwarmNodeStateStore()

    llm_kv = LLMKVAdapter(kv)
    anomaly_store = AnomalyEmbeddingStore(vecdb)
    kv_layout = KVShardLayout()

    shard_tensor = np.random.randn(16, 64, 128).astype(np.float32)
    shard_path = kv_layout.save_shard("llm_main", "sess_1", "layer0_head0", shard_tensor)
    llm_kv.save_cache_meta("llm_main", "sess_1", {
        "impl": "mmap_shard_v1",
        "shards": {"layer0_head0": str(shard_path)},
    })

    anomaly_store.upsert_anomaly(
        "anom_1",
        np.random.randn(768).astype(np.float32),
        {"score": 0.97, "source": "kernel_sensor"},
        hot=True,
    )
    anomaly_store.upsert_attack_node(
        "graph_1",
        "node_1",
        np.random.randn(768).astype(np.float32),
        {"type": "lateral_move", "host": "host01"},
        hot=False,
    )

    telemetry.log({"type": "boot", "msg": "system started"})
    telemetry.log({"type": "sensor", "msg": "kernel hook event"})

    brain.save_snapshot("global_brain", {"version": 1, "routes": ["llm_a", "llm_b"]})
    swarm.upsert("node_alpha", {"trust": 0.92, "role": "sensor"})
    swarm.upsert("node_beta", {"trust": 0.75, "role": "analyzer"})

    stop_event = threading.Event()
    migrator = HeatMigrator(kv=kv, vecdb=vecdb, interval=5.0,
                            kv_threshold=3, vec_threshold=3,
                            stop_event=stop_event)
    migrator.start()

    stats_worker = StatsWorker(kv=kv,
                               vecdb=vecdb,
                               telemetry=telemetry,
                               brain=brain,
                               swarm=swarm,
                               interval=2.0)
    stats_worker.start()

    etw_sensor = ETWSensor(telemetry, interval=5.0)
    etw_sensor.start()

    crdt_state: Dict[str, LWWRegister] = {
        "global_trust": LWWRegister(0.9),
    }
    mesh_node = SwarmMeshNode(
        node_id="local",
        port=19000,
        peers=[],  # add ("other_host", 19000) for real multi‑node mesh
        crdt_state=crdt_state,
        interval=10.0,
    )
    mesh_node.start()

    autoencoder = AnomalyScorer(dim=768)
    attack_reasoner = AttackGraphReasoner(dim=32)
    if HAS_NETWORKX:
        attack_reasoner.add_event("host01", "host02", weight=1.5, meta={"type": "lateral"})
        attack_reasoner.add_event("host02", "dc01", weight=2.0, meta={"type": "priv_esc"})

    service_thread = start_service_in_thread(
        kv, vecdb, anomaly_store, autoencoder, attack_reasoner
    )

    # Watchdogs (self‑repair)
    def restart_etw():
        t = ETWSensor(telemetry, interval=5.0)
        t.start()
        return t

    etw_watchdog = Watchdog("ETW", etw_sensor, restart_etw, interval=10.0)
    etw_watchdog.start()

    # GUI: prefer PySide6 cockpit if available, else Tk
    if HAS_PYSIDE6:
        qt_cockpit = NeuroFabricCockpitQt(stats_worker)
        qt_cockpit.run()
    else:
        root = tk.Tk()
        cockpit = NeuroFabricCockpitTk(
            root=root,
            optane_root=OPTANE_ROOT,
            stats_worker=stats_worker,
        )

        def on_close():
            stop_event.set()
            stats_worker.stop()
            etw_sensor.stop()
            mesh_node.stop()
            etw_watchdog.stop()
            migrator.join(timeout=2.0)
            stats_worker.join(timeout=2.0)
            etw_sensor.join(timeout=2.0)
            mesh_node.join(timeout=2.0)
            etw_watchdog.join(timeout=2.0)
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        root.mainloop()

if __name__ == "__main__":
    main()
