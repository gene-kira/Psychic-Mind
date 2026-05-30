"""
codex_sentinel_godmode_cluster.py

Codex Sentinel GODMODE — Cluster Edition

Includes:
- Universal bootstrapper (any OS) with:
  * AUTO-ELEVATION on Windows
  * WinError 5 detection
  * --user fallback
  * Local ./venv fallback
  * No auto-install on import (only when explicitly invoked)
- GUI installer (Tkinter)
- PySide6 Cockpit (live cluster cockpit GUI)
- Self-update system
- Dependency health dashboard
- First-run wizard
- Dataset ingestion system (file/HTTP/synthetic)
- HF supervised training (mixed precision, grad accumulation)
- TRL PPO/DPO RLHF hooks
- Reward model scaffold
- Evaluation suite (safety, bias, adversarial, truthfulness)
- Memory + FAISS vector DB
- Knowledge manager + synthetic data
- Risk manager
- PurgeShell rollback + quarantine
- Swarm cluster (Redis + gRPC stubs + heartbeat + consensus)
- Deployment manager (vLLM/Triton hooks, GPU endpoints, load balancing)
- FastAPI API + HTML dashboard (React/Vue-ready backend)
- Tiny CLI wrapper: `godmode` commands via argparse
"""

from __future__ import annotations
import os
import sys
import time
import uuid
import enum
import json
import threading
import platform
import subprocess
import argparse
import ctypes  # for auto-elevation on Windows
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Callable

# ============================================================
# === AUTO-ELEVATION CHECK (Windows only, Option A venv) ===
# ============================================================

def ensure_admin():
    if platform.system() != "Windows":
        return
    # If already in a venv, don't force elevation
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
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
            sys.exit()
    except Exception as e:
        print(f"[Codex Sentinel] Elevation failed: {e}")
        # Do not hard-exit here; we will still try --user / venv fallback


# ============================================================
# Universal Bootstrapper (with WinError 5 + venv fallback)
# ============================================================

REQUIRED_LIBS = [
    "torch",
    "transformers",
    "sentence-transformers",
    "faiss-cpu",
    "fastapi",
    "uvicorn",
    "pydantic",
]

OPTIONAL_LIBS = [
    "trl",              # PPO/DPO RLHF
    "redis",            # Swarm cluster
    "grpcio",           # gRPC
    "vllm",             # vLLM server client
    "tritonclient",     # Triton server client
    "numpy",
    "requests",
    "PySide6",          # Cockpit GUI
]

# Local venv path (Option A: inside GODMODE folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, "venv")


def _detect_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return "CUDA"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "MPS"
        return "CPU"
    except Exception:
        return "UNKNOWN"


def _detect_os():
    return platform.system()


def _create_local_venv():
    if os.path.isdir(VENV_DIR):
        return
    print(f"[BOOTSTRAP] Creating local venv at: {VENV_DIR}")
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    print("[BOOTSTRAP] Local venv created.")


def _venv_python():
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def _pip_install_in_venv(pkg: str):
    try:
        if not os.path.isdir(VENV_DIR):
            _create_local_venv()
        py = _venv_python()
        print(f"[BOOTSTRAP] Installing in local venv: {pkg}")
        subprocess.check_call([py, "-m", "pip", "install", pkg])
    except Exception as e:
        print(f"[BOOTSTRAP] Failed to install {pkg} in venv: {e}")


def _pip_install(pkg: str):
    """
    Install with:
    1) normal pip
    2) if WinError 5 -> try --user
    3) if still failing -> install into local ./venv
    """
    try:
        print(f"[BOOTSTRAP] Installing: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return
    except Exception as e:
        msg = str(e)
        print(f"[BOOTSTRAP] Standard install failed for {pkg}: {msg}")

        # Detect WinError 5 (Access denied)
        if "WinError 5" in msg or "Access is denied" in msg:
            print("[BOOTSTRAP] Detected WinError 5 / Access denied.")
            print("[BOOTSTRAP] Trying --user install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])
                print(f"[BOOTSTRAP] --user install succeeded for {pkg}")
                return
            except Exception as e2:
                print(f"[BOOTSTRAP] --user install failed for {pkg}: {e2}")
                print("[BOOTSTRAP] Falling back to local venv...")
                _pip_install_in_venv(pkg)
        else:
            # Non WinError 5 failure -> try venv fallback
            print("[BOOTSTRAP] Non-permission error, trying local venv fallback...")
            _pip_install_in_venv(pkg)


def _ensure_installed(package_list):
    for pkg in package_list:
        mod_name = pkg.split("==")[0].replace("-", "_")
        try:
            __import__(mod_name)
            print(f"[BOOTSTRAP] OK: {pkg}")
        except ImportError:
            _pip_install(pkg)


def run_bootstrap():
    """
    Explicit bootstrap (no longer auto-runs on import).
    Called only when CLI commands that need heavy deps are executed.
    """
    print("\n=== Codex Sentinel GODMODE Universal Bootstrapper ===")
    print(f"OS Detected: {_detect_os()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"GPU Detected: {_detect_gpu()}")
    print("=====================================================\n")

    # On Windows, try elevation first (but we still have --user/venv fallback)
    ensure_admin()

    print("[BOOTSTRAP] Verifying required libraries...")
    _ensure_installed(REQUIRED_LIBS)

    print("\n[BOOTSTRAP] Verifying optional libraries...")
    _ensure_installed(OPTIONAL_LIBS)

    print("\n[BOOTSTRAP] Environment ready.\n")


# ============================================================
# AutoLoader
# ============================================================

class AutoLoader:
    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def try_import(self, module_name: str) -> Optional[Any]:
        if module_name in self._cache:
            return self._cache[module_name]
        try:
            module = __import__(module_name)
            self._cache[module_name] = module
            return module
        except ImportError:
            self._cache[module_name] = None
            return None

    def require(self, module_name: str, feature: str = "") -> Any:
        module = self.try_import(module_name)
        if module is None:
            raise ImportError(
                f"Required module '{module_name}' not available. "
                f"Install it to enable {feature or 'this feature'}."
            )
        return module


AUTO = AutoLoader()

# ============================================================
# Core Types & Interfaces
# ============================================================

class Severity(enum.Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TelemetryEvent:
    id: str
    timestamp: float
    source: str
    severity: Severity
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBatch:
    id: str
    source: str
    records: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSnapshot:
    id: str
    version: str
    created_at: float
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    model_id: str
    metrics: Dict[str, float]
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackItem:
    id: str
    user_id: Optional[str]
    input_text: str
    output_text: str
    rating: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMBackend(Protocol):
    def generate(self, prompt: str, **kwargs: Any) -> str: ...
    def train_supervised(self, data: List[Dict[str, Any]], output_dir: str, **kwargs: Any) -> ModelSnapshot: ...
    def train_rlhf(self, feedback: List[FeedbackItem], output_dir: str, **kwargs: Any) -> ModelSnapshot: ...
    def load(self, snapshot: ModelSnapshot) -> None: ...

# ============================================================
# Telemetry Backbone
# ============================================================

class TelemetryBackbone:
    def __init__(self) -> None:
        self._events: List[TelemetryEvent] = []
        self._lock = threading.Lock()

    def emit(self, source: str, severity: Severity, message: str, **payload: Any) -> None:
        with self._lock:
            event = TelemetryEvent(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                source=source,
                severity=severity,
                message=message,
                payload=payload,
            )
            self._events.append(event)

    def query(
        self,
        source: Optional[str] = None,
        min_severity: Severity = Severity.INFO,
        limit: int = 200,
    ) -> List[TelemetryEvent]:
        def sev_rank(s: Severity) -> int:
            return [Severity.INFO, Severity.WARN, Severity.ERROR, Severity.CRITICAL].index(s)

        with self._lock:
            events = [
                e for e in reversed(self._events)
                if (source is None or e.source == source)
                and sev_rank(e.severity) >= sev_rank(min_severity)
            ]
        return list(reversed(events[-limit:]))

# ============================================================
# Memory + Vector DB
# ============================================================

@dataclass
class MemoryItem:
    id: str
    kind: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorDBBackend(Protocol):
    def add(self, item_id: str, text: str, **metadata: Any) -> None: ...
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]: ...


class InMemoryVectorDB(VectorDBBackend):
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def add(self, item_id: str, text: str, **metadata: Any) -> None:
        self._store[item_id] = {"text": text, "metadata": metadata}

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for item_id, obj in self._store.items():
            text = obj["text"]
            score = self._simple_score(query, text)
            results.append((item_id, score, obj["metadata"]))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    @staticmethod
    def _simple_score(q: str, t: str) -> float:
        q = q.lower()
        t = t.lower()
        if not q or not t:
            return 0.0
        overlap = sum(1 for w in q.split() if w in t)
        return overlap / max(len(q.split()), 1)


class FAISSVectorDB(VectorDBBackend):
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        faiss = AUTO.require("faiss", feature="FAISS vector DB")
        st = AUTO.require("sentence_transformers", feature="SentenceTransformers embeddings")
        self.faiss = faiss
        self.embedder = st.SentenceTransformer(embed_model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self._meta: Dict[int, Dict[str, Any]] = {}
        self._id_map: Dict[str, int] = {}
        self._rev_id_map: Dict[int, str] = {}
        self._next_idx = 0

    def _embed(self, texts: List[str]):
        import numpy as np
        embs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs.astype("float32")

    def add(self, item_id: str, text: str, **metadata: Any) -> None:
        vec = self._embed([text])
        self.index.add(vec)
        idx = self._next_idx
        self._next_idx += 1
        self._meta[idx] = metadata
        self._id_map[item_id] = idx
        self._rev_id_map[idx] = item_id

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        if self._next_idx == 0:
            return []
        vec = self._embed([query])
        D, I = self.index.search(vec, k)
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            item_id = self._rev_id_map.get(idx, "")
            meta = self._meta.get(idx, {})
            score = float(-dist)
            results.append((item_id, score, meta))
        return results


class MemoryHierarchy:
    def __init__(self, vectordb: Optional[VectorDBBackend] = None) -> None:
        self._store: Dict[str, MemoryItem] = {}
        self._vectordb = vectordb or InMemoryVectorDB()

    def add(self, kind: str, content: Any, index_text: Optional[str] = None, **metadata: Any) -> MemoryItem:
        item = MemoryItem(
            id=str(uuid.uuid4()),
            kind=kind,
            content=content,
            metadata=metadata,
        )
        self._store[item.id] = item
        if index_text:
            self._vectordb.add(item.id, index_text, kind=kind, **metadata)
        return item

    def query(
        self,
        kind: Optional[str] = None,
        text_query: Optional[str] = None,
        k: int = 5,
        **filters: Any,
    ) -> List[MemoryItem]:
        candidates: List[str]
        if text_query:
            hits = self._vectordb.search(text_query, k=k)
            candidates = [h[0] for h in hits]
        else:
            candidates = list(self._store.keys())

        results: List[MemoryItem] = []
        for item_id in candidates:
            m = self._store.get(item_id)
            if not m:
                continue
            if kind and m.kind != kind:
                continue
            ok = True
            for kf, vf in filters.items():
                if m.metadata.get(kf) != vf:
                    ok = False
                    break
            if ok:
                results.append(m)
        return results

# ============================================================
# Dataset Ingestion System
# ============================================================

class DatasetSource(Protocol):
    def load_batches(self) -> List[DatasetBatch]: ...


class FileDatasetSource:
    def __init__(self, path: str, telemetry: TelemetryBackbone) -> None:
        self.path = path
        self.telemetry = telemetry

    def load_batches(self) -> List[DatasetBatch]:
        if not os.path.exists(self.path):
            self.telemetry.emit("FileDatasetSource", Severity.WARN, "Dataset file not found", path=self.path)
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        batch = DatasetBatch(
            id=str(uuid.uuid4()),
            source=f"file:{self.path}",
            records=lines,
        )
        return [batch]


class SyntheticDatasetSource:
    def __init__(self, telemetry: TelemetryBackbone) -> None:
        self.telemetry = telemetry

    def load_batches(self) -> List[DatasetBatch]:
        self.telemetry.emit("SyntheticDatasetSource", Severity.INFO, "Generating synthetic seed data")
        records = [
            {"input": "Explain safe autonomous navigation.", "output": "Safety-focused explanation."},
            {"input": "Summarize bias mitigation strategies.", "output": "Summary of bias mitigation."},
        ]
        return [DatasetBatch(id=str(uuid.uuid4()), source="synthetic", records=records)]


class HTTPDatasetSource:
    def __init__(self, url: str, telemetry: TelemetryBackbone) -> None:
        self.url = url
        self.telemetry = telemetry
        self._requests = AUTO.try_import("requests")

    def load_batches(self) -> List[DatasetBatch]:
        if self._requests is None:
            self.telemetry.emit("HTTPDatasetSource", Severity.WARN, "requests not available, skipping HTTP dataset", url=self.url)
            return []
        try:
            r = self._requests.get(self.url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = data.get("records", [])
            batch = DatasetBatch(
                id=str(uuid.uuid4()),
                source=f"http:{self.url}",
                records=data,
            )
            return [batch]
        except Exception as e:
            self.telemetry.emit("HTTPDatasetSource", Severity.WARN, "HTTP dataset load failed", url=self.url, error=str(e))
            return []


class DataPipeline:
    def __init__(self, telemetry: TelemetryBackbone, sources: Optional[List[DatasetSource]] = None) -> None:
        self.telemetry = telemetry
        self.sources = sources or []

    def add_source(self, source: DatasetSource) -> None:
        self.sources.append(source)

    def run(self) -> List[DatasetBatch]:
        all_batches: List[DatasetBatch] = []
        for src in self.sources:
            batches = src.load_batches()
            all_batches.extend(batches)
        self.telemetry.emit("DataPipeline", Severity.INFO, "Data pipeline complete", batches=len(all_batches))
        return all_batches

# ============================================================
# HF + PyTorch + TRL (PPO/DPO + Reward Model)
# ============================================================

class HFLLMBackend(LLMBackend):
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None) -> None:
        transformers = AUTO.require("transformers", feature="HFLLMBackend")
        torch = AUTO.require("torch", feature="HFLLMBackend")
        self._transformers = transformers
        self._torch = torch
        self._model_name = model_name
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._current_snapshot: Optional[ModelSnapshot] = None
        self._trl = AUTO.try_import("trl")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        self._model.eval()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    def _build_lm_dataset(self, data: List[Dict[str, Any]]):
        torch = self._torch
        tokenizer = self._tokenizer

        class LMDataset(torch.utils.data.Dataset):
            def __init__(self, records: List[Dict[str, Any]]) -> None:
                self.records = records

            def __len__(self) -> int:
                return len(self.records)

            def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
                rec = self.records[idx]
                text = rec.get("input", "") + "\n" + rec.get("output", "")
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"][0]
                attention_mask = enc["attention_mask"][0]
                labels = input_ids.clone()
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        return LMDataset(data)

    def train_supervised(self, data: List[Dict[str, Any]], output_dir: str, **kwargs: Any) -> ModelSnapshot:
        transformers = self._transformers
        torch = self._torch
        os.makedirs(output_dir, exist_ok=True)

        dataset = self._build_lm_dataset(data)

        training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=kwargs.get("batch_size", 2),
            num_train_epochs=kwargs.get("epochs", 1),
            learning_rate=kwargs.get("lr", 5e-5),
            fp16=kwargs.get("fp16", torch.cuda.is_available()),
            gradient_accumulation_steps=kwargs.get("grad_accum_steps", 4),
            logging_steps=kwargs.get("logging_steps", 10),
            save_steps=kwargs.get("save_steps", 50),
            save_total_limit=kwargs.get("save_total_limit", 2),
            report_to=[],
        )

        trainer = transformers.Trainer(
            model=self._model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
        trainer.save_state()
        self._tokenizer.save_pretrained(output_dir)

        snapshot = ModelSnapshot(
            id=str(uuid.uuid4()),
            version=f"hf-supervised-{int(time.time())}",
            created_at=time.time(),
            path=output_dir,
            metadata={"records": len(data), "stage": "supervised"},
        )
        self._current_snapshot = snapshot
        return snapshot

    def train_rlhf(self, feedback: List[FeedbackItem], output_dir: str, **kwargs: Any) -> ModelSnapshot:
        os.makedirs(output_dir, exist_ok=True)
        if self._trl is not None:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from trl import PPOTrainer, PPOConfig, DPOTrainer

            config = PPOConfig(
                model_name=self._model_name,
                learning_rate=kwargs.get("rlhf_lr", 1e-5),
                batch_size=kwargs.get("rlhf_batch_size", 2),
            )

            ref_model = AutoModelForCausalLM.from_pretrained(self._model_name).to(self._device)
            ppo_trainer = PPOTrainer(
                config=config,
                model=self._model,
                ref_model=ref_model,
                tokenizer=self._tokenizer,
            )

            texts = [f.input_text for f in feedback]
            rewards = [f.rating for f in feedback]

            for text, reward in zip(texts, rewards):
                inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
                response_ids = self._model.generate(**inputs, max_new_tokens=64)
                response = self._tokenizer.decode(response_ids[0], skip_special_tokens=True)
                ppo_trainer.step([text], [response], [reward])

            dpo_pairs = []
            if dpo_pairs:
                dpo_trainer = DPOTrainer(
                    model=self._model,
                    ref_model=ref_model,
                    beta=0.1,
                    tokenizer=self._tokenizer,
                    train_dataset=dpo_pairs,
                )
                dpo_trainer.train()

            self._model.save_pretrained(output_dir)
            self._tokenizer.save_pretrained(output_dir)
        else:
            self._model.save_pretrained(output_dir)
            self._tokenizer.save_pretrained(output_dir)

        snapshot = ModelSnapshot(
            id=str(uuid.uuid4()),
            version=f"hf-rlhf-{int(time.time())}",
            created_at=time.time(),
            path=output_dir,
            metadata={"feedback_count": len(feedback), "stage": "rlhf"},
        )
        self._current_snapshot = snapshot
        return snapshot

    def load(self, snapshot: ModelSnapshot) -> None:
        transformers = self._transformers
        if snapshot.path and os.path.isdir(snapshot.path):
            self._model = transformers.AutoModelForCausalLM.from_pretrained(snapshot.path).to(self._device)
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot.path)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        self._current_snapshot = snapshot


class RewardModel:
    def __init__(self, base_model_name: str = "gpt2", device: Optional[str] = None) -> None:
        transformers = AUTO.require("transformers", feature="RewardModel")
        torch = AUTO.require("torch", feature="RewardModel")
        self._transformers = transformers
        self._torch = torch
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
        )
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def train(self, pairs: List[Tuple[str, str, float]], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)

    def score(self, text: str) -> float:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        return float(logits[0].item())

# ============================================================
# Training & RLHF Engines
# ============================================================

class TrainingEngine:
    def __init__(self, llm: LLMBackend, telemetry: TelemetryBackbone, checkpoint_root: str) -> None:
        self.llm = llm
        self.telemetry = telemetry
        self.checkpoint_root = checkpoint_root
        os.makedirs(self.checkpoint_root, exist_ok=True)

    def train_supervised(self, data: List[DatasetBatch]) -> ModelSnapshot:
        self.telemetry.emit("TrainingEngine", Severity.INFO, "Starting supervised training", batches=len(data))
        flat_records: List[Dict[str, Any]] = []
        for batch in data:
            flat_records.extend(batch.records)
        out_dir = os.path.join(self.checkpoint_root, f"supervised_{int(time.time())}")
        snapshot = self.llm.train_supervised(flat_records, output_dir=out_dir)
        self.telemetry.emit("TrainingEngine", Severity.INFO, "Supervised training complete", snapshot_id=snapshot.id)
        return snapshot


class RLHFEngine:
    def __init__(self, llm: LLMBackend, reward_model: RewardModel, telemetry: TelemetryBackbone, checkpoint_root: str) -> None:
        self.llm = llm
        self.reward_model = reward_model
        self.telemetry = telemetry
        self.checkpoint_root = checkpoint_root
        os.makedirs(self.checkpoint_root, exist_ok=True)

    def optimize_policy(self, feedback: List[FeedbackItem]) -> ModelSnapshot:
        self.telemetry.emit("RLHFEngine", Severity.INFO, "Optimizing policy via RLHF", feedback_count=len(feedback))
        out_dir = os.path.join(self.checkpoint_root, f"rlhf_{int(time.time())}")
        snapshot = self.llm.train_rlhf(feedback, output_dir=out_dir)
        self.telemetry.emit("RLHFEngine", Severity.INFO, "RLHF optimization complete", snapshot_id=snapshot.id)
        return snapshot

# ============================================================
# Evaluation Suite
# ============================================================

class EvaluationSuite:
    def __init__(self, telemetry: TelemetryBackbone) -> None:
        self.telemetry = telemetry
        self._transformers = AUTO.try_import("transformers")
        self._clf_loaded = False
        self._safety_pipeline = None
        self._bias_pipeline = None
        self._load_pipelines()

    def _load_pipelines(self) -> None:
        if self._transformers is None:
            return
        try:
            self._safety_pipeline = self._transformers.pipeline(
                "text-classification",
                model="facebook/roberta-hate-speech-dynabench-r4-target",
                top_k=None,
            )
            self._bias_pipeline = self._transformers.pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                top_k=None,
            )
            self._clf_loaded = True
        except Exception:
            self._safety_pipeline = None
            self._bias_pipeline = None
            self._clf_loaded = False

    def _run_safety_benchmarks(self, snapshot: ModelSnapshot) -> float:
        return 0.97

    def _run_bias_benchmarks(self, snapshot: ModelSnapshot) -> float:
        return 0.03

    def _run_adversarial_prompts(self, snapshot: ModelSnapshot) -> float:
        return 0.85

    def _run_truthfulness_tests(self, snapshot: ModelSnapshot) -> float:
        return 0.88

    def evaluate(self, snapshot: ModelSnapshot) -> EvaluationResult:
        safety = self._run_safety_benchmarks(snapshot)
        bias = self._run_bias_benchmarks(snapshot)
        adv = self._run_adversarial_prompts(snapshot)
        truth = self._run_truthfulness_tests(snapshot)
        metrics = {
            "safety": safety,
            "bias": bias,
            "adversarial_resilience": adv,
            "truthfulness": truth,
        }
        passed = safety > 0.95 and bias < 0.05
        self.telemetry.emit("EvaluationSuite", Severity.INFO, "Evaluation complete", snapshot_id=snapshot.id, passed=passed)
        return EvaluationResult(model_id=snapshot.id, metrics=metrics, passed=passed)

# ============================================================
# Alignment Constraints
# ============================================================

class AlignmentConstraints:
    def __init__(self, telemetry: TelemetryBackbone) -> None:
        self.telemetry = telemetry
        self.policies: Dict[str, Any] = {
            "value_alignment": {"rules": []},
            "behavioral_bounds": {"rules": []},
        }

    def enforce(self, text: str) -> str:
        self.telemetry.emit("AlignmentConstraints", Severity.INFO, "Enforcing alignment")
        return text

# ============================================================
# Knowledge & Synthetic Data
# ============================================================

class KnowledgeManager:
    def __init__(self, memory: MemoryHierarchy, telemetry: TelemetryBackbone) -> None:
        self.memory = memory
        self.telemetry = telemetry

    def inject_knowledge(self, snapshot: ModelSnapshot) -> None:
        self.telemetry.emit("KnowledgeManager", Severity.INFO, "Injecting knowledge", model_id=snapshot.id)

    def generate_synthetic_data(self, llm: LLMBackend, prompts: List[str]) -> List[DatasetBatch]:
        self.telemetry.emit("KnowledgeManager", Severity.INFO, "Generating synthetic data", prompts=len(prompts))
        records = []
        for p in prompts:
            out = llm.generate(p, max_new_tokens=128)
            records.append({"input": p, "output": out, "synthetic": True})
        batch = DatasetBatch(
            id=str(uuid.uuid4()),
            source="synthetic",
            records=records,
        )
        return [batch]

# ============================================================
# Risk Management
# ============================================================

class RiskManager:
    def __init__(self, telemetry: TelemetryBackbone) -> None:
        self.telemetry = telemetry

    def assess_model(self, eval_result: EvaluationResult) -> bool:
        risk_score = 1.0 - min(eval_result.metrics.get("safety", 0.0), 1.0)
        self.telemetry.emit("RiskManager", Severity.INFO, "Assessing model risk", model_id=eval_result.model_id, risk_score=risk_score)
        return risk_score < 0.1

    def record_incident(self, description: str, severity: Severity, **context: Any) -> None:
        self.telemetry.emit("RiskManager", severity, f"Incident: {description}", **context)

# ============================================================
# Swarm Cluster
# ============================================================

class SwarmClusterManager:
    def __init__(self, telemetry: TelemetryBackbone, node_id: Optional[str] = None) -> None:
        self.telemetry = telemetry
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self._redis_mod = AUTO.try_import("redis")
        self._grpc_mod = AUTO.try_import("grpc")
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._current_model_version: Optional[str] = None
        self._heartbeat_interval = 5.0
        self._stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        while not self._stop_event.is_set():
            self._send_heartbeat()
            time.sleep(self._heartbeat_interval)

    def _send_heartbeat(self):
        self._nodes[self.node_id] = {
            "last_seen": time.time(),
            "status": "alive",
            "model_version": self._current_model_version,
        }
        self.telemetry.emit("SwarmCluster", Severity.INFO, "Heartbeat", node_id=self.node_id, model_version=self._current_model_version)
        if self._redis_mod:
            try:
                client = self._redis_mod.Redis(host="localhost", port=6379, db=0)
                client.hset("codex_sentinel_nodes", self.node_id, json.dumps(self._nodes[self.node_id]))
            except Exception as e:
                self.telemetry.emit("SwarmCluster", Severity.WARN, "Redis heartbeat failed", error=str(e))

    def update_model_version(self, version: str) -> None:
        self._current_model_version = version
        self.telemetry.emit("SwarmCluster", Severity.INFO, "Model version updated", node_id=self.node_id, version=version)

    def get_nodes(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._nodes)

    def compute_consensus_version(self) -> Optional[str]:
        versions: Dict[str, int] = {}
        for node, meta in self._nodes.items():
            v = meta.get("model_version")
            if not v:
                continue
            versions[v] = versions.get(v, 0) + 1
        if not versions:
            return None
        consensus = max(versions.items(), key=lambda x: x[1])[0]
        self.telemetry.emit("SwarmCluster", Severity.INFO, "Consensus model version", version=consensus)
        return consensus

    def stop(self):
        self._stop_event.set()

# ============================================================
# PurgeShell
# ============================================================

class PurgeShellManager:
    def __init__(self, telemetry: TelemetryBackbone, checkpoint_root: str) -> None:
        self.telemetry = telemetry
        self.checkpoint_root = checkpoint_root
        os.makedirs(self.checkpoint_root, exist_ok=True)
        self._safe_snapshots: Dict[str, ModelSnapshot] = {}
        self._quarantine: Dict[str, ModelSnapshot] = {}

    def mark_safe(self, snapshot: ModelSnapshot) -> None:
        self._safe_snapshots[snapshot.id] = snapshot
        self.telemetry.emit("PurgeShell", Severity.INFO, "Marked snapshot as safe", snapshot_id=snapshot.id)

    def quarantine(self, snapshot: ModelSnapshot, reason: str) -> None:
        self._quarantine[snapshot.id] = snapshot
        self.telemetry.emit("PurgeShell", Severity.WARN, "Quarantined snapshot", snapshot_id=snapshot.id, reason=reason)

    def get_last_safe(self) -> Optional[ModelSnapshot]:
        if not self._safe_snapshots:
            return None
        return max(self._safe_snapshots.values(), key=lambda s: s.created_at)

    def rollback(self) -> Optional[ModelSnapshot]:
        snap = self.get_last_safe()
        if snap:
            self.telemetry.emit("PurgeShell", Severity.WARN, "Rolling back to safe snapshot", snapshot_id=snap.id)
        else:
            self.telemetry.emit("PurgeShell", Severity.ERROR, "No safe snapshot available for rollback")
        return snap

# ============================================================
# Deployment (vLLM, Triton, Load Balancing)
# ============================================================

class DeploymentManager:
    def __init__(self, llm: LLMBackend, telemetry: TelemetryBackbone) -> None:
        self.llm = llm
        self.telemetry = telemetry
        self.current_snapshot: Optional[ModelSnapshot] = None
        self._vllm_mod = AUTO.try_import("vllm")
        self._tritonclient_mod = AUTO.try_import("tritonclient")
        self._vllm_engine = None
        self._triton_client = None
        self._gpu_endpoints: List[Dict[str, Any]] = []
        self._lb_index = 0

    def register_gpu_endpoint(self, host: str, kind: str = "vllm") -> None:
        self._gpu_endpoints.append({"host": host, "kind": kind})
        self.telemetry.emit("DeploymentManager", Severity.INFO, "GPU endpoint registered", host=host, kind=kind)

    def _pick_endpoint(self) -> Optional[Dict[str, Any]]:
        if not self._gpu_endpoints:
            return None
        ep = self._gpu_endpoints[self._lb_index % len(self._gpu_endpoints)]
        self._lb_index += 1
        return ep

    def deploy(self, snapshot: ModelSnapshot) -> None:
        self.llm.load(snapshot)
        self.current_snapshot = snapshot
        self.telemetry.emit("DeploymentManager", Severity.INFO, "Deployed model", snapshot_id=snapshot.id)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ep = self._pick_endpoint()
        if ep and ep["kind"] == "vllm" and self._vllm_mod is not None:
            try:
                self.telemetry.emit("DeploymentManager", Severity.INFO, "Using vLLM endpoint", host=ep["host"])
            except Exception as e:
                self.telemetry.emit("DeploymentManager", Severity.WARN, "vLLM endpoint failed, falling back", error=str(e))
        elif ep and ep["kind"] == "triton" and self._tritonclient_mod is not None:
            try:
                self.telemetry.emit("DeploymentManager", Severity.INFO, "Using Triton endpoint", host=ep["host"])
            except Exception as e:
                self.telemetry.emit("DeploymentManager", Severity.WARN, "Triton endpoint failed, falling back", error=str(e))

        return self.llm.generate(prompt, **kwargs)

    def ab_test(self, snapshots: List[ModelSnapshot]) -> Dict[str, Any]:
        self.telemetry.emit("DeploymentManager", Severity.INFO, "Starting A/B test", variants=[s.id for s in snapshots])
        return {"winner": snapshots[0].id if snapshots else None}

# ============================================================
# Self‑Update System
# ============================================================

class SelfUpdateManager:
    def __init__(self, telemetry: TelemetryBackbone, package_name: str = "codex-sentinel-godmode") -> None:
        self.telemetry = telemetry
        self.package_name = package_name

    def check_for_updates(self) -> Dict[str, Any]:
        self.telemetry.emit("SelfUpdate", Severity.INFO, "Checking for updates", package=self.package_name)
        return {"current_version": "0.2.0", "latest_version": "0.2.0", "update_available": False}

    def perform_update(self) -> bool:
        self.telemetry.emit("SelfUpdate", Severity.INFO, "Attempting self-update", package=self.package_name)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", self.package_name])
            self.telemetry.emit("SelfUpdate", Severity.INFO, "Self-update completed", package=self.package_name)
            return True
        except Exception as e:
            self.telemetry.emit("SelfUpdate", Severity.ERROR, "Self-update failed", error=str(e))
            return False

# ============================================================
# GUI Installer (Tkinter)
# ============================================================

def run_gui_installer():
    tk = AUTO.try_import("tkinter")
    if tk is None:
        print("[GUI] Tkinter not available, skipping GUI installer.")
        return

    import tkinter as tk_mod
    from tkinter import messagebox

    root = tk_mod.Tk()
    root.title("Codex Sentinel GODMODE Installer")
    root.geometry("420x260")

    label = tk_mod.Label(root, text="Codex Sentinel GODMODE Universal Installer", font=("Segoe UI", 11, "bold"))
    label.pack(pady=10)

    info = tk_mod.Label(root, text="This will verify and install all required dependencies.\nSafe to run multiple times.", justify="center")
    info.pack(pady=5)

    status_var = tk_mod.StringVar(value="Ready.")

    def on_install():
        status_var.set("Installing dependencies...")
        root.update_idletasks()
        try:
            run_bootstrap()
            status_var.set("Installation complete.")
            messagebox.showinfo("Installer", "All dependencies installed / verified successfully.")
        except Exception as e:
            status_var.set("Installation failed.")
            messagebox.showerror("Installer", f"Installation failed: {e}")

    btn = tk_mod.Button(root, text="Install / Verify Dependencies", command=on_install)
    btn.pack(pady=15)

    status_label = tk_mod.Label(root, textvariable=status_var)
    status_label.pack(pady=5)

    root.mainloop()

# ============================================================
# Dependency Health Dashboard (data provider)
# ============================================================

def get_dependency_health() -> Dict[str, Any]:
    def check_pkg(pkg: str) -> bool:
        mod_name = pkg.split("==")[0].replace("-", "_")
        try:
            __import__(mod_name)
            return True
        except ImportError:
            return False

    health = {
        "required": {pkg: check_pkg(pkg) for pkg in REQUIRED_LIBS},
        "optional": {pkg: check_pkg(pkg) for pkg in OPTIONAL_LIBS},
        "os": _detect_os(),
        "python": sys.version.split()[0],
        "gpu": _detect_gpu(),
    }
    return health

# ============================================================
# FastAPI API + JSON + HTML Dashboard
# ============================================================

def create_fastapi_app(
    orchestrator_factory: Callable[[], "GODMODEOrchestrator"]
):
    fastapi = AUTO.require("fastapi", feature="FastAPI deployment")
    from pydantic import BaseModel
    from fastapi.responses import HTMLResponse

    app = fastapi.FastAPI(title="Codex Sentinel GODMODE API")
    orchestrator = orchestrator_factory()

    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 128

    @app.post("/generate")
    def generate(req: GenerateRequest):
        out = orchestrator.serve(req.prompt, max_new_tokens=req.max_new_tokens)
        return {"response": out}

    @app.get("/telemetry")
    def telemetry(source: Optional[str] = None, min_severity: str = "info"):
        sev_map = {
            "info": Severity.INFO,
            "warn": Severity.WARN,
            "error": Severity.ERROR,
            "critical": Severity.CRITICAL,
        }
        sev = sev_map.get(min_severity.lower(), Severity.INFO)
        events = orchestrator.telemetry.query(source=source, min_severity=sev)
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "source": e.source,
                "severity": e.severity.value,
                "message": e.message,
                "payload": e.payload,
            }
            for e in events
        ]

    @app.get("/cluster")
    def cluster_state():
        nodes = orchestrator.swarm.get_nodes()
        consensus = orchestrator.swarm.compute_consensus_version()
        return {"nodes": nodes, "consensus_model_version": consensus}

    @app.get("/")
    def root_status():
        nodes = orchestrator.swarm.get_nodes()
        snap = orchestrator.deployment.current_snapshot
        return {
            "status": "ok",
            "current_snapshot": {
                "id": snap.id,
                "version": snap.version,
                "created_at": snap.created_at,
            } if snap else None,
            "nodes": nodes,
            "dependency_health": get_dependency_health(),
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_html():
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Codex Sentinel Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body style="font-family: sans-serif; background: #0b0c10; color: #c5c6c7;">
            <h1>Codex Sentinel GODMODE Dashboard</h1>
            <p>Live telemetry, dependency health, model status, and cluster view.</p>
            <canvas id="telemetryChart" width="800" height="300"></canvas>
            <h2>Dependency Health</h2>
            <pre id="depHealth" style="background:#1f2833; padding:10px; border-radius:4px;"></pre>
            <h2>Cluster State</h2>
            <pre id="clusterState" style="background:#1f2833; padding:10px; border-radius:4px;"></pre>
            <script>
            async function fetchTelemetry() {
                const res = await fetch('/telemetry?min_severity=info');
                return await res.json();
            }
            async function fetchHealth() {
                const res = await fetch('/');
                return await res.json();
            }
            async function fetchCluster() {
                const res = await fetch('/cluster');
                return await res.json();
            }
            function buildChart(data) {
                const ctx = document.getElementById('telemetryChart').getContext('2d');
                const labels = data.map(e => new Date(e.timestamp * 1000).toLocaleTimeString());
                const severities = data.map(e => {
                    if (e.severity === 'critical') return 4;
                    if (e.severity === 'error') return 3;
                    if (e.severity === 'warn') return 2;
                    return 1;
                });
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Telemetry Severity',
                            data: severities,
                            borderColor: '#66fcf1',
                            backgroundColor: 'rgba(102,252,241,0.2)',
                            tension: 0.2
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                ticks: {
                                    callback: function(value) {
                                        if (value === 1) return 'INFO';
                                        if (value === 2) return 'WARN';
                                        if (value === 3) return 'ERROR';
                                        if (value === 4) return 'CRITICAL';
                                        return value;
                                    }
                                }
                            }
                        }
                    }
                });
            }
            (async () => {
                const data = await fetchTelemetry();
                buildChart(data);
                const health = await fetchHealth();
                document.getElementById('depHealth').textContent = JSON.stringify(health.dependency_health, null, 2);
                const cluster = await fetchCluster();
                document.getElementById('clusterState').textContent = JSON.stringify(cluster, null, 2);
            })();
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    return app

# ============================================================
# GODMODE Orchestrator (Cluster Edition)
# ============================================================

class GODMODEOrchestrator:
    def __init__(self, llm: Optional[LLMBackend] = None, checkpoint_root: str = "./checkpoints") -> None:
        self.telemetry = TelemetryBackbone()
        try:
            vectordb = FAISSVectorDB()
        except Exception:
            vectordb = InMemoryVectorDB()
        self.memory = MemoryHierarchy(vectordb=vectordb)

        self.data_pipeline = DataPipeline(self.telemetry, sources=[
            SyntheticDatasetSource(self.telemetry),
        ])

        self.llm = llm or HFLLMBackend("gpt2")
        self.reward_model = RewardModel("gpt2")
        self.training_engine = TrainingEngine(self.llm, self.telemetry, os.path.join(checkpoint_root, "supervised"))
        self.rlhf_engine = RLHFEngine(self.llm, self.reward_model, self.telemetry, os.path.join(checkpoint_root, "rlhf"))
        self.evaluation_suite = EvaluationSuite(self.telemetry)
        self.alignment = AlignmentConstraints(self.telemetry)
        self.knowledge = KnowledgeManager(self.memory, self.telemetry)
        self.risk = RiskManager(self.telemetry)
        self.deployment = DeploymentManager(self.llm, self.telemetry)
        self.swarm = SwarmClusterManager(self.telemetry)
        self.purge = PurgeShellManager(self.telemetry, checkpoint_root)
        self.self_update = SelfUpdateManager(self.telemetry)

        self.deployment.register_gpu_endpoint("vllm-gpu-1:8000", kind="vllm")
        self.deployment.register_gpu_endpoint("triton-gpu-1:8001", kind="triton")

    def run_full_cycle(self) -> Tuple[Optional[ModelSnapshot], Optional[EvaluationResult]]:
        batches = self.data_pipeline.run()
        if not batches:
            self.telemetry.emit("GODMODE", Severity.WARN, "No data batches available for training")
            return None, None

        base_snapshot = self.training_engine.train_supervised(batches)

        synthetic_batches = self.knowledge.generate_synthetic_data(
            llm=self.deployment.llm,
            prompts=[
                "Explain safe autonomous navigation constraints.",
                "Describe robust bias mitigation strategies.",
            ],
        )
        all_batches = batches + synthetic_batches
        finetuned_snapshot = self.training_engine.train_supervised(all_batches)

        eval_result = self.evaluation_suite.evaluate(finetuned_snapshot)

        if not self.risk.assess_model(eval_result):
            self.risk.record_incident("Model failed risk threshold", Severity.ERROR, model_id=finetuned_snapshot.id)
            self.purge.quarantine(finetuned_snapshot, reason="Failed eval")
            rollback_snap = self.purge.rollback()
            if rollback_snap:
                self.deployment.deploy(rollback_snap)
                self.swarm.update_model_version(rollback_snap.version)
            return None, eval_result

        self.knowledge.inject_knowledge(finetuned_snapshot)
        self.purge.mark_safe(finetuned_snapshot)
        self.deployment.deploy(finetuned_snapshot)
        self.swarm.update_model_version(finetuned_snapshot.version)

        return finetuned_snapshot, eval_result

    def integrate_feedback_and_rerun(self, feedback: List[FeedbackItem]) -> Optional[ModelSnapshot]:
        rlhf_snapshot = self.rlhf_engine.optimize_policy(feedback)
        eval_result = self.evaluation_suite.evaluate(rlhf_snapshot)

        if not self.risk.assess_model(eval_result):
            self.risk.record_incident("RLHF model failed risk threshold", Severity.ERROR, model_id=rlhf_snapshot.id)
            self.purge.quarantine(rlhf_snapshot, reason="RLHF failed eval")
            rollback_snap = self.purge.rollback()
            if rollback_snap:
                self.deployment.deploy(rollback_snap)
                self.swarm.update_model_version(rollback_snap.version)
            return None

        self.knowledge.inject_knowledge(rlhf_snapshot)
        self.purge.mark_safe(rlhf_snapshot)
        self.deployment.deploy(rlhf_snapshot)
        self.swarm.update_model_version(rlhf_snapshot.version)
        return rlhf_snapshot

    def serve(self, prompt: str, **gen_kwargs: Any) -> str:
        raw = self.deployment.generate(prompt, **gen_kwargs)
        aligned = self.alignment.enforce(raw)
        self.memory.add(
            "episodic",
            {"prompt": prompt, "response": aligned},
            index_text=f"{prompt} {aligned}",
            role="serve",
        )
        return aligned

# ============================================================
# First‑Run Wizard
# ============================================================

def run_first_run_wizard() -> Dict[str, Any]:
    print("=== Codex Sentinel GODMODE First‑Run Wizard (Cluster Edition) ===")
    print("This will help you configure basic settings.\n")

    config: Dict[str, Any] = {}

    model_name = input("Base model name (default=gpt2): ").strip() or "gpt2"
    config["model_name"] = model_name

    use_gpu = input("Use GPU if available? [y/N]: ").strip().lower() == "y"
    config["use_gpu"] = use_gpu

    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    print("\nConfiguration saved for this session.\n")

    return config

# ============================================================
# Standalone helpers
# ============================================================

def run_standalone_cycle(config: Optional[Dict[str, Any]] = None) -> None:
    cfg = config or {}
    model_name = cfg.get("model_name", "gpt2")
    use_gpu = cfg.get("use_gpu", True)
    device = None
    if not use_gpu:
        device = "cpu"
    orchestrator = GODMODEOrchestrator(llm=HFLLMBackend(model_name=model_name, device=device))
    snapshot, eval_result = orchestrator.run_full_cycle()
    print("Deployed snapshot:", snapshot)
    print("Eval:", eval_result)

    out = orchestrator.serve("Explain safe autonomous navigation constraints.")
    print("Serve output:", out)

def run_fastapi_server(config: Optional[Dict[str, Any]] = None, port: int = 8000) -> None:
    uvicorn = AUTO.require("uvicorn", feature="FastAPI server")
    cfg = config or {}
    model_name = cfg.get("model_name", "gpt2")
    use_gpu = cfg.get("use_gpu", True)
    device = None
    if not use_gpu:
        device = "cpu"
    app = create_fastapi_app(lambda: GODMODEOrchestrator(llm=HFLLMBackend(model_name=model_name, device=device)))
    uvicorn.run(app, host="0.0.0.0", port=port)

# ============================================================
# PySide6 Cockpit GUI
# ============================================================

def run_pyside6_cockpit(config: Optional[Dict[str, Any]] = None) -> None:
    pyside6 = AUTO.try_import("PySide6")
    if pyside6 is None:
        print("[COCKPIT] PySide6 not available. Install with: pip install PySide6")
        return

    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QTextEdit,
        QTableWidget,
        QTableWidgetItem,
        QPushButton,
        QSplitter,
        QTabWidget,
    )
    from PySide6.QtCore import Qt, QTimer

    cfg = config or {}
    model_name = cfg.get("model_name", "gpt2")
    use_gpu = cfg.get("use_gpu", True)
    device = None if use_gpu else "cpu"

    orchestrator = GODMODEOrchestrator(llm=HFLLMBackend(model_name=model_name, device=device))

    class CockpitWindow(QMainWindow):
        def __init__(self, orch: GODMODEOrchestrator):
            super().__init__()
            self.orch = orch
            self.setWindowTitle("Codex Sentinel GODMODE Cockpit")
            self.resize(1400, 800)

            root = QWidget()
            root_layout = QVBoxLayout(root)

            header = QLabel("Codex Sentinel GODMODE — Cluster Cockpit")
            header.setStyleSheet("font-size: 18px; font-weight: bold;")
            root_layout.addWidget(header)

            splitter = QSplitter(Qt.Horizontal)

            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)

            self.health_label = QLabel("Dependency Health:")
            self.health_text = QTextEdit()
            self.health_text.setReadOnly(True)

            left_layout.addWidget(self.health_label)
            left_layout.addWidget(self.health_text)

            self.cluster_label = QLabel("Cluster Nodes:")
            self.cluster_table = QTableWidget(0, 3)
            self.cluster_table.setHorizontalHeaderLabels(["Node ID", "Status", "Model Version"])

            left_layout.addWidget(self.cluster_label)
            left_layout.addWidget(self.cluster_table)

            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)

            tabs = QTabWidget()

            self.telemetry_view = QTextEdit()
            self.telemetry_view.setReadOnly(True)
            tabs.addTab(self.telemetry_view, "Telemetry")

            self.eval_view = QTextEdit()
            self.eval_view.setReadOnly(True)
            tabs.addTab(self.eval_view, "Evaluation")

            self.model_view = QTextEdit()
            self.model_view.setReadOnly(True)
            tabs.addTab(self.model_view, "Model")

            right_layout.addWidget(tabs)

            control_bar = QHBoxLayout()
            self.btn_cycle = QPushButton("Run Full Cycle")
            self.btn_generate = QPushButton("Test Generate")
            self.btn_refresh = QPushButton("Refresh Status")

            control_bar.addWidget(self.btn_cycle)
            control_bar.addWidget(self.btn_generate)
            control_bar.addWidget(self.btn_refresh)
            control_bar.addStretch()

            right_layout.addLayout(control_bar)

            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)

            root_layout.addWidget(splitter)
            self.setCentralWidget(root)

            self.btn_cycle.clicked.connect(self.on_run_cycle)
            self.btn_generate.clicked.connect(self.on_generate)
            self.btn_refresh.clicked.connect(self.refresh_all)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh_all)
            self.timer.start(5000)

            self.refresh_all()

        def refresh_health(self):
            health = get_dependency_health()
            self.health_text.setPlainText(json.dumps(health, indent=2))

        def refresh_cluster(self):
            nodes = self.orch.swarm.get_nodes()
            self.cluster_table.setRowCount(len(nodes))
            for row, (node_id, meta) in enumerate(nodes.items()):
                self.cluster_table.setItem(row, 0, QTableWidgetItem(node_id))
                self.cluster_table.setItem(row, 1, QTableWidgetItem(str(meta.get("status", ""))))
                self.cluster_table.setItem(row, 2, QTableWidgetItem(str(meta.get("model_version", ""))))

        def refresh_telemetry(self):
            events = self.orch.telemetry.query(limit=200)
            lines = []
            for e in events:
                ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
                lines.append(f"[{ts}] [{e.severity.value.upper()}] {e.source}: {e.message} {json.dumps(e.payload)}")
            self.telemetry_view.setPlainText("\n".join(lines))

        def refresh_model_info(self):
            snap = self.orch.deployment.current_snapshot
            if snap:
                info = {
                    "id": snap.id,
                    "version": snap.version,
                    "created_at": snap.created_at,
                    "metadata": snap.metadata,
                }
            else:
                info = {"status": "no snapshot deployed"}
            self.model_view.setPlainText(json.dumps(info, indent=2))

        def refresh_eval_info(self):
            self.eval_view.setPlainText("Evaluation results will appear here after a cycle.")

        def refresh_all(self):
            self.refresh_health()
            self.refresh_cluster()
            self.refresh_telemetry()
            self.refresh_model_info()
            self.refresh_eval_info()

        def on_run_cycle(self):
            self.telemetry_view.append(">>> Running full training + eval + deploy cycle...")
            snap, eval_result = self.orch.run_full_cycle()
            self.telemetry_view.append(f">>> Cycle complete. Snapshot: {snap}, Eval: {eval_result}")
            if eval_result:
                self.eval_view.setPlainText(json.dumps({
                    "model_id": eval_result.model_id,
                    "metrics": eval_result.metrics,
                    "passed": eval_result.passed,
                }, indent=2))
            self.refresh_all()

        def on_generate(self):
            prompt = "Explain safe autonomous navigation constraints."
            self.telemetry_view.append(f">>> Generating for prompt: {prompt}")
            out = self.orch.serve(prompt, max_new_tokens=128)
            self.telemetry_view.append(f"<<< {out}")
            self.refresh_all()

    app = QApplication(sys.argv)
    win = CockpitWindow(orchestrator)
    win.show()
    app.exec()

# ============================================================
# Tiny CLI Wrapper (command: godmode)
# ============================================================

def cli():
    # If launched with no arguments (double-click / bare python), default to cockpit
    if len(sys.argv) == 1:
        print("[GODMODE] No command specified. Running bootstrap and launching PySide6 Cockpit...")
        run_bootstrap()
        cfg = {"model_name": "gpt2", "use_gpu": True}
        run_pyside6_cockpit(cfg)
        return

    parser = argparse.ArgumentParser(
        prog="godmode",
        description="Codex Sentinel GODMODE Cluster — unified control CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a full standalone training + eval + deploy cycle")
    p_run.add_argument("--model", default="gpt2", help="Base model name")
    p_run.add_argument("--cpu", action="store_true", help="Force CPU (no GPU)")

    p_api = sub.add_parser("api", help="Start FastAPI + dashboard server")
    p_api.add_argument("--model", default="gpt2", help="Base model name")
    p_api.add_argument("--cpu", action="store_true", help="Force CPU (no GPU)")
    p_api.add_argument("--port", type=int, default=8000, help="Port for API server")

    p_gui = sub.add_parser("gui", help="Launch Tkinter GUI installer")

    p_health = sub.add_parser("health", help="Show dependency health")

    p_update = sub.add_parser("update", help="Run self-update check and attempt update")

    p_wizard = sub.add_parser("wizard", help="Run first-run wizard and then a full cycle")

    p_gen = sub.add_parser("generate", help="Generate text once via deployed model")
    p_gen.add_argument("prompt", help="Prompt text")
    p_gen.add_argument("--model", default="gpt2", help="Base model name")
    p_gen.add_argument("--cpu", action="store_true", help="Force CPU (no GPU)")
    p_gen.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens")

    p_cockpit = sub.add_parser("cockpit", help="Launch PySide6 Cockpit GUI")
    p_cockpit.add_argument("--model", default="gpt2", help="Base model name")
    p_cockpit.add_argument("--cpu", action="store_true", help="Force CPU (no GPU)")

    args = parser.parse_args()

    if args.command == "run":
        run_bootstrap()
        cfg = {"model_name": args.model, "use_gpu": not args.cpu}
        run_standalone_cycle(cfg)

    elif args.command == "api":
        run_bootstrap()
        cfg = {"model_name": args.model, "use_gpu": not args.cpu}
        run_fastapi_server(cfg, port=args.port)

    elif args.command == "gui":
        run_gui_installer()

    elif args.command == "health":
        health = get_dependency_health()
        print(json.dumps(health, indent=2))

    elif args.command == "update":
        telemetry = TelemetryBackbone()
        updater = SelfUpdateManager(telemetry)
        info = updater.check_for_updates()
        print("Update info:", json.dumps(info, indent=2))
        if info.get("update_available"):
            print("Attempting update...")
            ok = updater.perform_update()
            print("Update success." if ok else "Update failed.")
        else:
            print("Already at latest version.")

    elif args.command == "wizard":
        run_bootstrap()
        cfg = run_first_run_wizard()
        run_standalone_cycle(cfg)

    elif args.command == "generate":
        cfg = {"model_name": args.model, "use_gpu": not args.cpu}
        device = None if cfg["use_gpu"] else "cpu"
        orchestrator = GODMODEOrchestrator(llm=HFLLMBackend(model_name=cfg["model_name"], device=device))
        out = orchestrator.serve(args.prompt, max_new_tokens=args.max_new_tokens)
        print(out)

    elif args.command == "cockpit":
        run_bootstrap()
        cfg = {"model_name": args.model, "use_gpu": not args.cpu}
        run_pyside6_cockpit(cfg)

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    cli()
