"""
Forklift + HF LLaMA Integration
with Router Training, FP8/INT8 Per-Channel Quantization,
KV-Aware Sparsity, Multi-GPU Sharding, Robust Loader,
and Full Tkinter Cockpit (structured, framed, organized)
Always runs with or without a real LLM (tiny fallback model).
"""

import sys
import os
import time
import threading
from typing import Dict, Tuple, List

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
# Optional deps for loader
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

# -------------------------
# Tkinter GUI
# -------------------------
try:
    import tkinter as tk
    from tkinter import scrolledtext, filedialog, ttk
except ImportError:
    print("Tkinter not available on this system.")
    sys.exit(1)


# =========================
# Config
# =========================

PRIMARY_MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # may be gated

HAS_CUDA = torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count()
DEFAULT_DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")

TILE_ROWS = 64
TILE_COLS = 64
ACTIVATION_SKIP_THRESHOLD = 1e-3  # base threshold

# Quantization mode: "int8" or "fp8"
QUANT_MODE = "fp8"  # "int8" or "fp8"

# Router training config
ROUTER_FEATURE_DIM = 3  # norm, mean, max
ROUTER_HIDDEN_DIM = 16
ROUTER_LR = 1e-3
ROUTER_TRAIN_STEPS = 200  # small demo

# Model queue (remote)
FALLBACK_MODELS = [
    PRIMARY_MODEL_NAME,
    "TheBloke/Llama-2-7B-Chat-GPTQ",
    "TheBloke/Llama-2-7B-Chat-AWQ",
    "NousResearch/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "EleutherAI/gpt-neo-1.3B",
]

MODEL_QUEUE_INDEX = 0
NEEDS_MANUAL_DOWNLOAD = False

CURRENT_TOKENIZER = None
CURRENT_MODEL = None
CURRENT_MODEL_NAME = "None"
IS_FALLBACK_MODEL = False

DOWNLOAD_THREAD = None
DOWNLOAD_RUNNING = False


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


# =========================
# Tiny fallback model (always available)
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
# Per-channel quantization helpers (INT8 + FP8-style)
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


def quantize_tile(tile: torch.Tensor):
    if QUANT_MODE == "int8":
        return quantize_per_channel_int8(tile, num_bits=8)
    elif QUANT_MODE == "fp8":
        return quantize_per_channel_fp8_emulation(tile)
    else:
        raise ValueError(f"Unknown QUANT_MODE: {QUANT_MODE}")


def dequantize_tile(q_tile: torch.Tensor, scale: torch.Tensor):
    if QUANT_MODE == "int8":
        return dequantize_per_channel_int8(q_tile, scale)
    elif QUANT_MODE == "fp8":
        return dequantize_per_channel_fp8_emulation(q_tile, scale)
    else:
        raise ValueError(f"Unknown QUANT_MODE: {QUANT_MODE}")


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
# Tile importance router
# =========================

class TileRouter(nn.Module):
    def __init__(self, in_dim=ROUTER_FEATURE_DIM, hidden_dim=ROUTER_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats: torch.Tensor):
        return self.net(feats).squeeze(-1)


ROUTER = TileRouter().to(DEFAULT_DEVICE)


# =========================
# Forklift executor
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
        factor = 1.0 + (extra / 64.0) ** 2
        return factor

    def linear(self, layer_name: str, weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor):
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

        for tr, tc in compute_tile_indices(W.shape, TILE_ROWS, TILE_COLS):
            rs, cs = tile_slice(tr, tc, TILE_ROWS, TILE_COLS)
            X_sub = X_flat[:, cs]

            self.tiles_considered += 1

            norm = X_sub.norm()
            mean = X_sub.mean()
            maxv = X_sub.abs().max()

            feats = torch.stack([norm, mean, maxv]).detach().to(DEFAULT_DEVICE)

            with torch.no_grad():
                score = self.router(feats.view(1, -1))[0].item()

            router_factor = max(0.1, 1.0 - 0.5 * torch.tanh(torch.tensor(score)).item())
            effective_thresh = ACTIVATION_SKIP_THRESHOLD * kv_scale * router_factor

            m = min(TILE_ROWS, out_dim - rs.start)
            n = min(TILE_COLS, in_dim - cs.start)
            tile_bytes = m * n * W.element_size()

            if norm.item() < effective_thresh:
                self.tiles_skipped += 1
                self.bytes_avoided += tile_bytes
                self.layer_skipped[layer_name] = self.layer_skipped.get(layer_name, 0) + tile_bytes

                if self.collect_router_data:
                    self.router_feats.append(feats)
                    self.router_labels.append(torch.tensor(0.0, device=DEFAULT_DEVICE))
                continue

            key = (layer_name, tr, tc)
            cached = self.cache.get(key)
            if cached is None:
                tile = W[rs, cs].contiguous()
                q_tile, scale = quantize_tile(tile)
                self.cache.put(key, q_tile, scale)
                self.layer_bytes[layer_name] = self.layer_bytes.get(layer_name, 0) + \
                                               q_tile.numel() * q_tile.element_size() + \
                                               scale.numel() * scale.element_size()
            else:
                q_tile, scale = cached

            tile = dequantize_tile(q_tile, scale)

            partial = X_sub @ tile.t()
            Y_flat[:, rs] += partial

            flops = matvec_flops(tile.shape[0], tile.shape[1]) * batch
            self.total_flops += flops
            self.layer_flops[layer_name] = self.layer_flops.get(layer_name, 0) + flops

            if self.collect_router_data:
                contrib_norm = partial.norm().detach()
                label = torch.sigmoid(contrib_norm / (1e-3 + norm))
                self.router_feats.append(feats)
                self.router_labels.append(label.to(DEFAULT_DEVICE))

        if B is not None:
            Y_flat += B

        Y = Y_flat.view(*orig_shape[:-1], out_dim)
        return Y

    def stats(self):
        hits = self.cache.hits
        misses = self.cache.misses
        total = hits + misses
        hit_rate = hits / total if total else 0.0

        return {
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
        return out

    def get_router_dataset(self):
        if not self.router_feats:
            return None, None
        X = torch.stack(self.router_feats, dim=0)
        y = torch.stack(self.router_labels, dim=0)
        return X, y


EXECUTOR = ForkliftExecutor(GLOBAL_CACHE, ROUTER)


# =========================
# ForkliftLinear wrapper
# =========================

class ForkliftLinear(nn.Module):
    def __init__(self, base: nn.Linear, name: str, executor: ForkliftExecutor):
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.name = name
        self.exec = executor

    def forward(self, x):
        return self.exec.linear(self.name, self.weight, self.bias, x)


def wrap_linear_modules(model: nn.Module, executor: ForkliftExecutor, prefix=""):
    for name, module in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            setattr(model, name, ForkliftLinear(module, full_name, executor))
        else:
            wrap_linear_modules(module, executor, full_name)


# =========================
# Model management helpers
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


# =========================
# Fast HF load helper (non-blocking-ish)
# =========================

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
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,
            local_files_only=False,
        )
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


# =========================
# Robust loader (remote + fallback + emergency)
# =========================

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


# =========================
# Manual downloader (GUI-triggered)
# =========================

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

    with torch.no_grad():
        if hasattr(model, "generate"):
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

    stats = EXECUTOR.stats()
    layer_stats = EXECUTOR.per_layer_stats()
    return text, stats, layer_stats


# =========================
# Router training
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


# =========================
# Tkinter Cockpit (structured, framed, with status + inspector)
# =========================

def gui_tkinter():
    global MODEL_QUEUE_INDEX

    root = tk.Tk()
    root.title("Forklift GPU Cockpit (Tkinter)")
    root.geometry("1100x750")

    # Top info + status
    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X, padx=10, pady=5)

    info_label = tk.Label(
        top_frame,
        text=f"CUDA: {HAS_CUDA}, GPUs: {NUM_GPUS}, QUANT_MODE: {QUANT_MODE}",
        font=("Consolas", 11)
    )
    info_label.pack(side=tk.LEFT)

    status_var = tk.StringVar()
    status_label = tk.Label(
        top_frame,
        textvariable=status_var,
        font=("Consolas", 11),
        fg="green"
    )
    status_label.pack(side=tk.RIGHT)

    def update_status():
        mode = "Fallback" if IS_FALLBACK_MODEL else "Real"
        status_var.set(f"Model: {CURRENT_MODEL_NAME}  |  Mode: {mode}")

    update_status()

    # Main horizontal split
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # Left column: Model + Downloads + Model controls
    left_frame = tk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

    # Right column: Generation + Output + Inspector
    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

    # ----- Model Queue Frame -----
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

    # ----- Model Control Frame -----
    model_ctrl_frame = tk.LabelFrame(left_frame, text="Model Control", padx=5, pady=5)
    model_ctrl_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 5))

    force_fallback_btn = tk.Button(model_ctrl_frame, text="Force Fallback Model", width=20)
    force_fallback_btn.pack(side=tk.TOP, fill=tk.X, pady=2)

    clear_model_btn = tk.Button(model_ctrl_frame, text="Clear Current Model", width=20)
    clear_model_btn.pack(side=tk.TOP, fill=tk.X, pady=2)

    inspect_model_btn = tk.Button(model_ctrl_frame, text="Inspect Model", width=20)
    inspect_model_btn.pack(side=tk.TOP, fill=tk.X, pady=2)

    # ----- Downloads Frame -----
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

    # ----- Generation Frame -----
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

    # ----- Output Frame -----
    out_frame = tk.LabelFrame(right_frame, text="Output", padx=5, pady=5)
    out_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    output_box = scrolledtext.ScrolledText(out_frame, height=14)
    output_box.pack(fill=tk.BOTH, expand=True)

    # ----- Model Inspector Frame -----
    inspector_frame = tk.LabelFrame(right_frame, text="Model Inspector", padx=5, pady=5)
    inspector_frame.pack(fill=tk.BOTH, expand=False, pady=(5, 0))

    inspector_box = scrolledtext.ScrolledText(inspector_frame, height=8)
    inspector_box.pack(fill=tk.BOTH, expand=True)

    # ----- Logging helper -----
    def log(text):
        output_box.insert(tk.END, text + "\n")
        output_box.see(tk.END)

    def inspector_log(text):
        inspector_box.insert(tk.END, text + "\n")
        inspector_box.see(tk.END)

    # ----- Callbacks -----

    def on_auto():
        global MODEL_QUEUE_INDEX
        MODEL_QUEUE_INDEX = 0
        log("Auto loading model from queue...")
        ok = load_llama_with_forklift_from_queue()
        if ok:
            log("Model loaded successfully.")
        else:
            log("All remote models failed; using emergency fallback.")
        update_status()

    def on_retry():
        global MODEL_QUEUE_INDEX
        MODEL_QUEUE_INDEX = 0
        log("Retrying model queue...")
        ok = load_llama_with_forklift_from_queue()
        if ok:
            log("Model loaded successfully.")
        else:
            log("All remote models failed; using emergency fallback.")
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

    def on_gen():
        prompt = prompt_box.get("1.0", tk.END).strip()
        if not prompt:
            log("Prompt is empty.")
            return
        log("Generating...")
        text, stats, _ = generate_with_forklift(prompt, max_new_tokens=64, collect_router_data=False)
        log("=== Output ===")
        log(text)
        log("=== Stats ===")
        for k, v in stats.items():
            log(f"{k}: {v}")
        update_status()

    def on_train():
        prompt = prompt_box.get("1.0", tk.END).strip()
        if not prompt:
            log("Prompt is empty.")
            return
        log("Collecting router data via generation...")
        _ = generate_with_forklift(prompt, max_new_tokens=128, collect_router_data=True)
        log("Training router...")
        train_router_on_collected_data()
        log("Router training complete.")
        update_status()

    def on_force_fallback():
        log("Forcing fallback model...")
        force_fallback_model()
        log("Fallback model active.")
        update_status()

    def on_clear_model():
        log("Clearing current model...")
        clear_current_model()
        log("Model cleared. Next generation will auto-load fallback.")
        update_status()

    def on_inspect_model():
        inspector_box.delete("1.0", tk.END)
        ensure_model_loaded()
        inspector_log(f"Model name: {CURRENT_MODEL_NAME}")
        inspector_log(f"Mode: {'Fallback' if IS_FALLBACK_MODEL else 'Real'}")
        try:
            total_params = sum(p.numel() for p in CURRENT_MODEL.parameters())
            trainable_params = sum(p.numel() for p in CURRENT_MODEL.parameters() if p.requires_grad)
            inspector_log(f"Total params: {total_params:,}")
            inspector_log(f"Trainable params: {trainable_params:,}")
        except Exception as e:
            inspector_log(f"Param inspection failed: {e}")

        inspector_log(f"Device: {next(CURRENT_MODEL.parameters()).device}")
        inspector_log(f"Quant mode: {QUANT_MODE}")
        stats = EXECUTOR.stats()
        inspector_log("--- Forklift Stats (last run) ---")
        for k, v in stats.items():
            inspector_log(f"{k}: {v}")
        inspector_log("--- Top Layers by bytes moved/skipped ---")
        for name, flops, bytes_moved, skipped, fpb, skip_ratio in EXECUTOR.per_layer_stats()[:10]:
            inspector_log(
                f"{name}: bytes={bytes_moved}, skipped={skipped}, fpb={fpb:.2f}, skip_ratio={skip_ratio:.2f}"
            )

    # Bind buttons
    auto_btn.config(command=on_auto)
    retry_btn.config(command=on_retry)
    browse_btn.config(command=on_browse)
    dl_button.config(command=on_download)
    gen_btn.config(command=on_gen)
    train_btn.config(command=on_train)
    force_fallback_btn.config(command=on_force_fallback)
    clear_model_btn.config(command=on_clear_model)
    inspect_model_btn.config(command=on_inspect_model)

    # Progress bar pulsing when download running
    def tick():
        if DOWNLOAD_RUNNING:
            val = prog_var.get()
            val = (val + 3) % 100
            prog_var.set(val)
        root.after(200, tick)

    tick()
    root.mainloop()


# =========================
# Entry
# =========================

def run_gui():
    gui_tkinter()


if __name__ == "__main__":
    print(f"CUDA available: {HAS_CUDA}, GPUs: {NUM_GPUS}, QUANT_MODE={QUANT_MODE}")
    run_gui()
