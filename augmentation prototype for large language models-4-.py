"""
Forklift Inference Engine – Ultra‑Dense Engineering Cockpit (Unified File)
--------------------------------------------------------------------------

Features:
- GPU‑aware (uses CUDA if available)
- Tiled matmul with global cache (multi‑layer reuse)
- Activation‑aware tile skipping
- Learned tile importance router (stub, but wired in)
- Quantization hooks (INT8 tiles)
- Tiny Transformer with attention + MLP
- KV‑cache skipping hook points
- Multi‑GPU awareness hook (single‑device core)
- Tkinter ultra‑dense cockpit:
    - Telemetry (hits, misses, FLOPs, bytes, skip ratio)
    - Per‑layer stats table
    - Per‑batch output norms (gauges)
    - Cache size slider
    - Router diagnostics (live score + norms)
"""

import sys
import threading
import time

try:
    import numpy as np
except ImportError:
    print("pip install numpy")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("pip install torch")
    sys.exit(1)

import tkinter as tk
from tkinter import ttk


# =========================
# Config
# =========================

WORLD_DEVICES = [d for d in range(torch.cuda.device_count())] or [None]
PRIMARY_DEVICE = torch.device(f"cuda:{WORLD_DEVICES[0]}") if WORLD_DEVICES[0] is not None else torch.device("cpu")

SEQ_LEN = 16
EMBED_DIM = 128
NUM_HEADS = 4
MLP_DIM = 256
NUM_LAYERS = 3
BATCH_SIZE = 4

TILE_ROWS = 64
TILE_COLS = 64

ACTIVATION_SKIP_THRESHOLD = 1e-3


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
# Quantization helpers
# =========================

def quantize_tensor(x: torch.Tensor, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    if x.numel() == 0:
        scale = torch.tensor(1.0, device=x.device)
        q = torch.zeros_like(x, dtype=torch.int8)
        return q, scale
    scale = x.abs().max() / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.clamp((x / scale).round(), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float32) * scale


# =========================
# Global tile cache
# =========================

class TileCache:
    def __init__(self, max_tiles):
        self.max_tiles = max_tiles
        self.cache = {}
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

    def put(self, key, tile):
        if key in self.cache:
            self.order.remove(key)
        self.cache[key] = tile
        self.order.append(key)
        self.bytes_moved += tile.numel() * tile.element_size()
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


GLOBAL_CACHE = TileCache(max_tiles=512)


# =========================
# Tile importance router (stub)
# =========================

class TileImportanceRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)
        with torch.no_grad():
            self.fc.weight.fill_(1.0)
            self.fc.bias.zero_()

    def forward(self, norm, mean, maxv):
        feats = torch.stack([norm, mean, maxv], dim=-1)
        score = self.fc(feats).squeeze(-1)
        return score


TILE_ROUTER = TileImportanceRouter().to(PRIMARY_DEVICE)


def train_tile_router_stub():
    # Placeholder for future training logic
    pass


# =========================
# Forklift executor
# =========================

class ForkliftExecutor:
    def __init__(self, cache: TileCache, router: TileImportanceRouter, quant_bits=8):
        self.cache = cache
        self.router = router
        self.quant_bits = quant_bits

        self.total_flops = 0
        self.layer_flops = {}
        self.layer_bytes = {}
        self.layer_skipped = {}

        self.tiles_considered = 0
        self.tiles_skipped = 0
        self.bytes_avoided = 0

        self.last_router_score = 0.0
        self.last_router_norm = 0.0

    def reset_stats(self, cache_size=None):
        self.cache.reset(cache_size)
        self.total_flops = 0
        self.layer_flops.clear()
        self.layer_bytes.clear()
        self.layer_skipped.clear()
        self.tiles_considered = 0
        self.tiles_skipped = 0
        self.bytes_avoided = 0
        self.last_router_score = 0.0
        self.last_router_norm = 0.0

    def linear(self, layer_name, weight, bias, x):
        W = weight.to(PRIMARY_DEVICE)
        B = bias.to(PRIMARY_DEVICE) if bias is not None else None
        X = x.to(PRIMARY_DEVICE)

        out_dim, in_dim = W.shape
        batch = X.shape[0]
        Y = torch.zeros(batch, out_dim, device=PRIMARY_DEVICE)

        for tr, tc in compute_tile_indices(W.shape, TILE_ROWS, TILE_COLS):
            rs, cs = tile_slice(tr, tc, TILE_ROWS, TILE_COLS)
            X_sub = X[:, cs]

            self.tiles_considered += 1

            norm = X_sub.norm()
            mean = X_sub.mean()
            maxv = X_sub.abs().max()

            with torch.no_grad():
                score = self.router(norm.view(1), mean.view(1), maxv.view(1))[0].item()
            self.last_router_score = score
            self.last_router_norm = norm.item()

            effective_thresh = ACTIVATION_SKIP_THRESHOLD * (1.0 + max(0.0, -score))

            m = min(TILE_ROWS, out_dim - rs.start)
            n = min(TILE_COLS, in_dim - cs.start)
            tile_bytes = m * n * W.element_size()

            if norm.item() < effective_thresh:
                self.tiles_skipped += 1
                self.bytes_avoided += tile_bytes
                self.layer_skipped[layer_name] = self.layer_skipped.get(layer_name, 0) + tile_bytes
                continue

            key = (layer_name, tr, tc)
            cached = self.cache.get(key)
            if cached is None:
                tile = W[rs, cs].contiguous()
                q_tile, scale = quantize_tensor(tile, num_bits=self.quant_bits)
                cached = (q_tile, scale)
                self.cache.put(key, q_tile)
                self.layer_bytes[layer_name] = self.layer_bytes.get(layer_name, 0) + q_tile.numel() * q_tile.element_size()
            else:
                q_tile = cached
                tile = W[rs, cs].contiguous()
                q_tile, scale = quantize_tensor(tile, num_bits=self.quant_bits)

            tile = dequantize_tensor(q_tile, scale)

            partial = X_sub @ tile.t()
            Y[:, rs] += partial

            flops = matvec_flops(tile.shape[0], tile.shape[1]) * batch
            self.total_flops += flops
            self.layer_flops[layer_name] = self.layer_flops.get(layer_name, 0) + flops

        if B is not None:
            Y += B

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
            "router_score": self.last_router_score,
            "router_norm": self.last_router_norm,
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


EXECUTOR = ForkliftExecutor(GLOBAL_CACHE, TILE_ROUTER)


# =========================
# Tiny Transformer with KV-cache hooks
# =========================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if kv_cache is not None:
            # KV-cache skipping hook (placeholder)
            pass

        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (D ** 0.5)
        att = F.softmax(att, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.blocks = nn.ModuleList([
            TransformerBlock(EMBED_DIM, NUM_HEADS, MLP_DIM)
            for _ in range(NUM_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx, kv_cache=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, kv_cache=kv_cache)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


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
        if x.dim() == 3:
            B, T, C = x.shape
            x_flat = x.view(B * T, C)
            y_flat = self.exec.linear(self.name, self.weight, self.bias, x_flat)
            return y_flat.view(B, T, -1)
        else:
            return self.exec.linear(self.name, self.weight, self.bias, x)


def wrap_linear_modules(model: nn.Module, executor: ForkliftExecutor, prefix=""):
    for name, module in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            setattr(model, name, ForkliftLinear(module, full_name, executor))
        else:
            wrap_linear_modules(module, executor, full_name)


# =========================
# Multi-GPU hook (stub)
# =========================

def shard_model_across_gpus(model: nn.Module):
    model.to(PRIMARY_DEVICE)
    return model


# =========================
# Tkinter Ultra‑Dense Cockpit
# =========================

class UltraDenseCockpit:
    def __init__(self, root, model, executor):
        self.root = root
        self.model = model.to(PRIMARY_DEVICE)
        self.exec = executor

        self.running = True

        root.title("Forklift Inference Engine – Ultra‑Dense Cockpit")

        main = ttk.Frame(root)
        main.pack(fill="both", expand=True, padx=5, pady=5)

        # Left column: Telemetry + Router diagnostics
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Telemetry").pack(anchor="w")
        self.telemetry_text = tk.Text(left, width=60, height=20)
        self.telemetry_text.pack(fill="both", expand=True)

        ttk.Label(left, text="Router diagnostics").pack(anchor="w")
        router_frame = ttk.Frame(left)
        router_frame.pack(fill="x")

        ttk.Label(router_frame, text="Last router score:").grid(row=0, column=0, sticky="w")
        self.router_score_var = tk.StringVar(value="0.0")
        ttk.Label(router_frame, textvariable=self.router_score_var).grid(row=0, column=1, sticky="w")

        ttk.Label(router_frame, text="Last router norm:").grid(row=1, column=0, sticky="w")
        self.router_norm_var = tk.StringVar(value="0.0")
        ttk.Label(router_frame, textvariable=self.router_norm_var).grid(row=1, column=1, sticky="w")

        # Cache size slider
        ttk.Label(left, text="Global cache size (tiles)").pack(anchor="w")
        self.cache_var = tk.IntVar(value=self.exec.cache.max_tiles)
        ttk.Scale(left, from_=32, to=4096, orient="horizontal",
                  variable=self.cache_var, command=self.on_cache_change).pack(fill="x")

        # Right column: Per-layer stats + core norms
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="Per-layer stats").pack(anchor="w")
        self.layer_tree = ttk.Treeview(right, columns=("flops", "bytes", "skipped", "fpb", "skip"),
                                       show="headings", height=15)
        for col, txt in zip(("flops", "bytes", "skipped", "fpb", "skip"),
                            ("FLOPs", "Bytes", "Skipped", "FLOPs/byte", "Skip ratio")):
            self.layer_tree.heading(col, text=txt)
        self.layer_tree.pack(fill="both", expand=True)

        ttk.Label(right, text="Per-batch output norms").pack(anchor="w")
        self.core_bars = []
        self.core_labels = []
        for i in range(BATCH_SIZE):
            row = ttk.Frame(right)
            row.pack(fill="x")
            lbl = ttk.Label(row, text=f"Core {i}:")
            lbl.pack(side="left")
            bar = ttk.Progressbar(row, length=200, maximum=1.0)
            bar.pack(side="left", fill="x", expand=True)
            self.core_bars.append(bar)
            self.core_labels.append(lbl)

        self.core_norms = [0.0] * BATCH_SIZE

        self.loop()

    def on_cache_change(self, _):
        new_size = int(self.cache_var.get())
        self.exec.reset_stats(new_size)

    def loop(self):
        if not self.running:
            return

        idx = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=PRIMARY_DEVICE)
        with torch.no_grad():
            logits = self.model(idx)

        norms = logits.norm(dim=(1, 2)).detach().cpu().numpy().tolist()
        self.core_norms = norms

        stats = self.exec.stats()

        self.telemetry_text.delete("1.0", "end")
        self.telemetry_text.insert("end", f"Device: {PRIMARY_DEVICE}\n")
        self.telemetry_text.insert("end", f"Hits: {stats['hits']}\n")
        self.telemetry_text.insert("end", f"Misses: {stats['misses']}\n")
        self.telemetry_text.insert("end", f"Hit rate: {stats['hit_rate']:.4f}\n")
        self.telemetry_text.insert("end", f"Bytes moved: {stats['bytes_moved']/1e6:.2f} MB\n")
        self.telemetry_text.insert("end", f"FLOPs: {stats['total_flops']/1e9:.3f} GF\n")
        self.telemetry_text.insert("end", f"FLOPs/byte: {stats['flops_per_byte']:.2f}\n")
        self.telemetry_text.insert("end", f"Tiles considered: {stats['tiles_considered']}\n")
        self.telemetry_text.insert("end", f"Tiles skipped: {stats['tiles_skipped']}\n")
        self.telemetry_text.insert("end", f"Skip ratio: {stats['skip_ratio']:.3f}\n")
        self.telemetry_text.insert("end", f"Bytes avoided: {stats['bytes_avoided']/1e6:.2f} MB\n")

        self.router_score_var.set(f"{stats['router_score']:.4f}")
        self.router_norm_var.set(f"{stats['router_norm']:.4f}")

        for row in self.layer_tree.get_children():
            self.layer_tree.delete(row)
        for name, flops, bytes_moved, skipped, fpb, skip_ratio in self.exec.per_layer_stats():
            self.layer_tree.insert("", "end", values=(
                f"{flops/1e9:.3f}G",
                f"{bytes_moved/1e6:.2f}M",
                f"{skipped/1e6:.2f}M",
                f"{fpb:.2f}",
                f"{skip_ratio:.2f}",
            ))

        max_norm = max(self.core_norms) if self.core_norms else 1.0
        scale = max(max_norm, 1e-6)
        for bar, n in zip(self.core_bars, self.core_norms):
            bar["value"] = n / scale

        self.root.after(50, self.loop)


# =========================
# Main
# =========================

def main():
    base = TinyTransformer()
    wrap_linear_modules(base, EXECUTOR)
    model = shard_model_across_gpus(base)

    root = tk.Tk()
    cockpit = UltraDenseCockpit(root, model, EXECUTOR)
    root.mainloop()


if __name__ == "__main__":
    main()
