"""
Forklift Wrapper Runtime for PyTorch (Tkinter Cockpit)
------------------------------------------------------

- Real PyTorch model (no synthetic matmul).
- Forklift wrapper intercepts nn.Linear layers.
- Tiles real weights, caches tiles, skips tiles based on activations.
- Tracks:
    - bytes moved
    - bytes avoided
    - hits / misses / hit rate
    - FLOPs / FLOPs per byte
    - per-layer stats
- Tkinter cockpit HUD:
    - Live telemetry
    - Per-layer table
    - Per-core (batch element) output norms
- Single unified file.

Run:
    pip install torch numpy
    python this_file.py
"""

# =========================
# Autoloader
# =========================

import sys

try:
    import numpy as np
except ImportError:
    print("Missing dependency: numpy")
    print("Install with: pip install numpy")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Missing dependency: torch")
    print("Install with: pip install torch")
    sys.exit(1)

import tkinter as tk
from tkinter import ttk
import threading
import time


# =========================
# Config
# =========================

TILE_ROWS = 128
TILE_COLS = 128

INPUT_DIM = 512
HIDDEN_DIM = 1024
NUM_LAYERS = 3
BATCH_SIZE = 4

ACTIVATION_SKIP_THRESHOLD = 1e-3


def matvec_flops(m, n):
    return 2 * m * n


# =========================
# Tiling helpers
# =========================

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
# Tile cache
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
        self.bytes_moved += tile.nbytes
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


# =========================
# Forklift executor
# =========================

class ForkliftExecutor:
    """
    Wraps real PyTorch Linear layers and executes them via tiled matmul
    with activation-based tile skipping and caching.
    """

    def __init__(self, cache_size=256):
        self.cache = TileCache(cache_size)
        self.total_flops = 0

        # Per-layer stats
        self.layer_flops = {}
        self.layer_bytes = {}
        self.layer_skipped = {}

        # Global tile stats
        self.tiles_considered = 0
        self.tiles_skipped = 0
        self.bytes_avoided = 0

    def reset_stats(self, cache_size=None):
        self.cache.reset(cache_size)
        self.total_flops = 0
        self.layer_flops.clear()
        self.layer_bytes.clear()
        self.layer_skipped.clear()
        self.tiles_considered = 0
        self.tiles_skipped = 0
        self.bytes_avoided = 0

    def linear(self, layer_name, weight, bias, x):
        """
        weight: torch.Tensor [out, in]
        x:      torch.Tensor [batch, in]
        """
        W = weight.detach().cpu().numpy()
        B = bias.detach().cpu().numpy() if bias is not None else None
        X = x.detach().cpu().numpy()

        out_dim, in_dim = W.shape
        batch = X.shape[0]
        Y = np.zeros((batch, out_dim), dtype=np.float32)

        for tr, tc in compute_tile_indices(W.shape, TILE_ROWS, TILE_COLS):
            rs, cs = tile_slice(tr, tc, TILE_ROWS, TILE_COLS)
            X_sub = X[:, cs]  # [batch, tile_in]

            # Decide if this tile is needed based on activation norm
            self.tiles_considered += 1
            act_norm = float(np.linalg.norm(X_sub))
            m = min(TILE_ROWS, out_dim - rs.start)
            n = min(TILE_COLS, in_dim - cs.start)
            tile_bytes = m * n * 4  # float32

            if act_norm < ACTIVATION_SKIP_THRESHOLD:
                self.tiles_skipped += 1
                self.bytes_avoided += tile_bytes
                self.layer_skipped[layer_name] = self.layer_skipped.get(layer_name, 0) + tile_bytes
                continue

            key = (layer_name, tr, tc)
            tile = self.cache.get(key)
            if tile is None:
                tile = W[rs, cs]
                self.cache.put(key, tile)
                self.layer_bytes[layer_name] = self.layer_bytes.get(layer_name, 0) + tile.nbytes

            # tile: [tile_out, tile_in], X_sub: [batch, tile_in]
            # Y[:, rs] += X_sub @ tile.T
            partial = X_sub @ tile.T
            Y[:, rs] += partial

            flops = matvec_flops(tile.shape[0], tile.shape[1]) * batch
            self.total_flops += flops
            self.layer_flops[layer_name] = self.layer_flops.get(layer_name, 0) + flops

        if B is not None:
            Y += B

        return torch.from_numpy(Y).to(x.device)

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


# =========================
# PyTorch model + wrapper
# =========================

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(nn.Linear(INPUT_DIM, HIDDEN_DIM))
        for _ in range(NUM_LAYERS - 1):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.act(x)
        return x


class ForkliftWrappedMLP(nn.Module):
    """
    Wraps a real PyTorch MLP and routes all Linear layers
    through the ForkliftExecutor.
    """

    def __init__(self, base_model, executor: ForkliftExecutor):
        super().__init__()
        self.base = base_model
        self.exec = executor

    def forward(self, x):
        for idx, layer in enumerate(self.base.layers):
            name = f"L{idx}"
            x = self.exec.linear(name, layer.weight, layer.bias, x)
            x = torch.relu(x)
        return x


# =========================
# Tkinter cockpit
# =========================

class CockpitGUI:
    def __init__(self, root, model, executor):
        self.root = root
        self.model = model
        self.exec = executor

        self.running = True

        root.title("Forklift PyTorch Wrapper Cockpit")

        main = ttk.Frame(root)
        main.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: telemetry
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Telemetry").pack()
        self.text = tk.Text(left, width=60, height=25)
        self.text.pack(fill="both", expand=True)

        # Cache size control
        self.cache_var = tk.IntVar(value=self.exec.cache.max_tiles)
        ttk.Label(left, text="Cache size (tiles)").pack()
        ttk.Scale(left, from_=16, to=4096, orient="horizontal",
                  variable=self.cache_var, command=self.on_cache_change).pack(fill="x")

        # Right: per-layer + core norms
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="Per-layer stats").pack()
        self.layer_tree = ttk.Treeview(right, columns=("flops", "bytes", "skipped", "fpb", "skip"),
                                       show="headings", height=15)
        for col, txt in zip(("flops", "bytes", "skipped", "fpb", "skip"),
                            ("FLOPs", "Bytes", "Skipped", "FLOPs/byte", "Skip ratio")):
            self.layer_tree.heading(col, text=txt)
        self.layer_tree.pack(fill="both", expand=True)

        ttk.Label(right, text="Per-core output norms").pack()
        self.core_bars = []
        for i in range(BATCH_SIZE):
            bar = ttk.Progressbar(right, length=200, maximum=1.0)
            bar.pack()
            self.core_bars.append(bar)

        self.core_norms = [0.0] * BATCH_SIZE

        # Start background loop
        self.loop()

    def on_cache_change(self, _):
        new_size = int(self.cache_var.get())
        self.exec.reset_stats(new_size)

    def loop(self):
        if not self.running:
            return

        # Generate real input and run real model
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        y = self.model(x)

        # Update core norms
        norms = torch.norm(y, dim=1).detach().cpu().numpy().tolist()
        self.core_norms = norms

        # Update telemetry
        stats = self.exec.stats()
        self.text.delete("1.0", "end")
        self.text.insert("end", f"Hits: {stats['hits']}\n")
        self.text.insert("end", f"Misses: {stats['misses']}\n")
        self.text.insert("end", f"Hit rate: {stats['hit_rate']:.4f}\n")
        self.text.insert("end", f"Bytes moved: {stats['bytes_moved']/1e6:.2f} MB\n")
        self.text.insert("end", f"FLOPs: {stats['total_flops']/1e9:.3f} GF\n")
        self.text.insert("end", f"FLOPs/byte: {stats['flops_per_byte']:.2f}\n")
        self.text.insert("end", f"Tiles considered: {stats['tiles_considered']}\n")
        self.text.insert("end", f"Tiles skipped: {stats['tiles_skipped']}\n")
        self.text.insert("end", f"Skip ratio: {stats['skip_ratio']:.3f}\n")
        self.text.insert("end", f"Bytes avoided: {stats['bytes_avoided']/1e6:.2f} MB\n")

        # Update per-layer table
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

        # Update core bars
        max_norm = max(self.core_norms) if self.core_norms else 1.0
        scale = max(max_norm, 1e-6)
        for bar, n in zip(self.core_bars, self.core_norms):
            bar["value"] = n / scale

        # Schedule next step
        self.root.after(50, self.loop)


# =========================
# Main
# =========================

def main():
    base = SimpleMLP()
    executor = ForkliftExecutor(cache_size=256)
    wrapped = ForkliftWrappedMLP(base, executor)

    root = tk.Tk()
    gui = CockpitGUI(root, wrapped, executor)
    root.mainloop()


if __name__ == "__main__":
    main()
