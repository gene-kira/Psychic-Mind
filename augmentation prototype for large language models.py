"""
Dual-Forklift LLM Cockpit HUD (Tkinter Edition)
------------------------------------------------
This is a full rewrite of the DearPyGUI cockpit using Tkinter,
because the user requested stability and DearPyGUI was crashing.

Features:
- Dual forklifts running in parallel (cache A vs cache B)
- Shared warehouse
- Tile skipping (activation-based)
- Per-layer stats
- Per-core output norms
- Live telemetry
- Autoregressive token loop
- Tkinter GUI (stable, built-in)
- Autoloader for required libraries
"""

# ============================================================
# AUTOLOADER
# ============================================================

try:
    import numpy as np
except ImportError:
    print("Missing dependency: numpy")
    print("Install with: pip install numpy")
    raise SystemExit

import tkinter as tk
from tkinter import ttk
import threading
import time


# ============================================================
# MODEL CONFIG
# ============================================================

TILE_ROWS = 128
TILE_COLS = 128

MODEL_INPUT_DIM = 512
MODEL_HIDDEN_DIM = 1024
NUM_DENSE_LAYERS = 3
NUM_EXPERTS = 4
TOP_K_EXPERTS = 2
NUM_CORES = 3

ACTIVATION_SKIP_THRESHOLD = 1e-3


def matvec_flops(m, n):
    return 2 * m * n


# ============================================================
# UTILITY: TILING
# ============================================================

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


# ============================================================
# WAREHOUSE
# ============================================================

class WeightWarehouse:
    def __init__(self):
        self.layers = {}

    def add_layer(self, name, W):
        self.layers[name] = W

    def get_layer_shape(self, name):
        return self.layers[name].shape

    def get_tile(self, name, tr, tc, tile_rows, tile_cols):
        W = self.layers[name]
        rs, cs = tile_slice(tr, tc, tile_rows, tile_cols)
        return W[rs, cs]


# ============================================================
# TILE CACHE
# ============================================================

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
        if new_size:
            self.max_tiles = new_size
        self.cache.clear()
        self.order.clear()
        self.hits = 0
        self.misses = 0
        self.bytes_moved = 0


# ============================================================
# FORKLIFT
# ============================================================

class Forklift:
    def __init__(self, warehouse, cache_size, name):
        self.name = name
        self.wh = warehouse
        self.cache = TileCache(cache_size)

        self.total_flops = 0
        self.layer_flops = {}
        self.layer_bytes = {}
        self.layer_skipped = {}

        self.tiles_considered = 0
        self.tiles_skipped = 0
        self.bytes_avoided = 0

    def reset(self, cache_size):
        self.cache.reset(cache_size)
        self.total_flops = 0
        self.layer_flops.clear()
        self.layer_bytes.clear()
        self.layer_skipped.clear()
        self.tiles_considered = 0
        self.tiles_skipped = 0
        self.bytes_avoided = 0

    def matvec(self, layer, x):
        rows, cols = self.wh.get_layer_shape(layer)
        y = np.zeros(rows, dtype=np.float32)

        for tr, tc in compute_tile_indices((rows, cols), TILE_ROWS, TILE_COLS):
            rs, cs = tile_slice(tr, tc, TILE_ROWS, TILE_COLS)
            x_sub = x[cs]

            self.tiles_considered += 1

            # Skip tile if activation is tiny
            if np.linalg.norm(x_sub) < ACTIVATION_SKIP_THRESHOLD:
                m = min(TILE_ROWS, rows - rs.start)
                n = min(TILE_COLS, cols - cs.start)
                bytes_est = m * n * 4
                self.tiles_skipped += 1
                self.bytes_avoided += bytes_est
                self.layer_skipped[layer] = self.layer_skipped.get(layer, 0) + bytes_est
                continue

            key = (layer, tr, tc)
            tile = self.cache.get(key)
            if tile is None:
                tile = self.wh.get_tile(layer, tr, tc, TILE_ROWS, TILE_COLS)
                self.cache.put(key, tile)
                self.layer_bytes[layer] = self.layer_bytes.get(layer, 0) + tile.nbytes

            flops = matvec_flops(tile.shape[0], tile.shape[1])
            self.total_flops += flops
            self.layer_flops[layer] = self.layer_flops.get(layer, 0) + flops

            y[rs] += tile @ x_sub

        return y

    def matvec_moe(self, base, experts, x):
        y = np.zeros(MODEL_HIDDEN_DIM, dtype=np.float32)
        for e in experts:
            y += self.matvec(f"{base}_{e}", x)
        return y

    def stats(self):
        hits = self.cache.hits
        misses = self.cache.misses
        total = hits + misses
        hit_rate = hits / total if total else 0

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


# ============================================================
# ROUTER
# ============================================================

class ExpertRouter:
    def __init__(self):
        self.G = np.random.randn(NUM_EXPERTS, MODEL_HIDDEN_DIM).astype(np.float32)

    def route(self, x):
        scores = self.G @ x
        return list(np.argsort(scores)[-TOP_K_EXPERTS:])


# ============================================================
# CORE
# ============================================================

class Core:
    def __init__(self, name, forklift, router):
        self.name = name
        self.forklift = forklift
        self.router = router

    def forward(self, x):
        h = self.forklift.matvec("L0", x)
        h = np.maximum(h, 0)

        experts = self.router.route(h)
        h = self.forklift.matvec_moe("L_moe_expert", experts, h)
        h = np.maximum(h, 0)

        for i in range(1, NUM_DENSE_LAYERS):
            h = self.forklift.matvec(f"L{i}", h)
            h = np.maximum(h, 0)

        return h


# ============================================================
# BUILD MODEL
# ============================================================

def build_warehouse():
    wh = WeightWarehouse()
    wh.add_layer("L0", np.random.randn(MODEL_HIDDEN_DIM, MODEL_INPUT_DIM).astype(np.float32))
    for i in range(1, NUM_DENSE_LAYERS):
        wh.add_layer(f"L{i}", np.random.randn(MODEL_HIDDEN_DIM, MODEL_HIDDEN_DIM).astype(np.float32))
    for e in range(NUM_EXPERTS):
        wh.add_layer(f"L_moe_expert_{e}", np.random.randn(MODEL_HIDDEN_DIM, MODEL_HIDDEN_DIM).astype(np.float32))
    return wh


def build_forklift_system(warehouse, cache_size, name):
    forklift = Forklift(warehouse, cache_size, name)
    router = ExpertRouter()
    cores = [Core(f"{name}_core{i}", forklift, router) for i in range(NUM_CORES)]
    states = [np.random.randn(MODEL_INPUT_DIM).astype(np.float32) for _ in range(NUM_CORES)]
    return forklift, cores, states


# ============================================================
# SIMULATION WRAPPER
# ============================================================

class ForkliftSim:
    def __init__(self, warehouse, cache_size, name):
        self.name = name
        self.forklift, self.cores, self.states = build_forklift_system(warehouse, cache_size, name)
        self.cache_size = cache_size
        self.core_norms = [0] * NUM_CORES

    def reset(self, new_size):
        self.cache_size = new_size
        self.forklift, self.cores, self.states = build_forklift_system(warehouse, new_size, self.name)

    def step(self):
        new_states = []
        norms = []
        for core, x in zip(self.cores, self.states):
            y = core.forward(x)
            norms.append(float(np.linalg.norm(y)))
            new_states.append(np.tanh(y[:MODEL_INPUT_DIM]))
        self.states = new_states
        self.core_norms = norms
        return self.forklift.stats()


# ============================================================
# TKINTER GUI
# ============================================================

class ForkliftPanel:
    def __init__(self, root, sim, title):
        self.sim = sim

        frame = ttk.LabelFrame(root, text=title)
        frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Cache slider
        self.cache_var = tk.IntVar(value=sim.cache_size)
        ttk.Label(frame, text="Cache size").pack()
        ttk.Scale(frame, from_=16, to=4096, orient="horizontal",
                  variable=self.cache_var, command=self.update_cache).pack(fill="x")

        # Telemetry
        self.text = tk.Text(frame, height=20, width=60)
        self.text.pack()

        # Core gauges
        self.core_bars = []
        for i in range(NUM_CORES):
            bar = ttk.Progressbar(frame, length=200, maximum=1.0)
            bar.pack()
            self.core_bars.append(bar)

    def update_cache(self, _):
        self.sim.reset(self.cache_var.get())

    def update(self):
        stats = self.sim.step()

        # Update telemetry text
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

        # Update core gauges
        max_norm = max(self.sim.core_norms)
        scale = max(max_norm, 1e-6)
        for bar, norm in zip(self.core_bars, self.sim.core_norms):
            bar["value"] = norm / scale


# ============================================================
# MAIN LOOP
# ============================================================

warehouse = build_warehouse()

simA = ForkliftSim(warehouse, 64, "A")
simB = ForkliftSim(warehouse, 512, "B")

root = tk.Tk()
root.title("Dual Forklift LLM Cockpit (Tkinter Edition)")

panelA = ForkliftPanel(root, simA, "Forklift A (small cache)")
panelB = ForkliftPanel(root, simB, "Forklift B (large cache)")


def loop():
    panelA.update()
    panelB.update()
    root.after(50, loop)


root.after(50, loop)
root.mainloop()
