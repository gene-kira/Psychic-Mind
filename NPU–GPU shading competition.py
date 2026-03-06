"""
Alien Dark-Neon NPU–GPU Cockpit
- Tkinter GUI (heatmap + controls)
- Predictive NPU–GPU shading competition loop
- Multi-NPU swarm (USB + on-chip) manager
- Lineage memory + adaptive thresholds
- Swappable real-organ adapters (GPU, Telemetry, UIAutomation)

Swap Dummy* and enumerate_npus() with your real organs/runtimes.
"""

import importlib
import math
import statistics
import time
from collections import deque
import tkinter as tk
from tkinter import ttk

# ──────────────────────────────────────────────────────────────────────────────
# Autoloader (extend if you want more libs)
# ──────────────────────────────────────────────────────────────────────────────

def autoload_libraries():
    required = []
    loaded = {}
    for name in required:
        try:
            loaded[name] = importlib.import_module(name)
        except ImportError:
            print(f"[AUTOLOADER] Missing library: {name}")
    return loaded

_ = autoload_libraries()

# ──────────────────────────────────────────────────────────────────────────────
# Lineage memory & adaptive thresholds
# ──────────────────────────────────────────────────────────────────────────────

class LineageMemory:
    def __init__(self, max_len=1024):
        self.history = deque(maxlen=max_len)

    def append(self, entry):
        self.history.append(entry)

    def recent_stats(self, key, window=120):
        if not self.history:
            return None
        data = [h[key] for h in list(self.history)[-window:] if key in h]
        if not data:
            return None
        return {
            "mean": statistics.mean(data),
            "max": max(data),
            "min": min(data),
        }


class AdaptiveThresholds:
    def __init__(self):
        self.base = {
            "BEAST": 0.90,
            "AGGRESSIVE": 0.70,
            "NORMAL": 0.50,
            "COAST": 0.30,
        }
        self.delta = {k: 0.0 for k in self.base}

    def get(self, mode):
        mode = mode.upper()
        return max(0.05, min(0.99, self.base.get(mode, 0.50) + self.delta.get(mode, 0.0)))

    def adapt(self, mode, lineage: LineageMemory):
        mode = mode.upper()
        stats = lineage.recent_stats("frame_time", window=120)
        if not stats:
            return
        target_ms = 16.7
        mean_ft = stats["mean"]
        if mean_ft > target_ms * 1.15:
            self.delta[mode] -= 0.01
        elif mean_ft < target_ms * 0.85:
            self.delta[mode] += 0.01
        self.delta[mode] = max(-0.30, min(0.30, self.delta[mode]))

# ──────────────────────────────────────────────────────────────────────────────
# ReplicaNPU-style heads (logic placeholder; you can swap to real calls)
# ──────────────────────────────────────────────────────────────────────────────

class PredictiveHead:
    def forward(self, state):
        ft = state["frame_time"]
        util = state["gpu_util"]
        vram_p = state["vram_pressure"]
        ctx = state["ui_context"]
        ctx_boost = 0.0
        if ctx["shader_compiling"]:
            ctx_boost += 0.5
        if ctx["overlay_active"]:
            ctx_boost += 0.2
        if ctx["is_cutscene"]:
            ctx_boost += 0.1
        return ft * (1.0 + util / 200.0 + vram_p / 300.0 + ctx_boost)


class IntegrityHead:
    def risk(self, cost_pred, state):
        target_ms = 16.7
        ratio = cost_pred / target_ms
        ratio = max(0.0, min(3.0, ratio))
        return ratio / 3.0


class PolicyHead:
    def shading_policy(self, risk, state):
        actions = {
            "resolution_scale": 1.0,
            "lod_bias": 0.0,
            "shader_quality": "HIGH",
            "neural_upscale": False,
            "neural_denoise": False,
            "neural_materials": False,
        }
        if risk < 0.3:
            return actions
        if 0.3 <= risk < 0.6:
            actions["resolution_scale"] = 0.9
            actions["lod_bias"] = 0.25
            actions["shader_quality"] = "MEDIUM"
        elif 0.6 <= risk < 0.85:
            actions["resolution_scale"] = 0.8
            actions["lod_bias"] = 0.5
            actions["shader_quality"] = "LOW"
            actions["neural_upscale"] = True
            actions["neural_denoise"] = True
        else:
            actions["resolution_scale"] = 0.7
            actions["lod_bias"] = 0.75
            actions["shader_quality"] = "VERY_LOW"
            actions["neural_upscale"] = True
            actions["neural_denoise"] = True
            actions["neural_materials"] = True
        return actions

# ──────────────────────────────────────────────────────────────────────────────
# Heatmap buffer
# ──────────────────────────────────────────────────────────────────────────────

class HeatmapBuffer:
    def __init__(self, max_frames=300):
        self.entries = deque(maxlen=max_frames)

    def add(self, frame_id, risk, winner, resolution_scale, shader_quality, mode):
        self.entries.append({
            "frame_id": frame_id,
            "risk": risk,
            "winner": winner,
            "resolution_scale": resolution_scale,
            "shader_quality": shader_quality,
            "mode": mode,
        })

    def export(self):
        return list(self.entries)

# ──────────────────────────────────────────────────────────────────────────────
# Dummy organs (standalone demo; swap with real ones)
# ──────────────────────────────────────────────────────────────────────────────

class DummyGPUImpl:
    def __init__(self):
        self._res_scale = 1.0
        self._lod_bias = 0.0
        self._quality = "HIGH"

    def render_frame(self):
        time.sleep(0.005)

    def utilization(self): return 70.0 + 10.0 * math.sin(time.time())
    def frame_time(self): return 18.0 + 2.0 * math.sin(time.time() * 0.7)
    def frame_time_delta(self): return 0.5
    def vram_pressure(self): return 0.75
    def temperature(self): return 70.0
    def pipeline_stalls(self): return 3
    def shader_stage_times(self): return {"vertex": 1.0, "pixel": 5.0, "compute": 2.0}
    def current_resolution(self): return (2560, 1440)
    def current_lod(self): return self._lod_bias

    def default_plan(self):
        return {"shader_quality": "HIGH"}

    def set_resolution_scale(self, scale): self._res_scale = scale
    def set_lod_bias(self, bias): self._lod_bias = bias
    def set_shader_quality(self, quality): self._quality = quality
    def enable_neural_upscaling(self, flag): ...
    def enable_neural_denoising(self, flag): ...
    def enable_neural_materials(self, flag): ...
    def apply_default_plan(self, plan): ...


class DummyTelemetrySpineImpl:
    def __init__(self):
        self.buffer = deque(maxlen=512)

    def push_gpu_state(self, state):
        self.buffer.append(state)


class DummyUIAutomationImpl:
    def is_gameplay(self): return True
    def is_menu(self): return False
    def is_cutscene(self): return False
    def overlay_active(self): return False
    def shader_compiling(self): return False

# ──────────────────────────────────────────────────────────────────────────────
# Real-organ adapters
# ──────────────────────────────────────────────────────────────────────────────

class GPUOrgan:
    def __init__(self, gpu_impl):
        self.gpu = gpu_impl

    def render_frame(self):
        self.gpu.render_frame()

    def util(self): return self.gpu.utilization()
    def frame_time(self): return self.gpu.frame_time()
    def frame_delta(self): return self.gpu.frame_time_delta()
    def vram_pressure(self): return self.gpu.vram_pressure()
    def temperature(self): return self.gpu.temperature()
    def stalls(self): return self.gpu.pipeline_stalls()
    def shader_times(self): return self.gpu.shader_stage_times()
    def resolution(self): return self.gpu.current_resolution()
    def lod(self): return self.gpu.current_lod()

    def default_plan(self):
        return self.gpu.default_plan()

    def set_resolution_scale(self, scale): self.gpu.set_resolution_scale(scale)
    def set_lod_bias(self, bias): self.gpu.set_lod_bias(bias)
    def set_shader_quality(self, quality): self.gpu.set_shader_quality(quality)
    def enable_neural_upscale(self, flag): self.gpu.enable_neural_upscaling(flag)
    def enable_neural_denoise(self, flag): self.gpu.enable_neural_denoising(flag)
    def enable_neural_materials(self, flag): self.gpu.enable_neural_materials(flag)
    def apply_default_plan(self): self.gpu.apply_default_plan(self.gpu.default_plan())


class TelemetrySpineAdapter:
    def __init__(self, spine_impl):
        self.spine = spine_impl

    def push(self, state):
        self.spine.push_gpu_state(state)


class UIAutomationOrgan:
    def __init__(self, ui_impl):
        self.ui = ui_impl

    def is_gameplay(self): return self.ui.is_gameplay()
    def is_menu(self): return self.ui.is_menu()
    def is_cutscene(self): return self.ui.is_cutscene()
    def overlay_active(self): return self.ui.overlay_active()
    def shader_compiling(self): return self.ui.shader_compiling()

# ──────────────────────────────────────────────────────────────────────────────
# Multi-NPU swarm (USB + on-chip)
# ──────────────────────────────────────────────────────────────────────────────

class NPUNode:
    def __init__(self, device_id, backend):
        self.device_id = device_id
        self.backend = backend
        self.predictive_head = PredictiveHead()
        self.integrity_head = IntegrityHead()
        self.policy_head = PolicyHead()

    def infer(self, state):
        # In your real system, ship 'state' to backend and run model there.
        cost_pred = self.predictive_head.forward(state)
        risk = self.integrity_head.risk(cost_pred, state)
        actions = self.policy_head.shading_policy(risk, state)
        return {
            "device_id": self.device_id,
            "cost_pred": cost_pred,
            "risk": risk,
            "actions": actions,
        }


class MultiNPUManager:
    def __init__(self, device_enumerator):
        self.nodes = []
        self._discover(device_enumerator)

    def _discover(self, device_enumerator):
        devices = device_enumerator()
        for dev_id, backend, kind in devices:
            print(f"[NPU] Found {kind} NPU: {dev_id}")
            self.nodes.append(NPUNode(dev_id, backend))

    def infer_swarm(self, state):
        if not self.nodes:
            return 0.0, {
                "resolution_scale": 1.0,
                "lod_bias": 0.0,
                "shader_quality": "HIGH",
                "neural_upscale": False,
                "neural_denoise": False,
                "neural_materials": False,
            }, []
        outputs = [node.infer(state) for node in self.nodes]
        avg_risk = statistics.mean(o["risk"] for o in outputs)
        agg_actions = self._aggregate_actions(outputs)
        return avg_risk, agg_actions, outputs

    def _aggregate_actions(self, outputs):
        res_scales = [o["actions"]["resolution_scale"] for o in outputs]
        lod_biases = [o["actions"]["lod_bias"] for o in outputs]
        qualities = [o["actions"]["shader_quality"] for o in outputs]

        def q_rank(q):
            order = ["VERY_LOW", "LOW", "MEDIUM", "HIGH"]
            return order.index(q) if q in order else 2

        worst_quality = min(qualities, key=q_rank)
        nu = any(o["actions"]["neural_upscale"] for o in outputs)
        nd = any(o["actions"]["neural_denoise"] for o in outputs)
        nm = any(o["actions"]["neural_materials"] for o in outputs)

        return {
            "resolution_scale": min(res_scales),
            "lod_bias": max(lod_biases),
            "shader_quality": worst_quality,
            "neural_upscale": nu,
            "neural_denoise": nd,
            "neural_materials": nm,
        }


class ReplicaNPUSwarm:
    def __init__(self, multi_npu_manager: MultiNPUManager):
        self.multi = multi_npu_manager
        self.lineage = LineageMemory()
        self.thresholds = AdaptiveThresholds()

    def step(self, state, mode):
        avg_risk, agg_actions, node_outputs = self.multi.infer_swarm(state)
        self.thresholds.adapt(mode, self.lineage)
        thr = self.thresholds.get(mode)
        return avg_risk, thr, agg_actions, node_outputs

# ──────────────────────────────────────────────────────────────────────────────
# Cockpit state
# ──────────────────────────────────────────────────────────────────────────────

class CockpitState:
    def __init__(self):
        self._mode = "NORMAL"
        self.force_gpu = False
        self.force_npu = False

    def mode(self): return self._mode
    def set_mode(self, m): self._mode = m.upper()

# ──────────────────────────────────────────────────────────────────────────────
# Predictive swarm loop
# ──────────────────────────────────────────────────────────────────────────────

def predictive_loop_swarm(frame_id, gpu: GPUOrgan, telemetry: TelemetrySpineAdapter,
                          replica_swarm: ReplicaNPUSwarm, cockpit: CockpitState,
                          ui_auto: UIAutomationOrgan, heatmap: HeatmapBuffer):
    gpu.render_frame()

    state = {
        "gpu_util": gpu.util(),
        "frame_time": gpu.frame_time(),
        "frame_delta": gpu.frame_delta(),
        "vram_pressure": gpu.vram_pressure(),
        "temperature": gpu.temperature(),
        "stalls": gpu.stalls(),
        "shader_times": gpu.shader_times(),
        "resolution": gpu.resolution(),
        "lod": gpu.lod(),
        "ui_context": {
            "is_gameplay": ui_auto.is_gameplay(),
            "is_menu": ui_auto.is_menu(),
            "is_cutscene": ui_auto.is_cutscene(),
            "overlay_active": ui_auto.overlay_active(),
            "shader_compiling": ui_auto.shader_compiling(),
        }
    }
    telemetry.push(state)

    mode = cockpit.mode()
    avg_risk, threshold, actions, node_outputs = replica_swarm.step(state, mode)

    if cockpit.force_gpu:
        winner = "GPU"
    elif cockpit.force_npu:
        winner = "NPU"
    else:
        winner = "NPU" if avg_risk >= threshold else "GPU"

    if winner == "NPU":
        gpu.set_resolution_scale(actions["resolution_scale"])
        gpu.set_lod_bias(actions["lod_bias"])
        gpu.set_shader_quality(actions["shader_quality"])
        gpu.enable_neural_upscale(actions["neural_upscale"])
        gpu.enable_neural_denoise(actions["neural_denoise"])
        gpu.enable_neural_materials(actions["neural_materials"])
    else:
        gpu.apply_default_plan()

    replica_swarm.lineage.append({
        "frame": frame_id,
        "risk": avg_risk,
        "winner": winner,
        "frame_time": state["frame_time"],
        "vram_pressure": state["vram_pressure"],
        "stalls": state["stalls"],
    })

    heatmap.add(
        frame_id=frame_id,
        risk=avg_risk,
        winner=winner,
        resolution_scale=actions["resolution_scale"] if winner == "NPU" else 1.0,
        shader_quality=actions["shader_quality"] if winner == "NPU" else "HIGH",
        mode=mode,
    )

    return {
        "frame": frame_id,
        "risk": avg_risk,
        "threshold": threshold,
        "winner": winner,
        "actions": actions,
        "state": state,
        "nodes": node_outputs,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Dark neon Tkinter cockpit
# ──────────────────────────────────────────────────────────────────────────────

class AlienCockpit(tk.Tk):
    def __init__(self, gpu, telemetry, replica_swarm, cockpit_state, ui_auto, heatmap):
        super().__init__()
        self.title("Alien NPU–GPU Cockpit")
        self.configure(bg="#050510")
        self.geometry("1100x700")

        self.gpu = gpu
        self.telemetry = telemetry
        self.replica_swarm = replica_swarm
        self.cockpit_state = cockpit_state
        self.ui_auto = ui_auto
        self.heatmap = heatmap

        self.frame_id = 0
        self.running = True

        self._build_style()
        self._build_layout()
        self.after(50, self._tick)

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Neon.TButton",
                        background="#101020",
                        foreground="#66ffcc",
                        borderwidth=1,
                        focusthickness=3,
                        focuscolor="#00ffaa")
        style.map("Neon.TButton",
                  background=[("active", "#202040")])
        style.configure("Neon.TLabel",
                        background="#050510",
                        foreground="#66ffcc")

    def _build_layout(self):
        self.canvas = tk.Canvas(self, bg="#050510", highlightthickness=0, height=320)
        self.canvas.pack(fill="x", padx=10, pady=10)

        bottom = tk.Frame(self, bg="#050510")
        bottom.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.Frame(bottom, bg="#050510")
        left.pack(side="left", fill="y", padx=(0, 10))

        right = tk.Frame(bottom, bg="#050510")
        right.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="MODE", fg="#66ffcc", bg="#050510",
                 font=("Consolas", 12, "bold")).pack(pady=(0, 5))
        for mode in ["BEAST", "AGGRESSIVE", "NORMAL", "COAST"]:
            b = ttk.Button(left, text=mode, style="Neon.TButton",
                           command=lambda m=mode: self._set_mode(m))
            b.pack(fill="x", pady=2)

        tk.Label(left, text="OVERRIDES", fg="#66ffcc", bg="#050510",
                 font=("Consolas", 12, "bold")).pack(pady=(10, 5))
        self.force_gpu_var = tk.BooleanVar(value=False)
        self.force_npu_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(left, text="Force GPU", style="Neon.TButton",
                        command=self._toggle_force_gpu,
                        variable=self.force_gpu_var).pack(fill="x", pady=2)
        ttk.Checkbutton(left, text="Force NPU", style="Neon.TButton",
                        command=self._toggle_force_npu,
                        variable=self.force_npu_var).pack(fill="x", pady=2)

        ttk.Button(left, text="Reset Thresholds", style="Neon.TButton",
                   command=self._reset_thresholds).pack(fill="x", pady=(10, 2))

        tk.Label(right, text="TELEMETRY", fg="#66ffcc", bg="#050510",
                 font=("Consolas", 12, "bold")).pack(anchor="w")

        self.lbl_mode = tk.Label(right, text="Mode: NORMAL", bg="#050510",
                                 fg="#66ffcc", font=("Consolas", 10))
        self.lbl_mode.pack(anchor="w")

        self.lbl_risk = tk.Label(right, text="Risk: 0.00", bg="#050510",
                                 fg="#ff66aa", font=("Consolas", 10))
        self.lbl_risk.pack(anchor="w")

        self.lbl_threshold = tk.Label(right, text="Threshold: 0.50", bg="#050510",
                                      fg="#ffaa66", font=("Consolas", 10))
        self.lbl_threshold.pack(anchor="w")

        self.lbl_winner = tk.Label(right, text="Winner: GPU", bg="#050510",
                                   fg="#66aaff", font=("Consolas", 10))
        self.lbl_winner.pack(anchor="w")

        self.lbl_ft = tk.Label(right, text="Frame Time: -- ms", bg="#050510",
                               fg="#66ffcc", font=("Consolas", 10))
        self.lbl_ft.pack(anchor="w")

        self.lbl_util = tk.Label(right, text="GPU Util: -- %", bg="#050510",
                                 fg="#66ffcc", font=("Consolas", 10))
        self.lbl_util.pack(anchor="w")

        self.lbl_vram = tk.Label(right, text="VRAM Pressure: --", bg="#050510",
                                 fg="#66ffcc", font=("Consolas", 10))
        self.lbl_vram.pack(anchor="w")

        self.lbl_ui = tk.Label(right, text="UI Context: gameplay", bg="#050510",
                               fg="#66ffcc", font=("Consolas", 10))
        self.lbl_ui.pack(anchor="w")

    def _set_mode(self, mode):
        self.cockpit_state.set_mode(mode)
        self.lbl_mode.config(text=f"Mode: {mode}")

    def _toggle_force_gpu(self):
        val = self.force_gpu_var.get()
        self.cockpit_state.force_gpu = val
        if val:
            self.force_npu_var.set(False)
            self.cockpit_state.force_npu = False

    def _toggle_force_npu(self):
        val = self.force_npu_var.get()
        self.cockpit_state.force_npu = val
        if val:
            self.force_gpu_var.set(False)
            self.cockpit_state.force_gpu = False

    def _reset_thresholds(self):
        self.replica_swarm.thresholds = AdaptiveThresholds()

    def _tick(self):
        if not self.running:
            return
        self.frame_id += 1
        snapshot = predictive_loop_swarm(
            frame_id=self.frame_id,
            gpu=self.gpu,
            telemetry=self.telemetry,
            replica_swarm=self.replica_swarm,
            cockpit=self.cockpit_state,
            ui_auto=self.ui_auto,
            heatmap=self.heatmap,
        )
        self._update_telemetry(snapshot)
        self._draw_heatmap()
        self.after(33, self._tick)

    def _update_telemetry(self, snap):
        risk = snap["risk"]
        thr = snap["threshold"]
        winner = snap["winner"]
        state = snap["state"]

        self.lbl_risk.config(text=f"Risk: {risk:.2f}")
        self.lbl_threshold.config(text=f"Threshold: {thr:.2f}")
        self.lbl_winner.config(text=f"Winner: {winner}",
                               fg="#ff4466" if winner == "NPU" else "#66aaff")
        self.lbl_ft.config(text=f"Frame Time: {state['frame_time']:.2f} ms")
        self.lbl_util.config(text=f"GPU Util: {state['gpu_util']:.1f} %")
        self.lbl_vram.config(text=f"VRAM Pressure: {state['vram_pressure']:.2f}")
        ctx = state["ui_context"]
        ctx_str = "gameplay" if ctx["is_gameplay"] else "menu" if ctx["is_menu"] else "cutscene" if ctx["is_cutscene"] else "unknown"
        if ctx["overlay_active"]:
            ctx_str += " + overlay"
        if ctx["shader_compiling"]:
            ctx_str += " + compiling"
        self.lbl_ui.config(text=f"UI Context: {ctx_str}")

    def _draw_heatmap(self):
        self.canvas.delete("all")
        entries = self.heatmap.export()
        if not entries:
            return

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        cols = len(entries)
        if cols == 0:
            return

        col_w = max(2, width / cols)
        row_h = height / 4.0

        for i, e in enumerate(entries):
            x0 = i * col_w
            x1 = x0 + col_w

            # Row 1: risk
            r = e["risk"]
            r = max(0.0, min(1.0, r))
            if r < 0.5:
                t = r / 0.5
                color1 = _blend("#000000", "#ffff00", t)
            else:
                t = (r - 0.5) / 0.5
                color1 = _blend("#ffff00", "#ff0000", t)
            self.canvas.create_rectangle(x0, 0 * row_h, x1, 1 * row_h,
                                         fill=color1, outline="")

            # Row 2: winner
            color2 = "#66aaff" if e["winner"] == "GPU" else "#ff4466"
            self.canvas.create_rectangle(x0, 1 * row_h, x1, 2 * row_h,
                                         fill=color2, outline="")

            # Row 3: resolution scale
            rs = e["resolution_scale"]
            rs = max(0.6, min(1.0, rs))
            t = (rs - 0.6) / 0.4
            color3 = _blend("#004400", "#00ff88", t)
            self.canvas.create_rectangle(x0, 2 * row_h, x1, 3 * row_h,
                                         fill=color3, outline="")

            # Row 4: shader quality
            q = e["shader_quality"]
            if q == "HIGH":
                color4 = "#00ffff"
            elif q == "MEDIUM":
                color4 = "#00aa99"
            elif q == "LOW":
                color4 = "#8844ff"
            else:
                color4 = "#ff00ff"
            self.canvas.create_rectangle(x0, 3 * row_h, x1, 4 * row_h,
                                         fill=color4, outline="")

        self.canvas.create_text(5, row_h * 0.5, text="RISK", anchor="w",
                                fill="#66ffcc", font=("Consolas", 8))
        self.canvas.create_text(5, row_h * 1.5, text="WIN", anchor="w",
                                fill="#66ffcc", font=("Consolas", 8))
        self.canvas.create_text(5, row_h * 2.5, text="RES", anchor="w",
                                fill="#66ffcc", font=("Consolas", 8))
        self.canvas.create_text(5, row_h * 3.5, text="QUAL", anchor="w",
                                fill="#66ffcc", font=("Consolas", 8))


def _blend(c1, c2, t):
    t = max(0.0, min(1.0, t))
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def rgb_to_hex(r, g, b):
        return f"#{r:02x}{g:02x}{b:02x}"
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return rgb_to_hex(r, g, b)

# ──────────────────────────────────────────────────────────────────────────────
# NPU enumeration stub (replace with real USB + on-chip discovery)
# ──────────────────────────────────────────────────────────────────────────────

def enumerate_npus():
    """
    Replace this with your real NPU discovery:
    - scan USB NPUs
    - query on-chip NPU
    Return list of (device_id, backend, kind).
    """
    devices = []
    # Example:
    # devices.append(("usb_npu_0", usb_backend_handle_0, "USB"))
    # devices.append(("onchip_npu_0", onchip_backend_handle, "ONCHIP"))
    return devices

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo wiring with dummy organs; swap with your real ones.
    real_gpu_impl = DummyGPUImpl()
    real_spine_impl = DummyTelemetrySpineImpl()
    real_ui_impl = DummyUIAutomationImpl()

    gpu = GPUOrgan(real_gpu_impl)
    telemetry = TelemetrySpineAdapter(real_spine_impl)
    ui_auto = UIAutomationOrgan(real_ui_impl)

    multi_npu = MultiNPUManager(enumerate_npus)
    replica_swarm = ReplicaNPUSwarm(multi_npu)

    cockpit_state = CockpitState()
    heatmap = HeatmapBuffer(max_frames=300)

    app = AlienCockpit(gpu, telemetry, replica_swarm, cockpit_state, ui_auto, heatmap)
    app.mainloop()

