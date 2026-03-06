import math
import random
import time
import sys
import platform
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import tkinter as tk
from tkinter import ttk

# =============================================================
# ReplicaNPU: Ultra-style organ (plasticity, integrity, auto-tune)
# =============================================================

class ReplicaNPU:
    """
    Ultra-style software-simulated Neural Processing Unit (NPU)
    - MAC / matmul / activations
    - Predictive heads (short/medium/long)
    - Plasticity + integrity
    - Auto-tuning knobs (appetite, thresholds, horizon, dampening, cache, threads)
    - Auto-calibration (baselines, stance thresholds, reinforcement weights, confidence scaling)
    - Meta-state hooks via external brain
    """

    SERIAL_VERSION = 1

    def __init__(
        self,
        name="npu",
        cores=8,
        frequency_ghz=1.2,
        memory_size=32,
        plasticity_decay=0.0005,
        integrity_threshold=0.4,
        confidence_horizon=32,
    ):
        self.name = name
        self.cores = cores
        self.frequency_ghz = frequency_ghz

        # Hardware metrics
        self.cycles = 0
        self.energy = 0.0

        # Temporal memory
        self.memory = deque(maxlen=memory_size)

        # Plasticity
        self.plasticity = 1.0
        self.plasticity_decay = plasticity_decay

        # Integrity
        self.integrity_threshold = integrity_threshold
        self.model_integrity = 1.0
        self.frozen = False
        self.integrity_history = deque(maxlen=128)

        # Predictive heads
        self.heads = {}
        self.confidence_horizon = confidence_horizon

        # Symbolic modulation
        self.symbolic_bias = {}

        # Auto-tuning knobs
        self.appetite = 0.5          # how aggressive to chase performance
        self.thresholds = 0.5        # risk threshold
        self.horizon = 0.5           # short vs long horizon bias
        self.dampening = 0.5         # how much to smooth changes
        self.cache_behavior = 0.5    # caching aggressiveness
        self.thread_expansion = 0.5  # how many threads/cores to open up

        # Auto-calibration state
        self.baseline_perf = 0.5
        self.stance_thresholds = {
            "Conservative": 0.3,
            "Balanced": 0.6,
            "Beast": 0.85,
        }
        self.reinforcement_weights = {
            "short": 1.0,
            "medium": 1.0,
            "long": 1.0,
        }
        self.confidence_scale = 1.0

        # Scheduler
        self.instruction_queue = deque()

        # Last auto-calibration timestamp
        self.last_calibration_ts = time.time()

    # =========================================================
    # Scheduler
    # =========================================================
    def schedule(self, fn, *args, priority=10):
        self.instruction_queue.append((priority, fn, args))

    def tick(self, budget=64, last_step_success=True):
        if self.instruction_queue:
            self.instruction_queue = deque(
                sorted(self.instruction_queue, key=lambda x: x[0])
            )

        executed = 0
        while self.instruction_queue and executed < budget:
            priority, fn, args = self.instruction_queue.popleft()
            try:
                fn(*args)
            except Exception:
                self.cycles += 1
                self.energy += 0.0001
            executed += 1

        # Outcome-aware plasticity decay
        if last_step_success:
            self.plasticity = max(0.1, self.plasticity - self.plasticity_decay * 0.5)
        else:
            self.plasticity = max(0.05, self.plasticity - self.plasticity_decay * 1.5)

    # =========================================================
    # Low-level ops
    # =========================================================
    def mac(self, a, b):
        self.cycles += 1
        self.energy += 0.001
        v = a * b
        if math.isnan(v) or math.isinf(v):
            v = 0.0
        return v

    def vector_mac(self, v1, v2):
        if len(v1) != len(v2):
            raise ValueError(f"vector_mac mismatch: {len(v1)} vs {len(v2)}")
        chunk = max(1, math.ceil(len(v1) / self.cores))
        acc = 0.0
        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                partial += self.mac(v1[j], v2[j])
            acc += partial
        return acc

    def matmul(self, A, B, budget=None):
        if not A or not B:
            return []
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])
        if cols_A != rows_B:
            raise ValueError(f"matmul mismatch: {cols_A} vs {rows_B}")
        result = [[0.0] * cols_B for _ in range(rows_A)]
        ops = 0
        for i in range(rows_A):
            for j in range(cols_B):
                col = [B[k][j] for k in range(rows_B)]
                result[i][j] = self.vector_mac(A[i], col)
                ops += 1
                if budget is not None and ops >= budget:
                    return result
        return result

    # =========================================================
    # Activations
    # =========================================================
    def relu(self, x):
        self.cycles += 1
        return max(0.0, x)

    def sigmoid(self, x):
        self.cycles += 2
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def activate(self, tensor, mode="relu"):
        if not tensor:
            return tensor
        rows = len(tensor)
        cols = len(tensor[0])
        for i in range(rows):
            for j in range(cols):
                if mode == "relu":
                    tensor[i][j] = self.relu(tensor[i][j])
                else:
                    tensor[i][j] = self.sigmoid(tensor[i][j])
        return tensor

    # =========================================================
    # Predictive heads
    # =========================================================
    def add_head(self, name, input_dim, lr=0.01, risk=1.0, organ=None):
        self.heads[name] = {
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "organ": organ,
            "history": deque(maxlen=self.confidence_horizon),
        }

    def _symbolic_modulation(self, name):
        return self.symbolic_bias.get(name, 0.0)

    def _predict_head_raw(self, head, x, name, record=True):
        if len(x) != len(head["w"]):
            raise ValueError(
                f"Head '{name}' input dim mismatch: {len(x)} vs {len(head['w'])}"
            )
        y = 0.0
        for i in range(len(x)):
            y += self.mac(x[i], head["w"][i])
        y += head["b"]
        y += self._symbolic_modulation(name)
        if record:
            head["history"].append(y)
            self.memory.append(y)
        return y

    def predict(self, x):
        preds = {}
        for name, head in self.heads.items():
            preds[name] = self._predict_head_raw(head, x, name, record=True)
        return preds

    def learn(self, x, targets):
        if self.frozen:
            return {}
        errors = {}
        for name, target in targets.items():
            if name not in self.heads:
                continue
            head = self.heads[name]
            pred = self._predict_head_raw(head, x, name, record=False)
            error = target - pred
            weighted_error = (
                error
                * head["risk"]
                * self.plasticity
                * self.model_integrity
                * self.reinforcement_weights.get(name, 1.0)
            )
            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted_error * x[i]
                self.cycles += 1
            head["b"] += head["lr"] * weighted_error
            self.energy += 0.005
            errors[name] = error
            final_pred = self._predict_head_raw(head, x, name, record=True)
            _ = final_pred
        return errors

    # =========================================================
    # Confidence & Integrity
    # =========================================================
    def confidence(self, name):
        if name not in self.heads:
            return 0.0
        h = self.heads[name]["history"]
        if len(h) < 2:
            return 0.5 * self.confidence_scale
        mean = sum(h) / len(h)
        var = sum((v - mean) ** 2 for v in h) / len(h)
        norm_var = min(1.0, var / (1.0 + abs(mean)))
        base_conf = max(0.0, min(1.0, 1.0 - norm_var))
        return max(0.0, min(1.0, base_conf * self.confidence_scale))

    def check_integrity(self, external_integrity=1.0, allow_recovery=True):
        self.model_integrity = max(0.0, min(1.0, external_integrity))
        self.integrity_history.append(self.model_integrity)
        self.frozen = self.model_integrity < self.integrity_threshold
        if allow_recovery and not self.frozen and self.model_integrity > self.integrity_threshold:
            self.micro_recovery(rate=0.005)

    # =========================================================
    # Plasticity Recovery
    # =========================================================
    def micro_recovery(self, rate=0.01):
        self.plasticity = min(1.0, self.plasticity + rate)

    # =========================================================
    # Symbolic Interface
    # =========================================================
    def set_symbolic_bias(self, name, value):
        self.symbolic_bias[name] = value

    # =========================================================
    # Auto-tuning & Auto-calibration
    # =========================================================
    def auto_tune(self, long_term_success):
        """
        Adjust appetite, thresholds, horizon, dampening, cache, thread_expansion
        based on long-term success (0..1).
        """
        alpha = 0.1
        self.appetite = (1 - alpha) * self.appetite + alpha * long_term_success
        self.thresholds = (1 - alpha) * self.thresholds + alpha * (1 - long_term_success)
        self.horizon = (1 - alpha) * self.horizon + alpha * long_term_success
        self.dampening = (1 - alpha) * self.dampening + alpha * (0.5 + 0.5 * (1 - long_term_success))
        self.cache_behavior = (1 - alpha) * self.cache_behavior + alpha * long_term_success
        self.thread_expansion = (1 - alpha) * self.thread_expansion + alpha * long_term_success

    def auto_calibrate(self, perf_metric):
        """
        Recompute baselines, stance thresholds, reinforcement weights, confidence scaling.
        """
        self.baseline_perf = 0.9 * self.baseline_perf + 0.1 * perf_metric
        # Simple stance threshold drift
        self.stance_thresholds["Conservative"] = max(0.1, self.baseline_perf * 0.5)
        self.stance_thresholds["Balanced"] = min(0.9, self.baseline_perf * 0.9)
        self.stance_thresholds["Beast"] = min(0.99, self.baseline_perf * 1.1)

        # Reinforcement weights: reward heads that correlate with perf
        self.reinforcement_weights["short"] = 0.8 + 0.4 * perf_metric
        self.reinforcement_weights["medium"] = 1.0
        self.reinforcement_weights["long"] = 1.2 - 0.4 * perf_metric

        # Confidence scaling
        self.confidence_scale = 0.8 + 0.4 * perf_metric

    def maybe_daily_calibration(self, perf_metric):
        now = time.time()
        if now - self.last_calibration_ts > 24 * 3600:
            self.auto_calibrate(perf_metric)
            self.last_calibration_ts = now

    # =========================================================
    # Stats
    # =========================================================
    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9) if self.frequency_ghz > 0 else 0.0
        return {
            "name": self.name,
            "cores": self.cores,
            "cycles": self.cycles,
            "estimated_time_sec": time_sec,
            "energy_units": round(self.energy, 6),
            "plasticity": round(self.plasticity, 3),
            "integrity": round(self.model_integrity, 3),
            "frozen": self.frozen,
            "confidence": {
                k: round(self.confidence(k), 3) for k in self.heads
            },
            "appetite": round(self.appetite, 3),
            "thresholds": round(self.thresholds, 3),
            "horizon": round(self.horizon, 3),
            "dampening": round(self.dampening, 3),
            "cache_behavior": round(self.cache_behavior, 3),
            "thread_expansion": round(self.thread_expansion, 3),
        }


# =============================================================
# Dummy GPU Organ
# =============================================================

class DummyGPUOrgan:
    """
    Dummy GPU backend:
    - Simulates utilization, frame time, VRAM pressure, temperature
    - No real GPU calls
    """

    def __init__(self, name="gpu0"):
        self.name = name
        self.utilization = 0.3
        self.vram_usage = 0.4
        self.temperature = 0.5
        self.frame_time_ms = 16.0

    def update(self):
        # Simple random walk with bounds
        def step(v, scale=0.05):
            v += random.uniform(-scale, scale)
            return max(0.0, min(1.0, v))

        self.utilization = step(self.utilization, 0.07)
        self.vram_usage = step(self.vram_usage, 0.05)
        self.temperature = step(self.temperature, 0.03)
        # Frame time inversely related to utilization (just for flavor)
        self.frame_time_ms = 10 + (1.0 - self.utilization) * 20

    def snapshot(self):
        return {
            "utilization": self.utilization,
            "vram_usage": self.vram_usage,
            "temperature": self.temperature,
            "frame_time_ms": self.frame_time_ms,
        }


# =============================================================
# NPU Node + Swarm
# =============================================================

class NPUNode:
    def __init__(self, name):
        self.name = name
        self.npu = ReplicaNPU(name=name, cores=8, frequency_ghz=1.5)
        # Multi-horizon heads
        self.npu.add_head("short", 3, lr=0.05, risk=1.5, organ="short_horizon")
        self.npu.add_head("medium", 3, lr=0.03, risk=1.0, organ="mid_horizon")
        self.npu.add_head("long", 3, lr=0.02, risk=0.7, organ="long_horizon")
        self.last_preds = {"short": 0.5, "medium": 0.5, "long": 0.5}

    def step(self, x, targets, integrity=1.0):
        preds = self.npu.predict(x)
        errs = self.npu.learn(x, targets)
        self.npu.micro_recovery()
        self.npu.check_integrity(external_integrity=integrity)
        self.npu.tick(budget=32, last_step_success=True)
        self.last_preds = preds
        return preds, errs

    def stats(self):
        s = self.npu.stats()
        s["node"] = self.name
        return s


class NPUSwarm:
    def __init__(self, count=2):
        self.nodes = [NPUNode(f"npu_{i}") for i in range(count)]

    def step_all(self, x, targets, integrity=1.0):
        all_preds = []
        all_errs = []
        for node in self.nodes:
            preds, errs = node.step(x, targets, integrity=integrity)
            all_preds.append(preds)
            all_errs.append(errs)
        return all_preds, all_errs

    def aggregate(self):
        # Simple average across nodes
        agg = {"short": 0.0, "medium": 0.0, "long": 0.0}
        if not self.nodes:
            return agg
        for node in self.nodes:
            for k, v in node.last_preds.items():
                agg[k] += v
        for k in agg:
            agg[k] /= len(self.nodes)
        return agg

    def stats(self):
        return [n.stats() for n in self.nodes]


# =============================================================
# Brain + Prediction Bus
# =============================================================

class PredictionBus:
    def __init__(self):
        self.current_risk = 0.5
        self.last_perf = 0.5


class Brain:
    """
    Auto-tuning brain:
    - Reads GPU + NPU swarm
    - Maintains meta-state + stance
    - Computes baseline + best-guess
    - Builds reasoning heatmap
    """

    META_STATES = ["Hyper-Flow", "Sentinel", "Recovery-Flow", "Deep-Dream"]
    STANCES = ["Conservative", "Balanced", "Beast"]

    def __init__(self, swarm: NPUSwarm, gpu: DummyGPUOrgan):
        self.swarm = swarm
        self.gpu = gpu
        self.meta_state = "Sentinel"
        self.stance = "Balanced"
        self.last_predictions = {
            "short": 0.5,
            "medium": 0.5,
            "long": 0.5,
            "baseline": 0.5,
            "best_guess": 0.5,
            "meta_conf": 0.5,
        }
        self.last_reasoning = []
        self.last_heatmap = {}
        self.model_integrity = 1.0

    def _compute_baseline(self, gpu_snap):
        # Baseline from GPU frame time + utilization
        util = gpu_snap["utilization"]
        ft = gpu_snap["frame_time_ms"]
        # Normalize frame time to 0..1 (rough)
        ft_norm = max(0.0, min(1.0, (40 - ft) / 30.0))
        baseline = 0.5 * util + 0.5 * ft_norm
        return max(0.0, min(1.0, baseline))

    def _compute_best_guess(self, agg_preds, baseline):
        # Weighted blend of short/medium/long + baseline
        w_short = 0.35
        w_med = 0.35
        w_long = 0.2
        w_base = 0.1
        best = (
            w_short * agg_preds["short"]
            + w_med * agg_preds["medium"]
            + w_long * agg_preds["long"]
            + w_base * baseline
        )
        return max(0.0, min(1.0, best))

    def _compute_meta_conf(self, swarm_stats):
        # Average confidence across nodes and heads
        confs = []
        for s in swarm_stats:
            for v in s["confidence"].values():
                confs.append(v)
        if not confs:
            return 0.5
        return max(0.0, min(1.0, sum(confs) / len(confs)))

    def _update_meta_state(self, best_guess, turbulence):
        # Simple meta-state transitions with inertia
        current = self.meta_state
        if current == "Hyper-Flow":
            if turbulence > 0.7:
                self.meta_state = "Sentinel"
        elif current == "Sentinel":
            if best_guess > 0.7 and turbulence < 0.5:
                self.meta_state = "Hyper-Flow"
            elif best_guess < 0.3:
                self.meta_state = "Recovery-Flow"
        elif current == "Recovery-Flow":
            if turbulence < 0.4:
                self.meta_state = "Deep-Dream"
        elif current == "Deep-Dream":
            if turbulence > 0.6:
                self.meta_state = "Sentinel"

    def _update_stance(self, best_guess, thresholds):
        if best_guess < thresholds["Conservative"]:
            self.stance = "Conservative"
        elif best_guess < thresholds["Balanced"]:
            self.stance = "Balanced"
        else:
            self.stance = "Beast"

    def update(self, prediction_bus: PredictionBus):
        gpu_snap = self.gpu.snapshot()
        agg_preds = self.swarm.aggregate()
        baseline = self._compute_baseline(gpu_snap)
        best_guess = self._compute_best_guess(agg_preds, baseline)
        swarm_stats = self.swarm.stats()
        meta_conf = self._compute_meta_conf(swarm_stats)

        # Turbulence: variance across nodes' best horizon
        node_best = []
        for s in swarm_stats:
            c = s["confidence"]
            if c:
                best_h = max(c, key=c.get)
                node_best.append(c[best_h])
        if node_best:
            m = sum(node_best) / len(node_best)
            var = sum((v - m) ** 2 for v in node_best) / len(node_best)
            turbulence = max(0.0, min(1.0, var))
        else:
            turbulence = 0.5

        # Meta-state + stance
        npu0 = self.swarm.nodes[0].npu
        self._update_meta_state(best_guess, turbulence)
        self._update_stance(best_guess, npu0.stance_thresholds)

        # Risk: inverse of best_guess * meta_conf
        risk = max(0.0, min(1.0, 1.0 - best_guess * meta_conf))

        # Auto-tune NPUs based on performance
        perf_metric = best_guess * meta_conf
        for node in self.swarm.nodes:
            node.npu.auto_tune(perf_metric)
            node.npu.maybe_daily_calibration(perf_metric)

        # Model integrity: simple function of perf + turbulence
        self.model_integrity = max(0.0, min(1.0, perf_metric * (1.0 - 0.5 * turbulence)))

        # Reasoning + heatmap
        self.last_predictions = {
            "short": agg_preds["short"],
            "medium": agg_preds["medium"],
            "long": agg_preds["long"],
            "baseline": baseline,
            "best_guess": best_guess,
            "meta_conf": meta_conf,
        }

        self.last_reasoning = [
            f"GPU util={gpu_snap['utilization']:.2f}, frame={gpu_snap['frame_time_ms']:.1f}ms",
            f"Swarm short/med/long={agg_preds['short']:.2f}/{agg_preds['medium']:.2f}/{agg_preds['long']:.2f}",
            f"Baseline={baseline:.2f}, Best-Guess={best_guess:.2f}, Meta-Conf={meta_conf:.2f}",
            f"Turbulence={turbulence:.2f}, Meta-State={self.meta_state}, Stance={self.stance}",
            f"Perf={perf_metric:.2f}, Risk={risk:.2f}, Integrity={self.model_integrity:.2f}",
        ]

        self.last_heatmap = {
            "gpu_util": gpu_snap["utilization"],
            "gpu_vram": gpu_snap["vram_usage"],
            "gpu_temp": gpu_snap["temperature"],
            "short": agg_preds["short"],
            "medium": agg_preds["medium"],
            "long": agg_preds["long"],
            "baseline": baseline,
            "best_guess": best_guess,
            "meta_conf": meta_conf,
            "turbulence": turbulence,
            "risk": risk,
            "best_guess_contributors": {
                "short": 0.35,
                "medium": 0.35,
                "long": 0.20,
                "baseline": 0.10,
                "weights": {
                    "short": 0.35,
                    "medium": 0.35,
                    "long": 0.20,
                    "baseline": 0.10,
                },
            },
        }

        prediction_bus.current_risk = risk
        prediction_bus.last_perf = perf_metric


# =============================================================
# Discovery: Device descriptor
# =============================================================

@dataclass
class DiscoveredDevice:
    bus: str              # "PCIe" or "USB"
    vendor_id: str        # hex string, e.g. "10DE"
    device_id: str        # hex string
    description: str      # human-readable
    path: str             # OS-specific path / identifier
    is_gpu: bool = False
    is_npu: bool = False
    backend_hint: Optional[str] = None  # e.g. "nvidia", "intel_npu", "coral", "habana"


# =============================================================
# Discovery: Vendor maps
# =============================================================

GPU_VENDORS = {
    "10DE": "nvidia",
    "1002": "amd",
    "1022": "amd",
    "8086": "intel",
}

NPU_USB_HINTS = [
    "coral", "edge tpu", "habana", "npu", "neural", "accelerator",
]

NPU_PCI_HINTS = [
    "habana", "gaudi", "npu", "neural", "accelerator",
]


# =============================================================
# Discovery: OS helpers
# =============================================================

def _run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode(errors="ignore")
    except Exception:
        return ""


# -------------------------
# Linux: lspci / lsusb
# -------------------------

def _probe_pci_linux() -> List[DiscoveredDevice]:
    out = _run_cmd(["lspci", "-nn"])
    devices = []
    for line in out.splitlines():
        if not line.strip():
            continue
        desc = line.strip()
        if "[" in desc and "]" in desc:
            bracket = desc.split("[")[-1].split("]")[0]
            if ":" in bracket:
                ven, dev = bracket.split(":", 1)
                ven = ven.upper()
                dev = dev.upper()
            else:
                ven, dev = "0000", "0000"
        else:
            ven, dev = "0000", "0000"

        vendor_name = GPU_VENDORS.get(ven)
        is_gpu = "vga" in desc.lower() or "3d controller" in desc.lower()
        is_npu = any(h in desc.lower() for h in NPU_PCI_HINTS)
        backend_hint = vendor_name if is_gpu else None
        if is_npu and backend_hint is None:
            backend_hint = "generic_npu"

        devices.append(
            DiscoveredDevice(
                bus="PCIe",
                vendor_id=ven,
                device_id=dev,
                description=desc,
                path=desc.split()[0],
                is_gpu=is_gpu,
                is_npu=is_npu,
                backend_hint=backend_hint,
            )
        )
    return devices


def _probe_usb_linux() -> List[DiscoveredDevice]:
    out = _run_cmd(["lsusb"])
    devices = []
    for line in out.splitlines():
        if "ID " not in line:
            continue
        parts = line.split("ID ", 1)[-1].split(None, 1)
        if not parts:
            continue
        ids = parts[0]
        desc = parts[1] if len(parts) > 1 else ""
        if ":" in ids:
            ven, dev = ids.split(":", 1)
            ven = ven.upper()
            dev = dev.upper()
        else:
            ven, dev = "0000", "0000"

        is_npu = any(h in desc.lower() for h in NPU_USB_HINTS)
        backend_hint = None
        if is_npu:
            if "coral" in desc.lower() or "edge tpu" in desc.lower():
                backend_hint = "coral_edgetpu"
            else:
                backend_hint = "generic_usb_npu"

        devices.append(
            DiscoveredDevice(
                bus="USB",
                vendor_id=ven,
                device_id=dev,
                description=desc.strip(),
                path=line.strip(),
                is_gpu=False,
                is_npu=is_npu,
                backend_hint=backend_hint,
            )
        )
    return devices


# -------------------------
# Windows: wmic / powershell
# -------------------------

def _probe_pci_windows() -> List[DiscoveredDevice]:
    out = _run_cmd(["wmic", "path", "Win32_VideoController", "get", "Name,PNPDeviceID", "/format:list"])
    devices = []
    current = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            if current:
                name = current.get("Name", "Unknown GPU")
                pnp = current.get("PNPDeviceID", "")
                ven = "0000"
                dev = "0000"
                if "VEN_" in pnp and "DEV_" in pnp:
                    try:
                        ven = pnp.split("VEN_")[1][:4].upper()
                        dev = pnp.split("DEV_")[1][:4].upper()
                    except Exception:
                        pass
                vendor_name = GPU_VENDORS.get(ven)
                devices.append(
                    DiscoveredDevice(
                        bus="PCIe",
                        vendor_id=ven,
                        device_id=dev,
                        description=name,
                        path=pnp,
                        is_gpu=True,
                        is_npu=False,
                        backend_hint=vendor_name or "generic_gpu",
                    )
                )
                current = {}
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            current[k] = v
    return devices


def _probe_usb_windows() -> List[DiscoveredDevice]:
    cmd = [
        "powershell",
        "-Command",
        "Get-PnpDevice -PresentOnly | Select-Object -Property InstanceId, FriendlyName",
    ]
    out = _run_cmd(cmd)
    devices = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "InstanceId" in line or "FriendlyName" in line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 1:
            continue
        instance = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        ven = "0000"
        dev = "0000"
        if "VEN_" in instance and "DEV_" in instance:
            try:
                ven = instance.split("VEN_")[1][:4].upper()
                dev = instance.split("DEV_")[1][:4].upper()
            except Exception:
                pass
        is_npu = any(h in name.lower() for h in NPU_USB_HINTS)
        backend_hint = "generic_usb_npu" if is_npu else None
        devices.append(
            DiscoveredDevice(
                bus="USB",
                vendor_id=ven,
                device_id=dev,
                description=name,
                path=instance,
                is_gpu=False,
                is_npu=is_npu,
                backend_hint=backend_hint,
            )
        )
    return devices


# -------------------------
# macOS: system_profiler
# -------------------------

def _probe_pci_macos() -> List[DiscoveredDevice]:
    out = _run_cmd(["system_profiler", "SPDisplaysDataType"])
    devices = []
    current_name = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Chipset Model:"):
            current_name = s.split(":", 1)[1].strip()
        if s.startswith("Vendor:") and current_name:
            desc = current_name
            devices.append(
                DiscoveredDevice(
                    bus="PCIe",
                    vendor_id="0000",
                    device_id="0000",
                    description=desc,
                    path=desc,
                    is_gpu=True,
                    is_npu=False,
                    backend_hint="generic_gpu",
                )
            )
            current_name = None
    return devices


def _probe_usb_macos() -> List[DiscoveredDevice]:
    out = _run_cmd(["system_profiler", "SPUSBDataType"])
    devices = []
    for line in out.splitlines():
        s = line.strip()
        if not s or ":" not in s:
            continue
        desc = s.rstrip(":")
        is_npu = any(h in desc.lower() for h in NPU_USB_HINTS)
        backend_hint = "generic_usb_npu" if is_npu else None
        devices.append(
            DiscoveredDevice(
                bus="USB",
                vendor_id="0000",
                device_id="0000",
                description=desc,
                path=desc,
                is_gpu=False,
                is_npu=is_npu,
                backend_hint=backend_hint,
            )
        )
    return devices


# =============================================================
# Discovery: Public API
# =============================================================

def discover_devices() -> List[DiscoveredDevice]:
    system = platform.system().lower()
    devices: List[DiscoveredDevice] = []
    if system == "linux":
        devices.extend(_probe_pci_linux())
        devices.extend(_probe_usb_linux())
    elif system == "windows":
        devices.extend(_probe_pci_windows())
        devices.extend(_probe_usb_windows())
    elif system == "darwin":
        devices.extend(_probe_pci_macos())
        devices.extend(_probe_usb_macos())
    else:
        pass
    return devices


def pick_gpu_device(devices: List[DiscoveredDevice]) -> Optional[DiscoveredDevice]:
    gpus = [d for d in devices if d.is_gpu]
    if not gpus:
        return None
    return gpus[0]


def pick_npu_devices(devices: List[DiscoveredDevice]) -> List[DiscoveredDevice]:
    return [d for d in devices if d.is_npu]


# =============================================================
# Integration: GPU + NPU backend factories
# =============================================================

def create_gpu_backend_from_discovery() -> DummyGPUOrgan:
    devices = discover_devices()
    gpu_dev = pick_gpu_device(devices)
    if gpu_dev:
        print(f"[discovery] GPU: {gpu_dev.description} (ven={gpu_dev.vendor_id}, dev={gpu_dev.device_id}) "
              f"backend={gpu_dev.backend_hint}")
        return DummyGPUOrgan(name=gpu_dev.description)
    else:
        print("[discovery] No GPU found, using DummyGPUOrgan")
        return DummyGPUOrgan(name="dummy_gpu")


def create_npu_swarm_from_discovery(default_count: int = 2) -> NPUSwarm:
    devices = discover_devices()
    npu_devs = pick_npu_devices(devices)
    if not npu_devs:
        print(f"[discovery] No NPU devices found, using simulated swarm x{default_count}")
        return NPUSwarm(count=default_count)

    print(f"[discovery] Found {len(npu_devs)} NPU-like devices:")
    swarm = NPUSwarm(count=0)
    swarm.nodes = []
    for idx, dev in enumerate(npu_devs):
        print(f"  - {dev.bus} {dev.description} (ven={dev.vendor_id}, dev={dev.device_id}) backend={dev.backend_hint}")
        node = NPUNode(name=f"{dev.backend_hint or 'npu'}_{idx}")
        swarm.nodes.append(node)
    return swarm


# =============================================================
# AlienCockpit GUI (dark neon)
# =============================================================

class AlienCockpit:
    def __init__(self, root):
        self.root = root
        self.root.title("Alien Cockpit - ReplicaNPU Swarm")
        self.root.configure(bg="#050510")

        # Organs (discovery-backed)
        self.gpu = create_gpu_backend_from_discovery()
        self.swarm = create_npu_swarm_from_discovery(default_count=2)
        self.brain = Brain(self.swarm, self.gpu)
        self.prediction_bus = PredictionBus()

        # Layout
        self._build_ui()

        # Synthetic input stream
        self._stream_idx = 0
        self._stream = [
            [0.2, 0.5, 0.1],
            [0.3, 0.4, 0.2],
            [0.4, 0.6, 0.3],
            [0.5, 0.7, 0.4],
        ]

        self._start_update_loop()

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Alien.TLabel", background="#050510", foreground="#66ffcc")
        style.configure("AlienTitle.TLabel", background="#050510", foreground="#ff00ff", font=("Consolas", 14, "bold"))

        top_frame = ttk.Frame(self.root, style="Alien.TFrame")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self.lbl_title = ttk.Label(top_frame, text="Alien Cockpit - ReplicaNPU Swarm", style="AlienTitle.TLabel")
        self.lbl_title.pack(side=tk.LEFT, padx=4)

        self.lbl_meta_state = ttk.Label(top_frame, text="Meta-State: ?", style="Alien.TLabel")
        self.lbl_meta_state.pack(side=tk.LEFT, padx=10)

        self.lbl_stance = ttk.Label(top_frame, text="Stance: ?", style="Alien.TLabel")
        self.lbl_stance.pack(side=tk.LEFT, padx=10)

        self.lbl_meta_conf = ttk.Label(top_frame, text="Meta-Confidence: ?", style="Alien.TLabel")
        self.lbl_meta_conf.pack(side=tk.LEFT, padx=10)

        self.lbl_model_integrity = ttk.Label(top_frame, text="Model Integrity: ?", style="Alien.TLabel")
        self.lbl_model_integrity.pack(side=tk.LEFT, padx=10)

        self.lbl_current_risk = ttk.Label(top_frame, text="Current Risk: ?", style="Alien.TLabel")
        self.lbl_current_risk.pack(side=tk.LEFT, padx=10)

        self.lbl_cache = ttk.Label(top_frame, text="Transport Cache Hit Rate: 0.00", style="Alien.TLabel")
        self.lbl_cache.pack(side=tk.LEFT, padx=10)

        main_frame = ttk.Frame(self.root, style="Alien.TFrame")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        chart_frame = ttk.Frame(main_frame, style="Alien.TFrame")
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas_chart = tk.Canvas(chart_frame, width=500, height=260, bg="#050510", highlightthickness=0)
        self.canvas_chart.pack(fill=tk.BOTH, expand=True)

        reason_frame = ttk.Frame(main_frame, style="Alien.TFrame")
        reason_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.txt_reason = tk.Text(
            reason_frame,
            width=50,
            height=20,
            bg="#050510",
            fg="#66ffcc",
            insertbackground="#66ffcc",
            font=("Consolas", 9),
            borderwidth=0,
        )
        self.txt_reason.pack(fill=tk.BOTH, expand=True)

    # =========================================================
    # Update loop
    # =========================================================
    def _start_update_loop(self):
        self._tick()
        self.root.after(250, self._start_update_loop)

    def _tick(self):
        self.gpu.update()

        x = self._stream[self._stream_idx % len(self._stream)]
        self._stream_idx += 1
        targets = {
            "short": 0.6 + 0.1 * random.uniform(-1, 1),
            "medium": 0.55 + 0.1 * random.uniform(-1, 1),
            "long": 0.5 + 0.1 * random.uniform(-1, 1),
        }

        self.swarm.step_all(x, targets, integrity=self.brain.model_integrity)
        self.brain.update(self.prediction_bus)
        self._update_gui()

    def _update_gui(self):
        p = self.brain.last_predictions
        self.lbl_meta_state.config(text=f"Meta-State: {self.brain.meta_state}")
        self.lbl_stance.config(text=f"Stance: {self.brain.stance}")
        self.lbl_meta_conf.config(text=f"Meta-Confidence: {p['meta_conf']:.2f}")
        self.lbl_model_integrity.config(text=f"Model Integrity: {self.brain.model_integrity:.2f}")
        self.lbl_current_risk.config(text=f"Current Risk: {self.prediction_bus.current_risk:.2f}")
        self.lbl_cache.config(text=f"Transport Cache Hit Rate: {0.75 + 0.1 * random.uniform(-1,1):.2f}")

        self._draw_chart()
        self._update_reasoning()

    def _draw_chart(self):
        self.canvas_chart.delete("all")
        w = int(self.canvas_chart["width"])
        h = int(self.canvas_chart["height"])

        self.canvas_chart.create_rectangle(0, 0, w, h, fill="#050510", outline="")

        p = self.brain.last_predictions
        short = p["short"]
        med = p["medium"]
        long = p["long"]
        baseline = p["baseline"]
        best_guess = p["best_guess"]

        def y_from_val(v):
            return h - int(v * (h - 20)) - 10

        x_short = w * 0.2
        x_med = w * 0.5
        x_long = w * 0.8

        y_short = y_from_val(short)
        y_med = y_from_val(med)
        y_long = y_from_val(long)
        y_base = y_from_val(baseline)
        y_best = y_from_val(best_guess)

        self.canvas_chart.create_line(0, y_base, w, y_base, fill="#444444", dash=(3, 3))

        self.canvas_chart.create_line(x_short, y_short, x_med, y_med, fill="#00ccff", width=2)
        self.canvas_chart.create_line(x_med, y_med, x_long, y_long, fill="#00ccff", width=2)

        stance_color = {
            "Conservative": "#66ff66",
            "Balanced": "#ffff66",
            "Beast": "#ff6666",
        }.get(self.brain.stance, "#ffffff")
        self.canvas_chart.create_line(x_short, y_med, x_long, y_med, fill=stance_color, width=1)

        self.canvas_chart.create_line(0, y_best, w, y_best, fill="#ff00ff", width=2)

        for i in range(0, w, 40):
            self.canvas_chart.create_line(i, 0, i, h, fill="#111122")
        for j in range(0, h, 40):
            self.canvas_chart.create_line(0, j, w, j, fill="#111122")

        self.canvas_chart.create_text(
            5,
            5,
            anchor="nw",
            fill="#aaaaaa",
            text="Short/Med/Long (cyan), Baseline (gray), Best-Guess (magenta), Stance band (green/yellow/red)",
            font=("Consolas", 8),
        )

    def _update_reasoning(self):
        self.txt_reason.delete("1.0", tk.END)
        self.txt_reason.insert(tk.END, "Reasoning Tail:\n")
        for line in self.brain.last_reasoning:
            self.txt_reason.insert(tk.END, f"  - {line}\n")

        self.txt_reason.insert(tk.END, "\nReasoning Heatmap:\n")
        for k, v in self.brain.last_heatmap.items():
            if k == "best_guess_contributors":
                continue
            if isinstance(v, float):
                self.txt_reason.insert(tk.END, f"  {k}: {v:.3f}\n")
            else:
                self.txt_reason.insert(tk.END, f"  {k}: {v}\n")

        self.txt_reason.insert(tk.END, "\nBest-Guess Contributors:\n")
        contrib = self.brain.last_heatmap.get("best_guess_contributors", {})
        for k, v in contrib.items():
            if k == "weights":
                continue
            self.txt_reason.insert(tk.END, f"  {k}: {v:.3f}\n")

        weights = contrib.get("weights", {})
        if weights:
            self.txt_reason.insert(tk.END, "\nWeights:\n")
            for k, w in weights.items():
                self.txt_reason.insert(tk.END, f"  {k}: {w:.3f}\n")


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = AlienCockpit(root)
    root.mainloop()

