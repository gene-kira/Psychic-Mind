#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
network_organism_soc_hardened.py

Unified hardened network organism with:
- Cross-platform HAL (Windows/Linux/macOS)
- NIC health sensors (multi-target latency/jitter/loss, median-based)
- Throughput meters (bytes/sec per NIC)
- Threat heatmap + GPU/NPU anomaly bridge
- AI cortex (heuristic + optional PyTorch/CuPy)
- Swarm mesh (queen/worker) with HMAC auth
- NetworkBrain (global / in-out / priority split)
- Gaming QoS hooks (Windows-focused)
- Persistent learning (history across reboots)
- System load awareness (CPU, memory)
- Consciousness modes: NEUTRAL / GAMING / DEEP_WATER
- Auto-gaming detection + deep-water detection
- Water/data physics engine (pressure, turbulence, flow)
- Hardened scoring (outlier rejection, history bias)
- PySide6 Enterprise SOC cockpit with tabbed interface:
  - Live NIC
  - Physics
  - Learning
  - Swarm
  - Security/Settings
"""

import sys
import subprocess
import time
import threading
import socket
import json
import platform
import statistics
import random
import os
import hmac
import hashlib
from enum import Enum
from pathlib import Path

# =========================
#  AUTOLOADER
# =========================

REQUIRED_LIBS = ["psutil"]
def ensure_libs():
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except Exception:
            pass

ensure_libs()
import psutil  # type: ignore

# Optional imports
try:
    from PySide6 import QtWidgets, QtCore, QtGui
    HAVE_QT = True
except Exception:
    HAVE_QT = False

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

try:
    import cupy as cp
    HAVE_CUPY = True
except Exception:
    HAVE_CUPY = False


# =========================
#  CONFIG & PATHS
# =========================

SWARM_PORT = 49231
SWARM_BROADCAST_INTERVAL = 3.0
NODE_TTL = 15.0

DEFAULT_PING_TARGETS = ["8.8.8.8", "1.1.1.1"]
PROBE_WINDOW = 20
PROBE_INTERVAL = 1.0
THROUGHPUT_INTERVAL = 1.0

ROLE_QUEEN = "queen"
ROLE_WORKER = "worker"

STATE_PATH = Path.home() / "network_brain_state.json"
HISTORY_PATH = Path.home() / "network_brain_history.json"

# Swarm security
SWARM_SHARED_KEY = b"change_this_to_a_long_random_secret"
SWARM_MAX_SKEW = 30.0  # seconds

class RoutingMode(Enum):
    GLOBAL_BEST = "global_best"
    IN_OUT_SPLIT = "in_out_split"
    PRIORITY_SPLIT = "priority_split"

class ConsciousnessMode(Enum):
    NEUTRAL = "neutral"
    GAMING = "gaming"
    DEEP_WATER = "deep_water"  # bulk / long-flow


# =========================
#  HAL (CROSS-PLATFORM, WINDOWS-FOCUSED)
# =========================

class NetworkHAL:
    def __init__(self):
        self.os = platform.system()

    def set_default_route(self, nic):
        if not nic:
            return
        try:
            if self.os == "Windows":
                subprocess.run(
                    ["netsh", "interface", "ipv4", "set", "interface", nic, "metric=5"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif self.os == "Linux":
                subprocess.run(
                    ["ip", "route", "replace", "default", "dev", nic],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif self.os == "Darwin":
                subprocess.run(
                    ["route", "change", "default", nic],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    def set_interface_metric(self, nic, metric):
        if not nic or self.os != "Windows":
            return
        try:
            subprocess.run(
                ["netsh", "interface", "ipv4", "set", "interface", nic, f"metric={metric}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def apply_global_best(self, nic):
        self.set_default_route(nic)

    def apply_in_out_split(self, in_nic, out_nic):
        if self.os == "Windows":
            if in_nic:
                self.set_interface_metric(in_nic, 5)
            if out_nic and out_nic != in_nic:
                self.set_interface_metric(out_nic, 15)
        else:
            if in_nic:
                self.set_default_route(in_nic)

    def apply_priority_split(self, high_prio_nic, low_prio_nic):
        if self.os == "Windows":
            if high_prio_nic:
                self.set_interface_metric(high_prio_nic, 5)
            if low_prio_nic and low_prio_nic != high_prio_nic:
                self.set_interface_metric(low_prio_nic, 25)
        else:
            if high_prio_nic:
                self.set_default_route(high_prio_nic)

    def apply_gaming_qos(self, game_process_names, high_prio_nic):
        if not high_prio_nic:
            return

        if self.os == "Windows":
            self.set_interface_metric(high_prio_nic, 5)
            try:
                ps_script = r'''
$games = @({names})
foreach ($g in $games) {{
  New-NetQosPolicy -Name "Game_$g" -AppPathNameMatchCondition "*$g*" -DSCPAction 46 -PolicyStore ActiveStore -ErrorAction SilentlyContinue
}}
'''.format(
                    names=",".join([f'"{n}"' for n in game_process_names])
                )
                subprocess.run(
                    ["powershell", "-Command", ps_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        elif self.os == "Linux":
            # Hook: tc + iptables marks
            pass
        elif self.os == "Darwin":
            # Hook: pf.conf tagging + altq
            pass


# =========================
#  NIC SENSORS (HARDENED, MULTI-TARGET)
# =========================

class NICProbe:
    def __init__(self, name, addr, ping_targets=None):
        self.name = name
        self.addr = addr
        self.ping_targets = ping_targets or DEFAULT_PING_TARGETS
        self.lat_samples = []
        self.loss_samples = []
        self.jitter_samples = []
        self.lock = threading.Lock()

    def _ping_once(self, target):
        try:
            if platform.system() == "Windows":
                cmd = ["ping", "-n", "1", "-w", "500", target]
            else:
                cmd = ["ping", "-c", "1", "-W", "1", target]
            start = time.time()
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            end = time.time()
            if out.returncode != 0:
                return None
            return (end - start) * 1000.0
        except Exception:
            return None

    def run(self, stop_event: threading.Event):
        while not stop_event.is_set():
            samples = []
            for t in self.ping_targets:
                lat = self._ping_once(t)
                if lat is not None:
                    samples.append(lat)
            with self.lock:
                if not samples:
                    self.loss_samples.append(1)
                else:
                    median_lat = statistics.median(samples)
                    self.lat_samples.append(median_lat)
                    self.loss_samples.append(0)
                if len(self.lat_samples) > PROBE_WINDOW:
                    self.lat_samples.pop(0)
                if len(self.loss_samples) > PROBE_WINDOW:
                    self.loss_samples.pop(0)
                if len(self.lat_samples) > 2:
                    diffs = [
                        abs(self.lat_samples[i] - self.lat_samples[i - 1])
                        for i in range(1, len(self.lat_samples))
                    ]
                    self.jitter_samples = diffs[-10:]
            time.sleep(PROBE_INTERVAL + random.uniform(-0.2, 0.2))

    def base_score(self):
        with self.lock:
            if not self.lat_samples:
                return 9999.0
            avg_lat = statistics.mean(self.lat_samples)
            loss = sum(self.loss_samples) / len(self.loss_samples)
            jitter = statistics.mean(self.jitter_samples) if self.jitter_samples else 0.0
            return avg_lat + (loss * 500.0) + jitter


class NICSensorCortex:
    def __init__(self):
        self.nics = self._discover_nics()
        self.probes = {}
        self.threads = []
        self.stop_event = threading.Event()
        self._start_probes()

    def _discover_nics(self):
        nics = {}
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family == psutil.AF_LINK:
                    continue
                if any(tag in name.lower() for tag in ["eth", "en", "ethernet", "wi-fi", "wifi", "wlan"]):
                    nics[name] = a.address
        return nics

    def _start_probes(self):
        for name, addr in self.nics.items():
            probe = NICProbe(name, addr)
            self.probes[name] = probe
            t = threading.Thread(target=probe.run, args=(self.stop_event,), daemon=True)
            t.start()
            self.threads.append(t)

    def get_scores(self):
        scores = {}
        for name, probe in self.probes.items():
            scores[name] = probe.base_score()
        return scores

    def stop(self):
        self.stop_event.set()


# =========================
#  THROUGHPUT METER
# =========================

class DataFlowMeter:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_counters = psutil.net_io_counters(pernic=True)
        self.throughput = {}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            time.sleep(THROUGHPUT_INTERVAL)
            now = psutil.net_io_counters(pernic=True)
            with self.lock:
                for nic, counters in now.items():
                    if nic not in self.last_counters:
                        continue
                    prev = self.last_counters[nic]
                    dt = THROUGHPUT_INTERVAL
                    tx_bytes = counters.bytes_sent - prev.bytes_sent
                    rx_bytes = counters.bytes_recv - prev.bytes_recv
                    tx_bps = tx_bytes * 8 / dt
                    rx_bps = rx_bytes * 8 / dt
                    self.throughput[nic] = {
                        "tx_bps": max(tx_bps, 0),
                        "rx_bps": max(rx_bps, 0),
                    }
                self.last_counters = now

    def get_throughput(self):
        with self.lock:
            return dict(self.throughput)

    def stop(self):
        self.stop_event.set()


# =========================
#  THREAT HEATMAP + GPU/NPU BRIDGE
# =========================

class ThreatHeatmap:
    def __init__(self):
        self.nic_threat = {}
        self.lock = threading.Lock()

    def update_threat(self, nic_name: str, score: float):
        with self.lock:
            self.nic_threat[nic_name] = score

    def get_threat(self, nic_name: str) -> float:
        with self.lock:
            return self.nic_threat.get(nic_name, 0.0)

    def apply(self, nic_scores: dict):
        out = {}
        for nic, base in nic_scores.items():
            threat = self.get_threat(nic)
            out[nic] = base + threat * 1000.0
        return out


class GPUAnomalyBridge:
    def __init__(self, heatmap: ThreatHeatmap):
        self.heatmap = heatmap

    def push_scores(self, nic_scores: dict):
        for nic, score in nic_scores.items():
            self.heatmap.update_threat(nic, float(score))


# =========================
#  AI CORTEX (PLUGGABLE, HARDENED)
# =========================

class AICortex:
    def __init__(self):
        self.backend = "heuristic"
        if HAVE_TORCH:
            self.backend = "torch"
        elif HAVE_CUPY:
            self.backend = "cupy"

    def score(self, nic_features: dict):
        if self.backend == "torch":
            raw = self._score_torch(nic_features)
        elif self.backend == "cupy":
            raw = self._score_cupy(nic_features)
        else:
            raw = self._score_heuristic(nic_features)
        return self._harden_scores(raw)

    def _score_heuristic(self, nic_features: dict):
        out = {}
        for nic, feats in nic_features.items():
            out[nic] = sum(feats)
        return out

    def _score_torch(self, nic_features: dict):
        out = {}
        for nic, feats in nic_features.items():
            x = torch.tensor(feats, dtype=torch.float32)
            out[nic] = float(x.mean().item())
        return out

    def _score_cupy(self, nic_features: dict):
        out = {}
        for nic, feats in nic_features.items():
            x = cp.asarray(feats, dtype=cp.float32)
            out[nic] = float(cp.mean(x).get())
        return out

    def _harden_scores(self, scores: dict):
        if not scores:
            return scores
        vals = list(scores.values())
        median_val = statistics.median(vals)
        mad = statistics.median([abs(v - median_val) for v in vals]) or 1.0
        out = {}
        for nic, v in scores.items():
            z = abs(v - median_val) / mad
            if z > 6.0:
                out[nic] = median_val
            else:
                out[nic] = v
        return out


# =========================
#  SWARM MESH (HMAC AUTH)
# =========================

def swarm_sign(payload: dict) -> dict:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    ts = time.time()
    msg = body + str(ts).encode("utf-8")
    sig = hmac.new(SWARM_SHARED_KEY, msg, hashlib.sha256).hexdigest()
    return {"body": payload, "ts": ts, "sig": sig}

def swarm_verify(wrapper: dict) -> dict | None:
    try:
        body = wrapper["body"]
        ts = float(wrapper["ts"])
        sig = wrapper["sig"]
    except Exception:
        return None
    if abs(time.time() - ts) > SWARM_MAX_SKEW:
        return None
    msg = json.dumps(body, sort_keys=True).encode("utf-8") + str(ts).encode("utf-8")
    expected = hmac.new(SWARM_SHARED_KEY, msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    return body

class SwarmMesh:
    def __init__(self, node_id: str, role: str):
        self.node_id = node_id
        self.role = role
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("", SWARM_PORT))
        self.peers = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def broadcast_loop(self, get_local_scores):
        while not self.stop_event.is_set():
            payload = {
                "node_id": self.node_id,
                "role": self.role,
                "ts": time.time(),
                "nic_scores": get_local_scores(),
            }
            wrapper = swarm_sign(payload)
            data = json.dumps(wrapper).encode("utf-8")
            try:
                self.sock.sendto(data, ("<broadcast>", SWARM_PORT))
            except Exception:
                pass
            time.sleep(SWARM_BROADCAST_INTERVAL)

    def listen_loop(self):
        self.sock.settimeout(1.0)
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                self._reap_peers()
                continue
            except Exception:
                continue
            try:
                wrapper = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            payload = swarm_verify(wrapper)
            if payload is None:
                continue
            node_id = payload.get("node_id")
            if not node_id or node_id == self.node_id:
                continue
            with self.lock:
                self.peers[node_id] = {
                    "role": payload.get("role", ROLE_WORKER),
                    "last_seen": payload.get("ts", time.time()),
                    "nic_scores": payload.get("nic_scores", {}),
                    "trusted": True,
                }

    def _reap_peers(self):
        now = time.time()
        with self.lock:
            dead = [nid for nid, info in self.peers.items()
                    if now - info["last_seen"] > NODE_TTL]
            for nid in dead:
                del self.peers[nid]

    def get_peer_snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.peers))

    def merge_scores(self, local_scores: dict):
        peers = self.get_peer_snapshot()
        agg = {}
        count = {}
        for nic, score in local_scores.items():
            agg[nic] = agg.get(nic, 0.0) + score
            count[nic] = count.get(nic, 0) + 1
        for nid, info in peers.items():
            if not info.get("trusted", False):
                continue
            for nic, score in info.get("nic_scores", {}).items():
                agg[nic] = agg.get(nic, 0.0) + score
                count[nic] = count.get(nic, 0) + 1
        if not agg:
            return local_scores
        return {nic: agg[nic] / count[nic] for nic in agg}

    def infer_queen_choice(self):
        peers = self.get_peer_snapshot()
        queen_scores = None
        for nid, info in peers.items():
            if info.get("role") == ROLE_QUEEN and info.get("trusted", False):
                queen_scores = info.get("nic_scores", {})
                break
        if not queen_scores:
            return None
        return min(queen_scores, key=queen_scores.get)

    def stop(self):
        self.stop_event.set()
        try:
            self.sock.close()
        except Exception:
            pass


# =========================
#  WATER / DATA PHYSICS ENGINE (HARDENED)
# =========================

class DataPhysicsEngine:
    def __init__(self):
        self.prev_pressure = {}
        self.alpha = 0.3  # smoothing factor

    def compute(self, nic_health: dict, throughput: dict):
        physics = {}
        for nic, health in nic_health.items():
            tx_bps = throughput.get(nic, {}).get("tx_bps", 0.0)
            rx_bps = throughput.get(nic, {}).get("rx_bps", 0.0)
            flow_mbps = (tx_bps + rx_bps) / 1e6

            raw_pressure = max(0.0, 100.0 - min(health, 100.0)) + flow_mbps
            prev = self.prev_pressure.get(nic, raw_pressure)
            pressure = self.alpha * raw_pressure + (1 - self.alpha) * prev
            self.prev_pressure[nic] = pressure

            turbulence = min(100.0, health / 2.0)
            physics[nic] = {
                "pressure": pressure,
                "turbulence": turbulence,
                "flow_mbps": flow_mbps,
            }
        return physics


# =========================
#  NETWORK BRAIN (WITH LEARNING + ALTERED STATES + HARDENING)
# =========================

class NetworkBrain:
    def __init__(self, node_id=None, role=ROLE_WORKER,
                 routing_mode: RoutingMode = RoutingMode.GLOBAL_BEST):
        self.node_id = node_id or f"node-{random.randint(1000, 9999)}"
        self.role = role
        self.routing_mode = routing_mode

        self.sensors = NICSensorCortex()
        self.flow = DataFlowMeter()
        self.heatmap = ThreatHeatmap()
        self.gpu_bridge = GPUAnomalyBridge(self.heatmap)
        self.ai_cortex = AICortex()
        self.swarm = SwarmMesh(self.node_id, self.role)
        self.hal = NetworkHAL()
        self.physics_engine = DataPhysicsEngine()

        self.stop_event = threading.Event()
        self.threads = []
        self.brain_state = {}
        self.brain_state_lock = threading.Lock()

        self.history = self._load_history()

        self.mode_consciousness = ConsciousnessMode.NEUTRAL
        self.manual_mode_override = False

        self.last_best_nic = None
        self.last_switch_time = 0.0
        self.switch_cooldown = 15.0  # seconds

    # ---- Persistence ----

    def _load_history(self):
        if HISTORY_PATH.exists():
            try:
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_history(self):
        try:
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass

    def _append_sample_to_history(self, state):
        t = time.localtime()
        hour = str(t.tm_hour)
        nic_health = state.get("nic_health", {})
        throughput = state.get("throughput", {})
        best = state.get("best_nic")

        bucket = self.history.setdefault(hour, {})
        for nic, health in nic_health.items():
            nic_entry = bucket.setdefault(nic, {
                "samples": 0,
                "avg_health": 0.0,
                "avg_tx_mbps": 0.0,
                "avg_rx_mbps": 0.0,
                "best_count": 0,
            })
            tx_mbps = throughput.get(nic, {}).get("tx_bps", 0.0) / 1e6
            rx_mbps = throughput.get(nic, {}).get("rx_bps", 0.0) / 1e6

            n = nic_entry["samples"] + 1
            nic_entry["avg_health"] = (nic_entry["avg_health"] * nic_entry["samples"] + health) / n
            nic_entry["avg_tx_mbps"] = (nic_entry["avg_tx_mbps"] * nic_entry["samples"] + tx_mbps) / n
            nic_entry["avg_rx_mbps"] = (nic_entry["avg_rx_mbps"] * nic_entry["samples"] + rx_mbps) / n
            nic_entry["samples"] = n
            if nic == best:
                nic_entry["best_count"] += 1

    def _predict_nic_bias_from_history(self):
        t = time.localtime()
        hour = str(t.tm_hour)
        bucket = self.history.get(hour, {})
        bias = {}
        for nic, stats in bucket.items():
            samples = stats.get("samples", 1)
            best_count = stats.get("best_count", 0)
            avg_health = stats.get("avg_health", 9999.0)
            success_ratio = best_count / max(samples, 1)
            bias[nic] = avg_health - success_ratio * 100.0
        return bias

    # ---- Consciousness control ----

    def set_consciousness_mode(self, mode: ConsciousnessMode):
        self.mode_consciousness = mode
        self.manual_mode_override = True

    def _detect_gaming(self):
        game_names = [
            "cs2.exe", "valorant.exe", "fortniteclient-win64-shipping.exe",
            "overwatch.exe", "r5apex.exe", "cod.exe", "mw3.exe"
        ]
        try:
            for p in psutil.process_iter(["name"]):
                name = (p.info.get("name") or "").lower()
                if any(g in name for g in game_names):
                    return True
        except Exception:
            pass
        return False

    # ---- State ----

    def snapshot_state(self):
        nic_health = self.sensors.get_scores()
        nic_threat = {n: self.heatmap.get_threat(n) for n in nic_health}
        merged = self.heatmap.apply(nic_health)
        throughput = self.flow.get_throughput()
        peers = self.swarm.get_peer_snapshot()
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        physics = self.physics_engine.compute(nic_health, throughput)
        best, second = self._decide(merged, throughput, peers, cpu, mem)
        state = {
            "node_id": self.node_id,
            "role": self.role,
            "routing_mode": self.routing_mode.value,
            "nic_health": nic_health,
            "nic_threat": nic_threat,
            "merged_scores": merged,
            "throughput": throughput,
            "peers": peers,
            "best_nic": best,
            "second_nic": second,
            "cpu_percent": cpu,
            "mem_percent": mem,
            "history": self.history,
            "consciousness_mode": self.mode_consciousness.value,
            "physics": physics,
        }
        with self.brain_state_lock:
            self.brain_state = state
        return state

    def get_state(self):
        with self.brain_state_lock:
            return dict(self.brain_state)

    def push_gpu_anomaly_scores(self, nic_scores: dict):
        self.gpu_bridge.push_scores(nic_scores)

    # ---- Decision ----

    def _decide(self, merged_scores: dict, throughput: dict, peers_snapshot: dict,
                cpu: float, mem: float):
        if not merged_scores:
            return None, None

        nic_features = {}
        for nic, score in merged_scores.items():
            tx = throughput.get(nic, {}).get("tx_bps", 0.0)
            rx = throughput.get(nic, {}).get("rx_bps", 0.0)
            nic_features[nic] = [
                score,
                tx / 1e6,
                rx / 1e6,
                cpu / 100.0,
                mem / 100.0,
            ]

        ai_raw = self.ai_cortex.score(nic_features)
        combined = {}

        for nic, base in merged_scores.items():
            tx_mbps = nic_features[nic][1]
            rx_mbps = nic_features[nic][2]
            ai = ai_raw.get(nic, 0.0)

            if self.mode_consciousness == ConsciousnessMode.GAMING:
                load_penalty = (tx_mbps + rx_mbps) * 4.0
                combined[nic] = base * 1.5 + ai + load_penalty
            elif self.mode_consciousness == ConsciousnessMode.DEEP_WATER:
                smoothness_reward = max(0.0, 50.0 - base)
                load_penalty = (tx_mbps + rx_mbps) * 0.5
                combined[nic] = base + ai + load_penalty - smoothness_reward
            else:
                load_penalty = (tx_mbps + rx_mbps) * 2.0
                combined[nic] = base + ai + load_penalty

        history_bias = self._predict_nic_bias_from_history()
        for nic in combined:
            combined[nic] += history_bias.get(nic, 0.0)

        consensus = self.swarm.merge_scores(combined)
        sorted_nics = sorted(consensus.items(), key=lambda x: x[1])
        best = sorted_nics[0][0]
        second = sorted_nics[1][0] if len(sorted_nics) > 1 else best

        now = time.time()
        if self.last_best_nic is not None and best != self.last_best_nic:
            if now - self.last_switch_time < self.switch_cooldown:
                best = self.last_best_nic
        if best != self.last_best_nic:
            self.last_best_nic = best
            self.last_switch_time = now

        if self.role == ROLE_WORKER:
            queen_choice = self.swarm.infer_queen_choice()
            if queen_choice and queen_choice in consensus:
                best = queen_choice

        return best, second

    # ---- Threads ----

    def _routing_control_thread(self):
        while not self.stop_event.is_set():
            if not self.manual_mode_override:
                if self._detect_gaming():
                    self.mode_consciousness = ConsciousnessMode.GAMING
                else:
                    cpu_now = psutil.cpu_percent(interval=None)
                    if cpu_now > 70:
                        self.mode_consciousness = ConsciousnessMode.DEEP_WATER
                    else:
                        self.mode_consciousness = ConsciousnessMode.NEUTRAL

            state = self.snapshot_state()
            self._append_sample_to_history(state)
            self._save_history()

            best = state["best_nic"]
            second = state["second_nic"]

            if self.routing_mode == RoutingMode.GLOBAL_BEST:
                self.hal.apply_global_best(best)
            elif self.routing_mode == RoutingMode.IN_OUT_SPLIT:
                self.hal.apply_in_out_split(best, second)
            elif self.routing_mode == RoutingMode.PRIORITY_SPLIT:
                self.hal.apply_priority_split(best, second)

            if self.mode_consciousness == ConsciousnessMode.GAMING and best:
                self.hal.apply_gaming_qos(
                    ["cs2.exe", "valorant.exe", "fortniteclient-win64-shipping.exe",
                     "overwatch.exe", "r5apex.exe", "cod.exe", "mw3.exe"],
                    best,
                )

            time.sleep(5.0)

    def _swarm_broadcast_thread(self):
        self.swarm.broadcast_loop(self.sensors.get_scores)

    def _swarm_listen_thread(self):
        self.swarm.listen_loop()

    def run(self):
        t1 = threading.Thread(target=self._swarm_broadcast_thread, daemon=True)
        t2 = threading.Thread(target=self._swarm_listen_thread, daemon=True)
        t3 = threading.Thread(target=self._routing_control_thread, daemon=True)
        self.threads.extend([t1, t2, t3])
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_event.set()
        self.sensors.stop()
        self.flow.stop()
        self.swarm.stop()


# =========================
#  PYSIDE6 ENTERPRISE SOC COCKPIT (TABBED)
# =========================

if HAVE_QT:
    class NetworkCockpit(QtWidgets.QWidget):
        def __init__(self, brain: NetworkBrain):
            super().__init__()
            self.brain = brain
            self.setWindowTitle("Network Brain SOC Console")
            self.resize(1400, 800)
            self._apply_soc_style()
            self._build_ui()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(1000)

        def _apply_soc_style(self):
            palette = self.palette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#0D1117"))
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#C9D1D9"))
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#010409"))
            palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#161B22"))
            palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#C9D1D9"))
            palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#21262D"))
            palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#C9D1D9"))
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#58A6FF"))
            palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#0D1117"))
            self.setPalette(palette)
            self.setAutoFillBackground(True)
            self.setStyleSheet("""
                QWidget {
                    font-family: "Segoe UI", "Consolas", monospace;
                    color: #C9D1D9;
                    background-color: #0D1117;
                }
                QTableWidget {
                    gridline-color: #30363D;
                    selection-background-color: #1F6FEB;
                    selection-color: #FFFFFF;
                    background-color: #010409;
                    alternate-background-color: #161B22;
                }
                QHeaderView::section {
                    background-color: #161B22;
                    color: #C9D1D9;
                    border: 1px solid #30363D;
                    padding: 4px;
                }
                QTextEdit {
                    border: 1px solid #30363D;
                    background-color: #010409;
                }
                QLabel {
                    font-size: 11px;
                }
                QPushButton {
                    border: 1px solid #30363D;
                    padding: 4px 10px;
                    background-color: #21262D;
                    border-radius: 3px;
                }
                QPushButton:checked {
                    background-color: #1F6FEB;
                    color: #FFFFFF;
                }
                QTabWidget::pane {
                    border: 1px solid #30363D;
                    top: -1px;
                }
                QTabBar::tab {
                    background: #161B22;
                    color: #C9D1D9;
                    padding: 6px 12px;
                    border: 1px solid #30363D;
                    border-bottom: none;
                }
                QTabBar::tab:selected {
                    background: #0D1117;
                    color: #FFFFFF;
                }
            """)

        def _build_ui(self):
            main_layout = QtWidgets.QVBoxLayout(self)

            # Top status bar
            status_bar = QtWidgets.QHBoxLayout()
            self.lbl_node = QtWidgets.QLabel()
            self.lbl_mode = QtWidgets.QLabel()
            self.lbl_sys = QtWidgets.QLabel()
            status_bar.addWidget(self.lbl_node)
            status_bar.addStretch()
            status_bar.addWidget(self.lbl_mode)
            status_bar.addStretch()
            status_bar.addWidget(self.lbl_sys)
            main_layout.addLayout(status_bar)

            # Consciousness control
            mode_bar = QtWidgets.QHBoxLayout()
            self.lbl_conscious = QtWidgets.QLabel()
            mode_bar.addWidget(self.lbl_conscious)

            self.btn_neutral = QtWidgets.QPushButton("Neutral")
            self.btn_gaming = QtWidgets.QPushButton("Gaming")
            self.btn_deep = QtWidgets.QPushButton("Deep Water")
            for b in (self.btn_neutral, self.btn_gaming, self.btn_deep):
                b.setCheckable(True)
                mode_bar.addWidget(b)

            self.btn_neutral.clicked.connect(
                lambda: self._set_mode_ui(ConsciousnessMode.NEUTRAL)
            )
            self.btn_gaming.clicked.connect(
                lambda: self._set_mode_ui(ConsciousnessMode.GAMING)
            )
            self.btn_deep.clicked.connect(
                lambda: self._set_mode_ui(ConsciousnessMode.DEEP_WATER)
            )

            mode_bar.addStretch()
            main_layout.addLayout(mode_bar)

            # Tabs
            self.tabs = QtWidgets.QTabWidget()
            main_layout.addWidget(self.tabs)

            # Live NIC tab
            self.tab_live = QtWidgets.QWidget()
            live_layout = QtWidgets.QVBoxLayout(self.tab_live)
            self.table_live = QtWidgets.QTableWidget()
            self.table_live.setColumnCount(11)
            self.table_live.setHorizontalHeaderLabels(
                [
                    "NIC", "Health", "Threat", "Merged",
                    "TX Mbps", "RX Mbps",
                    "CPU%", "MEM%", "Role",
                    "Best", "Second"
                ]
            )
            self.table_live.horizontalHeader().setStretchLastSection(True)
            self.table_live.setAlternatingRowColors(True)
            live_layout.addWidget(self.table_live)
            self.tabs.addTab(self.tab_live, "Live NIC")

            # Physics tab
            self.tab_phys = QtWidgets.QWidget()
            phys_layout = QtWidgets.QVBoxLayout(self.tab_phys)
            self.table_phys = QtWidgets.QTableWidget()
            self.table_phys.setColumnCount(4)
            self.table_phys.setHorizontalHeaderLabels(
                ["NIC", "Pressure", "Turbulence", "Flow Mbps"]
            )
            self.table_phys.horizontalHeader().setStretchLastSection(True)
            self.table_phys.setAlternatingRowColors(True)
            phys_layout.addWidget(self.table_phys)
            self.tabs.addTab(self.tab_phys, "Physics")

            # Learning tab
            self.tab_learn = QtWidgets.QWidget()
            learn_layout = QtWidgets.QVBoxLayout(self.tab_learn)
            self.table_learn = QtWidgets.QTableWidget()
            self.table_learn.setColumnCount(7)
            self.table_learn.setHorizontalHeaderLabels(
                ["Hour", "NIC", "Samples", "Avg Health", "Avg TX Mbps",
                 "Avg RX Mbps", "Success Ratio"]
            )
            self.table_learn.horizontalHeader().setStretchLastSection(True)
            self.table_learn.setAlternatingRowColors(True)
            learn_layout.addWidget(self.table_learn)
            self.tabs.addTab(self.tab_learn, "Learning")

            # Swarm tab
            self.tab_swarm = QtWidgets.QWidget()
            swarm_layout = QtWidgets.QVBoxLayout(self.tab_swarm)
            self.txt_swarm = QtWidgets.QTextEdit()
            self.txt_swarm.setReadOnly(True)
            swarm_layout.addWidget(self.txt_swarm)
            self.tabs.addTab(self.tab_swarm, "Swarm")

            # Security / Settings tab
            self.tab_sec = QtWidgets.QWidget()
            sec_layout = QtWidgets.QVBoxLayout(self.tab_sec)
            self.lbl_sec_info = QtWidgets.QLabel(
                "Security posture:\n"
                "- Swarm: HMAC-authenticated, time-bounded\n"
                "- Sensors: multi-target, median-based\n"
                "- Scoring: outlier rejection, history bias\n"
                "- Routing: cooldown, fail-safe bias"
            )
            self.lbl_sec_info.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
            sec_layout.addWidget(self.lbl_sec_info)
            sec_layout.addStretch()
            self.tabs.addTab(self.tab_sec, "Security")

        def _set_mode_ui(self, mode: ConsciousnessMode):
            self.brain.set_consciousness_mode(mode)
            self.btn_neutral.setChecked(mode == ConsciousnessMode.NEUTRAL)
            self.btn_gaming.setChecked(mode == ConsciousnessMode.GAMING)
            self.btn_deep.setChecked(mode == ConsciousnessMode.DEEP_WATER)

        def refresh(self):
            state = self.brain.get_state()
            nic_health = state.get("nic_health", {})
            nic_threat = state.get("nic_threat", {})
            merged = state.get("merged_scores", {})
            throughput = state.get("throughput", {})
            physics = state.get("physics", {})
            best = state.get("best_nic")
            second = state.get("second_nic")
            role = state.get("role")
            mode = state.get("routing_mode")
            cpu = state.get("cpu_percent", 0.0)
            mem = state.get("mem_percent", 0.0)
            history = state.get("history", {})
            mode_conscious = state.get("consciousness_mode", "neutral")
            peers = state.get("peers", {})

            self.lbl_node.setText(
                f"Node: {state.get('node_id')} | Role: {role}"
            )
            self.lbl_mode.setText(
                f"Routing: {mode} | Consciousness: {mode_conscious.upper()} | AI: {self.brain.ai_cortex.backend.upper()}"
            )
            self.lbl_sys.setText(
                f"CPU: {cpu:.1f}% | MEM: {mem:.1f}%"
            )
            self.lbl_conscious.setText(
                f"Consciousness: {mode_conscious.upper()}"
            )

            if mode_conscious == ConsciousnessMode.NEUTRAL.value:
                self.btn_neutral.setChecked(True)
                self.btn_gaming.setChecked(False)
                self.btn_deep.setChecked(False)
            elif mode_conscious == ConsciousnessMode.GAMING.value:
                self.btn_neutral.setChecked(False)
                self.btn_gaming.setChecked(True)
                self.btn_deep.setChecked(False)
            else:
                self.btn_neutral.setChecked(False)
                self.btn_gaming.setChecked(False)
                self.btn_deep.setChecked(True)

            # Live NIC tab
            self.table_live.setRowCount(len(nic_health))
            for row, nic in enumerate(sorted(nic_health.keys())):
                tx_mbps = throughput.get(nic, {}).get("tx_bps", 0.0) / 1e6
                rx_mbps = throughput.get(nic, {}).get("rx_bps", 0.0) / 1e6
                self.table_live.setItem(row, 0, QtWidgets.QTableWidgetItem(nic))
                self.table_live.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{nic_health[nic]:.1f}"))
                self.table_live.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{nic_threat.get(nic, 0.0):.2f}"))
                self.table_live.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{merged.get(nic, 0.0):.1f}"))
                self.table_live.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{tx_mbps:.2f}"))
                self.table_live.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{rx_mbps:.2f}"))
                self.table_live.setItem(row, 6, QtWidgets.QTableWidgetItem(f"{cpu:.1f}"))
                self.table_live.setItem(row, 7, QtWidgets.QTableWidgetItem(f"{mem:.1f}"))
                self.table_live.setItem(row, 8, QtWidgets.QTableWidgetItem(role))
                self.table_live.setItem(row, 9, QtWidgets.QTableWidgetItem("YES" if nic == best else ""))
                self.table_live.setItem(row, 10, QtWidgets.QTableWidgetItem("YES" if nic == second else ""))

            # Physics tab
            self.table_phys.setRowCount(len(physics))
            for row, nic in enumerate(sorted(physics.keys())):
                phys = physics[nic]
                self.table_phys.setItem(row, 0, QtWidgets.QTableWidgetItem(nic))
                self.table_phys.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{phys.get('pressure', 0.0):.1f}"))
                self.table_phys.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{phys.get('turbulence', 0.0):.1f}"))
                self.table_phys.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{phys.get('flow_mbps', 0.0):.2f}"))

            # Learning tab
            rows = []
            for hour, bucket in history.items():
                for nic, stats in bucket.items():
                    samples = stats.get("samples", 0)
                    avg_health = stats.get("avg_health", 0.0)
                    avg_tx = stats.get("avg_tx_mbps", 0.0)
                    avg_rx = stats.get("avg_rx_mbps", 0.0)
                    best_count = stats.get("best_count", 0)
                    success_ratio = best_count / max(samples, 1) if samples > 0 else 0.0
                    rows.append((int(hour), nic, samples, avg_health, avg_tx, avg_rx, success_ratio))

            rows.sort(key=lambda r: (r[0], r[1]))
            self.table_learn.setRowCount(len(rows))
            for i, (hour, nic, samples, avg_health, avg_tx, avg_rx, success_ratio) in enumerate(rows):
                self.table_learn.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{hour:02d}"))
                self.table_learn.setItem(i, 1, QtWidgets.QTableWidgetItem(nic))
                self.table_learn.setItem(i, 2, QtWidgets.QTableWidgetItem(str(samples)))
                self.table_learn.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{avg_health:.1f}"))
                self.table_learn.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{avg_tx:.2f}"))
                self.table_learn.setItem(i, 5, QtWidgets.QTableWidgetItem(f"{avg_rx:.2f}"))
                self.table_learn.setItem(i, 6, QtWidgets.QTableWidgetItem(f"{success_ratio:.2f}"))

            # Swarm tab
            self.txt_swarm.setPlainText(json.dumps(peers, indent=2))


# =========================
#  MAIN
# =========================

def main():
    role = ROLE_QUEEN
    mode = RoutingMode.PRIORITY_SPLIT

    brain = NetworkBrain(role=role, routing_mode=mode)
    brain.run()

    if HAVE_QT:
        app = QtWidgets.QApplication(sys.argv)
        cockpit = NetworkCockpit(brain)
        cockpit.show()
        app.exec()
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    brain.stop()

if __name__ == "__main__":
    main()
