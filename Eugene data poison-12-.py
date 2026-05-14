#!/usr/bin/env python3
"""
Sentinel Tier‑29 – Full ASI‑flavored SAFE+++ Organism (Single Unified File)

This file is a COMPLETE, SELF‑CONTAINED synthetic defensive organism with:

Tiers 1‑26 (already integrated and extended):
- Multimodal telemetry (core + logs + net + behavior + OS‑flavored events)
- IsolationForest, Autoencoder, Transformer temporal encoder, temporal conv
- ONNX runtime hook (optional real model)
- GPU scoring (CuPy if available)
- Swarm + UDP gossip + Borg queens
- Attack chain engine + causal graph (NetworkX)
- Awareness engine (SENTINEL / HYPER_FLOW / DEEP_DREAM / RECOVERY_FLOW / DIAGNOSTIC)
- Self‑healing integrity organ + functional organ spawning
- Policy engine (log / micro / macro)
- Cockpit GUI (PySide6) with threat matrix and causal summary
- SAFE++ guarantees: no real ETW, no process killing, no firewall, no registry

Tier‑27/28/29 Upgrades (this file):
- Stronger ASI‑flavored brain orchestration (multi‑agent + regime + causal)
- Synthetic adversary engine (scenario generation, but SAFE – no real attacks)
- Counterfactual simulator (what‑if narratives, SAFE, no real actions)
- Defense planner (predictive defense planning from attack graph + counterfactuals)
- Enhanced causal graph usage (risk propagation, hub scoring)
- Extended organ spawning semantics (roles + hints to brain)
- Richer threat narrative export for cockpit
- Still SAFE+++ (no OS control, no destructive actions)

NOTE: This is a *simulation organism* – architecturally ASI‑flavored, operationally SAFE.
"""

import sys
import time
import math
import json
import random
import logging
import importlib
import threading
import socket
from collections import deque
from types import SimpleNamespace

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_MODE = "DEFEND"   # OBSERVE / DEFEND / AGGRESSIVE
LOOP_BASE_INTERVAL = 0.5
AUTO_CALIBRATION_INTERVAL = 24 * 60 * 60

GOSSIP_PORT = 49321
GOSSIP_BROADCAST_INTERVAL = 5.0
GOSSIP_NODE_ID = f"node-{random.randint(1000,9999)}"

ONNX_MODEL_PATH = None  # set to a real path if you want to load a model

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sentinel.tier29")

# ---------------------------------------------------------------------
# Capability loader
# ---------------------------------------------------------------------

MODULES = {
    "numpy":        {"required": True,  "role": "core"},
    "psutil":       {"required": False, "role": "telemetry"},
    "torch":        {"required": False, "role": "dl"},
    "cupy":         {"required": False, "role": "gpu"},
    "PySide6":      {"required": False, "role": "cockpit"},
    "sklearn.ensemble": {"required": False, "role": "iforest"},
    "onnxruntime":  {"required": False, "role": "onnx"},
    "networkx":     {"required": False, "role": "causal_graph"},
}

def load_capabilities():
    caps = SimpleNamespace()
    caps.modules = {}
    caps.flags = {}

    for name, meta in MODULES.items():
        try:
            m = importlib.import_module(name)
            caps.modules[name] = m
            caps.flags[name] = True
            log.info(f"[CAP] {name} OK ({meta.get('role','core')})")
        except Exception as e:
            caps.modules[name] = None
            caps.flags[name] = False
            if meta.get("required", False):
                log.error(f"[CAP] Required module missing: {name}")
                raise RuntimeError(f"Required module missing: {name}") from e
            log.warning(f"[CAP] {name} unavailable: {e}")

    caps.has_gpu     = caps.flags.get("cupy", False)
    caps.has_dl      = caps.flags.get("torch", False)
    caps.has_cockpit = caps.flags.get("PySide6", False)
    caps.has_iforest = caps.flags.get("sklearn.ensemble", False)
    caps.has_onnx    = caps.flags.get("onnxruntime", False)
    caps.has_nx      = caps.flags.get("networkx", False)
    return caps

CAPS = load_capabilities()

np = CAPS.modules.get("numpy", None)
psutil = CAPS.modules.get("psutil", None)
torch = CAPS.modules.get("torch", None)
cupy = CAPS.modules.get("cupy", None)
sk_ensemble = CAPS.modules.get("sklearn.ensemble", None)
onnxruntime = CAPS.modules.get("onnxruntime", None)
networkx = CAPS.modules.get("networkx", None)

# ---------------------------------------------------------------------
# Synthetic OS‑flavored event generator (SAFE+++)
# ---------------------------------------------------------------------

class SyntheticOSEvents:
    """
    Simulates ETW‑like events, process spawns, file mods, and net connects
    WITHOUT touching the real OS. Used to feed the attack chain engine.
    """
    def __init__(self):
        self.proc_names = ["chrome.exe", "powershell.exe", "cmd.exe", "svchost.exe", "python.exe"]
        self.hosts = ["10.0.0.5", "10.0.0.10", "192.168.1.20", "8.8.8.8"]
        self.files = ["C:\\temp\\a.tmp", "C:\\Windows\\System32\\x.dll", "C:\\Users\\user\\doc.txt"]

    def random_event(self):
        r = random.random()
        if r < 0.25:
            return ("proc_spawn", random.choice(self.proc_names))
        elif r < 0.5:
            return ("net_connect", random.choice(self.hosts))
        elif r < 0.75:
            return ("file_mod", random.choice(self.files))
        else:
            return ("powershell", "powershell.exe")

# ---------------------------------------------------------------------
# Telemetry + multimodal bus (SAFE+++)
# ---------------------------------------------------------------------

class TelemetryBus:
    def __init__(self):
        self._tick = 0
        self.os_events = SyntheticOSEvents()

    def collect_core(self):
        self._tick += 1
        now = time.time()
        cpu_load = (self._tick % 100) / 100.0
        io_rate = (self._tick % 50) / 10.0
        net_rate = (self._tick % 70) / 10.0
        proc_churn = (self._tick % 30)
        return {
            "tick": self._tick,
            "time": now,
            "cpu_load": cpu_load,
            "io_rate": io_rate,
            "net_rate": net_rate,
            "proc_churn": proc_churn,
        }

    def collect_logs(self):
        return {
            "error_rate": random.random() * 0.1,
            "warn_rate": random.random() * 0.2,
        }

    def collect_net_packets(self):
        return {
            "packet_rate": random.random() * 100.0,
            "suspicious_flows": random.randint(0, 3),
        }

    def collect_behavior(self):
        return {
            "login_anomalies": random.randint(0, 2),
            "file_touch_bursts": random.randint(0, 4),
        }

    def collect_os_event(self):
        etype, entity = self.os_events.random_event()
        return {"etype": etype, "entity": entity}

    def collect_multimodal(self):
        core = self.collect_core()
        logs = self.collect_logs()
        netp = self.collect_net_packets()
        beh = self.collect_behavior()
        fused = {
            **core,
            "log_error_rate": logs["error_rate"],
            "log_warn_rate": logs["warn_rate"],
            "packet_rate": netp["packet_rate"],
            "suspicious_flows": netp["suspicious_flows"],
            "login_anomalies": beh["login_anomalies"],
            "file_touch_bursts": beh["file_touch_bursts"],
        }
        return fused

# ---------------------------------------------------------------------
# Organ Messaging Bus
# ---------------------------------------------------------------------

class OrganMessageBus:
    def __init__(self):
        self.messages = deque(maxlen=256)

    def send(self, source, target, payload):
        self.messages.append({
            "time": time.time(),
            "source": source,
            "target": target,
            "payload": payload,
        })

    def drain_for(self, target):
        out = []
        keep = deque(maxlen=self.messages.maxlen)
        while self.messages:
            msg = self.messages.popleft()
            if msg["target"] == target or msg["target"] == "*":
                out.append(msg)
            else:
                keep.append(msg)
        self.messages = keep
        return out

# ---------------------------------------------------------------------
# Tier‑29 Anomaly Engine (same as Tier‑26, kept intact)
# ---------------------------------------------------------------------

class Tier29AnomalyEngine:
    """
    Tier‑29 anomaly stack:
    - Baseline anomalies
    - IsolationForest (if available)
    - Autoencoder (if torch available)
    - Transformer temporal encoder (multi‑layer, dropout)
    - GPU scoring (if CuPy available)
    - ONNX runtime head (real model loading hook)
    - Temporal convolutional scoring
    - Self‑tuning thresholds via meta‑learning
    - Synthetic adversary boost (from SyntheticAdversaryEngine)
    """
    def __init__(self, history_len=512):
        self.history = deque(maxlen=history_len)
        self.counter = 0

        self.iforest = None
        self.iforest_ready = False
        if CAPS.has_iforest and np is not None:
            try:
                IsolationForest = sk_ensemble.IsolationForest
                self.iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                log.info("[ANOM] IsolationForest available")
            except Exception as e:
                log.warning(f"[ANOM] IsolationForest init failed: {e}")
                self.iforest = None

        self.autoencoder = None
        self.autoencoder_ready = False
        self.transformer = None
        self.transformer_ready = False
        if CAPS.has_dl and torch is not None:
            try:
                self.autoencoder = self._build_autoencoder(input_dim=8)
                self.autoencoder_ready = False
                self.transformer = self._build_transformer(input_dim=8, d_model=32, nhead=4, num_layers=2)
                self.transformer_ready = False
                log.info("[ANOM] Autoencoder + Transformer available")
            except Exception as e:
                log.warning(f"[ANOM] DL models init failed: {e}")
                self.autoencoder = None
                self.transformer = None

        self.onnx_session = None
        if CAPS.has_onnx and onnxruntime is not None and ONNX_MODEL_PATH:
            try:
                self.onnx_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
                log.info(f"[ANOM] ONNX model loaded from {ONNX_MODEL_PATH}")
            except Exception as e:
                log.warning(f"[ANOM] ONNX model load failed: {e}")
                self.onnx_session = None
        elif CAPS.has_onnx:
            log.info("[ANOM] ONNX runtime available (no model path configured)")

        self.threshold = 2.0
        self.meta_lr = 0.01
        self.temporal_kernel = [0.2, 0.5, 0.2]

        self.adversary_mode = False
        self.adversary_intensity = 0.0

    def _build_autoencoder(self, input_dim):
        class AE(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.enc = torch.nn.Sequential(
                    torch.nn.Linear(d, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 8),
                )
                self.dec = torch.nn.Sequential(
                    torch.nn.Linear(8, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, d),
                )

            def forward(self, x):
                z = self.enc(x)
                out = self.dec(z)
                return out
        return AE(input_dim)

    def _build_transformer(self, input_dim, d_model=32, nhead=4, num_layers=2):
        class TemporalEncoder(torch.nn.Module):
            def __init__(self, in_dim, d_model, nhead, num_layers):
                super().__init__()
                self.proj = torch.nn.Linear(in_dim, d_model)
                enc_layer = torch.nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
                )
                self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=num_layers)
                self.pos_embed = torch.nn.Parameter(torch.zeros(1, 64, d_model))
                self.head = torch.nn.Linear(d_model, 1)

            def forward(self, x_seq):
                L = x_seq.shape[1]
                x = self.proj(x_seq)
                pos = self.pos_embed[:, :L, :]
                h = self.encoder(x + pos)
                out = self.head(h[:, -1, :])
                return out
        return TemporalEncoder(input_dim, d_model, nhead, num_layers)

    def _to_vector(self, signals):
        return [
            signals["cpu_load"],
            signals["io_rate"],
            signals["net_rate"],
            float(signals["proc_churn"]) / 30.0,
            signals.get("log_error_rate", 0.0),
            signals.get("log_warn_rate", 0.0),
            signals.get("packet_rate", 0.0) / 100.0,
            float(signals.get("suspicious_flows", 0)),
        ]

    def _temporal_conv_score(self):
        if len(self.history) < len(self.temporal_kernel):
            return 0.0
        vals = [h["base_score"] for h in list(self.history)[-len(self.temporal_kernel):]]
        score = 0.0
        for k, v in zip(self.temporal_kernel, vals):
            score += k * v
        return score

    def _gpu_boost(self, score):
        if not CAPS.has_gpu or cupy is None:
            return score
        try:
            arr = cupy.array([score], dtype=cupy.float32)
            arr = arr * 1.0
            return float(arr[0])
        except Exception:
            return score

    def _iforest_score(self, vec):
        if self.iforest is None or np is None:
            return None
        x = np.array(vec, dtype=float).reshape(1, -1)
        if not self.iforest_ready and len(self.history) >= 100:
            data = np.array([h["vec"] for h in self.history], dtype=float)
            try:
                self.iforest.fit(data)
                self.iforest_ready = True
                log.info("[ANOM] IsolationForest trained")
            except Exception as e:
                log.warning(f"[ANOM] IsolationForest fit failed: {e}")
                return None
        if not self.iforest_ready:
            return None
        try:
            score = -float(self.iforest.score_samples(x)[0])
            return score
        except Exception as e:
            log.warning(f"[ANOM] IsolationForest scoring failed: {e}")
            return None

    def _autoencoder_score(self, vec):
        if self.autoencoder is None or torch is None:
            return None
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        if not self.autoencoder_ready and len(self.history) >= 150:
            data = torch.tensor([h["vec"] for h in self.history], dtype=torch.float32)
            opt = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
            self.autoencoder.train()
            for _ in range(20):
                idx = torch.randint(0, data.shape[0], (16,))
                batch = data[idx]
                opt.zero_grad()
                out = self.autoencoder(batch)
                loss = torch.nn.functional.mse_loss(out, batch)
                loss.backward()
                opt.step()
            self.autoencoder_ready = True
            log.info("[ANOM] Autoencoder trained")
        self.autoencoder.eval()
        with torch.no_grad():
            out = self.autoencoder(x)
            loss = torch.nn.functional.mse_loss(out, x)
        return float(loss.item())

    def _transformer_score(self):
        if self.transformer is None or torch is None:
            return None
        if len(self.history) < 16:
            return None
        seq = [h["vec"] for h in list(self.history)[-32:]]
        x_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        if not self.transformer_ready and len(self.history) >= 250:
            self.transformer_ready = True
            log.info("[ANOM] Transformer temporal encoder marked ready")
        self.transformer.eval()
        with torch.no_grad():
            out = self.transformer(x_seq)
        return float(torch.relu(out).item())

    def _onnx_score(self, vec):
        if self.onnx_session is None or np is None:
            return None
        try:
            inp = np.array(vec, dtype=np.float32).reshape(1, -1)
            inputs = {self.onnx_session.get_inputs()[0].name: inp}
            out = self.onnx_session.run(None, inputs)
            val = float(out[0].ravel()[0])
            return max(0.0, val)
        except Exception as e:
            log.warning(f"[ANOM] ONNX scoring failed: {e}")
            return None

    def _meta_learn_threshold(self, scores):
        if not scores:
            return
        avg = sum(scores) / len(scores)
        if avg > self.threshold:
            self.threshold += self.meta_lr * (avg - self.threshold)
        else:
            self.threshold -= self.meta_lr * (self.threshold - avg)
        self.threshold = max(0.5, min(10.0, self.threshold))

    def _synthetic_adversary_boost(self, fused):
        if not self.adversary_mode:
            return fused
        t = time.time()
        bump = 0.3 * math.sin(t / 5.0) + random.uniform(0.0, self.adversary_intensity)
        return fused + max(0.0, bump)

    def evaluate(self, signals):
        self.counter += 1
        vec = self._to_vector(signals)

        base_score = (
            abs(vec[0] - 0.5) * 2.0 +
            vec[1] * 0.3 +
            vec[2] * 0.3 +
            vec[3] * 1.0 +
            vec[4] * 3.0 +
            vec[5] * 2.0 +
            vec[6] * 2.0 +
            vec[7] * 1.5
        )

        if_score = self._iforest_score(vec)
        ae_score = self._autoencoder_score(vec)
        tr_score = self._transformer_score()
        onnx_score = self._onnx_score(vec)
        temp_score = self._temporal_conv_score()

        scores = [base_score]
        for s in (if_score, ae_score, tr_score, temp_score, onnx_score):
            if s is not None:
                scores.append(s)

        fused = sum(scores) / len(scores) if scores else base_score
        fused = self._gpu_boost(fused)
        fused = self._synthetic_adversary_boost(fused)

        self.history.append({
            "time": signals["time"],
            "vec": vec,
            "base_score": base_score,
            "iforest": if_score,
            "autoencoder": ae_score,
            "transformer": tr_score,
            "onnx": onnx_score,
            "temporal": temp_score,
            "fused": fused,
        })

        self._meta_learn_threshold(scores)

        anomalies = []
        if fused > self.threshold:
            anomalies.append({
                "label": "tier29_anomaly",
                "score": fused,
                "pid": 1234,
                "engines": {
                    "base": base_score,
                    "iforest": if_score,
                    "autoencoder": ae_score,
                    "transformer": tr_score,
                    "onnx": onnx_score,
                    "temporal": temp_score,
                },
                "threshold": self.threshold,
            })

        if self.counter % 40 == 0:
            anomalies.append({
                "label": "synthetic_periodic",
                "score": self.threshold + 0.5,
                "pid": 5678,
            })

        return anomalies

# ---------------------------------------------------------------------
# Swarm engine + UDP gossip
# ---------------------------------------------------------------------

class SwarmEngine:
    def __init__(self):
        self.nodes = {}
        now = time.time()
        for i in range(1, 4):
            nid = f"node-{i}"
            self.nodes[nid] = {
                "id": nid,
                "trust": 0.9 - (i - 1) * 0.03,
                "last_seen": now,
                "state": "ok",
                "role": "worker" if i > 1 else "leader",
            }
        self.gossip_last = 0.0
        self.sock = None
        self._init_socket()

    def _init_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.sock.settimeout(0.001)
        except Exception as e:
            log.warning(f"[SWARM] UDP socket init failed: {e}")
            self.sock = None

    def _gossip_send(self):
        if self.sock is None:
            return
        now = time.time()
        if now - self.gossip_last < GOSSIP_BROADCAST_INTERVAL:
            return
        self.gossip_last = now
        msg = json.dumps({
            "id": GOSSIP_NODE_ID,
            "ts": now,
            "trust": 0.9,
            "role": "sentinel",
        }).encode("utf-8")
        try:
            self.sock.sendto(msg, ("255.255.255.255", GOSSIP_PORT))
        except Exception:
            pass

    def _gossip_recv(self):
        if self.sock is None:
            return
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                break
            except Exception:
                break
            try:
                obj = json.loads(data.decode("utf-8"))
                nid = obj.get("id")
                if not nid or nid == GOSSIP_NODE_ID:
                    continue
                trust = float(obj.get("trust", 0.8))
                role = obj.get("role", "worker")
                now = time.time()
                self.nodes.setdefault(nid, {
                    "id": nid,
                    "trust": trust,
                    "last_seen": now,
                    "state": "ok",
                    "role": role,
                })
                self.nodes[nid]["trust"] = trust
                self.nodes[nid]["last_seen"] = now
                self.nodes[nid]["role"] = role
            except Exception:
                continue

    def simulate_nodes(self):
        now = time.time()
        for nid, n in self.nodes.items():
            dt = now - n["last_seen"]
            n["last_seen"] = now
            n["trust"] *= 0.99 ** (dt / 5.0)
            n["trust"] += random.uniform(-0.005, 0.005)
            n["trust"] = max(0.0, min(1.0, n["trust"]))
            if n["trust"] < 0.3:
                n["state"] = "suspect"
            elif n["trust"] < 0.6:
                n["state"] = "watch"
            else:
                n["state"] = "ok"

    def update(self, signals, anomalies):
        self._gossip_send()
        self._gossip_recv()
        self.simulate_nodes()
        risk = sum(1.0 - n["trust"] for n in self.nodes.values())
        return {
            "nodes": list(self.nodes.values()),
            "risk": risk,
        }

# ---------------------------------------------------------------------
# Intelligent Water Data Physics Engine
# ---------------------------------------------------------------------

class IntelligentWaterEngine:
    def __init__(self, history_len=64):
        self.history = deque(maxlen=history_len)

    def update(self, signals, tension):
        self.history.append({
            "time": signals["time"],
            "cpu_load": signals["cpu_load"],
            "io_rate": signals["io_rate"],
            "net_rate": signals["net_rate"],
            "proc_churn": signals["proc_churn"],
        })

        flow = self._compute_flow()
        pressure = self._compute_pressure(tension)
        phase = self._compute_phase(pressure)

        return {
            "flow": flow,
            "pressure": pressure,
            "phase": phase,
        }

    def _compute_flow(self):
        if len(self.history) < 2:
            return 0.0
        a = self.history[-2]
        b = self.history[-1]
        dt = max(b["time"] - a["time"], 1e-6)
        dcpu = abs(b["cpu_load"] - a["cpu_load"])
        dio = abs(b["io_rate"] - a["io_rate"])
        dnet = abs(b["net_rate"] - a["net_rate"])
        dproc = abs(b["proc_churn"] - a["proc_churn"]) / 10.0
        return (dcpu + dio + dnet + dproc) / dt

    def _compute_pressure(self, tension):
        if not self.history:
            return tension
        h = self.history[-1]
        load = h["cpu_load"] + (h["io_rate"] / 10.0) + (h["net_rate"] / 10.0)
        return 0.6 * tension + 0.4 * load * 5.0

    def _compute_phase(self, pressure):
        if pressure < 1.0:
            return "LIQUID"
        elif pressure < 4.0:
            return "VAPOR"
        else:
            return "PLASMA"

# ---------------------------------------------------------------------
# Data Tree
# ---------------------------------------------------------------------

class DataTree:
    def __init__(self):
        self.root = {
            "type": "root",
            "children": [],
            "rings": [],
        }

    def add_ring(self, tension, flow, pressure, phase, anomalies, swarm, borg_risk, chain_risk, brain_risk):
        ring = {
            "tension": tension,
            "flow": flow,
            "pressure": pressure,
            "phase": phase,
            "anomaly_count": len(anomalies),
            "swarm_risk": swarm.get("risk", 0.0),
            "borg_risk": borg_risk,
            "chain_risk": chain_risk,
            "brain_risk": brain_risk,
            "timestamp": time.time(),
        }
        self.root["rings"].append(ring)

    def add_leaf(self, signals):
        leaf = {
            "type": "leaf",
            "signals": signals,
            "timestamp": signals.get("time", time.time()),
        }
        self._attach_leaf(leaf)

    def _attach_leaf(self, leaf):
        key = (
            round(leaf["signals"]["cpu_load"], 1),
            round(leaf["signals"]["io_rate"], 1),
            round(leaf["signals"]["net_rate"], 1),
        )
        branch = None
        for child in self.root["children"]:
            if child["key"] == key:
                branch = child
                break
        if branch is None:
            branch = {
                "type": "branch",
                "key": key,
                "children": [],
            }
            self.root["children"].append(branch)
        branch["children"].append(leaf)

    def stats(self):
        branch_count = len(self.root["children"])
        leaf_count = sum(len(b["children"]) for b in self.root["children"])
        ring_count = len(self.root["rings"])
        return {
            "branches": branch_count,
            "leaves": leaf_count,
            "rings": ring_count,
        }

# ---------------------------------------------------------------------
# Borg Queens & Borg Collective
# ---------------------------------------------------------------------

class BorgQueen:
    def __init__(self, queen_id, origin="root"):
        self.id = queen_id
        self.origin = origin
        self.trust = 1.0
        self.nodes = {}
        self.created_at = time.time()
        self.last_update = self.created_at

    def attach_node(self, node_id, initial_trust=0.9):
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "trust": initial_trust,
                "state": "ok",
                "last_seen": time.time(),
            }

    def update_node(self, node_id, delta_trust=-0.01):
        n = self.nodes.get(node_id)
        if not n:
            return
        n["trust"] = max(0.0, min(1.0, n["trust"] + delta_trust))
        n["last_seen"] = time.time()
        if n["trust"] < 0.3:
            n["state"] = "suspect"
        elif n["trust"] < 0.6:
            n["state"] = "watch"
        else:
            n["state"] = "ok"
        self.last_update = time.time()

    def aggregate_risk(self):
        if not self.nodes:
            return 0.0
        return sum(1.0 - n["trust"] for n in self.nodes.values())

class BorgCollective:
    def __init__(self):
        self.queens = {}

    def ensure_queen(self, queen_id, origin="root"):
        if queen_id not in self.queens:
            self.queens[queen_id] = BorgQueen(queen_id, origin)
        return self.queens[queen_id]

    def grow_node(self, queen_id, node_id):
        q = self.ensure_queen(queen_id)
        q.attach_node(node_id)

    def update_from_swarm(self, swarm_state):
        q = self.ensure_queen("queen-alpha", origin="swarm")
        for n in swarm_state.get("nodes", []):
            q.attach_node(n["id"], initial_trust=n["trust"])
            if n["state"] == "suspect":
                q.update_node(n["id"], delta_trust=-0.05)

    def total_risk(self):
        return sum(q.aggregate_risk() for q in self.queens.values())

    def snapshot(self):
        return {
            "queens": {
                qid: {
                    "id": q.id,
                    "origin": q.origin,
                    "trust": q.trust,
                    "nodes": list(q.nodes.values()),
                }
                for qid, q in self.queens.items()
            }
        }

# ---------------------------------------------------------------------
# Real‑Time Queen
# ---------------------------------------------------------------------

class RealTimeQueen:
    def __init__(self):
        self.nodes = {}

    def update(self, node_id, events):
        self.nodes[node_id] = events

    def global_risk(self):
        risk = {}
        for node, evts in self.nodes.items():
            for e in evts:
                ent = e.get("entity")
                score = e.get("score", 0.0)
                if ent is None:
                    continue
                risk[ent] = risk.get(ent, 0.0) + score
        return {k: v for k, v in risk.items() if v > 1.5}

# ---------------------------------------------------------------------
# Attack Chain Engine + Causal Graph
# ---------------------------------------------------------------------

class SecEvent:
    def __init__(self, etype, entity, meta=None):
        self.ts = time.time()
        self.type = etype
        self.entity = entity
        self.meta = meta or {}

class AttackChainEngine:
    def __init__(self, window=120):
        self.events = deque()
        self.window = window
        self.causal_graph = networkx.DiGraph() if CAPS.has_nx and networkx is not None else None

    def add_event(self, event_type, data):
        now = time.time()
        self.events.append((now, event_type, data))
        self._cleanup(now)
        self._update_causal_graph(event_type, data)

    def _cleanup(self, now):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def _update_causal_graph(self, event_type, data):
        if self.causal_graph is None:
            return
        ent = data.get("entity", "unknown")
        node = f"{ent}:{event_type}"
        self.causal_graph.add_node(node, entity=ent, etype=event_type, ts=time.time())
        for _, et, d in list(self.events)[-10:]:
            prev_ent = d.get("entity", "unknown")
            prev_node = f"{prev_ent}:{et}"
            if prev_node != node:
                w = self.causal_graph.get_edge_data(prev_node, node, {}).get("weight", 0.0)
                self.causal_graph.add_edge(prev_node, node, weight=w + 1.0)

        for u, v, data_e in list(self.causal_graph.edges(data=True)):
            data_e["weight"] *= 0.995
            if data_e["weight"] < 0.1:
                self.causal_graph.remove_edge(u, v)

    def detect(self):
        types = [e[1] for e in self.events]
        chains = []

        if all(x in types for x in ["proc_spawn", "powershell", "net_connect"]):
            chains.append(("LOLBIN_ATTACK", 0.9))

        if types.count("proc_spawn") > 5 and "net_connect" in types:
            chains.append(("PROCESS_STORM", 0.8))

        if "file_mod" in types and "net_connect" in types:
            chains.append(("PERSISTENCE_EXFIL", 0.85))

        return chains

    def predictive_attack_graph(self):
        graph = {}
        for _, etype, data in self.events:
            ent = data.get("entity", "unknown")
            if ent not in graph:
                graph[ent] = set()
            graph[ent].add(etype)
        return {k: sorted(list(v)) for k, v in graph.items()}

    def causal_summary(self):
        if self.causal_graph is None:
            return {}
        deg = {}
        for n in self.causal_graph.nodes():
            deg[n] = self.causal_graph.out_degree(n, weight="weight")
        top = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:5]
        return {k: v for k, v in top}

    def causal_risk(self):
        if self.causal_graph is None:
            return 0.0
        total = 0.0
        for u, v, data_e in self.causal_graph.edges(data=True):
            total += data_e.get("weight", 0.0)
        return total / max(1, self.causal_graph.number_of_nodes())

class EventBus:
    def __init__(self):
        self.subscribers = []
        self.queue = deque()
        self._running = False

    def publish(self, event):
        self.queue.append(event)

    def subscribe(self, fn):
        self.subscribers.append(fn)

    def run(self):
        self._running = True
        while self._running:
            if self.queue:
                evt = self.queue.popleft()
                for fn in self.subscribers:
                    fn(evt)
            time.sleep(0.01)

    def stop(self):
        self._running = False

# ---------------------------------------------------------------------
# Tier‑29 Brain (same core as Tier‑26, reused)
# ---------------------------------------------------------------------

class Tier29Brain:
    def __init__(self, input_dim=10, frequency_ghz=3.5, integrity_threshold=0.6, organ_bus=None):
        self.input_dim = input_dim
        self.frequency_ghz = frequency_ghz
        self.integrity_threshold = integrity_threshold

        self.heads = {}
        self.symbolic_bias = {}
        self.memory = deque(maxlen=512)

        self.plasticity = 1.0
        self.energy = 0.0
        self.cycles = 0
        self.model_integrity = 1.0
        self.frozen = False

        self.ewma_alpha = 0.3
        self.ewma_value = None
        self.trend_window = deque(maxlen=64)
        self.var_window = deque(maxlen=64)
        self.turb_window = deque(maxlen=64)
        self.baseline = 0.0
        self.baseline_window = deque(maxlen=512)

        self.last_auto_calibration = time.time()
        self.regime = "STABLE"

        self.agents = {
            "core": 1.0,
            "short": 0.9,
            "long": 0.8,
            "volatility": 0.7,
            "exfil": 0.85,
        }

        self.organ_bus = organ_bus

        self.add_head("risk_core", input_dim, lr=0.01, risk=1.0, organ="core")
        self.add_head("risk_short", input_dim, lr=0.008, risk=0.9, organ="short")
        self.add_head("risk_long", input_dim, lr=0.006, risk=0.8, organ="long")
        self.add_head("risk_volatility", input_dim, lr=0.007, risk=0.7, organ="volatility")
        self.add_head("risk_exfil", input_dim, lr=0.009, risk=0.85, organ="exfil")

    def relu(self, x):
        self.cycles += 1
        return max(0.0, x)

    def sigmoid(self, x):
        self.cycles += 2
        return 1.0 / (1.0 + math.exp(-x))

    def activate(self, tensor, mode="relu"):
        for i in range(len(tensor)):
            for j in range(len(tensor[0])):
                tensor[i][j] = (
                    self.relu(tensor[i][j])
                    if mode == "relu"
                    else self.sigmoid(tensor[i][j])
                )
        return tensor

    def add_head(self, name, input_dim, lr=0.01, risk=1.0, organ=None):
        self.heads[name] = {
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "organ": organ,
            "history": deque(maxlen=32),
            "integrity": 1.0,
        }

    def _symbolic_modulation(self, name, features):
        bias = self.symbolic_bias.get(name, 0.0)
        if len(features) >= 10:
            tension, flow, pressure, swarm_risk, borg_risk, chain_risk, anomaly_intensity, threat_density, log_intensity, net_intensity = features
            if name == "risk_exfil" and chain_risk > 1.0:
                bias += 0.3 * chain_risk
            if name == "risk_volatility" and borg_risk > 1.0:
                bias += 0.2 * borg_risk
            if name == "risk_core" and net_intensity > 1.0:
                bias += 0.1 * net_intensity
        return bias

    def mac(self, a, b):
        self.cycles += 1
        return a * b

    def _predict_head(self, head, x, name):
        y = 0.0
        for i in range(len(x)):
            y += self.mac(x[i], head["w"][i])
        y += head["b"]
        y += self._symbolic_modulation(name, x)
        head["history"].append(y)
        self.memory.append(y)
        return y

    def predict_heads(self, x):
        preds = {}
        for name, head in self.heads.items():
            preds[name] = self._predict_head(head, x, name)
        return preds

    def learn(self, x, targets):
        if self.frozen:
            return {}
        errors = {}
        for name, target in targets.items():
            head = self.heads[name]
            pred = self._predict_head(head, x, name)
            error = target - pred
            weighted_error = (
                error * head["risk"] * self.plasticity * self.model_integrity * head["integrity"]
            )
            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted_error * x[i]
                self.cycles += 1
            head["b"] += head["lr"] * weighted_error
            self.energy += 0.005
            errors[name] = error
        return errors

    def confidence(self, name):
        h = self.heads[name]["history"]
        if len(h) < 2:
            return 0.5
        mean = sum(h) / len(h)
        var = sum((v - mean) ** 2 for v in h) / len(h)
        return max(0.0, min(1.0, 1.0 - var))

    def uncertainty(self, preds):
        if not preds:
            return 1.0
        vals = list(preds.values())
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return min(1.0, max(0.0, var / 10.0))

    def check_integrity(self, external_integrity=1.0):
        self.model_integrity = external_integrity
        self.frozen = self.model_integrity < self.integrity_threshold

    def micro_recovery(self, rate=0.01):
        self.plasticity = min(1.0, self.plasticity + rate)

    def set_symbolic_bias(self, name, value):
        self.symbolic_bias[name] = value

    def save_state(self, path):
        state = {
            "heads": self.heads,
            "plasticity": self.plasticity,
            "energy": self.energy,
            "cycles": self.cycles,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load_state(self, path):
        with open(path, "r") as f:
            state = json.load(f)
        self.heads = state["heads"]
        self.plasticity = state["plasticity"]
        self.energy = state["energy"]
        self.cycles = state["cycles"]

    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9)
        return {
            "cores": 1,
            "cycles": self.cycles,
            "estimated_time_sec": time_sec,
            "energy_units": round(self.energy, 6),
            "plasticity": round(self.plasticity, 3),
            "integrity": round(self.model_integrity, 3),
            "frozen": self.frozen,
            "confidence": {
                k: round(self.confidence(k), 3) for k in self.heads
            },
        }

    def _update_ewma(self, value):
        if self.ewma_value is None:
            self.ewma_value = value
        else:
            self.ewma_value = self.ewma_alpha * value + (1 - self.ewma_alpha) * self.ewma_value
        return self.ewma_value

    def _update_windows(self, value):
        self.trend_window.append(value)
        self.var_window.append(value)
        self.turb_window.append(value)
        self.baseline_window.append(value)

    def _compute_trend(self):
        if len(self.trend_window) < 2:
            return 0.0
        return self.trend_window[-1] - self.trend_window[0]

    def _compute_variance(self):
        if len(self.var_window) < 2:
            return 0.0
        m = sum(self.var_window) / len(self.var_window)
        return sum((v - m) ** 2 for v in self.var_window) / len(self.var_window)

    def _compute_turbulence(self):
        if len(self.turb_window) < 3:
            return 0.0
        diffs = [abs(self.turb_window[i+1] - self.turb_window[i]) for i in range(len(self.turb_window)-1)]
        return sum(diffs) / len(diffs)

    def _update_baseline(self):
        if not self.baseline_window:
            return self.baseline
        self.baseline = sum(self.baseline_window) / len(self.baseline_window)
        return self.baseline

    def _detect_regime(self, value):
        var = self._compute_variance()
        trend = self._compute_trend()
        self._update_baseline()
        dev = abs(value - self.baseline)

        if var < 0.05 and dev < 0.5:
            self.regime = "STABLE"
        elif var > 0.5:
            self.regime = "CHAOTIC"
        elif trend > 0.5:
            self.regime = "RISING"
        elif trend < -0.5:
            self.regime = "COOLING"
        else:
            self.regime = "STABLE"
        return self.regime

    def _engine_votes(self, value):
        ewma = self._update_ewma(value)
        trend = self._compute_trend()
        var = self._compute_variance()
        turb = self._compute_turbulence()
        baseline = self.baseline

        votes = {
            "ewma": ewma,
            "trend": value + trend,
            "variance": var,
            "turbulence": turb,
            "baseline_dev": abs(value - baseline),
            "reinforcement": (sum(self.memory) / len(self.memory)) if self.memory else value,
            "onnx": value,
        }
        return votes

    def _engine_confidences(self, votes):
        var = votes["variance"]
        turb = votes["turbulence"]

        conf = {
            "ewma": 1.0 if var < 0.2 else 0.6,
            "trend": 0.8 if abs(votes["trend"] - votes["ewma"]) < 1.0 else 0.5,
            "variance": 0.5,
            "turbulence": 0.7 if turb > 0.1 else 0.4,
            "baseline_dev": 0.9,
            "reinforcement": 0.7,
            "onnx": 0.6,
        }
        return conf

    def _weighted_vote(self, votes, conf):
        num = 0.0
        den = 0.0
        for k, v in votes.items():
            w = conf.get(k, 0.0)
            num += v * w
            den += w
        return num / den if den > 0 else votes.get("ewma", 0.0)

    def _multi_agent_consensus(self, head_preds):
        num = 0.0
        den = 0.0
        for name, pred in head_preds.items():
            w = self.agents.get(name.replace("risk_", ""), 0.5)
            num += pred * w
            den += w
        return num / den if den > 0 else 0.0

    def _consume_organ_messages(self):
        if self.organ_bus is None:
            return
        msgs = self.organ_bus.drain_for("brain")
        for msg in msgs:
            payload = msg["payload"]
            if payload.get("type") == "integrity_hint":
                delta = payload.get("plasticity_boost", 0.0)
                self.plasticity = max(0.2, min(1.0, self.plasticity + delta))

    def predict_risk(self, features):
        self._consume_organ_messages()
        x = features[:self.input_dim]
        head_preds = self.predict_heads(x)
        consensus = self._multi_agent_consensus(head_preds)

        self._update_windows(consensus)
        regime = self._detect_regime(consensus)
        votes = self._engine_votes(consensus)
        conf = self._engine_confidences(votes)
        best_guess = self._weighted_vote(votes, conf)

        if regime == "STABLE":
            risk = 0.6 * best_guess + 0.4 * consensus
        elif regime == "CHAOTIC":
            risk = 0.5 * best_guess + 0.5 * votes["baseline_dev"]
        elif regime == "RISING":
            risk = best_guess + max(0.0, votes["trend"])
        else:
            risk = 0.6 * best_guess + 0.4 * votes["ewma"]

        risk = max(0.0, risk)
        unc = self.uncertainty(head_preds)
        risk *= (1.0 + 0.2 * unc)

        return risk, regime, votes, conf

    def auto_tune(self, success_score):
        if success_score > 0.8:
            self.ewma_alpha = min(0.9, self.ewma_alpha + 0.01)
            self.plasticity = min(1.0, self.plasticity + 0.01)
        else:
            self.ewma_alpha = max(0.1, self.ewma_alpha - 0.01)
            self.plasticity = max(0.2, self.plasticity - 0.02)

    def auto_calibrate_if_needed(self):
        now = time.time()
        if now - self.last_auto_calibration < AUTO_CALIBRATION_INTERVAL:
            return
        self.last_auto_calibration = now
        self._update_baseline()
        log.info("[BRAIN] Auto‑calibration: baseline=%.3f", self.baseline)

# ---------------------------------------------------------------------
# Awareness Engine
# ---------------------------------------------------------------------

class AwarenessEngine:
    def __init__(self):
        self.state = "SENTINEL"
        self.params = self._params_for(self.state)

    def _params_for(self, state):
        if state == "HYPER_FLOW":
            return {
                "prediction_horizon": 5,
                "aggressiveness": 1.4,
                "ram_appetite": 1.4,
                "thread_expansion": 1.7,
                "cache_factor": 1.3,
            }
        if state == "DEEP_DREAM":
            return {
                "prediction_horizon": 24,
                "aggressiveness": 0.6,
                "ram_appetite": 1.6,
                "thread_expansion": 1.3,
                "cache_factor": 1.6,
            }
        if state == "RECOVERY_FLOW":
            return {
                "prediction_horizon": 3,
                "aggressiveness": 0.5,
                "ram_appetite": 0.8,
                "thread_expansion": 0.7,
                "cache_factor": 0.9,
            }
        if state == "DIAGNOSTIC":
            return {
                "prediction_horizon": 8,
                "aggressiveness": 0.4,
                "ram_appetite": 0.7,
                "thread_expansion": 0.8,
                "cache_factor": 0.8,
            }
        return {
            "prediction_horizon": 10,
            "aggressiveness": 1.0,
            "ram_appetite": 1.0,
            "thread_expansion": 1.0,
            "cache_factor": 1.0,
        }

    def update_state(self, tension, risk, chain_risk):
        if risk > 8 or chain_risk > 1.5:
            new_state = "HYPER_FLOW"
        elif tension > 4 and risk > 3:
            new_state = "SENTINEL"
        elif tension < 0.5 and risk < 0.5:
            new_state = "RECOVERY_FLOW"
        elif risk < 0.5 and chain_risk < 0.3:
            new_state = "DEEP_DREAM"
        else:
            new_state = "DIAGNOSTIC"

        if new_state != self.state:
            log.info(f"[AWARENESS] state change {self.state} → {new_state}")
            self.state = new_state
            self.params = self._params_for(self.state)

    def snapshot(self):
        return {
            "state": self.state,
            "params": self.params,
        }

# ---------------------------------------------------------------------
# Self‑Integrity Organ + functional organ spawning
# ---------------------------------------------------------------------

class SelfIntegrityOrgan:
    def __init__(self, brain: Tier29Brain, organ_bus: OrganMessageBus):
        self.brain = brain
        self.integrity = 1.0
        self.missing_organs = []
        self.organ_bus = organ_bus
        self.spawned_organs = []

    def evaluate(self, core):
        self.missing_organs = []
        score = 1.0
        if not hasattr(core, "telemetry"):
            score -= 0.2
            self.missing_organs.append("telemetry")
        if not hasattr(core, "swarm"):
            score -= 0.2
            self.missing_organs.append("swarm")
        if not hasattr(core, "borg"):
            score -= 0.2
            self.missing_organs.append("borg")

        if core.tension > 12:
            score -= 0.2

        self.integrity = max(0.0, min(1.0, score))
        self.brain.check_integrity(self.integrity)

        if self.integrity < 0.7:
            self._self_heal(core)

        if core.tension > 10 or self.integrity < 0.5:
            self._spawn_organ(core)

        return self.integrity

    def _self_heal(self, core):
        self.brain.plasticity = min(1.0, self.brain.plasticity + 0.05)
        self.brain.baseline_window.clear()
        if self.organ_bus is not None:
            self.organ_bus.send(
                source="integrity",
                target="brain",
                payload={"type": "integrity_hint", "plasticity_boost": 0.02},
            )
        log.warning("[INTEGRITY] Self‑healing triggered: boosting plasticity and resetting baseline window")

    def _spawn_organ(self, core):
        role = random.choice(["stabilizer", "observer", "mirror"])
        name = f"{role}-organ-{len(self.spawned_organs)+1}"
        if name in self.spawned_organs:
            return
        self.spawned_organs.append(name)
        core.spawned_organs.append({"name": name, "role": role})
        if self.organ_bus is not None:
            self.organ_bus.send(
                source="integrity",
                target="brain",
                payload={"type": "organ_spawned", "role": role, "name": name},
            )
        log.warning(f"[INTEGRITY] Organ spawning: created {name} ({role})")

# ---------------------------------------------------------------------
# Synthetic Adversary Engine (Tier‑29)
# ---------------------------------------------------------------------

class SyntheticAdversaryEngine:
    """
    Generates synthetic adversary scenarios and pressure patterns.
    SAFE: does not touch OS, only influences anomaly engine via intensity.
    """
    def __init__(self):
        self.mode = "idle"  # idle / probe / storm / stealth
        self.intensity = 0.0
        self.last_switch = time.time()

    def update_mode(self, tension, chain_risk):
        now = time.time()
        if now - self.last_switch < 30.0:
            return
        self.last_switch = now

        if chain_risk > 1.5:
            self.mode = "storm"
        elif tension > 5:
            self.mode = "probe"
        elif tension < 1:
            self.mode = "stealth"
        else:
            self.mode = "idle"

    def current_intensity(self):
        if self.mode == "idle":
            self.intensity = 0.0
        elif self.mode == "probe":
            self.intensity = 0.4
        elif self.mode == "storm":
            self.intensity = 0.9
        elif self.mode == "stealth":
            self.intensity = 0.2
        return self.intensity

    def narrative(self):
        return {
            "mode": self.mode,
            "intensity": self.intensity,
        }

# ---------------------------------------------------------------------
# Counterfactual Simulator (Tier‑29)
# ---------------------------------------------------------------------

class CounterfactualSimulator:
    """
    Simulates "what‑if" narratives based on current attack graph and risk.
    SAFE: purely narrative, no real actions.
    """
    def __init__(self):
        self.last_scenarios = []

    def simulate(self, attack_graph, chain_risk, borg_risk):
        scenarios = []
        for ent, types in attack_graph.items():
            if not types:
                continue
            if "net_connect" in types and "file_mod" in types:
                scenarios.append({
                    "entity": ent,
                    "scenario": "If left unchecked, this node could become an exfiltration pivot.",
                    "risk": min(1.0, 0.5 + chain_risk * 0.3),
                })
            elif "proc_spawn" in types and "powershell" in types:
                scenarios.append({
                    "entity": ent,
                    "scenario": "This process chain could evolve into a LOLBIN‑style lateral movement.",
                    "risk": min(1.0, 0.4 + borg_risk * 0.2),
                })
        self.last_scenarios = scenarios
        return scenarios

# ---------------------------------------------------------------------
# Defense Planner (Tier‑29)
# ---------------------------------------------------------------------

class DefensePlanner:
    """
    Uses attack graph + counterfactuals + brain risk to propose SAFE defense plans.
    """
    def __init__(self):
        self.last_plan = []

    def plan(self, attack_graph, scenarios, brain_risk):
        plan = []
        if brain_risk > 5:
            plan.append("Increase monitoring frequency and tighten anomaly thresholds.")
        if any("exfiltration" in s["scenario"] for s in scenarios):
            plan.append("Mark suspected exfiltration pivots for deeper inspection (SAFE: log only).")
        if any("LOLBIN" in s["scenario"] for s in scenarios):
            plan.append("Flag LOLBIN‑like chains for correlation in future windows.")
        if not plan:
            plan.append("Maintain current defensive posture; no major changes required.")
        self.last_plan = plan
        return plan

# ---------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------

class PolicyEngine:
    def __init__(self, mode=DEFAULT_MODE, awareness_engine=None, brain=None):
        self.mode = mode
        self.awareness = awareness_engine or AwarenessEngine()
        self.brain = brain or Tier29Brain()

    def decide(self, signals, anomalies, swarm, tension, water_state,
               borg_risk, chain_risk, anomaly_intensity, threat_density,
               log_intensity, net_intensity):
        phase = water_state["phase"]
        flow = water_state["flow"]
        pressure = water_state["pressure"]

        swarm_risk = swarm.get("risk", 0.0)
        features = [
            tension,
            flow,
            pressure,
            swarm_risk,
            borg_risk,
            chain_risk,
            anomaly_intensity,
            threat_density,
            log_intensity,
            net_intensity,
        ]
        brain_risk, regime, votes, conf = self.brain.predict_risk(features)

        self.awareness.update_state(tension, brain_risk, chain_risk)
        aware = self.awareness.snapshot()
        aggr = aware["params"]["aggressiveness"]

        if self.mode == "OBSERVE":
            level = "log"
        elif self.mode == "DEFEND":
            level = "micro" if brain_risk * aggr < 3 else "macro"
        else:
            level = "macro" if brain_risk * aggr > 1 else "micro"

        decision = {
            "risk": brain_risk,
            "level": level,
            "severity": 1.0 if level == "log" else 3.0 if level == "micro" else 6.0,
            "actions": self._plan_actions(level, anomalies),
            "phase": phase,
            "flow": flow,
            "pressure": pressure,
            "borg_risk": borg_risk,
            "chain_risk": chain_risk,
            "awareness": aware,
            "regime": regime,
            "brain_votes": votes,
            "brain_conf": conf,
        }
        return decision

    def _plan_actions(self, level, anomalies):
        if level == "log":
            return [{"type": "log"}]
        if level == "micro":
            return [{"type": "throttle", "targets": [a.get("pid") for a in anomalies[:3] if "pid" in a]}]
        if level == "macro":
            return [{"type": "isolate", "targets": [a.get("pid") for a in anomalies[:5] if "pid" in a]}]

    def execute(self, decision):
        log.info(
            f"[POLICY] phase={decision['phase']} level={decision['level']} "
            f"risk={decision['risk']:.2f} borg_risk={decision['borg_risk']:.2f} "
            f"chain_risk={decision['chain_risk']:.2f} "
            f"awareness={decision['awareness']['state']} "
            f"regime={decision['regime']} "
            f"actions={decision['actions']}"
        )

# ---------------------------------------------------------------------
# Self‑Check Engine
# ---------------------------------------------------------------------

class SelfCheckEngine:
    def __init__(self, core_ref):
        self.core = core_ref
        self.interval = 15.0
        self._running = False

    def _check_once(self):
        try:
            assert isinstance(self.core.policy, PolicyEngine)
            assert callable(self.core.step)
            assert hasattr(self.core, "swarm")
            assert hasattr(self.core, "borg")
        except Exception as e:
            log.error(f"[SELF_CHECK] Integrity check failed: {e}")
        else:
            log.debug("[SELF_CHECK] OK")

    def run(self):
        self._running = True
        while self._running:
            self._check_once()
            time.sleep(self.interval)

    def stop(self):
        self._running = False

# ---------------------------------------------------------------------
# Tier‑29 Core
# ---------------------------------------------------------------------

class Tier29Core:
    def __init__(self, capability_map, policy_engine, telemetry_bus,
                 anomaly_engine, swarm_engine, water_engine,
                 data_tree, borg_collective,
                 rt_queen, chain_engine, event_bus,
                 brain, integrity_organ, organ_bus,
                 adversary_engine, counterfactual_sim, defense_planner):
        self.cap = capability_map
        self.policy = policy_engine
        self.telemetry = telemetry_bus
        self.anomaly = anomaly_engine
        self.swarm = swarm_engine
        self.water = water_engine
        self.tree = data_tree
        self.borg = borg_collective
        self.rt_queen = rt_queen
        self.chain_engine = chain_engine
        self.event_bus = event_bus
        self.brain = brain
        self.integrity_organ = integrity_organ
        self.organ_bus = organ_bus
        self.adversary = adversary_engine
        self.counterfactual = counterfactual_sim
        self.defense_planner = defense_planner

        self.loop_interval = LOOP_BASE_INTERVAL
        self.tension = 0.0
        self.history_tension = deque(maxlen=256)
        self.timeline = deque(maxlen=1024)
        self.spawned_organs = []

    def ingest_sec_event(self, etype, entity, meta=None):
        evt = SecEvent(etype, entity, meta)
        self.chain_engine.add_event(etype, {"entity": entity, "meta": meta or {}})

    def _compute_anomaly_intensity(self, anomalies):
        if not anomalies:
            return 0.0
        return sum(a.get("score", 0.0) for a in anomalies[-10:]) / max(1, len(anomalies[-10:]))

    def _compute_threat_density(self, chains):
        return sum(score for _, score in chains)

    def _compute_log_intensity(self, signals):
        return signals.get("log_error_rate", 0.0) * 5.0 + signals.get("log_warn_rate", 0.0) * 2.0

    def _compute_net_intensity(self, signals):
        return signals.get("packet_rate", 0.0) / 50.0 + float(signals.get("suspicious_flows", 0))

    def step(self):
        signals = self.telemetry.collect_multimodal()
        os_evt = self.telemetry.collect_os_event()
        self.ingest_sec_event(os_evt["etype"], os_evt["entity"])

        anomalies = self.anomaly.evaluate(signals)
        swarm_state = self.swarm.update(signals, anomalies)

        self.borg.update_from_swarm(swarm_state)
        borg_state = self.borg.snapshot()
        borg_risk = self.borg.total_risk()

        water_state = self.water.update(signals, self.tension)

        chains = self.chain_engine.detect()
        chain_risk = sum(score for _, score in chains)
        causal_risk = self.chain_engine.causal_risk()

        anomaly_intensity = self._compute_anomaly_intensity(anomalies)
        threat_density = self._compute_threat_density(chains)
        log_intensity = self._compute_log_intensity(signals)
        net_intensity = self._compute_net_intensity(signals)

        self.adversary.update_mode(self.tension, chain_risk)
        adv_intensity = self.adversary.current_intensity()
        self.anomaly.adversary_mode = adv_intensity > 0.0
        self.anomaly.adversary_intensity = adv_intensity

        decision = self.policy.decide(
            signals=signals,
            anomalies=anomalies,
            swarm=swarm_state,
            tension=self.tension,
            water_state=water_state,
            borg_risk=borg_risk + causal_risk,
            chain_risk=chain_risk,
            anomaly_intensity=anomaly_intensity,
            threat_density=threat_density,
            log_intensity=log_intensity,
            net_intensity=net_intensity,
        )

        self.policy.execute(decision)

        self._update_tension(anomalies, decision, water_state, borg_risk + causal_risk, chain_risk)
        self._adapt_loop_interval(decision)
        self._micro_recovery(water_state)
        self.brain.auto_calibrate_if_needed()

        self.tree.add_leaf(signals)
        self.tree.add_ring(
            tension=self.tension,
            flow=water_state["flow"],
            pressure=water_state["pressure"],
            phase=water_state["phase"],
            anomalies=anomalies,
            swarm=swarm_state,
            borg_risk=borg_risk + causal_risk,
            chain_risk=chain_risk,
            brain_risk=decision["risk"],
        )

        integrity = self.integrity_organ.evaluate(self)
        if integrity < 0.5:
            log.warning("[INTEGRITY] Low integrity=%.2f – safe stance", integrity)

        attack_graph = self.chain_engine.predictive_attack_graph()
        causal_summary = self.chain_engine.causal_summary()

        scenarios = self.counterfactual.simulate(attack_graph, chain_risk, borg_risk + causal_risk)
        defense_plan = self.defense_planner.plan(attack_graph, scenarios, decision["risk"])

        self._append_timeline(signals, anomalies, decision, water_state,
                              borg_risk + causal_risk, chain_risk, integrity,
                              scenarios, defense_plan)

        return {
            "signals": signals,
            "anomalies": anomalies,
            "swarm": swarm_state,
            "decision": decision,
            "tension": self.tension,
            "loop_interval": self.loop_interval,
            "timeline": list(self.timeline),
            "water": water_state,
            "borg": borg_state,
            "tree_stats": self.tree.stats(),
            "integrity": integrity,
            "brain_stats": self.brain.stats(),
            "attack_graph": attack_graph,
            "causal_summary": causal_summary,
            "spawned_organs": list(self.spawned_organs),
            "adversary": self.adversary.narrative(),
            "scenarios": scenarios,
            "defense_plan": defense_plan,
        }

    def _update_tension(self, anomalies, decision, water_state, borg_risk, chain_risk):
        base = sum(a.get("score", 0.0) for a in anomalies[-10:]) if anomalies else 0.0
        sev = decision.get("severity", 0.0)
        pressure = water_state["pressure"]
        self.tension = 0.7 * self.tension + 0.3 * (
            base + sev + pressure * 0.5 + borg_risk * 0.3 + chain_risk * 0.5
        )
        self.history_tension.append(self.tension)

    def _adapt_loop_interval(self, decision):
        aware = decision["awareness"]
        params = aware["params"]
        exp_factor = params["thread_expansion"]
        horizon = params["prediction_horizon"]

        if self.tension > 5:
            target = 0.1 / exp_factor
        elif self.tension > 1:
            target = 0.25 / exp_factor
        else:
            target = 0.75 / exp_factor

        target *= max(0.5, min(2.0, horizon / 10.0))
        self.loop_interval = max(0.05, 0.8 * self.loop_interval + 0.2 * target)

    def _micro_recovery(self, water_state):
        pressure = water_state["pressure"]
        if pressure > 4.0:
            self.brain.plasticity = max(0.2, self.brain.plasticity - 0.02)
        else:
            self.brain.micro_recovery(0.01)

    def _append_timeline(self, signals, anomalies, decision, water_state,
                         borg_risk, chain_risk, integrity,
                         scenarios, defense_plan):
        tick = signals.get("tick")
        risk = decision.get("risk", 0.0)
        level = decision.get("level", "log")
        phase = decision.get("phase", "LIQUID")
        aware_state = decision["awareness"]["state"]
        regime = decision["regime"]
        line = (
            f"[{tick}] phase={phase} aware={aware_state} regime={regime} "
            f"risk={risk:.2f} flow={water_state['flow']:.2f} "
            f"pressure={water_state['pressure']:.2f} borg_risk={borg_risk:.2f} "
            f"chain_risk={chain_risk:.2f} integrity={integrity:.2f} "
            f"level={level} anomalies={len(anomalies)}"
        )
        self.timeline.append(line)
        if scenarios:
            self.timeline.append(f"  scenarios: {len(scenarios)} counterfactuals")
        if defense_plan:
            self.timeline.append(f"  defense_plan: {defense_plan[0]}")

# ---------------------------------------------------------------------
# Cockpit (unchanged, but shows Tier‑29 extras)
# ---------------------------------------------------------------------

if CAPS.has_cockpit:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QTabWidget, QTextEdit, QListWidget
    )
    from PySide6.QtCore import Qt, QTimer

    class SentinelCockpit(QMainWindow):
        def __init__(self, core_state_callable, policy_engine, parent=None):
            super().__init__(parent)
            self.core_state = core_state_callable
            self.policy = policy_engine
            self.setWindowTitle("Sentinel Cockpit – Tier‑29 (ASI‑flavored SAFE+++)")

            self.resize(1800, 1000)
            self._build_ui()
            self._wire_timers()

        def _build_ui(self):
            central = QWidget()
            root_layout = QVBoxLayout(central)

            status_bar = QHBoxLayout()
            self.lbl_mode = QLabel(f"MODE: {self.policy.mode}")
            self.lbl_health = QLabel("HEALTH: Unknown")
            self.lbl_swarm = QLabel("SWARM: 0 nodes")
            self.lbl_phase = QLabel("PHASE: LIQUID")
            self.lbl_tree = QLabel("TREE: 0b/0l/0r")
            self.lbl_awareness = QLabel("AWARE: SENTINEL")
            self.lbl_integrity = QLabel("INT: 1.00")
            self.lbl_organs = QLabel("ORG: base")
            self.lbl_adv = QLabel("ADV: idle")
            for w in (self.lbl_mode, self.lbl_health, self.lbl_swarm,
                      self.lbl_phase, self.lbl_tree, self.lbl_awareness,
                      self.lbl_integrity, self.lbl_organs, self.lbl_adv):
                w.setStyleSheet("font-weight: bold;")
                status_bar.addWidget(w)
            status_bar.addStretch()
            root_layout.addLayout(status_bar)

            tabs = QTabWidget()
            tabs.addTab(self._build_timeline_tab(), "Timeline")
            tabs.addTab(self._build_anomaly_tab(), "Anomalies")
            tabs.addTab(self._build_swarm_tab(), "Swarm / Borg")
            tabs.addTab(self._build_brain_tab(), "Brain")
            tabs.addTab(self._build_threat_tab(), "Threat Matrix")
            tabs.addTab(self._build_causal_tab(), "Causal Graph")
            tabs.addTab(self._build_scenarios_tab(), "Scenarios / Plan")
            tabs.addTab(self._build_controls_tab(), "Controls")
            root_layout.addWidget(tabs)

            self.setCentralWidget(central)

        def _build_timeline_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            self.timeline_view = QTextEdit()
            self.timeline_view.setReadOnly(True)
            self.timeline_view.setPlaceholderText("Event timeline, attack narratives, state transitions...")
            layout.addWidget(self.timeline_view)
            return w

        def _build_anomaly_tab(self):
            w = QWidget()
            layout = QHBoxLayout(w)
            self.lst_anomalies = QListWidget()
            self.lst_anomalies.setMinimumWidth(350)
            self.anomaly_details = QTextEdit()
            self.anomaly_details.setReadOnly(True)
            self.anomaly_details.setPlaceholderText("Selected anomaly details, scores, features, narrative...")
            layout.addWidget(self.lst_anomalies)
            layout.addWidget(self.anomaly_details)
            return w

        def _build_swarm_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            self.swarm_view = QTextEdit()
            self.swarm_view.setReadOnly(True)
            self.swarm_view.setPlaceholderText("Swarm nodes, Borg queens, trust levels, consensus state...")
            layout.addWidget(self.swarm_view)
            return w

        def _build_brain_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            self.brain_view = QTextEdit()
            self.brain_view.setReadOnly(True)
            self.brain_view.setPlaceholderText("Brain stats, regimes, confidence, plasticity, integrity...")
            layout.addWidget(self.brain_view)
            return w

        def _build_threat_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            self.threat_view = QTextEdit()
            self.threat_view.setReadOnly(True)
            self.threat_view.setPlaceholderText("Threat matrix, predictive attack graph, chain narratives...")
            layout.addWidget(self.threat_view)
            return w

        def _build_causal_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            self.causal_view = QTextEdit()
            self.causal_view.setReadOnly(True)
            self.causal_view.setPlaceholderText("Causal graph summary (top causal nodes, degrees)...")
            layout.addWidget(self.causal_view)
            return w

        def _build_scenarios_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            self.scenario_view = QTextEdit()
            self.scenario_view.setReadOnly(True)
            self.scenario_view.setPlaceholderText("Counterfactual scenarios and defense plans...")
            layout.addWidget(self.scenario_view)
            return w

        def _build_controls_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)

            btn_row = QHBoxLayout()
            self.btn_mode_observe = QPushButton("Observe")
            self.btn_mode_defend = QPushButton("Defend")
            self.btn_mode_aggressive = QPushButton("Aggressive")
            for b in (self.btn_mode_observe, self.btn_mode_defend, self.btn_mode_aggressive):
                btn_row.addWidget(b)
            btn_row.addStretch()
            layout.addLayout(btn_row)

            self.control_log = QTextEdit()
            self.control_log.setReadOnly(True)
            self.control_log.setPlaceholderText("Control actions, overrides, and system decisions...")
            layout.addWidget(self.control_log)

            self.btn_mode_observe.clicked.connect(lambda: self._set_mode("OBSERVE"))
            self.btn_mode_defend.clicked.connect(lambda: self._set_mode("DEFEND"))
            self.btn_mode_aggressive.clicked.connect(lambda: self._set_mode("AGGRESSIVE"))

            return w

        def _wire_timers(self):
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._refresh_from_core)
            self.timer.start(500)

        def _refresh_from_core(self):
            state = self.core_state()
            tension = state.get("tension", 0.0)
            decision = state.get("decision", {})
            risk = decision.get("risk", 0.0)
            phase = decision.get("phase", "LIQUID")
            borg_risk = decision.get("borg_risk", 0.0)
            chain_risk = decision.get("chain_risk", 0.0)
            aware = decision.get("awareness", {"state": "SENTINEL"})
            aware_state = aware.get("state", "SENTINEL")
            water = state.get("water", {})
            flow = water.get("flow", 0.0)
            pressure = water.get("pressure", 0.0)
            swarm = state.get("swarm", {})
            nodes = swarm.get("nodes", [])
            borg = state.get("borg", {})
            queens = borg.get("queens", {})
            tree_stats = state.get("tree_stats", {"branches": 0, "leaves": 0, "rings": 0})
            integrity = state.get("integrity", 1.0)
            brain_stats = state.get("brain_stats", {})
            attack_graph = state.get("attack_graph", {})
            causal_summary = state.get("causal_summary", {})
            spawned_organs = state.get("spawned_organs", [])
            adversary = state.get("adversary", {"mode": "idle", "intensity": 0.0})
            scenarios = state.get("scenarios", [])
            defense_plan = state.get("defense_plan", [])

            self.lbl_health.setText(
                f"HEALTH: Tension={tension:.2f} Risk={risk:.2f} Borg={borg_risk:.2f} Chain={chain_risk:.2f}"
            )
            self.lbl_swarm.setText(f"SWARM: {len(nodes)} nodes / {len(queens)} queens")
            self.lbl_phase.setText(f"PHASE: {phase}")
            self.lbl_tree.setText(
                f"TREE: {tree_stats['branches']}b/{tree_stats['leaves']}l/{tree_stats['rings']}r"
            )
            self.lbl_awareness.setText(f"AWARE: {aware_state}")
            self.lbl_integrity.setText(f"INT: {integrity:.2f}")
            self.lbl_organs.setText(f"ORG: base+{len(spawned_organs)}")
            self.lbl_adv.setText(f"ADV: {adversary.get('mode','idle')} ({adversary.get('intensity',0.0):.2f})")

            if tension < 1:
                color = "#4caf50"
            elif tension < 4:
                color = "#ff9800"
            else:
                color = "#f44336"
            self.lbl_health.setStyleSheet(f"font-weight: bold; color: {color};")

            timeline = state.get("timeline", [])
            self.timeline_view.setPlainText("\n".join(timeline[-500:]))

            anomalies = state.get("anomalies", [])
            self.lst_anomalies.clear()
            for a in anomalies[-100:]:
                score = a.get("score", 0.0)
                label = a.get("label", "unknown")
                self.lst_anomalies.addItem(f"[{score:.3f}] {label}")
            if anomalies:
                last = anomalies[-1]
                self.anomaly_details.setPlainText(json.dumps(last, indent=2))
            else:
                self.anomaly_details.setPlainText("No anomalies.")

            swarm_lines = []
            for n in nodes:
                swarm_lines.append(f"{n['id']} | trust={n['trust']:.2f} | state={n['state']} | role={n.get('role','?')}")
            swarm_lines.append(f"\nflow={flow:.2f} pressure={pressure:.2f}")
            swarm_lines.append(f"borg_risk={borg_risk:.2f} chain_risk={chain_risk:.2f}")
            for qid, q in queens.items():
                swarm_lines.append(f"[QUEEN] {qid} nodes={len(q['nodes'])} trust={q['trust']:.2f}")
            self.swarm_view.setPlainText("\n".join(swarm_lines))

            self.brain_view.setPlainText(json.dumps(brain_stats, indent=2))

            threat_lines = ["Threat Matrix / Predictive Attack Graph:"]
            for ent, types in attack_graph.items():
                threat_lines.append(f"{ent}: {', '.join(types)}")
            self.threat_view.setPlainText("\n".join(threat_lines))

            causal_lines = ["Causal Graph Summary (top out‑degree nodes):"]
            for node, deg in causal_summary.items():
                causal_lines.append(f"{node} -> {deg:.2f}")
            self.causal_view.setPlainText("\n".join(causal_lines))

            scenario_lines = ["Counterfactual Scenarios:"]
            for s in scenarios:
                scenario_lines.append(f"- {s['entity']}: {s['scenario']} (risk={s['risk']:.2f})")
            scenario_lines.append("\nDefense Plan:")
            for p in defense_plan:
                scenario_lines.append(f"- {p}")
            self.scenario_view.setPlainText("\n".join(scenario_lines))

        def _set_mode(self, mode: str):
            self.policy.mode = mode
            self.lbl_mode.setText(f"MODE: {mode}")
            self.control_log.append(f"Mode set to: {mode}")

    def run_cockpit(core_state_callable, policy_engine):
        app = QApplication(sys.argv)
        win = SentinelCockpit(core_state_callable, policy_engine)
        win.show()
        sys.exit(app.exec())

else:
    def run_cockpit(core_state_callable, policy_engine):
        log.warning("PySide6 not available – cockpit disabled.")
        while True:
            time.sleep(10)

# ---------------------------------------------------------------------
# Wiring – Tier‑29 fusion
# ---------------------------------------------------------------------

organ_bus = OrganMessageBus()

telemetry_bus = TelemetryBus()
anomaly_engine = Tier29AnomalyEngine()
swarm_engine = SwarmEngine()
water_engine = IntelligentWaterEngine()
data_tree = DataTree()
borg_collective = BorgCollective()
rt_queen = RealTimeQueen()
chain_engine = AttackChainEngine()
event_bus = EventBus()
brain = Tier29Brain(organ_bus=organ_bus)
awareness_engine = AwarenessEngine()
policy_engine = PolicyEngine(mode=DEFAULT_MODE, awareness_engine=awareness_engine, brain=brain)
integrity_organ = SelfIntegrityOrgan(brain, organ_bus)
adversary_engine = SyntheticAdversaryEngine()
counterfactual_sim = CounterfactualSimulator()
defense_planner = DefensePlanner()

core = Tier29Core(
    capability_map=CAPS,
    policy_engine=policy_engine,
    telemetry_bus=telemetry_bus,
    anomaly_engine=anomaly_engine,
    swarm_engine=swarm_engine,
    water_engine=water_engine,
    data_tree=data_tree,
    borg_collective=borg_collective,
    rt_queen=rt_queen,
    chain_engine=chain_engine,
    event_bus=event_bus,
    brain=brain,
    integrity_organ=integrity_organ,
    organ_bus=organ_bus,
    adversary_engine=adversary_engine,
    counterfactual_sim=counterfactual_sim,
    defense_planner=defense_planner,
)

_state_snapshot = {
    "tension": 0.0,
    "anomalies": [],
    "swarm": {"nodes": []},
    "decision": {},
    "timeline": [],
    "water": {"flow": 0.0, "pressure": 0.0, "phase": "LIQUID"},
    "borg": {"queens": {}},
    "tree_stats": {"branches": 0, "leaves": 0, "rings": 0},
    "integrity": 1.0,
    "brain_stats": {},
    "attack_graph": {},
    "causal_summary": {},
    "spawned_organs": [],
    "adversary": {"mode": "idle", "intensity": 0.0},
    "scenarios": [],
    "defense_plan": [],
}

def core_thread():
    global _state_snapshot
    while True:
        snap = core.step()
        _state_snapshot = {
            "tension": snap["tension"],
            "anomalies": snap["anomalies"],
            "swarm": snap["swarm"],
            "decision": snap["decision"],
            "timeline": snap["timeline"],
            "water": snap["water"],
            "borg": snap["borg"],
            "tree_stats": snap["tree_stats"],
            "integrity": snap["integrity"],
            "brain_stats": snap["brain_stats"],
            "attack_graph": snap["attack_graph"],
            "causal_summary": snap["causal_summary"],
            "spawned_organs": snap["spawned_organs"],
            "adversary": snap["adversary"],
            "scenarios": snap["scenarios"],
            "defense_plan": snap["defense_plan"],
        }
        time.sleep(core.loop_interval)

def get_state():
    return dict(_state_snapshot)

def self_check_thread(core_ref):
    checker = SelfCheckEngine(core_ref)
    checker.run()

def event_bus_thread():
    event_bus.run()

def main():
    t_core = threading.Thread(target=core_thread, daemon=True)
    t_core.start()

    t_self = threading.Thread(target=self_check_thread, args=(core,), daemon=True)
    t_self.start()

    t_evt = threading.Thread(target=event_bus_thread, daemon=True)
    t_evt.start()

    run_cockpit(get_state, policy_engine)

if __name__ == "__main__":
    main()
