import importlib
import subprocess
import sys
import threading
import time
import json
import socket
import struct
import random
import os
import platform
import ctypes
import math
from collections import deque

# ============================================================
# 🔺 ELEVATION HELPERS (WINDOWS)
# ============================================================

def is_admin():
    try:
        return os.getuid() == 0
    except AttributeError:
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except Exception:
            return False

if platform.system() == "Windows" and not is_admin():
    script_path = os.path.abspath(sys.argv[0])
    try:
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, f'"{script_path}"', None, 1
        )
        sys.exit()
    except Exception:
        pass

def ensure_admin_v1():
    try:
        is_admin_flag = ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        is_admin_flag = False
    if not is_admin_flag:
        params = " ".join([f'"{arg}"' for arg in sys.argv])
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, params, None, 1
        )
        sys.exit()

def ensure_admin_v2():
    try:
        is_admin_flag = ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        is_admin_flag = False

    if not is_admin_flag:
        print("[Codex Sentinel] Elevation required. Relaunching as administrator...")
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, " ".join(sys.argv), None, 1
        )
        sys.exit()

def ensure_admin_auto():
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
        sys.exit()

if platform.system() == "Windows":
    ensure_admin_auto()

# ============================================================
# CONFIG
# ============================================================

POLICY_BUS_HOST = "127.0.0.1"
POLICY_BUS_PORT = 5555

SWARM_ENABLED = True
SWARM_GROUP = "239.10.10.10"
SWARM_PORT = 5556
SWARM_NODE_ID = f"node-{random.randint(1000, 9999)}"

CRYPTO_KEY_FILE = "swarm.key"

# ============================================================
# AUTOLOADER
# ============================================================

REQUIRED_LIBS = [
    "numpy",
    "psutil",
    "torch",
    "pynvml",
    "tk",
    "cryptography",
]

def autoload_libraries():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            if lib == "tk":
                continue
            print(f"[AUTOLOADER] Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

autoload_libraries()

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import pynvml
import tkinter as tk
from tkinter import ttk
from cryptography.fernet import Fernet

# ============================================================
# CRYPTO ORGAN
# ============================================================

class CryptoOrgan:
    def __init__(self, key_path=CRYPTO_KEY_FILE):
        self.key_path = key_path
        self.fernet = self._load_or_create_key()

    def _load_or_create_key(self):
        key = None
        if os.path.exists(self.key_path):
            try:
                with open(self.key_path, "rb") as f:
                    key = f.read().strip()
            except Exception:
                key = None
        if not key:
            key = Fernet.generate_key()
            try:
                with open(self.key_path, "wb") as f:
                    f.write(key)
            except Exception as e:
                print(f"[CRYPTO] Failed to persist key: {e}")
        return Fernet(key)

    def encrypt_dict(self, payload: dict) -> bytes:
        data = json.dumps(payload).encode("utf-8")
        return self.fernet.encrypt(data)

    def decrypt_bytes(self, blob: bytes):
        try:
            data = self.fernet.decrypt(blob)
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

# ============================================================
# USB / PCIe NPU DISCOVERY ORGAN
# ============================================================

class NPUDiscoveryOrgan:
    def __init__(self):
        self.devices = []
        self.scan()

    def scan(self):
        self.devices = []
        try:
            if sys.platform.startswith("win"):
                cmd = ["wmic", "path", "win32_pnpentity", "get", "Name"]
                out = subprocess.check_output(
                    cmd,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="ignore"
                )
                for line in out.splitlines():
                    name = line.strip()
                    if not name:
                        continue
                    lowered = name.lower()
                    if any(tag in lowered for tag in ["npu", "neural", "ai accelerator", "vpu"]):
                        self.devices.append(name)
            else:
                pass
        except Exception as e:
            print(f"[NPU DISCOVERY] Error during scan: {e}")

    def summary(self):
        if not self.devices:
            return "No explicit NPU devices detected (best-effort scan)."
        return " | ".join(self.devices)

    def count(self):
        return max(1, len(self.devices))

# ============================================================
# SOCKET POLICY BUS (LOCAL SUBSCRIBERS)
# ============================================================

class SocketPolicyBus:
    def __init__(self, host=POLICY_BUS_HOST, port=POLICY_BUS_PORT, crypto: CryptoOrgan | None = None):
        self.host = host
        self.port = port
        self.crypto = crypto
        self.clients = []
        self.lock = threading.Lock()
        self._stop = False
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

    def _server_loop(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                s.listen(5)
                print(f"[POLICY BUS] Listening on {self.host}:{self.port}")
                while not self._stop:
                    s.settimeout(1.0)
                    try:
                        conn, addr = s.accept()
                    except socket.timeout:
                        continue
                    print(f"[POLICY BUS] Client connected from {addr}")
                    with self.lock:
                        self.clients.append(conn)
        except Exception as e:
            print(f"[POLICY BUS] Server error: {e}")

    def broadcast(self, state: dict):
        if self.crypto:
            blob = self.crypto.encrypt_dict(state)
            data = blob + b"\n"
        else:
            data = (json.dumps(state) + "\n").encode("utf-8")

        dead = []
        with self.lock:
            for c in self.clients:
                try:
                    c.sendall(data)
                except Exception:
                    dead.append(c)
            for c in dead:
                try:
                    c.close()
                except Exception:
                    pass
                self.clients.remove(c)

    def stop(self):
        self._stop = True
        with self.lock:
            for c in self.clients:
                try:
                    c.close()
                except Exception:
                    pass
            self.clients.clear()

# ============================================================
# SWARM SYNC ORGAN (MULTI-NODE MESH VIA UDP MULTICAST)
# ============================================================

class SwarmSyncOrgan:
    def __init__(self, group=SWARM_GROUP, port=SWARM_PORT, node_id=SWARM_NODE_ID, crypto: CryptoOrgan | None = None):
        self.group = group
        self.port = port
        self.node_id = node_id
        self.crypto = crypto
        self._stop = False
        self.peers = {}
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

    def _recv_loop(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", self.port))
            except OSError:
                sock.bind((self.group, self.port))

            mreq = struct.pack("4sl", socket.inet_aton(self.group), socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

            while not self._stop:
                sock.settimeout(1.0)
                try:
                    data, addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue

                if self.crypto:
                    msg = self.crypto.decrypt_bytes(data)
                    if not msg:
                        continue
                else:
                    try:
                        msg = json.loads(data.decode("utf-8"))
                    except Exception:
                        continue

                nid = msg.get("node_id")
                if not nid or nid == self.node_id:
                    continue
                self.peers[nid] = {
                    "last_state": msg,
                    "last_seen": time.time(),
                    "addr": addr,
                }
        except Exception as e:
            print(f"[SWARM] Receiver error: {e}")

    def broadcast(self, state: dict):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            ttl = struct.pack("b", 1)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
            msg = dict(state)
            msg["node_id"] = self.node_id

            if self.crypto:
                blob = self.crypto.encrypt_dict(msg)
                payload = blob
            else:
                payload = json.dumps(msg).encode("utf-8")

            sock.sendto(payload, (self.group, self.port))
            sock.close()
        except Exception as e:
            print(f"[SWARM] Broadcast error: {e}")

    def snapshot_peers(self):
        now = time.time()
        return {
            nid: info for nid, info in self.peers.items()
            if now - info["last_seen"] < 10.0
        }

    def stop(self):
        self._stop = True

# ============================================================
# GPU ORGAN
# ============================================================

class GPUOrgan:
    def __init__(self, index=0):
        pynvml.nvmlInit()
        self.device_index = index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def get_name(self):
        name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(name, bytes):
            return name.decode()
        return name

    def get_utilization(self):
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return util.gpu, util.memory

    def get_memory_fraction(self):
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        if mem.total == 0:
            return 0.0
        return mem.used / mem.total

    def get_temperature(self):
        try:
            return pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            return 0

    def get_power_usage_and_limit(self):
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0
            return power, limit
        except pynvml.NVMLError:
            return 0.0, 0.0

    def apply_power_policy(self, power_scale: float):
        try:
            min_limit_mw, max_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)
            min_limit = min_limit_mw / 1000.0
            max_limit = max_limit_mw / 1000.0
            target = min_limit + (max_limit - min_limit) * float(np.clip(power_scale, 0.0, 1.0))
            print(f"[GPU POLICY] Target power limit: {target:.1f} W (scale={power_scale:.2f})")
            pynvml.nvmlDeviceSetPowerManagementLimit(self.handle, int(target * 1000))
        except pynvml.NVMLError as e:
            print(f"[GPU POLICY] Could not set power limit: {str(e)}")

# ============================================================
# GPU FAN ORGAN
# ============================================================

class GPUFanOrgan:
    def __init__(self, gpu_index=0, min_fan=20, max_fan=100, target_temp=70.0, max_temp=85.0):
        self.gpu_index = gpu_index
        self.min_fan = min_fan
        self.max_fan = max_fan
        self.target_temp = target_temp
        self.max_temp = max_temp

    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def compute_fan_target(self, temp, mode, stance):
        base = 0
        if temp <= self.target_temp:
            base = self.min_fan
        elif temp >= self.max_temp:
            base = self.max_fan
        else:
            frac = (temp - self.target_temp) / max(self.max_temp - self.target_temp, 1.0)
            base = self.min_fan + frac * (self.max_fan - self.min_fan)

        stance_bias = {
            "Conservative": +10,
            "Balanced": 0,
            "Beast": -5,
        }.get(stance, 0)

        mode_bias = {
            "COAST": -5,
            "NORMAL": 0,
            "AGGRESSIVE": +5,
            "BEAST": +10,
        }.get(mode, 0)

        fan = base + stance_bias + mode_bias
        return int(self._clamp(fan, self.min_fan, self.max_fan))

    def set_fan_speed(self, percent):
        percent = self._clamp(percent, self.min_fan, self.max_fan)
        cmd = ["nvidia-smi", "-i", str(self.gpu_index), "--fan", f"{percent}"]
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            print(f"[GPU FAN] Set fan to {percent}%")
        except Exception as e:
            print(f"[GPU FAN] Failed to set fan speed: {e}")

# ============================================================
# REPLICA NPU (MERGED VARIANTS)
# ============================================================

class ReplicaNPU:
    """
    Fully combined, predictive, hardware-style Neural Processing Unit (NPU)
    """

    def __init__(
        self,
        cores=8,
        frequency_ghz=1.2,
        memory_size=32,
        plasticity_decay=0.0005,
        integrity_threshold=0.4,
    ):
        self.cores = cores
        self.frequency_ghz = frequency_ghz

        self.cycles = 0
        self.energy = 0.0

        self.memory = deque(maxlen=memory_size)

        self.plasticity = 1.0
        self.plasticity_decay = plasticity_decay

        self.integrity_threshold = integrity_threshold
        self.model_integrity = 1.0
        self.frozen = False

        self.heads = {}
        self.symbolic_bias = {}
        self.instruction_queue = deque()

    # -------------------------
    # Instruction Scheduler
    # -------------------------
    def schedule(self, fn, *args):
        self.instruction_queue.append((fn, args))

    def tick(self, budget=64):
        executed = 0
        while self.instruction_queue and executed < budget:
            fn, args = self.instruction_queue.popleft()
            fn(*args)
            executed += 1

        self.plasticity = max(0.1, self.plasticity - self.plasticity_decay)

    # -------------------------
    # Low-level NPU operations
    # -------------------------
    def mac(self, a, b):
        self.cycles += 1
        self.energy += 0.001
        return a * b

    def vector_mac(self, v1, v2):
        assert len(v1) == len(v2)
        chunk = math.ceil(len(v1) / self.cores)
        acc = 0.0

        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                partial += self.mac(v1[j], v2[j])
            acc += partial

        return acc

    # -------------------------
    # Tensor operations
    # -------------------------
    def matmul(self, A, B):
        result = [[0.0] * len(B[0]) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                col = [B[k][j] for k in range(len(B))]
                result[i][j] = self.vector_mac(A[i], col)
        return result

    # -------------------------
    # Activations
    # -------------------------
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

    # -------------------------
    # Predictive Heads
    # -------------------------
    def add_head(self, name, input_dim, lr=0.01, risk=1.0, organ=None):
        self.heads[name] = {
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "organ": organ,
            "history": deque(maxlen=32),
        }

    def _symbolic_modulation(self, name):
        return self.symbolic_bias.get(name, 0.0)

    def _predict_head(self, head, x, name):
        y = 0.0
        for i in range(len(x)):
            y += self.mac(x[i], head["w"][i])
        y += head["b"]
        y += self._symbolic_modulation(name)

        head["history"].append(y)
        self.memory.append(y)
        return y

    def predict(self, x):
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
                error * head["risk"] * self.plasticity * self.model_integrity
            )

            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted_error * x[i]
                self.cycles += 1

            head["b"] += head["lr"] * weighted_error
            self.energy += 0.005
            errors[name] = error

        return errors

    # -------------------------
    # Confidence & Integrity
    # -------------------------
    def confidence(self, name):
        h = self.heads[name]["history"]
        if len(h) < 2:
            return 0.5
        mean = sum(h) / len(h)
        var = sum((v - mean) ** 2 for v in h) / len(h)
        return max(0.0, min(1.0, 1.0 - var))

    def check_integrity(self, external_integrity=1.0):
        self.model_integrity = external_integrity
        self.frozen = self.model_integrity < self.integrity_threshold

    # -------------------------
    # Plasticity Recovery
    # -------------------------
    def micro_recovery(self, rate=0.01):
        self.plasticity = min(1.0, self.plasticity + rate)

    # -------------------------
    # Symbolic Interface
    # -------------------------
    def set_symbolic_bias(self, name, value):
        self.symbolic_bias[name] = value

    # -------------------------
    # Serialization
    # -------------------------
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

    # -------------------------
    # Stats
    # -------------------------
    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9)
        return {
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
        }

# ============================================================
# BRAIN ORGAN (WRAPS ReplicaNPU)
# ============================================================

class BrainOrgan:
    def __init__(self, history_len=64):
        self.npu = ReplicaNPU(cores=16, frequency_ghz=1.5, memory_size=history_len)
        self.input_dim = 8  # gpu_util, temp, power_frac, mode_idx, swarm_load, integrity, plasticity, npu_conf
        self.npu.add_head("short", self.input_dim, lr=0.05, risk=1.5, organ="short_horizon")
        self.npu.add_head("medium", self.input_dim, lr=0.03, risk=1.0, organ="mid_horizon")
        self.npu.add_head("long", self.input_dim, lr=0.02, risk=0.7, organ="long_horizon")
        self.npu.add_head("baseline", self.input_dim, lr=0.01, risk=0.5, organ="baseline")
        self.meta_state = "Stable"
        self.stance = "Balanced"
        self.last_predictions = {}
        self.last_reasoning = []
        self.last_heatmap = {}

    def _mode_index(self, mode: str) -> float:
        mapping = {"COAST": 0.0, "NORMAL": 0.33, "AGGRESSIVE": 0.66, "BEAST": 1.0}
        return mapping.get(mode, 0.5)

    def _swarm_load(self, peers: dict) -> float:
        if not peers:
            return 0.0
        utils = []
        for _, info in peers.items():
            st = info.get("last_state", {})
            gu = st.get("gpu_util")
            if gu is not None:
                utils.append(gu / 100.0)
        if not utils:
            return 0.0
        return float(np.clip(np.mean(utils), 0.0, 1.0))

    def _build_input(self, gpu_util, temp, power, power_limit, mode, swarm_peers, npu_raid):
        power_frac = (power / power_limit) if power_limit > 0 else 0.0
        mode_idx = self._mode_index(mode)
        swarm_load = self._swarm_load(swarm_peers)
        integrity = npu_raid.integrity
        plasticity = npu_raid.plasticity
        npu_conf = npu_raid.confidence
        x = [
            gpu_util / 100.0,
            min(1.0, temp / 100.0),
            power_frac,
            mode_idx,
            swarm_load,
            integrity,
            plasticity,
            npu_conf,
        ]
        return x

    def _derive_meta_state_and_stance(self, preds):
        short = preds.get("short", 0.5)
        medium = preds.get("medium", 0.5)
        long = preds.get("long", 0.5)
        baseline = preds.get("baseline", 0.5)

        vol = abs(short - long)
        avg = (short + medium + long) / 3.0

        if vol < 0.1 and avg < 0.4:
            meta = "Stable"
        elif vol < 0.2 and avg < 0.7:
            meta = "Tense"
        else:
            meta = "Volatile"

        if avg < 0.3:
            stance = "Conservative"
        elif avg < 0.7:
            stance = "Balanced"
        else:
            stance = "Beast"

        return meta, stance

    def update(self, gpu_util, temp, power, power_limit, mode, swarm_peers, npu_raid):
        x = self._build_input(gpu_util, temp, power, power_limit, mode, swarm_peers, npu_raid)
        preds = self.npu.predict(x)

        target_short = gpu_util / 100.0
        target_medium = (gpu_util / 100.0 + temp / 100.0) / 2.0
        target_long = (temp / 100.0 + power / max(power_limit, 1.0)) / 2.0 if power_limit > 0 else temp / 100.0
        target_baseline = 0.5

        targets = {
            "short": target_short,
            "medium": target_medium,
            "long": target_long,
            "baseline": target_baseline,
        }
        errs = self.npu.learn(x, targets)
        self.npu.micro_recovery()
        self.npu.check_integrity(external_integrity=npu_raid.integrity)

        meta, stance = self._derive_meta_state_and_stance(preds)
        self.meta_state = meta
        self.stance = stance
        self.last_predictions = preds

        self.last_reasoning = [
            f"short={preds.get('short', 0.0):.3f}, medium={preds.get('medium', 0.0):.3f}, long={preds.get('long', 0.0):.3f}",
            f"baseline={preds.get('baseline', 0.0):.3f}, meta={meta}, stance={stance}",
            f"errors: { {k: round(v, 3) for k, v in errs.items()} }",
        ]

        self.last_heatmap = {
            "volatility": abs(preds.get("short", 0.5) - preds.get("long", 0.5)),
            "avg_risk": (preds.get("short", 0.5) + preds.get("medium", 0.5) + preds.get("long", 0.5)) / 3.0,
            "best_guess_contributors": {
                "short": preds.get("short", 0.5),
                "medium": preds.get("medium", 0.5),
                "long": preds.get("long", 0.5),
                "baseline": preds.get("baseline", 0.5),
                "weights": {
                    "short": 0.4,
                    "medium": 0.3,
                    "long": 0.2,
                    "baseline": 0.1,
                },
            },
        }

        avg_risk = self.last_heatmap["avg_risk"]
        bias = (avg_risk - 0.5) * 0.4
        self.npu.set_symbolic_bias("short", bias)
        self.npu.set_symbolic_bias("medium", bias * 0.7)
        self.npu.set_symbolic_bias("long", bias * 0.5)
        self.npu.set_symbolic_bias("baseline", bias * 0.3)

        return preds

    def stance_bias_for_power(self):
        mapping = {
            "Conservative": 0.8,
            "Balanced": 1.0,
            "Beast": 1.2,
        }
        return mapping.get(self.stance, 1.0)

# ============================================================
# NPU ORGAN + RAID (TORCH-BASED CONTROLLER)
# ============================================================

class NPUControllerNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class NPUOrgan:
    def __init__(self, target_util=0.8, max_temp=80.0):
        self.model = NPUControllerNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.target_util = target_util
        self.max_temp = max_temp
        self.confidence_val = 0.5
        self.integrity = 1.0
        self.plasticity = 0.5

    def _normalize_inputs(self, gpu_util, mem_util, mem_frac, temp, power, power_limit):
        gpu_util_n = gpu_util / 100.0
        mem_util_n = mem_util / 100.0
        mem_frac_n = mem_frac
        temp_n = min(1.0, temp / max(self.max_temp, 1.0)) if self.max_temp > 0 else 0.0
        power_frac = (power / power_limit) if power_limit > 0 else 0.0
        return np.array([gpu_util_n, mem_util_n, mem_frac_n, temp_n, power_frac], dtype=np.float32)

    def infer(self, gpu_util, mem_util, mem_frac, temp, power, power_limit):
        x = self._normalize_inputs(gpu_util, mem_util, mem_frac, temp, power, power_limit)
        x_t = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x_t)
        perf_raw, power_raw = out[0]
        perf = torch.sigmoid(perf_raw).item()
        power_scale = torch.sigmoid(power_raw).item()

        self.confidence_val = 0.5 + 0.5 * (1.0 - abs(perf - self.target_util))
        self.plasticity = float(np.clip(self.plasticity + np.random.uniform(-0.01, 0.01), 0.0, 1.0))

        return perf, power_scale

    def learn_from_outcome(self, gpu_util, temp):
        gpu_util_n = gpu_util / 100.0
        util_error = (gpu_util_n - self.target_util)
        temp_penalty = max(0.0, (temp - self.max_temp) / max(self.max_temp, 1.0))

        reward = - (util_error ** 2) - temp_penalty

        target_perf = self.target_util
        target_power_scale = 1.0 if temp < self.max_temp else 0.5

        x = torch.tensor([[gpu_util_n, gpu_util_n, gpu_util_n,
                           min(1.0, temp / max(self.max_temp, 1.0)), 0.0]], dtype=torch.float32)
        out = self.model(x)
        perf_raw, power_raw = out[0]
        perf = torch.sigmoid(perf_raw)
        power_scale = torch.sigmoid(power_raw)

        loss = (perf - target_perf) ** 2 + (power_scale - target_power_scale) ** 2 - 0.1 * reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.integrity = float(np.clip(self.integrity - 0.01 * float(loss.item()), 0.0, 1.0))


class NPURaidEnsemble:
    def __init__(self, count: int, target_util=0.8, max_temp=80.0):
        self.npus = [NPUOrgan(target_util=target_util, max_temp=max_temp) for _ in range(count)]

    @property
    def confidence(self):
        return float(np.mean([n.confidence_val for n in self.npus]))

    @property
    def integrity(self):
        return float(np.mean([n.integrity for n in self.npus]))

    @property
    def plasticity(self):
        return float(np.mean([n.plasticity for n in self.npus]))

    def infer(self, gpu_util, mem_util, mem_frac, temp, power, power_limit):
        perfs = []
        scales = []
        for n in self.npus:
            p, s = n.infer(gpu_util, mem_util, mem_frac, temp, power, power_limit)
            perfs.append(p)
            scales.append(s)
        return float(np.mean(perfs)), float(np.mean(scales))

    def learn_from_outcome(self, gpu_util, temp):
        for n in self.npus:
            n.learn_from_outcome(gpu_util, temp)

# ============================================================
# SCHEDULER
# ============================================================

class Scheduler:
    def __init__(self, gpu: GPUOrgan, fan: GPUFanOrgan, npu_raid: NPURaidEnsemble,
                 brain: BrainOrgan, policy_bus: SocketPolicyBus, swarm: SwarmSyncOrgan | None):
        self.gpu = gpu
        self.fan = fan
        self.npu_raid = npu_raid
        self.brain = brain
        self.policy_bus = policy_bus
        self.swarm = swarm
        self.mode = "NORMAL"
        self.last_perf = 0.0
        self.last_power_scale = 1.0

    def step(self):
        gpu_util, mem_util = self.gpu.get_utilization()
        mem_frac = self.gpu.get_memory_fraction()
        temp = self.gpu.get_temperature()
        power, power_limit = self.gpu.get_power_usage_and_limit()

        swarm_peers = self.swarm.snapshot_peers() if self.swarm is not None else {}

        perf, power_scale = self.npu_raid.infer(
            gpu_util, mem_util, mem_frac, temp, power, power_limit
        )

        brain_preds = self.brain.update(
            gpu_util=gpu_util,
            temp=temp,
            power=power,
            power_limit=power_limit,
            mode=self.mode,
            swarm_peers=swarm_peers,
            npu_raid=self.npu_raid,
        )

        stance_bias = self.brain.stance_bias_for_power()
        power_scale = float(np.clip(power_scale * stance_bias, 0.2, 1.2))

        if perf > 0.85:
            self.mode = "BEAST"
        elif perf > 0.6:
            self.mode = "AGGRESSIVE"
        elif perf > 0.3:
            self.mode = "NORMAL"
        else:
            self.mode = "COAST"

        self.gpu.apply_power_policy(power_scale)
        self.npu_raid.learn_from_outcome(gpu_util, temp)

        fan_target = self.fan.compute_fan_target(temp, self.mode, self.brain.stance)
        self.fan.set_fan_speed(fan_target)

        self.last_perf = perf
        self.last_power_scale = power_scale

        state = {
            "mode": self.mode,
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "mem_frac": mem_frac,
            "temp": temp,
            "power": power,
            "power_limit": power_limit,
            "perf": perf,
            "power_scale": power_scale,
            "fan_target": fan_target,
            "npu_confidence": self.npu_raid.confidence,
            "npu_integrity": self.npu_raid.integrity,
            "npu_plasticity": self.npu_raid.plasticity,
            "brain_meta_state": self.brain.meta_state,
            "brain_stance": self.brain.stance,
            "brain_preds": brain_preds,
            "timestamp": time.time(),
        }

        self.policy_bus.broadcast(state)
        if self.swarm is not None:
            self.swarm.broadcast(state)

        return state

# ============================================================
# GUI COCKPIT
# ============================================================

class CockpitGUI:
    def __init__(self, scheduler: Scheduler, gpu: GPUOrgan,
                 npu_discovery: NPUDiscoveryOrgan, swarm: SwarmSyncOrgan | None, brain: BrainOrgan):
        self.scheduler = scheduler
        self.gpu = gpu
        self.npu_discovery = npu_discovery
        self.swarm = swarm
        self.brain = brain

        self.root = tk.Tk()
        self.root.title("NPU RAID → GPU Swarm Controller")
        self.root.geometry("900x650")

        self.title_label = ttk.Label(
            self.root,
            text=f"GPU: {self.gpu.get_name()}",
            font=("Arial", 14, "bold")
        )
        self.title_label.pack(pady=5)

        self.npu_disc_label = ttk.Label(
            self.root,
            text=f"NPU Discovery: {self.npu_discovery.summary()}",
            wraplength=880,
            justify="center"
        )
        self.npu_disc_label.pack(pady=5)

        self.swarm_label = ttk.Label(
            self.root,
            text=f"Swarm Node ID: {SWARM_NODE_ID} | Group: {SWARM_GROUP}:{SWARM_PORT}",
            wraplength=880,
            justify="center"
        )
        self.swarm_label.pack(pady=5)

        frame_top = ttk.Frame(self.root)
        frame_top.pack(fill="x", pady=5)

        self.mode_label = ttk.Label(frame_top, text="Mode: ???", font=("Arial", 12))
        self.mode_label.grid(row=0, column=0, padx=5, sticky="w")

        self.brain_meta_label = ttk.Label(frame_top, text="Meta-State: ???", font=("Arial", 12))
        self.brain_meta_label.grid(row=0, column=1, padx=5, sticky="w")

        self.brain_stance_label = ttk.Label(frame_top, text="Stance: ???", font=("Arial", 12))
        self.brain_stance_label.grid(row=0, column=2, padx=5, sticky="w")

        frame_stats = ttk.Frame(self.root)
        frame_stats.pack(fill="x", pady=5)

        self.gpu_util_label = ttk.Label(frame_stats, text="GPU Util: ???")
        self.gpu_util_label.grid(row=0, column=0, padx=5, sticky="w")

        self.mem_util_label = ttk.Label(frame_stats, text="Mem Util: ???")
        self.mem_util_label.grid(row=0, column=1, padx=5, sticky="w")

        self.mem_frac_label = ttk.Label(frame_stats, text="VRAM Used: ???")
        self.mem_frac_label.grid(row=0, column=2, padx=5, sticky="w")

        self.temp_label = ttk.Label(frame_stats, text="Temp: ???")
        self.temp_label.grid(row=1, column=0, padx=5, sticky="w")

        self.power_label = ttk.Label(frame_stats, text="Power: ???")
        self.power_label.grid(row=1, column=1, padx=5, sticky="w")

        self.fan_label = ttk.Label(frame_stats, text="Fan Target: ???")
        self.fan_label.grid(row=1, column=2, padx=5, sticky="w")

        self.policy_label = ttk.Label(self.root, text="AI Perf / PowerScale: ???")
        self.policy_label.pack(pady=5)

        self.npu_label = ttk.Label(self.root, text="NPU RAID [Conf / Int / Plast]: ???")
        self.npu_label.pack(pady=5)

        self.swarm_peers_label = ttk.Label(self.root, text="Swarm Peers: none", justify="left")
        self.swarm_peers_label.pack(pady=5)

        frame_bottom = ttk.Frame(self.root)
        frame_bottom.pack(fill="both", expand=True, pady=5)

        self.canvas_chart = tk.Canvas(frame_bottom, width=500, height=200, bg="#111111", highlightthickness=0)
        self.canvas_chart.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.txt_reason = tk.Text(frame_bottom, width=50, height=12, bg="#111111", fg="#dddddd")
        self.txt_reason.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        frame_bottom.columnconfigure(0, weight=1)
        frame_bottom.columnconfigure(1, weight=1)
        frame_bottom.rowconfigure(0, weight=1)

        self.status_label = ttk.Label(self.root, text="Status: running", foreground="green")
        self.status_label.pack(pady=5)

        self.update_interval_ms = 1000
        self._stop = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_loop()

    def on_close(self):
        self._stop = True
        self.root.destroy()

    def _draw_chart(self, preds):
        self.canvas_chart.delete("all")
        w = int(self.canvas_chart["width"])
        h = int(self.canvas_chart["height"])

        self.canvas_chart.create_rectangle(0, 0, w, h, fill="#111111", outline="")

        short = preds.get("short", 0.5)
        med = preds.get("medium", 0.5)
        long = preds.get("long", 0.5)
        baseline = preds.get("baseline", 0.5)
        best_guess = (short * 0.4 + med * 0.3 + long * 0.2 + baseline * 0.1)

        def y_from_val(v):
            return h - int(v * (h - 10)) - 5

        x_short = w * 0.2
        x_med = w * 0.5
        x_long = w * 0.8

        y_short = y_from_val(short)
        y_med = y_from_val(med)
        y_long = y_from_val(long)
        y_base = y_from_val(baseline)
        y_best = y_from_val(best_guess)

        self.canvas_chart.create_line(0, y_base, w, y_base, fill="#555555", dash=(2, 2))

        self.canvas_chart.create_line(x_short, y_short, x_med, y_med, fill="#00ccff", width=2)
        self.canvas_chart.create_line(x_med, y_med, x_long, y_long, fill="#00ccff", width=2)

        stance_color = {
            "Conservative": "#66ff66",
            "Balanced": "#ffff66",
            "Beast": "#ff6666",
        }.get(self.brain.stance, "#ffffff")
        self.canvas_chart.create_line(x_short, y_med, x_long, y_med, fill=stance_color, width=1)

        self.canvas_chart.create_line(0, y_best, w, y_best, fill="#ff00ff", width=2)

        self.canvas_chart.create_text(5, 5, anchor="nw", fill="#aaaaaa",
                                      text="Short/Med/Long (cyan), Baseline (gray), Best-Guess (magenta)")

    def _update_reasoning(self):
        self.txt_reason.delete("1.0", tk.END)
        self.txt_reason.insert(tk.END, "Reasoning Tail:\n")
        for line in self.brain.last_reasoning:
            self.txt_reason.insert(tk.END, f"  - {line}\n")

        self.txt_reason.insert(tk.END, "\nReasoning Heatmap:\n")
        for k, v in self.brain.last_heatmap.items():
            if k == "best_guess_contributors":
                continue
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

    def update_loop(self):
        if self._stop:
            return

        try:
            state = self.scheduler.step()

            self.mode_label.config(text=f"Mode: {state['mode']}")
            self.brain_meta_label.config(text=f"Meta-State: {state['brain_meta_state']}")
            self.brain_stance_label.config(text=f"Stance: {state['brain_stance']}")

            self.gpu_util_label.config(text=f"GPU Util: {state['gpu_util']:.1f}%")
            self.mem_util_label.config(text=f"Mem Util: {state['mem_util']:.1f}%")
            self.mem_frac_label.config(text=f"VRAM Used: {state['mem_frac']*100:.1f}%")
            self.temp_label.config(text=f"Temp: {state['temp']:.1f} °C")
            self.power_label.config(
                text=f"Power: {state['power']:.1f} / {state['power_limit']:.1f} W"
            )
            self.fan_label.config(text=f"Fan Target: {state['fan_target']}%")

            self.policy_label.config(
                text=f"AI Perf: {state['perf']:.2f} | PowerScale: {state['power_scale']:.2f}"
            )
            self.npu_label.config(
                text=f"NPU RAID [Conf / Int / Plast]: "
                     f"{state['npu_confidence']:.2f} / {state['npu_integrity']:.2f} / {state['npu_plasticity']:.2f}"
            )

            if self.swarm is not None:
                peers = self.swarm.snapshot_peers()
                if peers:
                    peer_strs = []
                    for nid, info in peers.items():
                        ls = time.strftime("%H:%M:%S", time.localtime(info["last_seen"]))
                        mode = info["last_state"].get("mode")
                        gu = info["last_state"].get("gpu_util")
                        peer_strs.append(f"{nid} [{mode} @ {gu:.1f}%] last={ls}")
                    self.swarm_peers_label.config(
                        text="Swarm Peers:\n" + "\n".join(peer_strs)
                    )
                else:
                    self.swarm_peers_label.config(text="Swarm Peers: none")
            else:
                self.swarm_peers_label.config(text="Swarm Peers: disabled")

            self._draw_chart(self.brain.last_predictions)
            self._update_reasoning()

            self.status_label.config(text="Status: running", foreground="green")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}", foreground="red")

        self.root.after(self.update_interval_ms, self.update_loop)

    def run(self):
        self.root.mainloop()

# ============================================================
# MAIN
# ============================================================

def main():
    npu_discovery = NPUDiscoveryOrgan()
    gpu = GPUOrgan(index=0)
    fan = GPUFanOrgan(gpu_index=0)
    raid_count = npu_discovery.count()
    print(f"[NPU RAID] Logical NPU controllers: {raid_count}")
    npu_raid = NPURaidEnsemble(count=raid_count, target_util=0.8, max_temp=80.0)
    brain = BrainOrgan(history_len=64)
    crypto = CryptoOrgan(key_path=CRYPTO_KEY_FILE)
    policy_bus = SocketPolicyBus(host=POLICY_BUS_HOST, port=POLICY_BUS_PORT, crypto=crypto)
    swarm = SwarmSyncOrgan(group=SWARM_GROUP, port=SWARM_PORT, node_id=SWARM_NODE_ID, crypto=crypto) if SWARM_ENABLED else None
    scheduler = Scheduler(gpu, fan, npu_raid, brain, policy_bus, swarm)
    gui = CockpitGUI(scheduler, gpu, npu_discovery, swarm, brain)
    try:
        gui.run()
    finally:
        policy_bus.stop()
        if swarm is not None:
            swarm.stop()

if __name__ == "__main__":
    main()

