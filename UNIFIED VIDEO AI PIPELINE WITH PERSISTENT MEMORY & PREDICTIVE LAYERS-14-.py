#!/usr/bin/env python3
# rebootcore_async_omega_plus_plus.py
#
# RebootCore – ASYNC OMEGA++ UNIFIED RUNTIME
# - Async scheduler (asyncio main loop)
# - Upgraded DQN training (epsilon-greedy, target updates, replay)
# - RewardEngine (mission + safety + smoothness)
# - Model saving/loading (DQN weights + epsilon)
# - Simulation bridge (stub for AirSim/Gazebo/Isaac)
# - Multi-agent RL (per-agent policies, shared buffer)
# - SafetySupervisor (hard limits, action clamping, fallback)
# - TrajectoryPlanner (high-level target → safe velocity)
# - Real 3D renderer (moderngl + glfw) with camera orbit
# - Swarm formations (line, circle, V, orbit) + DSL
# - YOLO multi-backend (ultralytics / YOLO-NAS / YOLOv10 stubs)
# - GPU video pipeline + GPU optical flow (if available)
# - Plugin sandboxing via multiprocessing
# - Real autopilot control loop abstraction
# - Single-GUI selection (Qt OR DearPyGUI)
# - Auto-elevation, auto-fix networking, auto-fix HTTP API
# - No save prompts, dummy frames if no camera

# === AUTO-ELEVATION CHECK (Windows only) ===
import ctypes
import os
import sys

def ensure_admin():
    if os.name != "nt":
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
        sys.exit()

ensure_admin()

import json
import time
import threading
import queue
import socket
import random
import asyncio
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

# ---------- Optional heavy deps ----------
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# YOLO backends
HAS_ULTRA = False
HAS_YOLO_NAS = False
HAS_YOLOV10 = False
YOLO_BACKEND = "stub"

try:
    from ultralytics import YOLO
    HAS_ULTRA = True
except ImportError:
    pass

try:
    from super_gradients.training import models as sg_models
    HAS_YOLO_NAS = True
except ImportError:
    pass

try:
    import yolov10  # type: ignore
    HAS_YOLOV10 = True
except ImportError:
    pass

# DearPyGUI
try:
    import dearpygui.dearpygui as dpg
    HAS_DPG = True
except ImportError:
    HAS_DPG = False

# Qt
HAS_QT = False
QT_WIDGETS = None
try:
    from PySide6 import QtWidgets as QT_WIDGETS
    from PySide6 import QtCore
    HAS_QT = True
except ImportError:
    try:
        from PyQt5 import QtWidgets as QT_WIDGETS
        from PyQt5 import QtCore
        HAS_QT = True
    except ImportError:
        HAS_QT = False

# MAVLink
try:
    from pymavlink import mavutil
    HAS_MAVLINK = True
except ImportError:
    HAS_MAVLINK = False

# CAN / OBD
try:
    import can
    HAS_CAN = True
except ImportError:
    HAS_CAN = False

try:
    import obd
    HAS_OBD = True
except ImportError:
    HAS_OBD = False

# 3D / moderngl
try:
    import moderngl
    import glfw
    HAS_MGL = True
except ImportError:
    HAS_MGL = False

from http.server import BaseHTTPRequestHandler, HTTPServer

# =========================
# CONFIG & SAVE MANAGEMENT
# =========================

DEFAULT_BASE_DIR = os.path.join(os.path.expanduser("~"), "RebootCore")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_BASE_DIR, "config.json")
DEFAULT_MEMORY_PATH = os.path.join(DEFAULT_BASE_DIR, "reboot_memory.json")
DEFAULT_PLUGIN_CONFIG_PATH = os.path.join(DEFAULT_BASE_DIR, "plugins.json")
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_BASE_DIR, "models")


@dataclass
class CoreConfig:
    base_dir: str = DEFAULT_BASE_DIR
    memory_path: str = DEFAULT_MEMORY_PATH
    plugin_config_path: str = DEFAULT_PLUGIN_CONFIG_PATH
    model_dir: str = DEFAULT_MODEL_DIR
    video_source_type: str = "auto"  # "auto", "camera", "none"
    camera_index: int = 0
    udp_broadcast_port: int = 50050
    udp_listen_port: int = 50051
    api_port: int = 8080
    target_fps: int = 30
    use_gpu_hud: bool = True
    enable_swarm_visualizer: bool = True
    enable_3d_map: bool = True
    enable_rl: bool = True
    enable_yolo: bool = True
    enable_plugins: bool = True
    enable_gui: bool = True
    headless: bool = False
    use_gpu_inference: bool = True
    gui_mode: str = "dpg"  # "dpg", "qt", "none"
    yolo_backend: str = "auto"  # "auto", "ultralytics", "yolo_nas", "yolov10", "stub"
    shadow_mode: bool = False  # RL observes but does not control actuators
    max_speed: float = 25.0
    max_accel: float = 5.0
    max_altitude: float = 150.0
    min_altitude: float = 5.0
    geofence_radius: float = 1000.0  # meters
    multi_agent: bool = True
    num_agents: int = 3  # for multi-agent RL


class ConfigManager:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = CoreConfig()
        self._ensure_base_dir()
        self.load_or_create()

    def _ensure_base_dir(self):
        base_dir = os.path.dirname(self.config_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

    def load_or_create(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.config = CoreConfig(**data)
            except Exception as e:
                print(f"[Config] Failed to load config, using defaults: {e}")
        else:
            self.save()

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.config), f, indent=2)
        except Exception as e:
            print(f"[Config] Failed to save config: {e}")


# =========================
# MEMORY / PERSISTENCE
# =========================

class MemoryManager:
    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self.state: Dict[str, Any] = {
            "rl_buffer": [],
            "rl_policy": {},
            "plugin_settings": {},
            "last_run": None,
        }
        self._ensure_dir()
        self.load()

    def _ensure_dir(self):
        base_dir = os.path.dirname(self.memory_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

    def load(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
                print("[Memory] Loaded memory state.")
            except Exception as e:
                print(f"[Memory] Failed to load memory: {e}")

    def save(self):
        try:
            self.state["last_run"] = time.time()
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
            print("[Memory] Saved memory state.")
        except Exception as e:
            print(f"[Memory] Failed to save memory: {e}")


# =========================
# VIDEO / FRAME PIPELINE + GPU OPTICAL FLOW
# =========================

class VideoManager:
    def __init__(self, cfg: CoreConfig):
        self.cfg = cfg
        self.cap = None
        self.no_video = False
        self.frame_queue: "queue.Queue[Any]" = queue.Queue(maxsize=3)
        self.running = False
        self.use_cuda = False
        self.prev_gray_gpu = None
        self.optical_flow = None
        self._init_gpu()

    def _init_gpu(self):
        if cv2 is not None and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.use_cuda = True
            print("[Video] CUDA path enabled for frames.")
            try:
                self.optical_flow = cv2.cuda_FarnebackOpticalFlow.create(
                    5, 0.5, False, 15, 3, 5, 1.2, 0
                )
                print("[Video] CUDA optical flow enabled.")
            except Exception as e:
                print(f"[Video] CUDA optical flow init failed: {e}")
                self.optical_flow = None

    def _init_camera(self):
        if cv2 is None:
            print("[Video] OpenCV not available, disabling camera.")
            return False
        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            print("[Video] Camera not available, disabling camera.")
            self.cap = None
            return False
        print("[Video] Camera initialized.")
        return True

    def _get_dummy_frame(self):
        if np is None or cv2 is None:
            return None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        t = int(time.time() * 50) % 640
        cv2.circle(frame, (t, 240), 20, (0, 255, 0), -1)
        cv2.putText(frame, "Dummy Frame", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return frame

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _compute_optical_flow(self, frame):
        if cv2 is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.use_cuda and self.optical_flow is not None:
            try:
                gray_gpu = cv2.cuda_GpuMat()
                gray_gpu.upload(gray)
                if self.prev_gray_gpu is None:
                    self.prev_gray_gpu = gray_gpu
                    return None
                flow_gpu = self.optical_flow.calc(self.prev_gray_gpu, gray_gpu, None)
                self.prev_gray_gpu = gray_gpu
                flow = flow_gpu.download()
                return flow
            except Exception as e:
                print(f"[Video] CUDA optical flow error, fallback to CPU: {e}")
                self.use_cuda = False
                self.optical_flow = None
        if not hasattr(self, "_prev_gray_cpu"):
            self._prev_gray_cpu = gray
            return None
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray_cpu, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self._prev_gray_cpu = gray
        return flow

    def _loop(self):
        use_camera = False
        if self.cfg.video_source_type in ("auto", "camera"):
            use_camera = self._init_camera()

        if not use_camera:
            self.no_video = True
            print("[Video] Running in dummy-frame mode (no camera).")

        frame_interval = 1.0 / max(1, self.cfg.target_fps)

        while self.running:
            start = time.perf_counter()
            frame = None

            if use_camera and self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    frame = None

            if frame is None:
                frame = self._get_dummy_frame()

            flow = None
            if frame is not None:
                flow = self._compute_optical_flow(frame)

            if frame is not None:
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put((frame, flow), timeout=0.002)
                except queue.Full:
                    pass

            elapsed = time.perf_counter() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None, None

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# =========================
# YOLO / OBJECT DETECTION (multi-backend)
# =========================

class ObjectDetector:
    def __init__(self, cfg: CoreConfig):
        self.enabled = cfg.enable_yolo
        self.backend = self._select_backend(cfg.yolo_backend)
        self.model = None
        self.device = "cuda" if cfg.use_gpu_inference and HAS_TORCH and torch.cuda.is_available() else "cpu"
        self._load_model()

    def _select_backend(self, mode: str) -> str:
        if mode == "ultralytics" and HAS_ULTRA:
            return "ultralytics"
        if mode == "yolo_nas" and HAS_YOLO_NAS:
            return "yolo_nas"
        if mode == "yolov10" and HAS_YOLOV10:
            return "yolov10"
        if mode == "auto":
            if HAS_YOLOV10:
                return "yolov10"
            if HAS_YOLO_NAS:
                return "yolo_nas"
            if HAS_ULTRA:
                return "ultralytics"
        return "stub"

    def _load_model(self):
        if self.backend == "ultralytics":
            try:
                self.model = YOLO("yolov8n.pt")
                self.model.to(self.device)
                print(f"[YOLO] ultralytics backend on {self.device}")
            except Exception as e:
                print(f"[YOLO] ultralytics load error: {e}")
                self.backend = "stub"
        elif self.backend == "yolo_nas":
            try:
                self.model = sg_models.get("yolo_nas_s", pretrained_weights="coco")
                print("[YOLO] YOLO-NAS backend loaded.")
            except Exception as e:
                print(f"[YOLO] YOLO-NAS load error: {e}")
                self.backend = "stub"
        elif self.backend == "yolov10":
            try:
                self.model = yolov10.YOLOv10("yolov10n.pt")  # type: ignore
                print("[YOLO] YOLOv10 backend loaded.")
            except Exception as e:
                print(f"[YOLO] YOLOv10 load error: {e}")
                self.backend = "stub"
        else:
            print("[YOLO] Using stub backend.")

    def _preprocess(self, frame):
        return frame

    def detect(self, frame):
        if not self.enabled or frame is None:
            return []
        if self.backend == "stub" or self.model is None:
            h, w = frame.shape[:2]
            return [{
                "label": "dummy_object",
                "conf": 0.9,
                "bbox": [w // 4, h // 4, w // 2, h // 2]
            }]
        try:
            if self.backend == "ultralytics":
                results = self.model(frame, verbose=False)
                dets = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = self.model.names.get(cls, str(cls))
                        dets.append({
                            "label": label,
                            "conf": conf,
                            "bbox": [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                        })
                return dets
            elif self.backend == "yolo_nas":
                frame_p = self._preprocess(frame)
                _ = self.model.predict(frame_p)
                return []
            elif self.backend == "yolov10":
                frame_p = self._preprocess(frame)
                _ = self.model(frame_p)  # type: ignore
                return []
        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
        return []


# =========================
# TRACKING / PHYSICS (STUB)
# =========================

class Tracker:
    def __init__(self):
        self.enabled = True

    def track(self, detections):
        if not self.enabled:
            return []
        return detections


class PhysicsEngine:
    def __init__(self):
        self.enabled = True

    def predict(self, tracked_objects, flow=None):
        if not self.enabled:
            return tracked_objects
        for obj in tracked_objects:
            obj["vx"] = 0.0
            obj["vy"] = 0.0
        return tracked_objects


# =========================
# SIMULATION BRIDGE (STUB)
# =========================

class SimulationBridge:
    """
    Stub for connecting to AirSim/Gazebo/Isaac.
    In real use, this would:
      - Step the sim
      - Provide synthetic sensor data
      - Apply actions to simulated vehicles
    """
    def __init__(self, cfg: CoreConfig):
        self.cfg = cfg
        self.enabled = False  # set True when wired to a real sim
        self.last_state = {}

    def step(self, actions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        actions: {agent_id: {"throttle": ..., "steer": ...}}
        returns: simulated state dict
        """
        if not self.enabled:
            return {}
        # Placeholder: real implementation would talk to sim
        return self.last_state


# =========================
# RL AUTOPILOT – DQN STYLE + TRAINING (UPGRADED, MULTI-AGENT)
# =========================

class DQN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RewardEngine:
    """
    Centralized reward computation:
      - Mission performance
      - Smoothness
      - Safety penalties
    """
    def __init__(self, cfg: CoreConfig):
        self.cfg = cfg

    def compute_reward(
        self,
        prev_state: Dict[str, Any],
        state: Dict[str, Any],
        action: Dict[str, float],
        agent_id: str = "agent0",
    ) -> float:
        reward = 0.0

        # 1) Perception / tracking quality
        dets = state.get("detections", [])
        if len(dets) > 0:
            reward += 0.05
        else:
            reward -= 0.02

        # 2) Smoothness (penalize large control changes)
        prev_action = prev_state.get("action", {}) if prev_state else {}
        prev_throttle = float(prev_action.get("throttle", 0.0))
        prev_steer = float(prev_action.get("steer", 0.0))
        d_throttle = abs(action["throttle"] - prev_throttle)
        d_steer = abs(action["steer"] - prev_steer)
        reward -= 0.01 * (d_throttle + d_steer)

        # 3) Safety: speed, altitude, flow, geofence
        can_state = state.get("can", {})
        speed = float(can_state.get("speed_kph", 0.0))
        if speed > self.cfg.max_speed:
            reward -= 0.1

        mav_state = state.get("mavlink", {})
        alt = float(mav_state.get("altitude", 50.0))
        if alt < self.cfg.min_altitude or alt > self.cfg.max_altitude:
            reward -= 0.1

        flow_mag = float(state.get("flow_mag", 0.0))
        if flow_mag > 5.0:
            reward -= 0.05

        # 4) Swarm cohesion (if multi-agent)
        swarm = state.get("swarm", {})
        positions = swarm.get("positions", [])
        if positions and self.cfg.multi_agent:
            # simple cohesion: reward if drones are not too far apart
            xs = [p.get("x", 0.0) for p in positions]
            ys = [p.get("y", 0.0) for p in positions]
            if len(xs) > 1:
                spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
                reward -= 0.0005 * spread

        return reward


class RLCore:
    """
    Multi-agent capable RL core.
    - Each agent has its own DQN
    - Shared replay buffer
    """
    def __init__(self, memory: MemoryManager, cfg: CoreConfig, reward_engine: RewardEngine):
        self.memory = memory
        self.cfg = cfg
        self.enabled = cfg.enable_rl
        self.altered_mode = False
        self.reward_engine = reward_engine

        self.device = torch.device("cuda" if (HAS_TORCH and cfg.use_gpu_inference and torch.cuda.is_available()) else "cpu") if HAS_TORCH else None

        self.agents: Dict[str, Dict[str, Any]] = {}
        self.gamma = 0.99
        self.train_interval = 10
        self.step_count = 0

        # Shared epsilon across agents for simplicity
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

        if HAS_TORCH:
            for i in range(cfg.num_agents if cfg.multi_agent else 1):
                agent_id = f"agent{i}"
                model = DQN().to(self.device)
                target_model = DQN().to(self.device)
                target_model.load_state_dict(model.state_dict())
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                self.agents[agent_id] = {
                    "model": model,
                    "target": target_model,
                    "optimizer": optimizer,
                }
        print(f"[RL] Multi-agent RL: {len(self.agents)} agents on device: {self.device}")

    def _build_state_vector(self, state: Dict[str, Any]):
        vec = []
        dets = state.get("detections", [])
        vec.append(float(len(dets)))
        if dets:
            d0 = dets[0]
            bbox = d0.get("bbox", [0, 0, 0, 0])
            vec.extend([float(x) for x in bbox])
        else:
            vec.extend([0.0, 0.0, 0.0, 0.0])

        can_state = state.get("can", {})
        vec.append(float(can_state.get("speed_kph", 0.0)))
        vec.append(float(can_state.get("rpm", 0.0)))

        mav_state = state.get("mavlink", {})
        vec.append(float(mav_state.get("altitude", 0.0)))
        mode = mav_state.get("mode", "UNKNOWN")
        vec.append(float(len(mode)))

        swarm = state.get("swarm", {})
        vec.append(float(swarm.get("num_drones", 0)))

        flow_mag = state.get("flow_mag", 0.0)
        vec.append(float(flow_mag))

        while len(vec) < 32:
            vec.append(0.0)
        return vec[:32]

    def select_action(self, state: Dict[str, Any], agent_id: str = "agent0") -> Dict[str, float]:
        if not self.enabled:
            return {"throttle": 0.0, "steer": 0.0}

        if not HAS_TORCH or agent_id not in self.agents:
            base = 0.1 if self.altered_mode else 0.0
            return {
                "throttle": base + random.uniform(-0.1, 0.1),
                "steer": random.uniform(-0.2, 0.2)
            }

        model = self.agents[agent_id]["model"]

        if random.random() < self.epsilon:
            throttle = random.uniform(-1.0, 1.0)
            steer = random.uniform(-1.0, 1.0)
        else:
            vec = self._build_state_vector(state)
            x = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = model(x)[0].cpu().numpy()
            throttle = float(torch.tanh(torch.tensor(q_values[0])).item())
            steer = float(torch.tanh(torch.tensor(q_values[1])).item())

        if self.altered_mode:
            throttle *= 1.2
            steer *= 1.2

        return {"throttle": throttle, "steer": steer}

    def store_transition(self, transition: Dict[str, Any]):
        buf = self.memory.state.setdefault("rl_buffer", [])
        buf.append(transition)
        if len(buf) > 200000:
            buf.pop(0)

    def _sample_batch(self, batch_size=64):
        buf = self.memory.state.get("rl_buffer", [])
        if len(buf) < batch_size:
            return None
        batch = random.sample(buf, batch_size)
        return batch

    def train_step(self):
        if not HAS_TORCH or not self.agents:
            return
        batch = self._sample_batch()
        if batch is None:
            return

        # For simplicity, train all agents on shared buffer
        for agent_id, agent in self.agents.items():
            model = agent["model"]
            target_model = agent["target"]
            optimizer = agent["optimizer"]

            states = []
            rewards = []
            next_states = []
            dones = []

            for t in batch:
                states.append(self._build_state_vector(t["state"]))
                rewards.append(t.get("reward", 0.0))
                ns = t.get("next_state", t["state"])
                next_states.append(self._build_state_vector(ns))
                dones.append(float(t.get("done", False)))

            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

            q_values = model(states)
            q_selected = q_values

            with torch.no_grad():
                q_next = target_model(next_states)
                q_next_max = q_next.max(dim=1)[0]
                target = rewards_t + self.gamma * (1.0 - dones_t) * q_next_max

            loss = F.mse_loss(q_selected.mean(dim=1), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if random.random() < 0.01:
                target_model.load_state_dict(model.state_dict())
                print(f"[RL] Target network updated for {agent_id}.")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    async def training_loop(self):
        while True:
            if self.enabled:
                self.train_step()
            await asyncio.sleep(0.5)

    # -------- Model persistence --------
    def save_models(self, model_dir: str):
        if not HAS_TORCH:
            return
        os.makedirs(model_dir, exist_ok=True)
        for agent_id, agent in self.agents.items():
            path = os.path.join(model_dir, f"{agent_id}_dqn.pt")
            torch.save({
                "model_state": agent["model"].state_dict(),
                "target_state": agent["target"].state_dict(),
                "epsilon": self.epsilon,
            }, path)
            print(f"[RL] Saved model for {agent_id} -> {path}")

    def load_models(self, model_dir: str):
        if not HAS_TORCH:
            return
        for agent_id, agent in self.agents.items():
            path = os.path.join(model_dir, f"{agent_id}_dqn.pt")
            if not os.path.exists(path):
                continue
            try:
                ckpt = torch.load(path, map_location=self.device)
                agent["model"].load_state_dict(ckpt["model_state"])
                agent["target"].load_state_dict(ckpt["target_state"])
                self.epsilon = ckpt.get("epsilon", self.epsilon)
                print(f"[RL] Loaded model for {agent_id} from {path}")
            except Exception as e:
                print(f"[RL] Failed to load model for {agent_id}: {e}")


# =========================
# TRAJECTORY PLANNER + SAFETY SUPERVISOR
# =========================

class TrajectoryPlanner:
    """
    Simple high-level planner:
      - Takes a target (e.g., waypoint)
      - Produces a desired velocity vector
    """
    def __init__(self, cfg: CoreConfig):
        self.cfg = cfg
        self.current_target: Optional[Tuple[float, float, float]] = None  # x, y, z

    def set_target(self, x: float, y: float, z: float):
        self.current_target = (x, y, z)
        print(f"[Planner] New target set: ({x}, {y}, {z})")

    def compute_desired_velocity(self, state: Dict[str, Any]) -> Dict[str, float]:
        if self.current_target is None:
            return {"vx": 0.0, "vy": 0.0, "vz": 0.0}

        swarm = state.get("swarm", {})
        positions = swarm.get("positions", [])
        if not positions:
            return {"vx": 0.0, "vy": 0.0, "vz": 0.0}

        # Use first drone as reference
        p0 = positions[0]
        x = float(p0.get("x", 0.0))
        y = float(p0.get("y", 0.0))
        z = float(p0.get("z", 10.0))

        tx, ty, tz = self.current_target
        dx = tx - x
        dy = ty - y
        dz = tz - z

        dist = max(1e-3, (dx**2 + dy**2 + dz**2) ** 0.5)
        max_v = self.cfg.max_speed

        vx = max(-max_v, min(max_v, dx / dist * max_v * 0.5))
        vy = max(-max_v, min(max_v, dy / dist * max_v * 0.5))
        vz = max(-max_v, min(max_v, dz / dist * max_v * 0.5))

        return {"vx": vx, "vy": vy, "vz": vz}


class SafetySupervisor:
    """
    Enforces hard safety constraints:
      - Speed limits
      - Altitude limits
      - Geofence
      - Action clamping
    """
    def __init__(self, cfg: CoreConfig):
        self.cfg = cfg

    def clamp_action(self, action: Dict[str, float]) -> Dict[str, float]:
        throttle = max(-1.0, min(1.0, action.get("throttle", 0.0)))
        steer = max(-1.0, min(1.0, action.get("steer", 0.0)))
        return {"throttle": throttle, "steer": steer}

    def check_state_safety(self, state: Dict[str, Any]) -> bool:
        # Returns True if safe, False if unsafe
        can_state = state.get("can", {})
        speed = float(can_state.get("speed_kph", 0.0))
        if speed > self.cfg.max_speed * 1.5:
            return False

        mav_state = state.get("mavlink", {})
        alt = float(mav_state.get("altitude", 50.0))
        if alt < self.cfg.min_altitude * 0.5 or alt > self.cfg.max_altitude * 1.5:
            return False

        # Geofence check (if we had GPS positions)
        # Placeholder: always safe for now
        return True

    def blend_with_baseline(
        self,
        rl_action: Dict[str, float],
        baseline_action: Dict[str, float],
        alpha: float = 0.2,
    ) -> Dict[str, float]:
        return {
            "throttle": (1 - alpha) * baseline_action.get("throttle", 0.0) + alpha * rl_action.get("throttle", 0.0),
            "steer": (1 - alpha) * baseline_action.get("steer", 0.0) + alpha * rl_action.get("steer", 0.0),
        }


# =========================
# AUTOPILOT FLIGHT STACK ABSTRACTION
# =========================

class FlightStackInterface:
    def __init__(self, core: "RebootCore"):
        self.core = core

    def send_attitude_target(self, roll: float, pitch: float, yaw: float, thrust: float):
        print(f"[FlightStack] Attitude target r={roll:.2f} p={pitch:.2f} y={yaw:.2f} t={thrust:.2f}")

    def send_velocity_target(self, vx: float, vy: float, vz: float):
        print(f"[FlightStack] Velocity target vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}")

    def apply_action(self, action: Dict[str, float], state: Dict[str, Any]):
        """
        High-level: RL action + planner → velocity target
        """
        planner = self.core.planner
        desired_vel = planner.compute_desired_velocity(state)
        # Map RL throttle/steer into a modifier on desired velocity
        scale_throttle = action["throttle"]
        scale_steer = action["steer"]

        vx = desired_vel["vx"] + scale_throttle * 2.0
        vy = desired_vel["vy"] + scale_steer * 2.0
        vz = desired_vel["vz"]

        self.send_velocity_target(vx, vy, vz)


# =========================
# PLUGIN SYSTEM WITH SANDBOXING (multiprocessing)
# =========================

class PluginBase:
    name: str = "BasePlugin"
    version: str = "0.0.1"

    def __init__(self, core_ref: "RebootCore"):
        self.core = core_ref

    def on_load(self):
        pass

    def on_unload(self):
        pass

    def on_tick(self, dt: float, state: Dict[str, Any]):
        pass

    def get_settings_schema(self) -> Dict[str, Any]:
        return {}

    def apply_settings(self, settings: Dict[str, Any]):
        pass


def plugin_worker_loop(plugin_class, settings, in_q: mp.Queue, out_q: mp.Queue):
    core_stub = None
    plugin = plugin_class(core_stub)
    plugin.apply_settings(settings)
    plugin.on_load()
    last_state = {}
    while True:
        try:
            msg = in_q.get()
        except EOFError:
            break
        if msg is None:
            break
        if msg["type"] == "tick":
            dt = msg["dt"]
            state = msg["state"]
            try:
                plugin.on_tick(dt, state)
            except Exception as e:
                state.setdefault("_plugin_errors", []).append(str(e))
            last_state = state
            out_q.put({"type": "state", "state": last_state})
        elif msg["type"] == "shutdown":
            break
    plugin.on_unload()


class MAVLinkPlugin(PluginBase):
    name = "MAVLinkPlugin"
    version = "0.5.0"

    def __init__(self, core_ref: "RebootCore"):
        super().__init__(core_ref)
        self.connected = False
        self.master = None
        self.connection_string = "udp:127.0.0.1:14550"

    def get_settings_schema(self):
        return {
            "connection_string": {
                "type": "string",
                "default": self.connection_string
            }
        }

    def apply_settings(self, settings: Dict[str, Any]):
        self.connection_string = settings.get("connection_string", self.connection_string)

    def on_load(self):
        if not HAS_MAVLINK:
            print("[Plugin:MAVLink] pymavlink not installed, stub mode.")
            return
        try:
            self.master = mavutil.mavlink_connection(self.connection_string)
            self.connected = True
            print(f"[Plugin:MAVLink] Connected to {self.connection_string}")
        except Exception as e:
            print(f"[Plugin:MAVLink] Failed to connect: {e}")
            self.connected = False

    def on_tick(self, dt: float, state: Dict[str, Any]):
        state.setdefault("mavlink", {})
        if not HAS_MAVLINK or not self.connected or self.master is None:
            state["mavlink"]["altitude"] = 100.0
            state["mavlink"]["mode"] = "AUTO"
            return
        try:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                return
            msg_type = msg.get_type()
            if msg_type == "HEARTBEAT":
                mode = mavutil.mode_string_v10(msg)
                state["mavlink"]["mode"] = mode
            elif msg_type == "GLOBAL_POSITION_INT":
                alt = msg.alt / 1000.0
                state["mavlink"]["altitude"] = alt
        except Exception as e:
            print(f"[Plugin:MAVLink] Error: {e}")


class CANBusPlugin(PluginBase):
    name = "CANBusPlugin"
    version = "0.5.0"

    def __init__(self, core_ref: "RebootCore"):
        super().__init__(core_ref)
        self.bus = None
        self.obd_conn = None
        self.channel = "can0"
        self.bustype = "socketcan"

    def get_settings_schema(self):
        return {
            "channel": {"type": "string", "default": self.channel},
            "bustype": {"type": "string", "default": self.bustype},
        }

    def apply_settings(self, settings: Dict[str, Any]):
        self.channel = settings.get("channel", self.channel)
        self.bustype = settings.get("bustype", self.bustype)

    def on_load(self):
        if HAS_CAN:
            try:
                self.bus = can.interface.Bus(channel=self.channel, bustype=self.bustype)
                print(f"[Plugin:CAN] Connected to CAN {self.channel} ({self.bustype})")
            except Exception as e:
                print(f"[Plugin:CAN] Failed to connect to CAN: {e}")
        if HAS_OBD:
            try:
                self.obd_conn = obd.OBD()
                print("[Plugin:CAN] OBD-II connected.")
            except Exception as e:
                print(f"[Plugin:CAN] Failed to connect OBD-II: {e}")

    def on_tick(self, dt: float, state: Dict[str, Any]):
        state.setdefault("can", {})
        if HAS_OBD and self.obd_conn is not None and self.obd_conn.is_connected():
            try:
                speed = self.obd_conn.query(obd.commands.SPEED)
                rpm = self.obd_conn.query(obd.commands.RPM)
                if not speed.is_null():
                    state["can"]["speed_kph"] = float(speed.value.magnitude)
                if not rpm.is_null():
                    state["can"]["rpm"] = float(rpm.value.magnitude)
            except Exception as e:
                print(f"[Plugin:CAN] OBD error: {e}")
        else:
            state["can"].setdefault("speed_kph", 42.0)
            state["can"].setdefault("rpm", 2500.0)


class SwarmPlugin(PluginBase):
    name = "SwarmPlugin"
    version = "0.5.0"

    def __init__(self, core_ref: "RebootCore"):
        super().__init__(core_ref)
        self.num_drones = 5
        self.positions: List[Dict[str, float]] = []
        self.formation = "line"  # "line", "circle", "v", "orbit"

    def get_settings_schema(self):
        return {
            "num_drones": {"type": "int", "default": self.num_drones},
            "formation": {"type": "string", "default": self.formation},
        }

    def apply_settings(self, settings: Dict[str, Any]):
        self.num_drones = int(settings.get("num_drones", self.num_drones))
        self.formation = settings.get("formation", self.formation)

    def _init_positions(self):
        self.positions = []
        for i in range(self.num_drones):
            self.positions.append({"id": i, "x": 0.0, "y": 0.0, "z": 10.0})

    def _apply_formation(self, t: float):
        if not self.positions:
            self._init_positions()
        if self.formation == "line":
            for i, p in enumerate(self.positions):
                p["x"] = i * 20.0
                p["y"] = 0.0
                p["z"] = 10.0
        elif self.formation == "circle":
            radius = 50.0
            for i, p in enumerate(self.positions):
                angle = 2.0 * 3.14159 * i / max(1, self.num_drones)
                p["x"] = radius * float(np.cos(angle)) if np is not None else radius
                p["y"] = radius * float(np.sin(angle)) if np is not None else 0.0
                p["z"] = 10.0
        elif self.formation == "v":
            for i, p in enumerate(self.positions):
                offset = i - self.num_drones // 2
                p["x"] = abs(offset) * 20.0
                p["y"] = offset * 20.0
                p["z"] = 10.0
        elif self.formation == "orbit":
            radius = 60.0
            for i, p in enumerate(self.positions):
                angle = 2.0 * 3.14159 * i / max(1, self.num_drones) + t * 0.2
                p["x"] = radius * float(np.cos(angle)) if np is not None else radius
                p["y"] = radius * float(np.sin(angle)) if np is not None else 0.0
                p["z"] = 10.0

    def on_tick(self, dt: float, state: Dict[str, Any]):
        state.setdefault("swarm", {})
        state["swarm"]["num_drones"] = self.num_drones
        t = time.time()
        self._apply_formation(t)
        state["swarm"]["positions"] = self.positions


class VehicleAutopilotPlugin(PluginBase):
    name = "VehicleAutopilotPlugin"
    version = "0.5.0"

    def on_tick(self, dt: float, state: Dict[str, Any]):
        state.setdefault("vehicle_autopilot", {})
        state["vehicle_autopilot"]["lane_offset"] = 0.0


class PluginManager:
    def __init__(self, core_ref: "RebootCore", plugin_config_path: str, memory: MemoryManager):
        self.core = core_ref
        self.plugin_config_path = plugin_config_path
        self.settings: Dict[str, Dict[str, Any]] = memory.state.setdefault("plugin_settings", {})
        self.memory = memory
        self._ensure_dir()
        self.processes: Dict[str, Dict[str, Any]] = {}
        self._load_builtin_plugins()

    def _ensure_dir(self):
        base_dir = os.path.dirname(self.plugin_config_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

    def save_settings(self):
        self.memory.state["plugin_settings"] = self.settings

    def _load_builtin_plugins(self):
        builtin = [MAVLinkPlugin, CANBusPlugin, SwarmPlugin, VehicleAutopilotPlugin]
        for cls in builtin:
            self._spawn_plugin_process(cls)

    def _spawn_plugin_process(self, cls):
        name = cls.name
        schema = cls(self.core).get_settings_schema()
        if name in self.settings:
            settings = self.settings[name]
        else:
            defaults = {}
            for k, v in schema.items():
                defaults[k] = v.get("default")
            self.settings[name] = defaults
            settings = defaults
        in_q = mp.Queue()
        out_q = mp.Queue()
        p = mp.Process(target=plugin_worker_loop, args=(cls, settings, in_q, out_q), daemon=True)
        p.start()
        self.processes[name] = {"proc": p, "in": in_q, "out": out_q}

    def tick(self, dt: float, state: Dict[str, Any]):
        for name, info in list(self.processes.items()):
            try:
                info["in"].put({"type": "tick", "dt": dt, "state": state.copy()}, block=False)
            except Exception as e:
                print(f"[Plugins] Failed to send tick to {name}: {e}")
            try:
                while True:
                    msg = info["out"].get_nowait()
                    if msg["type"] == "state":
                        for k, v in msg["state"].items():
                            if k.startswith("_"):
                                continue
                            state[k] = v
            except queue.Empty:
                pass

    def shutdown(self):
        for name, info in self.processes.items():
            try:
                info["in"].put({"type": "shutdown"}, block=False)
            except:
                pass
            try:
                info["proc"].terminate()
            except:
                pass

    def get_marketplace_listing(self) -> List[Dict[str, Any]]:
        return [
            {"name": "MAVLinkPlugin", "version": "0.5.0", "type": "MAVLink"},
            {"name": "CANBusPlugin", "version": "0.5.0", "type": "CAN"},
            {"name": "SwarmPlugin", "version": "0.5.0", "type": "Swarm"},
            {"name": "VehicleAutopilotPlugin", "version": "0.5.0", "type": "Autopilot"},
        ]


# =========================
# SWARM COMMAND LANGUAGE
# =========================

class SwarmCommandLanguage:
    def __init__(self, core: "RebootCore"):
        self.core = core

    def execute(self, cmd: str):
        parts = cmd.strip().split()
        if not parts:
            return
        verb = parts[0]
        args = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("", 1)
                try:
                    args[k] = float(v)
                except ValueError:
                    args[k] = v
        swarm_state = self.core.last_state.get("swarm", {})
        positions = swarm_state.get("positions", [])
        if verb == "move_all":
            dx = float(args.get("x", 0.0))
            dy = float(args.get("y", 0.0))
            dz = float(args.get("z", 0.0))
            for p in positions:
                p["x"] += dx
                p["y"] += dy
                p["z"] += dz
        elif verb == "offset":
            target_id = int(args.get("id", -1))
            dx = float(args.get("dx", 0.0))
            dy = float(args.get("dy", 0.0))
            dz = float(args.get("dz", 0.0))
            for p in positions:
                if p["id"] == target_id:
                    p["x"] += dx
                    p["y"] += dy
                    p["z"] += dz
        elif verb == "formation":
            formation = args.get("type", "line")
            self.core.set_swarm_formation(str(formation))
        elif verb == "target":
            tx = float(args.get("x", 0.0))
            ty = float(args.get("y", 0.0))
            tz = float(args.get("z", 10.0))
            self.core.planner.set_target(tx, ty, tz)
        self.core.last_state.setdefault("swarm", {})["positions"] = positions
        print(f"[SwarmDSL] Executed: {cmd}")


# =========================
# AUTO-FIX NETWORKING + WATCHDOG
# =========================

class NetworkManager:
    def __init__(self, cfg: CoreConfig):
        self.cfg = cfg
        self.broadcast_sock = None
        self.listen_sock = None
        self.running = False
        self.control_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.bound_broadcast_port = None
        self.bound_listen_port = None
        self.network_enabled = True
        self._watchdog_thread = None

    def _safe_socket(self, family, type, proto=0):
        try:
            return socket.socket(family, type, proto)
        except Exception as e:
            print(f"[Net] Socket creation failed: {e}")
            return None

    def _try_bind(self, sock, addr, port, description):
        try:
            sock.bind((addr, port))
            print(f"[Net] Bound {description} on {addr}:{port}")
            return True
        except OSError as e:
            print(f"[Net] Failed to bind {description} on {addr}:{port} -> {e}")
            return False

    def _find_open_port(self, start_port, description):
        for offset in range(0, 50):
            port = start_port + offset
            test = self._safe_socket(socket.AF_INET, socket.SOCK_DGRAM)
            if test is None:
                continue
            try:
                test.bind(("127.0.0.1", port))
                test.close()
                print(f"[Net] Found open {description} port: {port}")
                return port
            except:
                test.close()
                continue
        print(f"[Net] No open {description} ports found in range {start_port}-{start_port+50}")
        return None

    def start(self):
        print("[Net] Starting auto-fix networking system...")
        self.running = True
        self._init_sockets()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _init_sockets(self):
        self.network_enabled = True
        self.broadcast_sock = self._safe_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listen_sock = self._safe_socket(socket.AF_INET, socket.SOCK_DGRAM)

        if not self.broadcast_sock or not self.listen_sock:
            print("[Net] Critical: Could not create sockets. Networking disabled.")
            self.network_enabled = False
            return

        try:
            self.broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception as e:
            print(f"[Net] Warning: Could not enable broadcast: {e}")

        listen_port = self._find_open_port(self.cfg.udp_listen_port, "listen")
        if listen_port is None:
            print("[Net] Listener disabled (no ports available).")
            self.network_enabled = False
            return

        if not self._try_bind(self.listen_sock, "0.0.0.0", listen_port, "listener"):
            print("[Net] Listener fallback to localhost...")
            if not self._try_bind(self.listen_sock, "127.0.0.1", listen_port, "listener"):
                print("[Net] Listener disabled (cannot bind).")
                self.network_enabled = False
                return

        self.bound_listen_port = listen_port

        broadcast_port = self._find_open_port(self.cfg.udp_broadcast_port, "broadcast")
        if broadcast_port is None:
            print("[Net] Broadcast disabled (no ports available).")
            self.network_enabled = False
            return

        self.bound_broadcast_port = broadcast_port

        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()

        print(f"[Net] Networking active. Listen={self.bound_listen_port}, Broadcast={self.bound_broadcast_port}")

    def _listen_loop(self):
        print(f"[Net] Listening on UDP port {self.bound_listen_port}")
        while self.running and self.network_enabled:
            try:
                data, addr = self.listen_sock.recvfrom(4096)
                msg = data.decode("utf-8", errors="ignore")
                try:
                    obj = json.loads(msg)
                    self.control_queue.put(obj)
                except json.JSONDecodeError:
                    pass
            except Exception:
                time.sleep(0.01)

    def broadcast_state(self, state: Dict[str, Any]):
        if not self.network_enabled or not self.broadcast_sock:
            return
        try:
            msg = json.dumps(state).encode("utf-8")
            self.broadcast_sock.sendto(msg, ("<broadcast>", self.bound_broadcast_port))
        except Exception as e:
            print(f"[Net] Broadcast error: {e}")

    def get_control_command(self) -> Optional[Dict[str, Any]]:
        try:
            return self.control_queue.get_nowait()
        except queue.Empty:
            return None

    def _watchdog_loop(self):
        while self.running:
            if not self.network_enabled:
                print("[Net] Watchdog: networking disabled, attempting restart...")
                try:
                    if self.listen_sock:
                        self.listen_sock.close()
                    if self.broadcast_sock:
                        self.broadcast_sock.close()
                except:
                    pass
                time.sleep(2.0)
                self._init_sockets()
            time.sleep(5.0)

    def stop(self):
        self.running = False
        try:
            if self.listen_sock:
                self.listen_sock.close()
            if self.broadcast_sock:
                self.broadcast_sock.close()
        except:
            pass
        print("[Net] Networking stopped.")


# =========================
# HTTP API WITH AUTO-FIX PORT
# =========================

class APIServer(BaseHTTPRequestHandler):
    core_ref: "RebootCore" = None

    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/status"):
            self._send_json(self.core_ref.get_status())
        elif self.path.startswith("/marketplace"):
            self._send_json(self.core_ref.plugin_manager.get_marketplace_listing())
        else:
            self._send_json({"error": "not_found"}, code=404)

    def do_POST(self):
        if self.path.startswith("/toggle"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                obj = {}
            self.core_ref.handle_toggle(obj)
            self._send_json({"ok": True})
        elif self.path.startswith("/swarm_cmd"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                obj = {}
            cmd = obj.get("cmd", "")
            self.core_ref.swarm_dsl.execute(cmd)
            self._send_json({"ok": True, "cmd": cmd})
        else:
            self._send_json({"error": "not_found"}, code=404)


def start_api_server(core: "RebootCore", port: int):
    APIServer.core_ref = core
    server = None
    for offset in range(0, 20):
        p = port + offset
        try:
            server = HTTPServer(("127.0.0.1", p), APIServer)
            print(f"[API] HTTP API running on port {p}")
            break
        except OSError as e:
            print(f"[API] Port {p} unavailable: {e}")
            continue
    if server is None:
        print("[API] Failed to bind any API port, API disabled.")
        return None
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# =========================
# 3D MAP – REAL RENDERER (moderngl + glfw, camera orbit)
# =========================

class Map3DManager:
    def __init__(self, core: "RebootCore"):
        self.core = core
        self.enabled = core.cfg.enable_3d_map and HAS_MGL and np is not None
        self.thread = None

    def start(self):
        if not self.enabled:
            print("[3D] 3D map disabled or moderngl/glfw/np not installed.")
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        if not glfw.init():
            print("[3D] glfw init failed.")
            return
        window = glfw.create_window(800, 600, "RebootCore 3D Swarm", None, None)
        if not window:
            glfw.terminate()
            print("[3D] glfw window creation failed.")
            return
        glfw.make_context_current(window)
        ctx = moderngl.create_context()

        prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(color, 1.0);
                }
            """,
        )

        vbo = ctx.buffer(reserve=4 * 3 * 256)
        vao = ctx.simple_vertex_array(prog, vbo, "in_pos")

        def perspective(fovy, aspect, znear, zfar):
            f = 1.0 / np.tan(fovy / 2.0)
            return np.array([
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (zfar + znear) / (znear - zfar), (2 * zfar * znear) / (znear - zfar)],
                [0, 0, -1, 0],
            ], dtype="f4")

        def look_at(eye, target, up):
            f = target - eye
            f = f / np.linalg.norm(f)
            u = up / np.linalg.norm(up)
            s = np.cross(f, u)
            s = s / np.linalg.norm(s)
            u = np.cross(s, f)
            m = np.eye(4, dtype="f4")
            m[0, :3] = s
            m[1, :3] = u
            m[2, :3] = -f
            translate = np.eye(4, dtype="f4")
            translate[:3, 3] = -eye
            return m @ translate

        while not glfw.window_should_close(window) and self.core.running:
            glfw.poll_events()
            swarm = self.core.last_state.get("swarm", {})
            positions = swarm.get("positions", [])
            verts = []
            for p in positions:
                x = float(p.get("x", 0.0))
                y = float(p.get("y", 0.0))
                z = float(p.get("z", 10.0))
                verts.extend([x, y, z])

            if verts:
                data = np.array(verts, dtype="f4")
                vbo.write(data.tobytes())

            width, height = glfw.get_framebuffer_size(window)
            ctx.viewport = (0, 0, width, height)
            ctx.clear(0.05, 0.05, 0.1)

            t = time.time()
            radius = 200.0
            eye = np.array([radius * np.cos(t * 0.2), radius * np.sin(t * 0.2), 120.0], dtype="f4")
            target = np.array([0.0, 0.0, 0.0], dtype="f4")
            up = np.array([0.0, 0.0, 1.0], dtype="f4")

            proj = perspective(np.radians(45.0), max(0.1, width / max(1.0, float(height))), 1.0, 1000.0)
            view = look_at(eye, target, up)
            mvp = proj @ view

            prog["mvp"].write(mvp.tobytes())
            prog["color"].value = (0.0, 1.0, 0.0)

            if verts:
                vao.render(mode=moderngl.POINTS)

            glfw.swap_buffers(window)
            time.sleep(0.02)

        glfw.terminate()
        print("[3D] 3D map stopped.")


# =========================
# GUI – DearPyGUI HUD + Qt Front-End (selectable)
# =========================

class DPGGUIManager:
    def __init__(self, core: "RebootCore"):
        self.core = core
        self.running = False

    def start(self):
        if not HAS_DPG:
            print("[GUI-DPG] DearPyGUI not installed, GUI disabled.")
            return
        if self.core.cfg.headless or self.core.cfg.gui_mode != "dpg":
            print("[GUI-DPG] Disabled by config.")
            return
        self.running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        dpg.create_context()
        dpg.create_viewport(title="RebootCore HUD / DPG", width=1400, height=800)

        with dpg.window(label="RebootCore Status", width=450, height=350, pos=(10, 10)):
            dpg.add_text("Status")
            dpg.add_text("", tag="status_text")
            dpg.add_button(label="Toggle YOLO", callback=lambda: self.core.toggle_yolo())
            dpg.add_button(label="Toggle RL", callback=lambda: self.core.toggle_rl())
            dpg.add_button(label="Toggle Altered Mode", callback=lambda: self.core.toggle_altered())
            dpg.add_slider_int(label="Target FPS", default_value=self.core.cfg.target_fps,
                               min_value=5, max_value=120,
                               callback=self._on_fps_change)

        with dpg.window(label="Plugin Marketplace", width=450, height=350, pos=(470, 10)):
            dpg.add_text("Available Plugins")
            dpg.add_listbox(items=[], tag="plugin_market_list", num_items=8)
            dpg.add_button(label="Refresh Marketplace", callback=self._refresh_marketplace)

        with dpg.window(label="Swarm Visualizer / 2D", width=450, height=350, pos=(930, 10)):
            dpg.add_text("Swarm Visualizer (2D)")
            with dpg.drawlist(width=430, height=300, tag="swarm_drawlist"):
                pass

        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running() and self.running:
            status = self.core.get_status()
            dpg.set_value("status_text", json.dumps(status, indent=2)[:1000])
            self._update_swarm_visualizer()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def _on_fps_change(self, sender, app_data):
        self.core.cfg.target_fps = int(app_data)

    def _refresh_marketplace(self):
        listing = self.core.plugin_manager.get_marketplace_listing()
        items = [f"{p['name']} ({p['type']}) v{p['version']}" for p in listing]
        dpg.configure_item("plugin_market_list", items=items)

    def _update_swarm_visualizer(self):
        if not dpg.does_item_exist("swarm_drawlist"):
            return
        dpg.delete_item("swarm_drawlist", children_only=True)
        swarm = self.core.last_state.get("swarm", {})
        positions = swarm.get("positions", [])
        for drone in positions:
            x = float(drone.get("x", 0.0)) * 0.5 + 200
            y = float(drone.get("y", 0.0)) * 0.5 + 150
            dpg.draw_circle((x, y), 6, color=(0, 255, 0, 255), fill=(0, 255, 0, 100))
            dpg.draw_text((x + 8, y - 8), f"ID {drone['id']}", color=(255, 255, 255, 255))


class QtMainWindow(QT_WIDGETS.QMainWindow if HAS_QT else object):
    def __init__(self, core: "RebootCore"):
        if not HAS_QT:
            return
        super().__init__()
        self.core = core
        self.setWindowTitle("RebootCore Qt Front-End")
        self.resize(800, 600)

        central = QT_WIDGETS.QWidget()
        layout = QT_WIDGETS.QVBoxLayout(central)

        self.status_label = QT_WIDGETS.QLabel("Status: ")
        layout.addWidget(self.status_label)

        btn_row = QT_WIDGETS.QHBoxLayout()
        self.btn_yolo = QT_WIDGETS.QPushButton("Toggle YOLO")
        self.btn_rl = QT_WIDGETS.QPushButton("Toggle RL")
        self.btn_alt = QT_WIDGETS.QPushButton("Toggle Altered")
        btn_row.addWidget(self.btn_yolo)
        btn_row.addWidget(self.btn_rl)
        btn_row.addWidget(self.btn_alt)
        layout.addLayout(btn_row)

        self.btn_yolo.clicked.connect(self.core.toggle_yolo)
        self.btn_rl.clicked.connect(self.core.toggle_rl)
        self.btn_alt.clicked.connect(self.core.toggle_altered)

        self.setCentralWidget(central)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_status)
        self.timer.start(500)

    def _update_status(self):
        status = self.core.get_status()
        self.status_label.setText("Status:\n" + json.dumps(status, indent=2)[:1000])


class QtGUIManager:
    def __init__(self, core: "RebootCore"):
        self.core = core
        self.thread = None

    def start(self):
        if not HAS_QT:
            print("[GUI-Qt] Qt not installed, Qt GUI disabled.")
            return
        if self.core.cfg.headless or self.core.cfg.gui_mode != "qt":
            print("[GUI-Qt] Disabled by config.")
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        app = QT_WIDGETS.QApplication(sys.argv)
        window = QtMainWindow(self.core)
        window.show()
        app.exec_()


# =========================
# CORE ORCHESTRATOR (ASYNC)
# =========================

class RebootCore:
    def __init__(self):
        self.cfg_manager = ConfigManager()
        self.cfg = self.cfg_manager.config

        self.memory = MemoryManager(self.cfg.memory_path)
        self.video = VideoManager(self.cfg)
        self.detector = ObjectDetector(self.cfg)
        self.tracker = Tracker()
        self.physics = PhysicsEngine()
        self.reward_engine = RewardEngine(self.cfg)
        self.rl = RLCore(self.memory, self.cfg, self.reward_engine)
        self.network = NetworkManager(self.cfg)
        self.plugin_manager = PluginManager(self, self.cfg.plugin_config_path, self.memory)
        self.sim_bridge = SimulationBridge(self.cfg)
        self.safety = SafetySupervisor(self.cfg)
        self.planner = TrajectoryPlanner(self.cfg)
        self.flight_stack = FlightStackInterface(self)
        self.swarm_dsl = SwarmCommandLanguage(self)
        self.map3d = Map3DManager(self)
        self.dpg_gui = DPGGUIManager(self)
        self.qt_gui = QtGUIManager(self)

        self.running = False
        self.last_state: Dict[str, Any] = {}
        self.api_server = None

        # Load RL models if present
        self.rl.load_models(self.cfg.model_dir)

    def toggle_yolo(self):
        self.detector.enabled = not self.detector.enabled
        print(f"[Core] YOLO enabled: {self.detector.enabled}")

    def toggle_rl(self):
        self.rl.enabled = not self.rl.enabled
        print(f"[Core] RL enabled: {self.rl.enabled}")

    def toggle_altered(self):
        self.rl.altered_mode = not self.rl.altered_mode
        print(f"[Core] Altered mode: {self.rl.altered_mode}")

    def handle_toggle(self, obj: Dict[str, Any]):
        if "yolo" in obj:
            self.detector.enabled = bool(obj["yolo"])
        if "rl" in obj:
            self.rl.enabled = bool(obj["rl"])
        if "altered" in obj:
            self.rl.altered_mode = bool(obj["altered"])

    def set_swarm_formation(self, formation: str):
        settings = self.memory.state.setdefault("plugin_settings", {})
        swarm_settings = settings.setdefault("SwarmPlugin", {})
        swarm_settings["formation"] = formation
        print(f"[Core] Swarm formation set to {formation}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "yolo_enabled": self.detector.enabled,
            "rl_enabled": self.rl.enabled,
            "altered_mode": self.rl.altered_mode,
            "video_no_camera": self.video.no_video,
            "target_fps": self.cfg.target_fps,
            "plugins": list(self.plugin_manager.processes.keys()),
            "last_state_keys": list(self.last_state.keys()),
            "epsilon": getattr(self.rl, "epsilon", None),
            "multi_agent": self.cfg.multi_agent,
        }

    async def start_async(self):
        print("[Core] Starting RebootCore ASYNC OMEGA++ ...")
        self.running = True
        self.video.start()
        self.network.start()
        self.api_server = start_api_server(self, self.cfg.api_port)
        self.dpg_gui.start()
        self.qt_gui.start()
        self.map3d.start()

        asyncio.create_task(self.rl.training_loop())
        await self._loop_async()

    def _get_baseline_action(self, state: Dict[str, Any]) -> Dict[str, float]:
        # Placeholder: baseline autopilot (e.g., PID, existing controller)
        # For now, just zero action
        return {"throttle": 0.0, "steer": 0.0}

    async def _loop_async(self):
        frame_interval = 1.0 / max(1, self.cfg.target_fps)
        last_time = time.perf_counter()

        prev_state = None
        prev_action = None

        while self.running:
            start = time.perf_counter()
            dt = start - last_time
            last_time = start

            frame, flow = self.video.get_latest_frame()
            detections = self.detector.detect(frame)
            tracked = self.tracker.track(detections)
            predicted = self.physics.predict(tracked, flow=flow)

            flow_mag = 0.0
            if flow is not None and np is not None:
                flow_mag = float(np.mean(np.linalg.norm(flow, axis=2)))

            state = {
                "detections": detections,
                "tracked": tracked,
                "predicted": predicted,
                "time": start,
                "flow_mag": flow_mag,
            }

            self.plugin_manager.tick(dt, state)

            # Multi-agent RL: for now, use agent0 as main controller
            agent_id = "agent0"
            rl_action = self.rl.select_action(state, agent_id=agent_id)
            rl_action = self.safety.clamp_action(rl_action)

            baseline_action = self._get_baseline_action(state)

            if self.cfg.shadow_mode:
                # Shadow mode: do not apply RL action, only record
                applied_action = baseline_action
            else:
                # Blend RL with baseline under safety
                if self.safety.check_state_safety(state):
                    applied_action = self.safety.blend_with_baseline(rl_action, baseline_action, alpha=0.2)
                else:
                    print("[Safety] Unsafe state detected, falling back to baseline.")
                    applied_action = baseline_action

            state["action"] = applied_action

            # Reward via RewardEngine
            reward = self.reward_engine.compute_reward(prev_state or state, state, applied_action, agent_id=agent_id)
            done = False  # could be set by mission logic

            if prev_state is not None and prev_action is not None:
                self.rl.store_transition({
                    "state": prev_state,
                    "action": prev_action,
                    "reward": reward,
                    "next_state": state,
                    "done": done,
                })

            prev_state = state
            prev_action = applied_action

            # Apply action to flight stack (or sim)
            if self.sim_bridge.enabled:
                # In sim mode, send actions to sim
                _ = self.sim_bridge.step({agent_id: applied_action})
            else:
                self.flight_stack.apply_action(applied_action, state)

            self.network.broadcast_state({
                "time": start,
                "action": applied_action,
                "status": self.get_status(),
            })

            cmd = self.network.get_control_command()
            if cmd:
                self.handle_toggle(cmd)

            self.last_state = state

            elapsed = time.perf_counter() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0)

        print("[Core] Main async loop stopped.")

    def stop(self):
        self.running = False
        self.video.stop()
        self.network.stop()
        self.plugin_manager.shutdown()
        self.memory.save()
        self.plugin_manager.save_settings()
        # Save RL models on shutdown
        self.rl.save_models(self.cfg.model_dir)
        print("[Core] Stopped and saved state.")


# =========================
# ENTRY POINT
# =========================

def main():
    core = RebootCore()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(core.start_async())
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt, shutting down...")
        core.stop()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
