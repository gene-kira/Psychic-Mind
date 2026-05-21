"""
UNIFIED VIDEO AI PIPELINE (RESILIENT, PREDICTIVE, PERSISTENT, CUDA-OPTIONAL)

Features:
- Autoloader with safe CUDA detection (GPU if available, CPU otherwise)
- YOLOv8 / RT-DETR-style detection backend with CUDA watchdog
- Camera + multi-camera fallback + webcam watchdog
- Screen capture fallback + "no video" mode + ghost frame predictor
- UIAutomation integration (Windows UI context)
- Keyboard action logger
- Predictive Bernoulli/Beta reasoning for missing details
- "Water" physics engine for smoothed object motion
- Self-improving RL loop with automatic "altered state" mode switching
- Reboot-persistent memory (local + SMB mirror)
"""

import importlib
import subprocess
import sys
import threading
import time
import json
import queue
import random
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ============================================================
# PERSISTENT MEMORY CONFIG (LOCAL + SMB MIRROR)
# ============================================================

MEMORY_LOCAL_PATH = os.path.abspath("reboot_memory.json")
# TODO: change this to your real SMB path
MEMORY_SMB_PATH = r"\\MY-NAS\AI-Memory\reboot_memory.json"

# ============================================================
# AUTOLOADER + SAFE CUDA DETECTION
# ============================================================

_autoload_lock = threading.Lock()

def load(module_name, package_name=None):
    with _autoload_lock:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            pkg = package_name if package_name else module_name
            print(f"[AUTOLOADER] Missing '{module_name}'. Installing '{pkg}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"[AUTOLOADER] Installed '{pkg}'. Retrying import...")
            return importlib.import_module(module_name)

def detect_cuda():
    """
    Safe CUDA detection.
    If CUDA is missing, broken, or half-installed → return CPU.
    Never crashes.
    """
    try:
        torch = load("torch")
    except Exception:
        print("[GPU] Torch not available. Using CPU.")
        return "cpu"

    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[GPU] CUDA available: {name}")
            return "cuda"
        else:
            print("[GPU] CUDA not available. Using CPU.")
            return "cpu"
    except Exception as e:
        print(f"[GPU] CUDA error: {e}. Using CPU.")
        return "cpu"

DEVICE = detect_cuda()

# ============================================================
# IMPORTS VIA AUTOLOADER
# ============================================================

cv2 = load("cv2", "opencv-python")
np = load("numpy")

ultralytics = None
try:
    ultralytics = load("ultralytics")
except Exception as e:
    print(f"[DETECTOR] Could not load ultralytics (YOLOv8): {e}")

dxcam = None
try:
    dxcam = load("dxcam")
except Exception as e:
    print(f"[SCREEN] dxcam not available: {e}")

mss = None
try:
    mss = load("mss")
except Exception as e:
    print(f"[SCREEN] mss not available: {e}")

pynput = None
try:
    pynput = load("pynput")
except Exception as e:
    print(f"[INPUT] pynput not available: {e}")

uiautomation = None
try:
    uiautomation = load("uiautomation")
except Exception as e:
    print(f"[UI] uiautomation not available: {e}")

try:
    gymnasium = load("gymnasium")
except Exception:
    gymnasium = None

# ============================================================
# RUNTIME CONFIG (PERSISTED)
# ============================================================

VIDEO_SOURCE_TYPE = "camera"  # "camera", "file", "screen"
VIDEO_SOURCE_PATH = 0         # 0 for default cam, or "video.mp4" for file

OUTPUT_TO_CONSOLE = True
SAVE_LOG = True
LOG_FILE = "video_state_log.jsonl"

USE_YOLO = True
USE_RL = True
USE_KEYBOARD_LOGGER = True

FPS_TARGET = 15.0

# RL mode: "normal" or "altered"
RL_MODE = "normal"

# ============================================================
# PERSISTENT MEMORY MANAGER
# ============================================================

class RebootMemoryManager:
    def __init__(self, local_path: str, smb_path: Optional[str] = None):
        self.local_path = local_path
        self.smb_path = smb_path

    def _ensure_dir(self, path: str):
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                print(f"[MEMORY] Could not create directory '{d}': {e}")

    def load(self) -> Optional[Dict[str, Any]]:
        for path in [self.local_path, self.smb_path]:
            if not path:
                continue
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    print(f"[MEMORY] Loaded state from {path}")
                    return data
            except Exception as e:
                print(f"[MEMORY] Failed to load from {path}: {e}")
        print("[MEMORY] No previous memory found.")
        return None

    def save(self, data: Dict[str, Any]):
        try:
            self._ensure_dir(self.local_path)
            with open(self.local_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"[MEMORY] Saved state to {self.local_path}")
        except Exception as e:
            print(f"[MEMORY] Failed to save to local {self.local_path}: {e}")

        if self.smb_path:
            try:
                self._ensure_dir(self.smb_path)
                with open(self.smb_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                print(f"[MEMORY] Mirrored state to {self.smb_path}")
            except Exception as e:
                print(f"[MEMORY] Failed to mirror to SMB {self.smb_path}: {e}")

memory_manager = RebootMemoryManager(MEMORY_LOCAL_PATH, MEMORY_SMB_PATH)
LOADED_MEMORY = memory_manager.load() or {}

def apply_loaded_config():
    global VIDEO_SOURCE_TYPE, VIDEO_SOURCE_PATH
    global OUTPUT_TO_CONSOLE, SAVE_LOG, LOG_FILE
    global USE_YOLO, USE_RL, USE_KEYBOARD_LOGGER, FPS_TARGET, RL_MODE

    cfg = LOADED_MEMORY.get("config", {})
    VIDEO_SOURCE_TYPE = cfg.get("VIDEO_SOURCE_TYPE", VIDEO_SOURCE_TYPE)
    VIDEO_SOURCE_PATH = cfg.get("VIDEO_SOURCE_PATH", VIDEO_SOURCE_PATH)
    OUTPUT_TO_CONSOLE = cfg.get("OUTPUT_TO_CONSOLE", OUTPUT_TO_CONSOLE)
    SAVE_LOG = cfg.get("SAVE_LOG", SAVE_LOG)
    LOG_FILE = cfg.get("LOG_FILE", LOG_FILE)
    USE_YOLO = cfg.get("USE_YOLO", USE_YOLO)
    USE_RL = cfg.get("USE_RL", USE_RL)
    USE_KEYBOARD_LOGGER = cfg.get("USE_KEYBOARD_LOGGER", USE_KEYBOARD_LOGGER)
    FPS_TARGET = cfg.get("FPS_TARGET", FPS_TARGET)
    RL_MODE = cfg.get("RL_MODE", RL_MODE)

apply_loaded_config()

# ============================================================
# DETECTION BACKENDS (YOLOv8 / RT-DETR STYLE) + GPU WATCHDOG
# ============================================================

class BaseDetector:
    def detect(self, frame) -> List[Dict[str, Any]]:
        raise NotImplementedError

class DummyDetector(BaseDetector):
    def detect(self, frame):
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        box_w, box_h = w // 4, h // 4
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2
        obj = {
            "id": 1,
            "class": "dummy_object",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "score": 0.9
        }
        return [obj]

class YOLOv8Detector(BaseDetector):
    def __init__(self, device="cpu"):
        if ultralytics is None:
            raise RuntimeError("Ultralytics not available")
        self.device = device
        self._build_model()

    def _build_model(self):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)
        print(f"[DETECTOR] YOLOv8 loaded on {self.device}")

    def _fallback_to_cpu(self):
        if self.device == "cuda":
            print("[GPU WATCHDOG] CUDA failure detected. Falling back to CPU.")
            self.device = "cpu"
            self._build_model()

    def detect(self, frame):
        try:
            results = self.model(frame, verbose=False)
            detections = []
            for r in results:
                for i, box in enumerate(r.boxes):
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    cls_name = self.model.names.get(cls_id, str(cls_id))
                    detections.append({
                        "id": i + 1,
                        "class": cls_name,
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "score": score
                    })
            return detections

        except Exception as e:
            msg = str(e)
            print(f"[DETECTOR] YOLO error: {msg}")

            if "CUDA" in msg or "cuda" in msg:
                self._fallback_to_cpu()
            return []

class RTDETRDetector(BaseDetector):
    def __init__(self, device="cpu"):
        self.device = device
        print(f"[DETECTOR] RT-DETR placeholder on {device}")

    def detect(self, frame):
        return []

def build_detector():
    if USE_YOLO and ultralytics is not None:
        try:
            return YOLOv8Detector(device=DEVICE)
        except Exception as e:
            print(f"[DETECTOR] YOLOv8 init failed: {e}. Falling back to DummyDetector.")
            return DummyDetector()
    else:
        print("[DETECTOR] YOLO disabled or unavailable. Using DummyDetector.")
        return DummyDetector()

# ============================================================
# SCREEN CAPTURE
# ============================================================

class ScreenCapture:
    def __init__(self):
        self.mode = None
        self.dx_cam = None
        self.mss_inst = None

        if dxcam is not None:
            try:
                self.dx_cam = dxcam.DXCamera()
                self.mode = "dxcam"
                print("[SCREEN] Using dxcam (DirectX).")
            except Exception as e:
                print(f"[SCREEN] dxcam init failed: {e}")

        if self.mode is None and mss is not None:
            try:
                self.mss_inst = mss.mss()
                self.mode = "mss"
                print("[SCREEN] Using mss (generic screen capture).")
            except Exception as e:
                print(f"[SCREEN] mss init failed: {e}")

        if self.mode is None:
            print("[SCREEN] No screen capture backend available.")

    def grab(self):
        if self.mode == "dxcam":
            frame = self.dx_cam.get_latest_frame()
            if frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return None
        elif self.mode == "mss":
            monitor = self.mss_inst.monitors[1]
            sct_img = self.mss_inst.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        else:
            return None

# ============================================================
# VIDEO MANAGER (MULTI-CAMERA + WATCHDOG + FALLBACK)
# ============================================================

class VideoManager:
    """
    Handles:
    - Multi-camera fallback (0,1,2,3)
    - Webcam watchdog (auto-reconnect)
    - Screen capture fallback
    - No-video mode
    """
    def __init__(self, preferred_source_type: str, preferred_path: Any):
        self.cam_indices = [0, 1, 2, 3]
        self.current_cam_index = None
        self.cap = None
        self.screen_cap = None
        self.mode = None  # "camera", "screen", "none"
        self.no_video = False

        self._init_video(preferred_source_type, preferred_path)

    def _try_open_camera(self, index: int):
        cam = cv2.VideoCapture(index)
        if cam.isOpened():
            print(f"[VIDEO] Camera {index} opened.")
            return cam
        cam.release()
        return None

    def _init_camera_chain(self):
        for idx in self.cam_indices:
            cam = self._try_open_camera(idx)
            if cam is not None:
                self.cap = cam
                self.current_cam_index = idx
                self.mode = "camera"
                return True
        return False

    def _init_screen(self):
        self.screen_cap = ScreenCapture()
        if self.screen_cap.mode is not None:
            self.mode = "screen"
            print("[VIDEO] Screen capture active.")
            return True
        return False

    def _init_video(self, preferred_source_type: str, preferred_path: Any):
        if self._init_camera_chain():
            return

        print("[VIDEO] No camera available. Trying screen capture...")
        if self._init_screen():
            return

        print("[VIDEO] No video sources available. Entering NO VIDEO MODE.")
        self.mode = "none"
        self.no_video = True

    def _switch_to_next_camera(self):
        print("[VIDEO] Switching to next camera...")
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        start_idx = self.cam_indices.index(self.current_cam_index) if self.current_cam_index in self.cam_indices else -1
        for offset in range(1, len(self.cam_indices) + 1):
            idx = self.cam_indices[(start_idx + offset) % len(self.cam_indices)]
            cam = self._try_open_camera(idx)
            if cam is not None:
                self.cap = cam
                self.current_cam_index = idx
                self.mode = "camera"
                print(f"[VIDEO] Switched to camera {idx}.")
                return True
        print("[VIDEO] No cameras available after switching. Trying screen capture...")
        if self._init_screen():
            return True
        print("[VIDEO] No video sources available. NO VIDEO MODE.")
        self.mode = "none"
        self.no_video = True
        return False

    def get_frame(self):
        """
        Returns (frame, is_real_frame)
        is_real_frame = False when using ghost frame.
        """
        if self.mode == "camera" and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame, True
            else:
                print("[VIDEO] Camera read failed. Webcam watchdog triggering...")
                if not self._switch_to_next_camera():
                    return None, False

        if self.mode == "screen" and self.screen_cap is not None:
            frame = self.screen_cap.grab()
            if frame is not None:
                return frame, True
            else:
                print("[VIDEO] Screen capture failed. NO VIDEO MODE.")
                self.mode = "none"
                self.no_video = True

        if self.mode == "none":
            return None, False

        return None, False

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None

# ============================================================
# UIAUTOMATION OBSERVER
# ============================================================

class UIAutomationObserver:
    def __init__(self):
        self.available = uiautomation is not None
        if self.available:
            print("[UI] UIAutomation enabled.")
        else:
            print("[UI] UIAutomation not available.")

    def snapshot(self) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            ctrl = uiautomation.GetForegroundControl()
            if ctrl is None:
                return None
            return {
                "name": ctrl.Name,
                "class_name": ctrl.ClassName,
                "control_type": str(ctrl.ControlType),
                "bounding_rectangle": list(ctrl.BoundingRectangle),
            }
        except Exception as e:
            print(f"[UI] UIAutomation snapshot error: {e}")
            return None

# ============================================================
# KEYBOARD LOGGER
# ============================================================

class KeyboardLogger:
    def __init__(self):
        self.events = queue.Queue()
        self.listener = None
        self.running = False

    def _on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        event = {"time": time.time(), "type": "key_down", "key": k}
        self.events.put(event)

    def _on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        event = {"time": time.time(), "type": "key_up", "key": k}
        self.events.put(event)

    def start(self):
        if pynput is None:
            print("[INPUT] pynput not available; keyboard logging disabled.")
            return
        from pynput import keyboard
        self.running = True
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
        print("[INPUT] Keyboard logger started.")

    def stop(self):
        self.running = False
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        print("[INPUT] Keyboard logger stopped.")

    def drain_events(self) -> List[Dict[str, Any]]:
        events = []
        while not self.events.empty():
            events.append(self.events.get())
        return events

class ControllerLogger:
    def __init__(self):
        pass

    def start(self):
        print("[INPUT] Controller logger stub (extend with gamepad library).")

    def stop(self):
        pass

    def drain_events(self):
        return []

# ============================================================
# PREDICTIVE REASONER (BERNOULLI / BETA)
# ============================================================

class PredictiveReasoner:
    def __init__(self):
        self.events = {
            "person_present": [1.0, 1.0],
            "vehicle_present": [1.0, 1.0],
            "enemy_present": [1.0, 1.0],
        }

    def update_from_objects(self, objects: List[Dict[str, Any]]):
        classes = [o.get("class", "") for o in objects]
        has_person = any("person" in c.lower() for c in classes)
        has_vehicle = any(c.lower() in ["car", "truck", "bus", "motorbike"] for c in classes)
        has_enemy = any(c.lower() in ["enemy", "opponent", "npc"] for c in classes)

        self._update_event("person_present", has_person)
        self._update_event("vehicle_present", has_vehicle)
        self._update_event("enemy_present", has_enemy)

    def _update_event(self, name: str, success: bool):
        if name not in self.events:
            self.events[name] = [1.0, 1.0]
        alpha, beta = self.events[name]
        if success:
            alpha += 1.0
        else:
            beta += 1.0
        self.events[name] = [alpha, beta]

    def get_probabilities(self) -> Dict[str, float]:
        probs = {}
        for name, (alpha, beta) in self.events.items():
            probs[name] = alpha / (alpha + beta)
        return probs

# ============================================================
# "WATER" PHYSICS ENGINE (SMOOTHED MOTION)
# ============================================================

class WaterPhysicsEngine:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.state = {}  # id -> {center, velocity}

    def update(self, objects: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        result = {}
        for obj in objects:
            oid = obj.get("id")
            x1, y1, x2, y2 = obj["bbox"]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            center = np.array([cx, cy], dtype=float)

            if oid in self.state:
                prev = self.state[oid]
                prev_center = prev["center"]
                velocity = center - prev_center
                new_center = self.alpha * center + (1 - self.alpha) * prev_center
            else:
                new_center = center
                velocity = np.array([0.0, 0.0])

            self.state[oid] = {
                "center": new_center,
                "velocity": velocity,
            }

            result[oid] = {
                "center": self.state[oid]["center"].tolist(),
                "velocity": self.state[oid]["velocity"].tolist(),
            }
        return result

# ============================================================
# RL LOOP SCAFFOLD + PERSISTENCE + ALTERED STATE
# ============================================================

@dataclass
class Transition:
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    done: bool

    def to_dict(self):
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Transition":
        return Transition(
            state=d["state"],
            action=d["action"],
            reward=d["reward"],
            next_state=d["next_state"],
            done=d["done"],
        )

@dataclass
class ReplayBuffer:
    capacity: int = 10000
    buffer: List[Transition] = field(default_factory=list)

    def add(self, transition: Transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def to_dict(self):
        return {
            "capacity": self.capacity,
            "buffer": [t.to_dict() for t in self.buffer],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ReplayBuffer":
        rb = ReplayBuffer(capacity=d.get("capacity", 10000))
        rb.buffer = [Transition.from_dict(t) for t in d.get("buffer", [])]
        return rb

class SimplePolicy:
    """
    Epsilon-greedy random policy.
    RL_MODE influences epsilon:
    - normal: lower epsilon
    - altered: higher epsilon (more exploratory)
    """
    def __init__(self, mode: str = "normal"):
        self.mode = mode

    def _epsilon(self) -> float:
        if self.mode == "altered":
            return 0.7
        return 0.3

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        eps = self._epsilon()
        if random.random() < eps:
            return {
                "throttle": random.uniform(-1, 1),
                "steer": random.uniform(-1, 1),
            }
        else:
            return {
                "throttle": random.uniform(0.2, 1.0),
                "steer": random.uniform(-0.2, 0.2),
            }

    def update(self, batch: List[Transition]):
        pass

    def to_dict(self):
        return {"mode": self.mode}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SimplePolicy":
        return SimplePolicy(mode=d.get("mode", "normal"))

class RLLearner:
    def __init__(self, saved_state: Optional[Dict[str, Any]] = None, mode: str = "normal"):
        if saved_state:
            self.buffer = ReplayBuffer.from_dict(saved_state.get("buffer", {}))
            self.policy = SimplePolicy.from_dict(saved_state.get("policy", {}))
            self.last_state = saved_state.get("last_state", None)
            self.last_action = saved_state.get("last_action", None)
            self.last_time = saved_state.get("last_time", None)
            print(f"[RL] Loaded RL state with {len(self.buffer.buffer)} transitions.")
        else:
            self.buffer = ReplayBuffer()
            self.policy = SimplePolicy(mode=mode)
            self.last_state = None
            self.last_action = None
            self.last_time = None

    def set_mode(self, mode: str):
        self.policy.mode = mode

    def on_new_state(self, state: Dict[str, Any], external_reward: Optional[float] = None):
        now = time.time()
        reward = external_reward if external_reward is not None else 0.0

        if self.last_state is not None and self.last_action is not None:
            transition = Transition(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=state,
                done=False,
            )
            self.buffer.add(transition)

        action = self.policy.act(state)
        self.last_state = state
        self.last_action = action
        self.last_time = now

        if len(self.buffer.buffer) >= 32:
            batch = self.buffer.sample(32)
            self.policy.update(batch)

        return action

    def to_dict(self):
        return {
            "buffer": self.buffer.to_dict(),
            "policy": self.policy.to_dict(),
            "last_state": self.last_state,
            "last_action": self.last_action,
            "last_time": self.last_time,
        }

# ============================================================
# LOGGER
# ============================================================

class StateLogger:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.file = open(filepath, "a", encoding="utf-8") if filepath else None
        self.lock = threading.Lock()

    def log(self, state):
        line = json.dumps(state)
        if self.file:
            with self.lock:
                self.file.write(line + "\n")
                self.file.flush()
        if OUTPUT_TO_CONSOLE:
            print(line)

    def close(self):
        if self.file:
            self.file.close()

# ============================================================
# GHOST FRAME PREDICTOR
# ============================================================

def make_ghost_frame(last_frame, physics_state: Dict[int, Dict[str, Any]]):
    """
    If no real frame arrives, create a "ghost" frame:
    - reuse last frame
    - overlay predicted centers from physics_state
    """
    if last_frame is None:
        return None
    frame = last_frame.copy()
    for oid, info in physics_state.items():
        cx, cy = info["center"]
        cv2.circle(frame, (int(cx), int(cy)), 10, (255, 0, 255), 2)
        cv2.putText(
            frame,
            f"ghost {oid}",
            (int(cx) + 5, int(cy) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 255),
            1,
        )
    cv2.putText(
        frame,
        "GHOST FRAME",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )
    return frame

# ============================================================
# MAIN LOOP
# ============================================================

def build_memory_snapshot(rl_learner: Optional[RLLearner]) -> Dict[str, Any]:
    cfg = {
        "VIDEO_SOURCE_TYPE": VIDEO_SOURCE_TYPE,
        "VIDEO_SOURCE_PATH": VIDEO_SOURCE_PATH,
        "OUTPUT_TO_CONSOLE": OUTPUT_TO_CONSOLE,
        "SAVE_LOG": SAVE_LOG,
        "LOG_FILE": LOG_FILE,
        "USE_YOLO": USE_YOLO,
        "USE_RL": USE_RL,
        "USE_KEYBOARD_LOGGER": USE_KEYBOARD_LOGGER,
        "FPS_TARGET": FPS_TARGET,
        "RL_MODE": RL_MODE,
    }
    rl_state = rl_learner.to_dict() if rl_learner is not None else None
    return {
        "config": cfg,
        "rl_state": rl_state,
        "timestamp": time.time(),
    }

def main():
    global RL_MODE

    saved_rl_state = LOADED_MEMORY.get("rl_state", None)

    video_manager = VideoManager(VIDEO_SOURCE_TYPE, VIDEO_SOURCE_PATH)
    detector = build_detector()
    logger = StateLogger(LOG_FILE if SAVE_LOG else None)

    kb_logger = KeyboardLogger() if USE_KEYBOARD_LOGGER else None
    if kb_logger is not None:
        kb_logger.start()

    rl_learner = RLLearner(saved_state=saved_rl_state, mode=RL_MODE) if USE_RL else None
    ui_observer = UIAutomationObserver()
    predictor = PredictiveReasoner()
    water_engine = WaterPhysicsEngine()

    frame_interval = 1.0 / FPS_TARGET
    last_time = time.time()
    last_objects = []
    last_frame = None
    consecutive_ghost_frames = 0

    try:
        while True:
            frame, is_real_frame = video_manager.get_frame()

            if frame is None:
                physics_state = water_engine.update(last_objects)
                frame = make_ghost_frame(last_frame, physics_state)
                is_real_frame = False
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                consecutive_ghost_frames += 1
            else:
                consecutive_ghost_frames = 0

            if is_real_frame:
                last_frame = frame.copy()

            now = time.time()
            timestamp = now

            try:
                objects = detector.detect(frame) if is_real_frame else last_objects
            except Exception as e:
                print(f"[MAIN] Detector error: {e}")
                objects = last_objects

            if not objects and last_objects:
                objects = last_objects
            last_objects = objects

            input_events = []
            if kb_logger is not None:
                input_events.extend(kb_logger.drain_events())

            ui_context = ui_observer.snapshot()
            predictor.update_from_objects(objects)
            pred_probs = predictor.get_probabilities()
            physics_state = water_engine.update(objects)

            # Automatic altered state switching
            if consecutive_ghost_frames > 10:
                RL_MODE = "altered"
            else:
                RL_MODE = "normal"

            if rl_learner is not None:
                rl_learner.set_mode(RL_MODE)

            state = {
                "timestamp": timestamp,
                "objects": objects,
                "objects_physics": physics_state,
                "input_events": input_events,
                "source_type": VIDEO_SOURCE_TYPE,
                "ui_context": ui_context,
                "predictions": pred_probs,
                "is_real_frame": is_real_frame,
                "rl_mode": RL_MODE,
            }

            control_suggestion = None
            if rl_learner is not None:
                action = rl_learner.on_new_state(state)
                control_suggestion = action
                state["control_suggestion"] = action

            logger.log(state)

            for obj in objects:
                x1, y1, x2, y2 = obj["bbox"]
                cls = obj["class"]
                score = obj["score"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{cls} {score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            if control_suggestion is not None:
                txt = f"throttle={control_suggestion['throttle']:.2f}, steer={control_suggestion['steer']:.2f}"
                cv2.putText(
                    frame,
                    txt,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            if ui_context is not None:
                title = ui_context.get("name", "")
                cv2.putText(
                    frame,
                    f"UI: {title[:40]}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

            cv2.putText(
                frame,
                f"MODE: {RL_MODE.upper()}  REAL:{is_real_frame}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )

            cv2.imshow("Unified Video AI", frame)

            elapsed = time.time() - last_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video_manager.release()
        cv2.destroyAllWindows()
        logger.close()
        if kb_logger is not None:
            kb_logger.stop()

        snapshot = build_memory_snapshot(rl_learner)
        memory_manager.save(snapshot)

if __name__ == "__main__":
    main()
