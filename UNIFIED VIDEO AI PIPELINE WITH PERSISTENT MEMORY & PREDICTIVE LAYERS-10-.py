"""
REBOOTCORE UNIFIED AI RUNTIME + DEARPYGUI FRONT-END

Modes:
    python rebootcore_unified.py
        -> Core only (video + HUD + API)

    python rebootcore_unified.py --gui
        -> DearPyGUI control console only (expects core running on port 8000)

    python rebootcore_unified.py --core-and-gui
        -> Core + DearPyGUI in one process (core in background thread)
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
import socket
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# ============================================================
# GLOBAL MEMORY CONFIG
# ============================================================

REBOOTCORE_FOLDER_NAME = "RebootCore"
MEMORY_FILE_NAME = "reboot_memory.json"

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
# IMPORTS VIA AUTOLOADER / STANDARD
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

speech_recognition = None
try:
    speech_recognition = load("speech_recognition", "SpeechRecognition")
except Exception as e:
    print(f"[VOICE] SpeechRecognition not available: {e}")

pymavlink = None
try:
    pymavlink = load("pymavlink")
except Exception as e:
    print(f"[MAVLINK] pymavlink not available: {e}")

python_can = None
try:
    python_can = load("can", "python-can")
except Exception as e:
    print(f"[CAN] python-can not available: {e}")

try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False
    print("[GUI] tkinter not available; Browse/SMB dialogs will be limited.")

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# DearPyGUI + requests for front-end
requests = load("requests")
dpg = load("dearpygui.dearpygui")

# ============================================================
# RUNTIME CONFIG (PERSISTED)
# ============================================================

VIDEO_SOURCE_TYPE = "camera"
VIDEO_SOURCE_PATH = 0

OUTPUT_TO_CONSOLE = True
SAVE_LOG = True
LOG_FILE = "video_state_log.jsonl"

USE_YOLO = True
USE_RL = True
USE_KEYBOARD_LOGGER = True

FPS_TARGET = 20.0
RL_MODE = "normal"

ENABLE_YOLO_RUNTIME = True
ENABLE_RL_RUNTIME = True
ALTERED_MANUAL_OVERRIDE = False

MEMORY_BASE_DIR = os.path.join("C:\\", REBOOTCORE_FOLDER_NAME)
CURRENT_SAVE_PATH_DISPLAY = "C:\\" + REBOOTCORE_FOLDER_NAME + "\\"

NETWORK_BROADCAST_PORT = 50555
NETWORK_LISTEN_PORT = 50556
NETWORK_BROADCAST_ENABLED = True

REMOTE_API_PORT = 8000

GUI_MODE = "opencv"  # placeholder for future Qt/DearPyGUI integration

# ============================================================
# PERSISTENT MEMORY MANAGER
# ============================================================

class RebootMemoryManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.file_name = MEMORY_FILE_NAME

    @property
    def local_path(self) -> str:
        return os.path.join(self.base_dir, self.file_name)

    def set_base_dir(self, base_dir: str):
        self.base_dir = base_dir
        print(f"[MEMORY] Base dir set to {self.base_dir}")

    def load(self) -> Optional[Dict[str, Any]]:
        path = self.local_path
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
        path = self.local_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"[MEMORY] Saved state to {path}")
        except Exception as e:
            print(f"[MEMORY] Failed to save to {path}: {e}")

def default_base_dir() -> str:
    return os.path.join("C:\\", REBOOTCORE_FOLDER_NAME)

MEMORY_BASE_DIR = default_base_dir()
memory_manager = RebootMemoryManager(MEMORY_BASE_DIR)
LOADED_MEMORY = memory_manager.load() or {}

def apply_loaded_config():
    global VIDEO_SOURCE_TYPE, VIDEO_SOURCE_PATH
    global OUTPUT_TO_CONSOLE, SAVE_LOG, LOG_FILE
    global USE_YOLO, USE_RL, USE_KEYBOARD_LOGGER, FPS_TARGET, RL_MODE
    global ENABLE_YOLO_RUNTIME, ENABLE_RL_RUNTIME, ALTERED_MANUAL_OVERRIDE
    global MEMORY_BASE_DIR, CURRENT_SAVE_PATH_DISPLAY
    global NETWORK_BROADCAST_ENABLED, GUI_MODE

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
    ENABLE_YOLO_RUNTIME = cfg.get("ENABLE_YOLO_RUNTIME", True)
    ENABLE_RL_RUNTIME = cfg.get("ENABLE_RL_RUNTIME", True)
    ALTERED_MANUAL_OVERRIDE = cfg.get("ALTERED_MANUAL_OVERRIDE", False)
    NETWORK_BROADCAST_ENABLED = cfg.get("NETWORK_BROADCAST_ENABLED", NETWORK_BROADCAST_ENABLED)
    GUI_MODE = cfg.get("GUI_MODE", GUI_MODE)

    MEMORY_BASE_DIR = cfg.get("MEMORY_BASE_DIR", MEMORY_BASE_DIR)
    memory_manager.set_base_dir(MEMORY_BASE_DIR)

    drive = os.path.splitdrive(MEMORY_BASE_DIR)[0] or "C:"
    CURRENT_SAVE_PATH_DISPLAY = f"{drive}\\{REBOOTCORE_FOLDER_NAME}\\"

apply_loaded_config()

# ============================================================
# DETECTION BACKENDS + GPU WATCHDOG
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
        return [{
            "id": -1,
            "class": "dummy_object",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "score": 0.9
        }]

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
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    cls_name = self.model.names.get(cls_id, str(cls_id))
                    detections.append({
                        "id": -1,
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
            dev = "cuda" if DEVICE == "cuda" else "cpu"
            return YOLOv8Detector(device=dev)
        except Exception as e:
            print(f"[DETECTOR] YOLOv8 init failed: {e}. Falling back to DummyDetector.")
            return DummyDetector()
    else:
        print("[DETECTOR] YOLO disabled or unavailable. Using DummyDetector.")
        return DummyDetector()

# ============================================================
# SIMPLE OBJECT TRACKER (IOU-BASED)
# ============================================================

class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.next_id = 1
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea <= 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        for det in detections:
            best_iou = 0.0
            best_id = None
            for tid, t in self.tracks.items():
                iou_val = self.iou(det["bbox"], t["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = tid
            if best_iou >= self.iou_threshold and best_id is not None:
                self.tracks[best_id]["bbox"] = det["bbox"]
                self.tracks[best_id]["age"] = 0
                det["id"] = best_id
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": det["bbox"], "age": 0}
                det["id"] = tid

        return detections

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
# VIDEO MANAGER WITH THREADED CAPTURE
# ============================================================

class VideoManager:
    def __init__(self, preferred_source_type: str, preferred_path: Any):
        self.cam_indices = [0, 1, 2, 3]
        self.current_cam_index = None
        self.cap = None
        self.screen_cap = None
        self.mode = None
        self.no_video = False

        self.frame_queue = queue.Queue(maxsize=3)
        self.capture_thread = None
        self.capture_running = False

        self._init_video(preferred_source_type, preferred_path)

    def _try_open_camera(self, index: int):
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
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
            self._start_capture_thread()
            return
        print("[VIDEO] No camera available. Trying screen capture...")
        if self._init_screen():
            self._start_capture_thread()
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
                self._start_capture_thread()
                return True
        print("[VIDEO] No cameras available after switching. Trying screen capture...")
        if self._init_screen():
            self._start_capture_thread()
            return True
        print("[VIDEO] No video sources available. NO VIDEO MODE.")
        self.mode = "none"
        self.no_video = True
        return False

    def _capture_loop(self):
        while self.capture_running:
            frame = None
            if self.mode == "camera" and self.cap is not None:
                ret, f = self.cap.read()
                if ret and f is not None:
                    frame = f
                else:
                    print("[VIDEO] Camera read failed in capture thread.")
                    if not self._switch_to_next_camera():
                        break
            elif self.mode == "screen" and self.screen_cap is not None:
                frame = self.screen_cap.grab()
                if frame is None:
                    print("[VIDEO] Screen capture failed in capture thread.")
                    self.mode = "none"
                    self.no_video = True
                    break
            else:
                break

            if frame is not None:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            time.sleep(0.001)

    def _start_capture_thread(self):
        if self.capture_thread is not None:
            return
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("[VIDEO] Capture thread started.")

    def get_frame(self):
        if self.no_video or self.mode == "none":
            return None, False
        try:
            frame = self.frame_queue.get(timeout=0.05)
            return frame, True
        except queue.Empty:
            return None, False

    def release(self):
        self.capture_running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)
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
        self.events.put({"time": time.time(), "type": "key_down", "key": k})

    def _on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        self.events.put({"time": time.time(), "type": "key_up", "key": k})

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
        print("[INPUT] Controller logger stub.")
    def stop(self):
        pass
    def drain_events(self):
        return []

# ============================================================
# PREDICTIVE REASONER
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
# "WATER" PHYSICS ENGINE
# ============================================================

class WaterPhysicsEngine:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.state = {}

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
            self.state[oid] = {"center": new_center, "velocity": velocity}
            result[oid] = {
                "center": self.state[oid]["center"].tolist(),
                "velocity": self.state[oid]["velocity"].tolist(),
            }
        return result

# ============================================================
# RL LOOP + AUTOPILOT MODEL INTERFACE
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
    capacity: int = 20000
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
        rb = ReplayBuffer(capacity=d.get("capacity", 20000))
        rb.buffer = [Transition.from_dict(t) for t in d.get("buffer", [])]
        return rb

class AutopilotModelInterface:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    def load_model(self, path: str):
        print(f"[AUTOPILOT] Placeholder load model from {path}")

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        objs = state.get("objects", [])
        throttle = 0.3
        steer = 0.0
        if objs:
            o = objs[0]
            x1, y1, x2, y2 = o["bbox"]
            cx = 0.5 * (x1 + x2)
            steer = (cx - 320.0) / 320.0
        throttle += random.uniform(-0.1, 0.1)
        steer += random.uniform(-0.05, 0.05)
        return {"throttle": float(max(-1.0, min(1.0, throttle))),
                "steer": float(max(-1.0, min(1.0, steer)))}

class SimplePolicy:
    def __init__(self, mode: str = "normal", autopilot: Optional[AutopilotModelInterface] = None):
        self.mode = mode
        self.autopilot = autopilot or AutopilotModelInterface(device=DEVICE)

    def _epsilon(self) -> float:
        return 0.7 if self.mode == "altered" else 0.3

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        eps = self._epsilon()
        if random.random() < eps:
            return {"throttle": random.uniform(-1, 1), "steer": random.uniform(-1, 1)}
        else:
            return self.autopilot.act(state)

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
        if len(self.buffer.buffer) >= 64:
            batch = self.buffer.sample(64)
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
            0.5,
            (255, 0, 255),
            2,
        )
    cv2.putText(
        frame,
        "GHOST FRAME",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
    )
    return frame

# ============================================================
# DRIVE DETECTION + FOLDER PICKER + SMB PATH
# ============================================================

def detect_drives() -> List[str]:
    drives = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if os.path.exists(root):
            drives.append(root)
    return drives

def browse_for_folder() -> Optional[str]:
    if not TK_AVAILABLE:
        print("[GUI] tkinter not available; cannot browse.")
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        root.after(1, lambda: root.focus_force())
        folder = filedialog.askdirectory(title="Select base folder for RebootCore", parent=root)
        root.destroy()
        if folder:
            return folder
    except Exception as e:
        print(f"[GUI] Folder browse error: {e}")
    return None

def prompt_smb_path(blocking_console_fallback: bool = True) -> Optional[str]:
    if TK_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes("-topmost", True)
            root.after(1, lambda: root.focus_force())
            path = simpledialog.askstring("SMB Network Path", "Enter SMB UNC path (e.g. \\\\SERVER\\Share\\Folder):", parent=root)
            root.destroy()
            if path:
                return path.strip()
        except Exception as e:
            print(f"[GUI] SMB path dialog error: {e}")

    if blocking_console_fallback:
        try:
            path = input("Enter SMB UNC path (e.g. \\\\SERVER\\Share\\Folder): ").strip()
            if path:
                return path
        except Exception:
            pass
    return None

def set_memory_base_from_drive_or_folder(selection: str, is_drive: bool):
    global MEMORY_BASE_DIR, CURRENT_SAVE_PATH_DISPLAY

    selection = os.path.normpath(selection)

    if selection.lower().endswith(REBOOTCORE_FOLDER_NAME.lower()):
        base_dir = selection
    else:
        base_dir = os.path.join(selection, REBOOTCORE_FOLDER_NAME)

    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception as e:
        print(f"[MEMORY] Failed to create base directory '{base_dir}': {e}")

    MEMORY_BASE_DIR = base_dir
    memory_manager.set_base_dir(MEMORY_BASE_DIR)

    drive = os.path.splitdrive(MEMORY_BASE_DIR)[0] or selection
    if drive.endswith("\\"):
        drive = drive[:-1]
    CURRENT_SAVE_PATH_DISPLAY = f"{drive}\\{REBOOTCORE_FOLDER_NAME}\\"

    print(f"[HUD] Memory base dir set to {MEMORY_BASE_DIR}")

# ============================================================
# STARTUP SAVE-LOCATION PROMPT
# ============================================================

def choose_initial_save_location():
    global LOADED_MEMORY

    print("\n=== RebootCore Save Location Setup ===")
    print("Select where to store reboot_memory.json\n")

    drives = detect_drives()
    for i, d in enumerate(drives):
        print(f"{i+1}. Drive: {d}")

    idx_offset = len(drives)
    smb_idx = idx_offset + 1
    browse_idx = idx_offset + 2
    keep_idx = idx_offset + 3

    print(f"{smb_idx}. SMB Network Path (anonymous)")
    print(f"{browse_idx}. Browse Local Folder")
    print(f"{keep_idx}. Keep current: {MEMORY_BASE_DIR}")
    print("")

    choice = None
    try:
        raw = input("Enter choice number: ").strip()
        if raw:
            choice = int(raw)
    except Exception:
        choice = None

    if choice is None:
        print("[SETUP] Invalid input. Keeping current location.")
        return

    if 1 <= choice <= len(drives):
        drive = drives[choice - 1]
        set_memory_base_from_drive_or_folder(drive, is_drive=True)
    elif choice == smb_idx:
        path = prompt_smb_path(blocking_console_fallback=True)
        if path:
            if os.path.exists(path):
                set_memory_base_from_drive_or_folder(path, is_drive=False)
            else:
                print(f"[SETUP] SMB path not reachable: {path}. Keeping current.")
        else:
            print("[SETUP] No SMB path entered. Keeping current.")
    elif choice == browse_idx:
        folder = browse_for_folder()
        if folder:
            set_memory_base_from_drive_or_folder(folder, is_drive=False)
        else:
            print("[SETUP] No folder selected. Keeping current.")
    elif choice == keep_idx:
        print("[SETUP] Keeping current location.")
    else:
        print("[SETUP] Invalid choice. Keeping current location.")

    new_mem = memory_manager.load()
    if new_mem is not None:
        LOADED_MEMORY = new_mem
        apply_loaded_config()
        print("[SETUP] Loaded memory from new location.")
    else:
        print("[SETUP] No existing memory at new location; starting fresh.")

# ============================================================
# CENTERED DRIVE POPUP (DARK THEME, GREEN HIGHLIGHT)
# ============================================================

class DrivePopup:
    def __init__(self):
        self.active = False
        self.rect = None
        self.item_rects: List[Tuple[int, int, int, int]] = []
        self.items: List[str] = []
        self.hover_index: Optional[int] = None

    def open(self, frame_shape):
        h, w, _ = frame_shape
        popup_w, popup_h = 400, 300
        x1 = (w - popup_w) // 2
        y1 = (h - popup_h) // 2
        x2 = x1 + popup_w
        y2 = y1 + popup_h
        self.rect = (x1, y1, x2, y2)

        drives = detect_drives()
        self.items = drives + ["SMB Network Path...", "Browse Local Folder..."]
        self.item_rects = []
        self.hover_index = None
        self.active = True
        print(f"[POPUP] Opened with items: {self.items}")

    def close(self):
        self.active = False
        self.rect = None
        self.item_rects = []
        self.items = []
        self.hover_index = None
        print("[POPUP] Closed.")

    def draw(self, frame):
        if not self.active or self.rect is None:
            return

        h, w, _ = frame.shape

        dim_overlay = frame.copy()
        cv2.rectangle(dim_overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(dim_overlay, 0.4, frame, 0.6, 0, frame)

        x1, y1, x2, y2 = self.rect

        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)

        title = "Select Save Location"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + 30
        cv2.putText(frame, title, (tx+1, ty+1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.line(frame, (x1 + 10, y1 + 40), (x2 - 10, y1 + 40), (80, 80, 80), 1)

        item_area_top = y1 + 55
        item_height = 32
        item_spacing = 6

        self.item_rects = []
        for i, item in enumerate(self.items):
            iy1 = item_area_top + i * (item_height + item_spacing)
            iy2 = iy1 + item_height
            if iy2 > y2 - 10:
                break

            rect = (x1 + 15, iy1, x2 - 15, iy2)
            self.item_rects.append(rect)

            rx1, ry1, rx2, ry2 = rect

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (45, 45, 45), -1)

            if self.hover_index == i:
                border_color = (0, 255, 0)
                text_color = (0, 255, 0)
            else:
                border_color = (160, 160, 160)
                text_color = (255, 255, 255)

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), border_color, 2)

            label = item
            (ltw, lth), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx = rx1 + 10
            ly = ry1 + (ry2 - ry1 + lth) // 2

            cv2.putText(frame, label, (lx+1, ly+1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

    def handle_click(self, x: int, y: int) -> bool:
        if not self.active or self.rect is None:
            return False

        x1, y1, x2, y2 = self.rect

        if not (x1 <= x <= x2 and y1 <= y <= y2):
            self.close()
            return True

        for idx, rect in enumerate(self.item_rects):
            rx1, ry1, rx2, ry2 = rect
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                item = self.items[idx]
                print(f"[POPUP] Selected item: {item}")

                if item == "SMB Network Path...":
                    path = prompt_smb_path(blocking_console_fallback=False)
                    if path:
                        if os.path.exists(path):
                            set_memory_base_from_drive_or_folder(path, is_drive=False)
                        else:
                            print(f"[POPUP] SMB path not reachable: {path}")

                elif item == "Browse Local Folder...":
                    folder = browse_for_folder()
                    if folder:
                        set_memory_base_from_drive_or_folder(folder, is_drive=False)

                else:
                    set_memory_base_from_drive_or_folder(item, is_drive=True)

                self.close()
                return True

        return True

    def handle_move(self, x: int, y: int):
        if not self.active or self.rect is None:
            return
        self.hover_index = None
        for idx, rect in enumerate(self.item_rects):
            rx1, ry1, rx2, ry2 = rect
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                self.hover_index = idx
                break

DRIVE_POPUP = DrivePopup()

# ============================================================
# GPU-ACCELERATED BLENDING (HUD)
# ============================================================

def gpu_blend(base_frame, overlay_frame, alpha=0.9):
    try:
        if DEVICE != "cuda":
            return None
        if not hasattr(cv2, "cuda"):
            return None
        g_base = cv2.cuda_GpuMat()
        g_overlay = cv2.cuda_GpuMat()
        g_base.upload(base_frame)
        g_overlay.upload(overlay_frame)
        g_out = cv2.cuda.addWeighted(g_overlay, alpha, g_base, 1.0 - alpha, 0)
        out = g_out.download()
        return out
    except Exception as e:
        print(f"[GPU HUD] CUDA blend failed: {e}")
        return None

# ============================================================
# PLUGIN SETTINGS PERSISTENCE
# ============================================================

class PluginSettingsStore:
    def __init__(self):
        self.settings: Dict[str, Dict[str, Any]] = LOADED_MEMORY.get("plugin_settings", {})

    def get(self, plugin_id: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.settings.get(plugin_id, default or {})

    def set(self, plugin_id: str, cfg: Dict[str, Any]):
        self.settings[plugin_id] = cfg

    def to_dict(self):
        return self.settings

PLUGIN_SETTINGS = PluginSettingsStore()

# ============================================================
# OVERLAY HUD
# ============================================================

@dataclass
class HUDButton:
    label: str
    rect: Tuple[int, int, int, int]
    active: bool = False

class ConfigHUD:
    def __init__(self):
        self.buttons: Dict[str, HUDButton] = {}
        self.last_frame_size = (0, 0)
        self.save_rect: Optional[Tuple[int, int, int, int]] = None
        self.panel_rect: Optional[Tuple[int, int, int, int]] = None

    def layout(self, frame):
        h, w, _ = frame.shape
        self.last_frame_size = (w, h)

        panel_x1 = w - 280
        panel_y1 = h - 260
        panel_x2 = w - 10
        panel_y2 = h - 10
        self.panel_rect = (panel_x1, panel_y1, panel_x2, panel_y2)

        base_x = panel_x1 + 20
        base_y = panel_y1 + 20
        bw = 140
        bh = 32
        gap = 8

        self.buttons["yolo"] = HUDButton("YOLO", (base_x, base_y, base_x + bw, base_y + bh))
        self.buttons["rl"] = HUDButton("RL", (base_x, base_y + (bh + gap), base_x + bw, base_y + (bh + gap) + bh))
        self.buttons["altered"] = HUDButton("ALTERED", (base_x, base_y + 2*(bh + gap), base_x + bw, base_y + 2*(bh + gap) + bh))
        self.buttons["fps_up"] = HUDButton("FPS+", (base_x, base_y + 3*(bh + gap), base_x + bw//2 - 4, base_y + 3*(bh + gap) + bh))
        self.buttons["fps_down"] = HUDButton("FPS-", (base_x + bw//2 + 4, base_y + 3*(bh + gap), base_x + bw, base_y + 3*(bh + gap) + bh))

        sv_x1 = base_x - 10
        sv_y1 = base_y + 4*(bh + gap)
        sv_x2 = base_x + bw + 10
        sv_y2 = sv_y1 + bh + 6
        self.save_rect = (sv_x1, sv_y1, sv_x2, sv_y2)

    def _ensure_layout(self, frame):
        if self.last_frame_size != (frame.shape[1], frame.shape[0]) or not self.buttons or self.save_rect is None:
            self.layout(frame)

    def draw(self, frame, yolo_on: bool, rl_on: bool, altered: bool, fps: float,
             ghost_count: int, save_path_display: str,
             swarm_state: Optional[List[Dict[str, Any]]] = None):
        self._ensure_layout(frame)
        overlay = frame.copy()

        if self.panel_rect is not None:
            px1, py1, px2, py2 = self.panel_rect
            cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 0, 0), -1)
            blended = gpu_blend(frame, overlay, alpha=0.55)
            if blended is not None:
                frame[:] = blended
            else:
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            overlay = frame.copy()

        def draw_button(btn: HUDButton, active: bool):
            x1, y1, x2, y2 = btn.rect
            color_bg = (60, 60, 60)
            color_border = (0, 255, 0) if active else (200, 200, 200)
            text_color = (255, 255, 255)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bg, -1)
            cv2.rectangle(overlay, (x1-1, y1-1), (x2+1, y2+1), (0, 255, 0) if active else (120, 120, 120), 1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_border, 2)

            text = btn.label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2

            cv2.putText(overlay, text, (tx+1, ty+1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(overlay, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

        draw_button(self.buttons["yolo"], yolo_on)
        draw_button(self.buttons["rl"], rl_on)
        draw_button(self.buttons["altered"], altered)
        draw_button(self.buttons["fps_up"], False)
        draw_button(self.buttons["fps_down"], False)

        if self.save_rect is not None:
            x1, y1, x2, y2 = self.save_rect
            text_color = (255, 255, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 60), -1)
            cv2.rectangle(overlay, (x1-1, y1-1), (x2+1, y2+1), (200, 200, 200), 1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 200), 2)

            text = "Save: " + save_path_display
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            tx = x1 + 6
            ty = y1 + (y2 - y1 + th) // 2

            cv2.putText(overlay, text, (tx+1, ty+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(overlay, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            cv2.putText(overlay, "v", (x2 - 18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        if swarm_state:
            mh, mw = 120, 120
            x0, y0 = 10, frame.shape[0] - mh - 10
            cv2.rectangle(overlay, (x0, y0), (x0+mw, y0+mh), (20, 20, 20), -1)
            cv2.rectangle(overlay, (x0, y0), (x0+mw, y0+mh), (200, 200, 200), 1)
            cv2.putText(overlay, "SWARM", (x0+5, y0+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            for d in swarm_state:
                sx = int(x0 + mw/2 + d["x"]*2)
                sy = int(y0 + mh/2 - d["y"]*2)
                cv2.circle(overlay, (sx, sy), 3, (0,255,255), -1)

        blended2 = gpu_blend(frame, overlay, alpha=0.9)
        if blended2 is not None:
            frame[:] = blended2
        else:
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        if self.panel_rect is not None:
            px1, py1, px2, py2 = self.panel_rect
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 1)

        info_lines = [
            f"YOLO: {'ON' if yolo_on else 'OFF'}",
            f"RL: {'ON' if rl_on else 'OFF'}",
            f"MODE: {'ALTERED' if altered else 'NORMAL'}",
            f"FPS: {fps:.1f}",
            f"GhostFrames: {ghost_count}",
            "Save button: change drive/SMB/folder.",
        ]
        x0, y0 = 10, 20
        for i, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (x0+1, y0 + i * 20 + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                frame,
                line,
                (x0, y0 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

    def handle_click(self, x: int, y: int, frame_shape):
        global ENABLE_YOLO_RUNTIME, ENABLE_RL_RUNTIME, ALTERED_MANUAL_OVERRIDE, RL_MODE, FPS_TARGET

        if DRIVE_POPUP.active:
            return

        for key, btn in self.buttons.items():
            x1, y1, x2, y2 = btn.rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key == "yolo":
                    ENABLE_YOLO_RUNTIME = not ENABLE_YOLO_RUNTIME
                    print(f"[HUD] YOLO runtime toggled to {ENABLE_YOLO_RUNTIME}")
                elif key == "rl":
                    ENABLE_RL_RUNTIME = not ENABLE_RL_RUNTIME
                    print(f"[HUD] RL runtime toggled to {ENABLE_RL_RUNTIME}")
                elif key == "altered":
                    ALTERED_MANUAL_OVERRIDE = not ALTERED_MANUAL_OVERRIDE
                    RL_MODE = "altered" if ALTERED_MANUAL_OVERRIDE else "normal"
                    print(f"[HUD] Altered manual override: {ALTERED_MANUAL_OVERRIDE}, RL_MODE={RL_MODE}")
                elif key == "fps_up":
                    FPS_TARGET = min(60.0, FPS_TARGET + 1.0)
                    print(f"[HUD] FPS_TARGET increased to {FPS_TARGET}")
                elif key == "fps_down":
                    FPS_TARGET = max(1.0, FPS_TARGET - 1.0)
                    print(f"[HUD] FPS_TARGET decreased to {FPS_TARGET}")
                return

        if self.save_rect is not None:
            x1, y1, x2, y2 = self.save_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                DRIVE_POPUP.open(frame_shape)
                return

HUD = ConfigHUD()

# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    frame_shape = param["frame_shape"]
    if event == cv2.EVENT_LBUTTONDOWN:
        if DRIVE_POPUP.active:
            consumed = DRIVE_POPUP.handle_click(x, y)
            if consumed:
                return
        HUD.handle_click(x, y, frame_shape)
    elif event == cv2.EVENT_MOUSEMOVE:
        if DRIVE_POPUP.active:
            DRIVE_POPUP.handle_move(x, y)

# ============================================================
# MEMORY SNAPSHOT
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
        "ENABLE_YOLO_RUNTIME": ENABLE_YOLO_RUNTIME,
        "ENABLE_RL_RUNTIME": ENABLE_RL_RUNTIME,
        "ALTERED_MANUAL_OVERRIDE": ALTERED_MANUAL_OVERRIDE,
        "MEMORY_BASE_DIR": MEMORY_BASE_DIR,
        "NETWORK_BROADCAST_ENABLED": NETWORK_BROADCAST_ENABLED,
        "GUI_MODE": GUI_MODE,
    }
    rl_state = rl_learner.to_dict() if rl_learner is not None else None
    return {
        "config": cfg,
        "rl_state": rl_state,
        "plugin_settings": PLUGIN_SETTINGS.to_dict(),
        "timestamp": time.time()
    }

# ============================================================
# MULTI-NODE NETWORKING (UDP)
# ============================================================

class NetworkBroadcaster:
    def __init__(self, port: int):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.enabled = True

    def send_state(self, state: Dict[str, Any]):
        if not self.enabled:
            return
        try:
            data = json.dumps({"type": "state", "payload": state}).encode("utf-8")
            self.sock.sendto(data, ("<broadcast>", self.port))
        except Exception as e:
            print(f"[NET] Broadcast error: {e}")

class NetworkListener:
    def __init__(self, port: int, control_queue: queue.Queue):
        self.port = port
        self.control_queue = control_queue
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", self.port))
        self.running = False
        self.thread = None

    def _loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
                msg = json.loads(data.decode("utf-8"))
                if msg.get("type") == "control":
                    self.control_queue.put(msg.get("payload", {}))
            except Exception:
                pass

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[NET] Listener started on port {self.port}")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.sock.close()

# ============================================================
# REMOTE CONTROL API (HTTP + MARKETPLACE)
# ============================================================

GLOBAL_STATUS = {
    "yolo": True,
    "rl": True,
    "mode": "normal",
    "fps": FPS_TARGET,
}

REMOTE_CONTROL_QUEUE = queue.Queue()

class PluginMarketplace:
    def __init__(self):
        self.catalog = [
            {
                "id": "drone_autopilot",
                "name": "Drone Autopilot",
                "description": "Simulated drone control + MAVLink hooks.",
                "version": "1.0",
                "author": "RebootCore",
            },
            {
                "id": "car_canbus",
                "name": "Car CAN-Bus Bridge",
                "description": "Simulated CAN bridge + OBD-II hooks.",
                "version": "1.0",
                "author": "RebootCore",
            },
            {
                "id": "swarm_manager",
                "name": "Multi-Drone Swarm Manager",
                "description": "Coordinates multiple virtual drones.",
                "version": "1.0",
                "author": "RebootCore",
            },
            {
                "id": "vehicle_autopilot",
                "name": "Vehicle Autopilot",
                "description": "High-level lane/target following logic.",
                "version": "1.0",
                "author": "RebootCore",
            },
        ]

    def list_plugins(self):
        return {"plugins": self.catalog}

PLUGIN_MARKETPLACE = PluginMarketplace()

class RemoteAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            self._send_json(GLOBAL_STATUS)
        elif parsed.path == "/toggle":
            qs = parse_qs(parsed.query)
            payload = {}
            for key in ["yolo", "rl", "mode", "fps"]:
                if key in qs:
                    val = qs[key][0]
                    payload[key] = val
            if payload:
                REMOTE_CONTROL_QUEUE.put(payload)
            self._send_json({"ok": True, "applied": payload})
        elif parsed.path == "/marketplace":
            self._send_json(PLUGIN_MARKETPLACE.list_plugins())
        else:
            self._send_json({"error": "unknown endpoint"}, code=404)

def start_remote_api_server(port: int):
    def run():
        server = HTTPServer(("0.0.0.0", port), RemoteAPIHandler)
        print(f"[API] Remote control API on port {port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        server.server_close()
    t = threading.Thread(target=run, daemon=True)
    t.start()

# ============================================================
# VOICE CONTROL (OPTIONAL)
# ============================================================

class VoiceController:
    def __init__(self, command_queue: queue.Queue):
        self.command_queue = command_queue
        self.running = False
        self.thread = None
        self.recognizer = None
        self.mic = None
        if speech_recognition is not None:
            try:
                self.recognizer = speech_recognition.Recognizer()
                self.mic = speech_recognition.Microphone()
                print("[VOICE] Voice control initialized.")
            except Exception as e:
                print(f"[VOICE] Init error: {e}")
                self.recognizer = None
                self.mic = None

    def _loop(self):
        if self.recognizer is None or self.mic is None:
            return
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
        while self.running:
            try:
                with self.mic as source:
                    audio = self.recognizer.listen(source, phrase_time_limit=4)
                text = self.recognizer.recognize_google(audio).lower()
                print(f"[VOICE] Heard: {text}")
                cmd = {}
                if "toggle yolo" in text:
                    cmd["yolo"] = "toggle"
                if "toggle r l" in text or "toggle rl" in text:
                    cmd["rl"] = "toggle"
                if "altered mode" in text:
                    cmd["mode"] = "altered"
                if "normal mode" in text:
                    cmd["mode"] = "normal"
                if "fps up" in text:
                    cmd["fps"] = "up"
                if "fps down" in text:
                    cmd["fps"] = "down"
                if cmd:
                    self.command_queue.put(cmd)
            except Exception:
                pass

    def start(self):
        if self.recognizer is None or self.mic is None:
            print("[VOICE] Voice control not available.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[VOICE] Voice control started.")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)

# ============================================================
# PLUGIN ARCHITECTURE + HOT-RELOAD
# ============================================================

class PluginHub:
    def __init__(self):
        self.plugins = []
        self.external_modules = {}
        self.hot_reload_interval = 5.0
        self.last_reload_time = 0.0
        self.plugins_dir = "plugins"

    def register_plugin(self, name: str, plugin_obj):
        self.plugins.append(plugin_obj)
        print(f"[PLUGIN] Registered internal plugin: {name}")

    def load_external_plugins(self):
        if not os.path.isdir(self.plugins_dir):
            return
        for fname in os.listdir(self.plugins_dir):
            if not fname.endswith(".py"):
                continue
            mod_name = fname[:-3]
            full_name = f"{self.plugins_dir}.{mod_name}"
            try:
                if full_name in self.external_modules:
                    mod = importlib.reload(self.external_modules[full_name])
                else:
                    mod = importlib.import_module(full_name)
                self.external_modules[full_name] = mod
                if hasattr(mod, "register"):
                    mod.register(self)
                    print(f"[PLUGIN] Loaded external plugin: {mod_name}")
            except Exception as e:
                print(f"[PLUGIN] Failed to load {mod_name}: {e}")

    def hot_reload_if_needed(self):
        now = time.time()
        if now - self.last_reload_time > self.hot_reload_interval:
            self.last_reload_time = now
            self.load_external_plugins()

    def on_state(self, state: Dict[str, Any]):
        for p in self.plugins:
            try:
                if hasattr(p, "on_state"):
                    p.on_state(state)
            except Exception as e:
                print(f"[PLUGIN] on_state error in {getattr(p, '__name__', str(p))}: {e}")

    def on_action(self, action: Dict[str, Any]):
        for p in self.plugins:
            try:
                if hasattr(p, "on_action"):
                    p.on_action(action)
            except Exception as e:
                print(f"[PLUGIN] on_action error in {getattr(p, '__name__', str(p))}: {e}")

    def on_draw(self, frame):
        for p in self.plugins:
            try:
                if hasattr(p, "on_draw"):
                    p.on_draw(frame)
            except Exception as e:
                print(f"[PLUGIN] on_draw error in {getattr(p, '__name__', str(p))}: {e}")

PLUGIN_HUB = PluginHub()

# ============================================================
# MAVLINK / CAN INTERFACES
# ============================================================

class MAVLinkInterface:
    def __init__(self):
        self.available = pymavlink is not None
        self.mode = "STABILIZE"
        self.armed = False
        self.altitude = 0.0

    def connect(self, device: str):
        print(f"[MAVLINK] Placeholder connect to {device}")

    def set_mode(self, mode: str):
        self.mode = mode

    def arm(self, arm: bool):
        self.armed = arm

    def update_from_action(self, action: Dict[str, Any]):
        self.altitude += action["throttle"] * 0.1

class CANInterface:
    def __init__(self):
        self.available = python_can is not None
        self.speed = 0.0
        self.rpm = 800
        self.gear = 1

    def connect(self, channel: str, bustype: str = "socketcan"):
        print(f"[CAN] Placeholder connect to {channel} ({bustype})")

    def decode_frame(self, frame_bytes: bytes):
        pass

    def update_from_action(self, action: Dict[str, Any]):
        self.speed += max(0.0, action["throttle"]) * 0.5
        self.speed = max(0.0, min(self.speed, 200))
        self.rpm = 800 + int(self.speed * 40)
        self.gear = min(6, max(1, int(self.speed // 20) + 1))

# ============================================================
# INTERNAL PLUGINS
# ============================================================

class DroneAutopilotPlugin:
    PLUGIN_ID = "drone_autopilot"

    DRONE_STATE = {
        "altitude": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "target_locked": False,
        "target_id": None,
    }

    def __init__(self):
        self.mav = MAVLinkInterface()
        self.cfg = PLUGIN_SETTINGS.get(self.PLUGIN_ID, {"enabled": True})
        print(f"[PLUGIN][DRONE] Settings: {self.cfg}")

    def on_state(self, state):
        if not self.cfg.get("enabled", True):
            return
        objs = state.get("objects", [])
        if objs:
            target = min(objs, key=lambda o: (o["bbox"][0] + o["bbox"][2]) / 2)
            self.DRONE_STATE["target_locked"] = True
            self.DRONE_STATE["target_id"] = target["id"]
        else:
            self.DRONE_STATE["target_locked"] = False
            self.DRONE_STATE["target_id"] = None

    def on_action(self, action):
        if not self.cfg.get("enabled", True):
            return
        self.DRONE_STATE["pitch"] = action["throttle"]
        self.DRONE_STATE["yaw"] = action["steer"]
        self.DRONE_STATE["vx"] = self.DRONE_STATE["yaw"] * 0.1
        self.DRONE_STATE["vz"] = self.DRONE_STATE["pitch"] * 0.1
        self.DRONE_STATE["altitude"] += self.DRONE_STATE["vz"]
        self.mav.update_from_action(action)

    def on_draw(self, frame):
        if not self.cfg.get("enabled", True):
            return
        txt = f"DRONE ALT {self.DRONE_STATE['altitude']:.1f}  MODE {self.mav.mode}"
        cv2.putText(frame, txt, (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if self.DRONE_STATE["target_locked"]:
            cv2.putText(frame, f"LOCKED → ID {self.DRONE_STATE['target_id']}",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

class CarCANBusPlugin:
    PLUGIN_ID = "car_canbus"

    def __init__(self):
        self.can = CANInterface()
        self.cfg = PLUGIN_SETTINGS.get(self.PLUGIN_ID, {"enabled": True})
        self.state = {"speed": 0.0, "rpm": 800, "gear": 1, "steer": 0.0, "throttle": 0.0}
        print(f"[PLUGIN][CAR] Settings: {self.cfg}")

    def on_state(self, state):
        pass

    def on_action(self, action):
        if not self.cfg.get("enabled", True):
            return
        self.state["steer"] = action["steer"]
        self.state["throttle"] = max(0.0, action["throttle"])
        self.can.update_from_action(action)
        self.state["speed"] = self.can.speed
        self.state["rpm"] = self.can.rpm
        self.state["gear"] = self.can.gear

    def on_draw(self, frame):
        if not self.cfg.get("enabled", True):
            return
        dash = f"SPEED {self.state['speed']:.1f}  RPM {self.state['rpm']}  GEAR {self.state['gear']}"
        cv2.putText(frame, dash, (frame.shape[1] - 350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        steer_txt = f"STEER {self.state['steer']:.2f}"
        cv2.putText(frame, steer_txt, (frame.shape[1] - 350, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

class SwarmManagerPlugin:
    PLUGIN_ID = "swarm_manager"

    def __init__(self):
        self.cfg = PLUGIN_SETTINGS.get(self.PLUGIN_ID, {"enabled": True})
        self.swarm = [
            {"id": 1, "x": 0.0, "y": 0.0},
            {"id": 2, "x": 10.0, "y": -5.0},
            {"id": 3, "x": -8.0, "y": 7.0},
        ]
        print(f"[PLUGIN][SWARM] Settings: {self.cfg}")

    def on_state(self, state):
        if not self.cfg.get("enabled", True):
            return
        for d in self.swarm:
            d["x"] += random.uniform(-0.1, 0.1)
            d["y"] += random.uniform(-0.1, 0.1)

    def on_action(self, action):
        pass

    def on_draw(self, frame):
        pass

    def get_swarm_state(self):
        return self.swarm if self.cfg.get("enabled", True) else []

class VehicleAutopilotPlugin:
    PLUGIN_ID = "vehicle_autopilot"

    def __init__(self):
        self.cfg = PLUGIN_SETTINGS.get(self.PLUGIN_ID, {"enabled": True})
        self.lane_center_offset = 0.0
        self.target_speed = 60.0
        print(f"[PLUGIN][VEHICLE] Settings: {self.cfg}")

    def on_state(self, state):
        if not self.cfg.get("enabled", True):
            return
        objs = state.get("objects", [])
        lane_objs = [o for o in objs if "lane" in o.get("class", "").lower()]
        if lane_objs:
            lane = lane_objs[0]
            x1, y1, x2, y2 = lane["bbox"]
            center = 0.5 * (x1 + x2)
            self.lane_center_offset = center - 320.0
        else:
            self.lane_center_offset *= 0.9

    def on_action(self, action):
        pass

    def on_draw(self, frame):
        if not self.cfg.get("enabled", True):
            return
        txt = f"AUTO LANE OFFSET {self.lane_center_offset:.1f}"
        cv2.putText(frame, txt, (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,180,255), 2)

DRONE_PLUGIN = DroneAutopilotPlugin()
CAR_PLUGIN = CarCANBusPlugin()
SWARM_PLUGIN = SwarmManagerPlugin()
VEHICLE_PLUGIN = VehicleAutopilotPlugin()

PLUGIN_HUB.register_plugin("drone_autopilot", DRONE_PLUGIN)
PLUGIN_HUB.register_plugin("car_canbus", CAR_PLUGIN)
PLUGIN_HUB.register_plugin("swarm_manager", SWARM_PLUGIN)
PLUGIN_HUB.register_plugin("vehicle_autopilot", VEHICLE_PLUGIN)

# ============================================================
# MAIN CORE LOOP
# ============================================================

def core_main():
    global RL_MODE, FPS_TARGET, ENABLE_YOLO_RUNTIME, ENABLE_RL_RUNTIME, GLOBAL_STATUS, NETWORK_BROADCAST_ENABLED

    saved_rl_state = LOADED_MEMORY.get("rl_state", None)

    video_manager = VideoManager(VIDEO_SOURCE_TYPE, VIDEO_SOURCE_PATH)
    detector = build_detector()
    tracker = SimpleTracker()
    logger = StateLogger(LOG_FILE if SAVE_LOG else None)

    kb_logger = KeyboardLogger() if USE_KEYBOARD_LOGGER else None
    if kb_logger is not None:
        kb_logger.start()

    rl_learner = RLLearner(saved_state=saved_rl_state, mode=RL_MODE) if USE_RL else None
    ui_observer = UIAutomationObserver()
    predictor = PredictiveReasoner()
    water_engine = WaterPhysicsEngine()

    cv2.namedWindow("Unified Video AI")
    param = {"frame_shape": (480, 640, 3)}
    cv2.setMouseCallback("Unified Video AI", mouse_callback, param)

    frame_interval = 1.0 / FPS_TARGET
    last_time = time.time()
    last_objects = []
    last_frame = None
    consecutive_ghost_frames = 0

    net_broadcaster = NetworkBroadcaster(NETWORK_BROADCAST_PORT)
    net_broadcaster.enabled = NETWORK_BROADCAST_ENABLED
    net_control_queue = queue.Queue()
    net_listener = NetworkListener(NETWORK_LISTEN_PORT, net_control_queue)
    net_listener.start()

    start_remote_api_server(REMOTE_API_PORT)

    voice_queue = queue.Queue()
    voice_controller = VoiceController(voice_queue)
    voice_controller.start()

    PLUGIN_HUB.load_external_plugins()

    try:
        while True:
            PLUGIN_HUB.hot_reload_if_needed()

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

            param["frame_shape"] = frame.shape

            while not REMOTE_CONTROL_QUEUE.empty():
                cmd = REMOTE_CONTROL_QUEUE.get()
                if "yolo" in cmd:
                    if cmd["yolo"] in ("1", "true"):
                        ENABLE_YOLO_RUNTIME = True
                    elif cmd["yolo"] in ("0", "false"):
                        ENABLE_YOLO_RUNTIME = False
                    elif cmd["yolo"] == "toggle":
                        ENABLE_YOLO_RUNTIME = not ENABLE_YOLO_RUNTIME
                if "rl" in cmd:
                    if cmd["rl"] in ("1", "true"):
                        ENABLE_RL_RUNTIME = True
                    elif cmd["rl"] in ("0", "false"):
                        ENABLE_RL_RUNTIME = False
                    elif cmd["rl"] == "toggle":
                        ENABLE_RL_RUNTIME = not ENABLE_RL_RUNTIME
                if "mode" in cmd:
                    RL_MODE = cmd["mode"]
                if "fps" in cmd:
                    try:
                        FPS_TARGET = float(cmd["fps"])
                    except Exception:
                        if cmd["fps"] == "up":
                            FPS_TARGET = min(60.0, FPS_TARGET + 1.0)
                        elif cmd["fps"] == "down":
                            FPS_TARGET = max(1.0, FPS_TARGET - 1.0)

            while not net_control_queue.empty():
                cmd = net_control_queue.get()
                if "yolo" in cmd:
                    ENABLE_YOLO_RUNTIME = bool(cmd["yolo"])
                if "rl" in cmd:
                    ENABLE_RL_RUNTIME = bool(cmd["rl"])
                if "mode" in cmd:
                    RL_MODE = cmd["mode"]
                if "fps" in cmd:
                    FPS_TARGET = float(cmd["fps"])

            while not voice_queue.empty():
                cmd = voice_queue.get()
                if "yolo" in cmd and cmd["yolo"] == "toggle":
                    ENABLE_YOLO_RUNTIME = not ENABLE_YOLO_RUNTIME
                if "rl" in cmd and cmd["rl"] == "toggle":
                    ENABLE_RL_RUNTIME = not ENABLE_RL_RUNTIME
                if "mode" in cmd:
                    RL_MODE = cmd["mode"]
                if "fps" in cmd:
                    if cmd["fps"] == "up":
                        FPS_TARGET = min(60.0, FPS_TARGET + 1.0)
                    elif cmd["fps"] == "down":
                        FPS_TARGET = max(1.0, FPS_TARGET - 1.0)

            now = time.time()
            timestamp = now

            if ENABLE_YOLO_RUNTIME:
                try:
                    detections = detector.detect(frame) if is_real_frame else last_objects
                except Exception as e:
                    print(f"[MAIN] Detector error: {e}")
                    detections = last_objects
            else:
                detections = []

            if not detections and last_objects:
                detections = last_objects

            tracked_objects = tracker.update(detections)
            last_objects = tracked_objects

            input_events = []
            if kb_logger is not None:
                input_events.extend(kb_logger.drain_events())

            ui_context = ui_observer.snapshot()
            predictor.update_from_objects(tracked_objects)
            pred_probs = predictor.get_probabilities()
            physics_state = water_engine.update(tracked_objects)

            if not ALTERED_MANUAL_OVERRIDE:
                if consecutive_ghost_frames > 10:
                    RL_MODE = "altered"
                else:
                    RL_MODE = "normal"

            if rl_learner is not None:
                rl_learner.set_mode(RL_MODE)

            state = {
                "timestamp": timestamp,
                "objects": tracked_objects,
                "objects_physics": physics_state,
                "input_events": input_events,
                "source_type": VIDEO_SOURCE_TYPE,
                "ui_context": ui_context,
                "predictions": pred_probs,
                "is_real_frame": is_real_frame,
                "rl_mode": RL_MODE,
            }

            control_suggestion = None
            if rl_learner is not None and ENABLE_RL_RUNTIME:
                action = rl_learner.on_new_state(state)
                control_suggestion = action
                state["control_suggestion"] = action
                PLUGIN_HUB.on_action(action)

            PLUGIN_HUB.on_state(state)
            logger.log(state)

            if NETWORK_BROADCAST_ENABLED:
                try:
                    net_broadcaster.send_state({
                        "timestamp": timestamp,
                        "rl_mode": RL_MODE,
                        "num_objects": len(tracked_objects),
                        "predictions": pred_probs,
                    })
                except Exception as e:
                    print(f"[NET] Broadcast error: {e}")

            for obj in tracked_objects:
                x1, y1, x2, y2 = obj["bbox"]
                cls = obj["class"]
                score = obj["score"]
                oid = obj.get("id", -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{oid}:{cls} {score:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if control_suggestion is not None:
                txt = f"throttle={control_suggestion['throttle']:.2f}, steer={control_suggestion['steer']:.2f}"
                cv2.putText(
                    frame,
                    txt,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            if ui_context is not None:
                title = ui_context.get("name", "")
                cv2.putText(
                    frame,
                    f"UI: {title[:40]}",
                    (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )

            swarm_state = SWARM_PLUGIN.get_swarm_state()

            HUD.draw(
                frame,
                yolo_on=ENABLE_YOLO_RUNTIME,
                rl_on=ENABLE_RL_RUNTIME,
                altered=(RL_MODE == "altered"),
                fps=FPS_TARGET,
                ghost_count=consecutive_ghost_frames,
                save_path_display=CURRENT_SAVE_PATH_DISPLAY,
                swarm_state=swarm_state,
            )

            DRIVE_POPUP.draw(frame)
            PLUGIN_HUB.on_draw(frame)

            GLOBAL_STATUS["yolo"] = ENABLE_YOLO_RUNTIME
            GLOBAL_STATUS["rl"] = ENABLE_RL_RUNTIME
            GLOBAL_STATUS["mode"] = RL_MODE
            GLOBAL_STATUS["fps"] = FPS_TARGET

            cv2.imshow("Unified Video AI", frame)

            frame_interval = 1.0 / FPS_TARGET
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
        voice_controller.stop()
        net_listener.stop()
        snapshot = build_memory_snapshot(rl_learner)
        memory_manager.save(snapshot)

# ============================================================
# DEARPYGUI FRONT-END (CONTROL CONSOLE)
# ============================================================

API_BASE = "http://127.0.0.1:8000"

status_lock = threading.Lock()
current_status: Dict[str, Any] = {
    "yolo": False,
    "rl": False,
    "mode": "normal",
    "fps": 20.0,
}
marketplace_data: Dict[str, Any] = {"plugins": []}
gui_running = True

def api_get(path: str, params: Dict[str, Any] = None) -> Any:
    url = API_BASE + path
    try:
        r = requests.get(url, params=params, timeout=1.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def refresh_status():
    global current_status
    data = api_get("/status")
    if "error" in data:
        return
    with status_lock:
        current_status.update(data)

def send_toggle(params: Dict[str, Any]):
    api_get("/toggle", params=params)

def refresh_marketplace():
    global marketplace_data
    data = api_get("/marketplace")
    if "error" in data:
        return
    marketplace_data = data

def polling_loop_gui():
    while gui_running:
        try:
            refresh_status()
            refresh_marketplace()
        except Exception:
            pass
        time.sleep(0.5)

def cb_toggle_yolo(sender, app_data, user_data):
    with status_lock:
        new_val = not current_status.get("yolo", False)
    send_toggle({"yolo": "1" if new_val else "0"})

def cb_toggle_rl(sender, app_data, user_data):
    with status_lock:
        new_val = not current_status.get("rl", False)
    send_toggle({"rl": "1" if new_val else "0"})

def cb_set_mode(sender, app_data, user_data):
    mode = app_data
    send_toggle({"mode": mode})

def cb_set_fps(sender, app_data, user_data):
    fps = float(app_data)
    send_toggle({"fps": str(fps)})

def cb_refresh_now(sender, app_data, user_data):
    refresh_status()
    refresh_marketplace()

def draw_status_panel():
    with dpg.group(horizontal=False):
        dpg.add_text("Core Status", color=(0, 255, 0))
        dpg.add_separator()
        dpg.add_text("Connection: ", tag="status_connection_text")
        dpg.add_text("YOLO: ", tag="status_yolo_text")
        dpg.add_text("RL: ", tag="status_rl_text")
        dpg.add_text("Mode: ", tag="status_mode_text")
        dpg.add_text("FPS: ", tag="status_fps_text")
        dpg.add_spacer(height=5)
        dpg.add_button(label="Refresh Now", callback=cb_refresh_now)

def draw_control_panel():
    with dpg.group(horizontal=False):
        dpg.add_text("Controls", color=(0, 200, 255))
        dpg.add_separator()
        dpg.add_button(label="Toggle YOLO", callback=cb_toggle_yolo)
        dpg.add_button(label="Toggle RL", callback=cb_toggle_rl)
        dpg.add_spacer(height=5)
        dpg.add_text("Mode:")
        dpg.add_radio_button(items=["normal", "altered"],
                             tag="mode_radio",
                             default_value="normal",
                             callback=cb_set_mode)
        dpg.add_spacer(height=5)
        dpg.add_text("FPS Target:")
        dpg.add_slider_float(tag="fps_slider",
                             default_value=20.0,
                             min_value=1.0,
                             max_value=60.0,
                             callback=cb_set_fps)

def draw_marketplace_panel():
    with dpg.group(horizontal=False):
        dpg.add_text("Plugin Marketplace", color=(255, 200, 0))
        dpg.add_separator()
        dpg.add_child_window(tag="marketplace_child", width=400, height=250, border=True)

def draw_swarm_panel():
    with dpg.group(horizontal=False):
        dpg.add_text("Swarm Visualizer (logical)", color=(200, 255, 200))
        dpg.add_separator()
        with dpg.drawlist(width=300, height=300, tag="swarm_drawlist"):
            dpg.draw_rectangle((0, 0), (300, 300), color=(80, 80, 80), fill=(20, 20, 20))
            dpg.draw_text((10, 10), "SWARM", color=(255, 255, 255), size=16)

def draw_save_panel():
    with dpg.group(horizontal=False):
        dpg.add_text("Save / Config", color=(180, 180, 255))
        dpg.add_separator()
        dpg.add_input_text(label="Current Save Path", tag="save_path_text", readonly=True)
        dpg.add_text("Note: path is managed by core HUD popup.\nYou can display it here for reference.")

def gui_update():
    with status_lock:
        st = dict(current_status)

    if st:
        dpg.set_value("status_connection_text", f"Connection: OK")
    else:
        dpg.set_value("status_connection_text", f"Connection: DISCONNECTED")

    dpg.set_value("status_yolo_text", f"YOLO: {'ON' if st.get('yolo') else 'OFF'}")
    dpg.set_value("status_rl_text", f"RL: {'ON' if st.get('rl') else 'OFF'}")
    dpg.set_value("status_mode_text", f"Mode: {st.get('mode', 'unknown')}")
    dpg.set_value("status_fps_text", f"FPS: {st.get('fps', 0):.1f}")

    dpg.set_value("mode_radio", st.get("mode", "normal"))
    dpg.set_value("fps_slider", float(st.get("fps", 20.0)))

    dpg.delete_item("marketplace_child", children_only=True)
    with dpg.group(parent="marketplace_child"):
        plugins: List[Dict[str, Any]] = marketplace_data.get("plugins", [])
        if not plugins:
            dpg.add_text("No plugins or core not responding.")
        else:
            for p in plugins:
                dpg.add_text(f"{p.get('name', 'Unknown')} ({p.get('id', '')})", color=(0, 255, 0))
                dpg.add_text(f"  v{p.get('version', '?')} by {p.get('author', 'Unknown')}")
                dpg.add_text(f"  {p.get('description', '')}")
                dpg.add_spacer(height=4)
                dpg.add_separator()

    dpg.delete_item("swarm_drawlist", children_only=True)
    with dpg.drawlist(parent="swarm_drawlist", width=300, height=300):
        dpg.draw_rectangle((0, 0), (300, 300), color=(80, 80, 80), fill=(20, 20, 20))
        dpg.draw_text((10, 10), "SWARM", color=(255, 255, 255), size=16)
        dpg.draw_circle((150, 150), 5, color=(0, 255, 255), fill=(0, 255, 255))
        dpg.draw_circle((190, 130), 5, color=(0, 255, 255), fill=(0, 255, 255))
        dpg.draw_circle((120, 190), 5, color=(0, 255, 255), fill=(0, 255, 255))

    dpg.set_value("save_path_text", "Managed by core HUD (drive popup)")

def gui_main():
    global gui_running

    dpg.create_context()
    dpg.create_viewport(title="RebootCore Control Console (DearPyGUI)", width=1100, height=700)

    with dpg.window(label="RebootCore Console", width=1080, height=680):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=260, height=640, border=True):
                draw_status_panel()
                dpg.add_spacer(height=10)
                draw_control_panel()
            with dpg.child_window(width=420, height=640, border=True):
                draw_marketplace_panel()
                dpg.add_spacer(height=10)
                draw_save_panel()
            with dpg.child_window(width=360, height=640, border=True):
                draw_swarm_panel()

    dpg.set_primary_window(dpg.last_container(), True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    t = threading.Thread(target=polling_loop_gui, daemon=True)
    t.start()

    dpg.set_frame_callback(0, lambda: None)
    dpg.set_render_callback(gui_update)

    try:
        dpg.start_dearpygui()
    finally:
        gui_running = False
        dpg.destroy_context()

# ============================================================
# ENTRY POINT
# ============================================================

def run_core_and_gui():
    core_thread = threading.Thread(target=core_main, daemon=True)
    core_thread.start()
    gui_main()

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--gui" in args:
        gui_main()
    elif "--core-and-gui" in args:
        choose_initial_save_location()
        run_core_and_gui()
    else:
        choose_initial_save_location()
        core_main()
