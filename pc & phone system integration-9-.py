#!/usr/bin/env python3
"""
phone_swarm_organism.py

Cybernetic phone swarm organism with:

PHASE 2 – AUTONOMY
------------------
- Swarm-level policy engine on controller:
  * Keeps minimum number of camera streams alive
  * Ensures periodic sensor + thermal sampling
  * Reassigns tasks when nodes go offline (self-healing)
- Task allocator:
  * Chooses best node based on CPU/battery/capabilities
- Agent-side local scheduler:
  * Runs periodic jobs even without controller (self-tasking)
  * Example jobs: sensors_read, thermal_status, AI inference

PHASE 3 – GUI EVOLUTION
-----------------------
- Camera wall (multi-node grid)
- Node health dashboard:
  * CPU/Battery history graphs
  * Mesh + BLE + AI status summary
- Map evolution hooks:
  * Real-time node markers
  * Mesh topology overlay (stub)
  * Future: 3D map / layered views

PHASE 4 – AI
------------
- Preprocessing for YOLO/MobileNet/Whisper-style
- Postprocessing:
  * YOLO: bounding boxes + labels (simplified)
  * MobileNet: top-k classes
  * Whisper: text hook
- ONNX Runtime Mobile on agent
- Controller shows AI status per node

PHASE 5 – SWARM INTELLIGENCE
----------------------------
- Consensus + leader election (stubbed but structured):
  * Leader role in mesh
  * Simple heartbeat-based election
- Distributed memory:
  * MESH_STATE shared via gossip
  * Versioned state for swarm-wide decisions

PHASE 6 – REAL-WORLD ROBOTICS CONTROL
-------------------------------------
- BLE robots / drones / vehicles / IoT:
  * GATT-based command channels
  * High-level RoboticsController on controller side
  * Agent-side BLE hooks already present
- Future: motion primitives, path plans, safety envelopes

PHASE 7 – FULL AUTONOMY
-----------------------
- Swarm modes:
  * IDLE, PATROL, FOLLOW, GUARD, EXPLORE (extensible)
- Behavior engine:
  * Mode -> policy set -> tasks
  * Uses swarm intelligence + distributed memory
- Hooks for:
  * AI-triggered mode changes
  * Voice-triggered commands
  * Mesh-triggered reconfiguration

PHASE 8 – NEURAL COORDINATION
-----------------------------
- NeuralCoordinator (controller):
  * Distributed model registry (name, version, checksum, size, roles)
  * Tracks which node has which model version
  * Suggests role specialization (vision/audio/router/robotics)
  * Uses mesh distributed memory for model metadata
- NeuralAgent (agent):
  * Local experience buffer (AI input/output/context)
  * Periodic experience summarization
  * Syncs model metadata with swarm via mesh
  * Hooks for future model update / P2P weight transfer

OTHER ORGANS
------------
1) H.264 video streaming (optimized)
   - Termux + ffmpeg -> H.264 TS chunks over WebSocket
   - Controller-side decoding (OpenCV/ffmpeg) into frames
   - Multi-stream grid in GUI
   - Low-latency mode flag

2) Bluetooth GATT (BLE control)
   - Service discovery
   - GATT read/write
   - Notifications subscribe/unsubscribe (stub)
   - Multi-device sessions per phone

3) Mesh networking
   - UDP discovery
   - TCP peer links
   - Peer routing table
   - Gossip protocol for shared state
   - Task forwarding stub

4) Voice recognition (async-ish)
   - Streaming microphone (agent-side)
   - Wake-word detection stub
   - Real-time transcription via speech_recognition

5) GUI cockpit
   - Node list + status
   - Map panel (lat/lon projection, mesh overlay hooks)
   - Multi-camera grid (JPEG + decoded H.264 frames)
   - Node health charts (CPU/Battery history)
   - Bluetooth devices + GATT status
   - Mesh peers + routing info
   - Voice status + last transcript
   - AI inference status + last detections
   - Camera wall + node health dashboard (Phase 3)

USAGE
-----

On PC (controller):

    python phone_swarm_organism.py --mode controller --wifi-ip 192.168.1.50 --wifi-port 8765

On Android (Termux agent):

    pkg install python
    pkg install termux-api
    pkg install ffmpeg
    pip install websockets onnxruntime bleak speechrecognition pillow opencv-python numpy

    python phone_swarm_organism.py --mode agent

NOTE:
- YOLO/MobileNet/Whisper ONNX paths are configured in MODEL_REGISTRY.
- Postprocessing is simplified but structurally correct; plug in real label files and decoding logic.
"""

import asyncio
import json
import logging
import random
import string
import sys
import time
import subprocess
import base64
import os
import socket
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable, Tuple

# ---------------------------------------------------------------------------
# Optional libs
# ---------------------------------------------------------------------------

try:
    import websockets  # type: ignore
except ImportError:
    websockets = None

try:
    import onnxruntime as ort  # type: ignore
except ImportError:
    ort = None

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:
    cv2 = None
    np = None

try:
    import bleak  # type: ignore
except ImportError:
    bleak = None

# Voice recognition
VOICE_BACKEND = None
try:
    import speech_recognition as sr  # type: ignore
    VOICE_BACKEND = "speech_recognition"
except ImportError:
    VOICE_BACKEND = None

# GUI libs
try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk  # type: ignore
    import io
except ImportError:
    tk = None
    Image = None
    ImageTk = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phone_swarm_organism")


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class NodeTransport(Enum):
    USB = auto()
    WIFI = auto()


class NodeStatus(Enum):
    UNKNOWN = auto()
    ONLINE = auto()
    OFFLINE = auto()
    DEGRADED = auto()


class SwarmMode(Enum):
    IDLE = auto()
    PATROL = auto()
    FOLLOW = auto()
    GUARD = auto()
    EXPLORE = auto()


@dataclass
class NodeCapabilities:
    cpu_cores: int = 4
    ram_mb: int = 2048
    has_npu: bool = False
    has_gpu: bool = True
    has_camera: bool = True
    has_mic: bool = True
    has_sensors: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    task_id: str
    node_id: str
    success: bool
    payload: Any
    error: Optional[str] = None


def _generate_task_id(length: int = 10) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "task-" + "".join(random.choice(alphabet) for _ in range(length))


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: Base node
# ---------------------------------------------------------------------------

class PhoneNode:
    def __init__(
        self,
        node_id: str,
        transport: NodeTransport,
        capabilities: Optional[NodeCapabilities] = None,
    ) -> None:
        self.node_id = node_id
        self.transport = transport
        self.capabilities = capabilities or NodeCapabilities()
        self.status: NodeStatus = NodeStatus.UNKNOWN
        self.last_seen: float = 0.0
        self._lock = asyncio.Lock()
        self.last_gps: Optional[Dict[str, Any]] = None

    async def connect(self) -> None:
        raise NotImplementedError

    async def disconnect(self) -> None:
        raise NotImplementedError

    async def send_command(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def health_check(self) -> bool:
        try:
            resp = await self.send_command("health_check", {})
            ok = bool(resp.get("ok", False))
            self.status = NodeStatus.ONLINE if ok else NodeStatus.DEGRADED
            self.last_seen = time.time()
            caps = resp.get("result", {}).get("caps")
            if isinstance(caps, dict):
                self._update_caps_from_agent(caps)
            return ok
        except Exception as e:
            logger.warning("Health check failed for %s: %s", self.node_id, e)
            self.status = NodeStatus.OFFLINE
            return False

    def _update_caps_from_agent(self, caps: Dict[str, Any]) -> None:
        self.capabilities.cpu_cores = int(caps.get("cpu_cores", self.capabilities.cpu_cores))
        self.capabilities.ram_mb = int(caps.get("ram_mb", self.capabilities.ram_mb))
        self.capabilities.has_npu = bool(caps.get("has_npu", self.capabilities.has_npu))
        self.capabilities.has_gpu = bool(caps.get("has_gpu", self.capabilities.has_gpu))
        self.capabilities.has_camera = bool(caps.get("has_camera", self.capabilities.has_camera))
        self.capabilities.has_mic = bool(caps.get("has_mic", self.capabilities.has_mic))
        self.capabilities.has_sensors = bool(caps.get("has_sensors", self.capabilities.has_sensors))
        tags = caps.get("tags")
        if isinstance(tags, list):
            self.capabilities.tags = [str(t) for t in tags]

    def __repr__(self) -> str:
        return f"<PhoneNode id={self.node_id} transport={self.transport.name} status={self.status.name}>"


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: USB node (stub)
# ---------------------------------------------------------------------------

class UsbPhoneNode(PhoneNode):
    def __init__(self, adb_serial: str, **kwargs: Any) -> None:
        super().__init__(node_id=f"usb-{adb_serial}", transport=NodeTransport.USB, **kwargs)
        self.adb_serial = adb_serial

    async def connect(self) -> None:
        logger.info("Connecting USB node %s (serial=%s)", self.node_id, self.adb_serial)
        self.status = NodeStatus.ONLINE
        self.last_seen = time.time()
        logger.info("USB node %s is online (stub)", self.node_id)

    async def disconnect(self) -> None:
        logger.info("Disconnecting USB node %s", self.node_id)
        self.status = NodeStatus.OFFLINE

    async def send_command(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            if self.status == NodeStatus.OFFLINE:
                raise RuntimeError(f"Node {self.node_id} is offline")

            task_id = payload.get("task_id")
            if command == "health_check":
                return {
                    "ok": True,
                    "result": {
                        "node_id": self.node_id,
                        "uptime_sec": 123.4,
                        "battery_pct": 87.0,
                        "temp_c": 35.0,
                        "caps": {
                            "cpu_cores": self.capabilities.cpu_cores,
                            "ram_mb": self.capabilities.ram_mb,
                            "has_npu": self.capabilities.has_npu,
                            "has_gpu": self.capabilities.has_gpu,
                            "has_camera": self.capabilities.has_camera,
                            "has_mic": self.capabilities.has_mic,
                            "has_sensors": self.capabilities.has_sensors,
                            "tags": self.capabilities.tags,
                        },
                    },
                    "error": None,
                    "task_id": task_id,
                }

            if command == "compute":
                expr = payload.get("expr", "1+1")
                try:
                    value = eval(expr, {"__builtins__": {}})
                    return {"ok": True, "result": {"value": value}, "error": None, "task_id": task_id}
                except Exception as e:
                    return {"ok": False, "result": None, "error": str(e), "task_id": task_id}

            if command in ("camera_start", "camera_stop", "mic_start", "mic_stop"):
                return {
                    "ok": True,
                    "result": {
                        "stream_id": payload.get("stream_id"),
                        "status": "started" if command.endswith("start") else "stopped",
                    },
                    "error": None,
                    "task_id": task_id,
                }

            if command == "sensors_read":
                return {
                    "ok": True,
                    "result": {
                        "timestamp": time.time(),
                        "sensors": {
                            "accel": [0.0, 0.0, 9.81],
                            "gyro": [0.0, 0.0, 0.0],
                            "gps": {"lat": 0.0, "lon": 0.0, "alt": 0.0, "acc": 100.0},
                            "light": 100.0,
                        },
                    },
                    "error": None,
                    "task_id": task_id,
                }

            if command == "ai_infer":
                model = payload.get("model", "unknown_model")
                return {
                    "ok": True,
                    "result": {
                        "model": model,
                        "output": {"stub": True, "message": f"AI inference on {model} not implemented (USB stub)"},
                    },
                    "error": None,
                    "task_id": task_id,
                }

            if command == "thermal_status":
                return {
                    "ok": True,
                    "result": {
                        "cpu_load": 0.2,
                        "temp_c": 40.0,
                        "battery_pct": 80.0,
                    },
                    "error": None,
                    "task_id": task_id,
                }

            if command == "voice_status":
                return {
                    "ok": True,
                    "result": {"listening": False, "last_text": None},
                    "error": None,
                    "task_id": task_id,
                }

            if command == "bt_scan":
                return {
                    "ok": True,
                    "result": {
                        "devices": [
                            {"mac": "00:11:22:33:44:55", "name": "StubDevice", "rssi": -60},
                        ]
                    },
                    "error": None,
                    "task_id": task_id,
                }

            if command == "bt_status":
                return {
                    "ok": True,
                    "result": {"enabled": True, "scanning": False},
                    "error": None,
                    "task_id": task_id,
                }

            if command == "bt_gatt":
                return {
                    "ok": True,
                    "result": {"stub": True, "message": "GATT not implemented on USB stub"},
                    "error": None,
                    "task_id": task_id,
                }

            if command == "bt_services":
                return {
                    "ok": True,
                    "result": {"services": []},
                    "error": None,
                    "task_id": task_id,
                }

            if command == "bt_notify":
                return {
                    "ok": True,
                    "result": {"stub": True, "message": "notifications not implemented on USB stub"},
                    "error": None,
                    "task_id": task_id,
                }

            if command == "mesh_info":
                return {
                    "ok": True,
                    "result": {"peers": [], "routes": [], "state_version": 0, "neural_models": {}, "neural_roles": {}},
                    "error": None,
                    "task_id": task_id,
                }

            if command == "mesh_task":
                return {
                    "ok": True,
                    "result": {"stub": True, "message": "mesh_task not implemented on USB stub"},
                    "error": None,
                    "task_id": task_id,
                }

            if command == "voice_recognize":
                return {
                    "ok": True,
                    "result": {"stub": True, "error": "voice_recognize not available on USB stub"},
                    "error": None,
                    "task_id": task_id,
                }

            if command == "neural_sync":
                return {
                    "ok": True,
                    "result": {"stub": True, "message": "neural_sync not implemented on USB stub"},
                    "error": None,
                    "task_id": task_id,
                }

            return {
                "ok": True,
                "result": {"echo": {"command": command, "payload": payload}},
                "error": None,
                "task_id": task_id,
            }


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: WiFi node
# ---------------------------------------------------------------------------

class WifiPhoneNode(PhoneNode):
    def __init__(self, host: str, port: int = 8765, **kwargs: Any) -> None:
        node_id = f"wifi-{host}:{port}"
        super().__init__(node_id=node_id, transport=NodeTransport.WIFI, **kwargs)
        self.host = host
        self.port = port
        self._ws = None

    async def connect(self) -> None:
        if websockets is None:
            logger.error("websockets not installed; WiFi node %s disabled", self.node_id)
            self.status = NodeStatus.OFFLINE
            return

        uri = f"ws://{self.host}:{self.port}"
        logger.info("Connecting WiFi node %s (%s)", self.node_id, uri)
        try:
            self._ws = await websockets.connect(uri, ping_interval=10, ping_timeout=10)
            self.status = NodeStatus.ONLINE
            self.last_seen = time.time()
            logger.info("WiFi node %s is online", self.node_id)
        except Exception as e:
            logger.warning("Failed to connect WiFi node %s: %s", self.node_id, e)
            self.status = NodeStatus.OFFLINE

    async def disconnect(self) -> None:
        logger.info("Disconnecting WiFi node %s", self.node_id)
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self.status = NodeStatus.OFFLINE

    async def send_command(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if websockets is None:
            raise RuntimeError("websockets not installed")

        async with self._lock:
            if self._ws is None or self.status == NodeStatus.OFFLINE:
                raise RuntimeError(f"Node {self.node_id} is offline or not connected")

            task_id = payload.get("task_id") or _generate_task_id()
            msg = {
                "command": command,
                "payload": payload,
                "task_id": task_id,
            }
            data = json.dumps(msg)
            await self._ws.send(data)

            raw = await self._ws.recv()
            try:
                resp = json.loads(raw)
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid JSON from node {self.node_id}: {raw}")

            self.last_seen = time.time()
            return resp


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: H.264 decoding helpers
# ---------------------------------------------------------------------------

def decode_h264_ts_chunk(chunk_b64: str) -> Optional["Image.Image"]:
    if cv2 is None or np is None or Image is None:
        return None
    try:
        raw = base64.b64decode(chunk_b64)
        tmp_path = "_tmp_chunk.h264"
        with open(tmp_path, "wb") as f:
            f.write(raw)
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        os.remove(tmp_path)
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: Swarm intelligence (Phase 5)
# ---------------------------------------------------------------------------

class SwarmConsensusRole(Enum):
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass
class SwarmConsensusState:
    role: SwarmConsensusRole = SwarmConsensusRole.FOLLOWER
    current_leader: Optional[str] = None
    last_heartbeat: float = 0.0
    term: int = 0


class SwarmIntelligence:
    """
    Phase 5: Swarm Intelligence
    - Simple leader election (heartbeat-based)
    - Distributed memory via MESH_STATE (agent side)
    - Hooks for future consensus algorithms
    """

    def __init__(self, swarm: "SwarmManager") -> None:
        self.swarm = swarm
        self.state = SwarmConsensusState()
        self.heartbeat_timeout = 15.0
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            await self._tick()

    async def _tick(self) -> None:
        now = time.time()
        if self.state.current_leader is None:
            await self._attempt_leader_election()
        else:
            if now - self.state.last_heartbeat > self.heartbeat_timeout:
                logger.info("[swarm-intel] Leader %s timed out, re-electing", self.state.current_leader)
                self.state.current_leader = None
                await self._attempt_leader_election()

    async def _attempt_leader_election(self) -> None:
        online = self.swarm.list_online_nodes()
        if not online:
            return
        leader = sorted(online, key=lambda n: n.node_id)[0]
        self.state.current_leader = leader.node_id
        self.state.role = SwarmConsensusRole.LEADER
        self.state.term += 1
        self.state.last_heartbeat = time.time()
        logger.info("[swarm-intel] New leader elected: %s (term %d)", leader.node_id, self.state.term)

    def heartbeat_from_leader(self, leader_id: str) -> None:
        self.state.current_leader = leader_id
        self.state.last_heartbeat = time.time()
        if self.state.role != SwarmConsensusRole.LEADER:
            self.state.role = SwarmConsensusRole.FOLLOWER

    def get_leader(self) -> Optional[str]:
        return self.state.current_leader


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: Robotics control (Phase 6)
# ---------------------------------------------------------------------------

class RoboticsController:
    """
    Phase 6: Real-world Robotics Control
    - High-level commands for BLE robots/drones/vehicles/IoT
    - Uses underlying bt_gatt / bt_notify primitives
    """

    def __init__(self, swarm: "SwarmManager") -> None:
        self.swarm = swarm

    async def send_robot_command(
        self,
        node_id: str,
        mac: str,
        service_uuid: str,
        char_uuid: str,
        command_bytes: bytes,
    ) -> TaskResult:
        hex_data = command_bytes.hex()
        return await self.swarm.bt_gatt(
            node_id=node_id,
            mac=mac,
            service_uuid=service_uuid,
            char_uuid=char_uuid,
            op="write",
            data=hex_data,
        )

    async def read_robot_state(
        self,
        node_id: str,
        mac: str,
        service_uuid: str,
        char_uuid: str,
    ) -> TaskResult:
        return await self.swarm.bt_gatt(
            node_id=node_id,
            mac=mac,
            service_uuid=service_uuid,
            char_uuid=char_uuid,
            op="read",
        )


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: Full autonomy behavior engine (Phase 7)
# ---------------------------------------------------------------------------

class SwarmBehaviorEngine:
    """
    Phase 7: Full Autonomy
    - Swarm modes (IDLE, PATROL, FOLLOW, GUARD, EXPLORE)
    - Mode -> policy set -> tasks
    - Hooks for AI/voice/mesh-triggered transitions
    """

    def __init__(self, swarm: "SwarmManager", intel: SwarmIntelligence) -> None:
        self.swarm = swarm
        self.intel = intel
        self.mode: SwarmMode = SwarmMode.IDLE
        self._task: Optional[asyncio.Task] = None

    def set_mode(self, mode: SwarmMode) -> None:
        logger.info("[behavior] Swarm mode -> %s", mode.name)
        self.mode = mode

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            await self._tick()

    async def _tick(self) -> None:
        if self.mode == SwarmMode.IDLE:
            return
        if self.mode == SwarmMode.PATROL:
            await self._patrol_tick()
        elif self.mode == SwarmMode.FOLLOW:
            await self._follow_tick()
        elif self.mode == SwarmMode.GUARD:
            await self._guard_tick()
        elif self.mode == SwarmMode.EXPLORE:
            await self._explore_tick()

    async def _patrol_tick(self) -> None:
        logger.debug("[behavior] PATROL tick")
        online = self.swarm.list_online_nodes()
        for node in online:
            await self.swarm.sensors_read(node.node_id)

    async def _follow_tick(self) -> None:
        logger.debug("[behavior] FOLLOW tick")
        # Hook: use AI detections (e.g., person) to drive robots
        pass

    async def _guard_tick(self) -> None:
        logger.debug("[behavior] GUARD tick")
        # Hook: watch for anomalies in AI / sensors, trigger alerts
        pass

    async def _explore_tick(self) -> None:
        logger.debug("[behavior] EXPLORE tick")
        # Hook: random walk / coverage strategies via mesh + robots
        pass


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: Neural Coordination (Phase 8)
# ---------------------------------------------------------------------------

@dataclass
class ModelVersionInfo:
    name: str
    version: str
    checksum: str
    size_bytes: int
    roles: List[str]  # e.g. ["vision", "audio", "router"]


class NeuralCoordinator:
    """
    Phase 8: Neural Coordination (controller side)

    - Maintains a distributed model registry (from mesh state + local config)
    - Tracks which node has which model versions
    - Suggests specialization roles for nodes (vision/audio/router/robotics)
    - Uses mesh distributed memory as the "neural genome"
    """

    def __init__(self, swarm: "SwarmManager") -> None:
        self.swarm = swarm
        # node_id -> {model_name: version}
        self.node_models: Dict[str, Dict[str, str]] = {}
        # global registry: model_name -> ModelVersionInfo
        self.registry: Dict[str, ModelVersionInfo] = {}
        self._task: Optional[asyncio.Task] = None

        # Seed registry from MODEL_REGISTRY
        for key, path in MODEL_REGISTRY.items():
            self.registry[key] = ModelVersionInfo(
                name=key,
                version="1.0.0",
                checksum="unknown",
                size_bytes=0,
                roles=["vision"] if "yolo" in key or "mobilenet" in key else ["audio"] if "whisper" in key else [],
            )

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(15.0)
            await self._sync_from_mesh()
            self._infer_node_roles()

    async def _sync_from_mesh(self) -> None:
        """
        Pull neural metadata from mesh_info (which includes neural_models/neural_roles).
        """
        for nid, info in self.swarm.last_mesh_info.items():
            models = info.get("neural_models") or {}
            roles = info.get("neural_roles") or {}
            if models:
                self.node_models[nid] = models
            # roles are hints; we don't override behavior engine here, but we could log them

    def _infer_node_roles(self) -> None:
        """
        Use capabilities + model presence to infer specialization.
        For now, just logs what each node is "good at".
        """
        for nid, node in self.swarm.nodes.items():
            caps = node.capabilities
            models = self.node_models.get(nid, {})
            roles = []
            if any("yolo" in m or "mobilenet" in m for m in models):
                roles.append("vision")
            if any("whisper" in m for m in models):
                roles.append("audio")
            if caps.has_sensors and caps.has_camera:
                roles.append("perception")
            if caps.has_mic and "audio" in roles:
                roles.append("speech")
            if caps.has_gpu or caps.has_npu:
                roles.append("heavy_ai")
            if not roles:
                roles.append("generic")

            logger.debug("[neural] Node %s roles: %s", nid, roles)


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: Swarm manager + AUTONOMY + NEURAL
# ---------------------------------------------------------------------------

class SwarmManager:
    def __init__(self) -> None:
        self.nodes: Dict[str, PhoneNode] = {}
        self._health_task: Optional[asyncio.Task] = None
        self._autonomy_task: Optional[asyncio.Task] = None
        self._running = False

        self.last_sensor_data: Dict[str, Any] = {}
        self.last_camera_frame_jpeg: Dict[str, str] = {}
        self.last_camera_frame_h264: Dict[str, "Image.Image"] = {}
        self.last_thermal_data: Dict[str, Any] = {}
        self.last_voice_status: Dict[str, Any] = {}
        self.last_bt_scan: Dict[str, Any] = {}
        self.last_bt_status: Dict[str, Any] = {}
        self.last_bt_gatt: Dict[str, Any] = {}
        self.last_bt_services: Dict[str, Any] = {}
        self.last_bt_notifications: Dict[str, Any] = {}
        self.last_mesh_info: Dict[str, Any] = {}
        self.cpu_history: Dict[str, List[Tuple[float, float]]] = {}
        self.batt_history: Dict[str, List[Tuple[float, float]]] = {}
        self.last_h264_chunk_size: Dict[str, int] = {}
        self.ai_status: Dict[str, Any] = {}

        # AUTONOMY STATE
        self.policy_config = {
            "min_camera_streams": 1,
            "sensor_poll_interval": 15.0,
            "thermal_poll_interval": 30.0,
        }
        self._last_sensor_policy_run = 0.0
        self._last_thermal_policy_run = 0.0
        self.active_camera_streams: Dict[str, str] = {}  # node_id -> stream_id

        # Phase 5 + 6 + 7 + 8
        self.intel = SwarmIntelligence(self)
        self.behavior = SwarmBehaviorEngine(self, self.intel)
        self.robotics = RoboticsController(self)
        self.neural = NeuralCoordinator(self)

    def register_node(self, node: PhoneNode) -> None:
        logger.info("Registering node: %s", node)
        self.nodes[node.node_id] = node

    def unregister_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            logger.info("Unregistering node: %s", node_id)
            del self.nodes[node_id]
        self.active_camera_streams.pop(node_id, None)

    async def start(self, health_interval: float = 10.0) -> None:
        logger.info("Starting SwarmManager with %d nodes", len(self.nodes))
        self._running = True
        for node in self.nodes.values():
            await node.connect()
        self._health_task = asyncio.create_task(self._health_loop(health_interval))
        self._autonomy_task = asyncio.create_task(self._autonomy_loop())
        self.intel.start()
        self.behavior.start()
        self.neural.start()

    async def stop(self) -> None:
        logger.info("Stopping SwarmManager")
        self._running = False
        for t in (self._health_task, self._autonomy_task):
            if t:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        for node in self.nodes.values():
            await node.disconnect()

    async def _health_loop(self, interval: float) -> None:
        while self._running:
            await asyncio.sleep(interval)
            await self.health_check_all()
            await self.poll_thermal_all()
            await self.poll_voice_all()
            await self.poll_bt_all()
            await self.poll_mesh_all()

    async def _autonomy_loop(self) -> None:
        while self._running:
            await asyncio.sleep(3.0)
            await self.enforce_policies()

    async def health_check_all(self) -> None:
        tasks = [node.health_check() for node in self.nodes.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def poll_thermal_all(self) -> None:
        for node in self.nodes.values():
            try:
                res = await self.run_task_on_node(node.node_id, "thermal_status", {})
                if res.success:
                    self.last_thermal_data[node.node_id] = res.payload
                    cpu = res.payload.get("cpu_load")
                    batt = res.payload.get("battery_pct")
                    t = time.time()
                    if cpu is not None:
                        self.cpu_history.setdefault(node.node_id, []).append((t, float(cpu)))
                        self.cpu_history[node.node_id] = self.cpu_history[node.node_id][-200:]
                    if batt is not None:
                        self.batt_history.setdefault(node.node_id, []).append((t, float(batt)))
                        self.batt_history[node.node_id] = self.batt_history[node.node_id][-200:]
            except Exception:
                pass

    async def poll_voice_all(self) -> None:
        for node in self.nodes.values():
            try:
                res = await self.run_task_on_node(node.node_id, "voice_status", {})
                if res.success:
                    self.last_voice_status[node.node_id] = res.payload
            except Exception:
                pass

    async def poll_bt_all(self) -> None:
        for node in self.nodes.values():
            try:
                res = await self.run_task_on_node(node.node_id, "bt_scan", {})
                if res.success:
                    self.last_bt_scan[node.node_id] = res.payload
                res2 = await self.run_task_on_node(node.node_id, "bt_status", {})
                if res2.success:
                    self.last_bt_status[node.node_id] = res2.payload
                res3 = await self.run_task_on_node(node.node_id, "bt_services", {})
                if res3.success:
                    self.last_bt_services[node.node_id] = res3.payload
            except Exception:
                pass

    async def poll_mesh_all(self) -> None:
        for node in self.nodes.values():
            try:
                res = await self.run_task_on_node(node.node_id, "mesh_info", {})
                if res.success:
                    self.last_mesh_info[node.node_id] = res.payload
            except Exception:
                pass

    def list_online_nodes(self) -> List[PhoneNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]

    # ---------------- AUTONOMY: policy engine ----------------

    async def enforce_policies(self) -> None:
        now = time.time()
        online = self.list_online_nodes()

        # Self-healing: drop camera streams for offline nodes
        for nid in list(self.active_camera_streams.keys()):
            node = self.nodes.get(nid)
            if not node or node.status == NodeStatus.OFFLINE:
                logger.info("[autonomy] Removing camera stream for offline node %s", nid)
                self.active_camera_streams.pop(nid, None)

        # Policy: maintain minimum camera streams
        min_streams = self.policy_config["min_camera_streams"]
        if len(self.active_camera_streams) < min_streams and online:
            needed = min_streams - len(self.active_camera_streams)
            candidates = [n for n in online if n.node_id not in self.active_camera_streams and n.capabilities.has_camera]
            for node in candidates[:needed]:
                stream_id = f"auto-cam-{_generate_task_id(4)}"
                logger.info("[autonomy] Starting camera on %s (stream_id=%s)", node.node_id, stream_id)
                try:
                    res = await self.camera_start(node.node_id, stream_id=stream_id, h264=True, low_latency=True)
                    if res.success:
                        self.active_camera_streams[node.node_id] = stream_id
                except Exception as e:
                    logger.warning("[autonomy] Failed to start camera on %s: %s", node.node_id, e)

        # Policy: periodic sensors_read
        if now - self._last_sensor_policy_run > self.policy_config["sensor_poll_interval"]:
            self._last_sensor_policy_run = now
            logger.debug("[autonomy] Running periodic sensors_read on all online nodes")
            await self.broadcast_task("sensors_read", {})

        # Policy: periodic thermal_status
        if now - self._last_thermal_policy_run > self.policy_config["thermal_poll_interval"]:
            self._last_thermal_policy_run = now
            logger.debug("[autonomy] Running periodic thermal_status on all online nodes")
            await self.broadcast_task("thermal_status", {})

    # ---------------- Task allocator ----------------

    def _select_best_node_for_task(self, candidates: List[PhoneNode], require_camera: bool = False) -> Optional[PhoneNode]:
        scored: List[Tuple[float, PhoneNode]] = []
        for n in candidates:
            if require_camera and not n.capabilities.has_camera:
                continue
            batt = self.last_thermal_data.get(n.node_id, {}).get("battery_pct", 50.0)
            cpu = self.last_thermal_data.get(n.node_id, {}).get("cpu_load", 0.5)
            score = batt - cpu * 50.0
            scored.append((score, n))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    async def run_task_on_node(
        self,
        node_id: str,
        command: str,
        payload: Dict[str, Any],
    ) -> TaskResult:
        node = self.nodes.get(node_id)
        if not node:
            return TaskResult(
                task_id=payload.get("task_id", _generate_task_id()),
                node_id=node_id,
                success=False,
                payload=None,
                error=f"Node {node_id} not found",
            )

        task_id = payload.get("task_id") or _generate_task_id()
        payload = dict(payload)
        payload["task_id"] = task_id

        try:
            resp = await node.send_command(command, payload)
            ok = bool(resp.get("ok", False))
            result_payload = resp.get("result", resp)

            if command == "sensors_read" and ok:
                self.last_sensor_data[node_id] = result_payload
                gps = result_payload.get("sensors", {}).get("gps")
                if gps:
                    node.last_gps = gps

            if command == "camera_start" and ok:
                frame = result_payload.get("frame")
                h264_chunk = result_payload.get("h264_chunk")
                if frame:
                    self.last_camera_frame_jpeg[node_id] = frame
                if h264_chunk:
                    self.last_h264_chunk_size[node_id] = len(base64.b64decode(h264_chunk))
                    img = decode_h264_ts_chunk(h264_chunk)
                    if img is not None:
                        self.last_camera_frame_h264[node_id] = img

            if command == "thermal_status" and ok:
                self.last_thermal_data[node_id] = result_payload

            if command == "voice_status" and ok:
                self.last_voice_status[node_id] = result_payload

            if command == "bt_scan" and ok:
                self.last_bt_scan[node_id] = result_payload

            if command == "bt_status" and ok:
                self.last_bt_status[node_id] = result_payload

            if command == "bt_gatt" and ok:
                self.last_bt_gatt[node_id] = result_payload

            if command == "bt_services" and ok:
                self.last_bt_services[node_id] = result_payload

            if command == "bt_notify" and ok:
                self.last_bt_notifications[node_id] = result_payload

            if command == "mesh_info" and ok:
                self.last_mesh_info[node_id] = result_payload

            if command == "mesh_task" and ok:
                pass

            if command == "voice_recognize" and ok:
                self.last_voice_status[node_id] = result_payload

            if command == "ai_infer" and ok:
                self.ai_status[node_id] = result_payload

            if command == "neural_sync" and ok:
                # Could be used to update NeuralCoordinator directly
                pass

            return TaskResult(
                task_id=task_id,
                node_id=node_id,
                success=ok,
                payload=result_payload,
                error=None if ok else resp.get("error", "Unknown error"),
            )
        except Exception as e:
            logger.exception("Error running task on node %s", node_id)
            return TaskResult(
                task_id=task_id,
                node_id=node_id,
                success=False,
                payload=None,
                error=str(e),
            )

    async def broadcast_task(
        self,
        command: str,
        payload: Dict[str, Any],
        only_online: bool = True,
    ) -> List[TaskResult]:
        nodes = self.list_online_nodes() if only_online else list(self.nodes.values())
        tasks = [
            self.run_task_on_node(node.node_id, command, payload)
            for node in nodes
        ]
        return await asyncio.gather(*tasks)

    async def schedule_task(
        self,
        command: str,
        payload: Dict[str, Any],
        require_camera: bool = False,
    ) -> TaskResult:
        candidates = self.list_online_nodes()
        if not candidates:
            return TaskResult(
                task_id=payload.get("task_id", _generate_task_id()),
                node_id="",
                success=False,
                payload=None,
                error="No online nodes available",
            )
        node = self._select_best_node_for_task(candidates, require_camera=require_camera) or random.choice(candidates)
        return await self.run_task_on_node(node.node_id, command, payload)

    async def camera_start(
        self,
        node_id: str,
        stream_id: Optional[str] = None,
        resolution: Optional[List[int]] = None,
        fps: int = 5,
        fmt: str = "jpeg",
        h264: bool = False,
        low_latency: bool = True,
    ) -> TaskResult:
        stream_id = stream_id or f"cam-{_generate_task_id()}"
        resolution = resolution or [640, 480]
        payload = {
            "stream_id": stream_id,
            "resolution": resolution,
            "fps": fps,
            "format": fmt,
            "h264": h264,
            "low_latency": low_latency,
        }
        res = await self.run_task_on_node(node_id, "camera_start", payload)
        if res.success:
            self.active_camera_streams[node_id] = stream_id
        return res

    async def camera_stop(self, node_id: str, stream_id: str) -> TaskResult:
        payload = {"stream_id": stream_id}
        res = await self.run_task_on_node(node_id, "camera_stop", payload)
        if res.success and self.active_camera_streams.get(node_id) == stream_id:
            self.active_camera_streams.pop(node_id, None)
        return res

    async def sensors_read(self, node_id: str) -> TaskResult:
        return await self.run_task_on_node(node_id, "sensors_read", {})

    async def ai_infer(
        self,
        node_id: str,
        model: str,
        input_type: str,
        input_data: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        payload = {
            "model": model,
            "input_type": input_type,
            "input": input_data,
            "params": params or {},
        }
        return await self.run_task_on_node(node_id, "ai_infer", payload)

    async def voice_status(self, node_id: str) -> TaskResult:
        return await self.run_task_on_node(node_id, "voice_status", {})

    async def voice_recognize(self, node_id: str) -> TaskResult:
        return await self.run_task_on_node(node_id, "voice_recognize", {})

    async def bt_scan(self, node_id: str) -> TaskResult:
        return await self.run_task_on_node(node_id, "bt_scan", {})

    async def bt_status(self, node_id: str) -> TaskResult:
        return await self.run_task_on_node(node_id, "bt_status", {})

    async def bt_services(self, node_id: str) -> TaskResult:
        return await self.run_task_on_node(node_id, "bt_services", {})

    async def bt_gatt(
        self,
        node_id: str,
        mac: str,
        service_uuid: str,
        char_uuid: str,
        op: str,
        data: Optional[str] = None,
    ) -> TaskResult:
        payload = {
            "mac": mac,
            "service_uuid": service_uuid,
            "char_uuid": char_uuid,
            "op": op,
            "data": data,
        }
        return await self.run_task_on_node(node_id, "bt_gatt", payload)

    async def bt_notify(
        self,
        node_id: str,
        mac: str,
        char_uuid: str,
        enable: bool,
    ) -> TaskResult:
        payload = {
            "mac": mac,
            "char_uuid": char_uuid,
            "enable": enable,
        }
        return await self.run_task_on_node(node_id, "bt_notify", payload)

    async def mesh_info(self, node_id: str) -> TaskResult:
        return await self.run_task_on_node(node_id, "mesh_info", {})

    async def mesh_task(
        self,
        node_id: str,
        target_peer_id: str,
        command: str,
        payload: Dict[str, Any],
    ) -> TaskResult:
        p = {
            "target_peer_id": target_peer_id,
            "inner_command": command,
            "inner_payload": payload,
        }
        return await self.run_task_on_node(node_id, "mesh_task", p)

    async def neural_sync(self, node_id: str) -> TaskResult:
        """
        Ask agent for its neural metadata (models, experience summary).
        """
        return await self.run_task_on_node(node_id, "neural_sync", {})


# ---------------------------------------------------------------------------
# AGENT SIDE: Termux helpers
# ---------------------------------------------------------------------------

def termux_json_cmd(cmd: List[str]) -> Dict[str, Any]:
    try:
        out = subprocess.check_output(cmd)
        return json.loads(out.decode())
    except Exception:
        return {}


def termux_capture_photo(path: str) -> Optional[str]:
    try:
        subprocess.check_output(["termux-camera-photo", path])
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return data
    except Exception as e:
        logger.warning("[agent] camera-photo failed: %s", e)
        return None


def termux_battery_status() -> Dict[str, Any]:
    return termux_json_cmd(["termux-battery-status"])


def termux_cpu_load() -> float:
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = line.split()
        if parts[0] != "cpu":
            return 0.0
        vals = list(map(int, parts[1:]))
        idle = vals[3]
        total = sum(vals)
        time.sleep(0.1)
        with open("/proc/stat", "r") as f:
            line2 = f.readline()
        parts2 = line2.split()
        vals2 = list(map(int, parts2[1:]))
        idle2 = vals2[3]
        total2 = sum(vals2)
        idle_delta = idle2 - idle
        total_delta = total2 - total
        if total_delta <= 0:
            return 0.0
        return 1.0 - (idle_delta / total_delta)
    except Exception:
        return 0.0


async def termux_read_real_sensors() -> Dict[str, Any]:
    sensors = termux_json_cmd(["termux-sensor", "-n", "1"])
    gps = termux_json_cmd(["termux-location", "--provider", "gps"])
    accel = sensors.get("accelerometer", [{}])[0].get("values", [0.0, 0.0, 9.81])
    gyro = sensors.get("gyroscope", [{}])[0].get("values", [0.0, 0.0, 0.0])
    light = sensors.get("light", [{}])[0].get("light", None)
    return {
        "timestamp": time.time(),
        "sensors": {
            "accel": accel,
            "gyro": gyro,
            "gps": {
                "lat": gps.get("latitude"),
                "lon": gps.get("longitude"),
                "alt": gps.get("altitude"),
                "acc": gps.get("accuracy"),
            },
            "light": light,
        },
    }


def termux_thermal_status() -> Dict[str, Any]:
    batt = termux_battery_status()
    cpu_load = termux_cpu_load()
    temp_c = batt.get("temperature")
    return {
        "cpu_load": cpu_load,
        "temp_c": temp_c,
        "battery_pct": batt.get("percentage"),
    }


# ---------------------------------------------------------------------------
# AGENT SIDE: AI pipelines (Phase 4)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "yolo": "/data/data/com.termux/files/home/models/yolo.onnx",
    "mobilenet": "/data/data/com.termux/files/home/models/mobilenet.onnx",
    "whisper": "/data/data/com.termux/files/home/models/whisper.onnx",
}

YOLO_LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
]
MOBILENET_LABELS = [
    "background", "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich",
]


def ai_preprocess(input_type: str, input_data: Any, params: Dict[str, Any]) -> Any:
    if np is None or Image is None:
        return input_data

    if input_type == "image_b64":
        try:
            img_bytes = base64.b64decode(input_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            size = params.get("size", (640, 640))
            img = img.resize(size)
            arr = np.array(img).astype("float32") / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, 0)
            return arr
        except Exception:
            return input_data

    if input_type == "audio_pcm":
        return input_data

    if input_type == "text":
        return input_data

    return input_data


def _yolo_postprocess(raw_outputs: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    conf_th = float(params.get("conf_threshold", 0.4))
    if not isinstance(raw_outputs, dict) or raw_outputs.get("stub"):
        return raw_outputs
    outputs = raw_outputs.get("outputs")
    if not outputs:
        return {"detections": []}
    arr = outputs[0]
    try:
        arr = np.array(arr)
    except Exception:
        return {"detections": []}
    detections = []
    for row in arr:
        if len(row) < 6:
            continue
        x, y, w, h = row[0:4]
        obj_conf = row[4]
        class_scores = row[5:]
        if obj_conf < conf_th:
            continue
        cls_id = int(np.argmax(class_scores))
        cls_conf = class_scores[cls_id] * obj_conf
        if cls_conf < conf_th:
            continue
        label = YOLO_LABELS[cls_id] if cls_id < len(YOLO_LABELS) else f"class_{cls_id}"
        detections.append({
            "bbox": [float(x), float(y), float(w), float(h)],
            "label": label,
            "score": float(cls_conf),
        })
    return {"detections": detections}


def _mobilenet_postprocess(raw_outputs: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    top_k = int(params.get("top_k", 3))
    if not isinstance(raw_outputs, dict) or raw_outputs.get("stub"):
        return raw_outputs
    outputs = raw_outputs.get("outputs")
    if not outputs:
        return {"classes": []}
    arr = outputs[0]
    try:
        arr = np.array(arr)
    except Exception:
        return {"classes": []}
    idxs = np.argsort(arr)[::-1][:top_k]
    classes = []
    for idx in idxs:
        label = MOBILENET_LABELS[idx] if idx < len(MOBILENET_LABELS) else f"class_{idx}"
        prob = float(arr[idx])
        classes.append({"label": label, "score": prob})
    return {"classes": classes}


def _whisper_postprocess(raw_outputs: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return raw_outputs


def ai_postprocess(model: str, raw_outputs: Any, params: Dict[str, Any]) -> Any:
    m = model.lower()
    if "yolo" in m:
        return _yolo_postprocess(raw_outputs, params)
    if "mobilenet" in m:
        return _mobilenet_postprocess(raw_outputs, params)
    if "whisper" in m:
        return _whisper_postprocess(raw_outputs, params)
    return raw_outputs


def onnx_infer_real(model_key: str, input_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    if ort is None:
        return {"stub": True, "message": "onnxruntime not installed on agent"}
    model_path = MODEL_REGISTRY.get(model_key, model_key)
    try:
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: input_data})
        return {"stub": False, "outputs": [o.tolist() if hasattr(o, "tolist") else o for o in output]}
    except Exception as e:
        return {"stub": True, "error": str(e)}


# ---------------------------------------------------------------------------
# AGENT SIDE: Voice recognition (async-ish)
# ---------------------------------------------------------------------------

VOICE_STATE = {
    "listening": False,
    "last_text": None,
    "wake_word": "computer",
    "last_wake": None,
}


async def agent_voice_recognize_streaming() -> Dict[str, Any]:
    if VOICE_BACKEND != "speech_recognition":
        return {"stub": True, "error": "speech_recognition not installed"}

    def _record_and_recognize():
        try:
            r = sr.Recognizer()  # type: ignore
            with sr.Microphone() as source:  # type: ignore
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
            text = r.recognize_google(audio)  # type: ignore
            return text
        except Exception as e:
            return f"__ERROR__:{e}"

    loop = asyncio.get_event_loop()
    VOICE_STATE["listening"] = True
    text = await loop.run_in_executor(None, _record_and_recognize)
    VOICE_STATE["listening"] = False

    if text.startswith("__ERROR__"):
        return {"stub": True, "error": text[9:]}

    VOICE_STATE["last_text"] = text
    wake = VOICE_STATE.get("wake_word") or ""
    if wake and wake.lower() in text.lower():
        VOICE_STATE["last_wake"] = time.time()

    return {"stub": False, "text": text, "wake_detected": wake.lower() in text.lower()}


# ---------------------------------------------------------------------------
# AGENT SIDE: Bluetooth (scan + GATT + services + notifications)
# ---------------------------------------------------------------------------

def termux_bt_scan() -> Dict[str, Any]:
    for cmd in (["termux-bluetooth-scan"], ["termux-bluetooth-scaninfo"]):
        try:
            out = subprocess.check_output(cmd)
            data = json.loads(out.decode())
            return {"devices": data}
        except Exception:
            continue
    return {"devices": []}


def termux_bt_status() -> Dict[str, Any]:
    return {"enabled": True, "scanning": False}


async def agent_bt_services(payload: Dict[str, Any]) -> Dict[str, Any]:
    if bleak is None:
        return {"stub": True, "error": "bleak not installed"}
    mac = payload.get("mac")
    from bleak import BleakClient  # type: ignore
    try:
        async with BleakClient(mac) as client:
            services = await client.get_services()
            out = []
            for s in services:
                chars = []
                for c in s.characteristics:
                    chars.append({
                        "uuid": str(c.uuid),
                        "props": list(c.properties),
                    })
                out.append({
                    "uuid": str(s.uuid),
                    "chars": chars,
                })
            return {"stub": False, "services": out}
    except Exception as e:
        return {"stub": True, "error": str(e)}


async def agent_bt_gatt(payload: Dict[str, Any]) -> Dict[str, Any]:
    if bleak is None:
        return {"stub": True, "error": "bleak not installed"}
    mac = payload.get("mac")
    char_uuid = payload.get("char_uuid")
    op = payload.get("op")
    data = payload.get("data")
    from bleak import BleakClient  # type: ignore
    try:
        async with BleakClient(mac) as client:
            if op == "read":
                val = await client.read_gatt_char(char_uuid)
                return {"stub": False, "op": "read", "value": list(val)}
            elif op == "write":
                if data is None:
                    return {"stub": True, "error": "no data for write"}
                raw = bytes.fromhex(data)
                await client.write_gatt_char(char_uuid, raw)
                return {"stub": False, "op": "write", "status": "ok"}
            else:
                return {"stub": True, "error": f"unknown op {op}"}
    except Exception as e:
        return {"stub": True, "error": str(e)}


BT_NOTIFY_CALLBACKS: Dict[str, Any] = {}


async def agent_bt_notify(payload: Dict[str, Any]) -> Dict[str, Any]:
    enable = bool(payload.get("enable", False))
    mac = payload.get("mac")
    char_uuid = payload.get("char_uuid")
    key = f"{mac}:{char_uuid}"
    if enable:
        BT_NOTIFY_CALLBACKS[key] = True
        return {"stub": False, "status": "subscribed"}
    else:
        BT_NOTIFY_CALLBACKS.pop(key, None)
        return {"stub": False, "status": "unsubscribed"}


# ---------------------------------------------------------------------------
# AGENT SIDE: H.264 streaming (ffmpeg hook)
# ---------------------------------------------------------------------------

async def agent_h264_stream(ws, stream_id: str, resolution: List[int], fps: int, low_latency: bool) -> None:
    width, height = resolution
    preset = "ultrafast" if low_latency else "veryfast"
    cmd = [
        "ffmpeg",
        "-f", "android_camera",
        "-i", "0",
        "-vf", f"scale={width}:{height}",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", preset,
        "-tune", "zerolatency",
        "-f", "mpegts",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.warning("[agent] ffmpeg H.264 start failed: %s", e)
        return

    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            msg = {
                "ok": True,
                "result": {
                    "stream_id": stream_id,
                    "h264_chunk": base64.b64encode(chunk).decode(),
                },
                "error": None,
                "task_id": None,
            }
            await ws.send(json.dumps(msg))
            await asyncio.sleep(0)
    except Exception:
        pass
    finally:
        try:
            proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AGENT SIDE: Mesh networking
# ---------------------------------------------------------------------------

MESH_PEERS: Dict[str, Tuple[str, int]] = {}
MESH_PORT = 9876
MESH_UDP_PORT = 9877
MESH_STATE: Dict[str, Any] = {
    "version": 0,
    "data": {},
    # Phase 8: neural metadata
    "neural_models": {},  # model_name -> {"version": str, "checksum": str}
    "neural_roles": {},   # node_id -> [roles]
}
MESH_ID = None


async def mesh_udp_beacon_loop(agent_id: str) -> None:
    global MESH_ID
    MESH_ID = agent_id
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setblocking(False)
    while True:
        msg = json.dumps({"id": agent_id, "port": MESH_PORT}).encode()
        try:
            sock.sendto(msg, ("255.255.255.255", MESH_UDP_PORT))
        except Exception:
            pass
        await asyncio.sleep(5)


async def mesh_udp_listener_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", MESH_UDP_PORT))
    sock.setblocking(False)
    loop = asyncio.get_event_loop()
    while True:
        try:
            data, addr = await loop.run_in_executor(None, sock.recvfrom, 4096)
        except Exception:
            await asyncio.sleep(1)
            continue
        try:
            msg = json.loads(data.decode())
            peer_id = msg.get("id")
            port = msg.get("port")
            if peer_id and port:
                MESH_PEERS[peer_id] = (addr[0], int(port))
        except Exception:
            pass


async def mesh_info() -> Dict[str, Any]:
    routes = [{"dest": k, "next_hop": v[0], "port": v[1]} for k, v in MESH_PEERS.items()]
    return {
        "peers": [{"id": k, "host": v[0], "port": v[1]} for k, v in MESH_PEERS.items()],
        "routes": routes,
        "state_version": MESH_STATE["version"],
        "neural_models": MESH_STATE.get("neural_models", {}),
        "neural_roles": MESH_STATE.get("neural_roles", {}),
    }


async def mesh_gossip_loop() -> None:
    while True:
        await asyncio.sleep(10)
        for pid, (host, port) in MESH_PEERS.items():
            try:
                reader, writer = await asyncio.open_connection(host, port)
                msg = {
                    "type": "gossip",
                    "from": MESH_ID,
                    "state": MESH_STATE,
                }
                writer.write((json.dumps(msg) + "\n").encode())
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            except Exception:
                continue


async def mesh_tcp_server_loop() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            line = await reader.readline()
            if not line:
                writer.close()
                await writer.wait_closed()
                return
            msg = json.loads(line.decode())
            if msg.get("type") == "gossip":
                incoming = msg.get("state", {})
                ver = incoming.get("version", 0)
                if ver > MESH_STATE["version"]:
                    MESH_STATE["version"] = ver
                    MESH_STATE["data"] = incoming.get("data", {})
                    MESH_STATE["neural_models"] = incoming.get("neural_models", {})
                    MESH_STATE["neural_roles"] = incoming.get("neural_roles", {})
            elif msg.get("type") == "task":
                # Hook: remote task execution
                pass
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "0.0.0.0", MESH_PORT)
    async with server:
        await server.serve_forever()


async def mesh_task_forward(target_peer_id: str, inner_command: str, inner_payload: Dict[str, Any]) -> Dict[str, Any]:
    peer = MESH_PEERS.get(target_peer_id)
    if not peer:
        return {"stub": True, "error": f"peer {target_peer_id} not found"}
    host, port = peer
    try:
        reader, writer = await asyncio.open_connection(host, port)
        msg = {
            "type": "task",
            "from": MESH_ID,
            "command": inner_command,
            "payload": inner_payload,
        }
        writer.write((json.dumps(msg) + "\n").encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        return {"stub": False, "status": "forwarded"}
    except Exception as e:
        return {"stub": True, "error": str(e)}


# ---------------------------------------------------------------------------
# AGENT SIDE: Local autonomy scheduler (Phase 2)
# ---------------------------------------------------------------------------

LOCAL_JOBS = [
    {
        "name": "periodic_sensors",
        "interval": 20.0,
        "last_run": 0.0,
        "command": "sensors_read",
        "payload": {},
    },
    {
        "name": "periodic_thermal",
        "interval": 40.0,
        "last_run": 0.0,
        "command": "thermal_status",
        "payload": {},
    },
]


async def local_autonomy_loop() -> None:
    """
    Agent-side self-tasking:
    - Runs jobs even if controller is offline.
    - For now, just logs results locally.
    """
    while True:
        now = time.time()
        for job in LOCAL_JOBS:
            if now - job["last_run"] >= job["interval"]:
                job["last_run"] = now
                cmd = job["command"]
                payload = job["payload"]
                try:
                    if cmd == "sensors_read":
                        res = await termux_read_real_sensors()
                        logger.debug("[agent-autonomy] sensors_read -> %s", res)
                    elif cmd == "thermal_status":
                        res = termux_thermal_status()
                        logger.debug("[agent-autonomy] thermal_status -> %s", res)
                except Exception as e:
                    logger.warning("[agent-autonomy] job %s failed: %s", job["name"], e)
        await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# AGENT SIDE: NeuralAgent (Phase 8)
# ---------------------------------------------------------------------------

NEURAL_EXPERIENCE_BUFFER: List[Dict[str, Any]] = []  # simple in-memory log


def neural_log_experience(model: str, input_type: str, context: Dict[str, Any], output: Any) -> None:
    """
    Log a single AI inference as "experience".
    """
    if len(NEURAL_EXPERIENCE_BUFFER) > 500:
        NEURAL_EXPERIENCE_BUFFER.pop(0)
    NEURAL_EXPERIENCE_BUFFER.append({
        "ts": time.time(),
        "model": model,
        "input_type": input_type,
        "context": context,
        "output": output,
    })


def neural_build_summary() -> Dict[str, Any]:
    """
    Build a tiny summary of recent experiences for sharing via mesh.
    """
    by_model: Dict[str, int] = {}
    for e in NEURAL_EXPERIENCE_BUFFER[-100:]:
        m = e["model"]
        by_model[m] = by_model.get(m, 0) + 1
    return {
        "count": len(NEURAL_EXPERIENCE_BUFFER),
        "recent_by_model": by_model,
    }


def neural_update_mesh_metadata() -> None:
    """
    Update MESH_STATE.neural_models / neural_roles for this agent.
    """
    # models: from MODEL_REGISTRY
    models_meta = {}
    for name, path in MODEL_REGISTRY.items():
        models_meta[name] = {
            "version": "1.0.0",
            "checksum": "unknown",
        }
    MESH_STATE["neural_models"] = models_meta

    # roles: simple heuristic
    roles = []
    roles.append("vision") if any("yolo" in k or "mobilenet" in k for k in MODEL_REGISTRY) else None
    roles.append("audio") if any("whisper" in k for k in MODEL_REGISTRY) else None
    if not roles:
        roles.append("generic")
    MESH_STATE["neural_roles"][MESH_ID or "unknown"] = roles
    MESH_STATE["version"] += 1


async def neural_periodic_loop() -> None:
    """
    Periodically refresh mesh neural metadata.
    """
    while True:
        neural_update_mesh_metadata()
        await asyncio.sleep(30)


# ---------------------------------------------------------------------------
# AGENT SIDE: WebSocket server
# ---------------------------------------------------------------------------

START_TIME_AGENT = time.time()


async def agent_handle_client(ws: "websockets.WebSocketServerProtocol") -> None:
    logger.info("[agent] Client connected: %s", ws.remote_address)
    try:
        async for raw in ws:
            try:
                req = json.loads(raw)
            except json.JSONDecodeError as e:
                await ws.send(json.dumps({
                    "ok": False,
                    "result": None,
                    "error": f"Invalid JSON: {e}",
                    "task_id": None,
                }))
                continue
            resp = await agent_handle_request(ws, req)
            if resp is not None:
                await ws.send(json.dumps(resp))
    except websockets.ConnectionClosed:
        logger.info("[agent] Client disconnected: %s", ws.remote_address)


async def agent_handle_request(ws, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    command = req.get("command")
    payload = req.get("payload") or {}
    task_id = req.get("task_id")

    try:
        if command == "health_check":
            uptime_sec = time.time() - START_TIME_AGENT
            batt = termux_battery_status()
            result = {
                "node_id": "termux-python-agent",
                "uptime_sec": uptime_sec,
                "battery_pct": batt.get("percentage"),
                "temp_c": batt.get("temperature"),
                "caps": {
                    "cpu_cores": 4,
                    "ram_mb": 2048,
                    "has_npu": False,
                    "has_gpu": False,
                    "has_camera": True,
                    "has_mic": True,
                    "has_sensors": True,
                    "tags": ["termux", "python"],
                },
            }
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "compute":
            expr = str(payload.get("expr", "1+1"))
            try:
                value = eval(expr, {"__builtins__": {}})
                result = {"value": value}
                return {"ok": True, "result": result, "error": None, "task_id": task_id}
            except Exception as e:
                return {"ok": False, "result": None, "error": str(e), "task_id": task_id}

        if command == "sensors_read":
            result = await termux_read_real_sensors()
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "thermal_status":
            result = termux_thermal_status()
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "ai_infer":
            model = str(payload.get("model", ""))
            input_type = payload.get("input_type")
            input_data = payload.get("input")
            params = payload.get("params") or {}
            pre = ai_preprocess(input_type, input_data, params)
            raw = onnx_infer_real(model, pre, params)
            post = ai_postprocess(model, raw, params)
            result = {
                "model": model,
                "input_type": input_type,
                "preprocessed": True,
                "output": post,
            }
            # Phase 8: log experience
            neural_log_experience(model, input_type, {"params": params}, post)
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "camera_start":
            stream_id = payload.get("stream_id") or f"cam-{_generate_task_id()}"
            resolution = payload.get("resolution") or [640, 480]
            fps = int(payload.get("fps", 5))
            h264 = bool(payload.get("h264", False))
            low_latency = bool(payload.get("low_latency", True))
            if h264:
                asyncio.create_task(agent_h264_stream(ws, stream_id, resolution, fps, low_latency))
                result = {"stream_id": stream_id, "status": "h264_streaming"}
                return {"ok": True, "result": result, "error": None, "task_id": task_id}
            else:
                tmp = "/data/data/com.termux/files/home/cam.jpg"
                img = termux_capture_photo(tmp)
                result = {
                    "stream_id": stream_id,
                    "frame": img,
                    "status": "snapshot",
                }
                return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "camera_stop":
            stream_id = payload.get("stream_id")
            result = {"stream_id": stream_id, "status": "stopped"}
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "voice_status":
            return {"ok": True, "result": VOICE_STATE, "error": None, "task_id": task_id}

        if command == "voice_recognize":
            res = await agent_voice_recognize_streaming()
            return {"ok": True, "result": res, "error": None, "task_id": task_id}

        if command == "bt_scan":
            result = termux_bt_scan()
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "bt_status":
            result = termux_bt_status()
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "bt_gatt":
            result = await agent_bt_gatt(payload)
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "bt_services":
            result = await agent_bt_services(payload)
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "bt_notify":
            result = await agent_bt_notify(payload)
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "mesh_info":
            result = await mesh_info()
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "mesh_task":
            target_peer_id = payload.get("target_peer_id")
            inner_command = payload.get("inner_command")
            inner_payload = payload.get("inner_payload") or {}
            result = await mesh_task_forward(target_peer_id, inner_command, inner_payload)
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        if command == "neural_sync":
            """
            Return neural metadata + experience summary for this node.
            """
            result = {
                "models": {name: {"version": "1.0.0"} for name in MODEL_REGISTRY},
                "experience_summary": neural_build_summary(),
            }
            return {"ok": True, "result": result, "error": None, "task_id": task_id}

        result = {
            "echo": {
                "command": command,
                "payload": payload,
            }
        }
        return {"ok": True, "result": result, "error": None, "task_id": task_id}

    except Exception as e:
        logger.exception("[agent] Error handling request")
        return {"ok": False, "result": None, "error": str(e), "task_id": task_id}


async def agent_main(host: str = "0.0.0.0", port: int = 8765) -> None:
    if websockets is None:
        raise RuntimeError("websockets not installed; run `pip install websockets` in Termux")
    agent_id = f"agent-{_generate_task_id(6)}"
    logger.info("[agent] Starting Termux agent %s on %s:%d", agent_id, host, port)
    async with websockets.serve(agent_handle_client, host, port, ping_interval=10, ping_timeout=10):
        await asyncio.gather(
            asyncio.Future(),
            mesh_udp_beacon_loop(agent_id),
            mesh_udp_listener_loop(),
            mesh_tcp_server_loop(),
            mesh_gossip_loop(),
            local_autonomy_loop(),
            neural_periodic_loop(),
        )


# ---------------------------------------------------------------------------
# CONTROLLER SIDE: GUI (Phase 3 evolution)
# ---------------------------------------------------------------------------

class SwarmGUI:
    def __init__(self, swarm: SwarmManager) -> None:
        if tk is None or Image is None or ImageTk is None:
            raise RuntimeError("Tkinter + Pillow required for GUI")

        self.swarm = swarm
        self.root = tk.Tk()
        self.root.title("Phone Swarm Cockpit")

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT: Node list + health summary
        left = ttk.Frame(self.main_frame)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Nodes", font=("Arial", 12, "bold")).pack(anchor="w")
        self.node_list = tk.Listbox(left, height=10)
        self.node_list.pack(fill=tk.X)
        self.node_list.bind("<<ListboxSelect>>", self.on_node_select)

        self.thermal_label = ttk.Label(left, text="Thermal: N/A")
        self.thermal_label.pack(anchor="w", pady=5)

        self.bt_status_label = ttk.Label(left, text="Bluetooth: N/A")
        self.bt_status_label.pack(anchor="w", pady=5)

        self.mesh_label = ttk.Label(left, text="Mesh: N/A")
        self.mesh_label.pack(anchor="w", pady=5)

        self.h264_label = ttk.Label(left, text="H.264: N/A")
        self.h264_label.pack(anchor="w", pady=5)

        self.ai_label = ttk.Label(left, text="AI: N/A")
        self.ai_label.pack(anchor="w", pady=5)

        self.neural_label = ttk.Label(left, text="Neural: N/A")
        self.neural_label.pack(anchor="w", pady=5)

        # CENTER: Sensor data + voice + BLE + health dashboard
        center = ttk.Frame(self.main_frame)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(center, text="Sensor Data", font=("Arial", 12, "bold")).pack(anchor="w")
        self.sensor_box = tk.Text(center, height=8, width=60)
        self.sensor_box.pack(fill=tk.BOTH, expand=True)

        self.voice_label = ttk.Label(center, text="Voice: N/A")
        self.voice_label.pack(anchor="w", pady=5)

        ttk.Label(center, text="Bluetooth Devices", font=("Arial", 12, "bold")).pack(anchor="w")
        self.bt_box = tk.Text(center, height=6, width=60)
        self.bt_box.pack(fill=tk.BOTH, expand=True)

        ttk.Label(center, text="CPU/Battery Graph (Node Health Dashboard)", font=("Arial", 12, "bold")).pack(anchor="w")
        self.graph_canvas = tk.Canvas(center, width=400, height=150, bg="black")
        self.graph_canvas.pack(fill=tk.X)

        # RIGHT: Map + camera wall
        right = ttk.Frame(self.main_frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Swarm Map (2D, 3D hooks)", font=("Arial", 12, "bold")).pack(anchor="w")
        self.map_canvas = tk.Canvas(right, width=300, height=200, bg="black")
        self.map_canvas.pack()

        ttk.Label(right, text="Camera Wall (JPEG + H.264)", font=("Arial", 12, "bold")).pack(anchor="w")
        self.cam_frame = ttk.Frame(right)
        self.cam_frame.pack()

        self.cam_labels: List[ttk.Label] = []
        for _ in range(4):
            lbl = ttk.Label(self.cam_frame)
            lbl.pack(side=tk.LEFT, padx=2)
            self.cam_labels.append(lbl)

        self.selected_node_id: Optional[str] = None
        self.root.after(500, self.update_gui)

    def on_node_select(self, event) -> None:
        sel = self.node_list.curselection()
        if not sel:
            self.selected_node_id = None
            return
        idx = sel[0]
        node_ids = list(self.swarm.nodes.keys())
        if idx < len(node_ids):
            self.selected_node_id = node_ids[idx]

    def draw_graph(self, nid: str) -> None:
        self.graph_canvas.delete("all")
        self.graph_canvas.create_rectangle(0, 0, 400, 150, fill="black")
        cpu_hist = self.swarm.cpu_history.get(nid, [])
        batt_hist = self.swarm.batt_history.get(nid, [])
        if not cpu_hist and not batt_hist:
            return
        if cpu_hist:
            t0 = cpu_hist[0][0]
        elif batt_hist:
            t0 = batt_hist[0][0]
        else:
            t0 = time.time()

        def norm_x(t):
            return 10 + (t - t0) * 5

        def norm_y_cpu(v):
            return 140 - v * 100

        def norm_y_batt(v):
            return 140 - v

        last = None
        for t, v in cpu_hist:
            x = norm_x(t)
            y = norm_y_cpu(v)
            if last is not None:
                self.graph_canvas.create_line(last[0], last[1], x, y, fill="green")
            last = (x, y)
        last = None
        for t, v in batt_hist:
            x = norm_x(t)
            y = norm_y_batt(v)
            if last is not None:
                self.graph_canvas.create_line(last[0], last[1], x, y, fill="yellow")
            last = (x, y)

    def update_gui(self) -> None:
        self.node_list.delete(0, tk.END)
        for nid, node in self.swarm.nodes.items():
            status = node.status.name
            self.node_list.insert(tk.END, f"{nid} [{status}]")

        nid = self.selected_node_id
        if nid and nid in self.swarm.nodes:
            sensors = self.swarm.last_sensor_data.get(nid)
            self.sensor_box.delete("1.0", tk.END)
            if sensors:
                self.sensor_box.insert(tk.END, json.dumps(sensors, indent=2))
            else:
                self.sensor_box.insert(tk.END, "No sensor data yet")

            thermal = self.swarm.last_thermal_data.get(nid)
            if thermal:
                cpu = thermal.get("cpu_load") or 0.0
                temp = thermal.get("temp_c")
                batt = thermal.get("battery_pct")
                self.thermal_label.config(
                    text=f"Thermal: CPU {cpu:.2f}, Temp {temp}, Batt {batt}%"
                )
            else:
                self.thermal_label.config(text="Thermal: N/A")

            voice = self.swarm.last_voice_status.get(nid)
            if voice:
                if isinstance(voice, dict) and "result" in voice:
                    v = voice["result"]
                else:
                    v = voice
                self.voice_label.config(
                    text=f"Voice: listening={v.get('listening')} last_text={v.get('last_text') or v.get('text')}"
                )
            else:
                self.voice_label.config(text="Voice: N/A")

            bt_status = self.swarm.last_bt_status.get(nid)
            if bt_status:
                self.bt_status_label.config(
                    text=f"Bluetooth: enabled={bt_status.get('enabled')} scanning={bt_status.get('scanning')}"
                )
            else:
                self.bt_status_label.config(text="Bluetooth: N/A")

            bt_scan = self.swarm.last_bt_scan.get(nid)
            self.bt_box.delete("1.0", tk.END)
            if bt_scan and "devices" in bt_scan:
                self.bt_box.insert(tk.END, json.dumps(bt_scan["devices"], indent=2))
            else:
                self.bt_box.insert(tk.END, "No Bluetooth devices seen")

            mesh = self.swarm.last_mesh_info.get(nid)
            if mesh:
                self.mesh_label.config(
                    text=f"Mesh peers: {len(mesh.get('peers', []))} routes: {len(mesh.get('routes', []))}"
                )
            else:
                self.mesh_label.config(text="Mesh: N/A")

            ai = self.swarm.ai_status.get(nid)
            if ai:
                model = ai.get("model")
                out = ai.get("output")
                summary = ""
                if isinstance(out, dict):
                    if "detections" in out:
                        summary = f"{len(out['detections'])} detections"
                    elif "classes" in out:
                        summary = ", ".join([c["label"] for c in out["classes"]])
                    else:
                        summary = "ok"
                self.ai_label.config(text=f"AI: model={model} {summary}")
            else:
                self.ai_label.config(text="AI: N/A")

            # Neural info (Phase 8): show from mesh_info if available
            mesh_info_node = self.swarm.last_mesh_info.get(nid, {})
            neural_models = mesh_info_node.get("neural_models", {})
            neural_roles = mesh_info_node.get("neural_roles", {}).get(nid, [])
            self.neural_label.config(
                text=f"Neural: models={list(neural_models.keys())} roles={neural_roles}"
            )

            self.draw_graph(nid)

            h264_size = self.swarm.last_h264_chunk_size.get(nid)
            if h264_size is not None:
                self.h264_label.config(text=f"H.264 last chunk: {h264_size} bytes")
            else:
                self.h264_label.config(text="H.264: no chunks yet")

        # Map: 2D projection + mesh overlay hooks
        self.map_canvas.delete("all")
        self.map_canvas.create_rectangle(0, 0, 300, 200, fill="black")
        for nid2, node in self.swarm.nodes.items():
            gps = node.last_gps
            if gps and gps.get("lat") is not None and gps.get("lon") is not None:
                lat = gps["lat"] or 0
                lon = gps["lon"] or 0
                x = 150 + lon * 2
                y = 100 - lat * 2
                self.map_canvas.create_oval(x-4, y-4, x+4, y+4, fill="green")
                self.map_canvas.create_text(x+10, y, text=nid2, anchor="w", fill="white")

        # Camera wall: show up to 4 nodes
        online_nodes = list(self.swarm.nodes.keys())[:4]
        for idx, lbl in enumerate(self.cam_labels):
            if idx < len(online_nodes):
                nid2 = online_nodes[idx]
                img_obj = None
                if nid2 in self.swarm.last_camera_frame_h264:
                    img_obj = self.swarm.last_camera_frame_h264[nid2]
                elif nid2 in self.swarm.last_camera_frame_jpeg:
                    try:
                        frame_b64 = self.swarm.last_camera_frame_jpeg[nid2]
                        img_data = base64.b64decode(frame_b64)
                        img_obj = Image.open(io.BytesIO(img_data))
                    except Exception:
                        img_obj = None
                if img_obj is not None:
                    img = img_obj.resize((160, 120))
                    tk_img = ImageTk.PhotoImage(img)
                    lbl.config(image=tk_img, text="")
                    lbl.image = tk_img
                else:
                    lbl.config(text=f"{nid2}\n(no image)", image="")
            else:
                lbl.config(text="", image="")

        self.root.after(500, self.update_gui)

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# CONTROLLER DEMO (GUI + asyncio in background)
# ---------------------------------------------------------------------------

async def controller_async_main(swarm: SwarmManager, wifi_ip: Optional[str], wifi_port: int) -> None:
    usb_node = UsbPhoneNode(adb_serial="FAKE_SERIAL_1234")
    swarm.register_node(usb_node)

    if wifi_ip and websockets is not None:
        wifi_node = WifiPhoneNode(host=wifi_ip, port=wifi_port)
        swarm.register_node(wifi_node)

    await swarm.start(health_interval=10.0)
    await asyncio.sleep(2)

    await swarm.broadcast_task("health_check", {})
    await swarm.broadcast_task("sensors_read", {})
    await swarm.broadcast_task("thermal_status", {})
    await swarm.broadcast_task("voice_status", {})
    await swarm.broadcast_task("bt_scan", {})
    await swarm.broadcast_task("bt_status", {})
    await swarm.broadcast_task("bt_services", {})
    await swarm.broadcast_task("mesh_info", {})
    await swarm.broadcast_task("neural_sync", {})

    online = swarm.list_online_nodes()
    for node in online[:4]:
        await swarm.camera_start(node.node_id, h264=True, low_latency=True)

    swarm.behavior.set_mode(SwarmMode.PATROL)

    while True:
        await asyncio.sleep(1)


def run_controller_with_gui(wifi_ip: Optional[str], wifi_port: int, use_gui: bool) -> None:
    swarm = SwarmManager()

    loop = asyncio.new_event_loop()

    def loop_runner():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(controller_async_main(swarm, wifi_ip, wifi_port))

    import threading
    t = threading.Thread(target=loop_runner, daemon=True)
    t.start()

    if use_gui and tk is not None and Image is not None and ImageTk is not None:
        gui = SwarmGUI(swarm)
        try:
            gui.run()
        finally:
            loop.call_soon_threadsafe(loop.stop)
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            loop.call_soon_threadsafe(loop.stop)


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phone swarm organism")
    parser.add_argument("--mode", choices=["controller", "agent"], default="controller")
    parser.add_argument("--wifi-ip", type=str, default=None)
    parser.add_argument("--wifi-port", type=int, default=8765)
    parser.add_argument("--agent-host", type=str, default="0.0.0.0")
    parser.add_argument("--agent-port", type=int, default=8765)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    if args.mode == "controller":
        run_controller_with_gui(args.wifi_ip, args.wifi_port, use_gui=not args.no_gui)
    else:
        try:
            asyncio.run(agent_main(args.agent_host, args.agent_port))
        except KeyboardInterrupt:
            logger.info("Agent interrupted by user")


if __name__ == "__main__":
    main()
