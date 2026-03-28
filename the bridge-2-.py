"""
Borg Hybrid Swarm Organism - Single File

Features:
- Cross-platform HybridBridge (Windows/Linux, WSL, sockets, shared files)
- Local + remote backends
- UIAutomation (Windows-only, auto-detected) with auto-click & popup suppression
- Vision (OCR + screenshot, optional)
- AIOrgan (role-based, persistent Q-table per node+role)
- AutonomousController (scout/analyst/archivist/commander)
- SwarmBus with pluggable backends:
    - file
    - Redis
    - ZeroMQ
    - NATS (stub)
    - raw UDP broadcast
- SwarmCoordinator:
    - heartbeats
    - simple leader election
    - anomaly consensus skeleton
- Auto-elevation on ALL Windows nodes
- COM initialization on all relevant threads for UIAutomation
"""

import os
import json
import time
import socket
import threading
import platform
import subprocess
from pathlib import Path
import ctypes
import sys

import psutil
import pythoncom

# Optional UIAutomation
try:
    import uiautomation as auto
except Exception:
    auto = None

# Optional Vision (OCR)
try:
    from PIL import ImageGrab
    import pytesseract
except Exception:
    ImageGrab = None
    pytesseract = None

# Optional Redis
try:
    import redis
except Exception:
    redis = None

# Optional ZeroMQ
try:
    import zmq
except Exception:
    zmq = None


# === AUTO-ELEVATION CHECK (ALL WINDOWS NODES) ===

def ensure_admin():
    try:
        if platform.system().lower().startswith("win"):
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


# === COM INITIALIZATION HELPER ===

def ensure_com_initialized():
    try:
        pythoncom.CoInitialize()
    except pythoncom.com_error:
        # Already initialized or not needed on this thread
        pass


# =========================
# Config
# =========================

SHARED_DIR = Path("./shared_bridge")
SHARED_DIR.mkdir(exist_ok=True)

SHARED_REQ = SHARED_DIR / "request.json"
SHARED_RES = SHARED_DIR / "response.json"

SOCKET_HOST = "127.0.0.1"
SOCKET_PORT = 50555

POLL_INTERVAL = 2.0  # seconds

# Swarm bus config: "file", "redis", "zeromq", "nats", "raw_udp"
SWARM_BUS_MODE = os.environ.get("BORG_SWARM_BUS", "file")

REDIS_URL = os.environ.get("BORG_REDIS_URL", "redis://localhost:6379/0")
ZEROMQ_ENDPOINT_PUB = os.environ.get("BORG_ZMQ_PUB", "tcp://127.0.0.1:5556")
ZEROMQ_ENDPOINT_SUB = os.environ.get("BORG_ZMQ_SUB", "tcp://127.0.0.1:5556")
NATS_ENDPOINT = os.environ.get("BORG_NATS_ENDPOINT", "127.0.0.1:4222")
RAW_UDP_BROADCAST = ("255.255.255.255", 50599)


# =========================
# Utilities
# =========================

def is_windows():
    return platform.system().lower().startswith("win")


def is_linux():
    return platform.system().lower().startswith("linux")


def has_wsl():
    if not is_windows():
        return False
    try:
        result = subprocess.run(
            ["wsl", "uname", "-a"],
            capture_output=True,
            text=True,
            timeout=2
        )
        return result.returncode == 0
    except Exception:
        return False


def run_cmd(cmd, use_shell=True):
    result = subprocess.run(
        cmd,
        shell=use_shell,
        capture_output=True,
        text=True
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


# =========================
# Local backends
# =========================

class LocalBackendBase:
    def get_processes(self):
        return [p.info for p in psutil.process_iter(['pid', 'name'])]

    def run_command(self, cmd):
        out, err, code = run_cmd(cmd)
        return {"stdout": out, "stderr": err, "code": code}


class WindowsLocalBackend(LocalBackendBase):
    def __init__(self):
        self.os = "windows"


class LinuxLocalBackend(LocalBackendBase):
    def __init__(self):
        self.os = "linux"


def create_local_backend():
    if is_windows():
        return WindowsLocalBackend()
    elif is_linux():
        return LinuxLocalBackend()
    else:
        raise RuntimeError("Unsupported OS")


# =========================
# Remote backends (transports)
# =========================

class RemoteBackendBase:
    def get_processes(self):
        raise NotImplementedError

    def run_command(self, cmd):
        raise NotImplementedError


class SharedFileRemoteBackend(RemoteBackendBase):
    def __init__(self):
        self.mode = "shared_files"

    def _send_request(self, payload):
        with SHARED_REQ.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        start = time.time()
        while time.time() - start < 5:
            if SHARED_RES.exists():
                try:
                    with SHARED_RES.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    SHARED_RES.unlink(missing_ok=True)
                    return data
                except Exception:
                    pass
            time.sleep(0.1)
        return {"error": "timeout waiting for shared response"}

    def get_processes(self):
        return self._send_request({"action": "process_list"})

    def run_command(self, cmd):
        return self._send_request({"action": "run", "cmd": cmd})


class SocketRemoteBackend(RemoteBackendBase):
    def __init__(self, host=SOCKET_HOST, port=SOCKET_PORT):
        self.mode = "socket"
        self.host = host
        self.port = port

    def _send(self, payload):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((self.host, self.port))
            data = json.dumps(payload).encode("utf-8")
            s.sendall(data + b"\n")
            chunks = []
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            s.close()
            if not chunks:
                return {"error": "no response"}
            resp = json.loads(b"".join(chunks).decode("utf-8"))
            return resp
        except Exception as e:
            return {"error": f"socket error: {e}"}

    def get_processes(self):
        return self._send({"action": "process_list"})

    def run_command(self, cmd):
        return self._send({"action": "run", "cmd": cmd})


class WSLRemoteBackend(RemoteBackendBase):
    def __init__(self):
        self.mode = "wsl"

    def get_processes(self):
        cmd = ["wsl", "ps", "-eo", "pid,comm"]
        out, err, code = run_cmd(cmd, use_shell=False)
        if code != 0:
            return {"error": err or "failed to query WSL processes"}
        procs = []
        for line in out.splitlines()[1:]:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                pid, name = parts
                procs.append({"pid": int(pid), "name": name})
        return procs

    def run_command(self, cmd):
        full_cmd = ["wsl", "bash", "-lc", cmd]
        out, err, code = run_cmd(full_cmd, use_shell=False)
        return {"stdout": out, "stderr": err, "code": code}


# =========================
# Servers (socket + shared file)
# =========================

class BridgeSocketServer(threading.Thread):
    def __init__(self, backend, host=SOCKET_HOST, port=SOCKET_PORT):
        super().__init__(daemon=True)
        self.backend = backend
        self.host = host
        self.port = port
        self._stop = threading.Event()

    def run(self):
        ensure_com_initialized()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.host, self.port))
        s.listen(5)
        s.settimeout(1)
        while not self._stop.is_set():
            try:
                conn, addr = s.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()
        s.close()

    def handle_client(self, conn):
        try:
            data = conn.recv(65536)
            if not data:
                conn.close()
                return
            req = json.loads(data.decode("utf-8"))
            action = req.get("action")
            if action == "process_list":
                resp = self.backend.get_processes()
            elif action == "run":
                cmd = req.get("cmd", "")
                resp = self.backend.run_command(cmd)
            else:
                resp = {"error": "unknown action"}
            conn.sendall(json.dumps(resp).encode("utf-8"))
        except Exception as e:
            try:
                conn.sendall(json.dumps({"error": str(e)}).encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()

    def stop(self):
        self._stop.set()


class SharedFileServer(threading.Thread):
    def __init__(self, backend):
        super().__init__(daemon=True)
        self.backend = backend
        self._stop = threading.Event()

    def run(self):
        ensure_com_initialized()
        while not self._stop.is_set():
            if SHARED_REQ.exists():
                try:
                    with SHARED_REQ.open("r", encoding="utf-8") as f:
                        req = json.load(f)
                    SHARED_REQ.unlink(missing_ok=True)
                    action = req.get("action")
                    if action == "process_list":
                        resp = self.backend.get_processes()
                    elif action == "run":
                        cmd = req.get("cmd", "")
                        resp = self.backend.run_command(cmd)
                    else:
                        resp = {"error": "unknown action"}
                    with SHARED_RES.open("w", encoding="utf-8") as f:
                        json.dump(resp, f)
                except Exception:
                    pass
            time.sleep(0.1)

    def stop(self):
        self._stop.set()


# =========================
# UIAutomation Organ
# =========================

class UIAutomationOrgan:
    def __init__(self):
        self.enabled = auto is not None and is_windows()
        self.auto = auto if self.enabled else None

    def get_active_window(self):
        if not self.enabled:
            return None
        ensure_com_initialized()
        w = self.auto.GetForegroundControl()
        return {
            "name": w.Name,
            "class": w.ClassName,
            "control_type": w.ControlTypeName,
            "rect": w.BoundingRectangle
        }

    def dump_ui_tree(self, depth=3):
        if not self.enabled:
            return None
        ensure_com_initialized()
        root = self.auto.GetRootControl()
        return self._walk(root, depth)

    def _walk(self, node, depth):
        if depth <= 0:
            return []
        children = []
        for c in node.GetChildren():
            children.append({
                "name": c.Name,
                "class": c.ClassName,
                "control_type": c.ControlTypeName,
                "children": self._walk(c, depth - 1)
            })
        return children

    def find_elements(self, text_keywords=None, control_type=None, max_depth=4):
        if not self.enabled:
            return []
        ensure_com_initialized()
        if text_keywords is None:
            text_keywords = []
        root = self.auto.GetRootControl()
        matches = []

        def walk(node, depth):
            if depth <= 0:
                return
            name = (node.Name or "").lower()
            ctype = (node.ControlTypeName or "").lower()
            if (not control_type or ctype == control_type.lower()) and any(
                kw.lower() in name for kw in text_keywords
            ):
                matches.append(node)
            for child in node.GetChildren():
                walk(child, depth - 1)

        walk(root, max_depth)
        return matches

    def click_elements(self, elements):
        if not self.enabled:
            return 0
        count = 0
        for el in elements:
            try:
                el.Click()
                count += 1
            except Exception:
                pass
        return count

    def suppress_popups(self):
        if not self.enabled:
            return {"clicked": 0}
        candidates = self.find_elements(
            text_keywords=["ok", "close", "yes", "no", "accept"],
            control_type="Button",
            max_depth=5
        )
        clicked = self.click_elements(candidates)
        return {"clicked": clicked}


# =========================
# Vision Organ
# =========================

class VisionOrgan:
    def __init__(self):
        self.enabled = ImageGrab is not None and pytesseract is not None

    def capture_screen_text(self):
        if not self.enabled:
            return None
        img = ImageGrab.grab()
        text = pytesseract.image_to_string(img)
        return text


# =========================
# AIOrgan (role-based, persistent Q-table)
# =========================

class AIOrgan:
    def __init__(self, node_id="default_node", role="scout", base_dir="./borg_ai"):
        self.node_id = node_id
        self.role = role
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.q_table = {}
        self._load_q_table()

    @property
    def q_path(self):
        return self.base_dir / f"qtable_{self.node_id}_{self.role}.json"

    def _load_q_table(self):
        if self.q_path.exists():
            try:
                with self.q_path.open("r", encoding="utf-8") as f:
                    self.q_table = json.load(f)
            except Exception:
                self.q_table = {}
        else:
            self.q_table = {}

    def save_q_table(self):
        try:
            with self.q_path.open("w", encoding="utf-8") as f:
                json.dump(self.q_table, f, indent=2)
        except Exception:
            pass

    def encode_state(self, ui_state, procs_local, procs_remote, vision_text=None):
        ui_tag = ui_state or "none"
        local_count = len(procs_local) if isinstance(procs_local, list) else 0
        remote_count = len(procs_remote) if isinstance(procs_remote, list) else 0

        vision_tag = "none"
        if vision_text:
            t = vision_text.lower()
            if "error" in t or "warning" in t:
                vision_tag = "error"
            elif "login" in t or "password" in t:
                vision_tag = "login"
            else:
                vision_tag = "other"

        return f"ui:{ui_tag}|l:{local_count}|r:{remote_count}|v:{vision_tag}"

    def classify_ui_state(self, ui_tree, vision_text=None):
        if not ui_tree and not vision_text:
            return "no_ui"

        flat = (json.dumps(ui_tree) if ui_tree else "").lower()
        if vision_text:
            flat += " " + vision_text.lower()

        if "password" in flat or "login" in flat:
            return "login_screen"
        if "error" in flat or "warning" in flat or "failed" in flat:
            return "error_popup"
        if "install" in flat or "setup" in flat:
            return "installer"
        return "normal"

    def available_actions(self):
        if self.role == "scout":
            return ["idle", "scan_processes", "focus_observation"]
        if self.role == "analyst":
            return ["idle", "scan_processes", "flag_anomaly"]
        if self.role == "archivist":
            return ["idle", "log_state"]
        if self.role == "commander":
            return ["idle", "suppress_popups", "focus_observation"]
        return ["idle"]

    def _ensure_state(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.available_actions()}

    def choose_action(self, state_key):
        self._ensure_state(state_key)
        return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def update_reward(self, state_key, action, reward):
        self._ensure_state(state_key)
        self.q_table[state_key][action] += reward

    def decide(self, ui_state, procs_local, procs_remote, vision_text=None):
        state_key = self.encode_state(ui_state, procs_local, procs_remote, vision_text)
        action = self.choose_action(state_key)
        return state_key, action


# =========================
# SwarmBus (pluggable, with subscribe)
# =========================

class SwarmBus:
    def __init__(self, mode=SWARM_BUS_MODE):
        self.mode = mode
        self.file_dir = Path("./borg_swarm")
        self.file_dir.mkdir(parents=True, exist_ok=True)

        self.redis_client = None
        self.zmq_ctx = None
        self.zmq_pub = None
        self.zmq_sub = None
        self.raw_udp_sock = None

        self.subscribers = {}  # topic -> list[callback]
        self.sub_thread = None
        self._stop_sub = threading.Event()

        if self.mode == "redis" and redis is not None:
            try:
                self.redis_client = redis.from_url(REDIS_URL)
            except Exception:
                self.mode = "file"

        if self.mode == "zeromq" and zmq is not None:
            try:
                self.zmq_ctx = zmq.Context()
                self.zmq_pub = self.zmq_ctx.socket(zmq.PUB)
                self.zmq_pub.bind(ZEROMQ_ENDPOINT_PUB)
                self.zmq_sub = self.zmq_ctx.socket(zmq.SUB)
                self.zmq_sub.connect(ZEROMQ_ENDPOINT_SUB)
                self.zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            except Exception:
                self.mode = "file"

        if self.mode == "raw_udp":
            try:
                self.raw_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.raw_udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            except Exception:
                self.mode = "file"

        if self.mode == "redis" and self.redis_client is not None:
            self.sub_thread = threading.Thread(target=self._redis_sub_loop, daemon=True)
            self.sub_thread.start()
        elif self.mode == "zeromq" and self.zmq_sub is not None:
            self.sub_thread = threading.Thread(target=self._zmq_sub_loop, daemon=True)
            self.sub_thread.start()

    def publish(self, topic, payload):
        data = json.dumps({"topic": topic, "payload": payload, "ts": time.time()})

        if self.mode == "redis" and self.redis_client is not None:
            try:
                self.redis_client.publish(topic, data)
                return
            except Exception:
                pass

        if self.mode == "zeromq" and self.zmq_pub is not None:
            try:
                self.zmq_pub.send_string(f"{topic} {data}")
                return
            except Exception:
                pass

        if self.mode == "nats":
            path = self.file_dir / f"nats_{topic}_{int(time.time())}.json"
            try:
                with path.open("w", encoding="utf-8") as f:
                    f.write(data)
                return
            except Exception:
                pass

        if self.mode == "raw_udp" and self.raw_udp_sock is not None:
            try:
                self.raw_udp_sock.sendto(data.encode("utf-8"), RAW_UDP_BROADCAST)
                return
            except Exception:
                pass

        path = self.file_dir / f"{topic}_{int(time.time())}.json"
        try:
            with path.open("w", encoding="utf-8") as f:
                f.write(data)
        except Exception:
            pass

    def subscribe(self, topic, callback):
        self.subscribers.setdefault(topic, []).append(callback)

    def _dispatch(self, topic, payload):
        for t, cbs in self.subscribers.items():
            if t == topic or t == "*":
                for cb in cbs:
                    try:
                        cb(topic, payload)
                    except Exception:
                        pass

    def _redis_sub_loop(self):
        ensure_com_initialized()
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.psubscribe("*")
            for msg in pubsub.listen():
                if self._stop_sub.is_set():
                    break
                if msg["type"] not in ("message", "pmessage"):
                    continue
                data = msg["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                try:
                    obj = json.loads(data)
                    topic = obj.get("topic") or msg.get("channel")
                    payload = obj.get("payload")
                    self._dispatch(topic, payload)
                except Exception:
                    pass
        except Exception:
            pass

    def _zmq_sub_loop(self):
        ensure_com_initialized()
        while not self._stop_sub.is_set():
            try:
                msg = self.zmq_sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.1)
                continue
            try:
                topic, data = msg.split(" ", 1)
                obj = json.loads(data)
                payload = obj.get("payload")
                self._dispatch(topic, payload)
            except Exception:
                pass

    def stop(self):
        self._stop_sub.set()


# =========================
# HybridBridge
# =========================

class HybridBridge:
    def __init__(self, remote_modes=("wsl", "socket", "shared_files")):
        self.local = create_local_backend()
        self.remote_modes = remote_modes
        self.remote_backends = self._init_remote_backends()

        self.ui = UIAutomationOrgan()
        self.vision = VisionOrgan()
        self.ai_helper = AIOrgan(node_id="helper", role="scout")

    def _init_remote_backends(self):
        backends = []
        for mode in self.remote_modes:
            if mode == "wsl" and has_wsl():
                backends.append(WSLRemoteBackend())
            elif mode == "socket":
                backends.append(SocketRemoteBackend())
            elif mode == "shared_files":
                backends.append(SharedFileRemoteBackend())
        return backends

    def get_local_processes(self):
        return self.local.get_processes()

    def run_on_local(self, cmd):
        return self.local.run_command(cmd)

    def get_remote_processes(self):
        results = []
        for rb in self.remote_backends:
            try:
                res = rb.get_processes()
                results.append({"mode": rb.mode, "data": res})
            except Exception as e:
                results.append({"mode": rb.mode, "error": str(e)})
        return results

    def run_on_remote(self, cmd):
        results = []
        for rb in self.remote_backends:
            try:
                res = rb.run_command(cmd)
                results.append({"mode": rb.mode, "data": res})
            except Exception as e:
                results.append({"mode": rb.mode, "error": str(e)})
        return results


# =========================
# SwarmCoordinator (heartbeats, leader election, anomaly consensus skeleton)
# =========================

class SwarmCoordinator(threading.Thread):
    def __init__(self, node_id, bus: SwarmBus, heartbeat_interval=5.0):
        super().__init__(daemon=True)
        self.node_id = node_id
        self.bus = bus
        self.heartbeat_interval = heartbeat_interval
        self.running = False

        self.leader_id = None
        self.nodes_seen = set()
        self.anomaly_counts = {}  # signature -> count

        self.bus.subscribe("heartbeat", self._on_heartbeat)
        self.bus.subscribe("leader", self._on_leader)
        self.bus.subscribe("anomaly", self._on_anomaly)

    def _on_heartbeat(self, topic, payload):
        nid = payload.get("node")
        if nid:
            self.nodes_seen.add(nid)

    def _on_leader(self, topic, payload):
        self.leader_id = payload.get("leader")

    def _on_anomaly(self, topic, payload):
        sig = payload.get("signature", "generic")
        self.anomaly_counts[sig] = self.anomaly_counts.get(sig, 0) + 1

    def _elect_leader(self):
        if not self.nodes_seen:
            return
        new_leader = sorted(self.nodes_seen)[0]
        if new_leader != self.leader_id:
            self.leader_id = new_leader
            self.bus.publish("leader", {"leader": self.leader_id})

    def run(self):
        ensure_com_initialized()
        self.running = True
        while self.running:
            self.bus.publish("heartbeat", {"node": self.node_id, "ts": time.time()})
            self._elect_leader()
            time.sleep(self.heartbeat_interval)

    def stop(self):
        self.running = False


# =========================
# AutonomousController
# =========================

class AutonomousController(threading.Thread):
    def __init__(self, bridge, node_id="node01", role="scout", interval=3.0, enable_vision=True, swarm_bus=None):
        super().__init__(daemon=True)
        self.bridge = bridge
        self.interval = interval
        self.running = False

        self.ui = bridge.ui
        self.vision = bridge.vision if enable_vision else None

        self.ai = AIOrgan(node_id=node_id, role=role)
        self.role = role
        self.node_id = node_id

        self.swarm_bus = swarm_bus

    def sense(self):
        local_procs = self.bridge.get_local_processes()
        remote_procs = self.bridge.get_remote_processes()

        ui_tree = None
        if self.ui is not None and getattr(self.ui, "enabled", False):
            try:
                ui_tree = self.ui.dump_ui_tree(depth=3)
            except Exception:
                ui_tree = None

        vision_text = None
        if self.vision is not None and getattr(self.vision, "enabled", False):
            try:
                vision_text = self.vision.capture_screen_text()
            except Exception:
                vision_text = None

        ui_state = self.ai.classify_ui_state(ui_tree, vision_text)

        return {
            "local_procs": local_procs,
            "remote_procs": remote_procs,
            "ui_tree": ui_tree,
            "ui_state": ui_state,
            "vision_text": vision_text,
        }

    def act(self, state_key, action, env):
        reward = 0.0

        if action == "idle":
            reward += 0.05

        elif action == "scan_processes":
            reward += 0.1

        elif action == "focus_observation":
            reward += 0.1

        elif action == "flag_anomaly":
            reward += 0.2
            if self.swarm_bus:
                sig = env["ui_state"]
                self.swarm_bus.publish("anomaly", {
                    "node": self.node_id,
                    "role": self.role,
                    "signature": sig,
                    "env": {"ui_state": env["ui_state"]},
                })

        elif action == "log_state":
            snapshot_dir = Path("./borg_logs")
            snapshot_dir.mkdir(exist_ok=True)
            snap_path = snapshot_dir / f"snapshot_{self.node_id}_{int(time.time())}.json"
            try:
                with snap_path.open("w", encoding="utf-8") as f:
                    json.dump(env, f, indent=2)
                reward += 0.3
            except Exception:
                pass

        elif action == "suppress_popups":
            if self.ui is not None and getattr(self.ui, "enabled", False):
                res = self.ui.suppress_popups()
                if res.get("clicked", 0) > 0:
                    reward += 0.5

        self.ai.update_reward(state_key, action, reward)
        return reward

    def run(self):
        ensure_com_initialized()
        self.running = True
        while self.running:
            try:
                env = self.sense()
                state_key, action = self.ai.decide(
                    env["ui_state"],
                    env["local_procs"],
                    env["remote_procs"],
                    env["vision_text"],
                )
                self.act(state_key, action, env)
                self.ai.save_q_table()
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False


# =========================
# Main entry
# =========================

def main():
    ensure_com_initialized()

    local_backend = create_local_backend()

    socket_server = BridgeSocketServer(local_backend)
    socket_server.start()

    shared_server = SharedFileServer(local_backend)
    shared_server.start()

    bridge = HybridBridge(remote_modes=("wsl", "socket", "shared_files"))

    swarm_bus = SwarmBus()
    node_id = os.environ.get("BORG_NODE_ID", "node01")

    coordinator = SwarmCoordinator(node_id=node_id, bus=swarm_bus)
    coordinator.start()

    scout = AutonomousController(bridge, node_id=node_id, role="scout", interval=3.0, swarm_bus=swarm_bus)
    analyst = AutonomousController(bridge, node_id=node_id, role="analyst", interval=5.0, swarm_bus=swarm_bus)
    archivist = AutonomousController(bridge, node_id=node_id, role="archivist", interval=10.0, swarm_bus=swarm_bus)
    commander = AutonomousController(bridge, node_id=node_id, role="commander", interval=4.0, swarm_bus=swarm_bus)

    scout.start()
    analyst.start()
    archivist.start()
    commander.start()

    print(f"Hybrid Borg swarm running with bus='{SWARM_BUS_MODE}'. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scout.stop()
        analyst.stop()
        archivist.stop()
        commander.stop()
        coordinator.stop()
        swarm_bus.stop()
        socket_server.stop()
        shared_server.stop()
        time.sleep(1)
        print("Stopped.")


if __name__ == "__main__":
    main()

