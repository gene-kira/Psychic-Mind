import importlib
import subprocess
import sys
import time
import json
import os
import random
import threading
import logging
import socket
from enum import Enum
from typing import List, Dict, Tuple, Optional

import tkinter as tk
from tkinter import ttk

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

# =========================
# AUTOLOADER
# =========================

OPTIONAL_LIBS = {
    "uiautomation": "uiautomation",
    "numpy": "numpy",
    "psutil": "psutil",
    "platform": "platform",
}

def safe_import(name, pip_name=None):
    try:
        return importlib.import_module(name)
    except ImportError:
        if not pip_name:
            logging.warning(f"Optional lib {name} not available.")
            return None
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            return importlib.import_module(name)
        except Exception as e:
            logging.error(f"Failed to install {pip_name}: {e}")
            return None

LIBS = {k: safe_import(k, v) for k, v in OPTIONAL_LIBS.items()}
uia = LIBS.get("uiautomation")
np = LIBS.get("numpy")
psutil = LIBS.get("psutil")
platform_mod = LIBS.get("platform")

# =========================
# CONFIG
# =========================

DEFAULT_CONFIG = {
    "memory_path": "swarm_memory.json",
    "genome_path": "swarm_genomes.json",
    "max_events": 2000,
    "target_fps": 30,
    "min_delay": 10,
    "max_delay": 120,
    "ui_automation_enabled": False,
    "initial_drones": 5,
    "max_drones": 12,
    "sandbox_mode": True,
    "snapshot_path": "swarm_snapshot.json",
    "network_enabled": False,
    "network_port": 50555,
    "network_broadcast_interval": 2.0,
    "plugins_dir": "plugins",
}

class Config:
    def __init__(self, path: str = "swarm_config.json"):
        self.path = path
        self.data = DEFAULT_CONFIG.copy()
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                user_cfg = json.load(f)
            self.data.update(user_cfg)
            logging.info("Config loaded.")
        except Exception as e:
            logging.error(f"Failed to load config: {e}")

    def get(self, key, default=None):
        return self.data.get(key, default)

# =========================
# UI DRIVER (SANDBOXED)
# =========================

class UIDriver:
    def __init__(self, enabled=True, sandbox=True):
        self.enabled = enabled and (uia is not None) and not sandbox
        self.sandbox = sandbox
        if self.sandbox:
            logging.info("UIDriver running in sandbox mode (no real UI actions).")

    def click_button_by_name(self, name: str) -> bool:
        if not self.enabled:
            return False
        try:
            btn = uia.Control(searchDepth=5, Name=name)
            if btn:
                btn.Click()
                return True
        except Exception as e:
            logging.error(f"UI click failed: {e}")
        return False

    def type_text(self, name: str, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            edit = uia.Control(searchDepth=5, Name=name)
            if edit:
                edit.SendKeys(text)
                return True
        except Exception as e:
            logging.error(f"UI type failed: {e}")
        return False

# =========================
# SYSTEM TELEMETRY (THREADED)
# =========================

class SystemTelemetry:
    def __init__(self):
        self.enabled = psutil is not None
        self.last_net = None
        self.lock = threading.Lock()
        self.current_snapshot = {
            "cpu": 0.0,
            "ram": 0.0,
            "disk": 0.0,
            "net": 0.0,
            "procs": 0,
        }
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            snap = self._snapshot_once()
            with self.lock:
                self.current_snapshot = snap
            time.sleep(0.5)

    def _snapshot_once(self) -> Dict:
        if not self.enabled:
            return {
                "cpu": random.uniform(5, 50),
                "ram": random.uniform(10, 70),
                "disk": random.uniform(5, 60),
                "net": random.uniform(0, 10),
                "procs": 50,
            }
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            net_io = psutil.net_io_counters()
            total_bytes = net_io.bytes_sent + net_io.bytes_recv
            if self.last_net is None:
                net = 0.0
            else:
                net = (total_bytes - self.last_net) / (1024 * 1024)
            self.last_net = total_bytes
            procs = len(psutil.pids())
            return {
                "cpu": cpu,
                "ram": ram,
                "disk": disk,
                "net": net,
                "procs": procs,
            }
        except Exception as e:
            logging.error(f"Telemetry error: {e}")
            return self.current_snapshot

    def snapshot(self) -> Dict:
        with self.lock:
            return dict(self.current_snapshot)

    def to_swarm_state(self, tele: Dict) -> Dict:
        priority = min(10.0, tele["cpu"] / 10.0 + tele["ram"] / 20.0)
        congestion = min(10.0, tele["ram"] / 15.0 + tele["disk"] / 20.0)
        risk = min(10.0, tele["disk"] / 15.0 + tele["net"] / 10.0)
        return {
            "priority": priority,
            "congestion": congestion,
            "risk": risk,
        }

    def stop(self):
        self.running = False

# =========================
# HARDWARE SCANNER
# =========================

class HardwareScanner:
    def __init__(self):
        self.info = self.scan()

    def scan(self) -> Dict:
        info = {}
        try:
            if platform_mod:
                info["system"] = platform_mod.system()
                info["release"] = platform_mod.release()
                info["version"] = platform_mod.version()
                info["machine"] = platform_mod.machine()
                info["processor"] = platform_mod.processor()
            if psutil:
                info["cpu_count"] = psutil.cpu_count(logical=True)
                info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
                info["disks"] = [
                    {"device": d.device, "mountpoint": d.mountpoint}
                    for d in psutil.disk_partitions()
                ]
                info["net_ifaces"] = list(psutil.net_if_addrs().keys())
        except Exception as e:
            logging.error(f"Hardware scan failed: {e}")
        return info

# =========================
# WATER PHYSICS ENGINE
# =========================

class WaterPhysicsEngine:
    def __init__(self):
        self.viscosity = 0.7
        self.max_pressure = 10.0

    def compute_flow(self, state: Dict) -> Dict:
        priority = state.get("priority", 5.0)
        congestion = state.get("congestion", 2.0)
        risk = state.get("risk", 1.0)

        pressure = priority - congestion
        pressure = max(0.0, min(self.max_pressure, pressure))
        resistance = risk * self.viscosity
        flow = pressure - resistance
        turbulence = abs(pressure - resistance)

        return {
            "flow": flow,
            "should_reroute": flow < 1.0,
            "turbulence": turbulence,
        }

# =========================
# WIRE MODEL (SPLINE)
# =========================

class WireModel:
    def __init__(self, origin: Tuple[float, float], target: Tuple[float, float]):
        self.origin = origin
        self.target = target
        self.control_points: List[Tuple[float, float]] = []
        self.color = "#00FFFF"
        self.thickness = 1.0

    def update_from_tension(self, tension: float, signal: float = 1.0):
        curvature = min(1.0, tension / 10.0)
        mid_x = (self.origin[0] + self.target[0]) / 2
        mid_y = (self.origin[1] + self.target[1]) / 2 - 100 * curvature

        self.control_points = [
            self.origin,
            (mid_x, mid_y),
            self.target,
        ]
        self.thickness = 1.0 + curvature * 3.0
        self.color = "#00FFFF" if signal > 0 else "#FF0000"

# =========================
# ROLES + VISUALS
# =========================

class DroneRole(Enum):
    SCOUT = "scout"
    OPERATOR = "operator"
    EXECUTOR = "executor"

ROLE_VISUAL = {
    DroneRole.SCOUT: {
        "glyph": "△",
        "color": "#00FFFF",
        "wire_factor": 0.5,
    },
    DroneRole.OPERATOR: {
        "glyph": "◉",
        "color": "#0088FF",
        "wire_factor": 1.0,
    },
    DroneRole.EXECUTOR: {
        "glyph": "⬢",
        "color": "#FF00AA",
        "wire_factor": 2.0,
    },
}

ROLE_THRESHOLDS = {
    "SCOUT→OPERATOR":  15,
    "OPERATOR→EXECUTOR": 30,
    "EXECUTOR→OPERATOR": -10,
    "OPERATOR→SCOUT": -20,
}

# =========================
# PERSONALITIES
# =========================

class DronePersonality(Enum):
    AGGRESSIVE = "aggressive"
    CAUTIOUS = "cautious"
    BALANCED = "balanced"

PERSONALITY_TRAITS = {
    DronePersonality.AGGRESSIVE: {"risk_bias": -1.0, "priority_bias": 2.0},
    DronePersonality.CAUTIOUS: {"risk_bias": 2.0, "priority_bias": -1.0},
    DronePersonality.BALANCED: {"risk_bias": 0.0, "priority_bias": 0.0},
}

# =========================
# WATER FIELD + HISTORY
# =========================

class WaterField:
    def __init__(self):
        self.global_congestion = 0.0
        self.global_risk = 0.0
        self.remote_pressure = 0.0  # from other nodes

    def update_from_swarm(self, swarm_state: List[Dict]):
        if not swarm_state:
            self.global_congestion = 0.0
            self.global_risk = 0.0
            return
        turbulences = [d["decision"]["turbulence"] for d in swarm_state]
        risks = [d["world"]["risk"] for d in swarm_state if "world" in d]
        self.global_congestion = sum(turbulences) / max(1, len(turbulences))
        self.global_risk = max(risks) if risks else 0.0

    def merge_remote_pressure(self, remote_pressure: float):
        self.remote_pressure = remote_pressure

    def get_local_state(self, base_state: Dict) -> Dict:
        return {
            "priority": base_state.get("priority", 5),
            "congestion": base_state.get("congestion", 2) + self.global_congestion + self.remote_pressure * 0.3,
            "risk": max(base_state.get("risk", 1), self.global_risk),
        }

class WaterfallHistory:
    def __init__(self, max_frames: int = 200):
        self.frames: List[Dict] = []
        self.max_frames = max_frames

    def push(self, field: WaterField):
        self.frames.append({
            "congestion": field.global_congestion + field.remote_pressure,
            "risk": field.global_risk,
            "timestamp": time.time(),
        })
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

# =========================
# DRONE BUS (COMMS)
# =========================

class DroneBus:
    def __init__(self):
        self.messages: List[Tuple[str, str, Dict]] = []

    def broadcast(self, sender_id: str, msg_type: str, payload: Dict):
        self.messages.append((sender_id, msg_type, payload))

    def collect(self, receiver_id: str) -> List[Tuple[str, str, Dict]]:
        msgs = self.messages[:]
        self.messages.clear()
        return msgs

# =========================
# EMERGENT SWARM BRAIN + GOALS
# =========================

class EmergentSwarmBrain:
    def __init__(self):
        self.collective_vector: Tuple[float, float] = (0.0, 0.0)
        self.confidence: float = 0.0
        self.predicted_vector: Tuple[float, float] = (0.0, 0.0)
        self.goal_vector: Tuple[float, float] = (0.0, 0.0)

    def set_goal(self, vec: Tuple[float, float]):
        self.goal_vector = vec

    def update(self, drone_reports: List[Dict]):
        vectors = [r["intent"] for r in drone_reports if "intent" in r]
        risks = [r["decision"]["turbulence"] for r in drone_reports]
        confs = [r.get("confidence", 0.5) for r in drone_reports]

        if vectors:
            x = sum(v[0] * c for v, c in zip(vectors, confs)) / len(vectors)
            y = sum(v[1] * c for v, c in zip(vectors, confs)) / len(vectors)
            self.collective_vector = (x, y)

        if risks:
            avg_turb = sum(risks) / len(risks)
            self.confidence = 1.0 / (1.0 + avg_turb)
        else:
            self.confidence = 1.0

        gx, gy = self.goal_vector
        cx, cy = self.collective_vector
        self.predicted_vector = (cx * 0.7 + gx * 0.3, cy * 0.7 + gy * 0.3)

# =========================
# HIVE-MIND CONSENSUS
# =========================

class HiveMindConsensus:
    def __init__(self):
        self.decision: Optional[str] = None

    def compute(self, drone_reports: List[Dict]) -> Optional[str]:
        votes: Dict[str, int] = {}
        for r in drone_reports:
            label = r.get("intent_label", "none")
            role = r.get("role", "scout")
            weight = 3 if role == "executor" else 2 if role == "operator" else 1
            votes[label] = votes.get(label, 0) + weight
        if votes:
            self.decision = max(votes, key=votes.get)
        return self.decision

# =========================
# OPERATOR OVERRIDE
# =========================

class OperatorOverride:
    def __init__(self):
        self.active = False
        self.target_drone_id: Optional[str] = None

    def engage(self, drone: "DataDrone"):
        drone.tether["signal"] = 0
        drone.state = "override"
        self.active = True
        self.target_drone_id = drone.id

    def release(self, drone: "DataDrone"):
        drone.tether["signal"] = 1
        if drone.state == "override":
            drone.state = "idle"
        self.active = False
        self.target_drone_id = None

# =========================
# SWARM MEMORY + PERSISTENCE + DREAMS
# =========================

class SwarmMemory:
    def __init__(self, max_events: int = 1000, path: str = "swarm_memory.json"):
        self.events: List[Dict] = []
        self.max_events = max_events
        self.path = path
        self.load()

    def record(self, event: Dict):
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def summarize(self) -> Dict:
        if not self.events:
            return {"avg_flow": 0.0, "avg_turbulence": 0.0}
        flows = [e.get("flow", 0.0) for e in self.events if "flow" in e]
        turbs = [e.get("turbulence", 0.0) for e in self.events if "turbulence" in e]
        return {
            "avg_flow": sum(flows) / max(1, len(flows)),
            "avg_turbulence": sum(turbs) / max(1, len(turbs)),
        }

    def dream_optimize(self):
        return self.summarize()

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.events, f)
        except Exception as e:
            logging.error(f"Failed to save memory: {e}")

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                self.events = json.load(f)
            logging.info("Swarm memory loaded.")
        except Exception as e:
            logging.error(f"Failed to load memory: {e}")
            self.events = []

# =========================
# EMOTIONAL STATE MODEL + ANOMALY STATE MACHINE
# =========================

class SwarmMood(Enum):
    CALM = "calm"
    FOCUSED = "focused"
    STRESSED = "stressed"
    PANICKED = "panicked"

class AnomalyLevel(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"

class SwarmEmotionEngine:
    def __init__(self):
        self.mood = SwarmMood.CALM
        self.anomaly_state = False
        self.anomaly_level = AnomalyLevel.NORMAL
        self.last_anomaly_reason = "none"

    def update(self, field: WaterField, consensus_confidence: float):
        pressure = field.global_congestion + field.global_risk + field.remote_pressure
        if pressure < 2 and consensus_confidence > 0.7:
            self.mood = SwarmMood.CALM
        elif pressure < 5:
            self.mood = SwarmMood.FOCUSED
        elif pressure < 8:
            self.mood = SwarmMood.STRESSED
        else:
            self.mood = SwarmMood.PANICKED

        if pressure < 5:
            self.anomaly_level = AnomalyLevel.NORMAL
            self.anomaly_state = False
            self.last_anomaly_reason = "none"
        elif pressure < 8:
            self.anomaly_level = AnomalyLevel.ELEVATED
            self.anomaly_state = True
            self.last_anomaly_reason = "elevated pressure"
        else:
            self.anomaly_level = AnomalyLevel.CRITICAL
            self.anomaly_state = True
            self.last_anomaly_reason = "critical pressure"

    def background_color(self) -> str:
        if self.mood == SwarmMood.CALM:
            return "#001533"
        if self.mood == SwarmMood.FOCUSED:
            return "#003344"
        if self.mood == SwarmMood.STRESSED:
            return "#665500"
        return "#660000"

# =========================
# GENOME PERSISTENCE
# =========================

class GenomeStore:
    def __init__(self, path: str = "swarm_genomes.json"):
        self.path = path
        self.genomes: Dict[str, Dict] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                self.genomes = json.load(f)
            logging.info("Genomes loaded.")
        except Exception as e:
            logging.error(f"Failed to load genomes: {e}")
            self.genomes = {}

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.genomes, f)
        except Exception as e:
            logging.error(f"Failed to save genomes: {e}")

    def get(self, drone_id: str) -> Optional[Dict]:
        return self.genomes.get(drone_id)

    def set(self, drone_id: str, genome: Dict):
        self.genomes[drone_id] = genome

# =========================
# RL BRAIN (PER-DRONE Q-LEARNING)
# =========================

class RLBrain:
    def __init__(self):
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.actions = ["scan", "hold", "execute"]
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.last_state: Optional[str] = None
        self.last_action: Optional[str] = None

    def _discretize_state(self, local_state: Dict) -> str:
        p = int(local_state.get("priority", 5))
        c = int(local_state.get("congestion", 2))
        r = int(local_state.get("risk", 1))
        return f"{p}:{c}:{r}"

    def select_action(self, local_state: Dict) -> str:
        s = self._discretize_state(local_state)
        self.q_table.setdefault(s, {a: 0.0 for a in self.actions})
        if random.random() < self.epsilon:
            a = random.choice(self.actions)
        else:
            q = self.q_table[s]
            a = max(q, key=q.get)
        self.last_state = s
        self.last_action = a
        return a

    def update(self, reward: float, new_state: Dict):
        if self.last_state is None or self.last_action is None:
            return
        s = self.last_state
        a = self.last_action
        ns = self._discretize_state(new_state)
        self.q_table.setdefault(ns, {ac: 0.0 for ac in self.actions})
        old_q = self.q_table[s][a]
        max_next = max(self.q_table[ns].values())
        new_q = old_q + self.alpha * (reward + self.gamma * max_next - old_q)
        self.q_table[s][a] = new_q

# =========================
# DATA DRONE + GENETIC EVOLUTION + RL
# =========================

class DataDrone:
    def __init__(
        self,
        id: str,
        physics_engine: WaterPhysicsEngine,
        ui_driver: UIDriver,
        origin_pos: Tuple[float, float],
        role: DroneRole = DroneRole.SCOUT,
        personality: DronePersonality = DronePersonality.BALANCED,
        genome: Optional[Dict] = None,
    ):
        self.id = id
        self.physics = physics_engine
        self.ui = ui_driver
        self.role = role
        self.personality = personality

        self.state = "idle"
        self.tether = {"tension": 0.0, "signal": 1.0}
        self.screen_pos = origin_pos
        self.wire_model = WireModel(origin_pos, origin_pos)

        self.performance = 0.0
        self.anomalies = 0.0
        self.stability = 0.0

        self.intent: Tuple[float, float] = (0.0, 0.0)
        self.intent_label: str = "none"
        self.confidence: float = 0.5

        self.genome = genome or {
            "risk_sensitivity": 1.0,
            "priority_weight": 1.0,
        }

        self.rl = RLBrain()

    def _clamp_genome(self):
        self.genome["risk_sensitivity"] = max(0.1, min(5.0, self.genome["risk_sensitivity"]))
        self.genome["priority_weight"] = max(0.1, min(5.0, self.genome["priority_weight"]))

    def sense(self, world_state: Dict):
        self.intent = (
            world_state.get("target_dx", 0.0),
            world_state.get("target_dy", 0.0),
        )
        self.intent_label = world_state.get("intent_label", "none")

    def decide(self, local_state: Dict) -> Dict:
        traits = PERSONALITY_TRAITS[self.personality]
        local_state = dict(local_state)
        local_state["priority"] += traits["priority_bias"] * self.genome["priority_weight"]
        local_state["risk"] += traits["risk_bias"] * self.genome["risk_sensitivity"]

        rl_action = self.rl.select_action(local_state)
        if rl_action == "scan":
            local_state["priority"] *= 0.9
        elif rl_action == "execute":
            local_state["priority"] *= 1.1

        flow = self.physics.compute_flow(local_state)
        self.tether["tension"] = flow["turbulence"]
        self.confidence = max(0.1, min(1.0, flow["flow"] / 10.0))
        return flow

    def act(self, decision: Dict):
        if self.tether["signal"] <= 0:
            self.state = "override"
            return

        if decision["flow"] <= 0:
            self.state = "blocked"
            self.anomalies += 0.1
            return

        self.state = "acting"

        if self.role == DroneRole.SCOUT:
            self.performance += 0.1
            return

        if self.role == DroneRole.OPERATOR:
            if decision["flow"] > 3 and self.ui.enabled:
                self.ui.click_button_by_name("Preview")
                self.performance += 0.5
            return

        if self.role == DroneRole.EXECUTOR:
            if decision["flow"] > 3 and self.ui.enabled:
                self.ui.click_button_by_name("Submit")
                self.performance += 1.0

    def update_wire_visuals(self):
        role_viz = ROLE_VISUAL[self.role]
        self.wire_model.update_from_tension(
            self.tether["tension"],
            self.tether["signal"]
        )
        self.wire_model.thickness *= role_viz["wire_factor"]
        self.wire_model.color = role_viz["color"]

    def evaluate_role(self):
        rfi = self.performance - self.anomalies + self.stability

        if self.role == DroneRole.SCOUT and rfi > ROLE_THRESHOLDS["SCOUT→OPERATOR"]:
            self.role = DroneRole.OPERATOR

        elif self.role == DroneRole.OPERATOR:
            if rfi > ROLE_THRESHOLDS["OPERATOR→EXECUTOR"]:
                self.role = DroneRole.EXECUTOR
            elif rfi < ROLE_THRESHOLDS["OPERATOR→SCOUT"]:
                self.role = DroneRole.SCOUT

        elif self.role == DroneRole.EXECUTOR and rfi < ROLE_THRESHOLDS["EXECUTOR→OPERATOR"]:
            self.role = DroneRole.OPERATOR

    def communicate(self, bus: DroneBus):
        bus.broadcast(self.id, "FLOW", {
            "turbulence": self.tether["tension"],
            "role": self.role.value,
        })

    def evolve_genome(self):
        if self.performance > self.anomalies + 5:
            self.genome["risk_sensitivity"] *= 0.98
            self.genome["priority_weight"] *= 1.02
        elif self.anomalies > self.performance + 2:
            self.genome["risk_sensitivity"] *= 1.02
            self.genome["priority_weight"] *= 0.98

        for k in self.genome:
            if random.random() < 0.01:
                self.genome[k] *= random.uniform(0.9, 1.1)

        self._clamp_genome()

    def loop(self, world_state: Dict, local_state: Dict, bus: DroneBus) -> Dict:
        self.sense(world_state)
        decision = self.decide(local_state)
        self.act(decision)
        self.update_wire_visuals()
        self.evaluate_role()
        self.communicate(bus)
        self.evolve_genome()

        reward = self.performance - self.anomalies
        self.rl.update(reward, local_state)

        report = {
            "id": self.id,
            "state": self.state,
            "tether": self.tether.copy(),
            "decision": decision,
            "intent": self.intent,
            "intent_label": self.intent_label,
            "confidence": self.confidence,
            "role": self.role.value,
            "world": local_state,
            "performance": self.performance,
            "anomalies": self.anomalies,
            "genome": dict(self.genome),
            "personality": self.personality.value,
        }
        return report

# =========================
# SWARM NETWORK NODE (MULTI-NODE GOSSIP)
# =========================

class SwarmNetworkNode:
    def __init__(self, config: Config):
        self.enabled = config.get("network_enabled", False)
        self.port = config.get("network_port", 50555)
        self.broadcast_interval = config.get("network_broadcast_interval", 2.0)
        self.last_broadcast = 0.0
        self.remote_pressure = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.sock = None
        if self.enabled:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind(("", self.port))
                self.sock.setblocking(False)
                self.running = True
                threading.Thread(target=self._recv_loop, daemon=True).start()
                logging.info(f"Swarm network node listening on UDP {self.port}")
            except Exception as e:
                logging.error(f"Network init failed: {e}")
                self.enabled = False

    def _recv_loop(self):
        while self.running and self.sock:
            try:
                data, _ = self.sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                if "pressure" in msg:
                    with self.lock:
                        self.remote_pressure = float(msg["pressure"])
            except BlockingIOError:
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Network recv error: {e}")
                time.sleep(0.5)

    def maybe_broadcast(self, field: WaterField):
        if not self.enabled or not self.sock:
            return
        now = time.time()
        if now - self.last_broadcast < self.broadcast_interval:
            return
        self.last_broadcast = now
        pressure = field.global_congestion + field.global_risk
        msg = json.dumps({"pressure": pressure})
        try:
            self.sock.sendto(msg.encode("utf-8"), ("127.0.0.1", self.port))
        except Exception as e:
            logging.error(f"Network send error: {e}")

    def get_remote_pressure(self) -> float:
        with self.lock:
            return self.remote_pressure

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

# =========================
# OS-LEVEL EVENT HOOKS (POLLING-BASED)
# =========================

class OSHooks:
    def __init__(self):
        self.last_procs = None
        self.last_cpu = None

    def check_events(self, tele: Dict) -> List[Dict]:
        events = []
        cpu = tele["cpu"]
        procs = tele["procs"]

        if self.last_cpu is not None and cpu - self.last_cpu > 40:
            events.append({"type": "cpu_spike", "delta": cpu - self.last_cpu})
        if self.last_procs is not None and procs - self.last_procs > 50:
            events.append({"type": "proc_surge", "delta": procs - self.last_procs})

        self.last_cpu = cpu
        self.last_procs = procs
        return events

# =========================
# PLUGIN MANAGER (ORGANS)
# =========================

class PluginManager:
    def __init__(self, plugins_dir: str):
        self.plugins_dir = plugins_dir
        self.plugins = []
        self.load_plugins()

    def load_plugins(self):
        if not os.path.isdir(self.plugins_dir):
            return
        for fname in os.listdir(self.plugins_dir):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(self.plugins_dir, fname)
            mod_name = f"plugin_{fname[:-3]}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if not spec or not spec.loader:
                continue
            try:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "register_plugin"):
                    self.plugins.append(mod)
                    logging.info(f"Loaded plugin organ: {fname}")
            except Exception as e:
                logging.error(f"Failed to load plugin {fname}: {e}")

    def tick(self, swarm: "Swarm", tele: Dict):
        for p in self.plugins:
            try:
                if hasattr(p, "tick_plugin"):
                    p.tick_plugin(swarm, tele)
            except Exception as e:
                logging.error(f"Plugin tick error: {e}")

# =========================
# SWARM
# =========================

class Swarm:
    def __init__(
        self,
        drones: List[DataDrone],
        water_field: WaterField,
        waterfall_history: WaterfallHistory,
        swarm_brain: EmergentSwarmBrain,
        consensus: HiveMindConsensus,
        memory: SwarmMemory,
        emotion_engine: SwarmEmotionEngine,
        bus: DroneBus,
        override: OperatorOverride,
        genome_store: GenomeStore,
        network_node: SwarmNetworkNode,
        os_hooks: OSHooks,
        hw_scanner: HardwareScanner,
        plugin_manager: PluginManager,
    ):
        self.drones = drones
        self.field = water_field
        self.history = waterfall_history
        self.brain = swarm_brain
        self.consensus = consensus
        self.memory = memory
        self.emotion = emotion_engine
        self.bus = bus
        self.override = override
        self.genome_store = genome_store
        self.network = network_node
        self.os_hooks = os_hooks
        self.hw_scanner = hw_scanner
        self.plugin_manager = plugin_manager

        self.telepathy_vector: Tuple[float, float] = (0.0, 0.0)
        self.telepathy_intent_label: Optional[str] = None
        self.mission_profile = "BALANCE"

    def inject_telepathy(self, vector: Tuple[float, float], label: str):
        self.telepathy_vector = vector
        self.telepathy_intent_label = label

    def clear_telepathy(self):
        self.telepathy_intent_label = None
        self.telepathy_vector = (0.0, 0.0)

    def set_mission_profile(self, profile: str):
        if profile in ("SCAN", "BALANCE", "EXECUTE"):
            self.mission_profile = profile

    def _apply_mission_profile(self, base_state: Dict, drone_id: str, role: DroneRole) -> Dict:
        state = dict(base_state)
        if self.mission_profile == "SCAN":
            state["priority"] *= 0.8
            state["risk"] *= 0.8
        elif self.mission_profile == "EXECUTE":
            state["priority"] *= 1.2
        return state

    def tick(self, world_states: Dict[str, Dict], tele: Dict) -> List[Dict]:
        events = self.os_hooks.check_events(tele)
        for e in events:
            self.memory.record({"event": e, "timestamp": time.time()})

        self.plugin_manager.tick(self, tele)

        reports: List[Dict] = []

        for d in self.drones:
            base_world = world_states.get(d.id, {})

            if self.telepathy_intent_label is not None:
                base_world["target_dx"] = self.telepathy_vector[0]
                base_world["target_dy"] = self.telepathy_vector[1]
                base_world["intent_label"] = self.telepathy_intent_label

            base_world["hw_profile"] = self.hw_scanner.info.get("cpu_count", 4)

            base_world = self._apply_mission_profile(base_world, d.id, d.role)
            local_state = self.field.get_local_state(base_world)
            report = d.loop(base_world, local_state, self.bus)
            reports.append(report)

            self.memory.record({
                "drone_id": d.id,
                "flow": report["decision"]["flow"],
                "turbulence": report["decision"]["turbulence"],
            })

            self.genome_store.set(d.id, d.genome)

        self.field.update_from_swarm(reports)
        if self.network.enabled:
            self.network.maybe_broadcast(self.field)
            self.field.merge_remote_pressure(self.network.get_remote_pressure())

        self.history.push(self.field)
        self.brain.update(reports)
        self.consensus.compute(reports)
        self.emotion.update(self.field, self.brain.confidence)

        self._anomaly_protocol()

        return reports

    def _anomaly_protocol(self):
        if self.emotion.anomaly_level == AnomalyLevel.NORMAL:
            return
        if self.emotion.anomaly_level == AnomalyLevel.ELEVATED:
            gx, gy = self.brain.goal_vector
            self.brain.set_goal((-gx * 0.5, -gy * 0.5))
        elif self.emotion.anomaly_level == AnomalyLevel.CRITICAL:
            self.brain.set_goal((0.0, 0.0))

    def dream_cycle(self):
        summary = self.memory.dream_optimize()
        self.memory.save()
        self.genome_store.save()
        return summary

    def snapshot_state(self) -> Dict:
        return {
            "drones": [
                {
                    "id": d.id,
                    "role": d.role.value,
                    "personality": d.personality.value,
                    "genome": d.genome,
                    "performance": d.performance,
                    "anomalies": d.anomalies,
                }
                for d in self.drones
            ],
            "mood": self.emotion.mood.value,
            "anomaly_level": self.emotion.anomaly_level.value,
            "consensus": self.consensus.decision,
            "goal_vector": self.brain.goal_vector,
            "mission_profile": self.mission_profile,
            "hw_info": self.hw_scanner.info,
        }

# =========================
# TKINTER COCKPIT
# =========================

class SwarmCockpit:
    def __init__(self, swarm: Swarm, telemetry: SystemTelemetry, config: Config):
        self.swarm = swarm
        self.telemetry = telemetry
        self.config = config

        self.root = tk.Tk()
        self.root.title("Swarm Tactical Map Cockpit")

        self.map_width = 800
        self.map_height = 400
        self.waterfall_width = 200
        self.waterfall_height = 200

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.map_canvas = tk.Canvas(
            self.main_frame,
            width=self.map_width,
            height=self.map_height,
            bg="#000000",
            highlightthickness=0,
        )
        self.map_canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")

        self.waterfall_canvas = tk.Canvas(
            self.main_frame,
            width=self.waterfall_width,
            height=self.waterfall_height,
            bg="#000000",
            highlightthickness=0,
        )
        self.waterfall_canvas.grid(row=1, column=0, sticky="nsew")

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=1, column=1, sticky="nsew")

        self.debug_frame = ttk.Frame(self.main_frame)
        self.debug_frame.grid(row=1, column=2, sticky="nsew")

        self.main_frame.rowconfigure(0, weight=3)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=1)

        self._build_controls()
        self._build_debug_panel()

        self.drone_draw_map: Dict[str, Dict[str, int]] = {}
        self.last_frame_time = time.time()
        self.target_fps = self.config.get("target_fps", 30)
        self.min_delay = self.config.get("min_delay", 10)
        self.max_delay = self.config.get("max_delay", 120)

        self.selected_drone_id: Optional[str] = None

        self._init_drone_graphics()

        self.window_3d = None
        try:
            self.window_3d = tk.Toplevel(self.root)
            self.window_3d.title("Swarm 3D View")
            self.canvas_3d = tk.Canvas(self.window_3d, width=400, height=400, bg="#000000")
            self.canvas_3d.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            logging.error(f"Failed to create 3D window: {e}")
            self.window_3d = None
            self.canvas_3d = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._loop()

    def _build_controls(self):
        ttk.Label(self.control_frame, text="Telepathy Vector X:").grid(row=0, column=0, sticky="w")
        self.tele_x = tk.DoubleVar(value=1.0)
        ttk.Entry(self.control_frame, textvariable=self.tele_x, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(self.control_frame, text="Telepathy Vector Y:").grid(row=1, column=0, sticky="w")
        self.tele_y = tk.DoubleVar(value=0.0)
        ttk.Entry(self.control_frame, textvariable=self.tele_y, width=8).grid(row=1, column=1, sticky="w")

        ttk.Label(self.control_frame, text="Intent Label:").grid(row=2, column=0, sticky="w")
        self.tele_label = tk.StringVar(value="execute")
        ttk.Entry(self.control_frame, textvariable=self.tele_label, width=12).grid(row=2, column=1, sticky="w")

        ttk.Button(self.control_frame, text="Inject Telepathy", command=self._on_inject_telepathy)\
            .grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Button(self.control_frame, text="Clear Telepathy", command=self._on_clear_telepathy)\
            .grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Label(self.control_frame, text="Override Drone:").grid(row=5, column=0, sticky="w")
        self.override_target = tk.StringVar()
        drone_ids = [d.id for d in self.swarm.drones]
        if drone_ids:
            self.override_target.set(drone_ids[0])
        self.override_menu = ttk.Combobox(self.control_frame, textvariable=self.override_target, values=drone_ids, state="readonly")
        self.override_menu.grid(row=5, column=1, sticky="ew")

        ttk.Button(self.control_frame, text="Engage Override", command=self._on_engage_override)\
            .grid(row=6, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Button(self.control_frame, text="Release Override", command=self._on_release_override)\
            .grid(row=7, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Button(self.control_frame, text="Run Dream Cycle", command=self._on_dream_cycle)\
            .grid(row=8, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Button(self.control_frame, text="Reload Config", command=self._on_reload_config)\
            .grid(row=9, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Label(self.control_frame, text="Mission Profile:").grid(row=10, column=0, sticky="w")
        self.mission_profile_var = tk.StringVar(value=self.swarm.mission_profile)
        self.mission_menu = ttk.Combobox(
            self.control_frame,
            textvariable=self.mission_profile_var,
            values=["SCAN", "BALANCE", "EXECUTE"],
            state="readonly"
        )
        self.mission_menu.grid(row=10, column=1, sticky="ew")
        self.mission_menu.bind("<<ComboboxSelected>>", self._on_mission_change)

        ttk.Button(self.control_frame, text="Snapshot State", command=self._on_snapshot)\
            .grid(row=11, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Button(self.control_frame, text="Add Drone", command=self._on_add_drone)\
            .grid(row=12, column=0, sticky="ew", pady=2)
        ttk.Button(self.control_frame, text="Remove Drone", command=self._on_remove_drone)\
            .grid(row=12, column=1, sticky="ew", pady=2)

        self.status_label = ttk.Label(self.control_frame, text="Status: Ready", anchor="w")
        self.status_label.grid(row=13, column=0, columnspan=2, sticky="ew")

        for i in range(14):
            self.control_frame.rowconfigure(i, weight=0)
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(1, weight=1)

    def _build_debug_panel(self):
        ttk.Label(self.debug_frame, text="Debug / Inspector").grid(row=0, column=0, sticky="w")

        ttk.Label(self.debug_frame, text="Select Drone:").grid(row=1, column=0, sticky="w")
        self.inspect_drone_var = tk.StringVar()
        drone_ids = [d.id for d in self.swarm.drones]
        if drone_ids:
            self.inspect_drone_var.set(drone_ids[0])
        self.inspect_menu = ttk.Combobox(
            self.debug_frame,
            textvariable=self.inspect_drone_var,
            values=drone_ids,
            state="readonly"
        )
        self.inspect_menu.grid(row=2, column=0, sticky="ew")
        self.inspect_menu.bind("<<ComboboxSelected>>", lambda e: self._update_inspector())

        self.inspect_text = tk.Text(self.debug_frame, height=10, width=40)
        self.inspect_text.grid(row=3, column=0, sticky="nsew")

        self.debug_frame.rowconfigure(3, weight=1)
        self.debug_frame.columnconfigure(0, weight=1)

    def _init_drone_graphics(self):
        for d in self.swarm.drones:
            x, y = d.screen_pos
            role_viz = ROLE_VISUAL[d.role]
            glyph = role_viz["glyph"]
            color = role_viz["color"]

            text_id = self.map_canvas.create_text(x, y, text=glyph, fill=color, font=("Consolas", 18))
            wire_id = self.map_canvas.create_line(x, y, x, y, fill=color, width=1, smooth=True)

            self.drone_draw_map[d.id] = {
                "glyph": text_id,
                "wire": wire_id,
            }

    def _on_inject_telepathy(self):
        vec = (self.tele_x.get(), self.tele_y.get())
        label = self.tele_label.get()
        self.swarm.inject_telepathy(vec, label)
        self.swarm.brain.set_goal(vec)
        self.status_label.config(text=f"Status: Telepathy injected ({vec}, '{label}')")

    def _on_clear_telepathy(self):
        self.swarm.clear_telepathy()
        self.swarm.brain.set_goal((0.0, 0.0))
        self.status_label.config(text="Status: Telepathy cleared")

    def _on_engage_override(self):
        target_id = self.override_target.get()
        for d in self.swarm.drones:
            if d.id == target_id:
                self.swarm.override.engage(d)
                self.status_label.config(text=f"Status: Override engaged on {target_id}")
                break

    def _on_release_override(self):
        target_id = self.override_target.get()
        for d in self.swarm.drones:
            if d.id == target_id:
                self.swarm.override.release(d)
                self.status_label.config(text=f"Status: Override released on {target_id}")
                break

    def _on_dream_cycle(self):
        summary = self.swarm.dream_cycle()
        self.status_label.config(
            text=f"Dream: avg_flow={summary['avg_flow']:.2f}, avg_turb={summary['avg_turbulence']:.2f}"
        )

    def _on_reload_config(self):
        self.config.load()
        self.target_fps = self.config.get("target_fps", 30)
        self.min_delay = self.config.get("min_delay", 10)
        self.max_delay = self.config.get("max_delay", 120)
        self.status_label.config(text="Status: Config reloaded")

    def _on_mission_change(self, event=None):
        profile = self.mission_profile_var.get()
        self.swarm.set_mission_profile(profile)
        self.status_label.config(text=f"Status: Mission profile set to {profile}")

    def _on_snapshot(self):
        path = self.config.get("snapshot_path", "swarm_snapshot.json")
        try:
            snap = self.swarm.snapshot_state()
            with open(path, "w") as f:
                json.dump(snap, f, indent=2)
            self.status_label.config(text=f"Status: Snapshot saved to {path}")
        except Exception as e:
            logging.error(f"Snapshot failed: {e}")
            self.status_label.config(text="Status: Snapshot failed")

    def _on_add_drone(self):
        max_drones = self.config.get("max_drones", 12)
        if len(self.swarm.drones) >= max_drones:
            self.status_label.config(text="Status: Max drones reached")
            return
        i = len(self.swarm.drones)
        x = 200 + i * (400 // max(1, max_drones - 1))
        y = 200
        role = [DroneRole.SCOUT, DroneRole.OPERATOR, DroneRole.EXECUTOR][i % 3]
        personality = [DronePersonality.AGGRESSIVE, DronePersonality.BALANCED, DronePersonality.CAUTIOUS][i % 3]
        drone_id = f"drone_{i+1}"
        d = DataDrone(drone_id, self.swarm.drones[0].physics, self.swarm.drones[0].ui, (x, y), role, personality)
        self.swarm.drones.append(d)

        text_id = self.map_canvas.create_text(x, y, text=ROLE_VISUAL[role]["glyph"], fill=ROLE_VISUAL[role]["color"], font=("Consolas", 18))
        wire_id = self.map_canvas.create_line(x, y, x, y, fill=ROLE_VISUAL[role]["color"], width=1, smooth=True)
        self.drone_draw_map[d.id] = {"glyph": text_id, "wire": wire_id}

        ids = [dr.id for dr in self.swarm.drones]
        self.override_menu["values"] = ids
        self.inspect_menu["values"] = ids
        self.status_label.config(text=f"Status: Added {drone_id}")

    def _on_remove_drone(self):
        if not self.swarm.drones:
            return
        d = self.swarm.drones.pop()
        ids = [dr.id for dr in self.swarm.drones]
        self.override_menu["values"] = ids
        self.inspect_menu["values"] = ids
        if ids:
            self.override_target.set(ids[0])
            self.inspect_drone_var.set(ids[0])
        self.status_label.config(text=f"Status: Removed {d.id}")

    def _on_close(self):
        self.telemetry.stop()
        if self.swarm.network:
            self.swarm.network.stop()
        self.root.destroy()

    def _update_map_background(self):
        color = self.swarm.emotion.background_color()
        if self.swarm.emotion.anomaly_state:
            color = "#550000"
        self.map_canvas.config(bg=color)

    def _draw_drones_and_wires(self, reports: List[Dict]):
        for d in self.swarm.drones:
            draw_ids = self.drone_draw_map.get(d.id)
            if not draw_ids:
                continue
            text_id = draw_ids["glyph"]
            wire_id = draw_ids["wire"]

            x, y = d.screen_pos
            role_viz = ROLE_VISUAL[d.role]
            glyph = role_viz["glyph"]
            color = d.wire_model.color

            self.map_canvas.coords(text_id, x, y)
            self.map_canvas.itemconfig(text_id, text=glyph, fill=role_viz["color"])

            cps = d.wire_model.control_points
            if len(cps) == 3:
                self.map_canvas.coords(
                    wire_id,
                    cps[0][0], cps[0][1],
                    cps[1][0], cps[1][1],
                    cps[2][0], cps[2][1],
                )
            self.map_canvas.itemconfig(wire_id, fill=color, width=d.wire_model.thickness)

            if d.state == "override":
                self.map_canvas.itemconfig(text_id, fill="#FFFFFF")

    def _draw_consensus_vector(self):
        self.map_canvas.delete("consensus_vector")
        cx, cy = self.map_width / 2, self.map_height / 2
        vx, vy = self.swarm.brain.collective_vector
        conf = self.swarm.brain.confidence

        length = 80 * conf
        ex = cx + vx * length
        ey = cy + vy * length

        self.map_canvas.create_line(
            cx, cy, ex, ey,
            fill="#FFFFFF",
            width=2,
            arrow=tk.LAST,
            tags="consensus_vector"
        )

        pvx, pvy = self.swarm.brain.predicted_vector
        pex = cx + pvx * length
        pey = cy + pvy * length
        self.map_canvas.create_line(
            cx, cy, pex, pey,
            fill="#00FF00",
            width=1,
            dash=(4, 2),
            arrow=tk.LAST,
            tags="consensus_vector"
        )

    def _draw_waterfall(self):
        self.waterfall_canvas.delete("all")
        frames = self.swarm.history.frames
        if not frames:
            return

        w = self.waterfall_width
        h = self.waterfall_height
        n = len(frames)
        bar_height = max(1, h // n)

        for i, f in enumerate(frames[-(h // bar_height):]):
            pressure = f["congestion"] + f["risk"]
            if pressure < 2:
                color = "#001533"
            elif pressure < 5:
                color = "#004466"
            elif pressure < 8:
                color = "#AAAA00"
            else:
                color = "#FF0000"

            y0 = h - (i + 1) * bar_height
            y1 = h - i * bar_height
            self.waterfall_canvas.create_rectangle(
                0, y0, w, y1,
                fill=color,
                outline=""
            )

    def _draw_3d_view(self, reports: List[Dict]):
        if not self.canvas_3d:
            return
        self.canvas_3d.delete("all")
        w = int(self.canvas_3d.winfo_width())
        h = int(self.canvas_3d.winfo_height())
        cx, cy = w // 2, h // 2

        for d in self.swarm.drones:
            z = max(0.1, 1.0 - d.tether["tension"] / 10.0)
            sx = cx + (d.screen_pos[0] - 400) * z * 0.5
            sy = cy + (d.screen_pos[1] - 200) * z * 0.5
            size = 10 * z
            color = ROLE_VISUAL[d.role]["color"]
            self.canvas_3d.create_oval(
                sx - size, sy - size, sx + size, sy + size,
                fill=color, outline=""
            )

    def _adaptive_delay(self, frame_time: float) -> int:
        dt = frame_time - self.last_frame_time
        self.last_frame_time = frame_time

        if dt <= 0:
            dt = 0.001

        current_fps = 1.0 / dt
        target_fps = self.target_fps

        if current_fps < target_fps * 0.8:
            delay = self.max_delay
        elif current_fps > target_fps * 1.2:
            delay = self.min_delay
        else:
            delay = int((self.min_delay + self.max_delay) / 2)

        return max(self.min_delay, min(self.max_delay, delay))

    def _update_inspector(self, reports: Optional[List[Dict]] = None):
        drone_id = self.inspect_drone_var.get()
        if not drone_id:
            return
        report = None
        if reports:
            for r in reports:
                if r["id"] == drone_id:
                    report = r
                    break
        if report is None:
            for d in self.swarm.drones:
                if d.id == drone_id:
                    report = {
                        "id": d.id,
                        "role": d.role.value,
                        "personality": d.personality.value,
                        "genome": d.genome,
                        "performance": d.performance,
                        "anomalies": d.anomalies,
                        "state": d.state,
                    }
                    break
        if report is None:
            return

        self.inspect_text.delete("1.0", tk.END)
        self.inspect_text.insert(tk.END, json.dumps(report, indent=2))

    def _update_debug_panel(self, tele: Dict):
        info = {
            "cpu": tele["cpu"],
            "ram": tele["ram"],
            "disk": tele["disk"],
            "net": tele["net"],
            "mood": self.swarm.emotion.mood.value,
            "anomaly_level": self.swarm.emotion.anomaly_level.value,
            "anomaly_reason": self.swarm.emotion.last_anomaly_reason,
            "consensus": self.swarm.consensus.decision,
            "goal_vector": self.swarm.brain.goal_vector,
            "mission_profile": self.swarm.mission_profile,
            "remote_pressure": self.swarm.field.remote_pressure,
        }
        if not self.inspect_drone_var.get():
            self.inspect_text.delete("1.0", tk.END)
            self.inspect_text.insert(tk.END, json.dumps(info, indent=2))

    def _loop(self):
        start = time.time()

        tele = self.telemetry.snapshot()
        base_state = self.telemetry.to_swarm_state(tele)

        world_states = {}
        for d in self.swarm.drones:
            if self.swarm.mission_profile == "SCAN":
                dx = random.uniform(-1, 1)
                dy = random.uniform(-1, 1)
            elif self.swarm.mission_profile == "EXECUTE":
                dx = 0.8 if d.role == DroneRole.EXECUTOR else 0.2
                dy = 0.0
            else:
                dx = 0.5 if d.id.endswith("1") else -0.5 if d.id.endswith("3") else 0.0
                dy = 0.0 if d.id.endswith("1") else 0.5 if d.id.endswith("2") else -0.5

            world_states[d.id] = {
                **base_state,
                "target_dx": dx,
                "target_dy": dy,
                "intent_label": "scan" if d.role == DroneRole.SCOUT else "prepare" if d.role == DroneRole.OPERATOR else "execute",
            }

        reports = self.swarm.tick(world_states, tele)

        self._update_map_background()
        self._draw_drones_and_wires(reports)
        self._draw_consensus_vector()
        self._draw_waterfall()
        self._draw_3d_view(reports)
        self._update_inspector(reports)
        self._update_debug_panel(tele)

        self.status_label.config(
            text=f"Status: Mood={self.swarm.emotion.mood.value}, Anom={self.swarm.emotion.anomaly_level.value}, CPU={tele['cpu']:.1f}%, RAM={tele['ram']:.1f}%"
        )

        end = time.time()
        delay = self._adaptive_delay(end)
        self.root.after(delay, self._loop)

    def run(self):
        self.root.mainloop()

# =========================
# BOOTSTRAP
# =========================

def main():
    config = Config()
    physics = WaterPhysicsEngine()
    ui_driver = UIDriver(
        enabled=config.get("ui_automation_enabled", False),
        sandbox=config.get("sandbox_mode", True),
    )

    genome_store = GenomeStore(path=config.get("genome_path", "swarm_genomes.json"))

    initial_drones = config.get("initial_drones", 5)
    max_drones = config.get("max_drones", 12)
    initial_drones = max(1, min(initial_drones, max_drones))

    drones: List[DataDrone] = []
    for i in range(initial_drones):
        x = 200 + i * (400 // max(1, max_drones - 1))
        y = 200
        role = [DroneRole.SCOUT, DroneRole.OPERATOR, DroneRole.EXECUTOR][i % 3]
        personality = [DronePersonality.AGGRESSIVE, DronePersonality.BALANCED, DronePersonality.CAUTIOUS][i % 3]
        drone_id = f"drone_{i+1}"
        genome = genome_store.get(drone_id)
        drones.append(
            DataDrone(drone_id, physics, ui_driver, (x, y), role, personality, genome)
        )

    field = WaterField()
    history = WaterfallHistory(max_frames=config.get("max_events", 2000))
    brain = EmergentSwarmBrain()
    consensus = HiveMindConsensus()
    memory = SwarmMemory(
        max_events=config.get("max_events", 2000),
        path=config.get("memory_path", "swarm_memory.json"),
    )
    emotion_engine = SwarmEmotionEngine()
    bus = DroneBus()
    override = OperatorOverride()
    telemetry = SystemTelemetry()
    hw_scanner = HardwareScanner()
    network_node = SwarmNetworkNode(config)
    os_hooks = OSHooks()
    plugin_manager = PluginManager(config.get("plugins_dir", "plugins"))

    swarm = Swarm(
        drones, field, history, brain, consensus,
        memory, emotion_engine, bus, override, genome_store,
        network_node, os_hooks, hw_scanner, plugin_manager
    )

    cockpit = SwarmCockpit(swarm, telemetry, config)
    cockpit.run()

if __name__ == "__main__":
    main()

