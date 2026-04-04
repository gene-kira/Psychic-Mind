import importlib
import subprocess
import sys
import time
from enum import Enum
from typing import List, Dict, Tuple, Optional

import tkinter as tk
from tkinter import ttk

# =========================
# AUTOLOADER
# =========================

OPTIONAL_LIBS = {
    "uiautomation": "uiautomation",
    "numpy": "numpy",
}

def safe_import(name, pip_name=None):
    try:
        return importlib.import_module(name)
    except ImportError:
        if not pip_name:
            return None
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            return importlib.import_module(name)
        except Exception:
            return None

LIBS = {k: safe_import(k, v) for k, v in OPTIONAL_LIBS.items()}
uia = LIBS.get("uiautomation")
np = LIBS.get("numpy")

# =========================
# UI DRIVER
# =========================

class UIDriver:
    def __init__(self, enabled=True):
        self.enabled = enabled and (uia is not None)

    def click_button_by_name(self, name: str) -> bool:
        if not self.enabled:
            return False
        btn = uia.Control(searchDepth=5, Name=name)
        if btn:
            btn.Click()
            return True
        return False

    def type_text(self, name: str, text: str) -> bool:
        if not self.enabled:
            return False
        edit = uia.Control(searchDepth=5, Name=name)
        if edit:
            edit.SendKeys(text)
            return True
        return False

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
# WATER FIELD + HISTORY
# =========================

class WaterField:
    def __init__(self):
        self.global_congestion = 0.0
        self.global_risk = 0.0

    def update_from_swarm(self, swarm_state: List[Dict]):
        if not swarm_state:
            self.global_congestion = 0.0
            self.global_risk = 0.0
            return
        turbulences = [d["decision"]["turbulence"] for d in swarm_state]
        risks = [d["world"]["risk"] for d in swarm_state if "world" in d]
        self.global_congestion = sum(turbulences) / max(1, len(turbulences))
        self.global_risk = max(risks) if risks else 0.0

    def get_local_state(self, base_state: Dict) -> Dict:
        return {
            "priority": base_state.get("priority", 5),
            "congestion": base_state.get("congestion", 2) + self.global_congestion,
            "risk": max(base_state.get("risk", 1), self.global_risk),
        }

class WaterfallHistory:
    def __init__(self, max_frames: int = 200):
        self.frames: List[Dict] = []
        self.max_frames = max_frames

    def push(self, field: WaterField):
        self.frames.append({
            "congestion": field.global_congestion,
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
# EMERGENT SWARM BRAIN
# =========================

class EmergentSwarmBrain:
    def __init__(self):
        self.collective_vector: Tuple[float, float] = (0.0, 0.0)
        self.confidence: float = 0.0

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
# SWARM MEMORY + DREAMS
# =========================

class SwarmMemory:
    def __init__(self, max_events: int = 1000):
        self.events: List[Dict] = []
        self.max_events = max_events

    def record(self, event: Dict):
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def summarize(self) -> Dict:
        if not self.events:
            return {"avg_flow": 0.0, "avg_turbulence": 0.0}
        flows = [e.get("flow", 0.0) for e in self.events]
        turbs = [e.get("turbulence", 0.0) for e in self.events]
        return {
            "avg_flow": sum(flows) / len(flows),
            "avg_turbulence": sum(turbs) / len(turbs),
        }

    def dream_optimize(self):
        return self.summarize()

# =========================
# EMOTIONAL STATE MODEL
# =========================

class SwarmMood(Enum):
    CALM = "calm"
    FOCUSED = "focused"
    STRESSED = "stressed"
    PANICKED = "panicked"

class SwarmEmotionEngine:
    def __init__(self):
        self.mood = SwarmMood.CALM

    def update(self, field: WaterField, consensus_confidence: float):
        pressure = field.global_congestion + field.global_risk
        if pressure < 2 and consensus_confidence > 0.7:
            self.mood = SwarmMood.CALM
        elif pressure < 5:
            self.mood = SwarmMood.FOCUSED
        elif pressure < 8:
            self.mood = SwarmMood.STRESSED
        else:
            self.mood = SwarmMood.PANICKED

    def background_color(self) -> str:
        if self.mood == SwarmMood.CALM:
            return "#001533"
        if self.mood == SwarmMood.FOCUSED:
            return "#003344"
        if self.mood == SwarmMood.STRESSED:
            return "#665500"
        return "#660000"

# =========================
# DATA DRONE
# =========================

class DataDrone:
    def __init__(
        self,
        id: str,
        physics_engine: WaterPhysicsEngine,
        ui_driver: UIDriver,
        origin_pos: Tuple[float, float],
        role: DroneRole = DroneRole.SCOUT,
    ):
        self.id = id
        self.physics = physics_engine
        self.ui = ui_driver
        self.role = role

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

    def sense(self, world_state: Dict):
        self.intent = (
            world_state.get("target_dx", 0.0),
            world_state.get("target_dy", 0.0),
        )
        self.intent_label = world_state.get("intent_label", "none")

    def decide(self, local_state: Dict) -> Dict:
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

    def loop(self, world_state: Dict, local_state: Dict, bus: DroneBus) -> Dict:
        self.sense(world_state)
        decision = self.decide(local_state)
        self.act(decision)
        self.update_wire_visuals()
        self.evaluate_role()
        self.communicate(bus)

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
        }
        return report

# =========================
# SWARM + DREAM CYCLES + TELEPATHY
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
        self.telepathy_vector: Tuple[float, float] = (0.0, 0.0)
        self.telepathy_intent_label: Optional[str] = None

    def inject_telepathy(self, vector: Tuple[float, float], label: str):
        self.telepathy_vector = vector
        self.telepathy_intent_label = label

    def clear_telepathy(self):
        self.telepathy_intent_label = None
        self.telepathy_vector = (0.0, 0.0)

    def tick(self, world_states: Dict[str, Dict]) -> List[Dict]:
        reports: List[Dict] = []

        for d in self.drones:
            base_world = world_states.get(d.id, {})

            if self.telepathy_intent_label is not None:
                base_world["target_dx"] = self.telepathy_vector[0]
                base_world["target_dy"] = self.telepathy_vector[1]
                base_world["intent_label"] = self.telepathy_intent_label

            local_state = self.field.get_local_state(base_world)
            report = d.loop(base_world, local_state, self.bus)
            reports.append(report)

            self.memory.record({
                "drone_id": d.id,
                "flow": report["decision"]["flow"],
                "turbulence": report["decision"]["turbulence"],
            })

        self.field.update_from_swarm(reports)
        self.history.push(self.field)
        self.brain.update(reports)
        self.consensus.compute(reports)
        self.emotion.update(self.field, self.brain.confidence)

        return reports

    def dream_cycle(self):
        return self.memory.dream_optimize()

# =========================
# TKINTER COCKPIT (TACTICAL MAP)
# =========================

class SwarmCockpit:
    def __init__(self, swarm: Swarm):
        self.swarm = swarm

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
        self.map_canvas.grid(row=0, column=0, columnspan=2, sticky="nsew")

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

        self.main_frame.rowconfigure(0, weight=3)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        self._build_controls()

        self.drone_draw_map: Dict[str, Dict[str, int]] = {}
        self.last_frame_time = time.time()
        self.target_fps = 30
        self.min_delay = 10
        self.max_delay = 100

        self._init_drone_graphics()
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

        self.status_label = ttk.Label(self.control_frame, text="Status: Ready", anchor="w")
        self.status_label.grid(row=9, column=0, columnspan=2, sticky="ew")

        for i in range(10):
            self.control_frame.rowconfigure(i, weight=0)
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(1, weight=1)

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
        self.status_label.config(text=f"Status: Telepathy injected ({vec}, '{label}')")

    def _on_clear_telepathy(self):
        self.swarm.clear_telepathy()
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

    def _update_map_background(self):
        color = self.swarm.emotion.background_color()
        self.map_canvas.config(bg=color)

    def _draw_drones_and_wires(self, reports: List[Dict]):
        for d in self.swarm.drones:
            draw_ids = self.drone_draw_map[d.id]
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

    def _loop(self):
        start = time.time()

        world_states = {}
        for d in self.swarm.drones:
            world_states[d.id] = {
                "priority": 5 + (hash(d.id) % 3),
                "congestion": 2 + (hash(d.id) % 2),
                "risk": 1 + (hash(d.id) % 3),
                "target_dx": 0.5 if d.id.endswith("1") else -0.5 if d.id.endswith("3") else 0.0,
                "target_dy": 0.0 if d.id.endswith("1") else 0.5 if d.id.endswith("2") else -0.5,
                "intent_label": "scan" if d.role == DroneRole.SCOUT else "prepare" if d.role == DroneRole.OPERATOR else "execute",
            }

        reports = self.swarm.tick(world_states)

        self._update_map_background()
        self._draw_drones_and_wires(reports)
        self._draw_consensus_vector()
        self._draw_waterfall()

        self.status_label.config(
            text=f"Status: Mood={self.swarm.emotion.mood.value}, Consensus={self.swarm.consensus.decision}"
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
    physics = WaterPhysicsEngine()
    ui_driver = UIDriver(enabled=False)  # set True if you want real UIAutomation

    drones = [
        DataDrone("drone_1", physics, ui_driver, (200, 200), DroneRole.SCOUT),
        DataDrone("drone_2", physics, ui_driver, (400, 200), DroneRole.OPERATOR),
        DataDrone("drone_3", physics, ui_driver, (600, 200), DroneRole.EXECUTOR),
    ]

    field = WaterField()
    history = WaterfallHistory()
    brain = EmergentSwarmBrain()
    consensus = HiveMindConsensus()
    memory = SwarmMemory()
    emotion_engine = SwarmEmotionEngine()
    bus = DroneBus()
    override = OperatorOverride()

    swarm = Swarm(
        drones, field, history, brain, consensus,
        memory, emotion_engine, bus, override
    )

    cockpit = SwarmCockpit(swarm)
    cockpit.run()

if __name__ == "__main__":
    main()

