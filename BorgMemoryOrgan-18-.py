"""
BORG-OS MEMORY BACKBONE ORGAN (v11.3 – User Activity Governor + Context-Aware Priority + Recall Timeline HUD + Persistent Memory + Encrypted Swarm + UIAutomation)

New vs v11.2:
- UserActivityGovernorOrgan: derives a global user_mode from prediction + threat/anomaly
- BorgMemoryOrgan.user_mode: "idle", "browsing", "typing", "coding", "gaming", "configuring", "under_stress"
- Context-aware priority: inject() boosts priority based on user_mode + metric
"""

# =========================
# Autoloader
# =========================

def autoload(modules):
    import importlib, subprocess, sys
    loaded = {}
    for m in modules:
        try:
            loaded[m] = importlib.import_module(m)
        except ImportError:
            if m in ("tkinter",):
                loaded[m] = None
                continue
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", m])
                loaded[m] = importlib.import_module(m)
            except Exception:
                loaded[m] = None
    return loaded


mods = autoload([
    "random",
    "dataclasses",
    "typing",
    "tkinter",
    "json",
    "sys",
    "socket",
    "threading",
    "time",
    "psutil",
    "os",
    "getpass",
    "platform",
    "math",
    "pynvml",
    "etw",
    "requests",
    "pickle",
    "pathlib",
    "hmac",
    "hashlib",
    "secrets",
    "base64",
    "cryptography",
    "comtypes",
])

random = mods["random"]
dataclasses = mods["dataclasses"]
typing = mods["typing"]
tkinter = mods["tkinter"]
json = mods["json"]
sys = mods["sys"]
socket = mods["socket"]
threading = mods["threading"]
time = mods["time"]
psutil = mods["psutil"]
os = mods["os"]
getpass = mods["getpass"]
platform = mods["platform"]
math = mods["math"]
pynvml = mods["pynvml"]
etw = mods["etw"]
requests = mods["requests"]
pickle = mods["pickle"]
pathlib = mods["pathlib"]
hmac = mods["hmac"]
hashlib = mods["hashlib"]
secrets = mods["secrets"]
base64 = mods["base64"]
cryptography = mods["cryptography"]
comtypes = mods["comtypes"]

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

if cryptography is not None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

# =========================
# Core Data Structures
# =========================

@dataclass
class Packet:
    pid: int
    value: int
    energy: float = 1.0
    priority: float = 1.0
    meta: dict = field(default_factory=dict)


class MemoryTile:
    def __init__(self, size: int, tile_id: int):
        self.size = size
        self.tile_id = tile_id
        self.cells: List[Optional[Packet]] = [None] * size

    def inject(self, index: int, packet: Packet):
        idx = index % self.size
        self.cells[idx] = packet

    def step(self, policy) -> List[Tuple[int, Packet]]:
        new_cells = [None] * self.size
        outgoing: List[Tuple[int, Packet]] = []

        for i, pkt in enumerate(self.cells):
            if pkt is None:
                continue

            pkt = policy.pre_transform(self.tile_id, i, pkt)
            if pkt is None:
                continue

            direction, dest_tile = policy.choose_route(self.tile_id, i, pkt)
            target_index = i + direction

            if target_index < 0 or target_index >= self.size or dest_tile != self.tile_id:
                pkt = policy.transform(pkt)
                if pkt is not None:
                    outgoing.append((dest_tile, pkt))
            else:
                pkt = policy.transform(pkt)
                if pkt is not None:
                    new_cells[target_index] = pkt

        self.cells = new_cells
        return outgoing

    def snapshot(self) -> List[Optional[int]]:
        return [p.value if p else None for p in self.cells]

    def snapshot_energy(self) -> List[float]:
        return [p.energy if p else 0.0 for p in self.cells]


# =========================
# Policies + Water Physics
# =========================

class BasePolicy:
    def __init__(self, p_loss=0.1, p_double=0.1):
        self.p_loss = p_loss
        self.p_double = p_double

    def pre_transform(self, tile_id: int, index: int, packet: Packet) -> Optional[Packet]:
        return packet

    def choose_route(self, tile_id: int, index: int, packet: Packet) -> Tuple[int, int]:
        direction = random.choice([-1, 1])
        return direction, tile_id

    def transform(self, packet: Packet) -> Optional[Packet]:
        r = random.random()
        if r < self.p_loss:
            packet.value = int(packet.value * 0.5)
            packet.energy *= 0.7
        elif r < self.p_loss + self.p_double:
            packet.value *= 2
            packet.energy *= 1.2
        return packet


class WaterPhysicsEngine:
    def __init__(self):
        self.pressure = 0.0
        self.turbulence = 0.0

    def update(self, packet_count: int, threat: float, forget_prob: float, misroute_prob: float):
        density = min(1.0, packet_count / 256.0)
        self.pressure = 0.5 * density + 0.5 * threat
        self.turbulence = (forget_prob + misroute_prob) / 2.0

    def flow_bias(self, energy: float, priority: float) -> int:
        score = energy * 0.5 + priority * 0.5 + self.pressure - self.turbulence
        if score > 1.5:
            return 1
        elif score < 0.5:
            return -1
        return random.choice([-1, 1])


class IntelligenceLayer(BasePolicy):
    def __init__(self, base: BasePolicy, num_tiles: int, water_engine: WaterPhysicsEngine):
        super().__init__(base.p_loss, base.p_double)
        self.base = base
        self.num_tiles = num_tiles
        self.water = water_engine

    def choose_route(self, tile_id: int, index: int, packet: Packet) -> Tuple[int, int]:
        base_dir, _ = self.base.choose_route(tile_id, index, packet)
        water_dir = self.water.flow_bias(packet.energy, packet.priority)

        if packet.priority > 1.0:
            direction = water_dir
        elif packet.priority < 1.0:
            direction = -water_dir
        else:
            direction = base_dir

        if packet.energy > 1.5 and random.random() < 0.3:
            dest_tile = (tile_id + 1) % self.num_tiles
        elif packet.energy < 0.5 and random.random() < 0.3:
            dest_tile = (tile_id - 1) % self.num_tiles
        else:
            dest_tile = tile_id

        return direction, dest_tile

    def transform(self, packet: Packet) -> Optional[Packet]:
        pkt = self.base.transform(packet)
        if pkt is None:
            return None
        pkt.energy = max(0.1, min(pkt.energy, 5.0))
        return pkt


class AlzheimerLayer(IntelligenceLayer):
    def __init__(
        self,
        base: BasePolicy,
        num_tiles: int,
        water_engine: WaterPhysicsEngine,
        forget_prob=0.05,
        misroute_prob=0.1,
        meta_corrupt_prob=0.15,
        ghost_prob=0.05,
        progression_rate=0.001,
        max_ghosts=200,
    ):
        super().__init__(base, num_tiles, water_engine)
        self.forget_prob = forget_prob
        self.misroute_prob = misroute_prob
        self.meta_corrupt_prob = meta_corrupt_prob
        self.ghost_prob = ghost_prob
        self.progression_rate = progression_rate
        self.max_ghosts = max_ghosts
        self.ghost_buffer: List[Packet] = []
        self.time = 0
        self.last_ghost_event = False
        self.last_misroute_event = False

    def tick_progression(self):
        self.time += 1
        factor = 1 + self.progression_rate * self.time
        self.forget_prob = min(0.9, self.forget_prob * factor)
        self.misroute_prob = min(0.9, self.misroute_prob * factor)
        self.meta_corrupt_prob = min(0.9, self.meta_corrupt_prob * factor)

    def pre_transform(self, tile_id: int, index: int, packet: Packet) -> Optional[Packet]:
        if random.random() < self.forget_prob:
            if random.random() < self.ghost_prob and len(self.ghost_buffer) < self.max_ghosts:
                self.ghost_buffer.append(packet)
            return None

        if random.random() < self.meta_corrupt_prob:
            packet.meta["tag"] = "corrupted"
            packet.meta["time"] = random.randint(-10_000, 10_000)

        return packet

    def choose_route(self, tile_id: int, index: int, packet: Packet) -> Tuple[int, int]:
        if random.random() < self.misroute_prob:
            direction = random.choice([-3, -2, -1, 1, 2, 3])
            dest_tile = random.randint(0, self.num_tiles - 1)
            self.last_misroute_event = True
            return direction, dest_tile

        self.last_misroute_event = False
        return super().choose_route(tile_id, index, packet)

    def transform(self, packet: Packet) -> Optional[Packet]:
        self.last_ghost_event = False
        pkt = super().transform(packet)
        if pkt is None:
            return None

        if self.ghost_buffer and random.random() < self.ghost_prob:
            ghost = random.choice(self.ghost_buffer)
            ghost.value = int(ghost.value * random.uniform(0.5, 1.5))
            ghost.energy *= random.uniform(0.5, 1.2)
            ghost.priority *= random.uniform(0.5, 1.5)
            ghost.energy = max(0.1, min(ghost.energy, 5.0))
            ghost.priority = max(0.1, min(ghost.priority, 5.0))
            self.last_ghost_event = True
            return ghost

        return pkt


# =========================
# Privilege, Personality, Altered States
# =========================

class PrivilegeProfile:
    def __init__(self):
        self.user = getpass.getuser()
        self.os = platform.system()
        self.hostname = platform.node()
        self.is_admin = self._detect_admin()

    def _detect_admin(self) -> bool:
        try:
            if self.os == "Windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except Exception:
            return False


class PersonalityProfile:
    def __init__(self, mode: str = "stable"):
        self.mode = mode.lower().strip()

    def apply_to_policy(self, policy: AlzheimerLayer):
        if self.mode == "stable":
            policy.forget_prob *= 0.5
            policy.misroute_prob *= 0.5
            policy.meta_corrupt_prob *= 0.7
            policy.progression_rate *= 0.5
        elif self.mode == "chaotic":
            policy.forget_prob *= 1.5
            policy.misroute_prob *= 2.0
            policy.meta_corrupt_prob *= 1.5
            policy.ghost_prob *= 1.5
        elif self.mode == "paranoid":
            policy.forget_prob *= 0.8
            policy.misroute_prob *= 1.5
            policy.ghost_prob *= 2.0
            policy.meta_corrupt_prob *= 1.2
        elif self.mode == "analytical":
            policy.forget_prob *= 0.4
            policy.misroute_prob *= 0.6
            policy.meta_corrupt_prob *= 0.5
            policy.progression_rate *= 0.3


class AlteredStateManager:
    def __init__(self, state: str = "focus"):
        self.state = state

    def apply(self, organ: "BorgMemoryOrgan"):
        p = organ.policy
        if self.state == "focus":
            p.forget_prob *= 0.7
            p.misroute_prob *= 0.7
        elif self.state == "dream":
            p.ghost_prob *= 2.0
            p.misroute_prob *= 1.5
        elif self.state == "panic":
            p.misroute_prob *= 1.8
            p.meta_corrupt_prob *= 1.3
        elif self.state == "trance":
            p.progression_rate *= 0.3
            organ.water_engine.turbulence *= 0.5


# =========================
# Auto Mode Selector
# =========================

class AutoModeSelector:
    def __init__(self):
        self.mode = "balanced"
        self.last_eval = 0.0
        self.eval_interval = 10.0

    def evaluate(self):
        now = time.time()
        if now - self.last_eval < self.eval_interval:
            return self.mode
        self.last_eval = now

        try:
            cpu_load = psutil.cpu_percent(interval=0.05)
            cores = psutil.cpu_count(logical=True) or 4
            mem = psutil.virtual_memory().percent
            temp = None
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                cpu_temps = temps.get("coretemp") or []
                if cpu_temps:
                    temp = sum(t.current for t in cpu_temps) / len(cpu_temps)

            if cpu_load > 80 or mem > 85 or (temp is not None and temp > 80):
                self.mode = "lightweight"
            elif cpu_load < 40 and mem < 70 and cores >= 8 and (temp is None or temp < 70):
                self.mode = "max"
            else:
                self.mode = "balanced"
        except Exception:
            self.mode = "balanced"

        return self.mode

    def scale_interval(self, base_interval: float) -> float:
        mode = self.evaluate()
        if mode == "lightweight":
            return base_interval * 1.8
        elif mode == "max":
            return max(0.2, base_interval * 0.6)
        return base_interval

    def gui_delay_ms(self, base_ms: int) -> int:
        mode = self.evaluate()
        if mode == "lightweight":
            factor = 2.0
        elif mode == "max":
            factor = 0.7
        else:
            factor = 1.0
        delay = int(base_ms * factor)
        return max(40, min(delay, 200))


# =========================
# Borg Memory Organ
# =========================

class BorgMemoryOrgan:
    def __init__(
        self,
        organ_id: str,
        num_tiles=4,
        tile_size=8,
        base_policy=None,
        profile: Optional[PrivilegeProfile] = None,
        personality: Optional[PersonalityProfile] = None,
        altered_state: Optional[AlteredStateManager] = None,
        mode_selector: Optional[AutoModeSelector] = None,
        state_path: Optional[str] = None,
    ):
        self.organ_id = organ_id
        self.num_tiles = num_tiles
        self.tiles = [MemoryTile(tile_size, t) for t in range(num_tiles)]
        self.water_engine = WaterPhysicsEngine()
        base = base_policy or BasePolicy(p_loss=0.1, p_double=0.1)
        self.policy = AlzheimerLayer(base, num_tiles=num_tiles, water_engine=self.water_engine)
        self.next_pid = 1
        self.debug_log: List[str] = []
        self.profile = profile or PrivilegeProfile()
        self.personality = personality or PersonalityProfile("stable")
        self.altered_state = altered_state or AlteredStateManager("focus")
        self.mode_selector = mode_selector or AutoModeSelector()

        # Telemetry snapshots
        self.last_telemetry_inject_time = 0.0
        self.last_cpu = 0
        self.last_mem = 0
        self.last_disk_read = 0
        self.last_disk_write = 0
        self.last_gpu_util = 0
        self.last_gpu_temp = 0
        self.last_gpu_power = 0
        self.last_gpu_fan = 0
        self.last_net_threat = 0

        # Rolling histories
        self.history_cpu: List[int] = []
        self.history_mem: List[int] = []
        self.history_gpu: List[int] = []
        self.history_threat: List[float] = []

        # Keepalive
        self.keepalive_enabled = True
        self.keepalive_interval_ticks = 5
        self._tick_counter = 0

        # Long-term memory
        self.long_term_memory: List[Packet] = []
        self.threat_level = 0.0
        self.health_history: List[float] = []

        # ETW counters
        self.etw_net_events = 0
        self.etw_proc_events = 0
        self.etw_disk_events = 0

        # Threat lineage
        self.threat_lineage: List[dict] = []
        self.firewall_organ = None

        # Profiler
        self.last_tick_duration = 0.0

        # Persistence
        self.state_path = state_path or str(pathlib.Path.cwd() / f"{self.organ_id}_state.pkl")

        # Anomaly tracking
        self.anomaly_score = 0.0
        self.anomaly_events: List[dict] = []

        # Predictive / UI state
        self.prediction: str = ""
        self.missing_details_hint: str = ""

        # Recall timeline
        self.recall_events: List[dict] = []

        # User activity mode (governor)
        self.user_mode: str = "idle"

        self._apply_privilege_profile()
        self._apply_personality_profile()
        self._apply_altered_state()

    # ---------- Persistence ----------

    def save_state(self):
        try:
            state = {
                "organ_id": self.organ_id,
                "tiles": [[(p.value, p.energy, p.priority, p.meta) if p else None for p in tile.cells] for tile in self.tiles],
                "policy": {
                    "forget_prob": self.policy.forget_prob,
                    "misroute_prob": self.policy.misroute_prob,
                    "meta_corrupt_prob": self.policy.meta_corrupt_prob,
                    "ghost_prob": self.policy.ghost_prob,
                    "progression_time": self.policy.time,
                },
                "threat_level": self.threat_level,
                "history_cpu": self.history_cpu,
                "history_mem": self.history_mem,
                "history_gpu": self.history_gpu,
                "history_threat": self.history_threat,
                "long_term_memory": [
                    (p.value, p.energy, p.priority, p.meta)
                    for p in self.long_term_memory[-2000:]
                ],
                "anomaly_score": self.anomaly_score,
                "recall_events": self.recall_events[-50:],
                "user_mode": self.user_mode,
            }
            with open(self.state_path, "wb") as f:
                pickle.dump(state, f)
            self.debug_log.append(f"State saved to {self.state_path}")
        except Exception as e:
            self.debug_log.append(f"State save error: {e}")

    def load_state(self):
        try:
            p = pathlib.Path(self.state_path)
            if not p.exists():
                self.debug_log.append("No saved state found.")
                return
            with open(self.state_path, "rb") as f:
                state = pickle.load(f)
            self._apply_loaded_state(state)
            self.debug_log.append(f"State loaded from {self.state_path}")
        except Exception as e:
            self.debug_log.append(f"State load error: {e}")

    def _apply_loaded_state(self, state: dict):
        tiles_state = state.get("tiles", [])
        for t_idx, tile_state in enumerate(tiles_state):
            if t_idx >= self.num_tiles:
                break
            tile = self.tiles[t_idx]
            for c_idx, cell in enumerate(tile_state):
                if c_idx >= tile.size:
                    break
                if cell is None:
                    tile.cells[c_idx] = None
                else:
                    value, energy, priority, meta = cell
                    tile.cells[c_idx] = Packet(
                        pid=self.next_pid,
                        value=value,
                        energy=energy,
                        priority=priority,
                        meta=meta,
                    )
                    self.next_pid += 1

        pol = state.get("policy", {})
        self.policy.forget_prob = pol.get("forget_prob", self.policy.forget_prob)
        self.policy.misroute_prob = pol.get("misroute_prob", self.policy.misroute_prob)
        self.policy.meta_corrupt_prob = pol.get("meta_corrupt_prob", self.policy.meta_corrupt_prob)
        self.policy.ghost_prob = pol.get("ghost_prob", self.policy.ghost_prob)
        self.policy.time = pol.get("progression_time", self.policy.time)

        self.threat_level = state.get("threat_level", self.threat_level)
        self.history_cpu = state.get("history_cpu", [])
        self.history_mem = state.get("history_mem", [])
        self.history_gpu = state.get("history_gpu", [])
        self.history_threat = state.get("history_threat", [])
        self.anomaly_score = state.get("anomaly_score", 0.0)

        ltm = state.get("long_term_memory", [])
        self.long_term_memory = []
        for i, (v, e, pr, m) in enumerate(ltm):
            self.long_term_memory.append(
                Packet(
                    pid=self.next_pid + i,
                    value=v,
                    energy=e,
                    priority=pr,
                    meta=m,
                )
            )
        self.next_pid += len(self.long_term_memory)

        self.recall_events = state.get("recall_events", [])
        self.user_mode = state.get("user_mode", "idle")

    # ---------- Core behavior ----------

    def _apply_privilege_profile(self):
        if self.profile.is_admin:
            self.policy.forget_prob *= 0.5
            self.policy.misroute_prob *= 0.5
            self.policy.meta_corrupt_prob *= 0.5
        else:
            self.policy.forget_prob *= 1.2
            self.policy.misroute_prob *= 1.2

    def _apply_personality_profile(self):
        self.personality.apply_to_policy(self.policy)

    def _apply_altered_state(self):
        self.altered_state.apply(self)

    def attach_firewall(self, fw):
        self.firewall_organ = fw

    def export_state(self) -> dict:
        return {
            "organ_id": self.organ_id,
            "tiles": [[(p.value, p.energy) if p else None for p in tile.cells] for tile in self.tiles],
            "time": self.policy.time,
            "threat_level": self.threat_level,
            "personality": self.personality.mode,
            "altered_state": self.altered_state.state,
            "user_mode": self.user_mode,
        }

    def import_state(self, state: dict):
        tiles_state = state.get("tiles", [])
        for t_idx, tile_state in enumerate(tiles_state):
            if t_idx >= self.num_tiles:
                break
            tile = self.tiles[t_idx]
            for c_idx, cell in enumerate(tile_state):
                if c_idx >= tile.size:
                    break
                if cell is None:
                    tile.cells[c_idx] = None
                else:
                    value, energy = cell
                    tile.cells[c_idx] = Packet(
                        pid=self.next_pid,
                        value=value,
                        energy=energy,
                        priority=1.0,
                        meta={"source": state.get("organ_id", "import")},
                    )
                    self.next_pid += 1

    def swarm_broadcast_firewall_blacklist(self):
        if self.firewall_organ:
            return {
                "type": "blacklist_update",
                "ips": self.firewall_organ.export_blacklist()
            }
        return None

    def set_forget_prob(self, value: float):
        self.policy.forget_prob = max(0.0, min(value, 1.0))

    def set_misroute_prob(self, value: float):
        self.policy.misroute_prob = max(0.0, min(value, 1.0))

    def set_meta_corrupt_prob(self, value: float):
        self.policy.meta_corrupt_prob = max(0.0, min(value, 1.0))

    def set_ghost_prob(self, value: float):
        self.policy.ghost_prob = max(0.0, min(value, 1.0))

    def set_loss_prob(self, value: float):
        self.policy.base.p_loss = max(0.0, min(value, 1.0))

    def set_double_prob(self, value: float):
        self.policy.base.p_double = max(0.0, min(value, 1.0))

    # ---------- User activity + priority logic ----------

    def update_user_mode(self, mode: str):
        mode = (mode or "").strip().lower()
        if not mode:
            return
        self.user_mode = mode

    def priority_boost(self, metric: Optional[str]) -> float:
        """
        Context-aware priority boost based on user_mode and metric.
        This is the core of better data-flow control.
        """
        if not metric:
            return 1.0

        m = metric.lower()
        u = self.user_mode

        # Default: no change
        boost = 1.0

        if u == "gaming":
            if m in ("gpu_util", "gpu_temp", "gpu_power", "gpu_fan", "game_load", "net_threat", "net_flow"):
                boost = 2.0
            elif m in ("ui_context", "missing_details"):
                boost = 0.7

        elif u == "typing":
            if m in ("ui_context", "missing_details"):
                boost = 2.0
            elif m in ("gpu_util", "game_load"):
                boost = 0.7

        elif u == "browsing":
            if m in ("browser_load", "net_flow", "net_threat"):
                boost = 1.8

        elif u == "coding":
            if m in ("ui_context", "cpu", "mem"):
                boost = 1.5

        elif u == "configuring":
            if m in ("ui_context", "missing_details", "anomaly_score"):
                boost = 1.7

        elif u == "under_stress":
            if m in ("anomaly_score", "net_threat", "recall", "prediction"):
                boost = 2.2

        elif u == "idle":
            if m in ("cpu", "mem", "gpu_util", "net_threat", "anomaly_score"):
                boost = 0.8

        return boost

    def inject(self, tile_id: int, index: int, value: int, priority: float = 1.0, meta=None):
        meta = meta or {}
        metric = meta.get("metric")
        boost = self.priority_boost(metric)
        eff_priority = max(0.1, min(priority * boost, 3.0))

        pkt = Packet(
            pid=self.next_pid,
            value=value,
            energy=1.0,
            priority=eff_priority,
            meta=meta,
        )
        self.next_pid += 1
        self.tiles[tile_id % self.num_tiles].inject(index, pkt)

        metric = pkt.meta.get("metric")
        if metric == "cpu":
            self.last_cpu = value
            self.last_telemetry_inject_time = time.time()
            self._push_history(self.history_cpu, value)
        elif metric == "mem":
            self.last_mem = value
            self.last_telemetry_inject_time = time.time()
            self._push_history(self.history_mem, value)
        elif metric == "disk_read":
            self.last_disk_read = value
        elif metric == "disk_write":
            self.last_disk_write = value
        elif metric == "gpu_util":
            self.last_gpu_util = value
            self._push_history(self.history_gpu, value)
        elif metric == "gpu_temp":
            self.last_gpu_temp = value
        elif metric == "gpu_power":
            self.last_gpu_power = value
        elif metric == "gpu_fan":
            self.last_gpu_fan = value
        elif metric == "net_threat":
            self.last_net_threat = value
            self.threat_lineage.append(pkt.meta)
            self.threat_lineage = self.threat_lineage[-20:]
        elif metric == "anomaly_score":
            self.anomaly_score = value
            self.anomaly_events.append(pkt.meta)
            self.anomaly_events = self.anomaly_events[-50:]
        elif metric == "prediction":
            self.prediction = pkt.meta.get("prediction", "")
        elif metric == "missing_details":
            self.missing_details_hint = pkt.meta.get("hint", "")
        elif metric == "recall":
            event = {
                "timestamp": pkt.meta.get("timestamp", time.time()),
                "score": pkt.meta.get("score", 0.0),
                "value": pkt.value,
                "context": pkt.meta.get("original_context", {}),
            }
            self.recall_events.append(event)
            self.recall_events = self.recall_events[-50:]

        return pkt.pid

    def _push_history(self, hist: List, value, max_len: int = 60):
        hist.append(value)
        if len(hist) > max_len:
            del hist[0]

    def _inject_keepalive(self):
        for t_id in range(self.num_tiles):
            self.inject(
                tile_id=t_id,
                index=random.randint(0, len(self.tiles[t_id].cells) - 1),
                value=0,
                priority=0.5,
                meta={"metric": "keepalive"},
            )

    def _consolidate_long_term_memory(self):
        if len(self.policy.ghost_buffer) > 10:
            for _ in range(min(5, len(self.policy.ghost_buffer))):
                pkt = self.policy.ghost_buffer.pop(0)
                pkt.meta["ltm"] = True
                pkt.meta.setdefault("timestamp", time.time())
                pkt.meta.setdefault("context", {
                    "cpu": self.last_cpu,
                    "mem": self.last_mem,
                    "gpu": self.last_gpu_util,
                    "threat": self.threat_level,
                    "prediction": self.prediction,
                })
                self.long_term_memory.append(pkt)
            if len(self.long_term_memory) > 5000:
                self.long_term_memory = self.long_term_memory[-5000:]

    def _update_threat_level(self):
        cpu_threat = max(0.0, (self.last_cpu - 70) / 30.0)
        mem_threat = max(0.0, (self.last_mem - 80) / 20.0)
        gpu_threat = max(0.0, (self.last_gpu_util - 80) / 20.0)
        disk_threat = max(0.0, (self.last_disk_write - 80) / 20.0)

        entropy = (self.policy.forget_prob + self.policy.misroute_prob) / 2.0
        etw_activity = min(
            1.0,
            (self.etw_net_events + self.etw_proc_events + self.etw_disk_events) / 500.0,
        )

        load_threat = max(cpu_threat, mem_threat, gpu_threat, disk_threat)

        self.threat_level = max(
            0.0,
            min(1.0, 0.3 * load_threat + 0.4 * entropy + 0.3 * etw_activity),
        )
        self._push_history(self.history_threat, self.threat_level * 100)

    def _learn_from_health(self):
        health = max(0.0, 1.0 - self.policy.forget_prob)
        self.health_history.append(health)
        if len(self.health_history) > 50:
            self.health_history.pop(0)

        avg_health = sum(self.health_history) / len(self.health_history)

        if avg_health < 0.4 and self.threat_level < 0.3:
            self.policy.forget_prob = max(0.0, self.policy.forget_prob * 0.95)
        if self.threat_level > 0.6:
            self.policy.misroute_prob = min(0.9, self.policy.misroute_prob * 1.05)

    def tick(self):
        t0 = time.time()
        self.policy.tick_progression()
        self._tick_counter += 1

        packet_count = sum(1 for t in self.tiles for c in t.cells if c is not None)
        self.water_engine.update(
            packet_count,
            self.threat_level,
            self.policy.forget_prob,
            self.policy.misroute_prob,
        )

        if self.keepalive_enabled and self._tick_counter % self.keepalive_interval_ticks == 0:
            self._inject_keepalive()

        outgoing_by_tile: Dict[int, List[Packet]] = {t: [] for t in range(self.num_tiles)}

        for t_id, tile in enumerate(self.tiles):
            outgoing = tile.step(self.policy)
            for dest_tile, pkt in outgoing:
                outgoing_by_tile[dest_tile % self.num_tiles].append(pkt)

        for dest_tile, packets in outgoing_by_tile.items():
            if not packets:
                continue
            tile = self.tiles[dest_tile]
            for pkt in packets:
                tile.inject(0, pkt)

        self._consolidate_long_term_memory()
        self._update_threat_level()
        self._learn_from_health()

        self.last_tick_duration = time.time() - t0

    def snapshot_values(self) -> List[List[Optional[int]]]:
        return [tile.snapshot() for tile in self.tiles]

    def snapshot_energy(self) -> List[List[float]]:
        return [tile.snapshot_energy() for tile in self.tiles]

    # ---------- IPC ----------

    def handle_ipc_message(self, msg: dict) -> dict:
        topic = msg.get("topic")
        cmd = msg.get("cmd")
        args = msg.get("args", {}) or {}
        mid = msg.get("id")

        resp: Dict[str, Any] = {"id": mid, "ok": True, "topic": topic, "cmd": cmd, "result": {}}

        try:
            if topic == "config":
                v = float(args["value"])
                if cmd == "set_forget_prob":
                    self.set_forget_prob(v)
                elif cmd == "set_misroute_prob":
                    self.set_misroute_prob(v)
                elif cmd == "set_meta_corrupt_prob":
                    self.set_meta_corrupt_prob(v)
                elif cmd == "set_ghost_prob":
                    self.set_ghost_prob(v)
                elif cmd == "set_loss_prob":
                    self.set_loss_prob(v)
                elif cmd == "set_double_prob":
                    self.set_double_prob(v)
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_config_cmd"

            elif topic == "inject" and cmd == "packet":
                tile_id = int(args.get("tile_id", 0))
                index = int(args.get("index", 0))
                value = int(args.get("value", 0))
                priority = float(args.get("priority", 1.0))
                meta = args.get("meta", {})
                pid = self.inject(tile_id, index, value, priority, meta)
                resp["result"]["pid"] = pid

            elif topic == "snapshot":
                if cmd == "values":
                    resp["result"]["values"] = self.snapshot_values()
                elif cmd == "energies":
                    resp["result"]["energies"] = self.snapshot_energy()
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_snapshot_cmd"

            elif topic == "state":
                if cmd == "full":
                    resp["result"]["state"] = self.export_state()
                elif cmd == "save":
                    self.save_state()
                elif cmd == "load":
                    self.load_state()
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_state_cmd"

            elif topic == "tick" and cmd == "step":
                steps = int(args.get("steps", 1))
                for _ in range(steps):
                    self.tick()

            elif topic == "swarm":
                if cmd == "import_state":
                    self.import_state(args.get("state", {}))
                elif cmd == "adjust":
                    param = args.get("param")
                    value = args.get("value")
                    if param == "forget_prob":
                        self.set_forget_prob(float(value))
                    elif param == "misroute_prob":
                        self.set_misroute_prob(float(value))
                    elif param == "meta_corrupt_prob":
                        self.set_meta_corrupt_prob(float(value))
                    elif param == "ghost_prob":
                        self.set_ghost_prob(float(value))
                    else:
                        resp["ok"] = False
                        resp["error"] = "unknown_swarm_param"
                elif cmd == "message":
                    msg_obj = args.get("msg", {})
                    if self.firewall_organ:
                        self.firewall_organ.receive_swarm_message(msg_obj)
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_swarm_cmd"

            elif topic == "debug":
                if cmd == "log":
                    resp["result"]["log"] = self.debug_log[-50:]
                elif cmd == "health":
                    resp["result"]["health"] = {
                        "time": self.policy.time,
                        "forget_prob": self.policy.forget_prob,
                        "misroute_prob": self.policy.misroute_prob,
                        "meta_corrupt_prob": self.policy.meta_corrupt_prob,
                        "ghost_prob": self.policy.ghost_prob,
                        "ghost_buffer_size": len(self.policy.ghost_buffer),
                        "threat_level": self.threat_level,
                        "last_cpu": self.last_cpu,
                        "last_mem": self.last_mem,
                        "last_disk_read": self.last_disk_read,
                        "last_disk_write": self.last_disk_write,
                        "last_gpu_util": self.last_gpu_util,
                        "last_gpu_temp": self.last_gpu_temp,
                        "last_gpu_power": self.last_gpu_power,
                        "last_gpu_fan": self.last_gpu_fan,
                        "last_net_threat": self.last_net_threat,
                        "etw_net_events": self.etw_net_events,
                        "etw_proc_events": self.etw_proc_events,
                        "etw_disk_events": self.etw_disk_events,
                        "mode": self.mode_selector.mode,
                        "tick_duration": self.last_tick_duration,
                        "anomaly_score": self.anomaly_score,
                        "prediction": self.prediction,
                        "missing_details_hint": self.missing_details_hint,
                        "user_mode": self.user_mode,
                    }
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_debug_cmd"

            elif topic == "control_panel":
                if cmd == "get_all":
                    resp["result"]["params"] = {
                        "forget_prob": self.policy.forget_prob,
                        "misroute_prob": self.policy.misroute_prob,
                        "meta_corrupt_prob": self.policy.meta_corrupt_prob,
                        "ghost_prob": self.policy.ghost_prob,
                        "loss_prob": self.policy.base.p_loss,
                        "double_prob": self.policy.base.p_double,
                        "threat_level": self.threat_level,
                        "personality": self.personality.mode,
                        "altered_state": self.altered_state.state,
                        "mode": self.mode_selector.mode,
                        "anomaly_score": self.anomaly_score,
                        "prediction": self.prediction,
                        "missing_details_hint": self.missing_details_hint,
                        "user_mode": self.user_mode,
                    }
                elif cmd == "set":
                    param = args.get("param")
                    value = args.get("value")
                    if param == "forget_prob":
                        self.set_forget_prob(float(value))
                    elif param == "misroute_prob":
                        self.set_misroute_prob(float(value))
                    elif param == "meta_corrupt_prob":
                        self.set_meta_corrupt_prob(float(value))
                    elif param == "ghost_prob":
                        self.set_ghost_prob(float(value))
                    elif param == "loss_prob":
                        self.set_loss_prob(float(value))
                    elif param == "double_prob":
                        self.set_double_prob(float(value))
                    elif param == "personality":
                        self.personality = PersonalityProfile(str(value))
                        self._apply_personality_profile()
                    elif param == "altered_state":
                        self.altered_state = AlteredStateManager(str(value))
                        self._apply_altered_state()
                    elif param == "user_mode":
                        self.update_user_mode(str(value))
                    else:
                        resp["ok"] = False
                        resp["error"] = "unknown_control_param"
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_control_cmd"

            elif topic == "meta":
                if cmd == "info":
                    resp["result"]["info"] = {
                        "organ_id": self.organ_id,
                        "num_tiles": self.num_tiles,
                        "tile_size": len(self.tiles[0].cells),
                        "time": self.policy.time,
                        "user": self.profile.user,
                        "is_admin": self.profile.is_admin,
                        "hostname": self.profile.hostname,
                        "os": self.profile.os,
                        "personality": self.personality.mode,
                        "altered_state": self.altered_state.state,
                        "mode": self.mode_selector.mode,
                        "user_mode": self.user_mode,
                    }
                elif cmd == "ping":
                    resp["result"]["pong"] = True
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_meta_cmd"

            else:
                resp["ok"] = False
                resp["error"] = "unknown_topic"

        except Exception as e:
            resp["ok"] = False
            resp["error"] = str(e)

        return resp


# =========================
# Telemetry Organs (Base)
# =========================

class BaseOrganThread:
    def __init__(self, memory: BorgMemoryOrgan, base_interval_sec: float):
        self.memory = memory
        self.base_interval_sec = base_interval_sec
        self._stop = False
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_wrapper, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop = True

    def _run_wrapper(self):
        while not self._stop:
            try:
                self._run_once()
            except Exception as e:
                self.memory.debug_log.append(f"{self.__class__.__name__} error: {e}")
            interval = self._scaled_interval()
            time.sleep(interval)

    def _scaled_interval(self) -> float:
        """
        Default scaling: use AutoModeSelector.
        UserActivityGovernor can still influence behavior via skipping work inside _run_once.
        """
        return self.memory.mode_selector.scale_interval(self.base_interval_sec)

    def _run_once(self):
        raise NotImplementedError


# =========================
# Telemetry Organs (Real)
# =========================

class SystemTelemetryOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 2.0):
        super().__init__(memory, interval_sec)

    def _run_once(self):
        cpu = int(psutil.cpu_percent())
        mem = int(psutil.virtual_memory().percent)
        self.memory.inject(0, 0, cpu, 1.2, {"metric": "cpu"})
        self.memory.inject(1, 0, mem, 1.0, {"metric": "mem"})


class DiskTelemetryOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 3.0):
        super().__init__(memory, interval_sec)
        self._last = psutil.disk_io_counters() if psutil else None

    def _run_once(self):
        if self._last is None:
            return
        now = psutil.disk_io_counters()
        read_delta = max(0, now.read_bytes - self._last.read_bytes)
        write_delta = max(0, now.write_bytes - self._last.write_bytes)
        self._last = now

        read_mb = int(read_delta / (1024 * 1024))
        write_mb = int(write_delta / (1024 * 1024))

        self.memory.inject(2, 0, read_mb, 1.0, {"metric": "disk_read"})
        self.memory.inject(3, 0, write_mb, 1.2, {"metric": "disk_write"})


class ProcessBehaviorOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 5.0):
        super().__init__(memory, interval_sec)

    def _run_once(self):
        procs = list(psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]))
        high_cpu = sum(1 for p in procs if p.info["cpu_percent"] and p.info["cpu_percent"] > 50)
        high_mem = sum(1 for p in procs if p.info["memory_percent"] and p.info["memory_percent"] > 10)
        score = min(100, high_cpu * 5 + high_mem * 3)
        self.memory.inject(0, 2, score, 1.0, {"metric": "proc_stress"})


class WindowsEventLogOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 10.0):
        super().__init__(memory, interval_sec)

    def start(self):
        if platform.system() == "Windows":
            super().start()

    def _run_once(self):
        self.memory.inject(1, 2, 1, 0.8, {"metric": "eventlog_pulse"})


class PacketSnifferOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 3.0):
        super().__init__(memory, interval_sec)
        self._last = psutil.net_io_counters() if psutil else None

    def _run_once(self):
        if self._last is None:
            return
        now = psutil.net_io_counters()
        sent_delta = max(0, now.packets_sent - self._last.packets_sent)
        recv_delta = max(0, now.packets_recv - self._last.packets_recv)
        self._last = now
        score = min(100, int((sent_delta + recv_delta) / 10))
        self.memory.inject(0, 3, score, 1.0, {"metric": "net_flow"})


class GPUTelemetryOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 2.0):
        super().__init__(memory, interval_sec)
        self._gpu_available = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_available = True
            except Exception:
                self._gpu_available = False

    def start(self):
        if self._gpu_available:
            super().start()

    def _run_once(self):
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)

        gpu_util = int(util.gpu)
        gpu_mem_used = int(mem.used / (1024 * 1024))
        gpu_temp = int(temp)

        self.memory.inject(0, 0, gpu_util, 1.2, {"metric": "gpu_util"})
        self.memory.inject(1, 0, gpu_mem_used, 1.0, {"metric": "gpu_mem"})
        self.memory.inject(2, 0, gpu_temp, 1.0, {"metric": "gpu_temp"})

        try:
            power = int(pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000)
            self.memory.inject(3, 0, power, 1.0, {"metric": "gpu_power"})
        except Exception:
            pass

        try:
            fan = int(pynvml.nvmlDeviceGetFanSpeed(self._handle))
            self.memory.inject(0, 1, fan, 1.0, {"metric": "gpu_fan"})
        except Exception:
            pass


class BrowserTelemetryOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 5.0, browsers=None):
        super().__init__(memory, interval_sec)
        self.browsers = set(browsers or ["chrome.exe", "msedge.exe", "firefox.exe"])

    def _run_once(self):
        procs = psutil.process_iter(["name", "cpu_percent"])
        cpu_sum = 0
        for p in procs:
            name = (p.info["name"] or "").lower()
            if any(b in name for b in self.browsers):
                cpu_sum += p.info["cpu_percent"] or 0
        score = min(100, int(cpu_sum))
        self.memory.inject(1, 3, score, 1.0, {"metric": "browser_load"})


class GameTelemetryOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 5.0, games=None):
        super().__init__(memory, interval_sec)
        self.games = set(games or ["steam.exe", "epicgameslauncher.exe"])

    def _run_once(self):
        procs = psutil.process_iter(["name", "cpu_percent"])
        cpu_sum = 0
        for p in procs:
            name = (p.info["name"] or "").lower()
            if any(g in name for g in self.games):
                cpu_sum += p.info["cpu_percent"] or 0
        score = min(100, int(cpu_sum))
        self.memory.inject(2, 3, score, 1.0, {"metric": "game_load"})


class HardwareSensorsOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 10.0):
        super().__init__(memory, interval_sec)

    def _run_once(self):
        temps = getattr(psutil, "sensors_temperatures", lambda: {})()
        cpu_temps = temps.get("coretemp") or []
        if cpu_temps:
            avg = int(sum(t.current for t in cpu_temps) / len(cpu_temps))
            self.memory.inject(3, 3, avg, 1.0, {"metric": "cpu_temp"})


class CloudAPIOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 30.0, url: str = ""):
        super().__init__(memory, interval_sec)
        self.url = url

    def start(self):
        if self.url and requests is not None:
            super().start()

    def _run_once(self):
        r = requests.get(self.url, timeout=5)
        val = len(r.content)
        score = min(100, int(val % 100))
        self.memory.inject(0, 4, score, 1.0, {"metric": "cloud_signal"})


class WeatherOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 60.0, url: str = ""):
        super().__init__(memory, interval_sec)
        self.url = url

    def start(self):
        if self.url and requests is not None:
            super().start()

    def _run_once(self):
        r = requests.get(self.url, timeout=5)
        temp = 20
        try:
            data = r.json()
            temp = int(data.get("main", {}).get("temp", 20))
        except Exception:
            pass
        self.memory.inject(1, 4, temp, 0.8, {"metric": "weather_temp"})


class StockMarketOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 60.0, url: str = ""):
        super().__init__(memory, interval_sec)
        self.url = url

    def start(self):
        if self.url and requests is not None:
            super().start()

    def _run_once(self):
        r = requests.get(self.url, timeout=5)
        score = 50
        try:
            data = r.json()
            score = int(data.get("score", 50))
        except Exception:
            pass
        self.memory.inject(2, 4, score, 1.0, {"metric": "market_signal"})


# =========================
# Network Firewall Organ (with CRDT)
# =========================

class NetworkFirewallOrgan(BaseOrganThread):
    def __init__(
        self,
        memory: BorgMemoryOrgan,
        interval_sec: float = 3.0,
        suspicious_ports=None,
        blacklist_ips=None,
    ):
        super().__init__(memory, interval_sec)
        self.suspicious_ports = set(suspicious_ports or {135, 445, 1433, 3389})
        self.blacklist_ips = set(blacklist_ips or set())

    def _run_once(self):
        conns = psutil.net_connections(kind="inet")
        net_threat, suspicious_count, blacklist_hits = self._compute_threat(conns)

        self.memory.inject(
            tile_id=0,
            index=1,
            value=int(net_threat * 100),
            priority=1.5 if net_threat > 0.5 else 1.0,
            meta={
                "metric": "net_threat",
                "origin_node": self.memory.organ_id,
                "origin_time": time.time(),
                "reason": f"{suspicious_count} suspicious ports, {blacklist_hits} blacklist hits",
            },
        )

        if net_threat > 0.7:
            self.memory.set_misroute_prob(
                min(0.9, self.memory.policy.misroute_prob * 1.05)
            )

        self.memory.debug_log.append(
            f"NetworkFirewallOrgan: net_threat={net_threat:.2f}, conns={len(conns)}"
        )

    def _compute_threat(self, conns):
        remote_count = 0
        suspicious_count = 0
        blacklist_hits = 0

        for c in conns:
            raddr = c.raddr
            if not raddr:
                continue
            remote_count += 1
            ip = raddr.ip
            port = raddr.port

            if port in self.suspicious_ports:
                suspicious_count += 1
            if ip in self.blacklist_ips:
                blacklist_hits += 1

        if remote_count == 0:
            return 0.0, 0, 0

        suspicious_ratio = suspicious_count / remote_count
        blacklist_ratio = blacklist_hits / max(1, remote_count)

        threat = 0.4 * suspicious_ratio + 0.6 * min(1.0, blacklist_ratio * 5.0)
        threat = max(0.0, min(threat, 1.0))
        return threat, suspicious_count, blacklist_hits

    def crdt_merge_blacklist(self, incoming_ips):
        before = len(self.blacklist_ips)
        self.blacklist_ips |= set(incoming_ips)
        after = len(self.blacklist_ips)
        self.memory.debug_log.append(
            f"Firewall CRDT merge: {before} → {after} entries"
        )

    def export_blacklist(self):
        return list(self.blacklist_ips)

    def receive_swarm_message(self, msg: dict):
        mtype = msg.get("type")

        if mtype == "blacklist_update":
            self.crdt_merge_blacklist(msg.get("ips", []))

        elif mtype == "suspicious_ports_update":
            new_ports = msg.get("ports", [])
            self.suspicious_ports.update(new_ports)
            self.memory.debug_log.append(f"Firewall: updated suspicious ports: {new_ports}")

        elif mtype == "threat_advisory":
            level = float(msg.get("level", 0))
            if level > 0.7:
                self.suspicious_ports.update({135, 445, 3389})
                self.memory.debug_log.append("Firewall: swarm high-threat advisory received")

        else:
            self.memory.debug_log.append(f"Firewall: unknown swarm msg {msg}")


# =========================
# Anomaly Detection Organ
# =========================

class AnomalyDetectionOrgan(BaseOrganThread):
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 5.0):
        super().__init__(memory, interval_sec)
        self.window: List[Tuple[float, float, float, float]] = []

    def _run_once(self):
        cpu = self.memory.last_cpu
        mem = self.memory.last_mem
        gpu = self.memory.last_gpu_util
        thr = self.memory.threat_level * 100.0

        self.window.append((cpu, mem, gpu, thr))
        if len(self.window) > 120:
            self.window.pop(0)

        if len(self.window) < 20:
            return

        means = [sum(x[i] for x in self.window) / len(self.window) for i in range(4)]
        stds = []
        for i in range(4):
            var = sum((x[i] - means[i]) ** 2 for x in self.window) / len(self.window)
            stds.append(max(1.0, math.sqrt(var)))

        z_cpu = abs(cpu - means[0]) / stds[0]
        z_mem = abs(mem - means[1]) / stds[1]
        z_gpu = abs(gpu - means[2]) / stds[2]
        z_thr = abs(thr - means[3]) / stds[3]

        combined = (z_cpu + z_mem + z_gpu + z_thr) / 4.0
        score = min(100, int(combined * 20))

        self.memory.inject(
            tile_id=3,
            index=2,
            value=score,
            priority=1.5 if score > 60 else 1.0,
            meta={
                "metric": "anomaly_score",
                "z_cpu": z_cpu,
                "z_mem": z_mem,
                "z_gpu": z_gpu,
                "z_thr": z_thr,
                "timestamp": time.time(),
            },
        )

        if score > 70:
            self.memory.threat_level = min(1.0, self.memory.threat_level + 0.1)
            self.memory.debug_log.append(f"AnomalyDetection: high anomaly score={score}")


# =========================
# Encrypted Swarm Networking Organ
# =========================

class SwarmCrypto:
    def __init__(self, key: bytes):
        self.key = key
        self.aes = AESGCM(key) if cryptography is not None else None
        self.chacha = ChaCha20Poly1305(key) if cryptography is not None else None

    def _hmac(self, data: bytes) -> bytes:
        return hmac.new(self.key, data, hashlib.sha256).digest()

    def encrypt(self, payload: dict) -> bytes:
        raw = json.dumps(payload).encode("utf-8")
        nonce = secrets.token_bytes(12)
        ts = int(time.time())
        body = json.dumps({"ts": ts, "data": base64.b64encode(raw).decode("ascii")}).encode("utf-8")

        if self.aes is not None:
            ct = self.aes.encrypt(nonce, body, None)
            mode = b"A"
        elif self.chacha is not None:
            ct = self.chacha.encrypt(nonce, body, None)
            mode = b"C"
        else:
            ct = body
            mode = b"N"

        mac = self._hmac(nonce + mode + ct)
        packet = nonce + mode + ct + mac
        return packet

    def decrypt(self, packet: bytes, max_age: int = 300) -> Optional[dict]:
        if len(packet) < 12 + 1 + 32:
            return None
        nonce = packet[:12]
        mode = packet[12:13]
        mac = packet[-32:]
        ct = packet[13:-32]

        if not hmac.compare_digest(mac, self._hmac(nonce + mode + ct)):
            return None

        if mode == b"A" and self.aes is not None:
            body = self.aes.decrypt(nonce, ct, None)
        elif mode == b"C" and self.chacha is not None:
            body = self.chacha.decrypt(nonce, ct, None)
        else:
            body = ct

        try:
            obj = json.loads(body.decode("utf-8"))
            ts = obj.get("ts", 0)
            if abs(time.time() - ts) > max_age:
                return None
            raw = base64.b64decode(obj.get("data", ""))
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None


class SwarmNetworkingOrgan(BaseOrganThread):
    def __init__(
        self,
        memory: BorgMemoryOrgan,
        interval_sec: float = 10.0,
        bind_port: int = 8888,
        peers: Optional[List[Tuple[str, int]]] = None,
        key: Optional[bytes] = None,
    ):
        super().__init__(memory, interval_sec)
        self.bind_port = bind_port
        self.peers = peers or [("127.0.0.1", bind_port)]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(("0.0.0.0", bind_port))
        except Exception as e:
            self.memory.debug_log.append(f"Swarm bind error: {e}")
        self.crypto = SwarmCrypto(key or self._default_key())
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _default_key(self) -> bytes:
        base = f"{self.memory.profile.user}@{self.memory.profile.hostname}".encode("utf-8")
        return hashlib.sha256(base).digest()

    def _recv_loop(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
                msg = self.crypto.decrypt(data)
                if msg is None:
                    continue
                self._handle_swarm_message(msg, addr)
            except Exception as e:
                self.memory.debug_log.append(f"Swarm recv error: {e}")

    def _handle_swarm_message(self, msg: dict, addr):
        mtype = msg.get("type")
        if mtype == "blacklist_update" and self.memory.firewall_organ:
            self.memory.firewall_organ.receive_swarm_message(msg)
        elif mtype == "threat_advisory" and self.memory.firewall_organ:
            self.memory.firewall_organ.receive_swarm_message(msg)
        elif mtype == "state_sync":
            pass
        self.memory.debug_log.append(f"Swarm message from {addr}: {mtype}")

    def _run_once(self):
        payloads = []

        if self.memory.firewall_organ:
            payloads.append({
                "type": "blacklist_update",
                "from": self.memory.organ_id,
                "ips": self.memory.firewall_organ.export_blacklist(),
                "time": time.time(),
            })

        payloads.append({
            "type": "threat_advisory",
            "from": self.memory.organ_id,
            "level": self.memory.threat_level,
            "anomaly_score": self.memory.anomaly_score,
            "time": time.time(),
        })

        for p in payloads:
            try:
                data = self.crypto.encrypt(p)
            except Exception as e:
                self.memory.debug_log.append(f"Swarm encrypt error: {e}")
                continue
            for host, port in self.peers:
                try:
                    self.sock.sendto(data, (host, port))
                except Exception as e:
                    self.memory.debug_log.append(f"Swarm send error: {e}")


# =========================
# AI Timing Manager for GUI
# =========================

class AITimingManager:
    def __init__(self):
        self.last_interaction = time.time()
        self.last_gui_duration = 0.0
        self.mode = "calm"  # calm or alert

    def record_interaction(self):
        self.last_interaction = time.time()

    def record_gui_duration(self, dt: float):
        self.last_gui_duration = dt

    def decide_mode(self, organ: BorgMemoryOrgan) -> str:
        now = time.time()
        idle = now - self.last_interaction
        threat = organ.threat_level
        anomaly = organ.anomaly_score / 100.0

        if threat > 0.6 or anomaly > 0.6 or organ.missing_details_hint:
            self.mode = "alert"
        elif idle > 30 and threat < 0.3 and anomaly < 0.3:
            self.mode = "calm"
        return self.mode

    def gui_delay_ms(self, organ: BorgMemoryOrgan, base_ms: int = 60) -> int:
        """
        Slightly faster base_ms to reduce click-to-update delay.
        """
        mode = self.decide_mode(organ)
        cpu = organ.last_cpu
        anomaly = organ.anomaly_score
        dt = self.last_gui_duration

        delay = base_ms

        if mode == "calm":
            delay *= 1.4
        else:
            delay *= 0.8

        if cpu > 80 or anomaly > 80 or dt > 0.05:
            delay *= 1.4

        delay = int(max(40, min(delay, 250)))
        return delay


# =========================
# UIAutomation Organ (COM-based)
# =========================

class UIAutomationOrgan(BaseOrganThread):
    """
    Uses comtypes + UIAutomationCore.dll to read:
    - active window
    - focused control
    - basic patterns (name, value)
    Produces high-level UI context for predictive/missing-details organs.
    """
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 1.0):
        super().__init__(memory, interval_sec)
        self.enabled = platform.system() == "Windows" and comtypes is not None
        self.uia = None
        self.root_element = None
        self.last_active_name = ""
        self.last_focus_name = ""
        self.last_focus_value = ""
        self.last_scan_missing = 0

        if self.enabled:
            try:
                from comtypes.client import CreateObject
                self.uia = CreateObject("UIAutomationClient.CUIAutomation8")
                self.root_element = self.uia.GetRootElement()
            except Exception as e:
                self.memory.debug_log.append(f"UIAutomation init error: {e}")
                self.enabled = False

    def start(self):
        if self.enabled:
            super().start()

    def _get_active_window_element(self):
        try:
            focused = self.uia.GetFocusedElement()
            hwnd = focused.CurrentNativeWindowHandle
            if hwnd:
                return self.uia.ElementFromHandle(hwnd)
        except Exception:
            pass
        return None

    def _run_once(self):
        if not self.enabled or self.uia is None:
            return

        try:
            focused = self.uia.GetFocusedElement()
        except Exception:
            focused = None

        active_name = ""
        focus_name = ""
        focus_value = ""

        try:
            if focused:
                focus_name = focused.CurrentName or ""
                try:
                    vp = focused.GetCurrentPattern(10002)  # ValuePattern
                    focus_value = vp.CurrentValue or ""
                except Exception:
                    focus_value = ""
        except Exception:
            pass

        try:
            active = self._get_active_window_element()
            if active:
                active_name = active.CurrentName or ""
        except Exception:
            pass

        self.last_active_name = active_name
        self.last_focus_name = focus_name
        self.last_focus_value = focus_value

        ctx = {
            "metric": "ui_context",
            "active": active_name,
            "focus": focus_name,
            "value_len": len(focus_value or ""),
            "timestamp": time.time(),
        }
        self.memory.inject(0, 5, min(100, len(focus_value or "")), 1.0, ctx)

    def scan_for_missing_fields(self, max_depth: int = 3) -> int:
        if not self.enabled or self.uia is None:
            return 0
        try:
            focused = self.uia.GetFocusedElement()
        except Exception:
            return 0
        if not focused:
            return 0

        missing = 0
        try:
            tree_scope = 4  # Descendants
            cond = self.uia.CreateTrueCondition()
            walker = focused.FindAll(tree_scope, cond)
            count = walker.Length
            for i in range(min(count, 200)):
                el = walker.GetElement(i)
                try:
                    name = el.CurrentName or ""
                    ctrl_type = el.CurrentControlType
                    if ctrl_type in (50004, 50025):  # edit, combo
                        val = ""
                        try:
                            vp = el.GetCurrentPattern(10002)
                            val = vp.CurrentValue or ""
                        except Exception:
                            pass
                        if not val and name:
                            missing += 1
                except Exception:
                    continue
        except Exception:
            pass

        self.last_scan_missing = missing
        return missing


# =========================
# Predictive Organ
# =========================

class PredictiveOrgan(BaseOrganThread):
    """
    Uses UIAutomationOrgan + system signals to infer user state.
    Injects 'prediction' packets into memory.
    """
    def __init__(self, memory: BorgMemoryOrgan, ui_org: UIAutomationOrgan, interval_sec: float = 2.0):
        super().__init__(memory, interval_sec)
        self.ui_org = ui_org
        self.last_prediction = ""

    def _run_once(self):
        cpu = self.memory.last_cpu
        mem = self.memory.last_mem
        gpu = self.memory.last_gpu_util
        threat = self.memory.threat_level
        anomaly = self.memory.anomaly_score / 100.0

        active = self.ui_org.last_active_name or ""
        focus = self.ui_org.last_focus_name or ""
        val_len = len(self.ui_org.last_focus_value or "")

        pred = "idle"

        if "chrome" in active.lower() or "edge" in active.lower() or "firefox" in active.lower():
            if cpu < 30 and val_len == 0:
                pred = "reading"
            elif val_len > 0:
                pred = "typing_in_browser"
        elif "explorer" in active.lower():
            pred = "file_navigation"
        elif "settings" in active.lower() or "control panel" in active.lower():
            pred = "configuring_system"
        elif "visual studio" in active.lower() or "code" in active.lower():
            pred = "coding"
        elif gpu > 60:
            pred = "gaming_or_rendering"

        if threat > 0.7 or anomaly > 0.7:
            pred = "under_stress"

        self.last_prediction = pred

        self.memory.inject(
            tile_id=1,
            index=5,
            value=min(100, int(cpu + mem + gpu) // 3),
            priority=1.2,
            meta={
                "metric": "prediction",
                "prediction": pred,
                "active": active,
                "focus": focus,
                "timestamp": time.time(),
            },
        )


# =========================
# Missing-Details Organ
# =========================

class MissingDetailsOrgan(BaseOrganThread):
    """
    Uses UIAutomationOrgan to look for empty required-like fields.
    Injects 'missing_details' packets and nudges anomaly/threat.
    """
    def __init__(self, memory: BorgMemoryOrgan, ui_org: UIAutomationOrgan, interval_sec: float = 4.0):
        super().__init__(memory, interval_sec)
        self.ui_org = ui_org

    def _run_once(self):
        missing = self.ui_org.scan_for_missing_fields()
        if missing <= 0:
            return

        hint = f"{missing} possible missing fields in current UI"
        self.memory.inject(
            tile_id=2,
            index=5,
            value=min(100, missing * 10),
            priority=1.5,
            meta={
                "metric": "missing_details",
                "missing_count": missing,
                "hint": hint,
                "timestamp": time.time(),
            },
        )

        self.memory.anomaly_score = min(100, self.memory.anomaly_score + missing * 2)
        self.memory.threat_level = min(1.0, self.memory.threat_level + missing * 0.01)
        self.memory.debug_log.append(f"MissingDetailsOrgan: {hint}")


# =========================
# Recall Organ (contextual recall)
# =========================

class RecallOrgan(BaseOrganThread):
    """
    Contextual recall:
    - Looks at current state (cpu/mem/gpu/threat/prediction)
    - Finds similar episodes in long_term_memory
    - Injects 'recall' packets to re-influence the cortex
    """
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 7.0):
        super().__init__(memory, interval_sec)

    def _run_once(self):
        if not self.memory.long_term_memory:
            return

        ctx_now = {
            "cpu": self.memory.last_cpu,
            "mem": self.memory.last_mem,
            "gpu": self.memory.last_gpu_util,
            "threat": self.memory.threat_level,
            "prediction": self.memory.prediction,
        }

        best = None
        best_score = 0.0

        for pkt in self.memory.long_term_memory[-1000:]:
            ctx = pkt.meta.get("context") or {}
            score = 0.0
            try:
                score += 1.0 - min(1.0, abs(ctx_now["cpu"] - ctx.get("cpu", 0)) / 100.0)
                score += 1.0 - min(1.0, abs(ctx_now["mem"] - ctx.get("mem", 0)) / 100.0)
                score += 1.0 - min(1.0, abs(ctx_now["gpu"] - ctx.get("gpu", 0)) / 100.0)
                score += 1.0 - min(1.0, abs(ctx_now["threat"] - ctx.get("threat", 0.0)))
                if ctx_now["prediction"] == ctx.get("prediction"):
                    score += 1.0
            except Exception:
                continue

            if score > best_score:
                best_score = score
                best = pkt

        if best is None or best_score < 2.5:
            return

        self.memory.inject(
            tile_id=3,
            index=1,
            value=best.value,
            priority=min(2.0, best.priority * 1.2),
            meta={
                "metric": "recall",
                "source_pid": best.pid,
                "score": best_score,
                "original_context": best.meta.get("context", {}),
                "timestamp": time.time(),
            },
        )


# =========================
# User Activity Governor Organ
# =========================

class UserActivityGovernorOrgan(BaseOrganThread):
    """
    Reads prediction + threat/anomaly and sets memory.user_mode:
    - idle
    - browsing
    - typing
    - coding
    - gaming
    - configuring
    - under_stress

    This drives context-aware priority in BorgMemoryOrgan.priority_boost().
    """
    def __init__(self, memory: BorgMemoryOrgan, interval_sec: float = 2.0):
        super().__init__(memory, interval_sec)

    def _run_once(self):
        pred = (self.memory.prediction or "").lower()
        threat = self.memory.threat_level
        anomaly = self.memory.anomaly_score / 100.0

        mode = "idle"

        if "gaming" in pred or "rendering" in pred:
            mode = "gaming"
        elif "typing" in pred:
            mode = "typing"
        elif "reading" in pred:
            mode = "browsing"
        elif "coding" in pred:
            mode = "coding"
        elif "configuring" in pred or "settings" in pred:
            mode = "configuring"
        elif "under_stress" in pred:
            mode = "under_stress"

        if threat > 0.7 or anomaly > 0.7:
            mode = "under_stress"

        self.memory.update_user_mode(mode)
        self.memory.debug_log.append(f"UserActivityGovernor: user_mode={mode}, pred={pred}, thr={threat:.2f}, anom={anomaly:.2f}")


# =========================
# GUI: Adaptive AI HUD with Recall Timeline
# =========================

class AdaptiveAIHUD:
    """
    Minimal by default (Calm Mode), expands when threat/anomaly high or user interacts (Alert Mode).
    - Calm: vital signs + single compact graph + status line
    - Alert: adds threat/anomaly details + lineage + prediction/missing-details hints + recall timeline
    """
    def __init__(self, organ: BorgMemoryOrgan, timing: AITimingManager, base_tick_ms=60):
        if tkinter is None:
            raise RuntimeError("Tkinter not available; GUI cannot be created.")
        self.organ = organ
        self.timing = timing
        self.base_tick_ms = base_tick_ms
        self.last_gui_update_duration = 0.0

        self.root = tkinter.Tk()
        self.root.title(f"Adaptive Cortex HUD - {organ.organ_id}")
        self.root.configure(bg="black")

        self.root.bind("<Button-1>", self._on_click)
        self.root.bind("<Motion>", self._on_motion)

        def mk_label(text, fg, font=("Consolas", 10)):
            lbl = tkinter.Label(self.root, text=text, fg=fg, bg="black", anchor="w", justify="left", font=font)
            lbl.pack(fill="x")
            return lbl

        self.lbl_vitals = mk_label("Vitals:", "white", ("Consolas", 10, "bold"))
        self.lbl_cpu = mk_label("CPU: 0%", "white")
        self.lbl_mem = mk_label("MEM: 0%", "white")
        self.lbl_gpu = mk_label("GPU: 0%", "cyan")
        self.lbl_threat = mk_label("Threat: 0.00", "red")
        self.lbl_anomaly = mk_label("Anomaly: 0", "yellow")
        self.lbl_mode = mk_label("Mode: balanced", "green")
        self.lbl_user_mode = mk_label("User Mode: idle", "magenta")

        self.lbl_graph = mk_label("Graph:", "white")

        self.lbl_status = mk_label("Status: Initializing...", "magenta")

        self.lbl_alert_header = mk_label("Alert Details:", "red", ("Consolas", 10, "bold"))
        self.lbl_alert_header.pack_forget()
        self.lbl_lineage = mk_label("", "white")
        self.lbl_lineage.pack_forget()
        self.lbl_prediction = mk_label("", "cyan")
        self.lbl_prediction.pack_forget()
        self.lbl_missing = mk_label("", "yellow")
        self.lbl_missing.pack_forget()

        self.lbl_recall_header = mk_label("Recall Timeline:", "orange", ("Consolas", 10, "bold"))
        self.lbl_recall_header.pack_forget()
        self.lbl_recall = mk_label("", "orange")
        self.lbl_recall.pack_forget()

        btn_frame = tkinter.Frame(self.root, bg="black")
        btn_frame.pack(fill="x")
        self.btn_save = tkinter.Button(btn_frame, text="Save State", command=self._on_save)
        self.btn_save.pack(side="left")
        self.btn_load = tkinter.Button(btn_frame, text="Load State", command=self._on_load)
        self.btn_load.pack(side="left")

        self._last_values = {}

        self._schedule_next()

    def _on_click(self, event):
        self.timing.record_interaction()

    def _on_motion(self, event):
        self.timing.record_interaction()

    def _on_save(self):
        self.timing.record_interaction()
        self.organ.save_state()

    def _on_load(self):
        self.timing.record_interaction()
        self.organ.load_state()

    def _schedule_next(self):
        delay = self.timing.gui_delay_ms(self.organ, self.base_tick_ms)
        self.root.after(delay, self._update)

    def _ascii_graph(self, values: List[float], width: int = 40, max_val: float = 100.0) -> str:
        if not values:
            return "-" * width
        vals = values[-width:]
        if not vals:
            return "-" * width
        chars = []
        for v in vals:
            ratio = max(0.0, min(1.0, v / max_val))
            if ratio < 0.2:
                ch = "."
            elif ratio < 0.4:
                ch = "-"
            elif ratio < 0.6:
                ch = "+"
            elif ratio < 0.8:
                ch = "*"
            else:
                ch = "#"
            chars.append(ch)
        if len(chars) < width:
            chars = [" "] * (width - len(chars)) + chars
        return "".join(chars)

    def _set_label(self, label, key: str, text: str):
        if self._last_values.get(key) == text:
            return
        self._last_values[key] = text
        label.config(text=text)

    def _status_text(self, organ: BorgMemoryOrgan, mode: str) -> str:
        t = organ.threat_level
        a = organ.anomaly_score / 100.0
        cpu = organ.last_cpu
        pred = organ.prediction or "unknown"

        if t < 0.3 and a < 0.3 and cpu < 60 and not organ.missing_details_hint:
            return f"Status: System Stable – {pred}"
        if organ.missing_details_hint:
            return f"Status: Attention – {organ.missing_details_hint}"
        if t > 0.7 or a > 0.7:
            return f"Status: ALERT – Threat/Anomaly High – {pred}"
        if cpu > 80:
            return f"Status: High Load – {pred}"
        if mode == "calm":
            return f"Status: Calm Monitoring – {pred}"
        return f"Status: Active Monitoring – {pred}"

    def _format_recall_timeline(self, events: List[dict], max_items: int = 5) -> str:
        if not events:
            return "(no recall events yet)"
        lines = []
        for e in events[-max_items:]:
            ts = e.get("timestamp", time.time())
            score = e.get("score", 0.0)
            ctx = e.get("context", {})
            cpu = ctx.get("cpu", 0)
            mem = ctx.get("mem", 0)
            thr = ctx.get("threat", 0.0)
            pred = ctx.get("prediction", "")
            t_str = time.strftime("%H:%M:%S", time.localtime(ts))
            line = f"[{t_str}] score={score:.1f} cpu={cpu}% mem={mem}% thr={thr:.2f} pred={pred}"
            lines.append(line)
        return "\n".join(lines)

    def _update(self):
        t0 = time.time()
        o = self.organ
        mode = self.timing.decide_mode(o)

        self._set_label(self.lbl_cpu, "cpu", f"CPU: {o.last_cpu}%")
        self._set_label(self.lbl_mem, "mem", f"MEM: {o.last_mem}%")
        self._set_label(self.lbl_gpu, "gpu", f"GPU: {o.last_gpu_util}%")
        self._set_label(self.lbl_threat, "threat", f"Threat: {o.threat_level:.2f}")
        self._set_label(self.lbl_anomaly, "anomaly", f"Anomaly: {o.anomaly_score}")
        self._set_label(self.lbl_mode, "mode", f"Mode: {o.mode_selector.mode}")
        self._set_label(self.lbl_user_mode, "user_mode", f"User Mode: {o.user_mode}")

        combined = []
        for i in range(max(len(o.history_cpu), len(o.history_mem), len(o.history_threat))):
            c = o.history_cpu[i] if i < len(o.history_cpu) else 0
            m = o.history_mem[i] if i < len(o.history_mem) else 0
            th = o.history_threat[i] if i < len(o.history_threat) else 0
            combined.append((c + m + th) / 3.0)
        graph_text = "Graph: " + self._ascii_graph(combined)
        self._set_label(self.lbl_graph, "graph", graph_text)

        status = self._status_text(o, mode)
        self._set_label(self.lbl_status, "status", status)

        if mode == "alert":
            if not self.lbl_alert_header.winfo_ismapped():
                self.lbl_alert_header.pack(before=self.lbl_status)
                self.lbl_lineage.pack(before=self.lbl_status)
                self.lbl_prediction.pack(before=self.lbl_status)
                self.lbl_missing.pack(before=self.lbl_status)
                self.lbl_recall_header.pack(before=self.lbl_status)
                self.lbl_recall.pack(before=self.lbl_status)

            lines = []
            for entry in o.threat_lineage[-3:]:
                lines.append(f"{entry.get('origin_node')} → {entry.get('reason')}")
            lineage_text = "Lineage:\n" + ("\n".join(lines) if lines else "(none)")
            self._set_label(self.lbl_lineage, "lineage", lineage_text)
            self._set_label(self.lbl_prediction, "prediction", f"Prediction: {o.prediction or '(none)'}")
            self._set_label(self.lbl_missing, "missing", f"Missing: {o.missing_details_hint or '(none)'}")

            recall_text = self._format_recall_timeline(o.recall_events)
            self._set_label(self.lbl_recall, "recall_timeline", recall_text)
        else:
            if self.lbl_alert_header.winfo_ismapped():
                self.lbl_alert_header.pack_forget()
                self.lbl_lineage.pack_forget()
                self.lbl_prediction.pack_forget()
                self.lbl_missing.pack_forget()
                self.lbl_recall_header.pack_forget()
                self.lbl_recall.pack_forget()

        self.last_gui_update_duration = time.time() - t0
        self.timing.record_gui_duration(self.last_gui_update_duration)

        self._schedule_next()

    def run(self):
        self.root.mainloop()


# =========================
# TCP Daemon
# =========================

def start_tcp_daemon(organ: BorgMemoryOrgan, host="127.0.0.1", port=7777):
    def handle_client(conn, addr):
        with conn:
            buf = b""
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                        resp = organ.handle_ipc_message(msg)
                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                    except Exception as e:
                        err = {"ok": False, "error": str(e)}
                        conn.sendall((json.dumps(err) + "\n").encode("utf-8"))

    def server_loop():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(5)
        organ.debug_log.append(f"TCP daemon listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

    threading.Thread(target=server_loop, daemon=True).start()


# =========================
# Main
# =========================

def main():
    organ = BorgMemoryOrgan(
        organ_id="memory_backbone_v11_3",
        personality=PersonalityProfile("analytical"),
        altered_state=AlteredStateManager("focus"),
        mode_selector=AutoModeSelector(),
    )

    organ.load_state()

    sys_org = SystemTelemetryOrgan(organ, 2.0)
    disk_org = DiskTelemetryOrgan(organ, 3.0)
    proc_org = ProcessBehaviorOrgan(organ, 5.0)
    evt_org = WindowsEventLogOrgan(organ, 10.0)
    sniff_org = PacketSnifferOrgan(organ, 3.0)
    gpu_org = GPUTelemetryOrgan(organ, 2.0)
    browser_org = BrowserTelemetryOrgan(organ, 5.0)
    game_org = GameTelemetryOrgan(organ, 5.0)
    hw_org = HardwareSensorsOrgan(organ, 10.0)
    cloud_org = CloudAPIOrgan(organ, 30.0, url="")
    weather_org = WeatherOrgan(organ, 60.0, url="")
    market_org = StockMarketOrgan(organ, 60.0, url="")
    fw_org = NetworkFirewallOrgan(
        organ,
        interval_sec=3.0,
        suspicious_ports={135, 445, 1433, 3389},
        blacklist_ips=set(),
    )
    anomaly_org = AnomalyDetectionOrgan(organ, 5.0)
    swarm_org = SwarmNetworkingOrgan(
        organ,
        interval_sec=10.0,
        bind_port=8888,
        peers=[("127.0.0.1", 8888)],
    )

    ui_org = UIAutomationOrgan(organ, 1.0)
    predictive_org = PredictiveOrgan(organ, ui_org, 2.0)
    missing_org = MissingDetailsOrgan(organ, ui_org, 4.0)
    recall_org = RecallOrgan(organ, 7.0)
    governor_org = UserActivityGovernorOrgan(organ, 2.0)

    organ.attach_firewall(fw_org)

    sys_org.start()
    disk_org.start()
    proc_org.start()
    evt_org.start()
    sniff_org.start()
    gpu_org.start()
    browser_org.start()
    game_org.start()
    hw_org.start()
    cloud_org.start()
    weather_org.start()
    market_org.start()
    fw_org.start()
    anomaly_org.start()
    swarm_org.start()
    ui_org.start()
    predictive_org.start()
    missing_org.start()
    recall_org.start()
    governor_org.start()

    start_tcp_daemon(organ)

    def tick_loop():
        last_save = time.time()
        while True:
            try:
                organ.tick()
            except Exception as e:
                organ.debug_log.append(f"Tick error: {e}")
            base = 0.25
            interval = organ.mode_selector.scale_interval(base)
            time.sleep(interval)

            if time.time() - last_save > 60:
                organ.save_state()
                last_save = time.time()

    threading.Thread(target=tick_loop, daemon=True).start()

    if tkinter is not None:
        timing = AITimingManager()
        hud = AdaptiveAIHUD(organ, timing, base_tick_ms=60)
        hud.run()
    else:
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()