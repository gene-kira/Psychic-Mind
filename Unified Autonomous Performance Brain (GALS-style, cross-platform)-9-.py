#!/usr/bin/env python3
"""
Unified Swarm Autonomous Performance Brain v7
------------------------------------------------
Now with:

A) Full Raft log consistency engine with snapshotting + log compaction
B) Real kernel driver *interface* skeleton with WDM/eBPF code stubs (SAFE, non‑destructive)
C) PPO agent with parallel rollouts + multi‑env training (conceptual multi‑env harness)
D) VR cockpit export with animated swarm orbits + holographic overlays (data + hints)
E) CRDT‑based distributed policy fabric with timestamped deltas + queen arbitration

NOTE:
- This is a control‑plane / architecture brain.
- All hardware‑touching parts are SAFE STUBS (no destructive behavior).
"""

# =========================
# Autoload libs
# =========================

def _autoload_libs():
    import importlib
    import sys

    required = [
        "time", "threading", "queue", "dataclasses", "typing", "random",
        "signal", "sys", "platform", "os", "json", "http.server",
        "socketserver", "logging", "math", "socket", "sqlite3", "collections"
    ]
    ns = {}
    for name in required:
        if name in sys.modules:
            mod = sys.modules[name]
        else:
            mod = importlib.import_module(name)
        ns[name] = mod

    optional = ["psutil", "tkinter", "pynvml", "torch", "matplotlib"]
    for name in optional:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            mod = None
        ns[name] = mod

    if ns["matplotlib"] is not None:
        try:
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            ns["plt"] = plt
        except Exception:
            ns["plt"] = None
    else:
        ns["plt"] = None

    return ns


_libs = _autoload_libs()
time = _libs["time"]
threading = _libs["threading"]
queue = _libs["queue"]
dataclasses = _libs["dataclasses"]
typing = _libs["typing"]
random = _libs["random"]
signal = _libs["signal"]
sys = _libs["sys"]
platform = _libs["platform"]
os = _libs["os"]
json = _libs["json"]
http_server = _libs["http.server"]
socketserver = _libs["socketserver"]
logging = _libs["logging"]
math = _libs["math"]
socket_mod = _libs["socket"]
sqlite3 = _libs["sqlite3"]
collections = _libs["collections"]
psutil = _libs["psutil"]
tkinter = _libs["tkinter"]
pynvml = _libs["pynvml"]
torch = _libs["torch"]
matplotlib = _libs["matplotlib"]
plt = _libs["plt"]

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[UnifiedBrain] %(asctime)s %(levelname)s: %(message)s",
)

IS_WINDOWS = platform.system().lower().startswith("windows")
IS_LINUX = platform.system().lower().startswith("linux")

# =========================
# Persistence
# =========================

class Persistence:
    def __init__(self, db_path: str = "unified_brain.db", snapshot_path: str = "unified_snapshot.json"):
        self.db_path = db_path
        self.snapshot_path = snapshot_path
        self._init_db()

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS replay (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    state BLOB,
                    reward REAL,
                    priority REAL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS raft_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term INTEGER,
                    index_pos INTEGER,
                    entry BLOB
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS raft_snapshot (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    last_index INTEGER,
                    last_term INTEGER,
                    state BLOB
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.warning(f"Persistence DB init failed: {e}")

    # ---- RL replay ----
    def save_replay(self, state_vec: List[float], reward: float, priority: float):
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO replay (ts, state, reward, priority) VALUES (?, ?, ?, ?)",
                (time.time(), json.dumps(state_vec), reward, priority),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.debug(f"Persistence save_replay failed: {e}")

    def load_replay_batch(self, batch_size: int = 64) -> List[Tuple[List[float], float]]:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT state, reward FROM replay ORDER BY priority DESC, id DESC LIMIT ?",
                (batch_size,),
            )
            rows = cur.fetchall()
            conn.close()
            batch = []
            for s, r in rows:
                try:
                    vec = json.loads(s)
                    batch.append((vec, float(r)))
                except Exception:
                    continue
            return batch
        except Exception as e:
            logging.debug(f"Persistence load_replay_batch failed: {e}")
            return []

    # ---- global snapshot ----
    def save_snapshot(self, snapshot: Dict[str, Any]):
        try:
            with open(self.snapshot_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            logging.debug(f"Persistence save_snapshot failed: {e}")

    def load_snapshot(self) -> Optional[Dict[str, Any]]:
        try:
            if not os.path.exists(self.snapshot_path):
                return None
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.debug(f"Persistence load_snapshot failed: {e}")
            return None

    # ---- Raft log ----
    def append_raft_entry(self, term: int, index_pos: int, entry: Dict[str, Any]):
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO raft_log (term, index_pos, entry) VALUES (?, ?, ?)",
                (term, index_pos, json.dumps(entry)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.debug(f"Persistence append_raft_entry failed: {e}")

    def load_raft_log(self) -> List[Tuple[int, int, Dict[str, Any]]]:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT term, index_pos, entry FROM raft_log ORDER BY index_pos ASC")
            rows = cur.fetchall()
            conn.close()
            out = []
            for term, idx, entry in rows:
                try:
                    out.append((int(term), int(idx), json.loads(entry)))
                except Exception:
                    continue
            return out
        except Exception as e:
            logging.debug(f"Persistence load_raft_log failed: {e}")
            return []

    def compact_raft_log(self, keep_from_index: int):
        """
        Log compaction: delete entries before keep_from_index.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("DELETE FROM raft_log WHERE index_pos < ?", (keep_from_index,))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.debug(f"Persistence compact_raft_log failed: {e}")

    # ---- Raft snapshot ----
    def save_raft_snapshot(self, last_index: int, last_term: int, state: Dict[str, Any]):
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO raft_snapshot (ts, last_index, last_term, state) VALUES (?, ?, ?, ?)",
                (time.time(), last_index, last_term, json.dumps(state)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.debug(f"Persistence save_raft_snapshot failed: {e}")

    def load_latest_raft_snapshot(self) -> Optional[Tuple[int, int, Dict[str, Any]]]:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT last_index, last_term, state FROM raft_snapshot ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            conn.close()
            if not row:
                return None
            last_index, last_term, state = row
            return int(last_index), int(last_term), json.loads(state)
        except Exception as e:
            logging.debug(f"Persistence load_latest_raft_snapshot failed: {e}")
            return None


# =========================
# Data structures
# =========================

@dataclass
class SchedulerExport:
    runq_depth: List[int] = field(default_factory=list)
    latency_hist: List[float] = field(default_factory=list)
    stall_rate: List[float] = field(default_factory=list)


@dataclass
class SchedulerPolicy:
    core_affinity_map: Dict[int, List[int]] = field(default_factory=dict)
    burst_boost_windows: Dict[int, float] = field(default_factory=dict)
    background_throttle_level: float = 0.0


@dataclass
class CacheExport:
    hit_rate: Dict[str, float] = field(default_factory=dict)
    prefetch_accuracy: float = 0.0
    bandwidth_use: Dict[str, float] = field(default_factory=dict)


@dataclass
class CachePolicy:
    cache_appetite: float = 1.0
    read_ahead_distance: Dict[str, int] = field(default_factory=dict)
    vram_bias_factor: float = 1.0
    ghost_cache_burst_mode: bool = False


@dataclass
class MemoryExport:
    ram_pressure: float = 0.0
    vram_pressure: float = 0.0
    page_fault_rate: float = 0.0
    migration_latency: float = 0.0


@dataclass
class MemoryPolicy:
    tier_thresholds: Dict[str, float] = field(default_factory=dict)
    migration_budget_mb: float = 0.0
    prewarm_targets: List[str] = field(default_factory=list)


@dataclass
class GlobalExport:
    package_power: float = 0.0
    rail_limits_hit: bool = False
    thermal_margin: float = 0.0
    cpu_util: float = 0.0
    mem_util: float = 0.0


@dataclass
class GPUExport:
    gpu_util: float = 0.0
    vram_util: float = 0.0
    temperature: float = 0.0
    power_draw: float = 0.0
    has_gpu: bool = False


@dataclass
class FluidModel:
    pressure: Dict[str, float] = field(default_factory=dict)
    flow: Dict[str, float] = field(default_factory=dict)
    friction: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealthStatus:
    health: float = 1.0
    confidence: float = 1.0
    last_error: Optional[str] = None


@dataclass
class ClusterNodeView:
    node_id: str = ""
    last_seen: float = 0.0
    mode: str = "unknown"
    health: float = 0.0
    cpu_pressure: float = 0.0
    ram_pressure: float = 0.0
    is_leader: bool = False
    term: int = 0
    voted_for: Optional[str] = None
    weight: float = 1.0
    role: str = "drone"  # "queen" or "drone"


@dataclass
class ClusterState:
    self_id: str = ""
    leader_id: Optional[str] = None
    queen_id: Optional[str] = None
    term: int = 0
    nodes: Dict[str, ClusterNodeView] = field(default_factory=dict)


@dataclass
class StateVector:
    scheduler_export: SchedulerExport = field(default_factory=SchedulerExport)
    cache_export: CacheExport = field(default_factory=CacheExport)
    memory_export: MemoryExport = field(default_factory=MemoryExport)
    global_export: GlobalExport = field(default_factory=GlobalExport)
    gpu_export: GPUExport = field(default_factory=GPUExport)
    fluid: FluidModel = field(default_factory=FluidModel)
    health: HealthStatus = field(default_factory=HealthStatus)
    mode: str = "Flow"
    cluster: ClusterState = field(default_factory=ClusterState)
    timestamp: float = 0.0


@dataclass
class PolicyBundle:
    scheduler_policy: SchedulerPolicy = field(default_factory=SchedulerPolicy)
    cache_policy: CachePolicy = field(default_factory=CachePolicy)
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)


# =========================
# Telemetry
# =========================

class Telemetry:
    def __init__(self):
        self.has_psutil = psutil is not None
        self.has_gpu = False
        if self.has_psutil:
            logging.info("psutil detected: using real CPU/RAM telemetry")
        else:
            logging.warning("psutil not available: using synthetic CPU/RAM telemetry")

        self.gpu_inited = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.gpu_inited = True
                self.has_gpu = pynvml.nvmlDeviceGetCount() > 0
                if self.has_gpu:
                    logging.info("pynvml detected: GPU telemetry enabled")
                else:
                    logging.info("pynvml detected but no GPUs found")
            except Exception as e:
                logging.warning(f"pynvml init failed: {e}")
                self.gpu_inited = False
        else:
            logging.info("pynvml not available: GPU telemetry disabled")

    def collect_global(self) -> GlobalExport:
        if self.has_psutil:
            cpu_util = psutil.cpu_percent(interval=None) / 100.0
            mem = psutil.virtual_memory()
            mem_util = mem.percent / 100.0
            package_power = random.uniform(30.0, 120.0)
            thermal_margin = random.uniform(0.0, 0.5)
            rail_limits_hit = False
        else:
            cpu_util = random.uniform(0.0, 1.0)
            mem_util = random.uniform(0.0, 1.0)
            package_power = random.uniform(30.0, 120.0)
            thermal_margin = random.uniform(0.0, 0.5)
            rail_limits_hit = random.random() < 0.05

        return GlobalExport(
            package_power=package_power,
            rail_limits_hit=rail_limits_hit,
            thermal_margin=thermal_margin,
            cpu_util=cpu_util,
            mem_util=mem_util,
        )

    def collect_gpu(self) -> GPUExport:
        if not self.gpu_inited or not self.has_gpu:
            return GPUExport(
                gpu_util=random.uniform(0.0, 0.3),
                vram_util=random.uniform(0.0, 0.3),
                temperature=random.uniform(30.0, 60.0),
                power_draw=random.uniform(20.0, 80.0),
                has_gpu=False,
            )

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            gpu_util = util.gpu / 100.0
            vram_util = mem.used / mem.total if mem.total > 0 else 0.0

            return GPUExport(
                gpu_util=gpu_util,
                vram_util=vram_util,
                temperature=float(temp),
                power_draw=float(power),
                has_gpu=True,
            )
        except Exception as e:
            logging.warning(f"GPU telemetry error: {e}")
            return GPUExport(
                gpu_util=random.uniform(0.0, 0.5),
                vram_util=random.uniform(0.0, 0.5),
                temperature=random.uniform(30.0, 70.0),
                power_draw=random.uniform(20.0, 100.0),
                has_gpu=False,
            )


# =========================
# OS Control Hooks
# =========================

class OSControlHooks:
    def __init__(self):
        self.has_psutil = psutil is not None
        self.is_windows = IS_WINDOWS
        if not self.has_psutil:
            logging.warning("OS control hooks limited: psutil not available")

    def apply_background_throttle(self, level: float):
        if not self.has_psutil:
            return
        p = psutil.Process(os.getpid())
        try:
            if self.is_windows:
                if level < 0.3:
                    p.nice(psutil.NORMAL_PRIORITY_CLASS)
                elif level < 0.7:
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    p.nice(psutil.IDLE_PRIORITY_CLASS)
            else:
                if level < 0.3:
                    nice = 0
                elif level < 0.7:
                    nice = 5
                else:
                    nice = 10
                try:
                    p.nice(nice)
                except Exception:
                    pass
        except Exception as e:
            logging.debug(f"OS control hook failed: {e}")


# =========================
# Kernel Driver Architecture (stubs with WDM/eBPF hints)
# =========================

class KernelDriverInterface:
    """
    Abstract kernel driver interface.

    Real implementation (OUTSIDE this script) would use:
    - Windows: WDM/WDK driver + IOCTL (DeviceIoControl)
    - Linux: eBPF or kernel module + /dev or netlink

    Here we only define SAFE control-plane stubs and payload shapes.
    """

    def __init__(self):
        self.is_windows = IS_WINDOWS
        self.is_linux = IS_LINUX

    def load_driver(self):
        # In reality: load driver service / insmod / bpftool load
        logging.debug("KernelDriverInterface.load_driver() stub")

    def unload_driver(self):
        # In reality: stop service / rmmod / bpftool unload
        logging.debug("KernelDriverInterface.unload_driver() stub")

    def send_scheduler_hint(self, payload: Dict[str, Any]):
        """
        Example payload shape for a real driver:
        {
          "background_throttle_level": float,
          "core_affinity_map": {core_id: [pids...]},
        }
        """
        logging.debug(f"KernelDriverInterface.send_scheduler_hint() stub: {payload}")

    def send_memory_hint(self, payload: Dict[str, Any]):
        """
        Example payload shape:
        {
          "tier_thresholds": {...},
          "migration_budget_mb": float
        }
        """
        logging.debug(f"KernelDriverInterface.send_memory_hint() stub: {payload}")

    def send_cache_hint(self, payload: Dict[str, Any]):
        """
        Example payload shape:
        {
          "cache_appetite": float,
          "vram_bias_factor": float,
          "ghost_cache_burst_mode": bool
        }
        """
        logging.debug(f"KernelDriverInterface.send_cache_hint() stub: {payload}")


class KernelHooks:
    def __init__(self, driver_iface: KernelDriverInterface):
        self.driver_iface = driver_iface

    def apply_scheduler_hint(self, policy: SchedulerPolicy):
        payload = {
            "background_throttle_level": policy.background_throttle_level,
            "core_affinity_map": policy.core_affinity_map,
        }
        self.driver_iface.send_scheduler_hint(payload)

    def apply_memory_hint(self, policy: MemoryPolicy):
        payload = {
            "tier_thresholds": policy.tier_thresholds,
            "migration_budget_mb": policy.migration_budget_mb,
        }
        self.driver_iface.send_memory_hint(payload)

    def apply_cache_hint(self, policy: CachePolicy):
        payload = {
            "cache_appetite": policy.cache_appetite,
            "vram_bias_factor": policy.vram_bias_factor,
            "ghost_cache_burst_mode": policy.ghost_cache_burst_mode,
        }
        self.driver_iface.send_cache_hint(payload)


# =========================
# Scheduler, Cache, Memory
# =========================

class SchedulerCore:
    def __init__(self, num_cores: int):
        self.num_cores = num_cores
        self.run_queues: Dict[int, queue.Queue] = {
            i: queue.Queue() for i in range(num_cores)
        }
        self.policy = SchedulerPolicy()
        self._lock = threading.Lock()

    def local_tick(self):
        with self._lock:
            for core_id in range(self.num_cores):
                _ = self.run_queues[core_id].qsize()

    def export_state(self) -> SchedulerExport:
        with self._lock:
            runq_depth = [self.run_queues[i].qsize() for i in range(self.num_cores)]
            latency_hist = [random.random() * 10 for _ in range(self.num_cores)]
            stall_rate = [random.random() for _ in range(self.num_cores)]
            return SchedulerExport(
                runq_depth=runq_depth,
                latency_hist=latency_hist,
                stall_rate=stall_rate,
            )

    def apply_policy(self, policy: SchedulerPolicy):
        with self._lock:
            self.policy = policy


class CachePolicyEngine:
    def __init__(self):
        self.policy = CachePolicy()
        self._lock = threading.Lock()
        self._hit_counters = {"L1": 0, "L2": 0, "L3": 0, "GHOST": 0}
        self._access_counters = {"L1": 1, "L2": 1, "L3": 1, "GHOST": 1}
        self._bandwidth_use = {"RAM": 0.0, "VRAM": 0.0, "SSD": 0.0}

    def local_tick(self):
        with self._lock:
            for level in self._hit_counters:
                self._hit_counters[level] += random.randint(0, 10)
                self._access_counters[level] += random.randint(1, 20)
            for dev in self._bandwidth_use:
                self._bandwidth_use[dev] = max(
                    0.0, min(1.0, self._bandwidth_use[dev] + random.uniform(-0.1, 0.1))
                )

    def export_state(self) -> CacheExport:
        with self._lock:
            hit_rate = {}
            for level in self._hit_counters:
                hits = self._hit_counters[level]
                acc = self._access_counters[level]
                hit_rate[level] = hits / acc if acc > 0 else 0.0
            prefetch_accuracy = random.uniform(0.5, 0.99)
            return CacheExport(
                hit_rate=hit_rate,
                prefetch_accuracy=prefetch_accuracy,
                bandwidth_use=dict(self._bandwidth_use),
            )

    def apply_policy(self, policy: CachePolicy):
        with self._lock:
            self.policy = policy


class UnifiedMemoryFabric:
    def __init__(self):
        self.policy = MemoryPolicy()
        self._lock = threading.Lock()
        self._ram_pressure = 0.0
        self._vram_pressure = 0.0
        self._page_fault_rate = 0.0
        self._migration_latency = 0.0

    def local_tick(self):
        with self._lock:
            self._ram_pressure = max(
                0.0, min(1.0, self._ram_pressure + random.uniform(-0.05, 0.05))
            )
            self._vram_pressure = max(
                0.0, min(1.0, self._vram_pressure + random.uniform(-0.05, 0.05))
            )
            self._page_fault_rate = max(
                0.0, self._page_fault_rate + random.uniform(-0.1, 0.1)
            )
            self._migration_latency = max(
                0.0, self._migration_latency + random.uniform(-0.2, 0.2)
            )

    def export_state(self) -> MemoryExport:
        with self._lock:
            return MemoryExport(
                ram_pressure=self._ram_pressure,
                vram_pressure=self._vram_pressure,
                page_fault_rate=self._page_fault_rate,
                migration_latency=self._migration_latency,
            )

    def apply_policy(self, policy: MemoryPolicy):
        with self._lock:
            self.policy = policy


# =========================
# Mode, Fluid, Health
# =========================

class ModeStateMachine:
    def __init__(self):
        self.mode = "Flow"
        self.last_mode = "Flow"

    def update(self, fluid: FluidModel, health: HealthStatus, global_export: GlobalExport, gpu: GPUExport) -> str:
        cpu_p = fluid.pressure.get("cpu", 0.0)
        ram_p = fluid.pressure.get("ram", 0.0)
        vram_p = fluid.pressure.get("vram", 0.0)
        thermal_margin = global_export.thermal_margin
        conf = health.confidence
        health_score = health.health

        if cpu_p > 0.85 or ram_p > 0.9 or vram_p > 0.9 or thermal_margin < 0.1 or health_score < 0.5:
            new_mode = "Survival"
        elif self.mode == "Survival" and (cpu_p < 0.7 and ram_p < 0.7 and vram_p < 0.7 and thermal_margin > 0.2):
            new_mode = "Recovery"
        elif cpu_p < 0.3 and ram_p < 0.3 and vram_p < 0.3 and conf > 0.8 and health_score > 0.8:
            new_mode = "Dream"
        else:
            new_mode = "Flow"

        if new_mode != self.mode:
            logging.info(f"Mode transition: {self.mode} -> {new_mode}")
            self.last_mode = self.mode
            self.mode = new_mode

        return self.mode


class FluidModelBuilder:
    def __init__(self):
        self.prev_cpu = None
        self.prev_ram = None
        self.prev_vram = None
        self.prev_time = None

    def build(self, state: StateVector) -> FluidModel:
        cpu_p = state.global_export.cpu_util
        ram_p = state.global_export.mem_util
        vram_p = state.gpu_export.vram_util if state.gpu_export.has_gpu else state.memory_export.vram_pressure
        io_p = state.memory_export.page_fault_rate / 10.0
        io_p = max(0.0, min(1.0, io_p))

        now = state.timestamp
        if self.prev_time is None:
            cpu_flow = ram_flow = vram_flow = io_flow = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            cpu_flow = (cpu_p - (self.prev_cpu or cpu_p)) / dt
            ram_flow = (ram_p - (self.prev_ram or ram_p)) / dt
            vram_flow = (vram_p - (self.prev_vram or vram_p)) / dt
            io_flow = 0.0

        self.prev_cpu = cpu_p
        self.prev_ram = ram_p
        self.prev_vram = vram_p
        self.prev_time = now

        friction_cpu = 0.3 + 0.7 * cpu_p
        friction_ram = 0.3 + 0.7 * ram_p
        friction_vram = 0.3 + 0.7 * vram_p
        friction_io = 0.3 + 0.7 * io_p

        return FluidModel(
            pressure={"cpu": cpu_p, "ram": ram_p, "vram": vram_p, "io": io_p},
            flow={"cpu": cpu_flow, "ram": ram_flow, "vram": vram_flow, "io": io_flow},
            friction={"cpu": friction_cpu, "ram": friction_ram, "vram": friction_vram, "io": friction_io},
        )


class HealthEstimator:
    def __init__(self):
        self.error_count = 0
        self.last_error_time = None

    def update(self, state: StateVector) -> HealthStatus:
        cpu_p = state.global_export.cpu_util
        ram_p = state.global_export.mem_util
        vram_p = state.fluid.pressure.get("vram", 0.0)
        thermal_margin = state.global_export.thermal_margin

        health = 1.0
        if cpu_p > 0.9 or ram_p > 0.95 or vram_p > 0.95:
            health -= 0.3
        if thermal_margin < 0.1:
            health -= 0.3
        health = max(0.0, min(1.0, health))

        confidence = 1.0
        if psutil is None:
            confidence -= 0.3
        if self.error_count > 0:
            confidence -= min(0.5, 0.1 * self.error_count)
        confidence = max(0.0, min(1.0, confidence))

        return HealthStatus(health=health, confidence=confidence, last_error=None)

    def register_error(self, msg: str):
        self.error_count += 1
        self.last_error_time = time.time()
        logging.warning(f"Health error registered: {msg}")


# =========================
# Deep RL Agent (PPO-style with parallel rollouts)
# =========================

class DeepRLAgent:
    """
    PPO-style RL agent:
    - PyTorch neural networks (policy + value) if available
    - GAE advantage estimation
    - Entropy regularization
    - PPO clipping
    - Minibatch training
    - Rollout buffer
    - Prioritized replay (SQLite)
    - Conceptual parallel rollouts / multi-env harness
    """

    def __init__(self, persistence: Persistence, input_dim: int = 8, hidden_dim: int = 64, num_envs: int = 4):
        self.persistence = persistence
        self.use_torch = torch is not None
        self.policy_net = None
        self.value_net = None
        self.optimizer = None
        self.gamma = 0.99
        self.lam = 0.95
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.clip_eps = 0.2
        self.lr = 1e-3
        self.rollout = []  # (state_vec, action, log_prob, reward, value, done)
        self.rollout_max_len = 512
        self.minibatch_size = 64
        self.ppo_epochs = 3
        self.last_action = {"cache_appetite": 1.0}
        self.num_envs = max(1, num_envs)

        if self.use_torch:
            try:
                class PolicyNet(torch.nn.Module):
                    def __init__(self, in_dim, hid_dim):
                        super().__init__()
                        self.fc1 = torch.nn.Linear(in_dim, hid_dim)
                        self.fc2 = torch.nn.Linear(hid_dim, hid_dim)
                        self.mu = torch.nn.Linear(hid_dim, 1)
                        self.log_std = torch.nn.Parameter(torch.zeros(1))

                    def forward(self, x):
                        x = torch.relu(self.fc1(x))
                        x = torch.relu(self.fc2(x))
                        mu = torch.tanh(self.mu(x))
                        std = torch.exp(self.log_std)
                        return mu, std

                class ValueNet(torch.nn.Module):
                    def __init__(self, in_dim, hid_dim):
                        super().__init__()
                        self.fc1 = torch.nn.Linear(in_dim, hid_dim)
                        self.fc2 = torch.nn.Linear(hid_dim, hid_dim)
                        self.out = torch.nn.Linear(hid_dim, 1)

                    def forward(self, x):
                        x = torch.relu(self.fc1(x))
                        x = torch.relu(self.fc2(x))
                        x = self.out(x)
                        return x

                self.policy_net = PolicyNet(input_dim, hidden_dim)
                self.value_net = ValueNet(input_dim, hidden_dim)
                params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
                self.optimizer = torch.optim.Adam(params, lr=self.lr)
                logging.info("DeepRLAgent: using PyTorch PPO-style networks with clipping + GAE + multi-env harness")
            except Exception as e:
                logging.warning(f"DeepRLAgent: torch init failed: {e}")
                self.use_torch = False

    def _encode_state(self, state: StateVector, reward: float) -> List[float]:
        cpu_p = state.fluid.pressure.get("cpu", 0.0)
        ram_p = state.fluid.pressure.get("ram", 0.0)
        vram_p = state.fluid.pressure.get("vram", 0.0)
        io_p = state.fluid.pressure.get("io", 0.0)
        health = state.health.health
        conf = state.health.confidence
        mode_val = {"Flow": 0.0, "Survival": 1.0, "Recovery": 0.5, "Dream": -0.5}.get(state.mode, 0.0)
        reward_clamped = max(-2.0, min(2.0, reward))
        return [cpu_p, ram_p, vram_p, io_p, health, conf, mode_val, reward_clamped]

    def _store_rollout(self, state_vec, action, log_prob, reward, value, done):
        self.rollout.append((state_vec, action, log_prob, reward, value, done))
        if len(self.rollout) > self.rollout_max_len:
            self.rollout.pop(0)

    def _train_from_rollout(self):
        if not self.use_torch or not self.rollout:
            return

        states = torch.tensor([r[0] for r in self.rollout], dtype=torch.float32)
        actions = torch.tensor([r[1] for r in self.rollout], dtype=torch.float32).unsqueeze(-1)
        old_log_probs = torch.tensor([r[2] for r in self.rollout], dtype=torch.float32).unsqueeze(-1)
        rewards = torch.tensor([r[3] for r in self.rollout], dtype=torch.float32)
        values = torch.tensor([r[4] for r in self.rollout], dtype=torch.float32)
        dones = torch.tensor([r[5] for r in self.rollout], dtype=torch.float32)

        # GAE
        advantages = []
        gae = 0.0
        with torch.no_grad():
            next_values = torch.cat([values[1:], values[-1:]], dim=0)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        idxs = list(range(dataset_size))

        for _ in range(self.ppo_epochs):
            random.shuffle(idxs)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = idxs[start:end]
                if not mb_idx:
                    continue
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx].unsqueeze(-1)
                mb_returns = returns[mb_idx].unsqueeze(-1)

                mu, std = self.policy_net(mb_states)
                dist = torch.distributions.Normal(mu, std)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.value_net(mb_states)
                value_loss = torch.mean((mb_returns - value_pred) ** 2)

                loss = self.value_coef * value_loss + policy_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def observe(self, state: StateVector, reward: float, done: bool = False):
        vec = self._encode_state(state, reward)
        priority = abs(reward) + 0.1
        self.persistence.save_replay(vec, reward, priority)

        if self.use_torch and self.policy_net is not None and self.value_net is not None:
            try:
                x = torch.tensor([vec], dtype=torch.float32)
                with torch.no_grad():
                    v = self.value_net(x).item()
                last_a = self.last_action.get("raw_action", 0.0)
                last_lp = self.last_action.get("log_prob", 0.0)
                self._store_rollout(vec, last_a, last_lp, reward, v, float(done))
                if len(self.rollout) >= self.rollout_max_len // 2:
                    self._train_from_rollout()
            except Exception as e:
                logging.debug(f"DeepRLAgent observe error: {e}")

    def act(self, base_cache_appetite: float, state: StateVector, reward: float) -> float:
        if self.use_torch and self.policy_net is not None:
            try:
                vec = self._encode_state(state, reward)
                x = torch.tensor([vec], dtype=torch.float32)
                mu, std = self.policy_net(x)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
                a_val = action.item()
                delta = a_val * 0.2
                self.last_action["raw_action"] = a_val
                self.last_action["log_prob"] = log_prob
            except Exception:
                delta = 0.0
        else:
            delta = 0.05 * state.health.confidence
            if state.mode == "Dream":
                delta *= 1.5
            elif state.mode == "Survival":
                delta *= 0.5

        new_appetite = base_cache_appetite + delta
        new_appetite = max(0.1, min(3.0, new_appetite))
        self.last_action["cache_appetite"] = new_appetite
        return new_appetite


# =========================
# Swarm Consensus (Raft + snapshotting + compaction + CRDT mesh + Queens)
# =========================

class RaftLog:
    def __init__(self, persistence: Persistence):
        self.entries: List[Dict[str, Any]] = []
        self.term: int = 0
        self.commit_index: int = -1
        self.persistence = persistence
        self.snapshot_last_index = -1
        self.snapshot_last_term = 0
        self._load_from_persistence()

    def _load_from_persistence(self):
        snap = self.persistence.load_latest_raft_snapshot()
        if snap is not None:
            self.snapshot_last_index, self.snapshot_last_term, _ = snap
        rows = self.persistence.load_raft_log()
        for term, idx, entry in rows:
            while len(self.entries) <= idx:
                self.entries.append({})
            self.entries[idx] = {"term": term, "entry": entry}
        if self.entries:
            self.term = max(e["term"] for e in self.entries if e)
            self.commit_index = len(self.entries) - 1

    def append(self, term: int, entry: Dict[str, Any]):
        index_pos = len(self.entries)
        self.entries.append({"term": term, "entry": entry})
        self.persistence.append_raft_entry(term, index_pos, entry)

    def get_last_index_term(self) -> Tuple[int, int]:
        if not self.entries:
            return self.snapshot_last_index, self.snapshot_last_term
        idx = len(self.entries) - 1
        return idx, self.entries[idx]["term"]

    def validate_append(self, prev_index: int, prev_term: int) -> bool:
        if prev_index == -1:
            return True
        if prev_index < self.snapshot_last_index:
            return prev_term == self.snapshot_last_term
        local_index = prev_index - self.snapshot_last_index - 1
        if local_index < 0 or local_index >= len(self.entries):
            return False
        return self.entries[local_index]["term"] == prev_term

    def resolve_conflict_and_append(self, prev_index: int, prev_term: int, term: int, entry: Dict[str, Any], index_pos: int):
        if not self.validate_append(prev_index, prev_term):
            if prev_index >= self.snapshot_last_index:
                local_index = prev_index - self.snapshot_last_index - 1
                if 0 <= local_index < len(self.entries):
                    self.entries = self.entries[:local_index + 1]
        local_index = index_pos - self.snapshot_last_index - 1
        while len(self.entries) < local_index:
            self.entries.append({})
        if len(self.entries) == local_index:
            self.entries.append({"term": term, "entry": entry})
        else:
            self.entries[local_index] = {"term": term, "entry": entry}
        self.persistence.append_raft_entry(term, index_pos, entry)

    def commit_up_to(self, index_pos: int):
        self.commit_index = max(self.commit_index, index_pos)

    def maybe_snapshot_and_compact(self, global_state: Dict[str, Any], threshold: int = 200):
        """
        If log grows beyond threshold, snapshot and compact.
        """
        last_index, last_term = self.get_last_index_term()
        if last_index - self.snapshot_last_index < threshold:
            return
        self.persistence.save_raft_snapshot(last_index, last_term, global_state)
        self.snapshot_last_index = last_index
        self.snapshot_last_term = last_term
        self.entries = []
        self.persistence.compact_raft_log(last_index + 1)


class SwarmConsensus:
    def __init__(self, node_id: str, persistence: Persistence):
        self.node_id = node_id
        self.term = 0
        self.leader_id: Optional[str] = None
        self.voted_for: Optional[str] = None
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(3.0, 5.0)
        self.log = RaftLog(persistence=persistence)

    def on_heartbeat(self, leader_id: str, term: int, commit_index: int):
        if term > self.term:
            self.term = term
            self.leader_id = leader_id
            self.voted_for = leader_id
        elif term == self.term and self.leader_id != leader_id:
            self.leader_id = leader_id
        self.last_heartbeat = time.time()
        self.log.commit_up_to(commit_index)

    def should_start_election(self) -> bool:
        return (time.time() - self.last_heartbeat) > self.election_timeout

    def start_election(self, peers: List[str]) -> Tuple[int, str]:
        self.term += 1
        self.leader_id = self.node_id
        self.voted_for = self.node_id
        self.last_heartbeat = time.time()
        return self.term, self.node_id

    def is_leader(self) -> bool:
        return self.leader_id == self.node_id or self.leader_id is None

    def append_log_entry(self, entry: Dict[str, Any]):
        if not self.is_leader():
            return
        self.log.append(self.term, entry)


class ClusterManager:
    """
    Multi-node awareness via UDP gossip + Raft-style consensus + CRDT mesh + Borg-style Queens.
    CRDT fabric uses timestamped deltas.
    """

    def __init__(self, node_id: str, persistence: Persistence, port: int = 50999):
        self.node_id = node_id
        self.port = port
        self.nodes: Dict[str, ClusterNodeView] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._sock = None
        self.consensus = SwarmConsensus(node_id=node_id, persistence=persistence)
        self._policy_mesh: Dict[str, Dict[str, Any]] = {}
        self._mesh_lock = threading.Lock()
        self.queen_id: Optional[str] = None
        self.persistence = persistence

    def start(self):
        try:
            self._sock = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_DGRAM)
            self._sock.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
            self._sock.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_BROADCAST, 1)
            self._sock.bind(("", self.port))
            self._thread.start()
            logging.info(f"ClusterManager listening on UDP port {self.port}")
        except Exception as e:
            logging.warning(f"ClusterManager failed to bind UDP: {e}")
            self._sock = None

    def stop(self):
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def _loop(self):
        while not self._stop.is_set() and self._sock:
            try:
                self._sock.settimeout(1.0)
                try:
                    data, addr = self._sock.recvfrom(8192)
                except socket_mod.timeout:
                    continue
                msg = json.loads(data.decode("utf-8"))
                msg_type = msg.get("type", "state")
                if msg_type == "state":
                    self._handle_state_msg(msg)
                elif msg_type == "heartbeat":
                    self._handle_heartbeat_msg(msg)
                elif msg_type == "policy":
                    self._handle_policy_msg(msg)
                elif msg_type == "raft_append":
                    self._handle_raft_append(msg)
            except Exception:
                continue

    def _handle_state_msg(self, msg: Dict[str, Any]):
        nid = msg.get("node_id", "")
        if not nid or nid == self.node_id:
            return
        now = time.time()
        view = ClusterNodeView(
            node_id=nid,
            last_seen=now,
            mode=msg.get("mode", "unknown"),
            health=msg.get("health", 0.0),
            cpu_pressure=msg.get("cpu_pressure", 0.0),
            ram_pressure=msg.get("ram_pressure", 0.0),
            is_leader=msg.get("is_leader", False),
            term=msg.get("term", 0),
            voted_for=msg.get("voted_for", None),
            weight=msg.get("weight", 1.0),
            role=msg.get("role", "drone"),
        )
        self.nodes[nid] = view
        self._recompute_leader_and_queen()

    def _handle_heartbeat_msg(self, msg: Dict[str, Any]):
        leader_id = msg.get("leader_id", "")
        term = msg.get("term", 0)
        commit_index = msg.get("commit_index", -1)
        if leader_id:
            self.consensus.on_heartbeat(leader_id, term, commit_index)

    def _handle_policy_msg(self, msg: Dict[str, Any]):
        nid = msg.get("node_id", "")
        if not nid:
            return
        with self._mesh_lock:
            # timestamped delta CRDT
            ts = msg.get("ts", time.time())
            pol = msg.get("policy", {})
            existing = self._policy_mesh.get(nid)
            if existing is None or ts >= existing.get("ts", 0):
                pol["ts"] = ts
                self._policy_mesh[nid] = pol

    def _handle_raft_append(self, msg: Dict[str, Any]):
        entry = msg.get("entry", {})
        term = msg.get("term", 0)
        index_pos = msg.get("index_pos", -1)
        prev_index = msg.get("prev_index", -1)
        prev_term = msg.get("prev_term", 0)
        if index_pos < 0:
            return
        self.consensus.log.resolve_conflict_and_append(prev_index, prev_term, term, entry, index_pos)

    def _recompute_leader_and_queen(self):
        now = time.time()
        pruned = {}
        for nid, v in self.nodes.items():
            if now - v.last_seen < 10.0:
                pruned[nid] = v
        self.nodes = pruned

        active_nodes = list(self.nodes.keys()) + [self.node_id]
        best_term = self.consensus.term
        best_leader = self.node_id
        for nid in active_nodes:
            term = self.nodes.get(nid, ClusterNodeView(term=0)).term
            if term > best_term or (term == best_term and nid < best_leader):
                best_term = term
                best_leader = nid
        self.consensus.term = best_term
        self.consensus.leader_id = best_leader

        best_score = -1.0
        queen = self.node_id
        for nid in active_nodes:
            v = self.nodes.get(nid)
            if v:
                score = v.health * 0.7 + v.weight * 0.3
            else:
                score = 0.5
            if score > best_score:
                best_score = score
                queen = nid
        self.queen_id = queen

    def broadcast_state(self, mode: str, health: float, cpu_p: float, ram_p: float, weight: float, role: str):
        if not self._sock:
            return
        is_leader = self.consensus.is_leader()
        msg = {
            "type": "state",
            "node_id": self.node_id,
            "mode": mode,
            "health": health,
            "cpu_pressure": cpu_p,
            "ram_pressure": ram_p,
            "is_leader": is_leader,
            "term": self.consensus.term,
            "voted_for": self.consensus.voted_for,
            "weight": weight,
            "role": role,
        }
        data = json.dumps(msg).encode("utf-8")
        try:
            self._sock.sendto(data, ("255.255.255.255", self.port))
        except Exception:
            pass

    def broadcast_heartbeat(self):
        if not self._sock:
            return
        msg = {
            "type": "heartbeat",
            "leader_id": self.consensus.leader_id or self.node_id,
            "term": self.consensus.term,
            "commit_index": self.consensus.log.commit_index,
        }
        data = json.dumps(msg).encode("utf-8")
        try:
            self._sock.sendto(data, ("255.255.255.255", self.port))
        except Exception:
            pass

    def broadcast_policy(self, policy: PolicyBundle):
        if not self._sock:
            return
        msg = {
            "type": "policy",
            "node_id": self.node_id,
            "ts": time.time(),
            "policy": {
                "scheduler": {
                    "background_throttle_level": policy.scheduler_policy.background_throttle_level
                },
                "cache": {
                    "cache_appetite": policy.cache_policy.cache_appetite,
                    "ghost_cache_burst_mode": policy.cache_policy.ghost_cache_burst_mode,
                },
                "memory": {
                    "migration_budget_mb": policy.memory_policy.migration_budget_mb
                },
            },
        }
        data = json.dumps(msg).encode("utf-8")
        try:
            self._sock.sendto(data, ("255.255.255.255", self.port))
        except Exception:
            pass

    def broadcast_raft_append(self, entry: Dict[str, Any]):
        if not self._sock:
            return
        last_index, last_term = self.consensus.log.get_last_index_term()
        msg = {
            "type": "raft_append",
            "node_id": self.node_id,
            "term": self.consensus.term,
            "index_pos": last_index + 1,
            "prev_index": last_index,
            "prev_term": last_term,
            "entry": entry,
        }
        data = json.dumps(msg).encode("utf-8")
        try:
            self._sock.sendto(data, ("255.255.255.255", self.port))
        except Exception:
            pass

    def maybe_start_election(self):
        if self.consensus.should_start_election():
            peers = list(self.nodes.keys())
            term, leader = self.consensus.start_election(peers)
            logging.info(f"Cluster election started: term={term}, leader={leader}")

    def get_cluster_state(self) -> ClusterState:
        now = time.time()
        pruned = {}
        for nid, view in self.nodes.items():
            if now - view.last_seen < 10.0:
                pruned[nid] = view
        self.nodes = pruned
        self._recompute_leader_and_queen()
        return ClusterState(
            self_id=self.node_id,
            leader_id=self.consensus.leader_id,
            queen_id=self.queen_id,
            term=self.consensus.term,
            nodes=pruned,
        )

    def merge_policies_crdt_with_queen(self, local_policy: PolicyBundle, local_weight: float, role: str) -> PolicyBundle:
        with self._mesh_lock:
            mesh = dict(self._policy_mesh)

        total_weight = local_weight
        agg_sched_throttle = local_policy.scheduler_policy.background_throttle_level * local_weight
        agg_cache_appetite = local_policy.cache_policy.cache_appetite * local_weight
        agg_migration_budget = local_policy.memory_policy.migration_budget_mb * local_weight
        ghost_mode = local_policy.cache_policy.ghost_cache_burst_mode

        queen_policy = None
        leader_policy = None

        for nid, pol in mesh.items():
            node_view = self.nodes.get(nid)
            if not node_view:
                continue
            w = max(0.1, node_view.weight)
            if nid == self.queen_id:
                queen_policy = pol
            if node_view.is_leader:
                leader_policy = pol
            total_weight += w
            s = pol.get("scheduler", {})
            c = pol.get("cache", {})
            m = pol.get("memory", {})
            agg_sched_throttle += s.get("background_throttle_level", 0.0) * w
            agg_cache_appetite += c.get("cache_appetite", 1.0) * w
            agg_migration_budget += m.get("migration_budget_mb", 128.0) * w
            ghost_mode = ghost_mode or c.get("ghost_cache_burst_mode", False)

        if queen_policy is not None:
            s = queen_policy.get("scheduler", {})
            c = queen_policy.get("cache", {})
            m = queen_policy.get("memory", {})
            return PolicyBundle(
                scheduler_policy=SchedulerPolicy(
                    core_affinity_map=local_policy.scheduler_policy.core_affinity_map,
                    burst_boost_windows=local_policy.scheduler_policy.burst_boost_windows,
                    background_throttle_level=s.get("background_throttle_level", 0.5),
                ),
                cache_policy=CachePolicy(
                    cache_appetite=c.get("cache_appetite", 1.5),
                    read_ahead_distance=local_policy.cache_policy.read_ahead_distance,
                    vram_bias_factor=local_policy.cache_policy.vram_bias_factor,
                    ghost_cache_burst_mode=c.get("ghost_cache_burst_mode", True),
                ),
                memory_policy=MemoryPolicy(
                    tier_thresholds=local_policy.memory_policy.tier_thresholds,
                    migration_budget_mb=m.get("migration_budget_mb", 512.0),
                    prewarm_targets=local_policy.memory_policy.prewarm_targets,
                ),
            )

        if leader_policy is not None and role == "drone":
            s = leader_policy.get("scheduler", {})
            c = leader_policy.get("cache", {})
            m = leader_policy.get("memory", {})
            return PolicyBundle(
                scheduler_policy=SchedulerPolicy(
                    core_affinity_map=local_policy.scheduler_policy.core_affinity_map,
                    burst_boost_windows=local_policy.scheduler_policy.burst_boost_windows,
                    background_throttle_level=s.get("background_throttle_level", 0.5),
                ),
                cache_policy=CachePolicy(
                    cache_appetite=c.get("cache_appetite", 1.2),
                    read_ahead_distance=local_policy.cache_policy.read_ahead_distance,
                    vram_bias_factor=local_policy.cache_policy.vram_bias_factor,
                    ghost_cache_burst_mode=c.get("ghost_cache_burst_mode", ghost_mode),
                ),
                memory_policy=MemoryPolicy(
                    tier_thresholds=local_policy.memory_policy.tier_thresholds,
                    migration_budget_mb=m.get("migration_budget_mb", 256.0),
                    prewarm_targets=local_policy.memory_policy.prewarm_targets,
                ),
            )

        if total_weight <= 0:
            return local_policy

        merged_sched = SchedulerPolicy(
            core_affinity_map=local_policy.scheduler_policy.core_affinity_map,
            burst_boost_windows=local_policy.scheduler_policy.burst_boost_windows,
            background_throttle_level=agg_sched_throttle / total_weight,
        )
        merged_cache = CachePolicy(
            cache_appetite=agg_cache_appetite / total_weight,
            read_ahead_distance=local_policy.cache_policy.read_ahead_distance,
            vram_bias_factor=local_policy.cache_policy.vram_bias_factor,
            ghost_cache_burst_mode=ghost_mode,
        )
        merged_mem = MemoryPolicy(
            tier_thresholds=local_policy.memory_policy.tier_thresholds,
            migration_budget_mb=agg_migration_budget / total_weight,
            prewarm_targets=local_policy.memory_policy.prewarm_targets,
        )
        return PolicyBundle(
            scheduler_policy=merged_sched,
            cache_policy=merged_cache,
            memory_policy=merged_mem,
        )


# =========================
# Policy Engine
# =========================

class PolicyEngine:
    def __init__(self, node_id: str, persistence: Persistence, cluster_manager: ClusterManager):
        self._last_policy = PolicyBundle()
        self._last_reward = 0.0
        self._cache_appetite = 1.0
        self.mode_machine = ModeStateMachine()
        self.fluid_builder = FluidModelBuilder()
        self.health_estimator = HealthEstimator()
        self.deep_rl = DeepRLAgent(persistence=persistence)
        self.node_id = node_id
        self.cluster_manager = cluster_manager

    def _compute_reward(self, state: StateVector) -> float:
        avg_stall = (
            sum(state.scheduler_export.stall_rate) / len(state.scheduler_export.stall_rate)
            if state.scheduler_export.stall_rate
            else 0.0
        )
        avg_hit = (
            sum(state.cache_export.hit_rate.values()) / len(state.cache_export.hit_rate)
            if state.cache_export.hit_rate
            else 0.0
        )
        ram_p = state.memory_export.ram_pressure
        cpu_u = state.global_export.cpu_util
        vram_p = state.fluid.pressure.get("vram", 0.0)

        reward = (avg_hit * 2.0) - (avg_stall + ram_p + cpu_u + 0.5 * vram_p)

        if state.cluster.leader_id and state.cluster.leader_id != self.node_id:
            reward *= 0.9

        return reward

    def compute_policy(self, state: StateVector, global_snapshot_for_raft: Dict[str, Any]) -> PolicyBundle:
        state.fluid = self.fluid_builder.build(state)
        state.health = self.health_estimator.update(state)
        state.mode = self.mode_machine.update(state.fluid, state.health, state.global_export, state.gpu_export)

        reward = self._compute_reward(state)
        self.deep_rl.observe(state, reward, done=False)

        base_appetite = self._cache_appetite
        new_appetite = self.deep_rl.act(base_appetite, state, reward)
        self._cache_appetite = new_appetite
        self._last_reward = reward

        avg_stall = (
            sum(state.scheduler_export.stall_rate) / len(state.scheduler_export.stall_rate)
            if state.scheduler_export.stall_rate
            else 0.0
        )

        if state.mode == "Survival":
            background_throttle = min(1.0, max(0.5, avg_stall * 2.0))
        elif state.mode == "Recovery":
            background_throttle = min(1.0, max(0.2, avg_stall * 1.2))
        elif state.mode == "Dream":
            background_throttle = min(0.5, avg_stall)
        else:
            background_throttle = min(1.0, max(0.0, avg_stall * 1.5))

        scheduler_policy = SchedulerPolicy(
            core_affinity_map={},
            burst_boost_windows={},
            background_throttle_level=background_throttle,
        )

        cache_policy = CachePolicy(
            cache_appetite=self._cache_appetite,
            read_ahead_distance={"SSD": 64, "RAM": 16},
            vram_bias_factor=1.2,
            ghost_cache_burst_mode=state.global_export.thermal_margin > 0.2,
        )

        ram_p = state.memory_export.ram_pressure
        vram_p = state.memory_export.vram_pressure

        tier_thresholds = {"RAM": 0.8, "VRAM": 0.9, "SSD": 0.95}
        if state.mode == "Survival":
            migration_budget_mb = 512.0
        elif state.mode == "Dream":
            migration_budget_mb = 128.0
        else:
            migration_budget_mb = 256.0 if (ram_p > 0.7 or vram_p > 0.7) else 64.0

        memory_policy = MemoryPolicy(
            tier_thresholds=tier_thresholds,
            migration_budget_mb=migration_budget_mb,
            prewarm_targets=["hot_region_1", "hot_region_2"],
        )

        local_bundle = PolicyBundle(
            scheduler_policy=scheduler_policy,
            cache_policy=cache_policy,
            memory_policy=memory_policy,
        )

        local_weight = (state.health.health + state.health.confidence) / 2.0
        role = "queen" if self.cluster_manager.queen_id == self.node_id else "drone"
        merged_bundle = self.cluster_manager.merge_policies_crdt_with_queen(local_bundle, local_weight, role)
        self._last_policy = merged_bundle

        if self.cluster_manager.consensus.is_leader():
            self.cluster_manager.consensus.append_log_entry(
                {"policy": "update", "timestamp": state.timestamp}
            )
            self.cluster_manager.broadcast_raft_append(
                {"policy": "update", "timestamp": state.timestamp}
            )
            # Raft snapshot + compaction
            self.cluster_manager.consensus.log.maybe_snapshot_and_compact(global_snapshot_for_raft)

        return merged_bundle


# =========================
# System Controller
# =========================

class SystemController:
    def __init__(self, num_cores: int, gcc_hz: float = 100.0):
        hostname = socket_mod.gethostname()
        pid = os.getpid()
        self.node_id = f"{hostname}-{pid}"

        self.persistence = Persistence()
        self.cluster_manager = ClusterManager(node_id=self.node_id, persistence=self.persistence)

        self.scheduler = SchedulerCore(num_cores=num_cores)
        self.cache_engine = CachePolicyEngine()
        self.memory_fabric = UnifiedMemoryFabric()
        self.telemetry = Telemetry()
        self.os_hooks = OSControlHooks()
        self.kernel_driver_iface = KernelDriverInterface()
        self.kernel_hooks = KernelHooks(self.kernel_driver_iface)
        self.policy_engine = PolicyEngine(
            node_id=self.node_id,
            persistence=self.persistence,
            cluster_manager=self.cluster_manager,
        )

        self.gcc_period = 1.0 / gcc_hz
        self._stop_flag = threading.Event()
        self._threads: List[threading.Thread] = []

        self._last_state: Optional[StateVector] = None
        self._last_policy: Optional[PolicyBundle] = None
        self._state_lock = threading.Lock()

        self.rest_port: Optional[int] = None
        self.rest_status: str = "initializing"

    def _scheduler_loop(self):
        while not self._stop_flag.is_set():
            try:
                self.scheduler.local_tick()
            except Exception as e:
                logging.error(f"Scheduler loop error: {e}")
            time.sleep(0.001)

    def _cache_loop(self):
        while not self._stop_flag.is_set():
            try:
                self.cache_engine.local_tick()
            except Exception as e:
                logging.error(f"Cache loop error: {e}")
            time.sleep(0.002)

    def _memory_loop(self):
        while not self._stop_flag.is_set():
            try:
                self.memory_fabric.local_tick()
            except Exception as e:
                logging.error(f"Memory loop error: {e}")
            time.sleep(0.005)

    def _consensus_loop(self):
        while not self._stop_flag.is_set():
            try:
                self.cluster_manager.maybe_start_election()
                if self.cluster_manager.consensus.is_leader():
                    self.cluster_manager.broadcast_heartbeat()
            except Exception:
                pass
            time.sleep(1.0)

    def _gcc_loop(self):
        while not self._stop_flag.is_set():
            start = time.time()
            try:
                state = self._collect_state()
                state.cluster = self.cluster_manager.get_cluster_state()
                # snapshot used for Raft snapshotting
                snap_for_raft = self._build_snapshot_for_raft(state)
                policy = self.policy_engine.compute_policy(state, snap_for_raft)

                self.scheduler.apply_policy(policy.scheduler_policy)
                self.cache_engine.apply_policy(policy.cache_policy)
                self.memory_fabric.apply_policy(policy.memory_policy)

                self.os_hooks.apply_background_throttle(
                    policy.scheduler_policy.background_throttle_level
                )
                self.kernel_hooks.apply_scheduler_hint(policy.scheduler_policy)
                self.kernel_hooks.apply_memory_hint(policy.memory_policy)
                self.kernel_hooks.apply_cache_hint(policy.cache_policy)

                weight = (state.health.health + state.health.confidence) / 2.0
                role = "queen" if self.cluster_manager.queen_id == self.node_id else "drone"
                self.cluster_manager.broadcast_state(
                    mode=state.mode,
                    health=state.health.health,
                    cpu_p=state.fluid.pressure.get("cpu", 0.0),
                    ram_p=state.fluid.pressure.get("ram", 0.0),
                    weight=weight,
                    role=role,
                )
                self.cluster_manager.broadcast_policy(policy)

                with self._state_lock:
                    self._last_state = state
                    self._last_policy = policy

                snap = self.get_snapshot()
                self.persistence.save_snapshot(snap)
            except Exception as e:
                logging.error(f"GCC loop error: {e}")

            elapsed = time.time() - start
            remaining = self.gcc_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _collect_state(self) -> StateVector:
        sched_export = self.scheduler.export_state()
        cache_export = self.cache_engine.export_state()
        mem_export = self.memory_fabric.export_state()
        global_export = self.telemetry.collect_global()
        gpu_export = self.telemetry.collect_gpu()
        return StateVector(
            scheduler_export=sched_export,
            cache_export=cache_export,
            memory_export=mem_export,
            global_export=global_export,
            gpu_export=gpu_export,
            timestamp=time.time(),
        )

    def _build_snapshot_for_raft(self, state: StateVector) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "mode": state.mode,
            "health": {
                "health": state.health.health,
                "confidence": state.health.confidence,
            },
            "fluid": {
                "pressure": state.fluid.pressure,
                "flow": state.fluid.flow,
                "friction": state.fluid.friction,
            },
            "global": {
                "cpu_util": state.global_export.cpu_util,
                "mem_util": state.global_export.mem_util,
                "thermal_margin": state.global_export.thermal_margin,
            },
        }

    def get_snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            state = self._last_state
            policy = self._last_policy
        if state is None or policy is None:
            snap = self.persistence.load_snapshot()
            if snap is not None:
                return snap
            return {"status": "initializing"}

        cluster_nodes = {
            nid: {
                "last_seen": v.last_seen,
                "mode": v.mode,
                "health": v.health,
                "cpu_pressure": v.cpu_pressure,
                "ram_pressure": v.ram_pressure,
                "is_leader": v.is_leader,
                "term": v.term,
                "weight": v.weight,
                "role": v.role,
            }
            for nid, v in state.cluster.nodes.items()
        }

        return {
            "status": "ok",
            "timestamp": state.timestamp,
            "node_id": self.node_id,
            "mode": state.mode,
            "health": {
                "health": state.health.health,
                "confidence": state.health.confidence,
            },
            "fluid": {
                "pressure": state.fluid.pressure,
                "flow": state.fluid.flow,
                "friction": state.fluid.friction,
            },
            "gpu": {
                "has_gpu": state.gpu_export.has_gpu,
                "gpu_util": state.gpu_export.gpu_util,
                "vram_util": state.gpu_export.vram_util,
                "temperature": state.gpu_export.temperature,
                "power_draw": state.gpu_export.power_draw,
            },
            "rest": {
                "port": self.rest_port,
                "status": self.rest_status,
            },
            "cluster": {
                "self_id": state.cluster.self_id,
                "leader_id": state.cluster.leader_id,
                "queen_id": state.cluster.queen_id,
                "term": state.cluster.term,
                "nodes": cluster_nodes,
                "commit_index": self.cluster_manager.consensus.log.commit_index,
            },
            "state": {
                "scheduler": {
                    "runq_depth": state.scheduler_export.runq_depth,
                    "stall_rate": state.scheduler_export.stall_rate,
                },
                "cache": {
                    "hit_rate": state.cache_export.hit_rate,
                    "prefetch_accuracy": state.cache_export.prefetch_accuracy,
                },
                "memory": {
                    "ram_pressure": state.memory_export.ram_pressure,
                    "vram_pressure": state.memory_export.vram_pressure,
                },
                "global": {
                    "cpu_util": state.global_export.cpu_util,
                    "mem_util": state.global_export.mem_util,
                    "thermal_margin": state.global_export.thermal_margin,
                },
            },
            "policy": {
                "scheduler": {
                    "background_throttle_level": policy.scheduler_policy.background_throttle_level
                },
                "cache": {
                    "cache_appetite": policy.cache_policy.cache_appetite,
                    "ghost_cache_burst_mode": policy.cache_policy.ghost_cache_burst_mode,
                },
                "memory": {
                    "migration_budget_mb": policy.memory_policy.migration_budget_mb
                },
            },
        }

    def export_vr_layout(self, path: str = "vr_layout.json"):
        """
        VR cockpit export: holographic overlays + animated swarm orbits (data-level).
        A real VR client can animate orbits based on 'orbit_phase' and 'orbit_radius'.
        """
        snap = self.get_snapshot()
        fluid = snap.get("fluid", {}).get("pressure", {})
        cluster = snap.get("cluster", {})
        nodes_meta = cluster.get("nodes", {})

        nodes = [
            {"id": "CPU", "x": 0.0, "y": 0.0, "z": fluid.get("cpu", 0.0), "type": "core"},
            {"id": "RAM", "x": 1.0, "y": 0.0, "z": fluid.get("ram", 0.0), "type": "core"},
            {"id": "VRAM", "x": 0.0, "y": 1.0, "z": fluid.get("vram", 0.0), "type": "core"},
            {"id": "IO", "x": 1.0, "y": 1.0, "z": fluid.get("io", 0.0), "type": "core"},
        ]

        angle_step = 2 * math.pi / max(1, len(nodes_meta) or 1)
        radius = 2.0
        idx = 0
        for nid, meta in nodes_meta.items():
            angle = idx * angle_step
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0.5 + 0.5 * meta.get("health", 0.0)
            role = meta.get("role", "drone")
            nodes.append(
                {
                    "id": f"node:{nid}",
                    "x": x,
                    "y": y,
                    "z": z,
                    "type": "queen" if role == "queen" else "drone",
                    "orbit_radius": radius,
                    "orbit_phase": angle,
                }
            )
            idx += 1

        edges = [
            {"from": "CPU", "to": "RAM"},
            {"from": "CPU", "to": "VRAM"},
            {"from": "RAM", "to": "IO"},
            {"from": "VRAM", "to": "IO"},
        ]
        for n in nodes:
            if n["id"].startswith("node:"):
                edges.append({"from": "CPU", "to": n["id"]})

        hud = {
            "mode": snap.get("mode"),
            "health": snap.get("health", {}),
            "cluster": {
                "leader_id": cluster.get("leader_id"),
                "queen_id": cluster.get("queen_id"),
                "term": cluster.get("term"),
            },
            "rest": snap.get("rest", {}),
        }

        layout = {"nodes": nodes, "edges": edges, "hud": hud, "timestamp": snap.get("timestamp")}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(layout, f, indent=2)
        except Exception as e:
            logging.debug(f"export_vr_layout failed: {e}")

    def start(self):
        self._stop_flag.clear()
        self.kernel_driver_iface.load_driver()
        self.cluster_manager.start()
        self._threads = [
            threading.Thread(target=self._scheduler_loop, daemon=True),
            threading.Thread(target=self._cache_loop, daemon=True),
            threading.Thread(target=self._memory_loop, daemon=True),
            threading.Thread(target=self._gcc_loop, daemon=True),
            threading.Thread(target=self._consensus_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()
        logging.info("Core controller started")

    def stop(self):
        self._stop_flag.set()
        self.cluster_manager.stop()
        self.kernel_driver_iface.unload_driver()
        for t in self._threads:
            t.join(timeout=1.0)
        logging.info("Core controller stopped")


# =========================
# REST API
# =========================

class RESTRequestHandler(http_server.BaseHTTPRequestHandler):
    controller: Optional[SystemController] = None

    def _send_json(self, obj: Any, code: int = 200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/state":
            if self.controller is None:
                self._send_json({"error": "controller not attached"}, 500)
                return
            snap = self.controller.get_snapshot()
            self._send_json(snap)
        elif self.path == "/meta":
            if self.controller is None:
                self._send_json({"error": "controller not attached"}, 500)
                return
            self._send_json(
                {
                    "rest_port": self.controller.rest_port,
                    "rest_status": self.controller.rest_status,
                    "platform": platform.platform(),
                    "node_id": self.controller.node_id,
                }
            )
        elif self.path == "/vr_layout":
            if self.controller is None:
                self._send_json({"error": "controller not attached"}, 500)
                return
            self.controller.export_vr_layout("vr_layout.json")
            try:
                with open("vr_layout.json", "r", encoding="utf-8") as f:
                    layout = json.load(f)
            except Exception:
                layout = {"error": "failed to read vr_layout.json"}
            self._send_json(layout)
        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, format, *args):
        return


class RESTServerThread(threading.Thread):
    def __init__(self, controller: SystemController, host: str = "127.0.0.1", preferred_port: int = 8080):
        super().__init__(daemon=True)
        self.controller = controller
        self.host = host
        self.preferred_port = preferred_port
        self.httpd = None
        self.port: Optional[int] = None
        self.status: str = "initializing"

    def _try_bind(self, host: str, port: int):
        try:
            httpd = socketserver.TCPServer((host, port), RESTRequestHandler)
            return httpd
        except OSError as e:
            logging.warning(f"REST bind failed on {host}:{port} with {e}")
            return None

    def run(self):
        RESTRequestHandler.controller = self.controller

        httpd = None
        chosen_port = None
        status = "failed"

        httpd = self._try_bind(self.host, self.preferred_port)
        if httpd:
            chosen_port = self.preferred_port
            status = "ok"
        else:
            for p in range(self.preferred_port + 1, self.preferred_port + 11):
                httpd = self._try_bind(self.host, p)
                if httpd:
                    chosen_port = p
                    status = "ok"
                    break

        if httpd is None:
            logging.info("Trying auto-port selection (port=0)")
            httpd = self._try_bind(self.host, 0)
            if httpd:
                chosen_port = httpd.server_address[1]
                status = "ok_auto"
            else:
                status = "bind_error"
                logging.error(
                    "REST server failed to bind any port. "
                    "Possible firewall/AV restriction (WinError 10013)."
                )

        self.httpd = httpd
        self.port = chosen_port
        self.status = status

        self.controller.rest_port = chosen_port
        self.controller.rest_status = status

        if httpd is None:
            return

        logging.info(f"REST server listening on {self.host}:{self.port} (status={self.status})")
        try:
            httpd.serve_forever()
        except Exception as e:
            logging.error(f"REST server error: {e}")

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            logging.info("REST server stopped")


# =========================
# GUI Cockpit
# =========================

class GUICockpit:
    def __init__(self, controller: SystemController, refresh_ms: int = 1000):
        if tkinter is None:
            raise RuntimeError("Tkinter not available")
        self.controller = controller
        self.refresh_ms = refresh_ms
        self.root = tkinter.Tk()
        self.root.title("UnifiedBrain Cockpit")

        self.label = tkinter.Label(self.root, text="Initializing...", font=("Consolas", 9), justify="left")
        self.label.pack(padx=10, pady=10)

        self.alert = tkinter.Label(self.root, text="", font=("Consolas", 9), fg="red", justify="left")
        self.alert.pack(padx=10, pady=5)

    def _update(self):
        snap = self.controller.get_snapshot()
        text = json.dumps(snap, indent=2)
        self.label.config(text=text)

        rest_status = snap.get("rest", {}).get("status", "unknown")
        rest_port = snap.get("rest", {}).get("port", None)
        mode = snap.get("mode", "unknown")
        health = snap.get("health", {}).get("health", 0.0)
        conf = snap.get("health", {}).get("confidence", 0.0)
        cluster = snap.get("cluster", {})
        queen_id = cluster.get("queen_id")

        if rest_status.startswith("ok"):
            rest_msg = f"REST OK on port {rest_port}"
            rest_color = "green"
        elif rest_status == "bind_error":
            rest_msg = "REST bind error: check firewall/AV (WinError 10013 likely)."
            rest_color = "red"
        else:
            rest_msg = f"REST status: {rest_status}"
            rest_color = "orange"

        self.alert.config(
            text=f"{rest_msg}\nMode: {mode} | Health: {health:.2f} | Confidence: {conf:.2f}\nQueen: {queen_id}",
            fg=rest_color,
        )

        self.root.after(self.refresh_ms, self._update)

    def run(self):
        self._update()
        self.root.mainloop()


# =========================
# 3D Cockpit
# =========================

class Visualization3D:
    def __init__(self, controller: SystemController, refresh_sec: float = 1.0):
        if matplotlib is None or plt is None:
            raise RuntimeError("matplotlib not available")
        self.controller = controller
        self.refresh_sec = refresh_sec

    def run(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("CPU pressure")
        ax.set_ylabel("RAM pressure")
        ax.set_zlabel("VRAM pressure")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        point, = ax.plot([0], [0], [0], "ro")

        def update(_):
            snap = self.controller.get_snapshot()
            fluid = snap.get("fluid", {}).get("pressure", {})
            cpu = fluid.get("cpu", 0.0)
            ram = fluid.get("ram", 0.0)
            vram = fluid.get("vram", 0.0)
            point.set_data([cpu], [ram])
            point.set_3d_properties([vram])
            return point,

        import matplotlib.animation as animation
        animation.FuncAnimation(fig, update, interval=int(self.refresh_sec * 1000))
        plt.show()


# =========================
# Windows Service wrapper (optional)
# =========================

UnifiedBrainService = None
if IS_WINDOWS:
    try:
        import win32serviceutil
        import win32service
        import win32event
        import servicemanager

        class UnifiedBrainService(win32serviceutil.ServiceFramework):
            _svc_name_ = "UnifiedBrainService"
            _svc_display_name_ = "Unified Swarm Autonomous Performance Brain"
            _svc_description_ = "Cross-platform autonomous scheduler/cache/memory/swarm controller"

            def __init__(self, args):
                super().__init__(args)
                self.stop_event = win32event.CreateEvent(None, 0, 0, None)
                self.controller = SystemController(num_cores=8, gcc_hz=100)
                self.rest_thread = RESTServerThread(self.controller)

            def SvcStop(self):
                self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
                win32event.SetEvent(self.stop_event)
                self.rest_thread.stop()
                self.controller.stop()

            def SvcDoRun(self):
                servicemanager.LogInfoMsg("UnifiedBrain starting...")
                self.controller.start()
                self.rest_thread.start()
                win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)
                servicemanager.LogInfoMsg("UnifiedBrain stopped.")

    except ImportError:
        UnifiedBrainService = None


# =========================
# Entrypoint
# =========================

def run_foreground_daemon(num_cores: int = 8, gcc_hz: float = 100.0, gui: bool = False, viz3d: bool = False):
    controller = SystemController(num_cores=num_cores, gcc_hz=gcc_hz)
    rest_server = RESTServerThread(controller)

    def handle_stop(signum, frame):
        rest_server.stop()
        controller.stop()
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, handle_stop)
        signal.signal(signal.SIGINT, handle_stop)
    except Exception:
        pass

    controller.start()
    rest_server.start()

    if gui and tkinter is not None:
        logging.info("Starting GUI cockpit")
        cockpit = GUICockpit(controller)
        cockpit.run()
    elif viz3d and matplotlib is not None and plt is not None:
        logging.info("Starting 3D visualization cockpit")
        viz = Visualization3D(controller)
        viz.run()
    else:
        logging.info("Running in autonomous mode. REST /state, /meta, /vr_layout available.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    args = sys.argv[1:]
    use_gui = "--gui" in args
    use_viz3d = "--viz3d" in args

    if IS_WINDOWS and UnifiedBrainService is not None and any(
        a in args for a in ("install", "start", "stop", "remove")
    ):
        win32serviceutil.HandleCommandLine(UnifiedBrainService)
    else:
        run_foreground_daemon(num_cores=8, gcc_hz=100.0, gui=use_gui, viz3d=use_viz3d)
