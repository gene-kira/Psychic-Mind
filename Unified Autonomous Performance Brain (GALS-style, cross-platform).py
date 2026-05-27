#!/usr/bin/env python3
"""
Unified Autonomous Performance Brain (GALS-style, cross-platform, extended, with diagnostics)

Features:
- Independent local domains (Scheduler, Cache, Memory)
- Global Coordination Clock (GCC) with PolicyEngine + simple RL
- Real telemetry via psutil (if available)
- OS control hooks (priority / niceness hints)
- REST API for state/policy inspection
- Auto-port selection, port fallback, basic port scanning, diagnostic logging
- REST endpoint exposing chosen port and status
- Optional GUI cockpit (Tkinter) with REST port + status alert
- Can run:
    * As a foreground daemon on any OS
    * As a Windows Service (if pywin32 is available and invoked with service args)
"""

# =========================
# Autoloader for libraries
# =========================

def _autoload_libs():
    import importlib
    import sys

    required = [
        "time",
        "threading",
        "queue",
        "dataclasses",
        "typing",
        "random",
        "signal",
        "sys",
        "platform",
        "os",
        "json",
        "http.server",
        "socketserver",
        "logging",
    ]

    ns = {}
    for name in required:
        if name in sys.modules:
            mod = sys.modules[name]
        else:
            mod = importlib.import_module(name)
        ns[name] = mod

    # Optional libs
    optional = ["psutil", "tkinter"]
    for name in optional:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            mod = None
        ns[name] = mod

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
psutil = _libs["psutil"]
tkinter = _libs["tkinter"]

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[UnifiedBrain] %(asctime)s %(levelname)s: %(message)s",
)


# =========================
# Shared data structures
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
    hit_rate: Dict[str, float] = field(default_factory=dict)  # L1/L2/L3/GHOST
    prefetch_accuracy: float = 0.0
    bandwidth_use: Dict[str, float] = field(default_factory=dict)  # RAM/VRAM/SSD


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
    tier_thresholds: Dict[str, float] = field(default_factory=dict)  # RAM/VRAM/SSD
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
class StateVector:
    scheduler_export: SchedulerExport = field(default_factory=SchedulerExport)
    cache_export: CacheExport = field(default_factory=CacheExport)
    memory_export: MemoryExport = field(default_factory=MemoryExport)
    global_export: GlobalExport = field(default_factory=GlobalExport)
    timestamp: float = 0.0


@dataclass
class PolicyBundle:
    scheduler_policy: SchedulerPolicy = field(default_factory=SchedulerPolicy)
    cache_policy: CachePolicy = field(default_factory=CachePolicy)
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)


# =========================
# Telemetry (real where possible)
# =========================

class Telemetry:
    def __init__(self):
        self.has_psutil = psutil is not None
        if self.has_psutil:
            logging.info("psutil detected: using real telemetry")
        else:
            logging.warning("psutil not available: using synthetic telemetry")

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


# =========================
# OS Control Hooks (safe hints)
# =========================

class OSControlHooks:
    def __init__(self):
        self.has_psutil = psutil is not None
        self.is_windows = platform.system().lower().startswith("windows")
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
# Scheduler Core (local domain)
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
            pass

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
            pass


# =========================
# Cache Policy Engine (local domain)
# =========================

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
            pass

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
            pass


# =========================
# Unified Memory Fabric (local domain)
# =========================

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
            pass

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
            pass


# =========================
# Policy Engine with simple RL
# =========================

class PolicyEngine:
    def __init__(self):
        self._last_policy = PolicyBundle()
        self._last_reward = 0.0
        self._cache_appetite = 1.0

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

        reward = (avg_hit * 2.0) - (avg_stall + ram_p + cpu_u)
        return reward

    def _rl_update(self, reward: float):
        delta = reward - self._last_reward
        if delta > 0:
            self._cache_appetite = min(2.0, self._cache_appetite + 0.05)
        else:
            self._cache_appetite = max(0.2, self._cache_appetite - 0.05)
        self._last_reward = reward

    def compute_policy(self, state: StateVector) -> PolicyBundle:
        reward = self._compute_reward(state)
        self._rl_update(reward)

        avg_stall = (
            sum(state.scheduler_export.stall_rate) / len(state.scheduler_export.stall_rate)
            if state.scheduler_export.stall_rate
            else 0.0
        )
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

        tier_thresholds = {
            "RAM": 0.8,
            "VRAM": 0.9,
            "SSD": 0.95,
        }

        migration_budget_mb = 256.0 if (ram_p > 0.7 or vram_p > 0.7) else 64.0

        memory_policy = MemoryPolicy(
            tier_thresholds=tier_thresholds,
            migration_budget_mb=migration_budget_mb,
            prewarm_targets=["hot_region_1", "hot_region_2"],
        )

        bundle = PolicyBundle(
            scheduler_policy=scheduler_policy,
            cache_policy=cache_policy,
            memory_policy=memory_policy,
        )
        self._last_policy = bundle
        return bundle


# =========================
# System Controller
# =========================

class SystemController:
    def __init__(self, num_cores: int, gcc_hz: float = 100.0):
        self.scheduler = SchedulerCore(num_cores=num_cores)
        self.cache_engine = CachePolicyEngine()
        self.memory_fabric = UnifiedMemoryFabric()
        self.policy_engine = PolicyEngine()
        self.telemetry = Telemetry()
        self.os_hooks = OSControlHooks()

        self.gcc_period = 1.0 / gcc_hz
        self._stop_flag = threading.Event()
        self._threads: List[threading.Thread] = []

        self._last_state: Optional[StateVector] = None
        self._last_policy: Optional[PolicyBundle] = None
        self._state_lock = threading.Lock()

        self.rest_port: Optional[int] = None
        self.rest_status: str = "initializing"

    # Local domain loops

    def _scheduler_loop(self):
        while not self._stop_flag.is_set():
            self.scheduler.local_tick()
            time.sleep(0.001)

    def _cache_loop(self):
        while not self._stop_flag.is_set():
            self.cache_engine.local_tick()
            time.sleep(0.002)

    def _memory_loop(self):
        while not self._stop_flag.is_set():
            self.memory_fabric.local_tick()
            time.sleep(0.005)

    # GCC loop

    def _gcc_loop(self):
        while not self._stop_flag.is_set():
            start = time.time()

            state = self._collect_state()
            policy = self.policy_engine.compute_policy(state)

            self.scheduler.apply_policy(policy.scheduler_policy)
            self.cache_engine.apply_policy(policy.cache_policy)
            self.memory_fabric.apply_policy(policy.memory_policy)

            self.os_hooks.apply_background_throttle(
                policy.scheduler_policy.background_throttle_level
            )

            with self._state_lock:
                self._last_state = state
                self._last_policy = policy

            elapsed = time.time() - start
            remaining = self.gcc_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _collect_state(self) -> StateVector:
        sched_export = self.scheduler.export_state()
        cache_export = self.cache_engine.export_state()
        mem_export = self.memory_fabric.export_state()
        global_export = self.telemetry.collect_global()

        return StateVector(
            scheduler_export=sched_export,
            cache_export=cache_export,
            memory_export=mem_export,
            global_export=global_export,
            timestamp=time.time(),
        )

    def get_snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            state = self._last_state
            policy = self._last_policy
        if state is None or policy is None:
            return {"status": "initializing"}

        return {
            "status": "ok",
            "timestamp": state.timestamp,
            "rest": {
                "port": self.rest_port,
                "status": self.rest_status,
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

    def start(self):
        self._stop_flag.clear()
        self._threads = [
            threading.Thread(target=self._scheduler_loop, daemon=True),
            threading.Thread(target=self._cache_loop, daemon=True),
            threading.Thread(target=self._memory_loop, daemon=True),
            threading.Thread(target=self._gcc_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()
        logging.info("Core controller started")

    def stop(self):
        self._stop_flag.set()
        for t in self._threads:
            t.join(timeout=1.0)
        logging.info("Core controller stopped")


# =========================
# REST API (http.server) with auto-port + diagnostics
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
                }
            )
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

        # 1) Try preferred port
        httpd = self._try_bind(self.host, self.preferred_port)
        if httpd:
            chosen_port = self.preferred_port
            status = "ok"
        else:
            # 2) Scan a small range
            for p in range(self.preferred_port + 1, self.preferred_port + 11):
                httpd = self._try_bind(self.host, p)
                if httpd:
                    chosen_port = p
                    status = "ok"
                    break

        # 3) Auto-port (0) if still no luck
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
# GUI Cockpit (Tkinter, optional)
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

        # UI alert for REST status
        rest_status = snap.get("rest", {}).get("status", "unknown")
        rest_port = snap.get("rest", {}).get("port", None)
        if rest_status.startswith("ok"):
            self.alert.config(
                text=f"REST OK on port {rest_port}",
                fg="green",
            )
        elif rest_status == "bind_error":
            self.alert.config(
                text="REST bind error: check firewall/AV (WinError 10013 likely).",
                fg="red",
            )
        else:
            self.alert.config(
                text=f"REST status: {rest_status}",
                fg="orange",
            )

        self.root.after(self.refresh_ms, self._update)

    def run(self):
        self._update()
        self.root.mainloop()


# =========================
# Windows Service wrapper (optional)
# =========================

IS_WINDOWS = platform.system().lower().startswith("windows")
UnifiedBrainService = None
if IS_WINDOWS:
    try:
        import win32serviceutil
        import win32service
        import win32event
        import servicemanager

        class UnifiedBrainService(win32serviceutil.ServiceFramework):
            _svc_name_ = "UnifiedBrainService"
            _svc_display_name_ = "Unified Autonomous Performance Brain"
            _svc_description_ = "Cross-platform autonomous scheduler/cache/memory controller"

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
# Cross-platform daemon entrypoint
# =========================

def run_foreground_daemon(num_cores: int = 8, gcc_hz: float = 100.0, gui: bool = False):
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
    else:
        logging.info("Running in autonomous mode. REST /state and /meta available.")
        while True:
            time.sleep(1.0)


# =========================
# Main
# =========================

if __name__ == "__main__":
    args = sys.argv[1:]
    use_gui = "--gui" in args

    if IS_WINDOWS and UnifiedBrainService is not None and any(
        a in args for a in ("install", "start", "stop", "remove")
    ):
        win32serviceutil.HandleCommandLine(UnifiedBrainService)
    else:
        run_foreground_daemon(num_cores=8, gcc_hz=100.0, gui=use_gui)
