#!/usr/bin/env python3
"""
Universal Negative-Space Organism v6

Features:
- Universal: Windows / Linux / macOS
- Config system (tunable thresholds, swarm, GUI, memory path)
- Persistent memory (JSON: event stats, trends, usage)
- Negative-Space Flow Theory core (unified brain)
- Telemetry: CPU, RAM, disk, network, GPU, thermal, processes, UI state
- Predictive brain:
    - smoothing + trend detection
    - Bernoulli spike predictors (CPU/MEM/DISK)
    - simple online linear regression for CPU/MEM/DISK
    - UI-intent-aware constraint shaping
- OS adapters with real, controlled actions:
    - Linux: CPU governor changes, process nice
    - Windows: process priority via psutil
    - macOS: process nice via psutil (QoS stub)
- Organs:
    - GPUOrgan
    - ThermalOrgan
    - DiskOrgan
    - NetworkOrgan
    - ProcessOrgan
    - SwarmOrgan (multi-machine via UDP broadcast)
    - CockpitOrgan (text)
    - GUICockpitOrgan (Tkinter, optional)
"""

import sys
import time
import platform
import traceback
from typing import Dict, Any, Optional, List
import threading as _threading_std

# ============================================================
# CONFIG SYSTEM
# ============================================================

CONFIG: Dict[str, Any] = {
    "poll_interval": 2.0,
    "memory_file": "nsf_memory.json",
    "swarm": {
        "enabled": True,
        "host": "<broadcast>",  # or specific subnet broadcast
        "port": 50505,
    },
    "gui": {
        "enabled": True,
        "title": "Negative-Space Cockpit",
        "refresh_ms": 1000,
    },
    "thresholds": {
        "cpu_spike_prob": 0.3,
        "mem_spike_prob": 0.3,
        "disk_spike_prob": 0.3,
    },
    "ui_patterns": {
        "gaming": ["game", "steam", "epic", "unity", "unreal"],
        "browsing": ["chrome", "edge", "firefox", "browser"],
        "creative_heavy": ["premiere", "resolve", "after effects", "blender"],
    },
}

# ============================================================
# AUTOLOADER
# ============================================================

def autoload(modules: List[str]) -> Dict[str, Any]:
    loaded = {}
    for name in modules:
        try:
            loaded[name] = __import__(name)
        except ImportError:
            loaded[name] = None
    return loaded

LIBS = autoload([
    "psutil",
    "subprocess",
    "GPUtil",
    "socket",
    "json",
    "threading",
    "os",
    "shutil",
    "uiautomation",
    "pywinauto",
    "win32gui",
    "win32api",
    "tkinter",
])

psutil = LIBS["psutil"]
GPUtil = LIBS["GPUtil"]
socket = LIBS["socket"]
json = LIBS["json"]
threading = LIBS["threading"] or _threading_std
uiautomation = LIBS["uiautomation"]
pywinauto = LIBS["pywinauto"]
win32gui = LIBS["win32gui"]
win32api = LIBS["win32api"]
tkinter = LIBS["tkinter"]

# ============================================================
# PERSISTENT MEMORY
# ============================================================

class PersistentMemory:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {
            "events": {
                "cpu_spike": {"success": 0, "total": 0},
                "mem_spike": {"success": 0, "total": 0},
                "disk_spike": {"success": 0, "total": 0},
            },
            "usage": {
                "runs": 0,
            },
        }
        self._load()

    def _load(self):
        if not json:
            return
        try:
            import os
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.data.update(loaded)
        except Exception:
            pass

    def save(self):
        if not json:
            return
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def get_events(self) -> Dict[str, Dict[str, int]]:
        return self.data.setdefault("events", {})

    def set_events(self, events: Dict[str, Dict[str, int]]) -> None:
        self.data["events"] = events

    def increment_runs(self):
        self.data.setdefault("usage", {}).setdefault("runs", 0)
        self.data["usage"]["runs"] += 1

# ============================================================
# BASE OS ADAPTER
# ============================================================

class OSAdapter:
    def __init__(self, libs: Dict[str, Any]):
        self.libs = libs

    def telemetry(self) -> Dict[str, Any]:
        raise NotImplementedError

    def apply_constraints(self, constraints: Dict[str, Any]) -> List[str]:
        raise NotImplementedError

    def read_ui_state(self) -> Dict[str, Any]:
        return {}

# ============================================================
# COMMON TELEMETRY HELPERS
# ============================================================

def collect_basic_telemetry() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not psutil:
        data["warning"] = "psutil_not_available"
        return data

    try:
        data["cpu_percent"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        data["mem_percent"] = vm.percent

        disk = psutil.disk_usage("/")
        data["disk_percent"] = disk.percent

        net = psutil.net_io_counters()
        data["net_bytes_sent"] = net.bytes_sent
        data["net_bytes_recv"] = net.bytes_recv
    except Exception:
        data["error"] = "psutil_telemetry_failed"
        data["traceback"] = traceback.format_exc()

    return data

def collect_thermal_telemetry() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not psutil:
        return data
    try:
        temps = psutil.sensors_temperatures()
        data["thermal_sensors"] = {k: [t.current for t in v] for k, v in temps.items()} if temps else {}
    except Exception:
        data["thermal_warning"] = "thermal_not_available_or_failed"
    return data

def collect_gpu_telemetry() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not GPUtil:
        data["gpu_warning"] = "GPUtil_not_available"
        return data
    try:
        gpus = GPUtil.getGPUs()
        info = []
        for g in gpus:
            info.append({
                "id": g.id,
                "load": g.load * 100.0,
                "mem_util": g.memoryUtil * 100.0,
                "temp": g.temperature,
                "name": g.name,
            })
        data["gpu_info"] = info
    except Exception:
        data["gpu_error"] = "gpu_telemetry_failed"
        data["gpu_traceback"] = traceback.format_exc()
    return data

def collect_process_telemetry(limit: int = 10) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not psutil:
        return data
    try:
        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
            info = p.info
            procs.append(info)
        procs.sort(key=lambda x: x.get("cpu_percent", 0.0), reverse=True)
        data["top_processes"] = procs[:limit]
    except Exception:
        data["process_warning"] = "process_telemetry_failed"
        data["process_traceback"] = traceback.format_exc()
    return data

# ============================================================
# WINDOWS ADAPTER (with UIAutomation)
# ============================================================

class WindowsAdapter(OSAdapter):
    def telemetry(self) -> Dict[str, Any]:
        data = {"os": "Windows"}
        data.update(collect_basic_telemetry())
        data.update(collect_thermal_telemetry())
        data.update(collect_gpu_telemetry())
        data.update(collect_process_telemetry())
        data["ui_state"] = self.read_ui_state()
        return data

    def read_ui_state(self) -> Dict[str, Any]:
        ui: Dict[str, Any] = {"source": None}
        if uiautomation:
            try:
                from uiautomation import GetForegroundControl
                ctrl = GetForegroundControl()
                ui["source"] = "uiautomation"
                ui["name"] = ctrl.Name
                ui["class"] = ctrl.ClassName
                ui["control_type"] = str(ctrl.ControlType)
                return ui
            except Exception:
                pass
        if pywinauto:
            try:
                from pywinauto import Desktop
                fg = Desktop(backend="uia").window(active_only=True)
                ui["source"] = "pywinauto"
                ui["name"] = fg.window_text()
                ui["class"] = fg.class_name()
                return ui
            except Exception:
                pass
        if win32gui:
            try:
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                ui["source"] = "win32gui"
                ui["name"] = title
                ui["hwnd"] = hwnd
                return ui
            except Exception:
                pass
        ui["source"] = "none"
        return ui

    def _set_process_priority(self, pid: int, level: str) -> str:
        if not psutil:
            return "Windows: psutil not available for priority change"
        try:
            p = psutil.Process(pid)
            import psutil as ps
            mapping = {
                "low": ps.IDLE_PRIORITY_CLASS,
                "below_normal": ps.BELOW_NORMAL_PRIORITY_CLASS,
                "normal": ps.NORMAL_PRIORITY_CLASS,
                "above_normal": ps.ABOVE_NORMAL_PRIORITY_CLASS,
                "high": ps.HIGH_PRIORITY_CLASS,
            }
            target = mapping.get(level, ps.NORMAL_PRIORITY_CLASS)
            p.nice(target)
            return f"Windows: set priority of PID {pid} to {level}"
        except PermissionError:
            return f"Windows: insufficient permissions to change priority of PID {pid}"
        except Exception as e:
            return f"Windows: failed to change priority of PID {pid}: {e}"

    def apply_constraints(self, constraints: Dict[str, Any]) -> List[str]:
        actions: List[str] = []
        cpu_policy = constraints.get("cpu_policy")
        if cpu_policy in ("throttle", "emergency_throttle"):
            actions.append("Windows: would hint power plan -> power_saver")
        elif cpu_policy == "balanced":
            actions.append("Windows: would hint power plan -> balanced")

        if constraints.get("background_network_heavy") is False:
            actions.append("Windows: would defer heavy background network tasks")

        gpu_policy = constraints.get("gpu_policy")
        if gpu_policy in ("reduce", "emergency_reduce"):
            actions.append("Windows: would reduce background GPU workloads")

        return actions

# ============================================================
# LINUX ADAPTER
# ============================================================

class LinuxAdapter(OSAdapter):
    def telemetry(self) -> Dict[str, Any]:
        data = {"os": "Linux"}
        data.update(collect_basic_telemetry())
        data.update(collect_thermal_telemetry())
        data.update(collect_gpu_telemetry())
        data.update(collect_process_telemetry())
        data["ui_state"] = self.read_ui_state()
        return data

    def _set_governor(self, governor: str) -> str:
        try:
            import glob
            paths = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
            if not paths:
                return "Linux: governor files not found"

            for path in paths:
                try:
                    with open(path, "w") as f:
                        f.write(governor)
                except PermissionError:
                    return "Linux: insufficient permissions to change governor"
                except Exception as e:
                    return f"Linux: failed to write governor: {e}"

            return f"Linux: governor set to {governor}"
        except Exception as e:
            return f"Linux: governor change failed: {e}"

    def _set_process_nice(self, pid: int, nice_value: int) -> str:
        if not psutil:
            return "Linux: psutil not available for nice change"
        try:
            p = psutil.Process(pid)
            p.nice(nice_value)
            return f"Linux: set nice of PID {pid} to {nice_value}"
        except PermissionError:
            return f"Linux: insufficient permissions to change nice of PID {pid}"
        except Exception as e:
            return f"Linux: failed to change nice of PID {pid}: {e}"

    def apply_constraints(self, constraints: Dict[str, Any]) -> List[str]:
        actions: List[str] = []
        cpu_policy = constraints.get("cpu_policy")

        if cpu_policy in ("throttle", "emergency_throttle"):
            result = self._set_governor("powersave")
            actions.append(result)
        elif cpu_policy == "balanced":
            result = self._set_governor("ondemand")
            actions.append(result)

        if constraints.get("background_network_heavy") is False:
            actions.append("Linux: would defer heavy background network tasks")

        gpu_policy = constraints.get("gpu_policy")
        if gpu_policy in ("reduce", "emergency_reduce"):
            actions.append("Linux: would reduce background GPU workloads")

        return actions

# ============================================================
# MACOS ADAPTER
# ============================================================

class MacOSAdapter(OSAdapter):
    def telemetry(self) -> Dict[str, Any]:
        data = {"os": "macOS"}
        data.update(collect_basic_telemetry())
        data.update(collect_thermal_telemetry())
        data.update(collect_gpu_telemetry())
        data.update(collect_process_telemetry())
        data["ui_state"] = self.read_ui_state()
        return data

    def _set_process_nice(self, pid: int, nice_value: int) -> str:
        if not psutil:
            return "macOS: psutil not available for nice change"
        try:
            p = psutil.Process(pid)
            p.nice(nice_value)
            return f"macOS: set nice of PID {pid} to {nice_value}"
        except PermissionError:
            return f"macOS: insufficient permissions to change nice of PID {pid}"
        except Exception as e:
            return f"macOS: failed to change nice of PID {pid}: {e}"

    def apply_constraints(self, constraints: Dict[str, Any]) -> List[str]:
        actions: List[str] = []
        cpu_policy = constraints.get("cpu_policy")
        if cpu_policy in ("throttle", "emergency_throttle"):
            actions.append("macOS: would hint QoS/App Nap for background CPU tasks")
        elif cpu_policy == "balanced":
            actions.append("macOS: would keep QoS balanced")

        if constraints.get("background_network_heavy") is False:
            actions.append("macOS: would defer heavy background network tasks")

        gpu_policy = constraints.get("gpu_policy")
        if gpu_policy in ("reduce", "emergency_reduce"):
            actions.append("macOS: would reduce background GPU workloads")

        return actions

# ============================================================
# ADAPTER FACTORY
# ============================================================

def get_adapter(libs: Dict[str, Any]) -> OSAdapter:
    system = platform.system().lower()
    if "windows" in system:
        return WindowsAdapter(libs)
    if "linux" in system:
        return LinuxAdapter(libs)
    if "darwin" in system:
        return MacOSAdapter(libs)
    raise RuntimeError(f"Unsupported OS: {system}")

# ============================================================
# NEGATIVE-SPACE ENGINE (PREDICTIVE BRAIN + SIMPLE ML)
# ============================================================

class NegativeSpaceEngine:
    def __init__(self, memory: PersistentMemory):
        self.last_constraints: Dict[str, Any] = {}
        self.last_net: Optional[Dict[str, int]] = None
        self.cpu_history: List[float] = []
        self.mem_history: List[float] = []
        self.disk_history: List[float] = []
        self.history_len = 10

        self.memory = memory
        self.events = self.memory.get_events()

        # Simple online linear regression state: (n, sum_x, sum_y, sum_xx, sum_xy)
        self.reg = {
            "cpu": {"n": 0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "sxy": 0.0},
            "mem": {"n": 0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "sxy": 0.0},
            "disk": {"n": 0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "sxy": 0.0},
        }
        self.t = 0

    def _smooth(self, history: List[float], value: float) -> float:
        history.append(value)
        if len(history) > self.history_len:
            history.pop(0)
        return sum(history) / len(history)

    def _trend(self, history: List[float]) -> float:
        if len(history) < 2:
            return 0.0
        return history[-1] - history[0]

    def _compute_net_delta(self, telemetry: Dict[str, Any]) -> Dict[str, float]:
        sent = telemetry.get("net_bytes_sent")
        recv = telemetry.get("net_bytes_recv")
        if sent is None or recv is None:
            return {"net_sent_delta": 0.0, "net_recv_delta": 0.0}

        if self.last_net is None:
            self.last_net = {"sent": sent, "recv": recv}
            return {"net_sent_delta": 0.0, "net_recv_delta": 0.0}

        delta_sent = max(0, sent - self.last_net["sent"])
        delta_recv = max(0, recv - self.last_net["recv"])
        self.last_net = {"sent": sent, "recv": recv}
        return {
            "net_sent_delta": float(delta_sent),
            "net_recv_delta": float(delta_recv),
        }

    def _update_bernoulli(self, name: str, condition_now: bool, threshold: float) -> bool:
        e = self.events.setdefault(name, {"success": 0, "total": 0})
        e["total"] += 1
        if condition_now:
            e["success"] += 1
        p = e["success"] / e["total"] if e["total"] > 0 else 0.0
        return p >= threshold

    def _update_reg(self, key: str, y: float):
        r = self.reg[key]
        self.t += 1
        x = float(self.t)
        r["n"] += 1
        r["sx"] += x
        r["sy"] += y
        r["sxx"] += x * x
        r["sxy"] += x * y

    def _predict_reg(self, key: str, steps_ahead: int = 5) -> float:
        r = self.reg[key]
        n = r["n"]
        if n < 2:
            return 0.0
        sx, sy, sxx, sxy = r["sx"], r["sy"], r["sxx"], r["sxy"]
        denom = (n * sxx - sx * sx)
        if denom == 0:
            return 0.0
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n
        future_x = float(self.t + steps_ahead)
        return a * future_x + b

    def compute_constraints(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}

        cpu_raw = float(telemetry.get("cpu_percent", 0.0) or 0.0)
        mem_raw = float(telemetry.get("mem_percent", 0.0) or 0.0)
        disk_raw = float(telemetry.get("disk_percent", 0.0) or 0.0)

        self._update_reg("cpu", cpu_raw)
        self._update_reg("mem", mem_raw)
        self._update_reg("disk", disk_raw)

        cpu_pred = self._predict_reg("cpu", steps_ahead=5)
        mem_pred = self._predict_reg("mem", steps_ahead=5)
        disk_pred = self._predict_reg("disk", steps_ahead=5)

        cpu = self._smooth(self.cpu_history, cpu_raw)
        mem = self._smooth(self.mem_history, mem_raw)
        disk = self._smooth(self.disk_history, disk_raw)

        cpu_trend = self._trend(self.cpu_history)
        mem_trend = self._trend(self.mem_history)
        disk_trend = self._trend(self.disk_history)

        net_delta = self._compute_net_delta(telemetry)
        net_sent = net_delta["net_sent_delta"]
        net_recv = net_delta["net_recv_delta"]
        net_total = net_sent + net_recv

        gpu_info = telemetry.get("gpu_info", [])
        max_gpu_load = max((g.get("load", 0.0) for g in gpu_info), default=0.0)
        max_gpu_temp = max((g.get("temp", 0.0) for g in gpu_info), default=0.0)

        thermal_sensors = telemetry.get("thermal_sensors", {})
        max_cpu_temp = 0.0
        if thermal_sensors:
            for _, temps in thermal_sensors.items():
                if temps:
                    max_cpu_temp = max(max_cpu_temp, max(temps))

        ui_state = telemetry.get("ui_state", {})
        ui_name = (ui_state.get("name") or "").lower()

        th = CONFIG["thresholds"]
        cpu_spike_pred = self._update_bernoulli("cpu_spike", cpu_raw >= 90.0, threshold=th["cpu_spike_prob"])
        mem_spike_pred = self._update_bernoulli("mem_spike", mem_raw >= 90.0, threshold=th["mem_spike_prob"])
        disk_spike_pred = self._update_bernoulli("disk_spike", disk_raw >= 90.0, threshold=th["disk_spike_prob"])

        # CPU policy (reactive + predictive + thermal + regression)
        if cpu >= 95 or max_cpu_temp >= 90 or cpu_pred >= 95 or (cpu_trend > 10 and cpu_spike_pred):
            constraints["cpu_policy"] = "emergency_throttle"
        elif cpu >= 90 or max_cpu_temp >= 85 or cpu_pred >= 90 or (cpu_trend > 5 and cpu_spike_pred):
            constraints["cpu_policy"] = "throttle"
        elif cpu >= 75 or max_cpu_temp >= 80 or cpu_pred >= 80 or (cpu_trend > 3 and cpu_spike_pred):
            constraints["cpu_policy"] = "conservative"
        else:
            constraints["cpu_policy"] = "balanced"

        # Memory policy
        if mem >= 95 or mem_pred >= 95 or (mem_trend > 5 and mem_spike_pred):
            constraints["spawn_heavy_jobs"] = False
            constraints["memory_pressure"] = "critical"
        elif mem >= 90 or mem_pred >= 90 or (mem_trend > 3 and mem_spike_pred):
            constraints["spawn_heavy_jobs"] = False
            constraints["memory_pressure"] = "high"
        elif mem >= 80 or mem_pred >= 80 or (mem_trend > 2 and mem_spike_pred):
            constraints["spawn_heavy_jobs"] = False
            constraints["memory_pressure"] = "elevated"
        else:
            constraints["spawn_heavy_jobs"] = True
            constraints["memory_pressure"] = "normal"

        # Disk policy
        if disk >= 95 or disk_pred >= 95 or (disk_trend > 5 and disk_spike_pred):
            constraints["disk_policy"] = "emergency_io_reduce"
        elif disk >= 90 or disk_pred >= 90 or (disk_trend > 3 and disk_spike_pred):
            constraints["disk_policy"] = "io_reduce"
        elif disk >= 80 or disk_pred >= 80 or (disk_trend > 2 and disk_spike_pred):
            constraints["disk_policy"] = "io_careful"
        else:
            constraints["disk_policy"] = "io_normal"

        # Network policy
        if net_total > 10_000_000:
            constraints["background_network_heavy"] = False
            constraints["network_pressure"] = "high"
        elif net_total > 1_000_000:
            constraints["background_network_heavy"] = False
            constraints["network_pressure"] = "elevated"
        else:
            constraints["background_network_heavy"] = True
            constraints["network_pressure"] = "normal"

        # GPU policy
        if max_gpu_load >= 95 or max_gpu_temp >= 85:
            constraints["gpu_policy"] = "emergency_reduce"
        elif max_gpu_load >= 85 or max_gpu_temp >= 80:
            constraints["gpu_policy"] = "reduce"
        elif max_gpu_load >= 70:
            constraints["gpu_policy"] = "careful"
        else:
            constraints["gpu_policy"] = "normal"

        # UI-intent hints
        patterns = CONFIG["ui_patterns"]
        if any(k in ui_name for k in patterns["gaming"]):
            constraints["ui_mode"] = "gaming"
            constraints["spawn_heavy_jobs"] = False
        elif any(k in ui_name for k in patterns["browsing"]):
            constraints["ui_mode"] = "browsing"
        elif any(k in ui_name for k in patterns["creative_heavy"]):
            constraints["ui_mode"] = "creative_heavy"
            constraints["disk_policy"] = "io_careful"

        self.last_constraints = constraints
        self.memory.set_events(self.events)
        return constraints

# ============================================================
# ORGANS
# ============================================================

class Organ:
    def tick(self, telemetry: Dict[str, Any], constraints: Dict[str, Any], actions: List[str]) -> None:
        raise NotImplementedError

class GPUOrgan(Organ):
    def tick(self, telemetry: Dict[str, Any], constraints: Dict[str, Any], actions: List[str]) -> None:
        gpu_info = telemetry.get("gpu_info", [])
        if not gpu_info:
            return
        policy = constraints.get("gpu_policy", "normal")
        if policy in ("reduce", "emergency_reduce"):
            actions.append(f"GPUOrgan: GPU policy={policy}, would reduce background GPU tasks")

class ThermalOrgan(Organ):
    def tick(self, telemetry: Dict[str, Any], constraints: Dict[str, Any], actions: List[str]) -> None:
        thermal = telemetry.get("thermal_sensors", {})
        if not thermal:
            return
        max_temp = 0.0
        for _, temps in thermal.items():
            if temps:
                max_temp = max(max_temp, max(temps))
        if max_temp >= 85:
            actions.append(f"ThermalOrgan: high temp {max_temp:.1f}C, enforcing conservative policies")

class DiskOrgan(Organ):
    def tick(self, telemetry: Dict[str, Any], constraints: Dict[str, Any], actions: List[str]) -> None:
        disk = telemetry.get("disk_percent", 0.0)
        policy = constraints.get("disk_policy", "io_normal")
        if policy != "io_normal":
            actions.append(f"DiskOrgan: disk={disk:.1f}%, policy={policy}, would slow heavy IO")

class NetworkOrgan(Organ):
    def tick(self, telemetry: Dict[str, Any], constraints: Dict[str, Any], actions: List[str]) -> None:
        net_sent = telemetry.get("net_bytes_sent", 0)
        net_recv = telemetry.get("net_bytes_recv", 0)
        pressure = constraints.get("network_pressure", "normal")
        if pressure != "normal":
            actions.append(f"NetworkOrgan: net_sent={net_sent}, net_recv={net_recv}, pressure={pressure}, would defer heavy network tasks")

class ProcessOrgan(Organ):
    def __init__(self, adapter: OSAdapter):
        self.adapter = adapter

    def tick(self, telemetry: Dict[str, Any], constraints: Dict[str, Any], actions: List[str]) -> None:
        top_procs = telemetry.get("top_processes", [])
        if not top_procs:
            return

        hog = top_procs[0]
        pid = hog.get("pid")
        name = hog.get("name")
        cpu = hog.get("cpu_percent", 0.0)

        if cpu < 50.0:
            return

        os_name = telemetry.get("os", "").lower()

        if "linux" in os_name and isinstance(self.adapter, LinuxAdapter):
            result = self.adapter._set_process_nice(pid, 10)
            actions.append(result)
        elif "windows" in os_name and isinstance(self.adapter, WindowsAdapter):
            result = self.adapter._set_process_priority(pid, "below_normal")
            actions.append(result)
        elif "macos" in os_name and isinstance(self.adapter, MacOSAdapter):
            result = self.adapter._set_process_nice(pid, 10)
            actions.append(result)
        else:
            actions.append(f"ProcessOrgan: would lower priority of PID {pid} ({name})")

class SwarmOrgan(Organ):
    def __init__(self, enabled: bool, host: str, port: int):
        self.enabled = enabled and socket is not None and json is not None and threading is not None
        self.host = host
        self.port = port
        if self.enabled:
            self._start_listener()

    def _start_listener(self):
        def listen():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", self.port))
                s.settimeout(0.5)
                while True:
                    try:
                        data, addr = s.recvfrom(4096)
                        _ = data, addr
                    except socket.timeout:
                        continue
                    except Exception:
                        break
            except Exception:
                pass

        t = threading.Thread(target=listen, daemon=True)
        t.start()

    def tick(self, telemetry: Dict[str, Any], constraints: Dict[str, Any], actions: List[str]) -> None:
        if not self.enabled:
            return
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.settimeout(0.01)
            packet = {
                "os": telemetry.get("os"),
                "cpu": telemetry.get("cpu_percent"),
                "mem": telemetry.get("mem_percent"),
                "constraints": constraints,
            }
            data = json.dumps(packet).encode("utf-8")
            s.sendto(data, (self.host, self.port))
            s.close()
        except Exception:
            actions.append("SwarmOrgan: broadcast failed (non-fatal)")

class CockpitOrgan(Organ):
    def _clear(self):
        print("\n" * 5)

    def tick(self, telemetry: Dict[str, Any], constraints: Dict[str, Any], actions: List[str]) -> None:
        self._clear()
        os_name = telemetry.get("os", "?")
        cpu = telemetry.get("cpu_percent", 0.0)
        mem = telemetry.get("mem_percent", 0.0)
        disk = telemetry.get("disk_percent", 0.0)
        net_sent = telemetry.get("net_bytes_sent", 0)
        net_recv = telemetry.get("net_bytes_recv", 0)
        ui_state = telemetry.get("ui_state", {})

        print("=== Negative-Space Cockpit (Text) ===")
        print(f"OS: {os_name}")
        print(f"CPU: {cpu:.1f}%   MEM: {mem:.1f}%   DISK: {disk:.1f}%")
        print(f"NET: sent={net_sent} bytes  recv={net_recv} bytes")
        print(f"UI: {ui_state}")
        print(f"Constraints: {constraints}")

        print("\nActions this tick:")
        if actions:
            for act in actions:
                print(f"  - {act}")
        else:
            print("  (none)")

        top_procs = telemetry.get("top_processes", [])[:5]
        print("\nTop processes (by CPU):")
        for p in top_procs:
            print(f"  PID {p.get('pid')}  {p.get('name')}  "
                  f"CPU {p.get('cpu_percent', 0.0):.1f}%  "
                  f"MEM {p.get('memory_percent', 0.0):.1f}%")

        gpu_info = telemetry.get("gpu_info", [])
        if gpu_info:
            print("\nGPU:")
            for g in gpu_info:
                print(f"  {g.get('name')}  load={g.get('load', 0.0):.1f}%  "
                      f"mem={g.get('mem_util', 0.0):.1f}%  temp={g.get('temp', 0.0):.1f}C")

        print("\n====================================")

class GUICockpitOrgan(Organ):
    def __init__(self, enabled: bool, refresh_ms: int):
        self.enabled = enabled and tkinter is not None
        self.refresh_ms = refresh_ms
        self._telemetry: Dict[str, Any] = {}
        self._constraints: Dict[str, Any] = {}
        self._actions: List[str] = []
        self._thread_started = False
        if self.enabled:
            self._start_gui_thread()

    def _start_gui_thread(self):
        def run_gui():
            root = tkinter.Tk()
            root.title(CONFIG["gui"]["title"])
            labels = {
                "os": tkinter.Label(root, text="OS: "),
                "cpu": tkinter.Label(root, text="CPU: "),
                "mem": tkinter.Label(root, text="MEM: "),
                "disk": tkinter.Label(root, text="DISK: "),
                "net": tkinter.Label(root, text="NET: "),
                "ui": tkinter.Label(root, text="UI: "),
                "constraints": tkinter.Label(root, text="Constraints: "),
                "actions": tkinter.Label(root, text="Actions: "),
            }
            for lbl in labels.values():
                lbl.pack(anchor="w")

            def update():
                t = self._telemetry
                c = self._constraints
                a = self._actions
                labels["os"]["text"] = f"OS: {t.get('os', '?')}"
                labels["cpu"]["text"] = f"CPU: {t.get('cpu_percent', 0.0):.1f}%"
                labels["mem"]["text"] = f"MEM: {t.get('mem_percent', 0.0):.1f}%"
                labels["disk"]["text"] = f"DISK: {t.get('disk_percent', 0.0):.1f}%"
                labels["net"]["text"] = f"NET: sent={t.get('net_bytes_sent', 0)} recv={t.get('net_bytes_recv', 0)}"
                labels["ui"]["text"] = f"UI: {t.get('ui_state', {})}"
                labels["constraints"]["text"] = f"Constraints: {c}"
                labels["actions"]["text"] = "Actions: " + "; ".join(a[-5:])
                root.after(self.refresh_ms, update)

            root.after(self.refresh_ms, update)
            root.mainloop()

        t = _threading_std.Thread(target=run_gui, daemon=True)
        t.start()
        self._thread_started = True

    def tick(self, telemetry: Dict[str, Any], constraints: Dict[str, Any], actions: List[str]) -> None:
        if not self.enabled:
            return
        self._telemetry = telemetry
        self._constraints = constraints
        self._actions = actions[:]

# ============================================================
# WRAPPERS
# ============================================================

class TelemetryOrganWrapper:
    def __init__(self, adapter: OSAdapter):
        self.adapter = adapter

    def read(self) -> Dict[str, Any]:
        try:
            return self.adapter.telemetry()
        except Exception:
            return {"error": "telemetry_exception", "traceback": traceback.format_exc()}

class ConstraintShifterOrganWrapper:
    def __init__(self, adapter: OSAdapter):
        self.adapter = adapter

    def apply(self, constraints: Dict[str, Any]) -> List[str]:
        try:
            return self.adapter.apply_constraints(constraints)
        except Exception:
            print("[NSF] Constraint application failed:")
            traceback.print_exc()
            return ["ConstraintShifter: apply failed"]

# ============================================================
# MAIN DAEMON
# ============================================================

class NegativeSpaceFlowDaemon:
    def __init__(self, poll_interval: float, memory: PersistentMemory):
        self.libs = LIBS
        self.adapter = get_adapter(self.libs)
        self.telemetry_organ = TelemetryOrganWrapper(self.adapter)
        self.engine = NegativeSpaceEngine(memory)
        self.constraint_shifter = ConstraintShifterOrganWrapper(self.adapter)
        self.poll_interval = poll_interval
        self.running = True
        self.memory = memory

        swarm_cfg = CONFIG["swarm"]
        gui_cfg = CONFIG["gui"]

        self.organs: List[Organ] = [
            GPUOrgan(),
            ThermalOrgan(),
            DiskOrgan(),
            NetworkOrgan(),
            ProcessOrgan(self.adapter),
            SwarmOrgan(enabled=swarm_cfg["enabled"], host=swarm_cfg["host"], port=swarm_cfg["port"]),
            CockpitOrgan(),
            GUICockpitOrgan(enabled=gui_cfg["enabled"], refresh_ms=gui_cfg["refresh_ms"]),
        ]

    def tick(self) -> None:
        telemetry = self.telemetry_organ.read()
        constraints = self.engine.compute_constraints(telemetry)
        actions = self.constraint_shifter.apply(constraints)

        for organ in self.organs:
            try:
                organ.tick(telemetry, constraints, actions)
            except Exception:
                print("[NSF] Organ tick failed:")
                traceback.print_exc()

    def run(self) -> None:
        print("[NSF] Negative-Space Flow Organism starting...")
        self.memory.increment_runs()
        try:
            while self.running:
                self.tick()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("[NSF] Stopping organism (KeyboardInterrupt).")
        except Exception:
            print("[NSF] Fatal error in main loop:")
            traceback.print_exc()
        finally:
            print("[NSF] Saving memory and exiting.")
            self.memory.save()
            print("[NSF] Organism exited.")

# ============================================================
# ENTRY POINT
# ============================================================

def main():
    if psutil is None:
        print("psutil is required. Install with: pip install psutil")
        sys.exit(1)

    memory = PersistentMemory(CONFIG["memory_file"])
    daemon = NegativeSpaceFlowDaemon(poll_interval=CONFIG["poll_interval"], memory=memory)
    daemon.run()

if __name__ == "__main__":
    main()

