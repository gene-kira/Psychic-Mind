"""
blacknight_spine_upgraded.py – Black Night spine with advanced organs and pluggable NetworkTelemetrySource

Upgrades:
1. HealthOrgan – continuous health scoring (deps, heartbeats, queues).
2. ConfidenceEngine – confidence scores for decisions (settings + network).
3. BehaviorMemoryOrgan – per-process behavior fingerprints.
4. Enhanced CockpitUI – tabs, colors, details pane, counters.
5. PolicySimulator – what-if impact of rules.
6. SafeModeController – fallback mode when errors/floods occur.
7. HotReloadOrgan – live config reload (JSON file).
8. ModuleRegistry – organs register name/version/status/heartbeat.
9. ContextEngine – enriches events with context.
10. FutureOrgan – local predictive hints about rules/behavior.

Network telemetry remains pluggable via NetworkTelemetrySource.
"""

from __future__ import annotations

import sys
import os
import time
import threading
import platform
import importlib
import logging
import json
import base64
import ctypes
import ctypes.wintypes
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Iterable, Tuple, Callable, Protocol

from importlib.metadata import version as pkg_version, PackageNotFoundError
from packaging.version import Version

# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("blacknight")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ============================================================================
# Dependency autoloader
# ============================================================================

@dataclass
class DependencySpec:
    pkg_name: str
    module_name: str
    min_version: Optional[str] = None
    required: bool = True
    feature: str = "core"
    description: str = ""


DEPENDENCIES: Dict[str, DependencySpec] = {
    "pywin32_win32api": DependencySpec(
        pkg_name="pywin32",
        module_name="win32api",
        feature="core",
        description="Win32 API access",
    ),
    "pywin32_win32con": DependencySpec(
        pkg_name="pywin32",
        module_name="win32con",
        feature="core",
        description="Win32 constants",
    ),
    "pywin32_win32security": DependencySpec(
        pkg_name="pywin32",
        module_name="win32security",
        feature="core",
        description="Security descriptors, ACLs, tokens",
    ),
    "pywin32_win32evtlog": DependencySpec(
        pkg_name="pywin32",
        module_name="win32evtlog",
        feature="core",
        description="Windows Event Log access",
    ),
    "pywin32_win32gui": DependencySpec(
        pkg_name="pywin32",
        module_name="win32gui",
        feature="core",
        description="Window handles / foreground window",
    ),
    "pywin32_win32process": DependencySpec(
        pkg_name="pywin32",
        module_name="win32process",
        feature="core",
        description="Process IDs from window handles",
    ),
    "cryptography": DependencySpec(
        pkg_name="cryptography",
        module_name="cryptography",
        min_version="41.0.0",
        feature="core",
        description="Crypto primitives",
    ),
    "psutil": DependencySpec(
        pkg_name="psutil",
        module_name="psutil",
        min_version="5.9.0",
        feature="core",
        description="Process/system telemetry",
    ),
    "uiautomation": DependencySpec(
        pkg_name="uiautomation",
        module_name="uiautomation",
        feature="automation",
        description="Windows UI Automation (optional)",
        required=False,
    ),
    "PyQt6": DependencySpec(
        pkg_name="PyQt6",
        module_name="PyQt6",
        min_version="6.5.0",
        required=False,
        feature="ui",
        description="Qt-based cockpit UI",
    ),
    "etw": DependencySpec(
        pkg_name="etw",
        module_name="etw",
        required=False,
        feature="etw",
        description="ETW tracing for registry/file events",
    ),
}


class DepsNamespace:
    pass


class DependencyError(RuntimeError):
    pass


def _ensure_windows() -> None:
    if platform.system().lower() != "windows":
        raise DependencyError(
            f"Black Night is designed for Windows. Detected: {platform.system()}"
        )


def _compare_versions(installed: str, required: str) -> bool:
    return Version(installed) >= Version(required)


def _filter_deps_by_features(features: Optional[Iterable[str]]) -> Dict[str, DependencySpec]:
    if not features:
        return DEPENDENCIES
    fs = set(features)
    return {k: v for k, v in DEPENDENCIES.items() if v.feature in fs}


def load_dependencies(
    strict: bool = True,
    features: Optional[Iterable[str]] = None,
) -> DepsNamespace:
    _ensure_windows()

    ns = DepsNamespace()
    errors: List[str] = []
    loaded_required = 0
    total_required = 0

    selected = _filter_deps_by_features(features)

    for key, spec in selected.items():
        if spec.required:
            total_required += 1

        try:
            if spec.min_version:
                try:
                    installed_ver = pkg_version(spec.pkg_name)
                except PackageNotFoundError:
                    raise DependencyError(
                        f"Package '{spec.pkg_name}' not installed "
                        f"(required >= {spec.min_version})"
                    )
                if not _compare_versions(installed_ver, spec.min_version):
                    raise DependencyError(
                        f"Package '{spec.pkg_name}' version {installed_ver} "
                        f"is below required {spec.min_version}"
                    )

            module = importlib.import_module(spec.module_name)
            attr_name = spec.module_name.split(".")[0]
            setattr(ns, attr_name, module)

            if spec.required:
                loaded_required += 1

            logger.info(
                "Loaded dependency '%s' (%s) [feature=%s]",
                spec.pkg_name,
                spec.module_name,
                spec.feature,
            )

        except Exception as exc:
            msg = f"Problem with '{spec.pkg_name}' ({spec.module_name}) [feature={spec.feature}]: {exc}"
            if spec.required:
                errors.append(msg)
                logger.error(msg)
            else:
                logger.warning(msg)

    if errors and strict:
        full = "\n".join(errors)
        raise DependencyError(
            "Required dependencies missing/invalid:\n"
            f"{full}\n\nFix environment before running Black Night."
        )

    ns._readiness = _compute_readiness(loaded_required, total_required)
    ns._loaded_required = loaded_required
    ns._total_required = total_required

    logger.info(
        "Environment readiness: %.2f (loaded %d / %d required)",
        ns._readiness,
        loaded_required,
        total_required,
    )

    return ns


def _compute_readiness(loaded_required: int, total_required: int) -> float:
    if total_required == 0:
        return 1.0
    return loaded_required / total_required


def get_deps(
    strict: bool = True,
    features: Optional[Iterable[str]] = None,
) -> DepsNamespace:
    return load_dependencies(strict=strict, features=features)


deps = get_deps()

win32api = deps.win32api
win32con = deps.win32con
win32security = deps.win32security
win32evtlog = deps.win32evtlog
win32gui = deps.win32gui
win32process = deps.win32process
psutil = deps.psutil
uia = getattr(deps, "uiautomation", None)
Qt = getattr(deps, "PyQt6", None)
etw_mod = getattr(deps, "etw", None)

# ============================================================================
# Helpers
# ============================================================================

def get_foreground_pid() -> Optional[int]:
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return pid
    except Exception:
        return None


def classify_process(pid: Optional[int]) -> str:
    if pid is None:
        return "unknown"
    try:
        if pid == 4:
            return "system"
        p = psutil.Process(pid)
        username = (p.username() or "").lower()
        if "system" in username:
            return "system"
        return "user"
    except Exception:
        return "unknown"


def get_process_name(pid: Optional[int]) -> Optional[str]:
    if pid is None:
        return None
    try:
        p = psutil.Process(pid)
        return p.name()
    except Exception:
        return None


def get_process_ancestry(pid: Optional[int]) -> List[int]:
    if pid is None:
        return []
    try:
        p = psutil.Process(pid)
        chain = []
        while True:
            chain.append(p.pid)
            if p.ppid() == 0 or p.ppid() == p.pid:
                break
            p = p.parent()
            if p is None:
                break
        return chain
    except Exception:
        return []

# ============================================================================
# Event bus
# ============================================================================

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Any]] = {}

    def subscribe(self, event_type: str, handler: Any):
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event_type: str, payload: Dict[str, Any]):
        for handler in self._subscribers.get(event_type, []):
            try:
                handler(payload)
            except Exception as exc:
                logger.error("Event handler error for %s: %s", event_type, exc)

# ============================================================================
# Module Registry
# ============================================================================

@dataclass
class ModuleInfo:
    name: str
    version: str
    status: str
    last_heartbeat: float
    error_count: int = 0


class ModuleRegistry:
    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self._lock = threading.Lock()

    def register(self, name: str, version: str):
        with self._lock:
            self.modules[name] = ModuleInfo(
                name=name,
                version=version,
                status="initializing",
                last_heartbeat=time.time(),
            )

    def heartbeat(self, name: str, status: str = "ok"):
        with self._lock:
            if name not in self.modules:
                self.register(name, "1.0")
            self.modules[name].status = status
            self.modules[name].last_heartbeat = time.time()

    def error(self, name: str):
        with self._lock:
            if name not in self.modules:
                self.register(name, "1.0")
            self.modules[name].error_count += 1
            self.modules[name].status = "error"

    def snapshot(self) -> List[ModuleInfo]:
        with self._lock:
            return list(self.modules.values())

# ============================================================================
# UIAutomation Organ
# ============================================================================

class UIAutomationOrgan:
    VERSION = "1.1"

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("UIAutomationOrgan", self.VERSION)
        self.last_input_time = time.time()
        self.signal = "unknown"
        self._stop = False

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            try:
                self._poll()
                self.registry.heartbeat("UIAutomationOrgan")
            except Exception as exc:
                logger.error("UIAutomation poll error: %s", exc)
                self.registry.error("UIAutomationOrgan")
            time.sleep(0.1)

    def _poll(self):
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ('cbSize', ctypes.wintypes.UINT),
                ('dwTime', ctypes.wintypes.DWORD),
            ]

        last_input = LASTINPUTINFO()
        last_input.cbSize = ctypes.sizeof(LASTINPUTINFO)

        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(last_input)):
            millis = win32api.GetTickCount() - last_input.dwTime
            seconds = millis / 1000.0
            if seconds < 0.5:
                self.last_input_time = time.time()

        if time.time() - self.last_input_time < 0.5:
            self.signal = "user_intent_confirmed"
        else:
            self.signal = "background_change"

# ============================================================================
# Policy Organ + Safe Mode
# ============================================================================

class SafeModeController:
    def __init__(self):
        self.safe_mode = False
        self.reason: Optional[str] = None

    def enter_safe_mode(self, reason: str):
        if not self.safe_mode:
            logger.warning("Entering SAFE MODE: %s", reason)
        self.safe_mode = True
        self.reason = reason

    def exit_safe_mode(self):
        if self.safe_mode:
            logger.info("Exiting SAFE MODE")
        self.safe_mode = False
        self.reason = None


class PolicyOrgan:
    VERSION = "1.2"

    def __init__(self, ui_organ: UIAutomationOrgan, safe_mode: SafeModeController, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("PolicyOrgan", self.VERSION)
        self.ui = ui_organ
        self.safe_mode = safe_mode
        self.mode = "degraded"
        self.maintenance_until: Optional[float] = None
        self.frozen: bool = False

    def start_maintenance(self, duration_seconds: int):
        self.maintenance_until = time.time() + duration_seconds
        logger.info("Maintenance window started for %d seconds", duration_seconds)

    def set_maintenance_until(self, ts: Optional[float]):
        self.maintenance_until = ts

    def set_frozen(self, frozen: bool = True):
        self.frozen = frozen
        if frozen:
            logger.warning("Policy entered FROZEN state due to tamper.")
        else:
            logger.info("Policy exited FROZEN state.")

    def _in_maintenance(self) -> bool:
        if self.maintenance_until is None:
            return False
        if time.time() <= self.maintenance_until:
            return True
        self.maintenance_until = None
        return False

    def evaluate(self) -> str:
        try:
            if self.safe_mode.safe_mode:
                self.mode = "safe_mode"
                return self.mode

            if self.frozen:
                self.mode = "frozen"
                return self.mode

            if self._in_maintenance():
                self.mode = "maintenance"
                return self.mode

            readiness = getattr(deps, "_readiness", 0.0)
            ui_signal = self.ui.signal

            if readiness >= 0.95 and ui_signal == "user_intent_confirmed":
                self.mode = "full_enforcement"
            else:
                self.mode = "degraded"

            self.registry.heartbeat("PolicyOrgan")
            return self.mode
        except Exception as exc:
            logger.error("Policy evaluation error: %s", exc)
            self.registry.error("PolicyOrgan")
            self.safe_mode.enter_safe_mode("Policy evaluation failure")
            self.mode = "safe_mode"
            return self.mode

# ============================================================================
# Settings Organ
# ============================================================================

@dataclass
class WatchedRegistryValue:
    hive: Any
    path: str
    name: str
    last_value: Any = None
    last_type: Optional[int] = None


@dataclass
class WatchedFile:
    path: str
    last_mtime: float = 0.0
    last_content: Optional[bytes] = None


class SettingsOrgan:
    VERSION = "1.2"

    def __init__(self, bus: EventBus, registry: ModuleRegistry):
        self.bus = bus
        self.registry = registry
        self.registry.register("SettingsOrgan", self.VERSION)
        self.registry_values: List[WatchedRegistryValue] = []
        self.files: List[WatchedFile] = []
        self._stop = False

    def add_registry_watch(self, hive, path: str, name: str):
        self.registry_values.append(WatchedRegistryValue(hive=hive, path=path, name=name))

    def add_file_watch(self, path: str):
        self.files.append(WatchedFile(path=path))

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            try:
                self._poll_registry()
                self._poll_files()
                self.registry.heartbeat("SettingsOrgan")
            except Exception as exc:
                logger.error("Settings poll error: %s", exc)
                self.registry.error("SettingsOrgan")
            time.sleep(0.5)

    def _poll_registry(self):
        pid = get_foreground_pid()
        proc_class = classify_process(pid)
        for item in self.registry_values:
            try:
                key = win32api.RegOpenKeyEx(item.hive, item.path, 0, win32con.KEY_READ)
                value, vtype = win32api.RegQueryValueEx(key, item.name)
                win32api.RegCloseKey(key)
            except Exception:
                value, vtype = None, None

            if item.last_value is None and item.last_type is None:
                item.last_value = value
                item.last_type = vtype
                continue

            if value != item.last_value or vtype != item.last_type:
                old_val = item.last_value
                old_type = item.last_type
                item.last_value = value
                item.last_type = vtype
                self.bus.publish("setting_changed", {
                    "kind": "registry",
                    "path": f"{item.path}\\{item.name}",
                    "name": item.name,
                    "old": (old_val, old_type),
                    "new": (value, vtype),
                    "pid": pid,
                    "proc_class": proc_class,
                    "hive": item.hive,
                    "source": "poll",
                })

    def _poll_files(self):
        pid = get_foreground_pid()
        proc_class = classify_process(pid)
        for item in self.files:
            try:
                mtime = os.path.getmtime(item.path)
            except FileNotFoundError:
                mtime = 0.0

            if item.last_mtime == 0.0:
                item.last_mtime = mtime
                try:
                    with open(item.path, "rb") as f:
                        item.last_content = f.read()
                except Exception:
                    item.last_content = None
                continue

            if mtime != item.last_mtime:
                old_content = item.last_content
                try:
                    with open(item.path, "rb") as f:
                        new_content = f.read()
                except Exception:
                    new_content = None

                item.last_mtime = mtime
                item.last_content = new_content

                self.bus.publish("setting_changed", {
                    "kind": "file",
                    "path": item.path,
                    "name": None,
                    "old": old_content,
                    "new": new_content,
                    "pid": pid,
                    "proc_class": proc_class,
                    "hive": None,
                    "source": "poll",
                })

# ============================================================================
# ETW Organ (optional)
# ============================================================================

class ETWOrgan:
    VERSION = "1.0"
    REGISTRY_PROVIDER = "{70EB4F03-C1DE-4F73-A051-33D13D5413BD}"

    def __init__(self, bus: EventBus, registry: ModuleRegistry):
        self.bus = bus
        self.registry = registry
        self.registry.register("ETWOrgan", self.VERSION)
        self.etw = etw_mod
        self._stop = False

    def start(self):
        if self.etw is None:
            logger.warning("ETW module not available; ETWOrgan is inactive.")
            return
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop = True

    def _loop(self):
        from etw import ETW

        def callback(event):
            try:
                hdr = event["EventHeader"]
                pid = hdr["ProcessId"]
                proc_class = classify_process(pid)
                data = event.get("EventData", {})
                key_name = data.get("KeyName") or data.get("KeyPath")
                if not key_name:
                    return

                self.bus.publish("setting_changed", {
                    "kind": "registry",
                    "path": key_name,
                    "name": None,
                    "old": None,
                    "new": None,
                    "pid": pid,
                    "proc_class": proc_class,
                    "hive": None,
                    "source": "etw",
                })
                self.registry.heartbeat("ETWOrgan")
            except Exception as exc:
                logger.error("ETW callback error: %s", exc)
                self.registry.error("ETWOrgan")

        trace = ETW(
            providers=[{
                "guid": self.REGISTRY_PROVIDER,
                "any": 0xFFFFFFFF,
                "all": 0,
            }],
            event_callback=callback,
        )

        logger.info("Starting ETW registry trace...")
        try:
            trace.start()
        except Exception as exc:
            logger.error("ETW trace error: %s", exc)
            self.registry.error("ETWOrgan")

# ============================================================================
# Per-path policy rules
# ============================================================================

@dataclass
class PolicyRule:
    path_prefix: str
    kind: str
    action: str   # "allow", "block", "monitor"
    severity: str # "low", "medium", "high", "critical"


DEFAULT_POLICY_RULES: List[PolicyRule] = [
    PolicyRule(
        path_prefix=r"Software\Microsoft\Windows\CurrentVersion\Run",
        kind="registry",
        action="block",
        severity="critical",
    ),
    PolicyRule(
        path_prefix=r"Software\Microsoft\Windows\CurrentVersion\Policies",
        kind="registry",
        action="block",
        severity="high",
    ),
    PolicyRule(
        path_prefix=r"C:\Temp\blacknight_test.cfg",
        kind="file",
        action="monitor",
        severity="low",
    ),
]

# ============================================================================
# Dynamic Policy
# ============================================================================

@dataclass
class DynamicRule:
    kind: str          # "network", "firewall", or "setting"
    proc_class: str    # "system", "user", "unknown"
    process_name: Optional[str]
    decision: str      # "allow" or "block"
    source: str        # "operator" / "auto_reputation"
    timestamp: float


class DynamicPolicy:
    VERSION = "1.1"

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("DynamicPolicy", self.VERSION)
        self.rules: List[DynamicRule] = []

    def add_rule(self, rule: DynamicRule):
        logger.info(
            "DynamicPolicy: %s %s/%s for kind=%s (source=%s)",
            rule.decision.upper(),
            rule.proc_class,
            rule.process_name or "*",
            rule.kind,
            rule.source,
        )
        self.rules.append(rule)
        self.registry.heartbeat("DynamicPolicy")

    def decide_for_process(self, kind: str, pid: Optional[int], proc_class: str) -> Optional[str]:
        pname = get_process_name(pid)
        for r in reversed(self.rules):
            if r.kind != kind:
                continue
            if r.proc_class != proc_class:
                continue
            if r.process_name and pname and r.process_name.lower() == pname.lower():
                return r.decision
        for r in reversed(self.rules):
            if r.kind != kind:
                continue
            if r.proc_class != proc_class:
                continue
            if r.process_name is None:
                return r.decision
        return None

# ============================================================================
# Confidence Engine
# ============================================================================

class ConfidenceEngine:
    VERSION = "1.0"

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("ConfidenceEngine", self.VERSION)

    def score_setting_decision(self, change: Dict[str, Any], decision: Dict[str, Any]) -> float:
        score = 0.5
        rule = decision.get("rule")
        mode = decision.get("mode", "degraded")
        ui_signal = decision.get("ui_signal", "unknown")

        if rule:
            if rule.severity in ("high", "critical"):
                score += 0.2
            if rule.action == "block":
                score += 0.1
        if mode == "full_enforcement":
            score += 0.1
        if ui_signal == "user_intent_confirmed":
            score += 0.1

        score = max(0.0, min(1.0, score))
        self.registry.heartbeat("ConfidenceEngine")
        return score

    def score_network_decision(self, pending: "PendingDecision") -> float:
        score = 0.5
        if pending.proc_class == "system":
            score += 0.1
        if pending.verdict_source == "auto_reputation":
            score += 0.2
        if pending.severity == "high":
            score += 0.1
        score = max(0.0, min(1.0, score))
        self.registry.heartbeat("ConfidenceEngine")
        return score

# ============================================================================
# Rollback Organ
# ============================================================================

class RollbackOrgan:
    VERSION = "1.1"

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("RollbackOrgan", self.VERSION)
        self.registry_baseline: Dict[str, Tuple[Any, Optional[int], Any]] = {}
        self.file_baseline: Dict[str, bytes] = {}

    def capture_baseline_registry(self, item: WatchedRegistryValue):
        try:
            key = win32api.RegOpenKeyEx(item.hive, item.path, 0, win32con.KEY_READ)
            value, vtype = win32api.RegQueryValueEx(key, item.name)
            win32api.RegCloseKey(key)
        except Exception:
            value, vtype = None, None
        self.registry_baseline[f"{item.path}\\{item.name}"] = (value, vtype, item.hive)
        self.registry.heartbeat("RollbackOrgan")

    def capture_baseline_file(self, path: str):
        try:
            with open(path, "rb") as f:
                self.file_baseline[path] = f.read()
        except Exception:
            self.file_baseline[path] = b""
        self.registry.heartbeat("RollbackOrgan")

    def rollback(self, change: Dict[str, Any]):
        kind = change["kind"]
        path = change["path"]

        if kind == "registry":
            baseline = self.registry_baseline.get(path, None)
            if baseline is None:
                return
            value, vtype, hive = baseline
            reg_path, name = path.rsplit("\\", 1)
            hive = hive or win32con.HKEY_CURRENT_USER
            try:
                key = win32api.RegOpenKeyEx(hive, reg_path, 0, win32con.KEY_SET_VALUE)
                win32api.RegSetValueEx(key, name, 0, vtype if vtype is not None else win32con.REG_SZ, value)
                win32api.RegCloseKey(key)
                logger.info("Rolled back registry %s", path)
                self.registry.heartbeat("RollbackOrgan")
            except Exception as exc:
                logger.error("Registry rollback error for %s: %s", path, exc)
                self.registry.error("RollbackOrgan")

        elif kind == "file":
            baseline = self.file_baseline.get(path, None)
            if baseline is None:
                return
            try:
                with open(path, "wb") as f:
                    f.write(baseline)
                logger.info("Rolled back file %s", path)
                self.registry.heartbeat("RollbackOrgan")
            except Exception as exc:
                logger.error("File rollback error for %s: %s", path, exc)
                self.registry.error("RollbackOrgan")

# ============================================================================
# RebootMemoryOrgan
# ============================================================================

class RebootMemoryOrgan:
    VERSION = "1.2"

    def __init__(self, registry: ModuleRegistry, path: str = None):
        self.registry = registry
        self.registry.register("RebootMemoryOrgan", self.VERSION)
        if path is None:
            base = os.path.join(os.path.expanduser("~"), "BlackNight")
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, "blacknight_state.json")
        self.path = path
        self._lock = threading.Lock()

        self.dynamic_policy: Optional[DynamicPolicy] = None
        self.rollback: Optional[RollbackOrgan] = None
        self.pending: Optional["PendingDecisionOrgan"] = None
        self.policy: Optional[PolicyOrgan] = None

    def _load_raw(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load state file: %s", exc)
            self.registry.error("RebootMemoryOrgan")
            return None

    def _save_raw(self, data: Dict[str, Any]):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)
            self.registry.heartbeat("RebootMemoryOrgan")
        except Exception as exc:
            logger.error("Failed to save state file: %s", exc)
            self.registry.error("RebootMemoryOrgan")

    def attach(self,
               dynamic_policy: DynamicPolicy,
               rollback: RollbackOrgan,
               pending: "PendingDecisionOrgan",
               policy: PolicyOrgan):
        self.dynamic_policy = dynamic_policy
        self.rollback = rollback
        self.pending = pending
        self.policy = policy

    def load_state(self):
        with self._lock:
            data = self._load_raw()
        if not data:
            logger.info("No previous state found; starting fresh.")
            return

        logger.info("Loading previous state from %s", self.path)

        if self.dynamic_policy:
            self.dynamic_policy.rules.clear()
            for r in data.get("dynamic_policy", []):
                try:
                    self.dynamic_policy.rules.append(DynamicRule(
                        kind=r["kind"],
                        proc_class=r["proc_class"],
                        process_name=r.get("process_name"),
                        decision=r["decision"],
                        source=r.get("source", "operator"),
                        timestamp=r.get("timestamp", time.time()),
                    ))
                except KeyError:
                    continue

        if self.rollback:
            self.rollback.registry_baseline.clear()
            for k, v in data.get("registry_baseline", {}).items():
                value = v.get("value")
                vtype = v.get("type")
                hive_name = v.get("hive", "HKEY_CURRENT_USER")
                hive = getattr(win32con, hive_name, win32con.HKEY_CURRENT_USER)
                self.rollback.registry_baseline[k] = (value, vtype, hive)

            self.rollback.file_baseline.clear()
            for path, b64 in data.get("file_baseline", {}).items():
                try:
                    self.rollback.file_baseline[path] = base64.b64decode(b64.encode("ascii"))
                except Exception:
                    self.rollback.file_baseline[path] = b""

        if self.pending:
            self.pending.pending.clear()
            for p in data.get("pending_decisions", []):
                try:
                    self.pending.pending.append(PendingDecision(
                        timestamp=p["timestamp"],
                        kind=p["kind"],
                        description=p["description"],
                        pid=p.get("pid"),
                        proc_class=p.get("proc_class", "unknown"),
                        process_name=p.get("process_name"),
                        dest_host=p.get("dest_host"),
                        dest_ip=p.get("dest_ip"),
                        severity=p.get("severity", "medium"),
                        decided=p.get("decided", False),
                        decision=p.get("decision"),
                        verdict_source=p.get("verdict_source"),
                        hint=p.get("hint"),
                    ))
                except KeyError:
                    continue

        if self.policy:
            self.policy.set_frozen(bool(data.get("tamper_frozen", False)))
            maint = data.get("maintenance_until", None)
            self.policy.set_maintenance_until(maint)

    def save_state(self):
        if not (self.dynamic_policy and self.rollback and self.pending and self.policy):
            return

        data: Dict[str, Any] = {
            "version": 2,
            "dynamic_policy": [],
            "registry_baseline": {},
            "file_baseline": {},
            "pending_decisions": [],
            "tamper_frozen": self.policy.frozen,
            "maintenance_until": self.policy.maintenance_until,
        }

        for r in self.dynamic_policy.rules:
            data["dynamic_policy"].append({
                "kind": r.kind,
                "proc_class": r.proc_class,
                "process_name": r.process_name,
                "decision": r.decision,
                "source": r.source,
                "timestamp": r.timestamp,
            })

        for path, (value, vtype, hive) in self.rollback.registry_baseline.items():
            hive_name = None
            for name in dir(win32con):
                if name.startswith("HKEY_") and getattr(win32con, name) == hive:
                    hive_name = name
                    break
            if hive_name is None:
                hive_name = "HKEY_CURRENT_USER"
            data["registry_baseline"][path] = {
                "value": value,
                "type": vtype,
                "hive": hive_name,
            }

        for path, content in self.rollback.file_baseline.items():
            try:
                b64 = base64.b64encode(content).decode("ascii")
            except Exception:
                b64 = ""
            data["file_baseline"][path] = b64

        for p in self.pending.pending:
            data["pending_decisions"].append({
                "timestamp": p.timestamp,
                "kind": p.kind,
                "description": p.description,
                "pid": p.pid,
                "proc_class": p.proc_class,
                "process_name": p.process_name,
                "dest_host": p.dest_host,
                "dest_ip": p.dest_ip,
                "severity": p.severity,
                "decided": p.decided,
                "decision": p.decision,
                "verdict_source": p.verdict_source,
                "hint": p.hint,
            })

        with self._lock:
            self._save_raw(data)

    def save_state_async(self):
        t = threading.Thread(target=self.save_state, daemon=True)
        t.start()

# ============================================================================
# Tamper Organ
# ============================================================================

class TamperOrgan:
    VERSION = "1.0"

    def __init__(self, policy: PolicyOrgan, bus: EventBus, self_paths: List[str],
                 memory: RebootMemoryOrgan, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("TamperOrgan", self.VERSION)
        self.policy = policy
        self.bus = bus
        self.self_paths = [p.lower() for p in self_paths]
        self.memory = memory

    def on_setting_changed(self, change: Dict[str, Any]):
        path = (change.get("path") or "").lower()
        for sp in self.self_paths:
            if path.startswith(sp):
                logger.warning("Tamper detected on protected path %s; freezing policy.", path)
                self.policy.set_frozen(True)
                self.memory.save_state_async()
                self.registry.heartbeat("TamperOrgan")
                break

# ============================================================================
# Threat Organ
# ============================================================================

@dataclass
class ThreatRecord:
    timestamp: float
    kind: str
    path: str
    allowed: bool
    mode: str
    ui_signal: str
    pid: Optional[int]
    proc_class: str
    rule_action: Optional[str]
    rule_severity: Optional[str]
    maintenance_allowed: bool
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


class ThreatOrgan:
    VERSION = "1.1"

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("ThreatOrgan", self.VERSION)
        self.records: List[ThreatRecord] = []

    def log_change(self, change: Dict[str, Any], decision: Dict[str, Any], confidence: float):
        rule: Optional[PolicyRule] = decision.get("rule")
        rec = ThreatRecord(
            timestamp=time.time(),
            kind=change["kind"],
            path=change["path"],
            allowed=decision["allowed"],
            mode=decision["mode"],
            ui_signal=decision["ui_signal"],
            pid=change.get("pid"),
            proc_class=change.get("proc_class", "unknown"),
            rule_action=rule.action if rule else None,
            rule_severity=rule.severity if rule else None,
            maintenance_allowed=decision.get("maintenance_allowed", False),
            confidence=confidence,
            details={
                "old": change.get("old"),
                "new": change.get("new"),
                "source": change.get("source", "poll"),
            },
        )
        self.records.append(rec)
        self.registry.heartbeat("ThreatOrgan")

    def recent(self, limit: int = 50) -> List[ThreatRecord]:
        return self.records[-limit:]

# ============================================================================
# ReputationOrgan
# ============================================================================

class ReputationOrgan:
    VERSION = "1.0"

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("ReputationOrgan", self.VERSION)
        self.trusted_dests: Dict[str, Dict[str, Any]] = {
            "MsMpEng.exe": {
                "hosts": {"wdcp.microsoft.com", "defender.microsoft.com", "update.microsoft.com"},
            },
            "chrome.exe": {
                "hosts": {"www.google.com", "accounts.google.com"},
            },
        }

    def evaluate_connection(self, process_name: Optional[str], dest_host: str, dest_ip: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        pname = process_name or ""
        trusted = self.trusted_dests.get(pname, None)
        self.registry.heartbeat("ReputationOrgan")
        if trusted:
            if dest_host in trusted["hosts"]:
                return None, None, f"{pname} talking to known host {dest_host}"
            else:
                return "block", "auto_reputation", f"{pname} talking to unexpected host {dest_host}"
        else:
            return None, None, f"Unknown process {pname or 'unknown'} to {dest_host}"

# ============================================================================
# Behavior Memory Organ
# ============================================================================

@dataclass
class BehaviorProfile:
    process_name: str
    total_events: int = 0
    blocked_events: int = 0
    allowed_events: int = 0
    unique_hosts: set = field(default_factory=set)


class BehaviorMemoryOrgan:
    VERSION = "1.0"

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("BehaviorMemoryOrgan", self.VERSION)
        self.profiles: Dict[str, BehaviorProfile] = {}

    def record_network_event(self, pending: "PendingDecision"):
        pname = pending.process_name or "unknown"
        profile = self.profiles.get(pname)
        if not profile:
            profile = BehaviorProfile(process_name=pname)
            self.profiles[pname] = profile
        profile.total_events += 1
        if pending.dest_host:
            profile.unique_hosts.add(pending.dest_host)
        if pending.decided:
            if pending.decision == "block":
                profile.blocked_events += 1
            elif pending.decision == "allow":
                profile.allowed_events += 1
        self.registry.heartbeat("BehaviorMemoryOrgan")

    def get_profile(self, process_name: str) -> Optional[BehaviorProfile]:
        return self.profiles.get(process_name)

# ============================================================================
# PendingDecision + Organs
# ============================================================================

@dataclass
class PendingDecision:
    timestamp: float
    kind: str          # "network" or "firewall"
    description: str
    pid: Optional[int]
    proc_class: str
    process_name: Optional[str]
    dest_host: Optional[str]
    dest_ip: Optional[str]
    severity: str
    decided: bool = False
    decision: Optional[str] = None  # "allow" / "block"
    verdict_source: Optional[str] = None  # "operator" / "auto_reputation"
    hint: Optional[str] = None
    confidence: float = 0.0


class PendingDecisionOrgan:
    VERSION = "1.1"

    def __init__(self, memory: RebootMemoryOrgan = None, registry: ModuleRegistry = None):
        self.registry = registry
        if self.registry:
            self.registry.register("PendingDecisionOrgan", self.VERSION)
        self.pending: List[PendingDecision] = []
        self.memory = memory

    def add(self, decision: PendingDecision):
        self.pending.append(decision)
        if self.memory:
            self.memory.save_state_async()
        if self.registry:
            self.registry.heartbeat("PendingDecisionOrgan")

    def list_pending(self, include_decided: bool = False) -> List[PendingDecision]:
        if include_decided:
            return list(self.pending)
        return [p for p in self.pending if not p.decided]

    def decide(self, index: int, decision: str, source: str = "operator"):
        if 0 <= index < len(self.pending):
            self.pending[index].decided = True
            self.pending[index].decision = decision
            self.pending[index].verdict_source = source
            logger.info("Pending decision %d marked as %s (%s)", index, decision, source)
            if self.memory:
                self.memory.save_state_async()
            if self.registry:
                self.registry.heartbeat("PendingDecisionOrgan")

    def index_of(self, obj: PendingDecision) -> int:
        try:
            return self.pending.index(obj)
        except ValueError:
            return -1

# ============================================================================
# NetworkTelemetrySource interface
# ============================================================================

class NetworkTelemetryEvent:
    def __init__(self,
                 pid: Optional[int],
                 dest_host: Optional[str],
                 dest_ip: Optional[str],
                 description: str = "Outbound connection"):
        self.pid = pid
        self.dest_host = dest_host or ""
        self.dest_ip = dest_ip or ""
        self.description = description


class NetworkTelemetrySource(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def on_event(self, callback: Callable[[NetworkTelemetryEvent], None]) -> None:
        ...


class DummyNetworkTelemetrySource:
    def __init__(self):
        self._callback: Optional[Callable[[NetworkTelemetryEvent], None]] = None
        self._stop = False

    def on_event(self, callback: Callable[[NetworkTelemetryEvent], None]) -> None:
        self._callback = callback

    def start(self) -> None:
        logger.info("DummyNetworkTelemetrySource started (no real telemetry).")
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop = True

    def _loop(self):
        while not self._stop:
            time.sleep(5.0)

# ============================================================================
# Context Engine
# ============================================================================

class ContextEngine:
    VERSION = "1.0"

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("ContextEngine", self.VERSION)

    def enrich_network(self, pending: PendingDecision) -> Dict[str, Any]:
        ancestry = get_process_ancestry(pending.pid)
        ctx = {
            "ancestry": ancestry,
            "user": None,
            "cpu_percent": None,
            "recent_events": None,
        }
        try:
            if pending.pid:
                p = psutil.Process(pending.pid)
                ctx["user"] = p.username()
                ctx["cpu_percent"] = p.cpu_percent(interval=0.0)
        except Exception:
            pass
        self.registry.heartbeat("ContextEngine")
        return ctx

# ============================================================================
# Future Organ (local predictive hints)
# ============================================================================

class FutureOrgan:
    VERSION = "1.0"

    def __init__(self, behavior_memory: BehaviorMemoryOrgan, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("FutureOrgan", self.VERSION)
        self.behavior_memory = behavior_memory

    def hint_for_process(self, process_name: Optional[str]) -> Optional[str]:
        if not process_name:
            return None
        profile = self.behavior_memory.get_profile(process_name)
        if not profile:
            return "No history yet; treat cautiously."
        if profile.blocked_events > profile.allowed_events:
            hint = "Process trending toward BLOCK behavior."
        elif profile.allowed_events > 0 and profile.blocked_events == 0:
            hint = "Process historically safe; low anomaly risk."
        else:
            hint = "Mixed behavior; monitor closely."
        self.registry.heartbeat("FutureOrgan")
        return hint

# ============================================================================
# Network / Firewall Organs
# ============================================================================

class NetworkOrgan:
    VERSION = "1.1"

    def __init__(self,
                 pending: PendingDecisionOrgan,
                 dynamic_policy: DynamicPolicy,
                 reputation: ReputationOrgan,
                 telemetry_source: NetworkTelemetrySource,
                 behavior_memory: BehaviorMemoryOrgan,
                 confidence_engine: ConfidenceEngine,
                 context_engine: ContextEngine,
                 registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("NetworkOrgan", self.VERSION)
        self.pending = pending
        self.dynamic_policy = dynamic_policy
        self.reputation = reputation
        self.telemetry_source = telemetry_source
        self.behavior_memory = behavior_memory
        self.confidence_engine = confidence_engine
        self.context_engine = context_engine

    def start(self):
        def handle_event(ev: NetworkTelemetryEvent):
            try:
                self._handle_event(ev)
                self.registry.heartbeat("NetworkOrgan")
            except Exception as exc:
                logger.error("NetworkOrgan event handling error: %s", exc)
                self.registry.error("NetworkOrgan")

        self.telemetry_source.on_event(handle_event)
        self.telemetry_source.start()

    def stop(self):
        self.telemetry_source.stop()

    def _handle_event(self, ev: NetworkTelemetryEvent):
        pid = ev.pid
        proc_class = classify_process(pid)
        pname = get_process_name(pid)

        dest_host = ev.dest_host
        dest_ip = ev.dest_ip
        desc = ev.description or "Outbound connection"

        auto_decision, verdict_source, hint = self.reputation.evaluate_connection(
            pname, dest_host, dest_ip
        )

        pd = PendingDecision(
            timestamp=time.time(),
            kind="network",
            description=desc,
            pid=pid,
            proc_class=proc_class,
            process_name=pname,
            dest_host=dest_host,
            dest_ip=dest_ip,
            severity="medium",
            decided=False,
            decision=None,
            verdict_source=verdict_source,
            hint=hint,
        )

        ctx = self.context_engine.enrich_network(pd)
        if ctx.get("cpu_percent") is not None:
            pd.hint = (pd.hint or "") + f" | CPU={ctx['cpu_percent']}"

        if auto_decision:
            pd.decided = True
            pd.decision = auto_decision
            pd.confidence = self.confidence_engine.score_network_decision(pd)
            self.pending.add(pd)

            rule = DynamicRule(
                kind="network",
                proc_class=proc_class,
                process_name=pname,
                decision=auto_decision,
                source=verdict_source or "auto_reputation",
                timestamp=time.time(),
            )
            self.dynamic_policy.add_rule(rule)
        else:
            pd.confidence = self.confidence_engine.score_network_decision(pd)
            self.pending.add(pd)

        self.behavior_memory.record_network_event(pd)

class FirewallOrgan:
    VERSION = "0.9"

    def __init__(self, pending: PendingDecisionOrgan, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("FirewallOrgan", self.VERSION)
        self.pending = pending
        self._stop = False

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            time.sleep(15.0)
            try:
                pid = get_foreground_pid()
                proc_class = classify_process(pid)
                pname = get_process_name(pid)
                dest_host = None
                dest_ip = None
                desc = "New firewall rule requested for inbound port 3389"

                pd = PendingDecision(
                    timestamp=time.time(),
                    kind="firewall",
                    description=desc,
                    pid=pid,
                    proc_class=proc_class,
                    process_name=pname,
                    dest_host=dest_host,
                    dest_ip=dest_ip,
                    severity="high",
                    decided=False,
                    decision=None,
                    verdict_source=None,
                    hint="Firewall change – operator review recommended",
                )
                self.pending.add(pd)
                self.registry.heartbeat("FirewallOrgan")
            except Exception as exc:
                logger.error("FirewallOrgan loop error: %s", exc)
                self.registry.error("FirewallOrgan")

# ============================================================================
# Change Validator
# ============================================================================

class ChangeValidator:
    VERSION = "1.1"

    def __init__(self, ui_organ: UIAutomationOrgan, policy_organ: PolicyOrgan,
                 rules: List[PolicyRule], dynamic_policy: DynamicPolicy,
                 confidence_engine: ConfidenceEngine, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("ChangeValidator", self.VERSION)
        self.ui = ui_organ
        self.policy = policy_organ
        self.rules = rules
        self.dynamic_policy = dynamic_policy
        self.confidence_engine = confidence_engine

    def _match_rule(self, change: Dict[str, Any]) -> Optional[PolicyRule]:
        kind = change["kind"]
        path = (change.get("path") or "").lower()
        best_match = None
        best_len = -1
        for rule in self.rules:
            if rule.kind != kind:
                continue
            prefix = rule.path_prefix.lower()
            if path.startswith(prefix) and len(prefix) > best_len:
                best_match = rule
                best_len = len(prefix)
        return best_match

    def validate(self, change: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        mode = self.policy.evaluate()
        ui_signal = self.ui.signal
        rule = self._match_rule(change)

        if mode == "maintenance":
            decision = {
                "allowed": True,
                "mode": mode,
                "ui_signal": ui_signal,
                "rule": rule,
                "maintenance_allowed": True,
            }
            conf = self.confidence_engine.score_setting_decision(change, decision)
            self.registry.heartbeat("ChangeValidator")
            return decision, conf

        if mode == "frozen" or mode == "safe_mode":
            decision = {
                "allowed": False,
                "mode": mode,
                "ui_signal": ui_signal,
                "rule": rule,
                "maintenance_allowed": False,
            }
            conf = self.confidence_engine.score_setting_decision(change, decision)
            self.registry.heartbeat("ChangeValidator")
            return decision, conf

        dyn_decision = self.dynamic_policy.decide_for_process(
            kind="setting",
            pid=change.get("pid"),
            proc_class=change.get("proc_class", "unknown"),
        )
        if dyn_decision is not None:
            allowed = (dyn_decision == "allow")
            decision = {
                "allowed": allowed,
                "mode": mode,
                "ui_signal": ui_signal,
                "rule": rule,
                "maintenance_allowed": False,
            }
            conf = self.confidence_engine.score_setting_decision(change, decision)
            self.registry.heartbeat("ChangeValidator")
            return decision, conf

        if rule:
            if rule.action == "block":
                allowed = False
            elif rule.action == "allow":
                allowed = True
            else:
                allowed = (mode == "full_enforcement" and ui_signal == "user_intent_confirmed")
        else:
            allowed = (mode == "full_enforcement" and ui_signal == "user_intent_confirmed")

        decision = {
            "allowed": allowed,
            "mode": mode,
            "ui_signal": ui_signal,
            "rule": rule,
            "maintenance_allowed": False,
        }
        conf = self.confidence_engine.score_setting_decision(change, decision)
        self.registry.heartbeat("ChangeValidator")
        return decision, conf

# ============================================================================
# Policy Simulator
# ============================================================================

class PolicySimulator:
    VERSION = "1.0"

    def __init__(self, threat_organ: ThreatOrgan, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("PolicySimulator", self.VERSION)
        self.threat_organ = threat_organ

    def simulate_rule(self, rule: PolicyRule) -> Dict[str, Any]:
        impacted = 0
        false_positives = 0
        for rec in self.threat_organ.records:
            if rec.kind != rule.kind:
                continue
            if not rec.path.lower().startswith(rule.path_prefix.lower()):
                continue
            impacted += 1
            if rec.allowed and rule.action == "block":
                false_positives += 1
        self.registry.heartbeat("PolicySimulator")
        return {
            "impacted": impacted,
            "false_positives": false_positives,
        }

# ============================================================================
# Health Organ
# ============================================================================

class HealthOrgan:
    VERSION = "1.0"

    def __init__(self, registry: ModuleRegistry, pending: PendingDecisionOrgan):
        self.registry = registry
        self.registry.register("HealthOrgan", self.VERSION)
        self.pending = pending
        self.health_score = 1.0
        self.last_eval = 0.0

    def evaluate(self):
        now = time.time()
        if now - self.last_eval < 2.0:
            return self.health_score
        self.last_eval = now

        score = getattr(deps, "_readiness", 0.0)

        modules = self.registry.snapshot()
        for m in modules:
            if m.status == "error":
                score -= 0.1
            if time.time() - m.last_heartbeat > 10.0:
                score -= 0.1

        pending_count = len(self.pending.list_pending(include_decided=False))
        if pending_count > 50:
            score -= 0.2
        elif pending_count > 10:
            score -= 0.1

        self.health_score = max(0.0, min(1.0, score))
        self.registry.heartbeat("HealthOrgan")
        return self.health_score

# ============================================================================
# Hot Reload Organ
# ============================================================================

class HotReloadOrgan:
    VERSION = "1.0"

    def __init__(self, config_path: str, dynamic_policy: DynamicPolicy, registry: ModuleRegistry):
        self.registry = registry
        self.registry.register("HotReloadOrgan", self.VERSION)
        self.config_path = config_path
        self.dynamic_policy = dynamic_policy
        self._stop = False
        self._last_mtime = 0.0

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            try:
                if os.path.exists(self.config_path):
                    mtime = os.path.getmtime(self.config_path)
                    if mtime != self._last_mtime:
                        self._last_mtime = mtime
                        self._reload()
                self.registry.heartbeat("HotReloadOrgan")
            except Exception as exc:
                logger.error("HotReloadOrgan error: %s", exc)
                self.registry.error("HotReloadOrgan")
            time.sleep(3.0)

    def _reload(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            logger.error("Failed to reload config: %s", exc)
            self.registry.error("HotReloadOrgan")
            return

        rules = cfg.get("dynamic_rules", [])
        self.dynamic_policy.rules.clear()
        for r in rules:
            try:
                self.dynamic_policy.rules.append(DynamicRule(
                    kind=r["kind"],
                    proc_class=r["proc_class"],
                    process_name=r.get("process_name"),
                    decision=r["decision"],
                    source=r.get("source", "hot_reload"),
                    timestamp=time.time(),
                ))
            except KeyError:
                continue
        logger.info("HotReloadOrgan applied %d dynamic rules from config.", len(self.dynamic_policy.rules))

# ============================================================================
# Cockpit UI (enhanced)
# ============================================================================

class CockpitUI:
    VERSION = "1.2"

    def __init__(self, ui_organ: UIAutomationOrgan, policy_organ: PolicyOrgan,
                 threat_organ: ThreatOrgan, pending_organ: PendingDecisionOrgan,
                 dynamic_policy: DynamicPolicy, memory: RebootMemoryOrgan,
                 health_organ: HealthOrgan, module_registry: ModuleRegistry,
                 simulator: PolicySimulator, future_organ: FutureOrgan):
        if Qt is None:
            raise RuntimeError("PyQt6 not available for cockpit UI")

        from PyQt6.QtWidgets import (
            QApplication, QWidget, QVBoxLayout, QLabel, QListWidget,
            QListWidgetItem, QHBoxLayout, QPushButton, QSplitter,
            QTabWidget, QTextEdit
        )
        from PyQt6.QtCore import QTimer, Qt as QtCoreQt
        from PyQt6.QtGui import QColor

        self.QApplication = QApplication
        self.QWidget = QWidget
        self.QVBoxLayout = QVBoxLayout
        self.QHBoxLayout = QHBoxLayout
        self.QLabel = QLabel
        self.QListWidget = QListWidget
        self.QListWidgetItem = QListWidgetItem
        self.QPushButton = QPushButton
        self.QSplitter = QSplitter
        self.QTabWidget = QTabWidget
        self.QTextEdit = QTextEdit
        self.QTimer = QTimer
        self.QtCoreQt = QtCoreQt
        self.QColor = QColor

        self.ui_organ = ui_organ
        self.policy_organ = policy_organ
        self.threat_organ = threat_organ
        self.pending_organ = pending_organ
        self.dynamic_policy = dynamic_policy
        self.memory = memory
        self.health_organ = health_organ
        self.module_registry = module_registry
        self.simulator = simulator
        self.future_organ = future_organ

        self.app = self.QApplication([])
        self.window = self.QWidget()
        self.window.setWindowTitle("Black Night Cockpit (Upgraded)")

        root = self.QVBoxLayout()

        top_bar = self.QHBoxLayout()
        self.label_ui = self.QLabel("UIAutomation: unknown")
        self.label_mode = self.QLabel("Policy mode: degraded")
        self.label_ready = self.QLabel(f"Readiness: {getattr(deps, '_readiness', 0.0):.2f}")
        self.label_health = self.QLabel("Health: 1.00")
        self.label_pending = self.QLabel("Pending: 0")
        top_bar.addWidget(self.label_ui)
        top_bar.addWidget(self.label_mode)
        top_bar.addWidget(self.label_ready)
        top_bar.addWidget(self.label_health)
        top_bar.addWidget(self.label_pending)
        root.addLayout(top_bar)

        tabs = self.QTabWidget()

        # Tab 1: Events
        events_tab = self.QWidget()
        events_layout = self.QVBoxLayout()
        splitter = self.QSplitter(self.QtCoreQt.Orientation.Horizontal)

        self.list_all = self.QListWidget()
        self.list_all.setMinimumWidth(400)
        splitter.addWidget(self.list_all)

        middle_panel = self.QWidget()
        middle_layout = self.QVBoxLayout()
        self.list_pending = self.QListWidget()
        self.list_pending.setMinimumWidth(400)
        btn_layout = self.QHBoxLayout()
        self.btn_allow = self.QPushButton("Allow")
        self.btn_block = self.QPushButton("Block")
        btn_layout.addWidget(self.btn_allow)
        btn_layout.addWidget(self.btn_block)
        middle_layout.addWidget(self.list_pending)
        middle_layout.addLayout(btn_layout)
        middle_panel.setLayout(middle_layout)
        splitter.addWidget(middle_panel)

        self.list_blocked = self.QListWidget()
        self.list_blocked.setMinimumWidth(400)
        splitter.addWidget(self.list_blocked)

        events_layout.addWidget(splitter)
        events_tab.setLayout(events_layout)
        tabs.addTab(events_tab, "Events")

        # Tab 2: Details
        details_tab = self.QWidget()
        details_layout = self.QVBoxLayout()
        self.details_text = self.QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        details_tab.setLayout(details_layout)
        tabs.addTab(details_tab, "Details")

        # Tab 3: Modules
        modules_tab = self.QWidget()
        modules_layout = self.QVBoxLayout()
        self.list_modules = self.QListWidget()
        modules_layout.addWidget(self.list_modules)
        modules_tab.setLayout(modules_layout)
        tabs.addTab(modules_tab, "Organs")

        root.addWidget(tabs)
        self.window.setLayout(root)

        self.list_pending.itemSelectionChanged.connect(self._on_pending_selected)
        self.list_pending.itemClicked.connect(self._on_pending_clicked)
        self.list_all.itemClicked.connect(self._on_all_clicked)
        self.list_blocked.itemClicked.connect(self._on_blocked_clicked)
        self.btn_allow.clicked.connect(self._on_allow_clicked)
        self.btn_block.clicked.connect(self._on_block_clicked)

        self._pending_index_map: Dict[int, int] = {}

        self.timer = self.QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(500)

    def _format_process(self, pid: Optional[int], proc_class: str, pname: Optional[str]) -> str:
        pid_str = f"pid={pid}" if pid is not None else "pid=?"
        name_str = pname or "unknown"
        return f"{name_str} ({pid_str}, class={proc_class})"

    def _format_dest(self, dest_host: Optional[str], dest_ip: Optional[str]) -> str:
        if not dest_host and not dest_ip:
            return ""
        if dest_host and dest_ip:
            return f" -> {dest_host} [{dest_ip}]"
        return f" -> {dest_host or dest_ip}"

    def _refresh(self):
        self.label_ui.setText(f"UIAutomation: {self.ui_organ.signal}")
        mode = self.policy_organ.evaluate()
        self.label_mode.setText(f"Policy mode: {mode}")
        self.label_ready.setText(f"Readiness: {getattr(deps, '_readiness', 0.0):.2f}")
        health = self.health_organ.evaluate()
        self.label_health.setText(f"Health: {health:.2f}")
        pending_count = len(self.pending_organ.list_pending(include_decided=False))
        self.label_pending.setText(f"Pending: {pending_count}")

        self.list_all.clear()
        for rec in self.threat_organ.recent(50):
            status = "ALLOWED" if rec.allowed else "BLOCKED"
            proc_str = self._format_process(rec.pid, rec.proc_class, get_process_name(rec.pid))
            src = rec.details.get("source", "poll")
            maint = " M" if rec.maintenance_allowed else ""
            rule = f" rule={rec.rule_action}/{rec.rule_severity}" if rec.rule_action else ""
            text = (
                f"{time.strftime('%H:%M:%S', time.localtime(rec.timestamp))} "
                f"[{status}{maint}] {rec.kind} {rec.path} {proc_str} "
                f"src={src}{rule} conf={rec.confidence:.2f}"
            )
            item = self.QListWidgetItem(text)
            if not rec.allowed:
                item.setForeground(self.QColor("red"))
            else:
                item.setForeground(self.QColor("green"))
            self.list_all.addItem(item)

        self.list_pending.clear()
        self._pending_index_map.clear()
        pending_all = self.pending_organ.list_pending(include_decided=True)
        for idx, p in enumerate(pending_all):
            if p.decided:
                continue
            proc_str = self._format_process(p.pid, p.proc_class, p.process_name)
            dest_str = self._format_dest(p.dest_host, p.dest_ip)
            hint_str = f" | {p.hint}" if p.hint else ""
            text = (
                f"{time.strftime('%H:%M:%S', time.localtime(p.timestamp))} "
                f"[PENDING] {p.kind} {p.description}{dest_str} {proc_str} "
                f"sev={p.severity} conf={p.confidence:.2f}{hint_str}"
            )
            item = self.QListWidgetItem(text)
            item.setForeground(self.QColor("orange"))
            self.list_pending.addItem(item)
            self._pending_index_map[self.list_pending.row(item)] = idx

        self.list_blocked.clear()
        for p in pending_all:
            if not p.decided or p.decision != "block":
                continue
            proc_str = self._format_process(p.pid, p.proc_class, p.process_name)
            dest_str = self._format_dest(p.dest_host, p.dest_ip)
            src = p.verdict_source or "operator"
            hint_str = f" | {p.hint}" if p.hint else ""
            text = (
                f"{time.strftime('%H:%M:%S', time.localtime(p.timestamp))} "
                f"[BLOCKED/{src}] {p.kind} {p.description}{dest_str} {proc_str} "
                f"sev={p.severity} conf={p.confidence:.2f}{hint_str}"
            )
            item = self.QListWidgetItem(text)
            item.setForeground(self.QColor("red"))
            self.list_blocked.addItem(item)

        self.list_modules.clear()
        for m in self.module_registry.snapshot():
            age = time.time() - m.last_heartbeat
            text = (
                f"{m.name} v{m.version} status={m.status} "
                f"errors={m.error_count} last={age:.1f}s"
            )
            item = self.QListWidgetItem(text)
            if m.status == "error":
                item.setForeground(self.QColor("red"))
            elif age > 10.0:
                item.setForeground(self.QColor("orange"))
            else:
                item.setForeground(self.QColor("green"))
            self.list_modules.addItem(item)

    def _on_pending_selected(self):
        row = self.list_pending.currentRow()
        has_selection = row >= 0
        self.btn_allow.setEnabled(has_selection)
        self.btn_block.setEnabled(has_selection)

    def _show_pending_details(self, idx: int):
        pending_all = self.pending_organ.list_pending(include_decided=True)
        if not (0 <= idx < len(pending_all)):
            return
        p = pending_all[idx]
        future_hint = self.future_organ.hint_for_process(p.process_name)
        text = []
        text.append(f"Kind: {p.kind}")
        text.append(f"Description: {p.description}")
        text.append(f"Process: {p.process_name} (pid={p.pid}, class={p.proc_class})")
        text.append(f"Destination: {p.dest_host} [{p.dest_ip}]")
        text.append(f"Severity: {p.severity}")
        text.append(f"Decided: {p.decided} ({p.decision})")
        text.append(f"Source: {p.verdict_source}")
        text.append(f"Confidence: {p.confidence:.2f}")
        text.append(f"Hint: {p.hint}")
        text.append(f"Future: {future_hint}")
        self.details_text.setPlainText("\n".join(text))

    def _on_pending_clicked(self, item):
        row = self.list_pending.currentRow()
        idx = self._pending_index_map.get(row)
        if idx is not None:
            self._show_pending_details(idx)

    def _on_all_clicked(self, item):
        self.details_text.setPlainText(item.text())

    def _on_blocked_clicked(self, item):
        self.details_text.setPlainText(item.text())

    def _apply_decision_to_policy(self, idx: int, decision: str):
        pending_all = self.pending_organ.list_pending(include_decided=True)
        if not (0 <= idx < len(pending_all)):
            return
        p = pending_all[idx]
        rule = DynamicRule(
            kind=p.kind,
            proc_class=p.proc_class,
            process_name=p.process_name,
            decision=decision,
            source="operator",
            timestamp=time.time(),
        )
        self.dynamic_policy.add_rule(rule)
        self.memory.save_state_async()

    def _on_allow_clicked(self):
        row = self.list_pending.currentRow()
        if row < 0:
            return
        idx = self._pending_index_map.get(row)
        if idx is None:
            return
        self.pending_organ.decide(idx, "allow", source="operator")
        self._apply_decision_to_policy(idx, "allow")
        self._show_pending_details(idx)

    def _on_block_clicked(self):
        row = self.list_pending.currentRow()
        if row < 0:
            return
        idx = self._pending_index_map.get(row)
        if idx is None:
            return
        self.pending_organ.decide(idx, "block", source="operator")
        self._apply_decision_to_policy(idx, "block")
        self._show_pending_details(idx)

    def run(self):
        self.window.show()
        self.app.exec()

# ============================================================================
# Black Night Spine (upgraded)
# ============================================================================

class BlackNightSpine:
    VERSION = "2.0"

    def __init__(self, telemetry_source: Optional[NetworkTelemetrySource] = None):
        self.module_registry = ModuleRegistry()
        self.module_registry.register("BlackNightSpine", self.VERSION)

        self.bus = EventBus()
        self.safe_mode = SafeModeController()

        self.ui_organ = UIAutomationOrgan(self.module_registry)
        self.policy_organ = PolicyOrgan(self.ui_organ, self.safe_mode, self.module_registry)
        self.dynamic_policy = DynamicPolicy(self.module_registry)
        self.settings_organ = SettingsOrgan(self.bus, self.module_registry)
        self.etw_organ = ETWOrgan(self.bus, self.module_registry)
        self.rollback_organ = RollbackOrgan(self.module_registry)
        self.threat_organ = ThreatOrgan(self.module_registry)
        self.confidence_engine = ConfidenceEngine(self.module_registry)
        self.memory_organ = RebootMemoryOrgan(self.module_registry)
        self.pending_organ = PendingDecisionOrgan(self.memory_organ, self.module_registry)
        self.reputation_organ = ReputationOrgan(self.module_registry)
        self.behavior_memory = BehaviorMemoryOrgan(self.module_registry)
        self.context_engine = ContextEngine(self.module_registry)
        self.future_organ = FutureOrgan(self.behavior_memory, self.module_registry)
        self.health_organ = HealthOrgan(self.module_registry, self.pending_organ)
        self.simulator = PolicySimulator(self.threat_organ, self.module_registry)

        if telemetry_source is None:
            telemetry_source = DummyNetworkTelemetrySource()
        self.network_organ = NetworkOrgan(
            self.pending_organ,
            self.dynamic_policy,
            self.reputation_organ,
            telemetry_source,
            self.behavior_memory,
            self.confidence_engine,
            self.context_engine,
            self.module_registry,
        )
        self.firewall_organ = FirewallOrgan(self.pending_organ, self.module_registry)

        self.memory_organ.attach(
            dynamic_policy=self.dynamic_policy,
            rollback=self.rollback_organ,
            pending=self.pending_organ,
            policy=self.policy_organ,
        )

        self.validator = ChangeValidator(
            self.ui_organ,
            self.policy_organ,
            DEFAULT_POLICY_RULES,
            self.dynamic_policy,
            self.confidence_engine,
            self.module_registry,
        )

        self.tamper_organ = TamperOrgan(
            self.policy_organ,
            self.bus,
            self_paths=[
                r"Software\BlackNight",
                r"C:\Temp\blacknight",
            ],
            memory=self.memory_organ,
            registry=self.module_registry,
        )

        self.bus.subscribe("setting_changed", self._on_setting_changed)
        self.bus.subscribe("setting_changed", self.tamper_organ.on_setting_changed)

        config_path = os.path.join(os.path.expanduser("~"), "BlackNight", "blacknight_config.json")
        self.hot_reload = HotReloadOrgan(config_path, self.dynamic_policy, self.module_registry)

    def _on_setting_changed(self, change: Dict[str, Any]):
        decision, conf = self.validator.validate(change)
        self.threat_organ.log_change(change, decision, conf)

        if not decision["allowed"] and not decision.get("maintenance_allowed", False):
            self.rollback_organ.rollback(change)
        self.memory_organ.save_state_async()

    def _boot_sanity_scan(self):
        pid = None
        proc_class = "unknown"

        for item in self.settings_organ.registry_values:
            path = f"{item.path}\\{item.name}"
            baseline = self.rollback_organ.registry_baseline.get(path)
            if baseline is None:
                continue
            base_val, base_type, _ = baseline
            try:
                key = win32api.RegOpenKeyEx(item.hive, item.path, 0, win32con.KEY_READ)
                cur_val, cur_type = win32api.RegQueryValueEx(key, item.name)
                win32api.RegCloseKey(key)
            except Exception:
                cur_val, cur_type = None, None

            if cur_val != base_val or cur_type != base_type:
                self.bus.publish("setting_changed", {
                    "kind": "registry",
                    "path": path,
                    "name": item.name,
                    "old": (base_val, base_type),
                    "new": (cur_val, cur_type),
                    "pid": pid,
                    "proc_class": proc_class,
                    "hive": item.hive,
                    "source": "boot_sanity",
                })

        for f in self.settings_organ.files:
            baseline = self.rollback_organ.file_baseline.get(f.path)
            if baseline is None:
                continue
            try:
                with open(f.path, "rb") as fh:
                    cur_content = fh.read()
            except Exception:
                cur_content = None

            if cur_content != baseline:
                self.bus.publish("setting_changed", {
                    "kind": "file",
                    "path": f.path,
                    "name": None,
                    "old": baseline,
                    "new": cur_content,
                    "pid": pid,
                    "proc_class": proc_class,
                    "hive": None,
                    "source": "boot_sanity",
                })

    def start(self):
        self.memory_organ.load_state()

        self.settings_organ.add_registry_watch(
            win32con.HKEY_CURRENT_USER,
            r"Software\BlackNight",
            "TestValue",
        )
        self.settings_organ.add_file_watch(r"C:\Temp\blacknight_test.cfg")

        if not self.rollback_organ.registry_baseline:
            for item in self.settings_organ.registry_values:
                self.rollback_organ.capture_baseline_registry(item)
        if not self.rollback_organ.file_baseline:
            for f in self.settings_organ.files:
                self.rollback_organ.capture_baseline_file(f.path)

        self.memory_organ.save_state_async()

        self._boot_sanity_scan()

        self.ui_organ.start()
        self.settings_organ.start()
        self.etw_organ.start()
        self.network_organ.start()
        self.firewall_organ.start()
        self.hot_reload.start()

        self.module_registry.heartbeat("BlackNightSpine")

    def run_cockpit(self):
        if Qt is None:
            raise RuntimeError("PyQt6 not available")
        cockpit = CockpitUI(
            self.ui_organ,
            self.policy_organ,
            self.threat_organ,
            self.pending_organ,
            self.dynamic_policy,
            self.memory_organ,
            self.health_organ,
            self.module_registry,
            self.simulator,
            self.future_organ,
        )
        cockpit.run()


if __name__ == "__main__":
    spine = BlackNightSpine(telemetry_source=DummyNetworkTelemetrySource())
    spine.start()
    spine.run_cockpit()
