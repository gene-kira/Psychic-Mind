"""
blacknight_spine_pluggable_net.py – Black Night spine with pluggable NetworkTelemetrySource

Architecture:
- UIAutomationOrgan (Win32 GetLastInputInfo)
- PolicyOrgan (full_enforcement / degraded / maintenance / frozen)
- SettingsOrgan (registry + file polling)
- ChangeValidator (per-path rules + DynamicPolicy)
- RollbackOrgan (registry + file rollback)
- ThreatOrgan (records all changes)
- TamperOrgan (frozen state on self-modification)
- RebootMemoryOrgan (persistent dynamic policy + pending + tamper state)
- DynamicPolicy (live rules from Allow/Block)
- ReputationOrgan (process → expected hosts)
- PendingDecisionOrgan (network/firewall decisions)
- NetworkTelemetrySource (interface – you implement real telemetry here)
- NetworkOrgan (consumes telemetry events, uses ReputationOrgan + DynamicPolicy)
- FirewallOrgan (simulated; can be made pluggable similarly)
- CockpitUI (All / Pending / Blocked panes with Allow/Block)
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
# UIAutomation Organ
# ============================================================================

class UIAutomationOrgan:
    def __init__(self):
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
            except Exception as exc:
                logger.error("UIAutomation poll error: %s", exc)
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
# Policy Organ
# ============================================================================

class PolicyOrgan:
    def __init__(self, ui_organ: UIAutomationOrgan):
        self.ui = ui_organ
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
    def __init__(self, bus: EventBus):
        self.bus = bus
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
            except Exception as exc:
                logger.error("Settings poll error: %s", exc)
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
# ETW Organ (optional, still for registry/file)
# ============================================================================

class ETWOrgan:
    REGISTRY_PROVIDER = "{70EB4F03-C1DE-4F73-A051-33D13D5413BD}"

    def __init__(self, bus: EventBus):
        self.bus = bus
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
            except Exception as exc:
                logger.error("ETW callback error: %s", exc)

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
    def __init__(self):
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
# Change Validator
# ============================================================================

class ChangeValidator:
    def __init__(self, ui_organ: UIAutomationOrgan, policy_organ: PolicyOrgan,
                 rules: List[PolicyRule], dynamic_policy: DynamicPolicy):
        self.ui = ui_organ
        self.policy = policy_organ
        self.rules = rules
        self.dynamic_policy = dynamic_policy

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

    def validate(self, change: Dict[str, Any]) -> Dict[str, Any]:
        mode = self.policy.evaluate()
        ui_signal = self.ui.signal
        rule = self._match_rule(change)

        if mode == "maintenance":
            return {
                "allowed": True,
                "mode": mode,
                "ui_signal": ui_signal,
                "rule": rule,
                "maintenance_allowed": True,
            }

        if mode == "frozen":
            return {
                "allowed": False,
                "mode": mode,
                "ui_signal": ui_signal,
                "rule": rule,
                "maintenance_allowed": False,
            }

        dyn_decision = self.dynamic_policy.decide_for_process(
            kind="setting",
            pid=change.get("pid"),
            proc_class=change.get("proc_class", "unknown"),
        )
        if dyn_decision is not None:
            allowed = (dyn_decision == "allow")
            return {
                "allowed": allowed,
                "mode": mode,
                "ui_signal": ui_signal,
                "rule": rule,
                "maintenance_allowed": False,
            }

        if rule:
            if rule.action == "block":
                allowed = False
            elif rule.action == "allow":
                allowed = True
            else:
                allowed = (mode == "full_enforcement" and ui_signal == "user_intent_confirmed")
        else:
            allowed = (mode == "full_enforcement" and ui_signal == "user_intent_confirmed")

        return {
            "allowed": allowed,
            "mode": mode,
            "ui_signal": ui_signal,
            "rule": rule,
            "maintenance_allowed": False,
        }

# ============================================================================
# Rollback Organ
# ============================================================================

class RollbackOrgan:
    def __init__(self):
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

    def capture_baseline_file(self, path: str):
        try:
            with open(path, "rb") as f:
                self.file_baseline[path] = f.read()
        except Exception:
            self.file_baseline[path] = b""

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
            except Exception as exc:
                logger.error("Registry rollback error for %s: %s", path, exc)

        elif kind == "file":
            baseline = self.file_baseline.get(path, None)
            if baseline is None:
                return
            try:
                with open(path, "wb") as f:
                    f.write(baseline)
                logger.info("Rolled back file %s", path)
            except Exception as exc:
                logger.error("File rollback error for %s: %s", path, exc)

# ============================================================================
# RebootMemoryOrgan
# ============================================================================

class RebootMemoryOrgan:
    def __init__(self, path: str = None):
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
            return None

    def _save_raw(self, data: Dict[str, Any]):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)
        except Exception as exc:
            logger.error("Failed to save state file: %s", exc)

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
            "version": 1,
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
    def __init__(self, policy: PolicyOrgan, bus: EventBus, self_paths: List[str], memory: RebootMemoryOrgan):
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
    details: Dict[str, Any] = field(default_factory=dict)


class ThreatOrgan:
    def __init__(self):
        self.records: List[ThreatRecord] = []

    def log_change(self, change: Dict[str, Any], decision: Dict[str, Any]):
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
            details={
                "old": change.get("old"),
                "new": change.get("new"),
                "source": change.get("source", "poll"),
            },
        )
        self.records.append(rec)

    def recent(self, limit: int = 50) -> List[ThreatRecord]:
        return self.records[-limit:]

# ============================================================================
# ReputationOrgan
# ============================================================================

class ReputationOrgan:
    """
    Local reputation map: process -> expected hosts.
    """

    def __init__(self):
        self.trusted_dests: Dict[str, Dict[str, Any]] = {
            "MsMpEng.exe": {
                "hosts": {"wdcp.microsoft.com", "defender.microsoft.com", "update.microsoft.com"},
            },
            "chrome.exe": {
                "hosts": {"www.google.com", "accounts.google.com"},
            },
        }

    def evaluate_connection(self, process_name: Optional[str], dest_host: str, dest_ip: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Returns (auto_decision, verdict_source, hint)
        auto_decision: "allow" / "block" / None
        verdict_source: "auto_reputation" or None
        hint: human-readable explanation
        """
        pname = process_name or ""
        trusted = self.trusted_dests.get(pname, None)
        if trusted:
            if dest_host in trusted["hosts"]:
                return None, None, f"{pname} talking to known host {dest_host}"
            else:
                return "block", "auto_reputation", f"{pname} talking to unexpected host {dest_host}"
        else:
            return None, None, f"Unknown process {pname or 'unknown'} to {dest_host}"

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


class PendingDecisionOrgan:
    def __init__(self, memory: RebootMemoryOrgan = None):
        self.pending: List[PendingDecision] = []
        self.memory = memory

    def add(self, decision: PendingDecision):
        self.pending.append(decision)
        if self.memory:
            self.memory.save_state_async()

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

    def index_of(self, obj: PendingDecision) -> int:
        try:
            return self.pending.index(obj)
        except ValueError:
            return -1

# ============================================================================
# NetworkTelemetrySource interface
# ============================================================================

class NetworkTelemetryEvent:
    """
    A single network event emitted by a telemetry source.
    """
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
    """
    Interface for real network telemetry.

    You implement:
      - start()
      - stop()
      - on_event(callback)

    The callback receives NetworkTelemetryEvent instances.
    """

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def on_event(self, callback: Callable[[NetworkTelemetryEvent], None]) -> None:
        ...


class DummyNetworkTelemetrySource:
    """
    Example stub that does nothing. Replace with your real implementation
    that emits NetworkTelemetryEvent objects.
    """

    def __init__(self):
        self._callback: Optional[Callable[[NetworkTelemetryEvent], None]] = None
        self._stop = False

    def on_event(self, callback: Callable[[NetworkTelemetryEvent], None]) -> None:
        self._callback = callback

    def start(self) -> None:
        # This stub does nothing. Your real implementation should:
        # - hook into your telemetry mechanism
        # - call self._callback(NetworkTelemetryEvent(...)) for each event
        logger.info("DummyNetworkTelemetrySource started (no real telemetry).")
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop = True

    def _loop(self):
        # No simulated events here – this is intentionally quiet.
        while not self._stop:
            time.sleep(5.0)

# ============================================================================
# Network / Firewall Organs (NetworkOrgan consumes NetworkTelemetrySource)
# ============================================================================

class NetworkOrgan:
    def __init__(self,
                 pending: PendingDecisionOrgan,
                 dynamic_policy: DynamicPolicy,
                 reputation: ReputationOrgan,
                 telemetry_source: NetworkTelemetrySource):
        self.pending = pending
        self.dynamic_policy = dynamic_policy
        self.reputation = reputation
        self.telemetry_source = telemetry_source

    def start(self):
        def handle_event(ev: NetworkTelemetryEvent):
            try:
                self._handle_event(ev)
            except Exception as exc:
                logger.error("NetworkOrgan event handling error: %s", exc)

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

        if auto_decision:
            pd.decided = True
            pd.decision = auto_decision
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
            self.pending.add(pd)


class FirewallOrgan:
    """
    Still simulated. You can later replace this with a pluggable FirewallTelemetrySource
    using the same pattern as NetworkTelemetrySource.
    """
    def __init__(self, pending: PendingDecisionOrgan):
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

# ============================================================================
# Cockpit UI
# ============================================================================

class CockpitUI:
    def __init__(self, ui_organ: UIAutomationOrgan, policy_organ: PolicyOrgan,
                 threat_organ: ThreatOrgan, pending_organ: PendingDecisionOrgan,
                 dynamic_policy: DynamicPolicy, memory: RebootMemoryOrgan):
        if Qt is None:
            raise RuntimeError("PyQt6 not available for cockpit UI")

        from PyQt6.QtWidgets import (
            QApplication, QWidget, QVBoxLayout, QLabel, QListWidget,
            QListWidgetItem, QHBoxLayout, QPushButton, QSplitter
        )
        from PyQt6.QtCore import QTimer, Qt as QtCoreQt

        self.QApplication = QApplication
        self.QWidget = QWidget
        self.QVBoxLayout = QVBoxLayout
        self.QHBoxLayout = QHBoxLayout
        self.QLabel = QLabel
        self.QListWidget = QListWidget
        self.QListWidgetItem = QListWidgetItem
        self.QPushButton = QPushButton
        self.QSplitter = QSplitter
        self.QTimer = QTimer
        self.QtCoreQt = QtCoreQt

        self.ui_organ = ui_organ
        self.policy_organ = policy_organ
        self.threat_organ = threat_organ
        self.pending_organ = pending_organ
        self.dynamic_policy = dynamic_policy
        self.memory = memory

        self.app = self.QApplication([])
        self.window = self.QWidget()
        self.window.setWindowTitle("Black Night Cockpit")

        root = self.QVBoxLayout()

        self.label_ui = self.QLabel("UIAutomation: unknown")
        self.label_mode = self.QLabel("Policy mode: degraded")
        self.label_ready = self.QLabel(f"Readiness: {getattr(deps, '_readiness', 0.0):.2f}")

        root.addWidget(self.label_ui)
        root.addWidget(self.label_mode)
        root.addWidget(self.label_ready)

        splitter = self.QSplitter(self.QtCoreQt.Orientation.Horizontal)

        # Left: all events
        self.list_all = self.QListWidget()
        self.list_all.setMinimumWidth(400)
        splitter.addWidget(self.list_all)

        # Middle: pending
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

        # Right: blocked
        self.list_blocked = self.QListWidget()
        self.list_blocked.setMinimumWidth(400)
        splitter.addWidget(self.list_blocked)

        root.addWidget(splitter)
        self.window.setLayout(root)

        self.list_pending.itemSelectionChanged.connect(self._on_pending_selected)
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

        # All events (settings changes)
        self.list_all.clear()
        for rec in self.threat_organ.recent(50):
            status = "ALLOWED" if rec.allowed else "BLOCKED"
            proc_str = self._format_process(rec.pid, rec.proc_class, get_process_name(rec.pid))
            src = rec.details.get("source", "poll")
            maint = " M" if rec.maintenance_allowed else ""
            rule = f" rule={rec.rule_action}/{rec.rule_severity}" if rec.rule_action else ""
            text = (
                f"{time.strftime('%H:%M:%S', time.localtime(rec.timestamp))} "
                f"[{status}{maint}] {rec.kind} {rec.path} {proc_str} src={src}{rule}"
            )
            self.list_all.addItem(self.QListWidgetItem(text))

        # Pending (only undecided)
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
                f"sev={p.severity}{hint_str}"
            )
            item = self.QListWidgetItem(text)
            self.list_pending.addItem(item)
            self._pending_index_map[self.list_pending.row(item)] = idx

        # Blocked list (auto + operator)
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
                f"sev={p.severity}{hint_str}"
            )
            self.list_blocked.addItem(self.QListWidgetItem(text))

    def _on_pending_selected(self):
        row = self.list_pending.currentRow()
        has_selection = row >= 0
        self.btn_allow.setEnabled(has_selection)
        self.btn_block.setEnabled(has_selection)

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

    def _on_block_clicked(self):
        row = self.list_pending.currentRow()
        if row < 0:
            return
        idx = self._pending_index_map.get(row)
        if idx is None:
            return
        self.pending_organ.decide(idx, "block", source="operator")
        self._apply_decision_to_policy(idx, "block")

    def run(self):
        self.window.show()
        self.app.exec()

# ============================================================================
# Black Night Spine (with pluggable NetworkTelemetrySource)
# ============================================================================

class BlackNightSpine:
    def __init__(self, telemetry_source: Optional[NetworkTelemetrySource] = None):
        self.bus = EventBus()
        self.ui_organ = UIAutomationOrgan()
        self.policy_organ = PolicyOrgan(self.ui_organ)
        self.dynamic_policy = DynamicPolicy()
        self.settings_organ = SettingsOrgan(self.bus)
        self.etw_organ = ETWOrgan(self.bus)
        self.rollback_organ = RollbackOrgan()
        self.threat_organ = ThreatOrgan()
        self.memory_organ = RebootMemoryOrgan()
        self.pending_organ = PendingDecisionOrgan(self.memory_organ)
        self.reputation_organ = ReputationOrgan()

        if telemetry_source is None:
            telemetry_source = DummyNetworkTelemetrySource()
        self.network_organ = NetworkOrgan(
            self.pending_organ,
            self.dynamic_policy,
            self.reputation_organ,
            telemetry_source,
        )
        self.firewall_organ = FirewallOrgan(self.pending_organ)

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
        )

        self.tamper_organ = TamperOrgan(
            self.policy_organ,
            self.bus,
            self_paths=[
                r"Software\BlackNight",
                r"C:\Temp\blacknight",
            ],
            memory=self.memory_organ,
        )

        self.bus.subscribe("setting_changed", self._on_setting_changed)
        self.bus.subscribe("setting_changed", self.tamper_organ.on_setting_changed)

    def _on_setting_changed(self, change: Dict[str, Any]):
        decision = self.validator.validate(change)
        self.threat_organ.log_change(change, decision)

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
        )
        cockpit.run()


if __name__ == "__main__":
    # Plug in your real NetworkTelemetrySource implementation here instead of DummyNetworkTelemetrySource.
    spine = BlackNightSpine(telemetry_source=DummyNetworkTelemetrySource())
    spine.start()
    spine.run_cockpit()

