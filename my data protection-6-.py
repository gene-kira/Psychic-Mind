# === AUTO-ELEVATION CHECK =====================================================
import ctypes
import os
import sys

def ensure_admin():
    try:
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
        print(f"[Guardian Sentinel] Elevation failed: {e}")
        sys.exit()

ensure_admin()

"""
GUARDIAN SENTINEL - TIER-2+ DEFENSIVE ORGANISM
----------------------------------------------
Defensive-only capabilities:

- Medium neon GUI (400x200), normal window
- GUI updates when new alerts arrive
- Tray notifications on alerts

Core subsystems:
- Telemetry + anomaly detection
- Egress-focused policy engine
- Shell exfil watcher (ML-assisted)
- USB watcher
- Browser upload watcher (behavioral)
- ETW integration via external agent (TCP/JSON, auto-disable)
- WinDivert integration via external helper (TCP/JSON, auto-disable)
- ML model wrapper (ONNX/Torch) with feature vector builder
- Threat-chain predictor v2 hooks (sequence-ready)
- Persona system v2 (adaptive via threat-chain + risk)
- Encrypted swarm mesh (Fernet)
- Swarm control protocol + consensus engine v2 (weights + quorum)
- Remote operator console over swarm (extended commands)
- Encrypted logs
- Payload protector (encrypt + mask)
- Self-healing watchdog
- Local CLI control socket (status, alerts, config) with auto-port fallback

EXTERNAL AGENT DESIGNS (REAL INTEGRATIONS)
------------------------------------------
1) Real ETW Agent (separate process/service)
   - Subscribes to ETW providers (e.g., PowerShell, Sysmon, Security)
   - Normalizes events into JSON lines over TCP:
       {
         "type": "etw",
         "provider": "Microsoft-Windows-PowerShell",
         "event_id": 4104,
         "level": "Information",
         "pid": 1234,
         "tid": 5678,
         "time": "...",
         "fields": { ... }
       }
   - Guardian connects to it via ETWAgentClient (below).

2) Real WinDivert Helper (separate process/service)
   - Uses WinDivert to capture packets
   - Extracts metadata (src/dst IP/port, proto, size, direction)
   - Emits JSON lines over TCP:
       {
         "type": "packet",
         "src": "10.0.0.5:54321",
         "dst": "8.8.8.8:53",
         "proto": "UDP",
         "size": 128,
         "direction": "outbound",
         "meta": { ... }
       }
   - Guardian connects via WinDivertClient (below).

3) Real ML Model Schema (for training)
   - Input features (per shell event):
       {
         "cmdline": str,
         "proc_name": str,
         "length": int,
         "space_count": int,
         "exfil_markers": int,
         "admin_markers": int,
         "data_markers": int,
         "is_powershell": 0/1,
         "is_cmd": 0/1
       }
   - Labels:
       - 0: normal
       - 1: admin/maintenance
       - 2: exfil/suspicious
   - Model: ONNX or Torch, 3-class softmax.

4) Remote Operator Console (over swarm)
   - Swarm messages of type "operator":
       {
         "type": "operator",
         "cmd": "ping|request_risk|set_persona|request_status|broadcast_message",
         "args": { ... },
         "from": "node-X",
         "ts": ...
       }

5) Swarm Consensus Engine v2
   - Each node broadcasts:
       {
         "type": "status",
         "node": "...",
         "ts": ...,
         "exfil_events": int,
         "persona": "...",
         "local_risk": "low|medium|high"
       }
   - Consensus:
       - Weighted by recency
       - Quorum: majority of active peers
       - Aggregated network_risk

6) Persona System v2
   - Persona can auto-adjust based on:
       - local anomaly score
       - network_risk from consensus
       - threat-chain predictions

7) Threat-Chain Predictor v2
   - Current implementation: Markov-style transitions
   - Hooks ready for sequence model (e.g., transformer) to replace predictor.
"""

import time
import json
import threading
import socket
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Protocol, Tuple

try:
    import numpy as np
except ImportError:
    print("[Autoloader] numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("[Autoloader] psutil is required. Install with: pip install psutil")
    sys.exit(1)

try:
    from cryptography.fernet import Fernet
except ImportError:
    print("[Autoloader] cryptography is required. Install with: pip install cryptography")
    sys.exit(1)

try:
    from PySide6.QtCore import Qt, Signal, QObject
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QLabel,
        QSystemTrayIcon, QMenu
    )
except ImportError:
    print("[Autoloader] PySide6 is required. Install with: pip install PySide6")
    sys.exit(1)

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ====== CONFIG / DATA MODELS ==================================================

@dataclass
class GuardianConfig:
    loop_hz: float = 1.0
    allowlisted_programs: List[str] = field(default_factory=lambda: [
        "explorer.exe", "System", "python.exe", "cmd.exe", "powershell.exe"
    ])
    suspicious_ports: List[int] = field(default_factory=lambda: [22, 3389, 443, 8080, 5985])
    local_ip_prefixes: List[str] = field(default_factory=lambda: ["10.", "192.168.", "172.16."])
    swarm_enabled: bool = True
    swarm_port: int = 49777
    swarm_group: str = "239.12.12.12"
    node_name: str = "node-1"
    swarm_key_path: str = "guardian_swarm.key"
    cli_port: int = 49888
    etw_agent_host: str = "127.0.0.1"
    etw_agent_port: int = 49901
    windivert_host: str = "127.0.0.1"
    windivert_port: int = 49902
    ml_model_path: str = "guardian_model.onnx"
    persona: str = "balanced"  # balanced / paranoid / relaxed / auto


@dataclass
class TelemetrySnapshot:
    timestamp: float
    cpu_percent: float
    ram_percent: float
    process_count: int
    connections: int
    top_processes: List[Dict[str, Any]]
    anomaly_score: float = 0.0


@dataclass
class Alert:
    timestamp: float
    level: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    category: str = "generic"


@dataclass
class WorldState:
    last_snapshot: Optional[TelemetrySnapshot] = None
    alerts: List[Alert] = field(default_factory=list)
    exfil_events: int = 0
    swarm_peers: int = 0
    last_event_text: str = "Idle"
    last_level: str = "INFO"
    peers: Dict[str, float] = field(default_factory=dict)
    node_id: str = "node-1"
    swarm_key: Optional[bytes] = None
    persona: str = "balanced"
    consensus_view: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    threat_chain_state: Dict[str, int] = field(default_factory=dict)
    network_risk: str = "low"
    local_risk: str = "low"


# ====== PROFILE MANAGER =======================================================

class ProfileManager:
    def __init__(self, path: str = "guardian_profiles.json"):
        self.path = path
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.active_profile_name: str = "Default"
        self._load_or_init()

    def _default_profile(self) -> Dict[str, Any]:
        return asdict(GuardianConfig())

    def _load_or_init(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.profiles = data.get("profiles", {})
                self.active_profile_name = data.get("active_profile", "Default")
            except Exception:
                self.profiles = {"Default": self._default_profile()}
                self.active_profile_name = "Default"
        else:
            self.profiles = {"Default": self._default_profile()}
            self.active_profile_name = "Default"
            self._save()

        if self.active_profile_name not in self.profiles:
            self.active_profile_name = list(self.profiles.keys())[0]

    def _save(self):
        data = {
            "profiles": self.profiles,
            "active_profile": self.active_profile_name
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ProfileManager] Failed to save profiles: {e}")

    def get_active_config(self) -> GuardianConfig:
        prof = self.profiles.get(self.active_profile_name, self._default_profile())
        return GuardianConfig(
            loop_hz=prof.get("loop_hz", 1.0),
            allowlisted_programs=list(prof.get("allowlisted_programs", [])),
            suspicious_ports=list(prof.get("suspicious_ports", [])),
            local_ip_prefixes=list(prof.get("local_ip_prefixes", [])),
            swarm_enabled=prof.get("swarm_enabled", True),
            swarm_port=prof.get("swarm_port", 49777),
            swarm_group=prof.get("swarm_group", "239.12.12.12"),
            node_name=prof.get("node_name", "node-1"),
            swarm_key_path=prof.get("swarm_key_path", "guardian_swarm.key"),
            cli_port=prof.get("cli_port", 49888),
            etw_agent_host=prof.get("etw_agent_host", "127.0.0.1"),
            etw_agent_port=prof.get("etw_agent_port", 49901),
            windivert_host=prof.get("windivert_host", "127.0.0.1"),
            windivert_port=prof.get("windivert_port", 49902),
            ml_model_path=prof.get("ml_model_path", "guardian_model.onnx"),
            persona=prof.get("persona", "balanced"),
        )


# ====== ENCRYPTED LOGGER ======================================================

class EncryptedLogger:
    def __init__(self, log_dir: str = "guardian_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.key_path = os.path.join(self.log_dir, ".log_key")
        self.log_path = os.path.join(self.log_dir, "events.log.enc")
        self.fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_path, "wb") as f:
            f.write(key)
        return key

    def log_alert(self, alert: Alert):
        line = json.dumps({
            "ts": alert.timestamp,
            "level": alert.level,
            "msg": alert.message,
            "cat": alert.category,
            "details": alert.details,
        }) + "\n"
        token = self.fernet.encrypt(line.encode("utf-8"))
        with open(self.log_path, "ab") as f:
            f.write(token + b"\n")


# ====== PAYLOAD PROTECTOR =====================================================

class PayloadProtector:
    def __init__(self, key_dir: str = "guardian_payload"):
        self.key_dir = key_dir
        os.makedirs(self.key_dir, exist_ok=True)
        self.key_path = os.path.join(self.key_dir, ".payload_key")
        self.fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_path, "wb") as f:
            f.write(key)
        return key

    def _glyph_transform(self, data: bytes) -> bytes:
        return data[::-1]

    def _chameleon_mask(self, data: bytes) -> bytes:
        header = b"[DECOY_HEADER]\n"
        footer = b"\n[DECOY_FOOTER]"
        return header + data + footer

    def encrypt_and_mask(self, data: bytes) -> bytes:
        masked = self._chameleon_mask(data)
        transformed = self._glyph_transform(masked)
        token = self.fernet.encrypt(transformed)
        return token


# ====== ANOMALY DETECTOR ======================================================

class AnomalyDetector:
    def __init__(self, window: int = 120):
        self.window = window
        self.cpu_hist: List[float] = []
        self.ram_hist: List[float] = []

    def update_and_score(self, cpu: float, ram: float) -> float:
        self.cpu_hist.append(cpu)
        self.ram_hist.append(ram)
        if len(self.cpu_hist) > self.window:
            self.cpu_hist.pop(0)
        if len(self.ram_hist) > self.window:
            self.ram_hist.pop(0)

        if len(self.cpu_hist) < 10:
            return 0.0

        cpu_arr = np.array(self.cpu_hist)
        ram_arr = np.array(self.ram_hist)

        cpu_mean, cpu_std = cpu_arr.mean(), cpu_arr.std() + 1e-6
        ram_mean, ram_std = ram_arr.mean(), ram_arr.std() + 1e-6

        z_cpu = abs(cpu - cpu_mean) / cpu_std
        z_ram = abs(ram - ram_mean) / ram_std

        return float((z_cpu + z_ram) / 2.0)


# ====== PERSONA SYSTEM v2 =====================================================

class PersonaManager:
    """
    Persona v2:
      - balanced / paranoid / relaxed / auto
      - auto: adjusts based on local_risk + network_risk + threat-chain predictions
    """

    def __init__(self, persona: str):
        self.persona = persona

    def set_persona(self, persona: str):
        self.persona = persona

    def adjust_thresholds(self, anomaly: float, world: WorldState) -> float:
        mode = self.persona
        if mode == "auto":
            if world.network_risk == "high" or world.local_risk == "high":
                mode = "paranoid"
            elif world.network_risk == "low" and world.local_risk == "low":
                mode = "relaxed"
            else:
                mode = "balanced"

        if mode == "paranoid":
            return anomaly * 1.5
        elif mode == "relaxed":
            return anomaly * 0.7
        return anomaly


# ====== FEATURE VECTOR BUILDER (ML) ==========================================

class FeatureVectorBuilder:
    def build_for_shell(self, cmdline: str, proc_name: str) -> np.ndarray:
        text = (cmdline or "").lower()
        proc = (proc_name or "").lower()

        exfil_markers = ["invoke-webrequest", "invoke-restmethod", "curl", "wget", "ftp", "upload", "post", "put"]
        admin_markers = ["get-wmiobject", "get-ciminstance", "schtasks", "reg add", "net user", "net localgroup"]
        data_markers = ["export-csv", "select-object", "out-file", "convertto-json"]

        features: List[float] = []
        features.append(sum(1 for m in exfil_markers if m in text))
        features.append(sum(1 for m in admin_markers if m in text))
        features.append(sum(1 for m in data_markers if m in text))
        features.append(float(len(text)))
        features.append(float(text.count(" ")))
        features.append(1.0 if "powershell" in proc or "pwsh" in proc else 0.0)
        features.append(1.0 if "cmd.exe" in proc else 0.0)

        return np.array(features, dtype=np.float32).reshape(1, -1)


# ====== ML MODEL WRAPPER ======================================================

class MLModelWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.backend = None
        self.session = None
        self.model = None
        self.fv_builder = FeatureVectorBuilder()
        self._init_model()

    def _init_model(self):
        if HAS_ONNX and self.model_path.endswith(".onnx") and os.path.exists(self.model_path):
            try:
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self.backend = "onnx"
                print("[ML] ONNX model loaded.")
                return
            except Exception as e:
                print(f"[ML] Failed to load ONNX model: {e}")

        if HAS_TORCH and self.model_path.endswith(".pt") and os.path.exists(self.model_path):
            try:
                self.model = torch.jit.load(self.model_path)
                self.model.eval()
                self.backend = "torch"
                print("[ML] Torch model loaded.")
                return
            except Exception as e:
                print(f"[ML] Failed to load Torch model: {e}")

        self.backend = "heuristic"
        print("[ML] No model loaded, using heuristic classifier.")

    def classify(self, features: Dict[str, Any]) -> Dict[str, float]:
        if self.backend in ("onnx", "torch"):
            return self._model_infer(features)
        else:
            return self._heuristic(features)

    def _model_infer(self, features: Dict[str, Any]) -> Dict[str, float]:
        cmdline = features.get("cmdline", "")
        proc_name = features.get("proc_name", "")
        vec = self.fv_builder.build_for_shell(cmdline, proc_name)

        if self.backend == "onnx":
            try:
                input_name = self.session.get_inputs()[0].name
                out = self.session.run(None, {input_name: vec})[0][0]
                exfil, admin, normal = float(out[0]), float(out[1]), float(out[2])
            except Exception as e:
                print(f"[ML] ONNX inference failed, fallback to heuristic: {e}")
                return self._heuristic(features)
        else:
            try:
                with torch.no_grad():
                    t = torch.from_numpy(vec)
                    out = self.model(t)[0].detach().cpu().numpy()
                exfil, admin, normal = float(out[0]), float(out[1]), float(out[2])
            except Exception as e:
                print(f"[ML] Torch inference failed, fallback to heuristic: {e}")
                return self._heuristic(features)

        total = exfil + admin + normal
        if total <= 0:
            total = 1.0
        return {
            "exfil": exfil / total,
            "admin": admin / total,
            "normal": normal / total,
        }

    def _heuristic(self, features: Dict[str, Any]) -> Dict[str, float]:
        text = (features.get("cmdline") or "").lower()
        score_exfil = 0.0
        score_admin = 0.0
        score_normal = 0.1

        exfil_markers = ["invoke-webrequest", "invoke-restmethod", "curl", "wget", "ftp", "upload", "post", "put"]
        admin_markers = ["get-wmiobject", "get-ciminstance", "schtasks", "reg add", "net user", "net localgroup"]

        if any(m in text for m in exfil_markers):
            score_exfil += 0.7
        if any(m in text for m in admin_markers):
            score_admin += 0.6
        if "select-object" in text or "export-csv" in text:
            score_exfil += 0.3

        total = score_exfil + score_admin + score_normal
        if total == 0:
            total = 1.0

        return {
            "exfil": score_exfil / total,
            "admin": score_admin / total,
            "normal": score_normal / total,
        }


# ====== THREAT-CHAIN PREDICTOR v2 HOOKS ======================================

class ThreatChainPredictor:
    """
    v2-ready:
      - current: Markov-style transitions
      - future: plug-in transformer/sequence model
    """

    def __init__(self):
        self.transitions: Dict[str, Dict[str, int]] = {}

    def observe(self, prev_cat: str, next_cat: str):
        if prev_cat not in self.transitions:
            self.transitions[prev_cat] = {}
        self.transitions[prev_cat][next_cat] = self.transitions[prev_cat].get(next_cat, 0) + 1

    def predict_next(self, current_cat: str) -> Optional[str]:
        if current_cat not in self.transitions:
            return None
        counts = self.transitions[current_cat]
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]


# ====== POLICY ENGINE =========================================================

class PolicyEngine:
    def __init__(self, cfg: GuardianConfig, persona_mgr: PersonaManager, world: WorldState):
        self.cfg = cfg
        self.persona_mgr = persona_mgr
        self.world = world

    def update_config(self, cfg: GuardianConfig):
        self.cfg = cfg

    def _is_local_ip(self, ip: str) -> bool:
        if not ip:
            return True
        for prefix in self.cfg.local_ip_prefixes:
            if ip.startswith(prefix):
                return True
        return False

    def evaluate(self, snap: TelemetrySnapshot, connections: List[Dict[str, Any]]) -> List[Alert]:
        alerts: List[Alert] = []
        now = snap.timestamp

        adj_anom = self.persona_mgr.adjust_thresholds(snap.anomaly_score, self.world)

        for p in snap.top_processes:
            name = (p["name"] or "").lower()
            if p["cpu"] > 50.0 and all(allowed.lower() != name for allowed in self.cfg.allowlisted_programs):
                alerts.append(Alert(
                    timestamp=now,
                    level="WARN",
                    message=f"High CPU by non-allowlisted process: {p['name']} (PID {p['pid']})",
                    details=p,
                    category="resource"
                ))

        for c in connections:
            raddr = c["raddr"]
            if not raddr:
                continue
            try:
                ip, port_str = raddr.split(":")
                port = int(port_str)
            except Exception:
                continue

            if port in self.cfg.suspicious_ports and not self._is_local_ip(ip):
                alerts.append(Alert(
                    timestamp=now,
                    level="WARN",
                    message=f"Suspicious outbound connection to {raddr} (PID {c['pid']})",
                    details=c,
                    category="network"
                ))

        if adj_anom > 3.0:
            alerts.append(Alert(
                timestamp=now,
                level="WARN",
                message=f"Anomaly score high: {snap.anomaly_score:.2f} (persona-adjusted {adj_anom:.2f})",
                details={"anomaly_score": snap.anomaly_score, "persona_adj": adj_anom},
                category="anomaly"
            ))

        # Local risk classification
        if snap.anomaly_score < 2.0 and self.world.exfil_events == 0:
            self.world.local_risk = "low"
        elif snap.anomaly_score < 4.0 and self.world.exfil_events < 3:
            self.world.local_risk = "medium"
        else:
            self.world.local_risk = "high"

        return alerts


# ====== PLUGIN SYSTEM =========================================================

class Plugin(Protocol):
    def on_snapshot(self, world: WorldState, snap: TelemetrySnapshot, conns: List[Dict[str, Any]]) -> List[Alert]:
        ...


# ====== ETW AGENT CLIENT ======================================================

class ETWAgentClient(threading.Thread):
    def __init__(self, host: str, port: int, on_event, max_failures: int = 3):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.on_event = on_event
        self.stopped = False
        self.max_failures = max_failures
        self.failures = 0

    def run(self):
        while not self.stopped:
            if self.failures >= self.max_failures:
                print(f"[ETW] Auto-disabled after {self.failures} failed connection attempts.")
                return
            try:
                print(f"[ETW] Connecting to ETW agent at {self.host}:{self.port} ...")
                with socket.create_connection((self.host, self.port), timeout=5.0) as s:
                    print("[ETW] Connected.")
                    self.failures = 0
                    s_file = s.makefile("r", encoding="utf-8")
                    for line in s_file:
                        if self.stopped:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except Exception:
                            continue
                        self.on_event(evt)
            except OSError as e:
                self.failures += 1
                print(f"[ETW] Connection failed ({e}). Attempt {self.failures}/{self.max_failures}.")
                time.sleep(3.0)
            except Exception as e:
                self.failures += 1
                print(f"[ETW] Unexpected error ({e}). Attempt {self.failures}/{self.max_failures}.")
                time.sleep(3.0)

    def stop(self):
        self.stopped = True


# ====== WINDIVERT CLIENT ======================================================

class WinDivertClient(threading.Thread):
    def __init__(self, host: str, port: int, on_packet, max_failures: int = 3):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.on_packet = on_packet
        self.stopped = False
        self.max_failures = max_failures
        self.failures = 0

    def run(self):
        while not self.stopped:
            if self.failures >= self.max_failures:
                print(f"[WinDivert] Auto-disabled after {self.failures} failed connection attempts.")
                return
            try:
                print(f"[WinDivert] Connecting to helper at {self.host}:{self.port} ...")
                with socket.create_connection((self.host, self.port), timeout=5.0) as s:
                    print("[WinDivert] Connected.")
                    self.failures = 0
                    s_file = s.makefile("r", encoding="utf-8")
                    for line in s_file:
                        if self.stopped:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            pkt = json.loads(line)
                        except Exception:
                            continue
                        self.on_packet(pkt)
            except OSError as e:
                self.failures += 1
                print(f"[WinDivert] Connection failed ({e}). Attempt {self.failures}/{self.max_failures}.")
                time.sleep(3.0)
            except Exception as e:
                self.failures += 1
                print(f"[WinDivert] Unexpected error ({e}). Attempt {self.failures}/{self.max_failures}.")
                time.sleep(3.0)

    def stop(self):
        self.stopped = True


# ====== USB WATCHER PLUGIN ====================================================

class USBWatcherPlugin:
    def __init__(self):
        self.known_devices = set()

    def on_snapshot(self, world: WorldState, snap: TelemetrySnapshot, conns: List[Dict[str, Any]]) -> List[Alert]:
        alerts: List[Alert] = []
        now = snap.timestamp
        current = set()

        try:
            for p in psutil.disk_partitions(all=False):
                if "removable" in p.opts.lower() or p.device.lower().startswith("\\\\.\\physicaldrive"):
                    current.add(p.device)
        except Exception:
            return alerts

        new_devices = current - self.known_devices
        removed = self.known_devices - current

        for d in new_devices:
            alerts.append(Alert(
                timestamp=now,
                level="INFO",
                message=f"New USB/removable drive detected: {d}",
                details={"device": d},
                category="usb"
            ))

        for d in removed:
            alerts.append(Alert(
                timestamp=now,
                level="INFO",
                message=f"USB/removable drive removed: {d}",
                details={"device": d},
                category="usb"
            ))

        self.known_devices = current
        return alerts


# ====== SHELL WATCHER / EXFIL DETECTOR =======================================

class ShellWatcherPlugin:
    def __init__(self, payload_protector: PayloadProtector, ml_wrapper: MLModelWrapper):
        self.last_seen_cmd: Dict[int, str] = {}
        self.payload_protector = payload_protector
        self.ml_wrapper = ml_wrapper

    def _looks_like_personal_data(self, cmdline: str) -> bool:
        tokens = cmdline.lower()
        patterns = [
            "phone", "ssn", "social", "biometric", "macaddress",
            "export-csv", "select-object", "get-wmiobject", "get-ciminstance"
        ]
        return any(p in tokens for p in patterns)

    def _looks_like_exfil(self, scores: Dict[str, float]) -> bool:
        return scores.get("exfil", 0.0) > 0.5

    def on_snapshot(self, world: WorldState, snap: TelemetrySnapshot, conns: List[Dict[str, Any]]) -> List[Alert]:
        alerts: List[Alert] = []
        now = snap.timestamp

        for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            name = (proc.info.get("name") or "").lower()
            if name not in ("powershell.exe", "pwsh.exe", "cmd.exe"):
                continue
            pid = proc.info.get("pid")
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
            except Exception:
                cmdline = ""

            if not cmdline:
                continue

            prev = self.last_seen_cmd.get(pid)
            if prev == cmdline:
                continue

            self.last_seen_cmd[pid] = cmdline

            features = {
                "cmdline": cmdline,
                "proc_name": name,
            }
            ai_scores = self.ml_wrapper.classify(features)
            is_exfil = self._looks_like_exfil(ai_scores)
            is_personal = self._looks_like_personal_data(cmdline)

            if not is_exfil:
                alerts.append(Alert(
                    timestamp=now,
                    level="INFO",
                    message=f"Shell activity ({name}, PID {pid})",
                    details={"cmdline": cmdline, "ai_scores": ai_scores},
                    category="shell"
                ))
                continue

            if is_personal:
                fake_payload = b"SIMULATED_PERSONAL_DATA"
                scrambled = self.payload_protector.encrypt_and_mask(fake_payload)
                world.exfil_events += 1
                alerts.append(Alert(
                    timestamp=now,
                    level="WARN",
                    message=f"Exfil attempt with personal data scrambled (PID {pid})",
                    details={
                        "cmdline": cmdline,
                        "scramble_mode": "encrypt+mask",
                        "scrambled_len": len(scrambled),
                        "ai_scores": ai_scores,
                    },
                    category="exfil"
                ))
            else:
                fake_payload = b"SIMULATED_GENERIC_DATA"
                scrambled = self.payload_protector.encrypt_and_mask(fake_payload)
                world.exfil_events += 1
                alerts.append(Alert(
                    timestamp=now,
                    level="WARN",
                    message=f"Suspicious exfil attempt masked (PID {pid})",
                    details={
                        "cmdline": cmdline,
                        "scramble_mode": "mask",
                        "scrambled_len": len(scrambled),
                        "ai_scores": ai_scores,
                    },
                    category="exfil"
                ))

        return alerts


# ====== BROWSER UPLOAD WATCHER ===============================================

class BrowserUploadWatcherPlugin:
    def __init__(self):
        self.browser_names = ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe"]

    def on_snapshot(self, world: WorldState, snap: TelemetrySnapshot, conns: List[Dict[str, Any]]) -> List[Alert]:
        alerts: List[Alert] = []
        now = snap.timestamp

        browser_pids = set()
        for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent"]):
            name = (proc.info.get("name") or "").lower()
            if name in self.browser_names and proc.info.get("cpu_percent", 0.0) > 10.0:
                browser_pids.add(proc.info.get("pid"))

        if not browser_pids:
            return alerts

        for c in conns:
            if c["pid"] in browser_pids and c["raddr"]:
                alerts.append(Alert(
                    timestamp=now,
                    level="INFO",
                    message=f"Browser outbound activity (PID {c['pid']}) to {c['raddr']}",
                    details=c,
                    category="browser"
                ))

        return alerts


# ====== TELEMETRY WORKER ======================================================

class TelemetryWorker(QObject):
    snapshot_ready = Signal(object, list)
    stopped = False

    def __init__(self, loop_hz: float):
        super().__init__()
        self.loop_hz = loop_hz
        self._t0 = time.time()

    def run(self):
        dt = 1.0 / max(0.1, self.loop_hz)
        while not self.stopped:
            t_rel = time.time() - self._t0
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            procs = list(psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]))
            procs_sorted = sorted(
                procs,
                key=lambda p: p.info.get("cpu_percent", 0.0),
                reverse=True
            )
            top = []
            for p in procs_sorted[:5]:
                info = p.info
                top.append({
                    "pid": info.get("pid"),
                    "name": info.get("name", "unknown"),
                    "cpu": info.get("cpu_percent", 0.0),
                    "mem": info.get("memory_percent", 0.0),
                })

            conns = []
            try:
                for c in psutil.net_connections(kind="inet"):
                    laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else ""
                    raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else ""
                    conns.append({
                        "pid": c.pid,
                        "laddr": laddr,
                        "raddr": raddr,
                        "status": c.status,
                    })
            except Exception:
                pass

            snap = TelemetrySnapshot(
                timestamp=t_rel,
                cpu_percent=cpu,
                ram_percent=ram,
                process_count=len(procs),
                connections=len(conns),
                top_processes=top,
            )
            self.snapshot_ready.emit(snap, conns)
            time.sleep(dt)


# ====== SWARM CRYPTO ==========================================================

class SwarmCrypto:
    def __init__(self, key_path: str):
        self.key_path = key_path
        self.key = self._load_or_create_key()
        self.fernet = Fernet(self.key)

    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_path, "wb") as f:
            f.write(key)
        return key

    def encrypt(self, payload: Dict[str, Any]) -> bytes:
        data = json.dumps(payload).encode("utf-8")
        return self.fernet.encrypt(data)

    def decrypt(self, token: bytes) -> Optional[Dict[str, Any]]:
        try:
            data = self.fernet.decrypt(token)
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None


# ====== SWARM CONSENSUS ENGINE v2 ============================================

class SwarmConsensusEngine:
    """
    v2:
      - Tracks peer states
      - Weighted by recency
      - Computes network_risk with quorum
    """

    def __init__(self, world: WorldState):
        self.world = world

    def update_peer_state(self, node: str, state: Dict[str, Any]):
        self.world.consensus_view[node] = state

    def compute_network_risk(self) -> str:
        now = time.time()
        weights = []
        risks = []
        for node, st in self.world.consensus_view.items():
            ts = st.get("ts", now)
            age = max(1.0, now - ts)
            w = 1.0 / age
            local_risk = st.get("local_risk", st.get("local_risk", "low"))
            risks.append(local_risk)
            weights.append(w)

        if not risks:
            return "low"

        score = 0.0
        total_w = sum(weights)
        for r, w in zip(risks, weights):
            if r == "low":
                score += 0.0 * w
            elif r == "medium":
                score += 0.5 * w
            else:
                score += 1.0 * w

        avg = score / total_w
        if avg < 0.25:
            return "low"
        if avg < 0.7:
            return "medium"
        return "high"


# ====== SWARM SYNC (ENCRYPTED, CONTROL + OPERATOR) ============================

class SwarmSync(threading.Thread):
    def __init__(self, cfg: GuardianConfig, world: WorldState, crypto: SwarmCrypto, on_control_cmd, on_operator_cmd, consensus: SwarmConsensusEngine):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.world = world
        self.crypto = crypto
        self.on_control_cmd = on_control_cmd
        self.on_operator_cmd = on_operator_cmd
        self.consensus = consensus
        self.stopped = False

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.sock.bind(("", self.cfg.swarm_port))
            print(f"[Swarm] Bound UDP multicast on port {self.cfg.swarm_port}")
        except OSError as e:
            print(f"[Swarm] Failed to bind port {self.cfg.swarm_port} ({e}). Swarm disabled.")
            self.sock = None
            self.stopped = True
            return

        mreq = socket.inet_aton(self.cfg.swarm_group) + socket.inet_aton("0.0.0.0")
        try:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            print(f"[Swarm] Joined multicast group {self.cfg.swarm_group}")
        except OSError as e:
            print(f"[Swarm] Failed to join multicast group {self.cfg.swarm_group} ({e}). Swarm disabled.")
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            self.stopped = True

    def run(self):
        if not self.sock:
            return

        self.sock.settimeout(1.0)
        last_broadcast = 0.0

        while not self.stopped:
            now = time.time()
            if now - last_broadcast > 3.0:
                msg = {
                    "type": "status",
                    "node": self.cfg.node_name,
                    "ts": now,
                    "exfil_events": self.world.exfil_events,
                    "persona": self.world.persona,
                    "local_risk": self.world.local_risk,
                }
                token = self.crypto.encrypt(msg)
                try:
                    self.sock.sendto(token, (self.cfg.swarm_group, self.cfg.swarm_port))
                except OSError as e:
                    print(f"[Swarm] Send failed ({e}).")
                last_broadcast = now

            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                self.world.swarm_peers = len(self.world.peers)
                continue
            except OSError as e:
                print(f"[Swarm] Receive failed ({e}). Stopping swarm.")
                break

            payload = self.crypto.decrypt(data)
            if not payload:
                continue

            ptype = payload.get("type")
            if ptype == "status":
                node = payload.get("node")
                if node and node != self.cfg.node_name:
                    self.world.peers[node] = payload.get("ts", time.time())
                    self.world.swarm_peers = len(self.world.peers)
                    self.consensus.update_peer_state(node, payload)
                    self.world.network_risk = self.consensus.compute_network_risk()
            elif ptype == "control":
                self.on_control_cmd(payload)
            elif ptype == "operator":
                self.on_operator_cmd(payload)

    def send_operator_command(self, cmd: str, args: Dict[str, Any]):
        msg = {
            "type": "operator",
            "cmd": cmd,
            "args": args,
            "from": self.cfg.node_name,
            "ts": time.time(),
        }
        token = self.crypto.encrypt(msg)
        try:
            self.sock.sendto(token, (self.cfg.swarm_group, self.cfg.swarm_port))
        except Exception as e:
            print(f"[Swarm] Failed to send operator command: {e}")

    def stop(self):
        self.stopped = True
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


# ====== WATCHDOG =============================================================

class Watchdog(threading.Thread):
    def __init__(self, guardian: "GuardianApp"):
        super().__init__(daemon=True)
        self.guardian = guardian
        self.stopped = False

    def run(self):
        while not self.stopped:
            time.sleep(3.0)
            if not self.guardian.worker_thread.is_alive():
                print("[Watchdog] Telemetry worker died, respawning.")
                self.guardian.spawn_worker()

    def stop(self):
        self.stopped = True


# ====== STATUS WINDOW =========================================================

class StatusWindow(QWidget):
    def __init__(self, guardian: "GuardianApp"):
        super().__init__()
        self.guardian = guardian

        self.setWindowTitle("Guardian Sentinel")
        self.setFixedSize(400, 200)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)

        self.setStyleSheet("""
            QWidget {
                background-color: #050608;
            }
            QLabel {
                color: #00eaff;
                font-family: Consolas, monospace;
                font-size: 11pt;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.status_label = QLabel("STATUS: ACTIVE")
        self.metrics_label = QLabel("CPU: ---% | RAM: ---% | ANOM: --")
        self.event_label = QLabel("LAST EVENT: Idle")
        self.swarm_label = QLabel("SWARM PEERS: 0")
        self.persona_label = QLabel("PERSONA: balanced")
        self.risk_label = QLabel("RISK: local=low, net=low")

        layout.addWidget(self.status_label)
        layout.addWidget(self.metrics_label)
        layout.addWidget(self.event_label)
        layout.addWidget(self.swarm_label)
        layout.addWidget(self.persona_label)
        layout.addWidget(self.risk_label)

    def update_status(self, snap: Optional[TelemetrySnapshot], world: WorldState, alert: Optional[Alert]):
        if snap:
            self.metrics_label.setText(
                f"CPU: {snap.cpu_percent:.0f}% | RAM: {snap.ram_percent:.0f}% | ANOM: {snap.anomaly_score:.2f}"
            )
        else:
            self.metrics_label.setText("CPU: ---% | RAM: ---% | ANOM: --")

        self.swarm_label.setText(f"SWARM PEERS: {world.swarm_peers}")
        self.persona_label.setText(f"PERSONA: {world.persona}")
        self.risk_label.setText(f"RISK: local={world.local_risk}, net={world.network_risk}")

        if alert:
            world.last_event_text = alert.message
            world.last_level = alert.level

        level = world.last_level
        msg = world.last_event_text

        if level == "WARN":
            color = "#ffcc00"
        elif level == "ERROR":
            color = "#ff5555"
        elif level == "INFO":
            color = "#00eaff"
        else:
            color = "#00ff88"

        self.event_label.setText(f"LAST EVENT: {msg}")
        self.event_label.setStyleSheet(f"color: {color}; font-family: Consolas, monospace; font-size: 11pt;")
        self.status_label.setText("STATUS: ACTIVE")


# ====== CLI SERVER (AUTO-PORT-FALLBACK) ======================================

class CLIServer(threading.Thread):
    def __init__(self, port: int, guardian: "GuardianApp"):
        super().__init__(daemon=True)
        self.base_port = port
        self.port = port
        self.guardian = guardian
        self.stopped = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        bound = False
        for offset in range(0, 11):
            try_port = self.base_port + offset
            try:
                self.sock.bind(("127.0.0.1", try_port))
                self.port = try_port
                bound = True
                print(f"[CLI] Bound on 127.0.0.1:{self.port}")
                break
            except OSError as e:
                print(f"[CLI] Failed to bind 127.0.0.1:{try_port} ({e}). Trying next port...")
        if not bound:
            print("[CLI] Failed to bind any CLI port. CLI disabled.")
            self.sock = None
            self.stopped = True
            return

        self.sock.listen(5)

    def run(self):
        if not self.sock:
            return

        while not self.stopped:
            try:
                client, addr = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self.handle_client, args=(client,), daemon=True).start()

    def handle_client(self, client: socket.socket):
        with client:
            f = client.makefile("rwb")
            f.write(f"Guardian CLI ready on port {self.port}.\n".encode("utf-8"))
            f.flush()
            for line in f:
                cmd = line.decode("utf-8", errors="ignore").strip()
                if not cmd:
                    continue
                if cmd == "status":
                    resp = self.guardian.cli_status()
                elif cmd.startswith("alerts"):
                    parts = cmd.split()
                    n = 10
                    if len(parts) > 1:
                        try:
                            n = int(parts[1])
                        except ValueError:
                            pass
                    resp = self.guardian.cli_alerts(n)
                elif cmd == "swarm":
                    resp = self.guardian.cli_swarm()
                elif cmd.startswith("persona"):
                    parts = cmd.split()
                    if len(parts) > 1:
                        resp = self.guardian.cli_set_persona(parts[1])
                    else:
                        resp = "Usage: persona balanced|paranoid|relaxed|auto\n"
                elif cmd == "risk":
                    resp = self.guardian.cli_risk()
                elif cmd.startswith("op "):
                    # send operator command over swarm
                    parts = cmd.split(maxsplit=2)
                    if len(parts) >= 2:
                        op_cmd = parts[1]
                        args = {}
                        if len(parts) == 3:
                            args["msg"] = parts[2]
                        resp = self.guardian.cli_operator(op_cmd, args)
                    else:
                        resp = "Usage: op <cmd> [msg]\n"
                elif cmd == "quit":
                    resp = "Shutting down node.\n"
                    f.write(resp.encode("utf-8"))
                    f.flush()
                    self.guardian.quit_app()
                    return
                else:
                    resp = "Unknown command.\n"
                f.write(resp.encode("utf-8"))
                f.flush()

    def stop(self):
        self.stopped = True
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


# ====== MAIN APP LOGIC ========================================================

class GuardianApp(QObject):
    def __init__(self, app: QApplication):
        super().__init__()
        self.qt_app = app
        self.profile_manager = ProfileManager()
        self.cfg = self.profile_manager.get_active_config()
        self.world = WorldState(node_id=self.cfg.node_name, persona=self.cfg.persona)
        self.swarm_crypto = SwarmCrypto(self.cfg.swarm_key_path)
        self.world.swarm_key = self.swarm_crypto.key

        self.persona_mgr = PersonaManager(self.cfg.persona)
        self.anomaly = AnomalyDetector()
        self.policy = PolicyEngine(self.cfg, self.persona_mgr, self.world)
        self.logger = EncryptedLogger()
        self.payload_protector = PayloadProtector()
        self.ml_wrapper = MLModelWrapper(self.cfg.ml_model_path)
        self.threat_chain = ThreatChainPredictor()
        self.consensus = SwarmConsensusEngine(self.world)

        self.status_window = StatusWindow(self)
        self.status_window.show()

        self.tray = QSystemTrayIcon(QIcon(), self.status_window)
        self.tray.setToolTip("Guardian Sentinel (ON)")
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show Status")
        quit_action = tray_menu.addAction("Quit")
        show_action.triggered.connect(self.show_status)
        quit_action.triggered.connect(self.quit_app)
        self.tray.setContextMenu(tray_menu)
        self.tray.show()

        self.plugins: List[Plugin] = [
            USBWatcherPlugin(),
            ShellWatcherPlugin(self.payload_protector, self.ml_wrapper),
            BrowserUploadWatcherPlugin(),
        ]

        self.worker = None
        self.worker_thread = None
        self.spawn_worker()

        self.etw_client = ETWAgentClient(self.cfg.etw_agent_host, self.cfg.etw_agent_port, self.on_etw_event)
        self.etw_client.start()

        self.windivert_client = WinDivertClient(self.cfg.windivert_host, self.cfg.windivert_port, self.on_packet_event)
        self.windivert_client.start()

        self.swarm: Optional[SwarmSync] = None
        if self.cfg.swarm_enabled:
            self.swarm = SwarmSync(
                self.cfg,
                self.world,
                self.swarm_crypto,
                self.on_swarm_control,
                self.on_swarm_operator,
                self.consensus
            )
            self.swarm.start()

        self.watchdog = Watchdog(self)
        self.watchdog.start()

        self.cli_server = CLIServer(self.cfg.cli_port, self)
        self.cli_server.start()

        self._latest_snap: Optional[TelemetrySnapshot] = None
        self._latest_conns: List[Dict[str, Any]] = []

    def spawn_worker(self):
        if self.worker is not None:
            self.worker.stopped = True
        self.worker = TelemetryWorker(self.cfg.loop_hz)
        self.worker.snapshot_ready.connect(self.on_snapshot)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()
        print("[Core] Telemetry worker started.")

    def on_snapshot(self, snap: TelemetrySnapshot, conns: list):
        snap.anomaly_score = self.anomaly.update_and_score(snap.cpu_percent, snap.ram_percent)
        self._latest_snap = snap
        self._latest_conns = conns
        self.world.last_snapshot = snap

        alerts = self.policy.evaluate(snap, conns)

        for plugin in self.plugins:
            try:
                alerts.extend(plugin.on_snapshot(self.world, snap, conns))
            except Exception as e:
                alerts.append(Alert(
                    timestamp=snap.timestamp,
                    level="ERROR",
                    message=f"Plugin error: {e}",
                    details={},
                    category="internal"
                ))

        if not alerts:
            return

        for a in alerts:
            self._record_alert(a)

        last_alert = alerts[-1]
        self.status_window.update_status(self._latest_snap, self.world, last_alert)

        if len(self.world.alerts) >= 2:
            prev_cat = self.world.alerts[-2].category
            self.threat_chain.observe(prev_cat, last_alert.category)

    def _record_alert(self, alert: Alert):
        self.world.alerts.append(alert)
        self.logger.log_alert(alert)
        self._notify_tray(alert)

    def _notify_tray(self, alert: Alert):
        title = f"Guardian: {alert.level}"
        msg = alert.message
        self.tray.showMessage(title, msg, QSystemTrayIcon.Information, 5000)

    def on_etw_event(self, evt: Dict[str, Any]):
        now = time.time()
        etype = evt.get("type", "etw")
        msg = f"ETW event: {etype}"
        alert = Alert(timestamp=now, level="INFO", message=msg, details=evt, category="etw")
        self._record_alert(alert)
        self.status_window.update_status(self._latest_snap, self.world, alert)

    def on_packet_event(self, pkt: Dict[str, Any]):
        now = time.time()
        msg = f"Packet event: {pkt.get('src')} -> {pkt.get('dst')}"
        alert = Alert(timestamp=now, level="INFO", message=msg, details=pkt, category="packet")
        self._record_alert(alert)
        self.status_window.update_status(self._latest_snap, self.world, alert)

    def on_swarm_control(self, payload: Dict[str, Any]):
        cmd = payload.get("cmd")
        args = payload.get("args", {})
        now = time.time()
        if cmd == "status":
            alert = Alert(timestamp=now, level="INFO", message="Remote status requested", details=payload, category="swarm")
            self._record_alert(alert)
        elif cmd == "set_persona":
            persona = args.get("persona")
            if persona in ("balanced", "paranoid", "relaxed", "auto"):
                self.persona_mgr.set_persona(persona)
                self.world.persona = persona
                alert = Alert(timestamp=now, level="INFO", message=f"Persona changed remotely to {persona}", details=payload, category="swarm")
                self._record_alert(alert)

    def on_swarm_operator(self, payload: Dict[str, Any]):
        cmd = payload.get("cmd")
        args = payload.get("args", {})
        now = time.time()
        if cmd == "ping":
            alert = Alert(timestamp=now, level="INFO", message="Remote operator ping", details=payload, category="operator")
            self._record_alert(alert)
        elif cmd == "request_risk":
            risk = self.consensus.compute_network_risk()
            alert = Alert(timestamp=now, level="INFO", message=f"Remote operator requested risk: {risk}", details={"risk": risk}, category="operator")
            self._record_alert(alert)
        elif cmd == "broadcast_message":
            msg = args.get("msg", "")
            alert = Alert(timestamp=now, level="INFO", message=f"Operator broadcast: {msg}", details=payload, category="operator")
            self._record_alert(alert)

    def cli_status(self) -> str:
        snap = self.world.last_snapshot
        if not snap:
            return "No snapshot yet.\n"
        next_cat = self.threat_chain.predict_next(self.world.alerts[-1].category) if self.world.alerts else None
        return (
            f"STATUS: ACTIVE\n"
            f"CPU: {snap.cpu_percent:.0f}% | RAM: {snap.ram_percent:.0f}% | ANOM: {snap.anomaly_score:.2f}\n"
            f"EXFIL EVENTS: {self.world.exfil_events}\n"
            f"SWARM PEERS: {self.world.swarm_peers}\n"
            f"PERSONA: {self.world.persona}\n"
            f"RISK: local={self.world.local_risk}, net={self.world.network_risk}\n"
            f"NEXT LIKELY ALERT CATEGORY: {next_cat or 'unknown'}\n"
        )

    def cli_alerts(self, n: int) -> str:
        last = self.world.alerts[-n:]
        lines = []
        for a in last:
            lines.append(f"{a.timestamp:.3f} [{a.level}] ({a.category}) {a.message}")
        return "\n".join(lines) + "\n"

    def cli_swarm(self) -> str:
        lines = [f"Peers: {len(self.world.peers)}"]
        for node, ts in self.world.peers.items():
            lines.append(f" - {node} (last: {ts:.3f})")
        return "\n".join(lines) + "\n"

    def cli_set_persona(self, persona: str) -> str:
        if persona not in ("balanced", "paranoid", "relaxed", "auto"):
            return "Invalid persona. Use: balanced|paranoid|relaxed|auto\n"
        self.persona_mgr.set_persona(persona)
        self.world.persona = persona
        return f"Persona set to {persona}\n"

    def cli_risk(self) -> str:
        risk = self.consensus.compute_network_risk()
        self.world.network_risk = risk
        return f"Network risk (from swarm consensus): {risk}\n"

    def cli_operator(self, cmd: str, args: Dict[str, Any]) -> str:
        if not self.swarm:
            return "Swarm disabled.\n"
        self.swarm.send_operator_command(cmd, args)
        return f"Operator command '{cmd}' sent.\n"

    def show_status(self):
        self.status_window.showNormal()
        self.status_window.raise_()
        self.status_window.activateWindow()

    def quit_app(self):
        self.stop()
        self.qt_app.quit()

    def stop(self):
        if self.worker:
            self.worker.stopped = True
        if hasattr(self, "etw_client") and self.etw_client:
            self.etw_client.stop()
        if hasattr(self, "windivert_client") and self.windivert_client:
            self.windivert_client.stop()
        if self.swarm:
            self.swarm.stop()
        if self.watchdog:
            self.watchdog.stop()
        if self.cli_server:
            self.cli_server.stop()


# ====== ENTRYPOINT ============================================================

def main():
    app = QApplication(sys.argv)
    guardian = GuardianApp(app)
    try:
        sys.exit(app.exec())
    finally:
        guardian.stop()


if __name__ == "__main__":
    main()
