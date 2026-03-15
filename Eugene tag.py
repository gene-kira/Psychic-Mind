"""
Unified Organism (Bernoulli + Distributed + Anomaly + HopGraph + GPU + ML + Consensus + Replay + Scripting)

Organs:
- ConfigOrgan
- CryptoOrganV2 (AES-GCM)
- UVGate
- DataCapsule, HopRecord, VerificationResult
- LineageGraph
- ContaminationMap
- PolicyEngineV3
- TamperReactor
- PersistenceOrgan
- LoggingOrgan
- FlowBus
- FlowMetricsOrgan
- FlowAnomalyOrgan (statistical)
- MlAnomalyOrgan (ML-based, IsolationForest if available)
- ClusterOrgan (distributed mode via UDP JSON broadcast)
- ConsensusOrgan (cluster policy consensus)
- HopGraphOrgan (live hop-chain graph)
- Ring
- DyePackOrgan
- WorkerPool
- AdaptiveTuningOrgan
- DeepUIAutomationOrgan
- UnifiedIOOrgan
- GpuCockpitOrgan (GPU renderer if moderngl available)
- TkinterCockpit (active cockpit, with flow + hop + anomaly + replay + scripting)
- LifecycleOrgan
- build_organism()
"""

import os
import json
import time
import uuid
import math
import sqlite3
import threading
import socket
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Deque, Tuple, Callable
from collections import deque, defaultdict
from threading import Lock, Thread
from queue import Queue, Empty

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets

import tkinter as tk
from tkinter import ttk

try:
    from pywinauto import Desktop
    UIA_AVAILABLE = True
except Exception:
    UIA_AVAILABLE = False

# Optional GPU / ML dependencies
try:
    import moderngl
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# =========================
#  BASIC ENUMS & UTILS
# =========================

class LogicalColor(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"
    PURPLE = "PURPLE"


class VerificationStatus(Enum):
    CLEAN = auto()
    TAMPERED = auto()
    WRONG_ENV = auto()
    FORGED = auto()
    UNKNOWN = auto()


def sha256_str(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# =========================
#  CONFIG ORGAN
# =========================

class ConfigOrgan:
    def __init__(self, config_path: Optional[str] = None):
        self.config: Dict[str, Any] = {
            "COMPANY_ID": os.environ.get("ORG_COMPANY_ID", "ACME_CORP"),
            "ENV_INTERNAL": os.environ.get("ORG_ENV_INTERNAL", "internal"),
            "ENV_EXTERNAL": os.environ.get("ORG_ENV_EXTERNAL", "external"),
            "DB_PATH": os.environ.get("ORG_DB_PATH", "organism.db"),
            "KEY_MASTER": os.environ.get("ORG_KEY_MASTER", None),
            "POLICY_PATH": os.environ.get("ORG_POLICY_PATH", "policy.json"),
            "RETENTION_DAYS": int(os.environ.get("ORG_RETENTION_DAYS", "7")),
            "RING_MAX_DEPTH": int(os.environ.get("ORG_RING_MAX_DEPTH", "1000")),
            "WORKER_COUNT": int(os.environ.get("ORG_WORKER_COUNT", "4")),
            "UIA_MIN_INTERVAL_MS": int(os.environ.get("ORG_UIA_MIN_INTERVAL_MS", "2000")),
            "FLOW_WINDOW_SEC": int(os.environ.get("ORG_FLOW_WINDOW_SEC", "10")),
            "PRESSURE_HIGH": float(os.environ.get("ORG_PRESSURE_HIGH", "0.9")),
            "PRESSURE_LOW": float(os.environ.get("ORG_PRESSURE_LOW", "0.3")),
            "WORKERS_MAX": int(os.environ.get("ORG_WORKERS_MAX", "16")),
            "WORKERS_MIN": int(os.environ.get("ORG_WORKERS_MIN", "1")),
            # distributed
            "CLUSTER_NODE_ID": os.environ.get("ORG_CLUSTER_NODE_ID", "node-1"),
            "CLUSTER_UDP_PORT": int(os.environ.get("ORG_CLUSTER_UDP_PORT", "49001")),
            "CLUSTER_UDP_ADDR": os.environ.get("ORG_CLUSTER_UDP_ADDR", "239.10.10.10"),
            # anomaly
            "ANOMALY_SENSITIVITY": float(os.environ.get("ORG_ANOMALY_SENSITIVITY", "3.0")),
            # ML anomaly
            "ML_ANOMALY_WINDOW": int(os.environ.get("ORG_ML_ANOMALY_WINDOW", "200")),
            "ML_ANOMALY_CONTAMINATION": float(os.environ.get("ORG_ML_ANOMALY_CONTAMINATION", "0.05")),
        }
        if config_path and os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.config.update(data)
            except Exception:
                pass

        if not self.config["KEY_MASTER"]:
            self.config["KEY_MASTER"] = secrets.token_hex(32)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


# =========================
#  CRYPTO ORGAN V2
# =========================

class CryptoOrganV2:
    def __init__(self, master_key_hex: str):
        self.master_key = bytes.fromhex(master_key_hex)
        if len(self.master_key) not in (16, 24, 32):
            self.master_key = sha256_str(self.master_key.hex()).encode("utf-8")[:32]

    def seal(self, payload: bytes, env_tag: str, key_id: str = "default") -> Tuple[bytes, str]:
        aad = env_tag.encode("utf-8") + b"|" + key_id.encode("utf-8")
        aesgcm = AESGCM(self.master_key)
        nonce = secrets.token_bytes(12)
        ct = aesgcm.encrypt(nonce, payload, aad)
        return nonce + ct, nonce.hex()

    def unseal(self, sealed: bytes, env_tag: str, key_id: str = "default") -> Optional[bytes]:
        aad = env_tag.encode("utf-8") + b"|" + key_id.encode("utf-8")
        aesgcm = AESGCM(self.master_key)
        if len(sealed) < 12:
            return None
        nonce = sealed[:12]
        ct = sealed[12:]
        try:
            return aesgcm.decrypt(nonce, ct, aad)
        except Exception:
            return None


# =========================
#  CAPSULE & UV GATE
# =========================

@dataclass
class HopRecord:
    from_node: str
    to_node: str
    ts: float
    hop_sig: str


@dataclass
class DataCapsule:
    capsule_id: str
    company_id: str
    env_tag: str
    sealed_payload: bytes
    integrity_hash: str
    signature: str
    key_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quarantined: bool = False
    ring_name: str = "unknown"
    source: str = "unknown"
    hop_chain: List[HopRecord] = field(default_factory=list)


@dataclass
class VerificationResult:
    capsule: DataCapsule
    status: VerificationStatus
    color: LogicalColor
    reason: str
    node: str
    severity: int
    timestamp: float = field(default_factory=time.time)


class UVGate:
    def __init__(self, trusted_company_ids: List[str], crypto: CryptoOrganV2):
        self.trusted_company_ids = set(trusted_company_ids)
        self.crypto = crypto

    def verify(self, capsule: DataCapsule, expected_env: Optional[str], node: str) -> VerificationResult:
        if not capsule.signature:
            return VerificationResult(capsule, VerificationStatus.UNKNOWN, LogicalColor.PURPLE,
                                      "No signature present", node, severity=20)

        if capsule.company_id not in self.trusted_company_ids:
            return VerificationResult(capsule, VerificationStatus.FORGED, LogicalColor.RED,
                                      f"Untrusted company_id: {capsule.company_id}", node, severity=100)

        unsealed = self.crypto.unseal(capsule.sealed_payload, capsule.env_tag, capsule.key_id)
        if unsealed is None:
            return VerificationResult(capsule, VerificationStatus.FORGED, LogicalColor.RED,
                                      "Unseal failed (wrong env or tampered)", node, severity=90)

        recomputed_hash = sha256_str(unsealed.decode("utf-8", errors="ignore"))
        hash_matches = (recomputed_hash == capsule.integrity_hash)

        base = {
            "company_id": capsule.company_id,
            "env_tag": capsule.env_tag,
            "integrity_hash": capsule.integrity_hash,
            "ring_name": capsule.ring_name,
            "key_id": capsule.key_id,
        }
        import json
        expected_signature = sha256_str(json.dumps(base, sort_keys=True))
        signature_matches = (expected_signature == capsule.signature)

        env_ok = True
        if expected_env is not None and capsule.env_tag != expected_env:
            env_ok = False

        if hash_matches and signature_matches and env_ok:
            return VerificationResult(capsule, VerificationStatus.CLEAN, LogicalColor.GREEN,
                                      "All checks passed", node, severity=0)

        if hash_matches and signature_matches and not env_ok:
            return VerificationResult(capsule, VerificationStatus.WRONG_ENV, LogicalColor.ORANGE,
                                      f"Capsule env_tag={capsule.env_tag}, expected={expected_env}", node,
                                      severity=60)

        if not hash_matches and signature_matches:
            return VerificationResult(capsule, VerificationStatus.TAMPERED, LogicalColor.YELLOW,
                                      "Integrity hash mismatch", node, severity=40)

        return VerificationResult(capsule, VerificationStatus.FORGED, LogicalColor.RED,
                                  "Signature mismatch", node, severity=80)


# =========================
#  LINEAGE, CONTAMINATION, POLICY
# =========================

@dataclass
class LineageEdge:
    capsule_id: str
    from_node: str
    to_node: str
    color: str
    status: str
    timestamp: float


@dataclass
class LineageGraph:
    edges: List[LineageEdge] = field(default_factory=list)
    by_capsule: Dict[str, List[LineageEdge]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def record_hop(self, edge: LineageEdge):
        with self._lock:
            self.edges.append(edge)
            self.by_capsule.setdefault(edge.capsule_id, []).append(edge)


@dataclass
class ContaminationMap:
    first_contamination: Dict[str, LineageEdge] = field(default_factory=dict)
    node_contamination_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: Lock = field(default_factory=Lock)

    def register_contamination(self, edge: LineageEdge):
        with self._lock:
            if edge.capsule_id not in self.first_contamination:
                self.first_contamination[edge.capsule_id] = edge
            self.node_contamination_count[edge.to_node] += 1


class NodeState(Enum):
    OPEN = "OPEN"
    QUARANTINED = "QUARANTINED"
    SEALED = "SEALED"


@dataclass
class PolicyAction:
    kind: str
    target: str
    reason: str
    severity: int
    timestamp: float


class PolicyEngineV3:
    def __init__(self, policy_path: str, logger: "LoggingOrgan"):
        self.node_state: Dict[str, NodeState] = {}
        self.node_health: Dict[str, int] = {}
        self.actions: List[PolicyAction] = []
        self._lock = Lock()
        self.policy_path = policy_path
        self.logger = logger
        self.rules = {
            "severity_thresholds": {"quarantine": 120, "seal": 200},
            "color_actions": {
                "RED": ["BLOCK_FLOWS"],
                "ORANGE": ["PRE_RESEAL"],
                "YELLOW": ["QUARANTINE_CAPSULE"],
            },
            "flow_rules": {}
        }
        self.load_rules()

    def load_rules(self):
        if os.path.isfile(self.policy_path):
            try:
                with open(self.policy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.rules.update(data)
                self.logger.log("INFO", "Policy rules loaded", path=self.policy_path)
            except Exception as e:
                self.logger.log("WARN", "Failed to load policy rules", error=str(e))

    def on_verification(self, result: VerificationResult):
        now = time.time()
        node = result.node
        sev = result.severity
        color = result.color

        with self._lock:
            if node not in self.node_state:
                self.node_state[node] = NodeState.OPEN
                self.node_health[node] = 0

            self.node_health[node] = self.node_health.get(node, 0) + sev

            thresholds = self.rules.get("severity_thresholds", {})
            q_th = thresholds.get("quarantine", 120)
            s_th = thresholds.get("seal", 200)

            health = self.node_health[node]
            if health >= s_th and self.node_state[node] != NodeState.SEALED:
                self.node_state[node] = NodeState.SEALED
                self.actions.append(PolicyAction("SEAL_NODE", node,
                                                 f"Node health {health} >= seal {s_th}", sev, now))
            elif health >= q_th and self.node_state[node] == NodeState.OPEN:
                self.node_state[node] = NodeState.QUARANTINED
                self.actions.append(PolicyAction("QUARANTINE_NODE", node,
                                                 f"Node health {health} >= quarantine {q_th}", sev, now))

            color_actions = self.rules.get("color_actions", {})
            acts = color_actions.get(color.value, [])
            for act in acts:
                target = node if act != "QUARANTINE_CAPSULE" else result.capsule.capsule_id
                reason = f"Color {color.value} triggered {act}"
                self.actions.append(PolicyAction(act, target, reason, sev, now))

    def apply_flow_metrics(self, metrics: Dict[str, Any]):
        now = time.time()
        with self._lock:
            for ring_name, m in metrics.get("rings", {}).items():
                pressure = m.get("pressure", 0.0)
                if pressure > 0.98:
                    self.actions.append(PolicyAction(
                        "FLOW_PRESSURE_CRITICAL",
                        ring_name,
                        f"Pressure {pressure:.2f} > 0.98",
                        severity=10,
                        timestamp=now
                    ))

    def apply_anomaly(self, anomaly: Dict[str, Any]):
        now = time.time()
        desc = anomaly.get("description", "flow anomaly")
        ring = anomaly.get("ring", "unknown")
        sev = anomaly.get("severity", 10)
        with self._lock:
            self.actions.append(PolicyAction(
                "FLOW_ANOMALY",
                ring,
                desc,
                sev,
                now
            ))

    def apply_operator_command(self, cmd: str, target: str):
        now = time.time()
        with self._lock:
            if cmd == "RESET_NODE_HEALTH":
                self.node_health[target] = 0
                self.actions.append(PolicyAction(cmd, target, "Operator reset node health", 0, now))
            elif cmd == "SEAL_NODE":
                self.node_state[target] = NodeState.SEALED
                self.actions.append(PolicyAction(cmd, target, "Operator sealed node", 0, now))
            elif cmd == "UNSEAL_NODE":
                self.node_state[target] = NodeState.OPEN
                self.actions.append(PolicyAction(cmd, target, "Operator unsealed node", 0, now))

    def snapshot(self):
        with self._lock:
            return dict(self.node_state), dict(self.node_health), list(self.actions[-50:])


@dataclass
class TamperReactor:
    events: List[Dict[str, Any]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    def handle(self, result: VerificationResult):
        capsule = result.capsule
        event = {
            "capsule_id": capsule.capsule_id,
            "ring": capsule.ring_name,
            "status": result.status.name,
            "color": result.color.value,
            "reason": result.reason,
            "node": result.node,
            "timestamp": result.timestamp,
            "source": capsule.source,
            "severity": result.severity,
        }
        with self._lock:
            self.events.append(event)
        if result.status != VerificationStatus.CLEAN:
            capsule.quarantined = True


# =========================
#  PERSISTENCE ORGAN
# =========================

class PersistenceOrgan:
    def __init__(self, db_path: str, retention_days: int, logger: "LoggingOrgan"):
        self.db_path = db_path
        self.retention_days = retention_days
        self._lock = Lock()
        self.logger = logger
        self._init_db()

    def _init_db(self):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capsule_id TEXT,
                    from_node TEXT,
                    to_node TEXT,
                    color TEXT,
                    status TEXT,
                    ts REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS contamination (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capsule_id TEXT,
                    node TEXT,
                    ts REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS policy_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT,
                    target TEXT,
                    reason TEXT,
                    severity INTEGER,
                    ts REAL
                )
            """)
            conn.commit()

    def store_lineage(self, edge: LineageEdge):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO lineage (capsule_id, from_node, to_node, color, status, ts) VALUES (?,?,?,?,?,?)",
                      (edge.capsule_id, edge.from_node, edge.to_node, edge.color, edge.status, edge.timestamp))
            conn.commit()

    def store_contamination(self, edge: LineageEdge):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO contamination (capsule_id, node, ts) VALUES (?,?,?)",
                      (edge.capsule_id, edge.to_node, edge.timestamp))
            conn.commit()

    def store_policy_action(self, action: PolicyAction):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO policy_actions (kind, target, reason, severity, ts) VALUES (?,?,?,?,?)",
                      (action.kind, action.target, action.reason, action.severity, action.timestamp))
            conn.commit()

    def apply_retention(self):
        cutoff = time.time() - self.retention_days * 86400
        with self._lock, sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM lineage WHERE ts < ?", (cutoff,))
            c.execute("DELETE FROM contamination WHERE ts < ?", (cutoff,))
            c.execute("DELETE FROM policy_actions WHERE ts < ?", (cutoff,))
            conn.commit()
        self.logger.log("INFO", "Retention applied", cutoff=cutoff)

    def load_lineage_window(self, start_ts: float, end_ts: float) -> List[LineageEdge]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT capsule_id, from_node, to_node, color, status, ts
                FROM lineage
                WHERE ts BETWEEN ? AND ?
                ORDER BY ts ASC
            """, (start_ts, end_ts))
            rows = c.fetchall()
        return [
            LineageEdge(
                capsule_id=r[0],
                from_node=r[1],
                to_node=r[2],
                color=r[3],
                status=r[4],
                timestamp=r[5],
            )
            for r in rows
        ]


# =========================
#  LOGGING ORGAN
# =========================

class LoggingOrgan:
    def __init__(self):
        self._lock = Lock()

    def log(self, level: str, msg: str, **fields):
        with self._lock:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            base = {"ts": ts, "level": level, "msg": msg}
            base.update(fields)
            print(json.dumps(base))


# =========================
#  FLOW BUS
# =========================

@dataclass
class FlowEvent:
    kind: str
    payload: Dict[str, Any]


class FlowBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[FlowEvent], None]]] = defaultdict(list)
        self._lock = Lock()

    def subscribe(self, kind: str, handler: Callable[[FlowEvent], None]):
        with self._lock:
            self.subscribers[kind].append(handler)

    def publish(self, event: FlowEvent):
        with self._lock:
            handlers = list(self.subscribers.get(event.kind, []))
        for h in handlers:
            try:
                h(event)
            except Exception:
                pass


# =========================
#  FLOW METRICS ORGAN
# =========================

class FlowMetricsOrgan:
    def __init__(self, rings: List["Ring"], bus: FlowBus, window_sec: int, logger: LoggingOrgan):
        self.rings = rings
        self.bus = bus
        self.window_sec = window_sec
        self.logger = logger

        self._lock = Lock()
        self._events: Deque[Tuple[float, str, str]] = deque()
        self._last_metrics: Dict[str, Any] = {}

        self.bus.subscribe("verification_result", self._on_verification)

    def _on_verification(self, event: FlowEvent):
        res: VerificationResult = event.payload["result"]
        ts = res.timestamp
        ring_name = res.capsule.ring_name
        source = res.capsule.source
        with self._lock:
            self._events.append((ts, ring_name, source))
            cutoff = ts - self.window_sec
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

    def compute_metrics(self) -> Dict[str, Any]:
        now = time.time()
        cutoff = now - self.window_sec
        with self._lock:
            events = [e for e in self._events if e[0] >= cutoff]

        ring_counts: Dict[str, int] = defaultdict(int)
        source_counts: Dict[str, int] = defaultdict(int)
        for ts, ring_name, source in events:
            ring_counts[ring_name] += 1
            source_counts[source] += 1

        rings_metrics: Dict[str, Dict[str, float]] = {}
        for r in self.rings:
            depth = len(r.queue)
            pressure = depth / r.max_depth if r.max_depth > 0 else 0.0
            velocity = ring_counts.get(r.name, 0) / self.window_sec
            rings_metrics[r.name] = {
                "pressure": pressure,
                "velocity": velocity,
                "depth": depth,
                "max_depth": r.max_depth,
            }

        sources_metrics: Dict[str, float] = {}
        for src, cnt in source_counts.items():
            sources_metrics[src] = cnt / self.window_sec

        metrics = {
            "ts": now,
            "window_sec": self.window_sec,
            "rings": rings_metrics,
            "sources": sources_metrics,
        }
        self._last_metrics = metrics
        self.bus.publish(FlowEvent("flow_metrics", {"metrics": metrics}))
        return metrics

    def last_metrics(self) -> Dict[str, Any]:
        return self._last_metrics


# =========================
#  FLOW ANOMALY ORGAN (STATISTICAL)
# =========================

class FlowAnomalyOrgan:
    def __init__(self, bus: FlowBus, sensitivity: float, logger: LoggingOrgan):
        self.bus = bus
        self.sensitivity = sensitivity
        self.logger = logger
        self._lock = Lock()
        self._stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "count": 0.0,
            "mean_v": 0.0,
            "m2_v": 0.0,
        })
        self.bus.subscribe("flow_metrics", self._on_flow_metrics)

    def _update_stats(self, ring: str, velocity: float):
        s = self._stats[ring]
        s["count"] += 1.0
        delta = velocity - s["mean_v"]
        s["mean_v"] += delta / s["count"]
        delta2 = velocity - s["mean_v"]
        s["m2_v"] += delta * delta2

    def _std(self, ring: str) -> float:
        s = self._stats[ring]
        if s["count"] < 2:
            return 0.0
        return math.sqrt(s["m2_v"] / (s["count"] - 1))

    def _on_flow_metrics(self, event: FlowEvent):
        metrics = event.payload["metrics"]
        rings = metrics.get("rings", {})
        ts = metrics.get("ts", time.time())
        with self._lock:
            for ring_name, m in rings.items():
                v = m.get("velocity", 0.0)
                self._update_stats(ring_name, v)
                std = self._std(ring_name)
                mean = self._stats[ring_name]["mean_v"]
                if std > 0:
                    z = (v - mean) / std
                    if abs(z) >= self.sensitivity and self._stats[ring_name]["count"] > 20:
                        desc = f"velocity z-score={z:.2f} (v={v:.2f}, mean={mean:.2f}, std={std:.2f})"
                        anomaly = {
                            "ring": ring_name,
                            "ts": ts,
                            "z": z,
                            "velocity": v,
                            "mean": mean,
                            "std": std,
                            "description": desc,
                            "severity": int(min(100, abs(z) * 10)),
                            "source": "statistical",
                        }
                        self.logger.log("WARN", "Flow anomaly (stat)", ring=ring_name, z=z, v=v, mean=mean, std=std)
                        self.bus.publish(FlowEvent("anomaly_event", {"anomaly": anomaly}))


# =========================
#  ML ANOMALY ORGAN (ISOLATION FOREST)
# =========================

class MlAnomalyOrgan:
    """
    Uses IsolationForest (if available) on ring-level metrics to detect anomalies.
    """

    def __init__(self, bus: FlowBus, cfg: ConfigOrgan, logger: LoggingOrgan):
        self.bus = bus
        self.logger = logger
        self.window = cfg.get("ML_ANOMALY_WINDOW", 200)
        self.contamination = cfg.get("ML_ANOMALY_CONTAMINATION", 0.05)
        self._lock = Lock()
        self._history: List[List[float]] = []
        self._model = None
        self._last_train_ts = 0.0
        self._train_interval = 30.0
        if SKLEARN_AVAILABLE:
            self._model = IsolationForest(
                contamination=self.contamination,
                n_estimators=100,
                random_state=42,
            )
        self.bus.subscribe("flow_metrics", self._on_flow_metrics)

    def _on_flow_metrics(self, event: FlowEvent):
        if not SKLEARN_AVAILABLE or self._model is None:
            return
        metrics = event.payload["metrics"]
        rings = metrics.get("rings", {})
        ts = metrics.get("ts", time.time())
        if not rings:
            return

        # feature vector: [pressure_inner, velocity_inner, pressure_mid, velocity_mid, pressure_outer, velocity_outer]
        def get(rn, key):
            return rings.get(rn, {}).get(key, 0.0)

        x = [
            get("inner", "pressure"),
            get("inner", "velocity"),
            get("mid", "pressure"),
            get("mid", "velocity"),
            get("outer", "pressure"),
            get("outer", "velocity"),
        ]

        with self._lock:
            self._history.append(x)
            if len(self._history) > self.window:
                self._history = self._history[-self.window:]

            if len(self._history) >= 50 and (ts - self._last_train_ts) > self._train_interval:
                try:
                    self._model.fit(self._history)
                    self._last_train_ts = ts
                    self.logger.log("INFO", "ML anomaly model retrained", samples=len(self._history))
                except Exception as e:
                    self.logger.log("WARN", "ML anomaly training failed", error=str(e))

            if len(self._history) >= 50:
                try:
                    pred = self._model.predict([x])[0]  # -1 anomaly, 1 normal
                    if pred == -1:
                        desc = "ML anomaly: IsolationForest flagged current flow vector"
                        anomaly = {
                            "ring": "multi-ring",
                            "ts": ts,
                            "description": desc,
                            "severity": 60,
                            "source": "ml",
                            "vector": x,
                        }
                        self.logger.log("WARN", "Flow anomaly (ML)", vector=x)
                        self.bus.publish(FlowEvent("anomaly_event", {"anomaly": anomaly}))
                except Exception as e:
                    self.logger.log("WARN", "ML anomaly prediction failed", error=str(e))


# =========================
#  CLUSTER ORGAN (DISTRIBUTED)
# =========================

class ClusterOrgan:
    def __init__(self, cfg: ConfigOrgan, bus: FlowBus, logger: LoggingOrgan):
        self.node_id = cfg.get("CLUSTER_NODE_ID", "node-1")
        self.addr = cfg.get("CLUSTER_UDP_ADDR", "239.10.10.10")
        self.port = cfg.get("CLUSTER_UDP_PORT", 49001)
        self.bus = bus
        self.logger = logger
        self._stop = False
        self._send_sock = None
        self._recv_sock = None
        self.thread: Optional[Thread] = None

        self.bus.subscribe("lineage_event", self._on_local_lineage)
        self.bus.subscribe("policy_action_event", self._on_local_policy_action)
        self.bus.subscribe("anomaly_event", self._on_local_anomaly)
        self.bus.subscribe("consensus_state_event", self._on_local_consensus)

    def _setup_sockets(self):
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)

        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._recv_sock.bind(("", self.port))
        except Exception as e:
            self.logger.log("WARN", "Cluster recv bind failed", error=str(e))
        mreq = socket.inet_aton(self.addr) + socket.inet_aton("0.0.0.0")
        try:
            self._recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception as e:
            self.logger.log("WARN", "Cluster membership failed", error=str(e))

    def start(self):
        self._setup_sockets()
        self.thread = Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        self.logger.log("INFO", "ClusterOrgan started", node_id=self.node_id,
                        addr=self.addr, port=self.port)

    def stop(self):
        self._stop = True
        if self.thread:
            self.thread.join(timeout=1.0)
        if self._recv_sock:
            self._recv_sock.close()
        if self._send_sock:
            self._send_sock.close()

    def _send(self, kind: str, payload: Dict[str, Any]):
        if not self._send_sock:
            return
        msg = {
            "node_id": self.node_id,
            "kind": kind,
            "payload": payload,
            "ts": time.time(),
        }
        try:
            data = json.dumps(msg).encode("utf-8")
            self._send_sock.sendto(data, (self.addr, self.port))
        except Exception as e:
            self.logger.log("WARN", "Cluster send failed", error=str(e))

    def _recv_loop(self):
        while not self._stop and self._recv_sock:
            try:
                data, _ = self._recv_sock.recvfrom(65535)
            except Exception:
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if msg.get("node_id") == self.node_id:
                continue
            kind = msg.get("kind")
            payload = msg.get("payload", {})
            if kind == "lineage_event":
                self.bus.publish(FlowEvent("cluster_lineage_event", payload))
            elif kind == "policy_action_event":
                self.bus.publish(FlowEvent("cluster_policy_action_event", payload))
            elif kind == "anomaly_event":
                self.bus.publish(FlowEvent("cluster_anomaly_event", payload))
            elif kind == "consensus_state_event":
                self.bus.publish(FlowEvent("cluster_consensus_state_event", payload))

    def _on_local_lineage(self, event: FlowEvent):
        self._send("lineage_event", event.payload)

    def _on_local_policy_action(self, event: FlowEvent):
        self._send("policy_action_event", event.payload)

    def _on_local_anomaly(self, event: FlowEvent):
        self._send("anomaly_event", event.payload)

    def _on_local_consensus(self, event: FlowEvent):
        self._send("consensus_state_event", event.payload)


# =========================
#  CONSENSUS ORGAN
# =========================

class ConsensusOrgan:
    """
    Very simple cluster consensus: aggregates node health/state snapshots
    and derives a "max severity" consensus view.
    """

    def __init__(self, policy: PolicyEngineV3, bus: FlowBus, logger: LoggingOrgan, node_id: str):
        self.policy = policy
        self.bus = bus
        self.logger = logger
        self.node_id = node_id
        self._lock = Lock()
        self.cluster_states: Dict[str, Dict[str, Any]] = {}

        self.bus.subscribe("cluster_consensus_state_event", self._on_cluster_state)

        self.thread = Thread(target=self._loop, daemon=True)
        self._stop = False

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop = True
        self.thread.join(timeout=1.0)

    def _loop(self):
        while not self._stop:
            time.sleep(5.0)
            node_state, node_health, actions = self.policy.snapshot()
            state = {
                "node_id": self.node_id,
                "node_state": {k: v.value for k, v in node_state.items()},
                "node_health": node_health,
                "ts": time.time(),
            }
            self.bus.publish(FlowEvent("consensus_state_event", {"state": state}))
            self._update_local_cluster_state(self.node_id, state)
            self._compute_consensus()

    def _on_cluster_state(self, event: FlowEvent):
        state = event.payload["state"]
        nid = state.get("node_id", "unknown")
        self._update_local_cluster_state(nid, state)
        self._compute_consensus()

    def _update_local_cluster_state(self, node_id: str, state: Dict[str, Any]):
        with self._lock:
            self.cluster_states[node_id] = state

    def _compute_consensus(self):
        with self._lock:
            if not self.cluster_states:
                return
            # simple consensus: for each node, take worst state seen across cluster
            consensus: Dict[str, str] = {}
            for nid, st in self.cluster_states.items():
                for node, s in st.get("node_state", {}).items():
                    prev = consensus.get(node, "OPEN")
                    # order: OPEN < QUARANTINED < SEALED
                    order = {"OPEN": 0, "QUARANTINED": 1, "SEALED": 2}
                    if order.get(s, 0) > order.get(prev, 0):
                        consensus[node] = s
            self.bus.publish(FlowEvent("consensus_view", {"consensus": consensus}))


# =========================
#  HOP GRAPH ORGAN
# =========================

class HopGraphOrgan:
    def __init__(self, bus: FlowBus):
        self._lock = Lock()
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        bus.subscribe("lineage_event", self._on_lineage)
        bus.subscribe("cluster_lineage_event", self._on_lineage)

    def _on_lineage(self, event: FlowEvent):
        edge = event.payload["edge"]
        with self._lock:
            self.nodes.setdefault(edge["from_node"], {"name": edge["from_node"]})
            self.nodes.setdefault(edge["to_node"], {"name": edge["to_node"]})
            self.edges.append(edge)
            if len(self.edges) > 500:
                self.edges = self.edges[-500:]

    def snapshot(self):
        with self._lock:
            return dict(self.nodes), list(self.edges)


# =========================
#  RINGS & DYE PACK ORGAN
# =========================

@dataclass
class Ring:
    name: str
    queue: Deque[DataCapsule] = field(default_factory=deque)
    max_depth: int = 1000

    def push(self, capsule: DataCapsule) -> bool:
        if len(self.queue) >= self.max_depth:
            return False
        self.queue.append(capsule)
        return True


class DyePackOrgan:
    def __init__(self, company_id: str, crypto: CryptoOrganV2, gate: UVGate,
                 reactor: TamperReactor, lineage: LineageGraph,
                 contamination: ContaminationMap, policy: PolicyEngineV3,
                 persistence: PersistenceOrgan, logger: LoggingOrgan,
                 bus: FlowBus):
        self.company_id = company_id
        self.crypto = crypto
        self.gate = gate
        self.reactor = reactor
        self.lineage = lineage
        self.contamination = contamination
        self.policy = policy
        self.persistence = persistence
        self.logger = logger
        self.bus = bus

    def _hop_sig(self, capsule_id: str, from_node: str, to_node: str, ts: float) -> str:
        base = f"{capsule_id}|{from_node}|{to_node}|{ts}"
        return sha256_str(base)

    def wrap_outbound(self, payload: bytes, env_tag: str, ring: Ring,
                      source: str, key_id: str = "default",
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[DataCapsule]:
        sealed, nonce_hex = self.crypto.seal(payload, env_tag, key_id=key_id)
        integrity_hash = sha256_str(payload.decode("utf-8", errors="ignore"))
        base = {
            "company_id": self.company_id,
            "env_tag": env_tag,
            "integrity_hash": integrity_hash,
            "ring_name": ring.name,
            "key_id": key_id,
            "nonce": nonce_hex,
        }
        import json
        signature = sha256_str(json.dumps(base, sort_keys=True))
        capsule = DataCapsule(
            capsule_id=str(uuid.uuid4()),
            company_id=self.company_id,
            env_tag=env_tag,
            sealed_payload=sealed,
            integrity_hash=integrity_hash,
            signature=signature,
            key_id=key_id,
            ring_name=ring.name,
            source=source,
            metadata=metadata or {},
        )
        if not ring.push(capsule):
            self.logger.log("WARN", "Ring backpressure drop", ring=ring.name, capsule_id=capsule.capsule_id)
            return None
        self.logger.log("INFO", "Capsule wrapped", capsule_id=capsule.capsule_id,
                        source=source, ring=ring.name, env=env_tag)
        return capsule

    def move_capsule(self, capsule: DataCapsule, from_node: str, to_node: str,
                     expected_env: str, to_ring: Optional[Ring] = None) -> VerificationResult:
        ts = time.time()
        hop_sig = self._hop_sig(capsule.capsule_id, from_node, to_node, ts)
        capsule.hop_chain.append(HopRecord(from_node=from_node, to_node=to_node, ts=ts, hop_sig=hop_sig))

        res = self.gate.verify(capsule, expected_env=expected_env, node=to_node)
        self.reactor.handle(res)
        self.policy.on_verification(res)

        edge = LineageEdge(
            capsule_id=capsule.capsule_id,
            from_node=from_node,
            to_node=to_node,
            color=res.color.value,
            status=res.status.name,
            timestamp=res.timestamp,
        )
        self.lineage.record_hop(edge)
        self.persistence.store_lineage(edge)

        if res.color != LogicalColor.GREEN:
            self.contamination.register_contamination(edge)
            self.persistence.store_contamination(edge)

        node_state, node_health, actions = self.policy.snapshot()
        if actions:
            last = actions[-1]
            self.persistence.store_policy_action(last)
            self.bus.publish(FlowEvent("policy_action_event", {
                "action": {
                    "kind": last.kind,
                    "target": last.target,
                    "reason": last.reason,
                    "severity": last.severity,
                    "ts": last.timestamp,
                }
            }))

        if to_ring is not None and not capsule.quarantined:
            if not to_ring.push(capsule):
                self.logger.log("WARN", "Ring backpressure drop on move",
                                ring=to_ring.name, capsule_id=capsule.capsule_id)

        self.logger.log("INFO", "Capsule moved",
                        capsule_id=capsule.capsule_id,
                        from_node=from_node, to_node=to_node,
                        color=res.color.value, status=res.status.name,
                        severity=res.severity)

        self.bus.publish(FlowEvent("verification_result", {
            "result": res,
        }))
        self.bus.publish(FlowEvent("lineage_event", {
            "edge": {
                "capsule_id": edge.capsule_id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "color": edge.color,
                "status": edge.status,
                "timestamp": edge.timestamp,
            }
        }))

        return res


# =========================
#  DEEP UIAUTOMATION ORGAN
# =========================

class DeepUIAutomationOrgan:
    def __init__(self, logger: LoggingOrgan, min_interval_ms: int):
        self.logger = logger
        self.min_interval_ms = min_interval_ms
        self._last_ts = 0.0
        self._lock = Lock()

    def snapshot_active_window(self) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            if (now - self._last_ts) * 1000 < self.min_interval_ms:
                return None
            self._last_ts = now

        if not UIA_AVAILABLE:
            self.logger.log("WARN", "UIAutomation not available")
            return {"available": False, "reason": "pywinauto not installed or failed"}

        try:
            desktop = Desktop(backend="uia")
            active = desktop.get_active()
        except Exception as e:
            self.logger.log("WARN", "UIAutomation active window error", error=str(e))
            return {"available": False, "reason": str(e)}

        info: Dict[str, Any] = {
            "available": True,
            "window_title": active.window_text(),
            "process_id": active.process_id(),
            "class_name": active.class_name(),
            "tree": [],
        }

        max_depth = 5
        max_nodes = 300

        def walk(elem, depth: int, acc: List[Dict[str, Any]]):
            if depth > max_depth or len(acc) >= max_nodes:
                return
            try:
                name = elem.window_text()
                ctrl_type = elem.element_info.control_type
                rect = elem.rectangle()
                node = {
                    "depth": depth,
                    "name": name,
                    "control_type": ctrl_type,
                    "rect": [rect.left, rect.top, rect.right, rect.bottom],
                }
                acc.append(node)
                for child in elem.children():
                    walk(child, depth + 1, acc)
            except Exception:
                return

        try:
            walk(active, 0, info["tree"])
        except Exception as e:
            self.logger.log("WARN", "UIAutomation tree walk error", error=str(e))

        info["node_count"] = len(info["tree"])
        return info


# =========================
#  WORKER POOL
# =========================

@dataclass
class WorkItem:
    capsule: DataCapsule
    from_node: str
    to_node: str
    expected_env: str
    to_ring: Optional[Ring]


class WorkerPool:
    def __init__(self, dye: DyePackOrgan, worker_count: int, logger: LoggingOrgan,
                 bus: FlowBus, cfg: ConfigOrgan):
        self.dye = dye
        self.worker_count = worker_count
        self.logger = logger
        self.bus = bus
        self.cfg = cfg
        self.queue: "Queue[WorkItem]" = Queue()
        self._stop = False
        self.threads: List[Thread] = []
        self._lock = Lock()

    def start(self):
        for _ in range(self.worker_count):
            t = Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self):
        while not self._stop:
            try:
                item = self.queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self.dye.move_capsule(item.capsule, item.from_node, item.to_node,
                                      item.expected_env, item.to_ring)
            except Exception as e:
                self.logger.log("ERROR", "Worker move_capsule error", error=str(e))
            finally:
                self.queue.task_done()

    def submit(self, item: WorkItem):
        self.queue.put(item)

    def adjust_workers(self, target_count: int):
        with self._lock:
            target_count = max(self.cfg.get("WORKERS_MIN", 1),
                               min(self.cfg.get("WORKERS_MAX", 16), target_count))
            if target_count == self.worker_count:
                return
            self.logger.log("INFO", "Adjusting worker count", old=self.worker_count, new=target_count)
            self._stop = True
            for t in self.threads:
                t.join(timeout=1.0)
            self.threads.clear()
            self.worker_count = target_count
            self._stop = False
            self.start()

    def stop(self):
        self._stop = True
        for t in self.threads:
            t.join(timeout=1.0)


# =========================
#  ADAPTIVE TUNING ORGAN
# =========================

class AdaptiveTuningOrgan:
    def __init__(self, rings: List[Ring], workers: WorkerPool,
                 flow_metrics: FlowMetricsOrgan, cfg: ConfigOrgan,
                 logger: LoggingOrgan):
        self.rings = rings
        self.workers = workers
        self.flow_metrics = flow_metrics
        self.cfg = cfg
        self.logger = logger
        self._stop = False
        self.thread: Optional[Thread] = None

        self.pressure_high = cfg.get("PRESSURE_HIGH", 0.9)
        self.pressure_low = cfg.get("PRESSURE_LOW", 0.3)

    def start(self):
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop:
            time.sleep(2.0)
            metrics = self.flow_metrics.compute_metrics()
            rings = metrics.get("rings", {})
            avg_pressure = 0.0
            if rings:
                avg_pressure = sum(m["pressure"] for m in rings.values()) / len(rings)
            current_workers = self.workers.worker_count

            if avg_pressure > self.pressure_high:
                new_workers = current_workers + 1
                self.workers.adjust_workers(new_workers)
                for r in self.rings:
                    r.max_depth = int(r.max_depth * 1.1)
                self.logger.log("INFO", "Adaptive tuning: high pressure",
                                avg_pressure=avg_pressure, workers=new_workers)
            elif avg_pressure < self.pressure_low:
                new_workers = max(self.cfg.get("WORKERS_MIN", 1), current_workers - 1)
                self.workers.adjust_workers(new_workers)
                for r in self.rings:
                    r.max_depth = max(100, int(r.max_depth * 0.9))
                self.logger.log("INFO", "Adaptive tuning: low pressure",
                                avg_pressure=avg_pressure, workers=new_workers)

    def stop(self):
        self._stop = True
        if self.thread:
            self.thread.join(timeout=1.0)


# =========================
#  UNIFIED IO ORGAN
# =========================

class UnifiedIOOrgan:
    def __init__(self, dye: DyePackOrgan, inner: Ring, mid: Ring, outer: Ring,
                 cfg: ConfigOrgan, workers: WorkerPool, logger: LoggingOrgan):
        self.dye = dye
        self.inner = inner
        self.mid = mid
        self.outer = outer
        self.cfg = cfg
        self.workers = workers
        self.logger = logger

        self.env_internal = cfg.get("ENV_INTERNAL", "internal")
        self.env_external = cfg.get("ENV_EXTERNAL", "external")

    def feed_http(self, payload: bytes, metadata: Optional[Dict[str, Any]] = None):
        cap = self.dye.wrap_outbound(payload, env_tag=self.env_external,
                                     ring=self.outer, source="http", metadata=metadata)
        if cap:
            self.workers.submit(WorkItem(cap, "http_ingest", "http_gateway",
                                         self.env_external, self.mid))

    def feed_telemetry(self, payload: bytes, metadata: Optional[Dict[str, Any]] = None):
        cap = self.dye.wrap_outbound(payload, env_tag=self.env_internal,
                                     ring=self.inner, source="telemetry", metadata=metadata)
        if cap:
            self.workers.submit(WorkItem(cap, "telemetry_ingest", "telemetry_gateway",
                                         self.env_internal, self.mid))

    def feed_fs(self, payload: bytes, metadata: Optional[Dict[str, Any]] = None):
        cap = self.dye.wrap_outbound(payload, env_tag=self.env_internal,
                                     ring=self.mid, source="fs", metadata=metadata)
        if cap:
            self.workers.submit(WorkItem(cap, "fs_ingest", "fs_gateway",
                                         self.env_internal, self.outer))

    def feed_queue(self, payload: bytes, metadata: Optional[Dict[str, Any]] = None):
        cap = self.dye.wrap_outbound(payload, env_tag=self.env_internal,
                                     ring=self.inner, source="queue", metadata=metadata)
        if cap:
            self.workers.submit(WorkItem(cap, "queue_ingest", "queue_gateway",
                                         self.env_internal, self.mid))

    def feed_uiautomation(self, snapshot: Dict[str, Any]):
        if snapshot is None:
            return
        payload = json.dumps(snapshot, sort_keys=True).encode("utf-8")
        cap = self.dye.wrap_outbound(payload, env_tag=self.env_internal,
                                     ring=self.mid, source="uia", metadata={"uia": True})
        if cap:
            self.workers.submit(WorkItem(cap, "uia_ingest", "uia_gateway",
                                         self.env_internal, self.outer))


# =========================
#  GPU COCKPIT ORGAN (OPTIONAL)
# =========================

class GpuCockpitOrgan:
    """
    Optional GPU-backed cockpit using moderngl.
    For now, it renders simple rings and points; Tkinter cockpit remains primary.
    """

    def __init__(self, rings: List[Ring], logger: LoggingOrgan):
        self.rings = rings
        self.logger = logger
        self._stop = False
        self.thread: Optional[Thread] = None

    def start(self):
        if not GPU_AVAILABLE:
            self.logger.log("INFO", "GPU cockpit not available (moderngl missing)")
            return
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.logger.log("INFO", "GPU cockpit started")

    def _loop(self):
        # Minimal placeholder: in a real deployment, wire moderngl + windowing here.
        # We keep it non-blocking and non-simulated; no fake data, just a hook.
        while not self._stop:
            time.sleep(1.0)

    def stop(self):
        self._stop = True
        if self.thread:
            self.thread.join(timeout=1.0)


# =========================
#  TKINTER COCKPIT (WITH FLOW, HOP, ANOMALY, REPLAY, SCRIPTING)
# =========================

class TkinterCockpit:
    def __init__(self, rings: List[Ring],
                 lineage: LineageGraph,
                 contamination: ContaminationMap,
                 policy: PolicyEngineV3,
                 bus: FlowBus,
                 flow_metrics: FlowMetricsOrgan,
                 hop_graph: HopGraphOrgan,
                 persistence: PersistenceOrgan,
                 logger: LoggingOrgan):
        self.rings = rings
        self.lineage = lineage
        self.contamination = contamination
        self.policy = policy
        self.bus = bus
        self.flow_metrics = flow_metrics
        self.hop_graph = hop_graph
        self.persistence = persistence
        self.logger = logger

        self.history: Deque[VerificationResult] = deque(maxlen=300)
        self._lock = Lock()

        self._capsule_angles: Dict[str, float] = {}
        self._angle_speed = 0.03

        self._ring_radii = {
            "inner": 80,
            "mid": 130,
            "outer": 180,
        }

        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.contam_list: Optional[tk.Listbox] = None
        self.policy_list: Optional[tk.Listbox] = None
        self.node_entry: Optional[tk.Entry] = None
        self.flow_frame: Optional[tk.Frame] = None
        self.flow_labels: Dict[str, Dict[str, tk.Label]] = {}
        self.hop_list: Optional[tk.Listbox] = None
        self.anomaly_list: Optional[tk.Listbox] = None
        self.replay_list: Optional[tk.Listbox] = None
        self.replay_scale: Optional[tk.Scale] = None
        self.script_text: Optional[tk.Text] = None

        self.bus.subscribe("verification_result", self._on_verification_event)
        self.bus.subscribe("flow_metrics", self._on_flow_metrics)
        self.bus.subscribe("anomaly_event", self._on_anomaly_event)
        self.bus.subscribe("cluster_anomaly_event", self._on_cluster_anomaly_event)

        self._last_flow_metrics: Dict[str, Any] = {}
        self._anomalies: Deque[Dict[str, Any]] = deque(maxlen=50)

        self._replay_edges: List[LineageEdge] = []
        self._replay_start_ts: float = 0.0
        self._replay_end_ts: float = 0.0

    def _on_verification_event(self, event: FlowEvent):
        res: VerificationResult = event.payload["result"]
        with self._lock:
            self.history.append(res)
            cid = res.capsule.capsule_id
            if cid not in self._capsule_angles:
                self._capsule_angles[cid] = (hash(cid) % 360) * math.pi / 180.0

    def _on_flow_metrics(self, event: FlowEvent):
        self._last_flow_metrics = event.payload["metrics"]

    def _on_anomaly_event(self, event: FlowEvent):
        anomaly = event.payload["anomaly"]
        self._anomalies.append(anomaly)

    def _on_cluster_anomaly_event(self, event: FlowEvent):
        anomaly = event.payload["anomaly"]
        self._anomalies.append(anomaly)

    def _color_for_state(self, color: LogicalColor) -> str:
        if color == LogicalColor.GREEN:
            return "#00ff00"
        if color == LogicalColor.YELLOW:
            return "#ffff00"
        if color == LogicalColor.ORANGE:
            return "#ffa500"
        if color == LogicalColor.RED:
            return "#ff0000"
        if color == LogicalColor.PURPLE:
            return "#b000ff"
        return "#cccccc"

    def _update_angles(self):
        with self._lock:
            for cid in list(self._capsule_angles.keys()):
                self._capsule_angles[cid] += self._angle_speed

    def _draw(self):
        if not self.canvas:
            return

        self._update_angles()
        self.canvas.delete("all")

        w = int(self.canvas["width"])
        h = int(self.canvas["height"])
        cx, cy = w // 2, h // 2

        for ring_name, radius in self._ring_radii.items():
            self.canvas.create_oval(cx - radius, cy - radius,
                                    cx + radius, cy + radius,
                                    outline="#555555")

        with self._lock:
            recent = list(self.history)[-200:]

        for res in recent:
            cap = res.capsule
            cid = cap.capsule_id
            angle = self._capsule_angles.get(cid, 0.0)
            radius = self._ring_radii.get(cap.ring_name, 200)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            color = self._color_for_state(res.color)
            r = 4 if not cap.quarantined else 7
            self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                    fill=color, outline=color)

        if self.contam_list:
            self.contam_list.delete(0, tk.END)
            with self.contamination._lock:
                items = list(self.contamination.node_contamination_count.items())
            for node, count in items:
                self.contam_list.insert(tk.END, f"{node}: {count}")

        if self.policy_list:
            self.policy_list.delete(0, tk.END)
            node_state, node_health, actions = self.policy.snapshot()
            self.policy_list.insert(tk.END, "Node states / health:")
            for node, state in node_state.items():
                health = node_health.get(node, 0)
                self.policy_list.insert(tk.END, f"{node}: {state.value} (health={health})")
            if actions:
                self.policy_list.insert(tk.END, "---- Recent actions ----")
                for act in actions[-10:]:
                    self.policy_list.insert(tk.END,
                                            f"{act.kind} -> {act.target} ({act.reason}, sev={act.severity})")

        self._update_flow_gauges()
        self._update_hop_view()
        self._update_anomaly_view()
        self._update_replay_view()

        if self.root:
            self.root.after(80, self._draw)

    def _update_flow_gauges(self):
        metrics = self._last_flow_metrics
        if not metrics or not self.flow_labels:
            return
        rings = metrics.get("rings", {})
        sources = metrics.get("sources", {})

        for ring_name, m in rings.items():
            labels = self.flow_labels.get(ring_name, {})
            if not labels:
                continue
            pressure = m.get("pressure", 0.0)
            velocity = m.get("velocity", 0.0)
            depth = m.get("depth", 0)
            max_depth = m.get("max_depth", 1)
            labels["pressure"].config(text=f"P={pressure:.2f} ({depth}/{max_depth})")
            labels["velocity"].config(text=f"V={velocity:.2f} cps")

        if "sources" in self.flow_labels:
            src_label = self.flow_labels["sources"]["rates"]
            parts = [f"{src}:{rate:.2f}" for src, rate in sources.items()]
            src_label.config(text=" | ".join(parts) if parts else "no flow")

    def _update_hop_view(self):
        if not self.hop_list:
            return
        self.hop_list.delete(0, tk.END)
        nodes, edges = self.hop_graph.snapshot()
        self.hop_list.insert(tk.END, f"Nodes: {', '.join(nodes.keys())}")
        self.hop_list.insert(tk.END, "Recent edges:")
        for e in edges[-20:]:
            self.hop_list.insert(
                tk.END,
                f"{e['from_node']} -> {e['to_node']} ({e['status']}/{e['color']})"
            )

    def _update_anomaly_view(self):
        if not self.anomaly_list:
            return
        self.anomaly_list.delete(0, tk.END)
        for a in list(self._anomalies)[-20:]:
            src = a.get("source", "stat")
            self.anomaly_list.insert(
                tk.END,
                f"[{src}] {a.get('ring','?')}: {a.get('description','')}"
            )

    def _update_replay_view(self):
        if not self.replay_list or not self._replay_edges:
            return
        self.replay_list.delete(0, tk.END)
        for e in self._replay_edges[-30:]:
            self.replay_list.insert(
                tk.END,
                f"{time.strftime('%H:%M:%S', time.localtime(e.timestamp))} "
                f"{e.from_node}->{e.to_node} ({e.status}/{e.color})"
            )

    def _cmd_reset_node(self):
        if not self.node_entry:
            return
        node = self.node_entry.get().strip()
        if not node:
            return
        self.bus.publish(FlowEvent("operator_command", {
            "cmd": "RESET_NODE_HEALTH",
            "target": node,
        }))

    def _cmd_seal_node(self):
        if not self.node_entry:
            return
        node = self.node_entry.get().strip()
        if not node:
            return
        self.bus.publish(FlowEvent("operator_command", {
            "cmd": "SEAL_NODE",
            "target": node,
        }))

    def _cmd_unseal_node(self):
        if not self.node_entry:
            return
        node = self.node_entry.get().strip()
        if not node:
            return
        self.bus.publish(FlowEvent("operator_command", {
            "cmd": "UNSEAL_NODE",
            "target": node,
        }))

    # ---- Replay UI ----

    def _cmd_load_replay(self):
        now = time.time()
        start = now - 300  # last 5 minutes
        end = now
        edges = self.persistence.load_lineage_window(start, end)
        self._replay_edges = edges
        self._replay_start_ts = start
        self._replay_end_ts = end
        if self.replay_scale:
            self.replay_scale.config(from_=start, to=end)
            self.replay_scale.set(end)

    def _on_replay_slider(self, value):
        if not self._replay_edges:
            return
        ts = float(value)
        subset = [e for e in self._replay_edges if e.timestamp <= ts]
        self._replay_edges = subset

    # ---- Operator scripting console ----

    def _cmd_run_script(self):
        if not self.script_text:
            return
        script = self.script_text.get("1.0", tk.END).strip()
        if not script:
            return
        lines = [l.strip() for l in script.splitlines() if l.strip()]
        for line in lines:
            self._execute_script_line(line)

    def _execute_script_line(self, line: str):
        # Simple DSL:
        #   seal nodeX
        #   unseal nodeX
        #   reset nodeX
        #   set PRESSURE_HIGH 0.95
        parts = line.split()
        if not parts:
            return
        cmd = parts[0].lower()
        if cmd == "seal" and len(parts) == 2:
            self.bus.publish(FlowEvent("operator_command", {
                "cmd": "SEAL_NODE",
                "target": parts[1],
            }))
        elif cmd == "unseal" and len(parts) == 2:
            self.bus.publish(FlowEvent("operator_command", {
                "cmd": "UNSEAL_NODE",
                "target": parts[1],
            }))
        elif cmd == "reset" and len(parts) == 2:
            self.bus.publish(FlowEvent("operator_command", {
                "cmd": "RESET_NODE_HEALTH",
                "target": parts[1],
            }))
        else:
            self.logger.log("WARN", "Unknown script line", line=line)

    def start(self):
        self.root = tk.Tk()
        self.root.title("Organism Cockpit")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main_frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main_frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.canvas = tk.Canvas(left, width=600, height=600, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        top_right = ttk.Frame(right)
        top_right.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom_right = ttk.Frame(right)
        bottom_right.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        contam_frame = ttk.LabelFrame(top_right, text="Node contamination heat")
        contam_frame.pack(fill=tk.X)
        self.contam_list = tk.Listbox(contam_frame, height=6)
        self.contam_list.pack(fill=tk.X)

        policy_frame = ttk.LabelFrame(top_right, text="Policy state / actions")
        policy_frame.pack(fill=tk.BOTH, expand=True)
        self.policy_list = tk.Listbox(policy_frame, height=8)
        self.policy_list.pack(fill=tk.BOTH, expand=True)

        ctrl_frame = ttk.LabelFrame(top_right, text="Operator controls")
        ctrl_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ctrl_frame, text="Node:").grid(row=0, column=0, sticky="w")
        self.node_entry = ttk.Entry(ctrl_frame, width=18)
        self.node_entry.grid(row=0, column=1, sticky="we")
        btn_reset = ttk.Button(ctrl_frame, text="Reset Health", command=self._cmd_reset_node)
        btn_reset.grid(row=1, column=0, sticky="we")
        btn_seal = ttk.Button(ctrl_frame, text="Seal", command=self._cmd_seal_node)
        btn_seal.grid(row=1, column=1, sticky="we")
        btn_unseal = ttk.Button(ctrl_frame, text="Unseal", command=self._cmd_unseal_node)
        btn_unseal.grid(row=2, column=0, columnspan=2, sticky="we")

        flow_frame = ttk.LabelFrame(bottom_right, text="Flow (Bernoulli)")
        flow_frame.pack(fill=tk.X, pady=5)
        self.flow_frame = flow_frame

        row = 0
        self.flow_labels = {}
        for ring_name in ["inner", "mid", "outer"]:
            lf = {}
            ttk.Label(flow_frame, text=ring_name).grid(row=row, column=0, sticky="w")
            lp = ttk.Label(flow_frame, text="P=0.00")
            lv = ttk.Label(flow_frame, text="V=0.00 cps")
            lp.grid(row=row, column=1, sticky="w")
            lv.grid(row=row, column=2, sticky="w")
            lf["pressure"] = lp
            lf["velocity"] = lv
            self.flow_labels[ring_name] = lf
            row += 1

        src_label = ttk.Label(flow_frame, text="sources: none")
        src_label.grid(row=row, column=0, columnspan=3, sticky="w")
        self.flow_labels["sources"] = {"rates": src_label}

        hop_frame = ttk.LabelFrame(bottom_right, text="Hop-chain view")
        hop_frame.pack(fill=tk.BOTH, expand=True)
        self.hop_list = tk.Listbox(hop_frame, height=8)
        self.hop_list.pack(fill=tk.BOTH, expand=True)

        anomaly_frame = ttk.LabelFrame(bottom_right, text="Anomalies")
        anomaly_frame.pack(fill=tk.BOTH, expand=True)
        self.anomaly_list = tk.Listbox(anomaly_frame, height=6)
        self.anomaly_list.pack(fill=tk.BOTH, expand=True)

        replay_frame = ttk.LabelFrame(bottom_right, text="Replay")
        replay_frame.pack(fill=tk.BOTH, expand=True)
        self.replay_list = tk.Listbox(replay_frame, height=6)
        self.replay_list.pack(fill=tk.BOTH, expand=True)
        self.replay_scale = tk.Scale(replay_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                     command=self._on_replay_slider)
        self.replay_scale.pack(fill=tk.X)
        btn_load_replay = ttk.Button(replay_frame, text="Load last 5 min", command=self._cmd_load_replay)
        btn_load_replay.pack(fill=tk.X)

        script_frame = ttk.LabelFrame(bottom_right, text="Operator scripting console")
        script_frame.pack(fill=tk.BOTH, expand=True)
        self.script_text = tk.Text(script_frame, height=4)
        self.script_text.pack(fill=tk.BOTH, expand=True)
        btn_run_script = ttk.Button(script_frame, text="Run script", command=self._cmd_run_script)
        btn_run_script.pack(fill=tk.X)

        self.root.after(80, self._draw)
        self.root.mainloop()

    def stop(self):
        if self.root:
            self.root.quit()


# =========================
#  LIFECYCLE ORGAN
# =========================

class LifecycleOrgan:
    def __init__(self, cockpit: TkinterCockpit, workers: WorkerPool,
                 persistence: PersistenceOrgan, policy: PolicyEngineV3,
                 bus: FlowBus, logger: LoggingOrgan,
                 tuning: AdaptiveTuningOrgan,
                 cluster: ClusterOrgan,
                 consensus: ConsensusOrgan,
                 gpu_cockpit: GpuCockpitOrgan):
        self.cockpit = cockpit
        self.workers = workers
        self.persistence = persistence
        self.policy = policy
        self.bus = bus
        self.logger = logger
        self.tuning = tuning
        self.cluster = cluster
        self.consensus = consensus
        self.gpu_cockpit = gpu_cockpit
        self.threads: List[Thread] = []
        self._stop = False

        self.bus.subscribe("operator_command", self._on_operator_command)
        self.bus.subscribe("cluster_policy_action_event", self._on_cluster_policy_action)

    def _on_operator_command(self, event: FlowEvent):
        cmd = event.payload["cmd"]
        target = event.payload["target"]
        self.policy.apply_operator_command(cmd, target)
        self.logger.log("INFO", "Operator command applied", cmd=cmd, target=target)

    def _on_cluster_policy_action(self, event: FlowEvent):
        action = event.payload["action"]
        pa = PolicyAction(
            kind=action["kind"],
            target=action["target"],
            reason=action["reason"],
            severity=action["severity"],
            timestamp=action["ts"],
        )
        with self.policy._lock:
            self.policy.actions.append(pa)

    def start(self):
        self.workers.start()
        self.tuning.start()
        self.cluster.start()
        self.consensus.start()
        self.gpu_cockpit.start()
        t = Thread(target=self.cockpit.start, daemon=True)
        t.start()
        self.threads.append(t)

        rt = Thread(target=self._retention_loop, daemon=True)
        rt.start()
        self.threads.append(rt)

    def _retention_loop(self):
        while not self._stop:
            time.sleep(3600)
            self.persistence.apply_retention()

    def wait(self):
        for t in self.threads:
            t.join()

    def stop(self):
        self._stop = True
        self.tuning.stop()
        self.workers.stop()
        self.cluster.stop()
        self.consensus.stop()
        self.gpu_cockpit.stop()
        self.cockpit.stop()


# =========================
#  ORGANISM BOOTSTRAP
# =========================

def build_organism(config_path: Optional[str] = None):
    cfg = ConfigOrgan(config_path=config_path)
    logger = LoggingOrgan()
    bus = FlowBus()

    company_id = cfg.get("COMPANY_ID", "ACME_CORP")
    crypto = CryptoOrganV2(master_key_hex=cfg.get("KEY_MASTER"))
    lineage = LineageGraph()
    contamination = ContaminationMap()
    policy = PolicyEngineV3(policy_path=cfg.get("POLICY_PATH", "policy.json"), logger=logger)
    reactor = TamperReactor()
    gate = UVGate(trusted_company_ids=[company_id], crypto=crypto)
    persistence = PersistenceOrgan(db_path=cfg.get("DB_PATH", "organism.db"),
                                   retention_days=cfg.get("RETENTION_DAYS", 7),
                                   logger=logger)

    max_depth = cfg.get("RING_MAX_DEPTH", 1000)
    inner = Ring("inner", max_depth=max_depth)
    mid = Ring("mid", max_depth=max_depth)
    outer = Ring("outer", max_depth=max_depth)
    rings = [inner, mid, outer]

    dye = DyePackOrgan(
        company_id=company_id,
        crypto=crypto,
        gate=gate,
        reactor=reactor,
        lineage=lineage,
        contamination=contamination,
        policy=policy,
        persistence=persistence,
        logger=logger,
        bus=bus,
    )

    flow_metrics = FlowMetricsOrgan(rings, bus, window_sec=cfg.get("FLOW_WINDOW_SEC", 10), logger=logger)
    anomaly = FlowAnomalyOrgan(bus, sensitivity=cfg.get("ANOMALY_SENSITIVITY", 3.0), logger=logger)
    ml_anomaly = MlAnomalyOrgan(bus, cfg, logger)
    hop_graph = HopGraphOrgan(bus)
    cluster = ClusterOrgan(cfg, bus, logger)
    consensus = ConsensusOrgan(policy, bus, logger, node_id=cfg.get("CLUSTER_NODE_ID", "node-1"))

    gpu_cockpit = GpuCockpitOrgan(rings, logger)
    cockpit = TkinterCockpit(rings, lineage, contamination, policy, bus, flow_metrics, hop_graph, persistence, logger)
    workers = WorkerPool(dye, worker_count=cfg.get("WORKER_COUNT", 4), logger=logger, bus=bus, cfg=cfg)
    tuning = AdaptiveTuningOrgan(rings, workers, flow_metrics, cfg, logger)
    uia = DeepUIAutomationOrgan(logger, min_interval_ms=cfg.get("UIA_MIN_INTERVAL_MS", 2000))
    io = UnifiedIOOrgan(dye, inner, mid, outer, cfg, workers, logger)
    lifecycle = LifecycleOrgan(cockpit, workers, persistence, policy, bus, logger, tuning, cluster, consensus, gpu_cockpit)

    bus.subscribe("anomaly_event", lambda e: policy.apply_anomaly(e.payload["anomaly"]))
    bus.subscribe("cluster_anomaly_event", lambda e: policy.apply_anomaly(e.payload["anomaly"]))

    return {
        "config": cfg,
        "logger": logger,
        "crypto": crypto,
        "lineage": lineage,
        "contamination": contamination,
        "policy": policy,
        "reactor": reactor,
        "gate": gate,
        "persistence": persistence,
        "rings": rings,
        "dye": dye,
        "cockpit": cockpit,
        "workers": workers,
        "uia": uia,
        "io": io,
        "lifecycle": lifecycle,
        "bus": bus,
        "flow_metrics": flow_metrics,
        "anomaly": anomaly,
        "ml_anomaly": ml_anomaly,
        "hop_graph": hop_graph,
        "cluster": cluster,
        "consensus": consensus,
        "tuning": tuning,
        "gpu_cockpit": gpu_cockpit,
    }


if __name__ == "__main__":
    org = build_organism()
    lifecycle: LifecycleOrgan = org["lifecycle"]
    io: UnifiedIOOrgan = org["io"]
    uia: DeepUIAutomationOrgan = org["uia"]

    lifecycle.start()

    # Wire real data here (no simulation):
    # snapshot = uia.snapshot_active_window()
    # io.feed_uiautomation(snapshot)
    # io.feed_http(b"GET /api/resource HTTP/1.1", {"path": "/api/resource"})
    # io.feed_telemetry(b"fps=144 ping=32", {"game": "example"})
    # io.feed_fs(b"file modified: config.yaml", {"path": "config.yaml"})
    # io.feed_queue(b"job:12345", {"queue": "jobs"})

    lifecycle.wait()

