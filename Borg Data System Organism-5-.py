"""
unified_borg_organism_vOmega.py

Single-file "synthetic data organism" node with:

Core:
- Universal cross-platform autoloader for dependencies
- Node identity
- Data capsule model
- In-process event bus
- Persona states and transitions
- Threat matrix integration
- Sync topology hooks
- External system handshake protocol

Borg system:
- Classic Borg Scheduler (Queen + executor workers)
- Borg Organism (Queen + sensor/motor workers)
- Borg Augmentation Engine (optimizer workers)
- GPU-accelerated optimizer (if torch+CUDA available)
- APU / NPU / LPU / DPU workers (simulated accelerators)

Evolution system:
- Genetic algorithms for self-mutation (config evolution)
- Reinforcement learning for long-term adaptation (policy tuning)
- Swarm-level consensus (simple majority-based consensus)
- Queen-of-Queens (cluster super-brain abstraction)

Persistence + Knowledge:
- Persistent storage of genetic config and RL Q-table
- Knowledge store:
    - Per-node stats
    - Per-partner stats
    - Anomaly scores
    - Bernoulli-style event probabilities
    - Missing-detail detection
- Rewards wired to real metrics:
    - Threat levels
    - Latency
    - Error rates
    - Anomaly scores

Predictive / Physics / Altered States:
- Bernoulli-style predictive engine for event likelihoods
- Missing-detail inference engine (best-guess filler)
- "Intelligent water" data physics engine (flow, pressure, turbulence metaphors)
- Altered states of consciousness:
    - Normal
    - Hypervigilant
    - Dreaming (speculative prediction mode)
    - Dissociated (failsafe minimal mode)

Real-time Queen:
- Real-time global risk aggregator across nodes

Higher cognition:
- Deep-learning prediction engine (MLP)
- Transformer-based prediction engine (sequence-aware, if torch available)
- Dream-simulation subsystem (speculative future capsules)
- Visual dashboard (FastAPI HTML + JSON views + charts)
- Plugin architecture (dynamic loading from ./plugins)
- Self-replication system (safe local clone + optional new port spawn)

Distributed / Cluster:
- Distributed storage integration (Redis optional, Cassandra stub)
- Simple Raft-style consensus stub (for future extension)
- Autonomous deployment stubs across machines (command generation)

Meta-evolution:
- Genetic plugin evolution (enable/disable plugins based on reward)
- Memory-consolidation "sleep" cycle (periodic deep save + cleanup)

Runs on Windows, Linux, macOS.
"""

# ============================================================
# 0. UNIVERSAL AUTOLOADER FOR ALL NECESSARY LIBRARIES
# ============================================================

import importlib
import subprocess
import sys
import platform

REQUIRED_LIBRARIES = [
    "fastapi",
    "uvicorn",
    "requests",
    "pydantic",
    "redis",  # optional, used if available
]

def install_package(package: str):
    try:
        print(f"[AUTOLOADER] Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"[AUTOLOADER] Failed to install {package}: {e}")

def autoload_libraries():
    os_name = platform.system().lower()
    print(f"[AUTOLOADER] Detected OS: {os_name}")
    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"[AUTOLOADER] Loaded: {lib}")
        except ImportError:
            print(f"[AUTOLOADER] Missing: {lib}")
            install_package(lib)
            try:
                importlib.import_module(lib)
                print(f"[AUTOLOADER] Successfully loaded after install: {lib}")
            except ImportError:
                print(f"[AUTOLOADER] ERROR: Could not load {lib} even after installation.")

autoload_libraries()

# ============================================================
# 1. Imports (safe after autoloader)
# ============================================================

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Callable, Optional
from enum import Enum
import time
import uuid
import threading
import requests
import uvicorn
import hashlib
import json
import argparse
import queue
import random
import os
import math
import shutil

# Optional GPU / DL support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_TORCH = False
    HAS_CUDA = False

# Optional Redis
try:
    import redis
    HAS_REDIS = True
except Exception:
    HAS_REDIS = False

# ============================================================
# 2. Node identity & config
# ============================================================

class NodeRole(str, Enum):
    SENSOR = "sensor"
    ANALYTICS = "analytics"
    ACTUATOR = "actuator"
    COORDINATOR = "coordinator"
    GATEWAY = "gateway"


class TrustTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Persona(str, Enum):
    OBSERVER = "observer"
    OPERATOR = "operator"
    AUTONOMOUS = "autonomous"
    CAUTIOUS = "cautious"
    QUARANTINED = "quarantined"


class ConsciousnessState(str, Enum):
    NORMAL = "normal"
    HYPERVIGILANT = "hypervigilant"
    DREAMING = "dreaming"
    DISSOCIATED = "dissociated"
    SLEEP = "sleep"


class NodeIdentity(BaseModel):
    node_id: str
    cluster_id: str
    role: NodeRole
    region: str
    trust_tier: TrustTier
    public_key_fingerprint: str
    created_at: float


def generate_node_identity(
    cluster_id: str,
    role: NodeRole,
    region: str,
    trust_tier: TrustTier = TrustTier.MEDIUM,
) -> NodeIdentity:
    raw_id = f"{cluster_id}-{role.value}-{uuid.uuid4()}"
    pk_fingerprint = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
    return NodeIdentity(
        node_id=f"org-{raw_id}",
        cluster_id=cluster_id,
        role=role,
        region=region,
        trust_tier=trust_tier,
        public_key_fingerprint=pk_fingerprint,
        created_at=time.time(),
    )

# ============================================================
# 3. Data capsule model
# ============================================================

class CapsuleType(str, Enum):
    TELEMETRY = "telemetry"
    EVENT = "event"
    COMMAND = "command"
    SYNC = "sync"
    ALERT = "alert"
    CONTROL = "control"


class CapsuleClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SECRET = "secret"


class DataCapsule(BaseModel):
    capsule_id: str
    capsule_type: CapsuleType
    origin_node_id: str
    target_scope: str
    timestamp: float
    priority: int
    ttl: float
    payload: Dict[str, Any]
    classification: CapsuleClassification = CapsuleClassification.INTERNAL
    integrity_hash: Optional[str] = None
    threat_tags: List[str] = []
    meta: Dict[str, Any] = {}


def compute_capsule_hash(capsule: DataCapsule) -> str:
    data = {
        "capsule_id": capsule.capsule_id,
        "capsule_type": capsule.capsule_type.value,
        "origin_node_id": capsule.origin_node_id,
        "target_scope": capsule.target_scope,
        "timestamp": capsule.timestamp,
        "priority": capsule.priority,
        "ttl": capsule.ttl,
        "payload": capsule.payload,
        "classification": capsule.classification.value,
        "meta": capsule.meta,
    }
    raw = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()

# ============================================================
# 4. Persona engine & threat matrix
# ============================================================

class ThreatScore(BaseModel):
    score: float
    tags: List[str] = []


def evaluate_threat(capsule: DataCapsule, identity: NodeIdentity) -> ThreatScore:
    score = 0.1
    tags: List[str] = []

    if capsule.capsule_type in {CapsuleType.COMMAND, CapsuleType.CONTROL}:
        score += 0.3
        tags.append("command_like")

    if capsule.target_scope.startswith("external:"):
        score += 0.3
        tags.append("external")

    if capsule.classification == CapsuleClassification.SECRET:
        score += 0.1
        tags.append("sensitive")

    score = min(1.0, max(0.0, score))
    return ThreatScore(score=score, tags=tags)


def persona_transition(current: Persona, threat: ThreatScore) -> Persona:
    s = threat.score
    if s >= 0.8:
        return Persona.QUARANTINED
    if s >= 0.5:
        return Persona.CAUTIOUS
    if s <= 0.2:
        if current in {Persona.OBSERVER, Persona.CAUTIOUS}:
            return Persona.OPERATOR
        if current == Persona.OPERATOR:
            return Persona.AUTONOMOUS
    return current


def consciousness_transition(current: ConsciousnessState, avg_threat: float, anomaly: float) -> ConsciousnessState:
    if current == ConsciousnessState.SLEEP:
        return ConsciousnessState.SLEEP
    if avg_threat > 0.7 or anomaly > 0.7:
        return ConsciousnessState.HYPERVIGILANT
    if avg_threat < 0.2 and anomaly < 0.2:
        if random.random() < 0.1:
            return ConsciousnessState.DREAMING
        return ConsciousnessState.NORMAL
    if avg_threat > 0.9 and anomaly > 0.9:
        return ConsciousnessState.DISSOCIATED
    return current

# ============================================================
# 5. In-process event bus
# ============================================================

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[DataCapsule], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Callable[[DataCapsule], None]):
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)

    def publish(self, topic: str, capsule: DataCapsule):
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        for h in handlers:
            try:
                h(capsule)
            except Exception as e:
                print(f"[EventBus] Handler error on topic {topic}: {e}")

# ============================================================
# 6. External handshake protocol models
# ============================================================

class ExternalCapabilities(BaseModel):
    supported_capsule_types: List[CapsuleType]
    max_rate_per_sec: int
    max_payload_bytes: int
    allowed_operations: List[str]


class ExternalHandshakeRequest(BaseModel):
    partner_id: str
    partner_name: str
    capabilities: ExternalCapabilities
    auth_token: str


class ExternalHandshakeResponse(BaseModel):
    accepted: bool
    reason: Optional[str] = None
    assigned_partner_id: Optional[str] = None
    allowed_capsule_types: List[CapsuleType] = []
    rate_limit_per_sec: int = 0
    channel: Optional[str] = None

# ============================================================
# 7. Persistence utilities
# ============================================================

STATE_DIR = "node_state"
GENETIC_STATE_FILE = os.path.join(STATE_DIR, "genetic_state.json")
RL_STATE_FILE = os.path.join(STATE_DIR, "rl_qtable.json")
KNOWLEDGE_STATE_FILE = os.path.join(STATE_DIR, "knowledge_store.json")
DL_MODEL_FILE = os.path.join(STATE_DIR, "dl_predictor.pt")
TRANSFORMER_MODEL_FILE = os.path.join(STATE_DIR, "transformer_predictor.pt")

def ensure_state_dir():
    if not os.path.isdir(STATE_DIR):
        os.makedirs(STATE_DIR, exist_ok=True)

def load_json(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[PERSISTENCE] Failed to save {path}: {e}")

# ============================================================
# 8. Knowledge store (per-node stats + anomalies + Bernoulli)
# ============================================================

class NodeStats(BaseModel):
    total_capsules: int = 0
    avg_threat: float = 0.0
    last_seen: float = 0.0
    anomaly_score: float = 0.0
    bernoulli_success: int = 0
    bernoulli_trials: int = 0


class PartnerStats(BaseModel):
    total_capsules: int = 0
    avg_threat: float = 0.0
    last_seen: float = 0.0
    anomaly_score: float = 0.0
    bernoulli_success: int = 0
    bernoulli_trials: int = 0


class KnowledgeStore:
    def __init__(self):
        self.node_stats: Dict[str, NodeStats] = {}
        self.partner_stats: Dict[str, PartnerStats] = {}
        self._lock = threading.Lock()

    def _bernoulli_update(self, is_good: bool, success: int, trials: int) -> (int, int):
        trials += 1
        if is_good:
            success += 1
        return success, trials

    def _bernoulli_prob(self, success: int, trials: int) -> float:
        if trials == 0:
            return 0.5
        return (success + 1) / (trials + 2)

    def update_from_capsule(self, capsule: DataCapsule, threat: ThreatScore):
        now = time.time()
        origin = capsule.origin_node_id
        is_good = threat.score < 0.5

        with self._lock:
            ns = self.node_stats.get(origin, NodeStats())
            ns.total_capsules += 1
            ns.avg_threat = ((ns.avg_threat * (ns.total_capsules - 1)) + threat.score) / ns.total_capsules
            ns.last_seen = now
            ns.bernoulli_success, ns.bernoulli_trials = self._bernoulli_update(
                is_good, ns.bernoulli_success, ns.bernoulli_trials
            )

            anomaly = 0.0
            if threat.score > 0.7:
                anomaly += 0.5
            if ns.total_capsules > 50 and ns.avg_threat > 0.5:
                anomaly += 0.3
            ns.anomaly_score = min(1.0, anomaly)
            self.node_stats[origin] = ns

            if capsule.target_scope.startswith("external:"):
                partner_id = capsule.target_scope.split("external:", 1)[-1]
                ps = self.partner_stats.get(partner_id, PartnerStats())
                ps.total_capsules += 1
                ps.avg_threat = ((ps.avg_threat * (ps.total_capsules - 1)) + threat.score) / ps.total_capsules
                ps.last_seen = now
                ps.bernoulli_success, ps.bernoulli_trials = self._bernoulli_update(
                    is_good, ps.bernoulli_success, ps.bernoulli_trials
                )
                panom = 0.0
                if threat.score > 0.7:
                    panom += 0.5
                if ps.total_capsules > 50 and ps.avg_threat > 0.5:
                    panom += 0.3
                ps.anomaly_score = min(1.0, panom)
                self.partner_stats[partner_id] = ps

    def get_global_anomaly(self) -> float:
        with self._lock:
            if not self.node_stats:
                return 0.0
            return sum(ns.anomaly_score for ns in self.node_stats.values()) / len(self.node_stats)

    def get_global_bernoulli_prob(self) -> float:
        with self._lock:
            total_success = sum(ns.bernoulli_success for ns in self.node_stats.values())
            total_trials = sum(ns.bernoulli_trials for ns in self.node_stats.values())
            if total_trials == 0:
                return 0.5
            return (total_success + 1) / (total_trials + 2)

    def infer_missing_details(self, capsule: DataCapsule) -> Dict[str, Any]:
        inferred = {}
        if "confidence" not in capsule.payload:
            p = self.get_global_bernoulli_prob()
            inferred["confidence"] = round(p, 3)
        if "risk_hint" not in capsule.payload:
            global_anom = self.get_global_anomaly()
            pressure = (global_anom + capsule.priority / 10.0) / 2.0
            inferred["risk_hint"] = round(pressure, 3)
        return inferred

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "node_stats": {k: v.dict() for k, v in self.node_stats.items()},
                "partner_stats": {k: v.dict() for k, v in self.partner_stats.items()},
            }

    def from_dict(self, data: Dict[str, Any]):
        with self._lock:
            self.node_stats = {k: NodeStats(**v) for k, v in data.get("node_stats", {}).items()}
            self.partner_stats = {k: PartnerStats(**v) for k, v in data.get("partner_stats", {}).items()}

# ============================================================
# 9. Deep-learning prediction engine (MLP)
# ============================================================

class DLPredictor:
    """
    Lightweight deep-learning prediction engine:
    - Input: [avg_threat, anomaly, priority, bernoulli_p]
    - Output: predicted risk score (0..1)
    """
    def __init__(self):
        self.model = None
        self.device = "cpu"
        if HAS_TORCH:
            self.device = "cuda" if HAS_CUDA else "cpu"
            self.model = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            self.loss_fn = nn.MSELoss()
            self._load()

    def _load(self):
        if not HAS_TORCH:
            return
        if os.path.isfile(DL_MODEL_FILE):
            try:
                self.model.load_state_dict(torch.load(DL_MODEL_FILE, map_location=self.device))
                print("[DL] Loaded predictor model")
            except Exception as e:
                print(f"[DL] Failed to load model: {e}")

    def save(self):
        if not HAS_TORCH:
            return
        try:
            torch.save(self.model.state_dict(), DL_MODEL_FILE)
        except Exception as e:
            print(f"[DL] Failed to save model: {e}")

    def predict(self, avg_threat: float, anomaly: float, priority: float, bernoulli_p: float) -> float:
        if not HAS_TORCH or self.model is None:
            return max(0.0, min(1.0, (avg_threat + anomaly + (1 - bernoulli_p)) / 3.0))
        x = torch.tensor([[avg_threat, anomaly, priority, bernoulli_p]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.model(x)
        return float(y.item())

    def train_step(self, avg_threat: float, anomaly: float, priority: float, bernoulli_p: float, target_risk: float):
        if not HAS_TORCH or self.model is None:
            return
        self.model.train()
        x = torch.tensor([[avg_threat, anomaly, priority, bernoulli_p]], dtype=torch.float32, device=self.device)
        y_true = torch.tensor([[target_risk]], dtype=torch.float32, device=self.device)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ============================================================
# 10. Transformer-based prediction engine
# ============================================================

class TransformerPredictor:
    """
    Sequence-aware transformer predictor:
    - Maintains a short history of [avg_threat, anomaly, error_rate, flow]
    - Predicts next-step risk
    """
    def __init__(self, seq_len: int = 8):
        self.seq_len = seq_len
        self.device = "cpu"
        self.model = None
        self.history: List[List[float]] = []
        if HAS_TORCH:
            self.device = "cuda" if HAS_CUDA else "cpu"
            d_model = 16
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=32)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(self.device)
            self.proj_in = nn.Linear(4, d_model).to(self.device)
            self.proj_out = nn.Linear(d_model, 1).to(self.device)
            self.optimizer = optim.Adam(
                list(self.encoder.parameters()) + list(self.proj_in.parameters()) + list(self.proj_out.parameters()),
                lr=1e-3,
            )
            self.loss_fn = nn.MSELoss()
            self._load()

    def _load(self):
        if not HAS_TORCH:
            return
        if os.path.isfile(TRANSFORMER_MODEL_FILE):
            try:
                state = torch.load(TRANSFORMER_MODEL_FILE, map_location=self.device)
                self.encoder.load_state_dict(state["encoder"])
                self.proj_in.load_state_dict(state["proj_in"])
                self.proj_out.load_state_dict(state["proj_out"])
                print("[TRANSFORMER] Loaded transformer predictor")
            except Exception as e:
                print(f"[TRANSFORMER] Failed to load: {e}")

    def save(self):
        if not HAS_TORCH:
            return
        try:
            state = {
                "encoder": self.encoder.state_dict(),
                "proj_in": self.proj_in.state_dict(),
                "proj_out": self.proj_out.state_dict(),
            }
            torch.save(state, TRANSFORMER_MODEL_FILE)
        except Exception as e:
            print(f"[TRANSFORMER] Failed to save: {e}")

    def _prepare_seq(self) -> Optional[torch.Tensor]:
        if not HAS_TORCH:
            return None
        if len(self.history) < self.seq_len:
            return None
        seq = self.history[-self.seq_len:]
        x = torch.tensor(seq, dtype=torch.float32, device=self.device)  # [L, 4]
        x = x.unsqueeze(1)  # [L, 1, 4]
        x = self.proj_in(x)  # [L, 1, d_model]
        return x

    def update_history(self, avg_threat: float, anomaly: float, error_rate: float, flow: float):
        self.history.append([avg_threat, anomaly, error_rate, flow])
        if len(self.history) > 128:
            self.history.pop(0)

    def predict(self) -> float:
        if not HAS_TORCH:
            return 0.5
        x = self._prepare_seq()
        if x is None:
            return 0.5
        with torch.no_grad():
            enc = self.encoder(x)  # [L, 1, d_model]
            last = enc[-1, 0, :]   # [d_model]
            y = self.proj_out(last)
            y = torch.sigmoid(y)
        return float(y.item())

    def train_step(self, target_risk: float):
        if not HAS_TORCH:
            return
        x = self._prepare_seq()
        if x is None:
            return
        self.encoder.train()
        self.proj_in.train()
        self.proj_out.train()
        enc = self.encoder(x)
        last = enc[-1, 0, :]
        y_pred = torch.sigmoid(self.proj_out(last))
        y_true = torch.tensor([target_risk], dtype=torch.float32, device=self.device)
        loss = self.loss_fn(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ============================================================
# 11. Distributed storage manager (Redis + Cassandra stub)
# ============================================================

class DistributedStorage:
    """
    Simple abstraction over Redis (if available) and a Cassandra stub.
    Used for:
    - Storing global metrics
    - Sharing knowledge snapshots
    - Future extension to full distributed state
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = None
        if HAS_REDIS:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                print("[DIST] Connected to Redis")
            except Exception as e:
                print(f"[DIST] Redis unavailable: {e}")
                self.redis_client = None
        # Cassandra stub (placeholder)
        self.cassandra_enabled = False

    def set_metric(self, key: str, value: Any):
        if self.redis_client is not None:
            try:
                self.redis_client.set(key, json.dumps(value))
            except Exception as e:
                print(f"[DIST] Failed to set metric {key}: {e}")

    def get_metric(self, key: str, default: Any = None) -> Any:
        if self.redis_client is None:
            return default
        try:
            v = self.redis_client.get(key)
            if v is None:
                return default
            return json.loads(v)
        except Exception:
            return default

    def publish_snapshot(self, channel: str, snapshot: Dict[str, Any]):
        if self.redis_client is not None:
            try:
                self.redis_client.publish(channel, json.dumps(snapshot))
            except Exception as e:
                print(f"[DIST] Failed to publish snapshot: {e}")

DISTRIBUTED_STORAGE = DistributedStorage()

# ============================================================
# 12. Evolution: Genetic algorithm (self-mutation) + persistence
# ============================================================

class GeneticConfig(BaseModel):
    num_exec_workers: int
    num_sensor_workers: int
    num_opt_workers: int
    exploration_rate: float
    mutation_rate: float


class GeneticEngine:
    def __init__(self):
        self.population: List[GeneticConfig] = []
        self.current: Optional[GeneticConfig] = None

    def init_population(self):
        self.population = [
            GeneticConfig(
                num_exec_workers=random.randint(1, 4),
                num_sensor_workers=random.randint(1, 4),
                num_opt_workers=random.randint(1, 3),
                exploration_rate=random.uniform(0.05, 0.3),
                mutation_rate=random.uniform(0.05, 0.3),
            )
            for _ in range(5)
        ]
        self.current = self.population[0]
        print(f"[GENETIC] Initialized population, current={self.current.dict()}")

    def mutate(self, cfg: GeneticConfig) -> GeneticConfig:
        def mutate_int(v, low, high):
            if random.random() < cfg.mutation_rate:
                return max(low, min(high, v + random.choice([-1, 1])))
            return v

        def mutate_float(v, low, high):
            if random.random() < cfg.mutation_rate:
                delta = random.uniform(-0.05, 0.05)
                return max(low, min(high, v + delta))
            return v

        return GeneticConfig(
            num_exec_workers=mutate_int(cfg.num_exec_workers, 1, 8),
            num_sensor_workers=mutate_int(cfg.num_sensor_workers, 1, 8),
            num_opt_workers=mutate_int(cfg.num_opt_workers, 1, 4),
            exploration_rate=mutate_float(cfg.exploration_rate, 0.01, 0.5),
            mutation_rate=mutate_float(cfg.mutation_rate, 0.01, 0.5),
        )

    def evolve(self, reward: float):
        if not self.current:
            self.init_population()
        if reward > 0.7:
            self.current.mutation_rate = max(0.01, self.current.mutation_rate * 0.9)
        else:
            self.current.mutation_rate = min(0.5, self.current.mutation_rate * 1.1)
        self.current = self.mutate(self.current)
        print(f"[GENETIC] Evolved config: {self.current.dict()}")
        return self.current

    def save_state(self):
        ensure_state_dir()
        data = self.current.dict() if self.current else None
        save_json(GENETIC_STATE_FILE, data)

    def load_state(self):
        data = load_json(GENETIC_STATE_FILE, None)
        if data:
            self.current = GeneticConfig(**data)
            print(f"[GENETIC] Loaded persisted config: {self.current.dict()}")

# ============================================================
# 13. Evolution: Reinforcement learning (simple Q-agent) + persistence
# ============================================================

class RLAction(str, Enum):
    SCALE_UP_EXEC = "scale_up_exec"
    SCALE_DOWN_EXEC = "scale_down_exec"
    SCALE_UP_SENSOR = "scale_up_sensor"
    SCALE_DOWN_SENSOR = "scale_down_sensor"
    SCALE_UP_OPT = "scale_up_opt"
    SCALE_DOWN_OPT = "scale_down_opt"
    NOOP = "noop"


class RLAgent:
    def __init__(self, exploration_rate: float = 0.1, learning_rate: float = 0.1, discount: float = 0.9):
        self.q_table: Dict[str, Dict[RLAction, float]] = {}
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount = discount

    def _ensure_state(self, state: str):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in RLAction}

    def choose_action(self, state: str) -> RLAction:
        self._ensure_state(state)
        if random.random() < self.exploration_rate:
            return random.choice(list(RLAction))
        actions = self.q_table[state]
        return max(actions, key=actions.get)

    def update(self, state: str, action: RLAction, reward: float, next_state: str):
        self._ensure_state(state)
        self._ensure_state(next_state)
        old_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_q = old_q + self.learning_rate * (reward + self.discount * next_max - old_q)
        self.q_table[state][action] = new_q

    def save_state(self):
        ensure_state_dir()
        data = {
            "q_table": {s: {a.value: v for a, v in acts.items()} for s, acts in self.q_table.items()},
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "discount": self.discount,
        }
        save_json(RL_STATE_FILE, data)

    def load_state(self):
        data = load_json(RL_STATE_FILE, None)
        if not data:
            return
        self.exploration_rate = data.get("exploration_rate", self.exploration_rate)
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.discount = data.get("discount", self.discount)
        q_raw = data.get("q_table", {})
        self.q_table = {}
        for s, acts in q_raw.items():
            self.q_table[s] = {RLAction(a): v for a, v in acts.items()}
        print(f"[RL] Loaded Q-table with {len(self.q_table)} states")

# ============================================================
# 14. Swarm-level consensus (simple majority) + Raft stub
# ============================================================

class ConsensusState(BaseModel):
    proposal_id: str
    value: Any
    votes_for: int = 0
    votes_against: int = 0
    total_expected: int = 1


class ConsensusManager:
    def __init__(self):
        self.current: Optional[ConsensusState] = None
        self._lock = threading.Lock()

    def propose(self, value: Any, expected_peers: int):
        with self._lock:
            self.current = ConsensusState(
                proposal_id=str(uuid.uuid4()),
                value=value,
                votes_for=1,
                votes_against=0,
                total_expected=expected_peers + 1,
            )
            print(f"[CONSENSUS] Proposed {self.current.value} (id={self.current.proposal_id})")

    def handle_vote(self, peer_id: str, accept: bool):
        with self._lock:
            if not self.current:
                return
            if accept:
                self.current.votes_for += 1
            else:
                self.current.votes_against += 1
            print(f"[CONSENSUS] Vote from {peer_id}: {'accept' if accept else 'reject'} "
                  f"(for={self.current.votes_for}, against={self.current.votes_against})")

    def decision(self) -> Optional[bool]:
        with self._lock:
            if not self.current:
                return None
            if self.current.votes_for + self.current.votes_against >= self.current.total_expected:
                result = self.current.votes_for > self.current.votes_against
                print(f"[CONSENSUS] Final decision: {result} for value={self.current.value}")
                return result
            return None


class RaftNode:
    """
    Minimal Raft-style stub:
    - Tracks term, voted_for, log length
    - Not a full implementation, but placeholder for future real consensus
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[Dict[str, Any]] = []

    def append_entry(self, entry: Dict[str, Any]):
        self.log.append(entry)

    def request_vote(self, candidate_id: str, term: int) -> bool:
        if term < self.current_term:
            return False
        if self.voted_for is None or self.voted_for == candidate_id:
            self.voted_for = candidate_id
            self.current_term = term
            return True
        return False

RAFT_NODE: Optional[RaftNode] = None

# ============================================================
# 15. Borg system (Queen + Workers, including GPU/APU/NPU/LPU/DPU)
# ============================================================

class WorkerRole(str, Enum):
    EXECUTOR = "executor"
    SENSOR = "sensor"
    OPTIMIZER = "optimizer"
    GPU_OPTIMIZER = "gpu_optimizer"
    APU = "apu"
    NPU = "npu"
    LPU = "lpu"
    DPU = "dpu"


class BorgTaskType(str, Enum):
    EXECUTION = "execution"
    SENSE = "sense"
    OPTIMIZE = "optimize"
    GPU_OPTIMIZE = "gpu_optimize"
    APU_TASK = "apu_task"
    NPU_TASK = "npu_task"
    LPU_TASK = "lpu_task"
    DPU_TASK = "dpu_task"


class BorgTask(BaseModel):
    task_id: str
    task_type: BorgTaskType
    payload: Dict[str, Any]
    created_at: float


class BorgWorker(threading.Thread):
    def __init__(self, worker_id: str, role: WorkerRole, task_queue: "queue.Queue[BorgTask]"):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.role = role
        self.task_queue = task_queue
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        print(f"[BORG-WORKER] {self.worker_id} ({self.role.value}) started")
        while self._running:
            try:
                task: BorgTask = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.handle_task(task)
            except Exception as e:
                print(f"[BORG-WORKER] {self.worker_id} error on task {task.task_id}: {e}")
            finally:
                self.task_queue.task_done()

    def handle_task(self, task: BorgTask):
        if self.role == WorkerRole.EXECUTOR and task.task_type == BorgTaskType.EXECUTION:
            print(f"[BORG-WORKER] {self.worker_id} EXECUTOR running: {task.payload}")
        elif self.role == WorkerRole.SENSOR and task.task_type == BorgTaskType.SENSE:
            print(f"[BORG-WORKER] {self.worker_id} SENSOR analyzing: {task.payload}")
        elif self.role == WorkerRole.OPTIMIZER and task.task_type == BorgTaskType.OPTIMIZE:
            print(f"[BORG-WORKER] {self.worker_id} OPTIMIZER tuning: {task.payload}")
        elif self.role == WorkerRole.GPU_OPTIMIZER and task.task_type == BorgTaskType.GPU_OPTIMIZE:
            if HAS_TORCH and HAS_CUDA:
                print(f"[BORG-WORKER] {self.worker_id} GPU-OPTIMIZER running on CUDA: {task.payload}")
                x = torch.randn(512, 512, device="cuda")
                y = torch.matmul(x, x)
                _ = y.sum().item()
            else:
                print(f"[BORG-WORKER] {self.worker_id} GPU requested but CUDA/torch not available")
        elif self.role == WorkerRole.APU and task.task_type == BorgTaskType.APU_TASK:
            print(f"[BORG-WORKER] {self.worker_id} APU processing mixed-signal task: {task.payload}")
        elif self.role == WorkerRole.NPU and task.task_type == BorgTaskType.NPU_TASK:
            print(f"[BORG-WORKER] {self.worker_id} NPU running neural-style inference: {task.payload}")
        elif self.role == WorkerRole.LPU and task.task_type == BorgTaskType.LPU_TASK:
            print(f"[BORG-WORKER] {self.worker_id} LPU evaluating logic/rules: {task.payload}")
        elif self.role == WorkerRole.DPU and task.task_type == BorgTaskType.DPU_TASK:
            print(f"[BORG-WORKER] {self.worker_id} DPU optimizing data flow: {task.payload}")
        else:
            print(f"[BORG-WORKER] {self.worker_id} mismatched task {task.task_type.value}")


class BorgQueen:
    def __init__(self):
        self.executor_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.sensor_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.optimizer_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.gpu_optimizer_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.apu_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.npu_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.lpu_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.dpu_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.workers: List[BorgWorker] = []

    def spawn_workers(self, cfg: GeneticConfig):
        self.workers.clear()
        for i in range(cfg.num_exec_workers):
            w = BorgWorker(worker_id=f"exec-{i}", role=WorkerRole.EXECUTOR, task_queue=self.executor_queue)
            w.start()
            self.workers.append(w)
        for i in range(cfg.num_sensor_workers):
            w = BorgWorker(worker_id=f"sensor-{i}", role=WorkerRole.SENSOR, task_queue=self.sensor_queue)
            w.start()
            self.workers.append(w)
        for i in range(cfg.num_opt_workers):
            w = BorgWorker(worker_id=f"opt-{i}", role=WorkerRole.OPTIMIZER, task_queue=self.optimizer_queue)
            w.start()
            self.workers.append(w)
        if HAS_TORCH and HAS_CUDA:
            w = BorgWorker(worker_id="gpu-opt-0", role=WorkerRole.GPU_OPTIMIZER, task_queue=self.gpu_optimizer_queue)
            w.start()
            self.workers.append(w)
            print("[BORG-QUEEN] GPU optimizer worker spawned")
        w_apu = BorgWorker(worker_id="apu-0", role=WorkerRole.APU, task_queue=self.apu_queue)
        w_apu.start()
        self.workers.append(w_apu)
        w_npu = BorgWorker(worker_id="npu-0", role=WorkerRole.NPU, task_queue=self.npu_queue)
        w_npu.start()
        self.workers.append(w_npu)
        w_lpu = BorgWorker(worker_id="lpu-0", role=WorkerRole.LPU, task_queue=self.lpu_queue)
        w_lpu.start()
        self.workers.append(w_lpu)
        w_dpu = BorgWorker(worker_id="dpu-0", role=WorkerRole.DPU, task_queue=self.dpu_queue)
        w_dpu.start()
        self.workers.append(w_dpu)
        print(f"[BORG-QUEEN] Total workers: {len(self.workers)}")

    def submit_execution(self, payload: Dict[str, Any]):
        task = BorgTask(
            task_id=str(uuid.uuid4()),
            task_type=BorgTaskType.EXECUTION,
            payload=payload,
            created_at=time.time(),
        )
        self.executor_queue.put(task)

    def submit_sense(self, payload: Dict[str, Any]):
        task = BorgTask(
            task_id=str(uuid.uuid4()),
            task_type=BorgTaskType.SENSE,
            payload=payload,
            created_at=time.time(),
        )
        self.sensor_queue.put(task)

    def submit_optimize(self, payload: Dict[str, Any]):
        task = BorgTask(
            task_id=str(uuid.uuid4()),
            task_type=BorgTaskType.OPTIMIZE,
            payload=payload,
            created_at=time.time(),
        )
        self.optimizer_queue.put(task)

    def submit_gpu_optimize(self, payload: Dict[str, Any]):
        task = BorgTask(
            task_id=str(uuid.uuid4()),
            task_type=BorgTaskType.GPU_OPTIMIZE,
            payload=payload,
            created_at=time.time(),
        )
        self.gpu_optimizer_queue.put(task)

    def submit_apu_task(self, payload: Dict[str, Any]):
        task = BorgTask(
            task_id=str(uuid.uuid4()),
            task_type=BorgTaskType.APU_TASK,
            payload=payload,
            created_at=time.time(),
        )
        self.apu_queue.put(task)

    def submit_npu_task(self, payload: Dict[str, Any]):
        task = BorgTask(
            task_id=str(uuid.uuid4()),
            task_type=BorgTaskType.NPU_TASK,
            payload=payload,
            created_at=time.time(),
        )
        self.npu_queue.put(task)

    def submit_lpu_task(self, payload: Dict[str, Any]):
        task = BorgTask(
            task_id=str(uuid.uuid4()),
            task_type=BorgTaskType.LPU_TASK,
            payload=payload,
            created_at=time.time(),
        )
        self.lpu_queue.put(task)

    def submit_dpu_task(self, payload: Dict[str, Any]):
        task = BorgTask(
            task_id=str(uuid.uuid4()),
            task_type=BorgTaskType.DPU_TASK,
            payload=payload,
            created_at=time.time(),
        )
        self.dpu_queue.put(task)

# ============================================================
# 16. Real-time Queen (global risk engine)
# ============================================================

class RealTimeQueen:
    def __init__(self):
        self.nodes: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def update(self, node: str, events: List[Dict[str, Any]]):
        with self._lock:
            self.nodes[node] = events

    def global_risk(self) -> Dict[str, float]:
        risk: Dict[str, float] = {}
        with self._lock:
            for node, evts in self.nodes.items():
                for e in evts:
                    ent = e.get("entity")
                    score = e.get("score", 0.0)
                    if ent is None:
                        continue
                    risk[ent] = risk.get(ent, 0.0) + score
        return {k: v for k, v in risk.items() if v > 1.5}

# ============================================================
# 17. Plugin architecture + genetic plugin evolution
# ============================================================

PLUGINS_DIR = "plugins"

class PluginManager:
    """
    Plugin system:
    - Loads Python modules from ./plugins
    - Looks for `register_plugin(node_context)` function
    - Tracks plugin performance scores for genetic evolution
    """
    def __init__(self):
        self.plugins = []
        self.plugin_scores: Dict[str, float] = {}

    def load_plugins(self, node_context: Dict[str, Any]):
        if not os.path.isdir(PLUGINS_DIR):
            return
        sys.path.insert(0, os.path.abspath(PLUGINS_DIR))
        for fname in os.listdir(PLUGINS_DIR):
            if not fname.endswith(".py"):
                continue
            mod_name = fname[:-3]
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "register_plugin") and callable(mod.register_plugin):
                    mod.register_plugin(node_context)
                    self.plugins.append(mod_name)
                    self.plugin_scores.setdefault(mod_name, 0.5)
                    print(f"[PLUGIN] Loaded plugin: {mod_name}")
            except Exception as e:
                print(f"[PLUGIN] Failed to load {mod_name}: {e}")

    def reward_plugin(self, name: str, reward: float):
        if name not in self.plugin_scores:
            return
        self.plugin_scores[name] = max(0.0, min(1.0, self.plugin_scores[name] * 0.9 + reward * 0.1))

    def evolve_plugins(self):
        if not self.plugins:
            return
        sorted_plugins = sorted(self.plugins, key=lambda p: self.plugin_scores.get(p, 0.5), reverse=True)
        keep = sorted_plugins[: max(1, len(sorted_plugins) // 2)]
        maybe_disable = sorted_plugins[len(keep):]
        for p in maybe_disable:
            if random.random() < 0.3:
                print(f"[PLUGIN] Evolution: disabling plugin {p} (score={self.plugin_scores.get(p,0.5):.2f})")
                self.plugins.remove(p)

PLUGIN_MANAGER = PluginManager()

# ============================================================
# 18. Queen-of-Queens (cluster super-brain) with DL + Transformer + reward wiring
# ============================================================

class SuperQueen:
    def __init__(
        self,
        queen: BorgQueen,
        genetic: GeneticEngine,
        rl: RLAgent,
        consensus: ConsensusManager,
        knowledge: KnowledgeStore,
        realtime_queen: RealTimeQueen,
        dl_predictor: DLPredictor,
        transformer_predictor: TransformerPredictor,
    ):
        self.queen = queen
        self.genetic = genetic
        self.rl = rl
        self.consensus = consensus
        self.knowledge = knowledge
        self.realtime_queen = realtime_queen
        self.dl_predictor = dl_predictor
        self.transformer_predictor = transformer_predictor
        self.consciousness_state: ConsciousnessState = ConsciousnessState.NORMAL

    def decide_and_evolve(self, avg_threat: float, avg_latency: float, error_rate: float, peer_count: int, flow: float):
        global_anomaly = self.knowledge.get_global_anomaly()
        bernoulli_p = self.knowledge.get_global_bernoulli_prob()

        self.transformer_predictor.update_history(avg_threat, global_anomaly, error_rate, flow)

        predicted_risk_mlp = self.dl_predictor.predict(
            avg_threat=avg_threat,
            anomaly=global_anomaly,
            priority=0.5,
            bernoulli_p=bernoulli_p,
        )
        predicted_risk_seq = self.transformer_predictor.predict()

        combined_pred = (predicted_risk_mlp + predicted_risk_seq) / 2.0

        raw_penalty = (avg_threat + avg_latency + error_rate + global_anomaly + combined_pred) / 5.0
        reward = max(0.0, 1.0 - raw_penalty)

        self.dl_predictor.train_step(
            avg_threat=avg_threat,
            anomaly=global_anomaly,
            priority=0.5,
            bernoulli_p=bernoulli_p,
            target_risk=raw_penalty,
        )
        self.transformer_predictor.train_step(target_risk=raw_penalty)

        cfg = self.genetic.evolve(reward)

        state = (
            f"thr:{round(avg_threat,2)}|lat:{round(avg_latency,2)}|"
            f"err:{round(error_rate,2)}|anom:{round(global_anomaly,2)}|p:{round(bernoulli_p,2)}"
        )
        action = self.rl.choose_action(state)
        print(f"[SUPER-QUEEN] State={state}, Action={action.value}, Reward={reward:.3f}, PredRisk={combined_pred:.3f}")

        next_state = state
        self.rl.update(state, action, reward, next_state)

        if action == RLAction.SCALE_UP_EXEC:
            cfg.num_exec_workers += 1
        elif action == RLAction.SCALE_DOWN_EXEC and cfg.num_exec_workers > 1:
            cfg.num_exec_workers -= 1
        elif action == RLAction.SCALE_UP_SENSOR:
            cfg.num_sensor_workers += 1
        elif action == RLAction.SCALE_DOWN_SENSOR and cfg.num_sensor_workers > 1:
            cfg.num_sensor_workers -= 1
        elif action == RLAction.SCALE_UP_OPT:
            cfg.num_opt_workers += 1
        elif action == RLAction.SCALE_DOWN_OPT and cfg.num_opt_workers > 1:
            cfg.num_opt_workers -= 1

        self.consensus.propose(value=cfg.dict(), expected_peers=peer_count)
        decision = self.consensus.decision()
        if decision is None or decision:
            print("[SUPER-QUEEN] Applying evolved config to Queen")
            self.queen.spawn_workers(cfg)
            self.genetic.save_state()
            self.rl.save_state()
            self.dl_predictor.save()
            self.transformer_predictor.save()
        else:
            print("[SUPER-QUEEN] Consensus rejected new config")

        self.consciousness_state = consciousness_transition(
            self.consciousness_state, avg_threat, global_anomaly
        )
        print(f"[SUPER-QUEEN] Consciousness state: {self.consciousness_state.value}")

        DISTRIBUTED_STORAGE.set_metric(
            f"node:{NODE_IDENTITY.node_id}:metrics",
            {
                "avg_threat": avg_threat,
                "avg_latency": avg_latency,
                "error_rate": error_rate,
                "anomaly": global_anomaly,
                "reward": reward,
                "predicted_risk": combined_pred,
            },
        )

        PLUGINS_TO_REWARD = list(PLUGIN_MANAGER.plugins)
        if PLUGINS_TO_REWARD:
            per_plugin_reward = reward / len(PLUGINS_TO_REWARD)
            for p in PLUGINS_TO_REWARD:
                PLUGIN_MANAGER.reward_plugin(p, per_plugin_reward)

# ============================================================
# 19. Dream-simulation subsystem
# ============================================================

class DreamEngine:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def _loop(self):
        while True:
            time.sleep(15)
            if SUPER_QUEEN.consciousness_state not in {ConsciousnessState.DREAMING, ConsciousnessState.SLEEP}:
                continue
            dream_capsule = DataCapsule(
                capsule_id=str(uuid.uuid4()),
                capsule_type=CapsuleType.EVENT,
                origin_node_id=NODE_IDENTITY.node_id,
                target_scope="internal:dream",
                timestamp=time.time(),
                priority=3,
                ttl=10.0,
                payload={
                    "dream": True,
                    "hypothesis": "future_anomaly",
                    "confidence": random.uniform(0.3, 0.9),
                },
                classification=CapsuleClassification.INTERNAL,
                meta={"mode": "dream"},
            )
            dream_capsule.integrity_hash = compute_capsule_hash(dream_capsule)
            print(f"[DREAM] Emitting speculative capsule {dream_capsule.capsule_id}")
            self.event_bus.publish("capsules", dream_capsule)

DREAM_ENGINE = DreamEngine(event_bus=None)  # wired later

# ============================================================
# 20. Memory-consolidation "sleep" cycle
# ============================================================

class SleepEngine:
    """
    Periodically enters SLEEP state:
    - Consolidates memory (knowledge, RL, genetic, DL, transformer)
    - Publishes snapshots to distributed storage
    - Then returns to NORMAL/DREAMING
    """
    def __init__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while True:
            time.sleep(120)
            print("[SLEEP] Entering memory-consolidation cycle")
            prev_state = SUPER_QUEEN.consciousness_state
            SUPER_QUEEN.consciousness_state = ConsciousnessState.SLEEP
            try:
                save_json(KNOWLEDGE_STATE_FILE, KNOWLEDGE_STORE.to_dict())
                GENETIC_ENGINE.save_state()
                RL_AGENT.save_state()
                DL_PREDICTOR.save()
                TRANSFORMER_PREDICTOR.save()
                snapshot = {
                    "node": NODE_IDENTITY.node_id,
                    "knowledge": KNOWLEDGE_STORE.to_dict(),
                    "genetic": GENETIC_ENGINE.current.dict() if GENETIC_ENGINE.current else None,
                }
                DISTRIBUTED_STORAGE.publish_snapshot("borg_snapshots", snapshot)
                print("[SLEEP] Memory consolidation complete")
            except Exception as e:
                print(f"[SLEEP] Error during consolidation: {e}")
            finally:
                SUPER_QUEEN.consciousness_state = prev_state
                print(f"[SLEEP] Exiting sleep, state={SUPER_QUEEN.consciousness_state.value}")

SLEEP_ENGINE = SleepEngine()

# ============================================================
# 21. BorgManager (integrates everything + metrics)
# ============================================================

class BorgManager:
    def __init__(self, queen: BorgQueen, super_queen: SuperQueen, knowledge: KnowledgeStore, realtime_queen: RealTimeQueen):
        self.queen = queen
        self.super_queen = super_queen
        self.knowledge = knowledge
        self.realtime_queen = realtime_queen
        self._opt_thread = threading.Thread(target=self._optimizer_loop, daemon=True)
        self._running = False
        self._threat_history: List[float] = []
        self._latency_history: List[float] = []
        self._error_count: int = 0
        self._total_capsules: int = 0
        self._flow_history: List[float] = []

    def start(self):
        self._running = True
        if not self.super_queen.genetic.current:
            self.super_queen.genetic.load_state()
            if not self.super_queen.genetic.current:
                self.super_queen.genetic.init_population()
        self.super_queen.rl.load_state()
        self.queen.spawn_workers(self.super_queen.genetic.current)
        self._opt_thread.start()

        def flow_loop():
            while True:
                time.sleep(10)
                flow = self._total_capsules
                self._flow_history.append(flow)
                if len(self._flow_history) > 50:
                    self._flow_history.pop(0)
                self._total_capsules = 0
        threading.Thread(target=flow_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def handle_capsule_for_borg(self, capsule: DataCapsule, threat: ThreatScore, latency: float, error: bool):
        self._threat_history.append(threat.score)
        self._latency_history.append(latency)
        if len(self._threat_history) > 200:
            self._threat_history.pop(0)
        if len(self._latency_history) > 200:
            self._latency_history.pop(0)

        self._total_capsules += 1
        if error:
            self._error_count += 1

        evt = {
            "entity": capsule.origin_node_id,
            "score": threat.score,
            "type": capsule.capsule_type.value,
        }
        self.realtime_queen.update(NODE_IDENTITY.node_id, [evt])

        if capsule.capsule_type in {CapsuleType.COMMAND, CapsuleType.CONTROL}:
            self.queen.submit_execution({
                "origin": capsule.origin_node_id,
                "command": capsule.payload,
                "meta": capsule.meta,
            })
            self.queen.submit_lpu_task({
                "origin": capsule.origin_node_id,
                "logic": capsule.payload,
            })
        elif capsule.capsule_type in {CapsuleType.TELEMETRY, CapsuleType.EVENT, CapsuleType.ALERT}:
            self.queen.submit_sense({
                "origin": capsule.origin_node_id,
                "data": capsule.payload,
                "type": capsule.capsule_type.value,
            })
            self.queen.submit_dpu_task({
                "origin": capsule.origin_node_id,
                "stream": capsule.payload,
            })
            self.queen.submit_apu_task({
                "origin": capsule.origin_node_id,
                "mixed_signal": capsule.payload,
            })
            if capsule.meta.get("neural_hint"):
                self.queen.submit_npu_task({
                    "origin": capsule.origin_node_id,
                    "neural_payload": capsule.payload,
                })
        elif capsule.capsule_type == CapsuleType.SYNC:
            self.queen.submit_optimize({
                "sync_info": capsule.payload,
                "source": capsule.origin_node_id,
            })
            if HAS_TORCH and HAS_CUDA:
                self.queen.submit_gpu_optimize({
                    "sync_info": capsule.payload,
                    "source": capsule.origin_node_id,
                    "gpu_hint": True,
                })

    def _optimizer_loop(self):
        while self._running:
            time.sleep(20)
            if not self._threat_history:
                avg_threat = 0.1
            else:
                avg_threat = sum(self._threat_history) / len(self._threat_history)
            if not self._latency_history:
                avg_latency = 0.1
            else:
                avg_latency = sum(self._latency_history) / len(self._latency_history)
            total_events = self._error_count + max(1, sum(self._flow_history) if self._flow_history else 1)
            error_rate = self._error_count / total_events
            flow = self._flow_history[-1] if self._flow_history else 0.0

            peer_count = len(SYNC_PEERS)
            self.super_queen.decide_and_evolve(avg_threat, avg_latency, error_rate, peer_count, flow)
            PLUGINS_EVOLUTION_CHANCE = 0.3
            if random.random() < PLUGINS_EVOLUTION_CHANCE:
                PLUGIN_MANAGER.evolve_plugins()

# ============================================================
# 22. Self-replication + autonomous deployment stubs
# ============================================================

def self_replicate(new_port: Optional[int] = None) -> Dict[str, Any]:
    try:
        src = os.path.abspath(sys.argv[0])
        base, ext = os.path.splitext(src)
        ts = int(time.time())
        dst = f"{base}_clone_{ts}{ext}"
        shutil.copy2(src, dst)
        cmd = None
        if new_port is not None:
            cmd = f"{sys.executable} {os.path.basename(dst)} --port {new_port}"
        print(f"[REPLICATE] Cloned to {dst}")
        return {"source": src, "clone": dst, "run_command": cmd}
    except Exception as e:
        print(f"[REPLICATE] Failed: {e}")
        return {"error": str(e)}

def deploy_to_host(host: str, port: int) -> Dict[str, Any]:
    """
    Autonomous deployment stub:
    - Generates a command that could be used to deploy on a remote host.
    - Real SSH/agent integration can be added later.
    """
    src = os.path.abspath(sys.argv[0])
    cmd = f"scp {src} {host}:/tmp/borg_node.py && ssh {host} '{sys.executable} /tmp/borg_node.py --port {port}'"
    print(f"[DEPLOY] Suggested deployment command:\n{cmd}")
    return {"host": host, "port": port, "command": cmd}

# ============================================================
# 23. FastAPI app & global state
# ============================================================

app = FastAPI(title="Unified Borg Organism vOmega", version="3.0.0")

NODE_IDENTITY: NodeIdentity
CURRENT_PERSONA: Persona
EVENT_BUS = EventBus()
EXTERNAL_PARTNERS: Dict[str, ExternalHandshakeRequest] = {}
SYNC_PEERS: List[str] = []

KNOWLEDGE_STORE = KnowledgeStore()
GENETIC_ENGINE = GeneticEngine()
RL_AGENT = RLAgent()
CONSENSUS_MANAGER = ConsensusManager()
BORG_QUEEN = BorgQueen()
REALTIME_QUEEN = RealTimeQueen()
DL_PREDICTOR = DLPredictor()
TRANSFORMER_PREDICTOR = TransformerPredictor()
SUPER_QUEEN = SuperQueen(BORG_QUEEN, GENETIC_ENGINE, RL_AGENT, CONSENSUS_MANAGER, KNOWLEDGE_STORE, REALTIME_QUEEN, DL_PREDICTOR, TRANSFORMER_PREDICTOR)
BORG_MANAGER = BorgManager(BORG_QUEEN, SUPER_QUEEN, KNOWLEDGE_STORE, REALTIME_QUEEN)

DREAM_ENGINE.event_bus = EVENT_BUS  # wire dream engine
RAFT_NODE = None  # will be set in main

# ============================================================
# 24. Core behaviors
# ============================================================

def handle_capsule(capsule: DataCapsule):
    global CURRENT_PERSONA

    start = time.time()
    error = False

    if capsule.integrity_hash:
        expected = compute_capsule_hash(capsule)
        if expected != capsule.integrity_hash:
            print(f"[{NODE_IDENTITY.node_id}] Integrity mismatch for capsule {capsule.capsule_id}")
            error = True

    threat = evaluate_threat(capsule, NODE_IDENTITY)
    KNOWLEDGE_STORE.update_from_capsule(capsule, threat)

    inferred = KNOWLEDGE_STORE.infer_missing_details(capsule)
    if inferred:
        capsule.payload.update(inferred)
        print(f"[{NODE_IDENTITY.node_id}] Inferred missing details: {inferred}")

    new_persona = persona_transition(CURRENT_PERSONA, threat)
    if new_persona != CURRENT_PERSONA:
        print(f"[{NODE_IDENTITY.node_id}] Persona change: {CURRENT_PERSONA.value} -> {new_persona.value} (threat={threat.score})")
        CURRENT_PERSONA = new_persona

    if CURRENT_PERSONA == Persona.QUARANTINED:
        print(f"[{NODE_IDENTITY.node_id}] QUARANTINED: dropping capsule {capsule.capsule_id}")
        latency = time.time() - start
        BORG_MANAGER.handle_capsule_for_borg(capsule, threat, latency, error=True)
        return

    print(f"[{NODE_IDENTITY.node_id}] [{CURRENT_PERSONA.value}] Received {capsule.capsule_type.value} from {capsule.origin_node_id}: {capsule.payload}")

    if capsule.capsule_type == CapsuleType.COMMAND and CURRENT_PERSONA in {Persona.OPERATOR, Persona.AUTONOMOUS}:
        print(f"[{NODE_IDENTITY.node_id}] Executing command (local): {capsule.payload}")
    elif capsule.capsule_type == CapsuleType.SYNC:
        print(f"[{NODE_IDENTITY.node_id}] Applying sync data: {capsule.payload}")

    latency = time.time() - start
    BORG_MANAGER.handle_capsule_for_borg(capsule, threat, latency, error)

def send_capsule(
    target_base_url: str,
    capsule_type: CapsuleType,
    payload: Dict[str, Any],
    target_scope: str,
    classification: CapsuleClassification = CapsuleClassification.INTERNAL,
    priority: int = 5,
    ttl: float = 30.0,
    meta: Dict[str, Any] = None,
):
    if meta is None:
        meta = {}

    capsule = DataCapsule(
        capsule_id=str(uuid.uuid4()),
        capsule_type=capsule_type,
        origin_node_id=NODE_IDENTITY.node_id,
        target_scope=target_scope,
        timestamp=time.time(),
        priority=priority,
        ttl=ttl,
        payload=payload,
        classification=classification,
        meta=meta,
    )
    capsule.integrity_hash = compute_capsule_hash(capsule)

    url = f"{target_base_url.rstrip('/')}/ingest"
    try:
        resp = requests.post(url, json=capsule.dict(), timeout=5)
        print(f"[{NODE_IDENTITY.node_id}] Sent {capsule_type.value} to {url}, status={resp.status_code}")
        try:
            print("Response:", resp.json())
        except Exception:
            print("Raw response:", resp.text)
    except Exception as e:
        print(f"[{NODE_IDENTITY.node_id}] Error sending capsule to {url}: {e}")

def sync_with_peers():
    while True:
        time.sleep(10)
        if not SYNC_PEERS:
            continue
        payload = {
            "node_id": NODE_IDENTITY.node_id,
            "persona": CURRENT_PERSONA.value,
            "timestamp": time.time(),
        }
        for peer in SYNC_PEERS:
            send_capsule(
                target_base_url=peer,
                capsule_type=CapsuleType.SYNC,
                payload=payload,
                target_scope=f"node:{peer}",
                classification=CapsuleClassification.INTERNAL,
                priority=3,
                ttl=10.0,
                meta={"sync": True},
            )

# ============================================================
# 25. Visual dashboard with charts
# ============================================================

@app.get("/", response_class=HTMLResponse)
def dashboard_root():
    html = f"""
    <html>
    <head>
        <title>Borg Organism Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; background:#05060a; color:#e0e0e0; }}
            .card {{ border:1px solid #333; border-radius:8px; padding:16px; margin:16px; background:#11131a; }}
            h1,h2,h3 {{ color:#7dd3fc; }}
            .tag {{ display:inline-block; padding:2px 6px; margin:2px; border-radius:4px; background:#1e293b; font-size:12px; }}
            pre {{ background:#020617; padding:8px; border-radius:4px; overflow-x:auto; }}
            canvas {{ background:#020617; border-radius:4px; }}
            a {{ color:#38bdf8; }}
        </style>
    </head>
    <body>
        <h1>Borg Organism vOmega – Node: {NODE_IDENTITY.node_id}</h1>
        <div class="card">
            <h2>Identity</h2>
            <p>Cluster: <b>{NODE_IDENTITY.cluster_id}</b> | Role: <b>{NODE_IDENTITY.role.value}</b> | Region: <b>{NODE_IDENTITY.region}</b></p>
            <p>Persona: <span class="tag">{CURRENT_PERSONA.value}</span> |
               Consciousness: <span class="tag">{SUPER_QUEEN.consciousness_state.value}</span></p>
        </div>
        <div class="card">
            <h2>Workers & Evolution</h2>
            <p>Workers: <b>{len(BORG_QUEEN.workers)}</b> | GPU: <b>{"ON" if (HAS_TORCH and HAS_CUDA) else "OFF"}</b></p>
            <pre id="genetic-config">{json.dumps(GENETIC_ENGINE.current.dict() if GENETIC_ENGINE.current else {}, indent=2)}</pre>
        </div>
        <div class="card">
            <h2>Knowledge & Risk</h2>
            <p>Global anomaly: <b id="anom-val">{round(KNOWLEDGE_STORE.get_global_anomaly(),3)}</b></p>
            <p>Bernoulli success p: <b id="bern-val">{round(KNOWLEDGE_STORE.get_global_bernoulli_prob(),3)}</b></p>
            <h3>Real-time global risk</h3>
            <pre id="risk-json">{json.dumps(REALTIME_QUEEN.global_risk(), indent=2)}</pre>
        </div>
        <div class="card">
            <h2>Metrics Charts</h2>
            <canvas id="chart" width="600" height="200"></canvas>
            <p style="font-size:12px;color:#9ca3af;">Chart shows recent avg_threat and anomaly from /dashboard/json.</p>
        </div>
        <div class="card">
            <h2>Plugins</h2>
            <p>Loaded plugins: <b id="plugins-list">{", ".join(PLUGIN_MANAGER.plugins) if PLUGIN_MANAGER.plugins else "None"}</b></p>
        </div>
        <div class="card">
            <h2>API Quick Links</h2>
            <ul>
                <li><a href="/identity">/identity</a></li>
                <li><a href="/capabilities">/capabilities</a></li>
                <li><a href="/partners">/partners</a></li>
                <li><a href="/dashboard/json">/dashboard/json</a></li>
            </ul>
        </div>
        <script>
            async function refreshDashboard() {{
                try {{
                    const res = await fetch('/dashboard/json');
                    const data = await res.json();
                    document.getElementById('anom-val').innerText = data.knowledge_global_anomaly.toFixed(3);
                    document.getElementById('bern-val').innerText = data.knowledge_bernoulli_p.toFixed(3);
                    document.getElementById('risk-json').innerText = JSON.stringify(data.realtime_risk, null, 2);
                    document.getElementById('plugins-list').innerText = data.plugins.length ? data.plugins.join(', ') : 'None';
                    drawChart(data.metrics_history);
                }} catch (e) {{
                    console.log('Dashboard refresh error', e);
                }}
            }}
            function drawChart(history) {{
                const canvas = document.getElementById('chart');
                if (!canvas || !history || !history.length) return;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0,0,canvas.width,canvas.height);
                const w = canvas.width;
                const h = canvas.height;
                const n = history.length;
                const maxVal = 1.0;
                ctx.strokeStyle = '#1f2937';
                ctx.beginPath();
                ctx.moveTo(0, h/2);
                ctx.lineTo(w, h/2);
                ctx.stroke();
                function plotLine(key, color, offset) {{
                    ctx.strokeStyle = color;
                    ctx.beginPath();
                    history.forEach((m, i) => {{
                        const x = (i/(n-1||1))*w;
                        const v = m[key] || 0;
                        const y = h - (v/maxVal)*h;
                        if (i===0) ctx.moveTo(x,y+offset);
                        else ctx.lineTo(x,y+offset);
                    }});
                    ctx.stroke();
                }}
                plotLine('avg_threat', '#f97316', 0);
                plotLine('anomaly', '#22c55e', 0);
            }}
            setInterval(refreshDashboard, 5000);
            refreshDashboard();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/dashboard/json", response_class=JSONResponse)
def dashboard_json():
    metrics_history = DISTRIBUTED_STORAGE.get_metric(f"node:{NODE_IDENTITY.node_id}:metrics_history", [])
    return {
        "identity": NODE_IDENTITY.dict(),
        "persona": CURRENT_PERSONA.value,
        "consciousness": SUPER_QUEEN.consciousness_state.value,
        "workers": len(BORG_QUEEN.workers),
        "gpu_enabled": HAS_TORCH and HAS_CUDA,
        "genetic_config": GENETIC_ENGINE.current.dict() if GENETIC_ENGINE.current else None,
        "knowledge": KNOWLEDGE_STORE.to_dict(),
        "knowledge_global_anomaly": KNOWLEDGE_STORE.get_global_anomaly(),
        "knowledge_bernoulli_p": KNOWLEDGE_STORE.get_global_bernoulli_prob(),
        "realtime_risk": REALTIME_QUEEN.global_risk(),
        "plugins": PLUGIN_MANAGER.plugins,
        "metrics_history": metrics_history,
    }

# ============================================================
# 26. API endpoints
# ============================================================

@app.get("/identity")
def get_identity():
    return {
        "identity": NODE_IDENTITY.dict(),
        "persona": CURRENT_PERSONA.value,
        "consciousness": SUPER_QUEEN.consciousness_state.value,
    }

@app.get("/capabilities")
def get_capabilities():
    return {
        "node_id": NODE_IDENTITY.node_id,
        "role": NODE_IDENTITY.role.value,
        "cluster_id": NODE_IDENTITY.cluster_id,
        "supported_capsule_types": [ct.value for ct in CapsuleType],
        "personas": [p.value for p in Persona],
        "current_persona": CURRENT_PERSONA.value,
        "consciousness": SUPER_QUEEN.consciousness_state.value,
        "borg": {
            "modes": ["scheduler", "organism", "optimizer"],
            "workers": len(BORG_QUEEN.workers),
            "gpu_enabled": HAS_TORCH and HAS_CUDA,
            "accelerators": ["GPU", "APU", "NPU", "LPU", "DPU"],
        },
        "evolution": {
            "genetic_config": GENETIC_ENGINE.current.dict() if GENETIC_ENGINE.current else None,
            "rl_exploration_rate": RL_AGENT.exploration_rate,
        },
        "knowledge": KNOWLEDGE_STORE.to_dict(),
        "realtime_risk": REALTIME_QUEEN.global_risk(),
        "plugins": PLUGIN_MANAGER.plugins,
    }

@app.post("/ingest")
def ingest_capsule(capsule: DataCapsule):
    EVENT_BUS.publish("capsules", capsule)
    return {
        "node_id": NODE_IDENTITY.node_id,
        "received_capsule_id": capsule.capsule_id,
        "persona": CURRENT_PERSONA.value,
        "consciousness": SUPER_QUEEN.consciousness_state.value,
        "status": "accepted",
    }

@app.post("/handshake/external", response_model=ExternalHandshakeResponse)
def external_handshake(req: ExternalHandshakeRequest):
    if not req.auth_token or len(req.auth_token) < 8:
        return ExternalHandshakeResponse(
            accepted=False,
            reason="invalid_auth_token",
        )

    EXTERNAL_PARTNERS[req.partner_id] = req

    allowed_types = [
        CapsuleType.TELEMETRY,
        CapsuleType.EVENT,
    ]
    if NODE_IDENTITY.trust_tier == TrustTier.HIGH and CURRENT_PERSONA != Persona.QUARANTINED:
        allowed_types.append(CapsuleType.COMMAND)

    channel = f"external.partner.{req.partner_id}"

    return ExternalHandshakeResponse(
        accepted=True,
        assigned_partner_id=req.partner_id,
        allowed_capsule_types=allowed_types,
        rate_limit_per_sec=min(req.capabilities.max_rate_per_sec, 50),
        channel=channel,
    )

@app.get("/partners")
def list_partners():
    return {
        "node_id": NODE_IDENTITY.node_id,
        "partners": list(EXTERNAL_PARTNERS.keys()),
    }

@app.post("/sync/peers")
def set_sync_peers(peers: List[str]):
    SYNC_PEERS.clear()
    SYNC_PEERS.extend(peers)
    return {"node_id": NODE_IDENTITY.node_id, "sync_peers": SYNC_PEERS}

@app.post("/replicate")
def api_replicate(new_port: Optional[int] = None):
    return self_replicate(new_port=new_port)

@app.post("/deploy")
def api_deploy(host: str, port: int):
    return deploy_to_host(host, port)

# ============================================================
# 27. Main / bootstrap
# ============================================================

def main():
    global NODE_IDENTITY, CURRENT_PERSONA, RAFT_NODE

    ensure_state_dir()

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cluster-id", type=str, default="alpha")
    parser.add_argument("--role", type=str, default="sensor",
                        choices=[r.value for r in NodeRole])
    parser.add_argument("--region", type=str, default="us-central")
    parser.add_argument("--trust", type=str, default="medium",
                        choices=[t.value for t in TrustTier])
    parser.add_argument("--sync-peer", action="append", default=[],
                        help="Base URL of a peer for sync, e.g. http://localhost:8001")
    parser.add_argument("--send-test-to", type=str, default=None,
                        help="Base URL of another node to send a test capsule to on startup")
    args = parser.parse_args()

    NODE_IDENTITY = generate_node_identity(
        cluster_id=args.cluster_id,
        role=NodeRole(args.role),
        region=args.region,
        trust_tier=TrustTier(args.trust),
    )
    CURRENT_PERSONA = Persona.OBSERVER
    RAFT_NODE = RaftNode(NODE_IDENTITY.node_id)

    EVENT_BUS.subscribe("capsules", handle_capsule)

    ks_data = load_json(KNOWLEDGE_STATE_FILE, None)
    if ks_data:
        KNOWLEDGE_STORE.from_dict(ks_data)
        print("[KNOWLEDGE] Loaded persisted knowledge store")

    SYNC_PEERS.extend(args.sync_peer)
    if SYNC_PEERS:
        t_sync = threading.Thread(target=sync_with_peers, daemon=True)
        t_sync.start()

    BORG_MANAGER.start()
    DREAM_ENGINE.start()
    SLEEP_ENGINE.start()

    node_context = {
        "NODE_IDENTITY": NODE_IDENTITY,
        "EVENT_BUS": EVENT_BUS,
        "send_capsule": send_capsule,
        "KNOWLEDGE_STORE": KNOWLEDGE_STORE,
        "BORG_QUEEN": BORG_QUEEN,
        "SUPER_QUEEN": SUPER_QUEEN,
        "BORG_MANAGER": BORG_MANAGER,
        "DISTRIBUTED_STORAGE": DISTRIBUTED_STORAGE,
    }
    PLUGIN_MANAGER.load_plugins(node_context)

    if args.send_test_to:
        def delayed_send():
            time.sleep(2)
            send_capsule(
                target_base_url=args.send_test_to,
                capsule_type=CapsuleType.TELEMETRY,
                payload={"temp": 42.5, "status": "nominal"},
                target_scope=f"node:{args.send_test_to}",
                classification=CapsuleClassification.INTERNAL,
                priority=5,
                ttl=30.0,
                meta={"test": True, "neural_hint": True},
            )
        threading.Thread(target=delayed_send, daemon=True).start()

    def persist_loop():
        history_key = f"node:{NODE_IDENTITY.node_id}:metrics_history"
        while True:
            time.sleep(30)
            save_json(KNOWLEDGE_STATE_FILE, KNOWLEDGE_STORE.to_dict())
            metrics = DISTRIBUTED_STORAGE.get_metric(f"node:{NODE_IDENTITY.node_id}:metrics", {})
            history = DISTRIBUTED_STORAGE.get_metric(history_key, [])
            if metrics:
                history.append({
                    "ts": time.time(),
                    "avg_threat": metrics.get("avg_threat", 0.0),
                    "anomaly": metrics.get("anomaly", 0.0),
                })
                if len(history) > 100:
                    history = history[-100:]
                DISTRIBUTED_STORAGE.set_metric(history_key, history)
    threading.Thread(target=persist_loop, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
