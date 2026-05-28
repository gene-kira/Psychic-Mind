"""
unified_evolving_borg_data_node.py

Single-file "evolving data organism" node with:
- Universal cross-platform autoloader for dependencies
- Node identity
- Data capsule model
- In-process event bus
- Persona states and transitions
- Threat matrix integration (stub)
- Sync topology hooks
- External system handshake protocol

Borg system:
- Classic Borg Scheduler (Queen + executor workers)
- Borg Organism (Queen + sensor/motor workers)
- Borg Augmentation Engine (optimizer workers)

Evolution system:
1. Genetic algorithms for self-mutation (config evolution)
2. Reinforcement learning for long-term adaptation (policy tuning)
3. Swarm-level consensus (simple majority-based consensus)
4. GPU-accelerated workers (if torch+CUDA available)
5. Queen-of-Queens (cluster super-brain abstraction)

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

# Optional GPU support
try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_TORCH = False
    HAS_CUDA = False

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
# 7. Evolution: Genetic algorithm (self-mutation)
# ============================================================

class GeneticConfig(BaseModel):
    num_exec_workers: int
    num_sensor_workers: int
    num_opt_workers: int
    exploration_rate: float  # for RL
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
        # Very simple: mutate current based on reward
        if reward > 0.7:
            print("[GENETIC] High reward, small mutation")
            self.current.mutation_rate = max(0.01, self.current.mutation_rate * 0.9)
        else:
            print("[GENETIC] Low reward, larger mutation")
            self.current.mutation_rate = min(0.5, self.current.mutation_rate * 1.1)
        self.current = self.mutate(self.current)
        print(f"[GENETIC] Evolved config: {self.current.dict()}")
        return self.current

# ============================================================
# 8. Evolution: Reinforcement learning (simple Q-agent)
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

# ============================================================
# 9. Swarm-level consensus (simple majority)
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
                votes_for=1,  # self-vote
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

# ============================================================
# 10. Borg system (Queen + Workers, including GPU workers)
# ============================================================

class WorkerRole(str, Enum):
    EXECUTOR = "executor"
    SENSOR = "sensor"
    OPTIMIZER = "optimizer"
    GPU_OPTIMIZER = "gpu_optimizer"


class BorgTaskType(str, Enum):
    EXECUTION = "execution"
    SENSE = "sense"
    OPTIMIZE = "optimize"
    GPU_OPTIMIZE = "gpu_optimize"


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
                # Example dummy GPU op
                x = torch.randn(1000, 1000, device="cuda")
                y = torch.matmul(x, x)
                _ = y.sum().item()
            else:
                print(f"[BORG-WORKER] {self.worker_id} GPU requested but CUDA/torch not available")
        else:
            print(f"[BORG-WORKER] {self.worker_id} mismatched task {task.task_type.value}")


class BorgQueen:
    def __init__(self):
        self.executor_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.sensor_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.optimizer_queue: "queue.Queue[BorgTask]" = queue.Queue()
        self.gpu_optimizer_queue: "queue.Queue[BorgTask]" = queue.Queue()
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

# ============================================================
# 11. Queen-of-Queens (cluster super-brain)
# ============================================================

class SuperQueen:
    """
    Abstracts a Queen-of-Queens:
    - Could coordinate multiple BorgQueens across nodes via sync capsules.
    - Here, it manages local Queen + evolution + RL + consensus.
    """
    def __init__(self, queen: BorgQueen, genetic: GeneticEngine, rl: RLAgent, consensus: ConsensusManager):
        self.queen = queen
        self.genetic = genetic
        self.rl = rl
        self.consensus = consensus

    def decide_and_evolve(self, avg_threat: float, avg_latency: float, peer_count: int):
        # Reward: low threat + low latency is good
        reward = max(0.0, 1.0 - (avg_threat + avg_latency) / 2.0)
        cfg = self.genetic.evolve(reward)

        state = f"threat:{round(avg_threat,2)}|lat:{round(avg_latency,2)}"
        action = self.rl.choose_action(state)
        print(f"[SUPER-QUEEN] State={state}, Action={action.value}, Reward={reward:.3f}")

        next_state = state  # simple
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
        else:
            print("[SUPER-QUEEN] Consensus rejected new config")

# ============================================================
# 12. BorgManager (integrates everything)
# ============================================================

class BorgManager:
    def __init__(self, queen: BorgQueen, super_queen: SuperQueen):
        self.queen = queen
        self.super_queen = super_queen
        self._opt_thread = threading.Thread(target=self._optimizer_loop, daemon=True)
        self._running = False
        self._threat_history: List[float] = []
        self._latency_history: List[float] = []

    def start(self):
        self._running = True
        if not self.super_queen.genetic.current:
            self.super_queen.genetic.init_population()
        self.queen.spawn_workers(self.super_queen.genetic.current)
        self._opt_thread.start()

    def stop(self):
        self._running = False

    def handle_capsule_for_borg(self, capsule: DataCapsule, threat: ThreatScore, latency: float):
        self._threat_history.append(threat.score)
        self._latency_history.append(latency)
        if len(self._threat_history) > 100:
            self._threat_history.pop(0)
        if len(self._latency_history) > 100:
            self._latency_history.pop(0)

        if capsule.capsule_type in {CapsuleType.COMMAND, CapsuleType.CONTROL}:
            self.queen.submit_execution({
                "origin": capsule.origin_node_id,
                "command": capsule.payload,
                "meta": capsule.meta,
            })
        elif capsule.capsule_type in {CapsuleType.TELEMETRY, CapsuleType.EVENT, CapsuleType.ALERT}:
            self.queen.submit_sense({
                "origin": capsule.origin_node_id,
                "data": capsule.payload,
                "type": capsule.capsule_type.value,
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
            peer_count = len(SYNC_PEERS)
            self.super_queen.decide_and_evolve(avg_threat, avg_latency, peer_count)

# ============================================================
# 13. FastAPI app & global state
# ============================================================

app = FastAPI(title="Unified Evolving Borg Data Node", version="1.0.0")

NODE_IDENTITY: NodeIdentity
CURRENT_PERSONA: Persona
EVENT_BUS = EventBus()
EXTERNAL_PARTNERS: Dict[str, ExternalHandshakeRequest] = {}
SYNC_PEERS: List[str] = []

GENETIC_ENGINE = GeneticEngine()
RL_AGENT = RLAgent()
CONSENSUS_MANAGER = ConsensusManager()
BORG_QUEEN = BorgQueen()
SUPER_QUEEN = SuperQueen(BORG_QUEEN, GENETIC_ENGINE, RL_AGENT, CONSENSUS_MANAGER)
BORG_MANAGER = BorgManager(BORG_QUEEN, SUPER_QUEEN)

# ============================================================
# 14. Core behaviors
# ============================================================

def handle_capsule(capsule: DataCapsule):
    global CURRENT_PERSONA

    start = time.time()

    if capsule.integrity_hash:
        expected = compute_capsule_hash(capsule)
        if expected != capsule.integrity_hash:
            print(f"[{NODE_IDENTITY.node_id}] Integrity mismatch for capsule {capsule.capsule_id}")
            return

    threat = evaluate_threat(capsule, NODE_IDENTITY)
    new_persona = persona_transition(CURRENT_PERSONA, threat)
    if new_persona != CURRENT_PERSONA:
        print(f"[{NODE_IDENTITY.node_id}] Persona change: {CURRENT_PERSONA.value} -> {new_persona.value} (threat={threat.score})")
        CURRENT_PERSONA = new_persona

    if CURRENT_PERSONA == Persona.QUARANTINED:
        print(f"[{NODE_IDENTITY.node_id}] QUARANTINED: dropping capsule {capsule.capsule_id}")
        return

    print(f"[{NODE_IDENTITY.node_id}] [{CURRENT_PERSONA.value}] Received {capsule.capsule_type.value} from {capsule.origin_node_id}: {capsule.payload}")

    if capsule.capsule_type == CapsuleType.COMMAND and CURRENT_PERSONA in {Persona.OPERATOR, Persona.AUTONOMOUS}:
        print(f"[{NODE_IDENTITY.node_id}] Executing command (local): {capsule.payload}")
    elif capsule.capsule_type == CapsuleType.SYNC:
        print(f"[{NODE_IDENTITY.node_id}] Applying sync data: {capsule.payload}")

    latency = time.time() - start
    BORG_MANAGER.handle_capsule_for_borg(capsule, threat, latency)


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
# 15. API endpoints
# ============================================================

@app.get("/identity")
def get_identity():
    return {
        "identity": NODE_IDENTITY.dict(),
        "persona": CURRENT_PERSONA.value,
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
        "borg": {
            "modes": ["scheduler", "organism", "optimizer"],
            "workers": len(BORG_QUEEN.workers),
            "gpu_enabled": HAS_TORCH and HAS_CUDA,
        },
        "evolution": {
            "genetic_config": GENETIC_ENGINE.current.dict() if GENETIC_ENGINE.current else None,
            "rl_exploration_rate": RL_AGENT.exploration_rate,
        },
    }


@app.post("/ingest")
def ingest_capsule(capsule: DataCapsule):
    EVENT_BUS.publish("capsules", capsule)
    return {
        "node_id": NODE_IDENTITY.node_id,
        "received_capsule_id": capsule.capsule_id,
        "persona": CURRENT_PERSONA.value,
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

# ============================================================
# 16. Main / bootstrap
# ============================================================

def main():
    global NODE_IDENTITY, CURRENT_PERSONA

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

    EVENT_BUS.subscribe("capsules", handle_capsule)

    SYNC_PEERS.extend(args.sync_peer)
    if SYNC_PEERS:
        t_sync = threading.Thread(target=sync_with_peers, daemon=True)
        t_sync.start()

    BORG_MANAGER.start()

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
                meta={"test": True},
            )
        threading.Thread(target=delayed_send, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
