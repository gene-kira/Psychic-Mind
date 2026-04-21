#!/usr/bin/env python3
# unified_swarm_queen.py

import sys
import subprocess
import importlib
import json
import time
import math
import hashlib
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone

# ---------------- autoloader ----------------

REQUIRED_LIBS = ["PySide6"]

def ensure_deps():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

ensure_deps()

from PySide6 import QtWidgets, QtCore

# ---------------- time helpers ----------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def seconds_since(ts_iso: str) -> float:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return 0.0

# ---------------- JSON helpers ----------------

def read_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

# ---------------- math helpers ----------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def rolling_avg(old: float, new: float, weight: float = 0.1) -> float:
    if old == 0.0:
        return new
    return (1.0 - weight) * old + weight * new

def average(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

# ---------------- fingerprint identity ----------------

def _bucket_float(x: float, scale: float = 10.0) -> int:
    return int(round(clamp(x, 0.0, 1.0) * scale))

def _bucket_port(port: Optional[int]) -> int:
    return int(port) // 1000 if port is not None else -1

def build_canonical_fingerprint_features(features: Dict[str, Any], ftype: str) -> Dict[str, Any]:
    canonical = {
        "type": ftype,
        "time_bucket": features.get("time_bucket", "UNKNOWN"),
        "keyword_hits": int(features.get("keyword_hits", 0)),
        "cmd_entropy_bucket": _bucket_float(features.get("cmd_entropy", 0.0)),
        "ml_score_bucket": _bucket_float(features.get("ml_score", 0.0)),
        "heuristic_score_bucket": _bucket_float(features.get("heuristic_score", 0.0)),
    }

    if ftype == "process":
        canonical["process_name"] = features.get("process_name", "").lower()
        canonical["parent_name"] = features.get("parent_name", "").lower()
        canonical["root_ancestor"] = features.get("root_ancestor", "").lower()
        canonical["lineage_depth"] = int(features.get("lineage_depth", 0))
    elif ftype == "ip":
        canonical["ip"] = features.get("ip", "")
        canonical["direction"] = features.get("direction", "unknown")
        canonical["port_bucket"] = _bucket_port(features.get("port"))
    elif ftype == "sequence":
        canonical["parent_name"] = features.get("parent_name", "").lower()
        canonical["child_name"] = features.get("child_name", "").lower()
        canonical["avg_delay_bucket"] = _bucket_float(features.get("avg_delay_norm", 0.0))
    elif ftype == "burst":
        canonical["burst_rate_bucket"] = _bucket_float(features.get("burst_rate_norm", 0.0))

    return canonical

def compute_fingerprint_id(canonical: Dict[str, Any]) -> str:
    s = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return "fp-" + h[:12]

# ---------------- dataclasses ----------------

@dataclass
class Fingerprint:
    fingerprint_id: str
    type: str
    version: int
    created_at: str
    node_id: str
    local_incident_id: str
    features: Dict[str, Any]
    context: Dict[str, Any]
    explanation: List[str]
    local_confidence: float

@dataclass
class SupportingNodeEntry:
    node_id: str
    reports: int = 0
    avg_local_confidence: float = 0.0

@dataclass
class Cluster:
    cluster_id: str
    type: str
    version: int
    fingerprint_ids: List[str] = field(default_factory=list)
    aggregate_features: Dict[str, Any] = field(default_factory=dict)
    supporting_nodes: Dict[str, SupportingNodeEntry] = field(default_factory=dict)
    swarm_confidence: float = 0.0
    last_updated: str = field(default_factory=now_utc_iso)

@dataclass
class NodeStats:
    reports_sent: int = 0
    clusters_contributed: int = 0
    estimated_true_positives: int = 0
    estimated_false_positives: int = 0
    avg_local_confidence: float = 0.0
    avg_swarm_confidence_of_reports: float = 0.0

@dataclass
class NodeReputation:
    node_id: str
    first_seen: str
    last_seen: str
    stats: NodeStats = field(default_factory=NodeStats)
    reputation_score: float = 0.5
    flags: Dict[str, bool] = field(default_factory=lambda: {"noisy": False, "under_observation": False})

@dataclass
class SwarmState:
    clusters: Dict[str, Cluster] = field(default_factory=dict)
    reputations: Dict[str, NodeReputation] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "clusters": {cid: asdict(c) for cid, c in self.clusters.items()},
            "reputations": {nid: asdict(r) for nid, r in self.reputations.items()},
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "SwarmState":
        clusters: Dict[str, Cluster] = {}
        for cid, c in data.get("clusters", {}).items():
            sne = {
                nid: SupportingNodeEntry(**e)
                for nid, e in c.get("supporting_nodes", {}).items()
            }
            clusters[cid] = Cluster(
                cluster_id=c["cluster_id"],
                type=c["type"],
                version=c.get("version", 1),
                fingerprint_ids=c.get("fingerprint_ids", []),
                aggregate_features=c.get("aggregate_features", {}),
                supporting_nodes=sne,
                swarm_confidence=c.get("swarm_confidence", 0.0),
                last_updated=c.get("last_updated", now_utc_iso()),
            )

        reputations: Dict[str, NodeReputation] = {}
        for nid, r in data.get("reputations", {}).items():
            stats = r.get("stats", {})
            reputations[nid] = NodeReputation(
                node_id=r["node_id"],
                first_seen=r.get("first_seen", now_utc_iso()),
                last_seen=r.get("last_seen", now_utc_iso()),
                stats=NodeStats(
                    reports_sent=stats.get("reports_sent", 0),
                    clusters_contributed=stats.get("clusters_contributed", 0),
                    estimated_true_positives=stats.get("estimated_true_positives", 0),
                    estimated_false_positives=stats.get("estimated_false_positives", 0),
                    avg_local_confidence=stats.get("avg_local_confidence", 0.0),
                    avg_swarm_confidence_of_reports=stats.get("avg_swarm_confidence_of_reports", 0.0),
                ),
                reputation_score=r.get("reputation_score", 0.5),
                flags=r.get("flags", {"noisy": False, "under_observation": False}),
            )

        return SwarmState(clusters=clusters, reputations=reputations)

def load_swarm_state(path: Path) -> SwarmState:
    data = read_json(path, default={})
    return SwarmState.from_json(data)

def save_swarm_state(path: Path, state: SwarmState):
    write_json(path, state.to_json())

# ---------------- cluster aggregation ----------------

def init_aggregate_features_from_fp(fp: Fingerprint) -> Dict[str, Any]:
    f = fp.features
    return {
        "avg_rarity": f.get("rarity", 0.0),
        "avg_ml_score": f.get("ml_score", 0.0),
        "avg_heuristic_score": f.get("heuristic_score", 0.0),
        "avg_water_pressure": f.get("water_pressure", 0.0),
        "dominant_process_name": f.get("process_name"),
        "dominant_parent_name": f.get("parent_name"),
        "dominant_time_bucket": f.get("time_bucket"),
        "dominant_ip": f.get("ip"),
    }

def update_aggregate_features(agg: Dict[str, Any], fp: Fingerprint, count: int):
    f = fp.features

    def upd(key_agg, key_fp):
        old = agg.get(key_agg, 0.0)
        new = f.get(key_fp, 0.0)
        agg[key_agg] = rolling_avg(old, new)

    upd("avg_rarity", "rarity")
    upd("avg_ml_score", "ml_score")
    upd("avg_heuristic_score", "heuristic_score")
    upd("avg_water_pressure", "water_pressure")

    for k_agg, k_fp in [
        ("dominant_process_name", "process_name"),
        ("dominant_parent_name", "parent_name"),
        ("dominant_time_bucket", "time_bucket"),
        ("dominant_ip", "ip"),
    ]:
        v = f.get(k_fp)
        if v:
            agg[k_agg] = v

def init_cluster_from_fingerprint(fp: Fingerprint) -> Cluster:
    cid = "cluster-" + fp.fingerprint_id[3:9]
    agg = init_aggregate_features_from_fp(fp)
    node_entry = SupportingNodeEntry(node_id=fp.node_id, reports=1,
                                     avg_local_confidence=fp.local_confidence)
    return Cluster(
        cluster_id=cid,
        type=fp.type,
        version=1,
        fingerprint_ids=[fp.fingerprint_id],
        aggregate_features=agg,
        supporting_nodes={fp.node_id: node_entry},
        swarm_confidence=0.0,
        last_updated=fp.created_at,
    )

def fingerprints_match_cluster(fp: Fingerprint, cluster: Cluster) -> bool:
    if fp.type != cluster.type:
        return False

    f = fp.features
    agg = cluster.aggregate_features

    pairs = [
        (f.get("process_name"), agg.get("dominant_process_name")),
        (f.get("parent_name"), agg.get("dominant_parent_name")),
        (f.get("time_bucket"), agg.get("dominant_time_bucket")),
        (f.get("ip"), agg.get("dominant_ip")),
    ]

    matches = 0
    for v_fp, v_agg in pairs:
        if v_fp and v_agg and str(v_fp).lower() == str(v_agg).lower():
            matches += 1

    return matches >= 1

def update_cluster_with_fingerprint(cluster: Cluster, fp: Fingerprint):
    cluster.fingerprint_ids.append(fp.fingerprint_id)
    count = len(cluster.fingerprint_ids)
    update_aggregate_features(cluster.aggregate_features, fp, count)
    entry = cluster.supporting_nodes.get(fp.node_id)
    if entry is None:
        entry = SupportingNodeEntry(node_id=fp.node_id, reports=0, avg_local_confidence=0.0)
    entry.reports += 1
    entry.avg_local_confidence = rolling_avg(entry.avg_local_confidence, fp.local_confidence)
    cluster.supporting_nodes[fp.node_id] = entry
    cluster.last_updated = max(cluster.last_updated, fp.created_at)

def merge_fingerprint_into_clusters(state: SwarmState, fp: Fingerprint) -> Cluster:
    for cluster in state.clusters.values():
        if fingerprints_match_cluster(fp, cluster):
            update_cluster_with_fingerprint(cluster, fp)
            return cluster
    cluster = init_cluster_from_fingerprint(fp)
    state.clusters[cluster.cluster_id] = cluster
    return cluster

# ---------------- reputation & swarm confidence ----------------

HIGH_CONF_THRESHOLD = 0.8
LOW_CONF_THRESHOLD = 0.3

def init_reputation(node_id: str) -> NodeReputation:
    now = now_utc_iso()
    return NodeReputation(node_id=node_id, first_seen=now, last_seen=now)

def compute_reputation_score(stats: NodeStats) -> float:
    tp = stats.estimated_true_positives
    fp = stats.estimated_false_positives
    total = max(1, tp + fp)
    precision = tp / total
    volume_factor = min(1.0, stats.reports_sent / 100.0)
    base = 0.5 * precision + 0.5 * volume_factor
    return clamp(base, 0.0, 1.0)

def get_or_init_reputation(state: SwarmState, node_id: str) -> NodeReputation:
    rep = state.reputations.get(node_id)
    if rep is None:
        rep = init_reputation(node_id)
        state.reputations[node_id] = rep
    rep.last_seen = now_utc_iso()
    return rep

def compute_node_anomaly_for_cluster(cluster: Cluster, node_id: str) -> float:
    entry = cluster.supporting_nodes.get(node_id)
    if not entry:
        return 1.0
    all_conf = [e.avg_local_confidence for e in cluster.supporting_nodes.values()]
    if not all_conf:
        return 0.0
    swarm_avg = average(all_conf)
    diff = abs(entry.avg_local_confidence - swarm_avg)
    return clamp(diff, 0.0, 1.0)

def compute_swarm_confidence(cluster: Cluster, reputations: Dict[str, NodeReputation]) -> float:
    if not cluster.supporting_nodes:
        return 0.0

    weights: List[float] = []
    scores: List[float] = []

    for node_id, entry in cluster.supporting_nodes.items():
        rep = reputations.get(node_id)
        rep_score = rep.reputation_score if rep else 0.5
        w = max(0.05, rep_score) * max(1, entry.reports)
        weights.append(w)
        scores.append(entry.avg_local_confidence)

    total_w = sum(weights)
    if total_w <= 0:
        return 0.0

    weighted = sum(s * w for s, w in zip(scores, weights)) / total_w
    node_factor = clamp(len(cluster.supporting_nodes) / 5.0, 0.0, 1.0)
    return clamp(0.7 * weighted + 0.3 * node_factor, 0.0, 1.0)

def classify_cluster_severity(cluster: Cluster) -> str:
    sc = cluster.swarm_confidence
    if sc >= 0.9:
        return "critical"
    if sc >= 0.7:
        return "high"
    if sc >= 0.4:
        return "medium"
    if sc > 0.0:
        return "low"
    return "unknown"

def update_reputation_with_report(rep: NodeReputation, fp: Fingerprint, cluster: Cluster):
    s = rep.stats
    s.reports_sent += 1
    s.clusters_contributed += 1
    s.avg_local_confidence = rolling_avg(s.avg_local_confidence, fp.local_confidence)
    s.avg_swarm_confidence_of_reports = rolling_avg(
        s.avg_swarm_confidence_of_reports,
        cluster.swarm_confidence,
    )

    if fp.local_confidence >= HIGH_CONF_THRESHOLD and cluster.swarm_confidence >= HIGH_CONF_THRESHOLD:
        s.estimated_true_positives += 1
    elif fp.local_confidence >= HIGH_CONF_THRESHOLD and cluster.swarm_confidence <= LOW_CONF_THRESHOLD:
        s.estimated_false_positives += 1

    rep.reputation_score = compute_reputation_score(s)
    rep.flags["noisy"] = s.estimated_false_positives > 10 and rep.reputation_score < 0.4
    rep.flags["under_observation"] = rep.flags["noisy"] or rep.reputation_score < 0.3

# ---------------- distributed trust decay ----------------

def apply_trust_decay(state: SwarmState, half_life_seconds: float = 86400.0):
    now = datetime.now(timezone.utc)
    for rep in state.reputations.values():
        last = datetime.fromisoformat(rep.last_seen.replace("Z", "+00:00"))
        dt = (now - last).total_seconds()
        if dt <= 0:
            continue
        decay_factor = 0.5 ** (dt / half_life_seconds)
        rep.reputation_score = clamp(0.5 + (rep.reputation_score - 0.5) * decay_factor, 0.0, 1.0)

# ---------------- high-level ingest API ----------------

def ingest_fingerprint(state: SwarmState, fp: Fingerprint) -> Cluster:
    rep = get_or_init_reputation(state, fp.node_id)
    cluster = merge_fingerprint_into_clusters(state, fp)
    cluster.swarm_confidence = compute_swarm_confidence(cluster, state.reputations)
    update_reputation_with_report(rep, fp, cluster)
    return cluster

def derive_orders_for_cluster(cluster: Cluster) -> Dict[str, Any]:
    severity = classify_cluster_severity(cluster)
    sc = cluster.swarm_confidence

    actions: List[str] = []
    if severity == "critical":
        actions = ["isolate_entity", "block_ip", "kill_process", "escalate_human"]
    elif severity == "high":
        actions = ["block_ip", "kill_process", "escalate_human"]
    elif severity == "medium":
        actions = ["increase_logging", "temporary_quarantine", "escalate_if_persistent"]
    elif severity == "low":
        actions = ["monitor", "tag_suspicious"]

    return {
        "cluster_id": cluster.cluster_id,
        "severity": severity,
        "swarm_confidence": sc,
        "aggregate_features": cluster.aggregate_features,
        "actions": actions,
        "supporting_nodes": list(cluster.supporting_nodes.keys()),
        "last_updated": cluster.last_updated,
    }

# ---------------- attack chain engine ----------------

class AttackChainEngine:
    def __init__(self, window: int = 120):
        self.window = window
        self.events: deque[Tuple[float, str, Dict[str, Any]]] = deque()

    def add_event(self, etype: str, data: Dict[str, Any]):
        now = time.time()
        self.events.append((now, etype, data))
        self._cleanup(now)

    def _cleanup(self, now: float):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def detect(self) -> List[Tuple[str, float]]:
        types = [e[1] for e in self.events]
        chains: List[Tuple[str, float]] = []

        if all(x in types for x in ["proc_spawn", "powershell", "net_connect"]):
            chains.append(("LOLBIN_ATTACK", 0.9))

        if types.count("proc_spawn") > 8 and "net_connect" in types:
            chains.append(("PROCESS_STORM", 0.8))

        if "file_mod" in types and "net_connect" in types:
            chains.append(("PERSISTENCE_EXFIL", 0.85))

        return chains

# ---------------- Queen consensus + narrative ----------------

@dataclass
class AttackNarrative:
    narrative_id: str
    entities: List[str]
    clusters: List[str]
    chains: List[str]
    severity: str
    global_score: float
    created_at: str
    last_updated: str

class QueenConsensus:
    def __init__(self, window: int = 180):
        self.window = window
        self.node_chains: Dict[str, List[Tuple[str, float]]] = {}
        self.global_events: deque[Tuple[float, str, Dict[str, Any]]] = deque()
        self.narratives: Dict[str, AttackNarrative] = {}

    def update_from_cluster_orders(self, orders: List[Dict[str, Any]]):
        now = time.time()
        for order in orders:
            ent = order["aggregate_features"].get("dominant_process_name") or \
                  order["aggregate_features"].get("dominant_ip") or "unknown"
            evt = {
                "entity": ent,
                "cluster_id": order["cluster_id"],
                "severity": order["severity"],
                "swarm_confidence": order["swarm_confidence"],
            }
            self.global_events.append((now, "cluster_order", evt))
        self._cleanup(now)

    def update_from_attack_chains(self, node_id: str, chains: List[Tuple[str, float]]):
        self.node_chains[node_id] = chains

    def _cleanup(self, now: float):
        cutoff = now - self.window
        while self.global_events and self.global_events[0][0] < cutoff:
            self.global_events.popleft()

    def fuse_swarm_and_chains(self) -> Dict[str, float]:
        chain_votes: Dict[str, float] = {}
        for node, chains in self.node_chains.items():
            for cname, score in chains:
                chain_votes[cname] = chain_votes.get(cname, 0.0) + score
        return {c: s for c, s in chain_votes.items() if s > 0.8}

    def reconstruct_narratives(self):
        by_entity: Dict[str, Dict[str, Any]] = {}
        for ts, etype, evt in self.global_events:
            if etype != "cluster_order":
                continue
            ent = evt["entity"]
            bucket = by_entity.setdefault(ent, {
                "clusters": set(),
                "chains": set(),
                "max_severity": "unknown",
                "max_score": 0.0,
            })
            bucket["clusters"].add(evt["cluster_id"])
            sev = evt["severity"]
            sev_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
            if sev_rank.get(sev, 0) > sev_rank.get(bucket["max_severity"], 0):
                bucket["max_severity"] = sev
            bucket["max_score"] = max(bucket["max_score"], evt["swarm_confidence"])

        fused_chains = self.fuse_swarm_and_chains()
        for ent, data in by_entity.items():
            ent_chains = [c for c in fused_chains.keys() if ent in c or True]  # simple association
            nid = "nar-" + hashlib.sha1(ent.encode("utf-8")).hexdigest()[:10]
            nar = AttackNarrative(
                narrative_id=nid,
                entities=[ent],
                clusters=list(data["clusters"]),
                chains=ent_chains,
                severity=data["max_severity"],
                global_score=data["max_score"],
                created_at=now_utc_iso(),
                last_updated=now_utc_iso(),
            )
            self.narratives[nid] = nar

    def get_active_narratives(self) -> List[AttackNarrative]:
        return list(self.narratives.values())

# ---------------- PySide6 cockpit for swarm orders ----------------

class SwarmCockpit(QtWidgets.QMainWindow):
    def __init__(self, state: SwarmState, queen: QueenConsensus):
        super().__init__()
        self.state = state
        self.queen = queen
        self.setWindowTitle("Swarm / Queen Cockpit")

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Cluster ID", "Severity", "Swarm Conf", "Entity", "Actions", "Supporting Nodes"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_view)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.refresh_btn)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.resize(1000, 600)
        self.refresh_view()

    def refresh_view(self):
        orders: List[Dict[str, Any]] = []
        for cluster in self.state.clusters.values():
            orders.append(derive_orders_for_cluster(cluster))

        self.queen.update_from_cluster_orders(orders)
        self.queen.reconstruct_narratives()

        self.table.setRowCount(len(orders))
        for row, order in enumerate(orders):
            agg = order["aggregate_features"]
            ent = agg.get("dominant_process_name") or agg.get("dominant_ip") or "unknown"
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(order["cluster_id"]))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(order["severity"]))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{order['swarm_confidence']:.2f}"))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(ent))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(", ".join(order["actions"])))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(", ".join(order["supporting_nodes"])))

# ---------------- example main ----------------

def main():
    state = SwarmState()
    queen = QueenConsensus()

    # demo: fake fingerprint
    features = {
        "process_name": "evil.exe",
        "parent_name": "powershell.exe",
        "time_bucket": "night",
        "ip": "10.0.0.13",
        "rarity": 0.9,
        "ml_score": 0.95,
        "heuristic_score": 0.9,
        "water_pressure": 0.8,
    }
    canonical = build_canonical_fingerprint_features(features, "process")
    fp_id = compute_fingerprint_id(canonical)
    fp = Fingerprint(
        fingerprint_id=fp_id,
        type="process",
        version=1,
        created_at=now_utc_iso(),
        node_id="node-1",
        local_incident_id="inc-1",
        features=features,
        context={},
        explanation=["demo"],
        local_confidence=0.92,
    )
    ingest_fingerprint(state, fp)

    app = QtWidgets.QApplication(sys.argv)
    cockpit = SwarmCockpit(state, queen)
    cockpit.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
