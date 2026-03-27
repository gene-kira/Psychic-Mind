from __future__ import annotations

import importlib
import os
import platform
import subprocess
import time
import json
import sqlite3
import hashlib
import base64
import random
import threading
import queue
import urllib.request
import http.client
import ctypes
import sys
import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any, List, Tuple

# Optional libs (loaded lazily where needed)
try:
    import psutil  # used by NodeServer telemetry
except Exception:
    psutil = None

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    Observer = None
    FileSystemEventHandler = object

MEMORY_FILE = "borg_memory.json"

# Simple defaults for Borg mesh config and related helpers
BORG_MESH_CONFIG = {
    "max_corridors": 1024,
    "unknown_bias": 0.3,
}


class MemoryManager:
    def record_mesh_event(self, evt: Dict[str, Any]):
        print(f"[MeshMemory] {evt}")


class BorgCommsRouter:
    def send_secure(self, channel: str, message: str, profile: str):
        print(f"[MeshComms] {channel} [{profile}]: {message}")


class SecurityGuardian:
    def disassemble(self, snippet: str) -> Dict[str, Any]:
        return {"entropy": 0.0, "pattern_flags": []}

    def reassemble(self, url: str, snippet: str, raw_pii_hits: int = 0) -> Dict[str, Any]:
        return {"status": "SAFE_FOR_TRAVEL"}

    def _pii_count(self, snippet: str) -> int:
        return 0


def privacy_filter(text: str):
    return text, {}


# === AUTO-ELEVATION CHECK ===
def ensure_admin():
    try:
        if platform.system().lower() == "windows":
            if not ctypes.windll.shell32.IsUserAnAdmin():
                script = os.path.abspath(sys.argv[0])
                params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable, f'"{script}" {params}', None, 1
                )
                sys.exit()
    except Exception as e:
        print(f"[Codex Sentinel] Elevation failed: {e}")
        sys.exit()


ensure_admin()

# =========================================================
# CONFIG
# =========================================================

DB_DIR = "sidestep_dbs"
os.makedirs(DB_DIR, exist_ok=True)

SWARM_DB = os.path.join(DB_DIR, "swarm.db")

default_nodes = {
    "A": ("127.0.0.1", 9101),
    "B": ("127.0.0.1", 9102),
    "C": ("127.0.0.1", 9103),
    "D": ("127.0.0.1", 9104),
    "E": ("127.0.0.1", 9105),
    "F": ("127.0.0.1", 9106),
    "G": ("127.0.0.1", 9107),
}

env_cfg = os.getenv("CLUSTER_CONFIG")
if env_cfg:
    try:
        cfg = json.loads(env_cfg)
        NODES = {n["id"]: (n["host"], int(n["port"])) for n in cfg}
    except Exception:
        NODES = default_nodes
else:
    NODES = default_nodes

DEFAULT_ORGANS = [
    "telemetry",
    "filesystem",
    "browser",
    "game",
    "external_api",
    "uiautomation",
    "heartbeat",
    "ml_inference",
    "gpu_inference",
    "guardian",
    "planner",
    "executor",
    "analyst",
    "nn_inference",
]
DEFAULT_LANES = [
    "system",
    "io",
    "activity",
    "telemetry",
    "signal",
    "focus",
    "control",
    "analysis",
    "guardian",
    "tasks",
]
DEFAULT_RITUALS = [
    "sample",
    "fs_event",
    "url_visit",
    "tick",
    "ping",
    "window_change",
    "alert",
    "classify",
    "gpu_score",
    "task_create",
    "task_assign",
    "task_result",
    "guardian_check",
    "nn_score",
]

WATCH_DIR = os.path.abspath(".")
BASE_ENCRYPTION_KEY = b"mythic_shared_key"

TRUSTED_NODE_IDS = set(NODES.keys())


def node_secret(node_id: str) -> bytes:
    return hashlib.sha256(BASE_ENCRYPTION_KEY + node_id.encode("utf-8")).digest()


# =========================================================
# UTILS / ENCRYPTION / SIGNING
# =========================================================

def now() -> float:
    return time.time()


def xor_encrypt(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def encrypt_payload_for(node_id: str, obj: Any) -> str:
    raw = json.dumps(obj).encode("utf-8")
    key = node_secret(node_id)
    enc = xor_encrypt(raw, key)
    return base64.b64encode(enc).decode("utf-8")


def decrypt_payload_from(node_id: str, s: str) -> Dict[str, Any]:
    try:
        key = node_secret(node_id)
        enc = base64.b64decode(s.encode("utf-8"))
        raw = xor_encrypt(enc, key)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def sign_message(sender_id: str, payload_str: str) -> str:
    key = node_secret(sender_id)
    h = hashlib.sha256(key + payload_str.encode("utf-8")).digest()
    return base64.b64encode(h).decode("utf-8")


def verify_message(sender_id: str, payload_str: str, signature: str) -> bool:
    if sender_id not in TRUSTED_NODE_IDS:
        return False
    expected = sign_message(sender_id, payload_str)
    return expected == signature


def db_path(node_id: str) -> str:
    return os.path.join(DB_DIR, f"sidestep_node_{node_id}.db")


def http_post(host: str, port: int, path: str, obj: Dict[str, Any], timeout: float = 1.0):
    data = json.dumps(obj).encode("utf-8")
    url = f"http://{host}:{port}{path}"
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


# =========================================================
# SWARM-WIDE DB
# =========================================================

def init_swarm_db():
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS swarm_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT,
            direction TEXT,
            peer TEXT,
            payload TEXT,
            priority TEXT,
            organ TEXT,
            lane TEXT,
            ritual TEXT,
            intent TEXT,
            role TEXT,
            task_id TEXT,
            ts REAL
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS automation_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            condition_json TEXT,
            action_json TEXT,
            enabled INTEGER
        )
    """
    )
    conn.commit()
    conn.close()


def log_swarm(node_id: str, direction: str, peer: str, msg: Dict[str, Any]):
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO swarm_messages(node_id, direction, peer, payload, priority,
                                   organ, lane, ritual, intent, role, task_id, ts)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """,
        (
            node_id,
            direction,
            peer,
            msg.get("payload", ""),
            msg.get("priority", "normal"),
            msg.get("organ", ""),
            msg.get("lane", ""),
            msg.get("ritual", ""),
            msg.get("intent", ""),
            msg.get("role", ""),
            msg.get("task_id", ""),
            now(),
        ),
    )
    conn.commit()
    conn.close()


def query_swarm(
    limit: int = 100, priority: Optional[str] = None, tag: Optional[str] = None
) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    base = (
        "SELECT node_id, direction, peer, payload, priority, organ, lane, ritual, "
        "intent, role, task_id, ts FROM swarm_messages"
    )
    params: List[Any] = []
    where: List[str] = []
    if priority:
        where.append("priority = ?")
        params.append(priority)
    if tag:
        where.append(
            "(organ = ? OR lane = ? OR ritual = ? OR intent = ? OR role = ?)"
        )
        params.extend([tag, tag, tag, tag, tag])
    if where:
        base += " WHERE " + " AND ".join(where)
    base += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    c.execute(base, params)
    rows = c.fetchall()
    conn.close()
    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "node": r[0],
                "direction": r[1],
                "peer": r[2],
                "payload": r[3],
                "priority": r[4],
                "organ": r[5],
                "lane": r[6],
                "ritual": r[7],
                "intent": r[8],
                "role": r[9],
                "task_id": r[10],
                "ts": r[11],
            }
        )
    return results


def add_automation_rule(
    name: str, condition: Dict[str, Any], action: Dict[str, Any], enabled: bool = True
):
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO automation_rules(name, condition_json, action_json, enabled)
        VALUES (?,?,?,?)
    """,
        (name, json.dumps(condition), json.dumps(action), 1 if enabled else 0),
    )
    conn.commit()
    conn.close()


def load_automation_rules() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    c.execute(
        "SELECT id, name, condition_json, action_json, enabled FROM automation_rules WHERE enabled=1"
    )
    rows = c.fetchall()
    conn.close()
    rules: List[Dict[str, Any]] = []
    for rid, name, cond, act, en in rows:
        rules.append(
            {
                "id": rid,
                "name": name,
                "condition": json.loads(cond),
                "action": json.loads(act),
                "enabled": bool(en),
            }
        )
    return rules


# =========================================================
# NODE DB + LEARNING + RL
# =========================================================

def init_db(node_id: str):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            direction TEXT,
            peer TEXT,
            payload TEXT,
            priority TEXT,
            organ TEXT,
            lane TEXT,
            ritual TEXT,
            intent TEXT,
            role TEXT,
            task_id TEXT,
            ts REAL
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS neighbors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            neighbor_id TEXT,
            host TEXT,
            port INTEGER,
            healthy INTEGER DEFAULT 1,
            last_fail REAL DEFAULT 0
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS constraints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organ TEXT,
            lane TEXT,
            ritual TEXT,
            priority TEXT
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS q_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            neighbor_id TEXT,
            state TEXT,
            action TEXT,
            q REAL
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_role TEXT,
            intent TEXT,
            success INTEGER,
            fail INTEGER
        )
    """
    )
    conn.commit()
    conn.close()


def log_message(node_id: str, direction: str, peer: str, msg: Dict[str, Any]):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO messages(direction, peer, payload, priority, organ, lane,
                             ritual, intent, role, task_id, ts)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """,
        (
            direction,
            peer,
            msg.get("payload", ""),
            msg.get("priority", "normal"),
            msg.get("organ", ""),
            msg.get("lane", ""),
            msg.get("ritual", ""),
            msg.get("intent", ""),
            msg.get("role", ""),
            msg.get("task_id", ""),
            now(),
        ),
    )
    conn.commit()
    conn.close()
    log_swarm(node_id, direction, peer, msg)


def add_neighbor(node_id: str, neighbor_id: str, host: str, port: int):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute(
        "INSERT INTO neighbors(neighbor_id, host, port, healthy, last_fail) VALUES (?,?,?,?,?)",
        (neighbor_id, host, port, 1, 0.0),
    )
    conn.commit()
    conn.close()


def mark_neighbor_health(node_id: str, neighbor_id: str, healthy: bool):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    if healthy:
        c.execute("UPDATE neighbors SET healthy=1 WHERE neighbor_id=?", (neighbor_id,))
    else:
        c.execute(
            "UPDATE neighbors SET healthy=0, last_fail=? WHERE neighbor_id=?",
            (now(), neighbor_id),
        )
    conn.commit()
    conn.close()


def get_neighbors(node_id: str, include_unhealthy: bool = False) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    if include_unhealthy:
        c.execute("SELECT neighbor_id, host, port, healthy, last_fail FROM neighbors")
    else:
        c.execute(
            "SELECT neighbor_id, host, port, healthy, last_fail FROM neighbors WHERE healthy=1"
        )
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "host": r[1],
            "port": r[2],
            "healthy": r[3],
            "last_fail": r[4],
        }
        for r in rows
    ]


def get_state_summary(node_id: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE direction='in'")
    in_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM messages WHERE direction='out'")
    out_count = c.fetchone()[0]
    c.execute(
        "SELECT payload, priority, organ, lane, ritual, intent, role, task_id, ts "
        "FROM messages ORDER BY id DESC LIMIT 5"
    )
    last_msgs = [
        {
            "payload": r[0],
            "priority": r[1],
            "organ": r[2],
            "lane": r[3],
            "ritual": r[4],
            "intent": r[5],
            "role": r[6],
            "task_id": r[7],
            "ts": r[8],
        }
        for r in c.fetchall()
    ]
    conn.close()
    neighbors = get_neighbors(node_id, include_unhealthy=True)
    return {
        "node_id": node_id,
        "in_count": in_count,
        "out_count": out_count,
        "last_messages": last_msgs,
        "neighbors": neighbors,
    }


def load_constraints(node_id: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT organ, lane, ritual, priority FROM constraints")
    rows = c.fetchall()
    conn.close()
    rules: List[Dict[str, Any]] = []
    for organ, lane, ritual, priority in rows:
        rules.append(
            {
                "organ": organ or None,
                "lane": lane or None,
                "ritual": ritual or None,
                "priority": priority,
            }
        )
    return rules


def add_constraint(
    node_id: str,
    organ: Optional[str],
    lane: Optional[str],
    ritual: Optional[str],
    priority: str,
):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute(
        "INSERT INTO constraints(organ, lane, ritual, priority) VALUES (?,?,?,?)",
        (organ, lane, ritual, priority),
    )
    conn.commit()
    conn.close()


def apply_constraints(node_id: str, msg: Dict[str, Any]) -> Dict[str, Any]:
    rules = load_constraints(node_id)
    for r in rules:
        if r["organ"] and r["organ"] != msg.get("organ"):
            continue
        if r["lane"] and r["lane"] != msg.get("lane"):
            continue
        if r["ritual"] and r["ritual"] != msg.get("ritual"):
            continue
        msg["priority"] = r["priority"]
        return msg
    return msg


def learn_priority_from_history(node_id: str, msg: Dict[str, Any]) -> Dict[str, Any]:
    if msg.get("priority") not in (None, "", "auto"):
        return msg
    organ = msg.get("organ", "")
    lane = msg.get("lane", "")
    ritual = msg.get("ritual", "")
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute(
        """
        SELECT priority, COUNT(*) as cnt
        FROM messages
        WHERE organ = ? OR lane = ? OR ritual = ?
        GROUP BY priority
    """,
        (organ, lane, ritual),
    )
    rows = c.fetchall()
    conn.close()
    if not rows:
        msg["priority"] = "normal"
        return msg
    best = max(rows, key=lambda r: r[1])[0]
    msg["priority"] = best
    return msg


def detect_missing_details(msg: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not msg.get("payload"):
        issues.append("missing_payload")
    if not msg.get("organ"):
        issues.append("missing_organ")
    if not msg.get("lane"):
        issues.append("missing_lane")
    if not msg.get("ritual"):
        issues.append("missing_ritual")
    if msg.get("priority") in (None, "", "auto"):
        issues.append("missing_priority")
    return issues


def get_q_value(
    node_id: str, neighbor_id: str, state: str = "default", action: str = "send"
) -> float:
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute(
        "SELECT q FROM q_values WHERE neighbor_id=? AND state=? AND action=?",
        (neighbor_id, state, action),
    )
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    return 0.0


def set_q_value(
    node_id: str, neighbor_id: str, state: str, action: str, q: float
):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute(
        "SELECT id FROM q_values WHERE neighbor_id=? AND state=? AND action=?",
        (neighbor_id, state, action),
    )
    row = c.fetchone()
    if row:
        c.execute("UPDATE q_values SET q=? WHERE id=?", (q, row[0]))
    else:
        c.execute(
            "INSERT INTO q_values(neighbor_id, state, action, q) VALUES (?,?,?,?)",
            (neighbor_id, state, action, q),
        )
    conn.commit()
    conn.close()


def update_q(
    node_id: str,
    neighbor_id: str,
    reward: float,
    state: str = "default",
    action: str = "send",
    alpha: float = 0.3,
    gamma: float = 0.9,
):
    old_q = get_q_value(node_id, neighbor_id, state, action)
    new_q = old_q + alpha * (reward + gamma * 0 - old_q)
    set_q_value(node_id, neighbor_id, state, action, new_q)


def choose_neighbor_rl(node_id: str) -> Optional[Dict[str, Any]]:
    neighbors = get_neighbors(node_id)
    if not neighbors:
        return None
    if random.random() < 0.2:
        return random.choice(neighbors)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for n in neighbors:
        q = get_q_value(node_id, n["id"])
        scored.append((q, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def update_agent_stats(node_id: str, role: str, intent: str, success: bool = True):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute(
        "SELECT id, success, fail FROM agent_stats WHERE agent_role=? AND intent=?",
        (role, intent),
    )
    row = c.fetchone()
    if row:
        sid, s, f = row
        if success:
            s += 1
        else:
            f += 1
        c.execute("UPDATE agent_stats SET success=?, fail=? WHERE id=?", (s, f, sid))
    else:
        c.execute(
            "INSERT INTO agent_stats(agent_role, intent, success, fail) VALUES (?,?,?,?)",
            (role, intent, 1 if success else 0, 0 if success else 1),
        )
    conn.commit()
    conn.close()


# =========================================================
# SIDESTEP ENGINE
# =========================================================

class SidestepEngine:
    def __init__(self):
        self.main_lane = queue.Queue()
        self.sidestep_buffer = queue.Queue()
        self.jump_lane = queue.Queue()

    def receive(self, msg):
        prio = msg.get("priority", "normal")
        if prio == "jump":
            self.jump_lane.put(msg)
        elif prio == "sidestep":
            self.sidestep_buffer.put(msg)
        else:
            self.main_lane.put(msg)

    def process_one(self):
        if not self.jump_lane.empty():
            return "jump", self.jump_lane.get()
        if not self.main_lane.empty():
            return "normal", self.main_lane.get()
        if not self.sidestep_buffer.empty():
            return "sidestep", self.sidestep_buffer.get()
        return None, None


# =========================================================
# MULTI-AGENT LAYER (PLANNER / GUARDIAN / EXECUTOR / ANALYST)
# =========================================================

def guardian_check(msg):
    payload = msg.get("payload", "").lower()
    organ = msg.get("organ", "")
    ritual = msg.get("ritual", "")
    alerts = []

    if organ == "filesystem" and ("delete" in payload or "remove" in payload):
        alerts.append("filesystem_delete")

    if organ == "telemetry" and "cpu=" in payload:
        try:
            parts = payload.split()
            cpu_part = [p for p in parts if p.startswith("cpu=")][0]
            cpu_val = float(cpu_part.split("=")[1])
            if cpu_val > 85.0:
                alerts.append("high_cpu")
        except Exception:
            pass

    if "error" in payload or "fail" in payload:
        alerts.append("error_signal")

    return alerts


def apply_guardian_policies(node, msg):
    alerts = guardian_check(msg)
    if not alerts:
        return msg, None

    msg["priority"] = "jump"
    alert_msg = {
        "from": node.node_id,
        "payload": f"GUARDIAN ALERT {alerts} on {msg.get('organ')}/{msg.get('ritual')}",
        "priority": "jump",
        "organ": "guardian",
        "lane": "guardian",
        "ritual": "alert",
        "intent": "guardian_alert",
        "role": "guardian",
    }
    return msg, alert_msg


def planner_agent(node, msg):
    if msg.get("intent") != "create_task":
        return None
    task_id = f"{node.node_id}-{int(now() * 1000)}"
    payload = msg.get("payload", "")
    new_msg = {
        "from": node.node_id,
        "payload": payload,
        "priority": "sidestep",
        "organ": "planner",
        "lane": "tasks",
        "ritual": "task_assign",
        "intent": "execute_task",
        "role": "executor",
        "task_id": task_id,
    }
    return new_msg


def executor_agent(node, msg):
    if msg.get("intent") != "execute_task":
        return None
    payload = msg.get("payload", "")
    result = f"executed:{payload}"
    result_msg = {
        "from": node.node_id,
        "payload": result,
        "priority": "normal",
        "organ": "executor",
        "lane": "tasks",
        "ritual": "task_result",
        "intent": "task_result",
        "role": "analyst",
        "task_id": msg.get("task_id", ""),
    }
    return result_msg


def analyst_agent(node, msg):
    if msg.get("intent") != "task_result":
        return None
    payload = msg.get("payload", "")
    summary = f"analysis:{payload}"
    summary_msg = {
        "from": node.node_id,
        "payload": summary,
        "priority": "normal",
        "organ": "analyst",
        "lane": "analysis",
        "ritual": "classify",
        "intent": "analysis_summary",
        "role": "guardian",
        "task_id": msg.get("task_id", ""),
    }
    return summary_msg


def run_agents(node, msg):
    out = []
    pm = planner_agent(node, msg)
    if pm:
        out.append(pm)
        update_agent_stats(node.node_id, "planner", msg.get("intent", ""), True)

    em = executor_agent(node, msg)
    if em:
        out.append(em)
        update_agent_stats(node.node_id, "executor", msg.get("intent", ""), True)

    am = analyst_agent(node, msg)
    if am:
        out.append(am)
        update_agent_stats(node.node_id, "analyst", msg.get("intent", ""), True)

    return out


def evaluate_automation_rules(node, msg):
    rules = load_automation_rules()
    triggered = []

    for r in rules:
        cond = r["condition"]
        ok = True
        if "organ" in cond and cond["organ"] != msg.get("organ"):
            ok = False
        if "lane" in cond and cond["lane"] != msg.get("lane"):
            ok = False
        if "ritual" in cond and cond["ritual"] != msg.get("ritual"):
            ok = False
        if "contains" in cond and cond["contains"].lower() not in msg.get(
            "payload", ""
        ).lower():
            ok = False
        if not ok:
            continue
        triggered.append(r)

    actions = []
    for r in triggered:
        act = r["action"]
        target_node = act.get("target_node")
        intent = act.get("intent")
        payload = act.get("payload", "")
        if target_node and intent:
            actions.append(
                {
                    "from": node.node_id,
                    "payload": payload,
                    "priority": act.get("priority", "normal"),
                    "organ": act.get("organ", "planner"),
                    "lane": act.get("lane", "tasks"),
                    "ritual": act.get("ritual", "task_create"),
                    "intent": intent,
                    "role": act.get("role", "planner"),
                    "target_node": target_node,
                }
            )
    return actions


# =========================================================
# AUTOLOADER
# =========================================================

class AutoLoader:
    def __init__(self):
        self._cache: Dict[str, Optional[Any]] = {}

    def load(self, module_name: str):
        if module_name in self._cache:
            return self._cache[module_name]
        try:
            mod = importlib.import_module(module_name)
            self._cache[module_name] = mod
            print(f"[AutoLoader] Loaded: {module_name}")
            return mod
        except ImportError:
            self._cache[module_name] = None
            print(f"[AutoLoader] Missing: {module_name}")
            return None

    def available(self, module_name: str) -> bool:
        return self.load(module_name) is not None


# =========================================================
# ELEVATION
# =========================================================

class Elevation:
    @staticmethod
    def is_elevated() -> bool:
        system = platform.system().lower()
        if system == "windows":
            try:
                import ctypes as _ct

                return bool(_ct.windll.shell32.IsUserAnAdmin())
            except Exception:
                return False
        try:
            return os.geteuid() == 0
        except AttributeError:
            return False

    @staticmethod
    def describe() -> str:
        return "Admin/Root" if Elevation.is_elevated() else "Normal User"


# =========================================================
# GPU DETECTION
# =========================================================

def detect_gpu(loader: AutoLoader) -> bool:
    for mod_name, attr in [("torch", "cuda"), ("cupy", None)]:
        mod = loader.load(mod_name)
        if mod:
            try:
                if (
                    mod_name == "torch"
                    and hasattr(mod, "cuda")
                    and mod.cuda.is_available()
                ):
                    print("[GPU] Detected via torch.cuda")
                    return True
                if mod_name == "cupy":
                    print("[GPU] Detected via cupy")
                    return True
            except Exception:
                pass
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            print("[GPU] Detected via nvidia-smi")
            return True
    except Exception:
        pass
    print("[GPU] No GPU detected (or not accessible)")
    return False


# =========================================================
# ENVIRONMENT PROFILE
# =========================================================

@dataclass
class EnvironmentProfile:
    os_name: str
    elevated: bool
    has_gpu: bool
    optional_libs: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def detect(cls, loader: AutoLoader):
        os_name = platform.system()
        elevated = Elevation.is_elevated()
        has_gpu = detect_gpu(loader)
        optional_libs = {
            "numpy": loader.available("numpy"),
            "psutil": loader.available("psutil"),
            "pywinauto": loader.available("pywinauto"),
            "winreg": loader.available("winreg"),
            "dbus": loader.available("dbus"),
        }
        return cls(os_name, elevated, has_gpu, optional_libs)


# =========================================================
# MEMORY STATE
# =========================================================

@dataclass
class MemoryState:
    loop_interval_seconds: int = 5
    history_cpu: List[float] = field(default_factory=list)
    history_mem: List[float] = field(default_factory=list)
    max_history: int = 100
    success_count: int = 0
    error_count: int = 0
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    max_anomalies: int = 200

    @classmethod
    def load(cls) -> "MemoryState":
        if not os.path.exists(MEMORY_FILE):
            print("[Memory] No existing memory file, starting fresh")
            return cls()
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("[Memory] Loaded memory from file")
            return cls(**data)
        except Exception as e:
            print(f"[Memory] Failed to load memory: {e}")
            return cls()

    def save(self):
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
            print("[Memory] Saved memory to file")
        except Exception as e:
            print(f"[Memory] Failed to save memory: {e}")

    def add_sample(self, cpu: float, mem: float):
        if cpu >= 0:
            self.history_cpu.append(cpu)
        if mem >= 0:
            self.history_mem.append(mem)
        if len(self.history_cpu) > self.max_history:
            self.history_cpu = self.history_cpu[-self.max_history :]
        if len(self.history_mem) > self.max_history:
            self.history_mem = self.history_mem[-self.max_history :]

    def moving_average(self, data: List[float], window: int) -> float:
        if not data:
            return 0.0
        window = max(1, min(window, len(data)))
        return sum(data[-window:]) / window

    def moving_std(self, data: List[float], window: int) -> float:
        if not data:
            return 0.0
        window = max(1, min(window, len(data)))
        segment = data[-window:]
        mean = sum(segment) / len(segment)
        var = sum((x - mean) ** 2 for x in segment) / max(1, len(segment) - 1)
        return var ** 0.5

    def predict_next(self, data: List[float], window: int = 5) -> float:
        return self.moving_average(data, window)

    def record_anomaly(self, kind: str, cpu: float, mem: float, score: float):
        entry = {
            "kind": kind,
            "cpu": cpu,
            "mem": mem,
            "score": score,
            "timestamp": time.time(),
        }
        self.anomalies.append(entry)
        if len(self.anomalies) > self.max_anomalies:
            self.anomalies = self.anomalies[-self.max_anomalies :]
        print(f"[Memory] Recorded anomaly: {entry}")


# =========================================================
# POLICY ENGINE
# =========================================================

@dataclass
class Policy:
    allow_giant_if_elevated: bool = True
    allow_gpu_tasks: bool = True
    allow_windows_organs: bool = True
    allow_linux_organs: bool = True
    max_agents: int = 6
    min_interval: int = 1
    max_interval: int = 30
    anomaly_sensitivity: float = 2.0

    def can_use_giant(self, env: EnvironmentProfile) -> bool:
        return self.allow_giant_if_elevated and env.elevated

    def can_use_gpu(self, env: EnvironmentProfile) -> bool:
        return self.allow_gpu_tasks and env.has_gpu

    def can_use_windows_organs(self, env: EnvironmentProfile) -> bool:
        return self.allow_windows_organs and env.os_name.lower().startswith("win")

    def can_use_linux_organs(self, env: EnvironmentProfile) -> bool:
        return self.allow_linux_organs and env.os_name.lower() == "linux"

    def adapt_from_predictions(
        self,
        memory: MemoryState,
        predicted_cpu: float,
        predicted_mem: float,
        cycle_errors: int,
        swarm_anomaly_vote: bool,
    ) -> None:
        print("[Policy] Adapting from predictions")
        print(
            f"[Policy] Current interval={memory.loop_interval_seconds}, "
            f"pred_cpu={predicted_cpu:.1f}, pred_mem={predicted_mem:.1f}, "
            f"errors={cycle_errors}, anomaly_vote={swarm_anomaly_vote}"
        )

        new_interval = memory.loop_interval_seconds

        if predicted_cpu > 80 or predicted_mem > 85:
            new_interval += 3

        if (
            predicted_cpu < 40
            and predicted_mem < 60
            and cycle_errors == 0
            and not swarm_anomaly_vote
        ):
            new_interval -= 1

        if cycle_errors > 0 or swarm_anomaly_vote:
            new_interval += 2

        new_interval = max(self.min_interval, min(self.max_interval, new_interval))
        print(f"[Policy] New interval={new_interval}")
        memory.loop_interval_seconds = new_interval


# =========================================================
# WINDOWS ORGANS
# =========================================================

class WindowsOrgans:
    def __init__(self, loader: AutoLoader):
        self.loader = loader
        self.pywinauto = loader.load("pywinauto")
        self.winreg = loader.load("winreg")
        self.psutil = loader.load("psutil")

    def registry_sample(self):
        if not self.winreg:
            print("[WinOrgans] winreg not available")
            return
        try:
            key = self.winreg.OpenKey(
                self.winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE", 0, self.winreg.KEY_READ
            )
            subkey_count, _, _ = self.winreg.QueryInfoKey(key)
            print(
                f"[WinOrgans] HKLM\\SOFTWARE has ~{subkey_count} subkeys (read-only sample)"
            )
            self.winreg.CloseKey(key)
        except Exception as e:
            print(f"[WinOrgans] Registry sample failed: {e}")

    def list_services(self):
        if not self.psutil:
            print("[WinOrgans] psutil not available for services")
            return
        try:
            services = getattr(self.psutil, "win_service_iter", None)
            if services is None:
                print("[WinOrgans] psutil.win_service_iter not available")
                return
            count = 0
            for svc in services():
                if count >= 5:
                    break
                print(
                    f"[WinOrgans] Service: {svc.name()} (status={svc.status()})"
                )
                count += 1
        except Exception as e:
            print(f"[WinOrgans] list_services failed: {e}")

    def ui_automation_sample(self):
        if not self.pywinauto:
            print("[WinOrgans] pywinauto not available for UIAutomation")
            return
        try:
            from pywinauto import Desktop

            desktop = Desktop(backend="uia")
            windows = desktop.windows()
            print(
                f"[WinOrgans] Top-level windows detected: {len(windows)} (showing up to 5)"
            )
            for w in windows[:5]:
                print(f"[WinOrgans] Window: {w.window_text()!r}")
        except Exception as e:
            print(f"[WinOrgans] UIAutomation sample failed: {e}")


# =========================================================
# LINUX ORGANS
# =========================================================

class LinuxOrgans:
    def __init__(self, loader: AutoLoader):
        self.loader = loader
        self.dbus = loader.load("dbus")

    def systemd_units_sample(self):
        try:
            result = subprocess.run(
                [
                    "systemctl",
                    "list-units",
                    "--type=service",
                    "--no-pager",
                    "--no-legend",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )
            if result.returncode != 0:
                print(f"[LinuxOrgans] systemctl error: {result.stderr.strip()}")
                return
            lines = result.stdout.strip().splitlines()
            print("[LinuxOrgans] systemd services (sample up to 5):")
            for line in lines[:5]:
                print(f"[LinuxOrgans] {line}")
        except FileNotFoundError:
            print("[LinuxOrgans] systemctl not found")
        except Exception as e:
            print(f"[LinuxOrgans] systemd_units_sample failed: {e}")

    def dbus_sample(self):
        if not self.dbus:
            print("[LinuxOrgans] dbus module not available")
            return
        try:
            bus = self.dbus.SessionBus()
            names = bus.list_names()
            print("[LinuxOrgans] D-Bus session names (sample up to 5):")
            for name in names[:5]:
                print(f"[LinuxOrgans] {name}")
        except Exception as e:
            print(f"[LinuxOrgans] dbus_sample failed: {e}")

    def procfs_sample(self):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                lines = [next(f).strip() for _ in range(5)]
            print("[LinuxOrgans] /proc/meminfo sample:")
            for line in lines:
                print(f"[LinuxOrgans] {line}")
        except Exception as e:
            print(f"[LinuxOrgans] procfs_sample failed: {e}")


# =========================================================
# BASE AGENT
# =========================================================

class BaseAgent:
    def __init__(
        self,
        name: str,
        role: str,
        loader: AutoLoader,
        env: EnvironmentProfile,
        policy: Policy,
    ):
        self.name = name
        self.role = role
        self.loader = loader
        self.env = env
        self.policy = policy
        self.mode = "ant"
        self.windows_organs = (
            WindowsOrgans(loader) if policy.can_use_windows_organs(env) else None
        )
        self.linux_organs = (
            LinuxOrgans(loader) if policy.can_use_linux_organs(env) else None
        )

    def shrink(self):
        print(f"[{self.name}][{self.role}] Shrinking -> ANT mode")
        self.mode = "ant"

    def grow(self):
        print(f"[{self.name}][{self.role}] Growing -> GIANT mode")
        self.mode = "giant"

    def decide_mode(self):
        if self.policy.can_use_giant(self.env) and self.role == "analyst":
            print(f"[{self.name}][{self.role}] Policy allows GIANT mode")
            self.grow()
        else:
            print(f"[{self.name}][{self.role}] Policy -> ANT mode")
            self.shrink()

    def _read_metrics(self) -> Tuple[float, float]:
        ps = self.loader.load("psutil")
        if not ps:
            print(f"[{self.name}][{self.role}] psutil missing -> no metrics")
            return -1.0, -1.0
        cpu = ps.cpu_percent(interval=0.2)
        mem = ps.virtual_memory().percent
        print(f"[{self.name}][{self.role}] CPU={cpu}%, MEM={mem}%")
        return cpu, mem

    def run(self, memory: MemoryState) -> Tuple[bool, float, float, bool]:
        print(
            f"[{self.name}][{self.role}] OS={self.env.os_name}, "
            f"Privilege={Elevation.describe()}, GPU={self.env.has_gpu}"
        )
        self.decide_mode()
        try:
            if self.mode == "ant":
                return self.run_ant(memory)
            else:
                return self.run_giant(memory)
        except Exception as e:
            print(f"[{self.name}][{self.role}] ERROR during run: {e}")
            return False, -1.0, -1.0, False

    def run_ant(self, memory: MemoryState) -> Tuple[bool, float, float, bool]:
        cpu, mem = self._read_metrics()
        if self.role == "scout":
            if self.windows_organs:
                self.windows_organs.registry_sample()
            if self.linux_organs:
                self.linux_organs.procfs_sample()
        anomaly = self.detect_anomaly(memory, cpu, mem)
        return True, cpu, mem, anomaly

    def run_giant(self, memory: MemoryState) -> Tuple[bool, float, float, bool]:
        cpu, mem = self._read_metrics()
        numpy = self.loader.load("numpy")
        if self.role == "analyst" and numpy:
            data = numpy.random.rand(50000)
            print(
                f"[{self.name}][{self.role}] Mean={data.mean():.4f}, Std={data.std():.4f}"
            )
        elif self.role == "analyst":
            print(f"[{self.name}][{self.role}] numpy missing -> reduced analysis")

        if self.policy.can_use_gpu(self.env) and self.role == "analyst":
            print(f"[{self.name}][{self.role}] GPU tasks allowed (placeholder)")

        if self.windows_organs and self.role == "analyst":
            self.windows_organs.list_services()
            self.windows_organs.ui_automation_sample()
        if self.linux_organs and self.role == "analyst":
            self.linux_organs.systemd_units_sample()
            self.linux_organs.dbus_sample()

        if self.role == "archivist":
            print(f"[{self.name}][{self.role}] Archiving metrics only")

        print(f"[{self.name}][{self.role}] Work done -> shrinking back to ANT")
        self.shrink()
        anomaly = self.detect_anomaly(memory, cpu, mem)
        return True, cpu, mem, anomaly

    def detect_anomaly(self, memory: MemoryState, cpu: float, mem: float) -> bool:
        if cpu < 0 or mem < 0:
            return False
        cpu_mean = memory.moving_average(memory.history_cpu, window=15)
        cpu_std = memory.moving_std(memory.history_cpu, window=15)
        mem_mean = memory.moving_average(memory.history_mem, window=15)
        mem_std = memory.moving_std(memory.history_mem, window=15)
        cpu_z = (cpu - cpu_mean) / cpu_std if cpu_std > 0 else 0.0
        mem_z = (mem - mem_mean) / mem_std if mem_std > 0 else 0.0
        score = max(abs(cpu_z), abs(mem_z))
        threshold = self.policy.anomaly_sensitivity
        is_anomaly = score >= threshold
        if is_anomaly:
            kind = f"{self.role}_anomaly"
            print(
                f"[{self.name}][{self.role}] Anomaly detected: score={score:.2f}, cpu={cpu}, mem={mem}"
            )
            memory.record_anomaly(kind, cpu, mem, score)
        else:
            print(f"[{self.name}][{self.role}] No anomaly: score={score:.2f}")
        return is_anomaly


# =========================================================
# SWARM
# =========================================================

class Swarm:
    def __init__(
        self, loader: AutoLoader, env: EnvironmentProfile, policy: Policy, node_id: str = "A"
    ):
        self.loader = loader
        self.env = env
        self.policy = policy
        self.node_id = node_id
        self.agents: List[BaseAgent] = []
        self.sidestep = SidestepEngine()
        self._build_swarm()

    def _build_swarm(self):
        roles = ["scout", "analyst", "archivist"]
        count = max(1, min(self.policy.max_agents, 12))
        for i in range(count):
            role = roles[i % len(roles)]
            name = f"{role.capitalize()}-{i+1}"
            agent = BaseAgent(name, role, self.loader, self.env, self.policy)
            self.agents.append(agent)
        print(
            f"[Swarm] Built swarm with {len(self.agents)} nodes for logical node {self.node_id}"
        )

    def run_all(self, memory: MemoryState) -> Tuple[int, float, float, bool]:
        print("[Swarm] Running all nodes")
        total_cpu = 0.0
        total_mem = 0.0
        count = 0
        errors = 0
        anomaly_votes = 0

        for agent in self.agents:
            print("--------------------------------------------------")
            success, cpu, mem, anomaly = agent.run(memory)

            msg = {
                "payload": f"role={agent.role},cpu={cpu},mem={mem}",
                "priority": "auto",
                "organ": "telemetry",
                "lane": "analysis",
                "ritual": "tick",
                "intent": "status",
                "role": agent.role,
                "task_id": "",
            }

            issues = detect_missing_details(msg)
            if issues:
                print(f"[Swarm] Message issues for {agent.name}: {issues}")

            msg = learn_priority_from_history(self.node_id, msg)
            msg = apply_constraints(self.node_id, msg)
            log_message(self.node_id, "out", "swarm", msg)
            update_agent_stats(
                self.node_id, agent.role, msg["intent"], success=success and not anomaly
            )

            if not success:
                errors += 1
            if cpu >= 0 and mem >= 0:
                total_cpu += cpu
                total_mem += mem
                count += 1
            if anomaly:
                anomaly_votes += 1

            self.sidestep.receive(msg)

        lane, m = self.sidestep.process_one()
        if m:
            print(f"[Swarm] Sidestep processed lane={lane} msg={m}")
            log_message(self.node_id, "internal", "sidestep", m)
            m, alert = apply_guardian_policies(self, m)
            if alert:
                self.sidestep.receive(alert)
            new_msgs = run_agents(self, m)
            for nm in new_msgs:
                self.sidestep.receive(nm)
            auto = evaluate_automation_rules(self, m)
            for am in auto:
                self.sidestep.receive(am)

        if count > 0:
            avg_cpu = total_cpu / count
            avg_mem = total_mem / count
        else:
            avg_cpu = 0.0
            avg_mem = 0.0

        swarm_anomaly = anomaly_votes > 0
        neighbor = choose_neighbor_rl(self.node_id)
        if neighbor:
            reward = -1.0 if swarm_anomaly or errors > 0 else 1.0
            update_q(self.node_id, neighbor["id"], reward)
            print(
                f"[Swarm] RL updated neighbor {neighbor['id']} with reward={reward}"
            )

        print(
            f"[Swarm] Summary: errors={errors}, avg_cpu={avg_cpu:.1f}, "
            f"avg_mem={avg_mem:.1f}, anomaly_votes={anomaly_votes}, "
            f"swarm_anomaly={swarm_anomaly}"
        )
        return errors, avg_cpu, avg_mem, swarm_anomaly


# =========================================================
# NEURAL INFERENCE ORGAN + ML PRIORITY
# =========================================================

class NeuralInferenceOrgan:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.weight = 0.5
        self.bias = 0.0
        self.lr = 0.01

    def _features(self, msg: Dict[str, Any]) -> float:
        p = msg.get("payload", "")
        length = len(p)
        has_error = 1.0 if ("error" in p.lower() or "fail" in p.lower()) else 0.0
        prio = msg.get("priority", "normal")
        prio_val = 0.0 if prio == "normal" else 0.5 if prio == "sidestep" else 1.0
        return 0.0005 * length + has_error + prio_val

    def infer(self, msg: Dict[str, Any]) -> Optional[float]:
        x = self._features(msg)
        y = self.weight * x + self.bias
        y = 1.0 / (1.0 + pow(2.71828, -y))
        return max(0.0, min(1.0, y))

    def train_step(self, msg: Dict[str, Any], label: float):
        x = self._features(msg)
        y = self.infer(msg)
        if y is None:
            return
        grad = (y - label)
        self.weight -= self.lr * grad * x
        self.bias -= self.lr * grad


def ml_infer_priority(msg: Dict[str, Any]) -> str:
    payload = msg.get("payload", "").lower()
    if "error" in payload or "fail" in payload:
        return "jump"
    if "warn" in payload or "slow" in payload:
        return "sidestep"
    return "normal"


def gpu_infer_score(msg: Dict[str, Any]) -> Optional[float]:
    return None


# =========================================================
# BORG-STYLE REGENERATION CHAMBER
# =========================================================

class RegenerationChamber:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.last_repair = 0.0
        self.cooldown = 30.0

    def _db_exists(self) -> bool:
        return os.path.exists(db_path(self.node_id))

    def _swarm_db_exists(self) -> bool:
        return os.path.exists(SWARM_DB)

    def _too_many_anomalies(self, memory: MemoryState) -> bool:
        if not memory.anomalies:
            return False
        recent = [a for a in memory.anomalies if now() - a["timestamp"] < 120]
        return len(recent) > 50

    def _too_many_errors(self, memory: MemoryState) -> bool:
        return memory.error_count > 100 and memory.error_count > memory.success_count * 2

    def _reset_memory(self, memory: MemoryState):
        print("[Regen] Resetting memory state (loop interval, anomalies, error counters)")
        memory.loop_interval_seconds = 5
        memory.anomalies.clear()
        memory.error_count = 0

    def _reinit_dbs(self):
        print("[Regen] Re-initializing node and swarm DBs")
        init_db(self.node_id)
        init_swarm_db()

    def _heal_neighbors(self):
        print("[Regen] Marking all neighbors as healthy (soft reset)")
        all_neighbors = get_neighbors(self.node_id, include_unhealthy=True)
        for n in all_neighbors:
            mark_neighbor_health(self.node_id, n["id"], True)

    def check_and_repair(self, memory: MemoryState):
        now_t = now()
        if now_t - self.last_repair < self.cooldown:
            return

        need_repair = False

        if not self._db_exists() or not self._swarm_db_exists():
            print("[Regen] DB missing -> repair needed")
            need_repair = True

        if self._too_many_anomalies(memory):
            print("[Regen] Excess anomalies -> repair needed")
            need_repair = True

        if self._too_many_errors(memory):
            print("[Regen] Excess errors -> repair needed")
            need_repair = True

        if not need_repair:
            return

        print("[Regen] ENTERING REGENERATION CYCLE")
        self._reset_memory(memory)
        self._reinit_dbs()
        self._heal_neighbors()
        memory.save()
        self.last_repair = now_t
        print("[Regen] REGENERATION COMPLETE")


# =========================================================
# NODE HTTP SERVER + NODESERVER
# =========================================================

class NodeHTTPHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if self.path == "/message":
            sender_id = payload.get("from_id")
            enc = payload.get("enc")
            sig = payload.get("sig")
            if not sender_id or not enc or not sig:
                self._send_json({"error": "invalid_message"}, code=400)
                return
            if sender_id not in TRUSTED_NODE_IDS:
                self._send_json({"error": "untrusted_sender"}, code=403)
                return
            raw_str = enc
            if not verify_message(sender_id, raw_str, sig):
                self._send_json({"error": "bad_signature"}, code=403)
                return
            msg = decrypt_payload_from(self.server.node.node_id, enc)
            self.server.node.receive_message(msg)
            self._send_json({"status": "ok"})
        elif self.path == "/neighbor":
            nid = payload.get("id")
            host = payload.get("host", "localhost")
            port = int(payload.get("port", 0))
            self.server.node.add_neighbor(nid, host, port)
            self._send_json({"status": "neighbor_added"})
        elif self.path == "/constraint":
            organ = payload.get("organ") or None
            lane = payload.get("lane") or None
            ritual = payload.get("ritual") or None
            priority = payload.get("priority") or "normal"
            self.server.node.add_constraint(organ, lane, ritual, priority)
            self._send_json({"status": "constraint_added"})
        else:
            self._send_json({"error": "unknown"}, code=404)

    def do_GET(self):
        if self.path == "/state":
            self._send_json(self.server.node.get_state())
        elif self.path == "/topology":
            self._send_json(self.server.node.get_topology())
        else:
            self._send_json({"error": "unknown"}, code=404)


class NodeServer:
    def __init__(self, node_id, port):
        self.node_id = node_id
        self.port = port
        init_db(node_id)
        init_swarm_db()
        self.engine = SidestepEngine()
        self.running = True
        self.nn_organ = NeuralInferenceOrgan(node_id)
        self.regen = RegenerationChamber(node_id)
        self.server = HTTPServer(("0.0.0.0", port), NodeHTTPHandler)
        self.server.node = self

    def log(self, msg):
        print(f"[Node {self.node_id}] {msg}")

    def receive_message(self, msg):
        issues = detect_missing_details(msg)
        if issues:
            self.log(f"Missing details: {issues}")
        msg = apply_constraints(self.node_id, msg)
        if msg.get("priority") in (None, "", "auto"):
            msg["priority"] = ml_infer_priority(msg)
        msg = learn_priority_from_history(self.node_id, msg)

        nn_score = self.nn_organ.infer(msg)
        if nn_score is not None:
            msg["nn_score"] = nn_score
            if nn_score > 0.8:
                msg["priority"] = "jump"
            elif nn_score > 0.5 and msg["priority"] == "normal":
                msg["priority"] = "sidestep"

        score = gpu_infer_score(msg)
        if score is not None and score > 0.8:
            msg["priority"] = "jump"

        msg, guardian_alert = apply_guardian_policies(self, msg)
        if guardian_alert:
            self.broadcast_message(guardian_alert)

        self.engine.receive(msg)

    def add_neighbor(self, neighbor_id, host, port):
        add_neighbor(self.node_id, neighbor_id, host, port)
        self.log(f"Neighbor added: {neighbor_id}@{host}:{port}")

    def add_constraint(self, organ, lane, ritual, priority):
        add_constraint(self.node_id, organ, lane, ritual, priority)
        self.log(
            f"Constraint added: organ={organ} lane={lane} ritual={ritual} -> {priority}"
        )

    def get_state(self):
        return get_state_summary(self.node_id)

    def get_topology(self):
        return {
            "node_id": self.node_id,
            "neighbors": get_neighbors(self.node_id, include_unhealthy=True),
        }

    def autonomous_behavior(self):
        neighbors = get_neighbors(self.node_id)
        for n in neighbors:
            payload = f"ping from {self.node_id}"
            msg = {
                "from": self.node_id,
                "payload": payload,
                "priority": "auto",
                "organ": "heartbeat",
                "lane": "control",
                "ritual": "ping",
            }
            msg = apply_constraints(self.node_id, msg)
            msg = learn_priority_from_history(self.node_id, msg)
            self.send_to_neighbor(n, msg)

    def broadcast_message(self, msg):
        neighbors = get_neighbors(self.node_id)
        for n in neighbors:
            self.send_to_neighbor(n, msg)

    def send_to_neighbor(self, neighbor, msg):
        host = neighbor["host"]
        port = neighbor["port"]
        nid = neighbor["id"]
        url = f"http://{host}:{port}/message"
        enc = encrypt_payload_for(nid, msg)
        sig = sign_message(self.node_id, enc)
        data = json.dumps({"from_id": self.node_id, "enc": enc, "sig": sig}).encode(
            "utf-8"
        )
        req = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=1)
            log_message(self.node_id, "out", nid, msg)
            mark_neighbor_health(self.node_id, nid, True)
            update_q(self.node_id, nid, reward=1.0)
            self.log(f"-> {nid} ({msg.get('priority')}): {msg.get('payload')}")
        except Exception as e:
            self.log(f"Send error to {nid}: {e}")
            mark_neighbor_health(self.node_id, nid, False)
            update_q(self.node_id, nid, reward=-1.0)

    def self_heal_topology(self):
        all_neighbors = get_neighbors(self.node_id, include_unhealthy=True)
        for n in all_neighbors:
            if not n["healthy"]:
                if now() - n["last_fail"] > 10:
                    try:
                        url = f"http://{n['host']}:{n['port']}/state"
                        urllib.request.urlopen(url, timeout=1)
                        mark_neighbor_health(self.node_id, n["id"], True)
                        self.log(f"Neighbor {n['id']} healed")
                    except Exception:
                        pass

    def telemetry_sensor_loop(self):
        if psutil is None:
            self.log("psutil not available, telemetry disabled")
            return
        while self.running:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            payload = f"cpu={cpu:.1f} mem={mem:.1f}"
            msg = {
                "from": self.node_id,
                "payload": payload,
                "priority": "auto",
                "organ": "telemetry",
                "lane": "system",
                "ritual": "sample",
            }
            n = choose_neighbor_rl(self.node_id)
            if n:
                self.send_to_neighbor(n, msg)
            time.sleep(3)

    def filesystem_sensor_loop(self):
        if Observer is None:
            self.log("watchdog not available, filesystem sensor disabled")
            return

        class FSHandler(FileSystemEventHandler):
            def __init__(self, node):
                super().__init__()
                self.node = node

            def on_any_event(self, event):
                payload = f"{event.event_type}: {event.src_path}"
                msg = {
                    "from": self.node.node_id,
                    "payload": payload,
                    "priority": "auto",
                    "organ": "filesystem",
                    "lane": "io",
                    "ritual": "fs_event",
                }
                n = choose_neighbor_rl(self.node.node_id)
                if n:
                    self.node.send_to_neighbor(n, msg)

        observer = Observer()
        handler = FSHandler(self)
        observer.schedule(handler, WATCH_DIR, recursive=True)
        observer.start()
        self.log(f"Filesystem sensor watching {WATCH_DIR}")
        try:
            while self.running:
                time.sleep(1)
        finally:
            observer.stop()
            observer.join()

    def uia_sensor_loop(self):
        if ctypes is None or os.name != "nt":
            self.log("UIAutomation not available")
            return

        user32 = ctypes.windll.user32

        def get_title():
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value

        last_title = None
        while self.running:
            title = get_title()
            if title and title != last_title:
                last_title = title
                payload = f"foreground={title}"
                msg = {
                    "from": self.node_id,
                    "payload": payload,
                    "priority": "auto",
                    "organ": "uiautomation",
                    "lane": "focus",
                    "ritual": "window_change",
                }
                n = choose_neighbor_rl(self.node_id)
                if n:
                    self.send_to_neighbor(n, msg)
            time.sleep(2)

    def process_loop(self):
        memory = MemoryState.load()
        while self.running:
            kind, msg = self.engine.process_one()
            if msg:
                peer = msg.get("from", "?")
                log_message(self.node_id, "in", peer, msg)
                self.log(f"{kind}-processed: {msg}")

                if msg.get("priority") == "jump":
                    self.nn_organ.train_step(msg, label=1.0)
                else:
                    self.nn_organ.train_step(msg, label=0.0)

                agent_outputs = run_agents(self, msg)
                for out in agent_outputs:
                    target_node = out.get("target_node")
                    if target_node and target_node in NODES:
                        host, port = NODES[target_node]
                        enc = encrypt_payload_for(target_node, out)
                        sig = sign_message(self.node_id, enc)
                        try:
                            http_post(
                                host,
                                port,
                                "/message",
                                {"from_id": self.node_id, "enc": enc, "sig": sig},
                            )
                            self.log(f"[AGENT] {self.node_id} -> {target_node}: {out}")
                        except Exception as e:
                            self.log(f"[AGENT ERROR] {e}")
                    else:
                        n = choose_neighbor_rl(self.node_id)
                        if n:
                            self.send_to_neighbor(n, out)

                auto_actions = evaluate_automation_rules(self, msg)
                for act in auto_actions:
                    target_node = act.get("target_node")
                    if target_node and target_node in NODES:
                        host, port = NODES[target_node]
                        m = act.copy()
                        m.pop("target_node", None)
                        enc = encrypt_payload_for(target_node, m)
                        sig = sign_message(self.node_id, enc)
                        try:
                            http_post(
                                host,
                                port,
                                "/message",
                                {"from_id": self.node_id, "enc": enc, "sig": sig},
                            )
                            self.log(f"[AUTO] rule -> {target_node}: {m}")
                        except Exception as e:
                            self.log(f"[AUTO ERROR] {e}")

                self.regen.check_and_repair(memory)

            if now() % 5 < 0.2:
                self.autonomous_behavior()
            self.self_heal_topology()
            time.sleep(0.1)

    def start(self):
        self.log(f"Listening on port {self.port}")
        t = threading.Thread(target=self.process_loop, daemon=True)
        t.start()
        threading.Thread(target=self.telemetry_sensor_loop, daemon=True).start()
        threading.Thread(target=self.filesystem_sensor_loop, daemon=True).start()
        threading.Thread(target=self.uia_sensor_loop, daemon=True).start()
        self.server.serve_forever()


# =========================================================
# Borg mesh — network within the network (overlay)
# ============================================================

class BorgMesh:
    def __init__(
        self,
        memory: MemoryManager,
        comms: BorgCommsRouter,
        guardian: SecurityGuardian,
    ):
        self.nodes = {}  # url -> {"state": discovered/built/enforced, "risk":0-100, "seen": int}
        self.edges = set()  # (src, dst)
        self.memory = memory
        self.comms = comms
        self.guardian = guardian
        self.max_corridors = BORG_MESH_CONFIG["max_corridors"]

    def _risk(self, snippet: str) -> int:
        dis = self.guardian.disassemble(snippet or "")
        base = int(dis["entropy"] * 12)
        base += len(dis["pattern_flags"]) * 10
        return max(0, min(100, base))

    def discover(self, url: str, snippet: str, links: list):
        risk = self._risk(snippet)
        node = self.nodes.get(
            url, {"state": "discovered", "risk": risk, "seen": 0}
        )
        node["state"] = "discovered"
        node["risk"] = risk
        node["seen"] += 1
        self.nodes[url] = node
        for l in links[:20]:
            self.edges.add((url, l))
        evt = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "discover",
            "url": url,
            "risk": risk,
            "links": len(links),
        }
        self.memory.record_mesh_event(evt)
        self.comms.send_secure(
            "mesh:discover", f"{url} risk={risk} links={len(links)}", "Default"
        )

    def build(self, url: str):
        if url not in self.nodes:
            return False
        self.nodes[url]["state"] = "built"
        evt = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "build",
            "url": url,
        }
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:build", f"{url} built", "Default")
        return True

    def enforce(self, url: str, snippet: str):
        if url not in self.nodes:
            return False
        verdict = self.guardian.reassemble(
            url,
            privacy_filter(snippet or "")[0],
            raw_pii_hits=self.guardian._pii_count(snippet or ""),
        )
        status = verdict.get("status", "HOSTILE")
        self.nodes[url]["state"] = "enforced"
        self.nodes[url]["risk"] = (
            0 if status == "SAFE_FOR_TRAVEL" else max(50, self.nodes[url]["risk"])
        )
        evt = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "enforce",
            "url": url,
            "status": status,
        }
        self.memory.record_mesh_event(evt)
        self.comms.send_secure(
            "mesh:enforce", f"{url} status={status}", "Default"
        )
        return True

    def stats(self):
        total = len(self.nodes)
        discovered = sum(
            1 for n in self.nodes.values() if n["state"] == "discovered"
        )
        built = sum(1 for n in self.nodes.values() if n["state"] == "built")
        enforced = sum(
            1 for n in self.nodes.values() if n["state"] == "enforced"
        )
        return {
            "total": total,
            "discovered": discovered,
            "built": built,
            "enforced": enforced,
            "corridors": len(self.edges),
        }


# ============================================================
# Borg roles — scanners, workers, enforcers
# ============================================================

class BorgScanner(threading.Thread):
    def __init__(
        self,
        mesh: BorgMesh,
        in_events: queue.Queue,
        out_ops: queue.Queue,
        label="SCANNER",
    ):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.in_events = in_events
        self.out_ops = out_ops
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                ev = self.in_events.get(timeout=1.0)
            except queue.Empty:
                continue
            unseen_links = [
                l
                for l in ev.links
                if l not in self.mesh.nodes
                and random.random() < BORG_MESH_CONFIG["unknown_bias"]
            ]
            self.mesh.discover(ev.url, ev.snippet, unseen_links or ev.links)
            self.out_ops.put(("build", ev.url))
            time.sleep(random.uniform(0.2, 0.6))


class BorgWorker(threading.Thread):
    def __init__(self, mesh: BorgMesh, ops_q: queue.Queue, label="WORKER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.ops_q = ops_q
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                op, url = self.ops_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if op == "build":
                if self.mesh.build(url):
                    self.ops_q.put(("enforce", url))
            elif op == "enforce":
                self.mesh.enforce(url, snippet="")
            time.sleep(random.uniform(0.2, 0.5))


class BorgEnforcer(threading.Thread):
    def __init__(
        self, mesh: BorgMesh, guardian: SecurityGuardian, label="ENFORCER"
    ):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.guardian = guardian
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            for url, meta in list(self.mesh.nodes.items()):
                if meta["state"] in ("built", "enforced") and random.random() < 0.15:
                    self.mesh.enforce(url, snippet="")
            time.sleep(1.2)


# =========================================================
# MAIN (BORG OS CONTINUOUS CYBERNETIC LOOP)
# =========================================================

def main():
    print("[Main] Starting Borg OS Hybrid Swarm Core (Full Assimilation Mode)")
    print(f"[Main] Privilege: {Elevation.describe()}")

    init_swarm_db()

    node_id = os.getenv("NODE_ID", "A")
    if node_id not in NODES:
        node_id = "A"

    init_db(node_id)

    memory = MemoryState.load()
    loader = AutoLoader()
    env = EnvironmentProfile.detect(loader)
    policy = Policy()
    swarm = Swarm(loader, env, policy, node_id=node_id)
    regen = RegenerationChamber(node_id)

    last_cycle_time = time.monotonic()

    while True:
        now_t = time.monotonic()
        elapsed = now_t - last_cycle_time

        if elapsed >= memory.loop_interval_seconds:
            print("==================================================")
            print(
                f"[Main] Swarm cycle starting (interval={memory.loop_interval_seconds}s, elapsed={elapsed:.2f}s)"
            )

            errors, avg_cpu, avg_mem, swarm_anomaly = swarm.run_all(memory)

            if avg_cpu >= 0 and avg_mem >= 0:
                memory.add_sample(avg_cpu, avg_mem)

            if errors == 0:
                memory.success_count += 1
            else:
                memory.error_count += errors

            pred_cpu = memory.predict_next(memory.history_cpu, window=5)
            pred_mem = memory.predict_next(memory.history_mem, window=5)
            print(
                f"[Main] Predictions: next_cpu={pred_cpu:.1f}, next_mem={pred_mem:.1f}"
            )

            policy.adapt_from_predictions(
                memory, pred_cpu, pred_mem, errors, swarm_anomaly
            )

            regen.check_and_repair(memory)
            memory.save()

            last_cycle_time = now_t
        # Full assimilation: no sleep here


if __name__ == "__main__":
    mode = os.getenv("BORG_MODE", "SWARM")
    if mode.upper() == "NODE":
        nid = os.getenv("NODE_ID", "A")
        host, port = NODES.get(nid, ("0.0.0.0", 9101))
        server = NodeServer(nid, port)
        server.start()
    else:
        main()

