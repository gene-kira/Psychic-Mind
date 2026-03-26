import os
import sys
import json
import time
import math
import queue
import random
import sqlite3
import threading
import subprocess
import base64
import hashlib

from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import urllib.error

# Optional imports
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    Observer = None
    FileSystemEventHandler = object

try:
    import ctypes
except ImportError:
    ctypes = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# === AUTO-ELEVATION CHECK ===
def ensure_admin():
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
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

"""
CLUSTER CONFIGURATION

You can override the default node map with an environment variable:

  CLUSTER_CONFIG='[
    {"id":"A","host":"10.0.0.10","port":9001},
    {"id":"B","host":"10.0.0.11","port":9002},
    {"id":"C","host":"10.0.0.12","port":9003},
    {"id":"D","host":"10.0.1.10","port":9004},
    {"id":"E","host":"10.0.1.11","port":9005},
    {"id":"F","host":"10.0.2.10","port":9006},
    {"id":"G","host":"10.0.2.11","port":9007}
  ]'

This lets you span multiple physical machines / subnets:

- Rack 1 / Machine 1: 10.0.0.10 (Node A, port 9001)
- Rack 1 / Machine 2: 10.0.0.11 (Node B, port 9002)
- Rack 1 / Machine 3: 10.0.0.12 (Node C, port 9003)
- Rack 2 / Machine 1: 10.0.1.10 (Node D, port 9004)
- Rack 2 / Machine 2: 10.0.1.11 (Node E, port 9005)
- Rack 3 / Machine 1: 10.0.2.10 (Node F, port 9006)
- Rack 3 / Machine 2: 10.0.2.11 (Node G, port 9007)

If CLUSTER_CONFIG is not set, we fall back to a single‑machine layout
with tuned ports in the 9100+ range.
"""

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
    "telemetry", "filesystem", "browser", "game", "external_api",
    "uiautomation", "heartbeat", "ml_inference", "gpu_inference",
    "guardian", "planner", "executor", "analyst", "nn_inference"
]
DEFAULT_LANES = [
    "system", "io", "activity", "telemetry", "signal",
    "focus", "control", "analysis", "guardian", "tasks"
]
DEFAULT_RITUALS = [
    "sample", "fs_event", "url_visit", "tick", "ping",
    "window_change", "alert", "classify", "gpu_score",
    "task_create", "task_assign", "task_result", "guardian_check",
    "nn_score"
]

WATCH_DIR = os.path.abspath(".")
BASE_ENCRYPTION_KEY = b"mythic_shared_key"

# Zero-trust: per-node secrets + allow-list
TRUSTED_NODE_IDS = set(NODES.keys())

def node_secret(node_id: str) -> bytes:
    return hashlib.sha256(BASE_ENCRYPTION_KEY + node_id.encode("utf-8")).digest()

# =========================================================
# UTILS / ENCRYPTION / SIGNING
# =========================================================

def now():
    return time.time()

def xor_encrypt(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))

def encrypt_payload_for(node_id: str, obj):
    raw = json.dumps(obj).encode("utf-8")
    key = node_secret(node_id)
    enc = xor_encrypt(raw, key)
    return base64.b64encode(enc).decode("utf-8")

def decrypt_payload_from(node_id: str, s):
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

def db_path(node_id):
    return os.path.join(DB_DIR, f"sidestep_node_{node_id}.db")

# =========================================================
# SWARM-WIDE DB
# =========================================================

def init_swarm_db():
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    c.execute("""
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
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS automation_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            condition_json TEXT,
            action_json TEXT,
            enabled INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_swarm(node_id, direction, peer, msg):
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO swarm_messages(node_id, direction, peer, payload, priority,
                                   organ, lane, ritual, intent, role, task_id, ts)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
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
        now()
    ))
    conn.commit()
    conn.close()

def query_swarm(limit=100, priority=None, tag=None):
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    base = "SELECT node_id, direction, peer, payload, priority, organ, lane, ritual, intent, role, task_id, ts FROM swarm_messages"
    params = []
    where = []
    if priority:
        where.append("priority = ?")
        params.append(priority)
    if tag:
        where.append("(organ = ? OR lane = ? OR ritual = ? OR intent = ? OR role = ?)")
        params.extend([tag, tag, tag, tag, tag])
    if where:
        base += " WHERE " + " AND ".join(where)
    base += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    c.execute(base, params)
    rows = c.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
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
            "ts": r[11]
        })
    return results

def add_automation_rule(name, condition, action, enabled=True):
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO automation_rules(name, condition_json, action_json, enabled)
        VALUES (?,?,?,?)
    """, (name, json.dumps(condition), json.dumps(action), 1 if enabled else 0))
    conn.commit()
    conn.close()

def load_automation_rules():
    conn = sqlite3.connect(SWARM_DB)
    c = conn.cursor()
    c.execute("SELECT id, name, condition_json, action_json, enabled FROM automation_rules WHERE enabled=1")
    rows = c.fetchall()
    conn.close()
    rules = []
    for rid, name, cond, act, en in rows:
        rules.append({
            "id": rid,
            "name": name,
            "condition": json.loads(cond),
            "action": json.loads(act),
            "enabled": bool(en)
        })
    return rules

# =========================================================
# NODE DB + LEARNING + RL
# =========================================================

def init_db(node_id):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("""
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
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS neighbors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            neighbor_id TEXT,
            host TEXT,
            port INTEGER,
            healthy INTEGER DEFAULT 1,
            last_fail REAL DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS constraints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organ TEXT,
            lane TEXT,
            ritual TEXT,
            priority TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS q_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            neighbor_id TEXT,
            state TEXT,
            action TEXT,
            q REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS agent_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_role TEXT,
            intent TEXT,
            success INTEGER,
            fail INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_message(node_id, direction, peer, msg):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("""
        INSERT INTO messages(direction, peer, payload, priority, organ, lane,
                             ritual, intent, role, task_id, ts)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
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
        now()
    ))
    conn.commit()
    conn.close()
    log_swarm(node_id, direction, peer, msg)

def add_neighbor(node_id, neighbor_id, host, port):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("INSERT INTO neighbors(neighbor_id, host, port, healthy, last_fail) VALUES (?,?,?,?,?)",
              (neighbor_id, host, port, 1, 0.0))
    conn.commit()
    conn.close()

def mark_neighbor_health(node_id, neighbor_id, healthy):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    if healthy:
        c.execute("UPDATE neighbors SET healthy=1 WHERE neighbor_id=?", (neighbor_id,))
    else:
        c.execute("UPDATE neighbors SET healthy=0, last_fail=? WHERE neighbor_id=?", (now(), neighbor_id))
    conn.commit()
    conn.close()

def get_neighbors(node_id, include_unhealthy=False):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    if include_unhealthy:
        c.execute("SELECT neighbor_id, host, port, healthy, last_fail FROM neighbors")
    else:
        c.execute("SELECT neighbor_id, host, port, healthy, last_fail FROM neighbors WHERE healthy=1")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "host": r[1], "port": r[2], "healthy": r[3], "last_fail": r[4]} for r in rows]

def get_state_summary(node_id):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE direction='in'")
    in_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM messages WHERE direction='out'")
    out_count = c.fetchone()[0]
    c.execute("SELECT payload, priority, organ, lane, ritual, intent, role, task_id, ts FROM messages ORDER BY id DESC LIMIT 5")
    last_msgs = [{
        "payload": r[0],
        "priority": r[1],
        "organ": r[2],
        "lane": r[3],
        "ritual": r[4],
        "intent": r[5],
        "role": r[6],
        "task_id": r[7],
        "ts": r[8]
    } for r in c.fetchall()]
    conn.close()
    neighbors = get_neighbors(node_id, include_unhealthy=True)
    return {
        "node_id": node_id,
        "in_count": in_count,
        "out_count": out_count,
        "last_messages": last_msgs,
        "neighbors": neighbors
    }

def load_constraints(node_id):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT organ, lane, ritual, priority FROM constraints")
    rows = c.fetchall()
    conn.close()
    rules = []
    for organ, lane, ritual, priority in rows:
        rules.append({
            "organ": organ or None,
            "lane": lane or None,
            "ritual": ritual or None,
            "priority": priority
        })
    return rules

def add_constraint(node_id, organ, lane, ritual, priority):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("INSERT INTO constraints(organ, lane, ritual, priority) VALUES (?,?,?,?)",
              (organ, lane, ritual, priority))
    conn.commit()
    conn.close()

def apply_constraints(node_id, msg):
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

def learn_priority_from_history(node_id, msg):
    if msg.get("priority") not in (None, "", "auto"):
        return msg
    organ = msg.get("organ", "")
    lane = msg.get("lane", "")
    ritual = msg.get("ritual", "")
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("""
        SELECT priority, COUNT(*) as cnt
        FROM messages
        WHERE organ = ? OR lane = ? OR ritual = ?
        GROUP BY priority
    """, (organ, lane, ritual))
    rows = c.fetchall()
    conn.close()
    if not rows:
        msg["priority"] = "normal"
        return msg
    best = max(rows, key=lambda r: r[1])[0]
    msg["priority"] = best
    return msg

def detect_missing_details(msg):
    issues = []
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

# RL helpers

def get_q_value(node_id, neighbor_id, state="default", action="send"):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT q FROM q_values WHERE neighbor_id=? AND state=? AND action=?",
              (neighbor_id, state, action))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    return 0.0

def set_q_value(node_id, neighbor_id, state, action, q):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT id FROM q_values WHERE neighbor_id=? AND state=? AND action=?",
              (neighbor_id, state, action))
    row = c.fetchone()
    if row:
        c.execute("UPDATE q_values SET q=? WHERE id=?", (q, row[0]))
    else:
        c.execute("INSERT INTO q_values(neighbor_id, state, action, q) VALUES (?,?,?,?)",
                  (neighbor_id, state, action, q))
    conn.commit()
    conn.close()

def update_q(node_id, neighbor_id, reward, state="default", action="send", alpha=0.3, gamma=0.9):
    old_q = get_q_value(node_id, neighbor_id, state, action)
    new_q = old_q + alpha * (reward + gamma * 0 - old_q)
    set_q_value(node_id, neighbor_id, state, action, new_q)

def choose_neighbor_rl(node_id):
    neighbors = get_neighbors(node_id)
    if not neighbors:
        return None
    if random.random() < 0.2:
        return random.choice(neighbors)
    scored = []
    for n in neighbors:
        q = get_q_value(node_id, n["id"])
        scored.append((q, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def update_agent_stats(node_id, role, intent, success=True):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT id, success, fail FROM agent_stats WHERE agent_role=? AND intent=?",
              (role, intent))
    row = c.fetchone()
    if row:
        sid, s, f = row
        if success:
            s += 1
        else:
            f += 1
        c.execute("UPDATE agent_stats SET success=?, fail=? WHERE id=?", (s, f, sid))
    else:
        c.execute("INSERT INTO agent_stats(agent_role, intent, success, fail) VALUES (?,?,?,?)",
                  (role, intent, 1 if success else 0, 0 if success else 1))
    conn.commit()
    conn.close()

# =========================================================
# SIMPLE ML / GPU / NN INFERENCE
# =========================================================

def ml_infer_priority(msg):
    payload = msg.get("payload", "")
    organ = msg.get("organ", "")
    ritual = msg.get("ritual", "")
    score = 0
    if "error" in payload.lower() or "fail" in payload.lower():
        score += 2
    if organ in ("filesystem", "telemetry"):
        score += 1
    if ritual in ("alert", "fs_event"):
        score += 1
    if "delete" in payload.lower():
        score += 2
    if score >= 3:
        return "jump"
    elif score == 2:
        return "sidestep"
    else:
        return "normal"

def gpu_infer_score(msg):
    if torch is None or not torch.cuda.is_available():
        return None
    payload = msg.get("payload", "")
    x = torch.tensor([len(payload)], dtype=torch.float32, device="cuda")
    w = torch.randn_like(x)
    score = torch.sigmoid(x * w).item()
    return score

class SimpleNN(nn.Module):
    def __init__(self, input_dim=8, hidden=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class NeuralInferenceOrgan:
    def __init__(self, node_id):
        self.node_id = node_id
        self.model = None
        self.device = "cpu"
        self.path = os.path.join(DB_DIR, f"nn_{node_id}.pt")
        if torch is not None and nn is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SimpleNN().to(self.device)
            if os.path.exists(self.path):
                try:
                    self.model.load_state_dict(torch.load(self.path, map_location=self.device))
                except Exception:
                    pass

    def features_from_msg(self, msg):
        payload = msg.get("payload", "")
        organ = msg.get("organ", "")
        ritual = msg.get("ritual", "")
        length = len(payload)
        has_error = 1.0 if "error" in payload.lower() else 0.0
        has_delete = 1.0 if "delete" in payload.lower() else 0.0
        is_fs = 1.0 if organ == "filesystem" else 0.0
        is_tel = 1.0 if organ == "telemetry" else 0.0
        is_alert = 1.0 if ritual == "alert" else 0.0
        is_fs_event = 1.0 if ritual == "fs_event" else 0.0
        return [length / 1000.0, has_error, has_delete, is_fs, is_tel, is_alert, is_fs_event, 1.0]

    def infer(self, msg):
        if self.model is None:
            return None
        x = torch.tensor([self.features_from_msg(msg)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.model(x).item()
        return y

    def train_step(self, msg, label):
        if self.model is None:
            return
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        x = torch.tensor([self.features_from_msg(msg)], dtype=torch.float32, device=self.device)
        y = torch.tensor([[label]], dtype=torch.float32, device=self.device)
        pred = self.model(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.save(self.model.state_dict(), self.path)

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
# + BEHAVIOR LEARNING
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
        "role": "guardian"
    }
    return msg, alert_msg

def planner_agent(node, msg):
    if msg.get("intent") != "create_task":
        return None
    task_id = f"{node.node_id}-{int(now()*1000)}"
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
        "task_id": task_id
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
        "task_id": msg.get("task_id", "")
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
        "task_id": msg.get("task_id", "")
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
        if "contains" in cond and cond["contains"].lower() not in msg.get("payload", "").lower():
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
            actions.append({
                "from": node.node_id,
                "payload": payload,
                "priority": act.get("priority", "normal"),
                "organ": act.get("organ", "planner"),
                "lane": act.get("lane", "tasks"),
                "ritual": act.get("ritual", "task_create"),
                "intent": intent,
                "role": act.get("role", "planner"),
                "target_node": target_node
            })
    return actions

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
        self.log(f"Constraint added: organ={organ} lane={lane} ritual={ritual} -> {priority}")

    def get_state(self):
        return get_state_summary(self.node_id)

    def get_topology(self):
        return {"node_id": self.node_id, "neighbors": get_neighbors(self.node_id, include_unhealthy=True)}

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
                "ritual": "ping"
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
        data = json.dumps({"from_id": self.node_id, "enc": enc, "sig": sig}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
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
                "ritual": "sample"
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
                    "ritual": "fs_event"
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
                    "ritual": "window_change"
                }
                n = choose_neighbor_rl(self.node_id)
                if n:
                    self.send_to_neighbor(n, msg)
            time.sleep(2)

    def process_loop(self):
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
                            http_post(host, port, "/message", {"from_id": self.node_id, "enc": enc, "sig": sig})
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
                            http_post(host, port, "/message", {"from_id": self.node_id, "enc": enc, "sig": sig})
                            self.log(f"[AUTO] rule -> {target_node}: {m}")
                        except Exception as e:
                            self.log(f"[AUTO ERROR] {e}")

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
# HTTP HELPERS (GUI & INTERNAL)
# =========================================================

def http_get(host, port, path):
    url = f"http://{host}:{port}{path}"
    with urllib.request.urlopen(url, timeout=1) as r:
        return json.loads(r.read().decode("utf-8"))

def http_post(host, port, path, obj):
    url = f"http://{host}:{port}{path}"
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1) as r:
        return json.loads(r.read().decode("utf-8"))

# =========================================================
# GUI
# =========================================================

class SwarmGUI:
    def __init__(self, root, node_procs):
        self.root = root
        self.node_procs = node_procs
        root.title("Mythic Distributed Brain / Zero-Trust Mesh / RL Swarm (A–G)")

        self.text = tk.Text(root, width=120, height=20)
        self.text.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        ttk.Label(root, text="From Node").grid(row=1, column=0, sticky="w")
        ttk.Label(root, text="To Node").grid(row=1, column=1, sticky="w")
        ttk.Label(root, text="Payload").grid(row=1, column=2, sticky="w")

        self.from_entry = ttk.Entry(root, width=5)
        self.to_entry = ttk.Entry(root, width=5)
        self.payload_entry = ttk.Entry(root, width=40)

        self.from_entry.grid(row=2, column=0, padx=5)
        self.to_entry.grid(row=2, column=1, padx=5)
        self.payload_entry.grid(row=2, column=2, padx=5)

        self.priority_var = tk.StringVar(value="auto")
        ttk.Label(root, text="Priority").grid(row=1, column=3, sticky="w")
        ttk.OptionMenu(root, self.priority_var, "auto", "auto", "normal", "sidestep", "jump").grid(row=2, column=3, padx=5)

        ttk.Label(root, text="Organ").grid(row=3, column=0, sticky="w")
        ttk.Label(root, text="Lane").grid(row=3, column=1, sticky="w")
        ttk.Label(root, text="Ritual").grid(row=3, column=2, sticky="w")

        self.organ_entry = ttk.Entry(root, width=10)
        self.lane_entry = ttk.Entry(root, width=10)
        self.ritual_entry = ttk.Entry(root, width=10)

        self.organ_entry.grid(row=4, column=0, padx=5)
        self.lane_entry.grid(row=4, column=1, padx=5)
        self.ritual_entry.grid(row=4, column=2, padx=5)

        ttk.Label(root, text="Intent").grid(row=3, column=3, sticky="w")
        self.intent_entry = ttk.Entry(root, width=10)
        self.intent_entry.grid(row=4, column=3, padx=5)

        ttk.Label(root, text="Role").grid(row=5, column=0, sticky="w")
        self.role_entry = ttk.Entry(root, width=10)
        self.role_entry.grid(row=5, column=1, padx=5)

        ttk.Button(root, text="Send", command=self.send_message).grid(row=5, column=3, pady=5)

        ttk.Button(root, text="Add Neighbor", command=self.add_neighbor).grid(row=6, column=0, pady=5)
        ttk.Button(root, text="Refresh State", command=self.refresh_state).grid(row=6, column=1, pady=5)
        ttk.Button(root, text="Add Constraint", command=self.add_constraint).grid(row=6, column=2, pady=5)

        self.canvas = tk.Canvas(root, width=450, height=260, bg="black")
        self.canvas.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

        ttk.Label(root, text="Query Limit").grid(row=7, column=2, sticky="w")
        self.query_limit = ttk.Entry(root, width=6)
        self.query_limit.insert(0, "50")
        self.query_limit.grid(row=7, column=3, sticky="w")

        ttk.Label(root, text="Filter Priority/Tag").grid(row=8, column=2, sticky="w")
        self.query_priority = ttk.Entry(root, width=10)
        self.query_priority.grid(row=8, column=3, sticky="w")

        ttk.Label(root, text="Filter Tag (organ/lane/intent/role)").grid(row=9, column=2, sticky="w")
        self.query_tag = ttk.Entry(root, width=10)
        self.query_tag.grid(row=9, column=3, sticky="w")

        ttk.Button(root, text="Run Swarm Query", command=self.run_query_swarm).grid(row=10, column=2, columnspan=2, pady=5)

        ttk.Button(root, text="Seed Automation Rules", command=self.seed_rules).grid(row=10, column=0, pady=5)

        self.running = True
        threading.Thread(target=self.topology_loop, daemon=True).start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, msg):
        self.text.insert(tk.END, msg + "\n")
        self.text.see(tk.END)

    def send_message(self):
        from_id = self.from_entry.get().strip()
        to_id = self.to_entry.get().strip()
        payload = self.payload_entry.get().strip()
        organ = self.organ_entry.get().strip()
        lane = self.lane_entry.get().strip()
        ritual = self.ritual_entry.get().strip()
        prio = self.priority_var.get()
        intent = self.intent_entry.get().strip()
        role = self.role_entry.get().strip()

        if not from_id or not to_id or not payload:
            self.log("[GUI] Missing from/to/payload")
            return
        if to_id not in NODES:
            self.log(f"[GUI] Unknown target node {to_id}")
            return

        host, port = NODES[to_id]
        msg = {
            "from": from_id,
            "payload": payload,
            "priority": prio,
            "organ": organ,
            "lane": lane,
            "ritual": ritual,
            "intent": intent,
            "role": role
        }
        enc = encrypt_payload_for(to_id, msg)
        sig = sign_message(from_id, enc)
        try:
            http_post(host, port, "/message", {"from_id": from_id, "enc": enc, "sig": sig})
            self.log(f"[GUI] {from_id} -> {to_id} ({prio}) [{intent}/{role}]: {payload}")
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def add_neighbor(self):
        from_id = self.from_entry.get().strip()
        to_id = self.to_entry.get().strip()
        if from_id not in NODES or to_id not in NODES:
            self.log("[GUI] Unknown node id")
            return
        host_from, port_from = NODES[from_id]
        host_to, port_to = NODES[to_id]
        try:
            http_post(host_from, port_from, "/neighbor",
                      {"id": to_id, "host": host_to, "port": port_to})
            self.log(f"[GUI] Neighbor added: {from_id} -> {to_id}")
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def add_constraint(self):
        node_id = self.from_entry.get().strip()
        if node_id not in NODES:
            self.log("[GUI] Unknown node id for constraint")
            return
        host, port = NODES[node_id]
        organ = self.organ_entry.get().strip() or None
        lane = self.lane_entry.get().strip() or None
        ritual = self.ritual_entry.get().strip() or None
        priority = self.priority_var.get()
        try:
            http_post(host, port, "/constraint",
                      {"organ": organ, "lane": lane, "ritual": ritual, "priority": priority})
            self.log(f"[GUI] Constraint added on {node_id}: {organ}/{lane}/{ritual} -> {priority}")
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def refresh_state(self):
        for nid, (host, port) in NODES.items():
            try:
                state = http_get(host, port, "/state")
                self.log(f"[STATE] {nid}: in={state['in_count']} out={state['out_count']} neighbors={state['neighbors']}")
            except Exception as e:
                self.log(f"[STATE ERROR] {nid}: {e}")

    def topology_loop(self):
        while self.running:
            self.draw_topology()
            time.sleep(2)

    def draw_topology(self):
        self.canvas.delete("all")
        cx, cy, r = 220, 130, 100
        ids = list(NODES.keys())
        positions = {}
        n = len(ids)
        node_states = {}
        for nid, (host, port) in NODES.items():
            try:
                state = http_get(host, port, "/state")
                node_states[nid] = state
            except Exception:
                node_states[nid] = None

        for i, nid in enumerate(ids):
            angle = 2 * math.pi * i / n
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            positions[nid] = (x, y)
            state = node_states.get(nid)
            if state is None:
                color = "red"
            else:
                in_c = state["in_count"]
                out_c = state["out_count"]
                if in_c + out_c > 200:
                    color = "lime"
                elif in_c + out_c > 50:
                    color = "yellow"
                else:
                    color = "darkorange"
            self.canvas.create_oval(x-15, y-15, x+15, y+15, fill=color)
            self.canvas.create_text(x, y, text=nid, fill="white")

        for nid, (host, port) in NODES.items():
            try:
                topo = http_get(host, port, "/topology")
                neighbors = topo.get("neighbors", [])
            except Exception:
                neighbors = []
            x1, y1 = positions[nid]
            for nb in neighbors:
                nid2 = nb["id"]
                if nid2 in positions:
                    x2, y2 = positions[nid2]
                    color = "cyan" if nb["healthy"] else "gray"
                    self.canvas.create_line(x1, y1, x2, y2, fill=color)

    def run_query_swarm(self):
        try:
            limit = int(self.query_limit.get().strip() or "50")
        except ValueError:
            limit = 50
        priority = self.query_priority.get().strip() or None
        tag = self.query_tag.get().strip() or None

        results = query_swarm(limit=limit, priority=priority, tag=tag)
        self.log(f"[SWARM QUERY] {len(results)} results")
        for r in results:
            self.log(f"[{r['node']} {r['direction']}] {r['peer']} | {r['priority']} | "
                     f"{r['organ']}/{r['lane']}/{r['ritual']} | {r['intent']}/{r['role']} | {r['payload']}")

    def seed_rules(self):
        add_automation_rule(
            "High CPU -> Guardian Task",
            {"organ": "telemetry", "contains": "cpu="},
            {
                "target_node": "A",
                "intent": "create_task",
                "payload": "Investigate high CPU",
                "priority": "sidestep",
                "organ": "planner",
                "lane": "tasks",
                "ritual": "task_create",
                "role": "planner"
            },
            enabled=True
        )
        add_automation_rule(
            "Filesystem delete -> Guardian Task",
            {"organ": "filesystem", "contains": "delete"},
            {
                "target_node": "B",
                "intent": "create_task",
                "payload": "Review filesystem delete",
                "priority": "sidestep",
                "organ": "planner",
                "lane": "tasks",
                "ritual": "task_create",
                "role": "planner"
            },
            enabled=True
        )
        self.log("[GUI] Seeded automation rules into swarm DB")

    def on_close(self):
        self.running = False
        for p in self.node_procs:
            try:
                p.terminate()
            except Exception:
                pass
        self.root.destroy()

# =========================================================
# ENTRYPOINTS
# =========================================================

def run_node_process(node_id, port):
    node = NodeServer(node_id, port)
    node.start()

def spawn_nodes():
    procs = []
    script = os.path.abspath(sys.argv[0])
    for nid, (_, port) in NODES.items():
        cmd = [sys.executable, script, "node", nid, str(port)]
        p = subprocess.Popen(cmd)
        procs.append(p)
    return procs

def run_gui():
    if tk is None or ttk is None:
        print("Tkinter not available")
        return
    init_swarm_db()
    node_procs = spawn_nodes()
    time.sleep(1.5)
    root = tk.Tk()
    gui = SwarmGUI(root, node_procs)
    root.mainloop()

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "node":
        if len(sys.argv) < 4:
            print("Usage: python mythic_swarm.py node <ID> <PORT>")
            sys.exit(1)
        node_id = sys.argv[2]
        port = int(sys.argv[3])
        run_node_process(node_id, port)
    else:
        run_gui()

if __name__ == "__main__":
    main()

