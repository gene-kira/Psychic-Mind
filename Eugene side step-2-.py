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

# =========================================================
# CONFIG
# =========================================================

DB_DIR = "sidestep_dbs"
os.makedirs(DB_DIR, exist_ok=True)

# 7-node swarm
NODES = {
    "A": ("localhost", 8001),
    "B": ("localhost", 8002),
    "C": ("localhost", 8003),
    "D": ("localhost", 8004),
    "E": ("localhost", 8005),
    "F": ("localhost", 8006),
    "G": ("localhost", 8007),
}

DEFAULT_ORGANS = ["telemetry", "filesystem", "browser", "game", "external_api", "uiautomation", "heartbeat"]
DEFAULT_LANES = ["system", "io", "activity", "telemetry", "signal", "focus", "control"]
DEFAULT_RITUALS = ["sample", "fs_event", "url_visit", "tick", "ping", "window_change", "alert"]

# =========================================================
# UTILS / DB
# =========================================================

def db_path(node_id):
    return os.path.join(DB_DIR, f"sidestep_node_{node_id}.db")

def now():
    return time.time()

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
            ts REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS neighbors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            neighbor_id TEXT,
            host TEXT,
            port INTEGER
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
    conn.commit()
    conn.close()

def log_message(node_id, direction, peer, msg):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("""
        INSERT INTO messages(direction, peer, payload, priority, organ, lane, ritual, ts)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        direction,
        peer,
        msg.get("payload", ""),
        msg.get("priority", "normal"),
        msg.get("organ", ""),
        msg.get("lane", ""),
        msg.get("ritual", ""),
        now()
    ))
    conn.commit()
    conn.close()

def add_neighbor(node_id, neighbor_id, host, port):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("INSERT INTO neighbors(neighbor_id, host, port) VALUES (?,?,?)",
              (neighbor_id, host, port))
    conn.commit()
    conn.close()

def get_neighbors(node_id):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT neighbor_id, host, port FROM neighbors")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "host": r[1], "port": r[2]} for r in rows]

def get_state_summary(node_id):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE direction='in'")
    in_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM messages WHERE direction='out'")
    out_count = c.fetchone()[0]
    c.execute("SELECT payload, priority, organ, lane, ritual, ts FROM messages ORDER BY id DESC LIMIT 5")
    last_msgs = [{
        "payload": r[0],
        "priority": r[1],
        "organ": r[2],
        "lane": r[3],
        "ritual": r[4],
        "ts": r[5]
    } for r in c.fetchall()]
    conn.close()
    return {
        "node_id": node_id,
        "in_count": in_count,
        "out_count": out_count,
        "last_messages": last_msgs,
        "neighbors": get_neighbors(node_id)
    }

def query_messages_all(node_ids, limit=50, priority=None, tag=None):
    results = []
    for nid in node_ids:
        conn = sqlite3.connect(db_path(nid))
        c = conn.cursor()
        base = "SELECT ?, direction, peer, payload, priority, organ, lane, ritual, ts FROM messages"
        params = [nid]
        where = []
        if priority:
            where.append("priority = ?")
            params.append(priority)
        if tag:
            where.append("(organ = ? OR lane = ? OR ritual = ?)")
            params.extend([tag, tag, tag])
        if where:
            base += " WHERE " + " AND ".join(where)
        base += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        c.execute(base, params)
        rows = c.fetchall()
        conn.close()
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
                "ts": r[8]
            })
    results.sort(key=lambda x: x["ts"], reverse=True)
    return results[:limit]

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
# NODE HTTP SERVER
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
            self.server.node.receive_message(payload)
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
        self.engine = SidestepEngine()
        self.running = True
        self.server = HTTPServer(("0.0.0.0", port), NodeHTTPHandler)
        self.server.node = self

    def log(self, msg):
        print(f"[Node {self.node_id}] {msg}")

    def receive_message(self, msg):
        issues = detect_missing_details(msg)
        if issues:
            self.log(f"Missing details: {issues}")
        msg = apply_constraints(self.node_id, msg)
        msg = learn_priority_from_history(self.node_id, msg)
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
        return {"node_id": self.node_id, "neighbors": get_neighbors(self.node_id)}

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

    def send_to_neighbor(self, neighbor, msg):
        host = neighbor["host"]
        port = neighbor["port"]
        nid = neighbor["id"]
        url = f"http://{host}:{port}/message"
        data = json.dumps(msg).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=1)
            log_message(self.node_id, "out", nid, msg)
            self.log(f"-> {nid} ({msg.get('priority')}): {msg.get('payload')}")
        except Exception as e:
            self.log(f"Send error to {nid}: {e}")

    def process_loop(self):
        while self.running:
            kind, msg = self.engine.process_one()
            if msg:
                peer = msg.get("from", "?")
                log_message(self.node_id, "in", peer, msg)
                self.log(f"{kind}-processed: {msg}")
            if now() % 5 < 0.2:
                self.autonomous_behavior()
            time.sleep(0.1)

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
            neighbors = get_neighbors(self.node_id)
            if neighbors:
                target = random.choice(neighbors)
                self.send_to_neighbor(target, msg)
            time.sleep(3)

    def start(self):
        self.log(f"Listening on port {self.port}")
        t = threading.Thread(target=self.process_loop, daemon=True)
        t.start()
        s = threading.Thread(target=self.telemetry_sensor_loop, daemon=True)
        s.start()
        self.server.serve_forever()

# =========================================================
# HTTP HELPERS (GUI SIDE)
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
        root.title("Mythic Federated Sidestep Swarm (Distributed A–G)")

        self.text = tk.Text(root, width=120, height=20)
        self.text.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Send panel
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

        # Tags
        ttk.Label(root, text="Organ").grid(row=3, column=0, sticky="w")
        ttk.Label(root, text="Lane").grid(row=3, column=1, sticky="w")
        ttk.Label(root, text="Ritual").grid(row=3, column=2, sticky="w")

        self.organ_entry = ttk.Entry(root, width=10)
        self.lane_entry = ttk.Entry(root, width=10)
        self.ritual_entry = ttk.Entry(root, width=10)

        self.organ_entry.grid(row=4, column=0, padx=5)
        self.lane_entry.grid(row=4, column=1, padx=5)
        self.ritual_entry.grid(row=4, column=2, padx=5)

        ttk.Button(root, text="Send", command=self.send_message).grid(row=4, column=3, pady=5)

        # Neighbor + state + constraints
        ttk.Button(root, text="Add Neighbor", command=self.add_neighbor).grid(row=5, column=0, pady=5)
        ttk.Button(root, text="Refresh State", command=self.refresh_state).grid(row=5, column=1, pady=5)
        ttk.Button(root, text="Add Constraint", command=self.add_constraint).grid(row=5, column=2, pady=5)

        # Topology canvas
        self.canvas = tk.Canvas(root, width=450, height=260, bg="black")
        self.canvas.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        # Query panel
        ttk.Label(root, text="Query Limit").grid(row=6, column=2, sticky="w")
        self.query_limit = ttk.Entry(root, width=6)
        self.query_limit.insert(0, "50")
        self.query_limit.grid(row=6, column=3, sticky="w")

        ttk.Label(root, text="Filter Priority").grid(row=7, column=2, sticky="w")
        self.query_priority = ttk.Entry(root, width=10)
        self.query_priority.grid(row=7, column=3, sticky="w")

        ttk.Label(root, text="Filter Tag").grid(row=8, column=2, sticky="w")
        self.query_tag = ttk.Entry(root, width=10)
        self.query_tag.grid(row=8, column=3, sticky="w")

        ttk.Button(root, text="Run Query", command=self.run_query).grid(row=9, column=2, columnspan=2, pady=5)

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
            "ritual": ritual
        }
        try:
            http_post(host, port, "/message", msg)
            self.log(f"[GUI] {from_id} -> {to_id} ({prio}): {payload}")
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
        for i, nid in enumerate(ids):
            angle = 2 * math.pi * i / n
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            positions[nid] = (x, y)
            self.canvas.create_oval(x-15, y-15, x+15, y+15, fill="darkorange")
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
                    self.canvas.create_line(x1, y1, x2, y2, fill="cyan")

    def run_query(self):
        try:
            limit = int(self.query_limit.get().strip() or "50")
        except ValueError:
            limit = 50
        priority = self.query_priority.get().strip() or None
        tag = self.query_tag.get().strip() or None

        results = query_messages_all(list(NODES.keys()), limit=limit, priority=priority, tag=tag)
        self.log(f"[QUERY] {len(results)} results")
        for r in results:
            self.log(f"[{r['node']} {r['direction']}] {r['peer']} | {r['priority']} | "
                     f"{r['organ']}/{r['lane']}/{r['ritual']} | {r['payload']}")

    def on_close(self):
        self.running = False
        # kill all node processes
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
        # spawn: python mythic_swarm.py node <id> <port>
        cmd = [sys.executable, script, "node", nid, str(port)]
        p = subprocess.Popen(cmd)  # console visible
        procs.append(p)
    return procs

def run_gui():
    if tk is None or ttk is None:
        print("Tkinter not available")
        return
    node_procs = spawn_nodes()
    # small delay to let nodes start
    time.sleep(1.5)
    root = tk.Tk()
    gui = SwarmGUI(root, node_procs)
    root.mainloop()

def main():
    # modes:
    #   no args          -> GUI
    #   node <id> <port> -> node server
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

