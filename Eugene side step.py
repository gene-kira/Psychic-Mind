import os
import time
import math
import queue
import random
import threading
import sqlite3
import tkinter as tk
from tkinter import ttk

# Optional imports for sensors
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

# =========================================================
# CONFIG
# =========================================================

NODE_IDS = ["A", "B", "C"]
DB_DIR = "sidestep_dbs"
os.makedirs(DB_DIR, exist_ok=True)

WATCH_DIR = os.path.abspath(".")  # directory to watch for FS events

# =========================================================
# UIAUTOMATION ADAPTER
# =========================================================

class UIAutomationAdapter:
    def trigger(self, node_id, msg):
        ritual = msg.get("ritual", "")
        organ = msg.get("organ", "")
        lane = msg.get("lane", "")
        payload = msg.get("payload", "")
        print(f"[UIA] Node {node_id} ritual={ritual} organ={organ} lane={lane} payload={payload}")

UIA = UIAutomationAdapter()

# =========================================================
# PERSISTENCE + LEARNING
# =========================================================

def db_path(node_id):
    return os.path.join(DB_DIR, f"sidestep_node_{node_id}.db")

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
            neighbor_id TEXT
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
        time.time()
    ))
    conn.commit()
    conn.close()

def add_neighbor(node_id, neighbor_id):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("INSERT INTO neighbors(neighbor_id) VALUES (?)", (neighbor_id,))
    conn.commit()
    conn.close()

def get_neighbors(node_id):
    conn = sqlite3.connect(db_path(node_id))
    c = conn.cursor()
    c.execute("SELECT neighbor_id FROM neighbors")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]

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

def query_messages_all(nodes, limit=50, priority=None, tag=None):
    results = []
    for nid in nodes:
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
# NODE + SWARM
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

class Node(threading.Thread):
    def __init__(self, node_id, swarm, gui_callback=None):
        super().__init__(daemon=True)
        self.node_id = node_id
        self.swarm = swarm
        self.engine = SidestepEngine()
        self.inbox = queue.Queue()
        self.gui_callback = gui_callback
        self.running = True
        init_db(node_id)

    def log(self, msg):
        if self.gui_callback:
            self.gui_callback(f"[Node {self.node_id}] {msg}")

    def receive_external(self, msg):
        issues = detect_missing_details(msg)
        if issues:
            self.log(f"Missing details: {issues}")
        msg = learn_priority_from_history(self.node_id, msg)
        self.engine.receive(msg)

    def autonomous_behavior(self):
        neighbors = get_neighbors(self.node_id)
        for n in neighbors:
            payload = f"ping from {self.node_id}"
            msg = {
                "from": self.node_id,
                "to": n,
                "payload": payload,
                "priority": "auto",
                "organ": "heartbeat",
                "lane": "control",
                "ritual": "ping"
            }
            msg = learn_priority_from_history(self.node_id, msg)
            self.swarm.send(self.node_id, n, msg)

    def run(self):
        while self.running:
            while not self.inbox.empty():
                msg = self.inbox.get()
                self.receive_external(msg)
            kind, msg = self.engine.process_one()
            if msg:
                peer = msg.get("from", "?")
                log_message(self.node_id, "in", peer, msg)
                self.log(f"{kind}-processed: {msg}")
                UIA.trigger(self.node_id, msg)
            if time.time() % 5 < 0.2:
                self.autonomous_behavior()
            time.sleep(0.1)

class Swarm:
    def __init__(self, node_ids, gui_callback=None):
        self.nodes = {}
        self.gui_callback = gui_callback
        for nid in node_ids:
            node = Node(nid, self, gui_callback)
            self.nodes[nid] = node
            node.start()

    def send(self, from_id, to_id, msg):
        if to_id not in self.nodes:
            if self.gui_callback:
                self.gui_callback(f"[Swarm] Unknown node {to_id}")
            return
        msg = dict(msg)
        msg["from"] = from_id
        self.nodes[to_id].inbox.put(msg)
        log_message(from_id, "out", to_id, msg)
        if self.gui_callback:
            self.gui_callback(f"[Swarm] {from_id} -> {to_id} ({msg.get('priority')}): {msg.get('payload')}")

# =========================================================
# LIVE DATA SENSORS
# =========================================================

class SystemTelemetrySensor(threading.Thread):
    def __init__(self, swarm, node_id="A", interval=2.0, gui_callback=None):
        super().__init__(daemon=True)
        self.swarm = swarm
        self.node_id = node_id
        self.interval = interval
        self.gui_callback = gui_callback
        self.running = True
        self.enabled = psutil is not None
        if not self.enabled and self.gui_callback:
            self.gui_callback("[Sensor:System] psutil not available, sensor disabled")

    def run(self):
        if not self.enabled:
            return
        while self.running:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            payload = f"cpu={cpu:.1f} mem={mem:.1f}"
            msg = {
                "payload": payload,
                "priority": "auto",
                "organ": "telemetry",
                "lane": "system",
                "ritual": "sample"
            }
            self.swarm.send(self.node_id, random.choice(list(self.swarm.nodes.keys())), msg)
            time.sleep(self.interval)

class FSHandler(FileSystemEventHandler):
    def __init__(self, sensor):
        super().__init__()
        self.sensor = sensor

    def on_any_event(self, event):
        self.sensor.handle_event(event)

class FileSystemSensor(threading.Thread):
    def __init__(self, swarm, node_id="B", gui_callback=None):
        super().__init__(daemon=True)
        self.swarm = swarm
        self.node_id = node_id
        self.gui_callback = gui_callback
        self.running = True
        self.enabled = Observer is not None
        if not self.enabled and self.gui_callback:
            self.gui_callback("[Sensor:FS] watchdog not available, sensor disabled")

    def handle_event(self, event):
        payload = f"{event.event_type}: {event.src_path}"
        msg = {
            "payload": payload,
            "priority": "auto",
            "organ": "filesystem",
            "lane": "io",
            "ritual": "fs_event"
        }
        self.swarm.send(self.node_id, random.choice(list(self.swarm.nodes.keys())), msg)

    def run(self):
        if not self.enabled:
            return
        observer = Observer()
        handler = FSHandler(self)
        observer.schedule(handler, WATCH_DIR, recursive=True)
        observer.start()
        if self.gui_callback:
            self.gui_callback(f"[Sensor:FS] Watching {WATCH_DIR}")
        try:
            while self.running:
                time.sleep(1)
        finally:
            observer.stop()
            observer.join()

class BrowserSensor(threading.Thread):
    def __init__(self, swarm, node_id="A", gui_callback=None):
        super().__init__(daemon=True)
        self.swarm = swarm
        self.node_id = node_id
        self.gui_callback = gui_callback
        self.running = True

    def run(self):
        urls = [
            "https://example.com",
            "https://news.ycombinator.com",
            "https://github.com",
            "https://docs.python.org"
        ]
        while self.running:
            url = random.choice(urls)
            payload = f"visit {url}"
            msg = {
                "payload": payload,
                "priority": "auto",
                "organ": "browser",
                "lane": "activity",
                "ritual": "url_visit"
            }
            self.swarm.send(self.node_id, random.choice(list(self.swarm.nodes.keys())), msg)
            time.sleep(5)

class GameSensor(threading.Thread):
    def __init__(self, swarm, node_id="B", gui_callback=None):
        super().__init__(daemon=True)
        self.swarm = swarm
        self.node_id = node_id
        self.gui_callback = gui_callback
        self.running = True

    def run(self):
        while self.running:
            hp = random.randint(0, 100)
            ammo = random.randint(0, 200)
            payload = f"hp={hp} ammo={ammo}"
            msg = {
                "payload": payload,
                "priority": "auto",
                "organ": "game",
                "lane": "telemetry",
                "ritual": "tick"
            }
            self.swarm.send(self.node_id, random.choice(list(self.swarm.nodes.keys())), msg)
            time.sleep(3)

class ExternalAPISensor(threading.Thread):
    def __init__(self, swarm, node_id="C", gui_callback=None):
        super().__init__(daemon=True)
        self.swarm = swarm
        self.node_id = node_id
        self.gui_callback = gui_callback
        self.running = True

    def run(self):
        while self.running:
            value = random.uniform(-1, 1)
            payload = f"signal={value:.3f}"
            msg = {
                "payload": payload,
                "priority": "auto",
                "organ": "external_api",
                "lane": "signal",
                "ritual": "sample"
            }
            self.swarm.send(self.node_id, random.choice(list(self.swarm.nodes.keys())), msg)
            time.sleep(4)

class UIAutomationSensor(threading.Thread):
    def __init__(self, swarm, node_id="C", gui_callback=None):
        super().__init__(daemon=True)
        self.swarm = swarm
        self.node_id = node_id
        self.gui_callback = gui_callback
        self.running = True
        self.enabled = ctypes is not None and os.name == "nt"
        if not self.enabled and self.gui_callback:
            self.gui_callback("[Sensor:UIA] ctypes/Windows not available, sensor disabled")

    def get_foreground_title(self):
        if not self.enabled:
            return None
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        return buff.value

    def run(self):
        if not self.enabled:
            return
        last_title = None
        while self.running:
            title = self.get_foreground_title()
            if title and title != last_title:
                last_title = title
                payload = f"foreground={title}"
                msg = {
                    "payload": payload,
                    "priority": "auto",
                    "organ": "uiautomation",
                    "lane": "focus",
                    "ritual": "window_change"
                }
                self.swarm.send(self.node_id, random.choice(list(self.swarm.nodes.keys())), msg)
            time.sleep(2)

# =========================================================
# GUI
# =========================================================

class SwarmGUI:
    def __init__(self, root):
        self.root = root
        root.title("Mythic Federated Sidestep Swarm (Live)")

        self.swarm = Swarm(NODE_IDS, gui_callback=self.log)

        # Sensors
        self.sensors = []
        self.sensors.append(SystemTelemetrySensor(self.swarm, node_id="A", gui_callback=self.log))
        self.sensors.append(FileSystemSensor(self.swarm, node_id="B", gui_callback=self.log))
        self.sensors.append(BrowserSensor(self.swarm, node_id="A", gui_callback=self.log))
        self.sensors.append(GameSensor(self.swarm, node_id="B", gui_callback=self.log))
        self.sensors.append(ExternalAPISensor(self.swarm, node_id="C", gui_callback=self.log))
        self.sensors.append(UIAutomationSensor(self.swarm, node_id="C", gui_callback=self.log))
        for s in self.sensors:
            s.start()

        # Log area
        self.text = tk.Text(root, width=110, height=18)
        self.text.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Send panel
        ttk.Label(root, text="From").grid(row=1, column=0, sticky="w")
        ttk.Label(root, text="To").grid(row=1, column=1, sticky="w")
        ttk.Label(root, text="Payload").grid(row=1, column=2, sticky="w")

        self.from_entry = ttk.Entry(root, width=5)
        self.to_entry = ttk.Entry(root, width=5)
        self.payload_entry = ttk.Entry(root, width=30)

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

        # Neighbor + state buttons
        ttk.Button(root, text="Add Neighbor", command=self.add_neighbor).grid(row=5, column=0, pady=5)
        ttk.Button(root, text="Refresh State", command=self.refresh_state).grid(row=5, column=1, pady=5)

        # Topology canvas
        self.canvas = tk.Canvas(root, width=400, height=260, bg="black")
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
        if from_id not in self.swarm.nodes or to_id not in self.swarm.nodes:
            self.log("[GUI] Unknown node id")
            return

        msg = {
            "payload": payload,
            "priority": prio,
            "organ": organ,
            "lane": lane,
            "ritual": ritual
        }
        self.swarm.send(from_id, to_id, msg)

    def add_neighbor(self):
        from_id = self.from_entry.get().strip()
        to_id = self.to_entry.get().strip()
        if from_id not in self.swarm.nodes or to_id not in self.swarm.nodes:
            self.log("[GUI] Unknown node id")
            return
        add_neighbor(from_id, to_id)
        self.log(f"[GUI] Neighbor added: {from_id} -> {to_id}")

    def refresh_state(self):
        for nid in NODE_IDS:
            state = get_state_summary(nid)
            self.log(f"[STATE] {nid}: in={state['in_count']} out={state['out_count']} neighbors={state['neighbors']}")

    def topology_loop(self):
        while self.running:
            self.draw_topology()
            time.sleep(2)

    def draw_topology(self):
        self.canvas.delete("all")
        cx, cy, r = 200, 130, 90
        positions = {}
        n = len(NODE_IDS)
        for i, nid in enumerate(NODE_IDS):
            angle = 2 * math.pi * i / n
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            positions[nid] = (x, y)
            self.canvas.create_oval(x-15, y-15, x+15, y+15, fill="darkorange")
            self.canvas.create_text(x, y, text=nid, fill="white")

        for nid in NODE_IDS:
            neighbors = get_neighbors(nid)
            x1, y1 = positions[nid]
            for nb in neighbors:
                if nb in positions:
                    x2, y2 = positions[nb]
                    self.canvas.create_line(x1, y1, x2, y2, fill="cyan")

    def run_query(self):
        try:
            limit = int(self.query_limit.get().strip() or "50")
        except ValueError:
            limit = 50
        priority = self.query_priority.get().strip() or None
        tag = self.query_tag.get().strip() or None

        results = query_messages_all(NODE_IDS, limit=limit, priority=priority, tag=tag)
        self.log(f"[QUERY] {len(results)} results")
        for r in results:
            self.log(f"[{r['node']} {r['direction']}] {r['peer']} | {r['priority']} | "
                     f"{r['organ']}/{r['lane']}/{r['ritual']} | {r['payload']}")

def main():
    root = tk.Tk()
    gui = SwarmGUI(root)
    root.mainloop()
    gui.running = False
    for s in gui.sensors:
        s.running = False

if __name__ == "__main__":
    main()

