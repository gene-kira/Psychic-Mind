# =========================
# USET-X UNIFIED BRAIN ORGANISM (TKINTER EDITION)
# =========================
# - FastAPI backend (organism) in child process
# - Tkinter operator console in main process
# - Unified engine + RL brain + camera in backend
# - Dark theme, horizontal tabs, minimal dark plot
# - Auto port selection: try 8000, else random 1024–65535
# - Backend writes chosen port to usetx_port.txt
# - GUI reads usetx_port.txt and connects automatically
# =========================

import sys, subprocess, importlib, threading, time, io, random, multiprocessing, platform, os, socket

# -------------------------
# AUTO-INSTALLER
# -------------------------
def install(pkg, imp=None):
    try:
        return importlib.import_module(imp or pkg)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            return importlib.import_module(imp or pkg)
        except Exception as e:
            print(f"[AutoInstall] Failed to install {pkg}: {e}")
            return None

# -------------------------
# CORE LIBS
# -------------------------
np = install("numpy")
import hashlib, zlib
requests = install("requests")
io_mod = io

# -------------------------
# IMAGE + CAMERA
# -------------------------
PIL_Image = install("PIL.Image", "PIL.Image")
cv2 = install("opencv-python", "cv2")

# -------------------------
# AI + LEARNING
# -------------------------
torch = install("torch")
if torch is None:
    raise RuntimeError("torch is required.")
nn = torch.nn
optim = torch.optim

# -------------------------
# HARDWARE / META-STATE
# -------------------------
psutil = install("psutil")

# -------------------------
# DATABASE (OPTIONAL)
# -------------------------
pymongo = install("pymongo")
if pymongo is not None:
    from pymongo import MongoClient
else:
    MongoClient = None

# -------------------------
# WEB SERVER
# -------------------------
fastapi = install("fastapi")
uvicorn = install("uvicorn")
if fastapi is None or uvicorn is None:
    raise RuntimeError("fastapi and uvicorn are required.")
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# -------------------------
# TKINTER UI
# -------------------------
import tkinter as tk
from tkinter import ttk
matplotlib = install("matplotlib")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
NODE_PEERS = []  # can be filled with other node URLs
NODE_ID = f"{platform.node()}-{os.getpid()}"
SWARM_DISCOVERY_INTERVAL = 10
CAMERA_INTERVAL = 1.0
RL_UPDATE_INTERVAL = 5.0
PORT_FILE = "usetx_port.txt"

# =========================
# PORT SELECTION
# =========================
def find_free_port(preferred=8000):
    # Try preferred first
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", preferred))
            return preferred
        except OSError:
            pass
    # Else random in 1024–65535
    while True:
        port = random.randint(1024, 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue

# =========================
# DB WRAPPERS (SAFE)
# =========================
def get_db():
    if MongoClient is None:
        return None, None, None
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=500)
        client.admin.command("ping")
        db = client["usetx"]
        return db, db["states"], db["meta"]
    except Exception as e:
        print(f"[DB] Mongo unavailable: {e}")
        return None, None, None

db, collection_states, collection_meta = get_db()

def safe_insert(coll, doc):
    if coll is None:
        return
    try:
        coll.insert_one(doc)
    except Exception as e:
        print(f"[DB] insert failed: {e}")

def safe_find(coll, *args, **kwargs):
    if coll is None:
        return []
    try:
        return list(coll.find(*args, **kwargs))
    except Exception as e:
        print(f"[DB] find failed: {e}")
        return []

def safe_count(coll):
    if coll is None:
        return 0
    try:
        return coll.count_documents({})
    except Exception as e:
        print(f"[DB] count failed: {e}")
        return 0

def safe_delete_many(coll, *args, **kwargs):
    if coll is None:
        return
    try:
        coll.delete_many(*args, **kwargs)
    except Exception as e:
        print(f"[DB] delete_many failed: {e}")

# =========================
# MODELS
# =========================
class TransitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return self.fc3(h)

class NavigatorBrain(nn.Module):
    def __init__(self, state_dim=3, action_dim=3):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.gamma = 0.95
        self.eps = 0.1

    def act(self, state_vec):
        if random.random() < self.eps:
            return random.randint(0, 2)
        with torch.no_grad():
            q_vals = self.q_net(state_vec.unsqueeze(0))
        return int(q_vals.argmax().item())

    def update(self, batch):
        if not batch:
            return None
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_vals = self.q_net(states)
        q_a = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_vals = self.q_net(next_states)
            max_next_q = next_q_vals.max(1)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = ((q_a - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

# =========================
# ENGINE
# =========================
class USETX:
    def __init__(self):
        self.net = TransitionNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.rl_brain = NavigatorBrain()
        self.rl_buffer = []

    def detect_objects(self, bytes_data):
        # Keep it simple and stable: pseudo-objects
        return [random.randint(0, 5)]

    def encode(self, b):
        return int.from_bytes(hashlib.sha256(b).digest(), "big")

    def complexity(self, b):
        return len(zlib.compress(b))

    def build_state(self, b, meta=None):
        base = {
            "objects": self.detect_objects(b),
            "complexity": self.complexity(b),
            "encoding": self.encode(b),
        }
        if meta:
            base["meta"] = meta
        return base

    def to_vector(self, state):
        obj_sum = sum(state.get("objects", [])) if state.get("objects") else 0
        comp = state.get("complexity", 0) % 10000
        enc = state.get("encoding", 0) % 10000
        return torch.tensor([obj_sum, comp, enc], dtype=torch.float32)

    def transition(self, vec):
        return self.net(vec)

    def train_step(self, vec):
        target = torch.tanh(vec)
        pred = self.net(vec)
        loss = ((pred - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def simulate(self, state, steps=5):
        vec = self.to_vector(state)
        history = []
        for _ in range(steps):
            loss = self.train_step(vec)
            vec = self.transition(vec)
            history.append({"vector": vec.detach().tolist(), "loss": loss})
        return history

    def rl_step(self, state):
        vec = self.to_vector(state)
        action = self.rl_brain.act(vec)
        with torch.no_grad():
            if action == 1:
                next_vec = vec + torch.tensor([1.0, 5.0, 5.0])
            elif action == 2:
                next_vec = vec - torch.tensor([1.0, 5.0, 5.0])
            else:
                next_vec = vec.clone()
        reward = -next_vec[1].item()
        done = False
        self.rl_buffer.append((vec, action, reward, next_vec, done))
        if len(self.rl_buffer) > 64:
            self.rl_buffer.pop(0)
        return {"action": action, "reward": reward, "next_vec": next_vec.tolist()}

    def rl_update(self):
        if not self.rl_buffer:
            return None
        batch = random.sample(self.rl_buffer, min(16, len(self.rl_buffer)))
        return self.rl_brain.update(batch)

# =========================
# META-STATE
# =========================
def get_hardware_meta():
    try:
        cpu = psutil.cpu_percent(interval=None) if psutil else None
        mem = psutil.virtual_memory() if psutil else None
        disk = psutil.disk_usage("/") if psutil else None
        return {
            "node_id": NODE_ID,
            "platform": platform.platform(),
            "cpu_percent": cpu if cpu is not None else None,
            "mem_percent": mem.percent if mem else None,
            "disk_percent": disk.percent if disk else None,
            "timestamp": time.time(),
        }
    except Exception as e:
        print(f"[Meta] Hardware scan failed: {e}")
        return {
            "node_id": NODE_ID,
            "platform": platform.platform(),
            "cpu_percent": None,
            "mem_percent": None,
            "disk_percent": None,
            "timestamp": time.time(),
        }

# =========================
# ROLE-BASED AGENTS
# =========================
class ScoutAgent:
    def __init__(self, engine: USETX):
        self.engine = engine

    def observe_frame(self, frame_bytes, meta=None):
        state = self.engine.build_state(frame_bytes, meta=meta)
        evolution = self.engine.simulate(state)
        rl_info = self.engine.rl_step(state)
        record = {
            "role": "scout",
            "state": state,
            "evolution": evolution,
            "rl": rl_info,
            "timestamp": time.time(),
        }
        safe_insert(collection_states, record)
        return record

class AnalystAgent:
    def __init__(self, engine: USETX):
        self.engine = engine

    def analyze_recent(self, limit=50):
        docs = safe_find(collection_states, {}, {"_id": 0})
        docs = sorted(docs, key=lambda d: d.get("timestamp", 0), reverse=True)[:limit]
        if not docs:
            return {"count": 0, "avg_complexity": None, "anomaly_score": None}
        complexities = [d["state"]["complexity"] for d in docs if "state" in d]
        if not complexities:
            return {"count": len(docs), "avg_complexity": None, "anomaly_score": None}
        avg_complexity = float(np.mean(complexities))
        anomaly_score = float(np.std(complexities))
        summary = {
            "type": "analysis_summary",
            "count": len(docs),
            "avg_complexity": avg_complexity,
            "anomaly_score": anomaly_score,
            "timestamp": time.time(),
        }
        safe_insert(collection_meta, summary)
        return summary

class ArchivistAgent:
    def prune_old(self, max_docs=1000):
        total = safe_count(collection_states)
        if total <= max_docs:
            return {"pruned": 0, "total": total}
        to_prune = total - max_docs
        docs = safe_find(collection_states, {}, {"_id": 1, "timestamp": 1})
        docs = sorted(docs, key=lambda d: d.get("timestamp", 0))[:to_prune]
        ids = [d["_id"] for d in docs if "_id" in d]
        if ids:
            safe_delete_many(collection_states, {"_id": {"$in": ids}})
        return {"pruned": len(ids), "total": safe_count(collection_states)}

# =========================
# SWARM HEARTBEAT
# =========================
def swarm_heartbeat():
    meta = get_hardware_meta()
    meta["type"] = "heartbeat"
    safe_insert(collection_meta, meta)
    for peer in NODE_PEERS:
        try:
            requests.post(f"{peer}/heartbeat", json=meta, timeout=1)
        except Exception:
            pass

# =========================
# CAMERA THREAD (BACKEND)
# =========================
def camera_loop(engine: USETX, scout: ScoutAgent):
    if cv2 is None:
        print("[Camera] OpenCV not available, skipping camera.")
        return
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Camera] No camera available.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buf = cv2.imencode(".jpg", frame)
            data = buf.tobytes()
            meta = get_hardware_meta()
            scout.observe_frame(data, meta=meta)
            time.sleep(CAMERA_INTERVAL)
    except Exception as e:
        print(f"[Camera] Failed: {e}")

# =========================
# BACKEND SERVER (CHILD PROCESS)
# =========================
def run_server():
    app = FastAPI()
    engine = USETX()
    scout = ScoutAgent(engine)
    analyst = AnalystAgent(engine)
    archivist = ArchivistAgent()
    last_rl_loss = {"value": None}

    @app.get("/")
    def root():
        return {"status": "USET-X swarm node running", "node_id": NODE_ID}

    @app.post("/process")
    async def process(file: UploadFile = File(...)):
        data = await file.read()
        meta = get_hardware_meta()
        record = scout.observe_frame(data, meta=meta)
        for peer in NODE_PEERS:
            try:
                requests.post(f"{peer}/ingest", json=record, timeout=1)
            except Exception:
                pass
        return JSONResponse(record)

    @app.post("/ingest")
    async def ingest(data: dict):
        data["ingested_from_peer"] = True
        safe_insert(collection_states, data)
        return {"status": "received", "node_id": NODE_ID}

    @app.get("/states")
    def get_states():
        data = safe_find(collection_states, {}, {"_id": 0})
        data = sorted(data, key=lambda d: d.get("timestamp", 0))
        return {"count": len(data), "data": data}

    @app.get("/meta")
    def get_meta():
        meta_docs = safe_find(collection_meta, {}, {"_id": 0})
        meta_docs = sorted(meta_docs, key=lambda d: d.get("timestamp", 0), reverse=True)[:20]
        hw_meta = get_hardware_meta()
        return {
            "hardware": hw_meta,
            "meta_docs": meta_docs,
            "node_id": NODE_ID,
            "last_rl_loss": last_rl_loss["value"],
        }

    @app.post("/heartbeat")
    async def heartbeat(info: dict):
        info["type"] = "heartbeat"
        info["timestamp"] = time.time()
        safe_insert(collection_meta, info)
        return {"status": "ok"}

    @app.get("/analyze")
    def analyze():
        return analyst.analyze_recent()

    @app.post("/prune")
    def prune(max_docs: int = 1000):
        return archivist.prune_old(max_docs=max_docs)

    def rl_loop():
        while True:
            loss = engine.rl_update()
            if loss is not None:
                last_rl_loss["value"] = loss
            time.sleep(RL_UPDATE_INTERVAL)

    threading.Thread(target=rl_loop, daemon=True).start()

    def heartbeat_loop():
        while True:
            swarm_heartbeat()
            time.sleep(SWARM_DISCOVERY_INTERVAL)

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    threading.Thread(target=camera_loop, args=(engine, scout), daemon=True).start()

    port = find_free_port(8000)
    try:
        with open(PORT_FILE, "w") as f:
            f.write(str(port))
    except Exception as e:
        print(f"[Server] Failed to write port file: {e}")

    print(f"[Server] Starting on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

# =========================
# TKINTER OPERATOR CONSOLE (MAIN PROCESS)
# =========================
class USETXConsole(tk.Tk):
    def __init__(self, port):
        super().__init__()
        self.title("USET-X Swarm Operator Console")
        self.geometry("1100x700")
        self.configure(bg="#111111")
        self.port = port
        self.base_url = f"http://127.0.0.1:{self.port}"

        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TNotebook", background="#111111", borderwidth=0)
        style.configure("TNotebook.Tab", background="#222222", foreground="#00ff66", padding=(10, 5))
        style.map("TNotebook.Tab", background=[("selected", "#00aa55")])
        style.configure("TFrame", background="#111111")
        style.configure("TLabel", background="#111111", foreground="#00ff66")
        style.configure("TButton", background="#222222", foreground="#00ff66")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.tab_states = ttk.Frame(notebook)
        self.tab_evolution = ttk.Frame(notebook)
        self.tab_meta = ttk.Frame(notebook)
        self.tab_logs = ttk.Frame(notebook)
        self.tab_controls = ttk.Frame(notebook)

        notebook.add(self.tab_states, text="States")
        notebook.add(self.tab_evolution, text="Evolution")
        notebook.add(self.tab_meta, text="Meta-State")
        notebook.add(self.tab_logs, text="Logs")
        notebook.add(self.tab_controls, text="Controls")

        self.build_states_tab()
        self.build_evolution_tab()
        self.build_meta_tab()
        self.build_logs_tab()
        self.build_controls_tab()

        self.after(2000, self.refresh_all)

    # ----- States Tab -----
    def build_states_tab(self):
        self.states_list = tk.Listbox(self.tab_states, bg="#111111", fg="#00ff66")
        self.states_list.pack(fill="both", expand=True, padx=10, pady=10)

    # ----- Evolution Tab -----
    def build_evolution_tab(self):
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor("#111111")
        self.ax.set_facecolor("#111111")
        self.ax.tick_params(colors="#cccccc")
        self.ax.spines["bottom"].set_color("#cccccc")
        self.ax.spines["top"].set_color("#cccccc")
        self.ax.spines["left"].set_color("#cccccc")
        self.ax.spines["right"].set_color("#cccccc")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_evolution)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # ----- Meta Tab -----
    def build_meta_tab(self):
        self.meta_text = tk.Text(self.tab_meta, bg="#111111", fg="#00ff66")
        self.meta_text.pack(fill="both", expand=True, padx=10, pady=10)

    # ----- Logs Tab -----
    def build_logs_tab(self):
        self.logs_text = tk.Text(self.tab_logs, bg="#111111", fg="#00ff66")
        self.logs_text.pack(fill="both", expand=True, padx=10, pady=10)

    # ----- Controls Tab -----
    def build_controls_tab(self):
        frame = ttk.Frame(self.tab_controls)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        btn_refresh = ttk.Button(frame, text="Refresh Now", command=self.refresh_all)
        btn_refresh.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        btn_analyze = ttk.Button(frame, text="Run Analysis", command=self.run_analysis)
        btn_analyze.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        btn_prune = ttk.Button(frame, text="Prune (1000 max)", command=self.prune_data)
        btn_prune.grid(row=0, column=2, padx=5, pady=5, sticky="w")

    # ----- Refresh Logic -----
    def refresh_all(self):
        self.update_states()
        self.update_meta()
        self.after(2000, self.refresh_all)

    def update_states(self):
        if requests is None:
            return
        try:
            res = requests.get(f"{self.base_url}/states", timeout=1)
            payload = res.json()
            data = payload.get("data", [])
            self.states_list.delete(0, tk.END)
            for i, s in enumerate(data[-20:]):
                st = s.get("state", {})
                comp = st.get("complexity", None)
                objs = st.get("objects", [])
                role = s.get("role", "unknown")
                self.states_list.insert(
                    tk.END,
                    f"{i}: role={role}, complexity={comp}, objs={objs}"
                )
            if data:
                last = data[-1].get("evolution", [])
                self.plot_evolution(last)
        except Exception as e:
            self.log(f"[States] Failed: {e}")

    def plot_evolution(self, evolution):
        self.ax.clear()
        self.ax.set_facecolor("#111111")
        self.ax.tick_params(colors="#cccccc")
        self.ax.spines["bottom"].set_color("#cccccc")
        self.ax.spines["top"].set_color("#cccccc")
        self.ax.spines["left"].set_color("#cccccc")
        self.ax.spines["right"].set_color("#cccccc")
        if not evolution:
            self.ax.set_title("No evolution data", color="#cccccc")
            self.canvas.draw()
            return
        arr = np.array([e["vector"] for e in evolution])
        for i in range(arr.shape[1]):
            self.ax.plot(arr[:, i], label=f"dim {i}")
        self.ax.set_title("State Evolution", color="#cccccc")
        self.ax.legend(facecolor="#111111", edgecolor="#cccccc", labelcolor="#cccccc")
        self.canvas.draw()

    def update_meta(self):
        if requests is None:
            return
        try:
            res = requests.get(f"{self.base_url}/meta", timeout=1)
            meta = res.json()
            hw = meta.get("hardware", {})
            last_rl_loss = meta.get("last_rl_loss", None)
            lines = []
            lines.append(f"Node ID: {meta.get('node_id')}")
            lines.append("---- Hardware ----")
            lines.append(f"CPU: {hw.get('cpu_percent')}%")
            lines.append(f"Mem: {hw.get('mem_percent')}%")
            lines.append(f"Disk: {hw.get('disk_percent')}%")
            lines.append("---- RL ----")
            lines.append(f"Last RL loss: {last_rl_loss}")
            lines.append("---- Meta Docs (latest) ----")
            for d in meta.get("meta_docs", []):
                t = time.strftime("%H:%M:%S", time.localtime(d.get("timestamp", 0)))
                if d.get("type") == "analysis_summary":
                    lines.append(
                        f"[{t}] Analysis: count={d.get('count')}, "
                        f"avg_complexity={d.get('avg_complexity'):.2f}, "
                        f"anomaly={d.get('anomaly_score'):.2f}"
                    )
                elif d.get("type") == "heartbeat":
                    lines.append(
                        f"[{t}] Heartbeat from {d.get('node_id')} "
                        f"CPU={d.get('cpu_percent')}% MEM={d.get('mem_percent')}%"
                    )
            self.meta_text.delete("1.0", tk.END)
            self.meta_text.insert(tk.END, "\n".join(lines))
        except Exception as e:
            self.log(f"[Meta] Failed: {e}")

    # ----- Controls -----
    def run_analysis(self):
        if requests is None:
            return
        try:
            res = requests.get(f"{self.base_url}/analyze", timeout=2)
            summary = res.json()
            self.log(
                f"[Analysis] count={summary.get('count')}, "
                f"avg_complexity={summary.get('avg_complexity')}, "
                f"anomaly={summary.get('anomaly_score')}"
            )
        except Exception as e:
            self.log(f"[Analysis] Failed: {e}")

    def prune_data(self):
        if requests is None:
            return
        try:
            res = requests.post(f"{self.base_url}/prune", params={"max_docs": 1000}, timeout=2)
            result = res.json()
            self.log(
                f"[Prune] pruned={result.get('pruned')}, total={result.get('total')}"
            )
        except Exception as e:
            self.log(f"[Prune] Failed: {e}")

    # ----- Logs -----
    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.logs_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.logs_text.see(tk.END)

# =========================
# MAIN ENTRY
# =========================
def wait_for_port_file(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(PORT_FILE):
            try:
                with open(PORT_FILE) as f:
                    return int(f.read().strip())
            except Exception:
                pass
        time.sleep(0.5)
    raise RuntimeError("Timed out waiting for backend port file.")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    server_proc = multiprocessing.Process(target=run_server)
    server_proc.start()

    try:
        port = wait_for_port_file()
        print(f"[Main] Backend running on port {port}")
    except Exception as e:
        print(f"[Main] Failed to get backend port: {e}")
        port = 8000  # fallback

    app = USETXConsole(port)
    app.mainloop()

