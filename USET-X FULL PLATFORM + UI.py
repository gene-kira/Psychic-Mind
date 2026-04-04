# =========================
# USET-X FULL PLATFORM + DASHBOARD
# =========================
import sys, subprocess, importlib, threading, time, io, random

# -------------------------
# AUTO-INSTALLER
# -------------------------
def install(pkg, imp=None):
    try:
        return importlib.import_module(imp or pkg)
    except ImportError:
        print(f"[AutoLoader] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(imp or pkg)

# -------------------------
# CORE
# -------------------------
np = install("numpy")
hashlib = install("hashlib")
zlib = install("zlib")
requests = install("requests")
io = install("io")

# -------------------------
# IMAGE + CAMERA
# -------------------------
Image = install("PIL.Image", "PIL.Image")
cv2 = install("opencv-python", "cv2")

# -------------------------
# AI + LEARNING
# -------------------------
torch = install("torch")
nn = torch.nn
optim = torch.optim
torchvision = install("torchvision")
T = install("torchvision.transforms", "torchvision.transforms")
models = install("torchvision.models.detection", "torchvision.models.detection")

# -------------------------
# DATABASE
# -------------------------
pymongo = install("pymongo")
from pymongo import MongoClient

# -------------------------
# WEB SERVER
# -------------------------
fastapi = install("fastapi")
uvicorn = install("uvicorn")
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# -------------------------
# DASHBOARD UI
# -------------------------
PyQt5 = install("PyQt5")
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout,
    QWidget, QLabel, QPushButton, QListWidget
)
from PyQt5.QtCore import QTimer
matplotlib = install("matplotlib")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
MONGO_URI = "mongodb://localhost:27017/"
NODE_PEERS = []

# =========================
# LEARNING MODEL
# =========================
class TransitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    def forward(self, x):
        return self.net(x)

# =========================
# ENGINE
# =========================
class USETX:

    def __init__(self, use_ai=False):
        print("[USET-X] Init engine...")
        self.use_ai = use_ai
        if use_ai:
            try:
                self.model = models.fasterrcnn_resnet50_fpn(pretrained=True)
                self.model.eval()
                self.transform = T.Compose([T.ToTensor()])
            except Exception as e:
                print("[USET-X] Failed to load AI model, using stub:", e)
                self.use_ai = False

        self.net = TransitionNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    # ---------- PERCEPTION ----------
    def detect_objects(self, bytes_data):
        if self.use_ai:
            try:
                img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                tensor = self.transform(img)
                with torch.no_grad():
                    preds = self.model([tensor])[0]
                return [int(l) for s, l in zip(preds["scores"], preds["labels"]) if s > 0.5]
            except:
                return []
        # fallback stub
        return [random.randint(0, 5)]

    # ---------- ENCODING ----------
    def encode(self, b):
        return int.from_bytes(hashlib.sha256(b).digest(), "big")

    # ---------- COMPLEXITY ----------
    def complexity(self, b):
        return len(zlib.compress(b))

    # ---------- STATE ----------
    def build_state(self, b):
        return {
            "objects": self.detect_objects(b),
            "complexity": self.complexity(b),
            "encoding": self.encode(b)
        }

    def to_vector(self, state):
        return torch.tensor([
            sum(state["objects"]) if state["objects"] else 0,
            state["complexity"] % 10000,
            state["encoding"] % 10000
        ], dtype=torch.float32)

    # ---------- LEARNING TRANSITION ----------
    def transition(self, vec):
        return self.net(vec)

    def train_step(self, vec):
        target = torch.tanh(vec)
        pred = self.net(vec)
        loss = ((pred - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def simulate(self, state, steps=5):
        vec = self.to_vector(state)
        history = []
        for _ in range(steps):
            loss = self.train_step(vec)
            vec = self.transition(vec)
            history.append({"vector": vec.detach().tolist(), "loss": loss})
        return history

# =========================
# DATABASE
# =========================
client = MongoClient(MONGO_URI)
db = client["usetx"]
collection = db["states"]

# =========================
# SERVER
# =========================
app = FastAPI()
engine = USETX(use_ai=False)  # set True for real AI if machine can handle it

# ---------- ROOT ----------
@app.get("/")
def root():
    return {"status": "USET-X platform running"}

# ---------- PROCESS ----------
@app.post("/process")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    state = engine.build_state(data)
    evolution = engine.simulate(state)
    record = {"state": state, "evolution": evolution}
    collection.insert_one(record)
    for peer in NODE_PEERS:
        try: requests.post(f"{peer}/ingest", json=record)
        except: pass
    return JSONResponse(record)

# ---------- INGEST ----------
@app.post("/ingest")
async def ingest(data: dict):
    collection.insert_one(data)
    return {"status": "received"}

# ---------- GET STATES ----------
@app.get("/states")
def get_states():
    data = list(collection.find({}, {"_id": 0}))
    return {"count": len(data), "data": data}

# =========================
# CAMERA STREAM (OPTIONAL)
# =========================
def camera_stream():
    try:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            _, buffer = cv2.imencode(".jpg", frame)
            data = buffer.tobytes()
            state = engine.build_state(data)
            evolution = engine.simulate(state)
            collection.insert_one({"state": state, "evolution": evolution})
            time.sleep(1)
    except:
        print("[Camera] No camera detected or failed.")

# =========================
# DASHBOARD
# =========================
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("USET-X Live Dashboard")
        self.setGeometry(100, 100, 900, 600)

        layout = QVBoxLayout()
        self.label = QLabel("Live State Universe")
        layout.addWidget(self.label)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.clicked.connect(self.update_data)
        layout.addWidget(self.refresh_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(2000)  # refresh every 2 sec

    def update_data(self):
        try:
            res = requests.get("http://127.0.0.1:8000/states")
            data = res.json()["data"]
            self.list_widget.clear()
            for i, s in enumerate(data[-10:]):  # last 10 states
                self.list_widget.addItem(
                    f"{i}: complexity={s['state']['complexity']}, objs={s['state']['objects']}"
                )
            if data:
                last = data[-1]["evolution"]
                self.plot(last)
        except Exception as e:
            self.label.setText(f"Error: {e}")

    def plot(self, evolution):
        self.ax.clear()
        arr = np.array([e["vector"] for e in evolution])
        for i in range(arr.shape[1]):
            self.ax.plot(arr[:, i])
        self.ax.set_title("State Evolution")
        self.canvas.draw()

# =========================
# RUN EVERYTHING
# =========================
def start_backend():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

def start_ui():
    app_qt = QApplication(sys.argv)
    win = Dashboard()
    win.show()
    sys.exit(app_qt.exec_())

if __name__ == "__main__":
    # backend in a thread
    threading.Thread(target=start_backend, daemon=True).start()
    # camera thread
    threading.Thread(target=camera_stream, daemon=True).start()
    # run dashboard in main thread
    start_ui()