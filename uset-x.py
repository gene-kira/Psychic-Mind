# =========================
# USET-X FULL PLATFORM
# =========================

import sys, subprocess, importlib

def install(pkg, imp=None):
    try:
        return importlib.import_module(imp or pkg)
    except ImportError:
        print(f"[AutoLoader] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(imp or pkg)

# Core
np = install("numpy")
io = install("io")
hashlib = install("hashlib")
zlib = install("zlib")
requests = install("requests")

# Imaging
Image = install("PIL.Image", "PIL.Image")
cv2 = install("opencv-python", "cv2")

# AI
torch = install("torch")
nn = torch.nn
optim = torch.optim

torchvision = install("torchvision")
T = install("torchvision.transforms", "torchvision.transforms")
models = install("torchvision.models.detection", "torchvision.models.detection")

# DB
pymongo = install("pymongo")
from pymongo import MongoClient

# Web
fastapi = install("fastapi")
uvicorn = install("uvicorn")
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# =========================
# CONFIG
# =========================
MONGO_URI = "mongodb://localhost:27017/"
NODE_PEERS = []  # add other server URLs

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

    def __init__(self):
        print("[USET-X] Init...")
        self.model = models.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])

        self.net = TransitionNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    # ---------- PERCEPTION ----------
    def detect_objects(self, bytes_data):
        try:
            img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            tensor = self.transform(img)
            with torch.no_grad():
                preds = self.model([tensor])[0]

            return [int(l) for s, l in zip(preds["scores"], preds["labels"]) if s > 0.5]
        except:
            return []

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
        target = torch.tanh(vec)  # simple target
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
            history.append({
                "vector": vec.detach().tolist(),
                "loss": loss
            })

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
engine = USETX()

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

    record = {
        "state": state,
        "evolution": evolution
    }

    # Save to DB
    collection.insert_one(record)

    # Broadcast to peers
    for peer in NODE_PEERS:
        try:
            requests.post(f"{peer}/ingest", json=record)
        except:
            pass

    return JSONResponse(record)

# ---------- INGEST FROM OTHER NODES ----------
@app.post("/ingest")
async def ingest(data: dict):
    collection.insert_one(data)
    return {"status": "received"}

# ---------- GET STATES ----------
@app.get("/states")
def get_states():
    data = list(collection.find({}, {"_id": 0}))
    return {"count": len(data), "data": data}

# ---------- COMPARE ----------
@app.get("/compare/{i}/{j}")
def compare(i: int, j: int):
    data = list(collection.find({}, {"_id": 0}))

    try:
        s1 = data[i]["state"]["encoding"]
        s2 = data[j]["state"]["encoding"]

        return {
            "distance": abs(s1 - s2),
            "hamming": bin(s1 ^ s2).count("1")
        }
    except:
        return {"error": "invalid indices"}

# =========================
# STREAMING (CAMERA)
# =========================
def camera_stream():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        data = buffer.tobytes()

        state = engine.build_state(data)
        evolution = engine.simulate(state)

        collection.insert_one({
            "state": state,
            "evolution": evolution
        })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import threading

    # Start camera thread (optional)
    threading.Thread(target=camera_stream, daemon=True).start()

    print("[USET-X] Running full platform...")
    uvicorn.run(app, host="0.0.0.0", port=8000)