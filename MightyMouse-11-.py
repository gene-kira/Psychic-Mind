import sys
import os
import time
import threading
import json
import socket
from math import atan2, degrees, sqrt
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib import request as urlrequest

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import psutil
    import win32gui
    import win32con
    import win32api
    import win32process
    import win32com.client
    import pythoncom
    HAVE_PYWIN32 = True
except ImportError:
    HAVE_PYWIN32 = False

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QCursor, QPixmap, QPainter, QColor, QPen

# =========================
# STORAGE ROOT SELECTION (D-Z, SMB via env, fallback C)
# =========================

def get_storage_root():
    smb_root = os.environ.get("MIGHTYMOUSE_SMB_ROOT")
    candidates = []
    if smb_root:
        candidates.append(smb_root)
    for d in range(ord('D'), ord('Z') + 1):
        drive = f"{chr(d)}:\\"
        candidates.append(drive)
    candidates.append("C:\\")
    for root in candidates:
        try:
            path = os.path.join(root, "MightyMouseData")
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".mm_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            return path
        except Exception:
            continue
    path = os.path.join(os.getcwd(), "MightyMouseData")
    os.makedirs(path, exist_ok=True)
    return path

STORAGE_ROOT = get_storage_root()
MEMORY_FILE = os.path.join(STORAGE_ROOT, "memory.json")
ML_DATA_FILE = os.path.join(STORAGE_ROOT, "enemy_traces.jsonl")
ML_MODEL_FILE = os.path.join(STORAGE_ROOT, "enemy_predictor.onnx")

# =========================
# EVENT BUS
# =========================

class EventBus:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type, handler):
        self.subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event_type, payload=None):
        for handler in self.subscribers.get(event_type, []):
            try:
                handler(payload)
            except Exception:
                pass

# =========================
# MEMORY + LEARNING (PERSISTENT)
# =========================

class MemoryStore:
    def __init__(self):
        self.data = {
            "web": {"high_risk_count": 0, "avg_score": 0.0, "samples": 0},
            "fs": {"high_risk_count": 0, "avg_score": 0.0, "samples": 0},
            "game": {
                "danger_peaks": [],
                "cluster_history": [],
                "future_danger_prediction": 0
            },
            "profiles": {"current": "combat"},
            "last_web_ts": 0.0,
            "last_fs_ts": 0.0,
            "last_game_ts": 0.0
        }
        self.load()

    def update_avg(self, domain, score, threshold=60):
        d = self.data[domain]
        d["samples"] += 1
        d["avg_score"] = ((d["avg_score"] * (d["samples"] - 1)) + score) / d["samples"]
        if score >= threshold:
            d["high_risk_count"] += 1

    def record_danger_peak(self, score):
        self.data["game"]["danger_peaks"].append((time.strftime("%H:%M:%S"), score))
        if len(self.data["game"]["danger_peaks"]) > 100:
            self.data["game"]["danger_peaks"].pop(0)

    def set_profile(self, name):
        self.data["profiles"]["current"] = name

    def get_profile(self):
        return self.data["profiles"]["current"]

    def set_future_danger(self, value):
        self.data["game"]["future_danger_prediction"] = value

    def mark_web_seen(self):
        self.data["last_web_ts"] = time.time()

    def mark_fs_seen(self):
        self.data["last_fs_ts"] = time.time()

    def mark_game_seen(self):
        self.data["last_game_ts"] = time.time()

    def save(self):
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def load(self):
        if not os.path.exists(MEMORY_FILE):
            return
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data.update(loaded)
        except Exception:
            pass

# =========================
# PLUGIN ARCHITECTURE
# =========================

class PluginBase:
    def __init__(self, name, bus, memory):
        self.name = name
        self.bus = bus
        self.memory = memory
        self.register()

    def register(self):
        pass

class DangerLoggerPlugin(PluginBase):
    def register(self):
        self.bus.subscribe("web_score", self.on_web_score)
        self.bus.subscribe("fs_score", self.on_fs_score)
        self.bus.subscribe("game_danger", self.on_game_danger)

    def on_web_score(self, score):
        if score is None:
            return
        self.memory.update_avg("web", score)

    def on_fs_score(self, score):
        if score is None:
            return
        self.memory.update_avg("fs", score)

    def on_game_danger(self, score):
        if score is None:
            return
        self.memory.record_danger_peak(score)

class AdaptiveThresholdPlugin(PluginBase):
    def __init__(self, name, bus, memory):
        self.web_threshold = 60
        self.fs_threshold = 60
        super().__init__(name, bus, memory)

    def register(self):
        self.bus.subscribe("tick", self.on_tick)

    def on_tick(self, _):
        web = self.memory.data["web"]
        fs = self.memory.data["fs"]
        self.web_threshold = max(40, 60 - int(web["avg_score"] / 10))
        self.fs_threshold = max(40, 60 - int(fs["avg_score"] / 10))

class DangerTimelinePredictorPlugin(PluginBase):
    def register(self):
        self.bus.subscribe("game_danger", self.on_game_danger)

    def on_game_danger(self, score):
        if score is None:
            return
        peaks = self.memory.data["game"]["danger_peaks"]
        peaks.append((time.time(), score))
        if len(peaks) > 50:
            peaks.pop(0)
        if len(peaks) >= 3:
            t1, s1 = peaks[-3]
            t2, s2 = peaks[-2]
            t3, s3 = peaks[-1]
            dt1 = t2 - t1
            dt2 = t3 - t2
            if dt1 > 0 and dt2 > 0:
                v1 = (s2 - s1) / dt1
                v2 = (s3 - s2) / dt2
                v = (v1 + v2) / 2
                future = s3 + v * 5.0
                self.memory.set_future_danger(int(max(0, min(100, future))))

# =========================
# CURSOR ENGINE
# =========================

class CursorEngine:
    def __init__(self, app):
        self.app = app

    def set_cursor(self, size=48, color=QColor(255, 255, 255), shape="circle"):
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(color, 3))
        if shape == "circle":
            p.drawEllipse(3, 3, size - 6, size - 6)
        elif shape == "diamond":
            points = [
                (size // 2, 3),
                (size - 3, size // 2),
                (size // 2, size - 3),
                (3, size // 2),
            ]
            p.drawPolygon(*[Qt.QPoint(x, y) for x, y in points])
        elif shape == "chevron":
            p.drawLine(3, size // 2, size // 2, 3)
            p.drawLine(size // 2, 3, size - 3, size // 2)
        p.end()
        self.app.setOverrideCursor(QCursor(pix))

    def set_state(self, profile, threat_level, future_danger):
        combined = max(threat_level, future_danger)
        if profile == "stealth":
            if combined < 40:
                self.set_cursor(32, QColor(120, 120, 120), "circle")
            else:
                self.set_cursor(40, QColor(255, 165, 0), "circle")
        elif profile == "combat":
            if combined < 40:
                self.set_cursor(40, QColor(0, 255, 0), "circle")
            elif combined < 70:
                self.set_cursor(48, QColor(255, 165, 0), "circle")
            else:
                self.set_cursor(56, QColor(255, 0, 0), "diamond")
        elif profile == "analysis":
            if combined < 40:
                self.set_cursor(36, QColor(0, 200, 255), "circle")
            else:
                self.set_cursor(44, QColor(255, 255, 0), "chevron")

# =========================
# TACTICAL HUD
# =========================

class TacticalHUD:
    ICONS = {
        "enemy": "⚠",
        "loot": "★",
        "objective": "◆",
        "hazard": "☣",
        "ally": "✚",
        "command": "▶",
        "info": "•",
        "sound": "♪",
        "predict": "⇢"
    }
    COLORS = {
        "enemy": "#ff4444",
        "loot": "#44ff44",
        "objective": "#ffff44",
        "hazard": "#ff5500",
        "ally": "#4488ff",
        "command": "#ffaa00",
        "info": "#ffffff",
        "sound": "#ff88ff",
        "predict": "#88ffdd"
    }

    def __init__(self):
        self.feed = []

    def push(self, kind, text):
        ts = time.strftime("%H:%M:%S")
        icon = self.ICONS.get(kind, "•")
        color = self.COLORS.get(kind, "#ffffff")
        line = f'<span style="color:{color}">[{ts}] {icon} {kind.upper()}: {text}</span>'
        self.feed.append(line)
        if len(self.feed) > 200:
            self.feed.pop(0)

    def get_html(self):
        return "<br>".join(self.feed)

# =========================
# WEB BRAIN
# =========================

class WebBrain:
    SUSPICIOUS_TLDS = {".ru", ".cn", ".tk", ".top", ".xyz"}

    def __init__(self, memory):
        self.latest_url = None
        self.reasons = []
        self.memory = memory

    def update_url(self, url):
        self.latest_url = url
        self.memory.mark_web_seen()

    def evaluate(self):
        self.reasons = []
        if not self.latest_url:
            return 0
        score = 0
        url = self.latest_url.lower()
        if url.startswith("http://") and "login" in url:
            score += 40
            self.reasons.append("Login over HTTP")
        for tld in self.SUSPICIOUS_TLDS:
            if url.endswith(tld):
                score += 30
                self.reasons.append(f"Suspicious TLD: {tld}")
                break
        if len(url) > 120:
            score += 20
            self.reasons.append("Unusually long URL")
        try:
            host = url.split("/")[2]
            socket.gethostbyname(host)
        except Exception:
            score += 30
            self.reasons.append("Host resolution failed")
        return max(0, min(100, score))

# =========================
# FILESYSTEM BRAIN
# =========================

class FilesystemBrain:
    HIGH_RISK_EXTS = {
        ".exe": 70, ".scr": 70, ".bat": 60, ".cmd": 60,
        ".ps1": 70, ".vbs": 70, ".js": 50, ".jar": 50,
        ".docm": 60, ".xlsm": 60, ".pptm": 60, ".zip": 20, ".rar": 20
    }

    def __init__(self, memory):
        self.hovered_path = None
        self.score = 0
        self.memory = memory

    def update_hover(self, path):
        self.hovered_path = path
        if path:
            self.score = self._score_path(path)
            self.memory.mark_fs_seen()
        else:
            self.score = 0

    def _score_path(self, path):
        path_lower = path.lower()
        for ext, val in self.HIGH_RISK_EXTS.items():
            if path_lower.endswith(ext):
                return val
        return 0

# =========================
# ML: ENEMY MOVEMENT DATA RECORDER
# =========================

class EnemyMovementRecorder:
    def __init__(self, path=ML_DATA_FILE):
        self.path = path
        self.lock = threading.Lock()

    def record_snapshot(self, minimap):
        if not minimap:
            return
        entry = {
            "timestamp": time.time(),
            "player": minimap.get("player", {}),
            "enemies": minimap.get("enemies", [])
        }
        line = json.dumps(entry)
        try:
            with self.lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass

# =========================
# ML: ENEMY MOVEMENT PREDICTOR (ONNX SCAFFOLD)
# =========================

class EnemyMovementPredictor:
    def __init__(self, hud):
        self.hud = hud
        self.session = None
        if ort is not None and os.path.exists(ML_MODEL_FILE):
            try:
                self.session = ort.InferenceSession(ML_MODEL_FILE, providers=["CPUExecutionProvider"])
            except Exception:
                self.session = None

    def predict_future_positions(self, minimap):
        if self.session is None or not minimap:
            return None
        enemies = minimap.get("enemies", [])
        if not enemies:
            return None
        vec = []
        for e in enemies:
            vec.append(float(e.get("x", 0)))
            vec.append(float(e.get("y", 0)))
        import numpy as np
        inp = np.array([vec], dtype=np.float32)
        try:
            inputs = {self.session.get_inputs()[0].name: inp}
            outputs = self.session.run(None, inputs)
            pred = outputs[0][0]
        except Exception:
            return None
        predicted = []
        for i in range(0, len(pred), 2):
            predicted.append({"x": float(pred[i]), "y": float(pred[i+1])})
        self._announce_prediction(enemies, predicted)
        return predicted

    def _announce_prediction(self, enemies, predicted):
        if not enemies or not predicted:
            return
        avg_dx = sum(p["x"] - e.get("x", 0) for p, e in zip(predicted, enemies)) / len(enemies)
        avg_dy = sum(p["y"] - e.get("y", 0) for p, e in zip(predicted, enemies)) / len(enemies)
        angle = (degrees(atan2(avg_dy, avg_dx)) + 360) % 360
        if angle < 45 or angle > 315:
            direction = "RIGHT"
        elif angle < 135:
            direction = "DOWN"
        elif angle < 225:
            direction = "LEFT"
        else:
            direction = "UP"
        self.hud.push("predict", f"Enemy movement forecast drifting {direction}")

# =========================
# GAME COGNITION
# =========================

class GameCognition:
    def __init__(self, hud, memory, recorder, predictor):
        self.minimap = None
        self.hud = hud
        self.memory = memory
        self.recorder = recorder
        self.predictor = predictor
        self.cones = "FRONT:0 RIGHT:0 BACK:0 LEFT:0"
        self.timeline = []
        self.enemy_history = {}
        self.clusters = []
        self.squad_command = ""
        self.heatmap = None
        self.sound_summary = ""
        self.predicted_positions = []

    def update_minimap(self, data):
        self.minimap = data
        self.memory.mark_game_seen()
        self.recorder.record_snapshot(data)
        self._update_enemy_history()
        self._update_cones()
        self._update_timeline()
        self._update_clusters()
        self._update_heatmap()
        self._update_squad_command()
        self._update_sounds()
        self._update_ml_prediction()

    def _update_enemy_history(self):
        if not self.minimap:
            return
        now = time.time()
        for idx, e in enumerate(self.minimap.get("enemies", [])):
            hist = self.enemy_history.setdefault(idx, [])
            hist.append((e.get("x", 0), e.get("y", 0), now))
            if len(hist) > 10:
                hist.pop(0)

    def _update_cones(self):
        if not self.minimap:
            self.cones = "FRONT:0 RIGHT:0 BACK:0 LEFT:0"
        else:
            player = self.minimap.get("player", {"x": 0, "y": 0, "yaw": 0})
            cones = {"FRONT": 0, "RIGHT": 0, "BACK": 0, "LEFT": 0}
            for e in self.minimap.get("enemies", []):
                dx = e.get("x", 0) - player["x"]
                dy = e.get("y", 0) - player["y"]
                angle = (degrees(atan2(dy, dx)) - player.get("yaw", 0) + 360) % 360
                if angle < 45 or angle > 315:
                    cones["FRONT"] += 1
                elif angle < 135:
                    cones["RIGHT"] += 1
                elif angle < 225:
                    cones["BACK"] += 1
                else:
                    cones["LEFT"] += 1
            self.cones = f"FRONT:{cones['FRONT']} RIGHT:{cones['RIGHT']} BACK:{cones['BACK']} LEFT:{cones['LEFT']}"

    def _update_timeline(self):
        enemies = self.minimap.get("enemies", []) if self.minimap else []
        score = min(100, len(enemies) * 10)
        self.timeline.append((time.strftime("%H:%M:%S"), score))
        if len(self.timeline) > 30:
            self.timeline.pop(0)
        self.memory.record_danger_peak(score)

    def _update_clusters(self):
        if not self.minimap:
            self.clusters = []
            return
        enemies = self.minimap.get("enemies", [])
        if not enemies:
            self.clusters = []
            return
        xs = [e.get("x", 0) for e in enemies]
        ys = [e.get("y", 0) for e in enemies]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        self.clusters = [{"center": (cx, cy), "count": len(enemies)}]
        self.memory.data["game"]["cluster_history"].append(
            {"time": time.strftime("%H:%M:%S"), "center": (cx, cy), "count": len(enemies)}
        )
        if len(self.memory.data["game"]["cluster_history"]) > 50:
            self.memory.data["game"]["cluster_history"].pop(0)

    def _update_heatmap(self):
        if not self.minimap:
            self.heatmap = None
            return
        enemies = self.minimap.get("enemies", [])
        if not enemies:
            self.heatmap = None
            return
        grid = [[0]*10 for _ in range(10)]
        for e in enemies:
            gx = int(min(9, max(0, e.get("x", 0) // 50)))
            gy = int(min(9, max(0, e.get("y", 0) // 50)))
            grid[gy][gx] += 1
        self.heatmap = grid

    def _update_squad_command(self):
        if not self.minimap:
            self.squad_command = ""
            return
        player = self.minimap.get("player", {"x": 0, "y": 0, "yaw": 0})
        enemies = self.minimap.get("enemies", [])
        allies = self.minimap.get("allies", [])
        cover = self.minimap.get("cover", [])
        cmd = self._generate_squad_command(enemies, allies, cover, player)
        self.squad_command = cmd
        self.hud.push("command", cmd)

    def _generate_squad_command(self, enemies, allies, cover, player):
        if not enemies:
            return "ADVANCE"
        density = len(enemies)
        if density >= 6:
            return "REGROUP WITH ALLIES"
        if cover:
            best, dist = self._nearest_cover(player, cover)
            if dist < 60:
                return "MOVE TO COVER"
        if any(self._cone_direction(e, player) == "RIGHT" for e in enemies):
            return "FLANK LEFT"
        return "HOLD POSITION"

    def _nearest_cover(self, player, cover_list):
        best = None
        best_dist = 99999
        for c in cover_list:
            cx = (c["x1"] + c["x2"]) / 2
            cy = (c["y1"] + c["y2"]) / 2
            dist = sqrt((cx - player["x"])**2 + (cy - player["y"])**2)
            if dist < best_dist:
                best_dist = dist
                best = (cx, cy)
        return best, best_dist

    def _cone_direction(self, enemy, player):
        dx = enemy.get("x", 0) - player["x"]
        dy = enemy.get("y", 0) - player["y"]
        angle = (degrees(atan2(dy, dx)) - player.get("yaw", 0) + 360) % 360
        if angle < 45 or angle > 315:
            return "FRONT"
        elif angle < 135:
            return "RIGHT"
        elif angle < 225:
            return "BACK"
        else:
            return "LEFT"

    def _update_sounds(self):
        if not self.minimap:
            self.sound_summary = ""
            return
        sounds = self.minimap.get("sounds", [])
        if not sounds:
            self.sound_summary = ""
            return
        player = self.minimap.get("player", {"x": 0, "y": 0, "yaw": 0})
        dirs = {"FRONT": 0, "RIGHT": 0, "BACK": 0, "LEFT": 0}
        for s in sounds:
            dx = s.get("x", 0) - player["x"]
            dy = s.get("y", 0) - player["y"]
            angle = (degrees(atan2(dy, dx)) - player.get("yaw", 0) + 360) % 360
            if angle < 45 or angle > 315:
                dirs["FRONT"] += s.get("intensity", 1)
            elif angle < 135:
                dirs["RIGHT"] += s.get("intensity", 1)
            elif angle < 225:
                dirs["BACK"] += s.get("intensity", 1)
            else:
                dirs["LEFT"] += s.get("intensity", 1)
        strongest = max(dirs, key=dirs.get)
        self.sound_summary = f"Sounds: {len(sounds)} strongest {strongest}"
        self.hud.push("sound", self.sound_summary)

    def _update_ml_prediction(self):
        if not self.predictor:
            return
        self.predicted_positions = self.predictor.predict_future_positions(self.minimap) or []

    def get_danger_score(self):
        if not self.timeline:
            return 0
        return self.timeline[-1][1]

# =========================
# PROFILES (AUTONOMOUS)
# =========================

class ProfileManager:
    PROFILES = ["stealth", "combat", "analysis"]

    def __init__(self, memory):
        self.memory = memory

    def get_current(self):
        return self.memory.get_profile()

    def autoselect_profile(self, mode, threat_level):
        profile = "stealth"
        if mode == "game":
            if threat_level >= 60:
                profile = "combat"
            else:
                profile = "stealth"
        elif mode in ("web", "filesystem"):
            if threat_level >= 50:
                profile = "analysis"
            else:
                profile = "stealth"
        else:
            profile = "stealth"
        self.memory.set_profile(profile)
        return profile

# =========================
# DIRECTX OVERLAY BRIDGE
# =========================

class DirectXOverlayBridge:
    def __init__(self, host="127.0.0.1", port=9797):
        self.host = host
        self.port = port
        self.enabled = True

    def send_state(self, game_cog, mode, profile, threat_level, future_danger):
        if not self.enabled:
            return
        state = {
            "mode": mode,
            "profile": profile,
            "threat_level": threat_level,
            "future_danger": future_danger,
            "cones": game_cog.cones,
            "timeline": game_cog.timeline[-5:],
            "clusters": game_cog.clusters,
            "predicted_positions": game_cog.predicted_positions,
            "sound_summary": game_cog.sound_summary,
        }
        payload = json.dumps(state).encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=0.05) as s:
                s.sendall(payload + b"\n")
        except Exception:
            pass

# =========================
# SYSTEM BACKBONE MONITOR (STANDALONE THREAD, COM-SAFE)
# =========================

class SystemBackboneMonitor:
    BROWSER_PROCESSES = {
        "chrome.exe", "msedge.exe", "firefox.exe", "opera.exe",
        "brave.exe", "vivaldi.exe"
    }
    GAME_PROCESSES = {
        "steam.exe", "epicgameslauncher.exe", "riotclientservices.exe",
        "battle.net.exe", "origin.exe", "uplay.exe"
    }
    BROWSER_CLASSES = {
        "Chrome_WidgetWin_1", "Chrome_WidgetWin_0",
        "MozillaWindowClass", "IEFrame", "ApplicationFrameWindow"
    }
    EXPLORER_CLASSES = {
        "CabinetWClass", "ExploreWClass"
    }
    GAME_CLASSES = {
        "UnrealWindow", "UnityWndClass", "DXGI", "SDL_app"
    }

    def __init__(self, orchestrator, interval=0.2):
        self.orch = orchestrator
        self.interval = interval
        self.running = HAVE_PYWIN32
        if not HAVE_PYWIN32:
            self.orch.log("[BACKBONE] pywin32 not available, backbone monitor disabled")

    def start(self):
        if not self.running:
            return
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()
        self.orch.log("[BACKBONE] Monitor started (200ms, hybrid detection)")

    def loop(self):
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        try:
            while self.running:
                try:
                    mode = self.detect_mode()
                    self.orch.set_backbone_mode(mode)
                except Exception as e:
                    self.orch.log(f"[BACKBONE ERROR] {e}")
                time.sleep(self.interval)
        finally:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def detect_mode(self):
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return "desktop"

        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc_name = ""
            try:
                proc = psutil.Process(pid)
                proc_name = proc.name().lower()
            except Exception:
                proc_name = ""
            cls = ""
            try:
                cls = win32gui.GetClassName(hwnd)
            except Exception:
                cls = ""

            if proc_name in self.BROWSER_PROCESSES:
                return "web"

            if cls in self.BROWSER_CLASSES:
                return "web"

            if cls in self.EXPLORER_CLASSES:
                return "filesystem"

            if proc_name in self.GAME_PROCESSES:
                return "game"

            if cls in self.GAME_CLASSES:
                return "game"

            if self.is_fullscreen(hwnd):
                return "game"

            return "desktop"
        except Exception:
            return "desktop"

    def is_fullscreen(self, hwnd):
        try:
            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect
            width = right - left
            height = bottom - top

            screen_w = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            screen_h = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

            if width == screen_w and height == screen_h:
                return True

            area_window = width * height
            area_screen = screen_w * screen_h
            if area_screen <= 0:
                return False
            coverage = area_window / area_screen
            if coverage >= 0.95:
                return True

            return False
        except Exception:
            return False

# =========================
# ORCHESTRATOR (BACKBONE-DRIVEN)
# =========================

class Orchestrator:
    def __init__(self, bus, memory, cursor, hud, web, fs, game, profiles, overlay_bridge):
        self.bus = bus
        self.memory = memory
        self.cursor = cursor
        self.hud = hud
        self.web = web
        self.fs = fs
        self.game = game
        self.profiles = profiles
        self.overlay_bridge = overlay_bridge
        self.mode = "desktop"
        self.logs = []
        self.backbone_mode = "desktop"

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.logs.append(line)
        if len(self.logs) > 400:
            self.logs.pop(0)
        print(line)

    def set_backbone_mode(self, mode):
        if mode not in ("desktop", "filesystem", "web", "game"):
            mode = "desktop"
        if mode != self.backbone_mode:
            self.log(f"[BACKBONE] Mode → {mode}")
        self.backbone_mode = mode

    def tick(self):
        self.bus.publish("tick", None)

        self.mode = self.backbone_mode

        threat_level = 0
        if self.mode == "filesystem":
            threat_level = self.fs.score
            self.bus.publish("fs_score", threat_level)
        elif self.mode == "web":
            threat_level = self.web.evaluate()
            self.bus.publish("web_score", threat_level)
        elif self.mode == "game":
            threat_level = self.game.get_danger_score()
            self.bus.publish("game_danger", threat_level)
        else:
            threat_level = 0

        future_danger = self.memory.data["game"]["future_danger_prediction"]
        effective_threat = max(threat_level, future_danger)

        profile = self.profiles.autoselect_profile(self.mode, effective_threat)

        self.cursor.set_state(profile, threat_level, future_danger)
        self.overlay_bridge.send_state(self.game, self.mode, profile, threat_level, future_danger)

# =========================
# MULTI-THREADED HTTP SERVERS
# =========================

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

# =========================
# LOCAL FEEDS
# =========================

def start_url_server(web_brain, orchestrator):
    class URLHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/url":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                url = data.get("url")
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    web_brain.update_url(url)
                    orchestrator.log(f"[URL FEED] {url}")
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK")
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Invalid URL")
            except Exception as e:
                orchestrator.log(f"[URL FEED ERROR] {e}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Bad Request")

        def log_message(self, format, *args):
            return

    def run():
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 8787), URLHandler)
            orchestrator.log("[URL FEED] Listening on http://127.0.0.1:8787/url")
            server.serve_forever()
        except OSError as e:
            orchestrator.log(f"[URL FEED] Failed to bind: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()

def start_game_event_server(hud, orchestrator):
    class GameEventHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/game":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                kind = data.get("kind", "info")
                text = data.get("text", "")
                hud.push(kind, text)
                orchestrator.log(f"[GAME EVENT] {kind} – {text}")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            except Exception as e:
                orchestrator.log(f"[GAME EVENT ERROR] {e}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Bad Request")

        def log_message(self, format, *args):
            return

    def run():
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 8788), GameEventHandler)
            orchestrator.log("[GAME EVENT] Listening on http://127.0.0.1:8788/game")
            server.serve_forever()
        except OSError as e:
            orchestrator.log(f"[GAME EVENT] Failed to bind: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()

def start_minimap_server(game_cognition, orchestrator):
    class MiniMapHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/minimap":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                game_cognition.update_minimap(data)
                orchestrator.log("[MINIMAP] Updated mini‑map data")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            except Exception as e:
                orchestrator.log(f"[MINIMAP ERROR] {e}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Bad Request")

        def log_message(self, format, *args):
            return

    def run():
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 8789), MiniMapHandler)
            orchestrator.log("[MINIMAP] Listening on http://127.0.0.1:8789/minimap")
            server.serve_forever()
        except OSError as e:
            orchestrator.log(f"[MINIMAP] Failed to bind: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()

def start_fs_hover_server(fs_brain, orchestrator):
    class FSHoverHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/fs_hover":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                path = data.get("path")
                if isinstance(path, str):
                    fs_brain.update_hover(path)
                    orchestrator.log(f"[FS HOVER] {path}")
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK")
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Invalid path")
            except Exception as e:
                orchestrator.log(f"[FS HOVER ERROR] {e}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Bad Request")

        def log_message(self, format, *args):
            return

    def run():
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 8790), FSHoverHandler)
            orchestrator.log("[FS HOVER] Listening on http://127.0.0.1:8790/fs_hover")
            server.serve_forever()
        except OSError as e:
            orchestrator.log(f"[FS HOVER] Failed to bind: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()

# =========================
# GAME HELPER SIM (OPTIONAL)
# =========================

class GameHelperClient:
    def __init__(self, base_url="http://127.0.0.1"):
        self.base_url = base_url

    def send_minimap(self, data):
        url = f"{self.base_url}:8789/minimap"
        self._post_json(url, data)

    def send_event(self, kind, text):
        url = f"{self.base_url}:8788/game"
        self._post_json(url, {"kind": kind, "text": text})

    def send_url(self, url_str):
        url = f"{self.base_url}:8787/url"
        self._post_json(url, {"url": url_str})

    def _post_json(self, url, payload):
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
            urlrequest.urlopen(req, timeout=0.5)
        except Exception:
            pass

def run_game_helper_sim():
    helper = GameHelperClient()
    yaw = 0
    while True:
        enemies = [
            {"x": 100, "y": 200},
            {"x": 250, "y": 300},
            {"x": 400, "y": 150},
        ]
        allies = [
            {"x": 50, "y": 220},
        ]
        cover = [
            {"x1": 180, "y1": 180, "x2": 220, "y2": 220},
        ]
        sounds = [
            {"x": 300, "y": 260, "intensity": 3},
            {"x": 80, "y": 190, "intensity": 1},
        ]
        minimap = {
            "player": {"x": 200, "y": 200, "yaw": yaw},
            "enemies": enemies,
            "allies": allies,
            "cover": cover,
            "sounds": sounds
        }
        helper.send_minimap(minimap)
        helper.send_event("enemy", "Enemy cluster detected ahead")
        yaw = (yaw + 15) % 360
        time.sleep(1.0)

# =========================
# PYTHON FILESYSTEM HOVER HELPER (IN-PROCESS, COM-SAFE)
# =========================

class FilesystemHoverHelper:
    def __init__(self, fs_brain, orchestrator, interval=0.3):
        self.fs_brain = fs_brain
        self.orch = orchestrator
        self.interval = interval
        self.running = HAVE_PYWIN32
        if not HAVE_PYWIN32:
            self.orch.log("[FS HOVER HELPER] pywin32 not available, helper disabled")

    def start(self):
        if not self.running:
            return
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()
        self.orch.log("[FS HOVER HELPER] Started")

    def loop(self):
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        try:
            shell = win32com.client.Dispatch("Shell.Application")
            while self.running:
                try:
                    path = self._get_hovered_path(shell)
                    if path:
                        self.fs_brain.update_hover(path)
                    time.sleep(self.interval)
                except Exception as e:
                    self.orch.log(f"[FS HOVER HELPER ERROR] {e}")
                    time.sleep(self.interval)
        finally:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def _get_hovered_path(self, shell):
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return None
        for w in shell.Windows():
            try:
                if int(w.HWND) == hwnd:
                    folder = w.Document.Folder
                    view = w.Document
                    items = view.SelectedItems()
                    if items.Count > 0:
                        item = items.Item(0)
                        return item.Path
            except Exception:
                continue
        return None

# =========================
# BROWSER URL WATCHER (REAL DEVTOOLS INTEGRATION)
# =========================

class BrowserURLWatcher:
    """
    Real browser URL capture using Chrome/Edge DevTools Protocol.
    Requires browser started with: --remote-debugging-port=9222
    Pure Python, no extensions.
    """
    def __init__(self, web_brain, orchestrator, interval=0.5, devtools_port=9222):
        self.web_brain = web_brain
        self.orch = orchestrator
        self.interval = interval
        self.devtools_port = devtools_port
        self.running = True

    def start(self):
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()
        self.orch.log(f"[BROWSER WATCHER] Started (DevTools port {self.devtools_port})")

    def loop(self):
        while self.running:
            try:
                self._poll_browser()
            except Exception as e:
                self.orch.log(f"[BROWSER WATCHER ERROR] {e}")
            time.sleep(self.interval)

    def _poll_browser(self):
        url = f"http://127.0.0.1:{self.devtools_port}/json"
        try:
            with urlrequest.urlopen(url, timeout=0.3) as resp:
                data = resp.read().decode("utf-8")
            tabs = json.loads(data)
        except Exception:
            return

        if not HAVE_PYWIN32:
            return

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return

        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid)
            proc_name = proc.name().lower()
        except Exception:
            proc_name = ""

        if proc_name not in ("chrome.exe", "msedge.exe", "brave.exe", "vivaldi.exe"):
            return

        active_url = None
        for t in tabs:
            if t.get("type") != "page":
                continue
            u = t.get("url")
            if isinstance(u, str) and u.startswith(("http://", "https://")):
                active_url = u
                break

        if active_url and active_url != self.web_brain.latest_url:
            self.web_brain.update_url(active_url)
            self.orch.log(f"[BROWSER WATCHER] URL → {active_url}")

# =========================
# GUI COCKPIT (READ-ONLY, AUTONOMOUS)
# =========================

class CockpitGUI(QWidget):
    def __init__(self, orch, hud, web, fs, game, profiles, memory):
        super().__init__()
        self.orch = orch
        self.hud = hud
        self.web = web
        self.fs = fs
        self.game = game
        self.profiles = profiles
        self.memory = memory

        self.setWindowTitle("Mighty Mouse – Autonomous Tactical HUD (Backbone + Heuristics)")
        self.resize(1200, 800)

        main_layout = QVBoxLayout()

        self.status_label = QLabel("Status: Running")
        self.mode_label = QLabel("Mode: desktop")
        self.profile_label = QLabel(f"Profile: {self.profiles.get_current()}")
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.mode_label)
        main_layout.addWidget(self.profile_label)

        self.hover_label = QLabel("Hovered path: (none)")
        self.fs_score_label = QLabel("FS Threat score: 0")
        main_layout.addWidget(self.hover_label)
        main_layout.addWidget(self.fs_score_label)

        self.web_url_label = QLabel("Web URL: (none)")
        self.web_reasons_view = QTextEdit()
        self.web_reasons_view.setReadOnly(True)
        main_layout.addWidget(self.web_url_label)
        main_layout.addWidget(self.web_reasons_view)

        self.tactical_label = QLabel("Tactical Feed:")
        self.tactical_view = QTextEdit()
        self.tactical_view.setReadOnly(True)
        main_layout.addWidget(self.tactical_label)
        main_layout.addWidget(self.tactical_view)

        self.cone_label = QLabel("3D Cone Visualization (textual):")
        self.cone_view = QLabel("FRONT:0 RIGHT:0 BACK:0 LEFT:0")
        main_layout.addWidget(self.cone_label)
        main_layout.addWidget(self.cone_view)

        self.timeline_label = QLabel("Danger Timeline:")
        self.timeline_view = QTextEdit()
        self.timeline_view.setReadOnly(True)
        main_layout.addWidget(self.timeline_label)
        main_layout.addWidget(self.timeline_view)

        self.future_label = QLabel("Predicted Future Danger (5s): 0")
        main_layout.addWidget(self.future_label)

        self.memory_label = QLabel("Memory / Learning Snapshot:")
        self.memory_view = QTextEdit()
        self.memory_view.setReadOnly(True)
        main_layout.addWidget(self.memory_label)
        main_layout.addWidget(self.memory_view)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        main_layout.addWidget(self.log_view)

        self.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(250)

    def update_gui(self):
        self.status_label.setText("Status: Running")
        self.mode_label.setText(f"Mode: {self.orch.mode}")
        self.profile_label.setText(f"Profile: {self.profiles.get_current()}")

        if self.fs.hovered_path:
            self.hover_label.setText(f"Hovered path: {self.fs.hovered_path}")
        else:
            self.hover_label.setText("Hovered path: (none)")
        self.fs_score_label.setText(f"FS Threat score: {self.fs.score}")

        if self.web.latest_url:
            self.web_url_label.setText(f"Web URL: {self.web.latest_url}")
            self.web_reasons_view.setPlainText("\n".join(self.web.reasons))
        else:
            self.web_url_label.setText("Web URL: (none)")
            self.web_reasons_view.setPlainText("")

        self.tactical_view.setHtml(self.hud.get_html())
        self.tactical_view.verticalScrollBar().setValue(
            self.tactical_view.verticalScrollBar().maximum()
        )

        self.cone_view.setText(self.game.cones)

        if self.game.timeline:
            lines = [f"{t} → {s}" for t, s in self.game.timeline]
            self.timeline_view.setPlainText("\n".join(lines))
        else:
            self.timeline_view.setPlainText("")

        future = self.memory.data["game"]["future_danger_prediction"]
        self.future_label.setText(f"Predicted Future Danger (5s): {future}")

        self.memory_view.setPlainText(json.dumps(self.memory.data, indent=2))

        self.log_view.setPlainText("\n".join(self.orch.logs))
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

# =========================
# MAIN
# =========================

def main():
    if "--game-helper" in sys.argv:
        run_game_helper_sim()
        return

    app = QApplication(sys.argv)

    bus = EventBus()
    memory = MemoryStore()

    cursor = CursorEngine(app)
    hud = TacticalHUD()
    web = WebBrain(memory)
    fs = FilesystemBrain(memory)

    recorder = EnemyMovementRecorder()
    predictor = EnemyMovementPredictor(hud)
    game = GameCognition(hud, memory, recorder, predictor)

    profiles = ProfileManager(memory)
    overlay_bridge = DirectXOverlayBridge()

    orch = Orchestrator(bus, memory, cursor, hud, web, fs, game, profiles, overlay_bridge)

    danger_logger = DangerLoggerPlugin("danger_logger", bus, memory)
    adaptive_threshold = AdaptiveThresholdPlugin("adaptive_threshold", bus, memory)
    danger_predictor = DangerTimelinePredictorPlugin("danger_predictor", bus, memory)

    gui = CockpitGUI(orch, hud, web, fs, game, profiles, memory)
    gui.show()

    start_url_server(web, orch)
    start_game_event_server(hud, orch)
    start_minimap_server(game, orch)
    start_fs_hover_server(fs, orch)

    hover_helper = FilesystemHoverHelper(fs, orch)
    hover_helper.start()

    backbone = SystemBackboneMonitor(orch, interval=0.2)
    backbone.start()

    browser_watcher = BrowserURLWatcher(web, orch, interval=0.5, devtools_port=9222)
    browser_watcher.start()

    def engine_loop():
        while True:
            orch.tick()
            memory.save()
            time.sleep(0.3)

    t = threading.Thread(target=engine_loop, daemon=True)
    t.start()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

