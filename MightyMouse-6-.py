import sys
import time
import threading
import json
import socket
from math import atan2, degrees, sqrt
from http.server import BaseHTTPRequestHandler, HTTPServer

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QCursor, QPixmap, QPainter, QColor, QPen

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
# MEMORY + LEARNING
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
            "profiles": {"current": "combat"}
        }

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
        self.pulse_state = 0

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
        "info": "•"
    }
    COLORS = {
        "enemy": "#ff4444",
        "loot": "#44ff44",
        "objective": "#ffff44",
        "hazard": "#ff5500",
        "ally": "#4488ff",
        "command": "#ffaa00",
        "info": "#ffffff"
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

    def __init__(self):
        self.latest_url = None
        self.reasons = []

    def update_url(self, url):
        self.latest_url = url

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

    def __init__(self):
        self.hovered_path = None
        self.score = 0

    def update_hover(self, path):
        self.hovered_path = path
        if path:
            self.score = self._score_path(path)
        else:
            self.score = 0

    def _score_path(self, path):
        path_lower = path.lower()
        for ext, val in self.HIGH_RISK_EXTS.items():
            if path_lower.endswith(ext):
                return val
        return 0

# =========================
# GAME COGNITION (PREDICTIVE)
# =========================

class GameCognition:
    def __init__(self, hud, memory):
        self.minimap = None
        self.hud = hud
        self.memory = memory
        self.cones = "FRONT:0 RIGHT:0 BACK:0 LEFT:0"
        self.timeline = []
        self.enemy_history = {}
        self.clusters = []
        self.squad_command = ""
        self.heatmap = None

    def update_minimap(self, data):
        self.minimap = data
        self._update_enemy_history()
        self._update_cones()
        self._update_timeline()
        self._update_clusters()
        self._update_heatmap()
        self._update_squad_command()

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

    def get_danger_score(self):
        if not self.timeline:
            return 0
        return self.timeline[-1][1]

# =========================
# PROFILES (AUTONOMOUS)
# =========================

class ProfileManager:
    def __init__(self, memory):
        self.memory = memory

    def get_current(self):
        return self.memory.get_profile()

    def auto_update_profile(self, global_threat):
        if global_threat < 30:
            self.memory.set_profile("stealth")
        elif global_threat < 60:
            self.memory.set_profile("analysis")
        else:
            self.memory.set_profile("combat")

# =========================
# ORCHESTRATOR (AUTONOMOUS MODES)
# =========================

class Orchestrator:
    def __init__(self, bus, memory, cursor, hud, web, fs, game, profiles):
        self.bus = bus
        self.memory = memory
        self.cursor = cursor
        self.hud = hud
        self.web = web
        self.fs = fs
        self.game = game
        self.profiles = profiles
        self.mode = "desktop"
        self.logs = []

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.logs.append(line)
        if len(self.logs) > 400:
            self.logs.pop(0)
        print(line)

    def auto_select_mode(self):
        if self.game.minimap is not None:
            self.mode = "game"
        elif self.web.latest_url:
            self.mode = "web"
        elif self.fs.hovered_path:
            self.mode = "filesystem"
        else:
            self.mode = "desktop"

    def tick(self):
        self.bus.publish("tick", None)

        self.auto_select_mode()

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
        global_threat = max(threat_level, future_danger)
        self.profiles.auto_update_profile(global_threat)
        profile = self.profiles.get_current()

        self.cursor.set_state(profile, threat_level, future_danger)

# =========================
# LOCAL FEEDS (BROWSER / GAME / MINIMAP)
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
            server = HTTPServer(("127.0.0.1", 8787), URLHandler)
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
            server = HTTPServer(("127.0.0.1", 8788), GameEventHandler)
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
            server = HTTPServer(("127.0.0.1", 8789), MiniMapHandler)
            orchestrator.log("[MINIMAP] Listening on http://127.0.0.1:8789/minimap")
            server.serve_forever()
        except OSError as e:
            orchestrator.log(f"[MINIMAP] Failed to bind: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()

# =========================
# GUI COCKPIT (READ‑ONLY MODE/PROFILE)
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

        self.setWindowTitle("Mighty Mouse – Autonomous Tactical Visor")
        self.resize(1200, 800)

        main_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        self.status_label = QLabel("Status: Running")
        self.mode_label = QLabel("Mode: desktop")
        self.profile_label = QLabel(f"Profile: {self.profiles.get_current()}")
        top_row.addWidget(self.status_label)
        top_row.addWidget(self.mode_label)
        top_row.addWidget(self.profile_label)
        main_layout.addLayout(top_row)

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

        self.cone_label = QLabel("3D Cone Visualization:")
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
        self.profile_label.setText(f"Profile: {self.profiles.get_current()}")
        self.mode_label.setText(f"Mode: {self.orch.mode}")

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
    app = QApplication(sys.argv)

    bus = EventBus()
    memory = MemoryStore()

    cursor = CursorEngine(app)
    hud = TacticalHUD()
    web = WebBrain()
    fs = FilesystemBrain()
    game = GameCognition(hud, memory)
    profiles = ProfileManager(memory)

    orch = Orchestrator(bus, memory, cursor, hud, web, fs, game, profiles)

    danger_logger = DangerLoggerPlugin("danger_logger", bus, memory)
    adaptive_threshold = AdaptiveThresholdPlugin("adaptive_threshold", bus, memory)
    danger_predictor = DangerTimelinePredictorPlugin("danger_predictor", bus, memory)

    gui = CockpitGUI(orch, hud, web, fs, game, profiles, memory)
    gui.show()

    start_url_server(web, orch)
    start_game_event_server(hud, orch)
    start_minimap_server(game, orch)

    def engine_loop():
        while True:
            orch.tick()
            time.sleep(0.3)

    t = threading.Thread(target=engine_loop, daemon=True)
    t.start()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

