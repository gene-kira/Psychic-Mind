# =========================
# MIGHTY MOUSE – FULL UPGRADED CODE (PREDICTIVE + WHAT-IF)
# =========================

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
import ipaddress
import statistics

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
# STORAGE ROOT SELECTION
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
# MEMORY + LEARNING
# =========================

class MemoryStore:
    def __init__(self):
        self.data = {
            "web": {
                "high_risk_count": 0,
                "avg_score": 0.0,
                "samples": 0,
                "trusted_ips": {},
                "domain_stats": {}
            },
            "fs": {"high_risk_count": 0, "avg_score": 0.0, "samples": 0},
            "game": {
                "danger_peaks": [],
                "cluster_history": [],
                "future_danger_prediction": 0,
                "future_danger_multi": {},   # horizon -> value
                "trusted_servers": {},
                "server_status": "UNKNOWN",
                "last_server_ip": None,
                "last_server_name": None
            },
            "profiles": {"current": "combat", "style": {"aggressive": 0, "cautious": 0}},
            "network": {
                "baseline": {},
                "anomalies": [],
                "baseline_start_ts": 0,
                "baseline_complete": False,
                "cone_counts": {
                    "FRONT": 0,
                    "RIGHT": 0,
                    "BACK": 0,
                    "LEFT": 0
                },
                "last_scan_ts": 0
            },
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

    def set_future_danger_multi(self, mapping):
        self.data["game"]["future_danger_multi"] = mapping

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
# GEOIP STUB
# =========================

def lookup_country_for_ip(ip: str) -> str:
    return "UNKNOWN"

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
        "predict": "⇢",
        "server": "⛨"
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
        "predict": "#88ffdd",
        "server": "#88aaff"
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
        if not self.feed:
            return '<span style="color:#666666">[No tactical events yet]</span>'
        return "<br>".join(self.feed)

# =========================
# WEB BRAIN (WITH MULTI-IP TRUST)
# =========================

class WebBrain:
    SUSPICIOUS_TLDS = {".ru", ".cn", ".tk", ".top", ".xyz"}

    KNOWN_GOOD_CIDRS = {
        "yahoo.com": ["74.6.0.0/16"],
        "google.com": ["142.250.0.0/15", "172.217.0.0/16"],
        "steamcommunity.com": ["162.254.0.0/16"],
    }

    def __init__(self, memory, hud):
        self.latest_url = None
        self.reasons = []
        self.memory = memory
        self.hud = hud
        self.trusted_ips = self.memory.data["web"].setdefault("trusted_ips", {})
        self.domain_stats = self.memory.data["web"].setdefault("domain_stats", {})

    def update_url(self, url):
        self.latest_url = url
        self.memory.mark_web_seen()

    def _extract_domain(self, url):
        try:
            host = url.split("/")[2]
            host = host.split(":")[0]
            parts = host.split(".")
            if len(parts) >= 2:
                return ".".join(parts[-2:])
            return host
        except Exception:
            return None

    def _resolve_ip(self, domain):
        try:
            return socket.gethostbyname(domain)
        except Exception:
            return None

    def _ip_in_cidrs(self, ip_str, cidr_list):
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except Exception:
            return False
        for cidr in cidr_list:
            try:
                net = ipaddress.ip_network(cidr, strict=False)
                if ip_obj in net:
                    return True
            except Exception:
                continue
        return False

    def _check_ip_trust(self, domain, ip_str):
        if not ip_str or not domain:
            return ("UNKNOWN", "No IP/domain")

        if domain in self.KNOWN_GOOD_CIDRS:
            cidrs = self.KNOWN_GOOD_CIDRS[domain]
            if self._ip_in_cidrs(ip_str, cidrs):
                return ("GOOD", f"{domain} → {ip_str} matches known-good range")
            else:
                return ("MISMATCH", f"{domain} → {ip_str} NOT in known-good range")

        learned = self.trusted_ips.setdefault(domain, [])

        if not learned:
            learned.append(ip_str)
            self.memory.save()
            return ("LEARNED", f"{domain} → {ip_str} learned as trusted")

        if ip_str in learned:
            return ("GOOD", f"{domain} → {ip_str} matches trusted IP list")

        learned.append(ip_str)
        self.memory.save()
        return ("LEARNED", f"{domain} → {ip_str} added to trusted IP list")

    def evaluate(self):
        self.reasons = []
        if not self.latest_url:
            self.reasons = ["No active URL"]
            return 0

        url = self.latest_url.lower()
        score = 0

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

        domain = self._extract_domain(url)
        ip_str = self._resolve_ip(domain) if domain else None
        status, msg = self._check_ip_trust(domain, ip_str)

        if status == "GOOD":
            self.hud.push("info", f"GOOD: {msg}")
        elif status == "LEARNED":
            self.hud.push("info", f"LEARNED: {msg}")
        elif status == "MISMATCH":
            self.hud.push("hazard", f"WARNING: {msg}")
            score += 40

        self.reasons.append(f"IP status: {status}")
        if ip_str:
            self.reasons.append(f"Resolved IP: {ip_str}")
        if domain:
            self.reasons.append(f"Domain: {domain}")

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
# ML: ENEMY MOVEMENT PREDICTOR
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
# PREDICTIVE ENGINE (MULTI-HORIZON)
# =========================

class PredictiveEngine:
    def __init__(self):
        pass

    def exponential_smooth(self, series, alpha=0.3):
        if not series:
            return 0.0
        s = series[0]
        for x in series[1:]:
            s = alpha * x + (1 - alpha) * s
        return s

    def linear_trend_forecast(self, series, horizons, step=1.0, clamp=(0, 100)):
        if not series:
            return {h: 0 for h in horizons}
        if len(series) == 1:
            base = series[-1]
            return {h: max(clamp[0], min(clamp[1], base)) for h in horizons}

        times = list(range(len(series)))
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(series)
        sum_xy = sum(x * y for x, y in zip(times, series))
        sum_x2 = sum(x * x for x in times)
        denom = n * sum_x2 - sum_x * sum_x
        if denom != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denom
        else:
            slope = 0.0

        last = series[-1]
        out = {}
        for h in horizons:
            steps_ahead = h / step
            pred = last + slope * steps_ahead
            pred = max(clamp[0], min(clamp[1], pred))
            out[h] = pred
        return out

# =========================
# GAME COGNITION
# =========================

class GameCognition:
    def __init__(self, hud, memory, recorder, predictor, predictive_engine):
        self.minimap = None
        self.hud = hud
        self.memory = memory
        self.recorder = recorder
        self.predictor = predictor
        self.predictive_engine = predictive_engine

        self.cones = "FRONT:0 RIGHT:0 BACK:0 LEFT:0"
        self.timeline = []
        self.enemy_history = {}
        self.clusters = []
        self.squad_command = ""
        self.heatmap = None
        self.sound_summary = ""
        self.predicted_positions = []
        self.danger_history = []
        self.future_forecasts = {}  # horizon -> value

    def update_minimap(self, data):
        self.minimap = data
        self.memory.mark_game_seen()
        self.recorder.record_snapshot(data)
        self._update_enemy_history()
        self._update_cones()
        self._update_timeline_and_prediction()
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

    def _update_timeline_and_prediction(self):
        enemies = self.minimap.get("enemies", []) if self.minimap else []
        base_score = min(100, len(enemies) * 10)
        self.timeline.append((time.strftime("%H:%M:%S"), base_score))
        if len(self.timeline) > 30:
            self.timeline.pop(0)

        self.danger_history.append(base_score)
        if len(self.danger_history) > 100:
            self.danger_history.pop(0)

        self.memory.record_danger_peak(base_score)

        if len(self.danger_history) >= 3:
            horizons = (5, 10, 20)
            forecasts = self.predictive_engine.linear_trend_forecast(self.danger_history[-20:], horizons)
            self.future_forecasts = {h: int(forecasts[h]) for h in horizons}
            self.memory.set_future_danger(self.future_forecasts.get(5, 0))
            self.memory.set_future_danger_multi(self.future_forecasts)

            recent = self.danger_history[-5:]
            last_now = recent[-1]
            future_5 = self.future_forecasts.get(5, 0)
            if future_5 >= 70 and last_now < 70:
                self.hud.push("predict", f"Danger spike expected in ~5s (trend rising to {future_5})")

            if len(self.danger_history) >= 10:
                mean = statistics.mean(self.danger_history[-10:])
                stdev = statistics.pstdev(self.danger_history[-10:])
                if stdev > 0 and base_score > mean + 2 * stdev:
                    self.hud.push("hazard", f"Anomalous danger spike detected (current {base_score}, mean {int(mean)})")

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
        if not self.minimap:
            return 0
        enemies = self.minimap.get("enemies", [])
        return min(100, len(enemies) * 10)

    def simulate_future_danger(self, enemies_count, horizon_seconds):
        base_series = list(self.danger_history[-20:])
        hypothetical = min(100, enemies_count * 10)
        base_series.append(hypothetical)
        forecasts = self.predictive_engine.linear_trend_forecast(base_series, (horizon_seconds,))
        return int(forecasts.get(horizon_seconds, 0))

# =========================
# PROFILES
# =========================

class ProfileManager:
    PROFILES = ["stealth", "combat", "analysis"]

    def __init__(self, memory):
        self.memory = memory

    def get_current(self):
        return self.memory.get_profile()

    def autoselect_profile(self, mode, threat_level):
        style = self.memory.data["profiles"].setdefault("style", {"aggressive": 0, "cautious": 0})
        aggressive = style.get("aggressive", 0)
        cautious = style.get("cautious", 0)

        profile = "stealth"
        if mode == "game":
            if threat_level >= 60:
                if aggressive > cautious:
                    profile = "combat"
                else:
                    profile = "stealth"
            else:
                if cautious > aggressive:
                    profile = "stealth"
                else:
                    profile = "combat"
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
            "future_danger_multi": game_cog.future_forecasts,
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
# SYSTEM BACKBONE MONITOR
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
# GAME SERVER TRUST WATCHER
# =========================

class GameServerWatcher:
    def __init__(self, memory, hud, orchestrator, interval=2.0):
        self.memory = memory
        self.hud = hud
        self.orch = orchestrator
        self.interval = interval
        self.running = HAVE_PYWIN32
        if not HAVE_PYWIN32:
            self.orch.log("[GAME SERVER] pywin32/psutil not available, server watcher disabled")

    def start(self):
        if not self.running:
            return
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()
        self.orch.log("[GAME SERVER] Watcher started")

    def loop(self):
        while self.running:
            try:
                self._scan_servers()
            except Exception as e:
                self.orch.log(f"[GAME SERVER ERROR] {e}")
            time.sleep(self.interval)

    def _scan_servers(self):
        trusted = self.memory.data["game"].setdefault("trusted_servers", {})
        last_status = self.memory.data["game"].get("server_status", "UNKNOWN")

        for proc in psutil.process_iter(attrs=["name", "pid"]):
            name = (proc.info["name"] or "").lower()
            if name in SystemBackboneMonitor.GAME_PROCESSES:
                try:
                    conns = proc.connections(kind="inet")
                except Exception:
                    continue
                remote_ips = [c.raddr.ip for c in conns if c.raddr]
                if not remote_ips:
                    continue
                ip = remote_ips[0]
                status, msg = self._check_server_trust(name, ip, trusted)
                self.memory.data["game"]["server_status"] = status
                self.memory.data["game"]["last_server_ip"] = ip
                self.memory.data["game"]["last_server_name"] = name

                if status != last_status:
                    if status == "GOOD":
                        self.hud.push("server", f"GOOD: {name} server {ip} matches trusted")
                    elif status == "LEARNED":
                        self.hud.push("server", f"LEARNED: {name} server {ip} stored as trusted")
                    elif status == "MISMATCH":
                        self.hud.push("hazard", f"WARNING: {name} server changed to {ip}")
                self.memory.save()
                return

        self.memory.data["game"]["server_status"] = "UNKNOWN"
        self.memory.data["game"]["last_server_ip"] = None
        self.memory.data["game"]["last_server_name"] = None

    def _check_server_trust(self, game_name, ip, trusted):
        ips = trusted.get(game_name, [])
        if not ips:
            trusted.setdefault(game_name, []).append(ip)
            return ("LEARNED", f"{game_name} → {ip} learned")
        if ip in ips:
            return ("GOOD", f"{game_name} → {ip} matches trusted")
        else:
            return ("MISMATCH", f"{game_name} → {ip} differs from trusted")

# =========================
# NETWORK ANOMALY MONITOR
# =========================

class NetworkAnomalyMonitor:
    BASELINE_DURATION = 3600
    SCAN_INTERVAL = 60

    def __init__(self, memory, hud, orchestrator):
        self.memory = memory
        self.hud = hud
        self.orch = orchestrator
        self.running = True

        net = self.memory.data.setdefault("network", {})
        net.setdefault("baseline", {})
        net.setdefault("anomalies", [])
        net.setdefault("baseline_start_ts", 0)
        net.setdefault("baseline_complete", False)
        net.setdefault("cone_counts", {"FRONT": 0, "RIGHT": 0, "BACK": 0, "LEFT": 0})
        net.setdefault("last_scan_ts", 0)

        if net["baseline_start_ts"] == 0:
            net["baseline_start_ts"] = time.time()

    def start(self):
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()
        self.orch.log("[NETWORK] Anomaly monitor started")

    def loop(self):
        while self.running:
            try:
                self.scan_once()
            except Exception as e:
                self.orch.log(f"[NETWORK ERROR] {e}")
            time.sleep(self.SCAN_INTERVAL)

    def scan_once(self):
        net = self.memory.data["network"]
        now = time.time()

        if not net["baseline_complete"]:
            if now - net["baseline_start_ts"] >= self.BASELINE_DURATION:
                net["baseline_complete"] = True
                self.orch.log("[NETWORK] Baseline learning complete")
                self.hud.push("info", "Network baseline learning complete")
        net["cone_counts"] = {"FRONT": 0, "RIGHT": 0, "BACK": 0, "LEFT": 0}

        try:
            conns = psutil.net_connections(kind="inet")
        except Exception:
            return

        for c in conns:
            if not c.raddr:
                continue
            ip = c.raddr.ip
            if not ip:
                continue

            country = lookup_country_for_ip(ip)
            baseline = net["baseline"]

            if not net["baseline_complete"]:
                if ip not in baseline:
                    baseline[ip] = {
                        "first_seen": now,
                        "country": country
                    }
                continue

            if ip in baseline:
                continue

            reason = "New outbound IP not in baseline"
            anomaly = {
                "ip": ip,
                "country": country,
                "reason": reason,
                "time": time.strftime("%H:%M:%S")
            }
            net["anomalies"].append(anomaly)
            if len(net["anomalies"]) > 200:
                net["anomalies"].pop(0)

            direction = self._map_country_to_cone(country)
            net["cone_counts"][direction] += 1

            self.hud.push(
                "hazard",
                f"Network anomaly: {ip} ({country}) – {reason} in {direction} cone"
            )
            self.orch.log(f"[NETWORK] Anomaly {ip} ({country}) → {direction}")

        net["last_scan_ts"] = now
        self.memory.save()

    def _map_country_to_cone(self, country: str) -> str:
        if country == "UNKNOWN":
            return "LEFT"
        return "FRONT"

# =========================
# ORCHESTRATOR
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
# MULTI-THREADED HTTP SERVER
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
# WHAT-IF SIMULATION SERVER
# =========================

def start_whatif_server(game_cognition, orchestrator):
    class WhatIfHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/whatif":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                mode = data.get("mode", "game")
                if mode != "game":
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Unsupported mode")
                    return
                enemies = int(data.get("enemies", 0))
                horizon = int(data.get("horizon", 5))
                horizon = max(1, min(60, horizon))
                predicted = game_cognition.simulate_future_danger(enemies, horizon)
                resp = {"mode": "game", "enemies": enemies, "horizon": horizon, "predicted_danger": predicted}
                orchestrator.log(f"[WHATIF] Game enemies={enemies} horizon={horizon}s → {predicted}")
                out = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)
            except Exception as e:
                orchestrator.log(f"[WHATIF ERROR] {e}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Bad Request")

        def log_message(self, format, *args):
            return

    def run():
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 8791), WhatIfHandler)
            orchestrator.log("[WHATIF] Listening on http://127.0.0.1:8791/whatif")
            server.serve_forever()
        except OSError as e:
            orchestrator.log(f"[WHATIF] Failed to bind: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()

# =========================
# GAME HELPER CLIENT
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
# FILESYSTEM HOVER HELPER
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
# BROWSER URL WATCHER (DEVTOOLS-BASED, OPTIONAL)
# =========================

class BrowserURLWatcher:
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

        if proc_name not in ("chrome.exe", "msedge.exe", "brave.exe", "vivaldi.exe", "opera.exe"):
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
# WEB SURFER (WINDOW-TITLE MODE ONLY)
# =========================

class WebSurfer:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.last_url = None
        self.orch.log("[WEBSURFER] Using window-title mode only (no Selenium)")

    def send_to_mighty_mouse(self, url):
        try:
            data = json.dumps({"url": url}).encode("utf-8")
            req = urlrequest.Request(
                "http://127.0.0.1:8787/url",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            urlrequest.urlopen(req, timeout=0.3)
        except Exception:
            pass

    def get_url_window_title(self):
        if not HAVE_PYWIN32:
            return None
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return None

            if "http://" in title.lower() or "https://" in title.lower():
                parts = title.split(" - ")
                for p in parts:
                    p = p.strip()
                    if p.startswith("http://") or p.startswith("https://"):
                        return p
            return None
        except Exception:
            return None

    def poll(self):
        url = self.get_url_window_title()
        if url and url != self.last_url:
            self.last_url = url
            self.orch.log(f"[WEBSURFER] URL detected → {url}")
            self.send_to_mighty_mouse(url)

# =========================
# GUI COCKPIT
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

        self.setWindowTitle("Mighty Mouse – Autonomous Tactical HUD (Predictive + Trust + Network)")
        self.resize(1200, 900)

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
        self.web_status_label = QLabel("Web IP Status: (none)")
        main_layout.addWidget(self.web_url_label)
        main_layout.addWidget(self.web_status_label)

        self.web_reasons_view = QTextEdit()
        self.web_reasons_view.setReadOnly(True)
        self.web_reasons_view.setStyleSheet("""
            QTextEdit {
                background-color: #101010;
                color: #DDDDDD;
                border: 1px solid #444444;
                padding: 4px;
            }
        """)
        self.web_reasons_view.setPlainText(
            "Waiting for web URL...\n"
            "Use DevTools port 9222, POST /url on 8787, or WebSurfer integration."
        )
        main_layout.addWidget(self.web_reasons_view)

        self.game_server_label = QLabel("Game Server Status: UNKNOWN (no server)")
        main_layout.addWidget(self.game_server_label)

        self.tactical_label = QLabel("Tactical Feed:")
        self.tactical_view = QTextEdit()
        self.tactical_view.setReadOnly(True)
        self.tactical_view.setStyleSheet("""
            QTextEdit {
                background-color: #101010;
                color: #DDDDDD;
                border: 1px solid #444444;
                padding: 4px;
            }
        """)
        main_layout.addWidget(self.tactical_label)
        main_layout.addWidget(self.tactical_view)

        self.server_trust_label = QLabel("Server Trust Panel:")
        main_layout.addWidget(self.server_trust_label)

        self.server_trust_box = QTextEdit()
        self.server_trust_box.setReadOnly(True)
        self.server_trust_box.setStyleSheet("""
            QTextEdit {
                background-color: #101010;
                color: #FFFFFF;
                border: 1px solid #00FF00;
                padding: 4px;
            }
        """)
        main_layout.addWidget(self.server_trust_box)

        self.net_cone_label = QLabel("Network Anomaly Cone (only threats):")
        self.net_cone_view = QLabel("FRONT:0 RIGHT:0 BACK:0 LEFT:0")
        main_layout.addWidget(self.net_cone_label)
        main_layout.addWidget(self.net_cone_view)

        self.timeline_label = QLabel("Danger Timeline:")
        self.timeline_view = QTextEdit()
        self.timeline_view.setReadOnly(True)
        self.timeline_view.setStyleSheet("""
            QTextEdit {
                background-color: #101010;
                color: #DDDDDD;
                border: 1px solid #444444;
                padding: 4px;
            }
        """)
        self.timeline_view.setPlainText("[No danger history yet]")
        main_layout.addWidget(self.timeline_label)
        main_layout.addWidget(self.timeline_view)

        self.future_label = QLabel("Predicted Future Danger (5s): 0")
        main_layout.addWidget(self.future_label)

        self.future_multi_label = QLabel("Multi-Horizon Danger (5/10/20s): 0 / 0 / 0")
        main_layout.addWidget(self.future_multi_label)

        self.log_label = QLabel("System Log:")
        main_layout.addWidget(self.log_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("""
            QTextEdit {
                background-color: #050505;
                color: #BBBBBB;
                border: 1px solid #333333;
                padding: 4px;
            }
        """)
        self.log_view.setPlainText("[Log will appear here]")
        main_layout.addWidget(self.log_view)

        self.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(250)

    def _color_for_status(self, status):
        if status == "GOOD":
            return "#00FF00"
        if status == "LEARNED":
            return "#00AAFF"
        if status == "MISMATCH":
            return "#FF3333"
        return "#FFFFFF"

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
            score = self.web.evaluate()

            self.web_url_label.setText(f"Web URL: {self.web.latest_url}")

            domain = None
            try:
                domain = self.web._extract_domain(self.web.latest_url)
            except Exception:
                domain = None

            ip_str = None
            if domain:
                try:
                    ip_str = socket.gethostbyname(domain)
                except Exception:
                    ip_str = "Unknown"

            status_line = next((r for r in self.web.reasons if r.startswith("IP status:")), None)
            if status_line:
                status = status_line.replace("IP status:", "").strip()
                self.web_status_label.setText(f"Web IP Status: {status} (score {score})")
            else:
                status = "UNKNOWN"
                self.web_status_label.setText(f"Web IP Status: UNKNOWN (score {score})")

            web_details = (
                f"Domain: {domain}\n"
                f"Resolved IP: {ip_str}\n"
                f"Status: {status}\n"
                f"Score: {score}\n\n"
            )

            self.web_reasons_view.setPlainText(web_details + "\n".join(self.web.reasons))
        else:
            self.web_url_label.setText("Web URL: (none)")
            self.web_status_label.setText("Web IP Status: (no URL)")
            self.web_reasons_view.setPlainText(
                "Waiting for web URL...\n"
                "Use DevTools port 9222, POST /url on 8787, or WebSurfer integration."
            )

        server_status = self.memory.data["game"].get("server_status", "UNKNOWN")
        last_ip = self.memory.data["game"].get("last_server_ip", None)
        last_name = self.memory.data["game"].get("last_server_name", None)

        if last_ip:
            self.game_server_label.setText(f"Game Server Status: {server_status} ({last_ip})")
        else:
            self.game_server_label.setText(f"Game Server Status: {server_status} (no server)")

        server_ip = last_ip if last_ip else "None"
        game_name = last_name if last_name else "(none)"

        color = self._color_for_status(server_status)
        border_color = color
        self.server_trust_box.setStyleSheet(f"""
            QTextEdit {{
                background-color: #101010;
                color: {color};
                border: 1px solid {border_color};
                padding: 4px;
            }}
        """)

        if server_status == "UNKNOWN":
            text = "⛨ No active game server detected\n"
        else:
            reason = ""
            if server_status == "GOOD":
                reason = "Matches trusted baseline"
            elif server_status == "LEARNED":
                reason = "First time seen, stored as trusted"
            elif server_status == "MISMATCH":
                reason = "Does not match trusted baseline"

            text = (
                f"⛨ {server_status} — {server_ip}\n"
                f"{reason}\n"
                f"Game: {game_name}\n"
            )

        self.server_trust_box.setPlainText(text)

        self.tactical_view.setHtml(self.hud.get_html())
        self.tactical_view.verticalScrollBar().setValue(
            self.tactical_view.verticalScrollBar().maximum()
        )

        net = self.memory.data.get("network", {})
        cone = net.get("cone_counts", {"FRONT": 0, "RIGHT": 0, "BACK": 0, "LEFT": 0})
        self.net_cone_view.setText(
            f"FRONT:{cone.get('FRONT', 0)} RIGHT:{cone.get('RIGHT', 0)} "
            f"BACK:{cone.get('BACK', 0)} LEFT:{cone.get('LEFT', 0)}"
        )

        if self.game.timeline:
            lines = [f"{t} → {s}" for t, s in self.game.timeline]
            self.timeline_view.setPlainText("\n".join(lines))
        else:
            self.timeline_view.setPlainText("[No danger history yet]")

        future = self.memory.data["game"]["future_danger_prediction"]
        self.future_label.setText(f"Predicted Future Danger (5s): {future}")

        multi = self.memory.data["game"].get("future_danger_multi", {})
        f5 = multi.get(5, 0)
        f10 = multi.get(10, 0)
        f20 = multi.get(20, 0)
        self.future_multi_label.setText(f"Multi-Horizon Danger (5/10/20s): {f5} / {f10} / {f20}")

        if self.orch.logs:
            self.log_view.setPlainText("\n".join(self.orch.logs[-80:]))
            self.log_view.verticalScrollBar().setValue(
                self.log_view.verticalScrollBar().maximum()
            )
        else:
            self.log_view.setPlainText("[No log entries yet]")

# =========================
# MAIN APPLICATION STARTUP
# =========================

def main():
    app = QApplication(sys.argv)

    bus = EventBus()
    memory = MemoryStore()
    hud = TacticalHUD()

    web = WebBrain(memory, hud)
    fs = FilesystemBrain(memory)

    recorder = EnemyMovementRecorder()
    predictor = EnemyMovementPredictor(hud)
    predictive_engine = PredictiveEngine()
    game = GameCognition(hud, memory, recorder, predictor, predictive_engine)

    profiles = ProfileManager(memory)
    cursor = CursorEngine(app)
    overlay_bridge = DirectXOverlayBridge()

    orch = Orchestrator(bus, memory, cursor, hud, web, fs, game, profiles, overlay_bridge)

    # START SERVERS
    start_url_server(web, orch)
    start_game_event_server(hud, orch)
    start_minimap_server(game, orch)
    start_fs_hover_server(fs, orch)
    start_whatif_server(game, orch)

    # BACKBONE MONITOR
    backbone = SystemBackboneMonitor(orch)
    backbone.start()

    # GAME SERVER WATCHER
    server_watcher = GameServerWatcher(memory, hud, orch)
    server_watcher.start()

    # NETWORK ANOMALY MONITOR
    netmon = NetworkAnomalyMonitor(memory, hud, orch)
    netmon.start()

    # FILESYSTEM HOVER HELPER
    fs_hover = FilesystemHoverHelper(fs, orch)
    fs_hover.start()

    # DEVTOOLS BROWSER WATCHER (OPTIONAL)
    browser_watch = BrowserURLWatcher(web, orch)
    browser_watch.start()

    # WEBSURFER (WINDOW-TITLE MODE ONLY)
    web_surfer = WebSurfer(orch)

    def web_surfer_loop():
        while True:
            try:
                web_surfer.poll()
            except Exception as e:
                orch.log(f"[WEBSURFER ERROR] {e}")
            time.sleep(0.5)

    t_ws = threading.Thread(target=web_surfer_loop, daemon=True)
    t_ws.start()
    orch.log("[WEBSURFER] Background loop started")

    # ORCHESTRATOR TICK LOOP
    def tick_loop():
        while True:
            try:
                orch.tick()
            except Exception as e:
                orch.log(f"[TICK ERROR] {e}")
            time.sleep(0.25)

    t_tick = threading.Thread(target=tick_loop, daemon=True)
    t_tick.start()
    orch.log("[TICK] Loop started")

    # GUI
    gui = CockpitGUI(orch, hud, web, fs, game, profiles, memory)
    gui.show()

    orch.log("[SYSTEM] Mighty Mouse fully initialized (Predictive + What-If)")
    sys.exit(app.exec_())

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    main()

