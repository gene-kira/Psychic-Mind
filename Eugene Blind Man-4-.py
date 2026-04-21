#!/usr/bin/env python3
# OmniQueen Swarm Guard + PySide6 Cockpit (Windows-focused, swarm, manual override)

import sys
import os
import json
import time
import platform
import threading
import subprocess
from datetime import datetime, timedelta
from collections import deque
from math import sqrt

# ---------------- autoloader ----------------

REQUIRED_LIBS = ["psutil", "yaml", "PySide6"]

def autoload(lib_name):
    try:
        return __import__(lib_name)
    except ImportError:
        try:
            print(f"[AUTOLOADER] Installing missing library: {lib_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib_name])
            return __import__(lib_name)
        except Exception as e:
            print(f"[AUTOLOADER] Failed to install {lib_name}: {e}")
            raise

psutil = autoload("psutil")
yaml = autoload("yaml")
PySide6 = autoload("PySide6")
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QListWidget, QListWidgetItem, QGroupBox
)
from PySide6.QtCore import QTimer, Qt

# ---------------- paths & constants ----------------

CONFIG_PATH = "omniqueen_config.yml"
STATE_PATH = "omniqueen_state.json"
REBOOT_MEMORY_PATH = "omniqueen_reboot_memory.json"
LOG_PATH = "omniqueen_events.log"

ALTERED_STATES = ["CALM", "ALERT", "WAR"]

# ---------------- core utils ----------------

def log_event(level, component, message, extra=None):
    event = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "component": component,
        "message": message,
        "extra": extra or {}
    }
    line = json.dumps(event, sort_keys=True)
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def sha256_file(path, chunk_size=65536):
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception as e:
        log_event("ERROR", "io", f"Failed to save {path}", {"error": str(e)})


def load_config():
    if not os.path.exists(CONFIG_PATH):
        default = {
            "node_id": platform.node(),
            "role": "node",  # "node" or "queen"
            "swarm": {
                "enabled": True,
                "swarm_dir": "swarm_state",
                "queen_orders_file": "queen_orders.json",
                "publish_interval": 15,
                "queen_interval": 20
            },
            "poll_interval_seconds": 5,
            "ai_anomaly": {
                "enabled": True,
                "sigma_multiplier": 2.0  # more aggressive
            },
            "file_integrity": {
                "enabled": True,
                "paths": [
                    r"C:\Windows\System32\drivers\etc\hosts"
                    if os.name == "nt" else "/etc/hosts"
                ]
            },
            "process_watch": {
                "enabled": True,
                "suspicious_keywords": [
                    "miner", "rat", "keylog", "hack", "bot",
                    "crypt", "stealer", "mimikatz", "cobalt"
                ],
                "rolling_window": 600
            },
            "net_watch": {
                "enabled": True,
                "rolling_window": 300
            },
            "auto_response": {
                "kill_extreme_processes": True,
                "firewall_block_extreme_ips": True,
                "log_only": False
            },
            "water_physics": {
                "enabled": True,
                "flow_window": 120
            },
            "manual_override": {
                "enabled": True,
                "panic_mode": False
            }
        }
        os.makedirs(default["swarm"]["swarm_dir"], exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(default, f)
        log_event("INFO", "config", f"Default config created at {CONFIG_PATH}")
        return default

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    swarm_dir = cfg.get("swarm", {}).get("swarm_dir", "swarm_state")
    os.makedirs(swarm_dir, exist_ok=True)
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
    except Exception as e:
        log_event("ERROR", "config", "Failed to save config", {"error": str(e)})


def load_state():
    return load_json(STATE_PATH, {})


def save_state(state):
    save_json(STATE_PATH, state)


def load_reboot_memory():
    return load_json(REBOOT_MEMORY_PATH, {
        "boot_count": 0,
        "last_boot_ts": None,
        "last_state": None,
        "environment_fingerprints": [],
        "threat_history": []
    })


def save_reboot_memory(mem):
    save_json(REBOOT_MEMORY_PATH, mem)

# ---------------- stats helpers ----------------

def update_rolling_stats(stats, value):
    if stats is None:
        stats = {"count": 0, "mean": 0.0, "m2": 0.0}
    count = stats["count"] + 1
    delta = value - stats["mean"]
    mean = stats["mean"] + delta / count
    delta2 = value - mean
    m2 = stats["m2"] + delta * delta2
    stats["count"] = count
    stats["mean"] = mean
    stats["m2"] = m2
    return stats


def stats_sigma(stats):
    if not stats or stats["count"] < 2:
        return 0.0
    variance = stats["m2"] / (stats["count"] - 1)
    return sqrt(max(variance, 0.0))

# ---------------- probes ----------------

def probe_environment():
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total": psutil.virtual_memory().total,
        "boot_time": psutil.boot_time(),
    }
    try:
        info["process_count"] = len(list(psutil.process_iter()))
    except Exception:
        info["process_count"] = None
    try:
        info["connection_count"] = len(psutil.net_connections(kind="inet"))
    except Exception:
        info["connection_count"] = None
    return info

# ---------------- water data physics ----------------

class WaterPhysicsEngine:
    def __init__(self, cfg_section):
        self.enabled = cfg_section.get("enabled", True) if cfg_section else True
        self.flow_window = cfg_section.get("flow_window", 120) if cfg_section else 120
        self.event_stream = deque()
        self.flow_stats = None

    def add_event(self, weight=1.0):
        if not self.enabled:
            return
        now = datetime.utcnow()
        self.event_stream.append((now, float(weight)))
        self._purge_old(now)

    def _purge_old(self, now):
        cutoff = now - timedelta(seconds=self.flow_window)
        while self.event_stream and self.event_stream[0][0] < cutoff:
            self.event_stream.popleft()

    def compute_flow(self):
        if not self.enabled:
            return {"flow_rate": 0.0, "turbulence": 0.0, "pressure": 0.0}
        now = datetime.utcnow()
        self._purge_old(now)
        if not self.event_stream:
            return {"flow_rate": 0.0, "turbulence": 0.0, "pressure": 0.0}
        total_weight = sum(w for _, w in self.event_stream)
        flow_rate = total_weight / float(self.flow_window)
        weights = [w for _, w in self.event_stream]
        mean_w = sum(weights) / len(weights)
        var = sum((w - mean_w) ** 2 for w in weights) / max(len(weights) - 1, 1)
        turbulence = sqrt(max(var, 0.0))
        pressure = flow_rate * (1.0 + turbulence)
        self.flow_stats = update_rolling_stats(self.flow_stats, pressure)
        return {"flow_rate": flow_rate, "turbulence": turbulence, "pressure": pressure}

# ---------------- borg memory ----------------

class BorgMemory:
    def __init__(self, state, cfg, node_id):
        self.state = state
        self.proc_stats = state.get("proc_stats", {})
        self.ip_stats = state.get("ip_stats", {})
        self.file_hashes = state.get("file_hashes", {})
        self.proc_count_stats = state.get("proc_count_stats", None)
        self.conn_count_stats = state.get("conn_count_stats", None)
        self.altered_state = state.get("altered_state", "CALM")
        self.threat_pressure_stats = state.get("threat_pressure_stats", None)
        self.suspect_ips = set(state.get("suspect_ips", []))
        self.suspect_procs = set(state.get("suspect_procs", []))
        self.node_id = node_id

        self.ai_cfg = cfg.get("ai_anomaly", {}) if cfg else {}
        self.water_engine = WaterPhysicsEngine(cfg.get("water_physics", {}))

    def touch_process(self, name):
        now = datetime.utcnow().timestamp()
        rec = self.proc_stats.get(name, {"count": 0, "last_seen": 0})
        rec["count"] += 1
        rec["last_seen"] = now
        self.proc_stats[name] = rec

    def touch_ip(self, ip):
        now = datetime.utcnow().timestamp()
        rec = self.ip_stats.get(ip, {"count": 0, "last_seen": 0})
        rec["count"] += 1
        rec["last_seen"] = now
        self.ip_stats[ip] = rec

    def set_file_hash(self, path, digest):
        self.file_hashes[path] = digest

    def get_file_hash(self, path):
        return self.file_hashes.get(path)

    def update_proc_count_baseline(self, count):
        self.proc_count_stats = update_rolling_stats(self.proc_count_stats, float(count))

    def update_conn_count_baseline(self, count):
        self.conn_count_stats = update_rolling_stats(self.conn_count_stats, float(count))

    def score_process(self, name, cmdline, suspicious_keywords):
        rec = self.proc_stats.get(name, {"count": 0, "last_seen": 0})
        count = rec["count"]
        rarity = 1.0 if count == 0 else 1.0 / (1.0 + count)

        keyword_boost = 0.0
        low_cmd = (cmdline or "").lower()
        low_name = (name or "").lower()
        for kw in suspicious_keywords:
            if kw in low_name or kw in low_cmd:
                keyword_boost += 0.4

        env_boost = 0.0
        if self.proc_count_stats and self.proc_count_stats["count"] > 5:
            sigma = stats_sigma(self.proc_count_stats)
            if sigma > 0:
                env_boost = min(0.5, sigma / (self.ai_cfg.get("sigma_multiplier", 2.0) * 6.0))

        flow = self.water_engine.compute_flow()
        pressure = flow["pressure"]
        self.threat_pressure_stats = update_rolling_stats(self.threat_pressure_stats, pressure)
        pressure_boost = min(0.5, pressure * 3.0)

        score = min(1.0, rarity * 0.4 + keyword_boost + env_boost + pressure_boost)
        return score

    def score_ip(self, ip):
        rec = self.ip_stats.get(ip, {"count": 0, "last_seen": 0})
        count = rec["count"]
        rarity = 1.0 if count == 0 else 1.0 / (1.0 + count)

        env_boost = 0.0
        if self.conn_count_stats and self.conn_count_stats["count"] > 5:
            sigma = stats_sigma(self.conn_count_stats)
            if sigma > 0:
                env_boost = min(0.5, sigma / (self.ai_cfg.get("sigma_multiplier", 2.0) * 6.0))

        flow = self.water_engine.compute_flow()
        pressure = flow["pressure"]
        self.threat_pressure_stats = update_rolling_stats(self.threat_pressure_stats, pressure)
        pressure_boost = min(0.5, pressure * 3.0)

        return min(1.0, rarity * 0.5 + env_boost + pressure_boost)

    def update_altered_state(self):
        sigma = stats_sigma(self.threat_pressure_stats) if self.threat_pressure_stats else 0.0
        mean = self.threat_pressure_stats["mean"] if self.threat_pressure_stats else 0.0
        pressure_level = mean + sigma

        prev_state = self.altered_state
        if pressure_level < 0.05:
            self.altered_state = "CALM"
        elif pressure_level < 0.2:
            self.altered_state = "ALERT"
        else:
            self.altered_state = "WAR"

        if self.altered_state != prev_state:
            log_event("INFO", "altered_state", "State transition", {
                "from": prev_state,
                "to": self.altered_state,
                "pressure_level": pressure_level
            })

    def register_threat_event(self, weight=1.0):
        self.water_engine.add_event(weight)
        self.update_altered_state()

    def mark_suspect_ip(self, ip):
        self.suspect_ips.add(ip)

    def mark_suspect_proc(self, name):
        self.suspect_procs.add(name)

    def flush_back(self):
        self.state["proc_stats"] = self.proc_stats
        self.state["ip_stats"] = self.ip_stats
        self.state["file_hashes"] = self.file_hashes
        self.state["proc_count_stats"] = self.proc_count_stats
        self.state["conn_count_stats"] = self.conn_count_stats
        self.state["altered_state"] = self.altered_state
        self.state["threat_pressure_stats"] = self.threat_pressure_stats
        self.state["suspect_ips"] = sorted(self.suspect_ips)
        self.state["suspect_procs"] = sorted(self.suspect_procs)

# ---------------- file integrity ----------------

def init_file_baseline(cfg, borg: BorgMemory):
    fi_cfg = cfg.get("file_integrity", {})
    if not fi_cfg.get("enabled", True):
        return

    for path in fi_cfg.get("paths", []):
        digest = sha256_file(path)
        if digest is None:
            log_event("WARN", "file_integrity", "Cannot hash file", {"path": path})
            continue
        if borg.get_file_hash(path) is None:
            borg.set_file_hash(path, digest)
            log_event("INFO", "file_integrity", "Baseline hash recorded", {"path": path, "hash": digest})


def check_file_integrity(cfg, borg: BorgMemory):
    fi_cfg = cfg.get("file_integrity", {})
    if not fi_cfg.get("enabled", True):
        return

    for path in fi_cfg.get("paths", []):
        old_hash = borg.get_file_hash(path)
        new_hash = sha256_file(path)
        if new_hash is None:
            continue
        if old_hash is None:
            borg.set_file_hash(path, new_hash)
            log_event("WARN", "file_integrity", "New file added to baseline", {"path": path, "hash": new_hash})
            continue
        if new_hash != old_hash:
            log_event("ALERT", "file_integrity", "File hash changed", {
                "path": path,
                "old_hash": old_hash,
                "new_hash": new_hash
            })
            borg.set_file_hash(path, new_hash)
            borg.register_threat_event(weight=2.0)

# ---------------- process watch ----------------

def describe_process(p):
    try:
        return {
            "pid": p.pid,
            "name": p.name(),
            "exe": p.exe() if p.exe() else None,
            "cmdline": " ".join(p.cmdline()) if p.cmdline() else "",
            "username": p.username()
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"pid": p.pid, "name": "unknown"}


def process_watch_loop(cfg_ref, state, borg: BorgMemory, stop_event):
    seen_pids = set()

    while not stop_event.is_set():
        cfg = cfg_ref["cfg"]
        pw_cfg = cfg.get("process_watch", {})
        ai_cfg = cfg.get("ai_anomaly", {})
        auto_cfg = cfg.get("auto_response", {})
        manual_cfg = cfg.get("manual_override", {})

        if not pw_cfg.get("enabled", True):
            time.sleep(3)
            continue

        suspicious_keywords = [k.lower() for k in pw_cfg.get("suspicious_keywords", [])]

        try:
            procs = list(psutil.process_iter(["pid", "name", "cmdline"]))
        except Exception as e:
            log_event("ERROR", "process_watch", "Failed to enumerate processes", {"error": str(e)})
            time.sleep(5)
            continue

        total_proc_count = len(procs)
        borg.update_proc_count_baseline(total_proc_count)

        for p in procs:
            pid = p.info["pid"]
            name = p.info.get("name") or ""
            cmdline = " ".join(p.info.get("cmdline") or [])

            borg.touch_process(name)

            if pid in seen_pids:
                continue
            seen_pids.add(pid)

            if not ai_cfg.get("enabled", True):
                continue

            score = borg.score_process(name, cmdline, suspicious_keywords)
            if score >= 0.85:
                desc = describe_process(p)
                log_event("ALERT", "process_ai", "High-score process detected", {
                    "score": score,
                    "process": desc,
                    "altered_state": borg.altered_state
                })
                borg.register_threat_event(weight=1.5)
                borg.mark_suspect_proc(name)

                if manual_cfg.get("panic_mode", False):
                    continue

                if auto_cfg.get("kill_extreme_processes", False) and borg.altered_state in ["ALERT", "WAR"]:
                    try:
                        p.terminate()
                        log_event("INFO", "process_ai", "Process terminated", {"pid": pid})
                    except Exception as e:
                        log_event("ERROR", "process_ai", "Failed to terminate process", {
                            "pid": pid,
                            "error": str(e)
                        })

        time.sleep(3)

# ---------------- firewall ----------------

def firewall_block_ip(ip):
    system = platform.system().lower()
    try:
        if system == "windows":
            cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name=OmniQueenBlock_{ip}",
                "dir=in", "action=block", f"remoteip={ip}"
            ]
        elif system == "linux":
            cmd = ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]
        elif system == "darwin":
            log_event("INFO", "firewall", "macOS block requested (pf not auto-wired)", {"ip": ip})
            return
        else:
            log_event("WARN", "firewall", "Unknown OS for firewall block", {"ip": ip})
            return

        log_event("INFO", "firewall", "Executing firewall block", {"ip": ip, "cmd": " ".join(cmd)})
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log_event("ERROR", "firewall", "Failed to apply firewall rule", {"ip": ip, "error": str(e)})

# ---------------- network watch ----------------

def net_watch_loop(cfg_ref, state, borg: BorgMemory, stop_event):
    recent_conns = deque()

    while not stop_event.is_set():
        cfg = cfg_ref["cfg"]
        nw_cfg = cfg.get("net_watch", {})
        ai_cfg = cfg.get("ai_anomaly", {})
        auto_cfg = cfg.get("auto_response", {})
        manual_cfg = cfg.get("manual_override", {})

        if not nw_cfg.get("enabled", True):
            time.sleep(2)
            continue

        rolling_window = nw_cfg.get("rolling_window", 300)

        now = datetime.utcnow()
        try:
            conns = psutil.net_connections(kind="inet")
        except Exception as e:
            log_event("ERROR", "net_watch", "Failed to read connections", {"error": str(e)})
            time.sleep(5)
            continue

        borg.update_conn_count_baseline(len(conns))

        for c in conns:
            if not c.raddr:
                continue
            l_ip = c.laddr.ip
            r_ip = c.raddr.ip
            borg.touch_ip(r_ip)
            conn_tuple = (l_ip, c.laddr.port, r_ip, c.raddr.port, c.status)
            recent_conns.append((now, conn_tuple))

            if ai_cfg.get("enabled", True):
                ip_score = borg.score_ip(r_ip)
                if ip_score >= 0.85:
                    log_event("ALERT", "net_ai", "High-score remote IP", {
                        "ip": r_ip,
                        "score": ip_score,
                        "altered_state": borg.altered_state
                    })
                    borg.register_threat_event(weight=1.2)
                    borg.mark_suspect_ip(r_ip)

                    if manual_cfg.get("panic_mode", False):
                        continue

                    if auto_cfg.get("firewall_block_extreme_ips", False) and borg.altered_state in ["ALERT", "WAR"]:
                        firewall_block_ip(r_ip)

        cutoff = now - timedelta(seconds=rolling_window)
        while recent_conns and recent_conns[0][0] < cutoff:
            recent_conns.popleft()

        time.sleep(2)

# ---------------- swarm sync ----------------

def swarm_publish_state(cfg, borg: BorgMemory):
    swarm_cfg = cfg.get("swarm", {})
    if not swarm_cfg.get("enabled", True):
        return
    swarm_dir = swarm_cfg.get("swarm_dir", "swarm_state")
    node_id = cfg.get("node_id", platform.node())
    path = os.path.join(swarm_dir, f"{node_id}.json")
    snapshot = {
        "node_id": node_id,
        "ts": datetime.utcnow().isoformat() + "Z",
        "altered_state": borg.altered_state,
        "suspect_ips": sorted(borg.suspect_ips),
        "suspect_procs": sorted(borg.suspect_procs),
        "proc_stats_size": len(borg.proc_stats),
        "ip_stats_size": len(borg.ip_stats)
    }
    save_json(path, snapshot)


def swarm_fetch_orders(cfg):
    swarm_cfg = cfg.get("swarm", {})
    if not swarm_cfg.get("enabled", True):
        return {"global_suspect_ips": [], "global_suspect_procs": []}
    swarm_dir = swarm_cfg.get("swarm_dir", "swarm_state")
    orders_file = swarm_cfg.get("queen_orders_file", "queen_orders.json")
    path = os.path.join(swarm_dir, orders_file)
    return load_json(path, {"global_suspect_ips": [], "global_suspect_procs": []})


def queen_aggregate_swarm(cfg):
    swarm_cfg = cfg.get("swarm", {})
    if not swarm_cfg.get("enabled", True):
        return
    swarm_dir = swarm_cfg.get("swarm_dir", "swarm_state")
    orders_file = swarm_cfg.get("queen_orders_file", "queen_orders.json")
    global_ips = set()
    global_procs = set()

    for fname in os.listdir(swarm_dir):
        if not fname.endswith(".json"):
            continue
        if fname == orders_file:
            continue
        path = os.path.join(swarm_dir, fname)
        data = load_json(path, {})
        for ip in data.get("suspect_ips", []):
            global_ips.add(ip)
        for proc in data.get("suspect_procs", []):
            global_procs.add(proc)

    orders = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "global_suspect_ips": sorted(global_ips),
        "global_suspect_procs": sorted(global_procs)
    }
    save_json(os.path.join(swarm_dir, orders_file), orders)
    log_event("INFO", "queen", "Updated queen orders", {
        "ips": len(global_ips),
        "procs": len(global_procs)
    })


def queen_loop(cfg_ref, stop_event):
    while not stop_event.is_set():
        cfg = cfg_ref["cfg"]
        queen_aggregate_swarm(cfg)
        swarm_cfg = cfg.get("swarm", {})
        interval = swarm_cfg.get("queen_interval", 20)
        time.sleep(interval)


def swarm_node_orders_apply(cfg, borg: BorgMemory):
    orders = swarm_fetch_orders(cfg)
    global_ips = orders.get("global_suspect_ips", [])
    global_procs = orders.get("global_suspect_procs", [])
    for ip in global_ips:
        if ip not in borg.suspect_ips:
            log_event("INFO", "swarm", "Adopting swarm suspect IP", {"ip": ip})
            borg.suspect_ips.add(ip)
    for proc in global_procs:
        if proc not in borg.suspect_procs:
            log_event("INFO", "swarm", "Adopting swarm suspect process", {"name": proc})
            borg.suspect_procs.add(proc)

# ---------------- reboot memory ----------------

def update_reboot_memory_on_start(reboot_mem, env_fingerprint, borg_state_snapshot):
    reboot_mem["boot_count"] = reboot_mem.get("boot_count", 0) + 1
    reboot_mem["last_boot_ts"] = datetime.utcnow().isoformat() + "Z"
    reboot_mem["last_state"] = borg_state_snapshot
    reboot_mem.setdefault("environment_fingerprints", []).append(env_fingerprint)
    save_reboot_memory(reboot_mem)


def record_threat_in_reboot_memory(reboot_mem, event):
    reboot_mem.setdefault("threat_history", []).append(event)
    save_reboot_memory(reboot_mem)

# ---------------- cockpit GUI ----------------

class Cockpit(QWidget):
    def __init__(self, cfg_ref, borg: BorgMemory, state_ref, parent=None):
        super().__init__(parent)
        self.cfg_ref = cfg_ref
        self.borg = borg
        self.state_ref = state_ref

        self.setWindowTitle("OmniQueen Swarm Guard Cockpit")
        self.resize(800, 500)

        main_layout = QVBoxLayout()

        # Status group
        status_group = QGroupBox("Status")
        status_layout = QHBoxLayout()
        self.state_label = QLabel("State: UNKNOWN")
        self.state_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.node_label = QLabel(f"Node: {self.cfg_ref['cfg'].get('node_id', platform.node())}")
        self.role_label = QLabel(f"Role: {self.cfg_ref['cfg'].get('role', 'node')}")
        status_layout.addWidget(self.state_label)
        status_layout.addWidget(self.node_label)
        status_layout.addWidget(self.role_label)
        status_layout.addStretch()
        status_group.setLayout(status_layout)

        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()

        self.kill_checkbox = QCheckBox("Kill extreme processes")
        self.block_checkbox = QCheckBox("Block extreme IPs")
        self.panic_checkbox = QCheckBox("Panic mode (no auto actions, force CALM)")

        controls_layout.addWidget(self.kill_checkbox)
        controls_layout.addWidget(self.block_checkbox)
        controls_layout.addWidget(self.panic_checkbox)

        self.apply_button = QPushButton("Apply Overrides")
        controls_layout.addWidget(self.apply_button)

        controls_group.setLayout(controls_layout)

        # Lists group
        lists_group = QGroupBox("Suspects")
        lists_layout = QHBoxLayout()

        self.proc_list = QListWidget()
        self.proc_list.setSelectionMode(QListWidget.NoSelection)
        self.proc_list.setAlternatingRowColors(True)
        self.proc_list.setWindowTitle("Suspect Processes")

        self.ip_list = QListWidget()
        self.ip_list.setSelectionMode(QListWidget.NoSelection)
        self.ip_list.setAlternatingRowColors(True)
        self.ip_list.setWindowTitle("Suspect IPs")

        lists_layout.addWidget(self.proc_list)
        lists_layout.addWidget(self.ip_list)
        lists_group.setLayout(lists_layout)

        main_layout.addWidget(status_group)
        main_layout.addWidget(controls_group)
        main_layout.addWidget(lists_group)

        self.setLayout(main_layout)

        self.apply_button.clicked.connect(self.apply_overrides)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_view)
        self.timer.start(1000)

        self.refresh_view()

    def apply_overrides(self):
        cfg = self.cfg_ref["cfg"]
        auto_cfg = cfg.get("auto_response", {})
        manual_cfg = cfg.get("manual_override", {})

        auto_cfg["kill_extreme_processes"] = self.kill_checkbox.isChecked()
        auto_cfg["firewall_block_extreme_ips"] = self.block_checkbox.isChecked()
        auto_cfg["log_only"] = not (auto_cfg["kill_extreme_processes"] or auto_cfg["firewall_block_extreme_ips"])

        manual_cfg["panic_mode"] = self.panic_checkbox.isChecked()

        cfg["auto_response"] = auto_cfg
        cfg["manual_override"] = manual_cfg

        if manual_cfg["panic_mode"]:
            self.borg.altered_state = "CALM"
            log_event("INFO", "manual_override", "Panic mode enabled, forcing CALM", {})
        else:
            log_event("INFO", "manual_override", "Panic mode disabled", {})

        save_config(cfg)
        self.cfg_ref["cfg"] = load_config()
        log_event("INFO", "cockpit", "Overrides applied", {
            "kill_extreme_processes": auto_cfg["kill_extreme_processes"],
            "firewall_block_extreme_ips": auto_cfg["firewall_block_extreme_ips"],
            "panic_mode": manual_cfg["panic_mode"]
        })

    def refresh_view(self):
        cfg = self.cfg_ref["cfg"]
        auto_cfg = cfg.get("auto_response", {})
        manual_cfg = cfg.get("manual_override", {})

        self.kill_checkbox.setChecked(auto_cfg.get("kill_extreme_processes", False))
        self.block_checkbox.setChecked(auto_cfg.get("firewall_block_extreme_ips", False))
        self.panic_checkbox.setChecked(manual_cfg.get("panic_mode", False))

        state = self.borg.altered_state
        self.state_label.setText(f"State: {state}")
        if state == "CALM":
            self.state_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        elif state == "ALERT":
            self.state_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFC107;")
        else:
            self.state_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #F44336;")

        self.proc_list.clear()
        for p in sorted(self.borg.suspect_procs):
            item = QListWidgetItem(p)
            self.proc_list.addItem(item)

        self.ip_list.clear()
        for ip in sorted(self.borg.suspect_ips):
            item = QListWidgetItem(ip)
            self.ip_list.addItem(item)

# ---------------- main ----------------

def main():
    cfg = load_config()
    cfg_ref = {"cfg": cfg}
    state = load_state()
    reboot_mem = load_reboot_memory()

    env_fingerprint = probe_environment()
    node_id = cfg.get("node_id", platform.node())
    role = cfg.get("role", "node")

    log_event("INFO", "core", "OmniQueen Swarm Guard starting", {
        "platform": env_fingerprint.get("platform"),
        "pid": os.getpid(),
        "node_id": node_id,
        "role": role
    })

    borg = BorgMemory(state, cfg, node_id)
    update_reboot_memory_on_start(reboot_mem, env_fingerprint, {
        "altered_state": borg.altered_state,
        "proc_stats_size": len(borg.proc_stats),
        "ip_stats_size": len(borg.ip_stats)
    })

    init_file_baseline(cfg, borg)

    stop_event = threading.Event()

    t_proc = threading.Thread(target=process_watch_loop, args=(cfg_ref, state, borg, stop_event), daemon=True)
    t_net = threading.Thread(target=net_watch_loop, args=(cfg_ref, state, borg, stop_event), daemon=True)
    t_proc.start()
    t_net.start()

    t_queen = None
    if role == "queen":
        t_queen = threading.Thread(target=queen_loop, args=(cfg_ref, stop_event), daemon=True)
        t_queen.start()

    swarm_cfg = cfg.get("swarm", {})
    publish_interval = swarm_cfg.get("publish_interval", 15)
    last_publish = time.time()

    poll_interval = cfg.get("poll_interval_seconds", 5)

    # GUI
    app = QApplication(sys.argv)
    cockpit = Cockpit(cfg_ref, borg, state)
    cockpit.show()

    def background_loop():
        nonlocal cfg_ref, state, borg, stop_event, publish_interval, last_publish, poll_interval
        try:
            while not stop_event.is_set():
                cfg_ref["cfg"] = load_config()
                cfg = cfg_ref["cfg"]
                check_file_integrity(cfg, borg)
                borg.update_altered_state()

                swarm_cfg_local = cfg.get("swarm", {})
                if swarm_cfg_local.get("enabled", True):
                    now = time.time()
                    if now - last_publish >= publish_interval:
                        swarm_publish_state(cfg, borg)
                        if role == "node":
                            swarm_node_orders_apply(cfg, borg)
                        last_publish = now

                borg.flush_back()
                save_state(state)
                time.sleep(poll_interval)
        except Exception as e:
            log_event("ERROR", "core", "Background loop crashed", {"error": str(e)})
            stop_event.set()

    t_bg = threading.Thread(target=background_loop, daemon=True)
    t_bg.start()

    try:
        app.exec()
    finally:
        stop_event.set()
        t_proc.join(timeout=3)
        t_net.join(timeout=3)
        if t_queen:
            t_queen.join(timeout=3)
        borg.flush_back()
        save_state(state)
        record_threat_in_reboot_memory(reboot_mem, {
            "ts": datetime.utcnow().isoformat() + "Z",
            "final_altered_state": borg.altered_state,
            "suspect_ips": sorted(borg.suspect_ips),
            "suspect_procs": sorted(borg.suspect_procs)
        })
        log_event("INFO", "core", "OmniQueen Swarm Guard stopped")


if __name__ == "__main__":
    main()
