#!/usr/bin/env python3
# Omniguard Borg Unified Sentinel
import sys
import os
import json
import time
import platform
import threading
import subprocess
from datetime import datetime, timedelta
from collections import deque, defaultdict
from math import sqrt

# ---------------- autoloader ----------------

REQUIRED_LIBS = ["psutil", "yaml"]

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

# ---------------- paths & constants ----------------

CONFIG_PATH = "omniguard_config.yml"
STATE_PATH = "omniguard_state.json"
REBOOT_MEMORY_PATH = "omniguard_reboot_memory.json"
LOG_PATH = "omniguard_events.log"

ALTERED_STATES = ["CALM", "ALERT", "WAR"]  # conceptual consciousness modes

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
            "poll_interval_seconds": 5,
            "ai_anomaly": {
                "enabled": True,
                "sigma_multiplier": 2.5
            },
            "file_integrity": {
                "enabled": True,
                "paths": [
                    "/etc/hosts" if os.name != "nt" else r"C:\Windows\System32\drivers\etc\hosts"
                ]
            },
            "process_watch": {
                "enabled": True,
                "suspicious_keywords": ["miner", "rat", "keylog", "hack", "bot"],
                "rolling_window": 600
            },
            "net_watch": {
                "enabled": True,
                "rolling_window": 300
            },
            "auto_response": {
                "kill_extreme_processes": False,
                "firewall_block_extreme_ips": False,
                "log_only": True
            },
            "water_physics": {
                "enabled": True,
                "flow_window": 120
            }
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(default, f)
        log_event("INFO", "config", f"Default config created at {CONFIG_PATH}")
        return default

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


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

# ---------------- simple stats helpers ----------------

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

# ---------------- probes: environment self-analysis ----------------

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
    # quick port/proc snapshot
    try:
        info["process_count"] = len(list(psutil.process_iter()))
    except Exception:
        info["process_count"] = None
    try:
        info["connection_count"] = len(psutil.net_connections(kind="inet"))
    except Exception:
        info["connection_count"] = None
    return info

# ---------------- water data physics engine ----------------

class WaterPhysicsEngine:
    """
    Treats events as water flowing through a channel:
    - flow_rate: events per second
    - turbulence: variance of flow
    - pressure: combined measure used to modulate altered state
    """
    def __init__(self, cfg_section):
        self.enabled = cfg_section.get("enabled", True) if cfg_section else True
        self.flow_window = cfg_section.get("flow_window", 120) if cfg_section else 120
        self.event_stream = deque()  # (ts, weight)
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
        # flow rate: total weight / window
        total_weight = sum(w for _, w in self.event_stream)
        flow_rate = total_weight / float(self.flow_window)
        # turbulence: std dev of weights
        weights = [w for _, w in self.event_stream]
        mean_w = sum(weights) / len(weights)
        var = sum((w - mean_w) ** 2 for w in weights) / max(len(weights) - 1, 1)
        turbulence = sqrt(max(var, 0.0))
        pressure = flow_rate * (1.0 + turbulence)
        # update rolling stats for flow if desired
        self.flow_stats = update_rolling_stats(self.flow_stats, pressure)
        return {"flow_rate": flow_rate, "turbulence": turbulence, "pressure": pressure}

# ---------------- borg memory / adaptive engine ----------------

class BorgMemory:
    """
    Adaptive borg memory:
    - Tracks process/IP frequencies
    - Rolling stats for process and connection counts
    - File hashes for integrity
    - Water physics engine for threat flow
    - Altered state of consciousness (CALM/ALERT/WAR)
    """
    def __init__(self, state, cfg):
        self.state = state
        self.proc_stats = state.get("proc_stats", {})
        self.ip_stats = state.get("ip_stats", {})
        self.file_hashes = state.get("file_hashes", {})
        self.proc_count_stats = state.get("proc_count_stats", None)
        self.conn_count_stats = state.get("conn_count_stats", None)
        self.altered_state = state.get("altered_state", "CALM")
        self.threat_pressure_stats = state.get("threat_pressure_stats", None)

        self.ai_cfg = cfg.get("ai_anomaly", {}) if cfg else {}
        self.water_engine = WaterPhysicsEngine(cfg.get("water_physics", {}))

    # basic memory
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

    # scoring
    def score_process(self, name, cmdline, suspicious_keywords):
        rec = self.proc_stats.get(name, {"count": 0, "last_seen": 0})
        count = rec["count"]
        rarity = 1.0 if count == 0 else 1.0 / (1.0 + count)

        keyword_boost = 0.0
        low_cmd = (cmdline or "").lower()
        low_name = (name or "").lower()
        for kw in suspicious_keywords:
            if kw in low_name or kw in low_cmd:
                keyword_boost += 0.3

        env_boost = 0.0
        if self.proc_count_stats and self.proc_count_stats["count"] > 5:
            sigma = stats_sigma(self.proc_count_stats)
            if sigma > 0:
                env_boost = min(0.4, sigma / (self.ai_cfg.get("sigma_multiplier", 2.5) * 10.0))

        # water physics influence
        flow = self.water_engine.compute_flow()
        pressure = flow["pressure"]
        self.threat_pressure_stats = update_rolling_stats(self.threat_pressure_stats, pressure)
        pressure_boost = min(0.4, pressure * 2.0)

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
                env_boost = min(0.4, sigma / (self.ai_cfg.get("sigma_multiplier", 2.5) * 10.0))

        flow = self.water_engine.compute_flow()
        pressure = flow["pressure"]
        self.threat_pressure_stats = update_rolling_stats(self.threat_pressure_stats, pressure)
        pressure_boost = min(0.4, pressure * 2.0)

        return min(1.0, rarity * 0.5 + env_boost + pressure_boost)

    # altered states of consciousness (mode machine)
    def update_altered_state(self):
        # derive from threat pressure stats
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

    def flush_back(self):
        self.state["proc_stats"] = self.proc_stats
        self.state["ip_stats"] = self.ip_stats
        self.state["file_hashes"] = self.file_hashes
        self.state["proc_count_stats"] = self.proc_count_stats
        self.state["conn_count_stats"] = self.conn_count_stats
        self.state["altered_state"] = self.altered_state
        self.state["threat_pressure_stats"] = self.threat_pressure_stats

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


def process_watch_loop(cfg, state, borg: BorgMemory, stop_event):
    pw_cfg = cfg.get("process_watch", {})
    ai_cfg = cfg.get("ai_anomaly", {})
    auto_cfg = cfg.get("auto_response", {})

    if not pw_cfg.get("enabled", True):
        return

    suspicious_keywords = [k.lower() for k in pw_cfg.get("suspicious_keywords", [])]

    seen_pids = set()

    while not stop_event.is_set():
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

# ---------------- network watch ----------------

def firewall_block_ip(ip):
    # placeholder: wire real OS-specific firewall commands here
    log_event("INFO", "firewall", "Requested block for IP (placeholder only)", {"ip": ip})


def net_watch_loop(cfg, state, borg: BorgMemory, stop_event):
    nw_cfg = cfg.get("net_watch", {})
    ai_cfg = cfg.get("ai_anomaly", {})
    auto_cfg = cfg.get("auto_response", {})

    if not nw_cfg.get("enabled", True):
        return

    rolling_window = nw_cfg.get("rolling_window", 300)
    recent_conns = deque()

    while not stop_event.is_set():
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
                    if auto_cfg.get("firewall_block_extreme_ips", False) and borg.altered_state in ["ALERT", "WAR"]:
                        firewall_block_ip(r_ip)

        cutoff = now - timedelta(seconds=rolling_window)
        while recent_conns and recent_conns[0][0] < cutoff:
            recent_conns.popleft()

        time.sleep(2)

# ---------------- reboot memory integration ----------------

def update_reboot_memory_on_start(reboot_mem, env_fingerprint, borg_state_snapshot):
    reboot_mem["boot_count"] = reboot_mem.get("boot_count", 0) + 1
    reboot_mem["last_boot_ts"] = datetime.utcnow().isoformat() + "Z"
    reboot_mem["last_state"] = borg_state_snapshot
    reboot_mem.setdefault("environment_fingerprints", []).append(env_fingerprint)
    save_reboot_memory(reboot_mem)


def record_threat_in_reboot_memory(reboot_mem, event):
    reboot_mem.setdefault("threat_history", []).append(event)
    save_reboot_memory(reboot_mem)

# ---------------- main loop ----------------

def main():
    cfg = load_config()
    state = load_state()
    reboot_mem = load_reboot_memory()

    env_fingerprint = probe_environment()
    log_event("INFO", "core", "Omniguard starting", {
        "platform": env_fingerprint.get("platform"),
        "pid": os.getpid()
    })

    borg = BorgMemory(state, cfg)
    update_reboot_memory_on_start(reboot_mem, env_fingerprint, {
        "altered_state": borg.altered_state,
        "proc_stats_size": len(borg.proc_stats),
        "ip_stats_size": len(borg.ip_stats)
    })

    init_file_baseline(cfg, borg)

    stop_event = threading.Event()

    t_proc = threading.Thread(target=process_watch_loop, args=(cfg, state, borg, stop_event), daemon=True)
    t_net = threading.Thread(target=net_watch_loop, args=(cfg, state, borg, stop_event), daemon=True)

    t_proc.start()
    t_net.start()

    poll_interval = cfg.get("poll_interval_seconds", 5)

    try:
        while True:
            cfg = load_config()  # hot reload
            check_file_integrity(cfg, borg)
            borg.update_altered_state()
            borg.flush_back()
            save_state(state)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        log_event("INFO", "core", "Shutdown requested by user")
    finally:
        stop_event.set()
        t_proc.join(timeout=3)
        t_net.join(timeout=3)
        borg.flush_back()
        save_state(state)
        # snapshot final state into reboot memory
        record_threat_in_reboot_memory(reboot_mem, {
            "ts": datetime.utcnow().isoformat() + "Z",
            "final_altered_state": borg.altered_state
        })
        log_event("INFO", "core", "Omniguard stopped")


if __name__ == "__main__":
    main()
