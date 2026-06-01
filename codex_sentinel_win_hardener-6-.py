#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# codex_sentinel_organism_tk_ultra_raft_etw.py
#
# Core capabilities:
#   - Windows hardening (home/server policy packs)
#   - Telemetry export + distributed storage (SQLite + optional Redis/MinIO)
#   - Real ETW ingestion path (Security/System providers) with optional etw bindings
#   - eBPF hook points (Linux)
#   - Async UDP event bus with signed messages (HMAC/PKI)
#   - Multi-node Raft consensus over UDP (leader election + heartbeats + log replication)
#   - Distributed threat matrix
#   - Autonomous policy evolution + rollback + System Restore
#   - Persistent settings store
#   - Plugin system + marketplace model
#   - Multi-process swarm nodes
#   - Tkinter cockpit HUD
#   - Optional GPU HUD overlay (pyglet/OpenGL)
#   - ML-based anomaly scoring (joblib)
#   - REST API for remote control
#   - CLI fallback
#

import argparse
import asyncio
import ctypes
import hashlib
import hmac
import json
import os
import platform
import queue
import random
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from multiprocessing import Process
from textwrap import dedent
from http.server import BaseHTTPRequestHandler, HTTPServer

import tkinter as tk
from tkinter import ttk

# Optional imports
try:
    import winreg  # type: ignore
except ImportError:
    winreg = None

try:
    import requests  # type: ignore
except ImportError:
    requests = None

# Optional voice control
try:
    import speech_recognition as sr  # type: ignore
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False

# Optional ETW bindings (real path)
HAS_ETW = False
ETW_AVAILABLE = False
try:
    if platform.system().lower() == "windows":
        # You can use packages like `etw` or `pywintrace`.
        # Here we try `etw` as an example; if not installed, we fall back to stub.
        import etw  # type: ignore
        from etw import ETW, ProviderInfo  # type: ignore
        HAS_ETW = True
        ETW_AVAILABLE = True
except Exception:
    HAS_ETW = False
    ETW_AVAILABLE = False

# Optional eBPF (Linux)
HAS_EBPF = platform.system().lower() == "linux"

# Optional GPU HUD (pyglet/OpenGL)
try:
    import pyglet  # type: ignore
    HAS_GPU_HUD = True
except Exception:
    HAS_GPU_HUD = False

# Optional ML model
try:
    import joblib  # type: ignore
    HAS_AI_MODEL = True
except Exception:
    HAS_AI_MODEL = False

# Optional cryptography for PKI
try:
    from cryptography.hazmat.primitives import hashes, serialization  # type: ignore
    from cryptography.hazmat.primitives.asymmetric import padding, rsa  # type: ignore
    HAS_PKI = True
except Exception:
    HAS_PKI = False

# Optional Redis
try:
    import redis  # type: ignore
    HAS_REDIS = True
except Exception:
    HAS_REDIS = False

# Optional MinIO (S3-compatible)
try:
    from minio import Minio  # type: ignore
    HAS_MINIO = True
except Exception:
    HAS_MINIO = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Auto-elevation (Windows)
# ---------------------------

def ensure_admin():
    if platform.system().lower() != "windows":
        return
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            script = os.path.abspath(sys.argv[0])
            params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
            ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                sys.executable,
                f'"{script}" {params}',
                None,
                1
            )
            sys.exit()
    except Exception as e:
        print(f"[Codex Sentinel] Elevation failed: {e}")
        sys.exit()


# ---------------------------
# Persistent settings store
# ---------------------------

class SettingsStore:
    def __init__(self, path=None):
        self.path = path or os.path.join(BASE_DIR, "codex_settings.json")
        self._lock = threading.Lock()
        self.data = {
            "swarm": {
                "enabled": False,
                "base_url": "https://swarm.example.local:9000",
                "api_key": "CHANGE_ME",
                "shared_secret": "CHANGE_ME_SECRET",
                "pki_enabled": False,
                "public_key_path": os.path.join(BASE_DIR, "swarm_pub.pem"),
                "private_key_path": os.path.join(BASE_DIR, "swarm_priv.pem"),
            },
            "auto_mode": True,
            "auto_interval": 300,
            "last_policy_pack": "home-1.0",
            "node_id": socket.gethostname(),
            "rest_api": {
                "enabled": False,
                "host": "127.0.0.1",
                "port": 8088,
            },
            "ai_model_path": os.path.join(BASE_DIR, "codex_ai_model.joblib"),
            "gpu_hud": {
                "enabled": True
            },
            "storage": {
                "sqlite_path": os.path.join(BASE_DIR, "codex_telemetry.db"),
                "redis_url": "redis://localhost:6379/0",
                "minio": {
                    "enabled": False,
                    "endpoint": "localhost:9000",
                    "access_key": "minioadmin",
                    "secret_key": "minioadmin",
                    "bucket": "codex-telemetry",
                    "secure": False,
                },
            },
            "plugin_marketplace": {
                "index_url": "https://plugins.example.local/index.json",
                "enabled": False,
            },
            "raft": {
                "cluster_nodes": [
                    # Example: {"id": "node1", "host": "127.0.0.1", "port": 50555}
                ],
                "election_timeout_min": 1500,
                "election_timeout_max": 3000,
                "heartbeat_interval": 500,
            },
        }
        self.load()

    def load(self):
        with self._lock:
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    for k, v in loaded.items():
                        if isinstance(v, dict) and k in self.data:
                            self.data[k].update(v)
                        else:
                            self.data[k] = v
            except Exception as e:
                print(f"[SETTINGS] Failed to load settings: {e}")

    def save(self):
        with self._lock:
            try:
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=2)
                os.replace(tmp, self.path)
            except Exception as e:
                print(f"[SETTINGS] Failed to save settings: {e}")

    def get(self, key, default=None):
        with self._lock:
            return self.data.get(key, default)

    def set(self, key, value):
        with self._lock:
            self.data[key] = value
        self.save()

    def get_nested(self, *keys, default=None):
        with self._lock:
            cur = self.data
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

    def set_nested(self, value, *keys):
        with self._lock:
            cur = self.data
            for k in keys[:-1]:
                if k not in cur or not isinstance(cur[k], dict):
                    cur[k] = {}
                cur = cur[k]
            cur[keys[-1]] = value
        self.save()


SETTINGS = SettingsStore()

# ---------------------------
# Policy packs (versioned)
# ---------------------------

POLICY_PACKS = {
    "home-1.0": {
        "profile": "home",
        "version": "1.0",
        "modules": [
            "defender",
            "smartscreen",
            "rdp",
            "smb_legacy",
            "firewall",
            "app_control_prep",
            "boot_bitlocker",
            "identity_lsa",
        ],
    },
    "server-1.0": {
        "profile": "server",
        "version": "1.0",
        "modules": [
            "defender",
            "smartscreen",
            "rdp",
            "smb_legacy",
            "firewall",
            "app_control_prep",
            "boot_bitlocker",
            "identity_lsa",
        ],
    },
}

# ---------------------------
# Plugin system + marketplace
# ---------------------------

PLUGIN_DIR = os.path.join(BASE_DIR, "plugins")
PLUGINS = {}  # name -> callable(profile, dry_run)

def load_plugins():
    if not os.path.isdir(PLUGIN_DIR):
        return
    sys.path.insert(0, PLUGIN_DIR)
    for fname in os.listdir(PLUGIN_DIR):
        if not fname.endswith(".py"):
            continue
        mod_name = os.path.splitext(fname)[0]
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "MODULE_NAME") and hasattr(mod, "apply_module"):
                PLUGINS[mod.MODULE_NAME] = mod.apply_module
                print(f"[PLUGIN] Loaded module plugin: {mod.MODULE_NAME}")
        except Exception as e:
            print(f"[PLUGIN] Failed to load {fname}: {e}")


def fetch_plugin_marketplace_index():
    cfg = SETTINGS.get("plugin_marketplace", {})
    if not cfg.get("enabled", False):
        return None
    url = cfg.get("index_url")
    if not url or requests is None:
        return None
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            index = resp.json()
            print(f"[PLUGIN MARKET] Index fetched: {len(index.get('plugins', []))} plugins.")
            return index
    except Exception as e:
        print(f"[PLUGIN MARKET] Fetch failed: {e}")
    return None


load_plugins()

# ---------------------------
# Swarm / sync configuration
# ---------------------------

SWARM_CONFIG = SETTINGS.get("swarm")

# ---------------------------
# Autonomy / threat / event bus / Raft
# ---------------------------

AUTO_MODE = SETTINGS.get("auto_mode", True)
AUTO_INTERVAL = SETTINGS.get("auto_interval", 300)
AUTO_THREAD = None
AUTO_STOP = False

THREAT_LEVEL = "LOW"      # LOW / MEDIUM / HIGH / CRITICAL
THREAT_REASON = ""
PENDING_ACTIONS = []      # list of dicts: {id, level, reason, suggestion}

EVENT_BUS_PORT = 50555

# Async event bus
EVENT_LOOP = None
EVENT_BUS_QUEUE: "queue.Queue[dict]" = queue.Queue()
EVENT_BUS_STOP = False

# Distributed threat matrix
THREAT_MATRIX = {}  # node_id -> {"level": str, "reason": str, "timestamp": int}

# Multi-process swarm nodes
SWARM_NODE_PROCESSES = {}
SWARM_NODE_LOCK = threading.Lock()

# Watchdog
WATCHDOG_THREAD = None
WATCHDOG_STOP = False

# Raft consensus state (multi-node)
RAFT_LOCK = threading.Lock()
RAFT_CONFIG = SETTINGS.get("raft", {})
RAFT_CLUSTER = RAFT_CONFIG.get("cluster_nodes", [])
NODE_ID = SETTINGS.get("node_id", socket.gethostname())

RAFT_STATE = {
    "current_term": 0,
    "voted_for": None,
    "log": [],  # list of {"term": int, "entry": dict}
    "commit_index": -1,
    "last_applied": -1,
    "role": "follower",  # follower / candidate / leader
    "leader_id": None,
    "election_deadline": 0,
    "heartbeat_interval": RAFT_CONFIG.get("heartbeat_interval", 500) / 1000.0,
}

# Snapshots / rollback
SNAPSHOT_DIR = os.path.join(BASE_DIR, "codex_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Distributed telemetry storage
TELEMETRY_STORE_DIR = os.path.join(BASE_DIR, "codex_telemetry_store")
os.makedirs(TELEMETRY_STORE_DIR, exist_ok=True)

# SQLite storage
SQLITE_PATH = SETTINGS.get_nested("storage", "sqlite_path")
os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)

# REST API
REST_SERVER = None
REST_THREAD = None

# AI model
AI_MODEL = None
if HAS_AI_MODEL:
    path = SETTINGS.get_nested("ai_model_path")
    if path and os.path.exists(path):
        try:
            AI_MODEL = joblib.load(path)
            print("[AI] Loaded anomaly model.")
        except Exception as e:
            print(f"[AI] Failed to load model: {e}")
            AI_MODEL = None

# PKI keys
PKI_PRIVATE_KEY = None
PKI_PUBLIC_KEY = None
if HAS_PKI and SWARM_CONFIG.get("pki_enabled", False):
    priv_path = SWARM_CONFIG.get("private_key_path")
    pub_path = SWARM_CONFIG.get("public_key_path")
    try:
        if os.path.exists(priv_path):
            with open(priv_path, "rb") as f:
                PKI_PRIVATE_KEY = serialization.load_pem_private_key(f.read(), password=None)
        else:
            PKI_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            with open(priv_path, "wb") as f:
                f.write(PKI_PRIVATE_KEY.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                ))
        if os.path.exists(pub_path):
            with open(pub_path, "rb") as f:
                PKI_PUBLIC_KEY = serialization.load_pem_public_key(f.read())
        else:
            PKI_PUBLIC_KEY = PKI_PRIVATE_KEY.public_key()
            with open(pub_path, "wb") as f:
                f.write(PKI_PUBLIC_KEY.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                ))
        print("[PKI] Swarm keys ready.")
    except Exception as e:
        print(f"[PKI] Key init failed: {e}")
        PKI_PRIVATE_KEY = None
        PKI_PUBLIC_KEY = None

# ---------------------------
# Utility / plumbing
# ---------------------------

def is_windows():
    return platform.system().lower() == "windows"


def run_ps(command, dry_run=False):
    print(f"[PS] {command}")
    if dry_run or not is_windows():
        return 0, "", ""
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            timeout=120
        )
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except Exception as e:
        print(f"[PS ERROR] {e}")
        return 1, "", str(e)


def set_reg_value(root, path, name, value, reg_type, dry_run=False):
    if winreg is None or root is None:
        print("[REG] winreg not available; skipping registry operation.")
        return

    print(f"[REG] {root}\\{path} :: {name} = {value} ({reg_type})")
    if dry_run:
        return

    try:
        key = winreg.CreateKey(root, path)
        winreg.SetValueEx(key, name, 0, reg_type, value)
        winreg.CloseKey(key)
    except Exception as e:
        print(f"[REG] Failed to set {path}\\{name}: {e}")


def get_reg_value(root, path, name, default=None):
    if winreg is None or root is None:
        return default
    try:
        key = winreg.OpenKey(root, path, 0, winreg.KEY_READ)
        value, _ = winreg.QueryValueEx(key, name)
        winreg.CloseKey(key)
        return value
    except Exception:
        return default


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


# ---------------------------
# Crypto signing & PKI for swarm messages
# ---------------------------

def sign_payload_hmac(payload: dict) -> dict:
    secret = SWARM_CONFIG.get("shared_secret", "CHANGE_ME_SECRET").encode("utf-8")
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return {"body": payload, "sig": sig, "alg": "HMAC"}


def verify_payload_hmac(wrapper: dict) -> dict | None:
    try:
        secret = SWARM_CONFIG.get("shared_secret", "CHANGE_ME_SECRET").encode("utf-8")
        body = wrapper["body"]
        sig = wrapper["sig"]
        raw = json.dumps(body, sort_keys=True).encode("utf-8")
        expected = hmac.new(secret, raw, hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return body
        else:
            print("[CRYPTO] HMAC signature mismatch; dropping message.")
            return None
    except Exception as e:
        print(f"[CRYPTO] HMAC verification error: {e}")
        return None


def sign_payload_pki(payload: dict) -> dict:
    if not (HAS_PKI and PKI_PRIVATE_KEY):
        return sign_payload_hmac(payload)
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    sig = PKI_PRIVATE_KEY.sign(
        body,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {"body": payload, "sig": sig.hex(), "alg": "PKI"}


def verify_payload_pki(wrapper: dict) -> dict | None:
    alg = wrapper.get("alg", "HMAC")
    if alg == "HMAC" or not (HAS_PKI and PKI_PUBLIC_KEY):
        return verify_payload_hmac(wrapper)
    try:
        body = wrapper["body"]
        sig = bytes.fromhex(wrapper["sig"])
        raw = json.dumps(body, sort_keys=True).encode("utf-8")
        PKI_PUBLIC_KEY.verify(
            sig,
            raw,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return body
    except Exception as e:
        print(f"[CRYPTO] PKI verification error: {e}")
        return None


def sign_payload(payload: dict) -> dict:
    if SWARM_CONFIG.get("pki_enabled", False) and HAS_PKI and PKI_PRIVATE_KEY:
        return sign_payload_pki(payload)
    return sign_payload_hmac(payload)


def verify_payload(wrapper: dict) -> dict | None:
    if wrapper.get("alg") == "PKI":
        return verify_payload_pki(wrapper)
    return verify_payload_hmac(wrapper)


# ---------------------------
# Organism / swarm hooks
# ---------------------------

def organism_notify(event, data=None):
    print(f"[ORGANISM] Event: {event}, Data: {str(data)[:200]}")
    send_event_bus({"event": event, "data": data})


# ---------------------------
# AI / anomaly scoring
# ---------------------------

def ai_feature_vector(telemetry):
    features = {
        "pua_unknown": telemetry["defender"].get("PUAProtection") in (0, "0", "unknown"),
        "lsa_weak": telemetry["identity_lsa"].get("RunAsPPL") in (0, "0", "unknown"),
        "smb1_on": telemetry["smb_legacy"].get("SMB1Protocol") in (1, "1"),
        "fw_disabled": False,
        "etw_events": len(telemetry["kernel"].get("etw_events", [])) > 0,
        "ebpf_events": len(telemetry["kernel"].get("ebpf_events", [])) > 0,
    }
    fw_profiles = telemetry["firewall"].get("profiles")
    if isinstance(fw_profiles, list):
        disabled = [p for p in fw_profiles if not p.get("Enabled")]
        features["fw_disabled"] = bool(disabled)

    vec = [
        int(features["pua_unknown"]),
        int(features["lsa_weak"]),
        int(features["smb1_on"]),
        int(features["fw_disabled"]),
        int(features["etw_events"]),
        int(features["ebpf_events"]),
    ]
    return vec, features


def ai_score_system_risk(context, telemetry=None):
    base_score = 0.2

    if telemetry:
        vec, features = ai_feature_vector(telemetry)

        score = base_score
        if features["pua_unknown"]:
            score += 0.15
        if features["lsa_weak"]:
            score += 0.2
        if features["smb1_on"]:
            score += 0.3
        if features["fw_disabled"]:
            score += 0.25
        if features["etw_events"] or features["ebpf_events"]:
            score += 0.2

        if AI_MODEL is not None:
            try:
                model_score = float(AI_MODEL.predict_proba([vec])[0][1])
                score = (score + model_score) / 2.0
            except Exception as e:
                print(f"[AI] Model inference error: {e}")
    else:
        score = base_score

    score = max(0.0, min(1.0, score))
    print(f"[AI] risk score for context '{context}': {score:.3f}")
    return score


# ---------------------------
# Telemetry collection
# ---------------------------

def collect_telemetry(profile, policy_pack_name, policy_pack):
    hostname = socket.gethostname()
    timestamp = int(time.time())

    telemetry = {
        "meta": {
            "hostname": hostname,
            "timestamp": timestamp,
            "profile": profile,
            "policy_pack": policy_pack_name,
            "policy_version": policy_pack.get("version"),
            "os": platform.platform(),
            "node_id": SETTINGS.get("node_id", hostname),
        },
        "defender": {},
        "firewall": {},
        "identity_lsa": {},
        "app_control": {},
        "smb_legacy": {},
        "kernel": {
            "etw_events": [],
            "ebpf_events": [],
        },
    }

    telemetry["defender"]["PUAProtection"] = get_reg_value(
        winreg.HKEY_LOCAL_MACHINE if winreg else None,
        r"SOFTWARE\Microsoft\Windows Defender\MpEngine",
        "PUAProtection",
        default="unknown",
    )

    telemetry["identity_lsa"]["RunAsPPL"] = get_reg_value(
        winreg.HKEY_LOCAL_MACHINE if winreg else None,
        r"SYSTEM\CurrentControlSet\Control\Lsa",
        "RunAsPPL",
        default="unknown",
    )

    telemetry["app_control"]["AuditMode"] = get_reg_value(
        winreg.HKEY_LOCAL_MACHINE if winreg else None,
        r"SYSTEM\CurrentControlSet\Control\CI\Policy",
        "AuditMode",
        default="unknown",
    )

    telemetry["smb_legacy"]["SMB1Protocol"] = get_reg_value(
        winreg.HKEY_LOCAL_MACHINE if winreg else None,
        r"SYSTEM\CurrentControlSet\Services\LanmanServer\Parameters",
        "SMB1",
        default="unknown",
    )

    try:
        code, out, _ = run_ps(
            "Get-NetFirewallProfile | Select-Object Name, Enabled | ConvertTo-Json",
            dry_run=False,
        )
        if code == 0 and out:
            telemetry["firewall"]["profiles"] = json.loads(out)
    except Exception:
        telemetry["firewall"]["profiles"] = "unknown"

    telemetry["kernel"]["etw_events"] = collect_etw_events()
    telemetry["kernel"]["ebpf_events"] = collect_ebpf_events()

    return telemetry


def export_telemetry(telemetry, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "codex_telemetry")

    os.makedirs(output_dir, exist_ok=True)

    hostname = telemetry["meta"]["hostname"]
    timestamp = telemetry["meta"]["timestamp"]
    filename = f"telemetry_{hostname}_{timestamp}.json"
    path = os.path.join(output_dir, filename)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        print(f"[TELEMETRY] Exported to: {path}")
    except Exception as e:
        print(f"[TELEMETRY] Failed to export telemetry: {e}")
    return path


def init_sqlite():
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT,
            ts INTEGER,
            level TEXT,
            reason TEXT,
            blob TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_sqlite()


def store_telemetry_distributed(telemetry, level=None, reason=None):
    node_id = telemetry["meta"]["node_id"]
    ts = telemetry["meta"]["timestamp"]
    blob = json.dumps(telemetry)

    # SQLite
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO telemetry (node_id, ts, level, reason, blob) VALUES (?, ?, ?, ?, ?)",
            (node_id, ts, level or "", reason or "", blob),
        )
        conn.commit()
        conn.close()
        print("[STORE] Telemetry stored in SQLite.")
    except Exception as e:
        print(f"[STORE] SQLite error: {e}")

    # Redis
    if HAS_REDIS:
        try:
            r = redis.from_url(SETTINGS.get_nested("storage", "redis_url"))
            key = f"telemetry:{node_id}:{ts}"
            r.set(key, blob)
        except Exception as e:
            print(f"[STORE] Redis error: {e}")

    # MinIO
    cfg = SETTINGS.get_nested("storage", "minio", default={})
    if cfg.get("enabled", False) and HAS_MINIO:
        try:
            client = Minio(
                cfg["endpoint"],
                access_key=cfg["access_key"],
                secret_key=cfg["secret_key"],
                secure=cfg.get("secure", False),
            )
            if not client.bucket_exists(cfg["bucket"]):
                client.make_bucket(cfg["bucket"])
            obj_name = f"{node_id}/{ts}.json"
            client.put_object(
                cfg["bucket"],
                obj_name,
                data=blob.encode("utf-8"),
                length=len(blob.encode("utf-8")),
                content_type="application/json",
            )
            print("[STORE] Telemetry stored in MinIO.")
        except Exception as e:
            print(f"[STORE] MinIO error: {e}")


# ---------------------------
# Real ETW ingestion
# ---------------------------

ETW_THREAD = None
ETW_STOP = False
ETW_BUFFER = []
ETW_LOCK = threading.Lock()

def etw_callback(event):
    """
    Called by ETW consumer for each event.
    """
    global ETW_BUFFER
    try:
        record = {
            "provider": event.get("provider_name", ""),
            "id": event.get("event_id", 0),
            "opcode": event.get("opcode", ""),
            "level": event.get("level", ""),
            "ts": int(time.time()),
        }
        with ETW_LOCK:
            ETW_BUFFER.append(record)
            if len(ETW_BUFFER) > 1000:
                ETW_BUFFER = ETW_BUFFER[-1000:]
    except Exception as e:
        print(f"[ETW CALLBACK ERROR] {e}")


def start_etw_listener():
    """
    Real ETW listener using `etw` package.
    Subscribes to Security and System providers.
    """
    global ETW_THREAD, ETW_STOP
    if not ETW_AVAILABLE:
        print("[ETW] etw package not available; using stub.")
        return
    if ETW_THREAD and ETW_THREAD.is_alive():
        return

    def run():
        global ETW_STOP
        print("[ETW] Listener started.")
        try:
            providers = [
                ProviderInfo("{54849625-5478-4994-A5BA-3E3B0328C30D}", any),  # Microsoft-Windows-Security-Auditing
                ProviderInfo("{9e814aad-3204-11d2-9a82-006008a86939}", any),  # System provider
            ]
            trace = ETW(providers=providers, event_callback=etw_callback)
            trace.start()
            while not ETW_STOP:
                time.sleep(1)
            trace.stop()
        except Exception as e:
            print(f"[ETW] Listener error: {e}")
        print("[ETW] Listener stopped.")

    ETW_STOP = False
    ETW_THREAD = threading.Thread(target=run, daemon=True)
    ETW_THREAD.start()


def collect_etw_events():
    if not ETW_AVAILABLE:
        return []
    with ETW_LOCK:
        events = list(ETW_BUFFER)
        ETW_BUFFER.clear()
    return events


# ---------------------------
# eBPF ingestion hooks
# ---------------------------

def collect_ebpf_events():
    if not HAS_EBPF:
        return []
    # Real integration: attach BPF programs via bcc/libbpf and push events here.
    # For now, keep a minimal stub to show pipeline.
    return []


# ---------------------------
# Swarm sync protocol + Raft consensus
# ---------------------------

def swarm_upload_telemetry(telemetry):
    if not SWARM_CONFIG.get("enabled", False):
        return None

    if requests is None:
        return None

    url = SWARM_CONFIG["base_url"].rstrip("/") + "/upload_telemetry"
    headers = {"X-API-Key": SWARM_CONFIG["api_key"]}
    try:
        wrapper = sign_payload(telemetry)
        resp = requests.post(url, headers=headers, json=wrapper, timeout=10)
        return resp.json()
    except Exception:
        return None


def swarm_fetch_policy_pack():
    if not SWARM_CONFIG.get("enabled", False):
        return None

    if requests is None:
        return None

    url = SWARM_CONFIG["base_url"].rstrip("/") + "/suggest_policy_pack"
    headers = {"X-API-Key": SWARM_CONFIG["api_key"]}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            wrapper = resp.json()
            body = verify_payload(wrapper) if isinstance(wrapper, dict) and "body" in wrapper else wrapper
            return body
        return None
    except Exception:
        return None


def raft_reset_election_deadline():
    with RAFT_LOCK:
        min_ms = RAFT_CONFIG.get("election_timeout_min", 1500)
        max_ms = RAFT_CONFIG.get("election_timeout_max", 3000)
        timeout = random.randint(min_ms, max_ms) / 1000.0
        RAFT_STATE["election_deadline"] = time.time() + timeout


def raft_become_follower(term, leader_id=None):
    with RAFT_LOCK:
        RAFT_STATE["current_term"] = term
        RAFT_STATE["role"] = "follower"
        RAFT_STATE["leader_id"] = leader_id
        RAFT_STATE["voted_for"] = None
        raft_reset_election_deadline()
        print(f"[RAFT] Become follower (term={term}, leader={leader_id})")


def raft_become_candidate():
    with RAFT_LOCK:
        RAFT_STATE["current_term"] += 1
        RAFT_STATE["role"] = "candidate"
        RAFT_STATE["leader_id"] = None
        RAFT_STATE["voted_for"] = NODE_ID
        raft_reset_election_deadline()
        term = RAFT_STATE["current_term"]
        print(f"[RAFT] Become candidate (term={term})")
    raft_broadcast_request_vote()


def raft_become_leader():
    with RAFT_LOCK:
        RAFT_STATE["role"] = "leader"
        RAFT_STATE["leader_id"] = NODE_ID
        print(f"[RAFT] Become leader (term={RAFT_STATE['current_term']})")
    raft_send_heartbeats()


def raft_last_log_index_term():
    with RAFT_LOCK:
        if not RAFT_STATE["log"]:
            return -1, 0
        idx = len(RAFT_STATE["log"]) - 1
        return idx, RAFT_STATE["log"][idx]["term"]


def raft_append_entry(entry):
    with RAFT_LOCK:
        RAFT_STATE["log"].append({"term": RAFT_STATE["current_term"], "entry": entry})
        print(f"[RAFT] Appended entry at index {len(RAFT_STATE['log']) - 1}")


def raft_broadcast_request_vote():
    last_index, last_term = raft_last_log_index_term()
    msg = {
        "type": "raft_request_vote",
        "term": RAFT_STATE["current_term"],
        "candidate_id": NODE_ID,
        "last_log_index": last_index,
        "last_log_term": last_term,
    }
    for node in RAFT_CLUSTER:
        if node["id"] == NODE_ID:
            continue
        send_event_bus({"raft": msg, "target": node["id"]})


def raft_handle_request_vote(msg):
    term = msg["term"]
    candidate_id = msg["candidate_id"]
    last_log_index = msg["last_log_index"]
    last_log_term = msg["last_log_term"]

    with RAFT_LOCK:
        if term < RAFT_STATE["current_term"]:
            vote_granted = False
        else:
            if term > RAFT_STATE["current_term"]:
                raft_become_follower(term)
            can_vote = (RAFT_STATE["voted_for"] is None or RAFT_STATE["voted_for"] == candidate_id)
            my_last_index, my_last_term = raft_last_log_index_term()
            up_to_date = (last_log_term > my_last_term) or (last_log_term == my_last_term and last_log_index >= my_last_index)
            vote_granted = can_vote and up_to_date
            if vote_granted:
                RAFT_STATE["voted_for"] = candidate_id
                raft_reset_election_deadline()

    reply = {
        "type": "raft_request_vote_reply",
        "term": RAFT_STATE["current_term"],
        "vote_granted": vote_granted,
        "voter_id": NODE_ID,
    }
    send_event_bus({"raft": reply, "target": candidate_id})


def raft_handle_request_vote_reply(msg):
    term = msg["term"]
    vote_granted = msg["vote_granted"]
    voter_id = msg["voter_id"]

    with RAFT_LOCK:
        if term < RAFT_STATE["current_term"]:
            return
        if term > RAFT_STATE["current_term"]:
            raft_become_follower(term)
            return
        if RAFT_STATE["role"] != "candidate":
            return

    if vote_granted:
        # Count votes
        votes = getattr(raft_handle_request_vote_reply, "_votes", set())
        votes.add(voter_id)
        raft_handle_request_vote_reply._votes = votes
        total_nodes = len(RAFT_CLUSTER) or 1
        if len(votes) + 1 > total_nodes // 2:  # +1 for self
            raft_become_leader()


def raft_send_heartbeats():
    with RAFT_LOCK:
        if RAFT_STATE["role"] != "leader":
            return
        term = RAFT_STATE["current_term"]
        commit_index = RAFT_STATE["commit_index"]
    msg = {
        "type": "raft_append_entries",
        "term": term,
        "leader_id": NODE_ID,
        "prev_log_index": -1,
        "prev_log_term": 0,
        "entries": [],
        "leader_commit": commit_index,
    }
    for node in RAFT_CLUSTER:
        if node["id"] == NODE_ID:
            continue
        send_event_bus({"raft": msg, "target": node["id"]})


def raft_handle_append_entries(msg):
    term = msg["term"]
    leader_id = msg["leader_id"]
    prev_log_index = msg["prev_log_index"]
    prev_log_term = msg["prev_log_term"]
    entries = msg["entries"]
    leader_commit = msg["leader_commit"]

    with RAFT_LOCK:
        if term < RAFT_STATE["current_term"]:
            success = False
        else:
            if term > RAFT_STATE["current_term"]:
                raft_become_follower(term, leader_id=leader_id)
            else:
                RAFT_STATE["leader_id"] = leader_id
                RAFT_STATE["role"] = "follower"
            raft_reset_election_deadline()

            if prev_log_index != -1:
                if prev_log_index >= len(RAFT_STATE["log"]) or RAFT_STATE["log"][prev_log_index]["term"] != prev_log_term:
                    success = False
                else:
                    success = True
            else:
                success = True

            if success and entries:
                RAFT_STATE["log"] = RAFT_STATE["log"][:prev_log_index + 1]
                for e in entries:
                    RAFT_STATE["log"].append(e)

            if leader_commit > RAFT_STATE["commit_index"]:
                RAFT_STATE["commit_index"] = min(leader_commit, len(RAFT_STATE["log"]) - 1)

    reply = {
        "type": "raft_append_entries_reply",
        "term": RAFT_STATE["current_term"],
        "success": success,
        "follower_id": NODE_ID,
    }
    send_event_bus({"raft": reply, "target": leader_id})


def raft_handle_append_entries_reply(msg):
    # For simplicity, we don't track per-follower matchIndex here.
    # In a full implementation, we'd track replication and advance commit_index.
    pass


def raft_consensus_tick():
    """
    Called periodically to drive elections and heartbeats.
    """
    with RAFT_LOCK:
        now = time.time()
        role = RAFT_STATE["role"]
        deadline = RAFT_STATE["election_deadline"]
        hb_interval = RAFT_STATE["heartbeat_interval"]

    if role in ("follower", "candidate") and now >= deadline:
        raft_become_candidate()
    elif role == "leader":
        # send heartbeats
        raft_send_heartbeats()
        raft_reset_election_deadline()


def consensus_merge_policy(local_pack, remote_pack):
    try:
        lv = float(local_pack.get("version", "0"))
    except Exception:
        lv = 0.0
    try:
        rv = float(remote_pack.get("version", "0"))
    except Exception:
        rv = 0.0

    raft_append_entry(remote_pack)

    if rv > lv:
        return remote_pack
    elif rv < lv:
        return local_pack
    else:
        lm = set(local_pack.get("modules", []))
        rm = set(remote_pack.get("modules", []))
        merged = sorted(lm | rm)
        merged_pack = dict(local_pack)
        merged_pack["modules"] = merged
        return merged_pack


# ---------------------------
# Hardening primitives
# ---------------------------

def harden_defender(dry_run=False):
    print_header("Defender / security stack hardening")

    run_ps("Set-MpPreference -PUAProtection Enabled", dry_run)
    run_ps("Set-MpPreference -MAPSReporting Advanced", dry_run)
    run_ps("Set-MpPreference -SubmitSamplesConsent SendSafeSamples", dry_run)
    run_ps("Set-MpPreference -DisableRealtimeMonitoring $false", dry_run)
    run_ps("Set-MpPreference -DisableBehaviorMonitoring $false", dry_run)
    run_ps("Set-MpPreference -DisableIOAVProtection $false", dry_run)
    run_ps("Set-MpPreference -DisableScriptScanning $false", dry_run)
    run_ps("Set-MpPreference -EnableNetworkProtection Enabled", dry_run)

    ai_score_system_risk("defender_hardening")
    organism_notify("defender_hardened")


def harden_smart_screen(dry_run=False):
    print_header("SmartScreen hardening")

    if winreg:
        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer",
            "SmartScreenEnabled",
            "RequireAdmin",
            winreg.REG_SZ,
            dry_run
        )

        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Policies\Microsoft\MicrosoftEdge\PhishingFilter",
            "EnabledV9",
            1,
            winreg.REG_DWORD,
            dry_run
        )

    ai_score_system_risk("smartscreen_hardening")
    organism_notify("smartscreen_hardened")


def harden_rdp(profile, dry_run=False):
    print_header("RDP / remote access hardening")

    if profile == "home":
        run_ps("Set-ItemProperty -Path 'HKLM:\\System\\CurrentControlSet\\Control\\Terminal Server' "
               "-Name 'fDenyTSConnections' -Value 1", dry_run)
        run_ps("Set-ItemProperty -Path 'HKLM:\\System\\CurrentControlSet\\Control\\Remote Assistance' "
               "-Name 'fAllowToGetHelp' -Value 0", dry_run)
    else:
        run_ps("Set-ItemProperty -Path 'HKLM:\\System\\CurrentControlSet\\Control\\Terminal Server\\WinStations\\RDP-Tcp' "
               "-Name 'UserAuthentication' -Value 1", dry_run)

    ai_score_system_risk(f"rdp_hardening_{profile}")
    organism_notify("rdp_hardened", {"profile": profile})


def harden_smb_and_legacy(dry_run=False):
    print_header("SMB / legacy protocol hardening")

    run_ps("Set-SmbServerConfiguration -EnableSMB1Protocol $false -Force", dry_run)
    run_ps("Disable-WindowsOptionalFeature -Online -FeatureName SMB1Protocol -NoRestart", dry_run)

    if winreg:
        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Services\Dnscache\Parameters",
            "EnableMulticast",
            0,
            winreg.REG_DWORD,
            dry_run
        )

    ai_score_system_risk("smb_legacy_hardening")
    organism_notify("smb_legacy_hardened")


def harden_firewall(profile, dry_run=False):
    print_header("Firewall hardening")

    run_ps("Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True", dry_run)

    if profile == "home":
        run_ps("Set-NetFirewallProfile -Profile Public,Private -DefaultInboundAction Block", dry_run)
    else:
        run_ps("Set-NetFirewallProfile -Profile Domain -DefaultInboundAction Block", dry_run)

    run_ps("Set-NetFirewallProfile -Profile Domain,Public,Private "
           "-LogAllowed True -LogBlocked True -LogFileName '%systemroot%\\system32\\LogFiles\\Firewall\\pfirewall.log'",
           dry_run)

    ai_score_system_risk(f"firewall_hardening_{profile}")
    organism_notify("firewall_hardened", {"profile": profile})


def prepare_app_control(dry_run=False):
    print_header("Application control (WDAC) preparation")

    if winreg:
        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\CI\Policy",
            "AuditMode",
            1,
            winreg.REG_DWORD,
            dry_run
        )

    print("[INFO] WDAC/app-control is NOT fully enforced by this script.")
    print("[INFO] It only prepares audit mode so you can build a safe allowlist.")
    ai_score_system_risk("app_control_prep")
    organism_notify("app_control_prep")


def harden_boot_and_bitlocker(profile, dry_run=False):
    print_header("Boot chain / BitLocker posture")

    cmd = "bcdedit /set {globalsettings} bootux disabled"
    print(f"[SUGGEST] To reduce external boot surface, consider: {cmd}")

    if profile == "home":
        print("[SUGGEST] On high-risk laptops, consider disabling WinRE temporarily:")
        print("          reagentc /disable")

    print("[SUGGEST] If BitLocker is enabled, configure a pre-boot PIN for stronger physical-access resistance.")
    ai_score_system_risk(f"boot_bitlocker_posture_{profile}")
    organism_notify("boot_bitlocker_posture", {"profile": profile})


def harden_identity_and_lsa(dry_run=False):
    print_header("Identity / LSA hardening")

    if winreg:
        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Lsa",
            "RunAsPPL",
            1,
            winreg.REG_DWORD,
            dry_run
        )

    print("[SUGGEST] Enable Credential Guard via Group Policy or Device Guard settings where supported.")
    ai_score_system_risk("identity_lsa_hardening")
    organism_notify("identity_lsa_hardened")


# ---------------------------
# System Restore (Windows rollback)
# ---------------------------

def create_system_restore_point(description="Codex Sentinel Policy Apply"):
    if not is_windows():
        return
    try:
        cmd = f'Checkpoint-Computer -Description "{description}" -RestorePointType "MODIFY_SETTINGS"'
        code, out, err = run_ps(cmd, dry_run=False)
        if code == 0:
            print("[RESTORE] System Restore point created.")
        else:
            print(f"[RESTORE] Failed to create restore point: {err}")
    except Exception as e:
        print(f"[RESTORE] Error creating restore point: {e}")


# ---------------------------
# Policy pack application + snapshots
# ---------------------------

def snapshot_state(policy_pack_name):
    try:
        ts = int(time.time())
        snap = {
            "timestamp": ts,
            "policy_pack_name": policy_pack_name,
            "policy_pack": POLICY_PACKS.get(policy_pack_name),
        }
        fname = f"snapshot_{policy_pack_name}_{ts}.json"
        path = os.path.join(SNAPSHOT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)
        print(f"[SNAPSHOT] Saved snapshot {path}")
    except Exception as e:
        print(f"[SNAPSHOT] Failed to snapshot: {e}")


def rollback_last_snapshot():
    try:
        snaps = [f for f in os.listdir(SNAPSHOT_DIR) if f.startswith("snapshot_") and f.endswith(".json")]
        if not snaps:
            print("[ROLLBACK] No snapshots available.")
            return None
        snaps.sort()
        last = snaps[-1]
        path = os.path.join(SNAPSHOT_DIR, last)
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        name = snap["policy_pack_name"]
        pack = snap["policy_pack"]
        if name and pack:
            POLICY_PACKS[name] = pack
            SETTINGS.set("last_policy_pack", name)
            print(f"[ROLLBACK] Restored policy pack {name} from snapshot.")
            return name
        else:
            print("[ROLLBACK] Snapshot invalid.")
            return None
    except Exception as e:
        print(f"[ROLLBACK] Failed: {e}")
        return None


def apply_policy_pack(policy_pack_name, dry_run=False):
    if policy_pack_name not in POLICY_PACKS:
        raise ValueError(f"Unknown policy pack: {policy_pack_name}")

    snapshot_state(policy_pack_name)
    create_system_restore_point(f"Codex Sentinel apply {policy_pack_name}")

    pack = POLICY_PACKS[policy_pack_name]
    profile = pack["profile"]
    modules = pack["modules"]

    print_header(f"APPLYING POLICY PACK: {policy_pack_name} (profile={profile}, version={pack['version']})")

    for module in modules:
        try:
            if module == "defender":
                harden_defender(dry_run=dry_run)
            elif module == "smartscreen":
                harden_smart_screen(dry_run=dry_run)
            elif module == "rdp":
                harden_rdp(profile=profile, dry_run=dry_run)
            elif module == "smb_legacy":
                harden_smb_and_legacy(dry_run=dry_run)
            elif module == "firewall":
                harden_firewall(profile=profile, dry_run=dry_run)
            elif module == "app_control_prep":
                prepare_app_control(dry_run=dry_run)
            elif module == "boot_bitlocker":
                harden_boot_and_bitlocker(profile=profile, dry_run=dry_run)
            elif module == "identity_lsa":
                harden_identity_and_lsa(dry_run=dry_run)
            elif module in PLUGINS:
                print(f"[PLUGIN] Applying plugin module: {module}")
                PLUGINS[module](profile, dry_run)
            else:
                print(f"[WARN] Unknown module in policy pack: {module}")
        except Exception as e:
            print(f"[POLICY ERROR] Module '{module}' failed: {e}")

    print_header(f"POLICY PACK {policy_pack_name} COMPLETE")
    organism_notify("policy_pack_applied", {"pack": policy_pack_name, "dry_run": dry_run})
    SETTINGS.set("last_policy_pack", policy_pack_name)
    return pack


# ---------------------------
# Policy diff visualizer
# ---------------------------

def diff_policy_packs(local_pack, remote_pack):
    lines = []
    lines.append(f"Local version : {local_pack.get('version')}")
    lines.append(f"Remote version: {remote_pack.get('version')}")
    lines.append("")

    local_modules = set(local_pack.get("modules", []))
    remote_modules = set(remote_pack.get("modules", []))

    added = remote_modules - local_modules
    removed = local_modules - remote_modules
    common = local_modules & remote_modules

    if added:
        lines.append("Modules added in remote:")
        for m in sorted(added):
            lines.append(f"  + {m}")
        lines.append("")

    if removed:
        lines.append("Modules removed in remote:")
        for m in sorted(removed):
            lines.append(f"  - {m}")
        lines.append("")

    if common:
        lines.append("Modules in both (unchanged or internally different):")
        for m in sorted(common):
            lines.append(f"  = {m}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------
# Autonomous anomaly detection + policy evolution
# ---------------------------

def detect_anomalies(telemetry):
    level = "LOW"
    reasons = []

    if telemetry["defender"].get("PUAProtection") in (0, "0", "unknown"):
        level = "MEDIUM"
        reasons.append("PUAProtection disabled or unknown")

    if telemetry["identity_lsa"].get("RunAsPPL") in (0, "0", "unknown"):
        if level in ("LOW", "MEDIUM"):
            level = "MEDIUM"
        reasons.append("LSA protection not enforced")

    smb1 = telemetry["smb_legacy"].get("SMB1Protocol")
    if smb1 in (1, "1"):
        level = "HIGH"
        reasons.append("SMB1 enabled")

    fw_profiles = telemetry["firewall"].get("profiles")
    if isinstance(fw_profiles, list):
        disabled = [p for p in fw_profiles if not p.get("Enabled")]
        if disabled:
            if level in ("LOW", "MEDIUM"):
                level = "HIGH"
            reasons.append("Firewall profile(s) disabled")

    etw_events = telemetry["kernel"].get("etw_events", [])
    ebpf_events = telemetry["kernel"].get("ebpf_events", [])
    if etw_events or ebpf_events:
        if level in ("LOW", "MEDIUM", "HIGH"):
            level = "CRITICAL"
        reasons.append("Kernel telemetry indicates suspicious activity")

    if level == "LOW":
        reason = "No significant anomalies detected"
    else:
        reason = "; ".join(reasons)

    return level, reason


def evolve_policy_locally(telemetry, current_pack_name):
    pack = POLICY_PACKS.get(current_pack_name)
    if not pack:
        return None

    modules = set(pack["modules"])
    suggestions = []

    if telemetry["smb_legacy"].get("SMB1Protocol") in (1, "1"):
        if "smb_legacy" not in modules:
            suggestions.append("Add smb_legacy module to disable SMB1")

    if telemetry["identity_lsa"].get("RunAsPPL") in (0, "0", "unknown"):
        if "identity_lsa" not in modules:
            suggestions.append("Add identity_lsa module to enforce LSA protection")

    if telemetry["defender"].get("PUAProtection") in (0, "0", "unknown"):
        if "defender" not in modules:
            suggestions.append("Add defender module to enforce PUAProtection")

    if not suggestions:
        return None

    try:
        new_version = str(round(float(pack["version"]) + 0.1, 1))
    except Exception:
        new_version = "1.1"

    suggestion = {
        "base_pack": current_pack_name,
        "new_version": new_version,
        "suggestions": suggestions,
    }
    return suggestion


def apply_policy_evolution_suggestion(suggestion):
    base = suggestion["base_pack"]
    new_version = suggestion["new_version"]
    base_pack = POLICY_PACKS.get(base)
    if not base_pack:
        return None

    new_name = f"{base_pack['profile']}-{new_version}"
    new_modules = set(base_pack["modules"])
    for s in suggestion["suggestions"]:
        if "smb_legacy" in s:
            new_modules.add("smb_legacy")
        if "identity_lsa" in s:
            new_modules.add("identity_lsa")
        if "defender" in s:
            new_modules.add("defender")

    POLICY_PACKS[new_name] = {
        "profile": base_pack["profile"],
        "version": new_version,
        "modules": sorted(new_modules),
    }
    print(f"[EVOLUTION] Created new local policy pack: {new_name}")
    return new_name


def queue_pending_action(level, reason, suggestion=None):
    global PENDING_ACTIONS
    action_id = int(time.time() * 1000)
    PENDING_ACTIONS.append({
        "id": action_id,
        "level": level,
        "reason": reason,
        "suggestion": suggestion,
    })
    return action_id


# ---------------------------
# Async event bus (UDP) + Raft messages
# ---------------------------

async def _async_send_event(payload):
    try:
        loop = asyncio.get_running_loop()
        wrapper = sign_payload(payload)
        data = json.dumps(wrapper).encode("utf-8")
        transport, _ = await loop.create_datagram_endpoint(
            lambda: asyncio.DatagramProtocol(),
            remote_addr=("255.255.255.255", EVENT_BUS_PORT),
            allow_broadcast=True,
        )
        transport.sendto(data)
        transport.close()
    except Exception as e:
        print(f"[BUS SEND ERROR] {e}")


def send_event_bus(payload):
    try:
        EVENT_BUS_QUEUE.put_nowait(payload)
    except Exception as e:
        print(f"[BUS QUEUE ERROR] {e}")


async def _event_bus_sender_loop():
    while not EVENT_BUS_STOP:
        try:
            payload = await asyncio.get_running_loop().run_in_executor(
                None, EVENT_BUS_QUEUE.get
            )
            await _async_send_event(payload)
        except Exception as e:
            print(f"[BUS SENDER LOOP ERROR] {e}")
        await asyncio.sleep(0.01)


class EventBusListenerProtocol(asyncio.DatagramProtocol):
    def __init__(self, hud=None):
        super().__init__()
        self.hud = hud

    def datagram_received(self, data, addr):
        try:
            wrapper = json.loads(data.decode("utf-8"))
            msg = verify_payload(wrapper)
            if msg is None:
                return
        except Exception:
            return

        # Raft messages
        if "raft" in msg:
            raft_msg = msg["raft"]
            t = raft_msg.get("type")
            if t == "raft_request_vote":
                raft_handle_request_vote(raft_msg)
            elif t == "raft_request_vote_reply":
                raft_handle_request_vote_reply(raft_msg)
            elif t == "raft_append_entries":
                raft_handle_append_entries(raft_msg)
            elif t == "raft_append_entries_reply":
                raft_handle_append_entries_reply(raft_msg)
            return

        node_id = msg.get("node_id") or msg.get("data", {}).get("node_id")
        if msg.get("event") == "threat_update" and node_id:
            level = msg.get("data", {}).get("level", "UNKNOWN")
            reason = msg.get("data", {}).get("reason", "")
            THREAT_MATRIX[node_id] = {
                "level": level,
                "reason": reason,
                "timestamp": int(time.time()),
            }

        if self.hud:
            try:
                self.hud.log(f"[BUS] {addr[0]}: {msg.get('event')}")
                self.hud.refresh_threat_matrix()
            except Exception:
                pass


def start_async_event_bus(hud=None):
    global EVENT_LOOP
    if EVENT_LOOP is not None:
        return

    def loop_thread():
        global EVENT_LOOP
        EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(EVENT_LOOP)
        try:
            listen = EVENT_LOOP.create_datagram_endpoint(
                lambda: EventBusListenerProtocol(hud=hud),
                local_addr=("", EVENT_BUS_PORT),
                allow_broadcast=True,
            )
            EVENT_LOOP.run_until_complete(listen)
            EVENT_LOOP.create_task(_event_bus_sender_loop())

            # Raft tick loop
            async def raft_loop():
                while True:
                    raft_consensus_tick()
                    await asyncio.sleep(0.2)
            EVENT_LOOP.create_task(raft_loop())

            EVENT_LOOP.run_forever()
        except Exception as e:
            print(f"[BUS LOOP ERROR] {e}")
        finally:
            try:
                EVENT_LOOP.close()
            except Exception:
                pass

    t = threading.Thread(target=loop_thread, daemon=True)
    t.start()


# ---------------------------
# Voice command control (optional)
# ---------------------------

def voice_listener(hud=None):
    if not HAS_VOICE:
        if hud:
            hud.log("[VOICE] speech_recognition not installed. Install with: pip install SpeechRecognition pyaudio")
        return

    r = sr.Recognizer()
    mic = None
    try:
        mic = sr.Microphone()
    except Exception as e:
        if hud:
            hud.log(f"[VOICE] Microphone error: {e}")
        return

    if hud:
        hud.log("[VOICE] Listening for commands: 'Codex, pause auto mode', 'Codex, resume auto mode', 'Codex, status'.")

    while True:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio).lower()
            except Exception:
                continue

            if "codex" not in text:
                continue

            if hud:
                hud.log(f"[VOICE] Heard: {text}")

            if "pause auto" in text or "pause autonomous" in text:
                hud.voice_pause_auto()
            elif "resume auto" in text:
                hud.voice_resume_auto()
            elif "status" in text:
                hud.voice_status()
        except Exception:
            continue


# ---------------------------
# Autonomous cycle engine
# ---------------------------

def autonomous_cycle(hud=None):
    global AUTO_MODE, AUTO_STOP, THREAT_LEVEL, THREAT_REASON

    while not AUTO_STOP:
        if AUTO_MODE:
            try:
                pack_name = SETTINGS.get("last_policy_pack", list(POLICY_PACKS.keys())[0])
                if pack_name not in POLICY_PACKS:
                    pack_name = list(POLICY_PACKS.keys())[0]
                pack = POLICY_PACKS[pack_name]
                profile = pack["profile"]

                if hud:
                    hud.log("[AUTO] Running autonomous hardening cycle…")

                apply_policy_pack(pack_name, dry_run=False)

                telemetry = collect_telemetry(profile, pack_name, pack)
                export_telemetry(telemetry)

                level, reason = detect_anomalies(telemetry)
                THREAT_LEVEL = level
                THREAT_REASON = reason
                if hud:
                    hud.update_threat_indicator(level, reason)

                store_telemetry_distributed(telemetry, level, reason)

                send_event_bus({
                    "event": "threat_update",
                    "node_id": SETTINGS.get("node_id"),
                    "data": {"level": level, "reason": reason},
                })

                if level in ("HIGH", "CRITICAL"):
                    suggestion = evolve_policy_locally(telemetry, pack_name)
                    action_id = queue_pending_action(level, reason, suggestion)
                    if hud:
                        hud.log(f"[AUTO] Anomaly detected (level={level}). Threat paused for admin review. Action ID: {action_id}")
                        hud.refresh_pending_actions()
                    AUTO_MODE = False
                    SETTINGS.set("auto_mode", False)
                    if hud:
                        hud.update_auto_button()
                else:
                    if hud:
                        hud.log(f"[AUTO] Threat level: {level} — {reason}")

                resp = swarm_upload_telemetry(telemetry)
                if hud:
                    hud.log(f"[AUTO] Swarm upload: {resp}")

                policy_resp = swarm_fetch_policy_pack()
                if policy_resp and "policy" in policy_resp:
                    remote_policy = policy_resp["policy"]
                    local_pack = pack
                    merged = consensus_merge_policy(local_pack, remote_policy)
                    hud.log(f"[AUTO] Swarm policy merged: version {merged.get('version')}")
                else:
                    if hud:
                        hud.log("[AUTO] No swarm policy received.")

            except Exception as e:
                if hud:
                    hud.log(f"[AUTO ERROR] {e}")

        time.sleep(AUTO_INTERVAL)


# ---------------------------
# Multi-process swarm nodes
# ---------------------------

def swarm_node_worker(node_id, port):
    print(f"[NODE {node_id}] Worker started on port {port} (pid={os.getpid()})")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("", port))
    except Exception as e:
        print(f"[NODE {node_id}] Bind error: {e}")
        return

    while True:
        try:
            s.settimeout(5.0)
            data, addr = s.recvfrom(65535)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[NODE {node_id}] Socket error: {e}")
            break

        try:
            wrapper = json.loads(data.decode("utf-8"))
            msg = verify_payload(wrapper)
            if msg is None:
                continue
        except Exception:
            continue

        print(f"[NODE {node_id}] From {addr[0]}: {msg.get('event')}")

    s.close()
    print(f"[NODE {node_id}] Worker exiting.")


def start_swarm_node(node_id, port=EVENT_BUS_PORT):
    with SWARM_NODE_LOCK:
        if node_id in SWARM_NODE_PROCESSES:
            return
        p = Process(target=swarm_node_worker, args=(node_id, port), daemon=True)
        p.start()
        SWARM_NODE_PROCESSES[node_id] = p
        print(f"[SWARM NODE] Started node {node_id} (pid={p.pid})")


def check_swarm_nodes():
    with SWARM_NODE_LOCK:
        dead = []
        for node_id, proc in SWARM_NODE_PROCESSES.items():
            if not proc.is_alive():
                dead.append(node_id)
        for node_id in dead:
            print(f"[SWARM NODE] Node {node_id} died; restarting.")
            del SWARM_NODE_PROCESSES[node_id]
            start_swarm_node(node_id)


# ---------------------------
# Watchdog daemonization
# ---------------------------

def watchdog_loop(hud=None):
    global WATCHDOG_STOP
    while not WATCHDOG_STOP:
        try:
            check_swarm_nodes()

            if AUTO_THREAD and not AUTO_THREAD.is_alive():
                if hud:
                    hud.log("[WATCHDOG] AUTO thread died; restarting.")
                restart_auto_thread(hud)

            if EVENT_LOOP and EVENT_LOOP.is_closed():
                if hud:
                    hud.log("[WATCHDOG] Event loop closed; restarting.")
                start_async_event_bus(hud)

        except Exception as e:
            print(f"[WATCHDOG ERROR] {e}")
        time.sleep(5)


def restart_auto_thread(hud=None):
    global AUTO_THREAD, AUTO_STOP
    AUTO_STOP = False
    AUTO_THREAD = threading.Thread(target=autonomous_cycle, args=(hud,), daemon=True)
    AUTO_THREAD.start()


def start_watchdog(hud=None):
    global WATCHDOG_THREAD
    if WATCHDOG_THREAD and WATCHDOG_THREAD.is_alive():
        return
    WATCHDOG_THREAD = threading.Thread(target=watchdog_loop, args=(hud,), daemon=True)
    WATCHDOG_THREAD.start()


# ---------------------------
# GPU HUD overlay (pyglet/OpenGL)
# ---------------------------

def start_gpu_hud_overlay():
    if not (HAS_GPU_HUD and SETTINGS.get_nested("gpu_hud", "enabled", default=True)):
        return

    def run():
        window = pyglet.window.Window(width=800, height=200, caption="Codex GPU HUD Overlay", resizable=False)
        label = pyglet.text.Label(
            "Codex Sentinel HUD Overlay",
            font_name="Consolas",
            font_size=14,
            x=10, y=window.height - 20,
            anchor_x="left", anchor_y="center",
            color=(0, 255, 200, 255),
        )

        @window.event
        def on_draw():
            window.clear()
            label.draw()

        pyglet.app.run()

    t = threading.Thread(target=run, daemon=True)
    t.start()


# ---------------------------
# REST API for remote control
# ---------------------------

class CodexRESTHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/status":
            self._send_json({
                "threat_level": THREAT_LEVEL,
                "threat_reason": THREAT_REASON,
                "auto_mode": AUTO_MODE,
                "node_id": SETTINGS.get("node_id"),
                "raft_role": RAFT_STATE["role"],
                "raft_term": RAFT_STATE["current_term"],
                "raft_leader": RAFT_STATE["leader_id"],
            })
        elif self.path == "/policy_packs":
            self._send_json({"packs": POLICY_PACKS})
        elif self.path == "/plugins":
            self._send_json({"plugins": list(PLUGINS.keys())})
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if self.path == "/auto":
            global AUTO_MODE
            mode = payload.get("mode")
            if mode == "on":
                AUTO_MODE = True
                SETTINGS.set("auto_mode", True)
            elif mode == "off":
                AUTO_MODE = False
                SETTINGS.set("auto_mode", False)
            self._send_json({"auto_mode": AUTO_MODE})
        elif self.path == "/apply_policy":
            name = payload.get("pack")
            if name in POLICY_PACKS:
                try:
                    apply_policy_pack(name, dry_run=False)
                    self._send_json({"status": "ok", "pack": name})
                except Exception as e:
                    self._send_json({"status": "error", "error": str(e)}, code=500)
            else:
                self._send_json({"status": "error", "error": "unknown pack"}, code=400)
        elif self.path == "/rollback":
            name = rollback_last_snapshot()
            if name:
                self._send_json({"status": "ok", "restored_pack": name})
            else:
                self._send_json({"status": "error", "error": "no snapshot"}, code=400)
        else:
            self._send_json({"error": "not found"}, code=404)


def start_rest_api():
    global REST_SERVER, REST_THREAD
    cfg = SETTINGS.get("rest_api", {})
    if not cfg.get("enabled", False):
        return

    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 8088))

    def run():
        global REST_SERVER
        REST_SERVER = HTTPServer((host, port), CodexRESTHandler)
        print(f"[REST] Listening on http://{host}:{port}")
        try:
            REST_SERVER.serve_forever()
        except Exception as e:
            print(f"[REST] Server error: {e}")

    REST_THREAD = threading.Thread(target=run, daemon=True)
    REST_THREAD.start()


# ---------------------------
# Tkinter Futuristic Cockpit HUD
# ---------------------------

class CodexHUD(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Codex Sentinel — Zero‑Day Hardening Cockpit")
        self.geometry("1400x850")
        self.configure(bg="#05060A")

        self.current_telemetry = None
        self.last_swarm_policy = None

        self._build_style()
        self._build_layout()

        global AUTO_THREAD
        AUTO_THREAD = threading.Thread(target=autonomous_cycle, args=(self,), daemon=True)
        AUTO_THREAD.start()

        start_async_event_bus(self)
        start_watchdog(self)
        start_rest_api()
        start_gpu_hud_overlay()
        start_etw_listener()

        self.voice_thread = threading.Thread(target=voice_listener, args=(self,), daemon=True)
        self.voice_thread.start()

        start_swarm_node(f"{SETTINGS.get('node_id')}-local")

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#05060A")
        style.configure("TLabel", background="#05060A", foreground="#E0E0E0", font=("Consolas", 10))
        style.configure("Title.TLabel", font=("Consolas", 16, "bold"), foreground="#00FFC8")
        style.configure("Section.TLabel", font=("Consolas", 11, "bold"), foreground="#00BFFF")
        style.configure("TButton", background="#101320", foreground="#E0E0E0", font=("Consolas", 10))
        style.map("TButton",
                  background=[("active", "#1A2035")],
                  foreground=[("active", "#FFFFFF")])
        style.configure("TCombobox", fieldbackground="#101320", background="#101320", foreground="#E0E0E0")

    def _build_layout(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        title = ttk.Label(top_frame, text="Codex Sentinel — Zero‑Day Hardening Cockpit", style="Title.TLabel")
        title.pack(side=tk.LEFT, padx=5)

        right_status = ttk.Frame(top_frame)
        right_status.pack(side=tk.RIGHT)

        self.threat_label = ttk.Label(right_status, text="Threat: LOW", foreground="#00FF00")
        self.threat_label.pack(side=tk.RIGHT, padx=5)

        self.swarm_label = ttk.Label(right_status, text="Swarm: idle", foreground="#AAAAAA")
        self.swarm_label.pack(side=tk.RIGHT, padx=5)

        mid_frame = ttk.Frame(self)
        mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_panel = ttk.Frame(mid_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        right_panel = ttk.Frame(mid_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        pack_label = ttk.Label(left_panel, text="Policy Pack", style="Section.TLabel")
        pack_label.pack(anchor="w", pady=(0, 4))

        self.pack_var = tk.StringVar(value=SETTINGS.get("last_policy_pack", list(POLICY_PACKS.keys())[0]))
        self.pack_combo = ttk.Combobox(left_panel, textvariable=self.pack_var, values=list(POLICY_PACKS.keys()), state="readonly")
        self.pack_combo.pack(fill=tk.X, pady=(0, 10))

        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)

        self.apply_btn = ttk.Button(btn_frame, text="Apply Pack", command=self.on_apply_pack)
        self.apply_btn.pack(fill=tk.X, pady=2)

        self.telemetry_btn = ttk.Button(btn_frame, text="Collect Telemetry", command=self.on_collect_telemetry)
        self.telemetry_btn.pack(fill=tk.X, pady=2)

        self.export_telemetry_btn = ttk.Button(btn_frame, text="Export Telemetry", command=self.on_export_telemetry)
        self.export_telemetry_btn.pack(fill=tk.X, pady=2)

        self.swarm_sync_btn = ttk.Button(btn_frame, text="Sync with Swarm", command=self.on_swarm_sync)
        self.swarm_sync_btn.pack(fill=tk.X, pady=2)

        self.diff_btn = ttk.Button(btn_frame, text="Show Policy Diff", command=self.on_show_diff)
        self.diff_btn.pack(fill=tk.X, pady=2)

        self.rollback_btn = ttk.Button(btn_frame, text="Rollback Snapshot", command=self.on_rollback)
        self.rollback_btn.pack(fill=tk.X, pady=2)

        self.auto_btn = ttk.Button(btn_frame, text="Pause AUTO Mode" if AUTO_MODE else "Resume AUTO Mode", command=self.toggle_auto)
        self.auto_btn.pack(fill=tk.X, pady=2)

        self.voice_btn = ttk.Button(btn_frame, text="Voice Control (passive)", command=self.on_voice_info)
        self.voice_btn.pack(fill=tk.X, pady=2)

        ttk.Label(left_panel, text="Node HUD", style="Section.TLabel").pack(anchor="w", pady=(15, 4))

        self.node_info = tk.Text(left_panel, height=8, bg="#05060A", fg="#00FFC8",
                                 insertbackground="#00FFC8", relief=tk.FLAT, font=("Consolas", 9))
        self.node_info.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        right_top = ttk.Frame(right_panel)
        right_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        right_bottom = ttk.Frame(right_panel)
        right_bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        telemetry_frame = ttk.LabelFrame(right_top, text="Telemetry Snapshot", padding=5)
        telemetry_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.telemetry_text = tk.Text(telemetry_frame, bg="#05060A", fg="#E0E0E0",
                                      insertbackground="#E0E0E0", relief=tk.FLAT, font=("Consolas", 9))
        self.telemetry_text.pack(fill=tk.BOTH, expand=True)

        threat_frame = ttk.LabelFrame(right_top, text="Distributed Threat Matrix", padding=5)
        threat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.threat_matrix_text = tk.Text(threat_frame, bg="#05060A", fg="#FFCC00",
                                          insertbackground="#FFCC00", relief=tk.FLAT, font=("Consolas", 9))
        self.threat_matrix_text.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(right_bottom, text="Event Log", padding=5)
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.log_text = tk.Text(log_frame, bg="#05060A", fg="#A0A0A0",
                                insertbackground="#A0A0A0", relief=tk.FLAT, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        pending_frame = ttk.LabelFrame(right_bottom, text="Pending Actions", padding=5)
        pending_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.pending_text = tk.Text(pending_frame, bg="#05060A", fg="#FF8888",
                                    insertbackground="#FF8888", relief=tk.FLAT, font=("Consolas", 9))
        self.pending_text.pack(fill=tk.BOTH, expand=True)

        self.refresh_pending_actions()
        self.refresh_threat_matrix()

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def update_threat_indicator(self, level, reason):
        color = {
            "LOW": "#00FF00",
            "MEDIUM": "#FFFF00",
            "HIGH": "#FF8800",
            "CRITICAL": "#FF0000",
        }.get(level, "#FFFFFF")
        self.threat_label.configure(text=f"Threat: {level}", foreground=color)
        self.log(f"[THREAT] {level} — {reason}")

    def update_auto_button(self):
        self.auto_btn.configure(text="Pause AUTO Mode" if AUTO_MODE else "Resume AUTO Mode")

    def refresh_pending_actions(self):
        self.pending_text.delete("1.0", tk.END)
        for a in PENDING_ACTIONS:
            self.pending_text.insert(
                tk.END,
                f"ID: {a['id']} | Level: {a['level']} | Reason: {a['reason']}\n"
            )
            if a["suggestion"]:
                self.pending_text.insert(
                    tk.END,
                    f"  Suggestion: {a['suggestion']}\n"
                )

    def refresh_threat_matrix(self):
        self.threat_matrix_text.delete("1.0", tk.END)
        for node_id, info in THREAT_MATRIX.items():
            ts = time.strftime("%H:%M:%S", time.localtime(info["timestamp"]))
            self.threat_matrix_text.insert(
                tk.END,
                f"{node_id} [{ts}] -> {info['level']} | {info['reason']}\n"
            )

    def voice_pause_auto(self):
        self.log("[VOICE] Pausing AUTO mode.")
        global AUTO_MODE
        AUTO_MODE = False
        SETTINGS.set("auto_mode", False)
        self.update_auto_button()

    def voice_resume_auto(self):
        self.log("[VOICE] Resuming AUTO mode.")
        global AUTO_MODE
        AUTO_MODE = True
        SETTINGS.set("auto_mode", True)
        self.update_auto_button()

    def voice_status(self):
        self.log(f"[VOICE] Status: Threat={THREAT_LEVEL}, AUTO={'ON' if AUTO_MODE else 'OFF'}")

    def on_apply_pack(self):
        pack_name = self.pack_var.get()
        try:
            apply_policy_pack(pack_name, dry_run=False)
            self.log(f"[HUD] Applied policy pack: {pack_name}")
        except Exception as e:
            self.log(f"[HUD ERROR] Failed to apply pack: {e}")

    def on_collect_telemetry(self):
        pack_name = self.pack_var.get()
        pack = POLICY_PACKS.get(pack_name)
        if not pack:
            self.log(f"[HUD ERROR] Unknown pack: {pack_name}")
            return
        profile = pack["profile"]
        try:
            telemetry = collect_telemetry(profile, pack_name, pack)
            self.current_telemetry = telemetry
            self.telemetry_text.delete("1.0", tk.END)
            self.telemetry_text.insert(tk.END, json.dumps(telemetry, indent=2))
            self.log("[HUD] Telemetry collected.")
        except Exception as e:
            self.log(f"[HUD ERROR] Telemetry collection failed: {e}")

    def on_export_telemetry(self):
        if not self.current_telemetry:
            self.log("[HUD] No telemetry to export.")
            return
        try:
            path = export_telemetry(self.current_telemetry)
            self.log(f"[HUD] Telemetry exported to {path}")
        except Exception as e:
            self.log(f"[HUD ERROR] Telemetry export failed: {e}")

    def on_swarm_sync(self):
        self.swarm_label.configure(text="Swarm: syncing…", foreground="#00BFFF")
        self.update_idletasks()
        try:
            pack_name = self.pack_var.get()
            local_pack = POLICY_PACKS.get(pack_name)
            if not local_pack:
                self.log(f"[HUD ERROR] Unknown pack: {pack_name}")
                return
            resp = swarm_fetch_policy_pack()
            if resp and "policy" in resp:
                remote_pack = resp["policy"]
                self.last_swarm_policy = remote_pack
                merged = consensus_merge_policy(local_pack, remote_pack)
                diff = diff_policy_packs(local_pack, remote_pack)
                self.log("[HUD] Swarm policy diff:\n" + diff)
                self.log(f"[HUD] Merged policy version: {merged.get('version')}")
            else:
                self.log("[HUD] No swarm policy received.")
        except Exception as e:
            self.log(f"[HUD ERROR] Swarm sync failed: {e}")
        finally:
            self.swarm_label.configure(text="Swarm: idle", foreground="#AAAAAA")

    def on_show_diff(self):
        if not self.last_swarm_policy:
            self.log("[HUD] No swarm policy cached.")
            return
        pack_name = self.pack_var.get()
        local_pack = POLICY_PACKS.get(pack_name)
        if not local_pack:
            self.log(f"[HUD ERROR] Unknown pack: {pack_name}")
            return
        diff = diff_policy_packs(local_pack, self.last_swarm_policy)
        self.log("[HUD] Policy diff:\n" + diff)

    def on_rollback(self):
        name = rollback_last_snapshot()
        if name:
            self.log(f"[HUD] Rolled back to snapshot policy pack: {name}")
            self.pack_var.set(name)
        else:
            self.log("[HUD] No snapshot to roll back to.")

    def toggle_auto(self):
        global AUTO_MODE
        AUTO_MODE = not AUTO_MODE
        SETTINGS.set("auto_mode", AUTO_MODE)
        self.update_auto_button()
        self.log(f"[HUD] AUTO mode {'enabled' if AUTO_MODE else 'paused'}.")

    def on_voice_info(self):
        self.log("[HUD] Voice control is passive; say 'Codex, status' or 'Codex, pause auto mode'.")


# ---------------------------
# CLI mode
# ---------------------------

def run_cli(args):
    pack_name = args.pack or SETTINGS.get("last_policy_pack", list(POLICY_PACKS.keys())[0])
    if pack_name not in POLICY_PACKS:
        print(f"[CLI] Unknown policy pack: {pack_name}")
        return

    pack = POLICY_PACKS[pack_name]
    profile = pack["profile"]

    if args.apply:
        apply_policy_pack(pack_name, dry_run=args.dry_run)

    if args.telemetry:
        telemetry = collect_telemetry(profile, pack_name, pack)
        export_telemetry(telemetry)
        level, reason = detect_anomalies(telemetry)
        store_telemetry_distributed(telemetry, level, reason)
        print(f"[CLI] Threat level: {level} — {reason}")

    if args.auto:
        print("[CLI] Starting autonomous cycle (Ctrl+C to stop)…")
        try:
            autonomous_cycle(hud=None)
        except KeyboardInterrupt:
            print("[CLI] Stopped.")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Codex Sentinel — Zero‑Day Hardening Cockpit (Ultra Raft + ETW Unified)"
    )
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of GUI.")
    parser.add_argument("--apply", action="store_true", help="Apply selected policy pack.")
    parser.add_argument("--telemetry", action="store_true", help="Collect and export telemetry.")
    parser.add_argument("--auto", action="store_true", help="Run autonomous cycle in CLI.")
    parser.add_argument("--dry-run", action="store_true", help="Do not actually change system settings.")
    parser.add_argument("--pack", type=str, help="Policy pack name (e.g., home-1.0).")

    args = parser.parse_args()

    if is_windows():
        ensure_admin()

    if args.cli:
        run_cli(args)
    else:
        app = CodexHUD()
        app.mainloop()


if __name__ == "__main__":
    main()
