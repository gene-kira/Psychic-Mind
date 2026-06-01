#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# codex_sentinel_organism_tk_ultra.py
#
# Unified organism:
#   - Zero-day-resistant Windows hardener (home + server)
#   - Policy pack system (versioned, evolvable)
#   - Telemetry export (JSON) + distributed storage
#   - Async event bus (UDP, non-blocking, signed messages)
#   - Hardened error boundaries + watchdog daemon
#   - Multi-node sync + Raft-style consensus skeleton
#   - Real-ish ETW/eBPF hooks (optional, pluggable)
#   - Distributed threat matrix
#   - Autonomous policy evolution engine + rollback/snapshots
#   - Persistent settings store
#   - Plugin system for hardening modules
#   - Multi-process swarm nodes
#   - Tkinter “Futuristic Cockpit HUD” GUI with telemetry panels
#   - Optional GPU-accelerated HUD overlay (stub)
#   - AI-based anomaly scoring (heuristic + pluggable model)
#   - REST API for remote control (optional)
#   - CLI fallback with --cli
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
import socket
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

# Optional ETW / eBPF (real integration would need extra libs)
try:
    import ctypes.wintypes as wintypes  # type: ignore
    HAS_ETW = platform.system().lower() == "windows"
except Exception:
    HAS_ETW = False

try:
    HAS_EBPF = platform.system().lower() == "linux"
except Exception:
    HAS_EBPF = False

# Optional GPU HUD (stub – could be wired to OpenGL/DirectX)
try:
    import pyglet  # type: ignore
    HAS_GPU_HUD = True
except Exception:
    HAS_GPU_HUD = False

# Optional AI model (placeholder – user can wire real model)
try:
    import joblib  # type: ignore
    HAS_AI_MODEL = True
except Exception:
    HAS_AI_MODEL = False

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
# Plugin system for modules
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


load_plugins()

# ---------------------------
# Swarm / sync configuration
# ---------------------------

SWARM_CONFIG = SETTINGS.get("swarm")

# ---------------------------
# Autonomy / threat / event bus
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

# Raft-style consensus (skeleton)
RAFT_STATE = {
    "term": 0,
    "leader": None,
    "log": [],  # list of {"term": int, "entry": dict}
}

RAFT_LOCK = threading.Lock()

# Snapshots / rollback
SNAPSHOT_DIR = os.path.join(BASE_DIR, "codex_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Distributed telemetry storage
TELEMETRY_STORE_DIR = os.path.join(BASE_DIR, "codex_telemetry_store")
os.makedirs(TELEMETRY_STORE_DIR, exist_ok=True)

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
            timeout=60
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
# Crypto signing for swarm messages
# ---------------------------

def sign_payload(payload: dict) -> dict:
    secret = SWARM_CONFIG.get("shared_secret", "CHANGE_ME_SECRET").encode("utf-8")
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return {"body": payload, "sig": sig}


def verify_payload(wrapper: dict) -> dict | None:
    try:
        secret = SWARM_CONFIG.get("shared_secret", "CHANGE_ME_SECRET").encode("utf-8")
        body = wrapper["body"]
        sig = wrapper["sig"]
        raw = json.dumps(body, sort_keys=True).encode("utf-8")
        expected = hmac.new(secret, raw, hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return body
        else:
            print("[CRYPTO] Signature mismatch; dropping message.")
            return None
    except Exception as e:
        print(f"[CRYPTO] Verification error: {e}")
        return None


# ---------------------------
# Organism / swarm hooks
# ---------------------------

def organism_notify(event, data=None):
    print(f"[ORGANISM] Event: {event}, Data: {str(data)[:200]}")
    send_event_bus({"event": event, "data": data})


# ---------------------------
# AI / anomaly scoring
# ---------------------------

def ai_score_system_risk(context, telemetry=None):
    """
    Hybrid heuristic + optional model-based scoring.
    Returns float in [0,1].
    """
    base_score = 0.2

    if telemetry:
        # Simple heuristic features
        features = {
            "pua_unknown": telemetry["defender"].get("PUAProtection") in (0, "0", "unknown"),
            "lsa_weak": telemetry["identity_lsa"].get("RunAsPPL") in (0, "0", "unknown"),
            "smb1_on": telemetry["smb_legacy"].get("SMB1Protocol") in (1, "1"),
            "fw_disabled": False,
        }
        fw_profiles = telemetry["firewall"].get("profiles")
        if isinstance(fw_profiles, list):
            disabled = [p for p in fw_profiles if not p.get("Enabled")]
            features["fw_disabled"] = bool(disabled)

        score = base_score
        if features["pua_unknown"]:
            score += 0.15
        if features["lsa_weak"]:
            score += 0.2
        if features["smb1_on"]:
            score += 0.3
        if features["fw_disabled"]:
            score += 0.25

        # Optional ML model
        if AI_MODEL is not None:
            try:
                vec = [
                    int(features["pua_unknown"]),
                    int(features["lsa_weak"]),
                    int(features["smb1_on"]),
                    int(features["fw_disabled"]),
                ]
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


def store_telemetry_distributed(telemetry):
    """
    Local distributed storage stub: writes to a store directory.
    In a real deployment, this could push to a DB, object store, or message bus.
    """
    try:
        node_id = telemetry["meta"]["node_id"]
        ts = telemetry["meta"]["timestamp"]
        fname = f"{node_id}_{ts}.json"
        path = os.path.join(TELEMETRY_STORE_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        print(f"[STORE] Telemetry stored at {path}")
    except Exception as e:
        print(f"[STORE] Failed to store telemetry: {e}")


# ---------------------------
# ETW / eBPF ingestion (stubs with hooks)
# ---------------------------

def collect_etw_events():
    if not HAS_ETW:
        return []
    # Real implementation would hook ETW providers and stream events.
    # Here we just return a placeholder event.
    return [
        {"provider": "Security", "id": 4624, "level": "Information", "ts": int(time.time())},
    ]


def collect_ebpf_events():
    if not HAS_EBPF:
        return []
    # Real implementation would attach eBPF programs and collect events.
    return [
        {"program": "sys_enter_execve", "pid": 1234, "ts": int(time.time())},
    ]


# ---------------------------
# Swarm sync protocol + consensus
# ---------------------------

def swarm_upload_telemetry(telemetry):
    if not SWARM_CONFIG.get("enabled", False):
        print("[SWARM] Upload disabled (SWARM_CONFIG.enabled = False).")
        return None

    if requests is None:
        print("[SWARM] 'requests' not installed; cannot upload telemetry.")
        return None

    url = SWARM_CONFIG["base_url"].rstrip("/") + "/upload_telemetry"
    headers = {"X-API-Key": SWARM_CONFIG["api_key"]}
    try:
        wrapper = sign_payload(telemetry)
        resp = requests.post(url, headers=headers, json=wrapper, timeout=10)
        print(f"[SWARM] Upload status: {resp.status_code}")
        return resp.json()
    except Exception as e:
        print(f"[SWARM] Upload failed: {e}")
        return None


def swarm_fetch_policy_pack():
    if not SWARM_CONFIG.get("enabled", False):
        print("[SWARM] Fetch disabled (SWARM_CONFIG.enabled = False).")
        return None

    if requests is None:
        print("[SWARM] 'requests' not installed; cannot fetch policy pack.")
        return None

    url = SWARM_CONFIG["base_url"].rstrip("/") + "/suggest_policy_pack"
    headers = {"X-API-Key": SWARM_CONFIG["api_key"]}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"[SWARM] Fetch status: {resp.status_code}")
        if resp.status_code == 200:
            wrapper = resp.json()
            body = verify_payload(wrapper) if isinstance(wrapper, dict) and "body" in wrapper else wrapper
            return body
        return None
    except Exception as e:
        print(f"[SWARM] Fetch failed: {e}")
        return None


def consensus_merge_policy(local_pack, remote_pack):
    """
    Simple version-based merge + Raft-style log append.
    """
    try:
        lv = float(local_pack.get("version", "0"))
    except Exception:
        lv = 0.0
    try:
        rv = float(remote_pack.get("version", "0"))
    except Exception:
        rv = 0.0

    with RAFT_LOCK:
        RAFT_STATE["term"] += 1
        RAFT_STATE["log"].append({"term": RAFT_STATE["term"], "entry": remote_pack})

    if rv > lv:
        print("[CONSENSUS] Remote policy newer; adopting remote.")
        return remote_pack
    elif rv < lv:
        print("[CONSENSUS] Local policy newer; keeping local.")
        return local_pack
    else:
        print("[CONSENSUS] Same version; merging modules.")
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
# Policy pack application + snapshots
# ---------------------------

def snapshot_state(policy_pack_name):
    """
    Snapshot current policy pack + timestamp for rollback.
    """
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
    """
    Roll back to the most recent snapshot (policy pack only).
    """
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
    """
    Heuristic anomaly detector + kernel telemetry.
    Returns (level, reason).
    """
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
    """
    Suggest local policy evolution based on anomalies.
    Returns a suggestion dict or None.
    """
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
    """
    Materialize a suggested policy evolution into POLICY_PACKS (local only).
    """
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
# Async event bus (UDP)
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
                store_telemetry_distributed(telemetry)

                level, reason = detect_anomalies(telemetry)
                THREAT_LEVEL = level
                THREAT_REASON = reason
                if hud:
                    hud.update_threat_indicator(level, reason)

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
# GPU HUD overlay (stub)
# ---------------------------

def start_gpu_hud_overlay():
    if not HAS_GPU_HUD:
        print("[GPU HUD] pyglet not installed; skipping GPU overlay.")
        return

    def run():
        window = pyglet.window.Window(width=800, height=200, caption="Codex GPU HUD Overlay")
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
            })
        elif self.path == "/policy_packs":
            self._send_json({"packs": POLICY_PACKS})
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
        print("[REST] REST API disabled in settings.")
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
        store_telemetry_distributed(telemetry)
        level, reason = detect_anomalies(telemetry)
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
        description="Codex Sentinel — Zero‑Day Hardening Cockpit (Ultra Unified)"
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
