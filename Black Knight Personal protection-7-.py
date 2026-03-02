from __future__ import annotations

import sys
import os
import json
import time
import secrets
import pathlib
import importlib
import hashlib
import subprocess
import platform
import threading
import queue
from typing import Optional, List, Dict, Any

import pythoncom  # COM init for worker threads

import tkinter as tk
from tkinter import ttk

from cryptography.fernet import Fernet
import psutil
import numpy as np
from sklearn.ensemble import IsolationForest

IS_WINDOWS = platform.system().lower().startswith("win")

if IS_WINDOWS:
    import uiautomation as auto
else:
    auto = None

# ============================================================
# Autoloader
# ============================================================

REQUIRED_LIBS = [
    ("cryptography.fernet", "cryptography"),
    ("psutil", "psutil"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
]

if IS_WINDOWS:
    REQUIRED_LIBS.append(("uiautomation", "uiautomation"))


def ensure_deps():
    missing = []
    for module_path, pip_name in REQUIRED_LIBS:
        try:
            importlib.import_module(module_path.split(".")[0])
        except ImportError:
            missing.append((module_path, pip_name))

    if missing:
        print("Missing dependencies:")
        for m, pip_name in missing:
            print(f" - {m} (install: pip install {pip_name})")
        sys.exit(1)


ensure_deps()

# ============================================================
# Safe process / localhost rules
# ============================================================

SAFE_PROCESS_NAMES = {
    "python.exe",
    "pythonw.exe",
    "py.exe",
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "opera_browser.exe",
    "steam.exe",
    "epicgameslauncher.exe",
    "epicgameslauncher.exe".lower(),
}

SAFE_LOCALHOST_IPS = {"127.0.0.1", "::1"}


def is_safe_process(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        p = psutil.Process(pid)
        name = (p.name() or "").lower()
        return name in SAFE_PROCESS_NAMES
    except Exception:
        return False


# ============================================================
# Root of Trust
# ============================================================

class RootOfTrustOrgan:
    def get_device_secret(self) -> bytes:
        raise NotImplementedError


class FileRootOfTrustOrgan(RootOfTrustOrgan):
    def __init__(self):
        self._cached_secret: Optional[bytes] = None

    def get_device_secret(self) -> bytes:
        if self._cached_secret is not None:
            return self._cached_secret
        secret_path = os.path.join(os.path.dirname(__file__), ".device_secret")
        if os.path.exists(secret_path):
            with open(secret_path, "rb") as f:
                self._cached_secret = f.read()
        else:
            self._cached_secret = os.urandom(32)
            with open(secret_path, "wb") as f:
                f.write(self._cached_secret)
        return self._cached_secret


class TPMRootOfTrustOrgan(RootOfTrustOrgan):
    def __init__(self, tpm_key_handle: Optional[int] = None):
        self._fallback = FileRootOfTrustOrgan()
        self._tpm_available = False
        self._tpm_key_handle = tpm_key_handle
        self._cached_secret: Optional[bytes] = None

        if IS_WINDOWS:
            try:
                from tpm2_pytss import ESAPI  # type: ignore
                self._ESAPI = ESAPI
                self._tpm_available = True
            except Exception:
                self._tpm_available = False
        else:
            self._tpm_available = False

    def get_device_secret(self) -> bytes:
        if self._cached_secret is not None:
            return self._cached_secret

        if not self._tpm_available or self._tpm_key_handle is None:
            self._cached_secret = self._fallback.get_device_secret()
            return self._cached_secret

        try:
            with self._ESAPI() as esapi:
                handle = self._tpm_key_handle
                pub, _ = esapi.ReadPublic(handle)
                pub_bytes = bytes(pub.marshal())
                self._cached_secret = hashlib.sha256(pub_bytes).digest()
                return self._cached_secret
        except Exception:
            self._cached_secret = self._fallback.get_device_secret()
            return self._cached_secret


# ============================================================
# Crypto + Store
# ============================================================

def base64_urlsafe_32(raw: bytes) -> bytes:
    import base64
    return base64.urlsafe_b64encode(raw[:32])


class BlackKnightCrypto:
    def __init__(self, device_secret: bytes):
        key_material = hashlib.sha256(device_secret).digest()
        self.key = base64_urlsafe_32(key_material)
        self.fernet = Fernet(self.key)

    def _mirror(self, data: bytes) -> bytes:
        return data[::-1]

    def _chameleon_pad(self, data: bytes) -> bytes:
        pad_len = secrets.randbelow(64)
        padding = secrets.token_bytes(pad_len)
        return pad_len.to_bytes(1, "big") + padding + data

    def _chameleon_unpad(self, data: bytes) -> bytes:
        pad_len = data[0]
        return data[1 + pad_len:]

    def encrypt_bytes(self, data: bytes) -> bytes:
        staged = self._mirror(data)
        staged = self._chameleon_pad(staged)
        return self.fernet.encrypt(staged)

    def decrypt_bytes(self, token: bytes) -> bytes:
        try:
            staged = self.fernet.decrypt(token)
            staged = self._chameleon_unpad(staged)
            return self._mirror(staged)
        except Exception as e:
            raise RuntimeError(f"decrypt_bytes failed: {e}")

    def encrypt_json(self, obj: Any) -> bytes:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return self.encrypt_bytes(raw)

    def decrypt_json(self, token: bytes) -> Any:
        try:
            raw = self.decrypt_bytes(token)
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"decrypt_json failed: {e}")


class BlackKnightStore:
    def __init__(self, crypto: BlackKnightCrypto, base_dir: Optional[str] = None):
        default_base = os.path.join(os.path.expanduser("~"), ".bk_net")
        self.crypto = crypto
        self.base_dir = pathlib.Path(base_dir or default_base)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.bin"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, str]]:
        if not self.index_path.exists():
            return {}
        try:
            with open(self.index_path, "rb") as f:
                token = f.read()
            return self.crypto.decrypt_json(token)
        except Exception:
            try:
                backup = self.index_path.with_suffix(".corrupt")
                if backup.exists():
                    backup.unlink()
                self.index_path.rename(backup)
            except Exception:
                pass
            return {}

    def _save_index(self):
        try:
            token = self.crypto.encrypt_json(self.index)
            with open(self.index_path, "wb") as f:
                f.write(token)
        except Exception:
            pass

    def _random_name(self) -> str:
        return secrets.token_hex(16)

    def set_blob(self, logical_key: str, data: bytes):
        entry = self.index.get(logical_key)
        if not entry:
            entry = {"file": self._random_name()}
            self.index[logical_key] = entry
        path = self.base_dir / entry["file"]
        try:
            token = self.crypto.encrypt_bytes(data)
            with open(path, "wb") as f:
                f.write(token)
            self._save_index()
        except Exception:
            pass

    def get_blob(self, logical_key: str) -> Optional[bytes]:
        entry = self.index.get(logical_key)
        if not entry:
            return None
        path = self.base_dir / entry["file"]
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                token = f.read()
            return self.crypto.decrypt_bytes(token)
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            self.index.pop(logical_key, None)
            self._save_index()
            return None

    def set_json(self, logical_key: str, obj: Any):
        self.set_blob(logical_key, json.dumps(obj).encode("utf-8"))

    def get_json(self, logical_key: str) -> Any:
        blob = self.get_blob(logical_key)
        if blob is None:
            return None
        try:
            return json.loads(blob.decode("utf-8"))
        except Exception:
            self.index.pop(logical_key, None)
            self._save_index()
            return None

    def reset_store(self):
        try:
            for p in self.base_dir.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        for sub in p.rglob("*"):
                            if sub.is_file():
                                sub.unlink()
                        p.rmdir()
                except Exception:
                    pass
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.index = {}
        self._save_index()


# ============================================================
# EventBus
# ============================================================

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List] = {}

    def subscribe(self, topic: str, callback):
        self.subscribers.setdefault(topic, []).append(callback)

    def publish(self, topic: str, payload: Any = None):
        for cb in self.subscribers.get(topic, []):
            try:
                cb(payload)
            except Exception:
                pass


# ============================================================
# AdaptiveAnomalyBrain / ConnectionPolicy
# ============================================================

class AdaptiveAnomalyBrain:
    def __init__(self):
        self.ip_stats: Dict[str, Dict[str, float]] = {}
        self.spike_factor = 4.0
        self.min_samples = 10

    def observe(self, ip: str, ts: float) -> Optional[str]:
        s = self.ip_stats.get(ip)
        if not s:
            s = {"count": 0.0, "last_ts": ts, "mean_interval": 0.0}
            self.ip_stats[ip] = s
            return None

        interval = ts - s["last_ts"]
        s["last_ts"] = ts
        s["count"] += 1

        if s["count"] == 1:
            s["mean_interval"] = interval
            return None

        n = s["count"]
        s["mean_interval"] = s["mean_interval"] + (interval - s["mean_interval"]) / n

        if n > self.min_samples and interval < (s["mean_interval"] / self.spike_factor):
            return f"Anomaly: {ip} interval {interval:.3f}s << mean {s['mean_interval']:.3f}s (n={int(n)})"

        return None


class ConnectionPolicy:
    def __init__(self):
        self.blocked_ports = {23, 3389}
        self.brain = AdaptiveAnomalyBrain()

    def evaluate(self, conn, item: Dict[str, Any]) -> Optional[str]:
        if conn.raddr and conn.raddr.ip in SAFE_LOCALHOST_IPS:
            return None
        if is_safe_process(conn.pid):
            return None

        if conn.raddr:
            if conn.raddr.port in self.blocked_ports:
                return f"Blocked port {conn.raddr.port} to {conn.raddr.ip}"
            verdict = self.brain.observe(conn.raddr.ip, time.time())
            if verdict:
                return verdict
        return None


# ============================================================
# MLAnomalyOrgan
# ============================================================

class MLAnomalyOrgan:
    def __init__(self, bus: EventBus, store: BlackKnightStore):
        self.bus = bus
        self.store = store
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=42,
            warm_start=True,
        )
        self._trained = False
        self._buffer: List[List[float]] = []

        self.bus.subscribe("connection_features", self.on_connection_features)

    def _features_from_conn(self, conn_item: Dict[str, Any]) -> List[float]:
        status_map = {
            "ESTABLISHED": 1.0,
            "SYN_SENT": 2.0,
            "SYN_RECV": 3.0,
            "FIN_WAIT1": 4.0,
            "FIN_WAIT2": 5.0,
            "TIME_WAIT": 6.0,
            "CLOSE": 7.0,
            "CLOSE_WAIT": 8.0,
            "LAST_ACK": 9.0,
            "LISTEN": 10.0,
            "CLOSING": 11.0,
        }
        status_code = status_map.get(conn_item["status"], 0.0)
        try:
            rport = float(conn_item["raddr"].split(":")[-1])
        except Exception:
            rport = 0.0
        pid = float(conn_item["pid"] or 0)
        return [rport, pid, status_code]

    def on_connection_features(self, conns: List[Dict[str, Any]]):
        if not conns:
            return

        filtered = []
        for c in conns:
            if c["raddr"].startswith("127.0.0.1:") or c["raddr"].startswith("[::1]:"):
                continue
            if is_safe_process(c["pid"]):
                continue
            filtered.append(c)

        if not filtered:
            return

        X = np.array([self._features_from_conn(c) for c in filtered], dtype=float)

        if not self._trained:
            self._buffer.extend(X.tolist())
            if len(self._buffer) >= 200:
                self.model.fit(np.array(self._buffer, dtype=float))
                self._trained = True
                self.store.set_json("ml_model_state", {"trained": True, "samples": len(self._buffer)})
            return

        preds = self.model.predict(X)
        for conn_item, pred in zip(filtered, preds):
            if pred == -1:
                verdict = f"ML anomaly: {conn_item['laddr']} -> {conn_item['raddr']} (pid={conn_item['pid']})"
                self.bus.publish("security_violation", verdict)
                self.store.set_json(f"ml_violation_{time.time()}", {
                    "verdict": verdict,
                    "conn": conn_item,
                })


# ============================================================
# Threaded organs (threading.Thread instead of QThread)
# ============================================================

class ConnectionMonitor(threading.Thread):
    def __init__(self, policy: ConnectionPolicy, bus: EventBus, store: BlackKnightStore, poll_interval: float = 3.0):
        super().__init__(daemon=True)
        self.policy = policy
        self.bus = bus
        self.store = store
        self.poll_interval = poll_interval
        self._running = True

    def run(self):
        if IS_WINDOWS:
            pythoncom.CoInitialize()
        try:
            while self._running:
                conns = []
                try:
                    for c in psutil.net_connections(kind="inet"):
                        if not c.raddr:
                            continue
                        if c.raddr.ip in SAFE_LOCALHOST_IPS:
                            continue
                        if is_safe_process(c.pid):
                            continue

                        item = {
                            "laddr": f"{c.laddr.ip}:{c.laddr.port}",
                            "raddr": f"{c.raddr.ip}:{c.raddr.port}",
                            "status": c.status,
                            "pid": c.pid,
                        }
                        conns.append(item)

                        verdict = self.policy.evaluate(c, item)
                        if verdict:
                            self.bus.publish("security_violation", verdict)
                            self.store.set_json(f"violation_{time.time()}", {
                                "verdict": verdict,
                                "conn": item,
                            })
                except Exception as e:
                    self.bus.publish("security_violation", f"Monitor error: {e}")

                self.bus.publish("connection_snapshot", conns)
                self.bus.publish("connection_features", conns)
                time.sleep(self.poll_interval)
        finally:
            if IS_WINDOWS:
                pythoncom.CoUninitialize()

    def stop(self):
        self._running = False


class ProcessProfileOrgan(threading.Thread):
    def __init__(self, bus: EventBus, interval: float = 10.0):
        super().__init__(daemon=True)
        self.bus = bus
        self.interval = interval
        self._running = True

    def run(self):
        if IS_WINDOWS:
            pythoncom.CoInitialize()
        try:
            while self._running:
                profiles: Dict[int, Dict[str, Any]] = {}

                for p in psutil.process_iter(attrs=["pid", "name", "ppid", "username", "create_time"]):
                    info = p.info
                    pid = info["pid"]
                    profiles[pid] = {
                        "pid": pid,
                        "name": info.get("name") or "",
                        "ppid": info.get("ppid") or 0,
                        "username": info.get("username") or "",
                        "create_time": info.get("create_time") or 0.0,
                        "conn_count": 0,
                        "remote_ports": set(),
                        "remote_ips": set(),
                    }

                try:
                    for c in psutil.net_connections(kind="inet"):
                        if not c.raddr or c.pid is None:
                            continue
                        if c.raddr.ip in SAFE_LOCALHOST_IPS:
                            continue
                        if is_safe_process(c.pid):
                            continue
                        pid = c.pid
                        if pid not in profiles:
                            continue
                        prof = profiles[pid]
                        prof["conn_count"] += 1
                        prof["remote_ports"].add(c.raddr.port)
                        prof["remote_ips"].add(c.raddr.ip)
                except Exception:
                    pass

                for prof in profiles.values():
                    prof["remote_port_count"] = len(prof["remote_ports"])
                    prof["remote_ip_count"] = len(prof["remote_ips"])
                    prof["remote_ports"] = list(prof["remote_ports"])
                    prof["remote_ips"] = list(prof["remote_ips"])

                snapshot = {
                    "ts": time.time(),
                    "profiles": list(profiles.values()),
                }

                self.bus.publish("process_profiles", snapshot)

                time.sleep(self.interval)
        finally:
            if IS_WINDOWS:
                pythoncom.CoUninitialize()

    def stop(self):
        self._running = False


class TemporalAnomalyOrgan:
    def __init__(self, bus: EventBus, store: BlackKnightStore):
        self.bus = bus
        self.store = store
        self.history: Dict[int, List[Dict[str, Any]]] = {}
        self.window_size = 12
        self.min_points = 5

        self.bus.subscribe("process_profiles", self.on_profiles)

    def on_profiles(self, snapshot: dict):
        if not snapshot:
            return
        ts = snapshot.get("ts", time.time())
        profiles = snapshot.get("profiles", [])

        for prof in profiles:
            pid = prof["pid"]
            if is_safe_process(pid):
                continue
            series = self.history.setdefault(pid, [])
            series.append({
                "ts": ts,
                "conn_count": prof["conn_count"],
                "remote_ip_count": prof["remote_ip_count"],
                "remote_port_count": prof["remote_port_count"],
            })
            if len(series) > self.window_size:
                series.pop(0)

            if len(series) >= self.min_points:
                self._check_temporal_anomaly(pid, prof, series)

    def _check_temporal_anomaly(self, pid: int, prof: dict, series: List[Dict[str, Any]]):
        conn_counts = [s["conn_count"] for s in series[:-1]]
        if not conn_counts:
            return
        mean_conn = sum(conn_counts) / len(conn_counts)
        current_conn = series[-1]["conn_count"]

        if mean_conn > 0 and current_conn > mean_conn * 5:
            verdict = (
                f"Temporal anomaly: process {prof['name']} (pid={pid}) "
                f"conn_count spike {current_conn} vs mean {mean_conn:.1f}"
            )
            self.bus.publish("security_violation", verdict)
            self.store.set_json(f"temporal_violation_{time.time()}", {
                "verdict": verdict,
                "pid": pid,
                "profile": prof,
                "history": series,
            })


class TemporalMLOrgan:
    def __init__(self, bus: EventBus, store: BlackKnightStore):
        self.bus = bus
        self.store = store
        self.history: Dict[int, List[Dict[str, Any]]] = {}
        self.window = 10
        self.min_points = 8
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.02,
            random_state=42,
            warm_start=True,
        )
        self.trained = False
        self.buffer: List[List[float]] = []

        self.bus.subscribe("process_profiles", self.on_profiles)

    def on_profiles(self, snapshot: dict):
        if not snapshot:
            return
        ts = snapshot.get("ts", time.time())
        profiles = snapshot.get("profiles", [])

        for prof in profiles:
            pid = prof["pid"]
            if is_safe_process(pid):
                continue
            series = self.history.setdefault(pid, [])
            series.append({
                "ts": ts,
                "conn_count": prof["conn_count"],
                "remote_ip_count": prof["remote_ip_count"],
                "remote_port_count": prof["remote_port_count"],
                "age": max(0.0, ts - (prof.get("create_time") or ts)),
            })
            if len(series) > self.window:
                series.pop(0)

            if len(series) >= self.min_points:
                self._evaluate_pid(pid, prof, series)

    def _sequence_vector(self, series: List[Dict[str, Any]]) -> List[float]:
        K = min(len(series), self.window)
        tail = series[-K:]
        vec: List[float] = []
        for s in tail:
            vec.extend([
                float(s["conn_count"]),
                float(s["remote_ip_count"]),
                float(s["remote_port_count"]),
                float(s["age"]),
            ])
        expected_len = self.window * 4
        while len(vec) < expected_len:
            vec.append(0.0)
        return vec

    def _evaluate_pid(self, pid: int, prof: dict, series: List[Dict[str, Any]]):
        x = self._sequence_vector(series)

        if not self.trained:
            self.buffer.append(x)
            if len(self.buffer) >= 200:
                X = np.array(self.buffer, dtype=float)
                self.model.fit(X)
                self.trained = True
                self.store.set_json("temporal_ml_state", {
                    "trained": True,
                    "samples": len(self.buffer),
                })
            return

        X = np.array([x], dtype=float)
        pred = self.model.predict(X)[0]
        if pred == -1:
            verdict = (
                f"Temporal ML anomaly: {prof['name']} (pid={pid}) "
                f"sequence deviates from learned behavior"
            )
            self.bus.publish("security_violation", verdict)
            self.store.set_json(f"temporal_ml_violation_{time.time()}", {
                "verdict": verdict,
                "pid": pid,
                "profile": prof,
                "series": series,
            })


class ProcessLineageOrgan(threading.Thread):
    def __init__(self, bus: EventBus, interval: float = 15.0):
        super().__init__(daemon=True)
        self.bus = bus
        self.interval = interval
        self._running = True

    def run(self):
        if IS_WINDOWS:
            pythoncom.CoInitialize()
        try:
            while self._running:
                try:
                    procs = {p.pid: p for p in psutil.process_iter(attrs=["pid", "name", "ppid", "exe"])}
                    for pid, p in procs.items():
                        if is_safe_process(pid):
                            continue
                        try:
                            chain = self._build_chain(pid, procs)
                            if self._is_suspicious_chain(chain):
                                msg = "Lineage anomaly: " + " -> ".join(
                                    f"{c['name']}({c['pid']})" for c in chain
                                )
                                self.bus.publish("security_violation", msg)
                                self.bus.publish("lineage_event", msg)
                        except Exception:
                            continue
                except Exception:
                    pass

                time.sleep(self.interval)
        finally:
            if IS_WINDOWS:
                pythoncom.CoUninitialize()

    def _build_chain(self, pid: int, procs: Dict[int, psutil.Process]) -> List[Dict[str, Any]]:
        chain: List[Dict[str, Any]] = []
        current = pid
        depth = 0
        max_depth = 6
        while current in procs and depth < max_depth:
            p = procs[current]
            info = p.info
            chain.append({
                "pid": info["pid"],
                "ppid": info.get("ppid") or 0,
                "name": info.get("name") or "",
                "exe": info.get("exe") or "",
            })
            current = info.get("ppid") or 0
            depth += 1
        chain.reverse()
        return chain

    def _is_suspicious_chain(self, chain: List[Dict[str, Any]]) -> bool:
        if len(chain) < 3:
            return False
        names = [c["name"].lower() for c in chain]
        office_like = any(n.startswith(("winword", "excel", "powerpnt")) for n in names)
        powershell_like = any("powershell" in n for n in names)
        unknown_leaf = chain[-1]["exe"] == ""
        if office_like and powershell_like and unknown_leaf:
            return True
        return False

    def stop(self):
        self._running = False


class KernelTelemetryOrgan(threading.Thread):
    def __init__(self, bus: EventBus):
        super().__init__(daemon=True)
        self.bus = bus
        self._running = True

    def run(self):
        if IS_WINDOWS:
            pythoncom.CoInitialize()
        try:
            while self._running:
                time.sleep(10)
                msg = f"KernelTelemetryOrgan heartbeat ({platform.system()})"
                self.bus.publish("kernel_telemetry", {"type": "heartbeat", "msg": msg})
        finally:
            if IS_WINDOWS:
                pythoncom.CoUninitialize()

    def stop(self):
        self._running = False


# ============================================================
# FirewallGuardian
# ============================================================

class FirewallGuardian:
    RULE_GROUP = "BlackKnightLockdown"
    ALLOW_GROUP = "BlackKnightAllow"

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.lockdown = False

    def _run_netsh(self, args: List[str]):
        if not IS_WINDOWS:
            return
        try:
            subprocess.run(["netsh"] + args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _ensure_block_rule_group(self):
        if not IS_WINDOWS:
            return
        self._run_netsh([
            "advfirewall", "firewall", "add", "rule",
            f"name={self.RULE_GROUP}",
            f"group={self.RULE_GROUP}",
            "dir=out", "action=block", "enable=yes", "profile=any"
        ])

    def _delete_block_rule_group(self):
        if not IS_WINDOWS:
            return
        self._run_netsh([
            "advfirewall", "firewall", "delete", "rule",
            f"group={self.RULE_GROUP}"
        ])

    def _ensure_allowlist_rules(self):
        if not IS_WINDOWS:
            return
        exe_candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files\Opera\launcher.exe",
            r"C:\Program Files (x86)\Steam\steam.exe",
            r"C:\Program Files (x86)\Epic Games\Launcher\Portal\Binaries\Win64\EpicGamesLauncher.exe",
            sys.executable,
        ]
        for exe in exe_candidates:
            if not os.path.exists(exe):
                continue
            self._run_netsh([
                "advfirewall", "firewall", "add", "rule",
                f"name={self.ALLOW_GROUP}_{os.path.basename(exe)}",
                f"group={self.ALLOW_GROUP}",
                "dir=out", "action=allow", "enable=yes", "profile=any",
                f"program={exe}"
            ])

    def _delete_allowlist_rules(self):
        if not IS_WINDOWS:
            return
        self._run_netsh([
            "advfirewall", "firewall", "delete", "rule",
            f"group={self.ALLOW_GROUP}"
        ])

    def set_lockdown(self, enabled: bool):
        self.lockdown = enabled
        if enabled:
            self._ensure_block_rule_group()
            self._ensure_allowlist_rules()
        else:
            self._delete_block_rule_group()
            self._delete_allowlist_rules()
        self.bus.publish("lockdown_changed", enabled)

    def is_lockdown(self) -> bool:
        return self.lockdown


# ============================================================
# SecurityPostureMonitor
# ============================================================

class SecurityPostureMonitor:
    def __init__(self, bus: EventBus, store: BlackKnightStore):
        self.bus = bus
        self.store = store
        self.last_violation: Optional[str] = None
        self.last_kernel_msg: Optional[str] = None
        self.bus.subscribe("security_violation", self.on_violation)
        self.bus.subscribe("lockdown_changed", self.on_lockdown_changed)
        self.bus.subscribe("kernel_telemetry", self.on_kernel_telemetry)

    def on_violation(self, verdict: str):
        self.last_violation = verdict
        self.store.set_json("last_violation", {"verdict": verdict, "ts": time.time()})

    def on_lockdown_changed(self, enabled: bool):
        self.store.set_json("lockdown_state", {"enabled": enabled, "ts": time.time()})

    def on_kernel_telemetry(self, payload: Dict[str, Any]):
        msg = payload.get("msg", "")
        self.last_kernel_msg = msg        # noqa
        self.store.set_json("kernel_telemetry", {"msg": msg, "ts": time.time()})


# ============================================================
# Fate Policy + AutoFill
# ============================================================

class FatePolicyOrgan:
    def __init__(self, store: BlackKnightStore):
        self.store = store
        self.policies = self.store.get_json("fate_policies") or {}

    def save(self):
        self.store.set_json("fate_policies", self.policies)

    def get_policy(self, domain: str) -> Dict[str, Any]:
        return self.policies.get(domain, {
            "trust": "gray",
            "allow_email": True,
            "allow_username": True,
            "allow_password": False,
        })

    def set_policy(self, domain: str, policy: Dict[str, Any]):
        self.policies[domain] = policy
        self.save()

    def is_allowed(self, domain: str, field_type: str) -> bool:
        pol = self.get_policy(domain)
        if pol.get("trust") == "blocked":
            return False
        if field_type == "email":
            return pol.get("allow_email", False)
        if field_type == "username":
            return pol.get("allow_username", False)
        if field_type == "password":
            return pol.get("allow_password", False)
        return False


class FateAutoFillOrgan(threading.Thread):
    def __init__(self, store: BlackKnightStore, bus: EventBus, policy: FatePolicyOrgan):
        super().__init__(daemon=True)
        self.store = store
        self.bus = bus
        self.policy = policy
        self._running = True

    def _infer_domain(self) -> str:
        if not IS_WINDOWS or auto is None:
            return "unknown"
        try:
            w = auto.GetForegroundControl()
            name = (w.Name or "").strip()
            if not name:
                return "unknown"
            return name.lower()
        except Exception:
            return "unknown"

    def _classify_field_type(self, name: str) -> str:
        n = name.lower()
        if "email" in n:
            return "email"
        if "user" in n or "login" in n:
            return "username"
        if "pass" in n:
            return "password"
        return "other"

    def run(self):
        if IS_WINDOWS and auto is not None:
            pythoncom.CoInitialize()
        try:
            if not IS_WINDOWS or auto is None:
                while self._running:
                    self.bus.publish("fate_fields_detected", [])
                    time.sleep(2)
                return

            while self._running:
                try:
                    window = auto.GetForegroundControl()
                    if not window:
                        time.sleep(1)
                        continue

                    edits = window.GetChildren(controlType=auto.ControlType.Edit)
                    fields = []

                    for e in edits:
                        name = e.Name or ""
                        aid = e.AutomationId or ""
                        rect = e.BoundingRectangle

                        fields.append({
                            "name": name,
                            "automation_id": aid,
                            "rect": str(rect),
                            "control": e
                        })

                    self.bus.publish("fate_fields_detected", fields)

                except Exception as e:
                    self.bus.publish("fate_event", f"AutoFill error: {e}")

                time.sleep(1)
        finally:
            if IS_WINDOWS and auto is not None:
                pythoncom.CoUninitialize()

    def fill_now(self, fields: List[Dict[str, Any]]):
        if not IS_WINDOWS or auto is None:
            self.bus.publish("fate_event", "Fate: auto-fill not available on this OS.")
            return

        domain = self._infer_domain()
        self.bus.publish("fate_event", f"Fate: inferred domain key '{domain}'")

        for f in fields:
            name = (f["name"] or "")
            ftype = self._classify_field_type(name)
            if ftype == "other":
                continue
            if not self.policy.is_allowed(domain, ftype):
                self.bus.publish(
                    "fate_event",
                    f"Fate: policy denies filling {ftype} on domain '{domain}'"
                )
                continue

            ctrl = f["control"]
            try:
                if ftype == "email":
                    email = self.store.get_json("user_email")
                    if email:
                        ctrl.SetValue(email)
                        self.bus.publish("fate_event", f"Filled email on '{domain}'")
                elif ftype == "username":
                    user = self.store.get_json("username")
                    if user:
                        ctrl.SetValue(user)
                        self.bus.publish("fate_event", f"Filled username on '{domain}'")
                elif ftype == "password":
                    pwd = self.store.get_json("password")
                    if pwd:
                        ctrl.SetValue(pwd)
                        self.bus.publish("fate_event", f"Filled password on '{domain}'")
            except Exception:
                pass


# ============================================================
# ThreatMatrixOrgan
# ============================================================

class ThreatMatrixOrgan:
    def __init__(self, bus: EventBus, store: BlackKnightStore):
        self.bus = bus
        self.store = store
        self.score = 0.0
        self.last_update = time.time()
        self.history: List[Dict[str, Any]] = []

        self.bus.subscribe("security_violation", self.on_security_violation)
        self.bus.subscribe("lineage_event", self.on_lineage_event)
        self.bus.subscribe("kernel_telemetry", self.on_kernel_telemetry)
        self.bus.subscribe("watchdog_alert", self.on_watchdog_alert)

    def _decay(self):
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        decay_rate = 0.1
        self.score = max(0.0, self.score - decay_rate * dt)

    def _bump(self, amount: float, kind: str, msg: str):
        self._decay()
        self.score += amount
        self.score = min(self.score, 100.0)
        event = {
            "ts": time.time(),
            "kind": kind,
            "msg": msg,
            "score": self.score,
        }
        self.history.append(event)
        if len(self.history) > 100:
            self.history.pop(0)
        self.store.set_json("threat_matrix_state", {
            "score": self.score,
            "last_event": event,
        })
        self.bus.publish("threat_matrix_update", {
            "score": self.score,
            "event": event,
            "history": self.history[-20:],
        })

    def on_security_violation(self, verdict: str):
        self._bump(15.0, "violation", verdict)

    def on_lineage_event(self, msg: str):
        self._bump(10.0, "lineage", msg)

    def on_kernel_telemetry(self, payload: Dict[str, Any]):
        msg = payload.get("msg", "")
        self._bump(1.0, "kernel", msg)

    def on_watchdog_alert(self, msg: str):
        self._bump(8.0, "watchdog", msg)


# ============================================================
# SwarmSyncOrgan
# ============================================================

class SwarmSyncOrgan(threading.Thread):
    def __init__(self, bus: EventBus, store: BlackKnightStore, node_id: str = "node-local"):
        super().__init__(daemon=True)
        self.bus = bus
        self.store = store
        self.node_id = node_id
        self._running = True

    def run(self):
        if IS_WINDOWS:
            pythoncom.CoInitialize()
        try:
            while self._running:
                try:
                    policies = self.store.get_json("fate_policies") or {}
                    ml_state = self.store.get_json("ml_model_state") or {}
                    temporal_state = self.store.get_json("temporal_ml_state") or {}
                    threat_state = self.store.get_json("threat_matrix_state") or {}

                    msg = (
                        f"Swarm sync from {self.node_id}: "
                        f"{len(policies)} policies, "
                        f"ML samples={ml_state.get('samples', 0)}, "
                        f"Temporal samples={temporal_state.get('samples', 0)}, "
                        f"Threat score={threat_state.get('score', 0)}"
                    )
                    self.bus.publish("swarm_event", msg)
                except Exception as e:
                    self.bus.publish("swarm_event", f"Swarm sync error: {e}")

                time.sleep(20)
        finally:
            if IS_WINDOWS:
                pythoncom.CoUninitialize()

    def stop(self):
        self._running = False


# ============================================================
# WatchdogOrgan
# ============================================================

class WatchdogOrgan(threading.Thread):
    def __init__(self, threads: List[threading.Thread], bus: EventBus, store: BlackKnightStore, firewall: FirewallGuardian):
        super().__init__(daemon=True)
        self.threads = threads
        self.bus = bus
        self.store = store
        self.firewall = firewall
        self._running = True
        self._baseline_index_mtime = self._get_index_mtime()
        self._last_corrupt_scan = 0.0

    def _get_index_mtime(self) -> float:
        try:
            return self.store.index_path.stat().st_mtime
        except Exception:
            return 0.0

    def _scan_for_corrupt_files(self) -> bool:
        now = time.time()
        if now - self._last_corrupt_scan < 30:
            return False
        self._last_corrupt_scan = now
        try:
            for p in self.store.base_dir.glob("*.corrupt"):
                return True
        except Exception:
            return False
        return False

    def run(self):
        if IS_WINDOWS:
            pythoncom.CoInitialize()
        try:
            while self._running:
                time.sleep(5)
                for t in self.threads:
                    if not t.is_alive():
                        msg = f"Watchdog: organ {t.__class__.__name__} not running."
                        self.bus.publish("watchdog_alert", msg)

                current_mtime = self._get_index_mtime()
                if self._baseline_index_mtime == 0.0:
                    self._baseline_index_mtime = current_mtime
                else:
                    if current_mtime != self._baseline_index_mtime:
                        msg = "Watchdog: possible tampering with encrypted store index."
                        self.bus.publish("watchdog_alert", msg)
                        self._baseline_index_mtime = current_mtime
                        if not self.firewall.is_lockdown():
                            self.firewall.set_lockdown(True)
                            self.store.set_json("lockdown_state", {"enabled": True, "ts": time.time()})
                            self.bus.publish("auto_lockdown", True)

                if self._scan_for_corrupt_files():
                    msg = "Watchdog: detected corrupted store files (no auto-reset, use cockpit button)."
                    self.bus.publish("watchdog_alert", msg)
        finally:
            if IS_WINDOWS:
                pythoncom.CoUninitialize()

    def stop(self):
        self._running = False


# ============================================================
# Tkinter NetCockpit (1:1 layout clone)
# ============================================================

class NetCockpitTk:
    def __init__(self, root: tk.Tk, bus: EventBus, firewall: FirewallGuardian, store: BlackKnightStore,
                 posture: SecurityPostureMonitor, autofill: FateAutoFillOrgan):
        self.root = root
        self.bus = bus
        self.firewall = firewall
        self.store = store
        self.posture = posture
        self.autofill = autofill

        self.last_fields: List[Dict[str, Any]] = []
        self.latest_profiles: Dict[int, Dict[str, Any]] = {}
        self.current_threat_score: float = 0.0
        self._lockdown_programmatic = False

        self.gui_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        self._wire_bus_to_queue()

        self._build_ui()

        self._restore_lockdown_state()

        self._poll_queue()
        self._refresh_violation_view_periodic()

    # ---------- Bus → queue bridge ----------

    def _wire_bus_to_queue(self):
        self.bus.subscribe("connection_snapshot", lambda payload: self.gui_queue.put(("snapshot", payload)))
        self.bus.subscribe("security_violation", lambda payload: self.gui_queue.put(("violation", payload)))
        self.bus.subscribe("kernel_telemetry", lambda payload: self.gui_queue.put(("kernel", payload)))
        self.bus.subscribe("watchdog_alert", lambda payload: self.gui_queue.put(("watchdog", payload)))
        self.bus.subscribe("process_profiles", lambda payload: self.gui_queue.put(("profiles", payload)))
        self.bus.subscribe("threat_matrix_update", lambda payload: self.gui_queue.put(("threat", payload)))
        self.bus.subscribe("swarm_event", lambda payload: self.gui_queue.put(("swarm", payload)))
        self.bus.subscribe("auto_lockdown", lambda payload: self.gui_queue.put(("auto_lockdown", payload)))
        self.bus.subscribe("fate_fields_detected", lambda payload: self.gui_queue.put(("fate_fields", payload)))
        self.bus.subscribe("fate_event", lambda payload: self.gui_queue.put(("fate_event", payload)))

    # ---------- UI construction ----------

    def _build_ui(self):
        self.root.title(f"Black Knight Organism ({platform.system()})")
        self.root.geometry("1600x900")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top layout
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_status = ttk.Label(
            top_frame,
            text=f"Black Knight: ACTIVE (device-bound, encrypted, heuristic+ML+temporal+lineage+ThreatMatrix+Swarm) [{platform.system()}]"
        )
        self.lbl_status.pack(side=tk.LEFT, padx=5, pady=5)

        self.lockdown_var = tk.BooleanVar(value=False)
        self.chk_lockdown = ttk.Checkbutton(
            top_frame,
            text="Lockdown",
            variable=self.lockdown_var,
            command=self.on_lockdown_toggled
        )
        self.chk_lockdown.pack(side=tk.RIGHT, padx=5, pady=5)

        # Threat layout
        threat_frame = ttk.Frame(main_frame)
        threat_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_threat_score = ttk.Label(threat_frame, text="Threat Score: 0.0 (GREEN)")
        self.lbl_threat_score.pack(side=tk.LEFT, padx=5, pady=5)

        # Splitter
        splitter = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        splitter.pack(fill=tk.BOTH, expand=True)

        # Left widget
        left_frame = ttk.Frame(splitter)
        splitter.add(left_frame, stretch="always")

        lbl_conn = ttk.Label(left_frame, text="Current Connections")
        lbl_conn.pack(side=tk.TOP, anchor="w", padx=5, pady=(5, 0))

        self.conn_list = tk.Listbox(left_frame)
        self.conn_list.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        lbl_proc = ttk.Label(left_frame, text="Process Fingerprints")
        lbl_proc.pack(side=tk.TOP, anchor="w", padx=5, pady=(5, 0))

        proc_split = tk.PanedWindow(left_frame, orient=tk.VERTICAL)
        proc_split.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        proc_top_frame = ttk.Frame(proc_split)
        proc_split.add(proc_top_frame, stretch="always")

        self.proc_list = tk.Listbox(proc_top_frame)
        self.proc_list.pack(fill=tk.BOTH, expand=True)

        proc_bottom_frame = ttk.Frame(proc_split)
        proc_split.add(proc_bottom_frame, stretch="always")

        self.proc_details = tk.Text(proc_bottom_frame, height=8)
        self.proc_details.pack(fill=tk.BOTH, expand=True)

        self.proc_list.bind("<<ListboxSelect>>", self.on_proc_selected)

        # Right widget
        right_frame = ttk.Frame(splitter)
        splitter.add(right_frame, stretch="always")

        lbl_violation = ttk.Label(right_frame, text="Last Security Violation")
        lbl_violation.pack(side=tk.TOP, anchor="w", padx=5, pady=(5, 0))

        self.txt_violation = tk.Text(right_frame, height=6)
        self.txt_violation.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        lbl_log = ttk.Label(right_frame, text="Kernel / Watchdog / Fate / Swarm / Events")
        lbl_log.pack(side=tk.TOP, anchor="w", padx=5, pady=(5, 0))

        self.txt_log = tk.Text(right_frame, height=10)
        self.txt_log.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom layout
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.btn_refresh = ttk.Button(bottom_frame, text="Refresh Now", command=self.request_refresh)
        self.btn_refresh.pack(side=tk.LEFT, padx=5)

        self.btn_focus_self = ttk.Button(bottom_frame, text="Focus Cockpit (uiautomation)", command=self.focus_self_uiautomation)
        self.btn_focus_self.pack(side=tk.LEFT, padx=5)

        self.btn_close_popups = ttk.Button(bottom_frame, text="Close Suspicious Popups", command=self.close_suspicious_popups)
        self.btn_close_popups.pack(side=tk.LEFT, padx=5)

        self.btn_fate_fill = ttk.Button(bottom_frame, text="Fate: Fill Now (manual)", command=self.on_fate_fill_now)
        self.btn_fate_fill.pack(side=tk.LEFT, padx=5)

        self.btn_reset_store = ttk.Button(bottom_frame, text="Reset Encrypted Store", command=self.on_reset_store)
        self.btn_reset_store.pack(side=tk.LEFT, padx=5)

    # ---------- Logging / threat label ----------

    def _log(self, msg: str, level: str = "info"):
        if level == "info":
            color = "white"
            prefix = "ℹ "
        elif level == "warn":
            color = "orange"
            prefix = "⚠ "
        elif level == "error":
            color = "red"
            prefix = "✖ "
        elif level == "success":
            color = "lightgreen"
            prefix = "✔ "
        elif level == "swarm":
            color = "cyan"
            prefix = "⧉ "
        elif level == "threat":
            color = "magenta"
            prefix = "☢ "
        else:
            color = "white"
            prefix = ""
        self.txt_log.insert(tk.END, f"{prefix}{msg}\n")
        self.txt_log.see(tk.END)

    def _update_threat_label(self):
        score = self.current_threat_score
        if score < 20:
            color = "green"
            level = "GREEN"
        elif score < 50:
            color = "orange"
            level = "YELLOW"
        elif score < 80:
            color = "red"
            level = "RED"
        else:
            color = "darkred"
            level = "CRITICAL"
        self.lbl_threat_score.config(text=f"Threat Score: {score:.1f} ({level})", foreground=color)

    # ---------- Event handlers (GUI side) ----------

    def _handle_snapshot(self, conns: List[Dict[str, Any]]):
        self.conn_list.delete(0, tk.END)
        for c in conns:
            self.conn_list.insert(
                tk.END,
                f"{c['laddr']} -> {c['raddr']} [{c['status']}] (pid={c['pid']})"
            )

    def _handle_violation(self, verdict: str):
        self.posture.on_violation(verdict)
        self._refresh_violation_view()
        self._log(f"VIOLATION: {verdict}", level="error")

    def _handle_kernel(self, payload: Dict[str, Any]):
        msg = payload.get("msg", "")
        self._log(f"KERNEL: {msg}", level="info")

    def _handle_watchdog(self, msg: str):
        self._log(f"WATCHDOG: {msg}", level="warn")

    def _handle_profiles(self, snapshot: dict):
        if not snapshot:
            return
        profiles = snapshot.get("profiles", [])
        self.latest_profiles = {p["pid"]: p for p in profiles}

        self.proc_list.delete(0, tk.END)
        for p in profiles:
            label = f"{p['name']} (pid={p['pid']}) conn={p['conn_count']} ips={p['remote_ip_count']} ports={p['remote_port_count']}"
            self.proc_list.insert(tk.END, label)

    def _handle_threat(self, payload: Dict[str, Any]):
        if not payload:
            return
        self.current_threat_score = float(payload.get("score", 0.0))
        event = payload.get("event", {})
        msg = event.get("msg", "")
        kind = event.get("kind", "threat")
        self._update_threat_label()
        self._log(f"ThreatMatrix [{kind}]: {msg} (score={self.current_threat_score:.1f})", level="threat")

    def _handle_swarm(self, msg: str):
        self._log(f"SWARM: {msg}", level="swarm")

    def _handle_auto_lockdown(self, enabled: bool):
        self._lockdown_programmatic = True
        self.lockdown_var.set(bool(enabled))
        self._lockdown_programmatic = False
        self._log(f"Auto-lockdown triggered: {enabled}", level="error")

    def _handle_fate_fields(self, fields: List[Dict[str, Any]]):
        self.last_fields = fields
        self._log(f"Fate: detected {len(fields)} input fields in active window", level="info")

    def _handle_fate_event(self, msg: str):
        self._log(f"FATE: {msg}", level="info")

    # ---------- Tk event wiring ----------

    def on_proc_selected(self, event):
        selection = self.proc_list.curselection()
        if not selection:
            self.proc_details.delete("1.0", tk.END)
            return
        row = selection[0]
        text = self.proc_list.get(row)
        try:
            pid_str = text.split("pid=")[1].split(")")[0]
            pid = int(pid_str)
        except Exception:
            self.proc_details.delete("1.0", tk.END)
            return

        prof = self.latest_profiles.get(pid)
        if not prof:
            self.proc_details.delete("1.0", tk.END)
            return

        details = [
            f"Name: {prof['name']}",
            f"PID: {prof['pid']}",
            f"PPID: {prof['ppid']}",
            f"User: {prof['username']}",
            f"Create Time: {time.ctime(prof['create_time']) if prof['create_time'] else 'N/A'}",
            f"Connection Count: {prof['conn_count']}",
            f"Remote IP Count: {prof['remote_ip_count']}",
            f"Remote Port Count: {prof['remote_port_count']}",
            f"Remote IPs: {', '.join(prof['remote_ips'])}",
            f"Remote Ports: {', '.join(str(p) for p in prof['remote_ports'])}",
        ]
        self.proc_details.delete("1.0", tk.END)
        self.proc_details.insert(tk.END, "\n".join(details))

    def _refresh_violation_view(self):
        if self.posture.last_violation:
            self.txt_violation.delete("1.0", tk.END)
            self.txt_violation.insert(tk.END, self.posture.last_violation)

    def _refresh_violation_view_periodic(self):
        self._refresh_violation_view()
        self.root.after(2000, self._refresh_violation_view_periodic)

    def on_lockdown_toggled(self):
        if self._lockdown_programmatic:
            return
        enabled = self.lockdown_var.get()
        self.firewall.set_lockdown(enabled)
        self.store.set_json("lockdown_state", {"enabled": enabled, "ts": time.time()})
        self._log(f"Lockdown set to {enabled} (note: full effect only on Windows)", level="warn")

    def request_refresh(self):
        self._log("Manual refresh requested (monitor is continuous).", level="info")

    def focus_self_uiautomation(self):
        if not IS_WINDOWS or auto is None:
            self._log("uiautomation focus not available on this OS.", level="warn")
            return
        try:
            window = auto.WindowControl(searchDepth=1, Name=self.root.title())
            if window.Exists(0, 0):
                window.SetActive()
                self._log("uiautomation: Cockpit window focused.", level="success")
            else:
                self._log("uiautomation: Cockpit window not found.", level="warn")
        except Exception as e:
            self._log(f"uiautomation error: {e}", level="error")

    def close_suspicious_popups(self):
        if not IS_WINDOWS or auto is None:
            self._log("uiautomation popup closing not available on this OS.", level="warn")
            return
        suspicious_keywords = ["Congratulations", "Warning", "Error", "Prize", "Alert"]
        try:
            root_ctrl = auto.GetRootControl()
            for w in root_ctrl.GetChildren():
                name = w.Name
                if not name:
                    continue
                if any(k.lower() in name.lower() for k in suspicious_keywords):
                    try:
                        w.Close()
                        self._log(f"uiautomation: Closed suspicious window '{name}'", level="success")
                    except Exception:
                        pass
        except Exception as e:
            self._log(f"uiautomation popup scan error: {e}", level="error")

    def on_fate_fill_now(self):
        if not self.last_fields:
            self._log("FATE: No fields detected to fill.", level="warn")
            return
        self._log("FATE: Manual fill requested.", level="info")
        self.autofill.fill_now(self.last_fields)

    def on_reset_store(self):
        self._log("Manual store reset requested.", level="warn")
        self.store.reset_store()
        self._log("Encrypted store reset complete.", level="success")

    # ---------- Queue polling ----------

    def _poll_queue(self):
        try:
            while True:
                event, payload = self.gui_queue.get_nowait()
                if event == "snapshot":
                    self._handle_snapshot(payload)
                elif event == "violation":
                    self._handle_violation(payload)
                elif event == "kernel":
                    self._handle_kernel(payload)
                elif event == "watchdog":
                    self._handle_watchdog(payload)
                elif event == "profiles":
                    self._handle_profiles(payload)
                elif event == "threat":
                    self._handle_threat(payload)
                elif event == "swarm":
                    self._handle_swarm(payload)
                elif event == "auto_lockdown":
                    self._handle_auto_lockdown(payload)
                elif event == "fate_fields":
                    self._handle_fate_fields(payload)
                elif event == "fate_event":
                    self._handle_fate_event(payload)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _restore_lockdown_state(self):
        state = self.store.get_json("lockdown_state")
        if state and state.get("enabled"):
            self._lockdown_programmatic = True
            self.lockdown_var.set(True)
            self._lockdown_programmatic = False
            self.firewall.set_lockdown(True)
        else:
            self._lockdown_programmatic = True
            self.lockdown_var.set(False)
            self._lockdown_programmatic = False
            self.firewall.set_lockdown(False)


# ============================================================
# Main
# ============================================================

def main():
    rot = TPMRootOfTrustOrgan(tpm_key_handle=None)
    device_secret = rot.get_device_secret()

    crypto = BlackKnightCrypto(device_secret)
    store = BlackKnightStore(crypto)

    bus = EventBus()

    policy = ConnectionPolicy()
    firewall = FirewallGuardian(bus)
    posture = SecurityPostureMonitor(bus, store)
    monitor = ConnectionMonitor(policy, bus, store, poll_interval=3.0)
    kernel = KernelTelemetryOrgan(bus)
    process_profiler = ProcessProfileOrgan(bus, interval=10.0)
    temporal_brain = TemporalAnomalyOrgan(bus, store)
    temporal_ml = TemporalMLOrgan(bus, store)
    ml_organ = MLAnomalyOrgan(bus, store)
    lineage = ProcessLineageOrgan(bus, interval=15.0)
    fate_policy = FatePolicyOrgan(store)
    autofill = FateAutoFillOrgan(store, bus, fate_policy)
    threat_matrix = ThreatMatrixOrgan(bus, store)
    swarm = SwarmSyncOrgan(bus, store, node_id="node-local")

    threads: List[threading.Thread] = [
        monitor,
        kernel,
        process_profiler,
        lineage,
        swarm,
        autofill,
    ]

    watchdog = WatchdogOrgan(
        threads=threads,
        bus=bus,
        store=store,
        firewall=firewall,
    )
    threads.append(watchdog)

    root = tk.Tk()
    cockpit = NetCockpitTk(root, bus, firewall, store, posture, autofill)

    # Start organs
    monitor.start()
    kernel.start()
    process_profiler.start()
    lineage.start()
    watchdog.start()
    autofill.start()
    swarm.start()

    try:
        root.mainloop()
    finally:
        # Stop threads
        monitor.stop()
        kernel.stop()
        process_profiler.stop()
        lineage.stop()
        watchdog.stop()
        autofill.stop()
        swarm.stop()

        # Join
        for t in threads:
            t.join(timeout=2.0)


if __name__ == "__main__":
    main()

