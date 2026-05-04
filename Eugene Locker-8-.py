#!/usr/bin/env python3
# borg_glyph_vault_unified_v7.py
#
# Concept demo ONLY:
# - Glyph codec
# - RSA + AES-GCM crypto
# - TLS-like handshake
# - Local vault
# - Borg Collective: 3 Queens + workers, all required to lock/unlock
# - Machine-bound via per-queen local secrets + OS/host/user
# - System profile fingerprint (MAC, OS, CPU, host)
# - User factor key (second password/PIN) in collective derivation
# - Biometric factor (simulated + API stubs)
# - Collective fingerprint + drift report
# - Hardware binding backend:
#       * TPM (tpm2_pytss) if available
#       * Secure Enclave / Keychain stubs
# - Telemetry:
#       * psutil-based system telemetry (CPU, mem, procs)
#       * ETW-like placeholder (no real kernel hooks)
# - Swarm:
#       * Local file export/import
#       * Local UDP broadcast stub for discovery
# - Anomaly engine:
#       * CPU scoring
#       * GPU/NPU scoring via PyTorch/CuPy if available
# - Autonomous defense:
#       * Policy engine
#       * Real enforcement inside the organism:
#           - lock/unlock allowed/blocked
#           - cooldowns
#           - escalation flags
# - Threat timeline:
#       * Text view
#       * Graph model (nodes/edges) in memory
# - UI:
#       * Tkinter tactical cockpit (active)
#       * PySide6 cockpit stub (future)
#
# NOT FOR REAL SECURITY USE.

import os
import json
import platform
import hashlib
import importlib
import uuid
import time
import threading
import random
import socket
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Optional libs
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from tpm2_pytss import ESAPI, TPM2B_PUBLIC, TPM2B_SENSITIVE_CREATE  # type: ignore
    TPM_AVAILABLE = True
except Exception:
    TPM_AVAILABLE = False

try:
    import PySide6  # noqa: F401
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

# -----------------------------
# Autoloader for cryptography
# -----------------------------

def autoload_crypto():
    try:
        importlib.import_module("cryptography")
    except ImportError as e:
        raise SystemExit(
            "This demo requires the 'cryptography' package.\n"
            "Install with: pip install cryptography"
        ) from e

autoload_crypto()

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# -----------------------------
# System profile fingerprint
# -----------------------------

def get_system_profile():
    mac_int = uuid.getnode()
    mac = ":".join(f"{(mac_int >> ele) & 0xff:02x}" for ele in range(40, -8, -8))
    profile = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "host": platform.node(),
        "mac": mac,
    }
    return profile


def profile_fingerprint(profile: dict) -> str:
    data = json.dumps(profile, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def diff_profiles(old: dict, new: dict) -> dict:
    diff = {}
    keys = set(old.keys()) | set(new.keys())
    for k in keys:
        if old.get(k) != new.get(k):
            diff[k] = {"old": old.get(k), "new": new.get(k)}
    return diff


# -----------------------------
# Glyph codec
# -----------------------------

class GlyphCodec:
    def __init__(self):
        base_glyphs = (
            "⟡⚚✶✺✹✸✷✦✧★☆✪✫✬✭✮✯✰❂❉❊❋✢✣✤✥❈❇❆❅"
            "☀☼☾☽☁☂☃☄★☆☉☊☋☌☍☎☏☑☒☘☙☚☛☜☝☞☟"
            "♠♣♥♦♤♧♡♢♩♪♫♬♭♮♯"
            "⚀⚁⚂⚃⚄⚅"
            "☰☱☲☳☴☵☶☷"
            "✁✂✃✄✅✆✇✈✉✌✍✎✏✐"
            "✑✒✓✔✕✖✗✘✙✚✛✜✝✞✟"
            "✠✡☮☯☸☹☺☻"
            "⚑⚐⚔⚕⚖⚗⚘⚙⚚⚛"
            "✡✢✣✤✥✦✧★☆"
        )
        glyphs = (base_glyphs * ((256 // len(base_glyphs)) + 1))[:256]
        self.byte_to_glyph = {i: glyphs[i] for i in range(256)}
        self.glyph_to_byte = {g: b for b, g in self.byte_to_glyph.items()}

    def encode(self, data: bytes) -> str:
        return "".join(self.byte_to_glyph[b] for b in data)

    def decode(self, s: str) -> bytes:
        return bytes(self.glyph_to_byte[g] for g in s)


# -----------------------------
# Crypto primitives
# -----------------------------

def generate_rsa_keypair(bits=2048):
    priv = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    pub = priv.public_key()
    return priv, pub


def rsa_serialize_public_key(pub) -> bytes:
    return pub.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def rsa_load_public_key(data: bytes):
    return serialization.load_der_public_key(data)


def rsa_encrypt(pub, plaintext: bytes) -> bytes:
    return pub.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def rsa_decrypt(priv, ciphertext: bytes) -> bytes:
    return priv.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def hkdf_sha256(secret: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    kdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info,
    )
    return kdf.derive(secret)


def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce + ct


def aes_gcm_decrypt(key: bytes, data: bytes, aad: bytes = b"") -> bytes:
    aesgcm = AESGCM(key)
    nonce, ct = data[:12], data[12:]
    return aesgcm.decrypt(nonce, ct, aad)


# -----------------------------
# TLS-like handshake (simplified)
# -----------------------------

class Server:
    def __init__(self):
        self.priv, self.pub = generate_rsa_keypair()
        self.session_key = None
        self.server_random = None

    def get_public_key_bytes(self) -> bytes:
        return rsa_serialize_public_key(self.pub)

    def process_client_hello(self, client_hello: dict) -> dict:
        self.server_random = os.urandom(32).hex()
        server_hello = {
            "server_random": self.server_random,
            "server_pubkey": self.get_public_key_bytes().hex(),
        }
        return server_hello

    def process_client_key_exchange(self, enc_pms_hex: str, client_random: str):
        enc_pms = bytes.fromhex(enc_pms_hex)
        pms = rsa_decrypt(self.priv, enc_pms)
        ms = hkdf_sha256(
            secret=pms,
            salt=bytes.fromhex(client_random + self.server_random),
            info=b"master secret",
            length=32,
        )
        self.session_key = hkdf_sha256(ms, salt=b"", info=b"session key", length=32)


class Client:
    def __init__(self):
        self.client_random = os.urandom(32).hex()
        self.server_random = None
        self.server_pub = None
        self.session_key = None

    def build_client_hello(self) -> dict:
        return {
            "client_random": self.client_random,
            "ciphers": ["RSA_AESGCM"],
        }

    def process_server_hello(self, server_hello: dict):
        self.server_random = server_hello["server_random"]
        self.server_pub = rsa_load_public_key(bytes.fromhex(server_hello["server_pubkey"]))

    def build_client_key_exchange(self) -> dict:
        pms = os.urandom(32)
        enc_pms = rsa_encrypt(self.server_pub, pms)
        ms = hkdf_sha256(
            secret=pms,
            salt=bytes.fromhex(self.client_random + self.server_random),
            info=b"master secret",
            length=32,
        )
        self.session_key = hkdf_sha256(ms, salt=b"", info=b"session key", length=32)
        return {"enc_pms": enc_pms.hex()}


# -----------------------------
# Vault (local password store)
# -----------------------------

class Vault:
    def __init__(self):
        self.entries = []
        self.master_key = None

    def set_master_password(self, password: str):
        digest = hashes.Hash(hashes.SHA256())
        digest.update(password.encode("utf-8"))
        self.master_key = digest.finalize()

    def add_entry(self, site: str, username: str, password: str):
        self.entries.append(
            {"site": site, "username": username, "password": password}
        )

    def encrypt_vault(self) -> bytes:
        if self.master_key is None:
            raise ValueError("Master key not set")
        data = json.dumps(self.entries).encode("utf-8")
        return aes_gcm_encrypt(self.master_key, data)

    def decrypt_vault(self, blob: bytes):
        if self.master_key is None:
            raise ValueError("Master key not set")
        data = aes_gcm_decrypt(self.master_key, blob)
        self.entries = json.loads(data.decode("utf-8"))


# -----------------------------
# Hardware binding backend
#   - TPM if available
#   - Fallback to local AES file
#   - Secure Enclave / Keychain stubs
# -----------------------------

class HardwareBindingBackend:
    """
    Hardware-bound sealing:
    - If TPM available: use TPM to seal/unseal a key (conceptual, simplified).
    - Else: local AES-GCM with label-derived key.
    Secure Enclave / Keychain are left as stubs for future platform-specific code.
    """

    def __init__(self, label: str):
        self.label = label
        home = os.path.expanduser("~")
        self.path = os.path.join(home, f".borg_hw_binding_{label}")
        self.use_tpm = TPM_AVAILABLE

    def seal(self, data: bytes) -> bytes:
        if self.use_tpm:
            # Conceptual TPM usage; real code would be more complex.
            # For safety and portability, we still write a local blob.
            key = hashlib.sha256((self.label + "_tpm").encode("utf-8")).digest()
        else:
            key = hashlib.sha256(self.label.encode("utf-8")).digest()
        blob = aes_gcm_encrypt(key, data)
        with open(self.path, "wb") as f:
            f.write(blob)
        return blob

    def unseal(self) -> bytes:
        if not os.path.exists(self.path):
            raise RuntimeError("Hardware binding blob missing")
        if self.use_tpm:
            key = hashlib.sha256((self.label + "_tpm").encode("utf-8")).digest()
        else:
            key = hashlib.sha256(self.label.encode("utf-8")).digest()
        with open(self.path, "rb") as f:
            blob = f.read()
        return aes_gcm_decrypt(key, blob)


# -----------------------------
# Biometric backend (API stubs)
# -----------------------------

class BiometricBackend:
    """
    Biometric factor:
    - For now: simulated via phrase-derived key.
    - Real integration points:
        * Windows Hello
        * Touch ID / Face ID
        * Linux PAM modules
    """

    def __init__(self):
        self.bio_key = None

    def enroll(self, phrase: str):
        h = hashlib.sha256()
        h.update(phrase.encode("utf-8"))
        self.bio_key = h.digest()

    def get_key(self) -> bytes:
        if self.bio_key is None:
            raise RuntimeError("Biometric not enrolled")
        return self.bio_key


# -----------------------------
# Borg Queens (machine-bound keys + workers + health)
# -----------------------------

class BorgQueen:
    def __init__(self, queen_id: str, workers: int = 4):
        self.queen_id = queen_id
        self.workers = workers
        home = os.path.expanduser("~")
        self.secret_path = os.path.join(home, f".borg_queen_{queen_id}_secret")
        self._status = "UNKNOWN"
        self._ensure_secret()

    def _ensure_secret(self):
        if os.path.exists(self.secret_path):
            self._status = "OK"
        else:
            with open(self.secret_path, "wb") as f:
                f.write(os.urandom(32))
            self._status = "NEW"

    def status(self) -> str:
        if not os.path.exists(self.secret_path):
            return "OFFLINE"
        return self._status

    def _local_secret(self) -> bytes:
        if not os.path.exists(self.secret_path):
            return None
        with open(self.secret_path, "rb") as f:
            return f.read()

    def get_key(self) -> bytes:
        secret = self._local_secret()
        if secret is None:
            raise RuntimeError(f"Queen {self.queen_id} secret missing")
        os_name = platform.system().encode("utf-8")
        host = platform.node().encode("utf-8")
        user = os.path.expanduser("~").encode("utf-8")
        h = hashlib.sha256()
        h.update(os_name + b"|" + host + b"|" + user + b"|" +
                 self.queen_id.encode("utf-8") + b"|" + secret)
        return h.digest()

    def fingerprint(self) -> str:
        return hashlib.sha256(self.get_key()).hexdigest()

    def lock_with_workers(self, data: bytes) -> bytes:
        blob = data
        base_key = self.get_key()
        for i in range(self.workers):
            worker_key = hashlib.sha256(base_key + i.to_bytes(1, "big")).digest()
            blob = aes_gcm_encrypt(worker_key, blob)
        return blob

    def unlock_with_workers(self, data: bytes) -> bytes:
        blob = data
        base_key = self.get_key()
        for i in reversed(range(self.workers)):
            worker_key = hashlib.sha256(base_key + i.to_bytes(1, "big")).digest()
            blob = aes_gcm_decrypt(worker_key, blob)
        return blob


# -----------------------------
# Anomaly engine (GPU/NPU capable)
# -----------------------------

class AnomalyEngine:
    """
    Anomaly scoring:
    - Tracks events
    - Scores bursts
    - Optional GPU/NPU scoring via PyTorch/CuPy
    """

    def __init__(self):
        self.events = []
        self.max_events = 500
        self.gpu_mode = False

    def enable_gpu(self):
        if TORCH_AVAILABLE or CUPY_AVAILABLE:
            self.gpu_mode = True

    def record(self, event_type: str, meta: dict):
        ts = time.time()
        self.events.append({"t": ts, "type": event_type, "meta": meta})
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def _cpu_score(self, window_sec: float) -> float:
        now = time.time()
        recent = [e for e in self.events if now - e["t"] <= window_sec]
        base_score = min(1.0, len(recent) / 50.0)
        return base_score

    def _gpu_score(self, window_sec: float) -> float:
        now = time.time()
        recent = [e for e in self.events if now - e["t"] <= window_sec]
        if not recent:
            return 0.0
        vals = [e["meta"].get("cpu_load", 0.0) for e in recent]
        if TORCH_AVAILABLE:
            t = torch.tensor(vals, dtype=torch.float32)
            score = float(torch.clamp(t.std() / 50.0, 0.0, 1.0))
            return score
        if CUPY_AVAILABLE:
            arr = cp.asarray(vals, dtype=cp.float32)
            score = float(cp.clip(arr.std() / 50.0, 0.0, 1.0).get())
            return score
        return self._cpu_score(window_sec)

    def score_recent(self, window_sec: float = 60.0) -> float:
        if self.gpu_mode:
            return self._gpu_score(window_sec)
        return self._cpu_score(window_sec)

    def rare_event_score(self, event_type: str) -> float:
        total = len(self.events)
        if total == 0:
            return 0.0
        count_type = sum(1 for e in self.events if e["type"] == event_type)
        freq = count_type / total
        return min(1.0, 1.0 - freq)


# -----------------------------
# Telemetry engine (psutil + ETW-like stub)
# -----------------------------

class TelemetryEngine:
    """
    System telemetry:
    - If psutil available: real CPU/mem/proc stats
    - Else: synthetic
    - Feeds AnomalyEngine
    """

    def __init__(self, anomaly_engine: AnomalyEngine, callback_log, callback_timeline):
        self.anomaly = anomaly_engine
        self.callback_log = callback_log
        self.callback_timeline = callback_timeline
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            if psutil:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                procs = len(psutil.pids())
                event_type = "telemetry_psutil"
                meta = {"cpu_load": cpu, "mem": mem, "proc_count": procs}
            else:
                event_type = random.choice(["telemetry_tick", "proc_spike", "io_burst"])
                meta = {
                    "cpu_load": round(10 + 70 * (time.time() % 1), 2),
                    "proc_count": 100 + int(time.time()) % 50,
                }

            self.anomaly.record(event_type, meta)
            score = self.anomaly.score_recent()
            self.callback_timeline(event_type, meta, score)
            if score > 0.5:
                self.callback_log(f"[ANOMALY] High recent activity score={score:.2f}")
            time.sleep(2.0)


# -----------------------------
# Swarm-node + networking (local UDP stub)
# -----------------------------

class SwarmNode:
    """
    Swarm node:
    - Export/import wrapped vault blobs
    - Node signatures
    - Local UDP broadcast stub for discovery (no remote I/O by default)
    """

    def __init__(self, node_id: str, udp_port: int = 49321):
        self.node_id = node_id
        home = os.path.expanduser("~")
        self.swarm_secret_path = os.path.join(home, ".borg_swarm_secret")
        self._ensure_swarm_secret()
        self.known_nodes = set([node_id])
        self.version_counter = 0
        self.udp_port = udp_port
        self.udp_enabled = False  # keep off by default for safety

    def _ensure_swarm_secret(self):
        if not os.path.exists(self.swarm_secret_path):
            with open(self.swarm_secret_path, "wb") as f:
                f.write(os.urandom(32))

    def _swarm_secret(self) -> bytes:
        with open(self.swarm_secret_path, "rb") as f:
            return f.read()

    def sign_meta(self, meta: dict) -> str:
        h = hashlib.sha256()
        h.update(self._swarm_secret())
        h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def export_blob(self, blob: bytes) -> bytes:
        self.version_counter += 1
        meta = {
            "node_id": self.node_id,
            "ts": time.time(),
            "version": self.version_counter,
        }
        sig = self.sign_meta(meta)
        obj = {
            "meta": meta,
            "sig": sig,
            "blob": blob.hex(),
        }
        return json.dumps(obj).encode("utf-8")

    def import_blob(self, data: bytes) -> bytes:
        obj = json.loads(data.decode("utf-8"))
        meta = obj["meta"]
        sig = obj["sig"]
        expected = self.sign_meta(meta)
        if sig != expected:
            raise RuntimeError("Swarm signature mismatch")
        self.known_nodes.add(meta["node_id"])
        return bytes.fromhex(obj["blob"])

    def consensus_status(self) -> str:
        if len(self.known_nodes) > 1:
            return f"QUORUM ({len(self.known_nodes)} nodes)"
        return "SINGLE NODE"

    def broadcast_stub(self, payload: bytes):
        if not self.udp_enabled:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.sendto(payload, ("255.255.255.255", self.udp_port))
        finally:
            sock.close()


# -----------------------------
# Threat timeline (graph + text)
# -----------------------------

class ThreatTimeline:
    """
    Threat timeline:
    - Linear event log
    - Graph model: nodes (events) + edges (causal links)
    """

    def __init__(self):
        self.events = []
        self.graph_nodes = {}
        self.graph_edges = []

    def add(self, kind: str, meta: dict, parent_id: str = None):
        eid = f"{int(time.time()*1000)}_{len(self.events)}"
        event = {
            "id": eid,
            "t": time.strftime("%H:%M:%S"),
            "kind": kind,
            "meta": meta,
        }
        self.events.append(event)
        self.graph_nodes[eid] = event
        if parent_id and parent_id in self.graph_nodes:
            self.graph_edges.append((parent_id, eid))
        if len(self.events) > 200:
            old = self.events.pop(0)
            self.graph_nodes.pop(old["id"], None)
            self.graph_edges = [
                (a, b) for (a, b) in self.graph_edges
                if a != old["id"] and b != old["id"]
            ]
        return eid

    def render_text(self) -> str:
        lines = []
        for e in self.events[-50:]:
            lines.append(f"{e['t']} [{e['kind']}] {e['meta']}")
        return "\n".join(lines)


# -----------------------------
# Policy engine for autonomous defense
# -----------------------------

class PolicyEngine:
    """
    Rule-based policy engine:
    - Evaluates anomaly score, unlock failures, drift, etc.
    - Produces actions: LOG, WARN, ESCALATE, LOCK_ORG
    """

    def __init__(self):
        self.rules = []
        self._build_default_rules()

    def _build_default_rules(self):
        self.rules.append({
            "name": "high_anomaly_escalation",
            "condition": lambda ctx: ctx.get("anomaly_score", 0) > 0.8,
            "action": "ESCALATE",
            "message": "High anomaly score detected. Recommend lock-down.",
        })
        self.rules.append({
            "name": "unlock_failure_escalation",
            "condition": lambda ctx: ctx.get("unlock_failures", 0) >= 3,
            "action": "LOCK_ORG",
            "message": "Multiple unlock failures. Locking organism (internal).",
        })

    def evaluate(self, context: dict):
        decisions = []
        for rule in self.rules:
            try:
                if rule["condition"](context):
                    decisions.append({
                        "rule": rule["name"],
                        "action": rule["action"],
                        "message": rule["message"],
                    })
            except Exception:
                continue
        return decisions


# -----------------------------
# Autonomous defense (with enforcement inside organism)
# -----------------------------

class AutonomousDefense:
    """
    Policy-driven autonomous defense:
    - Evaluates context
    - Logs decisions
    - Marks timeline
    - Enforces internal lock state:
        * can block lock/unlock operations
        * can require cooldown
    """

    def __init__(self, anomaly_engine: AnomalyEngine, policy_engine: PolicyEngine,
                 log_callback, timeline_callback, context_provider):
        self.anomaly = anomaly_engine
        self.policy = policy_engine
        self.log = log_callback
        self.timeline = timeline_callback
        self.context_provider = context_provider
        self.running = False
        self.thread = None
        self.org_locked = False
        self.cooldown_until = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            ctx = self.context_provider()
            ctx["anomaly_score"] = self.anomaly.score_recent()
            decisions = self.policy.evaluate(ctx)
            for d in decisions:
                self.log(f"[DEFENSE] Policy '{d['rule']}' => {d['action']}: {d['message']}")
                self.timeline("defense_decision", d)
                if d["action"] == "LOCK_ORG":
                    self.org_locked = True
                    self.cooldown_until = time.time() + 60  # 60s cooldown
                elif d["action"] == "ESCALATE":
                    # escalation is advisory; no extra enforcement here
                    pass
            time.sleep(5.0)

    def can_operate(self) -> bool:
        if self.org_locked and time.time() < self.cooldown_until:
            return False
        if self.org_locked and time.time() >= self.cooldown_until:
            self.org_locked = False
        return True


# -----------------------------
# Borg Collective wrapper
# -----------------------------

class BorgCollectiveWrapper:
    def __init__(self, queens, binding_backend: HardwareBindingBackend):
        if len(queens) != 3:
            raise ValueError("BorgCollectiveWrapper requires exactly 3 Queens")
        self.queens = queens
        self.binding_backend = binding_backend

    def _collective_fingerprint(self, user_factor_key: bytes, bio_key: bytes) -> str:
        h = hashlib.sha256()
        for q in self.queens:
            h.update(q.get_key())
        h.update(user_factor_key)
        h.update(bio_key)
        return h.hexdigest()

    def wrap(self, plaintext: bytes, meta: dict,
             user_factor_key: bytes, bio_key: bytes) -> bytes:
        if not user_factor_key:
            raise RuntimeError("User factor key is required for Borg wrap")
        if not bio_key:
            raise RuntimeError("Biometric key is required for Borg wrap")

        master_key = os.urandom(32)
        aes = AESGCM(master_key)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext, b"")

        wrapped_master = master_key
        for q in self.queens:
            wrapped_master = q.lock_with_workers(wrapped_master)

        sealed = self.binding_backend.seal(wrapped_master)

        profile = get_system_profile()
        obj = {
            "version": 4,
            "algo": "AES-256-GCM",
            "sealed_master": sealed.hex(),
            "ciphertext": (nonce + ct).hex(),
            "meta": {
                **meta,
                "collective_fp": self._collective_fingerprint(user_factor_key, bio_key),
                "queens": [q.queen_id for q in self.queens],
                "profile": profile,
                "profile_fp": profile_fingerprint(profile),
            },
        }
        return json.dumps(obj).encode("utf-8")

    def unwrap(self, blob: bytes, user_factor_key: bytes, bio_key: bytes) -> bytes:
        if not user_factor_key:
            raise RuntimeError("User factor key is required for Borg unwrap")
        if not bio_key:
            raise RuntimeError("Biometric key is required for Borg unwrap")

        obj = json.loads(blob.decode("utf-8"))
        meta = obj.get("meta", {})
        expected_fp = meta.get("collective_fp")
        current_fp = self._collective_fingerprint(user_factor_key, bio_key)
        if expected_fp is not None and expected_fp != current_fp:
            raise RuntimeError("Collective fingerprint mismatch: Queens/user factor/biometric changed")

        stored_profile = meta.get("profile", {})
        stored_profile_fp = meta.get("profile_fp")
        current_profile = get_system_profile()
        current_profile_fp = profile_fingerprint(current_profile)

        if stored_profile_fp and stored_profile_fp != current_profile_fp:
            drift = diff_profiles(stored_profile, current_profile)
            raise RuntimeError(
                "System profile drift detected: " + json.dumps(drift, indent=2)
            )

        sealed_master = bytes.fromhex(obj["sealed_master"])
        c = bytes.fromhex(obj["ciphertext"])
        nonce, ct = c[:12], c[12:]

        wrapped_master = self.binding_backend.unseal()

        for q in reversed(self.queens):
            wrapped_master = q.unlock_with_workers(wrapped_master)

        master_key = wrapped_master
        aes = AESGCM(master_key)
        return aes.decrypt(nonce, ct, b"")


# -----------------------------
# PySide6 cockpit stub
# -----------------------------

class PySideCockpitStub:
    def __init__(self):
        self.available = PYSIDE_AVAILABLE

    def launch(self):
        # Placeholder: real PySide6/QML UI would go here.
        pass


# -----------------------------
# GUI overview (Tkinter Tactical Cockpit)
# -----------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Borg Glyph Vault – Tactical Cockpit (Concept Demo v7)")

        self.codec = GlyphCodec()
        self.server = None
        self.client = None
        self.vault = Vault()
        self.encrypted_vault_blob = None
        self.wrapped_vault_blob = None

        self.user_factor_key = None
        self.biometric_backend = BiometricBackend()

        self.queen1 = BorgQueen("alpha", workers=4)
        self.queen2 = BorgQueen("beta", workers=4)
        self.queen3 = BorgQueen("gamma", workers=4)

        self.binding_backend = HardwareBindingBackend("borg_binding")
        self.collective = BorgCollectiveWrapper(
            [self.queen1, self.queen2, self.queen3],
            self.binding_backend,
        )

        self.anomaly_engine = AnomalyEngine()
        self.anomaly_engine.enable_gpu()
        self.timeline = ThreatTimeline()
        self.telemetry_engine = TelemetryEngine(self.anomaly_engine, self.log_msg, self.timeline_event)
        self.swarm_node = SwarmNode(node_id=platform.node())

        self.unlock_failures = 0
        self.policy_engine = PolicyEngine()
        self.autonomous_defense = AutonomousDefense(
            self.anomaly_engine,
            self.policy_engine,
            self.log_msg,
            self.timeline_event_simple,
            self._defense_context,
        )

        self._build_ui()
        self.log_msg("[INFO] Concept demo. Do NOT use for real secrets.")
        self.update_queen_status()
        self.telemetry_engine.start()
        self.autonomous_defense.start()

    def _build_ui(self):
        frm_top = tk.Frame(self.root)
        frm_top.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(frm_top, text="Master password:").grid(row=0, column=0, sticky="w")
        self.entry_master = tk.Entry(frm_top, show="*")
        self.entry_master.grid(row=0, column=1, sticky="ew", padx=5)
        tk.Button(frm_top, text="Set Master", command=self.set_master).grid(row=0, column=2, padx=5)

        tk.Label(frm_top, text="User factor PIN:").grid(row=1, column=0, sticky="w")
        self.entry_pin = tk.Entry(frm_top, show="*")
        self.entry_pin.grid(row=1, column=1, sticky="ew", padx=5)
        tk.Button(frm_top, text="Set PIN", command=self.set_pin).grid(row=1, column=2, padx=5)

        tk.Label(frm_top, text="Biometric phrase:").grid(row=2, column=0, sticky="w")
        self.entry_bio = tk.Entry(frm_top, show="*")
        self.entry_bio.grid(row=2, column=1, sticky="ew", padx=5)
        tk.Button(frm_top, text="Enroll Bio", command=self.set_bio).grid(row=2, column=2, padx=5)

        frm_top.columnconfigure(1, weight=1)

        frm_queens = tk.Frame(self.root)
        frm_queens.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_q1 = tk.Label(frm_queens, text="Queen alpha: ?", width=20)
        self.lbl_q1.pack(side=tk.LEFT, padx=5)
        self.lbl_q2 = tk.Label(frm_queens, text="Queen beta: ?", width=20)
        self.lbl_q2.pack(side=tk.LEFT, padx=5)
        self.lbl_q3 = tk.Label(frm_queens, text="Queen gamma: ?", width=20)
        self.lbl_q3.pack(side=tk.LEFT, padx=5)

        frm_mid = tk.Frame(self.root)
        frm_mid.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(frm_mid, text="Init Server", command=self.init_server).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid, text="Run Handshake", command=self.run_handshake).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid, text="Add Demo Entry", command=self.add_demo_entry).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid, text="Encrypt Vault", command=self.encrypt_vault).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid, text="Decrypt Vault", command=self.decrypt_vault).pack(side=tk.LEFT, padx=5)

        frm_mid2 = tk.Frame(self.root)
        frm_mid2.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(frm_mid2, text="Borg Lock Vault", command=self.borg_lock_vault).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid2, text="Borg Unlock Vault", command=self.borg_unlock_vault).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid2, text="Show Glyph Ciphertext", command=self.show_glyph_ciphertext).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid2, text="Swarm Export", command=self.swarm_export).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid2, text="Swarm Import", command=self.swarm_import).pack(side=tk.LEFT, padx=5)

        frm_status = tk.Frame(self.root)
        frm_status.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_anomaly = tk.Label(frm_status, text="Anomaly score: 0.00", width=25)
        self.lbl_anomaly.pack(side=tk.LEFT, padx=5)
        tk.Button(frm_status, text="Refresh Anomaly", command=self.refresh_anomaly).pack(side=tk.LEFT, padx=5)

        self.lbl_swarm = tk.Label(frm_status, text="Swarm: SINGLE NODE", width=25)
        self.lbl_swarm.pack(side=tk.LEFT, padx=5)

        frm_bottom = tk.PanedWindow(self.root, orient=tk.VERTICAL)
        frm_bottom.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log = scrolledtext.ScrolledText(frm_bottom, height=16)
        frm_bottom.add(self.log)

        self.timeline_view = scrolledtext.ScrolledText(frm_bottom, height=10)
        frm_bottom.add(self.timeline_view)

    def log_msg(self, msg: str):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def timeline_event(self, kind: str, meta: dict, score: float):
        self.timeline.add(kind, {"meta": meta, "score": round(score, 2)})
        self.render_timeline()

    def timeline_event_simple(self, kind: str, meta: dict):
        self.timeline.add(kind, meta)
        self.render_timeline()

    def render_timeline(self):
        self.timeline_view.delete("1.0", tk.END)
        self.timeline_view.insert(tk.END, self.timeline.render_text())
        self.timeline_view.see(tk.END)

    def update_queen_status(self):
        def color_for(status):
            if status in ("OK", "NEW"):
                return "green"
            if status == "OFFLINE":
                return "red"
            return "yellow"

        s1 = self.queen1.status()
        s2 = self.queen2.status()
        s3 = self.queen3.status()

        self.lbl_q1.config(text=f"Queen alpha: {s1}", bg=color_for(s1))
        self.lbl_q2.config(text=f"Queen beta: {s2}", bg=color_for(s2))
        self.lbl_q3.config(text=f"Queen gamma: {s3}", bg=color_for(s3))

        self.log_msg(f"[QUEEN] alpha status={s1}, fp={self.queen1.fingerprint()[:12]}...")
        self.log_msg(f"[QUEEN] beta  status={s2}, fp={self.queen2.fingerprint()[:12]}...")
        self.log_msg(f"[QUEEN] gamma status={s3}, fp={self.queen3.fingerprint()[:12]}...")

    def set_master(self):
        pw = self.entry_master.get()
        if not pw:
            messagebox.showwarning("Warning", "Enter a master password")
            return
        self.vault.set_master_password(pw)
        self.log_msg("[VAULT] Master password set (demo hash).")

    def set_pin(self):
        pin = self.entry_pin.get()
        if not pin:
            messagebox.showwarning("Warning", "Enter a PIN")
            return
        h = hashlib.sha256()
        h.update(pin.encode("utf-8"))
        self.user_factor_key = h.digest()
        self.log_msg("[USER] User factor PIN set (derived key).")

    def set_bio(self):
        phrase = self.entry_bio.get()
        if not phrase:
            messagebox.showwarning("Warning", "Enter a biometric phrase")
            return
        self.biometric_backend.enroll(phrase)
        self.log_msg("[BIO] Biometric phrase enrolled (simulated).")

    def init_server(self):
        self.server = Server()
        self.client = Client()
        self.log_msg("[TLS] Server and Client initialized with fresh keys/randoms.")

    def run_handshake(self):
        if self.server is None or self.client is None:
            messagebox.showwarning("Warning", "Init server/client first")
            return

        ch = self.client.build_client_hello()
        self.log_msg(f"[TLS] ClientHello: {ch}")

        sh = self.server.process_client_hello(ch)
        self.log_msg(f"[TLS] ServerHello: {sh}")

        self.client.process_server_hello(sh)
        ckx = self.client.build_client_key_exchange()
        self.log_msg(f"[TLS] ClientKeyExchange: {ckx}")

        self.server.process_client_key_exchange(
            enc_pms_hex=ckx["enc_pms"],
            client_random=self.client.client_random,
        )

        self.log_msg("[TLS] Session keys derived (client & server).")

    def add_demo_entry(self):
        self.vault.add_entry("example.com", "user@example.com", "SuperSecret123!")
        self.log_msg("[VAULT] Added demo entry: example.com / user@example.com")

    def encrypt_vault(self):
        try:
            blob = self.vault.encrypt_vault()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.encrypted_vault_blob = blob
        self.log_msg(f"[VAULT] Encrypted vault ({len(blob)} bytes).")

    def decrypt_vault(self):
        if self.encrypted_vault_blob is None:
            messagebox.showwarning("Warning", "No encrypted vault blob yet")
            return
        try:
            self.vault.decrypt_vault(self.encrypted_vault_blob)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.log_msg(f"[VAULT] Decrypted vault. Entries: {len(self.vault.entries)}")

    def borg_lock_vault(self):
        if not self.autonomous_defense.can_operate():
            messagebox.showerror("Defense Lock", "Organism is in lock-down cooldown.")
            self.log_msg("[DEFENSE] Lock operation blocked by autonomous defense.")
            return

        if self.encrypted_vault_blob is None:
            messagebox.showwarning("Warning", "Encrypt vault first")
            return
        if self.user_factor_key is None:
            messagebox.showwarning("Warning", "Set user factor PIN first")
            return
        try:
            bio_key = self.biometric_backend.get_key()
        except Exception as e:
            messagebox.showwarning("Warning", f"Biometric not ready: {e}")
            return

        statuses = [self.queen1.status(), self.queen2.status(), self.queen3.status()]
        if any(s == "OFFLINE" for s in statuses):
            messagebox.showerror("Error", "All 3 Queens must be ONLINE to lock.")
            return

        meta = {
            "note": "borg_collective_lock_demo",
        }
        try:
            self.wrapped_vault_blob = self.collective.wrap(
                self.encrypted_vault_blob, meta, self.user_factor_key, bio_key
            )
        except Exception as e:
            messagebox.showerror("Error", f"Borg lock failed: {e}")
            return

        self.log_msg(f"[BORG] Vault locked by 3 Queens ({len(self.wrapped_vault_blob)} bytes).")
        self.anomaly_engine.record("borg_lock", {"size": len(self.wrapped_vault_blob)})
        self.timeline.add("borg_lock", {"size": len(self.wrapped_vault_blob)})
        self.render_timeline()

    def borg_unlock_vault(self):
        if not self.autonomous_defense.can_operate():
            messagebox.showerror("Defense Lock", "Organism is in lock-down cooldown.")
            self.log_msg("[DEFENSE] Unlock operation blocked by autonomous defense.")
            return

        if self.wrapped_vault_blob is None:
            messagebox.showwarning("Warning", "No Borg-locked vault yet")
            return
        if self.user_factor_key is None:
            messagebox.showwarning("Warning", "Set user factor PIN first")
            return
        try:
            bio_key = self.biometric_backend.get_key()
        except Exception as e:
            messagebox.showwarning("Warning", f"Biometric not ready: {e}")
            return

        statuses = [self.queen1.status(), self.queen2.status(), self.queen3.status()]
        if any(s == "OFFLINE" for s in statuses):
            messagebox.showerror("Error", "All 3 Queens must be ONLINE to unlock.")
            return

        try:
            self.encrypted_vault_blob = self.collective.unwrap(
                self.wrapped_vault_blob, self.user_factor_key, bio_key
            )
        except Exception as e:
            msg = str(e)
            self.unlock_failures += 1
            self.log_msg(f"[DRIFT] {msg}")
            messagebox.showerror("Borg unlock failed", msg)
            self.anomaly_engine.record("borg_unlock_failed", {"reason": msg})
            self.timeline.add("borg_unlock_failed", {"reason": msg})
            self.render_timeline()
            return

        self.log_msg("[BORG] Vault unlocked by 3 Queens. You can now decrypt vault with master password.")
        self.anomaly_engine.record("borg_unlock", {"ok": True})
        self.timeline.add("borg_unlock", {"ok": True})
        self.render_timeline()
        self.unlock_failures = 0

    def show_glyph_ciphertext(self):
        if self.wrapped_vault_blob is None and self.encrypted_vault_blob is None:
            messagebox.showwarning("Warning", "No ciphertext yet")
            return
        blob = self.wrapped_vault_blob or self.encrypted_vault_blob
        glyphs = self.codec.encode(blob)
        self.log_msg("[GLYPH] Ciphertext as glyphs:")
        self.log_msg(glyphs[:800] + ("..." if len(glyphs) > 800 else ""))

    def swarm_export(self):
        if self.wrapped_vault_blob is None:
            messagebox.showwarning("Warning", "No Borg-locked vault to export")
            return
        data = self.swarm_node.export_blob(self.wrapped_vault_blob)
        path = os.path.join(os.path.expanduser("~"), "borg_swarm_export.json")
        with open(path, "wb") as f:
            f.write(data)
        self.log_msg(f"[SWARM] Exported Borg blob to {path}")
        self.lbl_swarm.config(text=f"Swarm: {self.swarm_node.consensus_status()}")

    def swarm_import(self):
        path = os.path.join(os.path.expanduser("~"), "borg_swarm_export.json")
        if not os.path.exists(path):
            messagebox.showwarning("Warning", f"No swarm export at {path}")
            return
        with open(path, "rb") as f:
            data = f.read()
        try:
            self.wrapped_vault_blob = self.swarm_node.import_blob(data)
        except Exception as e:
            messagebox.showerror("Error", f"Swarm import failed: {e}")
            return
        self.log_msg("[SWARM] Imported Borg blob from swarm export.")
        self.lbl_swarm.config(text=f"Swarm: {self.swarm_node.consensus_status()}")

    def refresh_anomaly(self):
        score = self.anomaly_engine.score_recent()
        self.lbl_anomaly.config(text=f"Anomaly score: {score:.2f}")
        self.log_msg(f"[ANOMALY] Recent score={score:.2f}")

    def _defense_context(self):
        return {
            "unlock_failures": self.unlock_failures,
        }

    def on_close(self):
        self.telemetry_engine.stop()
        self.autonomous_defense.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
