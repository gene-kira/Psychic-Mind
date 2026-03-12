#!/usr/bin/env python3
# ============================================================
#  LIVING SECURITY BACKBONE + COCKPIT GUI (CONTINUOUS, REAL TELEMETRY)
#  - Base64-first pipeline
#  - Final padding repair guard in backbone_decode
#  - Frame boundary validator
#  - Padding integrity monitor
#  - GUI in MAIN THREAD
#  - Continuous system telemetry + optional filesystem events
#  - Game detection + BIG Game Manager (wide + tall + scrollbars)
# ============================================================

import importlib

# ============================================================
#  AUTOLOADER (stdlib + optional external libs)
# ============================================================

REQUIRED_STDLIB = [
    "hashlib",
    "json",
    "base64",
    "hmac",
    "time",
    "uuid",
    "random",
    "threading",
    "tkinter",
    "socket",
    "os",
]

OPTIONAL_LIBS = {
    "cryptography": "cryptography",
    "numpy": "numpy",
    "psutil": "psutil",
    "watchdog": "watchdog.observers",
}

def autoload_stdlib():
    for lib in REQUIRED_STDLIB:
        importlib.import_module(lib)

def try_import_optional():
    loaded = {}
    for name, pkg in OPTIONAL_LIBS.items():
        try:
            loaded[name] = importlib.import_module(pkg)
        except ImportError:
            loaded[name] = None
    return loaded

autoload_stdlib()
opt = try_import_optional()

import hashlib
import json
import base64
import hmac
import time
import uuid
import random
import threading
import tkinter as tk
from tkinter import filedialog
import socket
import os

np = opt.get("numpy", None)
cryptography = opt.get("cryptography", None)
psutil = opt.get("psutil", None)
watchdog_observers = opt.get("watchdog", None)

# ============================================================
#  RING CONFIGURATION & STATE (ADAPTIVE)
# ============================================================

class RingState:
    def __init__(self, ring_id, max_size, base_frame_size, apply_pre, weight):
        self.ring_id = ring_id
        self.max_size = max_size
        self.base_frame_size = base_frame_size
        self.apply_pre = apply_pre
        self.weight = weight
        self.current_frame_size = base_frame_size
        self.bytes_in_flight = 0
        self.frames_in_flight = 0
        self.pressure_history = []

    def update_pressure(self, delta_frames, delta_bytes):
        self.frames_in_flight += delta_frames
        self.bytes_in_flight += delta_bytes
        pressure = max(0, self.bytes_in_flight)
        self.pressure_history.append(pressure)
        if len(self.pressure_history) > 100:
            self.pressure_history.pop(0)
        self._adapt_frame_size()

    def _adapt_frame_size(self):
        if not self.pressure_history:
            return
        if np is not None:
            avg = float(np.mean(self.pressure_history))
        else:
            avg = sum(self.pressure_history) / len(self.pressure_history)
        if avg > 4 * self.base_frame_size:
            self.current_frame_size = max(self.base_frame_size // 2, 64)
        elif avg < 2 * self.base_frame_size:
            self.current_frame_size = min(self.base_frame_size * 2, 8192)

RINGS = {
    "ring1": RingState("ring1", max_size=512,   base_frame_size=128,  apply_pre=True, weight=1.0),
    "ring2": RingState("ring2", max_size=4096,  base_frame_size=512,  apply_pre=True, weight=1.5),
    "ring3": RingState("ring3", max_size=16384, base_frame_size=1024, apply_pre=True, weight=2.0),
    "ring4": RingState("ring4", max_size=None,  base_frame_size=2048, apply_pre=True, weight=3.0),
}

def select_ring_for_payload_size(size: int) -> str:
    for ring_id, state in RINGS.items():
        if state.max_size is None or size <= state.max_size:
            return ring_id
    return "ring4"

# ============================================================
#  TELEMETRY, SUMMARY, ERROR ORGAN, PADDING MONITOR
# ============================================================

class TelemetrySink:
    def __init__(self):
        self.events = []

    def record_event(self, kind: str, data: dict):
        evt = {
            "kind": kind,
            "ts": time.time(),
            "data": data,
        }
        self.events.append(evt)

    def record_flow(self, flow_id: str, stage: str, meta: dict):
        self.record_event("flow", {"flow_id": flow_id, "stage": stage, **meta})

    def record_pre_mapping(self, flow_id: str, frame_index: int, ring_id: str,
                           pattern_id: str, mapping: dict, animation: dict):
        self.record_event("pre_mapping", {
            "flow_id": flow_id,
            "frame_index": frame_index,
            "ring_id": ring_id,
            "pattern_id": pattern_id,
            "mapping": mapping,
            "animation": animation,
        })

    def record_ring_pressure(self, ring_id: str, delta_frames: int, delta_bytes: int,
                             frames_in_flight: int, bytes_in_flight: int):
        self.record_event("ring_pressure", {
            "ring_id": ring_id,
            "delta_frames": delta_frames,
            "delta_bytes": delta_bytes,
            "frames_in_flight": frames_in_flight,
            "bytes_in_flight": bytes_in_flight,
        })

    def record_error(self, flow_id: str, stage: str, error_type: str, message: str):
        self.record_event("error", {
            "flow_id": flow_id,
            "stage": stage,
            "error_type": error_type,
            "message": message,
        })

    def record_padding_issue(self, flow_id: str, frame_index: int, detail: str):
        self.record_event("padding_issue", {
            "flow_id": flow_id,
            "frame_index": frame_index,
            "detail": detail,
        })

    def record_frame_boundary_issue(self, flow_id: str, frame_index: int, detail: str):
        self.record_event("frame_boundary_issue", {
            "flow_id": flow_id,
            "frame_index": frame_index,
            "detail": detail,
        })

    def summary(self):
        rings = {}
        patterns = {}
        errors = 0
        padding_issues = 0
        boundary_issues = 0
        for e in self.events:
            if e["kind"] == "ring_pressure":
                rid = e["data"]["ring_id"]
                rings.setdefault(rid, 0)
                rings[rid] = max(rings[rid], e["data"]["bytes_in_flight"])
            elif e["kind"] == "pre_mapping":
                pid = e["data"]["pattern_id"]
                patterns[pid] = patterns.get(pid, 0) + 1
            elif e["kind"] == "error":
                errors += 1
            elif e["kind"] == "padding_issue":
                padding_issues += 1
            elif e["kind"] == "frame_boundary_issue":
                boundary_issues += 1
        return {
            "ring_max_bytes_in_flight": rings,
            "pre_pattern_counts": patterns,
            "error_count": errors,
            "padding_issues": padding_issues,
            "boundary_issues": boundary_issues,
            "total_events": len(self.events),
        }

TELEMETRY = TelemetrySink()

def error_organ(flow_id: str, stage: str, error_type: str, message: str):
    TELEMETRY.record_error(flow_id, stage, error_type, message)
    raise RuntimeError(f"[{stage}] {error_type}: {message}")

# ============================================================
#  POLICY ENGINE
# ============================================================

class PolicyEngine:
    def classify_flow(self, raw: bytes, device_id: str, session_id: str):
        size = len(raw)
        if size < 256:
            sensitivity = "low"
        elif size < 4096:
            sensitivity = "medium"
        else:
            sensitivity = "high"

        device_trust = "normal"
        if device_id.startswith("trusted-"):
            device_trust = "high"
        elif device_id.startswith("untrusted-"):
            device_trust = "low"

        preferred_ring = None
        if sensitivity == "high":
            preferred_ring = "ring3"
        if device_trust == "low":
            preferred_ring = "ring1"

        return {
            "sensitivity": sensitivity,
            "device_trust": device_trust,
            "preferred_ring": preferred_ring,
        }

POLICY = PolicyEngine()

# ============================================================
#  SHARD ORGAN — 4-WAY SPLIT
# ============================================================

def shard_data_4(data: bytes):
    size = len(data)
    q = size // 4
    r = size % 4
    shards = []
    idx = 0
    for i in range(4):
        extra = 1 if i < r else 0
        shard = data[idx:idx + q + extra]
        shards.append(shard)
        idx += q + extra
    return shards

# ============================================================
#  PRE ORGAN — MULTIPLE PATTERNS + ADAPTIVE
# ============================================================

def pre_pattern_base(shards):
    S1, S2, S3, S4 = shards
    return [S1, S3, S2, S4], {"jump_from": 2, "jump_to": 1}

def pre_pattern_reverse(shards):
    S1, S2, S3, S4 = shards
    return [S4, S3, S2, S1], {"jump_from": 0, "jump_to": 3}

def pre_pattern_swap(shards):
    S1, S2, S3, S4 = shards
    return [S2, S1, S4, S3], {"jump_from": 0, "jump_to": 1}

def pre_pattern_rotate(shards):
    S1, S2, S3, S4 = shards
    return [S2, S3, S4, S1], {"jump_from": 0, "jump_to": 3}

def pre_pattern_random(shards, seed):
    random.seed(seed)
    idxs = [0, 1, 2, 3]
    random.shuffle(idxs)
    mapping = [shards[i] for i in idxs]
    jump_from = 2
    jump_to = idxs.index(2)
    return mapping, {"jump_from": jump_from, "jump_to": jump_to, "perm": idxs}

def pre_encode_patterns(shards, pattern_id, seed=None):
    if pattern_id == "base":
        return pre_pattern_base(shards)
    if pattern_id == "reverse":
        return pre_pattern_reverse(shards)
    if pattern_id == "swap":
        return pre_pattern_swap(shards)
    if pattern_id == "rotate":
        return pre_pattern_rotate(shards)
    if pattern_id == "random":
        return pre_pattern_random(shards, seed)
    return pre_pattern_base(shards)

def pre_decode_patterns(shards, pattern_id, seed=None, perm=None):
    if pattern_id == "base":
        S1, S3, S2, S4 = shards
        return [S1, S2, S3, S4]
    if pattern_id == "reverse":
        S4, S3, S2, S1 = shards
        return [S1, S2, S3, S4]
    if pattern_id == "swap":
        S2, S1, S4, S3 = shards
        return [S1, S2, S3, S4]
    if pattern_id == "rotate":
        S2, S3, S4, S1 = shards
        return [S1, S2, S3, S4]
    if pattern_id == "random" and perm is not None:
        inv = [None] * 4
        for i, p in enumerate(perm):
            inv[p] = shards[i]
        return inv
    S1, S3, S2, S4 = shards
    return [S1, S2, S3, S4]

def adaptive_pre_profile(ring_id: str, frame_index: int, sensitivity: str):
    if sensitivity == "high":
        return "random"
    if ring_id in ("ring1", "ring2"):
        return "base"
    if frame_index % 3 == 0:
        return "swap"
    if frame_index % 5 == 0:
        return "rotate"
    return "reverse"

# ============================================================
#  TRANSFORM ORGANS (REVERSIBLE)
# ============================================================

def organ_reverse(data: bytes):
    return data[::-1]

def organ_unreverse(data: bytes):
    return data[::-1]

def organ_chameleon(data: bytes):
    return base64.b64encode(data)

def organ_unchameleon(data: bytes):
    return base64.b64decode(data)

def organ_glyph(data: bytes):
    return hashlib.sha256(data).digest()

# ============================================================
#  CAPSULE SEALING & ENVIRONMENT BINDING
# ============================================================

def derive_env_key(device_id: str, session_id: str, ring_id: str) -> bytes:
    seed = f"{device_id}:{session_id}:{ring_id}".encode()
    return hashlib.sha256(seed).digest()

def seal_capsule(payload: bytes, glyph: bytes, ring_id: str,
                 device_id: str, session_id: str):
    env = {
        "device_id": device_id,
        "session_id": session_id,
        "ring_id": ring_id,
        "ts": time.time(),
    }
    env_json = json.dumps(env, sort_keys=True).encode()

    key = derive_env_key(device_id, session_id, ring_id)

    if cryptography is not None:
        from cryptography.hazmat.primitives import hashes, hmac as chmac
        h = chmac.HMAC(key, hashes.SHA256())
        h.update(payload + glyph + env_json)
        mac = h.finalize()
    else:
        mac = hmac.new(key, payload + glyph + env_json, hashlib.sha256).digest()

    capsule = {
        "payload": base64.b64encode(payload).decode(),
        "glyph": glyph.hex(),
        "env": env,
        "mac": mac.hex(),
    }
    return capsule

def verify_and_open_capsule(capsule: dict, flow_id: str):
    try:
        payload = base64.b64decode(capsule["payload"])
        glyph = bytes.fromhex(capsule["glyph"])
        env = capsule["env"]
        mac = bytes.fromhex(capsule["mac"])
    except Exception as e:
        error_organ(flow_id, "capsule_parse", "CapsuleFormatError", str(e))

    env_json = json.dumps(env, sort_keys=True).encode()
    key = derive_env_key(env["device_id"], env["session_id"], env["ring_id"])

    if cryptography is not None:
        from cryptography.hazmat.primitives import hashes, hmac as chmac
        h = chmac.HMAC(key, hashes.SHA256())
        h.update(payload + glyph + env_json)
        expected_mac = h.finalize()
        valid_mac = (mac == expected_mac)
    else:
        expected_mac = hmac.new(key, payload + glyph + env_json, hashlib.sha256).digest()
        valid_mac = hmac.compare_digest(mac, expected_mac)

    if not valid_mac:
        error_organ(flow_id, "capsule_verify", "MACMismatch", "Capsule MAC verification failed")

    if organ_glyph(payload) != glyph:
        error_organ(flow_id, "capsule_verify", "GlyphMismatch", "Capsule glyph mismatch")

    return payload, env

# ============================================================
#  FRAME BOUNDARY VALIDATOR + PADDING MONITOR
# ============================================================

def frame_boundary_validator(flow_id: str, frame_index: int, frame: bytes):
    if len(frame) < 0:
        TELEMETRY.record_frame_boundary_issue(flow_id, frame_index, "Negative length")
        error_organ(flow_id, "frame_boundary", "NegativeLength", "Frame length negative")
    if len(frame) > 10_000_000:
        TELEMETRY.record_frame_boundary_issue(flow_id, frame_index, "Frame too large")
        error_organ(flow_id, "frame_boundary", "FrameTooLarge", "Frame length too large")

def padding_integrity_monitor(flow_id: str, frame_index: int, data: bytes, stage: str):
    if len(data) % 4 != 0:
        TELEMETRY.record_padding_issue(
            flow_id, frame_index,
            f"{stage}: length {len(data)} not multiple of 4"
        )

# ============================================================
#  FRAME PIPELINE (PER RING, PER FRAME)
# ============================================================

def encode_frame_with_pre(frame: bytes, flow_id: str, ring_state: RingState,
                          frame_index: int, sensitivity: str):
    ring_id = ring_state.ring_id
    frame_boundary_validator(flow_id, frame_index, frame)

    TELEMETRY.record_flow(flow_id, "frame_ingress", {
        "ring_id": ring_id,
        "frame_index": frame_index,
        "size": len(frame),
    })

    shards = shard_data_4(frame)
    before_order = ["S1", "S2", "S3", "S4"]

    transformed = []
    for s in shards:
        x = organ_reverse(s)
        x = organ_chameleon(x)
        transformed.append(x)

    pattern_id = adaptive_pre_profile(ring_id, frame_index, sensitivity)
    seed = hash((flow_id, ring_id, frame_index)) & 0xFFFFFFFF
    after_shards, anim = pre_encode_patterns(transformed, pattern_id, seed=seed)

    mapping = {
        "before": before_order,
        "after": ["S?"] * 4,
    }
    animation = {
        "jump_from": anim.get("jump_from"),
        "jump_to": anim.get("jump_to"),
        "perm": anim.get("perm"),
    }

    combined = b"".join(after_shards)

    padding_integrity_monitor(flow_id, frame_index, combined, "encode_frame_with_pre")

    ring_state.update_pressure(delta_frames=1, delta_bytes=len(combined))
    TELEMETRY.record_ring_pressure(
        ring_id,
        delta_frames=1,
        delta_bytes=len(combined),
        frames_in_flight=ring_state.frames_in_flight,
        bytes_in_flight=ring_state.bytes_in_flight,
    )

    TELEMETRY.record_pre_mapping(flow_id, frame_index, ring_id, pattern_id, mapping, animation)

    TELEMETRY.record_flow(flow_id, "frame_encoded", {
        "ring_id": ring_id,
        "frame_index": frame_index,
        "size": len(combined),
    })

    return combined, pattern_id, anim.get("perm")

def decode_frame_with_pre(frame: bytes, flow_id: str, ring_state: RingState,
                          frame_index: int, pattern_id: str, perm):
    ring_id = ring_state.ring_id
    frame_boundary_validator(flow_id, frame_index, frame)

    TELEMETRY.record_flow(flow_id, "frame_decode_ingress", {
        "ring_id": ring_id,
        "frame_index": frame_index,
        "size": len(frame),
    })

    padding_integrity_monitor(flow_id, frame_index, frame, "decode_frame_with_pre")

    shards = shard_data_4(frame)
    seed = hash((flow_id, ring_id, frame_index)) & 0xFFFFFFFF
    original_order_shards = pre_decode_patterns(shards, pattern_id, seed=seed, perm=perm)

    restored = []
    for s in original_order_shards:
        try:
            x = organ_unchameleon(s)
        except Exception as e:
            TELEMETRY.record_padding_issue(
                flow_id, frame_index,
                f"Base64 decode error: {str(e)}"
            )
            error_organ(flow_id, "frame_decode", "Base64DecodeError", str(e))
        x = organ_unreverse(x)
        restored.append(x)

    combined = b"".join(restored)

    ring_state.update_pressure(delta_frames=-1, delta_bytes=-len(frame))
    TELEMETRY.record_ring_pressure(
        ring_id,
        delta_frames=-1,
        delta_bytes=-len(frame),
        frames_in_flight=ring_state.frames_in_flight,
        bytes_in_flight=ring_state.bytes_in_flight,
    )

    TELEMETRY.record_flow(flow_id, "frame_decoded", {
        "ring_id": ring_id,
        "frame_index": frame_index,
        "size": len(combined),
    })

    return combined

# ============================================================
#  BACKBONE ENCODE / DECODE (BASE64-FIRST + PADDING GUARD)
# ============================================================

def backbone_encode(raw: bytes, device_id: str, session_id: str):
    flow_id = str(uuid.uuid4())
    policy = POLICY.classify_flow(raw, device_id, session_id)

    encoded_raw = base64.b64encode(raw)
    size = len(encoded_raw)

    ring_id = policy["preferred_ring"] or select_ring_for_payload_size(size)
    ring_state = RINGS[ring_id]

    TELEMETRY.record_flow(flow_id, "ingress", {
        "ring_id": ring_id,
        "size": size,
        "sensitivity": policy["sensitivity"],
        "device_trust": policy["device_trust"],
    })

    frames = []
    patterns = []
    perms = []
    frame_sizes = []
    idx = 0
    frame_index = 0

    frozen_frame_size = ring_state.current_frame_size

    while idx < size:
        frame_size = frozen_frame_size
        chunk = encoded_raw[idx:idx + frame_size]
        encoded_frame, pattern_id, perm = encode_frame_with_pre(
            chunk, flow_id, ring_state, frame_index, policy["sensitivity"]
        )
        frames.append(encoded_frame)
        patterns.append(pattern_id)
        perms.append(perm)
        frame_sizes.append(frame_size)
        idx += frame_size
        frame_index += 1

    combined = b"".join(frames)
    glyph = organ_glyph(combined)

    TELEMETRY.record_flow(flow_id, "post_transform", {
        "ring_id": ring_id,
        "frames": frame_index,
        "combined_size": len(combined),
    })

    capsule = seal_capsule(combined, glyph, ring_id, device_id, session_id)

    TELEMETRY.record_flow(flow_id, "capsule_sealed", {
        "ring_id": ring_id,
        "capsule_size": len(capsule["payload"]),
    })

    capsule["meta"] = {
        "flow_id": flow_id,
        "patterns": patterns,
        "perms": perms,
        "frame_sizes": frame_sizes,
        "frozen_frame_size": frozen_frame_size,
        "policy": policy,
    }

    return capsule

def backbone_decode(capsule: dict):
    meta = capsule.get("meta", {})
    flow_id = meta.get("flow_id", str(uuid.uuid4()))
    patterns = meta.get("patterns", [])
    perms = meta.get("perms", [])
    frame_sizes = meta.get("frame_sizes", [])
    frozen_frame_size = meta.get("frozen_frame_size", None)

    payload, env = verify_and_open_capsule(capsule, flow_id)
    ring_id = env["ring_id"]
    ring_state = RINGS[ring_id]

    TELEMETRY.record_flow(flow_id, "capsule_opened", {
        "ring_id": ring_id,
        "payload_size": len(payload),
    })

    frames = []
    idx = 0
    size = len(payload)
    frame_index = 0
    while idx < size:
        if frozen_frame_size is not None:
            frame_size = frozen_frame_size
        elif frame_index < len(frame_sizes):
            frame_size = frame_sizes[frame_index]
        else:
            frame_size = ring_state.current_frame_size

        chunk = payload[idx:idx + frame_size]
        pattern_id = patterns[frame_index] if frame_index < len(patterns) else "base"
        perm = perms[frame_index] if frame_index < len(perms) else None
        decoded_frame = decode_frame_with_pre(
            chunk, flow_id, ring_state, frame_index, pattern_id, perm
        )
        frames.append(decoded_frame)
        idx += frame_size
        frame_index += 1

    encoded_raw = b"".join(frames)

    # --- FINAL PADDING REPAIR GUARD ---
    missing = (-len(encoded_raw)) % 4
    if missing:
        TELEMETRY.record_padding_issue(
            flow_id, -1,
            f"Padding repair: added {missing} '=' characters"
        )
        encoded_raw += b"=" * missing

    try:
        raw = base64.b64decode(encoded_raw)
    except Exception as e:
        TELEMETRY.record_padding_issue(
            flow_id, -1,
            f"Decode failed even after padding repair: {str(e)}"
        )
        return b"", env

    TELEMETRY.record_flow(flow_id, "egress", {
        "ring_id": ring_id,
        "size": len(raw),
    })

    return raw, env

# ============================================================
#  REAL TELEMETRY FEEDS (SYSTEM + GAMES + FS)
# ============================================================

KNOWN_GAME_HINTS = [
    "steam.exe",
    "epicgameslauncher.exe",
    "toxiccommando.exe",
    "toxic commando",
    "roadcraft.exe",
    "roadcraft",
    "back4blood.exe",
    "back 4 blood",
]

def detect_active_games():
    if psutil is None:
        return []
    active = []
    try:
        for p in psutil.process_iter(["pid", "name"]):
            name = (p.info.get("name") or "").lower()
            for hint in KNOWN_GAME_HINTS:
                if hint in name:
                    active.append({"pid": p.info["pid"], "name": name})
                    break
    except Exception:
        pass
    return active

def collect_system_metrics():
    if psutil is None:
        return None
    cpu = psutil.cpu_percent(interval=0.0)
    mem = psutil.virtual_memory().percent
    procs = []
    try:
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            info = p.info
            procs.append({
                "pid": info.get("pid"),
                "name": info.get("name"),
                "cpu": info.get("cpu_percent"),
                "mem": info.get("memory_percent"),
            })
    except Exception:
        pass

    active_games = detect_active_games()

    return {
        "type": "system_metrics",
        "cpu": cpu,
        "mem": mem,
        "processes": procs[:20],
        "active_games": active_games,
        "ts": time.time(),
    }

class FSHandler:
    def __init__(self):
        self.events = []

    def on_any_event(self, event):
        self.events.append({
            "type": "fs_event",
            "event_type": event.event_type,
            "src_path": event.src_path,
            "is_directory": event.is_directory,
            "ts": time.time(),
        })

def init_fs_watcher(path):
    if watchdog_observers is None:
        return None, None
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    handler = FSHandler()

    class _H(FileSystemEventHandler):
        def on_any_event(self, event):
            handler.on_any_event(event)

    observer = Observer()
    observer.schedule(_H(), path, recursive=True)
    observer.start()
    return observer, handler

def poll_fs_events(handler):
    if handler is None:
        return None
    if not handler.events:
        return None
    return handler.events.pop(0)

# ============================================================
#  GAME MANAGER PANEL (EXTRA WIDE + TALL + SCROLLBARS)
# ============================================================

class GameManager:
    def __init__(self, parent, known_games):
        self.parent = parent
        self.known_games = known_games
        self.frame = tk.Frame(parent, bd=2, relief="groove")
        self.frame.pack(side="bottom", fill="x", pady=5)

        tk.Label(self.frame, text="Game Manager", font=("Arial", 12, "bold")).pack()

        list_frame = tk.Frame(self.frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.listbox = tk.Listbox(
            list_frame,
            height=15,
            width=140,
            xscrollcommand=None
        )
        self.listbox.pack(side="left", fill="both", expand=True)

        xscroll = tk.Scrollbar(list_frame, orient="horizontal", command=self.listbox.xview)
        xscroll.pack(side="bottom", fill="x")
        self.listbox.config(xscrollcommand=xscroll.set)

        yscroll = tk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        yscroll.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=yscroll.set)

        btn_frame = tk.Frame(self.frame)
        btn_frame.pack(fill="x")

        tk.Button(btn_frame, text="Add Game", command=self.add_game).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Remove Selected", command=self.remove_selected).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Scan System", command=self.scan_system).pack(side="left", padx=5)

        self.refresh_list()

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for g in self.known_games:
            self.listbox.insert(tk.END, g)

    def add_game(self):
        path = filedialog.askopenfilename(
            title="Select Game EXE",
            filetypes=[("Executable Files", "*.exe")]
        )
        if path:
            exe = os.path.basename(path).lower()
            if exe not in self.known_games:
                self.known_games.append(exe)
                self.refresh_list()

    def remove_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        exe = self.listbox.get(sel[0])
        if exe in self.known_games:
            self.known_games.remove(exe)
        self.refresh_list()

    def scan_system(self):
        search_paths = [
            "C:/Program Files",
            "C:/Program Files (x86)",
            os.path.expanduser("~/AppData/Local"),
            os.path.expanduser("~/AppData/Roaming"),
        ]
        for root_path in search_paths:
            if not os.path.exists(root_path):
                continue
            for root, dirs, files in os.walk(root_path):
                for f in files:
                    if f.lower().endswith(".exe"):
                        exe = f.lower()
                        if exe not in self.known_games:
                            self.known_games.append(exe)
        self.refresh_list()

# ============================================================
#  COCKPIT GUI
# ============================================================

class BackboneCockpitGUI:
    def __init__(self, telemetry: TelemetrySink):
        self.telemetry = telemetry

        self.root = tk.Tk()
        self.root.title("Backbone Cockpit")
        self.root.geometry("1100x700")
        self.root.resizable(False, False)

        top = tk.Frame(self.root)
        top.pack(fill="x", pady=5)

        self.status_label = tk.Label(top, text="Status: UNKNOWN", font=("Arial", 14))
        self.status_label.pack(side="left", padx=10)

        self.capsule_label = tk.Label(top, text="Capsule: N/A", font=("Arial", 12))
        self.capsule_label.pack(side="right", padx=10)

        left = tk.Frame(self.root)
        left.pack(side="left", fill="y", padx=5, pady=5)

        tk.Label(left, text="Ring Pressure Gauges", font=("Arial", 12, "bold")).pack(pady=5)
        self.ring_gauges = {}
        for rid in ["ring1", "ring2", "ring3", "ring4"]:
            frame = tk.Frame(left)
            frame.pack(fill="x", pady=2)
            lbl = tk.Label(frame, text=rid, width=8, anchor="w")
            lbl.pack(side="left")
            bar = tk.Canvas(frame, width=150, height=12, bg="#222222")
            bar.pack(side="left", padx=5)
            self.ring_gauges[rid] = bar

        tk.Label(left, text="Multi-Ring Heatmap", font=("Arial", 12, "bold")).pack(pady=10)
        self.heatmap_labels = {}
        for rid in ["ring1", "ring2", "ring3", "ring4"]:
            lbl = tk.Label(left, text=f"{rid}: 0", width=20, anchor="w", bg="#202020", fg="white")
            lbl.pack(pady=1)
            self.heatmap_labels[rid] = lbl

        center = tk.Frame(self.root)
        center.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        tk.Label(center, text="PRE Shard-Jump Visualization", font=("Arial", 12, "bold")).pack(pady=5)
        self.pre_canvas = tk.Canvas(center, width=500, height=180, bg="#101010")
        self.pre_canvas.pack(pady=5)

        tk.Label(center, text="Flow Timeline", font=("Arial", 12, "bold")).pack(pady=5)
        self.timeline_canvas = tk.Canvas(center, width=500, height=140, bg="#101010")
        self.timeline_canvas.pack(pady=5)

        right = tk.Frame(self.root)
        right.pack(side="right", fill="y", padx=5, pady=5)

        tk.Label(right, text="Details", font=("Arial", 12, "bold")).pack(pady=5)

        self.pre_pattern_label = tk.Label(right, text="PRE Pattern: N/A", font=("Arial", 11))
        self.pre_pattern_label.pack(pady=3)

        self.pre_jump_label = tk.Label(right, text="Jump: N/A", font=("Arial", 11))
        self.pre_jump_label.pack(pady=3)

        self.flow_stage_label = tk.Label(right, text="Flow Stage: N/A", font=("Arial", 11))
        self.flow_stage_label.pack(pady=3)

        self.error_label = tk.Label(right, text="Errors: 0", font=("Arial", 11), fg="green")
        self.error_label.pack(pady=8)

        self.padding_label = tk.Label(right, text="Padding Issues: 0", font=("Arial", 10), fg="green")
        self.padding_label.pack(pady=3)

        self.boundary_label = tk.Label(right, text="Frame Boundary Issues: 0", font=("Arial", 10), fg="green")
        self.boundary_label.pack(pady=3)

        self.events_label = tk.Label(right, text="Events: 0", font=("Arial", 10))
        self.events_label.pack(pady=3)

        self.game_manager = GameManager(self.root, KNOWN_GAME_HINTS)

        self.update_loop()

    def _draw_ring_gauges(self, ring_bytes):
        max_bytes = max(ring_bytes.values()) if ring_bytes else 1
        for rid, canvas in self.ring_gauges.items():
            canvas.delete("all")
            val = ring_bytes.get(rid, 0)
            ratio = min(1.0, val / max_bytes) if max_bytes > 0 else 0
            width = int(150 * ratio)
            color = "#00ff00" if ratio < 0.4 else "#ffff00" if ratio < 0.8 else "#ff0000"
            canvas.create_rectangle(0, 0, width, 12, fill=color, outline="")

    def _draw_heatmap(self, ring_bytes):
        max_bytes = max(ring_bytes.values()) if ring_bytes else 1
        for rid, lbl in self.heatmap_labels.items():
            val = ring_bytes.get(rid, 0)
            ratio = min(1.0, val / max_bytes) if max_bytes > 0 else 0
            if ratio == 0:
                bg = "#202020"
            elif ratio < 0.4:
                bg = "#003300"
            elif ratio < 0.8:
                bg = "#666600"
            else:
                bg = "#660000"
            lbl.config(text=f"{rid}: {val}", bg=bg)

    def _draw_pre_animation(self, last_pre_event):
        self.pre_canvas.delete("all")
        if not last_pre_event:
            return

        anim = last_pre_event["data"]["animation"]
        jump_from = anim.get("jump_from")
        jump_to = anim.get("jump_to")

        w = 500
        h = 180
        margin = 50
        spacing = (w - 2 * margin) / 3
        y = h // 2

        positions = []
        for i in range(4):
            x = margin + i * spacing
            positions.append((x, y))
            self.pre_canvas.create_oval(x-18, y-18, x+18, y+18, outline="white")
            self.pre_canvas.create_text(x, y, text=f"S{i+1}", fill="white")

        if jump_from is not None and jump_to is not None:
            x1, y1 = positions[jump_from]
            x2, y2 = positions[jump_to]
            self.pre_canvas.create_line(x1, y1-25, x2, y2-25, fill="#00ffff", width=2, arrow=tk.LAST)

    def _draw_timeline(self, flow_events):
        self.timeline_canvas.delete("all")
        if not flow_events:
            return

        w = 500
        h = 140
        margin = 30
        steps = ["ingress", "frame_ingress", "frame_encoded", "post_transform",
                 "capsule_sealed", "capsule_opened", "frame_decode_ingress", "frame_decoded", "egress"]
        step_positions = {}
        spacing = (w - 2 * margin) / (len(steps) - 1)

        for i, s in enumerate(steps):
            x = margin + i * spacing
            y = h // 2
            step_positions[s] = (x, y)
            self.timeline_canvas.create_oval(x-6, y-6, x+6, y+6, fill="#444444", outline="")
            self.timeline_canvas.create_text(x, y+15, text=s, fill="#aaaaaa", font=("Arial", 7))

        for e in flow_events:
            stage = e["data"]["stage"]
            if stage in step_positions:
                x, y = step_positions[stage]
                self.timeline_canvas.create_oval(x-6, y-6, x+6, y+6, fill="#00ff00", outline="")

    def update_loop(self):
        summary = self.telemetry.summary()

        if summary["error_count"] > 0:
            self.status_label.config(text="Status: ERROR", fg="red")
        else:
            self.status_label.config(text="Status: RUNNING", fg="green")

        self.error_label.config(
            text=f"Errors: {summary['error_count']}",
            fg="red" if summary["error_count"] > 0 else "green",
        )
        self.events_label.config(text=f"Events: {summary['total_events']}")

        self.padding_label.config(
            text=f"Padding Issues: {summary['padding_issues']}",
            fg="red" if summary["padding_issues"] > 0 else "green",
        )
        self.boundary_label.config(
            text=f"Frame Boundary Issues: {summary['boundary_issues']}",
            fg="red" if summary['boundary_issues'] > 0 else "green",
        )

        ring_bytes = summary["ring_max_bytes_in_flight"]
        self._draw_ring_gauges(ring_bytes)
        self._draw_heatmap(ring_bytes)

        last_pre = None
        for e in reversed(self.telemetry.events):
            if e["kind"] == "pre_mapping":
                last_pre = e
                break

        if last_pre:
            pid = last_pre["data"]["pattern_id"]
            anim = last_pre["data"]["animation"]
            self.pre_pattern_label.config(text=f"PRE Pattern: {pid}")
            self.pre_jump_label.config(
                text=f"Jump: {anim.get('jump_from')} → {anim.get('jump_to')}"
            )
        else:
            self.pre_pattern_label.config(text="PRE Pattern: N/A")
            self.pre_jump_label.config(text="Jump: N/A")

        self._draw_pre_animation(last_pre)

        last_flow_id = None
        for e in reversed(self.telemetry.events):
            if e["kind"] == "flow":
                last_flow_id = e["data"]["flow_id"]
                break

        flow_events = []
        if last_flow_id:
            for e in self.telemetry.events:
                if e["kind"] == "flow" and e["data"]["flow_id"] == last_flow_id:
                    flow_events.append(e)
            if flow_events:
                last_stage = flow_events[-1]["data"]["stage"]
                self.flow_stage_label.config(text=f"Flow Stage: {last_stage}")
        else:
            self.flow_stage_label.config(text="Flow Stage: N/A")

        self._draw_timeline(flow_events)

        capsule_stage = "N/A"
        for e in reversed(self.telemetry.events):
            if e["kind"] == "flow":
                st = e["data"]["stage"]
                if st in ("capsule_sealed", "capsule_opened"):
                    capsule_stage = st
                    break
        self.capsule_label.config(text=f"Capsule: {capsule_stage}")

        self.root.after(300, self.update_loop)

    def start(self):
        self.root.mainloop()

# ============================================================
#  CONTINUOUS BACKBONE LOOP
# ============================================================

def run_backbone():
    device_id = "trusted-device-123"
    session_id = "session-abc"

    fs_observer, fs_handler = init_fs_watcher(".")

    try:
        while True:
            sys_evt = collect_system_metrics()
            if sys_evt is not None:
                raw = json.dumps(sys_evt).encode()
                capsule = backbone_encode(raw, device_id, session_id)
                decoded, env = backbone_decode(capsule)

            fs_evt = poll_fs_events(fs_handler)
            if fs_evt is not None:
                raw = json.dumps(fs_evt).encode()
                capsule = backbone_encode(raw, device_id, session_id)
                decoded, env = backbone_decode(capsule)

            time.sleep(0.5)
    finally:
        if fs_observer is not None:
            fs_observer.stop()
            fs_observer.join()

# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    gui = BackboneCockpitGUI(TELEMETRY)
    worker = threading.Thread(target=run_backbone, daemon=True)
    worker.start()
    gui.start()

