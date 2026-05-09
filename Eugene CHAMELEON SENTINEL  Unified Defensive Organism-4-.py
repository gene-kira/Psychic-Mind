#!/usr/bin/env python3
"""
CHAMELEON SENTINEL T8 — Unified Defensive Organism

Features:
- Autoloader
- Reboot Memory
- 3 AI Queens
- Worker Swarm (patrol + anomaly sensing)
- Optimized Guardian (data physics risk engine)
- Optional GPU/NPU acceleration (if CuPy/Torch available)
- Core Vault + Honeypot Vault
- Fake Identity + Activity History
- Sandbox Routing (core vs honeypot)
- Swarm Sync (local + simple distributed via shared folder)
- Kernel Sensor Stubs + ETW-style placeholder (safe, defensive)
- Location Guardian (website + game server IP/region consistency)
- Tkinter Tactical Cockpit GUI:
    - Status panel
    - Threat timeline
    - Threat heatmap
    - Risk timeline graph
    - Worker swarm visualization
    - Location monitor panel
- Verified Master Password Input
- Master Password Change
- PIN Storage + Change
- Local Remote API (read-only, localhost only)
- Simple website verification (allowlist + location consistency)
"""

import os, sys, json, time, base64, random, importlib, threading, socket
from pathlib import Path
from getpass import getpass
from typing import Dict, Any, List
from urllib.parse import urlparse

# ============================================================
#  AUTLOADER
# ============================================================

REQUIRED_LIBS = ["cryptography"]

def autoload_libraries():
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            missing.append(lib)

    if missing:
        print("\n[!] Missing libraries:", ", ".join(missing))
        print("[!] Install them with:")
        print("    pip install " + " ".join(missing))
        sys.exit(1)

autoload_libraries()

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

import tkinter as tk
from tkinter import ttk

# Optional GPU/NPU libs
try:
    cupy = importlib.import_module("cupy")
except ImportError:
    cupy = None

try:
    torch = importlib.import_module("torch")
except ImportError:
    torch = None

# Remote API (local only)
import http.server
import socketserver

# ============================================================
#  DIRECTORIES / FILES
# ============================================================

CORE_DIR = Path.home() / ".chameleon_core"
HONEYPOT_DIR = Path.home() / ".chameleon_honeypot"
REBOOT_MEMORY = Path.home() / ".chameleon_reboot_memory.json"
SWARM_STATE = Path.home() / ".chameleon_swarm_state.json"

SWARM_PEERS_DIR = Path.home() / ".chameleon_swarm_peers"
SWARM_PEERS_DIR.mkdir(exist_ok=True)

VAULT_FILE = "chameleon_vault.bin"
SALT_FILE = "chameleon_salt.bin"
META_FILE = "chameleon_meta.json"
PIN_FILE = "chameleon_pin.bin"

LOCATION_DB_FILE = Path.home() / ".chameleon_location_db.json"

IDLE_LOCK_SECONDS = 300
MAX_FAILED_ATTEMPTS = 5
BURST_WINDOW_SECONDS = 60
RISK_THRESHOLD_BASE = 5.0

current_mode = "core"

# ============================================================
#  THREAT TIMELINE
# ============================================================

THREAT_LOG_MAX = 200
threat_log: List[Dict[str, Any]] = []

def log_event(kind: str, detail: str, level: str = "info"):
    evt = {
        "ts": time.time(),
        "kind": kind,
        "detail": detail,
        "level": level,
    }
    threat_log.append(evt)
    if len(threat_log) > THREAT_LOG_MAX:
        del threat_log[0]

# ============================================================
#  REBOOT MEMORY
# ============================================================

def load_reboot_memory() -> Dict[str, Any]:
    if not REBOOT_MEMORY.exists():
        return {}
    try:
        with open(REBOOT_MEMORY, "r") as f:
            data = json.load(f)
        global threat_log
        threat_log = data.get("threat_log", [])
        return data
    except:
        return {}

def save_reboot_memory(data: Dict[str, Any]):
    data = dict(data)
    data["threat_log"] = threat_log[-50:]
    with open(REBOOT_MEMORY, "w") as f:
        json.dump(data, f, indent=2)

# ============================================================
#  SWARM SYNC (LOCAL + DISTRIBUTED VIA SHARED FOLDER)
# ============================================================

def load_swarm_state() -> Dict[str, Any]:
    if not SWARM_STATE.exists():
        return {}
    try:
        with open(SWARM_STATE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_swarm_state(data: Dict[str, Any]):
    with open(SWARM_STATE, "w") as f:
        json.dump(data, f, indent=2)

def update_swarm_risk(risk: float):
    state = load_swarm_state()
    history = state.get("risk_history", [])
    history.append({"ts": time.time(), "risk": risk})
    history = history[-200:]
    state["risk_history"] = history
    state["last_risk"] = risk
    save_swarm_state(state)
    peer_file = SWARM_PEERS_DIR / f"peer_{socket.gethostname()}.json"
    with open(peer_file, "w") as f:
        json.dump({"last_risk": risk, "ts": time.time()}, f)

def get_swarm_risk_hint() -> float:
    state = load_swarm_state()
    local_hint = float(state.get("last_risk", 0.0))
    peer_hints = []
    for p in SWARM_PEERS_DIR.glob("peer_*.json"):
        try:
            with open(p, "r") as f:
                d = json.load(f)
            peer_hints.append(float(d.get("last_risk", 0.0)))
        except:
            continue
    if peer_hints:
        avg_peer = sum(peer_hints) / len(peer_hints)
        return max(local_hint, avg_peer)
    return local_hint

def get_risk_history() -> List[Dict[str, Any]]:
    state = load_swarm_state()
    return state.get("risk_history", [])

# ============================================================
#  GUARDIAN (OPTIMIZED DATA PHYSICS ENGINE)
# ============================================================

class GuardianState:
    def __init__(self):
        mem = load_reboot_memory()
        self.failed_attempts = mem.get("failed_attempts", 0)
        self.access_count = mem.get("access_count", 0)
        self.last_access_ts = mem.get("last_access_ts", 0.0)
        self.altered_state = mem.get("altered_state", False)
        self.dynamic_threshold = mem.get("dynamic_threshold", RISK_THRESHOLD_BASE)

    def persist(self):
        save_reboot_memory({
            "failed_attempts": self.failed_attempts,
            "access_count": self.access_count,
            "last_access_ts": self.last_access_ts,
            "altered_state": self.altered_state,
            "dynamic_threshold": self.dynamic_threshold
        })

    def record_success(self):
        self.access_count += 1
        self.last_access_ts = time.time()
        self.persist()

    def record_failure(self):
        self.failed_attempts += 1
        if self.failed_attempts >= MAX_FAILED_ATTEMPTS:
            if not self.altered_state:
                log_event("state", "Guardian entered altered state due to failures", "alert")
            self.altered_state = True
        self.persist()

    def _cpu_risk_components(self) -> List[float]:
        now = time.time()
        comps = []
        comps.append(self.failed_attempts * 2.0)
        if self.last_access_ts and (now - self.last_access_ts) < BURST_WINDOW_SECONDS:
            comps.append(3.0)
        else:
            comps.append(0.0)
        hour = time.localtime(now).tm_hour
        comps.append(2.0 if (hour < 6 or hour > 23) else 0.0)
        swarm_hint = get_swarm_risk_hint()
        comps.append(min(swarm_hint, 5.0) * 0.5)
        return comps

    def compute_risk_score(self) -> float:
        comps = self._cpu_risk_components()
        if cupy is not None:
            try:
                arr = cupy.array(comps, dtype=cupy.float32)
                score = float(arr.sum().get())
                return score
            except Exception:
                pass
        if torch is not None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                t = torch.tensor(comps, dtype=torch.float32, device=device)
                score = float(t.sum().cpu().item())
                return score
            except Exception:
                pass
        return float(sum(comps))

guardian = GuardianState()

# ============================================================
#  AI QUEENS (3-LAYER BRAIN)
# ============================================================

class QueenRiskOracle:
    def decide_environment(self):
        risk = guardian.compute_risk_score()
        update_swarm_risk(risk)
        threshold = guardian.dynamic_threshold
        if guardian.altered_state or risk >= threshold:
            log_event("risk", f"Risk {risk:.2f} >= threshold {threshold:.2f}, using HONEYPOT", "warn")
            return "honeypot"
        log_event("risk", f"Risk {risk:.2f} below threshold {threshold:.2f}, using CORE", "info")
        return "core"

class QueenBehaviorOracle:
    def adjust_thresholds(self):
        state = load_swarm_state()
        hist = state.get("risk_history", [])
        if not hist:
            return
        avg = sum(h["risk"] for h in hist) / len(hist)
        old = guardian.dynamic_threshold
        if avg > RISK_THRESHOLD_BASE * 1.5:
            guardian.dynamic_threshold = max(3.0, guardian.dynamic_threshold - 0.5)
        elif avg < RISK_THRESHOLD_BASE * 0.5:
            guardian.dynamic_threshold = min(10.0, guardian.dynamic_threshold + 0.5)
        if guardian.dynamic_threshold != old:
            log_event("tuning", f"Adjusted dynamic threshold {old:.2f} -> {guardian.dynamic_threshold:.2f}", "info")
        guardian.persist()

class QueenContinuityOracle:
    def restore_state(self):
        pass

queen_risk = QueenRiskOracle()
queen_behavior = QueenBehaviorOracle()
queen_continuity = QueenContinuityOracle()

# ============================================================
#  KERNEL SENSOR STUBS + ETW-STYLE PLACEHOLDER
# ============================================================

class KernelSensor:
    def __init__(self):
        self.last_events: List[Dict[str, Any]] = []

    def poll(self):
        if random.random() < 0.1:
            evt = {
                "ts": time.time(),
                "type": "process",
                "detail": "benign_process_activity"
            }
            self.last_events.append(evt)
            self.last_events = self.last_events[-20:]

    def get_recent_events(self) -> List[Dict[str, Any]]:
        return list(self.last_events)

class WindowsETWSensor:
    def __init__(self):
        self.enabled = False

    def start(self):
        self.enabled = True
        log_event("kernel", "ETW sensor placeholder started (no real hooks)", "info")

    def stop(self):
        self.enabled = False
        log_event("kernel", "ETW sensor placeholder stopped", "info")

kernel_sensor = KernelSensor()
etw_sensor = WindowsETWSensor()

# ============================================================
#  WORKER SWARM
# ============================================================

class Worker:
    def __init__(self, wid: int):
        self.wid = wid
        self.last_risk = 0.0
        self.status = "idle"

    def patrol(self):
        kernel_sensor.poll()
        risk = guardian.compute_risk_score()
        self.last_risk = risk
        self.status = "patrolling"
        if risk > guardian.dynamic_threshold and not guardian.altered_state:
            guardian.altered_state = True
            guardian.persist()
            log_event("state", f"Guardian entered altered state (risk {risk:.2f})", "alert")

workers: List[Worker] = [Worker(i) for i in range(10)]

def run_workers():
    for w in workers:
        w.patrol()

# ============================================================
#  SANDBOX ENVIRONMENT
# ============================================================

def init_vault_env(mode: str):
    global current_mode
    current_mode = mode
    base = CORE_DIR if mode == "core" else HONEYPOT_DIR
    base.mkdir(mode=0o700, exist_ok=True)
    if base.is_symlink():
        print("[-] Vault directory is a symlink. Refusing.")
        sys.exit(1)
    os.chdir(base)
    print(f"[*] Environment: {mode.upper()} @ {base}")
    log_event("mode", f"Environment selected: {mode.upper()}", "info")

# ============================================================
#  CRYPTO HELPERS
# ============================================================

def _load_or_create_salt() -> bytes:
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE, "rb") as f:
            return f.read()
    salt = os.urandom(16)
    with open(SALT_FILE, "wb") as f:
        f.write(salt)
    return salt

def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=200_000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())
    return base64.urlsafe_b64encode(key)

def get_fernet(password: str) -> Fernet:
    salt = _load_or_create_salt()
    key = _derive_key(password, salt)
    return Fernet(key)

# ============================================================
#  PIN STORAGE
# ============================================================

def save_pin(pin: str):
    with open(PIN_FILE, "w") as f:
        f.write(pin)

def load_pin() -> str:
    if not os.path.exists(PIN_FILE):
        return ""
    with open(PIN_FILE, "r") as f:
        return f.read().strip()

# ============================================================
#  VAULT + META
# ============================================================

def load_vault(fernet: Fernet) -> Dict[str, Any]:
    if not os.path.exists(VAULT_FILE):
        return {}
    with open(VAULT_FILE, "rb") as f:
        encrypted = f.read()
    try:
        decrypted = fernet.decrypt(encrypted)
    except InvalidToken:
        guardian.record_failure()
        log_event("auth", "Invalid vault password", "warn")
        raise ValueError("Invalid password or corrupted vault.")
    guardian.record_success()
    log_event("auth", "Vault decrypted successfully", "info")
    return json.loads(decrypted.decode())

def save_vault(fernet: Fernet, data: Dict[str, Any]):
    plaintext = json.dumps(data, indent=2).encode()
    encrypted = fernet.encrypt(plaintext)
    with open(VAULT_FILE, "wb") as f:
        f.write(encrypted)
    log_event("vault", "Vault saved", "info")

def load_meta() -> Dict[str, Any]:
    if not os.path.exists(META_FILE):
        return {}
    with open(META_FILE, "r") as f:
        return json.load(f)

def save_meta(meta: Dict[str, Any]):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

# ============================================================
#  FAKE IDENTITY + ACTIVITY HISTORY
# ============================================================

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan"]
LAST_NAMES = ["Smith", "Johnson", "Brown", "Davis"]
DOMAINS = ["example.com", "mailservice.net"]
CITIES = ["New York", "Chicago", "Seattle"]
STATES = ["NY", "IL", "WA"]
USER_AGENTS = ["Chrome/124 Windows", "Edge/123 Windows"]
IP_BLOCKS = ["192.0.2.", "198.51.100."]

def random_phone():
    return f"+1-555-{random.randint(1000,9999)}"

def random_ip():
    return random.choice(IP_BLOCKS) + str(random.randint(1,254))

def random_email(first,last):
    return f"{first.lower()}.{last.lower()}@{random.choice(DOMAINS)}"

def random_address():
    return f"{random.randint(100,9999)} Oak St, {random.choice(CITIES)}, {random.choice(STATES)} {random.randint(10000,99999)}"

def generate_fake_activity():
    now = time.time()
    return {
        "logins": [
            {
                "timestamp": now - random.randint(1,7*86400),
                "ip": random_ip(),
                "user_agent": random.choice(USER_AGENTS),
                "status": random.choice(["success","failed"])
            }
            for _ in range(5)
        ]
    }

def generate_fake_identity():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return {
        "full_name": {"value": f"{first} {last}", "honey": True},
        "email": {"value": random_email(first,last), "honey": True},
        "phone": {"value": random_phone(), "honey": True},
        "address": {"value": random_address(), "honey": True},
        "activity": {"value": generate_fake_activity(), "honey": True}
    }

def seed_honeypot(fernet):
    try:
        vault = load_vault(fernet)
    except:
        vault = {}
    if vault:
        return
    vault.update(generate_fake_identity())
    save_vault(fernet, vault)
    log_event("honeypot", "Honeypot seeded with fake identity", "info")
    print("[*] Honeypot seeded.")

# ============================================================
#  PASSWORD / PIN HELPERS
# ============================================================

def prompt_verified_input(label: str) -> str:
    while True:
        v1 = getpass(f"{label}: ")
        v2 = getpass(f"Confirm {label}: ")
        if v1 == v2:
            return v1
        print("[!] Values do not match. Try again.")

def change_master_password():
    print("\n=== Change Master Password ===")
    old_pw = getpass("Enter current master password: ")
    try:
        old_fernet = get_fernet(old_pw)
        vault = load_vault(old_fernet)
    except Exception:
        print("[!] Incorrect password. Aborting.")
        log_event("auth", "Failed master password change (bad current password)", "warn")
        return
    new_pw = prompt_verified_input("New master password")
    new_fernet = get_fernet(new_pw)
    save_vault(new_fernet, vault)
    log_event("auth", "Master password changed", "info")
    print("[+] Master password updated successfully.")

def change_pin():
    print("\n=== Change PIN ===")
    current = load_pin()
    if current:
        old_pin = getpass("Enter current PIN: ")
        if old_pin != current:
            print("[!] Incorrect PIN.")
            log_event("auth", "Failed PIN change (bad current PIN)", "warn")
            return
    new_pin = prompt_verified_input("New PIN")
    save_pin(new_pin)
    log_event("auth", "PIN changed", "info")
    print("[+] PIN updated successfully.")

# ============================================================
#  GUARDIAN DECISION
# ============================================================

def guardian_allows(label: str) -> bool:
    risk = guardian.compute_risk_score()
    meta = load_meta()
    lm = meta.get(label, {})
    now = time.time()
    if now - lm.get("last_access_ts", 0) < 5:
        risk += 2.0
    lm["last_access_ts"] = now
    lm["count"] = lm.get("count", 0) + 1
    meta[label] = lm
    save_meta(meta)
    if guardian.altered_state or risk >= guardian.dynamic_threshold:
        log_event("access", f"Denied access to '{label}' (risk {risk:.2f})", "warn")
        return False
    log_event("access", f"Allowed access to '{label}' (risk {risk:.2f})", "info")
    return True

# ============================================================
#  VAULT OPERATIONS
# ============================================================

def add_secret(fernet):
    vault = load_vault(fernet)
    label = input("Label: ").strip()
    value = input("Value: ").strip()
    vault[label] = {"value": value, "honey": False}
    save_vault(fernet, vault)
    log_event("vault", f"Added REAL secret '{label}'", "info")

def add_honey(fernet):
    vault = load_vault(fernet)
    label = input("Honey label: ").strip()
    value = input("Fake value: ").strip()
    vault[label] = {"value": value, "honey": True}
    save_vault(fernet, vault)
    log_event("vault", f"Added HONEY secret '{label}'", "info")

def list_labels(fernet):
    vault = load_vault(fernet)
    if not vault:
        print("Vault empty.")
        return
    for k,v in vault.items():
        print(f"{k} [{'HONEY' if v.get('honey') else 'REAL'}]")

def get_secret(fernet):
    vault = load_vault(fernet)
    label = input("Label: ").strip()
    if label not in vault:
        print("Not found.")
        log_event("access", f"Requested unknown label '{label}'", "info")
        return
    if not guardian_allows(label):
        print("[!] Guardian denies access — showing decoy if available.")
        if vault[label].get("honey"):
            print(vault[label]["value"])
        else:
            print("Access denied.")
        return
    print(vault[label]["value"])

# ============================================================
#  LOCATION GUARDIAN (WEBSITES + GAMES)
# ============================================================

ALLOWED_DOMAINS = {
    "example.com",
    "microsoft.com",
    "github.com",
}

def load_location_db() -> Dict[str, Any]:
    if not LOCATION_DB_FILE.exists():
        return {"websites": {}, "games": {}}
    try:
        with open(LOCATION_DB_FILE, "r") as f:
            return json.load(f)
    except:
        return {"websites": {}, "games": {}}

def save_location_db(db: Dict[str, Any]):
    with open(LOCATION_DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

def simple_ip_region(ip: str) -> str:
    if ip.startswith("192.0.2."):
        return "TEST-NET-1"
    if ip.startswith("198.51.100."):
        return "TEST-NET-2"
    if ip.startswith("203.0.113."):
        return "TEST-NET-3"
    try:
        first_octet = int(ip.split(".")[0])
        if 1 <= first_octet <= 126:
            return "Region-A"
        if 128 <= first_octet <= 191:
            return "Region-B"
        if 192 <= first_octet <= 223:
            return "Region-C"
    except:
        pass
    return "Unknown"

def resolve_host(host: str) -> str:
    try:
        ip = socket.gethostbyname(host)
        return ip
    except Exception:
        return ""

def verify_website(url: str) -> Dict[str, Any]:
    db = load_location_db()
    websites = db.get("websites", {})
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        scheme = parsed.scheme or ""
        domain_ok = any(host.endswith(d) for d in ALLOWED_DOMAINS)
        ip = resolve_host(host) if host else ""
        region = simple_ip_region(ip) if ip else "Unknown"

        prev = websites.get(host, {})
        changed_ip = prev.get("ip") and prev.get("ip") != ip
        changed_region = prev.get("region") and prev.get("region") != region

        websites[host] = {
            "ip": ip,
            "region": region,
            "last_ts": time.time(),
        }
        db["websites"] = websites
        save_location_db(db)

        if changed_ip or changed_region:
            log_event(
                "location",
                f"Website {host} changed location: ip {prev.get('ip')} -> {ip}, region {prev.get('region')} -> {region}",
                "warn"
            )
        else:
            log_event("location", f"Website {host} verified: ip={ip}, region={region}", "info")

        result = {
            "url": url,
            "scheme": scheme,
            "host": host,
            "ip": ip,
            "region": region,
            "allowed_domain": domain_ok,
            "changed_ip": bool(changed_ip),
            "changed_region": bool(changed_region),
        }
        return result
    except Exception as e:
        log_event("web", f"Website verification error: {e}", "warn")
        return {"url": url, "error": str(e)}

def verify_game_server(name: str, host_or_ip: str) -> Dict[str, Any]:
    db = load_location_db()
    games = db.get("games", {})
    try:
        if any(c.isalpha() for c in host_or_ip):
            ip = resolve_host(host_or_ip)
            host = host_or_ip
        else:
            ip = host_or_ip
            host = ""
        region = simple_ip_region(ip) if ip else "Unknown"

        prev = games.get(name, {})
        changed_ip = prev.get("ip") and prev.get("ip") != ip
        changed_region = prev.get("region") and prev.get("region") != region

        games[name] = {
            "host": host,
            "ip": ip,
            "region": region,
            "last_ts": time.time(),
        }
        db["games"] = games
        save_location_db(db)

        if changed_ip or changed_region:
            log_event(
                "location",
                f"Game {name} changed server: ip {prev.get('ip')} -> {ip}, region {prev.get('region')} -> {region}",
                "warn"
            )
        else:
            log_event("location", f"Game {name} server verified: ip={ip}, region={region}", "info")

        return {
            "game": name,
            "host": host,
            "ip": ip,
            "region": region,
            "changed_ip": bool(changed_ip),
            "changed_region": bool(changed_region),
        }
    except Exception as e:
        log_event("location", f"Game verification error: {e}", "warn")
        return {"game": name, "error": str(e)}

def get_location_summary() -> Dict[str, Any]:
    db = load_location_db()
    return {
        "websites": db.get("websites", {}),
        "games": db.get("games", {}),
    }

# ============================================================
#  GUI TACTICAL COCKPIT
# ============================================================

def start_gui():
    root = tk.Tk()
    root.title("Chameleon Sentinel T8 Cockpit")

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    main = ttk.Frame(root, padding=5)
    main.grid(row=0, column=0, sticky="nsew")
    main.rowconfigure(0, weight=1)
    main.columnconfigure(0, weight=1)
    main.columnconfigure(1, weight=1)

    # Left: status + timeline
    left = ttk.Frame(main, padding=5)
    left.grid(row=0, column=0, sticky="nsew")
    left.rowconfigure(3, weight=1)

    labels = {}
    values = {}

    fields = [
        ("Mode", "mode"),
        ("Risk score", "risk"),
        ("Dynamic threshold", "threshold"),
        ("Failed attempts", "failed"),
        ("Access count", "access"),
        ("Altered state", "altered"),
        ("Swarm last risk", "swarm_risk"),
    ]

    for i, (text, key) in enumerate(fields):
        labels[key] = ttk.Label(left, text=text + ": ")
        labels[key].grid(row=i, column=0, sticky="w")
        values[key] = ttk.Label(left, text="")
        values[key].grid(row=i, column=1, sticky="w")

    sep = ttk.Separator(left, orient="horizontal")
    sep.grid(row=len(fields), column=0, columnspan=2, sticky="ew", pady=(5, 3))

    lbl_timeline = ttk.Label(left, text="Threat timeline:")
    lbl_timeline.grid(row=len(fields)+1, column=0, columnspan=2, sticky="w")

    timeline = tk.Listbox(left, height=12, width=80)
    timeline.grid(row=len(fields)+2, column=0, columnspan=2, sticky="nsew")

    # Right: heatmap + risk graph + worker swarm + location monitor
    right = ttk.Frame(main, padding=5)
    right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(3, weight=1)

    # Heatmap
    heat_frame = ttk.LabelFrame(right, text="Threat heatmap (counts by level)")
    heat_frame.grid(row=0, column=0, sticky="ew", pady=(0,5))
    heat_info = ttk.Label(heat_frame, text="")
    heat_info.grid(row=0, column=0, sticky="w")

    # Risk timeline graph
    graph_frame = ttk.LabelFrame(right, text="Risk timeline")
    graph_frame.grid(row=1, column=0, sticky="nsew", pady=(0,5))
    graph_frame.rowconfigure(0, weight=1)
    graph_frame.columnconfigure(0, weight=1)
    canvas = tk.Canvas(graph_frame, height=150, bg="black")
    canvas.grid(row=0, column=0, sticky="nsew")

    # Worker swarm visualization
    swarm_frame = ttk.LabelFrame(right, text="Worker swarm")
    swarm_frame.grid(row=2, column=0, sticky="nsew", pady=(0,5))
    swarm_list = tk.Listbox(swarm_frame, height=6, width=40)
    swarm_list.grid(row=0, column=0, sticky="nsew")

    # Location monitor
    loc_frame = ttk.LabelFrame(right, text="Location monitor (websites + games)")
    loc_frame.grid(row=3, column=0, sticky="nsew")
    loc_list = tk.Listbox(loc_frame, height=8, width=60)
    loc_list.grid(row=0, column=0, sticky="nsew")

    def format_event(evt):
        ts = time.strftime("%H:%M:%S", time.localtime(evt["ts"]))
        lvl = evt["level"].upper()
        kind = evt["kind"]
        detail = evt["detail"]
        return f"[{ts}] [{lvl}] {kind}: {detail}"

    def compute_heatmap_counts():
        counts = {"info": 0, "warn": 0, "alert": 0}
        for evt in threat_log[-100:]:
            lvl = evt.get("level", "info")
            if lvl in counts:
                counts[lvl] += 1
        return counts

    def draw_risk_graph():
        canvas.delete("all")
        history = get_risk_history()[-50:]
        if not history:
            return
        w = int(canvas.winfo_width() or 1)
        h = int(canvas.winfo_height() or 1)
        max_risk = max(h["risk"] for h in history) or 1.0
        step = max(1, w // max(1, len(history)-1))
        points = []
        for i, entry in enumerate(history):
            x = i * step
            y = h - int((entry["risk"] / max_risk) * (h-10))
            points.append((x, y))
        for i in range(1, len(points)):
            x1,y1 = points[i-1]
            x2,y2 = points[i]
            canvas.create_line(x1,y1,x2,y2, fill="lime", width=2)

    def refresh():
        try:
            risk = guardian.compute_risk_score()
            swarm_hint = get_swarm_risk_hint()
            values["mode"].config(text=current_mode.upper())
            values["risk"].config(text=f"{risk:.2f}")
            values["threshold"].config(text=f"{guardian.dynamic_threshold:.2f}")
            values["failed"].config(text=str(guardian.failed_attempts))
            values["access"].config(text=str(guardian.access_count))
            values["altered"].config(text="YES" if guardian.altered_state else "NO")
            values["swarm_risk"].config(text=f"{swarm_hint:.2f}")

            timeline.delete(0, tk.END)
            for evt in threat_log[-50:]:
                timeline.insert(tk.END, format_event(evt))

            counts = compute_heatmap_counts()
            heat_info.config(text=f"INFO: {counts['info']}   WARN: {counts['warn']}   ALERT: {counts['alert']}")

            draw_risk_graph()

            swarm_list.delete(0, tk.END)
            for w in workers:
                swarm_list.insert(tk.END, f"Worker {w.wid}: {w.status}, last_risk={w.last_risk:.2f}")

            loc_list.delete(0, tk.END)
            loc_db = get_location_summary()
            loc_list.insert(tk.END, "--- Websites ---")
            for host, info in loc_db.get("websites", {}).items():
                loc_list.insert(
                    tk.END,
                    f"{host}: ip={info.get('ip')}, region={info.get('region')}"
                )
            loc_list.insert(tk.END, "--- Games ---")
            for name, info in loc_db.get("games", {}).items():
                loc_list.insert(
                    tk.END,
                    f"{name}: ip={info.get('ip')}, region={info.get('region')}"
                )

        except Exception as e:
            values["mode"].config(text=f"ERR: {e}")
        root.after(1000, refresh)

    refresh()
    root.mainloop()

# ============================================================
#  REMOTE API (LOCALHOST ONLY, READ-ONLY)
# ============================================================

class SentinelAPIHandler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, obj: Any, code: int = 200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            self._send_json({"error": "local access only"}, 403)
            return

        if self.path == "/status":
            risk = guardian.compute_risk_score()
            swarm_hint = get_swarm_risk_hint()
            self._send_json({
                "mode": current_mode,
                "risk": risk,
                "dynamic_threshold": guardian.dynamic_threshold,
                "failed_attempts": guardian.failed_attempts,
                "access_count": guardian.access_count,
                "altered_state": guardian.altered_state,
                "swarm_risk": swarm_hint,
            })
        elif self.path == "/timeline":
            self._send_json(threat_log[-100:])
        elif self.path == "/workers":
            self._send_json([
                {"id": w.wid, "status": w.status, "last_risk": w.last_risk}
                for w in workers
            ])
        elif self.path == "/locations":
            self._send_json(get_location_summary())
        else:
            self._send_json({"error": "unknown endpoint"}, 404)

def start_api_server():
    try:
        with socketserver.TCPServer(("127.0.0.1", 8765), SentinelAPIHandler) as httpd:
            log_event("api", "Local API server started on 127.0.0.1:8765", "info")
            httpd.serve_forever()
    except OSError:
        log_event("api", "API server failed to bind (maybe already running)", "warn")

# ============================================================
#  MAIN
# ============================================================

def main():
    print("=== CHAMELEON SENTINEL T8 ===")

    queen_continuity.restore_state()
    queen_behavior.adjust_thresholds()
    etw_sensor.start()
    run_workers()

    mode = queen_risk.decide_environment()
    init_vault_env(mode)

    gui_thread = threading.Thread(target=start_gui, daemon=True)
    gui_thread.start()

    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()

    password = prompt_verified_input("Master password")
    fernet = get_fernet(password)

    if mode == "honeypot":
        seed_honeypot(fernet)

    last = time.time()

    while True:
        if time.time() - last > IDLE_LOCK_SECONDS:
            print("[*] Auto-lock.")
            password = prompt_verified_input("Re-enter master password")
            fernet = get_fernet(password)
            guardian.failed_attempts = 0
            guardian.altered_state = False
            guardian.persist()
            log_event("auth", "Vault auto-locked and re-authenticated", "info")
            last = time.time()

        print("\n1) Add secret")
        print("2) Add honey secret")
        print("3) List labels")
        print("4) Get secret")
        print("5) Change master password")
        print("6) Change PIN")
        print("7) Verify website")
        print("8) Verify game server")
        print("9) Exit")
        c = input("> ").strip()
        last = time.time()

        if c == "1":
            add_secret(fernet)
        elif c == "2":
            add_honey(fernet)
        elif c == "3":
            list_labels(fernet)
        elif c == "4":
            get_secret(fernet)
        elif c == "5":
            change_master_password()
        elif c == "6":
            change_pin()
        elif c == "7":
            url = input("Website URL: ").strip()
            info = verify_website(url)
            print(json.dumps(info, indent=2))
        elif c == "8":
            name = input("Game name: ").strip()
            host_or_ip = input("Game host or IP: ").strip()
            info = verify_game_server(name, host_or_ip)
            print(json.dumps(info, indent=2))
        elif c == "9":
            log_event("system", "Sentinel shutting down", "info")
            etw_sensor.stop()
            break

if __name__ == "__main__":
    main()
