#!/usr/bin/env python3
"""
CHAMELEON SENTINEL — Unified Defensive Organism
Includes:
- Autoloader
- Reboot Memory
- 3 AI Queens
- Worker Swarm
- Guardian (data physics)
- Core Vault + Honeypot Vault
- Fake Identity + Activity History
- Sandbox Routing
- Tkinter GUI Status Dashboard
- Verified Master Password Input
- Master Password Change
- PIN Storage + Change
"""

import os, sys, json, time, base64, random, importlib, threading
from pathlib import Path
from getpass import getpass
from typing import Dict, Any, List

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

# ============================================================
#  DIRECTORIES / FILES
# ============================================================

CORE_DIR = Path.home() / ".chameleon_core"
HONEYPOT_DIR = Path.home() / ".chameleon_honeypot"
REBOOT_MEMORY = Path.home() / ".chameleon_reboot_memory.json"

VAULT_FILE = "chameleon_vault.bin"
SALT_FILE = "chameleon_salt.bin"
META_FILE = "chameleon_meta.json"
PIN_FILE = "chameleon_pin.bin"

IDLE_LOCK_SECONDS = 300
MAX_FAILED_ATTEMPTS = 5
BURST_WINDOW_SECONDS = 60
RISK_THRESHOLD = 5.0

current_mode = "core"


# ============================================================
#  REBOOT MEMORY
# ============================================================

def load_reboot_memory() -> Dict[str, Any]:
    if not REBOOT_MEMORY.exists():
        return {}
    try:
        with open(REBOOT_MEMORY, "r") as f:
            return json.load(f)
    except:
        return {}

def save_reboot_memory(data: Dict[str, Any]):
    with open(REBOOT_MEMORY, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
#  GUARDIAN (DATA PHYSICS ENGINE)
# ============================================================

class GuardianState:
    def __init__(self):
        mem = load_reboot_memory()
        self.failed_attempts = mem.get("failed_attempts", 0)
        self.access_count = mem.get("access_count", 0)
        self.last_access_ts = mem.get("last_access_ts", 0.0)
        self.altered_state = mem.get("altered_state", False)

    def persist(self):
        save_reboot_memory({
            "failed_attempts": self.failed_attempts,
            "access_count": self.access_count,
            "last_access_ts": self.last_access_ts,
            "altered_state": self.altered_state
        })

    def record_success(self):
        self.access_count += 1
        self.last_access_ts = time.time()
        self.persist()

    def record_failure(self):
        self.failed_attempts += 1
        if self.failed_attempts >= MAX_FAILED_ATTEMPTS:
            self.altered_state = True
        self.persist()

    def compute_risk_score(self) -> float:
        now = time.time()
        score = 0.0

        score += self.failed_attempts * 2.0

        if self.last_access_ts and (now - self.last_access_ts) < BURST_WINDOW_SECONDS:
            score += 3.0

        hour = time.localtime(now).tm_hour
        if hour < 6 or hour > 23:
            score += 2.0

        return score

guardian = GuardianState()


# ============================================================
#  AI QUEENS (3-LAYER BRAIN)
# ============================================================

class QueenRiskOracle:
    def decide_environment(self):
        risk = guardian.compute_risk_score()
        if guardian.altered_state or risk >= RISK_THRESHOLD:
            return "honeypot"
        return "core"

class QueenBehaviorOracle:
    def adjust_thresholds(self):
        # Placeholder for adaptive behavior
        pass

class QueenContinuityOracle:
    def restore_state(self):
        # Already handled by GuardianState loading reboot memory
        pass

queen_risk = QueenRiskOracle()
queen_behavior = QueenBehaviorOracle()
queen_continuity = QueenContinuityOracle()


# ============================================================
#  WORKER SWARM
# ============================================================

class Worker:
    def patrol(self):
        risk = guardian.compute_risk_score()
        if risk > RISK_THRESHOLD:
            guardian.altered_state = True
            guardian.persist()

workers = [Worker() for _ in range(10)]

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
        raise ValueError("Invalid password or corrupted vault.")
    guardian.record_success()
    return json.loads(decrypted.decode())

def save_vault(fernet: Fernet, data: Dict[str, Any]):
    plaintext = json.dumps(data, indent=2).encode()
    encrypted = fernet.encrypt(plaintext)
    with open(VAULT_FILE, "wb") as f:
        f.write(encrypted)

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
    print("[*] Honeypot seeded.")


# ============================================================
#  PASSWORD / PIN HELPERS
# ============================================================

def prompt_verified_input(label: str) -> str:
    """
    Ask user for a value twice and verify they match.
    Used for master password or PIN.
    """
    while True:
        v1 = getpass(f"{label}: ")
        v2 = getpass(f"Confirm {label}: ")
        if v1 == v2:
            return v1
        print("[!] Values do not match. Try again.")

def change_master_password():
    """
    Safely change the master password.
    Re-encrypts the vault with the new key.
    """
    print("\n=== Change Master Password ===")

    old_pw = getpass("Enter current master password: ")
    try:
        old_fernet = get_fernet(old_pw)
        vault = load_vault(old_fernet)
    except Exception:
        print("[!] Incorrect password. Aborting.")
        return

    new_pw = prompt_verified_input("New master password")
    new_fernet = get_fernet(new_pw)
    save_vault(new_fernet, vault)
    print("[+] Master password updated successfully.")

def change_pin():
    print("\n=== Change PIN ===")
    current = load_pin()
    if current:
        old_pin = getpass("Enter current PIN: ")
        if old_pin != current:
            print("[!] Incorrect PIN.")
            return
    new_pin = prompt_verified_input("New PIN")
    save_pin(new_pin)
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

    if guardian.altered_state or risk >= RISK_THRESHOLD:
        return False
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

def add_honey(fernet):
    vault = load_vault(fernet)
    label = input("Honey label: ").strip()
    value = input("Fake value: ").strip()
    vault[label] = {"value": value, "honey": True}
    save_vault(fernet, vault)

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
#  GUI DASHBOARD
# ============================================================

def start_gui():
    root = tk.Tk()
    root.title("Chameleon Sentinel Monitor")

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")

    lbl_mode = ttk.Label(frm, text="Mode: ")
    lbl_mode.grid(row=0, column=0, sticky="w")
    val_mode = ttk.Label(frm, text="")
    val_mode.grid(row=0, column=1, sticky="w")

    lbl_risk = ttk.Label(frm, text="Risk score: ")
    lbl_risk.grid(row=1, column=0, sticky="w")
    val_risk = ttk.Label(frm, text="")
    val_risk.grid(row=1, column=1, sticky="w")

    lbl_failed = ttk.Label(frm, text="Failed attempts: ")
    lbl_failed.grid(row=2, column=0, sticky="w")
    val_failed = ttk.Label(frm, text="")
    val_failed.grid(row=2, column=1, sticky="w")

    lbl_access = ttk.Label(frm, text="Access count: ")
    lbl_access.grid(row=3, column=0, sticky="w")
    val_access = ttk.Label(frm, text="")
    val_access.grid(row=3, column=1, sticky="w")

    lbl_altered = ttk.Label(frm, text="Altered state: ")
    lbl_altered.grid(row=4, column=0, sticky="w")
    val_altered = ttk.Label(frm, text="")
    val_altered.grid(row=4, column=1, sticky="w")

    def refresh():
        try:
            risk = guardian.compute_risk_score()
            val_mode.config(text=current_mode.upper())
            val_risk.config(text=f"{risk:.2f}")
            val_failed.config(text=str(guardian.failed_attempts))
            val_access.config(text=str(guardian.access_count))
            val_altered.config(text="YES" if guardian.altered_state else "NO")
        except Exception as e:
            val_mode.config(text=f"ERR: {e}")
        root.after(1000, refresh)

    refresh()
    root.mainloop()


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=== CHAMELEON SENTINEL ===")

    queen_continuity.restore_state()
    run_workers()

    mode = queen_risk.decide_environment()
    init_vault_env(mode)

    gui_thread = threading.Thread(target=start_gui, daemon=True)
    gui_thread.start()

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
            last = time.time()

        print("\n1) Add secret")
        print("2) Add honey secret")
        print("3) List labels")
        print("4) Get secret")
        print("5) Change master password")
        print("6) Change PIN")
        print("7) Exit")
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
            break

if __name__ == "__main__":
    main()
