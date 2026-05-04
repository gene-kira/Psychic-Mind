#!/usr/bin/env python3
# borg_glyph_vault_unified_v2.py
#
# Concept demo ONLY:
# - Glyph codec
# - RSA + AES-GCM crypto
# - TLS-like handshake
# - Local vault
# - Borg Collective: 3 Queens + workers, all required to lock/unlock
# - Machine-bound via per-queen local secrets + OS/host/user
# - Collective fingerprint + Queen health
# - Tkinter GUI overview with Queen status
#
# NOT FOR REAL SECURITY USE.

import os
import json
import platform
import hashlib
import importlib
import tkinter as tk
from tkinter import scrolledtext, messagebox

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
# Borg Queens (machine-bound keys + workers + health)
# -----------------------------

class BorgQueen:
    """
    Each Queen has its own local secret file and derives a key from:
    - OS name
    - hostname
    - username
    - queen_id
    - queen-specific local secret
    Workers: multiple derived worker keys wrap/unwarp the blob.
    """

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
            # First-time init: create secret; in real life you'd warn the operator
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
# Borg Collective wrapper (3 Queens + collective fingerprint)
# -----------------------------

class BorgCollectiveWrapper:
    """
    Wrap:
      plaintext -> master_key -> AES-GCM(ciphertext)
      master_key is then wrapped sequentially by Queen1, Queen2, Queen3
      using their workers.
    Unwrap:
      reverse sequence: Queen3, Queen2, Queen1.
    If any Queen's key changes or fingerprint mismatches, unwrap fails.
    """

    def __init__(self, queens):
        if len(queens) != 3:
            raise ValueError("BorgCollectiveWrapper requires exactly 3 Queens")
        self.queens = queens

    def _collective_fingerprint(self) -> str:
        h = hashlib.sha256()
        for q in self.queens:
            h.update(q.get_key())
        return h.hexdigest()

    def wrap(self, plaintext: bytes, meta: dict) -> bytes:
        master_key = os.urandom(32)
        aes = AESGCM(master_key)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext, b"")

        wrapped_master = master_key
        for q in self.queens:
            wrapped_master = q.lock_with_workers(wrapped_master)

        obj = {
            "version": 1,
            "algo": "AES-256-GCM",
            "wrapped_master": wrapped_master.hex(),
            "ciphertext": (nonce + ct).hex(),
            "meta": {
                **meta,
                "collective_fp": self._collective_fingerprint(),
                "queens": [q.queen_id for q in self.queens],
            },
        }
        return json.dumps(obj).encode("utf-8")

    def unwrap(self, blob: bytes) -> bytes:
        obj = json.loads(blob.decode("utf-8"))
        meta = obj.get("meta", {})
        expected_fp = meta.get("collective_fp")
        current_fp = self._collective_fingerprint()
        if expected_fp is not None and expected_fp != current_fp:
            raise RuntimeError("Collective fingerprint mismatch: Queens changed or moved to another system")

        wrapped_master = bytes.fromhex(obj["wrapped_master"])
        c = bytes.fromhex(obj["ciphertext"])
        nonce, ct = c[:12], c[12:]

        for q in reversed(self.queens):
            wrapped_master = q.unlock_with_workers(wrapped_master)

        master_key = wrapped_master
        aes = AESGCM(master_key)
        return aes.decrypt(nonce, ct, b"")


# -----------------------------
# GUI overview (Tkinter)
# -----------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Borg Glyph Vault (Concept Demo)")

        self.codec = GlyphCodec()
        self.server = None
        self.client = None
        self.vault = Vault()
        self.encrypted_vault_blob = None
        self.wrapped_vault_blob = None

        self.queen1 = BorgQueen("alpha", workers=4)
        self.queen2 = BorgQueen("beta", workers=4)
        self.queen3 = BorgQueen("gamma", workers=4)
        self.collective = BorgCollectiveWrapper([self.queen1, self.queen2, self.queen3])

        frm_top = tk.Frame(root)
        frm_top.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(frm_top, text="Master password:").pack(side=tk.LEFT)
        self.entry_master = tk.Entry(frm_top, show="*")
        self.entry_master.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        tk.Button(frm_top, text="Set Master", command=self.set_master).pack(side=tk.LEFT, padx=5)

        frm_queens = tk.Frame(root)
        frm_queens.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_q1 = tk.Label(frm_queens, text="Queen alpha: ?", width=20)
        self.lbl_q1.pack(side=tk.LEFT, padx=5)
        self.lbl_q2 = tk.Label(frm_queens, text="Queen beta: ?", width=20)
        self.lbl_q2.pack(side=tk.LEFT, padx=5)
        self.lbl_q3 = tk.Label(frm_queens, text="Queen gamma: ?", width=20)
        self.lbl_q3.pack(side=tk.LEFT, padx=5)

        frm_mid = tk.Frame(root)
        frm_mid.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(frm_mid, text="Init Server", command=self.init_server).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid, text="Run Handshake", command=self.run_handshake).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid, text="Add Demo Entry", command=self.add_demo_entry).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid, text="Encrypt Vault", command=self.encrypt_vault).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid, text="Decrypt Vault", command=self.decrypt_vault).pack(side=tk.LEFT, padx=5)

        frm_mid2 = tk.Frame(root)
        frm_mid2.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(frm_mid2, text="Borg Lock Vault", command=self.borg_lock_vault).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid2, text="Borg Unlock Vault", command=self.borg_unlock_vault).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid2, text="Show Glyph Ciphertext", command=self.show_glyph_ciphertext).pack(side=tk.LEFT, padx=5)

        self.log = scrolledtext.ScrolledText(root, height=24)
        self.log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_msg("[INFO] Concept demo. Do NOT use for real secrets.")
        self.update_queen_status()

    def log_msg(self, msg: str):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

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
        if self.encrypted_vault_blob is None:
            messagebox.showwarning("Warning", "Encrypt vault first")
            return

        statuses = [self.queen1.status(), self.queen2.status(), self.queen3.status()]
        if any(s == "OFFLINE" for s in statuses):
            messagebox.showerror("Error", "All 3 Queens must be ONLINE to lock.")
            return

        meta = {
            "note": "borg_collective_lock_demo",
            "os": platform.system(),
            "host": platform.node(),
        }
        try:
            self.wrapped_vault_blob = self.collective.wrap(self.encrypted_vault_blob, meta)
        except Exception as e:
            messagebox.showerror("Error", f"Borg lock failed: {e}")
            return

        self.log_msg(f"[BORG] Vault locked by 3 Queens ({len(self.wrapped_vault_blob)} bytes).")

    def borg_unlock_vault(self):
        if self.wrapped_vault_blob is None:
            messagebox.showwarning("Warning", "No Borg-locked vault yet")
            return

        statuses = [self.queen1.status(), self.queen2.status(), self.queen3.status()]
        if any(s == "OFFLINE" for s in statuses):
            messagebox.showerror("Error", "All 3 Queens must be ONLINE to unlock.")
            return

        try:
            self.encrypted_vault_blob = self.collective.unwrap(self.wrapped_vault_blob)
        except Exception as e:
            messagebox.showerror("Error", f"Borg unlock failed: {e}")
            return

        self.log_msg("[BORG] Vault unlocked by 3 Queens. You can now decrypt vault with master password.")

    def show_glyph_ciphertext(self):
        if self.wrapped_vault_blob is None and self.encrypted_vault_blob is None:
            messagebox.showwarning("Warning", "No ciphertext yet")
            return
        blob = self.wrapped_vault_blob or self.encrypted_vault_blob
        glyphs = self.codec.encode(blob)
        self.log_msg("[GLYPH] Ciphertext as glyphs:")
        self.log_msg(glyphs[:800] + ("..." if len(glyphs) > 800 else ""))


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
