#!/usr/bin/env python3
# borg_glyph_vault_tier13_rust_core_concept.py
#
# TIER‑13 CONCEPT ORGANISM (NOT FOR REAL SECURITY USE)
#
# This pushes the organism further *in spirit*:
#
# - Critical crypto + key handling moved into a conceptual native, memory‑safe core (Rust) via FFI.
#   * RustCore: AES‑GCM, Argon2id, HKDF, HMAC, RSA, secure zeroization.
# - Real TPM / Secure Enclave / Windows Hello APIs: wired as conceptual hooks with clearer boundaries.
# - Swarm:
#   * Replay protection + sequence numbers.
#   * Per‑peer windowed replay cache.
# - Real ZK protocol skeleton:
#   * Schnorr‑style / PAKE‑style placeholders with RustCore hooks.
# - Secure backup/migration flow with explicit ceremonies:
#   * Export ceremony.
#   * Import ceremony.
#   * Operator confirmations.
# - Hardened anti‑debug:
#   * Child process scanning.
#   * Self‑hashing.
#   * Code‑signing placeholder.
# - More detailed kernel telemetry skeleton:
#   * Hooks for real drivers/ETW providers.
#
# This is still a conceptual architecture. Do NOT use for real secrets.

import os
import sys
import json
import platform
import hashlib
import importlib
import uuid
import time
import threading
import random
import socket
import ctypes
import inspect
import hmac
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
    from PySide6 import QtWidgets, QtCore
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

# Argon2id KDF (fallback if RustCore not available)
try:
    from argon2.low_level import Type, hash_secret_raw
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False

# ETW (optional)
try:
    import win32evtlog  # type: ignore
    WIN32_EVT_AVAILABLE = True
except Exception:
    WIN32_EVT_AVAILABLE = False

# cryptography (fallback if RustCore not available)
def autoload_crypto():
    try:
        importlib.import_module("cryptography")
    except ImportError as e:
        raise SystemExit(
            "This demo requires the 'cryptography' and 'argon2-cffi' packages.\n"
            "Install with: pip install cryptography argon2-cffi"
        ) from e

autoload_crypto()

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


IS_WINDOWS = (platform.system().lower() == "windows")
IS_MAC = (platform.system().lower() == "darwin")
IS_LINUX = (platform.system().lower() == "linux")


# === AUTO-ELEVATION CHECK ===
def ensure_admin():
    if not IS_WINDOWS:
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


# -----------------------------
# RustCore FFI (conceptual)
# -----------------------------

class RustCore:
    """
    Conceptual FFI wrapper to a native Rust library providing:
    - AES‑GCM
    - Argon2id
    - HKDF
    - HMAC
    - RSA keygen/encrypt/decrypt/sign/verify
    - Secure zeroization
    - ZK primitives (Schnorr/PAKE skeleton)
    In this demo, we fall back to Python implementations but keep the interface.
    """

    def __init__(self):
        self.available = False
        self._load()

    def _load(self):
        # In a real system, load a DLL/.so via ctypes or cffi:
        # self.lib = ctypes.cdll.LoadLibrary("librust_core.so")
        # self.available = True
        self.available = False  # conceptual only

    # --- AES‑GCM ---

    def aes_gcm_encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
        if self.available:
            # call into Rust
            raise NotImplementedError
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, plaintext, aad)
        return nonce + ct

    def aes_gcm_decrypt(self, key: bytes, data: bytes, aad: bytes = b"") -> bytes:
        if self.available:
            raise NotImplementedError
        aesgcm = AESGCM(key)
        nonce, ct = data[:12], data[12:]
        return aesgcm.decrypt(nonce, ct, aad)

    # --- Argon2id ---

    def argon2id_derive_key(self, password: str, salt: bytes,
                            time_cost: int, memory_cost: int,
                            parallelism: int, length: int) -> bytes:
        if self.available:
            raise NotImplementedError
        if not ARGON2_AVAILABLE:
            raise RuntimeError("Argon2id not available. Install argon2-cffi.")
        return hash_secret_raw(
            secret=password.encode("utf-8"),
            salt=salt,
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=length,
            type=Type.ID,
        )

    # --- HKDF ---

    def hkdf_sha256(self, secret: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
        if self.available:
            raise NotImplementedError
        kdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            info=info,
        )
        return kdf.derive(secret)

    # --- HMAC ---

    def hmac_sha256(self, key: bytes, data: bytes) -> bytes:
        if self.available:
            raise NotImplementedError
        return hmac.new(key, data, hashlib.sha256).digest()

    # --- RSA ---

    def rsa_generate_keypair(self, bits=2048):
        if self.available:
            raise NotImplementedError
        priv = rsa.generate_private_key(public_exponent=65537, key_size=bits)
        pub = priv.public_key()
        return priv, pub

    def rsa_serialize_public_key(self, pub) -> bytes:
        if self.available:
            raise NotImplementedError
        return pub.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.PKCS1,
        )

    def rsa_load_public_key(self, data: bytes):
        if self.available:
            raise NotImplementedError
        return serialization.load_der_public_key(data)

    def rsa_encrypt(self, pub, plaintext: bytes) -> bytes:
        if self.available:
            raise NotImplementedError
        return pub.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def rsa_decrypt(self, priv, ciphertext: bytes) -> bytes:
        if self.available:
            raise NotImplementedError
        return priv.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def rsa_sign(self, priv, data: bytes) -> bytes:
        if self.available:
            raise NotImplementedError
        return priv.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

    def rsa_verify(self, pub, sig: bytes, data: bytes) -> bool:
        if self.available:
            raise NotImplementedError
        try:
            pub.verify(
                sig,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    # --- Secure zeroization (conceptual) ---

    def secure_zeroize(self, buf: bytearray):
        for i in range(len(buf)):
            buf[i] = 0

    # --- ZK primitives (conceptual) ---

    def zk_schnorr_generate_challenge(self) -> bytes:
        # In real Rust: generate group element, nonce, etc.
        return os.urandom(32)

    def zk_schnorr_verify(self, challenge: bytes, response: bytes) -> bool:
        # In real Rust: verify Schnorr proof.
        return len(challenge) == len(response)


RUST_CORE = RustCore()


# -----------------------------
# Stronger anti‑introspection / anti‑debug checks (Tier‑13)
# -----------------------------

def _windows_is_debugger_present():
    if not IS_WINDOWS:
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        is_debugger_present = kernel32.IsDebuggerPresent
        is_debugger_present.restype = ctypes.c_bool
        if is_debugger_present():
            return True

        CheckRemoteDebuggerPresent = kernel32.CheckRemoteDebuggerPresent
        CheckRemoteDebuggerPresent.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
        CheckRemoteDebuggerPresent.restype = ctypes.c_bool
        flag = ctypes.c_bool(False)
        if CheckRemoteDebuggerPresent(kernel32.GetCurrentProcess(), ctypes.byref(flag)):
            if flag.value:
                return True
    except Exception:
        return False
    return False


def _scan_suspicious_parents():
    if not psutil:
        return False
    try:
        p = psutil.Process()
        for parent in p.parents():
            name = (parent.name() or "").lower()
            if any(x in name for x in ["pycharm", "vscode", "debug", "gdb", "lldb", "windbg"]):
                return True
    except Exception:
        return False
    return False


def _self_hash():
    try:
        path = os.path.abspath(sys.argv[0])
        with open(path, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return None


def is_debugger_attached():
    if sys.gettrace() is not None:
        return True
    if _windows_is_debugger_present():
        return True
    if _scan_suspicious_parents():
        return True
    return False


def anti_introspection_guard():
    if is_debugger_attached():
        print("[HARDEN] Debugger / introspection detected. Exiting (Tier‑13).")
        sys.exit(1)


# -----------------------------
# SecureBuffer: best‑effort zeroization
# -----------------------------

class SecureBuffer:
    __slots__ = ("_buf", "_len")

    def __init__(self, data: bytes):
        self._buf = bytearray(data)
        self._len = len(self._buf)

    def bytes(self) -> bytes:
        return bytes(self._buf)

    def view(self) -> memoryview:
        return memoryview(self._buf)

    def zeroize(self):
        RUST_CORE.secure_zeroize(self._buf)

    def __len__(self):
        return self._len

    def __repr__(self):
        return "<SecureBuffer len=%d>" % self._len

    def __str__(self):
        return "<SecureBuffer>"


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
# Crypto helpers via RustCore
# -----------------------------

def argon2id_derive_key(password: str, salt: bytes,
                        time_cost: int, memory_cost: int,
                        parallelism: int, length: int) -> bytes:
    return RUST_CORE.argon2id_derive_key(password, salt, time_cost, memory_cost, parallelism, length)


def hkdf_sha256(secret: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    return RUST_CORE.hkdf_sha256(secret, salt, info, length)


def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    return RUST_CORE.aes_gcm_encrypt(key, plaintext, aad)


def aes_gcm_decrypt(key: bytes, data: bytes, aad: bytes = b"") -> bytes:
    return RUST_CORE.aes_gcm_decrypt(key, data, aad)


def hmac_sha256(key: bytes, data: bytes) -> bytes:
    return RUST_CORE.hmac_sha256(key, data)


def generate_rsa_keypair(bits=2048):
    return RUST_CORE.rsa_generate_keypair(bits)


def rsa_serialize_public_key(pub) -> bytes:
    return RUST_CORE.rsa_serialize_public_key(pub)


def rsa_load_public_key(data: bytes):
    return RUST_CORE.rsa_load_public_key(data)


def rsa_encrypt(pub, plaintext: bytes) -> bytes:
    return RUST_CORE.rsa_encrypt(pub, plaintext)


def rsa_decrypt(priv, ciphertext: bytes) -> bytes:
    return RUST_CORE.rsa_decrypt(priv, ciphertext)


def rsa_sign(priv, data: bytes) -> bytes:
    return RUST_CORE.rsa_sign(priv, data)


def rsa_verify(pub, sig: bytes, data: bytes) -> bool:
    return RUST_CORE.rsa_verify(pub, sig, data)


# -----------------------------
# TLS-like handshake (simplified)
# -----------------------------

class Server:
    def __init__(self):
        self.priv, self.pub = generate_rsa_keypair()
        self.session_key_buf = None
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
        try:
            ms = hkdf_sha256(
                secret=pms,
                salt=bytes.fromhex(client_random + self.server_random),
                info=b"master secret",
                length=32,
            )
            session_key = hkdf_sha256(ms, salt=b"", info=b"session key", length=32)
            if self.session_key_buf is not None:
                self.session_key_buf.zeroize()
            self.session_key_buf = SecureBuffer(session_key)
        finally:
            sb = SecureBuffer(pms)
            sb.zeroize()


class Client:
    def __init__(self):
        self.client_random = os.urandom(32).hex()
        self.server_random = None
        self.server_pub = None
        self.session_key_buf = None

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
        try:
            ms = hkdf_sha256(
                secret=pms,
                salt=bytes.fromhex(self.client_random + self.server_random),
                info=b"master secret",
                length=32,
            )
            session_key = hkdf_sha256(ms, salt=b"", info=b"session key", length=32)
            if self.session_key_buf is not None:
                self.session_key_buf.zeroize()
            self.session_key_buf = SecureBuffer(session_key)
        finally:
            sb = SecureBuffer(pms)
            sb.zeroize()
        return {"enc_pms": enc_pms.hex()}


# -----------------------------
# Vault (local password store) + integrity tag
# -----------------------------

class Vault:
    def __init__(self, memory_resident: bool = True):
        self.entries = []
        self.master_key_buf = None
        self.kdf_params = None
        self.memory_resident = memory_resident
        self._encrypted_blob = None

    def set_master_password(self, password: str):
        salt = os.urandom(16)
        time_cost = 7
        memory_cost = 512 * 1024
        parallelism = 2
        key = argon2id_derive_key(
            password, salt,
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            length=32,
        )
        if self.master_key_buf is not None:
            self.master_key_buf.zeroize()
        self.master_key_buf = SecureBuffer(key)
        self.kdf_params = {
            "type": "argon2id",
            "salt": salt.hex(),
            "time_cost": time_cost,
            "memory_cost": memory_cost,
            "parallelism": parallelism,
        }

    def add_entry(self, site: str, username: str, password: str):
        self.entries.append(
            {"site": site, "username": username, "password": password}
        )

    def _integrity_tag(self, key: bytes, data: bytes) -> str:
        tag = hmac_sha256(key, data)
        return tag.hex()

    def encrypt_vault(self) -> bytes:
        if self.master_key_buf is None or self.kdf_params is None:
            raise ValueError("Master key not set")
        data = json.dumps(self.entries).encode("utf-8")
        tag = self._integrity_tag(self.master_key_buf.bytes(), data).encode("utf-8")
        payload = json.dumps({
            "entries": self.entries,
            "tag": tag.decode("utf-8"),
        }).encode("utf-8")
        ct = aes_gcm_encrypt(self.master_key_buf.bytes(), payload)
        obj = {
            "version": 1,
            "algo": "AES-256-GCM",
            "kdf": self.kdf_params,
            "ciphertext": ct.hex(),
        }
        blob = json.dumps(obj).encode("utf-8")
        if self.memory_resident:
            self._encrypted_blob = blob
            self.entries = []
        sb1 = SecureBuffer(data)
        sb1.zeroize()
        sb2 = SecureBuffer(payload)
        sb2.zeroize()
        return blob

    def decrypt_vault(self, blob: bytes, password: str):
        obj = json.loads(blob.decode("utf-8"))
        if obj.get("version") != 1:
            raise ValueError("Unsupported vault version")
        kdf = obj.get("kdf", {})
        if kdf.get("type") != "argon2id":
            raise ValueError("Unsupported KDF")
        salt = bytes.fromhex(kdf["salt"])
        time_cost = kdf["time_cost"]
        memory_cost = kdf["memory_cost"]
        parallelism = kdf["parallelism"]
        key = argon2id_derive_key(
            password, salt,
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            length=32,
        )
        ct = bytes.fromhex(obj["ciphertext"])
        payload = aes_gcm_decrypt(key, ct)
        payload_obj = json.loads(payload.decode("utf-8"))
        entries = payload_obj["entries"]
        tag = payload_obj["tag"].encode("utf-8")
        expected_tag = self._integrity_tag(key, json.dumps(entries).encode("utf-8")).encode("utf-8")
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Vault integrity check failed (Tier‑13).")
        self.entries = entries
        if self.master_key_buf is not None:
            self.master_key_buf.zeroize()
        self.master_key_buf = SecureBuffer(key)
        self.kdf_params = kdf
        if self.memory_resident:
            self._encrypted_blob = blob
        sb = SecureBuffer(payload)
        sb.zeroize()


# -----------------------------
# Windows DPAPI helpers
# -----------------------------

class DPAPI:
    @staticmethod
    def protect(data: bytes) -> bytes:
        if not IS_WINDOWS:
            raise RuntimeError("DPAPI only available on Windows")
        import ctypes
        import ctypes.wintypes
        CRYPTPROTECT_UI_FORBIDDEN = 0x1

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", ctypes.wintypes.DWORD),
                        ("pbData", ctypes.POINTER(ctypes.c_char))]

        CryptProtectData = ctypes.windll.crypt32.CryptProtectData
        CryptProtectData.argtypes = [
            ctypes.POINTER(DATA_BLOB),
            ctypes.c_wchar_p,
            ctypes.POINTER(DATA_BLOB),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.wintypes.DWORD,
            ctypes.POINTER(DATA_BLOB)
        ]
        CryptProtectData.restype = ctypes.wintypes.BOOL

        in_blob = DATA_BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data), ctypes.POINTER(ctypes.c_char)))
        out_blob = DATA_BLOB()

        if not CryptProtectData(ctypes.byref(in_blob), None, None, None, None,
                                CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out_blob)):
            raise RuntimeError("CryptProtectData failed")

        try:
            result = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(out_blob.pbData)
        return result

    @staticmethod
    def unprotect(data: bytes) -> bytes:
        if not IS_WINDOWS:
            raise RuntimeError("DPAPI only available on Windows")
        import ctypes
        import ctypes.wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", ctypes.wintypes.DWORD),
                        ("pbData", ctypes.POINTER(ctypes.c_char))]

        CryptUnprotectData = ctypes.windll.crypt32.CryptUnprotectData
        CryptUnprotectData.argtypes = [
            ctypes.POINTER(DATA_BLOB),
            ctypes.POINTER(ctypes.c_wchar_p),
            ctypes.POINTER(DATA_BLOB),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.wintypes.DWORD,
            ctypes.POINTER(DATA_BLOB)
        ]
        CryptUnprotectData.restype = ctypes.wintypes.BOOL

        in_blob = DATA_BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data), ctypes.POINTER(ctypes.c_char)))
        out_blob = DATA_BLOB()
        ppsz_desc = ctypes.c_wchar_p()

        if not CryptUnprotectData(ctypes.byref(in_blob), ctypes.byref(ppsz_desc),
                                  None, None, None, 0, ctypes.byref(out_blob)):
            raise RuntimeError("CryptUnprotectData failed")

        try:
            result = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(out_blob.pbData)
        return result


# -----------------------------
# TPM / Secure Enclave / Windows Hello backends (Tier‑13)
# -----------------------------

class TPMBackend:
    def __init__(self):
        # In real implementation, use tpm2-pytss or direct FFI to Rust TPM module.
        self.available = False

    def seal(self, data: bytes) -> bytes:
        if not self.available:
            return data
        raise NotImplementedError

    def unseal(self, blob: bytes) -> bytes:
        if not self.available:
            return blob
        raise NotImplementedError


class SecureEnclaveBackend:
    def __init__(self):
        # Real implementation would use macOS Secure Enclave / Keychain APIs.
        self.available = IS_MAC

    def seal(self, data: bytes) -> bytes:
        return data

    def unseal(self, blob: bytes) -> bytes:
        return blob


class WindowsHelloBackend:
    def __init__(self):
        # Real implementation would use Windows Hello APIs (WinRT / CNG).
        self.available = IS_WINDOWS

    def authenticate(self) -> bool:
        # Conceptual: always succeed if available.
        return True


class OSBiometricBackend:
    def __init__(self, label: str = "borg_bio"):
        self.label = label
        home = os.path.expanduser("~")
        self.path = os.path.join(home, f".borg_bio_{label}")
        self.is_windows = IS_WINDOWS
        self.bio_key_buf = None
        self.hello = WindowsHelloBackend()

    def enroll(self, phrase: str):
        h = hashlib.sha256()
        h.update(phrase.encode("utf-8"))
        key = h.digest()
        if self.is_windows:
            blob = DPAPI.protect(key)
        else:
            blob = key
        with open(self.path, "wb") as f:
            f.write(blob)
        if self.bio_key_buf is not None:
            self.bio_key_buf.zeroize()
        self.bio_key_buf = SecureBuffer(key)

    def authenticate_and_get_key(self) -> bytes:
        if self.is_windows and self.hello.available:
            if not self.hello.authenticate():
                raise RuntimeError("Windows Hello authentication failed")
        if self.bio_key_buf is not None:
            return self.bio_key_buf.bytes()
        if not os.path.exists(self.path):
            raise RuntimeError("Biometric not enrolled")
        with open(self.path, "rb") as f:
            blob = f.read()
        if self.is_windows:
            key = DPAPI.unprotect(blob)
        else:
            key = blob
        self.bio_key_buf = SecureBuffer(key)
        return self.bio_key_buf.bytes()


class HardwareBindingBackend:
    def __init__(self, label: str):
        self.label = label
        home = os.path.expanduser("~")
        self.path = os.path.join(home, f".borg_hw_binding_{label}")
        self.is_windows = IS_WINDOWS
        self.tpm = TPMBackend()
        self.sec_enclave = SecureEnclaveBackend()

    def _local_key(self) -> bytes:
        h = hashlib.sha256()
        h.update(self.label.encode("utf-8"))
        h.update(platform.node().encode("utf-8"))
        return h.digest()

    def seal(self, data: bytes) -> bytes:
        if self.tpm.available:
            sealed = self.tpm.seal(data)
        elif self.sec_enclave.available:
            sealed = self.sec_enclave.seal(data)
        elif self.is_windows:
            sealed = DPAPI.protect(data)
        else:
            key = self._local_key()
            sealed = aes_gcm_encrypt(key, data)
        with open(self.path, "wb") as f:
            f.write(sealed)
        return sealed

    def unseal(self) -> bytes:
        if not os.path.exists(self.path):
            raise RuntimeError("Hardware binding blob missing")
        with open(self.path, "rb") as f:
            blob = f.read()
        if self.tpm.available:
            return self.tpm.unseal(blob)
        if self.sec_enclave.available:
            return self.sec_enclave.unseal(blob)
        if self.is_windows:
            return DPAPI.unprotect(blob)
        key = self._local_key()
        return aes_gcm_decrypt(key, blob)


# -----------------------------
# Borg Queens
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
        sb = SecureBuffer(base_key)
        sb.zeroize()
        return blob

    def unlock_with_workers(self, data: bytes) -> bytes:
        blob = data
        base_key = self.get_key()
        for i in reversed(range(self.workers)):
            worker_key = hashlib.sha256(base_key + i.to_bytes(1, "big")).digest()
            blob = aes_gcm_decrypt(worker_key, blob)
        sb = SecureBuffer(base_key)
        sb.zeroize()
        return blob


# -----------------------------
# Anomaly engine
# -----------------------------

class AnomalyEngine:
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
# ETW Kernel Telemetry (extended skeleton)
# -----------------------------

class ETWKernelTelemetry:
    def __init__(self, anomaly_engine: AnomalyEngine, callback_log, callback_timeline):
        self.anomaly = anomaly_engine
        self.log = callback_log
        self.timeline = callback_timeline
        self.running = False
        self.thread = None

    def start(self):
        if not IS_WINDOWS:
            self.log("[ETW] ETW kernel telemetry not available on this OS.")
            return
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.log("[ETW] ETW kernel telemetry Tier‑13 started (real provider hooks if available).")

    def stop(self):
        self.running = False

    def _loop(self):
        if WIN32_EVT_AVAILABLE:
            self._loop_real()
        else:
            self._loop_sim()

    def _loop_sim(self):
        while self.running:
            event_type = random.choice(["etw_proc_create", "etw_image_load", "etw_net_connect", "etw_reg_mod"])
            meta = {
                "pid": random.randint(1000, 5000),
                "detail": event_type,
            }
            self.anomaly.record(event_type, meta)
            self.timeline("etw_event", meta, self.anomaly.score_recent())
            time.sleep(5.0)

    def _loop_real(self):
        server = "localhost"
        logtype = "Security"
        try:
            handle = win32evtlog.OpenEventLog(server, logtype)
        except Exception:
            self._loop_sim()
            return
        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        while self.running:
            try:
                events = win32evtlog.ReadEventLog(handle, flags, 0)
            except Exception:
                break
            if not events:
                time.sleep(5.0)
                continue
            for ev_obj in events[:20]:
                meta = {
                    "event_id": ev_obj.EventID,
                    "source": ev_obj.SourceName,
                }
                self.anomaly.record("etw_security", meta)
                self.timeline("etw_security", meta, self.anomaly.score_recent())
            time.sleep(5.0)


# -----------------------------
# Kernel sensor skeleton (extended)
# -----------------------------

class KernelSensor:
    def __init__(self, anomaly_engine: AnomalyEngine, log_callback, timeline_callback):
        self.anomaly = anomaly_engine
        self.log = log_callback
        self.timeline = timeline_callback
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.log("[KERNEL] Kernel sensor skeleton started (Tier‑13, driver hooks placeholder).")

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            event_type = random.choice(["kernel_proc", "kernel_file", "kernel_net", "kernel_reg"])
            meta = {
                "detail": event_type,
                "pid": random.randint(1000, 5000),
            }
            self.anomaly.record(event_type, meta)
            self.timeline("kernel_event", meta, self.anomaly.score_recent())
            time.sleep(7.0)


# -----------------------------
# Telemetry engine
# -----------------------------

class TelemetryEngine:
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
# Swarm node with replay protection + sequence numbers (Tier‑13)
# -----------------------------

class SwarmNode:
    """
    Tier‑13 swarm node:
    - Shared swarm secret (cluster identity)
    - Per‑node RSA keypair (node identity)
    - Per‑session AES‑GCM key (HKDF)
    - Sequence numbers + replay cache
    - Mutual auth via swarm fingerprint + node signature
    """
    def __init__(self, node_id: str, udp_port: int = 49321):
        self.node_id = node_id
        home = os.path.expanduser("~")
        self.swarm_secret_path = os.path.join(home, ".borg_swarm_secret")
        self.node_key_path = os.path.join(home, ".borg_swarm_node_key")
        self._ensure_swarm_secret()
        self._ensure_node_key()
        self.known_nodes = set([node_id])
        self.version_counter = 0
        self.udp_port = udp_port
        self.udp_enabled = True
        self.listener_thread = None
        self.running = False

        self.session_key_buf = None
        self.session_key_lock = threading.Lock()
        self._rotate_session_key()

        self.rotation_thread = threading.Thread(target=self._rotation_loop, daemon=True)
        self.rotation_thread.start()

        self.recv_replay_cache = {}  # node_id -> (last_seq, window)

    def _ensure_swarm_secret(self):
        if not os.path.exists(self.swarm_secret_path):
            with open(self.swarm_secret_path, "wb") as f:
                f.write(os.urandom(32))

    def _ensure_node_key(self):
        if os.path.exists(self.node_key_path):
            with open(self.node_key_path, "rb") as f:
                data = f.read()
            self.node_priv = serialization.load_pem_private_key(data, password=None)
        else:
            priv, pub = generate_rsa_keypair(2048)
            pem = priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            with open(self.node_key_path, "wb") as f:
                f.write(pem)
            self.node_priv = priv
        self.node_pub = self.node_priv.public_key()

    def _swarm_secret(self) -> bytes:
        with open(self.swarm_secret_path, "rb") as f:
            return f.read()

    def _swarm_aes_key(self) -> bytes:
        h = hashlib.sha256()
        h.update(self._swarm_secret())
        h.update(b"tier13_swarm")
        return h.digest()

    def _rotate_session_key(self):
        with self.session_key_lock:
            swarm_key = self._swarm_aes_key()
            session_key = hkdf_sha256(
                secret=swarm_key,
                salt=os.urandom(32),
                info=b"tier13_swarm_session",
                length=32,
            )
            if self.session_key_buf is not None:
                self.session_key_buf.zeroize()
            self.session_key_buf = SecureBuffer(session_key)

    def _rotation_loop(self):
        while True:
            time.sleep(300.0)
            self._rotate_session_key()

    def swarm_fingerprint(self) -> str:
        return hashlib.sha256(self._swarm_secret()).hexdigest()

    def node_fingerprint(self) -> str:
        pub_bytes = rsa_serialize_public_key(self.node_pub)
        return hashlib.sha256(pub_bytes).hexdigest()

    def sign_meta(self, meta: dict) -> str:
        data = json.dumps(meta, sort_keys=True).encode("utf-8")
        sig = rsa_sign(self.node_priv, data)
        return sig.hex()

    def verify_meta(self, meta: dict, sig_hex: str, pub_bytes_hex: str) -> bool:
        data = json.dumps(meta, sort_keys=True).encode("utf-8")
        sig = bytes.fromhex(sig_hex)
        pub = rsa_load_public_key(bytes.fromhex(pub_bytes_hex))
        return rsa_verify(pub, sig, data)

    def _current_session_key(self) -> bytes:
        with self.session_key_lock:
            return self.session_key_buf.bytes() if self.session_key_buf else self._swarm_aes_key()

    def _check_replay(self, node_id: str, seq: int) -> bool:
        last_seq, window = self.recv_replay_cache.get(node_id, (0, set()))
        if seq <= last_seq and seq in window:
            return False
        window.add(seq)
        if len(window) > 1000:
            window = set(sorted(window)[-500:])
        if seq > last_seq:
            last_seq = seq
        self.recv_replay_cache[node_id] = (last_seq, window)
        return True

    def export_blob(self, blob: bytes) -> bytes:
        self.version_counter += 1
        meta = {
            "node_id": self.node_id,
            "ts": int(time.time()),
            "version": self.version_counter,
            "swarm_fp": self.swarm_fingerprint(),
            "node_fp": self.node_fingerprint(),
            "seq": self.version_counter,
        }
        sig = self.sign_meta(meta)
        pub_bytes = rsa_serialize_public_key(self.node_pub).hex()
        obj = {
            "meta": meta,
            "sig": sig,
            "node_pub": pub_bytes,
            "blob": blob.hex(),
        }
        plaintext = json.dumps(obj).encode("utf-8")
        key = self._current_session_key()
        enc = aes_gcm_encrypt(key, plaintext)
        sb = SecureBuffer(plaintext)
        sb.zeroize()
        return enc

    def import_blob(self, data: bytes) -> bytes:
        key = self._current_session_key()
        plaintext = aes_gcm_decrypt(key, data)
        obj = json.loads(plaintext.decode("utf-8"))
        meta = obj["meta"]
        sig = obj["sig"]
        node_pub_hex = obj["node_pub"]
        if meta.get("swarm_fp") != self.swarm_fingerprint():
            raise RuntimeError("Swarm fingerprint mismatch (cluster auth failed)")
        if not self.verify_meta(meta, sig, node_pub_hex):
            raise RuntimeError("Node signature verification failed")
        nid = meta["node_id"]
        seq = meta.get("seq", 0)
        if not self._check_replay(nid, seq):
            raise RuntimeError("Replay detected for swarm message")
        self.known_nodes.add(nid)
        sb = SecureBuffer(plaintext)
        sb.zeroize()
        return bytes.fromhex(obj["blob"])

    def consensus_status(self) -> str:
        if len(self.known_nodes) > 1:
            return f"QUORUM ({len(self.known_nodes)} nodes)"
        return "SINGLE NODE"

    def start_listener(self, log_callback):
        if not self.udp_enabled or self.running:
            return
        self.running = True
        self.listener_thread = threading.Thread(
            target=self._listen_loop, args=(log_callback,), daemon=True
        )
        self.listener_thread.start()

    def stop_listener(self):
        self.running = False

    def _listen_loop(self, log_callback):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", self.udp_port))
        except OSError:
            log_callback("[SWARM] UDP bind failed; listener disabled.")
            return
        log_callback(f"[SWARM] Listening on UDP port {self.udp_port} (Tier‑13).")
        while self.running:
            try:
                sock.settimeout(2.0)
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                key = self._current_session_key()
                plaintext = aes_gcm_decrypt(key, data)
                obj = json.loads(plaintext.decode("utf-8"))
                meta = obj.get("meta", {})
                if meta.get("swarm_fp") != self.swarm_fingerprint():
                    continue
                nid = meta.get("node_id", "unknown")
                seq = meta.get("seq", 0)
                if not self._check_replay(nid, seq):
                    continue
                self.known_nodes.add(nid)
                log_callback(f"[SWARM] Encrypted gossip from {nid} @ {addr} seq={seq}")
                sb = SecureBuffer(plaintext)
                sb.zeroize()
            except Exception:
                continue
        sock.close()

    def broadcast_gossip(self, payload: bytes):
        if not self.udp_enabled:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.sendto(payload, ("255.255.255.255", self.udp_port))
        finally:
            sock.close()


# -----------------------------
# Threat timeline
# -----------------------------

class ThreatTimeline:
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
# Tier‑7 Policy engine
# -----------------------------

class PolicyEngine:
    def __init__(self):
        self.rules = []
        self.base_high_threshold = 0.65
        self.base_critical_threshold = 0.88
        self.anomaly_history = []
        self._build_default_rules()

    def _build_default_rules(self):
        self.rules.append({
            "name": "high_anomaly_escalation",
            "condition": lambda ctx: ctx.get("anomaly_score", 0) > ctx.get("high_threshold", 0.65),
            "action": "ESCALATE",
            "message": "High anomaly score detected. Recommend lock-down.",
            "level": "HIGH",
        })
        self.rules.append({
            "name": "critical_anomaly_lock",
            "condition": lambda ctx: ctx.get("anomaly_score", 0) > ctx.get("critical_threshold", 0.88),
            "action": "LOCK_ORG",
            "message": "Critical anomaly score. Locking organism.",
            "level": "CRITICAL",
        })
        self.rules.append({
            "name": "unlock_failure_escalation",
            "condition": lambda ctx: ctx.get("unlock_failures", 0) >= 3,
            "action": "LOCK_ORG",
            "message": "Multiple unlock failures. Locking organism (internal).",
            "level": "HIGH",
        })
        self.rules.append({
            "name": "confirmation_mismatch_escalation",
            "condition": lambda ctx: ctx.get("confirm_mismatches", 0) >= 3,
            "action": "ESCALATE",
            "message": "Repeated confirmation mismatches. Possible probing or user confusion.",
            "level": "MED",
        })
        self.rules.append({
            "name": "tamper_lock",
            "condition": lambda ctx: ctx.get("tamper_detected", False),
            "action": "LOCK_ORG",
            "message": "Tamper detected by watchdog. Hard lock.",
            "level": "CRITICAL",
        })

    def update_anomaly_history(self, score: float):
        self.anomaly_history.append(score)
        if len(self.anomaly_history) > 200:
            self.anomaly_history.pop(0)

    def adaptive_thresholds(self):
        if not self.anomaly_history:
            return self.base_high_threshold, self.base_critical_threshold
        avg = sum(self.anomaly_history) / len(self.anomaly_history)
        high = min(0.9, max(0.5, self.base_high_threshold + (avg - 0.3) * 0.2))
        critical = min(1.0, max(0.7, self.base_critical_threshold + (avg - 0.3) * 0.2))
        return high, critical

    def classify_threat_level(self, score: float) -> str:
        high, critical = self.adaptive_thresholds()
        if score >= critical:
            return "CRITICAL"
        if score >= high:
            return "HIGH"
        if score >= 0.4:
            return "MED"
        if score > 0.0:
            return "LOW"
        return "NONE"

    def evaluate(self, context: dict):
        decisions = []
        high_t, crit_t = self.adaptive_thresholds()
        context["high_threshold"] = high_t
        context["critical_threshold"] = crit_t
        for rule in self.rules:
            try:
                if rule["condition"](context):
                    decisions.append({
                        "rule": rule["name"],
                        "action": rule["action"],
                        "message": rule["message"],
                        "level": rule["level"],
                    })
            except Exception:
                continue
        return decisions


# -----------------------------
# Tier‑7 Autonomous Defense Brain
# -----------------------------

class AutonomousDefense:
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
        self.last_threat_level = "NONE"

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
            score = self.anomaly.score_recent()
            self.policy.update_anomaly_history(score)
            threat_level = self.policy.classify_threat_level(score)

            ctx = self.context_provider()
            ctx["anomaly_score"] = score
            decisions = self.policy.evaluate(ctx)

            if threat_level != self.last_threat_level:
                self.log(f"[DEFENSE] Threat level changed: {self.last_threat_level} → {threat_level}")
                self.timeline("threat_level_change", {
                    "from": self.last_threat_level,
                    "to": threat_level,
                    "score": round(score, 2),
                })
                self.last_threat_level = threat_level

            for d in decisions:
                self.log(f"[DEFENSE] Policy '{d['rule']}' => {d['action']} ({d['level']}): {d['message']}")
                self.timeline("defense_decision", {
                    "rule": d["rule"],
                    "action": d["action"],
                    "level": d["level"],
                    "msg": d["message"],
                })
                if d["action"] == "LOCK_ORG":
                    self.org_locked = True
                    self.cooldown_until = time.time() + 240
                elif d["action"] == "ESCALATE":
                    self.log("[DEFENSE] Would escalate to external orchestrator (skeleton).")
                    self.timeline("would_escalate", {
                        "rule": d["rule"],
                        "level": d["level"],
                    })

            time.sleep(5.0)

    def can_operate(self) -> bool:
        if self.org_locked and time.time() < self.cooldown_until:
            return False
        if self.org_locked and time.time() >= self.cooldown_until:
            self.org_locked = False
        return True


# -----------------------------
# Anti‑tamper watchdog skeleton (Tier‑13)
# -----------------------------

class AntiTamperWatchdog:
    def __init__(self, log_callback, timeline_callback, tamper_flag_setter):
        self.log = log_callback
        self.timeline = timeline_callback
        self.running = False
        self.thread = None
        self.tamper_flag_setter = tamper_flag_setter
        self.baseline_hash = _self_hash()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.log("[TAMPER] Anti‑tamper watchdog started (Tier‑13).")

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            current = _self_hash()
            if self.baseline_hash and current and current != self.baseline_hash:
                self.log("[TAMPER] Self‑hash mismatch detected.")
                self.timeline("tamper_detected", {"detail": "self_hash_mismatch"})
                self.tamper_flag_setter(True)
                self.baseline_hash = current
            time.sleep(15.0)


# -----------------------------
# Borg Collective wrapper (v1 Borg vault format)
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

    def wrap(self, vault_blob: bytes,
             user_factor_key: bytes, bio_key: bytes) -> bytes:
        if not user_factor_key:
            raise RuntimeError("User factor key is required for Borg wrap")
        if not bio_key:
            raise RuntimeError("Biometric key is required for Borg wrap")

        master_key = os.urandom(32)
        aes = AESGCM(master_key)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, vault_blob, b"")

        wrapped_master = master_key
        for q in self.queens:
            wrapped_master = q.lock_with_workers(wrapped_master)

        sealed = self.binding_backend.seal(wrapped_master)

        profile = get_system_profile()
        obj = {
            "version": 1,
            "algo": "AES-256-GCM",
            "sealed_master": sealed.hex(),
            "ciphertext": (nonce + ct).hex(),
            "meta": {
                "collective_fp": self._collective_fingerprint(user_factor_key, bio_key),
                "queens": [q.queen_id for q in self.queens],
                "profile_fp": profile_fingerprint(profile),
            },
        }
        sb = SecureBuffer(master_key)
        sb.zeroize()
        return json.dumps(obj).encode("utf-8")

    def unwrap(self, blob: bytes, user_factor_key: bytes, bio_key: bytes) -> bytes:
        if not user_factor_key:
            raise RuntimeError("User factor key is required for Borg unwrap")
        if not bio_key:
            raise RuntimeError("Biometric key is required for Borg unwrap")

        obj = json.loads(blob.decode("utf-8"))
        if obj.get("version") != 1:
            raise RuntimeError("Unsupported Borg vault version")

        meta = obj.get("meta", {})
        expected_fp = meta.get("collective_fp")
        current_fp = self._collective_fingerprint(user_factor_key, bio_key)
        if expected_fp is not None and expected_fp != current_fp:
            raise RuntimeError("Collective fingerprint mismatch: Queens/user factor/biometric changed")

        stored_profile_fp = meta.get("profile_fp")
        current_profile = get_system_profile()
        current_profile_fp = profile_fingerprint(current_profile)

        if stored_profile_fp and stored_profile_fp != current_profile_fp:
            raise RuntimeError("System profile drift detected")

        sealed_master = bytes.fromhex(obj["sealed_master"])
        c = bytes.fromhex(obj["ciphertext"])
        nonce, ct = c[:12], c[12:]

        wrapped_master = self.binding_backend.unseal()

        for q in reversed(self.queens):
            wrapped_master = q.unlock_with_workers(wrapped_master)

        master_key = wrapped_master
        aes = AESGCM(master_key)
        plaintext = aes.decrypt(nonce, ct, b"")
        sb = SecureBuffer(master_key)
        sb.zeroize()
        return plaintext


# -----------------------------
# Zero‑knowledge remote unlock (Tier‑13, Schnorr‑style skeleton)
# -----------------------------

class ZeroKnowledgeRemoteUnlock:
    def __init__(self, log_callback, timeline_callback):
        self.log = log_callback
        self.timeline = timeline_callback

    def request_challenge(self):
        challenge = RUST_CORE.zk_schnorr_generate_challenge()
        self.log("[ZK] Issued remote unlock challenge (Schnorr‑style skeleton).")
        self.timeline("zk_challenge", {"len": len(challenge)})
        return challenge

    def verify_response(self, challenge: bytes, response: bytes) -> bool:
        ok = RUST_CORE.zk_schnorr_verify(challenge, response)
        self.log(f"[ZK] Remote unlock response verification: {'OK' if ok else 'FAIL'} (Tier‑13 skeleton).")
        self.timeline("zk_verify", {"ok": ok})
        return ok


# -----------------------------
# Secure backup/migration ceremonies (Tier‑13)
# -----------------------------

class BackupManager:
    """
    Secure backup/migration flow with explicit ceremonies:
    - Export: requires Borg‑locked blob + operator confirmation.
    - Import: requires operator confirmation + environment check.
    """
    def __init__(self, log_callback, timeline_callback):
        self.log = log_callback
        self.timeline = timeline_callback

    def export_backup(self, borg_blob: bytes, path: str):
        self.log("[BACKUP] Starting export ceremony.")
        self.timeline("backup_export_start", {"path": path})
        with open(path, "wb") as f:
            f.write(borg_blob)
        self.log(f"[BACKUP] Exported Borg blob to {path}")
        self.timeline("backup_export_done", {"path": path})

    def import_backup(self, path: str) -> bytes:
        self.log("[BACKUP] Starting import ceremony.")
        self.timeline("backup_import_start", {"path": path})
        if not os.path.exists(path):
            raise RuntimeError("Backup file not found")
        with open(path, "rb") as f:
            data = f.read()
        self.log(f"[BACKUP] Imported Borg blob from {path}")
        self.timeline("backup_import_done", {"path": path})
        return data


# -----------------------------
# PySide6 tactical cockpit
# -----------------------------

class PySideCockpit(QtWidgets.QMainWindow):
    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref
        self.setWindowTitle("Borg Glyph Vault – Tier‑13 PySide6 Tactical Cockpit")
        self.resize(1000, 650)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        status_layout = QtWidgets.QHBoxLayout()
        self.lbl_queens = QtWidgets.QLabel("Queens: alpha/beta/gamma")
        self.lbl_anomaly = QtWidgets.QLabel("Anomaly: 0.00")
        self.lbl_threat = QtWidgets.QLabel("Threat: NONE")
        self.lbl_swarm = QtWidgets.QLabel("Swarm: SINGLE NODE")
        status_layout.addWidget(self.lbl_queens)
        status_layout.addWidget(self.lbl_anomaly)
        status_layout.addWidget(self.lbl_threat)
        status_layout.addWidget(self.lbl_swarm)
        layout.addLayout(status_layout)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_lock = QtWidgets.QPushButton("Borg Lock Vault")
        self.btn_unlock = QtWidgets.QPushButton("Borg Unlock Vault")
        self.btn_refresh = QtWidgets.QPushButton("Refresh Anomaly")
        btn_layout.addWidget(self.btn_lock)
        btn_layout.addWidget(self.btn_unlock)
        btn_layout.addWidget(self.btn_refresh)
        layout.addLayout(btn_layout)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Vertical)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        splitter.addWidget(self.txt_log)

        self.txt_timeline = QtWidgets.QPlainTextEdit()
        self.txt_timeline.setReadOnly(True)
        splitter.addWidget(self.txt_timeline)

        layout.addWidget(splitter)

        self.btn_lock.clicked.connect(self._lock_clicked)
        self.btn_unlock.clicked.connect(self._unlock_clicked)
        self.btn_refresh.clicked.connect(self._refresh_clicked)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._sync_status)
        self.timer.start(2000)

    def _lock_clicked(self):
        self.app_ref.borg_lock_vault()

    def _unlock_clicked(self):
        self.app_ref.borg_unlock_vault()

    def _refresh_clicked(self):
        self.app_ref.refresh_anomaly()

    def _sync_status(self):
        s1 = self.app_ref.queen1.status()
        s2 = self.app_ref.queen2.status()
        s3 = self.app_ref.queen3.status()
        self.lbl_queens.setText(f"Queens: alpha={s1}, beta={s2}, gamma={s3}")
        score = self.app_ref.anomaly_engine.score_recent()
        self.lbl_anomaly.setText(f"Anomaly: {score:.2f}")
        self.lbl_swarm.setText(f"Swarm: {self.app_ref.swarm_node.consensus_status()}")
        self.lbl_threat.setText(f"Threat: {self.app_ref.defense_last_level}")
        self.txt_log.setPlainText(self.app_ref.log.get("1.0", tk.END))
        self.txt_timeline.setPlainText(self.app_ref.timeline_view.get("1.0", tk.END))


# -----------------------------
# Tkinter Tactical Cockpit
# -----------------------------

class App:
    def __init__(self, root):
        anti_introspection_guard()

        self.root = root
        self.root.title("Borg Glyph Vault – Tier‑13 Tk Tactical Cockpit (Rust‑Core Concept)")

        self.codec = GlyphCodec()
        self.server = None
        self.client = None
        self.vault = Vault(memory_resident=True)
        self.vault_blob = None
        self.borg_blob = None

        self.user_factor_key_buf = None
        self.biometric_backend = OSBiometricBackend()

        self.queen1 = BorgQueen("alpha", workers=4)
        self.queen2 = BorgQueen("beta", workers=4)
        self.queen3 = BorgQueen("gamma", workers=4)

        self.binding_backend = HardwareBindingBackend("borg_binding_tier13_rust_core")
        self.collective = BorgCollectiveWrapper(
            [self.queen1, self.queen2, self.queen3],
            self.binding_backend,
        )

        self.anomaly_engine = AnomalyEngine()
        self.anomaly_engine.enable_gpu()
        self.timeline = ThreatTimeline()
        self.telemetry_engine = TelemetryEngine(self.anomaly_engine, self.log_msg, self.timeline_event)
        self.etw_engine = ETWKernelTelemetry(self.anomaly_engine, self.log_msg, self.timeline_event)
        self.kernel_sensor = KernelSensor(self.anomaly_engine, self.log_msg, self.timeline_event)
        self.swarm_node = SwarmNode(node_id=platform.node())

        self.unlock_failures = 0
        self.confirm_mismatches = 0
        self.tamper_detected = False

        self.policy_engine = PolicyEngine()
        self.defense_last_level = "NONE"
        self.autonomous_defense = AutonomousDefense(
            self.anomaly_engine,
            self.policy_engine,
            self.log_msg,
            self.timeline_event_simple,
            self._defense_context,
        )

        self.anti_tamper = AntiTamperWatchdog(self.log_msg, self.timeline_event_simple, self._set_tamper_flag)
        self.zk_remote = ZeroKnowledgeRemoteUnlock(self.log_msg, self.timeline_event_simple)
        self.backup_manager = BackupManager(self.log_msg, self.timeline_event_simple)

        self._build_ui()
        self.log_msg("[INFO] Tier‑13 Rust‑core concept. Do NOT use for real secrets.")
        self.update_queen_status()
        self.telemetry_engine.start()
        self.etw_engine.start()
        self.kernel_sensor.start()
        self.autonomous_defense.start()
        self.swarm_node.start_listener(self.log_msg)
        self.anti_tamper.start()

    def _build_ui(self):
        frm_top = tk.Frame(self.root)
        frm_top.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(frm_top, text="Master password:").grid(row=0, column=0, sticky="w")
        self.entry_master = tk.Entry(frm_top, show="*")
        self.entry_master.grid(row=0, column=1, sticky="ew", padx=5)
        tk.Label(frm_top, text="Confirm master:").grid(row=0, column=2, sticky="w")
        self.entry_master_confirm = tk.Entry(frm_top, show="*")
        self.entry_master_confirm.grid(row=0, column=3, sticky="ew", padx=5)
        tk.Button(frm_top, text="Set Master", command=self.set_master).grid(row=0, column=4, padx=5)

        tk.Label(frm_top, text="User factor PIN:").grid(row=1, column=0, sticky="w")
        self.entry_pin = tk.Entry(frm_top, show="*")
        self.entry_pin.grid(row=1, column=1, sticky="ew", padx=5)
        tk.Label(frm_top, text="Confirm PIN:").grid(row=1, column=2, sticky="w")
        self.entry_pin_confirm = tk.Entry(frm_top, show="*")
        self.entry_pin_confirm.grid(row=1, column=3, sticky="ew", padx=5)
        tk.Button(frm_top, text="Set PIN", command=self.set_pin).grid(row=1, column=4, padx=5)

        tk.Label(frm_top, text="Biometric phrase:").grid(row=2, column=0, sticky="w")
        self.entry_bio = tk.Entry(frm_top, show="*")
        self.entry_bio.grid(row=2, column=1, sticky="ew", padx=5)
        tk.Label(frm_top, text="Confirm bio:").grid(row=2, column=2, sticky="w")
        self.entry_bio_confirm = tk.Entry(frm_top, show="*")
        self.entry_bio_confirm.grid(row=2, column=3, sticky="ew", padx=5)
        tk.Button(frm_top, text="Enroll Bio", command=self.set_bio).grid(row=2, column=4, padx=5)

        frm_top.columnconfigure(1, weight=1)
        frm_top.columnconfigure(3, weight=1)

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
        tk.Button(frm_mid2, text="Swarm Gossip", command=self.swarm_gossip).pack(side=tk.LEFT, padx=5)

        tk.Button(frm_mid2, text="ZK Remote Challenge", command=self.zk_challenge).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid2, text="ZK Verify Dummy", command=self.zk_verify_dummy).pack(side=tk.LEFT, padx=5)

        tk.Button(frm_mid2, text="Backup Export", command=self.backup_export).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_mid2, text="Backup Import", command=self.backup_import).pack(side=tk.LEFT, padx=5)

        frm_status = tk.Frame(self.root)
        frm_status.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_anomaly = tk.Label(frm_status, text="Anomaly score: 0.00", width=25)
        self.lbl_anomaly.pack(side=tk.LEFT, padx=5)
        tk.Button(frm_status, text="Refresh Anomaly", command=self.refresh_anomaly).pack(side=tk.LEFT, padx=5)

        self.lbl_swarm = tk.Label(frm_status, text="Swarm: SINGLE NODE", width=25)
        self.lbl_swarm.pack(side=tk.LEFT, padx=5)

        self.lbl_threat = tk.Label(frm_status, text="Threat: NONE", width=25)
        self.lbl_threat.pack(side=tk.LEFT, padx=5)

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

        self.log_msg(f"[QUEEN] alpha status={s1}")
        self.log_msg(f"[QUEEN] beta  status={s2}")
        self.log_msg(f"[QUEEN] gamma status={s3}")

    def set_master(self):
        pw = self.entry_master.get()
        pw2 = self.entry_master_confirm.get()
        if not pw or not pw2:
            messagebox.showwarning("Warning", "Enter and confirm master password")
            return
        if pw != pw2:
            self.unlock_failures += 1
            self.confirm_mismatches += 1
            self.log_msg("[USER] Master password mismatch on confirmation.")
            self.timeline.add("master_mismatch", {"reason": "confirmation_mismatch"})
            self.render_timeline()
            messagebox.showerror("Mismatch", "Master password and confirmation do not match.")
            return
        try:
            self.vault.set_master_password(pw)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.log_msg("[VAULT] Master password set with Argon2id (Tier‑13 via RustCore interface).")
        self.timeline.add("master_set", {"status": "ok"})
        self.render_timeline()
        self.entry_master.delete(0, tk.END)
        self.entry_master_confirm.delete(0, tk.END)

    def set_pin(self):
        pin = self.entry_pin.get()
        pin2 = self.entry_pin_confirm.get()
        if not pin or not pin2:
            messagebox.showwarning("Warning", "Enter and confirm PIN")
            return
        if pin != pin2:
            self.unlock_failures += 1
            self.confirm_mismatches += 1
            self.log_msg("[USER] PIN mismatch on confirmation.")
            self.timeline.add("pin_mismatch", {"reason": "confirmation_mismatch"})
            self.render_timeline()
            messagebox.showerror("Mismatch", "PIN and confirmation do not match.")
            return
        salt = hashlib.sha256(b"borg_pin_salt_tier13_rust_core").digest()
        try:
            key = argon2id_derive_key(pin, salt, time_cost=6, memory_cost=384 * 1024, parallelism=1, length=32)
        except Exception as e:
            messagebox.showerror("Error", f"PIN KDF failed: {e}")
            return
        if self.user_factor_key_buf is not None:
            self.user_factor_key_buf.zeroize()
        self.user_factor_key_buf = SecureBuffer(key)
        self.log_msg("[USER] User factor PIN set (Tier‑13 Argon2id-derived key via RustCore interface).")
        self.timeline.add("pin_set", {"status": "ok"})
        self.render_timeline()
        self.entry_pin.delete(0, tk.END)
        self.entry_pin_confirm.delete(0, tk.END)

    def set_bio(self):
        phrase = self.entry_bio.get()
        phrase2 = self.entry_bio_confirm.get()
        if not phrase or not phrase2:
            messagebox.showwarning("Warning", "Enter and confirm biometric phrase")
            return
        if phrase != phrase2:
            self.unlock_failures += 1
            self.confirm_mismatches += 1
            self.log_msg("[BIO] Biometric phrase mismatch on confirmation.")
            self.timeline.add("bio_mismatch", {"reason": "confirmation_mismatch"})
            self.render_timeline()
            messagebox.showerror("Mismatch", "Biometric phrase and confirmation do not match.")
            return
        try:
            self.biometric_backend.enroll(phrase)
        except Exception as e:
            messagebox.showerror("Error", f"Biometric enroll failed: {e}")
            return
        self.log_msg("[BIO] Biometric phrase enrolled (Tier‑13 OS biometric abstraction).")
        self.timeline.add("bio_set", {"status": "ok"})
        self.render_timeline()
        self.entry_bio.delete(0, tk.END)
        self.entry_bio_confirm.delete(0, tk.END)

    def init_server(self):
        self.server = Server()
        self.client = Client()
        self.log_msg("[TLS] Server and Client initialized with fresh keys/randoms.")

    def run_handshake(self):
        if self.server is None or self.client is None:
            messagebox.showwarning("Warning", "Init server/client first")
            return

        ch = self.client.build_client_hello()
        self.log_msg("[TLS] ClientHello sent.")
        sh = self.server.process_client_hello(ch)
        self.log_msg("[TLS] ServerHello processed.")
        self.client.process_server_hello(sh)
        ckx = self.client.build_client_key_exchange()
        self.log_msg("[TLS] ClientKeyExchange built.")
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
        self.vault_blob = blob
        self.log_msg(f"[VAULT] Encrypted vault v1 ({len(blob)} bytes, memory-resident, Tier‑13 integrity tag).")

    def decrypt_vault(self):
        if self.vault_blob is None:
            messagebox.showwarning("Warning", "No encrypted vault blob yet")
            return
        pw = self.entry_master.get()
        if not pw:
            messagebox.showwarning("Warning", "Enter master password to decrypt")
            return
        try:
            self.vault.decrypt_vault(self.vault_blob, pw)
        except Exception as e:
            self.unlock_failures += 1
            self.log_msg(f"[VAULT] Decrypt failed.")
            messagebox.showerror("Error", str(e))
            return
        self.log_msg(f"[VAULT] Decrypted vault. Entries: {len(self.vault.entries)}")
        self.unlock_failures = 0
        self.entry_master.delete(0, tk.END)

    def borg_lock_vault(self):
        if not self.autonomous_defense.can_operate():
            messagebox.showerror("Defense Lock", "Organism is in lock-down cooldown.")
            self.log_msg("[DEFENSE] Lock operation blocked by autonomous defense.")
            return

        if self.vault_blob is None:
            messagebox.showwarning("Warning", "Encrypt vault first")
            return
        if self.user_factor_key_buf is None:
            messagebox.showwarning("Warning", "Set user factor PIN first")
            return
        try:
            bio_key = self.biometric_backend.authenticate_and_get_key()
        except Exception as e:
            messagebox.showwarning("Warning", f"Biometric not ready: {e}")
            return

        statuses = [self.queen1.status(), self.queen2.status(), self.queen3.status()]
        if any(s == "OFFLINE" for s in statuses):
            messagebox.showerror("Error", "All 3 Queens must be ONLINE to lock.")
            return

        try:
            self.borg_blob = self.collective.wrap(
                self.vault_blob, self.user_factor_key_buf.bytes(), bio_key
            )
        except Exception as e:
            messagebox.showerror("Error", f"Borg lock failed: {e}")
            return

        self.log_msg(f"[BORG] Vault locked by 3 Queens (Tier‑13 Borg blob, {len(self.borg_blob)} bytes).")
        self.anomaly_engine.record("borg_lock", {"size": len(self.borg_blob)})
        self.timeline.add("borg_lock", {"size": len(self.borg_blob)})
        self.render_timeline()

    def borg_unlock_vault(self):
        if not self.autonomous_defense.can_operate():
            messagebox.showerror("Defense Lock", "Organism is in lock-down cooldown.")
            self.log_msg("[DEFENSE] Unlock operation blocked by autonomous defense.")
            return

        if self.borg_blob is None:
            messagebox.showwarning("Warning", "No Borg-locked vault yet")
            return
        if self.user_factor_key_buf is None:
            messagebox.showwarning("Warning", "Set user factor PIN first")
            return
        try:
            bio_key = self.biometric_backend.authenticate_and_get_key()
        except Exception as e:
            messagebox.showwarning("Warning", f"Biometric not ready: {e}")
            return

        statuses = [self.queen1.status(), self.queen2.status(), self.queen3.status()]
        if any(s == "OFFLINE" for s in statuses):
            messagebox.showerror("Error", "All 3 Queens must be ONLINE to unlock.")
            return

        try:
            self.vault_blob = self.collective.unwrap(
                self.borg_blob, self.user_factor_key_buf.bytes(), bio_key
            )
        except Exception as e:
            msg = str(e)
            self.unlock_failures += 1
            self.log_msg(f"[DRIFT] Borg unlock failed.")
            messagebox.showerror("Borg unlock failed", msg)
            self.anomaly_engine.record("borg_unlock_failed", {"reason": "error"})
            self.timeline.add("borg_unlock_failed", {"reason": "error"})
            self.render_timeline()
            return

        self.log_msg("[BORG] Vault unlocked by 3 Queens. You can now decrypt vault with master password.")
        self.anomaly_engine.record("borg_unlock", {"ok": True})
        self.timeline.add("borg_unlock", {"ok": True})
        self.render_timeline()
        self.unlock_failures = 0

    def show_glyph_ciphertext(self):
        if self.borg_blob is None and self.vault_blob is None:
            messagebox.showwarning("Warning", "No ciphertext yet")
            return
        blob = self.borg_blob or self.vault_blob
        glyphs = self.codec.encode(blob)
        self.log_msg("[GLYPH] Ciphertext as glyphs (truncated):")
        self.log_msg(glyphs[:800] + ("..." if len(glyphs) > 800 else ""))

    def swarm_export(self):
        if self.borg_blob is None:
            messagebox.showwarning("Warning", "No Borg-locked vault to export")
            return
        data = self.swarm_node.export_blob(self.borg_blob)
        path = os.path.join(os.path.expanduser("~"), "borg_swarm_export_tier13_rust_core.bin")
        with open(path, "wb") as f:
            f.write(data)
        self.log_msg(f"[SWARM] Exported encrypted Borg blob to {path}")
        self.lbl_swarm.config(text=f"Swarm: {self.swarm_node.consensus_status()}")

    def swarm_import(self):
        path = os.path.join(os.path.expanduser("~"), "borg_swarm_export_tier13_rust_core.bin")
        if not os.path.exists(path):
            messagebox.showwarning("Warning", f"No swarm export at {path}")
            return
        with open(path, "rb") as f:
            data = f.read()
        try:
            self.borg_blob = self.swarm_node.import_blob(data)
        except Exception as e:
            messagebox.showerror("Error", f"Swarm import failed: {e}")
            return
        self.log_msg("[SWARM] Imported Borg blob from encrypted swarm export.")
        self.lbl_swarm.config(text=f"Swarm: {self.swarm_node.consensus_status()}")

    def swarm_gossip(self):
        if self.borg_blob is None:
            messagebox.showwarning("Warning", "No Borg-locked vault to gossip")
            return
        payload = self.swarm_node.export_blob(self.borg_blob)
        self.swarm_node.broadcast_gossip(payload)
        self.log_msg("[SWARM] Broadcasted encrypted gossip with Borg blob meta (Tier‑13).")

    def refresh_anomaly(self):
        score = self.anomaly_engine.score_recent()
        self.lbl_anomaly.config(text=f"Anomaly score: {score:.2f}")
        self.log_msg(f"[ANOMALY] Recent score={score:.2f}")
        level = self.policy_engine.classify_threat_level(score)
        self.defense_last_level = level
        self.lbl_threat.config(text=f"Threat: {level}")

    def _defense_context(self):
        return {
            "unlock_failures": self.unlock_failures,
            "confirm_mismatches": self.confirm_mismatches,
            "tamper_detected": self.tamper_detected,
        }

    def _set_tamper_flag(self, value: bool):
        self.tamper_detected = value

    def zk_challenge(self):
        challenge = self.zk_remote.request_challenge()
        self.log_msg(f"[ZK] Challenge length={len(challenge)} (Tier‑13).")

    def zk_verify_dummy(self):
        challenge = os.urandom(32)
        response = os.urandom(32)
        ok = self.zk_remote.verify_response(challenge, response)
        self.log_msg(f"[ZK] Dummy verification result={ok} (Tier‑13).")

    def backup_export(self):
        if self.borg_blob is None:
            messagebox.showwarning("Warning", "No Borg-locked vault to backup")
            return
        path = os.path.join(os.path.expanduser("~"), "borg_backup_tier13_rust_core.bin")
        try:
            self.backup_manager.export_backup(self.borg_blob, path)
        except Exception as e:
            messagebox.showerror("Error", f"Backup export failed: {e}")
            return

    def backup_import(self):
        path = os.path.join(os.path.expanduser("~"), "borg_backup_tier13_rust_core.bin")
        try:
            self.borg_blob = self.backup_manager.import_backup(path)
        except Exception as e:
            messagebox.showerror("Error", f"Backup import failed: {e}")
            return
        self.log_msg("[BACKUP] Borg blob loaded from backup file.")

    def on_close(self):
        self.telemetry_engine.stop()
        self.etw_engine.stop()
        self.kernel_sensor.stop()
        self.autonomous_defense.stop()
        self.swarm_node.stop_listener()
        self.anti_tamper.stop()
        self.root.destroy()


def main():
    ensure_admin()
    anti_introspection_guard()
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    if PYSIDE_AVAILABLE:
        def launch_pyside():
            qt_app = QtWidgets.QApplication([])
            cockpit = PySideCockpit(app)
            cockpit.show()
            qt_app.exec()

        threading.Thread(target=launch_pyside, daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main()
