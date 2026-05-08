#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Firmware Policy Organ v4 – Xeon-enable Intelligence Cockpit

- No direct NVRAM/firmware writes. Command generator + intelligence only.
- PySide6 cockpit (operator-grade, dark theme).
- Smarter: CPU/board detection (cpuinfo, WMI, dmidecode), board knowledge engine.
- More complete: multi-method schema, Xeon compatibility matrix, board fingerprints.
- More autonomous: validation, JSON schema validation, profile diffing, import/export.
- More integrated: plugin system, external config hooks, logging/status callbacks.
- IFR parser (template) + UEFI variable reader (safe/read-only stubs).
"""

import sys
import os
import json
import platform
import subprocess
import traceback
from copy import deepcopy
from datetime import datetime
from importlib import import_module
from types import ModuleType

# --- autoloader for required libraries --------------------------------------

def _require(module_name, import_expr, friendly_name=None, optional=False):
    try:
        return import_expr()
    except ImportError:
        if optional:
            return None
        print(f"[FATAL] Missing Python module: {module_name}")
        if friendly_name:
            print(f"        Install it with: pip install {friendly_name}")
        else:
            print(f"        Install it with: pip install {module_name}")
        sys.exit(1)

# GUI
QtWidgets = _require("PySide6.QtWidgets", lambda: __import__("PySide6.QtWidgets", fromlist=["*"]))
QtCore = _require("PySide6.QtCore", lambda: __import__("PySide6.QtCore", fromlist=["*"]))
QtGui = _require("PySide6.QtGui", lambda: __import__("PySide6.QtGui", fromlist=["*"]))

# Optional helpers
wmi = _require("wmi", lambda: __import__("wmi"), friendly_name="wmi", optional=True)
cpuinfo = _require("cpuinfo", lambda: __import__("cpuinfo"), friendly_name="py-cpuinfo", optional=True)
jsonschema = _require("jsonschema", lambda: __import__("jsonschema"), friendly_name="jsonschema", optional=True)

# --- base CONFIG (in-memory default) ----------------------------------------

BASE_CONFIG = {
    "Skylake Xeon E3 v5/v6": [
        {
            "board_name": "Example-Z170-Skylake",
            "vendor_hint": "ASUS",
            "product_hint": "Z170",
            "cpu_hint": "E3-12",
            "fingerprint": {
                "vendor": "ASUSTeK COMPUTER INC.",
                "product": "Z170",
            },
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x123",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "danger": 3,
                    "notes": "Tested manually with setup_var.efi on Example Z170 board."
                },
                {
                    "name": "RU.EFI template",
                    "id": "ru_efi",
                    "uefi_var": "Setup",
                    "offset_hex": "0x123",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "danger": 4,
                    "notes": "Use RU.EFI to navigate to 'Setup' NVRAM variable and edit this offset."
                },
            ],
        },
    ],
    "Kaby Lake Xeon": [
        {
            "board_name": "Example-Z270-Kaby",
            "vendor_hint": "ASUS",
            "product_hint": "Z270",
            "cpu_hint": "E3-12",
            "fingerprint": {
                "vendor": "ASUSTeK COMPUTER INC.",
                "product": "Z270",
            },
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x234",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "danger": 3,
                    "notes": "Kaby Xeon E3 profile; confirm IFR before use."
                },
            ],
        },
    ],
    "Coffee Lake Xeon": [
        {
            "board_name": "Example-Z370-Coffee",
            "vendor_hint": "MSI",
            "product_hint": "Z370",
            "cpu_hint": "E-21",
            "fingerprint": {
                "vendor": "MSI",
                "product": "Z370",
            },
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x345",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "danger": 3,
                    "notes": "Coffee Xeon/W profile; board-specific."
                },
            ],
        },
    ],
    "W-series Xeon": [
        {
            "board_name": "Example-W480-Workstation",
            "vendor_hint": "ASUS",
            "product_hint": "W480",
            "cpu_hint": "W-12",
            "fingerprint": {
                "vendor": "ASUSTeK COMPUTER INC.",
                "product": "W480",
            },
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x456",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "danger": 3,
                    "notes": "W-series workstation board; verify with UEFITool/IFR."
                },
                {
                    "name": "Chipsec template",
                    "id": "chipsec",
                    "uefi_var": "Setup",
                    "offset_hex": "0x456",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "danger": 4,
                    "notes": "Use Chipsec to read/write this variable/offset; script template only."
                },
            ],
        },
    ],
    "LGA1151 ES/QS Xeon": [
        {
            "board_name": "Example-ES-QS-1151",
            "vendor_hint": "Gigabyte",
            "product_hint": "Z170",
            "cpu_hint": "ES",
            "fingerprint": {
                "vendor": "Gigabyte Technology Co., Ltd.",
                "product": "Z170",
            },
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x567",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "danger": 4,
                    "notes": "ES/QS profile; microcode/stepping dependent."
                },
            ],
        },
    ],
}

PROFILE_FILE = "xeon_profiles.json"

XEON_MATRIX = [
    {"family": "Skylake Xeon E3 v5/v6", "socket": "LGA1151", "chipsets": "C232/C236, Z170 (with policy tweak)"},
    {"family": "Kaby Lake Xeon",        "socket": "LGA1151", "chipsets": "C236/C246, Z270 (with policy tweak)"},
    {"family": "Coffee Lake Xeon",      "socket": "LGA1151", "chipsets": "C246, Z370/Z390 (with policy tweak)"},
    {"family": "W-series Xeon",         "socket": "LGA1200+", "chipsets": "W480/W580, some Z-series (with policy tweak)"},
    {"family": "LGA1151 ES/QS Xeon",    "socket": "LGA1151", "chipsets": "Varies; highly stepping/board dependent"},
]

MODES = {
    "Enable Xeon features": "enable",
    "Restore stock / disable Xeon": "disable",
}

# --- JSON schema for profiles -----------------------------------------------

PROFILE_SCHEMA = {
    "type": "object",
    "patternProperties": {
        ".*": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["board_name", "methods"],
                "properties": {
                    "board_name": {"type": "string"},
                    "vendor_hint": {"type": "string"},
                    "product_hint": {"type": "string"},
                    "cpu_hint": {"type": "string"},
                    "fingerprint": {
                        "type": "object",
                        "properties": {
                            "vendor": {"type": "string"},
                            "product": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                    "methods": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "name",
                                "id",
                                "uefi_var",
                                "offset_hex",
                                "value_enable_hex",
                                "value_disable_hex",
                            ],
                            "properties": {
                                "name": {"type": "string"},
                                "id": {"type": "string"},
                                "uefi_var": {"type": "string"},
                                "offset_hex": {"type": "string"},
                                "value_enable_hex": {"type": "string"},
                                "value_disable_hex": {"type": "string"},
                                "danger": {"type": "integer"},
                                "notes": {"type": "string"},
                            },
                            "additionalProperties": True,
                        },
                    },
                },
                "additionalProperties": True,
            },
        }
    },
    "additionalProperties": True,
}

def validate_profile_schema(config):
    if not jsonschema:
        return True, "jsonschema not installed; schema validation skipped."
    try:
        jsonschema.validate(instance=config, schema=PROFILE_SCHEMA)
        return True, "Profile schema valid."
    except Exception as e:
        return False, f"Schema validation failed: {e}"

# --- detection helpers -------------------------------------------------------

def detect_cpu_string():
    try:
        if cpuinfo:
            info = cpuinfo.get_cpu_info()
            return info.get("brand_raw", "").strip()
    except Exception:
        pass
    try:
        return platform.processor() or platform.machine()
    except Exception:
        return ""

def detect_board_info_windows():
    vendor = ""
    product = ""
    if wmi and sys.platform.startswith("win"):
        try:
            c = wmi.WMI()
            for board in c.Win32_BaseBoard():
                vendor = (board.Manufacturer or "").strip()
                product = (board.Product or "").strip()
                break
        except Exception:
            pass
    return vendor, product

def detect_board_info_dmidecode():
    vendor = ""
    product = ""
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        return vendor, product
    try:
        out = subprocess.check_output(["dmidecode", "-t", "baseboard"], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Manufacturer:"):
                vendor = line.split(":", 1)[1].strip()
            elif line.startswith("Product Name:"):
                product = line.split(":", 1)[1].strip()
        return vendor, product
    except Exception:
        return "", ""

def detect_board_info():
    vendor, product = detect_board_info_windows()
    if not vendor and not product:
        vendor, product = detect_board_info_dmidecode()
    return vendor, product

# --- board knowledge engine --------------------------------------------------

def fingerprint_match_score(entry_fp, vendor, product):
    score = 0
    v = (vendor or "").upper()
    p = (product or "").upper()
    if not entry_fp:
        return 0
    ev = (entry_fp.get("vendor") or "").upper()
    ep = (entry_fp.get("product") or "").upper()
    if ev and ev in v:
        score += 3
    if ep and ep in p:
        score += 3
    return score

def board_knowledge_explain(entry, vendor, product, cpu_str):
    parts = []
    vh = (entry.get("vendor_hint") or "").upper()
    ph = (entry.get("product_hint") or "").upper()
    ch = (entry.get("cpu_hint") or "").upper()
    fp = entry.get("fingerprint") or {}
    v = (vendor or "").upper()
    p = (product or "").upper()
    c = (cpu_str or "").upper()

    if vh and vh in v:
        parts.append(f"Vendor hint '{vh}' matches detected vendor '{vendor}'.")
    if ph and ph in p:
        parts.append(f"Product hint '{ph}' matches detected product '{product}'.")
    if ch and ch in c:
        parts.append(f"CPU hint '{ch}' matches detected CPU '{cpu_str}'.")
    if fp.get("vendor") and fp["vendor"].upper() in v:
        parts.append(f"Fingerprint vendor '{fp['vendor']}' matches detected vendor '{vendor}'.")
    if fp.get("product") and fp["product"].upper() in p:
        parts.append(f"Fingerprint product '{fp['product']}' matches detected product '{product}'.")
    if not parts:
        parts.append("No strong fingerprint match; selection is heuristic.")
    return " ".join(parts)

def auto_select_profile(config):
    cpu_str = detect_cpu_string().upper()
    vendor, product = detect_board_info()
    vendor_u = vendor.upper()
    product_u = product.upper()

    best = None  # (family, board_name, score, explanation)

    for family, boards in config.items():
        for entry in boards:
            vh = (entry.get("vendor_hint") or "").upper()
            ph = (entry.get("product_hint") or "").upper()
            ch = (entry.get("cpu_hint") or "").upper()
            fp = entry.get("fingerprint") or {}

            score = 0
            if vh and vh in vendor_u:
                score += 2
            if ph and ph in product_u:
                score += 2
            if ch and ch in cpu_str:
                score += 2
            score += fingerprint_match_score(fp, vendor, product)

            if score > 0:
                explanation = board_knowledge_explain(entry, vendor, product, cpu_str)
                if best is None or score > best[2]:
                    best = (family, entry["board_name"], score, explanation)

    if best:
        return best
    return None, None, 0, "No matching profile; manual selection required."

# --- IFR parser (template / stub) -------------------------------------------

def parse_ifr_dump(text):
    """
    Template IFR parser.

    This is intentionally minimal and non-destructive:
    - Accepts IFR text dump as input.
    - Could be extended to locate specific question IDs / offsets.
    - For now, returns a simple structure with lines and a placeholder.
    """
    lines = text.splitlines()
    return {
        "line_count": len(lines),
        "sample": lines[:20],
        "notes": "Extend this parser to extract question IDs, varstores, and offsets."
    }

# --- UEFI variable reader (safe mode stub) ----------------------------------

def read_uefi_variable_safe(var_name):
    """
    Read-only UEFI variable stub.

    This is a placeholder; real implementations are platform-specific and risky.
    Intentionally does NOT write anything, and may simply return None.
    """
    # You could integrate with efivarfs (Linux) or Windows APIs in a safe, read-only way.
    # For now, we just return a stub.
    return {
        "var_name": var_name,
        "status": "not_implemented",
        "notes": "Implement platform-specific read-only UEFI variable access if desired."
    }

# --- validation & command generation ----------------------------------------

def _validate_hex(label, value):
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    v = value.strip()
    if not v.startswith("0x") and not v.startswith("0X"):
        raise ValueError(f"{label} must start with 0x (got: {value})")
    try:
        iv = int(v, 16)
    except ValueError:
        raise ValueError(f"{label} is not valid hex: {value}")
    if "Offset" in label and iv > 0xFFFF:
        raise ValueError(f"{label} looks suspiciously large: {value}")
    return v

def generate_method_commands(family, board_entry, method_entry, mode_key):
    mode = MODES.get(mode_key)
    if mode not in ("enable", "disable"):
        raise ValueError("Invalid mode")

    method_id = method_entry.get("id")
    uefi_var = method_entry.get("uefi_var")
    offset = _validate_hex("Offset", method_entry.get("offset_hex"))
    val_enable = _validate_hex("Enable value", method_entry.get("value_enable_hex"))
    val_disable = _validate_hex("Disable value", method_entry.get("value_disable_hex"))
    notes = method_entry.get("notes", "")
    danger = int(method_entry.get("danger", 3))

    value = val_enable if mode == "enable" else val_disable

    header = []
    header.append(f"# Family : {family}")
    header.append(f"# Board  : {board_entry.get('board_name')}")
    header.append(f"# Method : {method_entry.get('name')} ({method_id})")
    header.append(f"# Var    : {uefi_var}")
    header.append(f"# Offset : {offset}")
    header.append(f"# Mode   : {mode}")
    header.append(f"# Danger : {danger}/5")
    if notes:
        header.append(f"# Notes  : {notes}")
    header.append("")

    if method_id == "grub_setup_var":
        lines = []
        lines.append("# --- GRUB setup_var method ---")
        lines.append("# Boot a GRUB shell with setup_var support (or setup_var.efi).")
        lines.append("# Then run the following command manually:")
        lines.append("")
        lines.extend(header)
        lines.append(f"setup_var {offset} {value}")
        lines.append("")
        lines.append("# After running, reboot and verify POST/CPU enumeration.")
        return "\n".join(lines)

    elif method_id == "ru_efi":
        lines = []
        lines.append("# --- RU.EFI template ---")
        lines.append("# 1. Boot RU.EFI.")
        lines.append("# 2. Locate the NVRAM variable named as below.")
        lines.append("# 3. Navigate to the given offset and change the byte to the value shown.")
        lines.append("")
        lines.extend(header)
        lines.append("# In RU.EFI, edit the byte at this offset to:")
        lines.append(f"#   {value}")
        lines.append("")
        lines.append("# Save changes in RU.EFI, then reboot.")
        return "\n".join(lines)

    elif method_id == "chipsec":
        lines = []
        lines.append("# --- Chipsec script template ---")
        lines.append("# This is a template snippet; adapt it into a Chipsec script.")
        lines.append("")
        lines.extend(header)
        lines.append("from chipsec.chipset import cs")
        lines.append("from chipsec.hal.uefi import UEFI")
        lines.append("")
        lines.append("c = cs()")
        lines.append("uefi = UEFI(c)")
        lines.append(f"var_name = '{uefi_var}'")
        lines.append("# Read variable, modify byte at offset, and write back.")
        lines.append(f"offset = {int(offset, 16)}")
        lines.append(f"value  = {int(value, 16)}  # {value}")
        lines.append("# Implement read/modify/write logic here.")
        lines.append("")
        lines.append("# Run this Chipsec script only after manual review.")
        return "\n".join(lines)

    else:
        lines = []
        lines.append("# Unsupported method in this profile.")
        lines.append("# Fill in method-specific command generation logic.")
        lines.extend(header)
        return "\n".join(lines)

def generate_all_commands_for_board(family, board_entry, mode_key):
    out = []
    methods = board_entry.get("methods", [])
    if not methods:
        raise ValueError("No methods defined for this board/profile.")
    for m in methods:
        out.append(generate_method_commands(family, board_entry, m, mode_key))
        out.append("\n" + "#" * 72 + "\n")
    return "\n".join(out).strip()

# --- profile export/import & diff -------------------------------------------

def export_config(config, path=PROFILE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def import_config(path=PROFILE_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def diff_configs(old, new):
    lines = []
    old_fams = set(old.keys())
    new_fams = set(new.keys())
    added_fams = new_fams - old_fams
    removed_fams = old_fams - new_fams

    if added_fams:
        lines.append("Added families:")
        for f in sorted(added_fams):
            lines.append(f"  + {f}")
    if removed_fams:
        lines.append("Removed families:")
        for f in sorted(removed_fams):
            lines.append(f"  - {f}")

    common = old_fams & new_fams
    for fam in sorted(common):
        old_boards = {b["board_name"] for b in old.get(fam, [])}
        new_boards = {b["board_name"] for b in new.get(fam, [])}
        added_b = new_boards - old_boards
        removed_b = old_boards - new_boards
        if added_b or removed_b:
            lines.append(f"Family: {fam}")
            for b in sorted(added_b):
                lines.append(f"  + board: {b}")
            for b in sorted(removed_b):
                lines.append(f"  - board: {b}")

    if not lines:
        lines.append("No structural differences detected (families/boards).")
    return "\n".join(lines)

# --- plugin system -----------------------------------------------------------

def load_plugins(plugin_dir="plugins"):
    plugins = []
    if not os.path.isdir(plugin_dir):
        return plugins
    for fname in os.listdir(plugin_dir):
        if not fname.endswith(".py"):
            continue
        mod_name = fname[:-3]
        full_name = f"{plugin_dir}.{mod_name}".replace(os.sep, ".")
        try:
            mod = import_module(full_name)
            if isinstance(mod, ModuleType):
                plugins.append(mod)
        except Exception:
            traceback.print_exc()
    return plugins

def apply_plugins_to_config(config, plugins):
    cfg = deepcopy(config)
    for mod in plugins:
        hook = getattr(mod, "extend_xeon_profiles", None)
        if callable(hook):
            try:
                cfg = hook(cfg)
            except Exception:
                traceback.print_exc()
    return cfg

# --- GUI organ (PySide6) ----------------------------------------------------

class FirmwarePolicyOrgan(QtWidgets.QMainWindow):
    """
    Integration hooks:
      - external_config: dict to override internal config.
      - on_log: callable(str) to receive log lines.
      - on_status: callable(str) to receive status updates.
    """

    def __init__(self, external_config=None, on_log=None, on_status=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Firmware Policy Organ – Xeon Enable Cockpit v4")
        self.resize(1280, 800)

        self.on_log_cb = on_log
        self.on_status_cb = on_status

        self.base_config = deepcopy(BASE_CONFIG)
        self.config = deepcopy(BASE_CONFIG)
        if external_config:
            self.config = external_config

        # Apply plugins
        self.plugins = load_plugins()
        if self.plugins:
            self.config = apply_plugins_to_config(self.config, self.plugins)

        self.families = list(self.config.keys())

        self.cpu_detected = detect_cpu_string()
        self.board_vendor_detected, self.board_product_detected = detect_board_info()

        self.log_lines = []

        self._build_ui()
        self._populate_initial()
        self._auto_select_if_possible()

    # --- UI ------------------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        # Top row: detection + selectors + actions
        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout)

        # Detection panel
        detect_group = QtWidgets.QGroupBox("Detected platform")
        top_layout.addWidget(detect_group)
        detect_layout = QtWidgets.QVBoxLayout(detect_group)

        cpu_label = self.cpu_detected or "<unknown>"
        board_label = (self.board_vendor_detected + " " + self.board_product_detected).strip() or "<unknown>"

        self.cpu_label = QtWidgets.QLabel(f"CPU   : {cpu_label}")
        self.board_label = QtWidgets.QLabel(f"Board : {board_label}")
        detect_layout.addWidget(self.cpu_label)
        detect_layout.addWidget(self.board_label)

        # Selectors
        sel_widget = QtWidgets.QWidget()
        sel_layout = QtWidgets.QFormLayout(sel_widget)
        top_layout.addWidget(sel_widget, stretch=1)

        self.family_combo = QtWidgets.QComboBox()
        self.family_combo.currentIndexChanged.connect(self._on_family_change)
        sel_layout.addRow("CPU family:", self.family_combo)

        self.board_combo = QtWidgets.QComboBox()
        self.board_combo.currentIndexChanged.connect(self._on_board_change)
        sel_layout.addRow("Board/profile:", self.board_combo)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(list(MODES.keys()))
        sel_layout.addRow("Mode:", self.mode_combo)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.currentIndexChanged.connect(self._update_method_preview)
        sel_layout.addRow("Method:", self.method_combo)

        # Buttons
        btn_group = QtWidgets.QGroupBox("Actions")
        top_layout.addWidget(btn_group)
        btn_layout = QtWidgets.QVBoxLayout(btn_group)

        self.generate_btn = QtWidgets.QPushButton("Generate commands")
        self.generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self.generate_btn)

        self.copy_btn = QtWidgets.QPushButton("Copy to clipboard")
        self.copy_btn.clicked.connect(self._on_copy)
        btn_layout.addWidget(self.copy_btn)

        self.inspect_btn = QtWidgets.QPushButton("Inspect profile")
        self.inspect_btn.clicked.connect(self._on_inspect)
        btn_layout.addWidget(self.inspect_btn)

        self.export_btn = QtWidgets.QPushButton("Export profiles")
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.export_btn)

        self.import_btn = QtWidgets.QPushButton("Import profiles")
        self.import_btn.clicked.connect(self._on_import)
        btn_layout.addWidget(self.import_btn)

        self.diff_btn = QtWidgets.QPushButton("Diff vs base")
        self.diff_btn.clicked.connect(self._on_diff)
        btn_layout.addWidget(self.diff_btn)

        self.builder_btn = QtWidgets.QPushButton("Profile builder")
        self.builder_btn.clicked.connect(self._on_builder)
        btn_layout.addWidget(self.builder_btn)

        self.schema_btn = QtWidgets.QPushButton("Validate schema")
        self.schema_btn.clicked.connect(self._on_schema_validate)
        btn_layout.addWidget(self.schema_btn)

        btn_layout.addStretch(1)

        # Center: split commands / right panels
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Left: commands
        cmd_group = QtWidgets.QGroupBox("Generated commands (run manually!)")
        cmd_layout = QtWidgets.QVBoxLayout(cmd_group)
        self.output_text = QtWidgets.QPlainTextEdit()
        self.output_text.setReadOnly(False)
        cmd_layout.addWidget(self.output_text)
        splitter.addWidget(cmd_group)

        # Right: log + preview + matrix
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        splitter.addWidget(right_widget)

        log_group = QtWidgets.QGroupBox("Organ log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, stretch=1)

        preview_group = QtWidgets.QGroupBox("Method preview / danger meter")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.preview_text = QtWidgets.QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        right_layout.addWidget(preview_group, stretch=1)

        matrix_group = QtWidgets.QGroupBox("Xeon compatibility matrix")
        matrix_layout = QtWidgets.QVBoxLayout(matrix_group)
        self.matrix_text = QtWidgets.QPlainTextEdit()
        self.matrix_text.setReadOnly(True)
        matrix_layout.addWidget(self.matrix_text)
        right_layout.addWidget(matrix_group, stretch=1)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready.")

        # Dark-ish palette
        self._apply_dark_palette()

    def _apply_dark_palette(self):
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(224, 224, 224))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(17, 17, 17))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 220))
        palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(224, 224, 224))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(51, 51, 51))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(224, 224, 224))
        palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(64, 128, 255))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        app.setPalette(palette)

    # --- internal helpers ----------------------------------------------------

    def _populate_initial(self):
        self.family_combo.clear()
        self.family_combo.addItems(self.families)
        self._refresh_boards()
        self._populate_matrix()

    def _populate_matrix(self):
        buf = []
        for row in XEON_MATRIX:
            buf.append(f"Family : {row['family']}")
            buf.append(f"Socket : {row['socket']}")
            buf.append(f"Chipsets: {row['chipsets']}")
            buf.append("")
        self.matrix_text.setPlainText("\n".join(buf))

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        self.log_text.appendPlainText(line)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        if self.on_log_cb:
            try:
                self.on_log_cb(line)
            except Exception:
                pass

    def _set_status(self, msg):
        self.status_bar.showMessage(msg)
        self._log(msg)
        if self.on_status_cb:
            try:
                self.on_status_cb(msg)
            except Exception:
                pass

    def _refresh_families(self):
        self.families = list(self.config.keys())
        self.family_combo.blockSignals(True)
        self.family_combo.clear()
        self.family_combo.addItems(self.families)
        self.family_combo.blockSignals(False)

    def _refresh_boards(self):
        family = self.family_combo.currentText()
        boards = self.config.get(family, [])
        names = [b["board_name"] for b in boards]
        self.board_combo.blockSignals(True)
        self.board_combo.clear()
        self.board_combo.addItems(names)
        self.board_combo.blockSignals(False)
        self._refresh_methods()
        self._update_method_preview()

    def _refresh_methods(self):
        board_entry = self._get_selected_board_entry()
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        self.method_combo.addItem("All methods")
        if board_entry:
            for m in board_entry.get("methods", []):
                self.method_combo.addItem(m.get("name"))
        self.method_combo.blockSignals(False)

    def _get_selected_board_entry(self):
        family = self.family_combo.currentText()
        board_name = self.board_combo.currentText()
        for entry in self.config.get(family, []):
            if entry.get("board_name") == board_name:
                return entry
        return None

    def _get_selected_method_entry(self, board_entry):
        name = self.method_combo.currentText()
        if name == "All methods":
            return None
        for m in board_entry.get("methods", []):
            if m.get("name") == name:
                return m
        return None

    def _auto_select_if_possible(self):
        fam, board, score, explanation = auto_select_profile(self.config)
        if fam and board:
            idx_f = self.family_combo.findText(fam)
            if idx_f >= 0:
                self.family_combo.setCurrentIndex(idx_f)
            self._refresh_boards()
            idx_b = self.board_combo.findText(board)
            if idx_b >= 0:
                self.board_combo.setCurrentIndex(idx_b)
            self._refresh_methods()
            self._update_method_preview()
            self._set_status(f"Auto-selected profile: {fam} / {board} (score {score}). {explanation}")
        else:
            self._set_status("No auto-selection match; choose family/board manually.")

    def _update_method_preview(self):
        board_entry = self._get_selected_board_entry()
        if not board_entry:
            self.preview_text.setPlainText("No board/profile selected.")
            return
        method_entry = self._get_selected_method_entry(board_entry)
        if method_entry is None:
            self.preview_text.setPlainText("All methods selected.\nDanger meter will vary per method.")
            return

        danger = int(method_entry.get("danger", 3))
        notes = method_entry.get("notes", "")
        method_id = method_entry.get("id")
        name = method_entry.get("name")

        buf = []
        buf.append(f"Method : {name} ({method_id})")
        buf.append(f"Danger : {danger}/5")
        buf.append("")

        if method_id == "grub_setup_var":
            buf.append("Preview:")
            buf.append("- Uses GRUB setup_var to flip a UEFI variable.")
            buf.append("- Requires booting a GRUB shell with setup_var support.")
        elif method_id == "ru_efi":
            buf.append("Preview:")
            buf.append("- Uses RU.EFI to edit NVRAM variable bytes directly.")
            buf.append("- Requires careful navigation and manual byte editing.")
        elif method_id == "chipsec":
            buf.append("Preview:")
            buf.append("- Uses Chipsec to script UEFI variable read/modify/write.")
            buf.append("- Requires Python + Chipsec and careful script review.")
        else:
            buf.append("Preview:")
            buf.append("- Custom/unknown method; review profile notes.")

        if notes:
            buf.append("")
            buf.append("Notes:")
            buf.append(notes)

        self.preview_text.setPlainText("\n".join(buf))

    # --- event handlers ------------------------------------------------------

    def _on_family_change(self, idx):
        self._refresh_boards()
        self._set_status(f"Family changed to: {self.family_combo.currentText()}")

    def _on_board_change(self, idx):
        self._refresh_methods()
        self._update_method_preview()
        self._set_status(f"Board/profile changed to: {self.board_combo.currentText()}")

    def _on_generate(self):
        family = self.family_combo.currentText()
        board_entry = self._get_selected_board_entry()
        mode_key = self.mode_combo.currentText()

        if not family:
            QtWidgets.QMessageBox.critical(self, "Error", "No CPU family selected.")
            return
        if not board_entry:
            QtWidgets.QMessageBox.critical(self, "Error", "No board/profile selected for this family.")
            return

        try:
            method_entry = self._get_selected_method_entry(board_entry)
            if method_entry is None:
                commands = generate_all_commands_for_board(family, board_entry, mode_key)
            else:
                commands = generate_method_commands(family, board_entry, method_entry, mode_key)
        except Exception as e:
            tb = traceback.format_exc()
            self._log(tb)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate commands:\n{e}")
            self._set_status("Generation failed.")
            return

        self.output_text.setPlainText(commands)
        self._set_status("Commands generated. Review carefully before running on target system.")

    def _on_copy(self):
        text = self.output_text.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.information(self, "Info", "Nothing to copy.")
            return
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(text)
        QtWidgets.QMessageBox.information(self, "Copied", "Commands copied to clipboard.")
        self._set_status("Commands copied to clipboard.")

    def _on_inspect(self):
        family = self.family_combo.currentText()
        board_entry = self._get_selected_board_entry()
        if not board_entry:
            QtWidgets.QMessageBox.information(self, "Profile", "No board/profile selected.")
            return

        lines = []
        lines.append(f"Family : {family}")
        lines.append(f"Board  : {board_entry.get('board_name')}")
        lines.append(f"Vendor hint : {board_entry.get('vendor_hint')}")
        lines.append(f"Product hint: {board_entry.get('product_hint')}")
        lines.append(f"CPU hint    : {board_entry.get('cpu_hint')}")
        fp = board_entry.get("fingerprint") or {}
        lines.append(f"Fingerprint vendor : {fp.get('vendor')}")
        lines.append(f"Fingerprint product: {fp.get('product')}")
        lines.append("")
        lines.append("Methods:")
        for m in board_entry.get("methods", []):
            lines.append(f"  - {m.get('name')} ({m.get('id')})")
            lines.append(f"      Var    : {m.get('uefi_var')}")
            lines.append(f"      Offset : {m.get('offset_hex')}")
            lines.append(f"      Enable : {m.get('value_enable_hex')}")
            lines.append(f"      Disable: {m.get('value_disable_hex')}")
            lines.append(f"      Danger : {m.get('danger', 3)}/5")
            if m.get("notes"):
                lines.append(f"      Notes  : {m.get('notes')}")
        info = "\n".join(lines)
        QtWidgets.QMessageBox.information(self, "Profile details", info)
        self._set_status("Profile inspected.")

    def _on_export(self):
        try:
            export_config(self.config, PROFILE_FILE)
            self._set_status(f"Profiles exported to {PROFILE_FILE}.")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(tb)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export profiles:\n{e}")
            self._set_status("Export failed.")

    def _on_import(self):
        try:
            new_cfg = import_config(PROFILE_FILE)
            if not new_cfg:
                QtWidgets.QMessageBox.information(self, "Import", f"No profile file found at {PROFILE_FILE}.")
                return
            ok, msg = validate_profile_schema(new_cfg)
            if not ok:
                QtWidgets.QMessageBox.warning(self, "Schema warning", f"Imported config failed schema validation:\n{msg}")
            self.config = new_cfg
            self._refresh_families()
            self._refresh_boards()
            self._set_status(f"Profiles imported from {PROFILE_FILE}. {msg}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(tb)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to import profiles:\n{e}")
            self._set_status("Import failed.")

    def _on_diff(self):
        try:
            diff = diff_configs(self.base_config, self.config)
            QtWidgets.QMessageBox.information(self, "Profile diff vs base", diff)
            self._set_status("Diff computed.")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(tb)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to diff profiles:\n{e}")
            self._set_status("Diff failed.")

    def _on_schema_validate(self):
        ok, msg = validate_profile_schema(self.config)
        if ok:
            QtWidgets.QMessageBox.information(self, "Schema", msg)
        else:
            QtWidgets.QMessageBox.warning(self, "Schema", msg)
        self._set_status(msg)

    def _on_builder(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Profile builder wizard")
        layout = QtWidgets.QFormLayout(dlg)

        fam_edit = QtWidgets.QLineEdit()
        board_edit = QtWidgets.QLineEdit()
        vendor_edit = QtWidgets.QLineEdit()
        product_edit = QtWidgets.QLineEdit()
        cpu_hint_edit = QtWidgets.QLineEdit()
        method_name_edit = QtWidgets.QLineEdit("GRUB setup_var")
        method_id_edit = QtWidgets.QLineEdit("grub_setup_var")
        uefi_var_edit = QtWidgets.QLineEdit("Setup")
        offset_edit = QtWidgets.QLineEdit("0x000")
        enable_edit = QtWidgets.QLineEdit("0x01")
        disable_edit = QtWidgets.QLineEdit("0x00")
        danger_edit = QtWidgets.QLineEdit("3")
        notes_edit = QtWidgets.QLineEdit("")

        layout.addRow("Family (existing or new):", fam_edit)
        layout.addRow("Board name:", board_edit)
        layout.addRow("Vendor fingerprint:", vendor_edit)
        layout.addRow("Product fingerprint:", product_edit)
        layout.addRow("CPU hint:", cpu_hint_edit)
        layout.addRow("Method name:", method_name_edit)
        layout.addRow("Method id:", method_id_edit)
        layout.addRow("UEFI var name:", uefi_var_edit)
        layout.addRow("Offset (hex):", offset_edit)
        layout.addRow("Enable value (hex):", enable_edit)
        layout.addRow("Disable value (hex):", disable_edit)
        layout.addRow("Danger (1-5):", danger_edit)
        layout.addRow("Notes:", notes_edit)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        layout.addRow(btn_box)

        def on_save():
            family = fam_edit.text().strip()
            board = board_edit.text().strip()
            if not family or not board:
                QtWidgets.QMessageBox.critical(dlg, "Error", "Family and board name are required.")
                return
            try:
                _validate_hex("Offset", offset_edit.text())
                _validate_hex("Enable value", enable_edit.text())
                _validate_hex("Disable value", disable_edit.text())
                d = int(danger_edit.text())
                if d < 1 or d > 5:
                    raise ValueError("Danger out of range")
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Error", f"Validation failed:\n{e}")
                return

            method = {
                "name": method_name_edit.text().strip() or "GRUB setup_var",
                "id": method_id_edit.text().strip() or "grub_setup_var",
                "uefi_var": uefi_var_edit.text().strip() or "Setup",
                "offset_hex": offset_edit.text().strip(),
                "value_enable_hex": enable_edit.text().strip(),
                "value_disable_hex": disable_edit.text().strip(),
                "danger": d,
                "notes": notes_edit.text().strip(),
            }
            entry = {
                "board_name": board,
                "vendor_hint": vendor_edit.text().strip(),
                "product_hint": product_edit.text().strip(),
                "cpu_hint": cpu_hint_edit.text().strip(),
                "fingerprint": {
                    "vendor": vendor_edit.text().strip(),
                    "product": product_edit.text().strip(),
                },
                "methods": [method],
            }

            if family not in self.config:
                self.config[family] = []
            self.config[family].append(entry)
            self._refresh_families()
            idx_f = self.family_combo.findText(family)
            if idx_f >= 0:
                self.family_combo.setCurrentIndex(idx_f)
            self._refresh_boards()
            idx_b = self.board_combo.findText(board)
            if idx_b >= 0:
                self.board_combo.setCurrentIndex(idx_b)
            self._refresh_methods()
            self._update_method_preview()
            self._set_status(f"Profile added: {family} / {board}")
            dlg.accept()

        btn_box.accepted.connect(on_save)
        btn_box.rejected.connect(dlg.reject)

        dlg.exec()

    # --- public API / integration hooks -------------------------------------

    def get_config(self):
        return deepcopy(self.config)

    def set_config(self, new_config):
        self.config = deepcopy(new_config)
        self._refresh_families()
        self._refresh_boards()
        self._set_status("Config replaced via integration hook.")

# --- main --------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = FirmwarePolicyOrgan()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
