#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Firmware Policy Organ: Xeon-enable command generator cockpit (v2)

- No direct NVRAM/firmware writes.
- Smarter: CPU/board detection via cpuinfo, WMI (Windows), dmidecode (Linux/macOS).
- Operator-grade: dark cockpit, status bar, log, method preview, danger meter, matrix panel.
- Autonomous: validation, basic profile diffing, profile builder wizard, export/import.
- More complete: multi-method schema, Xeon compatibility matrix.
- Integrated: class-based organ with hooks for embedding into a larger system.
"""

import sys
import os
import json
import platform
import subprocess
import traceback
from datetime import datetime
from copy import deepcopy

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

tk = _require("tkinter", lambda: __import__("tkinter"))
ttk = _require("tkinter.ttk", lambda: __import__("tkinter.ttk", fromlist=["*"]))
messagebox = _require("tkinter.messagebox", lambda: __import__("tkinter.messagebox", fromlist=["*"]))

wmi = _require("wmi", lambda: __import__("wmi"), friendly_name="wmi", optional=True)
cpuinfo = _require("cpuinfo", lambda: __import__("cpuinfo"), friendly_name="py-cpuinfo", optional=True)

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
                    "danger": 3,  # 1–5
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
    # Simple illustrative matrix; extend as needed.
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
        # Requires dmidecode installed and root privileges typically.
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

def auto_select_profile(config):
    cpu_str = detect_cpu_string().upper()
    vendor, product = detect_board_info()
    vendor_u = vendor.upper()
    product_u = product.upper()

    best = None  # (family, board_name, score)

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
                if best is None or score > best[2]:
                    best = (family, entry["board_name"], score)

    if best:
        return best[0], best[1]
    return None, None

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
    # simple sanity: offset not insanely large
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
    # Very simple diff: list added/removed families/boards.
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

# --- GUI organ --------------------------------------------------------------

class FirmwarePolicyOrgan:
    def __init__(self, root=None, external_config=None):
        self._own_root = False
        if root is None:
            root = tk.Tk()
            self._own_root = True
        self.root = root
        self.root.title("Firmware Policy Organ – Xeon Enable Cockpit")
        self.root.geometry("1200x720")

        # config
        self.base_config = deepcopy(BASE_CONFIG)
        self.config = deepcopy(BASE_CONFIG)
        if external_config:
            self.config = external_config

        self.families = list(self.config.keys())

        self.family_var = tk.StringVar(value=self.families[0] if self.families else "")
        self.board_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value=list(MODES.keys())[0])
        self.method_var = tk.StringVar(value="All methods")

        self.cpu_detected = detect_cpu_string()
        self.board_vendor_detected, self.board_product_detected = detect_board_info()

        self.status_var = tk.StringVar(value="Ready.")
        self.log_lines = []

        self._build_style()
        self._build_layout()
        self._refresh_boards()
        self._auto_select_if_possible()

    # --- style / layout -----------------------------------------------------

    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background="#1e1e1e", foreground="#e0e0e0", fieldbackground="#2b2b2b")
        style.configure("TLabel", background="#1e1e1e", foreground="#e0e0e0")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabelframe", background="#1e1e1e", foreground="#e0e0e0")
        style.configure("TLabelframe.Label", background="#1e1e1e", foreground="#e0e0e0")
        style.configure("TButton", background="#333333", foreground="#e0e0e0")
        style.map("TButton", background=[("active", "#444444")])

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Top row: detection + selectors + actions
        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 10))

        # Detection info
        detect_frame = ttk.Labelframe(top, text="Detected platform")
        detect_frame.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 10))

        cpu_label = self.cpu_detected or "<unknown>"
        board_label = (self.board_vendor_detected + " " + self.board_product_detected).strip() or "<unknown>"

        ttk.Label(detect_frame, text=f"CPU   : {cpu_label}").pack(anchor="w", padx=5, pady=2)
        ttk.Label(detect_frame, text=f"Board : {board_label}").pack(anchor="w", padx=5, pady=2)

        # Selectors
        sel_frame = ttk.Frame(top)
        sel_frame.grid(row=0, column=1, sticky="nsew")

        ttk.Label(sel_frame, text="CPU family:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.family_combo = ttk.Combobox(
            sel_frame,
            textvariable=self.family_var,
            values=self.families,
            state="readonly",
            width=35,
        )
        self.family_combo.grid(row=0, column=1, sticky="w")
        self.family_combo.bind("<<ComboboxSelected>>", self._on_family_change)

        ttk.Label(sel_frame, text="Board/profile:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.board_combo = ttk.Combobox(
            sel_frame,
            textvariable=self.board_var,
            values=[],
            state="readonly",
            width=35,
        )
        self.board_combo.grid(row=1, column=1, sticky="w", pady=(5, 0))
        self.board_combo.bind("<<ComboboxSelected>>", self._on_board_change)

        ttk.Label(sel_frame, text="Mode:").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.mode_combo = ttk.Combobox(
            sel_frame,
            textvariable=self.mode_var,
            values=list(MODES.keys()),
            state="readonly",
            width=35,
        )
        self.mode_combo.grid(row=2, column=1, sticky="w", pady=(5, 0))

        ttk.Label(sel_frame, text="Method:").grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.method_combo = ttk.Combobox(
            sel_frame,
            textvariable=self.method_var,
            values=["All methods"],
            state="readonly",
            width=35,
        )
        self.method_combo.grid(row=3, column=1, sticky="w", pady=(5, 0))

        # Buttons
        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=0, column=2, rowspan=3, padx=(20, 0), sticky="n")

        self.generate_btn = ttk.Button(btn_frame, text="Generate commands", command=self._on_generate)
        self.generate_btn.pack(fill="x", pady=(0, 5))

        self.copy_btn = ttk.Button(btn_frame, text="Copy to clipboard", command=self._on_copy)
        self.copy_btn.pack(fill="x", pady=(0, 5))

        self.inspect_btn = ttk.Button(btn_frame, text="Inspect profile", command=self._on_inspect)
        self.inspect_btn.pack(fill="x", pady=(0, 5))

        self.export_btn = ttk.Button(btn_frame, text="Export profiles", command=self._on_export)
        self.export_btn.pack(fill="x", pady=(0, 5))

        self.import_btn = ttk.Button(btn_frame, text="Import profiles", command=self._on_import)
        self.import_btn.pack(fill="x", pady=(0, 5))

        self.diff_btn = ttk.Button(btn_frame, text="Diff vs base", command=self._on_diff)
        self.diff_btn.pack(fill="x", pady=(0, 5))

        self.builder_btn = ttk.Button(btn_frame, text="Profile builder", command=self._on_builder)
        self.builder_btn.pack(fill="x", pady=(0, 5))

        # Center: output + log + side panels
        center = ttk.Frame(main)
        center.pack(fill="both", expand=True)

        # Left: commands
        output_frame = ttk.Labelframe(center, text="Generated commands (run manually!)")
        output_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.output_text = tk.Text(output_frame, wrap="word", height=20, bg="#111111", fg="#e0e0e0", insertbackground="#e0e0e0")
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Right: log + method preview + matrix
        right = ttk.Frame(center)
        right.pack(side="right", fill="both", expand=False)

        log_frame = ttk.Labelframe(right, text="Organ log")
        log_frame.pack(fill="both", expand=True, padx=(5, 0))

        self.log_text = tk.Text(log_frame, wrap="word", width=40, height=10, bg="#111111", fg="#a0a0a0", insertbackground="#e0e0e0")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        preview_frame = ttk.Labelframe(right, text="Method preview / danger meter")
        preview_frame.pack(fill="both", expand=True, padx=(5, 0), pady=(5, 0))

        self.preview_text = tk.Text(preview_frame, wrap="word", width=40, height=8, bg="#111111", fg="#e0e0e0", insertbackground="#e0e0e0")
        self.preview_text.pack(fill="both", expand=True, padx=5, pady=5)

        matrix_frame = ttk.Labelframe(right, text="Xeon compatibility matrix")
        matrix_frame.pack(fill="both", expand=True, padx=(5, 0), pady=(5, 0))

        self.matrix_text = tk.Text(matrix_frame, wrap="word", width=40, height=8, bg="#111111", fg="#c0c0c0", insertbackground="#e0e0e0")
        self.matrix_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._populate_matrix()

        # Status bar
        status_frame = ttk.Frame(main)
        status_frame.pack(fill="x", pady=(5, 0))
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x")

    # --- internal helpers ----------------------------------------------------

    def _populate_matrix(self):
        self.matrix_text.delete("1.0", "end")
        for row in XEON_MATRIX:
            self.matrix_text.insert("end", f"Family : {row['family']}\n")
            self.matrix_text.insert("end", f"Socket : {row['socket']}\n")
            self.matrix_text.insert("end", f"Chipsets: {row['chipsets']}\n\n")
        self.matrix_text.config(state="disabled")

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")

    def _set_status(self, msg):
        self.status_var.set(msg)
        self._log(msg)

    def _refresh_families(self):
        self.families = list(self.config.keys())
        self.family_combo["values"] = self.families
        if self.families and self.family_var.get() not in self.families:
            self.family_var.set(self.families[0])

    def _refresh_boards(self):
        family = self.family_var.get()
        boards = self.config.get(family, [])
        names = [b["board_name"] for b in boards]
        self.board_combo["values"] = names
        if names:
            if self.board_var.get() not in names:
                self.board_var.set(names[0])
        else:
            self.board_var.set("")
        self._refresh_methods()
        self._update_method_preview()

    def _refresh_methods(self):
        board_entry = self._get_selected_board_entry()
        methods = ["All methods"]
        if board_entry:
            for m in board_entry.get("methods", []):
                methods.append(m.get("name"))
        self.method_combo["values"] = methods
        if methods:
            if self.method_var.get() not in methods:
                self.method_var.set(methods[0])

    def _get_selected_board_entry(self):
        family = self.family_var.get()
        board_name = self.board_var.get()
        for entry in self.config.get(family, []):
            if entry.get("board_name") == board_name:
                return entry
        return None

    def _get_selected_method_entry(self, board_entry):
        name = self.method_var.get()
        if name == "All methods":
            return None
        for m in board_entry.get("methods", []):
            if m.get("name") == name:
                return m
        return None

    def _auto_select_if_possible(self):
        fam, board = auto_select_profile(self.config)
        if fam and board:
            self.family_var.set(fam)
            self._refresh_boards()
            self.board_var.set(board)
            self._refresh_methods()
            self._update_method_preview()
            self._set_status(f"Auto-selected profile: {fam} / {board}")
        else:
            self._set_status("No auto-selection match; choose family/board manually.")

    def _update_method_preview(self):
        self.preview_text.config(state="normal")
        self.preview_text.delete("1.0", "end")
        board_entry = self._get_selected_board_entry()
        if not board_entry:
            self.preview_text.insert("end", "No board/profile selected.\n")
            self.preview_text.config(state="disabled")
            return
        method_entry = self._get_selected_method_entry(board_entry)
        if method_entry is None:
            self.preview_text.insert("end", "All methods selected.\n")
            self.preview_text.insert("end", "Danger meter will vary per method.\n")
            self.preview_text.config(state="disabled")
            return

        danger = int(method_entry.get("danger", 3))
        notes = method_entry.get("notes", "")
        method_id = method_entry.get("id")
        name = method_entry.get("name")

        self.preview_text.insert("end", f"Method : {name} ({method_id})\n")
        self.preview_text.insert("end", f"Danger : {danger}/5\n\n")

        if method_id == "grub_setup_var":
            self.preview_text.insert("end", "Preview:\n- Uses GRUB setup_var to flip a UEFI variable.\n")
            self.preview_text.insert("end", "- Requires booting a GRUB shell with setup_var support.\n")
        elif method_id == "ru_efi":
            self.preview_text.insert("end", "Preview:\n- Uses RU.EFI to edit NVRAM variable bytes directly.\n")
            self.preview_text.insert("end", "- Requires careful navigation and manual byte editing.\n")
        elif method_id == "chipsec":
            self.preview_text.insert("end", "Preview:\n- Uses Chipsec to script UEFI variable read/modify/write.\n")
            self.preview_text.insert("end", "- Requires Python + Chipsec and careful script review.\n")
        else:
            self.preview_text.insert("end", "Preview:\n- Custom/unknown method; review profile notes.\n")

        if notes:
            self.preview_text.insert("end", f"\nNotes:\n{notes}\n")

        self.preview_text.config(state="disabled")

    # --- event handlers ------------------------------------------------------

    def _on_family_change(self, event=None):
        self._refresh_boards()
        self._set_status(f"Family changed to: {self.family_var.get()}")

    def _on_board_change(self, event=None):
        self._refresh_methods()
        self._update_method_preview()
        self._set_status(f"Board/profile changed to: {self.board_var.get()}")

    def _on_generate(self):
        family = self.family_var.get()
        board_entry = self._get_selected_board_entry()
        mode_key = self.mode_var.get()

        if not family:
            messagebox.showerror("Error", "No CPU family selected.")
            return
        if not board_entry:
            messagebox.showerror("Error", "No board/profile selected for this family.")
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
            messagebox.showerror("Error", f"Failed to generate commands:\n{e}")
            self._set_status("Generation failed.")
            return

        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", commands)
        self._set_status("Commands generated. Review carefully before running on target system.")

    def _on_copy(self):
        text = self.output_text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Info", "Nothing to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Commands copied to clipboard.")
        self._set_status("Commands copied to clipboard.")

    def _on_inspect(self):
        family = self.family_var.get()
        board_entry = self._get_selected_board_entry()
        if not board_entry:
            messagebox.showinfo("Profile", "No board/profile selected.")
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
        messagebox.showinfo("Profile details", info)
        self._set_status("Profile inspected.")

    def _on_export(self):
        try:
            export_config(self.config, PROFILE_FILE)
            self._set_status(f"Profiles exported to {PROFILE_FILE}.")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(tb)
            messagebox.showerror("Error", f"Failed to export profiles:\n{e}")
            self._set_status("Export failed.")

    def _on_import(self):
        try:
            new_cfg = import_config(PROFILE_FILE)
            if not new_cfg:
                messagebox.showinfo("Import", f"No profile file found at {PROFILE_FILE}.")
                return
            self.config = new_cfg
            self._refresh_families()
            self._refresh_boards()
            self._set_status(f"Profiles imported from {PROFILE_FILE}.")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(tb)
            messagebox.showerror("Error", f"Failed to import profiles:\n{e}")
            self._set_status("Import failed.")

    def _on_diff(self):
        try:
            diff = diff_configs(self.base_config, self.config)
            messagebox.showinfo("Profile diff vs base", diff)
            self._set_status("Diff computed.")
        except Exception as e:
            tb = traceback.format_exc()
            self._log(tb)
            messagebox.showerror("Error", f"Failed to diff profiles:\n{e}")
            self._set_status("Diff failed.")

    def _on_builder(self):
        # Simple profile builder wizard: minimal fields.
        builder = tk.Toplevel(self.root)
        builder.title("Profile builder wizard")

        fam_var = tk.StringVar(value="")
        board_var = tk.StringVar(value="")
        vendor_var = tk.StringVar(value="")
        product_var = tk.StringVar(value="")
        cpu_hint_var = tk.StringVar(value="")
        method_name_var = tk.StringVar(value="GRUB setup_var")
        method_id_var = tk.StringVar(value="grub_setup_var")
        uefi_var_var = tk.StringVar(value="Setup")
        offset_var = tk.StringVar(value="0x000")
        enable_var = tk.StringVar(value="0x01")
        disable_var = tk.StringVar(value="0x00")
        danger_var = tk.StringVar(value="3")
        notes_var = tk.StringVar(value="")

        row = 0
        def add_label_entry(text, var):
            nonlocal row
            ttk.Label(builder, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(builder, textvariable=var, width=40).grid(row=row, column=1, sticky="w", padx=5, pady=2)
            row += 1

        add_label_entry("Family (existing or new):", fam_var)
        add_label_entry("Board name:", board_var)
        add_label_entry("Vendor fingerprint:", vendor_var)
        add_label_entry("Product fingerprint:", product_var)
        add_label_entry("CPU hint:", cpu_hint_var)
        add_label_entry("Method name:", method_name_var)
        add_label_entry("Method id:", method_id_var)
        add_label_entry("UEFI var name:", uefi_var_var)
        add_label_entry("Offset (hex):", offset_var)
        add_label_entry("Enable value (hex):", enable_var)
        add_label_entry("Disable value (hex):", disable_var)
        add_label_entry("Danger (1-5):", danger_var)
        add_label_entry("Notes:", notes_var)

        def on_save():
            family = fam_var.get().strip()
            board = board_var.get().strip()
            if not family or not board:
                messagebox.showerror("Error", "Family and board name are required.")
                return
            try:
                _validate_hex("Offset", offset_var.get())
                _validate_hex("Enable value", enable_var.get())
                _validate_hex("Disable value", disable_var.get())
                d = int(danger_var.get())
                if d < 1 or d > 5:
                    raise ValueError("Danger out of range")
            except Exception as e:
                messagebox.showerror("Error", f"Validation failed:\n{e}")
                return

            method = {
                "name": method_name_var.get().strip() or "GRUB setup_var",
                "id": method_id_var.get().strip() or "grub_setup_var",
                "uefi_var": uefi_var_var.get().strip() or "Setup",
                "offset_hex": offset_var.get().strip(),
                "value_enable_hex": enable_var.get().strip(),
                "value_disable_hex": disable_var.get().strip(),
                "danger": d,
                "notes": notes_var.get().strip(),
            }
            entry = {
                "board_name": board,
                "vendor_hint": vendor_var.get().strip(),
                "product_hint": product_var.get().strip(),
                "cpu_hint": cpu_hint_var.get().strip(),
                "fingerprint": {
                    "vendor": vendor_var.get().strip(),
                    "product": product_var.get().strip(),
                },
                "methods": [method],
            }

            if family not in self.config:
                self.config[family] = []
            self.config[family].append(entry)
            self._refresh_families()
            self.family_var.set(family)
            self._refresh_boards()
            self.board_var.set(board)
            self._refresh_methods()
            self._update_method_preview()
            self._set_status(f"Profile added: {family} / {board}")
            builder.destroy()

        ttk.Button(builder, text="Save profile", command=on_save).grid(row=row, column=0, columnspan=2, pady=10)

    # --- public API / integration hooks -------------------------------------

    def get_config(self):
        return deepcopy(self.config)

    def set_config(self, new_config):
        self.config = deepcopy(new_config)
        self._refresh_families()
        self._refresh_boards()
        self._set_status("Config replaced via integration hook.")

    def run(self):
        if self._own_root:
            self.root.mainloop()

# --- main --------------------------------------------------------------------

def main():
    organ = FirmwarePolicyOrgan()
    organ.run()

if __name__ == "__main__":
    main()
