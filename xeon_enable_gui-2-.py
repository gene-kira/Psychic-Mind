#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Firmware Policy Organ: Xeon-enable command generator cockpit

- No direct NVRAM/firmware writes.
- Smarter: tries to auto-detect CPU + board and auto-select profile.
- Operator-grade: dark-ish theme, status bar, log pane, profile inspector.
- Autonomous: validation, sanity checks, basic diff-style info.
- More complete Xeon support: multi-method schema (GRUB, RU.EFI, Chipsec templates).
- Integrated organ: can be run standalone or imported as part of a larger system.
"""

import sys
import platform
import traceback
from datetime import datetime

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

# Optional helpers for detection
wmi = _require("wmi", lambda: __import__("wmi"), friendly_name="wmi", optional=True)
cpuinfo = _require("cpuinfo", lambda: __import__("cpuinfo"), friendly_name="py-cpuinfo", optional=True)

# --- configuration schema ----------------------------------------------------
# You fill in real offsets/values per board after manual reverse/validation.
# This file NEVER guesses them for you.

CONFIG = {
    "Skylake Xeon E3 v5/v6": [
        {
            "board_name": "Example-Z170-Skylake",
            "vendor_hint": "ASUS",
            "product_hint": "Z170",
            "cpu_hint": "E3-12",
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x123",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "notes": "Tested manually with setup_var.efi on Example Z170 board."
                },
                {
                    "name": "RU.EFI template",
                    "id": "ru_efi",
                    "uefi_var": "Setup",
                    "offset_hex": "0x123",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
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
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x234",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
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
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x345",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
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
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x456",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "notes": "W-series workstation board; verify with UEFITool/IFR."
                },
                {
                    "name": "Chipsec template",
                    "id": "chipsec",
                    "uefi_var": "Setup",
                    "offset_hex": "0x456",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
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
            "methods": [
                {
                    "name": "GRUB setup_var",
                    "id": "grub_setup_var",
                    "uefi_var": "Setup",
                    "offset_hex": "0x567",
                    "value_enable_hex": "0x01",
                    "value_disable_hex": "0x00",
                    "notes": "ES/QS profile; microcode/stepping dependent."
                },
            ],
        },
    ],
}

FAMILIES = list(CONFIG.keys())

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

def detect_board_info():
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
    # On non-Windows, you could add dmidecode parsing here manually if you want.
    return vendor, product

def auto_select_profile():
    cpu_str = detect_cpu_string().upper()
    vendor, product = detect_board_info()
    vendor_u = vendor.upper()
    product_u = product.upper()

    best_family = None
    best_board = None

    for family, boards in CONFIG.items():
        for entry in boards:
            vh = (entry.get("vendor_hint") or "").upper()
            ph = (entry.get("product_hint") or "").upper()
            ch = (entry.get("cpu_hint") or "").upper()

            score = 0
            if vh and vh in vendor_u:
                score += 2
            if ph and ph in product_u:
                score += 2
            if ch and ch in cpu_str:
                score += 2

            if score > 0:
                if best_family is None or score > best_family[2]:
                    best_family = (family, entry["board_name"], score)

    if best_family:
        return best_family[0], best_family[1]
    return None, None

# --- validation & command generation ----------------------------------------

def _validate_hex(label, value):
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    v = value.strip()
    if not v.startswith("0x") and not v.startswith("0X"):
        raise ValueError(f"{label} must start with 0x (got: {value})")
    try:
        int(v, 16)
    except ValueError:
        raise ValueError(f"{label} is not valid hex: {value}")
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

    value = val_enable if mode == "enable" else val_disable

    header = []
    header.append(f"# Family : {family}")
    header.append(f"# Board  : {board_entry.get('board_name')}")
    header.append(f"# Method : {method_entry.get('name')} ({method_id})")
    header.append(f"# Var    : {uefi_var}")
    header.append(f"# Offset : {offset}")
    header.append(f"# Mode   : {mode}")
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

# --- GUI organ --------------------------------------------------------------

class FirmwarePolicyOrgan:
    def __init__(self, root=None):
        self._own_root = False
        if root is None:
            root = tk.Tk()
            self._own_root = True
        self.root = root
        self.root.title("Firmware Policy Organ – Xeon Enable Cockpit")
        self.root.geometry("1000x650")

        self.family_var = tk.StringVar(value=FAMILIES[0] if FAMILIES else "")
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

        # Top info + selectors
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
            values=FAMILIES,
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

        # Output + log
        center = ttk.Frame(main)
        center.pack(fill="both", expand=True)

        output_frame = ttk.Labelframe(center, text="Generated commands (run manually!)")
        output_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.output_text = tk.Text(output_frame, wrap="word", height=20, bg="#111111", fg="#e0e0e0", insertbackground="#e0e0e0")
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

        log_frame = ttk.Labelframe(center, text="Organ log")
        log_frame.pack(side="right", fill="both", expand=False)

        self.log_text = tk.Text(log_frame, wrap="word", width=40, height=20, bg="#111111", fg="#a0a0a0", insertbackground="#e0e0e0")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Status bar
        status_frame = ttk.Frame(main)
        status_frame.pack(fill="x", pady=(5, 0))
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x")

    # --- internal helpers ----------------------------------------------------

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")

    def _set_status(self, msg):
        self.status_var.set(msg)
        self._log(msg)

    def _refresh_boards(self):
        family = self.family_var.get()
        boards = CONFIG.get(family, [])
        names = [b["board_name"] for b in boards]
        self.board_combo["values"] = names
        if names:
            self.board_var.set(names[0])
        else:
            self.board_var.set("")
        self._refresh_methods()

    def _refresh_methods(self):
        board_entry = self._get_selected_board_entry()
        methods = ["All methods"]
        if board_entry:
            for m in board_entry.get("methods", []):
                methods.append(m.get("name"))
        self.method_combo["values"] = methods
        if methods:
            self.method_var.set(methods[0])

    def _get_selected_board_entry(self):
        family = self.family_var.get()
        board_name = self.board_var.get()
        for entry in CONFIG.get(family, []):
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
        fam, board = auto_select_profile()
        if fam and board:
            self.family_var.set(fam)
            self._refresh_boards()
            self.board_var.set(board)
            self._refresh_methods()
            self._set_status(f"Auto-selected profile: {fam} / {board}")
        else:
            self._set_status("No auto-selection match; choose family/board manually.")

    # --- event handlers ------------------------------------------------------

    def _on_family_change(self, event=None):
        self._refresh_boards()
        self._set_status(f"Family changed to: {self.family_var.get()}")

    def _on_board_change(self, event=None):
        self._refresh_methods()
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
        lines.append("")
        lines.append("Methods:")
        for m in board_entry.get("methods", []):
            lines.append(f"  - {m.get('name')} ({m.get('id')})")
            lines.append(f"      Var    : {m.get('uefi_var')}")
            lines.append(f"      Offset : {m.get('offset_hex')}")
            lines.append(f"      Enable : {m.get('value_enable_hex')}")
            lines.append(f"      Disable: {m.get('value_disable_hex')}")
            if m.get("notes"):
                lines.append(f"      Notes  : {m.get('notes')}")
        info = "\n".join(lines)
        messagebox.showinfo("Profile details", info)
        self._set_status("Profile inspected.")

    # --- public API ----------------------------------------------------------

    def run(self):
        if self._own_root:
            self.root.mainloop()

# --- main --------------------------------------------------------------------

def main():
    organ = FirmwarePolicyOrgan()
    organ.run()

if __name__ == "__main__":
    main()
