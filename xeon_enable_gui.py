#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xeon-enable command generator cockpit

- No direct NVRAM/firmware writes.
- You select: family → board/profile → mode (enable/restore).
- It generates commands (e.g. GRUB setup_var) for you to run manually.
"""

# --- autoloader for required libraries --------------------------------------
import sys

def _require(module_name, import_expr, friendly_name=None):
    try:
        return import_expr()
    except ImportError as e:
        print(f"[FATAL] Missing Python module: {module_name}")
        if friendly_name:
            print(f"        Install it with: pip install {friendly_name}")
        else:
            print(f"        Install it with: pip install {module_name}")
        sys.exit(1)

tk = _require("tkinter", lambda: __import__("tkinter"))
ttk = _require("tkinter.ttk", lambda: __import__("tkinter.ttk", fromlist=["*"]))
messagebox = _require("tkinter.messagebox", lambda: __import__("tkinter.messagebox", fromlist=["*"]))

# --- configuration schema ----------------------------------------------------
# You fill in real offsets/values per board after manual reverse/validation.
# This file NEVER guesses them for you.

CONFIG = {
    "Skylake Xeon E3 v5/v6": [
        {
            "board_name": "Example-Z170-Skylake",
            "uefi_var": "Setup",
            "method": "grub_setup_var",
            "offset_hex": "0x123",          # replace with real offset
            "value_enable_hex": "0x01",     # Xeon enabled
            "value_disable_hex": "0x00",    # stock
            "notes": "Tested manually with setup_var.efi on Example Z170 board."
        },
    ],
    "Kaby Lake Xeon": [
        {
            "board_name": "Example-Z270-Kaby",
            "uefi_var": "Setup",
            "method": "grub_setup_var",
            "offset_hex": "0x234",
            "value_enable_hex": "0x01",
            "value_disable_hex": "0x00",
            "notes": "Kaby Xeon E3 profile; confirm IFR before use."
        },
    ],
    "Coffee Lake Xeon": [
        {
            "board_name": "Example-Z370-Coffee",
            "uefi_var": "Setup",
            "method": "grub_setup_var",
            "offset_hex": "0x345",
            "value_enable_hex": "0x01",
            "value_disable_hex": "0x00",
            "notes": "Coffee Xeon/W profile; board-specific."
        },
    ],
    "W-series Xeon": [
        {
            "board_name": "Example-W480-Workstation",
            "uefi_var": "Setup",
            "method": "grub_setup_var",
            "offset_hex": "0x456",
            "value_enable_hex": "0x01",
            "value_disable_hex": "0x00",
            "notes": "W-series workstation board; verify with UEFITool/IFR."
        },
    ],
    "LGA1151 ES/QS Xeon": [
        {
            "board_name": "Example-ES-QS-1151",
            "uefi_var": "Setup",
            "method": "grub_setup_var",
            "offset_hex": "0x567",
            "value_enable_hex": "0x01",
            "value_disable_hex": "0x00",
            "notes": "ES/QS profile; microcode/stepping dependent."
        },
    ],
}

FAMILIES = list(CONFIG.keys())

MODES = {
    "Enable Xeon features": "enable",
    "Restore stock / disable Xeon": "disable",
}

# --- command generation core -------------------------------------------------

def generate_commands(family, board_entry, mode_key):
    """
    Build human-run commands for the selected profile.
    No direct firmware writes here.
    """
    mode = MODES.get(mode_key)
    if mode not in ("enable", "disable"):
        raise ValueError("Invalid mode")

    method = board_entry.get("method")
    uefi_var = board_entry.get("uefi_var")
    offset = board_entry.get("offset_hex")
    val_enable = board_entry.get("value_enable_hex")
    val_disable = board_entry.get("value_disable_hex")
    notes = board_entry.get("notes", "")

    if method == "grub_setup_var":
        value = val_enable if mode == "enable" else val_disable
        lines = []
        lines.append("# --- GRUB setup_var method ---")
        lines.append("# Boot a GRUB shell with setup_var support (or setup_var.efi).")
        lines.append("# Then run the following command(s) manually:")
        lines.append("")
        lines.append(f"# Family : {family}")
        lines.append(f"# Board  : {board_entry.get('board_name')}")
        lines.append(f"# Var    : {uefi_var}")
        lines.append(f"# Offset : {offset}")
        lines.append(f"# Mode   : {mode}")
        if notes:
            lines.append(f"# Notes  : {notes}")
        lines.append("")
        # Classic setup_var syntax: setup_var <offset> <value>
        lines.append(f"setup_var {offset} {value}")
        lines.append("")
        lines.append("# After running, reboot and verify POST/CPU enumeration.")
        return "\n".join(lines)

    else:
        # Placeholder for other methods (RU.EFI, Chipsec, etc.)
        lines = []
        lines.append("# Unsupported method in this profile.")
        lines.append("# Fill in method-specific command generation logic.")
        lines.append(f"# Method: {method}")
        if notes:
            lines.append(f"# Notes : {notes}")
        return "\n".join(lines)

# --- GUI ---------------------------------------------------------------------

class XeonEnableGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Xeon Enable Command Generator")
        self.root.geometry("800x500")

        self.family_var = tk.StringVar(value=FAMILIES[0] if FAMILIES else "")
        self.board_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value=list(MODES.keys())[0])

        self._build_layout()
        self._refresh_boards()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Top controls frame
        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 10))

        # Family selector
        ttk.Label(top, text="CPU family:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.family_combo = ttk.Combobox(
            top,
            textvariable=self.family_var,
            values=FAMILIES,
            state="readonly",
            width=30,
        )
        self.family_combo.grid(row=0, column=1, sticky="w")
        self.family_combo.bind("<<ComboboxSelected>>", self._on_family_change)

        # Board selector
        ttk.Label(top, text="Board/profile:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.board_combo = ttk.Combobox(
            top,
            textvariable=self.board_var,
            values=[],
            state="readonly",
            width=30,
        )
        self.board_combo.grid(row=1, column=1, sticky="w", pady=(5, 0))

        # Mode selector
        ttk.Label(top, text="Mode:").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.mode_combo = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=list(MODES.keys()),
            state="readonly",
            width=30,
        )
        self.mode_combo.grid(row=2, column=1, sticky="w", pady=(5, 0))

        # Buttons
        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=0, column=2, rowspan=3, padx=(20, 0), sticky="n")

        self.generate_btn = ttk.Button(btn_frame, text="Generate commands", command=self._on_generate)
        self.generate_btn.pack(fill="x", pady=(0, 5))

        self.copy_btn = ttk.Button(btn_frame, text="Copy to clipboard", command=self._on_copy)
        self.copy_btn.pack(fill="x")

        # Output text
        output_frame = ttk.LabelFrame(main, text="Generated commands (run manually!)")
        output_frame.pack(fill="both", expand=True)

        self.output_text = tk.Text(output_frame, wrap="word", height=20)
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

    def _refresh_boards(self):
        family = self.family_var.get()
        boards = CONFIG.get(family, [])
        names = [b["board_name"] for b in boards]
        self.board_combo["values"] = names
        if names:
            self.board_var.set(names[0])
        else:
            self.board_var.set("")

    def _on_family_change(self, event=None):
        self._refresh_boards()

    def _get_selected_board_entry(self):
        family = self.family_var.get()
        board_name = self.board_var.get()
        for entry in CONFIG.get(family, []):
            if entry.get("board_name") == board_name:
                return entry
        return None

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
            commands = generate_commands(family, board_entry, mode_key)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate commands:\n{e}")
            return

        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", commands)

    def _on_copy(self):
        text = self.output_text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Info", "Nothing to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Commands copied to clipboard.")

# --- main --------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = XeonEnableGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
