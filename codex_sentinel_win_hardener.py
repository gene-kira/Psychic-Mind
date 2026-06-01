#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# codex_sentinel_unified_tk.py
#
# Unified:
#   - Zero-day-resistant Windows hardener (home + server)
#   - Policy pack system (versioned)
#   - Telemetry export (JSON)
#   - Node→swarm sync protocol (skeleton)
#   - Policy diff visualizer
#   - Tkinter “Futuristic Cockpit HUD” GUI (default mode)
#   - CLI fallback with --cli
#   - Hooks for organism/swarm integration
#
# GUI (default):
#   python codex_sentinel_unified_tk.py
#
# CLI:
#   python codex_sentinel_unified_tk.py --cli --profile home --dry-run --export-telemetry
#

import argparse
import ctypes
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from textwrap import dedent

import tkinter as tk
from tkinter import ttk, filedialog

# Optional imports
try:
    import winreg  # type: ignore
except ImportError:
    winreg = None

try:
    import requests  # type: ignore
except ImportError:
    requests = None


# === AUTO-ELEVATION CHECK ===
def ensure_admin():
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


ensure_admin()


# ---------------------------
# Policy packs (versioned)
# ---------------------------

POLICY_PACKS = {
    "home-1.0": {
        "profile": "home",
        "version": "1.0",
        "modules": [
            "defender",
            "smartscreen",
            "rdp",
            "smb_legacy",
            "firewall",
            "app_control_prep",
            "boot_bitlocker",
            "identity_lsa",
        ],
    },
    "server-1.0": {
        "profile": "server",
        "version": "1.0",
        "modules": [
            "defender",
            "smartscreen",
            "rdp",
            "smb_legacy",
            "firewall",
            "app_control_prep",
            "boot_bitlocker",
            "identity_lsa",
        ],
    },
}


# ---------------------------
# Swarm / sync configuration
# ---------------------------

SWARM_CONFIG = {
    "enabled": False,          # set True when you wire it up
    "base_url": "https://swarm.example.local:9000",
    "api_key": "CHANGE_ME",    # placeholder
}


# ---------------------------
# Utility / plumbing
# ---------------------------

def is_windows():
    return platform.system().lower() == "windows"


def run_ps(command, dry_run=False):
    print(f"[PS] {command}")
    if dry_run:
        return 0, "", ""
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def set_reg_value(root, path, name, value, reg_type, dry_run=False):
    if winreg is None or root is None:
        print("[REG] winreg not available; skipping registry operation.")
        return

    print(f"[REG] {root}\\{path} :: {name} = {value} ({reg_type})")
    if dry_run:
        return

    try:
        key = winreg.CreateKey(root, path)
        winreg.SetValueEx(key, name, 0, reg_type, value)
        winreg.CloseKey(key)
    except Exception as e:
        print(f"[REG] Failed to set {path}\\{name}: {e}")


def get_reg_value(root, path, name, default=None):
    if winreg is None or root is None:
        return default
    try:
        key = winreg.OpenKey(root, path, 0, winreg.KEY_READ)
        value, _ = winreg.QueryValueEx(key, name)
        winreg.CloseKey(key)
        return value
    except Exception:
        return default


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


# ---------------------------
# Organism / swarm hooks (stubs)
# ---------------------------

def organism_notify(event, data=None):
    print(f"[ORGANISM] Event: {event}, Data: {str(data)[:200]}")


# ---------------------------
# AI / “borg tech” hooks (stubs)
# ---------------------------

def ai_score_system_risk(context, telemetry=None):
    score = 0.42
    print(f"[AI] (stub) risk score for context '{context}': {score}")
    if telemetry is not None:
        try:
            size = len(json.dumps(telemetry))
        except Exception:
            size = 0
        print(f"[AI] (stub) telemetry size: {size} bytes")
    return score


# ---------------------------
# Telemetry collection
# ---------------------------

def collect_telemetry(profile, policy_pack_name, policy_pack):
    hostname = socket.gethostname()
    timestamp = int(time.time())

    telemetry = {
        "meta": {
            "hostname": hostname,
            "timestamp": timestamp,
            "profile": profile,
            "policy_pack": policy_pack_name,
            "policy_version": policy_pack.get("version"),
            "os": platform.platform(),
        },
        "defender": {},
        "firewall": {},
        "identity_lsa": {},
        "app_control": {},
        "smb_legacy": {},
    }

    telemetry["defender"]["PUAProtection"] = get_reg_value(
        winreg.HKEY_LOCAL_MACHINE if winreg else None,
        r"SOFTWARE\Microsoft\Windows Defender\MpEngine",
        "PUAProtection",
        default="unknown",
    )

    telemetry["identity_lsa"]["RunAsPPL"] = get_reg_value(
        winreg.HKEY_LOCAL_MACHINE if winreg else None,
        r"SYSTEM\CurrentControlSet\Control\Lsa",
        "RunAsPPL",
        default="unknown",
    )

    telemetry["app_control"]["AuditMode"] = get_reg_value(
        winreg.HKEY_LOCAL_MACHINE if winreg else None,
        r"SYSTEM\CurrentControlSet\Control\CI\Policy",
        "AuditMode",
        default="unknown",
    )

    telemetry["smb_legacy"]["SMB1Protocol"] = get_reg_value(
        winreg.HKEY_LOCAL_MACHINE if winreg else None,
        r"SYSTEM\CurrentControlSet\Services\LanmanServer\Parameters",
        "SMB1",
        default="unknown",
    )

    try:
        code, out, _ = run_ps(
            "Get-NetFirewallProfile | Select-Object Name, Enabled | ConvertTo-Json",
            dry_run=False,
        )
        if code == 0 and out:
            telemetry["firewall"]["profiles"] = json.loads(out)
    except Exception:
        telemetry["firewall"]["profiles"] = "unknown"

    return telemetry


def export_telemetry(telemetry, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codex_telemetry")

    os.makedirs(output_dir, exist_ok=True)

    hostname = telemetry["meta"]["hostname"]
    timestamp = telemetry["meta"]["timestamp"]
    filename = f"telemetry_{hostname}_{timestamp}.json"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print(f"[TELEMETRY] Exported to: {path}")
    return path


# ---------------------------
# Swarm sync protocol (skeleton)
# ---------------------------

def swarm_upload_telemetry(telemetry):
    if not SWARM_CONFIG["enabled"]:
        print("[SWARM] Upload disabled (SWARM_CONFIG.enabled = False).")
        return None

    if requests is None:
        print("[SWARM] 'requests' not installed; cannot upload telemetry.")
        return None

    url = SWARM_CONFIG["base_url"].rstrip("/") + "/upload_telemetry"
    headers = {"X-API-Key": SWARM_CONFIG["api_key"]}
    try:
        resp = requests.post(url, headers=headers, json=telemetry, timeout=10)
        print(f"[SWARM] Upload status: {resp.status_code}")
        return resp.json()
    except Exception as e:
        print(f"[SWARM] Upload failed: {e}")
        return None


def swarm_fetch_policy_pack():
    if not SWARM_CONFIG["enabled"]:
        print("[SWARM] Fetch disabled (SWARM_CONFIG.enabled = False).")
        return None

    if requests is None:
        print("[SWARM] 'requests' not installed; cannot fetch policy pack.")
        return None

    url = SWARM_CONFIG["base_url"].rstrip("/") + "/suggest_policy_pack"
    headers = {"X-API-Key": SWARM_CONFIG["api_key"]}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"[SWARM] Fetch status: {resp.status_code}")
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        print(f"[SWARM] Fetch failed: {e}")
        return None


# ---------------------------
# Hardening primitives
# ---------------------------

def harden_defender(dry_run=False):
    print_header("Defender / security stack hardening")

    run_ps("Set-MpPreference -PUAProtection Enabled", dry_run)
    run_ps("Set-MpPreference -MAPSReporting Advanced", dry_run)
    run_ps("Set-MpPreference -SubmitSamplesConsent SendSafeSamples", dry_run)
    run_ps("Set-MpPreference -DisableRealtimeMonitoring $false", dry_run)
    run_ps("Set-MpPreference -DisableBehaviorMonitoring $false", dry_run)
    run_ps("Set-MpPreference -DisableIOAVProtection $false", dry_run)
    run_ps("Set-MpPreference -DisableScriptScanning $false", dry_run)
    run_ps("Set-MpPreference -EnableNetworkProtection Enabled", dry_run)

    ai_score_system_risk("defender_hardening")
    organism_notify("defender_hardened")


def harden_smart_screen(dry_run=False):
    print_header("SmartScreen hardening")

    if winreg:
        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer",
            "SmartScreenEnabled",
            "RequireAdmin",
            winreg.REG_SZ,
            dry_run
        )

        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Policies\Microsoft\MicrosoftEdge\PhishingFilter",
            "EnabledV9",
            1,
            winreg.REG_DWORD,
            dry_run
        )

    ai_score_system_risk("smartscreen_hardening")
    organism_notify("smartscreen_hardened")


def harden_rdp(profile, dry_run=False):
    print_header("RDP / remote access hardening")

    if profile == "home":
        run_ps("Set-ItemProperty -Path 'HKLM:\\System\\CurrentControlSet\\Control\\Terminal Server' "
               "-Name 'fDenyTSConnections' -Value 1", dry_run)
        run_ps("Set-ItemProperty -Path 'HKLM:\\System\\CurrentControlSet\\Control\\Remote Assistance' "
               "-Name 'fAllowToGetHelp' -Value 0", dry_run)
    else:
        run_ps("Set-ItemProperty -Path 'HKLM:\\System\\CurrentControlSet\\Control\\Terminal Server\\WinStations\\RDP-Tcp' "
               "-Name 'UserAuthentication' -Value 1", dry_run)

    ai_score_system_risk(f"rdp_hardening_{profile}")
    organism_notify("rdp_hardened", {"profile": profile})


def harden_smb_and_legacy(dry_run=False):
    print_header("SMB / legacy protocol hardening")

    run_ps("Set-SmbServerConfiguration -EnableSMB1Protocol $false -Force", dry_run)
    run_ps("Disable-WindowsOptionalFeature -Online -FeatureName SMB1Protocol -NoRestart", dry_run)

    if winreg:
        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Services\Dnscache\Parameters",
            "EnableMulticast",
            0,
            winreg.REG_DWORD,
            dry_run
        )

    ai_score_system_risk("smb_legacy_hardening")
    organism_notify("smb_legacy_hardened")


def harden_firewall(profile, dry_run=False):
    print_header("Firewall hardening")

    run_ps("Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True", dry_run)

    if profile == "home":
        run_ps("Set-NetFirewallProfile -Profile Public,Private -DefaultInboundAction Block", dry_run)
    else:
        run_ps("Set-NetFirewallProfile -Profile Domain -DefaultInboundAction Block", dry_run)

    run_ps("Set-NetFirewallProfile -Profile Domain,Public,Private "
           "-LogAllowed True -LogBlocked True -LogFileName '%systemroot%\\system32\\LogFiles\\Firewall\\pfirewall.log'",
           dry_run)

    ai_score_system_risk(f"firewall_hardening_{profile}")
    organism_notify("firewall_hardened", {"profile": profile})


def prepare_app_control(dry_run=False):
    print_header("Application control (WDAC) preparation")

    if winreg:
        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\CI\Policy",
            "AuditMode",
            1,
            winreg.REG_DWORD,
            dry_run
        )

    print("[INFO] WDAC/app-control is NOT fully enforced by this script.")
    print("[INFO] It only prepares audit mode so you can build a safe allowlist.")
    ai_score_system_risk("app_control_prep")
    organism_notify("app_control_prep")


def harden_boot_and_bitlocker(profile, dry_run=False):
    print_header("Boot chain / BitLocker posture")

    cmd = "bcdedit /set {globalsettings} bootux disabled"
    print(f"[SUGGEST] To reduce external boot surface, consider: {cmd}")

    if profile == "home":
        print("[SUGGEST] On high-risk laptops, consider disabling WinRE temporarily:")
        print("          reagentc /disable")

    print("[SUGGEST] If BitLocker is enabled, configure a pre-boot PIN for stronger physical-access resistance.")
    ai_score_system_risk(f"boot_bitlocker_posture_{profile}")
    organism_notify("boot_bitlocker_posture", {"profile": profile})


def harden_identity_and_lsa(dry_run=False):
    print_header("Identity / LSA hardening")

    if winreg:
        set_reg_value(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Lsa",
            "RunAsPPL",
            1,
            winreg.REG_DWORD,
            dry_run
        )

    print("[SUGGEST] Enable Credential Guard via Group Policy or Device Guard settings where supported.")
    ai_score_system_risk("identity_lsa_hardening")
    organism_notify("identity_lsa_hardened")


# ---------------------------
# Policy pack application
# ---------------------------

def apply_policy_pack(policy_pack_name, dry_run=False):
    if policy_pack_name not in POLICY_PACKS:
        raise ValueError(f"Unknown policy pack: {policy_pack_name}")

    pack = POLICY_PACKS[policy_pack_name]
    profile = pack["profile"]
    modules = pack["modules"]

    print_header(f"APPLYING POLICY PACK: {policy_pack_name} (profile={profile}, version={pack['version']})")

    for module in modules:
        if module == "defender":
            harden_defender(dry_run=dry_run)
        elif module == "smartscreen":
            harden_smart_screen(dry_run=dry_run)
        elif module == "rdp":
            harden_rdp(profile=profile, dry_run=dry_run)
        elif module == "smb_legacy":
            harden_smb_and_legacy(dry_run=dry_run)
        elif module == "firewall":
            harden_firewall(profile=profile, dry_run=dry_run)
        elif module == "app_control_prep":
            prepare_app_control(dry_run=dry_run)
        elif module == "boot_bitlocker":
            harden_boot_and_bitlocker(profile=profile, dry_run=dry_run)
        elif module == "identity_lsa":
            harden_identity_and_lsa(dry_run=dry_run)
        else:
            print(f"[WARN] Unknown module in policy pack: {module}")

    print_header(f"POLICY PACK {policy_pack_name} COMPLETE")
    organism_notify("policy_pack_applied", {"pack": policy_pack_name, "dry_run": dry_run})
    return pack


# ---------------------------
# Policy diff visualizer
# ---------------------------

def diff_policy_packs(local_pack, remote_pack):
    lines = []
    lines.append(f"Local version : {local_pack.get('version')}")
    lines.append(f"Remote version: {remote_pack.get('version')}")
    lines.append("")

    local_modules = set(local_pack.get("modules", []))
    remote_modules = set(remote_pack.get("modules", []))

    added = remote_modules - local_modules
    removed = local_modules - remote_modules
    common = local_modules & remote_modules

    if added:
        lines.append("Modules added in remote:")
        for m in sorted(added):
            lines.append(f"  + {m}")
        lines.append("")

    if removed:
        lines.append("Modules removed in remote:")
        for m in sorted(removed):
            lines.append(f"  - {m}")
        lines.append("")

    if common:
        lines.append("Modules in both (unchanged or internally different):")
        for m in sorted(common):
            lines.append(f"  = {m}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------
# Tkinter Futuristic Cockpit HUD
# ---------------------------

class CodexHUD(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Codex Sentinel — Zero‑Day Hardening Cockpit")
        self.geometry("1200x750")
        self.configure(bg="#05060A")

        self.current_telemetry = None
        self.last_swarm_policy = None

        self._build_style()
        self._build_layout()

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#05060A")
        style.configure("TLabel", background="#05060A", foreground="#E0E0E0", font=("Consolas", 10))
        style.configure("Title.TLabel", font=("Consolas", 16, "bold"), foreground="#00FFC8")
        style.configure("Section.TLabel", font=("Consolas", 11, "bold"), foreground="#00BFFF")
        style.configure("TButton", background="#101320", foreground="#E0E0E0", font=("Consolas", 10))
        style.map("TButton",
                  background=[("active", "#1A2035")],
                  foreground=[("active", "#FFFFFF")])
        style.configure("TCombobox", fieldbackground="#101320", background="#101320", foreground="#E0E0E0")

    def _build_layout(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        title = ttk.Label(top_frame, text="Codex Sentinel — Zero‑Day Hardening Cockpit", style="Title.TLabel")
        title.pack(side=tk.LEFT, padx=5)

        right_status = ttk.Frame(top_frame)
        right_status.pack(side=tk.RIGHT)

        self.swarm_label = ttk.Label(right_status, text="Swarm: idle", foreground="#AAAAAA")
        self.swarm_label.pack(side=tk.RIGHT, padx=5)

        mid_frame = ttk.Frame(self)
        mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_panel = ttk.Frame(mid_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        right_panel = ttk.Frame(mid_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        pack_label = ttk.Label(left_panel, text="Policy Pack", style="Section.TLabel")
        pack_label.pack(anchor="w", pady=(0, 4))

        self.pack_var = tk.StringVar(value=list(POLICY_PACKS.keys())[0])
        self.pack_combo = ttk.Combobox(left_panel, textvariable=self.pack_var, values=list(POLICY_PACKS.keys()), state="readonly")
        self.pack_combo.pack(fill=tk.X, pady=(0, 10))

        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)

        self.apply_btn = ttk.Button(btn_frame, text="Apply Pack", command=self.on_apply_pack)
        self.apply_btn.pack(fill=tk.X, pady=2)

        self.telemetry_btn = ttk.Button(btn_frame, text="Collect Telemetry", command=self.on_collect_telemetry)
        self.telemetry_btn.pack(fill=tk.X, pady=2)

        self.export_telemetry_btn = ttk.Button(btn_frame, text="Export Telemetry", command=self.on_export_telemetry)
        self.export_telemetry_btn.pack(fill=tk.X, pady=2)

        self.swarm_sync_btn = ttk.Button(btn_frame, text="Sync with Swarm", command=self.on_swarm_sync)
        self.swarm_sync_btn.pack(fill=tk.X, pady=2)

        self.diff_btn = ttk.Button(btn_frame, text="Show Policy Diff", command=self.on_show_diff)
        self.diff_btn.pack(fill=tk.X, pady=2)

        ttk.Label(left_panel, text="Node HUD", style="Section.TLabel").pack(anchor="w", pady=(15, 4))

        self.node_info = tk.Text(left_panel, height=10, bg="#05060A", fg="#00FFC8",
                                 insertbackground="#00FFC8", relief=tk.FLAT, font=("Consolas", 9))
        self.node_info.pack(fill=tk.BOTH, expand=True)
        self._update_node_info()

        upper_right = ttk.Frame(right_panel)
        upper_right.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        lower_right = ttk.Frame(right_panel)
        lower_right.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(5, 0))

        ttk.Label(upper_right, text="Telemetry Snapshot", style="Section.TLabel").pack(anchor="w")

        self.telemetry_view = tk.Text(upper_right, bg="#05060A", fg="#E0E0E0",
                                      insertbackground="#E0E0E0", relief=tk.FLAT, font=("Consolas", 9))
        self.telemetry_view.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        ttk.Label(lower_right, text="Event Log", style="Section.TLabel").pack(anchor="w")

        self.log_view = tk.Text(lower_right, bg="#05060A", fg="#00BFFF",
                                insertbackground="#00BFFF", relief=tk.FLAT, font=("Consolas", 9))
        self.log_view.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        self._log_banner()

    def _log_banner(self):
        self.log("[Codex Sentinel] HUD online. Awaiting orders, operator.")

    def _update_node_info(self):
        hostname = socket.gethostname()
        os_str = platform.platform()
        self.node_info.delete("1.0", tk.END)
        self.node_info.insert(tk.END, f"Host: {hostname}\n")
        self.node_info.insert(tk.END, f"OS  : {os_str}\n")
        self.node_info.insert(tk.END, f"Mode: GUI cockpit\n")
        self.node_info.insert(tk.END, f"Policy packs: {', '.join(POLICY_PACKS.keys())}\n")

    def log(self, text):
        self.log_view.insert(tk.END, text + "\n")
        self.log_view.see(tk.END)

    def on_apply_pack(self):
        pack_name = self.pack_var.get()
        self.log(f"[ACTION] Applying policy pack: {pack_name}")

        def worker():
            try:
                apply_policy_pack(pack_name, dry_run=False)
                self.log(f"[OK] Policy pack {pack_name} applied.")
            except Exception as e:
                self.log(f"[ERROR] Failed to apply pack: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def on_collect_telemetry(self):
        pack_name = self.pack_var.get()
        pack = POLICY_PACKS[pack_name]
        profile = pack["profile"]

        self.log("[ACTION] Collecting telemetry...")
        telemetry = collect_telemetry(profile, pack_name, pack)
        self.current_telemetry = telemetry
        self.telemetry_view.delete("1.0", tk.END)
        self.telemetry_view.insert(tk.END, json.dumps(telemetry, indent=2))
        self.log("[OK] Telemetry collected.")

    def on_export_telemetry(self):
        if not self.current_telemetry:
            self.log("[WARN] No telemetry in memory; collecting first.")
            self.on_collect_telemetry()
        if not self.current_telemetry:
            self.log("[ERROR] Telemetry collection failed.")
            return

        path = export_telemetry(self.current_telemetry)
        self.log(f"[OK] Telemetry exported to: {path}")

    def on_swarm_sync(self):
        if not self.current_telemetry:
            self.log("[WARN] No telemetry in memory; collecting first.")
            self.on_collect_telemetry()
        if not self.current_telemetry:
            self.log("[ERROR] Telemetry collection failed.")
            return

        self.log("[ACTION] Uploading telemetry to swarm (if enabled)...")

        def worker():
            resp = swarm_upload_telemetry(self.current_telemetry)
            self.log(f"[SWARM] Upload response: {resp}")
            self.swarm_label.config(text="Swarm: telemetry uploaded")

            self.log("[ACTION] Fetching suggested policy pack from swarm (if enabled)...")
            policy_resp = swarm_fetch_policy_pack()
            if policy_resp and "policy" in policy_resp:
                self.last_swarm_policy = policy_resp["policy"]
                self.log(f"[SWARM] Received policy: {self.last_swarm_policy.get('version')}")
                self.swarm_label.config(text="Swarm: policy received")
            else:
                self.log("[SWARM] No policy received.")
                self.swarm_label.config(text="Swarm: no policy")

        threading.Thread(target=worker, daemon=True).start()

    def on_show_diff(self):
        pack_name = self.pack_var.get()
        local_pack = POLICY_PACKS[pack_name]

        if not self.last_swarm_policy:
            self.log("[WARN] No swarm policy cached. Run 'Sync with Swarm' first.")
            return

        diff_text = diff_policy_packs(local_pack, self.last_swarm_policy)
        self.log("[DIFF] Local vs Swarm policy:")
        self.log(diff_text)


# ---------------------------
# Main
# ---------------------------

def main():
    if not is_windows():
        print("This script is intended for Windows only.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Codex Sentinel Unified (Tk HUD): Hardener + GUI + Telemetry + Swarm sync."
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of GUI."
    )
    parser.add_argument(
        "--profile",
        choices=["home", "server"],
        help="Which profile to apply (CLI mode)."
    )
    parser.add_argument(
        "--policy-pack",
        help="Policy pack name (default: <profile>-1.0 in CLI mode)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done, but do not change anything (CLI mode)."
    )
    parser.add_argument(
        "--export-telemetry",
        action="store_true",
        help="Export telemetry JSON after applying (or simulating) the policy pack (CLI mode)."
    )

    args = parser.parse_args()

    if not args.cli:
        app = CodexHUD()
        app.mainloop()
        return

    if not args.profile:
        print("CLI mode requires --profile (home or server).")
        sys.exit(1)

    default_pack = f"{args.profile}-1.0"
    policy_pack_name = args.policy_pack or default_pack

    print(dedent(f"""
        codex_sentinel_unified_tk.py
        Mode         : CLI
        Profile      : {args.profile}
        Policy pack  : {policy_pack_name}
        Dry-run      : {args.dry_run}
        Telemetry    : {args.export_telemetry}
    """).strip())

    pack = apply_policy_pack(policy_pack_name, dry_run=args.dry_run)

    if args.export_telemetry:
        telemetry = collect_telemetry(args.profile, policy_pack_name, pack)
        path = export_telemetry(telemetry)
        ai_score_system_risk("post_hardening_telemetry", telemetry=telemetry)
        print(f"[INFO] Telemetry exported to: {path}")


if __name__ == "__main__":
    main()
