#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Workstation / Xeon Hardware Lab v3.1 (Unified)

Safe engineering-style workstation planning + simulation cockpit:

SMARTER:
- CPU / Board / GPU / Memory compatibility engine
- PCIe topology simulation (slots, lanes, mapping)
- VRM power modeling (headroom heuristic)
- BIOS version awareness (local "latest BIOS" table)
- Microcode awareness (CPU min vs board current)
- GPU database (TDP, PCIe requirements)
- Memory compatibility engine (speed, ECC, capacity)
- Realistic-ish chipset rule table
- Suggested build generator (GPU + memory auto-selection by workload)

MORE OPERATOR-GRADE:
- Tactical cockpit layout (sidebar + stacked panels)
- Animated panel transitions
- Live system fingerprint map
- Node sync simulation (logical, not networked)
- Timeline visualization

MORE OEM-STYLE:
- Chipset rules (heuristic but structured)
- Xeon workstation families
- BIOS constraints
- VRM current limits (modeled in board profiles)

All operations are read-only and simulation-based.
"""

import sys
import os
import json
import platform
import subprocess
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from PySide6 import QtWidgets, QtCore, QtGui

# Optional hardware detection libs
try:
    import cpuinfo
except Exception:
    cpuinfo = None

try:
    import wmi
except Exception:
    wmi = None

# -----------------------------
# BASE DATABASES
# -----------------------------

CPU_DB: Dict[str, Dict[str, Any]] = {
    "Xeon W-2295": {
        "socket": "LGA2066",
        "tdp": 165,
        "cores": 18,
        "supported_chipsets": ["C422", "X299"],
        "min_bios": 1103,
        "family": "Xeon W",
        "min_microcode": 0x5000020,
        "max_memory_speed": 2933,
        "ecc_support": True,
    },
    "Xeon E3-1270 v6": {
        "socket": "LGA1151",
        "tdp": 72,
        "cores": 4,
        "supported_chipsets": ["C236", "Z270"],
        "min_bios": 2000,
        "family": "Xeon E3",
        "min_microcode": 0x000000C2,
        "max_memory_speed": 2400,
        "ecc_support": True,
    },
    "Core i7-6700K": {
        "socket": "LGA1151",
        "tdp": 91,
        "cores": 4,
        "supported_chipsets": ["Z170", "Z270"],
        "min_bios": 1800,
        "family": "Core i7",
        "min_microcode": 0x000000C2,
        "max_memory_speed": 2133,
        "ecc_support": False,
    },
}

BOARD_DB: Dict[str, Dict[str, Any]] = {
    "ASUS WS C422 PRO": {
        "socket": "LGA2066",
        "chipset": "C422",
        "max_tdp": 200,
        "bios_version": 1201,
        "pci_lanes": 48,
        "vendor": "ASUSTeK",
        "product": "WS C422 PRO",
        "vrm_current_limit": 220,
        "microcode_version": 0x5000025,
        "max_memory_speed": 2933,
        "ecc_support": True,
        "memory_slots": 8,
        "max_memory_gb": 512,
        "pcie_slots": [
            {"name": "PCIEX16_1", "type": "x16", "lanes": 16, "wired_to": "CPU"},
            {"name": "PCIEX16_2", "type": "x16", "lanes": 16, "wired_to": "CPU"},
            {"name": "PCIEX16_3", "type": "x16", "lanes": 16, "wired_to": "PCH"},
        ],
    },
    "ASUS Z170-A": {
        "socket": "LGA1151",
        "chipset": "Z170",
        "max_tdp": 95,
        "bios_version": 3805,
        "pci_lanes": 16,
        "vendor": "ASUSTeK",
        "product": "Z170-A",
        "vrm_current_limit": 140,
        "microcode_version": 0x000000C6,
        "max_memory_speed": 3466,
        "ecc_support": False,
        "memory_slots": 4,
        "max_memory_gb": 64,
        "pcie_slots": [
            {"name": "PCIEX16_1", "type": "x16", "lanes": 16, "wired_to": "CPU"},
            {"name": "PCIEX16_2", "type": "x16", "lanes": 4, "wired_to": "PCH"},
            {"name": "PCIEX1_1", "type": "x1", "lanes": 1, "wired_to": "PCH"},
        ],
    },
    "ASUS W480": {
        "socket": "LGA1200",
        "chipset": "W480",
        "max_tdp": 125,
        "bios_version": 1001,
        "pci_lanes": 20,
        "vendor": "ASUSTeK",
        "product": "W480",
        "vrm_current_limit": 180,
        "microcode_version": 0x000000E0,
        "max_memory_speed": 2933,
        "ecc_support": True,
        "memory_slots": 4,
        "max_memory_gb": 128,
        "pcie_slots": [
            {"name": "PCIEX16_1", "type": "x16", "lanes": 16, "wired_to": "CPU"},
            {"name": "PCIEX4_1", "type": "x4", "lanes": 4, "wired_to": "PCH"},
        ],
    },
}

GPU_DB: Dict[str, Dict[str, Any]] = {
    "RTX 4090": {
        "tdp": 450,
        "slot_type": "x16",
        "min_lanes": 8,
        "recommended_psu": 1000,
        "tier": "ultra",
    },
    "RTX 3080": {
        "tdp": 320,
        "slot_type": "x16",
        "min_lanes": 8,
        "recommended_psu": 750,
        "tier": "high",
    },
    "Quadro RTX 4000": {
        "tdp": 160,
        "slot_type": "x16",
        "min_lanes": 8,
        "recommended_psu": 550,
        "tier": "pro",
    },
}

MEMORY_DB: Dict[str, Dict[str, Any]] = {
    "DDR4-2133 16GB ECC": {
        "type": "DDR4",
        "speed": 2133,
        "size_gb": 16,
        "ecc": True,
    },
    "DDR4-3200 16GB Non-ECC": {
        "type": "DDR4",
        "speed": 3200,
        "size_gb": 16,
        "ecc": False,
    },
    "DDR4-2933 32GB ECC": {
        "type": "DDR4",
        "speed": 2933,
        "size_gb": 32,
        "ecc": True,
    },
}

LATEST_BIOS: Dict[str, int] = {
    "ASUS WS C422 PRO": 1301,
    "ASUS Z170-A": 3805,
    "ASUS W480": 1202,
}

# Chipset rule table (heuristic OEM-style)
CHIPSET_RULES: Dict[str, Dict[str, Any]] = {
    "C422": {
        "families": ["Xeon W"],
        "ecc_required": True,
        "workloads": ["rendering", "virtualization", "balanced"],
    },
    "C236": {
        "families": ["Xeon E3", "Core i7"],
        "ecc_optional": True,
        "workloads": ["balanced", "virtualization"],
    },
    "Z170": {
        "families": ["Core i7"],
        "gaming_preferred": True,
        "workloads": ["gaming", "balanced"],
    },
    "Z270": {
        "families": ["Core i7", "Xeon E3"],
        "workloads": ["gaming", "balanced"],
    },
    "W480": {
        "families": ["Xeon W", "Core i7"],
        "ecc_optional": True,
        "workloads": ["balanced", "virtualization", "rendering"],
    },
    "X299": {
        "families": ["Xeon W", "Core i9"],
        "workloads": ["rendering", "balanced"],
    },
}

INVENTORY: List[Dict[str, Any]] = []
TIMELINE: List[str] = []

# -----------------------------
# HARDWARE DETECTION
# -----------------------------

class HardwareDetector:
    @staticmethod
    def detect_cpu() -> str:
        try:
            if cpuinfo:
                brand = cpuinfo.get_cpu_info().get("brand_raw")
                if brand:
                    return brand
        except Exception:
            pass

        proc = platform.processor()
        if proc:
            return proc
        return "Unknown CPU"

    @staticmethod
    def detect_board() -> Tuple[str, str]:
        if os.name == "nt" and wmi is not None:
            try:
                c = wmi.WMI()
                for b in c.Win32_BaseBoard():
                    vendor = getattr(b, "Manufacturer", "Unknown")
                    product = getattr(b, "Product", "Unknown")
                    return vendor, product
            except Exception:
                pass

        if os.name == "posix":
            try:
                out = subprocess.check_output(
                    ["dmidecode", "-t", "baseboard"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                vendor = "Unknown"
                product = "Unknown"
                for line in out.splitlines():
                    line = line.strip()
                    if line.startswith("Manufacturer:"):
                        vendor = line.split(":", 1)[1].strip()
                    elif line.startswith("Product Name:"):
                        product = line.split(":", 1)[1].strip()
                return vendor or "Unknown", product or "Unknown"
            except Exception:
                pass

        return "Unknown Vendor", "Unknown Board"

    @staticmethod
    def fingerprint() -> Dict[str, Any]:
        vendor, product = HardwareDetector.detect_board()
        cpu = HardwareDetector.detect_cpu()
        return {
            "cpu": cpu,
            "board_vendor": vendor,
            "board_product": product,
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
        }

# -----------------------------
# PROFILE DATABASE
# -----------------------------

class ProfileDB:
    def __init__(self):
        self.cpu_db = deepcopy(CPU_DB)
        self.board_db = deepcopy(BOARD_DB)
        self.gpu_db = deepcopy(GPU_DB)
        self.mem_db = deepcopy(MEMORY_DB)

    def get_cpu_names(self) -> List[str]:
        return sorted(self.cpu_db.keys())

    def get_board_names(self) -> List[str]:
        return sorted(self.board_db.keys())

    def get_gpu_names(self) -> List[str]:
        return sorted(self.gpu_db.keys())

    def get_mem_names(self) -> List[str]:
        return sorted(self.mem_db.keys())

    def add_cpu(self, name: str, profile: Dict[str, Any]):
        self.cpu_db[name] = profile

    def add_board(self, name: str, profile: Dict[str, Any]):
        self.board_db[name] = profile

    def export_to_file(self, path: str) -> None:
        data = {
            "cpus": self.cpu_db,
            "boards": self.board_db,
            "gpus": self.gpu_db,
            "memory": self.mem_db,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def import_from_file(self, path: str) -> Tuple[bool, str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return False, f"Failed to read file: {e}"

        if not isinstance(data, dict):
            return False, "Invalid file format (root must be object)."

        cpus = data.get("cpus")
        boards = data.get("boards")
        gpus = data.get("gpus", {})
        mems = data.get("memory", {})

        if not isinstance(cpus, dict) or not isinstance(boards, dict):
            return False, "Invalid schema: 'cpus' and 'boards' must be objects."

        for name, cpu in cpus.items():
            if "socket" not in cpu or "tdp" not in cpu:
                return False, f"CPU '{name}' missing required fields."

        for name, board in boards.items():
            if "socket" not in board or "chipset" not in board:
                return False, f"Board '{name}' missing required fields."

        self.cpu_db = cpus
        self.board_db = boards
        self.gpu_db = gpus
        self.mem_db = mems
        return True, "Import successful."

    @staticmethod
    def diff_profiles(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        diff = {}
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            va = a.get(k, "<missing>")
            vb = b.get(k, "<missing>")
            if va != vb:
                diff[k] = (va, vb)
        return diff

# -----------------------------
# RULE ENGINE (CPU/BOARD/GPU/MEM)
# -----------------------------

class RuleEngine:
    def evaluate(
        self,
        cpu: Dict[str, Any],
        board: Dict[str, Any],
        gpu: Dict[str, Any] = None,
        mem: Dict[str, Any] = None,
        workload: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = {
            "score": 100,
            "compatible": True,
            "warnings": [],
            "errors": [],
            "vrm_headroom": None,
        }

        # Socket
        if cpu.get("socket") != board.get("socket"):
            result["compatible"] = False
            result["score"] -= 70
            result["errors"].append("Socket mismatch")

        # TDP vs board max
        if cpu.get("tdp", 0) > board.get("max_tdp", 0):
            result["score"] -= 40
            result["warnings"].append("CPU TDP exceeds board max TDP (power delivery risk)")

        # BIOS version
        if cpu.get("min_bios", 0) > board.get("bios_version", 0):
            result["score"] -= 30
            result["warnings"].append("Board BIOS below CPU minimum requirement (update recommended)")

        # Chipset rules
        chipset = board.get("chipset")
        family = cpu.get("family", "")
        if chipset in CHIPSET_RULES:
            rule = CHIPSET_RULES[chipset]
            allowed_families = rule.get("families", [])
            if allowed_families and not any(f in family for f in allowed_families):
                result["score"] -= 25
                result["warnings"].append(f"CPU family '{family}' not ideal for chipset {chipset} (rule table).")
            if workload and "workloads" in rule and workload not in rule["workloads"]:
                result["score"] -= 10
                result["warnings"].append(f"Chipset {chipset} not tuned for workload '{workload}' (rule table).")
        else:
            result["score"] -= 5
            result["warnings"].append(f"Unknown chipset {chipset}; rules not defined.")

        # Microcode
        if cpu.get("min_microcode", 0) > board.get("microcode_version", 0):
            result["score"] -= 15
            result["warnings"].append("Board microcode may be below CPU minimum (firmware update recommended)")

        # VRM modeling
        vrm_limit = board.get("vrm_current_limit", 0)
        cpu_tdp = cpu.get("tdp", 0)
        if vrm_limit > 0 and cpu_tdp > 0:
            estimated_current = cpu_tdp / (1.2 * 0.9)
            headroom = vrm_limit - estimated_current
            result["vrm_headroom"] = round(headroom, 1)
            if headroom < 0:
                result["score"] -= 35
                result["warnings"].append("VRM current headroom negative (high risk under load)")
            elif headroom < 20:
                result["score"] -= 10
                result["warnings"].append("VRM headroom low; heavy workloads may stress VRM")

        # GPU compatibility
        if gpu is not None:
            slots = board.get("pcie_slots", [])
            has_slot = any(
                s["type"] == gpu.get("slot_type", "x16") and s["lanes"] >= gpu.get("min_lanes", 8)
                for s in slots
            )
            if not has_slot:
                result["score"] -= 25
                result["warnings"].append("Board may not have a suitable PCIe slot for GPU")

        # Memory compatibility
        if mem is not None:
            mem_speed = mem.get("speed", 0)
            mem_ecc = mem.get("ecc", False)
            mem_size = mem.get("size_gb", 0)

            max_cpu_speed = cpu.get("max_memory_speed", 0)
            max_board_speed = board.get("max_memory_speed", 0)
            ecc_cpu = cpu.get("ecc_support", False)
            ecc_board = board.get("ecc_support", False)
            slots = board.get("memory_slots", 0)
            max_mem = board.get("max_memory_gb", 0)

            if mem_speed > max_cpu_speed or mem_speed > max_board_speed:
                result["score"] -= 10
                result["warnings"].append("Memory speed exceeds CPU/board rated maximum (will downclock)")

            if mem_ecc and not (ecc_cpu and ecc_board):
                result["score"] -= 10
                result["warnings"].append("ECC memory used but CPU/board may not fully support ECC")

            if slots > 0 and max_mem > 0:
                total_mem = mem_size * slots
                if total_mem > max_mem:
                    result["score"] -= 10
                    result["warnings"].append("Configured memory may exceed board maximum capacity")

        if result["score"] < 0:
            result["score"] = 0

        return result

# -----------------------------
# SIMULATOR
# -----------------------------

class Simulator:
    def __init__(self, db: ProfileDB):
        self.db = db
        self.engine = RuleEngine()

    def run(
        self,
        cpu_name: str,
        board_name: str,
        gpu_name: str = None,
        mem_name: str = None,
        workload: Optional[str] = None,
    ) -> Dict[str, Any]:
        cpu = self.db.cpu_db[cpu_name]
        board = self.db.board_db[board_name]
        gpu = self.db.gpu_db[gpu_name] if gpu_name else None
        mem = self.db.mem_db[mem_name] if mem_name else None

        result = self.engine.evaluate(cpu, board, gpu, mem, workload)
        report = self.build_report(cpu_name, board_name, cpu, board, result, gpu_name, mem_name)
        return report

    @staticmethod
    def build_report(
        cpu_name: str,
        board_name: str,
        cpu: Dict[str, Any],
        board: Dict[str, Any],
        result: Dict[str, Any],
        gpu_name: str,
        mem_name: str,
    ) -> Dict[str, Any]:
        return {
            "cpu": cpu_name,
            "board": board_name,
            "gpu": gpu_name,
            "memory": mem_name,
            "score": result["score"],
            "status": "COMPATIBLE" if result["compatible"] else "INCOMPATIBLE",
            "warnings": result["warnings"],
            "errors": result["errors"],
            "vrm_headroom_amps": result["vrm_headroom"],
            "cpu_profile": cpu,
            "board_profile": board,
        }

# -----------------------------
# SUGGESTED BUILD ENGINE
# -----------------------------

class SuggestedBuildEngine:
    def __init__(self, db: ProfileDB):
        self.db = db
        self.engine = RuleEngine()

    def suggest(self, workload: str) -> Optional[Dict[str, Any]]:
        best = None
        best_score = -1

        for cpu_name, cpu in self.db.cpu_db.items():
            if not self._cpu_matches_workload(cpu, workload):
                continue

            for board_name, board in self.db.board_db.items():
                if cpu.get("socket") != board.get("socket"):
                    continue

                # Choose GPU candidate
                gpu_name = self._pick_gpu(workload)
                gpu = self.db.gpu_db.get(gpu_name) if gpu_name else None

                # Choose memory candidate
                mem_name = self._pick_memory(cpu, board, workload)
                mem = self.db.mem_db.get(mem_name) if mem_name else None

                result = self.engine.evaluate(cpu, board, gpu, mem, workload)
                score = result["score"]

                if score > best_score:
                    best_score = score
                    best = {
                        "cpu": cpu_name,
                        "board": board_name,
                        "gpu": gpu_name,
                        "memory": mem_name,
                        "score": score,
                        "result": result,
                    }

        return best

    def _cpu_matches_workload(self, cpu: Dict[str, Any], workload: str) -> bool:
        cores = cpu.get("cores", 0)
        family = cpu.get("family", "")

        if workload == "rendering":
            return cores >= 8 or "Xeon W" in family
        if workload == "gaming":
            return "Xeon" not in family
        if workload == "virtualization":
            return cores >= 8 or cpu.get("ecc_support", False)
        return True

    def _pick_gpu(self, workload: str) -> Optional[str]:
        if workload == "gaming":
            return "RTX 3080"
        if workload == "rendering":
            return "RTX 4090"
        if workload == "virtualization":
            return "Quadro RTX 4000"
        return "RTX 3080"

    def _pick_memory(self, cpu: Dict[str, Any], board: Dict[str, Any], workload: str) -> Optional[str]:
        candidates = list(self.db.mem_db.items())
        best_name = None
        best_score = -1

        for name, mem in candidates:
            score = 0
            if mem.get("ecc"):
                score += 10 if (cpu.get("ecc_support") and board.get("ecc_support")) else -5
            if workload in ("rendering", "virtualization") and mem.get("size_gb", 0) >= 32:
                score += 10
            if workload == "gaming" and mem.get("speed", 0) >= 3000 and not mem.get("ecc"):
                score += 10
            if score > best_score:
                best_score = score
                best_name = name

        return best_name

# -----------------------------
# UPGRADE ADVISOR
# -----------------------------

def recommend(cpu_profile: Dict[str, Any], workload: str) -> str:
    cores = cpu_profile.get("cores", 0)
    family = cpu_profile.get("family", "")

    if workload == "rendering":
        if cores < 8:
            return "Rendering: Consider higher core-count CPU or Xeon W-class workstation."
        return "Rendering: CPU is reasonable; ensure cooling, RAM capacity, and fast storage."

    if workload == "gaming":
        if cores >= 8 and "Xeon" in family:
            return "Gaming: Xeon works, but a high-clock Core i7/i9 may offer better FPS."
        return "Gaming: Prioritize single-core boost and strong GPU pairing."

    if workload == "virtualization":
        if cores < 8:
            return "Virtualization: More cores and RAM recommended for multiple VMs."
        return "Virtualization: CPU is suitable; ensure ECC RAM and storage IOPS."

    return "Balanced: Configuration appears reasonable; tune based on specific workloads."

# -----------------------------
# XEON MATRIX
# -----------------------------

def build_xeon_matrix(db: ProfileDB) -> List[Dict[str, Any]]:
    rows = []
    engine = RuleEngine()
    for cpu_name, cpu in db.cpu_db.items():
        if "Xeon" not in cpu_name and "Xeon" not in cpu.get("family", ""):
            continue
        for board_name, board in db.board_db.items():
            result = engine.evaluate(cpu, board)
            rows.append({
                "cpu": cpu_name,
                "board": board_name,
                "chipset": board.get("chipset"),
                "score": result["score"],
                "status": "OK" if result["compatible"] else "NO",
            })
    return rows

# -----------------------------
# BIOS "FETCHER" (LOCAL TABLE)
# -----------------------------

def get_latest_bios(board_name: str) -> int:
    return LATEST_BIOS.get(board_name, board_name and 1000 or 0)

# -----------------------------
# PLUGIN SYSTEM
# -----------------------------

class PluginManager:
    def __init__(self, db: ProfileDB):
        self.db = db
        self.plugins: List[Any] = []
        self.load_plugins()

    def load_plugins(self):
        self.plugins.clear()
        if not os.path.isdir("plugins"):
            return
        for f in os.listdir("plugins"):
            if not f.endswith(".py"):
                continue
            name = f[:-3]
            try:
                mod = __import__(f"plugins.{name}", fromlist=[name])
                self.plugins.append(mod)
            except Exception:
                pass

    def run_post_simulation_hooks(self, report: Dict[str, Any]) -> List[str]:
        messages = []
        for p in self.plugins:
            hook = getattr(p, "post_simulation", None)
            if callable(hook):
                try:
                    msg = hook(report)
                    if isinstance(msg, str) and msg.strip():
                        messages.append(msg.strip())
                except Exception:
                    continue
        return messages

# -----------------------------
# NODE SYNC SIMULATION
# -----------------------------

class NodeManager:
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self._seed_nodes()

    def _seed_nodes(self):
        for i in range(3):
            self.nodes.append({
                "name": f"Node-{i+1}",
                "last_sync": None,
                "profiles_version": 1,
            })

    def sync_all(self) -> List[str]:
        logs = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        max_version = max(n["profiles_version"] for n in self.nodes)
        for n in self.nodes:
            if n["profiles_version"] < max_version:
                n["profiles_version"] = max_version
            n["last_sync"] = now
            logs.append(f"[{now}] Synced {n['name']} -> profiles v{n['profiles_version']}")
        return logs

# -----------------------------
# GUI: MAIN WINDOW
# -----------------------------

class HardwareLab(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Workstation / Xeon Hardware Lab v3.1")
        self.resize(1400, 850)

        self.db = ProfileDB()
        self.sim = Simulator(self.db)
        self.suggest_engine = SuggestedBuildEngine(self.db)
        self.plugins = PluginManager(self.db)
        self.nodes = NodeManager()

        self.fingerprint = HardwareDetector.fingerprint()

        self._build_ui()
        self._log_event("Lab started")

    # ---------------- showEvent FIX ----------------

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.refresh_profile_view()
        self.refresh_diff_tab()
        self.refresh_matrix_tab()
        self.refresh_inventory_tab()
        self.refresh_node_view()
        self.refresh_timeline_view()
        self._refresh_fingerprint_view()
        self.refresh_topology_view()

    # ---------------- UI BUILD ----------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)

        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.addItems([
            "Dashboard",
            "OEM Simulator",
            "PCIe / VRM View",
            "Profiles",
            "Xeon Matrix",
            "Profile Diff",
            "Inventory / Scenarios",
            "Node Sync",
            "Timeline",
            "Methods & Risk",
        ])
        self.sidebar.currentRowChanged.connect(self._change_panel)
        self.sidebar.setCurrentRow(0)

        self.stack = QtWidgets.QStackedWidget()

        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(self.stack, 1)

        self._build_panel_dashboard()
        self._build_panel_oem_sim()
        self._build_panel_topology()
        self._build_panel_profiles()
        self._build_panel_matrix()
        self._build_panel_diff()
        self._build_panel_inventory()
        self._build_panel_nodes()
        self._build_panel_timeline()
        self._build_panel_methods()

        self._setup_animation()

    def _setup_animation(self):
        self.effect = QtWidgets.QGraphicsOpacityEffect(self.stack)
        self.stack.setGraphicsEffect(self.effect)
        self.anim = QtCore.QPropertyAnimation(self.effect, b"opacity")
        self.anim.setDuration(220)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)

    def _change_panel(self, index: int):
        self.stack.setCurrentIndex(index)
        self.anim.stop()
        self.effect.setOpacity(0.0)
        self.anim.start()

    # ---------------- PANEL: DASHBOARD ----------------

    def _build_panel_dashboard(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        title = QtWidgets.QLabel("Live System Fingerprint")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        self.fp_view = QtWidgets.QTextEdit()
        self.fp_view.setReadOnly(True)
        layout.addWidget(self.fp_view)

        self.stack.addWidget(w)

    def _refresh_fingerprint_view(self):
        self.fp_view.setText(json.dumps(self.fingerprint, indent=4))

    # ---------------- PANEL: OEM SIMULATOR ----------------

    def _build_panel_oem_sim(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)

        left = QtWidgets.QVBoxLayout()

        self.cpu_select = QtWidgets.QComboBox()
        self.cpu_select.addItems(self.db.get_cpu_names())

        self.board_select = QtWidgets.QComboBox()
        self.board_select.addItems(self.db.get_board_names())

        self.gpu_select = QtWidgets.QComboBox()
        self.gpu_select.addItem("<none>")
        self.gpu_select.addItems(self.db.get_gpu_names())

        self.mem_select = QtWidgets.QComboBox()
        self.mem_select.addItem("<none>")
        self.mem_select.addItems(self.db.get_mem_names())

        self.workload_select = QtWidgets.QComboBox()
        self.workload_select.addItems(["gaming", "rendering", "virtualization", "balanced"])

        self.run_btn = QtWidgets.QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)

        self.suggest_btn = QtWidgets.QPushButton("Suggest Build for Workload")
        self.suggest_btn.clicked.connect(self.suggest_build)

        self.advice_box = QtWidgets.QTextEdit()
        self.advice_box.setReadOnly(True)

        left.addWidget(QtWidgets.QLabel("CPU Model"))
        left.addWidget(self.cpu_select)
        left.addWidget(QtWidgets.QLabel("Motherboard"))
        left.addWidget(self.board_select)
        left.addWidget(QtWidgets.QLabel("GPU"))
        left.addWidget(self.gpu_select)
        left.addWidget(QtWidgets.QLabel("Memory Profile"))
        left.addWidget(self.mem_select)
        left.addWidget(QtWidgets.QLabel("Workload"))
        left.addWidget(self.workload_select)
        left.addWidget(self.run_btn)
        left.addWidget(self.suggest_btn)
        left.addWidget(QtWidgets.QLabel("Upgrade Advisor"))
        left.addWidget(self.advice_box)

        right = QtWidgets.QVBoxLayout()
        self.report_box = QtWidgets.QTextEdit()
        self.report_box.setReadOnly(True)

        self.plugin_output = QtWidgets.QTextEdit()
        self.plugin_output.setReadOnly(True)
        self.plugin_output.setPlaceholderText("Plugin messages (if any)")

        right.addWidget(QtWidgets.QLabel("OEM Compatibility Report"))
        right.addWidget(self.report_box)
        right.addWidget(QtWidgets.QLabel("Plugin Hooks"))
        right.addWidget(self.plugin_output)

        layout.addLayout(left, 1)
        layout.addLayout(right, 2)

        self.stack.addWidget(w)

    def run_simulation(self):
        cpu_name = self.cpu_select.currentText()
        board_name = self.board_select.currentText()
        gpu_name = self.gpu_select.currentText()
        if gpu_name == "<none>":
            gpu_name = None
        mem_name = self.mem_select.currentText()
        if mem_name == "<none>":
            mem_name = None

        workload = self.workload_select.currentText()
        report = self.sim.run(cpu_name, board_name, gpu_name, mem_name, workload)
        self.report_box.setText(json.dumps(report, indent=4))

        advice = recommend(self.db.cpu_db[cpu_name], workload)
        self.advice_box.setText(advice)

        INVENTORY.append({
            "time": str(datetime.now()),
            "cpu": cpu_name,
            "board": board_name,
            "gpu": gpu_name,
            "memory": mem_name,
            "workload": workload,
            "score": report["score"],
            "status": report["status"],
        })

        plugin_msgs = self.plugins.run_post_simulation_hooks(report)
        self.plugin_output.setText("\n".join(plugin_msgs) if plugin_msgs else "No plugin messages.")

        self._log_event(
            f"Simulation: CPU={cpu_name}, Board={board_name}, GPU={gpu_name}, "
            f"MEM={mem_name}, Workload={workload}, Score={report['score']}"
        )
        self.refresh_inventory_tab()

    def suggest_build(self):
        workload = self.workload_select.currentText()
        suggestion = self.suggest_engine.suggest(workload)
        if not suggestion:
            QtWidgets.QMessageBox.warning(self, "Suggestion", "No suitable build found for this workload.")
            return

        cpu_name = suggestion["cpu"]
        board_name = suggestion["board"]
        gpu_name = suggestion["gpu"]
        mem_name = suggestion["memory"]

        self.cpu_select.setCurrentText(cpu_name)
        self.board_select.setCurrentText(board_name)
        if gpu_name:
            self.gpu_select.setCurrentText(gpu_name)
        else:
            self.gpu_select.setCurrentIndex(0)
        if mem_name:
            self.mem_select.setCurrentText(mem_name)
        else:
            self.mem_select.setCurrentIndex(0)

        report = self.sim.run(cpu_name, board_name, gpu_name, mem_name, workload)
        self.report_box.setText(json.dumps(report, indent=4))

        advice = recommend(self.db.cpu_db[cpu_name], workload)
        self.advice_box.setText("Suggested build:\n" + advice)

        self._log_event(
            f"Suggested build: CPU={cpu_name}, Board={board_name}, GPU={gpu_name}, "
            f"MEM={mem_name}, Workload={workload}, Score={suggestion['score']}"
        )

    # ---------------- PANEL: PCIe / VRM VIEW ----------------

    def _build_panel_topology(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("PCIe Topology & VRM Modeling"))

        top = QtWidgets.QHBoxLayout()

        self.topology_board_select = QtWidgets.QComboBox()
        self.topology_board_select.addItems(self.db.get_board_names())
        self.topology_board_select.currentTextChanged.connect(self.refresh_topology_view)

        top.addWidget(QtWidgets.QLabel("Board:"))
        top.addWidget(self.topology_board_select)
        top.addStretch()

        layout.addLayout(top)

        self.topology_view = QtWidgets.QTextEdit()
        self.topology_view.setReadOnly(True)
        layout.addWidget(self.topology_view)

        self.stack.addWidget(w)

    def refresh_topology_view(self):
        name = self.topology_board_select.currentText()
        if not name:
            return
        board = self.db.board_db[name]
        lines = []
        lines.append(f"Board: {name}")
        lines.append(f"Chipset: {board.get('chipset')}")
        lines.append(f"PCIe Lanes (CPU+PCH): {board.get('pci_lanes')}")
        lines.append("")
        lines.append("PCIe Slots:")
        for s in board.get("pcie_slots", []):
            lines.append(f"  - {s['name']}: {s['type']} ({s['lanes']} lanes, wired to {s['wired_to']})")
        lines.append("")
        lines.append("VRM Model:")
        lines.append(f"  VRM Current Limit: {board.get('vrm_current_limit')} A")
        lines.append(f"  Max TDP (board rating): {board.get('max_tdp')} W")
        lines.append("")
        latest = get_latest_bios(name)
        lines.append(f"BIOS Version (current): {board.get('bios_version')}")
        lines.append(f"BIOS Version (latest known): {latest}")
        if latest > board.get("bios_version", 0):
            lines.append("  -> BIOS update available (recommended for new CPUs).")
        self.topology_view.setText("\n".join(lines))

    # ---------------- PANEL: PROFILES ----------------

    def _build_panel_profiles(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export Profiles")
        self.btn_import = QtWidgets.QPushButton("Import Profiles")
        self.btn_new_cpu = QtWidgets.QPushButton("New CPU Profile")
        self.btn_new_board = QtWidgets.QPushButton("New Board Profile")

        self.btn_export.clicked.connect(self.export_profiles)
        self.btn_import.clicked.connect(self.import_profiles)
        self.btn_new_cpu.clicked.connect(self.new_cpu_profile)
        self.btn_new_board.clicked.connect(self.new_board_profile)

        btn_row.addWidget(self.btn_export)
        btn_row.addWidget(self.btn_import)
        btn_row.addWidget(self.btn_new_cpu)
        btn_row.addWidget(self.btn_new_board)
        btn_row.addStretch()

        layout.addLayout(btn_row)

        self.profile_view = QtWidgets.QTextEdit()
        self.profile_view.setReadOnly(True)
        layout.addWidget(self.profile_view)

        self.stack.addWidget(w)

    def refresh_profile_view(self):
        data = {
            "cpus": self.db.cpu_db,
            "boards": self.db.board_db,
            "gpus": self.db.gpu_db,
            "memory": self.db.mem_db,
        }
        self.profile_view.setText(json.dumps(data, indent=4))

        self.cpu_select.clear()
        self.cpu_select.addItems(self.db.get_cpu_names())
        self.board_select.clear()
        self.board_select.addItems(self.db.get_board_names())
        self.gpu_select.clear()
        self.gpu_select.addItem("<none>")
        self.gpu_select.addItems(self.db.get_gpu_names())
        self.mem_select.clear()
        self.mem_select.addItem("<none>")
        self.mem_select.addItems(self.db.get_mem_names())

        self.topology_board_select.clear()
        self.topology_board_select.addItems(self.db.get_board_names())

    def export_profiles(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Profiles", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            self.db.export_to_file(path)
            QtWidgets.QMessageBox.information(self, "Export", "Profiles exported successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Failed", str(e))

    def import_profiles(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Profiles", "", "JSON Files (*.json)")
        if not path:
            return
        ok, msg = self.db.import_from_file(path)
        if ok:
            QtWidgets.QMessageBox.information(self, "Import", msg)
            self.refresh_profile_view()
            self.refresh_diff_tab()
            self.refresh_matrix_tab()
        else:
            QtWidgets.QMessageBox.critical(self, "Import Failed", msg)

    def new_cpu_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New CPU Profile", "CPU Name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("CPU Profile Wizard")
        form = QtWidgets.QFormLayout(dlg)

        socket_edit = QtWidgets.QLineEdit("LGA1151")
        tdp_edit = QtWidgets.QLineEdit("65")
        cores_edit = QtWidgets.QLineEdit("4")
        chipsets_edit = QtWidgets.QLineEdit("Z170,Z270")
        bios_edit = QtWidgets.QLineEdit("1800")
        family_edit = QtWidgets.QLineEdit("Xeon / Core")
        microcode_edit = QtWidgets.QLineEdit("0x000000C2")
        mem_speed_edit = QtWidgets.QLineEdit("2666")
        ecc_edit = QtWidgets.QLineEdit("True/False")

        form.addRow("Socket:", socket_edit)
        form.addRow("TDP:", tdp_edit)
        form.addRow("Cores:", cores_edit)
        form.addRow("Supported Chipsets (comma):", chipsets_edit)
        form.addRow("Min BIOS:", bios_edit)
        form.addRow("Family:", family_edit)
        form.addRow("Min Microcode (hex):", microcode_edit)
        form.addRow("Max Memory Speed:", mem_speed_edit)
        form.addRow("ECC Support (True/False):", ecc_edit)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        try:
            profile = {
                "socket": socket_edit.text().strip(),
                "tdp": int(tdp_edit.text().strip()),
                "cores": int(cores_edit.text().strip()),
                "supported_chipsets": [c.strip() for c in chipsets_edit.text().split(",") if c.strip()],
                "min_bios": int(bios_edit.text().strip()),
                "family": family_edit.text().strip(),
                "min_microcode": int(microcode_edit.text().strip(), 16),
                "max_memory_speed": int(mem_speed_edit.text().strip()),
                "ecc_support": ecc_edit.text().strip().lower().startswith("t"),
            }
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", str(e))
            return

        self.db.add_cpu(name, profile)
        self.refresh_profile_view()
        self.refresh_diff_tab()
        self.refresh_matrix_tab()

    def new_board_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New Board Profile", "Board Name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Board Profile Wizard")
        form = QtWidgets.QFormLayout(dlg)

        socket_edit = QtWidgets.QLineEdit("LGA1151")
        chipset_edit = QtWidgets.QLineEdit("Z170")
        max_tdp_edit = QtWidgets.QLineEdit("95")
        bios_edit = QtWidgets.QLineEdit("3000")
        lanes_edit = QtWidgets.QLineEdit("16")
        vendor_edit = QtWidgets.QLineEdit("ASUSTeK")
        product_edit = QtWidgets.QLineEdit(name)
        vrm_edit = QtWidgets.QLineEdit("150")
        microcode_edit = QtWidgets.QLineEdit("0x000000C6")
        mem_speed_edit = QtWidgets.QLineEdit("3200")
        ecc_edit = QtWidgets.QLineEdit("True/False")
        slots_edit = QtWidgets.QLineEdit("4")
        max_mem_edit = QtWidgets.QLineEdit("64")

        form.addRow("Socket:", socket_edit)
        form.addRow("Chipset:", chipset_edit)
        form.addRow("Max TDP:", max_tdp_edit)
        form.addRow("BIOS Version:", bios_edit)
        form.addRow("PCIe Lanes:", lanes_edit)
        form.addRow("Vendor:", vendor_edit)
        form.addRow("Product:", product_edit)
        form.addRow("VRM Current Limit (A):", vrm_edit)
        form.addRow("Microcode Version (hex):", microcode_edit)
        form.addRow("Max Memory Speed:", mem_speed_edit)
        form.addRow("ECC Support (True/False):", ecc_edit)
        form.addRow("Memory Slots:", slots_edit)
        form.addRow("Max Memory (GB):", max_mem_edit)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        try:
            profile = {
                "socket": socket_edit.text().strip(),
                "chipset": chipset_edit.text().strip(),
                "max_tdp": int(max_tdp_edit.text().strip()),
                "bios_version": int(bios_edit.text().strip()),
                "pci_lanes": int(lanes_edit.text().strip()),
                "vendor": vendor_edit.text().strip(),
                "product": product_edit.text().strip(),
                "vrm_current_limit": int(vrm_edit.text().strip()),
                "microcode_version": int(microcode_edit.text().strip(), 16),
                "max_memory_speed": int(mem_speed_edit.text().strip()),
                "ecc_support": ecc_edit.text().strip().lower().startswith("t"),
                "memory_slots": int(slots_edit.text().strip()),
                "max_memory_gb": int(max_mem_edit.text().strip()),
                "pcie_slots": [],
            }
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", str(e))
            return

        self.db.add_board(name, profile)
        self.refresh_profile_view()
        self.refresh_diff_tab()
        self.refresh_matrix_tab()

    # ---------------- PANEL: XEON MATRIX ----------------

    def _build_panel_matrix(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("Xeon / Workstation Compatibility Matrix"))

        self.matrix_view = QtWidgets.QTextEdit()
        self.matrix_view.setReadOnly(True)
        layout.addWidget(self.matrix_view)

        self.stack.addWidget(w)

    def refresh_matrix_tab(self):
        rows = build_xeon_matrix(self.db)
        lines = []
        for r in rows:
            lines.append(
                f"{r['cpu']}  ->  {r['board']}  "
                f"[Chipset: {r['chipset']}]  Score: {r['score']}  Status: {r['status']}"
            )
        self.matrix_view.setText("\n".join(lines) if lines else "No Xeon entries in database.")

    # ---------------- PANEL: PROFILE DIFF ----------------

    def _build_panel_diff(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()

        cpu_box = QtWidgets.QGroupBox("CPU Diff")
        cpu_layout = QtWidgets.QVBoxLayout(cpu_box)
        self.cpu_diff_a = QtWidgets.QComboBox()
        self.cpu_diff_b = QtWidgets.QComboBox()
        self.cpu_diff_a.addItems(self.db.get_cpu_names())
        self.cpu_diff_b.addItems(self.db.get_cpu_names())
        self.btn_cpu_diff = QtWidgets.QPushButton("Diff CPUs")
        self.btn_cpu_diff.clicked.connect(self.run_cpu_diff)
        cpu_layout.addWidget(self.cpu_diff_a)
        cpu_layout.addWidget(self.cpu_diff_b)
        cpu_layout.addWidget(self.btn_cpu_diff)

        board_box = QtWidgets.QGroupBox("Board Diff")
        board_layout = QtWidgets.QVBoxLayout(board_box)
        self.board_diff_a = QtWidgets.QComboBox()
        self.board_diff_b = QtWidgets.QComboBox()
        self.board_diff_a.addItems(self.db.get_board_names())
        self.board_diff_b.addItems(self.db.get_board_names())
        self.btn_board_diff = QtWidgets.QPushButton("Diff Boards")
        self.btn_board_diff.clicked.connect(self.run_board_diff)
        board_layout.addWidget(self.board_diff_a)
        board_layout.addWidget(self.board_diff_b)
        board_layout.addWidget(self.btn_board_diff)

        top.addWidget(cpu_box)
        top.addWidget(board_box)
        top.addStretch()

        layout.addLayout(top)

        self.diff_output = QtWidgets.QTextEdit()
        self.diff_output.setReadOnly(True)
        layout.addWidget(self.diff_output)

        self.stack.addWidget(w)

    def refresh_diff_tab(self):
        if not hasattr(self, "cpu_diff_a"):
            return
        self.cpu_diff_a.clear()
        self.cpu_diff_b.clear()
        self.cpu_diff_a.addItems(self.db.get_cpu_names())
        self.cpu_diff_b.addItems(self.db.get_cpu_names())

        self.board_diff_a.clear()
        self.board_diff_b.clear()
        self.board_diff_a.addItems(self.db.get_board_names())
        self.board_diff_b.addItems(self.db.get_board_names())

    def run_cpu_diff(self):
        a = self.cpu_diff_a.currentText()
        b = self.cpu_diff_b.currentText()
        pa = self.db.cpu_db[a]
        pb = self.db.cpu_db[b]
        diff = self.db.diff_profiles(pa, pb)
        self._render_diff("CPU", a, b, diff)

    def run_board_diff(self):
        a = self.board_diff_a.currentText()
        b = self.board_diff_b.currentText()
        pa = self.db.board_db[a]
        pb = self.db.board_db[b]
        diff = self.db.diff_profiles(pa, pb)
        self._render_diff("Board", a, b, diff)

    def _render_diff(self, kind: str, a: str, b: str, diff: Dict[str, Tuple[Any, Any]]):
        lines = [f"{kind} Diff: {a}  vs  {b}", "-" * 50]
        if not diff:
            lines.append("No differences.")
        else:
            for k, (va, vb) in diff.items():
                lines.append(f"{k}: {va!r}  ->  {vb!r}")
        self.diff_output.setText("\n".join(lines))

    # ---------------- PANEL: INVENTORY / SCENARIOS ----------------

    def _build_panel_inventory(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("Simulated Config Inventory"))

        self.inventory_view = QtWidgets.QTextEdit()
        self.inventory_view.setReadOnly(True)
        layout.addWidget(self.inventory_view)

        self.stack.addWidget(w)

    def refresh_inventory_tab(self):
        lines = []
        for item in INVENTORY:
            lines.append(
                f"[{item['time']}] CPU={item['cpu']}  Board={item['board']}  "
                f"GPU={item['gpu']}  MEM={item['memory']}  "
                f"Workload={item['workload']}  Score={item['score']}  Status={item['status']}"
            )
        self.inventory_view.setText("\n".join(lines) if lines else "No simulations recorded yet.")

    # ---------------- PANEL: NODE SYNC ----------------

    def _build_panel_nodes(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("Node Sync Simulation"))

        self.node_view = QtWidgets.QTextEdit()
        self.node_view.setReadOnly(True)
        layout.addWidget(self.node_view)

        btn = QtWidgets.QPushButton("Sync All Nodes")
        btn.clicked.connect(self.sync_nodes)
        layout.addWidget(btn)

        self.stack.addWidget(w)

    def refresh_node_view(self):
        lines = []
        for n in self.nodes.nodes:
            lines.append(
                f"{n['name']}: profiles v{n['profiles_version']}  "
                f"last_sync={n['last_sync'] or 'never'}"
            )
        self.node_view.setText("\n".join(lines))

    def sync_nodes(self):
        logs = self.nodes.sync_all()
        for l in logs:
            self._log_event(l)
        self.refresh_node_view()
        QtWidgets.QMessageBox.information(self, "Node Sync", "Nodes synchronized (simulated).")

    # ---------------- PANEL: TIMELINE ----------------

    def _build_panel_timeline(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("Event Timeline"))

        self.timeline_view = QtWidgets.QTextEdit()
        self.timeline_view.setReadOnly(True)
        layout.addWidget(self.timeline_view)

        self.stack.addWidget(w)

    def _log_event(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        TIMELINE.append(f"[{ts}] {msg}")
        self.refresh_timeline_view()

    def refresh_timeline_view(self):
        self.timeline_view.setText("\n".join(TIMELINE))

    # ---------------- PANEL: METHODS & RISK ----------------

    def _build_panel_methods(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)

        content = []
        content.append("Method Overview (Informational Only)")
        content.append("-" * 60)
        content.append("1) OEM-style Compatibility Simulation (this tool)")
        content.append("   - Risk: LOW")
        content.append("   - Description: Simulation of CPU/board/GPU/memory using sockets, TDP, VRM, BIOS, chipset rules.")
        content.append("")
        content.append("2) Firmware / BIOS Modifications")
        content.append("   - Risk: EXTREME")
        content.append("   - Description: Not performed or guided by this tool. Can brick hardware, void warranties.")
        content.append("")
        content.append("3) CPUID / Microcode Manipulation")
        content.append("   - Risk: EXTREME")
        content.append("   - Description: Not performed or guided by this tool. Deep platform-level behavior.")
        content.append("")
        content.append("4) Registry / OS-Level Cosmetic Tweaks")
        content.append("   - Risk: MEDIUM")
        content.append("   - Description: Cosmetic only; does not change real hardware behavior. Not used here.")
        content.append("")
        content.append("Danger Meter (for this tool):")
        content.append("   - All operations are read-only and simulation-based.")
        content.append("   - No firmware flashing, no CPUID override, no registry writes.")
        content.append("   - Safe for workstation planning and what-if analysis.")

        text.setText("\n".join(content))
        layout.addWidget(text)

        self.stack.addWidget(w)

# -----------------------------
# MAIN
# -----------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = HardwareLab()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
