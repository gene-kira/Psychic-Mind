#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Workstation / Xeon Hardware Lab v4.0 (Unified)

Safe engineering-style workstation planning + simulation cockpit:

SMARTER PLATFORM MODELING:
- CPU / Board / GPU / Memory compatibility engine
- PCIe topology + bifurcation simulation
- NUMA topology (single vs multi-socket heuristic)
- Memory channel bandwidth estimation
- DMI / PCIe bottleneck simulation
- VRM power modeling (headroom heuristic)
- BIOS version awareness (local "latest BIOS" table)
- Microcode awareness (CPU min vs board current)
- GPU database (TDP, PCIe requirements, tier)
- Memory compatibility engine (speed, ECC, capacity)
- Storage subsystem modeling (NVMe lanes, RAID, PCIe gen)
- Multi-CPU workstation support (Xeon Scalable, EPYC-style profiles)

WORKLOAD-SPECIFIC SCORING:
- Blender (rendering)
- Unreal Engine (gaming / real-time)
- Virtualization density
- AI inference

POWER SUPPLY SIZING:
- CPU TDP
- GPU TDP
- Board overhead
- VRM efficiency (heuristic)
- PSU derating curves (headroom modeling)

OPERATOR-GRADE COCKPIT:
- Tactical layout (sidebar + stacked panels)
- Animated panel transitions
- Live system fingerprint map
- Node sync simulation
- Timeline visualization
- Profile diff, Xeon matrix, inventory

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
        "sockets_required": 1,
        "numa_nodes": 1,
        "memory_channels": 4,
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
        "sockets_required": 1,
        "numa_nodes": 1,
        "memory_channels": 2,
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
        "sockets_required": 1,
        "numa_nodes": 1,
        "memory_channels": 2,
    },
    # Multi-CPU style examples (heuristic)
    "Xeon Scalable 6248R (Dual)": {
        "socket": "LGA3647",
        "tdp": 205,
        "cores": 24,
        "supported_chipsets": ["C621"],
        "min_bios": 1000,
        "family": "Xeon Scalable",
        "min_microcode": 0x02000057,
        "max_memory_speed": 2933,
        "ecc_support": True,
        "sockets_required": 2,
        "numa_nodes": 2,
        "memory_channels": 6,
    },
    "EPYC 7542 (Single)": {
        "socket": "SP3",
        "tdp": 225,
        "cores": 32,
        "supported_chipsets": ["SP3-WS"],
        "min_bios": 1000,
        "family": "EPYC",
        "min_microcode": 0x08001137,
        "max_memory_speed": 3200,
        "ecc_support": True,
        "sockets_required": 1,
        "numa_nodes": 4,
        "memory_channels": 8,
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
            {"name": "PCIEX16_1", "type": "x16", "lanes": 16, "wired_to": "CPU", "bifurcation": "x16/x8x8"},
            {"name": "PCIEX16_2", "type": "x16", "lanes": 16, "wired_to": "CPU", "bifurcation": "x16/x8x8"},
            {"name": "PCIEX16_3", "type": "x16", "lanes": 16, "wired_to": "PCH", "bifurcation": "x16/x8x4x4"},
        ],
        "cpu_sockets": 1,
        "numa_nodes": 1,
        "dmi_bandwidth_gbps": 8.0,
        "nvme_slots": 2,
        "pcie_gen": 3,
        "raid_levels": ["0", "1", "10"],
        "base_board_power": 60,
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
            {"name": "PCIEX16_1", "type": "x16", "lanes": 16, "wired_to": "CPU", "bifurcation": "x16/x8x8"},
            {"name": "PCIEX16_2", "type": "x16", "lanes": 4, "wired_to": "PCH", "bifurcation": "x4"},
            {"name": "PCIEX1_1", "type": "x1", "lanes": 1, "wired_to": "PCH", "bifurcation": "x1"},
        ],
        "cpu_sockets": 1,
        "numa_nodes": 1,
        "dmi_bandwidth_gbps": 8.0,
        "nvme_slots": 1,
        "pcie_gen": 3,
        "raid_levels": ["0", "1", "10"],
        "base_board_power": 45,
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
            {"name": "PCIEX16_1", "type": "x16", "lanes": 16, "wired_to": "CPU", "bifurcation": "x16/x8x8"},
            {"name": "PCIEX4_1", "type": "x4", "lanes": 4, "wired_to": "PCH", "bifurcation": "x4"},
        ],
        "cpu_sockets": 1,
        "numa_nodes": 1,
        "dmi_bandwidth_gbps": 8.0,
        "nvme_slots": 2,
        "pcie_gen": 3,
        "raid_levels": ["0", "1", "5", "10"],
        "base_board_power": 50,
    },
    # Multi-CPU style board
    "Supermicro X11DPi-NT (Dual 3647)": {
        "socket": "LGA3647",
        "chipset": "C621",
        "max_tdp": 205,
        "bios_version": 1200,
        "pci_lanes": 80,
        "vendor": "Supermicro",
        "product": "X11DPi-NT",
        "vrm_current_limit": 400,
        "microcode_version": 0x02000057,
        "max_memory_speed": 2933,
        "ecc_support": True,
        "memory_slots": 16,
        "max_memory_gb": 2048,
        "pcie_slots": [
            {"name": "PCIEX16_1", "type": "x16", "lanes": 16, "wired_to": "CPU0", "bifurcation": "x16/x8x8"},
            {"name": "PCIEX16_2", "type": "x16", "lanes": 16, "wired_to": "CPU1", "bifurcation": "x16/x8x8"},
            {"name": "PCIEX16_3", "type": "x16", "lanes": 16, "wired_to": "CPU0", "bifurcation": "x16/x8x4x4"},
        ],
        "cpu_sockets": 2,
        "numa_nodes": 2,
        "dmi_bandwidth_gbps": 10.0,
        "nvme_slots": 4,
        "pcie_gen": 3,
        "raid_levels": ["0", "1", "5", "10"],
        "base_board_power": 90,
    },
    "EPYC SP3-WS Board": {
        "socket": "SP3",
        "chipset": "SP3-WS",
        "max_tdp": 280,
        "bios_version": 1000,
        "pci_lanes": 128,
        "vendor": "Generic",
        "product": "EPYC SP3-WS",
        "vrm_current_limit": 450,
        "microcode_version": 0x08001137,
        "max_memory_speed": 3200,
        "ecc_support": True,
        "memory_slots": 8,
        "max_memory_gb": 2048,
        "pcie_slots": [
            {"name": "PCIEX16_1", "type": "x16", "lanes": 16, "wired_to": "CPU", "bifurcation": "x16/x8x8"},
            {"name": "PCIEX16_2", "type": "x16", "lanes": 16, "wired_to": "CPU", "bifurcation": "x16/x8x8"},
            {"name": "PCIEX16_3", "type": "x16", "lanes": 16, "wired_to": "CPU", "bifurcation": "x16/x8x4x4"},
        ],
        "cpu_sockets": 1,
        "numa_nodes": 4,
        "dmi_bandwidth_gbps": 16.0,
        "nvme_slots": 6,
        "pcie_gen": 4,
        "raid_levels": ["0", "1", "5", "10"],
        "base_board_power": 100,
    },
}

GPU_DB: Dict[str, Dict[str, Any]] = {
    "RTX 4090": {
        "tdp": 450,
        "slot_type": "x16",
        "min_lanes": 8,
        "recommended_psu": 1000,
        "tier": "ultra",
        "pcie_gen": 4,
    },
    "RTX 3080": {
        "tdp": 320,
        "slot_type": "x16",
        "min_lanes": 8,
        "recommended_psu": 750,
        "tier": "high",
        "pcie_gen": 4,
    },
    "Quadro RTX 4000": {
        "tdp": 160,
        "slot_type": "x16",
        "min_lanes": 8,
        "recommended_psu": 550,
        "tier": "pro",
        "pcie_gen": 3,
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

STORAGE_DB: Dict[str, Dict[str, Any]] = {
    "Single NVMe": {
        "nvme_count": 1,
        "raid_level": "0",
        "pcie_gen_required": 3,
    },
    "Dual NVMe RAID1": {
        "nvme_count": 2,
        "raid_level": "1",
        "pcie_gen_required": 3,
    },
    "Quad NVMe RAID10": {
        "nvme_count": 4,
        "raid_level": "10",
        "pcie_gen_required": 3,
    },
    "Dual NVMe Gen4": {
        "nvme_count": 2,
        "raid_level": "0",
        "pcie_gen_required": 4,
    },
}

LATEST_BIOS: Dict[str, int] = {
    "ASUS WS C422 PRO": 1301,
    "ASUS Z170-A": 3805,
    "ASUS W480": 1202,
    "Supermicro X11DPi-NT (Dual 3647)": 1300,
    "EPYC SP3-WS Board": 1100,
}

CHIPSET_RULES: Dict[str, Dict[str, Any]] = {
    "C422": {
        "families": ["Xeon W"],
        "ecc_required": True,
        "workloads": ["rendering", "virtualization", "balanced", "blender", "ai_inference"],
    },
    "C236": {
        "families": ["Xeon E3", "Core i7"],
        "ecc_optional": True,
        "workloads": ["balanced", "virtualization"],
    },
    "Z170": {
        "families": ["Core i7"],
        "gaming_preferred": True,
        "workloads": ["gaming", "balanced", "unreal"],
    },
    "Z270": {
        "families": ["Core i7", "Xeon E3"],
        "workloads": ["gaming", "balanced", "unreal"],
    },
    "W480": {
        "families": ["Xeon W", "Core i7"],
        "ecc_optional": True,
        "workloads": ["balanced", "virtualization", "rendering", "blender"],
    },
    "X299": {
        "families": ["Xeon W", "Core i9"],
        "workloads": ["rendering", "balanced", "blender"],
    },
    "C621": {
        "families": ["Xeon Scalable"],
        "ecc_required": True,
        "workloads": ["virtualization", "rendering", "ai_inference"],
    },
    "SP3-WS": {
        "families": ["EPYC"],
        "ecc_required": True,
        "workloads": ["virtualization", "rendering", "ai_inference"],
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
        self.storage_db = deepcopy(STORAGE_DB)

    def get_cpu_names(self) -> List[str]:
        return sorted(self.cpu_db.keys())

    def get_board_names(self) -> List[str]:
        return sorted(self.board_db.keys())

    def get_gpu_names(self) -> List[str]:
        return sorted(self.gpu_db.keys())

    def get_mem_names(self) -> List[str]:
        return sorted(self.mem_db.keys())

    def get_storage_names(self) -> List[str]:
        return sorted(self.storage_db.keys())

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
            "storage": self.storage_db,
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
        storage = data.get("storage", {})

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
        self.storage_db = storage
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
# PLATFORM MODELING
# -----------------------------

class PlatformModel:
    @staticmethod
    def pcie_bifurcation(board: Dict[str, Any]) -> List[str]:
        lines = []
        for s in board.get("pcie_slots", []):
            bif = s.get("bifurcation", "n/a")
            lines.append(
                f"{s['name']}: {s['type']} ({s['lanes']} lanes, {s['wired_to']}, bifurcation={bif})"
            )
        return lines

    @staticmethod
    def numa_topology(cpu: Dict[str, Any], board: Dict[str, Any]) -> Dict[str, Any]:
        cpu_nodes = cpu.get("numa_nodes", 1)
        board_nodes = board.get("numa_nodes", 1)
        sockets_req = cpu.get("sockets_required", 1)
        sockets_board = board.get("cpu_sockets", 1)
        return {
            "cpu_numa_nodes": cpu_nodes,
            "board_numa_nodes": board_nodes,
            "cpu_sockets_required": sockets_req,
            "board_cpu_sockets": sockets_board,
            "numa_mismatch": cpu_nodes != board_nodes or sockets_req > sockets_board,
        }

    @staticmethod
    def memory_bandwidth(cpu: Dict[str, Any], board: Dict[str, Any]) -> float:
        channels = cpu.get("memory_channels", 2)
        speed = min(cpu.get("max_memory_speed", 2133), board.get("max_memory_speed", 2133))
        # Very rough: GB/s ≈ channels * speed * 8 bytes / 1e3
        return round(channels * speed * 8 / 1000.0, 1)

    @staticmethod
    def dmi_bottleneck(board: Dict[str, Any], gpu: Dict[str, Any], storage: Dict[str, Any]) -> Dict[str, Any]:
        dmi_bw = board.get("dmi_bandwidth_gbps", 8.0)
        nvme_count = storage.get("nvme_count", 0) if storage else 0
        pcie_gen = board.get("pcie_gen", 3)
        # Rough NVMe bandwidth per drive
        nvme_bw = nvme_count * (3.5 if pcie_gen == 3 else 7.0)
        # GPU traffic (heuristic)
        gpu_bw = 0.0
        if gpu:
            gpu_bw = 8.0 if gpu.get("pcie_gen", 4) >= 4 else 4.0
        total = nvme_bw + gpu_bw
        saturated = total > dmi_bw
        return {
            "dmi_bandwidth_gbps": dmi_bw,
            "estimated_nvme_traffic_gbps": nvme_bw,
            "estimated_gpu_traffic_gbps": gpu_bw,
            "estimated_total_traffic_gbps": total,
            "dmi_saturated": saturated,
        }

# -----------------------------
# POWER SUPPLY MODELING
# -----------------------------

class PowerEngine:
    @staticmethod
    def estimate_system_power(cpu: Dict[str, Any], board: Dict[str, Any], gpu: Dict[str, Any], storage: Dict[str, Any]) -> Dict[str, Any]:
        cpu_tdp = cpu.get("tdp", 0)
        gpu_tdp = gpu.get("tdp", 0) if gpu else 0
        board_power = board.get("base_board_power", 50)
        nvme_count = storage.get("nvme_count", 0) if storage else 0
        nvme_power = nvme_count * 7  # ~7W per NVMe
        misc = 40  # fans, RAM, etc.

        raw_total = cpu_tdp + gpu_tdp + board_power + nvme_power + misc
        # VRM efficiency ~90%, PSU derating target 70% load
        vrm_loss = raw_total * (1 / 0.9 - 1)
        recommended_psu = int((raw_total + vrm_loss) / 0.7 + 50)

        return {
            "cpu_tdp": cpu_tdp,
            "gpu_tdp": gpu_tdp,
            "board_power": board_power,
            "nvme_power": nvme_power,
            "misc_power": misc,
            "raw_total_power": int(raw_total),
            "vrm_loss_power": int(vrm_loss),
            "recommended_psu_watts": recommended_psu,
        }

    @staticmethod
    def evaluate_psu_margin(power_info: Dict[str, Any], gpu: Dict[str, Any]) -> Tuple[int, List[str]]:
        rec = power_info["recommended_psu_watts"]
        warnings = []
        if gpu:
            gpu_rec = gpu.get("recommended_psu", 0)
            if rec < gpu_rec:
                warnings.append(f"Estimated PSU ({rec}W) is below GPU vendor recommendation ({gpu_rec}W).")
        return rec, warnings

# -----------------------------
# WORKLOAD-SPECIFIC SCORING
# -----------------------------

class WorkloadScorer:
    @staticmethod
    def score(workload: str, cpu: Dict[str, Any], board: Dict[str, Any], gpu: Dict[str, Any],
              mem_bw: float, dmi_info: Dict[str, Any]) -> Tuple[int, List[str]]:
        score_delta = 0
        notes: List[str] = []
        cores = cpu.get("cores", 0)
        family = cpu.get("family", "")
        gpu_tier = gpu.get("tier", "") if gpu else ""
        numa_nodes = cpu.get("numa_nodes", 1)

        if workload in ("rendering", "blender"):
            if cores >= 16:
                score_delta += 15
                notes.append("Rendering: High core count is ideal.")
            if mem_bw < 50:
                score_delta -= 10
                notes.append("Rendering: Memory bandwidth may be limiting.")
        elif workload in ("gaming", "unreal"):
            if "Xeon" in family and cores > 8:
                score_delta -= 5
                notes.append("Gaming: Xeon is functional but not optimal for high FPS.")
            if gpu_tier in ("ultra", "high"):
                score_delta += 10
                notes.append("Gaming: Strong GPU tier for Unreal/real-time workloads.")
        elif workload in ("virtualization", "virtualization_density"):
            if cores < 16:
                score_delta -= 10
                notes.append("Virtualization: More cores recommended for high VM density.")
            if numa_nodes > 1:
                score_delta += 5
                notes.append("Virtualization: NUMA-aware multi-socket platform can help scaling.")
        elif workload in ("ai_inference", "ai"):
            if gpu_tier in ("ultra", "high"):
                score_delta += 15
                notes.append("AI: Strong GPU tier for inference workloads.")
            if mem_bw < 60:
                score_delta -= 5
                notes.append("AI: Memory bandwidth may limit large models.")

        if dmi_info.get("dmi_saturated"):
            score_delta -= 10
            notes.append("Platform: DMI/PCIe link may be saturated under heavy IO.")

        return score_delta, notes

# -----------------------------
# RULE ENGINE (CPU/BOARD/GPU/MEM/STORAGE)
# -----------------------------

class RuleEngine:
    def evaluate(
        self,
        cpu: Dict[str, Any],
        board: Dict[str, Any],
        gpu: Dict[str, Any] = None,
        mem: Dict[str, Any] = None,
        storage: Dict[str, Any] = None,
        workload: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = {
            "score": 100,
            "compatible": True,
            "warnings": [],
            "errors": [],
            "vrm_headroom": None,
            "platform": {},
            "power": {},
            "workload_notes": [],
        }

        # Socket + multi-CPU
        if cpu.get("socket") != board.get("socket"):
            result["compatible"] = False
            result["score"] -= 70
            result["errors"].append("Socket mismatch")

        sockets_req = cpu.get("sockets_required", 1)
        sockets_board = board.get("cpu_sockets", 1)
        if sockets_req > sockets_board:
            result["compatible"] = False
            result["score"] -= 50
            result["errors"].append("Board does not support required CPU socket count")

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

            if board.get("pcie_gen", 3) < gpu.get("pcie_gen", 3):
                result["score"] -= 5
                result["warnings"].append("Board PCIe generation below GPU capability (bandwidth limited).")

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

        # Storage modeling
        if storage is not None:
            nvme_count = storage.get("nvme_count", 0)
            if nvme_count > board.get("nvme_slots", 0):
                result["score"] -= 15
                result["warnings"].append("Requested NVMe count exceeds board NVMe slots.")
            raid_level = storage.get("raid_level", "0")
            if raid_level not in board.get("raid_levels", []):
                result["score"] -= 10
                result["warnings"].append(f"Board does not advertise RAID level {raid_level} support.")
            if board.get("pcie_gen", 3) < storage.get("pcie_gen_required", 3):
                result["score"] -= 5
                result["warnings"].append("Board PCIe generation may limit NVMe performance.")

        # Platform modeling
        mem_bw = PlatformModel.memory_bandwidth(cpu, board)
        numa_info = PlatformModel.numa_topology(cpu, board)
        dmi_info = PlatformModel.dmi_bottleneck(board, gpu, storage)
        pcie_bifurcation = PlatformModel.pcie_bifurcation(board)

        result["platform"] = {
            "memory_bandwidth_gbps": mem_bw,
            "numa": numa_info,
            "dmi": dmi_info,
            "pcie_bifurcation": pcie_bifurcation,
        }

        # Power modeling
        power_info = PowerEngine.estimate_system_power(cpu, board, gpu, storage)
        psu_rec, psu_warnings = PowerEngine.evaluate_psu_margin(power_info, gpu)
        result["power"] = {
            "power_model": power_info,
            "recommended_psu_watts": psu_rec,
        }
        result["warnings"].extend(psu_warnings)

        # Workload-specific scoring
        if workload:
            wl_score, wl_notes = WorkloadScorer.score(workload, cpu, board, gpu, mem_bw, dmi_info)
            result["score"] += wl_score
            result["workload_notes"] = wl_notes

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
        storage_name: str = None,
        workload: Optional[str] = None,
    ) -> Dict[str, Any]:
        cpu = self.db.cpu_db[cpu_name]
        board = self.db.board_db[board_name]
        gpu = self.db.gpu_db[gpu_name] if gpu_name else None
        mem = self.db.mem_db[mem_name] if mem_name else None
        storage = self.db.storage_db[storage_name] if storage_name else None

        result = self.engine.evaluate(cpu, board, gpu, mem, storage, workload)
        report = self.build_report(cpu_name, board_name, cpu, board, result, gpu_name, mem_name, storage_name)
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
        storage_name: str,
    ) -> Dict[str, Any]:
        return {
            "cpu": cpu_name,
            "board": board_name,
            "gpu": gpu_name,
            "memory": mem_name,
            "storage": storage_name,
            "score": result["score"],
            "status": "COMPATIBLE" if result["compatible"] else "INCOMPATIBLE",
            "warnings": result["warnings"],
            "errors": result["errors"],
            "vrm_headroom_amps": result["vrm_headroom"],
            "cpu_profile": cpu,
            "board_profile": board,
            "platform": result["platform"],
            "power": result["power"],
            "workload_notes": result["workload_notes"],
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
                if cpu.get("sockets_required", 1) > board.get("cpu_sockets", 1):
                    continue

                gpu_name = self._pick_gpu(workload)
                gpu = self.db.gpu_db.get(gpu_name) if gpu_name else None

                mem_name = self._pick_memory(cpu, board, workload)
                mem = self.db.mem_db.get(mem_name) if mem_name else None

                storage_name = self._pick_storage(board, workload)
                storage = self.db.storage_db.get(storage_name) if storage_name else None

                result = self.engine.evaluate(cpu, board, gpu, mem, storage, workload)
                score = result["score"]

                if score > best_score:
                    best_score = score
                    best = {
                        "cpu": cpu_name,
                        "board": board_name,
                        "gpu": gpu_name,
                        "memory": mem_name,
                        "storage": storage_name,
                        "score": score,
                        "result": result,
                    }

        return best

    def _cpu_matches_workload(self, cpu: Dict[str, Any], workload: str) -> bool:
        cores = cpu.get("cores", 0)
        family = cpu.get("family", "")

        if workload in ("rendering", "blender"):
            return cores >= 8 or "Xeon" in family or "EPYC" in family
        if workload in ("gaming", "unreal"):
            return "Xeon Scalable" not in family and "EPYC" not in family
        if workload in ("virtualization", "virtualization_density"):
            return cores >= 12 or cpu.get("ecc_support", False)
        if workload in ("ai_inference", "ai"):
            return cores >= 8
        return True

    def _pick_gpu(self, workload: str) -> Optional[str]:
        if workload in ("gaming", "unreal"):
            return "RTX 3080"
        if workload in ("rendering", "blender", "ai_inference", "ai"):
            return "RTX 4090"
        if workload in ("virtualization", "virtualization_density"):
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
            if workload in ("rendering", "blender", "virtualization", "virtualization_density", "ai_inference"):
                if mem.get("size_gb", 0) >= 32:
                    score += 10
            if workload in ("gaming", "unreal") and mem.get("speed", 0) >= 3000 and not mem.get("ecc"):
                score += 10
            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    def _pick_storage(self, board: Dict[str, Any], workload: str) -> Optional[str]:
        if workload in ("gaming", "unreal"):
            return "Single NVMe"
        if workload in ("rendering", "blender"):
            return "Dual NVMe RAID1"
        if workload in ("virtualization", "virtualization_density"):
            return "Quad NVMe RAID10"
        if workload in ("ai_inference", "ai"):
            if board.get("pcie_gen", 3) >= 4:
                return "Dual NVMe Gen4"
            return "Dual NVMe RAID1"
        return "Single NVMe"

# -----------------------------
# UPGRADE ADVISOR
# -----------------------------

def recommend(cpu_profile: Dict[str, Any], workload: str) -> str:
    cores = cpu_profile.get("cores", 0)
    family = cpu_profile.get("family", "")

    if workload in ("rendering", "blender"):
        if cores < 8:
            return "Rendering: Consider higher core-count CPU or Xeon/EPYC-class workstation."
        return "Rendering: CPU is reasonable; ensure cooling, RAM capacity, and fast NVMe storage."

    if workload in ("gaming", "unreal"):
        if cores >= 8 and "Xeon" in family:
            return "Gaming: Xeon works, but a high-clock Core i7/i9 may offer better FPS."
        return "Gaming: Prioritize single-core boost and strong GPU pairing."

    if workload in ("virtualization", "virtualization_density"):
        if cores < 12:
            return "Virtualization: More cores and RAM recommended for high VM density."
        return "Virtualization: CPU is suitable; ensure ECC RAM and storage IOPS."

    if workload in ("ai_inference", "ai"):
        return "AI: Strong GPU and high memory bandwidth recommended; ensure PSU headroom."

    return "Balanced: Configuration appears reasonable; tune based on specific workloads."

# -----------------------------
# XEON MATRIX
# -----------------------------

def build_xeon_matrix(db: ProfileDB) -> List[Dict[str, Any]]:
    rows = []
    engine = RuleEngine()
    for cpu_name, cpu in db.cpu_db.items():
        if "Xeon" not in cpu_name and "Xeon" not in cpu.get("family", "") and "EPYC" not in cpu.get("family", ""):
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
        self.setWindowTitle("Workstation / Xeon Hardware Lab v4.0")
        self.resize(1450, 900)

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
        self.sidebar.setFixedWidth(230)
        self.sidebar.addItems([
            "Dashboard",
            "OEM Simulator",
            "PCIe / VRM / Storage View",
            "Profiles",
            "Xeon / EPYC Matrix",
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

        self.storage_select = QtWidgets.QComboBox()
        self.storage_select.addItem("<none>")
        self.storage_select.addItems(self.db.get_storage_names())

        self.workload_select = QtWidgets.QComboBox()
        self.workload_select.addItems([
            "gaming",
            "unreal",
            "rendering",
            "blender",
            "virtualization",
            "virtualization_density",
            "ai_inference",
            "balanced",
        ])

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
        left.addWidget(QtWidgets.QLabel("Storage Profile"))
        left.addWidget(self.storage_select)
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
        storage_name = self.storage_select.currentText()
        if storage_name == "<none>":
            storage_name = None

        workload = self.workload_select.currentText()
        report = self.sim.run(cpu_name, board_name, gpu_name, mem_name, storage_name, workload)
        self.report_box.setText(json.dumps(report, indent=4))

        advice = recommend(self.db.cpu_db[cpu_name], workload)
        self.advice_box.setText(advice)

        INVENTORY.append({
            "time": str(datetime.now()),
            "cpu": cpu_name,
            "board": board_name,
            "gpu": gpu_name,
            "memory": mem_name,
            "storage": storage_name,
            "workload": workload,
            "score": report["score"],
            "status": report["status"],
        })

        plugin_msgs = self.plugins.run_post_simulation_hooks(report)
        self.plugin_output.setText("\n".join(plugin_msgs) if plugin_msgs else "No plugin messages.")

        self._log_event(
            f"Simulation: CPU={cpu_name}, Board={board_name}, GPU={gpu_name}, "
            f"MEM={mem_name}, Storage={storage_name}, Workload={workload}, Score={report['score']}"
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
        storage_name = suggestion["storage"]

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
        if storage_name:
            self.storage_select.setCurrentText(storage_name)
        else:
            self.storage_select.setCurrentIndex(0)

        report = self.sim.run(cpu_name, board_name, gpu_name, mem_name, storage_name, workload)
        self.report_box.setText(json.dumps(report, indent=4))

        advice = recommend(self.db.cpu_db[cpu_name], workload)
        self.advice_box.setText("Suggested build:\n" + advice)

        self._log_event(
            f"Suggested build: CPU={cpu_name}, Board={board_name}, GPU={gpu_name}, "
            f"MEM={mem_name}, Storage={storage_name}, Workload={workload}, Score={suggestion['score']}"
        )

    # ---------------- PANEL: PCIe / VRM / STORAGE VIEW ----------------

    def _build_panel_topology(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("PCIe Topology, VRM, Storage & BIOS Modeling"))

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
        lines.append(f"Vendor: {board.get('vendor')}  Product: {board.get('product')}")
        lines.append(f"Chipset: {board.get('chipset')}")
        lines.append(f"CPU Sockets: {board.get('cpu_sockets')}  NUMA Nodes: {board.get('numa_nodes')}")
        lines.append(f"PCIe Lanes (CPU+PCH): {board.get('pci_lanes')}")
        lines.append(f"PCIe Generation: Gen{board.get('pcie_gen')}")
        lines.append("")
        lines.append("PCIe Slots + Bifurcation:")
        for s in board.get("pcie_slots", []):
            lines.append(
                f"  - {s['name']}: {s['type']} ({s['lanes']} lanes, wired to {s['wired_to']}, bifurcation={s.get('bifurcation','n/a')})"
            )
        lines.append("")
        lines.append("VRM Model:")
        lines.append(f"  VRM Current Limit: {board.get('vrm_current_limit')} A")
        lines.append(f"  Max TDP (board rating): {board.get('max_tdp')} W")
        lines.append("")
        lines.append("Storage / NVMe:")
        lines.append(f"  NVMe Slots: {board.get('nvme_slots')}")
        lines.append(f"  RAID Levels: {', '.join(board.get('raid_levels', []))}")
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
            "storage": self.db.storage_db,
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
        self.storage_select.clear()
        self.storage_select.addItem("<none>")
        self.storage_select.addItems(self.db.get_storage_names())

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
        family_edit = QtWidgets.QLineEdit("Xeon / Core / EPYC")
        microcode_edit = QtWidgets.QLineEdit("0x000000C2")
        mem_speed_edit = QtWidgets.QLineEdit("2666")
        ecc_edit = QtWidgets.QLineEdit("True/False")
        sockets_req_edit = QtWidgets.QLineEdit("1")
        numa_nodes_edit = QtWidgets.QLineEdit("1")
        mem_channels_edit = QtWidgets.QLineEdit("2")

        form.addRow("Socket:", socket_edit)
        form.addRow("TDP:", tdp_edit)
        form.addRow("Cores:", cores_edit)
        form.addRow("Supported Chipsets (comma):", chipsets_edit)
        form.addRow("Min BIOS:", bios_edit)
        form.addRow("Family:", family_edit)
        form.addRow("Min Microcode (hex):", microcode_edit)
        form.addRow("Max Memory Speed:", mem_speed_edit)
        form.addRow("ECC Support (True/False):", ecc_edit)
        form.addRow("Sockets Required:", sockets_req_edit)
        form.addRow("NUMA Nodes:", numa_nodes_edit)
        form.addRow("Memory Channels:", mem_channels_edit)

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
                "sockets_required": int(sockets_req_edit.text().strip()),
                "numa_nodes": int(numa_nodes_edit.text().strip()),
                "memory_channels": int(mem_channels_edit.text().strip()),
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
        sockets_edit = QtWidgets.QLineEdit("1")
        numa_nodes_edit = QtWidgets.QLineEdit("1")
        dmi_bw_edit = QtWidgets.QLineEdit("8.0")
        nvme_slots_edit = QtWidgets.QLineEdit("1")
        pcie_gen_edit = QtWidgets.QLineEdit("3")
        raid_levels_edit = QtWidgets.QLineEdit("0,1,10")
        base_power_edit = QtWidgets.QLineEdit("50")

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
        form.addRow("CPU Sockets:", sockets_edit)
        form.addRow("NUMA Nodes:", numa_nodes_edit)
        form.addRow("DMI Bandwidth (Gbps):", dmi_bw_edit)
        form.addRow("NVMe Slots:", nvme_slots_edit)
        form.addRow("PCIe Gen:", pcie_gen_edit)
        form.addRow("RAID Levels (comma):", raid_levels_edit)
        form.addRow("Base Board Power (W):", base_power_edit)

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
                "cpu_sockets": int(sockets_edit.text().strip()),
                "numa_nodes": int(numa_nodes_edit.text().strip()),
                "dmi_bandwidth_gbps": float(dmi_bw_edit.text().strip()),
                "nvme_slots": int(nvme_slots_edit.text().strip()),
                "pcie_gen": int(pcie_gen_edit.text().strip()),
                "raid_levels": [r.strip() for r in raid_levels_edit.text().split(",") if r.strip()],
                "base_board_power": int(base_power_edit.text().strip()),
            }
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid Input", str(e))
            return

        self.db.add_board(name, profile)
        self.refresh_profile_view()
        self.refresh_diff_tab()
        self.refresh_matrix_tab()

    # ---------------- PANEL: XEON / EPYC MATRIX ----------------

    def _build_panel_matrix(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("Xeon / EPYC Workstation Compatibility Matrix"))

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
        self.matrix_view.setText("\n".join(lines) if lines else "No Xeon/EPYC entries in database.")

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
                f"GPU={item['gpu']}  MEM={item['memory']}  Storage={item['storage']}  "
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
        content.append("   - Description: Simulation of CPU/board/GPU/memory/storage using sockets, TDP, VRM, BIOS, chipset rules,")
        content.append("     PCIe topology, NUMA, memory bandwidth, DMI/PCIe bottlenecks, PSU sizing.")
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
