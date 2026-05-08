#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Workstation / Xeon Hardware Lab v6.0

Adds on top of v5.0:
- GPU thermal + power modeling
- Real-time GPU sensors (best-effort via nvidia-smi)
- Fan curve simulation (CPU + GPU, heuristic)
- Boost clock prediction (CPU + GPU, heuristic)
- PCIe lane allocator visualizer (multi-GPU + NVMe)
- ML-style workload predictor (rule-based classifier)
- Full build BOM generator
- Cooling system advisor
- Power transient modeling (spikes)
- Multi-GPU + NVLink modeling

All previous features preserved:
- CPU/board/VRM/NUMA/PCIe/DMI/RAID/PSU/thermal
- GPU compute scoring
- Suggested build engine
- AI-style build optimizer
- GUI cockpit with multiple panels
"""

import sys
import os
import json
import platform
import subprocess
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, List, Optional

from PySide6 import QtWidgets, QtCore, QtGui

# Optional libs
try:
    import cpuinfo
except Exception:
    cpuinfo = None

try:
    import wmi
except Exception:
    wmi = None

try:
    import psutil
except Exception:
    psutil = None

# -----------------------------
# DATABASES
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
        "base_clock_ghz": 3.0,
        "boost_clock_ghz": 4.6,
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
        "base_clock_ghz": 3.8,
        "boost_clock_ghz": 4.2,
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
        "base_clock_ghz": 4.0,
        "boost_clock_ghz": 4.2,
    },
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
        "base_clock_ghz": 3.0,
        "boost_clock_ghz": 4.0,
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
        "base_clock_ghz": 2.9,
        "boost_clock_ghz": 3.4,
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
        "cooling_capacity_watts": 250,
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
        "cooling_capacity_watts": 180,
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
        "cooling_capacity_watts": 220,
    },
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
        "cooling_capacity_watts": 450,
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
        "cooling_capacity_watts": 500,
    },
}

GPU_DB = {
    "RTX 4090": {
        "tdp": 450,
        "slot_type": "x16",
        "min_lanes": 8,
        "recommended_psu": 1000,
        "tier": "ultra",
        "pcie_gen": 4,
        "fp32_tflops": 82.0,
        "base_clock_mhz": 2235,
        "boost_clock_mhz": 2520,
        "thermal_limit_c": 83,
        "transient_spike_factor": 1.4,
        "nvlink": False,
    },
    "RTX 3080": {
        "tdp": 320,
        "slot_type": "x16",
        "min_lanes": 8,
        "recommended_psu": 750,
        "tier": "high",
        "pcie_gen": 4,
        "fp32_tflops": 29.8,
        "base_clock_mhz": 1440,
        "boost_clock_mhz": 1710,
        "thermal_limit_c": 83,
        "transient_spike_factor": 1.35,
        "nvlink": False,
    },
    "Quadro RTX 4000": {
        "tdp": 160,
        "slot_type": "x16",
        "min_lanes": 8,
        "recommended_psu": 550,
        "tier": "pro",
        "pcie_gen": 3,
        "fp32_tflops": 7.1,
        "base_clock_mhz": 1005,
        "boost_clock_mhz": 1545,
        "thermal_limit_c": 80,
        "transient_spike_factor": 1.25,
        "nvlink": False,
    },
    "RTX 6000 Ada": {
        "tdp": 300,
        "slot_type": "x16",
        "min_lanes": 16,
        "recommended_psu": 800,
        "tier": "pro_ultra",
        "pcie_gen": 4,
        "fp32_tflops": 91.0,
        "base_clock_mhz": 2200,
        "boost_clock_mhz": 2500,
        "thermal_limit_c": 82,
        "transient_spike_factor": 1.35,
        "nvlink": True,
    },
}

MEMORY_DB = {
    "DDR4-2133 16GB ECC": {"type": "DDR4", "speed": 2133, "size_gb": 16, "ecc": True},
    "DDR4-3200 16GB Non-ECC": {"type": "DDR4", "speed": 3200, "size_gb": 16, "ecc": False},
    "DDR4-2933 32GB ECC": {"type": "DDR4", "speed": 2933, "size_gb": 32, "ecc": True},
}

STORAGE_DB = {
    "Single NVMe": {"nvme_count": 1, "raid_level": "0", "pcie_gen_required": 3},
    "Dual NVMe RAID1": {"nvme_count": 2, "raid_level": "1", "pcie_gen_required": 3},
    "Quad NVMe RAID10": {"nvme_count": 4, "raid_level": "10", "pcie_gen_required": 3},
    "Dual NVMe Gen4": {"nvme_count": 2, "raid_level": "0", "pcie_gen_required": 4},
}

LATEST_BIOS = {
    "ASUS WS C422 PRO": 1301,
    "ASUS Z170-A": 3805,
    "ASUS W480": 1202,
    "Supermicro X11DPi-NT (Dual 3647)": 1300,
    "EPYC SP3-WS Board": 1100,
}

CHIPSET_RULES = {
    "C422": {"families": ["Xeon W"], "ecc_required": True,
             "workloads": ["rendering", "virtualization", "balanced", "blender", "ai_inference"]},
    "C236": {"families": ["Xeon E3", "Core i7"], "ecc_optional": True,
             "workloads": ["balanced", "virtualization"]},
    "Z170": {"families": ["Core i7"], "gaming_preferred": True,
             "workloads": ["gaming", "balanced", "unreal"]},
    "Z270": {"families": ["Core i7", "Xeon E3"],
             "workloads": ["gaming", "balanced", "unreal"]},
    "W480": {"families": ["Xeon W", "Core i7"], "ecc_optional": True,
             "workloads": ["balanced", "virtualization", "rendering", "blender"]},
    "X299": {"families": ["Xeon W", "Core i9"],
             "workloads": ["rendering", "balanced", "blender"]},
    "C621": {"families": ["Xeon Scalable"], "ecc_required": True,
             "workloads": ["virtualization", "rendering", "ai_inference"]},
    "SP3-WS": {"families": ["EPYC"], "ecc_required": True,
               "workloads": ["virtualization", "rendering", "ai_inference"]},
}

INVENTORY: List[Dict[str, Any]] = []
TIMELINE: List[str] = []

# -----------------------------
# HARDWARE DETECTOR + SENSORS
# -----------------------------

class HardwareDetector:
    @staticmethod
    def detect_cpu():
        try:
            if cpuinfo:
                brand = cpuinfo.get_cpu_info().get("brand_raw")
                if brand:
                    return brand
        except:
            pass
        return platform.processor() or "Unknown CPU"

    @staticmethod
    def detect_board():
        if os.name == "nt" and wmi:
            try:
                c = wmi.WMI()
                for b in c.Win32_BaseBoard():
                    return getattr(b, "Manufacturer", "Unknown"), getattr(b, "Product", "Unknown")
            except:
                pass

        if os.name == "posix":
            try:
                out = subprocess.check_output(["dmidecode", "-t", "baseboard"], text=True)
                vendor = "Unknown"
                product = "Unknown"
                for line in out.splitlines():
                    line = line.strip()
                    if line.startswith("Manufacturer:"):
                        vendor = line.split(":", 1)[1].strip()
                    elif line.startswith("Product Name:"):
                        product = line.split(":", 1)[1].strip()
                return vendor, product
            except:
                pass

        return "Unknown Vendor", "Unknown Board"

    @staticmethod
    def fingerprint():
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

    @staticmethod
    def read_cpu_sensors():
        data = {
            "cpu_load_percent": None,
            "cpu_temp_c": None,
        }
        if not psutil:
            return data

        try:
            data["cpu_load_percent"] = psutil.cpu_percent(interval=0.1)
        except:
            pass

        try:
            temps = psutil.sensors_temperatures()
            for name, entries in temps.items():
                for e in entries:
                    label = (e.label or "").lower()
                    if "cpu" in label or "package" in label or "core 0" in label:
                        data["cpu_temp_c"] = e.current
                        break
                if data["cpu_temp_c"] is not None:
                    break
        except:
            pass

        return data

    @staticmethod
    def read_gpu_sensors():
        """
        Best-effort GPU sensors via nvidia-smi.
        Returns list of dicts: [{index, name, temp_c, power_w, util_percent}, ...]
        """
        gpus = []
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "temp_c": float(parts[2]),
                        "power_w": float(parts[3]),
                        "util_percent": float(parts[4]),
                    })
        except:
            pass
        return gpus

    @staticmethod
    def read_all_sensors():
        cpu = HardwareDetector.read_cpu_sensors()
        gpu = HardwareDetector.read_gpu_sensors()
        return {"cpu": cpu, "gpu": gpu}

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

    def get_cpu_names(self):
        return sorted(self.cpu_db.keys())

    def get_board_names(self):
        return sorted(self.board_db.keys())

    def get_gpu_names(self):
        return sorted(self.gpu_db.keys())

    def get_mem_names(self):
        return sorted(self.mem_db.keys())

    def get_storage_names(self):
        return sorted(self.storage_db.keys())

    def add_cpu(self, name, profile):
        self.cpu_db[name] = profile

    def add_board(self, name, profile):
        self.board_db[name] = profile

    def export_to_file(self, path):
        data = {
            "cpus": self.cpu_db,
            "boards": self.board_db,
            "gpus": self.gpu_db,
            "memory": self.mem_db,
            "storage": self.storage_db,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def import_from_file(self, path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            return False, f"Failed to read file: {e}"

        if not isinstance(data, dict):
            return False, "Invalid file format."

        cpus = data.get("cpus")
        boards = data.get("boards")
        gpus = data.get("gpus", {})
        mems = data.get("memory", {})
        storage = data.get("storage", {})

        if not isinstance(cpus, dict) or not isinstance(boards, dict):
            return False, "Invalid schema."

        self.cpu_db = cpus
        self.board_db = boards
        self.gpu_db = gpus
        self.mem_db = mems
        self.storage_db = storage
        return True, "Import successful."

    @staticmethod
    def diff_profiles(a, b):
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
    def pcie_bifurcation(board):
        lines = []
        for s in board.get("pcie_slots", []):
            lines.append(
                f"{s['name']}: {s['type']} ({s['lanes']} lanes, {s['wired_to']}, bifurcation={s.get('bifurcation','n/a')})"
            )
        return lines

    @staticmethod
    def numa_topology(cpu, board):
        return {
            "cpu_numa_nodes": cpu.get("numa_nodes", 1),
            "board_numa_nodes": board.get("numa_nodes", 1),
            "cpu_sockets_required": cpu.get("sockets_required", 1),
            "board_cpu_sockets": board.get("cpu_sockets", 1),
            "numa_mismatch": (
                cpu.get("numa_nodes", 1) != board.get("numa_nodes", 1)
                or cpu.get("sockets_required", 1) > board.get("cpu_sockets", 1)
            ),
        }

    @staticmethod
    def memory_bandwidth(cpu, board):
        channels = cpu.get("memory_channels", 2)
        speed = min(cpu.get("max_memory_speed", 2133), board.get("max_memory_speed", 2133))
        return round(channels * speed * 8 / 1000.0, 1)

    @staticmethod
    def dmi_bottleneck(board, gpus, storage):
        dmi_bw = board.get("dmi_bandwidth_gbps", 8.0)
        nvme_count = storage.get("nvme_count", 0) if storage else 0
        pcie_gen = board.get("pcie_gen", 3)

        nvme_bw = nvme_count * (3.5 if pcie_gen == 3 else 7.0)
        gpu_bw = 0.0
        for gpu in gpus or []:
            if gpu and gpu.get("pcie_gen", 4) >= 4:
                gpu_bw += 8.0
            else:
                gpu_bw += 4.0

        total = nvme_bw + gpu_bw
        return {
            "dmi_bandwidth_gbps": dmi_bw,
            "estimated_nvme_traffic_gbps": nvme_bw,
            "estimated_gpu_traffic_gbps": gpu_bw,
            "estimated_total_traffic_gbps": total,
            "dmi_saturated": total > dmi_bw,
        }

    @staticmethod
    def pcie_lane_allocation(board, gpus, storage):
        """
        Simple allocator: assign GPUs to x16 slots, NVMe to remaining lanes.
        Returns list of allocations.
        """
        slots = board.get("pcie_slots", [])
        allocations = []
        remaining_lanes = board.get("pci_lanes", 0)

        # GPUs first
        gpu_index = 0
        for gpu in gpus or []:
            if not gpu:
                continue
            assigned = None
            for s in slots:
                if s["type"] == "x16" and s["lanes"] >= gpu.get("min_lanes", 8):
                    assigned = s["name"]
                    break
            lanes_used = gpu.get("min_lanes", 8)
            remaining_lanes -= lanes_used
            allocations.append({
                "device": f"GPU{gpu_index}",
                "model": gpu.get("model_name", "GPU"),
                "slot": assigned or "shared",
                "lanes": lanes_used,
            })
            gpu_index += 1

        # NVMe
        nvme_count = storage.get("nvme_count", 0) if storage else 0
        for i in range(nvme_count):
            lanes_used = 4
            remaining_lanes -= lanes_used
            allocations.append({
                "device": f"NVMe{i}",
                "model": "NVMe SSD",
                "slot": "PCH/CPU shared",
                "lanes": lanes_used,
            })

        return {
            "allocations": allocations,
            "remaining_lanes": remaining_lanes,
        }

# -----------------------------
# POWER + THERMAL MODELING
# -----------------------------

class PowerEngine:
    @staticmethod
    def estimate_system_power(cpu, board, gpus, storage):
        cpu_tdp = cpu.get("tdp", 0)
        gpu_tdp = sum(g.get("tdp", 0) for g in gpus if g)
        board_power = board.get("base_board_power", 50)
        nvme_count = storage.get("nvme_count", 0) if storage else 0
        nvme_power = nvme_count * 7
        misc = 40

        raw_total = cpu_tdp + gpu_tdp + board_power + nvme_power + misc

        # Transient spikes (GPU heavy)
        spike_factor = 1.0
        for g in gpus:
            if g:
                spike_factor = max(spike_factor, g.get("transient_spike_factor", 1.0))
        transient_peak = raw_total * spike_factor

        vrm_loss = raw_total * (1 / 0.9 - 1)
        recommended_psu = int((transient_peak + vrm_loss) / 0.7 + 50)

        return {
            "cpu_tdp": cpu_tdp,
            "gpu_tdp": gpu_tdp,
            "board_power": board_power,
            "nvme_power": nvme_power,
            "misc_power": misc,
            "raw_total_power": int(raw_total),
            "transient_peak_power": int(transient_peak),
            "vrm_loss_power": int(vrm_loss),
            "recommended_psu_watts": recommended_psu,
        }

    @staticmethod
    def evaluate_psu_margin(power_info, gpus):
        rec = power_info["recommended_psu_watts"]
        warnings = []
        max_gpu_rec = 0
        for g in gpus:
            if g:
                max_gpu_rec = max(max_gpu_rec, g.get("recommended_psu", 0))
        if max_gpu_rec and rec < max_gpu_rec:
            warnings.append(
                f"Estimated PSU ({rec}W) is below highest GPU vendor recommendation ({max_gpu_rec}W)."
            )
        return rec, warnings

class ThermalModel:
    @staticmethod
    def estimate_cpu_thermal(cpu, board, sensors):
        cpu_tdp = cpu.get("tdp", 0)
        cooling_cap = board.get("cooling_capacity_watts", cpu_tdp + 50)
        vrm_limit = board.get("vrm_current_limit", 0)

        headroom_watts = cooling_cap - cpu_tdp
        headroom_ratio = headroom_watts / cooling_cap if cooling_cap > 0 else 0

        temp = sensors.get("cpu", {}).get("cpu_temp_c")
        temp_note = None
        if temp is not None:
            if temp > 90:
                temp_note = "CPU temperature is very high; thermal throttling likely."
            elif temp > 80:
                temp_note = "CPU temperature is elevated; consider better cooling."
            elif temp > 70:
                temp_note = "CPU temperature is moderate; monitor under sustained load."
            else:
                temp_note = "CPU temperature appears comfortable."

        vrm_note = None
        if vrm_limit:
            est_current = cpu_tdp / (1.2 * 0.9)
            if est_current > vrm_limit:
                vrm_note = "VRM current limit exceeded under full TDP."
            elif est_current > vrm_limit * 0.9:
                vrm_note = "VRM current near limit; limited OC headroom."
            else:
                vrm_note = "VRM current within comfortable range."

        return {
            "cooling_capacity_watts": cooling_cap,
            "cpu_tdp": cpu_tdp,
            "thermal_headroom_watts": headroom_watts,
            "thermal_headroom_ratio": round(headroom_ratio, 2),
            "sensor_cpu_temp_c": temp,
            "temp_note": temp_note,
            "vrm_note": vrm_note,
        }

    @staticmethod
    def estimate_gpu_thermal(gpus, sensors):
        gpu_sensors = sensors.get("gpu", [])
        results = []
        for idx, g in enumerate(gpus):
            if not g:
                continue
            t_limit = g.get("thermal_limit_c", 80)
            sensor = None
            for s in gpu_sensors:
                if s["index"] == idx:
                    sensor = s
                    break
            temp = sensor["temp_c"] if sensor else None
            note = None
            if temp is not None:
                if temp > t_limit:
                    note = f"GPU{idx} temperature above thermal limit; throttling likely."
                elif temp > t_limit - 5:
                    note = f"GPU{idx} temperature near thermal limit; consider more airflow."
                else:
                    note = f"GPU{idx} temperature within comfortable range."
            results.append({
                "index": idx,
                "model": g.get("model_name", "GPU"),
                "thermal_limit_c": t_limit,
                "sensor_temp_c": temp,
                "note": note,
            })
        return results

class FanCurveSimulator:
    @staticmethod
    def cpu_fan_percent(temp_c):
        if temp_c is None:
            return None
        if temp_c < 40:
            return 20
        if temp_c < 60:
            return 40
        if temp_c < 75:
            return 60
        if temp_c < 85:
            return 80
        return 100

    @staticmethod
    def gpu_fan_percent(temp_c, limit_c):
        if temp_c is None:
            return None
        if temp_c < limit_c - 30:
            return 25
        if temp_c < limit_c - 15:
            return 50
        if temp_c < limit_c - 5:
            return 75
        return 100

class BoostPredictor:
    @staticmethod
    def cpu_boost_estimate(cpu, sensors):
        base = cpu.get("base_clock_ghz", 3.0)
        boost = cpu.get("boost_clock_ghz", 4.0)
        temp = sensors.get("cpu", {}).get("cpu_temp_c")
        load = sensors.get("cpu", {}).get("cpu_load_percent")

        if temp is None or load is None:
            return boost

        if temp > 90 or load > 95:
            return round(base + (boost - base) * 0.3, 2)
        if temp > 80 or load > 80:
            return round(base + (boost - base) * 0.6, 2)
        return boost

    @staticmethod
    def gpu_boost_estimate(gpu, sensor):
        base = gpu.get("base_clock_mhz", 1500)
        boost = gpu.get("boost_clock_mhz", 1800)
        temp = sensor.get("temp_c") if sensor else None
        util = sensor.get("util_percent") if sensor else None

        if temp is None or util is None:
            return boost

        if temp > gpu.get("thermal_limit_c", 80) or util > 95:
            return int(base + (boost - base) * 0.4)
        if temp > gpu.get("thermal_limit_c", 80) - 5 or util > 85:
            return int(base + (boost - base) * 0.7)
        return boost

# -----------------------------
# GPU COMPUTE SCORING
# -----------------------------

class GPUComputeScorer:
    @staticmethod
    def score_gpu_for_workload(gpu, workload):
        if not gpu:
            return 0, "No GPU selected."

        tflops = gpu.get("fp32_tflops", 0.0)
        tier = gpu.get("tier", "")

        base_score = tflops
        notes = [f"GPU FP32 compute: {tflops} TFLOPs (tier={tier})"]

        if workload in ("ai_inference", "ai", "blender", "rendering", "unreal"):
            base_score *= 1.2
            notes.append("Workload benefits strongly from GPU compute; boosting GPU score.")
        elif workload in ("gaming",):
            base_score *= 1.0
            notes.append("Gaming: GPU compute important but also latency/driver; neutral scaling.")
        else:
            base_score *= 0.8
            notes.append("Workload not heavily GPU-bound; reduced weight.")

        return int(base_score), "; ".join(notes)

# -----------------------------
# WORKLOAD SCORING
# -----------------------------

class WorkloadScorer:
    @staticmethod
    def score(workload, cpu, board, gpus, mem_bw, dmi_info):
        score_delta = 0
        notes = []
        cores = cpu.get("cores", 0)
        family = cpu.get("family", "")
        gpu_tiers = [g.get("tier", "") for g in gpus if g]
        numa_nodes = cpu.get("numa_nodes", 1)

        if workload in ("rendering", "blender"):
            if cores >= 16:
                score_delta += 15
                notes.append("Rendering: High core count ideal.")
            if mem_bw < 50:
                score_delta -= 10
                notes.append("Rendering: Memory bandwidth may limit performance.")
            if len(gpu_tiers) >= 2:
                score_delta += 5
                notes.append("Rendering: Multi-GPU beneficial for some workloads.")

        elif workload in ("gaming", "unreal"):
            if "Xeon" in family and cores > 8:
                score_delta -= 5
                notes.append("Gaming: Xeon functional but not optimal for FPS.")
            if any(t in ("ultra", "high") for t in gpu_tiers):
                score_delta += 10
                notes.append("Gaming: Strong GPU tier.")

        elif workload in ("virtualization", "virtualization_density"):
            if cores < 12:
                score_delta -= 10
                notes.append("Virtualization: More cores recommended.")
            if numa_nodes > 1:
                score_delta += 5
                notes.append("Virtualization: NUMA-aware scaling beneficial.")

        elif workload in ("ai_inference", "ai"):
            if any(t in ("ultra", "high", "pro_ultra") for t in gpu_tiers):
                score_delta += 15
                notes.append("AI: Strong GPU tier.")
            if mem_bw < 60:
                score_delta -= 5
                notes.append("AI: Memory bandwidth may limit model throughput.")
            if any(g.get("nvlink") for g in gpus if g):
                score_delta += 5
                notes.append("AI: NVLink-capable GPU beneficial for multi-GPU scaling.")

        if dmi_info.get("dmi_saturated"):
            score_delta -= 10
            notes.append("Platform: DMI/PCIe link may saturate under heavy IO.")

        return score_delta, notes

# -----------------------------
# ML-STYLE WORKLOAD PREDICTOR
# -----------------------------

class WorkloadPredictor:
    """
    Rule-based classifier that guesses workload from:
    - GPU utilization
    - CPU utilization
    - Selected GPU tier
    - Memory bandwidth
    """

    @staticmethod
    def predict(cpu, gpus, mem_bw, sensors):
        cpu_load = sensors.get("cpu", {}).get("cpu_load_percent") or 0
        gpu_sensors = sensors.get("gpu", [])
        max_gpu_util = max((g.get("util_percent", 0) for g in gpu_sensors), default=0)
        gpu_tiers = [g.get("tier", "") for g in gpus if g]

        if max_gpu_util > 80 and cpu_load < 60 and any(t in ("ultra", "high") for t in gpu_tiers):
            return "gaming"
        if cpu_load > 80 and max_gpu_util > 50 and mem_bw > 50:
            return "blender"
        if cpu_load > 70 and mem_bw > 60 and len(gpus) >= 1 and any(t in ("pro_ultra", "ultra") for t in gpu_tiers):
            return "ai_inference"
        if cpu_load > 60 and mem_bw < 40:
            return "virtualization_density"
        return "balanced"

# -----------------------------
# RULE ENGINE
# -----------------------------

class RuleEngine:
    def evaluate(self, cpu, board, gpus=None, mem=None, storage=None, workload=None, sensors=None):
        gpus = gpus or []
        result = {
            "score": 100,
            "compatible": True,
            "warnings": [],
            "errors": [],
            "vrm_headroom": None,
            "platform": {},
            "power": {},
            "workload_notes": [],
            "thermal": {},
            "gpu_thermal": [],
            "gpu_compute": {},
            "pcie_allocation": {},
            "fan_curves": {},
            "boost": {},
        }

        if cpu.get("socket") != board.get("socket"):
            result["compatible"] = False
            result["score"] -= 70
            result["errors"].append("Socket mismatch")

        if cpu.get("sockets_required", 1) > board.get("cpu_sockets", 1):
            result["compatible"] = False
            result["score"] -= 50
            result["errors"].append("Board does not support required CPU socket count")

        if cpu.get("tdp", 0) > board.get("max_tdp", 0):
            result["score"] -= 40
            result["warnings"].append("CPU TDP exceeds board max TDP")

        if cpu.get("min_bios", 0) > board.get("bios_version", 0):
            result["score"] -= 30
            result["warnings"].append("Board BIOS below CPU minimum requirement")

        chipset = board.get("chipset")
        family = cpu.get("family", "")
        if chipset in CHIPSET_RULES:
            rule = CHIPSET_RULES[chipset]
            if rule.get("families") and not any(f in family for f in rule["families"]):
                result["score"] -= 25
                result["warnings"].append(f"CPU family '{family}' not ideal for chipset {chipset}")
            if workload and workload not in rule.get("workloads", []):
                result["score"] -= 10
                result["warnings"].append(f"Chipset {chipset} not tuned for workload '{workload}'")
        else:
            result["score"] -= 5
            result["warnings"].append(f"Unknown chipset {chipset}")

        if cpu.get("min_microcode", 0) > board.get("microcode_version", 0):
            result["score"] -= 15
            result["warnings"].append("Board microcode may be below CPU minimum")

        vrm_limit = board.get("vrm_current_limit", 0)
        cpu_tdp = cpu.get("tdp", 0)
        if vrm_limit and cpu_tdp:
            estimated_current = cpu_tdp / (1.2 * 0.9)
            headroom = vrm_limit - estimated_current
            result["vrm_headroom"] = round(headroom, 1)
            if headroom < 0:
                result["score"] -= 35
                result["warnings"].append("VRM headroom negative")
            elif headroom < 20:
                result["score"] -= 10
                result["warnings"].append("VRM headroom low")

        if gpus:
            slots = board.get("pcie_slots", [])
            for idx, gpu in enumerate(gpus):
                if not gpu:
                    continue
                has_slot = any(
                    s["type"] == gpu.get("slot_type", "x16") and s["lanes"] >= gpu.get("min_lanes", 8)
                    for s in slots
                )
                if not has_slot:
                    result["score"] -= 25
                    result["warnings"].append(f"No suitable PCIe slot for GPU{idx}")
                if board.get("pcie_gen", 3) < gpu.get("pcie_gen", 3):
                    result["score"] -= 5
                    result["warnings"].append(f"Board PCIe gen below GPU{idx} capability")

        if mem:
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
                result["warnings"].append("Memory speed exceeds CPU/board rating")

            if mem_ecc and not (ecc_cpu and ecc_board):
                result["score"] -= 10
                result["warnings"].append("ECC memory used but CPU/board may not support ECC")

            if slots and max_mem:
                total_mem = mem_size * slots
                if total_mem > max_mem:
                    result["score"] -= 10
                    result["warnings"].append("Memory capacity exceeds board maximum")

        if storage:
            nvme_count = storage.get("nvme_count", 0)
            if nvme_count > board.get("nvme_slots", 0):
                result["score"] -= 15
                result["warnings"].append("NVMe count exceeds board slots")

            raid_level = storage.get("raid_level", "0")
            if raid_level not in board.get("raid_levels", []):
                result["score"] -= 10
                result["warnings"].append(f"Board does not support RAID level {raid_level}")

            if board.get("pcie_gen", 3) < storage.get("pcie_gen_required", 3):
                result["score"] -= 5
                result["warnings"].append("Board PCIe gen may limit NVMe performance")

        mem_bw = PlatformModel.memory_bandwidth(cpu, board)
        numa_info = PlatformModel.numa_topology(cpu, board)
        dmi_info = PlatformModel.dmi_bottleneck(board, gpus, storage)
        pcie_bifurcation = PlatformModel.pcie_bifurcation(board)
        pcie_alloc = PlatformModel.pcie_lane_allocation(board, gpus, storage)

        result["platform"] = {
            "memory_bandwidth_gbps": mem_bw,
            "numa": numa_info,
            "dmi": dmi_info,
            "pcie_bifurcation": pcie_bifurcation,
        }
        result["pcie_allocation"] = pcie_alloc

        power_info = PowerEngine.estimate_system_power(cpu, board, gpus, storage)
        psu_rec, psu_warnings = PowerEngine.evaluate_psu_margin(power_info, gpus)
        result["power"] = {
            "power_model": power_info,
            "recommended_psu_watts": psu_rec,
        }
        result["warnings"].extend(psu_warnings)

        if sensors is None:
            sensors = {"cpu": {}, "gpu": []}
        cpu_thermal = ThermalModel.estimate_cpu_thermal(cpu, board, sensors)
        gpu_thermal = ThermalModel.estimate_gpu_thermal(gpus, sensors)
        result["thermal"] = cpu_thermal
        result["gpu_thermal"] = gpu_thermal

        if cpu_thermal.get("temp_note"):
            result["warnings"].append(cpu_thermal["temp_note"])
        if cpu_thermal.get("vrm_note"):
            result["warnings"].append(cpu_thermal["vrm_note"])
        for gt in gpu_thermal:
            if gt.get("note"):
                result["warnings"].append(gt["note"])

        if workload:
            wl_score, wl_notes = WorkloadScorer.score(workload, cpu, board, gpus, mem_bw, dmi_info)
            result["score"] += wl_score
            result["workload_notes"] = wl_notes

            # GPU compute scoring: use strongest GPU
            main_gpu = None
            if gpus:
                main_gpu = max((g for g in gpus if g), key=lambda x: x.get("fp32_tflops", 0), default=None)
            gpu_score, gpu_note = GPUComputeScorer.score_gpu_for_workload(main_gpu, workload)
            result["score"] += int(gpu_score / 10)
            result["gpu_compute"] = {"score": gpu_score, "note": gpu_note}

        # Fan curves + boost prediction
        cpu_temp = sensors.get("cpu", {}).get("cpu_temp_c")
        cpu_fan = FanCurveSimulator.cpu_fan_percent(cpu_temp)
        cpu_boost = BoostPredictor.cpu_boost_estimate(cpu, sensors)

        gpu_fans = []
        gpu_boosts = []
        gpu_sensors = sensors.get("gpu", [])
        for idx, g in enumerate(gpus):
            if not g:
                gpu_fans.append(None)
                gpu_boosts.append(None)
                continue
            sensor = None
            for s in gpu_sensors:
                if s["index"] == idx:
                    sensor = s
                    break
            temp = sensor["temp_c"] if sensor else None
            fan = FanCurveSimulator.gpu_fan_percent(temp, g.get("thermal_limit_c", 80))
            boost = BoostPredictor.gpu_boost_estimate(g, sensor)
            gpu_fans.append(fan)
            gpu_boosts.append(boost)

        result["fan_curves"] = {
            "cpu_fan_percent": cpu_fan,
            "gpu_fan_percent": gpu_fans,
        }
        result["boost"] = {
            "cpu_boost_ghz_est": cpu_boost,
            "gpu_boost_mhz_est": gpu_boosts,
        }

        if result["score"] < 0:
            result["score"] = 0

        return result

# -----------------------------
# SIMULATOR
# -----------------------------

class Simulator:
    def __init__(self, db):
        self.db = db
        self.engine = RuleEngine()

    def run(self, cpu_name, board_name, gpu1_name=None, gpu2_name=None,
            mem_name=None, storage_name=None, workload=None, sensors=None):
        cpu = self.db.cpu_db[cpu_name]
        board = self.db.board_db[board_name]
        gpus = []
        if gpu1_name:
            g = deepcopy(self.db.gpu_db[gpu1_name])
            g["model_name"] = gpu1_name
            gpus.append(g)
        if gpu2_name:
            g2 = deepcopy(self.db.gpu_db[gpu2_name])
            g2["model_name"] = gpu2_name
            g2["nvlink_peer"] = bool(g2.get("nvlink"))
            gpus.append(g2)

        mem = self.db.mem_db[mem_name] if mem_name else None
        storage = self.db.storage_db[storage_name] if storage_name else None

        result = self.engine.evaluate(cpu, board, gpus, mem, storage, workload, sensors)
        return {
            "cpu": cpu_name,
            "board": board_name,
            "gpu_primary": gpu1_name,
            "gpu_secondary": gpu2_name,
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
            "thermal": result["thermal"],
            "gpu_thermal": result["gpu_thermal"],
            "gpu_compute": result["gpu_compute"],
            "pcie_allocation": result["pcie_allocation"],
            "fan_curves": result["fan_curves"],
            "boost": result["boost"],
        }

# -----------------------------
# AI-STYLE BUILD OPTIMIZER
# -----------------------------

class AIBuildOptimizer:
    def __init__(self, db: ProfileDB):
        self.db = db
        self.engine = RuleEngine()

    def optimize(self, workload: str) -> Optional[Dict[str, Any]]:
        best = None
        best_score = -1

        for cpu_name, cpu in self.db.cpu_db.items():
            for board_name, board in self.db.board_db.items():
                if cpu.get("socket") != board.get("socket"):
                    continue
                if cpu.get("sockets_required", 1) > board.get("cpu_sockets", 1):
                    continue

                for gpu1_name, gpu1 in self.db.gpu_db.items():
                    # Optionally try second GPU for heavy workloads
                    gpu2_name = None
                    if workload in ("blender", "ai_inference") and gpu1.get("nvlink"):
                        gpu2_name = gpu1_name

                    mem_name = self._pick_memory(cpu, board, workload)
                    storage_name = self._pick_storage(board, workload)

                    mem = self.db.mem_db.get(mem_name)
                    storage = self.db.storage_db.get(storage_name)

                    sensors = HardwareDetector.read_all_sensors()
                    gpus = []
                    g1 = deepcopy(gpu1)
                    g1["model_name"] = gpu1_name
                    gpus.append(g1)
                    if gpu2_name:
                        g2 = deepcopy(self.db.gpu_db[gpu2_name])
                        g2["model_name"] = gpu2_name
                        g2["nvlink_peer"] = bool(g2.get("nvlink"))
                        gpus.append(g2)

                    result = self.engine.evaluate(cpu, board, gpus, mem, storage, workload, sensors)

                    score = result["score"]
                    score += self._efficiency_bonus(cpu, board, gpus, result)
                    score -= self._overkill_penalty(cpu, board, gpus, workload)

                    if score > best_score:
                        best_score = score
                        best = {
                            "cpu": cpu_name,
                            "board": board_name,
                            "gpu_primary": gpu1_name,
                            "gpu_secondary": gpu2_name,
                            "memory": mem_name,
                            "storage": storage_name,
                            "score": score,
                            "engine_result": result,
                        }

        return best

    def _efficiency_bonus(self, cpu, board, gpus, result):
        power = result["power"]["power_model"]
        total = power["raw_total_power"]
        rec_psu = result["power"]["recommended_psu_watts"]
        ratio = total / rec_psu if rec_psu else 1.0
        bonus = 0
        if 0.4 < ratio < 0.7:
            bonus += 5
        if any(g and g.get("tier") in ("pro", "high", "pro_ultra") for g in gpus) and cpu.get("cores", 0) >= 8:
            bonus += 5
        return bonus

    def _overkill_penalty(self, cpu, board, gpus, workload):
        penalty = 0
        if workload in ("gaming", "unreal"):
            if "EPYC" in cpu.get("family", "") or "Xeon Scalable" in cpu.get("family", ""):
                penalty += 10
            if len(gpus) > 1:
                penalty += 5
        if workload in ("balanced",) and any(g and g.get("tier") == "ultra" for g in gpus):
            penalty += 5
        return penalty

    def _pick_memory(self, cpu, board, workload):
        best_name = None
        best_score = -1
        for name, mem in self.db.mem_db.items():
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

    def _pick_storage(self, board, workload):
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
# SUGGESTED BUILD ENGINE (v4-style)
# -----------------------------

class SuggestedBuildEngine:
    def __init__(self, db):
        self.db = db
        self.engine = RuleEngine()

    def suggest(self, workload):
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

                gpu1_name = self._pick_gpu(workload)
                gpu2_name = None
                if workload in ("blender", "ai_inference") and self.db.gpu_db[gpu1_name].get("nvlink"):
                    gpu2_name = gpu1_name

                mem_name = self._pick_memory(cpu, board, workload)
                storage_name = self._pick_storage(board, workload)

                mem = self.db.mem_db.get(mem_name)
                storage = self.db.storage_db.get(storage_name)

                sensors = HardwareDetector.read_all_sensors()
                gpus = []
                g1 = deepcopy(self.db.gpu_db[gpu1_name])
                g1["model_name"] = gpu1_name
                gpus.append(g1)
                if gpu2_name:
                    g2 = deepcopy(self.db.gpu_db[gpu2_name])
                    g2["model_name"] = gpu2_name
                    g2["nvlink_peer"] = bool(g2.get("nvlink"))
                    gpus.append(g2)

                result = self.engine.evaluate(cpu, board, gpus, mem, storage, workload, sensors)
                score = result["score"]

                if score > best_score:
                    best_score = score
                    best = {
                        "cpu": cpu_name,
                        "board": board_name,
                        "gpu_primary": gpu1_name,
                        "gpu_secondary": gpu2_name,
                        "memory": mem_name,
                        "storage": storage_name,
                        "score": score,
                        "result": result,
                    }

        return best

    def _cpu_matches_workload(self, cpu, workload):
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

    def _pick_gpu(self, workload):
        if workload in ("gaming", "unreal"):
            return "RTX 3080"
        if workload in ("rendering", "blender", "ai_inference", "ai"):
            return "RTX 6000 Ada"
        if workload in ("virtualization", "virtualization_density"):
            return "Quadro RTX 4000"
        return "RTX 3080"

    def _pick_memory(self, cpu, board, workload):
        best_name = None
        best_score = -1
        for name, mem in self.db.mem_db.items():
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

    def _pick_storage(self, board, workload):
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

def recommend(cpu_profile, workload):
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
# XEON / EPYC MATRIX
# -----------------------------

def build_xeon_matrix(db):
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
# BIOS FETCHER
# -----------------------------

def get_latest_bios(board_name):
    return LATEST_BIOS.get(board_name, 1000)

# -----------------------------
# PLUGIN SYSTEM
# -----------------------------

class PluginManager:
    def __init__(self, db):
        self.db = db
        self.plugins = []
        self.load_plugins()

    def load_plugins(self):
        self.plugins.clear()
        if not os.path.isdir("plugins"):
            return
        for f in os.listdir("plugins"):
            if f.endswith(".py"):
                try:
                    mod = __import__(f"plugins.{f[:-3]}", fromlist=[f[:-3]])
                    self.plugins.append(mod)
                except:
                    pass

    def run_post_simulation_hooks(self, report):
        messages = []
        for p in self.plugins:
            hook = getattr(p, "post_simulation", None)
            if callable(hook):
                try:
                    msg = hook(report)
                    if isinstance(msg, str) and msg.strip():
                        messages.append(msg.strip())
                except:
                    pass
        return messages

# -----------------------------
# NODE SYNC SIMULATION
# -----------------------------

class NodeManager:
    def __init__(self):
        self.nodes = []
        self._seed_nodes()

    def _seed_nodes(self):
        for i in range(3):
            self.nodes.append({
                "name": f"Node-{i+1}",
                "last_sync": None,
                "profiles_version": 1,
            })

    def sync_all(self):
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
# BOM GENERATOR + COOLING ADVISOR
# -----------------------------

class BOMGenerator:
    @staticmethod
    def generate(cpu_name, board_name, gpu1_name, gpu2_name, mem_name, storage_name, power_info):
        bom = []
        if cpu_name:
            bom.append({"category": "CPU", "item": cpu_name})
        if board_name:
            bom.append({"category": "Motherboard", "item": board_name})
        if gpu1_name:
            bom.append({"category": "GPU", "item": gpu1_name})
        if gpu2_name:
            bom.append({"category": "GPU (Secondary)", "item": gpu2_name})
        if mem_name:
            bom.append({"category": "Memory", "item": mem_name})
        if storage_name:
            bom.append({"category": "Storage", "item": storage_name})
        bom.append({"category": "PSU (Recommended)", "item": f"{power_info['recommended_psu_watts']}W unit"})
        return bom

class CoolingAdvisor:
    @staticmethod
    def advise(cpu_thermal, gpu_thermal):
        notes = []
        if cpu_thermal["thermal_headroom_ratio"] < 0.15:
            notes.append("CPU cooling: Headroom is low; consider higher-end air or 240mm+ AIO.")
        elif cpu_thermal["thermal_headroom_ratio"] < 0.3:
            notes.append("CPU cooling: Adequate, but not ideal for heavy sustained loads.")
        else:
            notes.append("CPU cooling: Good headroom for sustained workloads.")

        hot_gpus = [g for g in gpu_thermal if g.get("sensor_temp_c") and g["sensor_temp_c"] > g["thermal_limit_c"] - 5]
        if hot_gpus:
            notes.append("GPU cooling: One or more GPUs near thermal limit; consider more case airflow or blower-style cards.")
        elif gpu_thermal:
            notes.append("GPU cooling: Temperatures appear acceptable for current load.")

        return notes

# -----------------------------
# GUI: MAIN WINDOW
# -----------------------------

class HardwareLab(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Workstation / Xeon Hardware Lab v6.0")
        self.resize(1600, 980)

        self.db = ProfileDB()
        self.sim = Simulator(self.db)
        self.suggest_engine = SuggestedBuildEngine(self.db)
        self.ai_optimizer = AIBuildOptimizer(self.db)
        self.plugins = PluginManager(self.db)
        self.nodes = NodeManager()
        self.predictor = WorkloadPredictor()

        self.fingerprint = HardwareDetector.fingerprint()

        self.ui_ready = False

        self._build_ui()

        self.ui_ready = True

        self._log_event("Lab started")

        self.sensor_timer = QtCore.QTimer(self)
        self.sensor_timer.timeout.connect(self._update_sensor_panel)
        self.sensor_timer.start(2000)

    def showEvent(self, event):
        super().showEvent(event)
        if not self.ui_ready:
            return

        self.refresh_profile_view()
        self.refresh_diff_tab()
        self.refresh_matrix_tab()
        self.refresh_inventory_tab()
        self.refresh_node_view()
        self.refresh_timeline_view()
        self._refresh_fingerprint_view()
        self.refresh_topology_view()
        self._update_sensor_panel()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)

        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setFixedWidth(270)
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

    def _change_panel(self, index):
        self.stack.setCurrentIndex(index)
        self.anim.stop()
        self.effect.setOpacity(0.0)
        self.anim.start()

    # ---------------- DASHBOARD ----------------

    def _build_panel_dashboard(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        title = QtWidgets.QLabel("Live System Fingerprint + Sensors + Workload Guess")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        self.fp_view = QtWidgets.QTextEdit()
        self.fp_view.setReadOnly(True)
        layout.addWidget(self.fp_view)

        self.sensor_view = QtWidgets.QTextEdit()
        self.sensor_view.setReadOnly(True)
        self.sensor_view.setFixedHeight(160)
        layout.addWidget(self.sensor_view)

        self.stack.addWidget(w)

    def _refresh_fingerprint_view(self):
        self.fp_view.setText(json.dumps(self.fingerprint, indent=4))

    def _update_sensor_panel(self):
        sensors = HardwareDetector.read_all_sensors()
        cpu = sensors["cpu"]
        gpus = sensors["gpu"]

        lines = []
        lines.append("Real-time Sensors (best-effort):")
        lines.append(f"  CPU Load: {cpu.get('cpu_load_percent')} %")
        lines.append(f"  CPU Temp: {cpu.get('cpu_temp_c')} °C")
        lines.append("")
        lines.append("GPU Sensors:")
        if not gpus:
            lines.append("  (No GPU sensor data)")
        else:
            for g in gpus:
                lines.append(
                    f"  GPU{g['index']} {g['name']}: Temp={g['temp_c']} °C, "
                    f"Power={g['power_w']} W, Util={g['util_percent']} %"
                )

        # Try to guess workload from current sensors and a default config
        # (just use first CPU + board + GPU in DB for bandwidth)
        cpu_profile = next(iter(self.db.cpu_db.values()))
        board_profile = next(iter(self.db.board_db.values()))
        mem_bw = PlatformModel.memory_bandwidth(cpu_profile, board_profile)
        gpu_profiles = []
        if self.db.gpu_db:
            g = next(iter(self.db.gpu_db.values()))
            gpu_profiles.append(g)
        guessed = self.predictor.predict(cpu_profile, gpu_profiles, mem_bw, sensors)
        lines.append("")
        lines.append(f"ML-style workload guess: {guessed}")

        self.sensor_view.setText("\n".join(lines))

    # ---------------- OEM SIMULATOR ----------------

    def _build_panel_oem_sim(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)

        left = QtWidgets.QVBoxLayout()

        self.cpu_select = QtWidgets.QComboBox()
        self.cpu_select.addItems(self.db.get_cpu_names())

        self.board_select = QtWidgets.QComboBox()
        self.board_select.addItems(self.db.get_board_names())

        self.gpu1_select = QtWidgets.QComboBox()
        self.gpu1_select.addItem("<none>")
        self.gpu1_select.addItems(self.db.get_gpu_names())

        self.gpu2_select = QtWidgets.QComboBox()
        self.gpu2_select.addItem("<none>")
        self.gpu2_select.addItems(self.db.get_gpu_names())

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
            "ai",
            "balanced",
        ])

        self.run_btn = QtWidgets.QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)

        self.suggest_btn = QtWidgets.QPushButton("Suggest Build (v4-style)")
        self.suggest_btn.clicked.connect(self.suggest_build)

        self.ai_opt_btn = QtWidgets.QPushButton("AI Optimize Build (v6)")
        self.ai_opt_btn.clicked.connect(self.ai_optimize_build)

        self.bom_btn = QtWidgets.QPushButton("Generate BOM + Cooling Advice")
        self.bom_btn.clicked.connect(self.generate_bom_and_cooling)

        self.advice_box = QtWidgets.QTextEdit()
        self.advice_box.setReadOnly(True)

        left.addWidget(QtWidgets.QLabel("CPU Model"))
        left.addWidget(self.cpu_select)
        left.addWidget(QtWidgets.QLabel("Motherboard"))
        left.addWidget(self.board_select)
        left.addWidget(QtWidgets.QLabel("GPU Primary"))
        left.addWidget(self.gpu1_select)
        left.addWidget(QtWidgets.QLabel("GPU Secondary (optional / NVLink)"))
        left.addWidget(self.gpu2_select)
        left.addWidget(QtWidgets.QLabel("Memory Profile"))
        left.addWidget(self.mem_select)
        left.addWidget(QtWidgets.QLabel("Storage Profile"))
        left.addWidget(self.storage_select)
        left.addWidget(QtWidgets.QLabel("Workload"))
        left.addWidget(self.workload_select)
        left.addWidget(self.run_btn)
        left.addWidget(self.suggest_btn)
        left.addWidget(self.ai_opt_btn)
        left.addWidget(self.bom_btn)
        left.addWidget(QtWidgets.QLabel("Upgrade Advisor / AI Notes / Cooling"))
        left.addWidget(self.advice_box)

        right = QtWidgets.QVBoxLayout()
        self.report_box = QtWidgets.QTextEdit()
        self.report_box.setReadOnly(True)

        self.plugin_output = QtWidgets.QTextEdit()
        self.plugin_output.setReadOnly(True)
        self.plugin_output.setPlaceholderText("Plugin messages (if any)")

        right.addWidget(QtWidgets.QLabel("OEM Compatibility + Thermal + GPU + PCIe Report"))
        right.addWidget(self.report_box)
        right.addWidget(QtWidgets.QLabel("Plugin Hooks"))
        right.addWidget(self.plugin_output)

        layout.addLayout(left, 1)
        layout.addLayout(right, 2)

        self.stack.addWidget(w)

    def run_simulation(self):
        cpu_name = self.cpu_select.currentText()
        board_name = self.board_select.currentText()
        gpu1_name = self.gpu1_select.currentText()
        if gpu1_name == "<none>":
            gpu1_name = None
        gpu2_name = self.gpu2_select.currentText()
        if gpu2_name == "<none>":
            gpu2_name = None
        mem_name = self.mem_select.currentText()
        if mem_name == "<none>":
            mem_name = None
        storage_name = self.storage_select.currentText()
        if storage_name == "<none>":
            storage_name = None

        workload = self.workload_select.currentText()
        sensors = HardwareDetector.read_all_sensors()
        report = self.sim.run(cpu_name, board_name, gpu1_name, gpu2_name, mem_name, storage_name, workload, sensors)
        self.report_box.setText(json.dumps(report, indent=4))

        advice = recommend(self.db.cpu_db[cpu_name], workload)
        extra = []
        if report["thermal"].get("temp_note"):
            extra.append(report["thermal"]["temp_note"])
        if report["thermal"].get("vrm_note"):
            extra.append(report["thermal"]["vrm_note"])
        for gt in report["gpu_thermal"]:
            if gt.get("note"):
                extra.append(gt["note"])
        if report["gpu_compute"].get("note"):
            extra.append("GPU: " + report["gpu_compute"]["note"])
        extra.append(f"CPU boost estimate: {report['boost']['cpu_boost_ghz_est']} GHz")
        extra.append(f"GPU boost estimates: {report['boost']['gpu_boost_mhz_est']}")
        self.advice_box.setText(advice + "\n\n" + "\n".join(extra))

        INVENTORY.append({
            "time": str(datetime.now()),
            "cpu": cpu_name,
            "board": board_name,
            "gpu_primary": gpu1_name,
            "gpu_secondary": gpu2_name,
            "memory": mem_name,
            "storage": storage_name,
            "workload": workload,
            "score": report["score"],
            "status": report["status"],
        })

        plugin_msgs = self.plugins.run_post_simulation_hooks(report)
        self.plugin_output.setText("\n".join(plugin_msgs) if plugin_msgs else "No plugin messages.")

        self._log_event(
            f"Simulation: CPU={cpu_name}, Board={board_name}, GPU1={gpu1_name}, GPU2={gpu2_name}, "
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
        gpu1_name = suggestion["gpu_primary"]
        gpu2_name = suggestion["gpu_secondary"]
        mem_name = suggestion["memory"]
        storage_name = suggestion["storage"]

        self.cpu_select.setCurrentText(cpu_name)
        self.board_select.setCurrentText(board_name)
        self.gpu1_select.setCurrentText(gpu1_name if gpu1_name else "<none>")
        self.gpu2_select.setCurrentText(gpu2_name if gpu2_name else "<none>")
        self.mem_select.setCurrentText(mem_name)
        self.storage_select.setCurrentText(storage_name)

        sensors = HardwareDetector.read_all_sensors()
        report = self.sim.run(cpu_name, board_name, gpu1_name, gpu2_name, mem_name, storage_name, workload, sensors)
        self.report_box.setText(json.dumps(report, indent=4))

        advice = recommend(self.db.cpu_db[cpu_name], workload)
        self.advice_box.setText("Suggested build (v4-style):\n" + advice)

        self._log_event(
            f"Suggested build: CPU={cpu_name}, Board={board_name}, GPU1={gpu1_name}, GPU2={gpu2_name}, "
            f"MEM={mem_name}, Storage={storage_name}, Workload={workload}, Score={suggestion['score']}"
        )

    def ai_optimize_build(self):
        workload = self.workload_select.currentText()
        suggestion = self.ai_optimizer.optimize(workload)
        if not suggestion:
            QtWidgets.QMessageBox.warning(self, "AI Optimizer", "No suitable build found.")
            return

        cpu_name = suggestion["cpu"]
        board_name = suggestion["board"]
        gpu1_name = suggestion["gpu_primary"]
        gpu2_name = suggestion["gpu_secondary"]
        mem_name = suggestion["memory"]
        storage_name = suggestion["storage"]

        self.cpu_select.setCurrentText(cpu_name)
        self.board_select.setCurrentText(board_name)
        self.gpu1_select.setCurrentText(gpu1_name if gpu1_name else "<none>")
        self.gpu2_select.setCurrentText(gpu2_name if gpu2_name else "<none>")
        self.mem_select.setCurrentText(mem_name)
        self.storage_select.setCurrentText(storage_name)

        sensors = HardwareDetector.read_all_sensors()
        report = self.sim.run(cpu_name, board_name, gpu1_name, gpu2_name, mem_name, storage_name, workload, sensors)
        self.report_box.setText(json.dumps(report, indent=4))

        advice = recommend(self.db.cpu_db[cpu_name], workload)
        self.advice_box.setText(
            f"AI-Optimized build (v6):\nScore={suggestion['score']}\n\n" + advice
        )

        self._log_event(
            f"AI-Optimized build: CPU={cpu_name}, Board={board_name}, GPU1={gpu1_name}, GPU2={gpu2_name}, "
            f"MEM={mem_name}, Storage={storage_name}, Workload={workload}, Score={suggestion['score']}"
        )

    def generate_bom_and_cooling(self):
        cpu_name = self.cpu_select.currentText()
        board_name = self.board_select.currentText()
        gpu1_name = self.gpu1_select.currentText()
        if gpu1_name == "<none>":
            gpu1_name = None
        gpu2_name = self.gpu2_select.currentText()
        if gpu2_name == "<none>":
            gpu2_name = None
        mem_name = self.mem_select.currentText()
        if mem_name == "<none>":
            mem_name = None
        storage_name = self.storage_select.currentText()
        if storage_name == "<none>":
            storage_name = None

        workload = self.workload_select.currentText()
        sensors = HardwareDetector.read_all_sensors()
        report = self.sim.run(cpu_name, board_name, gpu1_name, gpu2_name, mem_name, storage_name, workload, sensors)

        power_info = report["power"]["power_model"]
        bom = BOMGenerator.generate(cpu_name, board_name, gpu1_name, gpu2_name, mem_name, storage_name, power_info)
        cooling_notes = CoolingAdvisor.advise(report["thermal"], report["gpu_thermal"])

        lines = ["Bill of Materials:"]
        for item in bom:
            lines.append(f"  - {item['category']}: {item['item']}")
        lines.append("")
        lines.append("Cooling Advisor:")
        for n in cooling_notes:
            lines.append(f"  - {n}")

        self.advice_box.setText("\n".join(lines))

    # ---------------- TOPOLOGY PANEL ----------------

    def _build_panel_topology(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("PCIe Topology, VRM, Storage & BIOS Modeling + Lane Allocator"))

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
        lines.append(f"  Cooling Capacity: {board.get('cooling_capacity_watts')} W")
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
        lines.append("")
        lines.append("PCIe Lane Allocator (example with 1x high-end GPU + 2x NVMe):")
        example_gpu = deepcopy(next(iter(self.db.gpu_db.values())))
        example_gpu["model_name"] = next(iter(self.db.gpu_db.keys()))
        example_storage = {"nvme_count": 2}
        alloc = PlatformModel.pcie_lane_allocation(board, [example_gpu], example_storage)
        for a in alloc["allocations"]:
            lines.append(
                f"  - {a['device']} ({a['model']}): {a['lanes']} lanes on {a['slot']}"
            )
        lines.append(f"  Remaining lanes: {alloc['remaining_lanes']}")
        self.topology_view.setText("\n".join(lines))

    # ---------------- PROFILES PANEL ----------------

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
        self.gpu1_select.clear()
        self.gpu1_select.addItem("<none>")
        self.gpu1_select.addItems(self.db.get_gpu_names())
        self.gpu2_select.clear()
        self.gpu2_select.addItem("<none>")
        self.gpu2_select.addItems(self.db.get_gpu_names())
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
        QtWidgets.QMessageBox.information(self, "New CPU", "Stub: extend to add custom CPU profiles.")

    def new_board_profile(self):
        QtWidgets.QMessageBox.information(self, "New Board", "Stub: extend to add custom board profiles.")

    # ---------------- MATRIX PANEL ----------------

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

    # ---------------- DIFF PANEL ----------------

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

    def _render_diff(self, kind, a, b, diff):
        lines = [f"{kind} Diff: {a}  vs  {b}", "-" * 50]
        if not diff:
            lines.append("No differences.")
        else:
            for k, (va, vb) in diff.items():
                lines.append(f"{k}: {va!r}  ->  {vb!r}")
        self.diff_output.setText("\n".join(lines))

    # ---------------- INVENTORY PANEL ----------------

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
                f"GPU1={item['gpu_primary']}  GPU2={item['gpu_secondary']}  "
                f"MEM={item['memory']}  Storage={item['storage']}  "
                f"Workload={item['workload']}  Score={item['score']}  Status={item['status']}"
            )
        self.inventory_view.setText("\n".join(lines) if lines else "No simulations recorded yet.")

    # ---------------- NODE SYNC PANEL ----------------

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

    # ---------------- TIMELINE PANEL ----------------

    def _build_panel_timeline(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(QtWidgets.QLabel("Event Timeline"))

        self.timeline_view = QtWidgets.QTextEdit()
        self.timeline_view.setReadOnly(True)
        layout.addWidget(self.timeline_view)

        self.stack.addWidget(w)

    def _log_event(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        TIMELINE.append(f"[{ts}] {msg}")
        self.refresh_timeline_view()

    def refresh_timeline_view(self):
        self.timeline_view.setText("\n".join(TIMELINE))

    # ---------------- METHODS & RISK PANEL ----------------

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
        content.append("     PCIe topology, NUMA, memory bandwidth, DMI/PCIe bottlenecks, PSU sizing, thermal modeling, GPU compute scoring,")
        content.append("     multi-GPU, NVLink, fan curves, boost prediction, and ML-style workload guessing.")
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
