#!/usr/bin/env python3
"""
POV Autonomous Privacy & Security Core

Autonomous + Privacy design:
- Starts monitoring automatically on launch (no user interaction required).
- No hostnames, usernames, IP addresses, MAC addresses, or interface details are read.
- No process names, command lines, or PIDs are read.
- No passwords, keystrokes, biometrics, or files are accessed.
- No data is written to disk (no logs, no caches).
- No network connections are opened or used.
- Only anonymous, aggregate system metrics are sampled:
  - CPU utilization (percent)
  - RAM utilization (percent)
  - Disk IO activity (bytes read+written)
  - Network IO rate (bytes/sec) as a scalar only
- Internal watchdog: if repeated sampling failures occur, the core shuts down cleanly.
"""

import sys
import subprocess
import importlib
import math
import threading
import time

# =========================
# Auto-loader for libraries
# =========================

REQUIRED_PACKAGES = ["psutil"]

def ensure_package(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[AUTOLOADER] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

psutil = ensure_package("psutil")

# =========================
# GUI + POV visualizer
# =========================

import tkinter as tk

PRIVACY_HARDENED = True  # explicit flag


class POVAutonomousPrivacyCore:
    def __init__(self, root):
        self.root = root
        self.root.title("POV Autonomous Privacy & Security Core")

        # --- Layout ---
        self.main_frame = tk.Frame(root, bg="#05060A")
        self.main_frame.pack(fill="both", expand=True)

        # Top privacy banner
        self.banner = tk.Label(
            self.main_frame,
            text="AUTONOMOUS • PRIVACY HARDENED • LOCAL ONLY • NO IDENTIFIERS COLLECTED",
            fg="#A0FFCF",
            bg="#07120F",
            font=("Consolas", 10, "bold")
        )
        self.banner.pack(fill="x", side="top")

        self.canvas = tk.Canvas(self.main_frame, bg="#05060A", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, side="top")

        control_frame = tk.Frame(self.main_frame, bg="#111318")
        control_frame.pack(fill="x", side="bottom")

        # --- Status labels ---
        self.status_label = tk.Label(control_frame, text="Status: Initializing…", fg="#A0FFB0", bg="#111318")
        self.status_label.pack(side="left", padx=10, pady=5)

        self.cpu_label = tk.Label(control_frame, text="CPU: --%", fg="#FFFFFF", bg="#111318")
        self.cpu_label.pack(side="left", padx=10)

        self.ram_label = tk.Label(control_frame, text="RAM: --%", fg="#FFFFFF", bg="#111318")
        self.ram_label.pack(side="left", padx=10)

        self.disk_label = tk.Label(control_frame, text="Disk: --%", fg="#FFFFFF", bg="#111318")
        self.disk_label.pack(side="left", padx=10)

        self.net_label = tk.Label(control_frame, text="Net: -- kB/s", fg="#FFFFFF", bg="#111318")
        self.net_label.pack(side="left", padx=10)

        self.sec_label = tk.Label(control_frame, text="Security Load: --", fg="#FFDD88", bg="#111318")
        self.sec_label.pack(side="left", padx=10)

        # No manual controls: autonomous
        self.info_label = tk.Label(
            control_frame,
            text="Autonomous mode: running continuously",
            fg="#8888FF",
            bg="#111318"
        )
        self.info_label.pack(side="right", padx=10)

        # --- POV buffers ---
        self.num_segments = 240
        self.cpu_buffer = [0.0] * self.num_segments
        self.ram_buffer = [0.0] * self.num_segments
        self.disk_buffer = [0.0] * self.num_segments
        self.net_buffer = [0.0] * self.num_segments
        self.sec_buffer = [0.0] * self.num_segments

        self.current_index = 0

        # --- Visualization parameters ---
        self.width = 900
        self.height = 700
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.inner_radius_cpu = 120
        self.outer_radius_cpu = 180

        self.inner_radius_ram = 190
        self.outer_radius_ram = 240

        self.inner_radius_disk = 250
        self.outer_radius_disk = 300

        self.inner_radius_net = 310
        self.outer_radius_net = 360

        self.running = True
        self.update_interval = 0.1  # seconds

        # For network speed calculation
        self.last_net = psutil.net_io_counters()
        self.last_net_time = time.time()

        # For security stress heuristic
        self.last_cpu = 0.0
        self.last_ram = 0.0
        self.last_disk = 0.0
        self.last_net_rate = 0.0

        # Watchdog
        self.error_count = 0
        self.max_errors = 10

        # Resize handling
        self.root.bind("<Configure>", self.on_resize)

        # Start autonomous loops
        self.status_label.config(text="Status: Running (Autonomous, Privacy Hardened)")
        threading.Thread(target=self.data_loop, daemon=True).start()
        self.schedule_redraw()

    # =========================
    # Control / Resize
    # =========================

    def on_resize(self, event):
        if event.width < 200 or event.height < 200:
            return
        self.width = event.width
        self.height = event.height - 60
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        base = min(self.width, self.height)
        scale = base / 800.0
        self.inner_radius_cpu = int(120 * scale)
        self.outer_radius_cpu = int(180 * scale)
        self.inner_radius_ram = int(190 * scale)
        self.outer_radius_ram = int(240 * scale)
        self.inner_radius_disk = int(250 * scale)
        self.outer_radius_disk = int(300 * scale)
        self.inner_radius_net = int(310 * scale)
        self.outer_radius_net = int(360 * scale)

        self.redraw()

    # =========================
    # Data sampling (anonymous)
    # =========================

    def data_loop(self):
        while self.running:
            try:
                # CPU and RAM
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent

                # Disk IO (aggregate only)
                disk_io = psutil.disk_io_counters()
                disk_activity = (disk_io.read_bytes + disk_io.write_bytes)

                # Network IO rate (aggregate only)
                now = time.time()
                net = psutil.net_io_counters()
                dt = max(now - self.last_net_time, 1e-6)
                bytes_sent = net.bytes_sent - self.last_net.bytes_sent
                bytes_recv = net.bytes_recv - self.last_net.bytes_recv
                net_rate = (bytes_sent + bytes_recv) / dt
                net_kb = net_rate / 1024.0

                self.last_net = net
                self.last_net_time = now

                # Normalize
                cpu_norm = min(max(cpu / 100.0, 0.0), 1.0)
                ram_norm = min(max(ram / 100.0, 0.0), 1.0)
                disk_norm = min(disk_activity / (100 * 1024 * 1024), 1.0)
                net_norm = min(net_rate / (10 * 1024 * 1024), 1.0)

                # Security stress heuristic
                stress = 0.0
                stress += abs(cpu_norm - self.last_cpu) * 1.5
                stress += abs(ram_norm - self.last_ram) * 1.0
                stress += abs(disk_norm - self.last_disk) * 1.2
                stress += abs(net_norm - self.last_net_rate) * 1.3
                stress += (cpu_norm + ram_norm + disk_norm + net_norm) / 4.0
                stress = min(max(stress, 0.0), 2.0) / 2.0

                self.last_cpu = cpu_norm
                self.last_ram = ram_norm
                self.last_disk = disk_norm
                self.last_net_rate = net_norm

                idx = self.current_index
                self.cpu_buffer[idx] = cpu_norm
                self.ram_buffer[idx] = ram_norm
                self.disk_buffer[idx] = disk_norm
                self.net_buffer[idx] = net_norm
                self.sec_buffer[idx] = stress

                self.current_index = (self.current_index + 1) % self.num_segments

                self.root.after(0, self.update_labels, cpu, ram, disk_norm, net_kb, stress)

                self.error_count = 0  # reset on success

            except Exception:
                self.error_count += 1
                if self.error_count >= self.max_errors:
                    self.running = False
                    self.root.after(0, self.status_label.config,
                                    {"text": "Status: Watchdog shutdown (sampling failures)", "fg": "#FF6B6B"})
                    break

            time.sleep(self.update_interval)

    def update_labels(self, cpu, ram, disk_norm, net_kb, stress):
        self.cpu_label.config(text=f"CPU: {cpu:.1f}%")
        self.ram_label.config(text=f"RAM: {ram:.1f}%")
        self.disk_label.config(text=f"Disk: {disk_norm*100:.1f}% est")
        self.net_label.config(text=f"Net: {net_kb:.1f} kB/s")
        self.sec_label.config(text=f"Security Load: {stress*100:.0f}%")

        if stress < 0.3:
            color = "#A0FFB0"
        elif stress < 0.7:
            color = "#FFD966"
        else:
            color = "#FF6B6B"
        self.sec_label.config(fg=color)

    # =========================
    # Visualization
    # =========================

    def schedule_redraw(self):
        if not self.running:
            return
        self.redraw()
        self.root.after(int(self.update_interval * 1000), self.schedule_redraw)

    def redraw(self):
        self.canvas.delete("all")

        self.draw_background()
        self.draw_ring(self.cpu_buffer, self.inner_radius_cpu, self.outer_radius_cpu, mode="cpu")
        self.draw_ring(self.ram_buffer, self.inner_radius_ram, self.outer_radius_ram, mode="ram")
        self.draw_ring(self.disk_buffer, self.inner_radius_disk, self.outer_radius_disk, mode="disk")
        self.draw_ring(self.net_buffer, self.inner_radius_net, self.outer_radius_net, mode="net")
        self.draw_security_core()

    def draw_background(self):
        max_r = self.outer_radius_net + 50
        self.canvas.create_oval(
            self.center_x - max_r,
            self.center_y - max_r,
            self.center_x + max_r,
            self.center_y + max_r,
            outline="#101320",
            width=2
        )

        self.canvas.create_line(
            self.center_x - 20, self.center_y,
            self.center_x + 20, self.center_y,
            fill="#151822", width=1
        )
        self.canvas.create_line(
            self.center_x, self.center_y - 20,
            self.center_x, self.center_y + 20,
            fill="#151822", width=1
        )

    def draw_ring(self, buffer, inner_r, outer_r, mode="cpu"):
        for i, intensity in enumerate(buffer):
            if intensity <= 0.01:
                continue
            idx_offset = (i - self.current_index) % self.num_segments
            angle = (2 * math.pi * idx_offset) / self.num_segments - math.pi / 2
            self.draw_segment(angle, intensity, inner_r, outer_r, mode)

    def draw_segment(self, angle, intensity, inner_r, outer_r, mode):
        x1 = self.center_x + inner_r * math.cos(angle)
        y1 = self.center_y + inner_r * math.sin(angle)
        x2 = self.center_x + outer_r * math.cos(angle)
        y2 = self.center_y + outer_r * math.sin(angle)

        if mode == "cpu":
            base_r, base_g, base_b = 80, 220, 255
        elif mode == "ram":
            base_r, base_g, base_b = 120, 255, 160
        elif mode == "disk":
            base_r, base_g, base_b = 255, 200, 120
        elif mode == "net":
            base_r, base_g, base_b = 255, 120, 180
        else:
            base_r, base_g, base_b = 200, 200, 200

        scale = intensity
        r = int(base_r * scale + 20)
        g = int(base_g * scale + 20)
        b = int(base_b * scale + 20)

        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        color = f"#{r:02x}{g:02x}{b:02x}"
        width = 1 + 4 * intensity

        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, capstyle="round")

    def draw_security_core(self):
        avg_stress = sum(self.sec_buffer) / len(self.sec_buffer)
        radius = 50 + int(30 * avg_stress)

        if avg_stress < 0.3:
            fill = "#07120F"
            outline = "#1EE3A1"
            text_color = "#A0FFCF"
        elif avg_stress < 0.7:
            fill = "#1A1407"
            outline = "#FFC857"
            text_color = "#FFE9A3"
        else:
            fill = "#190707"
            outline = "#FF4B4B"
            text_color = "#FFB3B3"

        self.canvas.create_oval(
            self.center_x - radius,
            self.center_y - radius,
            self.center_x + radius,
            self.center_y + radius,
            fill=fill,
            outline=outline,
            width=3
        )

        self.canvas.create_text(
            self.center_x,
            self.center_y,
            text="POV\nAUTONOMOUS\nCORE",
            fill=text_color,
            font=("Consolas", 11, "bold")
        )


# =========================
# Entry point
# =========================

def main():
    root = tk.Tk()
    app = POVAutonomousPrivacyCore(root)
    root.geometry("900x700")
    root.mainloop()

if __name__ == "__main__":
    main()
