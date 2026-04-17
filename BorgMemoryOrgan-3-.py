"""
BORG-OS MEMORY BACKBONE ORGAN (v2.0 – Distributed, Privilege-Aware, Telemetry-Fed)

Includes:
- Flow-based addressing, intelligent routing, Alzheimer-like degradation
- Tkinter GUI visualizer with overlays
- Borg-OS control panel (live tuning)
- JSON IPC protocol over stdin/stdout and TCP daemon
- Swarm integration hooks (local + remote)
- Privilege-aware behavior
- System telemetry organ (CPU/RAM) feeding real data
"""

# =========================
# Autoloader
# =========================

def autoload(modules):
    import importlib, subprocess, sys
    loaded = {}
    for m in modules:
        try:
            loaded[m] = importlib.import_module(m)
        except ImportError:
            # tkinter is stdlib; if missing, we just skip GUI
            if m in ("tkinter",):
                loaded[m] = None
                continue
            subprocess.check_call([sys.executable, "-m", "pip", "install", m])
            loaded[m] = importlib.import_module(m)
    return loaded


mods = autoload([
    "random",
    "dataclasses",
    "typing",
    "tkinter",
    "json",
    "sys",
    "socket",
    "threading",
    "time",
    "psutil",
    "os",
    "getpass",
    "platform",
])
random = mods["random"]
dataclasses = mods["dataclasses"]
typing = mods["typing"]
tkinter = mods["tkinter"]  # may be None
json = mods["json"]
sys = mods["sys"]
socket = mods["socket"]
threading = mods["threading"]
time = mods["time"]
psutil = mods["psutil"]
os = mods["os"]
getpass = mods["getpass"]
platform = mods["platform"]

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

# =========================
# Core Data Structures
# =========================

@dataclass
class Packet:
    pid: int
    value: int
    energy: float = 1.0
    priority: float = 1.0
    meta: dict = field(default_factory=dict)


class MemoryTile:
    """
    A tile is a small local memory region with cells that hold packets.
    """

    def __init__(self, size: int, tile_id: int):
        self.size = size
        self.tile_id = tile_id
        self.cells: List[Optional[Packet]] = [None] * size

    def inject(self, index: int, packet: Packet):
        idx = index % self.size
        self.cells[idx] = packet

    def step(self, policy) -> List[Tuple[int, Packet]]:
        """
        Advance packets one step according to policy.
        Returns list of (dest_tile_id, packet) for packets leaving this tile.
        """
        new_cells = [None] * self.size
        outgoing: List[Tuple[int, Packet]] = []

        for i, pkt in enumerate(self.cells):
            if pkt is None:
                continue

            pkt = policy.pre_transform(self.tile_id, i, pkt)
            if pkt is None:
                continue  # forgotten

            direction, dest_tile = policy.choose_route(self.tile_id, i, pkt)
            target_index = i + direction

            if target_index < 0 or target_index >= self.size or dest_tile != self.tile_id:
                pkt = policy.transform(pkt)
                if pkt is not None:
                    outgoing.append((dest_tile, pkt))
            else:
                pkt = policy.transform(pkt)
                if pkt is not None:
                    new_cells[target_index] = pkt

        self.cells = new_cells
        return outgoing

    def snapshot(self) -> List[Optional[int]]:
        return [p.value if p else None for p in self.cells]

    def snapshot_energy(self) -> List[float]:
        return [p.energy if p else 0.0 for p in self.cells]


# =========================
# Policies: Base, Intelligence, Alzheimer
# =========================

class BasePolicy:
    """
    Base flow policy: loss/double, simple random direction.
    """

    def __init__(self, p_loss=0.1, p_double=0.1):
        self.p_loss = p_loss
        self.p_double = p_double

    def pre_transform(self, tile_id: int, index: int, packet: Packet) -> Optional[Packet]:
        return packet

    def choose_route(self, tile_id: int, index: int, packet: Packet) -> Tuple[int, int]:
        direction = random.choice([-1, 1])
        return direction, tile_id

    def transform(self, packet: Packet) -> Optional[Packet]:
        r = random.random()
        if r < self.p_loss:
            packet.value = int(packet.value * 0.5)
            packet.energy *= 0.7
        elif r < self.p_loss + self.p_double:
            packet.value *= 2
            packet.energy *= 1.2
        return packet


class IntelligenceLayer(BasePolicy):
    """
    Adds intelligent routing:
    - Higher priority/energy packets are routed more 'directly'
    - Low energy packets drift more randomly
    """

    def __init__(self, base: BasePolicy, num_tiles: int):
        super().__init__(base.p_loss, base.p_double)
        self.base = base
        self.num_tiles = num_tiles

    def choose_route(self, tile_id: int, index: int, packet: Packet) -> Tuple[int, int]:
        base_dir, _ = self.base.choose_route(tile_id, index, packet)

        if packet.priority > 1.0:
            direction = 1
        elif packet.priority < 1.0:
            direction = -1
        else:
            direction = base_dir

        if packet.energy > 1.5 and random.random() < 0.3:
            dest_tile = (tile_id + 1) % self.num_tiles
        elif packet.energy < 0.5 and random.random() < 0.3:
            dest_tile = (tile_id - 1) % self.num_tiles
        else:
            dest_tile = tile_id

        return direction, dest_tile

    def transform(self, packet: Packet) -> Optional[Packet]:
        pkt = self.base.transform(packet)
        if pkt is None:
            return None
        pkt.energy = max(0.1, min(pkt.energy, 5.0))
        return pkt


class AlzheimerLayer(IntelligenceLayer):
    """
    Adds Alzheimer-like behavior:
    - Forgetting
    - Misrouting
    - Meta corruption
    - Ghosts
    - Progression over time
    """

    def __init__(
        self,
        base: BasePolicy,
        num_tiles: int,
        forget_prob=0.05,
        misroute_prob=0.1,
        meta_corrupt_prob=0.15,
        ghost_prob=0.05,
        progression_rate=0.001,
        max_ghosts=100,
    ):
        super().__init__(base, num_tiles)
        self.forget_prob = forget_prob
        self.misroute_prob = misroute_prob
        self.meta_corrupt_prob = meta_corrupt_prob
        self.ghost_prob = ghost_prob
        self.progression_rate = progression_rate
        self.max_ghosts = max_ghosts
        self.ghost_buffer: List[Packet] = []
        self.time = 0
        # For overlays
        self.last_ghost_event = False

    def tick_progression(self):
        self.time += 1
        factor = 1 + self.progression_rate * self.time
        self.forget_prob = min(0.9, self.forget_prob * factor)
        self.misroute_prob = min(0.9, self.misroute_prob * factor)
        self.meta_corrupt_prob = min(0.9, self.meta_corrupt_prob * factor)

    def pre_transform(self, tile_id: int, index: int, packet: Packet) -> Optional[Packet]:
        if random.random() < self.forget_prob:
            if random.random() < self.ghost_prob and len(self.ghost_buffer) < self.max_ghosts:
                self.ghost_buffer.append(packet)
            return None

        if random.random() < self.meta_corrupt_prob:
            packet.meta["tag"] = "corrupted"
            packet.meta["time"] = random.randint(-10_000, 10_000)

        return packet

    def choose_route(self, tile_id: int, index: int, packet: Packet) -> Tuple[int, int]:
        if random.random() < self.misroute_prob:
            direction = random.choice([-3, -2, -1, 1, 2, 3])
            dest_tile = random.randint(0, self.num_tiles - 1)
            return direction, dest_tile

        return super().choose_route(tile_id, index, packet)

    def transform(self, packet: Packet) -> Optional[Packet]:
        self.last_ghost_event = False
        pkt = super().transform(packet)
        if pkt is None:
            return None

        if self.ghost_buffer and random.random() < self.ghost_prob:
            ghost = random.choice(self.ghost_buffer)
            ghost.value = int(ghost.value * random.uniform(0.5, 1.5))
            ghost.energy *= random.uniform(0.5, 1.2)
            ghost.priority *= random.uniform(0.5, 1.5)
            ghost.energy = max(0.1, min(ghost.energy, 5.0))
            ghost.priority = max(0.1, min(ghost.priority, 5.0))
            self.last_ghost_event = True
            return ghost

        return pkt


# =========================
# Privilege Profile
# =========================

class PrivilegeProfile:
    def __init__(self):
        self.user = getpass.getuser()
        self.os = platform.system()
        self.hostname = platform.node()
        self.is_admin = self._detect_admin()

    def _detect_admin(self) -> bool:
        try:
            if self.os == "Windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except Exception:
            return False


# =========================
# Borg Memory Organ (Backbone + IPC)
# =========================

class BorgMemoryOrgan:
    """
    Central memory fabric organ:
    - multiple tiles
    - Alzheimer + intelligence policy
    - swarm hooks
    - live control knobs
    - JSON IPC handler
    """

    def __init__(
        self,
        organ_id: str,
        num_tiles=4,
        tile_size=8,
        base_policy=None,
        profile: Optional[PrivilegeProfile] = None,
    ):
        self.organ_id = organ_id
        self.num_tiles = num_tiles
        self.tiles = [MemoryTile(tile_size, t) for t in range(num_tiles)]
        base = base_policy or BasePolicy(p_loss=0.1, p_double=0.1)
        self.policy = AlzheimerLayer(base, num_tiles=num_tiles)
        self.next_pid = 1
        self.debug_log: List[str] = []
        self.profile = profile or PrivilegeProfile()
        self.last_telemetry_inject_time = 0.0

        self._apply_privilege_profile()

    # ---- Privilege-aware tuning ----

    def _apply_privilege_profile(self):
        # Example: admin → more stable, non-admin → slightly noisier
        if self.profile.is_admin:
            self.policy.forget_prob *= 0.5
            self.policy.misroute_prob *= 0.5
            self.policy.meta_corrupt_prob *= 0.5
        else:
            self.policy.forget_prob *= 1.2
            self.policy.misroute_prob *= 1.2

    # ---- Swarm Hooks ----

    def export_state(self) -> dict:
        return {
            "organ_id": self.organ_id,
            "tiles": [[(p.value, p.energy) if p else None for p in tile.cells] for tile in self.tiles],
            "time": self.policy.time,
        }

    def import_state(self, state: dict):
        tiles_state = state.get("tiles", [])
        for t_idx, tile_state in enumerate(tiles_state):
            if t_idx >= self.num_tiles:
                break
            tile = self.tiles[t_idx]
            for c_idx, cell in enumerate(tile_state):
                if c_idx >= tile.size:
                    break
                if cell is None:
                    tile.cells[c_idx] = None
                else:
                    value, energy = cell
                    tile.cells[c_idx] = Packet(
                        pid=self.next_pid,
                        value=value,
                        energy=energy,
                        priority=1.0,
                        meta={"source": state.get("organ_id", "import")},
                    )
                    self.next_pid += 1

    def receive_message(self, msg: dict):
        if msg.get("type") == "adjust_forget_prob":
            self.set_forget_prob(msg.get("value", self.policy.forget_prob))

    # ---- Live control knobs ----

    def set_forget_prob(self, value: float):
        self.policy.forget_prob = max(0.0, min(value, 1.0))

    def set_misroute_prob(self, value: float):
        self.policy.misroute_prob = max(0.0, min(value, 1.0))

    def set_meta_corrupt_prob(self, value: float):
        self.policy.meta_corrupt_prob = max(0.0, min(value, 1.0))

    def set_ghost_prob(self, value: float):
        self.policy.ghost_prob = max(0.0, min(value, 1.0))

    def set_loss_prob(self, value: float):
        self.policy.base.p_loss = max(0.0, min(value, 1.0))

    def set_double_prob(self, value: float):
        self.policy.base.p_double = max(0.0, min(value, 1.0))

    # ---- Core API ----

    def inject(self, tile_id: int, index: int, value: int, priority: float = 1.0, meta=None):
        pkt = Packet(
            pid=self.next_pid,
            value=value,
            energy=1.0,
            priority=priority,
            meta=meta or {},
        )
        self.next_pid += 1
        self.tiles[tile_id % self.num_tiles].inject(index, pkt)

        # Mark telemetry injections for overlays
        if pkt.meta.get("metric") in ("cpu", "mem"):
            self.last_telemetry_inject_time = time.time()

        return pkt.pid

    def tick(self):
        self.policy.tick_progression()

        outgoing_by_tile: Dict[int, List[Packet]] = {t: [] for t in range(self.num_tiles)}

        for t_id, tile in enumerate(self.tiles):
            outgoing = tile.step(self.policy)
            for dest_tile, pkt in outgoing:
                outgoing_by_tile[dest_tile % self.num_tiles].append(pkt)

        for dest_tile, packets in outgoing_by_tile.items():
            for pkt in packets:
                self.tiles[dest_tile].inject(0, pkt)

    def snapshot_values(self) -> List[List[Optional[int]]]:
        return [tile.snapshot() for tile in self.tiles]

    def snapshot_energy(self) -> List[List[float]]:
        return [tile.snapshot_energy() for tile in self.tiles]

    # ---- JSON IPC Protocol ----

    def handle_ipc_message(self, msg: dict) -> dict:
        topic = msg.get("topic")
        cmd = msg.get("cmd")
        args = msg.get("args", {}) or {}
        mid = msg.get("id")

        resp: Dict[str, Any] = {"id": mid, "ok": True, "topic": topic, "cmd": cmd, "result": {}}

        try:
            if topic == "config":
                v = float(args["value"])
                if cmd == "set_forget_prob":
                    self.set_forget_prob(v)
                elif cmd == "set_misroute_prob":
                    self.set_misroute_prob(v)
                elif cmd == "set_meta_corrupt_prob":
                    self.set_meta_corrupt_prob(v)
                elif cmd == "set_ghost_prob":
                    self.set_ghost_prob(v)
                elif cmd == "set_loss_prob":
                    self.set_loss_prob(v)
                elif cmd == "set_double_prob":
                    self.set_double_prob(v)
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_config_cmd"

            elif topic == "inject" and cmd == "packet":
                tile_id = int(args.get("tile_id", 0))
                index = int(args.get("index", 0))
                value = int(args.get("value", 0))
                priority = float(args.get("priority", 1.0))
                meta = args.get("meta", {})
                pid = self.inject(tile_id, index, value, priority, meta)
                resp["result"]["pid"] = pid

            elif topic == "snapshot":
                if cmd == "values":
                    resp["result"]["values"] = self.snapshot_values()
                elif cmd == "energies":
                    resp["result"]["energies"] = self.snapshot_energy()
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_snapshot_cmd"

            elif topic == "state" and cmd == "full":
                resp["result"]["state"] = self.export_state()

            elif topic == "tick" and cmd == "step":
                steps = int(args.get("steps", 1))
                for _ in range(steps):
                    self.tick()

            elif topic == "swarm":
                if cmd == "import_state":
                    self.import_state(args.get("state", {}))
                elif cmd == "adjust":
                    param = args.get("param")
                    value = args.get("value")
                    if param == "forget_prob":
                        self.set_forget_prob(float(value))
                    elif param == "misroute_prob":
                        self.set_misroute_prob(float(value))
                    elif param == "meta_corrupt_prob":
                        self.set_meta_corrupt_prob(float(value))
                    elif param == "ghost_prob":
                        self.set_ghost_prob(float(value))
                    else:
                        resp["ok"] = False
                        resp["error"] = "unknown_swarm_param"
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_swarm_cmd"

            elif topic == "debug":
                if cmd == "log":
                    resp["result"]["log"] = self.debug_log[-50:]
                elif cmd == "health":
                    resp["result"]["health"] = {
                        "time": self.policy.time,
                        "forget_prob": self.policy.forget_prob,
                        "misroute_prob": self.policy.misroute_prob,
                        "meta_corrupt_prob": self.policy.meta_corrupt_prob,
                        "ghost_prob": self.policy.ghost_prob,
                        "ghost_buffer_size": len(self.policy.ghost_buffer),
                    }
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_debug_cmd"

            elif topic == "control_panel":
                if cmd == "get_all":
                    resp["result"]["params"] = {
                        "forget_prob": self.policy.forget_prob,
                        "misroute_prob": self.policy.misroute_prob,
                        "meta_corrupt_prob": self.policy.meta_corrupt_prob,
                        "ghost_prob": self.policy.ghost_prob,
                        "loss_prob": self.policy.base.p_loss,
                        "double_prob": self.policy.base.p_double,
                    }
                elif cmd == "set":
                    param = args.get("param")
                    value = float(args.get("value"))
                    if param == "forget_prob":
                        self.set_forget_prob(value)
                    elif param == "misroute_prob":
                        self.set_misroute_prob(value)
                    elif param == "meta_corrupt_prob":
                        self.set_meta_corrupt_prob(value)
                    elif param == "ghost_prob":
                        self.set_ghost_prob(value)
                    elif param == "loss_prob":
                        self.set_loss_prob(value)
                    elif param == "double_prob":
                        self.set_double_prob(value)
                    else:
                        resp["ok"] = False
                        resp["error"] = "unknown_control_param"
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_control_cmd"

            elif topic == "meta":
                if cmd == "info":
                    resp["result"]["info"] = {
                        "organ_id": self.organ_id,
                        "num_tiles": self.num_tiles,
                        "tile_size": len(self.tiles[0].cells),
                        "time": self.policy.time,
                        "user": self.profile.user,
                        "is_admin": self.profile.is_admin,
                        "hostname": self.profile.hostname,
                        "os": self.profile.os,
                    }
                elif cmd == "ping":
                    resp["result"]["pong"] = True
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_meta_cmd"

            else:
                resp["ok"] = False
                resp["error"] = "unknown_topic"

        except Exception as e:
            resp["ok"] = False
            resp["error"] = str(e)

        return resp


# =========================
# GUI: Visualizer + Control Panel
# =========================

class MemoryVisualizer:
    """
    Simple GUI: each tile is a row, each cell is a rectangle.
    Color intensity = energy, text = value.
    Overlays:
    - Tile halo on ghost events
    - Telemetry pulse when recent system metrics injected
    """

    def __init__(self, organ: BorgMemoryOrgan, root=None, cell_size=40, tick_ms=300):
        if tkinter is None:
            raise RuntimeError("Tkinter not available; GUI cannot be created.")
        self.organ = organ
        self.cell_size = cell_size
        self.tick_ms = tick_ms

        self.root = root or tkinter.Tk()
        self.window = tkinter.Toplevel(self.root)
        title_suffix = "ADMIN" if self.organ.profile.is_admin else "USER"
        self.window.title(f"Borg Memory Organ - {organ.organ_id} [{title_suffix}]")

        rows = organ.num_tiles
        cols = len(organ.tiles[0].cells)
        self.canvas = tkinter.Canvas(
            self.window,
            width=cols * cell_size,
            height=rows * cell_size,
            bg="black",
        )
        self.canvas.pack()

        self.rects = []
        self.texts = []
        for r in range(rows):
            row_rects = []
            row_texts = []
            for c in range(cols):
                x0 = c * cell_size
                y0 = r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="black", outline="gray")
                txt = self.canvas.create_text(
                    x0 + cell_size / 2,
                    y0 + cell_size / 2,
                    text="",
                    fill="white",
                    font=("Consolas", 10),
                )
                row_rects.append(rect)
                row_texts.append(txt)
            self.rects.append(row_rects)
            self.texts.append(row_texts)

        # Overlays per tile (ghost halo)
        self.tile_overlays = []
        for r in range(rows):
            rect = self.canvas.create_rectangle(
                0,
                r * cell_size,
                cols * cell_size,
                (r + 1) * cell_size,
                outline="",
                fill="",
            )
            self.tile_overlays.append(rect)

        # Telemetry overlay (full canvas pulse)
        self.telemetry_overlay = self.canvas.create_rectangle(
            0,
            0,
            cols * cell_size,
            rows * cell_size,
            outline="",
            fill="",
        )

        self.window.after(self.tick_ms, self.update)

    def energy_to_color(self, e: float) -> str:
        e = max(0.0, min(e, 5.0))
        g = int(50 + (205 * (e / 5.0)))
        return f"#{0:02x}{g:02x}{0:02x}"

    def update(self):
        self.organ.tick()

        values = self.organ.snapshot_values()
        energies = self.organ.snapshot_energy()

        for r, row in enumerate(values):
            for c, val in enumerate(row):
                e = energies[r][c]
                color = self.energy_to_color(e) if val is not None else "#000000"
                self.canvas.itemconfig(self.rects[r][c], fill=color)
                self.canvas.itemconfig(self.texts[r][c], text=str(val) if val is not None else "")

        # Ghost halo overlay (if any ghost event occurred this tick)
        ghost_active = isinstance(self.organ.policy, AlzheimerLayer) and self.organ.policy.last_ghost_event
        for t_id in range(self.organ.num_tiles):
            if ghost_active:
                self.canvas.itemconfig(self.tile_overlays[t_id], fill="#00ff0040")
            else:
                self.canvas.itemconfig(self.tile_overlays[t_id], fill="")

        # Telemetry pulse overlay
        now = time.time()
        if now - self.organ.last_telemetry_inject_time < 1.0:
            self.canvas.itemconfig(self.telemetry_overlay, fill="#0000ff20")
        else:
            self.canvas.itemconfig(self.telemetry_overlay, fill="")

        self.window.after(self.tick_ms, self.update)


class BorgControlPanel:
    """
    Borg-OS style control panel for live tuning of the memory organ.
    """

    def __init__(self, organ: BorgMemoryOrgan, root=None):
        if tkinter is None:
            raise RuntimeError("Tkinter not available; control panel cannot be created.")
        self.organ = organ
        self.root = root or tkinter.Tk()
        self.window = tkinter.Toplevel(self.root)
        self.window.title(f"Control Panel - {organ.organ_id}")

        self._add_slider("Forget prob", 0, 1, organ.policy.forget_prob, self.organ.set_forget_prob)
        self._add_slider("Misroute prob", 0, 1, organ.policy.misroute_prob, self.organ.set_misroute_prob)
        self._add_slider("Meta corrupt prob", 0, 1, organ.policy.meta_corrupt_prob, self.organ.set_meta_corrupt_prob)
        self._add_slider("Ghost prob", 0, 1, organ.policy.ghost_prob, self.organ.set_ghost_prob)
        self._add_slider("Loss prob", 0, 1, organ.policy.base.p_loss, self.organ.set_loss_prob)
        self._add_slider("Double prob", 0, 1, organ.policy.base.p_double, self.organ.set_double_prob)

    def _add_slider(self, label, frm, to, initial, callback):
        frame = tkinter.Frame(self.window)
        frame.pack(fill="x", padx=5, pady=2)

        lab = tkinter.Label(frame, text=label, width=18, anchor="w")
        lab.pack(side="left")

        var = tkinter.DoubleVar(value=initial)

        def on_change(_):
            callback(var.get())

        slider = tkinter.Scale(
            frame,
            from_=frm,
            to=to,
            resolution=0.01,
            orient="horizontal",
            variable=var,
            command=on_change,
            length=200,
        )
        slider.pack(side="left")


# =========================
# Borg-OS Organ Client Template (Protocol User)
# =========================

class MemoryBackboneClient:
    """
    Template for other Borg-OS organs to talk to the backbone in-process.
    In a real system this would talk over pipes/sockets; here we call handle_ipc_message directly.
    """

    def __init__(self, organ: BorgMemoryOrgan):
        self.organ = organ

    def send(self, topic: str, cmd: str, args: dict = None, mid: str = None) -> dict:
        msg = {
            "id": mid,
            "topic": topic,
            "cmd": cmd,
            "args": args or {},
        }
        return self.organ.handle_ipc_message(msg)

    # Convenience wrappers
    def set_param(self, param: str, value: float):
        return self.send("control_panel", "set", {"param": param, "value": value})

    def inject_packet(self, tile_id: int, index: int, value: int, priority: float = 1.0, meta=None):
        return self.send("inject", "packet", {
            "tile_id": tile_id,
            "index": index,
            "value": value,
            "priority": priority,
            "meta": meta or {},
        })

    def snapshot_values(self):
        return self.send("snapshot", "values")

    def snapshot_energies(self):
        return self.send("snapshot", "energies")

    def tick(self, steps: int = 1):
        return self.send("tick", "step", {"steps": steps})


# =========================
# Remote Organ Client (TCP)
# =========================

class RemoteOrganClient:
    """
    Client for talking to a remote BorgMemoryOrgan daemon over TCP.
    """

    def __init__(self, host="127.0.0.1", port=7777):
        self.host = host
        self.port = port

    def _send_raw(self, msg: dict) -> dict:
        line = json.dumps(msg) + "\n"
        with socket.create_connection((self.host, self.port), timeout=5) as s:
            s.sendall(line.encode("utf-8"))
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
        if not data:
            return {"ok": False, "error": "no_response"}
        return json.loads(data.decode("utf-8"))

    def send(self, topic: str, cmd: str, args: dict = None, mid: str = None) -> dict:
        msg = {
            "id": mid,
            "topic": topic,
            "cmd": cmd,
            "args": args or {},
        }
        return self._send_raw(msg)

    def export_state(self) -> dict:
        resp = self.send("state", "full", {})
        return resp.get("result", {}).get("state", {})

    def import_state(self, state: dict) -> dict:
        return self.send("swarm", "import_state", {"state": state})

    def set_param(self, param: str, value: float) -> dict:
        return self.send("control_panel", "set", {"param": param, "value": value})

    def inject_packet(self, tile_id: int, index: int, value: int, priority: float = 1.0, meta=None) -> dict:
        return self.send("inject", "packet", {
            "tile_id": tile_id,
            "index": index,
            "value": value,
            "priority": priority,
            "meta": meta or {},
        })

    def tick(self, steps: int = 1) -> dict:
        return self.send("tick", "step", {"steps": steps})


# =========================
# Test Harness
# =========================

def run_test_harness():
    """
    Simple test harness that sends JSON commands to a single organ.
    """
    organ = BorgMemoryOrgan(organ_id="test-organ", num_tiles=2, tile_size=4)
    client = MemoryBackboneClient(organ)

    print("=== TEST: inject packets ===")
    print(client.inject_packet(0, 1, 10, priority=1.2, meta={"tag": "test1"}))
    print(client.inject_packet(1, 2, 20, priority=0.8, meta={"tag": "test2"}))

    print("=== TEST: snapshot before ticks ===")
    print(client.snapshot_values())

    print("=== TEST: tick 5 steps ===")
    print(client.tick(steps=5))

    print("=== TEST: snapshot after ticks ===")
    print(client.snapshot_values())

    print("=== TEST: adjust forget_prob ===")
    print(client.set_param("forget_prob", 0.3))

    print("=== TEST: health ===")
    print(organ.handle_ipc_message({"topic": "debug", "cmd": "health", "args": {}, "id": "health1"}))


# =========================
# Swarm Orchestrator (Network-aware)
# =========================

class SwarmOrchestrator:
    """
    Manages multiple backbone organs as a swarm.
    Can manage local instances or remote daemons via RemoteOrganClient.
    """

    def __init__(self):
        self.local_organs: Dict[str, BorgMemoryOrgan] = {}
        self.remote_organs: Dict[str, RemoteOrganClient] = {}

    def add_local_organ(self, organ_id: str, num_tiles=3, tile_size=8) -> BorgMemoryOrgan:
        organ = BorgMemoryOrgan(organ_id=organ_id, num_tiles=num_tiles, tile_size=tile_size)
        self.local_organs[organ_id] = organ
        return organ

    def add_remote_organ(self, organ_id: str, host="127.0.0.1", port=7777) -> RemoteOrganClient:
        client = RemoteOrganClient(host=host, port=port)
        self.remote_organs[organ_id] = client
        return client

    def broadcast_config(self, param: str, value: float):
        # Local
        for organ in self.local_organs.values():
            if param == "forget_prob":
                organ.set_forget_prob(value)
            elif param == "misroute_prob":
                organ.set_misroute_prob(value)
            elif param == "meta_corrupt_prob":
                organ.set_meta_corrupt_prob(value)
            elif param == "ghost_prob":
                organ.set_ghost_prob(value)
            elif param == "loss_prob":
                organ.set_loss_prob(value)
            elif param == "double_prob":
                organ.set_double_prob(value)
        # Remote
        for client in self.remote_organs.values():
            client.set_param(param, value)

    def tick_all(self, steps: int = 1):
        for _ in range(steps):
            for organ in self.local_organs.values():
                organ.tick()
            for client in self.remote_organs.values():
                client.tick(1)

    def export_swarm_state(self) -> Dict[str, dict]:
        state = {}
        for oid, organ in self.local_organs.items():
            state[oid] = organ.export_state()
        for oid, client in self.remote_organs.items():
            state[oid] = client.export_state()
        return state


# =========================
# System Telemetry Organ
# =========================

class SystemTelemetryOrgan:
    """
    Samples real system data (CPU, RAM) and injects into a backbone via RemoteOrganClient.
    """

    def __init__(self, client: RemoteOrganClient, tile_id=0):
        self.client = client
        self.tile_id = tile_id

    def sample_and_inject(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        cpu_val = int(cpu)
        mem_val = int(mem)

        self.client.inject_packet(self.tile_id, 0, cpu_val, priority=1.5, meta={"metric": "cpu"})
        self.client.inject_packet(self.tile_id, 1, mem_val, priority=1.2, meta={"metric": "mem"})

    def run_loop(self, interval=1.0):
        while True:
            self.sample_and_inject()
            self.client.tick(steps=1)
            time.sleep(interval)


def run_telemetry_process(host="127.0.0.1", port=7777, interval=1.0, tile_id=0):
    client = RemoteOrganClient(host=host, port=port)
    organ = SystemTelemetryOrgan(client, tile_id=tile_id)
    organ.run_loop(interval=interval)


# =========================
# IPC Loop (stdin/stdout)
# =========================

def run_ipc_loop(organ: BorgMemoryOrgan):
    """
    IPC mode: read JSON lines from stdin, write JSON responses to stdout.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            resp = {"ok": False, "error": f"invalid_json: {e}"}
            print(json.dumps(resp), flush=True)
            continue

        resp = organ.handle_ipc_message(msg)
        print(json.dumps(resp), flush=True)


# =========================
# TCP Daemon
# =========================

class MemoryDaemon:
    """
    TCP daemon exposing BorgMemoryOrgan over newline-delimited JSON.
    """

    def __init__(self, organ: BorgMemoryOrgan, host="127.0.0.1", port=7777):
        self.organ = organ
        self.host = host
        self.port = port

    def handle_client(self, conn: socket.socket, addr):
        with conn:
            buf = ""
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data.decode("utf-8")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                        resp = self.organ.handle_ipc_message(msg)
                    except Exception as e:
                        resp = {"ok": False, "error": str(e)}
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

    def serve_forever(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen()
        print(f"[MemoryDaemon] Listening on {self.host}:{self.port}")
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
            t.start()


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--test" in args:
        run_test_harness()
        sys.exit(0)

    # Swarm demo: local only, but using new orchestrator
    if "--swarm-demo" in args:
        swarm = SwarmOrchestrator()
        o1 = swarm.add_local_organ("node-1")
        o2 = swarm.add_local_organ("node-2")
        o1.inject(0, 1, 10, priority=1.5, meta={"tag": "n1"})
        o2.inject(1, 2, 20, priority=0.7, meta={"tag": "n2"})
        swarm.broadcast_config("forget_prob", 0.2)
        swarm.tick_all(steps=5)
        print("Swarm state:", swarm.export_swarm_state())
        sys.exit(0)

    # Telemetry client mode (connects to remote daemon)
    if "--telemetry" in args:
        host = "127.0.0.1"
        port = 7777
        interval = 1.0
        tile_id = 0
        for a in args:
            if a.startswith("--host="):
                host = a.split("=", 1)[1]
            elif a.startswith("--port="):
                port = int(a.split("=", 1)[1])
            elif a.startswith("--interval="):
                interval = float(a.split("=", 1)[1])
            elif a.startswith("--tile="):
                tile_id = int(a.split("=", 1)[1])
        run_telemetry_process(host=host, port=port, interval=interval, tile_id=tile_id)
        sys.exit(0)

    # Default organ
    organ = BorgMemoryOrgan(organ_id="backbone-memory", num_tiles=3, tile_size=8)

    # Seed some flows
    organ.inject(0, 2, 10, priority=1.5, meta={"tag": "alpha"})
    organ.inject(1, 4, 20, priority=0.8, meta={"tag": "beta"})
    organ.inject(2, 1, -5, priority=2.0, meta={"tag": "gamma"})

    # TCP daemon mode
    if "--daemon" in args:
        host = "0.0.0.0"
        port = 7777
        for a in args:
            if a.startswith("--host="):
                host = a.split("=", 1)[1]
            elif a.startswith("--port="):
                port = int(a.split("=", 1)[1])
        daemon = MemoryDaemon(organ, host=host, port=port)
        daemon.serve_forever()

    # IPC stdin/stdout mode
    elif "--ipc" in args:
        run_ipc_loop(organ)

    # GUI mode (visualizer + control panel)
    elif tkinter is not None:
        root = tkinter.Tk()
        root.withdraw()

        viz = MemoryVisualizer(organ, root=root, cell_size=40, tick_ms=200)
        panel = BorgControlPanel(organ, root=root)

        root.mainloop()
    else:
        # Headless fallback
        for step in range(20):
            print(f"Step {step}")
            for t_id, tile_state in enumerate(organ.snapshot_values()):
                print(f"  Tile {t_id}: {tile_state}")
            organ.tick()