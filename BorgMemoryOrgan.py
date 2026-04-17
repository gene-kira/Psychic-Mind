"""
borg_memory_backbone.py

A living memory backbone organ with:
- Flow-based addressing
- Intelligent routing
- Alzheimer-like degradation with progression
- Stabilizers for entropy and ghosts
- Tkinter GUI visualizer
- Borg-OS control panel (live tuning)
- Swarm integration hooks
- JSON IPC protocol over stdin/stdout
- Autoloader for dependencies
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


mods = autoload(["random", "dataclasses", "typing", "tkinter", "json", "sys"])
random = mods["random"]
dataclasses = mods["dataclasses"]
typing = mods["typing"]
tkinter = mods["tkinter"]  # may be None
json = mods["json"]
sys = mods["sys"]

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
            return ghost

        return pkt


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
    ):
        self.organ_id = organ_id
        self.num_tiles = num_tiles
        self.tiles = [MemoryTile(tile_size, t) for t in range(num_tiles)]
        base = base_policy or BasePolicy(p_loss=0.1, p_double=0.1)
        self.policy = AlzheimerLayer(base, num_tiles=num_tiles)
        self.next_pid = 1
        self.debug_log: List[str] = []

    # ---- Swarm Hooks ----

    def export_state(self) -> dict:
        return {
            "organ_id": self.organ_id,
            "tiles": [[(p.value, p.energy) if p else None for p in tile.cells] for tile in self.tiles],
            "time": self.policy.time,
        }

    def import_state(self, state: dict):
        # Placeholder for future swarm sync logic
        pass

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
    # Protocol: one JSON object per line on stdin/stdout.
    # Message format:
    # {
    #   "topic": "config" | "inject" | "snapshot" | "state",
    #   "cmd": "...",
    #   "args": {...},
    #   "id": "optional-correlation-id"
    # }

    def handle_ipc_message(self, msg: dict) -> dict:
        topic = msg.get("topic")
        cmd = msg.get("cmd")
        args = msg.get("args", {}) or {}
        mid = msg.get("id")

        resp: Dict[str, Any] = {"id": mid, "ok": True, "topic": topic, "cmd": cmd}

        try:
            if topic == "config":
                if cmd == "set_forget_prob":
                    self.set_forget_prob(float(args["value"]))
                elif cmd == "set_misroute_prob":
                    self.set_misroute_prob(float(args["value"]))
                elif cmd == "set_meta_corrupt_prob":
                    self.set_meta_corrupt_prob(float(args["value"]))
                elif cmd == "set_ghost_prob":
                    self.set_ghost_prob(float(args["value"]))
                elif cmd == "set_loss_prob":
                    self.set_loss_prob(float(args["value"]))
                elif cmd == "set_double_prob":
                    self.set_double_prob(float(args["value"]))
                else:
                    resp["ok"] = False
                    resp["error"] = "unknown_config_cmd"

            elif topic == "inject":
                tile_id = int(args.get("tile_id", 0))
                index = int(args.get("index", 0))
                value = int(args.get("value", 0))
                priority = float(args.get("priority", 1.0))
                meta = args.get("meta", {})
                pid = self.inject(tile_id, index, value, priority, meta)
                resp["pid"] = pid

            elif topic == "snapshot":
                resp["values"] = self.snapshot_values()
                resp["energies"] = self.snapshot_energy()

            elif topic == "state":
                resp["state"] = self.export_state()

            elif topic == "tick":
                steps = int(args.get("steps", 1))
                for _ in range(steps):
                    self.tick()

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
    """

    def __init__(self, organ: BorgMemoryOrgan, root=None, cell_size=40, tick_ms=300):
        if tkinter is None:
            raise RuntimeError("Tkinter not available; GUI cannot be created.")
        self.organ = organ
        self.cell_size = cell_size
        self.tick_ms = tick_ms

        self.root = root or tkinter.Tk()
        self.window = tkinter.Toplevel(self.root)
        self.window.title(f"Borg Memory Organ - {organ.organ_id}")

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
# Entry Point
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


if __name__ == "__main__":
    organ = BorgMemoryOrgan(organ_id="backbone-memory", num_tiles=3, tile_size=8)

    # Seed some flows
    organ.inject(0, 2, 10, priority=1.5, meta={"tag": "alpha"})
    organ.inject(1, 4, 20, priority=0.8, meta={"tag": "beta"})
    organ.inject(2, 1, -5, priority=2.0, meta={"tag": "gamma"})

    args = sys.argv[1:]

    if "--ipc" in args:
        # Headless IPC backbone mode
        run_ipc_loop(organ)

    elif tkinter is not None:
        # GUI backbone mode: visualizer + control panel
        root = tkinter.Tk()
        root.withdraw()

        viz = MemoryVisualizer(organ, root=root, cell_size=40, tick_ms=200)
        panel = BorgControlPanel(organ, root=root)

        root.mainloop()
    else:
        # Pure text fallback
        for step in range(20):
            print(f"Step {step}")
            for t_id, tile_state in enumerate(organ.snapshot_values()):
                print(f"  Tile {t_id}: {tile_state}")
            organ.tick()