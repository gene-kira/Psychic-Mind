#!/usr/bin/env python3
"""
UNIFIED AUTOPILOT – FULL PROTOTYPE (SINGLE FILE)
------------------------------------------------
Features:
- Autoloader for dependencies
- Minimal, smartphone-style adaptive GUI
- Modes: WALKING, AUTOMOBILE, AIRCRAFT, BOAT, SUBMARINE, DRONE
- Automatic mode detection (simulated)
- Manual override
- AI Intent Engine (simple command parsing)
- Voice input (stubbed; optional real integration)
- Voice output (stubbed; optional real integration)
- Simulated GPS / sensor fusion
- Blind navigation guidance (simulated)
- Vehicle API stubs (car, aircraft, boat, submarine, drone)
"""

import importlib
import threading
import time
import random
import sys

# ---------------------------------------------------------------------------
# AUTOLOADER
# ---------------------------------------------------------------------------

REQUIRED = ["tkinter", "queue"]

OPTIONAL = {
    "speech_recognition": "voice input",
    "pyttsx3": "voice output",
}

def load_module(name, required=True):
    try:
        return importlib.import_module(name)
    except ImportError:
        if required:
            print(f"[AUTOLOADER] Missing required module: {name}")
            raise
        else:
            print(f"[AUTOLOADER] Optional module not found: {name} ({OPTIONAL.get(name, '')})")
            return None

modules = {m: load_module(m, required=True) for m in REQUIRED}
tk = modules["tkinter"]
queue = modules["queue"]

sr = load_module("speech_recognition", required=False)
pyttsx3 = load_module("pyttsx3", required=False)

# ---------------------------------------------------------------------------
# VEHICLE API STUBS
# ---------------------------------------------------------------------------

class VehicleBase:
    def __init__(self, name):
        self.name = name
        self.connected = False

    def connect(self):
        self.connected = True
        print(f"[VEHICLE] {self.name}: connected.")

    def disconnect(self):
        self.connected = False
        print(f"[VEHICLE] {self.name}: disconnected.")

    def send_command(self, cmd):
        if not self.connected:
            print(f"[VEHICLE] {self.name}: not connected, cannot send command.")
            return
        print(f"[VEHICLE] {self.name}: command -> {cmd}")

class CarAPI(VehicleBase):
    def __init__(self):
        super().__init__("Car")

class AircraftAPI(VehicleBase):
    def __init__(self):
        super().__init__("Aircraft")

class BoatAPI(VehicleBase):
    def __init__(self):
        super().__init__("Boat")

class SubmarineAPI(VehicleBase):
    def __init__(self):
        super().__init__("Submarine")

class DroneAPI(VehicleBase):
    def __init__(self):
        super().__init__("Drone")

# ---------------------------------------------------------------------------
# SIMULATED GPS / SENSOR FUSION
# ---------------------------------------------------------------------------

class SensorFusion:
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.heading = 0.0

    def update(self):
        # Simulate movement / drift
        self.lat += random.uniform(-0.0001, 0.0001)
        self.lon += random.uniform(-0.0001, 0.0001)
        self.alt = max(0.0, self.alt + random.uniform(-1.0, 1.0))
        self.heading = (self.heading + random.uniform(-5.0, 5.0)) % 360.0

    def get_status(self):
        return f"Lat: {self.lat:.5f}, Lon: {self.lon:.5f}, Alt: {self.alt:.1f}m, Heading: {self.heading:.1f}°"

# ---------------------------------------------------------------------------
# BLIND NAVIGATION (SIMULATED)
# ---------------------------------------------------------------------------

class BlindNavigator:
    def __init__(self):
        self.last_instruction = "Standing by."

    def compute_instruction(self, mode):
        if mode == "WALKING":
            self.last_instruction = random.choice([
                "Walk forward 3 steps.",
                "Turn slightly left.",
                "Stop and wait.",
                "Turn right 90 degrees.",
            ])
        else:
            self.last_instruction = "AI guiding via vehicle instruments."

        return self.last_instruction

# ---------------------------------------------------------------------------
# AI INTENT ENGINE
# ---------------------------------------------------------------------------

class IntentEngine:
    """
    Very simple text-based intent parser.
    Commands (examples):
    - 'switch to car' / 'automobile mode'
    - 'aircraft mode'
    - 'boat mode'
    - 'submarine mode'
    - 'drone mode'
    - 'walking mode'
    - 'manual override on/off'
    - 'status'
    """

    def __init__(self, state, msg_queue):
        self.state = state
        self.msg_queue = msg_queue

    def handle_text(self, text: str):
        t = text.lower().strip()
        if not t:
            return

        self.msg_queue.put(f"[INTENT] Heard: {t}")

        if "manual override" in t and "on" in t:
            if not self.state.manual_override:
                self.state.manual_override = True
                self.state.status = "Manual override engaged (via voice)."
        elif "manual override" in t and ("off" in t or "automatic" in t):
            if self.state.manual_override:
                self.state.manual_override = False
                self.state.status = "Automatic mode restored (via voice)."
        elif "car" in t or "automobile" in t:
            self.state.set_mode("AUTOMOBILE")
        elif "aircraft" in t or "plane" in t or "airplane" in t:
            self.state.set_mode("AIRCRAFT")
        elif "boat" in t or "ship" in t:
            self.state.set_mode("BOAT")
        elif "submarine" in t or "sub" in t:
            self.state.set_mode("SUBMARINE")
        elif "drone" in t:
            self.state.set_mode("DRONE")
        elif "walk" in t or "walking" in t or "on foot" in t:
            self.state.set_mode("WALKING")
        elif "status" in t:
            self.msg_queue.put(f"[INTENT] Status requested.")
        else:
            self.msg_queue.put(f"[INTENT] Command not recognized.")

# ---------------------------------------------------------------------------
# VOICE I/O (OPTIONAL)
# ---------------------------------------------------------------------------

class VoiceIO(threading.Thread):
    """
    Optional voice input using speech_recognition and voice output using pyttsx3.
    If libraries are missing, this becomes a no-op.
    """

    def __init__(self, intent_engine, msg_queue):
        super().__init__(daemon=True)
        self.intent_engine = intent_engine
        self.msg_queue = msg_queue
        self.running = True
        self.recognizer = sr.Recognizer() if sr else None
        self.engine = pyttsx3.init() if pyttsx3 else None

    def speak(self, text):
        if self.engine:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            self.msg_queue.put(f"[VOICE-OUT] {text}")

    def run(self):
        if not self.recognizer:
            self.msg_queue.put("[VOICE] speech_recognition not available; voice input disabled.")
            return

        mic = None
        try:
            mic = sr.Microphone()
        except Exception as e:
            self.msg_queue.put(f"[VOICE] No microphone available: {e}")
            return

        self.msg_queue.put("[VOICE] Voice listener started. Say commands like 'car mode', 'manual override on', 'status'.")

        while self.running:
            try:
                with mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                try:
                    text = self.recognizer.recognize_google(audio)
                    self.intent_engine.handle_text(text)
                except sr.UnknownValueError:
                    self.msg_queue.put("[VOICE] Could not understand audio.")
                except sr.RequestError as e:
                    self.msg_queue.put(f"[VOICE] Recognition error: {e}")
            except Exception:
                # Timeout or other non-fatal issues
                pass

# ---------------------------------------------------------------------------
# AUTOPILOT STATE MACHINE
# ---------------------------------------------------------------------------

class AutopilotState:
    MODES = [
        "WALKING",
        "AUTOMOBILE",
        "AIRCRAFT",
        "BOAT",
        "SUBMARINE",
        "DRONE"
    ]

    def __init__(self):
        self.mode = "WALKING"
        self.manual_override = False
        self.status = "System initialized."
        self.running = True

        # Subsystems
        self.sensors = SensorFusion()
        self.blind_nav = BlindNavigator()

        # Vehicle APIs
        self.car = CarAPI()
        self.aircraft = AircraftAPI()
        self.boat = BoatAPI()
        self.submarine = SubmarineAPI()
        self.drone = DroneAPI()

    def set_mode(self, mode):
        if mode not in self.MODES:
            return
        self.mode = mode
        self.status = f"Mode switched to {mode}."

    def toggle_manual(self):
        self.manual_override = not self.manual_override
        if self.manual_override:
            self.status = "Manual override engaged."
        else:
            self.status = "Automatic mode restored."

# ---------------------------------------------------------------------------
# BACKGROUND BRAIN
# ---------------------------------------------------------------------------

class AutopilotBrain(threading.Thread):
    """
    Simulates:
    - Automatic mode detection (when not in manual override)
    - Sensor updates
    - Blind navigation instructions
    - Vehicle API usage (stubbed)
    """

    def __init__(self, state, msg_queue):
        super().__init__(daemon=True)
        self.state = state
        self.msg_queue = msg_queue

    def run(self):
        while self.state.running:
            # Update sensors
            self.state.sensors.update()
            sensor_status = self.state.sensors.get_status()
            self.msg_queue.put(f"[SENSORS] {sensor_status}")

            # Automatic mode detection (simulated)
            if not self.state.manual_override:
                if random.random() < 0.2:  # 20% chance to change mode
                    new_mode = random.choice(self.state.MODES)
                    if new_mode != self.state.mode:
                        self.state.set_mode(new_mode)
                        self.msg_queue.put(f"[AUTO] Detected mode: {new_mode}")

            # Blind navigation instruction
            instruction = self.state.blind_nav.compute_instruction(self.state.mode)
            self.msg_queue.put(f"[BLIND-NAV] {instruction}")

            # Vehicle API stub usage
            if self.state.mode == "AUTOMOBILE":
                self.state.car.connect()
                self.state.car.send_command("Maintain lane, follow route.")
            elif self.state.mode == "AIRCRAFT":
                self.state.aircraft.connect()
                self.state.aircraft.send_command("Maintain altitude and heading.")
            elif self.state.mode == "BOAT":
                self.state.boat.connect()
                self.state.boat.send_command("Hold course, avoid obstacles.")
            elif self.state.mode == "SUBMARINE":
                self.state.submarine.connect()
                self.state.submarine.send_command("Maintain depth and heading.")
            elif self.state.mode == "DRONE":
                self.state.drone.connect()
                self.state.drone.send_command("Hover and await instructions.")
            else:
                # WALKING or others: no vehicle
                pass

            self.msg_queue.put(f"[STATUS] {self.state.status}")
            time.sleep(3)

# ---------------------------------------------------------------------------
# GUI PANELS
# ---------------------------------------------------------------------------

class CockpitPanels:

    @staticmethod
    def walking(frame, sensors, blind_instruction):
        text = (
            "WALKING MODE\n"
            "Blind Navigation Active\n\n"
            f"{sensors}\n\n"
            f"Instruction:\n{blind_instruction}"
        )
        return tk.Label(frame, text=text, fg="white", bg="#222", justify="left")

    @staticmethod
    def automobile(frame, sensors):
        text = (
            "AUTOMOBILE MODE\n"
            "Speed • RPM • GPS (simulated)\n\n"
            f"{sensors}"
        )
        return tk.Label(frame, text=text, fg="white", bg="#222", justify="left")

    @staticmethod
    def aircraft(frame, sensors):
        text = (
            "AIRCRAFT MODE\n"
            "Altitude • Heading • VSI (simulated)\n\n"
            f"{sensors}"
        )
        return tk.Label(frame, text=text, fg="white", bg="#222", justify="left")

    @staticmethod
    def boat(frame, sensors):
        text = (
            "BOAT MODE\n"
            "Compass • Depth • Waves (simulated)\n\n"
            f"{sensors}"
        )
        return tk.Label(frame, text=text, fg="white", bg="#222", justify="left")

    @staticmethod
    def submarine(frame, sensors):
        text = (
            "SUBMARINE MODE\n"
            "Depth • Sonar • Ballast (simulated)\n\n"
            f"{sensors}"
        )
        return tk.Label(frame, text=text, fg="white", bg="#222", justify="left")

    @staticmethod
    def drone(frame, sensors):
        text = (
            "DRONE MODE\n"
            "Altitude • Battery • Camera (simulated)\n\n"
            f"{sensors}"
        )
        return tk.Label(frame, text=text, fg="white", bg="#222", justify="left")

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class AutopilotGUI:

    def __init__(self, root, state, msg_queue, intent_engine):
        self.root = root
        self.state = state
        self.msg_queue = msg_queue
        self.intent_engine = intent_engine

        self.last_blind_instruction = "Standing by."

        self.root.title("Unified Autopilot")
        self.root.geometry("360x640")
        self.root.configure(bg="#111")

        # Mode label
        self.mode_label = tk.Label(root, text="Mode: WALKING", fg="#00FFAA", bg="#111", font=("Helvetica", 16))
        self.mode_label.pack(pady=10)

        # Cockpit frame
        self.cockpit_frame = tk.Frame(root, bg="#222", width=320, height=260)
        self.cockpit_frame.pack(pady=10)
        self.cockpit_frame.pack_propagate(False)

        # Status box
        self.status_box = tk.Text(root, height=12, width=40, bg="#000", fg="#0F0", font=("Consolas", 9))
        self.status_box.pack(pady=10)
        self.status_box.insert("end", "System initialized.\n")
        self.status_box.config(state="disabled")

        # Manual override button
        self.btn_manual = tk.Button(root, text="Manual Override", command=self.toggle_manual, bg="#550000", fg="white")
        self.btn_manual.pack(pady=5)

        # Text command entry (for testing intent engine without voice)
        self.cmd_entry = tk.Entry(root, width=30)
        self.cmd_entry.pack(pady=5)
        self.cmd_entry.insert(0, "Type command here (e.g., 'car mode')")
        self.cmd_entry.bind("<Return>", self.on_text_command)

        self.btn_cmd = tk.Button(root, text="Send Command", command=self.on_text_command_btn, bg="#333", fg="white")
        self.btn_cmd.pack(pady=5)

        # Exit button
        self.btn_exit = tk.Button(root, text="Exit", command=self.exit, bg="#333", fg="white")
        self.btn_exit.pack(pady=5)

        # Start UI loops
        self.update_ui()
        self.root.after(200, self.poll_messages)

    def toggle_manual(self):
        self.state.toggle_manual()
        self.append(f"[USER] Manual override toggled.")

    def exit(self):
        self.state.running = False
        self.append("[USER] Exiting system...")
        self.root.after(500, self.root.destroy)

    def append(self, text):
        self.status_box.config(state="normal")
        self.status_box.insert("end", text + "\n")
        self.status_box.see("end")
        self.status_box.config(state="disabled")

    def on_text_command(self, event=None):
        text = self.cmd_entry.get()
        self.intent_engine.handle_text(text)
        self.cmd_entry.delete(0, "end")

    def on_text_command_btn(self):
        self.on_text_command()

    def update_ui(self):
        # Update mode label
        self.mode_label.config(text=f"Mode: {self.state.mode}")

        # Capture latest blind instruction from state
        self.last_blind_instruction = self.state.blind_nav.last_instruction

        # Clear cockpit frame
        for widget in self.cockpit_frame.winfo_children():
            widget.destroy()

        sensors = self.state.sensors.get_status()

        if self.state.mode == "WALKING":
            panel = CockpitPanels.walking(self.cockpit_frame, sensors, self.last_blind_instruction)
        elif self.state.mode == "AUTOMOBILE":
            panel = CockpitPanels.automobile(self.cockpit_frame, sensors)
        elif self.state.mode == "AIRCRAFT":
            panel = CockpitPanels.aircraft(self.cockpit_frame, sensors)
        elif self.state.mode == "BOAT":
            panel = CockpitPanels.boat(self.cockpit_frame, sensors)
        elif self.state.mode == "SUBMARINE":
            panel = CockpitPanels.submarine(self.cockpit_frame, sensors)
        elif self.state.mode == "DRONE":
            panel = CockpitPanels.drone(self.cockpit_frame, sensors)
        else:
            panel = tk.Label(self.cockpit_frame, text="UNKNOWN MODE", fg="white", bg="#222")

        panel.pack(expand=True, fill="both")

        self.root.after(500, self.update_ui)

    def poll_messages(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self.append(msg)
        except queue.Empty:
            pass

        self.root.after(200, self.poll_messages)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    state = AutopilotState()
    msg_queue = queue.Queue()

    intent_engine = IntentEngine(state, msg_queue)

    brain = AutopilotBrain(state, msg_queue)
    brain.start()

    voice_thread = None
    if sr is not None:
        voice_thread = VoiceIO(intent_engine, msg_queue)
        voice_thread.start()
    else:
        msg_queue.put("[VOICE] Voice libraries not available; skipping voice thread.")

    root = tk.Tk()
    gui = AutopilotGUI(root, state, msg_queue, intent_engine)
    root.mainloop()

    state.running = False
    if voice_thread:
        voice_thread.running = False

if __name__ == "__main__":
    main()
