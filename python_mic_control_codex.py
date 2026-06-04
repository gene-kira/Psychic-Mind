# === AUTO-ELEVATION CHECK ===
import ctypes
import os
import sys

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

# ============================================================
# Tier‑4 Codex Autoloader + Python Mic Control GUI
# ============================================================

import time
import subprocess
import importlib
import json

# ----------------------------
# TIER‑4 CODEX AUTOLOADER
# ----------------------------

REQUIRED_LIBS = {
    "PyQt5": {
        "pip": "pyqt5",
        "threat_level": "low"
    },
    "psutil": {
        "pip": "psutil",
        "threat_level": "low"
    }
    # winreg is built-in
}

CODEX_NODE_SYNC_FILE = os.path.join(
    os.path.expanduser("~"),
    ".codex_purge_shell",
    "python_mic_node_sync.json"
)

GLYPHS = ["◐", "◓", "◑", "◒"]


def codex_log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[CODEX {ts}] {msg}"
    print(line)
    return line


def ensure_node_sync_dir():
    d = os.path.dirname(CODEX_NODE_SYNC_FILE)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def write_node_sync(status):
    ensure_node_sync_dir()
    try:
        data = {
            "node": "python_mic_control",
            "status": status,
            "timestamp": time.time()
        }
        with open(CODEX_NODE_SYNC_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        codex_log(f"Node-sync write failed: {e}")


def threat_aware_validate(lib_name):
    meta = REQUIRED_LIBS.get(lib_name, {})
    threat = meta.get("threat_level", "unknown")
    if threat not in ("low", "medium"):
        codex_log(f"THREAT ALERT: {lib_name} threat level {threat}")
    else:
        codex_log(f"Threat check OK for {lib_name} ({threat})")


def install_lib(pip_name):
    codex_log(f"Installing dependency via pip: {pip_name}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def tier4_autoload():
    write_node_sync("autoload_start")
    glyph_index = 0
    for lib, meta in REQUIRED_LIBS.items():
        threat_aware_validate(lib)
        try:
            importlib.import_module(lib)
            codex_log(f"{GLYPHS[glyph_index % len(GLYPHS)]} Resurrection check: {lib} already present")
        except ImportError:
            codex_log(f"{GLYPHS[glyph_index % len(GLYPHS)]} Resurrection needed: {lib} missing")
            install_lib(meta["pip"])
            importlib.invalidate_caches()
            importlib.import_module(lib)
            codex_log(f"{GLYPHS[glyph_index % len(GLYPHS)]} Resurrection complete: {lib} loaded")
        glyph_index += 1
    write_node_sync("autoload_complete")


tier4_autoload()

# ============================================================
# Now safe to import heavy deps
# ============================================================

import winreg
import psutil
from PyQt5 import QtWidgets, QtGui, QtCore

# ============================================================
# CODEX SYSTEM SCANNER — AUTO-DETECT MIC BLOCKING METHOD
# ============================================================

def codex_system_scan():
    result = {
        "win32_privacy_supported": False,
        "nonpackaged_exists": False,
        "uwp_privacy_model": False,
        "python_uses_lowlevel_audio": False,
        "registry_block_allowed": False,
        "recommended_mode": "unknown"
    }

    try:
        winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
        )
        result["win32_privacy_supported"] = True
    except:
        result["win32_privacy_supported"] = False

    try:
        winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone\NonPackaged"
        )
        result["nonpackaged_exists"] = True
    except:
        result["nonpackaged_exists"] = False

    try:
        out = subprocess.check_output("tasklist /m python.exe", shell=True, text=True)
        lowlevel = ["audioses.dll", "mmdevapi.dll", "dsound.dll", "wdmaud.drv"]
        if any(dll.lower() in out.lower() for dll in lowlevel):
            result["python_uses_lowlevel_audio"] = True
    except:
        result["python_uses_lowlevel_audio"] = False

    if result["win32_privacy_supported"] and result["nonpackaged_exists"]:
        result["registry_block_allowed"] = True

    if not result["win32_privacy_supported"] and not result["nonpackaged_exists"]:
        result["uwp_privacy_model"] = True

    if result["registry_block_allowed"]:
        result["recommended_mode"] = "registry"
    elif result["uwp_privacy_model"]:
        result["recommended_mode"] = "uwp_broker"
    elif result["python_uses_lowlevel_audio"]:
        result["recommended_mode"] = "device_interceptor"
    else:
        result["recommended_mode"] = "watchdog_only"

    codex_log(f"System Scan Result: {result}")
    return result


SYSTEM_STATUS = codex_system_scan()

# ============================================================
# CONFIG
# ============================================================

PYTHON_PATHS = [
    r"C:\Users\<you>\AppData\Local\Programs\Python\Python3x\python.exe",
    r"C:\Users\<you>\anaconda3\python.exe",
    r"C:\Python3x\python.exe"
]

REG_BASE = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone\NonPackaged"
LOG_FILE = r"C:\ProgramData\PythonMicControl\python_mic_log.txt"

log_dir = os.path.dirname(LOG_FILE)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

def write_log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    return line.strip()


# ============================================================
# REGISTRY HELPERS
# ============================================================

def ensure_parent_keys():
    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore"
    mic = base + r"\microphone"
    nonpack = mic + r"\NonPackaged"
    try:
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, base)
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, mic)
        winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, nonpack)
    except Exception as e:
        write_log(f"Parent key creation error: {e}")


def block_python_mic():
    ensure_parent_keys()
    for path in PYTHON_PATHS:
        if os.path.exists(path):
            safe = path.replace("\\", "_")
            try:
                key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\{safe}")
                winreg.SetValueEx(key, "Value", 0, winreg.REG_SZ, "Deny")
                winreg.SetValueEx(key, "LastUsedTimeStart", 0, winreg.REG_QWORD, 0)
                winreg.SetValueEx(key, "LastUsedTimeStop", 0, winreg.REG_QWORD, 0)
                write_log(f"Mic access DENIED for {path}")
            except PermissionError:
                write_log("ERROR: Run as Administrator")


def allow_python_mic():
    for path in PYTHON_PATHS:
        safe = path.replace("\\", "_")
        try:
            winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\{safe}")
            write_log(f"Mic access ALLOWED for {path}")
        except FileNotFoundError:
            pass
        except PermissionError:
            write_log("ERROR: Run as Administrator")


def get_status():
    for path in PYTHON_PATHS:
        safe = path.replace("\\", "_")
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"{REG_BASE}\\{safe}")
            val, _ = winreg.QueryValueEx(key, "Value")
            if val == "Deny":
                return "Blocked"
        except FileNotFoundError:
            continue
    return "Allowed"


# ============================================================
# WATCHDOG THREAD
# ============================================================

class Watchdog(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.enabled = True

    def run(self):
        while True:
            if self.enabled and get_status() == "Blocked":
                for proc in psutil.process_iter(['pid', 'name', 'exe']):
                    try:
                        if proc.info['name'] and proc.info['name'].lower() == "python.exe":
                            msg = write_log(f"[WATCHDOG] Killing python PID {proc.pid} at {proc.info.get('exe')}")
                            self.log_signal.emit(msg)
                            proc.kill()
                    except Exception:
                        pass
            time.sleep(2)


# ============================================================
# DEVICE-LEVEL MIC INTERCEPTOR (PROCESS-LEVEL KILLER)
# ============================================================

class DeviceLevelInterceptor(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.enabled = False

    def run(self):
        while True:
            if self.enabled:
                for proc in psutil.process_iter(['pid', 'name', 'exe']):
                    try:
                        if proc.info['name'] and proc.info['name'].lower() == "python.exe":
                            msg = write_log(f"[DEVICE-INTERCEPTOR] Killing python PID {proc.pid} at {proc.info.get('exe')}")
                            self.log_signal.emit(msg)
                            proc.kill()
                    except Exception:
                        pass
            time.sleep(1.0)


# ============================================================
# ANIMATED OVERLAY (SPLASH)
# ============================================================

class CodexSplash(QtWidgets.QSplashScreen):
    def __init__(self):
        pixmap = QtGui.QPixmap(400, 200)
        super().__init__(pixmap)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.counter = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(120)

    def animate(self):
        self.counter += 1
        pixmap = QtGui.QPixmap(400, 200)
        pixmap.fill(QtGui.QColor(10, 10, 20))
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QColor(0, 255, 180))
        painter.setFont(QtGui.QFont("Consolas", 12))
        glyph = GLYPHS[self.counter % len(GLYPHS)]
        text = f"Codex Purge Shell Node: python_mic_control\nResurrection Cycle {self.counter} {glyph}"
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, text)
        painter.end()
        self.setPixmap(pixmap)


# ============================================================
# MAIN GUI
# ============================================================

class MicControlGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Python Microphone Control — Codex Panel")
        self.setGeometry(200, 200, 700, 500)

        layout = QtWidgets.QVBoxLayout()

        self.status_label = QtWidgets.QLabel()
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self.status_label)

        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_block = QtWidgets.QPushButton("Disable Python Mic")
        self.btn_block.clicked.connect(self.disable_mic)
        btn_layout.addWidget(self.btn_block)

        self.btn_allow = QtWidgets.QPushButton("Enable Python Mic")
        self.btn_allow.clicked.connect(self.enable_mic)
        btn_layout.addWidget(self.btn_allow)

        self.btn_watchdog = QtWidgets.QPushButton("Toggle Watchdog")
        self.btn_watchdog.clicked.connect(self.toggle_watchdog)
        btn_layout.addWidget(self.btn_watchdog)

        layout.addLayout(btn_layout)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                self.log_box.setText(f.read())

        self.watchdog = Watchdog()
        self.watchdog.log_signal.connect(self.append_log)
        self.watchdog.start()

        self.device_interceptor = DeviceLevelInterceptor()
        self.device_interceptor.log_signal.connect(self.append_log)
        self.device_interceptor.start()

        self.tray = QtWidgets.QSystemTrayIcon(self)
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        self.tray.setIcon(icon)
        self.tray.setVisible(True)

        menu = QtWidgets.QMenu()
        menu.addAction("Show Window", self.show)
        menu.addAction("Exit", self.exit_app)
        self.tray.setContextMenu(menu)

        self.update_status()

    def append_log(self, text):
        self.log_box.append(text)

    def update_status(self):
        status = get_status()
        if status == "Blocked":
            self.status_label.setText("Python Mic: BLOCKED")
            self.status_label.setStyleSheet("color: red; font-size: 22px; font-weight: bold;")
        else:
            self.status_label.setText("Python Mic: ALLOWED")
            self.status_label.setStyleSheet("color: lime; font-size: 22px; font-weight: bold;")

    def disable_mic(self):
        mode = SYSTEM_STATUS.get("recommended_mode", "watchdog_only")

        if mode == "registry":
            block_python_mic()
            write_log("Registry mic block applied.")
        elif mode == "uwp_broker":
            write_log("UWP privacy model detected — registry mic block ignored. Using watchdog-only mode.")
        elif mode == "device_interceptor":
            write_log("Device-level interceptor mode: killing any python.exe while blocked.")
            self.device_interceptor.enabled = True
        else:
            write_log("Fallback: watchdog-only mode active.")

        self.update_status()

    def enable_mic(self):
        allow_python_mic()
        self.device_interceptor.enabled = False
        write_log("Device-level interceptor disabled (Python allowed).")
        self.update_status()

    def toggle_watchdog(self):
        self.watchdog.enabled = not self.watchdog.enabled
        state = "ENABLED" if self.watchdog.enabled else "DISABLED"
        msg = write_log(f"Watchdog {state}")
        self.append_log(msg)

    def exit_app(self):
        self.watchdog.enabled = False
        self.device_interceptor.enabled = False
        self.watchdog.terminate()
        self.device_interceptor.terminate()
        QtWidgets.QApplication.quit()

    # Codex Purge Shell integration hook
    def start_from_codex(self):
        self.show()


# ============================================================
# MAIN ENTRY
# ============================================================

def main():
    app = QtWidgets.QApplication(sys.argv)

    splash = CodexSplash()
    splash.show()
    QtWidgets.QApplication.processEvents()
    time.sleep(1.5)
    splash.close()

    window = MicControlGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
