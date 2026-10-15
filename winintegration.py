"""
The parts of being a normal Windows app that are not audio: one running copy per user, Start with Windows, and the
checks the first-run setup shows (VB-CABLE present, microphone privacy). Plain ctypes and winreg, no Qt here, so the
launcher can use it before anything heavy loads. Every function is safe to call off Windows (it reports "unknown").
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from appinfo import APP_NAME

# The installer uses the mutex name too (Inno Setup AppMutex) to see that the app is running.
APP_MUTEX = "V-COMMS-AppMutex"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
STARTUP_APPROVED_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"
BACKGROUND_ARG = "--background"
MMDEVICES = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio"
PKEY_DEVICE_NAME = "{a45c254e-df1c-4efd-8020-67d146a850e0},2"          # "CABLE Input"
PKEY_INTERFACE_NAME = "{b3f8fa53-0004-438e-9003-51a46e139bfc},6"       # "VB-Audio Virtual Cable"
CONSENT_MIC = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
PRIVACY_MIC_URI = "ms-settings:privacy-microphone"
SOUND_SETTINGS_URI = "ms-settings:sound"
VB_CABLE_URL = "https://vb-audio.com/Cable/"

ERROR_ALREADY_EXISTS = 183
SYNCHRONIZE = 0x00100000
_mutex_handle = None


def _winreg():
    if sys.platform != "win32":
        return None
    import winreg

    return winreg


# ------------------------------------------------------------------------------------------------ single instance
def claim_instance() -> bool:
    """Creates the app mutex for this session. True when this is the only running copy. The handle lives as long as
    the process, so Windows releases it even after a crash."""
    global _mutex_handle
    if sys.platform != "win32":
        return True
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, APP_MUTEX)
    already = kernel32.GetLastError() == ERROR_ALREADY_EXISTS
    if handle:
        _mutex_handle = handle
    return not already


def instance_running() -> bool:
    """Whether some copy holds the app mutex (used by --quit to wait for the other copy to end)."""
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenMutexW.restype = ctypes.c_void_p
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, APP_MUTEX)
    if not handle:
        return False
    kernel32.CloseHandle(ctypes.c_void_p(handle))
    return True


def allow_other_to_take_focus() -> None:
    """Lets the running copy bring its window to the front when this (second) copy asks it to."""
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.AllowSetForegroundWindow(-1)  # ASFW_ANY
        except Exception:
            pass


def server_name() -> str:
    """The local socket the running copy listens on: one per Windows user."""
    return f"{APP_NAME}-{os.environ.get('USERNAME', 'user')}-instance"


# ------------------------------------------------------------------------------------------------ start with Windows
def launch_command() -> str:
    """How Windows should start this copy at sign-in: the installed exe, or the launcher script from source."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" {BACKGROUND_ARG}'
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    exe = pythonw if pythonw.exists() else Path(sys.executable)
    script = Path(__file__).resolve().with_name("vcomms_main.py")
    return f'"{exe}" "{script}" {BACKGROUND_ARG}'


def autostart_enabled() -> bool:
    """On when the Run entry exists and Task Manager's Startup apps page has not switched it off."""
    winreg = _winreg()
    if winreg is None:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, APP_NAME)
    except OSError:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, STARTUP_APPROVED_KEY) as key:
            data, _ = winreg.QueryValueEx(key, APP_NAME)
            if isinstance(data, bytes) and data[:1] not in (b"\x02", b"\x06"):  # 3 or 7 = disabled in Task Manager
                return False
    except OSError:
        pass
    return True


def set_autostart(enabled: bool) -> bool:
    """Adds or removes the per-user Run entry (no admin needed). Returns whether the change stuck."""
    winreg = _winreg()
    if winreg is None:
        return False
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            if enabled:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, launch_command())
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
        if enabled:  # turning it on here also undoes an earlier "Disable" in Task Manager
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, STARTUP_APPROVED_KEY, 0, winreg.KEY_SET_VALUE) as key:
                    winreg.DeleteValue(key, APP_NAME)
            except OSError:
                pass
    except OSError:
        return False
    return autostart_enabled() == enabled


# ------------------------------------------------------------------------------------------------ setup checks
def cable_status() -> Dict[str, str]:
    """
    Whether VB-CABLE's two endpoints exist, read from Windows' own audio device list (no audio API, no admin):
    {"input": ..., "output": ...} with "ok", "disabled", "unplugged" or "missing" ("unknown" off Windows).
    "input" is the playback device V-COMMS writes to; "output" is the recording device VALORANT listens to.
    """
    winreg = _winreg()
    if winreg is None:
        return {"input": "unknown", "output": "unknown"}
    states = {1: "ok", 2: "disabled", 4: "missing", 8: "unplugged"}
    result = {"input": "missing", "output": "missing"}
    for flow, role, wanted in (("Render", "input", "CABLE Input"), ("Capture", "output", "CABLE Output")):
        try:
            root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, rf"{MMDEVICES}\{flow}")
        except OSError:
            continue
        with root:
            i = 0
            while True:
                try:
                    guid = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                try:
                    with winreg.OpenKey(root, guid) as dev, winreg.OpenKey(root, guid + r"\Properties") as props:
                        name = winreg.QueryValueEx(props, PKEY_DEVICE_NAME)[0]
                        iface = winreg.QueryValueEx(props, PKEY_INTERFACE_NAME)[0]
                        state = int(winreg.QueryValueEx(dev, "DeviceState")[0])
                except OSError:
                    continue
                if name == wanted and "VB-Audio" in str(iface):
                    found = states.get(state, "disabled")
                    if result[role] != "ok":  # an active copy wins over a disabled one
                        result[role] = found
    return result


def mic_privacy() -> str:
    """
    Whether Windows lets desktop apps use the microphone:
      "ok", "device_off" (an admin turned microphone access off for this PC), "all_apps_off" ("Microphone access"
      is off), "desktop_apps_off" ("Let desktop apps access your microphone" is off), or "unknown".
    """
    winreg = _winreg()
    if winreg is None:
        return "unknown"

    def value(hive, path) -> Optional[str]:
        try:
            with winreg.OpenKey(hive, path) as key:
                return str(winreg.QueryValueEx(key, "Value")[0])
        except OSError:
            return None

    if value(winreg.HKEY_LOCAL_MACHINE, CONSENT_MIC) == "Deny":
        return "device_off"
    if value(winreg.HKEY_CURRENT_USER, CONSENT_MIC) == "Deny":
        return "all_apps_off"
    if value(winreg.HKEY_CURRENT_USER, CONSENT_MIC + r"\NonPackaged") == "Deny":
        return "desktop_apps_off"
    return "ok"


def open_uri(uri: str) -> bool:
    """Opens a Settings page (ms-settings:...) or a web page with the default handler."""
    try:
        os.startfile(uri)  # type: ignore[attr-defined]
        return True
    except (OSError, AttributeError):
        return False
