"""
Plain-text description of this PC and the current audio setup. It is shown to the user in full before a problem
report is sent, so keep it short and free of anything personal: no audio, no files, and the Windows user name in
paths is replaced with %USERNAME%.
"""

from __future__ import annotations

import getpass
import os
import platform
import re
import sys
import sysconfig
from pathlib import Path
from typing import Dict, List

import applog
from appinfo import APP_NAME, APP_VERSION

LOG_LINES_IN_REPORT = 50


def scrub(text: str) -> str:
    """Replaces the Windows user name in any path (C:\\Users\\<name>\\..., /Users/<name>/...) with %USERNAME%."""
    names = set()
    for candidate in (os.environ.get("USERNAME"), _safe_getuser(), Path.home().name):
        if candidate and len(candidate) >= 2:
            names.add(candidate)
    for name in sorted(names, key=len, reverse=True):
        pattern = re.compile(r"(?<=[\\/])" + re.escape(name) + r"(?=[\\/]|\s|$|['\"])", re.IGNORECASE)
        text = pattern.sub("%USERNAME%", text)
    return text


def _safe_getuser() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return ""


def _registry_value(path: str, name: str):
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            return winreg.QueryValueEx(key, name)[0]
    except Exception:
        return None


def windows_description() -> str:
    if sys.platform != "win32":
        return platform.platform()
    v = sys.getwindowsversion()
    name = "Windows 11" if v.build >= 22000 else "Windows 10"
    key = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    display = _registry_value(key, "DisplayVersion")
    revision = _registry_value(key, "UBR")
    build = f"build {v.build}" + (f".{revision}" if revision is not None else "")
    return f"{name}{' ' + str(display) if display else ''} ({build})"


def cpu_description() -> str:
    name = _registry_value(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString")
    name = " ".join(str(name).split()) if name else (platform.processor() or "unknown CPU")
    return f"{name} ({os.cpu_count() or '?'} logical cores)"


def architecture_description() -> str:
    """What the PC is versus what this Python was built for. An x64 Python on an ARM64 PC runs emulated."""
    host = platform.machine() or "unknown"
    build = sysconfig.get_platform()
    emulated = host.upper() == "ARM64" and build.lower().endswith("amd64")
    return f"{host} PC, Python build {build}" + (", x64 code emulated on ARM64" if emulated else "")


def build_diagnostics(snapshot: Dict, log_lines: int = LOG_LINES_IN_REPORT) -> str:
    """
    ``snapshot`` is assembled by the EngineBridge from the live engine: devices, sample rate, block size, preset,
    DSP statistics and the Lite mode state. Nothing else is collected.
    """
    s = snapshot
    stats = s.get("engine", {})
    on_off = lambda flag: "on" if flag else "off"
    flags = s.get("flags", {})
    block = s.get("block_size", 0)
    rate = s.get("sample_rate", 0) or 1
    lines: List[str] = [
        f"{APP_NAME} {APP_VERSION}",
        f"Windows: {windows_description()}",
        f"CPU: {cpu_description()}",
        f"Architecture: {architecture_description()}",
        f"Python {platform.python_version()}  |  {s.get('qt_versions', 'Qt: unknown')}",
        f"Microphone: {s.get('input', '-') or '-'}",
        f"Cable output: {s.get('output', '-') or '-'}",
        f"Headphone monitor: {s.get('monitor', '-') or '-'}",
        f"Sample rate: {rate} Hz  |  Block size: {block} samples ({block / rate * 1000.0:.1f} ms)",
        f"Preset: {s.get('preset', '-')}  |  Bypass: {on_off(flags.get('bypass_effects'))}  |  "
        f"Muted: {on_off(flags.get('mic_muted'))}  |  Hear Myself: {on_off(flags.get('hear_myself'))}  |  "
        f"AI model: {on_off(flags.get('enable_onnx'))}",
        f"Lite mode: {on_off(s.get('lite_mode'))}  |  Stability buffer: {on_off(s.get('stability_buffer'))}",
    ]
    if stats.get("blocks"):
        lines.append(
            f"Audio engine: {'running' if stats.get('running') else 'stopped'}  |  "
            f"DSP time per block: mean {stats['mean_dsp_ms']:.2f} ms, worst {stats['worst_dsp_ms']:.2f} ms "
            f"over {stats['blocks']} blocks  |  dropouts: {stats.get('xruns', 0)}"
        )
    else:
        lines.append("Audio engine: not started since launch (no DSP timing yet)")
    ai = s.get("ai_model") or {}
    if ai.get("loaded") or ai.get("errors"):
        line = f"AI model: {'loaded (' + ai.get('kind', '') + ')' if ai.get('loaded') else 'not loaded'}"
        if ai.get("errors"):
            line += f"  |  {ai['errors']} conversion errors, last: {str(ai.get('last_error', ''))[:200]}"
        lines.append(line)

    log = applog.tail(log_lines)
    lines.append("")
    lines.append(f"Last {len(log)} log lines:" if log else "No log file yet.")
    lines.extend(log)
    return scrub("\n".join(lines))
