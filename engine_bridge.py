"""
EngineBridge: the only door between the Fluent UI and the audio engine.

The bridge is a QObject that lives on its own thread (see ``start()``), owns ``engine.AudioEngineManager`` and the
Win32 hotkey listener, and talks to the UI exclusively through Qt signals and slots carrying plain Python values:

  UI -> bridge   slots such as ``apply_preset(str)`` and ``set_param(str, float)``; the window connects them to
                 pyqtSignals declared on the views. The views never import the engine or call the bridge.
  bridge -> UI   signals such as ``status_updated(dict)`` and ``params_changed(dict)``; the window connects them to
                 slots on the views (``update_status(data: dict)`` and friends).

The real-time audio callback never touches Qt: it overwrites a few plain numbers, and a ~30 Hz QTimer on the bridge
thread reads them (``AudioEngineManager.read_telemetry``) and emits ``status_updated``.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

import engine as eng
import presets as pr

TELEMETRY_INTERVAL_MS = 33      # ~30 Hz meters and latency readout
PARAM_FLUSH_INTERVAL_MS = 25    # slider drags are coalesced to at most ~40 snapshot rebuilds per second

BLOCK_SIZES = (128, 256, 512)
DEVICE_ROLES = ("input", "output", "monitor")
FLAG_NAMES = ("bypass_effects", "mic_muted", "hear_myself", "enable_onnx")
PARAM_NAMES = tuple(f for f in eng.VoiceParameters.__dataclass_fields__ if f != "name")

# What the UI shows for each preset: (one-line description, Fluent icon name). The DSP numbers live in
# engine.TACTICAL_PRESETS.
PRESET_META = {
    "Deep Shadow (Omen)": ("Sub-harmonic pitch and formant drop with a dark, roomy tail.", "QUIET_HOURS"),
    "Tactical Comms (Radio)": ("Tight 420 Hz to 2.9 kHz band-pass with an overdriven comms crunch.", "MEGAPHONE"),
    "Cybernetic Agent (KAY/O)": ("Phase-continuous 58 Hz ring modulator with a shifted vocal tract.", "ROBOT"),
    "Pitch Clean (Up / Helium-Lite)": ("Clean upward pitch and formant shift. No grit.", "UP"),
}
DEFAULT_HOTKEY_PREFIX = "Ctrl + Alt + "


def _split_preset_name(name: str):
    """'Deep Shadow (Omen)' -> ('Deep Shadow', 'Omen')."""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    return (m.group(1), m.group(2)) if m else (name, "")


def _summary(p: eng.VoiceParameters) -> str:
    """The few settings that make a custom preset sound the way it does, for its tile."""
    bits = []
    if abs(p.pitch_semitones) >= 0.05:
        bits.append(f"Pitch {p.pitch_semitones:+.1f} st")
    if abs(p.formant_semitones) >= 0.05:
        bits.append(f"Formant {p.formant_semitones:+.1f} st")
    if p.ring_mod_hz >= 0.5 and p.ring_mod_mix >= 0.005:
        bits.append(f"Ring mod {p.ring_mod_hz:.0f} Hz")
    if p.drive_db >= 0.05:
        bits.append(f"Drive {p.drive_db:.1f} dB")
    if p.reverb_mix >= 0.005:
        bits.append(f"Reverb {p.reverb_mix * 100:.0f}%")
    return ", ".join(bits[:3]) or "Plain voice, no effects."


class EngineBridge(QObject):
    # ---- bridge -> UI --------------------------------------------------------------------------------------
    ready = pyqtSignal()                  # initial state has been published
    catalog_ready = pyqtSignal(object)    # {"hotkeys": [...], "onnx_available": bool, "preset_rules": {...}, ...}
    presets_changed = pyqtSignal(object)  # {"presets": [tile data], "custom_names": [str]}: built-ins, then the player's own
    preset_changed = pyqtSignal(str)      # active preset name
    params_changed = pyqtSignal(object)   # {voice parameter: value}
    flags_changed = pyqtSignal(object)    # {bypass_effects, mic_muted, hear_myself, enable_onnx, monitor_volume}
    devices_changed = pyqtSignal(object)  # {"inputs": [(id, label)], "outputs": [...], "selected": {role: id}, ...}
    engine_state_changed = pyqtSignal(object)  # {"state": stopped|starting|running|error, "text": str, ...}
    status_updated = pyqtSignal(object)   # ~30 Hz: meters, DSP time, load, estimated round trip
    path_test_state = pyqtSignal(object)  # {"running": bool}
    path_test_finished = pyqtSignal(object)    # {"passed": bool, "steps": [{"status", "stage", "text"}]}
    onnx_changed = pyqtSignal(object)     # {"available", "enabled", "path", "message"}
    profile_changed = pyqtSignal(object)  # {"path": str, "autoloaded": bool}
    hotkeys_changed = pyqtSignal(object)  # {hotkey id: registered}
    notify = pyqtSignal(object)           # {"level": success|info|warning|error, "title": str, "text": str}

    # ---- internal (emitted from other threads, delivered on the bridge thread) -------------------------------
    _hotkey_pressed = pyqtSignal(int)
    _path_test_done = pyqtSignal(object)
    _shutdown_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._thread = QThread()
        self._thread.setObjectName("vcomms-engine-bridge")
        self.moveToThread(self._thread)
        self._thread.started.connect(self._initialize)
        self._hotkey_pressed.connect(self._on_hotkey)
        self._path_test_done.connect(self._on_path_test_done)
        self._shutdown_requested.connect(self._shutdown)

        self.config: eng.AppConfig = eng.AppConfig()
        self.engine: Optional[eng.AudioEngineManager] = None
        self._hotkeys: Optional[eng.Win32HotkeyListener] = None
        self._inputs: List = []
        self._outputs: List = []
        self._selected: Dict[str, Optional[int]] = {r: None for r in DEVICE_ROLES}
        self._pending: Dict[str, float] = {}
        self._path_test_running = False
        self._running_block_size = self.config.block_size
        self._profile_path = Path(eng.PROFILE_FILENAME)
        self._profile_autoloaded = False
        self._library = pr.PresetLibrary(self._profile_path.with_name(pr.LIBRARY_FILENAME))  # next to the profile
        self._flush_timer: Optional[QTimer] = None
        self._telemetry_timer: Optional[QTimer] = None

    # ======================================================================================================
    # lifecycle (call from the GUI thread)
    # ======================================================================================================

    def start(self) -> None:
        self._thread.start()

    def request_shutdown(self, timeout_ms: int = 5000) -> None:
        """Stops hotkeys and audio on the bridge thread and waits for it to finish."""
        if self._thread.isRunning():
            self._shutdown_requested.emit()
            self._thread.wait(timeout_ms)

    @pyqtSlot()
    def _initialize(self) -> None:
        cfg = self.config
        self._profile_autoloaded = self._profile_path.is_file() and eng.load_profile_into(cfg, self._profile_path)
        self.engine = eng.AudioEngineManager(cfg)
        self._library.load()
        # so a preset hotkey (or a click on a saved preset) never has to design filters mid-game
        eng.prewarm_filter_designs(cfg.sample_rate, extra=self._library.items.values())

        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(PARAM_FLUSH_INTERVAL_MS)
        self._flush_timer.timeout.connect(self._flush_params)
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(TELEMETRY_INTERVAL_MS)
        self._telemetry_timer.timeout.connect(self._tick)

        if self._profile_autoloaded:
            self.engine.dsp.update_parameters(cfg.current_params)
        elif cfg.active_preset in eng.TACTICAL_PRESETS:
            cfg.current_params = eng.preset_parameters(cfg.active_preset)
            self.engine.dsp.update_parameters(cfg.current_params)
        self._scan_devices()

        self._hotkeys = eng.Win32HotkeyListener(self._hotkey_pressed.emit)
        self._hotkeys.start()
        QTimer.singleShot(250, self._publish_hotkey_state)  # RegisterHotKey runs on the listener thread

        self.catalog_ready.emit(self._catalog())
        self._publish_everything()
        self._telemetry_timer.start()
        self.ready.emit()
        if self._library.load_error:
            self._set_damaged_library_aside()

    @pyqtSlot()
    def _shutdown(self) -> None:
        for timer in (self._flush_timer, self._telemetry_timer):
            if timer is not None:
                timer.stop()
        if self._hotkeys is not None:
            self._hotkeys.stop()
        if self.engine is not None:
            self.engine.stop_engine()
        QThread.currentThread().quit()

    # ======================================================================================================
    # payloads
    # ======================================================================================================

    def _preset_items(self) -> List[dict]:
        """Tile data: the four built-ins (Ctrl+Alt+F1..F4), then the player's own presets in the order they made them."""
        items = []
        for n, name in enumerate(eng.TACTICAL_PRESETS, start=1):
            title, tag = _split_preset_name(name)
            blurb, icon = PRESET_META.get(name, ("", "MICROPHONE"))
            items.append(
                {
                    "key": name,
                    "title": title,
                    "tag": tag,
                    "hotkey": f"{DEFAULT_HOTKEY_PREFIX}F{n}",
                    "blurb": blurb,
                    "icon": icon,
                    "custom": False,
                }
            )
        for name, params in self._library.items.items():
            items.append(
                {
                    "key": name,
                    "title": name,
                    "tag": "Custom",
                    "hotkey": "Custom",
                    "blurb": _summary(params),
                    "icon": "TAG",
                    "custom": True,
                }
            )
        return items

    def _presets_payload(self) -> dict:
        return {"presets": self._preset_items(), "custom_names": self._library.names()}

    def _params_for(self, name: str) -> Optional[eng.VoiceParameters]:
        """A fresh copy of a built-in or custom preset's parameters, or None when there is no such preset."""
        if name in eng.TACTICAL_PRESETS:
            return eng.preset_parameters(name)
        return self._library.get(name)

    def _catalog(self) -> dict:
        hotkeys = []
        for n, name in enumerate(eng.TACTICAL_PRESETS, start=1):
            title, _tag = _split_preset_name(name)
            blurb, icon = PRESET_META.get(name, ("", "MICROPHONE"))
            hotkeys.append(
                {
                    "id": getattr(eng.Win32HotkeyListener, f"HK_PRESET_{n}", None),
                    "keys": f"{DEFAULT_HOTKEY_PREFIX}F{n}",
                    "action": f"Preset: {title}",
                    "detail": blurb,
                    "icon": icon,
                }
            )
        hotkeys.append(
            {
                "id": eng.Win32HotkeyListener.HK_BYPASS,
                "keys": f"{DEFAULT_HOTKEY_PREFIX}F9",
                "action": "Bypass toggle",
                "detail": "Instant passthrough to your raw natural voice.",
                "icon": "SPEED_OFF",
            }
        )
        hotkeys.append(
            {
                "id": eng.Win32HotkeyListener.HK_MUTE,
                "keys": f"{DEFAULT_HOTKEY_PREFIX}F10",
                "action": "Hardware mute toggle",
                "detail": "Zero-fills the output buffer instantly.",
                "icon": "MUTE",
            }
        )
        return {
            "hotkeys": hotkeys,
            "block_sizes": [
                {"samples": b, "ms": b / self.config.sample_rate * 1000.0} for b in BLOCK_SIZES
            ],
            "onnx_available": eng.ONNX_AVAILABLE,
            "profile_file": eng.PROFILE_FILENAME,
            "preset_file": pr.LIBRARY_FILENAME,
            "sample_rate": self.config.sample_rate,
            # what the naming dialogs need to validate a name as the player types it
            "preset_rules": {
                "reserved": sorted(pr.reserved_names()),
                "name_max": pr.NAME_MAX,
                "max_custom": pr.MAX_CUSTOM_PRESETS,
            },
        }

    def _flags_payload(self) -> dict:
        cfg = self.config
        return {
            "bypass_effects": cfg.bypass_effects,
            "mic_muted": cfg.mic_muted,
            "hear_myself": cfg.hear_myself,
            "enable_onnx": cfg.enable_onnx,
            "monitor_volume": cfg.monitor_volume,
        }

    def _params_payload(self) -> dict:
        d = asdict(self.config.current_params)
        d.pop("name", None)
        return d

    def _devices_payload(self) -> dict:
        return {
            "inputs": list(self._inputs),
            "outputs": list(self._outputs),
            "selected": dict(self._selected),
            "block_size": self.config.block_size,
            "block_sizes": list(BLOCK_SIZES),
        }

    def _block_size(self) -> int:
        """The buffer size the audio streams actually use (a change in the UI only applies at the next start)."""
        return self._running_block_size if self.engine and self.engine.is_running else self.config.block_size

    def _engine_payload(self) -> dict:
        cfg = self.config
        running = bool(self.engine and self.engine.is_running)
        error = (self.engine.last_error if self.engine else "") or ""
        block = self._block_size()
        return {
            "state": "running" if running else ("error" if error else "stopped"),
            "running": running,
            "text": self.engine.status_text if self.engine else "STOPPED",
            "error": error,
            "block_size": block,
            "sample_rate": cfg.sample_rate,
            "block_ms": block / cfg.sample_rate * 1000.0,
        }

    def _onnx_payload(self, message: str = "") -> dict:
        cfg = self.config
        return {
            "available": eng.ONNX_AVAILABLE,
            "enabled": cfg.enable_onnx,
            "path": cfg.onnx_model_path,
            "message": message,
        }

    def _publish_everything(self) -> None:
        self.presets_changed.emit(self._presets_payload())
        self.preset_changed.emit(self.config.active_preset)
        self.params_changed.emit(self._params_payload())
        self.flags_changed.emit(self._flags_payload())
        self.devices_changed.emit(self._devices_payload())
        self.engine_state_changed.emit(self._engine_payload())
        self.onnx_changed.emit(self._onnx_payload())
        self.profile_changed.emit(
            {"path": str(self._profile_path.resolve()) if self._profile_autoloaded else "",
             "autoloaded": self._profile_autoloaded}
        )

    def _notify(self, level: str, title: str, text: str = "") -> None:
        self.notify.emit({"level": level, "title": title, "text": text})

    # ======================================================================================================
    # devices
    # ======================================================================================================

    def _scan_devices(self, refresh: bool = False) -> None:
        if refresh and not (self.engine and self.engine.is_running) and eng.SOUNDDEVICE_AVAILABLE:
            # PortAudio only lists devices it saw at start-up; re-initialising it picks up newly plugged ones.
            try:
                eng.sd._terminate()
                eng.sd._initialize()
            except Exception:
                try:
                    eng.sd._initialize()  # never leave PortAudio shut down
                except Exception:
                    pass
        self._inputs, self._outputs = eng.AudioEngineManager.list_audio_devices()
        in_t, out_t, mon_t = eng.pick_device_targets(self.config, self._inputs, self._outputs)
        self._selected["input"] = self._inputs[in_t][0] if self._inputs else None
        self._selected["output"] = self._outputs[out_t][0] if self._outputs else None
        self._selected["monitor"] = self._outputs[mon_t][0] if self._outputs else None

    def _label_for(self, role: str) -> str:
        devices = self._inputs if role == "input" else self._outputs
        idx = self._selected.get(role)
        return next((label for i, label in devices if i == idx), "")

    def _sync_device_names(self) -> None:
        cfg = self.config
        cfg.input_device_name = self._label_for("input")
        cfg.output_device_name = self._label_for("output")
        cfg.monitor_device_name = self._label_for("monitor")

    @pyqtSlot(str, int)
    def select_device(self, role: str, device_index: int) -> None:
        devices = self._inputs if role == "input" else self._outputs
        if role not in DEVICE_ROLES or all(i != device_index for i, _ in devices):
            return
        self._selected[role] = device_index
        setattr(self.config, f"{role}_device_name", self._label_for(role))

    @pyqtSlot(int)
    def set_block_size(self, samples: int) -> None:
        if samples in BLOCK_SIZES:
            self.config.block_size = int(samples)

    @pyqtSlot()
    def rescan_devices(self) -> None:
        if self.engine and self.engine.is_running or self._path_test_running:
            self._notify("warning", "Stop first", "Stop the audio engine before rescanning devices.")
            return
        self._scan_devices(refresh=True)
        self.devices_changed.emit(self._devices_payload())
        self._notify(
            "info",
            "Devices rescanned",
            f"{len(self._inputs)} inputs and {len(self._outputs)} outputs found.",
        )

    # ======================================================================================================
    # engine
    # ======================================================================================================

    @pyqtSlot()
    def toggle_engine(self) -> None:
        if self._path_test_running:
            self._notify("warning", "Test in progress", "Wait for the audio path test to finish.")
            return
        if self.engine.is_running:
            self.engine.stop_engine()
            self.engine_state_changed.emit(self._engine_payload())
            return

        in_idx, out_idx = self._selected["input"], self._selected["output"]
        if in_idx is None or out_idx is None:
            self._notify("error", "Pick your devices", "Please select valid input and output devices.")
            return

        self._sync_device_names()
        self.engine_state_changed.emit({**self._engine_payload(), "state": "starting", "text": "Starting…"})
        ok = self.engine.start_engine(in_idx, out_idx, self._selected["monitor"])
        self._running_block_size = self.config.block_size  # the buffer size only changes at the next start
        self.engine_state_changed.emit(self._engine_payload())
        if not ok:
            self._notify("error", "Couldn't start the audio engine", self.engine.last_error or self.engine.status_text)
        else:
            self._check_monitor()

    @pyqtSlot(bool)
    def set_telemetry_active(self, active: bool) -> None:
        """The window pauses the meters while it is hidden in the tray (hotkeys and audio keep running)."""
        if self._telemetry_timer is None:
            return
        if active and not self._telemetry_timer.isActive():
            self._telemetry_timer.start()
        elif not active:
            self._telemetry_timer.stop()

    def _tick(self) -> None:
        cfg = self.config
        t = self.engine.read_telemetry()
        block_size = self._block_size()
        block_ms = block_size / cfg.sample_rate * 1000.0
        peak = t["peak_dsp_ms"] or t["dsp_ms"]
        self.status_updated.emit(
            {
                "running": self.engine.is_running,
                "in_db": t["in_db"],
                "out_db": t["out_db"],
                "dsp_ms": t["dsp_ms"],
                "peak_dsp_ms": peak,
                "block_ms": block_ms,
                "block_size": block_size,
                "load": peak / block_ms if block_ms else 0.0,
                "roundtrip_ms": block_ms * 2.0 + t["dsp_ms"] + 6.0,
                "xruns": int(t["xruns"]),
                "tag": "MUTED" if cfg.mic_muted else ("BYPASS" if cfg.bypass_effects else cfg.active_preset),
            }
        )

    # ======================================================================================================
    # voice controls
    # ======================================================================================================

    @pyqtSlot(str)
    def apply_preset(self, name: str) -> None:
        params = self._params_for(name)
        if params is None:
            return
        self._pending.clear()
        cfg = self.config
        cfg.active_preset = name
        cfg.current_params = params
        self.engine.dsp.update_parameters(cfg.current_params)
        self.preset_changed.emit(name)
        self.params_changed.emit(self._params_payload())

    @pyqtSlot()
    def reset_to_preset(self) -> None:
        if self._params_for(self.config.active_preset) is None:
            self._notify("info", "Nothing to reset to", "The active preset is no longer available. Pick another one.")
            return
        self.apply_preset(self.config.active_preset)

    @pyqtSlot(str, float)
    def set_param(self, name: str, value: float) -> None:
        """Slider moves arrive far faster than filters should be redesigned, so they are coalesced."""
        if name not in PARAM_NAMES:
            return
        self._pending[name] = float(value)
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def _flush_params(self) -> None:
        if not self._pending:
            return
        d = asdict(self.config.current_params)
        d.update(self._pending)
        self._pending.clear()
        self.config.current_params = eng.VoiceParameters(**d)
        self.engine.dsp.update_parameters(self.config.current_params)

    @pyqtSlot(str, bool)
    def set_flag(self, name: str, value: bool) -> None:
        if name not in FLAG_NAMES:
            return
        setattr(self.config, name, bool(value))
        self.flags_changed.emit(self._flags_payload())
        if name == "hear_myself" and value:
            self._check_monitor()

    def _check_monitor(self) -> None:
        """Hear Myself plays through a second stream, which only opens on a device other than the cable output."""
        if not self.config.hear_myself:
            return
        if self._selected["monitor"] is None or self._selected["monitor"] == self._selected["output"]:
            self._notify(
                "warning",
                "Hear Myself needs another device",
                "Pick headphones that differ from the Virtual Cable Output, then restart the audio engine.",
            )
        elif self.engine.is_running and self.engine.monitor_stream is None:
            self._notify("warning", "No monitor stream is open", "Restart the audio engine to open the headphone monitor.")

    @pyqtSlot(float)
    def set_monitor_volume(self, volume: float) -> None:
        self.config.monitor_volume = min(max(float(volume), 0.0), 1.0)

    # ======================================================================================================
    # the player's own presets (a JSON library next to the profile) and preset files
    # ======================================================================================================

    def _library_failed(self, title: str, exc: Exception) -> None:
        if isinstance(exc, pr.PresetError):
            self._notify("error", title, str(exc))
        else:
            reason = getattr(exc, "strerror", None) or str(exc)
            self._notify("error", title, f"{pr.LIBRARY_FILENAME} could not be written ({reason}).")

    def _set_damaged_library_aside(self) -> None:
        """A preset file that cannot be read is kept as ``custom_presets.json.bad`` so the next save cannot destroy it."""
        path = self._library.path
        aside = path.with_name(path.name + ".bad")
        try:
            shutil.copy2(path, aside)
            kept = f"A copy was kept as {aside.name}."
        except OSError:
            kept = ""
        self._notify("warning", "Custom presets could not be loaded", f"{self._library.load_error}. {kept}".strip())

    @pyqtSlot(str)
    def save_preset(self, name: str) -> None:
        """Saves the current voice settings (sliders included) under ``name`` and selects it."""
        self._flush_params()
        try:
            saved = self._library.add(name, self.config.current_params)
        except (pr.PresetError, OSError) as exc:
            self._library_failed("Couldn't save the preset", exc)
            return
        self.engine.dsp.prepare_parameters(self._library.items[saved])
        self.config.active_preset = saved
        self.presets_changed.emit(self._presets_payload())
        self.preset_changed.emit(saved)
        self._notify("success", "Preset saved", f"{saved} is on the Live page.")

    @pyqtSlot(str, str)
    def rename_preset(self, old: str, new: str) -> None:
        try:
            renamed = self._library.rename(old, new)
        except (pr.PresetError, OSError) as exc:
            self._library_failed("Couldn't rename the preset", exc)
            return
        if self.config.active_preset == old:
            self.config.active_preset = renamed
        self.presets_changed.emit(self._presets_payload())
        self.preset_changed.emit(self.config.active_preset)
        self._notify("success", "Preset renamed", f"{old} is now {renamed}.")

    @pyqtSlot(str)
    def delete_preset(self, name: str) -> None:
        was_active = self.config.active_preset == name
        try:
            self._library.delete(name)
        except (pr.PresetError, OSError) as exc:
            self._library_failed("Couldn't delete the preset", exc)
            return
        self.presets_changed.emit(self._presets_payload())
        fallback = next(iter(eng.TACTICAL_PRESETS))
        if was_active:
            self.apply_preset(fallback)  # the voice would otherwise keep a preset that no longer exists
        title = _split_preset_name(fallback)[0]
        self._notify("success", "Preset deleted", f"{name} was removed." + (f" Switched to {title}." if was_active else ""))

    def _write_export(self, name: str, params: eng.VoiceParameters, path: str) -> None:
        try:
            Path(path).write_text(pr.export_text(name, params), encoding="utf-8")
        except OSError as exc:
            self._notify("error", "Couldn't export the preset", f"{Path(path).name} could not be written ({exc.strerror or exc}).")
            return
        self._notify("success", "Preset exported", Path(path).name)

    @pyqtSlot(str, str)
    def export_preset(self, key: str, path: str) -> None:
        params = self._params_for(key)
        if params is None:
            self._notify("error", "Couldn't export the preset", "That preset no longer exists.")
            return
        self._write_export(key, params, path)

    @pyqtSlot(str)
    def export_current(self, path: str) -> None:
        """The voice exactly as it sounds now, sliders included, whichever preset it started from."""
        self._flush_params()
        cfg = self.config
        name = cfg.active_preset
        if name not in self._library.items:
            name = f"My {_split_preset_name(name)[0]}"
        self._write_export(name, cfg.current_params, path)

    @pyqtSlot(str)
    def import_preset(self, path: str) -> None:
        """Reads a preset file as data only. A name that is taken (or reserved) gets a number: 'Name (2)'."""
        try:
            wanted, params = pr.parse_preset_file(Path(path))
            saved = self._library.add(pr.unique_name(wanted, self._library.names()), params)
        except (pr.PresetError, OSError) as exc:
            self._library_failed("Couldn't import the preset", exc)
            return
        self.engine.dsp.prepare_parameters(self._library.items[saved])
        self.presets_changed.emit(self._presets_payload())
        self.apply_preset(saved)
        renamed = " (that name was taken)" if saved != pr.normalize_name(wanted) else ""
        self._notify("success", "Preset imported", f"{saved} is selected{renamed}.")

    @pyqtSlot(bool)
    def reset_all(self, delete_presets: bool) -> None:
        """
        Everything back to the first-launch state: effects, switches, devices (auto-detected again), buffer size and
        the saved startup profile. The player's presets are kept unless ``delete_presets``. A running engine keeps
        its streams until it is restarted, as when the device boxes are changed by hand.
        """
        if self._path_test_running:
            self._notify("warning", "Test in progress", "Wait for the audio path test to finish, then reset.")
            return
        cfg = self.config
        running = bool(self.engine.is_running)
        self._pending.clear()
        defaults = eng.AppConfig()
        for f in dataclasses.fields(defaults):          # in place: the audio engine holds this same object
            if f.name != "current_params":
                setattr(cfg, f.name, getattr(defaults, f.name))
        cfg.current_params = eng.preset_parameters(cfg.active_preset)
        self.engine.dsp.update_parameters(cfg.current_params)
        self.engine.dsp.onnx_streamer.unload()

        problems = []
        if delete_presets and self._library.items:
            try:
                self._library.clear()
            except OSError as exc:
                problems.append(f"{pr.LIBRARY_FILENAME} could not be updated ({exc.strerror or exc})")
        self._scan_devices(refresh=not running)
        if self._profile_path.is_file():                # the file that is loaded on every start
            try:
                eng.save_profile(cfg, self._profile_path)
            except OSError as exc:
                problems.append(f"{self._profile_path.name} could not be rewritten ({exc.strerror or exc})")

        self._publish_everything()
        if problems:
            self._notify("warning", "Reset, with a problem", "; ".join(problems) + ".")
            return
        kept = "Custom presets were deleted." if delete_presets else "Your custom presets were kept."
        later = " Devices and buffer size apply when you restart the audio engine." if running else ""
        self._notify("success", "Settings reset", f"Everything is back to its defaults. {kept}{later}")

    # ======================================================================================================
    # hotkeys
    # ======================================================================================================

    @pyqtSlot(int)
    def _on_hotkey(self, hk_id: int) -> None:
        presets = list(eng.TACTICAL_PRESETS.keys())
        listener = eng.Win32HotkeyListener
        slot = {
            listener.HK_PRESET_1: 0,
            listener.HK_PRESET_2: 1,
            listener.HK_PRESET_3: 2,
            listener.HK_PRESET_4: 3,
        }.get(hk_id)
        if slot is not None:
            if slot < len(presets):
                self.apply_preset(presets[slot])
        elif hk_id == listener.HK_BYPASS:
            self.set_flag("bypass_effects", not self.config.bypass_effects)
        elif hk_id == listener.HK_MUTE:
            self.set_flag("mic_muted", not self.config.mic_muted)

    def _publish_hotkey_state(self) -> None:
        if self._hotkeys is None:
            return
        registered = dict(self._hotkeys.registered)
        self.hotkeys_changed.emit(registered)
        taken = sum(1 for ok in registered.values() if not ok)
        if taken:
            self._notify(
                "warning",
                "Some hotkeys are unavailable",
                f"{taken} of {len(registered)} are used by another app, maybe a second V-COMMS. See Settings.",
            )

    # ======================================================================================================
    # audio path test
    # ======================================================================================================

    @pyqtSlot()
    def run_path_test(self) -> None:
        if self._path_test_running:
            return
        if self.engine.is_running:
            self._notify("warning", "Stop the audio engine first", "Stop the audio engine before running the test.")
            return
        in_idx, out_idx = self._selected["input"], self._selected["output"]
        if in_idx is None or out_idx is None:
            self._notify("error", "Pick your devices", "Please select valid input and output devices.")
            return

        cfg = self.config
        params = eng.VoiceParameters(**asdict(cfg.current_params))
        sample_rate, block_size = cfg.sample_rate, cfg.block_size
        self._path_test_running = True
        self.path_test_state.emit({"running": True})

        def worker() -> None:
            try:
                passed, lines = eng.run_audio_path_test(in_idx, out_idx, params, sample_rate, block_size)
            except Exception as exc:
                passed, lines = False, [f"[FAIL] Audio error: {exc}"]
            self._path_test_done.emit({"passed": passed, "lines": lines})

        threading.Thread(target=worker, daemon=True, name="vcomms-path-test").start()

    @pyqtSlot(object)
    def _on_path_test_done(self, result: dict) -> None:
        self._path_test_running = False
        steps = []
        for line in result["lines"]:
            m = re.match(r"^\[(PASS|FAIL|SKIP)\]\s*([^:]+):\s*(.*)$", line)
            if m:
                steps.append({"status": m.group(1).lower(), "stage": m.group(2).strip(), "text": m.group(3)})
            else:
                steps.append({"status": "fail" if "FAIL" in line else "info", "stage": "", "text": line})
        self.path_test_state.emit({"running": False})
        self.path_test_finished.emit({"passed": result["passed"], "steps": steps})

    # ======================================================================================================
    # profiles & ONNX
    # ======================================================================================================

    @pyqtSlot(str)
    def save_profile(self, path: str) -> None:
        self._sync_device_names()
        try:
            eng.save_profile(self.config, Path(path))
        except Exception as exc:
            self._notify("error", "Couldn't save the profile", str(exc))
            return
        self._notify("success", "Profile saved", Path(path).name)

    @pyqtSlot(str)
    def load_profile(self, path: str) -> None:
        if not eng.load_profile_into(self.config, Path(path)):
            self._notify("error", "Couldn't load the profile", f"{Path(path).name} is not a valid V-COMMS profile.")
            return
        self._pending.clear()
        self._scan_devices()
        self.engine.dsp.update_parameters(self.config.current_params)
        self._publish_everything()
        self.profile_changed.emit({"path": str(Path(path).resolve()), "autoloaded": False})
        self._notify("success", "Profile loaded", Path(path).name)

    @pyqtSlot(str)
    def load_onnx_model(self, path: str) -> None:
        self.config.onnx_model_path = path
        ok, message = self.engine.dsp.onnx_streamer.load_model(path)
        self.onnx_changed.emit(self._onnx_payload(message))
        self._notify("success" if ok else "error", "ONNX Runtime", message)
