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
import itertools
import re
import shutil
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

import appdata
import applog
import diagnostics
import engine as eng
import hotkeys as hk
import presets as pr
import reporting

TELEMETRY_INTERVAL_MS = 33      # ~30 Hz meters and latency readout
LITE_TELEMETRY_INTERVAL_MS = 100  # Lite mode: 10 Hz is plenty for the numbers and bars, and a third of the painting
PARAM_FLUSH_INTERVAL_MS = 25    # slider drags are coalesced to at most ~40 snapshot rebuilds per second
HOTKEY_CAPTURE_TIMEOUT_MS = 15000  # safety net: hotkeys are switched off while a new one is being picked, never for longer

BLOCK_SIZES = (128, 256, 512)
DEVICE_ROLES = ("input", "output", "monitor")
FLAG_NAMES = ("bypass_effects", "mic_muted", "hear_myself", "enable_onnx", "lite_mode", "stability_buffer")
PARAM_NAMES = tuple(f for f in eng.VoiceParameters.__dataclass_fields__ if f != "name")

# What the UI shows for each preset: (one-line description, Fluent icon name). The DSP numbers live in
# engine.TACTICAL_PRESETS.
PRESET_META = {
    "Deep Shadow (Omen)": ("Sub-harmonic pitch and formant drop with a dark, roomy tail.", "QUIET_HOURS"),
    "Tactical Comms (Radio)": ("Tight 420 Hz to 2.9 kHz band-pass with an overdriven comms crunch.", "MEGAPHONE"),
    "Cybernetic Agent (KAY/O)": ("Phase-continuous 58 Hz ring modulator with a shifted vocal tract.", "ROBOT"),
    "Pitch Clean (Up / Helium-Lite)": ("Clean upward pitch and formant shift. No grit.", "UP"),
    "Female Natural (Clear)": ("Everyday female pitch with a clean, slightly warm top.", "PEOPLE"),
    "Female Bright (Young)": ("Higher and brighter, with a lifted vocal tract.", "BRIGHTNESS"),
    "Female Husky (Low)": ("Low alto with a dark, breathy grain. Closest to real female formants.", "MICROPHONE"),
    "Female Cute (Airy)": ("High, light and airy, with a touch of room.", "HEART"),
}
FEMALE_PRESETS = tuple(name for name in eng.TACTICAL_PRESETS if name.startswith("Female "))

# The Settings rows of the two switch hotkeys (the four preset hotkeys are described by the presets they select).
SWITCH_HOTKEYS = {
    "bypass": ("Bypass toggle", "Instant passthrough to your raw natural voice.", "SPEED_OFF"),
    "mute": ("Hardware mute toggle", "Zero-fills the output buffer instantly.", "MUTE"),
}


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
    hubert_download_requested = pyqtSignal(str)   # path of model waiting for HuBERT base model
    profile_changed = pyqtSignal(object)  # {"path": str, "autoloaded": bool}
    hotkeys_changed = pyqtSignal(object)  # {"items": [{id, text, bound, registered, conflict}], "all_default": bool}
    hotkey_capture_ready = pyqtSignal(str)  # action id: every hotkey is switched off, so the page can listen for keys
    notify = pyqtSignal(object)           # {"level": success|info|warning|error, "title": str, "text": str}
    report_ready = pyqtSignal(object)     # {"diagnostics": str, "wait_s": int, "sending": bool}: the form can be opened
    report_finished = pyqtSignal(object)  # reporting.submit() result: sent | fallback | rate_limited

    # ---- internal (emitted from other threads, delivered on the bridge thread) -------------------------------
    _hotkey_pressed = pyqtSignal(int)
    _hotkey_result = pyqtSignal(object)   # the listener thread's answer to apply / suspend / resume
    _path_test_done = pyqtSignal(object)
    _report_done = pyqtSignal(object)
    _shutdown_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._thread = QThread()
        self._thread.setObjectName("vcomms-engine-bridge")
        self.moveToThread(self._thread)
        self._thread.started.connect(self._initialize)
        self._hotkey_pressed.connect(self._on_hotkey)
        self._hotkey_result.connect(self._on_hotkey_result)
        self._path_test_done.connect(self._on_path_test_done)
        self._report_done.connect(self._on_report_done)
        self._shutdown_requested.connect(self._shutdown)

        self.config: eng.AppConfig = eng.AppConfig()
        self.engine: Optional[eng.AudioEngineManager] = None
        self._hotkeys: Optional[eng.Win32HotkeyListener] = None
        self._hk_registered: Dict[str, Optional[bool]] = {}   # action id -> does Windows accept its combination?
        self._hk_conflicts: Dict[str, str] = {}               # action id -> why the last change was refused
        self._hk_pending: Dict[int, dict] = {}                # command token -> what to do with the answer
        self._hk_tokens = itertools.count(1)
        self._hk_file_unreadable = False                      # hotkeys.json exists but is not a hotkeys file
        self._hk_suspended = False                            # a hotkey is being picked: nothing is registered
        self._hk_baseline: Dict[str, str] = hk.default_bindings()  # what is on disk (or was loaded at start)
        self._hk_watchdog: Optional[QTimer] = None
        self._inputs: List = []
        self._outputs: List = []
        self._selected: Dict[str, Optional[int]] = {r: None for r in DEVICE_ROLES}
        self._pending: Dict[str, float] = {}
        self._path_test_running = False
        self._running_block_size = self.config.block_size
        self._profile_path = appdata.profile_path()  # %APPDATA%\V-COMMS; presets sit next to it
        self._profile_autoloaded = False
        self._library = pr.PresetLibrary(self._profile_path.with_name(pr.LIBRARY_FILENAME))  # next to the profile
        self._report_limiter = reporting.RateLimiter()
        self._report_sending = False
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
        saved = hk.load_store()
        self._hk_file_unreadable = saved is None and hk.store_path().exists()
        if saved is not None:
            cfg.hotkeys = saved  # the player's last choice in Settings wins over the hotkeys inside the startup profile
        self._hk_baseline = dict(cfg.hotkeys)
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
        self._apply_lite()

        if self._profile_autoloaded:
            self.engine.dsp.update_parameters(cfg.current_params)
        elif cfg.active_preset in eng.TACTICAL_PRESETS:
            cfg.current_params = eng.preset_parameters(cfg.active_preset)
            self.engine.dsp.update_parameters(cfg.current_params)
        self._scan_devices()

        self._hk_watchdog = QTimer(self)
        self._hk_watchdog.setSingleShot(True)
        self._hk_watchdog.setInterval(HOTKEY_CAPTURE_TIMEOUT_MS)
        self._hk_watchdog.timeout.connect(self._hotkey_capture_timed_out)
        # RegisterHotKey runs on the listener thread; the answer arrives as _hotkey_result
        self._hotkeys = eng.Win32HotkeyListener(self._hotkey_pressed.emit, hk.to_table(cfg.hotkeys), self._hotkey_result.emit)
        self._hotkeys.start()

        self.catalog_ready.emit(self._catalog())
        self._publish_everything()
        self._telemetry_timer.start()
        self.ready.emit()
        if self._library.load_error:
            self._set_damaged_library_aside()
        if self._hk_file_unreadable:
            applog.get_logger().warning("%s could not be read; the hotkeys come from the profile or the defaults", hk.store_path())
            self._notify(
                "warning",
                "Your saved hotkeys could not be read",
                f"{hk.store_path().name} is damaged, so the hotkeys from the profile or the defaults are used. Changing a hotkey saves a new file.",
            )

    @pyqtSlot()
    def _shutdown(self) -> None:
        for timer in (self._flush_timer, self._telemetry_timer, self._hk_watchdog):
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

    def _preset_hotkey(self, n: int):
        """(action id, chip text) of the hotkey that selects built-in preset number ``n``; none for presets past the
        fourth, which have no hotkey."""
        action = f"preset_{n}"
        if action not in hk.ACTION_BY_ID:
            return None, ""
        serial = self.config.hotkeys.get(action, "")
        return action, (hk.combo_text(serial) if serial else "No hotkey")

    def _preset_items(self) -> List[dict]:
        """Tile data: the built-ins (the first four carry the hotkeys set in Settings, Ctrl+Alt+F1..F4 at first, then
        the female voices), then the player's own presets in the order they made them."""
        items = []
        for n, name in enumerate(eng.TACTICAL_PRESETS, start=1):
            title, tag = _split_preset_name(name)
            blurb, icon = PRESET_META.get(name, ("", "MICROPHONE"))
            action, chip = self._preset_hotkey(n)
            female = name in FEMALE_PRESETS
            items.append(
                {
                    "key": name,
                    "title": title,
                    "tag": tag,
                    "hotkey": chip if action else ("Female" if female else ""),
                    "hotkey_action": action,
                    "blurb": blurb,
                    "icon": icon,
                    "custom": False,
                    "group": "female" if female else "tactical",
                }
            )
        for name, params in self._library.items.items():
            items.append(
                {
                    "key": name,
                    "title": name,
                    "tag": "Custom",
                    "hotkey": "Custom",
                    "hotkey_action": None,
                    "blurb": _summary(params),
                    "icon": "TAG",
                    "custom": True,
                    "group": "custom",
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

    def _action_label(self, action_id: str) -> str:
        """How a hotkey is named on screen: the preset it selects, or what it toggles."""
        if action_id in SWITCH_HOTKEYS:
            return SWITCH_HOTKEYS[action_id][0]
        if action_id.startswith("preset_"):
            n = int(action_id.split("_")[1])
            names = list(eng.TACTICAL_PRESETS)
            if n <= len(names):
                return f"Preset: {_split_preset_name(names[n - 1])[0]}"
        action = hk.ACTION_BY_ID.get(action_id)
        return action.label if action else action_id

    def _hotkey_rows(self) -> List[dict]:
        """One Settings row per hotkey action, in the order of ``hotkeys.ACTIONS``."""
        names = list(eng.TACTICAL_PRESETS)
        rows = []
        for action in hk.ACTIONS:
            if action.id in SWITCH_HOTKEYS:
                label, detail, icon = SWITCH_HOTKEYS[action.id]
            else:
                n = int(action.id.split("_")[1])
                if n > len(names):
                    continue
                blurb, icon = PRESET_META.get(names[n - 1], ("", "MICROPHONE"))
                label, detail = self._action_label(action.id), blurb
            rows.append(
                {
                    "id": action.id,
                    "action": label,
                    "detail": detail,
                    "icon": icon,
                    "default_text": hk.combo_text(action.default),
                }
            )
        return rows

    def _catalog(self) -> dict:
        return {
            "hotkeys": self._hotkey_rows(),
            "block_sizes": [
                {"samples": b, "ms": b / self.config.sample_rate * 1000.0} for b in BLOCK_SIZES
            ],
            "onnx_available": eng.ONNX_AVAILABLE,
            "profile_file": str(self._profile_path),
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
            "lite_mode": cfg.lite_mode,
            "stability_buffer": cfg.stability_buffer,
        }

    def _apply_lite(self) -> None:
        """Lite mode slows the meters down; the window handles the rest (Mica, fades, memory) when it hears about it."""
        if self._telemetry_timer is not None:
            self._telemetry_timer.setInterval(
                LITE_TELEMETRY_INTERVAL_MS if self.config.lite_mode else TELEMETRY_INTERVAL_MS
            )

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
            "driver_latency_ms": list(self.engine.stream_latency_ms) if running else [0.0, 0.0],
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
        self._publish_hotkeys()
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
        if self.engine and self.engine.is_running:
            if role == "monitor":
                if self.config.hear_myself:
                    self.engine.start_monitor(device_index)
            else:
                in_idx, out_idx = self._selected["input"], self._selected["output"]
                if in_idx is not None and out_idx is not None:
                    self._sync_device_names()
                    ok = self.engine.start_engine(in_idx, out_idx, self._selected["monitor"])
                    self.engine_state_changed.emit(self._engine_payload())
                    if not ok:
                        self._notify("error", "Couldn't switch audio device", self.engine.last_error or self.engine.status_text)
                    else:
                        self._check_ai_model()

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
            applog.get_logger().info("Audio engine stopped")
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
            applog.get_logger().warning("Audio engine could not start: %s", self.engine.last_error or self.engine.status_text)
            self._notify("error", "Couldn't start the audio engine", self.engine.last_error or self.engine.status_text)
        else:
            applog.get_logger().info("Audio engine started: %s", self.engine.status_text)
            self._check_monitor()
            self._check_ai_model()

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
        driver_ms = sum(self.engine.stream_latency_ms) if self.engine.is_running else 0.0
        driver_ms = driver_ms if driver_ms > 0.0 else 6.0  # until the driver reports its buffering, assume a typical 6 ms
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
                "roundtrip_ms": block_ms * 2.0 + t["dsp_ms"] + driver_ms,
                "driver_ms": driver_ms,
                "xruns": int(t["xruns"]) + int(t["stalls"]),   # "dropouts" on screen: the driver's flags plus real stalls
                "tag": "MUTED" if cfg.mic_muted else ("BYPASS" if cfg.bypass_effects else ((f"AI: {Path(cfg.onnx_model_path).stem}" if cfg.onnx_model_path else "AI Voice Model") if cfg.enable_onnx else cfg.active_preset)),
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
        if name == "lite_mode":
            self._apply_lite()
        elif name == "enable_onnx":
            if value:
                if self.engine and self.engine.dsp and not self.engine.dsp.onnx_streamer.active:
                    if self.config.onnx_model_path:
                        ok, msg = self.engine.dsp.onnx_streamer.load_model(self.config.onnx_model_path)
                        if not ok:
                            setattr(self.config, "enable_onnx", False)
                            if msg.startswith("NEED_HUBERT"):  # same as picking the model: offer the download
                                self.hubert_download_requested.emit(self.config.onnx_model_path)
                            else:
                                self._notify("error", "Model Load Failed", msg)
                            self.flags_changed.emit(self._flags_payload())
                            self.onnx_changed.emit(self._onnx_payload(msg))
                            return
                    else:
                        setattr(self.config, "enable_onnx", False)
                        self._notify("warning", "No Model Selected", "Select an .onnx voice model first.")
                        self.flags_changed.emit(self._flags_payload())
                        self.onnx_changed.emit(self._onnx_payload("No model selected."))
                        return
            self.onnx_changed.emit(self._onnx_payload())
            self._tick()
        self.flags_changed.emit(self._flags_payload())
        if name == "hear_myself":
            if self.engine and self.engine.is_running:
                if value:
                    mon_idx = self._selected.get("monitor")
                    out_idx = self._selected.get("output")
                    if mon_idx is None or mon_idx == out_idx:
                        self._notify(
                            "warning",
                            "Hear Myself needs another device",
                            "Pick headphones that differ from the Virtual Cable Output.",
                        )
                    else:
                        ok = self.engine.start_monitor(mon_idx)
                        if not ok:
                            self._notify("warning", "Couldn't open monitor", "Check that your headphones are plugged in and ready.")
                else:
                    self.engine.stop_monitor()
        elif name == "stability_buffer" and self.engine and self.engine.is_running:
            self._notify("info", "Applies at the next start", "Restart the audio engine to use the new buffer.")

    def _check_monitor(self) -> None:
        """Hear Myself plays through a second stream, which only opens on a device other than the cable output."""
        if not self.config.hear_myself:
            return
        if self._selected["monitor"] is None or self._selected["monitor"] == self._selected["output"]:
            self._notify(
                "warning",
                "Hear Myself needs another device",
                "Pick headphones that differ from the Virtual Cable Output.",
            )
        elif self.engine and self.engine.is_running and self.engine.monitor_stream is None:
            ok = self.engine.start_monitor(self._selected["monitor"])
            if not ok:
                self._notify("warning", "Couldn't open monitor", "Check that your headphones are plugged in and ready.")

    def _check_ai_model(self) -> None:
        """The AI voice is on but its model did not load when the engine started. The output is silent on purpose
        then (the raw voice must never reach the game), so say why instead of leaving the player talking into nothing."""
        cfg = self.config
        if not (cfg.enable_onnx and self.engine and self.engine.is_running) or self.engine.dsp.onnx_streamer.active:
            return
        message = self.engine.onnx_message or "No model selected."
        applog.get_logger().warning("AI voice model not loaded at start: %s", message)
        if message.startswith("NEED_HUBERT"):
            self.hubert_download_requested.emit(cfg.onnx_model_path)
        else:
            self._notify(
                "error",
                "AI voice model not loaded",
                f"{message} Your voice stays silent until a model loads or the AI voice is switched off.",
            )
        self.onnx_changed.emit(self._onnx_payload(message))

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
        applog.get_logger().warning("Custom presets could not be loaded: %s", self._library.load_error)
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
        self._apply_lite()
        self._hk_conflicts.clear()                      # the defaults came back with the loop above
        self._hotkey_commit(persist=True, restored=True)
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

    def _hotkeys_payload(self) -> dict:
        cfg = self.config
        items = []
        for action in hk.ACTIONS:
            serial = cfg.hotkeys.get(action.id, "")
            items.append(
                {
                    "id": action.id,
                    "text": hk.combo_text(serial),
                    "bound": bool(serial),
                    "registered": self._hk_registered.get(action.id) if serial else None,
                    "conflict": self._hk_conflicts.get(action.id, ""),
                }
            )
        return {"items": items, "all_default": hk.normalize_bindings(cfg.hotkeys) == hk.default_bindings()}

    def _publish_hotkeys(self) -> None:
        self.hotkeys_changed.emit(self._hotkeys_payload())

    # ---- changing a hotkey -----------------------------------------------------------------------------------
    # The Settings page asks for a capture first: every registration is dropped (a registered combination never
    # reaches the page, it would fire its action instead), then the page listens, and ends with one of change /
    # clear / cancel, which put the registrations back. All RegisterHotKey calls happen on the listener thread; this
    # thread only queues commands and reads the answers, so neither the GUI nor the audio thread ever waits for them.

    def _hotkey_command(self, **context) -> int:
        token = next(self._hk_tokens)
        self._hk_pending[token] = context
        return token

    def _hotkey_commit(self, **context) -> None:
        """Sends the bindings in ``config.hotkeys`` to the listener thread (and ends any capture)."""
        self._hk_watchdog.stop()
        self._hk_suspended = False
        if self._hotkeys is None:
            return
        context["requested"] = dict(self.config.hotkeys)
        self._hotkeys.apply(hk.to_table(self.config.hotkeys), self._hotkey_command(**context), resume=True)

    def _resume_hotkeys(self) -> None:
        self._hk_watchdog.stop()
        if self._hotkeys is not None and self._hk_suspended:
            self._hk_suspended = False
            self._hotkeys.resume(self._hotkey_command())

    @pyqtSlot(str)
    def begin_hotkey_capture(self, action_id: str) -> None:
        if self._hotkeys is None or action_id not in hk.ACTION_BY_ID:
            return
        self._hk_suspended = True
        self._hk_watchdog.start()
        self._hotkeys.suspend(self._hotkey_command(capture=action_id))

    @pyqtSlot()
    def cancel_hotkey_capture(self) -> None:
        self._resume_hotkeys()

    def _hotkey_capture_timed_out(self) -> None:
        if self._hk_suspended:
            applog.get_logger().warning("A hotkey capture was left open; the hotkeys were switched back on")
            self._resume_hotkeys()

    @pyqtSlot(str, int, int)
    def change_hotkey(self, action_id: str, mods: int, vk: int) -> None:
        if action_id not in hk.ACTION_BY_ID:
            self._resume_hotkeys()
            return
        combo = hk.Combo(mods, vk)
        try:
            warning = hk.validate_combo(self.config.hotkeys, action_id, combo, self._action_label)
        except hk.HotkeyError as exc:
            self._notify("error", f"Couldn't change {self._action_label(action_id)}", exc.message)
            self._resume_hotkeys()
            self._publish_hotkeys()
            return
        self.config.hotkeys[action_id] = combo.serial()
        self._hk_conflicts.pop(action_id, None)
        self._hotkey_commit(action=action_id, persist=True, warning=warning)

    @pyqtSlot(str)
    def clear_hotkey(self, action_id: str) -> None:
        if action_id not in hk.ACTION_BY_ID:
            self._resume_hotkeys()
            return
        self.config.hotkeys[action_id] = ""
        self._hk_conflicts.pop(action_id, None)
        self._hotkey_commit(action=action_id, persist=True)

    @pyqtSlot()
    def restore_default_hotkeys(self) -> None:
        self.config.hotkeys = hk.default_bindings()
        self._hk_conflicts.clear()
        self._hotkey_commit(persist=True, restored=True)

    @pyqtSlot(object)
    def _on_hotkey_result(self, result: dict) -> None:
        """The listener thread's answer. Its table is the truth about what is bound now: a change Windows refused
        (another program owns the combination) has been put back on the previous combination there."""
        context = self._hk_pending.pop(result.get("token"), None) or {}
        kind = result["kind"]
        if result.get("error"):
            applog.get_logger().error("Hotkey listener: %s", result["error"])
            self._notify("error", "Hotkeys hit a problem", f"{result['error']}. Settings > Report a problem sends it to the developer.")
        if kind == "suspend":
            if "capture" in context and self._hk_suspended:
                self.hotkey_capture_ready.emit(context["capture"])
            return
        cfg = self.config
        for action in hk.ACTIONS:
            if action.hk_id in result["table"]:
                cfg.hotkeys[action.id] = hk.from_table_entry(result["table"][action.hk_id])
        registered = result["registered"]
        self._hk_registered = {a.id: registered.get(a.hk_id) for a in hk.ACTIONS}

        requested = context.get("requested", {})
        for hk_id in result["reverted"]:
            action = hk.ACTION_BY_HK_ID[hk_id]
            label = self._action_label(action.id)
            tried, kept = hk.combo_text(requested.get(action.id, "")), hk.combo_text(cfg.hotkeys.get(action.id, ""))
            error = result["errors"].get(hk_id, 0)
            why = (
                "another program already uses it"
                if error == eng.Win32HotkeyListener.ERROR_HOTKEY_ALREADY_REGISTERED
                else f"Windows refused it (error {error})"
            )
            self._hk_conflicts[action.id] = f"{tried} is used by another program. Still {kept}."
            self._notify("error", f"Couldn't use {tried} for {label}", f"{why[0].upper()}{why[1:]}. {label} stays {kept}.")
            applog.get_logger().warning("Hotkey %s: %s refused (error %s); kept %s", action.id, tried, error, kept)

        if kind == "startup":
            bound = [ok for ok in registered.values()]
            taken = sum(1 for ok in bound if not ok)
            if taken:
                held = ", ".join(
                    f"{hk.ACTION_BY_HK_ID[i].id} {hk.combo_text(cfg.hotkeys.get(hk.ACTION_BY_HK_ID[i].id, ''))} "
                    f"(error {result['errors'].get(i, '?')})"
                    for i, ok in registered.items()
                    if not ok
                )
                applog.get_logger().warning("Hotkeys another program holds, not registered at start: %s", held)
                self._notify(
                    "warning",
                    "Some hotkeys are unavailable",
                    f"{taken} of {len(bound)} are used by another app, maybe a second V-COMMS. See Settings.",
                )
        warning, changed = context.get("warning"), context.get("action")
        if warning and changed and changed not in {hk.ACTION_BY_HK_ID[i].id for i in result["reverted"]}:
            now = hk.combo_text(cfg.hotkeys.get(changed, ""))
            self._notify("warning", f"{self._action_label(changed)} is now {now}", warning)
        if context.get("persist"):
            self._persist_hotkeys()
        self._publish_hotkeys()

    def _persist_hotkeys(self) -> None:
        """Saves the bindings the moment they change, so a restart keeps them without saving a profile."""
        cfg = self.config
        if cfg.hotkeys == self._hk_baseline:
            return
        try:
            hk.save_store(cfg.hotkeys)
        except OSError as exc:
            applog.get_logger().warning("Hotkeys could not be saved: %s", exc)
            self._notify(
                "warning",
                "Couldn't save your hotkeys",
                f"{hk.store_path()} could not be written ({exc.strerror or exc}). They work until you quit V-COMMS.",
            )
            return
        self._hk_baseline = dict(cfg.hotkeys)

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
        hotkeys_before = dict(self.config.hotkeys)
        if not eng.load_profile_into(self.config, Path(path)):
            self._notify("error", "Couldn't load the profile", f"{Path(path).name} is not a valid V-COMMS profile.")
            return
        self._pending.clear()
        self._scan_devices()
        self._apply_lite()
        self.engine.dsp.update_parameters(self.config.current_params)
        if self.config.hotkeys != hotkeys_before:       # the profile carries its own hotkeys
            self._hk_conflicts.clear()
            self._hotkey_commit(persist=True)
        self._publish_everything()
        self.profile_changed.emit({"path": str(Path(path).resolve()), "autoloaded": False})
        self._notify("success", "Profile loaded", Path(path).name)

    @pyqtSlot(str)
    def load_onnx_model(self, path: str) -> None:
        if not path:
            return
        self.config.onnx_model_path = path
        ok, message = self.engine.dsp.onnx_streamer.load_model(path)
        if ok:
            self.config.enable_onnx = True
        else:
            self.config.enable_onnx = False
        self.flags_changed.emit(self._flags_payload())
        self.onnx_changed.emit(self._onnx_payload(message))
        self._tick()
        if ok:
            self._notify("success", "ONNX Voice Model", message)
        else:
            if message.startswith("NEED_HUBERT"):
                self.hubert_download_requested.emit(path)
            else:
                self._notify("error", "ONNX Voice Model", message)

    # ======================================================================================================
    # problem reports (Settings > Report a problem)
    # ======================================================================================================

    def _diagnostics_snapshot(self, qt_versions: str) -> dict:
        cfg = self.config
        return {
            "input": self._label_for("input"),
            "output": self._label_for("output"),
            "monitor": self._label_for("monitor"),
            "sample_rate": cfg.sample_rate,
            "block_size": self._block_size(),
            "preset": cfg.active_preset,
            "flags": self._flags_payload(),
            "lite_mode": cfg.lite_mode,
            "stability_buffer": cfg.stability_buffer,
            "engine": self.engine.read_session_stats() if self.engine else {},
            "ai_model": self._ai_model_state(),
            "qt_versions": qt_versions,
        }

    def _ai_model_state(self) -> dict:
        streamer = self.engine.dsp.onnx_streamer if self.engine else None
        if streamer is None:
            return {}
        return {
            "loaded": streamer.active,
            "kind": ("RVC" if streamer.is_rvc else "direct") if streamer.active else "",
            "errors": streamer.error_count,
            "last_error": streamer.last_error,
        }

    @pyqtSlot(str)
    def prepare_report(self, qt_versions: str) -> None:
        """Gathers the diagnostics (the form shows them in full before anything is sent) and opens the form."""
        text = diagnostics.build_diagnostics(self._diagnostics_snapshot(qt_versions))
        self.report_ready.emit(
            {"diagnostics": text, "wait_s": self._report_limiter.wait_seconds(), "sending": self._report_sending}
        )

    @pyqtSlot(object)
    def submit_report(self, payload: dict) -> None:
        """Delivers the report on a worker thread, so the network never makes the UI (or the audio) wait."""
        if self._report_sending:
            self._notify("info", "Still sending", "The previous report has not finished yet.")
            return
        title = str(payload.get("title", ""))[:200]
        body = str(payload.get("body", ""))[: reporting.BODY_MAX]
        reply_to = str(payload.get("reply_to", "")).strip()
        if reply_to and not reporting.valid_email(reply_to):
            reply_to = ""
        self._report_sending = True
        applog.get_logger().info("Problem report: sending (%d characters)", len(body))

        def worker() -> None:
            try:
                result = reporting.submit(title, body, reply_to, self._report_limiter)
            except Exception as exc:  # the player's report must never be lost to a surprise
                result = {
                    "status": "fallback",
                    "reason": "failed",
                    "detail": f"unexpected error ({type(exc).__name__})",
                    "mailto": reporting.mailto_url(reporting.report_config.MAILTO_ADDRESS, title, body),
                    "text": reporting.report_text(title, body),
                    "address": reporting.report_config.MAILTO_ADDRESS,
                }
            self._report_done.emit(result)

        threading.Thread(target=worker, daemon=True, name="vcomms-report").start()

    @pyqtSlot(object)
    def _on_report_done(self, result: dict) -> None:
        self._report_sending = False
        detail = f" ({result['detail']})" if result.get("detail") else ""
        applog.get_logger().info("Problem report: %s%s", result.get("status"), detail)
        self.report_finished.emit(result)
