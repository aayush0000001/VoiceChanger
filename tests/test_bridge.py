"""EngineBridge on its own QThread, driven exactly like the UI drives it: queued slot calls in, signals out."""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import Q_ARG, QCoreApplication, QMetaObject, Qt

import appdata
import engine as eng
import hotkeys as hk
import presets as pr
from conftest import ITERATIONS, SEEDS

SIGNALS = ["ready", "catalog_ready", "presets_changed", "preset_changed", "params_changed", "flags_changed",
           "devices_changed", "engine_state_changed", "status_updated", "path_test_state", "path_test_finished",
           "onnx_changed", "hubert_download_requested", "profile_changed", "hotkeys_changed", "hotkey_capture_ready",
           "notify", "report_ready", "report_finished"]


def pump(until=None, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        QCoreApplication.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.002)
    return until is None


class Bridge:
    """An EngineBridge with every signal recorded, and queued calls into its slots."""

    def __init__(self):
        from engine_bridge import EngineBridge

        self.obj = EngineBridge()
        self.got = defaultdict(list)
        for name in SIGNALS:
            getattr(self.obj, name).connect(lambda *args, n=name: self.got[n].append(args[0] if len(args) == 1 else args))
        self.obj.start()
        assert pump(lambda: self.got["ready"]), "the bridge never became ready"

    def call(self, slot, *args):
        def q(value):
            if isinstance(value, bool):
                return Q_ARG(bool, value)
            if isinstance(value, int):
                return Q_ARG(int, value)
            if isinstance(value, float):
                return Q_ARG(float, value)
            if isinstance(value, str):
                return Q_ARG(str, value)
            return Q_ARG("PyQt_PyObject", value)

        QMetaObject.invokeMethod(self.obj, slot, Qt.ConnectionType.QueuedConnection, *[q(a) for a in args])

    def wait(self, signal, count=1, timeout=5.0, where=None):
        ok = pump(lambda: len([v for v in self.got[signal] if where is None or where(v)]) >= count, timeout)
        assert ok, f"{signal} x{count} never arrived (got {self.got[signal][-3:]})"
        return self.got[signal][-1]

    def last(self, signal):
        return self.got[signal][-1]

    def notes(self, level=None):
        return [n for n in self.got["notify"] if level is None or n["level"] == level]

    def close(self):
        self.obj.request_shutdown()
        assert not self.obj._thread.isRunning()


@pytest.fixture
def bridge(isolated_home, fake_sd, qapp):
    b = Bridge()
    yield b
    b.close()


def test_startup_publishes_everything(bridge):
    got = bridge.got
    for name in ("catalog_ready", "presets_changed", "preset_changed", "params_changed", "flags_changed",
                 "devices_changed", "engine_state_changed", "onnx_changed", "hotkeys_changed", "profile_changed"):
        assert got[name], name
    catalog = got["catalog_ready"][-1]
    assert [r["id"] for r in catalog["hotkeys"]] == [a.id for a in hk.ACTIONS]
    assert catalog["onnx_available"] is True and catalog["preset_rules"]["max_custom"] == pr.MAX_CUSTOM_PRESETS
    presets = got["presets_changed"][-1]["presets"]
    assert [p["key"] for p in presets] == list(eng.TACTICAL_PRESETS)
    assert [p["hotkey"] for p in presets[:4]] == ["Ctrl + Alt + F1", "Ctrl + Alt + F2", "Ctrl + Alt + F3", "Ctrl + Alt + F4"]
    devices = got["devices_changed"][-1]
    assert devices["selected"] == {"input": 0, "output": 2, "monitor": 1}  # mic, CABLE Input, a real output
    assert got["engine_state_changed"][-1]["state"] == "stopped"
    assert got["profile_changed"][-1] == {"path": "", "autoloaded": False}
    assert bridge.notes("error") == []


@pytest.mark.parametrize("seed", SEEDS)
def test_presets_and_sliders(bridge, seed):
    rnd = random.Random(seed)
    name = rnd.choice(list(eng.TACTICAL_PRESETS))
    bridge.call("apply_preset", name)
    bridge.wait("preset_changed", where=lambda v: v == name)
    expected = {k: v for k, v in vars(eng.TACTICAL_PRESETS[name]).items() if k != "name"}
    # wait for this preset's params_changed itself: the start-up one is already recorded, and the two queued signals
    # can arrive in separate event-loop passes
    assert bridge.wait("params_changed", where=lambda v: v == expected) == expected
    # a slider drag: many moves, coalesced into one snapshot rebuild
    key = rnd.choice(list(eng.PARAM_RANGES))
    lo, hi = eng.PARAM_RANGES[key]
    values = [rnd.uniform(lo, hi) for _ in range(30)]
    for v in values:
        bridge.call("set_param", key, float(v))
    bridge.call("set_param", "not_a_param", 1.0)
    assert pump(lambda: abs(getattr(bridge.obj.engine.dsp._snapshot.params, key) - values[-1]) < 1e-9)
    assert abs(getattr(bridge.obj.config.current_params, key) - values[-1]) < 1e-9
    bridge.call("reset_to_preset")
    assert pump(lambda: bridge.obj.engine.dsp._snapshot.params == eng.TACTICAL_PRESETS[name])
    bridge.call("apply_preset", "No Such Preset")
    pump(timeout=0.05)
    assert bridge.obj.config.active_preset == name


@pytest.mark.parametrize("seed", SEEDS)
def test_engine_start_stop_and_device_switch(bridge, fake_sd, seed):
    bridge.call("set_flag", "hear_myself", bool(seed % 2))
    bridge.call("toggle_engine")
    state = bridge.wait("engine_state_changed", where=lambda v: v["state"] == "running")
    assert state["block_size"] == 256 and state["driver_latency_ms"] == [10.0, 20.0]
    duplex = [s for s in fake_sd.streams if s.kind == "duplex"][-1]
    assert duplex.started and duplex.kwargs["device"] == (0, 2)
    for _ in range(5):
        duplex.run_block(np.stack([np.full(256, 0.1, np.float32)] * 2, axis=1))
    bridge.call("select_device", "input", 5)  # switching the mic restarts the streams
    bridge.wait("engine_state_changed", count=2, where=lambda v: v["state"] == "running")
    assert [s for s in fake_sd.streams if s.kind == "duplex"][-1].kwargs["device"] == (5, 2)
    bridge.call("select_device", "input", 999)  # not a device: ignored
    bridge.call("rescan_devices")
    bridge.wait("notify", where=lambda n: n["title"] == "Stop first")
    bridge.call("set_block_size", 512)
    bridge.call("set_block_size", 333)  # not offered: ignored
    bridge.call("toggle_engine")
    bridge.wait("engine_state_changed", count=2, where=lambda v: v["state"] == "stopped")  # 1st: the startup state
    assert bridge.obj.config.block_size == 512
    assert all(s.closed for s in fake_sd.streams)
    if seed % 2:
        assert any(s.kind == "output" for s in fake_sd.streams)  # Hear Myself opened the headphone stream


def test_engine_open_failure_is_reported(bridge, fake_sd):
    fake_sd.fail_open = "Device unavailable"
    bridge.call("toggle_engine")
    note = bridge.wait("notify", where=lambda n: n["level"] == "error")
    assert note["title"] == "Couldn't start the audio engine" and "Device unavailable" in note["text"]
    assert bridge.last("engine_state_changed")["state"] == "error"


def test_hear_myself_on_the_cable_is_refused(bridge):
    bridge.call("select_device", "monitor", 2)  # the cable itself
    bridge.call("toggle_engine")
    bridge.wait("engine_state_changed", where=lambda v: v["state"] == "running")
    bridge.call("set_flag", "hear_myself", True)
    bridge.wait("notify", where=lambda n: n["title"] == "Hear Myself needs another device")
    assert bridge.obj.engine.monitor_stream is None
    bridge.call("toggle_engine")


@pytest.mark.parametrize("seed", SEEDS)
def test_custom_preset_lifecycle(bridge, isolated_home, seed):
    rnd = random.Random(seed)
    bridge.call("set_param", "pitch_semitones", 7.5)
    name = f"Mine {seed}"
    bridge.call("save_preset", name)
    bridge.wait("notify", where=lambda n: n["title"] == "Preset saved")
    assert bridge.last("presets_changed")["custom_names"] == [name]
    lib_file = appdata.presets_path()
    assert json.loads(lib_file.read_text())["presets"][0]["params"]["pitch_semitones"] == 7.5
    bridge.call("save_preset", "Omen")  # a built-in's name
    bridge.wait("notify", where=lambda n: n["title"] == "Couldn't save the preset")
    renamed = f"Renamed {seed}"
    bridge.call("rename_preset", name, renamed)
    bridge.wait("preset_changed", where=lambda v: v == renamed)
    export = isolated_home / "cwd" / "out.json"
    bridge.call("export_preset", renamed, str(export))
    bridge.wait("notify", where=lambda n: n["title"] == "Preset exported")
    bridge.call("import_preset", str(export))  # the same name is taken: imported as "Renamed N (2)"
    bridge.wait("notify", where=lambda n: n["title"] == "Preset imported")
    assert bridge.last("presets_changed")["custom_names"] == [renamed, f"{renamed} (2)"]
    bridge.call("export_current", str(isolated_home / "cwd" / "current.json"))
    bridge.wait("notify", count=2, where=lambda n: n["title"] == "Preset exported")
    bridge.call("delete_preset", f"{renamed} (2)")  # it was active: the voice falls back to the first built-in
    bridge.wait("preset_changed", where=lambda v: v == "Deep Shadow (Omen)")
    bridge.call("import_preset", str(isolated_home / "cwd" / "missing.json"))
    bridge.wait("notify", where=lambda n: n["title"] == "Couldn't import the preset")
    delete_all = bool(rnd.getrandbits(1))
    bridge.call("reset_all", delete_all)
    bridge.wait("notify", where=lambda n: n["title"] == "Settings reset")
    assert bridge.obj._library.names() == ([] if delete_all else [renamed])


def test_profile_save_and_load(bridge, isolated_home):
    bridge.call("apply_preset", "Tactical Comms (Radio)")
    bridge.call("set_block_size", 128)
    path = isolated_home / "cwd" / "profile.json"
    bridge.call("save_profile", str(path))
    bridge.wait("notify", where=lambda n: n["title"] == "Profile saved")
    data = json.loads(path.read_text())
    assert data["active_preset"] == "Tactical Comms (Radio)" and data["block_size"] == 128
    assert data["output_device_name"].startswith("CABLE Input")
    bridge.call("apply_preset", "Female Cute (Airy)")
    bridge.call("load_profile", str(path))
    bridge.wait("notify", where=lambda n: n["title"] == "Profile loaded")
    assert bridge.obj.config.active_preset == "Tactical Comms (Radio)"
    path.write_text("{ broken")
    bridge.call("load_profile", str(path))
    bridge.wait("notify", where=lambda n: n["title"] == "Couldn't load the profile")


def test_startup_profile_is_autoloaded(isolated_home, fake_sd, qapp):
    cfg = eng.AppConfig(active_preset="Female Husky (Low)", current_params=eng.preset_parameters("Female Husky (Low)"))
    appdata.roaming_dir().mkdir(parents=True, exist_ok=True)
    eng.save_profile(cfg, appdata.profile_path())
    b = Bridge()
    try:
        assert b.last("preset_changed") == "Female Husky (Low)"
        assert b.last("profile_changed")["autoloaded"] is True
        assert b.obj.engine.dsp._snapshot.params == eng.TACTICAL_PRESETS["Female Husky (Low)"]
    finally:
        b.close()


def test_ai_model_paths(bridge, onnx_models):
    bridge.call("load_onnx_model", str(onnx_models["rvc_no_hubert"]))
    assert bridge.wait("hubert_download_requested") == str(onnx_models["rvc_no_hubert"])
    assert bridge.obj.config.enable_onnx is False
    bridge.call("load_onnx_model", str(onnx_models["rvc_okada"]))
    msg = bridge.wait("onnx_changed", where=lambda v: v["message"].startswith("RVC Loaded"))
    assert msg["enabled"] and bridge.obj.config.enable_onnx
    bridge.wait("notify", where=lambda n: n["title"] == "ONNX Voice Model" and n["level"] == "success")
    bridge.call("set_flag", "enable_onnx", False)
    bridge.call("set_flag", "enable_onnx", True)  # already loaded: just on again
    pump(lambda: bridge.obj.config.enable_onnx)
    status = bridge.wait("status_updated", where=lambda s: s["tag"].startswith("AI:"))
    assert status["tag"] == "AI: rvc_okada"
    bridge.call("prepare_report", "Qt test")
    diag = bridge.wait("report_ready")["diagnostics"]
    assert "AI model: loaded (RVC)" in diag


def test_ai_voice_that_cannot_load_at_start_is_explained(bridge, onnx_models):
    """New: the AI voice is on in the profile but its model fails at engine start. The output is silent on purpose,
    and now the player is told why (or offered the HuBERT download) instead of talking into nothing."""
    bridge.obj.config.onnx_model_path = str(onnx_models["rvc_wrong_dim"])
    bridge.obj.config.enable_onnx = True
    bridge.call("toggle_engine")
    note = bridge.wait("notify", where=lambda n: n["title"] == "AI voice model not loaded")
    assert "test chunk" in note["text"] and "silent" in note["text"]
    bridge.call("toggle_engine")
    bridge.wait("engine_state_changed", count=2, where=lambda v: v["state"] == "stopped")
    bridge.obj.config.onnx_model_path = str(onnx_models["rvc_no_hubert"])
    bridge.call("toggle_engine")
    assert bridge.wait("hubert_download_requested") == str(onnx_models["rvc_no_hubert"])
    bridge.call("toggle_engine")


def test_switching_the_ai_voice_on_offers_the_hubert_download(bridge, onnx_models):
    bridge.obj.config.onnx_model_path = str(onnx_models["rvc_no_hubert"])
    bridge.call("set_flag", "enable_onnx", True)
    assert bridge.wait("hubert_download_requested") == str(onnx_models["rvc_no_hubert"])
    assert not [n for n in bridge.notes("error") if n["title"] == "Model Load Failed"]
    assert bridge.last("flags_changed")["enable_onnx"] is False


@pytest.mark.parametrize("seed", SEEDS)
def test_hotkey_change_flow(bridge, isolated_home, seed):
    """The listener thread does not run off Windows, so its answers are played back by hand."""
    rnd = random.Random(seed)
    action = rnd.choice(hk.ACTIONS).id
    bridge.call("begin_hotkey_capture", action)
    pump(lambda: bridge.obj._hk_suspended)
    token = max(bridge.obj._hk_pending)
    bridge.obj._hotkey_result.emit({"token": token, "kind": "suspend", "suspended": True, "table": {}, "registered": {},
                                    "errors": {}, "reverted": [], "error": ""})
    assert bridge.wait("hotkey_capture_ready") == action
    combo = hk.Combo(hk.MOD_CONTROL | hk.MOD_SHIFT, 0x70 + rnd.randint(0, 10))
    bridge.call("change_hotkey", action, combo.mods, combo.vk)
    assert pump(lambda: bridge.obj.config.hotkeys[action] == combo.serial())
    token = max(bridge.obj._hk_pending)
    table = hk.to_table(bridge.obj.config.hotkeys)
    bridge.obj._hotkey_result.emit({"token": token, "kind": "apply", "suspended": False, "table": table,
                                    "registered": {i: True for i, c in table.items() if c}, "errors": {}, "reverted": [],
                                    "error": ""})
    assert pump(lambda: hk.load_store() is not None and hk.load_store()[action] == combo.serial())
    other = next(a.id for a in hk.ACTIONS if a.id != action)
    bridge.call("change_hotkey", other, combo.mods, combo.vk)  # a duplicate is refused before Windows sees it
    bridge.wait("notify", where=lambda n: n["title"].startswith("Couldn't change"))
    bridge.call("change_hotkey", action, hk.MOD_WIN, 0x41)
    bridge.wait("notify", count=2, where=lambda n: n["title"].startswith("Couldn't change"))
    bridge.call("restore_default_hotkeys")
    assert pump(lambda: bridge.obj.config.hotkeys == hk.default_bindings())


def test_hotkey_presses_switch_presets_and_toggles(bridge):
    listener = eng.Win32HotkeyListener
    for hk_id, preset in zip((listener.HK_PRESET_1, listener.HK_PRESET_2, listener.HK_PRESET_3, listener.HK_PRESET_4),
                             list(eng.TACTICAL_PRESETS)[:4]):
        bridge.obj._hotkey_pressed.emit(hk_id)
        bridge.wait("preset_changed", where=lambda v, p=preset: v == p)
    bridge.obj._hotkey_pressed.emit(listener.HK_BYPASS)
    assert pump(lambda: bridge.obj.config.bypass_effects)
    bridge.obj._hotkey_pressed.emit(listener.HK_MUTE)
    assert pump(lambda: bridge.obj.config.mic_muted)
    bridge.obj._hotkey_pressed.emit(listener.HK_MUTE)
    assert pump(lambda: not bridge.obj.config.mic_muted)


def test_problem_report(bridge):
    bridge.call("prepare_report", "Qt 6 | PyQt 6")
    ready = bridge.wait("report_ready")
    assert ready["wait_s"] == 0 and not ready["sending"] and "V-COMMS" in ready["diagnostics"]
    bridge.call("submit_report", {"title": "T", "body": "日本語 " * 2000, "reply_to": "not an email"})
    result = bridge.wait("report_finished", timeout=10)  # used to hang forever on this body
    assert result["status"] == "fallback" and result["reason"] == "no_key"
    bridge.call("prepare_report", "Qt")
    assert bridge.wait("report_ready", count=2)["wait_s"] > 0  # one per minute


@pytest.mark.parametrize("seed", SEEDS)
def test_path_test(bridge, fake_sd, seed):
    fake_sd.rec_result = (0.2 * np.sin(2 * np.pi * 300 * np.arange(96000) / 48000)).astype(np.float32)
    fake_sd.playrec_fn = (lambda sent: sent * 0.9) if seed % 2 else (lambda sent: np.zeros_like(sent))
    bridge.call("run_path_test")
    result = bridge.wait("path_test_finished", timeout=20)
    assert result["passed"] is bool(seed % 2)
    assert [s["stage"] for s in result["steps"]] == ["Mic", "DSP", "Cable"]
    assert bridge.last("path_test_state") == {"running": False}


def test_flags_and_telemetry(bridge):
    for flag in ("bypass_effects", "mic_muted", "lite_mode", "stability_buffer"):
        bridge.call("set_flag", flag, True)
        bridge.wait("flags_changed", where=lambda f, k=flag: f[k] is True)
    bridge.call("set_flag", "nonsense", True)
    bridge.call("set_monitor_volume", 7.0)
    assert pump(lambda: bridge.obj.config.monitor_volume == 1.0)
    assert bridge.obj._telemetry_timer.interval() == 100  # Lite mode: 10 Hz meters
    status = bridge.wait("status_updated", where=lambda s: s["tag"] == "MUTED")
    assert set(status) >= {"running", "in_db", "out_db", "dsp_ms", "load", "roundtrip_ms", "xruns"}
    bridge.call("set_telemetry_active", False)
    pump(timeout=0.05)
    count = len(bridge.got["status_updated"])
    pump(timeout=0.3)
    assert len(bridge.got["status_updated"]) == count
