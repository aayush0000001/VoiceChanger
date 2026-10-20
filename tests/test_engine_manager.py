"""AudioEngineManager (streams, callbacks, telemetry), the audio path test, the hotkey listener's registration logic,
profiles and device selection. Audio hardware is the scripted fakes.FakeSoundDevice."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest

import engine as eng
import hotkeys as hk
from conftest import ITERATIONS, SEEDS
from fakes import CABLE_IN, CABLE_OUT, HEADPHONES, MIC, SPEAKERS


def tone(n, f=440.0, amp=0.3, sr=48000):
    return (amp * np.sin(2 * np.pi * f * np.arange(n) / sr)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ devices
def test_device_list_prefers_wasapi_and_drops_duplicates(fake_sd):
    inputs, outputs = eng.AudioEngineManager.list_audio_devices()
    assert all(label.endswith("[Windows WASAPI]") for _, label in inputs + outputs)
    assert [i for i, _ in inputs] == [0, 3, 5]
    assert [i for i, _ in outputs] == [1, 2, 4]
    fake_sd.hostapis = [{"name": "MME"}, {"name": "DirectSound"}]  # no WASAPI: everything, deduplicated by name
    for d in fake_sd.devices:
        d["hostapi"] = 0 if d["hostapi"] == 0 else 1
    inputs, outputs = eng.AudioEngineManager.list_audio_devices()
    names = [label.rsplit(" [", 1)[0] for _, label in inputs]
    assert len(names) == len(set(names))


def test_no_sounddevice_means_no_devices(monkeypatch):
    monkeypatch.setattr(eng, "SOUNDDEVICE_AVAILABLE", False)
    assert eng.AudioEngineManager.list_audio_devices() == ([], [])
    manager = eng.AudioEngineManager(eng.AppConfig())
    assert manager.start_engine(0, 1, None) is False and "not installed" in manager.last_error


LABELS_IN = [(0, "Microphone (Realtek Audio) [Windows WASAPI]"), (5, "Headset Microphone (USB Headset) [Windows WASAPI]"),
             (7, "Headset [Windows WASAPI]"), (8, "USB Headset [Windows WASAPI]")]
LABELS_OUT = [(1, "Speakers (Realtek Audio) [Windows WASAPI]"), (2, "CABLE Input (VB-Audio Virtual Cable) [Windows WASAPI]"),
              (9, "CABLE In 16ch (VB-Audio Virtual Cable) [Windows WASAPI]"), (4, "Headphones (USB Headset) [Windows WASAPI]")]


@pytest.mark.parametrize("seed", SEEDS)
def test_pick_device_targets(seed):
    rng = np.random.default_rng(seed)
    ins = list(LABELS_IN)
    outs = list(LABELS_OUT)
    order_in, order_out = rng.permutation(len(ins)), rng.permutation(len(outs))
    ins, outs = [ins[i] for i in order_in], [outs[i] for i in order_out]
    pos_in = {label: i for i, (_, label) in enumerate(ins)}
    pos_out = {label: i for i, (_, label) in enumerate(outs)}
    cable = pos_out["CABLE Input (VB-Audio Virtual Cable) [Windows WASAPI]"]

    # nothing saved: first mic, the cable, and a real (non-cable) output for the monitor
    i, o, m = eng.pick_device_targets(eng.AppConfig(), ins, outs)
    assert (i, o) == (0, cable) and m != o and "CABLE" not in outs[m][1]
    # exact labels win over containment ("Headset" is inside "USB Headset")
    cfg = eng.AppConfig(input_device_name="Headset [Windows WASAPI]", output_device_name=outs[cable][1],
                        monitor_device_name="Headphones (USB Headset) [Windows WASAPI]")
    i, o, m = eng.pick_device_targets(cfg, ins, outs)
    assert i == pos_in["Headset [Windows WASAPI]"] and o == cable
    assert m == pos_out["Headphones (USB Headset) [Windows WASAPI]"]
    # an older profile saved the name without the API suffix: containment still finds it
    cfg = eng.AppConfig(input_device_name="Microphone (Realtek Audio)")
    assert eng.pick_device_targets(cfg, ins, outs)[0] == pos_in["Microphone (Realtek Audio) [Windows WASAPI]"]
    # a saved output that is unplugged falls back to the cable, not to the speakers
    cfg = eng.AppConfig(output_device_name="Old USB DAC [Windows WASAPI]", monitor_device_name="Gone [Windows WASAPI]")
    i, o, m = eng.pick_device_targets(cfg, ins, outs)
    assert o == cable and m != o
    # results are always valid positions, even for empty or one-entry lists
    assert eng.pick_device_targets(eng.AppConfig(), [], []) == (0, 0, 0)
    assert eng.pick_device_targets(eng.AppConfig(), ins[:1], outs[:1]) == (0, 0, 0)


# ------------------------------------------------------------------------------------------------ the engine
def started_engine(fake_sd, **cfg):
    config = eng.AppConfig(**cfg)
    statuses = []
    manager = eng.AudioEngineManager(config, on_status=statuses.append)
    assert manager.start_engine(MIC, CABLE_IN, HEADPHONES), manager.last_error
    return manager, fake_sd.streams[-1] if not config.hear_myself else fake_sd.streams[-2], statuses


def test_start_opens_the_right_stream(fake_sd):
    manager, stream, statuses = started_engine(fake_sd)
    kw = stream.kwargs
    assert kw["device"] == (MIC, CABLE_IN) and kw["channels"] == (2, 2) and kw["blocksize"] == 256
    assert kw["latency"] == "low" and kw["samplerate"] == 48000 and kw["dtype"] == "float32"
    assert all(isinstance(s, eng.sd.WasapiSettings) and s.auto_convert and not s.exclusive for s in kw["extra_settings"])
    assert stream.started and manager.is_running and statuses[-1].startswith("ACTIVE")
    assert manager.stream_latency_ms == (10.0, 20.0) and abs(manager.stall_threshold_s - 0.025) < 1e-9
    manager.stop_engine()
    assert stream.closed and not manager.is_running and statuses[-1] == "STOPPED"


def test_stability_buffer_asks_for_more_latency(fake_sd):
    manager, stream, _ = started_engine(fake_sd, stability_buffer=True)
    assert stream.kwargs["latency"] == eng.STABILITY_LATENCY_S
    manager.stop_engine()


@pytest.mark.parametrize("seed", SEEDS)
def test_callback_processes_and_meters(fake_sd, seed):
    rng = np.random.default_rng(seed)
    manager, stream, _ = started_engine(fake_sd)
    manager.dsp.update_parameters(eng.preset_parameters(list(eng.TACTICAL_PRESETS)[seed % 8]))
    layout = seed % 3
    for _ in range(30):
        mono = tone(256, float(rng.uniform(100, 400)), amp=float(rng.uniform(0.05, 0.5)))
        if layout == 0:
            indata = np.stack([mono, mono], axis=1)          # both channels
        elif layout == 1:
            indata = np.stack([mono, np.zeros(256, np.float32)], axis=1)  # mic on the left only
        else:
            indata = np.stack([np.zeros(256, np.float32), mono], axis=1)  # mic on the right only
        out = stream.run_block(indata)
        assert np.all(np.isfinite(out)) and np.max(np.abs(out)) <= 0.891 + 1e-6
        assert np.array_equal(out[:, 0], out[:, 1])
    t = manager.read_telemetry()
    assert -40 < t["in_db"] < 0 and t["out_db"] > -80 and t["dsp_ms"] > 0 and t["peak_dsp_ms"] >= t["dsp_ms"]
    assert manager.read_telemetry()["peak_dsp_ms"] == 0.0  # the peak resets on every read
    stats = manager.read_session_stats()
    assert stats["blocks"] == 30 and stats["running"] and 0 < stats["mean_dsp_ms"] <= stats["worst_dsp_ms"]
    manager.stop_engine()


@pytest.mark.parametrize("seed", SEEDS)
def test_mute_bypass_and_monitor(fake_sd, seed):
    rng = np.random.default_rng(seed)
    manager, stream, _ = started_engine(fake_sd, hear_myself=True, monitor_volume=float(rng.uniform(0.1, 1)))
    monitor = fake_sd.streams[-1]
    assert monitor.kind == "output" and monitor.kwargs["device"] == HEADPHONES
    mono = tone(256)
    indata = np.stack([mono, mono], axis=1)
    cfg = manager.config
    cfg.mic_muted = True
    assert not stream.run_block(indata).any() and manager.latest_out_db == -80.0
    cfg.mic_muted, cfg.bypass_effects = False, True
    assert np.allclose(stream.run_block(indata)[:, 0], mono)  # raw voice, unprocessed
    cfg.bypass_effects = False
    heard = []
    for _ in range(12):
        sent = stream.run_block(indata)[:, 0].copy()
        heard.append((sent, monitor.run_block()))
    sent, got = heard[-1]
    assert got.shape == (256, 2) and np.array_equal(got[:, 0], got[:, 1])
    assert np.abs(got).max() > 0  # the processed voice reaches the headphones ...
    assert np.abs(got).max() <= np.abs(np.concatenate([h[0] for h in heard])).max() * cfg.monitor_volume + 1e-6  # ... at the monitor volume
    cfg.hear_myself = False
    assert not monitor.run_block().any()
    manager.stop_engine()
    assert monitor.closed


def test_monitor_never_on_the_cable(fake_sd):
    manager, _, _ = started_engine(fake_sd)
    assert manager.start_monitor(CABLE_IN) is False and manager.monitor_stream is None
    assert manager.start_monitor(None) is False
    manager.stop_engine()
    assert manager.start_monitor(HEADPHONES) is False  # not running


def test_stalls_and_xruns_are_counted(fake_sd, monkeypatch):
    manager, stream, _ = started_engine(fake_sd)
    clock = iter(np.cumsum([0.0053] * 5 + [0.2] + [0.0053] * 5))
    monkeypatch.setattr(eng.time, "perf_counter", lambda: float(next(clock, 999.0)))
    for _ in range(10):
        stream.run_block(np.zeros((256, 2), np.float32))
    stream.run_block(np.zeros((256, 2), np.float32), status=1)
    t = manager.read_telemetry()
    assert t["stalls"] >= 1 and t["xruns"] == 1
    monkeypatch.undo()
    manager.stop_engine()


def test_stream_open_failure_is_reported(fake_sd):
    fake_sd.fail_open = "Invalid device"
    statuses = []
    manager = eng.AudioEngineManager(eng.AppConfig(), on_status=statuses.append)
    assert manager.start_engine(MIC, CABLE_IN, None) is False
    assert manager.last_error == "Invalid device" and "Invalid device" in statuses[-1] and not manager.is_running


@pytest.mark.parametrize("seed", SEEDS)
def test_restart_cycles_leave_nothing_open(fake_sd, seed):
    manager = eng.AudioEngineManager(eng.AppConfig(hear_myself=bool(seed % 2)))
    for _ in range(5):
        assert manager.start_engine(MIC, CABLE_IN, HEADPHONES)
    manager.stop_engine()
    assert all(s.closed for s in fake_sd.streams)


def test_ai_model_only_loads_when_the_ai_voice_is_on(fake_sd, onnx_models):
    manager = eng.AudioEngineManager(eng.AppConfig(onnx_model_path=str(onnx_models["rvc_okada"]), enable_onnx=False))
    assert manager.start_engine(MIC, CABLE_IN, None)
    assert not manager.dsp.onnx_streamer.active and manager.onnx_message == ""
    manager.config.enable_onnx = True
    assert manager.start_engine(MIC, CABLE_IN, None)
    assert manager.dsp.onnx_streamer.active and manager.onnx_message.startswith("RVC Loaded")
    manager.config.onnx_model_path = str(onnx_models["rvc_no_hubert"])
    assert manager.start_engine(MIC, CABLE_IN, None)
    assert not manager.dsp.onnx_streamer.active and manager.onnx_message.startswith("NEED_HUBERT")
    out = fake_sd.streams[-1].run_block(np.stack([tone(256)] * 2, axis=1))
    assert not out.any()  # AI voice on without a model: silence, never the raw microphone
    manager.stop_engine()


# ------------------------------------------------------------------------------------------------ audio path test
def mic_take(level):
    return tone(96000, 300, amp=level)


@pytest.mark.parametrize("seed", SEEDS)
def test_audio_path_test(fake_sd, seed):
    rng = np.random.default_rng(seed)
    delay = int(rng.integers(100, 4000))
    fake_sd.rec_result = mic_take(float(rng.uniform(0.05, 0.5)))
    fake_sd.playrec_fn = lambda sent: np.concatenate([np.zeros(delay, np.float32), sent])[: len(sent)] * 0.9
    params = eng.preset_parameters(list(eng.TACTICAL_PRESETS)[seed % 8])
    passed, lines = eng.run_audio_path_test(MIC, CABLE_IN, params)
    assert passed, lines
    assert [l.split("]")[0] + "]" for l in lines] == ["[PASS]", "[PASS]", "[PASS]"]
    ms = float(lines[2].split("about ")[1].split(" ms")[0])
    assert abs(ms - delay / 48) < 2


def test_audio_path_test_failures(fake_sd):
    fake_sd.rec_result = np.zeros(96000, np.float32)  # dead mic
    fake_sd.playrec_fn = lambda sent: np.zeros_like(sent)  # cable delivers nothing
    passed, lines = eng.run_audio_path_test(MIC, CABLE_IN, eng.VoiceParameters())
    assert not passed and lines[0].startswith("[FAIL] Mic") and lines[-1].startswith("[FAIL] Cable: nothing arrived")
    fake_sd.rec_result = mic_take(0.3)
    passed, lines = eng.run_audio_path_test(MIC, SPEAKERS, eng.VoiceParameters())
    assert passed and lines[-1].startswith("[SKIP]")  # not VB-CABLE: the game-side check cannot run
    fake_sd.playrec_fn = lambda sent: (0.5 * np.sign(np.sin(2 * np.pi * 333 * np.arange(len(sent)) / 48000))).astype(np.float32)
    passed, lines = eng.run_audio_path_test(MIC, CABLE_IN, eng.VoiceParameters())
    assert not passed and "distorted" in lines[-1]
    fake_sd.devices[CABLE_OUT]["max_input_channels"] = 0
    passed, lines = eng.run_audio_path_test(MIC, CABLE_IN, eng.VoiceParameters())
    assert not passed and "could not find" in lines[-1]


# ------------------------------------------------------------------------------------------------ hotkey listener
class FakeUser32:
    """RegisterHotKey / UnregisterHotKey as Windows does them: one owner per combination."""

    def __init__(self, taken=()):
        self.taken = set(taken)        # combinations another program holds
        self.mine = {}                 # hk_id -> (mods, vk) registered by the listener
        self.calls = 0

    def RegisterHotKey(self, hwnd, hk_id, mods, vk):  # noqa: N802
        self.calls += 1
        combo = (mods & ~eng.Win32HotkeyListener.MOD_NOREPEAT, vk)
        assert mods & eng.Win32HotkeyListener.MOD_NOREPEAT
        if combo in self.taken or combo in self.mine.values() or hk_id in self.mine:
            return False
        self.mine[hk_id] = combo
        return True

    def UnregisterHotKey(self, hwnd, hk_id):  # noqa: N802
        return self.mine.pop(hk_id, None) is not None


@pytest.fixture
def listener(monkeypatch):
    monkeypatch.setattr(ctypes, "get_last_error", lambda: eng.Win32HotkeyListener.ERROR_HOTKEY_ALREADY_REGISTERED, raising=False)
    results = []
    lst = eng.Win32HotkeyListener(lambda _id: None, on_result=results.append)
    lst._owner = threading.get_ident()
    lst._user32 = FakeUser32()
    return lst, results


def combos_of(table):
    return {k: v for k, v in table.items() if v is not None}


@pytest.mark.parametrize("seed", SEEDS)
def test_hotkey_sync_apply_suspend_resume(listener, seed):
    lst, results = listener
    rng = np.random.default_rng(seed)
    user32 = lst._user32
    assert lst._sync() == {}
    assert user32.mine == combos_of(lst._wanted)
    # a random new set of bindings, with one combination held by another program
    keys = [k for k in hk.KEYS if k in hk.STANDALONE_OK]
    table = {a.hk_id: (int(rng.choice([1, 2, 3, 6])), int(rng.choice(keys))) for a in hk.ACTIONS}
    seen = set()
    for hk_id, combo in list(table.items()):  # keep them distinct, like the bridge guarantees
        if combo in seen:
            table[hk_id] = None
        else:
            seen.add(combo)
    victim = next(i for i, c in table.items() if c is not None and c != lst._wanted.get(i))
    user32.taken.add(table[victim])
    before = dict(lst._wanted)
    refused, reverted = lst._apply(table, resume=False)
    assert victim in refused and victim in reverted
    taken_over = before[victim] in [c for i, c in lst._wanted.items() if i != victim]
    # put back on what worked before, unless another action took that over in the same change: then Not set
    assert lst._wanted[victim] == (None if taken_over else before[victim])
    wanted = [c for c in lst._wanted.values() if c is not None]
    assert len(wanted) == len(set(wanted)), "two actions on one combination"
    assert user32.mine == {i: c for i, c in combos_of(lst._wanted).items() if c not in user32.taken}
    lst._publish("tok", "apply", refused, reverted)
    assert results[-1]["token"] == "tok" and results[-1]["reverted"] == reverted
    if lst._wanted[victim] is None:  # Not set: nothing to report as registered or refused
        assert victim not in results[-1]["registered"]
    else:
        assert results[-1]["registered"][victim] is (lst._wanted[victim] in user32.mine.values())
    # suspend drops everything (so the Settings page can read a key press), resume brings it all back
    lst._suspended = True
    lst._sync()
    assert user32.mine == {}
    lst._suspended = False
    lst._sync()
    assert set(user32.mine.values()) == {c for c in combos_of(lst._wanted).values() if c not in user32.taken}


def test_refused_change_never_reverts_onto_a_taken_combination(listener):
    """Swap two actions while another program holds one action's new combination: the refused action must not be
    put back on the combination its partner just took (that made two actions want Ctrl+Alt+F1)."""
    lst, _ = listener
    lst._sync()
    f1, f2 = lst._wanted[101], lst._wanted[102]
    lst._user32.taken.add(f2)                       # another program grabs Ctrl+Alt+F2 ...
    lst._user32.UnregisterHotKey(None, 102)         # ... after we released it
    lst._active.pop(102)
    refused, reverted = lst._apply({101: f2, 102: f1}, resume=False)
    assert reverted == [101] and lst._wanted[101] is None and lst._wanted[102] == f1
    assert lst._user32.mine[102] == f1 and 101 not in lst._user32.mine


def test_hotkeys_can_swap_combinations(listener):
    lst, _ = listener
    lst._sync()
    a, b = lst._wanted[101], lst._wanted[102]
    refused, reverted = lst._apply({101: b, 102: a}, resume=False)
    assert refused == {} and reverted == []
    assert lst._user32.mine[101] == b and lst._user32.mine[102] == a


def test_hotkey_commands_run_on_the_listener_thread(listener):
    lst, results = listener
    lst._sync()
    lst.suspend(token=1)
    lst.resume(token=2)
    lst.apply({105: None}, token=3)
    lst._run_commands()
    assert [r["token"] for r in results] == [1, 2, 3]
    assert results[0]["suspended"] and not results[1]["suspended"]
    assert results[2]["table"][105] is None and 105 not in lst._user32.mine
    lst._owner = -1  # a call from another thread is refused loudly, never half done
    with pytest.raises(RuntimeError):
        lst._register(101, (2, 0x70))


def test_a_bug_in_a_command_does_not_end_the_pump(listener):
    lst, results = listener
    lst._post(("apply", 9, {101: "garbage"}, False))
    lst._run_commands()
    assert results[-1]["error"] and results[-1]["token"] == 9


@pytest.mark.skipif(sys.platform == "win32", reason="the real message pump would run")
def test_listener_off_windows_just_reports_ready():
    lst = eng.Win32HotkeyListener(lambda _id: None)
    lst.start()
    assert lst.ready.wait(2)
    lst.join(2)
    assert not lst.is_alive()
    lst.stop()


# ------------------------------------------------------------------------------------------------ profiles
@pytest.mark.parametrize("seed", SEEDS)
def test_profile_round_trip(tmp_path, seed):
    rng = np.random.default_rng(seed)
    cfg = eng.AppConfig(
        block_size=int(rng.choice([128, 256, 512])),
        input_device_name=f"Mic {seed} [Windows WASAPI]",
        hear_myself=bool(rng.integers(2)),
        monitor_volume=float(rng.uniform(0, 1)),
        active_preset=list(eng.TACTICAL_PRESETS)[seed % 8],
        lite_mode=bool(rng.integers(2)),
        current_params=eng.VoiceParameters(**{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in eng.PARAM_RANGES.items()}),
    )
    cfg.hotkeys["bypass"] = "Ctrl+Shift+B"
    path = tmp_path / "p.json"
    eng.save_profile(cfg, path)
    loaded = eng.AppConfig()
    assert eng.load_profile_into(loaded, path)
    assert asdict(loaded) == asdict(cfg)
    assert [p.name for p in tmp_path.iterdir()] == ["p.json"]  # no temp file left behind


JUNK = [None, True, False, 0, -1, 1.5, 1e300, float("nan"), "x", "", [], {}, [1, 2], {"a": 1}, 999999, -999999]


@pytest.mark.parametrize("seed", SEEDS)
def test_profile_garbage_never_reaches_the_engine(tmp_path, seed):
    """Hand-edited or damaged profiles: wrong types are skipped, numbers clamped, nothing applied half-way."""
    rng = np.random.default_rng(seed)
    raw = {}
    for f in fields(eng.AppConfig):
        if rng.random() < 0.8:
            raw[f.name] = JUNK[int(rng.integers(len(JUNK)))]
    raw["current_params"] = {k: JUNK[int(rng.integers(len(JUNK)))] for k in eng.PARAM_RANGES}
    raw["current_params"]["bogus"] = 1
    raw["unknown_setting"] = 5
    path = tmp_path / "junk.json"
    path.write_text(json.dumps(raw, allow_nan=True))
    cfg = eng.AppConfig()
    defaults = asdict(eng.AppConfig())
    assert eng.load_profile_into(cfg, path) is True
    for f in fields(eng.AppConfig):
        value, default = getattr(cfg, f.name), defaults[f.name]
        if f.name == "current_params":
            for k, (lo, hi) in eng.PARAM_RANGES.items():
                v = getattr(value, k)
                assert isinstance(v, float) and lo <= v <= hi and np.isfinite(v)
        elif f.name == "hotkeys":
            assert set(value) == {a.id for a in hk.ACTIONS}
        else:
            assert type(value) is type(default), (f.name, value)
    assert 32 <= cfg.block_size <= eng.MAX_BLOCK_SIZE and 16000 <= cfg.sample_rate <= 96000
    assert 0.0 <= cfg.monitor_volume <= 1.0


@pytest.mark.parametrize("content", [b"", b"{", b"[]", b"null", b"\xff\xfe", b'"text"', b"[" * 5000])
def test_unreadable_profiles_change_nothing(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    cfg = eng.AppConfig()
    before = asdict(cfg)
    assert eng.load_profile_into(cfg, path) is False
    assert asdict(cfg) == before
    assert eng.load_profile_into(cfg, tmp_path / "missing.json") is False


def test_profile_with_bom_and_old_hotkeyless_format(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"active_preset": "Tactical Comms (Radio)", "block_size": 512}), encoding="utf-8-sig")
    cfg = eng.AppConfig()
    cfg.hotkeys["mute"] = "Ctrl+Alt+M"
    assert eng.load_profile_into(cfg, path)
    assert cfg.active_preset == "Tactical Comms (Radio)" and cfg.block_size == 512
    assert cfg.hotkeys["mute"] == "Ctrl+Alt+M"  # no hotkeys in the file: the current ones stay


def test_save_profile_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    eng.save_profile(eng.AppConfig(block_size=128), path)
    good = path.read_text()

    def disk_full(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(eng.os, "replace", disk_full)
    with pytest.raises(OSError):
        eng.save_profile(eng.AppConfig(block_size=512), path)
    assert path.read_text() == good
    assert [p.name for p in tmp_path.iterdir()] == ["p.json"]


# ------------------------------------------------------------------------------------------------ misc
def test_prewarm_pins_every_preset():
    eng._PINNED_DESIGNS.clear()
    extra = [eng.VoiceParameters(hpf_cutoff_hz=333.0, lpf_cutoff_hz=4321.0)]
    eng.prewarm_filter_designs(48000, extra=extra)
    chain = eng.RealTimeDSPChain(48000)
    pinned = dict(eng._PINNED_DESIGNS)
    for p in list(eng.TACTICAL_PRESETS.values()) + extra:
        snap = chain._compile_snapshot(p)
        assert any(snap.band_sos is v for v in pinned.values())


@pytest.mark.skipif(sys.platform == "win32", reason="it would really pin the test process to a few cores")
def test_process_isolation_is_a_no_op_off_windows():
    eng.configure_windows_process_isolation()  # must not raise
    assert eng.allowed_cpu_count() == len(os.sched_getaffinity(0))


def test_wasapi_settings_only_for_wasapi(fake_sd):
    assert isinstance(eng.wasapi_shared_settings(MIC), eng.sd.WasapiSettings)
    assert eng.wasapi_shared_settings(6) is None  # MME device
    assert eng.wasapi_shared_settings(99) is None  # unknown device
