"""ONNXVoiceStreamer: f0 extraction, model loading, the asynchronous RVC worker and its rings, with real ONNX Runtime
sessions on the synthetic models from onnx_models.py."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from scipy import signal

import engine as eng
from conftest import ITERATIONS, SEEDS

SR16 = 16000


def voice16(f, rng, kind="harm", n=2880, level=0.3):
    t = np.arange(n) / SR16
    if kind == "sine":
        x = np.sin(2 * np.pi * f * t)
    elif kind == "harm":
        x = sum((1 / k) * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28)) for k in range(1, int(7000 / f)))
    elif kind == "vowel":  # glottal pulses through three formant resonators
        x = (np.diff(np.floor(t * f), prepend=0) > 0).astype(float)
        for fc, bw in ((700, 110), (1220, 120), (2600, 160)):
            r = np.exp(-np.pi * bw / SR16)
            x = signal.lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / SR16), r * r], x)
    else:  # vibrato
        ph = 2 * np.pi * np.cumsum(f * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))) / SR16
        x = sum((0.6**k) * np.sin(k * ph) for k in range(1, 6))
    x = x / (np.max(np.abs(x)) + 1e-9) * level
    return (x + 0.003 * rng.standard_normal(n)).astype(np.float32)


def tone_freq(y, sr=48000):
    seg = y * np.hanning(len(y))
    return float(np.argmax(np.abs(np.fft.rfft(seg, 1 << 18)))) * sr / (1 << 18)


def streamed_freq(out, first, sr=48000):
    """The stand-in generator restarts its sine at every 160 ms chunk, so a whole-signal FFT shows lines every 6.25 Hz
    and its peak can snap to one of them. Measure each chunk's steady middle (after the 20 ms crossfade) instead."""
    hop, overlap = eng.ONNXVoiceStreamer.HOP_SAMPLES, eng.ONNXVoiceStreamer.OVERLAP_SAMPLES
    found = []
    start = first + overlap + 400
    while start + hop - overlap - 800 <= len(out):
        seg = out[start : start + hop - overlap - 800]
        if np.abs(seg).max() > 0.05:
            found.append(tone_freq(seg, sr))
        start += hop
    return float(np.median(found)) if found else 0.0


# ------------------------------------------------------------------------------------------------ f0
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", ["sine", "harm", "vowel", "vibrato"])
def test_f0_tracks_voices_from_65_to_800_hz(seed, kind):
    """The old autocorrelation picked the shortest lag for voices strong in their fundamental: 80 Hz read as 1143 Hz."""
    rng = np.random.default_rng(seed)
    streamer = eng.ONNXVoiceStreamer(48000)
    f = float(rng.uniform(65, 800))
    nsff0, pitch = streamer._extract_f0(voice16(f, rng, kind, level=float(rng.uniform(0.05, 0.6))), target_len=18)
    est = nsff0.ravel()
    assert nsff0.shape == (1, 18) and nsff0.dtype == np.float32
    assert pitch.shape == (1, 18) and pitch.dtype == np.int64
    good = np.abs(est - f) < 0.03 * f
    # The first three frames share one analysis window (it cannot start before the chunk), so a single wrong window
    # costs three frames: allowed once (it happens at the onset of ~0.25 % of high, sample-quantised synthetic vowels).
    assert good.sum() >= 15, (kind, f, est)
    assert np.all(est < 1.5 * f), "an octave-up error"


def test_f0_accuracy_over_many_voices():
    """In aggregate (4 kinds x 4 * ITERATIONS voices): the old autocorrelation got 84 % of frames right."""
    streamer = eng.ONNXVoiceStreamer(48000)
    rng = np.random.default_rng(12345)
    ok = total = 0
    for _ in range(4 * ITERATIONS):
        for kind in ("sine", "harm", "vowel", "vibrato"):
            f = float(rng.uniform(65, 800))
            est = streamer._extract_f0(voice16(f, rng, kind, level=float(rng.uniform(0.05, 0.6))), 18)[0].ravel()
            ok += int(np.sum(np.abs(est - f) < 0.03 * f))
            total += len(est)
    assert ok / total >= 0.995, ok / total


@pytest.mark.parametrize("seed", SEEDS)
def test_f0_is_zero_for_silence_and_noise(seed):
    rng = np.random.default_rng(seed)
    streamer = eng.ONNXVoiceStreamer(48000)
    assert not streamer._extract_f0(np.zeros(2880, np.float32), 18)[0].any()
    noise = (rng.standard_normal(2880) * rng.uniform(0.01, 0.3)).astype(np.float32)
    assert np.mean(streamer._extract_f0(noise, 18)[0] > 0) <= 0.2


@pytest.mark.parametrize("seed", SEEDS)
def test_f0_semitone_shift_and_coarse_bins(seed):
    rng = np.random.default_rng(seed)
    streamer = eng.ONNXVoiceStreamer(48000)
    f = float(rng.uniform(100, 300))
    semis = float(rng.choice([-1, 1]) * rng.uniform(0.1, 12))
    x = voice16(f, rng)
    base = streamer._extract_f0(x, 18)[0]
    assert np.array_equal(streamer._extract_f0(x, 18, semitones=0.04)[0], base)  # +-0.05 st dead zone
    nsff0, pitch = streamer._extract_f0(x, 18, semitones=semis)
    assert np.allclose(nsff0, base * 2 ** (semis / 12), rtol=1e-5)
    # RVC's coarse pitch: mel scale from 50 Hz (bin 1) to 1100 Hz (bin 255); unvoiced is bin 1
    mel = 1127 * np.log(1 + nsff0 / 700)
    lo, hi = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    ref = np.clip(np.round((mel - lo) * 254 / (hi - lo) + 1), 1, 255)
    ref[nsff0 <= 0] = 1
    assert np.array_equal(pitch, ref.astype(np.int64))


@pytest.mark.parametrize("seed", SEEDS)
def test_f0_any_length(seed):
    rng = np.random.default_rng(seed)
    streamer = eng.ONNXVoiceStreamer(48000)
    n = int(rng.integers(400, 8000))
    frames = int(rng.integers(1, 60))
    nsff0, pitch = streamer._extract_f0(voice16(150, rng, n=n), target_len=frames)
    assert nsff0.shape == pitch.shape == (1, frames)
    assert np.all(np.isfinite(nsff0))
    nsff0, _ = streamer._extract_f0(voice16(150, rng, n=n))  # default frame count
    assert nsff0.shape == (1, max(1, n // 320))


# ------------------------------------------------------------------------------------------------ loading
def test_load_failures_are_reported(onnx_models, tmp_path):
    streamer = eng.ONNXVoiceStreamer(48000)
    assert streamer.load_model(str(tmp_path / "missing.onnx")) == (False, "Model file not found.")
    ok, msg = streamer.load_model(str(onnx_models["rvc_no_hubert"]))
    assert not ok and msg.startswith("NEED_HUBERT")
    ok, msg = streamer.load_model(str(onnx_models["rvc_unknown_input"]))
    assert not ok and "'mystery'" in msg, msg
    ok, msg = streamer.load_model(str(onnx_models["rvc_wrong_dim"]))  # reported at load, not as endless silence
    assert not ok and "test chunk" in msg, msg
    assert not streamer.active and streamer._worker_thread is None
    garbage = tmp_path / "garbage.onnx"
    garbage.write_bytes(b"not a model at all")
    ok, msg = streamer.load_model(str(garbage))
    assert not ok and msg


@pytest.mark.parametrize("model", ["rvc_okada", "rvc_webui", "rvc_fp16"])
def test_rvc_exports_convert_at_the_right_pitch(onnx_models, model):
    """Three export conventions (nsff0/sid, pitchf/ds/rnd, fp16 features): each converts, at the input pitch."""
    streamer = eng.ONNXVoiceStreamer(48000)
    ok, msg = streamer.load_model(str(onnx_models[model]))
    assert ok, msg
    assert streamer.is_rvc and streamer.active and msg.startswith("RVC Loaded")
    try:
        rng = np.random.default_rng(0)
        for _ in range(ITERATIONS):
            f = float(rng.uniform(80, 400))
            semis = float(rng.uniform(-7, 7))
            streamer.set_pitch(semis)
            x = (0.2 * signal.sawtooth(2 * np.pi * f * np.arange(eng.ONNXVoiceStreamer.CHUNK_SAMPLES) / 48000)).astype(np.float32)
            y = streamer._convert(x)
            assert len(y) == eng.ONNXVoiceStreamer.CHUNK_SAMPLES and y.dtype == np.float32
            want = f * 2 ** (semis / 12)
            assert abs(tone_freq(y[960:-960]) - want) / want < 0.02
    finally:
        streamer.unload()


def test_direct_model_runs_in_the_callback(onnx_models):
    streamer = eng.ONNXVoiceStreamer(48000)
    ok, msg = streamer.load_model(str(onnx_models["direct"]))
    assert ok and not streamer.is_rvc and msg.startswith("Direct")
    rng = np.random.default_rng(0)
    for _ in range(ITERATIONS):
        x = rng.standard_normal(256).astype(np.float32)
        expected = x * 0.5
        streamer.process_into(x)
        assert np.allclose(x, expected, atol=1e-6)  # the model halves its input; the newest 256 samples come back
    streamer.unload()


def test_session_threads_follow_the_allowed_cores():
    rt = eng._load_onnxruntime()
    opts = eng.ONNXVoiceStreamer._session_options(rt)
    assert opts.intra_op_num_threads == min(8, max(2, eng.allowed_cpu_count()))
    assert eng.allowed_cpu_count() >= 1


# ------------------------------------------------------------------------------------------------ the worker
def _stream(streamer, signal_48k, block=256, realtime=True):
    out = np.zeros_like(signal_48k)
    for i in range(0, len(signal_48k) - block + 1, block):
        x = signal_48k[i : i + block].copy()
        streamer.process_into(x)
        out[i : i + block] = x
        if realtime:
            time.sleep(block / 48000 * 0.8)
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_worker_streams_converted_voice_and_never_leaks_the_raw_one(onnx_models, seed):
    rng = np.random.default_rng(seed)
    streamer = eng.ONNXVoiceStreamer(48000)
    assert streamer.load_model(str(onnx_models["rvc_okada"]))[0]
    try:
        f = float(rng.uniform(90, 300))
        semis = float(rng.choice([0.0, rng.uniform(-6, 6)]))
        streamer.set_pitch(semis)
        n = 48000
        raw = (0.2 * signal.sawtooth(2 * np.pi * f * np.arange(n) / 48000)).astype(np.float32)
        out = _stream(streamer, raw)
        first = np.flatnonzero(np.abs(out) > 1e-6)
        assert first.size, "nothing was converted"
        assert not out[: first[0]].any()  # before the first converted chunk: exact silence, not the microphone
        assert first[0] >= eng.ONNXVoiceStreamer.CHUNK_SAMPLES - 256  # it cannot be ready before one chunk was heard
        assert len(out) - first[0] > 2 * eng.ONNXVoiceStreamer.HOP_SAMPLES
        want = f * 2 ** (semis / 12)
        assert abs(streamed_freq(out, first[0]) - want) / want < 0.02
        assert streamer.error_count == 0, streamer.last_error
    finally:
        streamer.unload()


def test_worker_goes_quiet_after_speech_and_fades(onnx_models):
    streamer = eng.ONNXVoiceStreamer(48000)
    assert streamer.load_model(str(onnx_models["rvc_okada"]))[0]
    try:
        for _ in range(ITERATIONS // 4 or 1):
            speech = (0.2 * signal.sawtooth(2 * np.pi * 150 * np.arange(24000) / 48000)).astype(np.float32)
            silence = np.zeros(48000, np.float32)
            out = _stream(streamer, np.concatenate([speech, silence]))
            assert np.abs(out[24000:36000]).max() > 0.05  # the hangover keeps the end of the word
            assert np.abs(out[-4800:]).max() < 1e-6       # then silence
    finally:
        streamer.unload()


@pytest.mark.parametrize("seed", SEEDS)
def test_reloading_never_leaves_two_workers(onnx_models, seed):
    """The old shared stop event let a worker still inside a slow inference run on after a reload, next to the new
    one, both consuming the same SPSC ring."""
    streamer = eng.ONNXVoiceStreamer(48000)
    assert streamer.load_model(str(onnx_models["rvc_okada"]))[0]
    real_convert = streamer._convert
    gate = threading.Event()

    def slow_convert(chunk):
        gate.wait(1.0)  # hold the first worker inside "inference"
        return real_convert(chunk)

    streamer._convert = slow_convert
    try:
        streamer.process_into(np.full(eng.ONNXVoiceStreamer.CHUNK_SAMPLES + 256, 0.1, np.float32))
        time.sleep(0.05 + 0.002 * (seed % 5))
        old_worker = streamer._worker_thread
        streamer._start_worker()  # what a reload does
        gate.set()
        old_worker.join(2.0)
        assert not old_worker.is_alive()
        workers = [t for t in threading.enumerate() if t.name == "vcomms-rvc-worker" and t.is_alive()]
        assert workers == [streamer._worker_thread]
    finally:
        streamer.unload()
    time.sleep(0.05)
    assert streamer._worker_thread is None


def test_worker_records_errors_instead_of_hiding_them(onnx_models):
    streamer = eng.ONNXVoiceStreamer(48000)
    assert streamer.load_model(str(onnx_models["rvc_okada"]))[0]

    def broken(_chunk):
        raise RuntimeError("GPU lost")

    streamer._convert = broken
    try:
        streamer.process_into(np.full(eng.ONNXVoiceStreamer.CHUNK_SAMPLES + 256, 0.1, np.float32))
        deadline = time.time() + 2
        while streamer.error_count == 0 and time.time() < deadline:
            time.sleep(0.01)
        assert streamer.error_count >= 1 and "GPU lost" in streamer.last_error
        assert streamer._worker_thread.is_alive()  # it keeps going
    finally:
        streamer.unload()


@pytest.mark.parametrize("seed", SEEDS)
def test_unload_is_quick_and_complete(onnx_models, seed):
    streamer = eng.ONNXVoiceStreamer(48000)
    assert streamer.load_model(str(onnx_models["rvc_webui" if seed % 2 else "rvc_okada"]))[0]
    t0 = time.perf_counter()
    streamer.unload()
    assert time.perf_counter() - t0 < 0.5
    assert not streamer.active and streamer.session is None and streamer.hubert_session is None
    assert streamer._rvc_inputs == [] and not streamer.is_rvc
    x = np.ones(256, np.float32)
    streamer.process_into(x)  # unloaded: leaves the block alone (the DSP chain decides to mute)
    assert np.all(x == 1)
