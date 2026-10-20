"""The real-time DSP blocks of engine.py, each run for every seed in SEEDS (20 by default)."""

from __future__ import annotations

import tracemalloc
from dataclasses import asdict

import numpy as np
import pytest
from scipy import signal

import engine as eng
from conftest import SEEDS

SR = 48000


def tone(freq, n, sr=SR, amp=0.3, phase=0.0):
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / sr + phase)).astype(np.float32)


def dominant_freq(x, sr=SR):
    seg = x * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(seg, 1 << 18))
    return float(np.argmax(spec)) * sr / (1 << 18)


def random_params(rng) -> eng.VoiceParameters:
    values = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in eng.PARAM_RANGES.items()}
    return eng.VoiceParameters(name="random", **values)


# ------------------------------------------------------------------------------------------------ Pade tanh
@pytest.mark.parametrize("seed", SEEDS)
def test_pade_tanh_matches_formula_and_stays_bounded(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, eng.MAX_BLOCK_SIZE))
    x = (rng.standard_normal(n) * rng.uniform(0.1, 10)).astype(np.float32)
    ref = np.clip(x.astype(np.float64), -3, 3)
    ref = ref * (27 + ref**2) / (27 + 9 * ref**2)
    y = x.copy()
    eng.fast_pade_tanh_inplace(y, np.empty(n, np.float32), np.empty(n, np.float32))
    assert y.dtype == np.float32
    assert np.allclose(y, ref, atol=1e-6)
    assert np.all(np.abs(y) <= 1.0 + 1e-6)
    assert np.max(np.abs(y - np.tanh(np.clip(x, -3, 3)))) < 0.025  # a close tanh approximation
    order = np.argsort(x)
    assert np.all(np.diff(y[order]) >= -1e-6)  # monotonic


# ------------------------------------------------------------------------------------------------ SPSC ring buffer
@pytest.mark.parametrize("seed", SEEDS)
def test_spsc_steady_state_is_a_delayed_copy(seed):
    rng = np.random.default_rng(seed)
    block = int(rng.choice([64, 128, 256, 512]))
    ring = eng.WaitFreeSpscRingBuffer(max_lag_samples=4 * block, target_cushion=block)
    gain = float(rng.uniform(0.1, 1.0))
    stream = (rng.standard_normal(block * 200) * 0.3).astype(np.float32)
    out = np.zeros_like(stream)
    for i in range(200):  # many wraps around the 16384-sample array
        ring.write(stream[i * block : (i + 1) * block], gain=gain)
        ring.read_into(out[i * block : (i + 1) * block])
    assert np.allclose(out, stream * gain, atol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_spsc_never_invents_samples_under_random_jitter(seed):
    """Random write and read sizes (clock drift, jitter): everything read is silence or a sample that was written, in
    stream order, never one from a lap ago, and the cursors never cross. A second ring fed 1.0 on the same schedule
    reveals the underrun fade factors, so the positions can be recovered exactly."""
    rng = np.random.default_rng(seed)
    ring = eng.WaitFreeSpscRingBuffer(max_lag_samples=1536, target_cushion=512)
    ones = eng.WaitFreeSpscRingBuffer(max_lag_samples=1536, target_cushion=512)
    counter = 1
    last_seen = 0
    for _ in range(400):
        for _ in range(int(rng.integers(0, 3))):
            n = int(rng.integers(1, 2049))
            ring.write(np.arange(counter, counter + n, dtype=np.float64).astype(np.float32))  # value = stream position
            ones.write(np.ones(n, np.float32))
            counter += n
        size = int(rng.integers(1, 2049))  # reads larger than the 512 cushion are the case the drift guard got wrong
        out, factor = np.zeros(size, np.float32), np.zeros(size, np.float32)
        ring.read_into(out)
        ones.read_into(factor)
        live = factor > 0
        pos = np.round(out[live].astype(np.float64) / factor[live])
        assert np.all(pos >= 1), "a sample nobody wrote"
        if pos.size:
            assert np.all(np.diff(pos) == 1), "samples skipped or out of order inside one read"
            assert pos[0] > last_seen, "a sample from the past came back"
            last_seen = pos[-1]
        assert 0 <= ring.available_read() == ones.available_read()


@pytest.mark.parametrize("seed", SEEDS)
def test_spsc_peek_advance_skip(seed):
    rng = np.random.default_rng(seed)
    ring = eng.WaitFreeSpscRingBuffer(capacity=4096)
    data = rng.standard_normal(3000).astype(np.float32)
    start = int(rng.integers(0, 4096))
    ring._write_pos = ring._read_pos = start  # start anywhere, so peeks cross the wrap
    ring.write(data)
    k = int(rng.integers(1, 1000))
    assert np.array_equal(ring.peek_chunk(k), data[:k])
    assert ring.available_read() == 3000
    off = int(rng.integers(0, 1000))
    assert np.array_equal(ring.peek_chunk(k, offset=off), data[off : off + k])
    ring.advance_read(k)
    assert np.array_equal(ring.read_chunk(10), data[k : k + 10])
    keep = int(rng.integers(1, 500))
    ring.skip_to_latest(keep)
    assert ring.available_read() == keep
    assert np.array_equal(ring.peek_chunk(keep), data[-keep:])


@pytest.mark.parametrize("seed", SEEDS)
def test_spsc_underrun_fades_and_reprimes(seed):
    rng = np.random.default_rng(seed)
    ring = eng.WaitFreeSpscRingBuffer(max_lag_samples=1536, target_cushion=512)
    out = np.zeros(256, np.float32)
    ring.write(np.ones(511, np.float32))
    ring.read_into(out)
    assert not out.any(), "played before the cushion was there"
    ring.write(np.ones(1, np.float32))
    ring.read_into(out)
    assert np.all(out == 1.0)
    partial = int(rng.integers(1, 255))
    ring.read_into(out)  # the last 256 of the cushion: empty now
    ring.write(np.ones(partial, np.float32))
    ring.read_into(out)  # only `partial` there: micro-underrun
    fade = min(16, partial)
    assert np.all(out[: partial - fade] == 1.0)
    assert np.allclose(out[partial - fade : partial], np.linspace(1, 0, fade))
    assert not out[partial:].any()
    assert ring.available_read() == 0
    for _ in range(101):  # sustained starvation unprimes
        ring.read_into(out)
    assert not ring._primed


def test_spsc_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        eng.WaitFreeSpscRingBuffer(capacity=1000)


@pytest.mark.parametrize("seed", SEEDS)
def test_spsc_underrun_path_allocates_nothing(seed):
    ring = eng.WaitFreeSpscRingBuffer(max_lag_samples=1536, target_cushion=64)
    out = np.zeros(256, np.float32)
    ring.write(np.ones(64 + seed % 150, np.float32))  # fewer than one 256 read: the partial-read path
    tracemalloc.start()
    ring.read_into(out)  # partial read with fade
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 512, f"{peak} bytes allocated in an underrun"


# ------------------------------------------------------------------------------------------------ mirror buffer
@pytest.mark.parametrize("seed", SEEDS)
def test_mirror_buffer_view_is_the_last_window(seed):
    rng = np.random.default_rng(seed)
    w = int(rng.choice([256, 1000, 4096]))
    mirror = eng.ZeroRollMirrorBuffer(window_size=w)
    history = np.zeros(w, np.float32)
    for _ in range(60):
        n = int(rng.integers(1, 2 * w))
        block = rng.standard_normal(n).astype(np.float32)
        view = mirror.push_and_get_view(block)
        history = np.concatenate([history, block])[-w:]
        assert view.flags["C_CONTIGUOUS"]
        assert view.base is not None  # a view into the mirror, not a copy
        assert np.array_equal(view, history)


# ------------------------------------------------------------------------------------------------ pitch shifter
def test_pitch_shifter_bypass_below_threshold():
    ps = eng.BitwiseCircularPitchShifter(SR)
    x = tone(220, 256)
    y = x.copy()
    ps.process_into(y, 0.04)
    assert np.array_equal(x, y)
    assert ps.write_idx == 256


def band_centroid(x, centre, sr=SR):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 1 << 18)) ** 2
    freqs = np.fft.rfftfreq(1 << 18, 1 / sr)
    band = (freqs > centre / 1.25) & (freqs < centre * 1.25)
    return float(np.sum(freqs[band] * spec[band]) / np.sum(spec[band]))


@pytest.mark.parametrize("seed", SEEDS)
def test_pitch_shifter_moves_the_pitch(seed):
    """A two-head delay line amplitude-modulates its output at the head sweep rate (up to ~28 Hz), so the single
    strongest FFT bin can sit on a sideband; the energy centroid around the target is the honest measure."""
    rng = np.random.default_rng(seed)
    semis = float(rng.choice([-1, 1]) * rng.uniform(2.0, 12.0))
    f = float(rng.uniform(150, 600))
    ps = eng.BitwiseCircularPitchShifter(SR)
    x = tone(f, SR)  # one second
    y = x.copy()
    for i in range(0, SR - 255, 256):
        ps.process_into(y[i : i + 256], semis)
    want = f * 2 ** (semis / 12)
    got = band_centroid(y[SR // 4 :], want)
    # up to ~10 % for shifts near an octave down, in the original implementation as well (same numbers): sidebands
    assert abs(got - want) / want < 0.12, (f, semis, got, want)
    assert abs(got - want) < abs(got - f)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 0.3 * 1.01


class ReferencePitchShifter:
    """The original (v1.2.7) implementation, one head at a time with fancy indexing, kept as the reference the
    optimised shifter must reproduce."""

    def __init__(self, sr=SR):
        self.win = float(int(sr * 0.036))
        self.buf = np.zeros(4096, np.float32)
        self.w = 0
        self.phase = 0.0

    def head(self, w_idxs, phases):
        read = (w_idxs - (phases * np.float32(self.win)).astype(np.float32)).astype(np.float32)
        fl = np.floor(read)
        i0 = fl.astype(np.int32) & 4095
        frac = read - fl
        y0 = self.buf[i0]
        return y0 + (self.buf[(i0 + 1) & 4095] - y0) * frac

    def process(self, x, semis):
        n = len(x)
        w_idxs = (self.w + np.arange(n, dtype=np.int32)) & 4095
        self.buf[w_idxs] = x
        self.w = (self.w + n) & 4095
        if abs(semis) < 0.05:
            return x
        step = (1.0 - 2.0 ** (semis / 12.0)) / self.win
        pa = np.mod(np.arange(1, n + 1, dtype=np.float32) * np.float32(step) + np.float32(self.phase), 1.0).astype(np.float32)
        self.phase = float(pa[-1])
        pb = np.mod(pa + np.float32(0.5), 1.0).astype(np.float32)
        va, vb = self.head(w_idxs, pa), self.head(w_idxs, pb)
        gain = (0.5 * (1.0 - np.cos(pa * np.float32(2 * np.pi)))).astype(np.float32)
        return (vb + gain * (va - vb)).astype(np.float32)


@pytest.mark.parametrize("seed", SEEDS)
def test_pitch_shifter_matches_the_original_algorithm(seed):
    rng = np.random.default_rng(seed)
    ref, new = ReferencePitchShifter(), eng.BitwiseCircularPitchShifter(SR)
    semis = float(rng.uniform(-12, 12))
    for _ in range(40):
        n = int(rng.choice([64, 128, 256, 333, 512, 1024, 2048]))
        if rng.random() < 0.1:
            semis = float(rng.uniform(-12, 12))
        x = (rng.standard_normal(n) * 0.3).astype(np.float32)
        want = ref.process(x.copy(), semis)
        got = x.copy()
        new.process_into(got, semis)
        assert np.max(np.abs(got - want)) < 2e-3  # float32 rounding of the read position on full-scale noise
        assert new.write_idx == ref.w and abs(new.phase - ref.phase) < 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_pitch_shifter_is_block_size_invariant(seed):
    rng = np.random.default_rng(seed)
    semis = float(rng.uniform(-12, 12))
    x = (rng.standard_normal(8192) * 0.3).astype(np.float32)
    a = eng.BitwiseCircularPitchShifter(SR)
    b = eng.BitwiseCircularPitchShifter(SR)
    ya, yb = x.copy(), x.copy()
    for i in range(0, 8192, 2048):
        a.process_into(ya[i : i + 2048], semis)
    pos = 0
    while pos < 8192:
        n = min(int(rng.choice([64, 100, 128, 256, 333, 512])), 8192 - pos)
        b.process_into(yb[pos : pos + n], semis)
        pos += n
    assert np.max(np.abs(ya - yb)) < 2e-3  # float32 phase rounding only


@pytest.mark.parametrize("seed", SEEDS)
def test_pitch_shifter_allocates_no_sample_buffers(seed):
    ps = eng.BitwiseCircularPitchShifter(SR)
    n = int(np.random.default_rng(seed).choice([128, 256, 512]))
    x = tone(200, n)
    for _ in range(5):
        ps.process_into(x.copy(), -4.5)  # first call per size builds the cached views
    buf = x.copy()
    tracemalloc.start()
    for _ in range(50):
        buf[:] = x
        ps.process_into(buf, -4.5)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Allowed: NumPy's internal buffer for the one float -> index cast (8 bytes x 2n), which the original paid twice.
    # Not allowed: the four block-sized arrays fancy indexing used to create.
    assert peak <= 2 * n * 8 + 4096, f"{peak} bytes: more than the index cast buffer was allocated"


# ------------------------------------------------------------------------------------------------ reverb
def test_reverb_off_is_passthrough():
    r = eng.VectorizedDarkReverb(SR)
    x = tone(300, 256)
    y = x.copy()
    r.process_into(y, 0.0)
    assert np.array_equal(x, y)


@pytest.mark.parametrize("seed", SEEDS)
def test_reverb_is_stable_and_decays(seed):
    rng = np.random.default_rng(seed)
    r = eng.VectorizedDarkReverb(SR)
    mix = float(rng.uniform(0.05, 0.6))
    x = np.zeros(SR * 2, np.float32)
    x[:256] = rng.standard_normal(256).astype(np.float32) * 0.5
    for i in range(0, len(x), 256):
        r.process_into(x[i : i + 256], mix)
    assert np.all(np.isfinite(x))
    early = np.sqrt(np.mean(x[: SR // 4] ** 2))
    late = np.sqrt(np.mean(x[-SR // 4 :] ** 2))
    assert late < early * 1e-3


@pytest.mark.parametrize("seed", SEEDS)
def test_reverb_any_block_size_up_to_max(seed):
    """Blocks longer than the shortest comb (1116) used to crash (2048) or read samples not written yet (1117+)."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(4096) * 0.3).astype(np.float32)
    big = int(rng.choice([1117, 1500, 2048]))
    a, b = eng.VectorizedDarkReverb(SR), eng.VectorizedDarkReverb(SR)
    ya, yb = x.copy(), x.copy()
    for i in range(0, 4096, big):
        a.process_into(ya[i : i + big], 0.3)
    for i in range(0, 4096, 128):
        b.process_into(yb[i : i + 128], 0.3)
    assert np.array_equal(ya, yb)


@pytest.mark.parametrize("seed", SEEDS)
def test_reverb_other_sample_rates(seed):
    rate = [16000, 22050, 32000, 44100, 48000, 96000][seed % 6]
    r = eng.VectorizedDarkReverb(rate)
    x = (np.random.default_rng(seed).standard_normal(2048) * 0.3).astype(np.float32)
    r.process_into(x, 0.4)
    assert np.all(np.isfinite(x))


# ------------------------------------------------------------------------------------------------ filters
@pytest.mark.parametrize("seed", SEEDS)
def test_peaking_eq_hits_its_gain(seed):
    rng = np.random.default_rng(seed)
    f0, q, gain = float(rng.uniform(150, 4000)), float(rng.uniform(0.5, 3)), float(rng.uniform(-12, 12))
    sos = eng.biquad_peaking_eq(f0, q, gain, SR)
    assert sos.shape == (6,) and sos.dtype == np.float32
    w, h = signal.sosfreqz(sos.reshape(1, 6).astype(np.float64), worN=[0.0, f0, SR / 2 * 0.999], fs=SR)
    db = 20 * np.log10(np.abs(h))
    assert abs(db[1] - gain) < 0.05
    assert abs(db[0]) < 0.05


@pytest.mark.parametrize("seed", SEEDS)
def test_design_cache_never_evicts_what_it_just_stored(seed):
    """The sweep cache used to be cleared whole when full: with 63 entries, a snapshot's band design went in (64) and
    its formant design then cleared it, so the next snapshot with the same band redesigned it during the drag."""
    rng = np.random.default_rng(seed)
    chain = eng.RealTimeDSPChain(SR)
    eng._SWEEP_DESIGNS.clear()
    while len(eng._SWEEP_DESIGNS) < eng._SWEEP_DESIGN_LIMIT - 1 - seed % 2:
        chain._cached_design(("filler", len(eng._SWEEP_DESIGNS)), lambda: np.zeros((1, 6), np.float32))
    p = random_params(rng)
    p.hpf_cutoff_hz = 20.0 + 1e-3 * seed  # a band no earlier test has pinned
    a = chain._compile_snapshot(p)
    p.formant_semitones = 2.0 if p.formant_semitones < 1.0 else -2.0  # the formant slider moves, the band stays
    b = chain._compile_snapshot(p)
    assert a.band_sos is b.band_sos, "the band-pass was redesigned although its sliders did not move"
    assert len(eng._SWEEP_DESIGNS) <= eng._SWEEP_DESIGN_LIMIT


@pytest.mark.parametrize("seed", SEEDS)
def test_snapshot_designs_are_cached_and_bounded(seed):
    rng = np.random.default_rng(seed)
    chain = eng.RealTimeDSPChain(SR)
    p = random_params(rng)
    a = chain._compile_snapshot(p)
    b = chain._compile_snapshot(p)
    assert a.band_sos is b.band_sos and a.formant_sos is b.formant_sos  # reused, not redesigned
    for _ in range(100):
        chain._compile_snapshot(random_params(rng))
    assert len(eng._SWEEP_DESIGNS) <= eng._SWEEP_DESIGN_LIMIT
    chain.prepare_parameters(p)
    key = (SR, "band") + tuple(
        float(np.clip(v / (0.5 * SR), lo, hi)) for v, lo, hi in ((p.hpf_cutoff_hz, 0.001, 0.95), (p.lpf_cutoff_hz, 0.01, 0.98))
    )
    assert key in eng._PINNED_DESIGNS


# ------------------------------------------------------------------------------------------------ the whole chain
@pytest.mark.parametrize("seed", SEEDS)
def test_dsp_chain_random_settings_are_safe(seed):
    rng = np.random.default_rng(seed)
    chain = eng.RealTimeDSPChain(SR)
    params = random_params(rng)
    chain.update_parameters(params)
    n = int(rng.choice([128, 256, 512, 2048]))
    loud = rng.uniform(0.01, 4.0)  # includes input far over full scale
    for _ in range(40):
        x = (rng.standard_normal(n) * loud).astype(np.float32)
        before = x.copy()
        y = chain.process_block(x)
        assert np.array_equal(x, before), "process_block changed its input"
        assert y.dtype == np.float32 and len(y) == n
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) <= 0.891 + 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_every_preset_stays_under_the_ceiling(seed):
    rng = np.random.default_rng(seed)
    name = list(eng.TACTICAL_PRESETS)[seed % len(eng.TACTICAL_PRESETS)]
    chain = eng.RealTimeDSPChain(SR)
    chain.update_parameters(eng.TACTICAL_PRESETS[name])
    f = float(rng.uniform(90, 300))
    voice = sum((0.3 / k) * np.sin(2 * np.pi * f * k * np.arange(SR) / SR) for k in range(1, 12)).astype(np.float32)
    out = np.concatenate([chain.process_block(voice[i : i + 256]).copy() for i in range(0, SR, 256)])
    assert np.all(np.isfinite(out)) and np.max(np.abs(out)) <= 0.891 + 1e-6
    assert np.sqrt(np.mean(out[SR // 2 :] ** 2)) > 0.01, "the preset silenced a normal voice"


@pytest.mark.parametrize("seed", SEEDS)
def test_noise_gate_closes_on_silence_and_opens_on_voice(seed):
    rng = np.random.default_rng(seed)
    chain = eng.RealTimeDSPChain(SR)
    chain.update_parameters(eng.VoiceParameters(gate_threshold_db=-40.0))
    hiss = lambda: (rng.standard_normal(256) * 10 ** (-70 / 20)).astype(np.float32)  # well under the threshold
    for _ in range(80):
        y = chain.process_block(hiss())
    assert np.max(np.abs(y)) < 1e-5
    for _ in range(10):
        y = chain.process_block(tone(200, 256, amp=0.2))
    assert np.max(np.abs(y)) > 0.1


@pytest.mark.parametrize("seed", SEEDS)
def test_ai_voice_without_a_model_is_silent_never_raw(seed):
    chain = eng.RealTimeDSPChain(SR)
    x = (np.random.default_rng(seed).standard_normal(256) * 0.3).astype(np.float32)
    y = chain.process_block(x, use_onnx=True)
    assert not y.any()


@pytest.mark.parametrize("seed", SEEDS)
def test_parameter_swap_mid_stream(seed):
    """RCU swap between blocks: every block after the swap uses the new snapshot, and nothing glitches to NaN."""
    rng = np.random.default_rng(seed)
    chain = eng.RealTimeDSPChain(SR)
    names = list(eng.TACTICAL_PRESETS)
    for i in range(60):
        if i % 5 == 0:
            p = eng.preset_parameters(names[int(rng.integers(len(names)))])
            chain.update_parameters(p)
            assert chain._snapshot.params == p and chain._snapshot.params is not p  # an immutable copy
        y = chain.process_block(tone(150, 256))
        assert np.all(np.isfinite(y))


def test_dsp_chain_is_fast_enough():
    chain = eng.RealTimeDSPChain(SR)
    x = tone(140, 256)
    import time

    worst = []
    for name, p in eng.TACTICAL_PRESETS.items():
        chain.update_parameters(p)
        for _ in range(20):
            chain.process_block(x)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            chain.process_block(x)
            times.append(time.perf_counter() - t0)
        worst.append(np.percentile(times, 99) * 1000)
    assert max(worst) < 256 / SR * 1000  # inside the 5.33 ms deadline even on a loaded CI box


def test_preset_table_is_consistent():
    for name, p in eng.TACTICAL_PRESETS.items():
        assert p.name == name
        for key, (lo, hi) in eng.PARAM_RANGES.items():
            assert lo <= getattr(p, key) <= hi, (name, key)
    assert set(eng.PARAM_RANGES) == {k for k in asdict(eng.VoiceParameters()) if k != "name"}
    assert eng.HOTKEY_PRESET_COUNT == 4
    copy = eng.preset_parameters("Deep Shadow (Omen)")
    copy.pitch_semitones = 99
    assert eng.TACTICAL_PRESETS["Deep Shadow (Omen)"].pitch_semitones != 99


def test_production_self_test_passes(capsys):
    assert eng.run_production_self_test() == 0
    out = capsys.readouterr().out
    assert "ALL PRODUCTION CHECKS PASSED" in out and "Memory audit" in out
