"""
V-COMMS // Real-Time Low-Latency Tactical Voice Engine for Windows 11  (engine only, no UI)

Nothing in this module imports Qt. The audio engine, the Win32 hotkeys, the profile helpers and the
self-test run (and can be tested) without a window; the Fluent GUI reaches all of it through
``engine_bridge.EngineBridge``.

Architecture:
  - Lock-Free Read-Copy-Update (RCU) DSP Parameter Swapping
  - Power-of-Two Bitwise Circular Dual-Head Pitch Shifter (Zero Heap Allocations)
  - Wait-Free Single-Producer Single-Consumer (SPSC) Monitor Ring Buffer
  - Zero-Roll 2x Mirrored Sliding Window Buffer for ONNX DirectML/CUDA Inference
  - Vanguard-Safe Win32 RegisterHotKey API (Zero WH_KEYBOARD_LL Hooks)
  - Hybrid CPU Affinity Pinning (Targets E-Cores on Intel 12th-14th Gen HX CPUs)
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import importlib.util
import json
import os
import platform
import queue
import sys
from collections import OrderedDict
import tempfile
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

import hotkeys as hk

# NumPy and SciPy start a BLAS thread pool per logical CPU and reserve about 60 MB of address space for each: over 700 MB
# of committed memory on a 12-core PC, for nothing, because the DSP only handles 256-sample vectors, which BLAS never
# splits across threads. This has to be set before NumPy is imported (a variable the player already set wins).
for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import numpy as np
from scipy import signal
from scipy.signal._signaltools import _sosfilt
from scipy.signal._sigtools import _linear_filter

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):  # OSError: the module is there but the PortAudio library is not
    sd = None
    SOUNDDEVICE_AVAILABLE = False

STABILITY_LATENCY_S = 0.06  # suggested stream latency, in seconds, while the Stability buffer is on

# onnxruntime is only needed once an AI model is loaded, and importing it costs tens of megabytes, so it is looked for
# here but imported on first use (see _load_onnxruntime).
ONNX_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None
ort = None


def _load_onnxruntime():
    global ort
    if ort is None:
        import onnxruntime  # noqa: PLC0415

        ort = onnxruntime
    return ort


# ============================================================================
# 1. DATA MODELS & TACTICAL PRESETS
# ============================================================================

MAX_BLOCK_SIZE = 2048  # Pre-allocation ceiling for all scratch buffers


@dataclass
class VoiceParameters:
    name: str = "Custom"
    pitch_semitones: float = 0.0        # -12.0 to +12.0 semitones
    formant_semitones: float = 0.0      # -6.0 to +6.0 semitones
    gate_threshold_db: float = -48.0    # -70.0 to -15.0 dBFS
    hpf_cutoff_hz: float = 85.0         # High-pass rumble cut (Hz)
    lpf_cutoff_hz: float = 16000.0      # Low-pass cut (Hz)
    ring_mod_hz: float = 0.0            # 0.0 = Off, 30-90Hz = Cybernetic
    ring_mod_mix: float = 0.0           # 0.0 to 1.0
    reverb_mix: float = 0.0             # 0.0 to 0.60
    drive_db: float = 0.0               # 0.0 to 18.0 dB
    output_gain_db: float = 0.0         # -12.0 to +12.0 dB


# The slider range of every voice parameter (the Voice page uses the same numbers). Anything that arrives from outside
# the app, such as an imported preset, is clamped to these before it can reach the DSP.
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "pitch_semitones": (-12.0, 12.0),
    "formant_semitones": (-6.0, 6.0),
    "gate_threshold_db": (-70.0, -15.0),
    "hpf_cutoff_hz": (20.0, 600.0),
    "lpf_cutoff_hz": (1500.0, 16000.0),
    "ring_mod_hz": (0.0, 100.0),
    "ring_mod_mix": (0.0, 1.0),
    "reverb_mix": (0.0, 0.6),
    "drive_db": (0.0, 18.0),
    "output_gain_db": (-12.0, 12.0),
}


TACTICAL_PRESETS: Dict[str, VoiceParameters] = {
    "Deep Shadow (Omen)": VoiceParameters(
        name="Deep Shadow (Omen)",
        pitch_semitones=-4.5,
        formant_semitones=-2.8,
        gate_threshold_db=-54.0,
        hpf_cutoff_hz=75.0,
        lpf_cutoff_hz=6800.0,
        ring_mod_hz=36.0,
        ring_mod_mix=0.14,
        reverb_mix=0.22,
        drive_db=2.5,
        output_gain_db=2.0,
    ),
    "Tactical Comms (Radio)": VoiceParameters(
        name="Tactical Comms (Radio)",
        pitch_semitones=0.0,
        formant_semitones=0.5,
        gate_threshold_db=-52.0,
        hpf_cutoff_hz=420.0,
        lpf_cutoff_hz=2900.0,
        ring_mod_hz=0.0,
        ring_mod_mix=0.0,
        reverb_mix=0.0,
        drive_db=11.0,
        output_gain_db=-1.5,
    ),
    "Cybernetic Agent (KAY/O)": VoiceParameters(
        name="Cybernetic Agent (KAY/O)",
        pitch_semitones=-2.0,
        formant_semitones=-1.5,
        gate_threshold_db=-54.0,
        hpf_cutoff_hz=90.0,
        lpf_cutoff_hz=9500.0,
        ring_mod_hz=58.0,
        ring_mod_mix=0.38,
        reverb_mix=0.08,
        drive_db=4.0,
        output_gain_db=1.0,
    ),
    "Pitch Clean (Up / Helium-Lite)": VoiceParameters(
        name="Pitch Clean (Up / Helium-Lite)",
        pitch_semitones=3.5,
        formant_semitones=1.8,
        gate_threshold_db=-54.0,
        hpf_cutoff_hz=95.0,
        lpf_cutoff_hz=16000.0,
        ring_mod_hz=0.0,
        ring_mod_mix=0.0,
        reverb_mix=0.0,
        drive_db=0.0,
        output_gain_db=0.5,
    ),
    # Female voices. Appended after the four above so saved profiles and the Ctrl+Alt+F1..F4 hotkeys keep pointing at
    # the same presets. The pitch shifter moves formants by the same ratio as pitch, so each one trades a female pitch
    # against formants that sit higher than a real woman's (measured: 1.3x to 1.8x a male voice, against about 1.25x).
    "Female Natural (Clear)": VoiceParameters(
        name="Female Natural (Clear)",
        pitch_semitones=6.0,
        formant_semitones=1.5,
        gate_threshold_db=-54.0,
        hpf_cutoff_hz=120.0,
        lpf_cutoff_hz=10000.0,
        ring_mod_hz=0.0,
        ring_mod_mix=0.0,
        reverb_mix=0.0,
        drive_db=0.0,
        output_gain_db=1.0,
    ),
    "Female Bright (Young)": VoiceParameters(
        name="Female Bright (Young)",
        pitch_semitones=8.0,
        formant_semitones=3.0,
        gate_threshold_db=-54.0,
        hpf_cutoff_hz=150.0,
        lpf_cutoff_hz=13000.0,
        ring_mod_hz=0.0,
        ring_mod_mix=0.0,
        reverb_mix=0.0,
        drive_db=0.0,
        output_gain_db=0.5,
    ),
    "Female Husky (Low)": VoiceParameters(
        name="Female Husky (Low)",
        pitch_semitones=4.5,
        formant_semitones=-1.5,
        gate_threshold_db=-54.0,
        hpf_cutoff_hz=100.0,
        lpf_cutoff_hz=6500.0,
        ring_mod_hz=0.0,
        ring_mod_mix=0.0,
        reverb_mix=0.06,
        drive_db=3.0,
        output_gain_db=1.0,
    ),
    "Female Cute (Airy)": VoiceParameters(
        name="Female Cute (Airy)",
        pitch_semitones=10.0,
        formant_semitones=4.5,
        gate_threshold_db=-54.0,
        hpf_cutoff_hz=180.0,
        lpf_cutoff_hz=15000.0,
        ring_mod_hz=0.0,
        ring_mod_mix=0.0,
        reverb_mix=0.05,
        drive_db=0.0,
        output_gain_db=0.0,
    ),
}
HOTKEY_PRESET_COUNT = 4  # the first four presets have Ctrl+Alt+F1..F4; the rest are picked from the window


@dataclass
class AppConfig:
    sample_rate: int = 48000
    block_size: int = 256
    input_device_name: str = ""
    output_device_name: str = ""
    monitor_device_name: str = ""
    hear_myself: bool = False
    monitor_volume: float = 0.75
    bypass_effects: bool = False
    mic_muted: bool = False
    active_preset: str = "Deep Shadow (Omen)"
    onnx_model_path: str = ""
    enable_onnx: bool = False
    lite_mode: bool = False          # for older PCs: slower meters, no fades or Mica, memory trimmed while hidden
    stability_buffer: bool = False   # asks Windows for a bigger audio buffer: fewer dropouts, a little more delay
    hotkeys: Dict[str, str] = field(default_factory=hk.default_bindings)  # action id -> "Ctrl+Alt+F1" ("" = Not set)
    current_params: VoiceParameters = field(
        default_factory=lambda: VoiceParameters(**asdict(TACTICAL_PRESETS["Deep Shadow (Omen)"]))
    )


class DSPImmutableSnapshot(NamedTuple):
    """
    Read-Copy-Update (RCU) immutable state container.
    Created off-thread by the UI and swapped via a single atomic pointer assignment
    so the real-time audio callback NEVER acquires a mutex lock.
    """
    params: VoiceParameters
    band_sos: np.ndarray
    formant_sos: np.ndarray
    thresh_lin: float
    drive_gain: float
    drive_norm: float
    out_gain: float
    has_formant: bool


# ============================================================================
# 2. TASK-SPECIFIC ZERO-ALLOCATION DATA STRUCTURES
# ============================================================================

def fast_pade_tanh_inplace(x: np.ndarray, scratch_x2: np.ndarray, scratch_denom: np.ndarray) -> None:
    """
    In-place Rational Pade Approximant for tanh(x):
        tanh(x) ~= x * (27 + x^2) / (27 + 9 * x^2)  for |x| <= 3
    Executes in O(N) vectorized algebraic ops with ZERO heap allocations (~4x faster than np.tanh).
    """
    np.clip(x, -3.0, 3.0, out=x)
    np.multiply(x, x, out=scratch_x2)               # x^2
    np.multiply(scratch_x2, 9.0, out=scratch_denom)  # 9 * x^2
    np.add(scratch_denom, 27.0, out=scratch_denom)   # 27 + 9 * x^2
    np.add(scratch_x2, 27.0, out=scratch_x2)         # 27 + x^2
    np.multiply(x, scratch_x2, out=x)                # x * (27 + x^2)
    np.divide(x, scratch_denom, out=x)               # x * (27 + x^2) / (27 + 9 * x^2)


class WaitFreeSpscRingBuffer:
    """
    Wait-Free Single-Producer Single-Consumer (SPSC) Ring Buffer.
    - Backed by a power-of-two float32 array for bitwise '& mask' indexing.
    - Uses monotonic integer cursors (_write_pos, _read_pos) so zero locks are ever acquired.
    - Adaptive pre-buffering (jitter cushion) prevents alternating-silence stutter between unsynced streams.
    - Preserves safety cushion during clock-drift correction instead of dropping to zero margin.
    """

    CAPACITY = 16384
    MASK = 16383
    UNDERRUN_FADE = 16
    # the micro-underrun fade-outs, one per length, made once so an underrun allocates nothing in the audio callback
    _FADE_RAMPS = tuple(np.linspace(1.0, 0.0, k, dtype=np.float32) for k in range(UNDERRUN_FADE + 1))

    def __init__(self, capacity: int = 16384, max_lag_samples: int = 1536, target_cushion: int = 512):
        if capacity <= 0 or capacity & (capacity - 1):
            raise ValueError("capacity must be a power of two (the cursors wrap with '& mask')")
        self.capacity = capacity
        self.mask = capacity - 1
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._max_lag = max_lag_samples
        self._target_cushion = target_cushion
        self._primed = False
        self._underrun_blocks: int = 0

    def reset(self) -> None:
        """Clears cursors and unprimes so next stream begins with fresh cushion."""
        self._write_pos = 0
        self._read_pos = 0
        self._primed = False
        self._underrun_blocks = 0
        self._buf.fill(0.0)

    def available_read(self) -> int:
        return self._write_pos - self._read_pos

    def write(self, data: np.ndarray, gain: float = 1.0) -> None:
        n = len(data)
        if n == 0 or n > self.capacity:
            return

        w_idx = self._write_pos & self.mask
        end = w_idx + n
        if end <= self.capacity:
            np.multiply(data, gain, out=self._buf[w_idx:end])
        else:
            first = self.capacity - w_idx
            np.multiply(data[:first], gain, out=self._buf[w_idx:])
            np.multiply(data[first:], gain, out=self._buf[: end & self.mask])

        # Single monotonic integer update (Producer-only write)
        self._write_pos += n

    def read_into(self, out: np.ndarray) -> None:
        n = len(out)
        w_pos = self._write_pos
        r_pos = self._read_pos
        avail = w_pos - r_pos

        cushion = min(self._target_cushion, max(n, self._max_lag // 2))

        # Initial priming / underrun recovery: wait until we have a healthy cushion
        if not self._primed:
            if avail >= cushion:
                self._primed = True
            else:
                out.fill(0.0)
                return

        if avail < n:
            # Handle micro-underrun smoothly without violent unpriming:
            # If partial audio exists, output it and zero the tail; otherwise output silence
            if avail > 0:
                r_idx = r_pos & self.mask
                end = r_idx + avail
                if end <= self.capacity:
                    out[:avail] = self._buf[r_idx:end]
                else:
                    first = self.capacity - r_idx
                    out[:first] = self._buf[r_idx:]
                    out[first:avail] = self._buf[: end & self.mask]
                fade_len = min(self.UNDERRUN_FADE, avail)
                out[avail - fade_len : avail] *= self._FADE_RAMPS[fade_len]
                out[avail:].fill(0.0)
                self._read_pos = r_pos + avail
            else:
                out.fill(0.0)

            self._underrun_blocks += 1
            # Unprime after brief starvation (~10 blocks = ~53ms) so subsequent speech
            # starts with a healthy 2-chunk cushion without underrunning on GPU latency
            if self._underrun_blocks > 10:
                self._primed = False
            return

        self._underrun_blocks = 0

        # Clock-drift guard: prevent buildup of side-tone latency, but retain cushion so
        # the very next callback does not immediately underflow
        if avail > self._max_lag:
            # never closer to the writer than this read: a block larger than the cushion would otherwise run past the
            # write cursor into audio from a lap ago (avail >= n here, so the cursor still only moves forward)
            r_pos = w_pos - max(cushion, n)
            self._read_pos = r_pos

        r_idx = r_pos & self.mask
        end = r_idx + n
        if end <= self.capacity:
            out[:] = self._buf[r_idx:end]
        else:
            first = self.capacity - r_idx
            out[:first] = self._buf[r_idx:]
            out[first:] = self._buf[: end & self.mask]

        # Single monotonic integer update (Consumer-only write)
        self._read_pos = r_pos + n

    def peek_chunk(self, n: int, offset: int = 0) -> np.ndarray:
        """Reads n samples starting from read cursor without advancing consumer read cursor."""
        out = np.empty(n, dtype=np.float32)
        r_pos = self._read_pos + offset
        r_idx = r_pos & self.mask
        end = r_idx + n
        if end <= self.capacity:
            out[:] = self._buf[r_idx:end]
        else:
            first = self.capacity - r_idx
            out[:first] = self._buf[r_idx:]
            out[first:] = self._buf[: end & self.mask]
        return out

    def advance_read(self, n: int) -> None:
        """Advances consumer read cursor by n samples."""
        self._read_pos += n

    def read_chunk(self, n: int) -> np.ndarray:
        out = self.peek_chunk(n)
        self._read_pos += n
        return out

    def skip_to_latest(self, keep_samples: int) -> None:
        """Discards older unread samples so only the most recent keep_samples remain unread."""
        w_pos = self._write_pos
        avail = w_pos - self._read_pos
        if avail > keep_samples:
            self._read_pos = w_pos - keep_samples


class ZeroRollMirrorBuffer:
    """
    2x Mirrored Sliding Window Buffer for Real-Time ONNX Streaming Inference.
    Eliminates O(W) np.roll() full-buffer copies by maintaining a double-length buffer (2 * W)
    where every incoming block of size N is written at both `head` and `head + W`.
    Any slice `buf[head : head + W]` is guaranteed to be a contiguous O(1) NumPy view!
    """

    def __init__(self, window_size: int = 4096):
        self.W = window_size
        self._mirror_buf = np.zeros(window_size * 2, dtype=np.float32)
        self._head = 0

    def push_and_get_view(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        w = self.W
        if n >= w:
            self._mirror_buf[:w] = block[-w:]
            self._mirror_buf[w:] = block[-w:]
            self._head = 0
            return self._mirror_buf[:w]

        h = self._head
        end = h + n
        if end <= w:
            self._mirror_buf[h:end] = block
            self._mirror_buf[h + w : end + w] = block
        else:
            first = w - h
            rem = n - first
            self._mirror_buf[h:w] = block[:first]
            self._mirror_buf[h + w : 2 * w] = block[:first]
            self._mirror_buf[:rem] = block[first:]
            self._mirror_buf[w : w + rem] = block[first:]

        self._head = end if end < w else (end - w)
        return self._mirror_buf[self._head : self._head + w]


class BitwiseCircularPitchShifter:
    """
    Power-of-Two Dual-Head Rotating Delay-Line Pitch Shifter.
    - Uses C = 4096 (2^12) circular buffer with bitwise '& 4095' wrapping.
    - Pre-allocates all scratch vectors at initialization -> process_into() creates no arrays (only NumPy's internal
      buffer for the one float -> index cast, which a block-sized copy never needs).
    - Both read heads are one (2, n) array, so every step of the interpolation is a single NumPy call for both.
    - Uses single-multiply FMA lerp: y = y0 + frac * (y1 - y0)
    - Uses complementary Hann identity: out = val_b + gain_a * (val_a - val_b)
      (eliminates second cosine evaluation and gain_b array completely).
    """

    CAPACITY = 4096
    MASK = 4095

    def __init__(self, sample_rate: int = 48000, window_ms: float = 36.0):
        self.sr = sample_rate
        self.win_size = float(int(sample_rate * (window_ms / 1000.0)))
        self.buffer = np.zeros(self.CAPACITY, dtype=np.float32)
        self.write_idx: int = 0
        self.phase: float = 0.0

        # Pre-allocated scratch arena for O(1) space complexity. The two-head arrays are kept flat and viewed as
        # (2, n), so the view is C-contiguous for every block size and np.take can write into it directly.
        two = 2 * MAX_BLOCK_SIZE
        self._arange_f32 = np.arange(MAX_BLOCK_SIZE, dtype=np.float32)
        self._steps_f32 = np.arange(1, MAX_BLOCK_SIZE + 1, dtype=np.float32)
        self._w_pos = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)
        self._gain_a = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)
        self._phases = np.zeros(two, dtype=np.float32)    # row 0: head A, row 1: head B (half a window later)
        self._read_pos = np.zeros(two, dtype=np.float32)  # read position, then its fractional part
        self._floor = np.zeros(two, dtype=np.float32)
        self._idx0 = np.zeros(two, dtype=np.intp)          # intp: np.take uses these without converting them
        self._idx1 = np.zeros(two, dtype=np.intp)
        self._val0 = np.zeros(two, dtype=np.float32)
        self._val1 = np.zeros(two, dtype=np.float32)
        self._views: Dict[int, tuple] = {}  # block size -> the scratch views for it (made once per size)

    def _views_for(self, n: int) -> tuple:
        views = self._views.get(n)
        if views is None:
            def two_rows(a: np.ndarray) -> np.ndarray:
                return a[: 2 * n].reshape(2, n)

            phases, val0, val1 = two_rows(self._phases), two_rows(self._val0), two_rows(self._val1)
            views = (
                phases, phases[0], phases[1], self._steps_f32[:n], self._arange_f32[:n], self._w_pos[:n],
                two_rows(self._read_pos), two_rows(self._floor), two_rows(self._idx0), two_rows(self._idx1),
                val0, val1, val0[0], val0[1], val1[0], self._gain_a[:n],
            )
            self._views[n] = views
        return views

    def _write(self, x: np.ndarray) -> None:
        """Appends ``x`` to the delay line (two slice copies at most, no index array)."""
        n = len(x)
        w = self.write_idx
        end = w + n
        if end <= self.CAPACITY:
            self.buffer[w:end] = x
        else:
            first = self.CAPACITY - w
            self.buffer[w:] = x[:first]
            self.buffer[: end - self.CAPACITY] = x[first:]
        self.write_idx = end & self.MASK

    def process_into(self, x: np.ndarray, semitones: float) -> None:
        """Processes `x` in-place with zero heap allocations."""
        n = len(x)
        w = self.write_idx
        self._write(x)

        if abs(semitones) < 0.05:
            return

        pitch_ratio = 2.0 ** (semitones / 12.0)
        phasor_step = (1.0 - pitch_ratio) / self.win_size

        (phases, phases_a, phases_b, steps, arange, w_pos, read_pos, floor, idx0, idx1,
         val0, val1, val_a, val_b, diff, gain_a) = self._views_for(n)
        np.multiply(steps, phasor_step, out=phases_a)
        np.add(phases_a, self.phase, out=phases_a)
        np.mod(phases_a, 1.0, out=phases_a)
        self.phase = float(phases_a[-1])
        np.add(phases_a, 0.5, out=phases_b)
        np.mod(phases_b, 1.0, out=phases_b)

        # Both heads: read position = write position - phase * window, split into integer index and fraction
        np.add(arange, w, out=w_pos)  # unmasked: '& MASK' on the integer part wraps it below
        np.multiply(phases, self.win_size, out=read_pos)
        np.subtract(w_pos, read_pos, out=read_pos)
        np.floor(read_pos, out=floor)
        np.copyto(idx0, floor, casting="unsafe")
        np.subtract(read_pos, floor, out=read_pos)  # Fractional part in [0.0, 1.0)
        np.bitwise_and(idx0, self.MASK, out=idx0)
        np.add(idx0, 1, out=idx1)
        np.bitwise_and(idx1, self.MASK, out=idx1)

        # Single-multiply lerp: val0 = y0 + frac * (y1 - y0)   ('clip' never fires: the indices are masked already)
        np.take(self.buffer, idx0, out=val0, mode="clip")
        np.take(self.buffer, idx1, out=val1, mode="clip")
        np.subtract(val1, val0, out=val1)
        np.multiply(val1, read_pos, out=val1)
        np.add(val0, val1, out=val0)

        # Raised cosine window for Head A: gain_a = 0.5 * (1.0 - cos(2*pi*phases_a))
        np.multiply(phases_a, 2.0 * np.pi, out=gain_a)
        np.cos(gain_a, out=gain_a)
        np.subtract(1.0, gain_a, out=gain_a)
        np.multiply(gain_a, 0.5, out=gain_a)

        # Complementary Hann identity: x = val_b + gain_a * (val_a - val_b)
        np.subtract(val_a, val_b, out=diff)
        np.multiply(gain_a, diff, out=diff)
        np.add(val_b, diff, out=x)


class VectorizedDarkReverb:
    """
    Parallel Schroeder Comb + True Sample-Accurate One-Pole IIR Damping Spatializer.
    A comb line can only hand out as many delayed samples as it is long, so blocks longer than the shortest comb
    (1116 samples at 48 kHz; the app's blocks are 128-512) are processed in pieces of at most that length. Every
    piece read is then causal and executes via C-level scipy.signal.lfilter with zero per-sample Python loops.
    """

    def __init__(self, sample_rate: int = 48000):
        scale = sample_rate / 48000.0
        self.comb_delays = [max(600, int(d * scale)) for d in (1116, 1356, 1492)]
        self.comb_bufs = [np.zeros(d, dtype=np.float32) for d in self.comb_delays]
        self.comb_idxs = [0, 0, 0]
        self._max_piece = min(self.comb_delays)

        # One-pole low-pass damping IIR coefficients: y[n] = (1 - d)*x[n] + d*y[n-1]
        damp = 0.45
        self._b_damp = np.array([1.0 - damp], dtype=np.float32)
        self._a_damp = np.array([1.0, -damp], dtype=np.float32)
        self._lp_zis = [np.zeros(1, dtype=np.float32) for _ in range(3)]
        self._feedback = np.float32(0.62)

        # Pre-allocated scratch buffers
        self._wet = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)
        self._delayed = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)
        self._write_scratch = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)

    def process_into(self, x: np.ndarray, mix: float) -> None:
        if mix <= 0.01:
            return
        n = len(x)
        if n <= self._max_piece:
            self._process_piece(x, mix)
            return
        for start in range(0, n, self._max_piece):
            self._process_piece(x[start : start + self._max_piece], mix)

    def _process_piece(self, x: np.ndarray, mix: float) -> None:
        n = len(x)
        wet = self._wet[:n]
        delayed = self._delayed[:n]
        write_scratch = self._write_scratch[:n]
        feedback = self._feedback

        for c_i, buf in enumerate(self.comb_bufs):
            d_len = self.comb_delays[c_i]
            idx = self.comb_idxs[c_i]

            # Read N delayed samples from circular comb line
            end = idx + n
            wraps = end > d_len
            if not wraps:
                delayed[:] = buf[idx:end]
            else:
                first = d_len - idx
                delayed[:first] = buf[idx:]
                delayed[first:] = buf[: end - d_len]

            # True sample-by-sample IIR low-pass damping in direct compiled C
            filtered, self._lp_zis[c_i] = _linear_filter(
                self._b_damp, self._a_damp, delayed, -1, self._lp_zis[c_i]
            )

            # Write x + feedback * filtered back to circular comb line (straight into the line when it does not wrap)
            target = buf[idx:end] if not wraps else write_scratch
            np.multiply(filtered, feedback, out=target)
            np.add(x, target, out=target)
            if wraps:
                first = d_len - idx
                buf[idx:] = write_scratch[:first]
                buf[: end - d_len] = write_scratch[first:]

            self.comb_idxs[c_i] = end if end < d_len else (end - d_len)
            if c_i == 0:
                np.copyto(wet, delayed)  # the first comb starts the sum: no separate zero-fill pass
            else:
                np.add(wet, delayed, out=wet)

        # In-place wet/dry blend: x = x + mix * (0.33 * wet - x)
        np.multiply(wet, 0.33, out=wet)
        np.subtract(wet, x, out=wet)
        np.multiply(wet, mix, out=wet)
        np.add(x, wet, out=x)


def allowed_cpu_count() -> int:
    """Logical CPUs this process may run on: fewer than ``os.cpu_count()`` once it is pinned (see
    configure_windows_process_isolation) or started with a restricted affinity."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            if kernel32.GetProcessAffinityMask(
                kernel32.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
            ):
                return max(1, bin(process_mask.value).count("1"))
        except Exception:
            pass
    return os.cpu_count() or 4


# ONNX tensor element types -> NumPy, for feeding a model in the precision it was exported with (fp16 exports exist)
_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}

# What each input of an RVC v2 generator carries, under the names the common exporters give it (RVC WebUI: phone,
# phone_lengths, pitch, pitchf, ds, rnd; w-okada / MoeSS: nsff0 and sid).
RVC_INPUT_KINDS = {
    "phone": "feats", "feats": "feats", "hubert": "feats",
    "phone_lengths": "length", "feats_lengths": "length",
    "pitch": "coarse",
    "pitchf": "f0", "nsff0": "f0",
    "sid": "speaker", "ds": "speaker",
    "rnd": "noise",
}


class ONNXVoiceStreamer:
    """
    Hardware-Accelerated ONNX Runtime Wrapper (NVIDIA CUDA / TensorRT / DirectML / CPU)
    supporting both Direct Waveform models and full RVC v2 Generative neural pipelines.
    Runs asynchronous neural inference on a dedicated worker thread with wait-free SPSC
    ring buffers, ensuring zero real-time audio callback blocking and zero raw voice leakage.
    """

    HOP_SAMPLES = 7680       # 160 ms @ 48kHz (2560 @ 16kHz = 8 HuBERT frames)
    OVERLAP_SAMPLES = 960    # 20 ms @ 48kHz (320 @ 16kHz = 1 HuBERT frame)
    CHUNK_SAMPLES = HOP_SAMPLES + OVERLAP_SAMPLES  # 8640 samples (180 ms = 9 HuBERT frames)

    # f0 search range and YIN thresholds (see _extract_f0)
    F0_MIN_HZ = 50.0
    F0_MAX_HZ = 1100.0
    YIN_THRESHOLD = 0.15        # a dip of the normalised difference below this is a period
    YIN_LOOSE_THRESHOLD = 0.35  # no dip that deep: the deepest one still counts if it is below this (breathy voice)

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.session = None
        self.hubert_session = None
        self.mirror_buf = ZeroRollMirrorBuffer(window_size=4096)
        self.active = False
        self.is_rvc = False
        self.input_name = ""
        self.hubert_input_name = "source"
        self._hubert_dtype = np.float32
        self._rvc_inputs: List[Tuple[str, str, type, int]] = []  # (name, kind, dtype, noise channels)
        self._in_ring = WaitFreeSpscRingBuffer(capacity=131072, max_lag_samples=48000, target_cushion=self.CHUNK_SAMPLES)
        self._out_ring = WaitFreeSpscRingBuffer(capacity=131072, max_lag_samples=48000, target_cushion=self.HOP_SAMPLES * 2)
        self._worker_thread: Optional[threading.Thread] = None
        # Every worker gets its own event: one still busy in a slow inference when the model is reloaded keeps
        # seeing its own (set) event and exits, instead of running on next to its replacement on the same rings.
        self._stop_event = threading.Event()
        self.current_pitch_semitones: float = 0.0
        self._speech_hangover: int = 0
        self.last_error = ""   # the last exception the worker hit (it keeps running); shown in problem reports
        self.error_count = 0

        # Precompute raised-cosine overlap-add crossfade envelopes
        self._fade_out = (np.cos(np.linspace(0.0, np.pi / 2.0, self.OVERLAP_SAMPLES, dtype=np.float32)) ** 2)
        self._fade_in = (np.sin(np.linspace(0.0, np.pi / 2.0, self.OVERLAP_SAMPLES, dtype=np.float32)) ** 2)
        self._rng = np.random.default_rng()

    def set_pitch(self, semitones: float) -> None:
        self.current_pitch_semitones = float(semitones)

    def _stop_worker(self, timeout: float = 0.2) -> None:
        self._stop_event.set()
        worker = self._worker_thread
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=timeout)
        self._worker_thread = None

    def _start_worker(self) -> None:
        self._stop_worker()
        self._in_ring.reset()
        self._out_ring.reset()
        stop = threading.Event()
        self._stop_event = stop
        self._worker_thread = threading.Thread(
            target=self._rvc_worker_loop, args=(stop,), daemon=True, name="vcomms-rvc-worker"
        )
        self._worker_thread.start()

    def _rvc_worker_loop(self, stop: Optional[threading.Event] = None) -> None:
        stop = stop or self._stop_event
        hop = self.HOP_SAMPLES
        overlap = self.OVERLAP_SAMPLES
        total_chunk = self.CHUNK_SAMPLES
        prev_tail: Optional[np.ndarray] = None

        while not stop.is_set():
            if not self.active or self.session is None or self.hubert_session is None:
                stop.wait(0.02)
                prev_tail = None
                continue

            avail = self._in_ring.available_read()
            if avail < total_chunk:
                stop.wait(0.005)
                continue

            # Real-Time Bound: If input accumulated a large backlog (e.g. system suspended or heavy stall),
            # skip older samples so latency never drifts past ~1 second
            if avail > hop * 4:
                self._in_ring.skip_to_latest(total_chunk + hop)
                prev_tail = None

            try:
                # Read total_chunk (180ms) and advance by hop (160ms) so consecutive chunks share 20ms overlap
                chunk = self._in_ring.peek_chunk(total_chunk)
                self._in_ring.advance_read(hop)

                # Polyphase anti-aliased 48k -> 16k decimation
                audio_16k = signal.resample_poly(chunk, 1, 3, padtype="line").astype(np.float32)

                # Extract Pitch & Energy for Pitch-Aware Voice Activity Gate
                chunk_rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
                nsff0, pitch = self._extract_f0(audio_16k, target_len=18, semitones=self.current_pitch_semitones)
                voiced_frames = int(np.count_nonzero(pitch > 1))

                # Pitch-Aware Voice Activity Gate:
                # 1. No voiced vocal cords vibrating across the chunk -> Pure silence / breath / room fan noise.
                # 2. Or chunk RMS below ambient room noise floor (0.006 = -44 dBFS).
                # HiFi-GAN generators synthesize resonant unvoiced breath/whoosh noise when fed unvoiced silence.
                # Gating unvoiced chunks completely eliminates trailing hiss/whoosh upon stopping speech.
                is_silence = (voiced_frames == 0) or (chunk_rms < 0.006)

                if is_silence:
                    if prev_tail is not None and len(prev_tail) == overlap:
                        # Smooth 20ms raised-cosine fade-out of previous speech tail, followed by silence
                        blended = np.concatenate([prev_tail * self._fade_out, np.zeros(hop - overlap, dtype=np.float32)])
                        self._out_ring.write(blended)
                        prev_tail = None
                    else:
                        # Continuous silence preserves healthy 2-chunk cushion so next word never underruns
                        self._out_ring.write(np.zeros(hop, dtype=np.float32))
                    stop.wait(0.005)
                    continue

                synth = self._convert(chunk, audio_16k=audio_16k, pitch=pitch, nsff0=nsff0)
                if stop.is_set():
                    break  # the model was reloaded or unloaded during inference: the rings belong to a newer worker

                # Dynamic Energy Envelope Shaping:
                # HiFi-GAN generators output normalized amplitude (~0.12 RMS) regardless of input level.
                # As the speaker naturally decays their voice into silence, scale synth to smoothly follow
                # the natural vocal decay, preventing unvoiced residual noise from overpowering the voice.
                ref_rms = 0.040
                if chunk_rms < ref_rms:
                    env_gain = min(1.0, max(0.0, chunk_rms / ref_rms) ** 1.5)
                    synth *= env_gain

                if len(synth) == total_chunk:
                    # Raised-cosine overlap-add crossfading with previous chunk tail eliminates clicks
                    if prev_tail is not None and len(prev_tail) == overlap:
                        blended_head = prev_tail * self._fade_out + synth[:overlap] * self._fade_in
                        ready = np.concatenate([blended_head, synth[overlap:hop]])
                    else:
                        head = synth[:overlap] * self._fade_in
                        ready = np.concatenate([head, synth[overlap:hop]])

                    prev_tail = synth[hop:total_chunk].copy()
                    self._out_ring.write(ready)
            except Exception as exc:
                if stop.is_set():
                    break  # unloaded mid-inference (its sessions went away): not a conversion error worth reporting
                prev_tail = None
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"

    def _convert(
        self,
        chunk: np.ndarray,
        audio_16k: Optional[np.ndarray] = None,
        pitch: Optional[np.ndarray] = None,
        nsff0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """One 180 ms chunk at 48 kHz through HuBERT and the RVC generator: CHUNK_SAMPLES of converted voice."""
        total_chunk = self.CHUNK_SAMPLES
        if audio_16k is None:
            audio_16k = signal.resample_poly(chunk, 1, 3, padtype="line").astype(np.float32)
        # 160-sample reflect padding handles HuBERT CNN receptive field margins
        padded_16k = np.pad(audio_16k, (160, 160), mode="reflect")
        h_inp = padded_16k.reshape(1, 1, -1).astype(self._hubert_dtype, copy=False)

        feats_50hz = self.hubert_session.run(None, {self.hubert_input_name: h_inp})[0]
        if feats_50hz.ndim == 2:
            feats_50hz = feats_50hz[np.newaxis, ...]

        # Double HuBERT 50 Hz frames to 100 Hz required by RVC v2 HiFi-GAN generators
        feats_100hz = np.repeat(feats_50hz, 2, axis=1)
        if nsff0 is None or pitch is None:
            nsff0, pitch = self._extract_f0(audio_16k, target_len=feats_100hz.shape[1], semitones=self.current_pitch_semitones)

        # Run RVC Generator (480x upsampling from 100 Hz frames = exact 48,000 Hz synthesis)
        out = self.session.run(None, self._rvc_feed(feats_100hz, pitch, nsff0))[0]
        synth = np.asarray(out, dtype=np.float32).ravel()
        if len(synth) != total_chunk and len(synth) > 0:
            synth = np.interp(
                np.linspace(0, len(synth), total_chunk, endpoint=False),
                np.arange(len(synth)),
                synth,
            ).astype(np.float32)
        return synth

    def _rvc_feed(self, feats: np.ndarray, pitch: np.ndarray, nsff0: np.ndarray) -> Dict[str, np.ndarray]:
        frames = feats.shape[1]
        feed: Dict[str, np.ndarray] = {}
        for name, kind, dtype, channels in self._rvc_inputs:
            if kind == "feats":
                value = feats
            elif kind == "length":
                value = np.array([frames])
            elif kind == "coarse":
                value = pitch
            elif kind == "f0":
                value = nsff0
            elif kind == "speaker":
                value = np.array([0])
            else:  # noise: RVC WebUI exports take the flow's random input from the caller
                value = self._rng.standard_normal((1, channels, frames))
            feed[name] = np.asarray(value).astype(dtype, copy=False)
        return feed

    def _plan_rvc_inputs(self, inputs) -> Optional[str]:
        """Reads what the generator wants; returns why it cannot be fed, or None."""
        plan = []
        for inp in inputs:
            kind = RVC_INPUT_KINDS.get(inp.name.lower())
            if kind is None:
                return f"The voice model has an input V-COMMS does not know how to fill: '{inp.name}'."
            channels = 192
            shape = getattr(inp, "shape", None) or []
            if kind == "noise" and len(shape) >= 2 and isinstance(shape[1], int) and shape[1] > 0:
                channels = shape[1]
            plan.append((inp.name, kind, _ORT_DTYPES.get(getattr(inp, "type", ""), np.float32), channels))
        if not any(kind == "feats" for _name, kind, _dtype, _channels in plan):
            return "The voice model has no HuBERT feature input (phone or feats)."
        self._rvc_inputs = plan
        return None

    @staticmethod
    def _session_options(rt):
        sess_opts = rt.SessionOptions()
        sess_opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        # As many threads as the cores this process may use (it is pinned to a few on bigger PCs), at most 8. They
        # must not spin-wait between runs: a spinning pool would burn those same cores and starve the audio thread.
        sess_opts.intra_op_num_threads = min(8, max(2, allowed_cpu_count()))
        sess_opts.inter_op_num_threads = 1
        try:
            sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        except Exception:
            pass
        return sess_opts

    @staticmethod
    def _format_provider(raw_prov: str) -> str:
        cleaned = raw_prov.replace("ExecutionProvider", "")
        if cleaned == "Dml":
            return "GPU (DirectML / RTX)"
        elif cleaned in ("CUDA", "Tensorrt"):
            return f"GPU ({cleaned})"
        elif cleaned == "CPU":
            return "CPU"
        return cleaned

    def load_model(self, model_path: str) -> Tuple[bool, str]:
        if not ONNX_AVAILABLE:
            return False, "onnxruntime is not installed."
        if not os.path.isfile(model_path):
            return False, "Model file not found."
        self.unload()
        try:
            rt = _load_onnxruntime()
            # Hardware Execution Provider Selection:
            # - On ARM64 (Qualcomm Snapdragon X Elite): Qualcomm Oryon 12-core CPU with NEON SIMD
            #   runs ~10x faster than DirectML HLSL shader emulation on Adreno GPU.
            # - On x86_64: Prioritize NVIDIA RTX GPU acceleration via CUDA/TensorRT, then DirectML for DirectX 12 GPUs, then CPU.
            is_arm64 = platform.machine().lower() in ("arm64", "aarch64")
            if is_arm64:
                providers = [
                    "CUDAExecutionProvider",
                    "TensorrtExecutionProvider",
                    "CPUExecutionProvider",
                    "DmlExecutionProvider",
                ]
            else:
                providers = [
                    "CUDAExecutionProvider",
                    "TensorrtExecutionProvider",
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ]
            available = [p for p in providers if p in rt.get_available_providers()]
            sess_opts = self._session_options(rt)
            self.session = rt.InferenceSession(model_path, sess_options=sess_opts, providers=available)
            inputs = self.session.get_inputs()
            input_names = [inp.name.lower() for inp in inputs]

            # Detect whether this is an RVC Synthesizer (phone, pitch, nsff0, sid)
            is_rvc = any(name in RVC_INPUT_KINDS and name != "sid" for name in input_names) or len(inputs) > 2
            if is_rvc:
                self.is_rvc = True
                problem = self._plan_rvc_inputs(inputs)
                if problem:
                    self.unload()
                    return False, problem
                # Look for HuBERT ContentVec feature extractor in models/ or model directory
                model_dir = os.path.dirname(os.path.abspath(model_path))
                appdata_models = os.path.join(os.environ.get("APPDATA", ""), "V-COMMS", "models")
                localapp_models = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "V-COMMS", "models")
                script_models = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                exe_models = os.path.join(os.path.dirname(sys.executable), "models")
                internal_models = os.path.join(os.path.dirname(sys.executable), "_internal", "models")
                candidates = [
                    os.path.join("models", "vec-768-layer-12.onnx"),
                    os.path.join(model_dir, "vec-768-layer-12.onnx"),
                    os.path.join(appdata_models, "vec-768-layer-12.onnx"),
                    os.path.join(localapp_models, "vec-768-layer-12.onnx"),
                    os.path.join(script_models, "vec-768-layer-12.onnx"),
                    os.path.join(exe_models, "vec-768-layer-12.onnx"),
                    os.path.join(internal_models, "vec-768-layer-12.onnx"),
                    "vec-768-layer-12.onnx",
                    os.path.join("models", "hubert_base.onnx"),
                    os.path.join(model_dir, "hubert_base.onnx"),
                    os.path.join(appdata_models, "hubert_base.onnx"),
                    "hubert_base.onnx",
                ]
                hubert_path = next((c for c in candidates if os.path.isfile(c) and os.path.getsize(c) > 10_000_000), None)
                if not hubert_path:
                    self.active = False
                    return False, "NEED_HUBERT:RVC model detected! HuBERT base model (vec-768-layer-12.onnx) is required."

                self.hubert_session = rt.InferenceSession(hubert_path, sess_options=sess_opts, providers=available)
                h_inps = self.hubert_session.get_inputs()
                self.hubert_input_name = h_inps[0].name if h_inps else "source"
                self._hubert_dtype = _ORT_DTYPES.get(getattr(h_inps[0], "type", ""), np.float32) if h_inps else np.float32

                # One silent chunk through the exact streaming path: compiles the GPU/CPU kernels for the streaming
                # shape before the first word, and a model that cannot run is reported now instead of going quiet.
                try:
                    self._convert(np.zeros(self.CHUNK_SAMPLES, dtype=np.float32))
                except Exception as exc:
                    self.unload()
                    return False, f"The voice model could not run a test chunk ({type(exc).__name__}: {exc})."

                self.active = True
                self._start_worker()
                prov_str = self._format_provider(self.session.get_providers()[0])
                return True, f"RVC Loaded on {prov_str}"
            else:
                self.is_rvc = False
                self.input_name = inputs[0].name
                self.active = True
                prov_str = self._format_provider(self.session.get_providers()[0])
                return True, f"Direct Model Loaded on {prov_str}"
        except Exception as exc:
            self.active = False
            return False, str(exc)

    def unload(self) -> None:
        """Forgets the model (and frees its memory). Stops the background worker thread."""
        self.active = False
        self._stop_worker()
        self.session = None
        self.hubert_session = None
        self.is_rvc = False
        self.input_name = ""
        self._rvc_inputs = []
        self._in_ring.reset()
        self._out_ring.reset()

    def _extract_f0(
        self,
        audio_16k: np.ndarray,
        target_len: Optional[int] = None,
        semitones: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pitch per RVC frame with YIN (de Cheveigne & Kawahara, 2002), all frames at once:
          d(t)  = sum_j (x_j - x_{j+t})^2 = e0 + e_t - 2 r(t)   (r through one FFT product per frame)
          d'(t) = d(t) * t / sum_{k<=t} d(k)                     (cumulative mean normalisation)
        The period is the first dip of d' below YIN_THRESHOLD, followed down to its bottom and refined by a parabola.
        A plain autocorrelation peak cannot be used: it favours the shortest lags, so a voice whose energy sits in
        its fundamental (a low male voice, a hum) reads as 1.1 kHz.
        """
        sr = 16000
        if target_len is None:
            target_len = max(1, len(audio_16k) // 320)
        min_lag = int(sr / self.F0_MAX_HZ)   # 14
        max_lag = int(sr / self.F0_MIN_HZ)   # 320
        win = max_lag * 2                    # 640-sample integration window: two periods of the lowest pitch
        span = win + max_lag + 1             # samples one frame needs
        nfft = 1 << int(np.ceil(np.log2(span)))  # 1024: j + t < nfft for every term, so the FFT product never wraps

        x = np.asarray(audio_16k, dtype=np.float32)
        hop = len(x) / float(target_len)
        # Frame i is centred on i * hop. At the chunk edges the window slides inward instead of reading padding: a
        # reflected (time-reversed) waveform is not periodic, so padded edge frames read as unvoiced, and the edge
        # frames are exactly the 20 ms the overlap-add crossfade blends with the neighbouring chunks.
        starts = np.rint(np.arange(target_len) * hop - win / 2).astype(np.intp)
        if len(x) < span:
            x = np.pad(x, (0, span - len(x)))
        np.clip(starts, 0, len(x) - span, out=starts)
        frames = np.lib.stride_tricks.sliding_window_view(x, span)[starts]

        spec = np.fft.rfft(frames, nfft, axis=1)
        spec_head = np.fft.rfft(frames[:, :win], nfft, axis=1)
        r = np.fft.irfft(np.conj(spec_head) * spec, nfft, axis=1)[:, : max_lag + 1]
        energy = np.zeros((target_len, span + 1), dtype=np.float64)
        np.cumsum(frames * frames, axis=1, out=energy[:, 1:])
        e0 = energy[:, win] - energy[:, 0]
        e_t = energy[:, win : win + max_lag + 1] - energy[:, : max_lag + 1]
        diff = np.maximum(e0[:, None] + e_t - 2.0 * r, 0.0)
        diff[:, 0] = 0.0
        running = np.cumsum(diff[:, 1:], axis=1)
        cmnd = np.ones_like(diff)
        np.divide(diff[:, 1:] * np.arange(1, max_lag + 1), running, out=cmnd[:, 1:], where=running > 1e-12)

        candidates = cmnd[:, min_lag:max_lag]
        below = candidates < self.YIN_THRESHOLD
        has_dip = below.any(axis=1)
        first = np.argmax(below, axis=1)
        bottom = np.empty_like(below)  # True where the curve stops falling
        np.greater_equal(candidates[:, 1:], candidates[:, :-1], out=bottom[:, :-1])
        bottom[:, -1] = True
        bottom &= np.arange(max_lag - min_lag) >= first[:, None]
        k = np.where(has_dip, np.argmax(bottom, axis=1), np.argmin(candidates, axis=1))
        rows = np.arange(target_len)
        voiced = (e0 > 1e-4) & (has_dip | (candidates[rows, k] < self.YIN_LOOSE_THRESHOLD))

        lag_i = min_lag + k  # always 1 <= lag_i < max_lag, so both neighbours exist
        left, mid, right = cmnd[rows, lag_i - 1], cmnd[rows, lag_i], cmnd[rows, lag_i + 1]
        curve = left - 2.0 * mid + right
        safe = np.abs(curve) > 1e-12
        shift = np.where(safe, 0.5 * (left - right) / np.where(safe, curve, 1.0), 0.0)
        lag = lag_i + np.clip(shift, -1.0, 1.0)
        f0 = np.where(voiced, sr / lag, 0.0).astype(np.float32)

        # 3-tap median filtering removes octave-jump glitches and pitch flicker
        if target_len >= 3:
            f0 = signal.medfilt(f0, kernel_size=3).astype(np.float32)

        if abs(semitones) > 0.05:
            f0 *= 2.0 ** (semitones / 12.0)

        # nsff0: continuous f0 in Hz
        nsff0 = f0.reshape(1, -1).astype(np.float32)
        # pitch: coarse quantized mel bins (1..255)
        f0_mel = 1127.0 * np.log(1.0 + np.maximum(0.0, f0) / 700.0)
        f0_mel[f0 <= 0] = 0.0
        f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
        f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
        f0_coarse = (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
        f0_coarse[f0 <= 0] = 1.0
        pitch = np.clip(np.round(f0_coarse), 1, 255).astype(np.int64).reshape(1, -1)
        return nsff0, pitch

    def process_into(self, x: np.ndarray) -> None:
        if not self.active or self.session is None:
            return
        n = len(x)
        if not self.is_rvc:
            # Direct Waveform Model: O(N) mirror write + O(1) contiguous 4096-sample view
            context_view = self.mirror_buf.push_and_get_view(x)
            try:
                inp = context_view.reshape(1, -1)
                out = self.session.run(None, {self.input_name: inp})[0]
                out_flat = np.asarray(out, dtype=np.float32).ravel()
                if len(out_flat) >= n:
                    x[:] = out_flat[-n:]
                else:
                    x.fill(0.0)
            except Exception:
                x.fill(0.0)
            return

        # RVC Model Pipeline: asynchronous inference via wait-free SPSC ring buffers
        # 1. Enqueue incoming microphone audio to the neural inference queue
        self._in_ring.write(x)

        # 2. Dequeue synthesized neural audio into x.
        # If the output buffer is empty or priming, read_into fills x with silence (0.0),
        # ensuring the user's raw microphone voice NEVER leaks into the output.
        self._out_ring.read_into(x)

# ============================================================================
# 3. WAIT-FREE REAL-TIME DSP PIPELINE
# ============================================================================

# Filter designs shared by every DSP chain. The ones the built-in presets need are pinned at start-up (see
# prewarm_filter_designs); designs made while a slider is dragged go into a small least-recently-used cache. (Emptying
# it whole when full could throw away the band design a snapshot had just stored, in favour of its formant design.)
_PINNED_DESIGNS: Dict[tuple, np.ndarray] = {}
_SWEEP_DESIGNS: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_SWEEP_DESIGN_LIMIT = 64


def biquad_peaking_eq(f0: float, q: float, gain_db: float, sr: int = 48000) -> np.ndarray:
    """Audio EQ Cookbook parametric peaking biquad (returns a 6-element SOS row: [b0, b1, b2, a0, a1, a2])."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / float(sr)
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    b = np.array([b0, b1, b2], dtype=np.float32) / a0
    a = np.array([a0, a1, a2], dtype=np.float32) / a0
    return signal.tf2sos(b, a)[0].astype(np.float32)


class RealTimeDSPChain:
    """
    Lock-Free Real-Time Audio Pipeline.
    - All filter states (`zi`) are native float32.
    - Uses direct compiled Cython `_sosfilt` and fused SOS passes for 10x-40x lower latency on E-cores.
    - Parameter updates use atomic RCU snapshot swapping (`self._snapshot`).
    - All vector operations use pre-allocated scratch buffers for O(1) memory complexity.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate
        self.pitch_shifter = BitwiseCircularPitchShifter(sample_rate)
        self.reverb = VectorizedDarkReverb(sample_rate)
        self.onnx_streamer = ONNXVoiceStreamer(sample_rate)

        # Persistent filter states for compiled C _sosfilt (2 sections per filter)
        self.ring_phase: float = 0.0
        self.gate_env: float = 0.0
        self.band_zi = np.zeros((1, 2, 2), dtype=np.float32)
        self.formant_zi = np.zeros((1, 2, 2), dtype=np.float32)

        # Pre-allocated scratch buffers for audio callback
        self._work_2d = np.zeros((1, MAX_BLOCK_SIZE), dtype=np.float32)
        self._scratch_a = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)
        self._scratch_b = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)
        self._time_steps = np.arange(MAX_BLOCK_SIZE, dtype=np.float32) / float(sample_rate)

        # Atomic RCU snapshot pointer
        self._snapshot: DSPImmutableSnapshot = self._compile_snapshot(VoiceParameters())

    def _cached_design(
        self, key: tuple, build: Callable[[], np.ndarray], pin: bool = False
    ) -> np.ndarray:
        """
        Designing a filter with scipy costs several milliseconds, and holds the GIL the whole time.
        Reuse earlier results so a control that does not touch a filter never redesigns it while running.
        """
        key = (self.sr,) + key
        sos = _PINNED_DESIGNS.get(key)
        if sos is None:
            sos = _SWEEP_DESIGNS.get(key)
            if sos is not None:
                if pin:
                    _PINNED_DESIGNS[key] = sos
                else:
                    _SWEEP_DESIGNS.move_to_end(key)  # just used: the last to go
        if sos is None:
            sos = build()
            if pin:
                _PINNED_DESIGNS[key] = sos
            else:
                _SWEEP_DESIGNS[key] = sos
                while len(_SWEEP_DESIGNS) > _SWEEP_DESIGN_LIMIT:
                    _SWEEP_DESIGNS.popitem(last=False)  # the least recently used design goes
        return sos

    def _compile_snapshot(self, params: VoiceParameters, pin: bool = False) -> DSPImmutableSnapshot:
        """Runs on the UI thread to pre-calculate all float32 SOS matrices and scalar gains."""
        nyq = 0.5 * self.sr
        hpf_norm = float(np.clip(params.hpf_cutoff_hz / nyq, 0.001, 0.95))
        lpf_norm = float(np.clip(params.lpf_cutoff_hz / nyq, 0.01, 0.98))

        # Fused Bandpass (HPF Rumble + LPF High-Cut) into a single 2-section SOS matrix
        band_sos = self._cached_design(
            ("band", hpf_norm, lpf_norm),
            lambda: np.asarray(
                np.vstack([
                    signal.butter(2, hpf_norm, btype="highpass", output="sos"),
                    signal.butter(2, lpf_norm, btype="lowpass", output="sos"),
                ]),
                dtype=np.float32,
            ),
            pin,
        )

        shift_factor = 2.0 ** (params.formant_semitones / 12.0)
        f1_target = float(np.clip(550.0 * shift_factor, 200.0, 1400.0))
        f2_target = float(np.clip(1650.0 * shift_factor, 700.0, 3800.0))

        # Formant Peaking Biquad EQ (smooth vocal tract resonance without destructive cancellation)
        def design_formant() -> np.ndarray:
            gain1 = float(np.clip(params.formant_semitones * 0.8, -4.5, 4.5))
            gain2 = float(np.clip(params.formant_semitones * 0.7, -4.0, 4.0))
            s1 = biquad_peaking_eq(f1_target, 1.3, gain1, self.sr)
            s2 = biquad_peaking_eq(f2_target, 1.5, gain2, self.sr)
            return np.asarray(np.vstack([s1, s2]), dtype=np.float32)

        formant_sos = self._cached_design(
            ("formant_peaking", f1_target, f2_target, params.formant_semitones),
            design_formant,
            pin,
        )

        thresh_lin = float(10.0 ** (params.gate_threshold_db / 20.0))
        drive_gain = float(10.0 ** (params.drive_db / 20.0))
        dg_c = min(max(drive_gain, 1.0), 3.0)
        pade_at_drive = dg_c * (27.0 + dg_c * dg_c) / (27.0 + 9.0 * dg_c * dg_c)
        drive_norm = float(1.0 / max(pade_at_drive, 1e-6))
        out_gain = float(10.0 ** (params.output_gain_db / 20.0))

        has_formant = bool(abs(params.formant_semitones) >= 0.1)

        return DSPImmutableSnapshot(
            params=VoiceParameters(**asdict(params)),
            band_sos=band_sos,
            formant_sos=formant_sos,
            thresh_lin=thresh_lin,
            drive_gain=drive_gain,
            drive_norm=drive_norm,
            out_gain=out_gain,
            has_formant=has_formant,
        )

    def update_parameters(self, new_params: VoiceParameters) -> None:
        """Wait-free atomic pointer swap from UI thread."""
        self._snapshot = self._compile_snapshot(new_params)

    def prepare_parameters(self, params: VoiceParameters) -> None:
        """Designs and keeps the filters these parameters need without applying them."""
        self._compile_snapshot(params, pin=True)

    def process_block(self, audio_in: np.ndarray, use_onnx: bool = False) -> np.ndarray:
        """
        Executes in real-time WASAPI callback.
        Wait-free (zero locks) and zero steady-state heap allocations.
        """
        snap = self._snapshot
        p = snap.params
        n = len(audio_in)

        x_2d = self._work_2d[:, :n]
        scratch_a = self._scratch_a[:n]
        scratch_b = self._scratch_b[:n]
        np.copyto(x_2d[0], audio_in)

        # 1. High-Pass Rumble + Low-Pass Filter fused in a single direct compiled C pass
        _sosfilt(snap.band_sos, x_2d, self.band_zi)
        x = x_2d[0]

        # 2. Smooth Hysteresis Noise Gate (Exponential release avoids cutting off word tails & whispers)
        np.multiply(x, x, out=scratch_a)
        block_rms = float(np.sqrt(np.mean(scratch_a) + 1e-12))
        target_gate = 1.0 if block_rms >= snap.thresh_lin else 0.0
        alpha = 0.65 if target_gate > self.gate_env else 0.08
        self.gate_env = alpha * target_gate + (1.0 - alpha) * self.gate_env
        np.multiply(x, self.gate_env, out=x)

        # 3. Optional ONNX AI Voice Model Inference (Zero-Roll Mirror Buffer / Asynchronous SPSC Neural Pipeline)
        if use_onnx:
            if self.onnx_streamer.active:
                self.onnx_streamer.set_pitch(p.pitch_semitones)
                self.onnx_streamer.process_into(x)
            else:
                # User selected AI Voice Conversion, but model is loading, failed to load, or not selected.
                # Output silence so raw voice NEVER leaks into game comms!
                x.fill(0.0)

        # 4. Preset Character Effects (Automatically bypassed when ONNX AI Model is active)
        if not use_onnx:
            # 4a. Bitwise Power-of-Two Circular Pitch Shifter (In-place, 36ms natural grain window)
            if abs(p.pitch_semitones) >= 0.05:
                self.pitch_shifter.process_into(x, p.pitch_semitones)

            # 4b. Formant Peaking Vocal-Tract Shaper in direct compiled C pass
            if snap.has_formant:
                _sosfilt(snap.formant_sos, x_2d, self.formant_zi)

            # 4c. Phase-Continuous Ring Modulator (In-place algebraic blend)
            if p.ring_mod_mix > 0.01 and p.ring_mod_hz > 1.0:
                rad_freq = 2.0 * np.pi * p.ring_mod_hz
                np.multiply(self._time_steps[:n], rad_freq, out=scratch_a)
                np.add(scratch_a, self.ring_phase, out=scratch_a)
                np.sin(scratch_a, out=scratch_a)
                self.ring_phase = float((self.ring_phase + rad_freq * (n / self.sr)) % (2.0 * np.pi))

                np.subtract(scratch_a, 1.0, out=scratch_a)
                np.multiply(scratch_a, p.ring_mod_mix, out=scratch_a)
                np.add(scratch_a, 1.0, out=scratch_a)
                np.multiply(x, scratch_a, out=x)

            # 4d. Comms Drive / Saturation (In-place Rational Pade Approximant)
            if p.drive_db > 0.1:
                np.multiply(x, snap.drive_gain, out=x)
                fast_pade_tanh_inplace(x, scratch_a, scratch_b)
                np.multiply(x, snap.drive_norm, out=x)

            # 4e. Vectorized Schroeder Dark Reverb (Direct _linear_filter damping)
            if p.reverb_mix > 0.01:
                self.reverb.process_into(x, p.reverb_mix)

        # 9. Output Gain & Soft-Knee Limiter (-1.0 dBFS Ceiling = 0.891)
        ceiling = np.float32(0.891)
        inv_ceiling = np.float32(1.0 / 0.891)
        np.multiply(x, snap.out_gain * inv_ceiling, out=x)
        fast_pade_tanh_inplace(x, scratch_a, scratch_b)
        np.multiply(x, ceiling, out=x)

        return x


# ============================================================================
# 4. WASAPI DUAL-STREAM AUDIO ENGINE MANAGER
# ============================================================================

class AudioEngineManager:
    """
    Owns the two WASAPI streams (mic -> DSP -> cable, plus the optional "Hear Myself" monitor).

    No Qt in here. The real-time callback only overwrites a few plain numbers (meter levels, DSP time,
    xrun count); whoever hosts the engine reads them with ``read_telemetry()`` on its own schedule,
    so the audio thread never touches a UI object or emits a signal.
    """

    def __init__(self, config: AppConfig, on_status: Optional[Callable[[str], None]] = None):
        self.config = config
        self.on_status = on_status  # called synchronously by start/stop_engine, on the caller's thread
        self.status_text = "STOPPED"
        self.last_error = ""
        self.dsp = RealTimeDSPChain(config.sample_rate)
        self.monitor_ring = WaitFreeSpscRingBuffer(max_lag_samples=1536)

        self.primary_stream = None
        self.monitor_stream = None
        self.is_running = False
        self.onnx_message = ""  # what loading the AI voice model said at the last start ("" when none was loaded)

        # Published by the audio callback (single attribute stores, atomic under the GIL)
        self.latest_in_db: float = -80.0
        self.latest_out_db: float = -80.0
        self.latest_dsp_ms: float = 0.0
        self.peak_dsp_ms: float = 0.0   # slowest block since the last read_telemetry()
        self.xruns: int = 0             # PortAudio under/overflow flags seen by the primary callback (WASAPI never sets them)
        # a callback that starts later than the stream's buffers can cover leaves a hole in the audio: that is a stall
        self.stalls: int = 0
        self._last_cb_t0: float = 0.0
        self.stall_threshold_s: float = 0.040   # re-derived from the stream's own output buffer when it starts
        # totals since the engine was last started (for problem reports); three plain numbers, written by the callback
        self._dsp_blocks: int = 0
        self._dsp_total_ms: float = 0.0
        self._dsp_worst_ms: float = 0.0
        self.stream_latency_ms: Tuple[float, float] = (0.0, 0.0)  # driver buffering reported by PortAudio when it started
        self._current_in_idx: Optional[int] = None
        self._current_out_idx: Optional[int] = None
        self._current_mon_idx: Optional[int] = None
        self._scratch_mono = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)

    def _set_status(self, text: str) -> None:
        self.status_text = text
        if self.on_status is not None:
            self.on_status(text)

    def read_telemetry(self) -> Dict[str, float]:
        """Plain-value snapshot of what the audio callback last published. Safe to call from any thread."""
        peak, self.peak_dsp_ms = self.peak_dsp_ms, 0.0
        return {
            "in_db": self.latest_in_db,
            "out_db": self.latest_out_db,
            "dsp_ms": self.latest_dsp_ms,
            "peak_dsp_ms": peak,
            "xruns": self.xruns,
            "stalls": self.stalls,
        }

    def read_session_stats(self) -> Dict[str, float]:
        """DSP timing totals since the engine was last started, for the diagnostics of a problem report."""
        blocks = self._dsp_blocks
        return {
            "running": self.is_running,
            "blocks": blocks,
            "mean_dsp_ms": self._dsp_total_ms / blocks if blocks else 0.0,
            "worst_dsp_ms": self._dsp_worst_ms,
            "xruns": self.xruns + self.stalls,   # dropouts: the driver's flags plus real stalls
        }

    @staticmethod
    def list_audio_devices() -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
        if not SOUNDDEVICE_AVAILABLE:
            return [], []
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        preferred_api_idx = None
        for idx, api in enumerate(hostapis):
            if "WASAPI" in api["name"].upper():
                preferred_api_idx = idx
                break

        inputs: List[Tuple[int, str]] = []
        outputs: List[Tuple[int, str]] = []
        seen_in = set()
        seen_out = set()

        for idx, dev in enumerate(devices):
            if preferred_api_idx is not None and dev["hostapi"] != preferred_api_idx:
                continue
            name = dev["name"].strip()
            api_name = hostapis[dev["hostapi"]]["name"]
            label = f"{name} [{api_name}]"
            if dev["max_input_channels"] > 0 and name not in seen_in:
                inputs.append((idx, label))
                seen_in.add(name)
            if dev["max_output_channels"] > 0 and name not in seen_out:
                outputs.append((idx, label))
                seen_out.add(name)

        if not inputs and not outputs:
            for idx, dev in enumerate(devices):
                name = dev["name"].strip()
                api_name = hostapis[dev["hostapi"]]["name"]
                label = f"{name} [{api_name}]"
                if dev["max_input_channels"] > 0 and name not in seen_in:
                    inputs.append((idx, label))
                    seen_in.add(name)
                if dev["max_output_channels"] > 0 and name not in seen_out:
                    outputs.append((idx, label))
                    seen_out.add(name)

        return inputs, outputs

    def _primary_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info, status) -> None:
        t0 = time.perf_counter()
        if status:
            self.xruns += 1
        last, self._last_cb_t0 = self._last_cb_t0, t0
        if last and t0 - last > self.stall_threshold_s:
            self.stalls += 1

        if indata.ndim > 1:
            if indata.shape[1] == 1:
                mono_in = indata[:, 0]
            else:
                ch0 = indata[:, 0]
                ch1 = indata[:, 1]
                e0 = float(np.dot(ch0, ch0))
                e1 = float(np.dot(ch1, ch1))
                if e0 > 1e-7 and e1 < 1e-8:
                    mono_in = ch0
                elif e1 > 1e-7 and e0 < 1e-8:
                    mono_in = ch1
                else:
                    mono_in = self._scratch_mono[:frames]
                    np.add(ch0, ch1, out=mono_in)
                    np.multiply(mono_in, 0.5, out=mono_in)
        else:
            mono_in = indata
        in_rms = float(np.sqrt(np.dot(mono_in, mono_in) / float(max(frames, 1)) + 1e-12))
        self.latest_in_db = max(-80.0, 20.0 * np.log10(in_rms))

        if self.config.mic_muted:
            outdata.fill(0.0)
            self.latest_out_db = -80.0
            dsp_ms = (time.perf_counter() - t0) * 1000.0
            self.latest_dsp_ms = dsp_ms
            if dsp_ms > self.peak_dsp_ms:
                self.peak_dsp_ms = dsp_ms
            self._dsp_blocks += 1
            self._dsp_total_ms += dsp_ms
            if dsp_ms > self._dsp_worst_ms:
                self._dsp_worst_ms = dsp_ms
            return

        if self.config.bypass_effects:
            processed = mono_in
        else:
            processed = self.dsp.process_block(mono_in, use_onnx=self.config.enable_onnx)

        out_rms = float(np.sqrt(np.dot(processed, processed) / float(max(frames, 1)) + 1e-12))
        self.latest_out_db = max(-80.0, 20.0 * np.log10(out_rms))

        if outdata.ndim > 1:
            outdata[:] = processed[:, None]  # one broadcast copy fills every channel
        else:
            outdata[:] = processed

        if self.config.hear_myself:
            self.monitor_ring.write(processed, gain=float(self.config.monitor_volume))

        dsp_ms = (time.perf_counter() - t0) * 1000.0
        self.latest_dsp_ms = dsp_ms
        if dsp_ms > self.peak_dsp_ms:
            self.peak_dsp_ms = dsp_ms
        self._dsp_blocks += 1
        self._dsp_total_ms += dsp_ms
        if dsp_ms > self._dsp_worst_ms:
            self._dsp_worst_ms = dsp_ms

    def _monitor_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if not self.config.hear_myself:
            outdata.fill(0.0)
            return
        if outdata.ndim > 1:
            self.monitor_ring.read_into(outdata[:, 0])
            for ch in range(1, outdata.shape[1]):
                outdata[:, ch] = outdata[:, 0]
        else:
            self.monitor_ring.read_into(outdata)

    def start_engine(self, in_idx: int, out_idx: int, mon_idx: Optional[int]) -> bool:
        self.last_error = ""
        if not SOUNDDEVICE_AVAILABLE:
            self.last_error = "sounddevice library not installed."
            self._set_status("ERROR: sounddevice library not installed.")
            return False

        self.stop_engine()
        self._current_in_idx = in_idx
        self._current_out_idx = out_idx
        self._current_mon_idx = mon_idx
        try:
            self.dsp = RealTimeDSPChain(self.config.sample_rate)
            self.dsp.update_parameters(self.config.current_params)
            # The AI model (and HuBERT, ~360 MB) is only loaded when the AI voice is on; switching it on later loads it
            # then (EngineBridge.set_flag). Loading it on every start costs seconds and memory for a voice not in use.
            self.onnx_message = ""
            if self.config.enable_onnx and self.config.onnx_model_path:
                _ok, self.onnx_message = self.dsp.onnx_streamer.load_model(self.config.onnx_model_path)

            in_info = sd.query_devices(in_idx)
            out_info = sd.query_devices(out_idx)
            in_ch = min(2, max(1, in_info["max_input_channels"]))
            out_ch = min(2, max(1, out_info["max_output_channels"]))

            # WASAPI shared mode with auto_convert: Windows converts between the engine's 48 kHz and whatever rate the
            # device runs at, so players never have to set 48 kHz by hand (measured: no added delay). Only for WASAPI
            # devices; an MME or DirectSound device refuses WASAPI settings and does its own conversion.
            extra_cfg = (wasapi_shared_settings(in_idx), wasapi_shared_settings(out_idx))

            # A numeric suggested latency really enlarges WASAPI's shared-mode buffer; "high" does not (it gives the
            # same buffer as "low"): more headroom against a busy PC for a few tens of ms more delay.
            latency = STABILITY_LATENCY_S if self.config.stability_buffer else "low"
            self.xruns = 0
            self.stalls = 0
            self._last_cb_t0 = 0.0
            self.peak_dsp_ms = 0.0
            self._dsp_blocks = 0
            self._dsp_total_ms = 0.0
            self._dsp_worst_ms = 0.0
            self.primary_stream = sd.Stream(
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                device=(in_idx, out_idx),
                channels=(in_ch, out_ch),
                dtype="float32",
                latency=latency,
                extra_settings=extra_cfg,
                callback=self._primary_callback,
            )
            self.primary_stream.start()
            try:  # what the driver really gave us, in milliseconds (input, output)
                lat = self.primary_stream.latency
                self.stream_latency_ms = (float(lat[0]) * 1000.0, float(lat[1]) * 1000.0)
            except Exception:
                self.stream_latency_ms = (0.0, 0.0)
            if self.stream_latency_ms[1] > 0.0:
                # measured on this PC: a hole starts once a stall passes about the output buffer; the gap between two
                # callbacks also holds one ordinary period, so 90% of the buffer is where the alarm sounds
                self.stall_threshold_s = max(0.025, 0.9 * self.stream_latency_ms[1] / 1000.0)

            self.is_running = True
            block_ms = (self.config.block_size / self.config.sample_rate) * 1000.0
            self._set_status(
                f"ACTIVE | {self.config.sample_rate}Hz | {self.config.block_size} spl ({block_ms:.1f}ms buf)"
            )

            if self.config.hear_myself and mon_idx is not None and mon_idx != out_idx:
                self.start_monitor(mon_idx)

            return True
        except Exception as exc:
            self.stop_engine()
            self.last_error = str(exc)
            self._set_status(f"Audio Stream Error: {exc}")
            return False

    def start_monitor(self, mon_idx: Optional[int]) -> bool:
        self.stop_monitor()
        if mon_idx is None or mon_idx == self._current_out_idx or not self.is_running:
            return False
        try:
            latency = STABILITY_LATENCY_S if self.config.stability_buffer else "low"
            mon_info = sd.query_devices(mon_idx)
            mon_ch = min(2, max(1, mon_info["max_output_channels"]))
            self._current_mon_idx = mon_idx
            self.monitor_ring.reset()
            self.monitor_stream = sd.OutputStream(
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                device=mon_idx,
                channels=mon_ch,
                dtype="float32",
                latency=latency,
                extra_settings=wasapi_shared_settings(mon_idx),
                callback=self._monitor_callback,
            )
            self.monitor_stream.start()
            return True
        except Exception as exc:
            self.monitor_stream = None
            return False

    def stop_monitor(self) -> None:
        if self.monitor_stream is not None:
            try:
                self.monitor_stream.stop()
                self.monitor_stream.close()
            except Exception:
                pass
            self.monitor_stream = None
        self.monitor_ring.reset()

    def stop_engine(self) -> None:
        self.is_running = False
        self.stop_monitor()
        if self.primary_stream is not None:
            try:
                self.primary_stream.stop()
                self.primary_stream.close()
            except Exception:
                pass
            self.primary_stream = None
        if self.dsp and self.dsp.onnx_streamer:
            self.dsp.onnx_streamer.unload()
        self.latest_in_db = -80.0
        self.latest_out_db = -80.0
        self._set_status("STOPPED")


def wasapi_shared_settings(dev_idx: int):
    """WasapiSettings(shared, auto_convert) for a WASAPI device, None for any other host API."""
    if not hasattr(sd, "WasapiSettings"):
        return None
    try:
        api_name = sd.query_hostapis(sd.query_devices(dev_idx)["hostapi"])["name"]
    except Exception:
        return None
    return sd.WasapiSettings(exclusive=False, auto_convert=True) if "WASAPI" in api_name.upper() else None


def run_audio_path_test(
    in_idx: int,
    out_idx: int,
    params: VoiceParameters,
    sample_rate: int = 48000,
    block_size: int = 256,
) -> Tuple[bool, List[str]]:
    """
    Verifies the full routing chain on real hardware, one stage at a time:
      1. Records 2 s from the selected mic and checks that it picks up sound.
      2. Runs that recording through the DSP chain and checks output and block timing.
      3. If the output is VB-CABLE, plays a 1 kHz tone into "CABLE Input" while recording
         "CABLE Output" (what Valorant hears) and checks that the tone arrives.
    Must not run while the audio engine holds the devices.
    """
    if not SOUNDDEVICE_AVAILABLE:
        return False, ["[FAIL] sounddevice library not installed."]

    lines: List[str] = []
    passed = True
    hostapis = sd.query_hostapis()

    def shared_settings(dev_idx: int):
        api_name = hostapis[sd.query_devices(dev_idx)["hostapi"]]["name"]
        if "WASAPI" in api_name.upper() and hasattr(sd, "WasapiSettings"):
            return sd.WasapiSettings(exclusive=False, auto_convert=True)
        return None

    def to_db(value: float) -> float:
        return max(-80.0, 20.0 * float(np.log10(value + 1e-12)))

    # 1. Microphone capture
    mic = sd.rec(
        int(2.0 * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=in_idx,
        extra_settings=shared_settings(in_idx),
    )
    sd.wait()
    mic = np.ascontiguousarray(mic[:, 0])
    mic_peak_db = to_db(float(np.max(np.abs(mic))))
    mic_rms_db = to_db(float(np.sqrt(np.mean(mic * mic))))
    if mic_peak_db < -50.0:
        passed = False
        lines.append(
            f"[FAIL] Mic: silent (peak {mic_peak_db:.1f} dBFS). Check the device and Windows mic privacy settings."
        )
    else:
        lines.append(f"[PASS] Mic: peak {mic_peak_db:.1f} dBFS, average {mic_rms_db:.1f} dBFS.")

    # 2. DSP chain on the recorded voice
    dsp = RealTimeDSPChain(sample_rate)
    dsp.update_parameters(params)
    n_blocks = len(mic) // block_size
    processed = np.zeros(n_blocks * block_size, dtype=np.float32)
    timings_ms: List[float] = []
    for i in range(n_blocks):
        block = mic[i * block_size : (i + 1) * block_size]
        t0 = time.perf_counter()
        processed[i * block_size : (i + 1) * block_size] = dsp.process_block(block)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    deadline_ms = (block_size / sample_rate) * 1000.0
    p99 = float(np.percentile(timings_ms, 99))
    if not np.all(np.isfinite(processed)) or float(np.max(np.abs(processed))) > 0.90:
        passed = False
        lines.append("[FAIL] DSP: output contains invalid samples or exceeds the limiter ceiling.")
    elif p99 >= deadline_ms:
        passed = False
        lines.append(
            f"[FAIL] DSP: slowest blocks take {p99:.2f} ms, over the {deadline_ms:.2f} ms budget. "
            "Expect crackling; try a larger buffer size."
        )
    else:
        lines.append(f"[PASS] DSP: slowest blocks take {p99:.2f} ms of a {deadline_ms:.2f} ms budget.")

    # 3. VB-CABLE loopback (what the game actually receives)
    out_info = sd.query_devices(out_idx)
    if "CABLE" not in out_info["name"].upper():
        lines.append(
            f"[SKIP] Cable: output '{out_info['name']}' is not VB-CABLE, so the game-side check was skipped."
        )
        return passed, lines

    cable_rec_idx = None
    for idx, dev in enumerate(sd.query_devices()):
        if (
            dev["max_input_channels"] > 0
            and dev["hostapi"] == out_info["hostapi"]
            and "CABLE OUTPUT" in dev["name"].upper()
        ):
            cable_rec_idx = idx
            break
    if cable_rec_idx is None:
        lines.append("[FAIL] Cable: could not find the 'CABLE Output' recording device.")
        return False, lines

    lead = int(0.25 * sample_rate)
    t = np.arange(int(1.5 * sample_rate), dtype=np.float32) / sample_rate
    tone = np.concatenate([np.zeros(lead, dtype=np.float32), 0.25 * np.sin(2.0 * np.pi * 1000.0 * t)])
    out_ch = min(2, max(1, out_info["max_output_channels"]))
    captured = sd.playrec(
        np.tile(tone[:, None], (1, out_ch)).astype(np.float32),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=(cable_rec_idx, out_idx),
        extra_settings=shared_settings(out_idx),
    )
    sd.wait()
    captured = captured[:, 0]

    onset = np.flatnonzero(np.abs(captured) > 0.025)
    if onset.size == 0:
        lines.append("[FAIL] Cable: nothing arrived on 'CABLE Output'. Check that VB-CABLE is enabled in Windows.")
        return False, lines

    segment = captured[onset[0] : onset[0] + sample_rate // 2]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    peak_hz = float(np.argmax(spectrum)) * sample_rate / len(segment)
    level_db = to_db(float(np.sqrt(np.mean(segment * segment))))
    latency_ms = (onset[0] - lead) * 1000.0 / sample_rate
    if abs(peak_hz - 1000.0) > 20.0:
        passed = False
        lines.append(f"[FAIL] Cable: signal arrived but is distorted (strongest tone at {peak_hz:.0f} Hz).")
    else:
        lines.append(
            f"[PASS] Cable: test tone reached 'CABLE Output' at {level_db:.1f} dBFS, "
            f"about {max(latency_ms, 0.0):.0f} ms after it was sent."
        )
    return passed, lines


# ============================================================================
# 5. VANGUARD-SAFE WIN32 GLOBAL HOTKEY THREAD (ZERO-HOOK RegisterHotKey API)
# ============================================================================

class Win32HotkeyListener(threading.Thread):
    """
    Global hotkeys through the Win32 RegisterHotKey API inside one dedicated OS message pump. It installs no
    WH_KEYBOARD_LL hook (it sees nothing but the combinations it registered, costs no CPU while idle and stays inside
    what Riot Vanguard accepts) and adds MOD_NOREPEAT, so holding a combination fires it once.

    ``bindings`` maps each hotkey id to ``(modifiers, virtual key)`` or None (Not set); the default is the factory set.
    ``callback_fn(hotkey_id)`` is called on this thread for every press: hand the id to your UI thread from there.

    Windows ties a registration to the thread that made it, so every RegisterHotKey / UnregisterHotKey call is made on
    this thread. Other threads change the bindings with ``apply`` / ``suspend`` / ``resume``: each queues a command and
    wakes the pump without waiting, and the outcome comes back through ``on_result(dict)``, also called on this thread
    (emit a Qt signal from it): ``{"token", "kind", "suspended", "table", "registered", "errors", "reverted", "error"}``.
    A change Windows refuses (another program owns the combination) is put back to the previous combination, and its
    id is listed in ``reverted``. ``registered`` maps each bound hotkey id to whether Windows accepts it right now.
    """

    WM_HOTKEY = 0x0312
    WM_QUIT = 0x0012
    WM_COMMANDS = 0x8001  # WM_APP + 1: commands are waiting in the queue
    MOD_ALT = 0x0001
    MOD_CONTROL = 0x0002
    MOD_SHIFT = 0x0004
    MOD_NOREPEAT = 0x4000
    ERROR_HOTKEY_ALREADY_REGISTERED = 1409

    HK_PRESET_1 = 101  # Ctrl + Alt + F1 by default
    HK_PRESET_2 = 102  # Ctrl + Alt + F2
    HK_PRESET_3 = 103  # Ctrl + Alt + F3
    HK_PRESET_4 = 104  # Ctrl + Alt + F4
    HK_BYPASS   = 105  # Ctrl + Alt + F9
    HK_MUTE     = 106  # Ctrl + Alt + F10

    @staticmethod
    def default_table() -> Dict[int, Optional[Tuple[int, int]]]:
        return hk.to_table(hk.default_bindings())

    def __init__(self, callback_fn, bindings: Optional[Dict[int, Optional[Tuple[int, int]]]] = None, on_result=None):
        super().__init__(daemon=True, name="vcomms-hotkeys")
        self.callback_fn = callback_fn
        self.on_result = on_result
        self.thread_id: Optional[int] = None
        self.running = True
        self.registered: Dict[int, bool] = {}
        self.errors: Dict[int, int] = {}                 # last Win32 error per hotkey id that Windows refused
        self.last_error = ""                             # an unexpected exception while running a command (the pump survives it)
        self.ready = threading.Event()                   # set once the first registrations have been attempted
        self._wanted: Dict[int, Optional[Tuple[int, int]]] = dict(
            bindings if bindings is not None else self.default_table()
        )
        self._active: Dict[int, Tuple[int, int]] = {}    # what Windows currently has registered for this thread
        self._suspended = False
        self._commands: "queue.SimpleQueue[tuple]" = queue.SimpleQueue()
        self._owner: Optional[int] = None
        self._user32 = None

    # ---- any thread ----------------------------------------------------------------------------------------
    def apply(self, bindings: Dict[int, Optional[Tuple[int, int]]], token=None, resume: bool = False) -> None:
        """Makes the registrations match ``bindings`` (``resume`` also ends a suspension)."""
        self._post(("apply", token, dict(bindings), resume))

    def suspend(self, token=None) -> None:
        """Unregisters everything (the bindings are remembered), so the combinations reach whoever is listening."""
        self._post(("suspend", token))

    def resume(self, token=None) -> None:
        self._post(("resume", token))

    def stop(self) -> None:
        self.running = False
        if sys.platform == "win32" and self.thread_id is not None and self._user32 is not None:
            self._user32.PostThreadMessageW(self.thread_id, self.WM_QUIT, 0, 0)

    def _post(self, command: tuple) -> None:
        self._commands.put(command)  # before looking at thread_id: run() drains the queue once it is up
        if sys.platform == "win32" and self.thread_id is not None and self._user32 is not None:
            self._user32.PostThreadMessageW(self.thread_id, self.WM_COMMANDS, 0, 0)

    # ---- this thread only ----------------------------------------------------------------------------------
    def _register(self, hk_id: int, combo: Tuple[int, int]) -> int:
        """0 when Windows accepted the combination, otherwise its Win32 error code."""
        if threading.get_ident() != self._owner:
            raise RuntimeError("hotkeys must be registered on the listener thread")
        mods, vk = combo
        if self._user32.RegisterHotKey(None, hk_id, mods | self.MOD_NOREPEAT, vk):
            self._active[hk_id] = combo
            self.errors.pop(hk_id, None)
            return 0
        error = ctypes.get_last_error() or 1
        self.errors[hk_id] = error
        return error

    def _unregister(self, hk_id: int) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("hotkeys must be unregistered on the listener thread")
        if self._active.pop(hk_id, None) is not None:
            self._user32.UnregisterHotKey(None, hk_id)

    def _sync(self) -> Dict[int, int]:
        """Brings the registrations in line with ``_wanted`` (none while suspended). Everything that changes is
        unregistered before anything new is registered, so two actions can swap combinations. Returns the Win32 error
        for every combination Windows refused."""
        refused: Dict[int, int] = {}
        for hk_id in list(self._active):
            if self._suspended or self._active[hk_id] != self._wanted.get(hk_id):
                self._unregister(hk_id)
        if not self._suspended:
            for hk_id, combo in self._wanted.items():
                if combo is not None and hk_id not in self._active:
                    error = self._register(hk_id, combo)
                    if error:
                        refused[hk_id] = error
        return refused

    def _apply(self, table, resume: bool) -> Tuple[Dict[int, int], List[int]]:
        previous = dict(self._wanted)
        if resume:
            self._suspended = False
        for hk_id, entry in table.items():
            self._wanted[hk_id] = tuple(entry) if entry else None
        refused = self._sync()
        reverted: List[int] = []
        for hk_id in refused:
            if previous.get(hk_id) != self._wanted.get(hk_id):  # a change Windows refused: keep what worked before
                back = previous.get(hk_id)
                if back is not None and any(o != hk_id and c == back for o, c in self._wanted.items()):
                    # another action took that combination over in this same change (a profile load or Restore
                    # defaults moves several at once): Not set, rather than two actions on one combination
                    back = None
                self._wanted[hk_id] = back
                reverted.append(hk_id)
                if back is not None:
                    self._register(hk_id, back)
        return refused, reverted

    def _publish(self, token, kind: str, refused: Dict[int, int], reverted: List[int]) -> None:
        if not self._suspended:  # while suspended nothing is registered on purpose: that is not "taken by another app"
            self.registered = {i: (i in self._active) for i, combo in self._wanted.items() if combo is not None}
        if self.on_result is None:
            return
        try:
            self.on_result(
                {
                    "token": token,
                    "kind": kind,
                    "suspended": self._suspended,
                    "table": dict(self._wanted),
                    "registered": dict(self.registered),
                    "errors": dict(refused),
                    "reverted": list(reverted),
                    "error": self.last_error,
                }
            )
        except Exception:
            pass  # the listener must outlive a bug in the receiver

    def _run_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            kind, token = command[0], command[1]
            self.last_error = ""
            try:
                if kind == "apply":
                    refused, reverted = self._apply(command[2], command[3])
                elif kind == "suspend":
                    self._suspended = True
                    refused, reverted = self._sync(), []
                else:  # resume
                    self._suspended = False
                    refused, reverted = self._sync(), []
            except Exception as exc:  # a bug here must not end the message pump: the hotkeys would just stop working
                refused, reverted = {}, []
                self.last_error = f"{type(exc).__name__}: {exc}"
            self._publish(token, kind, refused, reverted)

    def run(self) -> None:
        if sys.platform != "win32":
            self.ready.set()
            return
        user32 = self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.UnregisterHotKey.restype = wintypes.BOOL
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = ctypes.c_int
        user32.PeekMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT,
        ]
        user32.PeekMessageW.restype = wintypes.BOOL
        user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.PostThreadMessageW.restype = wintypes.BOOL
        user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        self._owner = threading.get_ident()

        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0x0400, 0x0400, 0)  # creates this thread's message queue first
        self.thread_id = int(kernel32.GetCurrentThreadId())

        refused = self._sync()
        self._publish(None, "startup", refused, [])
        self.ready.set()
        self._run_commands()  # anything queued before the pump existed

        while self.running:
            status = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if status == 0 or status == -1:
                break
            if msg.message == self.WM_HOTKEY:
                try:
                    self.callback_fn(int(msg.wParam))
                except Exception:
                    pass
            elif msg.message == self.WM_COMMANDS:
                self._run_commands()
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        for hk_id in list(self._active):
            self._unregister(hk_id)


# ============================================================================
# 6. PROFILES, PRESET COPIES & DEFAULT DEVICE SELECTION (UI-agnostic helpers)
# ============================================================================

PROFILE_FILENAME = "tactical_voice_profile.json"  # auto-loaded from the folder the app is launched from


def preset_parameters(name: str) -> VoiceParameters:
    """A fresh, independent copy of a tactical preset's parameters."""
    return VoiceParameters(**asdict(TACTICAL_PRESETS[name]))


def prewarm_filter_designs(sample_rate: int = 48000, extra: Iterable[VoiceParameters] = ()) -> None:
    """
    Designs the filters every built-in preset (and any ``extra`` ones, such as the player's saved presets) needs, a
    few ms each, while nothing is running, so switching presets later, by hotkey in the middle of a match, only
    swaps a pointer instead of redesigning them.
    """
    chain = RealTimeDSPChain(sample_rate)
    for params in list(TACTICAL_PRESETS.values()) + list(extra):
        chain.prepare_parameters(params)


# Numbers a profile may set, and where they must lie. A value outside a range (or of the wrong type) is ignored and
# the current setting kept: profiles are hand-editable files, and a block size of 999999 or a 192 kHz rate (the
# pitch shifter's 36 ms window would no longer fit its delay line) must never reach the audio callback.
PROFILE_LIMITS: Dict[str, Tuple[float, float]] = {
    "sample_rate": (16000, 96000),
    "block_size": (32, MAX_BLOCK_SIZE),
    "monitor_volume": (0.0, 1.0),
}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _profile_value(name: str, current, value):
    """``value`` checked against the type (and limits) of the setting it replaces; ``current`` when it does not fit."""
    if isinstance(current, bool):
        return value if isinstance(value, bool) else current
    if isinstance(current, str):
        return value if isinstance(value, str) else current
    if not _is_number(value):
        return current
    lo, hi = PROFILE_LIMITS.get(name, (float("-inf"), float("inf")))
    if isinstance(current, int):
        if float(value) != int(value) or not lo <= value <= hi:
            return current
        return int(value)
    return float(min(max(float(value), lo), hi))


def _profile_params(raw: dict) -> VoiceParameters:
    """Voice parameters from a profile: missing ones take their defaults, every number is clamped to its slider."""
    params = VoiceParameters()
    if isinstance(raw.get("name"), str):
        params.name = raw["name"]
    for key, (lo, hi) in PARAM_RANGES.items():
        value = raw.get(key)
        if _is_number(value):
            setattr(params, key, float(min(max(float(value), lo), hi)))
    return params


def load_profile_into(config: AppConfig, path: Path) -> bool:
    """
    Applies a saved JSON profile onto ``config`` in place (the audio engine keeps seeing the same object). Everything
    is checked first and then applied in one go: a file that cannot be read changes nothing, and a single bad value
    is skipped instead of leaving the settings half loaded.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, RecursionError):
        return False
    if not isinstance(raw, dict):
        return False
    updates = {}
    for f in fields(AppConfig):
        if f.name in ("current_params", "hotkeys") or f.name not in raw:
            continue
        updates[f.name] = _profile_value(f.name, getattr(config, f.name), raw[f.name])
    params_raw = raw.get("current_params")
    if isinstance(params_raw, dict) and params_raw:
        updates["current_params"] = _profile_params(params_raw)
    if isinstance(raw.get("hotkeys"), dict):  # a profile from before hotkeys could be changed has none: keep the current ones
        updates["hotkeys"] = hk.normalize_bindings(raw["hotkeys"])
    for name, value in updates.items():
        setattr(config, name, value)
    return True


def save_profile(config: AppConfig, path: Path) -> None:
    """Atomic, like the preset library and hotkeys.json: the old file stays whole until the new one is complete."""
    path = Path(path)
    data = json.dumps(asdict(config), indent=2)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _find_device(saved: str, entries: List[Tuple[int, str]]) -> Optional[int]:
    """Position of a saved device: its exact label first, then the first label that contains it; None when it is gone.
    (Containment alone would take "USB Headset" for a saved "Headset".)"""
    if not saved:
        return None
    for i, (_, label) in enumerate(entries):
        if label == saved:
            return i
    return next((i for i, (_, label) in enumerate(entries) if saved in label), None)


def pick_device_targets(
    config: AppConfig,
    inputs: List[Tuple[int, str]],
    outputs: List[Tuple[int, str]],
) -> Tuple[int, int, int]:
    """
    Which entries (positions in ``inputs`` / ``outputs``, not device ids) to preselect for mic, cable and monitor.
    A saved device name wins; otherwise (nothing saved, or the saved device is unplugged) the mic is the first input,
    the cable is the first "CABLE Input" (outputs are listed WASAPI-first, so that is the WASAPI endpoint) and the
    monitor is the first output that is neither the cable output nor another virtual cable.
    """
    in_target = _find_device(config.input_device_name, inputs)
    in_target = 0 if in_target is None else in_target

    cable_idx = next((i for i, (_, label) in enumerate(outputs) if "CABLE INPUT" in label.upper()), None)
    out_target = _find_device(config.output_device_name, outputs)
    if out_target is None:
        out_target = cable_idx if cable_idx is not None else 0

    mon_target = _find_device(config.monitor_device_name, outputs)
    if mon_target is None:
        others = [i for i in range(len(outputs)) if i != out_target]
        physical = [i for i in others if "CABLE" not in outputs[i][1].upper()]
        mon_target = (physical or others or [0])[0]

    return in_target, out_target, mon_target


# ============================================================================
# 7. AUTOMATED SELF-TEST, MEMORY AUDIT & LATENCY BENCHMARK SUITE
# ============================================================================

def run_production_self_test() -> int:
    """
    Executes automated unit tests, numerical stability checks, SPSC ring buffer wraps,
    Zero-Roll mirror contiguity checks, and a 1,000-block latency & memory leak audit.
    """
    print("=" * 72)
    print("V-COMMS // AUTOMATED SELF-TEST & REAL-TIME COMPLEXITY BENCHMARK")
    print("=" * 72)

    sr = 48000
    block_size = 256
    block_deadline_ms = (block_size / sr) * 1000.0
    dsp = RealTimeDSPChain(sample_rate=sr)

    # 1. Test Wait-Free SPSC Ring Buffer wrap-around & drift protection
    spsc = WaitFreeSpscRingBuffer(max_lag_samples=1024)
    test_in = np.linspace(-0.8, 0.8, block_size, dtype=np.float32)
    test_out = np.zeros(block_size, dtype=np.float32)
    for _ in range(120):  # Force multiple wrap-arounds across 16,384 boundary
        spsc.write(test_in, gain=0.5)
        spsc.read_into(test_out)
    assert np.allclose(test_out, test_in * 0.5, atol=1e-6), "SPSC Ring Buffer wrap-around failed!"
    spsc.reset()
    spsc.write(test_in)
    peeked = spsc.peek_chunk(block_size)
    assert np.allclose(peeked, test_in, atol=1e-6), "peek_chunk failed!"
    assert spsc.available_read() == block_size, "peek_chunk modified read cursor!"
    spsc.advance_read(block_size)
    assert spsc.available_read() == 0, "advance_read failed!"
    print("[PASS] WaitFreeSpscRingBuffer: Wrap-around, drift guard & peek/advance verified.")

    # 2. Test ZeroRollMirrorBuffer contiguity & exact sample order
    mirror = ZeroRollMirrorBuffer(window_size=1024)
    stream_Reference = np.arange(3000, dtype=np.float32)
    for i in range(0, 3000, 250):
        view = mirror.push_and_get_view(stream_Reference[i : i + 250])
        assert view.flags["C_CONTIGUOUS"], "ZeroRollMirrorBuffer view is not C-contiguous!"
    assert np.array_equal(view, stream_Reference[-1024:]), "ZeroRollMirrorBuffer sample mismatch!"
    print("[PASS] ZeroRollMirrorBuffer: O(1) C-contiguous sliding window verified.")

    # 3. Benchmark all 4 Tactical Presets + Memory Allocation Audit
    t = np.arange(block_size, dtype=np.float32) / sr
    synth_voice = (0.35 * np.sin(2.0 * np.pi * 140.0 * t) + 0.15 * np.sin(2.0 * np.pi * 420.0 * t)).astype(
        np.float32
    )

    # Verify zero heap allocation during block processing
    tracemalloc.start()
    for preset_name, preset_params in TACTICAL_PRESETS.items():
        dsp.update_parameters(preset_params)
        for _ in range(20):
            _ = dsp.process_block(synth_voice, use_onnx=False)
    snap_before = tracemalloc.take_snapshot()
    for _ in range(50):
        _ = dsp.process_block(synth_voice, use_onnx=False)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    only_engine = [tracemalloc.Filter(True, __file__)]
    retained = sum(
        stat.size_diff
        for stat in snap_after.filter_traces(only_engine).compare_to(snap_before.filter_traces(only_engine), "lineno")
    )
    assert retained < 16 * 1024, f"DSP chain retained {retained} bytes over 50 blocks (memory leak)!"
    print(f"[PASS] Memory audit: {max(retained, 0)} bytes retained by the DSP chain over 50 blocks.")

    timings_ms: List[float] = []
    for preset_name, preset_params in TACTICAL_PRESETS.items():
        dsp.update_parameters(preset_params)
        # Warm up caches
        for _ in range(20):
            _ = dsp.process_block(synth_voice, use_onnx=False)

        for _ in range(250):
            t0 = time.perf_counter()
            out = dsp.process_block(synth_voice, use_onnx=False)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

            assert out.dtype == np.float32, f"{preset_name} output dtype is not float32!"
            assert np.all(np.isfinite(out)), f"{preset_name} produced NaN/Inf!"
            assert float(np.max(np.abs(out))) <= 0.90, f"{preset_name} exceeded -1.0 dBFS limiter ceiling!"
        print(f"[PASS] Preset '{preset_name}': 250 blocks clean | Max Peak: {np.max(np.abs(out)):.3f}")

    arr_ms = np.array(timings_ms)
    p50 = float(np.percentile(arr_ms, 50))
    p95 = float(np.percentile(arr_ms, 95))
    p99 = float(np.percentile(arr_ms, 99))
    duty_cycle = (p50 / block_deadline_ms) * 100.0

    print("-" * 72)
    print(f"Hardware Block Deadline (256 spl @ 48kHz) : {block_deadline_ms:.2f} ms")
    print(f"Callback Execution Latency (P50 Median)   : {p50:.3f} ms ({duty_cycle:.1f}% duty cycle)")
    print(f"Callback Execution Latency (P95)          : {p95:.3f} ms")
    print(f"Callback Execution Latency (P99)          : {p99:.3f} ms")
    print("=" * 72)
    assert p99 < block_deadline_ms, "P99 callback time exceeded hardware buffer deadline!"
    print("ALL PRODUCTION CHECKS PASSED.")
    return 0


def configure_windows_process_isolation() -> None:
    """Pins process to the final 2-4 E-cores on hybrid Intel HX CPUs and boosts priority."""
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL

        handle = kernel32.GetCurrentProcess()

        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS)

        total_threads = os.cpu_count() or 24
        if total_threads >= 12:
            # On 12+ thread CPUs (e.g. Intel HX/H/P i5/i7/i9), pin to the last 4 threads (all E-cores)
            # providing ample headroom for audio + GUI threads while keeping P-cores 100% free for games
            ecore_mask = (15 << (total_threads - 4))
            kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(ecore_mask))
        elif total_threads >= 8:
            # On 8-11 thread CPUs, pin to the last 2 threads
            ecore_mask = (3 << (total_threads - 2))
            kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(ecore_mask))
    except Exception:
        pass
