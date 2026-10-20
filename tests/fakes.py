"""A scripted stand-in for the sounddevice module: a fixed device list, streams whose callbacks the test drives, and
rec / playrec answers the test sets."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

WASAPI, MME = 0, 1

DEVICES = [
    {"name": "Microphone (Realtek Audio)", "hostapi": WASAPI, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Speakers (Realtek Audio)", "hostapi": WASAPI, "max_input_channels": 0, "max_output_channels": 2},
    {"name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": WASAPI, "max_input_channels": 0, "max_output_channels": 2},
    {"name": "CABLE Output (VB-Audio Virtual Cable)", "hostapi": WASAPI, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Headphones (USB Headset)", "hostapi": WASAPI, "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Headset Microphone (USB Headset)", "hostapi": WASAPI, "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Microphone (Realtek Audio)", "hostapi": MME, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Speakers (Realtek Audio)", "hostapi": MME, "max_input_channels": 0, "max_output_channels": 2},
]
HOSTAPIS = [
    {"name": "Windows WASAPI", "default_input_device": 0, "default_output_device": 1},
    {"name": "MME", "default_input_device": 6, "default_output_device": 7},
]
MIC, SPEAKERS, CABLE_IN, CABLE_OUT, HEADPHONES, HEADSET_MIC = 0, 1, 2, 3, 4, 5


class WasapiSettings:
    def __init__(self, exclusive=False, auto_convert=False):
        self.exclusive, self.auto_convert = exclusive, auto_convert


class FakeStream:
    def __init__(self, sd: "FakeSoundDevice", kind: str, **kwargs):
        if sd.fail_open:
            raise RuntimeError(sd.fail_open)
        self.sd, self.kind, self.kwargs = sd, kind, kwargs
        self.callback: Callable = kwargs["callback"]
        self.started = self.closed = False
        self.latency = (0.010, 0.020) if kind == "duplex" else 0.020
        sd.streams.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def run_block(self, indata: Optional[np.ndarray] = None, status=0) -> np.ndarray:
        """Calls the stream's callback once like PortAudio would and returns what it wrote."""
        frames = self.kwargs["blocksize"]
        channels = self.kwargs["channels"]
        if self.kind == "duplex":
            in_ch, out_ch = channels
            outdata = np.zeros((frames, out_ch), dtype=np.float32)
            self.callback(np.ascontiguousarray(indata, dtype=np.float32), outdata, frames, None, status)
        else:
            outdata = np.zeros((frames, channels), dtype=np.float32)
            self.callback(outdata, frames, None, status)
        return outdata


class FakeSoundDevice:
    WasapiSettings = WasapiSettings

    def __init__(self):
        self.devices = [dict(d) for d in DEVICES]
        self.hostapis = [dict(h) for h in HOSTAPIS]
        self.streams: List[FakeStream] = []
        self.fail_open = ""
        self.rec_result: Optional[np.ndarray] = None
        self.playrec_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.initialized = True

    def query_devices(self, device=None):
        if device is None:
            return [dict(d) for d in self.devices]
        return dict(self.devices[device])

    def query_hostapis(self, index=None):
        if index is None:
            return [dict(h) for h in self.hostapis]
        return dict(self.hostapis[index])

    def Stream(self, **kwargs):  # noqa: N802 (sounddevice's names)
        return FakeStream(self, "duplex", **kwargs)

    def OutputStream(self, **kwargs):  # noqa: N802
        return FakeStream(self, "output", **kwargs)

    def rec(self, frames, samplerate, channels, dtype, device=None, extra_settings=None):
        data = self.rec_result if self.rec_result is not None else np.zeros(frames, np.float32)
        return np.asarray(data[:frames], dtype=np.float32).reshape(-1, 1)

    def playrec(self, data, samplerate, channels, dtype, device=None, extra_settings=None):
        out = self.playrec_fn(np.asarray(data)[:, 0]) if self.playrec_fn else np.zeros(len(data), np.float32)
        return np.asarray(out, dtype=np.float32).reshape(-1, 1)

    def wait(self):
        return None

    def _terminate(self):
        self.initialized = False

    def _initialize(self):
        self.initialized = True
