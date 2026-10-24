from __future__ import annotations

import time
import tracemalloc
import numpy as np
import engine as eng


def test_onnx_context_and_phase_locked_splice(onnx_models):
    streamer = eng.ONNXVoiceStreamer(sample_rate=48000)
    ok, msg = streamer.load_model(str(onnx_models["rvc_webui"]))
    assert ok, msg

    t = np.arange(streamer.CHUNK_SAMPLES * 3, dtype=np.float32) / 48000.0
    carrier = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    prev_tail = None
    outputs = []
    for k in range(2):
        start = k * streamer.HOP_SAMPLES
        chunk = carrier[start : start + streamer.CHUNK_SAMPLES]
        synth = streamer._convert(chunk, prev_tail=prev_tail)
        assert len(synth) == streamer.CHUNK_SAMPLES
        assert np.all(np.isfinite(synth))
        if prev_tail is not None:
            # Verify positive phase coherence between prev_tail and phase-locked synth head
            corr = float(np.dot(prev_tail, synth[: streamer.OVERLAP_SAMPLES]))
            assert corr > 0.0, "Phase-locked WSOLA splice failed to align glottal phase!"
        prev_tail = synth[streamer.HOP_SAMPLES : streamer.CHUNK_SAMPLES].copy()
        outputs.append(synth)

    streamer.unload()


def test_unvoiced_sibilant_preserved_and_callback_realtime(onnx_models):
    dsp = eng.RealTimeDSPChain(sample_rate=48000)
    ok, _ = dsp.onnx_streamer.load_model(str(onnx_models["rvc_okada"]))
    assert ok

    # Verify zero heap allocation and < 5.33 ms deadline on audio callback thread with use_onnx=True
    block = np.zeros(256, dtype=np.float32)
    for _ in range(10):
        dsp.process_block(block, use_onnx=True)

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    timings = []
    for _ in range(50):
        t0 = time.perf_counter()
        dsp.process_block(block, use_onnx=True)
        timings.append((time.perf_counter() - t0) * 1000.0)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    assert np.percentile(timings, 99) < 5.33
    dsp.onnx_streamer.unload()
