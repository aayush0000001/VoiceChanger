"""
Small ONNX models with the same contracts as the real ones, so the whole RVC path can run through ONNX Runtime
in the tests without the 360 MB HuBERT download:

  hubert:  source [1, 1, N] -> Conv(768 ch, kernel 400, stride 320) -> [1, T, 768]     (wav2vec2 front-end geometry)
           with a 12 MB constant inside, because the engine only accepts a HuBERT file over 10 MB
  rvc_*:   an RVC v2 generator stand-in that synthesises a sine at the frame pitch (nsff0 / pitchf) at 480 samples
           per frame: the output frequency proves the pitch made it through the pipeline. Every other input is
           consumed (times zero), so ONNX Runtime requires them exactly like a real export does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = [helper.make_opsetid("", 17)]


def _save(graph, path: Path) -> Path:
    model = helper.make_model(graph, opset_imports=OPSET)
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def build_hubert(path: Path, dtype=TensorProto.FLOAT) -> Path:
    rng = np.random.default_rng(1)
    weight = (rng.standard_normal((768, 1, 400)) * 0.02).astype(np.float32)
    padding = np.zeros(3_000_000, dtype=np.float32)  # pushes the file past the engine's 10 MB HuBERT check
    nodes = []
    src = "source"
    if dtype == TensorProto.FLOAT16:
        nodes.append(helper.make_node("Cast", ["source"], ["source32"], to=TensorProto.FLOAT))
        src = "source32"
    nodes += [
        helper.make_node("Conv", [src, "W"], ["conv"], strides=[320], kernel_shape=[400]),
        helper.make_node("Transpose", ["conv"], ["feats"], perm=[0, 2, 1]),
        helper.make_node("ReduceSum", ["pad"], ["pad_sum"], keepdims=0),
        helper.make_node("Mul", ["pad_sum", "zero"], ["pad_zero"]),
        helper.make_node("Add", ["feats", "pad_zero"], ["units"]),
    ]
    graph = helper.make_graph(
        nodes,
        "hubert_stub",
        [helper.make_tensor_value_info("source", dtype, [1, 1, "N"])],
        [helper.make_tensor_value_info("units", TensorProto.FLOAT, [1, "T", 768])],
        [numpy_helper.from_array(weight, "W"), numpy_helper.from_array(padding, "pad"),
         numpy_helper.from_array(np.array(0.0, np.float32), "zero")],
    )
    return _save(graph, path)


def build_rvc(path: Path, naming: str = "okada", feats_dtype=TensorProto.FLOAT, feat_dim: int = 768,
              extra_input: str = "") -> Path:
    """naming 'okada': phone, phone_lengths, pitch, nsff0, sid.  'webui': phone, phone_lengths, pitch, pitchf, ds, rnd."""
    f0_name, spk_name = ("nsff0", "sid") if naming == "okada" else ("pitchf", "ds")
    inputs = [
        helper.make_tensor_value_info("phone", feats_dtype, [1, "T", feat_dim]),
        helper.make_tensor_value_info("phone_lengths", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("pitch", TensorProto.INT64, [1, "T"]),
        helper.make_tensor_value_info(f0_name, TensorProto.FLOAT, [1, "T"]),
        helper.make_tensor_value_info(spk_name, TensorProto.INT64, [1]),
    ]
    if naming == "webui":
        inputs.append(helper.make_tensor_value_info("rnd", TensorProto.FLOAT, [1, 192, "T"]))
    if extra_input:
        inputs.append(helper.make_tensor_value_info(extra_input, TensorProto.FLOAT, [1]))
    consumed = []
    nodes = [
        # sine at the frame pitch, 480 samples per frame
        helper.make_node("Unsqueeze", [f0_name, "ax2"], ["f0_3d"]),
        helper.make_node("Shape", [f0_name], ["f0_shape"]),
        helper.make_node("Concat", ["f0_shape", "c480"], ["up_shape"], axis=0),
        helper.make_node("Expand", ["f0_3d", "up_shape"], ["f0_up"]),
        helper.make_node("Reshape", ["f0_up", "flat"], ["f0_flat"]),
        helper.make_node("Mul", ["f0_flat", "w_step"], ["dphase"]),
        helper.make_node("CumSum", ["dphase", "ax_scalar"], ["phase"]),
        helper.make_node("Sin", ["phase"], ["wave"]),
        helper.make_node("Mul", ["wave", "amp"], ["wave_amp"]),
    ]
    for name, kind in (("phone", feats_dtype), ("phone_lengths", TensorProto.INT64), ("pitch", TensorProto.INT64),
                       (spk_name, TensorProto.INT64)):
        nodes.append(helper.make_node("Cast", [name], [f"{name}_f"], to=TensorProto.FLOAT))
        consumed.append(f"{name}_f")
    if naming == "webui":
        consumed.append("rnd")
    if extra_input:
        consumed.append(extra_input)
    prev = "wave_amp"
    for i, name in enumerate(consumed):
        nodes += [
            helper.make_node("ReduceSum", [name], [f"s{i}"], keepdims=0),
            helper.make_node("Mul", [f"s{i}", "zero"], [f"z{i}"]),
            helper.make_node("Add", [prev, f"z{i}"], [f"acc{i}"]),
        ]
        prev = f"acc{i}"
    nodes.append(helper.make_node("Unsqueeze", [prev, "ax1"], ["audio"]))
    graph = helper.make_graph(
        nodes,
        f"rvc_stub_{naming}",
        inputs,
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "S"])],
        [
            numpy_helper.from_array(np.array([2], np.int64), "ax2"),
            numpy_helper.from_array(np.array([1], np.int64), "ax1"),
            numpy_helper.from_array(np.array(1, np.int64), "ax_scalar"),  # CumSum takes its axis as a scalar
            numpy_helper.from_array(np.array([480], np.int64), "c480"),
            numpy_helper.from_array(np.array([1, -1], np.int64), "flat"),
            numpy_helper.from_array(np.array(2.0 * np.pi / 48000.0, np.float32), "w_step"),
            numpy_helper.from_array(np.array(0.3, np.float32), "amp"),
            numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        ],
    )
    return _save(graph, path)


def build_direct(path: Path) -> Path:
    graph = helper.make_graph(
        [helper.make_node("Mul", ["audio", "half"], ["out"])],
        "direct_stub",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 4096])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4096])],
        [numpy_helper.from_array(np.array(0.5, np.float32), "half")],
    )
    return _save(graph, path)


def build_all(folder: Path) -> Dict[str, Path]:
    folder = Path(folder)
    with_hubert = folder / "with_hubert"
    fp16_hubert = folder / "fp16_hubert"
    no_hubert = folder / "no_hubert"
    for d in (with_hubert, fp16_hubert, no_hubert):
        d.mkdir(parents=True, exist_ok=True)
    build_hubert(with_hubert / "vec-768-layer-12.onnx")
    build_hubert(fp16_hubert / "vec-768-layer-12.onnx", dtype=TensorProto.FLOAT16)
    return {
        "rvc_okada": build_rvc(with_hubert / "rvc_okada.onnx", "okada"),
        "rvc_webui": build_rvc(with_hubert / "rvc_webui.onnx", "webui"),
        "rvc_fp16": build_rvc(fp16_hubert / "rvc_fp16.onnx", "okada", feats_dtype=TensorProto.FLOAT16),
        "rvc_unknown_input": build_rvc(with_hubert / "rvc_unknown.onnx", "okada", extra_input="mystery"),
        "rvc_wrong_dim": build_rvc(with_hubert / "rvc_wrong_dim.onnx", "okada", feat_dim=256),
        "rvc_no_hubert": build_rvc(no_hubert / "rvc_alone.onnx", "okada"),
        "direct": build_direct(no_hubert / "direct.onnx"),
        "hubert": with_hubert / "vec-768-layer-12.onnx",
    }
