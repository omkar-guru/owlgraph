"""Produce the fp16 and int8 ONNX graphs that TensorRT 11 builds from.

TensorRT 11 networks are strongly typed, so precision is decided here rather
than at build time:

fp16
    Cast the graph's weights and activations to fp16.  Cheap and lossless to
    apply; the accuracy question is whether fp16 range suffices, not whether the
    builder honours it.

int8
    Run calibrated post-training quantization, which inserts explicit
    QuantizeLinear/DequantizeLinear pairs.  Calibration needs real frames from
    the evaluation domain - activation ranges taken from the wrong distribution
    are the usual cause of an int8 accuracy collapse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# TensorRT requires these in int32/int64/fp32 regardless of the surrounding
# graph, so they are kept out of the fp16 conversion. Range appears once the
# position encoding is interpolated for an off-native input resolution.
FP16_OP_BLOCK_LIST = ("Range",)


def to_fp16_onnx(
    src: Path,
    dst: Path,
    keep_io_fp32: bool = False,
    op_block_list: tuple[str, ...] = FP16_OP_BLOCK_LIST,
) -> Path:
    """Convert an fp32 ONNX graph to fp16, I/O included.

    I/O is converted too, rather than left fp32 behind boundary casts.  A
    strongly-typed TensorRT 11 network will not insert conversions on its own,
    so an fp32 input feeding the first fp16 convolution is a hard build failure
    ("`input` and `kernel` must be of same type") rather than a silent cast.
    :class:`~sggpipeline.detect.trt_runner.TRTRunner` reads each binding's dtype
    off the engine, so the host side needs no change.
    """
    import onnx
    from onnxconverter_common import float16

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(src))
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_fp32,
        disable_shape_infer=False,
        op_block_list=list(op_block_list),
    )
    _retarget_stale_fp32_casts(converted)
    onnx.save(converted, str(dst))
    return dst


def _retarget_stale_fp32_casts(model) -> int:
    """Point leftover ``Cast(to=float32)`` nodes at float16.

    HF's vision embedding does ``pixel_values.to(patch_embedding.weight.dtype)``,
    which traces to a Cast with the dtype *frozen to whatever the weights were at
    export time* - fp32.  The fp16 converter rewrites initializers and value
    types but not that attribute, leaving an fp32 tensor feeding an fp16
    convolution.  A strongly-typed TensorRT network rejects that outright.

    Only applied when no fp32 initializer survives conversion, so a graph where
    the converter deliberately kept a region in fp32 is left alone.
    """
    from onnx import TensorProto

    initializer_dtype = {t.name: t.data_type for t in model.graph.initializer}

    # Map each tensor to the nodes consuming it, so a cast is only retargeted
    # when its consumers genuinely want fp16. Blanket rewriting would also flip
    # the casts the converter deliberately places around blocked ops such as
    # Range, which must stay fp32 for TensorRT.
    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def wants_fp16(node) -> bool:
        weight_types = [
            initializer_dtype[name] for name in node.input if name in initializer_dtype
        ]
        if not weight_types:
            return False
        return all(dtype == TensorProto.FLOAT16 for dtype in weight_types)

    rewritten = 0
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        targets = consumers.get(node.output[0], [])
        if not targets or not all(wants_fp16(t) for t in targets):
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT16
                rewritten += 1
    return rewritten


class CalibrationReader:
    """Feeds preprocessed frames to modelopt's PTQ pass, one at a time.

    Yields the same two inputs the graph was exported with.  ``query_embeds`` is
    held constant at the real AG text embeddings: quantizing the class head
    against random query vectors would calibrate it for a distribution it never
    sees at inference.

    Frames are produced **lazily**.  Materializing the whole calibration set is
    the obvious implementation and it does not fit: 384 frames at 960x960 fp32
    is 3.96 GB of resident host memory, before modelopt's own graph copies and
    the ONNX Runtime session, which is enough to take down a 15 GB machine.
    Pass either a preprocessed array or a list of image paths plus a callable
    that turns one path into a ``(1, 3, S, S)`` array.
    """

    def __init__(
        self,
        pixel_values: np.ndarray | None = None,
        query_embeds: np.ndarray | None = None,
        image_paths: list | None = None,
        loader=None,
    ):
        if query_embeds is None:
            raise ValueError("query_embeds is required")
        self.query_embeds = query_embeds.astype(np.float32)
        self.image_paths = list(image_paths) if image_paths is not None else None
        self.loader = loader

        if self.image_paths is not None:
            if loader is None:
                raise ValueError("image_paths requires a loader callable")
            self.pixel_values = None
            self._count = len(self.image_paths)
        else:
            if pixel_values is None or pixel_values.ndim != 4:
                raise ValueError("Expected (N,3,S,S) pixel values or image_paths")
            self.pixel_values = pixel_values.astype(np.float32)
            self._count = int(pixel_values.shape[0])
        self._index = 0

    def __len__(self) -> int:
        return self._count

    def _sample(self, index: int) -> np.ndarray:
        if self.pixel_values is not None:
            return self.pixel_values[index : index + 1]
        return np.ascontiguousarray(
            self.loader(self.image_paths[index]), dtype=np.float32
        )

    def get_next(self) -> dict | None:
        if self._index >= self._count:
            return None
        batch = {
            "pixel_values": self._sample(self._index),
            "query_embeds": self.query_embeds,
        }
        self._index += 1
        return batch

    def rewind(self) -> None:
        self._index = 0

    def get_first(self) -> dict:
        """Return the first sample without consuming the iteration.

        modelopt calls this to shape-probe the graph before calibrating, so it
        must not advance the cursor that :meth:`get_next` walks.
        """
        return {"pixel_values": self._sample(0), "query_embeds": self.query_embeds}

    def set_range(self, start: int) -> None:
        """Used by ONNX Runtime's calibrator to restart iteration."""
        self._index = start


def to_quantized_onnx(
    src: Path,
    dst: Path,
    calibration: CalibrationReader,
    mode: str = "int8",
    calibration_method: str | None = None,
    op_types_to_exclude: tuple[str, ...] = ("LayerNormalization", "Softmax"),
    calibration_eps: tuple[str, ...] = ("cuda:0", "cpu"),
    high_precision_dtype: str = "fp16",
) -> Path:
    """Calibrated int8 PTQ, writing an ONNX graph with explicit Q/DQ nodes.

    LayerNorm and Softmax are left in higher precision by default.  Quantizing
    them is where transformer PTQ usually goes wrong: both have wide dynamic
    range and sit on the residual path, so the error compounds across every
    block of the vision tower.

    Calibration runs on GPU when available; on CPU it is slow enough on the
    large checkpoint to dominate the whole build.

    ``mode`` selects int8 or fp8. fp8 (E4M3) keeps an exponent field, so the
    wide-dynamic-range activations that make int8 hard on transformer residual
    paths degrade far more gracefully - but it still needs calibration, to fix
    the per-tensor scaling factors.
    """
    from modelopt.onnx.quantization import quantize

    if mode not in ("int8", "fp8"):
        raise ValueError(f"Unsupported quantization mode {mode!r}; use int8 or fp8")
    # 'entropy' is an int8 histogram method and is not meaningful for fp8,
    # whose scales come from observed maxima.
    if calibration_method is None:
        calibration_method = "entropy" if mode == "int8" else "max"

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    quantize(
        onnx_path=str(src),
        quantize_mode=mode,
        # A reader, not a data dict: modelopt slices axis 0 of every array in a
        # dict as the batch dimension, which would turn the fixed rank-2
        # query_embeds input into 53 bogus samples of shape (512,).
        calibration_data_reader=calibration,
        calibration_method=calibration_method,
        calibration_eps=list(calibration_eps),
        op_types_to_exclude=list(op_types_to_exclude),
        # Everything the quantizer leaves alone becomes fp16 rather than fp32.
        # In a strongly-typed TensorRT network an fp32 remainder is not a
        # fallback, it is what actually runs, which would quietly turn the
        # "int8" engine into a mostly-fp32 one.
        high_precision_dtype=high_precision_dtype,
        output_path=str(dst),
    )
    if not dst.exists():
        raise RuntimeError(f"Quantization produced no output at {dst}")
    return dst


def to_int8_onnx(src: Path, dst: Path, calibration: CalibrationReader, **kwargs) -> Path:
    """Backwards-compatible int8 entry point."""
    return to_quantized_onnx(src, dst, calibration, mode="int8", **kwargs)


def to_fp8_onnx(src: Path, dst: Path, calibration: CalibrationReader, **kwargs) -> Path:
    """Calibrated fp8 (E4M3) post-training quantization."""
    return to_quantized_onnx(src, dst, calibration, mode="fp8", **kwargs)


def summarize_quantization(onnx_path: Path) -> dict:
    """Count Q/DQ nodes, so an 'int8' graph that is silently fp32 is visible."""
    import onnx

    model = onnx.load(str(onnx_path))
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {
        "path": str(onnx_path),
        "size_mb": round(Path(onnx_path).stat().st_size / (1 << 20), 2),
        "total_nodes": len(model.graph.node),
        "quantize_nodes": counts.get("QuantizeLinear", 0),
        "dequantize_nodes": counts.get("DequantizeLinear", 0),
        "trt_fp8_nodes": counts.get("TRT_FP8QuantizeLinear", 0)
        + counts.get("TRT_FP8DequantizeLinear", 0),
        "cast_nodes": counts.get("Cast", 0),
    }
