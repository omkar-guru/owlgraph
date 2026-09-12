"""Build TensorRT engines from the exported ONNX graph.

TensorRT 11 builds **strongly-typed** networks only.  The legacy controls -
``BuilderFlag.FP16``, ``BuilderFlag.INT8`` and ``IInt8EntropyCalibrator2`` - no
longer exist, so precision is not a builder flag here.  It is a property of the
ONNX graph handed to the builder:

fp16
    The ONNX graph is cast to fp16, and TensorRT honours those types.
int8
    The ONNX graph carries explicit QuantizeLinear/DequantizeLinear pairs placed
    by a calibrated PTQ pass (see :mod:`sggpipeline.detect.quantize`).

Both therefore go through the same builder; only the input file differs.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt


def _logger(verbose: bool) -> trt.Logger:
    return trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float = 6.0,
    optimization_level: int = 3,
    timing_cache_path: Path | None = None,
    verbose: bool = False,
) -> Path:
    """Compile one ONNX file into a serialized TensorRT engine.

    The engine is written next to its inputs and is specific to this GPU, this
    driver and this TensorRT version; it is a build artifact, never a checkpoint.
    """
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = _logger(verbose)
    builder = trt.Builder(logger)

    # Strongly typed: the parsed graph's dtypes are authoritative. Without this
    # flag TensorRT is free to run everything in fp32 and quietly discard both
    # the fp16 casts and the Q/DQ placement.
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as fh:
        if not parser.parse(fh.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    config.builder_optimization_level = optimization_level
    if verbose:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    cache = _load_timing_cache(config, timing_cache_path)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            f"TensorRT failed to build an engine for {onnx_path}. "
            "Re-run with verbose=True for the builder log."
        )

    engine_path.write_bytes(serialized)
    _save_timing_cache(config, cache, timing_cache_path)
    return engine_path


def _load_timing_cache(config, path: Path | None):
    """Reuse kernel-timing measurements across builds to cut build time."""
    if path is None:
        return None
    path = Path(path)
    data = path.read_bytes() if path.exists() else b""
    cache = config.create_timing_cache(data)
    config.set_timing_cache(cache, ignore_mismatch=False)
    return cache


def _save_timing_cache(config, cache, path: Path | None) -> None:
    if path is None or cache is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(memoryview(cache.serialize()))


def engine_summary(engine_path: Path) -> dict:
    """Report the I/O signature and precisions TensorRT actually chose."""
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize engine: {engine_path}")

    tensors = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        tensors.append(
            {
                "name": name,
                "mode": engine.get_tensor_mode(name).name,
                "dtype": engine.get_tensor_dtype(name).name,
                "shape": tuple(engine.get_tensor_shape(name)),
            }
        )
    return {
        "engine_path": str(engine_path),
        "size_mb": round(Path(engine_path).stat().st_size / (1 << 20), 2),
        "num_layers": engine.num_layers,
        "device_memory_mb": round(engine.device_memory_size_v2 / (1 << 20), 2),
        "tensors": tensors,
    }
