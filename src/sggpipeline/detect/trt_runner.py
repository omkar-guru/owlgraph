"""Run a serialized TensorRT engine, using torch tensors as the device buffers.

Binding torch allocations to TensorRT avoids a second memory allocator and lets
the benchmark time the engine on the same CUDA stream torch already uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
}


class TRTRunner:
    """Thin execution wrapper around one engine with static shapes."""

    def __init__(self, engine_path: Path, device: str = "cuda"):
        self.engine_path = Path(engine_path)
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not deserialize engine: {self.engine_path}")
        self.engine = engine
        self.context = engine.create_execution_context()

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # Output buffers are allocated once and reused across frames, so steady
        # state timing does not include an allocation per call.
        self.outputs = {
            name: torch.empty(
                tuple(self.context.get_tensor_shape(name)),
                dtype=self._torch_dtype(name),
                device=self.device,
            )
            for name in self.output_names
        }

    def _torch_dtype(self, name: str) -> torch.dtype:
        dtype = self.engine.get_tensor_dtype(name)
        if dtype not in _TRT_TO_TORCH:
            raise RuntimeError(f"Unsupported TensorRT dtype {dtype} for tensor {name}")
        return _TRT_TO_TORCH[dtype]

    def infer(self, feeds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Execute one forward pass; returns the engine's own output buffers."""
        missing = set(self.input_names) - set(feeds)
        if missing:
            raise KeyError(f"Missing engine inputs: {sorted(missing)}")

        stream = torch.cuda.current_stream(self.device)
        for name in self.input_names:
            tensor = feeds[name].to(
                device=self.device, dtype=self._torch_dtype(name)
            ).contiguous()
            feeds[name] = tensor  # keep alive until execution completes
            self.context.set_tensor_address(name, tensor.data_ptr())
        for name, buf in self.outputs.items():
            self.context.set_tensor_address(name, buf.data_ptr())

        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.outputs

    def infer_numpy(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convenience path for evaluation code that works in numpy."""
        torch_feeds = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in feeds.items()}
        outputs = self.infer(torch_feeds)
        torch.cuda.current_stream(self.device).synchronize()
        return {k: v.detach().float().cpu().numpy() for k, v in outputs.items()}


class TorchRunner:
    """Eager-PyTorch fallback with the same interface, as a correctness reference.

    The TensorRT numbers mean little without something to compare them against;
    this is the unoptimized baseline both accuracy and latency are measured from.
    """

    def __init__(self, graph, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.graph = graph.eval()
        self.device = torch.device(device)
        self.dtype = dtype
        self.input_names = ["pixel_values", "query_embeds"]
        self.output_names = ["pred_logits", "pred_boxes", "objectness"]

    @torch.no_grad()
    def infer(self, feeds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pixel_values = feeds["pixel_values"].to(self.device, self.dtype)
        query_embeds = feeds["query_embeds"].to(self.device, self.dtype)
        outputs = self.graph(pixel_values, query_embeds)
        return dict(zip(self.output_names, outputs, strict=True))

    def infer_numpy(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch_feeds = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in feeds.items()}
        outputs = self.infer(torch_feeds)
        torch.cuda.synchronize()
        return {k: v.detach().float().cpu().numpy() for k, v in outputs.items()}
