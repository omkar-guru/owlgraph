"""YOLO26 through the same TensorRT path as OWLv2, for a like-for-like comparison.

Comparing an ultralytics ``model.predict()`` call against a TensorRT engine
measures two different things: one includes Python-side letterboxing, NMS and
result-object construction, the other is a bare engine execution. This module
puts YOLO on the same footing - GPU preprocessing, a strongly-typed fp16 TRT
engine, GPU postprocessing - so the two can be timed at matching scope.

YOLO26 exports with ``end2end=False``: the head emits ``(1, 84, 8400)`` raw
predictions (4 box + 80 class scores over 8400 anchors) and **still needs NMS**.
That cost is real and is measured as part of postprocessing rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# Ultralytics letterboxes with grey 114 on a 0-255 scale.
LETTERBOX_FILL = 114.0 / 255.0


@dataclass(slots=True)
class LetterboxGeometry:
    """Mapping from letterboxed 640x640 coordinates back to the source image."""

    scale: float
    pad_x: float
    pad_y: float


class YoloPreprocessor:
    """Letterbox to a square on the GPU, matching ultralytics' geometry.

    YOLO letterboxes - aspect-preserving resize, then **centred** grey padding -
    which is a different convention from OWLv2's pad-bottom-right-then-resize.
    Using the wrong one silently shifts every box, so the geometry is returned
    explicitly rather than reconstructed later.
    """

    def __init__(self, image_size: int = 640, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.image_size = int(image_size)
        self.device = torch.device(device)
        self.dtype = dtype

    def __call__(self, images: list) -> tuple[torch.Tensor, list[LetterboxGeometry]]:
        tensors, geometries = [], []
        for image in images:
            array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
            chw = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
            batch, geometry = self._letterbox(chw.unsqueeze(0).to(self.device))
            tensors.append(batch)
            geometries.append(geometry)
        return torch.cat(tensors, dim=0).to(self.dtype).contiguous(), geometries

    def _letterbox(self, x: torch.Tensor) -> tuple[torch.Tensor, LetterboxGeometry]:
        x = x.float() * (1.0 / 255.0)
        _, _, height, width = x.shape
        size = self.image_size
        scale = min(size / height, size / width)
        new_h, new_w = round(height * scale), round(width * scale)

        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                          align_corners=False, antialias=True)
        pad_y, pad_x = (size - new_h) / 2, (size - new_w) / 2
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        bottom, right = size - new_h - top, size - new_w - left
        x = F.pad(x, (left, right, top, bottom), value=LETTERBOX_FILL)
        return x, LetterboxGeometry(scale=scale, pad_x=left, pad_y=top)


@dataclass(slots=True)
class YoloDetections:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


def decode(
    raw: torch.Tensor,
    geometry: LetterboxGeometry,
    image_size: tuple[int, int],
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.7,
    max_detections: int = 100,
) -> YoloDetections:
    """Decode ``(1, 84, 8400)`` head output into boxes in source-image pixels.

    NMS runs on the GPU via torchvision, class-wise, which is what ultralytics
    does; doing it globally would merge overlapping objects of different classes.
    """
    from torchvision.ops import batched_nms

    predictions = raw.float().squeeze(0).transpose(0, 1)  # (8400, 84)
    boxes_cxcywh, class_scores = predictions[:, :4], predictions[:, 4:]

    scores, labels = class_scores.max(dim=1)
    keep = scores >= conf_threshold
    if not keep.any():
        empty = np.zeros((0,), dtype=np.float32)
        return YoloDetections(np.zeros((0, 4), dtype=np.float32), empty,
                              np.zeros((0,), dtype=np.int64))

    boxes_cxcywh, scores, labels = boxes_cxcywh[keep], scores[keep], labels[keep]
    cx, cy, w, h = boxes_cxcywh.unbind(dim=1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    selected = batched_nms(xyxy, scores, labels, iou_threshold)[:max_detections]
    xyxy, scores, labels = xyxy[selected], scores[selected], labels[selected]

    # Undo letterbox: remove the centred padding, then the uniform scale.
    xyxy[:, [0, 2]] -= geometry.pad_x
    xyxy[:, [1, 3]] -= geometry.pad_y
    xyxy /= geometry.scale

    width, height = image_size
    xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, width)
    xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, height)

    return YoloDetections(
        boxes=xyxy.cpu().numpy().astype(np.float32),
        scores=scores.cpu().numpy().astype(np.float32),
        labels=labels.cpu().numpy().astype(np.int64),
    )


def export_and_build(weights: str, image_size: int, ws, rebuild: bool = False):
    """ultralytics -> ONNX -> fp16 -> strongly-typed TensorRT engine.

    Ultralytics' own ``format="engine"`` export targets the pre-11 TensorRT API,
    so the graph is routed through this project's builder instead, which handles
    the strongly-typed requirements.
    """
    from pathlib import Path

    from ultralytics import YOLO

    from .quantize import to_fp16_onnx
    from .trt_build import build_engine

    stem = Path(weights).stem
    engine_path = ws.engine(f"{stem}_fp16.plan")
    if engine_path.exists() and not rebuild:
        return engine_path

    fp32 = ws.onnx(f"{stem}_fp32.onnx")
    if not fp32.exists() or rebuild:
        produced = YOLO(weights).export(
            format="onnx", imgsz=image_size, opset=17, dynamic=False, verbose=False
        )
        Path(produced).replace(fp32)

    fp16 = to_fp16_onnx(fp32, ws.onnx(f"{stem}_fp16.onnx"))
    return build_engine(fp16, engine_path, workspace_gb=6.0,
                        timing_cache_path=ws.cache("timing.cache"))
