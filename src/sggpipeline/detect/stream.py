"""The Stage 1 detector as a video stream: the interface later stages consume.

``StreamingDetector`` owns the logic that was duplicated across benchmark
scripts:

* the first frame after ``reset()`` runs the unmerged engine, seeding the prior;
* every later frame runs the merged engine, with its merge plan built from the
  previous frame's objectness (self-fed: the streaming test found no lock-in
  within 5 s, so no periodic refresh);
* raw outputs become per-class, duplicate-suppressed detections;
* each detection carries the feature vector of the patch that produced it.

The last point is the Stage 1 -> Stage 2 bridge. OWLv2 predicts one box per
patch, so a detection's own descriptor is that patch's row in the feature map
the heads read: no pooling and no second encoder. Engines must be built with the
``patch_features`` output (``build_engines.py --features``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .fast_preprocess import GpuOwlv2Preprocessor
from .merged_export import MergeIndexer
from .owlv2 import Detections, postprocess_device, unpack_detections
from .trt_runner import TRTRunner

PLAN_INPUTS = ("unmerged_idx", "member_patches", "assign")


@dataclass
class FrameResult:
    detections: Detections  # native-pixel boxes, with patch_index
    features: np.ndarray  # (K, D) float32, row i describes detection i
    merged: bool  # False for the seeding frame


class StreamingDetector:
    """Merged engine on a stream, seeded by the unmerged engine.

    With ``merged_engine=None`` every frame runs the unmerged engine, which suits
    sparse sampling (e.g. 1 frame/s) where a previous-frame prior would be stale.
    """

    def __init__(self, seed_engine: Path, merged_engine: Path | None, queries,
                 num_classes: int, image_size: int = 960, fraction: float = 0.5,
                 dilate: bool = True, score_threshold: float = 0.1,
                 nms_iou: float | None = 0.7, max_detections: int = 100,
                 min_box_side: float = 1.0, protect_score: float | None = 0.3,
                 device: str = "cuda"):
        self.seed = TRTRunner(seed_engine, device=device)
        self.merged = TRTRunner(merged_engine, device=device) if merged_engine else None
        for runner in filter(None, (self.seed, self.merged)):
            if "patch_features" not in runner.output_names:
                raise ValueError(f"{runner.engine_path.name} has no patch_features output; "
                                 "rebuild with build_engines.py --features")
        self.pre = GpuOwlv2Preprocessor(image_size, device=device)
        self.indexer = MergeIndexer(image_size // 16, fraction, dilate, device=device)
        self.query = torch.from_numpy(queries.embeds).to(device)
        self.owner = torch.as_tensor(queries.owner, device=device)
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.min_box_side = min_box_side
        # Windows inside the previous frame's detections at or above this score
        # are merged last. On the full AG test split this recovered ~58% of
        # merging's mAP loss (0.1043 -> 0.1064 vs 0.1079 unmerged), mostly on
        # large plain objects. None disables it.
        self.protect_score = protect_score
        self.reset()

    def reset(self) -> None:
        """Call between videos: the next frame is re-seeded with the unmerged engine."""
        self._prior = None
        self._protect = None

    def __call__(self, frame) -> FrameResult:
        """``frame``: PIL image, HWC uint8 array, or CHW uint8 tensor."""
        chw = self._as_chw(frame)
        height, width = chw.shape[1:]
        px = self.pre.preprocess_tensor(chw.unsqueeze(0))

        use_merged = self.merged is not None and self._prior is not None
        feeds = {"pixel_values": px, "query_embeds": self.query}
        if use_merged:
            feeds.update(zip(PLAN_INPUTS, self.indexer(self._prior, self._protect,
                                                       (width, height))))
        out = (self.merged if use_merged else self.seed).infer(feeds)
        if self.merged is not None:
            self._prior = out["objectness"][0].float().clone()

        # On the GPU; detections and their features then cross to the host in one
        # copy, since every separate copy is another wait on the device.
        packed = postprocess_device(out["pred_logits"], out["pred_boxes"], out["objectness"],
                                    self.owner, self.num_classes, (width, height),
                                    self.score_threshold, self.max_detections, self.nms_iou)
        # Clipping at the image border can collapse a box to zero width; later
        # stages require positive-area boxes, so such slivers are dropped here.
        sides = packed[:, 2:4] - packed[:, 0:2]
        packed = packed[(sides >= self.min_box_side).all(dim=1)]
        features = out["patch_features"][0].index_select(0, packed[:, 7].long()).float()
        host = torch.cat([packed, features], dim=1).cpu().numpy()
        det = unpack_detections(host[:, :8])

        if self.protect_score is not None:
            self._protect = det.boxes[det.scores >= self.protect_score]
        return FrameResult(det, np.ascontiguousarray(host[:, 8:]), use_merged)

    @staticmethod
    def _as_chw(frame) -> torch.Tensor:
        if isinstance(frame, torch.Tensor):
            return frame
        array = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
