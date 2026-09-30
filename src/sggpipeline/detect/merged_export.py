"""TensorRT-exportable OWLv2 with early, previous-frame-guided token merging.

Only the early-merge schedule is exportable as a static graph, and it is the one
that won (``prev50``: lossless at half the encoder compute, no lock-in when
self-fed). With a fixed budget the token count never changes - 2,251 at 960px
and 50% - so the engine keeps fixed shapes.

The per-frame merge decision stays *outside* the engine. ``MergeIndexer`` turns
the previous frame's objectness into three index tensors (a handful of GPU ops
over 900 windows), and the engine consumes them as ordinary inputs:

``unmerged_idx``    (Nu,)     patches kept as individual tokens
``member_patches``  (Wm, 4)   the four patches of each merged window
``assign``          (N,)      token index of every patch, for unmerging

No data-dependent shapes or control flow enter the graph.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .token_merging import (
    WindowGrid,
    _head_features,
    layer_forward,
    plan_indices,
    token_sizes,
)

INPUT_NAMES = ["pixel_values", "query_embeds", "unmerged_idx", "member_patches", "assign"]
OUTPUT_NAMES = ["pred_logits", "pred_boxes", "objectness"]


class MergedDetectionGraph(nn.Module):
    """OWLv2 vision tower and heads with a fixed number of merged windows."""

    def __init__(self, model, num_merged_windows: int, proportional: bool = True,
                 with_features: bool = False):
        super().__init__()
        self.model = model
        self.with_features = with_features
        self.side = model.num_patches_height
        num_patches = self.side * self.side
        num_unmerged = num_patches - 4 * num_merged_windows
        # Token sizes depend only on the counts, never on which windows merged,
        # because plan_indices fixes the order. So they are a constant here.
        # Without proportional attention there is no bias and heads keep their
        # native width.
        sizes = token_sizes(num_unmerged, num_merged_windows).log() if proportional else None
        self.register_buffer("log_sizes", sizes)

    def forward(self, pixel_values, query_embeds, unmerged_idx, member_patches, assign):
        vision = self.model.owlv2.vision_model
        x = vision.pre_layernorm(vision.embeddings(pixel_values))
        patches = x[:, 1:]
        tokens = torch.cat(
            [x[:, :1], patches[:, unmerged_idx], patches[:, member_patches].mean(dim=2)], dim=1
        )
        for layer in vision.encoder.layers:
            tokens = layer_forward(layer, tokens, self.log_sizes)

        feats = _head_features(self.model, torch.cat([tokens[:, :1], tokens[:, assign]], dim=1))
        fmap = feats.reshape(1, self.side, self.side, -1)
        logits = self.model.class_predictor(feats, query_embeds.unsqueeze(0), None)[0]
        boxes = self.model.box_predictor(feats, fmap)
        objectness = self.model.objectness_predictor(feats)
        if self.with_features:
            # Unmerged per-patch features. Object patches are never merged under
            # the chosen schedule, so detections read intact descriptors.
            return logits, boxes, objectness, feats
        return logits, boxes, objectness


class MergeIndexer:
    """Previous-frame objectness -> the engine's three index inputs.

    Mirrors the eager ``merged_forward`` early stage exactly: max-pool patch
    scores into 2x2 windows, optionally dilate, merge the lowest-scoring fraction.
    """

    def __init__(self, side: int, fraction: float, dilate: bool = True, device="cuda"):
        self.grid = WindowGrid(side, device)
        self.num_merged = round(fraction * self.grid.num_windows)
        self.dilate = dilate

    def __call__(self, prior_objectness: torch.Tensor, protect_boxes=None,
                 image_size: tuple[int, int] | None = None):
        """Merge plan from the previous frame's objectness.

        ``protect_boxes`` (xyxy native pixels, with ``image_size``) are the
        previous frame's confident detections: windows whose centre lies inside
        one are merged last. Patch objectness alone scores the interior of big
        plain surfaces (a tabletop, the floor) as background; on the full test
        split that is where merging's accuracy loss concentrated. The budget is
        fixed by the engine's static shape, so protection reorders which
        windows merge and never changes how many.
        """
        scores = self.grid.window_max(prior_objectness.reshape(-1), self.dilate)
        if protect_boxes is not None and len(protect_boxes):
            protected = self.windows_inside(protect_boxes, image_size, scores.device)
            scores = torch.where(protected, scores + 1e4, scores)
        window_merged = torch.zeros(self.grid.num_windows, dtype=torch.bool,
                                    device=scores.device)
        window_merged[scores.argsort()[: self.num_merged]] = True
        return plan_indices(window_merged, self.grid)

    def windows_inside(self, boxes, image_size: tuple[int, int], device) -> torch.Tensor:
        """(W,) mask of windows whose centre lies inside any box.

        Windows tile the bottom/right-padded square the preprocessor builds, so a
        window spans ``max(width, height) / windows-per-side`` native pixels.
        """
        half = self.grid.half
        cell = max(image_size) / half
        centres = (torch.arange(half, device=device, dtype=torch.float32) + 0.5) * cell
        cy, cx = torch.meshgrid(centres, centres, indexing="ij")
        cx, cy = cx.reshape(-1, 1), cy.reshape(-1, 1)
        b = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)
        inside = (cx >= b[:, 0]) & (cx <= b[:, 2]) & (cy >= b[:, 1]) & (cy <= b[:, 3])
        return inside.any(dim=1)


def export_merged_onnx(model, num_prompts: int, image_size: int, fraction: float,
                       out_path: Path, opset: int = 17, device: str = "cuda",
                       proportional: bool = True, with_features: bool = False) -> Path:
    """Trace the merged graph with static shapes, fp32, for later precision passes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    side = model.num_patches_height
    indexer = MergeIndexer(side, fraction, device=device)
    graph = MergedDetectionGraph(model, indexer.num_merged, proportional,
                                 with_features).eval().to(device)

    # Any valid plan traces the same graph; only the index *values* vary per frame.
    unmerged, members, assign = indexer(torch.rand(side * side, device=device))
    embed_dim = model.config.text_config.hidden_size
    query = torch.randn(num_prompts, embed_dim, device=device)
    query = query / query.norm(dim=-1, keepdim=True)
    pixels = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        torch.onnx.export(
            graph, (pixels, query, unmerged, members, assign), str(out_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES + (["patch_features"] if with_features else []),
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    return out_path
