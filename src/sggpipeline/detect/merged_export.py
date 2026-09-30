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

    def __init__(self, model, num_merged_windows: int, proportional: bool = True):
        super().__init__()
        self.model = model
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

    def __call__(self, prior_objectness: torch.Tensor):
        scores = self.grid.window_max(prior_objectness.reshape(-1), self.dilate)
        window_merged = torch.zeros(self.grid.num_windows, dtype=torch.bool,
                                    device=scores.device)
        window_merged[scores.argsort()[: self.num_merged]] = True
        return plan_indices(window_merged, self.grid)


def export_merged_onnx(model, num_prompts: int, image_size: int, fraction: float,
                       out_path: Path, opset: int = 17, device: str = "cuda",
                       proportional: bool = True) -> Path:
    """Trace the merged graph with static shapes, fp32, for later precision passes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    side = model.num_patches_height
    indexer = MergeIndexer(side, fraction, device=device)
    graph = MergedDetectionGraph(model, indexer.num_merged, proportional).eval().to(device)

    # Any valid plan traces the same graph; only the index *values* vary per frame.
    unmerged, members, assign = indexer(torch.rand(side * side, device=device))
    embed_dim = model.config.text_config.hidden_size
    query = torch.randn(num_prompts, embed_dim, device=device)
    query = query / query.norm(dim=-1, keepdim=True)
    pixels = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        torch.onnx.export(
            graph, (pixels, query, unmerged, members, assign), str(out_path),
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    return out_path
