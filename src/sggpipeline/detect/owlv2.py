"""OWLv2 loading, a TensorRT-exportable detection graph, and pre/post-processing.

OWLv2 is a two-tower model.  The text tower only ever sees the fixed AG prompt
set, so it runs **once** at setup and its output becomes an ordinary input
tensor to the exported graph.  Only the vision tower and the prediction heads go
into the TensorRT engine, which is also what the per-frame latency numbers time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

BASE_MODEL = "google/owlv2-base-patch16-ensemble"
LARGE_MODEL = "google/owlv2-large-patch14-ensemble"

MODEL_ALIASES = {"base": BASE_MODEL, "large": LARGE_MODEL}


def resolve_model_id(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def load_owlv2(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.float32):
    """Load an OWLv2 checkpoint and its processor in eval mode."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_id = resolve_model_id(model_id)
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id, dtype=dtype)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


@torch.no_grad()
def encode_text_queries(
    model, processor, prompts: list[str], device: str = "cuda"
) -> torch.Tensor:
    """Embed the prompt set once with the text tower.

    Returns ``(num_prompts, embed_dim)``.  The class head re-normalizes its query
    input, so normalizing here is idempotent and just keeps the tensor tidy.
    """
    tokens = processor(text=[prompts], return_tensors="pt", padding=True, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # get_text_features returns the full text-model output; the projected
    # embedding is written back onto pooler_output.
    text_outputs = model.owlv2.get_text_features(
        input_ids=input_ids, attention_mask=attention_mask
    )
    text_embeds = getattr(text_outputs, "pooler_output", text_outputs)
    text_embeds = text_embeds / (text_embeds.norm(p=2, dim=-1, keepdim=True) + 1e-6)
    return text_embeds.detach().float()


class Owlv2DetectionGraph(nn.Module):
    """Vision tower + heads, with the text embeddings supplied as an input.

    Kept deliberately free of Python control flow and dynamic shapes so it traces
    cleanly to ONNX and gives TensorRT a single static profile to optimize.
    """

    def __init__(self, model, interpolate_pos_encoding: bool = False,
                 with_features: bool = False):
        super().__init__()
        self.model = model
        # Required whenever the input is not the checkpoint's native resolution:
        # the position embeddings are learned for a fixed patch grid and must be
        # resampled to the new one, or every patch is located wrongly.
        self.interpolate_pos_encoding = interpolate_pos_encoding
        # Also return the per-patch features the heads read. Downstream stages
        # (identity, pair head) branch from these, so exposing them costs one
        # extra output instead of a second encoder.
        self.with_features = with_features

    def forward(self, pixel_values: torch.Tensor, query_embeds: torch.Tensor):
        # (B, H, W, D) grid of per-patch visual features.
        feature_map, _ = self.model.image_embedder(
            pixel_values=pixel_values,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
        )
        batch, height, width, dim = feature_map.shape
        image_feats = feature_map.reshape(batch, height * width, dim)

        # Class head compares patch embeddings against the text queries.
        query = query_embeds.unsqueeze(0).expand(batch, -1, -1)
        pred_logits, _ = self.model.class_predictor(image_feats, query, None)

        # Boxes are cxcywh, normalized to the padded square the processor built.
        pred_boxes = self.model.box_predictor(
            image_feats, feature_map, interpolate_pos_encoding=self.interpolate_pos_encoding
        )

        # Class-agnostic objectness, useful for top-k pruning later in the pipeline.
        objectness = self.model.objectness_predictor(image_feats)

        if self.with_features:
            return pred_logits, pred_boxes, objectness, image_feats
        return pred_logits, pred_boxes, objectness


@torch.no_grad()
def retarget_resolution(model, image_size: int) -> int:
    """Resample the learned position grid so the graph is static at a new size.

    OWLv2 learns one position embedding per patch of a fixed grid, so running at
    a different input size needs them resampled.  Doing that *at export time*
    rather than with ``interpolate_pos_encoding`` at runtime matters for
    TensorRT: the dynamic path emits a ``Range`` op that TensorRT requires in
    fp32, which then collides with the surrounding fp16 tensors in a
    strongly-typed network.  Baking the grid in removes the op entirely and
    saves the interpolation work on every frame.

    Returns the new patch-grid side length.
    """
    from torch import nn

    vision = model.owlv2.vision_model
    embeddings = vision.embeddings
    config = model.config.vision_config

    patch = config.patch_size
    if image_size % patch:
        raise ValueError(f"image_size {image_size} is not a multiple of patch size {patch}")

    old_side = config.image_size // patch
    new_side = image_size // patch
    if new_side == old_side:
        return new_side

    weight = embeddings.position_embedding.weight  # (1 + old_side^2, dim)
    class_pos, patch_pos = weight[:1], weight[1:]
    dim = weight.shape[1]

    # Bicubic on the 2-D grid, matching how the dynamic path resamples.
    grid = patch_pos.reshape(1, old_side, old_side, dim).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(
        grid.float(), size=(new_side, new_side), mode="bicubic", align_corners=False
    )
    grid = grid.permute(0, 2, 3, 1).reshape(new_side * new_side, dim).to(weight.dtype)

    new_weight = torch.cat([class_pos, grid], dim=0)
    num_positions = new_weight.shape[0]

    embeddings.position_embedding = nn.Embedding(num_positions, dim).to(
        weight.device, weight.dtype
    )
    embeddings.position_embedding.weight.copy_(new_weight)
    embeddings.num_patches = new_side * new_side
    embeddings.num_positions = num_positions
    embeddings.position_ids = torch.arange(num_positions, device=weight.device).expand((1, -1))

    # Several grid-dependent values are cached on the model at construction
    # time, not read from the config per call. All of them have to move together
    # or the graph silently reshapes to the old grid.
    config.image_size = image_size
    model.config.vision_config.image_size = image_size
    model.num_patches_height = new_side
    model.num_patches_width = new_side
    model.box_bias = model.compute_box_bias(new_side, new_side).to(weight.device)
    return new_side


@dataclass(slots=True)
class Detections:
    """Post-processed detections for one frame, in native image pixels."""

    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) class indices
    objectness: np.ndarray  # (N,)
    # Patch that produced each detection. OWLv2 predicts one box per patch, so
    # this row of the patch-feature map is that detection's own descriptor.
    patch_index: np.ndarray | None = None  # (N,)


def preprocess_sizes(width: int, height: int) -> float:
    """Side length of the square OWLv2's processor pads the image to.

    The processor pads to a square at the bottom/right *before* resizing, so
    normalized box coordinates map back by a single scale with no offset.
    Getting this wrong silently costs mAP rather than raising, so it lives in
    one named function.
    """
    return float(max(width, height))


def postprocess(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    objectness: np.ndarray,
    prompt_owner: np.ndarray,
    num_classes: int,
    image_size: tuple[int, int],
    score_threshold: float = 0.05,
    max_detections: int = 100,
    nms_iou: float | None = None,
) -> Detections:
    """Turn raw head outputs into per-class detections in original pixels.

    ``prompt_owner`` maps each text query back to its class, so a class written
    with several surface forms ("a cup" / "a glass" / "a bottle") is scored by
    its best-matching prompt rather than an arbitrary one.

    ``nms_iou`` enables per-class duplicate suppression. Off by default so Stage 1
    scores stay comparable; a tracker needs it, because below ~0.3 score about a
    tenth of detections are same-class near-duplicates of another detection.
    """
    logits = np.asarray(pred_logits, dtype=np.float32).reshape(-1, pred_logits.shape[-1])
    boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    obj = np.asarray(objectness, dtype=np.float32).reshape(-1)
    scores_per_prompt = _sigmoid(logits)

    # Max-pool prompt scores into class scores.
    num_patches = scores_per_prompt.shape[0]
    class_scores = np.full((num_patches, num_classes), -np.inf, dtype=np.float32)
    for prompt_idx, class_idx in enumerate(prompt_owner):
        np.maximum(
            class_scores[:, class_idx],
            scores_per_prompt[:, prompt_idx],
            out=class_scores[:, class_idx],
        )

    best_class = class_scores.argmax(axis=1)
    best_score = class_scores[np.arange(num_patches), best_class]

    keep = best_score >= score_threshold
    if not keep.any():
        empty_f = np.zeros((0, 4), dtype=np.float32)
        empty = np.zeros((0,), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int64)
        return Detections(empty_f, empty, empty_i, empty, empty_i)

    patch = np.flatnonzero(keep)
    boxes, best_score = boxes[keep], best_score[keep]
    best_class, obj = best_class[keep], _sigmoid(obj[keep])

    order = np.argsort(-best_score)
    if nms_iou is None:
        order = order[:max_detections]
    boxes, best_score = boxes[order], best_score[order]
    best_class, obj, patch = best_class[order], obj[order], patch[order]

    # cxcywh (normalized to the padded square) -> xyxy in native pixels.
    scale = preprocess_sizes(*image_size)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack(
        [(cx - w / 2) * scale, (cy - h / 2) * scale,
         (cx + w / 2) * scale, (cy + h / 2) * scale],
        axis=1,
    )
    width, height = image_size
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, width)
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, height)

    if nms_iou is not None:
        from torchvision.ops import batched_nms

        # batched_nms returns indices in descending score order.
        kept = batched_nms(torch.from_numpy(xyxy), torch.from_numpy(best_score),
                           torch.from_numpy(best_class), nms_iou).numpy()[:max_detections]
        xyxy, best_score, best_class = xyxy[kept], best_score[kept], best_class[kept]
        obj, patch = obj[kept], patch[kept]

    return Detections(
        boxes=xyxy.astype(np.float32),
        scores=best_score.astype(np.float32),
        labels=best_class.astype(np.int64),
        objectness=obj.astype(np.float32),
        patch_index=patch.astype(np.int64),
    )


@torch.no_grad()
def postprocess_device(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    objectness: torch.Tensor,
    prompt_owner: torch.Tensor,
    num_classes: int,
    image_size: tuple[int, int],
    score_threshold: float = 0.05,
    max_detections: int = 100,
    nms_iou: float | None = None,
) -> torch.Tensor:
    """``postprocess`` kept on the tensors' device, packed as one (K, 8) tensor.

    Columns: x1, y1, x2, y2, score, label, objectness, patch index. Packing lets
    a caller bring everything to the host in a single copy - each separate
    ``.cpu()`` is a device synchronisation, and on the streaming path those
    waits were most of what postprocessing still cost.
    """
    logits = pred_logits.reshape(-1, pred_logits.shape[-1]).float()
    boxes = pred_boxes.reshape(-1, 4).float()
    obj = objectness.reshape(-1).float()
    num_patches = logits.shape[0]
    owner = prompt_owner.to(logits.device).expand(num_patches, -1)
    class_scores = torch.full((num_patches, num_classes), float("-inf"), device=logits.device)
    class_scores = class_scores.scatter_reduce(1, owner, torch.sigmoid(logits), reduce="amax")
    best_score, best_class = class_scores.max(dim=1)

    patch = torch.nonzero(best_score >= score_threshold).squeeze(1)
    patch = patch[torch.argsort(best_score[patch], descending=True, stable=True)]
    if nms_iou is None:
        patch = patch[:max_detections]
    score, label = best_score[patch], best_class[patch]

    scale = preprocess_sizes(*image_size)
    cx, cy, w, h = boxes[patch].unbind(-1)
    xyxy = torch.stack([(cx - w / 2) * scale, (cy - h / 2) * scale,
                        (cx + w / 2) * scale, (cy + h / 2) * scale], dim=1)
    width, height = image_size
    xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, width)
    xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, height)

    if nms_iou is not None:
        from torchvision.ops import batched_nms

        kept = batched_nms(xyxy, score, label, nms_iou)[:max_detections]
        xyxy, score, label, patch = xyxy[kept], score[kept], label[kept], patch[kept]

    # Labels (< 64) and patch indices (< 2^24) are exact in float32.
    return torch.cat([xyxy, score[:, None], label[:, None].float(),
                      torch.sigmoid(obj[patch])[:, None], patch[:, None].float()], dim=1)


def unpack_detections(packed: np.ndarray) -> Detections:
    """Host-side (K, 8) array from ``postprocess_device`` -> ``Detections``."""
    packed = np.asarray(packed, dtype=np.float32).reshape(-1, 8)
    return Detections(
        boxes=np.ascontiguousarray(packed[:, :4]),
        scores=np.ascontiguousarray(packed[:, 4]),
        labels=packed[:, 5].astype(np.int64),
        objectness=np.ascontiguousarray(packed[:, 6]),
        patch_index=packed[:, 7].astype(np.int64),
    )


def postprocess_torch(pred_logits, pred_boxes, objectness, prompt_owner, num_classes,
                      image_size, score_threshold: float = 0.05, max_detections: int = 100,
                      nms_iou: float | None = None) -> Detections:
    """``postprocess`` on the tensors' own device, returning numpy detections.

    Same semantics as the numpy reference (and tested against it), but on the GPU
    only the surviving detections cross to the host, in one copy, instead of the
    full 3600 x prompts score grid.
    """
    packed = postprocess_device(pred_logits, pred_boxes, objectness, prompt_owner, num_classes,
                                image_size, score_threshold, max_detections, nms_iou)
    return unpack_detections(packed.cpu().numpy())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
