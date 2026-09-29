"""Selective 2x2 token merging inside the OWLv2 vision tower (eager PyTorch).

Research implementation for measuring whether merging background tokens beats
simply lowering the input resolution. It is not the deployable path: the
TensorRT version comes only if this wins.

Mechanics:

* Patches are grouped into non-overlapping 2x2 windows. A merged window becomes
  one token, the mean of its four patch tokens.
* Merging can happen before block 1 (guided by an external score, e.g. the
  previous video frame's final objectness) and again before a later block
  (guided by the model's own objectness head applied to that block's features,
  which the early-objectness probe found reliable from block 8 on).
* Attention is **proportional**: a merged token is weighted as the four patches
  it stands for, by adding log(size) to its attention logits, as in ToMe.
* Before the detection heads every merged token is copied back to its four
  patches, because OWLv2 predicts one box per patch on a fixed grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MergePlan:
    """When and how much to merge. Fractions are of all 2x2 windows."""

    early_fraction: float = 0.0
    early_scores: torch.Tensor | None = None  # per-patch, higher = keep
    early_dilate: bool = True
    late_block: int | None = None  # merge before this block index (= after that many blocks)
    late_total_fraction: float = 0.0  # cumulative fraction merged after the late stage


class WindowGrid:
    """Index tables relating patches (raster order) to their 2x2 windows."""

    def __init__(self, side: int, device):
        if side % 2:
            raise ValueError(f"patch grid side {side} is not divisible by 2")
        self.side, self.half = side, side // 2
        self.num_patches, self.num_windows = side * side, (side // 2) ** 2

        wr, wc = torch.meshgrid(torch.arange(self.half, device=device),
                                torch.arange(self.half, device=device), indexing="ij")
        wr, wc = wr.flatten(), wc.flatten()
        offsets = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device)
        rows = wr[:, None] * 2 + offsets[:, 0]
        cols = wc[:, None] * 2 + offsets[:, 1]
        self.patches_of_window = rows * side + cols  # (W, 4)

        p = torch.arange(self.num_patches, device=device)
        self.window_of_patch = (p // side // 2) * self.half + (p % side) // 2

    def window_max(self, patch_scores: torch.Tensor, dilate: bool = False) -> torch.Tensor:
        """Score each window by its best patch; optionally grow by one window."""
        grid = patch_scores.float().reshape(1, 1, self.side, self.side)
        windows = F.max_pool2d(grid, kernel_size=2, stride=2)
        if dilate:
            windows = F.max_pool2d(windows, kernel_size=3, stride=1, padding=1)
        return windows.flatten()


def plan_indices(window_merged, grid: WindowGrid):
    """Index tensors describing the token layout for a set of merged windows.

    Token order is fixed: CLS, then unmerged patches in raster order, then one
    token per merged window in window order. Both the eager path and the exported
    engine use this layout, so it is defined in exactly one place.

    Returns ``unmerged`` (patch indices kept individually), ``member_patches``
    (the four patch indices of each merged window) and ``assign`` (token index
    for every patch, used to unmerge).
    """
    device = window_merged.device
    unmerged = (~window_merged[grid.window_of_patch]).nonzero().squeeze(1)
    member_patches = grid.patches_of_window[window_merged.nonzero().squeeze(1)]  # (Wm, 4)
    nu, nm = unmerged.numel(), member_patches.shape[0]
    assign = torch.empty(grid.num_patches, dtype=torch.long, device=device)
    assign[unmerged] = 1 + torch.arange(nu, device=device)
    assign[member_patches] = (1 + nu + torch.arange(nm, device=device))[:, None].expand(-1, 4)
    return unmerged, member_patches, assign


def token_sizes(num_unmerged: int, num_merged: int, device=None) -> torch.Tensor:
    """Patches represented by each token, in ``plan_indices`` order."""
    return torch.cat([torch.ones(1 + num_unmerged, device=device),
                      torch.full((num_merged,), 4.0, device=device)])


def compose(cls_token, patch_feats, window_merged, grid: WindowGrid):
    """Build the token sequence for a given set of merged windows.

    ``patch_feats`` is one feature per patch. Windows merged at an earlier stage
    hold four identical copies, so their mean is the existing merged token.
    Returns the tokens, each token's size in patches, and a patch -> token index.
    """
    unmerged, member_patches, assign = plan_indices(window_merged, grid)
    merged_tokens = patch_feats[:, member_patches].mean(dim=2)
    tokens = torch.cat([cls_token, patch_feats[:, unmerged], merged_tokens], dim=1)
    sizes = token_sizes(unmerged.numel(), member_patches.shape[0], patch_feats.device)
    return tokens, sizes, assign


def proportional_attention(q, k, v, log_sizes, scale):
    """Scaled dot-product attention with log(size) added to each key's logit.

    An additive attention mask would add the bias directly, but it pushes
    PyTorch off its fused attention kernels and would slow the merged model for
    reasons unrelated to merging. Instead the bias is folded into one extra head
    dimension: q gets a constant 1, k gets log(size)/scale, so
    ``(q.k + log(size)/scale) * scale = q.k * scale + log(size)``. The extra
    dimension is padded to a multiple of 8 so fused kernels still apply.
    """
    if log_sizes is None:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    b, h, t, d = q.shape
    pad = (-(d + 1)) % 8
    ones = torch.ones(b, h, t, 1, dtype=q.dtype, device=q.device)
    bias = (log_sizes / scale).to(q.dtype).view(1, 1, t, 1).expand(b, h, t, 1)
    zq = torch.zeros(b, h, t, pad, dtype=q.dtype, device=q.device)
    q_aug = torch.cat([q, ones, zq], dim=-1)
    k_aug = torch.cat([k, bias, zq], dim=-1)
    v_aug = torch.cat([v, torch.zeros(b, h, t, 1 + pad, dtype=v.dtype, device=v.device)], dim=-1)
    out = F.scaled_dot_product_attention(q_aug, k_aug, v_aug, scale=scale)
    return out[..., :d]


def layer_forward(layer, x, log_sizes):
    """One OWLv2 encoder layer (pre-norm attention, then MLP), size-aware."""
    attn = layer.self_attn
    b, t, _ = x.shape
    h = layer.layer_norm1(x)
    shape = (b, t, attn.num_heads, attn.head_dim)
    q = attn.q_proj(h).view(shape).transpose(1, 2)
    k = attn.k_proj(h).view(shape).transpose(1, 2)
    v = attn.v_proj(h).view(shape).transpose(1, 2)
    o = proportional_attention(q, k, v, log_sizes, attn.scale)
    x = x + attn.out_proj(o.transpose(1, 2).reshape(b, t, -1))
    return x + layer.mlp(layer.layer_norm2(x))


def _head_features(model, tokens):
    """The post-processing OWLv2 applies before its heads (see image_embedder)."""
    e = model.owlv2.vision_model.post_layernorm(tokens)
    return model.layer_norm(e[:, 1:] * e[:, :1])


@torch.no_grad()
def merged_forward(model, pixel_values, query, plan: MergePlan, grid: WindowGrid) -> dict:
    """Full detection forward pass with selective merging.

    Returns the three head outputs (same shapes as the unmerged graph) and the
    token count entering each block, from which compute is derived exactly.
    """
    vision = model.owlv2.vision_model
    dtype = vision.embeddings.patch_embedding.weight.dtype
    x = vision.pre_layernorm(vision.embeddings(pixel_values.to(dtype)))

    window_merged = torch.zeros(grid.num_windows, dtype=torch.bool, device=x.device)
    assign = torch.arange(grid.num_patches, device=x.device) + 1
    sizes = None

    if plan.early_fraction > 0:
        if plan.early_scores is None:
            raise ValueError("early merging needs early_scores")
        scores = grid.window_max(plan.early_scores, plan.early_dilate)
        window_merged[scores.argsort()[: round(plan.early_fraction * grid.num_windows)]] = True
        x, sizes, assign = compose(x[:, :1], x[:, 1:], window_merged, grid)

    token_counts = []
    for i, layer in enumerate(vision.encoder.layers):
        if plan.late_block is not None and i == plan.late_block:
            obj = model.objectness_predictor(_head_features(model, x))[0]
            scores = grid.window_max(obj[assign - 1])
            scores[window_merged] = float("inf")  # already merged, not candidates
            extra = round(plan.late_total_fraction * grid.num_windows) - int(window_merged.sum())
            if extra > 0:
                window_merged[scores.argsort()[:extra]] = True
                x, sizes, assign = compose(x[:, :1], x[:, assign], window_merged, grid)
        token_counts.append(x.shape[1])
        log_sizes = None if sizes is None else sizes.log()
        x = layer_forward(layer, x, log_sizes)

    # Unmerge: every patch gets its token back, restoring the fixed grid.
    feats = _head_features(model, torch.cat([x[:, :1], x[:, assign]], dim=1))
    fmap = feats.reshape(1, grid.side, grid.side, -1)
    return {
        "pred_logits": model.class_predictor(feats, query.unsqueeze(0).to(feats.dtype), None)[0],
        "pred_boxes": model.box_predictor(feats, fmap),
        "objectness": model.objectness_predictor(feats),
        "token_counts": token_counts,
        "windows_merged": int(window_merged.sum()),
        "window_merged": window_merged,
    }


def encoder_flops(token_counts, hidden: int = 768, mlp_ratio: int = 4) -> float:
    """Matmul FLOPs of the encoder given the tokens entering each block.

    Per block: QKV + output projections 8*T*d^2, MLP 4*mlp_ratio*T*d^2, and the
    two attention products 4*T^2*d. Exact for a given schedule of token counts.
    """
    return float(sum(8 * t * hidden**2 + 4 * mlp_ratio * t * hidden**2 + 4 * t * t * hidden
                     for t in token_counts))
