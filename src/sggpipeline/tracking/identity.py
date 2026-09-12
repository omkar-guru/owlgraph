"""Instance descriptor and cross-frame supervised contrastive objective."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class IdentityHead(nn.Module):
    """Small adapter over frozen visual features; does not alter semantic scores."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("Head dimensions must be positive")
        self.dimensions = dict(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)


def identity_loss(descriptors: torch.Tensor, instance_ids: torch.Tensor,
                  frame_ids: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss using globally scoped instance IDs.

    Positive pairs share a physical instance across DIFFERENT frames. All other
    known instances are negatives, including those of the same semantic class.
    IDs < 0 are unknown and excluded entirely. The caller must namespace instance
    and frame IDs by video; category labels must never be supplied as instance IDs.
    """
    if not 0 < temperature < float("inf"):
        raise ValueError("Temperature must be positive and finite")
    if descriptors.ndim != 2 or instance_ids.shape != descriptors.shape[:1] or frame_ids.shape != instance_ids.shape:
        raise ValueError("Expected descriptors (N,D), instance IDs (N), frame IDs (N)")
    if not torch.isfinite(descriptors).all():
        raise ValueError("Descriptors must be finite")
    known = instance_ids >= 0
    same = instance_ids[:, None] == instance_ids[None, :]
    positive = same & (frame_ids[:, None] != frame_ids[None, :])
    candidate = known[:, None] & known[None, :] & (~same | positive)
    positive &= candidate
    # Require both positive and negative evidence for each included anchor.
    usable = positive.any(dim=1) & (candidate & ~same).any(dim=1)
    if not usable.any():
        raise ValueError("Identity batches need cross-frame positives and different-instance negatives")
    z = F.normalize(descriptors, dim=-1)
    logits = (z @ z.T / temperature)[usable]
    candidate, positive = candidate[usable], positive[usable]
    log_probability = logits - torch.logsumexp(logits.masked_fill(~candidate, -torch.inf), dim=1, keepdim=True)
    return -(log_probability.masked_fill(~positive, 0).sum(dim=1) / positive.sum(dim=1)).mean()
