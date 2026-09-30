"""The relationship head: SG-ViT-style routing and classification over detections.

Per frame, given up to N detections (boxes, labels, per-detection features from
the frozen detector):

1. **Role features.** Each detection gets a *subject* and an *object* projection
   of its feature plus a label embedding. Roles are separate because the pair is
   directed: person-holds-cup is not cup-holds-person.
2. **Router.** Every ordered pair (i, j), i != j, gets a cheap score: a dot
   product between i's subject key and j's object key, plus a small term on
   their relative geometry. Only the top-K pairs go further; the rest are never
   classified, which is where the saving is - and why pair-selection recall has
   to be measured.
3. **Pair embedding.** Selected pairs combine both role features and a richer
   geometry encoding into one embedding.
4. **Predicate classifier.** ``text``: cosine similarity between the projected
   pair embedding and predicate text embeddings, so a predicate never seen as a
   label can still be scored. ``closed``: a fixed linear layer, the control that
   cannot score unseen predicates at all.

Predicates follow Action Genome's three groups: attention (exactly one of 3,
softmax), spatial (any of 6) and contacting (any of 17), both sigmoid.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..ag.relations import ATTENTION, CONTACTING, PREDICATES, SPATIAL
from .pair_head import GEOMETRY_DIM, pair_geometry

NUM_PREDICATES = len(PREDICATES)
GROUP_SLICES = {
    "attention": slice(0, len(ATTENTION)),
    "spatial": slice(len(ATTENTION), len(ATTENTION) + len(SPATIAL)),
    "contacting": slice(len(ATTENTION) + len(SPATIAL), NUM_PREDICATES),
}


def all_pair_geometry(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    """(B, N, N, G) geometry of every ordered pair (subject i, object j)."""
    b, n, _ = boxes.shape
    subj = boxes[:, :, None, :].expand(b, n, n, 4).reshape(-1, 4)
    obj = boxes[:, None, :, :].expand(b, n, n, 4).reshape(-1, 4)
    size = image_size[:, None, :].expand(b, n * n, 2).reshape(-1, 2)
    return pair_geometry(subj, obj, size).reshape(b, n, n, GEOMETRY_DIM)


def predicate_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over attention, sigmoid over spatial and contacting."""
    a = GROUP_SLICES["attention"]
    return torch.cat([logits[..., a].softmax(dim=-1), logits[..., a.stop:].sigmoid()], dim=-1)


class RelationshipHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, predicate_embeds: torch.Tensor | None,
                 classifier: str = "text", hidden: int = 512, key_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        if classifier not in ("text", "closed"):
            raise ValueError("classifier must be 'text' or 'closed'")
        if classifier == "text" and predicate_embeds is None:
            raise ValueError("the text classifier needs predicate embeddings")
        self.classifier = classifier
        self.key_dim = key_dim
        self.role_subject = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
        self.role_object = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
        self.label = nn.Embedding(num_classes, hidden)
        self.subject_key = nn.Linear(hidden, key_dim)
        self.object_key = nn.Linear(hidden, key_dim)
        self.router_geometry = nn.Sequential(nn.Linear(GEOMETRY_DIM, 64), nn.GELU(), nn.Linear(64, 1))
        self.geometry = nn.Sequential(nn.Linear(GEOMETRY_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.trunk = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        if classifier == "text":
            self.register_buffer("predicate_embeds", F.normalize(predicate_embeds.float(), dim=1))
            self.to_text = nn.Linear(hidden, predicate_embeds.shape[1])
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            # One bias per group, never per predicate: a per-predicate bias would
            # learn each seen predicate's frequency and leave unseen ones at an
            # arbitrary value, biasing any held-out comparison.
            self.group_bias = nn.Parameter(torch.zeros(2))
        else:
            self.closed = nn.Linear(hidden, NUM_PREDICATES)

    # -- per detection ---------------------------------------------------------
    def roles(self, features: torch.Tensor, labels: torch.Tensor):
        lab = self.label(labels)
        return self.role_subject(features) + lab, self.role_object(features) + lab

    # -- router ------------------------------------------------------------------
    def route(self, subj: torch.Tensor, obj: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """(B, N, N) pair logits from (B, N, H) roles and (B, N, N, G) geometry."""
        keys = self.subject_key(subj) @ self.object_key(obj).transpose(1, 2)
        return keys / math.sqrt(self.key_dim) + self.router_geometry(geometry).squeeze(-1)

    # -- classifier --------------------------------------------------------------
    def classify(self, subj: torch.Tensor, obj: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """(P, 26) predicate logits for P selected pairs."""
        h = self.trunk(subj + obj + self.geometry(geometry))
        if self.classifier == "closed":
            return self.closed(h)
        z = F.normalize(self.to_text(h), dim=-1)
        logits = self.logit_scale.exp() * z @ self.predicate_embeds.T
        bias = torch.zeros(NUM_PREDICATES, device=logits.device, dtype=logits.dtype)
        bias[GROUP_SLICES["spatial"]] = self.group_bias[0]
        bias[GROUP_SLICES["contacting"]] = self.group_bias[1]
        return logits + bias


def classification_loss(logits: torch.Tensor, targets: torch.Tensor,
                        seen: torch.Tensor | None = None) -> torch.Tensor:
    """Attention cross-entropy plus spatial/contacting BCE, over seen predicates only.

    Held-out predicates contribute no gradient anywhere: their columns are
    dropped from the BCE, and a pair whose true attention predicate is held out
    is dropped from the attention term, which is taken over seen columns only.
    """
    if seen is None:
        seen = torch.ones(NUM_PREDICATES, dtype=torch.bool, device=logits.device)
    a = GROUP_SLICES["attention"]
    att_logits = logits[:, a].masked_fill(~seen[a], float("-inf"))
    att_target = targets[:, a].float().argmax(dim=1)
    keep = seen[a][att_target] & targets[:, a].any(dim=1)
    loss = logits.new_zeros(())
    if keep.any():
        loss = loss + F.cross_entropy(att_logits[keep], att_target[keep])
    rest = slice(a.stop, NUM_PREDICATES)
    cols = seen[rest]
    if cols.any():
        loss = loss + F.binary_cross_entropy_with_logits(
            logits[:, rest][:, cols], targets[:, rest][:, cols].float())
    return loss


def router_loss(pair_logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor,
                pos_weight: float = 50.0) -> torch.Tensor:
    """BCE over valid ordered pairs; positives are rare (~2 in ~1,000), so upweighted."""
    weight = torch.where(positive, torch.full_like(pair_logits, pos_weight),
                         torch.ones_like(pair_logits))
    return F.binary_cross_entropy_with_logits(pair_logits[valid], positive[valid].float(),
                                              weight=weight[valid])
