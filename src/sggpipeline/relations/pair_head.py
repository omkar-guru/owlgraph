"""Person-object relationship head over frozen Stage 1 features, and its metrics.

The head follows the SG-ViT idea from plan.md: separate *subject* and *object*
projections of shared detector features (the role matters - person-holds-cup is
not cup-holds-person), combined with relative box geometry and the object's
label into one pair embedding, then classified into Action Genome's three
predicate groups (attention: one of 3; spatial: any of 6; contacting: any of 17).
It is small on purpose: the question is what the frozen features already carry.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..ag.relations import ATTENTION, CONTACTING, SPATIAL

GEOMETRY_DIM = 15
GROUPS = (("attention", len(ATTENTION)), ("spatial", len(SPATIAL)),
          ("contacting", len(CONTACTING)))


def pair_geometry(subject: torch.Tensor, obj: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    """Relative geometry of (N,4) xyxy subject/object boxes, scale-free.

    Both boxes normalised by image size, the object's centre offset and log size
    ratio relative to the subject, and overlap measures (IoU, and intersection as
    a share of each box - "the cup is inside the person's box" vs the reverse).
    """
    wh = image_size.repeat(1, 2).float()
    s, o = subject / wh, obj / wh
    s_wh = (s[:, 2:] - s[:, :2]).clamp(min=1e-4)
    o_wh = (o[:, 2:] - o[:, :2]).clamp(min=1e-4)
    offset = ((o[:, :2] + o[:, 2:]) - (s[:, :2] + s[:, 2:])) / 2 / s_wh
    log_ratio = torch.log(o_wh / s_wh)
    lo = torch.maximum(s[:, :2], o[:, :2])
    hi = torch.minimum(s[:, 2:], o[:, 2:])
    inter = (hi - lo).clamp(min=0).prod(dim=1, keepdim=True)
    s_area, o_area = s_wh.prod(dim=1, keepdim=True), o_wh.prod(dim=1, keepdim=True)
    iou = inter / (s_area + o_area - inter).clamp(min=1e-8)
    return torch.cat([s, o, offset, log_ratio, iou, inter / o_area, inter / s_area], dim=1)


class PairHead(nn.Module):
    """Directed pair embedding -> attention / spatial / contacting logits.

    ``feature_dim=0`` gives the control that sees only geometry and the object
    label: if the full head cannot beat it, the image features add nothing.
    """

    def __init__(self, feature_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        if feature_dim:
            self.subject = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
            self.object = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
        self.geometry = nn.Sequential(nn.Linear(GEOMETRY_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.label = nn.Embedding(num_classes, hidden)
        self.trunk = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, n) for name, n in GROUPS})

    def forward(self, subject_feat, object_feat, geometry, labels) -> dict[str, torch.Tensor]:
        h = self.geometry(geometry) + self.label(labels)
        if self.feature_dim:
            h = h + self.subject(subject_feat) + self.object(object_feat)
        h = self.trunk(h)
        return {name: head(h) for name, head in self.heads.items()}


def pair_loss(logits: dict, attention, spatial, contacting) -> torch.Tensor:
    return (F.cross_entropy(logits["attention"], attention)
            + F.binary_cross_entropy_with_logits(logits["spatial"], spatial.float())
            + F.binary_cross_entropy_with_logits(logits["contacting"], contacting.float()))


def predicate_scores(logits: dict) -> torch.Tensor:
    """(N, 26) probabilities in PREDICATES order: softmax for attention, sigmoid else."""
    return torch.cat([logits["attention"].softmax(dim=1), logits["spatial"].sigmoid(),
                      logits["contacting"].sigmoid()], dim=1)


def ground_truth(attention: np.ndarray, spatial: np.ndarray, contacting: np.ndarray) -> np.ndarray:
    """(N, 26) multi-hot of true predicates in PREDICATES order."""
    att = np.zeros((len(attention), len(ATTENTION)), dtype=bool)
    att[np.arange(len(attention)), attention] = True
    return np.concatenate([att, spatial.astype(bool), contacting.astype(bool)], axis=1)


def recall_metrics(scores: np.ndarray, truth: np.ndarray, pair_frame: np.ndarray,
                   ks=(10, 20, 50)) -> dict:
    """Recall@K and mean Recall@K over frames, with and without a group constraint.

    Candidates are (pair, predicate) with the predicted probability as score.
    ``with_constraint`` keeps only each pair's best predicate *per group*, so a
    pair contributes at most 3 guesses; ``no_constraint`` lets all 26 compete.
    Per frame, the top K candidates are matched against that frame's true
    (pair, predicate) set. Recall@K is averaged over frames; mean Recall@K
    averages, per predicate, its recall over frames containing it, then averages
    over predicates - so rare predicates count as much as common ones.

    These follow the usual scene-graph definitions, but have not been checked
    line-for-line against any published Action Genome evaluation code.
    """
    num_pred = scores.shape[1]
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(pair_frame)) + 1, [len(pair_frame)]])
    group_slices = [slice(0, len(ATTENTION)), slice(len(ATTENTION), len(ATTENTION) + len(SPATIAL)),
                    slice(len(ATTENTION) + len(SPATIAL), num_pred)]
    constrained = np.zeros_like(scores, dtype=bool)
    for g in group_slices:
        best = scores[:, g].argmax(axis=1) + g.start
        constrained[np.arange(len(scores)), best] = True

    out = {}
    for mode, allowed in (("with_constraint", constrained), ("no_constraint", np.ones_like(constrained))):
        recalls = {k: [] for k in ks}
        per_pred = {k: [[] for _ in range(num_pred)] for k in ks}
        for a, b in zip(bounds[:-1], bounds[1:]):
            s, t, m = scores[a:b], truth[a:b], allowed[a:b]
            if not t.any():
                continue
            cand = np.argwhere(m)
            order = np.argsort(-s[m], kind="stable")
            cand = cand[order]
            for k in ks:
                top = cand[:k]
                hit = np.zeros_like(t)
                hit[top[:, 0], top[:, 1]] = True
                hit &= t
                recalls[k].append(hit.sum() / t.sum())
                present = t.any(axis=0)
                for p in np.flatnonzero(present):
                    per_pred[k][p].append(hit[:, p].sum() / t[:, p].sum())
        for k in ks:
            out[f"{mode}/R@{k}"] = float(np.mean(recalls[k]))
            per = [float(np.mean(v)) for v in per_pred[k][:] if v]
            out[f"{mode}/mR@{k}"] = float(np.mean(per))
            if k == ks[-1]:
                out[f"{mode}/per_predicate_R@{k}"] = [float(np.mean(v)) if v else None
                                                      for v in per_pred[k]]
    return out
