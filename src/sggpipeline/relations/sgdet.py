"""Matching detections to Action Genome ground truth, and scene-graph recall.

One set of rules serves training targets and evaluation alike:

* A detection **matches** a ground-truth box when their IoU >= 0.5 *and* the
  labels agree.
* Ground-truth box 0 of every frame is the person (the subject of every AG
  relation); boxes 1.. are objects, each with its true predicates.
* A detected ordered pair (i, j) is **positive** when i matches the person and j
  matches some object; its predicate targets are the union over matched objects.

Evaluation is standard SGDet: a predicted triplet (i, j, p) recalls the true
triplet (object k, p) when i matches the person and j matches object k. Recall@R
is the share of a frame's true triplets hit by its top R predictions, averaged
over frames; mean Recall@R averages that per predicate first (over frames
containing the predicate), so rare predicates weigh as much as common ones.

Also reported, because the head cannot recover what earlier stages dropped:
**object recall** (true boxes matched by any kept detection) and **pair recall**
(true person-object pairs whose two matching detections are among the router's
selected pairs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import GROUP_SLICES, NUM_PREDICATES

MATCH_IOU = 0.5


def batched_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, N, 4) x (B, G, 4) xyxy -> (B, N, G) IoU, in one vectorised pass."""
    lo = torch.maximum(a[:, :, None, :2], b[:, None, :, :2])
    hi = torch.minimum(a[:, :, None, 2:], b[:, None, :, 2:])
    inter = (hi - lo).clamp(min=0).prod(dim=-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(dim=-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(dim=-1)
    return inter / (area_a[:, :, None] + area_b[:, None, :] - inter).clamp(min=1e-9)


def match_matrix(det_boxes, det_labels, det_valid, gt_boxes, gt_labels, gt_valid,
                 iou: float = MATCH_IOU) -> torch.Tensor:
    """(B, N, G) bool: detection n matches ground-truth box g."""
    ious = batched_iou(det_boxes, gt_boxes)
    same = det_labels[:, :, None] == gt_labels[:, None, :]
    return (ious >= iou) & same & det_valid[:, :, None] & gt_valid[:, None, :]


def pair_targets(match: torch.Tensor, gt_predicates: torch.Tensor):
    """Positive ordered pairs and their predicate targets.

    ``match``: (B, N, G); ``gt_predicates``: (B, G, 26) with row 0 the person.
    Returns ``positive`` (B, N, N) and ``targets`` (B, N, 26) - the predicates for
    detection j *as an object*; a positive pair (i, j) takes row j.
    """
    subject = match[:, :, 0]
    obj_match = match[:, :, 1:].float()
    obj_predicates = torch.einsum("bng,bgp->bnp", obj_match, gt_predicates[:, 1:].float()) > 0
    is_object = match[:, :, 1:].any(dim=-1)
    n = match.shape[1]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=match.device)
    positive = subject[:, :, None] & is_object[:, None, :] & off_diagonal
    return positive, obj_predicates


def constrained_mask(probs: torch.Tensor) -> torch.Tensor:
    """Keep only each pair's best predicate per group (graph constraint)."""
    mask = torch.zeros_like(probs, dtype=torch.bool)
    for sl in GROUP_SLICES.values():
        best = probs[:, sl].argmax(dim=1) + sl.start
        mask[torch.arange(len(probs), device=probs.device), best] = True
    return mask


@dataclass
class SGRecall:
    """Accumulates SGDet / PredCls recall over frames."""

    ks: tuple[int, ...] = (20, 50, 100)
    recall: dict = field(default_factory=dict)
    per_pred: dict = field(default_factory=dict)
    pair_hits: int = 0
    pair_bound_hits: int = 0
    pairs: int = 0
    object_hits: int = 0
    objects: int = 0

    def __post_init__(self):
        for mode in ("with_constraint", "no_constraint"):
            for k in self.ks:
                self.recall[(mode, k)] = []
                self.per_pred[(mode, k)] = [[] for _ in range(NUM_PREDICATES)]

    def add_frame(self, pairs: torch.Tensor, probs: torch.Tensor, pair_scores: torch.Tensor,
                  det_scores: torch.Tensor, match: torch.Tensor, gt_predicates: torch.Tensor,
                  gt_valid: torch.Tensor, det_valid: torch.Tensor) -> None:
        """One frame. ``pairs`` (P, 2) selected (subject, object) detection indices;
        ``probs`` (P, 26); ``pair_scores`` (P,) router probabilities; ``match``
        (N, G); ``gt_predicates`` (G, 26) with row 0 the person."""
        truth = gt_predicates[1:] & gt_valid[1:, None]  # (G-1, 26) true triplets
        obj_match = match[:, 1:]  # (N, G-1)
        subj_match = match[:, 0]  # (N,)

        # Object recall: every true box, person included, matched by any detection.
        self.objects += int(gt_valid.sum())
        self.object_hits += int((match.any(dim=0) & gt_valid).sum())

        # Pair recall: true person-object pairs that reached the classifier.
        related = truth.any(dim=1)  # objects with at least one predicate
        self.pairs += int(related.sum())
        if len(pairs):
            sel = subj_match[pairs[:, 0], None] & obj_match[pairs[:, 1]]  # (P, G-1)
            self.pair_hits += int((sel.any(dim=0) & related).sum())
        valid_subj = subj_match & det_valid
        reachable = (obj_match & det_valid[:, None]).any(dim=0)
        # Upper bound: some detection pair matches, if every pair were classified.
        self.pair_bound_hits += int((reachable & related & bool(valid_subj.any())).sum())

        n_true = int(truth.sum())
        if n_true == 0:
            return
        if len(pairs) == 0:
            for key in self.recall:
                self.recall[key].append(0.0)
                for p in np.flatnonzero(truth.any(dim=0).cpu().numpy()):
                    self.per_pred[key][p].append(0.0)
            return
        base = pair_scores[:, None] * probs * det_scores[pairs[:, 0], None] * det_scores[pairs[:, 1], None]
        hit_map = subj_match[pairs[:, 0], None] & obj_match[pairs[:, 1]]  # (P, G-1)
        present = truth.any(dim=0).cpu().numpy()
        for mode in ("with_constraint", "no_constraint"):
            allowed = constrained_mask(probs) if mode == "with_constraint" else torch.ones_like(probs, dtype=torch.bool)
            pi, pp = torch.nonzero(allowed, as_tuple=True)
            order = torch.argsort(base[pi, pp], descending=True)
            pi, pp = pi[order], pp[order]
            for k in self.ks:
                ti, tp = pi[:k], pp[:k]
                onehot = torch.zeros(len(ti), NUM_PREDICATES, device=probs.device)
                onehot[torch.arange(len(ti)), tp] = 1.0
                hit = (hit_map[ti].float().T @ onehot) > 0  # (G-1, 26)
                hit &= truth
                self.recall[(mode, k)].append(float(hit.sum()) / n_true)
                hits_p = hit.sum(dim=0).cpu().numpy()
                true_p = truth.sum(dim=0).cpu().numpy()
                for p in np.flatnonzero(present):
                    self.per_pred[(mode, k)][p].append(hits_p[p] / true_p[p])

    def summary(self, seen: np.ndarray | None = None) -> dict:
        """Recall/mean recall per mode and K; with ``seen``, also split mR by it."""
        out = {"object_recall": self.object_hits / max(self.objects, 1),
               "pair_recall": self.pair_hits / max(self.pairs, 1),
               "pair_recall_upper_bound": self.pair_bound_hits / max(self.pairs, 1),
               "frames": len(next(iter(self.recall.values())))}
        for (mode, k), values in self.recall.items():
            out[f"{mode}/R@{k}"] = float(np.mean(values)) if values else float("nan")
            per = [float(np.mean(v)) if v else None for v in self.per_pred[(mode, k)]]
            present = [x for x in per if x is not None]
            out[f"{mode}/mR@{k}"] = float(np.mean(present)) if present else float("nan")
            out[f"{mode}/per_predicate_R@{k}"] = per
            if seen is not None:
                for name, mask in (("seen", seen), ("unseen", ~seen)):
                    vals = [per[i] for i in range(NUM_PREDICATES) if mask[i] and per[i] is not None]
                    out[f"{mode}/mR@{k}_{name}"] = float(np.mean(vals)) if vals else float("nan")
        return out
