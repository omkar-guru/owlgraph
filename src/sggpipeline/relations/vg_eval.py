"""Visual Genome (VG150) scene-graph recall, ported from the reference evaluator.

A line-for-line port of the metric code in Scene-Graph-Benchmark.pytorch
(``maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py``: ``SGRecall``,
``SGNoGraphConstraintRecall``, ``SGMeanRecall``), which SG-ViT reports it
replicated "with numerical accuracy". Published VG150 numbers - SG-ViT's
included - are therefore comparable with these. What the port keeps:

* IoU with the +1 pixel convention (``boxlist_iou``'s ``TO_REMOVE = 1``); a
  triplet matches when subject label, predicate and object label agree and both
  boxes reach IoU >= 0.5.
* Ground-truth relations exactly as listed, duplicates included; images without
  any are skipped.
* **Graph constraint**: one predicate per pair (its best), pairs ranked by
  predicate score x subject score x object score, top K pairs.
* **No graph constraint**: every (pair, predicate) ranked by subject score x
  object score x predicate score, top 100.
* Recall@K per image = distinct ground-truth relations hit / relations; averaged
  over images. Mean recall: per predicate, hits/count on each image containing
  it, averaged over those images, then over all predicates (0 for a predicate
  absent from the split).

Predicate indices are 0-based here (the reference's column 0 is background).
Also reported, as for Action Genome: object recall and pair recall (true
subject-object pairs whose detections both reached the classifier), which bound
what the relationship head can recover.
"""

from __future__ import annotations

import numpy as np

KS = (20, 50, 100)


def iou_plus_one(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) x (M, 4) xyxy -> (N, M) IoU, areas counted in whole pixels (+1)."""
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt + 1, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def triplet_hits(gt_rels, gt_classes, gt_boxes, pred_rels, pred_classes, pred_boxes,
                 iou_thres: float = 0.5) -> np.ndarray:
    """(G, P) bool: prediction p recalls ground-truth relation g.

    ``*_rels`` are (n, 3) rows of (subject index, object index, predicate).
    Same decision as the reference's ``_compute_pred_matches`` (non-phrdet).
    """
    if len(gt_rels) == 0 or len(pred_rels) == 0:
        return np.zeros((len(gt_rels), len(pred_rels)), bool)
    same = ((gt_classes[gt_rels[:, 0]][:, None] == pred_classes[pred_rels[:, 0]][None])
            & (gt_rels[:, 2][:, None] == pred_rels[:, 2][None])
            & (gt_classes[gt_rels[:, 1]][:, None] == pred_classes[pred_rels[:, 1]][None]))
    ious = iou_plus_one(gt_boxes, pred_boxes)
    sub = ious[gt_rels[:, 0]][:, pred_rels[:, 0]] >= iou_thres
    obj = ious[gt_rels[:, 1]][:, pred_rels[:, 1]] >= iou_thres
    return same & sub & obj


class VGRecall:
    """Accumulates VG150 SGDet (or PredCls) recall over images."""

    def __init__(self, num_predicates: int = 50, ks: tuple[int, ...] = KS, iou_thres: float = 0.5):
        self.num_predicates = num_predicates
        self.ks = ks
        self.iou_thres = iou_thres
        self.recall = {(m, k): [] for m in ("with_constraint", "no_constraint") for k in ks}
        self.collect = {key: [[] for _ in range(num_predicates)] for key in self.recall}
        self.objects = self.object_hits = 0
        self.pairs = self.pair_hits = self.pair_bound_hits = 0

    def add_image(self, gt_rels: np.ndarray, gt_classes: np.ndarray, gt_boxes: np.ndarray,
                  pair_idx: np.ndarray, rel_scores: np.ndarray, pred_classes: np.ndarray,
                  pred_boxes: np.ndarray, obj_scores: np.ndarray) -> None:
        """One image.

        ``gt_rels`` (M, 3) (subject, object, predicate) into the ground-truth
        boxes; ``pair_idx`` (P, 2) the pairs that reached the classifier, into the
        detections; ``rel_scores`` (P, C) their predicate scores.
        """
        # Diagnostics first: what the detector and the pair budget let through.
        ious = iou_plus_one(gt_boxes, pred_boxes) if len(pred_boxes) else np.zeros((len(gt_boxes), 0))
        match = (ious >= self.iou_thres) & (gt_classes[:, None] == pred_classes[None])  # (G_obj, N)
        self.objects += len(gt_classes)
        self.object_hits += int(match.any(axis=1).sum())
        if len(gt_rels) == 0:
            return
        pairs = np.unique(gt_rels[:, :2], axis=0)
        self.pairs += len(pairs)
        if len(pair_idx):
            sel = match[pairs[:, 0]][:, pair_idx[:, 0]] & match[pairs[:, 1]][:, pair_idx[:, 1]]
            self.pair_hits += int(sel.any(axis=1).sum())
        # Reachable with every pair: distinct detections matching subject and object.
        both = match[pairs[:, 0]][:, :, None] & match[pairs[:, 1]][:, None, :]
        both &= ~np.eye(match.shape[1], dtype=bool)[None]
        self.pair_bound_hits += int(both.reshape(len(pairs), -1).any(axis=1).sum())

        top = max(self.ks)
        if len(pair_idx):
            s, o = pair_idx[:, 0], pair_idx[:, 1]
            obj_pair = obj_scores[s] * obj_scores[o]
            # Graph constraint: each pair's best predicate, pairs by triple score.
            label, score = rel_scores.argmax(axis=1), rel_scores.max(axis=1)
            order = np.argsort(-(score * obj_pair), kind="stable")[:top]
            gc = np.column_stack([s[order], o[order], label[order]])
            # No graph constraint: every (pair, predicate), top 100 overall.
            overall = obj_pair[:, None] * rel_scores
            flat = np.argsort(-overall.ravel(), kind="stable")[:top]
            pi, pp = np.unravel_index(flat, overall.shape)
            ngc = np.column_stack([s[pi], o[pi], pp])
        else:
            gc = ngc = np.zeros((0, 3), np.int64)

        counts = np.bincount(gt_rels[:, 2], minlength=self.num_predicates)
        present = np.flatnonzero(counts)
        for mode, preds in (("with_constraint", gc), ("no_constraint", ngc)):
            hits = triplet_hits(gt_rels, gt_classes, gt_boxes, preds, pred_classes, pred_boxes,
                                self.iou_thres)
            for k in self.ks:
                hit = hits[:, :k].any(axis=1)
                self.recall[(mode, k)].append(hit.sum() / len(gt_rels))
                per = np.bincount(gt_rels[hit, 2], minlength=self.num_predicates)
                for p in present:
                    self.collect[(mode, k)][p].append(per[p] / counts[p])

    def summary(self, seen: np.ndarray | None = None) -> dict:
        out = {"object_recall": self.object_hits / max(self.objects, 1),
               "pair_recall": self.pair_hits / max(self.pairs, 1),
               "pair_recall_upper_bound": self.pair_bound_hits / max(self.pairs, 1),
               "images": len(self.recall[("with_constraint", self.ks[0])])}
        for (mode, k), values in self.recall.items():
            out[f"{mode}/R@{k}"] = float(np.mean(values)) if values else float("nan")
            per = [float(np.mean(v)) if v else 0.0 for v in self.collect[(mode, k)]]
            out[f"{mode}/mR@{k}"] = float(np.mean(per))
            out[f"{mode}/per_predicate_R@{k}"] = per
            if seen is not None:
                for name, mask in (("seen", seen), ("unseen", ~seen)):
                    vals = [per[i] for i in range(self.num_predicates) if mask[i]]
                    out[f"{mode}/mR@{k}_{name}"] = float(np.mean(vals)) if vals else float("nan")
        return out
