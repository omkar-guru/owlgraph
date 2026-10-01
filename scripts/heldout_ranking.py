"""Held-out predicates, measured by ranking rather than recall.

Recall on held-out predicates rewards calibration, not knowledge. The closed
classifier's held-out columns never receive a gradient, so their logits stay
near their initial value (sigmoid ~0.5) while trained columns learn to say ~0
for absent predicates; on Action Genome, where a frame has a handful of pairs
and the top 50 triplets cover nearly every (pair, predicate), a constant 0.5
column is "recalled" almost perfectly. The text classifier's held-out columns
sit low and rarely win a graph-constrained slot. Neither number says whether
the classifier knows which pairs carry the predicate.

Average precision per predicate does: rank every test pair (PredCls: true
person-object pairs, ground-truth boxes) by that predicate's probability and
score the ranking against the pairs that carry it. A per-column offset cannot
change it, and an uninformed column scores its positive rate (chance).

Held-out *objects* use the same measure on the pairs whose object is one of the
classes removed from training: can the head still read the predicates of a
person-laptop pair when it never saw a laptop? A head that takes the class as a
text embedding has a meaningful input for it; one with learned per-class
vectors has an untrained one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sggpipeline.ag.classes import AG_OBJECT_CLASSES
from sggpipeline.ag.relations import PREDICATES
from sggpipeline.pipeline import Workspace, write_report
from sggpipeline.relations.model import all_pair_geometry, load_relationship_head
from sggpipeline.relations.predicates import load_or_build

from train_relationships import HOLDOUTS, OBJECT_HOLDOUTS, predcls_frames


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Non-interpolated AP: mean precision at each positive, ranked by score."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = labels[order].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


@torch.no_grad()
def pair_probabilities(model, data) -> tuple[np.ndarray, np.ndarray]:
    """(P, 26) probabilities and targets for every true person-object pair."""
    probs, targets, objects = [], [], []
    for a in range(0, len(data), 512):
        idx = torch.arange(a, min(a + 512, len(data)), device=data.feats.device)
        b = data.batch(idx)
        subj, obj = model.roles(b["feats"].float(), b["labels"])
        geo = all_pair_geometry(b["boxes"], b["sizes"])
        fi, oi = torch.nonzero(b["valid"][:, 1:], as_tuple=True)
        oi = oi + 1  # slot 0 is the person, the subject of every AG relation
        si = torch.zeros_like(oi)
        logits = model.classify(subj[fi, si], obj[fi, oi], geo[fi, si, oi])
        probs.append(model.probabilities(logits).cpu())
        targets.append(b["gt_pred"][fi, oi].cpu())
        objects.append(b["labels"][fi, oi].cpu())
    return torch.cat(probs).numpy(), torch.cat(targets).numpy(), torch.cat(objects).numpy()


def map_over(probs, targets, rows, min_pos: int = 10) -> tuple[float, int]:
    """Mean AP over predicates with at least ``min_pos`` positives among ``rows``."""
    aps = [average_precision(probs[rows, p], targets[rows, p]) for p in range(targets.shape[1])
           if targets[rows, p].sum() >= min_pos]
    return float(np.mean(aps)), len(aps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    ws = Workspace(Path(args.artifacts))
    load = lambda p: dict(np.load(p, allow_pickle=False))  # noqa: E731
    test_np = load(ws.root / "sgdet" / "test.npz")
    data = predcls_frames(test_np, load(ws.root / "pair_features" / "test.npz"), "cuda")
    embeds = torch.from_numpy(load_or_build(ws.cache("predicate_embeds_base.npy"))).cuda()
    held = [PREDICATES.index(p) for p in HOLDOUTS["four"]]
    seen = np.ones(len(PREDICATES), bool)
    seen[held] = False

    report = {}

    def head(name, classifier="text"):
        path = ws.root / "sgdet" / f"{name}.pt"
        return load_relationship_head(path, len(AG_OBJECT_CLASSES), embeds, classifier) if path.exists() else None

    # 1. Held-out predicates (heads trained with per-class label vectors).
    for classifier in ("text", "closed"):
        for holdout in ("none", "four"):
            name = f"rel_{classifier}_{holdout}_idlabel"
            model = head(name, classifier)
            if model is None:
                continue
            probs, targets, _ = pair_probabilities(model, data)
            ap_ = np.array([average_precision(probs[:, p], targets[:, p]) for p in range(len(PREDICATES))])
            chance = targets.mean(axis=0)
            report[name] = {"per_predicate_ap": dict(zip(PREDICATES, ap_.tolist())),
                            "mAP_seen": float(np.nanmean(ap_[seen])),
                            "mAP_held_out_four": float(np.nanmean(ap_[held])),
                            "chance_per_held_out": dict(zip(HOLDOUTS["four"], chance[held].tolist())),
                            "pairs": len(targets)}
            print(f"{name:<26} mAP seen {report[name]['mAP_seen']:.3f}  held-out predicates "
                  f"{report[name]['mAP_held_out_four']:.3f}  ("
                  + ", ".join(f"{PREDICATES[p]} {ap_[p]:.3f} vs chance {chance[p]:.3f}" for p in held)
                  + ")", flush=True)

    # 2. Held-out objects: predicates of pairs whose object class was never trained on.
    hidden = np.array([AG_OBJECT_CLASSES.index(c) for c in OBJECT_HOLDOUTS["four"]])
    for name in ("rel_text_none", "rel_text_none_objfour", "rel_text_none_idlabel_objfour"):
        model = head(name)
        if model is None:
            continue
        probs, targets, objects = pair_probabilities(model, data)
        unseen = np.isin(objects, hidden)
        m_unseen, n_unseen = map_over(probs, targets, unseen)
        m_seen, n_seen = map_over(probs, targets, ~unseen)
        per_class = {AG_OBJECT_CLASSES[c]: map_over(probs, targets, objects == c)[0] for c in hidden}
        report[name] = {"mAP_pairs_with_held_out_objects": m_unseen, "predicates_scored_unseen": n_unseen,
                        "mAP_pairs_with_other_objects": m_seen, "predicates_scored_seen": n_seen,
                        "per_held_out_object": per_class, "pairs_with_held_out_objects": int(unseen.sum())}
        print(f"{name:<30} predicate mAP on pairs with held-out objects {m_unseen:.3f} ({unseen.sum()} pairs; "
              + ", ".join(f"{k} {v:.3f}" for k, v in per_class.items()) + f"); other pairs {m_seen:.3f}",
              flush=True)

    report["protocol"] = ("PredCls test pairs (person -> each ground-truth object); per-predicate "
                          "average precision over all pairs, or over pairs grouped by object class "
                          "(predicates with >= 10 positives); chance = positive rate; held-out objects: "
                          + ", ".join(OBJECT_HOLDOUTS["four"]))
    write_report(report, ws.result("relationships_heldout_ranking.json"))
    print("HELDOUT_RANKING_DONE", flush=True)


if __name__ == "__main__":
    main()
