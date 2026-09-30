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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sggpipeline.ag.classes import AG_OBJECT_CLASSES
from sggpipeline.ag.relations import PREDICATES
from sggpipeline.pipeline import Workspace, write_report
from sggpipeline.relations.model import RelationshipHead, all_pair_geometry
from sggpipeline.relations.predicates import load_or_build

from train_relationships import HOLDOUTS, predcls_frames


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
    probs, targets = [], []
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
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


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
    for classifier in ("text", "closed"):
        for holdout in ("none", "four"):
            name = f"rel_{classifier}_{holdout}"
            model = RelationshipHead(768, len(AG_OBJECT_CLASSES), embeds, classifier=classifier).cuda()
            model.load_state_dict(torch.load(ws.root / "sgdet" / f"{name}.pt", map_location="cuda",
                                             weights_only=True))
            model.eval()
            probs, targets = pair_probabilities(model, data)
            ap_ = np.array([average_precision(probs[:, p], targets[:, p]) for p in range(len(PREDICATES))])
            chance = targets.mean(axis=0)
            report[name] = {"per_predicate_ap": dict(zip(PREDICATES, ap_.tolist())),
                            "mAP_seen": float(np.nanmean(ap_[seen])),
                            "mAP_held_out_four": float(np.nanmean(ap_[held])),
                            "chance_held_out_four": float(chance[held].mean()), "pairs": len(targets)}
            print(f"{name:<18} mAP seen {report[name]['mAP_seen']:.3f}  held-out four "
                  f"{report[name]['mAP_held_out_four']:.3f}  (chance {chance[held].mean():.3f}; "
                  + ", ".join(f"{PREDICATES[p]} {ap_[p]:.3f}" for p in held) + ")", flush=True)
    report["protocol"] = ("PredCls test pairs (person -> each ground-truth object); per-predicate "
                          "average precision over all pairs; chance = positive rate")
    write_report(report, ws.result("relationships_heldout_ranking.json"))
    print("HELDOUT_RANKING_DONE", flush=True)


if __name__ == "__main__":
    main()
