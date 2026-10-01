"""Query-based evaluation: score the system the way it is used.

Standard SGDet gives every detection one label - the best of all class names -
and a triplet counts only under that exact label. A deployed system is asked
for something specific ("a person drinking from a cup") and scores each box
against that query alone, so near-synonyms in the vocabulary cannot steal a box.

Here every (predicate, object class) query is scored independently on every
Action Genome test frame:

    frame score = max over the head's top-K pairs of
                  router prob x predicate prob x person score(subject) x class score(object)

Object scores come from two sources, everything else held fixed:

``argmax``  the benchmark's format - a detection scores for its best class only;
``query``   each class scored on its own, from OWLv2's class head applied to the
            detection's cached features (no competition between names).

The gap between the two is what the label format costs. Two retrieval measures,
mean average precision over queries with enough positive frames:

``frame``     does the frame contain the triplet;
``grounded``  ... and does the top-scoring pair match it (IoU >= 0.5 for the
              person and for an object of the queried class carrying the predicate).

The head's routing and role features use the detector's best label as in
deployment; only the final scoring is per query. Queries whose object class is
one of the held-out objects (``train_relationships.py --holdout-objects four``)
are also reported on their own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sggpipeline.ag.classes import AG_OBJECT_CLASSES
from sggpipeline.ag.relations import PREDICATES
from sggpipeline.detect.owlv2 import load_owlv2
from sggpipeline.pipeline import Workspace, load_queries, write_report
from sggpipeline.relations.model import GROUP_SLICES, all_pair_geometry, load_relationship_head
from sggpipeline.relations.predicates import load_or_build
from sggpipeline.relations.sgdet import batched_iou

from heldout_ranking import average_precision
from train_relationships import OBJECT_HOLDOUTS, Frames, pair_mask

PERSON = AG_OBJECT_CLASSES.index("person")


def retrieval_ap(scores: np.ndarray, hits: np.ndarray, n_pos: int) -> float:
    """AP of a ranking where only ``hits`` count as true positives, out of ``n_pos``."""
    order = np.argsort(-scores, kind="stable")
    tp = hits[order].astype(np.float64)
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    return float((precision * tp).sum() / n_pos)


@torch.no_grad()
def collect(model, owl, query, owner, data: Frames, budget: int, batch: int):
    """Per frame and (predicate, class): best score per mode, positives and grounding."""
    c_n, p_n = len(AG_OBJECT_CLASSES), len(PREDICATES)
    out = {k: [] for k in ("argmax", "query", "argmax_hit", "query_hit", "positive")}
    agree = total = 0
    n = data.feats.shape[1]
    for a in tqdm(range(0, len(data), batch), desc="frames", mininterval=60):
        idx = torch.arange(a, min(a + batch, len(data)), device=data.feats.device)
        b = data.batch(idx)
        bsz = len(idx)
        # Independent per-class scores for every kept detection.
        logits = owl.class_predictor(b["feats"].float(), query[None].expand(bsz, -1, -1), None)[0]
        per_class = torch.full((bsz, n, c_n), float("-inf"), device=logits.device)
        per_class = per_class.scatter_reduce(2, owner.expand(bsz, n, -1), logits, reduce="amax").sigmoid()
        per_class = per_class * b["valid"][..., None]
        best_score, best_label = per_class.max(dim=2)
        agree += int(((best_label == b["labels"]) & b["valid"]).sum())
        total += int(b["valid"].sum())
        argmax = torch.zeros_like(per_class).scatter_(2, b["labels"][..., None], b["scores"][..., None])
        argmax = argmax * b["valid"][..., None]

        # The head, exactly as deployed: route, keep the top pairs, classify.
        subj, obj = model.roles(b["feats"].float(), b["labels"])
        geo = all_pair_geometry(b["boxes"], b["sizes"])
        route = model.route(subj, obj, geo).masked_fill(~pair_mask(b["valid"]), float("-inf")).reshape(bsz, -1)
        top = torch.topk(route, min(budget, route.shape[1]), dim=1)
        si, oi = top.indices // n, top.indices % n
        fi = torch.arange(bsz, device=idx.device)[:, None].expand_as(si)
        probs = model.probabilities(model.classify(subj[fi, si], obj[fi, oi], geo[fi, si, oi]))
        pair = torch.sigmoid(top.values)[..., None] * probs  # (B, K, P)

        # Ground truth: frame positives and which detection pairs would be correct.
        m = batched_iou(b["boxes"], b["gt_boxes"]) >= 0.5
        m &= b["valid"][:, :, None] & b["gt_valid"][:, None, :]
        person_ok = m[:, :, 0]
        onehot = torch.nn.functional.one_hot(b["gt_labels"], c_n).bool() & b["gt_valid"][..., None]
        onehot[:, 0] = False  # slot 0 is the person, never the object
        triplet = (b["gt_pred"][..., :, None] & onehot[..., None, :]).float()  # (B, G, P, C)
        out["positive"].append(triplet.amax(dim=1).bool().cpu())
        obj_ok = torch.einsum("bng,bgpc->bnpc", m.float(), triplet) > 0  # (B, N, P, C)
        correct = person_ok.gather(1, si)[..., None, None] & obj_ok[fi, oi]  # (B, K, P, C)

        for mode, scores in (("argmax", argmax), ("query", per_class)):
            s_subj = scores[..., PERSON].gather(1, si)  # (B, K)
            s_obj = scores[fi, oi]  # (B, K, C)
            full = pair[..., None] * s_subj[..., None, None] * s_obj[:, :, None, :]  # (B, K, P, C)
            best, arg = full.max(dim=1)
            out[mode].append(best.cpu())
            out[f"{mode}_hit"].append(correct.gather(1, arg[:, None]).squeeze(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}, agree / max(total, 1)


def summarise(res: dict, min_pos: int) -> dict:
    pos = res["positive"]
    counts = pos.sum(axis=0)
    queries = [(p, c) for p in range(pos.shape[1]) for c in range(pos.shape[2]) if counts[p, c] >= min_pos]
    hidden = {AG_OBJECT_CLASSES.index(x) for x in OBJECT_HOLDOUTS["four"]}
    groups = {name: set(range(sl.start, sl.stop)) for name, sl in GROUP_SLICES.items()}
    per = []
    for p, c in queries:
        row = {"predicate": PREDICATES[p], "object": AG_OBJECT_CLASSES[c], "positives": int(counts[p, c]),
               "chance": float(pos[:, p, c].mean())}
        for mode in ("argmax", "query"):
            s = res[mode][:, p, c]
            row[f"{mode}_frame_ap"] = average_precision(s, pos[:, p, c])
            row[f"{mode}_grounded_ap"] = retrieval_ap(s, res[f"{mode}_hit"][:, p, c] & pos[:, p, c],
                                                      int(counts[p, c]))
        per.append(row)

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    subsets = {"all": per,
               "held_out_objects": [r for r in per if AG_OBJECT_CLASSES.index(r["object"]) in hidden],
               "other_objects": [r for r in per if AG_OBJECT_CLASSES.index(r["object"]) not in hidden]}
    subsets.update({f"predicate_group_{g}": [r for r in per if PREDICATES.index(r["predicate"]) in ids]
                    for g, ids in groups.items()})
    summary = {name: {"queries": len(rows), "chance": mean(rows, "chance"),
                      **{f"{m}_{k}_mAP": mean(rows, f"{m}_{k}_ap")
                         for m in ("argmax", "query") for k in ("frame", "grounded")}}
               for name, rows in subsets.items()}
    return {"summary": summary, "per_query": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--checkpoints", nargs="+", default=["rel_text_none"])
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--min-positives", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    ws = Workspace(Path(args.artifacts))
    test = Frames(dict(np.load(ws.root / "sgdet" / "test.npz", allow_pickle=False)), "cuda")
    embeds = torch.from_numpy(load_or_build(ws.cache("predicate_embeds_base.npy"))).cuda()
    owl, _ = load_owlv2("base", device="cuda", dtype=torch.float32)
    q = load_queries(ws, "base")
    query = torch.from_numpy(q.embeds).cuda()
    owner = torch.as_tensor(q.owner, device="cuda")

    report = {"budget": args.budget, "min_positive_frames": args.min_positives, "frames": len(test)}
    for name in args.checkpoints:
        model = load_relationship_head(ws.root / "sgdet" / f"{name}.pt", len(AG_OBJECT_CLASSES), embeds)
        res, agree = collect(model, owl, query, owner, test, args.budget, args.batch)
        out = summarise(res, args.min_positives)
        out["recomputed_label_agreement"] = agree
        report[name] = out
        print(f"\n{name}: {len(out['per_query'])} queries (>= {args.min_positives} positive frames of "
              f"{len(test)}); class head reproduces cached labels on {agree:.3f} of detections", flush=True)
        print(f"  {'subset':<28}{'queries':>8}{'chance':>8}{'frame AP argmax':>17}{'query':>8}"
              f"{'grounded argmax':>17}{'query':>8}")
        for subset, s in out["summary"].items():
            print(f"  {subset:<28}{s['queries']:>8}{s['chance']:>8.3f}{s['argmax_frame_mAP']:>17.3f}"
                  f"{s['query_frame_mAP']:>8.3f}{s['argmax_grounded_mAP']:>17.3f}{s['query_grounded_mAP']:>8.3f}",
                  flush=True)
    write_report(report, ws.result("relationships_query_retrieval.json"))
    print("QUERY_RETRIEVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
