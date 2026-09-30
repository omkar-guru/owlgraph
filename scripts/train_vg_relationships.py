"""Train the relationship head on VG150 and score it against SG-ViT's published numbers.

Same head as Action Genome (``RelationshipHead``: directed roles, router, top-K
pairs, text-embedding predicate classifier) with Visual Genome's differences:
any detection can be the subject, and there are 50 predicates in one
multi-label group. Inputs are the frozen detector's top-64 zero-shot detections
(``cache_vg.py``).

Evaluation uses the port of the reference VG evaluator (``relations/vg_eval.py``)
that SG-ViT replicated, on the canonical 26,446 test images, with the checkpoint
chosen on the canonical 5,000-image validation split:

``SGDet``    detected boxes and labels; the router keeps the top K ordered pairs,
             K in {64, 128, 256, 512, all}. Graph-constrained and unconstrained
             R@20/50/100 and mR@20/50/100, plus object and pair recall.
``PredCls``  ground-truth boxes and labels, every pair classified.

A relation's predicate score is router probability x predicate probability; the
reference then multiplies in both object scores.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from sggpipeline.pipeline import Workspace, write_report
from sggpipeline.relations.model import (
    PredicateSchema, RelationshipHead, all_pair_geometry, classification_loss, router_loss)
from sggpipeline.relations.predicates import GENERIC_TEMPLATES, load_or_build
from sggpipeline.relations.sgdet import any_subject_targets, match_matrix, relation_map
from sggpipeline.relations.vg_eval import VGRecall
from sggpipeline.vg.data import vocabulary

PAIR_BUDGETS = (64, 128, 256, 512, None)  # None = every ordered pair

# SG-ViT (Salzmann et al., ECCV 2024) on VG150 test, graph-constrained (Table 2).
SGVIT_PUBLISHED = {"B/32": {"mR@50": 15.0, "mR@100": 18.1},
                   "B/16": {"mR@50": 15.7, "mR@100": 19.3},
                   "L/14": {"mR@50": 17.8, "mR@100": 21.8}}


class Images:
    """A split on the GPU: detections, ground truth and padded relation lists."""

    def __init__(self, a: dict, device: str, predcls: bool = False):
        t = lambda k, dt=None: torch.from_numpy(np.ascontiguousarray(a[k])).to(device, dt)  # noqa: E731
        f = len(a["image_ids"])
        self.image_ids = a["image_ids"]
        self.sizes = t("image_sizes")
        self.gt_boxes = t("gt_boxes")
        self.gt_labels = t("gt_labels", torch.long)
        g = self.gt_boxes.shape[1]
        self.gt_valid = torch.arange(g, device=device)[None] < t("gt_count", torch.long)[:, None]
        self.gt_rels = t("gt_rels", torch.long)
        if predcls:  # ground truth stands in for the detections
            self.feats, self.boxes, self.labels = t("gt_feats"), self.gt_boxes, self.gt_labels
            self.scores = torch.ones(f, g, device=device)
            self.valid = self.gt_valid
        else:
            self.feats = t("det_feats")
            self.boxes = t("det_boxes")
            self.labels = t("det_labels", torch.long)
            self.scores = t("det_scores", torch.float32)
            n = self.feats.shape[1]
            self.valid = torch.arange(n, device=device)[None] < t("det_count", torch.long)[:, None]

    def __len__(self):
        return len(self.feats)

    def batch(self, idx: torch.Tensor) -> dict:
        return {k: getattr(self, k)[idx] for k in ("feats", "boxes", "labels", "scores", "valid", "sizes",
                                                   "gt_boxes", "gt_labels", "gt_valid", "gt_rels")}


def pair_mask(valid: torch.Tensor) -> torch.Tensor:
    n = valid.shape[1]
    return valid[:, :, None] & valid[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=valid.device)


def train_epoch(model, data: Images, opt, batch: int, gen, num_predicates: int) -> float:
    model.train()
    perm = torch.randperm(len(data), device=data.feats.device, generator=gen)
    total, steps = 0.0, 0
    for a in range(0, len(data), batch):
        b = data.batch(perm[a:a + batch])
        match = match_matrix(b["boxes"], b["labels"], b["valid"], b["gt_boxes"], b["gt_labels"], b["gt_valid"])
        rel = relation_map(b["gt_rels"], b["gt_boxes"].shape[1], num_predicates)
        valid = pair_mask(b["valid"])
        positive, targets = any_subject_targets(match, rel)
        positive &= valid
        subj, obj = model.roles(b["feats"].float(), b["labels"])
        geo = all_pair_geometry(b["boxes"], b["sizes"])
        loss = router_loss(model.route(subj, obj, geo), positive, valid)
        bi, si, oi = torch.nonzero(positive, as_tuple=True)
        if len(bi):
            logits = model.classify(subj[bi, si], obj[bi, oi], geo[bi, si, oi])
            loss = loss + classification_loss(logits, targets[bi, si, oi], schema=model.schema)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, data: Images, budgets=PAIR_BUDGETS, batch: int = 64) -> dict:
    """Route, classify every valid pair once, and score each budget's prefix."""
    model.eval()
    accs = {k: VGRecall(len(model.schema)) for k in budgets}
    n = data.feats.shape[1]
    for a in range(0, len(data), batch):
        idx = torch.arange(a, min(a + batch, len(data)), device=data.feats.device)
        b = data.batch(idx)
        subj, obj = model.roles(b["feats"].float(), b["labels"])
        geo = all_pair_geometry(b["boxes"], b["sizes"])
        route = model.route(subj, obj, geo).masked_fill(~pair_mask(b["valid"]), float("-inf")).reshape(len(idx), -1)
        order = torch.argsort(route, dim=1, descending=True)
        n_valid = torch.isfinite(route).sum(dim=1)
        keep = torch.arange(order.shape[1], device=idx.device)[None] < n_valid[:, None]
        fi = torch.arange(len(idx), device=idx.device)[:, None].expand_as(order)
        si, oi = order // n, order % n
        logits = model.classify(subj[fi[keep], si[keep]], obj[fi[keep], oi[keep]],
                                geo[fi[keep], si[keep], oi[keep]])
        rel_scores = (model.probabilities(logits) * torch.sigmoid(route.gather(1, order)[keep])[:, None]).cpu().numpy()
        si_h, oi_h = si.cpu().numpy(), oi.cpu().numpy()
        counts = n_valid.tolist()
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
        host = {k: b[k].cpu().numpy() for k in ("boxes", "labels", "scores", "valid", "gt_boxes",
                                                 "gt_labels", "gt_valid", "gt_rels")}
        for f in range(len(idx)):
            c, s0 = counts[f], starts[f]
            nv, gv = int(host["valid"][f].sum()), int(host["gt_valid"][f].sum())
            rels = host["gt_rels"][f]
            rels = rels[rels[:, 0] >= 0]
            pairs = np.stack([si_h[f, :c], oi_h[f, :c]], axis=1)
            for k, acc in accs.items():
                m = c if k is None else min(k, c)
                acc.add_image(rels, host["gt_labels"][f, :gv], host["gt_boxes"][f, :gv], pairs[:m],
                              rel_scores[s0:s0 + m], host["labels"][f, :nv], host["boxes"][f, :nv],
                              host["scores"][f, :nv])
    return accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--vg-root", default="/workspace/vg150")
    ap.add_argument("--classifier", choices=["text", "closed"], default="text")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-budget", type=int, default=256, help="pair budget used for model selection")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    ws = Workspace(Path(args.artifacts))
    name = f"vg_{args.classifier}"
    classes, predicates = vocabulary(Path(args.vg_root))
    schema = PredicateSchema(predicates, (("predicate", slice(0, len(predicates))),))

    load = lambda s: dict(np.load(ws.root / "vg150" / f"{s}.npz", allow_pickle=False))  # noqa: E731
    train = Images(load("train"), device)
    val_np = load("val")
    val = Images(val_np, device)
    print(f"{name}: train {len(train)} images, val {len(val)}", flush=True)

    embeds = torch.from_numpy(load_or_build(ws.cache("predicate_embeds_vg150.npy"), phrases=list(predicates),
                                            templates=GENERIC_TEMPLATES)).to(device)
    model = RelationshipHead(768, len(classes), embeds, classifier=args.classifier, schema=schema).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    history, best, best_state = [], -1.0, None
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        loss = train_epoch(model, train, opt, args.batch, gen, len(schema))
        sched.step()
        s = evaluate(model, val, budgets=(args.select_budget,))[args.select_budget].summary()
        score = s["with_constraint/mR@100"]
        history.append({"epoch": epoch + 1, "loss": loss, "val_R@100": s["with_constraint/R@100"],
                        "val_mR@100": score, "val_pair_recall": s["pair_recall"],
                        "seconds": time.perf_counter() - t0})
        print(f"  epoch {epoch + 1:2d} loss {loss:.4f} val wc R@100 {s['with_constraint/R@100']:.4f} "
              f"mR@100 {score:.4f} pair recall@{args.select_budget} {s['pair_recall']:.4f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if np.isfinite(score) and score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ws.root / "vg150" / f"{name}.pt")
    del train
    torch.cuda.empty_cache()

    test_np = load("test")
    sgdet = {("all" if k is None else k): acc.summary()
             for k, acc in evaluate(model, Images(test_np, device)).items()}
    pred = evaluate(model, Images(test_np, device, predcls=True), budgets=(None,))[None].summary()
    write_report({"variant": name, "classifier": args.classifier, "history": history,
                  "sgdet_by_pair_budget": sgdet, "predcls": pred, "sgvit_published_vg150": SGVIT_PUBLISHED,
                  "protocol": "port of Scene-Graph-Benchmark sgg_eval (SG-ViT's evaluator): +1 IoU >= 0.5 "
                              "with labels; graph constraint = best predicate per pair; relation score = "
                              "router x predicate, times both object scores; 64 zero-shot detections; "
                              "canonical 26,446 test images; checkpoint chosen on the 5,000 val images"},
                 ws.result(f"relationships_{name}.json"))

    keys = ["object_recall", "pair_recall", "with_constraint/R@50", "with_constraint/R@100",
            "with_constraint/mR@50", "with_constraint/mR@100", "no_constraint/mR@50", "no_constraint/mR@100"]
    short = lambda k: k.replace("with_constraint/", "wc ").replace("no_constraint/", "nc ")  # noqa: E731
    print(f"\n{name} VG150 SGDet (test, {sgdet['all']['images']} images)")
    print(f"{'pairs':>6}" + "".join(f"{short(k):>14}" for k in keys))
    for k, s in sgdet.items():
        print(f"{str(k):>6}" + "".join(f"{s[key]:>14.4f}" for key in keys))
    print(f"{name} VG150 PredCls: wc R@50 {pred['with_constraint/R@50']:.4f} R@100 "
          f"{pred['with_constraint/R@100']:.4f} mR@50 {pred['with_constraint/mR@50']:.4f} mR@100 "
          f"{pred['with_constraint/mR@100']:.4f}")
    print("SG-ViT published (wc): " + ", ".join(f"{m} mR@50 {v['mR@50']} mR@100 {v['mR@100']}"
                                                for m, v in SGVIT_PUBLISHED.items()))
    print("VG_RELATIONSHIPS_DONE", flush=True)


if __name__ == "__main__":
    main()
