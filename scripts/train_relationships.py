"""Train and evaluate the full relationship head (router + pair classifier).

Training data: the frozen detector's top-32 detections per keyframe
(``cache_sgdet.py``). Each step scores every ordered pair with the router
(trained against pairs that truly carry a relation) and classifies the truly
related pairs (teacher forcing), so the classifier learns from every positive
whatever the router currently ranks.

Evaluation, test split, scored once with the checkpoint chosen on held-out
*training* videos:

``SGDet``    the detector's own boxes and labels; the router keeps the top K
             ordered pairs, K in {32, 64, 128, 256, all}. Reports object recall,
             pair recall (true pairs reaching the classifier) and relationship
             Recall / mean Recall @ 20/50/100, with and without the graph
             constraint.
``PredCls``  ground-truth boxes and labels (features from
             ``cache_pair_features.py``), all pairs classified.

``--holdout`` hides predicates during training - their labels contribute no
gradient - and reports seen and unseen mean recall separately. The text
classifier can score them through their phrase embeddings; the closed
classifier cannot, which is what makes it the control.

Triplet score = router probability x predicate probability x subject score x
object score.
"""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from sggpipeline.ag.classes import AG_OBJECT_CLASSES
from sggpipeline.ag.relations import PREDICATES
from sggpipeline.pipeline import Workspace, write_report
from sggpipeline.relations.model import (
    RelationshipHead, all_pair_geometry, classification_loss, predicate_probabilities,
    router_loss)
from sggpipeline.relations.predicates import load_or_build
from sggpipeline.relations.sgdet import SGRecall, match_matrix, pair_targets

HOLDOUTS = {"none": (), "four": ("drinking_from", "lying_on", "wiping", "beneath")}
PAIR_BUDGETS = (32, 64, 128, 256, None)  # None = every ordered pair


class Frames:
    """A split held on the GPU, detections and ground truth padded per frame."""

    def __init__(self, arrays: dict, device: str, index: np.ndarray | None = None):
        sel = slice(None) if index is None else index
        t = lambda k, dt=None: torch.from_numpy(np.ascontiguousarray(arrays[k][sel])).to(device, dt)  # noqa: E731
        self.video_ids = arrays["video_ids"][sel]
        self.feats = t("det_feats")
        self.boxes = t("det_boxes")
        self.labels = t("det_labels", torch.long)
        self.scores = t("det_scores", torch.float32)
        n = self.feats.shape[1]
        self.valid = torch.arange(n, device=device)[None] < t("det_count", torch.long)[:, None]
        self.sizes = t("image_sizes")
        self.gt_boxes = t("gt_boxes")
        self.gt_labels = t("gt_labels", torch.long)
        g = self.gt_boxes.shape[1]
        self.gt_valid = torch.arange(g, device=device)[None] < t("gt_count", torch.long)[:, None]
        self.gt_pred = t("gt_pred")

    def __len__(self):
        return len(self.feats)

    def batch(self, idx: torch.Tensor) -> dict:
        return {k: getattr(self, k)[idx] for k in ("feats", "boxes", "labels", "scores", "valid",
                                                   "sizes", "gt_boxes", "gt_labels", "gt_valid",
                                                   "gt_pred")}


def pair_mask(valid: torch.Tensor) -> torch.Tensor:
    n = valid.shape[1]
    off = ~torch.eye(n, dtype=torch.bool, device=valid.device)
    return valid[:, :, None] & valid[:, None, :] & off


def train_epoch(model, data: Frames, opt, seen: torch.Tensor, batch: int, gen) -> float:
    model.train()
    perm = torch.randperm(len(data), device=data.feats.device, generator=gen)
    total, steps = 0.0, 0
    for a in range(0, len(data), batch):
        b = data.batch(perm[a:a + batch])
        match = match_matrix(b["boxes"], b["labels"], b["valid"], b["gt_boxes"], b["gt_labels"],
                             b["gt_valid"])
        positive, obj_targets = pair_targets(match, b["gt_pred"])
        subj, obj = model.roles(b["feats"].float(), b["labels"])
        geo = all_pair_geometry(b["boxes"], b["sizes"])
        valid = pair_mask(b["valid"])
        loss = router_loss(model.route(subj, obj, geo), positive, valid)
        bi, si, oi = torch.nonzero(positive, as_tuple=True)
        if len(bi):
            logits = model.classify(subj[bi, si], obj[bi, oi], geo[bi, si, oi])
            loss = loss + classification_loss(logits, obj_targets[bi, oi], seen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss)
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, data: Frames, budgets=PAIR_BUDGETS, batch: int = 128,
             score_detections: bool = True) -> dict[str, SGRecall]:
    """Route, classify the top-K pairs and accumulate recall for every budget."""
    model.eval()
    accs = {k: SGRecall() for k in budgets}
    n = data.feats.shape[1]
    for a in range(0, len(data), batch):
        idx = torch.arange(a, min(a + batch, len(data)), device=data.feats.device)
        b = data.batch(idx)
        match = match_matrix(b["boxes"], b["labels"], b["valid"], b["gt_boxes"], b["gt_labels"],
                             b["gt_valid"])
        subj, obj = model.roles(b["feats"].float(), b["labels"])
        geo = all_pair_geometry(b["boxes"], b["sizes"])
        route = model.route(subj, obj, geo).masked_fill(~pair_mask(b["valid"]), float("-inf"))
        order = torch.argsort(route.reshape(len(idx), -1), dim=1, descending=True)
        n_valid = torch.isfinite(route.reshape(len(idx), -1)).sum(dim=1)
        # Classify every valid pair once; each budget then takes a prefix.
        fi = torch.arange(len(idx), device=idx.device)[:, None].expand_as(order)
        si, oi = order // n, order % n
        keep = torch.arange(order.shape[1], device=idx.device)[None] < n_valid[:, None]
        logits = model.classify(subj[fi[keep], si[keep]], obj[fi[keep], oi[keep]],
                                geo[fi[keep], si[keep], oi[keep]])
        probs = predicate_probabilities(logits)
        pair_prob = torch.sigmoid(route.reshape(len(idx), -1).gather(1, order)[keep])
        counts = n_valid.tolist()
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for f in range(len(idx)):
            s0, c = int(starts[f]), counts[f]
            pairs = torch.stack([si[f, :c], oi[f, :c]], dim=1)
            scores = b["scores"][f] if score_detections else torch.ones_like(b["scores"][f])
            for k, acc in accs.items():
                m = c if k is None else min(k, c)
                acc.add_frame(pairs[:m], probs[s0:s0 + m], pair_prob[s0:s0 + m], scores,
                              match[f], b["gt_pred"][f], b["gt_valid"][f], b["valid"][f])
    return accs


def predcls_frames(sgdet_test: dict, pair_features: dict, device: str) -> Frames:
    """Ground-truth boxes and labels as the 'detections', with their patch features."""
    if not np.array_equal(sgdet_test["frame_keys"], pair_features["frame_keys"]):
        raise RuntimeError("PredCls needs the SGDet and pair-feature caches in the same frame order")
    gt_count = sgdet_test["gt_count"].astype(np.int64)
    f, g = sgdet_test["gt_boxes"].shape[:2]
    feats = np.zeros((f, g, 768), np.float16)
    feats[:, 0] = pair_features["subject_patch"]
    pair_frame = pair_features["pair_frame"]
    slot = np.arange(len(pair_frame)) - np.searchsorted(pair_frame, pair_frame)
    feats[pair_frame, 1 + slot] = pair_features["object_patch"]
    arrays = dict(sgdet_test)
    arrays.update(det_feats=feats, det_boxes=sgdet_test["gt_boxes"],
                  det_labels=sgdet_test["gt_labels"], det_scores=np.ones((f, g), np.float16),
                  det_count=gt_count.astype(np.int16))
    return Frames(arrays, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--classifier", choices=["text", "closed"], default="text")
    ap.add_argument("--holdout", choices=sorted(HOLDOUTS), default="none")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-frames", type=int, default=None, help="smoke tests")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    ws = Workspace(Path(args.artifacts))
    name = f"rel_{args.classifier}_{args.holdout}"

    load = lambda p: dict(np.load(p, allow_pickle=False))  # noqa: E731
    train_np = load(ws.root / "sgdet" / "train.npz")
    test_np = load(ws.root / "sgdet" / "test.npz")
    if args.max_train_frames:
        train_np = {k: (v[: args.max_train_frames] if v.ndim and len(v) == len(train_np["frame_keys"]) else v)
                    for k, v in train_np.items()}
    held_out = np.array([zlib.crc32(v.encode()) % 20 == 0 for v in train_np["video_ids"]])
    train = Frames(train_np, device, np.flatnonzero(~held_out))
    val = Frames(train_np, device, np.flatnonzero(held_out))
    test = Frames(test_np, device)
    predcls = predcls_frames(test_np, load(ws.root / "pair_features" / "test.npz"), device)
    print(f"{name}: train {len(train)} frames, val {len(val)} ({len(set(val.video_ids))} held-out "
          f"train videos), test {len(test)}", flush=True)

    seen = torch.ones(len(PREDICATES), dtype=torch.bool, device=device)
    for p in HOLDOUTS[args.holdout]:
        seen[PREDICATES.index(p)] = False
    embeds = torch.from_numpy(load_or_build(ws.cache("predicate_embeds_base.npy"))).to(device)
    model = RelationshipHead(768, len(AG_OBJECT_CLASSES), embeds, classifier=args.classifier).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    history, best, best_state = [], -1.0, None
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        loss = train_epoch(model, train, opt, seen, args.batch, gen)
        sched.step()
        acc = evaluate(model, val, budgets=(128,))[128].summary()
        score = acc["with_constraint/mR@50"]
        history.append({"epoch": epoch + 1, "loss": loss, "val_R@50": acc["with_constraint/R@50"],
                        "val_mR@50": score, "val_pair_recall@128": acc["pair_recall"],
                        "seconds": time.perf_counter() - t0})
        print(f"  epoch {epoch + 1:2d} loss {loss:.4f} val wc R@50 {acc['with_constraint/R@50']:.4f} "
              f"mR@50 {score:.4f} pair recall@128 {acc['pair_recall']:.4f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if np.isfinite(score) and score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ws.root / "sgdet" / f"{name}.pt")

    seen_np = seen.cpu().numpy() if args.holdout != "none" else None
    sgdet = {("all" if k is None else k): acc.summary(seen_np) for k, acc in evaluate(model, test).items()}
    pred = evaluate(model, predcls, budgets=(None,), score_detections=False)[None].summary(seen_np)
    write_report({"variant": name, "classifier": args.classifier,
                  "held_out_predicates": list(HOLDOUTS[args.holdout]), "history": history,
                  "sgdet_by_pair_budget": sgdet, "predcls": pred,
                  "protocol": "IoU>=0.5 and label match; triplet score = router x predicate x "
                              "subject x object; budget 32 detections; test scored once"},
               ws.result(f"relationships_{name}.json"))

    keys = ["object_recall", "pair_recall", "with_constraint/R@20", "with_constraint/R@50",
            "with_constraint/mR@50", "no_constraint/R@50", "no_constraint/mR@50"]
    print(f"\n{name} SGDet (test)")
    print(f"{'pairs':>6}" + "".join(f"{k.replace('with_constraint/', 'wc ').replace('no_constraint/', 'nc '):>14}" for k in keys))
    for k, s in sgdet.items():
        print(f"{str(k):>6}" + "".join(f"{s[key]:>14.4f}" for key in keys))
    print(f"{name} PredCls (test): wc R@20 {pred['with_constraint/R@20']:.4f} R@50 "
          f"{pred['with_constraint/R@50']:.4f} mR@50 {pred['with_constraint/mR@50']:.4f}")
    if seen_np is not None:
        for label, s in (("SGDet@128", sgdet[128]), ("PredCls", pred)):
            print(f"{name} {label} nc mR@50 seen {s['no_constraint/mR@50_seen']:.4f} "
                  f"unseen {s['no_constraint/mR@50_unseen']:.4f}")
    print("RELATIONSHIPS_DONE", flush=True)


if __name__ == "__main__":
    main()
