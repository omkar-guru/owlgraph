"""Train and evaluate the pair head on cached frozen-detector features (PredCls).

Ground-truth boxes and object labels are given; only relationships are
predicted. That isolates the relationship head from detection errors and asks
one question: do the frozen OWLv2 features carry relationship information beyond
what box geometry and the object's label already give?

Variants, all scored on the same test pairs:

``frequency``       no model: each predicate's training frequency given the object class
``geometry_label``  the head with geometry and label only - the control
``pool``            + features averaged under each ground-truth box
``patch``           + the best-matching patch's feature (what deployment provides)

Model selection uses a held-out 5% of *training* videos (whole videos, so no
leakage between near-duplicate keyframes); the test split is scored once per
variant with the selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from sggpipeline.ag.classes import AG_OBJECT_CLASSES
from sggpipeline.ag.relations import ATTENTION, PREDICATES
from sggpipeline.relations.pair_head import (
    PairHead, ground_truth, pair_geometry, pair_loss, predicate_scores, recall_metrics)
from sggpipeline.pipeline import Workspace, write_report


def load(path: Path, device: str) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    pf = torch.from_numpy(d["pair_frame"]).to(device)
    sizes = torch.from_numpy(d["image_sizes"]).to(device)[pf]
    person = torch.from_numpy(d["person_boxes"]).to(device)[pf]
    obj_boxes = torch.from_numpy(d["object_boxes"]).to(device)
    return {
        "np": d,
        "pair_frame": d["pair_frame"],
        "video_of_pair": d["video_ids"][d["pair_frame"]],
        "geometry": pair_geometry(person, obj_boxes, sizes),
        "labels": torch.from_numpy(d["object_labels"]).to(device),
        "attention": torch.from_numpy(d["attention"]).to(device),
        "spatial": torch.from_numpy(d["spatial"]).to(device),
        "contacting": torch.from_numpy(d["contacting"]).to(device),
        "subject": {k: torch.from_numpy(d[f"subject_{k}"]).to(device)[pf] for k in ("pool", "patch")},
        "object": {k: torch.from_numpy(d[f"object_{k}"]).to(device) for k in ("pool", "patch")},
        "truth": ground_truth(d["attention"], d["spatial"], d["contacting"]),
    }


def subset(data: dict, mask: np.ndarray) -> dict:
    idx = torch.from_numpy(np.flatnonzero(mask)).to(data["labels"].device)
    out = {"pair_frame": data["pair_frame"][mask], "truth": data["truth"][mask]}
    for k in ("geometry", "labels", "attention", "spatial", "contacting"):
        out[k] = data[k][idx]
    out["subject"] = {k: v[idx] for k, v in data["subject"].items()}
    out["object"] = {k: v[idx] for k, v in data["object"].items()}
    return out


def frequency_scores(train: dict, target: dict) -> np.ndarray:
    """P(predicate | object class) from training pairs; no model at all."""
    num_classes = len(AG_OBJECT_CLASSES)
    labels = train["labels"].cpu().numpy()
    table = np.zeros((num_classes, len(PREDICATES)))
    for c in range(num_classes):
        rows = train["truth"][labels == c]
        if len(rows):
            table[c] = rows.mean(axis=0)
    table[:, :len(ATTENTION)] /= table[:, :len(ATTENTION)].sum(axis=1, keepdims=True).clip(min=1e-9)
    return table[target["labels"].cpu().numpy()]


@torch.no_grad()
def model_scores(model, data: dict, feature: str | None, batch: int = 16384) -> np.ndarray:
    model.eval()
    out = []
    for a in range(0, len(data["labels"]), batch):
        sl = slice(a, a + batch)
        s = data["subject"][feature][sl].float() if feature else None
        o = data["object"][feature][sl].float() if feature else None
        out.append(predicate_scores(model(s, o, data["geometry"][sl], data["labels"][sl])).cpu())
    return torch.cat(out).numpy()


def train_variant(train: dict, val: dict, feature: str | None, epochs: int, seed: int,
                  device: str) -> tuple[PairHead, list[dict]]:
    torch.manual_seed(seed)
    model = PairHead(768 if feature else 0, len(AG_OBJECT_CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n, batch = len(train["labels"]), 4096
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    history, best, best_state = [], -1.0, None
    g = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device, generator=g)
        total = 0.0
        for a in range(0, n, batch):
            idx = perm[a:a + batch]
            s = train["subject"][feature][idx].float() if feature else None
            o = train["object"][feature][idx].float() if feature else None
            logits = model(s, o, train["geometry"][idx], train["labels"][idx])
            loss = pair_loss(logits, train["attention"][idx], train["spatial"][idx],
                             train["contacting"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        m = recall_metrics(model_scores(model, val, feature), val["truth"], val["pair_frame"],
                           ks=(20,))
        score = m["with_constraint/mR@20"]
        history.append({"epoch": epoch + 1, "train_loss": total / n,
                        "val_R@20": m["with_constraint/R@20"], "val_mR@20": score})
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  {feature or 'geometry_label'} epoch {epoch + 1:2d} loss {total / n:.4f} "
              f"val R@20 {m['with_constraint/R@20']:.4f} mR@20 {score:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    ws = Workspace(Path(args.artifacts))
    feat_dir = ws.root / "pair_features"
    full_train = load(feat_dir / "train.npz", device)
    test = load(feat_dir / "test.npz", device)

    held_out = np.array([zlib.crc32(v.encode()) % 20 == 0 for v in full_train["video_of_pair"]])
    train, val = subset(full_train, ~held_out), subset(full_train, held_out)
    print(f"train pairs {len(train['labels'])}, val pairs {len(val['labels'])} "
          f"({len(set(full_train['video_of_pair'][held_out]))} held-out train videos), "
          f"test pairs {len(test['labels'])}", flush=True)

    iou = full_train["np"]["object_match_iou"]
    results = {"setting": "PredCls: ground-truth boxes and labels given",
               "train_object_best_patch_iou": {"median": float(np.median(iou)),
                                               "share_ge_0.5": float((iou >= 0.5).mean())},
               "variants": {}}
    results["variants"]["frequency"] = {
        "test": recall_metrics(frequency_scores(train, test), test["truth"], test["pair_frame"])}
    for feature in (None, "pool", "patch"):
        name = feature or "geometry_label"
        model, history = train_variant(train, val, feature, args.epochs, args.seed, device)
        metrics = recall_metrics(model_scores(model, test, feature), test["truth"], test["pair_frame"])
        results["variants"][name] = {"history": history, "test": metrics}
        torch.save(model.state_dict(), ws.root / "pair_features" / f"pair_head_{name}.pt")

    write_report(results, ws.result("pair_head_predcls.json"))
    cols = ["with_constraint/R@10", "with_constraint/R@20", "with_constraint/mR@10",
            "with_constraint/mR@20", "no_constraint/R@20", "no_constraint/R@50",
            "no_constraint/mR@20", "no_constraint/mR@50"]
    print("\n" + f"{'variant':<16}" + "".join(f"{c.replace('with_constraint/', 'wc ').replace('no_constraint/', 'nc '):>10}" for c in cols))
    for name, v in results["variants"].items():
        print(f"{name:<16}" + "".join(f"{v['test'][c]:>10.4f}" for c in cols))
    print("PAIR_HEAD_DONE", flush=True)


if __name__ == "__main__":
    main()
