"""Why does OWLv2 lose strict-IoU accuracy at reduced input resolution?

The resolution sweep showed mAP@50 nearly intact at 640 while mAP@[.5:.95]
collapsed - the detector still finds and classifies objects, but boxes them
imprecisely. That leaves two very different causes, with very different fixes:

**Systematic bias** - boxes consistently shifted or mis-scaled. Correctable with
a per-class affine calibration for essentially zero cost and no training.

**Random jitter** - boxes scattered around the right answer. No calibration can
help; it needs a learned refinement head that re-regresses from pooled features.

This script separates them. For each ground-truth box it takes the best-matching
prediction and measures the residual in a scale-invariant form:

* centre offset as a fraction of GT width/height
* log size ratio, so over- and under-estimation are symmetric

The **mean** of each residual is the bias; the **standard deviation** is the
jitter. Comparing 640 against 960 on identical frames shows which one grew.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import postprocess
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.pipeline import Workspace, load_queries, write_report


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0).astype(np.float32)


def residuals(pred_boxes, pred_labels, gt_boxes, gt_labels, match_iou=0.3):
    """Scale-invariant residuals for each GT box's best same-class prediction.

    A loose 0.3 match threshold is used on purpose: the question is how well a
    *found* object is localised, so requiring 0.5 would discard exactly the
    badly-localised cases under investigation.
    """
    out = []
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return out
    ious = iou_matrix(gt_boxes, pred_boxes)
    same_class = gt_labels[:, None] == pred_labels[None, :]
    ious = np.where(same_class, ious, 0.0)

    for gi in range(len(gt_boxes)):
        pi = int(ious[gi].argmax())
        best = float(ious[gi, pi])
        if best < match_iou:
            continue
        gx1, gy1, gx2, gy2 = gt_boxes[gi]
        px1, py1, px2, py2 = pred_boxes[pi]
        gw, gh = max(gx2 - gx1, 1e-6), max(gy2 - gy1, 1e-6)
        pw, ph = max(px2 - px1, 1e-6), max(py2 - py1, 1e-6)
        out.append({
            "iou": best,
            "dx": ((px1 + px2) / 2 - (gx1 + gx2) / 2) / gw,
            "dy": ((py1 + py2) / 2 - (gy1 + gy2) / 2) / gh,
            "log_w": float(np.log(pw / gw)),
            "log_h": float(np.log(ph / gh)),
            "label": int(gt_labels[gi]),
        })
    return out


def collect(engine_name, size, frames, ws, num_classes, conf, max_det):
    runner = TRTRunner(ws.engine(engine_name))
    pre = GpuOwlv2Preprocessor(size, device="cuda")
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()

    all_res = []
    for frame in tqdm(frames, desc=f"owlv2@{size}", leave=False):
        image = load_image(frame.image_path)
        raw = runner.infer({"pixel_values": pre([image]), "query_embeds": query})
        torch.cuda.synchronize()
        arrays = {k: v.detach().float().cpu().numpy() for k, v in raw.items()}
        det = postprocess(arrays["pred_logits"], arrays["pred_boxes"], arrays["objectness"],
                          queries.owner, num_classes, image.size, conf, max_det)
        all_res.extend(residuals(det.boxes, det.labels, frame.boxes, frame.labels))
    return all_res


def summarize(res, label, classes):
    if not res:
        return {"label": label, "matches": 0}
    arr = {k: np.array([r[k] for r in res], dtype=np.float64)
           for k in ("iou", "dx", "dy", "log_w", "log_h")}
    summary = {
        "label": label,
        "matches": len(res),
        "mean_iou": float(arr["iou"].mean()),
        "median_iou": float(np.median(arr["iou"])),
        "frac_iou_ge_50": float((arr["iou"] >= 0.5).mean()),
        "frac_iou_ge_75": float((arr["iou"] >= 0.75).mean()),
    }
    for key in ("dx", "dy", "log_w", "log_h"):
        summary[f"{key}_bias"] = float(arr[key].mean())
        summary[f"{key}_jitter"] = float(arr[key].std())

    per_class = defaultdict(list)
    for r in res:
        per_class[r["label"]].append(r)
    summary["per_class"] = {
        classes[k]: {
            "n": len(v),
            "mean_iou": float(np.mean([x["iou"] for x in v])),
            "log_w_bias": float(np.mean([x["log_w"] for x in v])),
            "log_h_bias": float(np.mean([x["log_h"] for x in v])),
        }
        for k, v in sorted(per_class.items(), key=lambda kv: -len(kv[1]))[:12]
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--max-frames", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--max-detections", type=int, default=100)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    ag = ActionGenome(root=Path(args.ag_root), split="test").load()
    available = [f for f in ag.frames if f.image_path]
    step = max(1, len(available) // args.max_frames)
    frames = available[::step][: args.max_frames]
    print(f"{len(frames)} frames, {sum(len(f.labels) for f in frames)} GT boxes\n")

    summaries = []
    for engine, size in (("base_fp16.plan", 960), ("base768_fp16.plan", 768),
                         ("base640_fp16.plan", 640)):
        res = collect(engine, size, frames, ws, len(ag.classes),
                      args.conf, args.max_detections)
        summaries.append(summarize(res, f"owlv2@{size}", ag.classes))

    write_report({"frames": len(frames), "summaries": summaries},
                 ws.result("box_error_diagnostic.json"))

    h = (f"{'variant':<14}{'matches':>9}{'meanIoU':>9}{'IoU>=.75':>10}"
         f"{'dx bias':>9}{'dx jit':>8}{'dy bias':>9}{'dy jit':>8}"
         f"{'logw bias':>11}{'logw jit':>10}{'logh bias':>11}{'logh jit':>10}")
    print(h + "\n" + "-" * len(h))
    for s in summaries:
        print(f"{s['label']:<14}{s['matches']:>9}{s['mean_iou']:>9.4f}"
              f"{s['frac_iou_ge_50'] and s['frac_iou_ge_75']:>10.4f}"
              f"{s['dx_bias']:>9.4f}{s['dx_jitter']:>8.4f}"
              f"{s['dy_bias']:>9.4f}{s['dy_jitter']:>8.4f}"
              f"{s['log_w_bias']:>11.4f}{s['log_w_jitter']:>10.4f}"
              f"{s['log_h_bias']:>11.4f}{s['log_h_jitter']:>10.4f}")

    print("\nInterpretation: bias shifting with resolution -> a calibration fixes it.")
    print("Jitter growing while bias stays flat -> only a learned refinement head helps.")


if __name__ == "__main__":
    main()
