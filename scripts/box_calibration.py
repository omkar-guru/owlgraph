"""Correct the systematic box inflation OWLv2 shows at reduced resolution.

The diagnostic found that low-resolution boxes are not randomly scattered - they
are consistently **too large**: log-size bias rises from ~0.00 at 960px to +0.149
(width) and +0.108 (height) at 640px, while jitter grows only slightly. A uniform
inflation pushes otherwise-correct boxes just past strict IoU thresholds, which
is why mAP@50 held up while mAP@[.5:.95] collapsed.

That shape of error is correctable in closed form. This fits a per-class affine
correction - a size scale and a centre shift - and applies it at postprocessing
for no measurable inference cost.

**Constants are fitted on the train split and applied to test.** Fitting them on
the evaluation split would be leakage: the correction would be tuned on the very
boxes it is then scored against.
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
from sggpipeline.detect.owlv2 import Detections, postprocess
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.evaluation.detection_eval import evaluate_detections
from sggpipeline.pipeline import Workspace, load_queries, write_report

from box_error_diagnostic import residuals


def sample_frames(ag_root, split, count):
    ag = ActionGenome(root=Path(ag_root), split=split).load()
    available = [f for f in ag.frames if f.image_path]
    step = max(1, len(available) // count)
    return available[::step][:count], ag.classes


def predict(runner, pre, query, queries, frames, num_classes, conf, max_det):
    """Raw (uncalibrated) detections plus the frame they came from."""
    out = []
    for frame in tqdm(frames, leave=False, desc="predict"):
        image = load_image(frame.image_path)
        frame.width, frame.height = image.size
        raw = runner.infer({"pixel_values": pre([image]), "query_embeds": query})
        torch.cuda.synchronize()
        a = {k: v.detach().float().cpu().numpy() for k, v in raw.items()}
        out.append(postprocess(a["pred_logits"], a["pred_boxes"], a["objectness"],
                               queries.owner, num_classes, image.size, conf, max_det))
    return out


def fit_calibration(dets, frames, num_classes, min_samples=15):
    """Per-class centre shift and log-size bias, with a global fallback.

    Classes with few matches fall back to the global constants: a bias estimated
    from three boxes is noise, and applying it would make those classes worse.
    """
    per_class = defaultdict(list)
    for det, frame in zip(dets, frames, strict=True):
        for r in residuals(det.boxes, det.labels, frame.boxes, frame.labels):
            per_class[r["label"]].append(r)

    every = [r for rs in per_class.values() for r in rs]
    if not every:
        raise RuntimeError("No matched boxes; cannot fit a calibration.")

    def constants(rows):
        return {
            "dx": float(np.mean([r["dx"] for r in rows])),
            "dy": float(np.mean([r["dy"] for r in rows])),
            "log_w": float(np.mean([r["log_w"] for r in rows])),
            "log_h": float(np.mean([r["log_h"] for r in rows])),
            "n": len(rows),
        }

    calibration = {"global": constants(every), "per_class": {}}
    for label, rows in per_class.items():
        if len(rows) >= min_samples:
            calibration["per_class"][int(label)] = constants(rows)
    return calibration


def apply_calibration(det, calibration, image_size):
    """Undo the measured bias: shrink by exp(-log bias), shift by -centre bias."""
    if len(det.boxes) == 0:
        return det
    boxes = det.boxes.astype(np.float64).copy()
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)

    g = calibration["global"]
    for i, label in enumerate(det.labels):
        c = calibration["per_class"].get(int(label), g)
        # Residuals were measured relative to GT size; the prediction's own size
        # is the only estimate available at inference, so correct with it.
        new_w = w[i] * np.exp(-c["log_w"])
        new_h = h[i] * np.exp(-c["log_h"])
        cx[i] -= c["dx"] * new_w
        cy[i] -= c["dy"] * new_h
        w[i], h[i] = new_w, new_h

    width, height = image_size
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    out[:, 0::2] = out[:, 0::2].clip(0, width)
    out[:, 1::2] = out[:, 1::2].clip(0, height)
    return Detections(out.astype(np.float32), det.scores, det.labels, det.objectness)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--fit-frames", type=int, default=600)
    ap.add_argument("--eval-frames", type=int, default=1200)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--max-detections", type=int, default=100)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()

    fit_frames, classes = sample_frames(args.ag_root, "train", args.fit_frames)
    eval_frames, _ = sample_frames(args.ag_root, "test", args.eval_frames)
    print(f"fit on {len(fit_frames)} TRAIN frames; evaluate on {len(eval_frames)} TEST frames\n")

    rows = []
    for engine, size in (("base640_fp16.plan", 640), ("base768_fp16.plan", 768),
                         ("base_fp16.plan", 960)):
        runner = TRTRunner(ws.engine(engine))
        pre = GpuOwlv2Preprocessor(size, device="cuda")

        fit_dets = predict(runner, pre, query, queries, fit_frames, len(classes),
                           args.conf, args.max_detections)
        calibration = fit_calibration(fit_dets, fit_frames, len(classes))
        g = calibration["global"]
        print(f"owlv2@{size} fitted bias: log_w={g['log_w']:+.4f} log_h={g['log_h']:+.4f} "
              f"(w x{np.exp(-g['log_w']):.3f}, h x{np.exp(-g['log_h']):.3f}) "
              f"from {g['n']} matches, {len(calibration['per_class'])} per-class")

        eval_dets = predict(runner, pre, query, queries, eval_frames, len(classes),
                            args.conf, args.max_detections)
        before = evaluate_detections(eval_frames, eval_dets, classes)
        corrected = [apply_calibration(d, calibration, (f.width, f.height))
                     for d, f in zip(eval_dets, eval_frames, strict=True)]
        after = evaluate_detections(eval_frames, corrected, classes)

        rows.append({"image_size": size, "calibration": calibration,
                     "before": {k: before[k] for k in ("mAP", "mAP_50", "mAP_75", "AR_100")},
                     "after": {k: after[k] for k in ("mAP", "mAP_50", "mAP_75", "AR_100")}})
        print(f"  mAP    {before['mAP']:.4f} -> {after['mAP']:.4f} "
              f"({100*(after['mAP']-before['mAP'])/max(before['mAP'],1e-9):+.1f}%)")
        print(f"  mAP@75 {before['mAP_75']:.4f} -> {after['mAP_75']:.4f}")
        print(f"  AR@100 {before['AR_100']:.4f} -> {after['AR_100']:.4f}\n")

    write_report({"fit_split": "train", "eval_split": "test", "results": rows},
                 ws.result("box_calibration.json"))

    h = f"{'res':>5}{'mAP before':>12}{'mAP after':>11}{'delta %':>10}{'mAP75 before':>14}{'mAP75 after':>13}"
    print(h + "\n" + "-" * len(h))
    for r in rows:
        b, a = r["before"], r["after"]
        print(f"{r['image_size']:>5}{b['mAP']:>12.4f}{a['mAP']:>11.4f}"
              f"{100*(a['mAP']-b['mAP'])/max(b['mAP'],1e-9):>9.1f}%"
              f"{b['mAP_75']:>14.4f}{a['mAP_75']:>13.4f}")


if __name__ == "__main__":
    main()
