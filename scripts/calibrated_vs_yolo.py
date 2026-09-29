"""Calibrated OWLv2 scored under the same protocol as normalized_compare.py.

The affine box calibration was first evaluated on all 36 AG classes, while the
YOLO26 comparison uses only the 14 classes reachable from COCO. This re-scores
calibrated OWLv2 on that shared set, with the same frames and ground truth, so
it can sit in the same table as YOLO26. Calibration constants are fitted on the
train split, as before.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.evaluation.detection_eval import evaluate_detections
from sggpipeline.pipeline import Workspace, load_queries, write_report

from box_calibration import apply_calibration, fit_calibration, predict, sample_frames
from yolo_vs_owlv2 import SHARED_AG_CLASSES, Detections


def to_shared(dets, frames, classes):
    """Remap to the shared class set, dropping frames with no shared GT."""
    shared = {n: i for i, n in enumerate(SHARED_AG_CLASSES)}
    remap = {o: shared[n] for o, n in enumerate(classes) if n in shared}
    out_frames, out_dets = [], []
    for det, frame in zip(dets, frames, strict=True):
        gt_mask = np.array([int(l) in remap for l in frame.labels], dtype=bool)
        if not gt_mask.any():
            continue
        frame.boxes = frame.boxes[gt_mask]
        frame.labels = np.array([remap[int(l)] for l in frame.labels[gt_mask]], dtype=np.int64)
        m = np.array([int(l) in remap for l in det.labels], dtype=bool)
        out_dets.append(Detections(det.boxes[m], det.scores[m],
                                   [remap[int(l)] for l in det.labels[m]]) if m.any()
                        else Detections(np.zeros((0, 4)), [], []))
        out_frames.append(frame)
    return out_frames, out_dets


def main() -> None:
    ws = Workspace(Path("artifacts"))
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()
    fit_frames, classes = sample_frames("acgdataset", "train", 600)

    rows = []
    for engine, size in (("base640_fp16.plan", 640), ("base768_fp16.plan", 768)):
        runner = TRTRunner(ws.engine(engine))
        pre = GpuOwlv2Preprocessor(size, device="cuda")
        calib = fit_calibration(
            predict(runner, pre, query, queries, fit_frames, len(classes), 0.05, 100),
            fit_frames, len(classes))

        eval_frames, _ = sample_frames("acgdataset", "test", 1200)
        raw = predict(runner, pre, query, queries, eval_frames, len(classes), 0.05, 100)
        sizes = [(f.width, f.height) for f in eval_frames]
        cal = [apply_calibration(d, calib, s) for d, s in zip(raw, sizes, strict=True)]

        # Cost of the correction itself, on real detection sets.
        t0 = time.perf_counter()
        for d, s in zip(raw[:300], sizes[:300]):
            apply_calibration(d, calib, s)
        calib_ms = (time.perf_counter() - t0) * 1000 / 300

        # to_shared mutates frame GT, so each scoring pass gets a fresh copy.
        frames_u, dets_u = to_shared(raw, eval_frames, classes)
        eval_frames, _ = sample_frames("acgdataset", "test", 1200)
        frames_c, dets_c = to_shared(cal, eval_frames, classes)
        before = evaluate_detections(frames_u, dets_u, SHARED_AG_CLASSES)
        after = evaluate_detections(frames_c, dets_c, SHARED_AG_CLASSES)
        rows.append({"image_size": size, "frames": len(frames_c), "calib_ms": calib_ms,
                     "before": {k: before[k] for k in ("mAP", "mAP_50", "mAP_75", "AR_100")},
                     "after": {k: after[k] for k in ("mAP", "mAP_50", "mAP_75", "AR_100")}})
        print(f"owlv2@{size} ({len(frames_c)} frames): "
              f"mAP {before['mAP']:.4f}->{after['mAP']:.4f}  "
              f"mAP50 {before['mAP_50']:.4f}->{after['mAP_50']:.4f}  "
              f"AR100 {before['AR_100']:.4f}->{after['AR_100']:.4f}  "
              f"calib cost {calib_ms:.3f} ms/frame", flush=True)

    write_report({"protocol": "14 shared COCO/AG classes, fit=train, eval=test",
                  "results": rows}, ws.result("calibrated_vs_yolo.json"))


if __name__ == "__main__":
    main()
