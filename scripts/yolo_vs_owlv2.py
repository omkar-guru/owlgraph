"""YOLO26 vs OWLv2-B/16 on the Action Genome classes both models can see.

A direct comparison is unfair in both directions and has to be constrained:

* YOLO26 is closed-vocabulary (COCO's 80 classes). Roughly half of AG's
  vocabulary - doorway, doorknob, broom, vacuum, groceries, blanket - has no
  COCO equivalent, so YOLO structurally cannot score on it.
* OWLv2 is open-vocabulary and does zero-shot text matching, while YOLO is
  trained directly on the COCO classes it shares with AG.

So both are scored **only on the 14 AG classes reachable from COCO**, with
ground truth restricted to the same set. Each model keeps its own natural
vocabulary at inference (YOLO predicts all 80 COCO classes, OWLv2 all 36 AG
prompts); predictions outside the shared set are discarded rather than the
vocabularies being trimmed, so neither model is handed an easier problem than
it would face in deployment.

This answers one question only: *what does open-vocabulary flexibility cost
against a specialist, on classes the specialist was trained for?* It says
nothing about the other 22 AG classes, where YOLO scores zero by construction.
"""

from __future__ import annotations

import argparse
import json
import time
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
from sggpipeline.evaluation.detection_eval import evaluate_detections
from sggpipeline.pipeline import Workspace, load_queries, write_report

# COCO surface name -> AG class name. Only unambiguous mappings are included.
# 'food' is deliberately excluded: AG separates 'food' from 'sandwich' while COCO
# has ten specific foodstuffs, and collapsing them would invent ground truth.
COCO_TO_AG: dict[str, str] = {
    "person": "person",
    "backpack": "bag",
    "handbag": "bag",
    "suitcase": "bag",
    "bed": "bed",
    "book": "book",
    "chair": "chair",
    "bottle": "cup/glass/bottle",
    "wine glass": "cup/glass/bottle",
    "cup": "cup/glass/bottle",
    "bowl": "dish",
    "laptop": "laptop",
    "cell phone": "phone/camera",
    "refrigerator": "refrigerator",
    "sandwich": "sandwich",
    "couch": "sofa/couch",
    "dining table": "table",
    "tv": "television",
}

SHARED_AG_CLASSES: tuple[str, ...] = tuple(sorted(set(COCO_TO_AG.values())))


class Detections:
    """Minimal detection record matching what evaluate_detections consumes."""

    def __init__(self, boxes, scores, labels):
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)


def restrict_ground_truth(frames, ag_classes: tuple[str, ...]):
    """Keep only boxes whose class is in the shared set, reindexed to it."""
    shared_index = {name: i for i, name in enumerate(SHARED_AG_CLASSES)}
    old_to_new = {
        old: shared_index[name]
        for old, name in enumerate(ag_classes)
        if name in shared_index
    }

    kept = []
    for frame in frames:
        mask = np.array([int(l) in old_to_new for l in frame.labels], dtype=bool)
        if not mask.any():
            continue
        frame.boxes = frame.boxes[mask]
        frame.labels = np.array(
            [old_to_new[int(l)] for l in frame.labels[mask]], dtype=np.int64
        )
        kept.append(frame)
    return kept


def run_owlv2(engine_name, image_size, frames, ws, ag_classes, threshold, max_det):
    """OWLv2 predicts its full 36-class AG vocabulary; we keep the shared ones."""
    runner = TRTRunner(ws.engine(engine_name))
    pre = GpuOwlv2Preprocessor(image_size, device="cuda")
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()

    shared_index = {name: i for i, name in enumerate(SHARED_AG_CLASSES)}
    old_to_new = {
        old: shared_index[name]
        for old, name in enumerate(ag_classes)
        if name in shared_index
    }

    out_dets = []
    for frame in tqdm(frames, desc=f"owlv2@{image_size}", leave=False):
        image = load_image(frame.image_path)
        frame.width, frame.height = image.size
        raw = runner.infer({"pixel_values": pre([image]), "query_embeds": query})
        torch.cuda.synchronize()
        arrays = {k: v.detach().float().cpu().numpy() for k, v in raw.items()}
        det = postprocess(
            arrays["pred_logits"], arrays["pred_boxes"], arrays["objectness"],
            prompt_owner=queries.owner, num_classes=len(ag_classes),
            image_size=image.size, score_threshold=threshold, max_detections=max_det,
        )
        mask = np.array([int(l) in old_to_new for l in det.labels], dtype=bool)
        if mask.any():
            out_dets.append(Detections(
                det.boxes[mask], det.scores[mask],
                [old_to_new[int(l)] for l in det.labels[mask]],
            ))
        else:
            out_dets.append(Detections(np.zeros((0, 4)), [], []))
    return out_dets


def run_yolo(weights, image_size, frames, threshold, max_det, device="cuda"):
    """YOLO26 predicts all 80 COCO classes; we map and keep the shared ones."""
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names  # {index: coco name}
    shared_index = {name: i for i, name in enumerate(SHARED_AG_CLASSES)}
    coco_to_new = {
        idx: shared_index[COCO_TO_AG[name]]
        for idx, name in names.items()
        if name in COCO_TO_AG
    }

    out_dets = []
    for frame in tqdm(frames, desc=f"{Path(weights).stem}@{image_size}", leave=False):
        result = model.predict(
            str(frame.image_path), imgsz=image_size, conf=threshold,
            max_det=max_det, device=device, half=True, verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.cls.numel() == 0:
            out_dets.append(Detections(np.zeros((0, 4)), [], []))
            continue
        cls = boxes.cls.cpu().numpy().astype(int)
        mask = np.array([c in coco_to_new for c in cls], dtype=bool)
        if not mask.any():
            out_dets.append(Detections(np.zeros((0, 4)), [], []))
            continue
        out_dets.append(Detections(
            boxes.xyxy.cpu().numpy()[mask],
            boxes.conf.cpu().numpy()[mask],
            [coco_to_new[c] for c in cls[mask]],
        ))
    return out_dets


def bench_yolo(weights, image_size, sample_image, iters=100, warmup=20, device="cuda"):
    """Per-frame latency for YOLO, timed the same way as the TensorRT engines."""
    from ultralytics import YOLO

    model = YOLO(weights)
    for _ in range(warmup):
        model.predict(sample_image, imgsz=image_size, device=device, half=True, verbose=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(sample_image, imgsz=image_size, device=device, half=True, verbose=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(times)
    return {"median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95))}


def bench_owlv2(engine_name, image_size, ws, iters=100, warmup=20):
    runner = TRTRunner(ws.engine(engine_name))
    queries = load_queries(ws, "base")
    feed = {
        "pixel_values": torch.randn(1, 3, image_size, image_size, device="cuda",
                                    dtype=runner._torch_dtype("pixel_values")),
        "query_embeds": torch.from_numpy(queries.embeds).cuda().to(
            runner._torch_dtype("query_embeds")),
    }
    for _ in range(warmup):
        runner.infer(dict(feed))
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); runner.infer(dict(feed)); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.asarray(times)
    return {"median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--score-threshold", type=float, default=0.05)
    ap.add_argument("--max-detections", type=int, default=100)
    ap.add_argument("--yolo", nargs="+", default=["yolo26s.pt", "yolo26m.pt", "yolo26x.pt"])
    ap.add_argument("--owlv2", nargs="+", default=["base640_fp16.plan:640", "base_fp16.plan:960"])
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    ag = ActionGenome(root=Path(args.ag_root), split="test").load()
    available = [f for f in ag.frames if f.image_path]
    step = max(1, len(available) // args.max_frames)
    frames = available[::step][: args.max_frames]
    ag_classes = ag.classes

    frames = restrict_ground_truth(frames, ag_classes)
    gt_boxes = sum(len(f.labels) for f in frames)
    print(f"shared classes ({len(SHARED_AG_CLASSES)}): {', '.join(SHARED_AG_CLASSES)}")
    print(f"scoring {len(frames)} frames, {gt_boxes} ground-truth boxes in the shared set\n")

    sample = str(frames[0].image_path)
    rows = []

    for spec in args.owlv2:
        name, _, size = spec.partition(":")
        size = int(size)
        dets = run_owlv2(name, size, frames, ws, ag_classes,
                         args.score_threshold, args.max_detections)
        metrics = evaluate_detections(frames, dets, SHARED_AG_CLASSES)
        timing = bench_owlv2(name, size, ws)
        rows.append({"model": f"OWLv2-B/16 fp16 TRT", "imgsz": size,
                     **{k: v for k, v in metrics.items() if k != "per_class_AP"}, **timing,
                     "per_class_AP": metrics.get("per_class_AP", {})})
        print(f"  OWLv2@{size}: mAP={metrics['mAP']:.4f} mAP50={metrics['mAP_50']:.4f} "
              f"AR100={metrics['AR_100']:.4f} {timing['median_ms']:.2f}ms")

    for weights in args.yolo:
        dets = run_yolo(weights, args.imgsz, frames, args.score_threshold, args.max_detections)
        metrics = evaluate_detections(frames, dets, SHARED_AG_CLASSES)
        timing = bench_yolo(weights, args.imgsz, sample)
        rows.append({"model": f"{Path(weights).stem} fp16 PyTorch", "imgsz": args.imgsz,
                     **{k: v for k, v in metrics.items() if k != "per_class_AP"}, **timing,
                     "per_class_AP": metrics.get("per_class_AP", {})})
        print(f"  {Path(weights).stem}@{args.imgsz}: mAP={metrics['mAP']:.4f} "
              f"mAP50={metrics['mAP_50']:.4f} AR100={metrics['AR_100']:.4f} "
              f"{timing['median_ms']:.2f}ms")

    write_report({"frames": len(frames), "gt_boxes": gt_boxes,
                  "shared_classes": list(SHARED_AG_CLASSES), "results": rows},
                 ws.result("yolo_vs_owlv2.json"))

    header = f"{'model':<26}{'res':>6}{'mAP':>9}{'mAP50':>9}{'AR100':>9}{'ms':>9}{'fps':>8}"
    print("\n" + header + "\n" + "-" * len(header))
    for r in rows:
        print(f"{r['model']:<26}{r['imgsz']:>6}{r['mAP']:>9.4f}{r['mAP_50']:>9.4f}"
              f"{r['AR_100']:>9.4f}{r['median_ms']:>9.2f}{1000/r['median_ms']:>8.1f}")


if __name__ == "__main__":
    main()
