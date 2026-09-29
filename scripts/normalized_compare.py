"""Normalized YOLO26 vs OWLv2 comparison: same runtime, same measurement scope.

Both models run as strongly-typed fp16 TensorRT engines with GPU preprocessing
and GPU postprocessing, and both are timed at two scopes:

``engine_ms``
    Engine execution alone. Compares the networks.
``end_to_end_ms``
    Preprocess + engine + postprocess. Compares what a stream actually costs.

Reporting only the first flatters whichever model has the heavier
postprocessing - and YOLO26 exports with ``end2end=False``, so it pays for NMS
while OWLv2 does not. Reporting only the second buries the network difference
under shared overhead. Both are needed.

Accuracy is scored on the 14 AG classes reachable from COCO, exactly as in
``yolo_vs_owlv2.py``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import postprocess as owlv2_postprocess
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.detect.yolo import YoloPreprocessor, decode as yolo_decode, export_and_build
from sggpipeline.evaluation.detection_eval import evaluate_detections
from sggpipeline.pipeline import Workspace, load_queries, write_report

from yolo_vs_owlv2 import COCO_TO_AG, SHARED_AG_CLASSES, Detections, restrict_ground_truth


def _stats(times_ms: list[float]) -> dict:
    arr = np.asarray(times_ms)
    return {"median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95))}


def _time_gpu(fn, iters=100, warmup=20) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return _stats(times)


def eval_owlv2(engine_name, size, frames, ws, ag_classes, conf, max_det):
    runner = TRTRunner(ws.engine(engine_name))
    pre = GpuOwlv2Preprocessor(size, device="cuda")
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()

    shared = {n: i for i, n in enumerate(SHARED_AG_CLASSES)}
    remap = {o: shared[n] for o, n in enumerate(ag_classes) if n in shared}

    dets = []
    for frame in tqdm(frames, desc=f"owlv2@{size}", leave=False):
        image = load_image(frame.image_path)
        frame.width, frame.height = image.size
        raw = runner.infer({"pixel_values": pre([image]), "query_embeds": query})
        torch.cuda.synchronize()
        arrays = {k: v.detach().float().cpu().numpy() for k, v in raw.items()}
        d = owlv2_postprocess(arrays["pred_logits"], arrays["pred_boxes"], arrays["objectness"],
                              queries.owner, len(ag_classes), image.size, conf, max_det)
        mask = np.array([int(l) in remap for l in d.labels], dtype=bool)
        dets.append(Detections(d.boxes[mask], d.scores[mask],
                               [remap[int(l)] for l in d.labels[mask]]) if mask.any()
                    else Detections(np.zeros((0, 4)), [], []))

    sample = load_image(frames[0].image_path)
    px = pre([sample])
    timing_engine = _time_gpu(lambda: runner.infer({"pixel_values": px, "query_embeds": query}))

    def full():
        p = pre([sample])
        o = runner.infer({"pixel_values": p, "query_embeds": query})
        a = {k: v.detach().float().cpu().numpy() for k, v in o.items()}
        owlv2_postprocess(a["pred_logits"], a["pred_boxes"], a["objectness"],
                          queries.owner, len(ag_classes), sample.size, conf, max_det)

    return dets, timing_engine, _time_gpu(full, iters=50, warmup=10)


def eval_yolo(weights, size, frames, ws, conf, iou, max_det):
    engine_path = export_and_build(weights, size, ws)
    runner = TRTRunner(engine_path)
    pre = YoloPreprocessor(size, device="cuda",
                           dtype=runner._torch_dtype("images"))
    out_name = runner.output_names[0]

    shared = {n: i for i, n in enumerate(SHARED_AG_CLASSES)}
    from ultralytics import YOLO
    names = YOLO(weights).names
    remap = {i: shared[COCO_TO_AG[n]] for i, n in names.items() if n in COCO_TO_AG}

    dets = []
    for frame in tqdm(frames, desc=f"{Path(weights).stem}@{size}", leave=False):
        image = load_image(frame.image_path)
        frame.width, frame.height = image.size
        batch, geoms = pre([image])
        raw = runner.infer({"images": batch})[out_name]
        d = yolo_decode(raw, geoms[0], image.size, conf, iou, max_det)
        mask = np.array([int(l) in remap for l in d.labels], dtype=bool)
        dets.append(Detections(d.boxes[mask], d.scores[mask],
                               [remap[int(l)] for l in d.labels[mask]]) if mask.any()
                    else Detections(np.zeros((0, 4)), [], []))

    sample = load_image(frames[0].image_path)
    batch, geoms = pre([sample])
    timing_engine = _time_gpu(lambda: runner.infer({"images": batch}))

    def full():
        b, g = pre([sample])
        r = runner.infer({"images": b})[out_name]
        yolo_decode(r, g[0], sample.size, conf, iou, max_det)

    return dets, timing_engine, _time_gpu(full, iters=50, warmup=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-detections", type=int, default=100)
    ap.add_argument("--yolo", nargs="+", default=["yolo26s.pt", "yolo26m.pt", "yolo26x.pt"])
    ap.add_argument("--owlv2", nargs="+", default=["base640_fp16.plan:640", "base_fp16.plan:960"])
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    ag = ActionGenome(root=Path(args.ag_root), split="test").load()
    available = [f for f in ag.frames if f.image_path]
    step = max(1, len(available) // args.max_frames)
    frames = restrict_ground_truth(available[::step][: args.max_frames], ag.classes)
    print(f"{len(frames)} frames, {sum(len(f.labels) for f in frames)} GT boxes, "
          f"{len(SHARED_AG_CLASSES)} shared classes\n")

    rows = []
    for spec in args.owlv2:
        name, _, size = spec.partition(":")
        dets, te, tf = eval_owlv2(name, int(size), frames, ws, ag.classes,
                                  args.conf, args.max_detections)
        m = evaluate_detections(frames, dets, SHARED_AG_CLASSES)
        rows.append({"model": "OWLv2-B/16", "imgsz": int(size), "runtime": "TRT fp16",
                     "mAP": m["mAP"], "mAP_50": m["mAP_50"], "AR_100": m["AR_100"],
                     "engine_ms": te["median_ms"], "end_to_end_ms": tf["median_ms"]})
        print(f"  OWLv2@{size}: mAP={m['mAP']:.4f} engine={te['median_ms']:.2f}ms "
              f"e2e={tf['median_ms']:.2f}ms")

    for weights in args.yolo:
        dets, te, tf = eval_yolo(weights, 640, frames, ws, args.conf, args.iou,
                                 args.max_detections)
        m = evaluate_detections(frames, dets, SHARED_AG_CLASSES)
        rows.append({"model": Path(weights).stem, "imgsz": 640, "runtime": "TRT fp16",
                     "mAP": m["mAP"], "mAP_50": m["mAP_50"], "AR_100": m["AR_100"],
                     "engine_ms": te["median_ms"], "end_to_end_ms": tf["median_ms"]})
        print(f"  {Path(weights).stem}: mAP={m['mAP']:.4f} engine={te['median_ms']:.2f}ms "
              f"e2e={tf['median_ms']:.2f}ms")

    write_report({"frames": len(frames), "shared_classes": list(SHARED_AG_CLASSES),
                  "results": rows}, ws.result("normalized_compare.json"))

    h = (f"{'model':<14}{'res':>5}{'mAP':>8}{'mAP50':>8}{'AR100':>8}"
         f"{'engine':>9}{'e2e ms':>9}{'e2e fps':>9}")
    print("\n" + h + "\n" + "-" * len(h))
    for r in rows:
        print(f"{r['model']:<14}{r['imgsz']:>5}{r['mAP']:>8.4f}{r['mAP_50']:>8.4f}"
              f"{r['AR_100']:>8.4f}{r['engine_ms']:>9.2f}{r['end_to_end_ms']:>9.2f}"
              f"{1000/r['end_to_end_ms']:>9.1f}")


if __name__ == "__main__":
    main()
