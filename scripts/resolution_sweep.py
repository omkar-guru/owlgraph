"""Accuracy/FPS trade-off across input resolutions, scored on Action Genome.

Speed alone cannot choose a resolution: the point of dropping to 768 or 640 is
only worth taking if detection quality survives it.  Every resolution is scored
on the *same* frames with the same postprocessing so the comparison isolates
input size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import load_owlv2, postprocess
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.evaluation.detection_eval import evaluate_detections
from sggpipeline.pipeline import Workspace, load_queries, write_report


def evaluate_engine(engine_name, image_size, frames, queries, num_classes,
                    score_threshold, max_detections, artifacts) -> dict:
    ws = Workspace(Path(artifacts))
    runner = TRTRunner(ws.engine(engine_name))
    pre = GpuOwlv2Preprocessor(image_size, device="cuda")
    query = torch.from_numpy(queries.embeds).cuda()

    detections = []
    for frame in tqdm(frames, desc=f"{engine_name}@{image_size}", leave=False):
        image = load_image(frame.image_path)
        frame.width, frame.height = image.size
        out = runner.infer({"pixel_values": pre([image]), "query_embeds": query})
        torch.cuda.synchronize()
        arrays = {k: v.detach().float().cpu().numpy() for k, v in out.items()}
        detections.append(
            postprocess(
                arrays["pred_logits"], arrays["pred_boxes"], arrays["objectness"],
                prompt_owner=queries.owner, num_classes=num_classes,
                image_size=image.size, score_threshold=score_threshold,
                max_detections=max_detections,
            )
        )
    return evaluate_detections(frames, detections, tuple(range(num_classes)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--score-threshold", type=float, default=0.05)
    ap.add_argument("--max-detections", type=int, default=100)
    ap.add_argument("--configs", nargs="+",
                    default=["base_fp16.plan:960", "base768_fp16.plan:768", "base640_fp16.plan:640"])
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    ag = ActionGenome(root=Path(args.ag_root), split=args.split).load()
    available = [f for f in ag.frames if f.image_path]
    # Spread the subset across videos rather than taking a contiguous block, so
    # a capped run is not dominated by a handful of scenes.
    step = max(1, len(available) // args.max_frames)
    frames = available[::step][: args.max_frames]
    print(f"scoring {len(frames)} frames from {len({f.video_id for f in frames})} videos "
          f"({len(available)} available)")

    queries = load_queries(ws, "base")
    rows = []
    for config in args.configs:
        name, _, size = config.partition(":")
        metrics = evaluate_engine(name, int(size), frames, queries, len(ag.classes),
                                  args.score_threshold, args.max_detections, args.artifacts)
        rows.append({"engine": name, "image_size": int(size), **{
            k: v for k, v in metrics.items() if k != "per_class_AP"}})
        print(f"  {name}@{size}: mAP={metrics['mAP']:.4f} mAP50={metrics['mAP_50']:.4f} "
              f"AR100={metrics['AR_100']:.4f} dets={metrics['num_detections']}")

    write_report({"frames": len(frames), "split": args.split, "results": rows},
                 ws.result("resolution_sweep.json"))
    print("\n" + f"{'engine':<24}{'res':>6}{'mAP':>9}{'mAP50':>9}{'AR100':>9}")
    for r in rows:
        print(f"{r['engine']:<24}{r['image_size']:>6}{r['mAP']:>9.4f}{r['mAP_50']:>9.4f}{r['AR_100']:>9.4f}")


if __name__ == "__main__":
    main()
