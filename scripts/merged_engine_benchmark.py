"""TensorRT benchmark: prev50 merged engine vs unmerged engines at 960/768/640.

All four are strongly-typed fp16 TensorRT engines. Timed at three scopes, all
with CUDA events on the same GPU in the same run:

``engine``      engine execution only
``index``       building the merge plan from the previous frame's objectness
                (merged engine only; runs outside the engine every frame)
``per_frame``   preprocess + index + engine: what a stream pays per frame

Accuracy uses the protocol of ``cascade_benchmark.py`` (600 test frames, one per
video, 36 classes, uncalibrated), with the prior taken from the unmerged 960
engine on the previous frame. The streaming test showed a self-fed prior
performs identically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.merged_export import MergeIndexer
from sggpipeline.detect.owlv2 import postprocess
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.evaluation.detection_eval import evaluate_detections
from sggpipeline.pipeline import Workspace, load_queries, write_report

from early_objectness_probe import decode_positions

ENGINES = {
    "960 unmerged": ("base_fp16.plan", 960, False),
    "960 merged50": ("base_merged50_fp16.plan", 960, True),
    "768 unmerged": ("base768_fp16.plan", 768, False),
    "640 unmerged": ("base640_fp16.plan", 640, False),
}


def cuda_ms(fn, iters=200, warmup=30) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.asarray(times)
    return {"median": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()
    runners = {k: TRTRunner(ws.engine(e)) for k, (e, _, _) in ENGINES.items()}
    pres = {k: GpuOwlv2Preprocessor(s, device="cuda") for k, (_, s, _) in ENGINES.items()}
    indexer = MergeIndexer(60, 0.5)
    prior_runner, prior_pre = runners["960 unmerged"], pres["960 unmerged"]

    def run(name, px, plan=None):
        feeds = {"pixel_values": px, "query_embeds": query}
        if plan is not None:
            feeds.update(unmerged_idx=plan[0], member_patches=plan[1], assign=plan[2])
        return runners[name].infer(feeds)

    ag = ActionGenome(root=Path(args.ag_root), split="test").load()
    classes = ag.classes
    available = [f for f in ag.frames if f.image_path and f.frame_index - 2 >= 0]
    step = max(1, len(available) // (args.frames * 2))
    frames, seen = [], set()
    for f in available[::step]:
        if f.video_id not in seen:
            seen.add(f.video_id)
            frames.append(f)
        if len(frames) == args.frames:
            break
    videos_dir = Path(args.ag_root) / "Charades_v1_480"

    dets = {k: [] for k in ENGINES}
    for frame in tqdm(frames, desc="frames"):
        pos = frame.frame_index - 2
        prior_img = decode_positions(videos_dir / frame.video_id, {pos})[pos]
        image = load_image(frame.image_path)
        frame.width, frame.height = image.size
        prior = run("960 unmerged", prior_pre([prior_img]))["objectness"][0].float()
        plan = indexer(prior)
        for name, (_, _, merged) in ENGINES.items():
            out = run(name, pres[name]([image]), plan if merged else None)
            torch.cuda.synchronize()
            a = {k: v.detach().float().cpu().numpy() for k, v in out.items()}
            dets[name].append(postprocess(a["pred_logits"], a["pred_boxes"], a["objectness"],
                                          queries.owner, len(classes), image.size, 0.05, 100))

    # Timing on one real frame with its real prior, as in a stream.
    frame = frames[0]
    pos = frame.frame_index - 2
    image = load_image(frame.image_path)
    prior_img = decode_positions(videos_dir / frame.video_id, {pos})[pos]
    prior = run("960 unmerged", prior_pre([prior_img]))["objectness"][0].float()
    plan = indexer(prior)
    index_ms = cuda_ms(lambda: indexer(prior))

    rows = []
    for name, (_, size, merged) in ENGINES.items():
        px = pres[name]([image])
        p = plan if merged else None
        engine_ms = cuda_ms(lambda: run(name, px, p))
        if merged:
            per_frame = cuda_ms(lambda: run(name, pres[name]([image]), indexer(prior)))
        else:
            per_frame = cuda_ms(lambda: run(name, pres[name]([image])))
        m = evaluate_detections(frames, dets[name], classes)
        rows.append({
            "config": name, "image_size": size,
            "engine_ms": engine_ms, "index_ms": index_ms if merged else None,
            "per_frame_ms": per_frame,
            **{k: m[k] for k in ("mAP", "mAP_50", "mAP_75", "mAP_small", "AR_100")},
        })

    write_report({"frames": len(frames), "protocol": "36 classes, uncalibrated, prior from "
                  "960 unmerged engine on the previous frame", "results": rows},
                 ws.result("merged_engine_benchmark.json"))

    print(f"\n{len(frames)} frames, one per video, 36 classes, uncalibrated; TensorRT fp16\n")
    h = (f"{'config':<14}{'engine':>9}{'p95':>7}{'index':>7}{'frame':>8}{'fps':>7}"
         f"{'mAP':>8}{'mAP75':>8}{'APsmall':>8}{'AR100':>8}")
    print(h + "\n" + "-" * len(h))
    for r in rows:
        idx = f"{r['index_ms']['median']:.2f}" if r["index_ms"] else "-"
        print(f"{r['config']:<14}{r['engine_ms']['median']:>9.2f}{r['engine_ms']['p95']:>7.2f}"
              f"{idx:>7}{r['per_frame_ms']['median']:>8.2f}"
              f"{1000 / r['per_frame_ms']['median']:>7.1f}{r['mAP']:>8.4f}{r['mAP_75']:>8.4f}"
              f"{r['mAP_small']:>8.4f}{r['AR_100']:>8.4f}")


if __name__ == "__main__":
    main()
