"""Full Action Genome test-split evaluation of the Stage 1 engines.

Earlier comparisons used 600-1,200 frames, one per video. This scores every test
keyframe that has a previous frame (~68k), so the final Stage 1 table does not
rest on a subsample.

Same protocol as ``merged_engine_benchmark.py``: 36 classes, uncalibrated,
TensorRT fp16, and the merged engine's prior taken from the unmerged 960 engine
on the immediately previous video frame. Each video is decoded once for all of
its priors.

Detections are saved per engine before scoring (pickle-free NPZ), so COCO
scoring can be rerun with ``--score-only`` without repeating inference.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.merged_export import MergeIndexer
from sggpipeline.detect.owlv2 import Detections, postprocess
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.evaluation.detection_eval import evaluate_detections
from sggpipeline.pipeline import Workspace, load_queries, write_report

from early_objectness_probe import decode_positions

ENGINES = {
    "960_unmerged": ("base_fp16.plan", 960, False),
    "960_merged50": ("base_merged50_np_fp16.plan", 960, True),
    "768_unmerged": ("base768_fp16.plan", 768, False),
    "640_unmerged": ("base640_fp16.plan", 640, False),
}


def save_detections(path: Path, frame_keys: list[str], dets: list[Detections]) -> None:
    offsets = np.cumsum([0] + [len(d.scores) for d in dets])
    np.savez_compressed(
        path, frame_keys=np.array(frame_keys), offsets=offsets,
        boxes=np.concatenate([d.boxes for d in dets]).astype(np.float32),
        scores=np.concatenate([d.scores for d in dets]).astype(np.float32),
        labels=np.concatenate([d.labels for d in dets]).astype(np.int64),
    )


def load_detections(path: Path) -> tuple[list[str], list[Detections]]:
    with np.load(path, allow_pickle=False) as z:
        keys, off = list(z["frame_keys"]), z["offsets"]
        boxes, scores, labels = z["boxes"], z["scores"], z["labels"]
    dets = [Detections(boxes[a:b], scores[a:b], labels[a:b], np.zeros(b - a, np.float32))
            for a, b in zip(off[:-1], off[1:], strict=True)]
    return keys, dets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--max-videos", type=int, default=None, help="for smoke tests")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    out_dir = ws.root / "full_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    ag = ActionGenome(root=Path(args.ag_root), split="test").load()
    classes = ag.classes
    by_video = defaultdict(list)
    for f in ag.frames:
        if f.image_path and f.frame_index - 2 >= 0:
            by_video[f.video_id].append(f)
    videos = sorted(by_video)[: args.max_videos]
    frames = [f for v in videos for f in by_video[v]]
    print(f"{len(frames)} frames from {len(videos)} videos", flush=True)

    if not args.score_only:
        queries = load_queries(ws, "base")
        query = torch.from_numpy(queries.embeds).cuda()
        runners = {k: TRTRunner(ws.engine(e)) for k, (e, _, _) in ENGINES.items()}
        pres = {k: GpuOwlv2Preprocessor(s, device="cuda") for k, (_, s, _) in ENGINES.items()}
        indexer = MergeIndexer(60, 0.5)
        videos_dir = Path(args.ag_root) / "Charades_v1_480"
        dets = {k: [] for k in ENGINES}

        def priors_for(video):
            return decode_positions(videos_dir / video, {f.frame_index - 2 for f in by_video[video]})

        t0 = time.perf_counter()
        # Decode priors a bounded window ahead on CPU threads while the GPU works.
        # Submitting every video at once would hold all ~68k prior frames in RAM.
        window = 16
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {v: pool.submit(priors_for, v) for v in videos[:window]}
            for i, video in enumerate(tqdm(videos, desc="videos", mininterval=30)):
                if i + window < len(videos):
                    ahead = videos[i + window]
                    futures[ahead] = pool.submit(priors_for, ahead)
                priors = futures.pop(video).result()
                for frame in by_video[video]:
                    image = load_image(frame.image_path)
                    frame.width, frame.height = image.size
                    prior_px = pres["960_unmerged"]([priors[frame.frame_index - 2]])
                    prior = runners["960_unmerged"].infer(
                        {"pixel_values": prior_px, "query_embeds": query})["objectness"][0].float()
                    plan = indexer(prior)
                    for name, (_, _, merged) in ENGINES.items():
                        feeds = {"pixel_values": pres[name]([image]), "query_embeds": query}
                        if merged:
                            feeds.update(unmerged_idx=plan[0], member_patches=plan[1], assign=plan[2])
                        out = runners[name].infer(feeds)
                        torch.cuda.synchronize()
                        a = {k: v.float().cpu().numpy() for k, v in out.items()}
                        dets[name].append(postprocess(a["pred_logits"], a["pred_boxes"],
                                                      a["objectness"], queries.owner,
                                                      len(classes), image.size, 0.05, 100))
        print(f"inference done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        keys = [f.frame_key for f in frames]
        for name in ENGINES:
            save_detections(out_dir / f"dets_{name}.npz", keys, dets[name])
    else:
        from PIL import Image

        for frame in frames:
            with Image.open(frame.image_path) as im:
                frame.width, frame.height = im.size

    rows = []
    for name, (_, size, _) in ENGINES.items():
        keys, dets = load_detections(out_dir / f"dets_{name}.npz")
        if keys != [f.frame_key for f in frames]:
            raise RuntimeError(f"Saved detections for {name} do not match the frame list")
        t0 = time.perf_counter()
        m = evaluate_detections(frames, dets, classes)
        rows.append({"engine": name, "image_size": size, "frames": len(frames),
                     "score_seconds": time.perf_counter() - t0,
                     **{k: m[k] for k in ("mAP", "mAP_50", "mAP_75", "mAP_small",
                                          "mAP_medium", "mAP_large", "AR_1", "AR_10", "AR_100")},
                     "per_class_AP": m["per_class_AP"]})
        print(f"  {name}: mAP {m['mAP']:.4f}  mAP50 {m['mAP_50']:.4f}  mAP75 {m['mAP_75']:.4f}  "
              f"APs {m['mAP_small']:.4f}  AR100 {m['AR_100']:.4f}", flush=True)

    write_report({"split": "test", "frames": len(frames), "videos": len(videos),
                  "protocol": "36 classes, uncalibrated, TensorRT fp16; merged prior from the "
                              "960 unmerged engine on the previous frame", "results": rows},
                 ws.result("full_split_eval.json"))
    print("FULL_SPLIT_DONE", flush=True)


if __name__ == "__main__":
    main()
