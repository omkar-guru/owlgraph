"""Does protecting the previous frame's detections recover merging's loss?

On the full test split, merging cost 3.3% mAP, concentrated in large plain
objects (table, floor, person, bed) whose interior windows score low objectness
and get merged. Here the previous frame's confident detections also guard their
windows: a window whose centre lies inside such a box is merged last.

Same frames and protocol as ``full_split_eval.py``. The merged engine runs three
ways per frame: unprotected (should reproduce 0.1043 mAP), and protecting boxes
the previous frame detected at score >= 0.3 and >= 0.15.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.merged_export import MergeIndexer
from sggpipeline.detect.owlv2 import postprocess, postprocess_torch
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.stream import PLAN_INPUTS
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.evaluation.detection_eval import evaluate_detections
from sggpipeline.pipeline import Workspace, load_queries, write_report

from early_objectness_probe import decode_positions
from full_split_eval import load_detections, save_detections

VARIANTS = {"unprotected": None, "protect_0.30": 0.30, "protect_0.15": 0.15}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--max-videos", type=int, default=None)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    out_dir = ws.root / "full_split"
    ag = ActionGenome(root=Path(args.ag_root), split="test").load()
    classes = ag.classes
    by_video = defaultdict(list)
    for f in ag.frames:
        if f.image_path and f.frame_index - 2 >= 0:
            by_video[f.video_id].append(f)
    videos = sorted(by_video)[: args.max_videos]
    frames = [f for v in videos for f in by_video[v]]
    print(f"{len(frames)} frames from {len(videos)} videos", flush=True)

    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()
    owner = torch.as_tensor(queries.owner, device="cuda")
    seed = TRTRunner(ws.engine("base_fp16.plan"))
    merged = TRTRunner(ws.engine("base_merged50_np_fp16.plan"))
    pre = GpuOwlv2Preprocessor(960, device="cuda")
    indexer = MergeIndexer(60, 0.5)
    videos_dir = Path(args.ag_root) / "Charades_v1_480"
    dets = {k: [] for k in VARIANTS}

    def priors_for(video):
        return decode_positions(videos_dir / video, {f.frame_index - 2 for f in by_video[video]})

    t0 = time.perf_counter()
    window = 16
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {v: pool.submit(priors_for, v) for v in videos[:window]}
        for i, video in enumerate(tqdm(videos, desc="videos", mininterval=60)):
            if i + window < len(videos):
                futures[videos[i + window]] = pool.submit(priors_for, videos[i + window])
            priors = futures.pop(video).result()
            for frame in by_video[video]:
                image = load_image(frame.image_path)
                frame.width, frame.height = image.size
                prior_img = priors[frame.frame_index - 2]
                prior_out = seed.infer({"pixel_values": pre([prior_img]), "query_embeds": query})
                objectness = prior_out["objectness"][0].float().clone()
                prior_dets = postprocess_torch(prior_out["pred_logits"], prior_out["pred_boxes"],
                                               prior_out["objectness"], owner, len(classes),
                                               prior_img.size, 0.15, 100)
                px = pre([image])
                for name, threshold in VARIANTS.items():
                    boxes = (prior_dets.boxes[prior_dets.scores >= threshold]
                             if threshold is not None else None)
                    plan = indexer(objectness, boxes, prior_img.size)
                    out = merged.infer({"pixel_values": px, "query_embeds": query,
                                        **dict(zip(PLAN_INPUTS, plan))})
                    torch.cuda.synchronize()
                    a = {k: v.float().cpu().numpy() for k, v in out.items()}
                    dets[name].append(postprocess(a["pred_logits"], a["pred_boxes"], a["objectness"],
                                                  queries.owner, len(classes), image.size, 0.05, 100))
    print(f"inference done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    keys = [f.frame_key for f in frames]
    rows = []
    for name in VARIANTS:
        path = out_dir / f"dets_merged50_{name}.npz"
        save_detections(path, keys, dets[name])
        _, saved = load_detections(path)
        m = evaluate_detections(frames, saved, classes)
        rows.append({"variant": name, **{k: m[k] for k in ("mAP", "mAP_50", "mAP_75", "mAP_small",
                                                            "mAP_medium", "mAP_large", "AR_100")},
                     "per_class_AP": m["per_class_AP"]})
        print(f"  {name}: mAP {m['mAP']:.4f}  mAP50 {m['mAP_50']:.4f}  mAP75 {m['mAP_75']:.4f}  "
              f"APl {m['mAP_large']:.4f}  AR100 {m['AR_100']:.4f}", flush=True)
    write_report({"frames": len(frames), "reference_960_unmerged_mAP": 0.1079, "results": rows},
                 ws.result("merge_protection_eval.json"))
    print("PROTECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
