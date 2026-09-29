"""Does previous-frame-guided merging lock in misses when it feeds itself?

The single-frame benchmark took the previous frame's objectness from an
*unmerged* pass. A real stream has only the previous *merged* pass. That closes
a loop: an object appearing inside a region the prior marked as background can
be missed, so its objectness stays low, so the region stays merged next frame,
and the object may never be picked up. This runs every consecutive frame of video
segments to measure that loop directly.

Per frame, in the same stream:

``ref``        unmerged model; supplies the pseudo ground truth and a clean prior
``selfF``      early fraction F, prior = its *own* objectness on the previous frame
``refpriorF``  early fraction F, prior = ``ref``'s objectness on the previous frame
               (what the single-frame benchmark measured)

``selfF`` vs ``refpriorF`` isolates lock-in: same merge budget, only the prior's
source differs. Each segment starts with one unmerged frame (a refresh), and
results are bucketed by frames since that refresh - so the same run shows what
any refresh interval would buy.

Pseudo ground truth: ``ref`` detections scored >= 0.3. AG labels only sparse
keyframes, so this measures what merging *loses relative to not merging* on every
frame, not absolute accuracy. An object is recalled if a same-class detection
scored >= 0.1 overlaps it at IoU >= 0.5. "Appearing" objects are pseudo-GT with no
same-class match (IoU >= 0.3) in the previous frame's pseudo-GT - the case lock-in
threatens most.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import load_owlv2, postprocess, preprocess_sizes
from sggpipeline.detect.token_merging import MergePlan, WindowGrid, merged_forward
from sggpipeline.pipeline import Workspace, load_queries, write_report

PSEUDO_GT_SCORE = 0.3
RECALL_SCORE = 0.1
MATCH_IOU = 0.5
APPEAR_IOU = 0.3
BUCKETS = ((1, 5), (6, 15), (16, 30), (31, 60), (61, 10_000))


def decode_segment(path: Path, start: int, length: int) -> list:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for pos, frame in enumerate(container.decode(stream)):
            if pos >= start:
                frames.append(frame.to_image())
            if len(frames) == length:
                break
    return frames


def bucket_of(t: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= t <= hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    raise ValueError(t)


def detections(out, queries, num_classes, size, threshold=0.05):
    return postprocess(out["pred_logits"].float().cpu().numpy(),
                       out["pred_boxes"].float().cpu().numpy(),
                       out["objectness"].float().cpu().numpy(),
                       queries.owner, num_classes, size, threshold, 100)


def match(gt_boxes, gt_labels, det, min_score, iou_thr):
    """Boolean per GT box: matched by a same-class detection above the score."""
    keep = det.scores >= min_score
    if not keep.any() or len(gt_boxes) == 0:
        return np.zeros(len(gt_boxes), dtype=bool)
    ious = box_iou(torch.as_tensor(gt_boxes), torch.as_tensor(det.boxes[keep])).numpy()
    same = gt_labels[:, None] == det.labels[keep][None, :]
    return ((ious >= iou_thr) & same).any(axis=1)


def centre_windows(boxes, image_size, grid: WindowGrid, input_size: int):
    """Window index containing each box centre, in model-input coordinates."""
    scale = input_size / preprocess_sizes(*image_size)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 * scale
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 * scale
    patch = input_size // grid.side
    col = np.clip((cx // patch).astype(int), 0, grid.side - 1)
    row = np.clip((cy // patch).astype(int), 0, grid.side - 1)
    return (row // 2) * grid.half + col // 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--configs", nargs="+", default=["50", "70nd"],
                    help="early merge percent; suffix 'nd' disables dilation")
    ap.add_argument("--videos", type=int, default=24)
    ap.add_argument("--start", type=int, default=30)
    ap.add_argument("--length", type=int, default=150)
    ap.add_argument("--image-size", type=int, default=960)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    queries = load_queries(ws, "base")
    model, _ = load_owlv2("base", device="cuda", dtype=torch.float16)
    grid = WindowGrid(model.num_patches_height, "cuda")
    pre = GpuOwlv2Preprocessor(args.image_size, device="cuda")
    query = torch.from_numpy(queries.embeds).cuda().half()

    ag = ActionGenome(root=Path(args.ag_root), split="test").load()
    num_classes = len(ag.classes)
    video_ids = sorted({f.video_id for f in ag.frames})
    step = max(1, len(video_ids) // args.videos)
    chosen = video_ids[::step][: args.videos]
    videos_dir = Path(args.ag_root) / "Charades_v1_480"

    configs = {}
    for spec in args.configs:
        dilate = not spec.endswith("nd")
        frac = int(spec.removesuffix("nd")) / 100
        configs[f"self{spec}"] = (frac, dilate, "self")
        configs[f"refprior{spec}"] = (frac, dilate, "ref")

    # stats[config][bucket] -> counters
    stats = {c: defaultdict(lambda: defaultdict(int)) for c in configs}
    iou_sum = defaultdict(float)
    frames_run = 0

    for video_id in tqdm(chosen, desc="segments"):
        images = decode_segment(videos_dir / video_id, args.start, args.length)
        if len(images) < 20:
            continue
        own_prior = {c: None for c in configs}
        ref_prior = None
        prev_pseudo = None

        for t, image in enumerate(images):
            px = pre([image]).half()
            ref = merged_forward(model, px, query, MergePlan(), grid)
            ref_det = detections(ref, queries, num_classes, image.size)
            conf = ref_det.scores >= PSEUDO_GT_SCORE
            pseudo_boxes, pseudo_labels = ref_det.boxes[conf], ref_det.labels[conf]

            if t == 0:
                # Refresh frame: every config runs unmerged, seeding its own prior.
                for c in configs:
                    own_prior[c] = ref["objectness"][0].float()
            else:
                appearing = np.ones(len(pseudo_boxes), dtype=bool)
                if prev_pseudo is not None and len(pseudo_boxes):
                    appearing = ~match(pseudo_boxes, pseudo_labels, prev_pseudo,
                                       0.0, APPEAR_IOU)
                bucket = bucket_of(t)
                for c, (frac, dilate, source) in configs.items():
                    prior = own_prior[c] if source == "self" else ref_prior
                    plan = MergePlan(early_fraction=frac, early_scores=prior, early_dilate=dilate)
                    out = merged_forward(model, px, query, plan, grid)
                    own_prior[c] = out["objectness"][0].float()
                    det = detections(out, queries, num_classes, image.size)
                    hit = match(pseudo_boxes, pseudo_labels, det, RECALL_SCORE, MATCH_IOU)
                    merged_mask = out["window_merged"].cpu().numpy()
                    in_merged = (merged_mask[centre_windows(pseudo_boxes, image.size, grid,
                                                            args.image_size)]
                                 if len(pseudo_boxes) else np.zeros(0, dtype=bool))
                    s = stats[c][bucket]
                    s["objects"] += len(pseudo_boxes)
                    s["recalled"] += int(hit.sum())
                    s["appearing"] += int(appearing.sum())
                    s["appearing_recalled"] += int((hit & appearing).sum())
                    s["centre_in_merged"] += int(in_merged.sum())
                    s["missed_in_merged"] += int((~hit & in_merged).sum())
                frames_run += 1

            ref_prior = ref["objectness"][0].float()
            prev_pseudo = type(ref_det)(ref_det.boxes[conf], ref_det.scores[conf],
                                        ref_det.labels[conf], ref_det.objectness[conf])

    def rate(num, den):
        return num / den if den else None

    rows = []
    for c in configs:
        buckets = {}
        total = defaultdict(int)
        for b, s in stats[c].items():
            buckets[b] = {
                "objects": s["objects"],
                "recall": rate(s["recalled"], s["objects"]),
                "appearing": s["appearing"],
                "appearing_recall": rate(s["appearing_recalled"], s["appearing"]),
                "centre_in_merged": rate(s["centre_in_merged"], s["objects"]),
            }
            for k, v in s.items():
                total[k] += v
        rows.append({
            "config": c,
            "recall": rate(total["recalled"], total["objects"]),
            "appearing_recall": rate(total["appearing_recalled"], total["appearing"]),
            "centre_in_merged": rate(total["centre_in_merged"], total["objects"]),
            "miss_share_in_merged": rate(total["missed_in_merged"],
                                         total["objects"] - total["recalled"]),
            "objects": total["objects"],
            "appearing": total["appearing"],
            "by_frames_since_refresh": buckets,
        })

    write_report({"image_size": args.image_size, "videos": len(chosen),
                  "segment": {"start": args.start, "length": args.length},
                  "frames_scored": frames_run, "pseudo_gt_score": PSEUDO_GT_SCORE,
                  "results": rows}, ws.result("merge_streaming_test.json"))

    def fmt(v):
        return "   -  " if v is None else f"{v:6.3f}"

    print(f"\n{frames_run} consecutive frames over {len(chosen)} segments @ {args.image_size}px; "
          f"pseudo-GT = unmerged detections >= {PSEUDO_GT_SCORE}\n")
    print(f"{'config':<12}{'recall':>8}{'appear':>8}{'ctr-merged':>11}{'miss-in-merged':>15}")
    for r in rows:
        print(f"{r['config']:<12}{fmt(r['recall']):>8}{fmt(r['appearing_recall']):>8}"
              f"{fmt(r['centre_in_merged']):>11}{fmt(r['miss_share_in_merged']):>15}")
    labels = [f"{lo}-{hi}" if hi < 10_000 else f"{lo}+" for lo, hi in BUCKETS]
    print("\nrecall by frames since refresh")
    print(f"{'config':<12}" + "".join(f"{l:>8}" for l in labels))
    for r in rows:
        b = r["by_frames_since_refresh"]
        print(f"{r['config']:<12}" + "".join(
            f"{fmt(b.get(l, {}).get('recall')):>8}" for l in labels))
    print("\nappearing-object recall by frames since refresh")
    for r in rows:
        b = r["by_frames_since_refresh"]
        print(f"{r['config']:<12}" + "".join(
            f"{fmt(b.get(l, {}).get('appearing_recall')):>8}" for l in labels))


if __name__ == "__main__":
    main()
