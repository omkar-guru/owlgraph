"""Cache frozen-detector features for every Action Genome person-object pair.

The pair head trains on features from the frozen Stage 1 detector, so they are
computed once. For each ground-truth box (the person and every labelled object)
two descriptors are stored, so the head can be trained on either:

``pool``   area-weighted mean of the 768-dim patch features under the box
``patch``  the feature of the single patch whose *predicted* box best matches the
           ground-truth box - what the deployed streaming detector hands later
           stages, so training on it avoids a train/deploy mismatch

plus that best match's IoU, so the quality of the correspondence is visible.
Features come from ``base_feat_fp16.plan`` (960px unmerged, ``--features``).
Output: one NPZ per split under ``artifacts/pair_features/``.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from sggpipeline.ag.relations import load_relation_frames
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import preprocess_sizes
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.pipeline import Workspace, load_queries

GRID = 60


def pool_boxes(fmap: torch.Tensor, boxes: torch.Tensor, side: float) -> torch.Tensor:
    """Area-weighted mean of a (G, G, D) token grid under xyxy pixel boxes.

    Same geometry as ``tracking.features.pool_box_features``: the grid covers the
    bottom/right-padded square of side ``side``.
    """
    edges = torch.linspace(0, side, GRID + 1, device=boxes.device)
    ox = (torch.minimum(boxes[:, 2:3], edges[None, 1:]) - torch.maximum(boxes[:, 0:1], edges[None, :-1])).clamp(min=0)
    oy = (torch.minimum(boxes[:, 3:4], edges[None, 1:]) - torch.maximum(boxes[:, 1:2], edges[None, :-1])).clamp(min=0)
    w = oy[:, :, None] * ox[:, None, :]
    return torch.einsum("bhw,hwd->bd", w, fmap) / w.sum(dim=(1, 2))[:, None].clamp(min=1e-6)


def prefetch(pool, frames, window: int = 64):
    """Decode images a bounded window ahead. ``pool.map`` would submit every frame
    at once and hold all ~218k decoded training images (~85 GB) in memory."""
    pending = deque()
    for frame in frames:
        pending.append(pool.submit(load_image, frame.image_path))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--max-frames", type=int, default=None, help="smoke tests")
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    out_dir = ws.root / "pair_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    query = torch.from_numpy(load_queries(ws, "base").embeds).cuda()
    runner = TRTRunner(ws.engine("base_feat_fp16.plan"))
    pre = GpuOwlv2Preprocessor(960, device="cuda")

    for split in args.splits:
        frames = load_relation_frames(Path(args.ag_root), split)[: args.max_frames]
        print(f"{split}: {len(frames)} frames, {sum(len(f.object_labels) for f in frames)} pairs",
              flush=True)
        subj = {"pool": [], "patch": []}
        obj = {"pool": [], "patch": []}
        subj_iou, obj_iou, sizes = [], [], []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=16) as pool:
            for frame, image in tqdm(zip(frames, prefetch(pool, frames)), total=len(frames),
                                     desc=split, mininterval=60):
                width, height = image.size
                side = preprocess_sizes(width, height)
                out = runner.infer({"pixel_values": pre([image]), "query_embeds": query})
                feats = out["patch_features"][0].float()  # (3600, 768)
                cx, cy, w, h = out["pred_boxes"][0].float().unbind(-1)
                pred = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * side
                gt = torch.from_numpy(np.vstack([frame.person_box[None], frame.object_boxes])).cuda()
                pooled = pool_boxes(feats.reshape(GRID, GRID, -1), gt, side)
                iou = box_iou(gt, pred)
                best_iou, best = iou.max(dim=1)
                patch = feats[best]
                subj["pool"].append(pooled[:1].half().cpu())
                subj["patch"].append(patch[:1].half().cpu())
                obj["pool"].append(pooled[1:].half().cpu())
                obj["patch"].append(patch[1:].half().cpu())
                subj_iou.append(best_iou[:1].cpu())
                obj_iou.append(best_iou[1:].cpu())
                sizes.append((width, height))
        minutes = (time.perf_counter() - t0) / 60
        counts = np.array([len(f.object_labels) for f in frames])
        np.savez(
            out_dir / f"{split}.npz",
            frame_keys=np.array([f.frame_key for f in frames]),
            video_ids=np.array([f.video_id for f in frames]),
            image_sizes=np.array(sizes, dtype=np.int32),
            pair_frame=np.repeat(np.arange(len(frames)), counts),
            person_boxes=np.stack([f.person_box for f in frames]),
            object_boxes=np.concatenate([f.object_boxes for f in frames]),
            object_labels=np.concatenate([f.object_labels for f in frames]),
            attention=np.concatenate([f.attention for f in frames]),
            spatial=np.concatenate([f.spatial for f in frames]),
            contacting=np.concatenate([f.contacting for f in frames]),
            subject_pool=torch.cat(subj["pool"]).numpy(),
            subject_patch=torch.cat(subj["patch"]).numpy(),
            object_pool=torch.cat(obj["pool"]).numpy(),
            object_patch=torch.cat(obj["patch"]).numpy(),
            subject_match_iou=torch.cat(subj_iou).numpy(),
            object_match_iou=torch.cat(obj_iou).numpy(),
            feature_source=np.array("owlv2-base-p16 960px unmerged TensorRT fp16, 768-dim "
                                    "pre-head patch features (base_feat_fp16.plan)"),
        )
        oi = torch.cat(obj_iou).numpy()
        print(f"{split}: cached in {minutes:.1f} min; object best-patch IoU median "
              f"{np.median(oi):.3f}, >=0.5 for {100 * (oi >= 0.5).mean():.1f}%", flush=True)
    print("PAIR_FEATURES_DONE", flush=True)


if __name__ == "__main__":
    main()
