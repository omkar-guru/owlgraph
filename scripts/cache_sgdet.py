"""Cache the frozen detector's detections for relationship training and SGDet.

For every Action Genome keyframe with a person and labelled relations, stores the
top N detections (plan.md's 32-object budget) from the 960px feature engine -
boxes, labels, scores and each detection's own 768-dim patch feature - next to
the ground truth, padded to fixed shapes so training batches stay on the GPU.

Detections: score >= 0.05, per-class NMS at 0.7, then the N highest scores.
Frames follow ``load_relation_frames`` order, the same as
``cache_pair_features.py``, so the two caches line up frame for frame.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sggpipeline.ag.classes import AG_OBJECT_CLASSES
from sggpipeline.ag.relations import PREDICATES, load_relation_frames
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import postprocess_device
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.pipeline import Workspace, load_queries

from cache_pair_features import prefetch

PERSON = AG_OBJECT_CLASSES.index("person")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--objects", type=int, default=32, help="detection budget per frame")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    out_dir = ws.root / "sgdet"
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()
    owner = torch.as_tensor(queries.owner, device="cuda")
    runner = TRTRunner(ws.engine("base_feat_fp16.plan"))
    pre = GpuOwlv2Preprocessor(960, device="cuda")
    n_obj = args.objects

    for split in args.splits:
        frames = load_relation_frames(Path(args.ag_root), split)[: args.max_frames]
        f = len(frames)
        g = 1 + max(len(fr.object_labels) for fr in frames)
        det_boxes = np.zeros((f, n_obj, 4), np.float32)
        det_labels = np.zeros((f, n_obj), np.int16)
        det_scores = np.zeros((f, n_obj), np.float16)
        det_count = np.zeros(f, np.int16)
        det_feats = np.zeros((f, n_obj, 768), np.float16)
        gt_boxes = np.zeros((f, g, 4), np.float32)
        gt_labels = np.zeros((f, g), np.int16)
        gt_count = np.zeros(f, np.int16)
        gt_pred = np.zeros((f, g, len(PREDICATES)), bool)
        sizes = np.zeros((f, 2), np.int32)
        print(f"{split}: {f} frames, up to {g - 1} objects per frame", flush=True)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=16) as pool:
            for i, (fr, image) in enumerate(tqdm(zip(frames, prefetch(pool, frames)), total=f,
                                                 desc=split, mininterval=60)):
                w, h = image.size
                sizes[i] = (w, h)
                out = runner.infer({"pixel_values": pre([image]), "query_embeds": query})
                packed = postprocess_device(out["pred_logits"], out["pred_boxes"], out["objectness"],
                                            owner, len(AG_OBJECT_CLASSES), (w, h), 0.05, n_obj, 0.7)
                sides = packed[:, 2:4] - packed[:, 0:2]
                packed = packed[(sides >= 1.0).all(dim=1)][:n_obj]
                feats = out["patch_features"][0].index_select(0, packed[:, 7].long())
                k = len(packed)
                det_count[i] = k
                host = packed.cpu().numpy()
                det_boxes[i, :k] = host[:, :4]
                det_scores[i, :k] = host[:, 4]
                det_labels[i, :k] = host[:, 5]
                det_feats[i, :k] = feats.half().cpu().numpy()
                m = len(fr.object_labels)
                gt_count[i] = 1 + m
                gt_boxes[i, 0], gt_labels[i, 0] = fr.person_box, PERSON
                gt_boxes[i, 1:1 + m], gt_labels[i, 1:1 + m] = fr.object_boxes, fr.object_labels
                gt_pred[i, 1 + np.arange(m), fr.attention] = True  # exactly one per object
                gt_pred[i, 1:1 + m, 3:9] = fr.spatial
                gt_pred[i, 1:1 + m, 9:] = fr.contacting
        np.savez(out_dir / f"{split}.npz", frame_keys=np.array([fr.frame_key for fr in frames]),
                 video_ids=np.array([fr.video_id for fr in frames]), image_sizes=sizes,
                 det_boxes=det_boxes, det_labels=det_labels, det_scores=det_scores,
                 det_count=det_count, det_feats=det_feats, gt_boxes=gt_boxes, gt_labels=gt_labels,
                 gt_count=gt_count, gt_pred=gt_pred,
                 settings=np.array(f"960px unmerged feature engine; score>=0.05; NMS 0.7; top {n_obj}"))
        print(f"{split}: cached in {(time.perf_counter() - t0) / 60:.1f} min; "
              f"mean detections/frame {det_count.mean():.1f}", flush=True)
    print("SGDET_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
