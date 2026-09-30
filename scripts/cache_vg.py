"""Cache the frozen detector's VG150 detections for relationship training and SGDet.

Same 960px feature engine as Action Genome. Its text queries are compiled in
(the AG vocabulary), so VG150's 150 class names are scored from the engine's
per-patch features with OWLv2's own class head - exactly the computation inside
the engine, with a different query set. The first image checks that claim: the
class head on the engine's features must reproduce the engine's AG logits.

Per image (only images with relations, the reference protocol):

* the top N detections (score >= threshold, per-class NMS) - boxes, labels,
  scores, and each detection's own 768-dim patch feature;
* for PredCls, each ground-truth box's feature: the patch whose predicted box
  overlaps it best (and that IoU), as ``cache_pair_features.py`` does for AG;
* the ground truth itself, padded to fixed shapes.

Detection is zero-shot: OWLv2 is not trained on VG150's labels here.
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

from sggpipeline.ag.classes import build_prompt_index
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import encode_text_queries, load_owlv2, postprocess_device, preprocess_sizes
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.pipeline import Workspace, load_queries
from sggpipeline.vg.data import iter_split, vocabulary


def prefetch(pool, images, window: int = 64):
    pending = deque()
    for item in images:
        pending.append((item, pool.submit(item.image)))
        if len(pending) >= window:
            item, fut = pending.popleft()
            yield item, fut.result()
    while pending:
        item, fut = pending.popleft()
        yield item, fut.result()


def pad(rows: list[np.ndarray], width: int, dtype, fill=0) -> np.ndarray:
    out = np.full((len(rows), width) + rows[0].shape[1:], fill, dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--vg-root", default="/workspace/vg150")
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    ap.add_argument("--objects", type=int, default=64, help="detections kept per image")
    ap.add_argument("--score", type=float, default=0.01)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    out_dir = ws.root / "vg150"
    out_dir.mkdir(parents=True, exist_ok=True)
    classes, predicates = vocabulary(Path(args.vg_root))
    model, processor = load_owlv2("base", device="cuda", dtype=torch.float32)
    prompts, owner = build_prompt_index(classes)
    vg_query = encode_text_queries(model, processor, prompts, "cuda")[None]
    owner = torch.as_tensor(owner, device="cuda")
    ag = load_queries(ws, "base")
    ag_query = torch.from_numpy(ag.embeds).cuda()
    runner = TRTRunner(ws.engine("base_feat_fp16.plan"))
    pre = GpuOwlv2Preprocessor(960, device="cuda")
    n_obj = args.objects
    checked = False

    for split in args.splits:
        t0 = time.perf_counter()
        det = {k: [] for k in ("boxes", "labels", "scores", "feats")}
        gt = {k: [] for k in ("boxes", "labels", "rels", "feats", "iou")}
        ids, sizes = [], []
        stream = iter_split(Path(args.vg_root), split)
        with ThreadPoolExecutor(max_workers=16) as pool, torch.no_grad():
            for i, (item, image) in enumerate(tqdm(prefetch(pool, stream), desc=split, mininterval=60)):
                if args.max_images and i >= args.max_images:
                    break
                w, h = image.size
                out = runner.infer({"pixel_values": pre([image]), "query_embeds": ag_query})
                feats = out["patch_features"].float()
                logits = model.class_predictor(feats, vg_query, None)[0]
                if not checked:
                    ref = model.class_predictor(feats, ag_query[None].float(), None)[0]
                    err = (ref - out["pred_logits"].float()).abs().max().item()
                    print(f"class head on engine features vs engine AG logits: max |diff| {err:.4f}",
                          flush=True)
                    assert err < 0.5, "engine features do not reproduce the engine's class head"
                    checked = True
                packed = postprocess_device(logits, out["pred_boxes"], out["objectness"], owner,
                                            len(classes), (w, h), args.score, n_obj, args.nms)
                sides = packed[:, 2:4] - packed[:, 0:2]
                packed = packed[(sides >= 1.0).all(dim=1)][:n_obj]
                host = packed.cpu().numpy()
                det["boxes"].append(host[:, :4])
                det["scores"].append(host[:, 4])
                det["labels"].append(host[:, 5].astype(np.int16))
                det["feats"].append(feats[0].index_select(0, packed[:, 7].long()).half().cpu().numpy())

                # PredCls features: the patch whose own predicted box best fits each GT box.
                cx, cy, bw, bh = out["pred_boxes"][0].float().unbind(-1)
                scale = preprocess_sizes(w, h)
                all_boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) * scale
                gt_boxes = torch.from_numpy(item.boxes).cuda()
                best_iou, best = box_iou(gt_boxes, all_boxes).max(dim=1)
                gt["feats"].append(feats[0, best].half().cpu().numpy())
                gt["iou"].append(best_iou.cpu().numpy())
                gt["boxes"].append(item.boxes)
                gt["labels"].append(item.labels.astype(np.int16))
                gt["rels"].append(item.relations.astype(np.int16))
                ids.append(item.image_id)
                sizes.append((w, h))

        f = len(ids)
        g = max(len(b) for b in gt["boxes"])
        r = max(len(x) for x in gt["rels"])
        arrays = dict(
            image_ids=np.array(ids), image_sizes=np.array(sizes, np.int32),
            det_count=np.array([len(b) for b in det["boxes"]], np.int16),
            det_boxes=pad(det["boxes"], n_obj, np.float32),
            det_labels=pad(det["labels"], n_obj, np.int16),
            det_scores=pad(det["scores"], n_obj, np.float16),
            det_feats=pad(det["feats"], n_obj, np.float16),
            gt_count=np.array([len(b) for b in gt["boxes"]], np.int16),
            gt_boxes=pad(gt["boxes"], g, np.float32),
            gt_labels=pad(gt["labels"], g, np.int16),
            rel_count=np.array([len(x) for x in gt["rels"]], np.int16),
            gt_rels=pad(gt["rels"], r, np.int16, fill=-1),
            gt_feat_iou=pad(gt["iou"], g, np.float32),
            settings=np.array(f"960px unmerged feature engine, zero-shot VG150 class queries; "
                              f"score>={args.score}; NMS {args.nms}; top {n_obj}"))
        if split != "train":  # PredCls is evaluated, never trained on
            arrays["gt_feats"] = pad(gt["feats"], g, np.float16)
        np.savez(out_dir / f"{split}.npz", **arrays)
        print(f"{split}: {f} images in {(time.perf_counter() - t0) / 60:.1f} min; up to {g} objects, "
              f"{r} relations; mean detections {arrays['det_count'].mean():.1f}; GT best-patch IoU "
              f"median {np.median(np.concatenate(gt['iou'])):.3f}", flush=True)
    print("VG_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
