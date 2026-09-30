"""The relationship head on the deployed stream: latency, and SGDet end to end.

The cached evaluation (``train_relationships.py``) scores the head on
detections from the unmerged engine. Deployment runs it on the merged streaming
detector (self-fed plans, box protection, per-detection features). This checks
that path directly:

``--latency``  sustained ms/frame: detector alone, then detector + relationship
               head at several pair budgets (frames resident on the GPU).
``--eval``     SGDet on test keyframes through the deployed path: the keyframe's
               previous video frame seeds the stream (unmerged), the keyframe
               runs merged with that prior, then the head. Same recall code and
               matching rules as the cached evaluation.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sggpipeline.ag.classes import AG_OBJECT_CLASSES
from sggpipeline.ag.relations import PREDICATES, load_relation_frames
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.stream import StreamingDetector
from sggpipeline.pipeline import Workspace, load_queries, write_report
from sggpipeline.relations.predicates import load_or_build
from sggpipeline.relations.runtime import RelationshipPredictor
from sggpipeline.relations.sgdet import SGRecall, match_matrix

from early_objectness_probe import decode_positions
from verify_bridge import decode

PERSON = AG_OBJECT_CLASSES.index("person")


def build(ws: Workspace, checkpoint: str, classifier: str, budget: int):
    queries = load_queries(ws, "base")
    detector = StreamingDetector(ws.engine("base_feat_fp16.plan"),
                                 ws.engine("base_merged50_np_feat_fp16.plan"), queries,
                                 len(AG_OBJECT_CLASSES), score_threshold=0.05)
    embeds = load_or_build(ws.cache("predicate_embeds_base.npy"))
    head = RelationshipPredictor(ws.root / "sgdet" / checkpoint, embeds, classifier,
                                 pair_budget=budget)
    return detector, head


def latency(ws, args) -> dict:
    frames = [f.cuda() for f in decode(Path(args.video), 300)]
    out = {}

    def sustained(fn, seconds=5.0):
        for i in range(50):
            fn(i)
        torch.cuda.synchronize()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fn(n)
            n += 1
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0) / n

    detector, head = build(ws, args.checkpoint, args.classifier, 128)
    detector.reset()
    out["detector_only"] = sustained(lambda i: detector(frames[i % len(frames)]))
    for budget in (64, 128, 256, 992):
        head.pair_budget = budget
        detector.reset()
        out[f"detector+relations@{budget}"] = sustained(
            lambda i: head(detector(frames[i % len(frames)])))
    for k, v in out.items():
        print(f"  {k:<28} {v:6.2f} ms/frame ({100 * v / 16:.0f}% of 16 ms)", flush=True)
    # A readable sample, to eyeball that the triplets make sense.
    detector.reset()
    head.pair_budget = 128
    for i in range(0, 120, 40):
        r = detector(frames[i])
        rel = head(r)
        names = [AG_OBJECT_CLASSES[l] for l in r.detections.labels]
        top = rel.triplets(r.detections.scores, top=4)
        print(f"  frame {i}: " + "; ".join(f"{names[s]} {p} {names[o]} ({sc:.2f})"
                                           for s, o, p, sc in top), flush=True)
    return out


def evaluate(ws, args) -> dict:
    frames = load_relation_frames(Path(args.ag_root), "test")
    frames = [f for f in frames if int(Path(f.frame_key).stem) - 2 >= 0]
    by_video = defaultdict(list)
    for f in frames:
        by_video[f.video_id].append(f)
    videos = sorted(by_video)[: args.max_videos]
    detector, head = build(ws, args.checkpoint, args.classifier, args.budget)
    acc = SGRecall()
    videos_dir = Path(args.ag_root) / "Charades_v1_480"
    for video in tqdm(videos, desc="videos", mininterval=60):
        vf = by_video[video]
        priors = decode_positions(videos_dir / video, {int(Path(f.frame_key).stem) - 2 for f in vf})
        for f in vf:
            detector.reset()
            detector(priors[int(Path(f.frame_key).stem) - 2])  # seed: previous frame, unmerged
            r = detector(load_image(f.image_path))  # keyframe: merged, prior from previous
            rel = head(r)
            det = r.detections
            n = min(len(det.scores), head.max_objects)
            m = len(f.object_labels)
            gt_boxes = torch.from_numpy(np.vstack([f.person_box[None], f.object_boxes]))[None]
            gt_labels = torch.from_numpy(np.concatenate([[PERSON], f.object_labels]))[None]
            gt_pred = torch.zeros(1 + m, len(PREDICATES), dtype=torch.bool)
            gt_pred[1 + torch.arange(m), torch.from_numpy(f.attention)] = True
            gt_pred[1:, 3:9] = torch.from_numpy(f.spatial)
            gt_pred[1:, 9:] = torch.from_numpy(f.contacting)
            det_valid = torch.ones(1, n, dtype=torch.bool)
            match = match_matrix(torch.from_numpy(det.boxes[:n])[None],
                                 torch.from_numpy(det.labels[:n])[None], det_valid, gt_boxes,
                                 gt_labels, torch.ones(1, 1 + m, dtype=torch.bool))[0]
            pairs = torch.from_numpy(np.stack([rel.subject, rel.object], axis=1)).long()
            acc.add_frame(pairs, torch.from_numpy(rel.predicate_probs), torch.from_numpy(rel.pair_score),
                          torch.from_numpy(det.scores[:n]), match, gt_pred,
                          torch.ones(1 + m, dtype=torch.bool), det_valid[0])
    s = acc.summary()
    print(f"deployed-path SGDet @{args.budget} pairs over {s['frames']} frames: object recall "
          f"{s['object_recall']:.4f}, pair recall {s['pair_recall']:.4f}, wc R@20 "
          f"{s['with_constraint/R@20']:.4f} R@50 {s['with_constraint/R@50']:.4f} mR@50 "
          f"{s['with_constraint/mR@50']:.4f}, nc R@50 {s['no_constraint/R@50']:.4f} "
          f"mR@50 {s['no_constraint/mR@50']:.4f}", flush=True)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--checkpoint", default="rel_text_none.pt")
    ap.add_argument("--classifier", default="text")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--video", default="acgdataset/Charades_v1_480/001YG.mp4")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--max-videos", type=int, default=None)
    args = ap.parse_args()
    ws = Workspace(Path(args.artifacts))
    report = {"checkpoint": args.checkpoint, "budget": args.budget}
    if args.latency:
        report["latency_ms"] = latency(ws, args)
    if args.eval:
        report["deployed_sgdet"] = evaluate(ws, args)
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    write_report(report, ws.result(f"relationship_stream_{gpu}.json"))
    print("REL_STREAM_DONE", flush=True)


if __name__ == "__main__":
    main()
