"""Verify the Stage 1 -> Stage 2 bridge on real video, and measure what it costs.

1. Correctness: both feature engines (``--features`` builds) against the eager
   fp32 model on real consecutive frames - detections *and* the new
   ``patch_features`` output - plus a check that adding the output did not change
   the detection outputs relative to the existing engines.
2. Stream sanity: ``StreamingDetector`` over whole videos - positive-area boxes,
   no same-class duplicates above the NMS threshold, finite nonzero features.
3. Cost: sustained per-frame time of the full streaming detector (engine, NMS,
   feature gather, host copies) next to the bare merged engine, so the bridge's
   share of the 16 ms budget is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import av
import numpy as np
import torch
from torchvision.ops import box_iou

from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.merged_export import MergedDetectionGraph, MergeIndexer
from sggpipeline.detect.owlv2 import Owlv2DetectionGraph, load_owlv2
from sggpipeline.detect.stream import PLAN_INPUTS, StreamingDetector
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.pipeline import Workspace, load_queries, write_report


def decode(path: Path, limit: int) -> list[torch.Tensor]:
    frames = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for f in c.decode(s):
            frames.append(torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous())
            if len(frames) >= limit:
                break
    return frames


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=240)
    args = ap.parse_args()
    ws = Workspace(Path(args.artifacts))
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()
    report = {}

    # 1. Correctness against eager fp32.
    model, _ = load_owlv2("base", device="cuda", dtype=torch.float32)
    pre = GpuOwlv2Preprocessor(960, device="cuda")
    seed_trt = TRTRunner(ws.engine("base_feat_fp16.plan"))
    merged_trt = TRTRunner(ws.engine("base_merged50_np_feat_fp16.plan"))
    plain_trt = TRTRunner(ws.engine("base_merged50_np_fp16.plan"))
    indexer = MergeIndexer(60, 0.5)
    seed_ref = Owlv2DetectionGraph(model, with_features=True).eval()
    merged_ref = MergedDetectionGraph(model, indexer.num_merged, proportional=False,
                                      with_features=True).eval().cuda()
    names = ("pred_logits", "pred_boxes", "objectness", "patch_features")
    frames = decode(Path(args.videos[0]), 200)
    checks = {"seed": [], "merged": [], "merged_vs_plain_engine": []}
    for i in range(1, 9):
        prev = pre.preprocess_tensor(frames[i * 20 - 1].unsqueeze(0).cuda())
        px = pre.preprocess_tensor(frames[i * 20].unsqueeze(0).cuda())
        with torch.no_grad():
            ref = dict(zip(names, seed_ref(px.float(), query)))
        out = seed_trt.infer({"pixel_values": px, "query_embeds": query})
        checks["seed"].append({n: cosine(out[n], ref[n]) for n in names})

        with torch.no_grad():
            prior = seed_ref(prev.float(), query)[2][0].float()
        plan = indexer(prior)
        with torch.no_grad():
            ref = dict(zip(names, merged_ref(px.float(), query, *plan)))
        feeds = {"pixel_values": px, "query_embeds": query, **dict(zip(PLAN_INPUTS, plan))}
        out = merged_trt.infer(dict(feeds))
        checks["merged"].append({n: cosine(out[n], ref[n]) for n in names})
        plain = plain_trt.infer(dict(feeds))
        checks["merged_vs_plain_engine"].append(
            {n: cosine(out[n], plain[n]) for n in names[:3]})
    summary = {k: {n: float(np.mean([c[n] for c in v])) for n in v[0]} for k, v in checks.items()}
    passed = all(val >= 0.999 for group in summary.values() for val in group.values())
    report["correctness"] = {"mean_cosine": summary, "passed": passed}
    for group, vals in summary.items():
        print(f"{group:<24}" + "  ".join(f"{n} {v:.5f}" for n, v in vals.items()), flush=True)
    print(f"correctness -> {'PASS' if passed else 'FAIL'}", flush=True)
    del model, seed_ref, merged_ref
    torch.cuda.empty_cache()

    # 2. Stream sanity over whole videos.
    detector = StreamingDetector(ws.engine("base_feat_fp16.plan"),
                                 ws.engine("base_merged50_np_feat_fp16.plan"), queries, 36)
    counts, dup_pairs, degenerate, bad_features, merged_frames = [], 0, 0, 0, 0
    for video in args.videos:
        detector.reset()
        for frame in decode(Path(video), args.frames):
            r = detector(frame)
            d = r.detections
            counts.append(len(d.scores))
            merged_frames += r.merged
            degenerate += int(((d.boxes[:, 2:] - d.boxes[:, :2]) <= 0).any(axis=1).sum())
            norms = np.linalg.norm(r.features, axis=1)
            bad_features += int((~np.isfinite(r.features).all(axis=1) | (norms < 1e-6)).sum())
            if len(d.scores) > 1:
                iou = box_iou(torch.from_numpy(d.boxes), torch.from_numpy(d.boxes)).numpy()
                np.fill_diagonal(iou, 0)
                dup_pairs += int(((iou > 0.7) & (d.labels[:, None] == d.labels[None, :])).sum() // 2)
            assert r.features.shape == (len(d.scores), 768)
    sanity = {"frames": len(counts), "merged_frames": merged_frames,
              "detections_per_frame": float(np.mean(counts)),
              "same_class_pairs_iou_gt_0_7": dup_pairs, "degenerate_boxes": degenerate,
              "bad_feature_rows": bad_features}
    sanity["passed"] = dup_pairs == 0 and degenerate == 0 and bad_features == 0
    report["stream_sanity"] = sanity
    print("stream sanity:", sanity, flush=True)

    # 3. Cost: full streaming detector vs bare merged engine, sustained.
    frames = decode(Path(args.videos[0]), 300)
    gpu_frames = [f.cuda() for f in frames]

    def sustained(fn, seconds=8.0):
        for i in range(50):
            fn(i)
        torch.cuda.synchronize()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fn(n)
            n += 1
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0) / n

    detector.reset()
    stream_ms = sustained(lambda i: detector(gpu_frames[i % len(gpu_frames)]))
    prior = plain_trt.infer({"pixel_values": pre.preprocess_tensor(gpu_frames[0].unsqueeze(0)),
                             "query_embeds": query, **dict(zip(PLAN_INPUTS, indexer(
                                 torch.rand(3600, device="cuda"))))})["objectness"][0].float().clone()

    def bare(i):
        nonlocal prior
        feeds = {"pixel_values": pre.preprocess_tensor(gpu_frames[i % len(gpu_frames)].unsqueeze(0)),
                 "query_embeds": query, **dict(zip(PLAN_INPUTS, indexer(prior)))}
        prior = plain_trt.infer(feeds)["objectness"][0].float().clone()

    bare_ms = sustained(bare)
    report["cost"] = {"streaming_detector_ms": stream_ms, "bare_merged_engine_ms": bare_ms,
                      "bridge_overhead_ms": stream_ms - bare_ms,
                      "note": "sustained, frames already on GPU; decode excluded"}
    print(f"cost: streaming detector {stream_ms:.2f} ms/frame vs bare merged engine "
          f"{bare_ms:.2f} ms/frame -> bridge adds {stream_ms - bare_ms:.2f} ms", flush=True)

    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    write_report(report, ws.result(f"verify_bridge_{gpu}.json"))


if __name__ == "__main__":
    main()
