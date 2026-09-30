"""Detector speed on this GPU, measured the way a 60 FPS stream would use it.

Needs only the engines (``build_engines.py``) and one or more video files - not
the Action Genome dataset. Engine timing depends on tensor shapes, not content,
and accuracy is a property of the engine that the correctness gate re-checks.

Per engine:

``engine``     one engine call, CUDA events
``frame``      preprocess + merge plan + engine for one frame, synchronised
``sustained``  a real stream: a decoder thread feeds pinned frames while the GPU
               preprocesses and runs the engine; the merged engine builds each
               frame's plan from its *own* previous output, as deployed.
               Wall-clock frames/second over the whole run.

Also: decode-only speed (the CPU side must sustain the target on its own),
CPU postprocessing cost per frame, and a fp16 matmul roofline for the GPU.

Correctness gate: the merged engine is compared with the eager fp32 model on
real consecutive frames; it must reach logit cosine >= 0.999.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import av
import numpy as np
import torch

from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.merged_export import MergeIndexer
from sggpipeline.detect.owlv2 import load_owlv2, postprocess
from sggpipeline.detect.token_merging import MergePlan, WindowGrid, merged_forward
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.evaluation.verify import compare_outputs
from sggpipeline.pipeline import Workspace, load_queries, write_report

ENGINES = {
    "960 unmerged": ("base_fp16.plan", 960, False),
    "960 merged50": ("base_merged50_np_fp16.plan", 960, True),
    "768 unmerged": ("base768_fp16.plan", 768, False),
    "640 unmerged": ("base640_fp16.plan", 640, False),
}


def decode_frames(paths: list[Path], limit: int) -> list[torch.Tensor]:
    """Decoded frames as CHW uint8 tensors, cycling through the given videos."""
    frames = []
    for path in paths:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for f in c.decode(s):
                frames.append(torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1)
                              .contiguous())
                if len(frames) >= limit:
                    return frames
    return frames


def decode_fps(paths: list[Path], limit: int) -> float:
    count, t0 = 0, time.perf_counter()
    for path in paths:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for f in c.decode(s):
                f.to_ndarray(format="rgb24")
                count += 1
                if count >= limit:
                    return count / (time.perf_counter() - t0)
    return count / (time.perf_counter() - t0)


def cuda_ms(fn, iters=300, warmup=50) -> dict:
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
    a = np.asarray(times)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}


def roofline_tflops(n=8192, iters=50) -> float:
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(10):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12


def sustained_fps(runner, pre, query, indexer, frames, seed_objectness, seconds=10.0) -> dict:
    """Decoder thread -> GPU preprocess -> engine, back to back, wall clock.

    The merged engine's plan for frame t comes from its own objectness at t-1,
    computed and consumed entirely on the GPU.
    """
    q: queue.Queue = queue.Queue(maxsize=16)
    stop = threading.Event()
    # Page-locked once and reused. Calling pin_memory() per frame allocates fresh
    # page-locked memory every time; when the CPU runs ahead of the GPU those
    # buffers cannot be recycled, and on the 5090 that alone turned a 4.9 ms/frame
    # engine into 7.8 ms/frame (stream_overhead_probe.py).
    pinned = [f.pin_memory() for f in frames]

    def producer():
        i = 0
        while not stop.is_set():
            q.put(pinned[i % len(pinned)])
            i += 1

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    objectness = seed_objectness
    feeds = {"query_embeds": query}

    def step():
        nonlocal objectness
        px = pre.preprocess_tensor(q.get().unsqueeze(0))
        feeds["pixel_values"] = px
        if indexer is not None:
            u, m, a = indexer(objectness)
            feeds.update(unmerged_idx=u, member_patches=m, assign=a)
        out = runner.infer(feeds)
        if indexer is not None:
            objectness = out["objectness"][0].float().clone()

    for _ in range(50):
        step()
    torch.cuda.synchronize()
    count, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        step()
        count += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set()
    return {"fps": count / elapsed, "ms_per_frame": 1000 * elapsed / count, "frames": count}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    videos = [Path(v) for v in args.videos]
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()
    gpu = torch.cuda.get_device_name(0)
    frames = decode_frames(videos, args.frames)
    h, w = frames[0].shape[1:]
    print(f"GPU: {gpu} | {len(frames)} frames of {w}x{h} from {len(videos)} video(s)", flush=True)

    # Correctness gate for the merged engine on real consecutive frames.
    model, _ = load_owlv2("base", device="cuda", dtype=torch.float32)
    grid = WindowGrid(60, "cuda")
    pre960 = GpuOwlv2Preprocessor(960, device="cuda")
    merged = TRTRunner(ws.engine(ENGINES["960 merged50"][0]))
    indexer = MergeIndexer(60, 0.5)
    cosines = []
    for i in range(1, 9):
        prev_px = pre960.preprocess_tensor(frames[i * 20 - 1].unsqueeze(0).cuda()).float()
        px = pre960.preprocess_tensor(frames[i * 20].unsqueeze(0).cuda()).float()
        prior = merged_forward(model, prev_px, query, MergePlan(), grid)["objectness"][0].float()
        ref = merged_forward(model, px, query, MergePlan(early_fraction=0.5, early_scores=prior,
                                                         proportional=False), grid)
        u, m, a = indexer(prior)
        out = merged.infer({"pixel_values": px, "query_embeds": query,
                            "unmerged_idx": u, "member_patches": m, "assign": a})
        torch.cuda.synchronize()
        eng = {k: v.float().cpu().numpy() for k, v in out.items()}
        refn = {k: ref[k].float().cpu().numpy() for k in ("pred_logits", "pred_boxes", "objectness")}
        cosines.append(compare_outputs(eng, refn)["pred_logits"]["cosine_similarity"])
    gate = float(np.mean(cosines))
    print(f"gate: merged engine vs eager fp32 logit cosine {gate:.5f} "
          f"-> {'PASS' if gate >= 0.999 else 'FAIL'}", flush=True)
    del model
    torch.cuda.empty_cache()

    tflops = roofline_tflops()
    dec = decode_fps(videos, 600)
    print(f"fp16 matmul roofline {tflops:.1f} TFLOPS | decode {dec:.0f} FPS (1 thread, "
          f"{w}x{h})", flush=True)

    rows = []
    sample = frames[len(frames) // 2].unsqueeze(0)
    for name, (engine, size, is_merged) in ENGINES.items():
        runner = TRTRunner(ws.engine(engine))
        pre = GpuOwlv2Preprocessor(size, device="cuda")
        px = pre.preprocess_tensor(sample.cuda())
        idx = MergeIndexer(size // 16, 0.5) if is_merged else None
        seed = runner.infer({"pixel_values": px, "query_embeds": query,
                             **(dict(zip(("unmerged_idx", "member_patches", "assign"),
                                         idx(torch.rand(3600, device="cuda"))))
                                if is_merged else {})})["objectness"][0].float().clone()
        plan = idx(seed) if is_merged else None
        feeds = {"pixel_values": px, "query_embeds": query}
        if is_merged:
            feeds.update(zip(("unmerged_idx", "member_patches", "assign"), plan))
        engine_ms = cuda_ms(lambda: runner.infer(feeds))

        def one_frame():
            f = {"pixel_values": pre.preprocess_tensor(sample.cuda()), "query_embeds": query}
            if is_merged:
                f.update(zip(("unmerged_idx", "member_patches", "assign"), idx(seed)))
            runner.infer(f)

        frame_ms = cuda_ms(one_frame)
        sus = sustained_fps(runner, pre, query, idx, frames, seed, args.seconds)

        out = runner.infer(feeds)
        torch.cuda.synchronize()
        arr = {k: v.float().cpu().numpy() for k, v in out.items()}
        t0 = time.perf_counter()
        for _ in range(50):
            postprocess(arr["pred_logits"], arr["pred_boxes"], arr["objectness"],
                        queries.owner, 36, (w, h), 0.05, 100)
        post_ms = (time.perf_counter() - t0) * 1000 / 50

        rows.append({"engine": name, "engine_ms": engine_ms, "frame_ms": frame_ms,
                     "sustained": sus, "cpu_postprocess_ms": post_ms})
        print(f"  {name:<14} engine {engine_ms['median']:6.2f} ms (p95 {engine_ms['p95']:.2f}) | "
              f"frame {frame_ms['median']:6.2f} | sustained {sus['fps']:6.1f} FPS | "
              f"postprocess {post_ms:.2f} ms CPU", flush=True)

    write_report({"gpu": gpu, "frame_size": [w, h], "gate_logit_cosine": gate,
                  "fp16_roofline_tflops": tflops, "decode_fps": dec, "results": rows},
                 ws.result(f"speed_benchmark_{gpu.replace(' ', '_')}.json"))

    print(f"\n{gpu}: fp16 roofline {tflops:.1f} TFLOPS, decode {dec:.0f} FPS\n")
    print(f"{'engine':<15}{'engine ms':>10}{'p95':>7}{'p99':>7}{'frame ms':>10}"
          f"{'sustained FPS':>15}{'ms/frame':>10}{'% of 16 ms':>12}")
    for r in rows:
        s = r["sustained"]
        print(f"{r['engine']:<15}{r['engine_ms']['median']:>10.2f}{r['engine_ms']['p95']:>7.2f}"
              f"{r['engine_ms']['p99']:>7.2f}{r['frame_ms']['median']:>10.2f}{s['fps']:>15.1f}"
              f"{s['ms_per_frame']:>10.2f}{100 * s['ms_per_frame'] / 16:>11.0f}%")


if __name__ == "__main__":
    main()
