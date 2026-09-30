"""Where does sustained streaming lose time relative to one-frame timing?

The unmerged 960 engine sustains ~8 ms/frame on the 5090 against a ~5 ms
per-frame time, reproducibly, while the merged engine shows no such gap. The
suspect is the benchmark harness: its producer calls ``pin_memory()`` on every
frame, a fresh page-locked allocation. When the CPU runs ahead of the GPU - as it
does in the unmerged loop, which never waits on the GPU - those buffers cannot be
recycled yet, so every frame pays a new allocation. The merged loop happens to
wait once per frame and so stays in step.

Same engine, three ways of feeding frames:

``A per-frame pin``  what speed_benchmark.py does
``B pre-pinned``     every frame page-locked once, buffers reused
``C on-GPU``         frames already resident on the device (the ceiling)

If B closes most of the A-to-C gap, the gap is the harness, not the engine.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import av
import torch

from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.merged_export import MergeIndexer
from sggpipeline.detect.stream import PLAN_INPUTS
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.pipeline import Workspace, load_queries, write_report


def decode(path: Path, limit: int) -> list[torch.Tensor]:
    out = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            out.append(torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous())
            if len(out) >= limit:
                break
    return out


def run(runner, pre, query, indexer, source, seconds: float) -> float:
    """Frames/second with a producer thread feeding ``source(i)`` through a queue."""
    q: queue.Queue = queue.Queue(maxsize=16)
    stop = threading.Event()

    def producer():
        i = 0
        while not stop.is_set():
            q.put(source(i))
            i += 1

    threading.Thread(target=producer, daemon=True).start()
    prior = None
    feeds = {"query_embeds": query}

    def step():
        nonlocal prior
        feeds["pixel_values"] = pre.preprocess_tensor(q.get().unsqueeze(0))
        if indexer is not None:
            plan = indexer(prior if prior is not None else torch.rand(3600, device="cuda"))
            feeds.update(zip(PLAN_INPUTS, plan))
        out = runner.infer(feeds)
        if indexer is not None:
            prior = out["objectness"][0].float().clone()

    for _ in range(50):
        step()
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        step()
        n += 1
    torch.cuda.synchronize()
    stop.set()
    return n / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--video", required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()
    ws = Workspace(Path(args.artifacts))
    query = torch.from_numpy(load_queries(ws, "base").embeds).cuda()
    pre = GpuOwlv2Preprocessor(960, device="cuda")
    frames = decode(Path(args.video), 300)
    pinned = [f.pin_memory() for f in frames]
    on_gpu = [f.cuda() for f in frames]
    sources = {
        "A per-frame pin": lambda i: frames[i % len(frames)].pin_memory(),
        "B pre-pinned": lambda i: pinned[i % len(pinned)],
        "C on-GPU": lambda i: on_gpu[i % len(on_gpu)],
    }
    rows = {}
    for engine, is_merged in (("base_fp16.plan", False), ("base_merged50_np_fp16.plan", True)):
        runner = TRTRunner(ws.engine(engine))
        indexer = MergeIndexer(60, 0.5) if is_merged else None
        rows[engine] = {name: run(runner, pre, query, indexer, src, args.seconds)
                        for name, src in sources.items()}
        print(engine, {k: f"{v:.1f} FPS ({1000 / v:.2f} ms)" for k, v in rows[engine].items()},
              flush=True)
    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    write_report(rows, ws.result(f"stream_overhead_{gpu}.json"))


if __name__ == "__main__":
    main()
