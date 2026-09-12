"""Measure concrete options for raising fp16 streaming FPS.

Each experiment is a self-contained build+measure so the winners can be adopted
and the losers discarded on evidence rather than intuition.  Run with the
machine otherwise idle: a concurrent CPU job moves these numbers by several ms.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.pipeline import Workspace, load_queries


def bench_engine(runner, feed, n=100, warmup=20) -> dict:
    """Median/p95 of a single engine call, timed with CUDA events."""
    for _ in range(warmup):
        runner.infer(dict(feed))
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        runner.infer(dict(feed))
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    arr = np.asarray(times)
    return {"median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95))}


def bench_throughput(runner, feed, batch: int, seconds: float = 3.0) -> dict:
    """Sustained frames/second with no per-call synchronization.

    This is the number that matters for a stream: back-to-back submission lets
    the GPU stay busy, which a latency-per-call measurement hides.
    """
    stream = torch.cuda.current_stream()
    for name, tensor in feed.items():
        runner.context.set_tensor_address(name, tensor.data_ptr())
    for name, buf in runner.outputs.items():
        runner.context.set_tensor_address(name, buf.data_ptr())

    for _ in range(20):
        runner.context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    calls = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            runner.context.execute_async_v3(stream.cuda_stream)
        calls += 10
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "batch": batch,
        "calls": calls,
        "fps": float(calls * batch / elapsed),
        "ms_per_frame": float(elapsed * 1000 / (calls * batch)),
    }


def make_feed(runner, queries, batch: int, image_size: int) -> dict:
    dtype = runner._torch_dtype("pixel_values")
    return {
        "pixel_values": torch.randn(batch, 3, image_size, image_size,
                                    device="cuda", dtype=dtype),
        "query_embeds": torch.from_numpy(queries.embeds).cuda().to(
            runner._torch_dtype("query_embeds")
        ),
    }


def run(engine_path: Path, queries, batch: int, image_size: int, label: str) -> dict:
    runner = TRTRunner(engine_path)
    feed = make_feed(runner, queries, batch, image_size)
    latency = bench_engine(runner, feed)
    throughput = bench_throughput(runner, feed, batch)
    return {
        "label": label,
        "engine": engine_path.name,
        "engine_mb": round(engine_path.stat().st_size / (1 << 20), 1),
        "device_mem_mb": round(runner.engine.device_memory_size_v2 / (1 << 20), 1),
        **latency,
        **throughput,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--engines", nargs="+", required=True,
                        help="engine filenames under artifacts/engines")
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--checkpoint", default="base")
    args = parser.parse_args()

    ws = Workspace(Path(args.artifacts))
    queries = load_queries(ws, args.checkpoint)

    rows = []
    for name in args.engines:
        path = ws.engine(name)
        if not path.exists():
            print(f"skip missing {path}")
            continue
        for batch in args.batches:
            try:
                rows.append(run(path, queries, batch, args.image_size, f"{name}@b{batch}"))
            except Exception as exc:
                rows.append({"label": f"{name}@b{batch}", "error": str(exc)})

    print(json.dumps(rows, indent=2))
    header = f"{'config':<34}{'lat ms':>9}{'p95':>8}{'ms/frame':>10}{'fps':>8}{'mem MB':>9}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f"{row['label']:<34}  ERROR: {row['error'][:60]}")
            continue
        print(f"{row['label']:<34}{row['median_ms']:>9.2f}{row['p95_ms']:>8.2f}"
              f"{row['ms_per_frame']:>10.2f}{row['fps']:>8.1f}{row['device_mem_mb']:>9.1f}")


if __name__ == "__main__":
    main()
