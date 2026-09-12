"""Per-frame latency measurement.

Three properties make these numbers comparable across the variants:

* CUDA events, not wall clock, so asynchronous launches are not mistimed.
* A warmup phase that is discarded, so one-off autotuning and allocation do not
  land in the reported distribution.
* Median and p95 rather than a mean, because a mean over a long-tailed latency
  distribution hides exactly the stalls that matter for streaming video.

Preprocessing is timed separately from the engine.  It runs on CPU and is shared
by every variant, so folding it in would compress the measured gap between them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass(slots=True)
class LatencyStats:
    """Timing summary for one configuration, all values in milliseconds."""

    label: str
    frames: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    fps_from_median: float
    peak_gpu_mem_mb: float

    def as_dict(self) -> dict:
        return asdict(self)


def _summarize(label: str, times_ms: list[float], peak_mem_mb: float) -> LatencyStats:
    arr = np.asarray(times_ms, dtype=np.float64)
    median = float(np.median(arr))
    return LatencyStats(
        label=label,
        frames=int(arr.size),
        mean_ms=float(arr.mean()),
        median_ms=median,
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        std_ms=float(arr.std()),
        fps_from_median=float(1000.0 / median) if median > 0 else float("nan"),
        peak_gpu_mem_mb=round(peak_mem_mb, 2),
    )


def benchmark_engine(
    runner,
    pixel_values: np.ndarray,
    query_embeds: np.ndarray,
    label: str,
    warmup: int = 20,
    iterations: int = 200,
    device: str = "cuda",
) -> LatencyStats:
    """Time the detector's forward pass, one frame at a time.

    Frames are cycled from a real preprocessed batch rather than regenerated, so
    the measurement isolates compute instead of host-side data preparation.
    """
    dev = torch.device(device)
    num_frames = pixel_values.shape[0]
    query = torch.from_numpy(np.ascontiguousarray(query_embeds))
    frames = [
        torch.from_numpy(np.ascontiguousarray(pixel_values[i : i + 1]))
        for i in range(num_frames)
    ]
    # Resident on device: host-to-device copies are measured separately.
    frames = [f.to(dev) for f in frames]
    query = query.to(dev)

    for i in range(warmup):
        runner.infer({"pixel_values": frames[i % num_frames], "query_embeds": query})
    torch.cuda.synchronize(dev)

    torch.cuda.reset_peak_memory_stats(dev)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []

    for i in range(iterations):
        feed = {"pixel_values": frames[i % num_frames], "query_embeds": query}
        start.record()
        runner.infer(feed)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    peak = torch.cuda.max_memory_allocated(dev) / (1 << 20)
    return _summarize(label, times_ms, peak)


def benchmark_preprocess(
    preprocessor, images: list, label: str, warmup: int = 5, iterations: int = 50
) -> LatencyStats:
    """Time CPU-side preprocessing, reported alongside but never inside engine time."""
    import time

    # A GPU preprocessor returns before its kernels finish, so the timer has to
    # synchronize or it measures launch overhead instead of the work.
    needs_sync = torch.cuda.is_available()

    for i in range(warmup):
        preprocessor([images[i % len(images)]])
    if needs_sync:
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        preprocessor([images[i % len(images)]])
        if needs_sync:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return _summarize(label, times_ms, 0.0)


def benchmark_h2d(
    pixel_values: np.ndarray, label: str, iterations: int = 100, device: str = "cuda"
) -> LatencyStats:
    """Time the host-to-device copy of one preprocessed frame.

    At 960x960x3 fp32 this is ~11MB per frame and is a real part of a streaming
    budget, so it is measured rather than assumed negligible.
    """
    dev = torch.device(device)
    host = torch.from_numpy(np.ascontiguousarray(pixel_values[:1])).pin_memory()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(10):
        host.to(dev, non_blocking=True)
    torch.cuda.synchronize(dev)

    times_ms: list[float] = []
    for _ in range(iterations):
        start.record()
        host.to(dev, non_blocking=True)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return _summarize(label, times_ms, 0.0)
