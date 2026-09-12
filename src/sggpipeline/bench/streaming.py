"""End-to-end streaming throughput, with decode and preprocessing overlapped.

Benchmarking the engine alone answers "how fast is the detector", not "how fast
can this ingest video".  A real stream also has to decode frames and preprocess
them, and those stages can run *while* the GPU is busy rather than before it.

The pipeline is two stages deep:

* a producer thread decodes frames and converts them to pinned uint8 tensors
  (pure CPU work, fully overlappable with the GPU);
* the consumer preprocesses and runs the engine.

Measuring the serial sum of stages overstates the cost of a pipelined system;
measuring the engine alone understates it.  Both are reported here.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(slots=True)
class StreamingStats:
    label: str
    frames: int
    seconds: float
    fps: float
    ms_per_frame: float
    decode_starved_pct: float

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def _decode_worker(
    video_path: Path, out: queue.Queue, stride: int, limit: int | None, stop: threading.Event
) -> None:
    """Decode frames to pinned CHW uint8 tensors, ready for a non-blocking copy."""
    import av

    produced = 0
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, frame in enumerate(container.decode(stream)):
                if stop.is_set():
                    break
                if stride > 1 and index % stride:
                    continue
                array = frame.to_ndarray(format="rgb24")
                tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
                # Pinned memory is what makes the host-to-device copy async.
                out.put(tensor.pin_memory())
                produced += 1
                if limit is not None and produced >= limit:
                    break
    finally:
        out.put(None)


def benchmark_streaming(
    runner,
    preprocessor,
    video_path: Path,
    query_embeds: np.ndarray,
    label: str,
    limit: int | None = 300,
    stride: int = 1,
    queue_size: int = 8,
    device: str = "cuda",
) -> StreamingStats:
    """Sustained frames/second for decode -> preprocess -> engine."""
    frames_queue: queue.Queue = queue.Queue(maxsize=queue_size)
    stop = threading.Event()
    worker = threading.Thread(
        target=_decode_worker,
        args=(Path(video_path), frames_queue, stride, limit, stop),
        daemon=True,
    )

    dev = torch.device(device)
    query = torch.from_numpy(query_embeds).to(dev)
    processed = 0
    starved = 0

    worker.start()
    # Prime the queue so startup decode latency is not counted as throughput.
    first = frames_queue.get()
    if first is None:
        return StreamingStats(label, 0, 0.0, 0.0, float("nan"), 0.0)

    torch.cuda.synchronize(dev)
    start = time.perf_counter()
    current = first
    while current is not None:
        batch = preprocessor.preprocess_tensor(current.unsqueeze(0))
        runner.infer({"pixel_values": batch, "query_embeds": query})
        processed += 1
        if frames_queue.empty():
            starved += 1
        current = frames_queue.get()
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - start

    stop.set()
    worker.join(timeout=2.0)

    return StreamingStats(
        label=label,
        frames=processed,
        seconds=round(elapsed, 3),
        fps=round(processed / elapsed, 2) if elapsed > 0 else float("nan"),
        ms_per_frame=round(elapsed * 1000 / processed, 3) if processed else float("nan"),
        # How often the GPU had to wait on the decoder: if this is high the
        # bottleneck is CPU decode, not the detector.
        decode_starved_pct=round(100.0 * starved / processed, 1) if processed else 0.0,
    )


def benchmark_decode_only(
    video_path: Path, limit: int | None = 300, stride: int = 1
) -> StreamingStats:
    """Decode throughput on its own, to see whether it caps the pipeline."""
    import av

    count = 0
    start = time.perf_counter()
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if stride > 1 and index % stride:
                continue
            frame.to_ndarray(format="rgb24")
            count += 1
            if limit is not None and count >= limit:
                break
    elapsed = time.perf_counter() - start
    return StreamingStats(
        label="decode-only",
        frames=count,
        seconds=round(elapsed, 3),
        fps=round(count / elapsed, 2) if elapsed else float("nan"),
        ms_per_frame=round(elapsed * 1000 / count, 3) if count else float("nan"),
        decode_starved_pct=0.0,
    )
