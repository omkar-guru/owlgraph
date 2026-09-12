"""Frame loading and OWLv2 preprocessing.

The HF ``Owlv2Processor`` is treated as the authority on preprocessing: it
rescales, pads the image to a square with grey at the bottom/right, resizes to
the checkpoint's native resolution and applies CLIP normalization.  Re-deriving
that by hand is a classic way to lose mAP invisibly, so it is not re-derived -
but it *is* timed separately from the engine, because it runs on CPU and would
otherwise be silently attributed to the detector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> Image.Image:
    """Read one frame as RGB."""
    with Image.open(path) as im:
        return im.convert("RGB")


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) without decoding pixel data."""
    with Image.open(path) as im:
        return im.size


class Owlv2Preprocessor:
    """Turns PIL frames into the exact tensor the exported graph expects."""

    def __init__(self, processor):
        self.processor = processor

    @property
    def image_size(self) -> int:
        """Square side length this checkpoint's vision tower consumes."""
        size = self.processor.image_processor.size
        # Depending on the transformers version this is a plain dict or a
        # SizeDict; both expose height/shortest_edge, one by key and one by attr.
        for key in ("height", "shortest_edge"):
            value = (
                size.get(key) if isinstance(size, dict) else getattr(size, key, None)
            )
            if value:
                return int(value)
        raise RuntimeError(f"Could not determine image size from processor size={size!r}")

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        """Return ``(B, 3, S, S)`` float32 pixel values."""
        out = self.processor(images=images, return_tensors="np")
        return np.ascontiguousarray(out["pixel_values"].astype(np.float32))


def iter_video_frames(video_path: Path, stride: int = 1, limit: int | None = None):
    """Decode frames from a video file, for latency runs on real footage.

    Benchmarking on decoded video rather than random noise matters: activation
    statistics drive quantized kernel selection, and noise is not representative.
    """
    import av

    count = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i % stride:
                continue
            yield frame.to_image()
            count += 1
            if limit is not None and count >= limit:
                return
