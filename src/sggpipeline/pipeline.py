"""Stage 1 orchestration: export, compile, benchmark and score OWLv2 variants.

A *variant* is one (checkpoint, precision) pair - the two this stage compares
are ``base`` at fp16 and ``large`` at int8.  Each variant owns an ONNX graph, a
TensorRT engine and a results record, all keyed by the same name so artifacts
never cross between variants.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .ag.classes import build_prompt_index
from .detect import owlv2 as owl
from .detect.export_onnx import export_onnx
from .detect.preprocess import Owlv2Preprocessor


@dataclass(slots=True)
class Variant:
    """One checkpoint compiled at one precision."""

    name: str
    checkpoint: str
    precision: str  # "fp16" | "int8" | "fp32"

    @property
    def onnx_fp32(self) -> str:
        return f"{self.checkpoint}_fp32.onnx"

    @property
    def onnx_final(self) -> str:
        return f"{self.checkpoint}_{self.precision}.onnx"

    @property
    def engine(self) -> str:
        return f"{self.name}.plan"


DEFAULT_VARIANTS = (
    Variant(name="base_fp16", checkpoint="base", precision="fp16"),
    Variant(name="base_fp8", checkpoint="base", precision="fp8"),
    Variant(name="large_int8", checkpoint="large", precision="int8"),
)


@dataclass
class Workspace:
    """Filesystem layout for build artifacts and results."""

    root: Path = Path("artifacts")

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for sub in ("onnx", "engines", "results", "cache"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def onnx(self, name: str) -> Path:
        return self.root / "onnx" / name

    def engine(self, name: str) -> Path:
        return self.root / "engines" / name

    def result(self, name: str) -> Path:
        return self.root / "results" / name

    def cache(self, name: str) -> Path:
        return self.root / "cache" / name


@dataclass
class TextQueries:
    """The AG prompt set, embedded once and shared by every variant."""

    prompts: list[str]
    owner: np.ndarray
    embeds: np.ndarray
    classes: tuple[str, ...]

    @property
    def num_prompts(self) -> int:
        return len(self.prompts)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def build_text_queries(model, processor, classes: tuple[str, ...], device="cuda") -> TextQueries:
    """Run the text tower once for the whole vocabulary."""
    prompts, owner = build_prompt_index(classes)
    embeds = owl.encode_text_queries(model, processor, prompts, device)
    return TextQueries(
        prompts=prompts,
        owner=np.asarray(owner, dtype=np.int64),
        embeds=embeds.cpu().numpy().astype(np.float32),
        classes=classes,
    )


def stage_export(variant: Variant, ws: Workspace, classes, device="cuda") -> dict:
    """Load the checkpoint, embed the vocabulary and write the fp32 ONNX graph."""
    model, processor = owl.load_owlv2(variant.checkpoint, device=device, dtype=torch.float32)
    pre = Owlv2Preprocessor(processor)
    queries = build_text_queries(model, processor, classes, device)

    np.savez(
        ws.cache(f"queries_{variant.checkpoint}.npz"),
        embeds=queries.embeds,
        owner=queries.owner,
        prompts=np.array(queries.prompts, dtype=object),
    )

    t0 = time.perf_counter()
    path = export_onnx(
        model,
        num_prompts=queries.num_prompts,
        image_size=pre.image_size,
        out_path=ws.onnx(variant.onnx_fp32),
        device=device,
    )
    return {
        "variant": variant.name,
        "checkpoint": owl.resolve_model_id(variant.checkpoint),
        "image_size": pre.image_size,
        "num_prompts": queries.num_prompts,
        "num_classes": queries.num_classes,
        "onnx_path": str(path),
        "onnx_size_mb": round(path.stat().st_size / (1 << 20), 2),
        "export_seconds": round(time.perf_counter() - t0, 2),
    }


def load_queries(ws: Workspace, checkpoint: str) -> TextQueries:
    """Re-read the cached text embeddings written by :func:`stage_export`."""
    data = np.load(ws.cache(f"queries_{checkpoint}.npz"), allow_pickle=True)
    prompts = list(data["prompts"])
    return TextQueries(
        prompts=prompts,
        owner=data["owner"],
        embeds=data["embeds"],
        classes=tuple(),
    )


def build_preprocessor(processor, device: str = "cuda", fast: bool = True):
    """Pick the GPU preprocessor by default, falling back to the stock one.

    The stock ``Owlv2ImageProcessor`` spends ~70% of its time in a CPU
    anti-aliasing convolution and costs several times the fp16 engine itself.
    The GPU path reproduces it to ~1e-6, so the slow one is kept only as the
    reference that claim is checked against.
    """
    pre = Owlv2Preprocessor(processor)
    if not fast:
        return pre
    from .detect.fast_preprocess import GpuOwlv2Preprocessor

    return GpuOwlv2Preprocessor(image_size=pre.image_size, device=device)


def write_report(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path
