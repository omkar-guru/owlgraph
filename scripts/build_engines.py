"""Build every engine the speed comparison uses, from scratch, on this GPU.

TensorRT engines are specific to one GPU, driver and TensorRT version, so a new
machine needs all of them rebuilt. This produces:

``base_fp16.plan``            960px, unmerged
``base768_fp16.plan``         768px, unmerged (position grid resampled at export)
``base640_fp16.plan``         640px, unmerged
``base_merged50_np_fp16.plan`` 960px, 50% of windows merged before block 1,
                              plain attention - the current best configuration

plus the text-query cache the runners read. Existing engines are kept unless
``--rebuild`` is given.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from sggpipeline.ag.classes import AG_OBJECT_CLASSES
from sggpipeline.detect.export_onnx import export_onnx
from sggpipeline.detect.merged_export import export_merged_onnx
from sggpipeline.detect.owlv2 import load_owlv2
from sggpipeline.detect.quantize import to_fp16_onnx
from sggpipeline.detect.trt_build import build_engine
from sggpipeline.pipeline import DEFAULT_VARIANTS, Workspace, load_queries, stage_export


def build(ws: Workspace, name: str, fp32_onnx: Path) -> None:
    t0 = time.perf_counter()
    fp16 = to_fp16_onnx(fp32_onnx, ws.onnx(f"{name}.onnx"))
    build_engine(fp16, ws.engine(f"{name}.plan"), workspace_gb=6.0,
                 timing_cache_path=ws.cache("timing.cache"))
    print(f"built {name}.plan in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    ws = Workspace(Path(args.artifacts))

    def needed(name: str) -> bool:
        return args.rebuild or not ws.engine(f"{name}.plan").exists()

    # 960 unmerged; also writes the text-query cache every runner needs.
    base = next(v for v in DEFAULT_VARIANTS if v.name == "base_fp16")
    if needed("base_fp16") or not ws.cache("queries_base.npz").exists():
        stage_export(base, ws, AG_OBJECT_CLASSES)
        build(ws, "base_fp16", ws.onnx("base_fp32.onnx"))
    num_prompts = len(load_queries(ws, "base").prompts)

    for size in (768, 640):
        name = f"base{size}_fp16"
        if needed(name):
            model, _ = load_owlv2("base", device="cuda", dtype=torch.float32)
            fp32 = export_onnx(model, num_prompts, size, ws.onnx(f"base{size}_fp32.onnx"),
                               native_image_size=960)
            build(ws, name, fp32)
            del model

    name = "base_merged50_np_fp16"
    if needed(name):
        model, _ = load_owlv2("base", device="cuda", dtype=torch.float32)
        fp32 = export_merged_onnx(model, num_prompts, 960, 0.5,
                                  ws.onnx("base_merged50_np_fp32.onnx"), proportional=False)
        build(ws, name, fp32)


if __name__ == "__main__":
    main()
