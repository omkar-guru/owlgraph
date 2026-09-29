"""Kernel-level profile: why is the merged engine only 20% faster?

The prev50 engine does 48% less encoder work than the unmerged 960 engine but
runs only 20% faster; effective throughput fell from ~61 to ~40 TFLOPS. A
TensorRT layer profile would mostly show a few large fused blocks, so this
records every CUDA kernel the engines launch (via CUPTI, through
``torch.profiler``) and compares them by category.

Categories are assigned by kernel-name patterns. Anything unmatched is listed
under ``other`` with its names shown, so a wrong guess is visible rather than
silently absorbed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.merged_export import MergeIndexer
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.pipeline import Workspace, load_queries, write_report

ITERS = 50
CATEGORIES = [
    ("attention", r"fmha|mha|flash|attention|softmax"),
    ("gemm", r"gemm|xmma|cutlass|matmul|sm\d+_|ampere|hopper|blackwell|cublas"),
    ("norm", r"layernorm|layer_norm|norm"),
    ("gather/scatter", r"gather|scatter|index"),
    ("copy/concat/reformat", r"copy|concat|slice|reformat|transpose|permute|shuffle|memcpy|memset"),
    ("conv", r"conv|implicit"),
    ("pointwise (myelin)", r"__myl|myelin|pointwise|elementwise|eltwise|reduce"),
]


def category(name: str) -> str:
    low = name.lower()
    for cat, pattern in CATEGORIES:
        if re.search(pattern, low):
            return cat
    return "other"


def kernel_times(fn) -> list[tuple[str, float, float]]:
    """(kernel name, ms per iteration, launches per iteration) for GPU kernels."""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        total_us = getattr(evt, "device_time_total", None) or getattr(evt, "cuda_time_total", 0)
        if total_us <= 0:
            continue
        rows.append((evt.key, total_us / 1000 / ITERS, evt.count / ITERS))
    return sorted(rows, key=lambda r: -r[1])


def main() -> None:
    ws = Workspace(Path("artifacts"))
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()
    frame = next(f for f in ActionGenome(root=Path("acgdataset"), split="test")
                 .load(max_frames=20).frames if f.image_path)
    px = GpuOwlv2Preprocessor(960, device="cuda")([load_image(frame.image_path)])

    unmerged = TRTRunner(ws.engine("base_fp16.plan"))
    merged = TRTRunner(ws.engine("base_merged50_fp16.plan"))
    prior = unmerged.infer({"pixel_values": px, "query_embeds": query})["objectness"][0].float()
    torch.cuda.synchronize()
    u_idx, members, assign = MergeIndexer(60, 0.5)(prior)

    engines = {
        "960 unmerged": lambda: unmerged.infer({"pixel_values": px, "query_embeds": query}),
        "960 merged50": lambda: merged.infer({"pixel_values": px, "query_embeds": query,
                                              "unmerged_idx": u_idx, "member_patches": members,
                                              "assign": assign}),
    }

    report = {}
    by_cat = {}
    for name, fn in engines.items():
        rows = kernel_times(fn)
        cats = defaultdict(lambda: [0.0, 0.0])
        others = []
        for kname, ms, n in rows:
            c = category(kname)
            cats[c][0] += ms
            cats[c][1] += n
            if c == "other":
                others.append((kname, ms))
        by_cat[name] = dict(cats)
        report[name] = {"total_kernel_ms": sum(r[1] for r in rows),
                        "kernels_per_iter": sum(r[2] for r in rows),
                        "by_category": {k: {"ms": v[0], "launches": v[1]} for k, v in cats.items()},
                        "top_kernels": [{"name": k, "ms": m, "launches": n} for k, m, n in rows[:25]],
                        "uncategorised": [{"name": k, "ms": m} for k, m in others[:15]]}

    write_report(report, ws.result("engine_kernel_profile.json"))

    names = list(engines)
    cats = sorted({c for n in names for c in by_cat[n]},
                  key=lambda c: -max(by_cat[n].get(c, [0])[0] for n in names))
    print(f"GPU kernel time per inference ({ITERS} iterations)\n")
    print(f"{'category':<24}" + "".join(f"{n + ' ms':>16}{'launches':>10}" for n in names)
          + f"{'merged/unmerged':>17}")
    for c in cats:
        vals = [by_cat[n].get(c, [0.0, 0.0]) for n in names]
        ratio = f"{vals[1][0] / vals[0][0]:.2f}" if vals[0][0] > 0 else "new"
        print(f"{c:<24}" + "".join(f"{v[0]:>16.3f}{v[1]:>10.0f}" for v in vals) + f"{ratio:>17}")
    tot = [report[n]["total_kernel_ms"] for n in names]
    print(f"{'TOTAL':<24}{tot[0]:>16.3f}{report[names[0]]['kernels_per_iter']:>10.0f}"
          f"{tot[1]:>16.3f}{report[names[1]]['kernels_per_iter']:>10.0f}{tot[1] / tot[0]:>17.2f}")

    for n in names:
        print(f"\ntop kernels - {n}")
        for k in report[n]["top_kernels"][:12]:
            print(f"  {k['ms']:7.3f} ms  x{k['launches']:<4.0f} [{category(k['name'])}] {k['name'][:95]}")


if __name__ == "__main__":
    main()
