"""Benchmark selective token merging against no merging and a lower resolution.

For each resolution the same frames run under four merge schedules:

``none``     no merging (reference, same code path as the merged runs)
``prev50``   50% of 2x2 windows merged before block 1, chosen by the previous
             frame's final objectness (dilated)
``b8_75``    75% of windows merged before block 9, chosen by the model's own
             objectness head at block 8
``cascade``  25% before block 1 via the previous frame, then up to 75% at block 8

plus a 768px no-merge run on the same frames, the baseline merging must beat.

Accuracy is AG detection on all 36 classes, uncalibrated for every row. Compute
is exact encoder matmul FLOPs from the measured token schedule. Latency is eager
PyTorch fp16 including the merge/unmerge bookkeeping: indicative of relative
cost, not the deployable TensorRT number.

The previous frame's objectness comes from an unmerged pass over that frame. In
a real stream it would come from that frame's own merged pass, which is slightly
less accurate in merged regions - so ``prev50`` and ``cascade`` are mildly
optimistic here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import (
    Owlv2DetectionGraph,
    load_owlv2,
    postprocess,
    retarget_resolution,
)
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.token_merging import (
    MergePlan,
    WindowGrid,
    encoder_flops,
    merged_forward,
    proportional_attention,
)
from sggpipeline.evaluation.detection_eval import evaluate_detections
from sggpipeline.pipeline import Workspace, load_queries, write_report

from early_objectness_probe import decode_positions

CONFIGS = {
    "none": dict(),
    "prev50": dict(early_fraction=0.50),
    "b8_75": dict(late_block=8, late_total_fraction=0.75),
    "cascade": dict(early_fraction=0.25, late_block=8, late_total_fraction=0.75),
}
RUNS = {960: list(CONFIGS), 640: list(CONFIGS), 768: ["none"]}


def gate_attention() -> float:
    """The folded-bias attention must equal a plain additive-mask attention."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 12, 37, 64, device="cuda") for _ in range(3))
    sizes = torch.randint(1, 5, (37,), device="cuda").float()
    scale = 64 ** -0.5
    ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=sizes.log().view(1, 1, 1, -1), scale=scale)
    got = proportional_attention(q, k, v, sizes.log(), scale)
    return float((ref - got).abs().max())


def gate_forward(model, grid, pre, image, query) -> float:
    """With merging off, the rewritten forward must reproduce the real graph."""
    px = pre([image]).float()
    ref = Owlv2DetectionGraph(model)(px, query)
    got = merged_forward(model, px, query, MergePlan(), grid)
    return max(float((got["pred_logits"] - ref[0]).abs().max()),
               float((got["pred_boxes"] - ref[1]).abs().max()),
               float((got["objectness"] - ref[2]).abs().max()))


def time_config(model, grid, px, query, plan, iters=100, warmup=20) -> float:
    for _ in range(warmup):
        merged_forward(model, px, query, plan, grid)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        merged_forward(model, px, query, plan, grid)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def run(configs: dict, runs: dict, num_frames: int, ag_root: str = "acgdataset",
        artifacts: str = "artifacts", out_name: str = "cascade_benchmark.json") -> list[dict]:
    """Score every (resolution, schedule) pair in ``runs`` on the same frames."""
    needs_prior = {n for n, c in configs.items() if c.get("early_fraction", 0) > 0}
    ws = Workspace(Path(artifacts))
    queries = load_queries(ws, "base")
    query32 = torch.from_numpy(queries.embeds).cuda()

    ag = ActionGenome(root=Path(ag_root), split="test").load()
    classes = ag.classes
    available = [f for f in ag.frames if f.image_path and f.frame_index - 2 >= 0]
    step = max(1, len(available) // (num_frames * 2))
    frames, seen = [], set()
    for f in available[::step]:
        if f.video_id not in seen:  # one frame per video: more scenes, one decode each
            seen.add(f.video_id)
            frames.append(f)
        if len(frames) == num_frames:
            break

    attn_gate = gate_attention()
    print(f"gate proportional attention max|diff|: {attn_gate:.2e}")
    if attn_gate > 1e-4:
        raise SystemExit("Folded-bias attention does not match the reference.")

    first_image = load_image(frames[0].image_path)
    setups = {}
    for size in runs:
        model, _ = load_owlv2("base", device="cuda", dtype=torch.float32)
        if size != 960:
            retarget_resolution(model, size)
        grid = WindowGrid(model.num_patches_height, "cuda")
        pre = GpuOwlv2Preprocessor(size, device="cuda")
        diff = gate_forward(model, grid, pre, first_image, query32)
        print(f"gate {size}px no-merge forward vs real graph max|diff|: {diff:.2e}")
        if diff > 1e-3:
            raise SystemExit(f"Rewritten forward does not match the real graph at {size}px.")
        model.half()
        setups[size] = (model, grid, pre)

    query = query32.half()
    dets = defaultdict(list)
    counts = {}
    videos_dir = Path(ag_root) / "Charades_v1_480"

    for frame in tqdm(frames, desc="frames"):
        prior_pos = frame.frame_index - 2  # 1-based numbering -> previous decode position
        prior_image = decode_positions(videos_dir / frame.video_id, {prior_pos})[prior_pos]
        image = load_image(frame.image_path)
        frame.width, frame.height = image.size

        for size, names in runs.items():
            model, grid, pre = setups[size]
            px = pre([image])
            prior_scores = None
            if needs_prior & set(names):
                prior = merged_forward(model, pre([prior_image]), query, MergePlan(), grid)
                prior_scores = prior["objectness"][0].float()
            for name in names:
                plan = MergePlan(**configs[name], early_scores=prior_scores)
                out = merged_forward(model, px, query, plan, grid)
                counts[(size, name)] = out["token_counts"]
                dets[(size, name)].append(postprocess(
                    out["pred_logits"].float().cpu().numpy(),
                    out["pred_boxes"].float().cpu().numpy(),
                    out["objectness"].float().cpu().numpy(),
                    queries.owner, len(classes), image.size, 0.05, 100))

    # Latency on one fixed frame, with its prior available as in a stream.
    prior_image = decode_positions(videos_dir / frames[0].video_id,
                                   {frames[0].frame_index - 2})[frames[0].frame_index - 2]
    rows = []
    for size, names in runs.items():
        model, grid, pre = setups[size]
        px = pre([first_image])
        prior_scores = merged_forward(model, pre([prior_image]), query, MergePlan(),
                                      grid)["objectness"][0].float()
        base_flops = encoder_flops(counts[(max(runs), "none")])
        for name in names:
            plan = MergePlan(**configs[name], early_scores=prior_scores)
            m = evaluate_detections(frames, dets[(size, name)], classes)
            rows.append({
                "image_size": size, "config": name,
                "tokens_block1": counts[(size, name)][0],
                "tokens_block12": counts[(size, name)][-1],
                "encoder_gflops": encoder_flops(counts[(size, name)]) / 1e9,
                "flops_vs_full": encoder_flops(counts[(size, name)]) / base_flops,
                "eager_fp16_ms": time_config(model, grid, px, query, plan),
                **{k: m[k] for k in ("mAP", "mAP_50", "mAP_75", "mAP_small", "AR_100")},
            })

    write_report({"frames": len(frames), "videos": len(frames), "configs": configs,
                  "gates": {"attention": attn_gate}, "results": rows},
                 ws.result(out_name))

    print(f"\n{len(frames)} frames, one per video, all 36 AG classes, uncalibrated\n")
    h = (f"{'res':>5} {'config':<9}{'tok b1':>7}{'tok b12':>8}{'GFLOPs':>8}{'vsfull':>7}"
         f"{'ms':>7}{'mAP':>8}{'mAP50':>8}{'mAP75':>8}{'APsmall':>8}{'AR100':>8}")
    print(h + "\n" + "-" * len(h))
    for r in rows:
        print(f"{r['image_size']:>5} {r['config']:<9}{r['tokens_block1']:>7}{r['tokens_block12']:>8}"
              f"{r['encoder_gflops']:>8.0f}{r['flops_vs_full']:>7.2f}{r['eager_fp16_ms']:>7.2f}"
              f"{r['mAP']:>8.4f}{r['mAP_50']:>8.4f}{r['mAP_75']:>8.4f}{r['mAP_small']:>8.4f}"
              f"{r['AR_100']:>8.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()
    run(CONFIGS, RUNS, args.frames, args.ag_root, args.artifacts)


if __name__ == "__main__":
    main()
