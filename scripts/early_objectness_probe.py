"""Does OWLv2 carry a usable objectness signal before its final layer?

Motivation: merge background tokens early to save compute, while keeping object
tokens at full resolution. That only works if something can tell, *before* the
last layer, which tokens matter. OWLv2 computes objectness once, at the end, so
this probes candidate early signals:

``after_blockK``
    The final objectness head applied to the features after transformer block K
    (0 = patch embeddings). The head was never trained on these features, so this
    measures whether it transfers, not whether it was designed to.
``prevN`` / ``prevN_dilated``
    The *final* objectness of the video frame N frames earlier, optionally
    grown by one window so objects that moved slightly stay protected. Free in a
    stream, because that frame was already processed.
``random``
    The floor any useful score must beat.

The decision metric is **object survival**. Merging is simulated as the proposed
rule: tokens are grouped into non-overlapping 2x2 windows, each window takes the
*max* score of its tokens (so one important token protects the window), and the
lowest-scoring X% of windows are merged. A ground-truth object "survives" if at
least one token that could still detect it - predicted box IoU >= 0.5 with the
object and class score >= 0.05 at the final layer - sits in an unmerged window.

This is a ceiling estimate: it holds token features fixed, whereas real merging
also changes the surviving tokens' features. A signal that fails here fails in
practice; one that passes still has to be tested with actual merging.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import (
    Owlv2DetectionGraph,
    load_owlv2,
    preprocess_sizes,
    retarget_resolution,
)
from sggpipeline.detect.preprocess import load_image
from sggpipeline.pipeline import Workspace, load_queries, write_report

MERGE_FRACTIONS = (0.25, 0.50, 0.75)
MATCH_IOU = 0.5
DETECT_SCORE = 0.05
PRIOR_OFFSETS = (1, 5)  # ~33 ms and ~167 ms earlier at 30 fps
STUFF_CLASSES = {"floor", "doorway", "window", "light", "door"}


@torch.no_grad()
def layerwise(model, pixel_values, query, with_heads: bool = True) -> dict:
    """Objectness after every block, plus the final boxes and class logits.

    Replicates ``image_embedder``'s post-processing (post-layernorm, class-token
    product, layer norm) on each intermediate hidden state, so the final entry is
    exactly what the real head sees - checked against the graph in ``main``.
    """
    vision = model.owlv2.vision_model
    out = vision(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    side = model.num_patches_height

    scores, feats = [], None
    for hidden in out.hidden_states:
        e = vision.post_layernorm(hidden)
        cls = torch.broadcast_to(e[:, :1, :], e[:, :-1].shape)
        feats = model.layer_norm(e[:, 1:, :] * cls)
        scores.append(model.objectness_predictor(feats)[0].float())

    result = {"layers": torch.stack(scores)}  # (num_blocks + 1, N)
    if with_heads:
        grid = feats.reshape(1, side, side, -1)
        result["boxes"] = model.box_predictor(feats, grid)[0].float()
        result["logits"] = model.class_predictor(feats, query.unsqueeze(0), None)[0][0].float()
    return result


def window_scores(token_scores: torch.Tensor, side: int, dilate: bool = False) -> torch.Tensor:
    """Max-pool token scores into 2x2 windows; optionally grow by one window."""
    grid = token_scores.reshape(1, 1, side, side)
    windows = F.max_pool2d(grid, kernel_size=2, stride=2)
    if dilate:
        windows = F.max_pool2d(windows, kernel_size=3, stride=1, padding=1)
    return windows.flatten()


def token_to_window(side: int, device) -> torch.Tensor:
    idx = torch.arange(side * side, device=device)
    return (idx // side // 2) * (side // 2) + (idx % side) // 2


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


def decode_positions(video_path: Path, positions: set[int]) -> dict:
    """Decode a video once, keeping only the requested 0-based frame positions."""
    wanted, last = set(positions), max(positions)
    frames = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for pos, frame in enumerate(container.decode(stream)):
            if pos in wanted:
                frames[pos] = frame.to_image()
            if pos >= last:
                break
    return frames


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--image-size", type=int, default=960)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    queries = load_queries(ws, "base")
    model, _ = load_owlv2("base", device="cuda", dtype=torch.float32)
    if args.image_size != 960:
        retarget_resolution(model, args.image_size)
    side = model.num_patches_height
    pre = GpuOwlv2Preprocessor(args.image_size, device="cuda")
    query = torch.from_numpy(queries.embeds).cuda()
    owner = torch.as_tensor(queries.owner, device="cuda")
    tok_win = token_to_window(side, "cuda")
    num_windows = (side // 2) ** 2

    ag = ActionGenome(root=Path(args.ag_root), split="test").load()
    classes = ag.classes
    available = [f for f in ag.frames if f.image_path]
    step = max(1, len(available) // args.frames)
    frames = [f for f in available[::step][: args.frames]
              if f.frame_index - 1 - max(PRIOR_OFFSETS) >= 0]

    videos_dir = Path(args.ag_root) / "Charades_v1_480"
    by_video = defaultdict(list)
    for f in frames:
        by_video[f.video_id].append(f)

    # Correctness gate: the per-layer replication must reproduce the real head.
    first = frames[0]
    px0 = pre([load_image(first.image_path)]).float()
    ref = Owlv2DetectionGraph(model)(px0, query)
    probe = layerwise(model, px0, query)
    gate = {
        "objectness": float((probe["layers"][-1] - ref[2][0]).abs().max()),
        "boxes": float((probe["boxes"] - ref[1][0]).abs().max()),
        "logits": float((probe["logits"] - ref[0][0]).abs().max()),
    }
    print(f"gate max|diff| vs real head: {gate}")
    if max(gate.values()) > 1e-3:
        raise SystemExit("Per-layer replication does not match the real head; aborting.")

    num_blocks = probe["layers"].shape[0] - 1
    methods = [f"after_block{k}" for k in range(num_blocks + 1)]
    methods += [f"prev{o}{s}" for o in PRIOR_OFFSETS for s in ("", "_dilated")]
    methods += ["random"]

    survived = {m: {x: 0 for x in MERGE_FRACTIONS} for m in methods}
    per_class = {m: defaultdict(lambda: [0, 0]) for m in methods}  # at 50%
    rho = defaultdict(list)
    detectable_total = undetectable_total = 0
    gen = torch.Generator(device="cuda").manual_seed(0)

    for video_id, vframes in tqdm(sorted(by_video.items()), desc="probe"):
        positions = {f.frame_index - 1 - o for f in vframes for o in PRIOR_OFFSETS}
        priors = decode_positions(videos_dir / video_id, positions)

        for frame in vframes:
            image = load_image(frame.image_path)
            r = layerwise(model, pre([image]).float(), query)
            final = r["layers"][-1]

            # Which tokens could still detect each GT object at the final layer.
            scale = preprocess_sizes(*image.size)
            cx, cy, w, h = r["boxes"].unbind(-1)
            tok_boxes = torch.stack([(cx - w / 2) * scale, (cy - h / 2) * scale,
                                     (cx + w / 2) * scale, (cy + h / 2) * scale], -1)
            prompt_scores = torch.sigmoid(r["logits"])
            class_scores = torch.zeros(prompt_scores.shape[0], len(classes), device="cuda")
            class_scores = class_scores.scatter_reduce(
                1, owner.expand_as(prompt_scores), prompt_scores, reduce="amax")

            gt = torch.as_tensor(frame.boxes, device="cuda")
            labels = torch.as_tensor(frame.labels, device="cuda")
            cand = (box_iou(gt, tok_boxes) >= MATCH_IOU) & (class_scores[:, labels].T >= DETECT_SCORE)
            detectable = cand.any(dim=1)
            detectable_total += int(detectable.sum())
            undetectable_total += int((~detectable).sum())
            if not detectable.any():
                continue
            cand, labels_d = cand[detectable], labels[detectable]

            token_scores = {f"after_block{k}": r["layers"][k] for k in range(num_blocks + 1)}
            token_scores["random"] = torch.rand(final.shape, device="cuda", generator=gen)
            for o in PRIOR_OFFSETS:
                prior_img = priors.get(frame.frame_index - 1 - o)
                if prior_img is None:
                    raise RuntimeError(f"Missing prior frame for {frame.frame_key}")
                prior = layerwise(model, pre([prior_img]).float(), query, with_heads=False)
                token_scores[f"prev{o}"] = prior["layers"][-1]

            for method in methods:
                base = method.removesuffix("_dilated")
                ws_ = window_scores(token_scores[base], side, dilate=method.endswith("_dilated"))
                order = ws_.argsort()
                if base != "random":
                    rho[method].append(spearman(token_scores[base], final))
                for x in MERGE_FRACTIONS:
                    merged = torch.zeros(num_windows, dtype=torch.bool, device="cuda")
                    merged[order[: int(round(x * num_windows))]] = True
                    alive = (cand & ~merged[tok_win][None, :]).any(dim=1)
                    survived[method][x] += int(alive.sum())
                    if x == 0.50:
                        for lab, ok in zip(labels_d.tolist(), alive.tolist(), strict=True):
                            per_class[method][lab][0] += int(ok)
                            per_class[method][lab][1] += 1

    total = detectable_total
    rows = []
    for m in methods:
        row = {"method": m,
               "spearman_vs_final": float(np.mean(rho[m])) if rho[m] else None,
               **{f"survival@{int(x*100)}": survived[m][x] / total for x in MERGE_FRACTIONS}}
        stuff = [v for k, v in per_class[m].items() if classes[k] in STUFF_CLASSES]
        things = [v for k, v in per_class[m].items() if classes[k] not in STUFF_CLASSES]
        # None, not 0, when a group has no objects: an empty group did not fail.
        row["stuff_survival@50"] = (sum(a for a, _ in stuff) / sum(b for _, b in stuff)
                                    if stuff else None)
        row["thing_survival@50"] = (sum(a for a, _ in things) / sum(b for _, b in things)
                                    if things else None)
        row["per_class_survival@50"] = {classes[k]: v[0] / v[1] for k, v in per_class[m].items()}
        rows.append(row)

    write_report({"image_size": args.image_size, "frames": len(frames),
                  "videos": len(by_video), "detectable_objects": detectable_total,
                  "undetectable_at_baseline": undetectable_total, "gate": gate,
                  "window": "2x2, max-pooled", "results": rows},
                 ws.result(f"early_objectness_{args.image_size}.json"))

    print(f"\n{len(frames)} frames / {len(by_video)} videos @ {args.image_size}px; "
          f"{detectable_total} detectable objects ({undetectable_total} not detectable even unmerged)")
    print("Tokens removed by merging X% of 2x2 windows: 0.75*X "
          "(25%->19%, 50%->38%, 75%->56%)\n")
    h = (f"{'method':<22}{'rho':>7}{'surv@25':>9}{'surv@50':>9}{'surv@75':>9}"
         f"{'things@50':>11}{'stuff@50':>10}")
    print(h + "\n" + "-" * len(h))
    for r_ in rows:
        rho_s = f"{r_['spearman_vs_final']:.3f}" if r_["spearman_vs_final"] is not None else "   -"
        print(f"{r_['method']:<22}{rho_s:>7}{r_['survival@25']:>9.3f}{r_['survival@50']:>9.3f}"
              f"{r_['survival@75']:>9.3f}{_fmt(r_['thing_survival@50']):>11}{_fmt(r_['stuff_survival@50']):>10}")


if __name__ == "__main__":
    main()
