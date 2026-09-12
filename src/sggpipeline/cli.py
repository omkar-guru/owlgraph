"""Command line entry point for the Stage 1 detector comparison.

Typical order::

    sgg export   --variant base_fp16      # checkpoint -> fp32 ONNX
    sgg prepare  --variant base_fp16      # fp32 ONNX -> fp16 / int8 ONNX
    sgg build    --variant base_fp16      # ONNX -> TensorRT engine
    sgg bench    --variant base_fp16      # per-frame latency
    sgg evaluate --variant base_fp16      # Action Genome mAP

``prepare`` for an int8 variant needs ``--ag-root``, because its calibration
frames must come from the evaluation domain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _variant_by_name(name: str):
    from .pipeline import DEFAULT_VARIANTS

    for variant in DEFAULT_VARIANTS:
        if variant.name == name:
            return variant
    known = ", ".join(v.name for v in DEFAULT_VARIANTS)
    raise SystemExit(f"Unknown variant {name!r}. Known variants: {known}")


def _echo(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=float))


# -- ag ----------------------------------------------------------------------
def cmd_validate_ag(args) -> None:
    """Inspect the downloaded annotations before trusting any score from them."""
    from .ag.dataset import ActionGenome

    ag = ActionGenome(root=Path(args.ag_root), split=args.split).load(
        max_frames=args.max_frames
    )
    report = ag.validate()
    counts = report.pop("label_counts")
    _echo(report)
    print("\nGround-truth boxes per class:")
    for name, count in counts.items():
        print(f"  {name:<22} {count}")


def cmd_extract_frames(args) -> None:
    """Decode the annotated keyframes out of the Charades videos."""
    from .ag.dataset import ActionGenome
    from .ag.extract_frames import extract_all

    ag_root = Path(args.ag_root)
    ag = ActionGenome(root=ag_root, split=args.split)
    frame_list = ag.annotation_dir / "frame_list.txt"
    if not frame_list.exists():
        raise SystemExit(f"frame_list.txt not found under {ag.annotation_dir}")

    videos = None
    if args.split != "all":
        # Only decode videos belonging to the requested split; the full list is
        # 288k frames across 9,848 videos and most of it is not needed yet.
        ag.load()
        videos = {frame.video_id for frame in ag.frames}
        if args.max_videos:
            # Evenly spaced rather than the first N, so a capped run still spans
            # the split instead of clustering on alphabetically early videos.
            ordered = sorted(videos)
            step = max(1, len(ordered) // args.max_videos)
            videos = set(ordered[::step][: args.max_videos])
        print(f"{args.split} split: {len(videos)} videos")

    summary = extract_all(
        frame_list=frame_list,
        videos_dir=Path(args.videos_dir),
        frames_dir=ag_root / args.frames_dirname,
        videos=videos,
        workers=args.workers,
        one_based=not args.zero_based,
        image_format=args.format,
        overwrite=args.overwrite,
    )
    _echo(summary)


# -- export / prepare / build ------------------------------------------------
def cmd_export(args) -> None:
    from .ag.classes import AG_OBJECT_CLASSES
    from .pipeline import Workspace, stage_export

    variant = _variant_by_name(args.variant)
    ws = Workspace(Path(args.artifacts))
    classes = AG_OBJECT_CLASSES
    if args.ag_root:
        from .ag.dataset import ActionGenome

        classes = ActionGenome(root=Path(args.ag_root), split=args.split).load(
            max_frames=args.max_frames
        ).classes
    _echo(stage_export(variant, ws, classes, device=args.device))


def cmd_prepare(args) -> None:
    """Produce the precision-specific ONNX graph TensorRT will compile."""
    from .detect.quantize import summarize_quantization, to_fp16_onnx
    from .pipeline import Workspace

    variant = _variant_by_name(args.variant)
    ws = Workspace(Path(args.artifacts))
    src = ws.onnx(variant.onnx_fp32)
    dst = ws.onnx(variant.onnx_final)

    if variant.precision == "fp16":
        to_fp16_onnx(src, dst)
    elif variant.precision == "int8":
        dst = _prepare_int8(args, variant, ws, src, dst)
    else:
        raise SystemExit(f"Unsupported precision {variant.precision!r}")
    _echo(summarize_quantization(dst))


def _prepare_int8(args, variant, ws, src: Path, dst: Path) -> Path:
    """Calibrate on real AG frames from the training split, then write Q/DQ.

    Calibration reads the data distribution, so calibrating on the split the
    model is then scored on would leak evaluation data into the model itself -
    a quieter version of training on the test set, and one that inflates int8
    results specifically. The split is therefore refused, not merely defaulted.
    """
    if not args.ag_root:
        raise SystemExit(
            "int8 calibration needs --ag-root: activation ranges must come from "
            "the evaluation domain, not from synthetic or unrelated images."
        )
    if args.calib_split == args.eval_split:
        raise SystemExit(
            f"Refusing to calibrate on {args.calib_split!r}, the same split "
            f"used for evaluation. Calibration observes the data distribution; "
            f"sharing a split with evaluation leaks test data into the engine. "
            f"Use --calib-split train with --eval-split test."
        )
    from .ag.dataset import ActionGenome
    from .detect.owlv2 import load_owlv2
    from .detect.preprocess import Owlv2Preprocessor, load_image
    from .detect.quantize import CalibrationReader, to_int8_onnx
    from .pipeline import load_queries

    import torch

    # Calibrate on the *train* split so the test split stays untouched. The
    # full split is indexed first and then sampled across videos: taking the
    # first N frames would draw them all from a handful of alphabetically early
    # videos, calibrating activation ranges on a few rooms and lighting setups.
    ag = ActionGenome(root=Path(args.ag_root), split=args.calib_split).load()
    if not len(ag):
        raise SystemExit(f"No frames found in AG split {args.calib_split!r}")
    ag.frames = _sample_across_videos(
        [f for f in ag.frames if f.image_path], args.calib_frames
    )
    if not len(ag):
        raise SystemExit(
            f"No extracted images for split {args.calib_split!r}. Run "
            f"'sgg extract-frames --split {args.calib_split}' first."
        )

    _, processor = load_owlv2(variant.checkpoint, device="cpu", dtype=torch.float32)
    from .pipeline import build_preprocessor

    pre = build_preprocessor(processor, device=args.device, fast=True)

    images = [load_image(f.image_path) for f in ag.frames]
    videos = {f.video_id for f in ag.frames}
    print(
        f"calibrating on {len(images)} frames from {len(videos)} videos "
        f"in split {args.calib_split!r}"
    )

    batch = pre(images)
    pixel_values = (
        batch.detach().float().cpu().numpy() if hasattr(batch, "detach") else np.asarray(batch)
    )
    queries = load_queries(ws, variant.checkpoint)

    # Record exactly what was calibrated on, so the claim of no overlap with the
    # evaluation split is auditable after the fact rather than taken on trust.
    from .pipeline import write_report

    used = [f.frame_key for f in ag.frames if f.image_path][: len(images)]
    write_report(
        {
            "variant": variant.name,
            "calibration_split": args.calib_split,
            "evaluation_split": args.eval_split,
            "num_frames": len(used),
            "num_videos": len({k.split("/")[0] for k in used}),
            "frame_keys": used,
        },
        ws.result(f"calibration_{variant.name}.json"),
    )

    reader = CalibrationReader(pixel_values, queries.embeds)
    return to_int8_onnx(src, dst, reader, calibration_method=args.calib_method)


def cmd_build(args) -> None:
    from .detect.trt_build import build_engine, engine_summary
    from .pipeline import Workspace

    variant = _variant_by_name(args.variant)
    ws = Workspace(Path(args.artifacts))
    engine = build_engine(
        ws.onnx(variant.onnx_final),
        ws.engine(variant.engine),
        workspace_gb=args.workspace_gb,
        timing_cache_path=ws.cache("timing.cache"),
        verbose=args.verbose,
    )
    _echo(engine_summary(engine))


# -- bench -------------------------------------------------------------------
def cmd_bench(args) -> None:
    """Per-frame latency for one compiled variant."""
    from .bench.latency import benchmark_engine, benchmark_h2d, benchmark_preprocess
    from .detect.owlv2 import load_owlv2
    from .detect.preprocess import Owlv2Preprocessor, iter_video_frames, load_image
    from .detect.trt_runner import TRTRunner
    from .pipeline import Workspace, build_preprocessor, load_queries, write_report

    import torch

    variant = _variant_by_name(args.variant)
    ws = Workspace(Path(args.artifacts))
    _, processor = load_owlv2(variant.checkpoint, device="cpu", dtype=torch.float32)
    pre = build_preprocessor(processor, device=args.device, fast=not args.slow_preprocess)

    images = _bench_images(args, pre)
    pixel_values = _as_numpy_batch(pre, images)
    queries = load_queries(ws, variant.checkpoint)

    runner = TRTRunner(ws.engine(variant.engine), device=args.device)
    engine_stats = benchmark_engine(
        runner, pixel_values, queries.embeds, f"{variant.name}:engine",
        warmup=args.warmup, iterations=args.iterations, device=args.device,
    )
    pre_stats = benchmark_preprocess(pre, images, f"{variant.name}:preprocess")
    h2d_stats = benchmark_h2d(pixel_values, f"{variant.name}:h2d", device=args.device)

    # The GPU preprocessor uploads the uint8 frame itself and hands back a tensor
    # already on device, so its transfer is inside `preprocess`. Adding the
    # standalone host-to-device figure as well would double-count it; that copy
    # is only a real pipeline stage when preprocessing runs on CPU.
    gpu_preprocess = not args.slow_preprocess
    stages = [pre_stats.median_ms, engine_stats.median_ms]
    if not gpu_preprocess:
        stages.append(h2d_stats.median_ms)

    report = {
        "variant": variant.name,
        "source": args.video or "synthetic",
        "num_distinct_frames": int(pixel_values.shape[0]),
        "image_size": pre.image_size,
        "preprocessor": "gpu" if gpu_preprocess else "cpu-transformers",
        "engine": engine_stats.as_dict(),
        "preprocess": pre_stats.as_dict(),
        "host_to_device": h2d_stats.as_dict(),
        "host_to_device_counted": not gpu_preprocess,
        "end_to_end_median_ms": round(sum(stages), 3),
    }
    _echo(report)
    write_report(report, ws.result(f"bench_{variant.name}.json"))


def _as_numpy_batch(pre, images) -> np.ndarray:
    """Preprocess to a numpy batch, whichever preprocessor is in use."""
    out = pre(images)
    if hasattr(out, "detach"):  # torch tensor from the GPU preprocessor
        return out.detach().float().cpu().numpy()
    return np.asarray(out)


def _bench_images(args, pre) -> list:
    """Real decoded frames when a video is given; noise only as a last resort."""
    from PIL import Image

    from .detect.preprocess import iter_video_frames

    if args.video:
        images = list(iter_video_frames(Path(args.video), stride=args.stride,
                                        limit=args.num_frames))
        if images:
            return images
        raise SystemExit(f"No frames decoded from {args.video}")
    print(
        "warning: benchmarking on synthetic noise. Quantized kernel timing "
        "depends on activation statistics, so pass --video for real numbers.",
        file=sys.stderr,
    )
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
        for _ in range(args.num_frames)
    ]


# -- evaluate ----------------------------------------------------------------
def cmd_evaluate(args) -> None:
    """Score one variant's detections against Action Genome ground truth."""
    from tqdm import tqdm

    from .ag.dataset import ActionGenome
    from .detect.owlv2 import load_owlv2, postprocess
    from .detect.preprocess import Owlv2Preprocessor, load_image
    from .detect.trt_runner import TRTRunner
    from .evaluation.detection_eval import evaluate_detections
    from .pipeline import Workspace, build_preprocessor, load_queries, write_report

    import torch

    variant = _variant_by_name(args.variant)
    ws = Workspace(Path(args.artifacts))
    ag = ActionGenome(root=Path(args.ag_root), split=args.split).load(
        max_frames=args.max_frames
    )
    frames = [f for f in ag.frames if f.image_path]
    if not frames:
        raise SystemExit("No AG frames with readable images; check --ag-root.")

    _, processor = load_owlv2(variant.checkpoint, device="cpu", dtype=torch.float32)
    pre = build_preprocessor(processor, device=args.device, fast=not args.slow_preprocess)
    queries = load_queries(ws, variant.checkpoint)
    runner = TRTRunner(ws.engine(variant.engine), device=args.device)

    detections = []
    for frame in tqdm(frames, desc=variant.name):
        image = load_image(frame.image_path)
        frame.width, frame.height = image.size
        outputs = runner.infer_numpy(
            {"pixel_values": pre([image]), "query_embeds": queries.embeds}
        )
        detections.append(
            postprocess(
                outputs["pred_logits"], outputs["pred_boxes"], outputs["objectness"],
                prompt_owner=queries.owner, num_classes=len(ag.classes),
                image_size=image.size, score_threshold=args.score_threshold,
                max_detections=args.max_detections,
            )
        )

    metrics = evaluate_detections(frames, detections, ag.classes)
    report = {"variant": variant.name, "split": args.split,
              "num_frames": len(frames), "metrics": metrics,
              "calibration_leakage": _check_calibration_leakage(ws, variant, frames)}
    _echo(report)
    write_report(report, ws.result(f"eval_{variant.name}.json"))


def cmd_report(args) -> None:
    """Render the comparison table from whatever results exist on disk."""
    from .bench.compare import load_results, render_report
    from .pipeline import DEFAULT_VARIANTS, Workspace

    ws = Workspace(Path(args.artifacts))
    names = args.variants or [v.name for v in DEFAULT_VARIANTS]
    results = load_results(ws.root / "results", names)
    if not results:
        raise SystemExit(f"No results found in {ws.root / 'results'}")
    report = render_report(results)
    print(report)
    (ws.root / "results" / "report.md").write_text(report + "\n")


def cmd_verify(args) -> None:
    """Check a compiled engine against the eager fp32 reference."""
    from PIL import Image
    import torch

    from .detect.owlv2 import Owlv2DetectionGraph, load_owlv2, postprocess
    from .detect.preprocess import Owlv2Preprocessor, iter_video_frames
    from .detect.trt_runner import TRTRunner, TorchRunner
    from .evaluation.verify import compare_detections, compare_outputs
    from .pipeline import Workspace, load_queries, write_report

    variant = _variant_by_name(args.variant)
    ws = Workspace(Path(args.artifacts))
    model, processor = load_owlv2(variant.checkpoint, device=args.device, dtype=torch.float32)
    pre = Owlv2Preprocessor(processor)
    queries = load_queries(ws, variant.checkpoint)

    if args.video:
        images = list(iter_video_frames(Path(args.video), stride=args.stride,
                                        limit=args.num_frames))
    else:
        rng = np.random.default_rng(0)
        images = [
            Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
            for _ in range(args.num_frames)
        ]

    engine = TRTRunner(ws.engine(variant.engine), device=args.device)
    reference = TorchRunner(Owlv2DetectionGraph(model), device=args.device)

    per_frame = []
    for image in images:
        feed = {"pixel_values": pre([image]), "query_embeds": queries.embeds}
        eo = engine.infer_numpy(dict(feed))
        ro = reference.infer_numpy(dict(feed))
        ed = postprocess(eo["pred_logits"], eo["pred_boxes"], eo["objectness"],
                         queries.owner, args.num_classes, image.size, args.score_threshold)
        rd = postprocess(ro["pred_logits"], ro["pred_boxes"], ro["objectness"],
                         queries.owner, args.num_classes, image.size, args.score_threshold)
        per_frame.append({"tensors": compare_outputs(eo, ro),
                          "detections": compare_detections(ed, rd)})

    summary = {
        "variant": variant.name,
        "frames": len(per_frame),
        "mean_logit_cosine": float(
            np.mean([f["tensors"]["pred_logits"]["cosine_similarity"] for f in per_frame])
        ),
        "mean_detection_match_rate": float(
            np.mean([f["detections"]["match_rate"] for f in per_frame])
        ),
        "mean_best_iou": float(
            np.mean([f["detections"]["mean_best_iou"] for f in per_frame])
        ),
        "per_frame": per_frame,
    }
    _echo({k: v for k, v in summary.items() if k != "per_frame"})
    write_report(summary, ws.result(f"verify_{variant.name}.json"))


def _sample_across_videos(frames: list, count: int) -> list:
    """Pick ``count`` frames spread over as many distinct videos as possible.

    Round-robins one frame per video before taking a second from any of them,
    so a small calibration set still covers many scenes rather than many frames
    of one scene.
    """
    if count is None or len(frames) <= count:
        return frames

    from collections import defaultdict

    by_video: dict[str, list] = defaultdict(list)
    for frame in frames:
        by_video[frame.video_id].append(frame)

    picked: list = []
    round_index = 0
    video_ids = sorted(by_video)
    while len(picked) < count:
        added = False
        for video_id in video_ids:
            bucket = by_video[video_id]
            if round_index < len(bucket):
                picked.append(bucket[round_index])
                added = True
                if len(picked) == count:
                    break
        if not added:
            break
        round_index += 1
    return picked


def _check_calibration_leakage(ws, variant, frames) -> dict:
    """Verify no evaluated frame or video was used to calibrate this engine.

    Checked at scoring time rather than at build time, because this is the point
    where a leak would actually corrupt a reported number.
    """
    manifest_path = ws.result(f"calibration_{variant.name}.json")
    if not manifest_path.exists():
        return {"checked": False, "reason": "no calibration manifest (not an int8 build)"}

    manifest = json.loads(manifest_path.read_text())
    calib_frames = set(manifest.get("frame_keys", []))
    calib_videos = {key.split("/")[0] for key in calib_frames}
    eval_frames = {f.frame_key for f in frames}
    eval_videos = {f.video_id for f in frames}

    frame_overlap = sorted(calib_frames & eval_frames)
    video_overlap = sorted(calib_videos & eval_videos)
    return {
        "checked": True,
        "calibration_split": manifest.get("calibration_split"),
        "calibration_frames": len(calib_frames),
        "overlapping_frames": len(frame_overlap),
        "overlapping_videos": len(video_overlap),
        "clean": not frame_overlap and not video_overlap,
        "examples": (frame_overlap[:5] or video_overlap[:5]) or None,
    }


# -- parser ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgg", description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--device", default="cuda")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-ag", help="inspect AG annotations")
    p.add_argument("--ag-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_validate_ag)

    p = sub.add_parser("extract-frames", help="decode AG keyframes from Charades videos")
    p.add_argument("--ag-root", required=True)
    p.add_argument("--videos-dir", required=True, help="directory of Charades .mp4 files")
    p.add_argument("--frames-dirname", default="frames")
    p.add_argument("--split", default="test", choices=["train", "test", "all"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-videos", type=int, default=None,
                   help="cap videos decoded, spread evenly across the split")
    p.add_argument("--format", default="png", choices=["png", "jpg"])
    p.add_argument("--zero-based", action="store_true",
                   help="treat AG frame numbers as 0-based (they are 1-based)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_extract_frames)

    p = sub.add_parser("export", help="checkpoint -> fp32 ONNX")
    p.add_argument("--variant", required=True)
    p.add_argument("--ag-root", default=None, help="derive vocabulary from data")
    p.add_argument("--split", default="test")
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("prepare", help="fp32 ONNX -> fp16/int8 ONNX")
    p.add_argument("--variant", required=True)
    p.add_argument("--ag-root", default=None)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--eval-split", default="test",
                   help="split reserved for evaluation; calibrating on it is refused")
    p.add_argument("--calib-frames", type=int, default=256)
    p.add_argument("--calib-method", default="entropy",
                   choices=["entropy", "max"])
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("build", help="ONNX -> TensorRT engine")
    p.add_argument("--variant", required=True)
    p.add_argument("--workspace-gb", type=float, default=6.0)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("bench", help="per-frame latency")
    p.add_argument("--variant", required=True)
    p.add_argument("--video", default=None, help="decode real frames from this file")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--slow-preprocess", action="store_true",
                   help="use the stock CPU image processor instead of the GPU one")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("evaluate", help="Action Genome mAP")
    p.add_argument("--variant", required=True)
    p.add_argument("--ag-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--score-threshold", type=float, default=0.05)
    p.add_argument("--max-detections", type=int, default=100)
    p.add_argument("--slow-preprocess", action="store_true",
                   help="use the stock CPU image processor instead of the GPU one")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("verify", help="engine vs eager fp32 reference")
    p.add_argument("--variant", required=True)
    p.add_argument("--video", default=None)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=36)
    p.add_argument("--score-threshold", type=float, default=0.1)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("report", help="comparison table from saved results")
    p.add_argument("--variants", nargs="*", default=None)
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
