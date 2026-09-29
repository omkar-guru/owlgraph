"""Export, compile and verify the prev50 token-merging engine.

Stages, each gated before the next runs:

1. export the static merged graph to ONNX (fp32), convert to fp16
2. build a strongly-typed TensorRT engine
3. verify engine outputs against the eager fp32 merged forward on real frames,
   using real previous-frame priors - the same bar the unmerged engine passed
   (logit cosine >= 0.999, detection match rate >= 0.85)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.merged_export import MergeIndexer, export_merged_onnx
from sggpipeline.detect.owlv2 import load_owlv2, postprocess
from sggpipeline.detect.preprocess import load_image
from sggpipeline.detect.quantize import summarize_quantization, to_fp16_onnx
from sggpipeline.detect.token_merging import MergePlan, WindowGrid, merged_forward
from sggpipeline.detect.trt_build import build_engine, engine_summary
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.evaluation.verify import compare_detections, compare_outputs
from sggpipeline.pipeline import Workspace, load_queries, write_report

from early_objectness_probe import decode_positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--image-size", type=int, default=960)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--verify-frames", type=int, default=8)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()
    tag = f"base_merged{int(round(args.fraction * 100))}"
    model, _ = load_owlv2("base", device="cuda", dtype=torch.float32)
    side = model.num_patches_height

    fp32 = export_merged_onnx(model, len(queries.prompts), args.image_size, args.fraction,
                              ws.onnx(f"{tag}_fp32.onnx"))
    fp16 = to_fp16_onnx(fp32, ws.onnx(f"{tag}_fp16.onnx"))
    print("onnx:", json.dumps(summarize_quantization(fp16)))

    engine_path = build_engine(fp16, ws.engine(f"{tag}_fp16.plan"), workspace_gb=6.0,
                               timing_cache_path=ws.cache("timing.cache"))
    summary = engine_summary(engine_path)
    print("engine:", json.dumps({k: summary[k] for k in ("size_mb", "num_layers",
                                                          "device_memory_mb")}))
    for t in summary["tensors"]:
        print(f"  {t['mode']:<6} {t['name']:<15} {t['dtype']:<6} {t['shape']}")

    # Verification on real frames with real previous-frame priors.
    runner = TRTRunner(engine_path)
    indexer = MergeIndexer(side, args.fraction)
    grid = WindowGrid(side, "cuda")
    pre = GpuOwlv2Preprocessor(args.image_size, device="cuda")
    ag = ActionGenome(root=Path(args.ag_root), split="test").load()
    candidates = [f for f in ag.frames if f.image_path and f.frame_index - 2 >= 0]
    step = max(1, len(candidates) // args.verify_frames)
    frames = candidates[::step][: args.verify_frames]
    videos_dir = Path(args.ag_root) / "Charades_v1_480"

    per_frame = []
    for frame in frames:
        pos = frame.frame_index - 2
        prior_img = decode_positions(videos_dir / frame.video_id, {pos})[pos]
        image = load_image(frame.image_path)
        prior = merged_forward(model, pre([prior_img]).float(), query, MergePlan(),
                               grid)["objectness"][0].float()
        px = pre([image]).float()
        ref = merged_forward(model, px, query,
                             MergePlan(early_fraction=args.fraction, early_scores=prior), grid)
        unmerged, members, assign = indexer(prior)
        out = runner.infer({"pixel_values": px, "query_embeds": query,
                            "unmerged_idx": unmerged, "member_patches": members,
                            "assign": assign})
        torch.cuda.synchronize()
        eng = {k: v.detach().float().cpu().numpy() for k, v in out.items()}
        refn = {"pred_logits": ref["pred_logits"].float().cpu().numpy(),
                "pred_boxes": ref["pred_boxes"].float().cpu().numpy(),
                "objectness": ref["objectness"].float().cpu().numpy()}
        n = len(ag.classes)
        ed = postprocess(eng["pred_logits"], eng["pred_boxes"], eng["objectness"],
                         queries.owner, n, image.size, 0.1)
        rd = postprocess(refn["pred_logits"], refn["pred_boxes"], refn["objectness"],
                         queries.owner, n, image.size, 0.1)
        per_frame.append({"tensors": compare_outputs(eng, refn),
                          "detections": compare_detections(ed, rd)})

    cos = float(np.mean([f["tensors"]["pred_logits"]["cosine_similarity"] for f in per_frame]))
    match = float(np.mean([f["detections"]["match_rate"] for f in per_frame]))
    iou = float(np.mean([f["detections"]["mean_best_iou"] for f in per_frame]))
    passed = cos >= 0.999 and match >= 0.85
    write_report({"engine": str(engine_path), "fraction": args.fraction,
                  "mean_logit_cosine": cos, "mean_detection_match_rate": match,
                  "mean_best_iou": iou, "passed": passed, "per_frame": per_frame},
                 ws.result(f"verify_{tag}.json"))
    print(f"\nverify vs eager fp32 on {len(per_frame)} frames: logit cosine {cos:.5f}, "
          f"detection match {match:.3f}, mean IoU {iou:.3f} -> {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
