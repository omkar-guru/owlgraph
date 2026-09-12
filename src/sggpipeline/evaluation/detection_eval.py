"""Detection accuracy on Action Genome, scored with COCO mAP.

AG is not COCO, and two of its properties shape how results must be read:

* Annotation is **sparse and incomplete** - only objects involved in an
  annotated interaction are labelled.  A detector is therefore penalised for
  correctly finding real objects AG chose not to annotate, so absolute mAP
  understates quality.  The comparison between precisions is still valid, since
  every variant is penalised identically.
* Person boxes come from a detector, not a human annotator, so the "person"
  class measures agreement with that detector rather than with ground truth.

Per-class AP is reported so these effects stay visible instead of being averaged
into a single number.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np


def build_coco_gt(frames, classes: tuple[str, ...]) -> dict:
    """Assemble AG ground truth into a COCO-format dict."""
    images, annotations = [], []
    ann_id = 1
    for image_id, frame in enumerate(frames, start=1):
        width, height = frame.width or 0, frame.height or 0
        images.append(
            {
                "id": image_id,
                "file_name": frame.frame_key,
                "width": width,
                "height": height,
            }
        )
        for box, label in zip(frame.boxes, frame.labels, strict=True):
            x1, y1, x2, y2 = (float(v) for v in box)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label) + 1,  # COCO ids are 1-based
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "info": {"description": "Action Genome detection ground truth"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(classes)
        ],
    }


def build_coco_detections(detections_per_frame: list) -> list[dict]:
    """Flatten per-frame detections into COCO result records."""
    results = []
    for image_id, det in enumerate(detections_per_frame, start=1):
        for box, score, label in zip(det.boxes, det.scores, det.labels, strict=True):
            x1, y1, x2, y2 = (float(v) for v in box)
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return results


def evaluate_detections(
    frames, detections_per_frame: list, classes: tuple[str, ...], quiet: bool = True
) -> dict:
    """Score detections with pycocotools and return the standard COCO metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_dict = build_coco_gt(frames, classes)
    results = build_coco_detections(detections_per_frame)
    if not results:
        return {"error": "no detections above threshold", "mAP": 0.0}

    sink = io.StringIO() if quiet else None
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()

    with ctx:
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)

        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = evaluator.stats
    metrics = {
        "mAP": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "num_gt_boxes": len(gt_dict["annotations"]),
        "num_detections": len(results),
        "num_images": len(gt_dict["images"]),
    }
    metrics["per_class_AP"] = _per_class_ap(evaluator, classes)
    return metrics


def _per_class_ap(evaluator, classes: tuple[str, ...]) -> dict[str, float]:
    """AP@[.5:.95] per class; NaN where a class has no ground truth."""
    precision = evaluator.eval["precision"]  # (iou, recall, cls, area, maxdet)
    out: dict[str, float] = {}
    for idx, name in enumerate(classes):
        if idx >= precision.shape[2]:
            break
        values = precision[:, :, idx, 0, -1]
        values = values[values > -1]
        out[name] = float(np.mean(values)) if values.size else float("nan")
    return out


def compare_metrics(baseline: dict, variant: dict, classes: tuple[str, ...]) -> dict:
    """Difference report between two precision variants."""
    deltas = {
        key: round(variant[key] - baseline[key], 5)
        for key in ("mAP", "mAP_50", "mAP_75", "AR_100")
        if key in baseline and key in variant
    }
    per_class = {}
    for name in classes:
        b = baseline.get("per_class_AP", {}).get(name)
        v = variant.get("per_class_AP", {}).get(name)
        if b is None or v is None or np.isnan(b) or np.isnan(v):
            continue
        per_class[name] = round(v - b, 5)
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:10]
    return {
        "delta": deltas,
        "relative_mAP_change_pct": (
            round(100.0 * deltas.get("mAP", 0.0) / baseline["mAP"], 2)
            if baseline.get("mAP") else None
        ),
        "largest_per_class_regressions": dict(worst),
    }


def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    return path
