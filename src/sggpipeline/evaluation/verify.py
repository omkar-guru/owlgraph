"""Compare a compiled engine against the eager-PyTorch fp32 reference.

mAP alone cannot tell a quantization problem from a preprocessing or
postprocessing problem - both show up as "the number got worse".  Comparing raw
head outputs on identical inputs separates them: if the logits already disagree,
the damage is in the engine, not in the scoring code.
"""

from __future__ import annotations

import numpy as np


def compare_outputs(
    engine_outputs: dict[str, np.ndarray], reference_outputs: dict[str, np.ndarray]
) -> dict:
    """Element-wise agreement between engine and reference head outputs."""
    report: dict[str, dict] = {}
    for key in ("pred_logits", "pred_boxes", "objectness"):
        a = np.asarray(engine_outputs[key], dtype=np.float64).ravel()
        b = np.asarray(reference_outputs[key], dtype=np.float64).ravel()
        diff = np.abs(a - b)
        denom = np.maximum(np.abs(b), 1e-6)
        report[key] = {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "max_rel_diff": float((diff / denom).max()),
            "cosine_similarity": float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            ),
        }
    return report


def compare_detections(engine_det, reference_det, iou_threshold: float = 0.5) -> dict:
    """How much the final detection sets agree, which is what mAP actually sees.

    Raw tensor drift can be large while the ranked detections stay identical, and
    the reverse is also possible near a decision boundary.  Both are reported.
    """
    matched = 0
    ious: list[float] = []
    for eng_box, eng_label in zip(engine_det.boxes, engine_det.labels, strict=True):
        best_iou, best_label = 0.0, None
        for ref_box, ref_label in zip(reference_det.boxes, reference_det.labels, strict=True):
            iou = _iou(eng_box, ref_box)
            if iou > best_iou:
                best_iou, best_label = iou, ref_label
        ious.append(best_iou)
        if best_iou >= iou_threshold and best_label == eng_label:
            matched += 1

    n_engine = len(engine_det.boxes)
    return {
        "engine_detections": n_engine,
        "reference_detections": len(reference_det.boxes),
        "matched_at_iou": matched,
        "match_rate": float(matched / n_engine) if n_engine else 0.0,
        "mean_best_iou": float(np.mean(ious)) if ious else 0.0,
        "top1_label_agrees": bool(
            n_engine
            and len(reference_det.labels)
            and engine_det.labels[0] == reference_det.labels[0]
        ),
    }


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
