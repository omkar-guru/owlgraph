"""Stage 1 postprocessing: patch provenance and optional duplicate suppression."""

import unittest

import numpy as np
import torch
from torchvision.ops import box_iou

from sggpipeline.detect.owlv2 import _sigmoid, postprocess

NUM_PATCHES, NUM_PROMPTS, NUM_CLASSES = 400, 7, 4
OWNER = np.array([0, 0, 1, 2, 2, 3, 3])


def raw_outputs(seed: int):
    rng = np.random.default_rng(seed)
    logits = rng.normal(-2.0, 1.5, (1, NUM_PATCHES, NUM_PROMPTS)).astype(np.float32)
    centres = rng.uniform(0.1, 0.9, (NUM_PATCHES, 2))
    sizes = rng.uniform(0.05, 0.3, (NUM_PATCHES, 2))
    # Clusters of near-identical boxes, as OWLv2 emits around one object.
    centres[1::4] = centres[::4][: len(centres[1::4])] + 0.002
    boxes = np.concatenate([centres, sizes], axis=1)[None].astype(np.float32)
    objectness = rng.normal(0, 1, (1, NUM_PATCHES)).astype(np.float32)
    return logits, boxes, objectness


class PostprocessTest(unittest.TestCase):
    def test_patch_index_points_at_the_producing_patch(self):
        logits, boxes, obj = raw_outputs(0)
        det = postprocess(logits, boxes, obj, OWNER, NUM_CLASSES, (640, 480), 0.05, 100)
        self.assertEqual(det.patch_index.shape, det.scores.shape)
        class_scores = np.full((NUM_PATCHES, NUM_CLASSES), -np.inf)
        for prompt, cls in enumerate(OWNER):
            class_scores[:, cls] = np.maximum(class_scores[:, cls], _sigmoid(logits[0, :, prompt]))
        for i, patch in enumerate(det.patch_index):
            self.assertEqual(det.labels[i], class_scores[patch].argmax())
            self.assertAlmostEqual(float(det.scores[i]), float(class_scores[patch].max()), places=5)
            cx, cy, w, h = boxes[0, patch] * 640
            expected = np.clip([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0, [640, 480, 640, 480])
            np.testing.assert_allclose(det.boxes[i], expected, atol=1e-3)

    def test_default_path_is_top_k_by_score(self):
        logits, boxes, obj = raw_outputs(1)
        det = postprocess(logits, boxes, obj, OWNER, NUM_CLASSES, (640, 480), 0.05, 50)
        self.assertLessEqual(len(det.scores), 50)
        self.assertTrue((np.diff(det.scores) <= 0).all())
        self.assertTrue((det.scores >= 0.05).all())

    def test_nms_removes_same_class_overlaps_only(self):
        logits, boxes, obj = raw_outputs(2)
        plain = postprocess(logits, boxes, obj, OWNER, NUM_CLASSES, (640, 480), 0.05, 1000)
        det = postprocess(logits, boxes, obj, OWNER, NUM_CLASSES, (640, 480), 0.05, 1000,
                          nms_iou=0.5)
        self.assertLess(len(det.scores), len(plain.scores))
        self.assertTrue((np.diff(det.scores) <= 0).all())
        self.assertTrue(set(det.patch_index) <= set(plain.patch_index))
        iou = box_iou(torch.from_numpy(det.boxes), torch.from_numpy(det.boxes)).numpy()
        np.fill_diagonal(iou, 0)
        same = det.labels[:, None] == det.labels[None, :]
        self.assertFalse(((iou > 0.5) & same).any())

    def test_nms_keeps_max_detections_after_suppression(self):
        logits, boxes, obj = raw_outputs(3)
        det = postprocess(logits, boxes, obj, OWNER, NUM_CLASSES, (640, 480), 0.05, 10,
                          nms_iou=0.5)
        self.assertEqual(len(det.scores), 10)

    def test_empty_result_has_patch_index(self):
        logits, boxes, obj = raw_outputs(4)
        det = postprocess(logits - 50, boxes, obj, OWNER, NUM_CLASSES, (640, 480), 0.05, 100)
        self.assertEqual(len(det.scores), 0)
        self.assertEqual(det.patch_index.shape, (0,))


if __name__ == "__main__":
    unittest.main()
