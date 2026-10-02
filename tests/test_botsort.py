"""Integration tests against the actual installed BoT-SORT implementation."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sggpipeline.tracking import BoTSORTConfig, OWLBoTSORT
from sggpipeline.tracking.experiment import association_metrics, main
from test_tracking import toy_cache


def detections(boxes, scores=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return SimpleNamespace(boxes=boxes, scores=np.full(len(boxes), .9) if scores is None else np.array(scores),
                           labels=np.zeros(len(boxes), dtype=np.int64))


@unittest.skipUnless(importlib.util.find_spec("lap"), "Install the tracking extra")
class BoTSORTTests(unittest.TestCase):
    def test_supplied_features_change_association_without_reid_model(self):
        with patch("ultralytics.trackers.utils.reid.ReID", side_effect=AssertionError("No extra encoder allowed")):
            boxes = detections([[0, 0, 10, 10], [3, 0, 13, 10]])
            normal, swapped = OWLBoTSORT(), OWLBoTSORT()
            features = np.eye(2, dtype=np.float32)
            original = features.copy()
            normal.update(boxes, features, 0)
            swapped.update(boxes, features, 0)
            a = normal.update(boxes, features, 1)
            b = swapped.update(boxes, features[::-1], 1)
            np.testing.assert_array_equal(a.track_ids, b.track_ids[::-1])
            np.testing.assert_array_equal(features, original)
            self.assertIsNone(normal._backend.gmc.method)
            self.assertEqual(normal._backend.args.model, "auto")

    def test_low_confidence_recovery_and_detection_row_indices(self):
        tracker = OWLBoTSORT()
        boxes = [[0, 0, 10, 10], [30, 0, 40, 10]]
        first = tracker.update(detections(boxes), np.eye(2), 0)
        second = tracker.update(detections(boxes[::-1], [.15, .9]), np.eye(2)[::-1], 1)
        np.testing.assert_array_equal(second.track_ids, first.track_ids[::-1])

    def test_filtered_and_tentative_detections_have_no_id(self):
        tracker = OWLBoTSORT()
        first = tracker.update(detections([[0, 0, 10, 10]], [.01]), np.ones((1, 3)), 0)
        self.assertEqual(first.track_ids[0], -1)
        tentative = tracker.update(detections([[0, 0, 10, 10]]), np.ones((1, 3)), 1)
        self.assertEqual(tentative.track_ids[0], -1)
        confirmed = tracker.update(detections([[0, 0, 10, 10]]), np.ones((1, 3)), 2)
        self.assertGreaterEqual(confirmed.track_ids[0], 0)
        self.assertTrue(confirmed.is_new[0])

    def test_occlusion_expiry_and_reset(self):
        tracker = OWLBoTSORT(BoTSORTConfig(track_buffer=1))
        box = detections([[0, 0, 10, 10]])
        first = tracker.update(box, np.ones((1, 3)), 0)
        self.assertEqual(len(tracker.update(detections([]), np.empty((0, 3)), 1).track_ids), 0)
        recovered = tracker.update(box, np.ones((1, 3)), 2)
        np.testing.assert_array_equal(first.track_ids, recovered.track_ids)
        tracker.update(detections([]), np.empty((0, 3)), 3)
        expired = tracker.update(detections([]), np.empty((0, 3)), 4)
        self.assertIn(int(first.track_ids[0]), expired.expired_ids)
        tracker.update(box, np.ones((1, 3)), 5)
        fresh = tracker.update(box, np.ones((1, 3)), 6)
        self.assertNotEqual(fresh.track_ids[0], first.track_ids[0])
        tracker.reset()
        np.testing.assert_array_equal(tracker.update(box, np.ones((1, 3)), 0).track_ids, first.track_ids)

    def test_cadence_and_feature_validation(self):
        tracker = OWLBoTSORT()
        box = detections([[0, 0, 10, 10]])
        tracker.update(box, np.ones((1, 3)), 0)
        tracker.update(box, np.ones((1, 3)), .1)
        with self.assertRaisesRegex(ValueError, "regularly"):
            tracker.update(box, np.ones((1, 3)), .3)
        with self.assertRaisesRegex(ValueError, "dimension"):
            tracker.update(box, np.ones((1, 4)), .2)
        with self.assertRaises(ValueError):
            tracker.update(box, np.zeros((1, 3)), .2)

    def test_independent_tracker_id_allocators(self):
        first, second = OWLBoTSORT(), OWLBoTSORT()
        one = detections([[0, 0, 10, 10]])
        first.update(one, np.ones((1, 3)), 0)
        second.update(one, np.ones((1, 3)), 0)
        second.reset()
        two = detections([[0, 0, 10, 10], [30, 0, 40, 10]])
        first.update(two, np.ones((2, 3)), 1)
        result = first.update(two, np.ones((2, 3)), 2)
        self.assertEqual(len(set(result.track_ids)), 2)
        self.assertNotIn(-1, result.track_ids)

    def test_unassigned_metrics_do_not_create_false_shared_identity(self):
        metrics = association_metrics(toy_cache(), np.full(6, -1))
        self.assertEqual(metrics["id_switches"], 0)
        self.assertEqual(metrics["cross_instance_transfers"], 0)
        self.assertEqual(metrics["assignment_coverage"], 0)
        self.assertEqual(metrics["unassigned_observations"], 6)

    def test_cached_evaluation_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, output = Path(directory) / "video.npz", Path(directory) / "report.json"
            toy_cache("heldout", "test").save(cache)
            main(["evaluate", "--tracker", "botsort", "--caches", str(cache), "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report["tracker"], "botsort")
            self.assertIn("backend_version", report)
            self.assertEqual(report["variants"]["appearance"]["totals"]["assignment_coverage"], 1)
            self.assertFalse(report["ambiguity_estimates_available"])


if __name__ == "__main__":
    unittest.main()
