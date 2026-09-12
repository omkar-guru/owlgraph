"""Synthetic correctness checks; these do not establish real tracking quality."""

from dataclasses import replace
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from sggpipeline.tracking.cache import DetectionCache, load_caches
from sggpipeline.tracking.experiment import association_metrics, compare, main, train_identity
from sggpipeline.tracking.features import pool_box_features
from sggpipeline.tracking.identity import IdentityHead, identity_loss
from sggpipeline.tracking.tracker import AssociationTracker, TrackerConfig, assign


def detections(boxes, labels=None):
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    return SimpleNamespace(boxes=boxes, labels=np.zeros(len(boxes), dtype=int) if labels is None else np.array(labels))


def toy_cache(video="train-video", split="train"):
    return DetectionCache(
        metadata=dict(schema_version=1, video_id=video, split=split,
                      feature_source="synthetic-4d-v1", identity_source="human"),
        timestamps=np.array([0., 1., 2.]), offsets=np.array([0, 2, 4, 6]),
        boxes=np.array([[0, 0, 10, 10], [30, 0, 40, 10]] * 3, dtype=np.float32),
        scores=np.ones(6), labels=np.zeros(6, dtype=np.int64), objectness=np.ones(6),
        features=np.array([[1, 0, 0, 0], [0, 1, 0, 0]] * 3, dtype=np.float32),
        instance_ids=np.array([0, 1] * 3))


class TrackerTests(unittest.TestCase):
    def test_assignment_matches_bruteforce_with_gates(self):
        rng = np.random.default_rng(13)
        for n, m in itertools.product(range(1, 5), repeat=2):
            for _ in range(4):
                cost, valid = rng.random((n, m)), rng.random((n, m)) > .4
                actual = assign(cost, valid)
                best = (0, 0.)
                for choices in itertools.product(range(-1, m), repeat=n):
                    matches = [(i, j) for i, j in enumerate(choices) if j >= 0]
                    if len({j for _, j in matches}) != len(matches) or any(not valid[i, j] for i, j in matches):
                        continue
                    best = min(best, (-len(matches), sum(cost[i, j] for i, j in matches)))
                self.assertEqual(len(actual), -best[0])
                self.assertAlmostEqual(sum(cost[i, j] for i, j in actual), best[1])

    def test_same_class_swap_preserves_identity(self):
        tracker = AssociationTracker()
        boxes = detections([[0, 0, 10, 10], [20, 0, 30, 10]])
        first = tracker.update(boxes, np.eye(2), 0)
        second = tracker.update(boxes, np.eye(2)[::-1], 1)
        np.testing.assert_array_equal(second.track_ids, first.track_ids[::-1])

    def test_missing_observation_and_expiry(self):
        tracker = AssociationTracker()
        box = detections([[0, 0, 10, 10]])
        first = tracker.update(box, np.ones((1, 2)), 0)
        gap = tracker.update(detections([]), np.empty((0, 2)), 1)
        self.assertEqual(len(gap.track_ids), 0)
        recovered = tracker.update(box, np.ones((1, 2)), 2)
        np.testing.assert_array_equal(recovered.track_ids, first.track_ids)
        expired = tracker.update(box, np.ones((1, 2)), 4.1)
        self.assertNotEqual(expired.track_ids[0], first.track_ids[0])
        self.assertEqual(expired.expired_ids, (int(first.track_ids[0]),))

    def test_timestamp_motion(self):
        tracker = AssociationTracker(TrackerConfig(max_age_seconds=10, max_center_distance=1))
        first = tracker.update(detections([[0, 0, 10, 10]]), np.ones((1, 2)), 0)
        tracker.update(detections([[10, 0, 20, 10]]), np.ones((1, 2)), 1)
        moved = tracker.update(detections([[50, 0, 60, 10]]), np.ones((1, 2)), 5)
        np.testing.assert_array_equal(moved.track_ids, first.track_ids)

    def test_ambiguity_retires_old_memory(self):
        tracker = AssociationTracker()
        tracker.update(detections([[0, 0, 10, 10], [0, 0, 10, 10]]), np.ones((2, 2)), 0)
        result = tracker.update(detections([[0, 0, 10, 10]]), np.ones((1, 2)), 1)
        self.assertTrue(result.uncertain[0])
        self.assertEqual(result.track_ids[0], 2)
        self.assertEqual(set(result.expired_ids), {0, 1})

    def test_labels_are_not_identity_and_reset(self):
        tracker = AssociationTracker()
        first = tracker.update(detections([[0, 0, 10, 10]], [0]), np.ones((1, 2)), 0)
        second = tracker.update(detections([[0, 0, 10, 10]], [1]), np.ones((1, 2)), 1)
        np.testing.assert_array_equal(first.track_ids, second.track_ids)
        tracker.reset()
        self.assertEqual(tracker.update(detections([[0, 0, 10, 10]]), np.ones((1, 2)), 0).track_ids[0], 0)

    def test_invalid_inputs(self):
        tracker = AssociationTracker()
        box = detections([[0, 0, 10, 10]])
        tracker.update(box, np.ones((1, 2)), 0)
        with self.assertRaises(ValueError):
            tracker.update(box, np.ones((1, 2)), 0)
        with self.assertRaises(ValueError):
            tracker.update(box, np.zeros((1, 2)), 1)
        with self.assertRaises(ValueError):
            tracker.update(box, np.ones((1, 3)), 1)

    def test_pooling_respects_bottom_right_padding(self):
        grid = np.array([[[1, 2], [3, 4]], [[99, 99], [99, 99]]], dtype=np.float32)
        pooled = pool_box_features(grid, np.array([[0, 0, 20, 10], [0, 0, 10, 10]]), (20, 10))
        np.testing.assert_allclose(pooled, [[2, 3], [1, 2]])
        self.assertEqual(pool_box_features(grid, np.empty((0, 4)), (20, 10)).shape, (0, 2))


class IdentityTests(unittest.TestCase):
    def test_contrastive_order_and_unknown_mask(self):
        ids, frames = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
        good = torch.tensor([[1., 0], [1, 0], [0, 1], [0, 1]], requires_grad=True)
        good_loss = identity_loss(good, ids, frames)
        self.assertLess(good_loss.item(), identity_loss(good[[0, 2, 1, 3]], ids, frames).item())
        extended = torch.cat([good, torch.tensor([[100., -50.]])])
        self.assertAlmostEqual(good_loss.item(), identity_loss(extended, torch.cat([ids, torch.tensor([-1])]),
                                                             torch.cat([frames, torch.tensor([2])])).item())
        good_loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())

    def test_same_frame_or_category_only_supervision_rejected(self):
        z = torch.randn(4, 8)
        with self.assertRaises(ValueError):
            identity_loss(z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, dtype=torch.long))
        with self.assertRaises(ValueError):
            identity_loss(z, torch.zeros(4, dtype=torch.long), torch.arange(4))

    def test_cache_and_training_contract(self):
        cache = toy_cache()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.npz"
            cache.save(path)
            restored = load_caches([path], require_identity=True)[0]
            np.testing.assert_array_equal(restored.features, cache.features)
            with self.assertRaises(ValueError):
                load_caches([path, path])
        with self.assertRaises(ValueError):
            train_identity([toy_cache(split="test")], steps=1)
        with self.assertRaises(ValueError):
            replace(cache, metadata={**cache.metadata, "identity_source": "none"}).validate()
        with self.assertRaises(ValueError):
            replace(cache, instance_ids=np.array([0, 0] * 3)).validate()
        with self.assertRaises(ValueError):
            replace(cache, offsets=np.array([0, 4, 2, 6])).validate()

    def test_training_updates_only_identity_head(self):
        cache = toy_cache()
        before = cache.features.copy()
        torch.manual_seed(0)
        initial = IdentityHead(4, 16, 8)
        trained, report = train_identity([cache], steps=12, hidden_dim=16, output_dim=8)
        self.assertLess(report["last_loss"], report["first_loss"])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial.parameters(), trained.parameters())))
        np.testing.assert_array_equal(cache.features, before)

    def test_metrics_count_transfer_and_fragmentation(self):
        metrics = association_metrics(toy_cache(), np.array([0, 1, 1, 0, 1, 0]))
        self.assertEqual(metrics["id_switches"], 2)
        self.assertEqual(metrics["cross_instance_transfers"], 2)
        self.assertEqual(metrics["extra_ids_per_instance_total"], 2)
        self.assertEqual(metrics["identity_links"], 4)

    def test_comparison_does_not_mutate_inputs(self):
        cache = toy_cache(split="test")
        original = cache.boxes.copy()
        report = compare([cache], TrackerConfig(), IdentityHead(4, 16, 8))
        self.assertEqual(set(report["variants"]), {"appearance", "identity"})
        self.assertEqual(report["variants"]["appearance"]["totals"]["id_switches"], 0)
        np.testing.assert_array_equal(cache.boxes, original)

    def test_cli_roundtrip_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, test_path = root / "train.npz", root / "test.npz"
            checkpoint, output = root / "identity.pt", root / "report.json"
            toy_cache().save(train_path)
            toy_cache("held-out", "test").save(test_path)
            main(["train", "--caches", str(train_path), "--output", str(checkpoint), "--steps", "2"])
            main(["evaluate", "--caches", str(test_path), "--checkpoint", str(checkpoint), "--output", str(output)])
            self.assertTrue(output.exists())
            toy_cache("train-video", "test").save(test_path)
            with self.assertRaisesRegex(ValueError, "overlap"):
                main(["evaluate", "--caches", str(test_path), "--checkpoint", str(checkpoint), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
