"""Pair head geometry, output layout and recall metrics on hand-worked cases."""

import unittest

import numpy as np
import torch

from sggpipeline.ag.relations import ATTENTION, CONTACTING, PREDICATES, SPATIAL
from sggpipeline.relations.pair_head import (
    GEOMETRY_DIM, PairHead, ground_truth, pair_geometry, predicate_scores, recall_metrics)

A, S = len(ATTENTION), len(SPATIAL)


def col(group: str, name: str) -> int:
    return {"attention": 0, "spatial": A, "contacting": A + S}[group] + \
        {"attention": ATTENTION, "spatial": SPATIAL, "contacting": CONTACTING}[group].index(name)


def two_pairs():
    """Pair 0 fully right; pair 1 right on attention/spatial, wrong on contacting."""
    truth = ground_truth(np.array([0, 1]),
                         np.eye(S, dtype=bool)[[2, 3]],  # in_front_of, behind
                         np.eye(len(CONTACTING), dtype=bool)[[5, 8]])  # holding, not_contacting
    scores = np.full((2, len(PREDICATES)), 0.01)
    scores[0, [col("attention", "looking_at"), col("spatial", "in_front_of"),
               col("contacting", "holding")]] = [0.99, 0.80, 0.70]
    scores[1, [col("attention", "not_looking_at"), col("spatial", "behind"),
               col("contacting", "holding")]] = [0.60, 0.50, 0.95]
    return scores, truth


class GeometryAndLayoutTest(unittest.TestCase):
    def test_identical_boxes(self):
        box = torch.tensor([[10., 20., 50., 80.]])
        g = pair_geometry(box, box, torch.tensor([[100, 100]]))
        self.assertEqual(g.shape, (1, GEOMETRY_DIM))
        np.testing.assert_allclose(g[0, 8:].numpy(), [0, 0, 0, 0, 1, 1, 1], atol=1e-6)

    def test_object_inside_subject(self):
        s = torch.tensor([[0., 0., 100., 100.]])
        o = torch.tensor([[40., 40., 60., 60.]])
        g = pair_geometry(s, o, torch.tensor([[200, 200]]))
        self.assertAlmostEqual(float(g[0, 13]), 1.0, places=5)  # all of the object is inside
        self.assertAlmostEqual(float(g[0, 14]), 0.04, places=5)  # 4% of the subject

    def test_scores_and_truth_follow_predicate_order(self):
        logits = {"attention": torch.tensor([[5., 0., 0.]]), "spatial": torch.zeros(1, S),
                  "contacting": torch.zeros(1, len(CONTACTING))}
        p = predicate_scores(logits)
        self.assertEqual(p.shape, (1, len(PREDICATES)))
        self.assertAlmostEqual(float(p[0, :A].sum()), 1.0, places=5)
        self.assertTrue(torch.allclose(p[0, A:], torch.full((len(PREDICATES) - A,), 0.5)))
        t = ground_truth(np.array([2]), np.eye(S, dtype=bool)[[0]], np.eye(len(CONTACTING), dtype=bool)[[16]])
        self.assertEqual(sorted(np.flatnonzero(t[0])), [2, A + 0, A + S + 16])

    def test_geometry_only_control_ignores_features(self):
        head = PairHead(feature_dim=0, num_classes=36)
        out = head(None, None, torch.zeros(3, GEOMETRY_DIM), torch.zeros(3, dtype=torch.long))
        self.assertEqual({k: v.shape for k, v in out.items()},
                         {"attention": (3, A), "spatial": (3, S), "contacting": (3, len(CONTACTING))})


class RecallTest(unittest.TestCase):
    def test_with_constraint_hand_worked(self):
        scores, truth = two_pairs()
        m = recall_metrics(scores, truth, np.array([0, 0]), ks=(10,))
        # 6 constrained guesses all fit in K=10; 5 of 6 true triplets are hit.
        self.assertAlmostEqual(m["with_constraint/R@10"], 5 / 6)
        # Per predicate: five predicates at recall 1, not_contacting at 0.
        self.assertAlmostEqual(m["with_constraint/mR@10"], 5 / 6)
        self.assertEqual(m["with_constraint/per_predicate_R@10"][col("contacting", "not_contacting")], 0.0)

    def test_no_constraint_top_k_cutoff(self):
        scores, truth = two_pairs()
        m = recall_metrics(scores, truth, np.array([0, 0]), ks=(2,))
        # Top 2 overall: pair0 looking_at (0.99, hit) and pair1 holding (0.95, miss).
        self.assertAlmostEqual(m["no_constraint/R@2"], 1 / 6)

    def test_recall_is_averaged_over_frames(self):
        scores, truth = two_pairs()
        perfect = np.where(truth, 0.9, 0.01)
        m = recall_metrics(np.vstack([scores, perfect]), np.vstack([truth, truth]),
                           np.array([0, 0, 1, 1]), ks=(10,))
        self.assertAlmostEqual(m["with_constraint/R@10"], (5 / 6 + 1) / 2)


if __name__ == "__main__":
    unittest.main()
