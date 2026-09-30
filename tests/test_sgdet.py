"""Relationship head: matching, targets, recall and losses on hand-worked cases."""

import unittest

import numpy as np
import torch

from sggpipeline.ag.relations import ATTENTION, CONTACTING, PREDICATES, SPATIAL
from sggpipeline.relations.model import (
    RelationshipHead, all_pair_geometry, classification_loss, predicate_probabilities)
from sggpipeline.relations.sgdet import SGRecall, match_matrix, pair_targets

A, S = len(ATTENTION), len(SPATIAL)
LOOK, NOT_LOOK = 0, 1
HOLD = A + S + CONTACTING.index("holding")
TOUCH = A + S + CONTACTING.index("touching")
PERSON, CUP, TABLE = 0, 10, 31


def frame():
    """GT: person, cup (looking_at + holding), table (not_looking_at).
    Detections: person, cup, a table-shaped box labelled wrongly, and junk."""
    gt_boxes = torch.tensor([[[0., 0, 10, 10], [20, 0, 30, 10], [40, 0, 60, 10]]])
    gt_labels = torch.tensor([[PERSON, CUP, TABLE]])
    gt_valid = torch.ones(1, 3, dtype=torch.bool)
    gt_pred = torch.zeros(1, 3, len(PREDICATES), dtype=torch.bool)
    gt_pred[0, 1, [LOOK, HOLD]] = True
    gt_pred[0, 2, NOT_LOOK] = True
    det_boxes = torch.tensor([[[0., 0, 10, 10], [20, 0, 30, 10], [40, 0, 60, 10], [70, 70, 80, 80]]])
    det_labels = torch.tensor([[PERSON, CUP, TABLE - 1, CUP]])
    det_valid = torch.ones(1, 4, dtype=torch.bool)
    det_scores = torch.tensor([0.9, 0.8, 0.7, 0.2])
    match = match_matrix(det_boxes, det_labels, det_valid, gt_boxes, gt_labels, gt_valid)
    return match, gt_pred, gt_valid, det_valid, det_scores


class MatchingTest(unittest.TestCase):
    def test_label_must_agree(self):
        match, *_ = frame()
        self.assertTrue(match[0, 0, 0] and match[0, 1, 1])
        self.assertFalse(match[0, 2].any())  # right box, wrong label
        self.assertFalse(match[0, 3].any())

    def test_positive_pairs_and_targets(self):
        match, gt_pred, *_ = frame()
        positive, targets = pair_targets(match, gt_pred)
        self.assertEqual(torch.nonzero(positive[0]).tolist(), [[0, 1]])
        self.assertEqual(sorted(torch.nonzero(targets[0, 1]).flatten().tolist()), [LOOK, HOLD])


class RecallTest(unittest.TestCase):
    def evaluate(self, touch_prob, ks):
        match, gt_pred, gt_valid, det_valid, det_scores = frame()
        pairs = torch.tensor([[0, 1], [0, 2]])
        probs = torch.full((2, len(PREDICATES)), 0.01)
        probs[0, LOOK], probs[0, HOLD], probs[0, TOUCH] = 0.9, 0.8, touch_prob
        probs[1, NOT_LOOK] = 0.9
        r = SGRecall(ks=ks)
        r.add_frame(pairs, probs, torch.ones(2), det_scores, match[0], gt_pred[0], gt_valid[0], det_valid[0])
        return r.summary()

    def test_recall_object_and_pair_recall(self):
        s = self.evaluate(touch_prob=0.1, ks=(50,))
        self.assertAlmostEqual(s["no_constraint/R@50"], 2 / 3)  # table triplet unreachable
        self.assertAlmostEqual(s["object_recall"], 2 / 3)
        self.assertAlmostEqual(s["pair_recall"], 1 / 2)
        self.assertAlmostEqual(s["pair_recall_upper_bound"], 1 / 2)

    def test_constraint_can_cost_a_hit(self):
        # touching outranks holding within the contacting group: the constraint
        # keeps only touching, so holding is missed; without it both survive.
        s = self.evaluate(touch_prob=0.85, ks=(50,))
        self.assertAlmostEqual(s["with_constraint/R@50"], 1 / 3)
        self.assertAlmostEqual(s["no_constraint/R@50"], 2 / 3)

    def test_top_k_cutoff(self):
        # Top 1 triplet is person-looking_at-cup (0.9 * 0.9 * 0.8); one of three hit.
        s = self.evaluate(touch_prob=0.1, ks=(1,))
        self.assertAlmostEqual(s["no_constraint/R@1"], 1 / 3)


class ModelAndLossTest(unittest.TestCase):
    def test_shapes_and_probabilities(self):
        embeds = torch.nn.functional.normalize(torch.randn(len(PREDICATES), 16), dim=1)
        for kind in ("text", "closed"):
            head = RelationshipHead(8, 36, embeds, classifier=kind, hidden=32, key_dim=8)
            feats, labels = torch.randn(2, 5, 8), torch.randint(0, 36, (2, 5))
            boxes = torch.tensor([[10., 10, 50, 50]]).repeat(2, 5, 1) + torch.arange(5.)[None, :, None]
            geo = all_pair_geometry(boxes, torch.tensor([[100, 100], [100, 100]]))
            subj, obj = head.roles(feats, labels)
            self.assertEqual(head.route(subj, obj, geo).shape, (2, 5, 5))
            logits = head.classify(subj[0, :3], obj[0, 1:4], geo[0, 0, 1:4])
            p = predicate_probabilities(logits)
            self.assertEqual(p.shape, (3, len(PREDICATES)))
            self.assertTrue(torch.allclose(p[:, :A].sum(dim=1), torch.ones(3)))

    def test_held_out_predicates_get_no_gradient(self):
        logits = torch.zeros(4, len(PREDICATES), requires_grad=True)
        targets = torch.zeros(4, len(PREDICATES), dtype=torch.bool)
        targets[:, LOOK] = True
        targets[:, HOLD] = True
        seen = torch.ones(len(PREDICATES), dtype=torch.bool)
        seen[HOLD] = False
        classification_loss(logits, targets, seen).backward()
        self.assertTrue(torch.all(logits.grad[:, HOLD] == 0))
        self.assertTrue(torch.any(logits.grad[:, LOOK] != 0))

    def test_held_out_attention_target_is_skipped(self):
        logits = torch.zeros(2, len(PREDICATES), requires_grad=True)
        targets = torch.zeros(2, len(PREDICATES), dtype=torch.bool)
        targets[:, NOT_LOOK] = True
        seen = torch.ones(len(PREDICATES), dtype=torch.bool)
        seen[NOT_LOOK] = False
        classification_loss(logits, targets, seen).backward()
        self.assertTrue(torch.all(logits.grad[:, :A] == 0))


if __name__ == "__main__":
    unittest.main()
