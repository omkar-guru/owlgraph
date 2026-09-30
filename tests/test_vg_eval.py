"""VG150 evaluator port and any-subject targets, against the reference logic."""

import unittest
from functools import reduce

import numpy as np
import torch

from sggpipeline.relations.model import PredicateSchema, RelationshipHead, all_pair_geometry, classification_loss
from sggpipeline.relations.sgdet import any_subject_targets, match_matrix, relation_map
from sggpipeline.relations.vg_eval import VGRecall, iou_plus_one, triplet_hits


def reference_pred_to_gt(gt_rels, gt_classes, gt_boxes, pred_rels, pred_classes, pred_boxes, thr=0.5):
    """The reference ``_compute_pred_matches`` (non-phrdet), transcribed."""
    gt_trip = np.column_stack((gt_classes[gt_rels[:, 0]], gt_rels[:, 2], gt_classes[gt_rels[:, 1]]))
    pr_trip = np.column_stack((pred_classes[pred_rels[:, 0]], pred_rels[:, 2], pred_classes[pred_rels[:, 1]]))
    gt_tb = np.column_stack((gt_boxes[gt_rels[:, 0]], gt_boxes[gt_rels[:, 1]]))
    pr_tb = np.column_stack((pred_boxes[pred_rels[:, 0]], pred_boxes[pred_rels[:, 1]]))
    keeps = (gt_trip[:, None, :] == pr_trip[None]).all(-1)
    pred_to_gt = [[] for _ in range(len(pred_rels))]
    for gi in np.where(keeps.any(1))[0]:
        keep = keeps[gi]
        boxes = pr_tb[keep]
        sub = iou_plus_one(gt_tb[gi, None, :4], boxes[:, :4])[0]
        obj = iou_plus_one(gt_tb[gi, None, 4:], boxes[:, 4:])[0]
        for i in np.where(keep)[0][(sub >= thr) & (obj >= thr)]:
            pred_to_gt[i].append(int(gi))
    return pred_to_gt


def random_image(rng, n_gt=6, n_rel=8, n_det=10, classes=4, preds=5):
    xy = rng.uniform(0, 80, (n_gt, 2))
    gt_boxes = np.concatenate([xy, xy + rng.uniform(10, 40, (n_gt, 2))], 1)
    gt_classes = rng.integers(0, classes, n_gt)
    gt_rels = np.column_stack([rng.integers(0, n_gt, n_rel), rng.integers(0, n_gt, n_rel), rng.integers(0, preds, n_rel)])
    gt_rels = gt_rels[gt_rels[:, 0] != gt_rels[:, 1]]
    src = rng.integers(0, n_gt, n_det)  # detections jittered from GT boxes
    det_boxes = gt_boxes[src] + rng.normal(0, 3, (n_det, 4))
    det_classes = np.where(rng.random(n_det) < 0.8, gt_classes[src], rng.integers(0, classes, n_det))
    return gt_rels, gt_classes, gt_boxes, det_classes, det_boxes


class TripletHitsTest(unittest.TestCase):
    def test_matches_reference_on_random_images(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            gt_rels, gt_classes, gt_boxes, det_classes, det_boxes = random_image(rng)
            n = len(det_boxes)
            pred_rels = np.column_stack([rng.integers(0, n, 40), rng.integers(0, n, 40), rng.integers(0, 5, 40)])
            ours = triplet_hits(gt_rels, gt_classes, gt_boxes, pred_rels, det_classes, det_boxes)
            ref = reference_pred_to_gt(gt_rels, gt_classes, gt_boxes, pred_rels, det_classes, det_boxes)
            for k in (1, 5, 20, 40):
                want = reduce(np.union1d, ref[:k])
                self.assertEqual(sorted(np.flatnonzero(ours[:, :k].any(1))), sorted(int(x) for x in want))

    def test_plus_one_iou(self):
        # 10x10 pixels inclusive each, overlapping in 5x10: 50 / 150.
        self.assertAlmostEqual(iou_plus_one(np.array([[0., 0, 9, 9]]), np.array([[5., 0, 14, 9]]))[0, 0], 1 / 3)


class VGRecallTest(unittest.TestCase):
    def setUp(self):
        # GT: 0 man, 1 horse, 2 hat; man riding horse (p1), man wearing hat (p2), duplicate of the first.
        self.gt_boxes = np.array([[0., 0, 50, 100], [40, 40, 140, 120], [10, 0, 30, 15]])
        self.gt_classes = np.array([0, 1, 2])
        self.gt_rels = np.array([[0, 1, 1], [0, 2, 2], [0, 1, 1]])

    def test_graph_constraint_and_duplicates(self):
        r = VGRecall(num_predicates=3, ks=(1, 2))
        pairs = np.array([[0, 1], [0, 2]])
        scores = np.array([[0.0, 0.9, 0.1], [0.0, 0.2, 0.8]])
        r.add_image(self.gt_rels, self.gt_classes, self.gt_boxes, pairs, scores, self.gt_classes,
                    self.gt_boxes, np.ones(3))
        s = r.summary()
        self.assertAlmostEqual(s["with_constraint/R@1"], 2 / 3)  # riding hits both duplicates
        self.assertAlmostEqual(s["with_constraint/R@2"], 1.0)
        self.assertAlmostEqual(s["with_constraint/mR@1"], (0.0 + 1.0 + 0.0) / 3)  # predicate 0 absent -> 0
        self.assertAlmostEqual(s["pair_recall"], 1.0)
        self.assertAlmostEqual(s["object_recall"], 1.0)

    def test_constraint_keeps_one_predicate_per_pair(self):
        r = VGRecall(num_predicates=3, ks=(2,))
        pairs = np.array([[0, 1], [0, 2]])
        scores = np.array([[0.0, 0.9, 0.1], [0.0, 0.85, 0.8]])  # hat pair's best is wrong
        r.add_image(self.gt_rels, self.gt_classes, self.gt_boxes, pairs, scores, self.gt_classes,
                    self.gt_boxes, np.ones(3))
        s = r.summary()
        self.assertAlmostEqual(s["with_constraint/R@2"], 2 / 3)
        self.assertAlmostEqual(s["no_constraint/R@2"], 2 / 3)  # top 2 overall: 0.9 and 0.85
        r2 = VGRecall(num_predicates=3, ks=(3,))
        r2.add_image(self.gt_rels, self.gt_classes, self.gt_boxes, pairs, scores, self.gt_classes,
                     self.gt_boxes, np.ones(3))
        self.assertAlmostEqual(r2.summary()["no_constraint/R@3"], 1.0)


class AnySubjectTargetsTest(unittest.TestCase):
    def test_targets_follow_matches(self):
        gt_boxes = torch.tensor([[[0., 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]])
        gt_labels = torch.tensor([[0, 1, 2]])
        det_boxes = torch.tensor([[[40., 0, 50, 10], [0, 0, 10, 10], [20, 0, 30, 10]]])  # permuted
        match = match_matrix(det_boxes, gt_labels[:, [2, 0, 1]], torch.ones(1, 3, dtype=torch.bool),
                             gt_boxes, gt_labels, torch.ones(1, 3, dtype=torch.bool))
        rels = torch.tensor([[[1, 2, 3], [1, 0, 0], [-1, -1, -1]]])  # GT 1 -> GT 2 (p3), GT 1 -> GT 0 (p0)
        positive, targets = any_subject_targets(match, relation_map(rels, 3, 4))
        self.assertEqual(sorted(map(tuple, torch.nonzero(positive[0]).tolist())), [(2, 0), (2, 1)])
        self.assertTrue(targets[0, 2, 0, 3] and targets[0, 2, 1, 0])
        self.assertEqual(int(targets.sum()), 2)


class SchemaTest(unittest.TestCase):
    def test_single_group_multilabel_head(self):
        schema = PredicateSchema(tuple(f"p{i}" for i in range(7)), (("predicate", slice(0, 7)),))
        embeds = torch.nn.functional.normalize(torch.randn(7, 16), dim=1)
        head = RelationshipHead(8, 5, embeds, hidden=32, key_dim=8, schema=schema)
        feats, labels = torch.randn(1, 4, 8), torch.randint(0, 5, (1, 4))
        boxes = torch.tensor([[10., 10, 50, 50]]).repeat(1, 4, 1) + torch.arange(4.)[None, :, None]
        geo = all_pair_geometry(boxes, torch.tensor([[100, 100]]))
        subj, obj = head.roles(feats, labels)
        logits = head.classify(subj[0, :2], obj[0, 2:], geo[0, :2, 2])
        self.assertEqual(logits.shape, (2, 7))
        self.assertTrue(torch.allclose(head.probabilities(logits), logits.sigmoid()))
        targets = torch.zeros(2, 7, dtype=torch.bool)
        targets[0, 3] = True
        want = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float())
        self.assertTrue(torch.allclose(classification_loss(logits, targets, schema=schema), want))


if __name__ == "__main__":
    unittest.main()
