"""Merge planner: fixed budget, and box protection reorders but never resizes it."""

import unittest

import torch

from sggpipeline.detect.merged_export import MergeIndexer


class MergeIndexerTest(unittest.TestCase):
    def setUp(self):
        self.indexer = MergeIndexer(side=8, fraction=0.5, dilate=False, device="cpu")  # 4x4 windows

    def merged_windows(self, plan):
        return {int(w) for w in self.indexer.grid.window_of_patch[plan[1][:, 0]]}

    def test_budget_is_fixed(self):
        prior = torch.rand(64)
        plan = self.indexer(prior)
        self.assertEqual(plan[1].shape, (8, 4))  # 8 of 16 windows merged
        self.assertEqual(plan[0].numel(), 64 - 32)

    def test_protected_windows_merge_last(self):
        prior = torch.zeros(64)  # everything looks like background
        # 128x128 image -> 32 px windows; protect the top-left 2x2 windows (64x64 px).
        plan = self.indexer(prior, protect_boxes=[[0, 0, 64, 64]], image_size=(128, 128))
        protected = {0, 1, 4, 5}
        self.assertFalse(protected & self.merged_windows(plan))
        self.assertEqual(plan[1].shape, (8, 4))

    def test_protection_cannot_shrink_the_budget(self):
        prior = torch.zeros(64)
        plan = self.indexer(prior, protect_boxes=[[0, 0, 128, 128]], image_size=(128, 128))
        self.assertEqual(plan[1].shape, (8, 4))  # all protected -> still exactly 8 merged

    def test_windows_inside_uses_padded_square(self):
        # 128x64 image pads to 128x128; a box on the lower half of the *padded*
        # square covers no real pixels, and its windows are the bottom two rows.
        mask = self.indexer.windows_inside([[0, 64, 128, 128]], (128, 64), "cpu")
        self.assertEqual(set(torch.nonzero(mask).flatten().tolist()), set(range(8, 16)))

    def test_no_boxes_is_unchanged(self):
        prior = torch.rand(64)
        a, b = self.indexer(prior), self.indexer(prior, protect_boxes=[], image_size=(128, 128))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
