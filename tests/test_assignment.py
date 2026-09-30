"""SciPy-backed gated assignment: same semantics as exhaustive search, and fast."""

import itertools
import time
import unittest

import numpy as np

from sggpipeline.tracking.tracker import assign


def brute_force(cost, valid):
    """(match count, total cost) of the best gated matching, by enumeration."""
    n, m = cost.shape
    best = (0, 0.0)
    for choices in itertools.product(range(-1, m), repeat=n):
        pairs = [(i, j) for i, j in enumerate(choices) if j >= 0]
        if len({j for _, j in pairs}) != len(pairs) or any(not valid[i, j] for i, j in pairs):
            continue
        best = min(best, (-len(pairs), sum(cost[i, j] for i, j in pairs)))
    return -best[0], best[1]


class AssignmentTest(unittest.TestCase):
    def check(self, cost, valid):
        pairs = assign(cost, valid)
        count, total = brute_force(cost, valid)
        self.assertEqual(len(pairs), count)
        self.assertAlmostEqual(sum(cost[i, j] for i, j in pairs), total, places=9)
        self.assertTrue(all(valid[i, j] for i, j in pairs))
        self.assertEqual(len({i for i, _ in pairs}), len(pairs))
        self.assertEqual(len({j for _, j in pairs}), len(pairs))

    def test_matches_exhaustive_search_on_random_gated_problems(self):
        rng = np.random.default_rng(7)
        for n, m in itertools.product(range(1, 6), range(1, 7)):
            if n > 5 or (n == 5 and m > 5):
                continue
            for density in (0.2, 0.5, 0.9):
                for _ in range(3):
                    self.check(rng.random((n, m)), rng.random((n, m)) < density)

    def test_cost_scale_does_not_trade_matches_for_cost(self):
        # A cheap single match must lose to two expensive ones: cardinality first.
        cost = np.array([[1e-6, 5.0], [5.0, 1e9]])
        valid = np.array([[True, True], [True, False]])
        self.check(cost, valid)
        self.assertEqual(len(assign(cost, valid)), 2)
        rng = np.random.default_rng(3)
        for scale in (1e-9, 1.0, 1e6):
            self.check(rng.random((4, 5)) * scale, rng.random((4, 5)) < 0.5)

    def test_degenerate_inputs(self):
        self.assertEqual(assign(np.zeros((0, 3)), np.zeros((0, 3), bool)), [])
        self.assertEqual(assign(np.zeros((3, 0)), np.zeros((3, 0), bool)), [])
        self.assertEqual(assign(np.ones((3, 3)), np.zeros((3, 3), bool)), [])
        self.check(np.zeros((3, 4)), np.ones((3, 4), bool))

    def test_fast_at_a_hundred_objects(self):
        rng = np.random.default_rng(0)
        cost, valid = rng.random((100, 100)), rng.random((100, 100)) < 0.7
        assign(cost, valid)
        start = time.perf_counter()
        for _ in range(5):
            assign(cost, valid)
        self.assertLess((time.perf_counter() - start) / 5, 0.020)


if __name__ == "__main__":
    unittest.main()
