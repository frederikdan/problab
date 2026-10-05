from math import ceil, log2
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import binom

from problab.statistics._quantiles import _quantile_confidence_interval


class QuantileBoundaryTests(unittest.TestCase):
    def test_binary_search_matches_exhaustive_feasible_indices(self):
        for n in (6, 8, 17, 32, 101):
            for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                for alpha in (0.05, 0.2, 0.8):
                    candidates = np.arange(1, n)
                    lower = candidates[binom.cdf(candidates - 1, n, q) <= alpha / 2]
                    upper = candidates[binom.sf(candidates, n, q) <= alpha / 2]
                    with self.subTest(n=n, q=q, alpha=alpha):
                        if not len(lower) or not len(upper):
                            with self.assertRaises(ValueError):
                                _quantile_confidence_interval(np.arange(n), q, alpha)
                            continue
                        with patch("problab.statistics._quantiles.binom.cdf", wraps=binom.cdf) as cdf:
                            with patch("problab.statistics._quantiles.binom.sf", wraps=binom.sf) as sf:
                                interval = _quantile_confidence_interval(np.arange(n)[::-1], q, alpha)
                        self.assertEqual(interval.lower, lower[-1] - 1)
                        self.assertEqual(interval.upper, upper[0])
                        self.assertLessEqual(cdf.call_count, ceil(log2(n - 1)))
                        self.assertLessEqual(sf.call_count, ceil(log2(n - 1)))

    def test_tail_probability_equal_to_threshold_is_feasible(self):
        n = 8
        alpha = 2 * binom.cdf(1, n, 0.5)
        interval = _quantile_confidence_interval(np.arange(n), 0.5, alpha)
        self.assertEqual((interval.lower, interval.upper), (1.0, 6.0))
