import unittest
from unittest.mock import patch

import numpy as np

from problab.statistics._quantiles import QuantileMethod, _quantile_confidence_interval


class QuantileConfidenceIntervalTests(unittest.TestCase):

    def test_quantile_method_lists_supported_numpy_methods(self):
        self.assertIn("linear", QuantileMethod.__args__)
        self.assertIn("inverted_cdf", QuantileMethod.__args__)

    def test_rejects_invalid_quantile_alpha_and_empty_samples(self):
        samples = np.arange(10)

        with self.assertRaises(ValueError):
            _quantile_confidence_interval(samples, q=0.0)
        with self.assertRaises(ValueError):
            _quantile_confidence_interval(samples, q=0.5, alpha=1.0)
        with self.assertRaises(ValueError):
            _quantile_confidence_interval(np.array([]), q=0.5)

    def test_rejects_sample_size_below_confidence_requirement(self):
        with self.assertRaisesRegex(ValueError, "At least 6 samples"):
            _quantile_confidence_interval(np.arange(5), q=0.5, alpha=0.05)

    @patch("problab.statistics._quantiles.binom.sf")
    @patch("problab.statistics._quantiles.binom.cdf")
    def test_uses_outer_order_statistics_when_tail_probabilities_are_large(
        self,
        cdf,
        sf,
    ):
        cdf.side_effect = lambda k, n, q: 0.01 if k == 0 else 1.0
        sf.side_effect = lambda k, n, q: 0.01 if k == n - 1 else 1.0
        samples = np.array([4, 1, 3, 2, 6, 5])

        interval = _quantile_confidence_interval(samples, q=0.5, alpha=0.05)

        self.assertEqual((interval.lower, interval.upper, interval.alpha), (1.0, 6.0, 0.05))
        self.assertLessEqual(cdf.call_count, 3)
        self.assertLessEqual(sf.call_count, 3)

    @patch("problab.statistics._quantiles.binom.sf")
    @patch("problab.statistics._quantiles.binom.cdf")
    def test_binary_search_selects_tail_boundary_order_statistics(
        self,
        cdf,
        sf,
    ):
        cdf.side_effect = lambda k, n, q: 0.01 if k <= 1 else 0.1
        sf.side_effect = lambda k, n, q: 0.01 if k >= 6 else 0.1
        samples = np.array([8, 1, 7, 2, 6, 3, 5, 4])

        interval = _quantile_confidence_interval(samples, q=0.5, alpha=0.05)

        self.assertEqual((interval.lower, interval.upper), (2.0, 7.0))
        self.assertLessEqual(cdf.call_count, 3)
        self.assertLessEqual(sf.call_count, 3)


if __name__ == "__main__":
    unittest.main()
