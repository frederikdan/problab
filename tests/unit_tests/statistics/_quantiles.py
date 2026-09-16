import unittest
from unittest.mock import call, patch

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
        cdf.return_value = 1.0
        sf.return_value = 1.0
        samples = np.array([4, 1, 3, 2, 6, 5])

        interval = _quantile_confidence_interval(samples, q=0.5, alpha=0.05)

        self.assertEqual((interval.lower, interval.upper, interval.alpha), (1.0, 6.0, 0.05))
        cdf.assert_called_once_with(1, 6, 0.5)
        sf.assert_called_once_with(4, 6, 0.5)

    @patch("problab.statistics._quantiles.binom.sf")
    @patch("problab.statistics._quantiles.binom.cdf")
    def test_moves_order_statistics_inward_until_tail_probability_exceeds_alpha(
        self,
        cdf,
        sf,
    ):
        cdf.side_effect = (0.01, 0.1)
        sf.side_effect = (0.01, 0.1)
        samples = np.array([8, 1, 7, 2, 6, 3, 5, 4])

        interval = _quantile_confidence_interval(samples, q=0.5, alpha=0.05)

        self.assertEqual((interval.lower, interval.upper), (2.0, 7.0))
        self.assertEqual(cdf.call_args_list, [call(1, 8, 0.5), call(2, 8, 0.5)])
        self.assertEqual(sf.call_args_list, [call(6, 8, 0.5), call(5, 8, 0.5)])


if __name__ == "__main__":
    unittest.main()
