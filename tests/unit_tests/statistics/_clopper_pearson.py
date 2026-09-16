import math
import unittest
from unittest.mock import patch

from problab.statistics._clopper_pearson import _confidence_interval


class ClopperPearsonTests(unittest.TestCase):

    @patch("problab.statistics._clopper_pearson.beta.ppf")
    def test_zero_samples_returns_undefined_interval_without_beta_calls(self, beta_ppf):
        interval = _confidence_interval(
            num_samples=0,
            num_successes=0,
            alpha=0.05,
        )

        self.assertTrue(math.isnan(interval.lower))
        self.assertTrue(math.isnan(interval.upper))
        self.assertEqual(interval.alpha, 0.05)
        beta_ppf.assert_not_called()

    @patch("problab.statistics._clopper_pearson.beta.ppf")
    def test_zero_successes_sets_lower_bound_to_zero(self, beta_ppf):
        beta_ppf.return_value = 0.3

        interval = _confidence_interval(
            num_samples=10,
            num_successes=0,
            alpha=0.1,
        )

        self.assertEqual(interval.lower, 0.0)
        self.assertEqual(interval.upper, 0.3)
        beta_ppf.assert_called_once_with(q=0.95, a=1, b=10)

    @patch("problab.statistics._clopper_pearson.beta.ppf")
    def test_all_successes_sets_upper_bound_to_one(self, beta_ppf):
        beta_ppf.return_value = 0.7

        interval = _confidence_interval(
            num_samples=10,
            num_successes=10,
            alpha=0.1,
        )

        self.assertEqual(interval.lower, 0.7)
        self.assertEqual(interval.upper, 1.0)
        beta_ppf.assert_called_once_with(q=0.05, a=10, b=1)

    @patch("problab.statistics._clopper_pearson.beta.ppf")
    def test_interior_success_count_uses_beta_lower_and_upper_quantiles(self, beta_ppf):
        beta_ppf.side_effect = (0.2, 0.8)

        interval = _confidence_interval(
            num_samples=10,
            num_successes=4,
            alpha=0.1,
        )

        self.assertEqual((interval.lower, interval.upper, interval.alpha), (0.2, 0.8, 0.1))
        self.assertEqual(
            beta_ppf.call_args_list[0].kwargs,
            {"q": 0.05, "a": 4, "b": 7},
        )
        self.assertEqual(
            beta_ppf.call_args_list[1].kwargs,
            {"q": 0.95, "a": 5, "b": 6},
        )

    def test_rejects_invalid_counts_and_alpha(self):
        with self.assertRaises(ValueError):
            _confidence_interval(num_samples=2, num_successes=3, alpha=0.05)
        with self.assertRaises(TypeError):
            _confidence_interval(num_samples=True, num_successes=0, alpha=0.05)
        with self.assertRaises(ValueError):
            _confidence_interval(num_samples=1, num_successes=0, alpha=0.0)


if __name__ == "__main__":
    unittest.main()
