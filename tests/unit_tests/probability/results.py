import math
import unittest
from unittest.mock import patch, sentinel

from problab.probability.results import ProbabilityResult


class ProbabilityResultTests(unittest.TestCase):

    def test_num_samples_uses_unconditioned_count_without_condition(self):
        result = ProbabilityResult(
            value=0.25,
            num_successes=25,
            num_unconditioned_samples=100,
        )

        self.assertEqual(result.num_samples, 100)
        self.assertEqual(str(result), "0.25")

    def test_num_samples_uses_conditioned_count_when_present(self):
        result = ProbabilityResult(
            value=0.5,
            num_successes=10,
            num_unconditioned_samples=100,
            num_conditioned_samples=20,
        )

        self.assertEqual(result.num_samples, 20)

    @patch("problab.probability.results._clopper_pearson._confidence_interval")
    def test_confidence_interval_delegates_counts_and_alpha(self, confidence_interval):
        confidence_interval.return_value = sentinel.interval
        result = ProbabilityResult(
            value=0.25,
            num_successes=25,
            num_unconditioned_samples=100,
        )

        interval = result.confidence_interval(alpha=0.1)

        self.assertIs(interval, sentinel.interval)
        confidence_interval.assert_called_once_with(
            num_samples=100,
            num_successes=25,
            alpha=0.1,
        )

    def test_constructor_allows_nan_for_zero_conditioned_samples(self):
        result = ProbabilityResult(
            value=math.nan,
            num_successes=0,
            num_unconditioned_samples=10,
            num_conditioned_samples=0,
        )

        self.assertTrue(math.isnan(result.value))
        self.assertEqual(result.num_samples, 0)

    def test_constructor_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ProbabilityResult(value=0.5, num_successes=1, num_unconditioned_samples=0)
        with self.assertRaises(ValueError):
            ProbabilityResult(value=0.5, num_successes=3, num_unconditioned_samples=2)
        with self.assertRaises(ValueError):
            ProbabilityResult(
                value=0.5,
                num_successes=1,
                num_unconditioned_samples=2,
                num_conditioned_samples=3,
            )


if __name__ == "__main__":
    unittest.main()
