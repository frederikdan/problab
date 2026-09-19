import unittest

from problab import ProbabilityResult


class ProbabilityResultConfidenceIntervalRegressionTests(unittest.TestCase):

    def test_confidence_interval_is_available_from_a_public_probability_result(self):
        result = ProbabilityResult(
            value=0.5,
            num_successes=5,
            num_unconditioned_samples=10,
        )

        interval = result.confidence_interval(alpha=0.1)

        self.assertLess(interval.lower, result.value)
        self.assertGreater(interval.upper, result.value)
        self.assertEqual(interval.alpha, 0.1)


if __name__ == "__main__":
    unittest.main()
