import unittest

from problab.probability.results import ProbabilityResult


class ProbabilityResultTests(unittest.TestCase):
    def test_confidence_interval_uses_clopper_pearson(self):
        result = ProbabilityResult(
            value=0.25,
            num_successes=25,
            num_unconditioned_samples=100,
        )

        interval = result.confidence_interval()

        self.assertLessEqual(interval.lower, result.value)
        self.assertGreaterEqual(interval.upper, result.value)
        self.assertEqual(interval.alpha, 0.05)


if __name__ == "__main__":
    unittest.main()
