import unittest

import numpy as np

from problab import ProbabilityResult


class ProbabilityResultPropertyTests(unittest.TestCase):

    def test_sample_count_and_confidence_interval_hold_for_many_valid_counts(self):
        generator = np.random.default_rng(19)

        for _ in range(100):
            unconditioned_samples = int(generator.integers(1, 1000))
            conditioned_samples = int(generator.integers(0, unconditioned_samples + 1))
            successes = int(generator.integers(0, conditioned_samples + 1))
            value = np.nan if conditioned_samples == 0 else successes / conditioned_samples

            result = ProbabilityResult(
                value=value,
                num_successes=successes,
                num_unconditioned_samples=unconditioned_samples,
                num_conditioned_samples=conditioned_samples,
            )
            interval = result.confidence_interval(alpha=0.1)

            self.assertEqual(result.num_samples, conditioned_samples)
            if conditioned_samples == 0:
                self.assertTrue(np.isnan(result.value))
                self.assertTrue(np.isnan(interval.lower))
                self.assertTrue(np.isnan(interval.upper))
            else:
                self.assertEqual(result.value, successes / conditioned_samples)
                self.assertLessEqual(interval.lower, interval.upper)
                self.assertGreaterEqual(interval.lower, 0.0)
                self.assertLessEqual(interval.upper, 1.0)


if __name__ == "__main__":
    unittest.main()
