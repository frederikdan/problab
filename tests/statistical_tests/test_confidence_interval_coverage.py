import unittest

import numpy as np
from scipy.stats import binom

from problab import CategoricalDistribution, NormalDistribution, P, RandomVariable


ALPHA = 0.1
FALSE_FAILURE_PROBABILITY = 1e-8


class ConfidenceIntervalCoverageStatisticalTests(unittest.TestCase):

    def assert_nominal_coverage(self, intervals_containing_truth: int, experiments: int) -> None:
        lower_count = int(
            binom.ppf(FALSE_FAILURE_PROBABILITY, experiments, 1 - ALPHA)
        )
        self.assertGreaterEqual(
            intervals_containing_truth,
            lower_count,
            f"Only {intervals_containing_truth}/{experiments} intervals contained the true value; "
            f"minimum count for the nominal {1 - ALPHA:.0%} coverage is {lower_count}.",
        )

    def test_probability_confidence_intervals_cover_true_categorical_probability(self):
        true_probability = 0.37
        variable = RandomVariable(
            CategoricalDistribution([0, 1], [1 - true_probability, true_probability])
        )
        generator = np.random.default_rng(501)
        experiments = 500
        covered = 0

        for _ in range(experiments):
            result = P(variable == 1, num_samples=60, rng=generator)
            interval = result.confidence_interval(alpha=ALPHA)
            covered += interval.lower <= true_probability <= interval.upper

        self.assert_nominal_coverage(covered, experiments)

    def test_random_variable_quantile_confidence_intervals_cover_true_median(self):
        variable = RandomVariable(NormalDistribution(0, 1))
        generator = np.random.default_rng(502)
        experiments = 180
        covered = 0

        for _ in range(experiments):
            interval = variable.quantile_confidence_interval(
                q=0.5,
                alpha=ALPHA,
                num_samples=80,
                rng=generator,
            )
            covered += interval.lower <= 0 <= interval.upper

        self.assert_nominal_coverage(covered, experiments)

    def test_distribution_quantile_confidence_intervals_cover_true_median(self):
        distribution = NormalDistribution(0, 1)
        generator = np.random.default_rng(503)
        experiments = 180
        covered = 0

        for _ in range(experiments):
            interval = distribution.quantile_confidence_interval(
                q=0.5,
                alpha=ALPHA,
                num_samples=80,
                rng=generator,
            )
            covered += interval.lower <= 0 <= interval.upper

        self.assert_nominal_coverage(covered, experiments)


if __name__ == "__main__":
    unittest.main()
