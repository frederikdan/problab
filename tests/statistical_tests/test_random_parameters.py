import math
import unittest

import numpy as np

from problab import (
    BinomialDistribution,
    CategoricalDistribution,
    NormalDistribution,
    P,
    PoissonDistribution,
    RandomVariable,
)


NUM_SAMPLES = 100_000


def binomial_tail(trials: int, probability: float, minimum: int) -> float:
    return sum(
        math.comb(trials, k) * probability ** k * (1 - probability) ** (trials - k)
        for k in range(minimum, trials + 1)
    )


def normal_cdf(value: float, mean: float, standard_deviation: float) -> float:
    return (1 + math.erf((value - mean) / (standard_deviation * math.sqrt(2)))) / 2


class RandomParameterStatisticalTests(unittest.TestCase):

    def assert_probability_matches(self, observed: float, expected: float, samples: int) -> None:
        standard_error = math.sqrt(expected * (1 - expected) / samples)
        self.assertLessEqual(
            abs(observed - expected),
            6 * standard_error,
            f"Observed {observed}, expected {expected}, n={samples}",
        )

    def test_random_binomial_probability_matches_weighted_analytical_tails(self):
        probability = RandomVariable(CategoricalDistribution([0.2, 0.8], [0.25, 0.75]))
        variable = RandomVariable(BinomialDistribution(4, probability))
        expected = 0.25 * binomial_tail(4, 0.2, 3) + 0.75 * binomial_tail(4, 0.8, 3)

        result = P(variable >= 3, num_samples=NUM_SAMPLES, rng=np.random.default_rng(301))

        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_random_binomial_trial_count_matches_weighted_analytical_tails(self):
        trials = RandomVariable(CategoricalDistribution([2, 5], [0.4, 0.6]))
        variable = RandomVariable(BinomialDistribution(trials, 0.35))
        expected = 0.4 * binomial_tail(2, 0.35, 3) + 0.6 * binomial_tail(5, 0.35, 3)

        result = P(variable >= 3, num_samples=NUM_SAMPLES, rng=np.random.default_rng(302))

        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_random_normal_mean_matches_weighted_analytical_cdfs(self):
        mean = RandomVariable(CategoricalDistribution([-2.0, 2.0], [0.4, 0.6]))
        variable = RandomVariable(NormalDistribution(mean, 1.0))
        expected = 0.4 * normal_cdf(0, -2.0, 1.0) + 0.6 * normal_cdf(0, 2.0, 1.0)

        result = P(variable <= 0, num_samples=NUM_SAMPLES, rng=np.random.default_rng(303))

        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_conditioning_on_shared_poisson_rate_matches_component_distribution(self):
        rate = RandomVariable(CategoricalDistribution([1.0, 4.0], [0.7, 0.3]))
        variable = RandomVariable(PoissonDistribution(rate))

        result = P(
            variable == 0,
            given=rate == 4.0,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(304),
        )

        self.assertGreater(result.num_conditioned_samples, 0)
        self.assert_probability_matches(
            result.num_conditioned_samples / NUM_SAMPLES,
            0.3,
            NUM_SAMPLES,
        )
        self.assert_probability_matches(
            result.value,
            math.exp(-4),
            result.num_conditioned_samples,
        )


if __name__ == "__main__":
    unittest.main()
