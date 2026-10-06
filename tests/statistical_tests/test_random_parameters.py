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
from problab.value_sets import HomogeneousNumericValueSet
from problab.value_sets.sets import NON_NEGATIVE_REALS, POSITIVE_REALS, REALS, UNIT_INTERVAL


NUM_SAMPLES = 100_000


def binomial_tail(trials: int, probability: float, minimum: int) -> float:
    return sum(
        math.comb(trials, k) * probability ** k * (1 - probability) ** (trials - k)
        for k in range(minimum, trials + 1)
    )


def normal_cdf(value: float, mean: float, standard_deviation: float) -> float:
    return (1 + math.erf((value - mean) / (standard_deviation * math.sqrt(2)))) / 2


class RandomParameterStatisticalTests(unittest.TestCase):

    def test_masked_batches_preserve_the_analytical_law_at_valid_positions(self):
        cases = (
            (POSITIVE_REALS, [1., -1., 2.], lambda p: NormalDistribution(0, p, parameter_risk_policy="ignore"),
             lambda x: x <= 1, 0.5 * normal_cdf(1, 0, 1) + 0.5 * normal_cdf(1, 0, 2)),
            (UNIT_INTERVAL, [0.2, -1., 0.8], lambda p: BinomialDistribution(4, p, parameter_risk_policy="ignore"),
             lambda x: x >= 3, 0.5 * binomial_tail(4, 0.2, 3) + 0.5 * binomial_tail(4, 0.8, 3)),
            (NON_NEGATIVE_REALS, [1., -1., 4.], lambda p: PoissonDistribution(p, parameter_risk_policy="ignore"),
             lambda x: x == 0, 0.5 * math.exp(-1) + 0.5 * math.exp(-4)),
        )
        for mathematical_set, values, constructor, event, expected in cases:
            realized = np.tile(values, 20_000)
            source = RandomVariable(CategoricalDistribution([1.], [1.]))
            parameter = source.apply(
                lambda x: realized.copy(), vectorized=True,
                mathematical_value_set=mathematical_set,
                realization_value_set=HomogeneousNumericValueSet(REALS.sympy_set, (np.float64,)),
            )
            distribution = constructor(parameter)
            with self.subTest(distribution=distribution.symbol):
                samples = RandomVariable(distribution).sample(
                    len(realized), rng=np.random.default_rng(842), numerical_error_policy="ignore",
                )
                invalid = np.tile([False, True, False], 20_000)
                np.testing.assert_array_equal(np.isnan(samples), invalid)
                observed = np.mean(event(samples[~invalid]))
                self.assert_probability_matches(observed, expected, np.count_nonzero(~invalid))

    def test_mixed_integer_float_normal_mean_matches_analytical_mixture(self):
        mean = RandomVariable(CategoricalDistribution([0, 2.0], [0.4, 0.6]))
        variable = RandomVariable(NormalDistribution(mean, 1.0, parameter_risk_policy="raise"))
        expected = 0.4 * normal_cdf(1, 0, 1) + 0.6 * normal_cdf(1, 2, 1)
        result = P(variable <= 1, num_samples=NUM_SAMPLES, rng=np.random.default_rng(305))
        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_mixed_integer_float_poisson_rate_matches_weighted_zero_probability(self):
        rate = RandomVariable(CategoricalDistribution([1, 4.0], [0.25, 0.75]))
        variable = RandomVariable(PoissonDistribution(rate, parameter_risk_policy="raise"))
        expected = 0.25 * math.exp(-1) + 0.75 * math.exp(-4)
        result = P(variable == 0, num_samples=NUM_SAMPLES, rng=np.random.default_rng(306))
        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

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
