import math
import unittest

import numpy as np

from problab import CategoricalDistribution, NormalDistribution, P, RandomVariable
from problab.functions import exp, floor, log, sin
from problab.value_sets.sets import INTEGERS


NUM_SAMPLES = 80_000


class DerivedFunctionStatisticalTests(unittest.TestCase):

    def assert_probability_matches(self, observed: float, expected: float) -> None:
        standard_error = math.sqrt(expected * (1 - expected) / NUM_SAMPLES)
        self.assertLessEqual(abs(observed - expected), 6 * standard_error)

    def test_floor_of_normal_matches_difference_of_analytical_cdfs(self):
        variable = RandomVariable(NormalDistribution(0.4, 1.2))

        result = P(
            floor(variable) == 1,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(601),
        )

        def cdf(value):
            return (1 + math.erf((value - 0.4) / (1.2 * math.sqrt(2)))) / 2

        self.assert_probability_matches(result.value, cdf(2) - cdf(1))

    def test_log_of_exponential_normal_matches_original_normal_tail(self):
        variable = RandomVariable(NormalDistribution(0.2, 0.8))
        threshold = 0.5
        expected = 1 - (
            1 + math.erf((threshold - 0.2) / (0.8 * math.sqrt(2)))
        ) / 2

        result = P(
            log(exp(variable)) > threshold,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(602),
        )

        self.assert_probability_matches(result.value, expected)

    def test_trigonometric_function_matches_weighted_finite_categories(self):
        variable = RandomVariable(
            CategoricalDistribution(
                [0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
                [0.1, 0.2, 0.3, 0.4],
            )
        )

        result = P(
            sin(variable) > 0.5,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(603),
        )

        self.assert_probability_matches(result.value, 0.2)

    def test_custom_function_of_two_independent_variables_matches_finite_sum(self):
        left = RandomVariable(CategoricalDistribution([0, 1, 2], [0.2, 0.3, 0.5]))
        right = RandomVariable(CategoricalDistribution([1, 2], [0.4, 0.6]))
        product = left.apply(
            lambda a, b: a * b,
            right,
            value_set=INTEGERS,
            function_name="product",
        )
        expected = 0.3 * 0.6 + 0.5 * 0.4

        result = P(
            product == 2,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(604),
            validate=True,
        )

        self.assert_probability_matches(result.value, expected)


if __name__ == "__main__":
    unittest.main()
