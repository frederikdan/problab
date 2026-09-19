import math
import unittest

import numpy as np
import sympy as sp

from problab import (
    BinomialDistribution,
    CategoricalDistribution,
    NormalDistribution,
    P,
    PoissonDistribution,
    RandomVariable,
)
from problab.functions import exp, sqrt


NUM_SAMPLES = 80_000
SIGMA_LIMIT = 6


def binomial_probability(trials: int, probability: float, successes: int) -> float:
    return math.comb(trials, successes) * probability ** successes * (
        1 - probability
    ) ** (trials - successes)


def normal_cdf(value: float, mean: float, standard_deviation: float) -> float:
    return (1 + math.erf((value - mean) / (standard_deviation * math.sqrt(2)))) / 2


class AnalyticalWorkflowStatisticalTests(unittest.TestCase):

    def assert_probability_matches(self, observed: float, expected: float, samples: int) -> None:
        self.assertGreaterEqual(expected, 0)
        self.assertLessEqual(expected, 1)
        standard_error = math.sqrt(expected * (1 - expected) / samples)
        self.assertLessEqual(
            abs(observed - expected),
            SIGMA_LIMIT * standard_error,
            f"Observed {observed}, expected {expected}, n={samples}, "
            f"standard error={standard_error}",
        )

    def test_binomial_point_probability_matches_finite_formula(self):
        variable = RandomVariable(BinomialDistribution(7, 0.35))
        expected = binomial_probability(7, 0.35, 3)

        result = P(variable == 3, num_samples=NUM_SAMPLES, rng=np.random.default_rng(201))

        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_poisson_lower_tail_matches_finite_formula(self):
        rate = 2.7
        variable = RandomVariable(PoissonDistribution(rate))
        expected = math.exp(-rate) * sum(rate ** k / math.factorial(k) for k in range(4))

        result = P(variable <= 3, num_samples=NUM_SAMPLES, rng=np.random.default_rng(202))

        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_normal_upper_tail_matches_error_function(self):
        mean, standard_deviation, threshold = -0.5, 1.8, 1.0
        variable = RandomVariable(NormalDistribution(mean, standard_deviation))
        expected = 1 - normal_cdf(threshold, mean, standard_deviation)

        result = P(variable > threshold, num_samples=NUM_SAMPLES, rng=np.random.default_rng(203))

        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_conditional_binomial_probability_matches_ratio_of_finite_sums(self):
        trials, probability = 5, 0.4
        variable = RandomVariable(BinomialDistribution(trials, probability))
        joint = sum(binomial_probability(trials, probability, k) for k in range(3, 6))
        condition = sum(binomial_probability(trials, probability, k) for k in range(1, 6))
        expected = joint / condition

        result = P(
            variable >= 3,
            given=variable >= 1,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(204),
        )

        self.assertGreater(result.num_conditioned_samples, 0)
        self.assert_probability_matches(result.value, expected, result.num_conditioned_samples)
        self.assert_probability_matches(
            result.num_conditioned_samples / NUM_SAMPLES,
            condition,
            NUM_SAMPLES,
        )

    def test_categorical_event_logic_matches_sum_of_category_weights(self):
        variable = RandomVariable(CategoricalDistribution([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]))
        event = ((variable >= 1) & (variable <= 2)) | (variable == 3)

        result = P(event, num_samples=NUM_SAMPLES, rng=np.random.default_rng(205))

        self.assert_probability_matches(result.value, 0.2 + 0.3 + 0.4, NUM_SAMPLES)

    def test_categorical_complement_and_conditional_probability_match_weights(self):
        variable = RandomVariable(
            CategoricalDistribution(["red", "blue", "green"], [0.2, 0.3, 0.5])
        )

        complement = P(~(variable == "red"), num_samples=NUM_SAMPLES, rng=np.random.default_rng(206))
        conditional = P(
            variable == "blue",
            given=variable != "red",
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(207),
        )

        self.assert_probability_matches(complement.value, 0.8, NUM_SAMPLES)
        self.assert_probability_matches(
            conditional.value,
            0.3 / 0.8,
            conditional.num_conditioned_samples,
        )

    def test_compound_object_category_probability_matches_its_weight(self):
        target = ["medium", 2]
        variable = RandomVariable(
            CategoricalDistribution(
                [("small", 1), target, {"size": "large"}],
                [0.2, 0.3, 0.5],
            )
        )

        result = P(
            variable == target,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(221),
            validate=True,
        )

        self.assert_probability_matches(result.value, 0.3, NUM_SAMPLES)

    def test_interval_closure_matches_discrete_category_weights(self):
        variable = RandomVariable(CategoricalDistribution([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]))
        cases = (
            ("both", 0.2 + 0.3),
            ("left", 0.2),
            ("right", 0.3),
            ("none", 0.0),
        )

        for offset, (closed, expected) in enumerate(cases):
            with self.subTest(closed=closed):
                result = P(
                    variable.is_in_interval(1, 2, closed=closed),
                    num_samples=NUM_SAMPLES,
                    rng=np.random.default_rng(208 + offset),
                )
                self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_finite_set_membership_matches_category_weights(self):
        variable = RandomVariable(CategoricalDistribution([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]))

        result = P(
            variable.is_in(sp.FiniteSet(1, 3)),
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(212),
        )

        self.assert_probability_matches(result.value, 0.2 + 0.4, NUM_SAMPLES)

    def test_affine_transform_of_binomial_matches_original_tail_probability(self):
        trials, probability = 6, 0.3
        variable = RandomVariable(BinomialDistribution(trials, probability))
        expected = sum(binomial_probability(trials, probability, k) for k in range(3))

        result = P(
            2 * variable + 1 <= 5,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(213),
        )

        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_nonlinear_function_composition_matches_discrete_weights(self):
        variable = RandomVariable(CategoricalDistribution([-3, -1, 2], [0.2, 0.3, 0.5]))

        result = P(
            sqrt(variable ** 2) > 1.5,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(214),
        )

        self.assert_probability_matches(result.value, 0.2 + 0.5, NUM_SAMPLES)

    def test_exponential_transform_of_normal_matches_analytical_cdf(self):
        variable = RandomVariable(NormalDistribution(0.2, 1.1))
        expected = normal_cdf(math.log(2), 0.2, 1.1)

        result = P(
            exp(variable) <= 2,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(215),
        )

        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)

    def test_reused_random_variable_keeps_its_dependence(self):
        variable = RandomVariable(CategoricalDistribution([0, 1, 2], [0.2, 0.3, 0.5]))

        result = P(
            variable + variable == 2,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(216),
        )

        self.assert_probability_matches(result.value, 0.3, NUM_SAMPLES)

    def test_complex_power_followed_by_absolute_value_matches_finite_weights(self):
        variable = RandomVariable(CategoricalDistribution([-4.0, -1.0, 9.0], [0.2, 0.3, 0.5]))

        result = P(
            abs(variable ** 0.5) > 1.5,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(218),
        )

        self.assert_probability_matches(result.value, 0.2 + 0.5, NUM_SAMPLES)

    def test_modulo_and_reverse_division_match_finite_category_weights(self):
        variable = RandomVariable(CategoricalDistribution([1, 2, 4], [0.2, 0.3, 0.5]))

        modulo = P(
            variable % 3 == 1,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(219),
        )
        reverse_division = P(
            10 / variable >= 5,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(220),
        )

        self.assert_probability_matches(modulo.value, 0.2 + 0.5, NUM_SAMPLES)
        self.assert_probability_matches(reverse_division.value, 0.2 + 0.3, NUM_SAMPLES)

    def test_sum_of_independent_binomials_matches_finite_convolution(self):
        left = RandomVariable(BinomialDistribution(2, 0.3))
        right = RandomVariable(BinomialDistribution(3, 0.4))
        expected = sum(
            binomial_probability(2, 0.3, a) * binomial_probability(3, 0.4, b)
            for a in range(3)
            for b in range(4)
            if a + b >= 3
        )

        result = P(
            left + right >= 3,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(217),
        )

        self.assert_probability_matches(result.value, expected, NUM_SAMPLES)


if __name__ == "__main__":
    unittest.main()
