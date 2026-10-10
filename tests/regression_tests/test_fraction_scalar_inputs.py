import unittest
from fractions import Fraction

import numpy as np

from problab import CategoricalDistribution, Mode, ProbabilityResult, RandomVariable


class FractionScalarInputRegressionTests(unittest.TestCase):
    def test_categorical_fraction_probabilities_merge_and_sample_as_floats(self):
        distribution = CategoricalDistribution(
            ["a", "a", "b"], [Fraction(1, 6), Fraction(1, 6), Fraction(2, 3)]
        )
        reference = CategoricalDistribution(["a", "b"], [1 / 3, 2 / 3])
        self.assertEqual(distribution.categories, ("a", "b"))
        self.assertEqual(distribution.probabilities, reference.probabilities)
        self.assertTrue(all(type(value) is float for value in distribution.probabilities))
        np.testing.assert_array_equal(
            distribution.sample(20, rng=np.random.default_rng(17), validate=True),
            reference.sample(20, rng=np.random.default_rng(17), validate=True),
        )

    def test_categorical_rejects_negative_fraction_even_when_it_rounds_to_zero(self):
        tiny = Fraction(1, 10**400)
        for probabilities in ((-tiny, 1 + tiny), (Fraction(1, 3), Fraction(1, 3))):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                CategoricalDistribution(["a", "b"], probabilities)

    def test_scalar_fraction_cdf_and_ppf_match_float_arguments(self):
        distribution = CategoricalDistribution([0, 1, 2], [0.25, 0.5, 0.25])
        for method in (distribution.cdf, distribution.ppf):
            for value in (Fraction(0), Fraction(1, 3), Fraction(1)):
                with self.subTest(method=method.__name__, value=value):
                    actual = method(value, mode=Mode.MONTE_CARLO, num_samples=100,
                                    rng=np.random.default_rng(23))
                    expected = method(float(value), mode=Mode.MONTE_CARLO, num_samples=100,
                                      rng=np.random.default_rng(23))
                    self.assertEqual(actual, expected)

    def test_ppf_rejects_out_of_range_fractions_that_round_to_valid_endpoints(self):
        distribution = CategoricalDistribution([0], [1.0])
        tiny = Fraction(1, 10**400)
        for value in (-tiny, 1 + tiny):
            with self.subTest(value=value), self.assertRaises(ValueError):
                distribution.ppf(value, mode=Mode.MONTE_CARLO, num_samples=1)

    def test_random_variable_interval_accepts_fraction_alpha(self):
        variable = RandomVariable(CategoricalDistribution([0, 1, 2], [0.25, 0.5, 0.25]))
        actual = variable.interval(Fraction(1, 10), num_samples=100, rng=np.random.default_rng(29))
        expected = variable.interval(0.1, num_samples=100, rng=np.random.default_rng(29))
        self.assertEqual(actual, expected)
        self.assertIs(type(actual.alpha), float)

    def test_quantile_intervals_accept_fraction_q_and_alpha_through_both_public_paths(self):
        distribution = CategoricalDistribution([0, 1, 2], [0.25, 0.5, 0.25])
        for source in (distribution, RandomVariable(distribution)):
            with self.subTest(source=type(source).__name__):
                actual = source.quantile_confidence_interval(
                    Fraction(1, 2), alpha=Fraction(1, 10), num_samples=100,
                    rng=np.random.default_rng(31),
                )
                expected = source.quantile_confidence_interval(
                    0.5, alpha=0.1, num_samples=100, rng=np.random.default_rng(31),
                )
                self.assertEqual(actual, expected)
                self.assertIs(type(actual.alpha), float)

    def test_probability_result_accepts_fraction_value_and_confidence_level(self):
        result = ProbabilityResult(value=Fraction(1, 3), num_successes=1, num_unconditioned_samples=3)
        actual = result.confidence_interval(Fraction(1, 10))
        expected = result.confidence_interval(0.1)
        self.assertEqual(actual, expected)
        self.assertIs(type(actual.alpha), float)
        with self.assertRaises(ValueError):
            ProbabilityResult(value=Fraction(2, 3), num_successes=1, num_unconditioned_samples=3)

    def test_open_probability_parameters_reject_fractions_rounding_to_endpoints(self):
        distribution = CategoricalDistribution([0], [1.0])
        variable = RandomVariable(distribution)
        result = ProbabilityResult(value=1.0, num_successes=1, num_unconditioned_samples=1)
        tiny = Fraction(1, 10**400)
        for value in (tiny, 1 - tiny):
            for call in (
                lambda: variable.interval(value, num_samples=10),
                lambda: distribution.quantile_confidence_interval(value, num_samples=10),
                lambda: distribution.quantile_confidence_interval(0.5, alpha=value, num_samples=10),
                lambda: result.confidence_interval(value),
            ):
                with self.subTest(value=value, call=call), self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
