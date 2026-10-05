import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from problab.distributions.discrete.categorical import CategoricalDistribution
from problab.value_sets import HomogeneousNumericValueSet, MixedNumericValueSet, ObjectValueSet


class _RecordingRng:

    def __init__(self, indices):
        self.indices = np.asarray(indices)
        self.calls = []

    def choice(self, population_size, size, p):
        self.calls.append((population_size, size, p))
        return self.indices


class CategoricalDistributionTests(unittest.TestCase):

    def test_numeric_categories_are_computed_once_and_reused_for_sampling(self):
        path = "problab.distributions.discrete.categorical._attempt_make_categories_numeric"
        with patch(path, return_value=np.array([1.0, 4.0])) as convert:
            distribution = CategoricalDistribution([1, 4.0], [0.5, 0.5])
            first = distribution._categories_numeric
            second = distribution._categories_numeric
            np.testing.assert_array_equal(distribution._sample(num_samples=2, rng=_RecordingRng([1, 0])), [4.0, 1.0])
        self.assertIs(first, second)
        convert.assert_called_once_with((1, 4.0), distribution.value_set)

    def test_lossy_conversion_falls_back_to_exact_object_samples(self):
        categories = (2 ** 63 - 1, 2 ** 63)
        distribution = CategoricalDistribution(categories, [0.5, 0.5])
        self.assertIsNone(distribution._categories_numeric)
        samples = distribution._sample(num_samples=2, rng=_RecordingRng([0, 1]))
        self.assertEqual(samples.dtype, np.dtype(object))
        self.assertEqual(tuple(samples), categories)
        self.assertIsInstance(distribution._realization_value_set, HomogeneousNumericValueSet)
        self.assertEqual(distribution._realization_value_set.dtype_types, (np.object_,))

    def test_realization_support_is_cached_and_matches_numeric_samples(self):
        distribution = CategoricalDistribution([1, 4.0], [0.5, 0.5])
        support = distribution._realization_value_set
        self.assertIs(support, distribution._realization_value_set)
        self.assertIsInstance(support, HomogeneousNumericValueSet)
        self.assertEqual(support.sympy_set, distribution.value_set.sympy_set)
        self.assertEqual(support.dtype_types, (np.float64,))

    def test_non_finite_categories_get_explicit_realization_permissions(self):
        distribution = CategoricalDistribution([np.inf, -np.inf, np.nan], [0.25, 0.25, 0.5])
        support = distribution._realization_value_set
        self.assertTrue(support.allows_positive_infinity)
        self.assertTrue(support.allows_negative_infinity)
        self.assertTrue(support.allows_nan)
        np.testing.assert_array_equal(distribution._sample(num_samples=3, rng=_RecordingRng([0, 1, 2])), [np.inf, -np.inf, np.nan])

    def test_object_categories_reuse_mathematical_support_for_realizations(self):
        distribution = CategoricalDistribution(["red", "blue"], [0.5, 0.5])
        self.assertIsNone(distribution._categories_numeric)
        self.assertIs(distribution._realization_value_set, distribution.value_set)

    def test_constructor_preserves_categories_and_probabilities(self):
        distribution = CategoricalDistribution(
            categories=["red", "blue"],
            probabilities=[0.25, 0.75],
        )

        self.assertEqual(distribution.categories, ("red", "blue"))
        self.assertEqual(distribution.probabilities, (0.25, 0.75))
        self.assertEqual(distribution.parameters, ())
        self.assertIsInstance(distribution.value_set, ObjectValueSet)

    def test_equal_categories_are_merged_and_probabilities_are_summed(self):
        distribution = CategoricalDistribution(
            categories=["red", "red", "blue"],
            probabilities=[0.2, 0.3, 0.5],
        )

        self.assertEqual(distribution.categories, ("red", "blue"))
        self.assertEqual(distribution.probabilities, (0.5, 0.5))

    def test_numeric_categories_use_numeric_value_set(self):
        distribution = CategoricalDistribution([1, 2, 3], [0.2, 0.3, 0.5])

        self.assertNotIsInstance(distribution.value_set, ObjectValueSet)
        self.assertTrue(distribution.value_set.contains(2))

    def test_mixed_numeric_categories_preserve_inputs_and_sample_lossless_numeric_array(self):
        distribution = CategoricalDistribution([1, 4.0], [0.5, 0.5])
        rng = _RecordingRng([0, 1])

        samples = distribution._sample(num_samples=2, rng=rng)

        self.assertIsInstance(distribution.value_set, MixedNumericValueSet)
        self.assertIs(type(distribution.categories[0]), int)
        self.assertIs(type(distribution.categories[1]), float)
        self.assertEqual(samples.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(samples, [1.0, 4.0])

    def test_sample_uses_rng_indices_and_configured_probabilities(self):
        distribution = CategoricalDistribution(
            categories=["red", "blue"],
            probabilities=[0.25, 0.75],
        )
        rng = _RecordingRng([1, 0, 1])

        samples = distribution._sample(num_samples=3, rng=rng)

        self.assertEqual(samples.tolist(), ["blue", "red", "blue"])
        self.assertEqual(rng.calls, [(2, 3, (0.25, 0.75))])

    def test_configuration_validation_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            CategoricalDistribution([], [])
        with self.assertRaises(ValueError):
            CategoricalDistribution(["red"], [0.5, 0.5])
        with self.assertRaises(TypeError):
            CategoricalDistribution(["red"], ["one"])
        with self.assertRaises(ValueError):
            CategoricalDistribution(["red"], [-0.1])
        with self.assertRaises(ValueError):
            CategoricalDistribution(["red"], [0.5])

    def test_fraction_and_integer_categories_use_mixed_numeric_values(self):
        distribution = CategoricalDistribution(
            categories=(1, Fraction(1, 2)),
            probabilities=(0.5, 0.5),
        )

        self.assertIsInstance(distribution.value_set, MixedNumericValueSet)
        self.assertTrue(distribution.value_set.contains(Fraction(1, 2)))


if __name__ == "__main__":
    unittest.main()
