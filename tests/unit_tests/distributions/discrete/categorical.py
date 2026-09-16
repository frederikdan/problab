import unittest
from fractions import Fraction

import numpy as np

from problab.distributions.discrete.categorical import CategoricalDistribution
from problab.value_sets import MixedNumericValueSet, ObjectValueSet


class _RecordingRng:

    def __init__(self, indices):
        self.indices = np.asarray(indices)
        self.calls = []

    def choice(self, population_size, size, p):
        self.calls.append((population_size, size, p))
        return self.indices


class CategoricalDistributionTests(unittest.TestCase):

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

    def test_mixed_numeric_categories_preserve_original_types(self):
        distribution = CategoricalDistribution([1, 4.0], [0.5, 0.5])
        rng = _RecordingRng([0, 1])

        samples = distribution._sample(num_samples=2, rng=rng)

        self.assertIsInstance(distribution.value_set, MixedNumericValueSet)
        self.assertEqual(type(samples[0]), int)
        self.assertEqual(type(samples[1]), float)

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
