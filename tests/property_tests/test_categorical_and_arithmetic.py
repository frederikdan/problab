import unittest

import numpy as np

from problab import CategoricalDistribution, RandomVariable


class CategoricalAndArithmeticPropertyTests(unittest.TestCase):

    def test_categorical_configuration_merges_equal_values_and_normalizes_probabilities(self):
        generator = np.random.default_rng(17)

        for _ in range(100):
            category_count = int(generator.integers(1, 9))
            categories = tuple(int(value) for value in generator.integers(-4, 5, category_count))
            raw_probabilities = generator.uniform(0.01, 1.0, category_count)
            probabilities = tuple(raw_probabilities / raw_probabilities.sum())

            distribution = CategoricalDistribution(categories, probabilities)

            expected_probabilities = {}
            for category, probability in zip(categories, probabilities):
                expected_probabilities[category] = (
                    expected_probabilities.get(category, 0.0) + probability
                )

            self.assertEqual(distribution.categories, tuple(expected_probabilities))
            np.testing.assert_allclose(
                distribution.probabilities,
                tuple(expected_probabilities.values()),
            )
            self.assertAlmostEqual(sum(distribution.probabilities), 1.0)

    def test_arithmetic_identities_hold_for_many_integer_categorical_variables(self):
        generator = np.random.default_rng(18)

        for value in generator.integers(-100, 101, 100):
            with self.subTest(value=int(value)):
                variable = RandomVariable(
                    CategoricalDistribution([int(value)], [1.0]),
                )

                for expression in (variable + 0, variable - 0, variable * 1, variable ** 1):
                    samples = expression.sample(num_samples=3, validate=True)
                    np.testing.assert_array_equal(samples, [value, value, value])

    def test_compound_categories_remain_atomic_values_through_sampling(self):
        categories = (
            ("small", 1),
            ["medium", 2],
            {"size": "large"},
        )

        for category in categories:
            with self.subTest(category=category):
                variable = RandomVariable(CategoricalDistribution([category], [1.0]))

                samples = variable.sample(num_samples=4, validate=True)

                self.assertTrue(all(value == category for value in samples))


if __name__ == "__main__":
    unittest.main()
