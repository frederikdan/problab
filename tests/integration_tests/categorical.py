import unittest
import warnings

import numpy as np

from problab import CategoricalDistribution, P, RandomVariable


class CategoricalRandomVariableIntegrationTests(unittest.TestCase):

    def test_declared_non_finite_categories_validate_under_every_sampling_policy(self):
        for value in (np.inf, -np.inf, np.nan, complex(np.inf, np.nan)):
            variable = RandomVariable(CategoricalDistribution([value], [1.0]))
            for policy in ("warn", "raise", "ignore"):
                with self.subTest(value=value, policy=policy), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    samples = variable.sample(num_samples=3, validate=True, numerical_error_policy=policy)
                    np.testing.assert_array_equal(samples, np.full(3, value))
                    self.assertEqual(caught, [])

    def test_tuple_category_is_compared_as_one_value(self):
        category = (1, 2)
        variable = RandomVariable(CategoricalDistribution([category], [1.0]))

        result = P(variable == category, num_samples=3)

        self.assertEqual(result.value, 1.0)

    def test_list_category_is_compared_as_one_value(self):
        category = [1, 2]
        variable = RandomVariable(CategoricalDistribution([category], [1.0]))

        result = P(variable == category, num_samples=3)

        self.assertEqual(result.value, 1.0)

    def test_object_categories_support_realization_validation(self):
        category = {"status": "finished"}
        variable = RandomVariable(CategoricalDistribution([category], [1.0]))

        samples = variable.sample(num_samples=3, validate=True)

        self.assertEqual(samples.tolist(), [category, category, category])

    def test_mixed_numeric_categories_produce_validated_numeric_samples(self):
        variable = RandomVariable(CategoricalDistribution([1, 4.0], [0.5, 0.5]))

        samples = variable.sample(
            num_samples=20,
            rng=np.random.default_rng(1),
            validate=True,
        )

        self.assertEqual(samples.dtype, np.dtype(np.float64))
        self.assertTrue(np.isin(samples, [1.0, 4.0]).all())
        np.testing.assert_array_equal((variable + 1).sample(num_samples=20, rng=np.random.default_rng(1), validate=True), samples + 1)


if __name__ == "__main__":
    unittest.main()
