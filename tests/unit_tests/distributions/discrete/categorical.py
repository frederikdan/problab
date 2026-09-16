import unittest
import numpy as np

from problab import CategoricalDistribution, P, RandomVariable
from problab.value_sets import MixedNumericValueSet


class CategoricalAtomicValueTests(unittest.TestCase):
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

    def test_unsafe_mixed_numeric_categories_keep_their_types(self):
        categories = (1, 4.0)
        distribution = CategoricalDistribution(categories, [0.5, 0.5])

        self.assertIsInstance(distribution.value_set, MixedNumericValueSet)
        self.assertTrue(distribution.value_set.contains(4.0))

        samples = RandomVariable(distribution).sample(
            num_samples=20,
            rng=np.random.default_rng(1),
        )
        self.assertTrue(any(type(value) is int for value in samples))
        self.assertTrue(any(type(value) is float for value in samples))


if __name__ == "__main__":
    unittest.main()
