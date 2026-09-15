import unittest

from problab import CategoricalDistribution, P, RandomVariable


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


if __name__ == "__main__":
    unittest.main()
