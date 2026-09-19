import unittest

from problab import CategoricalDistribution, P, RandomVariable


class CategoricalValueRegressionTests(unittest.TestCase):

    def test_duplicate_categories_are_merged_before_sampling(self):
        distribution = CategoricalDistribution(
            categories=["red", "red", "blue"],
            probabilities=[0.2, 0.3, 0.5],
        )

        self.assertEqual(distribution.categories, ("red", "blue"))
        self.assertEqual(distribution.probabilities, (0.5, 0.5))

    def test_compound_category_is_an_atomic_value_during_probability_evaluation(self):
        category = ["large", {"priority": 1}]
        variable = RandomVariable(CategoricalDistribution([category], [1.0]))

        result = P(variable == category, num_samples=5, validate=True)

        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.num_successes, 5)


if __name__ == "__main__":
    unittest.main()
