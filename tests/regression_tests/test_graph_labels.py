import unittest

from problab import CategoricalDistribution, PoissonDistribution, RandomVariable
from problab.functions import sqrt


class GraphLabelRegressionTests(unittest.TestCase):

    def test_poisson_name_identifies_the_distribution(self):
        self.assertIn("Poisson", PoissonDistribution(2.0).name)

    def test_categorical_name_identifies_the_distribution(self):
        self.assertIn("Categorical", CategoricalDistribution(["red"], [1.0]).name)

    def test_mathematical_function_node_names_the_function(self):
        variable = RandomVariable(CategoricalDistribution([4.0], [1.0]), name="height")

        result = sqrt(variable)

        self.assertIn("sqrt", result._node.name)
        self.assertIn("height", result._node.name)


if __name__ == "__main__":
    unittest.main()
