import unittest

import numpy as np

import problab
from problab import (
    BinomialDistribution,
    CategoricalDistribution,
    NormalDistribution,
    RandomVariable,
)
from problab.distributions import NormalDistribution as PackageNormalDistribution
from problab.random_variables import RandomVariable as PackageRandomVariable


class PublicContractTests(unittest.TestCase):

    def test_public_import_paths_share_class_identity(self):
        self.assertIs(NormalDistribution, PackageNormalDistribution)
        self.assertIs(RandomVariable, PackageRandomVariable)

    def test_star_import_exposes_exactly_the_top_level_public_names(self):
        namespace = {}

        exec("from problab import *", namespace)

        self.assertEqual(set(namespace) - {"__builtins__"}, set(problab.__all__))

    def test_distribution_parameters_preserve_public_random_variable_inputs(self):
        parameter = RandomVariable(CategoricalDistribution([1.0], [1.0]))

        distribution = NormalDistribution(parameter, 2.0)

        self.assertIs(distribution.parameters[0], parameter)
        self.assertEqual(distribution.parameters[1], 2.0)

    def test_categorical_configuration_is_read_only_and_outside_graph_parameters(self):
        distribution = CategoricalDistribution(["red", "blue"], [0.25, 0.75])

        self.assertEqual(distribution.parameters, ())
        self.assertEqual(distribution.categories, ("red", "blue"))
        self.assertEqual(distribution.probabilities, (0.25, 0.75))
        with self.assertRaises(AttributeError):
            distribution.categories = ("green",)
        with self.assertRaises(AttributeError):
            distribution.probabilities = (1.0,)

    def test_public_distribution_sample_accepts_count_and_seeded_generator(self):
        distribution = BinomialDistribution(3, 1.0)

        samples = distribution.sample(num_samples=5, rng=np.random.default_rng(701))

        np.testing.assert_array_equal(samples, [3, 3, 3, 3, 3])

    def test_public_distribution_sample_can_validate_realizations(self):
        distribution = CategoricalDistribution(["red"], [1.0])

        samples = distribution.sample(validate=True)

        self.assertEqual(samples.tolist(), ["red"])

    def test_random_variable_can_raise_its_graph_size_limit(self):
        expression = RandomVariable(CategoricalDistribution([1], [1.0]))
        for _ in range(110):
            expression = expression + 1

        samples = expression.sample(num_samples=1, max_graph_size=250)

        self.assertEqual(samples.tolist(), [111])


if __name__ == "__main__":
    unittest.main()
