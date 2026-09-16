import unittest

import problab
import problab.distributions as distributions
import problab.distributions.continuous as continuous
import problab.distributions.discrete as discrete
import problab.functions as functions
import problab.probability as probability
import problab.random_variables as random_variables
import problab.value_sets as value_sets


class PublicImportTests(unittest.TestCase):

    def test_top_level_exports(self):
        expected = {
            "RandomVariable",
            "Distribution",
            "NormalDistribution",
            "BinomialDistribution",
            "PoissonDistribution",
            "CategoricalDistribution",
            "P",
            "NodeGraph",
            "ProbabilityResult",
            "ConfidenceInterval",
            "ProbabilityInterval",
            "Mode",
            "ValueSet",
            "NumericValueSet",
            "HomogeneousNumericValueSet",
            "MixedNumericValueSet",
            "ObjectValueSet",
        }

        self.assertEqual(set(problab.__all__), expected)
        for name in expected:
            self.assertTrue(hasattr(problab, name))

    def test_distribution_package_exports(self):
        self.assertEqual(
            set(distributions.__all__),
            {
                "Distribution",
                "Mode",
                "NormalDistribution",
                "BinomialDistribution",
                "CategoricalDistribution",
                "PoissonDistribution",
            },
        )
        self.assertEqual(continuous.__all__, ["NormalDistribution"])
        self.assertEqual(
            set(discrete.__all__),
            {"BinomialDistribution", "CategoricalDistribution", "PoissonDistribution"},
        )

    def test_function_package_exports(self):
        expected_names = {
            "absolute", "ceil", "floor", "sign", "sqrt",
            "exp", "log", "log2", "log10",
            "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh",
            "cos", "cosh", "sin", "sinh", "tan", "tanh",
        }

        self.assertEqual(set(functions.__all__), expected_names)
        self.assertTrue(all(callable(getattr(functions, name)) for name in expected_names))

    def test_probability_package_lazy_exports(self):
        expected_names = {
            "P",
            "ProbabilityResult",
            "ConfidenceInterval",
            "ProbabilityInterval",
        }

        self.assertEqual(set(probability.__all__), expected_names)
        self.assertTrue(all(hasattr(probability, name) for name in expected_names))
        self.assertTrue(expected_names.issubset(dir(probability)))
        with self.assertRaises(AttributeError):
            getattr(probability, "not_a_probability_export")

    def test_random_variable_package_lazy_exports(self):
        self.assertEqual(random_variables.__all__, ["RandomVariable", "NodeGraph"])
        self.assertTrue(all(hasattr(random_variables, name) for name in random_variables.__all__))
        self.assertTrue(set(random_variables.__all__).issubset(dir(random_variables)))
        with self.assertRaises(AttributeError):
            getattr(random_variables, "not_a_random_variable_export")

    def test_value_set_package_exports_core_types_and_sets(self):
        expected_names = {
            "ValueSet",
            "NumericValueSet",
            "HomogeneousNumericValueSet",
            "MixedNumericValueSet",
            "ObjectValueSet",
            "REALS",
            "INTEGERS",
            "COMPLEXES",
            "UNKNOWN_VALUE_SET",
        }

        self.assertTrue(expected_names.issubset(value_sets.__all__))
        self.assertTrue(all(hasattr(value_sets, name) for name in expected_names))


if __name__ == "__main__":
    unittest.main()
