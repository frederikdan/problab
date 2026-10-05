import unittest
from unittest.mock import patch

import numpy as np

from problab import CategoricalDistribution, NormalDistribution, NodeGraph, P, RandomVariable
from problab.functions import exp


class AuditEdgeCaseTests(unittest.TestCase):
    def test_normal_rejects_python_and_numpy_booleans_with_type_error(self):
        for value in (True, np.bool_(True)):
            for parameter in ("mean", "std"):
                with self.subTest(value=type(value).__name__, parameter=parameter):
                    arguments = {"mean": 0.0, "std": 1.0, parameter: value}
                    with self.assertRaises(TypeError):
                        NormalDistribution(**arguments)

    def test_public_graph_rejects_non_integer_limits(self):
        variable = RandomVariable(CategoricalDistribution([1], [1.0]))
        for limit in (True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(TypeError):
                NodeGraph((variable._node,), max_size=limit)

    def test_complement_conditioning_returns_zero(self):
        variable = RandomVariable(CategoricalDistribution([0, 1], [0.5, 0.5]))
        event = variable == 1
        result = P(event, given=~event, num_samples=100, rng=np.random.default_rng(42))
        self.assertGreater(result.num_conditioned_samples, 0)
        self.assertEqual(result.value, 0.0)

    def test_mixed_numeric_categories_support_exponential(self):
        variable = RandomVariable(CategoricalDistribution([1, 2.0], [0.5, 0.5]))
        samples = exp(variable).sample(num_samples=20, rng=np.random.default_rng(42))
        self.assertTrue(np.all(np.isclose(samples, np.exp(1)) | np.isclose(samples, np.exp(2))))

    def test_multiplication_underflow_passes_realization_validation(self):
        variable = RandomVariable(CategoricalDistribution([1e-200], [1.0]))
        np.testing.assert_array_equal((variable * variable).sample(validate=True), [0.0])

    def test_sample_third_positional_argument_still_means_validate(self):
        variable = RandomVariable(CategoricalDistribution([1], [1.0]))
        rng = np.random.default_rng(42)
        with patch("problab.random_variables.base._RealizationContext") as context:
            variable.sample(1, rng, True)
        self.assertIs(context.call_args.kwargs["validate"], True)

    def test_graph_limit_error_reports_requested_limit(self):
        variable = RandomVariable(CategoricalDistribution([1], [1.0])) + 2
        with self.assertRaisesRegex(ValueError, "maximum size of 1 nodes"):
            variable.sample(max_graph_size=1)

    def test_non_numeric_cdf_rejects_scalar_and_array_consistently(self):
        distribution = CategoricalDistribution(["red"], [1.0])
        for value in (1.0, np.array([1.0])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                distribution.cdf(value, num_samples=1)


if __name__ == "__main__":
    unittest.main()
