import unittest

from problab import CategoricalDistribution, RandomVariable
from problab.functions import exp, tanh


class FloatBoundaryValidationRegressionTests(unittest.TestCase):

    def test_tanh_rounding_to_one_remains_valid(self):
        variable = RandomVariable(CategoricalDistribution([20.0], [1.0]))

        samples = tanh(variable).sample(num_samples=2, validate=True)

        self.assertEqual(samples.tolist(), [1.0, 1.0])

    def test_exponential_underflow_to_zero_remains_valid(self):
        variable = RandomVariable(CategoricalDistribution([-1000.0], [1.0]))

        samples = exp(variable).sample(num_samples=2, validate=True)

        self.assertEqual(samples.tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
