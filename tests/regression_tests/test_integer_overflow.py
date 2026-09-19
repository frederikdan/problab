import unittest

import numpy as np

from problab import CategoricalDistribution, RandomVariable


class IntegerOverflowRegressionTests(unittest.TestCase):

    def test_fixed_width_integer_addition_does_not_silently_wrap(self):
        variable = RandomVariable(CategoricalDistribution([np.int8(100)], [1.0]))

        try:
            samples = (variable + variable).sample(num_samples=4)
        except OverflowError:
            return

        self.assertEqual(samples.tolist(), [200, 200, 200, 200])

    def test_fixed_width_integer_multiplication_does_not_silently_wrap(self):
        variable = RandomVariable(CategoricalDistribution([np.int8(100)], [1.0]))

        try:
            samples = (variable * variable).sample(num_samples=4)
        except OverflowError:
            return

        self.assertEqual(samples.tolist(), [10000, 10000, 10000, 10000])


if __name__ == "__main__":
    unittest.main()
