import unittest

import numpy as np

from problab import CategoricalDistribution, RandomVariable
from problab.functions import sqrt


class RealPowerRegressionTests(unittest.TestCase):

    def test_square_of_a_negative_value_remains_compatible_with_sqrt(self):
        source = RandomVariable(CategoricalDistribution([-2], [1.0]))

        samples = sqrt(source ** 2).sample(num_samples=4, validate=True)

        self.assertTrue(np.issubdtype(samples.dtype, np.floating))
        np.testing.assert_array_equal(samples, [2.0, 2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
