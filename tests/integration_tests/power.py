import unittest

import numpy as np

from problab import CategoricalDistribution, NormalDistribution, RandomVariable
from problab.functions import sqrt


class RandomVariablePowerIntegrationTests(unittest.TestCase):

    def test_real_power_uses_real_samples(self):
        variable = RandomVariable(NormalDistribution(0, 1))

        for exponent in (2, 2.0):
            squared = variable ** exponent
            square_root = sqrt(squared)
            samples = square_root.sample(num_samples=32, rng=np.random.default_rng(1))

            self.assertTrue(np.issubdtype(samples.dtype, np.floating))
            self.assertTrue(np.all(samples >= 0))

    def test_complex_power_keeps_complex_samples(self):
        variable = RandomVariable(CategoricalDistribution([-4.0], [1.0]))

        samples = (variable ** 0.5).sample(num_samples=4)

        self.assertTrue(np.iscomplexobj(samples))
        np.testing.assert_allclose(samples, 2j)


if __name__ == "__main__":
    unittest.main()
