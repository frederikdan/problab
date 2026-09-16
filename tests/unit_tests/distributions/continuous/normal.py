import unittest
from unittest.mock import patch, sentinel

import numpy as np

from problab.distributions.continuous.normal import NormalDistribution
from problab.value_sets.sets import REALS


class NormalDistributionTests(unittest.TestCase):
    def test_configuration_and_public_properties(self):
        distribution = NormalDistribution(mean=2.0, std=3.0)

        self.assertEqual(distribution.parameters, (2.0, 3.0))
        self.assertEqual(distribution.symbol, "N")
        self.assertEqual(distribution.name, "N(2.0, 3.0)")
        self.assertIs(distribution.value_set, REALS)

    @patch("problab.distributions.continuous.normal.norm.rvs")
    def test_sample_delegates_to_scipy_with_parameters(self, rvs):
        distribution = NormalDistribution(mean=2.0, std=3.0)
        rvs.return_value = sentinel.samples

        samples = distribution._sample(
            np.array(2.0),
            np.array(3.0),
            num_samples=16,
            rng=sentinel.rng,
        )

        self.assertIs(samples, sentinel.samples)
        rvs.assert_called_once_with(
            loc=np.array(2.0),
            scale=np.array(3.0),
            size=16,
            random_state=sentinel.rng,
        )

    def test_invalid_mean_and_standard_deviation_are_rejected(self):
        with self.assertRaises(TypeError):
            NormalDistribution(mean="two", std=1.0)
        with self.assertRaises(TypeError):
            NormalDistribution(mean=0.0, std="one")
        with self.assertRaises(ValueError):
            NormalDistribution(mean=0.0, std=0.0)
        with self.assertRaises(ValueError):
            NormalDistribution(mean=0.0, std=-1.0)


if __name__ == "__main__":
    unittest.main()
