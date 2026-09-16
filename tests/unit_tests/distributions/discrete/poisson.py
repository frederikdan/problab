import unittest
from unittest.mock import patch, sentinel

import numpy as np

from problab.distributions.discrete.poisson import PoissonDistribution
from problab.value_sets.sets import NATURALS_0


class PoissonDistributionTests(unittest.TestCase):
    def test_configuration_and_support(self):
        distribution = PoissonDistribution(mu=3.0)

        self.assertEqual(distribution.parameters, (3.0,))
        self.assertEqual(distribution.symbol, "unnamed_distribution")
        self.assertEqual(distribution.name, "unnamed_distribution(3.0)")
        self.assertIs(distribution.value_set, NATURALS_0)

    @patch("problab.distributions.discrete.poisson.poisson.rvs")
    def test_sample_delegates_to_scipy_with_parameters(self, rvs):
        distribution = PoissonDistribution(mu=3.0)
        rvs.return_value = sentinel.samples

        samples = distribution._sample(
            np.array(3.0),
            num_samples=100,
            rng=sentinel.rng,
        )

        self.assertIs(samples, sentinel.samples)
        rvs.assert_called_once_with(
            mu=np.array(3.0),
            size=100,
            random_state=sentinel.rng,
        )

    def test_invalid_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            PoissonDistribution(mu=-1.0)
        with self.assertRaises(TypeError):
            PoissonDistribution(mu="three")


if __name__ == "__main__":
    unittest.main()
