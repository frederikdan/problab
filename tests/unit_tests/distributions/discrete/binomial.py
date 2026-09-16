import unittest
from unittest.mock import patch, sentinel

import numpy as np
import sympy as sp

from problab.distributions.discrete.binomial import BinomialDistribution


class BinomialDistributionTests(unittest.TestCase):
    def test_configuration_and_finite_support(self):
        distribution = BinomialDistribution(n=4, p=0.25)

        self.assertEqual(distribution.parameters, (4, 0.25))
        self.assertEqual(distribution.symbol, "Bin")
        self.assertEqual(distribution.name, "Bin(4, 0.25)")
        self.assertEqual(distribution.value_set.sympy_set, sp.FiniteSet(0, 1, 2, 3, 4))
        self.assertEqual(distribution.value_set.dtype_types, (np.integer,))

    def test_zero_n_has_single_value_support(self):
        distribution = BinomialDistribution(n=0, p=0.8)

        self.assertEqual(distribution.value_set.sympy_set, sp.FiniteSet(0))

    @patch("problab.distributions.discrete.binomial.binom.rvs")
    def test_sample_delegates_to_scipy_with_parameters(self, rvs):
        distribution = BinomialDistribution(n=4, p=0.25)
        rvs.return_value = sentinel.samples

        samples = distribution._sample(
            np.array(4),
            np.array(0.25),
            num_samples=100,
            rng=sentinel.rng,
        )

        self.assertIs(samples, sentinel.samples)
        rvs.assert_called_once_with(
            n=np.array(4),
            p=np.array(0.25),
            size=100,
            random_state=sentinel.rng,
        )

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(TypeError):
            BinomialDistribution(n=1.0, p=0.5)
        with self.assertRaises(ValueError):
            BinomialDistribution(n=-1, p=0.5)
        with self.assertRaises(ValueError):
            BinomialDistribution(n=1, p=-0.1)
        with self.assertRaises(ValueError):
            BinomialDistribution(n=1, p=1.1)


if __name__ == "__main__":
    unittest.main()
