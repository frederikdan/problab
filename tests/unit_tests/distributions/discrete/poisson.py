import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np

from problab.distributions.discrete.poisson import PoissonDistribution
from problab.value_sets.sets import NATURALS_0, NON_NEGATIVE_REALS, UNKNOWN_VALUE_SET


class PoissonDistributionTests(unittest.TestCase):
    def test_class_parameter_domain_matches_constructor(self):
        distribution = PoissonDistribution(3.)
        self.assertEqual(list(distribution._valid_parameter_sets), ["mu"])
        self.assertIs(distribution._valid_parameter_sets["mu"], NON_NEGATIVE_REALS)
        self.assertNotIn("_valid_parameter_sets", vars(distribution))

    def test_cached_realization_support_distinguishes_safe_and_unknown_rates(self):
        for unknown in (False, True):
            distribution = PoissonDistribution(3.)
            if unknown:
                distribution._parameter_nodes = (Mock(_realization_value_set=UNKNOWN_VALUE_SET),)
            with self.subTest(unknown=unknown):
                support = distribution._realization_value_set
                self.assertIs(distribution._realization_value_set, support)
                self.assertEqual(support.sympy_set, NATURALS_0.sympy_set)
                self.assertIs(support.allows_nan, unknown)
                self.assertEqual(np.float64 in support.dtype_types, unknown)
                self.assertEqual(np.object_ in support.dtype_types, unknown)
                self.assertFalse(support.allows_positive_infinity)
                self.assertFalse(support.allows_negative_infinity)

    def test_runtime_validation_delegates_to_rate_validator(self):
        distribution = PoissonDistribution(3.)
        mu = np.array([0., 3.])
        with patch(
            "problab.distributions.discrete.poisson._validate_poisson_mu_realizations",
            return_value=sentinel.mask,
        ) as check:
            self.assertEqual(distribution._validate_parameter_realizations(mu), {"mu": sentinel.mask})
        check.assert_called_once()
        self.assertIs(check.call_args.args[0], mu)

    def test_configuration_and_support(self):
        distribution = PoissonDistribution(mu=3.0)

        self.assertEqual(distribution.parameters, (3.0,))
        self.assertEqual(distribution.symbol, "Poisson")
        self.assertEqual(distribution.name, "Poisson(3.0)")
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
