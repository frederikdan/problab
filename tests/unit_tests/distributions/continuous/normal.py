import unittest
from unittest.mock import patch, sentinel

import numpy as np

from problab.distributions.continuous.normal import NormalDistribution
from problab.value_sets.sets import REALS, POSITIVE_REALS


class NormalDistributionTests(unittest.TestCase):
    def test_class_parameter_domains_match_constructor_order(self):
        distribution = NormalDistribution(0., 1.)
        self.assertEqual(list(distribution._valid_parameter_sets), ["mean", "std"])
        self.assertIs(distribution._valid_parameter_sets["mean"], REALS)
        self.assertIs(distribution._valid_parameter_sets["std"], POSITIVE_REALS)
        self.assertNotIn("_valid_parameter_sets", vars(distribution))

    def test_realization_support_is_cached_and_tracks_parameter_risk(self):
        for guaranteed in (True, False):
            distribution = NormalDistribution(0., 1.)
            with self.subTest(guaranteed=guaranteed), patch.object(
                distribution, "_parameter_realizations_guaranteed_valid", return_value=guaranteed,
            ) as check:
                support = distribution._realization_value_set
                self.assertIs(distribution._realization_value_set, support)
                self.assertEqual(support.sympy_set, REALS.sympy_set)
                self.assertEqual(support.dtype_types, (np.floating,))
                self.assertTrue(support.allows_positive_infinity)
                self.assertTrue(support.allows_negative_infinity)
                self.assertIs(support.allows_nan, not guaranteed)
                self.assertIs(distribution.value_set, REALS)
                check.assert_called_once_with()

    def test_runtime_validation_delegates_each_parameter_to_its_validator(self):
        distribution = NormalDistribution(0., 1.)
        mean, std = np.array([0., 1.]), np.array([1., 2.])
        mean_mask, std_mask = np.array([True, False]), np.array([True, True])
        path = "problab.distributions.continuous.normal"
        with patch(path + "._validate_normal_mean_realizations", return_value=mean_mask) as mean_check, patch(
            path + "._validate_normal_std_realizations", return_value=std_mask,
        ) as std_check:
            masks = distribution._validate_parameter_realizations(mean, std)
        self.assertEqual(set(masks), {"mean", "std"})
        self.assertIs(masks["mean"], mean_mask)
        self.assertIs(masks["std"], std_mask)
        mean_check.assert_called_once()
        std_check.assert_called_once()
        self.assertIs(mean_check.call_args.args[0], mean)
        self.assertIs(std_check.call_args.args[0], std)

    def test_configuration_and_public_properties(self):
        distribution = NormalDistribution(mean=2.0, std=3.0)

        self.assertEqual(distribution.parameters, (2.0, 3.0))
        self.assertEqual(distribution.symbol, NormalDistribution.symbol)
        self.assertEqual(distribution.name, "Normal(2.0, 3.0)")
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
