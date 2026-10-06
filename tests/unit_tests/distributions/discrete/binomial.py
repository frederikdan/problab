import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np
import sympy as sp

from problab.distributions.discrete.binomial import BinomialDistribution
from problab.value_sets import HomogeneousNumericValueSet
from problab.value_sets.sets import NATURALS_0, UNIT_INTERVAL, UNKNOWN_VALUE_SET


class BinomialDistributionTests(unittest.TestCase):
    def test_class_parameter_domains_match_constructor_order(self):
        distribution = BinomialDistribution(4, 0.25)
        self.assertEqual(list(distribution._valid_parameter_sets), ["n", "p"])
        self.assertIs(distribution._valid_parameter_sets["n"], NATURALS_0)
        self.assertIs(distribution._valid_parameter_sets["p"], UNIT_INTERVAL)
        self.assertNotIn("_valid_parameter_sets", vars(distribution))

    def test_realization_support_uses_trial_realizations_without_changing_math(self):
        distribution = BinomialDistribution(4, 0.25)
        distribution._parameter_nodes = (
            Mock(_realization_value_set=HomogeneousNumericValueSet(sp.FiniteSet(5), (np.int64,))),
            distribution._parameter_nodes[1],
        )
        support = distribution._realization_value_set
        self.assertIs(distribution._realization_value_set, support)
        self.assertEqual(distribution.value_set.sympy_set, sp.Range(0, 5))
        self.assertEqual(support.sympy_set, sp.Range(0, 6))
        self.assertEqual(support.dtype_types, (np.integer,))
        self.assertFalse(support.allows_nan)
        self.assertFalse(support.allows_positive_infinity)
        self.assertFalse(support.allows_negative_infinity)

    def test_risky_support_declares_object_fallback_only_for_possible_large_counts(self):
        for n_support, allows_object in (
            (HomogeneousNumericValueSet(sp.Interval(-1, 5), (np.float64,)), False),
            (HomogeneousNumericValueSet(sp.FiniteSet(2**53), (np.int64,)), False),
            (HomogeneousNumericValueSet(sp.FiniteSet(2**53 + 1), (np.int64,)), True),
            (UNKNOWN_VALUE_SET, True),
        ):
            distribution = BinomialDistribution(4, 0.25)
            distribution._parameter_nodes = (
                Mock(_realization_value_set=n_support),
                Mock(_realization_value_set=HomogeneousNumericValueSet(sp.Interval(0, 1), (np.float64,), allows_nan=True)),
            )
            with self.subTest(n_support=n_support):
                support = distribution._realization_value_set
                self.assertTrue(support.allows_nan)
                self.assertIn(np.float64, support.dtype_types)
                self.assertEqual(np.object_ in support.dtype_types, allows_object)

    def test_runtime_validation_delegates_each_parameter_to_its_validator(self):
        distribution = BinomialDistribution(4, 0.25)
        n, p = np.array([2, 4]), np.array([0.2, 0.4])
        path = "problab.distributions.discrete.binomial"
        with patch(path + "._validate_binomial_n_realizations", return_value=sentinel.n_mask) as n_check, patch(
            path + "._validate_binomial_p_realizations", return_value=sentinel.p_mask,
        ) as p_check:
            masks = distribution._validate_parameter_realizations(n, p)
        self.assertEqual(masks, {"n": sentinel.n_mask, "p": sentinel.p_mask})
        n_check.assert_called_once()
        p_check.assert_called_once()
        self.assertIs(n_check.call_args.args[0], n)
        self.assertIs(p_check.call_args.args[0], p)

    def test_configuration_and_symbolic_finite_support(self):
        distribution = BinomialDistribution(n=4, p=0.25)

        self.assertEqual(distribution.parameters, (4, 0.25))
        self.assertEqual(distribution.symbol, BinomialDistribution.symbol)
        self.assertEqual(distribution.name, "Binomial(4, 0.25)")
        self.assertEqual(distribution.value_set.sympy_set, sp.Range(0, 5))
        self.assertEqual(distribution.value_set.dtype_types, (np.integer,))

    def test_zero_n_has_single_value_support(self):
        distribution = BinomialDistribution(n=0, p=0.8)

        self.assertEqual(distribution.value_set.sympy_set, sp.Range(0, 1))

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
