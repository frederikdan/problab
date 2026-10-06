"""Intended sampling behavior when a permitted parameter risk actually occurs."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from problab import (
    BinomialDistribution,
    CategoricalDistribution,
    NormalDistribution,
    PoissonDistribution,
    RandomVariable,
)
from problab.value_sets import HomogeneousNumericValueSet
from problab.value_sets.sets import (
    NATURALS_0,
    NON_NEGATIVE_REALS,
    POSITIVE_REALS,
    REALS,
    UNIT_INTERVAL,
)


class ParameterRealizationPolicyRegressionTests(unittest.TestCase):
    def parameter(self, support, values):
        values = np.asarray(values)
        source = RandomVariable(CategoricalDistribution([1.0], [1.0]))
        return source.apply(
            lambda x: values.copy(),
            mathematical_value_set=support,
            realization_value_set=HomogeneousNumericValueSet(
                REALS.sympy_set, (values.dtype.type,),
                allows_positive_infinity=True,
                allows_negative_infinity=True,
                allows_nan=True,
            ),
            vectorized=True,
        )

    def cases(self):
        # The mathematical declaration is valid; the machine outputs can be invalid.
        return (
            ("Normal mean", REALS, [0.0, np.nan, 0.0], lambda x: NormalDistribution(x, 1, parameter_risk_policy="ignore")),
            ("Normal std", POSITIVE_REALS, [1.0, 0.0, 1.0], lambda x: NormalDistribution(0, x, parameter_risk_policy="ignore")),
            ("Binomial n", NATURALS_0, [2.0, -1.0, 2.0], lambda x: BinomialDistribution(x, 0.5, parameter_risk_policy="ignore")),
            ("Binomial p", UNIT_INTERVAL, [0.5, 2.0, 0.5], lambda x: BinomialDistribution(2, x, parameter_risk_policy="ignore")),
            ("Poisson mu", NON_NEGATIVE_REALS, [1.0, -1.0, 1.0], lambda x: PoissonDistribution(x, parameter_risk_policy="ignore")),
        )

    def variable(self, mathematical_set, values, constructor):
        source = RandomVariable(CategoricalDistribution([1.0], [1.0]))
        parameter = source.apply(lambda x: np.array(values), mathematical_value_set=mathematical_set,
                                 realization_value_set=HomogeneousNumericValueSet(REALS.sympy_set, (np.float64,), allows_nan=True),
                                 vectorized=True, function_name="parameter")
        return RandomVariable(constructor(parameter))

    def test_raise_policy_rejects_invalid_realized_parameters_with_informative_error(self):
        for name, support, values, constructor in self.cases():
            variable = self.variable(support, values, constructor)
            with self.subTest(parameter=name), self.assertRaisesRegex(ValueError, r"parameter|'(?:mean|std|n|p|mu)'"):
                variable.sample(num_samples=3, rng=np.random.default_rng(42), numerical_error_policy="raise")

    def test_warn_policy_warns_and_marks_only_invalid_positions_nan(self):
        for name, support, values, constructor in self.cases():
            variable = self.variable(support, values, constructor)
            with self.subTest(parameter=name), self.assertWarns(RuntimeWarning):
                samples = variable.sample(num_samples=3, rng=np.random.default_rng(42), numerical_error_policy="warn", validate=True)
                np.testing.assert_array_equal(np.isnan(samples), [False, True, False])
                self.assertTrue(np.isfinite(samples[[0, 2]]).all())

    def test_ignore_policy_marks_only_invalid_positions_nan_without_warning(self):
        for name, support, values, constructor in self.cases():
            variable = self.variable(support, values, constructor)
            with self.subTest(parameter=name), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                samples = variable.sample(num_samples=3, rng=np.random.default_rng(42), numerical_error_policy="ignore", validate=True)
                np.testing.assert_array_equal(np.isnan(samples), [False, True, False])
                self.assertEqual(caught, [])

    def test_all_invalid_batches_skip_the_backend_under_every_policy(self):
        cases = (
            (NormalDistribution, (0, self.parameter(POSITIVE_REALS, [0., -1., np.nan])), "problab.distributions.continuous.normal.norm.rvs"),
            (BinomialDistribution, (self.parameter(NATURALS_0, [-1., 1.5, np.inf]), 0.5), "problab.distributions.discrete.binomial.binom.rvs"),
            (PoissonDistribution, (self.parameter(NON_NEGATIVE_REALS, [-1., np.nan, np.inf]),), "problab.distributions.discrete.poisson.poisson.rvs"),
        )
        for constructor, parameters, backend_path in cases:
            variable = RandomVariable(constructor(*parameters, parameter_risk_policy="ignore"))
            for policy in ("raise", "warn", "ignore"):
                with self.subTest(distribution=constructor.__name__, policy=policy), patch(backend_path) as backend:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        if policy == "raise":
                            with self.assertRaisesRegex(ValueError, "parameter"):
                                variable.sample(3, numerical_error_policy=policy, validate=True)
                        else:
                            samples = variable.sample(3, numerical_error_policy=policy, validate=True)
                            self.assertTrue(np.isnan(samples).all())
                        self.assertEqual(len(caught), int(policy == "warn"))
                    backend.assert_not_called()

    def test_combined_masks_filter_every_parameter_and_preserve_sample_order(self):
        n = self.parameter(NATURALS_0, [2., -1., 4., 5.])
        p = self.parameter(UNIT_INTERVAL, [0.1, 0.2, 2., 0.4])
        variable = RandomVariable(BinomialDistribution(n, p, parameter_risk_policy="ignore"))
        rng = np.random.default_rng(42)
        path = "problab.distributions.discrete.binomial.binom.rvs"
        with patch(path, return_value=np.array([1, 3], dtype=np.int64)) as backend:
            samples = variable.sample(4, rng=rng, numerical_error_policy="ignore", validate=True)
        np.testing.assert_array_equal(samples, [1., np.nan, np.nan, 3.])
        np.testing.assert_array_equal(backend.call_args.kwargs["n"], [2., 5.])
        np.testing.assert_array_equal(backend.call_args.kwargs["p"], [0.1, 0.4])
        self.assertEqual(backend.call_args.kwargs["size"], 2)
        self.assertIs(backend.call_args.kwargs["random_state"], rng)

    def test_raise_stops_mixed_batches_before_backend_sampling(self):
        for _, support, values, constructor in self.cases():
            variable = self.variable(support, values, constructor)
            with self.subTest(constructor=constructor), patch.object(variable._distribution, "_sample") as backend:
                with self.assertRaisesRegex(ValueError, "parameter"):
                    variable.sample(3, numerical_error_policy="raise")
                backend.assert_not_called()

    def test_large_integer_samples_remain_exact_beside_nan(self):
        large = 2**53 + 1
        p = self.parameter(UNIT_INTERVAL, [1., -1., 1.])
        binomial = BinomialDistribution(large + 2, p, parameter_risk_policy="ignore")
        mu = self.parameter(NON_NEGATIVE_REALS, [1., -1., 1.])
        poisson = PoissonDistribution(mu, parameter_risk_policy="ignore")
        for distribution in (binomial, poisson):
            with self.subTest(distribution=distribution.symbol), patch.object(
                distribution, "_sample", return_value=np.array([large, large + 2], dtype=np.int64)
            ):
                samples = RandomVariable(distribution).sample(3, numerical_error_policy="ignore", validate=True)
            self.assertEqual(samples.dtype, np.dtype(object))
            self.assertEqual(samples[0], large)
            self.assertEqual(samples[2], large + 2)
            self.assertTrue(np.isnan(samples[1]))

    def test_valid_integer_batches_preserve_backend_dtype_and_generator_sequence(self):
        from scipy.stats import binom, poisson
        cases = (
            (BinomialDistribution(4, 0.25), lambda rng: binom.rvs(4, 0.25, size=20, random_state=rng)),
            (PoissonDistribution(2), lambda rng: poisson.rvs(2, size=20, random_state=rng)),
        )
        for distribution, expected in cases:
            with self.subTest(distribution=distribution.symbol):
                samples = distribution.sample(20, rng=np.random.default_rng(42), validate=True)
                self.assertEqual(samples.dtype.kind, "i")
                self.assertFalse(distribution._realization_value_set.allows_nan)
                np.testing.assert_array_equal(samples, expected(np.random.default_rng(42)))

    def test_backend_limits_are_checked_before_sampling(self):
        from problab.validation.distributions.discrete._poisson import _POISSON_MU_MAX
        cases = (
            (NATURALS_0, [2., float(2**63), 2.], lambda p: BinomialDistribution(p, 0.5, parameter_risk_policy="ignore")),
            (NON_NEGATIVE_REALS, [1., np.nextafter(_POISSON_MU_MAX, np.inf), 1.], lambda p: PoissonDistribution(p, parameter_risk_policy="ignore")),
        )
        for support, values, constructor in cases:
            variable = RandomVariable(constructor(self.parameter(support, values)))
            with self.subTest(support=support):
                samples = variable.sample(3, numerical_error_policy="ignore", validate=True)
                np.testing.assert_array_equal(np.isnan(samples), [False, True, False])

    def test_construction_policy_detects_finite_backend_limit_risks(self):
        constructors = (
            lambda policy: BinomialDistribution(2**63, 0.5, parameter_risk_policy=policy),
            lambda policy: PoissonDistribution(1e20, parameter_risk_policy=policy),
        )
        for constructor in constructors:
            with self.assertRaisesRegex(ValueError, "Cannot guarantee"):
                constructor("raise")
            with self.assertWarns(RuntimeWarning):
                constructor("warn")
            distribution = constructor("ignore")
            self.assertTrue(distribution._realization_value_set.allows_nan)
            samples = RandomVariable(distribution).sample(2, numerical_error_policy="ignore", validate=True)
            self.assertTrue(np.isnan(samples).all())
