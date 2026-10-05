"""Intended sampling behavior when a permitted parameter risk actually occurs."""
import unittest
import warnings

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
