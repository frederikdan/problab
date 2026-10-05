import unittest
import warnings

import numpy as np
import sympy as sp

from problab import CategoricalDistribution, RandomVariable
from problab.functions import basic, exponential, trigonometric
from problab.value_sets import HomogeneousNumericValueSet
from problab.value_sets.sets import REALS


class NumericalFunctionIntegrationTests(unittest.TestCase):
    def variable(self, value):
        return RandomVariable(CategoricalDistribution([value], [1.0]))

    def test_all_predefined_functions_match_native_values_and_validate_concrete_dtypes(self):
        for module, name, value in (
            (basic, "absolute", -0.5), (basic, "sqrt", 0.5), (basic, "floor", 0.5), (basic, "ceil", 0.5), (basic, "sign", -0.5),
            (exponential, "exp", 0.5), (exponential, "expm1", 0.5), (exponential, "log", 0.5),
            (exponential, "log2", 0.5), (exponential, "log10", 0.5), (exponential, "log1p", 0.5),
            (trigonometric, "sin", 0.5), (trigonometric, "cos", 0.5), (trigonometric, "tan", 0.5),
            (trigonometric, "arcsin", 0.5), (trigonometric, "arccos", 0.5), (trigonometric, "arctan", 0.5),
            (trigonometric, "sinh", 0.5), (trigonometric, "cosh", 0.5), (trigonometric, "tanh", 0.5),
            (trigonometric, "arcsinh", 0.5), (trigonometric, "arccosh", 2.0), (trigonometric, "arctanh", 0.5),
        ):
            for dtype in (np.float16, np.float32, np.float64):
                with self.subTest(function=name, dtype=dtype):
                    original = dtype(value)
                    expected = getattr(np, name)(np.array([original]))
                    result = getattr(module, name)(self.variable(original))
                    samples = result.sample(num_samples=3, validate=True, numerical_error_policy="raise")
                    self.assertEqual(samples.dtype, expected.dtype)
                    np.testing.assert_array_equal(samples, np.repeat(expected, 3))

    def test_binary_functions_support_scalar_and_variable_inputs_with_mixed_precision(self):
        for function, numpy_function in ((basic.hypot, np.hypot), (exponential.logaddexp, np.logaddexp)):
            for left, right in ((3.0, self.variable(np.float32(4))),
                                (self.variable(np.float32(3)), self.variable(np.float64(4)))):
                with self.subTest(function=function.__name__, left=type(left)):
                    samples = function(left, right).sample(num_samples=3, validate=True)
                    expected = numpy_function(np.array([3.0]), np.array([4.0]))
                    np.testing.assert_allclose(samples, np.repeat(expected, 3), rtol=1e-6)

    def test_inverse_trigonometric_endpoints_keep_exact_math_and_rounded_realization_support(self):
        for function, value, math_interval in ((trigonometric.arcsin, 1, sp.Interval(-sp.pi / 2, sp.pi / 2)),
                                               (trigonometric.arccos, -1, sp.Interval(0, sp.pi)),
                                               (trigonometric.arctan, 1e10, sp.Interval.open(-sp.pi / 2, sp.pi / 2))):
            with self.subTest(function=function.__name__):
                expression = function(self.variable(np.float32(value)))
                self.assertEqual(expression._node.value_set.sympy_set, math_interval)
                self.assertTrue(expression._node._realization_value_set.sympy_set.is_superset(math_interval))
                samples = expression.sample(num_samples=3, validate=True)
                self.assertTrue(np.isfinite(samples).all())

    def test_exponential_overflow_policy_does_not_change_mathematical_support(self):
        expression = exponential.exp(self.variable(1000.0))
        self.assertFalse(expression._node.value_set.contains(0.0))
        self.assertFalse(expression._node.value_set.contains(np.inf))
        self.assertTrue(expression._node._realization_value_set.allows_positive_infinity)
        for policy in ("warn", "ignore"):
            with self.subTest(policy=policy), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                samples = expression.sample(num_samples=3, validate=True, numerical_error_policy=policy)
                np.testing.assert_array_equal(samples, [np.inf] * 3)
                self.assertEqual(bool(caught), policy == "warn")
        with self.assertRaises(FloatingPointError):
            expression.sample(numerical_error_policy="raise", validate=True)

    def test_underflow_is_allowed_but_logarithms_apply_policy_to_realized_zero(self):
        underflowed = exponential.exp(self.variable(-1000.0))
        np.testing.assert_array_equal(underflowed.sample(num_samples=3, validate=True, numerical_error_policy="raise"), [0.0] * 3)
        # log2/log10 do not have the inverse simplification of log(exp(X)).
        for function in (exponential.log2, exponential.log10):
            expression = function(underflowed)
            for policy in ("warn", "ignore"):
                with self.subTest(function=function.__name__, policy=policy), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    np.testing.assert_array_equal(expression.sample(num_samples=3, validate=True, numerical_error_policy=policy), [-np.inf] * 3)
                    self.assertEqual(bool(caught), policy == "warn")
            with self.assertRaises(FloatingPointError):
                expression.sample(numerical_error_policy="raise")

    def test_arctanh_handles_rounded_tanh_endpoint_under_each_policy(self):
        expression = trigonometric.arctanh(trigonometric.tanh(self.variable(1000.0)))
        self.assertTrue(expression._node._realization_value_set.allows_positive_infinity)
        for policy in ("warn", "ignore"):
            with self.subTest(policy=policy), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                np.testing.assert_array_equal(expression.sample(num_samples=3, validate=True, numerical_error_policy=policy), [np.inf] * 3)
                self.assertEqual(bool(caught), policy == "warn")
        with self.assertRaises(FloatingPointError):
            expression.sample(numerical_error_policy="raise")

    def test_realization_domain_violations_declare_nan_without_widening_math_support(self):
        source = self.variable(1.0)
        argument = source.apply(lambda x: np.full_like(x, 1.1), vectorized=True,
            mathematical_value_set=HomogeneousNumericValueSet(sp.Interval(-1, 1), (np.float64,)),
            realization_value_set=HomogeneousNumericValueSet(sp.Interval(-2, 2), (np.float64,)))
        for function in (trigonometric.arcsin, trigonometric.arccos):
            expression = function(argument)
            with self.subTest(function=function.__name__):
                self.assertTrue(expression._node._realization_value_set.allows_nan)
                self.assertFalse(expression._node.value_set.allows_nan)
                samples = expression.sample(num_samples=3, validate=True, numerical_error_policy="ignore")
                self.assertTrue(np.isnan(samples).all())
                with self.assertRaises(FloatingPointError):
                    expression.sample(num_samples=3, numerical_error_policy="raise")
