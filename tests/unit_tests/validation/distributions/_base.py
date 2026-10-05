import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import sympy as sp

from problab.value_sets import HomogeneousNumericValueSet, ObjectValueSet
from problab.value_sets._unknown import _UnknownValueSet
from problab.value_sets.sets import POSITIVE_REALS

from problab.validation.distributions._base import (
    _validate_cdf_input,
    _validate_ppf_input,
    _validate_quantile_method,
    _validate_real_input,
    _require_supported_parameter_realization_dtypes,
    _validate_parameter_realization_value_set,
    _validate_parameter_risk_policy,
)


class DistributionBaseValidationTests(unittest.TestCase):

    def test_parameter_risk_policy_accepts_three_modes_and_rejects_invalid_values(self):
        for value in ("warn", "raise", "ignore"):
            _validate_parameter_risk_policy(value)
        for value in (None, True, 1):
            with self.subTest(value=value), self.assertRaises(TypeError):
                _validate_parameter_risk_policy(value)
        for value in ("WARN", "", "unknown"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate_parameter_risk_policy(value)

    def test_parameter_dtype_rules_broadcast_and_report_distribution_and_input_name(self):
        first = SimpleNamespace(name="mean", _realization_value_set=SimpleNamespace(dtype_types=(np.float64,)))
        for types in (None, (np.object_,), (np.float32, np.complex64)):
            second = SimpleNamespace(name="scale", _realization_value_set=SimpleNamespace(dtype_types=types))
            with self.subTest(types=types), self.assertRaisesRegex(TypeError, "Normal.*'scale'"):
                _require_supported_parameter_realization_dtypes((first, second), distribution_name="Normal",
                                                               supported_input_types=((np.integer, np.floating),))

    def test_parameter_dtype_validation_allows_per_parameter_rules_and_rejects_mismatched_arity(self):
        nodes = tuple(SimpleNamespace(name=str(i), _realization_value_set=SimpleNamespace(dtype_types=(t,)))
                      for i, t in enumerate((np.int64, np.float32)))
        _require_supported_parameter_realization_dtypes(nodes, distribution_name="Custom",
                                                       supported_input_types=((np.integer,), (np.floating,)))
        with self.assertRaises(ValueError):
            _require_supported_parameter_realization_dtypes(nodes, distribution_name="Custom", supported_input_types=())
        _require_supported_parameter_realization_dtypes((object(),), distribution_name="Custom", supported_input_types=None)

    def check_risk(self, value_set, policy, valid_value_set=POSITIVE_REALS):
        return _validate_parameter_realization_value_set(
            SimpleNamespace(_realization_value_set=value_set), parameter_name="std",
            distribution_name="CustomNormal", valid_value_set=valid_value_set, parameter_risk_policy=policy,
        )

    def test_safe_parameter_support_produces_no_warning_under_any_policy(self):
        safe = HomogeneousNumericValueSet(sp.Interval(1, 2), (np.float64,))
        for policy in ("warn", "raise", "ignore"):
            with self.subTest(policy=policy), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.check_risk(safe, policy)
            self.assertEqual(caught, [])

    def test_risky_finite_unknown_and_object_support_follow_risk_policy(self):
        for support in (HomogeneousNumericValueSet(sp.Interval(0, 1), (np.float64,)),
                        HomogeneousNumericValueSet(_UnknownValueSet(), (np.float64,)), ObjectValueSet((1,))):
            with self.subTest(support=support):
                with self.assertRaisesRegex(ValueError, "'std'.*CustomNormal"):
                    self.check_risk(support, "raise")
                with self.assertWarnsRegex(RuntimeWarning, "'std'.*CustomNormal"):
                    self.check_risk(support, "warn")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.check_risk(support, "ignore")
                self.assertEqual(caught, [])

    def test_each_exceptional_value_permission_is_checked_independently(self):
        for flag in ("allows_positive_infinity", "allows_negative_infinity", "allows_nan"):
            support = HomogeneousNumericValueSet(sp.Interval(1, 2), (np.float64,), **{flag: True})
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.check_risk(support, "raise")
            target = HomogeneousNumericValueSet(sp.Interval(0, 3), (np.float64,), **{flag: True})
            self.check_risk(support, "raise", target)

    def test_real_input_accepts_real_scalars_and_numeric_non_complex_arrays(self):
        _validate_real_input(1.0, "x")
        _validate_cdf_input(np.array([1, 2], dtype=np.int64))

    def test_real_input_rejects_bool_text_and_complex_values(self):
        with self.assertRaises(TypeError):
            _validate_real_input(True, "x")
        with self.assertRaises(TypeError):
            _validate_real_input(np.array(["x"]), "x")
        with self.assertRaises(TypeError):
            _validate_real_input(np.array([1 + 1j]), "x")

    def test_ppf_input_requires_values_in_closed_unit_interval(self):
        _validate_ppf_input(np.array([0.0, 0.5, 1.0]))

        with self.assertRaises(ValueError):
            _validate_ppf_input(np.array([np.nan]))
        with self.assertRaises(ValueError):
            _validate_ppf_input(-0.1)

    def test_quantile_method_requires_supported_name(self):
        _validate_quantile_method("linear")

        with self.assertRaises(TypeError):
            _validate_quantile_method(1)
        with self.assertRaises(ValueError):
            _validate_quantile_method("unsupported")


if __name__ == "__main__":
    unittest.main()
