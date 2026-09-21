import unittest
from unittest.mock import patch, sentinel

import numpy as np
import sympy as sp

from problab._operations import (
    _SIN,
    _ARCSIN,
    _COS,
    _ARCCOS,
    _TAN,
    _ARCTAN,
    _SINH,
    _COSH,
    _TANH,
    _ARCSINH,
    _ARCCOSH,
    _ARCTANH,
)
from problab.functions import trigonometric
from problab.value_sets.sets import NON_NEGATIVE_REALS, REALS


class TrigonometricFunctionTests(unittest.TestCase):

    def test_sin_validates_real_input_and_declares_closed_unit_interval(self):
        self._assert_real_function(
            trigonometric.sin,
            _SIN,
            sp.Interval(-1, 1),
        )

    def test_cos_validates_real_input_and_declares_closed_unit_interval(self):
        self._assert_real_function(
            trigonometric.cos,
            _COS,
            sp.Interval(-1, 1),
        )

    def test_tan_validates_real_input_and_declares_real_output(self):
        self._assert_real_function(trigonometric.tan, _TAN, REALS)

    def test_arcsin_validates_closed_unit_interval_and_declares_output_range(self):
        self._assert_domain_function(
            trigonometric.arcsin,
            _ARCSIN,
            input_domain=sp.Interval(-1, 1),
            output_domain=sp.Interval(-sp.pi / 2, sp.pi / 2),
        )

    def test_arccos_validates_closed_unit_interval_and_declares_output_range(self):
        self._assert_domain_function(
            trigonometric.arccos,
            _ARCCOS,
            input_domain=sp.Interval(-1, 1),
            output_domain=sp.Interval(0, sp.pi),
        )

    def test_arctan_declares_open_math_range_and_closed_realization_range(self):
        self._assert_real_function(
            trigonometric.arctan,
            _ARCTAN,
            sp.Interval(-sp.pi / 2, sp.pi / 2),
            mathematical_domain=sp.Interval.open(-sp.pi / 2, sp.pi / 2),
        )

    def test_sinh_validates_real_input_and_declares_real_output(self):
        self._assert_real_function(trigonometric.sinh, _SINH, REALS)

    def test_cosh_validates_real_input_and_declares_output_at_least_one(self):
        self._assert_real_function(
            trigonometric.cosh,
            _COSH,
            sp.Interval(1, sp.oo),
        )

    def test_tanh_declares_open_math_range_and_closed_realization_range(self):
        self._assert_real_function(
            trigonometric.tanh,
            _TANH,
            sp.Interval(-1, 1),
            mathematical_domain=sp.Interval.open(-1, 1),
        )

    def test_arcsinh_validates_real_input_and_declares_real_output(self):
        self._assert_real_function(trigonometric.arcsinh, _ARCSINH, REALS)

    def test_arccosh_validates_lower_bound_and_declares_non_negative_output(self):
        self._assert_domain_function(
            trigonometric.arccosh,
            _ARCCOSH,
            input_domain=sp.Interval(1, sp.oo),
            output_value_set=NON_NEGATIVE_REALS,
        )

    def test_arctanh_validates_open_unit_interval_and_declares_real_output(self):
        self._assert_domain_function(
            trigonometric.arctanh,
            _ARCTANH,
            input_domain=sp.Interval.open(-1, 1),
            output_value_set=REALS,
        )

    def _assert_real_function(
        self,
        function,
        operation,
        output_value_set,
        mathematical_domain=None,
    ):
        with (
            patch("problab.functions.trigonometric._validate_real_valued") as validate_real_valued,
            patch("problab.functions.trigonometric._apply", return_value=sentinel.result) as apply,
        ):
            result = function(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        actual_value_set = apply.call_args.kwargs["mathematical_value_set"]
        realization_value_set = apply.call_args.kwargs["realization_value_set"]
        if mathematical_domain is not None:
            self.assertEqual(actual_value_set.sympy_set, mathematical_domain)
            self.assertEqual(realization_value_set.sympy_set, output_value_set)
        elif isinstance(output_value_set, sp.Set):
            self.assertEqual(actual_value_set.sympy_set, output_value_set)
            self.assertEqual(actual_value_set.dtype_types, (np.floating,))
        else:
            self.assertIs(actual_value_set, output_value_set)
        self._assert_apply_call(
            apply,
            operation,
            actual_value_set,
            realization_value_set,
        )

    def _assert_domain_function(
        self,
        function,
        operation,
        input_domain,
        output_domain=None,
        output_value_set=None,
    ):
        with (
            patch("problab.functions.trigonometric._validate_domain") as validate_domain,
            patch("problab.functions.trigonometric._apply", return_value=sentinel.result) as apply,
        ):
            result = function(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_domain.assert_called_once_with(x=sentinel.x, domain=input_domain)
        if output_value_set is None:
            output_value_set = apply.call_args.kwargs["mathematical_value_set"]
            self.assertEqual(output_value_set.sympy_set, output_domain)
            self.assertEqual(output_value_set.dtype_types, (np.floating,))
        self._assert_apply_call(
            apply,
            operation,
            output_value_set,
            apply.call_args.kwargs["realization_value_set"],
        )

    def _assert_apply_call(self, apply, operation, mathematical_value_set, realization_value_set):
        expected = dict(
            x=sentinel.x,
            operation=operation,
            mathematical_value_set=mathematical_value_set,
            realization_value_set=realization_value_set,
        )
        apply.assert_called_once_with(**expected)


if __name__ == "__main__":
    unittest.main()
