import unittest
from unittest.mock import patch, sentinel

import numpy as np
import sympy as sp

from problab.functions import trigonometric
from problab.value_sets.sets import NON_NEGATIVE_REALS, REALS


class TrigonometricFunctionTests(unittest.TestCase):

    def test_sin_validates_real_input_and_declares_closed_unit_interval(self):
        self._assert_real_function(
            trigonometric.sin,
            np.sin,
            sp.Interval(-1, 1),
        )

    def test_cos_validates_real_input_and_declares_closed_unit_interval(self):
        self._assert_real_function(
            trigonometric.cos,
            np.cos,
            sp.Interval(-1, 1),
        )

    def test_tan_validates_real_input_and_declares_real_output(self):
        self._assert_real_function(trigonometric.tan, np.tan, REALS)

    def test_arcsin_validates_closed_unit_interval_and_declares_output_range(self):
        self._assert_domain_function(
            trigonometric.arcsin,
            np.arcsin,
            input_domain=sp.Interval(-1, 1),
            output_domain=sp.Interval(-sp.pi / 2, sp.pi / 2),
        )

    def test_arccos_validates_closed_unit_interval_and_declares_output_range(self):
        self._assert_domain_function(
            trigonometric.arccos,
            np.arccos,
            input_domain=sp.Interval(-1, 1),
            output_domain=sp.Interval(0, sp.pi),
        )

    def test_arctan_validates_real_input_and_declares_open_output_range(self):
        self._assert_real_function(
            trigonometric.arctan,
            np.arctan,
            sp.Interval.open(-sp.pi / 2, sp.pi / 2),
        )

    def test_sinh_validates_real_input_and_declares_real_output(self):
        self._assert_real_function(trigonometric.sinh, np.sinh, REALS)

    def test_cosh_validates_real_input_and_declares_output_at_least_one(self):
        self._assert_real_function(
            trigonometric.cosh,
            np.cosh,
            sp.Interval(1, sp.oo),
        )

    def test_tanh_validates_real_input_and_declares_open_unit_interval(self):
        self._assert_real_function(
            trigonometric.tanh,
            np.tanh,
            sp.Interval.open(-1, 1),
        )

    def test_arcsinh_validates_real_input_and_declares_real_output(self):
        self._assert_real_function(trigonometric.arcsinh, np.arcsinh, REALS)

    def test_arccosh_validates_lower_bound_and_declares_non_negative_output(self):
        self._assert_domain_function(
            trigonometric.arccosh,
            np.arccosh,
            input_domain=sp.Interval(1, sp.oo),
            output_value_set=NON_NEGATIVE_REALS,
        )

    def test_arctanh_validates_open_unit_interval_and_declares_real_output(self):
        self._assert_domain_function(
            trigonometric.arctanh,
            np.arctanh,
            input_domain=sp.Interval.open(-1, 1),
            output_value_set=REALS,
        )

    def _assert_real_function(self, function, numpy_function, output_value_set):
        with (
            patch("problab.functions.trigonometric._validate_real_valued") as validate_real_valued,
            patch("problab.functions.trigonometric._apply", return_value=sentinel.result) as apply,
        ):
            result = function(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        actual_value_set = apply.call_args.kwargs["value_set"]
        if isinstance(output_value_set, sp.Set):
            self.assertEqual(actual_value_set.sympy_set, output_value_set)
            self.assertEqual(actual_value_set.dtype_types, (np.floating,))
        else:
            self.assertIs(actual_value_set, output_value_set)
        self._assert_apply_call(apply, numpy_function, actual_value_set)

    def _assert_domain_function(
        self,
        function,
        numpy_function,
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
            output_value_set = apply.call_args.kwargs["value_set"]
            self.assertEqual(output_value_set.sympy_set, output_domain)
            self.assertEqual(output_value_set.dtype_types, (np.floating,))
        self._assert_apply_call(apply, numpy_function, output_value_set)

    def _assert_apply_call(self, apply, numpy_function, value_set):
        apply.assert_called_once_with(
            x=sentinel.x,
            function=numpy_function,
            value_set=value_set,
        )


if __name__ == "__main__":
    unittest.main()
