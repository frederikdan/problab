import unittest
from unittest.mock import call, patch, sentinel

import numpy as np
import sympy as sp

from problab.operations._arithmetic import (_ABS)
from problab.operations._function import (_CEIL, _FLOOR, _HYPOT, _SIGN, _SQRT)
from problab.functions import basic
from problab.value_sets.sets import INTEGERS, NON_NEGATIVE_REALS


class BasicFunctionTests(unittest.TestCase):

    def test_sqrt_validates_non_negative_domain_and_applies_sqrt_operation(self):
        with (
            patch("problab.functions.basic._validate_domain") as validate_domain,
            patch("problab.functions.basic._apply", return_value=sentinel.result) as apply,
        ):
            result = basic.sqrt(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_domain.assert_called_once_with(
            x=sentinel.x,
            domain=NON_NEGATIVE_REALS.sympy_set,
        )
        apply.assert_called_once_with(
            x=sentinel.x,
            operation=_SQRT,
            mathematical_value_set=NON_NEGATIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

    def test_absolute_validates_real_input_and_declares_non_negative_output(self):
        with (
            patch("problab.functions.basic._validate_real_valued") as validate_real_valued,
            patch("problab.functions.basic._apply", return_value=sentinel.result) as apply,
        ):
            result = basic.absolute(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        apply.assert_called_once_with(
            x=sentinel.x,
            operation=_ABS,
            mathematical_value_set=NON_NEGATIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

    def test_floor_validates_real_input_and_declares_integer_output(self):
        with (
            patch("problab.functions.basic._validate_real_valued") as validate_real_valued,
            patch("problab.functions.basic._apply", return_value=sentinel.result) as apply,
        ):
            result = basic.floor(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        apply.assert_called_once_with(
            x=sentinel.x,
            operation=_FLOOR,
            mathematical_value_set=INTEGERS,
            realization_value_set=INTEGERS,
        )

    def test_ceil_validates_real_input_and_declares_integer_output(self):
        with (
            patch("problab.functions.basic._validate_real_valued") as validate_real_valued,
            patch("problab.functions.basic._apply", return_value=sentinel.result) as apply,
        ):
            result = basic.ceil(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        apply.assert_called_once_with(
            x=sentinel.x,
            operation=_CEIL,
            mathematical_value_set=INTEGERS,
            realization_value_set=INTEGERS,
        )

    def test_sign_validates_real_input_and_declares_three_possible_outputs(self):
        with (
            patch("problab.functions.basic._validate_real_valued") as validate_real_valued,
            patch("problab.functions.basic._apply", return_value=sentinel.result) as apply,
        ):
            result = basic.sign(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        self.assertEqual(apply.call_args.kwargs["x"], sentinel.x)
        self.assertIs(apply.call_args.kwargs["operation"], _SIGN)
        mathematical_value_set = apply.call_args.kwargs["mathematical_value_set"]
        realization_value_set = apply.call_args.kwargs["realization_value_set"]
        self.assertEqual(mathematical_value_set.sympy_set, sp.FiniteSet(-1, 0, 1))
        self.assertEqual(mathematical_value_set.dtype_types, (np.integer, np.floating))
        self.assertEqual(realization_value_set, mathematical_value_set)

    def test_hypot_validates_both_inputs_and_declares_non_negative_output(self):
        with (
            patch("problab.functions.basic._validate_real_valued") as validate_real_valued,
            patch("problab.functions.basic._apply", return_value=sentinel.result) as apply,
        ):
            result = basic.hypot(sentinel.x, sentinel.y)

        self.assertIs(result, sentinel.result)
        self.assertEqual(validate_real_valued.call_args_list, [call(sentinel.x), call(sentinel.y)])
        apply.assert_called_once_with(
            sentinel.x,
            _HYPOT,
            NON_NEGATIVE_REALS,
            NON_NEGATIVE_REALS,
            sentinel.y,
        )


if __name__ == "__main__":
    unittest.main()
