import unittest
from unittest.mock import patch, sentinel

import numpy as np
import sympy as sp

from problab.functions import basic
from problab.value_sets.sets import INTEGERS, NON_NEGATIVE_REALS


class BasicFunctionTests(unittest.TestCase):

    def test_sqrt_validates_non_negative_domain_and_applies_numpy_sqrt(self):
        with (
            patch("problab.functions.basic._validate_domain") as validate_domain,
            patch("problab.functions.basic._apply_scalar_or_rv", return_value=sentinel.result) as apply,
        ):
            result = basic.sqrt(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_domain.assert_called_once_with(
            x=sentinel.x,
            domain=NON_NEGATIVE_REALS.sympy_set,
        )
        apply.assert_called_once_with(
            x=sentinel.x,
            function=np.sqrt,
            value_set=NON_NEGATIVE_REALS,
        )

    def test_absolute_validates_real_input_and_declares_non_negative_output(self):
        with (
            patch("problab.functions.basic._validate_real_valued") as validate_real_valued,
            patch("problab.functions.basic._apply_scalar_or_rv", return_value=sentinel.result) as apply,
        ):
            result = basic.absolute(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        apply.assert_called_once_with(
            x=sentinel.x,
            function=np.abs,
            value_set=NON_NEGATIVE_REALS,
        )

    def test_floor_validates_real_input_and_declares_integer_output(self):
        with (
            patch("problab.functions.basic._validate_real_valued") as validate_real_valued,
            patch("problab.functions.basic._apply_scalar_or_rv", return_value=sentinel.result) as apply,
        ):
            result = basic.floor(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        apply.assert_called_once_with(
            x=sentinel.x,
            function=np.floor,
            value_set=INTEGERS,
        )

    def test_ceil_validates_real_input_and_declares_integer_output(self):
        with (
            patch("problab.functions.basic._validate_real_valued") as validate_real_valued,
            patch("problab.functions.basic._apply_scalar_or_rv", return_value=sentinel.result) as apply,
        ):
            result = basic.ceil(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        apply.assert_called_once_with(
            x=sentinel.x,
            function=np.ceil,
            value_set=INTEGERS,
        )

    def test_sign_validates_real_input_and_declares_three_possible_outputs(self):
        with (
            patch("problab.functions.basic._validate_real_valued") as validate_real_valued,
            patch("problab.functions.basic._apply_scalar_or_rv", return_value=sentinel.result) as apply,
        ):
            result = basic.sign(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        self.assertEqual(apply.call_args.kwargs["x"], sentinel.x)
        self.assertIs(apply.call_args.kwargs["function"], np.sign)
        value_set = apply.call_args.kwargs["value_set"]
        self.assertEqual(value_set.sympy_set, sp.FiniteSet(-1, 0, 1))
        self.assertEqual(value_set.dtype_types, (np.integer, np.floating))


if __name__ == "__main__":
    unittest.main()
