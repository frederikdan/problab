import unittest
from unittest.mock import call, patch, sentinel

from problab._operations import _EXP, _EXPM1, _LOG, _LOG1P, _LOG2, _LOG10, _LOGADDEXP
from problab.functions import exponential
from problab.value_sets.sets import NON_NEGATIVE_REALS, POSITIVE_REALS, REALS


class ExponentialFunctionTests(unittest.TestCase):

    def test_exp_declares_positive_math_support_and_non_negative_realization_support(self):
        with (
            patch("problab.functions.exponential._validate_real_valued") as validate_real_valued,
            patch("problab.functions.exponential._apply", return_value=sentinel.result) as apply,
        ):
            result = exponential.exp(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        apply.assert_called_once_with(
            x=sentinel.x,
            operation=_EXP,
            mathematical_value_set=POSITIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

    def test_log_validates_positive_domain_and_applies_log_operation(self):
        self._assert_log_function(exponential.log, _LOG)

    def test_log2_validates_positive_domain_and_applies_log2_operation(self):
        self._assert_log_function(exponential.log2, _LOG2)

    def test_log10_validates_positive_domain_and_applies_log10_operation(self):
        self._assert_log_function(exponential.log10, _LOG10)

    def test_log1p_validates_domain_and_applies_log1p_operation(self):
        with (
            patch("problab.functions.exponential._validate_domain") as validate_domain,
            patch("problab.functions.exponential._apply", return_value=sentinel.result) as apply,
        ):
            result = exponential.log1p(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_domain.assert_called_once_with(
            x=sentinel.x,
            domain=exponential.GT_NEG_ONE_REALS.sympy_set,
        )
        apply.assert_called_once_with(
            x=sentinel.x,
            operation=_LOG1P,
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

    def test_expm1_validates_real_input_and_declares_its_range(self):
        with (
            patch("problab.functions.exponential._validate_real_valued") as validate_real_valued,
            patch("problab.functions.exponential._apply", return_value=sentinel.result) as apply,
        ):
            result = exponential.expm1(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        apply.assert_called_once_with(
            x=sentinel.x,
            operation=_EXPM1,
            mathematical_value_set=exponential.GT_NEG_ONE_REALS,
            realization_value_set=exponential.GE_NEG_ONE_REALS,
        )

    def test_logaddexp_validates_both_inputs_and_applies_binary_operation(self):
        with (
            patch("problab.functions.exponential._validate_real_valued") as validate_real_valued,
            patch("problab.functions.exponential._apply", return_value=sentinel.result) as apply,
        ):
            result = exponential.logaddexp(sentinel.x, sentinel.y)

        self.assertIs(result, sentinel.result)
        self.assertEqual(validate_real_valued.call_args_list, [call(sentinel.x), call(sentinel.y)])
        apply.assert_called_once_with(sentinel.x, _LOGADDEXP, REALS, REALS, sentinel.y)

    def _assert_log_function(self, function, operation):
        with (
            patch("problab.functions.exponential._validate_domain") as validate_domain,
            patch("problab.functions.exponential._apply", return_value=sentinel.result) as apply,
        ):
            result = function(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_domain.assert_called_once_with(
            x=sentinel.x,
            domain=POSITIVE_REALS.sympy_set,
        )
        apply.assert_called_once_with(
            x=sentinel.x,
            operation=operation,
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )


if __name__ == "__main__":
    unittest.main()
