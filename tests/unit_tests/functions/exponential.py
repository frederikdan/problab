import unittest
from unittest.mock import patch, sentinel

import numpy as np

from problab.functions import exponential
from problab.value_sets.sets import POSITIVE_REALS, REALS


class ExponentialFunctionTests(unittest.TestCase):

    def test_exp_validates_real_input_and_declares_positive_output(self):
        with (
            patch("problab.functions.exponential._validate_real_valued") as validate_real_valued,
            patch("problab.functions.exponential._apply_scalar_or_rv", return_value=sentinel.result) as apply,
        ):
            result = exponential.exp(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_real_valued.assert_called_once_with(sentinel.x)
        apply.assert_called_once_with(
            x=sentinel.x,
            function=np.exp,
            value_set=POSITIVE_REALS,
        )

    def test_log_validates_positive_domain_and_applies_numpy_log(self):
        self._assert_log_function(exponential.log, np.log)

    def test_log2_validates_positive_domain_and_applies_numpy_log2(self):
        self._assert_log_function(exponential.log2, np.log2)

    def test_log10_validates_positive_domain_and_applies_numpy_log10(self):
        self._assert_log_function(exponential.log10, np.log10)

    def _assert_log_function(self, function, numpy_function):
        with (
            patch("problab.functions.exponential._validate_domain") as validate_domain,
            patch("problab.functions.exponential._apply_scalar_or_rv", return_value=sentinel.result) as apply,
        ):
            result = function(sentinel.x)

        self.assertIs(result, sentinel.result)
        validate_domain.assert_called_once_with(
            x=sentinel.x,
            domain=POSITIVE_REALS.sympy_set,
        )
        apply.assert_called_once_with(
            x=sentinel.x,
            function=numpy_function,
            value_set=REALS,
        )


if __name__ == "__main__":
    unittest.main()
