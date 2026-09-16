import unittest
from unittest.mock import Mock, sentinel

import numpy as np

from problab.functions._utils import _apply, _apply_scalar_or_rv
from problab.random_variables.base import RandomVariable
from problab.value_sets.sets import REALS


class FunctionUtilityTests(unittest.TestCase):

    def test_apply_scalar_or_rv_returns_scalar_function_result(self):
        result = _apply_scalar_or_rv(
            x=3,
            function=lambda value: value + 1,
            value_set=REALS,
        )

        self.assertEqual(result, 4)

    def test_apply_scalar_or_rv_delegates_random_variable_to_apply(self):
        random_variable = RandomVariable.__new__(RandomVariable)
        random_variable.apply = Mock(return_value=sentinel.result)

        result = _apply_scalar_or_rv(
            x=random_variable,
            function=sentinel.function,
            value_set=REALS,
        )

        self.assertIs(result, sentinel.result)
        random_variable.apply.assert_called_once_with(
            function=sentinel.function,
            value_set=REALS,
            vectorized=True,
        )

    def test_apply_converts_scalar_result_to_float(self):
        result = _apply(
            x=3,
            function=lambda value: np.int64(value + 1),
            value_set=REALS,
        )

        self.assertEqual(result, 4.0)
        self.assertIs(type(result), float)

    def test_apply_delegates_random_variable_to_apply(self):
        random_variable = RandomVariable.__new__(RandomVariable)
        random_variable.apply = Mock(return_value=sentinel.result)

        result = _apply(
            x=random_variable,
            function=sentinel.function,
            value_set=REALS,
        )

        self.assertIs(result, sentinel.result)
        random_variable.apply.assert_called_once_with(
            function=sentinel.function,
            value_set=REALS,
            vectorized=True,
        )


if __name__ == "__main__":
    unittest.main()
