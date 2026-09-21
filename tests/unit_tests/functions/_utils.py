import unittest
from unittest.mock import Mock, sentinel

import numpy as np

from problab._operations import _FunctionOperation
from problab.functions._utils import _apply
from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes import _ConstantNode
from problab.value_sets.sets import REALS


class FunctionUtilityTests(unittest.TestCase):

    def test_apply_converts_scalar_operation_result_to_float(self):
        operation = _FunctionOperation(
            operation=lambda value: np.int64(value + 1),
            name_func=lambda name: f"increment({name})",
        )

        result = _apply(
            x=3,
            operation=operation,
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertEqual(result, 4.0)
        self.assertIs(type(result), float)

    def test_apply_delegates_random_variable_to_private_operation_method(self):
        random_variable = RandomVariable.__new__(RandomVariable)
        random_variable._apply_operation = Mock(return_value=sentinel.result)
        operation = _FunctionOperation(
            operation=sentinel.function,
            name_func=lambda name: f"f({name})",
        )

        result = _apply(
            x=random_variable,
            operation=operation,
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(result, sentinel.result)
        random_variable._apply_operation.assert_called_once_with(
            operation=operation,
            mathematical_value_set=REALS,
            realization_value_set=REALS,
            vectorized=True,
        )

    def test_apply_builds_node_for_mixed_scalar_and_random_variable_inputs(self):
        random_variable = RandomVariable._from_node(_ConstantNode(3))
        operation = _FunctionOperation(
            operation=np.hypot,
            name_func=lambda x, y: f"hypot({x}, {y})",
        )

        result = _apply(
            4,
            operation,
            REALS,
            REALS,
            random_variable,
        )

        self.assertIsInstance(result, RandomVariable)
        self.assertIs(result._node._inputs[1], random_variable._node)
        self.assertEqual(result._node._inputs[0].value, 4)


if __name__ == "__main__":
    unittest.main()
