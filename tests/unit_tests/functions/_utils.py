import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np

from problab.operations._function import (_FunctionOperation)
from problab.functions._utils import _apply
from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes import _ConstantNode
from problab.value_sets.sets import REALS


class FunctionUtilityTests(unittest.TestCase):

    def test_apply_checks_named_nodes_before_executing_scalar_callback(self):
        function = Mock(return_value=5.0)
        operation = _FunctionOperation(operation=function, name_func=lambda x, y: f"f({x}, {y})",
                                       supported_input_types=((np.floating,),))
        with patch("problab.functions._utils._require_supported_operation_inputs", side_effect=TypeError("dtype")) as require:
            with self.assertRaisesRegex(TypeError, "dtype"):
                _apply(2, operation, REALS, REALS, 3)
        nodes = require.call_args.args[0]
        self.assertEqual(tuple(node.value for node in nodes), (2, 3))
        self.assertEqual(require.call_args.kwargs, dict(operation_name="f(2, 3)", supported_input_types=operation.supported_input_types))
        function.assert_not_called()

    def test_apply_passes_all_nodes_to_dtype_check_for_multiple_random_variables(self):
        first, second = (RandomVariable._from_node(_ConstantNode(value)) for value in (3, 4))
        operation = _FunctionOperation(operation=np.hypot, name_func=lambda x, y: f"hypot({x}, {y})")
        with patch("problab.functions._utils._require_supported_operation_inputs") as require:
            _apply(first, operation, REALS, REALS, second)
        self.assertEqual(require.call_args.args[0], (first._node, second._node))

    def test_apply_rejects_object_storage_before_ufunc_execution(self):
        operation = _FunctionOperation(operation=Mock(), name_func=str, supported_input_types=((np.number,),))
        variable = RandomVariable._from_node(_ConstantNode([1, 2]))
        with self.assertRaises(TypeError):
            _apply(variable, operation, REALS, REALS)
        operation.operation.assert_not_called()

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
        random_variable = RandomVariable._from_node(_ConstantNode(3))
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
