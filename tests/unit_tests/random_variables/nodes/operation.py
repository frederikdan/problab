import unittest
from unittest.mock import Mock

import numpy as np

from problab._operations import _ADD, _EXP, _FunctionOperation
from problab.random_variables.nodes.operation import _OperationNode
from problab.value_sets.sets import NON_NEGATIVE_REALS, REALS


class OperationNodeTests(unittest.TestCase):

    def test_properties_expose_operation_inputs_and_value_set(self):
        first = object()
        second = object()
        node = _OperationNode(
            operation=_ADD,
            inputs=(first, second),
            name="(first + second)",
            mathematical_value_set=REALS,
        )

        self.assertEqual(node.name, "(first + second)")
        self.assertEqual(repr(node), "OperationNode((first + second))")
        self.assertEqual(node.dependencies, {first, second})
        self.assertIs(node.value_set, REALS)
        self.assertIs(node._realization_value_set, REALS)
        self.assertIs(node._operation, _ADD)

    def test_properties_expose_distinct_mathematical_and_realization_sets(self):
        node = _OperationNode(
            operation=_EXP,
            inputs=(),
            name="exp(x)",
            mathematical_value_set=REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

        self.assertIs(node.value_set, REALS)
        self.assertIs(node._realization_value_set, NON_NEGATIVE_REALS)

    def test_evaluate_realizes_inputs_then_calls_operation_descriptor(self):
        first = object()
        second = object()
        context = Mock()
        context.evaluate.side_effect = (
            np.array([1, 2]),
            np.array([3, 4]),
        )
        operation_function = Mock(return_value=np.array([4, 6]))
        operation = _FunctionOperation(
            operation=operation_function,
            name_func=lambda left, right: f"sum({left}, {right})",
        )
        node = _OperationNode(operation, (first, second), "sum", REALS)

        result = node._evaluate(context)

        np.testing.assert_array_equal(result, [4, 6])
        self.assertEqual(context.evaluate.call_args_list[0].args, (first,))
        self.assertEqual(context.evaluate.call_args_list[1].args, (second,))
        operation_function.assert_called_once()
        np.testing.assert_array_equal(operation_function.call_args.args[0], [1, 2])
        np.testing.assert_array_equal(operation_function.call_args.args[1], [3, 4])


if __name__ == "__main__":
    unittest.main()
