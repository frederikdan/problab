import unittest
import warnings
from unittest.mock import Mock

import numpy as np

from problab.operations._arithmetic import (_ADD)
from problab.operations._function import (_EXP, _FunctionOperation)
from problab.random_variables.nodes.operation import _OperationNode
from problab.value_sets.sets import NON_NEGATIVE_REALS, REALS


class OperationNodeTests(unittest.TestCase):

    def test_evaluation_uses_context_policy_for_overflow_division_and_invalid_values(self):
        for operation, arguments in ((np.exp, (np.array([1000.0]),)),
                                     (np.divide, (np.array([1.0]), np.array([0.0]))),
                                     (np.sqrt, (np.array([-1.0]),))):
            for policy in ("warn", "raise", "ignore"):
                with self.subTest(operation=operation.__name__, policy=policy):
                    context = Mock(numerical_error_policy=policy)
                    context.evaluate.side_effect = arguments
                    node = _OperationNode(_FunctionOperation(operation=operation, name_func=str),
                                          tuple(object() for _ in arguments), "operation", REALS)
                    original = np.geterr().copy()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        if policy == "raise":
                            with self.assertRaises(FloatingPointError):
                                node._evaluate(context)
                        else:
                            self.assertTrue(np.any(~np.isfinite(node._evaluate(context))))
                            self.assertEqual(bool(caught), policy == "warn")
                    self.assertEqual(np.geterr(), original)

    def test_evaluation_ignores_underflow_even_when_global_policy_raises(self):
        context = Mock(numerical_error_policy="raise")
        context.evaluate.return_value = np.array([-1000.0])
        node = _OperationNode(_EXP, (object(),), "exp", REALS)
        with np.errstate(under="raise"):
            np.testing.assert_array_equal(node._evaluate(context), [0.0])
            self.assertEqual(np.geterr()["under"], "raise")

    def test_evaluation_restores_numpy_policy_after_arbitrary_callable_error(self):
        def fail(values):
            self.assertEqual(np.geterr(), dict(over="ignore", divide="ignore", invalid="ignore", under="ignore"))
            raise RuntimeError("callback failed")
        node = _OperationNode(_FunctionOperation(operation=fail, name_func=str), (object(),), "fail", REALS)
        context = Mock(numerical_error_policy="ignore")
        original = np.geterr().copy()
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            node._evaluate(context)
        self.assertEqual(np.geterr(), original)

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
        context = Mock(numerical_error_policy="warn")
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
