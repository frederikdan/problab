import unittest
from unittest.mock import Mock
import numpy as np
import sympy as sp

from problab.operations._arithmetic import (
    _ABS,
    _ADD,
    _DIVIDE,
    _MULTIPLY,
    _NEGATIVE,
    _POWER,
    _REAL_POWER,
    _SUBTRACT,
)
from problab.operations._function import (
    _ARCCOS,
    _ARCSIN,
    _ARCTAN,
    _COS,
    _EXP,
    _EXPM1,
    _HYPOT,
    _LOG,
    _LOG1P,
    _LOGADDEXP,
    _SIN,
    _SQRT,
    _TAN,
)
from problab.random_variables.nodes import _ConstantNode, _OperationNode
from problab.random_variables.nodes._simplification import _simplify_or_create_node
from problab.operations._logical import _AND, _OR, _INVERT
from problab.operations._function import _FunctionOperation
from problab.value_sets import HomogeneousNumericValueSet, ObjectValueSet
from problab.value_sets.sets import BOOLEANS
from problab.value_sets.sets import NON_NEGATIVE_REALS, POSITIVE_REALS, REALS


def _arithmetic_node(operation, inputs):
    return _OperationNode(
        operation=operation,
        inputs=inputs,
        name=operation.name_func(*(node.name for node in inputs)),
        mathematical_value_set=operation.infer_mathematical_value_set(
            *(node.value_set for node in inputs),
        ),
        realization_value_set=operation.infer_realization_value_set(
            *(node._realization_value_set for node in inputs),
        ),
    )


def _function_node(operation, inputs, mathematical_value_set, realization_value_set):
    return _OperationNode(
        operation=operation,
        inputs=inputs,
        name=operation.name_func(*(node.name for node in inputs)),
        mathematical_value_set=mathematical_value_set,
        realization_value_set=HomogeneousNumericValueSet(
            sympy_set=realization_value_set.sympy_set, dtype_types=(np.float64,),
        ),
    )


class SimplificationTests(unittest.TestCase):

    def test_boolean_idempotence_and_double_negation_reuse_original_node(self):
        original = _ConstantNode(True)
        inverted = _OperationNode(_INVERT, (original,), "~E", BOOLEANS)
        for operation, inputs in ((_AND, (original, original)), (_OR, (original, original)), (_INVERT, (inverted,))):
            with self.subTest(operation=operation):
                result = _simplify_or_create_node(operation, inputs, BOOLEANS, BOOLEANS)
                self.assertIs(result, original)

    def test_boolean_complements_simplify_in_both_orders(self):
        original = _ConstantNode(True)
        inverted = _OperationNode(_INVERT, (original,), "~E", BOOLEANS)
        for operation, expected in ((_AND, False), (_OR, True)):
            for inputs in ((original, inverted), (inverted, original)):
                with self.subTest(operation=operation, inputs=inputs):
                    result = _simplify_or_create_node(operation, inputs, BOOLEANS, BOOLEANS)
                    self.assertIsInstance(result, _ConstantNode)
                    self.assertIs(result.value, expected)

    def test_boolean_rules_use_node_identity_rather_than_equal_names(self):
        first, second = _ConstantNode(True), _ConstantNode(True)
        inverted = _OperationNode(_INVERT, (second,), "~E", BOOLEANS)
        for operation, inputs in ((_AND, (first, second)), (_OR, (first, inverted))):
            with self.subTest(operation=operation):
                result = _simplify_or_create_node(operation, inputs, BOOLEANS, BOOLEANS)
                self.assertIsInstance(result, _OperationNode)
                self.assertEqual(result._inputs, inputs)

    def test_function_factory_passes_realization_sets_to_inferer_and_keeps_math_support(self):
        first, second = _ConstantNode(2), _ConstantNode(3)
        output = HomogeneousNumericValueSet(sp.S.Reals, (np.float64,), allows_nan=True)
        infer = Mock(return_value=output)
        operation = _FunctionOperation(operation=np.hypot, name_func=lambda x, y: f"custom({x}, {y})",
                                       infer_realization_value_set=infer)
        result = _simplify_or_create_node(operation, (first, second), REALS, NON_NEGATIVE_REALS)
        infer.assert_called_once_with(NON_NEGATIVE_REALS, first._realization_value_set, second._realization_value_set)
        self.assertIs(result.value_set, REALS)
        self.assertIs(result._realization_value_set, output)

    def test_numeric_function_inference_rejects_object_input_or_output_support(self):
        operation = _FunctionOperation(operation=np.exp, name_func=str, infer_realization_value_set=Mock())
        for input_node, output_set in ((_ConstantNode([1, 2]), REALS), (_ConstantNode(1), ObjectValueSet((1,)))):
            with self.subTest(input_node=input_node, output_set=output_set), self.assertRaises(TypeError):
                _simplify_or_create_node(operation, (input_node,), REALS, output_set)
        operation.infer_realization_value_set.assert_not_called()

    def test_custom_function_without_inferer_preserves_supplied_supports(self):
        operation = _FunctionOperation(operation=lambda x: x, name_func=lambda x: f"custom({x})")
        result = _simplify_or_create_node(operation, (_ConstantNode(2),), POSITIVE_REALS, NON_NEGATIVE_REALS)
        self.assertIs(result.value_set, POSITIVE_REALS)
        self.assertIs(result._realization_value_set, NON_NEGATIVE_REALS)

    def test_creates_operation_node_when_no_rule_applies(self):
        left = _ConstantNode(2)
        right = _ConstantNode(3)

        node = _simplify_or_create_node(
            operation=_ADD,
            inputs=(left, right),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIsInstance(node, _OperationNode)
        self.assertIs(node._operation, _ADD)
        self.assertEqual(node.name, "(2 + 3)")

    def test_addition_by_zero_returns_original_node(self):
        original_node = _ConstantNode(2)
        zero_node = _ConstantNode(0)

        node = _simplify_or_create_node(
            operation=_ADD,
            inputs=(original_node, zero_node),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(node, original_node)

    def test_zero_plus_addition_returns_original_node(self):
        original_node = _ConstantNode(2)

        node = _simplify_or_create_node(
            operation=_ADD,
            inputs=(_ConstantNode(0), original_node),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(node, original_node)

    def test_subtraction_by_zero_returns_original_node(self):
        original_node = _ConstantNode(2)

        node = _simplify_or_create_node(
            operation=_SUBTRACT,
            inputs=(original_node, _ConstantNode(0)),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(node, original_node)

    def test_multiplication_by_one_returns_original_node_in_both_orders(self):
        original_node = _ConstantNode(2)

        for inputs in (
            (original_node, _ConstantNode(1)),
            (_ConstantNode(1), original_node),
        ):
            with self.subTest(inputs=inputs):
                node = _simplify_or_create_node(
                    operation=_MULTIPLY,
                    inputs=inputs,
                    mathematical_value_set=REALS,
                    realization_value_set=REALS,
                )

                self.assertIs(node, original_node)

    def test_division_by_one_returns_original_node(self):
        original_node = _ConstantNode(2)

        node = _simplify_or_create_node(
            operation=_DIVIDE,
            inputs=(original_node, _ConstantNode(1)),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(node, original_node)

    def test_power_to_one_returns_base_node_for_both_power_operations(self):
        base_node = _ConstantNode(2)

        for operation in (_POWER, _REAL_POWER):
            with self.subTest(operation=operation):
                node = _simplify_or_create_node(
                    operation=operation,
                    inputs=(base_node, _ConstantNode(1)),
                    mathematical_value_set=REALS,
                    realization_value_set=REALS,
                )

                self.assertIs(node, base_node)

    def test_double_negation_returns_original_node(self):
        original_node = _ConstantNode(2)
        negative_node = _arithmetic_node(_NEGATIVE, (original_node,))

        node = _simplify_or_create_node(
            operation=_NEGATIVE,
            inputs=(negative_node,),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(node, original_node)

    def test_nested_absolute_value_returns_inner_absolute_value_node(self):
        original_node = _ConstantNode(-2)
        absolute_node = _arithmetic_node(_ABS, (original_node,))

        node = _simplify_or_create_node(
            operation=_ABS,
            inputs=(absolute_node,),
            mathematical_value_set=NON_NEGATIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

        self.assertIs(node, absolute_node)

    def test_log_of_exponential_returns_real_input_node(self):
        original_node = _ConstantNode(2)
        exponential_node = _OperationNode(
            operation=_EXP,
            inputs=(original_node,),
            name="exp(2)",
            mathematical_value_set=POSITIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

        node = _simplify_or_create_node(
            operation=_LOG,
            inputs=(exponential_node,),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(node, original_node)

    def test_exponential_of_logarithm_returns_positive_input_node(self):
        original_node = _ConstantNode(2)
        logarithm_node = _OperationNode(
            operation=_LOG,
            inputs=(original_node,),
            name="log(2)",
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        node = _simplify_or_create_node(
            operation=_EXP,
            inputs=(logarithm_node,),
            mathematical_value_set=POSITIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

        self.assertIs(node, original_node)

    def test_square_root_of_real_square_creates_absolute_value_node(self):
        original_node = _ConstantNode(-2)
        square_node = _arithmetic_node(
            _REAL_POWER,
            (original_node, _ConstantNode(2)),
        )

        node = _simplify_or_create_node(
            operation=_SQRT,
            inputs=(square_node,),
            mathematical_value_set=NON_NEGATIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

        self.assertIsInstance(node, _OperationNode)
        self.assertIs(node._operation, _ABS)
        self.assertEqual(node._inputs, (original_node,))
        self.assertEqual(
            node.value_set,
            _ABS.infer_mathematical_value_set(original_node.value_set),
        )
        self.assertEqual(
            node._realization_value_set,
            _ABS.infer_mathematical_value_set(original_node._realization_value_set),
        )

    def test_square_root_of_absolute_value_square_reuses_absolute_value_node(self):
        original_node = _ConstantNode(-2)
        absolute_node = _arithmetic_node(_ABS, (original_node,))
        square_node = _arithmetic_node(
            _REAL_POWER,
            (absolute_node, _ConstantNode(2)),
        )

        node = _simplify_or_create_node(
            operation=_SQRT,
            inputs=(square_node,),
            mathematical_value_set=NON_NEGATIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

        self.assertIs(node, absolute_node)

    def test_sine_of_arcsine_returns_input_in_closed_unit_interval(self):
        original_node = _ConstantNode(0)
        arcsine_node = _function_node(_ARCSIN, (original_node,), REALS, REALS)

        node = _simplify_or_create_node(
            operation=_SIN,
            inputs=(arcsine_node,),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(node, original_node)

    def test_cosine_of_arccosine_returns_input_in_closed_unit_interval(self):
        original_node = _ConstantNode(0)
        arccosine_node = _function_node(_ARCCOS, (original_node,), REALS, REALS)

        node = _simplify_or_create_node(
            operation=_COS,
            inputs=(arccosine_node,),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(node, original_node)

    def test_tangent_of_arctangent_returns_real_input(self):
        original_node = _ConstantNode(2)
        arctangent_node = _function_node(_ARCTAN, (original_node,), REALS, REALS)

        node = _simplify_or_create_node(
            operation=_TAN,
            inputs=(arctangent_node,),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIs(node, original_node)

    def test_logarithm_of_one_plus_input_creates_log1p_node(self):
        original_node = _ConstantNode(2)
        addition_node = _arithmetic_node(_ADD, (_ConstantNode(1), original_node))

        node = _simplify_or_create_node(
            operation=_LOG,
            inputs=(addition_node,),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIsInstance(node, _OperationNode)
        self.assertIs(node._operation, _LOG1P)
        self.assertEqual(node._inputs, (original_node,))

    def test_exponential_minus_one_creates_expm1_node(self):
        original_node = _ConstantNode(2)
        exponential_node = _function_node(
            _EXP,
            (original_node,),
            POSITIVE_REALS,
            NON_NEGATIVE_REALS,
        )

        node = _simplify_or_create_node(
            operation=_SUBTRACT,
            inputs=(exponential_node, _ConstantNode(1)),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIsInstance(node, _OperationNode)
        self.assertIs(node._operation, _EXPM1)
        self.assertEqual(node._inputs, (original_node,))

    def test_square_root_of_sum_of_squares_creates_hypot_node(self):
        left_node = _ConstantNode(3)
        right_node = _ConstantNode(4)
        addition_node = _arithmetic_node(
            _ADD,
            (
                _arithmetic_node(_REAL_POWER, (left_node, _ConstantNode(2))),
                _arithmetic_node(_REAL_POWER, (right_node, _ConstantNode(2))),
            ),
        )

        node = _simplify_or_create_node(
            operation=_SQRT,
            inputs=(addition_node,),
            mathematical_value_set=NON_NEGATIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

        self.assertIsInstance(node, _OperationNode)
        self.assertIs(node._operation, _HYPOT)
        self.assertEqual(node._inputs, (left_node, right_node))

    def test_logarithm_of_sum_of_exponentials_creates_logaddexp_node(self):
        left_node = _ConstantNode(2)
        right_node = _ConstantNode(3)
        addition_node = _arithmetic_node(
            _ADD,
            (
                _function_node(_EXP, (left_node,), POSITIVE_REALS, NON_NEGATIVE_REALS),
                _function_node(_EXP, (right_node,), POSITIVE_REALS, NON_NEGATIVE_REALS),
            ),
        )

        node = _simplify_or_create_node(
            operation=_LOG,
            inputs=(addition_node,),
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )

        self.assertIsInstance(node, _OperationNode)
        self.assertIs(node._operation, _LOGADDEXP)
        self.assertEqual(node._inputs, (left_node, right_node))

    def test_logarithm_of_one_plus_exponential_creates_softplus_node(self):
        original_node = _ConstantNode(2)
        addition_node = _arithmetic_node(
            _ADD,
            (
                _ConstantNode(1),
                _function_node(_EXP, (original_node,), POSITIVE_REALS, NON_NEGATIVE_REALS),
            ),
        )

        node = _simplify_or_create_node(
            operation=_LOG,
            inputs=(addition_node,),
            mathematical_value_set=POSITIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
        )

        self.assertIsInstance(node, _OperationNode)
        self.assertIs(node._operation, _LOGADDEXP)
        self.assertEqual(node._inputs[0].value, 0)
        self.assertIs(node._inputs[1], original_node)


if __name__ == "__main__":
    unittest.main()
