import unittest

from problab._operations import (
    _ABS,
    _ADD,
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
    _DIVIDE,
    _MULTIPLY,
    _NEGATIVE,
    _POWER,
    _REAL_POWER,
    _SIN,
    _SQRT,
    _SUBTRACT,
    _TAN,
)
from problab.random_variables.nodes import _ConstantNode, _OperationNode
from problab.random_variables.nodes._simplification import _simplify_or_create_node
from problab.value_sets.sets import NON_NEGATIVE_REALS, POSITIVE_REALS, REALS


def _arithmetic_node(operation, inputs):
    return _OperationNode(
        operation=operation,
        inputs=inputs,
        name=operation.name_func(*(node.name for node in inputs)),
        mathematical_value_set=operation.infer_output_value_set(
            *(node.value_set for node in inputs),
        ),
        realization_value_set=operation.infer_output_value_set(
            *(node._realization_value_set for node in inputs),
        ),
    )


def _function_node(operation, inputs, mathematical_value_set, realization_value_set):
    return _OperationNode(
        operation=operation,
        inputs=inputs,
        name=operation.name_func(*(node.name for node in inputs)),
        mathematical_value_set=mathematical_value_set,
        realization_value_set=realization_value_set,
    )


class SimplificationTests(unittest.TestCase):

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
            _ABS.infer_output_value_set(original_node.value_set),
        )
        self.assertEqual(
            node._realization_value_set,
            _ABS.infer_output_value_set(original_node._realization_value_set),
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
