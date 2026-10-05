import sympy as sp

from problab.operations._base import _Operation
from problab.operations._arithmetic import (
    _ADD,
    _SUBTRACT,
    _MULTIPLY,
    _DIVIDE,
    _POWER,
    _REAL_POWER,
    _NEGATIVE,
    _ABS,
)
from problab.operations._function import (
    _FunctionOperation,
    _EXP,
    _LOG,
    _SQRT,
    _SIN,
    _ARCSIN,
    _COS,
    _ARCCOS,
    _TAN,
    _ARCTAN,
    _LOG1P,
    _EXPM1,
    _HYPOT,
    _LOGADDEXP,
)
from problab.operations._logical import (
    _AND,
    _OR,
    _INVERT,
)
from problab.random_variables.nodes.base import _Node
from problab.random_variables.nodes.constant import _ConstantNode
from problab.random_variables.nodes.operation import _OperationNode
from problab.random_variables.nodes._utils import _node_is_constant, _node_is_operation, _node_is_square_operation
from problab.value_sets import REALS, POSITIVE_REALS, NON_NEGATIVE_REALS
from problab.value_sets.sets import (
    GT_NEG_ONE_REALS,
    GE_NEG_ONE_REALS,
)
from problab.value_sets._utils import is_known_subset
from problab.value_sets.base import NumericValueSet, ValueSet


def _simplify_or_create_node(
    operation: _Operation,
    inputs: tuple[_Node, ...],
    mathematical_value_set: ValueSet,
    realization_value_set: ValueSet,
) -> _Node:

    simplified_node = _simplify_operation(
        operation=operation,
        inputs=inputs,
    )

    if simplified_node is not None:
        return simplified_node

    if (
        isinstance(operation, _FunctionOperation)
        and operation.infer_realization_value_set is not None
    ):
        input_sets = tuple(node._realization_value_set for node in inputs)
        if not isinstance(realization_value_set, NumericValueSet) or not all(
            isinstance(value_set, NumericValueSet) for value_set in input_sets
        ):
            raise TypeError("Numerical function inference requires numeric value sets.")
        realization_value_set = operation.infer_realization_value_set(
            realization_value_set,
            *input_sets,
        )

    return _OperationNode(
        operation=operation,
        inputs=inputs,
        name=operation.name_func(*(input_node.name for input_node in inputs)),
        mathematical_value_set=mathematical_value_set,
        realization_value_set=realization_value_set,
    )


def _simplify_operation(
    operation: _Operation,
    inputs: tuple[_Node, ...],
) -> _Node | None:

    if operation is _ADD:
        left, right = inputs

        if _node_is_constant(left, 0):
            return right

        if _node_is_constant(right, 0):
            return left

    if operation is _SUBTRACT:
        left, right = inputs

        if _node_is_constant(right, 0):
            return left

        if _node_is_constant(right, 1) and _node_is_operation(left, _EXP):
            original_node = left._inputs[0]

            if is_known_subset(original_node.value_set, REALS):
                return _simplify_or_create_node(
                    operation=_EXPM1,
                    inputs=(original_node,),
                    mathematical_value_set=GT_NEG_ONE_REALS,
                    realization_value_set=GE_NEG_ONE_REALS,
                )

    if operation is _MULTIPLY:
        left, right = inputs

        if _node_is_constant(left, 1):
            return right

        if _node_is_constant(right, 1):
            return left

    if operation is _DIVIDE:
        left, right = inputs

        if _node_is_constant(right, 1):
            return left

    if operation is _POWER or operation is _REAL_POWER:
        base, exponent = inputs

        if _node_is_constant(exponent, 1):
            return base

    if operation is _NEGATIVE:
        input_node = inputs[0]

        if _node_is_operation(input_node, _NEGATIVE):
            return input_node._inputs[0]

    if operation is _ABS:
        input_node = inputs[0]

        if _node_is_operation(input_node, _ABS):
            return input_node

    if operation is _LOG:
        input_node = inputs[0]

        if _node_is_operation(input_node, _EXP):
            original_node = input_node._inputs[0]

            if is_known_subset(original_node.value_set, REALS):
                return original_node

        if _node_is_operation(input_node, _ADD):
            left, right = input_node._inputs

            if _node_is_constant(left, 1) and _node_is_operation(right, _EXP):
                original_node = right._inputs[0]

                if is_known_subset(original_node.value_set, REALS):
                    return _simplify_or_create_node(
                        operation=_LOGADDEXP,
                        inputs=(_ConstantNode(0), original_node),
                        mathematical_value_set=POSITIVE_REALS,
                        realization_value_set=NON_NEGATIVE_REALS,
                    )

            if _node_is_constant(right, 1) and _node_is_operation(left, _EXP):
                original_node = left._inputs[0]

                if is_known_subset(original_node.value_set, REALS):
                    return _simplify_or_create_node(
                        operation=_LOGADDEXP,
                        inputs=(_ConstantNode(0), original_node),
                        mathematical_value_set=POSITIVE_REALS,
                        realization_value_set=NON_NEGATIVE_REALS,
                    )

            if _node_is_operation(left, _EXP) and _node_is_operation(right, _EXP):
                left_original_node = left._inputs[0]
                right_original_node = right._inputs[0]

                if (
                    is_known_subset(left_original_node.value_set, REALS)
                    and is_known_subset(right_original_node.value_set, REALS)
                ):
                    return _simplify_or_create_node(
                        operation=_LOGADDEXP,
                        inputs=(left_original_node, right_original_node),
                        mathematical_value_set=REALS,
                        realization_value_set=REALS,
                    )

            if _node_is_constant(left, 1):
                original_node = right
            elif _node_is_constant(right, 1):
                original_node = left
            else:
                original_node = None

            if (
                original_node is not None
                and is_known_subset(
                    original_node.value_set,
                    GT_NEG_ONE_REALS,
                )
            ):
                return _simplify_or_create_node(
                    operation=_LOG1P,
                    inputs=(original_node,),
                    mathematical_value_set=REALS,
                    realization_value_set=REALS,
                )

    if operation is _EXP:
        input_node = inputs[0]

        if _node_is_operation(input_node, _LOG):
            original_node = input_node._inputs[0]

            if is_known_subset(original_node.value_set, POSITIVE_REALS):
                return original_node

    if operation is _SQRT:
        input_node = inputs[0]

        if _node_is_operation(input_node, _ADD):
            left, right = input_node._inputs

            if _node_is_square_operation(left) and _node_is_square_operation(right):
                left_original_node = left._inputs[0]
                right_original_node = right._inputs[0]

                if (
                    is_known_subset(left_original_node.value_set, REALS)
                    and is_known_subset(right_original_node.value_set, REALS)
                ):
                    return _simplify_or_create_node(
                        operation=_HYPOT,
                        inputs=(left_original_node, right_original_node),
                        mathematical_value_set=NON_NEGATIVE_REALS,
                        realization_value_set=NON_NEGATIVE_REALS,
                    )

        if _node_is_square_operation(input_node):
            original_node = input_node._inputs[0]

            if is_known_subset(original_node.value_set, REALS):
                return _simplify_or_create_node(
                    operation=_ABS,
                    inputs=(original_node,),
                    mathematical_value_set=_ABS.infer_mathematical_value_set(
                        original_node.value_set,
                    ),
                    realization_value_set=_ABS.infer_realization_value_set(
                        original_node._realization_value_set,
                    ),
                )

    if operation is _SIN:
        input_node = inputs[0]

        if _node_is_operation(input_node, _ARCSIN):
            original_node = input_node._inputs[0]

            if is_known_subset(original_node.value_set, sp.Interval(-1, 1)):
                return original_node

    if operation is _COS:
        input_node = inputs[0]

        if _node_is_operation(input_node, _ARCCOS):
            original_node = input_node._inputs[0]

            if is_known_subset(original_node.value_set, sp.Interval(-1, 1)):
                return original_node

    if operation is _TAN:
        input_node = inputs[0]

        if _node_is_operation(input_node, _ARCTAN):
            original_node = input_node._inputs[0]

            if is_known_subset(original_node.value_set, REALS):
                return original_node

    if operation is _INVERT:
        input_node = inputs[0]

        if _node_is_operation(input_node, _INVERT):
            return input_node._inputs[0]

    if operation is _AND:
        left, right = inputs

        if left is right:
            return left

        if (
                _node_is_operation(left, _INVERT)
                and left._inputs[0] is right
        ) or (
                _node_is_operation(right, _INVERT)
                and right._inputs[0] is left
        ):
            return _ConstantNode(False)

    if operation is _OR:
        left, right = inputs

        if left is right:
            return left

        if (
                _node_is_operation(left, _INVERT)
                and left._inputs[0] is right
        ) or (
                _node_is_operation(right, _INVERT)
                and right._inputs[0] is left
        ):
            return _ConstantNode(True)


    return None
