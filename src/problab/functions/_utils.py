from numbers import Real

from problab._utils import _float_if_fraction
from problab.operations._base import _Operation
from problab.operations._arithmetic import _ArithmeticOperation
from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes import _ConstantNode
from problab.random_variables.nodes._simplification import _simplify_or_create_node
from problab.validation._common import _require_supported_operation_inputs
from problab.value_sets.base import NumericValueSet
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet

def _apply(
    x: RandomVariable | Real,
    operation: _Operation,
    mathematical_value_set: NumericValueSet,
    realization_value_set: NumericValueSet,
    *others: RandomVariable | Real,
) -> RandomVariable | float:
    arguments = (x, *others)

    input_nodes = tuple(
        argument._node
        if isinstance(argument, RandomVariable)
        else _ConstantNode(argument)
        for argument in arguments
    )

    _require_supported_operation_inputs(
        input_nodes,
        operation_name=operation.name_func(
            *(node.name for node in input_nodes)
        ),
        supported_input_types=operation.supported_input_types,
    )

    if not any(isinstance(argument, RandomVariable) for argument in arguments):
        return float(operation.operation(
            *(_float_if_fraction(argument) for argument in arguments)
        ))

    if isinstance(operation, _ArithmeticOperation):
        inferred_set = operation.infer_realization_value_set(
            *(node._realization_value_set for node in input_nodes)
        )
        realization_value_set = HomogeneousNumericValueSet(
            sympy_set=realization_value_set.sympy_set,
            dtype_types=inferred_set.dtype_types,
            allows_positive_infinity=inferred_set.allows_positive_infinity,
            allows_negative_infinity=inferred_set.allows_negative_infinity,
            allows_nan=inferred_set.allows_nan,
        )

    if isinstance(x, RandomVariable) and not others:
        return x._apply_operation(
            operation=operation,
            mathematical_value_set=mathematical_value_set,
            realization_value_set=realization_value_set,
            vectorized=True,
        )

    return RandomVariable._from_node(
        _simplify_or_create_node(
            operation=operation,
            inputs=input_nodes,
            mathematical_value_set=mathematical_value_set,
            realization_value_set=realization_value_set,
        )
    )
