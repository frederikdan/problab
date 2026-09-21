from numbers import Real

from problab._operations import _Operation
from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes import _ConstantNode
from problab.random_variables.nodes._simplification import _simplify_or_create_node
from problab.value_sets.base import NumericValueSet

def _apply(
    x: RandomVariable | Real,
    operation: _Operation,
    mathematical_value_set: NumericValueSet,
    realization_value_set: NumericValueSet,
    *others: RandomVariable | Real,
) -> RandomVariable | float:
    arguments = (x, *others)

    if not any(isinstance(argument, RandomVariable) for argument in arguments):
        return float(operation.operation(*arguments))

    if isinstance(x, RandomVariable) and not others:
        return x._apply_operation(
            operation=operation,
            mathematical_value_set=mathematical_value_set,
            realization_value_set=realization_value_set,
            vectorized=True,
        )

    input_nodes = tuple(
        argument._node
        if isinstance(argument, RandomVariable)
        else _ConstantNode(argument)
        for argument in arguments
    )

    return RandomVariable._from_node(
        _simplify_or_create_node(
            operation=operation,
            inputs=input_nodes,
            mathematical_value_set=mathematical_value_set,
            realization_value_set=realization_value_set,
        )
    )
