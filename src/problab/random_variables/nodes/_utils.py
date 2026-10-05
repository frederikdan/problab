from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
import sympy as sp

from problab.operations._base import _Operation
from problab.operations._arithmetic import (
    _POWER,
    _REAL_POWER,
)
from problab.random_variables.nodes.base import _Node
from problab.random_variables.nodes.operation import _OperationNode
from problab.value_sets._unknown import _UnknownValueSet
from problab.value_sets._utils import _to_sympy_value
from problab.value_sets.base import ValueSet
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.object_value_set import ObjectValueSet

if TYPE_CHECKING:
    from problab.random_variables.nodes.constant import _ConstantNode


def _constant_array(value: Any) -> np.ndarray:
    array = np.asarray(value)

    if array.ndim == 0:
        return array

    atomic_array = np.empty((), dtype=object)
    atomic_array[()] = value
    return atomic_array


def _constant_value_set(value: Any, array: np.ndarray) -> ValueSet:
    try:
        sympy_set = sp.FiniteSet(_to_sympy_value(value))
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        sympy_set = _UnknownValueSet()

    if isinstance(sympy_set, _UnknownValueSet):
        return ObjectValueSet(objects=(value,))

    return HomogeneousNumericValueSet(
        sympy_set=sympy_set,
        dtype_types=(array.dtype.type,),
    )


def _node_is_constant(
    node: _Node,
    value: Any,
) -> TypeGuard[_ConstantNode]:
    from problab.random_variables.nodes.constant import _ConstantNode

    return (
        isinstance(node, _ConstantNode)
        and node.value == value
    )

def _node_is_operation(
    node: _Node,
    operation: _Operation,
) -> TypeGuard[_OperationNode]:
    return (
        isinstance(node, _OperationNode)
        and node._operation is operation
    )

def _node_is_square_operation(
    node: _Node,
) -> TypeGuard[_OperationNode]:
    return (
        isinstance(node, _OperationNode)
        and node._operation in (_POWER, _REAL_POWER)
        and len(node._inputs) == 2
        and _node_is_constant(node._inputs[1], 2)
    )
