from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from problab.random_variables.nodes.base import _Node
from problab.value_sets.base import ValueSet
from problab.value_sets.sets import UNKNOWN_VALUE_SET

if TYPE_CHECKING:
    from problab.random_variables._context import _RealizationContext


class _OperationNode(_Node):

    def __init__(
            self,
            operation: Callable[..., np.ndarray],
            inputs: tuple[_Node, ...],
            name: str,
            value_set: ValueSet = UNKNOWN_VALUE_SET,
    ) -> None:
        super().__init__()

        self._operation = operation
        self._inputs = inputs
        self._value_set = value_set
        self._name = name

    def __repr__(self) -> str:
        return f"OperationNode({self.name})"

    @property
    def value_set(self) -> ValueSet:
        return self._value_set

    @property
    def dependencies(self) -> set[_Node]:
        return set(self._inputs)

    def _evaluate(self, context: _RealizationContext) -> np.ndarray:
        values = tuple(context.evaluate(node) for node in self._inputs)
        return self._operation(*values)
