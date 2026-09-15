from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from problab.random_variables.nodes.base import _Node
from problab.random_variables.nodes._utils import _constant_array, _constant_value_set

if TYPE_CHECKING:
    from problab.random_variables._context import _RealizationContext


T = TypeVar("T")


class _ConstantNode(_Node, Generic[T]):

    def __init__(self, value: T) -> None:
        super().__init__()

        self._value = value
        self._array = _constant_array(value)
        self._name = str(value)
        self._value_set = _constant_value_set(value, self._array)

    def __repr__(self) -> str:
        return f"ConstantNode({self.name})"

    @property
    def value(self) -> T:
        return self._value

    @property
    def value_set(self):
        return self._value_set

    @property
    def dependencies(self) -> set[_Node]:
        return set()

    def _evaluate(self, context: _RealizationContext) -> np.ndarray:
        return self._array
