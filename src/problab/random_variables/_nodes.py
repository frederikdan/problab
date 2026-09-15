from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Any

import numpy as np
import sympy as sp

from problab.value_sets.base import ValueSet
from problab.value_sets.sets import UNKNOWN_VALUE_SET

if TYPE_CHECKING:
    from problab.distributions.base import Distribution
    from problab.random_variables._context import _RealizationContext


T = TypeVar('T')


class _Node(ABC):

    def __init__(self):
        self._name = None
        self._extended_name = self._name

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @property
    @abstractmethod
    def value_set(self) -> ValueSet:
        ...

    @property
    @abstractmethod
    def dependencies(self) -> set[_Node]:
        # Only top level of dependencies not a graph of dependencies of dependencies.
        ...

    @abstractmethod
    def _evaluate(self, context: _RealizationContext) -> np.ndarray:
        ...

    @property
    def name(self) -> str:
        return self._name

    @property
    def extended_name(self) -> str:
        return self._extended_name

    @property
    def has_dependencies(self) -> bool:
        return bool(self.dependencies)


class _ConstantNode(_Node, Generic[T]):

    def __init__(self, value: T) -> None:

        super().__init__()

        self._value = value
        self._name = str(value)
        self._value_set = ValueSet(
            sympy_set=sp.FiniteSet(value),
            dtype_types=(np.asarray(value).dtype.type,),
        )

    def __repr__(self) -> str:
        return f"ConstantNode({self.name})"

    @property
    def value(self) -> T:
        return self._value

    @property
    def value_set(self) -> ValueSet:
        return self._value_set

    @property
    def dependencies(self) -> set[_Node]:
        return set()

    def _evaluate(self, context: _RealizationContext) -> np.ndarray:
        return np.asarray(self._value)


class _DistributionNode(_Node):

    def __init__(self,
                 distribution: Distribution,
                 rv_name: str
                 ) -> None:

        super().__init__()

        self._distribution = distribution
        self._name = f"{rv_name}"
        self._extended_name = self._name + f" ~ {distribution.name}"

    def __repr__(self) -> str:
        return f"DistributionNode({self.name})"

    @property
    def distribution(self) -> Distribution:
        return self._distribution

    @property
    def value_set(self) -> ValueSet:
        return self._distribution.value_set

    @property
    def dependencies(self) -> set[_Node]:
        return self._distribution._node_dependencies

    def _evaluate(self, context: _RealizationContext) -> np.ndarray:
        return self._distribution._evaluate(context)


class _OperationNode(_Node):

    def __init__(self,
                 operation: Callable[..., np.ndarray],
                 inputs: tuple[_Node, ...],
                 name: str,
                 value_set: ValueSet = UNKNOWN_VALUE_SET
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

        values = tuple(
            context.evaluate(node)
            for node in self._inputs
        )

        return self._operation(*values)





