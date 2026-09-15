from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from problab.value_sets.base import ValueSet

if TYPE_CHECKING:
    from problab.random_variables._context import _RealizationContext


class _Node(ABC):

    def __init__(self) -> None:
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
        # Only top-level dependencies, not a graph of dependencies of dependencies.
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
