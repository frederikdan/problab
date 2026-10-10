from __future__ import annotations

from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import sympy as sp

from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets._utils import _detect_non_finite_values, _to_sympy_value
from problab.value_sets.base import ValueSet, NumericValueSet
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
        self._mathematical_value_set = _constant_value_set(value, self._array)

    def __repr__(self) -> str:
        return f"ConstantNode({self.name})"

    @property
    def value(self) -> T:
        return self._value

    @property
    def value_set(self):
        return self._mathematical_value_set

    @cached_property
    def _realization_value_set(self) -> ValueSet:
        if not isinstance(self._mathematical_value_set, NumericValueSet):
            return self._mathematical_value_set

        positive_infinity, negative_infinity, nan = (
            _detect_non_finite_values(self._array)
        )

        sympy_set = self._mathematical_value_set.sympy_set

        if isinstance(self._value, Fraction):
            sympy_set = sp.FiniteSet(_to_sympy_value(self._array[()]))

        return HomogeneousNumericValueSet(
            sympy_set=sympy_set,
            dtype_types=(self._array.dtype.type,),
            allows_positive_infinity=positive_infinity,
            allows_negative_infinity=negative_infinity,
            allows_nan=nan,
        )

    @property
    def dependencies(self) -> set[_Node]:
        return set()

    def _evaluate(self, context: _RealizationContext) -> np.ndarray:
        return self._array
