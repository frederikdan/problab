from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import sympy as sp

from ._unknown import _UnknownValueSet


class ValueSet(ABC):

    dtype_types: tuple[type[np.generic], ...] | None

    @abstractmethod
    def contains(self, values: Any) -> bool | np.ndarray:
        ...


class NumericValueSet(ValueSet, ABC):
    sympy_set: sp.Set | _UnknownValueSet

    allows_positive_infinity: bool = False
    allows_negative_infinity: bool = False
    allows_nan: bool = False

def __getattr__(name: str):

    if name == "HomogeneousNumericValueSet":
        from .homogeneous_numeric_value_set import (
            HomogeneousNumericValueSet,
        )

        return HomogeneousNumericValueSet

    if name == "MixedNumericValueSet":
        from .mixed_numeric_value_set import MixedNumericValueSet

        return MixedNumericValueSet

    if name == "ObjectValueSet":
        from .object_value_set import ObjectValueSet

        return ObjectValueSet

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "ValueSet",
    "NumericValueSet",
    "_UnknownValueSet",
]