import dataclasses
from decimal import Decimal
from numbers import Number
from typing import Any

import numpy as np
import sympy as sp

from ._unknown import _UnknownValueSet
from ._utils import _to_sympy_value
from .base import NumericValueSet


@dataclasses.dataclass(frozen=True)
class MixedNumericValueSet(NumericValueSet):

    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise TypeError("'values' must be a tuple.")

        if not self.values:
            raise ValueError("'values' must contain at least one value.")

        if not all(
                isinstance(value, (bool, np.bool_, Number, Decimal))
                for value in self.values
        ):
            raise TypeError("'values' must contain only numeric or Boolean values.")

    @property
    def dtype_types(self) -> None:
        return None

    @property
    def sympy_set(self) -> sp.Set | _UnknownValueSet:
        try:
            return sp.FiniteSet(*(_to_sympy_value(value) for value in self.values))
        except (AttributeError, TypeError, ValueError, sp.SympifyError):
            return _UnknownValueSet()

    def contains(self, values: Any) -> bool | np.ndarray:
        array = np.asarray(values, dtype=object)
        scalar = array.ndim == 0
        flat_values = array.reshape(-1)

        result = np.asarray(
            [
                any(value == candidate for candidate in self.values)
                for value in flat_values
            ],
            dtype=bool,
        )

        if scalar:
            return bool(result[0])

        return result.reshape(array.shape)
