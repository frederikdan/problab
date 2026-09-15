"""Value sets represented by SymPy sets."""

import dataclasses
from typing import Any

import numpy as np
import sympy as sp

from problab.validation.value_sets._base import _validate_value_set_configuration

from ._unknown import _UnknownValueSet
from .base import ValueSet


@dataclasses.dataclass(frozen=True)
class NumericValueSet(ValueSet):

    sympy_set: sp.Set | _UnknownValueSet
    dtype_types: tuple[type[np.generic], ...] | None

    def __post_init__(self) -> None:
        _validate_value_set_configuration(
            sympy_set=self.sympy_set,
            dtype_types=self.dtype_types,
        )

    def contains(self, values: Any) -> bool | np.ndarray:

        array = np.asarray(values)
        scalar = array.ndim == 0
        flat_values = array.reshape(-1)

        if isinstance(self.sympy_set, _UnknownValueSet):
            result = np.zeros(flat_values.shape, dtype=bool)
        else:
            result = np.empty(flat_values.shape, dtype=bool)
            for index, value in enumerate(flat_values):
                try:
                    if (
                            isinstance(value, (float, np.floating))
                            and np.isfinite(value)
                            and value == np.trunc(value)
                    ):
                        symbolic_value = sp.Integer(int(value))
                    else:
                        symbolic_value = sp.sympify(value)
                    result[index] = self.sympy_set.contains(symbolic_value) is sp.true
                except (AttributeError, TypeError, ValueError, NotImplementedError, sp.SympifyError):
                    result[index] = False

        if scalar:
            return bool(result[0])
        return result.reshape(array.shape)
