import dataclasses
from typing import Any

import numpy as np
import sympy as sp

from problab.validation.value_sets._base import (
    _validate_value_set_configuration,
)
from ._unknown import _UnknownValueSet
from ._utils import _to_sympy_value, _contains_non_finite_value
from .base import NumericValueSet


@dataclasses.dataclass(frozen=True)
class HomogeneousNumericValueSet(NumericValueSet):

    sympy_set: sp.Set | _UnknownValueSet
    dtype_types: tuple[type[np.generic], ...] | None

    allows_positive_infinity: bool = dataclasses.field(
        default=False, kw_only=True,
    )
    allows_negative_infinity: bool = dataclasses.field(
        default=False, kw_only=True,
    )
    allows_nan: bool = dataclasses.field(
        default=False, kw_only=True,
    )

    def __post_init__(self) -> None:
        _validate_value_set_configuration(
            sympy_set=self.sympy_set,
            dtype_types=self.dtype_types,
            allows_positive_infinity=self.allows_positive_infinity,
            allows_negative_infinity=self.allows_negative_infinity,
            allows_nan=self.allows_nan,
        )

    def contains(self, values: Any) -> bool | np.ndarray:
        array = np.asarray(values)
        scalar = array.ndim == 0
        flat_values = array.reshape(-1)

        result = np.empty(flat_values.shape, dtype=bool)

        for index, value in enumerate(flat_values):
            non_finite_result = _contains_non_finite_value(value, self)

            if non_finite_result is not None:
                result[index] = non_finite_result

            elif isinstance(self.sympy_set, _UnknownValueSet):
                result[index] = False

            else:
                try:
                    symbolic_value = _to_sympy_value(value)
                    result[index] = (
                        self.sympy_set.contains(symbolic_value) is sp.true
                    )
                except (TypeError, ValueError, NotImplementedError):
                    result[index] = False

        if scalar:
            return bool(result[0])

        return result.reshape(array.shape)