import dataclasses

import numpy as np
import sympy as sp

from src.problab.validation.value_sets._base import _validate_value_set_configuration
from src.problab.value_sets._unknown import _UnknownValueSet

@dataclasses.dataclass(frozen=True)
class ValueSet:
    sympy_set: sp.Set | _UnknownValueSet
    dtype_types: tuple[type[np.generic], ...] | None

    def __post_init__(self) -> None:
        _validate_value_set_configuration(
            sympy_set=self.sympy_set,
            dtype_types=self.dtype_types,
        )
