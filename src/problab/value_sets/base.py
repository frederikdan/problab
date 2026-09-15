import dataclasses
from typing import TypeAlias

import numpy as np
import sympy as sp

class _UnknownValueSet:

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNKNOWN_VALUE_SET"

@dataclasses.dataclass(frozen=True)
class ValueSet:
    sympy_set: sp.Set | _UnknownValueSet
    dtype_types: tuple[type[np.generic], ...] | None