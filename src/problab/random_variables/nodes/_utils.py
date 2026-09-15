from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from problab.value_sets._unknown import _UnknownValueSet
from problab.value_sets.base import ValueSet


def _sympy_constant_value(value: Any) -> sp.Basic:
    if (
            isinstance(value, (float, np.floating))
            and np.isfinite(value)
            and value == np.trunc(value)
    ):
        return sp.Integer(int(value))

    return sp.sympify(value)


def _constant_array(value: Any) -> np.ndarray:
    array = np.asarray(value)

    if array.ndim == 0:
        return array

    atomic_array = np.empty((), dtype=object)
    atomic_array[()] = value
    return atomic_array


def _constant_value_set(value: Any, array: np.ndarray) -> ValueSet:
    try:
        sympy_set = sp.FiniteSet(_sympy_constant_value(value))
    except (AttributeError, TypeError, ValueError, sp.SympifyError):
        sympy_set = _UnknownValueSet()

    return ValueSet(
        sympy_set=sympy_set,
        dtype_types=(array.dtype.type,),
    )
