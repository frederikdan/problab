import numpy as np
import sympy as sp

from problab.value_sets._unknown import _UnknownValueSet


def _validate_value_set_configuration(sympy_set: sp.Set | _UnknownValueSet,
                                      dtype_types: tuple[type[np.generic], ...] | None,
                                      *,
                                      allows_positive_infinity: bool = False,
                                      allows_negative_infinity: bool = False,
                                      allows_nan: bool = False,
                                      ) -> None:

    if not isinstance(sympy_set, (sp.Set, _UnknownValueSet)):
        raise TypeError("'sympy_set' must be a SymPy set or UNKNOWN_VALUE_SET.")

    for name, value in (
        ("allows_positive_infinity", allows_positive_infinity),
        ("allows_negative_infinity", allows_negative_infinity),
        ("allows_nan", allows_nan),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"'{name}' must be a bool.")

    if dtype_types is None:
        return

    if not isinstance(dtype_types, tuple):
        raise TypeError("'dtype_types' must be a tuple of NumPy scalar types or None.")

    if len(dtype_types) == 0:
        raise ValueError("'dtype_types' must contain at least one NumPy scalar type.")

    if not all(isinstance(dtype_type, type) and issubclass(dtype_type, np.generic) for dtype_type in dtype_types):
        raise TypeError("'dtype_types' must contain only NumPy scalar types.")
