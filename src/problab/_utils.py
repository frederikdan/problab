from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray

from problab._config import RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE


def _float_if_fraction(value: Any) -> Any:
    if isinstance(value, Fraction):
        return float(value)

    return value


def _convert_fractions_in_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind != "O":
        return values

    if not any(isinstance(value, Fraction) for value in values.flat):
        return values

    converted = values.copy()
    for index, value in enumerate(values.flat):
        converted.flat[index] = _float_if_fraction(value)

    items = tuple(converted.flat)
    try:
        numeric = np.asarray(items)
    except (TypeError, ValueError, OverflowError):
        return converted

    if numeric.dtype.kind not in "iufc":
        return converted

    if not _conversion_preserves_values(items, numeric):
        return converted

    return numeric.reshape(values.shape)


def _conversion_preserves_values(
    categories: tuple[Any, ...],
    converted: NDArray[Any],
) -> bool:

    def _exact_real_value(value) -> Fraction:
        if isinstance(value, (int, np.integer)):
            return Fraction(int(value))

        if isinstance(value, (float, np.floating)):
            return Fraction(*value.as_integer_ratio())

        return Fraction(value)

    def _real_values_equal(original, result) -> bool:
        if isinstance(original, (float, np.floating)):
            if np.isnan(original):
                return (
                    isinstance(result, (float, np.floating))
                    and bool(np.isnan(result))
                )

            if np.isinf(original):
                return bool(original == result)

        return _exact_real_value(original) == _exact_real_value(result)

    def _numeric_values_equal(original, result) -> bool:
        if isinstance(original, (complex, np.complexfloating)):
            original_real, original_imag = original.real, original.imag
        else:
            original_real, original_imag = original, 0

        if isinstance(result, (complex, np.complexfloating)):
            result_real, result_imag = result.real, result.imag
        else:
            result_real, result_imag = result, 0

        return (
            _real_values_equal(original_real, result_real)
            and _real_values_equal(original_imag, result_imag)
        )

    if converted.shape != (len(categories),):
        return False

    try:
        return all(
            _numeric_values_equal(original, result)
            for original, result in zip(categories, converted)
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _is_close(left, right) -> bool:
    return bool(np.isclose(
        _float_if_fraction(left),
        _float_if_fraction(right),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    ))
