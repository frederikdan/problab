from fractions import Fraction
from typing import Any

import numpy as np

from problab._config import RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE


def _float_if_fraction(value: Any) -> Any:
    if isinstance(value, Fraction):
        return float(value)

    return value


def _is_close(left, right) -> bool:
    return bool(np.isclose(
        _float_if_fraction(left),
        _float_if_fraction(right),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    ))
