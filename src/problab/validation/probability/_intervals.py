from numbers import Real
from math import isnan

import numpy as np


def _validate_interval_bound(value: Real, name: str, allow_nan: bool) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"'{name}' must be a real number.")

    if not allow_nan and isnan(value):
        raise ValueError(f"'{name}' must not be NaN.")


def _validate_confidence_interval_configuration(
        lower: Real,
        upper: Real,
        alpha: Real,
        ) -> None:

    _validate_interval_bound(lower, "lower", allow_nan=True)
    _validate_interval_bound(upper, "upper", allow_nan=True)

    lower_is_nan = isnan(lower)
    upper_is_nan = isnan(upper)

    if lower_is_nan != upper_is_nan:
        raise ValueError("'lower' and 'upper' must either both be NaN or neither be NaN.")

    if not lower_is_nan and lower > upper:
        raise ValueError("'lower' must be less than or equal to 'upper'.")

    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise TypeError("'alpha' must be a real number.")

    if not 0 < alpha < 1:
        raise ValueError("'alpha' must be between 0 and 1.")


def _validate_probability_interval_configuration(
        lower: Real,
        upper: Real,
        alpha: Real,
        is_estimate: bool,
        ) -> None:

    _validate_interval_bound(lower, "lower", allow_nan=False)
    _validate_interval_bound(upper, "upper", allow_nan=False)

    if lower > upper:
        raise ValueError("'lower' must be less than or equal to 'upper'.")

    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise TypeError("'alpha' must be a real number.")

    if not 0 < alpha < 1:
        raise ValueError("'alpha' must be between 0 and 1.")

    if not isinstance(is_estimate, bool):
        raise TypeError("'is_estimate' must be a bool.")
