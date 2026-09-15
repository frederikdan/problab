from typing import get_args

import numpy as np
from numbers import Real

from src.problab.statistics.quantiles import QuantileMethod


def _validate_real_input(value: Real | np.ndarray, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"'{name}' must be a real number or a NumPy array of real numbers.")

    if isinstance(value, Real):
        return

    if not isinstance(value, np.ndarray):
        raise TypeError(f"'{name}' must be a real number or a NumPy array of real numbers.")

    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"'{name}' must be a real number or a NumPy array of real numbers.")


def _validate_cdf_input(value: Real | np.ndarray) -> None:
    _validate_real_input(value, "x")


def _validate_ppf_input(value: Real | np.ndarray) -> None:
    _validate_real_input(value, "q")

    values = np.asarray(value)

    if np.any(np.isnan(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("'q' must be between 0 and 1.")


def _validate_quantile_method(value: QuantileMethod) -> None:
    if not isinstance(value, str):
        raise TypeError("'quantile_method' must be a string.")

    if value not in get_args(QuantileMethod):
        raise ValueError(f"'quantile_method' must be one of {get_args(QuantileMethod)}.")
