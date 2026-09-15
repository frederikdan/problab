import numpy as np
from numbers import Real
from enum import Enum


def _validate_num_samples(value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("'num_samples' must be an integer.")

    if value < 1:
        raise ValueError("'num_samples' must be at least 1.")


def _validate_rng(value: np.random.Generator | None) -> None:
    if value is not None and not isinstance(value, np.random.Generator):
        raise TypeError("'rng' must be a np.random.Generator or None.")


def _validate_validate(value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError("'validate' must be a bool.")


def _validate_alpha(value: Real) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("'alpha' must be a real number.")

    if not 0 < value < 1:
        raise ValueError("'alpha' must be between 0 and 1.")


def _validate_q(value: Real) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("'q' must be a real number.")

    if not 0 < value < 1:
        raise ValueError("'q' must be between 0 and 1.")


def _validate_enum(value, enum_type: type[Enum], name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"'{name}' must be a {enum_type.__name__}.")


def _validate_max_size(value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("'max_size' must be an integer.")

    if value < 1:
        raise ValueError("'max_size' must be at least 1.")
