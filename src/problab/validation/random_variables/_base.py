from collections.abc import Callable
from numbers import Real

import sympy as sp

from problab.distributions.base import Distribution
from problab.value_sets.base import ValueSet


def _validate_distribution(value: Distribution) -> None:
    if not isinstance(value, Distribution):
        raise TypeError("'distribution' must be a Distribution.")


def _validate_name(value: str | None) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError("'name' must be a string or None.")


def _validate_interval_bound(value: Real) -> None:
    if not isinstance(value, Real):
        raise TypeError("Interval bounds must be real numbers.")


def _validate_closed(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("'closed' must be a string.")

    if value not in {"both", "left", "right", "none"}:
        raise ValueError("'closed' must be one of 'both', 'left', 'right', or 'none'.")


def _validate_target_set(value: sp.Set | tuple | list) -> None:
    if isinstance(value, sp.Set):
        return

    if isinstance(value, (tuple, list)) and len(value) == 2:
        return

    raise TypeError("'target_set' must be a SymPy set or a two-element tuple or list.")


def _validate_function(value: Callable) -> None:
    if not callable(value):
        raise TypeError("'function' must be callable.")


def _validate_value_set(value: ValueSet) -> None:
    if not isinstance(value, ValueSet):
        raise TypeError("'value_set' must be a ValueSet.")


def _validate_function_name(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("'function_name' must be a string.")


def _validate_vectorized(value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError("'vectorized' must be a bool.")


def _validate_others(value: tuple) -> None:
    from problab.random_variables.base import RandomVariable

    if not all(isinstance(other, RandomVariable) for other in value):
        raise TypeError("'others' must contain only RandomVariable instances.")
