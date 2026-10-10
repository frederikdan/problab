from collections.abc import Iterable
from numbers import Real
from typing import Any
import numpy as np

from problab._utils import _float_if_fraction, _is_close


def _validate_categories(value) -> None:
    if not isinstance(value, Iterable):
        raise TypeError("'categories' must be an iterable.")


def _validate_probabilities(value) -> None:
    if not isinstance(value, Iterable):
        raise TypeError("'probabilities' must be an iterable.")


def _validate_categorical_configuration(
    categories: tuple[Any, ...],
    probabilities: tuple[float, ...],
) -> None:
    if len(categories) == 0:
        raise ValueError("'categories' must contain at least one value.")

    if len(categories) != len(probabilities):
        raise ValueError("'categories' and 'probabilities' must have the same length.")

    if any(
        isinstance(probability, (bool, np.bool_))
        for probability in probabilities
    ):
        raise TypeError("'probabilities' must not contain booleans.")

    if not all(isinstance(probability, Real) for probability in probabilities):
        raise TypeError("'probabilities' must contain only real numbers.")

    numeric_probabilities = tuple(
        _float_if_fraction(probability)
        for probability in probabilities
    )

    if not all(np.isfinite(probability) for probability in numeric_probabilities):
        raise ValueError("'probabilities' must be finite.")

    if any(probability < 0 for probability in probabilities):
        raise ValueError("'probabilities' cannot contain negative values.")

    if not _is_close(sum(numeric_probabilities), 1.0):
        raise ValueError("'probabilities' must sum to 1.")
