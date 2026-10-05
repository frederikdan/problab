from __future__ import annotations

import numpy as np
from numbers import Real
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from problab.random_variables.nodes.base import _Node


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


def _require_supported_operation_inputs(input_nodes: tuple[_Node, ...],
                                        *,
                                        operation_name: str,
                                        supported_input_types: tuple[tuple[type[np.generic], ...], ...] | None,
                                        ) -> None:

    if supported_input_types is None:
        return

    if len(supported_input_types) == 1:
        supported_input_types *= len(input_nodes)
    elif len(supported_input_types) != len(input_nodes):
        raise ValueError("The number of input dtype rules does not match the inputs.")

    for node, allowed_types in zip(input_nodes, supported_input_types):
        dtype_types = node._realization_value_set.dtype_types

        if dtype_types is None or any(
            not any(
                issubclass(dtype_type, allowed_type)
                for allowed_type in allowed_types
            )
            for dtype_type in dtype_types
        ):
            raise TypeError(f"Cannot apply {operation_name}: input {node.name!r} may sample to an unsupported dtype.")


def _validate_numerical_error_policy(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("'numerical_error_policy' must be a string.")

    if value not in ("warn", "raise", "ignore"):
        raise ValueError("'numerical_error_policy' must be 'warn', 'raise', or 'ignore'.")