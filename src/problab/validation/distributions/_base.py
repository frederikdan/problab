import warnings
from typing import get_args, Literal

import numpy as np
from numbers import Real
from collections.abc import Mapping
import numpy as np

from problab._utils import _float_if_fraction
from problab.random_variables.nodes import _Node
from problab.statistics._quantiles import QuantileMethod
from problab.value_sets._utils import is_known_subset
from problab.value_sets.base import ValueSet, NumericValueSet


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

    values = np.asarray(_float_if_fraction(value))

    if np.any(np.isnan(values)):
        raise ValueError("'x' must not be NaN.")

def _validate_ppf_input(value: Real | np.ndarray) -> None:
    _validate_real_input(value, "q")

    values = np.asarray(value)

    if (
        np.any(np.isnan(np.asarray(_float_if_fraction(value))))
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("'q' must be between 0 and 1.")


def _validate_quantile_method(value: QuantileMethod) -> None:
    if not isinstance(value, str):
        raise TypeError("'quantile_method' must be a string.")

    if value not in get_args(QuantileMethod):
        raise ValueError(f"'quantile_method' must be one of {get_args(QuantileMethod)}.")


def _require_supported_parameter_realization_dtypes(input_nodes: tuple[_Node, ...],
                                                    *,
                                                    distribution_name: str,
                                                    supported_input_types: tuple[tuple[type[np.generic], ...], ...] | None
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
            raise TypeError(
                f"Cannot construct {distribution_name}: parameter input "
                f"{node.name!r} may sample to an unsupported dtype."
            )


def _validate_parameter_realization_value_set(
    node: _Node,
    *,
    parameter_name: str,
    distribution_name: str,
    valid_value_set: NumericValueSet,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    if parameter_risk_policy == "ignore":
        return

    realization_set = node._realization_value_set

    is_valid = (
        isinstance(realization_set, NumericValueSet)
        and is_known_subset(realization_set, valid_value_set)
        and (
            not realization_set.allows_positive_infinity
            or valid_value_set.allows_positive_infinity
        )
        and (
            not realization_set.allows_negative_infinity
            or valid_value_set.allows_negative_infinity
        )
        and (
            not realization_set.allows_nan
            or valid_value_set.allows_nan
        )
    )

    if is_valid:
        return

    message = (
        f"Cannot guarantee that parameter {parameter_name!r} "
        f"of {distribution_name} produces valid numerical values "
        f"within {valid_value_set.sympy_set}."
    )

    if parameter_risk_policy == "raise":
        raise ValueError(message)

    warnings.warn(message, RuntimeWarning, stacklevel=4)


def _validate_parameter_risk_policy(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("'parameter_risk_policy' must be a string.")

    if value not in ("warn", "raise", "ignore"):
        raise ValueError("'parameter_risk_policy' must be 'warn', 'raise', or 'ignore'.")
