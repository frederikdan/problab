from numbers import Real
from typing import Literal

import numpy as np

from problab.random_variables.nodes import _ConstantNode
from problab.validation.distributions._base import _require_supported_parameter_realization_dtypes, \
    _validate_parameter_realization_value_set
from problab.value_sets._utils import is_known_subset
from problab.value_sets.sets import POSITIVE_REALS, REALS


def _validate_normal_mean(
    value,
    *,
    instance,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    from problab.random_variables.base import RandomVariable

    if isinstance(value, RandomVariable):
        mean_node = value._node
    elif isinstance(value, Real):
        mean_node = _ConstantNode(value)
    else:
        raise TypeError("'mean' must be a RandomVariable or a real number.")

    if not is_known_subset(mean_node.value_set, REALS):
        raise ValueError("'mean' must contain only real values.")

    _require_supported_parameter_realization_dtypes(
        (mean_node,),
        distribution_name=instance.symbol,
        supported_input_types=((np.integer, np.floating),),
    )

    _validate_parameter_realization_value_set(
        mean_node,
        parameter_name="mean",
        distribution_name=instance.symbol,
        valid_value_set=REALS,
        parameter_risk_policy=parameter_risk_policy,
    )


def _validate_normal_std(
    value,
    *,
    instance,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    from problab.random_variables.base import RandomVariable

    if isinstance(value, RandomVariable):
        std_node = value._node
    elif isinstance(value, Real):
        std_node = _ConstantNode(value)
    else:
        raise TypeError("'std' must be a RandomVariable or a real number.")

    if not is_known_subset(std_node.value_set, POSITIVE_REALS):
        raise ValueError("'std' must contain only positive real values.")

    _require_supported_parameter_realization_dtypes(
        (std_node,),
        distribution_name=instance.symbol,
        supported_input_types=((np.integer, np.floating),),
    )

    _validate_parameter_realization_value_set(
        std_node,
        parameter_name="std",
        distribution_name=instance.symbol,
        valid_value_set=POSITIVE_REALS,
        parameter_risk_policy=parameter_risk_policy,
    )
