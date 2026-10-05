from numbers import Real
from typing import Literal

import numpy as np

from problab.random_variables.nodes import _ConstantNode
from problab.validation.distributions._base import (
    _require_supported_parameter_realization_dtypes,
    _validate_parameter_realization_value_set,
)
from problab.value_sets._utils import is_known_subset
from problab.value_sets.sets import NATURALS_0, UNIT_INTERVAL


def _validate_binomial_n(
    value,
    *,
    instance,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    from problab.random_variables.base import RandomVariable

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("'n' must be a RandomVariable or an integer, not a boolean.")

    if isinstance(value, RandomVariable):
        n_node = value._node
    elif isinstance(value, (int, np.integer)):
        n_node = _ConstantNode(value)
    else:
        raise TypeError("'n' must be a RandomVariable or an integer.")

    if not is_known_subset(n_node.value_set, NATURALS_0):
        raise ValueError("'n' must be a positive integer or 0.")

    _require_supported_parameter_realization_dtypes(
        (n_node,),
        distribution_name=instance.symbol,
        supported_input_types=((np.integer, np.floating),),
    )

    _validate_parameter_realization_value_set(
        n_node,
        parameter_name="n",
        distribution_name=instance.symbol,
        valid_value_set=NATURALS_0,
        parameter_risk_policy=parameter_risk_policy,
    )


def _validate_binomial_p(
    value,
    *,
    instance,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    from problab.random_variables.base import RandomVariable

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("'p' must be a RandomVariable or a real number, not a boolean.")

    if isinstance(value, RandomVariable):
        p_node = value._node
    elif isinstance(value, Real):
        p_node = _ConstantNode(value)
    else:
        raise TypeError("'p' must be a RandomVariable or a real number.")

    if not is_known_subset(p_node.value_set, UNIT_INTERVAL):
        raise ValueError("'p' must be in the interval [0, 1].")

    _require_supported_parameter_realization_dtypes(
        (p_node,),
        distribution_name=instance.symbol,
        supported_input_types=((np.integer, np.floating),),
    )

    _validate_parameter_realization_value_set(
        p_node,
        parameter_name="p",
        distribution_name=instance.symbol,
        valid_value_set=UNIT_INTERVAL,
        parameter_risk_policy=parameter_risk_policy,
    )
