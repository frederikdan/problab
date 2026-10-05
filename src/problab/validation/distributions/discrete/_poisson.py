from numbers import Real
from typing import Literal

import numpy as np

from problab.random_variables.nodes import _ConstantNode
from problab.validation.distributions._base import (
    _require_supported_parameter_realization_dtypes,
    _validate_parameter_realization_value_set,
)
from problab.value_sets._utils import is_known_subset
from problab.value_sets.sets import NON_NEGATIVE_REALS


def _validate_poisson_mu(
    value,
    *,
    instance,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    from problab.random_variables.base import RandomVariable

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("'mu' must be a RandomVariable or a real number, not a boolean.")

    if isinstance(value, RandomVariable):
        mu_node = value._node
    elif isinstance(value, Real):
        mu_node = _ConstantNode(value)
    else:
        raise TypeError("'mu' must be a RandomVariable or a real number.")

    if not is_known_subset(mu_node.value_set, NON_NEGATIVE_REALS):
        raise ValueError("'mu' must be non-negative.")

    _require_supported_parameter_realization_dtypes(
        (mu_node,),
        distribution_name=instance.symbol,
        supported_input_types=((np.integer, np.floating),),
    )

    _validate_parameter_realization_value_set(
        mu_node,
        parameter_name="mu",
        distribution_name=instance.symbol,
        valid_value_set=NON_NEGATIVE_REALS,
        parameter_risk_policy=parameter_risk_policy,
    )
