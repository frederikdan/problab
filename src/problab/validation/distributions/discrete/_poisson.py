from numbers import Real
from typing import Literal

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from problab.random_variables.nodes import _ConstantNode
from problab.validation.distributions._base import (
    _require_supported_parameter_realization_dtypes,
    _validate_parameter_realization_value_set,
)
from problab.value_sets._utils import is_known_subset
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet


# NumPy Generator.poisson reserves ten standard deviations below int64.max.
_POISSON_MU_MAX = float(np.iinfo(np.int64).max) - 10 * np.sqrt(float(np.iinfo(np.int64).max))
_POISSON_MU_REALIZATION_SET = HomogeneousNumericValueSet(
    sympy_set=sp.Interval(0, sp.Rational(float(_POISSON_MU_MAX))),
    dtype_types=(np.integer, np.floating),
)


def _validate_poisson_mu(
    value,
    *,
    instance,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    from problab.random_variables.base import RandomVariable

    valid_set = instance._valid_parameter_sets["mu"]

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("'mu' must be a RandomVariable or a real number, not a boolean.")

    if isinstance(value, RandomVariable):
        mu_node = value._node
    elif isinstance(value, Real):
        mu_node = _ConstantNode(value)
    else:
        raise TypeError("'mu' must be a RandomVariable or a real number.")

    if not is_known_subset(mu_node.value_set, valid_set):
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
        valid_value_set=_POISSON_MU_REALIZATION_SET,
        parameter_risk_policy=parameter_risk_policy,
    )


def _validate_poisson_mu_realizations(values: np.ndarray) -> NDArray[np.bool_]:
    limit = int(_POISSON_MU_MAX) if values.dtype.kind in "iu" else _POISSON_MU_MAX
    return np.isfinite(values) & (values >= 0) & (values <= limit)
