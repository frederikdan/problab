from numbers import Real

import numpy as np
import sympy as sp

from src.problab.random_variables.base import RandomVariable
from src.problab.value_sets._utils import is_known_subset


def _validate_real_valued(x: RandomVariable | Real) -> None:
    if isinstance(x, RandomVariable):
        if not is_known_subset(x._node.value_set, sp.S.Reals):
            raise ValueError("'x' must contain only real values.")
    elif isinstance(x, (bool, np.bool_)) or not isinstance(x, Real):
        raise TypeError("'x' must be a RandomVariable or Real.")


def _validate_domain(x: RandomVariable | Real, domain: sp.Set) -> None:
    _validate_real_valued(x)

    if isinstance(x, RandomVariable):
        if not is_known_subset(x._node.value_set, domain):
            raise ValueError(f"'x' must contain only values in {domain}.")
    elif x not in domain:
        raise ValueError(f"'x' must be in {domain}.")
