from numbers import Real
from typing import Callable

import sympy as sp

from src.problab.random_variables.base import (
    RandomVariable,
    is_known_subset,
    ValueSet,
)



def _require_real_valued(x: RandomVariable | Real) -> None:

    if isinstance(x, RandomVariable):
        if not is_known_subset(x._node.value_set, sp.S.Reals):
            raise ValueError("'x' must contain only real values.")

    elif not isinstance(x, Real):
        raise TypeError("'x' must be a RandomVariable or Real.")


def _require_domain(
        x: RandomVariable | Real,
        domain: sp.Set
) -> None:

    _require_real_valued(x)

    if isinstance(x, RandomVariable):
        if not is_known_subset(x._node.value_set, domain):
            raise ValueError(f"'x' must contain only values in {domain}.")

    elif x not in domain:
        raise ValueError(f"'x' must be in {domain}.")


def _apply_scalar_or_rv(
        x: RandomVariable | Real,
        function: Callable,
        value_set: ValueSet,
) -> RandomVariable | Real:

    if isinstance(x, RandomVariable):
        return x.apply(
            function=function,
            value_set=value_set,
            vectorized=True,
        )

    return function(x)


def _apply(
        x: RandomVariable | Real,
        function,
        value_set: ValueSet,
) -> RandomVariable | float:

    if isinstance(x, RandomVariable):
        return x.apply(
            function=function,
            value_set=value_set,
            vectorized=True,
        )

    return float(function(x))