from typing import Callable

from numbers import Real

from problab.random_variables.base import RandomVariable
from problab.value_sets.base import ValueSet


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
