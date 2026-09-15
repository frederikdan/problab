from typing import Callable

from numbers import Real

from problab.random_variables.base import RandomVariable
from problab.value_sets.numeric_value_set import NumericValueSet


def _apply_scalar_or_rv(
        x: RandomVariable | Real,
        function: Callable,
        value_set: NumericValueSet,
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
        value_set: NumericValueSet,
) -> RandomVariable | float:

    if isinstance(x, RandomVariable):
        return x.apply(
            function=function,
            value_set=value_set,
            vectorized=True,
        )

    return float(function(x))
