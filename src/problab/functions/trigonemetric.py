from numbers import Real

import numpy as np
import sympy as sp

from src.problab.functions._utils import _require_real_valued, _apply, _require_domain
from src.problab.random_variables.base import RandomVariable
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import REALS, NON_NEGATIVE_REALS


def sin(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.sin,
        value_set=ValueSet(sp.Interval(-1, 1), (np.floating,)),
    )


def cos(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.cos,
        value_set=ValueSet(sp.Interval(-1, 1), (np.floating,)),
    )


def tan(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.tan,
        value_set=REALS,
    )


def arcsin(x: RandomVariable | Real) -> RandomVariable | float:
    _require_domain(
        x=x,
        domain=sp.Interval(-1, 1),
    )

    return _apply(
        x=x,
        function=np.arcsin,
        value_set=ValueSet(sp.Interval(-sp.pi / 2, sp.pi / 2), (np.floating,)),
    )


def arccos(x: RandomVariable | Real) -> RandomVariable | float:
    _require_domain(
        x=x,
        domain=sp.Interval(-1, 1),
    )

    return _apply(
        x=x,
        function=np.arccos,
        value_set=ValueSet(sp.Interval(0, sp.pi), (np.floating,)),
    )


def arctan(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.arctan,
        value_set=ValueSet(sp.Interval.open(-sp.pi / 2, sp.pi / 2), (np.floating,)),
    )


def sinh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.sinh,
        value_set=REALS,
    )


def cosh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.cosh,
        value_set=ValueSet(sp.Interval(1, sp.oo), (np.floating,)),
    )


def tanh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.tanh,
        value_set=ValueSet(sp.Interval.open(-1, 1), (np.floating,)),
    )


def arcsinh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.arcsinh,
        value_set=REALS,
    )


def arccosh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_domain(
        x=x,
        domain=sp.Interval(1, sp.oo),
    )

    return _apply(
        x=x,
        function=np.arccosh,
        value_set=NON_NEGATIVE_REALS,
    )


def arctanh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_domain(
        x=x,
        domain=sp.Interval.open(-1, 1),
    )

    return _apply(
        x=x,
        function=np.arctanh,
        value_set=REALS,
    )