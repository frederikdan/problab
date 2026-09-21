from numbers import Real

import numpy as np
import sympy as sp

from problab._operations import (
    _SIN,
    _ARCSIN,
    _COS,
    _ARCCOS,
    _TAN,
    _ARCTAN,
    _SINH,
    _COSH,
    _TANH,
    _ARCSINH,
    _ARCCOSH,
    _ARCTANH,
)
from problab.functions._utils import _apply
from problab.random_variables.base import RandomVariable
from problab.validation.functions._common import _validate_domain, _validate_real_valued
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.sets import REALS, NON_NEGATIVE_REALS


def sin(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_SIN,
        mathematical_value_set=HomogeneousNumericValueSet(sp.Interval(-1, 1), (np.floating,)),
        realization_value_set=HomogeneousNumericValueSet(sp.Interval(-1, 1), (np.floating,)),
    )


def cos(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_COS,
        mathematical_value_set=HomogeneousNumericValueSet(sp.Interval(-1, 1), (np.floating,)),
        realization_value_set=HomogeneousNumericValueSet(sp.Interval(-1, 1), (np.floating,)),
    )


def tan(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_TAN,
        mathematical_value_set=REALS,
        realization_value_set=REALS,
    )


def arcsin(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_domain(
        x=x,
        domain=sp.Interval(-1, 1),
    )

    return _apply(
        x=x,
        operation=_ARCSIN,
        mathematical_value_set=HomogeneousNumericValueSet(sp.Interval(-sp.pi / 2, sp.pi / 2), (np.floating,)),
        realization_value_set=HomogeneousNumericValueSet(sp.Interval(-sp.pi / 2, sp.pi / 2), (np.floating,)),
    )


def arccos(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_domain(
        x=x,
        domain=sp.Interval(-1, 1),
    )

    return _apply(
        x=x,
        operation=_ARCCOS,
        mathematical_value_set=HomogeneousNumericValueSet(sp.Interval(0, sp.pi), (np.floating,)),
        realization_value_set=HomogeneousNumericValueSet(sp.Interval(0, sp.pi), (np.floating,)),
    )


def arctan(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_ARCTAN,
        mathematical_value_set=HomogeneousNumericValueSet(
            sp.Interval.open(-sp.pi / 2, sp.pi / 2),
            (np.floating,),
        ),
        realization_value_set=HomogeneousNumericValueSet(
            sp.Interval(-sp.pi / 2, sp.pi / 2),
            (np.floating,),
        ),
    )


def sinh(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_SINH,
        mathematical_value_set=REALS,
        realization_value_set=REALS,
    )


def cosh(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_COSH,
        mathematical_value_set=HomogeneousNumericValueSet(sp.Interval(1, sp.oo), (np.floating,)),
        realization_value_set=HomogeneousNumericValueSet(sp.Interval(1, sp.oo), (np.floating,)),
    )


def tanh(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_TANH,
        mathematical_value_set=HomogeneousNumericValueSet(
            sp.Interval.open(-1, 1),
            (np.floating,),
        ),
        realization_value_set=HomogeneousNumericValueSet(
            sp.Interval(-1, 1),
            (np.floating,),
        ),
    )


def arcsinh(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_ARCSINH,
        mathematical_value_set=REALS,
        realization_value_set=REALS,
    )


def arccosh(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_domain(
        x=x,
        domain=sp.Interval(1, sp.oo),
    )

    return _apply(
        x=x,
        operation=_ARCCOSH,
        mathematical_value_set=NON_NEGATIVE_REALS,
        realization_value_set=NON_NEGATIVE_REALS,
    )


def arctanh(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_domain(
        x=x,
        domain=sp.Interval.open(-1, 1),
    )

    return _apply(
        x=x,
        operation=_ARCTANH,
        mathematical_value_set=REALS,
        realization_value_set=REALS,
    )
