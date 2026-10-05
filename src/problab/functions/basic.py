from numbers import Real

import numpy as np
import sympy as sp

from problab.operations._function import (
    _SQRT,
    _FLOOR,
    _CEIL,
    _SIGN,
    _HYPOT,
)
from problab.operations._arithmetic import _ABS
from problab.random_variables.base import RandomVariable
from problab.functions._utils import _apply
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.validation.functions._common import _validate_domain, _validate_real_valued

from problab.value_sets.sets import NON_NEGATIVE_REALS, INTEGERS


def sqrt(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_domain(
        x=x,
        domain=NON_NEGATIVE_REALS.sympy_set,
    )

    return _apply(
        x=x,
        operation=_SQRT,
        mathematical_value_set=NON_NEGATIVE_REALS,
        realization_value_set=NON_NEGATIVE_REALS,
    )


def absolute(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_ABS,
        mathematical_value_set=NON_NEGATIVE_REALS,
        realization_value_set=NON_NEGATIVE_REALS,
    )


def floor(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_FLOOR,
        mathematical_value_set=INTEGERS,
        realization_value_set=INTEGERS,
    )


def ceil(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_CEIL,
        mathematical_value_set=INTEGERS,
        realization_value_set=INTEGERS,
    )


def sign(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_SIGN,
        mathematical_value_set=HomogeneousNumericValueSet(
            sympy_set=sp.FiniteSet(-1, 0, 1),
            dtype_types=(np.integer, np.floating),
        ),
        realization_value_set=HomogeneousNumericValueSet(
            sympy_set=sp.FiniteSet(-1, 0, 1),
            dtype_types=(np.integer, np.floating),
        ),
    )


def hypot(
    x: RandomVariable | Real,
    y: RandomVariable | Real,
) -> RandomVariable | Real:
    _validate_real_valued(x)
    _validate_real_valued(y)

    return _apply(x, _HYPOT, NON_NEGATIVE_REALS, NON_NEGATIVE_REALS, y)
