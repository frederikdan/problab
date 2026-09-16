import numpy as np
import sympy as sp

from numbers import Real

from problab.random_variables.base import RandomVariable
from problab.functions._utils import _apply_scalar_or_rv
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.validation.functions._common import _validate_domain, _validate_real_valued

from problab.value_sets.sets import NON_NEGATIVE_REALS, INTEGERS


def sqrt(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_domain(
        x=x,
        domain=NON_NEGATIVE_REALS.sympy_set,
    )

    return _apply_scalar_or_rv(
        x=x,
        function=np.sqrt,
        value_set=NON_NEGATIVE_REALS,
    )


def absolute(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.abs,
        value_set=NON_NEGATIVE_REALS,
    )


def floor(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.floor,
        value_set=INTEGERS,
    )


def ceil(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.ceil,
        value_set=INTEGERS,
    )


def sign(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.sign,
        value_set=HomogeneousNumericValueSet(
            sympy_set=sp.FiniteSet(-1, 0, 1),
            dtype_types=(np.integer, np.floating),
        ),
    )
