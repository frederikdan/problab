import numpy as np
import sympy as sp

from numbers import Real

from src.problab.random_variables.base import RandomVariable
from src.problab.functions._utils import (
    _apply_scalar_or_rv,
    _require_domain,
    _require_real_valued,
)
from src.problab.value_sets.base import ValueSet

from src.problab.value_sets.sets import NON_NEGATIVE_REALS, INTEGERS


def sqrt(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_domain(
        x=x,
        domain=NON_NEGATIVE_REALS.sympy_set,
    )

    return _apply_scalar_or_rv(
        x=x,
        function=np.sqrt,
        value_set=NON_NEGATIVE_REALS,
    )


def absolute(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.abs,
        value_set=NON_NEGATIVE_REALS,
    )


def floor(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.floor,
        value_set=INTEGERS,
    )


def ceil(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.ceil,
        value_set=INTEGERS,
    )


def sign(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.sign,
        value_set=ValueSet(
            sympy_set=sp.FiniteSet(-1, 0, 1),
            dtype_types=(np.integer, np.floating),
        ),
    )