from numbers import Real

import numpy as np

from src.problab.random_variables.base import RandomVariable
from src.problab.functions._utils import _apply_scalar_or_rv
from src.problab.value_sets.sets import POSITIVE_REALS, REALS
from src.problab.validation.functions._common import _validate_domain, _validate_real_valued


def exp(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.exp,
        value_set=POSITIVE_REALS,
    )


def log(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_domain(
        x=x,
        domain=POSITIVE_REALS.sympy_set,
    )

    return _apply_scalar_or_rv(
        x=x,
        function=np.log,
        value_set=REALS,
    )


def log2(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_domain(
        x=x,
        domain=POSITIVE_REALS.sympy_set,
    )

    return _apply_scalar_or_rv(
        x=x,
        function=np.log2,
        value_set=REALS,
    )


def log10(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_domain(
        x=x,
        domain=POSITIVE_REALS.sympy_set,
    )

    return _apply_scalar_or_rv(
        x=x,
        function=np.log10,
        value_set=REALS,
    )
