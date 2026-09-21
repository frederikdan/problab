from numbers import Real

from problab._operations import _EXP, _LOG, _LOG2, _LOG10, _LOG1P, _EXPM1, _LOGADDEXP
from problab.random_variables.base import RandomVariable
from problab.functions._utils import _apply
from problab.value_sets.sets import (
    GE_NEG_ONE_REALS,
    GT_NEG_ONE_REALS,
    NON_NEGATIVE_REALS,
    POSITIVE_REALS,
    REALS,
)
from problab.validation.functions._common import _validate_domain, _validate_real_valued


def exp(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_EXP,
        mathematical_value_set=POSITIVE_REALS,
        realization_value_set=NON_NEGATIVE_REALS,
    )


def log(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_domain(
        x=x,
        domain=POSITIVE_REALS.sympy_set,
    )

    return _apply(
        x=x,
        operation=_LOG,
        mathematical_value_set=REALS,
        realization_value_set=REALS,
    )


def log2(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_domain(
        x=x,
        domain=POSITIVE_REALS.sympy_set,
    )

    return _apply(
        x=x,
        operation=_LOG2,
        mathematical_value_set=REALS,
        realization_value_set=REALS,
    )


def log10(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_domain(
        x=x,
        domain=POSITIVE_REALS.sympy_set,
    )

    return _apply(
        x=x,
        operation=_LOG10,
        mathematical_value_set=REALS,
        realization_value_set=REALS,
    )


def log1p(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_domain(
        x=x,
        domain=GT_NEG_ONE_REALS.sympy_set,
    )

    return _apply(
        x=x,
        operation=_LOG1P,
        mathematical_value_set=REALS,
        realization_value_set=REALS,
    )


def expm1(x: RandomVariable | Real) -> RandomVariable | Real:
    _validate_real_valued(x)

    return _apply(
        x=x,
        operation=_EXPM1,
        mathematical_value_set=GT_NEG_ONE_REALS,
        realization_value_set=GE_NEG_ONE_REALS,
    )


def logaddexp(
    x: RandomVariable | Real,
    y: RandomVariable | Real,
) -> RandomVariable | Real:
    _validate_real_valued(x)
    _validate_real_valued(y)

    return _apply(x, _LOGADDEXP, REALS, REALS, y)
