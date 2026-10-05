from numbers import Real

import numpy as np
import sympy as sp

from problab.operations._function import (
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
from problab.random_variables.nodes import _ConstantNode
from problab.validation.functions._common import _validate_domain, _validate_real_valued
from problab.value_sets._inference import _infer_dtype_types
from problab.value_sets._realization_inference import _infer_rounded_interval_value_set
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

    input_node = (
        x._node
        if isinstance(x, RandomVariable)
        else _ConstantNode(x)
    )

    output_dtype_types = _infer_dtype_types(
        _ARCSIN.operation,
        (input_node._realization_value_set,),
    )

    if output_dtype_types is None:
        raise TypeError("Cannot determine the output dtype for arcsin().")

    mathematical_interval = sp.Interval(-sp.pi / 2, sp.pi / 2)

    return _apply(
        x=x,
        operation=_ARCSIN,
        mathematical_value_set=HomogeneousNumericValueSet(
            sympy_set=mathematical_interval,
            dtype_types=(np.floating,),
        ),
        realization_value_set=_infer_rounded_interval_value_set(
            mathematical_interval,
            output_dtype_types,
        ),
    )


def arccos(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_domain(
        x=x,
        domain=sp.Interval(-1, 1),
    )

    input_node = (
        x._node
        if isinstance(x, RandomVariable)
        else _ConstantNode(x)
    )

    output_dtype_types = _infer_dtype_types(
        _ARCCOS.operation,
        (input_node._realization_value_set,),
    )

    if output_dtype_types is None:
        raise TypeError("Cannot determine the output dtype for arccos().")

    mathematical_interval = sp.Interval(0, sp.pi)

    return _apply(
        x=x,
        operation=_ARCCOS,
        mathematical_value_set=HomogeneousNumericValueSet(
            sympy_set=mathematical_interval,
            dtype_types=(np.floating,),
        ),
        realization_value_set=_infer_rounded_interval_value_set(
            mathematical_interval,
            output_dtype_types,
        ),
    )


def arctan(x: RandomVariable | Real) -> RandomVariable | float:
    _validate_real_valued(x)

    input_node = (
        x._node
        if isinstance(x, RandomVariable)
        else _ConstantNode(x)
    )

    output_dtype_types = _infer_dtype_types(
        _ARCTAN.operation,
        (input_node._realization_value_set,),
    )

    if output_dtype_types is None:
        raise TypeError("Cannot determine the output dtype for arctan().")

    mathematical_interval = sp.Interval.open(-sp.pi / 2, sp.pi / 2)

    return _apply(
        x=x,
        operation=_ARCTAN,
        mathematical_value_set=HomogeneousNumericValueSet(
            sympy_set=mathematical_interval,
            dtype_types=(np.floating,),
        ),
        realization_value_set=_infer_rounded_interval_value_set(
            mathematical_interval,
            output_dtype_types,
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
