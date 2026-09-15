import numpy as np
import sympy as sp

from src.problab.value_sets.base import _UnknownValueSet, ValueSet

n = sp.Symbol("n", integer=True)

_INTEGER_DTYPES = (np.integer, np.floating)
_REAL_DTYPES = (np.integer, np.floating)
_COMPLEX_DTYPES = (np.integer, np.floating, np.complexfloating)

UNKNOWN_VALUE_SET        = ValueSet(sympy_set=_UnknownValueSet(),                                                             dtype_types=None)
BOOLEANS                 = ValueSet(sympy_set=sp.FiniteSet(False, True),                                                dtype_types=(np.bool_,))
REALS                    = ValueSet(sympy_set=sp.S.Reals,                                                                     dtype_types=_REAL_DTYPES)
POSITIVE_REALS           = ValueSet(sympy_set=sp.Interval.open(0, sp.oo),                                                  dtype_types=_REAL_DTYPES)
NEGATIVE_REALS           = ValueSet(sympy_set=sp.Interval.open(-sp.oo, 0),                                                 dtype_types=_REAL_DTYPES)
NON_NEGATIVE_REALS       = ValueSet(sympy_set=sp.Interval(0, sp.oo),                                                     dtype_types=_REAL_DTYPES)
NON_POSITIVE_REALS       = ValueSet(sympy_set=sp.Interval(-sp.oo, 0),                                                    dtype_types=_REAL_DTYPES)
INTEGERS                 = ValueSet(sympy_set=sp.S.Integers,                                                                  dtype_types=_INTEGER_DTYPES)
POSITIVE_INTEGERS        = ValueSet(sympy_set=sp.S.Naturals,                                                                  dtype_types=_INTEGER_DTYPES)
NEGATIVE_INTEGERS        = ValueSet(sympy_set=sp.Intersection(sp.S.Integers, sp.Interval.open(-sp.oo, 0)),           dtype_types=_INTEGER_DTYPES)
NATURALS_0               = ValueSet(sympy_set=sp.S.Naturals0,                                                                 dtype_types=_INTEGER_DTYPES)
COMPLEXES                = ValueSet(sympy_set=sp.S.Complexes,                                                                 dtype_types=_COMPLEX_DTYPES)
UNIT_INTERVAL            = ValueSet(sympy_set=sp.Interval(0, 1),                                                    dtype_types=_REAL_DTYPES)
ZERO                     = ValueSet(sympy_set=sp.FiniteSet(0),                                                                dtype_types=_COMPLEX_DTYPES)
ONE                      = ValueSet(sympy_set=sp.FiniteSet(1),                                                                dtype_types=_COMPLEX_DTYPES)
NON_ZERO_REALS           = ValueSet(sympy_set=REALS.sympy_set - ZERO.sympy_set,                                               dtype_types=_REAL_DTYPES)
NON_ZERO_COMPLEXES       = ValueSet(sympy_set=COMPLEXES.sympy_set - ZERO.sympy_set,                                           dtype_types=_COMPLEX_DTYPES)
NON_INTEGER_REALS        = ValueSet(sympy_set=REALS.sympy_set - INTEGERS.sympy_set,                                           dtype_types=(np.floating,))
EVEN_INTEGERS            = ValueSet(sympy_set=sp.ImageSet(sp.Lambda(n, 2 * n), INTEGERS.sympy_set),                     dtype_types=_INTEGER_DTYPES)
ODD_INTEGERS             = ValueSet(sympy_set=sp.ImageSet(sp.Lambda(n, 2 * n + 1), INTEGERS.sympy_set),                 dtype_types=_INTEGER_DTYPES)
POSITIVE_EVEN_INTEGERS   = ValueSet(sympy_set=sp.Intersection(EVEN_INTEGERS.sympy_set, POSITIVE_INTEGERS.sympy_set),    dtype_types=_INTEGER_DTYPES)
NEGATIVE_EVEN_INTEGERS   = ValueSet(sympy_set=sp.Intersection(EVEN_INTEGERS.sympy_set, NEGATIVE_INTEGERS.sympy_set),    dtype_types=_INTEGER_DTYPES)
POSITIVE_ODD_INTEGERS    = ValueSet(sympy_set=sp.Intersection(ODD_INTEGERS.sympy_set, POSITIVE_INTEGERS.sympy_set),     dtype_types=_INTEGER_DTYPES)
NEGATIVE_ODD_INTEGERS    = ValueSet(sympy_set=sp.Intersection(ODD_INTEGERS.sympy_set, NEGATIVE_INTEGERS.sympy_set),     dtype_types=_INTEGER_DTYPES)
NON_POSITIVE_INTEGERS    = ValueSet(sympy_set=sp.Intersection(INTEGERS.sympy_set, NON_POSITIVE_REALS.sympy_set),        dtype_types=_INTEGER_DTYPES)
NATURALS                 = POSITIVE_INTEGERS