import numpy as np
import sympy as sp

from problab.value_sets._unknown import _UnknownValueSet
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet

n = sp.Symbol("n", integer=True)

_INTEGER_DTYPES = (np.integer, np.floating)
_REAL_DTYPES = (np.integer, np.floating)
_COMPLEX_DTYPES = (np.integer, np.floating, np.complexfloating)

UNKNOWN_VALUE_SET        = HomogeneousNumericValueSet(sympy_set=_UnknownValueSet(), dtype_types=None)
BOOLEANS                 = HomogeneousNumericValueSet(sympy_set=sp.FiniteSet(False, True), dtype_types=(np.bool_,))
REALS                    = HomogeneousNumericValueSet(sympy_set=sp.S.Reals, dtype_types=_REAL_DTYPES)
POSITIVE_REALS           = HomogeneousNumericValueSet(sympy_set=sp.Interval.open(0, sp.oo), dtype_types=_REAL_DTYPES)
NEGATIVE_REALS           = HomogeneousNumericValueSet(sympy_set=sp.Interval.open(-sp.oo, 0), dtype_types=_REAL_DTYPES)
NON_NEGATIVE_REALS       = HomogeneousNumericValueSet(sympy_set=sp.Interval(0, sp.oo), dtype_types=_REAL_DTYPES)
NON_POSITIVE_REALS       = HomogeneousNumericValueSet(sympy_set=sp.Interval(-sp.oo, 0), dtype_types=_REAL_DTYPES)
INTEGERS                 = HomogeneousNumericValueSet(sympy_set=sp.S.Integers, dtype_types=_INTEGER_DTYPES)
POSITIVE_INTEGERS        = HomogeneousNumericValueSet(sympy_set=sp.S.Naturals, dtype_types=_INTEGER_DTYPES)
NEGATIVE_INTEGERS        = HomogeneousNumericValueSet(sympy_set=sp.Intersection(sp.S.Integers, sp.Interval.open(-sp.oo, 0)), dtype_types=_INTEGER_DTYPES)
NATURALS_0               = HomogeneousNumericValueSet(sympy_set=sp.S.Naturals0, dtype_types=_INTEGER_DTYPES)
COMPLEXES                = HomogeneousNumericValueSet(sympy_set=sp.S.Complexes, dtype_types=_COMPLEX_DTYPES)
UNIT_INTERVAL            = HomogeneousNumericValueSet(sympy_set=sp.Interval(0, 1), dtype_types=_REAL_DTYPES)
ZERO                     = HomogeneousNumericValueSet(sympy_set=sp.FiniteSet(0), dtype_types=_COMPLEX_DTYPES)
ONE                      = HomogeneousNumericValueSet(sympy_set=sp.FiniteSet(1), dtype_types=_COMPLEX_DTYPES)
NON_ZERO_REALS           = HomogeneousNumericValueSet(sympy_set=REALS.sympy_set - ZERO.sympy_set, dtype_types=_REAL_DTYPES)
NON_ZERO_COMPLEXES       = HomogeneousNumericValueSet(sympy_set=COMPLEXES.sympy_set - ZERO.sympy_set, dtype_types=_COMPLEX_DTYPES)
NON_INTEGER_REALS        = HomogeneousNumericValueSet(sympy_set=REALS.sympy_set - INTEGERS.sympy_set, dtype_types=(np.floating,))
EVEN_INTEGERS            = HomogeneousNumericValueSet(sympy_set=sp.ImageSet(sp.Lambda(n, 2 * n), INTEGERS.sympy_set), dtype_types=_INTEGER_DTYPES)
ODD_INTEGERS             = HomogeneousNumericValueSet(sympy_set=sp.ImageSet(sp.Lambda(n, 2 * n + 1), INTEGERS.sympy_set), dtype_types=_INTEGER_DTYPES)
POSITIVE_EVEN_INTEGERS   = HomogeneousNumericValueSet(sympy_set=sp.Intersection(EVEN_INTEGERS.sympy_set, POSITIVE_INTEGERS.sympy_set), dtype_types=_INTEGER_DTYPES)
NEGATIVE_EVEN_INTEGERS   = HomogeneousNumericValueSet(sympy_set=sp.Intersection(EVEN_INTEGERS.sympy_set, NEGATIVE_INTEGERS.sympy_set), dtype_types=_INTEGER_DTYPES)
POSITIVE_ODD_INTEGERS    = HomogeneousNumericValueSet(sympy_set=sp.Intersection(ODD_INTEGERS.sympy_set, POSITIVE_INTEGERS.sympy_set), dtype_types=_INTEGER_DTYPES)
NEGATIVE_ODD_INTEGERS    = HomogeneousNumericValueSet(sympy_set=sp.Intersection(ODD_INTEGERS.sympy_set, NEGATIVE_INTEGERS.sympy_set), dtype_types=_INTEGER_DTYPES)
NON_POSITIVE_INTEGERS    = HomogeneousNumericValueSet(sympy_set=sp.Intersection(INTEGERS.sympy_set, NON_POSITIVE_REALS.sympy_set), dtype_types=_INTEGER_DTYPES)
NATURALS                 = POSITIVE_INTEGERS
