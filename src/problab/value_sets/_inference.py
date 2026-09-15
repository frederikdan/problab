from functools import wraps
from itertools import product
from typing import Callable

import numpy as np
import sympy as sp

from problab.value_sets._utils import is_known_subset
from problab.value_sets.base import ValueSet, _UnknownValueSet
from problab.value_sets.sets import UNKNOWN_VALUE_SET, ZERO, ONE, POSITIVE_REALS, REALS, NON_NEGATIVE_REALS, \
    POSITIVE_EVEN_INTEGERS, NON_ZERO_REALS, NEGATIVE_EVEN_INTEGERS, POSITIVE_ODD_INTEGERS, NEGATIVE_ODD_INTEGERS, \
    NEGATIVE_REALS, NON_INTEGER_REALS, COMPLEXES, NON_ZERO_COMPLEXES, NON_POSITIVE_REALS, INTEGERS, NATURALS_0, \
    NEGATIVE_INTEGERS, POSITIVE_INTEGERS, NON_POSITIVE_INTEGERS

_BUILTIN_DTYPES = tuple(np.dtype(code) for code in np.typecodes["All"])


def _infer_dtype_types(operation: np.ufunc | None, inputs: tuple[ValueSet, ...]) -> tuple[type[np.generic], ...] | None:
    if operation is None:
        return None

    input_dtypes = []
    for value_set in inputs:
        if value_set.dtype_types is None:
            return None

        candidates = []
        for allowed in value_set.dtype_types:
            matches = [dtype for dtype in _BUILTIN_DTYPES if np.issubdtype(dtype, allowed)]
            if not matches:
                return None
            candidates.extend(matches)
        if not candidates:
            return None
        input_dtypes.append(candidates)

    output_types = []
    for combination in product(*input_dtypes):
        try:
            output_dtype = operation.resolve_dtypes((*combination, None))[-1]
        except (TypeError, ValueError):
            return None
        if output_dtype.type not in output_types:
            output_types.append(output_dtype.type)

    return tuple(output_types)


def _arithmetic_inference(*, dtype_operation: np.ufunc | None = None, preserves_integers: bool = False):
    def decorate(infer: Callable[..., ValueSet]) -> Callable[..., ValueSet]:
        @wraps(infer)
        def wrapped(*operands: ValueSet, **named_operands: ValueSet) -> ValueSet:
            inputs = (*operands, *named_operands.values())

            if not all(is_known_subset(value_set, COMPLEXES) for value_set in inputs):
                return UNKNOWN_VALUE_SET

            result = infer(*operands, **named_operands)
            if isinstance(result.sympy_set, _UnknownValueSet):
                return result

            sympy_set = result.sympy_set
            if preserves_integers and all(is_known_subset(value_set, INTEGERS) for value_set in inputs):
                sympy_set = sp.Intersection(sympy_set, INTEGERS.sympy_set)

            dtype_types = _infer_dtype_types(dtype_operation, inputs)
            return ValueSet(sympy_set=sympy_set, dtype_types=dtype_types)

        return wrapped

    return decorate


# Binary operations

@_arithmetic_inference(dtype_operation=np.add, preserves_integers=True)
def _infer_add_value_set(left_set: ValueSet, right_set: ValueSet) -> ValueSet:

    if left_set.sympy_set == ZERO.sympy_set:
        return right_set

    if right_set.sympy_set == ZERO.sympy_set:
        return left_set

    if (
        (is_known_subset(left_set, POSITIVE_REALS) and is_known_subset(right_set, NON_NEGATIVE_REALS))
        or (is_known_subset(left_set, NON_NEGATIVE_REALS) and is_known_subset(right_set, POSITIVE_REALS))
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(left_set, NON_NEGATIVE_REALS)
        and is_known_subset(right_set, NON_NEGATIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        (is_known_subset(left_set, NEGATIVE_REALS) and is_known_subset(right_set, NON_POSITIVE_REALS))
        or (is_known_subset(left_set, NON_POSITIVE_REALS) and is_known_subset(right_set, NEGATIVE_REALS))
    ):
        return NEGATIVE_REALS

    if (
        is_known_subset(left_set, NON_POSITIVE_REALS)
        and is_known_subset(right_set, NON_POSITIVE_REALS)
    ):
        return NON_POSITIVE_REALS

    if (
        is_known_subset(left_set, INTEGERS)
        and is_known_subset(right_set, INTEGERS)
    ):
        return INTEGERS

    if (
        is_known_subset(left_set, REALS)
        and is_known_subset(right_set, REALS)
    ):
        return REALS

    if (
        is_known_subset(left_set, COMPLEXES)
        and is_known_subset(right_set, COMPLEXES)
    ):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


@_arithmetic_inference(dtype_operation=np.subtract, preserves_integers=True)
def _infer_subtract_value_set(left_set: ValueSet, right_set: ValueSet) -> ValueSet:

    if right_set.sympy_set == ZERO.sympy_set:
        return left_set

    if (
        is_known_subset(left_set, POSITIVE_REALS)
        and is_known_subset(right_set, NEGATIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(left_set, NON_NEGATIVE_REALS)
        and is_known_subset(right_set, NON_POSITIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(left_set, NEGATIVE_REALS)
        and is_known_subset(right_set, POSITIVE_REALS)
    ):
        return NEGATIVE_REALS

    if (
        is_known_subset(left_set, NON_POSITIVE_REALS)
        and is_known_subset(right_set, NON_NEGATIVE_REALS)
    ):
        return NON_POSITIVE_REALS

    if (
        is_known_subset(left_set, INTEGERS)
        and is_known_subset(right_set, INTEGERS)
    ):
        return INTEGERS

    if (
        is_known_subset(left_set, REALS)
        and is_known_subset(right_set, REALS)
    ):
        return REALS

    if (
        is_known_subset(left_set, COMPLEXES)
        and is_known_subset(right_set, COMPLEXES)
    ):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


@_arithmetic_inference(dtype_operation=np.multiply, preserves_integers=True)
def _infer_multiply_value_set(left_set: ValueSet, right_set: ValueSet) -> ValueSet:

    if left_set.sympy_set == ZERO.sympy_set or right_set.sympy_set == ZERO.sympy_set:
        return ZERO

    if left_set.sympy_set == ONE.sympy_set:
        return right_set

    if right_set.sympy_set == ONE.sympy_set:
        return left_set

    if (
        is_known_subset(left_set, POSITIVE_REALS)
        and is_known_subset(right_set, POSITIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(left_set, NEGATIVE_REALS)
        and is_known_subset(right_set, NEGATIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(left_set, POSITIVE_REALS)
        and is_known_subset(right_set, NEGATIVE_REALS)
        or is_known_subset(left_set, NEGATIVE_REALS)
        and is_known_subset(right_set, POSITIVE_REALS)
    ):
        return NEGATIVE_REALS

    if (
        is_known_subset(left_set, NON_NEGATIVE_REALS)
        and is_known_subset(right_set, NON_NEGATIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(left_set, NON_POSITIVE_REALS)
        and is_known_subset(right_set, NON_POSITIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(left_set, INTEGERS)
        and is_known_subset(right_set, INTEGERS)
    ):
        return INTEGERS

    if (
        is_known_subset(left_set, NON_ZERO_REALS)
        and is_known_subset(right_set, NON_ZERO_REALS)
    ):
        return NON_ZERO_REALS

    if (
        is_known_subset(left_set, REALS)
        and is_known_subset(right_set, REALS)
    ):
        return REALS

    if (
        is_known_subset(left_set, NON_ZERO_COMPLEXES)
        and is_known_subset(right_set, NON_ZERO_COMPLEXES)
    ):
        return NON_ZERO_COMPLEXES

    if (
        is_known_subset(left_set, COMPLEXES)
        and is_known_subset(right_set, COMPLEXES)
    ):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


@_arithmetic_inference(dtype_operation=np.divide)
def _infer_divide_value_set(numerator_set: ValueSet, denominator_set: ValueSet) -> ValueSet:

    if not is_known_subset(denominator_set, NON_ZERO_COMPLEXES):
        return UNKNOWN_VALUE_SET

    if numerator_set.sympy_set == ZERO.sympy_set:
        return ZERO

    if denominator_set.sympy_set == ONE.sympy_set:
        return numerator_set

    if (
        is_known_subset(numerator_set, POSITIVE_REALS)
        and is_known_subset(denominator_set, POSITIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(numerator_set, NEGATIVE_REALS)
        and is_known_subset(denominator_set, NEGATIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(numerator_set, POSITIVE_REALS)
        and is_known_subset(denominator_set, NEGATIVE_REALS)
        or is_known_subset(numerator_set, NEGATIVE_REALS)
        and is_known_subset(denominator_set, POSITIVE_REALS)
    ):
        return NEGATIVE_REALS

    if (
        is_known_subset(numerator_set, NON_ZERO_REALS)
        and is_known_subset(denominator_set, NON_ZERO_REALS)
    ):
        return NON_ZERO_REALS

    if (
        is_known_subset(numerator_set, REALS)
        and is_known_subset(denominator_set, NON_ZERO_REALS)
    ):
        return REALS

    if (
        is_known_subset(numerator_set, NON_ZERO_COMPLEXES)
        and is_known_subset(denominator_set, NON_ZERO_COMPLEXES)
    ):
        return NON_ZERO_COMPLEXES

    if (
        is_known_subset(numerator_set, COMPLEXES)
        and is_known_subset(denominator_set, NON_ZERO_COMPLEXES)
    ):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


@_arithmetic_inference(dtype_operation=np.floor_divide)
def _infer_floor_divide_value_set(numerator_set: ValueSet, denominator_set: ValueSet) -> ValueSet:

    if not is_known_subset(numerator_set, REALS) or not is_known_subset(denominator_set, NON_ZERO_REALS):
        return UNKNOWN_VALUE_SET

    if numerator_set.sympy_set == ZERO.sympy_set:
        return ZERO

    if (
        is_known_subset(numerator_set, NON_NEGATIVE_REALS)
        and is_known_subset(denominator_set, POSITIVE_REALS)
    ):
        return NATURALS_0

    if (
        is_known_subset(numerator_set, NON_POSITIVE_REALS)
        and is_known_subset(denominator_set, NEGATIVE_REALS)
    ):
        return NATURALS_0

    if (
        is_known_subset(numerator_set, NEGATIVE_REALS)
        and is_known_subset(denominator_set, POSITIVE_REALS)
    ):
        return NEGATIVE_INTEGERS

    if (
        is_known_subset(numerator_set, POSITIVE_REALS)
        and is_known_subset(denominator_set, NEGATIVE_REALS)
    ):
        return NEGATIVE_INTEGERS

    if (
        is_known_subset(numerator_set, REALS)
        and is_known_subset(denominator_set, NON_ZERO_REALS)
    ):
        return INTEGERS

    return UNKNOWN_VALUE_SET


# _MODULO also applies np.where with a NaN branch, which can change dtype.
@_arithmetic_inference()
def _infer_modulo_value_set(dividend_set: ValueSet, divisor_set: ValueSet) -> ValueSet:

    if not is_known_subset(dividend_set, REALS) or not is_known_subset(divisor_set, NON_ZERO_REALS):
        return UNKNOWN_VALUE_SET

    if dividend_set.sympy_set == ZERO.sympy_set:
        return ZERO

    if (
        is_known_subset(dividend_set, INTEGERS)
        and is_known_subset(divisor_set, POSITIVE_INTEGERS)
    ):
        return NATURALS_0

    if (
        is_known_subset(dividend_set, INTEGERS)
        and is_known_subset(divisor_set, NEGATIVE_INTEGERS)
    ):
        return NON_POSITIVE_INTEGERS

    if (
        is_known_subset(dividend_set, REALS)
        and is_known_subset(divisor_set, POSITIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(dividend_set, REALS)
        and is_known_subset(divisor_set, NEGATIVE_REALS)
    ):
        return NON_POSITIVE_REALS

    return UNKNOWN_VALUE_SET


# _POWER/scimath performs casts depending on the realized signs of its inputs.
@_arithmetic_inference()
def _infer_power_value_set(base_set: ValueSet, exponent_set: ValueSet) -> ValueSet:

    # Match NumPy's convention, including 0 ** 0 == 1.
    if exponent_set.sympy_set == ZERO.sympy_set:
        return ONE

    if exponent_set.sympy_set == ONE.sympy_set:
        return base_set

    if base_set.sympy_set == ONE.sympy_set:
        return ONE

    base_is_nonzero = is_known_subset(base_set, NON_ZERO_COMPLEXES)
    exponent_is_nonnegative_integer = is_known_subset(exponent_set, NATURALS_0)
    exponent_is_positive_real = is_known_subset(exponent_set, POSITIVE_REALS)

    if not base_is_nonzero and not (exponent_is_nonnegative_integer or exponent_is_positive_real):
        return UNKNOWN_VALUE_SET

    if base_set.sympy_set == ZERO.sympy_set and exponent_is_positive_real:
        return ZERO

    if is_known_subset(base_set, INTEGERS) and exponent_is_nonnegative_integer:
        if is_known_subset(base_set, POSITIVE_INTEGERS):
            return POSITIVE_INTEGERS
        if is_known_subset(base_set, NATURALS_0) or is_known_subset(exponent_set, POSITIVE_EVEN_INTEGERS):
            return NATURALS_0
        if is_known_subset(base_set, NEGATIVE_INTEGERS) and is_known_subset(exponent_set, POSITIVE_ODD_INTEGERS):
            return NEGATIVE_INTEGERS
        return INTEGERS

    if (
        is_known_subset(base_set, POSITIVE_REALS)
        and is_known_subset(exponent_set, REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(base_set, NON_NEGATIVE_REALS)
        and is_known_subset(exponent_set, POSITIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(base_set, REALS)
        and is_known_subset(exponent_set, POSITIVE_EVEN_INTEGERS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(base_set, NON_ZERO_REALS)
        and is_known_subset(exponent_set, NEGATIVE_EVEN_INTEGERS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(base_set, REALS)
        and is_known_subset(exponent_set, POSITIVE_ODD_INTEGERS)
    ):
        return REALS

    if (
        is_known_subset(base_set, NON_ZERO_REALS)
        and is_known_subset(exponent_set, NEGATIVE_ODD_INTEGERS)
    ):
        return NON_ZERO_REALS

    if (
        is_known_subset(base_set, NEGATIVE_REALS)
        and is_known_subset(exponent_set, NON_INTEGER_REALS)
    ):
        return ValueSet(sympy_set=COMPLEXES.sympy_set - REALS.sympy_set, dtype_types=(np.complexfloating,))

    if base_is_nonzero:
        return NON_ZERO_COMPLEXES

    return COMPLEXES


# Unary operations

@_arithmetic_inference(dtype_operation=np.negative, preserves_integers=True)
def _infer_negative_value_set(value_set: ValueSet) -> ValueSet:

    if value_set.sympy_set == ZERO.sympy_set:
        return ZERO

    if is_known_subset(value_set, POSITIVE_REALS):
        return NEGATIVE_REALS

    if is_known_subset(value_set, NEGATIVE_REALS):
        return POSITIVE_REALS

    if is_known_subset(value_set, NON_NEGATIVE_REALS):
        return NON_POSITIVE_REALS

    if is_known_subset(value_set, NON_POSITIVE_REALS):
        return NON_NEGATIVE_REALS

    if is_known_subset(value_set, INTEGERS):
        return INTEGERS

    if is_known_subset(value_set, REALS):
        return REALS

    if is_known_subset(value_set, COMPLEXES):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


@_arithmetic_inference(dtype_operation=np.absolute, preserves_integers=True)
def _infer_absolute_value_set(value_set: ValueSet) -> ValueSet:

    if value_set.sympy_set == ZERO.sympy_set:
        return ZERO

    if is_known_subset(value_set, NON_NEGATIVE_REALS):
        return value_set

    if is_known_subset(value_set, NON_ZERO_REALS):
        return POSITIVE_REALS

    if is_known_subset(value_set, REALS):
        return NON_NEGATIVE_REALS

    if is_known_subset(value_set, NON_ZERO_COMPLEXES):
        return POSITIVE_REALS

    if is_known_subset(value_set, COMPLEXES):
        return NON_NEGATIVE_REALS

    return UNKNOWN_VALUE_SET
