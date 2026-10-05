from functools import cache
from itertools import product
from typing import cast

import numpy as np
import sympy as sp

from problab.value_sets import _mathematical_inference as mathematical
from problab.value_sets._inference import _infer_dtype_types, _input_dtype_candidates
from problab.value_sets._unknown import _UnknownValueSet
from problab.value_sets._utils import is_known_subset
from problab.value_sets.base import NumericValueSet
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.sets import (
    COMPLEXES,
    EVEN_INTEGERS,
    INTEGERS,
    NON_NEGATIVE_REALS,
    NON_POSITIVE_REALS,
    ONE,
    ODD_INTEGERS,
    REALS,
    ZERO,
)


def _real_bounds(value_set: NumericValueSet) -> tuple[sp.Expr, sp.Expr] | None:
    if isinstance(value_set.sympy_set, _UnknownValueSet) or not is_known_subset(value_set, REALS):
        return None
    try:
        return sp.sympify(value_set.sympy_set.inf), sp.sympify(value_set.sympy_set.sup)
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        return None


def _dtype_limits(dtype_type: type[np.generic]) -> tuple[sp.Expr, sp.Expr] | None:
    dtype = np.dtype(dtype_type)
    if np.issubdtype(dtype, np.integer):
        integer_limits = np.iinfo(dtype)
        return sp.Integer(integer_limits.min), sp.Integer(integer_limits.max)
    if np.issubdtype(dtype, np.inexact):
        floating_limits = np.finfo(dtype.char)
        return (
            sp.Rational(*floating_limits.min.as_integer_ratio()),
            sp.Rational(*floating_limits.max.as_integer_ratio()),
        )
    return None


def _may_contain_zero(value_set: NumericValueSet) -> bool:
    if isinstance(value_set.sympy_set, _UnknownValueSet):
        return True
    try:
        return value_set.sympy_set.contains(sp.Integer(0)) is not sp.false
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        return True


def _may_be_positive(value_set: NumericValueSet) -> bool:
    if value_set.allows_positive_infinity:
        return True
    bounds = _real_bounds(value_set)
    if bounds is not None and (bounds[1] <= 0) is sp.true:
        return False
    return not is_known_subset(value_set, NON_POSITIVE_REALS)


def _may_be_negative(value_set: NumericValueSet) -> bool:
    if value_set.allows_negative_infinity:
        return True
    bounds = _real_bounds(value_set)
    if bounds is not None and (bounds[0] >= 0) is sp.true:
        return False
    return not is_known_subset(value_set, NON_NEGATIVE_REALS)


def _may_use_complex_dtype(dtype_types: tuple[type[np.generic], ...] | None) -> bool:
    return dtype_types is None or any(
        issubclass(dtype_type, np.complexfloating)
        or issubclass(np.complexfloating, dtype_type)
        for dtype_type in dtype_types
    )


def _allow_underflow_to_zero(value_set: NumericValueSet) -> NumericValueSet:
    if isinstance(value_set.sympy_set, _UnknownValueSet):
        return value_set

    dtype_types = value_set.dtype_types
    if dtype_types is not None and all(
        issubclass(dtype_type, np.integer) for dtype_type in dtype_types
    ):
        return value_set

    if is_known_subset(value_set, INTEGERS):
        return value_set

    return HomogeneousNumericValueSet(
        sympy_set=sp.Union(value_set.sympy_set, ZERO.sympy_set),
        dtype_types=dtype_types,
        allows_positive_infinity=value_set.allows_positive_infinity,
        allows_negative_infinity=value_set.allows_negative_infinity,
        allows_nan=value_set.allows_nan,
    )


def _infer_add_value_set(
    left_set: NumericValueSet,
    right_set: NumericValueSet,
) -> NumericValueSet:

    result = mathematical._infer_add_value_set(left_set, right_set)
    dtype_types = result.dtype_types

    positive = left_set.allows_positive_infinity or right_set.allows_positive_infinity
    negative = left_set.allows_negative_infinity or right_set.allows_negative_infinity
    nan = (
        left_set.allows_nan or right_set.allows_nan
        or (left_set.allows_positive_infinity and right_set.allows_negative_infinity)
        or (left_set.allows_negative_infinity and right_set.allows_positive_infinity)
    )
    left_bounds, right_bounds = _real_bounds(left_set), _real_bounds(right_set)
    minimum = maximum = None
    if left_bounds is not None and right_bounds is not None:
        minimum = left_bounds[0] + right_bounds[0]
        maximum = left_bounds[1] + right_bounds[1]

    integer_overflow = False
    if dtype_types is None:
        positive = negative = True
    for dtype_type in dtype_types or ():
        limits = _dtype_limits(dtype_type)
        below = limits is None or minimum is None or (minimum >= limits[0]) is not sp.true
        above = limits is None or maximum is None or (maximum <= limits[1]) is not sp.true
        negative |= below
        positive |= above
        integer_overflow |= issubclass(dtype_type, np.integer) and (below or above)
    if integer_overflow:
        dtype_types = tuple(dict.fromkeys((*dtype_types, np.float64)))

    return HomogeneousNumericValueSet(
        sympy_set=result.sympy_set,
        dtype_types=dtype_types,
        allows_positive_infinity=positive,
        allows_negative_infinity=negative,
        allows_nan=nan,
    )


def _infer_subtract_value_set(left_set: NumericValueSet, right_set: NumericValueSet) -> NumericValueSet:
    result = mathematical._infer_subtract_value_set(left_set, right_set)
    dtype_types = result.dtype_types

    positive = left_set.allows_positive_infinity or right_set.allows_negative_infinity
    negative = left_set.allows_negative_infinity or right_set.allows_positive_infinity
    nan = (
        left_set.allows_nan or right_set.allows_nan
        or (left_set.allows_positive_infinity and right_set.allows_positive_infinity)
        or (left_set.allows_negative_infinity and right_set.allows_negative_infinity)
    )
    left_bounds, right_bounds = _real_bounds(left_set), _real_bounds(right_set)
    minimum = maximum = None
    if left_bounds is not None and right_bounds is not None:
        minimum = left_bounds[0] - right_bounds[1]
        maximum = left_bounds[1] - right_bounds[0]

    integer_overflow = False
    if dtype_types is None:
        positive = negative = True
    for dtype_type in dtype_types or ():
        limits = _dtype_limits(dtype_type)
        below = limits is None or minimum is None or (minimum >= limits[0]) is not sp.true
        above = limits is None or maximum is None or (maximum <= limits[1]) is not sp.true
        negative |= below
        positive |= above
        integer_overflow |= issubclass(dtype_type, np.integer) and (below or above)
    if integer_overflow:
        dtype_types = tuple(dict.fromkeys((*dtype_types, np.float64)))

    return HomogeneousNumericValueSet(
        sympy_set=result.sympy_set,
        dtype_types=dtype_types,
        allows_positive_infinity=positive,
        allows_negative_infinity=negative,
        allows_nan=nan,
    )


def _infer_multiply_value_set(left_set: NumericValueSet, right_set: NumericValueSet) -> NumericValueSet:
    result = mathematical._infer_multiply_value_set(left_set, right_set)
    if left_set.sympy_set not in (ZERO.sympy_set, ONE.sympy_set) and right_set.sympy_set not in (
        ZERO.sympy_set, ONE.sympy_set,
    ):
        result = _allow_underflow_to_zero(result)
    dtype_types = result.dtype_types
    left_infinity = left_set.allows_positive_infinity or left_set.allows_negative_infinity
    right_infinity = right_set.allows_positive_infinity or right_set.allows_negative_infinity
    positive = (
        (left_set.allows_positive_infinity and _may_be_positive(right_set))
        or (left_set.allows_negative_infinity and _may_be_negative(right_set))
        or (right_set.allows_positive_infinity and _may_be_positive(left_set))
        or (right_set.allows_negative_infinity and _may_be_negative(left_set))
    )
    negative = (
        (left_set.allows_positive_infinity and _may_be_negative(right_set))
        or (left_set.allows_negative_infinity and _may_be_positive(right_set))
        or (right_set.allows_positive_infinity and _may_be_negative(left_set))
        or (right_set.allows_negative_infinity and _may_be_positive(left_set))
    )
    nan = (
        left_set.allows_nan or right_set.allows_nan
        or (left_infinity and _may_contain_zero(right_set))
        or (right_infinity and _may_contain_zero(left_set))
    )
    minimum = maximum = None
    left_bounds, right_bounds = _real_bounds(left_set), _real_bounds(right_set)
    if left_set.sympy_set == ZERO.sympy_set or right_set.sympy_set == ZERO.sympy_set:
        minimum = maximum = sp.Integer(0)
    elif left_bounds is not None and right_bounds is not None:
        try:
            products = tuple(a * b for a, b in product(left_bounds, right_bounds))
            minimum, maximum = sp.Min(*products), sp.Max(*products)
        except (TypeError, ValueError, NotImplementedError):
            pass

    integer_overflow = False
    finite_overflow = dtype_types is None
    positive_product = (
        (_may_be_positive(left_set) and _may_be_positive(right_set))
        or (_may_be_negative(left_set) and _may_be_negative(right_set))
    )
    negative_product = (
        (_may_be_positive(left_set) and _may_be_negative(right_set))
        or (_may_be_negative(left_set) and _may_be_positive(right_set))
    )
    if dtype_types is None:
        positive |= positive_product
        negative |= negative_product
    for dtype_type in dtype_types or ():
        limits = _dtype_limits(dtype_type)
        below = limits is None or minimum is None or (minimum >= limits[0]) is not sp.true
        above = limits is None or maximum is None or (maximum <= limits[1]) is not sp.true
        negative |= below and negative_product
        positive |= above and positive_product
        finite_overflow |= below or above
        integer_overflow |= issubclass(dtype_type, np.integer) and (below or above)
    if integer_overflow:
        dtype_types = tuple(dict.fromkeys((*dtype_types, np.float64)))
    if _may_use_complex_dtype(dtype_types) and (finite_overflow or left_infinity or right_infinity):
        positive = negative = nan = True

    return HomogeneousNumericValueSet(
        sympy_set=result.sympy_set,
        dtype_types=dtype_types,
        allows_positive_infinity=positive,
        allows_negative_infinity=negative,
        allows_nan=nan,
    )


def _infer_divide_value_set(numerator_set: NumericValueSet, denominator_set: NumericValueSet) -> NumericValueSet:
    result = mathematical._infer_divide_value_set(numerator_set, denominator_set)
    if numerator_set.sympy_set != ZERO.sympy_set and denominator_set.sympy_set != ONE.sympy_set:
        result = _allow_underflow_to_zero(result)
    dtype_types = _infer_dtype_types(np.divide, (numerator_set, denominator_set))
    numerator_infinity = numerator_set.allows_positive_infinity or numerator_set.allows_negative_infinity
    denominator_infinity = denominator_set.allows_positive_infinity or denominator_set.allows_negative_infinity
    numerator_zero = _may_contain_zero(numerator_set)
    denominator_zero = _may_contain_zero(denominator_set)
    positive = (
        (numerator_set.allows_positive_infinity and _may_be_positive(denominator_set))
        or (numerator_set.allows_negative_infinity and _may_be_negative(denominator_set))
    )
    negative = (
        (numerator_set.allows_positive_infinity and _may_be_negative(denominator_set))
        or (numerator_set.allows_negative_infinity and _may_be_positive(denominator_set))
    )
    nan = (
        numerator_set.allows_nan or denominator_set.allows_nan
        or (numerator_infinity and denominator_infinity)
        or (numerator_zero and denominator_zero)
    )
    # The symbolic set does not distinguish +0.0 from -0.0.
    if denominator_zero and (_may_be_positive(numerator_set) or _may_be_negative(numerator_set)):
        positive = negative = True

    minimum = maximum = None
    numerator_bounds = _real_bounds(numerator_set)
    denominator_bounds = _real_bounds(denominator_set)
    if numerator_set.sympy_set == ZERO.sympy_set:
        minimum = maximum = sp.Integer(0)
    elif numerator_bounds is not None and denominator_bounds is not None:
        if (denominator_bounds[0] > 0) is sp.true or (denominator_bounds[1] < 0) is sp.true:
            try:
                quotients = tuple(a / b for a, b in product(numerator_bounds, denominator_bounds))
                minimum, maximum = sp.Min(*quotients), sp.Max(*quotients)
            except (TypeError, ValueError, NotImplementedError):
                pass

    positive_quotient = (
        (_may_be_positive(numerator_set) and _may_be_positive(denominator_set))
        or (_may_be_negative(numerator_set) and _may_be_negative(denominator_set))
    )
    negative_quotient = (
        (_may_be_positive(numerator_set) and _may_be_negative(denominator_set))
        or (_may_be_negative(numerator_set) and _may_be_positive(denominator_set))
    )
    finite_overflow = dtype_types is None
    if dtype_types is None:
        positive |= positive_quotient
        negative |= negative_quotient
    for dtype_type in dtype_types or ():
        limits = _dtype_limits(dtype_type)
        below = limits is None or minimum is None or (minimum >= limits[0]) is not sp.true
        above = limits is None or maximum is None or (maximum <= limits[1]) is not sp.true
        negative |= below and negative_quotient
        positive |= above and positive_quotient
        finite_overflow |= below or above
    if _may_use_complex_dtype(dtype_types) and (
        finite_overflow or numerator_infinity or denominator_infinity or denominator_zero
    ):
        positive = negative = nan = True

    return HomogeneousNumericValueSet(
        sympy_set=result.sympy_set,
        dtype_types=dtype_types,
        allows_positive_infinity=positive,
        allows_negative_infinity=negative,
        allows_nan=nan,
    )


def _infer_floor_divide_value_set(numerator_set: NumericValueSet, denominator_set: NumericValueSet) -> NumericValueSet:
    return mathematical._infer_floor_divide_value_set(numerator_set, denominator_set)


def _infer_modulo_value_set(dividend_set: NumericValueSet, divisor_set: NumericValueSet) -> NumericValueSet:
    result = mathematical._infer_modulo_value_set(dividend_set, divisor_set)
    dtype_types = _infer_dtype_types(np.remainder, (dividend_set, divisor_set))
    if dtype_types is not None:
        dtype_types = tuple(dict.fromkeys(
            np.result_type(np.empty(0, dtype=dtype_type), np.nan).type
            for dtype_type in dtype_types
        ))

    return HomogeneousNumericValueSet(
        sympy_set=result.sympy_set,
        dtype_types=dtype_types,
        allows_positive_infinity=divisor_set.allows_positive_infinity and _may_be_negative(dividend_set),
        allows_negative_infinity=divisor_set.allows_negative_infinity and _may_be_positive(dividend_set),
        allows_nan=(
            dividend_set.allows_nan or divisor_set.allows_nan
            or dividend_set.allows_positive_infinity or dividend_set.allows_negative_infinity
            or _may_contain_zero(divisor_set)
        ),
    )


def _infer_power_dtype_types(
    base_set: NumericValueSet,
    exponent_set: NumericValueSet,
    *,
    allow_complex: bool,
) -> tuple[type[np.generic], ...] | None:
    candidates = _input_dtype_candidates((base_set, exponent_set))
    if candidates is None:
        return None

    exponent_may_be_negative = _may_be_negative(exponent_set)
    exponent_may_be_nonnegative = _may_be_positive(exponent_set) or _may_contain_zero(exponent_set)
    base_may_be_negative = _may_be_negative(base_set)
    output_types = []

    for base_dtype, exponent_dtype in product(*candidates):
        integer_base = np.issubdtype(base_dtype, np.integer)
        integer_exponent = np.issubdtype(exponent_dtype, np.integer)

        try:
            if integer_base and integer_exponent and exponent_may_be_nonnegative:
                output_type = np.power.resolve_dtypes((base_dtype, exponent_dtype, None))[-1].type
                if output_type not in output_types:
                    output_types.append(output_type)

                if issubclass(output_type, np.integer):
                    below, above = _integer_power_overflow(
                        base_set, exponent_set, base_dtype, exponent_dtype, np.dtype(output_type),
                    )
                    if (below or above) and np.float64 not in output_types:
                        output_types.append(np.float64)

            if integer_base and integer_exponent and not exponent_may_be_negative:
                continue

            base_dtypes = [base_dtype]
            if integer_base and exponent_may_be_negative:
                if not exponent_may_be_nonnegative or integer_exponent:
                    base_dtypes = []
                base_dtypes.append(np.dtype(np.float64))

            if allow_complex and base_may_be_negative:
                base_dtypes.extend(
                    np.dtype(np.complex64)
                    if issubclass(dtype.type, (
                        np.float32, np.int8, np.int16, np.uint8, np.uint16, np.complex64,
                    ))
                    else np.dtype(np.complex128)
                    for dtype in tuple(base_dtypes)
                )

            exponent_dtypes = [exponent_dtype]
            if allow_complex and exponent_may_be_negative:
                exponent_dtypes.append(
                    np.multiply.resolve_dtypes((exponent_dtype, float, None))[-1]
                )

            for base_dtype, exponent_dtype in product(base_dtypes, exponent_dtypes):
                output_type = np.power.resolve_dtypes((base_dtype, exponent_dtype, None))[-1].type
                if output_type not in output_types:
                    output_types.append(output_type)
        except (TypeError, ValueError):
            return None

    return tuple(output_types)


def _integer_power_overflow(
    base_set: NumericValueSet,
    exponent_set: NumericValueSet,
    base_dtype: np.dtype,
    exponent_dtype: np.dtype,
    result_dtype: np.dtype,
) -> tuple[bool, bool]:
    limits = np.iinfo(result_dtype)
    base_bounds, exponent_bounds = _real_bounds(base_set), _real_bounds(exponent_set)
    if base_bounds is None or exponent_bounds is None:
        return True, True
    try:
        base_limits = np.iinfo(base_dtype)
        exponent_limits = np.iinfo(exponent_dtype)
        base_minimum = int(sp.ceiling(sp.Max(base_bounds[0], sp.Integer(base_limits.min))))
        base_maximum = int(sp.floor(sp.Min(base_bounds[1], sp.Integer(base_limits.max))))
        exponent_minimum = int(sp.ceiling(sp.Max(exponent_bounds[0], sp.Integer(0))))
        exponent_maximum = int(sp.floor(sp.Min(exponent_bounds[1], sp.Integer(exponent_limits.max))))
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        return True, True
    if base_minimum > base_maximum or exponent_minimum > exponent_maximum:
        return False, False

    bases = {base_minimum, base_maximum}
    bases.update(value for value in (-1, 0, 1) if base_minimum <= value <= base_maximum)
    exponents = {exponent_minimum, exponent_maximum}
    if exponent_minimum < exponent_maximum:
        exponents.update((exponent_minimum + 1, exponent_maximum - 1))

    below = above = False
    magnitude_bits = max(abs(limits.min), limits.max).bit_length()
    for base, exponent in product(bases, exponents):
        # Avoid constructing enormous Python integers merely to infer overflow.
        if abs(base) >= 2 and exponent > magnitude_bits:
            if base < 0 and exponent % 2:
                below = True
            else:
                above = True
        else:
            value = pow(base, exponent)
            below |= value < limits.min
            above |= value > limits.max
    return below, above


def _infer_power_result(
    base_set: NumericValueSet,
    exponent_set: NumericValueSet,
    *,
    allow_complex: bool,
) -> NumericValueSet:
    result = mathematical._infer_power_value_set(base_set, exponent_set)
    dtype_types = _infer_power_dtype_types(base_set, exponent_set, allow_complex=allow_complex)
    complex_output = _may_use_complex_dtype(dtype_types)
    candidates = _input_dtype_candidates((base_set, exponent_set))
    odd_exponent = is_known_subset(exponent_set, ODD_INTEGERS)
    even_exponent = is_known_subset(exponent_set, EVEN_INTEGERS)
    floating_branch = candidates is None
    if candidates is not None:
        floating_branch = any(
            not (np.issubdtype(base_dtype, np.integer) and np.issubdtype(exponent_dtype, np.integer))
            or _may_be_negative(exponent_set)
            for base_dtype, exponent_dtype in product(*candidates)
        )
    if floating_branch and not complex_output:
        exponent_bounds = _real_bounds(exponent_set)
        for dtype_type in dtype_types or ():
            if not issubclass(dtype_type, np.floating):
                continue
            exact_integer_limit = sp.Integer(2) ** (np.finfo(np.dtype(dtype_type).char).nmant + 1)
            if exponent_bounds is None or any(
                (sp.Abs(bound) <= exact_integer_limit) is not sp.true for bound in exponent_bounds
            ):
                # Casting a large integer exponent to floating point can change its parity.
                odd_exponent = even_exponent = False
    exponent_exceptional = (
        exponent_set.allows_positive_infinity or exponent_set.allows_negative_infinity or exponent_set.allows_nan
    )
    base_exceptional = base_set.allows_positive_infinity or base_set.allows_negative_infinity or base_set.allows_nan
    zero_exponent = exponent_set.sympy_set == ZERO.sympy_set and not exponent_exceptional
    unit_base = base_set.sympy_set == ONE.sympy_set and not base_exceptional and not complex_output

    positive = negative = False
    nan = False
    if not zero_exponent and not unit_base:
        nan = base_set.allows_nan or exponent_set.allows_nan
        base_zero = _may_contain_zero(base_set)
        exponent_negative = _may_be_negative(exponent_set)
        if base_zero and exponent_negative:
            # Both signs of floating-point zero have the same symbolic value.
            positive = negative = True
        if exponent_set.allows_positive_infinity or exponent_set.allows_negative_infinity:
            positive = True
        if (base_set.allows_positive_infinity or base_set.allows_negative_infinity) and _may_be_positive(exponent_set):
            positive |= base_set.allows_positive_infinity or (
                base_set.allows_negative_infinity and not odd_exponent
            )
            negative |= base_set.allows_negative_infinity and not even_exponent
        if not allow_complex and _may_be_negative(base_set) and not is_known_subset(exponent_set, INTEGERS):
            nan = True

        if candidates is None:
            positive = negative = nan = True
        else:
            for base_dtype, exponent_dtype in product(*candidates):
                if np.issubdtype(base_dtype, np.integer) and np.issubdtype(exponent_dtype, np.integer):
                    output_type = np.power.resolve_dtypes((base_dtype, exponent_dtype, None))[-1].type
                    if issubclass(output_type, np.integer):
                        below, above = _integer_power_overflow(
                            base_set, exponent_set, base_dtype, exponent_dtype, np.dtype(output_type),
                        )
                        negative |= below
                        positive |= above

        log_maximum = None
        base_bounds, exponent_bounds = _real_bounds(base_set), _real_bounds(exponent_set)
        if base_bounds is not None and exponent_bounds is not None:
            try:
                minimum_magnitude = (
                    sp.Integer(0)
                    if (base_bounds[0] <= 0) is sp.true and (base_bounds[1] >= 0) is sp.true
                    else sp.Min(*(sp.Abs(value) for value in base_bounds))
                )
                maximum_magnitude = sp.Max(*(sp.Abs(value) for value in base_bounds))
                # At zero use the real logarithm's limiting value, not SymPy's zoo.
                logarithms = tuple(
                    -sp.oo if magnitude == 0 else sp.log(magnitude)
                    for magnitude in (minimum_magnitude, maximum_magnitude)
                )
                terms = tuple(
                    sp.Integer(0) if exponent == 0 or logarithm == 0 else exponent * logarithm
                    for exponent, logarithm in product(exponent_bounds, logarithms)
                )
                log_maximum = sp.Max(*terms)
            except (TypeError, ValueError, NotImplementedError):
                pass

        floating_overflow = dtype_types is None
        for dtype_type in dtype_types or ():
            if not issubclass(dtype_type, np.inexact):
                continue
            limits = _dtype_limits(dtype_type)
            floating_overflow |= log_maximum is None or (log_maximum <= sp.log(limits[1])) is not sp.true
        if floating_overflow:
            positive |= _may_be_positive(base_set) or (
                _may_be_negative(base_set) and not odd_exponent
            )
            negative |= _may_be_negative(base_set) and not even_exponent
        if complex_output and (floating_overflow or base_exceptional or exponent_exceptional):
            positive = negative = nan = True

    sympy_set = result.sympy_set
    if not isinstance(sympy_set, _UnknownValueSet) and sympy_set.is_subset(sp.S.Reals) is not True:
        sympy_set = COMPLEXES.sympy_set
    result = HomogeneousNumericValueSet(
        sympy_set=sympy_set,
        dtype_types=dtype_types,
        allows_positive_infinity=positive,
        allows_negative_infinity=negative,
        allows_nan=nan,
    )
    if exponent_set.sympy_set not in (ZERO.sympy_set, ONE.sympy_set) and base_set.sympy_set not in (
        ZERO.sympy_set, ONE.sympy_set,
    ):
        result = _allow_underflow_to_zero(result)
    return result


def _infer_power_value_set(base_set: NumericValueSet, exponent_set: NumericValueSet) -> NumericValueSet:
    return _infer_power_result(base_set, exponent_set, allow_complex=True)


def _infer_real_power_value_set(base_set: NumericValueSet, exponent_set: NumericValueSet) -> NumericValueSet:
    return _infer_power_result(base_set, exponent_set, allow_complex=False)


def _infer_negative_value_set(value_set: NumericValueSet) -> NumericValueSet:
    result = mathematical._infer_negative_value_set(value_set)
    dtype_types = result.dtype_types

    positive = value_set.allows_negative_infinity
    negative = value_set.allows_positive_infinity
    bounds = _real_bounds(value_set)
    integer_overflow = False
    for dtype_type in dtype_types or ():
        if not issubclass(dtype_type, np.integer):
            continue
        limits = _dtype_limits(dtype_type)
        below = limits is None or bounds is None or (-bounds[1] >= limits[0]) is not sp.true
        above = limits is None or bounds is None or (-bounds[0] <= limits[1]) is not sp.true
        negative |= below
        positive |= above
        integer_overflow |= below or above
    if dtype_types is None:
        positive = negative = True
    if integer_overflow:
        dtype_types = tuple(dict.fromkeys((*dtype_types, np.float64)))

    return HomogeneousNumericValueSet(
        sympy_set=result.sympy_set,
        dtype_types=dtype_types,
        allows_positive_infinity=positive,
        allows_negative_infinity=negative,
        allows_nan=value_set.allows_nan,
    )


def _infer_absolute_value_set(value_set: NumericValueSet) -> NumericValueSet:
    result = mathematical._infer_absolute_value_set(value_set)
    dtype_types = result.dtype_types

    positive = value_set.allows_positive_infinity or value_set.allows_negative_infinity
    maximum = None
    bounds = _real_bounds(value_set)
    if bounds is not None:
        maximum = sp.Max(sp.Abs(bounds[0]), sp.Abs(bounds[1]))
    elif isinstance(value_set.sympy_set, sp.FiniteSet) and value_set.sympy_set:
        try:
            maximum = sp.Max(*(sp.Abs(value) for value in value_set.sympy_set))
        except (TypeError, ValueError, NotImplementedError):
            pass

    integer_overflow = False
    if dtype_types is None:
        positive = True
    for dtype_type in dtype_types or ():
        limits = _dtype_limits(dtype_type)
        above = limits is None or maximum is None or (maximum <= limits[1]) is not sp.true
        positive |= above
        integer_overflow |= issubclass(dtype_type, np.integer) and above
    if integer_overflow:
        dtype_types = tuple(dict.fromkeys((*dtype_types, np.float64)))

    return HomogeneousNumericValueSet(
        sympy_set=result.sympy_set,
        dtype_types=dtype_types,
        allows_positive_infinity=positive,
        allows_negative_infinity=False,
        allows_nan=value_set.allows_nan,
    )


def _function_result(
    operation: np.ufunc,
    result_set: NumericValueSet,
    *inputs: NumericValueSet,
    positive: bool = False,
    negative: bool = False,
    nan: bool = False,
) -> HomogeneousNumericValueSet:
    return HomogeneousNumericValueSet(
        sympy_set=result_set.sympy_set,
        dtype_types=_infer_dtype_types(operation, inputs),
        allows_positive_infinity=positive,
        allows_negative_infinity=negative,
        allows_nan=nan,
    )


def _may_contain_value(value_set: NumericValueSet, value: int) -> bool:
    if isinstance(value_set.sympy_set, _UnknownValueSet):
        return True
    try:
        return value_set.sympy_set.contains(sp.Integer(value)) is not sp.false
    except (TypeError, ValueError, NotImplementedError):
        return True


def _may_exceed_floating_limit(
    maximum: sp.Expr | None,
    dtype_types: tuple[type[np.generic], ...] | None,
) -> bool:
    if maximum is None or dtype_types is None:
        return True
    for dtype_type in dtype_types:
        dtype = np.dtype(dtype_type)
        if not np.issubdtype(dtype, np.floating):
            return True
        # Leave one representable step below the limit for rounding in the ufunc.
        safe_maximum = np.nextafter(np.finfo(dtype.char).max, dtype.type(0))
        limit = sp.Rational(*safe_maximum.as_integer_ratio())
        if (maximum <= limit) is not sp.true:
            return True
    return False


def _infer_exp_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    bounds = _real_bounds(value_set)
    maximum = None if bounds is None else sp.exp(bounds[1])
    return _function_result(
        np.exp, result_set, value_set,
        positive=value_set.allows_positive_infinity or _may_exceed_floating_limit(
            maximum, _infer_dtype_types(np.exp, (value_set,)),
        ),
        nan=value_set.allows_nan,
    )


def _infer_expm1_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    bounds = _real_bounds(value_set)
    maximum = None if bounds is None else sp.exp(bounds[1]) - 1
    return _function_result(
        np.expm1, result_set, value_set,
        positive=value_set.allows_positive_infinity or _may_exceed_floating_limit(
            maximum, _infer_dtype_types(np.expm1, (value_set,)),
        ),
        nan=value_set.allows_nan,
    )


def _infer_log_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.log, result_set, value_set,
        positive=value_set.allows_positive_infinity,
        negative=_may_contain_zero(value_set),
        nan=value_set.allows_nan or _may_be_negative(value_set),
    )


def _infer_log2_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    result = _infer_log_value_set(result_set, value_set)
    return _function_result(
        np.log2, result_set, value_set,
        positive=result.allows_positive_infinity,
        negative=result.allows_negative_infinity,
        nan=result.allows_nan,
    )


def _infer_log10_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    result = _infer_log_value_set(result_set, value_set)
    return _function_result(
        np.log10, result_set, value_set,
        positive=result.allows_positive_infinity,
        negative=result.allows_negative_infinity,
        nan=result.allows_nan,
    )


def _infer_log1p_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.log1p, result_set, value_set,
        positive=value_set.allows_positive_infinity,
        negative=_may_contain_value(value_set, -1),
        nan=(
            value_set.allows_nan or value_set.allows_negative_infinity
            or not is_known_subset(value_set, sp.Interval(-1, sp.oo))
        ),
    )


def _infer_sqrt_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.sqrt, result_set, value_set,
        positive=value_set.allows_positive_infinity,
        nan=value_set.allows_nan or _may_be_negative(value_set),
    )


def _infer_floor_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.floor, result_set, value_set,
        positive=value_set.allows_positive_infinity,
        negative=value_set.allows_negative_infinity,
        nan=value_set.allows_nan,
    )


def _infer_ceil_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.ceil, result_set, value_set,
        positive=value_set.allows_positive_infinity,
        negative=value_set.allows_negative_infinity,
        nan=value_set.allows_nan,
    )


def _infer_sign_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(np.sign, result_set, value_set, nan=value_set.allows_nan)


def _infer_sin_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.sin, result_set, value_set,
        nan=(value_set.allows_nan or value_set.allows_positive_infinity or value_set.allows_negative_infinity),
    )


def _infer_cos_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.cos, result_set, value_set,
        nan=(value_set.allows_nan or value_set.allows_positive_infinity or value_set.allows_negative_infinity),
    )


def _infer_tan_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    # A range spanning poles cannot supply a finite bound for the tangent.
    bounds = _real_bounds(value_set)
    minimum = maximum = None
    if bounds is not None:
        lower, upper = bounds
        if (lower > -sp.pi / 2) is sp.true and (upper < sp.pi / 2) is sp.true:
            minimum, maximum = sp.tan(lower), sp.tan(upper)
    dtypes = _infer_dtype_types(np.tan, (value_set,))
    return _function_result(
        np.tan, result_set, value_set,
        positive=_may_exceed_floating_limit(maximum, dtypes),
        negative=_may_exceed_floating_limit(None if minimum is None else -minimum, dtypes),
        nan=(value_set.allows_nan or value_set.allows_positive_infinity or value_set.allows_negative_infinity),
    )


def _infer_arcsin_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.arcsin, result_set, value_set,
        nan=(
            value_set.allows_nan or value_set.allows_positive_infinity or value_set.allows_negative_infinity
            or not is_known_subset(value_set, sp.Interval(-1, 1))
        ),
    )


def _infer_arccos_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.arccos, result_set, value_set,
        nan=(
            value_set.allows_nan or value_set.allows_positive_infinity or value_set.allows_negative_infinity
            or not is_known_subset(value_set, sp.Interval(-1, 1))
        ),
    )


def _infer_arctan_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(np.arctan, result_set, value_set, nan=value_set.allows_nan)


def _infer_sinh_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    bounds = _real_bounds(value_set)
    minimum = None if bounds is None else sp.sinh(bounds[0])
    maximum = None if bounds is None else sp.sinh(bounds[1])
    dtypes = _infer_dtype_types(np.sinh, (value_set,))
    return _function_result(
        np.sinh, result_set, value_set,
        positive=value_set.allows_positive_infinity or _may_exceed_floating_limit(maximum, dtypes),
        negative=value_set.allows_negative_infinity or _may_exceed_floating_limit(
            None if minimum is None else -minimum, dtypes,
        ),
        nan=value_set.allows_nan,
    )


def _infer_cosh_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    bounds = _real_bounds(value_set)
    maximum = None if bounds is None else sp.cosh(sp.Max(sp.Abs(bounds[0]), sp.Abs(bounds[1])))
    return _function_result(
        np.cosh, result_set, value_set,
        positive=(
            value_set.allows_positive_infinity or value_set.allows_negative_infinity
            or _may_exceed_floating_limit(maximum, _infer_dtype_types(np.cosh, (value_set,)))
        ),
        nan=value_set.allows_nan,
    )


def _infer_tanh_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(np.tanh, result_set, value_set, nan=value_set.allows_nan)


def _infer_arcsinh_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.arcsinh, result_set, value_set,
        positive=value_set.allows_positive_infinity,
        negative=value_set.allows_negative_infinity,
        nan=value_set.allows_nan,
    )


def _infer_arccosh_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.arccosh, result_set, value_set,
        positive=value_set.allows_positive_infinity,
        nan=(
            value_set.allows_nan or value_set.allows_negative_infinity
            or not is_known_subset(value_set, sp.Interval(1, sp.oo))
        ),
    )


def _infer_arctanh_value_set(
    result_set: NumericValueSet,
    value_set: NumericValueSet,
) -> NumericValueSet:
    return _function_result(
        np.arctanh, result_set, value_set,
        positive=_may_contain_value(value_set, 1),
        negative=_may_contain_value(value_set, -1),
        nan=(
            value_set.allows_nan or value_set.allows_positive_infinity or value_set.allows_negative_infinity
            or not is_known_subset(value_set, sp.Interval(-1, 1))
        ),
    )


def _infer_hypot_value_set(
    result_set: NumericValueSet,
    left_set: NumericValueSet,
    right_set: NumericValueSet,
) -> NumericValueSet:
    left_bounds, right_bounds = _real_bounds(left_set), _real_bounds(right_set)
    maximum = None
    if left_bounds is not None and right_bounds is not None:
        left_magnitude = sp.Max(sp.Abs(left_bounds[0]), sp.Abs(left_bounds[1]))
        right_magnitude = sp.Max(sp.Abs(right_bounds[0]), sp.Abs(right_bounds[1]))
        maximum = sp.sqrt(left_magnitude ** 2 + right_magnitude ** 2)
    return _function_result(
        np.hypot, result_set, left_set, right_set,
        positive=(
            left_set.allows_positive_infinity or left_set.allows_negative_infinity
            or right_set.allows_positive_infinity or right_set.allows_negative_infinity
            or _may_exceed_floating_limit(maximum, _infer_dtype_types(np.hypot, (left_set, right_set)))
        ),
        nan=left_set.allows_nan or right_set.allows_nan,
    )


def _infer_logaddexp_value_set(
    result_set: NumericValueSet,
    left_set: NumericValueSet,
    right_set: NumericValueSet,
) -> NumericValueSet:
    # The stable ufunc stays finite for finite inputs, including the largest floats.
    return _function_result(
        np.logaddexp, result_set, left_set, right_set,
        positive=left_set.allows_positive_infinity or right_set.allows_positive_infinity,
        negative=left_set.allows_negative_infinity and right_set.allows_negative_infinity,
        nan=left_set.allows_nan or right_set.allows_nan,
    )


@cache
def _infer_rounded_interval_value_set(
    mathematical_interval: sp.Interval,
    dtype_types: tuple[type[np.generic], ...],
) -> HomogeneousNumericValueSet:
    if not dtype_types:
        raise ValueError("'dtype_types' must contain at least one dtype.")

    if not isinstance(mathematical_interval, sp.Interval):
        raise TypeError("'mathematical_interval' must be a SymPy Interval.")

    if (
        mathematical_interval.start.is_finite is not True
        or mathematical_interval.end.is_finite is not True
    ):
        raise ValueError("The interval must have finite endpoints.")

    lower_bounds = []
    upper_bounds = []

    for dtype_type in dtype_types:
        dtype = np.dtype(dtype_type)

        if not np.issubdtype(dtype, np.floating):
            raise TypeError("Rounded interval bounds require floating-point dtypes.")

        dtype = cast(np.dtype[np.floating], dtype)
        precision = max(50, np.finfo(dtype).precision + 20)  # Use 20 extra decimal digits beyond the dtype's precision to reduce rounding error in the intermediate calculation.

        lower = dtype.type(
            str(sp.N(mathematical_interval.start, precision))  # Using text avoids first rounding the value to an ordinary Python float.
        )
        upper = dtype.type(
            str(sp.N(mathematical_interval.end, precision))
        )

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"The endpoints cannot be represented as finite values in {dtype}.")

        exact_lower = sp.Rational(*lower.as_integer_ratio())
        exact_upper = sp.Rational(*upper.as_integer_ratio())

        if exact_lower > mathematical_interval.start:
            lower = np.nextafter(lower, dtype.type(-np.inf))
            exact_lower = sp.Rational(*lower.as_integer_ratio())

        if exact_upper < mathematical_interval.end:
            upper = np.nextafter(upper, dtype.type(np.inf))
            exact_upper = sp.Rational(*upper.as_integer_ratio())

        lower_bounds.append(exact_lower)
        upper_bounds.append(exact_upper)

    return HomogeneousNumericValueSet(
        sympy_set=sp.Interval(
            min(lower_bounds),
            max(upper_bounds),
        ),
        dtype_types=dtype_types,
    )
