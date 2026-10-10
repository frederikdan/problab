import dataclasses
import warnings
from typing import Callable

import numpy as np

from problab.operations._base import (
    _Operation,
    _NUMERIC_INPUT_TYPES,
    _REAL_NUMERIC_INPUT_TYPES,
)
from problab.value_sets.base import NumericValueSet
from problab.value_sets.sets import COMPLEXES, REALS
from problab.value_sets import _mathematical_inference as mathematical
from problab.value_sets import _realization_inference as realization


@dataclasses.dataclass(frozen=True)
class _ArithmeticOperation(_Operation):
    infer_mathematical_value_set: Callable[..., NumericValueSet]
    infer_realization_value_set: Callable[..., NumericValueSet]
    valid_input_value_set: NumericValueSet


def _convert_integer_result(
    result: np.ndarray,
    result_dtype: np.dtype,
    operation_name: str,
) -> np.ndarray:

    result = np.asarray(result, dtype=object)
    limits = np.iinfo(result_dtype)

    below_minimum = result < limits.min
    above_maximum = result > limits.max
    overflowed = below_minimum | above_maximum

    if not np.any(overflowed):
        return result.astype(result_dtype)

    policy = np.geterr()["over"]
    message = f"Integer {operation_name} overflowed for dtype {result_dtype}."

    if policy == "raise":
        raise OverflowError(message)

    if policy == "warn":
        warnings.warn(
            f"{message} The result array is converted to float64, "
            "with overflowing entries replaced by signed infinity; "
            "other large integers may be rounded.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif policy != "ignore":
        raise OverflowError(message)

    converted = np.empty(result.shape, dtype=np.float64)
    converted[below_minimum] = -np.inf
    converted[above_maximum] = np.inf
    converted[~overflowed] = result[~overflowed].astype(np.float64)

    return converted


def _apply_binary_arithmetic(operation, name, x, y):
    left = np.asarray(x)
    right = np.asarray(y)
    result_dtype = np.result_type(left, right)

    if not (
        np.issubdtype(left.dtype, np.integer)
        and np.issubdtype(right.dtype, np.integer)
        and np.issubdtype(result_dtype, np.integer)
    ):
        return operation(x, y)

    result = operation(left.astype(object), right.astype(object))

    return _convert_integer_result(
        result=result,
        result_dtype=result_dtype,
        operation_name=name,
    )

def _apply_unary_arithmetic(operation, name, x):
    values = np.asarray(x)

    if not np.issubdtype(values.dtype, np.integer):
        return operation(x)

    result = operation(values.astype(object))

    return _convert_integer_result(
        result=result,
        result_dtype=values.dtype,
        operation_name=name,
    )

def _divide(x, y):
    return np.divide(x, y)


def _add(x, y):
    return _apply_binary_arithmetic(np.add, "addition", x, y)


def _subtract(x, y):
    return _apply_binary_arithmetic(np.subtract, "subtraction", x, y)


def _multiply(x, y):
    return _apply_binary_arithmetic(np.multiply, "multiplication", x, y)


def _modulo(x, y):
    result = np.asarray(np.mod(x, y))
    zero_divisors = np.broadcast_to(np.equal(y, 0), result.shape)

    if not np.any(zero_divisors):
        return result

    if result.dtype.kind in "iu":
        dtype = np.float64

        if np.any(result > 2**53) or (
            result.dtype.kind == "i"
            and np.any(result < -(2**53))
        ):
            dtype = object

        result = result.astype(dtype)

    result[zero_divisors] = np.nan
    return result


def _power(x, y, allow_complex: bool = True):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.issubdtype(x.dtype, np.integer) and np.any(y < 0):
        x = x.astype(float)

    if np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer):
        return _apply_binary_arithmetic(np.power, "power", x, y)

    if allow_complex:
        return np.emath.power(x, y)

    return np.power(x, y)


def _real_power(x, y):
    return _power(x, y, allow_complex=False)


def _negative(x):
    return _apply_unary_arithmetic(np.negative, "negation", x)


def _absolute(x):
    return _apply_unary_arithmetic(np.absolute, "absolute value", x)


# Arithmetic operations

_ADD = _ArithmeticOperation(
    operation=_add,
    name_func=lambda a, b: f"({a} + {b})",
    valid_input_value_set=COMPLEXES,
    infer_mathematical_value_set=mathematical._infer_add_value_set,
    infer_realization_value_set=realization._infer_add_value_set,
    supported_input_types=_NUMERIC_INPUT_TYPES,
)

_SUBTRACT = _ArithmeticOperation(
    operation=_subtract,
    name_func=lambda a, b: f"({a} - {b})",
    valid_input_value_set=COMPLEXES,
    infer_mathematical_value_set=mathematical._infer_subtract_value_set,
    infer_realization_value_set=realization._infer_subtract_value_set,
    supported_input_types=_NUMERIC_INPUT_TYPES,
)

_MULTIPLY = _ArithmeticOperation(
    operation=_multiply,
    name_func=lambda a, b: f"({a} * {b})",
    valid_input_value_set=COMPLEXES,
    infer_mathematical_value_set=mathematical._infer_multiply_value_set,
    infer_realization_value_set=realization._infer_multiply_value_set,
    supported_input_types=_NUMERIC_INPUT_TYPES,
)

_DIVIDE = _ArithmeticOperation(
    operation=_divide,
    name_func=lambda a, b: f"({a} / {b})",
    valid_input_value_set=COMPLEXES,
    infer_mathematical_value_set=mathematical._infer_divide_value_set,
    infer_realization_value_set=realization._infer_divide_value_set,
    supported_input_types=_NUMERIC_INPUT_TYPES,
)

_MODULO = _ArithmeticOperation(
    operation=_modulo,
    name_func=lambda a, b: f"({a} mod {b})",
    valid_input_value_set=REALS,
    infer_mathematical_value_set=mathematical._infer_modulo_value_set,
    infer_realization_value_set=realization._infer_modulo_value_set,
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_POWER = _ArithmeticOperation(
    operation=_power,
    name_func=lambda a, b: f"({a} ** {b})",
    valid_input_value_set=COMPLEXES,
    infer_mathematical_value_set=mathematical._infer_power_value_set,
    infer_realization_value_set=realization._infer_power_value_set,
    supported_input_types=_NUMERIC_INPUT_TYPES,
)

_REAL_POWER = _ArithmeticOperation(
    operation=_real_power,
    name_func=lambda a, b: f"({a} ** {b})",
    valid_input_value_set=COMPLEXES,
    infer_mathematical_value_set=mathematical._infer_power_value_set,
    infer_realization_value_set=realization._infer_real_power_value_set,
    supported_input_types=_NUMERIC_INPUT_TYPES,
)

_NEGATIVE = _ArithmeticOperation(
    operation=_negative,
    name_func=lambda x: f"(-{x})",
    valid_input_value_set=COMPLEXES,
    infer_mathematical_value_set=mathematical._infer_negative_value_set,
    infer_realization_value_set=realization._infer_negative_value_set,
    supported_input_types=_NUMERIC_INPUT_TYPES,
)

_ABS = _ArithmeticOperation(
    operation=_absolute,
    name_func=lambda x: f"abs({x})",
    valid_input_value_set=COMPLEXES,
    infer_mathematical_value_set=mathematical._infer_absolute_value_set,
    infer_realization_value_set=realization._infer_absolute_value_set,
    supported_input_types=_NUMERIC_INPUT_TYPES,
)
