import dataclasses
import operator
from typing import Callable

import numpy as np

from problab.value_sets.base import NumericValueSet
from problab.value_sets._inference import (_infer_add_value_set,
                                               _infer_subtract_value_set,
                                               _infer_multiply_value_set,
                                               _infer_divide_value_set,
                                               _infer_floor_divide_value_set,
                                               _infer_modulo_value_set,
                                               _infer_power_value_set,
                                               _infer_negative_value_set,
                                               _infer_absolute_value_set)
from problab.value_sets.sets import COMPLEXES, REALS


@dataclasses.dataclass(frozen=True)
class _Operation:
    operation: Callable
    name_func: Callable[..., str]

@dataclasses.dataclass(frozen=True)
class _ArithmeticOperation(_Operation):
    infer_output_value_set: Callable[..., NumericValueSet]
    valid_input_value_set: NumericValueSet

@dataclasses.dataclass(frozen=True)
class _LogicalOperation(_Operation):
    pass

@dataclasses.dataclass(frozen=True)
class _ComparisonOperation(_Operation):
    pass

@dataclasses.dataclass(frozen=True)
class _FunctionOperation(_Operation):
    pass


def _apply_binary_arithmetic(operation, name, x, y):
    left = np.asarray(x)
    right = np.asarray(y)
    result_dtype = np.result_type(left, right)

    if not (
        np.issubdtype(left.dtype, np.integer)
        and np.issubdtype(right.dtype, np.integer)
        and np.issubdtype(result_dtype, np.integer)
    ):
        try:
            with np.errstate(over="raise"):
                return operation(x, y)
        except FloatingPointError as error:
            raise OverflowError(f"Integer or floating-point {name} overflowed.") from error

    result = operation(left.astype(object), right.astype(object))
    limits = np.iinfo(result_dtype)
    result = np.asarray(result, dtype=object)

    if np.any(result < limits.min) or np.any(result > limits.max):
        raise OverflowError(f"Integer {name} overflowed for dtype {result_dtype}.")

    return result.astype(result_dtype)


def _apply_unary_arithmetic(operation, name, x):
    values = np.asarray(x)

    if not np.issubdtype(values.dtype, np.integer):
        try:
            with np.errstate(over="raise"):
                return operation(x)
        except FloatingPointError as error:
            raise OverflowError(f"Integer or floating-point {name} overflowed.") from error

    result = np.asarray(operation(values.astype(object)), dtype=object)
    limits = np.iinfo(values.dtype)

    if np.any(result < limits.min) or np.any(result > limits.max):
        raise OverflowError(f"Integer {name} overflowed for dtype {values.dtype}.")

    return result.astype(values.dtype)


def _divide(x, y):
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="raise"):
            return np.divide(x, y)
    except FloatingPointError as error:
        raise OverflowError("Integer or floating-point division overflowed.") from error


def _add(x, y):
    return _apply_binary_arithmetic(np.add, "addition", x, y)


def _subtract(x, y):
    return _apply_binary_arithmetic(np.subtract, "subtraction", x, y)


def _multiply(x, y):
    return _apply_binary_arithmetic(np.multiply, "multiplication", x, y)


def _modulo(x, y):
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="raise"):
            result = np.mod(x, y)
    except FloatingPointError as error:
        raise OverflowError("Integer or floating-point modulo overflowed.") from error

    return np.where(y == 0, np.nan, result)

def _power(x, y, allow_complex: bool = True):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.issubdtype(x.dtype, np.integer) and np.any(y < 0):
        x = x.astype(float)

    if np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer):
        return _apply_binary_arithmetic(np.power, "power", x, y)

    try:
        with np.errstate(over="raise"):
            if allow_complex:
                return np.emath.power(x, y)
            else:
                return np.power(x, y)
    except FloatingPointError as error:
        raise OverflowError("Integer or floating-point power overflowed.") from error


def _real_power(x, y):
    return _power(x, y, allow_complex=False)


def _negative(x):
    return _apply_unary_arithmetic(np.negative, "negation", x)


def _absolute(x):
    return _apply_unary_arithmetic(np.absolute, "absolute value", x)



# Arithmetic operations
_ADD         = _ArithmeticOperation(operation=_add, name_func=lambda a, b: f"({a} + {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_add_value_set)
_SUBTRACT    = _ArithmeticOperation(operation=_subtract, name_func=lambda a, b: f"({a} - {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_subtract_value_set)
_MULTIPLY    = _ArithmeticOperation(operation=_multiply, name_func=lambda a, b: f"({a} * {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_multiply_value_set)
_DIVIDE      = _ArithmeticOperation(operation=_divide, name_func=lambda a, b: f"({a} / {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_divide_value_set)
_MODULO      = _ArithmeticOperation(operation=_modulo, name_func=lambda a, b: f"({a} mod {b})", valid_input_value_set=REALS, infer_output_value_set=_infer_modulo_value_set)
_POWER       = _ArithmeticOperation(operation=_power, name_func=lambda a, b: f"({a} ** {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_power_value_set)
_REAL_POWER  = _ArithmeticOperation(operation=_real_power,name_func=lambda a, b: f"({a} ** {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_power_value_set,)
_NEGATIVE    = _ArithmeticOperation(operation=_negative, name_func=lambda x: f"(-{x})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_negative_value_set)
_ABS         = _ArithmeticOperation(operation=_absolute, name_func=lambda x: f"abs({x})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_absolute_value_set)

# Logical operations
_AND         = _LogicalOperation(operation=operator.and_, name_func=lambda a, b: f"{{{a} & {b}}}")
_OR          = _LogicalOperation(operation=operator.or_, name_func=lambda a, b: f"{{{a} | {b}}}")
_XOR         = _LogicalOperation(operation=operator.xor, name_func=lambda a, b: f"{{{a} ^ {b}}}")
_INVERT      = _LogicalOperation(operation=operator.invert, name_func=lambda x: f"{{~{x}}}")

# Comparisons
_LT          = _ComparisonOperation(operation=operator.lt, name_func=lambda a, b: f"{{{a} < {b}}}")
_GT          = _ComparisonOperation(operation=operator.gt, name_func=lambda a, b: f"{{{a} > {b}}}")
_LTE         = _ComparisonOperation(operation=operator.le, name_func=lambda a, b: f"{{{a} <= {b}}}")
_GTE         = _ComparisonOperation(operation=operator.ge, name_func=lambda a, b: f"{{{a} >= {b}}}")
_EQ          = _ComparisonOperation(operation=operator.eq, name_func=lambda a, b: f"{{{a} = {b}}}")
_NEQ         = _ComparisonOperation(operation=operator.ne, name_func=lambda a, b: f"{{{a} != {b}}}")

# Function operations
_EXP        = _FunctionOperation(operation=np.exp, name_func=lambda x: f"exp({x})")
_LOG        = _FunctionOperation(operation=np.log, name_func=lambda x: f"log({x})")
_SQRT       = _FunctionOperation(operation=np.sqrt, name_func=lambda x: f"sqrt({x})")

_SIN        = _FunctionOperation(operation=np.sin, name_func=lambda x: f"sin({x})")
_ARCSIN     = _FunctionOperation(operation=np.arcsin, name_func=lambda x: f"arcsin({x})")
_COS        = _FunctionOperation(operation=np.cos, name_func=lambda x: f"cos({x})")
_ARCCOS     = _FunctionOperation(operation=np.arccos, name_func=lambda x: f"arccos({x})")
_TAN        = _FunctionOperation(operation=np.tan, name_func=lambda x: f"tan({x})")
_ARCTAN     = _FunctionOperation(operation=np.arctan, name_func=lambda x: f"arctan({x})")
_SINH       = _FunctionOperation(operation=np.sinh, name_func=lambda x: f"sinh({x})")
_COSH       = _FunctionOperation(operation=np.cosh, name_func=lambda x: f"cosh({x})")
_TANH       = _FunctionOperation(operation=np.tanh, name_func=lambda x: f"tanh({x})")
_ARCSINH    = _FunctionOperation(operation=np.arcsinh, name_func=lambda x: f"arcsinh({x})")
_ARCCOSH    = _FunctionOperation(operation=np.arccosh, name_func=lambda x: f"arccosh({x})")
_ARCTANH    = _FunctionOperation(operation=np.arctanh, name_func=lambda x: f"arctanh({x})")
_LOG2       = _FunctionOperation(operation=np.log2, name_func=lambda x: f"log2({x})")
_LOG10      = _FunctionOperation(operation=np.log10, name_func=lambda x: f"log10({x})")
_FLOOR      = _FunctionOperation(operation=np.floor, name_func=lambda x: f"floor({x})")
_CEIL       = _FunctionOperation(operation=np.ceil, name_func=lambda x: f"ceil({x})")
_SIGN       = _FunctionOperation(operation=np.sign, name_func=lambda x: f"sign({x})")

_LOG1P      = _FunctionOperation(operation=np.log1p, name_func=lambda x: f"log1p({x})")
_EXPM1      = _FunctionOperation(operation=np.expm1, name_func=lambda x: f"expm1({x})")
_HYPOT      = _FunctionOperation(operation=np.hypot, name_func=lambda x, y: f"hypot({x}, {y})")
_LOGADDEXP  = _FunctionOperation(operation=np.logaddexp, name_func=lambda x, y: f"logaddexp({x}, {y})")
