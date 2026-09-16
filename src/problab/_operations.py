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

def _divide_values(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(x, y)
def _modulo_values(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.mod(x, y)

    return np.where(y == 0, np.nan, result)

def _power_values(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.issubdtype(x.dtype, np.integer) and np.any(y < 0):
        x = x.astype(float)

    return np.emath.power(x, y)


# Arithmetic operations
_ADD         = _ArithmeticOperation(operation=operator.add, name_func=lambda a, b: f"({a} + {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_add_value_set)
_SUBTRACT    = _ArithmeticOperation(operation=operator.sub, name_func=lambda a, b: f"({a} - {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_subtract_value_set)
_MULTIPLY    = _ArithmeticOperation(operation=operator.mul, name_func=lambda a, b: f"({a} * {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_multiply_value_set)
_DIVIDE      = _ArithmeticOperation(operation=_divide_values, name_func=lambda a, b: f"({a} / {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_divide_value_set)
_MODULO      = _ArithmeticOperation(operation=_modulo_values, name_func=lambda a, b: f"({a} mod {b})", valid_input_value_set=REALS, infer_output_value_set=_infer_modulo_value_set)
_POWER       = _ArithmeticOperation(operation=_power_values, name_func=lambda a, b: f"({a} ** {b})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_power_value_set)
_NEGATIVE    = _ArithmeticOperation(operation=operator.neg, name_func=lambda x: f"(-{x})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_negative_value_set)
_ABS         = _ArithmeticOperation(operation=operator.abs, name_func=lambda x: f"abs({x})", valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_absolute_value_set)

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
