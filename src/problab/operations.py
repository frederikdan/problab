import dataclasses
import operator
from typing import Callable

import numpy as np

from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.inference import (_infer_add_value_set,
                                              _infer_subtract_value_set,
                                              _infer_multiply_value_set,
                                              _infer_divide_value_set,
                                              _infer_floor_divide_value_set,
                                              _infer_modulo_value_set,
                                              _infer_power_value_set,
                                              _infer_negative_value_set,
                                              _infer_absolute_value_set)
from src.problab.value_sets.sets import COMPLEXES, REALS


@dataclasses.dataclass(frozen=True)
class Operation:
    operation: Callable
    name_func: Callable[..., str]

@dataclasses.dataclass(frozen=True)
class ArithmeticOperation(Operation):
    infer_output_value_set: Callable[..., ValueSet]
    valid_input_value_set: ValueSet

@dataclasses.dataclass(frozen=True)
class LogicalOperation(Operation):
    pass

@dataclasses.dataclass(frozen=True)
class ComparisonOperation(Operation):
    pass

def _DIVIDE(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(x, y)
def _MODULO(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.mod(x, y)

    return np.where(y == 0, np.nan, result)

def _POWER(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.issubdtype(x.dtype, np.integer) and np.any(y < 0):
        x = x.astype(float)

    return np.emath.power(x, y)


# Arithmetic operations
ADD         = ArithmeticOperation(operation=operator.add,   name_func=lambda a, b: f"({a} + {b})",      valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_add_value_set)
SUBTRACT    = ArithmeticOperation(operation=operator.sub,   name_func=lambda a, b: f"({a} - {b})",      valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_subtract_value_set)
MULTIPLY    = ArithmeticOperation(operation=operator.mul,   name_func=lambda a, b: f"({a} * {b})",      valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_multiply_value_set)
DIVIDE      = ArithmeticOperation(operation=_DIVIDE,        name_func=lambda a, b: f"({a} / {b})",      valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_divide_value_set)
MODULO      = ArithmeticOperation(operation=_MODULO,        name_func=lambda a, b: f"({a} mod {b})",    valid_input_value_set=REALS,     infer_output_value_set=_infer_modulo_value_set)
POWER       = ArithmeticOperation(operation=_POWER,         name_func=lambda a, b: f"({a} ** {b})",     valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_power_value_set)
NEGATIVE    = ArithmeticOperation(operation=operator.neg,   name_func=lambda x: f"(-{x})",              valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_negative_value_set)
ABS         = ArithmeticOperation(operation=operator.abs,   name_func=lambda x: f"abs({x})",            valid_input_value_set=COMPLEXES, infer_output_value_set=_infer_absolute_value_set)

# Logical operations
AND         = LogicalOperation(operation=operator.and_,     name_func=lambda a, b: f"{{{a} & {b}}}")
OR          = LogicalOperation(operation=operator.or_,      name_func=lambda a, b: f"{{{a} | {b}}}")
XOR         = LogicalOperation(operation=operator.xor,      name_func=lambda a, b: f"{{{a} ^ {b}}}")
INVERT      = LogicalOperation(operation=operator.invert,   name_func=lambda x: f"{{~{x}}}")

# Comparisons
LT          = ComparisonOperation(operation=operator.lt,    name_func=lambda a, b: f"{{{a} < {b}}}")
GT          = ComparisonOperation(operation=operator.gt,    name_func=lambda a, b: f"{{{a} > {b}}}")
LTE         = ComparisonOperation(operation=operator.le,    name_func=lambda a, b: f"{{{a} <= {b}}}")
GTE         = ComparisonOperation(operation=operator.ge,    name_func=lambda a, b: f"{{{a} >= {b}}}")
EQ          = ComparisonOperation(operation=operator.eq,    name_func=lambda a, b: f"{{{a} = {b}}}")
NEQ         = ComparisonOperation(operation=operator.ne,    name_func=lambda a, b: f"{{{a} != {b}}}")
