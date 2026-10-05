import dataclasses
from typing import Callable

import numpy as np

from problab.operations._base import (
    _Operation,
    _REAL_NUMERIC_INPUT_TYPES,
)
from problab.value_sets import _realization_inference as realization
from problab.value_sets.base import NumericValueSet


@dataclasses.dataclass(frozen=True)
class _FunctionOperation(_Operation):
    infer_realization_value_set: Callable[..., NumericValueSet] | None = (
        dataclasses.field(default=None, kw_only=True)
    )


# Function operations
_EXP = _FunctionOperation(
    operation=np.exp,
    infer_realization_value_set=realization._infer_exp_value_set,
    name_func=lambda x: f"exp({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_LOG = _FunctionOperation(
    operation=np.log,
    infer_realization_value_set=realization._infer_log_value_set,
    name_func=lambda x: f"log({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_SQRT = _FunctionOperation(
    operation=np.sqrt,
    infer_realization_value_set=realization._infer_sqrt_value_set,
    name_func=lambda x: f"sqrt({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_SIN = _FunctionOperation(
    operation=np.sin,
    infer_realization_value_set=realization._infer_sin_value_set,
    name_func=lambda x: f"sin({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_ARCSIN = _FunctionOperation(
    operation=np.arcsin,
    infer_realization_value_set=realization._infer_arcsin_value_set,
    name_func=lambda x: f"arcsin({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_COS = _FunctionOperation(
    operation=np.cos,
    infer_realization_value_set=realization._infer_cos_value_set,
    name_func=lambda x: f"cos({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_ARCCOS = _FunctionOperation(
    operation=np.arccos,
    infer_realization_value_set=realization._infer_arccos_value_set,
    name_func=lambda x: f"arccos({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_TAN = _FunctionOperation(
    operation=np.tan,
    infer_realization_value_set=realization._infer_tan_value_set,
    name_func=lambda x: f"tan({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_ARCTAN = _FunctionOperation(
    operation=np.arctan,
    infer_realization_value_set=realization._infer_arctan_value_set,
    name_func=lambda x: f"arctan({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_SINH = _FunctionOperation(
    operation=np.sinh,
    infer_realization_value_set=realization._infer_sinh_value_set,
    name_func=lambda x: f"sinh({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_COSH = _FunctionOperation(
    operation=np.cosh,
    infer_realization_value_set=realization._infer_cosh_value_set,
    name_func=lambda x: f"cosh({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_TANH = _FunctionOperation(
    operation=np.tanh,
    infer_realization_value_set=realization._infer_tanh_value_set,
    name_func=lambda x: f"tanh({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_ARCSINH = _FunctionOperation(
    operation=np.arcsinh,
    infer_realization_value_set=realization._infer_arcsinh_value_set,
    name_func=lambda x: f"arcsinh({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_ARCCOSH = _FunctionOperation(
    operation=np.arccosh,
    infer_realization_value_set=realization._infer_arccosh_value_set,
    name_func=lambda x: f"arccosh({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_ARCTANH = _FunctionOperation(
    operation=np.arctanh,
    infer_realization_value_set=realization._infer_arctanh_value_set,
    name_func=lambda x: f"arctanh({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_LOG2 = _FunctionOperation(
    operation=np.log2,
    infer_realization_value_set=realization._infer_log2_value_set,
    name_func=lambda x: f"log2({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_LOG10 = _FunctionOperation(
    operation=np.log10,
    infer_realization_value_set=realization._infer_log10_value_set,
    name_func=lambda x: f"log10({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_FLOOR = _FunctionOperation(
    operation=np.floor,
    infer_realization_value_set=realization._infer_floor_value_set,
    name_func=lambda x: f"floor({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_CEIL = _FunctionOperation(
    operation=np.ceil,
    infer_realization_value_set=realization._infer_ceil_value_set,
    name_func=lambda x: f"ceil({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_SIGN = _FunctionOperation(
    operation=np.sign,
    infer_realization_value_set=realization._infer_sign_value_set,
    name_func=lambda x: f"sign({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_LOG1P = _FunctionOperation(
    operation=np.log1p,
    infer_realization_value_set=realization._infer_log1p_value_set,
    name_func=lambda x: f"log1p({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_EXPM1 = _FunctionOperation(
    operation=np.expm1,
    infer_realization_value_set=realization._infer_expm1_value_set,
    name_func=lambda x: f"expm1({x})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_HYPOT = _FunctionOperation(
    operation=np.hypot,
    infer_realization_value_set=realization._infer_hypot_value_set,
    name_func=lambda x, y: f"hypot({x}, {y})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)

_LOGADDEXP = _FunctionOperation(
    operation=np.logaddexp,
    infer_realization_value_set=realization._infer_logaddexp_value_set,
    name_func=lambda x, y: f"logaddexp({x}, {y})",
    supported_input_types=_REAL_NUMERIC_INPUT_TYPES,
)
