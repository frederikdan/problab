import numpy as np

from problab.operations._function import _FunctionOperation
from problab.operations._base import _NUMERIC_INPUT_TYPES

# Statistical operations
_MEAN = _FunctionOperation(
    operation=np.mean,
    name_func=lambda x: f"mean({x})",
    supported_input_types=_NUMERIC_INPUT_TYPES,
)

_VARIANCE = _FunctionOperation(
    operation=np.var,
    name_func=lambda x: f"variance({x})",
    supported_input_types=_NUMERIC_INPUT_TYPES,
)

_STD = _FunctionOperation(
    operation=np.std,
    name_func=lambda x: f"std({x})",
    supported_input_types=_NUMERIC_INPUT_TYPES,
)
