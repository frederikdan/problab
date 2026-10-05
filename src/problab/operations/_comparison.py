import dataclasses
import operator

import numpy as np

from problab.operations._base import _Operation


@dataclasses.dataclass(frozen=True)
class _ComparisonOperation(_Operation):
    pass


# Comparisons
_LT = _ComparisonOperation(
    operation=operator.lt,
    name_func=lambda a, b: f"{{{a} < {b}}}",
    supported_input_types=((np.integer, np.floating, np.object_),),
)

_GT = _ComparisonOperation(
    operation=operator.gt,
    name_func=lambda a, b: f"{{{a} > {b}}}",
    supported_input_types=((np.integer, np.floating, np.object_),),
)

_LTE = _ComparisonOperation(
    operation=operator.le,
    name_func=lambda a, b: f"{{{a} <= {b}}}",
    supported_input_types=((np.integer, np.floating, np.object_),),
)

_GTE = _ComparisonOperation(
    operation=operator.ge,
    name_func=lambda a, b: f"{{{a} >= {b}}}",
    supported_input_types=((np.integer, np.floating, np.object_),),
)

_EQ = _ComparisonOperation(
    operation=operator.eq,
    name_func=lambda a, b: f"{{{a} = {b}}}",
)

_NEQ = _ComparisonOperation(
    operation=operator.ne,
    name_func=lambda a, b: f"{{{a} != {b}}}",
)
