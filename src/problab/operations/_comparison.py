import dataclasses
import operator

import numpy as np

from problab.operations._base import _Operation
from problab.value_sets._comparison import _objects_equal


@dataclasses.dataclass(frozen=True)
class _ComparisonOperation(_Operation):
    pass


def _equal(left, right):
    left = np.asarray(left)
    right = np.asarray(right)

    if left.dtype.kind != "O" and right.dtype.kind != "O":
        return np.equal(left, right)

    left, right = np.broadcast_arrays(left, right)

    return np.fromiter(
        (
            _objects_equal(left_value, right_value)
            for left_value, right_value in zip(left.flat, right.flat)
        ),
        dtype=bool,
        count=left.size,
    ).reshape(left.shape)


def _not_equal(left, right):
    return np.logical_not(_equal(left, right))


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
    operation=_equal,
    name_func=lambda a, b: f"{{{a} = {b}}}",
)

_NEQ = _ComparisonOperation(
    operation=_not_equal,
    name_func=lambda a, b: f"{{{a} != {b}}}",
)

