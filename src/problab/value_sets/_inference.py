from functools import wraps
from itertools import product
from typing import Callable

import numpy as np
import sympy as sp

from problab.value_sets._utils import is_known_subset
from problab.value_sets.base import NumericValueSet, ValueSet
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets._unknown import _UnknownValueSet
from problab.value_sets.sets import COMPLEXES, INTEGERS, UNKNOWN_VALUE_SET

_BUILTIN_DTYPES = tuple(np.dtype(code) for code in np.typecodes["All"])


def _input_dtype_candidates(
    inputs: tuple[ValueSet, ...],
) -> tuple[tuple[np.dtype, ...], ...] | None:
    input_dtypes = []

    for value_set in inputs:
        if value_set.dtype_types is None:
            return None

        candidates = []

        for allowed in value_set.dtype_types:
            matches = [
                dtype
                for dtype in _BUILTIN_DTYPES
                if dtype.kind in "biufcO"
                and np.issubdtype(dtype, allowed)
            ]

            if not matches:
                return None

            candidates.extend(matches)

        if not candidates:
            return None

        unique_candidates = {
            dtype.type: dtype
            for dtype in candidates
        }
        input_dtypes.append(tuple(unique_candidates.values()))

    return tuple(input_dtypes)


def _infer_dtype_types(
    operation: np.ufunc | None,
    inputs: tuple[ValueSet, ...],
) -> tuple[type[np.generic], ...] | None:

    if operation is None:
        return None

    input_dtypes = _input_dtype_candidates(inputs)
    if input_dtypes is None:
        return None

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
    def decorate(infer: Callable[..., NumericValueSet]) -> Callable[..., NumericValueSet]:
        @wraps(infer)
        def wrapped(*operands: NumericValueSet, **named_operands: NumericValueSet) -> NumericValueSet:
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
            return HomogeneousNumericValueSet(sympy_set=sympy_set, dtype_types=dtype_types)

        return wrapped

    return decorate


