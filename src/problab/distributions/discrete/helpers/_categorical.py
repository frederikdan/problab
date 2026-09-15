from numbers import Number
from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from problab.value_sets.numeric_value_set import NumericValueSet
from problab.value_sets.object_value_set import ObjectValueSet
from problab.value_sets.base import ValueSet


def _is_numeric_category(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_, Number))


def _build_untyped_category_configuration(
        categories: tuple[Any, ...],
) -> tuple[NDArray[Any], ObjectValueSet]:

    values = np.empty(len(categories), dtype=object)

    # Assign one item at a time so tuples and lists stay single objects.
    for index, category in enumerate(categories):
        values[index] = category

    values.flags.writeable = False

    value_set = ObjectValueSet(objects=categories)

    return values, value_set


def _infer_categorical_configuration(
        categories: tuple[Any, ...],
) -> tuple[NDArray[Any], ValueSet]:

    if not all(_is_numeric_category(category) for category in categories):
        return _build_untyped_category_configuration(categories)

    try:
        dtype = np.result_type(
            *(np.asarray(category).dtype for category in categories)
        )

        if dtype.kind == "O":
            return _build_untyped_category_configuration(categories)

        symbolic_categories = tuple(
            sp.sympify(category)
            for category in categories
        )

        sympy_set = sp.FiniteSet(*symbolic_categories)
        values = np.asarray(categories, dtype=dtype)

    except (TypeError, ValueError, sp.SympifyError):
        return _build_untyped_category_configuration(categories)

    values.flags.writeable = False

    value_set = NumericValueSet(
        sympy_set=sympy_set,
        dtype_types=(dtype.type,),
    )

    return values, value_set
