from decimal import Decimal
from numbers import Number
from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.mixed_numeric_value_set import MixedNumericValueSet
from problab.value_sets.object_value_set import ObjectValueSet
from problab.value_sets.base import ValueSet
from problab.value_sets._comparison import _objects_equal


def _is_numeric_category(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_, Number, Decimal))


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


def _build_mixed_numeric_category_configuration(
        categories: tuple[Any, ...],
) -> tuple[NDArray[Any], MixedNumericValueSet]:

    values = np.empty(len(categories), dtype=object)

    for index, category in enumerate(categories):
        values[index] = category

    values.flags.writeable = False

    return values, MixedNumericValueSet(values=categories)


def _infer_categorical_configuration(
        categories: tuple[Any, ...],
) -> tuple[NDArray[Any], ValueSet]:

    if not all(_is_numeric_category(category) for category in categories):
        return _build_untyped_category_configuration(categories)

    if len({type(category) for category in categories}) > 1:
        return _build_mixed_numeric_category_configuration(categories)

    try:
        dtype = np.result_type(
            *(np.asarray(category).dtype for category in categories)
        )

        if dtype.kind == "O":
            return _build_mixed_numeric_category_configuration(categories)

        symbolic_categories = tuple(
            sp.sympify(category)
            for category in categories
        )

        sympy_set = sp.FiniteSet(*symbolic_categories)
        values = np.asarray(categories, dtype=dtype)

    except (TypeError, ValueError, sp.SympifyError):
        if all(_is_numeric_category(category) for category in categories):
            return _build_mixed_numeric_category_configuration(categories)
        return _build_untyped_category_configuration(categories)

    values.flags.writeable = False

    value_set = HomogeneousNumericValueSet(
        sympy_set=sympy_set,
        dtype_types=(dtype.type,),
    )

    return values, value_set

def _merge_equal_categories(
    categories: tuple[Any, ...],
    probabilities: tuple[float, ...],
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    merged_categories = []
    merged_probabilities = []

    for category, probability in zip(categories, probabilities):
        for index, existing in enumerate(merged_categories):
            if _objects_equal(category, existing):
                merged_probabilities[index] += probability
                break
        else:
            merged_categories.append(category)
            merged_probabilities.append(probability)

    return (
        tuple(merged_categories),
        tuple(merged_probabilities),
    )
