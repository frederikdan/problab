import numpy as np
import sympy as sp

from problab.value_sets.base import ValueSet
from problab.value_sets.base import NumericValueSet
from problab.value_sets.object_value_set import ObjectValueSet
from problab.value_sets._unknown import _UnknownValueSet


def is_known_subset(subset: ValueSet | sp.Set, superset: ValueSet | sp.Set) -> bool:

    if isinstance(subset, ObjectValueSet) or isinstance(superset, ObjectValueSet):
        return False

    subset_set = subset.sympy_set if isinstance(subset, NumericValueSet) else subset
    superset_set = superset.sympy_set if isinstance(superset, NumericValueSet) else superset

    if isinstance(subset_set, _UnknownValueSet) or isinstance(superset_set, _UnknownValueSet):
        return False

    return subset_set.is_subset(superset_set) is True


def validate_as_subset(values: np.ndarray,
                       target_set: ValueSet
                       ) -> None:

    if isinstance(target_set, ObjectValueSet):
        flat_values = np.asarray(values, dtype=object).reshape(-1)
        membership = np.asarray(target_set.contains(flat_values), dtype=bool).reshape(-1)
        for index, (value, is_member) in enumerate(zip(flat_values, membership)):
            if not is_member:
                raise ValueError(
                    f"Value at index {index}, {value!r}, is outside the target set {target_set}."
                )
        return

    if not isinstance(target_set, NumericValueSet):
        raise TypeError("'target_set' must be a ValueSet.")

    if isinstance(target_set.sympy_set, _UnknownValueSet):
        raise ValueError("Cannot validate membership: the target set is unknown.")

    for index, value in enumerate(values):
        try:
            if isinstance(value, (float, np.floating)) and np.isfinite(value) and value == np.trunc(value):
                symbolic_value = sp.Integer(int(value))
            else:
                symbolic_value = sp.sympify(value)

            result = target_set.sympy_set.contains(symbolic_value)
        except (TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(f"Could not validate value at index {index}: {value!r} against {target_set}.") from error

        if result is sp.false:
            raise ValueError(f"Value at index {index}, {value!r}, is outside the target set {target_set}.")

        if result is not sp.true:
            raise ValueError(f"Could not determine membership for value at index {index}: {value!r} in {target_set}.")
