from decimal import Decimal
from numbers import Number
from typing import Any

import numpy as np
import sympy as sp

from problab.value_sets.base import ValueSet
from problab.value_sets.base import NumericValueSet
from problab.value_sets.object_value_set import ObjectValueSet
from problab.value_sets._unknown import _UnknownValueSet


def _to_sympy_value(value: Any) -> sp.Basic:
    if not isinstance(value, (bool, np.bool_, Number, Decimal)):
        raise TypeError("Only numeric and Boolean values can be converted.")

    try:
        if (
            isinstance(value, (float, np.floating))
            and np.isfinite(value)
            and value == np.trunc(value)
        ):
            return sp.Integer(int(value))

        return sp.sympify(value)

    except (TypeError, ValueError, NotImplementedError, sp.SympifyError) as error:
        raise ValueError(
            f"Could not convert value {value!r} to SymPy."
        ) from error

def is_known_subset(subset: ValueSet | sp.Set, superset: ValueSet | sp.Set) -> bool:

    if isinstance(subset, ObjectValueSet) or isinstance(superset, ObjectValueSet):
        return False

    subset_set = subset.sympy_set if isinstance(subset, NumericValueSet) else subset
    superset_set = superset.sympy_set if isinstance(superset, NumericValueSet) else superset

    if isinstance(subset_set, _UnknownValueSet) or isinstance(superset_set, _UnknownValueSet):
        return False

    return subset_set.is_subset(superset_set) is True


def validate_as_subset(
    values: np.ndarray,
    target_set: ValueSet,
) -> None:
    if isinstance(target_set, ObjectValueSet):
        flat_values = np.asarray(values, dtype=object).reshape(-1)
        membership = np.asarray(
            target_set.contains(flat_values),
            dtype=bool,
        ).reshape(-1)

        for index, (value, is_member) in enumerate(
            zip(flat_values, membership)
        ):
            if not is_member:
                raise ValueError(
                    f"Value at index {index}, {value!r}, "
                    f"is outside the target set {target_set}."
                )

        return

    if not isinstance(target_set, NumericValueSet):
        raise TypeError("'target_set' must be a ValueSet.")

    for index, value in enumerate(values):
        non_finite_result = _contains_non_finite_value(value, target_set)

        if non_finite_result is not None:
            if not non_finite_result:
                raise ValueError(
                    f"Value at index {index}, {value!r}, "
                    f"is outside the target set {target_set}."
                )
            continue

        if isinstance(target_set.sympy_set, _UnknownValueSet):
            raise ValueError(
                "Cannot validate membership: the target set is unknown."
            )

        try:
            symbolic_value = _to_sympy_value(value)
            result = target_set.sympy_set.contains(symbolic_value)

        except (TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                f"Could not validate value at index {index}: "
                f"{value!r} against {target_set}."
            ) from error

        if result is sp.false:
            raise ValueError(
                f"Value at index {index}, {value!r}, "
                f"is outside the target set {target_set}."
            )

        if result is not sp.true:
            raise ValueError(
                f"Could not determine membership for value at index "
                f"{index}: {value!r} in {target_set}."
            )


def _contains_non_finite_value(
    value: Any,
    target_set: NumericValueSet,
) -> bool | None:

    if not isinstance(
        value,
        (float, complex, np.floating, np.complexfloating),
    ):
        return None

    components = (
        (value.real, value.imag)
        if isinstance(value, (complex, np.complexfloating))
        else (value,)
    )

    found_non_finite = False

    for component in components:
        if np.isnan(component):
            allowed = target_set.allows_nan
        elif np.isposinf(component):
            allowed = target_set.allows_positive_infinity
        elif np.isneginf(component):
            allowed = target_set.allows_negative_infinity
        else:
            continue

        found_non_finite = True

        if not allowed:
            return False

    return True if found_non_finite else None


def _detect_non_finite_values(
    values: np.ndarray,
) -> tuple[bool, bool, bool]:

    positive_infinity = False
    negative_infinity = False
    nan = False

    for value in values.reshape(-1):
        components = (
            (value.real, value.imag)
            if isinstance(value, (complex, np.complexfloating))
            else (value,)
        )

        for component in components:
            if not isinstance(component, (float, np.floating)):
                continue

            positive_infinity |= bool(np.isposinf(component))
            negative_infinity |= bool(np.isneginf(component))
            nan |= bool(np.isnan(component))

    return positive_infinity, negative_infinity, nan