"""Common interface for sets of values used by random variables."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ._unknown import _UnknownValueSet


class ValueSet(ABC):
    """Common interface for numeric and object value sets."""

    dtype_types: tuple[type[np.generic], ...] | None

    @abstractmethod
    def contains(self, values: Any) -> bool | np.ndarray:
        """Return whether one or more values belong to this set."""


def __getattr__(name: str):
    """Keep the old import path working while classes live in own modules."""

    if name == "NumericValueSet":
        from .numeric_value_set import NumericValueSet

        return NumericValueSet

    if name == "ObjectValueSet":
        from .object_value_set import ObjectValueSet

        return ObjectValueSet

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ValueSet", "NumericValueSet", "ObjectValueSet", "_UnknownValueSet"]
