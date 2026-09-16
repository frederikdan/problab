"""Value sets for categorical values treated as opaque Python objects."""

import dataclasses
from typing import Any

import numpy as np

from ._comparison import _objects_equal
from .base import ValueSet





@dataclasses.dataclass(frozen=True)
class ObjectValueSet(ValueSet):

    objects: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.objects, tuple):
            raise TypeError("'objects' must be a tuple.")
        if len(self.objects) == 0:
            raise ValueError("'objects' must contain at least one value.")

    @property
    def dtype_types(self) -> tuple[type[np.generic], ...]:
        return (np.object_,)

    def contains(self, values: Any) -> bool | np.ndarray:
        array = np.asarray(values, dtype=object)
        scalar = array.ndim == 0
        flat_values = array.reshape(-1)
        result = np.asarray(
            [any(_objects_equal(value, candidate) for candidate in self.objects)
             for value in flat_values],
            dtype=bool,
        )

        if scalar:
            return bool(result[0])
        return result.reshape(array.shape)
