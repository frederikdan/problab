from numbers import Real, Number
from typing import Iterable, Any

import numpy as np
import sympy as sp
from numpy._typing import NDArray

from src.problab.distributions.base import Distribution
from src.problab.random_variables.context import RealizationContext
from src.problab.value_sets.base import ValueSet


class CategoricalDistribution(Distribution):

    @staticmethod
    def _prepare_category_values(categories: tuple[Any, ...]) -> NDArray[Any]:

        first_category = categories[0]
        first_category_is_numeric = isinstance(first_category, Number)
        all_categories_are_same_type = all(type(c) is type(first_category) for c in categories)

        if first_category_is_numeric and all_categories_are_same_type:
            category_values = np.asarray(categories)
        else:
            category_values = np.empty(len(categories), dtype=object)
            category_values[:] = categories

        category_values.flags.writeable = False
        return category_values

    def __init__(self,
                 categories: Iterable[Any],
                 probabilities: Iterable[float],
                 ) -> None:

        categories = tuple(categories)
        self._probabilities = tuple(probabilities)

        if len(categories) == 0:
            raise ValueError("'categories' must contain at least one value.")

        if len(categories) != len(self._probabilities):
            raise ValueError("'categories' and 'probabilities' must have the same length.")

        if not all(isinstance(p, Real) for p in self._probabilities):
            raise TypeError("'probabilities' must contain only real numbers.")

        if not all(np.isfinite(p) for p in self._probabilities):
            raise ValueError("'probabilities' must be finite.")

        if any(p < 0 for p in self._probabilities):
            raise ValueError("'probabilities' cannot contain negative values.")

        if not np.isclose(sum(self._probabilities), 1.0):
            raise ValueError("'probabilities' must sum to 1.")

        sympy_set = sp.FiniteSet(*categories)

        self._categories = self._prepare_category_values(categories)

        self._value_set = ValueSet(
            sympy_set=sympy_set,
            dtype_types=(self._categories.dtype.type,),
        )

        super().__init__(parameters=None)

    @property
    def value_set(self) -> ValueSet:
        return self._value_set

    def _sample(self,
                *parameters: np.ndarray,
                num_samples: int,
                rng: np.random.Generator
                ) -> NDArray[Any]:

        indices = rng.choice(
            len(self._categories),
            size=num_samples,
            p=self._probabilities,
        )

        return self._categories[indices]


