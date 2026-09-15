from numbers import Number
from typing import Iterable, Any

import numpy as np
import sympy as sp
from numpy._typing import NDArray

from problab.distributions.base import Distribution
from problab.validation._decorator import _validate_parameters
from problab.validation.distributions.discrete._categorical import _validate_categories, _validate_probabilities, \
    _validate_categorical_configuration
from problab.value_sets.base import ValueSet


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

    @_validate_parameters(
        categories=_validate_categories,
        probabilities=_validate_probabilities,
    )
    def __init__(self,
                 categories: Iterable[Any],
                 probabilities: Iterable[float],
                 ) -> None:

        categories = tuple(categories)
        probabilities = tuple(probabilities)

        _validate_categorical_configuration(categories, probabilities)

        total_probability = sum(probabilities)
        self._probabilities = tuple(
            probability / total_probability
            for probability in probabilities
        )

        sympy_set = sp.FiniteSet(*categories)

        self._category_inputs = categories
        self._categories = self._prepare_category_values(categories)

        self._value_set = ValueSet(
            sympy_set=sympy_set,
            dtype_types=(self._categories.dtype.type,),
        )

        super().__init__(parameters=None)

    @property
    def categories(self) -> tuple[Any, ...]:
        return self._category_inputs

    @property
    def probabilities(self) -> tuple[float, ...]:
        return self._probabilities

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


