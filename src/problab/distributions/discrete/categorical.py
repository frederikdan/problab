from functools import cached_property
from typing import Iterable, Any

import numpy as np
from numpy._typing import NDArray

from problab.value_sets._utils import _detect_non_finite_values
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.distributions.base import Distribution
from problab.distributions.discrete.helpers._categorical import _prepare_categories_and_mathematical_value_set, \
    _merge_equal_categories, _attempt_make_categories_numeric
from problab.validation._decorator import _validate_parameters
from problab.validation.distributions.discrete._categorical import _validate_categories, _validate_probabilities, \
    _validate_categorical_configuration
from problab.value_sets.base import ValueSet, NumericValueSet


class CategoricalDistribution(Distribution):


    @_validate_parameters(
        categories=_validate_categories,
        probabilities=_validate_probabilities,
    )
    def __init__(self,
                 categories: Iterable[Any],
                 probabilities: Iterable[float]
                 ) -> None:

        categories = tuple(categories)
        probabilities = tuple(probabilities)

        _validate_categorical_configuration(categories, probabilities)

        categories, probabilities = _merge_equal_categories(categories,probabilities)

        total_probability = sum(probabilities)

        self._probabilities = tuple(
            probability / total_probability
            for probability in probabilities
        )

        self._category_inputs = categories

        self._categories, self._mathematical_value_set = _prepare_categories_and_mathematical_value_set(categories)


        self._categories_numeric = _attempt_make_categories_numeric(categories, self._mathematical_value_set)

        super().__init__(parameters=None, symbol="Categorical")

    @property
    def categories(self) -> tuple[Any, ...]:
        return self._category_inputs

    @property
    def probabilities(self) -> tuple[float, ...]:
        return self._probabilities

    @property
    def value_set(self) -> ValueSet:
        return self._mathematical_value_set

    @cached_property
    def _realization_value_set(self) -> ValueSet:
        if not isinstance(self._mathematical_value_set, NumericValueSet):
            return self._mathematical_value_set

        categories = self._categories_numeric
        if categories is None:
            categories = self._categories

        positive_infinity, negative_infinity, nan = (
            _detect_non_finite_values(categories)
        )

        return HomogeneousNumericValueSet(
            sympy_set=self._mathematical_value_set.sympy_set,
            dtype_types=(categories.dtype.type,),
            allows_positive_infinity=positive_infinity,
            allows_negative_infinity=negative_infinity,
            allows_nan=nan,
        )

    def _sample(self,
                *parameters: np.ndarray,
                num_samples: int,
                rng: np.random.Generator
                ) -> NDArray[Any]:

        categories = self._categories_numeric

        if categories is None:
            categories = self._categories

        indices = rng.choice(
            len(categories),
            size=num_samples,
            p=self._probabilities,
        )

        return categories[indices]


