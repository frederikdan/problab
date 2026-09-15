from typing import Iterable, Any

import numpy as np
from numpy._typing import NDArray

from problab.distributions.base import Distribution
from problab.distributions.discrete.helpers._categorical import _infer_categorical_configuration
from problab.validation._decorator import _validate_parameters
from problab.validation.distributions.discrete._categorical import _validate_categories, _validate_probabilities, \
    _validate_categorical_configuration
from problab.value_sets.base import ValueSet


class CategoricalDistribution(Distribution):

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

        _validate_categorical_configuration(
            categories,
            probabilities,
        )

        total_probability = sum(probabilities)

        self._probabilities = tuple(
            probability / total_probability
            for probability in probabilities
        )

        self._category_inputs = categories

        self._categories, self._value_set = (
            _infer_categorical_configuration(categories)
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


