from numbers import Real

import numpy as np
from scipy.stats import poisson

from problab.distributions.base import Distribution
from problab.random_variables.base import RandomVariable
from problab.validation._decorator import _validate_parameters
from problab.validation.distributions.discrete._poisson import _validate_poisson_mu
from problab.value_sets.numeric_value_set import NumericValueSet
from problab.value_sets.sets import NATURALS_0

class PoissonDistribution(Distribution):

    @_validate_parameters(
        mu=_validate_poisson_mu,
    )
    def __init__(
            self,
            mu: RandomVariable | Real,
        ) -> None:

        self._value_set = NATURALS_0

        super().__init__(parameters=(mu,))

    @property
    def value_set(self) -> NumericValueSet:
        return self._value_set

    def _sample(self,
                *parameters: np.ndarray,
                num_samples: int,
                rng: np.random.Generator,
                ) -> np.ndarray:

        mu, = parameters

        return poisson.rvs(
            mu=mu,
            size=num_samples,
            random_state=rng
        )
