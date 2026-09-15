from numbers import Real

import numpy as np
from scipy.stats import poisson

from src.problab.distributions.base import Distribution
from src.problab.random_variables.base import RandomVariable
from src.problab.validation._decorator import _validate_parameters
from src.problab.validation.distributions.discrete._poisson import _validate_poisson_mu
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import NATURALS_0

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
    def value_set(self) -> ValueSet:
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
