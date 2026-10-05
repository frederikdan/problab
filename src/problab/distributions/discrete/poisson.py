from numbers import Real
from typing import ClassVar, Literal

import numpy as np
from scipy.stats import poisson

from problab._config import DEF_PARAMETER_RISK_POLICY
from problab.distributions.base import Distribution
from problab.random_variables.base import RandomVariable
from problab.validation._decorator import _validate_parameters
from problab.validation.distributions._base import _validate_parameter_risk_policy
from problab.validation.distributions.discrete._poisson import _validate_poisson_mu
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.sets import NATURALS_0

class PoissonDistribution(Distribution):

    symbol: ClassVar[str] = "Poisson"

    @_validate_parameters(
        validator_arguments=("parameter_risk_policy",),
        parameter_risk_policy=_validate_parameter_risk_policy,
        mu=_validate_poisson_mu,
    )
    def __init__(
            self,
            mu: RandomVariable | Real,
            *,
            parameter_risk_policy: Literal["warn", "raise", "ignore"] = DEF_PARAMETER_RISK_POLICY,
        ) -> None:

        self._mathematical_value_set = NATURALS_0

        super().__init__(parameters=(mu,))

    @property
    def value_set(self) -> HomogeneousNumericValueSet:
        return self._mathematical_value_set

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
