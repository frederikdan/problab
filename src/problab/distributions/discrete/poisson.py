from functools import cached_property
from numbers import Real
from typing import ClassVar, Literal

import numpy as np
from scipy.stats import poisson

from problab._config import DEF_PARAMETER_RISK_POLICY
from problab.distributions.base import Distribution
from problab.random_variables.base import RandomVariable
from problab.validation._decorator import _validate_parameters
from problab.validation.distributions._base import _validate_parameter_risk_policy
from problab.validation.distributions.discrete._poisson import (
    _validate_poisson_mu, _validate_poisson_mu_realizations,
    _POISSON_MU_REALIZATION_SET,
)
from problab.value_sets.base import NumericValueSet
from problab.value_sets._utils import is_known_subset
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.sets import NATURALS_0, NON_NEGATIVE_REALS

class PoissonDistribution(Distribution):

    symbol: ClassVar[str] = "Poisson"
    _valid_parameter_sets: ClassVar[dict[str, NumericValueSet]] = {
        "mu": NON_NEGATIVE_REALS,
    }

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

    @cached_property
    def _realization_value_set(self) -> HomogeneousNumericValueSet:
        allows_nan = (
            not self._parameter_realizations_guaranteed_valid()
            or not is_known_subset(
                self._parameter_nodes[0]._realization_value_set,
                _POISSON_MU_REALIZATION_SET,
            )
        )
        return HomogeneousNumericValueSet(
            sympy_set=NATURALS_0.sympy_set,
            dtype_types=(np.integer, np.float64, np.object_) if allows_nan else (np.integer,),
            allows_nan=allows_nan,
        )

    def _validate_parameter_realizations(
        self,
        *parameters: np.ndarray,
    ) -> dict[str, np.ndarray]:
        mu, = parameters
        return {"mu": _validate_poisson_mu_realizations(mu)}

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
