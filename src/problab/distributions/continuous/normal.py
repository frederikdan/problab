from functools import cached_property
from numbers import Real
from typing import Any, ClassVar, Literal

import numpy as np
from numpy._typing import NDArray
from scipy.stats import norm

from problab.value_sets.base import NumericValueSet
from problab._config import DEF_PARAMETER_RISK_POLICY
from problab.distributions.base import Distribution
from problab.random_variables.base import RandomVariable
from problab.validation._decorator import _validate_parameters
from problab.validation.distributions._base import _validate_parameter_risk_policy
from problab.validation.distributions.continuous._normal import (
    _validate_normal_mean,
    _validate_normal_std,
    _validate_normal_mean_realizations,
    _validate_normal_std_realizations,
)
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.sets import REALS, POSITIVE_REALS


class NormalDistribution(Distribution):

    symbol: ClassVar[str] = "Normal"
    _valid_parameter_sets: ClassVar[dict[str, NumericValueSet]] = {
        "mean": REALS,
        "std": POSITIVE_REALS,
    }

    @_validate_parameters(
        validator_arguments=("parameter_risk_policy",),
        parameter_risk_policy=_validate_parameter_risk_policy,
        mean=_validate_normal_mean,
        std=_validate_normal_std,
    )
    def __init__(self,
                 mean: RandomVariable | Real,
                 std: RandomVariable | Real,
                 *,
                 parameter_risk_policy: Literal["warn", "raise", "ignore"] = DEF_PARAMETER_RISK_POLICY,
    ) -> None:

        self._mathematical_value_set = REALS

        super().__init__(parameters=(mean, std))

    @property
    def value_set(self) -> HomogeneousNumericValueSet:
        return self._mathematical_value_set

    @cached_property
    def _realization_value_set(self) -> HomogeneousNumericValueSet:
        return HomogeneousNumericValueSet(
            sympy_set=REALS.sympy_set,
            dtype_types=(np.floating,),
            allows_positive_infinity=True,
            allows_negative_infinity=True,
            allows_nan=not self._parameter_realizations_guaranteed_valid(),
        )

    def _validate_parameter_realizations(
            self,
            *parameters: np.ndarray,
    ) -> dict[str, np.ndarray]:
        mean, std = parameters

        return {
            "mean": _validate_normal_mean_realizations(mean),
            "std": _validate_normal_std_realizations(std),
        }

    def _sample(self,
                *parameters: np.ndarray,
                num_samples: int,
                rng: np.random.Generator,
                ) -> NDArray[np.floating[Any]]:

        mean, std = parameters

        return norm.rvs(
            loc=mean,
            scale=std,
            size=num_samples,
            random_state=rng,
        )



