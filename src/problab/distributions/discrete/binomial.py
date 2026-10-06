from functools import cached_property
from numbers import Real
from typing import Any, ClassVar, Literal

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.stats import binom

from problab._config import DEF_PARAMETER_RISK_POLICY
from problab.distributions.base import Distribution
from problab.random_variables.base import RandomVariable
from problab.validation._decorator import _validate_parameters
from problab.validation.distributions._base import _validate_parameter_risk_policy
from problab.validation.distributions.discrete._binomial import (
    _validate_binomial_n, _validate_binomial_p,
    _validate_binomial_n_realizations, _validate_binomial_p_realizations,
    _BINOMIAL_N_MAX, _BINOMIAL_N_REALIZATION_SET,
)
from problab.value_sets.base import NumericValueSet
from problab.value_sets._utils import is_known_subset
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.sets import NATURALS_0, UNIT_INTERVAL

class BinomialDistribution(Distribution):

    symbol: ClassVar[str] = "Binomial"
    _valid_parameter_sets: ClassVar[dict[str, NumericValueSet]] = {
        "n": NATURALS_0,
        "p": UNIT_INTERVAL,
    }

    @_validate_parameters(
        validator_arguments=("parameter_risk_policy",),
        parameter_risk_policy=_validate_parameter_risk_policy,
        n=_validate_binomial_n,
        p=_validate_binomial_p,
    )
    def __init__(
            self,
            n: RandomVariable | int,
            p: RandomVariable | Real,
            *,
            parameter_risk_policy: Literal["warn", "raise", "ignore"] = DEF_PARAMETER_RISK_POLICY,
        ) -> None:

        super().__init__(parameters=(n, p))

        n_node, _ = self._parameter_nodes

        n_max = n_node.value_set.sympy_set.sup  # validation ensures that this is a NumericValueSet which has sympy_set

        if n_max == sp.oo:
            self._mathematical_value_set = NATURALS_0
        else:
            self._mathematical_value_set = HomogeneousNumericValueSet(
                sympy_set=sp.Range(0, int(n_max) + 1),
                dtype_types=(np.integer,),
            )

    @property
    def value_set(self) -> HomogeneousNumericValueSet:
        return self._mathematical_value_set

    @cached_property
    def _realization_value_set(self) -> HomogeneousNumericValueSet:
        n_support = self._parameter_nodes[0]._realization_value_set
        allows_nan = (
            not self._parameter_realizations_guaranteed_valid()
            or not is_known_subset(n_support, _BINOMIAL_N_REALIZATION_SET)
        )
        n_max = int(_BINOMIAL_N_MAX)
        try:
            upper = n_support.sympy_set.sup
            if upper.is_finite is True:
                n_max = max(0, min(n_max, int(sp.floor(upper))))
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            pass

        dtype_types = (np.integer,)
        if allows_nan:
            dtype_types += (np.float64,)
            if n_max > 2**53:
                dtype_types += (np.object_,)

        return HomogeneousNumericValueSet(
            sympy_set=sp.Range(0, n_max + 1),
            dtype_types=dtype_types,
            allows_nan=allows_nan,
        )

    def _validate_parameter_realizations(
        self,
        *parameters: np.ndarray,
    ) -> dict[str, np.ndarray]:
        n, p = parameters
        return {
            "n": _validate_binomial_n_realizations(n),
            "p": _validate_binomial_p_realizations(p),
        }

    def _sample(self,
                *parameters: np.ndarray,
                num_samples: int,
                rng: np.random.Generator,
                ) -> NDArray[np.integer[Any]]:

        n, p = parameters

        return binom.rvs(
            n=n,
            p=p,
            size=num_samples,
            random_state=rng
        )
