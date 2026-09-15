from numbers import Real
from typing import Any

import numpy as np
from numpy._typing import NDArray
from scipy.stats import norm

from src.problab.distributions.base import Distribution
from src.problab.random_variables.base import RandomVariable
from src.problab.random_variables.context import RealizationContext
from src.problab.random_variables.nodes import ConstantNode, DistributionNode
from src.problab.validation._decorator import _validate_parameters
from src.problab.validation.distributions.continuous._normal import _validate_normal_mean, _validate_normal_std
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import REALS

class NormalDistribution(Distribution):

    @_validate_parameters(
        mean=_validate_normal_mean,
        std=_validate_normal_std,
    )
    def __init__(self,
                 mean: RandomVariable | Real,
                 std: RandomVariable | Real
                 ) -> None:

        if isinstance(mean, RandomVariable):
            mean_node = mean._node
        else:  # mean is a real number
            mean_node = ConstantNode(mean)

        self._mean = mean_node

        if isinstance(std, RandomVariable):
            std_node = std._node
        else:  # std is a real number
            std_node = ConstantNode(std)

        self._std = std_node

        self._value_set = REALS

        super().__init__(parameters=(self._mean, self._std), symbol="N")

    @property
    def value_set(self) -> ValueSet:
        return self._value_set

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



