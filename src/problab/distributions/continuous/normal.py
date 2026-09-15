from numbers import Real
from typing import Any

import numpy as np
from numpy._typing import NDArray
from scipy.stats import norm

from src.problab.distributions.base import Distribution
from src.problab.random_variables.base import RandomVariable
from src.problab.random_variables.context import RealizationContext
from src.problab.random_variables.nodes import ConstantNode, DistributionNode
from src.problab.value_sets._utils import is_known_subset
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import REALS, POSITIVE_REALS

class NormalDistribution(Distribution):

    def __init__(self,
                 mean: RandomVariable | Real,
                 std: RandomVariable | Real
                 ) -> None:

        if isinstance(mean, RandomVariable):
            mean_node = mean._node
        elif isinstance(mean, Real):
            mean_node = ConstantNode(mean)
        else:
            raise TypeError("'mean' must be a RandomVariable or Real.")

        if not is_known_subset(mean_node.value_set, REALS):
            raise ValueError("'mean' must contain only real values.")

        self._mean = mean_node

        if isinstance(std, RandomVariable):
            std_node = std._node
        elif isinstance(std, Real):
            std_node = ConstantNode(std)
        else:
            raise TypeError("'std' must be a RandomVariable or Real.")

        if not is_known_subset(std_node.value_set, POSITIVE_REALS):
            raise ValueError("'std' must contain only positive real values.")

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



