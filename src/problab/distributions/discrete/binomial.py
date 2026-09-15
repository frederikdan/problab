from numbers import Real
from typing import Any

import numpy as np
import sympy as sp
from numpy._typing import NDArray
from scipy.stats import binom

from src.problab.distributions.base import Distribution
from src.problab.random_variables.base import RandomVariable
from src.problab.random_variables.context import RealizationContext
from src.problab.random_variables.nodes import ConstantNode
from src.problab.value_sets._utils import is_known_subset
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import NATURALS_0, UNIT_INTERVAL

class BinomialDistribution(Distribution):

    def __init__(
            self,
            n: RandomVariable | int,
            p: RandomVariable | Real
        ) -> None:

        if isinstance(n, RandomVariable):
            n_node = n._node
        elif isinstance(n, int):
            n_node = ConstantNode(n)
        else:
            raise TypeError("'n' must be a RandomVariable or int.")

        if not is_known_subset(n_node.value_set, NATURALS_0):
            raise ValueError("'n' must be a positive integer or 0.")

        self._n = n_node

        n_max = n_node.value_set.sympy_set.sup

        if n_max == sp.oo:
            self._value_set = NATURALS_0
        else:
            self._value_set = ValueSet(
                sympy_set=sp.FiniteSet(*range(int(n_max) + 1)),
                dtype_types=(np.integer,),
            )

        if isinstance(p, RandomVariable):
            p_node = p._node
        elif isinstance(p, Real):
            p_node = ConstantNode(p)
        else:
            raise TypeError("'p' must be a RandomVariable or Real.")

        if not is_known_subset(p_node.value_set, UNIT_INTERVAL):
            raise ValueError("'p' must be in the interval [0, 1].")

        self._p = p_node

        super().__init__(parameters=(self._n, self._p), symbol="Bin")

    @property
    def value_set(self) -> ValueSet:
        return self._value_set

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
