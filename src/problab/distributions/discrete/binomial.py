from numbers import Real
from typing import Any

import numpy as np
import sympy as sp
from numpy._typing import NDArray
from scipy.stats import binom

from src.problab.distributions.base import Distribution
from src.problab.random_variables.base import RandomVariable
from src.problab.random_variables.nodes import ConstantNode
from src.problab.validation._decorator import _validate_parameters
from src.problab.validation.distributions.discrete._binomial import _validate_binomial_n, _validate_binomial_p
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import NATURALS_0

class BinomialDistribution(Distribution):

    @_validate_parameters(
        n=_validate_binomial_n,
        p=_validate_binomial_p,
    )
    def __init__(
            self,
            n: RandomVariable | int,
            p: RandomVariable | Real
        ) -> None:

        if isinstance(n, RandomVariable):
            n_node = n._node
        else:  # n is an integer
            n_node = ConstantNode(n)

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
        else:  # p is a real number
            p_node = ConstantNode(p)

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
