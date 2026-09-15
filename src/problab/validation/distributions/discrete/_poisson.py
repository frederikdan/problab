from numbers import Real

from src.problab.random_variables.nodes import ConstantNode
from src.problab.value_sets._utils import is_known_subset
from src.problab.value_sets.sets import NON_NEGATIVE_REALS


def _validate_poisson_mu(value) -> None:
    from src.problab.random_variables.base import RandomVariable

    if isinstance(value, RandomVariable):
        mu_node = value._node
    elif isinstance(value, Real):
        mu_node = ConstantNode(value)
    else:
        raise TypeError("'mu' must be a RandomVariable or a real number.")

    if not is_known_subset(mu_node.value_set, NON_NEGATIVE_REALS):
        raise ValueError("'mu' must be non-negative.")
