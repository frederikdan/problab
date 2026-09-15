from numbers import Real

from problab.random_variables.nodes import _ConstantNode
from problab.value_sets._utils import is_known_subset
from problab.value_sets.sets import NATURALS_0, UNIT_INTERVAL


def _validate_binomial_n(value) -> None:
    from problab.random_variables.base import RandomVariable

    if isinstance(value, RandomVariable):
        n_node = value._node
    elif isinstance(value, int):
        n_node = _ConstantNode(value)
    else:
        raise TypeError("'n' must be a RandomVariable or an integer.")

    if not is_known_subset(n_node.value_set, NATURALS_0):
        raise ValueError("'n' must be a positive integer or 0.")


def _validate_binomial_p(value) -> None:
    from problab.random_variables.base import RandomVariable

    if isinstance(value, RandomVariable):
        p_node = value._node
    elif isinstance(value, Real):
        p_node = _ConstantNode(value)
    else:
        raise TypeError("'p' must be a RandomVariable or a real number.")

    if not is_known_subset(p_node.value_set, UNIT_INTERVAL):
        raise ValueError("'p' must be in the interval [0, 1].")
