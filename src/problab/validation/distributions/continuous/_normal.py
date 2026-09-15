from numbers import Real

from src.problab.random_variables.nodes import ConstantNode
from src.problab.value_sets._utils import is_known_subset
from src.problab.value_sets.sets import POSITIVE_REALS, REALS


def _validate_normal_mean(value) -> None:
    from src.problab.random_variables.base import RandomVariable

    if isinstance(value, RandomVariable):
        mean_node = value._node
    elif isinstance(value, Real):
        mean_node = ConstantNode(value)
    else:
        raise TypeError("'mean' must be a RandomVariable or a real number.")

    if not is_known_subset(mean_node.value_set, REALS):
        raise ValueError("'mean' must contain only real values.")


def _validate_normal_std(value) -> None:
    from src.problab.random_variables.base import RandomVariable

    if isinstance(value, RandomVariable):
        std_node = value._node
    elif isinstance(value, Real):
        std_node = ConstantNode(value)
    else:
        raise TypeError("'std' must be a RandomVariable or a real number.")

    if not is_known_subset(std_node.value_set, POSITIVE_REALS):
        raise ValueError("'std' must contain only positive real values.")
