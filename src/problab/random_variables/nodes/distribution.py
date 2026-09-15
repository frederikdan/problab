from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from problab.random_variables.nodes.base import _Node

if TYPE_CHECKING:
    from problab.distributions.base import Distribution
    from problab.random_variables._context import _RealizationContext


class _DistributionNode(_Node):

    def __init__(self, distribution: Distribution, rv_name: str) -> None:
        super().__init__()

        self._distribution = distribution
        self._name = f"{rv_name}"
        self._extended_name = self._name + f" ~ {distribution.name}"

    def __repr__(self) -> str:
        return f"DistributionNode({self.name})"

    @property
    def distribution(self) -> Distribution:
        return self._distribution

    @property
    def value_set(self):
        return self._distribution.value_set

    @property
    def dependencies(self) -> set[_Node]:
        return self._distribution._node_dependencies

    def _evaluate(self, context: _RealizationContext) -> np.ndarray:
        return self._distribution._evaluate(context)
