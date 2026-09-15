from typing import Any, TypeVar

import numpy as np

from src.problab.random_variables._config import DEF_MAX_GRAPH_SIZE
from src.problab.random_variables.graph import NodeGraph
from src.problab.random_variables.nodes import Node

T = TypeVar('T')


class RealizationContext:

    def __init__(
        self,
        root_node: Node,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:

        if num_samples < 1:
            raise ValueError("'num_samples' must be at least 1.")

        self._root_node = root_node
        self._graph = NodeGraph(root_node, max_size=DEF_MAX_GRAPH_SIZE)

        if not self._graph.is_complete:
            raise ValueError(f"RandomVariable dependency graph exceeds the maximum size of {DEF_MAX_GRAPH_SIZE} nodes.")

        self._remaining_dependants = {
            node: self._graph.num_dependents(node)
            for node in self._graph.nodes
        }

        self._num_samples = num_samples
        self._rng = np.random.default_rng() if rng is None else rng
        self._realizations: dict[Node, np.ndarray] = {}

    def __contains__(self, item):
        return item in self._realizations

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def evaluate(self, node: Node) -> np.ndarray:

        if node not in self._realizations:
            self._realizations[node] = node._evaluate(self)

            for dependency in node.dependencies:
                self._remaining_dependants[dependency] -= 1
                if self._remaining_dependants[dependency] == 0:
                    self._realizations.pop(dependency)

        return self._realizations[node]


