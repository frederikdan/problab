from typing import TypeVar

import numpy as np

from src.problab.random_variables._config import DEF_MAX_GRAPH_SIZE
from src.problab.random_variables.graph import NodeGraph
from src.problab.random_variables._nodes import _Node
from src.problab.value_sets._utils import validate_as_subset

T = TypeVar('T')


class _RealizationContext:

    def __init__(self,
                 root_node: _Node,
                 num_samples: int = 1,
                 rng: np.random.Generator | None = None,
                 validate: bool = False,
                 ) -> None:

        if isinstance(num_samples, (bool, np.bool_)) or not isinstance(num_samples, (int, np.integer)):
            raise TypeError("'num_samples' must be an integer.")

        if num_samples < 1:
            raise ValueError("'num_samples' must be at least 1.")

        num_samples = int(num_samples)

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
        self._realizations: dict[_Node, np.ndarray] = {}
        self._validate = validate


    def __contains__(self, item):
        return item in self._realizations

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def evaluate(self, node: _Node) -> np.ndarray:

        if node not in self._realizations:

            node_realizations = np.asarray(node._evaluate(self))

            if node_realizations.ndim == 0:
                node_realizations = np.broadcast_to(node_realizations, (self.num_samples,))

            if node_realizations.shape != (self.num_samples,):
                raise ValueError(
                    f"Expected sample shape ({self.num_samples},), "
                    f"got {node_realizations.shape}."
                )

            dtype_types = node.value_set.dtype_types
            if dtype_types is not None:
                if not any(np.issubdtype(node_realizations.dtype, dtype_type) for dtype_type in dtype_types):
                    raise TypeError(
                        f"Node {node.name!r} returned dtype {node_realizations.dtype}; expected one of {dtype_types}.")

            if self._validate:
                validate_as_subset(values=node_realizations, target_set=node.value_set)

            self._realizations[node] = node_realizations

            for dependency in node.dependencies:
                self._remaining_dependants[dependency] -= 1
                if self._remaining_dependants[dependency] == 0:
                    self._realizations.pop(dependency)

        return self._realizations[node]


