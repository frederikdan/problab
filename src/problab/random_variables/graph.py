from typing import Any
import networkx as nx
import matplotlib.pyplot as plt

from problab.random_variables.nodes import _Node
from problab.validation._common import _validate_max_size
from problab.validation._decorator import _validate_parameters


class NodeGraph:

    @_validate_parameters(max_size=_validate_max_size)
    def __init__(self,
                 root_nodes: tuple[_Node, ...],
                 max_size: int
                 ) -> None:

        self._root_nodes = root_nodes
        self._max_size = max_size
        self._graph = nx.DiGraph()
        self._is_complete = True
        self._build_graph()


    @property
    def size(self) -> int:
        # number of nodes in graph
        return self._graph.number_of_nodes()

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @property
    def nodes(self) -> set[_Node]:
        return set(self._graph.nodes)

    @property
    def nx_graph(self) -> nx.DiGraph:
        return self._graph

    def num_dependants(self, node: _Node) -> int:
        return self._graph.in_degree(node)  # for one input node this always returns an int

    def num_dependencies(self, node: _Node) -> int:
        return self._graph.out_degree(node)  # for one input node this always returns an int

    def plot(self, use_extended_names: bool = False) -> None:

        labels = {
            node: node.extended_name if use_extended_names else node.name
            for node in self.nodes
        }

        nx.draw(
            self._graph,
            with_labels=True,
            labels=labels,
        )

        plt.show()

    def _build_graph(self) -> None:
        for root_node in self._root_nodes:
            self._add_node_recursive(root_node)

    def _add_node_recursive(self, node: _Node) -> None:

        if node in self._graph:
            return

        if self._graph.number_of_nodes() >= self._max_size:
            self._is_complete = False
            return

        self._graph.add_node(node)

        for child_node in node.dependencies:
            self._add_node_recursive(child_node)

            if child_node in self._graph:
                self._graph.add_edge(node, child_node)



