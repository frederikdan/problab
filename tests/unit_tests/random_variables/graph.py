import unittest
from unittest.mock import patch

import numpy as np

from problab.random_variables.graph import NodeGraph
from problab.random_variables.nodes.base import _Node
from problab.value_sets.sets import REALS


class _StubNode(_Node):

    def __init__(self, name, dependencies=()):
        super().__init__()
        self._name = name
        self._extended_name = f"extended {name}"
        self._dependencies = set(dependencies)

    def __repr__(self):
        return f"StubNode({self.name})"

    @property
    def value_set(self):
        return REALS

    @property
    def dependencies(self):
        return self._dependencies

    def _evaluate(self, context):
        return np.array([1.0])


class NodeGraphTests(unittest.TestCase):

    def test_graph_contains_nodes_and_edges_from_root_to_dependencies(self):
        leaf = _StubNode("leaf")
        middle = _StubNode("middle", (leaf,))
        root = _StubNode("root", (middle, leaf))

        graph = NodeGraph(root, max_size=3)

        self.assertTrue(graph.is_complete)
        self.assertEqual(graph.size, 3)
        self.assertEqual(graph.nodes, {root, middle, leaf})
        self.assertEqual(set(graph.nx_graph.edges), {(root, middle), (root, leaf), (middle, leaf)})
        self.assertEqual(graph.num_dependencies(root), 2)
        self.assertEqual(graph.num_dependents(leaf), 2)

    def test_graph_stops_when_maximum_size_is_reached(self):
        leaf = _StubNode("leaf")
        middle = _StubNode("middle", (leaf,))
        root = _StubNode("root", (middle,))

        graph = NodeGraph(root, max_size=2)

        self.assertFalse(graph.is_complete)
        self.assertEqual(graph.size, 2)
        self.assertEqual(graph.nodes, {root, middle})
        self.assertEqual(set(graph.nx_graph.edges), {(root, middle)})

    def test_constructor_rejects_non_positive_maximum_size(self):
        with self.assertRaises(ValueError):
            NodeGraph(_StubNode("root"), max_size=0)

    @patch("problab.random_variables.graph.plt.show")
    @patch("problab.random_variables.graph.nx.draw")
    def test_plot_uses_selected_node_labels(self, draw, show):
        node = _StubNode("root")
        graph = NodeGraph(node, max_size=1)

        graph.plot(use_extended_names=True)

        draw.assert_called_once_with(
            graph.nx_graph,
            with_labels=True,
            labels={node: "extended root"},
        )
        show.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
