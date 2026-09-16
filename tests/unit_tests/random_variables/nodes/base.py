import unittest

import numpy as np

from problab.random_variables.nodes.base import _Node
from problab.value_sets.sets import REALS


class _StubNode(_Node):

    def __init__(self, name="node", dependencies=()):
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


class NodeBaseTests(unittest.TestCase):

    def test_base_node_remains_abstract(self):
        with self.assertRaises(TypeError):
            _Node()

    def test_name_and_extended_name_are_exposed(self):
        node = _StubNode("value")

        self.assertEqual(node.name, "value")
        self.assertEqual(node.extended_name, "extended value")
        self.assertEqual(str(node), "value")

    def test_has_dependencies_reflects_direct_dependencies(self):
        leaf = _StubNode("leaf")

        self.assertFalse(leaf.has_dependencies)
        self.assertTrue(_StubNode("parent", dependencies=(leaf,)).has_dependencies)


if __name__ == "__main__":
    unittest.main()
