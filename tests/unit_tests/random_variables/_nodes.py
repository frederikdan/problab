import unittest

from problab.random_variables import _nodes
from problab.random_variables.nodes import (
    _ConstantNode,
    _DistributionNode,
    _Node,
    _OperationNode,
)


class NodeCompatibilityImportTests(unittest.TestCase):

    def test_compatibility_module_reexports_current_node_types(self):
        self.assertIs(_nodes._Node, _Node)
        self.assertIs(_nodes._ConstantNode, _ConstantNode)
        self.assertIs(_nodes._DistributionNode, _DistributionNode)
        self.assertIs(_nodes._OperationNode, _OperationNode)
        self.assertEqual(
            _nodes.__all__,
            ["_Node", "_ConstantNode", "_DistributionNode", "_OperationNode"],
        )


if __name__ == "__main__":
    unittest.main()
