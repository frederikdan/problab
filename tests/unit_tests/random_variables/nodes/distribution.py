import unittest
from unittest.mock import Mock, sentinel

from problab.random_variables.nodes.distribution import _DistributionNode
from problab.value_sets.sets import REALS


class DistributionNodeTests(unittest.TestCase):

    def test_properties_are_forwarded_from_distribution(self):
        dependency = sentinel.dependency
        distribution = Mock()
        distribution.name = "Normal(0, 1)"
        distribution.value_set = REALS
        distribution._node_dependencies = {dependency}

        node = _DistributionNode(distribution, rv_name="X")

        self.assertIs(node.distribution, distribution)
        self.assertEqual(node.name, "X")
        self.assertEqual(node.extended_name, "X ~ Normal(0, 1)")
        self.assertEqual(repr(node), "DistributionNode(X)")
        self.assertIs(node.value_set, REALS)
        self.assertEqual(node.dependencies, {dependency})

    def test_evaluate_delegates_to_distribution(self):
        distribution = Mock()
        distribution._evaluate.return_value = sentinel.samples
        node = _DistributionNode(distribution, rv_name="X")

        result = node._evaluate(sentinel.context)

        self.assertIs(result, sentinel.samples)
        distribution._evaluate.assert_called_once_with(sentinel.context)


if __name__ == "__main__":
    unittest.main()
