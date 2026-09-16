import unittest
from unittest.mock import sentinel

import numpy as np

from problab.random_variables.nodes.constant import _ConstantNode
from problab.value_sets import ObjectValueSet


class ConstantNodeTests(unittest.TestCase):

    def test_numeric_constant_exposes_value_name_and_empty_dependencies(self):
        node = _ConstantNode(3)

        self.assertEqual(node.value, 3)
        self.assertEqual(node.name, "3")
        self.assertEqual(repr(node), "ConstantNode(3)")
        self.assertEqual(node.dependencies, set())
        self.assertTrue(node.value_set.contains(3))

    def test_evaluate_returns_scalar_array_for_scalar_value(self):
        node = _ConstantNode(3)

        result = node._evaluate(sentinel.context)

        self.assertEqual(result.shape, ())
        self.assertEqual(result.item(), 3)

    def test_compound_constant_remains_atomic_object(self):
        value = [1, 2]
        node = _ConstantNode(value)

        result = node._evaluate(sentinel.context)

        self.assertEqual(result.shape, ())
        self.assertIs(result.item(), value)
        self.assertIsInstance(node.value_set, ObjectValueSet)


if __name__ == "__main__":
    unittest.main()
