import unittest
from unittest.mock import sentinel

import numpy as np

from problab.random_variables.nodes.constant import _ConstantNode
from problab.value_sets import HomogeneousNumericValueSet, ObjectValueSet


class ConstantNodeTests(unittest.TestCase):

    def test_realization_support_is_cached_and_records_actual_exceptional_values(self):
        for value, flags in ((np.inf, (True, False, False)), (-np.inf, (False, True, False)),
                             (np.nan, (False, False, True)), (complex(np.inf, np.nan), (True, False, True))):
            with self.subTest(value=value):
                node = _ConstantNode(value)
                support = node._realization_value_set
                self.assertIs(support, node._realization_value_set)
                self.assertIsInstance(support, HomogeneousNumericValueSet)
                self.assertEqual(support.dtype_types, (np.asarray(value).dtype.type,))
                self.assertEqual((support.allows_positive_infinity, support.allows_negative_infinity, support.allows_nan), flags)
                self.assertTrue(support.contains(value))

    def test_object_constant_reuses_object_support_without_numeric_detection(self):
        node = _ConstantNode(object())
        self.assertIs(node._realization_value_set, node.value_set)

    def test_numeric_constant_exposes_value_name_and_empty_dependencies(self):
        node = _ConstantNode(3)

        self.assertEqual(node.value, 3)
        self.assertEqual(node.name, "3")
        self.assertEqual(repr(node), "ConstantNode(3)")
        self.assertEqual(node.dependencies, set())
        self.assertTrue(node.value_set.contains(3))
        self.assertEqual(node._realization_value_set, node.value_set)

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
