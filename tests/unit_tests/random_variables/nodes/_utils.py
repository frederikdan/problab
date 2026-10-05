import unittest

import numpy as np
import sympy as sp

from problab.operations._arithmetic import (_ADD, _REAL_POWER)
from problab.random_variables.nodes import _ConstantNode, _OperationNode
from problab.random_variables.nodes._utils import (
    _constant_array,
    _constant_value_set,
    _node_is_constant,
    _node_is_operation,
    _node_is_square_operation,
)
from problab.value_sets import ObjectValueSet
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet


class NodeUtilityTests(unittest.TestCase):

    def test_constant_array_keeps_scalar_array(self):
        array = _constant_array(3)

        self.assertEqual(array.shape, ())
        self.assertEqual(array.item(), 3)

    def test_constant_array_keeps_compound_value_atomic(self):
        value = [1, 2]
        array = _constant_array(value)

        self.assertEqual(array.shape, ())
        self.assertEqual(array.dtype, object)
        self.assertIs(array.item(), value)

    def test_constant_value_set_for_numeric_value_uses_sympy_and_dtype(self):
        array = _constant_array(2.0)

        value_set = _constant_value_set(2.0, array)

        self.assertIsInstance(value_set, HomogeneousNumericValueSet)
        self.assertEqual(value_set.sympy_set, sp.FiniteSet(2))
        self.assertEqual(value_set.dtype_types, (np.float64,))

    def test_constant_value_set_for_unrepresentable_object_preserves_object(self):
        value = [1, 2]
        value_set = _constant_value_set(value, _constant_array(value))

        self.assertIsInstance(value_set, ObjectValueSet)
        self.assertEqual(value_set.objects, (value,))

    def test_node_type_predicates_recognize_matching_constant_and_operation_nodes(self):
        base_node = _ConstantNode(2)
        exponent_node = _ConstantNode(2)
        square_node = _OperationNode(
            operation=_REAL_POWER,
            inputs=(base_node, exponent_node),
            name="(2 ** 2)",
            mathematical_value_set=HomogeneousNumericValueSet(
                sympy_set=sp.FiniteSet(4),
                dtype_types=(np.int64,),
            ),
        )

        self.assertTrue(_node_is_constant(base_node, 2))
        self.assertFalse(_node_is_constant(base_node, 3))
        self.assertTrue(_node_is_operation(square_node, _REAL_POWER))
        self.assertFalse(_node_is_operation(square_node, _ADD))
        self.assertTrue(_node_is_square_operation(square_node))


if __name__ == "__main__":
    unittest.main()
