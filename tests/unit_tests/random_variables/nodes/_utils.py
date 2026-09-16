import unittest

import numpy as np
import sympy as sp

from problab.random_variables.nodes._utils import (
    _constant_array,
    _constant_value_set,
    _sympy_constant_value,
)
from problab.value_sets import ObjectValueSet
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet


class NodeUtilityTests(unittest.TestCase):

    def test_sympy_constant_value_converts_integral_float_to_integer(self):
        self.assertEqual(_sympy_constant_value(2.0), sp.Integer(2))
        self.assertEqual(_sympy_constant_value(np.float64(-3.0)), sp.Integer(-3))

    def test_sympy_constant_value_preserves_non_integral_float(self):
        self.assertEqual(_sympy_constant_value(2.5), sp.Float(2.5))

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


if __name__ == "__main__":
    unittest.main()
