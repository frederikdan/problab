import unittest
from fractions import Fraction

import numpy as np
import sympy as sp

from problab.value_sets.mixed_numeric_value_set import MixedNumericValueSet


class MixedNumericValueSetTests(unittest.TestCase):

    def test_preserves_values_and_supports_scalar_and_array_membership(self):
        value_set = MixedNumericValueSet((1, Fraction(1, 2)))

        self.assertIsNone(value_set.dtype_types)
        self.assertTrue(value_set.contains(Fraction(1, 2)))
        self.assertFalse(value_set.contains(2))
        np.testing.assert_array_equal(value_set.contains(np.array([1, 2], dtype=object)), [True, False])
        self.assertEqual(value_set.sympy_set, sp.FiniteSet(1, sp.Rational(1, 2)))

    def test_constructor_requires_non_empty_tuple(self):
        with self.assertRaises(TypeError):
            MixedNumericValueSet([1])
        with self.assertRaises(ValueError):
            MixedNumericValueSet(())


if __name__ == "__main__":
    unittest.main()
