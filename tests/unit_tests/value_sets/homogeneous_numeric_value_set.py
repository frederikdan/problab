import unittest

import numpy as np
import sympy as sp

from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet


class HomogeneousNumericValueSetTests(unittest.TestCase):

    def test_contains_returns_bool_for_scalar_and_array_for_array(self):
        value_set = HomogeneousNumericValueSet(sp.Interval(0, 1), (np.floating,))

        self.assertTrue(value_set.contains(0.5))
        self.assertFalse(value_set.contains(2.0))
        np.testing.assert_array_equal(value_set.contains(np.array([0.0, 2.0])), [True, False])

    def test_constructor_validates_configuration(self):
        with self.assertRaises(TypeError):
            HomogeneousNumericValueSet("reals", (np.floating,))
        with self.assertRaises(ValueError):
            HomogeneousNumericValueSet(sp.S.Reals, ())


if __name__ == "__main__":
    unittest.main()
