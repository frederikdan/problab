import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import sympy as sp

from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets._unknown import _UnknownValueSet


class HomogeneousNumericValueSetTests(unittest.TestCase):

    def test_non_finite_values_are_rejected_by_default_and_enabled_independently(self):
        for flag, value in (("allows_positive_infinity", np.inf), ("allows_negative_infinity", -np.inf), ("allows_nan", np.nan)):
            with self.subTest(flag=flag):
                self.assertFalse(HomogeneousNumericValueSet(sp.S.Reals, (np.float64,)).contains(value))
                support = HomogeneousNumericValueSet(sp.S.Reals, (np.float64,), **{flag: True})
                self.assertTrue(support.contains(value))
                self.assertIs(type(support.contains(value)), bool)

    def test_contains_preserves_array_shape_with_finite_and_exceptional_values(self):
        support = HomogeneousNumericValueSet(sp.Interval(0, 1), (np.float64,), allows_nan=True)
        np.testing.assert_array_equal(support.contains(np.array([[0, np.nan], [2, np.inf]])), [[True, True], [False, False]])
        result = support.contains(np.empty((2, 0)))
        self.assertEqual(result.shape, (2, 0))
        self.assertEqual(result.dtype, np.dtype(bool))

    def test_unknown_finite_support_does_not_override_non_finite_permissions(self):
        support = HomogeneousNumericValueSet(_UnknownValueSet(), None, allows_nan=True)
        self.assertTrue(support.contains(np.nan))
        self.assertFalse(support.contains(1.0))

    def test_exceptional_permissions_are_keyword_only_and_frozen(self):
        with self.assertRaises(TypeError):
            HomogeneousNumericValueSet(sp.S.Reals, (np.float64,), True)
        support = HomogeneousNumericValueSet(sp.S.Reals, (np.float64,), allows_nan=True)
        with self.assertRaises(FrozenInstanceError):
            support.allows_nan = False

    def test_exceptional_permissions_require_booleans_even_with_unknown_dtype(self):
        for dtype_types in ((np.float64,), None):
            for flag in ("allows_positive_infinity", "allows_negative_infinity", "allows_nan"):
                with self.subTest(dtype_types=dtype_types, flag=flag), self.assertRaises(TypeError):
                    HomogeneousNumericValueSet(sp.S.Reals, dtype_types, **{flag: "yes"})

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
