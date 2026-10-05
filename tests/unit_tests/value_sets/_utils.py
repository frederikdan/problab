import unittest
from itertools import product

import numpy as np
import sympy as sp

from problab.value_sets._utils import (
    _to_sympy_value,
    is_known_subset,
    validate_as_subset,
    _contains_non_finite_value,
    _detect_non_finite_values,
)
from problab.value_sets import HomogeneousNumericValueSet
from problab.value_sets._unknown import _UnknownValueSet
from problab.value_sets.object_value_set import ObjectValueSet
from problab.value_sets.sets import INTEGERS, REALS, UNKNOWN_VALUE_SET


class ValueSetUtilityTests(unittest.TestCase):

    def test_detect_non_finite_values_distinguishes_sign_nan_and_complex_components(self):
        for values, expected in (
            (np.array([1, 2]), (False, False, False)),
            (np.array([1.0, np.inf]), (True, False, False)),
            (np.array([-np.inf]), (False, True, False)),
            (np.array([np.nan]), (False, False, True)),
            (np.array([complex(np.inf, -np.inf), complex(0, np.nan)]), (True, True, True)),
            (np.empty(0), (False, False, False)),
            (np.array([[1.0, np.inf], [-np.inf, np.nan]]), (True, True, True)),
        ):
            with self.subTest(values=values):
                result = _detect_non_finite_values(values)
                self.assertEqual(result, expected)
                self.assertTrue(all(type(flag) is bool for flag in result))

    def test_detect_non_finite_values_leaves_compound_categories_atomic(self):
        array = np.empty(2, dtype=object)
        array[0] = [np.inf, np.nan]
        array[1] = {"value": -np.inf}
        self.assertEqual(_detect_non_finite_values(array), (False, False, False))

    def test_non_finite_membership_uses_each_permission_independently(self):
        for positive, negative, nan in product((False, True), repeat=3):
            support = HomogeneousNumericValueSet(sp.S.Complexes, (np.number,),
                allows_positive_infinity=positive, allows_negative_infinity=negative, allows_nan=nan)
            for value, expected in ((np.inf, positive), (-np.inf, negative), (np.nan, nan),
                                    (complex(np.inf, -np.inf), positive and negative),
                                    (complex(np.inf, np.nan), positive and nan)):
                with self.subTest(flags=(positive, negative, nan), value=value):
                    self.assertIs(_contains_non_finite_value(value, support), expected)

    def test_non_finite_membership_returns_none_for_finite_and_opaque_values(self):
        for value in (1, 1.5, 1 + 2j, "red", [np.inf]):
            with self.subTest(value=value):
                self.assertIsNone(_contains_non_finite_value(value, REALS))

    def test_validate_as_subset_accepts_only_declared_exceptional_values(self):
        support = HomogeneousNumericValueSet(sp.Interval(0, 1), (np.float64,),
                                           allows_positive_infinity=True, allows_nan=True)
        validate_as_subset(np.array([0.5, np.inf, np.nan]), support)
        with self.assertRaisesRegex(ValueError, "index 1"):
            validate_as_subset(np.array([0.5, -np.inf]), support)
        with self.assertRaises(ValueError):
            validate_as_subset(np.array([2.0]), support)

    def test_validate_as_subset_accepts_declared_nan_even_if_finite_support_is_unknown(self):
        support = HomogeneousNumericValueSet(_UnknownValueSet(), (np.float64,), allows_nan=True)
        validate_as_subset(np.array([np.nan]), support)
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_as_subset(np.array([0.0]), support)

    def test_validate_as_subset_requires_all_non_finite_complex_components(self):
        support = HomogeneousNumericValueSet(sp.S.Complexes, (np.complex128,), allows_positive_infinity=True)
        validate_as_subset(np.array([complex(np.inf, 1)]), support)
        with self.assertRaises(ValueError):
            validate_as_subset(np.array([complex(np.inf, np.nan)]), support)

    def test_to_sympy_value_converts_integral_float_to_integer(self):
        self.assertEqual(_to_sympy_value(2.0), sp.Integer(2))
        self.assertEqual(_to_sympy_value(np.float64(-3.0)), sp.Integer(-3))

    def test_to_sympy_value_preserves_non_integral_float(self):
        self.assertEqual(_to_sympy_value(2.5), sp.Float(2.5))

    def test_is_known_subset_handles_numeric_object_and_unknown_sets(self):
        self.assertTrue(is_known_subset(INTEGERS, REALS))
        self.assertTrue(is_known_subset(sp.S.Integers, sp.S.Reals))
        self.assertFalse(is_known_subset(ObjectValueSet(("red",)), REALS))
        self.assertFalse(is_known_subset(UNKNOWN_VALUE_SET, REALS))

    def test_validate_as_subset_accepts_integer_valued_floats_for_integer_set(self):
        validate_as_subset(np.array([-2.0, 0.0, 3.0]), INTEGERS)

    def test_validate_as_subset_rejects_outside_or_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "index 1"):
            validate_as_subset(np.array([1, 1.5]), INTEGERS)
        with self.assertRaises(ValueError):
            validate_as_subset(np.array([1]), UNKNOWN_VALUE_SET)

    def test_validate_as_subset_uses_object_membership(self):
        value_set = ObjectValueSet(("red", "blue"))

        validate_as_subset(np.array(["red", "blue"], dtype=object), value_set)
        with self.assertRaisesRegex(ValueError, "index 1"):
            validate_as_subset(np.array(["red", "green"], dtype=object), value_set)


if __name__ == "__main__":
    unittest.main()
