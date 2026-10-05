import unittest
import numpy as np
from problab.operations import _comparison as comparison


class ComparisonOperationTests(unittest.TestCase):
    def test_comparisons_apply_elementwise_with_correct_names(self):
        for name, symbol, expected in (
            ("_LT", "<", [True, False, False]), ("_GT", ">", [False, False, True]),
            ("_LTE", "<=", [True, True, False]), ("_GTE", ">=", [False, True, True]),
            ("_EQ", "=", [False, True, False]), ("_NEQ", "!=", [True, False, True]),
        ):
            with self.subTest(operation=name):
                descriptor = getattr(comparison, name)
                self.assertIsInstance(descriptor, comparison._ComparisonOperation)
                self.assertEqual(descriptor.name_func("X", "Y"), f"{{X {symbol} Y}}")
                np.testing.assert_array_equal(descriptor.operation(np.array([1, 2, 3]), 2), expected)

    def test_ordered_comparisons_support_real_and_object_dtypes(self):
        for descriptor in (comparison._LT, comparison._GT, comparison._LTE, comparison._GTE):
            self.assertEqual(descriptor.supported_input_types, ((np.integer, np.floating, np.object_),))

    def test_equality_has_no_dtype_restriction_and_accepts_objects(self):
        for descriptor in (comparison._EQ, comparison._NEQ):
            self.assertIsNone(descriptor.supported_input_types)
        np.testing.assert_array_equal(comparison._EQ.operation(np.array(["red", "blue"], dtype=object), "red"), [True, False])
