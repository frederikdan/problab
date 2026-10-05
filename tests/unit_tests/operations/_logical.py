import unittest
import numpy as np
from problab.operations import _logical as logical


class LogicalOperationTests(unittest.TestCase):
    def test_binary_operations_cover_complete_boolean_truth_tables(self):
        for descriptor, symbol, expected in (
            (logical._AND, "&", [False, False, False, True]),
            (logical._OR, "|", [False, True, True, True]),
            (logical._XOR, "^", [False, True, True, False]),
        ):
            with self.subTest(operation=symbol):
                np.testing.assert_array_equal(descriptor.operation(
                    np.array([False, False, True, True]), np.array([False, True, False, True])), expected)
                self.assertEqual(descriptor.name_func("A", "B"), f"{{A {symbol} B}}")
                self.assertEqual(descriptor.supported_input_types, ((np.bool_,),))

    def test_inversion_uses_boolean_dtype_requirement_and_name(self):
        np.testing.assert_array_equal(logical._INVERT.operation(np.array([True, False])), [False, True])
        self.assertEqual(logical._INVERT.name_func("E"), "{~E}")
        self.assertEqual(logical._INVERT.supported_input_types, ((np.bool_,),))
