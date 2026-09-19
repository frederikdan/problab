import unittest

import numpy as np

from problab._operations import (
    _ADD,
    _AND,
    _ABS,
    _DIVIDE,
    _EQ,
    _GTE,
    _INVERT,
    _LT,
    _MODULO,
    _NEGATIVE,
    _OR,
    _POWER,
    _Operation,
    _ArithmeticOperation,
    _ComparisonOperation,
    _LogicalOperation,
)


class OperationTests(unittest.TestCase):

    def test_operation_constants_have_expected_types_and_names(self):
        self.assertIsInstance(_ADD, _ArithmeticOperation)
        self.assertIsInstance(_AND, _LogicalOperation)
        self.assertIsInstance(_EQ, _ComparisonOperation)
        self.assertIsInstance(_ADD, _Operation)
        self.assertEqual(_ADD.name_func("x", "y"), "(x + y)")
        self.assertEqual(_AND.name_func("a", "b"), "{a & b}")
        self.assertEqual(_EQ.name_func("a", "b"), "{a = b}")

    def test_divide_and_modulo_handle_zero_without_warnings(self):
        np.testing.assert_array_equal(
            _DIVIDE.operation(np.array([1.0, 0.0]), np.array([0.0, 0.0])),
            [np.inf, np.nan],
        )
        result = _MODULO.operation(np.array([3.0, 1.0]), np.array([2.0, 0.0]))
        self.assertEqual(result[0], 1.0)
        self.assertTrue(np.isnan(result[1]))

    def test_power_promotes_integer_base_for_negative_exponents_and_handles_complex_output(self):
        reciprocal = _POWER.operation(np.array([2], dtype=int), np.array([-1]))
        complex_root = _POWER.operation(np.array([-4.0]), np.array([0.5]))

        np.testing.assert_allclose(reciprocal, [0.5])
        np.testing.assert_allclose(complex_root, [2j])

    def test_logical_and_comparison_operations_apply_elementwise(self):
        left = np.array([True, False])
        right = np.array([False, False])
        values = np.array([1, 3])

        np.testing.assert_array_equal(_AND.operation(left, right), [False, False])
        np.testing.assert_array_equal(_OR.operation(left, right), [True, False])
        np.testing.assert_array_equal(_INVERT.operation(left), [False, True])
        np.testing.assert_array_equal(_LT.operation(values, 2), [True, False])
        np.testing.assert_array_equal(_GTE.operation(values, 2), [False, True])

    def test_unary_arithmetic_operations_preserve_expected_values_and_names(self):
        values = np.array([-2.0, 3.0])

        np.testing.assert_array_equal(_NEGATIVE.operation(values), [2.0, -3.0])
        np.testing.assert_array_equal(_ABS.operation(values), [2.0, 3.0])
        self.assertEqual(_NEGATIVE.name_func("X"), "(-X)")
        self.assertEqual(_ABS.name_func("X"), "abs(X)")


if __name__ == "__main__":
    unittest.main()
