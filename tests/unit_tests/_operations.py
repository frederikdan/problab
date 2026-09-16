import unittest

import numpy as np

from problab._operations import (
    _ADD,
    _AND,
    _DIVIDE,
    _EQ,
    _MODULO,
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


if __name__ == "__main__":
    unittest.main()
