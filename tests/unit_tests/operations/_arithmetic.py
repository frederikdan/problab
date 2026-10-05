import unittest
import warnings

import numpy as np

from problab.operations import _arithmetic as arithmetic
from problab.value_sets import _mathematical_inference as mathematical
from problab.value_sets import _realization_inference as realization


class ArithmeticOperationTests(unittest.TestCase):
    def test_descriptors_connect_each_operation_to_both_support_inferers(self):
        for name, suffix, expression in (
            ("_ADD", "add", "(X + Y)"),
            ("_SUBTRACT", "subtract", "(X - Y)"),
            ("_MULTIPLY", "multiply", "(X * Y)"),
            ("_DIVIDE", "divide", "(X / Y)"),
            ("_MODULO", "modulo", "(X mod Y)"),
            ("_POWER", "power", "(X ** Y)"),
            ("_REAL_POWER", "real_power", "(X ** Y)"),
        ):
            with self.subTest(operation=name):
                descriptor = getattr(arithmetic, name)
                self.assertIsInstance(descriptor, arithmetic._ArithmeticOperation)
                self.assertEqual(descriptor.name_func("X", "Y"), expression)
                self.assertIs(descriptor.infer_mathematical_value_set,
                              getattr(mathematical, f"_infer_{'power' if suffix == 'real_power' else suffix}_value_set"))
                self.assertIs(descriptor.infer_realization_value_set,
                              getattr(realization, f"_infer_{suffix}_value_set"))
                self.assertIsNotNone(descriptor.supported_input_types)

    def test_bounded_integer_arithmetic_preserves_dtype_and_broadcasts(self):
        for dtype in (np.int8, np.int64, np.uint8, np.uint64):
            for function, expected in (
                (arithmetic._add, [[4, 5], [5, 6]]),
                (arithmetic._subtract, [[0, 1], [-1, 0]]) if np.issubdtype(dtype, np.signedinteger)
                else (arithmetic._subtract, [[0, 1], [0, 1]]),
                (arithmetic._multiply, [[4, 6], [6, 9]]),
            ):
                with self.subTest(dtype=dtype, operation=function.__name__):
                    left = np.array([2, 3], dtype=dtype)
                    right = np.array([[2], [3]], dtype=dtype)
                    if function is arithmetic._subtract and np.issubdtype(dtype, np.unsignedinteger):
                        right = np.array([[2], [2]], dtype=dtype)
                    with np.errstate(over="raise"):
                        result = function(left, right)
                    np.testing.assert_array_equal(result, expected)
                    self.assertEqual(result.dtype, np.dtype(dtype))

    def test_integer_overflow_raises_for_binary_and_unary_operations(self):
        for function, arguments in (
            (arithmetic._add, (np.array([127], dtype=np.int8), np.array([1], dtype=np.int8))),
            (arithmetic._subtract, (np.array([-128], dtype=np.int8), np.array([1], dtype=np.int8))),
            (arithmetic._multiply, (np.array([100], dtype=np.int8), np.array([2], dtype=np.int8))),
            (arithmetic._power, (np.array([-3], dtype=np.int8), np.array([5], dtype=np.int8))),
            (arithmetic._negative, (np.array([-128], dtype=np.int8),)),
            (arithmetic._absolute, (np.array([-128], dtype=np.int8),)),
            (arithmetic._negative, (np.array([1], dtype=np.uint8),)),
        ):
            with self.subTest(operation=function.__name__, dtype=arguments[0].dtype):
                with np.errstate(over="raise"), self.assertRaises(OverflowError):
                    function(*arguments)

    def test_integer_overflow_warns_and_converts_entire_array_to_float64(self):
        with np.errstate(over="warn"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = arithmetic._add(np.array([-120, 120, 2], dtype=np.int8),
                                     np.array([-20, 20, 3], dtype=np.int8))
        np.testing.assert_array_equal(result, [-np.inf, np.inf, 5.0])
        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RuntimeWarning)
        for text in ("addition", "int8", "float64", "rounded"):
            self.assertIn(text, str(caught[0].message))

    def test_integer_overflow_ignore_returns_infinity_without_warning(self):
        with np.errstate(over="ignore"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = arithmetic._multiply(np.array([200], dtype=np.uint8),
                                          np.array([2], dtype=np.uint8))
        np.testing.assert_array_equal(result, [np.inf])
        self.assertEqual(caught, [])

    def test_no_overflow_preserves_integers_too_large_for_float64(self):
        value = 2 ** 53 + 1
        for policy in ("warn", "ignore", "raise"):
            with self.subTest(policy=policy), np.errstate(over=policy):
                result = arithmetic._add(np.array([value], dtype=np.int64),
                                         np.array([2], dtype=np.int64))
                self.assertEqual(result.dtype, np.dtype(np.int64))
                self.assertEqual(int(result[0]), value + 2)

    def test_overflow_fallback_can_round_unaffected_large_integers_as_warning_states(self):
        large = 2 ** 53 + 1
        with np.errstate(over="warn"), self.assertWarnsRegex(RuntimeWarning, "other large integers may be rounded"):
            result = arithmetic._add(np.array([np.iinfo(np.int64).max, large], dtype=np.int64),
                                     np.array([1, 0], dtype=np.int64))
        np.testing.assert_array_equal(result, [np.inf, float(large)])
        self.assertEqual(int(result[1]), 2 ** 53)

    def test_overflow_conversion_handles_scalar_and_empty_results(self):
        with np.errstate(over="ignore"):
            result = arithmetic._convert_integer_result(np.array(128, dtype=object), np.dtype("int8"), "test")
        self.assertEqual(result.shape, ())
        self.assertTrue(np.isposinf(result))
        empty = arithmetic._convert_integer_result(np.array([], dtype=object), np.dtype("int8"), "test")
        self.assertEqual(empty.dtype, np.dtype("int8"))
        self.assertEqual(empty.size, 0)

    def test_mixed_signed_unsigned_inputs_follow_numpy_float_promotion(self):
        result = arithmetic._add(np.array([-1], dtype=np.int64), np.array([2], dtype=np.uint64))
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(result, [1.0])

    def test_floating_overflow_honors_ambient_numpy_policy(self):
        for function, arguments in (
            (arithmetic._add, (np.array([1e308]), np.array([1e308]))),
            (arithmetic._subtract, (np.array([-1e308]), np.array([1e308]))),
            (arithmetic._multiply, (np.array([1e308]), np.array([2.0]))),
            (arithmetic._power, (np.array([1e308]), np.array([2.0]))),
        ):
            with self.subTest(operation=function.__name__):
                with np.errstate(over="raise"), self.assertRaises(FloatingPointError):
                    function(*arguments)
                with np.errstate(over="ignore"):
                    self.assertTrue(np.isinf(function(*arguments)).all())

    def test_division_distinguishes_zero_division_from_invalid_zero_over_zero(self):
        with np.errstate(divide="raise"), self.assertRaises(FloatingPointError):
            arithmetic._divide(np.array([1.0]), np.array([0.0]))
        with np.errstate(invalid="raise"), self.assertRaises(FloatingPointError):
            arithmetic._divide(np.array([0.0]), np.array([0.0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            result = arithmetic._divide(np.array([1.0, -1.0, 0.0]), np.zeros(3))
        np.testing.assert_array_equal(result, [np.inf, -np.inf, np.nan])

    def test_modulo_zero_is_nan_when_errors_are_ignored(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = arithmetic._modulo(np.array([3, 1]), np.array([2, 0]))
        np.testing.assert_array_equal(result, [1.0, np.nan])

    def test_negative_integer_exponents_promote_base_to_floating(self):
        result = arithmetic._power(np.array([2, 4]), np.array([-1, -2]))
        np.testing.assert_array_equal(result, [0.5, 0.0625])
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_ordinary_power_supports_complex_roots_and_real_power_rejects_them(self):
        np.testing.assert_allclose(arithmetic._power(np.array([-4.0]), np.array([0.5])), [2j])
        with np.errstate(invalid="raise"), self.assertRaises(FloatingPointError):
            arithmetic._real_power(np.array([-4.0]), np.array([0.5]))
        result = arithmetic._real_power(np.array([4.0]), np.array([0.5]))
        np.testing.assert_array_equal(result, [2.0])
        self.assertFalse(np.iscomplexobj(result))

    def test_unary_arithmetic_names_and_complex_absolute_values(self):
        self.assertEqual(arithmetic._NEGATIVE.name_func("X"), "(-X)")
        self.assertEqual(arithmetic._ABS.name_func("X"), "abs(X)")
        np.testing.assert_array_equal(arithmetic._negative(np.array([-2.0, 3.0])), [2.0, -3.0])
        np.testing.assert_array_equal(arithmetic._absolute(np.array([3 + 4j])), [5.0])
