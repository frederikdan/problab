import math
import unittest
from fractions import Fraction

import numpy as np

from problab._config import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE
from problab._utils import _convert_fractions_in_array, _is_close


class NumericalUtilityTests(unittest.TestCase):

    def test_fraction_array_conversion_reuses_arrays_without_fractions(self):
        for values in (np.array([1, 2]), np.array([0.5], dtype=np.float32),
                       np.array([1 + 2j]), np.array(["label"]),
                       np.array([2**60 + 1], dtype=object), np.empty((0, 2), dtype=object)):
            with self.subTest(dtype=values.dtype, shape=values.shape):
                self.assertIs(_convert_fractions_in_array(values), values)

    def test_fraction_array_conversion_preserves_shape_and_readonly_source(self):
        for shape in ((), (1,), (1, 1)):
            values = np.array(Fraction(1, 3), dtype=object).reshape(shape)
            values.flags.writeable = False
            result = _convert_fractions_in_array(values)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, np.dtype(np.float64))
            np.testing.assert_array_equal(result, np.full(shape, float(Fraction(1, 3))))
            self.assertIsInstance(values.flat[0], Fraction)
            self.assertFalse(values.flags.writeable)
        values = np.array([[Fraction(1, 2), 2], [np.inf, np.nan]], dtype=object)
        np.testing.assert_array_equal(_convert_fractions_in_array(values), [[0.5, 2.0], [np.inf, np.nan]])

    def test_fraction_array_conversion_retains_large_integers_in_object_storage(self):
        for large in (2**53 + 1, -(2**53 + 1), 2**64 - 1, 10**400):
            with self.subTest(large=large):
                values = np.array([Fraction(1, 3), large], dtype=object)
                result = _convert_fractions_in_array(values)
                self.assertEqual(result.dtype, np.dtype(object))
                self.assertEqual(result[0], float(Fraction(1, 3)))
                self.assertIs(type(result[0]), float)
                self.assertEqual(result[1], large)
                self.assertIs(type(result[1]), int)
                self.assertIsInstance(values[0], Fraction)

    def test_fraction_array_conversion_preserves_nonnumeric_objects(self):
        for value in ("label", [1, 2], object()):
            values = np.empty(2, dtype=object)
            values[0] = Fraction(1, 3)
            values[1] = value
            result = _convert_fractions_in_array(values)
            self.assertEqual(result.dtype, np.dtype(object))
            self.assertEqual(result[0], float(Fraction(1, 3)))
            self.assertIs(result[1], value)

    def test_fraction_array_conversion_matches_scalar_underflow_and_overflow(self):
        tiny = np.array([Fraction(1, 10**400)], dtype=object)
        np.testing.assert_array_equal(_convert_fractions_in_array(tiny), [0.0])
        huge = np.array([Fraction(10**400)], dtype=object)
        with self.assertRaises(OverflowError):
            _convert_fractions_in_array(huge)
        self.assertIsInstance(huge[0], Fraction)

    def test_is_close_accepts_exact_and_within_tolerance_values(self):
        self.assertTrue(_is_close(1.0, 1.0))
        self.assertTrue(_is_close(0.0, ABSOLUTE_TOLERANCE / 2))
        self.assertTrue(_is_close(1.0, 1.0 + RELATIVE_TOLERANCE / 2))

    def test_is_close_rejects_nan_and_values_outside_tolerance(self):
        self.assertFalse(_is_close(math.nan, 1.0))
        self.assertFalse(
            _is_close(
                1.0,
                1.0 + 2 * RELATIVE_TOLERANCE + ABSOLUTE_TOLERANCE,
            )
        )


if __name__ == "__main__":
    unittest.main()
