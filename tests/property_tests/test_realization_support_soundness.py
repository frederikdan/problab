import unittest

import numpy as np
import sympy as sp

from problab.operations import _arithmetic as arithmetic
from problab.value_sets import HomogeneousNumericValueSet
from problab.value_sets._unknown import _UnknownValueSet


class RealizationSupportSoundnessTests(unittest.TestCase):
    def support(self, values):
        exact = tuple(sp.Integer(int(v)) if np.issubdtype(values.dtype, np.integer)
                      else sp.Rational(*v.as_integer_ratio()) for v in values)
        return HomogeneousNumericValueSet(sp.FiniteSet(*exact), (values.dtype.type,))

    def assert_supported(self, inferred, actual):
        if inferred.dtype_types is not None:
            self.assertTrue(any(np.issubdtype(actual.dtype, t) for t in inferred.dtype_types), (actual.dtype, inferred))
        for value in actual.reshape(-1):
            if np.isfinite(value) and isinstance(inferred.sympy_set, _UnknownValueSet):
                # An unknown finite support makes no containment promise.
                continue
            self.assertTrue(inferred.contains(value), (value, inferred))

    def test_integer_arithmetic_support_covers_deterministic_boundary_batches(self):
        left = np.array([-128, -127, -100, -3, -1, 0, 1, 3, 100, 127], dtype=np.int8)
        right = np.array([127, -3, 100, -128, 0, 1, -1, 3, 2, 127], dtype=np.int8)
        for operation in (arithmetic._ADD, arithmetic._SUBTRACT, arithmetic._MULTIPLY,
                          arithmetic._DIVIDE, arithmetic._MODULO):
            with self.subTest(operation=operation.name_func("X", "Y")), np.errstate(all="ignore"):
                inferred = operation.infer_realization_value_set(self.support(left), self.support(right))
                self.assert_supported(inferred, operation.operation(left, right))

    def test_integer_power_support_covers_positive_negative_even_odd_and_zero_cases(self):
        bases = np.array([-3, -2, -1, 0, 1, 2, 3, 12], dtype=np.int8)
        for exponent in (0, 1, 2, 3, 7, 8, 9):
            powers = np.full(bases.shape, exponent, dtype=np.int8)
            for operation in (arithmetic._POWER, arithmetic._REAL_POWER):
                with self.subTest(exponent=exponent, operation=operation), np.errstate(all="ignore"):
                    inferred = operation.infer_realization_value_set(self.support(bases), self.support(powers))
                    self.assert_supported(inferred, operation.operation(bases, powers))

    def test_unary_integer_support_covers_signed_minimum_and_unsigned_values(self):
        for values in (np.array([-128, -127, 0, 127], dtype=np.int8), np.array([0, 1, 255], dtype=np.uint8)):
            for operation in (arithmetic._NEGATIVE, arithmetic._ABS):
                with self.subTest(dtype=values.dtype, operation=operation), np.errstate(over="ignore"):
                    self.assert_supported(operation.infer_realization_value_set(self.support(values)), operation.operation(values))

    def test_floating_arithmetic_support_covers_overflow_underflow_and_zero_division(self):
        for dtype in (np.float16, np.float32, np.float64):
            maximum = np.finfo(dtype).max
            tiny = np.nextafter(dtype(0), dtype(1))
            left = np.array([maximum, -maximum, tiny, -tiny, 0, 1], dtype=dtype)
            right = np.array([maximum, maximum, tiny, tiny, 0, 0], dtype=dtype)
            for operation in (arithmetic._ADD, arithmetic._SUBTRACT, arithmetic._MULTIPLY,
                              arithmetic._DIVIDE, arithmetic._MODULO):
                with self.subTest(dtype=dtype, operation=operation), np.errstate(all="ignore"):
                    inferred = operation.infer_realization_value_set(self.support(left), self.support(right))
                    self.assert_supported(inferred, operation.operation(left, right))
