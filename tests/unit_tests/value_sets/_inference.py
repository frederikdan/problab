import unittest
from unittest.mock import Mock

import numpy as np
import sympy as sp

from problab.value_sets._inference import (
    _arithmetic_inference, _infer_dtype_types, _input_dtype_candidates,
)
from problab.value_sets import HomogeneousNumericValueSet, ObjectValueSet
from problab.value_sets.sets import UNKNOWN_VALUE_SET


class SharedInferenceTests(unittest.TestCase):
    def value_set(self, dtype_types, sympy_set=sp.S.Reals):
        return HomogeneousNumericValueSet(sympy_set, dtype_types)

    def test_concrete_dtype_candidates_preserve_input_order_and_remove_duplicates(self):
        candidates = _input_dtype_candidates((self.value_set((np.float32, np.float32)), self.value_set((np.int8,))))
        self.assertEqual(candidates, ((np.dtype("float32"),), (np.dtype("int8"),)))

    def test_unknown_and_unsupported_dtype_candidates_return_none(self):
        class Unsupported(np.generic):
            pass
        for dtype_types in (None, (Unsupported,)):
            with self.subTest(dtype_types=dtype_types):
                self.assertIsNone(_input_dtype_candidates((self.value_set(dtype_types),)))

    def test_numeric_families_exclude_datetime_and_timedelta_storage(self):
        for family in (np.integer, np.number):
            with self.subTest(family=family):
                candidates = _input_dtype_candidates((self.value_set((family,)),))[0]
                self.assertTrue(candidates)
                self.assertTrue(all(dtype.kind in "iufc" for dtype in candidates), candidates)

    def test_dtype_resolution_matches_numpy_for_concrete_numeric_inputs(self):
        for operation, types, expected in (
            (np.add, (np.float32, np.float32), np.float32),
            (np.add, (np.int64, np.uint64), np.float64),
            (np.divide, (np.int32, np.int32), np.float64),
            (np.absolute, (np.complex64,), np.float32),
            (np.exp, (np.float64,), np.float64),
        ):
            with self.subTest(operation=operation.__name__, types=types):
                self.assertEqual(_infer_dtype_types(operation, tuple(self.value_set((t,)) for t in types)), (expected,))

    def test_real_numeric_families_can_resolve_exponential_output_dtypes(self):
        types = _infer_dtype_types(np.exp, (self.value_set((np.integer, np.floating)),))
        self.assertIsNotNone(types)
        self.assertIn(np.float64, types)
        self.assertTrue(all(issubclass(t, np.floating) for t in types))

    def test_abs_of_generic_numeric_family_can_resolve_real_dtypes(self):
        types = _infer_dtype_types(np.absolute, (self.value_set((np.number,), sp.S.Complexes),))
        self.assertIsNotNone(types)
        self.assertTrue(all(not issubclass(t, np.complexfloating) for t in types))

    def test_missing_ufunc_or_unsupported_loop_leaves_dtype_unspecified(self):
        values = (self.value_set((np.object_,)),)
        self.assertIsNone(_infer_dtype_types(None, values))
        self.assertEqual(_infer_dtype_types(np.exp, values), (np.object_,))
        self.assertIsNone(_infer_dtype_types(np.bitwise_and, (self.value_set((np.float64,)),) * 2))

    def test_arithmetic_decorator_rejects_unknown_and_object_support_before_delegate(self):
        delegate = Mock(return_value=self.value_set((np.float64,)))
        infer = _arithmetic_inference(dtype_operation=np.add)(delegate)
        for invalid in (UNKNOWN_VALUE_SET, ObjectValueSet(("red",))):
            with self.subTest(value_set=invalid):
                self.assertIs(infer(invalid, self.value_set((np.float64,))), UNKNOWN_VALUE_SET)
        delegate.assert_not_called()

    def test_arithmetic_decorator_keeps_unknown_result_and_function_metadata(self):
        def original(left, right):
            """Infer finite support."""
            return UNKNOWN_VALUE_SET
        infer = _arithmetic_inference(dtype_operation=np.add)(original)
        self.assertIs(infer(self.value_set((np.float64,)), self.value_set((np.float64,))), UNKNOWN_VALUE_SET)
        self.assertEqual(infer.__name__, original.__name__)
        self.assertEqual(infer.__doc__, original.__doc__)

    def test_arithmetic_decorator_intersects_integer_inputs_and_resolves_named_operands(self):
        def infer(left, right):
            return self.value_set((np.floating,), sp.Interval(2, 5))
        wrapped = _arithmetic_inference(dtype_operation=np.add, preserves_integers=True)(infer)
        integers = self.value_set((np.int8,), sp.S.Integers)
        result = wrapped(integers, right=integers)
        self.assertEqual(result.sympy_set.symmetric_difference(sp.FiniteSet(2, 3, 4, 5)), sp.EmptySet)
        self.assertEqual(result.dtype_types, (np.int8,))
