import unittest
from itertools import product

import numpy as np
import sympy as sp

from problab._operations import _ADD, _SUBTRACT, _MULTIPLY, _DIVIDE, _MODULO, _POWER, _NEGATIVE, _ABS
from problab.random_variables._context import _RealizationContext
from problab.random_variables._nodes import _ConstantNode, _OperationNode
from problab.value_sets import _inference as inference
from problab.value_sets import sets as sets
from problab.value_sets._utils import is_known_subset
from problab.value_sets.base import ValueSet, _UnknownValueSet


class ValueSetInferenceTests(unittest.TestCase):
    def constant_set(self, value):
        return ValueSet(sp.FiniteSet(value), (np.asarray(value).dtype.type,))

    def assert_result_contains(self, value_set, values):
        self.assertIsInstance(value_set, ValueSet)
        self.assertNotIsInstance(value_set.sympy_set, _UnknownValueSet)
        if value_set.dtype_types is not None:
            self.assertTrue(any(np.issubdtype(values.dtype, allowed) for allowed in value_set.dtype_types))
        for value in values:
            # Exact rationals avoid SymPy's undecidable Float membership in Integers.
            if np.iscomplexobj(value):
                symbolic_value = sp.Rational(float(value.real)) + sp.I * sp.Rational(float(value.imag))
            else:
                symbolic_value = sp.Rational(float(value))
            self.assertIs(value_set.sympy_set.contains(symbolic_value), sp.true, (value_set, value))

    def test_addition_identity_is_zero_not_one(self):
        two = self.constant_set(2)
        result = inference._infer_add_value_set(two, sets.ONE)
        self.assertIs(result.sympy_set.contains(3), sp.true)
        for left, right in ((two, sets.ZERO), (sets.ZERO, two)):
            self.assertEqual(inference._infer_add_value_set(left, right).sympy_set, two.sympy_set)

    def test_zero_division_returns_wrapper(self):
        for infer in (inference._infer_divide_value_set, inference._infer_floor_divide_value_set, inference._infer_modulo_value_set):
            self.assertEqual(infer(sets.ZERO, sets.ONE).sympy_set, sets.ZERO.sympy_set)

    def test_possible_zero_divisors_are_unknown(self):
        for infer, divisor in product(
            (inference._infer_divide_value_set, inference._infer_floor_divide_value_set, inference._infer_modulo_value_set),
            (sets.ZERO, sets.INTEGERS, sets.REALS, sets.UNKNOWN_VALUE_SET),
        ):
            with self.subTest(infer=infer.__name__, divisor=divisor):
                self.assertIs(infer(sets.ZERO, divisor), sets.UNKNOWN_VALUE_SET)

    def test_nonreal_fractional_power(self):
        result = inference._infer_power_value_set(sets.NEGATIVE_REALS, sets.NON_INTEGER_REALS)
        self.assertIsInstance(result, ValueSet)
        self.assertIs(result.sympy_set.contains(sp.I), sp.true)
        self.assertIs(result.sympy_set.contains(1), sp.false)

    def test_power_undefined_at_zero(self):
        for base, exponent in product(
            (sets.ZERO, sets.REALS, sets.COMPLEXES),
            (sets.NEGATIVE_REALS, sets.REALS, sets.COMPLEXES),
        ):
            self.assertIs(inference._infer_power_value_set(base, exponent), sets.UNKNOWN_VALUE_SET)

    def test_power_zero_and_one_conventions(self):
        self.assertEqual(inference._infer_power_value_set(sets.ZERO, sets.ZERO).sympy_set, sets.ONE.sympy_set)
        self.assertEqual(inference._infer_power_value_set(sets.ZERO, sets.POSITIVE_REALS).sympy_set, sets.ZERO.sympy_set)
        self.assertEqual(inference._infer_power_value_set(sets.ONE, sets.COMPLEXES).sympy_set, sets.ONE.sympy_set)

    def test_unknown_inputs_remain_unknown(self):
        binary = (
            inference._infer_add_value_set, inference._infer_subtract_value_set,
            inference._infer_multiply_value_set, inference._infer_divide_value_set,
            inference._infer_floor_divide_value_set, inference._infer_modulo_value_set,
            inference._infer_power_value_set,
        )
        unknown = ValueSet(_UnknownValueSet(), (np.floating,))
        for infer in binary:
            self.assertIs(infer(unknown, sets.ONE), sets.UNKNOWN_VALUE_SET)
            self.assertIs(infer(sets.ONE, unknown), sets.UNKNOWN_VALUE_SET)
        for infer in (inference._infer_negative_value_set, inference._infer_absolute_value_set):
            self.assertIs(infer(unknown), sets.UNKNOWN_VALUE_SET)

    def test_integer_information_and_sign_are_preserved(self):
        cases = (
            (inference._infer_add_value_set, (sets.NATURALS_0, sets.POSITIVE_INTEGERS), sets.POSITIVE_INTEGERS),
            (inference._infer_subtract_value_set, (sets.POSITIVE_INTEGERS, sets.NEGATIVE_INTEGERS), sets.POSITIVE_INTEGERS),
            (inference._infer_multiply_value_set, (sets.NEGATIVE_INTEGERS, sets.NEGATIVE_INTEGERS), sets.POSITIVE_INTEGERS),
            (inference._infer_negative_value_set, (sets.POSITIVE_INTEGERS,), sets.NEGATIVE_INTEGERS),
            (inference._infer_absolute_value_set, (sets.NEGATIVE_INTEGERS,), sets.POSITIVE_INTEGERS),
            (inference._infer_power_value_set, (sets.POSITIVE_INTEGERS, sets.NATURALS_0), sets.POSITIVE_INTEGERS),
            (inference._infer_power_value_set, (sets.INTEGERS, sets.NATURALS_0), sets.INTEGERS),
        )
        for infer, operands, expected in cases:
            with self.subTest(infer=infer.__name__):
                self.assertTrue(is_known_subset(infer(*operands), expected))

    def test_identity_rules_allow_numpy_dtype_promotion(self):
        for operation, left, right in (
                (_ADD, 0j, 2), (_ADD, 2, 0j), (_SUBTRACT, 2, 0j),
                (_MULTIPLY, 1j * 0 + 1, 2), (_MULTIPLY, 2, 1.0),
                (_DIVIDE, 2, 1), (_POWER, 2, 1.0),
        ):
            with self.subTest(operation=operation, left=left, right=right):
                result = operation.infer_output_value_set(self.constant_set(left), self.constant_set(right))
                actual = operation.operation(np.array([left]), np.array([right]))
                self.assert_result_contains(result, actual)

    def test_real_output_from_complex_absolute_value(self):
        result = inference._infer_absolute_value_set(self.constant_set(3 + 4j))
        self.assert_result_contains(result, np.abs(np.array([3 + 4j])))
        self.assertFalse(any(np.issubdtype(np.dtype(complex), allowed) for allowed in result.dtype_types))

    def test_object_arithmetic_representation(self):
        operand = ValueSet(sp.FiniteSet(sp.Rational(1, 2)), (np.object_,))
        result = inference._infer_add_value_set(operand, operand)
        self.assertIn(np.object_, result.dtype_types)

    def test_unknown_dtype_stays_unspecified(self):
        operand = ValueSet(sp.S.Integers, None)
        result = inference._infer_add_value_set(operand, sets.ONE)
        self.assertIsNone(result.dtype_types)
        self.assertTrue(is_known_subset(result, sets.INTEGERS))

    def test_exact_dtype_rules_follow_the_operation(self):
        cases = (
            (_ADD, np.int64, np.int64, np.int64),
            (_ADD, np.float32, np.float32, np.float32),
            (_ADD, np.int64, np.uint64, np.float64),
            (_DIVIDE, np.int64, np.int64, np.float64),
            (_MULTIPLY, np.complex64, np.float32, np.complex64),
        )
        for operation, left_dtype, right_dtype, expected_dtype in cases:
            with self.subTest(operation=operation, left=left_dtype, right=right_dtype):
                left = ValueSet(sp.FiniteSet(2), (left_dtype,))
                right = ValueSet(sp.FiniteSet(1), (right_dtype,))
                result = operation.infer_output_value_set(left, right)
                self.assertEqual(result.dtype_types, (expected_dtype,))

        result = _ABS.infer_output_value_set(ValueSet(sp.FiniteSet(1j), (np.complex64,)))
        self.assertEqual(result.dtype_types, (np.float32,))

    def test_integer_family_includes_signed_unsigned_promotion(self):
        operand = ValueSet(sp.S.Integers, (np.integer,))
        result = _ADD.infer_output_value_set(operand, operand)
        self.assertIn(np.float64, result.dtype_types)
        self.assertNotIn(np.complex128, result.dtype_types)

    def test_custom_operations_do_not_guess_dtype(self):
        for operation in (_MODULO, _POWER):
            result = operation.infer_output_value_set(self.constant_set(2), self.constant_set(1))
            self.assertIsNone(result.dtype_types)

    def test_unsupported_representation_leaves_dtype_unknown(self):
        operand = ValueSet(sp.S.Reals, (np.void,))
        self.assertIsNone(_ADD.infer_output_value_set(operand, operand).dtype_types)

    def test_float_families_include_extended_precision(self):
        operand = ValueSet(sp.S.Reals, (np.floating,))
        result = _ADD.infer_output_value_set(operand, operand)
        for dtype in (np.float16, np.float32, np.float64, np.longdouble):
            actual = np.add(np.array([1], dtype=dtype), np.array([2], dtype=dtype))
            self.assertTrue(any(np.issubdtype(actual.dtype, allowed) for allowed in result.dtype_types))

    def test_representative_binary_results(self):
        domains = (
            (sets.POSITIVE_INTEGERS, np.array([1, 2, 3])),
            (sets.NEGATIVE_INTEGERS, np.array([-1, -2, -3])),
            (sets.NATURALS_0, np.array([0, 1, 2])),
            (sets.NON_POSITIVE_INTEGERS, np.array([0, -1, -2])),
            (sets.POSITIVE_REALS, np.array([0.5, 1.5, 2.5])),
            (sets.NEGATIVE_REALS, np.array([-0.5, -1.5, -2.5])),
        )
        operations = (
            (_ADD.infer_output_value_set, _ADD.operation),
            (_SUBTRACT.infer_output_value_set, _SUBTRACT.operation),
            (_MULTIPLY.infer_output_value_set, _MULTIPLY.operation),
            (_DIVIDE.infer_output_value_set, _DIVIDE.operation),
            (_MODULO.infer_output_value_set, _MODULO.operation),
            (inference._infer_floor_divide_value_set, np.floor_divide),
        )
        for (left_set, left), (right_set, right), (infer, operation) in product(domains, domains, operations):
            result = infer(left_set, right_set)
            if isinstance(result.sympy_set, _UnknownValueSet):
                continue
            with self.subTest(infer=infer.__name__, left=left_set, right=right_set):
                self.assert_result_contains(result, operation(left, right))

    def test_representative_unary_results(self):
        for value_set, values in (
            (sets.INTEGERS, np.array([-2, 0, 2])),
            (sets.REALS, np.array([-0.5, 0, 0.5])),
            (sets.COMPLEXES, np.array([3 + 4j, 0, -3 - 4j])),
        ):
            for operation in (_NEGATIVE, _ABS):
                self.assert_result_contains(operation.infer_output_value_set(value_set), operation.operation(values))

    def test_representative_power_results(self):
        cases = (
            (sets.POSITIVE_REALS, [0.5, 2.0], sets.REALS, [-1.0, 0.5]),
            (sets.NON_NEGATIVE_REALS, [0.0, 2.0], sets.POSITIVE_REALS, [0.5, 2.0]),
            (sets.INTEGERS, [-2, 0], sets.POSITIVE_EVEN_INTEGERS, [2, 4]),
            (sets.REALS, [-0.5, 0.0], sets.POSITIVE_EVEN_INTEGERS, [2, 4]),
            (sets.NON_ZERO_REALS, [-2.0, 2.0], sets.NEGATIVE_EVEN_INTEGERS, [-2, -4]),
            (sets.NEGATIVE_INTEGERS, [-2, -3], sets.POSITIVE_ODD_INTEGERS, [1, 3]),
            (sets.REALS, [-0.5, 0.5], sets.POSITIVE_ODD_INTEGERS, [1, 3]),
            (sets.NON_ZERO_REALS, [-2.0, 2.0], sets.NEGATIVE_ODD_INTEGERS, [-1, -3]),
            (sets.NEGATIVE_REALS, [-4.0, -9.0], sets.NON_INTEGER_REALS, [0.5, 0.5]),
            (sets.NON_ZERO_COMPLEXES, [1j, -1j], sets.COMPLEXES, [2, 3]),
            (sets.COMPLEXES, [0j, 1j], sets.NATURALS_0, [0, 2]),
        )
        for base_set, bases, exponent_set, exponents in cases:
            with self.subTest(base=base_set, exponent=exponent_set):
                result = _POWER.infer_output_value_set(base_set, exponent_set)
                actual = _POWER.operation(np.asarray(bases), np.asarray(exponents))
                self.assert_result_contains(result, actual)

    def test_context_accepts_promoted_outputs(self):
        for operation, left, right, expected in (
            (_DIVIDE, 2, 1, 2.0),
            (_ADD, 2, 0j, 2 + 0j),
            (_POWER, -4, sp.Rational(1, 2), 2j),
            (_POWER, 2, -1, 0.5),
        ):
            # Use a float sample for the fractional exponent, with its exact mathematical set.
            left_node = _ConstantNode(left)
            right_node = _ConstantNode(float(right) if isinstance(right, sp.Rational) else right)
            exponent_set = ValueSet(sp.FiniteSet(right), right_node.value_set.dtype_types)
            node = _OperationNode(
                operation=operation.operation,
                inputs=(left_node, right_node),
                name="inference regression",
                value_set=operation.infer_output_value_set(left_node.value_set, exponent_set),
            )
            actual = _RealizationContext(node, num_samples=3).evaluate(node)
            np.testing.assert_allclose(actual, np.full(3, expected), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
