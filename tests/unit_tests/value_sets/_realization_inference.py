import unittest

import numpy as np
import sympy as sp

from problab.value_sets import _realization_inference as inference
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet


class RealizationInferenceTests(unittest.TestCase):

    def test_bounded_integer_operations_preserve_dtype_when_overflow_is_impossible(self):
        for infer, values in ((inference._infer_add_value_set, (2, 3)),
                              (inference._infer_subtract_value_set, (5, 3)),
                              (inference._infer_multiply_value_set, (2, 3)),
                              (inference._infer_power_value_set, (2, 3)),
                              (inference._infer_real_power_value_set, (2, 3)),
                              (inference._infer_negative_value_set, (3,)),
                              (inference._infer_absolute_value_set, (-3,))):
            with self.subTest(infer=infer.__name__):
                result = infer(*(self.value_set(v, np.int8) for v in values))
                self.assertEqual(result.dtype_types, (np.int8,))
                self.assert_flags(result)

    def test_underflow_support_includes_zero_for_multiplication_division_and_power(self):
        for infer, values in ((inference._infer_multiply_value_set, (1e-200, 1e-200)),
                              (inference._infer_divide_value_set, (1e-200, 1e200)),
                              (inference._infer_power_value_set, (1e-200, 2)),
                              (inference._infer_real_power_value_set, (1e-200, 2))):
            with self.subTest(infer=infer.__name__):
                result = infer(*(self.value_set(v) for v in values))
                self.assertTrue(result.contains(0.0))

    def test_function_inference_preserves_finite_template_and_resolves_output_dtype(self):
        template = HomogeneousNumericValueSet(sp.Interval(-10, 10), (np.floating,))
        for name, values in (
            ("exp", (0,)), ("expm1", (0,)), ("log", (1,)), ("log2", (1,)), ("log10", (1,)),
            ("log1p", (0,)), ("sqrt", (1,)), ("floor", (0,)), ("ceil", (0,)), ("sign", (0,)),
            ("sin", (0,)), ("cos", (0,)), ("tan", (0,)), ("arcsin", (0,)), ("arccos", (0,)),
            ("arctan", (0,)), ("sinh", (0,)), ("cosh", (0,)), ("tanh", (0,)), ("arcsinh", (0,)),
            ("arccosh", (1,)), ("arctanh", (0,)), ("hypot", (3, 4)), ("logaddexp", (0, 1)),
        ):
            with self.subTest(function=name):
                result = getattr(inference, f"_infer_{name}_value_set")(template, *(self.value_set(v, np.float32) for v in values))
                self.assertEqual(result.sympy_set, template.sympy_set)
                self.assertEqual(result.dtype_types, (np.float32,))
                self.assert_flags(result)

    def test_exponential_and_hyperbolic_functions_declare_possible_overflow(self):
        template = HomogeneousNumericValueSet(sp.S.Reals, (np.floating,))
        for name, value, positive, negative in (("exp", 1000, True, False), ("expm1", 1000, True, False),
                                               ("sinh", 1000, True, False), ("sinh", -1000, False, True),
                                               ("cosh", -1000, True, False)):
            with self.subTest(function=name, value=value):
                result = getattr(inference, f"_infer_{name}_value_set")(template, self.value_set(value))
                self.assert_flags(result, positive, negative)

    def test_logs_declare_zero_division_invalid_negative_inputs_and_input_infinity(self):
        template = HomogeneousNumericValueSet(sp.S.Reals, (np.floating,))
        for name in ("log", "log2", "log10"):
            infer = getattr(inference, f"_infer_{name}_value_set")
            for value, positive, negative, nan in ((0, False, True, False), (-1, False, False, True), (1, False, False, False)):
                with self.subTest(function=name, value=value):
                    self.assert_flags(infer(template, self.value_set(value)), positive, negative, nan)
            self.assertTrue(infer(template, self.value_set(1, allows_positive_infinity=True)).allows_positive_infinity)
            # Negative infinity is an invalid logarithm input even if finite support is positive.
            self.assertTrue(infer(template, self.value_set(1, allows_negative_infinity=True)).allows_nan)

    def test_log1p_and_sqrt_distinguish_endpoints_from_invalid_domains(self):
        template = HomogeneousNumericValueSet(sp.S.Reals, (np.floating,))
        self.assert_flags(inference._infer_log1p_value_set(template, self.value_set(-1)), negative=True)
        self.assert_flags(inference._infer_log1p_value_set(template, self.value_set(-2)), nan=True)
        self.assert_flags(inference._infer_sqrt_value_set(template, self.value_set(-1)), nan=True)
        self.assert_flags(inference._infer_sqrt_value_set(template, self.value_set(0)))

    def test_inverse_trigonometric_and_hyperbolic_domains_declare_invalid_machine_inputs(self):
        template = HomogeneousNumericValueSet(sp.S.Reals, (np.floating,))
        for name in ("arcsin", "arccos"):
            self.assert_flags(getattr(inference, f"_infer_{name}_value_set")(template, self.value_set(2)), nan=True)
        self.assert_flags(inference._infer_arccosh_value_set(template, self.value_set(0)), nan=True)
        self.assert_flags(inference._infer_arctanh_value_set(template, self.value_set(1)), positive=True)
        self.assert_flags(inference._infer_arctanh_value_set(template, self.value_set(-1)), negative=True)
        self.assert_flags(inference._infer_arctanh_value_set(template, self.value_set(2)), nan=True)

    def test_infinite_trigonometric_inputs_produce_nan_but_bounded_inverse_outputs_do_not(self):
        template = HomogeneousNumericValueSet(sp.S.Reals, (np.floating,))
        for flag in ("allows_positive_infinity", "allows_negative_infinity"):
            operand = self.value_set(0, **{flag: True})
            for name in ("sin", "cos", "tan"):
                with self.subTest(function=name, flag=flag):
                    self.assertTrue(getattr(inference, f"_infer_{name}_value_set")(template, operand).allows_nan)
            for name in ("arctan", "tanh", "sign"):
                self.assert_flags(getattr(inference, f"_infer_{name}_value_set")(template, operand))

    def test_rounding_functions_and_arcsinh_preserve_signed_infinity_flags(self):
        template = HomogeneousNumericValueSet(sp.S.Reals, (np.floating,))
        operand = self.value_set(0, allows_positive_infinity=True, allows_negative_infinity=True, allows_nan=True)
        for name in ("floor", "ceil", "arcsinh"):
            with self.subTest(function=name):
                self.assert_flags(getattr(inference, f"_infer_{name}_value_set")(template, operand), True, True, True)

    def test_all_function_inferers_propagate_possible_nan_inputs(self):
        template = HomogeneousNumericValueSet(sp.S.Reals, (np.floating,))
        for name in ("exp", "expm1", "log", "log2", "log10", "log1p", "sqrt", "floor", "ceil", "sign",
                     "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
                     "arcsinh", "arccosh", "arctanh", "hypot", "logaddexp"):
            operands = (self.value_set(1, allows_nan=True),)
            if name in ("hypot", "logaddexp"):
                operands += (self.value_set(1),)
            with self.subTest(function=name):
                self.assertTrue(getattr(inference, f"_infer_{name}_value_set")(template, *operands).allows_nan)
                if name in ("hypot", "logaddexp"):
                    self.assertTrue(getattr(inference, f"_infer_{name}_value_set")(template, *operands[::-1]).allows_nan)

    def test_hypot_accounts_for_both_operands_and_floating_overflow(self):
        template = HomogeneousNumericValueSet(sp.Interval(0, sp.oo), (np.floating,))
        maximum = sp.Rational(*np.finfo(np.float64).max.as_integer_ratio())
        self.assert_flags(inference._infer_hypot_value_set(template, self.value_set(maximum), self.value_set(maximum)), positive=True)
        for reverse in (False, True):
            operands = (self.value_set(1, allows_negative_infinity=True), self.value_set(0))
            if reverse:
                operands = operands[::-1]
            self.assert_flags(inference._infer_hypot_value_set(template, *operands), positive=True)

    def test_logaddexp_is_stable_for_large_finite_values_and_tracks_signed_infinities(self):
        template = HomogeneousNumericValueSet(sp.S.Reals, (np.floating,))
        self.assert_flags(inference._infer_logaddexp_value_set(template, self.value_set(1e308), self.value_set(1e308)))
        negative = self.value_set(0, allows_negative_infinity=True)
        self.assert_flags(inference._infer_logaddexp_value_set(template, negative, negative), negative=True)
        self.assert_flags(inference._infer_logaddexp_value_set(template, negative, self.value_set(0)))

    def test_rounded_intervals_enclose_numpy_endpoint_values_for_each_precision(self):
        for dtype in (np.float16, np.float32, np.float64, np.longdouble):
            for interval, operation, values in (
                (sp.Interval(-sp.pi / 2, sp.pi / 2), np.arcsin, [-1, 1]),
                (sp.Interval(0, sp.pi), np.arccos, [-1, 1]),
                (sp.Interval.open(-sp.pi / 2, sp.pi / 2), np.arctan, [-np.inf, np.inf]),
            ):
                with self.subTest(dtype=dtype, operation=operation.__name__):
                    result = inference._infer_rounded_interval_value_set(interval, (dtype,))
                    self.assertTrue(result.sympy_set.is_superset(interval))
                    self.assertFalse(result.sympy_set.left_open)
                    self.assertFalse(result.sympy_set.right_open)
                    for value in operation(np.array(values, dtype=dtype)):
                        self.assertIs(result.sympy_set.contains(sp.Rational(*value.as_integer_ratio())), sp.true)

    def test_rounded_intervals_combine_precisions_and_cache_equal_requests(self):
        interval = sp.Interval(-sp.pi / 2, sp.pi / 2)
        result = inference._infer_rounded_interval_value_set(interval, (np.float32, np.float64))
        self.assertIs(result, inference._infer_rounded_interval_value_set(interval, (np.float32, np.float64)))
        for dtype in (np.float32, np.float64):
            self.assertTrue(result.sympy_set.is_superset(inference._infer_rounded_interval_value_set(interval, (dtype,)).sympy_set))

    def test_rounded_interval_rejects_empty_nonfloating_and_nonfinite_configurations(self):
        for interval, dtypes, exception in ((sp.Interval(0, 1), (), ValueError),
                                           (sp.Interval(0, 1), (np.int32,), TypeError),
                                           (sp.FiniteSet(0), (np.float64,), TypeError),
                                           (sp.Interval(0, sp.oo), (np.float64,), ValueError)):
            with self.subTest(interval=interval, dtypes=dtypes), self.assertRaises(exception):
                inference._infer_rounded_interval_value_set(interval, dtypes)
    def value_set(self, value, dtype=np.float64, **flags):
        return HomogeneousNumericValueSet(sp.FiniteSet(value), (dtype,), **flags)

    def assert_flags(self, value_set, positive=False, negative=False, nan=False):
        self.assertEqual(
            (value_set.allows_positive_infinity, value_set.allows_negative_infinity, value_set.allows_nan),
            (positive, negative, nan),
        )

    def test_addition_preserves_integer_overflow_direction(self):
        for value, positive, negative in ((120, True, False), (-120, False, True)):
            with self.subTest(value=value):
                operand = self.value_set(value, np.int8)
                result = inference._infer_add_value_set(operand, operand)
                self.assert_flags(result, positive, negative)
                self.assertEqual(result.dtype_types, (np.int8, np.float64))

    def test_addition_declares_floating_overflow(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                maximum = sp.Rational(*np.finfo(dtype).max.as_integer_ratio())
                operand = self.value_set(maximum, dtype)
                self.assert_flags(inference._infer_add_value_set(operand, operand), positive=True)

    def test_addition_and_subtraction_handle_infinite_cancellation(self):
        positive = self.value_set(1, allows_positive_infinity=True)
        negative = self.value_set(-1, allows_negative_infinity=True)
        self.assert_flags(inference._infer_add_value_set(positive, negative), True, True, True)
        self.assert_flags(inference._infer_subtract_value_set(positive, positive), True, True, True)

    def test_subtraction_preserves_integer_overflow_direction(self):
        for left, right, positive, negative in ((120, -120, True, False), (-120, 120, False, True)):
            with self.subTest(left=left, right=right):
                result = inference._infer_subtract_value_set(
                    self.value_set(left, np.int8), self.value_set(right, np.int8),
                )
                self.assert_flags(result, positive, negative)
                self.assertIn(np.float64, result.dtype_types)

    def test_multiplication_handles_overflow_and_zero_times_infinity(self):
        self.assert_flags(
            inference._infer_multiply_value_set(self.value_set(-100, np.int8), self.value_set(2, np.int8)),
            negative=True,
        )
        self.assert_flags(
            inference._infer_multiply_value_set(
                self.value_set(0), self.value_set(1, allows_positive_infinity=True),
            ),
            nan=True,
        )

    def test_division_handles_signed_zero_and_zero_over_zero(self):
        self.assert_flags(
            inference._infer_divide_value_set(self.value_set(1), self.value_set(0)),
            positive=True, negative=True,
        )
        self.assert_flags(
            inference._infer_divide_value_set(self.value_set(0), self.value_set(0)),
            nan=True,
        )

    def test_division_declares_overflow_and_infinity_over_infinity(self):
        self.assert_flags(
            inference._infer_divide_value_set(self.value_set(sp.Integer(10) ** 300), self.value_set(sp.Rational(1, 10**300))),
            positive=True,
        )
        operand = self.value_set(1, allows_positive_infinity=True)
        self.assert_flags(inference._infer_divide_value_set(operand, operand), positive=True, nan=True)

    def test_modulo_handles_zero_divisor_and_infinite_inputs(self):
        self.assert_flags(inference._infer_modulo_value_set(self.value_set(1), self.value_set(0)), nan=True)
        self.assert_flags(
            inference._infer_modulo_value_set(self.value_set(-1), self.value_set(1, allows_positive_infinity=True)),
            positive=True,
        )
        self.assert_flags(
            inference._infer_modulo_value_set(self.value_set(1, allows_positive_infinity=True), self.value_set(2)),
            nan=True,
        )

    def test_power_distinguishes_odd_and_even_integer_overflow(self):
        for exponent, positive, negative in ((7, False, False), (8, True, False), (9, False, True)):
            with self.subTest(exponent=exponent):
                result = inference._infer_real_power_value_set(
                    self.value_set(-2, np.int8), self.value_set(exponent, np.int8),
                )
                self.assert_flags(result, positive, negative)

    def test_power_infers_large_exponents_without_constructing_the_result(self):
        result = inference._infer_real_power_value_set(
            self.value_set(2, np.int64), self.value_set(np.iinfo(np.int64).max, np.int64),
        )
        self.assert_flags(result, positive=True)
        self.assertIn(np.float64, result.dtype_types)

    def test_positive_exponent_range_does_not_imply_negative_powers_of_zero(self):
        base = HomogeneousNumericValueSet(sp.Range(-1, 5), (np.int8,))
        exponent = HomogeneousNumericValueSet(sp.Range(7, 8), (np.int8,))
        self.assert_flags(inference._infer_real_power_value_set(base, exponent), positive=True)

    def test_floating_power_allows_infinity_sign_changes_from_exponent_rounding(self):
        result = inference._infer_real_power_value_set(
            self.value_set(-2, np.float64), self.value_set(2**53 + 1, np.int64),
        )
        self.assertTrue(result.allows_positive_infinity)

    def test_power_preserves_real_numpy_identity_rules_for_nan(self):
        result = inference._infer_real_power_value_set(
            self.value_set(2, allows_nan=True), self.value_set(0),
        )
        self.assert_flags(result)
        self.assertEqual(result.sympy_set, sp.FiniteSet(1))
        result = inference._infer_real_power_value_set(
            self.value_set(1), self.value_set(2, allows_nan=True),
        )
        self.assert_flags(result)

    def test_power_distinguishes_real_invalid_input_from_complex_output(self):
        base, exponent = self.value_set(-2), self.value_set(sp.Rational(1, 2))
        self.assert_flags(inference._infer_real_power_value_set(base, exponent), nan=True)
        self.assert_flags(inference._infer_power_value_set(base, exponent))

    def test_negation_handles_signed_minimum_unsigned_values_and_infinity(self):
        self.assert_flags(inference._infer_negative_value_set(self.value_set(-128, np.int8)), positive=True)
        self.assert_flags(inference._infer_negative_value_set(self.value_set(1, np.uint8)), negative=True)
        self.assert_flags(
            inference._infer_negative_value_set(self.value_set(1, allows_positive_infinity=True, allows_nan=True)),
            negative=True, nan=True,
        )

    def test_absolute_maps_both_infinity_signs_to_positive_infinity(self):
        self.assert_flags(inference._infer_absolute_value_set(self.value_set(-128, np.int8)), positive=True)
        self.assert_flags(
            inference._infer_absolute_value_set(self.value_set(-1, allows_negative_infinity=True, allows_nan=True)),
            positive=True, nan=True,
        )

    def test_bounded_real_operations_keep_exceptional_flags_disabled(self):
        two, three = self.value_set(2), self.value_set(3)
        for infer in (
            inference._infer_add_value_set, inference._infer_subtract_value_set,
            inference._infer_multiply_value_set, inference._infer_divide_value_set,
            inference._infer_modulo_value_set, inference._infer_power_value_set,
            inference._infer_real_power_value_set,
        ):
            with self.subTest(infer=infer.__name__):
                self.assert_flags(infer(two, three))
        self.assert_flags(inference._infer_negative_value_set(two))
        self.assert_flags(inference._infer_absolute_value_set(two))

    def test_complex_overflow_can_declare_nan_components(self):
        operand = self.value_set(sp.Integer(10) ** 300 * (1 + sp.I), np.complex128)
        for infer in (
            inference._infer_multiply_value_set, inference._infer_divide_value_set,
            inference._infer_power_value_set,
        ):
            with self.subTest(infer=infer.__name__):
                self.assertTrue(infer(operand, operand).allows_nan)

    def test_underflow_support_expansion_preserves_exceptional_flags(self):
        original = HomogeneousNumericValueSet(
            sp.Interval.open(0, sp.oo), (np.float64,),
            allows_positive_infinity=True, allows_negative_infinity=True, allows_nan=True,
        )
        result = inference._allow_underflow_to_zero(original)
        self.assert_flags(result, True, True, True)
        self.assertIs(result.sympy_set.contains(0), sp.true)


if __name__ == "__main__":
    unittest.main()
