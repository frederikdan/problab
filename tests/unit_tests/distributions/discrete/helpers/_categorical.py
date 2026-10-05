import unittest
from fractions import Fraction
from decimal import Decimal

import numpy as np

from problab.distributions.discrete.helpers._categorical import (
    _prepare_mixed_numeric_categories_and_mathematical_value_set,
    _build_untyped_category_configuration,
    _prepare_categories_and_mathematical_value_set,
    _is_numeric_category,
    _merge_equal_categories,
    _attempt_make_categories_numeric,
    _conversion_preserves_values,
)
from problab.value_sets import (
    HomogeneousNumericValueSet,
    MixedNumericValueSet,
    ObjectValueSet,
)


class CategoricalHelperTests(unittest.TestCase):

    def test_lossless_numeric_conversion_accepts_exact_value_preserving_promotions(self):
        for categories, dtype in (((1, 4.0), np.float64), ((Fraction(1, 2),), np.float64),
                                  ((Decimal("0.5"),), np.float64), ((np.float32(0.1),), np.float64),
                                  ((1 + 2j,), np.complex128), ((1, 2 + 3j), np.complex128)):
            with self.subTest(categories=categories, dtype=dtype):
                self.assertTrue(_conversion_preserves_values(categories, np.asarray(categories, dtype=dtype)))

    def test_lossless_numeric_conversion_rejects_rounding_of_integers_fractions_and_float_components(self):
        for categories, converted in (
            ((2 ** 53 + 1,), np.array([2 ** 53 + 1], dtype=np.float64)),
            ((Fraction(1, 3),), np.array([1 / 3])),
            ((Decimal("0.1"),), np.array([0.1])),
            ((1.0 + 1.0000000001j,), np.array([1 + 1j], dtype=np.complex64)),
            ((0.1,), np.array([0.1], dtype=np.float32)),
        ):
            with self.subTest(categories=categories):
                self.assertFalse(_conversion_preserves_values(categories, converted))

    def test_lossless_numeric_conversion_distinguishes_infinity_sign_and_matches_nan(self):
        for original, result, expected in (
            (np.inf, np.inf, True), (-np.inf, -np.inf, True), (np.inf, -np.inf, False),
            (np.nan, np.nan, True), (np.nan, 0.0, False), (1.0, np.inf, False),
            (complex(np.inf, np.nan), complex(np.inf, np.nan), True),
            (complex(np.inf, np.nan), complex(-np.inf, np.nan), False),
        ):
            with self.subTest(original=original, result=result):
                self.assertIs(_conversion_preserves_values((original,), np.asarray([result])), expected)

    def test_lossless_numeric_conversion_checks_shape_and_rejects_nonnumeric_objects(self):
        for array in (np.array(1), np.array([[1]]), np.array([1, 2])):
            with self.subTest(shape=array.shape):
                self.assertFalse(_conversion_preserves_values((1,), array))
        self.assertFalse(_conversion_preserves_values((object(),), np.array([1])))

    def test_attempt_numeric_conversion_returns_readonly_native_array_for_safe_mixture(self):
        categories = (1, np.float32(0.5), 2 + 3j)
        values = _attempt_make_categories_numeric(categories, MixedNumericValueSet(categories))
        self.assertEqual(values.dtype, np.dtype(np.complex128))
        np.testing.assert_array_equal(values, [1, 0.5, 2 + 3j])
        self.assertFalse(values.flags.writeable)

    def test_attempt_numeric_conversion_retains_objects_when_conversion_is_lossy_or_unsupported(self):
        for categories in ((2 ** 63 - 1, 2 ** 63), (1, Fraction(1, 3)), (Decimal("0.1"),)):
            with self.subTest(categories=categories):
                self.assertIsNone(_attempt_make_categories_numeric(categories, MixedNumericValueSet(categories)))
        self.assertIsNone(_attempt_make_categories_numeric(("red",), ObjectValueSet(("red",))))

    def test_homogeneous_large_integer_categories_fall_back_without_precision_loss(self):
        categories = (2 ** 63 - 1, 2 ** 63)
        values, support = _prepare_categories_and_mathematical_value_set(categories)
        self.assertIsInstance(support, MixedNumericValueSet)
        self.assertEqual(values.dtype, np.dtype(object))
        self.assertEqual(tuple(values), categories)

    def test_homogeneous_numpy_categories_preserve_original_precision(self):
        values, support = _prepare_categories_and_mathematical_value_set((np.float32(0.5), np.float32(1.5)))
        self.assertEqual(values.dtype, np.dtype(np.float32))
        self.assertEqual(support.dtype_types, (np.float32,))
        self.assertFalse(values.flags.writeable)
    def test_numeric_category_detection(self):
        self.assertTrue(_is_numeric_category(1))
        self.assertTrue(_is_numeric_category(np.float64(1.0)))
        self.assertTrue(_is_numeric_category(Fraction(1, 2)))
        self.assertFalse(_is_numeric_category("1"))
        self.assertFalse(_is_numeric_category((1, 2)))

    def test_untyped_categories_remain_atomic_objects(self):
        categories = ([1, 2], {"status": "done"})

        values, value_set = _build_untyped_category_configuration(categories)

        self.assertIsInstance(value_set, ObjectValueSet)
        self.assertEqual(values.dtype, object)
        self.assertIs(values[0], categories[0])
        self.assertIs(values[1], categories[1])
        self.assertFalse(values.flags.writeable)

    def test_mixed_numeric_categories_remain_atomic_objects(self):
        categories = (1, 4.0)

        values, value_set = _prepare_mixed_numeric_categories_and_mathematical_value_set(categories)

        self.assertIsInstance(value_set, MixedNumericValueSet)
        self.assertEqual(values.dtype, object)
        self.assertEqual(type(values[0]), int)
        self.assertEqual(type(values[1]), float)
        self.assertFalse(values.flags.writeable)

    def test_inference_selects_homogeneous_mixed_or_object_value_sets(self):
        _, homogeneous = _prepare_categories_and_mathematical_value_set((1, 2, 3))
        _, mixed = _prepare_categories_and_mathematical_value_set((1, 4.0))
        _, objects = _prepare_categories_and_mathematical_value_set(("red", "blue"))

        self.assertIsInstance(homogeneous, HomogeneousNumericValueSet)
        self.assertIsInstance(mixed, MixedNumericValueSet)
        self.assertIsInstance(objects, ObjectValueSet)

    def test_equal_categories_are_merged_in_order(self):
        categories, probabilities = _merge_equal_categories(
            ("red", "red", "blue"),
            (0.2, 0.3, 0.5),
        )

        self.assertEqual(categories, ("red", "blue"))
        self.assertEqual(probabilities, (0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
