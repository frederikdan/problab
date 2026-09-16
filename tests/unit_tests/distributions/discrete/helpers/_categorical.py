import unittest
from fractions import Fraction

import numpy as np

from problab.distributions.discrete.helpers._categorical import (
    _build_mixed_numeric_category_configuration,
    _build_untyped_category_configuration,
    _infer_categorical_configuration,
    _is_numeric_category,
    _merge_equal_categories,
)
from problab.value_sets import (
    HomogeneousNumericValueSet,
    MixedNumericValueSet,
    ObjectValueSet,
)


class CategoricalHelperTests(unittest.TestCase):
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

        values, value_set = _build_mixed_numeric_category_configuration(categories)

        self.assertIsInstance(value_set, MixedNumericValueSet)
        self.assertEqual(values.dtype, object)
        self.assertEqual(type(values[0]), int)
        self.assertEqual(type(values[1]), float)
        self.assertFalse(values.flags.writeable)

    def test_inference_selects_homogeneous_mixed_or_object_value_sets(self):
        _, homogeneous = _infer_categorical_configuration((1, 2, 3))
        _, mixed = _infer_categorical_configuration((1, 4.0))
        _, objects = _infer_categorical_configuration(("red", "blue"))

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
