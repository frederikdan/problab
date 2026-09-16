import unittest

import numpy as np

from problab.value_sets.object_value_set import ObjectValueSet


class ObjectValueSetTests(unittest.TestCase):

    def test_contains_preserves_object_identity_and_array_shape(self):
        category = ["large", "blue"]
        value_set = ObjectValueSet(("small", category))
        scalar_category = np.empty((), dtype=object)
        scalar_category[()] = category
        values = np.empty(2, dtype=object)
        values[:] = [category, "other"]

        self.assertTrue(value_set.contains(scalar_category))
        self.assertFalse(value_set.contains("other"))
        np.testing.assert_array_equal(value_set.contains(values), [True, False])
        self.assertEqual(value_set.dtype_types, (np.object_,))

    def test_constructor_requires_non_empty_tuple(self):
        with self.assertRaises(TypeError):
            ObjectValueSet(["small"])
        with self.assertRaises(ValueError):
            ObjectValueSet(())


if __name__ == "__main__":
    unittest.main()
