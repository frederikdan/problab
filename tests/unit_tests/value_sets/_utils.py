import unittest

import numpy as np
import sympy as sp

from problab.value_sets._utils import is_known_subset, validate_as_subset
from problab.value_sets.object_value_set import ObjectValueSet
from problab.value_sets.sets import INTEGERS, REALS, UNKNOWN_VALUE_SET


class ValueSetUtilityTests(unittest.TestCase):

    def test_is_known_subset_handles_numeric_object_and_unknown_sets(self):
        self.assertTrue(is_known_subset(INTEGERS, REALS))
        self.assertTrue(is_known_subset(sp.S.Integers, sp.S.Reals))
        self.assertFalse(is_known_subset(ObjectValueSet(("red",)), REALS))
        self.assertFalse(is_known_subset(UNKNOWN_VALUE_SET, REALS))

    def test_validate_as_subset_accepts_integer_valued_floats_for_integer_set(self):
        validate_as_subset(np.array([-2.0, 0.0, 3.0]), INTEGERS)

    def test_validate_as_subset_rejects_outside_or_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "index 1"):
            validate_as_subset(np.array([1, 1.5]), INTEGERS)
        with self.assertRaises(ValueError):
            validate_as_subset(np.array([1]), UNKNOWN_VALUE_SET)

    def test_validate_as_subset_uses_object_membership(self):
        value_set = ObjectValueSet(("red", "blue"))

        validate_as_subset(np.array(["red", "blue"], dtype=object), value_set)
        with self.assertRaisesRegex(ValueError, "index 1"):
            validate_as_subset(np.array(["red", "green"], dtype=object), value_set)


if __name__ == "__main__":
    unittest.main()
