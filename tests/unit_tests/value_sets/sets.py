import unittest

from problab.value_sets.sets import (
    BOOLEANS,
    COMPLEXES,
    INTEGERS,
    NATURALS,
    POSITIVE_REALS,
    REALS,
    UNIT_INTERVAL,
    UNKNOWN_VALUE_SET,
)
from problab.value_sets._unknown import _UnknownValueSet


class PredefinedValueSetTests(unittest.TestCase):

    def test_core_numeric_sets_describe_expected_membership(self):
        self.assertTrue(BOOLEANS.contains(True))
        self.assertFalse(BOOLEANS.contains(2))
        self.assertTrue(INTEGERS.contains(2))
        self.assertTrue(REALS.contains(-1.5))
        self.assertTrue(POSITIVE_REALS.contains(0.1))
        self.assertFalse(POSITIVE_REALS.contains(0))
        self.assertTrue(NATURALS.contains(1))
        self.assertTrue(UNIT_INTERVAL.contains(1))
        self.assertTrue(COMPLEXES.contains(1j))

    def test_unknown_value_set_uses_unknown_symbolic_support(self):
        self.assertIsInstance(UNKNOWN_VALUE_SET.sympy_set, _UnknownValueSet)
        self.assertIsNone(UNKNOWN_VALUE_SET.dtype_types)


if __name__ == "__main__":
    unittest.main()
