import unittest

from problab.value_sets.base import NumericValueSet, ValueSet
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.object_value_set import ObjectValueSet


class ValueSetBaseTests(unittest.TestCase):

    def test_value_set_remains_abstract(self):
        with self.assertRaises(TypeError):
            ValueSet()

    def test_numeric_value_set_is_value_set_marker_class(self):
        self.assertTrue(issubclass(NumericValueSet, ValueSet))

    def test_dynamic_attributes_load_concrete_value_set_classes(self):
        import problab.value_sets.base as base

        self.assertIs(base.HomogeneousNumericValueSet, HomogeneousNumericValueSet)
        self.assertIs(base.ObjectValueSet, ObjectValueSet)
        with self.assertRaises(AttributeError):
            getattr(base, "MissingValueSet")


if __name__ == "__main__":
    unittest.main()
