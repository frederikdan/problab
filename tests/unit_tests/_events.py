import unittest
from unittest.mock import Mock

import numpy as np

from problab._events import _Event
from problab.random_variables.nodes import _ConstantNode
from problab.value_sets.sets import BOOLEANS


class EventTests(unittest.TestCase):

    def test_constructor_exposes_boolean_node_name_and_representation(self):
        event = _Event(_ConstantNode(True))

        self.assertEqual(event.name, "True")
        self.assertEqual(str(event), "True")
        self.assertEqual(repr(event), "Event(ConstantNode(True))")

    def test_constructor_rejects_non_boolean_node(self):
        with self.assertRaises(ValueError):
            _Event(_ConstantNode(1))

    def test_logical_operations_build_boolean_operation_events(self):
        left = _Event(_ConstantNode(True))
        right = _Event(_ConstantNode(False))

        conjunction = left & right
        disjunction = left | right
        inversion = ~left

        self.assertEqual(conjunction.name, "{True & False}")
        self.assertEqual(disjunction.name, "{True | False}")
        self.assertEqual(inversion.name, "{~True}")
        self.assertIs(conjunction._node.value_set, BOOLEANS)

    def test_event_has_no_single_truth_value(self):
        with self.assertRaises(TypeError):
            bool(_Event(_ConstantNode(True)))


if __name__ == "__main__":
    unittest.main()
