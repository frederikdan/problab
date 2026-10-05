import unittest
from unittest.mock import Mock, patch

import numpy as np

from problab._events import _Event
from problab.random_variables.nodes import _ConstantNode
from problab.value_sets.sets import BOOLEANS


class EventTests(unittest.TestCase):

    def test_logical_operations_check_node_dtypes_before_simplifying(self):
        event = _Event(_ConstantNode(True))
        for function in (lambda: event & event, lambda: event | event, lambda: ~event):
            with self.subTest(operation=function):
                with patch("problab._events._require_supported_operation_inputs", side_effect=TypeError("unsupported")) as require:
                    with self.assertRaisesRegex(TypeError, "unsupported"):
                        function()
                self.assertTrue(require.call_args.args[0])
                self.assertTrue(all(node is event._node for node in require.call_args.args[0]))
                self.assertEqual(require.call_args.kwargs["supported_input_types"], ((np.bool_,),))

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

    def test_repeated_conjunction_reuses_the_event_node(self):
        event = _Event(_ConstantNode(True))

        result = event & event

        self.assertIs(result._node, event._node)

    def test_repeated_disjunction_reuses_the_event_node(self):
        event = _Event(_ConstantNode(True))

        result = event | event

        self.assertIs(result._node, event._node)

    def test_double_inversion_reuses_the_event_node(self):
        event = _Event(_ConstantNode(True))

        result = ~(~event)

        self.assertIs(result._node, event._node)

    def test_event_and_its_inverse_simplifies_to_false(self):
        event = _Event(_ConstantNode(True))

        result = event & ~event

        self.assertIsInstance(result._node, _ConstantNode)
        self.assertIs(result._node.value, False)

    def test_inverse_and_event_simplifies_to_false(self):
        event = _Event(_ConstantNode(True))

        result = ~event & event

        self.assertIsInstance(result._node, _ConstantNode)
        self.assertIs(result._node.value, False)

    def test_event_or_its_inverse_simplifies_to_true(self):
        event = _Event(_ConstantNode(False))

        result = event | ~event

        self.assertIsInstance(result._node, _ConstantNode)
        self.assertIs(result._node.value, True)

    def test_inverse_or_event_simplifies_to_true(self):
        event = _Event(_ConstantNode(False))

        result = ~event | event

        self.assertIsInstance(result._node, _ConstantNode)
        self.assertIs(result._node.value, True)

    def test_logical_operations_evaluate_each_boolean_array(self):
        left = _Event(_ConstantNode(True))
        right = _Event(_ConstantNode(False))
        context = Mock(numerical_error_policy="warn")
        context.evaluate.side_effect = (
            np.array([True, False]),
            np.array([False, False]),
        )

        np.testing.assert_array_equal((left & right)._node._evaluate(context), [False, False])

        context.evaluate.side_effect = (
            np.array([True, False]),
            np.array([False, True]),
        )
        np.testing.assert_array_equal((left | right)._node._evaluate(context), [True, True])

        context.evaluate.side_effect = (np.array([True, False]),)
        np.testing.assert_array_equal((~left)._node._evaluate(context), [False, True])

    def test_logical_operations_reject_non_events(self):
        event = _Event(_ConstantNode(True))

        self.assertIs(event.__and__(True), NotImplemented)
        self.assertIs(event.__or__(False), NotImplemented)

    def test_event_has_no_single_truth_value(self):
        with self.assertRaises(TypeError):
            bool(_Event(_ConstantNode(True)))


if __name__ == "__main__":
    unittest.main()
