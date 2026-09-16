import math
import unittest
from unittest.mock import call, patch

import numpy as np

from problab._events import _Event
from problab.probability.probability import P
from problab.random_variables.nodes import _OperationNode
from problab.value_sets.sets import BOOLEANS


def _event(name: str) -> _Event:
    return _Event(
        _OperationNode(
            operation=lambda: np.array([], dtype=bool),
            inputs=(),
            name=name,
            value_set=BOOLEANS,
        )
    )


class ProbabilityFunctionTests(unittest.TestCase):

    @patch("problab.probability.probability._RealizationContext")
    def test_unconditional_probability_counts_true_values(self, realization_context):
        event = _event("event")
        rng = np.random.default_rng(1)
        realization_context.return_value.evaluate.return_value = np.array(
            [True, False, True, True],
        )

        result = P(event, num_samples=4, rng=rng, validate=True)

        self.assertEqual(result.value, 0.75)
        self.assertEqual(result.num_successes, 3)
        self.assertEqual(result.num_unconditioned_samples, 4)
        self.assertIsNone(result.num_conditioned_samples)
        realization_context.assert_called_once_with(
            root_node=event._node,
            num_samples=4,
            rng=rng,
            validate=True,
        )
        realization_context.return_value.evaluate.assert_called_once_with(event._node)

    @patch("problab.probability.probability._RealizationContext")
    def test_conditional_probability_counts_joint_successes_within_condition(
        self,
        realization_context,
    ):
        event = _event("event")
        given = _event("given")
        rng = np.random.default_rng(2)
        realization_context.return_value.evaluate.side_effect = (
            np.array([True, False, True, True]),
            np.array([True, False, False, True]),
        )

        result = P(event, given=given, num_samples=4, rng=rng)

        self.assertEqual(result.value, 2 / 3)
        self.assertEqual(result.num_successes, 2)
        self.assertEqual(result.num_unconditioned_samples, 4)
        self.assertEqual(result.num_conditioned_samples, 3)
        root_node = realization_context.call_args.kwargs["root_node"]
        self.assertIsInstance(root_node, _OperationNode)
        self.assertEqual(root_node.dependencies, {event._node, given._node})
        realization_context.return_value.evaluate.assert_has_calls(
            [call(given._node), call(root_node)],
        )

    @patch("problab.probability.probability._RealizationContext")
    def test_conditional_probability_returns_nan_when_condition_never_occurs(
        self,
        realization_context,
    ):
        event = _event("event")
        given = _event("given")
        realization_context.return_value.evaluate.return_value = np.zeros(4, dtype=bool)

        result = P(event, given=given, num_samples=4)

        self.assertTrue(math.isnan(result.value))
        self.assertEqual(result.num_successes, 0)
        self.assertEqual(result.num_conditioned_samples, 0)
        realization_context.return_value.evaluate.assert_called_once_with(given._node)

    def test_rejects_invalid_public_arguments(self):
        event = _event("event")

        with self.assertRaises(TypeError):
            P("event")
        with self.assertRaises(TypeError):
            P(event, given="given")
        with self.assertRaises(ValueError):
            P(event, num_samples=0)
        with self.assertRaises(TypeError):
            P(event, validate="yes")


if __name__ == "__main__":
    unittest.main()
