import unittest

import numpy as np

from problab._events import _Event
from problab.random_variables.nodes import _OperationNode
from problab.validation.probability._probability import _validate_event, _validate_given
from problab.value_sets.sets import BOOLEANS


def _event():
    return _Event(
        _OperationNode(
            operation=lambda: np.array([], dtype=bool),
            inputs=(),
            name="event",
            value_set=BOOLEANS,
        )
    )


class ProbabilityEventValidationTests(unittest.TestCase):

    def test_event_and_given_accept_event_and_none(self):
        event = _event()

        _validate_event(event)
        _validate_given(event)
        _validate_given(None)

    def test_event_and_given_reject_other_values(self):
        with self.assertRaises(TypeError):
            _validate_event("event")
        with self.assertRaises(TypeError):
            _validate_given("event")


if __name__ == "__main__":
    unittest.main()
