import math
import unittest

from problab.validation.probability._intervals import (
    _validate_confidence_interval_configuration,
    _validate_interval_bound,
    _validate_probability_interval_configuration,
)


class ProbabilityIntervalValidationTests(unittest.TestCase):

    def test_interval_bound_allows_nan_only_when_requested(self):
        _validate_interval_bound(math.nan, "bound", allow_nan=True)

        with self.assertRaises(ValueError):
            _validate_interval_bound(math.nan, "bound", allow_nan=False)
        with self.assertRaises(TypeError):
            _validate_interval_bound("bound", "bound", allow_nan=False)

    def test_confidence_interval_configuration_accepts_ordered_or_two_nan_bounds(self):
        _validate_confidence_interval_configuration(0.1, 0.2, 0.05)
        _validate_confidence_interval_configuration(math.nan, math.nan, 0.05)

        with self.assertRaises(ValueError):
            _validate_confidence_interval_configuration(math.nan, 0.2, 0.05)
        with self.assertRaises(ValueError):
            _validate_confidence_interval_configuration(0.2, 0.1, 0.05)
        with self.assertRaises(ValueError):
            _validate_confidence_interval_configuration(0.1, 0.2, 1.0)

    def test_probability_interval_configuration_requires_ordered_finite_bounds_and_bool_flag(self):
        _validate_probability_interval_configuration(0.1, 0.2, 0.05, True)

        with self.assertRaises(ValueError):
            _validate_probability_interval_configuration(math.nan, 0.2, 0.05, True)
        with self.assertRaises(ValueError):
            _validate_probability_interval_configuration(0.2, 0.1, 0.05, True)
        with self.assertRaises(TypeError):
            _validate_probability_interval_configuration(0.1, 0.2, 0.05, 1)


if __name__ == "__main__":
    unittest.main()
