import math
import unittest

import numpy as np

from problab._config import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE
from problab.validation.probability._results import (
    _validate_non_negative_integer,
    _validate_probability_result_configuration,
)


class ProbabilityResultValidationTests(unittest.TestCase):

    def test_non_negative_integer_accepts_numpy_integer(self):
        _validate_non_negative_integer(np.int64(0), "count")

        with self.assertRaises(TypeError):
            _validate_non_negative_integer(True, "count")
        with self.assertRaises(ValueError):
            _validate_non_negative_integer(-1, "count")

    def test_configuration_accepts_unconditional_and_zero_conditioned_results(self):
        _validate_probability_result_configuration(0.5, 5, 10, None)
        _validate_probability_result_configuration(math.nan, 0, 10, 0)

    def test_configuration_rejects_invalid_counts_and_probability(self):
        with self.assertRaises(ValueError):
            _validate_probability_result_configuration(0.5, 1, 0, None)
        with self.assertRaises(ValueError):
            _validate_probability_result_configuration(0.5, 4, 3, None)
        with self.assertRaises(ValueError):
            _validate_probability_result_configuration(0.5, 1, 3, 4)
        with self.assertRaises(ValueError):
            _validate_probability_result_configuration(1.1, 1, 3, None)
        with self.assertRaises(ValueError):
            _validate_probability_result_configuration(0.5, 0, 3, 0)

    def test_configuration_uses_tolerance_for_probability_consistency(self):
        expected_value = 1 / 3

        _validate_probability_result_configuration(
            expected_value + RELATIVE_TOLERANCE * expected_value / 2,
            1,
            3,
            None,
        )

        with self.assertRaises(ValueError):
            _validate_probability_result_configuration(
                expected_value + 2 * RELATIVE_TOLERANCE * expected_value + ABSOLUTE_TOLERANCE,
                1,
                3,
                None,
            )


if __name__ == "__main__":
    unittest.main()
