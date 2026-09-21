import math
import unittest

from problab._config import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE
from problab._utils import _is_close


class NumericalUtilityTests(unittest.TestCase):

    def test_is_close_accepts_exact_and_within_tolerance_values(self):
        self.assertTrue(_is_close(1.0, 1.0))
        self.assertTrue(_is_close(0.0, ABSOLUTE_TOLERANCE / 2))
        self.assertTrue(_is_close(1.0, 1.0 + RELATIVE_TOLERANCE / 2))

    def test_is_close_rejects_nan_and_values_outside_tolerance(self):
        self.assertFalse(_is_close(math.nan, 1.0))
        self.assertFalse(
            _is_close(
                1.0,
                1.0 + 2 * RELATIVE_TOLERANCE + ABSOLUTE_TOLERANCE,
            )
        )


if __name__ == "__main__":
    unittest.main()
