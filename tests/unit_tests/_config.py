import unittest

from problab._config import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
    DEF_NUMERICAL_ERROR_POLICY,
    DEF_PARAMETER_RISK_POLICY,
)


class NumericalConfigTests(unittest.TestCase):

    def test_default_numerical_and_parameter_policies_warn(self):
        self.assertEqual(DEF_NUMERICAL_ERROR_POLICY, "warn")
        self.assertEqual(DEF_PARAMETER_RISK_POLICY, "warn")

    def test_tolerances_are_positive_and_less_than_one(self):
        self.assertGreater(RELATIVE_TOLERANCE, 0)
        self.assertGreater(ABSOLUTE_TOLERANCE, 0)
        self.assertLess(RELATIVE_TOLERANCE, 1)
        self.assertLess(ABSOLUTE_TOLERANCE, 1)


if __name__ == "__main__":
    unittest.main()
