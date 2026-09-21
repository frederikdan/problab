import unittest

from problab._config import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE


class NumericalConfigTests(unittest.TestCase):

    def test_tolerances_are_positive_and_less_than_one(self):
        self.assertGreater(RELATIVE_TOLERANCE, 0)
        self.assertGreater(ABSOLUTE_TOLERANCE, 0)
        self.assertLess(RELATIVE_TOLERANCE, 1)
        self.assertLess(ABSOLUTE_TOLERANCE, 1)


if __name__ == "__main__":
    unittest.main()
