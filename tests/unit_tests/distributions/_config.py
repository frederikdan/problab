import unittest

from problab.distributions._config import (
    DEF_ALPHA,
    DEF_DISTRIBUTION_SYMBOL_NAME,
    DEF_NUM_SAMPLES,
)


class DistributionConfigTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(DEF_NUM_SAMPLES, 100_000)
        self.assertEqual(DEF_ALPHA, 0.05)
        self.assertEqual(DEF_DISTRIBUTION_SYMBOL_NAME, "unnamed_distribution")


if __name__ == "__main__":
    unittest.main()
