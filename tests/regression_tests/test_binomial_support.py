import unittest

import sympy as sp

from problab import BinomialDistribution


class BinomialSupportRegressionTests(unittest.TestCase):

    def test_finite_binomial_support_uses_symbolic_range(self):
        support = BinomialDistribution(1000, 0.5).value_set.sympy_set

        self.assertFalse(isinstance(support, sp.FiniteSet), "Binomial support expands all 1001 values")
        self.assertTrue(bool(support.contains(0)))
        self.assertTrue(bool(support.contains(1000)))
        self.assertFalse(bool(support.contains(-1)))
        self.assertFalse(bool(support.contains(1001)))


if __name__ == "__main__":
    unittest.main()
