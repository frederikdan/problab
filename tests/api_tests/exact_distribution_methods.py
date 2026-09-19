import math
import unittest

from problab import (
    BinomialDistribution,
    CategoricalDistribution,
    Mode,
    NormalDistribution,
    PoissonDistribution,
)


class ExactDistributionMethodTests(unittest.TestCase):

    def setUp(self):
        self.cases = (
            ("normal", NormalDistribution(1.0, 2.0), 1.0, 4.0, 0.5, 1.0),
            ("binomial", BinomialDistribution(4, 0.25), 1.0, 0.75, 0.75 ** 4, 1.0),
            ("poisson", PoissonDistribution(2.0), 2.0, 2.0, math.exp(-2.0), 2.0),
            ("numeric categorical", CategoricalDistribution([1, 3], [0.25, 0.75]), 2.5, 0.75, 0.25, 3.0),
        )

    def test_exact_mean_matches_analytical_mean(self):
        for name, distribution, expected, _, _, _ in self.cases:
            with self.subTest(distribution=name):
                self.assertAlmostEqual(distribution.mean(mode=Mode.EXACT), expected)

    def test_exact_variance_matches_analytical_variance(self):
        for name, distribution, _, expected, _, _ in self.cases:
            with self.subTest(distribution=name):
                self.assertAlmostEqual(distribution.variance(mode=Mode.EXACT), expected)

    def test_exact_standard_deviation_matches_square_root_of_variance(self):
        for name, distribution, _, variance, _, _ in self.cases:
            with self.subTest(distribution=name):
                self.assertAlmostEqual(distribution.std(mode=Mode.EXACT), math.sqrt(variance))

    def test_exact_cdf_matches_analytical_probability(self):
        thresholds = (1.0, 0.0, 0.0, 2.0)
        for (name, distribution, _, _, expected, _), threshold in zip(self.cases, thresholds):
            with self.subTest(distribution=name):
                self.assertAlmostEqual(distribution.cdf(threshold, mode=Mode.EXACT), expected)

    def test_exact_ppf_matches_analytical_median(self):
        for name, distribution, _, _, _, expected in self.cases:
            with self.subTest(distribution=name):
                self.assertAlmostEqual(distribution.ppf(0.5, mode=Mode.EXACT), expected)


if __name__ == "__main__":
    unittest.main()
