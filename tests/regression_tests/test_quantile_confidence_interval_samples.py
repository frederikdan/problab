import unittest

from problab import CategoricalDistribution, RandomVariable


class QuantileConfidenceIntervalSampleRegressionTests(unittest.TestCase):

    def test_insufficient_samples_are_rejected_before_constructing_an_interval(self):
        variable = RandomVariable(CategoricalDistribution([1], [1.0]))

        with self.assertRaisesRegex(ValueError, "Insufficient samples"):
            variable.quantile_confidence_interval(
                q=0.5,
                alpha=0.05,
                num_samples=1,
            )


if __name__ == "__main__":
    unittest.main()
