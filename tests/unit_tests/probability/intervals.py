import math
import unittest

from problab.probability.intervals import ConfidenceInterval, ProbabilityInterval


class ConfidenceIntervalTests(unittest.TestCase):

    def test_constructor_preserves_configuration(self):
        interval = ConfidenceInterval(lower=0.2, upper=0.4, alpha=0.05)

        self.assertEqual(interval.lower, 0.2)
        self.assertEqual(interval.upper, 0.4)
        self.assertEqual(interval.alpha, 0.05)

    def test_constructor_allows_two_nan_bounds_for_an_undefined_interval(self):
        interval = ConfidenceInterval(
            lower=math.nan,
            upper=math.nan,
            alpha=0.05,
        )

        self.assertTrue(math.isnan(interval.lower))
        self.assertTrue(math.isnan(interval.upper))

    def test_constructor_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ConfidenceInterval(lower=0.4, upper=0.2, alpha=0.05)
        with self.assertRaises(ValueError):
            ConfidenceInterval(lower=math.nan, upper=0.2, alpha=0.05)
        with self.assertRaises(ValueError):
            ConfidenceInterval(lower=0.2, upper=0.4, alpha=1.0)


class ProbabilityIntervalTests(unittest.TestCase):

    def test_properties_report_nominal_coverage_and_bounds(self):
        interval = ProbabilityInterval(
            lower=1.0,
            upper=3.0,
            alpha=0.1,
            is_estimate=True,
        )

        self.assertEqual(interval.probability, 0.9)
        self.assertEqual(str(interval), "[1.0, 3.0]")
        self.assertTrue(interval.is_estimate)

    def test_constructor_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ProbabilityInterval(upper=1.0, lower=2.0, alpha=0.05, is_estimate=False)
        with self.assertRaises(ValueError):
            ProbabilityInterval(lower=1.0, upper=2.0, alpha=0.0, is_estimate=False)
        with self.assertRaises(TypeError):
            ProbabilityInterval(lower=1.0, upper=2.0, alpha=0.05, is_estimate=1)


if __name__ == "__main__":
    unittest.main()
