import unittest
import warnings

import numpy as np

from problab import CategoricalDistribution, RandomVariable


class ArithmeticExceptionalSupportTests(unittest.TestCase):
    def variable(self, value):
        return RandomVariable(CategoricalDistribution((value,), (1.0,)))

    def expressions(self):
        left = self.variable(np.int8(120))
        first, second = left + left, left + left
        return (
            ("addition", first, np.inf),
            ("subtraction", self.variable(np.int8(-120)) - left, -np.inf),
            ("multiplication", self.variable(np.int8(100)) * self.variable(np.int8(2)), np.inf),
            ("division", self.variable(1e300) / self.variable(1e-300), np.inf),
            ("modulo", self.variable(1.) % self.variable(0.), np.nan),
            ("power", self.variable(np.int8(12)) ** np.int8(2), np.inf),
            ("rounded_exponent", self.variable(-2.) ** np.int64(2**53 + 1), np.inf),
            ("negation", -self.variable(np.int8(-128)), np.inf),
            ("absolute", abs(self.variable(np.int8(-128))), np.inf),
            ("propagated_nan", first - second, np.nan),
            ("zero_times_infinity", self.variable(0.) * first, np.nan),
        )

    def test_warn_and_ignore_allow_declared_exceptional_samples_with_validation(self):
        for name, expression, expected in self.expressions():
            for policy in ("warn", "ignore"):
                with self.subTest(operation=name, policy=policy):
                    with warnings.catch_warnings(record=True) as recorded:
                        warnings.simplefilter("always")
                        samples = expression.sample(
                            num_samples=3, validate=True, numerical_error_policy=policy,
                        )
                    np.testing.assert_array_equal(samples, np.full(3, expected))
                    self.assertEqual(bool(recorded), policy == "warn")

    def test_raise_stops_exceptional_arithmetic_before_output_validation(self):
        for name, expression, _ in self.expressions():
            with self.subTest(operation=name):
                with self.assertRaises((OverflowError, FloatingPointError)):
                    expression.sample(
                        num_samples=3, validate=True, numerical_error_policy="raise",
                    )


if __name__ == "__main__":
    unittest.main()
