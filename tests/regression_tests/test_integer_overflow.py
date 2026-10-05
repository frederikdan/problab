import unittest
import warnings

import numpy as np

from problab import CategoricalDistribution, RandomVariable


class IntegerOverflowRegressionTests(unittest.TestCase):

    def test_fixed_width_arithmetic_raises_instead_of_silently_wrapping(self):
        variable = RandomVariable(CategoricalDistribution([np.int8(100)], [1.0]))
        for result in (variable + variable, variable * variable):
            with self.subTest(operation=result.name), self.assertRaises(OverflowError):
                result.sample(num_samples=4, numerical_error_policy="raise", validate=True)

    def test_warn_and_ignore_replace_overflow_with_infinity_and_pass_validation(self):
        variable = RandomVariable(CategoricalDistribution([np.int8(100)], [1.0]))
        for policy in ("warn", "ignore"):
            for result in (variable + variable, variable * variable):
                with self.subTest(policy=policy, operation=result.name):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        samples = result.sample(num_samples=4, numerical_error_policy=policy, validate=True)
                    self.assertEqual(samples.dtype, np.dtype(np.float64))
                    np.testing.assert_array_equal(samples, [np.inf] * 4)
                    self.assertEqual(len(caught), 1 if policy == "warn" else 0)
                    if caught:
                        self.assertIs(caught[0].category, RuntimeWarning)

    def test_bounded_integer_arithmetic_preserves_integer_dtype(self):
        variable = RandomVariable(CategoricalDistribution([np.int8(5)], [1.0]))
        for result, expected in ((variable + variable, 10), (variable * variable, 25)):
            with self.subTest(operation=result.name):
                samples = result.sample(num_samples=3, numerical_error_policy="raise", validate=True)
                self.assertEqual(samples.dtype, np.dtype(np.int8))
                np.testing.assert_array_equal(samples, [expected] * 3)


if __name__ == "__main__":
    unittest.main()
