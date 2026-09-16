import unittest

from problab.validation.distributions.discrete._binomial import (
    _validate_binomial_n,
    _validate_binomial_p,
)


class BinomialDistributionValidationTests(unittest.TestCase):

    def test_n_accepts_natural_zero_and_rejects_invalid_values(self):
        _validate_binomial_n(0)
        _validate_binomial_n(2)

        with self.assertRaises(ValueError):
            _validate_binomial_n(-1)
        with self.assertRaises(TypeError):
            _validate_binomial_n(2.0)

    def test_p_accepts_closed_unit_interval_and_rejects_invalid_values(self):
        _validate_binomial_p(0.0)
        _validate_binomial_p(1.0)

        with self.assertRaises(ValueError):
            _validate_binomial_p(-0.1)
        with self.assertRaises(ValueError):
            _validate_binomial_p(1.1)
        with self.assertRaises(TypeError):
            _validate_binomial_p("half")


if __name__ == "__main__":
    unittest.main()
