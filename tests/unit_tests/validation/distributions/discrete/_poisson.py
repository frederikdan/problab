import unittest

from problab.validation.distributions.discrete._poisson import _validate_poisson_mu


class PoissonDistributionValidationTests(unittest.TestCase):

    def test_mu_accepts_non_negative_real_and_rejects_invalid_values(self):
        _validate_poisson_mu(0.0)
        _validate_poisson_mu(1.5)

        with self.assertRaises(ValueError):
            _validate_poisson_mu(-0.1)
        with self.assertRaises(TypeError):
            _validate_poisson_mu("rate")


if __name__ == "__main__":
    unittest.main()
