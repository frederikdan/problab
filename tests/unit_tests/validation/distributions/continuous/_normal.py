import unittest

from problab.validation.distributions.continuous._normal import (
    _validate_normal_mean,
    _validate_normal_std,
)


class NormalDistributionValidationTests(unittest.TestCase):

    def test_mean_accepts_real_scalar_and_rejects_invalid_values(self):
        _validate_normal_mean(0.0)

        with self.assertRaises(TypeError):
            _validate_normal_mean("zero")
        with self.assertRaises(TypeError):
            _validate_normal_mean(1j)

    def test_std_requires_positive_real_scalar(self):
        _validate_normal_std(1.0)

        with self.assertRaises(ValueError):
            _validate_normal_std(0.0)
        with self.assertRaises(ValueError):
            _validate_normal_std(-1.0)
        with self.assertRaises(TypeError):
            _validate_normal_std("one")


if __name__ == "__main__":
    unittest.main()
