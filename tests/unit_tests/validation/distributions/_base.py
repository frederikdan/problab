import unittest

import numpy as np

from problab.validation.distributions._base import (
    _validate_cdf_input,
    _validate_ppf_input,
    _validate_quantile_method,
    _validate_real_input,
)


class DistributionBaseValidationTests(unittest.TestCase):

    def test_real_input_accepts_real_scalars_and_numeric_non_complex_arrays(self):
        _validate_real_input(1.0, "x")
        _validate_cdf_input(np.array([1, 2], dtype=np.int64))

    def test_real_input_rejects_bool_text_and_complex_values(self):
        with self.assertRaises(TypeError):
            _validate_real_input(True, "x")
        with self.assertRaises(TypeError):
            _validate_real_input(np.array(["x"]), "x")
        with self.assertRaises(TypeError):
            _validate_real_input(np.array([1 + 1j]), "x")

    def test_ppf_input_requires_values_in_closed_unit_interval(self):
        _validate_ppf_input(np.array([0.0, 0.5, 1.0]))

        with self.assertRaises(ValueError):
            _validate_ppf_input(np.array([np.nan]))
        with self.assertRaises(ValueError):
            _validate_ppf_input(-0.1)

    def test_quantile_method_requires_supported_name(self):
        _validate_quantile_method("linear")

        with self.assertRaises(TypeError):
            _validate_quantile_method(1)
        with self.assertRaises(ValueError):
            _validate_quantile_method("unsupported")


if __name__ == "__main__":
    unittest.main()
