import enum
import unittest

import numpy as np

from problab.validation._common import (
    _validate_alpha,
    _validate_enum,
    _validate_max_size,
    _validate_num_samples,
    _validate_q,
    _validate_rng,
    _validate_validate,
)


class _Mode(enum.Enum):
    VALUE = "value"


class CommonValidationTests(unittest.TestCase):

    def test_num_samples_accepts_numpy_integer_and_rejects_invalid_values(self):
        _validate_num_samples(np.int64(1))

        with self.assertRaises(TypeError):
            _validate_num_samples(True)
        with self.assertRaises(TypeError):
            _validate_num_samples(1.0)
        with self.assertRaises(ValueError):
            _validate_num_samples(0)

    def test_rng_accepts_generator_or_none(self):
        _validate_rng(None)
        _validate_rng(np.random.default_rng(1))

        with self.assertRaises(TypeError):
            _validate_rng("rng")

    def test_validate_requires_bool(self):
        _validate_validate(True)

        with self.assertRaises(TypeError):
            _validate_validate(1)

    def test_alpha_and_q_require_open_unit_interval(self):
        _validate_alpha(0.5)
        _validate_q(np.float64(0.5))

        for validator in (_validate_alpha, _validate_q):
            with self.subTest(validator=validator.__name__):
                with self.assertRaises(TypeError):
                    validator(True)
                with self.assertRaises(ValueError):
                    validator(0.0)
                with self.assertRaises(ValueError):
                    validator(1.0)

    def test_enum_requires_declared_member(self):
        _validate_enum(_Mode.VALUE, _Mode, "mode")

        with self.assertRaises(TypeError):
            _validate_enum("value", _Mode, "mode")

    def test_max_size_requires_positive_integer(self):
        _validate_max_size(np.int64(1))

        with self.assertRaises(TypeError):
            _validate_max_size(False)
        with self.assertRaises(ValueError):
            _validate_max_size(0)


if __name__ == "__main__":
    unittest.main()
