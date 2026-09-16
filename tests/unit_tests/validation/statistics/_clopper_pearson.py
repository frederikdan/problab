import unittest

import numpy as np

from problab.validation.statistics._clopper_pearson import (
    _validate_confidence_interval_configuration,
    _validate_confidence_interval_num_samples,
    _validate_num_successes,
)


class ClopperPearsonValidationTests(unittest.TestCase):

    def test_count_validators_accept_non_negative_integers(self):
        _validate_confidence_interval_num_samples(np.int64(0))
        _validate_num_successes(0)

    def test_count_validators_reject_booleans_and_negative_values(self):
        for validator in (_validate_confidence_interval_num_samples, _validate_num_successes):
            with self.subTest(validator=validator.__name__):
                with self.assertRaises(TypeError):
                    validator(True)
                with self.assertRaises(ValueError):
                    validator(-1)

    def test_configuration_rejects_successes_above_samples(self):
        _validate_confidence_interval_configuration(2, 2)

        with self.assertRaises(ValueError):
            _validate_confidence_interval_configuration(2, 3)


if __name__ == "__main__":
    unittest.main()
