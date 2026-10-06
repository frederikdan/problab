import unittest

import numpy as np

from problab.validation.distributions.continuous._normal import (
    _validate_normal_mean_realizations, _validate_normal_std_realizations,
)
from problab.validation.distributions.discrete._binomial import (
    _validate_binomial_n_realizations, _validate_binomial_p_realizations,
)
from problab.validation.distributions.discrete._poisson import _validate_poisson_mu_realizations
from problab.value_sets.sets import NATURALS_0, NON_NEGATIVE_REALS, POSITIVE_REALS, REALS, UNIT_INTERVAL


class ParameterMaskPropertyTests(unittest.TestCase):
    def test_runtime_masks_agree_with_value_sets_away_from_backend_limits(self):
        rng = np.random.default_rng(918)
        generated = np.concatenate((
            rng.uniform(-3, 3, 40), rng.integers(-3, 4, 20),
            [0., -0., 1., np.nan, np.inf, -np.inf],
        ))
        cases = (
            (_validate_normal_mean_realizations, REALS),
            (_validate_normal_std_realizations, POSITIVE_REALS),
            (_validate_binomial_n_realizations, NATURALS_0),
            (_validate_binomial_p_realizations, UNIT_INTERVAL),
            (_validate_poisson_mu_realizations, NON_NEGATIVE_REALS),
        )
        for dtype in (np.float16, np.float32, np.float64):
            values = generated.astype(dtype)
            for validator, valid_set in cases:
                with self.subTest(dtype=dtype, validator=validator.__name__):
                    expected = valid_set.contains(values)
                    original = values.copy()
                    with np.errstate(all="raise"):
                        actual = validator(values)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(values, original)
                    self.assertEqual(actual.dtype, np.dtype(bool))
                    self.assertEqual(actual.shape, values.shape)
