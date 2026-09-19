import unittest

import numpy as np

from problab import Mode, NormalDistribution


class CdfNanRegressionTests(unittest.TestCase):

    def test_scalar_nan_is_rejected_by_cdf(self):
        distribution = NormalDistribution(0, 1)

        with self.assertRaises(ValueError):
            distribution.cdf(
                float("nan"),
                mode=Mode.MONTE_CARLO,
                num_samples=8,
                rng=np.random.default_rng(1),
            )

    def test_array_nan_is_rejected_by_cdf(self):
        distribution = NormalDistribution(0, 1)

        with self.assertRaises(ValueError):
            distribution.cdf(
                np.array([0.0, np.nan]),
                mode=Mode.MONTE_CARLO,
                num_samples=8,
                rng=np.random.default_rng(1),
            )


if __name__ == "__main__":
    unittest.main()
