import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import binom

from problab.statistics._quantiles import _quantile_confidence_interval


class QuantileSearchCostRegressionTests(unittest.TestCase):

    def test_binomial_tail_search_uses_sublinear_number_of_calls(self):
        samples = np.arange(1000)

        with patch("problab.statistics._quantiles.binom.cdf", wraps=binom.cdf) as cdf:
            with patch("problab.statistics._quantiles.binom.sf", wraps=binom.sf) as sf:
                interval = _quantile_confidence_interval(samples, q=0.5, alpha=0.05)

        self.assertLessEqual(interval.lower, interval.upper)
        self.assertLessEqual(cdf.call_count + sf.call_count, 32)


if __name__ == "__main__":
    unittest.main()
