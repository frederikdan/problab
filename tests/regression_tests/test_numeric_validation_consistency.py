import unittest

import numpy as np

from problab import BinomialDistribution, CategoricalDistribution, NormalDistribution, PoissonDistribution


class NumericValidationConsistencyRegressionTests(unittest.TestCase):

    def test_numpy_integer_is_accepted_as_binomial_trial_count(self):
        distribution = BinomialDistribution(np.int64(3), 0.5)

        self.assertEqual(distribution.parameters[0], np.int64(3))

    def test_numpy_float_is_accepted_as_distribution_parameter(self):
        normal = NormalDistribution(np.float32(1.5), np.float32(2.0))
        poisson = PoissonDistribution(np.float32(2.0))

        self.assertEqual(normal.parameters, (np.float32(1.5), np.float32(2.0)))
        self.assertEqual(poisson.parameters, (np.float32(2.0),))

    def test_boolean_is_rejected_as_binomial_trial_count(self):
        with self.assertRaises(TypeError):
            BinomialDistribution(True, 0.5)

    def test_boolean_is_rejected_as_probability_or_rate(self):
        for name, constructor in (
            ("binomial p", lambda: BinomialDistribution(3, True)),
            ("poisson rate", lambda: PoissonDistribution(True)),
            ("categorical probabilities", lambda: CategoricalDistribution(["a", "b"], [True, False])),
        ):
            with self.subTest(parameter=name):
                with self.assertRaises(TypeError):
                    constructor()


if __name__ == "__main__":
    unittest.main()
