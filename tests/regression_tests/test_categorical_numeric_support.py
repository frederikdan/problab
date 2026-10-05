import unittest

import numpy as np
import sympy as sp

from problab import BinomialDistribution, CategoricalDistribution, RandomVariable
from problab.functions import sqrt
from problab.value_sets._utils import is_known_subset
from problab.value_sets.sets import INTEGERS


class CategoricalNumericSupportRegressionTests(unittest.TestCase):
    def test_integer_valued_float_categories_declare_integer_mathematical_support(self):
        for categories in ((2.0,), (1, 2.0)):
            with self.subTest(categories=categories):
                distribution = CategoricalDistribution(categories, [1 / len(categories)] * len(categories))
                self.assertTrue(is_known_subset(distribution.value_set, INTEGERS))

    def test_integer_valued_float_category_can_be_a_binomial_trial_count(self):
        trials = RandomVariable(CategoricalDistribution([2.0], [1.0]))
        variable = RandomVariable(BinomialDistribution(trials, 1.0, parameter_risk_policy="raise"))
        np.testing.assert_array_equal(variable.sample(num_samples=3, validate=True), [2, 2, 2])

    def test_integer_valued_categorical_exponent_keeps_real_square_support(self):
        base = RandomVariable(CategoricalDistribution([-2.0], [1.0]))
        exponent = RandomVariable(CategoricalDistribution([2.0], [1.0]))
        squared = base ** exponent
        self.assertTrue(is_known_subset(squared._node.value_set, sp.S.Reals))
        np.testing.assert_array_equal(sqrt(squared).sample(num_samples=3, validate=True), [2.0] * 3)

    def test_large_integer_object_fallback_preserves_public_samples_and_numeric_support(self):
        large = 2 ** 63 - 1
        distribution = CategoricalDistribution([large, 2 ** 63], [1.0, 0.0])
        samples = distribution.sample(num_samples=3, validate=True)
        self.assertEqual(samples.dtype, np.dtype(object))
        self.assertEqual(samples.tolist(), [large] * 3)
        self.assertTrue(is_known_subset(distribution.value_set, INTEGERS))
