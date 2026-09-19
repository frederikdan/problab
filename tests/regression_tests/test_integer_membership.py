import unittest

import numpy as np
import sympy as sp

from problab import CategoricalDistribution, P, RandomVariable
from problab.value_sets import INTEGERS


class IntegerMembershipRegressionTests(unittest.TestCase):

    def test_integer_value_set_contains_integer_valued_float(self):
        self.assertTrue(INTEGERS.contains(-2.0))

    def test_integer_value_set_checks_each_float_in_array(self):
        result = INTEGERS.contains(np.array([-2.0, 0.0, 2.5]))

        np.testing.assert_array_equal(result, [True, True, False])

    def test_integer_valued_float_matches_integer_set_event(self):
        variable = RandomVariable(CategoricalDistribution([-2.0], [1.0]))

        result = P(variable.is_in(sp.S.Integers), num_samples=4, validate=True)

        self.assertEqual(result.value, 1.0)


if __name__ == "__main__":
    unittest.main()
