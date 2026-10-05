import unittest
import numpy as np
from problab.operations._statistical import _MEAN, _VARIANCE, _STD


class StatisticalOperationTests(unittest.TestCase):
    def test_descriptors_use_population_statistics_and_numeric_dtype_rules(self):
        for operation, name, expected in ((_MEAN, "mean", 2.0), (_VARIANCE, "variance", 1.0), (_STD, "std", 1.0)):
            with self.subTest(operation=name):
                self.assertEqual(operation.operation(np.array([1.0, 3.0])), expected)
                self.assertEqual(operation.name_func("X"), f"{name}(X)")
                self.assertEqual(operation.supported_input_types, ((np.number,),))

    def test_complex_mean_is_complex_while_variance_and_std_are_real(self):
        values = np.array([1 + 1j, 3 + 3j])
        self.assertEqual(_MEAN.operation(values), 2 + 2j)
        self.assertAlmostEqual(_VARIANCE.operation(values), 2.0)
        self.assertAlmostEqual(_STD.operation(values), 2 ** 0.5)
