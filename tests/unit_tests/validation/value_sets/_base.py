import unittest

import numpy as np
import sympy as sp

from problab.validation.value_sets._base import _validate_value_set_configuration
from problab.value_sets._unknown import _UnknownValueSet


class ValueSetConfigurationValidationTests(unittest.TestCase):

    def test_accepts_sympy_or_unknown_set_and_numpy_dtype_types(self):
        _validate_value_set_configuration(sp.S.Reals, (np.integer, np.floating))
        _validate_value_set_configuration(_UnknownValueSet(), None)

    def test_rejects_invalid_sympy_set(self):
        with self.assertRaises(TypeError):
            _validate_value_set_configuration("reals", None)

    def test_rejects_invalid_dtype_configuration(self):
        with self.assertRaises(TypeError):
            _validate_value_set_configuration(sp.S.Reals, [np.integer])
        with self.assertRaises(ValueError):
            _validate_value_set_configuration(sp.S.Reals, ())
        with self.assertRaises(TypeError):
            _validate_value_set_configuration(sp.S.Reals, (int,))


if __name__ == "__main__":
    unittest.main()
