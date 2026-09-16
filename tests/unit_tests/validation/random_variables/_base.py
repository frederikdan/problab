import unittest
from unittest.mock import Mock

import sympy as sp

from problab.distributions.base import Distribution
from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes import _ConstantNode
from problab.validation.random_variables._base import (
    _validate_closed,
    _validate_distribution,
    _validate_function,
    _validate_function_name,
    _validate_interval_bound,
    _validate_name,
    _validate_others,
    _validate_target_set,
    _validate_value_set,
    _validate_vectorized,
)
from problab.value_sets.sets import REALS


class RandomVariableValidationTests(unittest.TestCase):

    def test_distribution_and_name_accept_expected_values(self):
        _validate_distribution(Mock(spec=Distribution))
        _validate_name(None)
        _validate_name("X")

        with self.assertRaises(TypeError):
            _validate_distribution("distribution")
        with self.assertRaises(TypeError):
            _validate_name(1)

    def test_interval_bounds_and_closure_are_validated(self):
        _validate_interval_bound(1.0)
        _validate_closed("both")

        with self.assertRaises(TypeError):
            _validate_interval_bound("one")
        with self.assertRaises(TypeError):
            _validate_closed(1)
        with self.assertRaises(ValueError):
            _validate_closed("closed")

    def test_target_set_accepts_sympy_set_or_two_element_sequence(self):
        _validate_target_set(sp.S.Reals)
        _validate_target_set((0, 1))
        _validate_target_set([0, 1])

        with self.assertRaises(TypeError):
            _validate_target_set((0, 1, 2))

    def test_function_value_set_and_name_validators(self):
        _validate_function(lambda value: value)
        _validate_value_set(REALS)
        _validate_function_name("function")
        _validate_vectorized(False)

        with self.assertRaises(TypeError):
            _validate_function("function")
        with self.assertRaises(TypeError):
            _validate_value_set("reals")
        with self.assertRaises(TypeError):
            _validate_function_name(1)
        with self.assertRaises(TypeError):
            _validate_vectorized(1)

    def test_others_requires_random_variables(self):
        variable = RandomVariable._from_node(_ConstantNode(1))

        _validate_others((variable,))

        with self.assertRaises(TypeError):
            _validate_others((1,))


if __name__ == "__main__":
    unittest.main()
