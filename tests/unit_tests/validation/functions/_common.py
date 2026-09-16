import unittest

import sympy as sp

from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes import _ConstantNode
from problab.validation.functions._common import _validate_domain, _validate_real_valued
from problab.value_sets import ObjectValueSet


class FunctionValidationTests(unittest.TestCase):

    def test_real_valued_accepts_real_scalar_and_real_random_variable(self):
        _validate_real_valued(1.0)
        _validate_real_valued(RandomVariable._from_node(_ConstantNode(1)))

    def test_real_valued_rejects_bool_text_and_object_random_variable(self):
        with self.assertRaises(TypeError):
            _validate_real_valued(True)
        with self.assertRaises(TypeError):
            _validate_real_valued("one")
        with self.assertRaises(ValueError):
            _validate_real_valued(
                RandomVariable._from_node(_ConstantNode("red")),
            )

    def test_domain_accepts_member_and_rejects_outside_values(self):
        _validate_domain(1.0, sp.Interval(0, sp.oo))

        with self.assertRaises(ValueError):
            _validate_domain(-1.0, sp.Interval(0, sp.oo))


if __name__ == "__main__":
    unittest.main()
