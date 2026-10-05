import unittest
import numpy as np
import sympy as sp
from problab.operations._arithmetic import _ADD, _DIVIDE, _POWER
from problab.random_variables.nodes import _ConstantNode, _OperationNode
from problab.random_variables._context import _RealizationContext
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet


class PromotedSupportIntegrationTests(unittest.TestCase):
    def test_context_accepts_promoted_outputs(self):
        for operation, left, right, expected in (
            (_DIVIDE, 2, 1, 2.0),
            (_ADD, 2, 0j, 2 + 0j),
            (_POWER, -4, sp.Rational(1, 2), 2j),
            (_POWER, 2, -1, 0.5),
        ):
            # Use a float sample for the fractional exponent, with its exact mathematical set.
            left_node = _ConstantNode(left)
            right_node = _ConstantNode(float(right) if isinstance(right, sp.Rational) else right)
            exponent_set = HomogeneousNumericValueSet(sp.FiniteSet(right), right_node.value_set.dtype_types)
            node = _OperationNode(
                operation=operation,
                inputs=(left_node, right_node),
                name="inference regression",
                mathematical_value_set=operation.infer_mathematical_value_set(left_node.value_set, exponent_set),
                realization_value_set=operation.infer_realization_value_set(left_node._realization_value_set, exponent_set),
            )
            actual = _RealizationContext((node,), num_samples=3).evaluate(node)
            np.testing.assert_allclose(actual, np.full(3, expected), atol=1e-14)

