import unittest
from unittest.mock import Mock, call, patch, sentinel

import numpy as np
import sympy as sp

from problab._operations import (
    _ABS,
    _ADD,
    _DIVIDE,
    _EQ,
    _GTE,
    _GT,
    _LTE,
    _LT,
    _MODULO,
    _MULTIPLY,
    _NEGATIVE,
    _NEQ,
    _POWER,
    _SUBTRACT,
)
from problab.distributions.base import Distribution
from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes import _ConstantNode, _Node, _OperationNode
from problab.value_sets import ObjectValueSet
from problab.value_sets.sets import COMPLEXES, REALS


class _StubNode(_Node):

    def __init__(self, name="X", value_set=REALS):
        super().__init__()
        self._name = name
        self._extended_name = name
        self._value_set = value_set

    def __repr__(self):
        return f"StubNode({self.name})"

    @property
    def value_set(self):
        return self._value_set

    @property
    def dependencies(self):
        return set()

    def _evaluate(self, context):
        return context.values[self]


def _random_variable(name="X", value_set=REALS):
    return RandomVariable._from_node(_StubNode(name, value_set), name=name)


class RandomVariableBaseTests(unittest.TestCase):

    @patch("problab.random_variables.base._DistributionNode")
    def test_constructor_creates_distribution_node_with_explicit_name(self, distribution_node):
        distribution = Mock(spec=Distribution)
        distribution_node.return_value = sentinel.node

        variable = RandomVariable(distribution, name="X")

        self.assertIs(variable._distribution, distribution)
        self.assertIs(variable._node, sentinel.node)
        self.assertEqual(variable.name, "X")
        distribution_node.assert_called_once_with(distribution, rv_name="X")

    def test_from_node_preserves_node_and_explicit_name(self):
        node = _StubNode("source")

        variable = RandomVariable._from_node(node, name="X")

        self.assertIs(variable._node, node)
        self.assertEqual(variable.name, "X")
        self.assertEqual(str(variable), "X")
        self.assertIn("name=X", repr(variable))

    def test_generated_names_are_distinct_when_not_supplied(self):
        first = RandomVariable._from_node(_StubNode("source"))
        second = RandomVariable._from_node(_StubNode("source"))

        self.assertRegex(first.name, r"RV_\d+")
        self.assertRegex(second.name, r"RV_\d+")
        self.assertNotEqual(first.name, second.name)

    @patch("problab.random_variables.base.NodeGraph")
    def test_dependency_graph_uses_default_graph_limit(self, node_graph):
        variable = _random_variable()

        graph = variable.dependency_graph

        self.assertIs(graph, node_graph.return_value)
        node_graph.assert_called_once_with(variable._node, max_size=100)

    @patch("problab.random_variables.base.NodeGraph")
    def test_plot_dependencies_delegates_to_graph(self, node_graph):
        variable = _random_variable()

        variable.plot_dependencies(max_size=7)

        node_graph.assert_called_once_with(variable._node, max_size=7)
        node_graph.return_value.plot.assert_called_once_with()

    @patch("problab.random_variables.base._RealizationContext")
    def test_sample_creates_context_and_evaluates_own_node(self, realization_context):
        variable = _random_variable()
        rng = np.random.default_rng(1)
        realization_context.return_value.evaluate.return_value = sentinel.samples

        samples = variable.sample(num_samples=3, rng=rng, validate=True)

        self.assertIs(samples, sentinel.samples)
        realization_context.assert_called_once_with(
            root_node=variable._node,
            num_samples=3,
            rng=rng,
            validate=True,
        )
        realization_context.return_value.evaluate.assert_called_once_with(variable._node)

    def test_is_real_or_complex_reflects_declared_value_set(self):
        self.assertTrue(_random_variable(value_set=REALS)._is_real_or_complex())
        self.assertFalse(
            _random_variable(value_set=ObjectValueSet(objects=("red",)))._is_real_or_complex(),
        )

    def test_realize_delegates_to_sample_defaults(self):
        variable = _random_variable()
        variable.sample = Mock(return_value=sentinel.samples)

        samples = variable.realize()

        self.assertIs(samples, sentinel.samples)
        variable.sample.assert_called_once_with()

    def test_binary_operation_builds_operation_node_with_scalar_constant(self):
        variable = _random_variable("X")

        result = variable + 2

        self.assertIsInstance(result, RandomVariable)
        self.assertIsInstance(result._node, _OperationNode)
        self.assertEqual(result._node.name, "(X + 2)")
        self.assertIs(result._node._operation, _ADD.operation)
        self.assertIs(result._node._inputs[0], variable._node)
        self.assertIsInstance(result._node._inputs[1], _ConstantNode)
        self.assertEqual(result._node._inputs[1].value, 2)

    def test_reverse_binary_operation_reverses_input_order(self):
        variable = _random_variable("X")

        result = 2 - variable

        self.assertEqual(result._node.name, "(2 - X)")
        self.assertEqual(result._node._inputs[1], variable._node)

    def test_public_arithmetic_methods_delegate_with_correct_operations(self):
        variable = _random_variable()
        variable._binary_operation = Mock(return_value=sentinel.result)

        cases = (
            ("__add__", _ADD, False),
            ("__radd__", _ADD, False),
            ("__sub__", _SUBTRACT, False),
            ("__rsub__", _SUBTRACT, True),
            ("__mul__", _MULTIPLY, False),
            ("__rmul__", _MULTIPLY, False),
            ("__truediv__", _DIVIDE, False),
            ("__rtruediv__", _DIVIDE, True),
            ("__mod__", _MODULO, False),
            ("__rmod__", _MODULO, True),
            ("__pow__", _POWER, False),
            ("__rpow__", _POWER, True),
        )

        for method_name, operation, reverse in cases:
            with self.subTest(method=method_name):
                variable._binary_operation.reset_mock()

                self.assertIs(getattr(variable, method_name)(2), sentinel.result)

                expected = call(2, operation, reverse=True) if reverse else call(2, operation)
                variable._binary_operation.assert_called_once_with(*expected.args, **expected.kwargs)

    def test_binary_operation_rejects_unsupported_operand(self):
        variable = _random_variable()

        self.assertIs(variable._binary_operation("text", _ADD), NotImplemented)

    def test_binary_operation_rejects_non_numeric_value_set(self):
        variable = _random_variable(value_set=ObjectValueSet(objects=("red",)))

        self.assertIs(variable._binary_operation(1, _ADD), NotImplemented)

    def test_real_power_uses_real_numpy_power_operation(self):
        variable = _random_variable(value_set=REALS)

        result = variable ** 2

        self.assertIs(result._node._operation, np.power)

    def test_unary_operation_builds_node(self):
        variable = _random_variable("X")

        result = -variable

        self.assertIsInstance(result._node, _OperationNode)
        self.assertEqual(result._node.name, "(-X)")
        self.assertEqual(result._node._inputs, (variable._node,))

    def test_public_unary_methods_delegate_with_correct_operations(self):
        variable = _random_variable()
        variable._unary_operation = Mock(return_value=sentinel.result)

        self.assertIs(-variable, sentinel.result)
        variable._unary_operation.assert_called_once_with(_NEGATIVE)

        variable._unary_operation.reset_mock()
        self.assertIs(abs(variable), sentinel.result)
        variable._unary_operation.assert_called_once_with(_ABS)

    def test_apply_wraps_non_vectorized_function_for_sample_arrays(self):
        variable = _random_variable("X")

        result = variable.apply(
            lambda value: value + 1,
            value_set=REALS,
            function_name="increment",
        )
        context = Mock()
        context.evaluate.return_value = np.array([1, 2])

        values = result._node._evaluate(context)

        np.testing.assert_array_equal(values, [2, 3])
        self.assertEqual(result._node.name, "increment(X)")
        context.evaluate.assert_called_once_with(variable._node)

    def test_apply_preserves_vectorized_function(self):
        variable = _random_variable()

        result = variable.apply(np.negative, value_set=REALS, vectorized=True)

        self.assertIs(result._node._operation, np.negative)

    def test_apply_passes_aligned_values_from_other_variables(self):
        variable = _random_variable("X")
        other = _random_variable("Y")
        result = variable.apply(
            lambda left, right: left + right,
            other,
            value_set=REALS,
            function_name="sum",
        )
        context = Mock()
        context.evaluate.side_effect = (
            np.array([1, 2]),
            np.array([10, 20]),
        )

        values = result._node._evaluate(context)

        np.testing.assert_array_equal(values, [11, 22])
        self.assertEqual(result._node.name, "sum(X, Y)")
        context.evaluate.assert_has_calls([call(variable._node), call(other._node)])

    def test_interval_uses_inverted_cdf_quantiles_of_samples(self):
        variable = _random_variable()
        variable.sample = Mock(return_value=np.array([1.0, 2.0, 3.0, 4.0]))
        rng = np.random.default_rng(2)

        interval = variable.interval(alpha=0.5, num_samples=4, rng=rng)

        self.assertEqual((interval.lower, interval.upper), (1.0, 3.0))
        self.assertEqual(interval.alpha, 0.5)
        self.assertTrue(interval.is_estimate)
        variable.sample.assert_called_once_with(num_samples=4, rng=rng)

    def test_is_in_interval_rejects_reversed_bounds(self):
        variable = _random_variable()

        with self.assertRaises(ValueError):
            variable.is_in_interval(2, 1)

    def test_is_in_interval_honors_each_closure_mode(self):
        variable = _random_variable("X")
        sample_values = np.array([0.0, 1.0, 2.0])
        context = Mock()

        def evaluate(node):
            if node is variable._node:
                return sample_values
            return node._evaluate(context)

        context.evaluate.side_effect = evaluate
        expected = {
            "both": [False, True, True],
            "left": [False, True, False],
            "right": [False, False, True],
            "none": [False, False, False],
        }

        for closed, values in expected.items():
            with self.subTest(closed=closed):
                event = variable.is_in_interval(1.0, 2.0, closed=closed)
                np.testing.assert_array_equal(event._node._evaluate(context), values)

    def test_is_in_delegates_tuple_and_list_ranges_with_expected_closure(self):
        variable = _random_variable()
        variable.is_in_interval = Mock(return_value=sentinel.event)

        self.assertIs(variable.is_in((1, 2)), sentinel.event)
        variable.is_in_interval.assert_called_once_with(1, 2, closed="none")

        variable.is_in_interval.reset_mock()
        self.assertIs(variable.is_in([1, 2]), sentinel.event)
        variable.is_in_interval.assert_called_once_with(1, 2, closed="both")

    def test_is_in_builds_boolean_event_for_sympy_set(self):
        variable = _random_variable("X")

        event = variable.is_in(sp.FiniteSet(1, 3))
        context = Mock()
        context.evaluate.return_value = np.array([1, 2, 3])

        values = event._node._evaluate(context)

        np.testing.assert_array_equal(values, [True, False, True])
        self.assertEqual(event.name, "{X in {1, 3}}")

    def test_comparison_builds_boolean_event(self):
        variable = _random_variable("X")

        event = variable < 2
        context = Mock()
        context.evaluate.side_effect = (
            np.array([1, 3]),
            np.array([2, 2]),
        )

        values = event._node._evaluate(context)

        np.testing.assert_array_equal(values, [True, False])
        self.assertEqual(event.name, "{X < 2}")

    def test_public_comparison_methods_delegate_with_correct_operations(self):
        variable = _random_variable()
        variable._inequality_comparison = Mock(return_value=sentinel.inequality_event)
        variable._equality_comparison = Mock(return_value=sentinel.equality_event)

        for method_name, operation in (
            ("__lt__", _LT),
            ("__le__", _LTE),
            ("__gt__", _GT),
            ("__ge__", _GTE),
        ):
            with self.subTest(method=method_name):
                variable._inequality_comparison.reset_mock()
                self.assertIs(getattr(variable, method_name)(2), sentinel.inequality_event)
                variable._inequality_comparison.assert_called_once_with(operation, 2)

        for method_name, operation in (("__eq__", _EQ), ("__ne__", _NEQ)):
            with self.subTest(method=method_name):
                variable._equality_comparison.reset_mock()
                self.assertIs(getattr(variable, method_name)("red"), sentinel.equality_event)
                variable._equality_comparison.assert_called_once_with(operation, "red")

    @patch("problab.random_variables.base._quantile_confidence_interval")
    def test_quantile_confidence_interval_delegates_sample_and_parameters(
        self,
        quantile_confidence_interval,
    ):
        variable = _random_variable()
        variable.sample = Mock(return_value=sentinel.samples)
        quantile_confidence_interval.return_value = sentinel.interval
        rng = np.random.default_rng(3)

        interval = variable.quantile_confidence_interval(
            q=0.25,
            alpha=0.1,
            num_samples=8,
            rng=rng,
        )

        self.assertIs(interval, sentinel.interval)
        variable.sample.assert_called_once_with(num_samples=8, rng=rng)
        quantile_confidence_interval.assert_called_once_with(
            samples=sentinel.samples,
            q=0.25,
            alpha=0.1,
        )

    def test_real_only_methods_reject_non_real_value_sets(self):
        variable = _random_variable(value_set=COMPLEXES)

        with self.assertRaises(TypeError):
            variable.interval(alpha=0.1)
        with self.assertRaises(TypeError):
            variable.is_in_interval(0, 1)
        with self.assertRaises(TypeError):
            variable.quantile_confidence_interval(q=0.5)


if __name__ == "__main__":
    unittest.main()
