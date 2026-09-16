import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np

from problab.random_variables._context import _RealizationContext
from problab.random_variables.nodes.base import _Node
from problab.value_sets.sets import REALS


class _StubNode(_Node):

    def __init__(self, name, value_set=REALS, dependencies=(), evaluator=None):
        super().__init__()
        self._name = name
        self._value_set = value_set
        self._dependencies = set(dependencies)
        self._evaluator = Mock() if evaluator is None else evaluator

    def __repr__(self):
        return f"StubNode({self.name})"

    @property
    def value_set(self):
        return self._value_set

    @property
    def dependencies(self):
        return self._dependencies

    def _evaluate(self, context):
        return self._evaluator(context)


class RealizationContextTests(unittest.TestCase):

    def test_constructor_stores_sample_count_and_supplied_rng(self):
        root = _StubNode("root")
        rng = np.random.default_rng(1)

        context = _RealizationContext(root, num_samples=np.int64(3), rng=rng)

        self.assertEqual(context.num_samples, 3)
        self.assertIs(context.rng, rng)

    def test_constructor_rejects_invalid_sample_count(self):
        root = _StubNode("root")

        with self.assertRaises(TypeError):
            _RealizationContext(root, num_samples=True)
        with self.assertRaises(ValueError):
            _RealizationContext(root, num_samples=0)

    def test_evaluate_caches_node_result(self):
        evaluator = Mock(return_value=np.array([1.0, 2.0]))
        node = _StubNode("root", evaluator=evaluator)
        context = _RealizationContext(node, num_samples=2)

        first = context.evaluate(node)
        second = context.evaluate(node)

        self.assertIs(first, second)
        evaluator.assert_called_once_with(context)
        self.assertIn(node, context)

    def test_evaluate_broadcasts_scalar_result_to_sample_count(self):
        node = _StubNode("root", evaluator=lambda context: np.array(2.0))
        context = _RealizationContext(node, num_samples=3)

        result = context.evaluate(node)

        np.testing.assert_array_equal(result, [2.0, 2.0, 2.0])

    def test_evaluate_rejects_wrong_sample_shape(self):
        node = _StubNode("root", evaluator=lambda context: np.array([1.0, 2.0]))
        context = _RealizationContext(node, num_samples=3)

        with self.assertRaises(ValueError):
            context.evaluate(node)

    def test_evaluate_rejects_dtype_outside_declared_family(self):
        node = _StubNode("root", evaluator=lambda context: np.array(["a"]))
        context = _RealizationContext(node, num_samples=1)

        with self.assertRaises(TypeError):
            context.evaluate(node)

    @patch("problab.random_variables._context.validate_as_subset")
    def test_evaluate_validates_result_when_requested(self, validate_as_subset):
        node = _StubNode("root", evaluator=lambda context: np.array([1.0]))
        context = _RealizationContext(node, num_samples=1, validate=True)

        context.evaluate(node)

        validate_as_subset.assert_called_once()
        self.assertIs(validate_as_subset.call_args.kwargs["target_set"], REALS)

    def test_evaluate_releases_dependency_after_last_dependant_is_realized(self):
        child = _StubNode("child", evaluator=lambda context: np.array([1.0]))
        root = _StubNode(
            "root",
            dependencies=(child,),
            evaluator=lambda context: context.evaluate(child) + 1,
        )
        context = _RealizationContext(root, num_samples=1)

        result = context.evaluate(root)

        np.testing.assert_array_equal(result, [2.0])
        self.assertIn(root, context)
        self.assertNotIn(child, context)

    @patch("problab.random_variables._context.NodeGraph")
    def test_constructor_rejects_incomplete_dependency_graph(self, node_graph):
        node_graph.return_value.is_complete = False

        with self.assertRaises(ValueError):
            _RealizationContext(sentinel.root_node)


if __name__ == "__main__":
    unittest.main()
