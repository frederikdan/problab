import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np
import sympy as sp
from problab.value_sets import HomogeneousNumericValueSet

from problab.random_variables._context import _RealizationContext
from problab.random_variables.nodes.base import _Node
from problab.value_sets.sets import NON_NEGATIVE_REALS, POSITIVE_REALS, REALS, UNKNOWN_VALUE_SET


class _StubNode(_Node):

    def __init__(
        self,
        name,
        value_set=REALS,
        realization_value_set=None,
        dependencies=(),
        evaluator=None,
    ):
        super().__init__()
        self._name = name
        self._value_set = value_set
        self.__realization_value_set = value_set if realization_value_set is None else realization_value_set
        self._dependencies = set(dependencies)
        self._evaluator = Mock() if evaluator is None else evaluator

    def __repr__(self):
        return f"StubNode({self.name})"

    @property
    def value_set(self):
        return self._value_set

    @property
    def _realization_value_set(self):
        return self.__realization_value_set

    @property
    def dependencies(self):
        return self._dependencies

    def _evaluate(self, context):
        return self._evaluator(context)


class RealizationContextTests(unittest.TestCase):

    def test_context_stores_default_and_explicit_numerical_error_policy(self):
        node = _StubNode("root")
        self.assertEqual(_RealizationContext((node,)).numerical_error_policy, "warn")
        for policy in ("warn", "raise", "ignore"):
            self.assertEqual(_RealizationContext((node,), numerical_error_policy=policy).numerical_error_policy, policy)

    def test_duplicate_requests_preserve_results_without_duplicate_evaluation(self):
        node = _StubNode("root", evaluator=Mock(return_value=np.array([1.0])))
        context = _RealizationContext((node, node))
        result = context.evaluate(node)
        self.assertIs(context.evaluate(node), result)
        node._evaluator.assert_called_once()

    def test_validate_uses_declared_exceptional_permissions_independently_of_policy(self):
        for policy in ("warn", "raise", "ignore"):
            for permitted in (False, True):
                support = HomogeneousNumericValueSet(sp.S.Reals, (np.float64,), allows_nan=permitted)
                node = _StubNode("root", realization_value_set=support, evaluator=lambda context: np.array([np.nan]))
                context = _RealizationContext((node,), validate=True, numerical_error_policy=policy)
                with self.subTest(policy=policy, permitted=permitted):
                    if permitted:
                        self.assertTrue(np.isnan(context.evaluate(node)[0]))
                    else:
                        with self.assertRaises(ValueError):
                            context.evaluate(node)
                        self.assertNotIn(node, context)

    def test_failed_parent_evaluation_does_not_release_dependencies_or_cache_failed_result(self):
        child = _StubNode("child", evaluator=Mock(return_value=np.array([2.0])))
        attempts = []
        def evaluate_parent(context):
            value = context.evaluate(child)
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("retry")
            return value + 1
        parent = _StubNode("parent", dependencies=(child,), evaluator=evaluate_parent)
        context = _RealizationContext((parent,))
        with self.assertRaisesRegex(RuntimeError, "retry"):
            context.evaluate(parent)
        self.assertNotIn(parent, context)
        self.assertIn(child, context)
        np.testing.assert_array_equal(context.evaluate(parent), [3.0])
        self.assertNotIn(child, context)
        child._evaluator.assert_called_once()

    def test_constructor_rejects_empty_requested_nodes(self):
        with self.assertRaisesRegex(ValueError, "at least one node"):
            _RealizationContext(())

    def test_constructor_applies_graph_limit_to_all_requested_nodes(self):
        with self.assertRaisesRegex(ValueError, "maximum size of 1 nodes"):
            _RealizationContext((_StubNode("first"), _StubNode("second")), max_graph_size=1)

    def test_evaluate_preserves_requested_dependency_in_either_order(self):
        for child_first in (True, False):
            with self.subTest(child_first=child_first):
                evaluator = Mock(side_effect=[np.array([2.0, 4.0]), np.array([8.0, 16.0])])
                child = _StubNode("child", evaluator=evaluator)
                parent = _StubNode("parent", dependencies=(child,),
                                   evaluator=lambda context: context.evaluate(child) + 1)
                context = _RealizationContext((parent, child), num_samples=2)

                if child_first:
                    child_values = context.evaluate(child)
                parent_values = context.evaluate(parent)
                if not child_first:
                    child_values = context.evaluate(child)

                np.testing.assert_array_equal(parent_values, child_values + 1)
                self.assertIs(context.evaluate(child), child_values)
                self.assertIs(context.evaluate(parent), parent_values)
                evaluator.assert_called_once_with(context)

    def test_evaluate_releases_shared_dependency_only_after_both_roots(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                evaluator = Mock(return_value=np.array([2.0, 4.0]))
                child = _StubNode("child", evaluator=evaluator)
                first = _StubNode("first", dependencies=(child,),
                                  evaluator=lambda context: context.evaluate(child) + 1)
                second = _StubNode("second", dependencies=(child,),
                                   evaluator=lambda context: context.evaluate(child) * 2)
                context = _RealizationContext((first, second), num_samples=2)
                earlier, later = (second, first) if reverse else (first, second)

                context.evaluate(earlier)
                self.assertIn(child, context)
                context.evaluate(later)
                self.assertNotIn(child, context)
                np.testing.assert_array_equal(context.evaluate(first), [3.0, 5.0])
                np.testing.assert_array_equal(context.evaluate(second), [4.0, 8.0])
                evaluator.assert_called_once_with(context)

    def test_evaluate_duplicate_requested_nodes_does_not_resample(self):
        evaluator = Mock(return_value=np.array([2.0]))
        child = _StubNode("child", evaluator=evaluator)
        root = _StubNode("root", dependencies=(child,),
                         evaluator=lambda context: context.evaluate(child) + 1)
        context = _RealizationContext((root, root))

        first = context.evaluate(root)
        self.assertIs(context.evaluate(root), first)
        self.assertNotIn(child, context)
        evaluator.assert_called_once_with(context)

    def test_constructor_stores_sample_count_and_supplied_rng(self):
        root = _StubNode("root")
        rng = np.random.default_rng(1)

        context = _RealizationContext((root,), num_samples=np.int64(3), rng=rng)

        self.assertEqual(context.num_samples, 3)
        self.assertIs(context.rng, rng)

    def test_constructor_creates_a_generator_when_rng_is_not_supplied(self):
        context = _RealizationContext((_StubNode("root"),))

        self.assertIsInstance(context.rng, np.random.Generator)

    def test_constructor_rejects_invalid_sample_count(self):
        root = _StubNode("root")

        with self.assertRaises(TypeError):
            _RealizationContext((root,), num_samples=True)
        with self.assertRaises(ValueError):
            _RealizationContext((root,), num_samples=0)

    def test_evaluate_caches_node_result(self):
        evaluator = Mock(return_value=np.array([1.0, 2.0]))
        node = _StubNode("root", evaluator=evaluator)
        context = _RealizationContext((node,), num_samples=2)

        first = context.evaluate(node)
        second = context.evaluate(node)

        self.assertIs(first, second)
        evaluator.assert_called_once_with(context)
        self.assertIn(node, context)

    def test_evaluate_broadcasts_scalar_result_to_sample_count(self):
        node = _StubNode("root", evaluator=lambda context: np.array(2.0))
        context = _RealizationContext((node,), num_samples=3)

        result = context.evaluate(node)

        np.testing.assert_array_equal(result, [2.0, 2.0, 2.0])

    def test_evaluate_rejects_wrong_sample_shape(self):
        node = _StubNode("root", evaluator=lambda context: np.array([1.0, 2.0]))
        context = _RealizationContext((node,), num_samples=3)

        with self.assertRaises(ValueError):
            context.evaluate(node)

    def test_evaluate_rejects_dtype_outside_declared_family(self):
        node = _StubNode("root", evaluator=lambda context: np.array(["a"]))
        context = _RealizationContext((node,), num_samples=1)

        with self.assertRaises(TypeError):
            context.evaluate(node)

    def test_evaluate_accepts_any_dtype_when_value_set_has_no_dtype_family(self):
        node = _StubNode(
            "root",
            value_set=UNKNOWN_VALUE_SET,
            evaluator=lambda context: np.array(["label"], dtype=object),
        )
        context = _RealizationContext((node,), num_samples=1)

        result = context.evaluate(node)

        np.testing.assert_array_equal(result, ["label"])

    @patch("problab.random_variables._context.validate_as_subset")
    def test_evaluate_validates_result_when_requested(self, validate_as_subset):
        node = _StubNode("root", evaluator=lambda context: np.array([1.0]))
        context = _RealizationContext((node,), num_samples=1, validate=True)

        context.evaluate(node)

        validate_as_subset.assert_called_once()
        self.assertIs(validate_as_subset.call_args.kwargs["target_set"], REALS)

    @patch("problab.random_variables._context.validate_as_subset")
    def test_evaluate_uses_realization_support_for_runtime_validation(self, validate_as_subset):
        node = _StubNode(
            "exp(X)",
            value_set=POSITIVE_REALS,
            realization_value_set=NON_NEGATIVE_REALS,
            evaluator=lambda context: np.array([0.0]),
        )
        context = _RealizationContext((node,), num_samples=1, validate=True)

        context.evaluate(node)

        self.assertIs(
            validate_as_subset.call_args.kwargs["target_set"],
            NON_NEGATIVE_REALS,
        )

    def test_evaluate_releases_dependency_after_last_dependant_is_realized(self):
        child = _StubNode("child", evaluator=lambda context: np.array([1.0]))
        root = _StubNode(
            "root",
            dependencies=(child,),
            evaluator=lambda context: context.evaluate(child) + 1,
        )
        context = _RealizationContext((root,), num_samples=1)

        result = context.evaluate(root)

        np.testing.assert_array_equal(result, [2.0])
        self.assertIn(root, context)
        self.assertNotIn(child, context)

    @patch("problab.random_variables._context.NodeGraph")
    def test_constructor_rejects_incomplete_dependency_graph(self, node_graph):
        node_graph.return_value.is_complete = False

        with self.assertRaises(ValueError):
            _RealizationContext((sentinel.root_node,))


if __name__ == "__main__":
    unittest.main()
