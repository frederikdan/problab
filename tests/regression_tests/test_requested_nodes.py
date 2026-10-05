import unittest
from unittest.mock import patch

import numpy as np

from problab import CategoricalDistribution, P, RandomVariable


class RequestedNodesRegressionTests(unittest.TestCase):

    def setUp(self):
        self.variable = RandomVariable(CategoricalDistribution([0, 1], [0.5, 0.5]))

    @patch.object(CategoricalDistribution, "_sample")
    def test_complement_conditioning_preserves_exact_condition_count(self, sample):
        sample.return_value = np.array([0, 1, 0, 0])
        event = self.variable == 1

        result = P(event, given=~event, num_samples=4, validate=True)

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.num_successes, 0)
        self.assertEqual(result.num_conditioned_samples, 3)
        self.assertEqual(result.num_unconditioned_samples, 4)
        sample.assert_called_once()

    @patch.object(CategoricalDistribution, "_sample")
    def test_identical_event_and_condition_reuse_one_realization(self, sample):
        sample.return_value = np.array([0, 1, 0, 0])
        event = self.variable == 1

        result = P(event, given=event, num_samples=4, validate=True)

        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.num_successes, 1)
        self.assertEqual(result.num_conditioned_samples, 1)
        sample.assert_called_once()

    @patch.object(CategoricalDistribution, "_sample")
    def test_constant_joint_with_unsatisfied_condition_returns_nan(self, sample):
        sample.return_value = np.ones(4, dtype=int)
        event = self.variable == 1

        result = P(event, given=~event, num_samples=4)

        self.assertTrue(np.isnan(result.value))
        self.assertEqual(result.num_successes, 0)
        self.assertEqual(result.num_conditioned_samples, 0)
        sample.assert_called_once()

    def test_dependency_graph_builds_from_single_variable_with_new_constructor(self):
        expression = self.variable + 2

        graph = expression.dependency_graph

        self.assertTrue(graph.is_complete)
        self.assertIn(expression._node, graph.nodes)
        self.assertIn(self.variable._node, graph.nodes)
        self.assertIn((expression._node, self.variable._node), graph.nx_graph.edges)

    @patch("problab.random_variables.graph.plt.show")
    @patch("problab.random_variables.graph.nx.draw")
    def test_plot_dependencies_builds_graph_with_new_constructor(self, draw, show):
        self.variable.plot_dependencies(max_size=1)

        self.assertEqual(set(draw.call_args.args[0].nodes), {self.variable._node})
        draw.assert_called_once()
        show.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
