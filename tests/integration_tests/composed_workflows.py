import unittest

import numpy as np

from problab import BinomialDistribution, CategoricalDistribution, P, RandomVariable


class ComposedWorkflowIntegrationTests(unittest.TestCase):

    def test_distribution_can_use_random_variable_parameters_through_sampling(self):
        trials = RandomVariable(CategoricalDistribution([3], [1.0]), name="trials")
        probability = RandomVariable(CategoricalDistribution([1.0], [1.0]), name="p")
        variable = RandomVariable(BinomialDistribution(trials, probability), name="X")

        samples = variable.sample(
            num_samples=8,
            rng=np.random.default_rng(4),
            validate=True,
        )

        np.testing.assert_array_equal(samples, np.full(8, 3))
        self.assertTrue(variable.dependency_graph.is_complete)
        self.assertGreater(variable.dependency_graph.size, 1)

    def test_probability_uses_one_shared_realization_for_related_events(self):
        variable = RandomVariable(
            CategoricalDistribution([0, 1], [0.5, 0.5]),
            name="X",
        )

        result = P(
            (variable + variable) == 2,
            given=variable == 1,
            num_samples=20,
            rng=np.random.default_rng(9),
            validate=True,
        )

        self.assertEqual(result.value, 1.0)
        self.assertGreater(result.num_conditioned_samples, 0)
        self.assertEqual(result.num_successes, result.num_conditioned_samples)

    def test_probability_result_provides_a_confidence_interval_after_evaluation(self):
        variable = RandomVariable(CategoricalDistribution(["ready"], [1.0]))

        result = P(variable == "ready", num_samples=12, validate=True)
        interval = result.confidence_interval(alpha=0.1)

        self.assertEqual(result.value, 1.0)
        self.assertEqual((result.num_successes, result.num_samples), (12, 12))
        self.assertLessEqual(interval.lower, result.value)
        self.assertEqual(interval.upper, 1.0)

    def test_interval_estimation_works_for_a_composed_random_variable(self):
        source = RandomVariable(CategoricalDistribution([2], [1.0]))
        variable = source * 3 - 1

        interval = variable.interval(
            alpha=0.1,
            num_samples=10,
            rng=np.random.default_rng(5),
        )

        self.assertEqual((interval.lower, interval.upper), (5.0, 5.0))
        self.assertTrue(interval.is_estimate)


if __name__ == "__main__":
    unittest.main()
