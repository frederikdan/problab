import math
import unittest

import numpy as np

from problab import (
    BinomialDistribution,
    CategoricalDistribution,
    NormalDistribution,
    P,
    PoissonDistribution,
    RandomVariable,
)


class DistributionSamplingStatisticalTests(unittest.TestCase):

    def assert_mean_within_six_standard_errors(
        self,
        samples: np.ndarray,
        expected_mean: float,
        population_variance: float,
    ) -> None:
        standard_error = math.sqrt(population_variance / len(samples))
        difference = abs(float(np.mean(samples)) - expected_mean)
        self.assertLessEqual(
            difference,
            6 * standard_error,
            f"Sample mean differs from {expected_mean} by {difference}, "
            f"which is more than six standard errors ({standard_error}).",
        )

    def test_normal_sample_mean_matches_configured_mean(self):
        mean = 3.5
        standard_deviation = 1.75
        variable = RandomVariable(NormalDistribution(mean, standard_deviation))

        samples = variable.sample(num_samples=100_000, rng=np.random.default_rng(101))

        self.assert_mean_within_six_standard_errors(
            samples,
            expected_mean=mean,
            population_variance=standard_deviation ** 2,
        )

    def test_binomial_sample_mean_matches_configured_mean(self):
        trials = 12
        probability = 0.3
        variable = RandomVariable(BinomialDistribution(trials, probability))

        samples = variable.sample(num_samples=100_000, rng=np.random.default_rng(102))

        self.assert_mean_within_six_standard_errors(
            samples,
            expected_mean=trials * probability,
            population_variance=trials * probability * (1 - probability),
        )

    def test_poisson_sample_mean_matches_configured_mean(self):
        rate = 4.5
        variable = RandomVariable(PoissonDistribution(rate))

        samples = variable.sample(num_samples=100_000, rng=np.random.default_rng(103))

        self.assert_mean_within_six_standard_errors(
            samples,
            expected_mean=rate,
            population_variance=rate,
        )

    def test_categorical_sample_frequencies_match_configured_probabilities(self):
        categories = ("small", "medium", "large")
        probabilities = (0.1, 0.3, 0.6)
        variable = RandomVariable(CategoricalDistribution(categories, probabilities))
        num_samples = 100_000

        samples = variable.sample(num_samples=num_samples, rng=np.random.default_rng(104))

        for category, probability in zip(categories, probabilities):
            with self.subTest(category=category):
                observed_frequency = float(np.mean(samples == category))
                standard_error = math.sqrt(probability * (1 - probability) / num_samples)
                self.assertLessEqual(
                    abs(observed_frequency - probability),
                    6 * standard_error,
                )

    def test_probability_estimate_matches_a_categorical_event_probability(self):
        probability = 0.3
        variable = RandomVariable(
            CategoricalDistribution(["small", "medium", "large"], [0.1, probability, 0.6]),
        )
        num_samples = 100_000

        result = P(
            variable == "medium",
            num_samples=num_samples,
            rng=np.random.default_rng(105),
        )

        standard_error = math.sqrt(probability * (1 - probability) / num_samples)
        self.assertLessEqual(abs(result.value - probability), 6 * standard_error)
        self.assertEqual(result.value, result.num_successes / num_samples)


if __name__ == "__main__":
    unittest.main()
