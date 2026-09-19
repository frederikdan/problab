import math
import statistics
import unittest

import numpy as np
from scipy.stats import chi2

from problab import BinomialDistribution, Mode, NormalDistribution, RandomVariable


NUM_SAMPLES = 100_000
DKW_ERROR_PROBABILITY = 1e-8


def binomial_cdf(value: int, trials: int, probability: float) -> float:
    return sum(
        math.comb(trials, k) * probability ** k * (1 - probability) ** (trials - k)
        for k in range(min(value, trials) + 1)
    )


def dkw_tolerance(samples: int) -> float:
    return math.sqrt(math.log(2 / DKW_ERROR_PROBABILITY) / (2 * samples))


class DistributionMethodStatisticalTests(unittest.TestCase):

    def test_normal_mean_variance_and_std_match_their_sampling_laws(self):
        mean, standard_deviation = 1.5, 2.0
        distribution = NormalDistribution(mean, standard_deviation)
        variance = standard_deviation ** 2

        observed_mean = distribution.mean(
            mode=Mode.AUTO,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(401),
        )
        observed_variance = distribution.variance(
            mode=Mode.MONTE_CARLO,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(402),
        )
        observed_std = distribution.std(
            mode=Mode.MONTE_CARLO,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(403),
        )

        self.assertLessEqual(
            abs(observed_mean - mean),
            6 * standard_deviation / math.sqrt(NUM_SAMPLES),
        )
        lower_variance = variance * chi2.ppf(1e-8, NUM_SAMPLES - 1) / NUM_SAMPLES
        upper_variance = variance * chi2.isf(1e-8, NUM_SAMPLES - 1) / NUM_SAMPLES
        self.assertLessEqual(lower_variance, observed_variance)
        self.assertLessEqual(observed_variance, upper_variance)
        self.assertLessEqual(math.sqrt(lower_variance), observed_std)
        self.assertLessEqual(observed_std, math.sqrt(upper_variance))

    def test_normal_cdf_array_matches_analytical_normal_cdf(self):
        reference = statistics.NormalDist(mu=0.4, sigma=1.3)
        thresholds = np.array([-1.0, 0.0, 1.5])
        distribution = NormalDistribution(0.4, 1.3)

        estimates = distribution.cdf(
            thresholds,
            mode=Mode.MONTE_CARLO,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(404),
        )

        self.assertEqual(estimates.shape, thresholds.shape)
        for threshold, observed in zip(thresholds, estimates):
            with self.subTest(threshold=threshold):
                expected = reference.cdf(float(threshold))
                standard_error = math.sqrt(expected * (1 - expected) / NUM_SAMPLES)
                self.assertLessEqual(abs(observed - expected), 6 * standard_error)

    def test_normal_ppf_array_lies_within_analytical_dkw_bounds(self):
        reference = statistics.NormalDist(mu=-0.2, sigma=1.5)
        quantiles = np.array([0.1, 0.5, 0.9])
        distribution = NormalDistribution(-0.2, 1.5)

        estimates = distribution.ppf(
            quantiles,
            mode=Mode.MONTE_CARLO,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(405),
        )

        self.assertEqual(estimates.shape, quantiles.shape)
        tolerance = dkw_tolerance(NUM_SAMPLES)
        for quantile, observed in zip(quantiles, estimates):
            with self.subTest(quantile=quantile):
                self.assertLessEqual(reference.inv_cdf(quantile - tolerance), observed)
                self.assertLessEqual(observed, reference.inv_cdf(quantile + tolerance))

    def test_binomial_cdf_and_ppf_match_finite_distribution(self):
        trials, probability = 7, 0.35
        distribution = BinomialDistribution(trials, probability)
        thresholds = np.array([1, 3, 5])
        quantiles = np.array([0.1, 0.5, 0.9])

        cdf_estimates = distribution.cdf(
            thresholds,
            mode=Mode.MONTE_CARLO,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(406),
        )
        ppf_estimates = distribution.ppf(
            quantiles,
            mode=Mode.MONTE_CARLO,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(407),
        )

        tolerance = dkw_tolerance(NUM_SAMPLES)
        for threshold, observed in zip(thresholds, cdf_estimates):
            with self.subTest(method="cdf", threshold=threshold):
                expected = binomial_cdf(int(threshold), trials, probability)
                self.assertLessEqual(abs(observed - expected), tolerance)
        for quantile, observed in zip(quantiles, ppf_estimates):
            with self.subTest(method="ppf", quantile=quantile):
                value = int(observed)
                self.assertEqual(observed, value)
                self.assertLessEqual(binomial_cdf(value - 1, trials, probability), quantile + tolerance)
                self.assertGreaterEqual(binomial_cdf(value, trials, probability), quantile - tolerance)

    def test_random_variable_probability_interval_matches_normal_quantiles(self):
        reference = statistics.NormalDist(mu=0.5, sigma=1.2)
        variable = RandomVariable(NormalDistribution(0.5, 1.2))
        alpha = 0.1

        interval = variable.interval(
            alpha=alpha,
            num_samples=NUM_SAMPLES,
            rng=np.random.default_rng(408),
        )

        tolerance = dkw_tolerance(NUM_SAMPLES)
        for observed, quantile in ((interval.lower, alpha / 2), (interval.upper, 1 - alpha / 2)):
            with self.subTest(quantile=quantile):
                self.assertLessEqual(reference.inv_cdf(quantile - tolerance), observed)
                self.assertLessEqual(observed, reference.inv_cdf(quantile + tolerance))


if __name__ == "__main__":
    unittest.main()
