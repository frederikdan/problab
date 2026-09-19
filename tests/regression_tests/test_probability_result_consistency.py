import unittest

from problab import ProbabilityResult


class ProbabilityResultConsistencyRegressionTests(unittest.TestCase):

    def test_probability_must_match_success_count(self):
        with self.assertRaises(ValueError):
            ProbabilityResult(
                value=0.9,
                num_successes=0,
                num_unconditioned_samples=10,
            )

    def test_conditional_probability_must_match_conditioned_count(self):
        with self.assertRaises(ValueError):
            ProbabilityResult(
                value=0.5,
                num_successes=2,
                num_unconditioned_samples=10,
                num_conditioned_samples=2,
            )


if __name__ == "__main__":
    unittest.main()
