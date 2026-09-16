import unittest

from problab.probability._config import (
    DEFAULT_PROB_NUM_SAMPLES,
    DEF_PROB_NUM_SAMPLES_MAX,
)


class ProbabilityConfigTests(unittest.TestCase):

    def test_default_sample_count_is_within_configured_limit(self):
        self.assertEqual(DEFAULT_PROB_NUM_SAMPLES, 100_000)
        self.assertEqual(DEF_PROB_NUM_SAMPLES_MAX, 1_000_000)
        self.assertLessEqual(DEFAULT_PROB_NUM_SAMPLES, DEF_PROB_NUM_SAMPLES_MAX)


if __name__ == "__main__":
    unittest.main()
