import unittest

from problab.validation.distributions.discrete._categorical import (
    _validate_categorical_configuration,
    _validate_categories,
    _validate_probabilities,
)


class CategoricalDistributionValidationTests(unittest.TestCase):

    def test_category_and_probability_inputs_require_iterables(self):
        _validate_categories(["red"])
        _validate_probabilities((1.0,))

        with self.assertRaises(TypeError):
            _validate_categories(1)
        with self.assertRaises(TypeError):
            _validate_probabilities(1)

    def test_configuration_accepts_matching_normalized_probabilities(self):
        _validate_categorical_configuration(("red", "blue"), (0.25, 0.75))

    def test_configuration_rejects_invalid_category_probability_pairs(self):
        with self.assertRaises(ValueError):
            _validate_categorical_configuration((), ())
        with self.assertRaises(ValueError):
            _validate_categorical_configuration(("red",), (0.5, 0.5))
        with self.assertRaises(TypeError):
            _validate_categorical_configuration(("red",), ("one",))
        with self.assertRaises(ValueError):
            _validate_categorical_configuration(("red",), (float("inf"),))
        with self.assertRaises(ValueError):
            _validate_categorical_configuration(("red",), (-0.1,))
        with self.assertRaises(ValueError):
            _validate_categorical_configuration(("red",), (0.5,))


if __name__ == "__main__":
    unittest.main()
