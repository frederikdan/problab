import unittest

from problab.validation._decorator import _validate_parameters


class ParameterDecoratorTests(unittest.TestCase):

    def test_validates_explicit_and_default_parameter_values(self):
        values = []

        @_validate_parameters(value=values.append)
        def function(value=3):
            return value * 2

        self.assertEqual(function(), 6)
        self.assertEqual(function(4), 8)
        self.assertEqual(values, [3, 4])

    def test_preserves_function_metadata(self):
        @_validate_parameters(value=lambda value: None)
        def function(value):
            """A documented function."""

        self.assertEqual(function.__name__, "function")
        self.assertEqual(function.__doc__, "A documented function.")

    def test_rejects_unknown_parameter_and_non_callable_validator(self):
        def function(value):
            return value

        with self.assertRaises(ValueError):
            _validate_parameters(other=lambda value: None)(function)
        with self.assertRaises(TypeError):
            _validate_parameters(value="validator")(function)

    def test_preserves_normal_argument_binding_errors(self):
        @_validate_parameters(value=lambda value: None)
        def function(value):
            return value

        with self.assertRaises(TypeError):
            function()


if __name__ == "__main__":
    unittest.main()
