import unittest
from inspect import signature
from unittest.mock import Mock

from problab.validation._decorator import _validate_parameters


class ParameterDecoratorTests(unittest.TestCase):

    def test_forwarded_arguments_use_explicit_values_and_function_defaults(self):
        received = []

        def validate(value, *, policy):
            received.append((value, policy))

        @_validate_parameters(validator_arguments=("policy",), value=validate)
        def function(value=2, policy="warn"):
            return value

        self.assertEqual(function(), 2)
        self.assertEqual(function(3, "ignore"), 3)
        self.assertEqual(function(value=4, policy="raise"), 4)
        self.assertEqual(received, [(2, "warn"), (3, "ignore"), (4, "raise")])

    def test_forwarded_settings_are_validated_before_dependent_parameters(self):
        calls = []

        def validate(value, *, policy):
            calls.append(("value", value, policy))

        @_validate_parameters(validator_arguments=("policy",), value=validate,
                              policy=lambda value: calls.append(("policy", value)))
        def function(value, *, policy="warn"):
            calls.append(("body", value))

        function(7)
        self.assertEqual(calls, [("policy", "warn"), ("value", 7, "warn"), ("body", 7)])

    def test_invalid_forwarded_setting_prevents_parameter_validation_and_body(self):
        validate = Mock()
        body = Mock()

        def reject(value):
            raise ValueError("invalid policy")

        @_validate_parameters(validator_arguments=("policy",), value=validate, policy=reject)
        def function(value, policy="warn"):
            body()

        with self.assertRaisesRegex(ValueError, "invalid policy"):
            function(7, policy="invalid")
        validate.assert_not_called()
        body.assert_not_called()

    def test_forwards_only_arguments_explicitly_requested_by_each_validator(self):
        plain = Mock()
        received = []

        def validate(value, *, policy):
            received.append((value, policy))

        @_validate_parameters(validator_arguments=("policy",), left=plain, right=validate)
        def function(left, right, policy="warn"):
            pass

        function(1, 2)
        plain.assert_called_once_with(1)
        self.assertEqual(received, [(2, "warn")])

    def test_forwards_instance_and_setting_together_without_distribution_specific_names(self):
        received = []

        def validate(value, *, instance, option):
            received.append((value, instance, option))

        class Example:
            @_validate_parameters(validator_arguments=("option",), value=validate)
            def __init__(self, value, *, option=5):
                self.value = value

        instance = Example(3, option=9)
        self.assertEqual(received, [(3, instance, 9)])

    def test_rejects_unknown_forwarded_argument_when_decorating(self):
        with self.assertRaisesRegex(ValueError, "Unknown forwarded argument"):
            _validate_parameters(validator_arguments=("missing",))(lambda value: value)

    def test_rejects_positional_only_forwarded_validator_arguments(self):
        def validate(value, policy, /):
            pass
        with self.assertRaisesRegex(TypeError, "keyword argument"):
            _validate_parameters(validator_arguments=("policy",), value=validate)(lambda value, policy: value)

    def test_rejects_positional_only_instance_argument(self):
        def validate(value, instance, /):
            pass
        def method(self, value):
            pass
        with self.assertRaisesRegex(TypeError, "keyword argument"):
            _validate_parameters(value=validate)(method)

    def test_falls_back_to_value_only_for_callable_without_inspectable_signature(self):
        class Validator:
            __signature__ = "unavailable"
            def __init__(self):
                self.values = []
            def __call__(self, value):
                self.values.append(value)
        validator = Validator()
        function = _validate_parameters(value=validator)(lambda value: value + 1)
        self.assertEqual(function(4), 5)
        self.assertEqual(validator.values, [4])

    def test_wrapping_preserves_signature_and_rejects_duplicate_or_extra_arguments(self):
        def original(value, *, option=3):
            return value
        wrapped = _validate_parameters(value=lambda value: None)(original)
        self.assertEqual(signature(wrapped), signature(original))
        self.assertIs(wrapped.__wrapped__, original)
        for args, kwargs in (((1,), {"value": 2}), ((1,), {"extra": 2})):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                wrapped(*args, **kwargs)

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

    def test_passes_instance_before_constructor_initializes_it(self):
        calls = []

        def validate(value, *, instance):
            calls.append((value, instance, hasattr(instance, "value")))

        class Distribution:
            symbol = "N"

            @_validate_parameters(value=validate, other=calls.append)
            def __init__(self, value=3, other=4):
                self.value = value

        distribution = Distribution()

        self.assertEqual(calls, [(3, distribution, False), 4])
        self.assertEqual(distribution.value, 3)

    def test_passes_actual_subclass_instance(self):
        calls = []

        def validate(value, *, instance):
            calls.append((value, instance))

        class Distribution:
            symbol = "Base"

            @_validate_parameters(value=validate)
            def __init__(self, value):
                pass

        class Subclass(Distribution):
            symbol = "Subclass"

        distribution = Subclass(value=5)

        self.assertEqual(calls, [(5, distribution)])

    def test_rejects_instance_validator_without_instance_method(self):
        def validate(value, *, instance):
            pass

        def function(value):
            pass

        with self.assertRaises(ValueError):
            _validate_parameters(value=validate)(function)


if __name__ == "__main__":
    unittest.main()
