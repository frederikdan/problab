from functools import wraps
from inspect import signature


def _validate_parameters(**validators):
    def decorate(function):
        function_signature = signature(function)

        for name, validator in validators.items():
            if name not in function_signature.parameters:
                raise ValueError(f"Unknown parameter {name!r} for {function.__qualname__}.")

            if not callable(validator):
                raise TypeError(f"The validator for {name!r} must be callable.")

        @wraps(function)
        def wrapped(*args, **kwargs):
            arguments = function_signature.bind(*args, **kwargs)
            arguments.apply_defaults()

            for name, validator in validators.items():
                validator(arguments.arguments[name])

            return function(*arguments.args, **arguments.kwargs)

        return wrapped

    return decorate