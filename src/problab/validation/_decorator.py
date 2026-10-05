from functools import wraps
from inspect import Parameter, signature


from functools import wraps
from inspect import Parameter, signature


def _validate_parameters(*,
                         validator_arguments: tuple[str, ...] = (),
                         **validators,
                         ):

    def decorate(function):
        function_signature = signature(function)
        arguments_requested_by_validators = {}

        for name in validator_arguments:
            if name not in function_signature.parameters:
                raise ValueError(
                    f"Unknown forwarded argument {name!r} "
                    f"for {function.__qualname__}."
                )

        for name, validator in validators.items():
            if name not in function_signature.parameters:
                raise ValueError(
                    f"Unknown parameter {name!r} "
                    f"for {function.__qualname__}."
                )

            if not callable(validator):
                raise TypeError(
                    f"The validator for {name!r} must be callable."
                )

            arguments_requested_by_validators[name] = []

            try:
                validator_signature = signature(validator)
            except (TypeError, ValueError):
                continue

            for argument_name in ("instance", *validator_arguments):
                parameter = validator_signature.parameters.get(argument_name)

                if parameter is None:
                    continue

                if parameter.kind not in (
                    Parameter.POSITIONAL_OR_KEYWORD,
                    Parameter.KEYWORD_ONLY,
                ):
                    raise TypeError(
                        f"{argument_name!r} must accept a keyword argument."
                    )

                if (
                    argument_name == "instance"
                    and "self" not in function_signature.parameters
                ):
                    raise ValueError(
                        "Validators requesting 'instance' require "
                        "an instance method with 'self'."
                    )

                arguments_requested_by_validators[name].append(argument_name)

        validation_order = (
            tuple(name for name in validators if name in validator_arguments)
            + tuple(name for name in validators if name not in validator_arguments)
        )

        @wraps(function)
        def wrapped(*args, **kwargs):
            arguments = function_signature.bind(*args, **kwargs)
            arguments.apply_defaults()

            for name in validation_order:
                validator_kwargs = {
                    argument_name: arguments.arguments[
                        "self" if argument_name == "instance" else argument_name
                    ]
                    for argument_name in arguments_requested_by_validators[name]
                }

                validators[name](
                    arguments.arguments[name],
                    **validator_kwargs,
                )

            return function(*arguments.args, **arguments.kwargs)

        return wrapped

    return decorate
