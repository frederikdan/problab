import dataclasses
import operator

from problab.operations._base import (
    _Operation,
    _BOOLEAN_INPUT_TYPES,
)


@dataclasses.dataclass(frozen=True)
class _LogicalOperation(_Operation):
    pass


# Logical operations
_AND = _LogicalOperation(
    operation=operator.and_,
    name_func=lambda a, b: f"{{{a} & {b}}}",
    supported_input_types=_BOOLEAN_INPUT_TYPES,
)

_OR = _LogicalOperation(
    operation=operator.or_,
    name_func=lambda a, b: f"{{{a} | {b}}}",
    supported_input_types=_BOOLEAN_INPUT_TYPES,
)

_XOR = _LogicalOperation(
    operation=operator.xor,
    name_func=lambda a, b: f"{{{a} ^ {b}}}",
    supported_input_types=_BOOLEAN_INPUT_TYPES,
)

_INVERT = _LogicalOperation(
    operation=operator.invert,
    name_func=lambda x: f"{{~{x}}}",
    supported_input_types=_BOOLEAN_INPUT_TYPES,
)
