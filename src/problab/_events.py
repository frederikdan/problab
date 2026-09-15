from __future__ import annotations

import sympy as sp

from problab._operations import _AND, _INVERT, _OR
from problab.random_variables._nodes import _Node, _OperationNode
from problab.value_sets._utils import is_known_subset
from problab.value_sets.sets import BOOLEANS


class _Event:

    @classmethod
    def _from_node(cls, node: _Node) -> _Event:
        return cls(node)

    def __init__(self, node: _Node) -> None:
        if not is_known_subset(node.value_set, BOOLEANS):
            raise ValueError("The value set of 'node' must be a subset of {False, True}.")

        self._node = node

    @property
    def name(self) -> str:
        return self._node.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Event({self._node!r})"

    def __and__(self, other: _Event) -> _Event:

        if not isinstance(other, _Event):
            return NotImplemented

        node_name = _AND.name_func(self._node.name, other.name)

        return _Event._from_node(_OperationNode(
            operation=_AND.operation,
            inputs=(self._node, other._node),
            name=node_name,
            value_set=BOOLEANS,
        ))

    def __or__(self, other: _Event) -> _Event:

        if not isinstance(other, _Event):
            return NotImplemented

        node_name = _OR.name_func(self._node.name, other.name)

        return _Event._from_node(_OperationNode(
            operation=_OR.operation,
            inputs=(self._node, other._node),
            name=node_name,
            value_set=BOOLEANS,
        ))

    def __invert__(self) -> _Event:

        node_name = _INVERT.name_func(self._node.name)

        return _Event._from_node(_OperationNode(
            operation=_INVERT.operation,
            inputs=(self._node,),
            name=node_name,
            value_set=BOOLEANS,
        ))

    def __bool__(self) -> bool:
        raise TypeError(
            "An Event has no single truth value. "
            "Combine events using '&', '|', and '~', "
            "with parentheses around each comparison."
        )
