from __future__ import annotations

import sympy as sp

from src.problab.operations import AND, INVERT, OR
from src.problab.random_variables.nodes import Node, OperationNode


class Event:

    def __init__(self, node: Node):
        self._node: Node = node

    @property
    def name(self) -> str:
        return self._node.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Event({self._node!r})"

    def __and__(self, other: Event) -> Event:

        if not isinstance(other, Event):
            return NotImplemented

        node_name = AND.name_func(self._node.name, other.name)

        return Event(OperationNode(
            operation=AND.operation,
            inputs=(self._node, other._node),
            name=node_name,
            value_set=sp.FiniteSet(False, True),
        ))

    def __or__(self, other: Event) -> Event:

        if not isinstance(other, Event):
            return NotImplemented

        node_name = OR.name_func(self._node.name, other.name)

        return Event(OperationNode(
            operation=OR.operation,
            inputs=(self._node, other._node),
            name=node_name,
            value_set=sp.FiniteSet(False, True),
        ))

    def __invert__(self) -> Event:

        node_name = INVERT.name_func(self._node.name)

        return Event(OperationNode(
            operation=INVERT.operation,
            inputs=(self._node,),
            name=node_name,
            value_set=sp.FiniteSet(False, True),
        ))

    def __bool__(self) -> bool:
        raise TypeError(
            "An Event has no single truth value. "
            "Combine events using '&', '|', and '~', "
            "with parentheses around each comparison."
        )
