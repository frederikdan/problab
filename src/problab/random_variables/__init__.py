"""Random variables and their public dependency graph."""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from .base import RandomVariable
    from .graph import NodeGraph

__all__ = ["RandomVariable", "NodeGraph"]


def __getattr__(name: str):
    # Distribution evaluation imports this package before Distribution is defined.
    modules = {"RandomVariable": ".base", "NodeGraph": ".graph"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(modules[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
