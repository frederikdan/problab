"""Probability evaluation, results, and intervals."""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from .intervals import ConfidenceInterval, ProbabilityInterval
    from .probability import P
    from .results import ProbabilityResult

__all__ = ["P", "ProbabilityResult", "ConfidenceInterval", "ProbabilityInterval"]


def __getattr__(name: str):
    # Intervals are imported while Distribution and RandomVariable initialize.
    modules = {
        "P": ".probability",
        "ProbabilityResult": ".results",
        "ConfidenceInterval": ".intervals",
        "ProbabilityInterval": ".intervals",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(modules[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
