"""Mathematical functions accepting scalars or random variables."""

from .basic import absolute, ceil, floor, hypot, sign, sqrt
from .exponential import exp, expm1, log, log1p, log2, log10, logaddexp
from .trigonometric import (
    arccos, arccosh, arcsin, arcsinh, arctan, arctanh,
    cos, cosh, sin, sinh, tan, tanh,
)

__all__ = [
    "absolute", "ceil", "floor", "hypot", "sign", "sqrt",
    "exp", "expm1", "log", "log1p", "log2", "log10", "logaddexp",
    "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh",
    "cos", "cosh", "sin", "sinh", "tan", "tanh",
]
