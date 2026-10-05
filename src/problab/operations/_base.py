import dataclasses
from typing import Callable
import numpy as np


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Operation:
    operation: Callable
    name_func: Callable[..., str]
    supported_input_types: tuple[tuple[type[np.generic], ...], ...] | None = None


_NUMERIC_INPUT_TYPES = ((np.number,),)
_REAL_NUMERIC_INPUT_TYPES = ((np.integer, np.floating),)
_BOOLEAN_INPUT_TYPES = ((np.bool_,),)
