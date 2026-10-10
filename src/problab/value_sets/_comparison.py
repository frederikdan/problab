from typing import Any

import numpy as np


def _objects_equal(left: Any, right: Any) -> bool:
    # Remember, when at some point you write comments/descriptions,
    # to be aware that the broad exception handling here is intentional.

    try:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return bool(np.array_equal(left, right))

        result = left == right
    except Exception:
        return False

    if isinstance(result, (bool, np.bool_)):
        return bool(result)

    try:
        return bool(np.all(result))
    except (TypeError, ValueError):
        return False
