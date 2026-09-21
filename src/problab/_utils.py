import numpy as np

from problab._config import RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE


def _is_close(left, right) -> bool:
    return bool(np.isclose(
        left,
        right,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    ))