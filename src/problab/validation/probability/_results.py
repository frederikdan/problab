from numbers import Real
from math import isfinite, isnan

import numpy as np


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"'{name}' must be an integer.")

    if value < 0:
        raise ValueError(f"'{name}' must be non-negative.")


def _validate_probability_result_configuration(
        value: Real,
        num_successes: int,
        num_unconditioned_samples: int,
        num_conditioned_samples: int | None,
        ) -> None:

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("'value' must be a real number.")

    _validate_non_negative_integer(num_successes, "num_successes")
    _validate_non_negative_integer(num_unconditioned_samples, "num_unconditioned_samples")

    if num_unconditioned_samples < 1:
        raise ValueError("'num_unconditioned_samples' must be at least 1.")

    if num_conditioned_samples is not None:
        _validate_non_negative_integer(num_conditioned_samples, "num_conditioned_samples")

        if num_conditioned_samples > num_unconditioned_samples:
            raise ValueError("'num_conditioned_samples' must not exceed 'num_unconditioned_samples'.")

    num_samples = num_unconditioned_samples if num_conditioned_samples is None else num_conditioned_samples

    if num_successes > num_samples:
        raise ValueError("'num_successes' must not exceed the number of samples.")

    if num_samples == 0:
        if not isnan(value):
            raise ValueError("'value' must be NaN when there are no conditioned samples.")
        return

    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("'value' must be between 0 and 1.")
