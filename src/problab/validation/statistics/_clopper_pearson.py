import numpy as np


def _validate_confidence_interval_num_samples(value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("'num_samples' must be an integer.")

    if value < 0:
        raise ValueError("'num_samples' must be non-negative.")


def _validate_num_successes(value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("'num_successes' must be an integer.")

    if value < 0:
        raise ValueError("'num_successes' must be non-negative.")


def _validate_confidence_interval_configuration(
        num_samples: int,
        num_successes: int,
        ) -> None:

    if num_successes > num_samples:
        raise ValueError("'num_successes' must be between 0 and 'num_samples'.")