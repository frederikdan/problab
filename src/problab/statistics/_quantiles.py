import numpy as np
from scipy.stats import binom
from typing import Literal, TypeAlias

from problab._utils import _float_if_fraction
from problab.distributions._config import DEF_ALPHA
from problab.probability.intervals import ConfidenceInterval

QuantileMethod: TypeAlias = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]


def _quantile_confidence_interval(samples: np.ndarray,
                                  q: float,
                                  alpha: float = DEF_ALPHA
                                  ) -> ConfidenceInterval:

    q = _float_if_fraction(q)
    alpha = _float_if_fraction(alpha)

    if not 0 < q < 1:
        raise ValueError("'q' must be between 0 and 1.")

    if not 0 < alpha < 1:
        raise ValueError("'alpha' must be between 0 and 1.")

    n = len(samples)

    if n == 0:
        raise ValueError("'samples' must contain at least one value.")

    n_min = int(np.ceil(
        np.log(alpha / 2) / np.log(max(q, 1 - q))
    ))

    if n < n_min:
        raise ValueError(
            f"Insufficient samples. At least {n_min} samples are required "
            f"for q={q} and alpha={alpha}."
        )

    lower_left = 1
    lower_right = n - 1

    while lower_left < lower_right:
        lower_midpoint = (lower_left + lower_right + 1) // 2

        if binom.cdf(lower_midpoint - 1, n, q) <= alpha / 2:
            lower_left = lower_midpoint
        else:
            lower_right = lower_midpoint - 1

    upper_left = 1
    upper_right = n - 1

    while upper_left < upper_right:
        upper_midpoint = (upper_left + upper_right) // 2

        # Computing the upper tail directly avoids loss of precision from 1 - CDF.
        if binom.sf(upper_midpoint, n, q) <= alpha / 2:
            upper_right = upper_midpoint
        else:
            upper_left = upper_midpoint + 1

    k_L = lower_left
    k_U = upper_right

    sorted_samples = np.sort(samples)

    lower = float(sorted_samples[k_L - 1])
    upper = float(sorted_samples[k_U])

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        alpha=alpha,
    )
