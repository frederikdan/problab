import numpy as np
from scipy.stats import beta

from problab.probability.intervals import ConfidenceInterval
from problab.validation._common import _validate_alpha
from problab.validation._decorator import _validate_parameters
from problab.validation.statistics._clopper_pearson import _validate_confidence_interval_num_samples, \
    _validate_num_successes, _validate_confidence_interval_configuration


@_validate_parameters(
    num_samples=_validate_confidence_interval_num_samples,
    num_successes=_validate_num_successes,
    alpha=_validate_alpha,
)
def _confidence_interval(
        num_samples: int,
        num_successes: int,
        alpha: float,
) -> ConfidenceInterval:

    _validate_confidence_interval_configuration(
        num_samples=num_samples,
        num_successes=num_successes,
    )

    n = num_samples
    k = num_successes

    if n == 0:
        return ConfidenceInterval(np.nan, np.nan, alpha)

    p_L = 0.0 if k == 0 else beta.ppf(
        q=alpha / 2,
        a=k,
        b=n - k + 1,
    )

    p_U = 1.0 if k == n else beta.ppf(
        q=1 - alpha / 2,
        a=k + 1,
        b=n - k,
    )

    return ConfidenceInterval(p_L, p_U, alpha)
