from dataclasses import dataclass

from src.problab.distributions._config import DEF_ALPHA
from src.problab.probability.intervals import ConfidenceInterval
from src.problab.statistics import clopper_pearson
from src.problab.validation._common import _validate_alpha
from src.problab.validation._decorator import _validate_parameters
from src.problab.validation.probability._results import _validate_probability_result_configuration


@dataclass(frozen=True)
class ProbabilityResult:
    value: float
    num_successes: int
    num_unconditioned_samples: int
    num_conditioned_samples: int | None = None

    def __post_init__(self) -> None:
        _validate_probability_result_configuration(
            value=self.value,
            num_successes=self.num_successes,
            num_unconditioned_samples=self.num_unconditioned_samples,
            num_conditioned_samples=self.num_conditioned_samples,
        )

    def __str__(self):
        return str(self.value)

    @property
    def num_samples(self) -> int:
        if self.num_conditioned_samples is None:
            return self.num_unconditioned_samples
        return self.num_conditioned_samples

    @_validate_parameters(alpha=_validate_alpha)
    def confidence_interval(self,
                            alpha: float = DEF_ALPHA
                            ) -> ConfidenceInterval:

        return clopper_pearson.confidence_interval(
            num_samples=self.num_samples,
            num_successes=self.num_successes,
            alpha=alpha,
        )
