from dataclasses import dataclass

from problab.validation.probability._intervals import _validate_confidence_interval_configuration, \
    _validate_probability_interval_configuration


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float
    alpha: float

    def __post_init__(self) -> None:
        _validate_confidence_interval_configuration(
            lower=self.lower,
            upper=self.upper,
            alpha=self.alpha,
        )

@dataclass(frozen=True)
class ProbabilityInterval():
    lower: float
    upper: float
    alpha: float
    is_estimate: bool

    def __post_init__(self) -> None:
        _validate_probability_interval_configuration(
            lower=self.lower,
            upper=self.upper,
            alpha=self.alpha,
            is_estimate=self.is_estimate,
        )

    @property
    def probability(self) -> float:
        return 1 - self.alpha

    def __str__(self) -> str:
        return f"[{self.lower}, {self.upper}]"
