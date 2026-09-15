from abc import ABC, abstractmethod
from typing import TypeVar, Any
import numpy as np
from enum import Enum
from collections.abc import Callable
from numbers import Real, Complex
from functools import partial

from src.problab.distributions._config import DEF_NUM_SAMPLES, DEF_ALPHA, DEF_DISTRIBUTION_SYMBOL_NAME
from src.problab.probability.intervals import ConfidenceInterval
from src.problab.random_variables.context import RealizationContext
from src.problab.random_variables.nodes import Node, ConstantNode, DistributionNode
from src.problab.statistics.quantiles import quantile_confidence_interval, QuantileMethod
from src.problab.validation._common import _validate_q, _validate_alpha, _validate_num_samples, _validate_rng, \
    _validate_enum
from src.problab.validation._decorator import _validate_parameters
from src.problab.validation.distributions._base import _validate_cdf_input, _validate_ppf_input, _validate_quantile_method
from src.problab.value_sets._utils import is_known_subset
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import REALS

T = TypeVar('T')


class Mode(Enum):
    AUTO = 1
    EXACT = 2
    MONTE_CARLO = 3


class Distribution(ABC):


    @property
    @abstractmethod
    def value_set(self) -> ValueSet:
        ...

    def sample(self, context: RealizationContext | None = None) -> np.ndarray:

        if context is None:
            root = DistributionNode(self, rv_name=self.name)
            context = RealizationContext(root_node=root)
            return context.evaluate(root)

        parameter_values = tuple(context.evaluate(parameter) for parameter in self.parameters)

        return self._sample(
            *parameter_values,
            num_samples=context.num_samples,
            rng=context.rng,
        )

    @abstractmethod
    def _sample(self,
                *parameters: np.ndarray,
                num_samples: int,
                rng: np.random.Generator
                ) -> np.ndarray:
        ...


    def __init__(self,
                 parameters: tuple[Node, ...] | None = None,
                 symbol: str = DEF_DISTRIBUTION_SYMBOL_NAME
                 ) -> None:
        self._parameters = parameters if parameters is not None else ()
        self._symbol = symbol


    @property
    def symbol(self) -> str:
        return self._symbol


    @property
    def name(self) -> str:
        return f"{self.symbol}({", ".join(parameter.name for parameter in self.parameters)})"


    @property
    def parameters(self) -> tuple[Node, ...]:
        return self._parameters


    @property
    def node_dependencies(self) -> set[Node]:
        dependencies = set(
            x for x in self.parameters
            if not isinstance(x, ConstantNode)
        )

        return dependencies

    @_validate_parameters(
        q=_validate_q,
        alpha=_validate_alpha,
        num_samples=_validate_num_samples,
        rng=_validate_rng,
    )
    def quantile_confidence_interval(self,
                                     q: float,
                                     alpha: float = DEF_ALPHA,
                                     num_samples: int = DEF_NUM_SAMPLES,
                                     rng: np.random.Generator | None = None
                                     ) -> ConfidenceInterval:

        if not is_known_subset(self.value_set, REALS):
            raise TypeError("Quantile confidence intervals require a real-valued distribution.")

        root_node = DistributionNode(self, rv_name=self.name)
        context = RealizationContext(
            root_node=root_node,
            num_samples=num_samples,
            rng=rng,
        )
        samples = context.evaluate(root_node)

        return quantile_confidence_interval(
            samples=samples,
            q=q,
            alpha=alpha,
        )


    def _monte_carlo(self,
                     operation: Callable[[np.ndarray], Complex | np.ndarray],
                     num_samples: int = DEF_NUM_SAMPLES,
                     rng: np.random.Generator | None = None
                     ) -> float | complex | np.ndarray:

        root_node = DistributionNode(self, rv_name=self.name)
        context = RealizationContext(
            root_node=root_node,
            num_samples=num_samples,
            rng=rng,
        )
        samples = context.evaluate(root_node)

        result = operation(samples)

        if np.ndim(result) == 0:
            return complex(result) if np.iscomplexobj(result) else float(result)

        return result

    @_validate_parameters(
        mode=partial(_validate_enum, enum_type=Mode, name="mode"),
        num_samples=_validate_num_samples,
        rng=_validate_rng,
    )
    def mean(self,
             mode: Mode = Mode.AUTO,
             num_samples: int = DEF_NUM_SAMPLES,
             rng: np.random.Generator | None = None
             ) -> float | complex:

        if mode == Mode.MONTE_CARLO:
            return self._monte_carlo(operation=np.mean, num_samples=num_samples, rng=rng)

        exact_mean = self._mean_exact()

        if mode == Mode.EXACT:
            if exact_mean is None:
                raise NotImplementedError("Exact mean is not available for this distribution.")
            return exact_mean

        assert mode is Mode.AUTO
        return exact_mean if exact_mean is not None else self._monte_carlo(
            operation=np.mean,
            num_samples=num_samples,
            rng=rng,
        )

    @_validate_parameters(
        mode=partial(_validate_enum, enum_type=Mode, name="mode"),
        num_samples=_validate_num_samples,
        rng=_validate_rng,
    )
    def variance(self,
                 mode: Mode = Mode.AUTO,
                 num_samples: int = DEF_NUM_SAMPLES,
                 rng: np.random.Generator | None = None
                 ) -> float:

        if mode == Mode.MONTE_CARLO:
            return self._monte_carlo(operation=np.var, num_samples=num_samples, rng=rng)

        exact_variance = self._variance_exact()

        if mode == Mode.EXACT:
            if exact_variance is None:
                raise NotImplementedError("Exact variance is not available for this distribution.")
            return exact_variance

        assert mode is Mode.AUTO
        return exact_variance if exact_variance is not None else self._monte_carlo(
            operation=np.var,
            num_samples=num_samples,
            rng=rng,
        )

    @_validate_parameters(
        mode=partial(_validate_enum, enum_type=Mode, name="mode"),
        num_samples=_validate_num_samples,
        rng=_validate_rng,
    )
    def std(self,
            mode: Mode = Mode.AUTO,
            num_samples: int = DEF_NUM_SAMPLES,
            rng: np.random.Generator | None = None
            ) -> float:

        if mode == Mode.MONTE_CARLO:
            return self._monte_carlo(operation=np.std, num_samples=num_samples, rng=rng)

        exact_variance = self._variance_exact()
        exact_std = (
            None if exact_variance is None
            else float(np.sqrt(exact_variance))
        )

        if mode == Mode.EXACT:
            if exact_std is None:
                raise NotImplementedError("Exact standard deviation is not available for this distribution.")
            return exact_std

        assert mode is Mode.AUTO
        return exact_std if exact_std is not None else self._monte_carlo(
            operation=np.std,
            num_samples=num_samples,
            rng=rng,
        )

    @_validate_parameters(
        x=_validate_cdf_input,
        mode=partial(_validate_enum, enum_type=Mode, name="mode"),
        num_samples=_validate_num_samples,
        rng=_validate_rng,
    )
    def cdf(self,
            x: Real | np.ndarray,
            mode: Mode = Mode.AUTO,
            num_samples: int = DEF_NUM_SAMPLES,
            rng: np.random.Generator | None = None
            ) -> float | np.ndarray:

        def monte_carlo() -> float | np.ndarray:

            if isinstance(x, np.ndarray): # if statement instead
                operation =  lambda samples: (
                        np.searchsorted(
                            np.sort(samples),
                            x,
                            side="right",
                        ) / len(samples)
                )
            else:
                operation = lambda samples: np.mean(samples <= x)

            return self._monte_carlo(
                operation=operation,
                num_samples=num_samples,
                rng=rng,
            )

        if mode == Mode.MONTE_CARLO:
            return monte_carlo()

        exact_cdf = self._cdf_exact(x)

        if mode == Mode.EXACT:
            if exact_cdf is None:
                raise NotImplementedError("Exact CDF is not available for this distribution.")
            return exact_cdf

        assert mode is Mode.AUTO
        return exact_cdf if exact_cdf is not None else monte_carlo()

    @_validate_parameters(
        q=_validate_ppf_input,
        mode=partial(_validate_enum, enum_type=Mode, name="mode"),
        num_samples=_validate_num_samples,
        rng=_validate_rng,
        quantile_method=_validate_quantile_method
    )
    def ppf(self,
            q: Real | np.ndarray,
            mode: Mode = Mode.AUTO,
            num_samples: int = DEF_NUM_SAMPLES,
            rng: np.random.Generator | None = None,
            quantile_method: QuantileMethod = "inverted_cdf"
            ) -> float | np.ndarray:

        def monte_carlo() -> float | np.ndarray:
            return self._monte_carlo(
                operation=lambda samples: np.quantile(
                    samples,
                    q,
                    method=quantile_method,
                ),
                num_samples=num_samples,
                rng=rng,
            )

        if mode == Mode.MONTE_CARLO:
            return monte_carlo()

        exact_ppf = self._ppf_exact(q)

        if mode == Mode.EXACT:
            if exact_ppf is None:
                raise NotImplementedError("Exact PPF is not available for this distribution.")
            return exact_ppf

        assert mode is Mode.AUTO
        return exact_ppf if exact_ppf is not None else monte_carlo()

    def _mean_exact(self) -> float | Complex | None:
       return None

    def _variance_exact(self) -> float | None:
        return None

    def _cdf_exact(self,
                   x: Real | np.ndarray
                   ) -> float | np.ndarray | None:
        return None

    def _ppf_exact(self,
                   q: Real | np.ndarray
                   ) -> float | np.ndarray | None:
        return None


class ContinuousDistribution(Distribution):


    @abstractmethod
    def pdf(self, x: Real | np.ndarray) -> float | np.ndarray:
        ...


class DiscreteDistribution(Distribution):

    @abstractmethod
    def pmf(self, x: Real | np.ndarray) -> float | np.ndarray:
        ...

