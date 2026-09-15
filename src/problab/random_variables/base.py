from __future__ import annotations

from itertools import count
from numbers import Real, Complex
from typing import Callable
import numpy as np
import sympy as sp

from src.problab.distributions._config import DEF_NUM_SAMPLES, DEF_ALPHA
from src.problab.distributions.base import Distribution
from src.problab.events import Event
from src.problab.operations import ADD, SUBTRACT, MULTIPLY, MODULO, LT, LTE, GT, GTE, EQ, NEQ, POWER, \
    ArithmeticOperation, NEGATIVE, ABS, DIVIDE, ComparisonOperation
from src.problab.probability.intervals import ProbabilityInterval, ConfidenceInterval
from src.problab.random_variables._config import DEF_MAX_GRAPH_SIZE
from src.problab.random_variables.context import RealizationContext
from src.problab.random_variables.graph import NodeGraph
from src.problab.random_variables.nodes import DistributionNode, Node, ConstantNode, OperationNode
from src.problab.statistics.quantiles import quantile_confidence_interval
from src.problab.value_sets._utils import is_known_subset, is_in
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import COMPLEXES, REALS, BOOLEANS


class RandomVariable:

    _count = count()

    def __init__(self,
                 distribution: Distribution,
                 name: str | None = None
                 ) -> None:

        self._distribution = distribution

        if name is None: self._name = "RV_" + str(next(RandomVariable._count) + 1)
        else: self._name = name

        self._node = DistributionNode(distribution, rv_name=self._name)

    @classmethod
    def _from_node(cls,
                   node: Node,
                   name: str | None = None
                   ) -> RandomVariable:

        rv = cls.__new__(cls)

        rv._node = node
        rv._name = name if name is not None else "RV_" + str(next(cls._count) + 1)

        return rv

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependency_graph(self) -> NodeGraph:
        return NodeGraph(self._node, max_size=DEF_MAX_GRAPH_SIZE)

    def plot_dependencies(self, max_size=DEF_MAX_GRAPH_SIZE) -> None:
        NodeGraph(self._node, max_size=max_size).plot()

    def realize(self):
        return self.sample()

    def sample(self, num_samples: int = 1,  rng: np.random.Generator | None = None) -> np.ndarray:

        if num_samples < 1:
            raise ValueError("Number of samples must be positive")

        return RealizationContext(root_node=self._node,
                                  num_samples=num_samples,
                                  rng=rng).evaluate(self._node)

    def _is_real_or_complex(self) -> bool:
        return is_known_subset(self._node.value_set, COMPLEXES)

    def _binary_operation(self,
                          other: RandomVariable | Complex,
                          operation: ArithmeticOperation,
                          reverse: bool = False
                          ) -> RandomVariable:

        if not isinstance(other, (RandomVariable, Complex)):
            return NotImplemented

        if not is_known_subset(
                self._node.value_set,
                operation.valid_value_set,
        ):
            return NotImplemented

        if isinstance(other, RandomVariable):
            if not is_known_subset(
                    other._node.value_set,
                    operation.valid_value_set,
            ):
                return NotImplemented
        else:
            other = RandomVariable._from_node(
                ConstantNode(other)
            )

        left_node, right_node = (
            (other._node, self._node)
            if reverse
            else (self._node, other._node)
        )

        value_set = operation.infer_value_set(
            left_node.value_set,
            right_node.value_set,
        )

        node_name = operation.name_func(left_node.name, right_node.name)

        return RandomVariable._from_node(
            OperationNode(
                operation=operation.operation,
                inputs=(left_node, right_node),
                name=node_name,
                value_set=value_set,
            )
        )

    def _unary_operation(self,
                         operation: ArithmeticOperation
                         ) -> RandomVariable:

        if not is_known_subset(
                self._node.value_set,
                operation.valid_value_set,
        ):
            return NotImplemented

        value_set = operation.infer_value_set(
            self._node.value_set
        )

        node_name = operation.name_func(self._node.name)

        return RandomVariable._from_node(
            OperationNode(
                operation=operation.operation,
                inputs=(self._node,),
                name=node_name,
                value_set=value_set,
            )
        )

    def interval(self,
                 alpha: float,
                 num_samples: int = DEF_NUM_SAMPLES,
                 rng: np.random.Generator | None = None
                 ) -> ProbabilityInterval:

        if not is_known_subset(self._node.value_set, REALS):
            raise TypeError("Probability intervals are only defined for real-valued random variables." )

        if not 0 < alpha < 1:
            raise ValueError("'alpha' must be between 0 and 1.")

        samples = self.sample(
            num_samples=num_samples,
            rng=rng,
        )

        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        lower = float(np.quantile(samples, lower_quantile))
        upper = float(np.quantile(samples, upper_quantile))

        return ProbabilityInterval(
            lower=lower,
            upper=upper,
            alpha=alpha,
            is_estimate=True
        )

    def is_in_interval(self,
              lower_bound: Real,
              upper_bound: Real,
              closed: str = "both"
              ) -> Event:

        if not is_known_subset(self._node.value_set, REALS):
            raise TypeError("is_in() requires a real-valued random variable.")

        if not isinstance(lower_bound, Real) or not isinstance(upper_bound, Real):
            raise TypeError("Interval bounds must be real numbers.")

        if not lower_bound <= upper_bound:
            raise ValueError("Lower bound must be less than or equal to upper bound.")

        if closed == "both":
            return (lower_bound <= self) & (self <= upper_bound)

        if closed == "left":
            return (lower_bound <= self) & (self < upper_bound)

        if closed == "right":
            return (lower_bound < self) & (self <= upper_bound)

        if closed == "none":
            return (lower_bound < self) & (self < upper_bound)

        raise ValueError("closed must be one of 'both', 'left', 'right', or 'none'.")


    def is_in(self,
              target_set: sp.Set | tuple[Real, Real] | list[Real],
              ) -> Event:

        if isinstance(target_set, (tuple, list)):
            if len(target_set) != 2:
                raise ValueError("An interval must contain exactly two bounds.")

            lower_bound, upper_bound = target_set
            closed = "none" if isinstance(target_set, tuple) else "both"

            return self.is_in_interval(
                lower_bound,
                upper_bound,
                closed=closed,
            )

        if not isinstance(target_set, sp.Set):
            raise TypeError(
                "'target_set' must be a SymPy set or a tuple/list of two bounds."
            )

        def contains(value) -> bool:
            result = target_set.contains(value)

            if result is sp.true:
                return True

            if result is sp.false:
                return False

            raise ValueError(
                f"Could not determine whether {value!r} belongs to {target_set}."
            )

        def operation(samples: np.ndarray) -> np.ndarray:
            return np.fromiter(
                (contains(value) for value in samples),
                dtype=bool,
                count=len(samples),
            )

        return Event(
            OperationNode(
                operation=operation,
                inputs=(self._node,),
                name=f"{{{self._node.name} in {target_set}}}",
                value_set=BOOLEANS,
            )
        )

    def apply(self,
              function: Callable,
              *others: RandomVariable,
              value_set: ValueSet,
              function_name: str = "f",
              vectorized: bool = False
              ) -> RandomVariable:

        if not callable(function):
            raise TypeError("'function' must be callable.")

        if not all(isinstance(other, RandomVariable) for other in others):
            raise TypeError("'others' must be RandomVariable instances.")

        inputs = (
            self._node,
            *(other._node for other in others),
        )

        if vectorized:
            operation = function
        else:
            operation = lambda *values: np.asarray([
                function(*realization)
                for realization in zip(*values)
            ])

        node_name = f"{function_name}({', '.join(map(str, inputs))})"

        return RandomVariable._from_node(
            OperationNode(
                operation=operation,
                inputs=inputs,
                name=node_name,
                value_set=value_set,
            )
        )

    def __add__(self, other):
        return self._binary_operation(other, ADD)

    def __radd__(self, other):
        return self._binary_operation(other, ADD)

    def __sub__(self, other):
        return self._binary_operation(other, SUBTRACT)

    def __rsub__(self, other):
        return self._binary_operation(other, SUBTRACT, reverse=True)

    def __mul__(self, other):
        return self._binary_operation(other, MULTIPLY)

    def __rmul__(self, other):
        return self._binary_operation(other, MULTIPLY)

    def __truediv__(self, other):
        return self._binary_operation(other, DIVIDE)

    def __rtruediv__(self, other):
        return self._binary_operation(other, DIVIDE, reverse=True)

    def __mod__(self, other: RandomVariable | Real) -> RandomVariable:
        return self._binary_operation(other, MODULO)

    def __rmod__(self, other: RandomVariable | Real) -> RandomVariable:
        return self._binary_operation(other, MODULO, reverse=True)

    def __pow__(self, other: RandomVariable | Complex) -> RandomVariable:
        return self._binary_operation(other, POWER)

    def __rpow__(self, other: RandomVariable | Complex) -> RandomVariable:
        return self._binary_operation(other, POWER, reverse=True)

    def __neg__(self) -> RandomVariable:
        return self._unary_operation(NEGATIVE)

    def __abs__(self) -> RandomVariable:
        return self._unary_operation(ABS)

    def _inequality_comparison(self, operator: ComparisonOperation, other: RandomVariable | Real) -> Event:

        if not is_known_subset(self._node.value_set, REALS):
            return NotImplemented

        if isinstance(other, Real):
            other_node = ConstantNode(other)

        elif isinstance(other, RandomVariable):
            if not is_known_subset(other._node.value_set, REALS):
                return NotImplemented

            other_node = other._node

        else:
            return NotImplemented

        node_name = operator.name_func(self._node.name, other_node.name)

        return Event(
            OperationNode(
                operation=operator.operation,
                inputs=(self._node, other_node),
                name=node_name,
                value_set=BOOLEANS,
            )
        )

    def _equality_comparison(self, operator: ComparisonOperation, other: RandomVariable | Real) -> Event:
        if isinstance(other, RandomVariable):
            other_node = other._node
        else:
            other_node = ConstantNode(other)

        node_name = operator.name_func(self._node.name, other_node.name)

        return Event(
            OperationNode(
                operation=operator.operation,
                inputs=(self._node, other_node),
                name=node_name,
                value_set=BOOLEANS,
            )
        )

    def __lt__(self, other: RandomVariable | Real) -> Event:
        return self._inequality_comparison(LT, other)

    def __le__(self, other: RandomVariable | Real) -> Event:
        return self._inequality_comparison(LTE, other)

    def __gt__(self, other: RandomVariable | Real) -> Event:
        return self._inequality_comparison(GT, other)

    def __ge__(self, other: RandomVariable | Real) -> Event:
        return self._inequality_comparison(GTE, other)

    def __eq__(self, other: RandomVariable | Real) -> Event:
        return self._equality_comparison(EQ, other)

    def __ne__(self, other: RandomVariable | Real) -> Event:
        return self._equality_comparison(NEQ, other)

    def __repr__(self):
        return (
            f"RandomVariable("
            f"name={self._name}, "
            f"value_set={self._node.value_set!r}"
            f")"
        )

    def __str__(self):
        return self.name

    def quantile_confidence_interval(self,
                                     q: float,
                                     alpha: float = DEF_ALPHA,
                                     num_samples: int = DEF_NUM_SAMPLES,
                                     rng: np.random.Generator | None = None
                                     ) -> ConfidenceInterval:

        samples = self.sample(
            num_samples=num_samples,
            rng=rng,
        )

        return quantile_confidence_interval(
            samples=samples,
            q=q,
            alpha=alpha,
        )

