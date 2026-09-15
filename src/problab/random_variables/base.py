from __future__ import annotations

from itertools import count
from numbers import Real, Complex
from typing import Callable
import numpy as np
import sympy as sp

from problab.distributions._config import DEF_NUM_SAMPLES, DEF_ALPHA
from problab.distributions.base import Distribution
from problab._events import _Event
from problab._operations import _ADD, _SUBTRACT, _MULTIPLY, _MODULO, _LT, _LTE, _GT, _GTE, _EQ, _NEQ, _POWER, \
    _ArithmeticOperation, _NEGATIVE, _ABS, _DIVIDE, _ComparisonOperation
from problab.probability.intervals import ProbabilityInterval, ConfidenceInterval
from problab.random_variables._config import DEF_MAX_GRAPH_SIZE
from problab.random_variables._context import _RealizationContext
from problab.random_variables.graph import NodeGraph
from problab.random_variables.nodes import _DistributionNode, _Node, _ConstantNode, _OperationNode
from problab.statistics._quantiles import _quantile_confidence_interval
from problab.validation._common import _validate_num_samples, _validate_rng, _validate_alpha, _validate_q, \
    _validate_max_size, _validate_validate
from problab.validation._decorator import _validate_parameters
from problab.validation.random_variables._base import _validate_distribution, _validate_name, \
    _validate_interval_bound, _validate_closed, _validate_target_set, _validate_function, _validate_others, \
    _validate_value_set, _validate_function_name, _validate_vectorized
from problab.value_sets._utils import is_known_subset
from problab.value_sets.base import ValueSet
from problab.value_sets.sets import COMPLEXES, REALS, BOOLEANS


class RandomVariable:

    _count = count()

    @_validate_parameters(
        distribution=_validate_distribution,
        name=_validate_name,
    )
    def __init__(self,
                 distribution: Distribution,
                 name: str | None = None,
                 ) -> None:

        self._distribution = distribution

        if name is None: self._name = "RV_" + str(next(RandomVariable._count) + 1)
        else: self._name = name

        self._node = _DistributionNode(distribution, rv_name=self._name)

    @classmethod
    def _from_node(cls,
                   node: _Node,
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

    @_validate_parameters(max_size=_validate_max_size)
    def plot_dependencies(self, max_size=DEF_MAX_GRAPH_SIZE) -> None:
        NodeGraph(self._node, max_size=max_size).plot()

    def realize(self):
        return self.sample()

    @_validate_parameters(
        num_samples=_validate_num_samples,
        rng=_validate_rng,
        validate=_validate_validate
    )
    def sample(self,
               num_samples: int = 1,
               rng: np.random.Generator | None = None,
               validate: bool = False,
               ) -> np.ndarray:

        num_samples = int(num_samples)

        return _RealizationContext(
            root_node=self._node,
            num_samples=num_samples,
            rng=rng,
            validate=validate
        ).evaluate(self._node)

    def _is_real_or_complex(self) -> bool:
        return is_known_subset(self._node.value_set, COMPLEXES)

    def _binary_operation(self,
                          other: RandomVariable | Complex,
                          operation: _ArithmeticOperation,
                          reverse: bool = False
                          ) -> RandomVariable:

        if not isinstance(other, (RandomVariable, Complex)):
            return NotImplemented

        if not is_known_subset(
                self._node.value_set,
                operation.valid_input_value_set,
        ):
            return NotImplemented

        if isinstance(other, RandomVariable):
            if not is_known_subset(
                    other._node.value_set,
                    operation.valid_input_value_set,
            ):
                return NotImplemented
        else:
            other = RandomVariable._from_node(
                _ConstantNode(other)
            )

        left_node, right_node = (
            (other._node, self._node)
            if reverse
            else (self._node, other._node)
        )

        value_set = operation.infer_output_value_set(
            left_node.value_set,
            right_node.value_set,
        )

        operation_function = operation.operation
        if operation is _POWER and is_known_subset(value_set, REALS):
            operation_function = np.power

        node_name = operation.name_func(left_node.name, right_node.name)

        return RandomVariable._from_node(
            _OperationNode(
                operation=operation_function,
                inputs=(left_node, right_node),
                name=node_name,
                value_set=value_set,
            )
        )

    def _unary_operation(self,
                         operation: _ArithmeticOperation
                         ) -> RandomVariable:

        if not is_known_subset(
                self._node.value_set,
                operation.valid_input_value_set,
        ):
            raise TypeError(f"The unary operation {operation.name_func('x')} requires a random variable "
                            f"whose value set is a known subset of {operation.valid_input_value_set.sympy_set}.")

        value_set = operation.infer_output_value_set(
            self._node.value_set
        )

        node_name = operation.name_func(self._node.name)

        return RandomVariable._from_node(
            _OperationNode(
                operation=operation.operation,
                inputs=(self._node,),
                name=node_name,
                value_set=value_set,
            )
        )

    @_validate_parameters(
        alpha=_validate_alpha,
        num_samples=_validate_num_samples,
        rng=_validate_rng,
    )
    def interval(self,
                 alpha: float,
                 num_samples: int = DEF_NUM_SAMPLES,
                 rng: np.random.Generator | None = None
                 ) -> ProbabilityInterval:

        if not is_known_subset(self._node.value_set, REALS):
            raise TypeError("Probability intervals are only defined for real-valued random variables." )

        samples = self.sample(
            num_samples=num_samples,
            rng=rng,
        )

        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        lower = float(np.quantile(samples, lower_quantile, method="inverted_cdf"))
        upper = float(np.quantile(samples, upper_quantile, method="inverted_cdf"))

        return ProbabilityInterval(
            lower=lower,
            upper=upper,
            alpha=alpha,
            is_estimate=True
        )

    @_validate_parameters(
        lower_bound=_validate_interval_bound,
        upper_bound=_validate_interval_bound,
        closed=_validate_closed,
    )
    def is_in_interval(self,
              lower_bound: Real,
              upper_bound: Real,
              closed: str = "both"
              ) -> _Event:

        if not is_known_subset(self._node.value_set, REALS):
            raise TypeError("is_in_interval() requires a real-valued random variable.")

        if not lower_bound <= upper_bound:
            raise ValueError("Lower bound must be less than or equal to upper bound.")

        if closed == "both":
            return (lower_bound <= self) & (self <= upper_bound)

        if closed == "left":
            return (lower_bound <= self) & (self < upper_bound)

        if closed == "right":
            return (lower_bound < self) & (self <= upper_bound)

        # closed == "none"
        return (lower_bound < self) & (self < upper_bound)

    @_validate_parameters(
        target_set=_validate_target_set,
    )
    def is_in(self,
              target_set: sp.Set | tuple[Real, Real] | list[Real],
              ) -> _Event:

        if isinstance(target_set, (tuple, list)):

            lower_bound, upper_bound = target_set
            closed = "none" if isinstance(target_set, tuple) else "both"

            return self.is_in_interval(
                lower_bound,
                upper_bound,
                closed=closed,
            )

        def contains(value) -> bool:
            result = target_set.contains(value)

            if result is sp.true:
                return True

            if result is sp.false:
                return False

            raise ValueError(f"Could not determine whether {value!r} belongs to {target_set}.")

        def operation(samples: np.ndarray) -> np.ndarray:
            return np.fromiter(
                (contains(value) for value in samples),
                dtype=bool,
                count=len(samples),
            )

        return _Event(
            _OperationNode(
                operation=operation,
                inputs=(self._node,),
                name=f"{{{self._node.name} in {target_set}}}",
                value_set=BOOLEANS,
            )
        )

    @_validate_parameters(
        function=_validate_function,
        others=_validate_others,
        value_set=_validate_value_set,
        function_name=_validate_function_name,
        vectorized=_validate_vectorized,
    )
    def apply(self,
              function: Callable,
              *others: RandomVariable,
              value_set: ValueSet,
              function_name: str = "f",
              vectorized: bool = False
              ) -> RandomVariable:

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
            _OperationNode(
                operation=operation,
                inputs=inputs,
                name=node_name,
                value_set=value_set,
            )
        )

    def __add__(self, other):
        return self._binary_operation(other, _ADD)

    def __radd__(self, other):
        return self._binary_operation(other, _ADD)

    def __sub__(self, other):
        return self._binary_operation(other, _SUBTRACT)

    def __rsub__(self, other):
        return self._binary_operation(other, _SUBTRACT, reverse=True)

    def __mul__(self, other):
        return self._binary_operation(other, _MULTIPLY)

    def __rmul__(self, other):
        return self._binary_operation(other, _MULTIPLY)

    def __truediv__(self, other):
        return self._binary_operation(other, _DIVIDE)

    def __rtruediv__(self, other):
        return self._binary_operation(other, _DIVIDE, reverse=True)

    def __mod__(self, other: RandomVariable | Real) -> RandomVariable:
        return self._binary_operation(other, _MODULO)

    def __rmod__(self, other: RandomVariable | Real) -> RandomVariable:
        return self._binary_operation(other, _MODULO, reverse=True)

    def __pow__(self, other: RandomVariable | Complex) -> RandomVariable:
        return self._binary_operation(other, _POWER)

    def __rpow__(self, other: RandomVariable | Complex) -> RandomVariable:
        return self._binary_operation(other, _POWER, reverse=True)

    def __neg__(self) -> RandomVariable:
        return self._unary_operation(_NEGATIVE)

    def __abs__(self) -> RandomVariable:
        return self._unary_operation(_ABS)

    def _inequality_comparison(self, operator: _ComparisonOperation, other: RandomVariable | Real) -> _Event:

        if not is_known_subset(self._node.value_set, REALS):
            return NotImplemented

        if isinstance(other, Real):
            other_node = _ConstantNode(other)

        elif isinstance(other, RandomVariable):
            if not is_known_subset(other._node.value_set, REALS):
                return NotImplemented

            other_node = other._node

        else:
            return NotImplemented

        node_name = operator.name_func(self._node.name, other_node.name)

        return _Event(
            _OperationNode(
                operation=operator.operation,
                inputs=(self._node, other_node),
                name=node_name,
                value_set=BOOLEANS,
            )
        )

    def _equality_comparison(self, operator: _ComparisonOperation, other: RandomVariable | Real) -> _Event:
        if isinstance(other, RandomVariable):
            other_node = other._node
        else:
            other_node = _ConstantNode(other)

        node_name = operator.name_func(self._node.name, other_node.name)

        return _Event(
            _OperationNode(
                operation=operator.operation,
                inputs=(self._node, other_node),
                name=node_name,
                value_set=BOOLEANS,
            )
        )

    def __lt__(self, other: RandomVariable | Real) -> _Event:
        return self._inequality_comparison(_LT, other)

    def __le__(self, other: RandomVariable | Real) -> _Event:
        return self._inequality_comparison(_LTE, other)

    def __gt__(self, other: RandomVariable | Real) -> _Event:
        return self._inequality_comparison(_GT, other)

    def __ge__(self, other: RandomVariable | Real) -> _Event:
        return self._inequality_comparison(_GTE, other)

    def __eq__(self, other: RandomVariable | Real) -> _Event:
        return self._equality_comparison(_EQ, other)

    def __ne__(self, other: RandomVariable | Real) -> _Event:
        return self._equality_comparison(_NEQ, other)

    def __repr__(self):
        return (
            f"RandomVariable("
            f"name={self._name}, "
            f"value_set={self._node.value_set!r}"
            f")"
        )

    def __str__(self):
        return self.name

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

        if not is_known_subset(self._node.value_set, REALS):
            raise TypeError("Quantile confidence intervals require a real-valued random variable.")

        samples = self.sample(
            num_samples=num_samples,
            rng=rng,
        )

        return _quantile_confidence_interval(
            samples=samples,
            q=q,
            alpha=alpha,
        )

