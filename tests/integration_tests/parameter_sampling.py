import unittest
import warnings
from unittest.mock import patch

import numpy as np

from problab import (
    BinomialDistribution, CategoricalDistribution, NormalDistribution,
    PoissonDistribution, RandomVariable,
)
from problab.random_variables._context import _RealizationContext
from problab.value_sets import HomogeneousNumericValueSet
from problab.value_sets.sets import POSITIVE_REALS, REALS, UNIT_INTERVAL


class ParameterSamplingIntegrationTests(unittest.TestCase):
    def parameter(self, values, mathematical_set=POSITIVE_REALS, calls=None):
        values = np.asarray(values, dtype=np.float64)
        source = RandomVariable(CategoricalDistribution([1.], [1.]))

        def realize(_):
            if calls is not None:
                calls.append(1)
            return values.copy()

        return source.apply(
            realize,
            mathematical_value_set=mathematical_set,
            realization_value_set=HomogeneousNumericValueSet(
                REALS.sympy_set, (np.float64,), allows_nan=True,
            ),
            vectorized=True,
        )

    def test_shared_parameter_is_evaluated_once_and_retained_without_filtering(self):
        for order in ((0, 1, 2), (2, 1, 0)):
            calls = []
            parameter = self.parameter([1., -1., 2.], calls=calls)
            normal = RandomVariable(NormalDistribution(0, parameter, parameter_risk_policy="ignore"))
            poisson = RandomVariable(PoissonDistribution(parameter, parameter_risk_policy="ignore"))
            nodes = (normal._node, poisson._node, parameter._node)
            context = _RealizationContext(
                requested_nodes=nodes, num_samples=3, rng=np.random.default_rng(40),
                numerical_error_policy="ignore", validate=True,
            )
            with self.subTest(order=order):
                for index in order:
                    context.evaluate(nodes[index])
                np.testing.assert_array_equal(context.evaluate(parameter._node), [1., -1., 2.])
                for node in nodes[:2]:
                    np.testing.assert_array_equal(np.isnan(context.evaluate(node)), [False, True, False])
                self.assertEqual(calls, [1])

    def test_nan_markers_propagate_through_a_dependent_normal_parameter(self):
        parameter = self.parameter([0., np.nan, 2.], REALS)
        first = RandomVariable(NormalDistribution(parameter, 1, parameter_risk_policy="ignore"))
        second = RandomVariable(NormalDistribution(first, 1, parameter_risk_policy="ignore"))
        samples = second.sample(3, numerical_error_policy="ignore", validate=True)
        np.testing.assert_array_equal(np.isnan(samples), [False, True, False])
        self.assertTrue(np.isfinite(samples[[0, 2]]).all())

    def test_runtime_policy_is_independent_of_construction_policy_and_output_validation(self):
        for construction_policy in ("warn", "ignore"):
            parameter = self.parameter([1., -1.])
            with warnings.catch_warnings(record=True) as construction_warnings:
                warnings.simplefilter("always")
                distribution = NormalDistribution(0, parameter, parameter_risk_policy=construction_policy)
            self.assertEqual(len(construction_warnings), int(construction_policy == "warn"))
            variable = RandomVariable(distribution)
            for validate in (False, True):
                for runtime_policy in ("warn", "ignore", "raise"):
                    with self.subTest(construction=construction_policy, runtime=runtime_policy, validate=validate):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            if runtime_policy == "raise":
                                with self.assertRaisesRegex(ValueError, "std"):
                                    variable.sample(2, numerical_error_policy=runtime_policy, validate=validate)
                            else:
                                samples = variable.sample(2, numerical_error_policy=runtime_policy, validate=validate)
                                np.testing.assert_array_equal(np.isnan(samples), [False, True])
                            self.assertEqual(len(caught), int(runtime_policy == "warn"))

    def test_single_invalid_sample_and_public_distribution_default_policy(self):
        parameter = self.parameter([0.])
        distribution = NormalDistribution(0, parameter, parameter_risk_policy="ignore")
        with self.assertWarnsRegex(RuntimeWarning, "1 of 1"):
            samples = distribution.sample(num_samples=1, validate=True)
        np.testing.assert_array_equal(samples, [np.nan])

    def test_valid_batches_under_risky_declarations_remain_integer_and_reproducible(self):
        p = self.parameter([0.2, 0.8, 0.5], UNIT_INTERVAL)
        distribution = BinomialDistribution(5, p, parameter_risk_policy="ignore")
        variable = RandomVariable(distribution)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = variable.sample(3, rng=np.random.default_rng(71), validate=True)
            second = variable.sample(3, rng=np.random.default_rng(71), validate=True)
        self.assertEqual(caught, [])
        self.assertEqual(first.dtype.kind, "i")
        np.testing.assert_array_equal(first, second)

    def test_output_validation_does_not_excuse_a_parameter_lying_about_support(self):
        source = RandomVariable(CategoricalDistribution([1.], [1.]))
        parameter = source.apply(
            lambda x: np.array([0.]), vectorized=True,
            mathematical_value_set=POSITIVE_REALS,
            realization_value_set=HomogeneousNumericValueSet(POSITIVE_REALS.sympy_set, (np.float64,)),
        )
        distribution = NormalDistribution(0, parameter, parameter_risk_policy="raise")
        with patch.object(distribution, "_sample") as backend:
            with self.assertRaisesRegex(ValueError, "outside the target set"):
                RandomVariable(distribution).sample(validate=True, numerical_error_policy="ignore")
            backend.assert_not_called()
