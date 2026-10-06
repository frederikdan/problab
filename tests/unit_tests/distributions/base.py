import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import numpy as np

from problab.operations._function import _FunctionOperation
from problab.operations._statistical import _MEAN, _STD, _VARIANCE
from problab.distributions.base import (
    Distribution,
    Mode,
    _ContinuousDistribution,
    _DiscreteDistribution,
)
from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes import _ConstantNode, _OperationNode
from problab.value_sets.sets import REALS, POSITIVE_REALS, UNKNOWN_VALUE_SET
from problab.value_sets import HomogeneousNumericValueSet, ObjectValueSet
import sympy as sp


class _StubDistribution(Distribution):

    symbol = "Stub"

    def __init__(
        self,
        parameters=(),
        exact_mean=None,
        exact_variance=None,
        exact_cdf=None,
        exact_ppf=None,
    ):
        self._sample_delegate = Mock()
        self._stub_exact_mean = exact_mean
        self._stub_exact_variance = exact_variance
        self._stub_exact_cdf = exact_cdf
        self._stub_exact_ppf = exact_ppf
        super().__init__(parameters=parameters)

    @property
    def value_set(self):
        return REALS

    def _sample(self, *parameters, num_samples, rng):
        return self._sample_delegate(*parameters, num_samples=num_samples, rng=rng)

    def _mean_exact(self):
        return self._stub_exact_mean

    def _variance_exact(self):
        return self._stub_exact_variance

    def _cdf_exact(self, x):
        return self._stub_exact_cdf

    def _ppf_exact(self, q):
        return self._stub_exact_ppf


class DistributionBaseTests(unittest.TestCase):

    def test_parameterless_defaults_have_no_masks_and_guarantee_validity(self):
        distribution = _StubDistribution()
        self.assertEqual(distribution._validate_parameter_realizations(), {})
        self.assertTrue(distribution._parameter_realizations_guaranteed_valid())

    def test_parameter_guarantee_requires_known_numeric_domain_membership(self):
        distribution = _StubDistribution()
        distribution._valid_parameter_sets = {"scale": POSITIVE_REALS}
        for support, expected in (
            (HomogeneousNumericValueSet(sp.Interval(1, 2), (np.float64,)), True),
            (HomogeneousNumericValueSet(sp.Interval(0, 2), (np.float64,)), False),
            (UNKNOWN_VALUE_SET, False),
            (ObjectValueSet((1,)), False),
        ):
            with self.subTest(support=support):
                distribution._parameter_nodes = (SimpleNamespace(_realization_value_set=support),)
                self.assertIs(distribution._parameter_realizations_guaranteed_valid(), expected)

    def test_parameter_guarantee_checks_each_exceptional_permission_independently(self):
        flags = ("allows_positive_infinity", "allows_negative_infinity", "allows_nan")
        distribution = _StubDistribution()
        for emitted in flags:
            support = HomogeneousNumericValueSet(sp.S.Reals, (np.float64,), **{emitted: True})
            distribution._parameter_nodes = (SimpleNamespace(_realization_value_set=support),)
            for allowed in (None, *flags):
                permission = {} if allowed is None else {allowed: True}
                distribution._valid_parameter_sets = {
                    "value": HomogeneousNumericValueSet(sp.S.Reals, (np.float64,), **permission),
                }
                with self.subTest(emitted=emitted, allowed=allowed):
                    self.assertIs(distribution._parameter_realizations_guaranteed_valid(), emitted == allowed)

    def test_parameter_guarantee_uses_mapping_order_and_rejects_missing_entries(self):
        distribution = _StubDistribution(parameters=(-1., 2.))
        distribution._valid_parameter_sets = {"mean": REALS, "scale": POSITIVE_REALS}
        self.assertTrue(distribution._parameter_realizations_guaranteed_valid())
        distribution._valid_parameter_sets = {"scale": POSITIVE_REALS, "mean": REALS}
        self.assertFalse(distribution._parameter_realizations_guaranteed_valid())
        distribution._valid_parameter_sets = {"mean": REALS}
        with self.assertRaises(ValueError):
            distribution._parameter_realizations_guaranteed_valid()

    def masked_distribution(self, policy, masks, output):
        count = len(next(iter(masks.values())))
        distribution = _StubDistribution(parameters=(1., 2.))
        arrays = (np.arange(count, dtype=float), np.arange(count, dtype=float) + 10)
        for values in arrays:
            values.flags.writeable = False
        context = Mock(num_samples=count, rng=sentinel.rng, numerical_error_policy=policy)
        context.evaluate.side_effect = arrays
        distribution._validate_parameter_realizations = Mock(return_value=masks)
        distribution._sample_delegate.return_value = output
        return distribution, context, arrays

    def test_evaluate_all_valid_masks_preserve_arrays_and_backend_result_identity(self):
        expected = np.array([1, 2, 3], dtype=np.int16)
        distribution, context, arrays = self.masked_distribution(
            "raise", {"a": np.ones(3, dtype=bool), "b": np.ones(3, dtype=bool)}, expected,
        )
        result = distribution._evaluate(context)
        self.assertIs(result, expected)
        for actual, original in zip(distribution._sample_delegate.call_args.args, arrays):
            self.assertIs(actual, original)
        self.assertEqual(distribution._sample_delegate.call_args.kwargs, {"num_samples": 3, "rng": sentinel.rng})

    def test_evaluate_combines_masks_without_mutating_shared_parameters_or_masks(self):
        masks = {"a": np.array([True, False, True, True]), "b": np.array([True, True, False, True])}
        for mask in masks.values():
            mask.flags.writeable = False
        for policy in ("warn", "ignore"):
            for dtype in (np.float16, np.float32, np.float64):
                with self.subTest(policy=policy, dtype=dtype):
                    output = np.array([7, 8], dtype=dtype)
                    distribution, context, arrays = self.masked_distribution(policy, masks, output)
                    originals = tuple(array.copy() for array in arrays)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result = distribution._evaluate(context)
                    self.assertEqual(result.dtype, output.dtype)
                    np.testing.assert_array_equal(result, [7, np.nan, np.nan, 8])
                    self.assertEqual(len(caught), int(policy == "warn"))
                    if caught:
                        self.assertIs(caught[0].category, RuntimeWarning)
                        self.assertIn("a, b", str(caught[0].message))
                        self.assertIn("2 of 4", str(caught[0].message))
                    distribution._sample_delegate.assert_called_once()
                    for actual, original, before in zip(distribution._sample_delegate.call_args.args, arrays, originals):
                        np.testing.assert_array_equal(actual, before[[0, 3]])
                        np.testing.assert_array_equal(original, before)
                    self.assertEqual(distribution._sample_delegate.call_args.kwargs, {"num_samples": 2, "rng": sentinel.rng})
        np.testing.assert_array_equal(masks["a"], [True, False, True, True])
        np.testing.assert_array_equal(masks["b"], [True, True, False, True])

    def test_evaluate_raise_reports_parameter_names_and_count_before_backend(self):
        distribution, context, _ = self.masked_distribution(
            "raise", {"first": np.array([False, True]), "second": np.array([True, False])}, None,
        )
        with self.assertRaisesRegex(ValueError, "Stub.*first, second.*2 of 2"):
            distribution._evaluate(context)
        distribution._sample_delegate.assert_not_called()

    def test_evaluate_all_invalid_returns_float64_nan_without_backend(self):
        for policy in ("warn", "ignore"):
            with self.subTest(policy=policy):
                distribution, context, _ = self.masked_distribution(policy, {"a": np.zeros(2, dtype=bool)}, None)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    samples = distribution._evaluate(context)
                self.assertEqual(samples.dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(samples, [np.nan, np.nan])
                self.assertEqual(len(caught), int(policy == "warn"))
                distribution._sample_delegate.assert_not_called()

    def test_evaluate_integer_nan_storage_preserves_exact_boundary_values(self):
        cases = (
            (np.array([0, 127], dtype=np.int8), np.float64),
            (np.array([-(2**53), 2**53], dtype=np.int64), np.float64),
            (np.array([0, 2**53], dtype=np.uint64), np.float64),
            (np.array([-(2**53) - 1, 1], dtype=np.int64), object),
            (np.array([2**53 + 1, np.iinfo(np.int64).max], dtype=np.int64), object),
            (np.array([2**53 + 1, np.iinfo(np.uint64).max], dtype=np.uint64), object),
        )
        for output, expected_dtype in cases:
            with self.subTest(output=output):
                distribution, context, _ = self.masked_distribution("ignore", {"a": np.array([True, False, True])}, output)
                result = distribution._evaluate(context)
                self.assertEqual(result.dtype, np.dtype(expected_dtype))
                self.assertEqual(int(result[0]), int(output[0]))
                self.assertEqual(int(result[2]), int(output[1]))
                self.assertTrue(np.isnan(result[1]))

    def test_evaluate_does_not_hide_unrelated_backend_errors(self):
        for mask in (np.array([True, True]), np.array([True, False])):
            with self.subTest(mask=mask):
                distribution, context, _ = self.masked_distribution("ignore", {"a": mask}, None)
                error = RuntimeError("unrelated backend failure")
                distribution._sample_delegate.side_effect = error
                with self.assertRaises(RuntimeError) as caught:
                    distribution._evaluate(context)
                self.assertIs(caught.exception, error)

    def test_cdf_and_ppf_reject_nonreal_support_before_exact_or_sampling_paths(self):
        class Unsupported(_StubDistribution):
            @property
            def value_set(self):
                return self.support
        for support in (ObjectValueSet(("red",)), HomogeneousNumericValueSet(sp.S.Complexes, (np.complex128,))):
            distribution = Unsupported(exact_cdf=0.5, exact_ppf=1.0)
            distribution.support = support
            distribution._monte_carlo = Mock()
            for mode in Mode:
                for method, argument in ((distribution.cdf, 1.0), (distribution.ppf, 0.5)):
                    with self.subTest(support=support, mode=mode, method=method.__name__):
                        with self.assertRaisesRegex(ValueError, "real-valued distribution"):
                            method(argument, mode=mode)
            distribution._monte_carlo.assert_not_called()

    def test_monte_carlo_statistics_forward_descriptors_sample_count_and_rng(self):
        distribution = _StubDistribution()
        distribution._monte_carlo = Mock(return_value=3.0)
        rng = np.random.default_rng(2)
        for method, operation in ((distribution.mean, _MEAN), (distribution.variance, _VARIANCE), (distribution.std, _STD)):
            with self.subTest(method=method.__name__):
                distribution._monte_carlo.reset_mock()
                method(mode=Mode.MONTE_CARLO, num_samples=7, rng=rng)
                distribution._monte_carlo.assert_called_once_with(operation=operation, num_samples=7, rng=rng)

    @patch("problab.distributions.base._RealizationContext")
    @patch("problab.distributions.base._DistributionNode")
    def test_statistics_check_realization_dtype_before_sampling(self, distribution_node, realization_context):
        root = Mock(name="root", _realization_value_set=Mock(dtype_types=(np.object_,)))
        root.name = "source"
        distribution_node.return_value = root
        with self.assertRaisesRegex(TypeError, "std.*source"):
            _StubDistribution().std(mode=Mode.MONTE_CARLO)
        realization_context.assert_not_called()

    @patch("problab.distributions.base._RealizationContext")
    @patch("problab.distributions.base._DistributionNode")
    def test_monte_carlo_preserves_complex_scalar_results(self, distribution_node, realization_context):
        distribution_node.return_value = Mock(_realization_value_set=Mock(dtype_types=(np.complex128,)))
        realization_context.return_value.evaluate.return_value = np.array([1 + 1j, 3 + 3j])
        result = _StubDistribution()._monte_carlo(_MEAN)
        self.assertIs(type(result), complex)
        self.assertEqual(result, 2 + 2j)

    def test_mode_members(self):
        self.assertEqual({Mode.AUTO, Mode.EXACT, Mode.MONTE_CARLO}, set(Mode))

    def test_constructor_preserves_parameters_and_builds_constant_nodes(self):
        distribution = _StubDistribution(parameters=(3, 4.0))

        self.assertEqual(distribution.parameters, (3, 4.0))
        self.assertEqual(distribution.symbol, "Stub")
        self.assertEqual(_StubDistribution.symbol, "Stub")
        self.assertNotIn("symbol", vars(distribution))
        self.assertNotIn("_symbol", vars(distribution))
        self.assertEqual(distribution.name, "Stub(3, 4.0)")
        self.assertEqual(distribution._node_dependencies, set())
        self.assertTrue(
            all(isinstance(node, _ConstantNode) for node in distribution._parameter_nodes)
        )

    def test_distribution_uses_mathematical_support_as_default_realization_support(self):
        distribution = _StubDistribution()

        self.assertIs(distribution._realization_value_set, distribution.value_set)

    def test_parameter_to_node_preserves_existing_node(self):
        node = _ConstantNode(3)

        self.assertIs(Distribution._parameter_to_node(node), node)

    def test_parameter_to_node_uses_a_random_variable_node_without_changing_public_input(self):
        variable = RandomVariable._from_node(_ConstantNode(3), name="X")
        distribution = _StubDistribution(parameters=(variable,))

        self.assertEqual(distribution.parameters, (variable,))
        self.assertIs(distribution._parameter_nodes[0], variable._node)

    def test_node_dependencies_include_variable_parameters_but_not_constants(self):
        parameter_node = _OperationNode(
            operation=_FunctionOperation(
                operation=lambda: np.array([1.0]),
                name_func=lambda: "parameter",
            ),
            inputs=(),
            name="parameter",
            mathematical_value_set=REALS,
            realization_value_set=REALS,
        )
        variable = RandomVariable._from_node(parameter_node, name="X")
        distribution = _StubDistribution(parameters=(variable, 3))

        self.assertEqual(distribution._node_dependencies, {parameter_node})

    @patch("problab.distributions.base._RealizationContext")
    @patch("problab.distributions.base._DistributionNode")
    def test_sample_creates_context_and_evaluates_distribution_node(
        self,
        distribution_node,
        realization_context,
    ):
        distribution = _StubDistribution()
        distribution_node.return_value = Mock(name="root", _realization_value_set=REALS)
        root_node = distribution_node.return_value
        realization_context.return_value.evaluate.return_value = sentinel.samples

        samples = Distribution.sample(distribution)

        self.assertIs(samples, sentinel.samples)
        distribution_node.assert_called_once_with(distribution, rv_name="Stub()")
        realization_context.assert_called_once_with(
            requested_nodes=(root_node,),
            num_samples=1,
            rng=None,
            validate=False,
        )
        realization_context.return_value.evaluate.assert_called_once_with(root_node)

    @patch("problab.distributions.base._RealizationContext")
    @patch("problab.distributions.base._DistributionNode")
    def test_sample_passes_count_rng_and_validation_to_context(
        self,
        distribution_node,
        realization_context,
    ):
        distribution = _StubDistribution()
        generator = np.random.default_rng(321)
        distribution_node.return_value = Mock(name="root", _realization_value_set=REALS)
        root_node = distribution_node.return_value

        distribution.sample(num_samples=5, rng=generator, validate=True)

        realization_context.assert_called_once_with(
            requested_nodes=(root_node,),
            num_samples=5,
            rng=generator,
            validate=True,
        )

    def test_evaluate_realizes_parameter_nodes_and_delegates_to_sample(self):
        distribution = _StubDistribution(parameters=(3,))
        context = Mock(num_samples=5, rng=sentinel.rng)
        parameter_values = np.array([3, 3, 3, 3, 3])
        context.evaluate.return_value = parameter_values
        distribution._sample_delegate.return_value = sentinel.samples

        samples = distribution._evaluate(context)

        self.assertIs(samples, sentinel.samples)
        context.evaluate.assert_called_once_with(distribution._parameter_nodes[0])
        distribution._sample_delegate.assert_called_once_with(
            parameter_values,
            num_samples=5,
            rng=sentinel.rng,
        )

    def test_exact_statistics_are_used_for_exact_and_auto_modes(self):
        distribution = _StubDistribution(exact_mean=4.0, exact_variance=9.0)

        self.assertEqual(distribution.mean(mode=Mode.EXACT), 4.0)
        self.assertEqual(distribution.variance(mode=Mode.EXACT), 9.0)
        self.assertEqual(distribution.std(mode=Mode.EXACT), 3.0)
        self.assertEqual(distribution.mean(mode=Mode.AUTO), 4.0)
        self.assertEqual(distribution.variance(mode=Mode.AUTO), 9.0)
        self.assertEqual(distribution.std(mode=Mode.AUTO), 3.0)

    def test_monte_carlo_statistics_delegate_to_monte_carlo(self):
        distribution = _StubDistribution()
        distribution._monte_carlo = Mock(side_effect=(4.0, 9.0, 3.0))

        self.assertEqual(distribution.mean(mode=Mode.MONTE_CARLO, num_samples=7), 4.0)
        self.assertEqual(distribution.variance(mode=Mode.MONTE_CARLO, num_samples=8), 9.0)
        self.assertEqual(distribution.std(mode=Mode.MONTE_CARLO, num_samples=9), 3.0)
        self.assertEqual(distribution._monte_carlo.call_count, 3)

    def test_auto_statistics_fall_back_to_monte_carlo_when_exact_values_are_unavailable(self):
        distribution = _StubDistribution()
        distribution._monte_carlo = Mock(side_effect=(4.0, 9.0, 3.0))

        self.assertEqual(distribution.mean(mode=Mode.AUTO, num_samples=7), 4.0)
        self.assertEqual(distribution.variance(mode=Mode.AUTO, num_samples=8), 9.0)
        self.assertEqual(distribution.std(mode=Mode.AUTO, num_samples=9), 3.0)
        self.assertEqual(distribution._monte_carlo.call_count, 3)

    def test_exact_mode_rejects_unavailable_statistics(self):
        distribution = _StubDistribution()

        with self.assertRaises(NotImplementedError):
            distribution.mean(mode=Mode.EXACT)
        with self.assertRaises(NotImplementedError):
            distribution.variance(mode=Mode.EXACT)
        with self.assertRaises(NotImplementedError):
            distribution.std(mode=Mode.EXACT)
        with self.assertRaises(NotImplementedError):
            distribution.cdf(1.0, mode=Mode.EXACT)
        with self.assertRaises(NotImplementedError):
            distribution.ppf(0.5, mode=Mode.EXACT)

    def test_cdf_and_ppf_use_exact_values_when_available(self):
        distribution = _StubDistribution(exact_cdf=0.25, exact_ppf=1.5)

        self.assertEqual(distribution.cdf(2.0, mode=Mode.EXACT), 0.25)
        self.assertEqual(distribution.cdf(2.0, mode=Mode.AUTO), 0.25)
        self.assertEqual(distribution.ppf(0.5, mode=Mode.EXACT), 1.5)
        self.assertEqual(distribution.ppf(0.5, mode=Mode.AUTO), 1.5)

    def test_cdf_and_ppf_delegate_to_monte_carlo_when_requested(self):
        distribution = _StubDistribution()
        distribution._monte_carlo = Mock(side_effect=(0.75, 2.0))

        self.assertEqual(distribution.cdf(3.0, mode=Mode.MONTE_CARLO, num_samples=7), 0.75)
        self.assertEqual(distribution.ppf(0.5, mode=Mode.MONTE_CARLO, num_samples=8), 2.0)
        self.assertEqual(distribution._monte_carlo.call_count, 2)

    def test_auto_cdf_and_ppf_fall_back_to_monte_carlo_when_exact_values_are_unavailable(self):
        distribution = _StubDistribution()
        distribution._monte_carlo = Mock(side_effect=(0.75, 2.0))

        self.assertEqual(distribution.cdf(3.0, mode=Mode.AUTO, num_samples=7), 0.75)
        self.assertEqual(distribution.ppf(0.5, mode=Mode.AUTO, num_samples=8), 2.0)
        self.assertEqual(distribution._monte_carlo.call_count, 2)

    def test_monte_carlo_cdf_operations_handle_scalar_and_array_inputs(self):
        distribution = _StubDistribution()

        def run_operation(operation, **kwargs):
            return operation.operation(np.array([1.0, 3.0, 5.0]))

        distribution._monte_carlo = Mock(side_effect=run_operation)

        self.assertEqual(distribution.cdf(3.0, mode=Mode.MONTE_CARLO), 2 / 3)
        np.testing.assert_array_equal(
            distribution.cdf(np.array([0.0, 3.0, 6.0]), mode=Mode.MONTE_CARLO),
            [0.0, 2 / 3, 1.0],
        )

    def test_monte_carlo_ppf_operation_uses_requested_quantile_method(self):
        distribution = _StubDistribution()

        def run_operation(operation, **kwargs):
            return operation.operation(np.array([1.0, 2.0, 3.0, 4.0]))

        distribution._monte_carlo = Mock(side_effect=run_operation)

        self.assertEqual(
            distribution.ppf(0.5, mode=Mode.MONTE_CARLO, quantile_method="inverted_cdf"),
            2.0,
        )

    @patch("problab.distributions.base._RealizationContext")
    @patch("problab.distributions.base._DistributionNode")
    def test_monte_carlo_realizes_samples_and_normalizes_scalar_results(
        self,
        distribution_node,
        realization_context,
    ):
        distribution = _StubDistribution()
        distribution_node.return_value = Mock(name="root", _realization_value_set=REALS)
        root_node = distribution_node.return_value
        realization_context.return_value.evaluate.return_value = np.array([1.0, 3.0])

        result = distribution._monte_carlo(_MEAN, num_samples=2, rng=sentinel.rng)

        self.assertEqual(result, 2.0)
        self.assertIsInstance(result, float)
        realization_context.assert_called_once_with(
            requested_nodes=(root_node,),
            num_samples=2,
            rng=sentinel.rng,
        )

    @patch("problab.distributions.base._RealizationContext")
    @patch("problab.distributions.base._DistributionNode")
    def test_monte_carlo_preserves_array_results(
        self,
        distribution_node,
        realization_context,
    ):
        distribution = _StubDistribution()
        distribution_node.return_value = Mock(name="root", _realization_value_set=REALS)
        root_node = distribution_node.return_value
        realization_context.return_value.evaluate.return_value = np.array([1.0, 3.0])

        result = distribution._monte_carlo(
            _FunctionOperation(operation=lambda samples: np.array([samples.min(), samples.max()]), name_func=lambda name: f"range({name})"),
        )

        np.testing.assert_array_equal(result, [1.0, 3.0])

    @patch("problab.distributions.base._quantile_confidence_interval")
    @patch("problab.distributions.base._RealizationContext")
    @patch("problab.distributions.base._DistributionNode")
    def test_quantile_confidence_interval_realizes_samples_and_delegates(
        self,
        distribution_node,
        realization_context,
        quantile_confidence_interval,
    ):
        distribution = _StubDistribution()
        distribution_node.return_value = Mock(name="root", _realization_value_set=REALS)
        root_node = distribution_node.return_value
        realization_context.return_value.evaluate.return_value = sentinel.samples
        quantile_confidence_interval.return_value = sentinel.interval
        rng = np.random.default_rng(1)

        interval = distribution.quantile_confidence_interval(
            q=0.5,
            alpha=0.1,
            num_samples=8,
            rng=rng,
        )

        self.assertIs(interval, sentinel.interval)
        realization_context.assert_called_once_with(
            requested_nodes=(root_node,),
            num_samples=8,
            rng=rng,
        )
        quantile_confidence_interval.assert_called_once_with(
            samples=sentinel.samples,
            q=0.5,
            alpha=0.1,
        )

    def test_validation_rejects_invalid_distribution_arguments(self):
        distribution = _StubDistribution()

        with self.assertRaises(TypeError):
            distribution.mean(mode="exact")
        with self.assertRaises(ValueError):
            distribution.cdf(1.0, mode=Mode.MONTE_CARLO, num_samples=0)
        with self.assertRaises(TypeError):
            distribution.cdf(1j)
        with self.assertRaises(ValueError):
            distribution.ppf(1.1)


class DistributionSubclassTests(unittest.TestCase):

    def test_continuous_and_discrete_base_classes_remain_abstract(self):
        with self.assertRaises(TypeError):
            _ContinuousDistribution()
        with self.assertRaises(TypeError):
            _DiscreteDistribution()


if __name__ == "__main__":
    unittest.main()
