import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np

from problab.distributions.base import (
    Distribution,
    Mode,
    _ContinuousDistribution,
    _DiscreteDistribution,
)
from problab.random_variables.nodes import _ConstantNode
from problab.value_sets.sets import REALS


class _StubDistribution(Distribution):

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
        super().__init__(parameters=parameters, symbol="Stub")

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

    def test_mode_members(self):
        self.assertEqual({Mode.AUTO, Mode.EXACT, Mode.MONTE_CARLO}, set(Mode))

    def test_constructor_preserves_parameters_and_builds_constant_nodes(self):
        distribution = _StubDistribution(parameters=(3, 4.0))

        self.assertEqual(distribution.parameters, (3, 4.0))
        self.assertEqual(distribution.symbol, "Stub")
        self.assertEqual(distribution.name, "Stub(3, 4.0)")
        self.assertEqual(distribution._node_dependencies, set())
        self.assertTrue(
            all(isinstance(node, _ConstantNode) for node in distribution._parameter_nodes)
        )

    def test_parameter_to_node_preserves_existing_node(self):
        node = _ConstantNode(3)

        self.assertIs(Distribution._parameter_to_node(node), node)

    @patch("problab.distributions.base._RealizationContext")
    @patch("problab.distributions.base._DistributionNode")
    def test_sample_creates_context_and_evaluates_distribution_node(
        self,
        distribution_node,
        realization_context,
    ):
        distribution = _StubDistribution()
        distribution_node.return_value = sentinel.root_node
        realization_context.return_value.evaluate.return_value = sentinel.samples

        samples = Distribution.sample(distribution)

        self.assertIs(samples, sentinel.samples)
        distribution_node.assert_called_once_with(distribution, rv_name="Stub()")
        realization_context.assert_called_once_with(root_node=sentinel.root_node)
        realization_context.return_value.evaluate.assert_called_once_with(sentinel.root_node)

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
        distribution_node.return_value = sentinel.root_node
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
            root_node=sentinel.root_node,
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
