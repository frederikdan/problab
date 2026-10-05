import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes.base import _Node
from problab.value_sets.sets import NON_NEGATIVE_REALS

from problab.validation.distributions.discrete._poisson import _validate_poisson_mu


class PoissonDistributionValidationTests(unittest.TestCase):

    def test_mu_delegates_dtype_and_risk_validation_with_instance_symbol(self):
        node = Mock(spec=_Node)
        node.name = "parameter"
        node.value_set = NON_NEGATIVE_REALS
        variable = RandomVariable._from_node(node)
        prefix = "problab.validation.distributions.discrete._poisson"
        with (
            patch(prefix + "._require_supported_parameter_realization_dtypes") as dtype_check,
            patch(prefix + "._validate_parameter_realization_value_set") as risk_check,
        ):
            _validate_poisson_mu(variable, instance=self.instance, parameter_risk_policy="ignore")
        dtype_check.assert_called_once_with(
            (node,), distribution_name=self.instance.symbol,
            supported_input_types=((np.integer, np.floating),),
        )
        risk_check.assert_called_once_with(
            node, parameter_name="mu", distribution_name=self.instance.symbol,
            valid_value_set=NON_NEGATIVE_REALS, parameter_risk_policy="ignore",
        )

    def setUp(self):
        self.instance = SimpleNamespace(symbol="CustomPoisson")

    def test_mu_rejects_object_realizations_with_symbol_and_node_name(self):
        node = Mock(spec=_Node)
        node.name = "X"
        node.value_set = NON_NEGATIVE_REALS
        node._realization_value_set = SimpleNamespace(dtype_types=(np.object_,))
        variable = RandomVariable._from_node(node)

        with self.assertRaisesRegex(TypeError, "CustomPoisson.*'X'"):
            _validate_poisson_mu(variable, instance=self.instance, parameter_risk_policy="raise")

    def test_mu_accepts_non_negative_real_and_rejects_invalid_values(self):
        _validate_poisson_mu(0.0, instance=self.instance, parameter_risk_policy="raise")
        _validate_poisson_mu(1.5, instance=self.instance, parameter_risk_policy="raise")

        with self.assertRaises(ValueError):
            _validate_poisson_mu(-0.1, instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(TypeError):
            _validate_poisson_mu("rate", instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(TypeError):
            _validate_poisson_mu(True, instance=self.instance, parameter_risk_policy="raise")


if __name__ == "__main__":
    unittest.main()
