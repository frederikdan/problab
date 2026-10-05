import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes.base import _Node
from problab.value_sets.sets import NATURALS_0, UNIT_INTERVAL

from problab.validation.distributions.discrete._binomial import (
    _validate_binomial_n,
    _validate_binomial_p,
)


class BinomialDistributionValidationTests(unittest.TestCase):

    def test_n_delegates_dtype_and_risk_validation_with_instance_symbol(self):
        node = Mock(spec=_Node)
        node.name = "parameter"
        node.value_set = NATURALS_0
        variable = RandomVariable._from_node(node)
        prefix = "problab.validation.distributions.discrete._binomial"
        with (
            patch(prefix + "._require_supported_parameter_realization_dtypes") as dtype_check,
            patch(prefix + "._validate_parameter_realization_value_set") as risk_check,
        ):
            _validate_binomial_n(variable, instance=self.instance, parameter_risk_policy="ignore")
        dtype_check.assert_called_once_with(
            (node,), distribution_name=self.instance.symbol,
            supported_input_types=((np.integer, np.floating),),
        )
        risk_check.assert_called_once_with(
            node, parameter_name="n", distribution_name=self.instance.symbol,
            valid_value_set=NATURALS_0, parameter_risk_policy="ignore",
        )

    def test_p_delegates_dtype_and_risk_validation_with_instance_symbol(self):
        node = Mock(spec=_Node)
        node.name = "parameter"
        node.value_set = UNIT_INTERVAL
        variable = RandomVariable._from_node(node)
        prefix = "problab.validation.distributions.discrete._binomial"
        with (
            patch(prefix + "._require_supported_parameter_realization_dtypes") as dtype_check,
            patch(prefix + "._validate_parameter_realization_value_set") as risk_check,
        ):
            _validate_binomial_p(variable, instance=self.instance, parameter_risk_policy="ignore")
        dtype_check.assert_called_once_with(
            (node,), distribution_name=self.instance.symbol,
            supported_input_types=((np.integer, np.floating),),
        )
        risk_check.assert_called_once_with(
            node, parameter_name="p", distribution_name=self.instance.symbol,
            valid_value_set=UNIT_INTERVAL, parameter_risk_policy="ignore",
        )

    def setUp(self):
        self.instance = SimpleNamespace(symbol="CustomBinomial")

    def test_numeric_parameters_reject_object_realizations_with_symbol_and_node_name(self):
        for validator, mathematical_set in (
            (_validate_binomial_n, NATURALS_0),
            (_validate_binomial_p, UNIT_INTERVAL),
        ):
            with self.subTest(validator=validator.__name__):
                node = Mock(spec=_Node)
                node.name = "X"
                node.value_set = mathematical_set
                node._realization_value_set = SimpleNamespace(dtype_types=(np.object_,))
                variable = RandomVariable._from_node(node)

                with self.assertRaisesRegex(TypeError, "CustomBinomial.*'X'"):
                    validator(variable, instance=self.instance, parameter_risk_policy="raise")

    def test_n_accepts_integer_support_with_floating_realizations(self):
        node = Mock(spec=_Node)
        node.name = "X"
        node.value_set = NATURALS_0
        node._realization_value_set = NATURALS_0

        _validate_binomial_n(RandomVariable._from_node(node), instance=self.instance, parameter_risk_policy="raise")

    def test_n_accepts_natural_zero_and_rejects_invalid_values(self):
        _validate_binomial_n(0, instance=self.instance, parameter_risk_policy="raise")
        _validate_binomial_n(2, instance=self.instance, parameter_risk_policy="raise")
        _validate_binomial_n(np.int64(2), instance=self.instance, parameter_risk_policy="raise")

        with self.assertRaises(ValueError):
            _validate_binomial_n(-1, instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(TypeError):
            _validate_binomial_n(2.0, instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(TypeError):
            _validate_binomial_n(True, instance=self.instance, parameter_risk_policy="raise")

    def test_p_accepts_closed_unit_interval_and_rejects_invalid_values(self):
        _validate_binomial_p(0.0, instance=self.instance, parameter_risk_policy="raise")
        _validate_binomial_p(1.0, instance=self.instance, parameter_risk_policy="raise")

        with self.assertRaises(ValueError):
            _validate_binomial_p(-0.1, instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(ValueError):
            _validate_binomial_p(1.1, instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(TypeError):
            _validate_binomial_p("half", instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(TypeError):
            _validate_binomial_p(True, instance=self.instance, parameter_risk_policy="raise")


if __name__ == "__main__":
    unittest.main()
