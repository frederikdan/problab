import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from problab.random_variables.base import RandomVariable
from problab.random_variables.nodes.base import _Node
from problab.value_sets.sets import REALS, POSITIVE_REALS

from problab.validation.distributions.continuous._normal import (
    _validate_normal_mean,
    _validate_normal_std,
)


class NormalDistributionValidationTests(unittest.TestCase):

    def test_mean_delegates_dtype_and_risk_validation_with_instance_symbol(self):
        node = Mock(spec=_Node)
        node.name = "parameter"
        node.value_set = REALS
        variable = RandomVariable._from_node(node)
        prefix = "problab.validation.distributions.continuous._normal"
        with (
            patch(prefix + "._require_supported_parameter_realization_dtypes") as dtype_check,
            patch(prefix + "._validate_parameter_realization_value_set") as risk_check,
        ):
            _validate_normal_mean(variable, instance=self.instance, parameter_risk_policy="ignore")
        dtype_check.assert_called_once_with(
            (node,), distribution_name=self.instance.symbol,
            supported_input_types=((np.integer, np.floating),),
        )
        risk_check.assert_called_once_with(
            node, parameter_name="mean", distribution_name=self.instance.symbol,
            valid_value_set=REALS, parameter_risk_policy="ignore",
        )

    def test_std_delegates_dtype_and_risk_validation_with_instance_symbol(self):
        node = Mock(spec=_Node)
        node.name = "parameter"
        node.value_set = POSITIVE_REALS
        variable = RandomVariable._from_node(node)
        prefix = "problab.validation.distributions.continuous._normal"
        with (
            patch(prefix + "._require_supported_parameter_realization_dtypes") as dtype_check,
            patch(prefix + "._validate_parameter_realization_value_set") as risk_check,
        ):
            _validate_normal_std(variable, instance=self.instance, parameter_risk_policy="ignore")
        dtype_check.assert_called_once_with(
            (node,), distribution_name=self.instance.symbol,
            supported_input_types=((np.integer, np.floating),),
        )
        risk_check.assert_called_once_with(
            node, parameter_name="std", distribution_name=self.instance.symbol,
            valid_value_set=POSITIVE_REALS, parameter_risk_policy="ignore",
        )

    def setUp(self):
        self.instance = SimpleNamespace(symbol="CustomNormal")

    def test_numeric_parameters_reject_object_realizations_with_symbol_and_node_name(self):
        for validator, mathematical_set in (
            (_validate_normal_mean, REALS),
            (_validate_normal_std, POSITIVE_REALS),
        ):
            with self.subTest(validator=validator.__name__):
                node = Mock(spec=_Node)
                node.name = "X"
                node.value_set = mathematical_set
                node._realization_value_set = SimpleNamespace(dtype_types=(np.object_,))
                variable = RandomVariable._from_node(node)

                with self.assertRaisesRegex(TypeError, "CustomNormal.*'X'"):
                    validator(variable, instance=self.instance, parameter_risk_policy="raise")

    def test_mean_accepts_real_scalar_and_rejects_invalid_values(self):
        _validate_normal_mean(0.0, instance=self.instance, parameter_risk_policy="raise")

        with self.assertRaises(TypeError):
            _validate_normal_mean("zero", instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(TypeError):
            _validate_normal_mean(1j, instance=self.instance, parameter_risk_policy="raise")

    def test_std_requires_positive_real_scalar(self):
        _validate_normal_std(1.0, instance=self.instance, parameter_risk_policy="raise")

        with self.assertRaises(ValueError):
            _validate_normal_std(0.0, instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(ValueError):
            _validate_normal_std(-1.0, instance=self.instance, parameter_risk_policy="raise")
        with self.assertRaises(TypeError):
            _validate_normal_std("one", instance=self.instance, parameter_risk_policy="raise")


if __name__ == "__main__":
    unittest.main()
