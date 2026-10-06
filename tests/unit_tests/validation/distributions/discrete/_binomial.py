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
    _validate_binomial_n_realizations,
    _validate_binomial_p_realizations,
    _BINOMIAL_N_MAX,
    _BINOMIAL_N_REALIZATION_SET,
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
            valid_value_set=_BINOMIAL_N_REALIZATION_SET, parameter_risk_policy="ignore",
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
        self.instance = SimpleNamespace(
            symbol="CustomBinomial",
            _valid_parameter_sets={"n": NATURALS_0, "p": UNIT_INTERVAL},
        )

    def test_realized_trial_counts_require_finite_nonnegative_integers(self):
        for dtype in (np.float16, np.float32, np.float64, np.longdouble):
            with self.subTest(dtype=dtype), np.errstate(all="raise"):
                values = np.array([0, 2, -1, 1.5, np.nan, np.inf, -np.inf], dtype=dtype)
                np.testing.assert_array_equal(
                    _validate_binomial_n_realizations(values),
                    [True, True, False, False, False, False, False],
                )

    def test_realized_trial_count_backend_boundary_preserves_integer_precision(self):
        for dtype in (np.int64, np.uint64):
            values = np.array([0, 2**53 + 1, _BINOMIAL_N_MAX], dtype=dtype)
            np.testing.assert_array_equal(_validate_binomial_n_realizations(values), [True] * 3)
        np.testing.assert_array_equal(
            _validate_binomial_n_realizations(np.array([int(_BINOMIAL_N_MAX) + 1], dtype=np.uint64)),
            [False],
        )
        boundary = np.float64(int(_BINOMIAL_N_MAX) + 1)
        np.testing.assert_array_equal(
            _validate_binomial_n_realizations(np.array([np.nextafter(boundary, 0), boundary])),
            [True, False],
        )

    def test_realized_probabilities_require_finite_closed_unit_interval(self):
        values = np.array([0, 0.5, 1, -0.1, 1.1, np.nan, np.inf, -np.inf])
        np.testing.assert_array_equal(
            _validate_binomial_p_realizations(values),
            [True, True, True, False, False, False, False, False],
        )

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
        node._realization_value_set = _BINOMIAL_N_REALIZATION_SET

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
