import enum
import unittest
from types import SimpleNamespace

import numpy as np

from problab.validation._common import (
    _validate_alpha,
    _validate_enum,
    _validate_max_size,
    _validate_num_samples,
    _validate_q,
    _validate_rng,
    _validate_validate,
    _validate_numerical_error_policy,
    _require_supported_operation_inputs,
)


class _Mode(enum.Enum):
    VALUE = "value"


class CommonValidationTests(unittest.TestCase):

    def test_numerical_error_policy_accepts_all_modes_and_rejects_invalid_values(self):
        for value in ("warn", "raise", "ignore"):
            _validate_numerical_error_policy(value)
        for value in (None, 1, True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                _validate_numerical_error_policy(value)
        for value in ("", "WARN", "print"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate_numerical_error_policy(value)

    def test_operation_dtype_validation_uses_realizations_and_node_names(self):
        node = SimpleNamespace(name="source", value_set=SimpleNamespace(dtype_types=(np.float64,)),
                               _realization_value_set=SimpleNamespace(dtype_types=(np.object_,)))
        with self.assertRaisesRegex(TypeError, "exp.*'source'"):
            _require_supported_operation_inputs((node,), operation_name="exp", supported_input_types=((np.floating,),))

    def test_operation_dtype_validation_broadcasts_single_rule_to_all_nodes(self):
        for unsupported in (None, (np.float64, np.object_)):
            nodes = (SimpleNamespace(name="X", _realization_value_set=SimpleNamespace(dtype_types=(np.float32,))),
                     SimpleNamespace(name="Y", _realization_value_set=SimpleNamespace(dtype_types=unsupported)))
            with self.subTest(dtype_types=unsupported), self.assertRaisesRegex(TypeError, "'Y'"):
                _require_supported_operation_inputs(nodes, operation_name="hypot", supported_input_types=((np.floating,),))

    def test_operation_dtype_validation_applies_separate_rules_and_checks_arity(self):
        nodes = tuple(SimpleNamespace(name=str(i), _realization_value_set=SimpleNamespace(dtype_types=(t,)))
                      for i, t in enumerate((np.int8, np.bool_)))
        _require_supported_operation_inputs(nodes, operation_name="custom", supported_input_types=((np.integer,), (np.bool_,)))
        with self.assertRaisesRegex(ValueError, "number of input dtype rules"):
            _require_supported_operation_inputs(nodes, operation_name="custom", supported_input_types=())

    def test_unrestricted_operation_does_not_read_input_dtypes(self):
        _require_supported_operation_inputs((object(),), operation_name="equal", supported_input_types=None)

    def test_num_samples_accepts_numpy_integer_and_rejects_invalid_values(self):
        _validate_num_samples(np.int64(1))

        with self.assertRaises(TypeError):
            _validate_num_samples(True)
        with self.assertRaises(TypeError):
            _validate_num_samples(1.0)
        with self.assertRaises(ValueError):
            _validate_num_samples(0)

    def test_rng_accepts_generator_or_none(self):
        _validate_rng(None)
        _validate_rng(np.random.default_rng(1))

        with self.assertRaises(TypeError):
            _validate_rng("rng")

    def test_validate_requires_bool(self):
        _validate_validate(True)

        with self.assertRaises(TypeError):
            _validate_validate(1)

    def test_alpha_and_q_require_open_unit_interval(self):
        _validate_alpha(0.5)
        _validate_q(np.float64(0.5))

        for validator in (_validate_alpha, _validate_q):
            with self.subTest(validator=validator.__name__):
                with self.assertRaises(TypeError):
                    validator(True)
                with self.assertRaises(ValueError):
                    validator(0.0)
                with self.assertRaises(ValueError):
                    validator(1.0)

    def test_enum_requires_declared_member(self):
        _validate_enum(_Mode.VALUE, _Mode, "mode")

        with self.assertRaises(TypeError):
            _validate_enum("value", _Mode, "mode")

    def test_max_size_requires_positive_integer(self):
        _validate_max_size(np.int64(1))

        with self.assertRaises(TypeError):
            _validate_max_size(False)
        with self.assertRaises(ValueError):
            _validate_max_size(0)


if __name__ == "__main__":
    unittest.main()
