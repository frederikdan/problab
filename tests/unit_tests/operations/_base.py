import dataclasses
import unittest

import numpy as np

from problab.operations._base import (
    _Operation, _BOOLEAN_INPUT_TYPES, _NUMERIC_INPUT_TYPES, _REAL_NUMERIC_INPUT_TYPES,
)


class OperationBaseTests(unittest.TestCase):
    def test_descriptor_retains_callable_name_and_optional_dtype_requirements(self):
        operation = _Operation(operation=np.add, name_func=lambda a, b: f"{a}+{b}")
        self.assertIs(operation.operation, np.add)
        self.assertEqual(operation.name_func("X", "Y"), "X+Y")
        self.assertIsNone(operation.supported_input_types)

    def test_descriptor_is_immutable(self):
        operation = _Operation(operation=np.add, name_func=str)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            operation.supported_input_types = ((np.object_,),)

    def test_descriptor_constructor_requires_keywords(self):
        with self.assertRaises(TypeError):
            _Operation(np.add, str)

    def test_shared_dtype_rules_describe_numeric_real_and_boolean_inputs(self):
        self.assertEqual(_NUMERIC_INPUT_TYPES, ((np.number,),))
        self.assertEqual(_REAL_NUMERIC_INPUT_TYPES, ((np.integer, np.floating),))
        self.assertEqual(_BOOLEAN_INPUT_TYPES, ((np.bool_,),))
