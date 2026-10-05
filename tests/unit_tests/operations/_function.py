import unittest
import numpy as np
from problab.operations import _function as functions
from problab.value_sets import _realization_inference as inference


class FunctionOperationTests(unittest.TestCase):
    def test_predefined_descriptors_connect_all_functions_to_their_inferers(self):
        names = ("exp", "log", "sqrt", "sin", "arcsin", "cos", "arccos", "tan", "arctan",
                 "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "log2", "log10",
                 "floor", "ceil", "sign", "log1p", "expm1", "hypot", "logaddexp")
        for name in names:
            with self.subTest(function=name):
                descriptor = getattr(functions, "_" + name.upper())
                self.assertIsInstance(descriptor, functions._FunctionOperation)
                self.assertIs(descriptor.operation, getattr(np, name))
                self.assertEqual(descriptor.supported_input_types, ((np.integer, np.floating),))
                self.assertIs(descriptor.infer_realization_value_set, getattr(inference, f"_infer_{name}_value_set"))
                args = ("X", "Y") if name in ("hypot", "logaddexp") else ("X",)
                self.assertEqual(descriptor.name_func(*args), f"{name}({', '.join(args)})")

    def test_custom_descriptor_does_not_require_an_inference_hook(self):
        descriptor = functions._FunctionOperation(operation=lambda x: x, name_func=str)
        self.assertIsNone(descriptor.infer_realization_value_set)
        self.assertIsNone(descriptor.supported_input_types)
