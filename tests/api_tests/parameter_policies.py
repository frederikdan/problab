import unittest
import warnings
from inspect import signature

import numpy as np
import sympy as sp

from problab import (
    BinomialDistribution,
    CategoricalDistribution,
    NormalDistribution,
    PoissonDistribution,
    RandomVariable,
)
from problab.value_sets import HomogeneousNumericValueSet
from problab.value_sets.sets import (
    NATURALS_0,
    NON_NEGATIVE_REALS,
    POSITIVE_REALS,
    REALS,
    UNIT_INTERVAL,
)


class DistributionParameterPolicyTests(unittest.TestCase):

    def test_parameter_metadata_is_private_and_does_not_add_constructor_arguments(self):
        import problab

        for cls, expected in (
            (NormalDistribution, ["mean", "std"]),
            (BinomialDistribution, ["n", "p"]),
            (PoissonDistribution, ["mu"]),
        ):
            with self.subTest(distribution=cls.__name__):
                self.assertEqual(list(cls._valid_parameter_sets), expected)
                self.assertEqual(list(signature(cls).parameters), [*expected, "parameter_risk_policy"])
        self.assertNotIn("_valid_parameter_sets", problab.__all__)

    def test_subclass_parameter_domains_are_read_before_constructor_initialization(self):
        class PositiveMeanNormal(NormalDistribution):
            _valid_parameter_sets = {"mean": POSITIVE_REALS, "std": POSITIVE_REALS}

        with self.assertRaises(ValueError):
            PositiveMeanNormal(-1., 1.)
        distribution = PositiveMeanNormal(1., 2.)
        self.assertEqual(distribution.parameters, (1., 2.))
        self.assertIs(NormalDistribution._valid_parameter_sets["mean"], REALS)
    def parameter(self, mathematical_set, realization_set, name="parameter"):
        source = RandomVariable(CategoricalDistribution([1.0], [1.0]), name=name)
        return source.apply(lambda x: x, mathematical_value_set=mathematical_set,
                            realization_value_set=realization_set, vectorized=True)

    def cases(self):
        return (
            ("Normal", "mean", REALS, lambda x, **kw: NormalDistribution(x, 1, **kw)),
            ("Normal", "std", POSITIVE_REALS, lambda x, **kw: NormalDistribution(0, x, **kw)),
            ("Binomial", "n", NATURALS_0, lambda x, **kw: BinomialDistribution(x, 0.5, **kw)),
            ("Binomial", "p", UNIT_INTERVAL, lambda x, **kw: BinomialDistribution(2, x, **kw)),
            ("Poisson", "mu", NON_NEGATIVE_REALS, lambda x, **kw: PoissonDistribution(x, **kw)),
        )

    def test_parameter_risk_policy_is_keyword_only_with_warn_default(self):
        for distribution in (NormalDistribution, BinomialDistribution, PoissonDistribution):
            with self.subTest(distribution=distribution.__name__):
                parameter = signature(distribution).parameters["parameter_risk_policy"]
                self.assertEqual(parameter.kind, parameter.KEYWORD_ONLY)
                self.assertEqual(parameter.default, "warn")

    def test_risky_parameters_warn_by_default_raise_or_allow_silently_when_requested(self):
        for symbol, name, mathematical_set, constructor in self.cases():
            risky = HomogeneousNumericValueSet(mathematical_set.sympy_set, (np.float64,), allows_nan=True)
            parameter = self.parameter(mathematical_set, risky, name)
            with self.subTest(distribution=symbol, parameter=name):
                with self.assertWarnsRegex(RuntimeWarning, f"'{name}'.*{symbol}"):
                    distribution = constructor(parameter)
                self.assertTrue(any(value is parameter for value in distribution.parameters))
                with self.assertRaisesRegex(ValueError, f"'{name}'.*{symbol}"):
                    constructor(parameter, parameter_risk_policy="raise")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    constructor(parameter, parameter_risk_policy="ignore")
                self.assertEqual(caught, [])

    def test_risk_policy_never_bypasses_mathematically_invalid_parameters(self):
        for policy in ("warn", "raise", "ignore"):
            for constructor in (lambda: NormalDistribution(0, -1, parameter_risk_policy=policy),
                                lambda: BinomialDistribution(-1, 0.5, parameter_risk_policy=policy),
                                lambda: BinomialDistribution(2, 1.1, parameter_risk_policy=policy),
                                lambda: PoissonDistribution(-1, parameter_risk_policy=policy)):
                with self.subTest(policy=policy, constructor=constructor), self.assertRaises(ValueError):
                    constructor()

    def test_risk_policy_never_bypasses_unsupported_realization_dtype(self):
        for symbol, name, mathematical_set, constructor in self.cases():
            parameter = self.parameter(mathematical_set, HomogeneousNumericValueSet(mathematical_set.sympy_set, (np.object_,)))
            for policy in ("warn", "raise", "ignore"):
                with self.subTest(distribution=symbol, parameter=name, policy=policy), self.assertRaises(TypeError):
                    constructor(parameter, parameter_risk_policy=policy)

    def test_invalid_policy_is_rejected_before_other_parameter_errors(self):
        for distribution, args in ((NormalDistribution, ("wrong", "wrong")),
                                   (BinomialDistribution, ("wrong", "wrong")), (PoissonDistribution, ("wrong",))):
            for policy, exception in ((None, TypeError), ("invalid", ValueError)):
                with self.subTest(distribution=distribution.__name__, policy=policy):
                    with self.assertRaisesRegex(exception, "parameter_risk_policy"):
                        distribution(*args, parameter_risk_policy=policy)

    def test_subclass_static_symbol_is_used_in_name_and_parameter_risk_message(self):
        class CustomNormal(NormalDistribution):
            symbol = "CustomN"
        parameter = self.parameter(POSITIVE_REALS, HomogeneousNumericValueSet(sp.Interval(0, 1), (np.float64,)))
        with self.assertWarnsRegex(RuntimeWarning, "'std'.*CustomN"):
            distribution = CustomNormal(0, parameter)
        self.assertTrue(distribution.name.startswith("CustomN("))
        self.assertNotIn("symbol", vars(distribution))

    def test_static_symbols_do_not_add_constructor_parameters(self):
        for distribution, symbol in ((NormalDistribution, "Normal"), (BinomialDistribution, "Binomial"),
                                     (PoissonDistribution, "Poisson"), (CategoricalDistribution, "Categorical")):
            with self.subTest(distribution=distribution.__name__):
                self.assertEqual(distribution.symbol, symbol)
                self.assertNotIn("symbol", signature(distribution).parameters)
