# Test gap inventory

This file maps unfinished behavior to active tests. A failing test is kept as a
visible signal that the corresponding library behavior is still missing. The
tests are not skipped or marked as expected failures.

| Missing or incorrect behavior | Test location |
| --- | --- |
| Integer-valued floating values belong to integer sets, including in arrays and events | `tests/regression_tests/test_integer_membership.py` |
| Fixed-width integer arithmetic must widen or raise instead of silently wrapping | `tests/regression_tests/test_integer_overflow.py` |
| Scalar and array CDF calls must reject NaN consistently | `tests/regression_tests/test_cdf_nan.py` |
| Declared support must remain valid at floating-point boundaries | `tests/regression_tests/test_float_boundary_validation.py` |
| A probability result's value must agree with its success count and effective sample count | `tests/regression_tests/test_probability_result_consistency.py` |
| Python and NumPy numeric inputs must follow the same validation rules, with booleans rejected as numeric parameters | `tests/regression_tests/test_numeric_validation_consistency.py` |
| Finite binomial support must have a symbolic representation | `tests/regression_tests/test_binomial_support.py` |
| Quantile confidence intervals must avoid a linear number of scalar binomial tail calls | `tests/regression_tests/test_quantile_search_cost.py` |
| Graph labels must identify distributions and mathematical functions | `tests/regression_tests/test_graph_labels.py` |
| Public distribution sampling must accept sample count, RNG, and validation controls and forward them to the realization context | `tests/api_tests/public_contracts.py`, `tests/unit_tests/distributions/base.py` |
| Users must be able to raise the graph size limit for a deep valid expression | `tests/api_tests/public_contracts.py` |
| Fixed-parameter distributions must provide exact mean, variance, standard deviation, CDF, and PPF | `tests/api_tests/exact_distribution_methods.py` |

The exact-method tests specify the next implementation target for fixed scalar
parameters. Random distribution parameters need a separate analytical contract.

The empty distribution modules (`beta`, `exponential`, `gamma`, both `uniform`
modules, and `geometric`) are placeholders outside the public API. Their
constructors and expected results must be defined before tests can meaningfully
specify them. The precise display format of every graph label and the policy
for exact methods with random parameters also remain design decisions.

Passing tests cannot prove that every possible use case works. Add a new case
here whenever a bug or intended behavior is identified.
