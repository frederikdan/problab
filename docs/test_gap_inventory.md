# Test gap inventory

This file maps unfinished behavior to active tests. A failing test is kept as a
visible signal that the corresponding library behavior is still missing. The
tests are not skipped or marked as expected failures.

| Missing or incorrect behavior | Test location |
| --- | --- |
| **Resolved:** integer-valued floating values belong to integer sets, including in arrays and events | `tests/regression_tests/test_integer_membership.py` |
| **Resolved:** fixed-width integer arithmetic raises instead of silently wrapping | `tests/regression_tests/test_integer_overflow.py` |
| **Resolved:** scalar and array CDF calls reject NaN consistently | `tests/regression_tests/test_cdf_nan.py` |
| **Resolved:** declared mathematical support remains separate from machine-realization support at floating-point boundaries | `tests/regression_tests/test_float_boundary_validation.py`, `tests/unit_tests/random_variables/_context.py` |
| **Resolved:** a probability result's value agrees with its success count and effective sample count within the configured tolerance | `tests/regression_tests/test_probability_result_consistency.py` |
| **Resolved:** Python and NumPy numeric inputs follow the same validation rules, with booleans rejected as numeric parameters | `tests/regression_tests/test_numeric_validation_consistency.py` |
| **Resolved:** finite binomial support has a symbolic representation | `tests/regression_tests/test_binomial_support.py` |
| **Resolved:** quantile confidence intervals use binary search and avoid a linear number of scalar binomial tail calls | `tests/regression_tests/test_quantile_search_cost.py` |
| **Resolved:** graph labels identify distributions and mathematical functions | `tests/regression_tests/test_graph_labels.py` |
| **Resolved:** public distribution sampling accepts sample count, RNG, and validation controls and forwards them to the realization context | `tests/api_tests/public_contracts.py`, `tests/unit_tests/distributions/base.py` |
| **Resolved:** users can raise the graph size limit for a deep valid expression | `tests/api_tests/public_contracts.py` |
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
