# Test gap inventory

Updated 2026-10-10 after the numeric, object, sampling, and partial Fraction work.

## Latest full verification (2026-10-10)

The library suite runs **659 test methods: 654 pass and 5 error**. The five
methods are the exact-distribution tests listed below; their four distribution
subtests produce **20 error reports**, all from the unimplemented F01 hooks.
There are no failure reports, skipped tests, or expected failures. The library
explorer's **7 extractor tests pass** when run with normal temporary-directory
access; the sandbox-only attempt was blocked by filesystem permissions.

| Test category | Methods | Passing | Failing methods |
| --- | ---: | ---: | ---: |
| `api_tests` | 27 | 22 | 5 |
| `integration_tests` | 25 | 25 | 0 |
| `property_tests` | 11 | 11 | 0 |
| `regression_tests` | 63 | 63 | 0 |
| `statistical_tests` | 40 | 40 | 0 |
| `unit_tests` | 493 | 493 | 0 |

The currently failing methods are:

- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_cdf_matches_analytical_probability`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_mean_matches_analytical_mean`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_ppf_matches_analytical_median`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_standard_deviation_matches_square_root_of_variance`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_variance_matches_analytical_variance`

The previously failing B04, B05, B13, B21, and B22 methods now pass. Their
historical entries below are retained. B16 remains open: scalar Fraction inputs
are supported, while the reviewed array helper is not yet integrated into
numerical entry points or Fraction-valued random-input support inference.

## Focused follow-up: Fraction scalar inputs (2026-10-10)

B16 now has eight public regression tests in
`tests/regression_tests/test_fraction_scalar_inputs.py`. All eight pass, along
with the affected unit suites: **139 focused methods passed**. This verifies
categorical probabilities, scalar CDF/PPF inputs, q/alpha conversions, interval
calculations, and ProbabilityResult consistency checks. Fraction arrays and
Fraction-valued random inputs remain unfinished. A subsequent helper-only step
adds five array-conversion unit tests; all 44 focused utility/categorical methods
pass. The array helper preserves shape, source arrays, and other large integers,
but is not yet integrated into numerical entry points. The library now contains
659 test methods. The full run above subsequently verified the combined changes.

## Previous full verification (2026-10-06)

The library suite runs **646 test methods: 630 pass and 16 fail or error**.
Unittest reports **10 failures and 26 errors**, because several methods report
more than one failing subtest. No tests are skipped or marked as expected
failures. The library explorer has **7 additional passing extractor tests**.

| Test category | Methods | Passing | Failing methods |
| --- | ---: | ---: | ---: |
| `api_tests` | 27 | 22 | 5 |
| `integration_tests` | 25 | 25 | 0 |
| `property_tests` | 11 | 11 | 0 |
| `regression_tests` | 55 | 50 | 5 |
| `statistical_tests` | 40 | 38 | 2 |
| `unit_tests` | 488 | 484 | 4 |

## Failures recorded on 2026-10-06

| Issue | Test location |
| --- | --- |
| **F01:** fixed-parameter exact mean, variance, standard deviation, CDF, and PPF hooks remain unimplemented | `tests/api_tests/exact_distribution_methods.py` |
| **B04:** the former third positional sampling argument is interpreted as graph size instead of validation | `tests/regression_tests/test_audit_edge_cases.py` |
| **B05:** Normal parameters reject Python and NumPy booleans with different exception types | `tests/regression_tests/test_audit_edge_cases.py` |
| **B13:** integer-valued categorical floats still use raw SymPy Float support, blocking integer count parameters and real-power inference | `tests/regression_tests/test_categorical_numeric_support.py` |
| **B21:** numeric dtype candidates include timedeltas, fail on some valid numeric loops, and omit longdouble on this Windows environment | `tests/unit_tests/value_sets/_inference.py`, `tests/unit_tests/value_sets/_mathematical_inference.py`, affected Binomial statistical workflows |
| **B22:** exceptional-value flags are not validated when `dtype_types=None` | `tests/unit_tests/value_sets/homogeneous_numeric_value_set.py` |

These remaining tests express intended behavior and remain active.

The 2026-10-05 baseline was 600 methods, with 578 passing and 22 failing.
F02 now passes for Normal mean/std, Binomial n/p, and Poisson mu. New checks
cover all-invalid batches, backend filtering, combined masks, exact integer
storage, finite backend limits, and unchanged all-valid sampling. Normal's
floating-only realization dtype also resolves its power integration test and
two exponential/logarithm workflows; the general B21 inference defect remains.
The follow-up suite adds 34 passing tests, including direct tests of class
parameter domains and guarantee checks, numeric mask/value-set agreement,
shared-node sampling, independent policy controls, storage boundaries, and
analytical probabilities at valid positions. Every failing method below is
unchanged by that coverage expansion.

### Every failing test method

- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_cdf_matches_analytical_probability`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_mean_matches_analytical_mean`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_ppf_matches_analytical_median`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_standard_deviation_matches_square_root_of_variance`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_variance_matches_analytical_variance`
- `regression_tests.test_audit_edge_cases.AuditEdgeCaseTests.test_normal_rejects_python_and_numpy_booleans_with_type_error`
- `regression_tests.test_audit_edge_cases.AuditEdgeCaseTests.test_sample_third_positional_argument_still_means_validate`
- `regression_tests.test_categorical_numeric_support.CategoricalNumericSupportRegressionTests.test_integer_valued_categorical_exponent_keeps_real_square_support`
- `regression_tests.test_categorical_numeric_support.CategoricalNumericSupportRegressionTests.test_integer_valued_float_categories_declare_integer_mathematical_support`
- `regression_tests.test_categorical_numeric_support.CategoricalNumericSupportRegressionTests.test_integer_valued_float_category_can_be_a_binomial_trial_count`
- `statistical_tests.test_analytical_workflows.AnalyticalWorkflowStatisticalTests.test_affine_transform_of_binomial_matches_original_tail_probability`
- `statistical_tests.test_analytical_workflows.AnalyticalWorkflowStatisticalTests.test_sum_of_independent_binomials_matches_finite_convolution`
- `unit_tests.value_sets._inference.SharedInferenceTests.test_numeric_families_exclude_datetime_and_timedelta_storage`
- `unit_tests.value_sets._inference.SharedInferenceTests.test_real_numeric_families_can_resolve_exponential_output_dtypes`
- `unit_tests.value_sets._mathematical_inference.MathematicalValueSetInferenceTests.test_float_families_include_extended_precision`
- `unit_tests.value_sets.homogeneous_numeric_value_set.HomogeneousNumericValueSetTests.test_exceptional_permissions_require_booleans_even_with_unknown_dtype`

## Passing checks for the recent changes

- Operations have matching unit files under `tests/unit_tests/operations/`. Tests cover descriptor metadata, supported dtypes, both inference hooks, Boolean truth tables, comparisons, and population statistics.
- Shared, mathematical, and realization inference have separate unit files. Realization tests cover signed overflow, the float64 integer fallback, underflow to zero, exceptional-value propagation, power parity, rounded interval bounds, and every predefined function inferer.
- Construction parameter-risk validation covers all five numeric parameters and all three policies. Tests check defaults, validation order, instance and argument forwarding, static symbols, subclass symbols, mathematical support, and supported realization dtypes.
- Categorical tests cover cached lossless conversion, precision-loss rejection, object fallback, exact large integers, signed infinity, NaN, complex components, public input preservation, and the dtype actually sampled.
- Numerical-function workflows check every unary predefined function at float16/float32/float64 precision, binary mixed inputs, rounded inverse-trigonometric endpoints, overflow, underflow, and invalid machine inputs.
- Requested-node graph and context tests cover union accounting, duplicate requests, retained requested dependencies, shared caches, failure/retry behavior, and conditional complements and identical conditions.
- Statistical tests cover mixed integer/float Normal means and Poisson rates against independently calculated mixture probabilities with six-standard-error bounds.
- Existing checks for integer-valued floating membership, CDF NaN rejection, probability-result consistency, finite binomial support, binary quantile search, graph labels, public sampling controls, and confidence intervals remain active.

## Scope still needing tests or a design decision

The earlier audit also lists bugs without complete persistent regressions; see
**T01** in [audit_todo.md](audit_todo.md). This update covers the changes since
the last commit and selected directly related regressions. It does not prove
every possible library use case works.

Empty distribution modules (`beta`, `exponential`, `gamma`, both `uniform`
modules, and `geometric`) remain placeholders outside the public API. Define
their constructors and behavior before writing meaningful behavior tests.
Exact methods with random parameters and dtype preservation by simplification
still need their separate design decisions.
