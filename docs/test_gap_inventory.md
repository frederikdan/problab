# Test gap inventory

Updated 2026-10-05 after reviewing all changes since commit `8de9839`.

## Latest verification

The library suite runs **600 test methods: 578 pass and 22 fail or error**.
Unittest reports **14 failures and 37 errors**, because several methods report
more than one failing subtest. No tests are skipped or marked as expected
failures. The library explorer has **7 additional passing extractor tests**.

| Test category | Methods | Passing | Failing methods |
| --- | ---: | ---: | ---: |
| `api_tests` | 25 | 20 | 5 |
| `integration_tests` | 19 | 18 | 1 |
| `property_tests` | 10 | 10 | 0 |
| `regression_tests` | 48 | 40 | 8 |
| `statistical_tests` | 39 | 35 | 4 |
| `unit_tests` | 459 | 455 | 4 |

## Remaining failures

| Issue | Test location |
| --- | --- |
| **F01:** fixed-parameter exact mean, variance, standard deviation, CDF, and PPF hooks remain unimplemented | `tests/api_tests/exact_distribution_methods.py` |
| **B04:** the former third positional sampling argument is interpreted as graph size instead of validation | `tests/regression_tests/test_audit_edge_cases.py` |
| **B05:** Normal parameters reject Python and NumPy booleans with different exception types | `tests/regression_tests/test_audit_edge_cases.py` |
| **B13:** integer-valued categorical floats still use raw SymPy Float support, blocking integer count parameters and real-power inference | `tests/regression_tests/test_categorical_numeric_support.py` |
| **B21:** numeric dtype candidates include timedeltas, fail on some valid numeric loops, and omit longdouble on this Windows environment | `tests/unit_tests/value_sets/_inference.py`, `tests/unit_tests/value_sets/_mathematical_inference.py`, `tests/integration_tests/power.py`, affected statistical workflows |
| **B22:** exceptional-value flags are not validated when `dtype_types=None` | `tests/unit_tests/value_sets/homogeneous_numeric_value_set.py` |
| **F02:** sampling does not yet consistently reject invalid parameter realizations under raise or warn/return NaN under warn/ignore, and distribution realization support does not describe those NaN outputs | `tests/regression_tests/test_parameter_realization_policy.py` |

These tests express intended behavior. They remain active; the library source
was not changed during this test update. The parameter-realization tests cover
Normal mean/std, Binomial n/p, and Poisson mu, with invalid values in only one
sample position so valid positions must still be sampled.

### Every failing test method

- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_cdf_matches_analytical_probability`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_mean_matches_analytical_mean`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_ppf_matches_analytical_median`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_standard_deviation_matches_square_root_of_variance`
- `api_tests.exact_distribution_methods.ExactDistributionMethodTests.test_exact_variance_matches_analytical_variance`
- `integration_tests.power.RandomVariablePowerIntegrationTests.test_real_power_uses_real_samples`
- `regression_tests.test_audit_edge_cases.AuditEdgeCaseTests.test_normal_rejects_python_and_numpy_booleans_with_type_error`
- `regression_tests.test_audit_edge_cases.AuditEdgeCaseTests.test_sample_third_positional_argument_still_means_validate`
- `regression_tests.test_categorical_numeric_support.CategoricalNumericSupportRegressionTests.test_integer_valued_categorical_exponent_keeps_real_square_support`
- `regression_tests.test_categorical_numeric_support.CategoricalNumericSupportRegressionTests.test_integer_valued_float_categories_declare_integer_mathematical_support`
- `regression_tests.test_categorical_numeric_support.CategoricalNumericSupportRegressionTests.test_integer_valued_float_category_can_be_a_binomial_trial_count`
- `regression_tests.test_parameter_realization_policy.ParameterRealizationPolicyRegressionTests.test_ignore_policy_marks_only_invalid_positions_nan_without_warning`
- `regression_tests.test_parameter_realization_policy.ParameterRealizationPolicyRegressionTests.test_raise_policy_rejects_invalid_realized_parameters_with_informative_error`
- `regression_tests.test_parameter_realization_policy.ParameterRealizationPolicyRegressionTests.test_warn_policy_warns_and_marks_only_invalid_positions_nan`
- `statistical_tests.test_analytical_workflows.AnalyticalWorkflowStatisticalTests.test_affine_transform_of_binomial_matches_original_tail_probability`
- `statistical_tests.test_analytical_workflows.AnalyticalWorkflowStatisticalTests.test_exponential_transform_of_normal_matches_analytical_cdf`
- `statistical_tests.test_analytical_workflows.AnalyticalWorkflowStatisticalTests.test_sum_of_independent_binomials_matches_finite_convolution`
- `statistical_tests.test_derived_functions.DerivedFunctionStatisticalTests.test_log_of_exponential_normal_matches_original_normal_tail`
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
