# ProbLab test inventory

Updated 2026-10-05. This inventory lists all 607 test methods: 600 in the
library test suite and 7 in the library explorer. Each entry gives the test
class and function name, followed by what it checks. A method using `subTest`
counts once here even when it exercises many cases.

Known failing tests remain in this list. See
[test_gap_inventory.md](test_gap_inventory.md) for the current run and failures.

## Unit tests

### `tests/unit_tests/_config.py`

- `NumericalConfigTests.test_default_numerical_and_parameter_policies_warn` — Checks that default numerical and parameter policies warn.
- `NumericalConfigTests.test_tolerances_are_positive_and_less_than_one` — Checks that the shared numerical tolerances are positive fractions.

### `tests/unit_tests/_events.py`

- `EventTests.test_logical_operations_check_node_dtypes_before_simplifying` — Checks that logical operations check node dtypes before simplifying.
- `EventTests.test_constructor_exposes_boolean_node_name_and_representation` — Checks that an event exposes its boolean node’s name and useful representation.
- `EventTests.test_constructor_rejects_non_boolean_node` — Checks that constructing an event from a non-boolean node raises an error.
- `EventTests.test_logical_operations_build_boolean_operation_events` — Checks that logical event operators construct boolean operation events.
- `EventTests.test_repeated_conjunction_reuses_the_event_node` — Checks that combining an event with itself reuses the original node.
- `EventTests.test_repeated_disjunction_reuses_the_event_node` — Checks that disjoining an event with itself reuses the original node.
- `EventTests.test_double_inversion_reuses_the_event_node` — Checks that double event inversion reuses the original node.
- `EventTests.test_event_and_its_inverse_simplifies_to_false` — Checks that an event combined with its inverse produces a constant false event.
- `EventTests.test_inverse_and_event_simplifies_to_false` — Checks the same contradiction rule when operand order is reversed.
- `EventTests.test_event_or_its_inverse_simplifies_to_true` — Checks that an event combined with its inverse by disjunction produces a constant true event.
- `EventTests.test_inverse_or_event_simplifies_to_true` — Checks the same tautology rule when operand order is reversed.
- `EventTests.test_logical_operations_evaluate_each_boolean_array` — Checks that logical event operators calculate elementwise Boolean array results.
- `EventTests.test_logical_operations_reject_non_events` — Checks that logical event operators reject operands that are not events.
- `EventTests.test_event_has_no_single_truth_value` — Checks that an event cannot be coerced into one Python Boolean value.

### `tests/unit_tests/_utils.py`

- `NumericalUtilityTests.test_is_close_accepts_exact_and_within_tolerance_values` — Checks that the shared numeric comparison accepts exact and configured-close values.
- `NumericalUtilityTests.test_is_close_rejects_nan_and_values_outside_tolerance` — Checks that the shared numeric comparison rejects NaN and materially different values.

### `tests/unit_tests/distributions/_config.py`

- `DistributionConfigTests.test_defaults` — Checks the default distribution configuration constants used throughout the package.

### `tests/unit_tests/distributions/base.py`

- `DistributionBaseTests.test_cdf_and_ppf_reject_nonreal_support_before_exact_or_sampling_paths` — Checks that object and complex distribution supports are rejected by CDF and PPF in every mode before Monte Carlo sampling, even when an exact hook supplies a value.
- `DistributionBaseTests.test_monte_carlo_statistics_forward_descriptors_sample_count_and_rng` — Checks that monte carlo statistics forward descriptors sample count and rng.
- `DistributionBaseTests.test_statistics_check_realization_dtype_before_sampling` — Checks that statistics check realization dtype before sampling.
- `DistributionBaseTests.test_monte_carlo_preserves_complex_scalar_results` — Checks that monte carlo preserves complex scalar results.
- `DistributionBaseTests.test_mode_members` — Checks that `Mode` has exactly the declared automatic, exact, and Monte Carlo options.
- `DistributionBaseTests.test_constructor_preserves_parameters_and_builds_constant_nodes` — Checks that a distribution keeps public parameters while creating corresponding constant parameter nodes.
- `DistributionBaseTests.test_distribution_uses_mathematical_support_as_default_realization_support` — Checks that distributions use their mathematical support for runtime validation unless they declare a separate realization support.
- `DistributionBaseTests.test_parameter_to_node_preserves_existing_node` — Checks that an already internal node is reused rather than wrapped again.
- `DistributionBaseTests.test_parameter_to_node_uses_a_random_variable_node_without_changing_public_input` — Checks that a random-variable parameter uses its internal node without replacing the public parameter object.
- `DistributionBaseTests.test_node_dependencies_include_variable_parameters_but_not_constants` — Checks that only non-constant parameter nodes become distribution dependencies.
- `DistributionBaseTests.test_sample_creates_context_and_evaluates_distribution_node` — Checks that `Distribution.sample()` creates a realization context and evaluates its root distribution node.
- `DistributionBaseTests.test_sample_passes_count_rng_and_validation_to_context` — Checks that public distribution sampling forwards count, generator, and validation arguments to the realization context.
- `DistributionBaseTests.test_evaluate_realizes_parameter_nodes_and_delegates_to_sample` — Checks that internal distribution evaluation realizes parameters before invoking `_sample()`.
- `DistributionBaseTests.test_exact_statistics_are_used_for_exact_and_auto_modes` — Checks that available exact mean and variance values are used in exact and automatic modes.
- `DistributionBaseTests.test_monte_carlo_statistics_delegate_to_monte_carlo` — Checks that mean, variance, and standard deviation delegate to Monte Carlo in Monte Carlo mode.
- `DistributionBaseTests.test_auto_statistics_fall_back_to_monte_carlo_when_exact_values_are_unavailable` — Checks that automatic statistics use Monte Carlo when an exact hook returns no result.
- `DistributionBaseTests.test_exact_mode_rejects_unavailable_statistics` — Checks that exact statistical methods report an unavailable exact implementation.
- `DistributionBaseTests.test_cdf_and_ppf_use_exact_values_when_available` — Checks that CDF and PPF use available exact-hook results in exact and automatic modes.
- `DistributionBaseTests.test_cdf_and_ppf_delegate_to_monte_carlo_when_requested` — Checks that explicitly requested Monte Carlo CDF and PPF use sampling.
- `DistributionBaseTests.test_auto_cdf_and_ppf_fall_back_to_monte_carlo_when_exact_values_are_unavailable` — Checks automatic CDF and PPF fallback when exact hooks have no answer.
- `DistributionBaseTests.test_monte_carlo_cdf_operations_handle_scalar_and_array_inputs` — Checks Monte Carlo CDF calculation for both scalar thresholds and threshold arrays.
- `DistributionBaseTests.test_monte_carlo_ppf_operation_uses_requested_quantile_method` — Checks that Monte Carlo PPF uses the requested NumPy quantile method.
- `DistributionBaseTests.test_monte_carlo_realizes_samples_and_normalizes_scalar_results` — Checks sampling through a context and scalar-result normalization in Monte Carlo helpers.
- `DistributionBaseTests.test_monte_carlo_preserves_array_results` — Checks that Monte Carlo helpers preserve a vector or array result.
- `DistributionBaseTests.test_quantile_confidence_interval_realizes_samples_and_delegates` — Checks that a distribution realizes samples and passes them with parameters to quantile-interval calculation.
- `DistributionBaseTests.test_validation_rejects_invalid_distribution_arguments` — Checks validation of mode, sample count, RNG, CDF input, PPF input, and quantile method arguments.
- `DistributionSubclassTests.test_continuous_and_discrete_base_classes_remain_abstract` — Checks that the specialized continuous and discrete base classes cannot be instantiated directly.

### `tests/unit_tests/distributions/continuous/normal.py`

- `NormalDistributionTests.test_configuration_and_public_properties` — Checks normal-distribution parameters, symbol, name, and real-valued support.
- `NormalDistributionTests.test_sample_delegates_to_scipy_with_parameters` — Checks that normal `_sample()` passes mean, standard deviation, sample count, and generator to SciPy.
- `NormalDistributionTests.test_invalid_mean_and_standard_deviation_are_rejected` — Checks normal constructor validation for invalid means and standard deviations.

### `tests/unit_tests/distributions/discrete/binomial.py`

- `BinomialDistributionTests.test_configuration_and_symbolic_finite_support` — Checks binomial parameters, name, and symbolic finite support for a fixed trial count.
- `BinomialDistributionTests.test_zero_n_has_single_value_support` — Checks that a binomial distribution with zero trials has only zero in its support.
- `BinomialDistributionTests.test_sample_delegates_to_scipy_with_parameters` — Checks that binomial `_sample()` forwards parameters, sample count, and generator to SciPy.
- `BinomialDistributionTests.test_invalid_parameters_are_rejected` — Checks binomial constructor validation for invalid trial counts and probabilities.

### `tests/unit_tests/distributions/discrete/categorical.py`

- `CategoricalDistributionTests.test_numeric_categories_are_computed_once_and_reused_for_sampling` — Checks that categorical numeric conversion runs once during construction and its cached array is reused by sampling.
- `CategoricalDistributionTests.test_lossy_conversion_falls_back_to_exact_object_samples` — Checks that large integers that cannot share a lossless native dtype remain exact object samples, with object storage declared by realization support.
- `CategoricalDistributionTests.test_realization_support_is_cached_and_matches_numeric_samples` — Checks that realization support is cached and matches numeric samples.
- `CategoricalDistributionTests.test_non_finite_categories_get_explicit_realization_permissions` — Checks that non finite categories get explicit realization permissions.
- `CategoricalDistributionTests.test_object_categories_reuse_mathematical_support_for_realizations` — Checks that object categories reuse mathematical support for realizations.
- `CategoricalDistributionTests.test_constructor_preserves_categories_and_probabilities` — Checks that categorical construction retains configured categories and probabilities.
- `CategoricalDistributionTests.test_equal_categories_are_merged_and_probabilities_are_summed` — Checks that equal categorical outcomes merge and their probabilities are summed.
- `CategoricalDistributionTests.test_numeric_categories_use_numeric_value_set` — Checks that homogeneous numeric categories receive a numeric value set.
- `CategoricalDistributionTests.test_mixed_numeric_categories_preserve_inputs_and_sample_lossless_numeric_array` — Checks that the public category inputs keep their original Python types while sampling uses a lossless native float64 array.
- `CategoricalDistributionTests.test_sample_uses_rng_indices_and_configured_probabilities` — Checks that categorical `_sample()` uses chosen indices and configured probability weights.
- `CategoricalDistributionTests.test_configuration_validation_rejects_invalid_inputs` — Checks categorical configuration validation for invalid categories and probability lists.
- `CategoricalDistributionTests.test_fraction_and_integer_categories_use_mixed_numeric_values` — Checks that fractions combined with integers use the mixed numeric value-set representation.

### `tests/unit_tests/distributions/discrete/helpers/_categorical.py`

- `CategoricalHelperTests.test_lossless_numeric_conversion_accepts_exact_value_preserving_promotions` — Checks exact equality after integer, Fraction, Decimal, float32, and complex promotions that can preserve their values.
- `CategoricalHelperTests.test_lossless_numeric_conversion_rejects_rounding_of_integers_fractions_and_float_components` — Checks that conversion rejects rounded large integers, non-binary fractions and decimals, and rounded real or imaginary components.
- `CategoricalHelperTests.test_lossless_numeric_conversion_distinguishes_infinity_sign_and_matches_nan` — Checks that conversion preserves signed infinity and NaN, including complex components, while rejecting changed values.
- `CategoricalHelperTests.test_lossless_numeric_conversion_checks_shape_and_rejects_nonnumeric_objects` — Checks that lossless numeric conversion checks shape and rejects nonnumeric objects.
- `CategoricalHelperTests.test_attempt_numeric_conversion_returns_readonly_native_array_for_safe_mixture` — Checks that attempt numeric conversion returns readonly native array for safe mixture.
- `CategoricalHelperTests.test_attempt_numeric_conversion_retains_objects_when_conversion_is_lossy_or_unsupported` — Checks that attempt numeric conversion retains objects when conversion is lossy or unsupported.
- `CategoricalHelperTests.test_homogeneous_large_integer_categories_fall_back_without_precision_loss` — Checks that homogeneous large integer categories fall back without precision loss.
- `CategoricalHelperTests.test_homogeneous_numpy_categories_preserve_original_precision` — Checks that homogeneous NumPy categories preserve original precision.
- `CategoricalHelperTests.test_numeric_category_detection` — Checks detection of category collections that are numeric.
- `CategoricalHelperTests.test_untyped_categories_remain_atomic_objects` — Checks that untyped categories are stored as atomic object values.
- `CategoricalHelperTests.test_mixed_numeric_categories_remain_atomic_objects` — Checks that mixed numeric categories remain separate atomic category values.
- `CategoricalHelperTests.test_inference_selects_homogeneous_mixed_or_object_value_sets` — Checks categorical inference chooses homogeneous numeric, mixed numeric, or object value sets appropriately.
- `CategoricalHelperTests.test_equal_categories_are_merged_in_order` — Checks equal categories merge while preserving their first-occurrence order.

### `tests/unit_tests/distributions/discrete/poisson.py`

- `PoissonDistributionTests.test_configuration_and_support` — Checks Poisson parameters, name, and natural-number support.
- `PoissonDistributionTests.test_sample_delegates_to_scipy_with_parameters` — Checks that Poisson `_sample()` forwards rate, sample count, and generator to SciPy.
- `PoissonDistributionTests.test_invalid_rate_is_rejected` — Checks Poisson constructor validation for invalid rates.

### `tests/unit_tests/functions/_utils.py`

- `FunctionUtilityTests.test_apply_checks_named_nodes_before_executing_scalar_callback` — Checks that apply checks named nodes before executing scalar callback.
- `FunctionUtilityTests.test_apply_passes_all_nodes_to_dtype_check_for_multiple_random_variables` — Checks that apply passes all nodes to dtype check for multiple random variables.
- `FunctionUtilityTests.test_apply_rejects_object_storage_before_ufunc_execution` — Checks that apply rejects object storage before ufunc execution.
- `FunctionUtilityTests.test_apply_converts_scalar_operation_result_to_float` — Checks the descriptor-based application helper converts scalar numerical results to `float`.
- `FunctionUtilityTests.test_apply_delegates_random_variable_to_private_operation_method` — Checks the helper delegates random-variable inputs to `_apply_operation()` with the shared descriptor.
- `FunctionUtilityTests.test_apply_builds_node_for_mixed_scalar_and_random_variable_inputs` — Checks the helper builds one operation node when an operation combines scalar and random-variable inputs.

### `tests/unit_tests/functions/basic.py`

- `BasicFunctionTests.test_sqrt_validates_non_negative_domain_and_applies_sqrt_operation` — Checks square root domain validation, shared operation selection, and declared output support.
- `BasicFunctionTests.test_absolute_validates_real_input_and_declares_non_negative_output` — Checks absolute value validates real inputs and declares non-negative support.
- `BasicFunctionTests.test_floor_validates_real_input_and_declares_integer_output` — Checks floor validates real inputs and declares integer support.
- `BasicFunctionTests.test_ceil_validates_real_input_and_declares_integer_output` — Checks ceiling validates real inputs and declares integer support.
- `BasicFunctionTests.test_sign_validates_real_input_and_declares_three_possible_outputs` — Checks sign validation and its three-element output support.
- `BasicFunctionTests.test_hypot_validates_both_inputs_and_declares_non_negative_output` — Checks `hypot` validates both real inputs and declares non-negative output support.

### `tests/unit_tests/functions/exponential.py`

- `ExponentialFunctionTests.test_exp_declares_positive_math_support_and_non_negative_realization_support` — Checks that exponential output is mathematically positive while machine realizations may include zero.
- `ExponentialFunctionTests.test_log_validates_positive_domain_and_applies_log_operation` — Checks natural-log domain validation and shared operation selection.
- `ExponentialFunctionTests.test_log2_validates_positive_domain_and_applies_log2_operation` — Checks base-two-log domain validation and shared operation selection.
- `ExponentialFunctionTests.test_log10_validates_positive_domain_and_applies_log10_operation` — Checks base-ten-log domain validation and shared operation selection.
- `ExponentialFunctionTests.test_log1p_validates_domain_and_applies_log1p_operation` — Checks `log1p` requires inputs greater than negative one and selects its stable operation descriptor.
- `ExponentialFunctionTests.test_expm1_validates_real_input_and_declares_its_range` — Checks `expm1` accepts real inputs and declares its open mathematical and closed realization lower bounds.
- `ExponentialFunctionTests.test_logaddexp_validates_both_inputs_and_applies_binary_operation` — Checks `logaddexp` validates both real inputs and uses its shared binary descriptor.

### `tests/unit_tests/functions/trigonometric.py`

- `TrigonometricFunctionTests.test_sin_validates_real_input_and_declares_closed_unit_interval` — Checks sine validation and its closed unit-interval support.
- `TrigonometricFunctionTests.test_cos_validates_real_input_and_declares_closed_unit_interval` — Checks cosine validation and its closed unit-interval support.
- `TrigonometricFunctionTests.test_tan_validates_real_input_and_declares_real_output` — Checks tangent validation and real-valued output support.
- `TrigonometricFunctionTests.test_arcsin_validates_closed_unit_interval_and_declares_output_range` — Checks arcsine input validation and declared output interval.
- `TrigonometricFunctionTests.test_arccos_validates_closed_unit_interval_and_declares_output_range` — Checks arccosine input validation and declared output interval.
- `TrigonometricFunctionTests.test_arctan_declares_open_math_range_and_closed_realization_range` — Checks that arctangent has an open mathematical range and a closed machine-realization range.
- `TrigonometricFunctionTests.test_sinh_validates_real_input_and_declares_real_output` — Checks hyperbolic sine validation and real-valued output support.
- `TrigonometricFunctionTests.test_cosh_validates_real_input_and_declares_output_at_least_one` — Checks hyperbolic cosine validation and lower-bounded output support.
- `TrigonometricFunctionTests.test_tanh_declares_open_math_range_and_closed_realization_range` — Checks that hyperbolic tangent has an open mathematical range and a closed machine-realization range.
- `TrigonometricFunctionTests.test_arcsinh_validates_real_input_and_declares_real_output` — Checks inverse hyperbolic sine validation and real output support.
- `TrigonometricFunctionTests.test_arccosh_validates_lower_bound_and_declares_non_negative_output` — Checks inverse hyperbolic cosine lower-bound validation and non-negative output support.
- `TrigonometricFunctionTests.test_arctanh_validates_open_unit_interval_and_declares_real_output` — Checks inverse hyperbolic tangent domain validation and real output support.

### `tests/unit_tests/operations/_arithmetic.py`

- `ArithmeticOperationTests.test_descriptors_connect_each_operation_to_both_support_inferers` — Checks that descriptors connect each operation to both support inferers.
- `ArithmeticOperationTests.test_bounded_integer_arithmetic_preserves_dtype_and_broadcasts` — Checks that bounded integer arithmetic preserves dtype and broadcasts.
- `ArithmeticOperationTests.test_integer_overflow_raises_for_binary_and_unary_operations` — Checks that integer overflow raises for binary and unary operations.
- `ArithmeticOperationTests.test_integer_overflow_warns_and_converts_entire_array_to_float64` — Checks that integer overflow warns and converts entire array to float64.
- `ArithmeticOperationTests.test_integer_overflow_ignore_returns_infinity_without_warning` — Checks that integer overflow ignore returns infinity without warning.
- `ArithmeticOperationTests.test_no_overflow_preserves_integers_too_large_for_float64` — Checks that no overflow preserves integers too large for float64.
- `ArithmeticOperationTests.test_overflow_fallback_can_round_unaffected_large_integers_as_warning_states` — Checks the documented whole-array float64 fallback, including rounding of a large non-overflowing integer when another entry overflows.
- `ArithmeticOperationTests.test_overflow_conversion_handles_scalar_and_empty_results` — Checks that overflow conversion handles scalar and empty results.
- `ArithmeticOperationTests.test_mixed_signed_unsigned_inputs_follow_numpy_float_promotion` — Checks that mixed signed unsigned inputs follow NumPy float promotion.
- `ArithmeticOperationTests.test_floating_overflow_honors_ambient_numpy_policy` — Checks that floating overflow honors ambient NumPy policy.
- `ArithmeticOperationTests.test_division_distinguishes_zero_division_from_invalid_zero_over_zero` — Checks that division distinguishes zero division from invalid zero over zero.
- `ArithmeticOperationTests.test_modulo_zero_is_nan_when_errors_are_ignored` — Checks that modulo zero is NaN when errors are ignored.
- `ArithmeticOperationTests.test_negative_integer_exponents_promote_base_to_floating` — Checks that negative integer exponents promote base to floating.
- `ArithmeticOperationTests.test_ordinary_power_supports_complex_roots_and_real_power_rejects_them` — Checks that ordinary power supports complex roots and real power rejects them.
- `ArithmeticOperationTests.test_unary_arithmetic_names_and_complex_absolute_values` — Checks that unary arithmetic names and complex absolute values.

### `tests/unit_tests/operations/_base.py`

- `OperationBaseTests.test_descriptor_retains_callable_name_and_optional_dtype_requirements` — Checks that descriptor retains callable name and optional dtype requirements.
- `OperationBaseTests.test_descriptor_is_immutable` — Checks that descriptor is immutable.
- `OperationBaseTests.test_descriptor_constructor_requires_keywords` — Checks that descriptor constructor requires keywords.
- `OperationBaseTests.test_shared_dtype_rules_describe_numeric_real_and_boolean_inputs` — Checks that shared dtype rules describe numeric real and boolean inputs.

### `tests/unit_tests/operations/_comparison.py`

- `ComparisonOperationTests.test_comparisons_apply_elementwise_with_correct_names` — Checks that comparisons apply elementwise with correct names.
- `ComparisonOperationTests.test_ordered_comparisons_support_real_and_object_dtypes` — Checks that ordered comparisons support real and object dtypes.
- `ComparisonOperationTests.test_equality_has_no_dtype_restriction_and_accepts_objects` — Checks that equality has no dtype restriction and accepts objects.

### `tests/unit_tests/operations/_function.py`

- `FunctionOperationTests.test_predefined_descriptors_connect_all_functions_to_their_inferers` — Checks that predefined descriptors connect all functions to their inferers.
- `FunctionOperationTests.test_custom_descriptor_does_not_require_an_inference_hook` — Checks that custom descriptor does not require an inference hook.

### `tests/unit_tests/operations/_logical.py`

- `LogicalOperationTests.test_binary_operations_cover_complete_boolean_truth_tables` — Checks that binary operations cover complete boolean truth tables.
- `LogicalOperationTests.test_inversion_uses_boolean_dtype_requirement_and_name` — Checks that inversion uses boolean dtype requirement and name.

### `tests/unit_tests/operations/_statistical.py`

- `StatisticalOperationTests.test_descriptors_use_population_statistics_and_numeric_dtype_rules` — Checks that descriptors use population statistics and numeric dtype rules.
- `StatisticalOperationTests.test_complex_mean_is_complex_while_variance_and_std_are_real` — Checks that complex mean is complex while variance and std are real.

### `tests/unit_tests/probability/_config.py`

- `ProbabilityConfigTests.test_default_sample_count_is_within_configured_limit` — Checks that the default probability sample count respects the configured maximum.

### `tests/unit_tests/probability/intervals.py`

- `ConfidenceIntervalTests.test_constructor_preserves_configuration` — Checks that a confidence interval preserves its bounds and alpha configuration.
- `ConfidenceIntervalTests.test_constructor_allows_two_nan_bounds_for_an_undefined_interval` — Checks that an undefined confidence interval may use two NaN bounds.
- `ConfidenceIntervalTests.test_constructor_rejects_invalid_configuration` — Checks probability-interval constructor validation for invalid bounds, alpha, and estimate flags.
- `ProbabilityIntervalTests.test_properties_report_nominal_coverage_and_bounds` — Checks that a probability interval reports its nominal coverage and configured bounds.
- `ProbabilityIntervalTests.test_constructor_rejects_invalid_configuration` — Checks probability-interval constructor validation for invalid bounds, alpha, and estimate flags.

### `tests/unit_tests/probability/probability.py`

- `ProbabilityFunctionTests.test_unconditional_probability_counts_true_values` — Checks unconditional `P` counts true event realizations and returns matching metadata.
- `ProbabilityFunctionTests.test_conditional_probability_counts_joint_successes_within_condition` — Checks conditional `P` counts joint successes over only conditioned realizations.
- `ProbabilityFunctionTests.test_conditional_probability_returns_nan_when_condition_never_occurs` — Checks conditional `P` returns NaN when the conditioning event has no realizations.
- `ProbabilityFunctionTests.test_rejects_invalid_public_arguments` — Checks public probability validation for event, condition, sample count, generator, and validation arguments.

### `tests/unit_tests/probability/results.py`

- `ProbabilityResultTests.test_num_samples_uses_unconditioned_count_without_condition` — Checks `ProbabilityResult.num_samples` selects the unconditioned count when there is no condition.
- `ProbabilityResultTests.test_num_samples_uses_conditioned_count_when_present` — Checks `ProbabilityResult.num_samples` selects the conditioned count when present.
- `ProbabilityResultTests.test_confidence_interval_delegates_counts_and_alpha` — Checks confidence-interval creation delegates the effective count, successes, and alpha correctly.
- `ProbabilityResultTests.test_constructor_allows_nan_for_zero_conditioned_samples` — Checks that zero conditioned samples may be represented by a NaN probability.
- `ProbabilityResultTests.test_constructor_rejects_invalid_configuration` — Checks `ProbabilityResult` construction rejects invalid counts and probability values.

### `tests/unit_tests/random_variables/_config.py`

- `RandomVariableConfigTests.test_default_graph_size_is_positive` — Checks that the default dependency-graph size limit is positive.

### `tests/unit_tests/random_variables/_context.py`

- `RealizationContextTests.test_context_stores_default_and_explicit_numerical_error_policy` — Checks that context stores default and explicit numerical error policy.
- `RealizationContextTests.test_duplicate_requests_preserve_results_without_duplicate_evaluation` — Checks that duplicate requests preserve results without duplicate evaluation.
- `RealizationContextTests.test_validate_uses_declared_exceptional_permissions_independently_of_policy` — Checks that validate uses declared exceptional permissions independently of policy.
- `RealizationContextTests.test_failed_parent_evaluation_does_not_release_dependencies_or_cache_failed_result` — Checks that failed parent evaluation does not release dependencies or cache failed result.
- `RealizationContextTests.test_constructor_rejects_empty_requested_nodes` — Checks that constructor rejects empty requested nodes.
- `RealizationContextTests.test_constructor_applies_graph_limit_to_all_requested_nodes` — Checks that constructor applies graph limit to all requested nodes.
- `RealizationContextTests.test_evaluate_preserves_requested_dependency_in_either_order` — Checks that evaluate preserves requested dependency in either order.
- `RealizationContextTests.test_evaluate_releases_shared_dependency_only_after_both_roots` — Checks that evaluate releases shared dependency only after both roots.
- `RealizationContextTests.test_evaluate_duplicate_requested_nodes_does_not_resample` — Checks that evaluate duplicate requested nodes does not resample.
- `RealizationContextTests.test_constructor_stores_sample_count_and_supplied_rng` — Checks that a realization context retains the requested sample count and supplied generator.
- `RealizationContextTests.test_constructor_creates_a_generator_when_rng_is_not_supplied` — Checks that a realization context creates a NumPy generator when none is supplied.
- `RealizationContextTests.test_constructor_rejects_invalid_sample_count` — Checks that invalid realization-context sample counts are rejected.
- `RealizationContextTests.test_evaluate_caches_node_result` — Checks that evaluating the same node twice reuses its cached result.
- `RealizationContextTests.test_evaluate_broadcasts_scalar_result_to_sample_count` — Checks that scalar node results broadcast to the configured sample length.
- `RealizationContextTests.test_evaluate_rejects_wrong_sample_shape` — Checks that a node result with an incompatible sample shape is rejected.
- `RealizationContextTests.test_evaluate_rejects_dtype_outside_declared_family` — Checks that a node result outside its declared dtype family is rejected.
- `RealizationContextTests.test_evaluate_accepts_any_dtype_when_value_set_has_no_dtype_family` — Checks that an unspecified dtype family accepts any result dtype.
- `RealizationContextTests.test_evaluate_validates_result_when_requested` — Checks that realization validation calls subset validation when requested.
- `RealizationContextTests.test_evaluate_uses_realization_support_for_runtime_validation` — Checks that runtime validation uses the node’s realization support rather than its mathematical support.
- `RealizationContextTests.test_evaluate_releases_dependency_after_last_dependant_is_realized` — Checks that a cached dependency is released after its last dependent has been realized.
- `RealizationContextTests.test_constructor_rejects_incomplete_dependency_graph` — Checks that a realization context rejects a graph truncated by its maximum-size limit.

### `tests/unit_tests/random_variables/_nodes.py`

- `NodeCompatibilityImportTests.test_compatibility_module_reexports_current_node_types` — Checks that the compatibility module re-exports the current internal node classes.

### `tests/unit_tests/random_variables/base.py`

- `RandomVariableBaseTests.test_sample_forwards_each_policy_and_validates_before_creating_context` — Checks that sample forwards each policy and validates before creating context.
- `RandomVariableBaseTests.test_binary_and_unary_operations_use_distinct_support_inference_inputs` — Checks that binary and unary operations use distinct support inference inputs.
- `RandomVariableBaseTests.test_ordered_comparisons_reject_complex_realizations_while_equality_allows_them` — Checks that ordered comparisons reject complex realizations while equality allows them.
- `RandomVariableBaseTests.test_constructor_creates_distribution_node_with_explicit_name` — Checks that random-variable construction creates a distribution node with the supplied name.
- `RandomVariableBaseTests.test_from_node_preserves_node_and_explicit_name` — Checks the internal node constructor retains the given node and explicit name.
- `RandomVariableBaseTests.test_generated_names_are_distinct_when_not_supplied` — Checks generated random-variable names are distinct when no name is supplied.
- `RandomVariableBaseTests.test_dependency_graph_uses_default_graph_limit` — Checks `dependency_graph` constructs a graph using the default maximum size.
- `RandomVariableBaseTests.test_plot_dependencies_delegates_to_graph` — Checks dependency plotting delegates to the graph with the supplied maximum size.
- `RandomVariableBaseTests.test_sample_creates_context_and_evaluates_own_node` — Checks sampling creates a realization context and evaluates the variable’s own node.
- `RandomVariableBaseTests.test_is_real_or_complex_reflects_declared_value_set` — Checks numeric capability detection reflects the node’s declared value set.
- `RandomVariableBaseTests.test_realize_delegates_to_sample_defaults` — Checks `realize()` calls sampling with its default arguments.
- `RandomVariableBaseTests.test_binary_operation_builds_operation_node_with_scalar_constant` — Checks a binary operation wraps a scalar operand as a constant node and builds the correct operation node.
- `RandomVariableBaseTests.test_reverse_binary_operation_reverses_input_order` — Checks reverse arithmetic uses the scalar node as the left input.
- `RandomVariableBaseTests.test_public_arithmetic_methods_delegate_with_correct_operations` — Checks public arithmetic operators choose their corresponding internal operations.
- `RandomVariableBaseTests.test_binary_operation_rejects_unsupported_operand` — Checks binary operations reject operands of unsupported Python types.
- `RandomVariableBaseTests.test_binary_operation_rejects_non_numeric_value_set` — Checks binary operations reject random variables with nonnumeric value sets.
- `RandomVariableBaseTests.test_real_power_uses_real_power_operation_descriptor` — Checks powers inferred as real use the shared real-power operation descriptor.
- `RandomVariableBaseTests.test_unary_operation_builds_node` — Checks a unary operation creates an operation node with expected input and value set.
- `RandomVariableBaseTests.test_binary_operation_reuses_node_when_additive_identity_applies` — Checks arithmetic construction reuses the original node for `X + 0`.
- `RandomVariableBaseTests.test_unary_operation_reuses_node_when_double_negation_applies` — Checks unary construction reuses the original node for `-(-X)`.
- `RandomVariableBaseTests.test_predefined_function_operations_use_simplifier` — Checks predefined `log(exp(X))` construction returns the original node.
- `RandomVariableBaseTests.test_public_unary_methods_delegate_with_correct_operations` — Checks public unary operators choose their corresponding internal operations.
- `RandomVariableBaseTests.test_apply_wraps_non_vectorized_function_for_sample_arrays` — Checks `apply()` adapts a scalar function to aligned sample arrays.
- `RandomVariableBaseTests.test_apply_preserves_vectorized_function` — Checks `apply()` passes a vectorized function through unchanged.
- `RandomVariableBaseTests.test_apply_defaults_unknown_supports_when_no_supports_are_given` — Checks `apply()` uses unknown mathematical and realization supports when neither support is supplied.
- `RandomVariableBaseTests.test_apply_defaults_realization_support_to_mathematical_support` — Checks `apply()` uses the mathematical support as the realization support when only the mathematical support is supplied.
- `RandomVariableBaseTests.test_apply_passes_aligned_values_from_other_variables` — Checks `apply()` passes aligned realizations from every additional random variable.
- `RandomVariableBaseTests.test_interval_uses_inverted_cdf_quantiles_of_samples` — Checks random-variable interval estimation uses inverted-CDF sample quantiles.
- `RandomVariableBaseTests.test_is_in_interval_rejects_reversed_bounds` — Checks interval membership rejects a lower bound greater than the upper bound.
- `RandomVariableBaseTests.test_is_in_interval_honors_each_closure_mode` — Checks interval membership creates the correct comparisons for all four closure modes.
- `RandomVariableBaseTests.test_is_in_delegates_tuple_and_list_ranges_with_expected_closure` — Checks tuple and list range shorthand choose open and closed interval semantics respectively.
- `RandomVariableBaseTests.test_is_in_builds_boolean_event_for_sympy_set` — Checks SymPy set membership builds a Boolean event node that evaluates correctly.
- `RandomVariableBaseTests.test_comparison_builds_boolean_event` — Checks comparison with a scalar builds a Boolean event node.
- `RandomVariableBaseTests.test_public_comparison_methods_delegate_with_correct_operations` — Checks public comparison operators choose their corresponding internal comparisons.
- `RandomVariableBaseTests.test_quantile_confidence_interval_delegates_sample_and_parameters` — Checks quantile confidence intervals sample once and pass the requested quantile and alpha parameters onward.
- `RandomVariableBaseTests.test_real_only_methods_reject_non_real_value_sets` — Checks interval and quantile methods reject variables with non-real supports.

### `tests/unit_tests/random_variables/graph.py`

- `NodeGraphTests.test_graph_unites_disconnected_roots` — Checks that graph unites disconnected roots.
- `NodeGraphTests.test_graph_counts_shared_dependencies_once_across_roots` — Checks that graph counts shared dependencies once across roots.
- `NodeGraphTests.test_graph_limit_applies_to_union_of_roots` — Checks that graph limit applies to union of roots.
- `NodeGraphTests.test_constructor_rejects_boolean_and_non_integer_limits` — Checks that constructor rejects boolean and non integer limits.
- `NodeGraphTests.test_constructor_accepts_numpy_integer_limit` — Checks that constructor accepts NumPy integer limit.
- `NodeGraphTests.test_graph_contains_nodes_and_edges_from_root_to_dependencies` — Checks a graph contains every reachable dependency and root-to-dependency edge.
- `NodeGraphTests.test_graph_stops_when_maximum_size_is_reached` — Checks graph construction marks itself incomplete when the maximum node count is reached.
- `NodeGraphTests.test_constructor_rejects_non_positive_maximum_size` — Checks graph construction rejects non-positive maximum sizes.
- `NodeGraphTests.test_graph_handles_cycles_without_duplicate_nodes_or_recursion` — Checks cyclic dependency input does not produce duplicate nodes or unbounded recursion.
- `NodeGraphTests.test_plot_uses_selected_node_labels` — Checks plotting uses standard or extended labels according to its option.

### `tests/unit_tests/random_variables/nodes/_simplification.py`

- `SimplificationTests.test_boolean_idempotence_and_double_negation_reuse_original_node` — Checks that boolean idempotence and double negation reuse original node.
- `SimplificationTests.test_boolean_complements_simplify_in_both_orders` — Checks that boolean complements simplify in both orders.
- `SimplificationTests.test_boolean_rules_use_node_identity_rather_than_equal_names` — Checks that boolean rules use node identity rather than equal names.
- `SimplificationTests.test_function_factory_passes_realization_sets_to_inferer_and_keeps_math_support` — Checks that function factory passes realization sets to inferer and keeps math support.
- `SimplificationTests.test_numeric_function_inference_rejects_object_input_or_output_support` — Checks that numeric function inference rejects object input or output support.
- `SimplificationTests.test_custom_function_without_inferer_preserves_supplied_supports` — Checks that custom function without inferer preserves supplied supports.
- `SimplificationTests.test_creates_operation_node_when_no_rule_applies` — Checks the simplification entry point creates an operation node when no identity applies.
- `SimplificationTests.test_addition_by_zero_returns_original_node` — Checks additive identity simplification reuses the original node.
- `SimplificationTests.test_zero_plus_addition_returns_original_node` — Checks the left additive identity reuses the original node.
- `SimplificationTests.test_subtraction_by_zero_returns_original_node` — Checks subtraction by zero reuses the original node.
- `SimplificationTests.test_multiplication_by_one_returns_original_node_in_both_orders` — Checks both multiplicative identity orders reuse the original node.
- `SimplificationTests.test_division_by_one_returns_original_node` — Checks division by one reuses the original node.
- `SimplificationTests.test_power_to_one_returns_base_node_for_both_power_operations` — Checks both power descriptors participate in the exponent-one identity rule.
- `SimplificationTests.test_double_negation_returns_original_node` — Checks double negation returns the original node.
- `SimplificationTests.test_nested_absolute_value_returns_inner_absolute_value_node` — Checks nested absolute value simplifies to the existing inner node.
- `SimplificationTests.test_log_of_exponential_returns_real_input_node` — Checks `log(exp(X))` simplifies for real-valued inputs.
- `SimplificationTests.test_exponential_of_logarithm_returns_positive_input_node` — Checks `exp(log(X))` simplifies for positive-valued inputs.
- `SimplificationTests.test_square_root_of_real_square_creates_absolute_value_node` — Checks `sqrt(X ** 2)` produces an absolute-value node for real inputs.
- `SimplificationTests.test_square_root_of_absolute_value_square_reuses_absolute_value_node` — Checks recursively created replacements are simplified before being returned.
- `SimplificationTests.test_sine_of_arcsine_returns_input_in_closed_unit_interval` — Checks `sin(arcsin(X))` returns `X` when its support is within the closed interval from negative one to one.
- `SimplificationTests.test_cosine_of_arccosine_returns_input_in_closed_unit_interval` — Checks `cos(arccos(X))` returns `X` when its support is within the closed interval from negative one to one.
- `SimplificationTests.test_tangent_of_arctangent_returns_real_input` — Checks `tan(arctan(X))` returns a real-valued `X`.
- `SimplificationTests.test_logarithm_of_one_plus_input_creates_log1p_node` — Checks `log(1 + X)` is replaced by the numerically stable `log1p(X)` node.
- `SimplificationTests.test_exponential_minus_one_creates_expm1_node` — Checks `exp(X) - 1` is replaced by the numerically stable `expm1(X)` node.
- `SimplificationTests.test_square_root_of_sum_of_squares_creates_hypot_node` — Checks `sqrt(X ** 2 + Y ** 2)` is replaced by the numerically stable `hypot(X, Y)` node.
- `SimplificationTests.test_logarithm_of_sum_of_exponentials_creates_logaddexp_node` — Checks `log(exp(X) + exp(Y))` is replaced by the numerically stable `logaddexp(X, Y)` node.
- `SimplificationTests.test_logarithm_of_one_plus_exponential_creates_softplus_node` — Checks `log(1 + exp(X))` is replaced by `logaddexp(0, X)`.

### `tests/unit_tests/random_variables/nodes/_utils.py`

- `NodeUtilityTests.test_constant_array_keeps_scalar_array` — Checks scalar constants are stored as scalar NumPy arrays.
- `NodeUtilityTests.test_constant_array_keeps_compound_value_atomic` — Checks compound constants become one atomic object-array item.
- `NodeUtilityTests.test_constant_value_set_for_numeric_value_uses_sympy_and_dtype` — Checks numeric constants infer a SymPy support and compatible dtype family.
- `NodeUtilityTests.test_constant_value_set_for_unrepresentable_object_preserves_object` — Checks non-SymPy objects retain object membership in their constant value set.
- `NodeUtilityTests.test_node_type_predicates_recognize_matching_constant_and_operation_nodes` — Checks node predicates recognize constants, operations, and real-power square expressions.

### `tests/unit_tests/random_variables/nodes/base.py`

- `NodeBaseTests.test_base_node_remains_abstract` — Checks that the internal base node cannot be instantiated directly.
- `NodeBaseTests.test_name_and_extended_name_are_exposed` — Checks nodes expose their normal and extended names.
- `NodeBaseTests.test_has_dependencies_reflects_direct_dependencies` — Checks dependency presence reflects whether a node has direct dependencies.

### `tests/unit_tests/random_variables/nodes/constant.py`

- `ConstantNodeTests.test_realization_support_is_cached_and_records_actual_exceptional_values` — Checks that realization support is cached and records actual exceptional values.
- `ConstantNodeTests.test_object_constant_reuses_object_support_without_numeric_detection` — Checks that object constant reuses object support without numeric detection.
- `ConstantNodeTests.test_numeric_constant_exposes_value_name_and_empty_dependencies` — Checks a numeric constant exposes its value, name, value set, and no dependencies.
- `ConstantNodeTests.test_evaluate_returns_scalar_array_for_scalar_value` — Checks constant-node evaluation returns its scalar array representation.
- `ConstantNodeTests.test_compound_constant_remains_atomic_object` — Checks a compound constant is stored as a single atomic object array value.

### `tests/unit_tests/random_variables/nodes/distribution.py`

- `DistributionNodeTests.test_properties_are_forwarded_from_distribution` — Checks a distribution node forwards its distribution, support, dependencies, and names.
- `DistributionNodeTests.test_evaluate_delegates_to_distribution` — Checks distribution-node evaluation delegates to the wrapped distribution.

### `tests/unit_tests/random_variables/nodes/operation.py`

- `OperationNodeTests.test_evaluation_uses_context_policy_for_overflow_division_and_invalid_values` — Checks that evaluation uses context policy for overflow division and invalid values.
- `OperationNodeTests.test_evaluation_ignores_underflow_even_when_global_policy_raises` — Checks that evaluation ignores underflow even when global policy raises.
- `OperationNodeTests.test_evaluation_restores_numpy_policy_after_arbitrary_callable_error` — Checks that evaluation restores NumPy policy after arbitrary callable error.
- `OperationNodeTests.test_properties_expose_operation_inputs_and_value_set` — Checks an operation node exposes its operation, input nodes, and declared value set.
- `OperationNodeTests.test_properties_expose_distinct_mathematical_and_realization_sets` — Checks an operation node can expose different mathematical and machine-realization supports.
- `OperationNodeTests.test_evaluate_realizes_inputs_then_calls_operation_descriptor` — Checks operation-node evaluation realizes inputs before calling its operation descriptor.

### `tests/unit_tests/statistics/_clopper_pearson.py`

- `ClopperPearsonTests.test_zero_samples_returns_undefined_interval_without_beta_calls` — Checks zero samples produce an undefined interval without requesting beta quantiles.
- `ClopperPearsonTests.test_zero_successes_sets_lower_bound_to_zero` — Checks zero successes force the Clopper–Pearson lower bound to zero.
- `ClopperPearsonTests.test_all_successes_sets_upper_bound_to_one` — Checks all successes force the Clopper–Pearson upper bound to one.
- `ClopperPearsonTests.test_interior_success_count_uses_beta_lower_and_upper_quantiles` — Checks an interior success count uses the correct beta lower and upper quantiles.
- `ClopperPearsonTests.test_rejects_invalid_counts_and_alpha` — Checks Clopper–Pearson calculation rejects invalid counts and alpha values.

### `tests/unit_tests/statistics/_quantiles.py`

- `QuantileConfidenceIntervalTests.test_quantile_method_lists_supported_numpy_methods` — Checks that the quantile-method type lists supported NumPy method names.
- `QuantileConfidenceIntervalTests.test_rejects_invalid_quantile_alpha_and_empty_samples` — Checks quantile intervals reject invalid quantiles, alpha values, and empty samples.
- `QuantileConfidenceIntervalTests.test_rejects_sample_size_below_confidence_requirement` — Checks quantile intervals reject samples below the confidence requirement.
- `QuantileConfidenceIntervalTests.test_uses_outer_order_statistics_when_tail_probabilities_are_large` — Checks that large tail probabilities select outer order statistics.
- `QuantileConfidenceIntervalTests.test_binary_search_selects_tail_boundary_order_statistics` — Checks that binary search selects the order statistics at the lower- and upper-tail boundaries.

### `tests/unit_tests/validation/_common.py`

- `CommonValidationTests.test_numerical_error_policy_accepts_all_modes_and_rejects_invalid_values` — Checks that numerical error policy accepts all modes and rejects invalid values.
- `CommonValidationTests.test_operation_dtype_validation_uses_realizations_and_node_names` — Checks that operation dtype validation uses realizations and node names.
- `CommonValidationTests.test_operation_dtype_validation_broadcasts_single_rule_to_all_nodes` — Checks that operation dtype validation broadcasts single rule to all nodes.
- `CommonValidationTests.test_operation_dtype_validation_applies_separate_rules_and_checks_arity` — Checks that operation dtype validation applies separate rules and checks arity.
- `CommonValidationTests.test_unrestricted_operation_does_not_read_input_dtypes` — Checks that unrestricted operation does not read input dtypes.
- `CommonValidationTests.test_num_samples_accepts_numpy_integer_and_rejects_invalid_values` — Checks sample-count validation accepts NumPy integers and rejects invalid values.
- `CommonValidationTests.test_rng_accepts_generator_or_none` — Checks RNG validation accepts a NumPy generator or `None` only.
- `CommonValidationTests.test_validate_requires_bool` — Checks realization-validation flags must be Python booleans.
- `CommonValidationTests.test_alpha_and_q_require_open_unit_interval` — Checks alpha and quantile arguments require the open interval from zero to one.
- `CommonValidationTests.test_enum_requires_declared_member` — Checks enum validation accepts only members of the requested enum class.
- `CommonValidationTests.test_max_size_requires_positive_integer` — Checks maximum-size validation requires a positive integer.

### `tests/unit_tests/validation/_decorator.py`

- `ParameterDecoratorTests.test_forwarded_arguments_use_explicit_values_and_function_defaults` — Checks that forwarded arguments use explicit values and function defaults.
- `ParameterDecoratorTests.test_forwarded_settings_are_validated_before_dependent_parameters` — Checks that forwarded settings are validated before dependent parameters.
- `ParameterDecoratorTests.test_invalid_forwarded_setting_prevents_parameter_validation_and_body` — Checks that invalid forwarded setting prevents parameter validation and body.
- `ParameterDecoratorTests.test_forwards_only_arguments_explicitly_requested_by_each_validator` — Checks that forwards only arguments explicitly requested by each validator.
- `ParameterDecoratorTests.test_forwards_instance_and_setting_together_without_distribution_specific_names` — Checks that forwards instance and setting together without distribution specific names.
- `ParameterDecoratorTests.test_rejects_unknown_forwarded_argument_when_decorating` — Checks that rejects unknown forwarded argument when decorating.
- `ParameterDecoratorTests.test_rejects_positional_only_forwarded_validator_arguments` — Checks that rejects positional only forwarded validator arguments.
- `ParameterDecoratorTests.test_rejects_positional_only_instance_argument` — Checks that rejects positional only instance argument.
- `ParameterDecoratorTests.test_falls_back_to_value_only_for_callable_without_inspectable_signature` — Checks that falls back to value only for callable without inspectable signature.
- `ParameterDecoratorTests.test_wrapping_preserves_signature_and_rejects_duplicate_or_extra_arguments` — Checks that wrapping preserves signature and rejects duplicate or extra arguments.
- `ParameterDecoratorTests.test_validates_explicit_and_default_parameter_values` — Checks the parameter-validation decorator validates both explicit and default arguments.
- `ParameterDecoratorTests.test_preserves_function_metadata` — Checks the decorator preserves wrapped function metadata.
- `ParameterDecoratorTests.test_rejects_unknown_parameter_and_non_callable_validator` — Checks decorator construction rejects unknown parameters and non-callable validators.
- `ParameterDecoratorTests.test_preserves_normal_argument_binding_errors` — Checks the decorator retains ordinary Python argument-binding errors.
- `ParameterDecoratorTests.test_passes_instance_before_constructor_initializes_it` — Checks that passes instance before constructor initializes it.
- `ParameterDecoratorTests.test_passes_actual_subclass_instance` — Checks that passes actual subclass instance.
- `ParameterDecoratorTests.test_rejects_instance_validator_without_instance_method` — Checks that rejects instance validator without instance method.

### `tests/unit_tests/validation/distributions/_base.py`

- `DistributionBaseValidationTests.test_parameter_risk_policy_accepts_three_modes_and_rejects_invalid_values` — Checks that parameter risk policy accepts three modes and rejects invalid values.
- `DistributionBaseValidationTests.test_parameter_dtype_rules_broadcast_and_report_distribution_and_input_name` — Checks that parameter dtype rules broadcast and report distribution and input name.
- `DistributionBaseValidationTests.test_parameter_dtype_validation_allows_per_parameter_rules_and_rejects_mismatched_arity` — Checks that parameter dtype validation allows per parameter rules and rejects mismatched arity.
- `DistributionBaseValidationTests.test_safe_parameter_support_produces_no_warning_under_any_policy` — Checks that safe parameter support produces no warning under any policy.
- `DistributionBaseValidationTests.test_risky_finite_unknown_and_object_support_follow_risk_policy` — Checks that risky finite unknown and object support follow risk policy.
- `DistributionBaseValidationTests.test_each_exceptional_value_permission_is_checked_independently` — Checks that each exceptional value permission is checked independently.
- `DistributionBaseValidationTests.test_real_input_accepts_real_scalars_and_numeric_non_complex_arrays` — Checks real-input validation accepts real scalars and numeric arrays without complex values.
- `DistributionBaseValidationTests.test_real_input_rejects_bool_text_and_complex_values` — Checks real-input validation rejects booleans, text, and complex values.
- `DistributionBaseValidationTests.test_ppf_input_requires_values_in_closed_unit_interval` — Checks PPF input validation requires values in the closed unit interval.
- `DistributionBaseValidationTests.test_quantile_method_requires_supported_name` — Checks quantile-method validation accepts only supported method names.

### `tests/unit_tests/validation/distributions/continuous/_normal.py`

- `NormalDistributionValidationTests.test_mean_delegates_dtype_and_risk_validation_with_instance_symbol` — Checks that mean delegates dtype and risk validation with instance symbol.
- `NormalDistributionValidationTests.test_std_delegates_dtype_and_risk_validation_with_instance_symbol` — Checks that std delegates dtype and risk validation with instance symbol.
- `NormalDistributionValidationTests.test_numeric_parameters_reject_object_realizations_with_symbol_and_node_name` — Checks that numeric parameters reject object realizations with symbol and node name.
- `NormalDistributionValidationTests.test_mean_accepts_real_scalar_and_rejects_invalid_values` — Checks normal-mean validation accepts real scalars and rejects invalid values.
- `NormalDistributionValidationTests.test_std_requires_positive_real_scalar` — Checks normal standard deviation validation requires a positive real scalar.

### `tests/unit_tests/validation/distributions/discrete/_binomial.py`

- `BinomialDistributionValidationTests.test_n_delegates_dtype_and_risk_validation_with_instance_symbol` — Checks that n delegates dtype and risk validation with instance symbol.
- `BinomialDistributionValidationTests.test_p_delegates_dtype_and_risk_validation_with_instance_symbol` — Checks that p delegates dtype and risk validation with instance symbol.
- `BinomialDistributionValidationTests.test_numeric_parameters_reject_object_realizations_with_symbol_and_node_name` — Checks that numeric parameters reject object realizations with symbol and node name.
- `BinomialDistributionValidationTests.test_n_accepts_integer_support_with_floating_realizations` — Checks that n accepts integer support with floating realizations.
- `BinomialDistributionValidationTests.test_n_accepts_natural_zero_and_rejects_invalid_values` — Checks binomial trial-count validation accepts natural numbers including zero and rejects invalid values.
- `BinomialDistributionValidationTests.test_p_accepts_closed_unit_interval_and_rejects_invalid_values` — Checks binomial probability validation accepts the closed unit interval and rejects invalid values.

### `tests/unit_tests/validation/distributions/discrete/_categorical.py`

- `CategoricalDistributionValidationTests.test_category_and_probability_inputs_require_iterables` — Checks categorical category and probability inputs must be iterable.
- `CategoricalDistributionValidationTests.test_configuration_accepts_matching_normalized_probabilities` — Checks categorical configuration accepts matching categories and normalized probability weights.
- `CategoricalDistributionValidationTests.test_configuration_rejects_invalid_category_probability_pairs` — Checks categorical configuration rejects invalid category and probability combinations.
- `CategoricalDistributionValidationTests.test_configuration_uses_configured_probability_tolerance` — Checks categorical probability totals use the shared configured tolerance.

### `tests/unit_tests/validation/distributions/discrete/_poisson.py`

- `PoissonDistributionValidationTests.test_mu_delegates_dtype_and_risk_validation_with_instance_symbol` — Checks that mu delegates dtype and risk validation with instance symbol.
- `PoissonDistributionValidationTests.test_mu_rejects_object_realizations_with_symbol_and_node_name` — Checks that mu rejects object realizations with symbol and node name.
- `PoissonDistributionValidationTests.test_mu_accepts_non_negative_real_and_rejects_invalid_values` — Checks Poisson rate validation accepts non-negative real values and rejects invalid values.

### `tests/unit_tests/validation/functions/_common.py`

- `FunctionValidationTests.test_real_valued_accepts_real_scalar_and_real_random_variable` — Checks function validation accepts real scalars and random variables with real supports.
- `FunctionValidationTests.test_real_valued_rejects_bool_text_and_object_random_variable` — Checks function validation rejects booleans, text, and object-valued random variables.
- `FunctionValidationTests.test_domain_accepts_member_and_rejects_outside_values` — Checks domain validation accepts members and rejects values outside the given domain.

### `tests/unit_tests/validation/probability/_intervals.py`

- `ProbabilityIntervalValidationTests.test_interval_bound_allows_nan_only_when_requested` — Checks interval-bound validation permits NaN only for explicitly allowed undefined intervals.
- `ProbabilityIntervalValidationTests.test_confidence_interval_configuration_accepts_ordered_or_two_nan_bounds` — Checks confidence-interval validation accepts ordered bounds or the special two-NaN case.
- `ProbabilityIntervalValidationTests.test_probability_interval_configuration_requires_ordered_finite_bounds_and_bool_flag` — Checks probability-interval validation requires ordered finite bounds and a Boolean estimate flag.

### `tests/unit_tests/validation/probability/_probability.py`

- `ProbabilityEventValidationTests.test_event_and_given_accept_event_and_none` — Checks probability event and condition validation accepts events and an optional `None` condition.
- `ProbabilityEventValidationTests.test_event_and_given_reject_other_values` — Checks probability event and condition validation rejects other value types.

### `tests/unit_tests/validation/probability/_results.py`

- `ProbabilityResultValidationTests.test_non_negative_integer_accepts_numpy_integer` — Checks result-count validation accepts NumPy integer values.
- `ProbabilityResultValidationTests.test_configuration_accepts_unconditional_and_zero_conditioned_results` — Checks result validation accepts valid unconditional results and NaN for zero conditioned samples.
- `ProbabilityResultValidationTests.test_configuration_rejects_invalid_counts_and_probability` — Checks result validation rejects invalid counts and probability values.
- `ProbabilityResultValidationTests.test_configuration_uses_tolerance_for_probability_consistency` — Checks result validation accepts only configured-close probability ratios.

### `tests/unit_tests/validation/random_variables/_base.py`

- `RandomVariableValidationTests.test_distribution_and_name_accept_expected_values` — Checks random-variable distribution and name validation accepts expected values.
- `RandomVariableValidationTests.test_interval_bounds_and_closure_are_validated` — Checks random-variable interval bounds and closure values are validated.
- `RandomVariableValidationTests.test_target_set_accepts_sympy_set_or_two_element_sequence` — Checks target-set validation accepts a SymPy set or a two-element range sequence.
- `RandomVariableValidationTests.test_function_value_set_and_name_validators` — Checks custom-function, output-value-set, and function-name validation.
- `RandomVariableValidationTests.test_realization_value_set_accepts_none_or_a_value_set` — Checks realization support validation accepts `None` or a value-set instance and rejects other values.
- `RandomVariableValidationTests.test_others_requires_random_variables` — Checks additional custom-function operands must be random variables.

### `tests/unit_tests/validation/statistics/_clopper_pearson.py`

- `ClopperPearsonValidationTests.test_count_validators_accept_non_negative_integers` — Checks Clopper–Pearson count validators accept non-negative integers.
- `ClopperPearsonValidationTests.test_count_validators_reject_booleans_and_negative_values` — Checks Clopper–Pearson count validators reject booleans and negative values.
- `ClopperPearsonValidationTests.test_configuration_rejects_successes_above_samples` — Checks Clopper–Pearson configuration rejects success counts above the sample count.

### `tests/unit_tests/validation/value_sets/_base.py`

- `ValueSetConfigurationValidationTests.test_accepts_sympy_or_unknown_set_and_numpy_dtype_types` — Checks value-set configuration accepts SymPy or unknown sets and NumPy dtype families.
- `ValueSetConfigurationValidationTests.test_rejects_invalid_sympy_set` — Checks value-set configuration rejects an invalid support object.
- `ValueSetConfigurationValidationTests.test_rejects_invalid_dtype_configuration` — Checks value-set configuration rejects invalid dtype-family declarations.

### `tests/unit_tests/value_sets/_comparison.py`

- `ObjectComparisonTests.test_compares_scalars_and_array_like_values` — Checks safe object equality works for scalars and array-like values.
- `ObjectComparisonTests.test_returns_false_when_equality_raises` — Checks safe object equality returns false when an object’s equality operation raises.

### `tests/unit_tests/value_sets/_inference.py`

- `SharedInferenceTests.test_concrete_dtype_candidates_preserve_input_order_and_remove_duplicates` — Checks that concrete dtype candidates preserve input order and remove duplicates.
- `SharedInferenceTests.test_unknown_and_unsupported_dtype_candidates_return_none` — Checks that unknown and unsupported dtype candidates return none.
- `SharedInferenceTests.test_numeric_families_exclude_datetime_and_timedelta_storage` — Checks that ordinary integer and numeric dtype families do not include datetime or timedelta arrays when inferring numeric operations.
- `SharedInferenceTests.test_dtype_resolution_matches_numpy_for_concrete_numeric_inputs` — Checks that dtype resolution matches NumPy for concrete numeric inputs.
- `SharedInferenceTests.test_real_numeric_families_can_resolve_exponential_output_dtypes` — Checks that exp can infer floating output dtypes for declared integer/floating families, as used by Normal random variables.
- `SharedInferenceTests.test_abs_of_generic_numeric_family_can_resolve_real_dtypes` — Checks that abs of generic numeric family can resolve real dtypes.
- `SharedInferenceTests.test_missing_ufunc_or_unsupported_loop_leaves_dtype_unspecified` — Checks that missing ufunc or unsupported loop leaves dtype unspecified.
- `SharedInferenceTests.test_arithmetic_decorator_rejects_unknown_and_object_support_before_delegate` — Checks that arithmetic decorator rejects unknown and object support before delegate.
- `SharedInferenceTests.test_arithmetic_decorator_keeps_unknown_result_and_function_metadata` — Checks that arithmetic decorator keeps unknown result and function metadata.
- `SharedInferenceTests.test_arithmetic_decorator_intersects_integer_inputs_and_resolves_named_operands` — Checks that arithmetic decorator intersects integer inputs and resolves named operands.

### `tests/unit_tests/value_sets/_mathematical_inference.py`

- `MathematicalValueSetInferenceTests.test_addition_identity_is_zero_not_one` — Checks the value-set inference identity for addition is zero.
- `MathematicalValueSetInferenceTests.test_zero_division_returns_wrapper` — Checks division by a known zero support returns the dedicated wrapper representation.
- `MathematicalValueSetInferenceTests.test_possible_zero_divisors_are_unknown` — Checks a divisor that may be zero produces an unknown inferred value set.
- `MathematicalValueSetInferenceTests.test_nonreal_fractional_power` — Checks a negative base with a non-integral fractional exponent infers a non-real result.
- `MathematicalValueSetInferenceTests.test_power_undefined_at_zero` — Checks power inference handles expressions undefined at zero.
- `MathematicalValueSetInferenceTests.test_power_zero_and_one_conventions` — Checks power inference follows the special exponent-zero and base-one conventions.
- `MathematicalValueSetInferenceTests.test_unknown_inputs_remain_unknown` — Checks unknown input supports remain unknown through inference.
- `MathematicalValueSetInferenceTests.test_integer_information_and_sign_are_preserved` — Checks inference preserves known integer and sign information when mathematically valid.
- `MathematicalValueSetInferenceTests.test_identity_rules_allow_numpy_dtype_promotion` — Checks inference identity rules permit NumPy dtype promotion.
- `MathematicalValueSetInferenceTests.test_real_output_from_complex_absolute_value` — Checks absolute value of a complex support infers a real output support.
- `MathematicalValueSetInferenceTests.test_object_arithmetic_representation` — Checks arithmetic on object representations uses the appropriate object-level value-set representation.
- `MathematicalValueSetInferenceTests.test_unknown_dtype_stays_unspecified` — Checks unknown support information leaves the dtype family unspecified.
- `MathematicalValueSetInferenceTests.test_exact_dtype_rules_follow_the_operation` — Checks exact inferred dtype rules correspond to the applied arithmetic operation.
- `MathematicalValueSetInferenceTests.test_integer_family_includes_signed_unsigned_promotion` — Checks integer dtype inference includes signed and unsigned promotion cases.
- `MathematicalValueSetInferenceTests.test_custom_operations_do_not_guess_dtype` — Checks custom operations do not invent a dtype family without a stated rule.
- `MathematicalValueSetInferenceTests.test_unsupported_representation_leaves_dtype_unknown` — Checks unsupported inferred representations keep dtype information unknown.
- `MathematicalValueSetInferenceTests.test_float_families_include_extended_precision` — Checks that arithmetic dtype inference covers NumPy longdouble as well as float16, float32, and float64. This currently exposes a Windows dtype-enumeration gap.
- `MathematicalValueSetInferenceTests.test_representative_binary_results` — Checks representative binary operations infer the intended support and dtype results.
- `MathematicalValueSetInferenceTests.test_representative_unary_results` — Checks representative unary operations infer the intended support and dtype results.
- `MathematicalValueSetInferenceTests.test_representative_power_results` — Checks representative power expressions infer their intended support and dtype results.

### `tests/unit_tests/value_sets/_realization_inference.py`

- `RealizationInferenceTests.test_bounded_integer_operations_preserve_dtype_when_overflow_is_impossible` — Checks that bounded integer operations preserve dtype when overflow is impossible.
- `RealizationInferenceTests.test_underflow_support_includes_zero_for_multiplication_division_and_power` — Checks that underflow support includes zero for multiplication division and power.
- `RealizationInferenceTests.test_function_inference_preserves_finite_template_and_resolves_output_dtype` — Checks all 24 predefined function inferers against safe inputs: each keeps its supplied finite support, resolves float32 output, and does not declare unnecessary exceptional values.
- `RealizationInferenceTests.test_exponential_and_hyperbolic_functions_declare_possible_overflow` — Checks that exponential and hyperbolic functions declare possible overflow.
- `RealizationInferenceTests.test_logs_declare_zero_division_invalid_negative_inputs_and_input_infinity` — Checks that logs declare zero division invalid negative inputs and input infinity.
- `RealizationInferenceTests.test_log1p_and_sqrt_distinguish_endpoints_from_invalid_domains` — Checks that log1p and sqrt distinguish endpoints from invalid domains.
- `RealizationInferenceTests.test_inverse_trigonometric_and_hyperbolic_domains_declare_invalid_machine_inputs` — Checks that inverse trigonometric and hyperbolic domains declare invalid machine inputs.
- `RealizationInferenceTests.test_infinite_trigonometric_inputs_produce_nan_but_bounded_inverse_outputs_do_not` — Checks that infinite trigonometric inputs produce NaN but bounded inverse outputs do not.
- `RealizationInferenceTests.test_rounding_functions_and_arcsinh_preserve_signed_infinity_flags` — Checks that rounding functions and arcsinh preserve signed infinity flags.
- `RealizationInferenceTests.test_all_function_inferers_propagate_possible_nan_inputs` — Checks that all function inferers propagate possible NaN inputs.
- `RealizationInferenceTests.test_hypot_accounts_for_both_operands_and_floating_overflow` — Checks that hypot accounts for both operands and floating overflow.
- `RealizationInferenceTests.test_logaddexp_is_stable_for_large_finite_values_and_tracks_signed_infinities` — Checks that logaddexp is stable for large finite values and tracks signed infinities.
- `RealizationInferenceTests.test_rounded_intervals_enclose_numpy_endpoint_values_for_each_precision` — Checks outward interval bounds against NumPy arcsin, arccos, and arctan endpoint results at all supported tested floating precisions.
- `RealizationInferenceTests.test_rounded_intervals_combine_precisions_and_cache_equal_requests` — Checks that rounded intervals combine precisions and cache equal requests.
- `RealizationInferenceTests.test_rounded_interval_rejects_empty_nonfloating_and_nonfinite_configurations` — Checks that rounded interval rejects empty nonfloating and nonfinite configurations.
- `RealizationInferenceTests.test_addition_preserves_integer_overflow_direction` — Checks that addition preserves integer overflow direction.
- `RealizationInferenceTests.test_addition_declares_floating_overflow` — Checks that addition declares floating overflow.
- `RealizationInferenceTests.test_addition_and_subtraction_handle_infinite_cancellation` — Checks that addition and subtraction handle infinite cancellation.
- `RealizationInferenceTests.test_subtraction_preserves_integer_overflow_direction` — Checks that subtraction preserves integer overflow direction.
- `RealizationInferenceTests.test_multiplication_handles_overflow_and_zero_times_infinity` — Checks that multiplication handles overflow and zero times infinity.
- `RealizationInferenceTests.test_division_handles_signed_zero_and_zero_over_zero` — Checks that division handles signed zero and zero over zero.
- `RealizationInferenceTests.test_division_declares_overflow_and_infinity_over_infinity` — Checks that division declares overflow and infinity over infinity.
- `RealizationInferenceTests.test_modulo_handles_zero_divisor_and_infinite_inputs` — Checks that modulo handles zero divisor and infinite inputs.
- `RealizationInferenceTests.test_power_distinguishes_odd_and_even_integer_overflow` — Checks that power distinguishes odd and even integer overflow.
- `RealizationInferenceTests.test_power_infers_large_exponents_without_constructing_the_result` — Checks that power infers large exponents without constructing the result.
- `RealizationInferenceTests.test_positive_exponent_range_does_not_imply_negative_powers_of_zero` — Checks that positive exponent range does not imply negative powers of zero.
- `RealizationInferenceTests.test_floating_power_allows_infinity_sign_changes_from_exponent_rounding` — Checks that floating power allows infinity sign changes from exponent rounding.
- `RealizationInferenceTests.test_power_preserves_real_numpy_identity_rules_for_nan` — Checks that power preserves real NumPy identity rules for NaN.
- `RealizationInferenceTests.test_power_distinguishes_real_invalid_input_from_complex_output` — Checks that power distinguishes real invalid input from complex output.
- `RealizationInferenceTests.test_negation_handles_signed_minimum_unsigned_values_and_infinity` — Checks that negation handles signed minimum unsigned values and infinity.
- `RealizationInferenceTests.test_absolute_maps_both_infinity_signs_to_positive_infinity` — Checks that absolute maps both infinity signs to positive infinity.
- `RealizationInferenceTests.test_bounded_real_operations_keep_exceptional_flags_disabled` — Checks that bounded real operations keep exceptional flags disabled.
- `RealizationInferenceTests.test_complex_overflow_can_declare_nan_components` — Checks that complex overflow can declare NaN components.
- `RealizationInferenceTests.test_underflow_support_expansion_preserves_exceptional_flags` — Checks that underflow support expansion preserves exceptional flags.

### `tests/unit_tests/value_sets/_unknown.py`

- `UnknownValueSetTests.test_representation_is_stable` — Checks the unknown value-set representation remains stable and readable.

### `tests/unit_tests/value_sets/_utils.py`

- `ValueSetUtilityTests.test_detect_non_finite_values_distinguishes_sign_nan_and_complex_components` — Checks that detect non finite values distinguishes sign NaN and complex components.
- `ValueSetUtilityTests.test_detect_non_finite_values_leaves_compound_categories_atomic` — Checks that detect non finite values leaves compound categories atomic.
- `ValueSetUtilityTests.test_non_finite_membership_uses_each_permission_independently` — Checks that non finite membership uses each permission independently.
- `ValueSetUtilityTests.test_non_finite_membership_returns_none_for_finite_and_opaque_values` — Checks that non finite membership returns none for finite and opaque values.
- `ValueSetUtilityTests.test_validate_as_subset_accepts_only_declared_exceptional_values` — Checks that validate as subset accepts only declared exceptional values.
- `ValueSetUtilityTests.test_validate_as_subset_accepts_declared_nan_even_if_finite_support_is_unknown` — Checks that validate as subset accepts declared NaN even if finite support is unknown.
- `ValueSetUtilityTests.test_validate_as_subset_requires_all_non_finite_complex_components` — Checks that validate as subset requires all non finite complex components.
- `ValueSetUtilityTests.test_to_sympy_value_converts_integral_float_to_integer` — Checks that a finite integral float becomes a SymPy integer value.
- `ValueSetUtilityTests.test_to_sympy_value_preserves_non_integral_float` — Checks that a non-integral float remains a floating SymPy value.
- `ValueSetUtilityTests.test_is_known_subset_handles_numeric_object_and_unknown_sets` — Checks known-subset detection across numeric, object, and unknown value sets.
- `ValueSetUtilityTests.test_validate_as_subset_accepts_integer_valued_floats_for_integer_set` — Checks subset validation accepts integer-valued floats for the integer set.
- `ValueSetUtilityTests.test_validate_as_subset_rejects_outside_or_unknown_values` — Checks subset validation rejects values outside or indeterminate for the target set.
- `ValueSetUtilityTests.test_validate_as_subset_uses_object_membership` — Checks subset validation uses object-category membership for object value sets.

### `tests/unit_tests/value_sets/base.py`

- `ValueSetBaseTests.test_numeric_base_declares_non_finite_permissions_false` — Checks that numeric base declares non finite permissions false.
- `ValueSetBaseTests.test_value_set_remains_abstract` — Checks that the base `ValueSet` class cannot be instantiated directly.
- `ValueSetBaseTests.test_numeric_value_set_is_value_set_marker_class` — Checks `NumericValueSet` retains its role as a `ValueSet` marker base class.
- `ValueSetBaseTests.test_dynamic_attributes_load_concrete_value_set_classes` — Checks lazy module attributes resolve concrete value-set classes correctly.

### `tests/unit_tests/value_sets/homogeneous_numeric_value_set.py`

- `HomogeneousNumericValueSetTests.test_non_finite_values_are_rejected_by_default_and_enabled_independently` — Checks that non finite values are rejected by default and enabled independently.
- `HomogeneousNumericValueSetTests.test_contains_preserves_array_shape_with_finite_and_exceptional_values` — Checks that contains preserves array shape with finite and exceptional values.
- `HomogeneousNumericValueSetTests.test_unknown_finite_support_does_not_override_non_finite_permissions` — Checks that unknown finite support does not override non finite permissions.
- `HomogeneousNumericValueSetTests.test_exceptional_permissions_are_keyword_only_and_frozen` — Checks that exceptional permissions are keyword only and frozen.
- `HomogeneousNumericValueSetTests.test_exceptional_permissions_require_booleans_even_with_unknown_dtype` — Checks that all three exceptional-value permissions reject non-boolean settings whether dtype information is known or unspecified.
- `HomogeneousNumericValueSetTests.test_contains_returns_bool_for_scalar_and_array_for_array` — Checks homogeneous numeric membership returns a Boolean scalar or elementwise Boolean array as appropriate.
- `HomogeneousNumericValueSetTests.test_constructor_validates_configuration` — Checks homogeneous numeric value-set construction validates support and dtype configuration.

### `tests/unit_tests/value_sets/mixed_numeric_value_set.py`

- `MixedNumericValueSetTests.test_preserves_values_and_supports_scalar_and_array_membership` — Checks mixed numeric value sets preserve values and support scalar and array membership.
- `MixedNumericValueSetTests.test_constructor_requires_non_empty_tuple` — Checks mixed numeric value sets require a non-empty tuple of values.

### `tests/unit_tests/value_sets/object_value_set.py`

- `ObjectValueSetTests.test_contains_preserves_object_identity_and_array_shape` — Checks object value-set membership respects object identity and input-array shape.
- `ObjectValueSetTests.test_constructor_requires_non_empty_tuple` — Checks object value sets require a non-empty tuple of categories.

### `tests/unit_tests/value_sets/sets.py`

- `PredefinedValueSetTests.test_core_numeric_sets_describe_expected_membership` — Checks predefined numeric sets recognize representative members and non-members.
- `PredefinedValueSetTests.test_unknown_value_set_uses_unknown_symbolic_support` — Checks the predefined unknown set uses the unknown symbolic-support sentinel.

## Integration tests

### `tests/integration_tests/categorical.py`

- `CategoricalRandomVariableIntegrationTests.test_declared_non_finite_categories_validate_under_every_sampling_policy` — Checks that explicit infinity and NaN categories sample and validate without warnings under all three policies, since sampling supplied values does not generate a new numerical error.
- `CategoricalRandomVariableIntegrationTests.test_tuple_category_is_compared_as_one_value` — Checks that a tuple category remains one categorical outcome during equality probability evaluation.
- `CategoricalRandomVariableIntegrationTests.test_list_category_is_compared_as_one_value` — Checks that a list category remains one categorical outcome during equality probability evaluation.
- `CategoricalRandomVariableIntegrationTests.test_object_categories_support_realization_validation` — Checks that opaque object categories can be sampled with realization validation enabled.
- `CategoricalRandomVariableIntegrationTests.test_mixed_numeric_categories_produce_validated_numeric_samples` — Checks native mixed-category sampling with validation and confirms that arithmetic receives the same numerical values.

### `tests/integration_tests/composed_workflows.py`

- `ComposedWorkflowIntegrationTests.test_distribution_can_use_random_variable_parameters_through_sampling` — Checks that a distribution samples correctly when one of its parameters is a random variable.
- `ComposedWorkflowIntegrationTests.test_probability_uses_one_shared_realization_for_related_events` — Checks that related events reuse a single realization rather than being sampled independently.
- `ComposedWorkflowIntegrationTests.test_probability_result_provides_a_confidence_interval_after_evaluation` — Checks that a public probability result can produce a confidence interval after an event evaluation.
- `ComposedWorkflowIntegrationTests.test_interval_estimation_works_for_a_composed_random_variable` — Checks interval estimation on a random variable built from another random variable.

### `tests/integration_tests/numerical_functions.py`

- `NumericalFunctionIntegrationTests.test_all_predefined_functions_match_native_values_and_validate_concrete_dtypes` — Checks every unary predefined function against NumPy at float16, float32, and float64 precision, including sample dtype and declared realization support.
- `NumericalFunctionIntegrationTests.test_binary_functions_support_scalar_and_variable_inputs_with_mixed_precision` — Checks that binary functions support scalar and variable inputs with mixed precision.
- `NumericalFunctionIntegrationTests.test_inverse_trigonometric_endpoints_keep_exact_math_and_rounded_realization_support` — Checks that inverse trigonometric endpoints keep exact math and rounded realization support.
- `NumericalFunctionIntegrationTests.test_exponential_overflow_policy_does_not_change_mathematical_support` — Checks that exponential overflow policy does not change mathematical support.
- `NumericalFunctionIntegrationTests.test_underflow_is_allowed_but_logarithms_apply_policy_to_realized_zero` — Checks that underflow is allowed but logarithms apply policy to realized zero.
- `NumericalFunctionIntegrationTests.test_arctanh_handles_rounded_tanh_endpoint_under_each_policy` — Checks that arctanh handles rounded tanh endpoint under each policy.
- `NumericalFunctionIntegrationTests.test_realization_domain_violations_declare_nan_without_widening_math_support` — Checks that realization domain violations declare NaN without widening math support.

### `tests/integration_tests/power.py`

- `RandomVariablePowerIntegrationTests.test_real_power_uses_real_samples` — Checks that a power expression inferred as real produces real-valued samples.
- `RandomVariablePowerIntegrationTests.test_complex_power_keeps_complex_samples` — Checks that a power expression requiring complex values preserves a complex sample representation.

### `tests/integration_tests/realization_support.py`

- `PromotedSupportIntegrationTests.test_context_accepts_promoted_outputs` — Checks that support inference and the realization context agree on floating and complex promotions in division, addition, and powers.

## Public API tests

### `tests/api_tests/exact_distribution_methods.py`

- `ExactDistributionMethodTests.test_exact_mean_matches_analytical_mean` — Checks the exact mean of fixed normal, binomial, Poisson, and numeric categorical distributions against their analytical means.
- `ExactDistributionMethodTests.test_exact_variance_matches_analytical_variance` — Checks the exact variance of the four implemented fixed-parameter distributions against analytical values.
- `ExactDistributionMethodTests.test_exact_standard_deviation_matches_square_root_of_variance` — Checks that exact standard deviation equals the square root of the analytical variance for each implemented distribution.
- `ExactDistributionMethodTests.test_exact_cdf_matches_analytical_probability` — Checks exact CDF values at representative normal, binomial, Poisson, and categorical points.
- `ExactDistributionMethodTests.test_exact_ppf_matches_analytical_median` — Checks exact inverse-CDF values at probability 0.5 against the known medians.

### `tests/api_tests/imports.py`

- `PublicImportTests.test_top_level_exports` — Checks that `problab.__all__` contains exactly the documented top-level public API and that every name is accessible.
- `PublicImportTests.test_distribution_package_exports` — Checks the public exports of the distributions package and its continuous and discrete subpackages.
- `PublicImportTests.test_function_package_exports` — Checks that every documented mathematical wrapper is publicly exported and callable.
- `PublicImportTests.test_probability_package_lazy_exports` — Checks the lazy public exports, directory listing, and missing-name behavior of `problab.probability`.
- `PublicImportTests.test_random_variable_package_lazy_exports` — Checks the lazy public exports, directory listing, and missing-name behavior of `problab.random_variables`.
- `PublicImportTests.test_value_set_package_exports_core_types_and_sets` — Checks that core value-set classes and predefined numeric sets are publicly available.

### `tests/api_tests/parameter_policies.py`

- `DistributionParameterPolicyTests.test_parameter_risk_policy_is_keyword_only_with_warn_default` — Checks that parameter risk policy is keyword only with warn default.
- `DistributionParameterPolicyTests.test_risky_parameters_warn_by_default_raise_or_allow_silently_when_requested` — Checks all five numeric distribution parameters follow the construction risk policy and preserve the original RandomVariable inputs.
- `DistributionParameterPolicyTests.test_risk_policy_never_bypasses_mathematically_invalid_parameters` — Checks that risk policy never bypasses mathematically invalid parameters.
- `DistributionParameterPolicyTests.test_risk_policy_never_bypasses_unsupported_realization_dtype` — Checks that risk policy never bypasses unsupported realization dtype.
- `DistributionParameterPolicyTests.test_invalid_policy_is_rejected_before_other_parameter_errors` — Checks that invalid policy is rejected before other parameter errors.
- `DistributionParameterPolicyTests.test_subclass_static_symbol_is_used_in_name_and_parameter_risk_message` — Checks that subclass static symbol is used in name and parameter risk message.
- `DistributionParameterPolicyTests.test_static_symbols_do_not_add_constructor_parameters` — Checks all four concrete distributions expose their static symbols and keep symbol out of their constructor signatures.

### `tests/api_tests/public_contracts.py`

- `PublicContractTests.test_public_import_paths_share_class_identity` — Checks that the top-level and subpackage import paths expose the same class objects.
- `PublicContractTests.test_star_import_exposes_exactly_the_top_level_public_names` — Checks that `from problab import *` exposes exactly the names declared in `problab.__all__`.
- `PublicContractTests.test_distribution_parameters_preserve_public_random_variable_inputs` — Checks that a distribution retains the original `RandomVariable` object supplied as a parameter.
- `PublicContractTests.test_categorical_configuration_is_read_only_and_outside_graph_parameters` — Checks categorical configuration properties, their immutability, and the empty graph-parameter tuple.
- `PublicContractTests.test_public_distribution_sample_accepts_count_and_seeded_generator` — Checks that public distribution sampling accepts a sample count and caller-supplied random generator.
- `PublicContractTests.test_public_distribution_sample_can_validate_realizations` — Checks that public distribution sampling accepts the realization-validation option.
- `PublicContractTests.test_random_variable_can_raise_its_graph_size_limit` — Checks that a deep, valid random-variable expression can request a larger graph limit for sampling.

## Property tests

### `tests/property_tests/test_categorical_and_arithmetic.py`

- `CategoricalAndArithmeticPropertyTests.test_categorical_configuration_merges_equal_values_and_normalizes_probabilities` — Checks across many generated configurations that equal categories merge and resulting probabilities sum to one.
- `CategoricalAndArithmeticPropertyTests.test_arithmetic_identities_hold_for_many_integer_categorical_variables` — Checks arithmetic identity rules across many generated integer categorical variables.
- `CategoricalAndArithmeticPropertyTests.test_compound_categories_remain_atomic_values_through_sampling` — Checks across generated compound categories that sampling preserves each category as one atomic object.

### `tests/property_tests/test_probability_results.py`

- `ProbabilityResultPropertyTests.test_sample_count_and_confidence_interval_hold_for_many_valid_counts` — Checks across many valid count combinations that sample-count selection and confidence-interval construction remain valid.

### `tests/property_tests/test_quantile_boundaries.py`

- `QuantileBoundaryTests.test_binary_search_matches_exhaustive_feasible_indices` — Checks that binary search matches exhaustive feasible indices.
- `QuantileBoundaryTests.test_tail_probability_equal_to_threshold_is_feasible` — Checks that tail probability equal to threshold is feasible.

### `tests/property_tests/test_realization_support_soundness.py`

- `RealizationSupportSoundnessTests.test_integer_arithmetic_support_covers_deterministic_boundary_batches` — Checks that declared realization support covers addition, subtraction, multiplication, division, and modulo results for a batch spanning int8 boundaries.
- `RealizationSupportSoundnessTests.test_integer_power_support_covers_positive_negative_even_odd_and_zero_cases` — Checks that integer power support covers positive negative even odd and zero cases.
- `RealizationSupportSoundnessTests.test_unary_integer_support_covers_signed_minimum_and_unsigned_values` — Checks that unary integer support covers signed minimum and unsigned values.
- `RealizationSupportSoundnessTests.test_floating_arithmetic_support_covers_overflow_underflow_and_zero_division` — Checks arithmetic support against actual floating outputs at extreme values, subnormal values, and zero divisors for float16, float32, and float64.

## Regression tests

### `tests/regression_tests/test_arithmetic_exceptional_support.py`

- `ArithmeticExceptionalSupportTests.test_warn_and_ignore_allow_declared_exceptional_samples_with_validation` — Checks that warn and ignore allow declared exceptional samples with validation.
- `ArithmeticExceptionalSupportTests.test_raise_stops_exceptional_arithmetic_before_output_validation` — Checks that raise stops exceptional arithmetic before output validation.

### `tests/regression_tests/test_audit_edge_cases.py`

- `AuditEdgeCaseTests.test_normal_rejects_python_and_numpy_booleans_with_type_error` — Checks that normal rejects python and NumPy booleans with type error.
- `AuditEdgeCaseTests.test_public_graph_rejects_non_integer_limits` — Checks that public graph rejects non integer limits.
- `AuditEdgeCaseTests.test_complement_conditioning_returns_zero` — Checks that complement conditioning returns zero.
- `AuditEdgeCaseTests.test_mixed_numeric_categories_support_exponential` — Checks that mixed numeric categories support exponential.
- `AuditEdgeCaseTests.test_multiplication_underflow_passes_realization_validation` — Checks that multiplication underflow passes realization validation.
- `AuditEdgeCaseTests.test_sample_third_positional_argument_still_means_validate` — Checks that sample third positional argument still means validate.
- `AuditEdgeCaseTests.test_graph_limit_error_reports_requested_limit` — Checks that graph limit error reports requested limit.
- `AuditEdgeCaseTests.test_non_numeric_cdf_rejects_scalar_and_array_consistently` — Checks that non numeric CDF rejects scalar and array consistently.

### `tests/regression_tests/test_binomial_support.py`

- `BinomialSupportRegressionTests.test_finite_binomial_support_uses_symbolic_range` — Checks that finite binomial support is represented symbolically without expanding every integer outcome.

### `tests/regression_tests/test_categorical_numeric_support.py`

- `CategoricalNumericSupportRegressionTests.test_integer_valued_float_categories_declare_integer_mathematical_support` — Checks that integer valued float categories declare integer mathematical support.
- `CategoricalNumericSupportRegressionTests.test_integer_valued_float_category_can_be_a_binomial_trial_count` — Checks that integer valued float category can be a binomial trial count.
- `CategoricalNumericSupportRegressionTests.test_integer_valued_categorical_exponent_keeps_real_square_support` — Checks that integer valued categorical exponent keeps real square support.
- `CategoricalNumericSupportRegressionTests.test_large_integer_object_fallback_preserves_public_samples_and_numeric_support` — Checks that large integer object fallback preserves public samples and numeric support.

### `tests/regression_tests/test_categorical_values.py`

- `CategoricalValueRegressionTests.test_duplicate_categories_are_merged_before_sampling` — Checks that duplicate categorical outcomes merge before sampling and their probabilities are added.
- `CategoricalValueRegressionTests.test_compound_category_is_an_atomic_value_during_probability_evaluation` — Checks that a compound category compares as one object during public probability evaluation.

### `tests/regression_tests/test_cdf_nan.py`

- `CdfNanRegressionTests.test_scalar_nan_is_rejected_by_cdf` — Checks that a scalar NaN input to CDF raises a validation error.
- `CdfNanRegressionTests.test_array_nan_is_rejected_by_cdf` — Checks that an array containing NaN is rejected by CDF validation.

### `tests/regression_tests/test_float_boundary_validation.py`

- `FloatBoundaryValidationRegressionTests.test_tanh_rounding_to_one_remains_valid` — Checks that `tanh` rounding to exactly one does not contradict its declared support during validation.
- `FloatBoundaryValidationRegressionTests.test_exponential_underflow_to_zero_remains_valid` — Checks that exponential underflow to exactly zero does not contradict its declared support during validation.

### `tests/regression_tests/test_graph_labels.py`

- `GraphLabelRegressionTests.test_poisson_name_identifies_the_distribution` — Checks that a Poisson distribution name identifies the Poisson family.
- `GraphLabelRegressionTests.test_categorical_name_identifies_the_distribution` — Checks that a categorical distribution name identifies the categorical family.
- `GraphLabelRegressionTests.test_mathematical_function_node_names_the_function` — Checks that a mathematical wrapper’s graph node name includes both the wrapper name and source variable name.

### `tests/regression_tests/test_integer_membership.py`

- `IntegerMembershipRegressionTests.test_integer_value_set_contains_integer_valued_float` — Checks that an integer-valued floating scalar is recognized as a member of the integer value set.
- `IntegerMembershipRegressionTests.test_integer_value_set_checks_each_float_in_array` — Checks that integer membership is evaluated independently for each floating array element.
- `IntegerMembershipRegressionTests.test_integer_valued_float_matches_integer_set_event` — Checks that an integer-valued floating category satisfies a public integer-set event with validation enabled.

### `tests/regression_tests/test_integer_overflow.py`

- `IntegerOverflowRegressionTests.test_fixed_width_arithmetic_raises_instead_of_silently_wrapping` — Checks that fixed width arithmetic raises instead of silently wrapping.
- `IntegerOverflowRegressionTests.test_warn_and_ignore_replace_overflow_with_infinity_and_pass_validation` — Checks that warn and ignore replace overflow with infinity and pass validation.
- `IntegerOverflowRegressionTests.test_bounded_integer_arithmetic_preserves_integer_dtype` — Checks that bounded integer arithmetic preserves integer dtype.

### `tests/regression_tests/test_numeric_validation_consistency.py`

- `NumericValidationConsistencyRegressionTests.test_numpy_integer_is_accepted_as_binomial_trial_count` — Checks that a NumPy integer is accepted wherever a binomial trial count is accepted.
- `NumericValidationConsistencyRegressionTests.test_numpy_float_is_accepted_as_distribution_parameter` — Checks that NumPy floating values are accepted as normal and Poisson parameters.
- `NumericValidationConsistencyRegressionTests.test_boolean_is_rejected_as_binomial_trial_count` — Checks that a boolean is rejected as a binomial trial count with a type error.
- `NumericValidationConsistencyRegressionTests.test_boolean_is_rejected_as_probability_or_rate` — Checks that booleans are rejected as binomial probabilities, Poisson rates, and categorical probabilities.

### `tests/regression_tests/test_parameter_realization_policy.py`

- `ParameterRealizationPolicyRegressionTests.test_raise_policy_rejects_invalid_realized_parameters_with_informative_error` — Requires invalid realized Normal mean/std, Binomial n/p, and Poisson mu to raise an informative ValueError under the sampling raise policy. This specifies behavior still missing from the implementation.
- `ParameterRealizationPolicyRegressionTests.test_warn_policy_warns_and_marks_only_invalid_positions_nan` — Requires the sampling warn policy to emit a RuntimeWarning and return NaN only for invalid parameter positions, while valid positions remain finite and validate=True succeeds. This behavior is not implemented yet.
- `ParameterRealizationPolicyRegressionTests.test_ignore_policy_marks_only_invalid_positions_nan_without_warning` — Requires the sampling ignore policy to return NaN only at invalid parameter positions, silently and with validate=True enabled. This behavior is not implemented yet.

### `tests/regression_tests/test_probability_result_confidence_interval.py`

- `ProbabilityResultConfidenceIntervalRegressionTests.test_confidence_interval_is_available_from_a_public_probability_result` — Checks that a public `ProbabilityResult` can construct its confidence interval.

### `tests/regression_tests/test_probability_result_consistency.py`

- `ProbabilityResultConsistencyRegressionTests.test_probability_must_match_success_count` — Checks that a reported unconditional probability agrees with its stated success count.
- `ProbabilityResultConsistencyRegressionTests.test_conditional_probability_must_match_conditioned_count` — Checks that a reported conditional probability agrees with successes and conditioned sample count.

### `tests/regression_tests/test_quantile_confidence_interval_samples.py`

- `QuantileConfidenceIntervalSampleRegressionTests.test_insufficient_samples_are_rejected_before_constructing_an_interval` — Checks that quantile confidence intervals reject an insufficient sample before creating bounds.

### `tests/regression_tests/test_quantile_search_cost.py`

- `QuantileSearchCostRegressionTests.test_binomial_tail_search_uses_sublinear_number_of_calls` — Checks that quantile-interval tail selection does not make a linear number of binomial CDF and survival-function calls.

### `tests/regression_tests/test_real_powers.py`

- `RealPowerRegressionTests.test_square_of_a_negative_value_remains_compatible_with_sqrt` — Checks that squaring a negative value produces a result accepted by the square-root wrapper.

### `tests/regression_tests/test_requested_nodes.py`

- `RequestedNodesRegressionTests.test_complement_conditioning_preserves_exact_condition_count` — Checks that complement conditioning preserves exact condition count.
- `RequestedNodesRegressionTests.test_identical_event_and_condition_reuse_one_realization` — Checks that identical event and condition reuse one realization.
- `RequestedNodesRegressionTests.test_constant_joint_with_unsatisfied_condition_returns_nan` — Checks that constant joint with unsatisfied condition returns NaN.
- `RequestedNodesRegressionTests.test_dependency_graph_builds_from_single_variable_with_new_constructor` — Checks that dependency graph builds from single variable with new constructor.
- `RequestedNodesRegressionTests.test_plot_dependencies_builds_graph_with_new_constructor` — Checks that plot dependencies builds graph with new constructor.

## Statistical tests

### `tests/statistical_tests/test_analytical_workflows.py`

- `AnalyticalWorkflowStatisticalTests.test_binomial_point_probability_matches_finite_formula` — Compares an estimated binomial point probability with the finite binomial formula.
- `AnalyticalWorkflowStatisticalTests.test_poisson_lower_tail_matches_finite_formula` — Compares an estimated Poisson lower-tail probability with the finite Poisson sum.
- `AnalyticalWorkflowStatisticalTests.test_normal_upper_tail_matches_error_function` — Compares an estimated normal upper-tail probability with the analytical error-function result.
- `AnalyticalWorkflowStatisticalTests.test_conditional_binomial_probability_matches_ratio_of_finite_sums` — Compares a conditional binomial probability with an independently calculated ratio of finite sums.
- `AnalyticalWorkflowStatisticalTests.test_categorical_event_logic_matches_sum_of_category_weights` — Compares categorical event logic with the sum of the corresponding category probabilities.
- `AnalyticalWorkflowStatisticalTests.test_categorical_complement_and_conditional_probability_match_weights` — Compares categorical complements and conditional probabilities with their known category weights.
- `AnalyticalWorkflowStatisticalTests.test_compound_object_category_probability_matches_its_weight` — Checks that an opaque compound category has the probability assigned to that exact category.
- `AnalyticalWorkflowStatisticalTests.test_interval_closure_matches_discrete_category_weights` — Checks that open and closed interval events include exactly the intended discrete category weights.
- `AnalyticalWorkflowStatisticalTests.test_finite_set_membership_matches_category_weights` — Checks that finite-set membership probability equals the total probability of the matching categories.
- `AnalyticalWorkflowStatisticalTests.test_affine_transform_of_binomial_matches_original_tail_probability` — Checks that an affine binomial transformation preserves the analytically equivalent tail probability.
- `AnalyticalWorkflowStatisticalTests.test_nonlinear_function_composition_matches_discrete_weights` — Checks a nonlinear random-variable composition against independently summed discrete weights.
- `AnalyticalWorkflowStatisticalTests.test_exponential_transform_of_normal_matches_analytical_cdf` — Compares the CDF of an exponentiated normal variable with its analytical transformed-normal CDF.
- `AnalyticalWorkflowStatisticalTests.test_reused_random_variable_keeps_its_dependence` — Checks that using the same random variable twice preserves dependence in the resulting event.
- `AnalyticalWorkflowStatisticalTests.test_complex_power_followed_by_absolute_value_matches_finite_weights` — Checks a complex power followed by absolute value against finite categorical probability weights.
- `AnalyticalWorkflowStatisticalTests.test_modulo_and_reverse_division_match_finite_category_weights` — Checks modulo and reverse-division expressions against independently summed categorical weights.
- `AnalyticalWorkflowStatisticalTests.test_sum_of_independent_binomials_matches_finite_convolution` — Checks the sum of independent binomials against the finite convolution of their probabilities.

### `tests/statistical_tests/test_confidence_interval_coverage.py`

- `ConfidenceIntervalCoverageStatisticalTests.test_probability_confidence_intervals_cover_true_categorical_probability` — Repeats categorical probability estimation to check that confidence-interval coverage matches its stated confidence level.
- `ConfidenceIntervalCoverageStatisticalTests.test_random_variable_quantile_confidence_intervals_cover_true_median` — Repeats random-variable median intervals to check coverage of the true median.
- `ConfidenceIntervalCoverageStatisticalTests.test_distribution_quantile_confidence_intervals_cover_true_median` — Repeats distribution median intervals to check coverage of the true median.

### `tests/statistical_tests/test_derived_functions.py`

- `DerivedFunctionStatisticalTests.test_floor_of_normal_matches_difference_of_analytical_cdfs` — Compares a floored normal-variable probability with the difference of analytical normal CDF values.
- `DerivedFunctionStatisticalTests.test_log_of_exponential_normal_matches_original_normal_tail` — Checks that taking log after exponentiating a normal variable restores the matching normal-tail probability.
- `DerivedFunctionStatisticalTests.test_trigonometric_function_matches_weighted_finite_categories` — Checks a trigonometric transformation of finite categories against their weighted analytical result.
- `DerivedFunctionStatisticalTests.test_custom_function_of_two_independent_variables_matches_finite_sum` — Checks a custom two-variable function against an independently calculated finite double sum.

### `tests/statistical_tests/test_distribution_methods.py`

- `DistributionMethodStatisticalTests.test_normal_mean_variance_and_std_match_their_sampling_laws` — Checks Monte Carlo normal mean, variance, and standard deviation using their correct sampling-law bounds.
- `DistributionMethodStatisticalTests.test_normal_cdf_array_matches_analytical_normal_cdf` — Checks a normal CDF array estimate against analytical normal CDF values.
- `DistributionMethodStatisticalTests.test_normal_ppf_array_lies_within_analytical_dkw_bounds` — Checks normal quantile estimates against DKW confidence bounds.
- `DistributionMethodStatisticalTests.test_binomial_cdf_and_ppf_match_finite_distribution` — Checks binomial CDF and inverse-CDF estimates against finite-distribution calculations.
- `DistributionMethodStatisticalTests.test_random_variable_probability_interval_matches_normal_quantiles` — Checks a random-variable probability interval against normal-distribution quantiles.

### `tests/statistical_tests/test_distribution_sampling.py`

- `DistributionSamplingStatisticalTests.test_normal_sample_mean_matches_configured_mean` — Checks that a large normal sample mean is statistically consistent with the configured mean.
- `DistributionSamplingStatisticalTests.test_binomial_sample_mean_matches_configured_mean` — Checks that a large binomial sample mean is statistically consistent with the configured mean.
- `DistributionSamplingStatisticalTests.test_poisson_sample_mean_matches_configured_mean` — Checks that a large Poisson sample mean is statistically consistent with the configured mean.
- `DistributionSamplingStatisticalTests.test_categorical_sample_frequencies_match_configured_probabilities` — Checks that categorical sample frequencies are statistically consistent with configured probabilities.
- `DistributionSamplingStatisticalTests.test_probability_estimate_matches_a_categorical_event_probability` — Checks that public probability estimation matches a known categorical event probability.

### `tests/statistical_tests/test_random_parameters.py`

- `RandomParameterStatisticalTests.test_mixed_integer_float_normal_mean_matches_analytical_mixture` — Checks that mixed integer float normal mean matches analytical mixture.
- `RandomParameterStatisticalTests.test_mixed_integer_float_poisson_rate_matches_weighted_zero_probability` — Checks that mixed integer float poisson rate matches weighted zero probability.
- `RandomParameterStatisticalTests.test_random_binomial_probability_matches_weighted_analytical_tails` — Checks a binomial with random probability against the weighted mixture of analytical tails.
- `RandomParameterStatisticalTests.test_random_binomial_trial_count_matches_weighted_analytical_tails` — Checks a binomial with random trial count against the weighted mixture of analytical tails.
- `RandomParameterStatisticalTests.test_random_normal_mean_matches_weighted_analytical_cdfs` — Checks a normal with random mean against the weighted mixture of analytical CDFs.
- `RandomParameterStatisticalTests.test_conditioning_on_shared_poisson_rate_matches_component_distribution` — Checks conditioning on a shared random Poisson rate against the selected component distribution.

## Library explorer tests

### `docs/library_explorer/test_build_graph.py`

- `SourceGraphTests.test_reexports_resolve_to_original_definition` — Checks that reexports resolve to original definition.
- `SourceGraphTests.test_alias_calls_inheritance_and_self_methods` — Checks that alias calls inheritance and self methods.
- `SourceGraphTests.test_dynamic_calls_are_not_invented` — Checks that dynamic calls are not invented.
- `SourceGraphTests.test_parameters_shadow_imported_names` — Checks that parameters shadow imported names.
- `SourceGraphTests.test_properties_fields_docstrings_and_source_locations` — Checks that properties fields docstrings and source locations.
- `SourceGraphTests.test_reads_source_without_executing_it_and_changes_hash` — Checks that reads source without executing it and changes hash.
- `SourceGraphTests.test_every_edge_and_parent_resolves` — Checks that every edge and parent resolves.
