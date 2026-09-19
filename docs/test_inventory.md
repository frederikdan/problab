# ProbLab test inventory

This inventory lists every discovered `unittest` test method in `tests` as of
2026-09-19. The 345 entries below are grouped first by test purpose and then by
test module. A method that uses `subTest` is one entry here because it is one
test method in discovery, even though it exercises several cases.

The inventory deliberately includes tests that currently fail. Those tests
specify bugs or unfinished behavior; see [test_gap_inventory.md](test_gap_inventory.md)
for their current status and intended implementation targets.

## Public API tests

### `tests/api_tests/exact_distribution_methods.py`

- `test_exact_mean_matches_analytical_mean` — Checks the exact mean of fixed normal, binomial, Poisson, and numeric categorical distributions against their analytical means.
- `test_exact_variance_matches_analytical_variance` — Checks the exact variance of the four implemented fixed-parameter distributions against analytical values.
- `test_exact_standard_deviation_matches_square_root_of_variance` — Checks that exact standard deviation equals the square root of the analytical variance for each implemented distribution.
- `test_exact_cdf_matches_analytical_probability` — Checks exact CDF values at representative normal, binomial, Poisson, and categorical points.
- `test_exact_ppf_matches_analytical_median` — Checks exact inverse-CDF values at probability 0.5 against the known medians.

### `tests/api_tests/imports.py`

- `test_top_level_exports` — Checks that `problab.__all__` contains exactly the documented top-level public API and that every name is accessible.
- `test_distribution_package_exports` — Checks the public exports of the distributions package and its continuous and discrete subpackages.
- `test_function_package_exports` — Checks that every documented mathematical wrapper is publicly exported and callable.
- `test_probability_package_lazy_exports` — Checks the lazy public exports, directory listing, and missing-name behavior of `problab.probability`.
- `test_random_variable_package_lazy_exports` — Checks the lazy public exports, directory listing, and missing-name behavior of `problab.random_variables`.
- `test_value_set_package_exports_core_types_and_sets` — Checks that core value-set classes and predefined numeric sets are publicly available.

### `tests/api_tests/public_contracts.py`

- `test_public_import_paths_share_class_identity` — Checks that the top-level and subpackage import paths expose the same class objects.
- `test_star_import_exposes_exactly_the_top_level_public_names` — Checks that `from problab import *` exposes exactly the names declared in `problab.__all__`.
- `test_distribution_parameters_preserve_public_random_variable_inputs` — Checks that a distribution retains the original `RandomVariable` object supplied as a parameter.
- `test_categorical_configuration_is_read_only_and_outside_graph_parameters` — Checks categorical configuration properties, their immutability, and the empty graph-parameter tuple.
- `test_public_distribution_sample_accepts_count_and_seeded_generator` — Checks that public distribution sampling accepts a sample count and caller-supplied random generator.
- `test_public_distribution_sample_can_validate_realizations` — Checks that public distribution sampling accepts the realization-validation option.
- `test_random_variable_can_raise_its_graph_size_limit` — Checks that a deep, valid random-variable expression can request a larger graph limit for sampling.

## Integration tests

### `tests/integration_tests/categorical.py`

- `test_tuple_category_is_compared_as_one_value` — Checks that a tuple category remains one categorical outcome during equality probability evaluation.
- `test_list_category_is_compared_as_one_value` — Checks that a list category remains one categorical outcome during equality probability evaluation.
- `test_object_categories_support_realization_validation` — Checks that opaque object categories can be sampled with realization validation enabled.
- `test_mixed_numeric_categories_keep_types_through_realization` — Checks that mixed numeric category values retain their original types after sampling.

### `tests/integration_tests/composed_workflows.py`

- `test_distribution_can_use_random_variable_parameters_through_sampling` — Checks that a distribution samples correctly when one of its parameters is a random variable.
- `test_probability_uses_one_shared_realization_for_related_events` — Checks that related events reuse a single realization rather than being sampled independently.
- `test_probability_result_provides_a_confidence_interval_after_evaluation` — Checks that a public probability result can produce a confidence interval after an event evaluation.
- `test_interval_estimation_works_for_a_composed_random_variable` — Checks interval estimation on a random variable built from another random variable.

### `tests/integration_tests/power.py`

- `test_real_power_uses_real_samples` — Checks that a power expression inferred as real produces real-valued samples.
- `test_complex_power_keeps_complex_samples` — Checks that a power expression requiring complex values preserves a complex sample representation.

## Property tests

### `tests/property_tests/test_categorical_and_arithmetic.py`

- `test_categorical_configuration_merges_equal_values_and_normalizes_probabilities` — Checks across many generated configurations that equal categories merge and resulting probabilities sum to one.
- `test_arithmetic_identities_hold_for_many_integer_categorical_variables` — Checks arithmetic identity rules across many generated integer categorical variables.
- `test_compound_categories_remain_atomic_values_through_sampling` — Checks across generated compound categories that sampling preserves each category as one atomic object.

### `tests/property_tests/test_probability_results.py`

- `test_sample_count_and_confidence_interval_hold_for_many_valid_counts` — Checks across many valid count combinations that sample-count selection and confidence-interval construction remain valid.

## Regression tests

### `tests/regression_tests/test_binomial_support.py`

- `test_finite_binomial_support_uses_symbolic_range` — Checks that finite binomial support is represented symbolically without expanding every integer outcome.

### `tests/regression_tests/test_categorical_values.py`

- `test_duplicate_categories_are_merged_before_sampling` — Checks that duplicate categorical outcomes merge before sampling and their probabilities are added.
- `test_compound_category_is_an_atomic_value_during_probability_evaluation` — Checks that a compound category compares as one object during public probability evaluation.

### `tests/regression_tests/test_cdf_nan.py`

- `test_scalar_nan_is_rejected_by_cdf` — Checks that a scalar NaN input to CDF raises a validation error.
- `test_array_nan_is_rejected_by_cdf` — Checks that an array containing NaN is rejected by CDF validation.

### `tests/regression_tests/test_float_boundary_validation.py`

- `test_tanh_rounding_to_one_remains_valid` — Checks that `tanh` rounding to exactly one does not contradict its declared support during validation.
- `test_exponential_underflow_to_zero_remains_valid` — Checks that exponential underflow to exactly zero does not contradict its declared support during validation.

### `tests/regression_tests/test_graph_labels.py`

- `test_poisson_name_identifies_the_distribution` — Checks that a Poisson distribution name identifies the Poisson family.
- `test_categorical_name_identifies_the_distribution` — Checks that a categorical distribution name identifies the categorical family.
- `test_mathematical_function_node_names_the_function` — Checks that a mathematical wrapper’s graph node name includes both the wrapper name and source variable name.

### `tests/regression_tests/test_integer_membership.py`

- `test_integer_value_set_contains_integer_valued_float` — Checks that an integer-valued floating scalar is recognized as a member of the integer value set.
- `test_integer_value_set_checks_each_float_in_array` — Checks that integer membership is evaluated independently for each floating array element.
- `test_integer_valued_float_matches_integer_set_event` — Checks that an integer-valued floating category satisfies a public integer-set event with validation enabled.

### `tests/regression_tests/test_integer_overflow.py`

- `test_fixed_width_integer_addition_does_not_silently_wrap` — Checks that fixed-width integer addition either widens safely or raises instead of wrapping.
- `test_fixed_width_integer_multiplication_does_not_silently_wrap` — Checks that fixed-width integer multiplication either widens safely or raises instead of wrapping.

### `tests/regression_tests/test_numeric_validation_consistency.py`

- `test_numpy_integer_is_accepted_as_binomial_trial_count` — Checks that a NumPy integer is accepted wherever a binomial trial count is accepted.
- `test_numpy_float_is_accepted_as_distribution_parameter` — Checks that NumPy floating values are accepted as normal and Poisson parameters.
- `test_boolean_is_rejected_as_binomial_trial_count` — Checks that a boolean is rejected as a binomial trial count with a type error.
- `test_boolean_is_rejected_as_probability_or_rate` — Checks that booleans are rejected as binomial probabilities, Poisson rates, and categorical probabilities.

### `tests/regression_tests/test_probability_result_confidence_interval.py`

- `test_confidence_interval_is_available_from_a_public_probability_result` — Checks that a public `ProbabilityResult` can construct its confidence interval.

### `tests/regression_tests/test_probability_result_consistency.py`

- `test_probability_must_match_success_count` — Checks that a reported unconditional probability agrees with its stated success count.
- `test_conditional_probability_must_match_conditioned_count` — Checks that a reported conditional probability agrees with successes and conditioned sample count.

### `tests/regression_tests/test_quantile_confidence_interval_samples.py`

- `test_insufficient_samples_are_rejected_before_constructing_an_interval` — Checks that quantile confidence intervals reject an insufficient sample before creating bounds.

### `tests/regression_tests/test_quantile_search_cost.py`

- `test_binomial_tail_search_uses_sublinear_number_of_calls` — Checks that quantile-interval tail selection does not make a linear number of binomial CDF and survival-function calls.

### `tests/regression_tests/test_real_powers.py`

- `test_square_of_a_negative_value_remains_compatible_with_sqrt` — Checks that squaring a negative value produces a result accepted by the square-root wrapper.

## Statistical tests

### `tests/statistical_tests/test_analytical_workflows.py`

- `test_binomial_point_probability_matches_finite_formula` — Compares an estimated binomial point probability with the finite binomial formula.
- `test_poisson_lower_tail_matches_finite_formula` — Compares an estimated Poisson lower-tail probability with the finite Poisson sum.
- `test_normal_upper_tail_matches_error_function` — Compares an estimated normal upper-tail probability with the analytical error-function result.
- `test_conditional_binomial_probability_matches_ratio_of_finite_sums` — Compares a conditional binomial probability with an independently calculated ratio of finite sums.
- `test_categorical_event_logic_matches_sum_of_category_weights` — Compares categorical event logic with the sum of the corresponding category probabilities.
- `test_categorical_complement_and_conditional_probability_match_weights` — Compares categorical complements and conditional probabilities with their known category weights.
- `test_compound_object_category_probability_matches_its_weight` — Checks that an opaque compound category has the probability assigned to that exact category.
- `test_interval_closure_matches_discrete_category_weights` — Checks that open and closed interval events include exactly the intended discrete category weights.
- `test_finite_set_membership_matches_category_weights` — Checks that finite-set membership probability equals the total probability of the matching categories.
- `test_affine_transform_of_binomial_matches_original_tail_probability` — Checks that an affine binomial transformation preserves the analytically equivalent tail probability.
- `test_nonlinear_function_composition_matches_discrete_weights` — Checks a nonlinear random-variable composition against independently summed discrete weights.
- `test_exponential_transform_of_normal_matches_analytical_cdf` — Compares the CDF of an exponentiated normal variable with its analytical transformed-normal CDF.
- `test_reused_random_variable_keeps_its_dependence` — Checks that using the same random variable twice preserves dependence in the resulting event.
- `test_complex_power_followed_by_absolute_value_matches_finite_weights` — Checks a complex power followed by absolute value against finite categorical probability weights.
- `test_modulo_and_reverse_division_match_finite_category_weights` — Checks modulo and reverse-division expressions against independently summed categorical weights.
- `test_sum_of_independent_binomials_matches_finite_convolution` — Checks the sum of independent binomials against the finite convolution of their probabilities.

### `tests/statistical_tests/test_confidence_interval_coverage.py`

- `test_probability_confidence_intervals_cover_true_categorical_probability` — Repeats categorical probability estimation to check that confidence-interval coverage matches its stated confidence level.
- `test_random_variable_quantile_confidence_intervals_cover_true_median` — Repeats random-variable median intervals to check coverage of the true median.
- `test_distribution_quantile_confidence_intervals_cover_true_median` — Repeats distribution median intervals to check coverage of the true median.

### `tests/statistical_tests/test_derived_functions.py`

- `test_floor_of_normal_matches_difference_of_analytical_cdfs` — Compares a floored normal-variable probability with the difference of analytical normal CDF values.
- `test_log_of_exponential_normal_matches_original_normal_tail` — Checks that taking log after exponentiating a normal variable restores the matching normal-tail probability.
- `test_trigonometric_function_matches_weighted_finite_categories` — Checks a trigonometric transformation of finite categories against their weighted analytical result.
- `test_custom_function_of_two_independent_variables_matches_finite_sum` — Checks a custom two-variable function against an independently calculated finite double sum.

### `tests/statistical_tests/test_distribution_methods.py`

- `test_normal_mean_variance_and_std_match_their_sampling_laws` — Checks Monte Carlo normal mean, variance, and standard deviation using their correct sampling-law bounds.
- `test_normal_cdf_array_matches_analytical_normal_cdf` — Checks a normal CDF array estimate against analytical normal CDF values.
- `test_normal_ppf_array_lies_within_analytical_dkw_bounds` — Checks normal quantile estimates against DKW confidence bounds.
- `test_binomial_cdf_and_ppf_match_finite_distribution` — Checks binomial CDF and inverse-CDF estimates against finite-distribution calculations.
- `test_random_variable_probability_interval_matches_normal_quantiles` — Checks a random-variable probability interval against normal-distribution quantiles.

### `tests/statistical_tests/test_distribution_sampling.py`

- `test_normal_sample_mean_matches_configured_mean` — Checks that a large normal sample mean is statistically consistent with the configured mean.
- `test_binomial_sample_mean_matches_configured_mean` — Checks that a large binomial sample mean is statistically consistent with the configured mean.
- `test_poisson_sample_mean_matches_configured_mean` — Checks that a large Poisson sample mean is statistically consistent with the configured mean.
- `test_categorical_sample_frequencies_match_configured_probabilities` — Checks that categorical sample frequencies are statistically consistent with configured probabilities.
- `test_probability_estimate_matches_a_categorical_event_probability` — Checks that public probability estimation matches a known categorical event probability.

### `tests/statistical_tests/test_random_parameters.py`

- `test_random_binomial_probability_matches_weighted_analytical_tails` — Checks a binomial with random probability against the weighted mixture of analytical tails.
- `test_random_binomial_trial_count_matches_weighted_analytical_tails` — Checks a binomial with random trial count against the weighted mixture of analytical tails.
- `test_random_normal_mean_matches_weighted_analytical_cdfs` — Checks a normal with random mean against the weighted mixture of analytical CDFs.
- `test_conditioning_on_shared_poisson_rate_matches_component_distribution` — Checks conditioning on a shared random Poisson rate against the selected component distribution.

## Unit tests

### `tests/unit_tests/_events.py`

- `test_constructor_exposes_boolean_node_name_and_representation` — Checks that an event exposes its boolean node’s name and useful representation.
- `test_constructor_rejects_non_boolean_node` — Checks that constructing an event from a non-boolean node raises an error.
- `test_logical_operations_build_boolean_operation_events` — Checks that logical event operators construct boolean operation events.
- `test_logical_operations_evaluate_each_boolean_array` — Checks that logical event operators calculate elementwise Boolean array results.
- `test_logical_operations_reject_non_events` — Checks that logical event operators reject operands that are not events.
- `test_event_has_no_single_truth_value` — Checks that an event cannot be coerced into one Python Boolean value.

### `tests/unit_tests/_operations.py`

- `test_operation_constants_have_expected_types_and_names` — Checks that internal operation constants expose their intended operation types and names.
- `test_divide_and_modulo_handle_zero_without_warnings` — Checks division and modulo behavior at zero without emitting numerical warnings.
- `test_power_promotes_integer_base_for_negative_exponents_and_handles_complex_output` — Checks that power promotes integer bases for negative exponents and preserves complex outputs where required.
- `test_logical_and_comparison_operations_apply_elementwise` — Checks that logical and comparison operations act elementwise on sample arrays.
- `test_unary_arithmetic_operations_preserve_expected_values_and_names` — Checks unary arithmetic values and human-readable operation names.

### `tests/unit_tests/distributions/_config.py`

- `test_defaults` — Checks the default distribution configuration constants used throughout the package.

### `tests/unit_tests/distributions/base.py`

- `test_mode_members` — Checks that `Mode` has exactly the declared automatic, exact, and Monte Carlo options.
- `test_constructor_preserves_parameters_and_builds_constant_nodes` — Checks that a distribution keeps public parameters while creating corresponding constant parameter nodes.
- `test_parameter_to_node_preserves_existing_node` — Checks that an already internal node is reused rather than wrapped again.
- `test_parameter_to_node_uses_a_random_variable_node_without_changing_public_input` — Checks that a random-variable parameter uses its internal node without replacing the public parameter object.
- `test_node_dependencies_include_variable_parameters_but_not_constants` — Checks that only non-constant parameter nodes become distribution dependencies.
- `test_sample_creates_context_and_evaluates_distribution_node` — Checks that `Distribution.sample()` creates a realization context and evaluates its root distribution node.
- `test_sample_passes_count_rng_and_validation_to_context` — Checks that public distribution sampling forwards count, generator, and validation arguments to the realization context.
- `test_evaluate_realizes_parameter_nodes_and_delegates_to_sample` — Checks that internal distribution evaluation realizes parameters before invoking `_sample()`.
- `test_exact_statistics_are_used_for_exact_and_auto_modes` — Checks that available exact mean and variance values are used in exact and automatic modes.
- `test_monte_carlo_statistics_delegate_to_monte_carlo` — Checks that mean, variance, and standard deviation delegate to Monte Carlo in Monte Carlo mode.
- `test_auto_statistics_fall_back_to_monte_carlo_when_exact_values_are_unavailable` — Checks that automatic statistics use Monte Carlo when an exact hook returns no result.
- `test_exact_mode_rejects_unavailable_statistics` — Checks that exact statistical methods report an unavailable exact implementation.
- `test_cdf_and_ppf_use_exact_values_when_available` — Checks that CDF and PPF use available exact-hook results in exact and automatic modes.
- `test_cdf_and_ppf_delegate_to_monte_carlo_when_requested` — Checks that explicitly requested Monte Carlo CDF and PPF use sampling.
- `test_auto_cdf_and_ppf_fall_back_to_monte_carlo_when_exact_values_are_unavailable` — Checks automatic CDF and PPF fallback when exact hooks have no answer.
- `test_monte_carlo_cdf_operations_handle_scalar_and_array_inputs` — Checks Monte Carlo CDF calculation for both scalar thresholds and threshold arrays.
- `test_monte_carlo_ppf_operation_uses_requested_quantile_method` — Checks that Monte Carlo PPF uses the requested NumPy quantile method.
- `test_monte_carlo_realizes_samples_and_normalizes_scalar_results` — Checks sampling through a context and scalar-result normalization in Monte Carlo helpers.
- `test_monte_carlo_preserves_array_results` — Checks that Monte Carlo helpers preserve a vector or array result.
- `test_quantile_confidence_interval_realizes_samples_and_delegates` — Checks that a distribution realizes samples and passes them with parameters to quantile-interval calculation.
- `test_validation_rejects_invalid_distribution_arguments` — Checks validation of mode, sample count, RNG, CDF input, PPF input, and quantile method arguments.
- `test_continuous_and_discrete_base_classes_remain_abstract` — Checks that the specialized continuous and discrete base classes cannot be instantiated directly.

### `tests/unit_tests/distributions/continuous/normal.py`

- `test_configuration_and_public_properties` — Checks normal-distribution parameters, symbol, name, and real-valued support.
- `test_sample_delegates_to_scipy_with_parameters` — Checks that normal `_sample()` passes mean, standard deviation, sample count, and generator to SciPy.
- `test_invalid_mean_and_standard_deviation_are_rejected` — Checks normal constructor validation for invalid means and standard deviations.

### `tests/unit_tests/distributions/discrete/binomial.py`

- `test_configuration_and_finite_support` — Checks binomial parameters, name, and support for a finite fixed trial count.
- `test_zero_n_has_single_value_support` — Checks that a binomial distribution with zero trials has only zero in its support.
- `test_sample_delegates_to_scipy_with_parameters` — Checks that binomial `_sample()` forwards parameters, sample count, and generator to SciPy.
- `test_invalid_parameters_are_rejected` — Checks binomial constructor validation for invalid trial counts and probabilities.

### `tests/unit_tests/distributions/discrete/categorical.py`

- `test_constructor_preserves_categories_and_probabilities` — Checks that categorical construction retains configured categories and probabilities.
- `test_equal_categories_are_merged_and_probabilities_are_summed` — Checks that equal categorical outcomes merge and their probabilities are summed.
- `test_numeric_categories_use_numeric_value_set` — Checks that homogeneous numeric categories receive a numeric value set.
- `test_mixed_numeric_categories_preserve_original_types` — Checks that mixed numeric categories preserve their exact original scalar types.
- `test_sample_uses_rng_indices_and_configured_probabilities` — Checks that categorical `_sample()` uses chosen indices and configured probability weights.
- `test_configuration_validation_rejects_invalid_inputs` — Checks categorical configuration validation for invalid categories and probability lists.
- `test_fraction_and_integer_categories_use_mixed_numeric_values` — Checks that fractions combined with integers use the mixed numeric value-set representation.

### `tests/unit_tests/distributions/discrete/helpers/_categorical.py`

- `test_numeric_category_detection` — Checks detection of category collections that are numeric.
- `test_untyped_categories_remain_atomic_objects` — Checks that untyped categories are stored as atomic object values.
- `test_mixed_numeric_categories_remain_atomic_objects` — Checks that mixed numeric categories remain separate atomic category values.
- `test_inference_selects_homogeneous_mixed_or_object_value_sets` — Checks categorical inference chooses homogeneous numeric, mixed numeric, or object value sets appropriately.
- `test_equal_categories_are_merged_in_order` — Checks equal categories merge while preserving their first-occurrence order.

### `tests/unit_tests/distributions/discrete/poisson.py`

- `test_configuration_and_support` — Checks Poisson parameters, name, and natural-number support.
- `test_sample_delegates_to_scipy_with_parameters` — Checks that Poisson `_sample()` forwards rate, sample count, and generator to SciPy.
- `test_invalid_rate_is_rejected` — Checks Poisson constructor validation for invalid rates.

### `tests/unit_tests/functions/_utils.py`

- `test_apply_scalar_or_rv_returns_scalar_function_result` — Checks scalar application returns the scalar function result unchanged.
- `test_apply_scalar_or_rv_delegates_random_variable_to_apply` — Checks random-variable scalar-or-RV application delegates to `RandomVariable.apply()`.
- `test_apply_converts_scalar_result_to_float` — Checks the generic application helper converts scalar numerical results to `float`.
- `test_apply_delegates_random_variable_to_apply` — Checks the generic application helper delegates random-variable inputs to `apply()`.

### `tests/unit_tests/functions/basic.py`

- `test_sqrt_validates_non_negative_domain_and_applies_numpy_sqrt` — Checks square root domain validation, NumPy delegation, and declared output support.
- `test_absolute_validates_real_input_and_declares_non_negative_output` — Checks absolute value validates real inputs and declares non-negative support.
- `test_floor_validates_real_input_and_declares_integer_output` — Checks floor validates real inputs and declares integer support.
- `test_ceil_validates_real_input_and_declares_integer_output` — Checks ceiling validates real inputs and declares integer support.
- `test_sign_validates_real_input_and_declares_three_possible_outputs` — Checks sign validation and its three-element output support.

### `tests/unit_tests/functions/exponential.py`

- `test_exp_validates_real_input_and_declares_positive_output` — Checks exponential validation and its declared positive-real output support.
- `test_log_validates_positive_domain_and_applies_numpy_log` — Checks natural-log domain validation and NumPy delegation.
- `test_log2_validates_positive_domain_and_applies_numpy_log2` — Checks base-two-log domain validation and NumPy delegation.
- `test_log10_validates_positive_domain_and_applies_numpy_log10` — Checks base-ten-log domain validation and NumPy delegation.

### `tests/unit_tests/functions/trigonometric.py`

- `test_sin_validates_real_input_and_declares_closed_unit_interval` — Checks sine validation and its closed unit-interval support.
- `test_cos_validates_real_input_and_declares_closed_unit_interval` — Checks cosine validation and its closed unit-interval support.
- `test_tan_validates_real_input_and_declares_real_output` — Checks tangent validation and real-valued output support.
- `test_arcsin_validates_closed_unit_interval_and_declares_output_range` — Checks arcsine input validation and declared output interval.
- `test_arccos_validates_closed_unit_interval_and_declares_output_range` — Checks arccosine input validation and declared output interval.
- `test_arctan_validates_real_input_and_declares_open_output_range` — Checks arctangent validation and declared open output interval.
- `test_sinh_validates_real_input_and_declares_real_output` — Checks hyperbolic sine validation and real-valued output support.
- `test_cosh_validates_real_input_and_declares_output_at_least_one` — Checks hyperbolic cosine validation and lower-bounded output support.
- `test_tanh_validates_real_input_and_declares_open_unit_interval` — Checks hyperbolic tangent validation and declared open unit interval.
- `test_arcsinh_validates_real_input_and_declares_real_output` — Checks inverse hyperbolic sine validation and real output support.
- `test_arccosh_validates_lower_bound_and_declares_non_negative_output` — Checks inverse hyperbolic cosine lower-bound validation and non-negative output support.
- `test_arctanh_validates_open_unit_interval_and_declares_real_output` — Checks inverse hyperbolic tangent domain validation and real output support.

### `tests/unit_tests/probability/_config.py`

- `test_default_sample_count_is_within_configured_limit` — Checks that the default probability sample count respects the configured maximum.

### `tests/unit_tests/probability/intervals.py`

- `test_constructor_preserves_configuration` — Checks that a confidence interval preserves its bounds and alpha configuration.
- `test_constructor_allows_two_nan_bounds_for_an_undefined_interval` — Checks that an undefined confidence interval may use two NaN bounds.
- `test_constructor_rejects_invalid_configuration` — Checks confidence-interval constructor validation for invalid configurations.
- `test_properties_report_nominal_coverage_and_bounds` — Checks that a probability interval reports its nominal coverage and configured bounds.
- `test_constructor_rejects_invalid_configuration` — Checks probability-interval constructor validation for invalid bounds, alpha, and estimate flags.

### `tests/unit_tests/probability/probability.py`

- `test_unconditional_probability_counts_true_values` — Checks unconditional `P` counts true event realizations and returns matching metadata.
- `test_conditional_probability_counts_joint_successes_within_condition` — Checks conditional `P` counts joint successes over only conditioned realizations.
- `test_conditional_probability_returns_nan_when_condition_never_occurs` — Checks conditional `P` returns NaN when the conditioning event has no realizations.
- `test_rejects_invalid_public_arguments` — Checks public probability validation for event, condition, sample count, generator, and validation arguments.

### `tests/unit_tests/probability/results.py`

- `test_num_samples_uses_unconditioned_count_without_condition` — Checks `ProbabilityResult.num_samples` selects the unconditioned count when there is no condition.
- `test_num_samples_uses_conditioned_count_when_present` — Checks `ProbabilityResult.num_samples` selects the conditioned count when present.
- `test_confidence_interval_delegates_counts_and_alpha` — Checks confidence-interval creation delegates the effective count, successes, and alpha correctly.
- `test_constructor_allows_nan_for_zero_conditioned_samples` — Checks that zero conditioned samples may be represented by a NaN probability.
- `test_constructor_rejects_invalid_configuration` — Checks `ProbabilityResult` construction rejects invalid counts and probability values.

### `tests/unit_tests/random_variables/_config.py`

- `test_default_graph_size_is_positive` — Checks that the default dependency-graph size limit is positive.

### `tests/unit_tests/random_variables/_context.py`

- `test_constructor_stores_sample_count_and_supplied_rng` — Checks that a realization context retains the requested sample count and supplied generator.
- `test_constructor_creates_a_generator_when_rng_is_not_supplied` — Checks that a realization context creates a NumPy generator when none is supplied.
- `test_constructor_rejects_invalid_sample_count` — Checks that invalid realization-context sample counts are rejected.
- `test_evaluate_caches_node_result` — Checks that evaluating the same node twice reuses its cached result.
- `test_evaluate_broadcasts_scalar_result_to_sample_count` — Checks that scalar node results broadcast to the configured sample length.
- `test_evaluate_rejects_wrong_sample_shape` — Checks that a node result with an incompatible sample shape is rejected.
- `test_evaluate_rejects_dtype_outside_declared_family` — Checks that a node result outside its declared dtype family is rejected.
- `test_evaluate_accepts_any_dtype_when_value_set_has_no_dtype_family` — Checks that an unspecified dtype family accepts any result dtype.
- `test_evaluate_validates_result_when_requested` — Checks that realization validation calls subset validation when requested.
- `test_evaluate_releases_dependency_after_last_dependant_is_realized` — Checks that a cached dependency is released after its last dependent has been realized.
- `test_constructor_rejects_incomplete_dependency_graph` — Checks that a realization context rejects a graph truncated by its maximum-size limit.

### `tests/unit_tests/random_variables/_nodes.py`

- `test_compatibility_module_reexports_current_node_types` — Checks that the compatibility module re-exports the current internal node classes.

### `tests/unit_tests/random_variables/base.py`

- `test_constructor_creates_distribution_node_with_explicit_name` — Checks that random-variable construction creates a distribution node with the supplied name.
- `test_from_node_preserves_node_and_explicit_name` — Checks the internal node constructor retains the given node and explicit name.
- `test_generated_names_are_distinct_when_not_supplied` — Checks generated random-variable names are distinct when no name is supplied.
- `test_dependency_graph_uses_default_graph_limit` — Checks `dependency_graph` constructs a graph using the default maximum size.
- `test_plot_dependencies_delegates_to_graph` — Checks dependency plotting delegates to the graph with the supplied maximum size.
- `test_sample_creates_context_and_evaluates_own_node` — Checks sampling creates a realization context and evaluates the variable’s own node.
- `test_is_real_or_complex_reflects_declared_value_set` — Checks numeric capability detection reflects the node’s declared value set.
- `test_realize_delegates_to_sample_defaults` — Checks `realize()` calls sampling with its default arguments.
- `test_binary_operation_builds_operation_node_with_scalar_constant` — Checks a binary operation wraps a scalar operand as a constant node and builds the correct operation node.
- `test_reverse_binary_operation_reverses_input_order` — Checks reverse arithmetic uses the scalar node as the left input.
- `test_public_arithmetic_methods_delegate_with_correct_operations` — Checks public arithmetic operators choose their corresponding internal operations.
- `test_binary_operation_rejects_unsupported_operand` — Checks binary operations reject operands of unsupported Python types.
- `test_binary_operation_rejects_non_numeric_value_set` — Checks binary operations reject random variables with nonnumeric value sets.
- `test_real_power_uses_real_numpy_power_operation` — Checks powers inferred as real use NumPy’s real power operation.
- `test_unary_operation_builds_node` — Checks a unary operation creates an operation node with expected input and value set.
- `test_public_unary_methods_delegate_with_correct_operations` — Checks public unary operators choose their corresponding internal operations.
- `test_apply_wraps_non_vectorized_function_for_sample_arrays` — Checks `apply()` adapts a scalar function to aligned sample arrays.
- `test_apply_preserves_vectorized_function` — Checks `apply()` passes a vectorized function through unchanged.
- `test_apply_passes_aligned_values_from_other_variables` — Checks `apply()` passes aligned realizations from every additional random variable.
- `test_interval_uses_inverted_cdf_quantiles_of_samples` — Checks random-variable interval estimation uses inverted-CDF sample quantiles.
- `test_is_in_interval_rejects_reversed_bounds` — Checks interval membership rejects a lower bound greater than the upper bound.
- `test_is_in_interval_honors_each_closure_mode` — Checks interval membership creates the correct comparisons for all four closure modes.
- `test_is_in_delegates_tuple_and_list_ranges_with_expected_closure` — Checks tuple and list range shorthand choose open and closed interval semantics respectively.
- `test_is_in_builds_boolean_event_for_sympy_set` — Checks SymPy set membership builds a Boolean event node that evaluates correctly.
- `test_comparison_builds_boolean_event` — Checks comparison with a scalar builds a Boolean event node.
- `test_public_comparison_methods_delegate_with_correct_operations` — Checks public comparison operators choose their corresponding internal comparisons.
- `test_quantile_confidence_interval_delegates_sample_and_parameters` — Checks quantile confidence intervals sample once and pass the requested quantile and alpha parameters onward.
- `test_real_only_methods_reject_non_real_value_sets` — Checks interval and quantile methods reject variables with non-real supports.

### `tests/unit_tests/random_variables/graph.py`

- `test_graph_contains_nodes_and_edges_from_root_to_dependencies` — Checks a graph contains every reachable dependency and root-to-dependency edge.
- `test_graph_stops_when_maximum_size_is_reached` — Checks graph construction marks itself incomplete when the maximum node count is reached.
- `test_constructor_rejects_non_positive_maximum_size` — Checks graph construction rejects non-positive maximum sizes.
- `test_graph_handles_cycles_without_duplicate_nodes_or_recursion` — Checks cyclic dependency input does not produce duplicate nodes or unbounded recursion.
- `test_plot_uses_selected_node_labels` — Checks plotting uses standard or extended labels according to its option.

### `tests/unit_tests/random_variables/nodes/base.py`

- `test_base_node_remains_abstract` — Checks that the internal base node cannot be instantiated directly.
- `test_name_and_extended_name_are_exposed` — Checks nodes expose their normal and extended names.
- `test_has_dependencies_reflects_direct_dependencies` — Checks dependency presence reflects whether a node has direct dependencies.

### `tests/unit_tests/random_variables/nodes/constant.py`

- `test_numeric_constant_exposes_value_name_and_empty_dependencies` — Checks a numeric constant exposes its value, name, value set, and no dependencies.
- `test_evaluate_returns_scalar_array_for_scalar_value` — Checks constant-node evaluation returns its scalar array representation.
- `test_compound_constant_remains_atomic_object` — Checks a compound constant is stored as a single atomic object array value.

### `tests/unit_tests/random_variables/nodes/distribution.py`

- `test_properties_are_forwarded_from_distribution` — Checks a distribution node forwards its distribution, support, dependencies, and names.
- `test_evaluate_delegates_to_distribution` — Checks distribution-node evaluation delegates to the wrapped distribution.

### `tests/unit_tests/random_variables/nodes/operation.py`

- `test_properties_expose_operation_inputs_and_value_set` — Checks an operation node exposes its operation, input nodes, and declared value set.
- `test_evaluate_realizes_inputs_then_calls_operation` — Checks operation-node evaluation realizes inputs before calling the wrapped operation.

### `tests/unit_tests/random_variables/nodes/_utils.py`

- `test_sympy_constant_value_converts_integral_float_to_integer` — Checks that a finite integral float becomes a SymPy integer constant.
- `test_sympy_constant_value_preserves_non_integral_float` — Checks that a non-integral float remains a floating SymPy value.
- `test_constant_array_keeps_scalar_array` — Checks scalar constants are stored as scalar NumPy arrays.
- `test_constant_array_keeps_compound_value_atomic` — Checks compound constants become one atomic object-array item.
- `test_constant_value_set_for_numeric_value_uses_sympy_and_dtype` — Checks numeric constants infer a SymPy support and compatible dtype family.
- `test_constant_value_set_for_unrepresentable_object_preserves_object` — Checks non-SymPy objects retain object membership in their constant value set.

### `tests/unit_tests/statistics/_clopper_pearson.py`

- `test_zero_samples_returns_undefined_interval_without_beta_calls` — Checks zero samples produce an undefined interval without requesting beta quantiles.
- `test_zero_successes_sets_lower_bound_to_zero` — Checks zero successes force the Clopper–Pearson lower bound to zero.
- `test_all_successes_sets_upper_bound_to_one` — Checks all successes force the Clopper–Pearson upper bound to one.
- `test_interior_success_count_uses_beta_lower_and_upper_quantiles` — Checks an interior success count uses the correct beta lower and upper quantiles.
- `test_rejects_invalid_counts_and_alpha` — Checks Clopper–Pearson calculation rejects invalid counts and alpha values.

### `tests/unit_tests/statistics/_quantiles.py`

- `test_quantile_method_lists_supported_numpy_methods` — Checks that the quantile-method type lists supported NumPy method names.
- `test_rejects_invalid_quantile_alpha_and_empty_samples` — Checks quantile intervals reject invalid quantiles, alpha values, and empty samples.
- `test_rejects_sample_size_below_confidence_requirement` — Checks quantile intervals reject samples below the confidence requirement.
- `test_uses_outer_order_statistics_when_tail_probabilities_are_large` — Checks that large tail probabilities select outer order statistics.
- `test_moves_order_statistics_inward_until_tail_probability_exceeds_alpha` — Checks that order statistics move inward until each tail exceeds the alpha threshold.

### `tests/unit_tests/validation/_common.py`

- `test_num_samples_accepts_numpy_integer_and_rejects_invalid_values` — Checks sample-count validation accepts NumPy integers and rejects invalid values.
- `test_rng_accepts_generator_or_none` — Checks RNG validation accepts a NumPy generator or `None` only.
- `test_validate_requires_bool` — Checks realization-validation flags must be Python booleans.
- `test_alpha_and_q_require_open_unit_interval` — Checks alpha and quantile arguments require the open interval from zero to one.
- `test_enum_requires_declared_member` — Checks enum validation accepts only members of the requested enum class.
- `test_max_size_requires_positive_integer` — Checks maximum-size validation requires a positive integer.

### `tests/unit_tests/validation/_decorator.py`

- `test_validates_explicit_and_default_parameter_values` — Checks the parameter-validation decorator validates both explicit and default arguments.
- `test_preserves_function_metadata` — Checks the decorator preserves wrapped function metadata.
- `test_rejects_unknown_parameter_and_non_callable_validator` — Checks decorator construction rejects unknown parameters and non-callable validators.
- `test_preserves_normal_argument_binding_errors` — Checks the decorator retains ordinary Python argument-binding errors.

### `tests/unit_tests/validation/distributions/_base.py`

- `test_real_input_accepts_real_scalars_and_numeric_non_complex_arrays` — Checks real-input validation accepts real scalars and numeric arrays without complex values.
- `test_real_input_rejects_bool_text_and_complex_values` — Checks real-input validation rejects booleans, text, and complex values.
- `test_ppf_input_requires_values_in_closed_unit_interval` — Checks PPF input validation requires values in the closed unit interval.
- `test_quantile_method_requires_supported_name` — Checks quantile-method validation accepts only supported method names.

### `tests/unit_tests/validation/distributions/continuous/_normal.py`

- `test_mean_accepts_real_scalar_and_rejects_invalid_values` — Checks normal-mean validation accepts real scalars and rejects invalid values.
- `test_std_requires_positive_real_scalar` — Checks normal standard deviation validation requires a positive real scalar.

### `tests/unit_tests/validation/distributions/discrete/_binomial.py`

- `test_n_accepts_natural_zero_and_rejects_invalid_values` — Checks binomial trial-count validation accepts natural numbers including zero and rejects invalid values.
- `test_p_accepts_closed_unit_interval_and_rejects_invalid_values` — Checks binomial probability validation accepts the closed unit interval and rejects invalid values.

### `tests/unit_tests/validation/distributions/discrete/_categorical.py`

- `test_category_and_probability_inputs_require_iterables` — Checks categorical category and probability inputs must be iterable.
- `test_configuration_accepts_matching_normalized_probabilities` — Checks categorical configuration accepts matching categories and normalized probability weights.
- `test_configuration_rejects_invalid_category_probability_pairs` — Checks categorical configuration rejects invalid category and probability combinations.

### `tests/unit_tests/validation/distributions/discrete/_poisson.py`

- `test_mu_accepts_non_negative_real_and_rejects_invalid_values` — Checks Poisson rate validation accepts non-negative real values and rejects invalid values.

### `tests/unit_tests/validation/functions/_common.py`

- `test_real_valued_accepts_real_scalar_and_real_random_variable` — Checks function validation accepts real scalars and random variables with real supports.
- `test_real_valued_rejects_bool_text_and_object_random_variable` — Checks function validation rejects booleans, text, and object-valued random variables.
- `test_domain_accepts_member_and_rejects_outside_values` — Checks domain validation accepts members and rejects values outside the given domain.

### `tests/unit_tests/validation/probability/_intervals.py`

- `test_interval_bound_allows_nan_only_when_requested` — Checks interval-bound validation permits NaN only for explicitly allowed undefined intervals.
- `test_confidence_interval_configuration_accepts_ordered_or_two_nan_bounds` — Checks confidence-interval validation accepts ordered bounds or the special two-NaN case.
- `test_probability_interval_configuration_requires_ordered_finite_bounds_and_bool_flag` — Checks probability-interval validation requires ordered finite bounds and a Boolean estimate flag.

### `tests/unit_tests/validation/probability/_probability.py`

- `test_event_and_given_accept_event_and_none` — Checks probability event and condition validation accepts events and an optional `None` condition.
- `test_event_and_given_reject_other_values` — Checks probability event and condition validation rejects other value types.

### `tests/unit_tests/validation/probability/_results.py`

- `test_non_negative_integer_accepts_numpy_integer` — Checks result-count validation accepts NumPy integer values.
- `test_configuration_accepts_unconditional_and_zero_conditioned_results` — Checks result validation accepts valid unconditional results and NaN for zero conditioned samples.
- `test_configuration_rejects_invalid_counts_and_probability` — Checks result validation rejects invalid counts and probability values.

### `tests/unit_tests/validation/random_variables/_base.py`

- `test_distribution_and_name_accept_expected_values` — Checks random-variable distribution and name validation accepts expected values.
- `test_interval_bounds_and_closure_are_validated` — Checks random-variable interval bounds and closure values are validated.
- `test_target_set_accepts_sympy_set_or_two_element_sequence` — Checks target-set validation accepts a SymPy set or a two-element range sequence.
- `test_function_value_set_and_name_validators` — Checks custom-function, output-value-set, and function-name validation.
- `test_others_requires_random_variables` — Checks additional custom-function operands must be random variables.

### `tests/unit_tests/validation/statistics/_clopper_pearson.py`

- `test_count_validators_accept_non_negative_integers` — Checks Clopper–Pearson count validators accept non-negative integers.
- `test_count_validators_reject_booleans_and_negative_values` — Checks Clopper–Pearson count validators reject booleans and negative values.
- `test_configuration_rejects_successes_above_samples` — Checks Clopper–Pearson configuration rejects success counts above the sample count.

### `tests/unit_tests/validation/value_sets/_base.py`

- `test_accepts_sympy_or_unknown_set_and_numpy_dtype_types` — Checks value-set configuration accepts SymPy or unknown sets and NumPy dtype families.
- `test_rejects_invalid_sympy_set` — Checks value-set configuration rejects an invalid support object.
- `test_rejects_invalid_dtype_configuration` — Checks value-set configuration rejects invalid dtype-family declarations.

### `tests/unit_tests/value_sets/_comparison.py`

- `test_compares_scalars_and_array_like_values` — Checks safe object equality works for scalars and array-like values.
- `test_returns_false_when_equality_raises` — Checks safe object equality returns false when an object’s equality operation raises.

### `tests/unit_tests/value_sets/_inference.py`

- `test_addition_identity_is_zero_not_one` — Checks the value-set inference identity for addition is zero.
- `test_zero_division_returns_wrapper` — Checks division by a known zero support returns the dedicated wrapper representation.
- `test_possible_zero_divisors_are_unknown` — Checks a divisor that may be zero produces an unknown inferred value set.
- `test_nonreal_fractional_power` — Checks a negative base with a non-integral fractional exponent infers a non-real result.
- `test_power_undefined_at_zero` — Checks power inference handles expressions undefined at zero.
- `test_power_zero_and_one_conventions` — Checks power inference follows the special exponent-zero and base-one conventions.
- `test_unknown_inputs_remain_unknown` — Checks unknown input supports remain unknown through inference.
- `test_integer_information_and_sign_are_preserved` — Checks inference preserves known integer and sign information when mathematically valid.
- `test_identity_rules_allow_numpy_dtype_promotion` — Checks inference identity rules permit NumPy dtype promotion.
- `test_real_output_from_complex_absolute_value` — Checks absolute value of a complex support infers a real output support.
- `test_object_arithmetic_representation` — Checks arithmetic on object representations uses the appropriate object-level value-set representation.
- `test_unknown_dtype_stays_unspecified` — Checks unknown support information leaves the dtype family unspecified.
- `test_exact_dtype_rules_follow_the_operation` — Checks exact inferred dtype rules correspond to the applied arithmetic operation.
- `test_integer_family_includes_signed_unsigned_promotion` — Checks integer dtype inference includes signed and unsigned promotion cases.
- `test_custom_operations_do_not_guess_dtype` — Checks custom operations do not invent a dtype family without a stated rule.
- `test_unsupported_representation_leaves_dtype_unknown` — Checks unsupported inferred representations keep dtype information unknown.
- `test_float_families_include_extended_precision` — Checks floating dtype families include supported extended-precision types.
- `test_representative_binary_results` — Checks representative binary operations infer the intended support and dtype results.
- `test_representative_unary_results` — Checks representative unary operations infer the intended support and dtype results.
- `test_representative_power_results` — Checks representative power expressions infer their intended support and dtype results.
- `test_context_accepts_promoted_outputs` — Checks realization validation accepts outputs with compatible promoted dtypes.

### `tests/unit_tests/value_sets/_unknown.py`

- `test_representation_is_stable` — Checks the unknown value-set representation remains stable and readable.

### `tests/unit_tests/value_sets/_utils.py`

- `test_is_known_subset_handles_numeric_object_and_unknown_sets` — Checks known-subset detection across numeric, object, and unknown value sets.
- `test_validate_as_subset_accepts_integer_valued_floats_for_integer_set` — Checks subset validation accepts integer-valued floats for the integer set.
- `test_validate_as_subset_rejects_outside_or_unknown_values` — Checks subset validation rejects values outside or indeterminate for the target set.
- `test_validate_as_subset_uses_object_membership` — Checks subset validation uses object-category membership for object value sets.

### `tests/unit_tests/value_sets/base.py`

- `test_value_set_remains_abstract` — Checks that the base `ValueSet` class cannot be instantiated directly.
- `test_numeric_value_set_is_value_set_marker_class` — Checks `NumericValueSet` retains its role as a `ValueSet` marker base class.
- `test_dynamic_attributes_load_concrete_value_set_classes` — Checks lazy module attributes resolve concrete value-set classes correctly.

### `tests/unit_tests/value_sets/homogeneous_numeric_value_set.py`

- `test_contains_returns_bool_for_scalar_and_array_for_array` — Checks homogeneous numeric membership returns a Boolean scalar or elementwise Boolean array as appropriate.
- `test_constructor_validates_configuration` — Checks homogeneous numeric value-set construction validates support and dtype configuration.

### `tests/unit_tests/value_sets/mixed_numeric_value_set.py`

- `test_preserves_values_and_supports_scalar_and_array_membership` — Checks mixed numeric value sets preserve values and support scalar and array membership.
- `test_constructor_requires_non_empty_tuple` — Checks mixed numeric value sets require a non-empty tuple of values.

### `tests/unit_tests/value_sets/object_value_set.py`

- `test_contains_preserves_object_identity_and_array_shape` — Checks object value-set membership respects object identity and input-array shape.
- `test_constructor_requires_non_empty_tuple` — Checks object value sets require a non-empty tuple of categories.

### `tests/unit_tests/value_sets/sets.py`

- `test_core_numeric_sets_describe_expected_membership` — Checks predefined numeric sets recognize representative members and non-members.
- `test_unknown_value_set_uses_unknown_symbolic_support` — Checks the predefined unknown set uses the unknown symbolic-support sentinel.
