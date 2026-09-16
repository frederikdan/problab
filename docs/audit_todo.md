# ProbLab audit TODO

Temporary checklist from the library audit.

Status: `[ ]` pending, `[x]` completed.

## Completed before this audit

- [x] Refactored the internal evaluation API to use underscore-prefixed node, context, event, and operation implementation names.
- [x] Kept `NodeGraph` public because `RandomVariable.dependency_graph` returns it.
- [x] Kept `Distribution.sample()` public with an internally created realization context.
- [x] Preserved original distribution parameter inputs through `Distribution.parameters` and converted them separately to internal parameter nodes.
- [x] Added `ValueSet` support for SymPy sets and NumPy dtype families.
- [x] Added optional realization validation through `validate=True`.
- [x] Added `RandomVariable.is_in()` and `is_in_interval()`.
- [x] Added a minimum-sample check for quantile confidence intervals.
- [x] Added reusable parameter validation and reorganized validators.
- [x] Decided that categorical graph parameters remain empty: `CategoricalDistribution.parameters == ()`.
- [x] Added read-only `CategoricalDistribution.categories` and `probabilities` properties.
- [x] Added explicit package-level public exports and documented the intended API.
- [x] Removed the unused `value_sets._utils.is_in()` helper and the unused categorical realization-context import.
- [x] Existing baseline suite passes: 22 tests.

## Necessary changes

### Six highest priority

- [x] Fix `ProbabilityResult.confidence_interval()`: it imports `_clopper_pearson` but calls `clopper_pearson`.
- [x] Standardize package imports. Internal modules now use `problab`, and package metadata defines the `src` layout for installation.
- [x] Make real-valued expressions keep a compatible real representation. Power nodes now use `np.power` when inference proves a real result and retain `_power_values` for complex-capable cases.
- [x] Treat each categorical item as one atomic opaque object, even if it is numeric, a tuple, list-like object, or another compound object. Numeric categories use `NumericValueSet`; other categories use `ObjectValueSet`, preserving category identity for sampling, equality, membership, and `validate=True` realization validation.
- [x] Define safe categorical support handling for ordinary labels. Object categories such as `"red apple"`, lists, tuples, and mappings no longer require a SymPy representation.
- [ ] Make set membership use the same numeric conversion as realization validation. Integer-valued floating samples such as `-2.0` should be recognized as members of `Integers`.

### Remaining necessary changes

- [ ] Define an overflow policy for fixed-width integer arithmetic. Operations such as `np.int8(100) + np.int8(100)` silently wrap around.
- [x] Explicitly support mixed numeric categorical values through `MixedNumericValueSet` while preserving their original types.
- [ ] Make CDF handling consistent for NaN and invalid domains. Scalar and array NaN inputs currently produce different results.
- [ ] Reconcile declared value sets with floating-point boundary behavior. Examples include `tanh(20.0) == 1.0` despite an open `(-1, 1)` support and `exp(-1000.0) == 0.0` despite positive support.
- [ ] Validate `ProbabilityResult` counts and value together; contradictory data such as `value=0.9` with `num_successes=0` is currently accepted.

## Recommended changes

- [ ] Add regression tests for the public API, distributions, random variables, graph dependencies, sampling, events, validation, categorical values, and all findings above.
- [ ] Rename `tests/tast_random_variables.py` to `tests/test_random_variables.py`; the current name is skipped by standard unittest discovery.
- [ ] Give `Distribution.sample()` controls matching `RandomVariable.sample()`: sample count, RNG, and optional validation.
- [ ] Make the graph evaluation limit configurable instead of always using the hard-coded maximum of 100 nodes.
- [ ] Represent binomial support symbolically instead of expanding every integer from zero through `n`.
- [ ] Improve quantile confidence interval performance; the current repeated scalar CDF search is slow for large samples.
- [ ] Make validation consistent for Python and NumPy numeric types, and reject booleans consistently where integers are expected.
- [ ] Clarify that `ProbabilityInterval.probability` is nominal coverage (`1 - alpha`), not the exact probability mass inside the returned interval.
- [ ] Improve graph labels for constants, operation nodes, mathematical wrappers, Poisson distributions, and categorical distributions.
- [ ] Implement exact distribution methods after the parameter and support contracts are stable. Random parameters require separate treatment from fixed scalar parameters.
- [x] Add package metadata and installation instructions once the import standardization is complete.

## Test organization progress

- [x] Create test-purpose folders for unit, integration, statistical, regression, property-based, and public API tests.
- [x] Mirror every `src/problab` Python file under `tests/unit_tests`.
- [x] Move the existing tests into their matching mirrored modules.
- [ ] Expand unit coverage to every mirrored module.
- [ ] Add integration, statistical, regression, property-based, and public API tests.

Progress: **3 of 5 setup and coverage phases complete (60%)**. The directory structure and migration are complete; broader test coverage remains.

## Verification baseline

- Existing suite: 28 tests pass.
- The audit found failures that are not covered by the current tests.
- `docs/mathematics/tmp.png` is unrelated and should remain untouched.
