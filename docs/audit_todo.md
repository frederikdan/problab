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

- [x] Add regression tests for the public API, distributions, random variables, graph dependencies, sampling, events, validation, categorical values, and all identified findings above.
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
- [x] Document the test-writing method and test-category boundaries in `docs/testing.md`.
- [x] Add isolated unit tests for the implemented modules in `src/problab/distributions`, including the base class, concrete distributions, configuration, and categorical helpers.
- [x] Add isolated unit tests for all modules in `src/problab/functions`, including public exports, helpers, and mathematical wrappers.
- [x] Add isolated unit tests for all modules in `src/problab/probability`, including `P`, results, intervals, and configuration.
- [x] Add isolated unit tests for all modules in `src/problab/random_variables`, including graph construction, realization, nodes, and `RandomVariable` operations.
- [x] Add isolated unit tests for all implemented modules in `src/problab/statistics`, including Clopper-Pearson and quantile confidence intervals.
- [x] Add isolated unit tests for all implemented modules in `src/problab/validation`, including every validator and the parameter-validation decorator.
- [x] Add isolated unit tests for all implemented modules in `src/problab/value_sets` and remaining top-level event and operation modules.
- [x] Move categorical graph and probability behavior checks into integration tests and package import checks into API tests.
- [x] Finish detailed unit-test gaps in the distribution base class, random-variable operations and realization, dependency graphs, events, and operation helpers.
- [x] Add integration tests for composed public workflows: random distribution parameters, shared event realizations, probability results, and derived random variables.
- [x] Add statistical tests for samplers and complete probabilistic workflows against analytical answers: events, conditions, arithmetic, functions, shared and independent variables, random parameters, distribution statistics, CDF/PPF, and interval coverage.
- [x] Add regression tests for confidence intervals, real powers, categorical values, duplicate categories, and quantile-confidence sample limits.
- [x] Add failing regression tests for the five unresolved necessary issues: integer-valued float membership, integer overflow, NaN CDF inputs, floating-point support boundaries, and inconsistent probability-result counts.
- [x] Add property tests for categorical configuration, arithmetic identities, atomic compound categories, and probability-result rules.
- [x] Expand public API tests beyond import checks, including deliberate red tests for sampling controls, graph limits, and exact methods.
- [x] Map every identified unresolved behavior to a failing test in `docs/test_gap_inventory.md`.

Progress: **100% of the current test-organization checklist**. Unit tests cover every implemented source module; integration, property, regression, API, and analytical statistical tests cover the identified behavior. New bugs and new features still need new tests as they are discovered or specified.

## Verification baseline

- Full discovery: 345 test methods; 14 assertion failures and 31 errors from deliberately active gap tests.
- The 20 exact-method subcases each report an error, so failure and error counts are larger than the number of distinct unfinished behaviors. `docs/test_gap_inventory.md` maps the failures to source behavior.
- `docs/mathematics/tmp.png` is unrelated and should remain untouched.
