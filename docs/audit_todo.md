# ProbLab to-do

Updated after the source audit on 2026-09-21. Keep completed items checked;
remove an item only when explicitly requested. Identifiers stay stable so
findings can be discussed one at a time.

The audit covered library source, exports and packaging, existing tests,
API/testing documentation, and the source explorer's code and extractor tests.
Runtime findings below were reproduced with local probes or existing tests.
The Python-version issue was checked against the source and Python's official
documentation; only Python 3.13 was available locally. This review cannot prove
that every possible bug has been found. The explorer UI was reviewed statically,
not exhaustively exercised in a browser.

## Confirmed defects already tracked

- [x] **B01 — Update the remaining graph callers.** `RandomVariable.dependency_graph`
  and `plot_dependencies()` in `src/problab/random_variables/base.py` now pass
  `(self._node,)` to `NodeGraph`. Their delegation, public-path regression, and
  composed-workflow tests pass.
- [ ] **B02 — Support mixed numeric arrays at numerical boundaries.**
  `exp(X)` fails for categorical inputs `[1, 2.0]`. The same representation
  also breaks random parameters in `NormalDistribution` and `PoissonDistribution`.
  Normalize supported numeric inputs at ufunc/SciPy boundaries without changing
  stored categorical objects or silently losing precision. Sources:
  `functions/_utils.py`, `_operations.py`, and concrete distribution `_sample()` methods.
- [ ] **B03 — Account for arithmetic underflow in realization support.**
  Multiplying a deterministic `1e-200` variable by itself, or raising it to
  power two, returns zero but fails `validate=True`. Give machine-support
  inference its own boundary rules for multiplication, division, and powers.
  Source: `value_sets/_inference.py`, used by `random_variables/base.py`.
- [x] **B04 — Preserve the former positional sampling argument.**
  `sample(1, rng, True)` currently interprets `True` as `max_graph_size`.
  Restore the former `validate` position or explicitly decide to change that
  contract and update its regression test. Source: `RandomVariable.sample()`.
  Completed 2026-10-07: `validate` is again the third positional argument;
  `max_graph_size` and `numerical_error_policy` are keyword-only. All 45
  RandomVariable unit, public-contract, and focused sampling regression tests
  pass. A scan of Python callers found no calls with more than three positional
  sample arguments. The full suite was not rerun.
- [x] **B05 — Reject boolean numeric parameters consistently.**
  Normal parameters reject Python booleans with `ValueError` and NumPy booleans
  with `TypeError`. Interval-bound validation also accepts Python booleans but
  rejects NumPy booleans. Apply explicit boolean rejection consistently in
  `validation/distributions/continuous/_normal.py` and
  `validation/random_variables/_base.py`.
  Completed 2026-10-07: added explicit Python/NumPy Boolean rejection to
  Normal mean/std and interval-bound validation. All 52 focused existing tests
  and 28 direct Boolean-rejection checks pass. No other validation behavior
  was changed; the full suite was not rerun.

## Additional confirmed defects

The most urgent new findings are B06–B09 and B19: executing labels, silently
changing values/results, returning probabilities outside the intended domain,
and claiming an unsupported Python version.

- [x] **B06 — Never evaluate string labels as Python/SymPy expressions.**
  `_constant_value_set()` sends arbitrary equality operands through
  `_to_sympy_value()` and `sp.sympify()`. Constructing `X == label` can execute
  code contained in that label; a harmless probe appended a marker to a list
  during event construction. Keep nonnumeric labels outside symbolic parsing
  and validate inputs to public numeric value sets before sympifying them.
  Sources: `random_variables/nodes/_utils.py`, `value_sets/_utils.py`, and
  `value_sets/mixed_numeric_value_set.py`.
  Completed 2026-10-10: nonnumeric constants bypass symbolic conversion;
  the shared converter and mixed numeric constructor reject nonnumeric inputs.
  All 38 focused existing tests pass. Direct checks for six text/byte labels
  confirm that symbolic conversion is never called on those paths. Text-label
  array storage remains a separate issue (B10). The full suite was not rerun.
- [ ] **B07 — Preserve large integer category values exactly.**
  `CategoricalDistribution([2**63 - 1, 2**63], [1.0, 0.0]).sample()` returns
  the second value after both categories are converted to `float64`; even
  realization validation passes. Check that dtype selection is lossless and
  fall back to mixed numeric storage when necessary. Source:
  `distributions/discrete/helpers/_categorical.py::_infer_categorical_configuration`.
- [x] **B08 — Avoid unconditional float conversion in integer modulo.**
  For deterministic `X = 2**60 + 1`, `X % (2**61)` returns a float representing
  `2**60`, losing one. `_modulo()` uses `np.where(..., np.nan, result)` even
  when no divisor is zero. Preserve exact integer results in that case and
  define a representation for batches containing zero divisors. Source: `_operations.py`.
  Completed 2026-10-10 in `operations/_arithmetic.py` and
  `value_sets/_realization_inference.py`: modulo preserves native storage when
  no divisor is zero. Integer batches needing NaN use float64 within +/-2**53
  and object storage otherwise. Realization inference declares matching types
  and conservative finite support for zero-containing or mixed-sign domains.
  All 49 arithmetic-operation and realization-inference tests pass, along with
  six validated public precision/storage cases and warn/raise policy checks.
  The full suite was not rerun.
- [ ] **B09 — Require real-valued support for CDF and quantile methods.**
  `CategoricalDistribution([1j], [1.0]).cdf(1.0, num_samples=3)` returns `1.0`
  using NumPy's complex ordering. A tuple category's `ppf()` can even return
  a tuple, outside its declared result contract. Reject unsupported support
  before sorting/comparing samples, consistently with existing interval methods.
  Source: `distributions/base.py::cdf` and `ppf`.
- [x] **B10 — Align nonnumeric constant storage with ObjectValueSet.**
  Comparing a categorical variable to `"N"`, `"None"`, or `"hello world"`
  raises a dtype error: the constant has a Unicode dtype while its support
  requires object dtype. Choose the array representation and value set together;
  these labels should compare as ordinary objects. Source:
  `random_variables/nodes/_utils.py::_constant_array` and `_constant_value_set`.
  Completed 2026-10-10: nonnumeric constants now use scalar object arrays.
  All 15 node-utility, constant-node, and categorical integration tests pass.
  Direct checks cover equality/inequality with realization validation for eight
  text/byte and compound values, plus unchanged numeric/Boolean storage.
  The full suite was not rerun.
- [x] **B11 — Preserve ragged compound constants as atomic objects.**
  A categorical value `[1, [2, 3]]` samples successfully, but constructing
  comparison with that same value raises from the initial `np.asarray(value)`.
  Build a scalar object array for compound values without first attempting a
  rectangular array conversion. Source: `random_variables/nodes/_utils.py::_constant_array`.
  Completed by the same storage change as B10 on 2026-10-10: compound values
  are assigned directly into scalar object arrays. Direct checks confirm
  preserved object identity and validated comparisons for `[1, [2, 3]]`
  and `(1, [2, 3])`.
- [x] **B12 — Make compound-category equality and merging agree.**
  Array categories can be sampled and merged, but comparing a sampled
  `np.array([1, 2])` category to the same array raises an ambiguous-truth error.
  Also, `[1]` and `[1, 1]` NumPy arrays merge because equality broadcasts before
  `_objects_equal()` reduces with `all`. Define shape-aware atomic array equality
  and use that policy for merging, containment, and event comparisons.
  Sources: `value_sets/_comparison.py`, categorical `_merge_equal_categories`,
  `random_variables/base.py::_equality_comparison`, and `_operations.py::_EQ/_NEQ`.
  Completed 2026-10-10: array comparisons use `np.array_equal`, and object-valued
  equality/inequality operations compare each sample through the shared helper;
  native nonobject comparisons retain the NumPy path. All 23 focused existing
  tests pass. Direct public checks cover distinct/duplicate array categories,
  their merged probabilities, validated per-sample equality/inequality, and
  deterministic probabilities. Scalar/empty comparisons also pass. The full
  suite was not rerun.
- [x] **B13 — Normalize numeric support construction consistently.**
  A categorical variable with category `2.0` satisfies `is_in(Integers)` but
  is rejected as a Binomial count parameter. A categorical exponent `2.0`
  also prevents the real-power inference needed for `sqrt(X ** exponent)`.
  Use the shared numeric-to-SymPy conversion when constructing categorical
  support and `MixedNumericValueSet.sympy_set`, which still use raw `sp.sympify`.
  Completed 2026-10-10: both construction paths now use `_to_sympy_value`,
  preserving category inputs while normalizing integral floats in symbolic
  support. All 24 categorical-support regression, helper, mixed-value-set,
  and categorical integration tests pass, including all three B13 regressions.
  The full suite was not rerun.
- [ ] **B14 — Allow correctly rounded inverse-trigonometric endpoints.**
  `arcsin(X)` for deterministic `X = np.float32(1)` fails realization validation
  because its rounded result exceeds exact `pi/2`. The same occurs for
  `arccos(np.float32(-1))` and large positive float32 inputs to `arctan`.
  Derive conservative machine bounds for the supported dtypes while retaining
  exact mathematical bounds. Source: `functions/trigonometric.py`.
- [x] **B15 — Validate that MixedNumericValueSet actually contains numbers.**
  `MixedNumericValueSet(values=("2",))` is accepted and reported as a subset
  of the reals while containing a string. Reject nonnumeric members before
  building symbolic support. Source: `value_sets/mixed_numeric_value_set.py`.
  Completed alongside B06 on 2026-10-10: constructor validation now rejects
  nonnumeric members before symbolic conversion, while preserving the existing
  Boolean category support. Covered by the focused tests and direct checks above.
- [ ] **B16 — Make accepted Real inputs match numerical backend support.**
  `Fraction(1, 2)` passes the numeric type checks but fails in categorical
  probability validation (`np.isfinite`), Normal sampling, and `log()`.
  Explicitly support and normalize such inputs, or narrow the declared/runtime
  contract and reject unsupported types at entry. Sources:
  `validation/distributions/discrete/_categorical.py`, Normal sampling, and
  `validation/functions/_common.py`/`functions/_utils.py`.
  Design decision agreed on 2026-10-10: accept `Fraction` inputs consistently
  across implemented real-valued APIs and convert them to supported numerical
  approximations for numerical calculations. Exact rational arithmetic is not
  guaranteed. Implementation must cover both direct inputs and Fraction-valued
  random inputs, keep mathematical and realization support accurate, and retain
  fast paths for native NumPy inputs. This item remains open pending implementation
  and verification.
  Partial progress on 2026-10-10: `_float_if_fraction` now converts Fraction
  constants when building their numerical arrays. Constant nodes retain exact
  mathematical support and describe the converted scalar in realization support.
  All 10 constant-node/helper tests passed, alongside five direct Fraction checks
  (including underflow) and validated Normal sampling with Fraction parameters.
  Scalar function evaluation in `_apply` now uses the same helper before calling
  the numerical operation. All 31 function unit tests and five public Fraction
  checks (`log`, `sqrt`, `sin`, `hypot`, and `logaddexp`) passed.
  Further scalar integration on 2026-10-10: the helper now covers categorical
  probability validation/storage, scalar CDF/PPF validation/evaluation, strict
  q/alpha validation, probability and quantile interval calculations, and
  `_is_close` (including ProbabilityResult consistency validation). Original
  negative categorical probabilities and out-of-range PPF inputs are still
  rejected even when conversion rounds them to valid endpoints. Strict q/alpha
  inputs that round to zero or one are rejected by their existing range checks.
  All 139 focused tests passed, including eight new public regression tests in
  `tests/regression_tests/test_fraction_scalar_inputs.py`. No full-suite rerun
  was performed for this step. Fraction arrays and Fraction-valued random inputs,
  their dtype checks, and matching realization inference remain open for review.
  Shared array helper added on 2026-10-10: `_convert_fractions_in_array` preserves
  native-array fast paths and source arrays, converts Fraction elements, and
  retains object storage when native conversion would change other values.
  `_conversion_preserves_values` moved unchanged to `problab._utils`; existing
  categorical code imports it there. All 44 focused utility/categorical tests
  passed, including five new array-helper tests. Integration of the array helper
  at numerical entry points remains pending individual review.
- [ ] **B17 — Handle deep graphs within the allowed size.**
  A chain of 600 additions to a deterministic variable raises `RecursionError`
  with `max_graph_size=1500`, although its 1201 nodes fit the limit. Graph
  construction/evaluation still recurse through Python calls. Use iterative
  traversal/evaluation, or explicitly validate and document a separate depth
  limit rather than failing with an incidental recursion error. Sources:
  `random_variables/graph.py`, `_context.py`, and `nodes/operation.py`.
- [ ] **B18 — Stabilize quantile minimum-sample calculations near zero.**
  `quantile_confidence_interval(q=1e-20, num_samples=1)` raises `OverflowError`
  instead of the intended insufficient-samples error. `1-q` rounds to one;
  similarly, the smallest positive float `alpha` underflows in `alpha/2`.
  Use stable logarithms (`log1p` and `log(alpha)-log(2)`) and handle infeasible
  sample requirements explicitly. Source: `statistics/_quantiles.py`.
- [ ] **B19 — Match Python compatibility metadata to source syntax.**
  `pyproject.toml` promises Python 3.11, but `Distribution.name` reuses double
  quotes inside a double-quoted f-string, syntax introduced in Python 3.12.
  Rewrite that expression for 3.11 or raise the declared minimum and test it.
  Source: `distributions/base.py:117`. See
  [Python's quote-reuse compatibility note](https://docs.python.org/3.12/whatsnew/3.12.html#pep-701-syntactic-formalization-of-f-strings).
- [ ] **B20 — Update documentation that describes an older API/status.**
  `docs/api_design.md` omits the exported numeric/object ValueSet subclasses
  and does not explain merged categorical properties. The opening of
  `docs/test_gap_inventory.md` still presents the fixed conditional-probability
  and graph-limit issues as current defects. Preserve historical review text
  with dated status notes and align current API descriptions with the code.

## Completed — keep these entries

- [x] **C01 — Preserve condition dependencies when a joint event simplifies to a constant.**
  Contexts now request both the joint event and the condition. Regression tests
  cover complements, identical event/condition nodes, exact condition counts,
  and no observed conditioned samples. Context tests cover cache retention and
  shared dependencies in either evaluation order.
- [x] **C02 — Validate public NodeGraph limits as positive integers.**
  The constructor uses `_validate_max_size`; boolean/fractional rejection and
  NumPy integer acceptance tests pass.

## Deferred feature

- [ ] **F01 — Implement exact distribution statistics.** Implement exact mean,
  variance, standard deviation, CDF, and PPF for fixed-parameter distributions.
  Define a separate contract for random parameters before extending exact
  methods to those distributions. Existing exact-method tests remain active.

- [x] **F02 — Apply the sampling policy to invalid distribution parameters.**
  Completed 2026-10-06 for Normal, Binomial, and Poisson. Per-parameter NumPy
  masks exclude invalid realized values before SciPy sampling. `raise` reports
  invalid parameters; `warn`/`ignore` sample valid positions and insert NaN at
  invalid positions. All-invalid batches skip the backend. Realization support
  declares these outputs, including finite backend limits and integer/NaN
  storage. Mixed batches use float64 for integers within +/-2**53 and exact
  object storage otherwise; all-valid batches retain their backend dtype.
  Active tests: `tests/regression_tests/test_parameter_realization_policy.py`.
  Design agreed 2026-10-06: add a runtime mask function per parameter alongside
  construction validators in the corresponding distribution validation module.
  Prefer vectorized NumPy checks, with alternatives when needed; shared
  evaluation combines masks and applies the policy before backend sampling.
  See [the runtime-validation design](adding_distributions.md#agreed-runtime-validation-design-2026-10-06).
  Categorical has no graph parameters and uses the default empty masks.

## Additional defects reproduced by the test update

- [x] **B21 — Enumerate numeric dtype candidates correctly.** Shared inference
  includes timedelta storage in `np.integer`/`np.number`, then fails integer
  bounds or power-loop resolution. It can also return unknown dtypes for
  `exp(NormalVariable)` and omits `np.longdouble` on this Windows environment.
  These failures now have unit, integration, and statistical tests. Sources:
  `value_sets/_inference.py` and callers in `_realization_inference.py`.
  Completed 2026-10-07: candidate matching now excludes nonnumeric storage
  other than explicitly supported Boolean/object types, and deduplication uses
  scalar types so Windows `longdouble` is retained. All 80 tests in shared,
  mathematical, and realization inference plus analytical workflows pass,
  including the previously failing B21 cases. The full suite was not rerun.
- [x] **B22 — Validate non-finite permissions when dtype information is unknown.**
  `HomogeneousNumericValueSet(..., dtype_types=None, allows_nan="yes")` is
  accepted because configuration validation returns before checking its flags.
  Validate all three Boolean settings regardless of dtype information.
  Completed 2026-10-07: the Boolean checks now run before the early return for
  unknown dtype information. All 10 configuration-validation and homogeneous
  numeric value-set tests pass, including the previously failing constructor
  test. The full suite was not rerun.

## Design recommendations — decisions needed before implementation

- [ ] **D01 — Adopt and enforce an annotation contract.** Start from
  `docs/review_findings.md`. In particular, declare the symbolic-set interface
  on `NumericValueSet`, use public `numpy.typing`, type decorators so they preserve
  signatures, and make equality operands/return annotations reflect the actual
  API. This is a proposal, not authorization for a naming or API refactor.
- [ ] **D02 — Decide whether simplification preserves dtype promotion.**
  Removing `X * 1.0` keeps X's integer dtype. With X equal to the largest int64,
  `(X * 1.0) + 1` then raises integer overflow, whereas the unsimplified NumPy
  expression produces a finite float. Decide which semantics ProbLab promises;
  retain required casts or document that symbolic identities can change machine
  arithmetic. Source: `random_variables/nodes/_simplification.py`.
- [ ] **D03 — Complete the agreed machine-domain and overflow policy.**
  Operations now honor `numerical_error_policy`; underflow is allowed, and
  declared machine support covers permitted infinity, NaN, and rounding.
  Tests for `log2(exp(X))` at X=-1000 pass under all three policies. Runtime
  distribution-parameter handling (F02) is complete as of 2026-10-06; broad
  dtype inference (B21) is complete as of 2026-10-07. Sources: `operations/_arithmetic.py`, function wrappers,
  distribution evaluation, and `value_sets/_realization_inference.py`.
- [ ] **D04 — Expose sampling configuration consistently.** Consider the same
  keyword controls across `P`, distribution sampling/statistics, and
  `RandomVariable` methods. Currently a graph that samples with an increased
  limit cannot use that limit through `P` or the interval/statistics methods.
- [ ] **D05 — Define atomic-object behavior for apply and contains.**
  `X.apply(lambda value: (value, 2))` creates a two-dimensional array and fails
  the sample-shape check; `ObjectValueSet.contains([1, 2])` treats the argument
  as a batch, even when `[1, 2]` is one stored category. Define/document how to
  distinguish an atomic object from a batch, and reuse an explicit packing
  helper where appropriate. These are contract decisions beyond categorical sampling.
- [ ] **D06 — Test the declared compatibility range.** Add checks for the minimum
  Python/dependency versions and a current version. Set dependency lower bounds
  from the APIs actually used (for example `ufunc.resolve_dtypes`) so an existing
  older environment is not silently accepted. Keep known failing behavior tests
  visible rather than weakening them to obtain a green build.

## Verification still to add

- [ ] **T01 — Turn the remaining reproductions into focused regression tests.**
  The suite now includes persistent checks for the recent numeric support,
  categorical conversion, graph accounting, rounding, and parameter-policy
  work. Tests cover B07, B13, B14, B21, B22, and F02; broader scenarios for
  B05 and the other audit findings still need focused regressions. In
  particular, B06, B08, B10–B12, and B15–B18 do not yet have complete
  persistent regression coverage. Keep the systematic test inventory current.

Latest full library run (2026-10-10): **659 test methods; 654 pass and 5
error**, producing **20 error reports** when subtests are counted. All five
erroring methods concern the unimplemented F01 exact-distribution hooks. The
explorer's **7 extractor tests pass** with normal temporary-directory access.
No tests are skipped or marked as expected failures. B16 remains open pending
array and Fraction-valued random-input integration. See
`docs/test_gap_inventory.md` for every failing method and its issue.

Previous full run (2026-10-06): 646 methods; 630 passed and 16 failed or errored,
producing 10 failure reports and 26 error reports across subtests.

Previous full run (2026-10-05): 600 methods; 578 passed and 22 failed or errored.
The F02 update adds 12 tests and resolves the three parameter-policy methods.
Normal's floating-only realization dtype also makes its previously failing
power and two exponential/logarithm workflows pass; at that stage B21 itself
remained open.
The follow-up coverage adds 34 more tests for class parameter metadata,
guarantee checks, mask delegation and agreement, shared graph evaluation,
policy independence, integer/NaN boundaries, and analytical masked sampling.
The 16 failing methods in the 2026-10-06 run match the pre-expansion list exactly; none
are skipped or marked as expected failures. The distribution-development guide
now includes complete worked code, the `_valid_parameter_sets` contract, and
specific tests and commands. Its Gamma examples were verified in memory only;
Gamma remains an unimplemented placeholder in the library.

Recent verification confirms the covered native mixed numeric scenarios (B02),
arithmetic underflow (B03), large integer category preservation (B07), CDF/PPF
support rejection (B09), and rounded inverse-trigonometric endpoints (B14)
now pass. Existing to-do entries are retained; no library behavior was changed
by this test update.
