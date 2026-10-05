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
- [ ] **B04 — Preserve the former positional sampling argument.**
  `sample(1, rng, True)` currently interprets `True` as `max_graph_size`.
  Restore the former `validate` position or explicitly decide to change that
  contract and update its regression test. Source: `RandomVariable.sample()`.
- [ ] **B05 — Reject boolean numeric parameters consistently.**
  Normal parameters reject Python booleans with `ValueError` and NumPy booleans
  with `TypeError`. Interval-bound validation also accepts Python booleans but
  rejects NumPy booleans. Apply explicit boolean rejection consistently in
  `validation/distributions/continuous/_normal.py` and
  `validation/random_variables/_base.py`.

## Additional confirmed defects

The most urgent new findings are B06–B09 and B19: executing labels, silently
changing values/results, returning probabilities outside the intended domain,
and claiming an unsupported Python version.

- [ ] **B06 — Never evaluate string labels as Python/SymPy expressions.**
  `_constant_value_set()` sends arbitrary equality operands through
  `_to_sympy_value()` and `sp.sympify()`. Constructing `X == label` can execute
  code contained in that label; a harmless probe appended a marker to a list
  during event construction. Keep nonnumeric labels outside symbolic parsing
  and validate inputs to public numeric value sets before sympifying them.
  Sources: `random_variables/nodes/_utils.py`, `value_sets/_utils.py`, and
  `value_sets/mixed_numeric_value_set.py`.
- [ ] **B07 — Preserve large integer category values exactly.**
  `CategoricalDistribution([2**63 - 1, 2**63], [1.0, 0.0]).sample()` returns
  the second value after both categories are converted to `float64`; even
  realization validation passes. Check that dtype selection is lossless and
  fall back to mixed numeric storage when necessary. Source:
  `distributions/discrete/helpers/_categorical.py::_infer_categorical_configuration`.
- [ ] **B08 — Avoid unconditional float conversion in integer modulo.**
  For deterministic `X = 2**60 + 1`, `X % (2**61)` returns a float representing
  `2**60`, losing one. `_modulo()` uses `np.where(..., np.nan, result)` even
  when no divisor is zero. Preserve exact integer results in that case and
  define a representation for batches containing zero divisors. Source: `_operations.py`.
- [ ] **B09 — Require real-valued support for CDF and quantile methods.**
  `CategoricalDistribution([1j], [1.0]).cdf(1.0, num_samples=3)` returns `1.0`
  using NumPy's complex ordering. A tuple category's `ppf()` can even return
  a tuple, outside its declared result contract. Reject unsupported support
  before sorting/comparing samples, consistently with existing interval methods.
  Source: `distributions/base.py::cdf` and `ppf`.
- [ ] **B10 — Align nonnumeric constant storage with ObjectValueSet.**
  Comparing a categorical variable to `"N"`, `"None"`, or `"hello world"`
  raises a dtype error: the constant has a Unicode dtype while its support
  requires object dtype. Choose the array representation and value set together;
  these labels should compare as ordinary objects. Source:
  `random_variables/nodes/_utils.py::_constant_array` and `_constant_value_set`.
- [ ] **B11 — Preserve ragged compound constants as atomic objects.**
  A categorical value `[1, [2, 3]]` samples successfully, but constructing
  comparison with that same value raises from the initial `np.asarray(value)`.
  Build a scalar object array for compound values without first attempting a
  rectangular array conversion. Source: `random_variables/nodes/_utils.py::_constant_array`.
- [ ] **B12 — Make compound-category equality and merging agree.**
  Array categories can be sampled and merged, but comparing a sampled
  `np.array([1, 2])` category to the same array raises an ambiguous-truth error.
  Also, `[1]` and `[1, 1]` NumPy arrays merge because equality broadcasts before
  `_objects_equal()` reduces with `all`. Define shape-aware atomic array equality
  and use that policy for merging, containment, and event comparisons.
  Sources: `value_sets/_comparison.py`, categorical `_merge_equal_categories`,
  `random_variables/base.py::_equality_comparison`, and `_operations.py::_EQ/_NEQ`.
- [ ] **B13 — Normalize numeric support construction consistently.**
  A categorical variable with category `2.0` satisfies `is_in(Integers)` but
  is rejected as a Binomial count parameter. A categorical exponent `2.0`
  also prevents the real-power inference needed for `sqrt(X ** exponent)`.
  Use the shared numeric-to-SymPy conversion when constructing categorical
  support and `MixedNumericValueSet.sympy_set`, which still use raw `sp.sympify`.
- [ ] **B14 — Allow correctly rounded inverse-trigonometric endpoints.**
  `arcsin(X)` for deterministic `X = np.float32(1)` fails realization validation
  because its rounded result exceeds exact `pi/2`. The same occurs for
  `arccos(np.float32(-1))` and large positive float32 inputs to `arctan`.
  Derive conservative machine bounds for the supported dtypes while retaining
  exact mathematical bounds. Source: `functions/trigonometric.py`.
- [ ] **B15 — Validate that MixedNumericValueSet actually contains numbers.**
  `MixedNumericValueSet(values=("2",))` is accepted and reported as a subset
  of the reals while containing a string. Reject nonnumeric members before
  building symbolic support. Source: `value_sets/mixed_numeric_value_set.py`.
- [ ] **B16 — Make accepted Real inputs match numerical backend support.**
  `Fraction(1, 2)` passes the numeric type checks but fails in categorical
  probability validation (`np.isfinite`), Normal sampling, and `log()`.
  Explicitly support and normalize such inputs, or narrow the declared/runtime
  contract and reject unsupported types at entry. Sources:
  `validation/distributions/discrete/_categorical.py`, Normal sampling, and
  `validation/functions/_common.py`/`functions/_utils.py`.
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

- [ ] **F02 — Apply the sampling policy to invalid distribution parameters.**
  Construction risk checks are implemented, but `_evaluate()` still passes
  invalid realized parameter arrays directly to SciPy. Under `raise`, reject
  them with an informative ValueError; under `warn`/`ignore`, return NaN only
  at invalid positions and sample valid positions. Update sampled dtypes and
  realization support so `validate=True` accepts these declared outputs.
  Active tests: `tests/regression_tests/test_parameter_realization_policy.py`.

## Additional defects reproduced by the test update

- [ ] **B21 — Enumerate numeric dtype candidates correctly.** Shared inference
  includes timedelta storage in `np.integer`/`np.number`, then fails integer
  bounds or power-loop resolution. It can also return unknown dtypes for
  `exp(NormalVariable)` and omits `np.longdouble` on this Windows environment.
  These failures now have unit, integration, and statistical tests. Sources:
  `value_sets/_inference.py` and callers in `_realization_inference.py`.
- [ ] **B22 — Validate non-finite permissions when dtype information is unknown.**
  `HomogeneousNumericValueSet(..., dtype_types=None, allows_nan="yes")` is
  accepted because configuration validation returns before checking its flags.
  Validate all three Boolean settings regardless of dtype information. The
  focused constructor test remains failing.

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
  distribution-parameter handling (F02) and broad dtype inference (B21) remain
  unfinished. Sources: `operations/_arithmetic.py`, function wrappers,
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

Latest full library run (2026-10-05): **600 test methods; 578 pass and 22
fail or error**, producing **14 failure reports and 37 error reports** when
subtests are counted. The explorer's **7 extractor tests pass**. See
`docs/test_gap_inventory.md` for every failing method and its issue.

Recent verification confirms the covered native mixed numeric scenarios (B02),
arithmetic underflow (B03), large integer category preservation (B07), CDF/PPF
support rejection (B09), and rounded inverse-trigonometric endpoints (B14)
now pass. Existing to-do entries are retained; no library behavior was changed
by this test update.
