# Source and test review

This review adds reproducible regression cases, without changing library behavior.
The exact-distribution methods remain deliberately deferred. This is not a claim
that passing tests establish correctness for all possible inputs.

## Confirmed defects

Each case below is reproduced in `tests/regression_tests/test_audit_edge_cases.py`.

| Defect | Source | Suggested correction |
| --- | --- | --- |
| `P(E, given=~E)` raises `KeyError` when the condition has positive probability. Simplification turns the joint event into a constant, so the context graph omits dependencies needed to evaluate the condition. | `probability/probability.py`, `random_variables/_context.py` | Construct a context containing both the condition and joint-event dependencies, or retain an unsimplified joint node for conditional evaluation. Preserve shared realizations. |
| `exp(X)` fails for numeric categories `[1, 2.0]` because the object array reaches a NumPy ufunc that cannot evaluate it. | `functions/_utils.py`, `_operations.py`, `distributions/discrete/helpers/_categorical.py` | Convert supported numeric realizations at the function boundary, or evaluate supported scalar objects individually. Preserve original categorical samples; do not indiscriminately coerce arbitrary objects. |
| Multiplying a deterministic `1e-200` variable by itself returns zero, but `validate=True` rejects it as outside strictly positive realization support. | `random_variables/base.py`, `value_sets/_inference.py` | Use machine-aware arithmetic support inference, including underflow to zero, separately from mathematical inference. Review division and powers for the same boundary issue. |
| Adding `max_graph_size` before `validate` breaks the existing positional call `X.sample(1, rng, True)`. | `random_variables/base.py` | Preserve the previous parameter positions and add the graph limit afterward, preferably as a keyword-only parameter. |
| Normal parameters reject Python `True` with `ValueError` but NumPy `bool_` with `TypeError`; Binomial and Poisson explicitly reject both with `TypeError`. | `validation/distributions/continuous/_normal.py` | Reject both boolean types before the `Real` check, consistently with the other distribution validators. |
| Public `NodeGraph` accepts `max_size=True` and `max_size=1.5`, although public sampling rejects these limits. | `random_variables/graph.py` | Apply the shared positive-integer validator at graph construction. |

## Test changes

- Added eight regression methods: six expose the defects above; two confirm
  the corrected graph-limit error message and scalar/array rejection for label CDFs.
- Added two quantile property methods: compare binary search to exhaustive feasible
  indices across 75 parameter combinations, assert the logarithmic call bound,
  and check equality at the probability threshold explicitly.
- Corrected the quantile unit-test fake to preserve feasible outer endpoints.
  Removed assertions prescribing the exact midpoint visitation order; retained
  assertions on the selected endpoints and maximum call count.
- Kept the deferred exact-method tests active, as requested.

## Proposed annotation contract

This is a proposal; no library annotation refactor is included in this review.

1. Annotate every public function, method and property, and module-level private
   helpers. Omit annotations for `self` and `cls`; use `-> None` for methods that
   do not return a value. Nested helpers may rely on clear local inference.
2. Describe actual accepted inputs and actual returned values. For example,
   categorical equality accepts arbitrary objects, so its operand should be
   `object`, rather than `RandomVariable | Real`. Do not use annotations as a
   substitute for runtime range or boolean validation.
3. Use shared scalar aliases where repeated: integer inputs should include
   `int | np.integer[Any]`; standard real scalar inputs should include Python
   integers/floats and NumPy integer/floating scalars. Explicitly decide whether
   additional `numbers.Real` implementations are supported before narrowing
   existing contracts. Keep runtime ABC checks separate from typing aliases.
4. Use `NDArray` from public `numpy.typing`, never `numpy._typing`. Use
   `NDArray[Any]` when categorical/object/real/complex samples are all possible,
   and precise dtype parameters where guaranteed. State shape requirements in
   docstrings or validation; `NDArray` alone does not encode sample count.
5. Distinguish scalar and array returns with overloads where useful. Public
   mathematical wrappers return `float` for scalar input and `RandomVariable`
   for random-variable input. Sample methods always return arrays.
6. Use `object` for an input that must be checked or narrowed; use `Any` only
   when unrestricted behavior is intentional. Type arbitrary `apply` callbacks
   with a suitable callable signature. Annotate the validation decorator with
   `ParamSpec` and a return type variable so decorating preserves signatures.
7. Use `ValueSet` for any supported set, `NumericValueSet` where symbolic numeric
   support is required, and `TypeGuard` for helpers that narrow node types.
   Explicitly narrow before accessing numeric-only fields.
8. Include `None` and `types.NotImplementedType` wherever a function can return
   them. In particular, binary operator methods that reject unsupported operands
   should not claim that every return is an event or random variable.
9. Use postponed annotations and `TYPE_CHECKING` imports where needed to avoid
   annotation-only circular imports. Prefer `collections.abc` for collection and
   callable interfaces, with built-in `list`, `tuple`, `dict`, and `set` generics.

Concrete inconsistencies currently include:

- `RandomVariable.realize()` and several arithmetic methods have no return annotations.
- `RandomVariable.__eq__()` and `_equality_comparison()` exclude label and compound
  objects that are intentionally accepted by the implementation.
- Event binary methods annotate only `_Event` despite returning `NotImplemented`.
- Basic/exponential wrappers annotate scalar returns as `Real`, trigonometric
  wrappers use `float`, and the common helper normalizes scalar results to `float`.
- `_mean_exact()` uses `numbers.Complex`, while `mean()` promises built-in `complex`.
- Normal, Binomial and Categorical import `NDArray` through `numpy._typing`, while
  categorical helpers use `numpy.typing`.
- `_validate_parameters` is untyped, so static checking cannot reliably preserve
  the decorated public signatures.
- `_to_sympy_value(value: Any) -> sp.Basic` is broader in input than its return
  guarantee: `sympify` may return Python containers for container inputs.

Adopt the contract first, then check it with a static checker and small typing
examples. Runtime tests alone cannot establish annotation consistency.
