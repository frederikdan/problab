# Adding a distribution

Internal developer checklist. Checked against the current code on 2026-10-05.
Gamma is the example below; it is not implemented by this note.

## 1. Define the public contract

- Choose parameter names, order, defaults, scalar types, and allowed ranges.
- Decide which parameters can be `RandomVariable` objects and which are fixed
  configuration. Random parameters describe a conditional distribution for
  each realization; they are not replaced with their means.
- For Gamma, a shape/scale API is one option: both parameters must be finite  
  and strictly positive. Scale and rate are different; rate is `1 / scale`.
  Decide separately whether to expose a location parameter.
- Follow the current concrete classes by inheriting from `Distribution`.
  `_ContinuousDistribution` additionally requires `pdf()`;
  `_DiscreteDistribution` requires `pmf()`. Choosing either means implementing
  that extra public method and its validation contract.

## 2. Create the files and exports

| Purpose | Gamma example, relative to the repository root |
| --- | --- |
| Implementation | `src/problab/distributions/continuous/gamma.py` (currently empty) |
| Parameter validators | `src/problab/validation/distributions/continuous/_gamma.py` |
| Implementation unit tests | `tests/unit_tests/distributions/continuous/gamma.py` (currently empty) |
| Validator unit tests | `tests/unit_tests/validation/distributions/continuous/_gamma.py` |

Export `GammaDistribution` through imports and `__all__` in all three packages:
`problab.distributions.continuous`, `problab.distributions`, and `problab`.
Internal modules should import from the defining module, not the root package.
Use a function-local import or `TYPE_CHECKING` where needed to avoid cycles.
Use `numpy.typing`, not NumPy's private `_typing` module, for new annotations.

## 3. Construct the distribution

- Define `symbol: ClassVar[str] = "Gamma"` on the class. The base class uses it
  for the distribution name; validators use `instance.symbol` in messages.
- Accept keyword-only `parameter_risk_policy`, annotated
  `Literal["warn", "raise", "ignore"]`, defaulting to
  `DEF_PARAMETER_RISK_POLICY` from `problab/_config.py`.
- Use this decorator pattern with distribution-specific validators:

```python
@_validate_parameters(
    validator_arguments=("parameter_risk_policy",),
    parameter_risk_policy=_validate_parameter_risk_policy,
    shape=_validate_gamma_shape,
    scale=_validate_gamma_scale,
)
```

- Call `super().__init__(parameters=(shape, scale))`. Preserve the original
  inputs: `parameters` must return the original scalars and the exact supplied
  random-variable objects. `_parameter_nodes` provides their graph form.
- Keep fixed configuration outside graph parameters, as Categorical does.
  The risk policy is a construction setting, not a graph parameter; it is
  forwarded before the constructor body and need not be stored on `self`.

## 4. Validate every graph parameter

Each validator receives `value`, keyword-only `instance`, and
`parameter_risk_policy`. Follow this order:

1. Check the accepted input type; explicitly reject booleans for numeric
   parameters. Use the random variable's `_node`, or create a `_ConstantNode`.
2. Check its mathematical `value_set` with `is_known_subset`. Gamma shape and
   scale must be known subsets of `POSITIVE_REALS`.
3. Call `_require_supported_parameter_realization_dtypes` with the node and
   the dtype families the backend actually supports. A `Real` annotation does
   not guarantee that SciPy accepts object arrays, `Fraction`, or `Decimal`.
4. Call `_validate_parameter_realization_value_set`, supplying the node,
   parameter name, `instance.symbol`, allowed set, and policy.

Mathematically invalid definitions and unsupported dtypes always raise.
Use `TypeError` for unsupported types and `ValueError` for invalid values.
For numerical risk, `raise` rejects construction, `warn` issues a
`RuntimeWarning`, and `ignore` allows construction silently. An unknown or
overly broad realization set means validity cannot be guaranteed; a warning
does not prove that a bad sample will occur.

Reuse the helpers in `validation/distributions/_base.py`; do not duplicate
dtype or exceptional-value checks in each distribution.

## 5. Declare both output value sets

- Store the ideal mathematical range in `_mathematical_value_set` and expose
  it through the public `value_set` property. Define boundary semantics
  explicitly: Gamma's usual support includes zero, although the ideal
  continuous draw equals zero with probability zero.
- Override `_realization_value_set` when machine outputs differ. Gamma can
  return zero through underflow, so its numerical range must include zero.
- Declare the actual output dtype families and any possible positive infinity,
  negative infinity, or `NaN` through their separate permission fields.
  Include exceptional values only when they can occur or cannot be ruled out.
- The inherited realization property simply returns the mathematical set;
  do not assume it accounts for rounding, underflow, overflow, or invalid
  realized parameters automatically.

The context always checks sample shape and dtype. `validate=True` additionally
checks samples against the realization set, not the mathematical set.

## 6. Implement sampling without breaking graph evaluation

Implement `_sample(*parameters, num_samples, rng)`; its parameter arrays arrive
in the same order passed to the base constructor. Return a numeric NumPy array
of shape `(num_samples,)` with a dtype permitted by the realization set.

For a shape/scale Gamma wrapper, the backend mapping is:

```python
shape, scale = parameters
return gamma.rvs(
    a=shape,
    loc=0,
    scale=scale,
    size=num_samples,
    random_state=rng,
)
```

Here `gamma` is imported from `scipy.stats`. See the
[SciPy Gamma documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html)
for parameter meanings, boundaries, and backend behavior.

Use the passed generator; do not create or seed another one. Do not call
`.sample()` on parameters or create another realization context. The existing
path is `sample()` -> context -> `_DistributionNode` -> `Distribution._evaluate()`
-> `_sample()`. It evaluates parameters together, preserves shared dependencies,
and supplies the generator and sample count. Scalar constants are broadcast by
the context; each sample must use its corresponding realized parameters.

## 7. Respect numerical policies and current gaps

`parameter_risk_policy` governs potential parameter problems at construction.
`numerical_error_policy` governs actual numerical failures during sampling.
One setting does not imply or override the other.

**Still pending in the shared library:** `Distribution._evaluate()` passes
realized parameters directly to `_sample()`. Runtime parameter-range checks,
sampling invalid entries as `NaN` under warn/ignore, and corresponding output
support/dtype declarations have not yet been implemented. Also, the context's
NumPy error policy currently wraps operation-node evaluation, not distribution
backend calls. `Distribution.sample()` does not yet expose that policy, although
`RandomVariable.sample()` does. Do not assume construction warnings, SciPy, or
`validate=True` provide this missing parameter handling.

When that shared handling is added, integrate new distributions with it.
Check the backend's own finite limits as well as mathematical domains; avoid
silently accepting invalid parameters or suppressing unrelated backend errors.

## 8. Decide which exact methods to implement

The optional hooks are `_mean_exact`, `_variance_exact`, `_cdf_exact(x)`, and
`_ppf_exact(q)`. Return `None` when a hook is unavailable. The base class then
uses Monte Carlo for `Mode.AUTO`, or raises `NotImplementedError` for
`Mode.EXACT`; it derives exact standard deviation from exact variance.

For fixed shape/scale Gamma, mean is `shape * scale` and variance is
`shape * scale**2`. Support scalar/array CDF and PPF inputs and their endpoints
if implementing those hooks. Random parameters require a separate analytical
derivation: an unconditional mixture generally cannot use the fixed-parameter
formula with random variables or their means substituted.

If adding `pdf()` or `pmf()`, validate its input type and `NaN` handling and
define values outside the support and at its boundaries. A density can be
infinite at a boundary without that being a sampling overflow.

## 9. Verify and document the addition

- **Unit:** constructor storage, symbol, support, every validator, accepted
  types, invalid ranges, supported dtypes, all risk policies, policy validation
  order, and exact-hook behavior. Patch `gamma.rvs` where the implementation
  uses it to check arguments, array order, size, generator, and returned result.
- **Integration:** distribution and random-variable sampling,
  random/shared parameters, events, conditional probability, sample shape,
  dtype, reproducibility, and `validate=True` at numerical boundaries.
- **Statistical:** compare public sampling results with independent analytical
  expectations. Gamma with shape one gives an exponential case; also check
  fixed-parameter moments and mixtures. Use statistically justified tolerances
  and fixed seeds, following `docs/testing.md`.
- **Property/regression:** cover general invariants and discovered bugs,
  especially parameter boundaries, rounding, underflow, and overflow. Keep
  failures for missing behavior visible.
- **API:** verify exports, constructor signatures, keyword-only policy,
  default settings, and public parameter identity.
- Update API descriptions/examples and `docs/test_inventory.md`. Refresh the
  library explorer snapshot if the change includes an updated source map.
- Confirm tests are discovered: mirrored unit filenames are not necessarily
  `test_*.py`; the all-suite loader in `tests/test_all.py` handles the unit tree.
  Record known unrelated failures separately rather than weakening tests.
