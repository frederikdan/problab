# Adding a distribution

Internal developer checklist. Updated for runtime parameter validation on 2026-10-06.
Gamma is the example below; it is not implemented by this note.

Read this together with [testing.md](testing.md). This guide covers the current
scalar-outcome distribution interface: one outcome per sample position.
Vector-valued distributions need a separate shape contract; they cannot simply
return an array of shape `(num_samples, dimension)` through the current context.

The Gamma code below is a worked sampling example, not a new public API or a
claim that Gamma's complete numerical behavior has been audited. Its shape/scale
contract and deliberately restricted input dtypes are example choices. Verify
the chosen backend across the project's supported platforms before releasing
a real implementation. Exact hooks are optional and omitted from the example.

## Responsibility map

| Component | What it owns | What it does not do |
| --- | --- | --- |
| Concrete distribution class | Public constructor, parameter metadata, mathematical and realization supports, backend delegation | Resample parameter variables or manage the graph cache |
| Its validation module | Construction validators, runtime mask functions, backend parameter limits | Apply sampling policy, sample, or replace invalid values |
| `Distribution` | Preserve parameters, construct parameter nodes, combine runtime masks, apply policy, restore sample positions | Infer a new distribution's output support or invent its validity rules |
| `_RealizationContext` | Shared realizations, RNG, sample count, caching, shape/dtype checks, optional support membership | Interpret distribution-specific parameters |
| Exact hooks | Analytical results for the explicitly supported cases | Silently substitute parameter means for random parameters |

Public distribution classes, `Distribution`, `RandomVariable`, `Mode`, and value
sets have public names. Validation functions, nodes, hooks, and metadata such as
`_valid_parameter_sets` remain private. New function examples here omit docstrings
in accordance with the current project workflow; prose explains their contracts.

## 1. Define the public contract

- Choose parameter names, order, defaults, scalar types, and allowed ranges.
- Decide which parameters can be `RandomVariable` objects and which are fixed
  configuration. Random parameters describe a conditional distribution for
  each realization; they are not replaced with their means.
- Explicitly decide which Python and NumPy scalar types are supported. Arrays
  arriving at `_sample()` are batches of realized parameters, not permission
  for users to pass arbitrary batch arrays to the constructor. Decide how to
  represent vector or compound configuration separately.
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

Add integration, statistical, regression, or property files in their matching
test folders as needed; see section 9 for required cases. New package directories
need their usual `__init__.py` files for imports and unittest discovery. Existing
`continuous` and `discrete` directories are already packages. A placeholder file
is not an exported or implemented distribution.

Export `GammaDistribution` through imports and `__all__` in all three packages:
`problab.distributions.continuous`, `problab.distributions`, and `problab`.
Internal modules should import from the defining module, not the root package.
Use a function-local import or `TYPE_CHECKING` where needed to avoid cycles.
Use `numpy.typing`, not NumPy's private `_typing` module, for new annotations.
Extend the explicit expected export lists in `tests/api_tests/imports.py` when
making the new class public, and test identity across its public import paths.
Do not export its validators or internal hooks.

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

### Define `_valid_parameter_sets` on the class

For the Gamma example, place this beside `symbol`:

```python
_valid_parameter_sets: ClassVar[dict[str, NumericValueSet]] = {
    "shape": POSITIVE_REALS,
    "scale": POSITIVE_REALS,
}
```

Import `NumericValueSet` from `problab.value_sets.base`, not from the root
`problab` package. Use lowercase for this private class metadata, consistently
with the public class attribute `symbol`. `ClassVar` describes class ownership;
it does not make the dictionary immutable. Treat it as fixed metadata. If a
subclass needs a different mapping, assign a new dictionary rather than mutating
the parent's dictionary, and update its validators and support rules together.

The mapping must satisfy all of these rules:

1. Each graph parameter has exactly one entry. Keys identify parameters in
   construction validators and runtime error messages.
2. Values are `NumericValueSet` objects describing mathematical valid domains.
   Use the separate permission flags for exceptional values; membership in
   `REALS`, for example, does not authorize infinity or NaN.
3. Insertion order matches `super().__init__(parameters=(...))`. For Gamma,
   `shape` must come before `scale`. The base helper pairs nodes with
   `.values()` by position; it does not match them by name. `zip(strict=True)`
   detects length mismatches when reached, but does not detect swapped entries.
4. Fixed configuration, `symbol`, and policy settings are not graph parameters
   and do not belong in this mapping. Categorical has `parameters=()` and uses
   the inherited empty mapping.
5. Define it on the class, not inside `__init__`. The decorator invokes
   construction validators before the constructor body runs, passing the
   not-yet-initialized instance. Validators may read class metadata at that
   point, but cannot rely on `_parameter_nodes` or other constructor fields.

Construction validation reads, for example,
`instance._valid_parameter_sets["shape"]`. The base helper
`_parameter_realizations_guaranteed_valid()` reads the same mapping when
comparing each node's declared realization support against its valid domain.

The mapping does **not** register validators, generate NumPy checks, enforce
backend limits, or determine output support. A new distribution must still wire
its constructor decorator and `_validate_parameter_realizations()` explicitly.
If a backend has a narrower finite domain, define that limit in its validation
module and account for it in both construction risk and output NaN inference.
Independent parameter sets also cannot prove cross-parameter conditions such
as `lower < upper`; see the runtime section for that case.

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

The decorator validates forwarded settings first, binds positional arguments,
applies constructor defaults, and supplies `instance` when the validator asks
for it. A validator returns `None` on success; its return value does not replace
the original parameter. Do not implement parameter normalization by returning
a different value from one of these validators.

### Worked validation module

The following is the complete validation-module example for
`src/problab/validation/distributions/continuous/_gamma.py`. Each parameter has
a construction function and a runtime function. Shape and scale share their
construction mechanics through a private helper in this same file.

The explicit dtype list is intentional: this example promises the usual
fixed-width integers and float16/32/64, not every NumPy floating implementation.
If a node declares a broader family that cannot be proved supported, the shared
dtype validator rejects it. Broaden that contract only after backend tests;
do not silently narrow a parameter array or assume `Real` implies backend
compatibility. A project implementation can make a different tested choice.

```python
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from problab.random_variables.nodes import _ConstantNode
from problab.validation.distributions._base import (
    _require_supported_parameter_realization_dtypes,
    _validate_parameter_realization_value_set,
)
from problab.value_sets._utils import is_known_subset


_GAMMA_PARAMETER_DTYPES = (
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
)


def _validate_gamma_positive_parameter(
    value,
    *,
    instance,
    parameter_name: str,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    from problab.random_variables.base import RandomVariable

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"'{parameter_name}' must not be a boolean.")

    if isinstance(value, RandomVariable):
        node = value._node
    elif isinstance(value, Real):
        node = _ConstantNode(value)
    else:
        raise TypeError(
            f"'{parameter_name}' must be a RandomVariable or a real number."
        )

    valid_set = instance._valid_parameter_sets[parameter_name]
    if not is_known_subset(node.value_set, valid_set):
        raise ValueError(f"'{parameter_name}' must be strictly positive.")

    _require_supported_parameter_realization_dtypes(
        (node,),
        distribution_name=instance.symbol,
        supported_input_types=(_GAMMA_PARAMETER_DTYPES,),
    )
    _validate_parameter_realization_value_set(
        node,
        parameter_name=parameter_name,
        distribution_name=instance.symbol,
        valid_value_set=valid_set,
        parameter_risk_policy=parameter_risk_policy,
    )


def _validate_gamma_shape(
    value,
    *,
    instance,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    _validate_gamma_positive_parameter(
        value,
        instance=instance,
        parameter_name="shape",
        parameter_risk_policy=parameter_risk_policy,
    )


def _validate_gamma_scale(
    value,
    *,
    instance,
    parameter_risk_policy: Literal["warn", "raise", "ignore"],
) -> None:
    _validate_gamma_positive_parameter(
        value,
        instance=instance,
        parameter_name="scale",
        parameter_risk_policy=parameter_risk_policy,
    )


def _validate_gamma_shape_realizations(values: np.ndarray) -> NDArray[np.bool_]:
    return np.isfinite(values) & (values > 0)


def _validate_gamma_scale_realizations(values: np.ndarray) -> NDArray[np.bool_]:
    return np.isfinite(values) & (values > 0)
```

### Agreed runtime-validation design (2026-10-06)

Keep construction and runtime parameter validators in the same distribution
validation module. For Normal, this is
`src/problab/validation/distributions/continuous/_normal.py`. Add a separate
private runtime-validation function for each parameter alongside its existing
construction validator. Define `_valid_parameter_sets` as a private class
attribute mapping parameter names to their mathematical valid sets, in the same
order as the constructor's `parameters` tuple. Construction validators read
these sets through `instance`. Keep stricter backend limits in the validation
module and use them for construction risk checks and runtime masks.

- Construction validators reason about mathematical and declared realization
  value sets, using `parameter_risk_policy` for potential numerical risks.
- Runtime validators check actual realized parameter arrays. Each returns a
  Boolean mask of the same shape, with `True` for valid positions. Prefer
  efficient vectorized NumPy checks; use another appropriate method when the
  constraint cannot be checked that way. Do not default to per-element SymPy
  membership checks in the sampling path.
- For example, Normal mean validity is `np.isfinite(mean)`; standard-deviation
  validity is `np.isfinite(std) & (std > 0)`. Check backend limits as well as
  mathematical constraints where relevant. Rules involving multiple parameters
  can have an additional joint check in the same validation module.
- Each concrete distribution connects its validators through
  `_validate_parameter_realizations(*parameters)`, returning named masks.
  `Distribution` supplies an empty default for distributions such as Categorical
  that have no sampled parameters. Override it for new parameterized samplers.
- Runtime validators identify valid positions; they do not sample, emit policy
  warnings, or replace values. Shared evaluation combines the masks and applies
  `numerical_error_policy` before calling the sampling backend. Under `raise`,
  invalid positions cause an informative error. Under `warn`/`ignore`, only valid
  positions are sampled and invalid positions become NaN in the original order;
  `warn` also emits a RuntimeWarning. All parameter arrays use the same combined
  mask. If all positions are invalid, the backend is not called.

Runtime mask functions receive arrays of shape `(context.num_samples,)`,
including broadcast arrays for scalar constants. Inputs can be read-only and
shared with other nodes; do not mutate them. Return a Boolean array of exactly
the same shape, not a scalar `all(...)`, shortened array, integer mask, or tuple
of indices. Check every parameter, even if construction used `raise`, and even
if output `validate=False`: neither setting replaces runtime validation.
Current shared evaluation trusts these mask contracts; tests must enforce them.

For a joint condition, add a separate mask entry with an informative label,
such as `"lower/upper"`, from a joint validator in the same validation module.
For example, a finite upper/lower pair might additionally require
`lower_values < upper_values`. Construction must reject provably invalid joint
definitions and apply the risk policy when joint numerical validity cannot be
guaranteed. `_parameter_realizations_guaranteed_valid()` only checks parameters
individually; also account for the joint condition when deciding `allows_nan`.

Avoid validators that produce warnings while inspecting invalid input: restrict
domain-sensitive calculations to candidate-valid positions, use stable forms,
and test under `np.errstate(all="raise")`. Counts require integer-valued checks,
not just positivity; do not convert large integers to float to check them. Near
machine limits, use bounds/comparisons that do not round an invalid value into
the valid range. The Binomial/Poisson validators are examples of these cases.

This deliberately permits expressing a domain both as a value set and as a fast
runtime check. Test their agreement at representative values and boundaries,
with separate coverage for any stricter backend limits. Reuse common numerical
checks where useful; a general translator from symbolic sets to fast masks is
not required. Realization support describes everything the sampling method can
return, including deliberately inserted NaN markers, rather than only successful
distribution outcomes. Keep mathematical support separate.

All-valid batches keep the backend dtype. In mixed batches, floating samples
keep their floating dtype. Integer samples use float64 when every valid output
is within +/-2**53; otherwise they use object storage to preserve exact integers
alongside NaN. All-invalid batches return float64 NaNs without a backend call.
Declare every possible storage dtype in realization support. A distribution
whose support includes this object fallback can be rejected by numeric
operations requiring native dtypes, even when one particular draw would fit in
float64. Object batches also require elementwise NaN inspection rather than
applying `np.isnan` directly to the object array.

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
- Normal, Binomial, and Poisson override it and reuse
  `_parameter_realizations_guaranteed_valid()` for domain and non-finite checks.
  Binomial and Poisson additionally account for backend parameter limits.
  Binomial's machine output bound comes from the realized trial-count support,
  since rounding can make it differ from the mathematical trial-count bound.

The context always checks sample shape and dtype. `validate=True` additionally
checks samples against the realization set, not the mathematical set.

Finite support, dtype, and exceptional-value permissions are independent:

| Field | Question to answer |
| --- | --- |
| `sympy_set` | Which finite numerical values can actually be returned after rounding or underflow? |
| `dtype_types` | Which NumPy scalar dtype classes can storage use on every path, including the all-invalid path? |
| `allows_positive_infinity` | Can positive overflow or an explicitly supported infinite output occur? |
| `allows_negative_infinity` | Can negative overflow or an explicitly supported infinite output occur? |
| `allows_nan` | Can parameter masking insert NaN, or can the backend itself legitimately produce it? |

Supply NumPy scalar classes such as `np.float64`, not dtype instances or strings.
Use concrete dtypes where known. A family such as `np.floating` is conservative
and may limit what downstream backend checks can guarantee. Do not copy Normal's
infinity flags to a bounded distribution, or assume an integer mathematical set
requires integer storage when NaNs can be inserted.

The shared guarantee helper checks declared finite membership and the three
permission flags; it does not sample, enforce dtype support, detect unregistered
backend constraints, or prove relationships between parameters. It returns
True for no parameters. Keep unknown risks conservative. The output support
must cover what sampling can return under warn/ignore even if a particular call
uses raise; the cached property has no per-call policy argument.

`@cached_property` is suitable when parameters and fixed configuration cannot
change after construction. If you introduce mutation, define how cached support
and graph metadata are invalidated; do not let them silently become stale.
Do not enable NaN just to hide an incorrect sampler or a failed test.

## 6. Implement sampling without breaking graph evaluation

Implement `_sample(*parameters, num_samples, rng)`; its parameter arrays arrive
in the same order passed to the base constructor. Return a numeric NumPy array
of shape `(num_samples,)` with a dtype permitted by the realization set.

### Worked distribution module

The following completes the sampling example in
`src/problab/distributions/continuous/gamma.py`, using the validation module
above. It inherits the optional exact hooks unchanged. The finite support is
nonnegative and positive infinity is allowed conservatively for overflow;
NaN permission covers invalid-parameter markers. If backend investigation finds
additional NaN-producing cases for valid parameters, either declare that risk
or handle/restrict it explicitly before adopting the example as a real class.

```python
from functools import cached_property
from numbers import Real
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gamma

from problab._config import DEF_PARAMETER_RISK_POLICY
from problab.distributions.base import Distribution
from problab.random_variables.base import RandomVariable
from problab.validation._decorator import _validate_parameters
from problab.validation.distributions._base import _validate_parameter_risk_policy
from problab.validation.distributions.continuous._gamma import (
    _validate_gamma_shape,
    _validate_gamma_scale,
    _validate_gamma_shape_realizations,
    _validate_gamma_scale_realizations,
)
from problab.value_sets.base import NumericValueSet
from problab.value_sets.homogeneous_numeric_value_set import HomogeneousNumericValueSet
from problab.value_sets.sets import NON_NEGATIVE_REALS, POSITIVE_REALS


class GammaDistribution(Distribution):
    symbol: ClassVar[str] = "Gamma"
    _valid_parameter_sets: ClassVar[dict[str, NumericValueSet]] = {
        "shape": POSITIVE_REALS,
        "scale": POSITIVE_REALS,
    }

    @_validate_parameters(
        validator_arguments=("parameter_risk_policy",),
        parameter_risk_policy=_validate_parameter_risk_policy,
        shape=_validate_gamma_shape,
        scale=_validate_gamma_scale,
    )
    def __init__(
        self,
        shape: RandomVariable | Real,
        scale: RandomVariable | Real,
        *,
        parameter_risk_policy: Literal["warn", "raise", "ignore"] = DEF_PARAMETER_RISK_POLICY,
    ) -> None:
        self._mathematical_value_set = NON_NEGATIVE_REALS
        super().__init__(parameters=(shape, scale))

    @property
    def value_set(self) -> HomogeneousNumericValueSet:
        return self._mathematical_value_set

    @cached_property
    def _realization_value_set(self) -> HomogeneousNumericValueSet:
        return HomogeneousNumericValueSet(
            sympy_set=NON_NEGATIVE_REALS.sympy_set,
            dtype_types=(np.float64,),
            allows_positive_infinity=True,
            allows_negative_infinity=False,
            allows_nan=not self._parameter_realizations_guaranteed_valid(),
        )

    def _validate_parameter_realizations(
        self,
        *parameters: np.ndarray,
    ) -> dict[str, np.ndarray]:
        shape, scale = parameters
        return {
            "shape": _validate_gamma_shape_realizations(shape),
            "scale": _validate_gamma_scale_realizations(scale),
        }

    def _sample(
        self,
        *parameters: np.ndarray,
        num_samples: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
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

`_sample()` receives the original batch when all parameters are valid, or only
valid positions and a reduced `num_samples` when some are invalid. The current
base implementation does not call it for zero valid positions. Always return an
array whose length matches the count passed to this call; do not cache the
original context count, sort parameters independently, or draw and discard
samples for invalid positions. Boolean indexing makes aligned copies of the
valid parameter positions without changing cached parameter arrays.

Sharing one random-variable parameter between several distributions must keep
its realizations shared. Repeated references to the same distribution node also
reuse samples within one context. Separate public `sample()` calls create new
contexts and therefore do not share caches, even when using the same generator.
Invalid-position filtering changes how many random draws are consumed; require
reproducibility for the same input and seed, not equivalence to sampling invalid
positions and discarding them. All-valid sampling should retain the original
backend sequence and dtype.

For distributions with object outcomes, use Categorical as the reference:
keep configured objects intact, pack each tuple/list/array as one element of an
object array, and declare `ObjectValueSet` as appropriate. Do not let `np.asarray`
turn atomic compound outcomes into an extra sample dimension. Numeric category
conversion must be lossless. The current shared NaN restoration was designed
for numeric outputs; Boolean, string, structured, or other new output kinds
need an explicit failure-storage contract before opting into parameter masking.

## 7. Respect numerical policies and current gaps

`parameter_risk_policy` governs potential parameter problems at construction.
`numerical_error_policy` governs actual numerical failures during sampling.
One setting does not imply or override the other.

`Distribution._evaluate()` now checks realized parameter masks before sampling.
Integrate new distributions through the hooks described above; construction
warnings alone do not handle invalid realized values. `validate=True` separately
checks output membership in the declared realization support.

Binomial checks the integer range used by SciPy's trial-count conversion.
Poisson checks NumPy Generator's rate ceiling, which reserves ten standard
deviations below the int64 limit; see
[NumPy's constraint definition](https://github.com/numpy/numpy/blob/main/numpy/random/_common.pyx).
Both construction risk checks and runtime validation account for these limits.
Unrelated backend exceptions are allowed to propagate.

**Still separate:** the context's NumPy error policy wraps operation-node
evaluation, not arithmetic inside distribution backend calls.
`Distribution.sample()` does not yet expose `numerical_error_policy`, although
`RandomVariable.sample()` does. These are not changed by the parameter masks.

Use the wrapper to select the runtime policy in a public example:

```python
variable = RandomVariable(GammaDistribution(2.0, 3.0))
samples = variable.sample(
    num_samples=100,
    rng=np.random.default_rng(42),
    numerical_error_policy="raise",
    validate=True,
)
```

Keep calls explicit with keywords. The former positional `validate` argument
on `RandomVariable.sample()` remains an unrelated known issue (B04).
`Distribution.sample()` and the statistics methods use the context's default
runtime policy; do not document a policy keyword those methods do not accept.

Construction policy, runtime policy, and output membership validation are three
separate controls. If a custom parameter declares a support that excludes the
invalid values it actually returns, `validate=True` can reject that parameter
node before distribution masking is reached. To test invalid-parameter handling,
give the parameter valid mathematical support and honest realization support
that includes its possible invalid machine values.

NaN markers remain in returned arrays; sampling does not drop, resample, impute,
or renormalize failed positions. Downstream statistics currently apply their
existing operations to the returned samples. They are not automatically
NaN-aware and do not promise an estimate conditioned on valid parameters.

## 8. Decide which exact methods to implement

The optional hooks are `_mean_exact`, `_variance_exact`, `_cdf_exact(x)`, and
`_ppf_exact(q)`. Return `None` when a hook is unavailable. The base class then
uses Monte Carlo for `Mode.AUTO`, or raises `NotImplementedError` for
`Mode.EXACT`; it derives exact standard deviation from exact variance.

| Hook | Input and result contract |
| --- | --- |
| `_mean_exact(self)` | Scalar real or complex mean, or `None` |
| `_variance_exact(self)` | Scalar real nonnegative variance, or `None` |
| `_cdf_exact(self, x)` | Scalar result for scalar `x`, shape-preserving array for array `x`, or `None` |
| `_ppf_exact(self, q)` | Scalar result for scalar `q`, shape-preserving array for array `q`, or `None` |

Public `cdf()` and `ppf()` first require real-valued mathematical support, even
if an exact hook is implemented. CDF rejects NaN inputs; PPF requires finite
probabilities in `[0, 1]`. Test both endpoints and backend conventions at them.
The base does not normalize exact-hook result types as it does some Monte Carlo
results, so follow the result contract explicitly. Do not implement a separate
`_std_exact` hook: none is called. `Mode.MONTE_CARLO` bypasses exact hooks.

For fixed shape/scale Gamma, mean is `shape * scale` and variance is
`shape * scale**2`. Support scalar/array CDF and PPF inputs and their endpoints
if implementing those hooks. Random parameters require a separate analytical
derivation: an unconditional mixture generally cannot use the fixed-parameter
formula with random variables or their means substituted.

Check `self.parameters` to distinguish scalar inputs from `RandomVariable`
inputs, or explicitly inspect constant parameter nodes if that is the agreed
contract. Do not use `_node_dependencies` alone to decide whether public inputs
were fixed: a variable can be backed by a constant node. If random parameters
are unsupported by an exact hook, return `None` for that case. Do not consume
RNG state from an exact hook. Decide how undefined moments or arithmetic
overflow are reported; `None` means the calculation is unavailable, not that a
defined exact answer happens to be zero or non-finite.

If adding `pdf()` or `pmf()`, validate its input type and `NaN` handling and
define values outside the support and at its boundaries. A density can be
infinite at a boundary without that being a sampling overflow.

## 9. Verify and document the addition

- **Unit:** constructor storage, symbol, support, every validator, accepted
  types, invalid ranges, supported dtypes, all risk policies, policy validation
  order, and exact-hook behavior. Patch `gamma.rvs` where the implementation
  uses it to check arguments, array order, size, generator, and returned result.
- When a construction-validator unit test supplies a fake `instance`, it now
  needs both `symbol` and `_valid_parameter_sets`. For example:

```python
from types import SimpleNamespace

instance = SimpleNamespace(
    symbol="CustomGamma",
    _valid_parameter_sets={
        "shape": POSITIVE_REALS,
        "scale": POSITIVE_REALS,
    },
)
```

The fixture should declare the intended contract independently. Do not weaken
assertions because a stale fake lacks the new class metadata. Construction
validators should be tested with fake nodes and patched dependencies; leave
complete `RandomVariable` workflows to integration/regression tests.

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

### Required cases for parameter handling

| Area | Cases and assertions |
| --- | --- |
| Class metadata | Mapping keys/order match graph parameters; policy is not a graph parameter; public scalar inputs and supplied variable objects are preserved; subclass symbols work before constructor initialization |
| Construction | Valid/invalid types, Python and NumPy booleans, invalid mathematical ranges, unknown support, unsupported dtypes; all three risk policies and validation order |
| Runtime masks | All five value patterns: valid, invalid, mixed, boundary, non-finite; signed/unsigned integers and supported float precisions; Boolean dtype, original shape, and no input mutation |
| Mathematical/backend agreement | Independent expected validity at boundaries; integer-valued versus fractional counts; exact integer limits and neighboring floats; stricter backend limits separately from mathematical validity |
| Shared evaluation | One mask per parameter; invalid entries in different parameters; no backend call under raise or with no valid positions; correct reduced parameter arrays, count, and identical RNG object |
| Output restoration | Original length/order; NaN only at invalid positions; no warnings for ignore, one RuntimeWarning for warn; float dtype preservation; integer/NaN conversion boundaries and exact large integers |
| Realization support | Safe parameters exclude masking NaNs; risky parameters allow them; correct finite support, dtype, and separate infinity flags; cached result if immutable; unknown or joint risk is not assumed safe |
| Graph integration | Shared parameter evaluated once across roots; explicitly requested parameter results retain their original arrays; scalar broadcasting and single-sample batches; repeatability with a seed |
| Policies versus validation | Invalid parameters handled even when `validate=False`; intended markers accepted with `validate=True`; a parameter lying about its own support is still rejected; construction and runtime policy are independent |
| Backend failures | Unrelated backend exceptions are not converted into masked parameter failures; all-valid calls preserve backend values/dtype and generator sequence |
| Parameterless/object distributions | Empty mask default still works; compound outcomes remain atomic; no unintended numeric conversion or extra dimensions |

Do not demand that every distribution allow every dtype or exceptional value.
Tests should establish its chosen contract. Test the guarantee helper and batch
assembly generically in `tests/unit_tests/distributions/base.py`; test concrete
rule delegation and support in the corresponding distribution unit file.

### Minimal backend delegation test for the worked example

This belongs in `tests/unit_tests/distributions/continuous/gamma.py` after the
example is implemented. It checks ProbLab's argument mapping, not SciPy's
sampling algorithm.

```python
import unittest
from unittest.mock import patch

import numpy as np

from problab.distributions.continuous.gamma import GammaDistribution


class GammaDistributionTests(unittest.TestCase):
    @patch("problab.distributions.continuous.gamma.gamma.rvs")
    def test_sample_passes_aligned_parameters_and_generator(self, rvs):
        distribution = GammaDistribution(2.0, 3.0)
        shape = np.array([2.0, 4.0])
        scale = np.array([3.0, 5.0])
        rng = np.random.default_rng(42)
        expected = np.array([1.0, 2.0])
        rvs.return_value = expected

        actual = distribution._sample(shape, scale, num_samples=2, rng=rng)

        self.assertIs(actual, expected)
        rvs.assert_called_once()
        self.assertIs(rvs.call_args.kwargs["a"], shape)
        self.assertIs(rvs.call_args.kwargs["scale"], scale)
        self.assertIs(rvs.call_args.kwargs["random_state"], rng)
        self.assertEqual(rvs.call_args.kwargs["loc"], 0)
        self.assertEqual(rvs.call_args.kwargs["size"], 2)
```

Use `tests/regression_tests/test_parameter_realization_policy.py` as the reference
for constructing a parameter with valid mathematical support but invalid
realizations. Add persistent cases for the new parameters; do not rely on an
interactive probe as the only record of a discovered defect.

### Commands and completion checklist

Run from the repository root using the existing virtual environment. These
focused Gamma commands apply only after its files and tests have been created:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.unit_tests.distributions.continuous.gamma tests.unit_tests.validation.distributions.continuous._gamma -v
.\.venv\Scripts\python.exe -B -m unittest tests.regression_tests.test_parameter_realization_policy -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m unittest discover -s docs/library_explorer -p "test_*.py" -v
```

Run new integration/statistical/property modules explicitly during development,
then verify they are discovered by the full command. Unit and integration
filenames need not start with `test_`; the custom loader handles their folders.
Regression, statistical, and property files should follow existing `test_*.py`
discovery conventions. Use fixed RNG seeds and analytical error bounds for
statistical checks, as described in [testing.md](testing.md).

Before declaring a new distribution complete:

- Its constructor, class metadata, validators, support declarations, and backend
  mapping agree on parameter order, domains, dtypes, and boundaries.
- Every runtime mask and all invalid-batch paths have persistent tests.
- Exact methods are either implemented for explicitly supported inputs or
  return `None`; public documentation says which behavior is available.
- Public exports and their tests are updated; examples use supported keywords.
- Mathematical, runtime, and backend limitations are documented, including
  precision, NaN handling, and downstream dtype restrictions.
- The full suite has been run, with known failures retained and new regressions
  investigated. Do not skip or mark intended behavior as expected failure to
  obtain a green run.
- Update [test_inventory.md](test_inventory.md), [test_gap_inventory.md](test_gap_inventory.md),
  [audit_todo.md](audit_todo.md), relevant API documentation, and user examples.
  Keep completed audit entries; record completion in the same to-do file.
- If shipping a refreshed source explorer snapshot, run
  `.\.venv\Scripts\python.exe -B docs/library_explorer/build_graph.py` and its
  extractor tests. It reads source statically; runtime example testing is still
  required. Do not touch `docs/mathematics/tmp.png`.

`pyproject.toml` currently declares Python >=3.11 and unbounded NumPy/SciPy/SymPy
dependencies. Compatibility defects and missing lower-bound testing remain
tracked in the audit. Do not interpret a single local passing run as proof of
the full declared compatibility range. Test supported versions when introducing
backend APIs or syntax with version requirements.
