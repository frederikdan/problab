# API rules

## Public API
- Intended for library users to call.
- Names have no leading underscore.
- Supported import paths are explicitly defined.
- Validate input types and constraints where needed for a clear contract.
- Validation may be delegated if it happens before the input is used.

## Internal implementation
- Intended only for use within the library.
- Names use a leading underscore.
- Helpers may use ordinary names inside explicitly internal modules.
- Trust inputs already validated by their callers.
- Keep checks needed to protect internal correctness.

## Extension hooks
- Intended for subclasses to implement, rather than users to call directly.
- Method names use a leading underscore.
- Define the required input and output contracts.
- Validate returned results at the boundary that consumes them.

## Validation conventions
- Use TypeError for unsupported input types.
- Use ValueError for invalid values of supported types.
- Avoid repeating validation along the same call path.
- Always enforce required sample shapes and declared dtype constraints.
- Enable expensive sample membership checks through validate=True.

## Package exports

The installed package uses `problab` as its import root. Public names are listed
explicitly in each public package's `__all__`:

| Package | Public exports |
| --- | --- |
| `problab` | `RandomVariable`, `Distribution`, `NormalDistribution`, `BinomialDistribution`, `PoissonDistribution`, `CategoricalDistribution`, `P`, `NodeGraph`, `ProbabilityResult`, `ConfidenceInterval`, `ProbabilityInterval`, `Mode`, `ValueSet` |
| `problab.distributions` | `Distribution`, `Mode`, and the four implemented concrete distributions |
| `problab.distributions.continuous` | `NormalDistribution` |
| `problab.distributions.discrete` | `BinomialDistribution`, `CategoricalDistribution`, `PoissonDistribution` |
| `problab.random_variables` | `RandomVariable`, `NodeGraph` |
| `problab.probability` | `P`, `ProbabilityResult`, `ConfidenceInterval`, `ProbabilityInterval` |
| `problab.value_sets` | `ValueSet` and the named value-set constants from `sets.py` |
| `problab.functions` | The scalar/random-variable functions from `basic.py`, `exponential.py`, and `trigonometric.py` |

Validation and statistical estimation helpers are internal. Empty distribution
modules do not define public distributions yet. Internal modules use the same
canonical `problab` package name to avoid duplicate class identities.

Random-variable and probability package exports load on demand to avoid circular
imports during distribution initialization.

## Distribution parameters and categorical configuration

`Distribution.parameters` returns a tuple of original graph parameter inputs.
For example, `NormalDistribution(mean=rv, std=1).parameters` returns `(rv, 1)`,
preserving the identity of `rv`. Internal `_parameter_nodes` holds the graph
representation of those inputs.

`CategoricalDistribution.parameters` returns `()`. Its fixed configuration is
available through two read-only properties:

- `categories`: a tuple of the supplied category objects, in sampling order.
- `probabilities`: a tuple of normalized probabilities used for sampling, in the
  same order. These may differ slightly from the supplied probabilities when
  their sum is accepted within the validation tolerance around one.

For example:

```python
from problab import CategoricalDistribution

distribution = CategoricalDistribution(categories=[10, 20], probabilities=[0.4, 0.6])
assert distribution.parameters == ()
assert distribution.categories == (10, 20)
assert distribution.probabilities == (0.4, 0.6)
```

Exact distribution methods remain extension hooks; exporting `Mode` does not
add implementations for `Mode.EXACT`.
