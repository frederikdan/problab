# Testing ProbLab

ProbLab keeps different kinds of tests in separate folders. Each kind answers
a different question.

```text
tests/
├── unit_tests/          # Tests one source file or small piece of code
├── integration_tests/   # Tests ProbLab parts working together
├── statistical_tests/   # Tests results that need many random samples
├── regression_tests/    # Keeps a test for a bug that was fixed
├── property_tests/      # Tries many generated inputs and checks general rules
└── api_tests/           # Tests the public imports and public API
```

## Unit tests

Every source file should have a matching unit-test file with the same path
inside `tests/unit_tests`.

```text
src/problab/distributions/discrete/binomial.py
tests/unit_tests/distributions/discrete/binomial.py
```

When writing a unit test, follow these rules:

1. First decide what the file itself is responsible for. This can include
   creating objects, checking input, returning values, or calling the next
   piece of code it depends on.
2. Usually call the public methods of the file. It is also fine to test a
   private method directly when that method is an important part of that file.
   For example, test a distribution's `_sample()` method directly, and test
   the helper functions in `_categorical.py` directly.
3. Do not test another library through ProbLab. Replace outside code with a
   small fake object or `unittest.mock.patch`. For example, when testing
   `BinomialDistribution._sample()`, replace SciPy's `binom.rvs` and check
   that ProbLab sends it the right `n`, `p`, sample size, and random generator.
   SciPy itself is responsible for generating the random numbers correctly.
4. Do not let random luck decide whether a unit test passes. Give the method a
   fixed random generator, or a small fake generator that returns known values.
5. Check what a user or the next piece of code can observe: returned values,
   types, declared value sets, raised errors, and calls to direct dependencies.
6. Test one small behavior per test. Name tests clearly, for example
   `test_sample_passes_parameters_to_scipy`.

Do not use `RandomVariable`, `P`, a realization context, or a dependency graph
when testing a distribution file by itself. If the point of the test is that
those parts work together, put the test in `integration_tests`.

### Testing numerical behavior

Keep mathematical support and realization support separate in tests. The first
describes ideal values; the second describes actual NumPy arrays, including
their dtype, rounding, underflow, and permitted infinity or NaN values.

- Test descriptors in `unit_tests/operations/`, mathematical inference in
  `unit_tests/value_sets/_mathematical_inference.py`, and machine support in
  `_realization_inference.py`. Shared dtype rules belong in `_inference.py`.
- Check `warn`, `raise`, and `ignore` separately. Capture warnings and check
  their category; check exceptions and returned arrays as appropriate. Use
  `np.errstate` so a test restores NumPy's settings after it finishes.
- For support inference, compare declared support with independently computed
  results or known boundary cases. A conservative support can be wider than
  necessary; require it to contain possible outputs rather than requiring the
  smallest possible set.
- Include signed integer limits, large integers, float16/float32/float64,
  complex components, zero divisors, underflow, and rounded endpoints where
  relevant. Check object fallback when numeric conversion would lose a value.
- Test construction `parameter_risk_policy` separately from sampling
  `numerical_error_policy`. Construction warns about a possibility; sampling
  handles the values that actually occur.
- Add a public workflow test when several pieces must cooperate. For example,
  test `log2(exp(X))` when exp underflows, and condition an event on its
  complement after the joint event simplifies to a constant.

Only keep a failing test when its expected behavior follows an agreed contract
or a clearly reproduced defect. Correct obsolete imports, mocks, and assumptions
about storage separately; do not count those as library defects. Record genuine
remaining failures in `test_gap_inventory.md` and list every test method in
`test_inventory.md`.

## Regression tests

Add a regression test when a bug is found, even before it is fixed. The test
should show the smallest public example and the behavior the library should
provide. A known bug can leave the test failing; keep that failure visible
until the implementation is corrected. Give the file and test a name that
describes the bug. Do not combine unrelated bugs in one test. Do not skip or
mark known regressions as expected failures: the failing result is the signal
that the bug remains.

## Statistical tests

Statistical tests check random samples against a known distribution. Use a
fixed random generator and a large sample. Calculate the allowed difference
from the expected result using the correct sampling law; do not use a guessed
tolerance. For an event with true probability `p` and `n` trials, its observed
frequency has standard error `sqrt(p * (1 - p) / n)`. The tests allow six
standard errors. For a conditional probability, use the observed number of
samples satisfying the condition as `n`.

Calculate expected probabilities independently: finite sums for discrete
distributions and mixtures, and the normal CDF for normal examples. Test
complete expressions, including dependent and independent variables, random
distribution parameters, events, arithmetic, functions, and conditions.

Different results need different bounds. The tests use the exact chi-square
sampling law for normal sample variance, a Dvoretzky-Kiefer-Wolfowitz (DKW)
bound for empirical quantiles and CDFs, and an exact binomial lower-tail bound
when checking confidence-interval coverage over repeated experiments.

Test ProbLab's sampling and probability interfaces, rather than trying to
retest NumPy or SciPy in isolation. Keep the sample size large enough for the
chosen limit and small enough for the normal test suite to remain quick. A
fixed seed makes failures repeatable; passing statistical tests do not prove
that every possible model is correct. Add an analytical case when a new
distribution or probabilistic operation is implemented.

## Property tests

Property tests check a rule across many valid inputs. They do not check one
specific example. For example, a categorical distribution must always merge
equal categories and leave probabilities summing to one.

Use a fixed `numpy.random.Generator` to make the generated inputs repeatable.
Generate only valid inputs unless the rule is specifically about rejecting bad
input. Keep the rule simple and observable: a value, type, support, count, or
interval bound. A failed generated case must be easy to reproduce from the
test's fixed seed.

## Where each test belongs

| Question | Test folder |
| --- | --- |
| Does `BinomialDistribution._sample()` send `n`, `p`, sample size, and RNG to SciPy? | Unit |
| Can `RandomVariable(CategoricalDistribution(...))` sample tuple categories correctly? | Integration |
| Does a categorical distribution return about 30% `"large"` values in a very large sample? | Statistical |
| Does an integer-valued float stay in `Integers` after a bug fix? | Regression |
| Do many generated valid probability lists create a categorical distribution? | Property |
| Can a user import every documented name from `problab`? | API |

## Running tests

Run all currently registered test folders:

```text
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The 2026-10-06 run contains 646 test methods: 630 pass and 16 fail or error.
Several methods report multiple failing subtests. This command currently exits
with failures from known, unfixed bugs. See
`docs/audit_todo.md` for the current count and `docs/test_gap_inventory.md` for
the issues those tests expose. Tests for deferred exact calculations and the
confirmed follow-up defects remain deliberately active until those behaviors
are corrected. Public sampling-control tests now pass.

Run all unit tests:

```text
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/unit_tests -p "*.py" -v
```

Run the distribution unit tests:

```text
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/unit_tests/distributions -p "*.py" -v
```

Run property tests:

```text
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/property_tests -p "*.py" -v
```

Run statistical tests:

```text
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/statistical_tests -p "test_*.py" -v
```
