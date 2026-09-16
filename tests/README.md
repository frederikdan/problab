# Test layout

The test tree is organized by test purpose:

- `unit_tests/` mirrors `src/problab/`.
- `integration_tests/` covers multiple library components working together.
- `statistical_tests/` checks sampling and probability estimates.
- `regression_tests/` preserves fixes for previously found bugs.
- `property_tests/` checks general rules across generated inputs.
- `api_tests/` checks the documented public interface.

Run the current unit-test tree with:

```text
.\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests/unit_tests -p "*.py" -v
```

The same unit tests are also available through the top-level test loader:

```text
.\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v
```
