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