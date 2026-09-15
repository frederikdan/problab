# ProbLab library explorer

An interactive source map created by Codex. The interface uses a Codex-inspired
graphite palette, restrained accents, expandable views, and an optional light theme.
Everything runs locally, without third-party JavaScript, fonts, build tools, or network services.

## Open

Open `index.html` directly in a browser. The included `graph-data.js` is a snapshot
of the library source at generation time.

For live source refresh, run this command from the repository root:

```powershell
.\.venv\Scripts\python.exe -B docs/library_explorer/serve.py
```

Visit **http://127.0.0.1:8765**. The refresh button reads the current Python files.
The server binds only to localhost. Stop it with Ctrl+C. Use `--port 8766` if needed.

## Explore

- **Architecture:** start with the top-level packages and internal expression modules.
- **Select a card:** inspect its description, signature, fields, and relationships.
- **Explore / double-click:** open its modules, classes, functions, and methods.
- **Map connections:** focus on the selected component and its dependencies/dependents.
- **Source:** read the actual source lines in the inspector.
- **Public API:** see definitions exported by the root `problab.__all__`.
- **Search (`/` or Ctrl/Cmd+K):** jump to any definition, including internal ones.
- **Graph / list:** use the graph for relationships and the list for scanning all definitions.
- **Graph pages:** at most nine cards appear together so names remain legible. Use the
  arrows above the graph for further components. The inspector lists all connections
  for the selection. Connections view keeps the central component on every page.
- **Drag / wheel / + / − / F:** pan, zoom, or fit the current graph.
- **Connection filters:** toggle imports, inheritance, calls, and references.
- **Internals:** include or hide names beginning with an underscore.

URLs preserve the scope, selected definition, view, and graph page. The theme is
saved locally when browser storage is available. Reduced-motion preferences disable animations.

## Refresh the portable snapshot

```powershell
.\.venv\Scripts\python.exe -B docs/library_explorer/build_graph.py
```

Reload the page afterward. Live refresh does not overwrite the committed snapshot;
run the build command before committing an updated snapshot. The analyzer uses the
standard library and does **not** import or execute ProbLab.

## What the map means

- Definitions, containment, source locations, signatures, docstrings, and fields
  come from Python's abstract syntax tree (`ast`).
- Internal imports and chains of re-exports are resolved to original definitions.
- `inherits` links a class to a resolvable base class.
- `calls` links an explicitly resolvable call to its target. Direct `self.method()`
  calls are linked when that method is defined on the enclosing class.
- `references` includes resolvable names in bodies, annotations, and decorators.
- Container-level edges aggregate connections between their descendants.
- An arrow points **from the component using something to its dependency**.
  It does not indicate execution order or sampling order.
- The dependency bar counts distinct referencing definitions outside a component.
  Bar lengths are relative to the most-referenced component on the current page.
  Overview HUB badges identify components with at least 65% of that maximum count.
  Public root exports receive an API badge and sort before other symbols.
  This is a navigational aid, not a claim about architectural quality.
- Brief top-level package descriptions are editorial; graph facts come from source.

This is static analysis, not a complete runtime call graph. Dynamic dispatch,
inherited method dispatch, computed attributes, external dependencies, and dynamic
imports may not resolve. Type-checking-only imports are included. Unknown call sites
are reported rather than assigned guessed targets. Empty placeholder modules remain
in the source data and search, but are omitted from the contents graph.

## Files and checks

| File | Purpose |
| --- | --- |
| `index.html` | Application structure |
| `styles.css` | Responsive dark/light design and animations |
| `app.js` | Graph navigation, search, inspector, and interactions |
| `build_graph.py` | AST analyzer and snapshot generator |
| `graph-data.js` | Generated definitions, relationships, and source snapshot |
| `serve.py` | Optional local server with a fresh-source endpoint |
| `test_build_graph.py` | Analyzer regression tests using temporary source fixtures |

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s docs/library_explorer -p "test_*.py" -v
node --check docs/library_explorer/app.js
```
