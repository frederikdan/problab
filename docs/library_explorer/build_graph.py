"""Extract a navigable graph from Python source without importing the library."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parents[1] / "src" / "problab"


def build_graph(source: Path = DEFAULT_SOURCE) -> dict:
    source = source.resolve()
    root = source.name
    nodes, trees, modules, bindings, exports, definitions = {}, {}, {}, {}, {}, {}
    local_names = {}
    edges = set()
    digest = hashlib.sha256()

    def add_node(id_, name, kind, parent, module, tree=None):
        line = getattr(tree, "lineno", 1)
        end = getattr(tree, "end_lineno", line)
        doc = ast.get_docstring(tree) if isinstance(tree, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None
        nodes[id_] = dict(id=id_, name=name, kind=kind, parent=parent, module=module,
                          line=line, endLine=end, doc=doc or "", signature="", public=False,
                          internal=name.startswith("_"), bases=[], fields=[])

    files = sorted(source.rglob("*.py"))
    for file in files:
        relative = file.relative_to(source)
        package = file.name == "__init__.py"
        parts = relative.parts[:-1] if package else (*relative.parts[:-1], file.stem)
        module = ".".join((root, *parts))
        content = file.read_text(encoding="utf-8-sig")
        digest.update(relative.as_posix().encode() + b"\0" + content.encode())
        tree = ast.parse(content, filename=str(relative))
        trees[module] = tree
        modules[module] = dict(path=f"src/{root}/{relative.as_posix()}", source=content)
        bindings[module] = {}
        exports[module] = []
        add_node(module, parts[-1] if parts else root, "package" if package else "module",
                 module.rpartition(".")[0] or None, module, tree)
        nodes[module]["endLine"] = len(content.splitlines())

    # Collect lexical definitions and imports, including imports under TYPE_CHECKING.
    def collect(statements, owner, module):
        for statement in statements:
            if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                id_ = f"{owner}.{statement.name}"
                is_class = isinstance(statement, ast.ClassDef)
                is_method = nodes[owner]["kind"] == "class"
                decorators = [ast.unparse(item) for item in statement.decorator_list]
                kind = "class" if is_class else "property" if "property" in decorators else "method" if is_method else "function"
                add_node(id_, statement.name, kind, owner, module, statement)
                definitions[id_] = statement
                if is_class:
                    nodes[id_]["bases"] = [ast.unparse(base) for base in statement.bases]
                    nodes[id_]["signature"] = f"class {statement.name}" + (f"({', '.join(nodes[id_]['bases'])})" if statement.bases else "")
                else:
                    local_names[id_] = {arg.arg for arg in (*statement.args.posonlyargs, *statement.args.args, *statement.args.kwonlyargs)}
                    for arg in (statement.args.vararg, statement.args.kwarg):
                        if arg:
                            local_names[id_].add(arg.arg)
                    nodes[id_]["signature"] = f"{'async ' if isinstance(statement, ast.AsyncFunctionDef) else ''}def {statement.name}({ast.unparse(statement.args)})"
                    if statement.returns:
                        nodes[id_]["signature"] += f" -> {ast.unparse(statement.returns)}"
                collect(statement.body, id_, module)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                if isinstance(statement, ast.ImportFrom):
                    package = module if nodes[module]["kind"] == "package" else module.rpartition(".")[0]
                    prefix = package.split(".")[:len(package.split(".")) - statement.level + 1] if statement.level else []
                    imported = ".".join([*prefix, *([statement.module] if statement.module else [])])
                    pairs = [(alias.asname or alias.name, f"{imported}.{alias.name}") for alias in statement.names if alias.name != "*"]
                else:
                    pairs = [(alias.asname or alias.name.split(".")[0], alias.name if alias.asname else alias.name.split(".")[0]) for alias in statement.names]
                for local, target in pairs:
                    bindings.setdefault(owner, {})[local] = target
                    if target.startswith(root + "."):
                        edges.add((owner, target, "imports"))
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if owner in local_names:
                        local_names[owner].add(target.id)
                    if target.id == "__all__" and owner == module:
                        try:
                            exports[module] = ast.literal_eval(statement.value)
                        except (ValueError, TypeError):
                            pass
                    elif nodes[owner]["kind"] == "class" and isinstance(statement, ast.AnnAssign):
                        nodes[owner]["fields"].append(f"{target.id}: {ast.unparse(statement.annotation)}")
                    elif owner == module and target.id.lstrip("_").isupper():
                        id_ = f"{owner}.{target.id}"
                        add_node(id_, target.id, "constant", owner, module, statement)
                        definitions[id_] = statement
                        nodes[id_]["signature"] = ast.unparse(statement)
            else:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.stmt):
                        collect([child], owner, module)
                    elif isinstance(child, ast.ExceptHandler):
                        collect(child.body, owner, module)

    for module, tree in trees.items():
        collect(tree.body, module, module)

    def resolve_path(path, seen=None):
        seen = set() if seen is None else seen
        if path in seen:
            return None
        seen.add(path)
        if path in nodes:
            return path
        parts = path.split(".")
        for index in range(len(parts) - 1, 0, -1):
            scope, name = ".".join(parts[:index]), parts[index]
            target = bindings.get(scope, {}).get(name)
            if target:
                return resolve_path(".".join([target, *parts[index + 1:]]), seen)
        return None

    def resolve_name(name, owner):
        scope = owner
        while scope:
            first_part = name.split(".")[0]
            if first_part in local_names.get(scope, set()) and first_part not in bindings.get(scope, {}):
                return None
            candidate = f"{scope}.{name}"
            found = resolve_path(candidate)
            if found:
                return found
            scope = nodes.get(scope, {}).get("parent")
        return resolve_path(name)

    def class_owner(owner):
        while owner in nodes:
            if nodes[owner]["kind"] == "class":
                return owner
            owner = nodes[owner]["parent"]
        return None

    def resolve_expression(expr, owner):
        if isinstance(expr, ast.Name):
            return resolve_name(expr.id, owner)
        if isinstance(expr, ast.Attribute):
            if isinstance(expr.value, ast.Name) and expr.value.id in {"self", "cls"}:
                cls = class_owner(owner)
                return resolve_path(f"{cls}.{expr.attr}") if cls else None
            parent = resolve_expression(expr.value, owner)
            return resolve_path(f"{parent}.{expr.attr}") if parent else None
        return None

    resolved_edges = set()
    for start, target, kind in edges:
        end = resolve_path(target)
        if end and start != end:
            resolved_edges.add((start, end, kind))

    def visit_body(tree, owner):
        if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and definitions.get(owner) is not tree:
            return
        if isinstance(tree, ast.Call):
            target = resolve_expression(tree.func, owner)
            if target and target != owner:
                resolved_edges.add((owner, target, "calls"))
            elif not target:
                nodes[owner]["unresolvedCalls"] = nodes[owner].get("unresolvedCalls", 0) + 1
        if isinstance(tree, (ast.Name, ast.Attribute)) and isinstance(tree.ctx, ast.Load):
            target = resolve_expression(tree, owner)
            if target and target != owner:
                resolved_edges.add((owner, target, "references"))
        for child in ast.iter_child_nodes(tree):
            visit_body(child, owner)

    for id_, definition in definitions.items():
        if isinstance(definition, ast.ClassDef):
            for base in definition.bases:
                expression = base.value if isinstance(base, ast.Subscript) else base
                target = resolve_expression(expression, id_)
                if target:
                    resolved_edges.add((id_, target, "inherits"))
        visit_body(definition, id_)

    for module, names in exports.items():
        for name in names:
            target = resolve_path(f"{module}.{name}")
            if target:
                nodes[target]["exportedFrom"] = sorted(set([*nodes[target].get("exportedFrom", []), module]))
                if module == root:
                    nodes[target]["public"] = True

    # Imports and calls already express the more specific connection.
    specific = {(a, b) for a, b, kind in resolved_edges if kind != "references"}
    resolved_edges = {(a, b, kind) for a, b, kind in resolved_edges if kind != "references" or (a, b) not in specific}
    incoming = Counter(b for _, b, _ in resolved_edges)
    for id_, node in nodes.items():
        node["incoming"] = incoming[id_]
        node["importance"] = incoming[id_] + (20 if node["public"] else 0)
        node["path"] = modules[node["module"]]["path"]
    counts = Counter(node["kind"] for node in nodes.values())
    return dict(root=root, generatedAt=datetime.now(timezone.utc).isoformat(), sourceHash=digest.hexdigest()[:12],
                nodes=list(nodes.values()), edges=[dict(source=a, target=b, kind=k) for a, b, k in sorted(resolved_edges)],
                modules=modules, counts=dict(counts),
                limitations="Static source map. Calls through runtime objects, dynamic imports and inherited method dispatch may be unresolved. Imports include TYPE_CHECKING. References include annotations and decorators. Arrows point from the user to its dependency; they are not execution order.")


def write_snapshot(source=DEFAULT_SOURCE, output=HERE / "graph-data.js"):
    graph = build_graph(source)
    payload = json.dumps(graph, ensure_ascii=True, separators=(",", ":"))
    output.write_text("// Generated by build_graph.py. Do not edit by hand.\nwindow.PROBLAB_GRAPH = " + payload + ";\n", encoding="utf-8")
    print(f"Mapped {len(graph['nodes'])} definitions and containers, {len(graph['edges'])} connections -> {output}")
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=HERE / "graph-data.js")
    args = parser.parse_args()
    write_snapshot(args.source, args.output)
