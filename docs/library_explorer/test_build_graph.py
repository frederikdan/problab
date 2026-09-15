"""Focused regression checks for the explorer's static analysis."""

import tempfile
import unittest
from pathlib import Path

from build_graph import build_graph


class SourceGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "example"
        self.source.mkdir()
        self.write("__init__.py", 'from .api import Derived\n__all__ = ["Derived"]\n')
        self.write("api/__init__.py", 'from .types import Derived\n')
        self.write("api/types.py", '''from ..base import Base as Parent
from ..helpers import helper as run

class Derived(Parent):
    """A documented subclass."""
    count: int

    @property
    def value(self):
        return self.compute()

    def compute(self):
        return run()

    def dynamic(self, receiver):
        return receiver.unknown()
''')
        self.write("base.py", 'class Base:\n    pass\n')
        self.write("helpers.py", 'def helper():\n    return 42\n')

    def write(self, name, content):
        file = self.source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")

    def graph(self):
        graph = build_graph(self.source)
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = {(edge["source"], edge["target"], edge["kind"]) for edge in graph["edges"]}
        return graph, nodes, edges

    def test_reexports_resolve_to_original_definition(self):
        _, nodes, edges = self.graph()
        self.assertTrue(nodes["example.api.types.Derived"]["public"])
        self.assertIn(("example", "example.api.types.Derived", "imports"), edges)
        self.assertNotIn("example.Derived", nodes)

    def test_alias_calls_inheritance_and_self_methods(self):
        _, _, edges = self.graph()
        self.assertIn(("example.api.types.Derived", "example.base.Base", "inherits"), edges)
        self.assertIn(("example.api.types.Derived.compute", "example.helpers.helper", "calls"), edges)
        self.assertIn(("example.api.types.Derived.value", "example.api.types.Derived.compute", "calls"), edges)
        self.assertNotIn(("example.api.types.Derived", "example.helpers.helper", "calls"), edges)

    def test_dynamic_calls_are_not_invented(self):
        _, nodes, edges = self.graph()
        self.assertEqual(nodes["example.api.types.Derived.dynamic"]["unresolvedCalls"], 1)
        self.assertFalse(any(a == "example.api.types.Derived.dynamic" and k == "calls" for a, _, k in edges))

    def test_parameters_shadow_imported_names(self):
        self.write("shadow.py", 'from .helpers import helper\ndef invoke(helper):\n    return helper()\n')
        _, nodes, edges = self.graph()
        self.assertEqual(nodes["example.shadow.invoke"]["unresolvedCalls"], 1)
        self.assertNotIn(("example.shadow.invoke", "example.helpers.helper", "calls"), edges)

    def test_properties_fields_docstrings_and_source_locations(self):
        graph, nodes, _ = self.graph()
        cls = nodes["example.api.types.Derived"]
        self.assertEqual(cls["doc"], "A documented subclass.")
        self.assertEqual(cls["fields"], ["count: int"])
        prop = nodes["example.api.types.Derived.value"]
        self.assertEqual(prop["kind"], "property")
        self.assertEqual(prop["parent"], cls["id"])
        self.assertIn("def value", graph["modules"][prop["module"]]["source"].splitlines()[prop["line"] - 1])

    def test_reads_source_without_executing_it_and_changes_hash(self):
        graph, _, _ = self.graph()
        self.write("danger.py", 'raise RuntimeError("This source must never execute")\n')
        changed, _, _ = self.graph()
        self.assertNotEqual(graph["sourceHash"], changed["sourceHash"])

    def test_every_edge_and_parent_resolves(self):
        graph, nodes, edges = self.graph()
        self.assertTrue(all(a in nodes and b in nodes for a, b, _ in edges))
        self.assertTrue(all(node["parent"] is None or node["parent"] in nodes for node in graph["nodes"]))


if __name__ == "__main__":
    unittest.main()
