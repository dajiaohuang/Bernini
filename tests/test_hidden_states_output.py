"""Regression checks for disabled hidden-state output."""

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "bernini" / "models" / "modeling_qwen2_5_vl.py"


class HiddenStatesOutputTests(unittest.TestCase):
    def test_postprocessing_is_guarded_when_hidden_states_are_disabled(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        comprehensions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.GeneratorExp)
            and isinstance(node.elt, ast.Call)
            and isinstance(node.elt.func, ast.Attribute)
            and node.elt.func.attr == "get_sp_hidden_states"
            and any(
                isinstance(name, ast.Name) and name.id == "all_hidden_states"
                for name in ast.walk(node)
            )
        ]
        self.assertEqual(len(comprehensions), 1)

        parent_map = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        guard = parent_map[comprehensions[0]]
        while not isinstance(guard, ast.If):
            guard = parent_map[guard]
        self.assertIsInstance(guard, ast.If)
        self.assertIsInstance(guard.test, ast.Compare)
        self.assertIsInstance(guard.test.left, ast.Name)
        self.assertEqual(guard.test.left.id, "all_hidden_states")
        self.assertIsInstance(guard.test.ops[0], ast.IsNot)
        self.assertIsInstance(guard.test.comparators[0], ast.Constant)
        self.assertIsNone(guard.test.comparators[0].value)

    def test_output_contract_keeps_disabled_value_as_none(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("all_hidden_states = () if output_hidden_states else None", source)
        self.assertIn("if all_hidden_states is not None:", source)


if __name__ == "__main__":
    unittest.main()
