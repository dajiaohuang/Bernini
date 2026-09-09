"""Regression checks for the VIT-only preprocessing initialization path."""

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "tools" / "preprocess_data.py"


class OnlyVitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.main = next(node for node in cls.tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    def test_vae_loader_is_guarded_by_only_vit(self):
        loader = next(
            node
            for node in ast.walk(self.main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "AutoencoderKLWan"
        )
        guard = next(
            node
            for node in ast.walk(self.main)
            if isinstance(node, ast.If) and node.body and loader in ast.walk(node)
        )
        self.assertIsInstance(guard.test, ast.UnaryOp)
        self.assertIsInstance(guard.test.op, ast.Not)
        self.assertIsInstance(guard.test.operand, ast.Attribute)
        self.assertEqual(guard.test.operand.attr, "only_vit")

    def test_vae_state_is_explicitly_empty_before_optional_setup(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("vae_model = None", source)
        self.assertIn("vae_transform = None", source)
        self.assertIn("if not args.only_vit:", source)


if __name__ == "__main__":
    unittest.main()
