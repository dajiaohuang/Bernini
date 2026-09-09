"""Regression checks for W&B initialization on fresh and resumed training."""

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "tasks" / "bernini_renderer" / "train_bernini_renderer.py"


class WandbResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.main = next(node for node in cls.tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    def test_init_is_outside_training_step_loop(self):
        init_nodes = [
            node
            for node in ast.walk(self.main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "init"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "wandb"
        ]
        self.assertEqual(len(init_nodes), 1)
        train_loop = next(
            node
            for node in ast.walk(self.main)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "epoch"
        )
        self.assertLess(init_nodes[0].lineno, train_loop.lineno)

    def test_logging_uses_initialized_run(self):
        log_calls = [
            node
            for node in ast.walk(self.main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "log"
        ]
        self.assertEqual(len(log_calls), 1)
        self.assertIsInstance(log_calls[0].func.value, ast.Name)
        self.assertEqual(log_calls[0].func.value.id, "wandb_run")
        guards = [
            node
            for node in ast.walk(self.main)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "wandb_run"
        ]
        self.assertTrue(any(isinstance(node.test.ops[0], ast.IsNot) for node in guards))


if __name__ == "__main__":
    unittest.main()
