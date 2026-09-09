"""Regression checks for the renderer Python API checkpoint defaults."""

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "bernini" / "pipeline.py"


class PipelineDefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.method = next(
            node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "from_pretrained"
        )

    def test_renderer_defaults_derive_diffusers_mode(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("load_ckpt_weights: bool = None", source)
        self.assertIn("load_ckpt_weights = high_noise_ckpt is not None", source)
        self.assertIn("high_noise_ckpt and low_noise_ckpt must be provided together", source)

    def test_module_parses(self):
        ast.parse(SOURCE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
