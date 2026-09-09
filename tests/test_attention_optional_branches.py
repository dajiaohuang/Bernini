"""Regression checks for optional Qwen attention branches."""

import ast
from pathlib import Path
import unittest

from packaging import version


SOURCE = Path(__file__).parents[1] / "bernini" / "models" / "modeling_qwen2_5_vl.py"


class OptionalAttentionBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.compile_kwargs = cls._load_compile_kwargs(cls.tree)
        cls.flash_class = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Qwen2_5_VLFlashAttention2"
        )
        cls.forward = next(
            node
            for node in cls.flash_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )

    def test_flex_compile_version_checks_use_defined_symbols(self):
        self.assertIn("from packaging import version", self.source)
        self.assertIn('torch_version = version.parse(torch.__version__.split("+", 1)[0])', self.source)
        self.assertNotIn("is_torch_less_or_equal(", self.source)
        self.assertNotIn("get_torch_version()", self.source)

    def test_flex_compile_options_cover_stable_dev_and_rc_2_6_versions(self):
        expected_special_case = {
            "dynamic": False,
            "mode": "max-autotune-no-cudagraphs",
        }
        for value in ("2.6.0", "2.6.0.dev20260101", "2.6.0rc1", "2.6.0+cu128"):
            with self.subTest(value=value):
                self.assertEqual(self.__class__.compile_kwargs(version.parse(value)), expected_special_case)
        self.assertEqual(self.__class__.compile_kwargs(version.parse("2.5.1")), {"dynamic": False})
        self.assertEqual(self.__class__.compile_kwargs(version.parse("2.6.1")), {})

    def test_flash_forward_initializes_attention_weights(self):
        initializations = [
            node
            for node in self.forward.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "attn_weights" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        ]
        self.assertEqual(len(initializations), 1)
        returns = [node for node in ast.walk(self.forward) if isinstance(node, ast.Return)]
        self.assertTrue(any(isinstance(node.value, ast.Tuple) for node in returns))

    @staticmethod
    def _load_compile_kwargs(tree):
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_flex_attention_compile_kwargs"
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="packaging", names=[ast.alias(name="version")]), helper],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(SOURCE), "exec"), namespace)
        return namespace["_flex_attention_compile_kwargs"]


if __name__ == "__main__":
    unittest.main()
