"""Regression checks for planner prediction-mask selection."""

import ast
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).parents[1] / "bernini" / "pipeline.py"


class MaskSelectionTests(unittest.TestCase):
    def test_empty_and_singleton_index_zero_masks_are_distinguished(self):
        empty = torch.zeros(4, dtype=torch.bool).nonzero(as_tuple=True)[0]
        singleton_zero = torch.tensor([True, False, False, False]).nonzero(as_tuple=True)[0]

        self.assertEqual(empty.numel(), 0)
        self.assertEqual(singleton_zero.tolist(), [0])
        self.assertNotEqual(singleton_zero.numel(), 0)

    def test_pipeline_uses_index_count_for_empty_check(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        source = SOURCE.read_text(encoding="utf-8")

        self.assertIn("pred_indices = mask_to_pred.nonzero(as_tuple=True)[0]", source)
        self.assertIn("if pred_indices.numel() == 0:", source)
        self.assertNotIn("mask_to_pred.nonzero(as_tuple=True)[0].sum()", source)
        self.assertIsNotNone(tree)


if __name__ == "__main__":
    unittest.main()
