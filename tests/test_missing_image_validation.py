"""Regression checks for missing image input validation."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


SOURCE = Path(__file__).parents[1] / "bernini" / "data_utils.py"


class MissingImageValidationTests(unittest.TestCase):
    def test_missing_path_raises_file_not_found_error(self):
        with self._load_data_utils():
            missing_path = str(Path(tempfile.gettempdir()) / "bernini-missing-input.png")
            with self.assertRaises(FileNotFoundError) as context:
                self.data_utils.generate_unified_inputs(
                    prompt="test", input_image_paths=[missing_path]
                )
            self.assertEqual(str(context.exception), f"Input image not found: {missing_path}")

    def test_existing_image_dimensions_are_recorded(self):
        with self._load_data_utils():
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = Path(tmpdir) / "input.png"
                Image.new("RGB", (7, 5), color=(1, 2, 3)).save(image_path)

                result = self.data_utils.generate_unified_inputs(
                    prompt="test", input_image_paths=[str(image_path)]
                )

        image_entry = next(item for item in json.loads(result) if item["type"] == "image")
        self.assertEqual(image_entry["width"], 7)
        self.assertEqual(image_entry["height"], 5)

    def _load_data_utils(self):
        class ModuleLoader:
            def __enter__(loader_self):
                loader_self.previous_decord = sys.modules.get("decord", mock.sentinel.missing)
                sys.modules["decord"] = types.ModuleType("decord")
                spec = importlib.util.spec_from_file_location("bernini_test_data_utils", SOURCE)
                loader_self.module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = loader_self.module
                spec.loader.exec_module(loader_self.module)
                self.data_utils = loader_self.module
                return loader_self

            def __exit__(loader_self, exc_type, exc_value, traceback):
                sys.modules.pop("bernini_test_data_utils", None)
                if loader_self.previous_decord is mock.sentinel.missing:
                    sys.modules.pop("decord", None)
                else:
                    sys.modules["decord"] = loader_self.previous_decord

        return ModuleLoader()


if __name__ == "__main__":
    unittest.main()
