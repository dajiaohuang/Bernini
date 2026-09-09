"""Regression checks for image and batch output paths."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


SOURCE = Path(__file__).parents[1] / "bernini" / "io_utils.py"


class OutputPathTests(unittest.TestCase):
    def test_single_frame_mp4_default_is_written_as_a_real_png(self):
        output = np.array(
            [[[[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]], [[0.25, 0.75, 0.0], [0.0, 1.0, 0.25]]]],
            dtype=np.float32,
        )
        with self._load_io_utils() as io_utils:
            with tempfile.TemporaryDirectory() as directory:
                requested = str(Path(directory) / "output.mp4")

                with mock.patch.object(
                    io_utils.imageio, "imwrite", side_effect=self._write_png, create=True
                ):
                    actual = io_utils.save_output(output, requested)

                self.assertEqual(Path(actual).suffix, ".png")
                self.assertFalse(Path(requested).exists())
                with Image.open(actual) as encoded:
                    self.assertEqual(encoded.format, "PNG")
                    self.assertEqual(encoded.size, (2, 2))
                    self.assertEqual(encoded.getpixel((0, 0)), (0, 127, 255))

    def test_batch_defaults_are_unique_but_explicit_outputs_are_preserved(self):
        with self._load_io_utils() as io_utils:
            self.assertEqual(
                io_utils.resolve_output_path("outputs/output.mp4", None, 0, 2),
                "outputs/output_0000.mp4",
            )
            self.assertEqual(
                io_utils.resolve_output_path("outputs/output.mp4", None, 1, 2),
                "outputs/output_0001.mp4",
            )
            self.assertEqual(
                io_utils.resolve_output_path("outputs/output.mp4", "custom.mp4", 1, 2),
                "custom.mp4",
            )

    @staticmethod
    def _write_png(path, array):
        Image.fromarray(array).save(path, format="PNG")

    def _load_io_utils(self):
        class ModuleLoader:
            def __enter__(loader_self):
                loader_self.previous_imageio = sys.modules.get("imageio", mock.sentinel.missing)
                if loader_self.previous_imageio is mock.sentinel.missing:
                    sys.modules["imageio"] = types.ModuleType("imageio")
                spec = importlib.util.spec_from_file_location("bernini_test_io_utils", SOURCE)
                loader_self.module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = loader_self.module
                spec.loader.exec_module(loader_self.module)
                return loader_self.module

            def __exit__(loader_self, exc_type, exc_value, traceback):
                sys.modules.pop("bernini_test_io_utils", None)
                if loader_self.previous_imageio is mock.sentinel.missing:
                    sys.modules.pop("imageio", None)

        return ModuleLoader()


if __name__ == "__main__":
    unittest.main()
