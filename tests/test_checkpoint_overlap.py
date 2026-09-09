"""Regression checks for rejecting unrelated transformer checkpoints."""

import unittest

from bernini.weights import _validate_state_dict_overlap


class CheckpointOverlapTests(unittest.TestCase):
    def test_unrelated_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no keys matching"):
            _validate_state_dict_overlap(
                {"totally_unrelated": object()},
                {"transformer.layer.weight": object()},
                "high-noise",
            )

    def test_matching_checkpoint_is_accepted_and_counted(self):
        overlap = _validate_state_dict_overlap(
            {"transformer.layer.weight": object(), "extra": object()},
            {"transformer.layer.weight": object(), "other": object()},
            "low-noise",
        )
        self.assertEqual(overlap, 1)


if __name__ == "__main__":
    unittest.main()
