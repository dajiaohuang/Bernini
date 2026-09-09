"""Regression checks for FlowMatchScheduler device handling."""

import importlib.util
from pathlib import Path
import unittest

import torch


_SCHEDULER_PATH = Path(__file__).parents[1] / "bernini" / "models" / "scheduler.py"
_SPEC = importlib.util.spec_from_file_location("bernini_scheduler_under_test", _SCHEDULER_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
FlowMatchScheduler = _MODULE.FlowMatchScheduler


class SchedulerDeviceTests(unittest.TestCase):
    def test_cpu_timesteps_remain_on_cpu_during_step(self):
        scheduler = FlowMatchScheduler(num_inference_steps=3)
        scheduler.set_timesteps(num_inference_steps=3, device="cpu", dtype=torch.float32)
        model_output = torch.ones(2, 2)
        sample = torch.zeros(2, 2)
        timestep = scheduler.timesteps[0].clone()

        result = scheduler.step(model_output, timestep, sample)

        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.shape, sample.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_scalar_timestep_uses_configured_cpu_schedule(self):
        scheduler = FlowMatchScheduler(num_inference_steps=3)
        scheduler.set_timesteps(num_inference_steps=3, device="cpu", dtype=torch.float32)
        sample = torch.zeros(1)

        result = scheduler.step(torch.ones(1), float(scheduler.timesteps[0]), sample)

        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
