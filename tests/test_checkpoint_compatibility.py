"""CPU-only checkpoint regression, independently runnable on newer Python/Torch."""
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_explicit_new_torch_default_and_all_rng_restoration(self):
        import torch
        from project_utils.reproduction import save_training_state, restore_training_state, load_dino_backbone
        model, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=.1, momentum=.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        head(model(torch.ones(1, 2))).sum().backward()
        optimizer.step()
        scheduler.step()
        load = torch.load
        calls = []
        def newer_default(*args, **kwargs):
            self.assertIn('weights_only', kwargs)
            calls.append(kwargs['weights_only'])
            return load(*args, **kwargs)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'internal.pt'
            save_training_state(path, model, head, optimizer, scheduler, 3)
            expected = (torch.rand(3), np.random.rand(3), random.random())
            with patch('torch.load', side_effect=newer_default):
                restored = restore_training_state(path, model, head, optimizer, scheduler)
                torch.save({'weight': model.weight.detach()}, path)
                load_dino_backbone(path)
            self.assertEqual(calls, [False, True])
            self.assertTrue(torch.equal(expected[0], torch.rand(3)))
            np.testing.assert_array_equal(expected[1], np.random.rand(3))
            self.assertEqual(expected[2], random.random())
            self.assertEqual(restored['epoch'], 3)
            self.assertEqual(scheduler.state_dict(), restored['scheduler'])
            self.assertTrue(optimizer.state)


if __name__ == '__main__':
    unittest.main()
