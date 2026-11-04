import copy
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from dean_integration.online_step_smoke import (
    audit_parameter_diff, build_online_pair, gradient_report, pseudo_target_fixture,
    run_online_step, snapshot_parameters, validate_gradients, validate_setup,
)


def fixture_groups():
    return {name: {"p": nn.Parameter(torch.tensor([1., 2.]), requires_grad=name in ("online_pet", "online_head"))}
            for name in ("offline_model", "offline_head", "online_frozen", "online_pet", "online_head")}


class ParameterBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.groups = fixture_groups()
        self.before = snapshot_parameters(self.groups)

    def update(self):
        pet, head = self.groups["online_pet"]["p"], self.groups["online_head"]["p"]
        optimizer = torch.optim.AdamW([pet, head], lr=1e-4)
        (pet.square().sum() + head.square().sum()).backward()
        gradients = gradient_report(self.groups)
        optimizer.step()
        return gradients

    def test_exact_cpu_snapshot_and_valid_single_step_boundary(self):
        self.assertEqual(validate_setup(self.groups), [])
        gradients = self.update()
        result = audit_parameter_diff(self.before, self.groups, gradients)
        self.assertEqual(result["status"], "PASS")
        for group in ("offline_model", "offline_head", "online_frozen"):
            self.assertEqual(result["groups"][group]["changed"], [])
        for group in ("online_pet", "online_head"):
            self.assertEqual(result["groups"][group]["gradient_nonzero_and_changed"], ["p"])
        self.assertTrue(torch.equal(self.before["online_pet"]["p"]["value"], torch.tensor([1., 2.])))

    def test_frozen_mutation_fails_closed(self):
        gradients = self.update()
        for group in ("offline_model", "offline_head", "online_frozen"):
            groups = copy.deepcopy(self.groups)
            with torch.no_grad():
                groups[group]["p"][0] += 1
            result = audit_parameter_diff(self.before, groups, gradients)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn(f"Frozen group changed: {group}", result["errors"])

    def test_missing_pet_update_and_decay_only_are_not_passes(self):
        gradients = self.update()
        with torch.no_grad():
            self.groups["online_pet"]["p"].copy_(self.before["online_pet"]["p"]["value"])
        result = audit_parameter_diff(self.before, self.groups, gradients)
        self.assertEqual(result["status"], "FAIL")
        gradients["online_pet"]["nonzero"] = []
        self.assertTrue(validate_gradients(gradients))

    def test_nonfinite_parameter_and_gradient_fail_closed(self):
        gradients = self.update()
        with torch.no_grad():
            self.groups["online_head"]["p"][0] = float("inf")
        self.groups["online_pet"]["p"].grad[0] = float("nan")
        result = audit_parameter_diff(self.before, self.groups, gradient_report(self.groups))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("Nonfinite gradient: online_pet", result["errors"])
        self.assertEqual(result["groups"]["online_head"]["nonfinite"], ["p"])

    def test_wrong_trainability_shared_storage_and_frozen_gradient_fail(self):
        self.groups["offline_head"]["p"].requires_grad_(True)
        self.assertTrue(validate_setup(self.groups))
        groups = fixture_groups()
        groups["online_frozen"]["p"] = nn.Parameter(groups["offline_model"]["p"].detach(), requires_grad=False)
        self.assertTrue(any("shared" in error for error in validate_setup(groups)))
        groups = fixture_groups()
        groups["offline_model"]["p"].grad = torch.zeros(2)
        self.assertIn("Frozen group received gradients: offline_model", validate_gradients(gradient_report(groups)))

    def test_changed_parameter_identity_and_trainability_fail(self):
        gradients = self.update()
        self.groups["online_head"]["p"].requires_grad_(False)
        self.groups["online_pet"]["extra"] = nn.Parameter(torch.ones(1))
        result = audit_parameter_diff(self.before, self.groups, gradients)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("Parameter names changed: online_pet", result["errors"])

    def test_fixture_uses_only_logits_and_valid_synthetic_slots(self):
        logits = torch.zeros(4, 160)
        logits[:, 7] = 10
        self.assertEqual(pseudo_target_fixture(logits).tolist(), [7, 160, 7, 161])
        with self.assertRaises(ValueError):
            pseudo_target_fixture(logits[:1])
        with self.assertRaises(ValueError):
            pseudo_target_fixture(torch.zeros(2, 162))


@unittest.skipUnless(os.environ.get("DEAN_UPSTREAM_CHECKOUT"), "External checkout is opt-in")
class ExternalCPUOneStepTests(unittest.TestCase):
    def test_native_tiny_vit_pet_head_original_loss_exactly_one_cpu_step(self):
        checkout = Path(os.environ["DEAN_UPSTREAM_CHECKOUT"])
        sys.path.insert(0, str(checkout))
        try:
            vision = importlib.import_module("net.vision_transformer")
        finally:
            sys.path.pop(0)
        torch.manual_seed(1)
        class TinyViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = vision.VisionTransformer(embedding_size=8, img_size=[16], patch_size=8,
                                                        embed_dim=8, depth=12, num_heads=2, mlp_ratio=2.)
                self.backbone.embedding = nn.Linear(8, 8)
            def forward(self, values):
                return self.backbone(values)
        models = build_online_pair(TinyViT(), checkout, embedding_dim=8)
        optimizers = []
        real_adam = torch.optim.AdamW
        def counted_optimizer(*args, **kwargs):
            optimizer = real_adam(*args, **kwargs)
            optimizer.step = Mock(wraps=optimizer.step)
            optimizers.append(optimizer)
            return optimizer
        with patch("dean_integration.online_step_smoke.torch.optim.AdamW", side_effect=counted_optimizer):
            report = run_online_step(models, torch.randn(2, 3, 16, 16), checkout)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(report["execution_device"], "cpu")
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["epochs"], 0)
        self.assertEqual(report["backward_calls"], 1)
        self.assertEqual(report["forward_calls"], {"offline": 1, "online": 1})
        self.assertEqual(len(optimizers), 1)
        optimizers[0].step.assert_called_once()
        self.assertEqual(report["logit_shape"], [2, 162])
        self.assertEqual(report["parameter_audit"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
