"""Small-dimension parity tests against original SelEx AST, and corrected guards."""
import ast
from collections import Counter
from dataclasses import replace
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from dean_integration.config import ROOT
from dean_integration.selex_config import ReferenceConfig, SOURCE_SHA, load_reference_config
from dean_integration.selex_hierarchy import EMAHierarchy, HierarchySnapshot
from dean_integration.selex_projector import SelExProjector
from dean_integration.selex_reference import (confusion_targets, expertise_losses, supervised_contrastive,
                                               unsupervised_expertise, view_major)


def small_config(**changes):
    return replace(ReferenceConfig(), output_dim=16, hidden_dim=12, bottleneck_dim=8, **changes)


def original_namespace():
    namespace = dict(torch=torch, nn=nn, F=F, trunc_normal_=nn.init.trunc_normal_, device=torch.device("cpu"))
    for path, names in (("methods/contrastive_training/contrastive_training.py", ("SupConLoss", "LabelSmoothingLoss", "info_nce_logits")),
                        ("models/vision_transformer.py", ("DINOHead",))):
        source = subprocess.check_output(["git", "show", SOURCE_SHA+":"+path], cwd=ROOT, text=True)
        tree = ast.parse(source)
        selected = [node for node in tree.body if getattr(node, "name", None) in names]
        exec(compile(ast.Module(body=selected, type_ignores=[]), "<pinned-SelEx-reference>", "exec"), namespace)
    return namespace


class ReferenceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = original_namespace()

    def setUp(self):
        self.config = small_config()

    def test_production_configuration_without_allocating_production_head(self):
        config = load_reference_config()
        self.assertEqual(config.output_dim, 65536)
        self.assertEqual((config.hidden_dim, config.bottleneck_dim), (2048, 256))
        self.assertEqual(config.gamma, 0.1)
        self.assertEqual(config.sse_weight, 0.35)
        for changes in (dict(gamma=-1), dict(gamma=float("nan")), dict(output_dim=0), dict(output_dim=True),
                        dict(smoothing=2), dict(projector_lr=0), dict(epsilon=0), dict(prototype_capacity=1.5)):
            with self.assertRaises(ValueError):
                ReferenceConfig(**changes)

    def test_projector_exact_reference_initialization_forward_backward(self):
        torch.manual_seed(7)
        ours = SelExProjector(self.config)
        torch.manual_seed(7)
        original = self.reference["DINOHead"](768, 16, nlayers=3, hidden_dim=12, bottleneck_dim=8)
        for key, tensor in ours.state_dict().items():
            self.assertTrue(torch.equal(tensor, original.state_dict()[key]), key)
        a = torch.randn(8, 768, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        actual = ours(a)
        expected = F.normalize(original(b), dim=-1, eps=self.config.epsilon)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
        self.assertFalse(ours.last_layer.weight_g.requires_grad)
        self.assertTrue(torch.allclose(actual.norm(dim=-1), torch.ones(8)))

    def test_supcon_full_and_prefix_scalar_gradient_parity(self):
        labels = torch.tensor([0, 0, 1, 1])
        for dimensions in (2, 8, 16):
            for scale in (0.3, 1.0):
                a = (torch.randn(4, 2, dimensions)*scale).requires_grad_(True)
                b = a.detach().clone().requires_grad_(True)
                actual = supervised_contrastive(a, labels, self.config)
                expected = self.reference["SupConLoss"]()(b, labels)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-6)

    def test_use_matches_reference_logits_soft_ce_on_nondegenerate_corrected_targets(self):
        # Corrected targets are supplied to the original kernel: its target
        # construction bugs are intentionally not copied into production.
        labels = torch.tensor([0, 0, 1, 1])
        geometry = torch.randn(4, 2, 768)
        bank = EMAHierarchy(self.config)
        snapshot = bank.update(geometry[:, 0], labels, torch.arange(4))
        for alpha in (0., 0.5, 1.):
            config = replace(self.config, smoothing=alpha)
            targets, valid = confusion_targets(geometry, labels, snapshot, config, Counter())
            self.assertTrue(valid.all())
            torch.testing.assert_close(targets.sum(1), torch.ones(8))
            self.assertTrue(torch.equal(targets.diag(), torch.zeros(8)))
            a = F.normalize(torch.randn(4, 2, 16), dim=-1).requires_grad_(True)
            b = a.detach().clone().requires_grad_(True)
            actual = unsupervised_expertise(a, geometry, labels, snapshot, config)
            logits, positive, similarity = self.reference["info_nce_logits"](
                view_major(b), targets, SimpleNamespace(n_views=2, temperature=1.))
            expected = self.reference["LabelSmoothingLoss"]()(logits, positive, similarity, alpha)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-6)

    def test_total_preserves_sse_half_and_reference_level_weights(self):
        config = self.config
        labels = torch.tensor([0, 0, 1, 1])
        geometry = torch.randn(4, 2, 768)
        snapshot = EMAHierarchy(config).update(geometry[:, 0], labels, torch.arange(4))
        q = F.normalize(torch.randn(4, 2, 16), dim=-1).requires_grad_()
        replay = F.normalize(torch.randn(4, 2, 16), dim=-1).requires_grad_()
        actual = expertise_losses(q, geometry, labels, snapshot, config, replay_projected=replay, replay_labels=labels)
        hierarchy = 0
        for i, assignments in enumerate(snapshot.assignments(labels)):
            _, compact = torch.unique(assignments, return_inverse=True)
            dim = 16//2**(i+1)
            hierarchy += self.reference["SupConLoss"]()(q[:, :, :dim], compact)/2**(i+1)
        gt = self.reference["SupConLoss"]()(replay, labels)
        torch.testing.assert_close(actual["sse"], (gt+hierarchy)/2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual["selex_total"], .65*actual["use"]+.35*actual["sse"])
        without = expertise_losses(q, geometry, labels, snapshot, config)
        self.assertEqual(float(without["sse_gt"]), 0)
        torch.testing.assert_close(actual["sse_hierarchy"], without["sse_hierarchy"])
        torch.testing.assert_close(actual["use"], without["use"])
        self.assertGreater(float(without["sse"]), 0)

    def test_empty_singleton_no_positive_terms_are_differentiable_zeros(self):
        for count, labels in ((0, []), (1, [0]), (3, [0, 1, 2])):
            q = torch.randn(count, 2, 8, requires_grad=True)
            counters = Counter()
            loss = supervised_contrastive(q, torch.tensor(labels, dtype=torch.long), self.config, counters)
            self.assertEqual(float(loss), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(q.grad).all())
            self.assertGreater(sum(counters.values()), 0)

    def test_skip_only_singleton_anchor_group_and_keep_valid_group(self):
        q = torch.randn(3, 2, 8, requires_grad=True)
        counters = Counter()
        loss = supervised_contrastive(q, torch.tensor([0, 0, 1]), self.config, counters)
        self.assertGreater(float(loss), 0)
        self.assertEqual(counters["sse.singleton_groups"], 1)
        self.assertEqual(counters["sse.no_positive_anchors"], 2)
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_zero_confusion_skips_only_use_and_preserves_sse(self):
        labels = torch.zeros(4, dtype=torch.long)
        geometry = torch.ones(4, 2, 768)
        snapshot = EMAHierarchy(self.config).update(geometry[:, 0], labels, torch.arange(4))
        q = torch.randn(4, 2, 16, requires_grad=True)
        counters = Counter()
        losses = expertise_losses(q, geometry, labels, snapshot, self.config, counters=counters)
        self.assertEqual(float(losses["use"]), 0)
        self.assertGreater(float(losses["sse"]), 0)
        self.assertEqual(counters["use.zero_target_rows"], 8)
        losses["selex_total"].backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_degenerate_empty_singleton_all_known_all_novel(self):
        for labels in ([], [0], [0, 0, 1, 1], [160, 160, 173, 173]):
            y = torch.tensor(labels, dtype=torch.long)
            z = torch.randn(len(y), 2, 768, requires_grad=True)
            q = torch.randn(len(y), 2, 16, requires_grad=True)
            snapshot = EMAHierarchy(self.config).update(z[:, 0], y, torch.arange(len(y)))
            losses = expertise_losses(q, z, y, snapshot, self.config)
            losses["selex_total"].backward()
            self.assertTrue(torch.isfinite(q.grad).all())
            self.assertIsNone(z.grad)  # Geometry/assignments are detached.
            self.assertTrue(all(torch.isfinite(v) for v in losses.values()))

    def test_structural_mismatches_are_not_silently_skipped(self):
        q = torch.randn(2, 2, 8)
        y = torch.tensor([0, 0])
        z = torch.randn(2, 2, 768)
        snapshot = EMAHierarchy(self.config).update(z[:, 0], y, torch.arange(2))
        calls = [lambda: supervised_contrastive(q[:, :1], y, self.config),
                 lambda: supervised_contrastive(q, y.float(), self.config),
                 lambda: supervised_contrastive(q, y[:1], self.config),
                 lambda: unsupervised_expertise(q, z[:, :, :2], y, snapshot, self.config),
                 lambda: unsupervised_expertise(q, z, torch.tensor([999, 999]), snapshot, self.config),
                 lambda: unsupervised_expertise(q, z, y, HierarchySnapshot((), (), (), ()), self.config),
                 lambda: expertise_losses(q, z, y, snapshot, self.config, replay_labels=y),
                 lambda: supervised_contrastive(torch.full_like(q, float("nan")), y, self.config)]
        for call in calls:
            with self.assertRaises(ValueError):
                call()

    def test_pairwise_axes_are_batch_axes_and_prefixes_alias_projection_storage(self):
        config = small_config()
        flat = torch.randn(8, 16, requires_grad=True)
        q = flat.reshape(2, 4, 16).transpose(0, 1)
        self.assertEqual(view_major(q).data_ptr(), flat.data_ptr())
        self.assertEqual(view_major(q[:, :, :8]).data_ptr(), flat.data_ptr())
        y = torch.tensor([0, 0, 1, 1])
        geometry = torch.randn(4, 2, 768)
        snapshot = EMAHierarchy(config).update(geometry[:, 0], y, torch.arange(4))
        cdist, eye = torch.cdist, torch.eye
        shapes = []
        def guarded_cdist(a, b, *args, **kwargs):
            self.assertEqual(a.shape[-2], 8)
            self.assertEqual(b.shape[-2], 8)
            out = cdist(a, b, *args, **kwargs)
            self.assertEqual(out.shape, (8, 8))
            shapes.append(tuple(out.shape))
            return out
        def guarded_eye(n, *args, **kwargs):
            self.assertEqual(n, 8)
            return eye(n, *args, **kwargs)
        with patch("torch.cdist", side_effect=guarded_cdist), patch("torch.eye", side_effect=guarded_eye):
            loss = expertise_losses(q, geometry, y, snapshot, config)["selex_total"]
            loss.backward()
        self.assertTrue(shapes)
        self.assertTrue(torch.isfinite(flat.grad).all())

    def test_prefix_halving_is_independent_of_weights_and_never_calls_zero_width(self):
        config = small_config(hierarchy_decay=.25)
        y = torch.arange(8)
        geometry = torch.randn(8, 2, 768)
        snapshot = EMAHierarchy(config).update(geometry[:, 0], y, torch.arange(8))
        import dean_integration.selex_reference as kernels
        original = kernels.supervised_contrastive
        for dimension in (1, 3, 16):
            flat = torch.randn(16, dimension, requires_grad=True)
            q = flat.reshape(2, 8, dimension).transpose(0, 1)
            widths = []
            def traced(features, *args, **kwargs):
                self.assertGreater(features.shape[-1], 0)
                self.assertEqual(features.data_ptr(), flat.data_ptr())
                widths.append(features.shape[-1])
                return original(features, *args, **kwargs)
            counters = Counter()
            with patch.object(kernels, "supervised_contrastive", side_effect=traced):
                loss = expertise_losses(q, geometry, y, snapshot, config, counters=counters)["selex_total"]
            expected = [dimension//2**(i+1) for i in range(len(snapshot.mappings)) if dimension//2**(i+1)]
            self.assertEqual(widths, expected)
            loss.backward()
            self.assertTrue(torch.isfinite(flat.grad).all())
            if dimension == 1:
                self.assertTrue(any("empty_dimensions" in k for k in counters))

    def test_low_precision_losses_use_fp32_accumulation_and_detached_768d_geometry(self):
        for dtype in (torch.float16, torch.bfloat16):
            y = torch.tensor([0, 0, 160, 160])
            geometry = torch.randn(4, 2, 768, dtype=dtype, requires_grad=True)
            bank = EMAHierarchy(self.config)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                snapshot = bank.update(geometry[:, 0], y, torch.arange(4))
            for center in bank.prototypes.values():
                self.assertEqual(center.shape, (768,))
                self.assertEqual(center.dtype, torch.float32)
                self.assertFalse(center.requires_grad)
            q = torch.randn(4, 2, 16, dtype=dtype, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                losses = expertise_losses(q, geometry, y, snapshot, self.config)
            self.assertEqual(losses["selex_total"].dtype, torch.float32)
            losses["selex_total"].backward()
            self.assertTrue(torch.isfinite(q.grad).all())
            self.assertIsNone(geometry.grad)

    def test_skipped_term_zero_cannot_overflow_on_finite_inputs(self):
        q = torch.full((2, 2, 8), 1e38, requires_grad=True)
        loss = supervised_contrastive(q, torch.tensor([0, 1]), self.config)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertTrue(torch.equal(q.grad, torch.zeros_like(q)))

    def test_config_rejects_unrepresentable_numerical_settings(self):
        for changes in (dict(epsilon=1e-60), dict(epsilon=2), dict(use_temperature=1e-60),
                        dict(sse_temperature=0), dict(base_temperature=-1), dict(gamma=1e100),
                        dict(hidden_dim=0), dict(bottleneck_dim=False), dict(hierarchy_decay=1)):
            with self.assertRaises(ValueError):
                ReferenceConfig(**changes)


if __name__ == "__main__":
    unittest.main()
