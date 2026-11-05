"""Phase-1 contracts: immutable sample identity and fail-closed HSSK routing."""
from copy import deepcopy
from dataclasses import asdict, FrozenInstanceError, replace
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
import yaml

from dean_integration.config import ROOT
from dean_integration.experiment_config import load_selex_config
from dean_integration.selex_config import ReferenceConfig, load_reference_config
from dean_integration.selex_hierarchy import EMAHierarchy
from dean_integration.selex_hierarchy_provider import (
    DeanPrototypeHierarchyProvider, HierarchyBuildResult, HierarchyBuildStatus,
    HierarchyProviderKind, HierarchySnapshot, HierarchySupportPool,
    HSSKImplementationIncomplete, OnlineHSSKHierarchyProvider,
    SampleHierarchyAssignment, create_provider, digest, provider_metadata)
from dean_integration.selex_reference import expertise_losses


def small(**kw):
    return ReferenceConfig(output_dim=16, hidden_dim=12, bottleneck_dim=8, **kw)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.center = torch.randn(768, requires_grad=True)
        self.rows = [SampleHierarchyAssignment(101, -1, (-3,)), SampleHierarchyAssignment(8, -2, (-3,))]
        self.snapshot = HierarchySnapshot("hssk_online", self.rows,
            ({-1: self.center, -2: self.center}, {-3: self.center}),
            ({-1: .2, -2: .2}, {-3: .1}), source_batch=7,
            diagnostics={"nested": {"count": 2}})

    def test_sparse_permuted_and_two_view_lookup(self):
        self.assertEqual(self.snapshot.assignments([8, 101])[0].tolist(), [-2, -1])
        self.assertEqual(self.snapshot.assignments([101, 8, 101, 8])[0].tolist(), [-1, -2, -1, -2])
        with self.assertRaises(ValueError):
            self.snapshot.assignments([0])

    def test_detached_frozen_copy_in_copy_out_hash(self):
        hashed = self.snapshot.snapshot_hash
        with torch.no_grad():
            self.center.fill_(999)
        for _ in range(15):
            self.snapshot.centers[0][-1].fill_(4)
            self.snapshot.diagnostics["nested"]["count"] = 999
            state = self.snapshot.state_dict()
            state["payload"]["assignments"][0]["leaf_id"] = 8
            self.assertEqual(self.snapshot.snapshot_hash, hashed)
            self.assertFalse(self.snapshot.centers[0][-1].requires_grad)
            self.assertIsNone(self.snapshot.centers[0][-1].grad_fn)
            self.assertNotEqual(float(self.snapshot.centers[0][-1][0]), 999)
        with self.assertRaises(TypeError):
            self.snapshot.sample_to_leaf[101] = 12
        with self.assertRaises(FrozenInstanceError):
            self.snapshot.snapshot_hash = "bad"

    def test_snapshot_state_identity_schema_hash_validation(self):
        s = self.snapshot.state_dict()
        other = HierarchySnapshot.from_state_dict(s)
        self.assertEqual(other.snapshot_hash, self.snapshot.snapshot_hash)
        for field, value in (("schema_version", "bad"), ("provider", "other")):
            bad = deepcopy(s)
            bad["payload"][field] = value
            bad["snapshot_hash"] = digest(bad["payload"])
            with self.assertRaises(ValueError):
                HierarchySnapshot.from_state_dict(bad)
        s["snapshot_hash"] = "bad"
        with self.assertRaises(ValueError):
            HierarchySnapshot.from_state_dict(s)

    def test_skip_is_not_valid_hierarchy(self):
        result = HierarchyBuildResult(HierarchyProviderKind.HSSK_ONLINE, HierarchyBuildStatus.SKIP, reason="support")
        with self.assertRaises(ValueError):
            result.require_snapshot()
        with self.assertRaises(ValueError):
            HierarchyBuildResult(HierarchyProviderKind.HSSK_ONLINE, HierarchyBuildStatus.BUILD)
        with self.assertRaises(ValueError):
            HierarchyBuildResult(HierarchyProviderKind.HSSK_ONLINE, HierarchyBuildStatus.SKIP, self.snapshot, "support")
        with self.assertRaises(ValueError):
            HierarchySnapshot("hssk_online", [])

    def test_pool_detaches_and_preserves_explicit_identity(self):
        f = torch.randn(2, 768, requires_grad=True)
        pool = HierarchySupportPool([101, 8], f)
        captured = pool.current_features
        pool.current_features.zero_()
        self.assertTrue(torch.equal(captured, pool.current_features))
        self.assertFalse(pool.current_features.requires_grad)
        with self.assertRaises(ValueError):
            HierarchySupportPool([8, 8], f)

    def test_inconsistent_tree_is_a_hard_structural_error(self):
        rows = [SampleHierarchyAssignment(8, -1, (-2,)), SampleHierarchyAssignment(90, -1, (-3,))]
        with self.assertRaisesRegex(ValueError, "inconsistent ancestor"):
            HierarchySnapshot("hssk_online", rows)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_legacy_output_loss_gradient_counter_rng_parity(self):
        cfg = small()
        old, wrapped = EMAHierarchy(cfg), DeanPrototypeHierarchyProvider(cfg)
        for batch in range(3):
            torch.manual_seed(10 + batch)
            pool = HierarchySupportPool([90, 1, 8, 71], torch.randn(4, 768))
            labels = torch.tensor([0, 0, 160, 160])
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            direct = old.update(pool.current_features, labels, pool.sample_ids)
            result = wrapped.build(pool, labels, context={"batch": batch})
            self.assertEqual(before[0], random.getstate())
            self.assertTrue(np.array_equal(before[1][1], np.random.get_state()[1]))
            self.assertTrue(torch.equal(before[2], torch.get_rng_state()))
            self.assertEqual(old.counters, wrapped.ema.counters)
            self.assertEqual(old.counts, wrapped.ema.counts)
            self.assertEqual(old.next_parent, wrapped.ema.next_parent)
            for a, b in zip(direct.assignments(labels), result.snapshot.assignments(pool.sample_ids)):
                self.assertTrue(torch.equal(a, b))
            for a, b in zip(direct.centers, result.snapshot.centers):
                for k in a:
                    self.assertTrue(torch.equal(a[k], b[k]))
            q = torch.randn(4, 2, 16, requires_grad=True)
            geometry = torch.randn(4, 2, 768)
            a = expertise_losses(q, geometry, labels, direct, cfg)
            q2 = q.detach().clone().requires_grad_()
            b = expertise_losses(q2, geometry, pool.sample_ids, result.snapshot, cfg)
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), k)
            a["selex_total"].backward()
            b["selex_total"].backward()
            self.assertTrue(torch.equal(q.grad, q2.grad))

    def test_original_raw_legacy_delegation_preserves_dtype_and_counter(self):
        cfg = small()
        old, wrapped = EMAHierarchy(cfg), DeanPrototypeHierarchyProvider(cfg)
        f, y, ids = torch.randn(2, 768, dtype=torch.float64), [0, 0], [19, 20]
        a = old.update(f, y, ids)
        b = wrapped.build_legacy(f, y, ids)
        self.assertEqual(old.counters, wrapped.ema.counters)
        self.assertTrue(torch.equal(a.centers[0][0], b.centers[0][0]))

    def test_legacy_state_resume_and_wrong_namespace(self):
        provider = DeanPrototypeHierarchyProvider(small())
        pool = HierarchySupportPool([8, 90], torch.randn(2, 768))
        provider.build(pool, [0, 160])
        restored = DeanPrototypeHierarchyProvider(small())
        restored.load_state_dict(provider.state_dict())
        self.assertEqual(restored.state_dict(), provider.state_dict())
        a, b = provider.build(pool, [0, 160]), restored.build(pool, [0, 160])
        self.assertEqual(a.snapshot.snapshot_hash, b.snapshot.snapshot_hash)
        with self.assertRaises(ValueError):
            restored.load_state_dict(OnlineHSSKHierarchyProvider().state_dict())

    def test_primary_factory_and_no_prototype_fallback_or_input_reads(self):
        cfg = load_selex_config(implementation="reference_v1")
        self.assertEqual(cfg.hierarchy_source, "hssk_online")
        class Poison:
            def __getattribute__(self, name):
                raise AssertionError("HSSK scaffold read a DEAN input")
        with patch("dean_integration.selex_hierarchy_provider.DeanPrototypeHierarchyProvider", side_effect=AssertionError("fallback")):
            provider = create_provider(cfg)
            self.assertIsInstance(provider, OnlineHSSKHierarchyProvider)
            with self.assertRaisesRegex(HSSKImplementationIncomplete, "HSSK implementation incomplete"):
                provider.build(Poison(), Poison(), Poison(), Poison(), Poison())
        self.assertEqual(provider_metadata(cfg)["hierarchy_provider_status"], "scaffold/incomplete")

    def test_state_isolation_and_corruption(self):
        hssk = OnlineHSSKHierarchyProvider()
        before = hssk.state_dict()
        legacy = DeanPrototypeHierarchyProvider(small())
        legacy.build_legacy(torch.randn(2, 768), [0, 0], [8, 90])
        self.assertEqual(before, hssk.state_dict())
        hssk.load_state_dict(before)
        for field in ("provider", "schema"):
            s = deepcopy(before)
            s["payload"][field] = "bad"
            s["state_hash"] = digest(s["payload"])
            with self.assertRaises(ValueError):
                hssk.load_state_dict(s)
        with self.assertRaises(ValueError):
            hssk.load_state_dict(legacy.state_dict())

    def test_historical_adapter_is_hash_identified_and_new_config_strict(self):
        with self.assertLogs("dean_integration.selex_config", level="WARNING") as logs:
            cfg = load_reference_config()
        self.assertEqual(cfg.hierarchy_source, "dean_prototypes")
        self.assertIn("ablation", logs.output[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"new.yaml"
            for changes in ({"hierarchy_source": "unknown"}, {"replay_per_class": 5},
                            {"hssk_degenerate_policy": "skip"}, {"selex_labeled_replay_per_class": 5},
                            {"insufficient_support_policy": "skip"}):
                path.write_text(yaml.safe_dump({**asdict(small()), **changes}))
                with self.assertRaises(ValueError):
                    load_reference_config(path)
            value = asdict(small())
            del value["hierarchy_source"]
            path.write_text(yaml.safe_dump(value))
            with self.assertRaises(ValueError):
                load_reference_config(path)

    def test_hssk_cli_blocks_before_training_and_emits_incomplete_provenance(self):
        spec = importlib.util.spec_from_file_location("hssk_phase1_cli", ROOT/"scripts/run_dean.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for action in ("train", "worker", "verify", "smoke"):
            argv = ["run_dean", "--action", action, "--upstream", ".", "--use_selex_online",
                    "--selex-implementation", "reference_v1"]
            output = io.StringIO()
            with patch("sys.argv", argv), patch.object(cli, "verify_checkout", return_value=ROOT), \
                 patch.object(cli, "check_assets", side_effect=AssertionError("asset read")), \
                 patch.object(cli, "bound_entrypoint", side_effect=AssertionError("training")), redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    cli.main()
            self.assertEqual(raised.exception.code, 1)
            report = json.loads(output.getvalue())
            self.assertEqual(report["hierarchy_source"], "hssk_online")
            self.assertEqual(report["hierarchy_provider_status"], "scaffold/incomplete")
            self.assertEqual(report["status"], "BLOCKED")

    def test_incomplete_controller_fails_before_rng_or_projector(self):
        from dean_integration.selex_real_online import RealSelExOnline
        cfg = small(hierarchy_source="hssk_online")
        with patch("dean_integration.selex_real_online.SelExRNG", side_effect=AssertionError("RNG")), \
             patch("dean_integration.selex_real_online.create_provider", side_effect=AssertionError("provider")):
            with self.assertRaises(HSSKImplementationIncomplete):
                RealSelExOnline(cfg, 1)

    def test_disabled_and_gamma_zero_hssk_bypass_provider_images_rng_and_loss_hooks(self):
        spec = importlib.util.spec_from_file_location("hssk_bypass_cli", ROOT/"scripts/run_dean.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        body = """
import copy, random, numpy as np, torch
assert _dean_real is None
model = torch.nn.Linear(4, 2)
baseline = copy.deepcopy(model)
x = torch.ones(2, 4)
y = torch.tensor([0, 1])
state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
expected_output = baseline(x)
expected_loss = torch.nn.functional.cross_entropy(expected_output, y)
expected_loss.backward()
actual_output = model(x)
actual_loss = torch.nn.functional.cross_entropy(actual_output, y)
actual_loss.backward()
assert torch.equal(actual_output, expected_output)
assert torch.equal(actual_loss, expected_loss)
assert all(torch.equal(a.grad,b.grad) for a,b in zip(model.parameters(),baseline.parameters()))
assert state[0] == random.getstate()
assert np.array_equal(state[1][1],np.random.get_state()[1])
assert torch.equal(state[2],torch.get_rng_state())
"""
        for enabled, gamma in ((False, .1), (True, 0.)):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint = root/"dino.pth"
                checkpoint.write_bytes(b"fixture")
                cfg = root/"hssk.yaml"
                cfg.write_text(yaml.safe_dump(asdict(small(hierarchy_source="hssk_online", gamma=gamma))))
                argv = ["run_dean", "--action", "worker", "--upstream", tmp, "--dataset-parent", tmp,
                        "--checkpoint", str(checkpoint), "--output", tmp, "--seed", "1",
                        "--selex-implementation", "reference_v1", "--selex-config", str(cfg)]
                if enabled:
                    argv.append("--use_selex_online")
                with patch("sys.argv", argv), patch.object(cli, "verify_checkout", return_value=root), \
                     patch.object(cli, "check_assets", return_value={"errors": []}), \
                     patch.object(cli.os, "chdir"), \
                     patch.object(cli, "bound_entrypoint", return_value=compile(body, "<official-bypass>", "exec")) as bound, \
                     patch("dean_integration.selex_real_online.create_provider", side_effect=AssertionError("provider")), \
                     patch("dean_integration.selex_real_online.SelExRNG", side_effect=AssertionError("RNG")), \
                     patch("dean_integration.selex_real_online.CurrentBatchImages.images", side_effect=AssertionError("image read")):
                    self.assertEqual(cli.main(), 0)
                    self.assertFalse(bound.call_args.kwargs["reference_selex"])


if __name__ == "__main__":
    unittest.main()
