"""CPU contracts for the DEAN experiment wrapper; no training/data required."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dean_integration.base_checkpoint import capture_rng, load_base, read_boundary, restore_rng, save_boundary, validate_checkpoint
from dean_integration.config import ROOT, UPSTREAM_SHA
from dean_integration.experiment_config import load_selex_config
from dean_integration.provenance import execute_run
from dean_integration.runtime import instrument_boundary, patch_experiment
from dean_integration.selex_online import SelExConfig, augment_online_loss, build_hierarchy, sse_loss, use_loss


class ProvenanceTests(unittest.TestCase):
    def test_complete_failed_spawn_error_and_interrupt(self):
        for result in (0, 7, OSError("spawn failed"), KeyboardInterrupt()):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                report = {"upstream_sha": UPSTREAM_SHA, "arguments": {"1": ["--seed", "1"]}, "checkpoint_sha256": "abc"}
                runner = Mock(side_effect=result) if isinstance(result, BaseException) else Mock(return_value=SimpleNamespace(returncode=result))
                jobs = [(1, run / "seed1", ["fake"])]
                if isinstance(result, BaseException):
                    with self.assertRaises(type(result)):
                        execute_run(run, report, jobs, runner)
                else:
                    self.assertEqual(execute_run(run, report, jobs, runner), result)
                aggregate = json.loads((run / "provenance.json").read_text())
                seed = json.loads((run / "seed1/execution.json").read_text())
                self.assertEqual(aggregate["status"], "COMPLETE" if result == 0 else "FAILED")
                self.assertEqual(seed["status"], aggregate["status"])
                for record in (aggregate, seed):
                    self.assertIsInstance(record["returncode"], int)
                    self.assertIn("start_time", record)
                    self.assertIn("end_time", record)
                    self.assertGreaterEqual(record["duration_seconds"], 0)
                    self.assertTrue(record["output_paths"])
                self.assertEqual(seed["seed"], 1)

    def test_failed_seed_stops_sweep_and_hashes_generated_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            def runner(command, **kwargs):
                (run / "seed1/cub_VIT_best_step_0.pth").write_bytes(b"fixture")
                return SimpleNamespace(returncode=2)
            execute_run(run, {}, [(1, run / "seed1", []), (3, run / "seed3", [])], runner)
            self.assertFalse((run / "seed3").exists())
            record = json.loads((run / "seed1/execution.json").read_text())
            self.assertEqual(len(next(iter(record["checkpoint_hashes"].values()))), 64)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.model, self.header = nn.Linear(4, 3), nn.Linear(3, 2)
        self.state = {"model_state_dict": copy.deepcopy(self.model.state_dict()),
                      "header_state_dict": copy.deepcopy(self.header.state_dict())}

    def test_valid_and_incompatible_checkpoint(self):
        self.assertIs(validate_checkpoint(self.state, self.model, self.header), self.state)
        variants = []
        for group in self.state:
            for mutation in ("missing", "unexpected", "shape", "dtype", "nan", "inf", "not_tensor"):
                value = copy.deepcopy(self.state)
                if mutation == "missing":
                    del value[group]["weight"]
                elif mutation == "unexpected":
                    value[group]["extra"] = torch.ones(1)
                elif mutation == "shape":
                    value[group]["weight"] = torch.ones(1)
                elif mutation == "dtype":
                    value[group]["weight"] = value[group]["weight"].double()
                elif mutation == "not_tensor":
                    value[group]["weight"] = []
                else:
                    value[group]["weight"].flatten()[0] = float(mutation)
                variants.append(value)
        variants += [{"teacher": {}}, {**self.state, "extra": {}}]
        for value in variants:
            with self.assertRaises(ValueError):
                validate_checkpoint(value, self.model, self.header)

    def test_tensor_only_load_and_no_partial_mutation(self):
        before = copy.deepcopy(self.model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cub_VIT_best_step_0.pth"
            torch.save(self.state, path)
            load_base(path, self.model, self.header)
            self.state["header_state_dict"]["bias"] = torch.ones(99)
            torch.save(self.state, path)
            with self.assertRaises(ValueError):
                load_base(path, self.model, self.header)
            with self.assertRaises(FileNotFoundError):
                load_base(path.with_name("missing.pth"), self.model, self.header)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_online_only_command_keeps_official_epochs_and_forwards_options(self):
        spec = importlib.util.spec_from_file_location("dean_runner_fixture", ROOT / "scripts/run_dean.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = SimpleNamespace(config=ROOT / "configs/dean/cub_official_v0.yaml", dataset_parent=Path("data"),
                               checkpoint=Path("dino.pth"), base_checkpoint=Path("cub_VIT_best_step_0.pth"),
                               boundary_rng=Path("rng.json"), selex_config=None, online_only=True, use_selex_online=False)
        command = module.worker_command(args, Path("external"), Path("output"), 1)
        self.assertIn("--online-only", command)
        self.assertIn(str(args.base_checkpoint.resolve()), command)
        self.assertIn(str(args.boundary_rng.resolve()), command)
        self.assertNotIn("--use_selex_online", command)
        self.assertNotIn("--base_epochs", command)
        args.use_selex_online = True
        args.selex_config = ROOT / "configs/dean/selex_online_synthetic_v1.yaml"
        enabled = module.worker_command(args, Path("external"), Path("out"), 1)
        self.assertIn("--use_selex_online", enabled)
        self.assertIn(str(args.selex_config.resolve()), enabled)
        args.online_only, args.base_checkpoint, args.boundary_rng = False, None, None
        self.assertNotIn("--online-only", module.worker_command(args, Path("external"), Path("out"), 1))


class StreamTests(unittest.TestCase):
    def test_boundary_roundtrip_matches_full_stream_after_base_rng_consumption(self):
        # Synthetic full base phase consumes all global RNG families.
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.device_count", return_value=0):
            for seed in (1, 3):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.random(), np.random.rand(23), torch.rand(79)
                state = capture_rng()
                def online():
                    # Eval iterator and LoRA/head setup consume RNG before shuffle upstream.
                    list(torch.utils.data.DataLoader(torch.arange(10), batch_size=3))
                    nn.Linear(4, 3)
                    batches = list(torch.utils.data.DataLoader(torch.arange(71), shuffle=True, batch_size=8, num_workers=0))
                    return torch.cat(batches), random.random(), np.random.rand(), torch.rand(3)
                full = online()
                torch.rand(100), random.random(), np.random.rand(10)
                restore_rng(json.loads(json.dumps(state)))
                resumed = online()
                self.assertTrue(torch.equal(full[0], resumed[0]))
                self.assertEqual(full[1:3], resumed[1:3])
                self.assertTrue(torch.equal(full[3], resumed[3]))

    def test_sidecar_hash_context_missing_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch("torch.cuda.is_available", return_value=False):
            path, sidecar = Path(tmp) / "base.pth", Path(tmp) / "rng.json"
            path.write_bytes(b"checkpoint")
            context = {"seed": 1, "upstream_sha": UPSTREAM_SHA}
            with self.assertRaisesRegex(ValueError, "sidecar"):
                read_boundary(path, sidecar, context)
            save_boundary(path, sidecar, context)
            self.assertIn("torch", read_boundary(path, sidecar, context))
            with self.assertRaises(FileExistsError):
                save_boundary(path, sidecar, context)
            with self.assertRaisesRegex(ValueError, "seed/protocol/assets"):
                read_boundary(path, sidecar, {"seed": 3})
            path.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "hash/version"):
                read_boundary(path, sidecar, context)


class SelExTests(unittest.TestCase):
    def test_disabled_is_identical_without_evaluating_auxiliary_inputs(self):
        features = torch.randn(8, 4, requires_grad=True)
        weight = torch.randn(3, 4, requires_grad=True)
        logits = features @ weight.T
        loss = F.cross_entropy(logits, torch.arange(8) % 3) + 0.7 * logits.square().mean()
        expected = torch.autograd.grad(loss, (features, weight), retain_graph=True)
        rng = torch.get_rng_state().clone()
        actual = augment_online_loss(loss, None, None, None, None, enabled=False)
        self.assertIs(actual, loss)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for a, b in zip(expected, torch.autograd.grad(actual, (features, weight))):
            self.assertTrue(torch.equal(a, b))

    def test_use_sse_and_integrated_forward_backward_finite(self):
        for classes in (1, 3, 7):
            for objective in ("use", "sse", "combined"):
                features = torch.randn(8, 4, requires_grad=True)
                prototypes = torch.randn(classes, 4, requires_grad=True)
                logits = features @ prototypes.T
                pseudo = torch.arange(8) % classes
                parents = build_hierarchy(prototypes)
                self.assertTrue(torch.equal(parents, build_hierarchy(prototypes)))
                if objective == "use":
                    loss = use_loss(logits, pseudo, parents)
                elif objective == "sse":
                    loss = sse_loss(features, pseudo, prototypes, parents)
                else:
                    loss = augment_online_loss(F.cross_entropy(logits, pseudo), features, logits, pseudo, prototypes,
                                               enabled=True, config=SelExConfig())
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(features.grad).all())
                if prototypes.grad is not None:
                    self.assertTrue(torch.isfinite(prototypes.grad).all())

    def test_configuration_validation(self):
        self.assertEqual(load_selex_config(ROOT / "configs/dean/selex_online_synthetic_v1.yaml"), SelExConfig())
        for weight in (-1, float("nan"), float("inf"), "invalid", None, True):
            with self.assertRaises(ValueError):
                SelExConfig(use_weight=weight)

    def test_bad_config_shape_and_invalid_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selex.yaml"
            for text in ("[]", "use_weight: 0.1", "use_weight: invalid\nsse_weight: 0.1"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_selex_config(path)
        for prototypes in (torch.empty(0, 4), torch.ones(4), torch.full((3, 4), float("nan"))):
            with self.assertRaises(ValueError):
                build_hierarchy(prototypes)


class RuntimeTests(unittest.TestCase):
    def fixture(self):
        return ast.parse("""
for epoch in range(0, args.nb_epochs):
    base_calls.append(epoch)
pth_pth = 'generated.pth'
checkpoint = torch.load(pth_pth)
model.load_state_dict(checkpoint['model_state_dict'])
header.load_state_dict(checkpoint['header_state_dict'])
loss = ce_loss + args.energy_hp * energy_loss
""")

    def test_online_skips_base_uses_strict_loader_and_retains_official_loss(self):
        tree = patch_experiment(self.fixture(), online_only=True)
        tree = instrument_boundary(tree, online_only=True)
        text = ast.unparse(tree)
        self.assertIn("range(0):", text)
        self.assertIn("_dean_load_base(pth_pth, model, header)", text)
        self.assertIn("loss = ce_loss + args.energy_hp * energy_loss", text)
        self.assertNotIn("_dean_augment", text)
        events = []
        namespace = dict(args=SimpleNamespace(nb_epochs=30, energy_hp=1.), base_calls=[],
                         _dean_base_path="completed.pth", _dean_boundary_rng={},
                         _dean_restore=lambda rng: events.append("restore"),
                         _dean_load_base=lambda path, model, head: (events.append(path) or {"model_state_dict": {}, "header_state_dict": {}}),
                         model=Mock(), header=Mock(), ce_loss=torch.tensor(2.), energy_loss=torch.tensor(3.))
        exec(compile(ast.fix_missing_locations(tree), "<fixture>", "exec"), namespace)
        self.assertEqual(namespace["base_calls"], [])
        self.assertEqual(events, ["restore", "completed.pth"])
        self.assertEqual(namespace["loss"].item(), 5.)

    def test_selex_patches_only_online_loss_and_rejects_source_drift(self):
        original = self.fixture()
        same = patch_experiment(copy.deepcopy(original))
        self.assertEqual(ast.dump(original), ast.dump(same))
        changed = patch_experiment(copy.deepcopy(original), use_selex_online=True)
        self.assertIn("enabled=True", ast.unparse(changed))
        with self.assertRaises(ValueError):
            patch_experiment(ast.parse("pass"), online_only=True)
        duplicated = self.fixture()
        duplicated.body.append(copy.deepcopy(duplicated.body[-1]))
        with self.assertRaises(ValueError):
            patch_experiment(duplicated, use_selex_online=True)

    def test_boundary_capture_does_not_advance_rng(self):
        with tempfile.TemporaryDirectory() as tmp, patch("torch.cuda.is_available", return_value=False):
            checkpoint = Path(tmp) / "base.pth"
            checkpoint.write_bytes(b"synthetic")
            before = json.dumps(capture_rng())
            save_boundary(checkpoint, Path(tmp) / "rng.json", {"seed": 1})
            self.assertEqual(before, json.dumps(capture_rng()))

    @unittest.skipUnless(os.environ.get("DEAN_UPSTREAM_CHECKOUT"), "Pinned external source is opt-in")
    def test_pinned_ast_hooks_and_exact_disabled_loss_outputs_gradients(self):
        from dean_integration.external import bound_entrypoint, verify_checkout
        checkout = verify_checkout(os.environ["DEAN_UPSTREAM_CHECKOUT"])
        path = checkout / "Ours_train_online.py"
        before = path.read_bytes()
        original = ast.parse(before.decode("utf-8"))
        for online in (False, True):
            for enabled in (False, True):
                bound_entrypoint(checkout, ROOT, online_only=online, use_selex_online=enabled, capture_boundary=True)
        for enabled in (False, True):
            tree = patch_experiment(copy.deepcopy(original), online_only=True, use_selex_online=enabled)
            expressions = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                           any(isinstance(t, ast.Name) and t.id == "loss" for t in n.targets)]
            online_loss = expressions[-1]
            if not enabled:
                self.assertEqual(ast.unparse(online_loss), "loss = ce_loss + args.energy_hp * energy_loss")
                feats = torch.randn(8, 4, requires_grad=True)
                weight = torch.randn(3, 4, requires_grad=True)
                logits = feats @ weight.T
                ce = F.cross_entropy(logits, torch.arange(8) % 3)
                energy = logits.square().mean()
                expected = ce + 1. * energy
                gradients = torch.autograd.grad(expected, (feats, weight), retain_graph=True)
                namespace = dict(ce_loss=ce, energy_loss=energy, args=SimpleNamespace(energy_hp=1.))
                exec(compile(ast.Module(body=[online_loss], type_ignores=[]), "<pinned-loss>", "exec"), namespace)
                self.assertTrue(torch.equal(expected, namespace["loss"]))
                for a, b in zip(gradients, torch.autograd.grad(namespace["loss"], (feats, weight))):
                    self.assertTrue(torch.equal(a, b))
        self.assertEqual(before, path.read_bytes())


class RunnerTests(unittest.TestCase):
    def runner_module(self):
        spec = importlib.util.spec_from_file_location("dean_cli_fixture", ROOT / "scripts/run_dean.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_cli_train_finalizes_and_refuses_existing_artifacts(self):
        module = self.runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "dino.pth"
            checkpoint.write_bytes(b"fixture")
            argv = ["run_dean", "--action", "train", "--upstream", tmp, "--dataset-parent", tmp,
                    "--checkpoint", str(checkpoint), "--output", tmp, "--seed", "1"]
            with patch("sys.argv", argv), patch.object(module, "verify_checkout", return_value=Path(tmp)), \
                 patch.object(module, "check_assets", return_value={"errors": []}), \
                 patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as process:
                self.assertEqual(module.main(), 0)
                report_path = Path(tmp) / "official/cub_v0/provenance.json"
                report = json.loads(report_path.read_text())
                self.assertEqual(report["status"], "COMPLETE")
                self.assertFalse(report["use_selex_online"])
                before = report_path.read_bytes()
                with self.assertRaises(SystemExit):
                    module.main()
                self.assertEqual(report_path.read_bytes(), before)
                self.assertEqual(process.call_count, 1)

    def test_online_cli_never_retrains_when_sidecar_missing(self):
        module = self.runner_module()
        with patch("sys.argv", ["run_dean", "--upstream", ".", "--online-only", "--seed", "1"]), \
             patch.object(module, "verify_checkout", return_value=ROOT), \
             patch.object(module.subprocess, "run") as process:
            with self.assertRaises(SystemExit):
                module.main()
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
