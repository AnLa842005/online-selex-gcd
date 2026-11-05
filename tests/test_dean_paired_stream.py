"""CPU paired-protocol contracts, including real DataLoader and AST wiring."""
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

from dean_integration.config import ROOT
from dean_integration.paired_stream import (
    PairedOnlineStream, canonical_bytes, make_manifest, read_manifest,
    reset_online_rng, validate_manifest, validate_online_seed,
)
from dean_integration.preflight import sha256
from dean_integration.provenance import execute_run
from dean_integration.runtime import instrument_boundary, patch_experiment


class IdDataset(torch.utils.data.Dataset):
    def __init__(self, count=71, offset=100):
        self.uq_idxs = np.arange(offset, offset + 3 * count, 3)

    def __len__(self):
        return len(self.uq_idxs)

    def __getitem__(self, index):
        return int(self.uq_idxs[index])


def context(seed=17):
    return dict(online_seed=seed, seed=1, upstream_sha="pinned", config_sha256="official",
                dino_sha256="dino", base_checkpoint_sha256="base",
                dataset_metadata={"images": "metadata"}, arguments=["--seed", "1", "--base_epochs", "30"])


def batches(stream, dataset, session=0, workers=0):
    return [b.tolist() for b in stream.loader(dataset, session, 8, workers)]


class ManifestTests(unittest.TestCase):
    def test_generation_independent_of_global_rng_and_seed_changes_order(self):
        datasets = [IdDataset(), IdDataset(19, 500)]
        first = make_manifest(datasets, 8, context())
        for seed in (1, 42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.random(), np.random.rand(100), torch.rand(200)
            before = torch.get_rng_state().clone()
            self.assertEqual(canonical_bytes(first), canonical_bytes(make_manifest(datasets, 8, context())))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        changed = make_manifest(datasets, 8, context(18))
        self.assertNotEqual(first["sessions"][0]["batches"], changed["sessions"][0]["batches"])

    def test_repeated_generation_reload_same_bytes_batches_and_hashes(self):
        dataset = IdDataset()
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"stream{i}.json" for i in range(3)]
            streams = [PairedOnlineStream(context(), paths[0]), PairedOnlineStream(context(), paths[1]),
                       PairedOnlineStream(context(), paths[2], paths[0])]
            received = []
            for stream in streams:
                torch.rand(51), random.random(), np.random.rand(21)
                stream.prepare([dataset], 8)
                received.append(batches(stream, dataset))
            self.assertEqual(received, [received[0]] * 3)
            self.assertEqual(len({sha256(p) for p in paths}), 1)
            self.assertEqual(received[0], streams[0].record["sessions"][0]["batches"])
            self.assertEqual(len(received[0][-1]), 7)
            before = torch.get_rng_state().clone()
            batches(streams[0], dataset)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_worker_generators_preserve_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = IdDataset(19)
            stream = PairedOnlineStream(context(), Path(tmp) / "stream.json")
            stream.prepare([dataset], 8)
            self.assertEqual(batches(stream, dataset, workers=2), stream.record["sessions"][0]["batches"])

    def test_reset_python_numpy_cpu_cuda_at_boundary(self):
        with patch("torch.cuda.manual_seed_all") as cuda_seed:
            reset_online_rng(17)
            first = (random.random(), np.random.rand(), torch.rand(3))
            torch.rand(99), random.random(), np.random.rand(45)
            reset_online_rng(17)
            second = (random.random(), np.random.rand(), torch.rand(3))
            cuda_seed.assert_called_with(17)
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))
        for seed in (None, -1, 2**32, True):
            with self.assertRaises(ValueError):
                validate_online_seed(seed)

    def test_malformed_and_incompatible_manifests_fail(self):
        dataset = IdDataset()
        valid = make_manifest([dataset], 8, context())
        variants = [[], {}, {**valid, "version": 2}, {**valid, "version": True},
                    {**valid, "batch_size": 4}, {**valid, "sessions": []},
                    {**valid, "context": context(18)}]
        for mutation in ("duplicate", "unknown", "missing", "boundary", "session", "float", "dataset"):
            value = copy.deepcopy(valid)
            session = value["sessions"][0]
            if mutation == "duplicate":
                session["batches"][0][0] = session["batches"][0][1]
            elif mutation == "unknown":
                session["batches"][0][0] = -999
            elif mutation == "missing":
                session["batches"][-1].pop()
            elif mutation == "boundary":
                session["batches"][1].append(session["batches"][0].pop())
            elif mutation == "session":
                session["session_id"] = 1
            elif mutation == "float":
                session["batches"][0][0] = float(session["batches"][0][0])
            else:
                session["dataset_sample_ids"].reverse()
            variants.append(value)
        for value in variants:
            with self.subTest(value=str(value)[:70]), self.assertRaisesRegex(ValueError, "manifest"):
                validate_manifest(value, context(), [dataset], 8)
        for key in ("dataset_metadata", "base_checkpoint_sha256", "arguments"):
            changed = {**context(), key: "different"}
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_manifest(valid, changed, [dataset], 8)
        with self.assertRaisesRegex(ValueError, "dataset"):
            validate_manifest(valid, context(), [IdDataset(offset=200)], 8)

    def test_bad_input_never_generates_fallback_or_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input.json", Path(tmp) / "output.json"
            source.write_text("not json")
            stream = PairedOnlineStream(context(), output, source)
            with self.assertRaisesRegex(ValueError, "Cannot read stream manifest"):
                stream.prepare([IdDataset()], 8)
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                read_manifest(source.with_name("missing.json"), context())
            source.write_bytes(canonical_bytes(make_manifest([IdDataset()], 8, context())))
            before = source.read_bytes()
            with self.assertRaises(FileExistsError):
                PairedOnlineStream(context(), source, source).prepare([IdDataset()], 8)
            self.assertEqual(source.read_bytes(), before)
            stream.prepare([IdDataset()], 8)
            with self.assertRaisesRegex(ValueError, "once"):
                stream.prepare([IdDataset()], 8)
            with self.assertRaisesRegex(ValueError, "differs"):
                stream.loader(IdDataset(offset=200), 0, 8, 0)


class PairedRuntimeTests(unittest.TestCase):
    def fixture(self):
        return ast.parse("""
for epoch in range(0, args.nb_epochs):
    raise AssertionError('base stage must not run')
pth_pth = 'unused.pth'
checkpoint = torch.load(pth_pth)
for sess_idx, inc_tr_dataset in enumerate(dt_train_inc_unlabelled):
    dlod_tr_now = torch.utils.data.DataLoader(copy.deepcopy(inc_tr_dataset), batch_size=args.inc_sz_batch, shuffle=True, num_workers=args.nb_workers)
    for batch in dlod_tr_now:
        received.append(batch.tolist())
        loss = ce_loss + args.energy_hp * energy_loss
""")

    def test_control_treatment_identical_incoming_batches_despite_different_rng_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            received, paths = [], []
            for enabled in (False, True):
                tree = patch_experiment(self.fixture(), paired_online=True, use_selex_online=enabled)
                tree = instrument_boundary(tree, paired_online=True)
                path = Path(tmp) / f"arm{enabled}.json"
                stream = PairedOnlineStream(context(), path, paths[0] if paths else None)
                def treatment(loss, *args, **kwargs):
                    random.random(), np.random.rand(37), torch.rand(65)
                    return loss + 1
                load = Mock(return_value={})
                namespace = dict(args=SimpleNamespace(nb_epochs=30, inc_sz_batch=8, nb_workers=0, energy_hp=1.),
                                 copy=copy, torch=torch, dt_train_inc_unlabelled=[IdDataset()], received=[],
                                 _dean_base_path="same-base.pth", _dean_load_base=load, model=Mock(), header=Mock(),
                                 _dean_paired=stream, _dean_augment=treatment, _dean_selex_config=None,
                                 ce_loss=torch.tensor(2.), energy_loss=torch.tensor(1.),
                                 feats=None, logits=None, on_y=None, header_now=SimpleNamespace(head=SimpleNamespace(weight=None)))
                exec(compile(ast.fix_missing_locations(tree), "<paired-integration>", "exec"), namespace)
                load.assert_called_once_with("same-base.pth", namespace["model"], namespace["header"])
                received.append(namespace["received"])
                paths.append(path)
            self.assertEqual(received[0], received[1])
            self.assertEqual(sha256(paths[0]), sha256(paths[1]))

    def test_full_and_strict_do_not_change_stream_loader(self):
        original = self.fixture()
        self.assertEqual(ast.dump(original), ast.dump(patch_experiment(copy.deepcopy(original))))
        strict = patch_experiment(copy.deepcopy(original), online_only=True)
        strict = instrument_boundary(strict, online_only=True)
        text = ast.unparse(strict)
        self.assertIn("shuffle=True", text)
        self.assertIn("_dean_restore(_dean_boundary_rng)", text)
        self.assertNotIn("_dean_paired", text)
        for function in (patch_experiment, instrument_boundary):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                function(copy.deepcopy(original), online_only=True, paired_online=True)
        drift = ast.parse(ast.unparse(original).replace("shuffle=True", "shuffle=False"))
        with self.assertRaisesRegex(ValueError, "changed"):
            patch_experiment(drift, paired_online=True)

    @unittest.skipUnless(os.environ.get("DEAN_UPSTREAM_CHECKOUT"), "Pinned external source is opt-in")
    def test_pinned_upstream_paired_compiles_with_both_arms_and_stays_clean(self):
        from dean_integration.external import bound_entrypoint, verify_checkout
        checkout = verify_checkout(os.environ["DEAN_UPSTREAM_CHECKOUT"])
        path = checkout / "Ours_train_online.py"
        before = path.read_bytes()
        for enabled in (False, True):
            bound_entrypoint(checkout, ROOT, paired_online=True, use_selex_online=enabled, capture_boundary=True)
        self.assertEqual(before, path.read_bytes())
        verify_checkout(checkout)


class PairedCliTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location("paired_cli", ROOT / "scripts/run_dean.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_modes_are_explicit_and_strict_sidecar_is_still_required(self):
        module = self.module()
        variants = [
            ["--online-only", "--base-checkpoint", "base.pth", "--seed", "1"],
            ["--paired-online", "--base-checkpoint", "base.pth", "--seed", "1"],
            ["--paired-online", "--base-checkpoint", "base.pth", "--seed", "1", "--online_seed", "1", "--boundary-rng", "rng.json"],
            ["--online_seed", "1"], ["--stream-manifest", "stream.json"],
            ["--paired-online", "--online-only"],
        ]
        for flags in variants:
            with self.subTest(flags=flags), patch("sys.argv", ["run_dean", "--upstream", ".", *flags]), \
                 patch.object(module, "verify_checkout", return_value=ROOT), \
                 patch.object(module.subprocess, "run") as runner:
                with self.assertRaises(SystemExit):
                    module.main()
                runner.assert_not_called()

    def test_cli_paired_without_sidecar_propagates_flags_and_finalizes_manifest(self):
        module = self.module()
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dino, base = root / "dino.pth", root / "cub_VIT_best_step_0.pth"
                dino.write_bytes(b"dino")
                base.write_bytes(b"base")
                argv = ["run_dean", "--action", "train", "--upstream", tmp, "--dataset-parent", tmp,
                        "--checkpoint", str(dino), "--base-checkpoint", str(base), "--output", tmp,
                        "--seed", "1", "--paired-online", "--online_seed", "17"]
                if enabled:
                    argv.append("--use_selex_online")
                def worker(command, **kwargs):
                    self.assertIn("--paired-online", command)
                    self.assertIn("--online_seed", command)
                    self.assertNotIn("--boundary-rng", command)
                    self.assertEqual("--use_selex_online" in command, enabled)
                    directory = Path(command[command.index("--output") + 1])
                    parent = json.loads((directory.parent / "provenance.json").read_text())
                    PairedOnlineStream(parent["stream_context"], directory / "stream_manifest.json").prepare([IdDataset()], 64)
                    return SimpleNamespace(returncode=0)
                with patch("sys.argv", argv), patch.object(module, "verify_checkout", return_value=root), \
                     patch.object(module, "check_assets", return_value={"errors": []}), \
                     patch("dean_integration.base_checkpoint.audit_base", return_value={"status": "PASS"}) as audit, \
                     patch.object(module.subprocess, "run", side_effect=worker):
                    self.assertEqual(module.main(), 0)
                    audit.assert_called_once_with(base, root)
                report = json.loads((root / "experiments/cub_paired_v1/provenance.json").read_text())
                self.assertEqual(report["status"], "COMPLETE")
                self.assertEqual(report["online_mode"], "paired")
                self.assertEqual(report["online_seed"], 17)
                manifest = root / "experiments/cub_paired_v1/seed1/stream_manifest.json"
                self.assertEqual(report["stream_manifest_sha256"], sha256(manifest))
                self.assertEqual(report["executions"][0]["stream_manifest"]["online_seed"], 17)
                self.assertEqual(report["arguments"]["1"][report["arguments"]["1"].index("--base_epochs") + 1], "30")

    def test_cli_treatment_reuses_control_manifest_with_identical_official_arguments(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dino, base = root / "dino.pth", root / "base.pth"
            dino.write_bytes(b"dino")
            base.write_bytes(b"base")
            reports, received = [], []
            source = root / "control/experiments/cub_paired_v1/seed1/stream_manifest.json"
            for enabled in (False, True):
                output = root / ("treatment" if enabled else "control")
                argv = ["run_dean", "--action", "train", "--upstream", tmp, "--dataset-parent", tmp,
                        "--checkpoint", str(dino), "--base-checkpoint", str(base), "--output", str(output),
                        "--seed", "1", "--paired-online", "--online_seed", "17"]
                if enabled:
                    argv += ["--use_selex_online", "--stream-manifest", str(source)]
                def worker(command, **kwargs):
                    directory = Path(command[command.index("--output") + 1])
                    report = json.loads((directory.parent / "provenance.json").read_text())
                    source_path = Path(command[command.index("--stream-manifest") + 1]) if "--stream-manifest" in command else None
                    stream = PairedOnlineStream(report["stream_context"], directory / "stream_manifest.json", source_path)
                    dataset = IdDataset()
                    stream.prepare([dataset], 64)
                    received.append([b.tolist() for b in stream.loader(dataset, 0, 64, 0)])
                    return SimpleNamespace(returncode=0)
                with patch("sys.argv", argv), patch.object(module, "verify_checkout", return_value=root), \
                     patch.object(module, "check_assets", return_value={"errors": []}), \
                     patch("dean_integration.base_checkpoint.audit_base", return_value={"status": "PASS"}), \
                     patch.object(module.subprocess, "run", side_effect=worker):
                    self.assertEqual(module.main(), 0)
                reports.append(json.loads((output / "experiments/cub_paired_v1/provenance.json").read_text()))
            self.assertEqual(received[0], received[1])
            for key in ("stream_context", "arguments", "base_checkpoint_sha256", "online_seed", "stream_manifest_sha256"):
                self.assertEqual(reports[0][key], reports[1][key], key)
            self.assertEqual(reports[1]["input_stream_manifest_sha256"], reports[0]["stream_manifest_sha256"])

    def test_paired_success_without_manifest_fails_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "without a stream manifest"):
                execute_run(root, {"online_mode": "paired", "online_seed": 17}, [(1, root / "seed1", [])],
                            Mock(return_value=SimpleNamespace(returncode=0)))
            report = json.loads((root / "provenance.json").read_text())
            self.assertEqual(report["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
