"""Phase-3 tree, verified supervision, frozen live loss and bounded smoke gates."""
import ast
from copy import deepcopy
from dataclasses import replace
import inspect
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dean_integration.selex_config import ReferenceConfig
from dean_integration.selex_hierarchy_provider import (OnlineHSSKHierarchyProvider, HierarchySupportPool,
    HierarchyBuildStatus, FullTrainingBlocked, ensure_provider_ready, digest)
from dean_integration.selex_hssk_online import LeafFitResult
from dean_integration.selex_hssk_hierarchy import coarsen_children
from dean_integration.selex_real_online import RealSelExOnline, CurrentBatchImages, LabeledReplay
from dean_integration.selex_reference import expertise_losses
from dean_integration.selex_replay_manifest import reconstruct, verified_manifest, write_manifest
from dean_integration.selex_smoke_gate import SmokeBudget, BoundedSmokeComplete
from dean_integration.runtime import patch_experiment, instrument_boundary, patch_bounded_smoke
from tests.test_dean_hssk_online import basis, replay_features


def cfg(**kw):
    return ReferenceConfig(hierarchy_source="hssk_online", output_dim=16, hidden_dim=12, bottleneck_dim=8, **kw)


def context(batch=0, **kw):
    return dict(batch=batch, seed=1, model_lineage="fixture-model", dataset="fixture-dataset",
                feature_version="dino768-eval-v1", seed_protocol="private-v1", **kw)


def support(offset=0):
    return HierarchySupportPool(range(offset, offset+16), torch.repeat_interleave(basis(8), 2, dim=0))


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(2, 768)*.01)
        self.lora_B = nn.Parameter(torch.zeros(768, 2))

    def forward(self, x):
        return x + F.linear(F.linear(x, self.lora_A), self.lora_B)


def engine_fixture(replay=True, output=None, dimensions=16):
    torch.manual_seed(7)
    model = Student()
    config = replace(cfg(), output_dim=dimensions)
    engine = RealSelExOnline(config, 17, output, bounded_smoke=True, context={k: v for k, v in context().items() if k not in ("batch", "seed")})
    def augment(x):
        return x+torch.rand_like(x)*.003
    if replay:
        _, manager, _ = replay_features(1)
        engine.replay_manager = manager
        engine.replay_image_factory = lambda ids: CurrentBatchImages(ids, lambda k: basis(1)[0], augment, lambda x: x)
    images = CurrentBatchImages([8, 19, 77, 90], lambda k: basis(1, 2 if k in (8, 19) else 3)[0], augment, lambda x: x)
    engine.start_session(model)
    return engine, model, images


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_higher_signature_and_inputs_only_child_centers(self):
        self.assertEqual(set(inspect.signature(coarsen_children).parameters), {"centers", "support_counts", "anchor_groups", "seed", "level"})
        with patch("dean_integration.selex_hssk_provider.coarsen_children", wraps=coarsen_children) as spy:
            result = OnlineHSSKHierarchyProvider(cfg()).build(support(), context=context())
        self.assertEqual(len(result.snapshot.mappings), 4)
        for call in spy.call_args_list:
            self.assertLessEqual(len(call.kwargs["centers"]), 8)
            self.assertTrue(all(v.shape == (768,) and not v.requires_grad for v in call.kwargs["centers"].values()))

    def test_nested_ancestors_negative_unique_ids_and_support(self):
        snap = OnlineHSSKHierarchyProvider(cfg()).build(support(), context=context()).snapshot
        previous, used = None, set()
        for level, mapping in enumerate(snap.mappings):
            group_ids = set(snap.centers[level])
            self.assertTrue(all(k < 0 for k in group_ids))
            self.assertFalse(group_ids & used)
            used |= group_ids
            if previous is not None:
                for child in set(previous.values()):
                    self.assertEqual(len({mapping[k] for k in previous if previous[k] == child}), 1)
                for parent in set(mapping.values()):
                    children = {previous[k] for k in mapping if mapping[k] == parent}
                    self.assertGreaterEqual(len(children), 2)
            self.assertTrue(all(v >= 2 for v in snap.diagnostics["support_counts"][level].values()))
            previous = mapping

    def test_positive_prefix_width_and_strict_reduction(self):
        for invalid in (1, "bad", None, True):
            with self.assertRaises(ValueError):
                replace(cfg(), output_dim=invalid)
        p = OnlineHSSKHierarchyProvider(replace(cfg(), output_dim=8))
        snap = p.build(support(), context=context()).snapshot
        self.assertEqual(len(snap.mappings), 3)
        sizes = [len(c) for c in snap.centers]
        self.assertTrue(all(b < a for a, b in zip(sizes, sizes[1:])))
        self.assertTrue(all(8//2**(i+1) > 0 for i in range(len(sizes))))

    def test_stop_when_free_child_support_cannot_halve(self):
        centers = dict(zip((-1, -2, -3), basis(3)))
        self.assertIsNone(coarsen_children(centers=centers, support_counts={k: 2 for k in centers},
                                         anchor_groups={k: () for k in centers}, seed=1, level=1))

    def test_dean_poison_not_read_and_prototype_never_called(self):
        class Poison:
            def __getattribute__(self, name):
                raise AssertionError("DEAN data read")
        with patch("dean_integration.selex_hierarchy_provider.DeanPrototypeHierarchyProvider", side_effect=AssertionError("fallback")):
            a = OnlineHSSKHierarchyProvider(cfg()).build(support(), Poison(), context=context())
            b = OnlineHSSKHierarchyProvider(cfg()).build(support(), torch.randn(100, 768), context=context())
        self.assertEqual(a.snapshot.snapshot_hash, b.snapshot.snapshot_hash)

    def test_parent_ids_monotonic_memory_updates_once_and_state_roundtrip(self):
        p = OnlineHSSKHierarchyProvider(cfg())
        first = p.build(support(), context=context()).snapshot
        before = p.memory.state_dict()
        with self.assertRaises(ValueError):
            p.build(support(), context=context())
        self.assertEqual(before, p.memory.state_dict())
        restored = OnlineHSSKHierarchyProvider(cfg())
        restored.load_state_dict(p.state_dict())
        a = p.build(support(), context=context(1)).snapshot
        b = restored.build(support(), context=context(1)).snapshot
        self.assertEqual(a.snapshot_hash, b.snapshot_hash)
        self.assertLess(max(a.centers[1]), min(first.centers[-1]))
        self.assertEqual(p.memory.state_dict(), restored.memory.state_dict())
        bad = p.state_dict(); bad["payload"]["last_status"] = "UNKNOWN"
        bad["state_hash"] = digest(bad["payload"])
        with self.assertRaises(ValueError):
            restored.load_state_dict(bad)

    def _force_failure(self):
        return patch("dean_integration.selex_hssk_provider.HSSKLeafFitter.fit",
                     return_value=LeafFitResult(dict(status="INSUFFICIENT_SUPPORT")))

    def test_reuse_preserves_tree_geometry_memory_and_allocator(self):
        p = OnlineHSSKHierarchyProvider(cfg())
        original = p.build(support(), context=context()).snapshot
        before = p.memory.state_dict()
        with self._force_failure():
            result = p.build(support(100), context=context(1))
        self.assertEqual(result.status, HierarchyBuildStatus.REUSE)
        self.assertEqual(p.memory.state_dict(), before)
        self.assertEqual(result.snapshot.source_batch, 0)
        self.assertEqual(set(result.snapshot.sample_to_leaf), set(range(100, 116)))
        for a, b in zip(original.centers, result.snapshot.centers):
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]))
        self.assertIn("reuse_hash", result.snapshot.diagnostics)

    def test_reuse_cannot_chain(self):
        p = OnlineHSSKHierarchyProvider(cfg())
        p.build(support(), context=context())
        with self._force_failure():
            self.assertEqual(p.build(support(100), context=context(1)).status, HierarchyBuildStatus.REUSE)
            self.assertEqual(p.build(support(200), context=context(2)).status, HierarchyBuildStatus.SKIP)

    def test_reuse_distance_support_and_identity_rejected_with_correct_policy(self):
        for policy in ("reuse_then_skip", "reuse_then_error"):
            for changed in ("distance", "identity", "singleton"):
                p = OnlineHSSKHierarchyProvider(cfg(insufficient_support_policy=policy))
                p.build(support(), context=context())
                pool, ctx = support(100), context(1)
                if changed == "distance":
                    pool = HierarchySupportPool(range(100, 116), torch.repeat_interleave(basis(8, 20), 2, dim=0))
                elif changed == "identity":
                    ctx["model_lineage"] = "other-model"
                else:
                    pool = HierarchySupportPool([100], basis(1))
                before = p.memory.state_dict()
                with self._force_failure():
                    if policy == "reuse_then_error":
                        with self.assertRaisesRegex(ValueError, "reuse rejected"):
                            p.build(pool, context=ctx)
                    else:
                        result = p.build(pool, context=ctx)
                        self.assertEqual(result.status, HierarchyBuildStatus.SKIP)
                        self.assertIsNone(result.snapshot)
                self.assertEqual(before, p.memory.state_dict())


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_single_build_frozen_15_epochs_fresh_views_and_finite_gradients(self):
        engine, model, images = engine_fixture()
        engine.prepare(model, images, object(), object())  # DEAN predictions never inspected.
        hashed, memory = engine.snapshot.snapshot_hash, engine.provider.memory.state_hash
        head = nn.Linear(768, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.add_param_group(dict(params=head.parameters()))
        engine.attach_optimizer(opt, head)
        for step in range(15):
            ce = F.cross_entropy(head(model(torch.stack([images.load_image(k) for k in images.sample_ids.tolist()]))), torch.tensor([0, 0, 1, 1]))
            loss = engine.augment(ce+ce*.01, ce, ce*.01, model, object(), images.sample_ids)
            opt.zero_grad(); loss.backward(); opt.step()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(engine.snapshot.snapshot_hash, hashed)
            self.assertEqual(engine.provider.memory.state_hash, memory)
        self.assertEqual(engine.provider.counters["hssk.build"], 1)
        self.assertGreater(float(model.lora_B.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in engine.projector.parameters() if p.grad is not None))
        self.assertEqual(opt.param_groups[-2]["name"], "selex_projector")
        with self.assertRaises(ValueError):
            engine.prepare(model, images, None, 2)

    def test_formula_verified_gt_only_and_view_identity(self):
        engine, model, images = engine_fixture()
        engine.prepare(model, images, None, 2)
        captured = {}
        def checked(q, geom, ids, snapshot, config, **kw):
            self.assertEqual(ids.tolist(), images.sample_ids.tolist())
            self.assertEqual(kw["replay_labels"].tolist(), [11, 11])
            self.assertEqual(q.shape, (4, 2, 16))
            self.assertEqual(snapshot.assignments(ids.repeat(2))[0].tolist(), snapshot.assignments(ids)[0].tolist()*2)
            actual = expertise_losses(q, geom, ids, snapshot, config, **kw)
            torch.testing.assert_close(actual["sse"], (actual["sse_gt"]+actual["sse_hierarchy"])/2)
            torch.testing.assert_close(actual["selex_total"], (1-config.sse_weight)*actual["use"]+config.sse_weight*actual["sse"])
            captured.update(actual)
            return actual
        ce = model.lora_B.sum()+2
        with patch("dean_integration.selex_hssk_runtime.expertise_losses", side_effect=checked):
            total = engine.augment(ce+.5, ce, ce*0+.5, model, torch.tensor([-999]), images.sample_ids)
        torch.testing.assert_close(total, ce+.5+engine.config.gamma*captured["selex_total"])
        with self.assertRaises(ValueError):
            engine.augment(ce, ce, ce*0, model, None, images.sample_ids, LabeledReplay(images, torch.zeros(4, dtype=torch.long)))

    def test_missing_replay_omits_only_gt_and_domains_do_not_consume_global_rng(self):
        engine, model, images = engine_fixture(False)
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        engine.prepare(model, images, None, 2)
        ce = model.lora_B.sum()+1
        total = engine.augment(ce, ce, ce*0, model, None, images.sample_ids)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(random.getstate(), state[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], state[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), state[2]))
        self.assertEqual(engine.counters["sse.gt.unavailable.no_verified_selection"], 1)
        current = engine.rng.cpu.clone()
        with engine.replay_rng.scope():
            torch.rand(100)
        self.assertTrue(torch.equal(current, engine.rng.cpu))

    def test_eval_targets_independent_of_current_augmentation_rng(self):
        a, model_a, images_a = engine_fixture(False)
        b, model_b, images_b = engine_fixture(False)
        with a.rng.scope():
            torch.rand(1000)
        a.prepare(model_a, images_a, None, 2)
        b.prepare(model_b, images_b, torch.tensor([999]), 999)
        self.assertEqual(a.snapshot.snapshot_hash, b.snapshot.snapshot_hash)
        self.assertTrue(model_a.training)
        self.assertTrue(all(not t.requires_grad for level in a.snapshot.centers for t in level.values()))

    def test_skip_is_whole_auxiliary_zero_without_augmentation(self):
        engine, model, _ = engine_fixture(False)
        images = CurrentBatchImages([8], lambda k: basis(1)[0], lambda x: (_ for _ in ()).throw(AssertionError("augmentation")), lambda x: x)
        engine.prepare(model, images, None, 1)
        self.assertEqual(engine.build_result.status, HierarchyBuildStatus.SKIP)
        ce = model.lora_B.sum()+1
        total = engine.augment(ce, ce, ce*0, model, None, images.sample_ids)
        self.assertTrue(torch.equal(total, ce))
        total.backward()
        self.assertTrue(torch.isfinite(model.lora_B.grad).all())

    def test_optimizer_state_survives_classifier_expansion(self):
        engine, model, images = engine_fixture(False)
        engine.prepare(model, images, None, 2)
        head = nn.Linear(768, 2)
        opt = torch.optim.AdamW(model.parameters())
        opt.add_param_group(dict(params=head.parameters()))
        engine.attach_optimizer(opt, head)
        ce = model.lora_B.sum()+1
        engine.augment(ce, ce, ce*0, model, None, images.sample_ids).backward(); opt.step()
        state = {p: {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in opt.state[p].items()}
                 for p in engine.projector.parameters() if p in opt.state}
        old = opt.param_groups.pop()
        for p in old["params"]: opt.state.pop(p, None)
        head = nn.Linear(768, 3)
        opt.add_param_group(dict(params=head.parameters()))
        engine.attach_optimizer(opt, head)
        for p in state:
            for k, v in state[p].items():
                self.assertTrue(torch.equal(v, opt.state[p][k]))


def metadata_fixture(root):
    images, labels, splits = [], [], []
    for i in range(1, 25):
        images.append(f"{i} class{(i-1)//6}/{i}.jpg\n")
        labels.append(f"{i} {(i-1)//6+1}\n")
        splits.append(f"{i} {int(i%6 != 0)}\n")
    for name, text in zip(("images.txt", "image_class_labels.txt", "train_test_split.txt"), (images, labels, splits)):
        (root/name).write_text("".join(text))


class ManifestAndGateTests(unittest.TestCase):
    def test_aggregate_provenance_rejects_missing_smoke_completion_or_snapshot(self):
        from types import SimpleNamespace
        from dean_integration.provenance import execute_run
        valid = dict(implementation="reference_v1", step=1, optimizer_groups=[dict(name="selex_projector")],
                     last_event="bounded_smoke_complete", full_train_blocked=True,
                     hierarchy_state=dict(status="BUILD", memory_hash="m", snapshot_hash="s"))
        for field in (None, "last_event", "hierarchy_state"):
            with tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp); child = run/"seed1"
                summary = deepcopy(valid)
                if field is not None: summary.pop(field)
                def runner(command, **kw):
                    (child/"selex_reference_summary.json").write_text(json.dumps(summary))
                    return SimpleNamespace(returncode=0)
                report = dict(real_selex_active=True, hierarchy_source="hssk_online", bounded_smoke=dict(optimizer_updates=1))
                if field is None:
                    self.assertEqual(execute_run(run, report, [(1, child, ["fixture"])], runner), 0)
                    self.assertEqual(report["status"], "COMPLETE")
                else:
                    with self.assertRaises(ValueError): execute_run(run, report, [(1, child, ["fixture"])], runner)
                    self.assertEqual(report["status"], "FAILED")

    def test_bounded_worker_fixture_wires_verified_replay_and_final_provenance(self):
        from dean_integration.config import ROOT
        spec = importlib.util.spec_from_file_location("phase3_smoke_cli", ROOT/"scripts/run_dean.py")
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        body = """
import numpy as np, torch
from pathlib import Path
from types import SimpleNamespace
from tests.test_dean_hssk_phase3 import Student
from tests.test_dean_hssk_online import basis
from dean_integration.selex_real_online import CurrentBatchImages
model=Student(); head=torch.nn.Linear(768,2)
_dean_real.start_session(model)
root=Path(_dean_real.output).parent/'CUB_200_2011'
labelled=SimpleNamespace(root=root,base_folder='images',uq_idxs=np.array(sorted(_dean_real.replay_manager.eligibility._records)),
    loader=lambda path:basis(1,int(Path(path).parent.name[5:]))[0])
_dean_real.configure_replay_from_cub(labelled,lambda is_train:(lambda x:x+torch.rand_like(x)*.001) if is_train else (lambda x:x))
images=CurrentBatchImages([100,101,102,103],lambda key:basis(1,32+key%2)[0],lambda x:x+torch.rand_like(x)*.001,lambda x:x)
_dean_real.prepare(model,images,None,2)
opt=torch.optim.AdamW(model.parameters());opt.add_param_group(dict(params=head.parameters()));_dean_real.attach_optimizer(opt,head)
for i in range(100):
    ce=torch.nn.functional.cross_entropy(head(model(torch.stack([images.load_image(k) for k in images.sample_ids.tolist()]))),torch.tensor([0,1,0,1]))
    loss=_dean_real.augment(ce,ce,ce*0,model,None,images.sample_ids)
    opt.zero_grad();loss.backward();opt.step();_dean_smoke.step_completed()
raise AssertionError('unbounded worker')
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dataset = root/"CUB_200_2011"; dataset.mkdir(); metadata_fixture(dataset)
            output = root/"out"; output.mkdir()
            dino, base, config = root/"dino.pth", root/"base.pth", root/"hssk.yaml"
            dino.write_bytes(b"fixture"); base.write_bytes(b"fixture")
            import yaml
            from dataclasses import asdict
            config.write_text(yaml.safe_dump(asdict(cfg())))
            argv = ["run_dean", "--action", "smoke-worker", "--upstream", tmp, "--dataset-parent", tmp,
                    "--checkpoint", str(dino), "--base-checkpoint", str(base), "--output", str(output),
                    "--paired-online", "--seed", "1", "--online_seed", "17", "--use_selex_online",
                    "--selex-implementation", "reference_v1", "--selex-config", str(config), "--smoke-updates", "1"]
            with patch("sys.argv", argv), patch.object(cli, "verify_checkout", return_value=root), \
                 patch.object(cli, "check_assets", return_value={"errors": []}), \
                 patch("dean_integration.base_checkpoint.audit_base", return_value={}), patch.object(cli.os, "chdir"), \
                 patch.object(cli, "bound_entrypoint", return_value=compile(body, "<bounded-smoke-fixture>", "exec")) as bound:
                self.assertEqual(cli.main(), 0)
                self.assertTrue(bound.call_args.kwargs["bounded_smoke"])
            summary = json.loads((output/"selex_reference_summary.json").read_text())
            self.assertEqual(summary["last_event"], "bounded_smoke_complete")
            self.assertEqual(summary["step"], 1)
            self.assertTrue(summary["full_train_blocked"])
            self.assertEqual(summary["hierarchy_state"]["status"], "BUILD")
            self.assertTrue(summary["hierarchy_state"]["snapshot_hash"])
            self.assertEqual(summary["hierarchy_state"]["replay"]["replay.reference_selected"], 8)
            audit = [json.loads(line) for line in (output/"smoke_diagnostics.jsonl").read_text().splitlines()]
            hierarchy = next(r for r in audit if r["event"] == "hierarchy")
            update = next(r for r in audit if r["event"] == "optimizer_update")
            self.assertEqual(len(hierarchy["replay_ids"]), 8)
            self.assertTrue(hierarchy["replay_selection_hash"])
            self.assertGreater(hierarchy["wall_seconds"], 0)
            self.assertGreater(update["wall_seconds"], 0)
            self.assertTrue(all(g["finite"] and g["nonzero"] for g in update["gradients"].values()))
            self.assertIn("git_sha", update)
            self.assertIsNone(update["cuda_peak_allocated"])
            self.assertEqual(audit[-1]["exit_code"], 0)

    def test_reconstruction_exact_pinned_seed0_per_class_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); metadata_fixture(root)
            numpy = np.random.get_state()
            a = reconstruct(root, .8, 3); b = reconstruct(root, .8, 3)
            self.assertEqual(a, b)
            self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
            expected = []
            for c in range(3):
                expected.extend(np.random.RandomState(0).choice(np.arange(c*5, c*5+5), 4, replace=False).tolist())
            self.assertEqual({r["sample_id"] for r in a["eligibility"]["records"]}, set(expected))

    def test_manifest_source_hash_split_and_dataset_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); metadata_fixture(root)
            value = reconstruct(root, .8, 3)
            path = root/"manifest.json"; write_manifest(path, value)
            _, verified = verified_manifest(root, path, digest(value), .8, 3)
            self.assertEqual(len(verified._records), 12)
            for field in ("upstream_sha", "source_tree_hash", "prop_samples"):
                bad = deepcopy(value); bad["source"][field] = "bad"
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): verified_manifest(root, path, digest(bad), .8, 3)
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): verified_manifest(root, path, "0"*64, .8, 3)
            (root/"images.txt").write_text((root/"images.txt").read_text().replace("1.jpg", "different.jpg"))
            with self.assertRaises(ValueError): verified_manifest(root, path, digest(value), .8, 3)

    def test_only_explicit_bounded_smoke_opens_gate(self):
        with self.assertRaisesRegex(FullTrainingBlocked, "Phase 4"):
            ensure_provider_ready(cfg())
        ensure_provider_ready(cfg(), bounded_smoke=True)
        for n in (0, 3, 16, True):
            with self.assertRaises(ValueError): SmokeBudget(n)
        budget = SmokeBudget(2); budget.step_completed()
        with self.assertRaises(BoundedSmokeComplete): budget.step_completed()
        self.assertEqual(budget.steps, 2)

    def test_smoke_ast_terminates_post_optimizer_and_suppresses_base_eval(self):
        from dean_integration.external import bound_entrypoint
        checkout = Path(__file__).resolve().parents[2]/"OCGCD-dean-audit"
        tree = ast.parse((checkout/"Ours_train_online.py").read_text(encoding="utf-8"))
        patch_experiment(tree, paired_online=True, reference_selex=True)
        instrument_boundary(tree, paired_online=True); patch_bounded_smoke(tree)
        text = ast.unparse(tree)
        self.assertIn("opt.step()\n                    _dean_smoke.step_completed()", text)
        self.assertIn("acc_0 = None", text)
        self.assertIn("base accuracy not measured", text)
        self.assertIsInstance(bound_entrypoint(checkout, ".", paired_online=True, reference_selex=True, bounded_smoke=True), type(compile("", "", "exec")))


if __name__ == "__main__": unittest.main()
