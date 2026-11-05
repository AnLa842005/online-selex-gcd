"""CPU integration: isolated views, true replay, EMA lifecycle and DEAN optimizer AST."""
import ast
from dataclasses import asdict, replace
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
import yaml

from dean_integration.config import ROOT
from dean_integration.experiment_config import load_selex_config
from dean_integration.paired_stream import PairedOnlineStream
from dean_integration.preflight import sha256
from dean_integration.provenance import execute_run
from dean_integration.runtime import patch_experiment
from dean_integration.selex_config import ReferenceConfig
from dean_integration.selex_real_online import CurrentBatchImages, LabeledReplay, RealSelExOnline, assert_optimizer_order
from dean_integration.selex_rng import SelExRNG


def config(**kwargs):
    return replace(ReferenceConfig(), output_dim=16, hidden_dim=12, bottleneck_dim=8, **kwargs)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 768, bias=False).requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(2, 4)*0.1)
        self.lora_B = nn.Parameter(torch.zeros(768, 2))
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return self.frozen(x) + F.linear(F.linear(x, self.lora_A), self.lora_B)


class Head(nn.Module):
    def __init__(self, classes=162):
        super().__init__()
        self.head = nn.Linear(768, classes)

    def forward(self, features):
        return self.head(features)

    def update(self, classes):
        previous = self.head
        self.head = nn.Linear(768, classes)
        with torch.no_grad():
            self.head.weight[:len(previous.weight)].copy_(previous.weight)
            self.head.bias[:len(previous.bias)].copy_(previous.bias)


def image_provider(ids=(20, 33, 71, 91), *, noise=True):
    def load(key):
        return torch.tensor([key/100., 0.4, -0.1, 0.9])

    def augment(image):
        if noise:
            return image + torch.rand(4)*0.03 + random.random()*0.01 + float(np.random.rand())*0.01
        return image
    return CurrentBatchImages(list(ids), load, augment, lambda image: image)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model, self.head = Student(), Head()
        self.engine = RealSelExOnline(config(), 17)
        self.engine.start_session(self.model)
        self.images = image_provider()
        self.labels = torch.tensor([0, 0, 160, 160])
        self.engine.prepare(self.model, self.images, self.labels, 162)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.optimizer.add_param_group(dict(params=self.head.parameters(), lr=1e-4))
        self.engine.attach_optimizer(self.optimizer, self.head)

    def step(self, replay=None):
        primary = torch.randn(4, 4)
        ce = F.cross_entropy(self.head(self.model(primary)), self.labels)
        ec = ce*0.01
        loss = self.engine.augment(ce+ec, ce, ec, self.model, self.labels, self.images.sample_ids, replay)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def test_finite_gradients_reach_lora_projector_classifier_not_frozen_backbone(self):
        before = self.model.frozen.weight.detach().clone()
        loss = self.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(before, self.model.frozen.weight))
        self.assertIsNone(self.model.frozen.weight.grad)
        self.assertGreater(float(self.model.lora_B.grad.abs().sum()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.engine.projector.parameters() if p.grad is not None), 0)
        self.assertIsNotNone(self.head.head.weight.grad)

    def test_two_fresh_views_every_update_and_hierarchy_fixed_for_15_updates(self):
        snapshot = self.engine.snapshot
        prototype_before = {key: value.clone() for key, value in self.engine.hierarchy.prototypes.items()}
        previous_views = None
        for _ in range(15):
            self.step()
            views = self.model.seen[-1]
            self.assertEqual(views.shape, (8, 4))
            self.assertFalse(torch.equal(views[:4], views[4:]))
            if previous_views is not None:
                self.assertFalse(torch.equal(previous_views, views))
            previous_views = views
            self.assertIs(self.engine.snapshot, snapshot)
        self.assertEqual(self.engine.hierarchy.counters["hierarchy_rebuilds"], 1)
        for key in prototype_before:
            self.assertTrue(torch.equal(prototype_before[key], self.engine.hierarchy.prototypes[key]))
        self.engine.prepare(self.model, self.images, self.labels, 162)
        self.assertEqual(self.engine.hierarchy.counters["hierarchy_rebuilds"], 2)
        self.assertIsNot(self.engine.snapshot, snapshot)
        self.assertEqual(self.engine.step, 0)

    def test_labeled_replay_separate_from_current_pseudo_labels(self):
        replay_images = image_provider((200, 210, 220, 230))
        replay = LabeledReplay(replay_images, torch.tensor([7, 7, 8, 8]))
        self.assertTrue(torch.isfinite(self.step(replay)))
        with self.assertRaisesRegex(ValueError, "explicit LabeledReplay"):
            self.step(self.labels)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.step(LabeledReplay(self.images, torch.tensor([0, 0, 160, 160])))
        with self.assertRaises(ValueError):
            LabeledReplay(replay_images, torch.tensor([-1, 0, 1, 1]))

    def test_empty_replay_omits_only_gt(self):
        empty = LabeledReplay(image_provider(()), torch.empty(0, dtype=torch.long))
        self.assertTrue(torch.isfinite(self.step(empty)))
        self.assertGreater(self.engine.counters["sse.gt.empty_replay"], 0)
        self.assertEqual(self.engine.step, 1)

    def test_primary_ce_rng_and_python_numpy_torch_states_are_isolated(self):
        torch_state = torch.get_rng_state().clone()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        loss = torch.tensor(1., requires_grad=True)
        self.engine.augment(loss, loss, loss*0, self.model, self.labels, self.images.sample_ids)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))

    def test_cpu_autocast_forward_backward_preserves_fp32_state_and_finite_gradients(self):
        observed = []
        hook = self.engine.projector.register_forward_hook(lambda module, inputs, output: observed.append((output.shape, output.dtype)))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.engine.prepare(self.model, self.images, self.labels, 162)
            ce = F.cross_entropy(self.head(self.model(torch.randn(4, 4))), self.labels)
            loss = self.engine.augment(ce, ce, ce*0, self.model, self.labels, self.images.sample_ids)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        hook.remove()
        self.assertEqual(observed, [(torch.Size([8, 16]), torch.float32)])
        self.assertTrue(torch.isfinite(loss))
        for p in list(self.model.parameters())+list(self.engine.projector.parameters()):
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        for p in self.engine.hierarchy.prototypes.values():
            self.assertEqual(p.dtype, torch.float32)
            self.assertFalse(p.requires_grad)

    def test_raw_cub_adapter_never_reads_gt_and_rejects_historical_ids(self):
        class Row:
            filepath = "bird.jpg"
            @property
            def target(self):
                raise AssertionError("GT must not be read")
        class Frame:
            def __init__(self):
                self.iloc = self
            def __len__(self):
                return 4
            def __getitem__(self, index):
                return Row()
        dataset = SimpleNamespace(uq_idxs=np.array([20, 33, 71, 91]), ys=[0, 0, 160, 160],
                                  data=Frame(), root="/not-opened", base_folder="cub", loader=lambda path: torch.ones(4))
        self.engine.prepare_batch(self.model, dataset, lambda is_train: (lambda x: x), 162)
        self.assertEqual(self.engine.current_labels, {20: 0, 33: 0, 71: 160, 91: 160})
        with self.assertRaisesRegex(ValueError, "historical"):
            self.engine.images.images([999], two_views=True)
        with self.assertRaisesRegex(ValueError, "differ"):
            self.engine.augment(torch.tensor(1.), torch.tensor(1.), torch.tensor(0.), self.model,
                                torch.tensor([1, 1, 160, 160]), self.images.sample_ids)

    def test_expanded_head_keeps_projector_group_and_state(self):
        self.step()
        parameter = next(self.engine.projector.parameters())
        state_before = self.optimizer.state[parameter]["exp_avg"].clone()
        del self.optimizer.param_groups[-1]
        self.head.update(164)
        self.optimizer.add_param_group(dict(params=self.head.parameters(), lr=1e-4))
        self.engine.attach_optimizer(self.optimizer, self.head)
        assert_optimizer_order(self.optimizer, self.engine.projector, self.head)
        self.assertTrue(torch.equal(state_before, self.optimizer.state[parameter]["exp_avg"]))
        self.assertEqual(self.optimizer.param_groups[-2]["name"], "selex_projector")
        self.assertEqual(self.optimizer.param_groups[-2]["lr"], 1e-4)
        groups = self.optimizer.param_groups
        groups[-1], groups[-2] = groups[-2], groups[-1]
        with self.assertRaises(ValueError):
            assert_optimizer_order(self.optimizer, self.engine.projector, self.head)

    def test_telemetry_reports_all_losses_skips_counts_groups_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = RealSelExOnline(config(), 17, tmp)
            engine.start_session(self.model)
            engine.prepare(self.model, self.images, self.labels, 162)
            optimizer = torch.optim.AdamW(self.model.parameters())
            optimizer.add_param_group(dict(params=self.head.parameters()))
            engine.attach_optimizer(optimizer, self.head)
            ce = torch.tensor(1., requires_grad=True)
            result = engine.augment(ce+.3, ce, torch.tensor(.3), self.model, self.labels, self.images.sample_ids)
            events = [json.loads(line) for line in (Path(tmp)/"selex_reference_metrics.jsonl").read_text().splitlines()]
            update = events[-1]
            for key in ("ce", "ec", "use", "sse", "selex_total", "final_total", "counters"):
                self.assertIn(key, update)
            self.assertAlmostEqual(update["final_total"], update["ce"]+update["ec"]+.1*update["selex_total"], places=5)
            summary = json.loads((Path(tmp)/"selex_reference_summary.json").read_text())
            self.assertEqual(summary["optimizer_groups"][-2]["name"], "selex_projector")
            self.assertGreater(summary["projector_trainable"], 0)
            self.assertTrue(torch.isfinite(result))


class RngAndCliTests(unittest.TestCase):
    def test_provenance_preserves_worker_failure_code_with_partial_research_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def worker(command, **kwargs):
                (root/"seed1/selex_reference_summary.json").write_text("partial JSON")
                return SimpleNamespace(returncode=9)
            code = execute_run(root, {"real_selex_active": True}, [(1, root/"seed1", [])], worker)
            self.assertEqual(code, 9)
            report = json.loads((root/"provenance.json").read_text())
            self.assertEqual(report["returncode"], 9)
            self.assertIn("selex_reference_error", report["executions"][0])

    def test_provenance_rejects_success_without_optimization_and_records_valid_summary(self):
        for step in (0, 1):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                summary = dict(implementation="reference_v1", config=asdict(config()), step=step,
                               optimizer_groups=[dict(name="selex_projector"), dict(name="dean_classifier")])
                def worker(command, **kwargs):
                    (root/"seed1/selex_reference_summary.json").write_text(json.dumps(summary))
                    return SimpleNamespace(returncode=0)
                report = dict(real_selex_active=True, selex_config=asdict(config()))
                if step == 0:
                    with self.assertRaisesRegex(ValueError, "without optimizer updates"):
                        execute_run(root, report, [(1, root/"seed1", [])], worker)
                else:
                    self.assertEqual(execute_run(root, report, [(1, root/"seed1", [])], worker), 0)
                    record = json.loads((root/"provenance.json").read_text())["executions"][0]
                    self.assertEqual(record["selex_reference"], summary)
                    self.assertEqual(len(record["selex_reference_summary_sha256"]), 64)

    def test_engine_construction_and_projector_init_preserve_all_host_rng_states(self):
        model = Student()
        before = torch.get_rng_state().clone()
        python, numpy = random.getstate(), np.random.get_state()
        engine = RealSelExOnline(config(), 17)
        engine.start_session(model)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(python, random.getstate())
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))

    def test_cuda_rng_scope_isolated_with_mocked_cuda_states_without_gpu_execution(self):
        rng = SelExRNG(17)
        external = torch.arange(16, dtype=torch.uint8)
        state = {"cuda": external.clone()}
        class CudaGenerator:
            def manual_seed(self, seed):
                self.value = torch.full((16,), seed % 255, dtype=torch.uint8)
                return self
            def get_state(self):
                return self.value.clone()
        def set_state(value, device=0):
            state["cuda"] = value.clone()
        with patch("torch.cuda.get_rng_state", side_effect=lambda device=0: state["cuda"].clone()), \
             patch("torch.cuda.set_rng_state", side_effect=set_state), \
             patch("torch.Generator", side_effect=lambda **kwargs: CudaGenerator()):
            with rng.scope("cuda:0"):
                self.assertFalse(torch.equal(external, state["cuda"]))
                state["cuda"].fill_(99)
            self.assertTrue(torch.equal(external, state["cuda"]))
            with rng.scope("cuda:0"):
                self.assertTrue(torch.equal(state["cuda"], torch.full((16,), 99, dtype=torch.uint8)))
            self.assertTrue(torch.equal(external, state["cuda"]))

    def test_private_rng_reproducible_independent_and_restored_on_exception(self):
        a, b = SelExRNG(17), SelExRNG(17)
        before = torch.get_rng_state().clone()
        def draw(rng):
            with rng.scope():
                return random.random(), float(np.random.rand()), torch.rand(4)
        x, y = draw(a), draw(b)
        self.assertEqual(x[:2], y[:2])
        self.assertTrue(torch.equal(x[2], y[2]))
        self.assertFalse(torch.equal(x[2], draw(a)[2]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        python = random.getstate()
        with self.assertRaises(RuntimeError):
            with a.scope():
                torch.rand(100), random.random(), np.random.rand()
                raise RuntimeError("fixture")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(python, random.getstate())

    def test_augmentation_rng_cannot_change_paired_manifest(self):
        class IDs:
            uq_idxs = np.arange(24)
            def __len__(self):
                return 24
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/"a.json", Path(tmp)/"b.json"]
            first = PairedOnlineStream(dict(online_seed=17), paths[0])
            first.prepare([IDs()], 8)
            with SelExRNG(17).scope():
                torch.rand(1000), random.random(), np.random.rand(100)
            second = PairedOnlineStream(dict(online_seed=17), paths[1])
            second.prepare([IDs()], 8)
            self.assertEqual(sha256(paths[0]), sha256(paths[1]))

    def test_reference_config_dispatch_and_no_unknown_fields(self):
        self.assertEqual(load_selex_config(implementation="reference_v1").gamma, .1)
        with self.assertRaises(ValueError):
            load_selex_config(implementation="literal_bug_compatible")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"config.yaml"
            path.write_text(yaml.safe_dump({**asdict(config()), "literal_bugs": True}))
            with self.assertRaises(ValueError):
                load_selex_config(path, "reference_v1")

    def test_disabled_and_gamma_zero_cli_do_not_construct_real_engine_or_patch_loss(self):
        spec = importlib.util.spec_from_file_location("reference_cli", ROOT/"scripts/run_dean.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for enabled, gamma in ((False, .1), (True, 0.)):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint = root/"dino.pth"
                checkpoint.write_bytes(b"fixture")
                path = root/"real.yaml"
                path.write_text(yaml.safe_dump(asdict(config(gamma=gamma))))
                argv = ["run_dean", "--action", "worker", "--upstream", tmp, "--dataset-parent", tmp,
                        "--checkpoint", str(checkpoint), "--output", tmp, "--seed", "1",
                        "--selex-implementation", "reference_v1", "--selex-config", str(path)]
                if enabled:
                    argv.append("--use_selex_online")
                code = compile("assert _dean_real is None", "<disabled-worker>", "exec")
                with patch("sys.argv", argv), patch.object(module, "verify_checkout", return_value=root), \
                     patch.object(module, "check_assets", return_value={"errors": []}), \
                     patch.object(module.os, "chdir"), \
                     patch.object(module, "bound_entrypoint", return_value=code) as bound, \
                     patch("dean_integration.selex_real_online.RealSelExOnline") as constructor:
                    self.assertEqual(module.main(), 0)
                    constructor.assert_not_called()
                    self.assertFalse(bound.call_args.kwargs["reference_selex"])
                    self.assertFalse(bound.call_args.kwargs["use_selex_online"])

    def test_real_worker_cli_executes_small_head_and_persists_research_telemetry(self):
        spec = importlib.util.spec_from_file_location("reference_active_cli", ROOT/"scripts/run_dean.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        body = """
import torch
from dean_integration.selex_real_online import CurrentBatchImages
model = torch.nn.Linear(4, 768)
head = torch.nn.Linear(768, 162)
_dean_real.start_session(model)
ids = torch.arange(4)
labels = torch.tensor([0, 0, 160, 160])
images = CurrentBatchImages(ids, lambda key: torch.ones(4)*(key+1), lambda x: x+torch.rand(4), lambda x: x)
_dean_real.prepare(model, images, labels, 162)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
opt.add_param_group(dict(params=head.parameters(), lr=1e-4))
_dean_real.attach_optimizer(opt, head)
ce = torch.nn.functional.cross_entropy(head(model(torch.ones(4, 4))), labels)
loss = _dean_real.augment(ce, ce, ce*0, model, labels, ids)
opt.zero_grad()
loss.backward()
opt.step()
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/"dino.pth"
            checkpoint.write_bytes(b"fixture")
            path = root/"real.yaml"
            path.write_text(yaml.safe_dump(asdict(config())))
            argv = ["run_dean", "--action", "worker", "--upstream", tmp, "--dataset-parent", tmp,
                    "--checkpoint", str(checkpoint), "--output", tmp, "--seed", "1",
                    "--selex-implementation", "reference_v1", "--selex-config", str(path), "--use_selex_online"]
            with patch("sys.argv", argv), patch.object(module, "verify_checkout", return_value=root), \
                 patch.object(module, "check_assets", return_value={"errors": []}), \
                 patch.object(module.os, "chdir"), \
                 patch.object(module, "bound_entrypoint", return_value=compile(body, "<real-worker-CPU>", "exec")) as bound:
                self.assertEqual(module.main(), 0)
                self.assertTrue(bound.call_args.kwargs["reference_selex"])
                self.assertFalse(bound.call_args.kwargs["use_selex_online"])
            summary = json.loads((root/"selex_reference_summary.json").read_text())
            self.assertEqual(summary["step"], 1)
            self.assertEqual(summary["config"]["output_dim"], 16)
            args = SimpleNamespace(config=path, dataset_parent=root, checkpoint=checkpoint, base_checkpoint=None,
                                   boundary_rng=None, selex_config=path, online_only=False, paired_online=False,
                                   use_selex_online=True, selex_implementation="reference_v1")
            command = module.worker_command(args, root, root, 1)
            self.assertIn("reference_v1", command)


@unittest.skipUnless(os.environ.get("DEAN_UPSTREAM_CHECKOUT"), "External DEAN is opt-in")
class PinnedAstTests(unittest.TestCase):
    def source(self):
        return Path(os.environ["DEAN_UPSTREAM_CHECKOUT"]) / "Ours_train_online.py"

    def test_all_real_hooks_compile_on_pinned_source_and_gamma_zero_bytecode_identical(self):
        from dean_integration.external import bound_entrypoint, verify_checkout
        verify_checkout(self.source().parent)
        raw = self.source().read_bytes()
        code = bound_entrypoint(self.source().parent, ROOT, paired_online=True, reference_selex=True, capture_boundary=True)
        self.assertIsNotNone(code)
        a = bound_entrypoint(self.source().parent, ROOT, paired_online=True)
        b = bound_entrypoint(self.source().parent, ROOT, paired_online=True, reference_selex=False)
        self.assertEqual(a.co_code, b.co_code)
        self.assertEqual(a.co_consts, b.co_consts)
        self.assertEqual(raw, self.source().read_bytes())
        tree = patch_experiment(ast.parse(raw.decode("utf-8")), reference_selex=True)
        text = ast.unparse(tree)
        self.assertEqual(text.count("_dean_real.attach_optimizer(opt, header_now)"), 2)
        self.assertEqual(text.count("_dean_real.prepare_batch("), 1)
        self.assertEqual(text.count("_dean_real.augment("), 1)

    def test_original_dean_optimizer_expansion_preserves_real_projector_adam_state(self):
        torch.set_num_threads(1)
        model, head = Student(), Head()
        engine = RealSelExOnline(config(), 17)
        engine.start_session(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.add_param_group(dict(params=head.parameters(), lr=1e-4))
        engine.attach_optimizer(opt, head)
        loss = engine.projector(model(torch.randn(4, 4))).sum() + head(model(torch.randn(4, 4))).square().mean()
        loss.backward()
        opt.step()
        projector_param = next(engine.projector.parameters())
        state = opt.state[projector_param]["exp_avg"].clone()
        head.update(164)
        tree = patch_experiment(ast.parse(self.source().read_text(encoding="utf-8")), reference_selex=True)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and n.lineno == 659)
        class CPU(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
                    node.func.attr = "cpu"
                return node
            def visit_Constant(self, node):
                if node.value == "cuda":
                    node.value = "cpu"
                return node
        node = CPU().visit(node)
        import copy
        ns = dict(torch=torch, copy=copy, opt=opt, header_now=head, batch_idx=1, nb_classes_k=2,
                  args=SimpleNamespace(lr=1e-4), _dean_real=engine)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<original-DEAN-optimizer-CPU>", "exec"), ns)
        assert_optimizer_order(opt, engine.projector, head)
        self.assertTrue(torch.equal(state, opt.state[projector_param]["exp_avg"]))
        self.assertEqual(opt.state[head.head.weight]["exp_avg"].shape, head.head.weight.shape)
        self.assertEqual(opt.state[head.head.bias]["exp_avg"].shape, head.head.bias.shape)
        opt.zero_grad()
        next_loss = engine.projector(model(torch.randn(4, 4))).sum() + head(model(torch.randn(4, 4))).square().mean()
        next_loss.backward()
        opt.step()
        self.assertEqual(int(opt.state[projector_param]["step"]), 2)


if __name__ == "__main__":
    unittest.main()
