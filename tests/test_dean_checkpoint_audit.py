import copy
import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import nn

from dean_integration.checkpoint_audit import audit_checkpoint, compare_teacher


def tiny_reference():
    # Deliberately tiny independent fixture with the mandatory 12-block structure.
    model = nn.Module()
    model.backbone = nn.Module()
    backbone = model.backbone
    backbone.patch_embed = nn.Linear(2, 2)
    backbone.cls_token = nn.Parameter(torch.zeros(1, 1, 2))
    backbone.pos_embed = nn.Parameter(torch.zeros(1, 5, 2))
    backbone.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(12))
    backbone.norm = nn.LayerNorm(2)
    return model


class CheckpointAuditTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_reference()
        self.state = copy.deepcopy(self.model.state_dict())

    def test_exact_teacher_and_projection_keys_are_reported_without_mutation(self):
        source = {**self.state, "head.last_layer.weight": torch.ones(4, 2)}
        before = copy.deepcopy(self.model.state_dict())
        report = compare_teacher({"teacher": source}, self.model)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["unexpected_keys"], ["head.last_layer.weight"])
        self.assertEqual(report["parameter_numel"]["ratio"], 1.)
        self.assertEqual(report["parameter_tensors"]["ratio"], 1.)
        self.assertTrue(all(item["pass"] for item in report["critical_groups"].values()))
        self.assertFalse(report["actual_weights_loaded"])
        self.assertEqual(report["key_transformations_applied"], [])
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))

    def test_bare_empty_and_non_mapping_teacher_fail_closed(self):
        for source in (self.state, {"teacher": {}}, {"teacher": []}, None):
            result = compare_teacher(source, self.model)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["teacher_mapping_valid"])
            self.assertEqual(result["matched_key_count"], 0)

    def test_prefix_is_diagnostic_never_automatically_stripped(self):
        for prefix in ("module.", "teacher.", "module.teacher."):
            report = compare_teacher({"teacher": {prefix + k: v for k, v in self.state.items()}}, self.model)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["matched_key_count"], 0)
            self.assertEqual(report["prefix_alternatives_not_applied"][f"strip_{prefix}"], len(self.state))

    def test_backbone_prefix_must_not_be_invented(self):
        bare = {k.removeprefix("backbone."): v for k, v in self.state.items()}
        report = compare_teacher({"teacher": bare}, self.model)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["prefix_alternatives_not_applied"]["add_backbone_to_bare_keys"], len(bare))

    def test_shape_mismatch_and_finite_check(self):
        self.state["backbone.cls_token"] = torch.zeros(1, 1, 3)
        report = compare_teacher({"teacher": self.state}, self.model)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["shape_mismatches"][0]["key"], "backbone.cls_token")
        self.assertFalse(report["critical_groups"]["cls_token"]["pass"])
        self.state["backbone.cls_token"] = torch.full((1, 1, 2), float("nan"))
        report = compare_teacher({"teacher": self.state}, self.model)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["nonfinite_keys"], ["backbone.cls_token"])

    def test_partial_backbone_fails_even_with_high_numel_coverage(self):
        del self.state["backbone.norm.bias"]
        report = compare_teacher({"teacher": self.state}, self.model)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("backbone.norm.bias", report["missing_keys"])
        self.assertFalse(report["critical_groups"]["final_norm"]["pass"])

    def test_local_checkpoint_cpu_load_sha_and_no_wrapping_bare_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pth"
            torch.save(self.state, path)
            with patch("dean_integration.checkpoint_audit.build_reference_model", return_value=self.model):
                report = audit_checkpoint(path, "fixture-checkout")
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(len(report["sha256"]), 64)
            self.assertEqual(report["checkpoint_top_level_key_count"], len(self.state))

    def test_preflight_requires_key_audit_not_teacher_presence_alone(self):
        from dean_integration.preflight import check_assets
        with tempfile.TemporaryDirectory() as directory:
            with patch("dean_integration.checkpoint_audit.audit_checkpoint", return_value={
                "status": "FAIL", "errors": ["shape mismatch fixture"]
            }) as audit:
                result = check_assets(directory, Path(directory) / "teacher.pth", directory, upstream="fixture")
            audit.assert_called_once()
            self.assertIn("Checkpoint: shape mismatch fixture", result["errors"])


@unittest.skipUnless(os.environ.get("DEAN_UPSTREAM_CHECKOUT"), "External checkout is opt-in")
class OfficialReferenceTests(unittest.TestCase):
    def test_official_loader_and_loss_keep_pseudo_targets_separate_from_gt(self):
        from dean_integration.smoke import execute_nodes
        checkout = Path(os.environ["DEAN_UPSTREAM_CHECKOUT"])
        tree = ast.parse((checkout / "dataset/cub.py").read_text(encoding="utf-8"))
        cub = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CUBirds")
        getter = next(node for node in cub.body if isinstance(node, ast.FunctionDef) and node.name == "__getitem__")
        namespace = {"os": os}
        exec(compile(ast.Module(body=[getter], type_ignores=[]), "<external-cub-getitem>", "exec"), namespace)
        sample = SimpleNamespace(filepath="opaque.jpg", target=177)
        dataset = SimpleNamespace(data=SimpleNamespace(iloc=[sample]), root="synthetic",
                                  base_folder="images", loader=lambda _: torch.ones(4),
                                  transform=None, target_transform=None, binary=True,
                                  ys=[3], uq_idxs=[0])
        first = namespace["__getitem__"](dataset, 0)
        sample.target = 199
        second = namespace["__getitem__"](dataset, 0)
        self.assertEqual((first[1], second[1]), (3, 3))  # pseudo target unchanged
        self.assertEqual((first[2], second[2]), (176, 198))  # GT changes independently
        train_tree = ast.parse((checkout / "Ours_train_online.py").read_text(encoding="utf-8"))
        # Assert actual CE call and pseudo-target replacement, rather than a local loss replica.
        ce_calls = [node for node in ast.walk(train_tree) if isinstance(node, ast.Assign)
                    and node.lineno == 765]
        self.assertEqual(ast.unparse(ce_calls[0].value), "F.cross_entropy(logits, on_y)")
        self.assertTrue(any(isinstance(node, ast.Assign) and node.lineno == 604
                            and ast.unparse(node.targets[0]) == "dset_tr_o.ys" for node in ast.walk(train_tree)))
        header = SimpleNamespace(head=nn.Linear(4, 4))
        with torch.no_grad():
            header.head.weight.fill_(.01)
            header.head.bias.fill_(2.)
        args = SimpleNamespace(use_EC=True, energy_hp=1.)
        context = {"torch": torch, "F": torch.nn.functional, "args": args,
                   "model_now": nn.Identity(), "header_now": header,
                   "nb_classes": 2, "on_x": torch.ones(2, 4), "on_y": torch.tensor([3, 3]),
                   "y_n_msk": torch.tensor([[0]]), "gt_y": torch.tensor([176, 198])}
        # Header is callable upstream; use its unchanged Linear for this tiny fixture.
        class FixtureHeader:
            head = header.head
            def __call__(self, values):
                return self.head(values)
        context["header_now"] = FixtureHeader()
        execute_nodes(train_tree, 741, 767, context)
        original_loss = context["loss"].detach().clone()
        original_grad = torch.autograd.grad(context["loss"], header.head.weight)[0]
        context["gt_y"] = torch.tensor([999, -999])
        execute_nodes(train_tree, 741, 767, context)
        self.assertTrue(torch.equal(original_loss, context["loss"].detach()))
        self.assertTrue(torch.equal(original_grad, torch.autograd.grad(context["loss"], header.head.weight)[0]))

    def test_construct_actual_upstream_reference_without_pretrained_download(self):
        from dean_integration.checkpoint_audit import build_reference_model
        model = build_reference_model(os.environ["DEAN_UPSTREAM_CHECKOUT"])
        self.assertEqual(model.backbone.patch_embed.proj.weight.shape, (768, 3, 16, 16))
        self.assertEqual(model.backbone.pos_embed.shape, (1, 197, 768))
        self.assertEqual(len(model.backbone.blocks), 12)
        self.assertFalse(hasattr(model.backbone, "embedding"))
        report = compare_teacher({"teacher": model.state_dict()}, model)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["parameter_tensors"]["total"], 150)


if __name__ == "__main__":
    unittest.main()
