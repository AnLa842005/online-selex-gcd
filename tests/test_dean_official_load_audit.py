import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dean_integration.official_load_audit import audit_full_checkpoint, observe_official_load
from test_dean_checkpoint_audit import tiny_reference


class OfficialLoadAuditTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_reference()
        self.teacher = {key: torch.full_like(value, .125) for key, value in self.model.state_dict().items()}

    def test_full_teacher_raw_load_changes_weights_and_passes(self):
        state = {"student": {}, "teacher": self.teacher, "early_stopping_epoch": 20}
        state["teacher"]["head.last_layer.weight"] = torch.ones(2, 2)
        untouched = copy.deepcopy(state)
        report = observe_official_load(state, self.model)
        self.assertEqual(report["decision"], "PASS")
        self.assertTrue(report["weights_changed_after_official_load"])
        raw = report["raw_official_matching"]
        self.assertEqual(raw["raw_exact_key_matches"], len(self.model.state_dict()))
        self.assertEqual(raw["parameter_numel"]["ratio"], 1.)
        self.assertEqual(raw["observed_official_load"]["returned_missing_keys"], [])
        self.assertEqual(raw["unexpected_keys"], ["head.last_layer.weight"])
        self.assertEqual(report["hypothetical_prefix_matching"], {})
        for key, value in untouched["teacher"].items():
            self.assertTrue(torch.equal(value, state["teacher"][key]))

    def test_module_prefix_diagnostic_cannot_turn_official_failure_into_pass(self):
        source = {"teacher": {"module." + k: v for k, v in self.teacher.items()}}
        before = copy.deepcopy(self.model.state_dict())
        report = observe_official_load(source, self.model)
        self.assertEqual(report["decision"], "BLOCKED_UPSTREAM_CHECKPOINT_PREFIX_MISMATCH")
        self.assertFalse(report["weights_changed_after_official_load"])
        self.assertEqual(report["raw_official_matching"]["raw_exact_key_matches"], 0)
        a = report["hypothetical_prefix_matching"]["A_strip_module_against_official_target"]
        b = report["hypothetical_prefix_matching"]["B_strip_module_backbone_against_bare_backbone"]
        self.assertEqual(a["matched_parameter_numel_ratio"], 1.)
        self.assertEqual(b["matched_parameter_numel_ratio"], 1.)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))

    def test_no_weight_change_is_blocked_even_with_complete_raw_matching(self):
        report = observe_official_load({"teacher": copy.deepcopy(self.model.state_dict())}, self.model)
        self.assertEqual(report["raw_official_matching"]["status"], "PASS")
        self.assertEqual(report["decision"], "BLOCKED_OFFICIAL_LOAD_DID_NOT_CHANGE_BACKBONE_WEIGHTS")

    def test_shape_mismatch_raises_officially_and_cannot_pass(self):
        self.teacher["backbone.cls_token"] = torch.ones(1, 1, 3)
        report = observe_official_load({"teacher": self.teacher}, self.model)
        self.assertTrue(report["decision"].startswith("BLOCKED"))
        self.assertEqual(report["raw_official_matching"]["raw_exact_key_matches"], len(self.teacher))
        self.assertEqual(report["raw_official_matching"]["raw_shape_compatible_matches"], len(self.teacher) - 1)
        self.assertIn("RuntimeError", report["raw_official_matching"]["observed_official_load"]["error"])

    def test_missing_teacher_does_not_attempt_load(self):
        with patch.object(self.model, "load_state_dict", wraps=self.model.load_state_dict) as loader:
            report = observe_official_load(self.teacher, self.model)
        loader.assert_not_called()
        self.assertTrue(report["decision"].startswith("BLOCKED"))

    def test_file_and_source_unchanged_and_cpu_trusted_loading(self):
        from dean_integration.preflight import sha256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trusted_fixture.pth"
            torch.save({"teacher": self.teacher}, path)
            before = sha256(path)
            with patch("dean_integration.official_load_audit.EXPECTED_FULL_CHECKPOINT_BYTES", path.stat().st_size), \
                 patch("dean_integration.official_load_audit.verify_checkout", return_value=Path(directory)), \
                 patch("dean_integration.official_load_audit.build_reference_model", return_value=self.model), \
                 patch("dean_integration.official_load_audit.torch.load", wraps=torch.load) as loader:
                result = audit_full_checkpoint(path, directory)
            self.assertEqual(result["decision"], "PASS")
            loader.assert_called_once_with(path.resolve(), map_location="cpu", weights_only=False)
            self.assertTrue(result["checkpoint_unchanged"])
            self.assertEqual(sha256(path), before)
            self.assertLessEqual(len(result["sample_teacher_keys"]), 20)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_wrong_size_rejected_before_checkpoint_unpickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pth"
            path.write_bytes(b"wrong size")
            with patch("dean_integration.official_load_audit.verify_checkout", return_value=Path(directory)), \
                 patch("dean_integration.official_load_audit.torch.load") as loader:
                with self.assertRaisesRegex(ValueError, "Expected 871077689 bytes"):
                    audit_full_checkpoint(path, directory)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
