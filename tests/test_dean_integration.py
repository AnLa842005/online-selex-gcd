import ast
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from dean_integration.adapters import LearnerBatch, OfficialProtocolAdapter, SSB100ProtocolAdapter
from dean_integration.config import BASE_SHA, OFFICIAL_ARGS, ROOT, UPSTREAM_SHA, config_digest, load_config
from dean_integration.external import bound_entrypoint, verify_checkout


class DeanIntegrationTests(unittest.TestCase):
    def config(self, name="official"):
        return load_config(ROOT / f"configs/dean/cub_{name}_v0.yaml")

    def test_provenance_and_official_contract(self):
        config = self.config()
        self.assertEqual(config["upstream"]["sha"], UPSTREAM_SHA)
        self.assertEqual(config["seeds"], [1, 3, 8, 2, 4])
        lock = json.loads((ROOT / "configs/dean/upstream_lock.json").read_text())
        self.assertEqual(lock["sha"], UPSTREAM_SHA)
        self.assertEqual(len(lock["files"]), 42)
        self.assertFalse(any("license" in name.lower() for name in lock["files"]))
        self.assertEqual(len(config_digest(config)), 64)
        args = OfficialProtocolAdapter.arguments(config, 1, "checkpoint.pth")
        self.assertEqual(args[:len(OFFICIAL_ARGS)], OFFICIAL_ARGS)
        with self.assertRaises(ValueError):
            OfficialProtocolAdapter.arguments(config, 42, "checkpoint.pth")

    def test_reject_modified_and_malformed_config(self):
        for value in ([], {"mode": "unknown"}, {**self.config(), "seeds": [1]}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.yaml"
                path.write_text(yaml.safe_dump(value))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_protocol_and_label_boundary(self):
        official, ssb = self.config(), self.config("ssb100")
        self.assertNotEqual(official["result_namespace"], ssb["result_namespace"])
        with self.assertRaises(ValueError):
            OfficialProtocolAdapter.arguments(ssb, 1, "checkpoint.pth")
        with self.assertRaises(ValueError):
            SSB100ProtocolAdapter(official)
        with self.assertRaises(NotImplementedError):
            SSB100ProtocolAdapter(ssb).run()
        self.assertEqual([f.name for f in dataclasses.fields(LearnerBatch)],
                         ["sample_ids", "inputs", "session_id"])
        with self.assertRaises(TypeError):
            LearnerBatch(("opaque",), [[0.]], 1, true_novel_label=4)

    def test_frozen_and_offline_files_byte_identical_to_task_base(self):
        # All pre-existing code/config/result files are frozen in DEAN-0.
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE_SHA],
                                        cwd=ROOT, text=True).splitlines()
        for name in names:
            if name.endswith((".py", ".yaml", ".json")) or name.startswith("requirements"):
                before = subprocess.check_output(["git", "show", f"{BASE_SHA}:{name}"], cwd=ROOT)
                after = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(before.replace(b"\r\n", b"\n"), after, name)
        tag = subprocess.check_output(["git", "rev-parse", "selex-cub-baseline-v1^{commit}"],
                                      cwd=ROOT, text=True).strip()
        self.assertEqual(tag, "a52cf64dcbc43a57069c708ce0ce616cdd045221")

    def test_offline_reproduction_config_unchanged(self):
        result = subprocess.check_output([os.sys.executable, "scripts/compare_reproduction_config.py"],
                                         cwd=ROOT, text=True)
        self.assertEqual(json.loads(result)["differences"], {})

    def test_python39_syntax(self):
        for folder in (ROOT / "dean_integration",):
            for path in folder.glob("*.py"):
                ast.parse(path.read_text(), feature_version=(3, 9))

    def test_reject_wrong_sha_dirty_checkout_and_origin(self):
        for answers in (["0" * 40], [UPSTREAM_SHA, " M Ours_train_online.py"],
                        [UPSTREAM_SHA, "", "https://example.org/unofficial.git"]):
            with patch("dean_integration.external.git", side_effect=answers):
                with self.assertRaises(ValueError):
                    verify_checkout(Path("synthetic-checkout"))

    def test_reject_invalid_cub_metadata(self):
        from dean_integration.preflight import cub_metadata
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("images", "image_class_labels", "train_test_split"):
                (root / f"{name}.txt").write_text("1 1\n1 1\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                cub_metadata(root)
            for name in ("images", "image_class_labels", "train_test_split"):
                (root / f"{name}.txt").write_text("1 1\n")
            with self.assertRaisesRegex(ValueError, "11788"):
                cub_metadata(root)


@unittest.skipUnless(os.environ.get("DEAN_UPSTREAM_CHECKOUT"), "External upstream checkout is opt-in")
class ExternalDeanTests(unittest.TestCase):
    def setUp(self):
        self.checkout = Path(os.environ["DEAN_UPSTREAM_CHECKOUT"])

    def test_external_components(self):
        from dean_integration.smoke import run_smoke
        self.assertEqual(run_smoke(self.checkout)["status"], "PASS")

    def test_path_binding_only(self):
        verify_checkout(self.checkout)
        code = bound_entrypoint(self.checkout, Path("/synthetic-parent"))
        original = compile((self.checkout / "Ours_train_online.py").read_text(encoding="utf-8"),
                           str(self.checkout / "Ours_train_online.py"), "exec")
        self.assertEqual(code.co_code, original.co_code)
        differences = [(a, b) for a, b in zip(original.co_consts, code.co_consts) if a != b]
        self.assertEqual(differences, [("Your data path..", str(Path("/synthetic-parent").resolve()))])
        # Official CLI flags are compared to the pinned upstream shell command.
        import shlex
        shell = (self.checkout / "run.sh").read_text().replace("\\\n", " ")
        command = next(line.strip() for line in shell.splitlines() if "python Ours_train_online.py" in line)
        upstream = shlex.split(command)[2:]
        ours = OfficialProtocolAdapter.arguments(load_config(ROOT / "configs/dean/cub_official_v0.yaml"), 1, "x")
        def flags(tokens):
            result, key = {}, None
            for token in tokens:
                if token.startswith("--"):
                    key = token
                    result[key] = []
                else:
                    result[key].append(token)
            return result
        expected, actual = flags(upstream), flags(ours)
        expected["--seed"] = ["1"]
        for key, value in expected.items():
            self.assertEqual(actual[key], value, key)


if __name__ == "__main__":
    unittest.main()
