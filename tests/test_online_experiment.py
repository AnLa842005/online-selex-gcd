from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from online_gcd.experiment import run_experiment
from online_gcd.prototype_bank import PrototypeBank
from online_gcd.state import load_state, save_state, state_json
from online_gcd.stream import load_config
from online_gcd.synthetic import SyntheticConfig, generate_synthetic
from scripts.run_synthetic_online_baseline import main

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/online_selex/synthetic_prototype_v0.yaml'


class SyntheticExperimentTests(unittest.TestCase):
    def test_default_schema_accuracy_determinism_and_final_reload(self):
        config = load_config(CONFIG, SyntheticConfig)
        first, bank = run_experiment(config)
        second, repeated = run_experiment(config)
        self.assertEqual(first, second)
        self.assertEqual(state_json(bank), state_json(repeated))
        self.assertEqual(set(first), {'schema_version', 'audience', 'baseline_version', 'dataset',
                                    'configuration', 'synthetic_checksum', 'stream_manifest_checksum',
                                    'environment', 'base_probe', 'sessions', 'summary'})
        self.assertEqual(first['schema_version'], 1)
        self.assertEqual(first['audience'], 'evaluator_only')
        self.assertEqual(len(first['sessions']), config.num_online_sessions)
        self.assertGreater(first['summary']['final_session_accuracy'], 0.95)
        self.assertGreater(first['summary']['final_probe_accuracy']['all']['value'], 0.95)
        self.assertGreater(first['summary']['average_online_accuracy'], 0.95)
        self.assertEqual(first['summary']['known_prototype_count'], config.num_known_classes)
        self.assertGreater(first['summary']['discovered_novel_cluster_count'], 0)
        self.assertEqual(first['summary']['final_cumulative_accuracy']['all']['count'],
                         (config.num_known_classes + config.num_novel_classes) * config.samples_per_class)
        data = generate_synthetic(config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'learner.json'
            save_state(bank, path)
            loaded = load_state(path, config.feature_dim)
            self.assertEqual([bank.predict(p.feature) for p in data.probes],
                             [loaded.predict(p.feature) for p in data.probes])
            self.assertEqual(state_json(bank), state_json(loaded))

    def test_midstream_save_load_continuation_matches_uninterrupted(self):
        config = load_config(CONFIG, SyntheticConfig)
        data = generate_synthetic(config)
        bank = PrototypeBank(config.feature_dim, config.threshold, config.momentum)
        bank.initialize_known(data.labelled_base_groups())
        for p in data.learner_session(1):
            bank.observe(p.sample_id, p.session_id, p.feature)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'midstream.json'
            save_state(bank, path)
            restored = load_state(path)
            for t in range(2, config.num_online_sessions + 1):
                for p in data.learner_session(t):
                    self.assertEqual(bank.observe(p.sample_id, t, p.feature),
                                     restored.observe(p.sample_id, t, p.feature))
            self.assertEqual(state_json(bank), state_json(restored))
            _, uninterrupted = run_experiment(config)
            self.assertEqual(state_json(bank), state_json(uninterrupted))

    def test_bank_interface_receives_only_base_labels_and_safe_online_observations(self):
        config = load_config(CONFIG, SyntheticConfig)
        calls = []
        class AuditedBank(PrototypeBank):
            def __init__(self, feature_dim, threshold, momentum):
                calls.append(('config', feature_dim, threshold, momentum))
                super().__init__(feature_dim, threshold, momentum)

            def initialize_known(self, groups):
                self_test.assertEqual(set(groups), set(range(config.num_known_classes)))
                super().initialize_known(groups)

            def observe(self, sample_id, session_id, feature):
                calls.append(('observation', sample_id, session_id))
                return super().observe(sample_id, session_id, feature)
        self_test = self
        with patch('online_gcd.experiment.PrototypeBank', AuditedBank):
            run_experiment(config)
        self.assertEqual(calls[0], ('config', config.feature_dim, config.threshold, config.momentum))
        self.assertEqual(len(calls) - 1,
                         (config.num_known_classes + config.num_novel_classes) * config.samples_per_class)

    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('sys.argv', ['run_synthetic_online_baseline.py', *map(str, arguments)]), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                main()
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_two_runs_byte_identical_outputs_and_seed_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for n in (1, 2):
                code, stdout, stderr = self.invoke(['--config', CONFIG, '--output-json', root / f'result{n}.json',
                                                   '--save-state', root / f'state{n}.json'])
                self.assertEqual(code, 0, stderr)
                self.assertIn('CPU', stdout)
            self.assertEqual((root / 'result1.json').read_bytes(), (root / 'result2.json').read_bytes())
            self.assertEqual((root / 'state1.json').read_bytes(), (root / 'state2.json').read_bytes())
            code, _, stderr = self.invoke(['--seed', '2', '--output-json', root / 'seed2.json'])
            self.assertEqual(code, 0, stderr)
            other = json.loads((root / 'seed2.json').read_text())
            self.assertEqual(other['configuration']['seed'], 2)
            first = json.loads((root / 'result1.json').read_text())
            self.assertNotEqual(first['synthetic_checksum'], other['synthetic_checksum'])
            code, _, stderr = self.invoke(['--output-json', root / 'result1.json'])
            self.assertEqual(code, 2)
            self.assertIn('new file', stderr)
            code, _, stderr = self.invoke(['--output-json', root / 'same.json', '--save-state', root / 'same.json'])
            self.assertEqual(code, 2)
            self.assertIn('separate output paths', stderr)
        self.assertEqual(self.invoke(['--seed', '-1'])[0], 2)

    def test_standalone_command_with_runtime_guards_against_external_workloads(self):
        # Execute the real CLI in a clean interpreter with prohibited access blocked.
        code = r'''
import builtins
import os
import platform
import runpy
import sys
# Python's Windows platform lookup may execute the harmless `ver` command.
# Cache it before the workload guard, so NumPy's platform check uses that cache.
platform.uname()
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'data', 'models', 'methods'} or name == 'online_gcd.ssb':
        raise AssertionError('Forbidden workload import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
def audit(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen'}:
        raise AssertionError('Forbidden external operation: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0]).lower().replace(chr(92), '/')
        if 'cub_200_2011' in path or '/ssb_splits/' in path or path.endswith(('.pt', '.pth', '.ckpt', '.pkl')):
            raise AssertionError('Forbidden data/checkpoint access: ' + path)
sys.addaudithook(audit)
sys.argv = ['scripts/run_synthetic_online_baseline.py']
runpy.run_path(sys.argv[0], run_name='__main__')
assert 'torch' not in sys.modules
assert 'online_gcd.ssb' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                                text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Infrastructure validation only', result.stdout)


if __name__ == '__main__':
    unittest.main()
