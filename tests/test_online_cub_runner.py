from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from online_cub_fixtures import fixture_cache
from online_gcd.artifacts import file_sha256, seal
from online_gcd.calibration import calibrate_known_base
from online_gcd.clean_protocol import CleanConfig
from online_gcd.cub_runner import run_cub_baseline
from online_gcd.evaluator import OnlineEvaluator
from online_gcd.feature_cache import write_cache
from online_gcd.state import state_json

ROOT = Path(__file__).resolve().parents[1]


class CalibrationTests(unittest.TestCase):
    def test_objective_candidates_tie_policy_and_counts(self):
        groups = {3: [[1, .1], [1, -.1]], 7: [[-1, .1], [-1, -.1]]}
        result = calibrate_known_base(groups, base_identity='base', cache_identity='cache')
        self.assertEqual(result['positive_count'], 4)
        self.assertEqual(result['negative_count'], 4)
        self.assertEqual(result['balanced_accuracy'], 1)
        best = max(r['balanced_accuracy'] for r in result['candidates'])
        self.assertEqual(result['threshold'], max(r['threshold'] for r in result['candidates']
                                                 if r['balanced_accuracy'] == best))
        positive = (1 - .01) / (1 + .01)
        self.assertLessEqual(result['threshold'], positive + 1e-14)
        self.assertEqual(result, calibrate_known_base(groups, base_identity='base', cache_identity='cache'))

    def test_loo_is_not_in_sample_mean(self):
        result = calibrate_known_base({0: [[1, 0], [0, 1]], 1: [[-1, 0], [0, -1]]},
                                      base_identity='b', cache_identity='c')
        # Same-class LOO similarity is exactly zero, not 1/sqrt(2).
        self.assertEqual(result['threshold'], 0)
        self.assertEqual(result['balanced_accuracy'], 1)

    def test_too_few_or_degenerate_base_samples_fail(self):
        for groups in ({0: [[1, 0], [1, 0]]},
                       {0: [[1, 0]], 1: [[0, 1], [0, 1]]},
                       {0: [[1, 0], [-1, 0]], 1: [[0, 1], [0, 1]]}):
            with self.assertRaises(ValueError):
                calibrate_known_base(groups, base_identity='b', cache_identity='c')


class RealFeatureRunnerTests(unittest.TestCase):
    def test_fake_cache_end_to_end_determinism_probe_immutability_and_calibration_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, manifests, cache, features, _ = fixture_cache(folder)
            before = file_sha256(cache / 'features.npy')
            base_rows = bundle['base']['records']
            checked = []
            original_probe = OnlineEvaluator._probe

            def audit_calibration(groups, **kwargs):
                self.assertEqual(set(groups), {1, 5})
                for c, values in groups.items():
                    np.testing.assert_array_equal(values, [features[r['feature_row']] for r in base_rows if r['known_class_id'] == c])
                checked.append('calibration')
                return calibrate_known_base(groups, **kwargs)

            def audit_probe(evaluator, bank, session):
                self.assertTrue(checked)
                snapshot = state_json(bank)
                eligible = [r for r in bundle['probe_evaluator']['records'] if r['eligibility_session'] <= session]
                with patch.object(bank, 'predict', wraps=bank.predict) as prediction:
                    report = original_probe(evaluator, bank, session)
                    self.assertEqual(prediction.call_count, len(eligible))
                self.assertEqual(state_json(bank), snapshot)
                return report

            with patch('online_gcd.cub_runner.calibrate_known_base', side_effect=audit_calibration), \
                    patch.object(OnlineEvaluator, '_probe', audit_probe):
                first, bank = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True)
            second, repeated = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True,
                                                calibration_record=first['calibration'])
            self.assertEqual(first['experiment'], 'fixture_only')
            self.assertEqual(len(first['sessions']), 5)
            self.assertEqual(first['summary']['cumulative_accuracy']['all']['count'], 180)
            self.assertGreater(first['summary']['discovered_novel_cluster_count'], 0)
            self.assertEqual(file_sha256(cache / 'features.npy'), before)
            self.assertEqual(state_json(bank), state_json(repeated))
            for result in (first, second):
                result.pop('performance')
                for row in result['sessions']:
                    row.pop('performance')
            self.assertEqual(first, second)
            wrong = {k: v for k, v in first['calibration'].items() if k != 'checksum'}
            wrong['threshold'] = -1
            with self.assertRaisesRegex(ValueError, 'calibration does not match'):
                run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True, calibration_record=seal(wrong))

    def test_changes_outside_base_cannot_change_calibrated_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, manifests, cache, features, _ = fixture_cache(folder)
            first, _ = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True)
            base_indices = {r['feature_row'] for r in bundle['base']['records']}
            changed = features.copy()
            for i in range(len(changed)):
                if i not in base_indices:
                    changed[i] = -changed[i]
            other_cache = Path(folder) / 'other_cache'
            write_cache(other_cache, bundle, [changed], {'kind': 'fixture'})
            second, _ = run_cub_baseline(manifests, other_cache, CleanConfig(), allow_fixture=True)
            for key in ('threshold', 'balanced_accuracy', 'candidates', 'base_features_sha256'):
                self.assertEqual(first['calibration'][key], second['calibration'][key])

    def test_cli_help_is_cpu_only_and_no_fixture_switch(self):
        for script in ('build_cub_online_manifests.py', 'extract_clean_dino_cub_features.py',
                       'run_cub_frozen_dino_baseline.py'):
            result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--help'],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('--allow-fixture', result.stdout)
        # A fresh process may not import torch or any model while running fake caches.
        code = '''
import builtins, sys, tempfile
sys.path.insert(0, 'tests')
original = builtins.__import__
def guard(name, *a, **kw):
    if name.split('.')[0] in ('torch', 'torchvision', 'models', 'data', 'methods'):
        raise AssertionError(name)
    return original(name, *a, **kw)
builtins.__import__ = guard
from online_cub_fixtures import fixture_cache
from online_gcd.cub_runner import run_cub_baseline
from online_gcd.clean_protocol import CleanConfig
with tempfile.TemporaryDirectory() as root:
    _, manifests, cache, _, _ = fixture_cache(root)
    result, _ = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True)
    assert result['experiment'] == 'fixture_only'
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
