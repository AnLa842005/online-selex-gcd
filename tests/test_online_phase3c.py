"""Phase 3C regression tests: fake inputs only, no external workloads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from online_cub_fixtures import fixture_cache
import reference_online_v0 as reference
from online_gcd.artifacts import digest
from online_gcd.calibration import calibrate_known_base
from online_gcd.clean_dino import ManifestImages, safe_image_path
from online_gcd.clean_protocol import CleanConfig
from online_gcd.cub_runner import run_cub_baseline
from online_gcd.experiment import run_experiment
from online_gcd.prototype_bank import PrototypeBank
from online_gcd.state import bank_from_state, state_dict, state_json
from online_gcd.synthetic import SyntheticConfig


class NetworkVolumeTests(unittest.TestCase):
    def bundle(self, path):
        return {'partition': {'source_records': [{'image_id': 1, 'path': path}],
                              'feature_sample_ids': ['cub-' + digest(['CUB-200-2011', 1])]}}

    def test_preflight_never_resolves_or_stats_images(self):
        with patch.object(Path, 'resolve', side_effect=AssertionError('slow realpath')), \
                patch.object(Path, 'is_file', side_effect=AssertionError('remote stat')), \
                patch('online_gcd.clean_dino.evaluation_transform', return_value=lambda x: x):
            dataset = ManifestImages('remote-volume', self.bundle('bird/image.jpg'))
            self.assertEqual(len(dataset), 1)
        with self.assertRaises(FileNotFoundError):
            dataset[0]
        with patch('online_gcd.clean_dino.evaluation_transform'), self.assertRaises(FileNotFoundError):
            ManifestImages('remote-volume', self.bundle('bird/image.jpg'), verify_images=True)

    def test_malicious_paths_rejected_before_path_normalization(self):
        for path in ('', '.', '..', 'a/./b', 'a/../b', 'a//b', 'a/', '/a',
                     'C:/a', 'C:a', '//server/share/a', r'\\server\share\a',
                     r'a\..\b', r'a\b', 'a\x00b', 'a:stream'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_image_path(path)
        self.assertEqual(safe_image_path('001.bird/image.jpg'), Path('001.bird/image.jpg'))


class VectorEquivalenceTests(unittest.TestCase):
    def test_sequential_ids_vectors_counts_state_reload_and_probes(self):
        rng = np.random.RandomState(42)
        for threshold in (-1., .4, .99, 1.):
            groups = {c: rng.normal(size=(4, 32)) for c in (2, 10, 9)}
            old, new = reference.PrototypeBank(32, threshold), PrototypeBank(32, threshold)
            for bank in (old, new):
                bank.initialize_known(groups)
            with threadpool_limits(limits=1):
                for i, feature in enumerate(rng.normal(size=(150, 32))):
                    self.assertEqual(old.observe(str(i), 1 + i // 30, feature),
                                     new.observe(str(i), 1 + i // 30, feature))
                    self.assertEqual(state_json(old), state_json(new))
                    np.testing.assert_allclose([p.vector for p in old.prototypes],
                                               [p.vector for p in new.prototypes], rtol=0, atol=1e-15)
                probes = rng.normal(size=(47, 32))
                before = state_json(new)
                expected = [old.predict(x) for x in probes]
                for batch in (1, 7, 256):
                    self.assertEqual(expected, new.predict_batch(probes, batch))
                self.assertEqual(before, state_json(new))
                restored = bank_from_state(state_dict(new))
                self.assertEqual(expected, restored.predict_batch(probes))
                self.assertEqual(before, state_json(restored))

    def test_exact_tie_and_inclusive_threshold(self):
        bank = PrototypeBank(2, threshold=0)
        bank.initialize_known({2: [[1, 0]], 10: [[1, 0]]})
        self.assertEqual(bank.predict_batch([[0, 1], [1, 0], [-1, 0]]),
                         ['known-10', 'known-10', None])
        self.assertEqual(bank.predict_batch([]), [])

    def test_calibration_exact_record_including_candidates(self):
        rng = np.random.RandomState(5)
        with threadpool_limits(limits=1):
            for dim in (2, 32, 768):
                groups = {c: rng.normal(size=(7, dim)) for c in range(5)}
                self.assertEqual(reference.calibrate_known_base(groups, base_identity='b', cache_identity='c'),
                                 calibrate_known_base(groups, base_identity='b', cache_identity='c'))

    def test_synthetic_every_metric_and_saved_state(self):
        config = SyntheticConfig()
        with threadpool_limits(limits=1):
            actual, bank = run_experiment(config)
        with patch('online_gcd.experiment.PrototypeBank', reference.PrototypeBank), \
                patch('online_gcd.experiment.OnlineEvaluator', reference.OnlineEvaluator), threadpool_limits(limits=1):
            expected, old = run_experiment(config)
        self.assertEqual(expected, actual)
        self.assertEqual(state_json(old), state_json(bank))

    def test_fake_cache_every_session_final_metric_state_and_progress(self):
        with tempfile.TemporaryDirectory() as root:
            _, manifests, cache, _, _ = fixture_cache(root)
            logs = []
            actual, bank = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True, progress=logs.append)
            # Reference evaluator has no timing instrumentation; isolate timing only.
            class ReferenceEvaluator(reference.OnlineEvaluator):
                last_timing = {'probe_evaluation_seconds': 0., 'metric_computation_seconds': 0.}
            with patch('online_gcd.cub_runner.PrototypeBank', reference.PrototypeBank), \
                    patch('online_gcd.cub_runner.calibrate_known_base', reference.calibrate_known_base), \
                    patch('online_gcd.cub_runner.OnlineEvaluator', ReferenceEvaluator):
                expected, old = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True, progress=None)
            for result in (actual, expected):
                result.pop('performance')
                for row in result['sessions']:
                    timing = row.pop('performance')
                    self.assertGreaterEqual(timing['stream_assignment_seconds'], 0)
                    self.assertGreaterEqual(timing['probe_evaluation_seconds'], 0)
                    self.assertGreaterEqual(timing['metric_computation_seconds'], 0)
            self.assertEqual(actual, expected)
            self.assertEqual(state_json(bank), state_json(old))
            for text in ('Feature cache validation', 'Threshold calibration', 'Known prototype initialization',
                         'Session 5 end', 'Probe 5 end'):
                self.assertTrue(any(text in line for line in logs), text)


class CalibrationImportTests(unittest.TestCase):
    def setUp(self):
        from online_gcd.artifacts import read_json, seal
        self.record = read_json(Path(__file__).absolute().parents[1] / 'results/cub_frozen_dino_online_seed1.json')
        # Artificial calibration diagnostics, never presented as real run values.
        self.artifact = seal({'schema_version': 1, 'method': 'known_only_loo_balanced_accuracy_v0',
            'base_manifest_checksum': 'a' * 64, 'cache_checksum': self.record['cache_checksum'],
            'feature_dim': 768, 'base_features_sha256': 'b' * 64, 'positive_count': 1498,
            'negative_count': 1498, 'known_class_count': 100,
            'objective': '0.5 * (same_class_acceptance + different_class_rejection)',
            'tie_break': 'largest_threshold', 'threshold': .5, 'balanced_accuracy': 1.,
            'candidates': [{'threshold': .5, 'positive_accepted': 1498, 'negative_rejected': 1498,
                            'balanced_accuracy': 1.}]})

    def test_import_changes_only_calibration_and_preserves_unknowns(self):
        from online_gcd.result_record import import_calibration
        from copy import deepcopy
        before = deepcopy(self.record)
        imported = import_calibration(self.record, self.artifact)
        self.assertEqual(self.record, before)
        self.assertEqual(imported['calibration']['status'], 'imported')
        self.assertEqual(imported['calibration']['artifact'], self.artifact)
        self.assertEqual({k: v for k, v in imported.items() if k != 'calibration'},
                         {k: v for k, v in before.items() if k != 'calibration'})
        self.assertIsNone(before['calibration']['threshold'])
        self.assertIsNone(imported['provenance']['run_commit'])
        with self.assertRaises(ValueError):
            import_calibration(imported, self.artifact)

    def test_bad_checksum_identity_counts_threshold_and_objective_rejected(self):
        from online_gcd.artifacts import seal
        from online_gcd.result_record import import_calibration
        with self.assertRaises(ValueError):
            import_calibration(self.record, dict(self.artifact, threshold=.4))
        for key, value in [('cache_checksum', 'c' * 64), ('known_class_count', 99),
                           ('positive_count', 1497), ('threshold', .4), ('balanced_accuracy', .9),
                           ('method', 'other'), ('base_features_sha256', 'missing')]:
            artifact = {k: v for k, v in self.artifact.items() if k != 'checksum'}
            artifact[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                import_calibration(self.record, seal(artifact))


if __name__ == '__main__':
    unittest.main()
