"""CPU-only consolidation control tests. All embeddings/metadata are artificial."""
from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from online_cub_fixtures import fixture_cache, overfragmented_fixture_cache
from online_gcd.artifacts import file_sha256
from online_gcd.clean_protocol import CleanConfig
from online_gcd.consolidation import ConsolidatingBank, ConsolidationOptions, EXPERIMENT, pairwise_cosine
from online_gcd.consolidation_calibration import calibrate_consolidation
from online_gcd.consolidation_evaluator import ConsolidationEvaluator
from online_gcd.cub_runner import run_cub_baseline
from online_gcd.experiment import run_experiment
from online_gcd.metrics import fragmentation
from online_gcd.prototype_bank import Prototype, PrototypeBank, normalize
from online_gcd.state import bank_from_state, load_state, save_state, state_dict, state_json
from online_gcd.stream import EvaluationRecord
from online_gcd.synthetic import LabelledEmbedding, SyntheticConfig


def semantic(value):
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k != 'performance' and not k.endswith('_seconds')}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def bank_fixture(vectors, counts=None, creation=None, threshold=.9, block_size=2):
    dimension = len(vectors[0])
    bank = ConsolidatingBank(dimension, .99999, .9, threshold, block_size)
    bank.initialize_known({0: [np.eye(dimension)[-1]], 1: [-np.eye(dimension)[-1]]})
    counts = counts or [1] * len(vectors)
    creation = creation or [1] * len(vectors)
    for i, (value, count, session) in enumerate(zip(vectors, counts, creation), 1):
        cid = f'novel-{i:06d}'
        bank._prototypes[cid] = Prototype(cid, tuple(normalize(value, dimension)), count, False, session, session)
    bank._next_novel_id = len(vectors) + 1
    bank._last_session = max(creation)
    bank._rebuild_cache()
    return bank


def angle(degrees):
    t = np.deg2rad(degrees)
    return [np.cos(t), np.sin(t), 0.]


class MergeTests(unittest.TestCase):
    def test_close_merges_far_and_known_never_merge(self):
        bank = bank_fixture([angle(0), angle(5), angle(90), [0, 0, 1]])
        known = [p for p in bank.prototypes if p.originated_known]
        report = bank.consolidate()
        self.assertEqual(report['merge_count'], 1)
        self.assertEqual(report['novel_prototypes_before'], 4)
        self.assertEqual(report['novel_prototypes_after'], 3)
        self.assertEqual([p for p in bank.prototypes if p.originated_known], known)
        self.assertIn('novel-000004', bank._prototypes)  # identical to known-0, still novel
        self.assertEqual(bank.resolve_alias('novel-000002'), 'novel-000001')

    def test_weighted_centroid_counts_and_session_metadata(self):
        bank = bank_fixture([angle(0), angle(30)], counts=[2, 6], creation=[2, 1], threshold=.8)
        expected = normalize(2 * np.asarray(angle(0)) + 6 * np.asarray(angle(30)), 3)
        report = bank.consolidate()
        selected = bank._prototypes['novel-000002']
        np.testing.assert_allclose(selected.vector, expected, rtol=0, atol=1e-15)
        self.assertEqual(selected.observation_count, 8)
        self.assertEqual((selected.creation_session, selected.last_updated_session), (1, 2))
        self.assertFalse(selected.originated_known)
        self.assertEqual(report['merge_history'][0]['retired_id'], 'novel-000001')

    def test_ties_use_pair_age_then_ids_and_survivor_age(self):
        bank = bank_fixture([angle(0)] * 4, creation=[2, 1, 3, 1])
        report = bank.consolidate()
        self.assertEqual(report['merge_history'][0]['survivor_id'], 'novel-000002')
        self.assertEqual(report['merge_history'][0]['retired_id'], 'novel-000001')
        self.assertEqual([p.cluster_id for p in bank.prototypes if not p.originated_known], ['novel-000002'])

    def test_recomputation_prevents_single_link_chain_collapse(self):
        bank = bank_fixture([angle(0), angle(20), angle(40)], threshold=float(np.cos(np.deg2rad(25))))
        report = bank.consolidate()
        self.assertEqual(report['merge_count'], 1)
        self.assertEqual(report['novel_prototypes_after'], 2)
        self.assertEqual(sum(p.observation_count for p in bank.prototypes if not p.originated_known), 3)

    def test_exact_threshold_is_inclusive(self):
        vectors = [angle(0), angle(20)]
        threshold = float(pairwise_cosine(np.asarray(vectors[:1]), np.asarray(vectors[1:]))[0, 0])
        bank = bank_fixture(vectors, threshold=threshold)
        # Derive from the actual normalized vectors to avoid fixture normalization rounding.
        bank.consolidation_threshold = float(pairwise_cosine(
            np.asarray([bank._prototypes['novel-000001'].vector]),
            np.asarray([bank._prototypes['novel-000002'].vector]))[0, 0])
        self.assertEqual(bank.consolidate()['merge_count'], 1)

    def test_empty_novel_and_single_novel_are_noops(self):
        bank = ConsolidatingBank(3, .9, .9, .9)
        bank.initialize_known({0: [[1, 0, 0]]})
        before = state_json(bank)
        self.assertEqual(bank.consolidate()['merge_count'], 0)
        self.assertEqual(state_json(bank), before)
        bank = bank_fixture([angle(0)])
        before = state_json(bank)
        self.assertEqual(bank.consolidate()['merge_count'], 0)
        self.assertEqual(state_json(bank), before)

    def test_reference_optimized_and_blocks_exact(self):
        rng = np.random.RandomState(17)
        values = rng.normal(size=(27, 12))
        reference = bank_fixture(values, threshold=.2, block_size=4)
        report = reference.consolidate(implementation='reference')
        expected = state_dict(reference)
        for block in (1, 3, 16, 100):
            bank = bank_fixture(values, threshold=.2, block_size=block)
            actual = bank.consolidate()
            self.assertEqual(report['merge_history'], actual['merge_history'])
            state = state_dict(bank)
            state['consolidation_block_size'] = expected['consolidation_block_size']
            self.assertEqual(state, expected)

    def test_thread_counts_identical_merge_ids_and_state(self):
        values = np.random.RandomState(19).normal(size=(24, 768))
        expected = None
        for threads in (1, 2, 4):
            with threadpool_limits(limits=threads):
                bank = bank_fixture(values, threshold=0., block_size=7)
                bank.consolidate()
                if expected is None:
                    expected = state_json(bank)
                self.assertEqual(state_json(bank), expected)

    def test_optimized_does_not_allocate_full_pairwise_matrix(self):
        bank = bank_fixture(np.random.RandomState(3).normal(size=(31, 16)), threshold=1., block_size=4)
        shapes = []
        def audit(left, right):
            shapes.append((len(left), len(right)))
            return pairwise_cosine(left, right)
        with patch('online_gcd.consolidation.pairwise_cosine', side_effect=audit):
            bank.consolidate()
        self.assertTrue(shapes)
        self.assertTrue(all(a <= 4 and b <= 4 for a, b in shapes))

    def test_invalid_options_and_degenerate_centroid_fail(self):
        for enabled, block in ((1, 2), (True, 0), (False, True)):
            with self.assertRaises(ValueError):
                ConsolidationOptions(enabled, block)
        bank = bank_fixture([angle(0), [-1, 0, 0]], threshold=-1.)
        before = state_json(bank)
        with self.assertRaises(ValueError):
            bank.consolidate()
        self.assertEqual(before, state_json(bank))


class LineageTests(unittest.TestCase):
    def chain(self):
        bank = bank_fixture([angle(0), angle(15), angle(16)], threshold=.9)
        bank.consolidate()
        self.assertEqual(bank.max_merge_chain_depth(), 2)
        return bank

    def test_transitive_alias_resolution_and_path_compression(self):
        bank = self.chain()
        bank.aliases['novel-000003'] = 'novel-000002'
        self.assertEqual(bank.resolve_alias('novel-000003'), 'novel-000001')
        self.assertEqual(bank.aliases['novel-000003'], 'novel-000002')
        bank.resolve_alias('novel-000003', compress=True)
        self.assertEqual(bank.aliases['novel-000003'], 'novel-000001')
        self.assertEqual(bank.max_merge_chain_depth(), 2)

    def test_save_load_and_resumed_observations_preserve_ids_and_state(self):
        bank = self.chain()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'state.json'
            save_state(bank, path)
            restored = load_state(path, 3)
            self.assertEqual(state_json(bank), state_json(restored))
            self.assertEqual(bank.predict_batch([angle(10), angle(90)]), restored.predict_batch([angle(10), angle(90)]))
            for i, feature in enumerate((angle(10), angle(90), angle(91))):
                self.assertEqual(bank.observe(str(i), 2, feature), restored.observe(str(i), 2, feature))
            self.assertEqual(bank.consolidate()['merge_history'], restored.consolidate()['merge_history'])
            self.assertEqual(state_json(bank), state_json(restored))
            self.assertEqual(bank._next_novel_id, 7)  # No reuse of retired IDs.

    def test_cycles_missing_targets_and_invalid_state_fail(self):
        saved = state_dict(self.chain())
        for aliases in ({'novel-000002': 'novel-000003', 'novel-000003': 'novel-000002'},
                        {'novel-000002': 'novel-999999', 'novel-000003': 'novel-000001'},
                        {'novel-000002': 'known-0', 'novel-000003': 'novel-000001'},
                        {'novel-000001': 'novel-000002', 'novel-000003': 'novel-000001'}):
            broken = deepcopy(saved)
            broken['aliases'] = aliases
            with self.assertRaises(ValueError):
                bank_from_state(broken)
        for key, value in [('merge_history', []), ('next_novel_id', 3), ('unknown', True)]:
            broken = deepcopy(saved)
            broken[key] = value
            with self.assertRaises(ValueError):
                bank_from_state(broken)

    def test_live_alias_cycles_and_missing_targets_fail_explicitly(self):
        bank = self.chain()
        bank.aliases['novel-000002'] = 'novel-000003'
        bank.aliases['novel-000003'] = 'novel-000002'
        with self.assertRaisesRegex(ValueError, 'cycle'):
            bank.resolve_alias('novel-000002', compress=True)
        bank.aliases['novel-000002'] = 'novel-999999'
        with self.assertRaisesRegex(ValueError, 'missing'):
            bank.resolve_alias('novel-000002')

    def test_future_predictions_canonical_and_history_unchanged(self):
        bank = bank_fixture([angle(0), angle(5)], threshold=.9)
        bank.threshold = .9
        historical = ['novel-000001', 'novel-000002']
        original = list(historical)
        bank.consolidate()
        self.assertEqual(bank.predict(angle(5)), 'novel-000001')
        self.assertEqual(bank.observe('future', 2, angle(5)), 'novel-000001')
        self.assertEqual(historical, original)


class MergeCalibrationTests(unittest.TestCase):
    def groups(self):
        return {0: [[1, 0, .25, 0], [1, 0, -.25, 0]],
                1: [[0, 1, 0, .25], [0, 1, 0, -.25]]}

    def test_counts_objective_candidates_ties_and_repeatability(self):
        groups = self.groups()
        record = calibrate_consolidation(groups, base_identity='b', cache_identity='c')
        self.assertEqual(record['positive_count'], 2)
        self.assertEqual(record['negative_count'], 4)
        self.assertEqual(record['known_class_count'], 2)
        self.assertEqual(record['base_sample_count'], 4)
        best = max(record['candidates'], key=lambda r: (r['balanced_accuracy'], r['threshold']))
        self.assertEqual(record['threshold'], best['threshold'])
        self.assertEqual(record, calibrate_consolidation(groups, base_identity='b', cache_identity='c'))
        expected = (1 - .25**2) / (1 + .25**2)
        self.assertAlmostEqual(record['threshold'], expected)
        for row in record['candidates']:
            self.assertEqual(row['balanced_accuracy'], .5 * (row['positive_accepted']/2 + row['negative_rejected']/4))

    def test_insufficient_samples_classes_or_degenerate_halves_fail(self):
        for groups in ({0: [[1, 0], [1, 0]]}, {0: [[1, 0]], 1: [[0, 1], [0, 1]]},
                       {0: [[1, 0], [-1, 0], [0, 1], [0, 1]], 1: [[0, 1], [0, 1]]}):
            with self.assertRaises(ValueError):
                calibrate_consolidation(groups, base_identity='b', cache_identity='c')

    def test_label_boundary_and_blend_thresholds_are_separate(self):
        self.assertEqual(list(inspect.signature(calibrate_consolidation).parameters),
                         ['labelled_groups', 'base_identity', 'cache_identity'])
        bank = bank_fixture([angle(0), angle(5)])
        with self.assertRaises(TypeError):
            bank.consolidate(labels=[1, 2])
        threshold = bank.threshold
        bank.consolidate()
        self.assertEqual(bank.threshold, threshold)
        self.assertNotEqual(bank.threshold, bank.consolidation_threshold)


class IntegrationTests(unittest.TestCase):
    def test_explicitly_disabled_synthetic_is_bit_identical(self):
        with threadpool_limits(limits=1):
            original, old = run_experiment(SyntheticConfig())
            disabled, bank = run_experiment(SyntheticConfig(), consolidation=ConsolidationOptions(False, 3))
        self.assertEqual(original, disabled)
        self.assertEqual(state_json(old), state_json(bank))
        self.assertIs(type(bank), PrototypeBank)

    def test_controlled_overfragmentation_reduced_without_cross_class_merge(self):
        groups = MergeCalibrationTests().groups()
        calibration = calibrate_consolidation(groups, base_identity='b', cache_identity='c')
        bank = ConsolidatingBank(4, .99999, .9, calibration['threshold'], 2)
        bank.initialize_known(groups)
        probes = tuple(LabelledEmbedding('p' + str(c), c, np.eye(4)[c]) for c in range(4))
        evaluator = ConsolidationEvaluator((0, 1), ((2, 1), (3, 1)), probes)
        evaluator.evaluate_base(bank)
        values = [[.1, 0, 1, 0], [-.1, 0, 1, 0], [0, .1, 0, 1], [0, -.1, 0, 1]]
        labels = [2, 2, 3, 3]
        emitted = {str(i): bank.observe(str(i), 1, feature) for i, feature in enumerate(values)}
        frozen = dict(emitted)
        from online_gcd.metrics import clustering_report
        events = []
        original_merge, original_probe = bank.consolidate, bank.predict_batch
        def raw_metrics(*args, **kwargs):
            events.append('raw')
            return clustering_report(*args, **kwargs)
        def merge(**kwargs):
            events.append('merge')
            self.assertEqual(evaluator._predictions, list(frozen.values()))
            return original_merge(**kwargs)
        def probe(*args, **kwargs):
            events.append('probe')
            return original_probe(*args, **kwargs)
        with patch('online_gcd.evaluator.clustering_report', side_effect=raw_metrics), \
                patch.object(bank, 'consolidate', side_effect=merge), \
                patch.object(bank, 'predict_batch', side_effect=probe):
            report = evaluator.evaluate_session(1, [EvaluationRecord(str(i), 1, c, None, False)
                                                   for i, c in enumerate(labels)], emitted, bank)
        self.assertEqual(events, ['raw', 'raw', 'merge', 'probe'])
        self.assertEqual(emitted, frozen)
        self.assertEqual(evaluator._predictions, list(frozen.values()))
        self.assertEqual(report['merge_count'], 2)
        self.assertEqual(report['novel_prototypes_after'], 2)
        self.assertEqual(report['raw_stream_fragmentation']['duplicate_cluster_count'], 2)
        self.assertEqual(report['active_probe_fragmentation']['duplicate_cluster_count'], 0)
        self.assertEqual(report['consolidated_probe_accuracy']['new']['value'], 1.)
        self.assertNotEqual(bank.resolve_alias(emitted['0']), bank.resolve_alias(emitted['2']))
        self.assertEqual(report['raw_stream_accuracy'], report['session_local']['hungarian_accuracy'])

    def test_fake_cache_both_modes_calibration_boundary_and_no_probe_labels(self):
        with tempfile.TemporaryDirectory() as root:
            bundle, manifests, cache, features, _ = fixture_cache(root)
            base_hash = file_sha256(cache / 'features.npy')
            baseline, old = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True, progress=None)
            disabled, bank = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True, progress=None,
                                               consolidation=ConsolidationOptions(False))
            self.assertEqual(semantic(baseline), semantic(disabled))
            self.assertEqual(state_json(old), state_json(bank))
            calls = []
            def audit(groups, **kwargs):
                self.assertEqual(set(groups), {1, 5})
                for c in groups:
                    expected = [features[r['feature_row']] for r in bundle['base']['records'] if r['known_class_id'] == c]
                    np.testing.assert_array_equal(groups[c], expected)
                calls.append('calibration')
                return calibrate_consolidation(groups, **kwargs)
            logs = []
            with patch('online_gcd.consolidation_calibration.calibrate_consolidation', side_effect=audit):
                ablation, merged = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True,
                                                    progress=logs.append, consolidation=ConsolidationOptions(True, 3))
            self.assertEqual(calls, ['calibration'])
            self.assertEqual(ablation['experiment'], EXPERIMENT)
            self.assertEqual(ablation['input_kind'], 'fixture')
            self.assertEqual(ablation['calibration'], baseline['calibration'])
            self.assertEqual(ablation['assignment_threshold'], baseline['calibration']['threshold'])
            self.assertEqual(file_sha256(cache / 'features.npy'), base_hash)
            for report in ablation['sessions']:
                self.assertEqual(report['prototypes_before_consolidation'] - report['merge_count'],
                                 report['prototypes_after_consolidation'])
                self.assertEqual(report['raw_stream_accuracy'], report['session_local']['hungarian_accuracy'])
                self.assertIn('active_probe_fragmentation', report)
            self.assertTrue(any('Consolidation calibration end' in row for row in logs))
            self.assertTrue(any('Consolidation similarity end' in row for row in logs))
            self.assertTrue(any('Consolidation merge end' in row for row in logs))
            self.assertEqual(state_json(merged), state_json(bank_from_state(state_dict(merged))))

    def test_multi_session_merges_preserve_raw_history_and_probe_eligibility(self):
        groups = MergeCalibrationTests().groups()
        calibration = calibrate_consolidation(groups, base_identity='b', cache_identity='c')
        bank = ConsolidatingBank(4, .99999, .9, calibration['threshold'], 2)
        bank.initialize_known(groups)
        probes = tuple(LabelledEmbedding('p' + str(c), c, np.eye(4)[c]) for c in range(4))
        evaluator = ConsolidationEvaluator((0, 1), ((2, 1), (3, 2)), probes)
        evaluator.evaluate_base(bank)
        first = {'a': bank.observe('a', 1, [.1, 0, 1, 0]),
                 'b': bank.observe('b', 1, [-.1, 0, 1, 0])}
        report1 = evaluator.evaluate_session(1, [EvaluationRecord(sid, 1, 2, None, False) for sid in first], first, bank)
        snapshot = deepcopy(report1)
        self.assertEqual(report1['probe']['eligible_class_count'], 3)
        self.assertEqual(report1['merge_count'], 1)
        second = {sid: bank.observe(sid, 2, feature) for sid, feature in [
            ('c', [1, 0, 0, 0]), ('d', [.03, 0, 1, 0]),
            ('e', [0, .1, 0, 1]), ('f', [0, -.1, 0, 1]) ]}
        labels = (0, 2, 3, 3)
        report2 = evaluator.evaluate_session(2, [EvaluationRecord(sid, 2, c, None, False)
                                                for sid, c in zip(second, labels)], second, bank)
        self.assertEqual(evaluator._predictions, [*first.values(), *second.values()])
        self.assertEqual(report1, snapshot)
        self.assertEqual(report2['novel_prototypes_after'], 2)
        self.assertEqual(report2['retired_cluster_count'], 3)
        self.assertEqual(report2['raw_historical_fragmentation']['duplicate_cluster_count'], 3)
        self.assertEqual(report2['probe']['eligible_class_count'], 4)
        self.assertEqual(report2['active_probe_fragmentation']['duplicate_cluster_count'], 0)
        self.assertEqual(state_json(bank), state_json(bank_from_state(state_dict(bank))))

    def test_fake_cache_requested_thread_counts_preserve_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            _, manifests, cache, _, _ = fixture_cache(root)
            records = []
            states = []
            for threads in (1, 4):
                record, bank = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True,
                                                progress=None, consolidation=ConsolidationOptions(True, 3),
                                                blas_threads=threads)
                self.assertEqual(record['performance']['blas_threads'], 1)
                self.assertEqual(record['performance']['requested_blas_threads'], threads)
                records.append(semantic(record))
                states.append(state_json(bank))
            self.assertEqual(records[0], records[1])
            self.assertEqual(states[0], states[1])

    def test_overfragmented_fake_cache_reference_matches_optimized_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:
            _, manifests, cache, _, _ = overfragmented_fixture_cache(root)
            baseline, _ = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True, progress=None)
            optimized, bank = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True,
                                               progress=None, consolidation=ConsolidationOptions(True, 2))
            original = ConsolidatingBank.consolidate
            def reference(bank, **kwargs):
                return original(bank, implementation='reference', **kwargs)
            with patch.object(ConsolidatingBank, 'consolidate', reference):
                expected, reference_bank = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True,
                                                            progress=None, consolidation=ConsolidationOptions(True, 2))
            self.assertGreater(sum(row['merge_count'] for row in optimized['sessions']), 0)
            self.assertLess(optimized['summary']['active_prototype_count'], baseline['summary']['active_prototype_count'])
            self.assertEqual(semantic(expected), semantic(optimized))
            self.assertEqual(state_json(bank), state_json(reference_bank))
            self.assertEqual(state_json(bank), state_json(bank_from_state(state_dict(bank))))

    def test_stream_and_probe_changes_cannot_change_merge_calibration(self):
        from dataclasses import replace
        from online_gcd.synthetic import generate_synthetic
        config = SyntheticConfig(samples_per_class=5, probe_samples_per_class=2)
        data = generate_synthetic(config)
        changed = replace(data,
                          features={sid: tuple(-x for x in value) for sid, value in data.features.items()},
                          probes=tuple(replace(p, feature=tuple(np.roll(p.feature, 1))) for p in data.probes))
        records = []
        with threadpool_limits(limits=1):
            for supplied in (data, changed):
                with patch('online_gcd.experiment.generate_synthetic', return_value=supplied):
                    record, _ = run_experiment(config, consolidation=ConsolidationOptions(True, 3))
                records.append(record)
        self.assertNotEqual(records[0]['synthetic_checksum'], records[1]['synthetic_checksum'])
        self.assertEqual(records[0]['consolidation_calibration'], records[1]['consolidation_calibration'])

    def test_consolidation_cli_blocks_gpu_dataset_checkpoint_and_network_workloads(self):
        import ast
        import subprocess
        import sys
        root = Path(__file__).absolute().parents[1]
        tree = ast.parse((root / 'tests/test_online_experiment.py').read_text())
        code = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str) and 'def guarded_import(' in n.value)
        code = code.replace("sys.argv = ['scripts/run_synthetic_online_baseline.py']",
                            "sys.argv = ['scripts/run_synthetic_online_baseline.py', '--consolidate-novel', '--consolidation-block-size', '3']")
        result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(EXPERIMENT, result.stdout)
        self.assertIn('not Online SelEx', result.stdout)

    def test_same_seed_and_threads_preserve_all_metrics_and_history(self):
        config = SyntheticConfig(samples_per_class=10, probe_samples_per_class=2, threshold=.999)
        expected = None
        state = None
        for threads in (1, 2, 4):
            with threadpool_limits(limits=threads):
                actual, bank = run_experiment(config, consolidation=ConsolidationOptions(True, 3))
            if expected is None:
                expected, state = semantic(actual), state_json(bank)
            self.assertEqual(semantic(actual), expected)
            self.assertEqual(state_json(bank), state)
            self.assertEqual(actual['experiment'], EXPERIMENT)


if __name__ == '__main__':
    unittest.main()
