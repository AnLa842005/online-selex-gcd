"""Known-base calibration amendments and trusted fixture orchestration tests."""
import ast
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from online_cub_fixtures import fixture_cache
from selex_fixtures import angle, hard_fake_cache, seeded_bank
from reference_selex_euclidean import ReferenceBank
from online_gcd.artifacts import digest, file_sha256, verify
from online_gcd.clean_protocol import CleanConfig
from online_gcd.consolidation import ConsolidationOptions
from online_gcd.cub_runner import run_cub_baseline
from online_gcd.experiment import run_experiment
from online_gcd.prototype_bank import PrototypeBank
from online_gcd.selex_calibration import calibrate_selex_base, select_threshold
from online_gcd.selex_euclidean import EXPERIMENT, SelexEuclideanBank, SelexOptions, experiment_identity
from online_gcd.selex_runner import SelexEvaluator, run_selex_fake_cache, run_selex_synthetic, stable_result
from online_gcd.state import bank_from_state, state_dict, state_json
from online_gcd.synthetic import LabelledEmbedding, SyntheticConfig, generate_synthetic as generate_raw_synthetic
from online_gcd.selex_synthetic import generate_selex_synthetic as generate_synthetic

ROOT = Path(__file__).absolute().parents[1]


def base_fixture():
    return {c: [(f'opaque-{c}-{i:03}', angle(theta)) for i, theta in enumerate(
                ([0, 2] + [40, 42] * 10) if c == 0 else ([150, 152] + [190, 192] * 10))]
            for c in (0, 1)}


class ApprovedCalibrationTests(unittest.TestCase):
    def test_a1_all_births_external_origins_and_eligible_counts(self):
        base = base_fixture()
        labels = {sid: c for c, rows in base.items() for sid, _ in rows}
        captured, tracked = [], {}
        class AuditBank(SelexEuclideanBank):
            def _rebuild_cache(self):
                super()._rebuild_cache()
                if self._collect_only:
                    # Pseudo seed IDs are deterministic, labels remain external.
                    for cid in self._active_ids:
                        tracked.setdefault(cid, 0 if int(cid[6:]) <= 2 else 1)
            def observe(self, sid, session, feature):
                self_outer.assertEqual(session, 1)
                self_outer.assertFalse(self.nodes)
                emitted = super().observe(sid, session, feature)
                tracked.setdefault(emitted, labels[sid])
                captured[:] = [self]
                return emitted
            def rebuild(self, **kwargs):
                raise AssertionError('circular calibration rebuild')
        self_outer = self
        with patch('online_gcd.selex_calibration.SelexEuclideanBank', AuditBank):
            record = calibrate_selex_base(base, .99995, base_identity='b', cache_identity='c')
        bank = captured[0]
        expected = [0, 0]
        born_positive = 0
        for (a, b), (_, count) in bank.edges.items():
            if count >= 3:
                same = tracked[a] == tracked[b]
                expected[0 if same else 1] += 1
                born_positive += same and (int(a[6:]) > 4 or int(b[6:]) > 4)
        self.assertGreater(born_positive, 0)
        self.assertEqual((record['positive_count'], record['negative_count']), tuple(expected))
        self.assertGreater(record['born_pseudo_leaf_count'], 0)
        self.assertEqual(record['pseudo_session_count'], 1)
        self.assertEqual(record['pseudo_hierarchy_rebuilds'], 0)
        self.assertEqual(record['diagnostic_leaf_scope'], 'all_initial_and_born')
        self.assertFalse(record['birth_origin_labels_in_learner'])
        for forbidden in ('origins', 'birth_origins', 'labels', 'known_label'):
            self.assertNotIn(forbidden, bank.__dict__)
        with self.assertRaises(ValueError):
            state_dict(bank)
        verify(record)

    def test_pseudo_assignment_birth_and_ema_exactly_match_baseline(self):
        base=base_fixture()
        base={c:[(sid,x*(3.14159 if i%2 else 7.)) for i,(sid,x) in enumerate(rows)] for c,rows in base.items()}
        originals={sid:x for rows in base.values() for sid,x in rows}
        self_outer=self
        class LifecycleAudit(SelexEuclideanBank):
            def _rebuild_cache(self):
                super()._rebuild_cache()
                if self._collect_only and self.last_session==0:
                    reference=PrototypeBank(self.feature_dim,self.threshold,.9)
                    reference._prototypes=deepcopy(self._prototypes)
                    reference._next_novel_id=self._next_novel_id
                    reference._rebuild_cache()
                    references[id(self)]=reference
            def observe(self,sid,session,feature):
                np.testing.assert_array_equal(feature,originals[sid])
                predicted=super().observe(sid,session,feature)
                reference=references[id(self)]
                self_outer.assertEqual(predicted,reference.observe(sid,session,feature))
                self_outer.assertEqual(self.prototypes,reference.prototypes)
                return predicted
        references={}
        with patch('online_gcd.selex_calibration.SelexEuclideanBank',LifecycleAudit):
            calibrate_selex_base(base,.99995,base_identity='b',cache_identity='c')

    def test_geometry_pairs_all_leaves_no_opportunity_gate_or_temperature_score(self):
        record = calibrate_selex_base(base_fixture(), .99995, base_identity='b', cache_identity='c', mode='geometry')
        n = record['pseudo_leaf_count']
        self.assertEqual(record['positive_count'] + record['negative_count'], n * (n - 1) // 2)
        self.assertIsNone(record['diagnostic_opportunity_gate'])
        self.assertIn('geometry', record['method'])
        self.assertTrue(-1 <= record['threshold'] <= 1)
        self.assertEqual(record['born_pseudo_leaf_count'], 4)

    def test_input_permutation_rng_and_blocks_preserve_calibration(self):
        base = base_fixture()
        permuted = {c: rows[::-1] for c, rows in reversed(list(base.items()))}
        expected = None
        for seed, supplied in ((3, base), (999, permuted)):
            np.random.seed(seed)
            record = calibrate_selex_base(supplied, .99995, base_identity='b', cache_identity='c')
            if expected is None:
                expected = record
            self.assertEqual(record, expected)
        for block in (1, 7, 256):
            record = calibrate_selex_base(base, .99995, base_identity='b', cache_identity='c',
                                           mode='geometry', block_size=block)
            self.assertEqual(record['threshold'], calibrate_selex_base(base, .99995, base_identity='b',
                             cache_identity='c', mode='geometry')['threshold'])

    def test_exact_balanced_objective_and_largest_threshold_tie(self):
        result = select_threshold([.8, .9], [.1, .2, .3, .4], lower=0.)
        self.assertEqual(result['threshold'], .8)
        self.assertEqual(result['balanced_accuracy'], 1.)
        for row in result['candidates']:
            self.assertEqual(row['balanced_accuracy'], .5 * (row['positive_accepted'] / 2 + row['negative_rejected'] / 4))
        self.assertEqual(select_threshold([1.], [1.], lower=0.)['threshold'], 1.)

    def test_fail_without_eligible_diagnostics_or_base_samples(self):
        for pos, neg in (([], [.5]), ([.5], []), ([], [])):
            with self.assertRaisesRegex(ValueError, 'no fallback'):
                select_threshold(pos, neg, lower=0.)
        base = base_fixture()
        base[0] = base[0][:4]
        with self.assertRaisesRegex(ValueError, 'five'):
            calibrate_selex_base(base, .9, base_identity='b', cache_identity='c')
        # Known-base high-dimensional dispersion creates evidence starvation.
        config = SyntheticConfig(feature_dim=768, samples_per_class=5, probe_samples_per_class=1)
        data = generate_raw_synthetic(config)
        base = {c: [(p.sample_id, p.feature) for p in data.base if p.class_id == c] for c in data.known_classes}
        with self.assertRaisesRegex(ValueError, 'eligible positive and negative'):
            calibrate_selex_base(base, .85, base_identity='b', cache_identity='c')


class FixtureRunnerTests(unittest.TestCase):
    def config(self):
        return SyntheticConfig(feature_dim=768, noise_std=.005, samples_per_class=5, probe_samples_per_class=2)

    def test_disabled_default_existing_baseline_and_control_unchanged(self):
        config = self.config()
        original, old = run_experiment(config)
        disabled, bank = run_experiment(config, consolidation=ConsolidationOptions(False))
        self.assertEqual(original, disabled)
        self.assertEqual(state_json(old), state_json(bank))
        with tempfile.TemporaryDirectory() as root:
            _, m, c, _, _ = fixture_cache(root)
            before, bank1 = run_cub_baseline(m, c, CleanConfig(), allow_fixture=True, progress=None,
                                             consolidation=ConsolidationOptions(True))
            run_selex_fake_cache(m, c, CleanConfig())
            after, bank2 = run_cub_baseline(m, c, CleanConfig(), allow_fixture=True, progress=None,
                                            consolidation=ConsolidationOptions(True))
            self.assertEqual(stable_result(before), stable_result(after))
            self.assertEqual(state_json(bank1), state_json(bank2))

    def test_lambda_zero_and_no_hierarchy_match_baseline_every_metric_and_leaf_state(self):
        config = self.config()
        fixture = generate_synthetic(config)
        calibrated, _ = run_selex_synthetic(config)
        self.assertEqual(calibrated['assignment_threshold'],calibrated['assignment_calibration']['threshold'])
        with patch('online_gcd.experiment.generate_synthetic',return_value=fixture):
            base, bank0 = run_experiment(replace(config,threshold=calibrated['assignment_threshold']))
        for options in (SelexOptions(mixing=0), SelexOptions(hierarchy_enabled=False)):
            result, bank = run_selex_synthetic(config, options=options)
            self.assertEqual(bank.prototypes, bank0.prototypes)
            self.assertEqual(bank._next_novel_id, bank0._next_novel_id)
            for a, b in zip(base['sessions'], result['sessions']):
                for key in ('session_local', 'cumulative', 'cumulative_fixed_mapping_accuracy'):
                    self.assertEqual(a[key], b[key])
                probe = {k: v for k, v in b['probe'].items() if k not in ('active_probe_fragmentation', 'incorrect_collapse')}
                self.assertEqual(a['probe'], probe)
            self.assertNotEqual(result['experiment'], EXPERIMENT)

    def test_three_way_hard_cache_reports_failures_not_research_improvement(self):
        with tempfile.TemporaryDirectory() as root:
            _, m, c, _, _ = hard_fake_cache(root)
            cache_hash = file_sha256(c / 'features.npy')
            baseline, _ = run_cub_baseline(m, c, CleanConfig(), allow_fixture=True, progress=None)
            control, _ = run_cub_baseline(m, c, CleanConfig(), allow_fixture=True, progress=None,
                                          consolidation=ConsolidationOptions(True))
            method, bank = run_selex_fake_cache(m, c, CleanConfig())
            self.assertGreater(sum(row['merge_count'] for row in control['sessions']), 0)
            self.assertGreater(len(bank.nodes), 0)
            self.assertEqual(method['summary']['active_prototype_count'], baseline['summary']['active_prototype_count'])
            self.assertLess(control['summary']['active_prototype_count'], baseline['summary']['active_prototype_count'])
            self.assertLess(control['summary']['probe_accuracy']['all']['value'], baseline['summary']['probe_accuracy']['all']['value'])
            self.assertTrue(method['validation_only'])
            for row in method['sessions']:
                for field in ('raw_stream_accuracy', 'raw_historical_fragmentation', 'active_probe_fragmentation',
                              'structural_mixed_origin_group_count', 'incorrect_collapse', 'learner_state_json_bytes'):
                    self.assertIn(field, row)
            self.assertEqual(file_sha256(c / 'features.npy'), cache_hash)
            self.assertEqual(state_json(bank), state_json(bank_from_state(state_dict(bank))))

    def test_reference_end_to_end_per_session_metrics_nodes_and_leaf_state(self):
        with tempfile.TemporaryDirectory() as root:
            _, m, c, _, _ = fixture_cache(root)
            actual, bank = run_selex_fake_cache(m, c, CleanConfig())
            # Independent scalar scoring/evidence, full candidate hierarchy oracle.
            with patch('online_gcd.selex_runner.SelexEuclideanBank', ReferenceBank):
                expected, reference = run_selex_fake_cache(m, c, CleanConfig(), reference=True)
            self.assertEqual(actual['summary'], expected['summary'])
            for a, b in zip(actual['sessions'], expected['sessions']):
                for key in ('session_local', 'cumulative', 'probe', 'active_prototype_count',
                            'active_structural_node_count', 'created_node_ids', 'retired_node_ids'):
                    self.assertEqual(a[key], b[key])
            self.assertEqual(bank.prototypes, reference.prototypes)
            self.assertEqual(bank.nodes, reference.nodes)
            self.assertEqual(bank.next_node_id, reference.next_node_id)
            for pair in bank.edges:
                self.assertEqual(bank.edges[pair][1], reference.edges[pair][1])
                self.assertAlmostEqual(bank.edges[pair][0], reference.edges[pair][0], delta=2e-13)

    def test_future_metadata_removal_and_novel_label_permutation_leave_behavior_identical(self):
        config = self.config()
        data = generate_synthetic(config)
        # Permute novel labels only in evaluator records, retaining eligibility.
        intro = dict(data.stream.class_introduction_schedule)
        classes_by_session = defaultdict(list)
        for c, sid in intro.items():
            classes_by_session[sid].append(c)
        permutation = {c: other for classes in classes_by_session.values()
                       for c, other in zip(sorted(classes), sorted(classes)[::-1])}
        stream = replace(data.stream, evaluator_sessions=tuple(
            tuple(replace(row, class_id=permutation.get(row.class_id, row.class_id)) for row in session)
            for session in data.stream.evaluator_sessions))
        permuted = replace(data, stream=stream, probes=tuple(replace(p, class_id=permutation.get(p.class_id,p.class_id)) for p in data.probes))
        snapshots = []
        states = []
        for supplied in (data, permuted):
            captured = []
            with patch('online_gcd.selex_runner.generate_synthetic', return_value=supplied):
                result, bank = run_selex_synthetic(config, snapshot_callback=lambda sid, state: captured.append(state))
            snapshots.append(captured)
            states.append(state_json(bank))
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(states[0], states[1])
        # Learner can consume a prefix with no future schedule or metadata at all.
        full = bank_from_state(snapshots[0][0])
        prefix = bank_from_state(snapshots[0][0])
        for observation in data.learner_session(2):
            self.assertEqual(full.observe(observation.sample_id, 2, observation.feature),
                             prefix.observe(observation.sample_id, 2, observation.feature))
        full.rebuild()
        prefix.rebuild()
        self.assertEqual(state_json(full), state_json(prefix))
        self.assertEqual(state_dict(full), snapshots[0][1])
        first = data.stream.evaluator_sessions[0]
        only_prefix = replace(data, features={r.sample_id:data.features[r.sample_id] for r in first},
                              stream=replace(data.stream,evaluator_sessions=(first,)))
        with patch('online_gcd.selex_runner.generate_synthetic',return_value=only_prefix):
            _, prefix_only = run_selex_synthetic(replace(config,num_online_sessions=1))
        self.assertEqual(state_dict(prefix_only),snapshots[0][0])

    def test_changed_future_features_and_probes_do_not_change_base_calibration(self):
        config = self.config()
        original = generate_synthetic(config)
        changed = replace(original, features={sid: tuple(-x for x in value) for sid,value in original.features.items()},
                          probes=tuple(replace(p, feature=tuple(np.roll(p.feature,1))) for p in original.probes))
        artifacts = []
        base_states = []
        from online_gcd import selex_runner
        call = selex_runner._calibrate
        def audit(*args, **kwargs):
            bank, assignment, structure = call(*args, **kwargs)
            base_states.append(state_json(bank))
            return bank, assignment, structure
        for data in (original, changed):
            with patch('online_gcd.selex_runner.generate_synthetic', return_value=data), \
                    patch('online_gcd.selex_runner._calibrate', side_effect=audit):
                result, _ = run_selex_synthetic(config)
                artifacts.append((result['assignment_calibration'],result['structural_calibration']))
        self.assertEqual(artifacts[0], artifacts[1])
        self.assertEqual(base_states[0], base_states[1])

    def test_raw_metrics_rebuild_probe_order_and_label_free_api(self):
        from online_gcd.metrics import clustering_report
        events = []
        class Audited(SelexEuclideanBank):
            def rebuild(self, **kwargs):
                events.append('hierarchy')
                return super().rebuild(**kwargs)
            def predict_batch(self, *args, **kwargs):
                events.append('probe')
                return super().predict_batch(*args, **kwargs)
            def observe(self, sid, session, feature):
                self_outer.assertIs(type(sid), str)
                self_outer.assertIs(type(session), int)
                self_outer.assertEqual(np.asarray(feature).shape, (768,))
                return super().observe(sid, session, feature)
        self_outer = self
        def raw(*args, **kwargs):
            events.append('raw')
            return clustering_report(*args, **kwargs)
        with patch('online_gcd.selex_runner.SelexEuclideanBank', Audited), \
                patch('online_gcd.evaluator.clustering_report', side_effect=raw):
            run_selex_synthetic(self.config())
        self.assertEqual(events[0], 'probe')
        self.assertEqual(events[1:], ['raw','raw','hierarchy','probe'] * 5)

    def test_threads_and_blocks_preserve_metrics_ids_and_state(self):
        config = self.config()
        expected = None
        for threads, block in ((1, 1), (2, 7), (4, 256)):
            with threadpool_limits(limits=threads):
                result, bank = run_selex_synthetic(config, options=SelexOptions(block_size=block))
            state = state_dict(bank)
            # Execution block size is recorded, not a semantic learner change.
            state['options']['block_size'] = 256
            state['config_checksum'] = digest(state['options'])
            # Calibration records include execution settings, so compare their
            # diagnostic objective/threshold rather than provenance hashes here.
            state['calibration_identities'] = {}
            semantic = (result['summary'], state)
            if expected is None:
                expected = semantic
            self.assertEqual(semantic, expected)

    def test_geometry_mode_uses_own_calibration_and_no_temporal_state(self):
        with tempfile.TemporaryDirectory() as root:
            _, m, c, _, _ = fixture_cache(root)
            result, bank = run_selex_fake_cache(m, c, CleanConfig(), options=SelexOptions(mode='geometry'))
        self.assertIsNone(bank.structure_threshold)
        self.assertEqual(bank.geometry_threshold, result['structural_calibration']['threshold'])
        self.assertIsNone(result['structural_calibration']['diagnostic_opportunity_gate'])
        self.assertFalse(bank.edges)
        self.assertIn('geometry', result['experiment'])
        self.assertEqual(state_json(bank), state_json(bank_from_state(state_dict(bank))))

    def test_production_cache_kind_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle, m, c, features, provenance = fixture_cache(root)
            spoofed = deepcopy(provenance)
            spoofed['extraction']['kind'] = 'frozen_dino'
            with patch('online_gcd.feature_cache.load_cache', return_value=(features, spoofed)), \
                    self.assertRaisesRegex(ValueError, 'fixture caches only'):
                run_selex_fake_cache(m, c, CleanConfig())

    def test_cli_byte_stable_semantics_and_state_and_workload_guard(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root)/'one', Path(root)/'two']
            for path in paths:
                result = subprocess.run([sys.executable, 'scripts/run_selex_euclidean_fixtures.py', '--output-dir', str(path)],
                                        cwd=ROOT, capture_output=True,text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            for name in ('euclidean.json','euclidean_state.json'):
                self.assertEqual((paths[0]/name).read_bytes(),(paths[1]/name).read_bytes())
            self.assertNotEqual((paths[0]/'euclidean_performance.json').read_bytes(),
                                (paths[1]/'euclidean_performance.json').read_bytes())
        for variant in ('geometry','lambda-zero','no-hierarchy'):
            with tempfile.TemporaryDirectory() as root:
                result = subprocess.run([sys.executable,'scripts/run_selex_euclidean_fixtures.py','--variant',variant,
                                         '--output-dir',str(Path(root)/'run')],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
        # Reuse the actual established workload guards, not a weaker fake guard.
        tree = ast.parse((ROOT/'tests/test_online_experiment.py').read_text())
        code = next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)
                    and 'def guarded_import(' in n.value)
        with tempfile.TemporaryDirectory() as root:
            argv = ['scripts/run_selex_euclidean_fixtures.py','--output-dir',str(Path(root)/'guarded')]
            code = code.replace("sys.argv = ['scripts/run_synthetic_online_baseline.py']", 'sys.argv = ' + repr(argv))
            result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    unittest.main()
