"""Schema/provenance checks for a small curated real-run report; no workloads."""
import json
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).absolute().parents[1]
RECORD = ROOT / 'results/cub_frozen_dino_consolidation_phase4a.json'


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('invalid JSON constant: ' + value)
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique, parse_constant=nonfinite)


class CuratedCUBResultTests(unittest.TestCase):
    def setUp(self):
        self.record = read_json(RECORD)
        self.reference = read_json(ROOT / 'results/cub_frozen_dino_online_seed1.json')

    def test_curated_schema_and_experiment_boundary(self):
        self.assertEqual(set(self.record), {'schema_version', 'record_kind', 'status', 'dataset', 'protocol',
            'seed', 'experiment', 'provenance', 'runtime', 'calibration_audit', 'control_off',
            'consolidation_on', 'interpretation', 'limitations'})
        self.assertIs(type(self.record['schema_version']), int)
        self.assertEqual(self.record['schema_version'], 1)
        self.assertEqual(self.record['record_kind'], 'curated_user_reported_phase4a_real_cub_control')
        self.assertEqual(self.record['status'], 'closed_negative_control')
        self.assertEqual(self.record['experiment'], 'frozen-dino-prototype-consolidation-v0')
        self.assertEqual(self.record['protocol'], self.reference['protocol'])
        self.assertEqual(self.record['dataset'], 'CUB-200-2011')
        self.assertIs(self.record['control_off']['consolidation_enabled'], False)
        self.assertIs(self.record['consolidation_on']['consolidation_enabled'], True)

    def test_control_off_exactly_reproduces_supplied_phase3b_metrics(self):
        summary = self.record['control_off']['summary']
        for key in ('final_session_accuracy', 'cumulative_accuracy', 'probe_accuracy'):
            self.assertEqual(set(summary[key]), {'all', 'old', 'new'})
            self.assertEqual(summary[key], {k: v['value'] for k, v in self.reference['summary'][key].items()})
        for key in ('average_online_accuracy', 'average_forgetting', 'active_prototype_count', 'discovered_novel_cluster_count'):
            self.assertEqual(summary[key], self.reference['summary'][key])

    def test_negative_control_has_no_claimed_accuracy_gain(self):
        off = self.record['control_off']['summary']
        on = self.record['consolidation_on']['summary']
        for key in ('final_session_accuracy', 'average_online_accuracy', 'probe_accuracy'):
            self.assertEqual(on[key], off[key])
        self.assertEqual(on['cumulative_accuracy']['all'], off['cumulative_accuracy']['all'])
        self.assertGreater(on['cumulative_accuracy']['old'], off['cumulative_accuracy']['old'])
        self.assertLess(on['cumulative_accuracy']['new'], off['cumulative_accuracy']['new'])
        self.assertIs(self.record['interpretation']['fragmentation_resolved'], False)
        self.assertIs(self.record['interpretation']['probe_new_improved'], False)
        self.assertIs(self.record['interpretation']['novel_probe_threshold_tuning_permitted'], False)

    def test_single_merge_bookkeeping_and_approximate_timing(self):
        sessions = self.record['consolidation_on']['sessions']
        self.assertEqual([r['session_id'] for r in sessions], [1, 2, 3, 4, 5])
        self.assertEqual([r['merge_count'] for r in sessions], [0, 0, 0, 1, 0])
        self.assertEqual([r['active_prototype_count'] for r in sessions], [315, 597, 938, 1374, 2065])
        self.assertEqual([r['consolidation_seconds_approx'] for r in sessions], [.025, .118, .345, .754, 1.781])
        on = self.record['consolidation_on']['summary']
        off = self.record['control_off']['summary']
        self.assertEqual(sum(r['merge_count'] for r in sessions), on['retired_cluster_count'])
        self.assertEqual(on['retired_cluster_count'], 1)
        self.assertEqual(on['ever_created_novel_cluster_count'] - on['retired_cluster_count'],
                         on['discovered_novel_cluster_count'])
        self.assertEqual(on['active_prototype_count'] - on['discovered_novel_cluster_count'],
                         self.reference['population']['known_classes'])
        self.assertEqual(off['active_prototype_count'] - on['active_prototype_count'], 2)
        self.assertEqual(off['discovered_novel_cluster_count'] - on['ever_created_novel_cluster_count'], 1)
        self.assertEqual(on['maximum_merge_chain_depth'], 1)
        self.assertEqual(self.record['control_off']['end_to_end_elapsed_seconds_approx'], 15.139)
        self.assertEqual(self.record['consolidation_on']['end_to_end_elapsed_seconds_approx'], 17.152)
        self.assertEqual(on['active_probe_fragmentation'], {'mean_excess_clusters': 4.5,
                         'duplicate_cluster_count': 900, 'fragmented_class_fraction': .93})

    def test_threshold_decimal_provenance_and_distinct_audit_claims(self):
        audit = self.record['calibration_audit']
        for name, decimal in [('assignment_threshold', '0.76694905900245858'),
                              ('consolidation_threshold', '0.84814506320210814')]:
            self.assertEqual(audit[name + '_reported_decimal'], decimal)
            self.assertEqual(audit[name], float(decimal))
            self.assertTrue(-1 <= audit[name] <= 1)
        self.assertNotEqual(audit['assignment_threshold'], audit['consolidation_threshold'])
        self.assertEqual(audit['reported_original_artifact_structural_import'], 'passed')
        self.assertEqual(audit['reported_original_artifact_strict_runtime_match'], 'failed')
        self.assertEqual(audit['reported_control_off_phase3b_metrics_match'], 'exact')
        self.assertEqual(audit['issue_status'], 'unresolved_numerical_runtime_compatibility_issue')
        self.assertIsNone(audit['cause'])
        self.assertEqual(audit['threshold_source'], 'runtime recalibration from labelled known base only')
        self.assertEqual(self.reference['calibration']['status'], 'pending_import')
        self.assertIsNone(self.reference['calibration']['threshold'])

    def test_unknowns_are_not_filled_from_inference(self):
        provenance = self.record['provenance']
        self.assertEqual(provenance['run_commit'], '64fc515894f30a0d48ce3f90294945408a5dd9c7')
        self.assertEqual(provenance['persisted_artifact_directory'], '/workspace/runs/cub_phase4a_control_20261102_062837')
        self.assertIsNone(provenance['raw_artifact_hashes'])
        self.assertIsNone(provenance['execution_timestamp_utc'])
        self.assertIsNone(provenance['cache_checksum'])
        self.assertIsNone(self.record['seed'])
        self.assertEqual(provenance['phase3b_reference_seed'], 1)
        self.assertIsNone(self.record['consolidation_on']['summary']['average_forgetting'])
        runtime = self.record['runtime']
        self.assertEqual(runtime['vcpus'], 8)
        self.assertEqual(runtime['python'], '3.13 virtual environment')
        self.assertEqual(runtime['cache_samples'], 11788)
        self.assertIs(runtime['pod_running_after_execution'], False)
        for key in ('exact_python_patch', 'numpy_version', 'blas_version'):
            self.assertIsNone(runtime[key])

    def test_summary_metrics_are_finite_fractions_and_counts_are_integers(self):
        for arm in ('control_off', 'consolidation_on'):
            summary = self.record[arm]['summary']
            for name in ('final_session_accuracy', 'cumulative_accuracy', 'probe_accuracy'):
                self.assertEqual(set(summary[name]), {'all', 'old', 'new'})
                for value in summary[name].values():
                    self.assertIs(type(value), float)
                    self.assertTrue(math.isfinite(value) and 0 <= value <= 1)
            for name in ('average_online_accuracy', 'average_forgetting'):
                value = summary[name]
                if value is not None:
                    self.assertTrue(type(value) is float and math.isfinite(value) and 0 <= value <= 1)
            for name in ('active_prototype_count', 'discovered_novel_cluster_count'):
                self.assertIs(type(summary[name]), int)
                self.assertGreater(summary[name], 0)

    def test_record_is_small_and_excludes_raw_state_and_per_class_maps(self):
        self.assertLess(RECORD.stat().st_size, 16 * 1024)
        forbidden = {'features', 'prototypes', 'aliases', 'raw_state', 'learner_state', 'per_class_accuracy',
                     'clusters_per_class', 'merge_history', 'source_records', 'predictions'}
        def walk(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(self.record)

    def test_task_document_relative_links_resolve(self):
        for name in ('CUB_PHASE4A_REAL_CONTROL_REPORT.md', 'ONLINE_SELEX_EUCLIDEAN_PHASE4B_SPEC.md',
                     'RESEARCH_ROADMAP.md', 'NOVEL_PROTOTYPE_CONSOLIDATION.md'):
            path = ROOT / 'docs' / name
            content = path.read_text(encoding='utf-8-sig')
            self.assertGreater(len(content), 200)
            for target in re.findall(r'\]\(([^)]+)\)', content):
                if '://' not in target and not target.startswith('#'):
                    with self.subTest(document=name, target=target):
                        self.assertTrue((path.parent / target.split('#')[0]).is_file())


if __name__ == '__main__':
    unittest.main()
