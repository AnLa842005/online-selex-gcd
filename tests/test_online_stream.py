from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from online_gcd.stream import (
    ProtocolConfig, SampleRecord, build_stream, introduction_schedule, load_config,
    validate_class_split,
)

ROOT = Path(__file__).resolve().parents[1]


class OnlineStreamTests(unittest.TestCase):
    def setUp(self):
        self.config = ProtocolConfig(num_known_classes=2, num_novel_classes=6,
                                     num_online_sessions=3, novel_classes_per_session=2)
        self.known = [10, 20]
        self.novel = [30, 40, 50, 60, 70, 80]
        # Opaque IDs intentionally contain no class names; no files are opened.
        self.records = [SampleRecord(f'image-{index * 7 + j:03}', class_id,
                                     f'private/class-{class_id}/{j}.jpg', index < 2 and j == 0)
                        for index, class_id in enumerate(self.known + self.novel)
                        for j in range(7)]

    def build(self, config=None, records=None):
        return build_stream(self.records if records is None else records,
                            self.known, self.novel, config or self.config)

    def test_same_seed_and_input_permutation_are_identical(self):
        self.assertEqual(self.build().to_json(), self.build().to_json())
        self.assertEqual(self.build().to_json(), self.build(records=reversed(self.records)).to_json())

    def test_other_seed_changes_actual_schedule_or_observations(self):
        first, second = self.build(), self.build(replace(self.config, seed=2))
        self.assertNotEqual((first.class_introduction_schedule, first.evaluator_sessions),
                            (second.class_introduction_schedule, second.evaluator_sessions))

    def test_invalid_class_lists(self):
        for known, novel, message in (
            ([10, 30], self.novel, 'disjoint'), ([], self.novel, 'non-empty'),
            (self.known, [], 'non-empty'), ([10, 10], self.novel, 'duplicate'),
            (self.known, [30] * 6, 'duplicate'), ([10], self.novel, 'counts'),
            ([True, 20], self.novel, 'integer'),
        ):
            with self.subTest(known=known, novel=novel), self.assertRaisesRegex(ValueError, message):
                introduction_schedule(known, novel, self.config)

    def test_introduction_coverage_and_no_early_novel(self):
        stream = self.build()
        schedule = dict(stream.class_introduction_schedule)
        self.assertEqual(set(schedule), set(self.novel))
        self.assertEqual(len(stream.class_introduction_schedule), len(self.novel))
        self.assertEqual(Counter(schedule.values()), {1: 2, 2: 2, 3: 2})
        for class_id, intro in schedule.items():
            observed = {r.session_id for session in stream.evaluator_sessions
                        for r in session if r.class_id == class_id}
            self.assertEqual(observed, set(range(intro, 4)))
        for session_id, records in enumerate(stream.evaluator_sessions, 1):
            self.assertTrue(set(self.known) <= {r.class_id for r in records})
            self.assertTrue(all(schedule.get(r.class_id, 1) <= session_id for r in records))

    def test_each_original_sample_once_and_allocation_counts(self):
        stream = self.build()
        originals = {r.sample_id: r for r in self.records}
        flattened = [r for records in stream.evaluator_sessions for r in records]
        self.assertEqual(Counter(r.sample_id for r in flattened), Counter(originals.keys()))
        intro = dict(stream.class_introduction_schedule)
        for class_id in self.known + self.novel:
            eligible = list(range(intro.get(class_id, 1), 4))
            expected = Counter(eligible[j % len(eligible)] for j in range(7))
            self.assertEqual(Counter(r.session_id for r in flattened if r.class_id == class_id), expected)
        for record in flattened:
            original = originals[record.sample_id]
            self.assertEqual((record.class_id, record.path, record.is_labelled_known),
                             (original.class_id, original.path, original.is_labelled_known))

    def test_contiguous_policy_has_same_counts_but_other_assignments(self):
        first = self.build()
        second = self.build(replace(self.config, sample_allocation='contiguous_chunks'))
        def counts(stream):
            return Counter((r.class_id, r.session_id) for s in stream.evaluator_sessions for r in s)
        self.assertEqual(counts(first), counts(second))
        self.assertNotEqual(first.evaluator_sessions, second.evaluator_sessions)
        self.assertEqual(len({r.sample_id for s in second.evaluator_sessions for r in s}), len(self.records))

    def test_reappearance_can_be_disabled(self):
        stream = self.build(replace(self.config, allow_seen_novel_reappearance=False))
        schedule = dict(stream.class_introduction_schedule)
        for session in stream.evaluator_sessions:
            for record in session:
                if record.class_id in schedule:
                    self.assertEqual(record.session_id, schedule[record.class_id])

    def test_sparse_metadata_fails_known_coverage_or_uses_earliest_novel_session(self):
        sparse = [r for r in self.records if r.sample_id == 'image-000' or r.class_id != 10]
        with self.assertRaisesRegex(ValueError, 'needs at least 3'):
            self.build(records=sparse)
        self.build(replace(self.config, known_classes_present_each_session=False), sparse)
        sparse = [r for r in self.records if r.class_id != 30] + [SampleRecord('sparse', 30)]
        stream = self.build(records=sparse)
        observation, = [r for s in stream.evaluator_sessions for r in s if r.class_id == 30]
        self.assertEqual(observation.session_id, dict(stream.class_introduction_schedule)[30])

    def test_learner_projection_has_no_labels_paths_config_or_schedule(self):
        stream = self.build()
        for session_id in range(1, 4):
            safe = stream.learner_session(session_id)
            self.assertEqual([r.sample_id for r in safe],
                             [r.sample_id for r in stream.evaluator_sessions[session_id - 1]])
            for record in safe:
                self.assertEqual(set(asdict(record)), {'sample_id', 'session_id'})
                self.assertEqual(record.session_id, session_id)
        for invalid in (0, 4, True, '1'):
            with self.assertRaisesRegex(ValueError, 'session_id'):
                stream.learner_session(invalid)

    def test_explicit_membership_is_known_only_even_in_oracle_config(self):
        stream = self.build(replace(self.config, expose_labelled_known_membership=True,
                                    reveal_total_novel_class_count=True))
        self.assertTrue(any(r.is_labelled_known for s in stream.evaluator_sessions for r in s))
        for session_id in range(1, 4):
            for safe, evaluator in zip(stream.learner_session(session_id), stream.evaluator_sessions[session_id - 1]):
                self.assertEqual(set(asdict(safe)), {'sample_id', 'session_id', 'is_labelled_known'})
                self.assertEqual(safe.is_labelled_known, evaluator.is_labelled_known)
                if evaluator.class_id in self.novel:
                    self.assertFalse(safe.is_labelled_known)

    def test_stable_export_and_independent_checksum(self):
        stream = self.build()
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / 'a.json', Path(folder) / 'b.json'
            stream.export(first)
            self.build().export(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotIn(b'\r', first.read_bytes())
            payload = json.loads(first.read_text())
            checksum = payload.pop('manifest_checksum')
            canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
            self.assertEqual(checksum, hashlib.sha256(canonical.encode('utf-8')).hexdigest())
            self.assertEqual(payload['audience'], 'evaluator_only')
            self.assertEqual(payload['manifest_kind'], 'sample_stream')
            with self.assertRaises(FileExistsError):
                stream.export(first)

    def test_checksum_detects_observation_changes(self):
        changed = list(self.records)
        changed[0] = replace(changed[0], sample_id='changed')
        self.assertNotEqual(self.build().checksum, self.build(records=changed).checksum)
        changed[0] = replace(self.records[0], path='another-private-path')
        self.assertNotEqual(self.build().checksum, self.build(records=changed).checksum)

    def test_global_python_and_numpy_rng_unchanged(self):
        import numpy as np
        before_python, before_numpy = random.getstate(), np.random.get_state()
        self.build().to_json()
        self.assertEqual(random.getstate(), before_python)
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])

    def test_bad_config_fails_clearly(self):
        cases = {
            'seed': [-1, True, '1'], 'num_online_sessions': [0, 2, 1.5],
            'num_known_classes': [0], 'num_novel_classes': [0, 9],
            'novel_classes_per_session': [0, 3], 'use_ssb_splits': [False, 'true'],
            'sample_ordering': ['random'], 'sample_allocation': ['duplicate'],
            'memory_budget': [-1, 1], 'encoder_update': ['trainable'],
            'replay_enabled': [True, 'false'], 'protocol_name': ['v99'],
            'dataset': ['other'], 'known_classes_present_each_session': ['true'],
            'allow_seen_novel_reappearance': [0], 'reveal_total_novel_class_count': [1],
            'expose_labelled_known_membership': ['false'],
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{key: value})
        for bad in (None, [], {}, dict(asdict(self.config), typo=True)):
            with self.assertRaisesRegex(ValueError, 'configuration'):
                ProtocolConfig.from_dict(bad)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'invalid.yaml'
            path.write_text('seed: [', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'invalid YAML'):
                load_config(path)

    def test_bad_records_fail_clearly(self):
        bad = (
            (self.records + [self.records[0]], 'duplicate sample_id'),
            ([replace(self.records[0], sample_id='')] + self.records[1:], 'sample_id'),
            ([replace(self.records[0], class_id=999)] + self.records[1:], 'outside'),
            ([replace(self.records[0], is_labelled_known='yes')] + self.records[1:], 'boolean'),
            (self.records + [SampleRecord('leak', 30, is_labelled_known=True)], 'novel samples'),
            ([replace(self.records[0], path=42)] + self.records[1:], 'path'),
            ([r for r in self.records if r.class_id != 30], 'no samples'),
            ([], 'no samples'), ([{}], 'SampleRecord'),
        )
        for records, message in bad:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.build(records=records)

    def test_tracked_config_five_by_twenty_and_optional_cub_validation(self):
        config = load_config(ROOT / 'configs/online_selex/cub_5sessions_seed1.yaml')
        # Synthetic partition, not numeric SSB membership assumptions.
        known, novel = list(range(100)), list(range(100, 200))
        schedule = introduction_schedule(known, novel, config, expected_cub=True)
        self.assertEqual(len(schedule), 100)
        self.assertEqual(len(dict(schedule)), 100)
        self.assertEqual(Counter(s for _, s in schedule), {i: 20 for i in range(1, 6)})
        self.assertEqual(set(c for c, _ in schedule), set(novel))
        with self.assertRaisesRegex(ValueError, '0..199'):
            validate_class_split(known, list(range(200, 300)), config, expected_cub=True)
        records = [SampleRecord(f'id-{c * 5 + j}', c) for c in known + novel for j in range(5)]
        stream = build_stream(records, known, novel, config, expected_cub=True)
        self.assertEqual(len(stream.evaluator_sessions), 5)
        self.assertEqual(sum(map(len, stream.evaluator_sessions)), len(records))
        for c, intro in schedule:
            self.assertIn(c, {r.class_id for r in stream.evaluator_sessions[intro - 1]})

    def test_schedule_only_is_explicit_and_has_no_fabricated_samples(self):
        manifest = build_stream(None, self.known, self.novel, self.config)
        self.assertEqual(manifest.manifest_kind, 'schedule_only')
        self.assertEqual(manifest.evaluator_sessions, ((), (), ()))
        self.assertEqual(manifest.learner_session(1), ())

    def test_core_has_no_third_party_runtime_dependency(self):
        result = subprocess.run([sys.executable, '-S', '-c',
            'from online_gcd.stream import ProtocolConfig, build_stream; '
            'm = build_stream(None, list(range(100)), list(range(100, 200)), ProtocolConfig()); '
            'assert len(m.class_introduction_schedule) == 100'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
