from dataclasses import asdict, replace
import random
import unittest

import numpy as np

from online_gcd.synthetic import SyntheticConfig, generate_synthetic


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.config = SyntheticConfig(feature_dim=8, num_known_classes=2, num_novel_classes=4,
                                      num_online_sessions=2, samples_per_class=6,
                                      base_samples_per_class=3, probe_samples_per_class=2)

    def test_seed_determinism_and_input_normalization(self):
        first, second = generate_synthetic(self.config), generate_synthetic(self.config)
        self.assertEqual(first.checksum, second.checksum)
        self.assertEqual(first.base, second.base)
        self.assertEqual(first.features, second.features)
        self.assertEqual(first.stream.to_json(), second.stream.to_json())
        other = generate_synthetic(replace(self.config, seed=2))
        self.assertNotEqual(first.checksum, other.checksum)
        self.assertNotEqual(first.base[0].feature, other.base[0].feature)
        for feature in [p.feature for p in first.base + first.probes] + list(first.features.values()):
            self.assertTrue(np.isfinite(feature).all())
            self.assertAlmostEqual(np.linalg.norm(feature), 1)

    def test_no_global_rng_mutation(self):
        py, numpy = random.getstate(), np.random.get_state()
        generate_synthetic(self.config)
        self.assertEqual(py, random.getstate())
        after = np.random.get_state()
        self.assertEqual(numpy[0], after[0])
        np.testing.assert_array_equal(numpy[1], after[1])
        self.assertEqual(numpy[2:], after[2:])

    def test_disjoint_populations_known_only_base_and_schedule(self):
        data = generate_synthetic(self.config)
        base = {p.sample_id for p in data.base}
        probes = {p.sample_id for p in data.probes}
        stream = set(data.features)
        self.assertFalse(base & stream or base & probes or probes & stream)
        self.assertEqual(len(base), 6)
        self.assertEqual(len(probes), 12)
        self.assertEqual(len(stream), 36)
        self.assertEqual({p.class_id for p in data.base}, set(data.known_classes))
        self.assertEqual(set(data.labelled_base_groups()), set(data.known_classes))
        schedule = dict(data.stream.class_introduction_schedule)
        flattened = []
        for t in range(1, 3):
            records = data.stream.evaluator_sessions[t - 1]
            self.assertTrue(set(data.known_classes) <= {p.class_id for p in records})
            for record in records:
                self.assertLessEqual(schedule.get(record.class_id, 1), t)
            for safe in data.learner_session(t):
                self.assertEqual(set(asdict(safe)), {'sample_id', 'session_id', 'feature'})
            flattened.extend(records)
        self.assertEqual(len({r.sample_id for r in flattened}), len(stream))
        for c, start in schedule.items():
            self.assertEqual({r.session_id for r in flattened if r.class_id == c}, set(range(start, 3)))

    def test_invalid_synthetic_config(self):
        for change in ({'feature_dim': 2}, {'num_novel_classes': 3}, {'samples_per_class': 1},
                       {'noise_std': -1}, {'noise_std': float('nan')}, {'seed': -1},
                       {'dataset': 'CUB-200-2011'}, {'base_samples_per_class': 0},
                       {'probe_samples_per_class': 0}, {'momentum': 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change)
        with self.assertRaises(ValueError):
            SyntheticConfig.from_dict({'seed': 1})


if __name__ == '__main__':
    unittest.main()
