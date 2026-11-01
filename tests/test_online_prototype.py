from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from online_gcd.prototype_bank import PrototypeBank, normalize
from online_gcd.state import bank_from_state, load_state, save_state, state_dict, state_json


class PrototypeTests(unittest.TestCase):
    def bank(self, threshold=0.8, momentum=0.9):
        bank = PrototypeBank(3, threshold, momentum)
        bank.initialize_known({7: [[2, 0, 0], [1, 0, 0]]})
        return bank

    def test_known_initialization_normalized_mean_and_metadata(self):
        bank = PrototypeBank(3)
        bank.initialize_known({7: [[2, 0, 0], [0, 5, 0]], 2: [[0, 0, 4]]})
        known = {p.cluster_id: p for p in bank.prototypes}
        np.testing.assert_allclose(known['known-7'].vector, np.array([1, 1, 0]) / np.sqrt(2))
        self.assertEqual(known['known-7'].observation_count, 2)
        self.assertTrue(all(p.originated_known and p.creation_session == p.last_updated_session == 0
                            for p in bank.prototypes))
        with self.assertRaisesRegex(ValueError, 'empty base'):
            bank.initialize_known({7: [[1, 0, 0]]})

    def test_near_attach_far_creation_monotonic_ids(self):
        bank = self.bank()
        self.assertEqual(bank.observe('a', 1, [1, 0.01, 0]), 'known-7')
        self.assertEqual(bank.observe('b', 1, [0, 1, 0]), 'novel-000001')
        self.assertEqual(bank.observe('c', 2, [0, 0, 1]), 'novel-000002')
        self.assertEqual(bank.observe('d', 2, [0, 1, 0]), 'novel-000001')
        self.assertEqual(len(bank.prototypes), 3)
        novel = next(p for p in bank.prototypes if p.cluster_id == 'novel-000001')
        self.assertEqual((novel.observation_count, novel.creation_session, novel.last_updated_session), (2, 1, 2))
        for p in bank.prototypes:
            self.assertAlmostEqual(np.linalg.norm(p.vector), 1)

    def test_ema_formula_and_counts(self):
        bank = self.bank(threshold=0.5, momentum=0.25)
        observation = np.array([0.8, 0.6, 0])
        expected = 0.25 * np.array([1, 0, 0]) + 0.75 * observation
        expected /= np.linalg.norm(expected)
        self.assertEqual(bank.observe('x', 3, observation), 'known-7')
        p, = bank.prototypes
        np.testing.assert_allclose(p.vector, expected, rtol=0, atol=1e-15)
        self.assertEqual((p.observation_count, p.creation_session, p.last_updated_session), (3, 0, 3))
        self.assertEqual(bank.last_session, 3)

    def test_ties_use_stable_id_order_and_threshold_is_inclusive(self):
        bank = PrototypeBank(3, threshold=0)
        bank.initialize_known({9: [[1, 0, 0]], 2: [[1, 0, 0]]})
        self.assertEqual(bank.predict([0, 1, 0]), 'known-2')
        self.assertEqual(bank.observe('tie', 1, [0, 1, 0]), 'known-2')

    def test_prediction_rejection_never_mutates(self):
        bank = self.bank()
        before = state_json(bank)
        self.assertEqual(bank.predict([1, 0, 0]), 'known-7')
        self.assertIsNone(bank.predict([-1, 0, 0]))
        self.assertEqual(state_json(bank), before)

    def test_invalid_vectors_and_session_input_do_not_mutate(self):
        bank = self.bank()
        bank.observe('first', 2, [1, 0, 0])
        before = state_json(bank)
        for feature in ([0, 0, 0], [1, 2], [[1, 0, 0]], [np.nan, 0, 1],
                        [np.inf, 0, 1], ['1', '0', '0'], [True, False, False]):
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                bank.observe('invalid', 2, feature)
            self.assertEqual(state_json(bank), before)
        for sid, session in (('', 2), ('x', 0), ('x', 1), ('x', True), ('x', 1.5)):
            with self.assertRaises(ValueError):
                bank.observe(sid, session, [1, 0, 0])
        self.assertEqual(state_json(bank), before)

    def test_invalid_parameters_and_initialization(self):
        for threshold in (-1.1, 1.1, np.nan, np.inf, True, '0.5'):
            with self.assertRaisesRegex(ValueError, 'threshold'):
                PrototypeBank(3, threshold=threshold)
        for momentum in (-0.1, 1.1, np.nan, True):
            with self.assertRaisesRegex(ValueError, 'momentum'):
                PrototypeBank(3, momentum=momentum)
        for dimension in (0, True, 1.2):
            with self.assertRaisesRegex(ValueError, 'feature_dim'):
                PrototypeBank(dimension)
        for groups in ({}, {1: []}, {True: [[1, 0, 0]]}, {1: [[1, 0, 0], [-1, 0, 0]]}):
            bank = PrototypeBank(3)
            with self.assertRaises(ValueError):
                bank.initialize_known(groups)
            self.assertEqual(bank.prototypes, ())
        with self.assertRaisesRegex(ValueError, 'initialize'):
            PrototypeBank(3).predict([1, 0, 0])

    def test_cancelling_ema_fails_atomically_and_large_finite_vectors_work(self):
        bank = self.bank(threshold=-1, momentum=0.5)
        before = state_json(bank)
        with self.assertRaisesRegex(ValueError, 'non-zero'):
            bank.observe('opposite', 1, [-1, 0, 0])
        self.assertEqual(state_json(bank), before)
        np.testing.assert_allclose(normalize([1e308, 1e308, 0], 3), [2**-0.5, 2**-0.5, 0])

    def test_no_merge_delete_or_replay_and_immutable_snapshots(self):
        bank = self.bank()
        for name in ('merge', 'split', 'delete', 'replay', 'replay_buffer'):
            self.assertFalse(hasattr(bank, name))
        before = {p.cluster_id for p in bank.prototypes}
        for i, feature in enumerate(([0, 1, 0], [0, 0, 1], [-1, 0, 0])):
            bank.observe(str(i), i + 1, feature)
            self.assertTrue(before <= {p.cluster_id for p in bank.prototypes})
            before = {p.cluster_id for p in bank.prototypes}
        self.assertNotIn('samples', state_dict(bank))
        with self.assertRaises(AttributeError):
            bank.prototypes[0].observation_count = 0


class LearnerStateTests(unittest.TestCase):
    def bank(self):
        bank = PrototypeBank(3)
        bank.initialize_known({5: [[1, 0, 0]]})
        bank.observe('opaque', 1, [0, 1, 0])
        return bank

    def test_roundtrip_exact_bytes_predictions_and_continuation(self):
        bank = self.bank()
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / 'first.json', Path(folder) / 'second.json'
            save_state(bank, first)
            loaded = load_state(first, expected_feature_dim=3)
            self.assertEqual(state_json(bank), state_json(loaded))
            for feature in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
                self.assertEqual(bank.predict(feature), loaded.predict(feature))
            self.assertEqual(bank.observe('next', 2, [0, 0, 1]), 'novel-000002')
            self.assertEqual(loaded.observe('next', 2, [0, 0, 1]), 'novel-000002')
            self.assertEqual(state_json(bank), state_json(loaded))
            save_state(loaded, second)
            self.assertNotIn(b'\r', second.read_bytes())
            with self.assertRaises(FileExistsError):
                save_state(loaded, first)
            with self.assertRaisesRegex(ValueError, 'dimension'):
                load_state(first, expected_feature_dim=4)

    def test_state_contains_only_learner_information(self):
        data = state_dict(self.bank())
        self.assertEqual(set(data), {'schema_version', 'baseline_version', 'feature_dim', 'threshold',
                                    'momentum', 'next_novel_id', 'last_session', 'prototypes'})
        for p in data['prototypes']:
            self.assertEqual(set(p), {'cluster_id', 'vector', 'observation_count', 'originated_known',
                                     'creation_session', 'last_updated_session'})
        self.assertNotIn('opaque', json.dumps(data))

    def test_malformed_state_rejected(self):
        original = json.loads(state_json(self.bank()))
        mutations = [
            lambda d: d.update(schema_version=True), lambda d: d.update(baseline_version='bad'),
            lambda d: d.update(evaluator_labels=[1]), lambda d: d.update(feature_dim=4),
            lambda d: d.update(next_novel_id=1), lambda d: d.update(next_novel_id=3),
            lambda d: d.update(last_session=0), lambda d: d.update(last_session=2),
            lambda d: d.update(prototypes=[]), lambda d: d['prototypes'].append(d['prototypes'][0]),
            lambda d: d['prototypes'][0].update(vector=[2, 0, 0]),
            lambda d: d['prototypes'][0].update(vector=[float('nan'), 0, 0]),
            lambda d: d['prototypes'][0].update(vector=[0, 0, 0]),
            lambda d: d['prototypes'][0].update(originated_known=1),
            lambda d: d['prototypes'][0].update(observation_count=0),
            lambda d: d['prototypes'][0].update(creation_session=1),
            lambda d: d['prototypes'][1].update(cluster_id='novel-000002'),
            lambda d: d['prototypes'][1].update(cluster_id='novel-0000001'),
            lambda d: d['prototypes'][1].update(creation_session=0),
            lambda d: d['prototypes'][1].update(last_updated_session=0),
            lambda d: d['prototypes'][1].update(hidden_class=10),
        ]
        for index, mutate in enumerate(mutations):
            data = deepcopy(original)
            mutate(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                bank_from_state(data)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.json'
            for content in ('{', '[]', '{"schema_version":1,"schema_version":1}', '{"value":NaN}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load_state(path)


if __name__ == '__main__':
    unittest.main()
