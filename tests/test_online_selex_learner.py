from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from selex_fixtures import angle, seeded_bank
from reference_selex_euclidean import ReferenceBank, distribution, groups, prune
from online_gcd.prototype_bank import PrototypeBank, normalize
from online_gcd.selex_euclidean import SelexEuclideanBank, SelexOptions
from online_gcd.selex_graph import complete_link_groups, geometry_edges, prune_edges
from online_gcd.selex_math import geometry_score, mixture, squared_distances
from online_gcd.state import bank_from_state, load_state, save_state, state_dict, state_json


class EuclideanMathTests(unittest.TestCase):
    def test_hand_mixture_group_balancing_reranks_without_lowering_gate(self):
        bank = seeded_bank([angle(math.degrees(math.acos(.95))), angle(math.degrees(math.acos(.75))),
                            angle(math.degrees(math.acos(.945)))], threshold=.7)
        pair = ('novel-000001', 'novel-000002')
        bank.edges[pair] = (3., 3)
        bank.rebuild()
        self.assertEqual(bank.predict(angle(0)), 'novel-000003')
        ids = [p.cluster_id for p in bank.prototypes if not p.originated_known]
        distances = [.1, .5, .11]
        actual = mixture(ids, distances, bank._memberships)
        expected = distribution(ids, distances, bank._memberships)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-16)
        self.assertEqual(int(actual.argmax()), 2)
        before = state_json(bank)
        self.assertIsNone(bank.predict(-angle(0)))
        self.assertEqual(before, state_json(bank))

    def test_distances_blocks_and_scalar_oracle(self):
        rng = np.random.default_rng(4)
        matrix, x = rng.normal(size=(31, 768)), rng.normal(size=768)
        expected = squared_distances(matrix, x, reference=True)
        for block in (1, 3, 256):
            np.testing.assert_array_equal(squared_distances(matrix, x, block), expected)
        self.assertAlmostEqual(geometry_score(angle(0), angle(30)), math.cos(math.pi / 6))
        self.assertEqual(geometry_score(angle(0), -angle(0)), -1.)

    def test_pruning_exact_degree_bound_and_forgotten_edge(self):
        ids = [f'novel-{i:06d}' for i in range(1, 17)]
        edges = {(a, b): (.5, 1) for i, a in enumerate(ids) for b in ids[i + 1:]}
        kept = prune_edges(edges)
        self.assertEqual(kept, prune(edges))
        for cid in ids:
            self.assertLessEqual(sum(cid in pair for pair in kept), 8)
        bank = seeded_bank([angle(i * 3) for i in range(12)])
        bank.edges = {pair: (1., 1) for pair in edges if all(cid in bank._prototypes for cid in pair)}
        bank.observe('one', 1, angle(0))
        self.assertLessEqual(len(bank.edges), 4 * 12)
        self.assertGreater(bank.pruned_edge_count, 0)
        # Forgotten edge is reintroduced from a fresh count, not resurrected data.
        bank.edges.clear()
        pending = bank._pending_evidence(normalize(angle(2), 768))
        bank.observe('two', 1, angle(2))
        self.assertTrue(all(bank.edges[pair][1] == 1 for pair, _ in pending if pair in bank.edges))

    def test_complete_link_anti_chain_ties_and_group_bound(self):
        ids = [f'novel-{i:06d}' for i in range(1, 12)]
        scores = {(ids[i], ids[j]): 1. for i in range(len(ids)) for j in range(i + 1, len(ids))}
        creation = {cid: 1 for cid in ids}
        actual = complete_link_groups(ids, scores, creation)
        self.assertEqual(actual, groups(ids, scores, creation))
        self.assertTrue(all(len(group) <= 8 for group in actual))
        self.assertEqual(actual[0], tuple(ids[:8]))
        chain = {(ids[0], ids[1]): .9, (ids[1], ids[2]): .8}
        self.assertEqual(complete_link_groups(ids[:3], chain, creation), [tuple(ids[:2])])

    def test_random_graph_heap_matches_independent_reference(self):
        rng = np.random.default_rng(15)
        for n in (3, 9, 26):
            ids = [f'novel-{i:06d}' for i in range(n)]
            raw = {(a, b): (float(rng.random()), 1) for i, a in enumerate(ids) for b in ids[i + 1:]}
            scores = {pair: v[0] for pair, v in prune_edges(raw).items()}
            ages = {cid: int(rng.integers(1, 4)) for cid in ids}
            self.assertEqual(complete_link_groups(ids, scores, ages), groups(ids, scores, ages))

    def test_geometry_edges_match_full_reference_without_pair_matrix(self):
        rng = np.random.default_rng(27)
        matrix = np.stack([normalize(x, 768) for x in rng.normal(size=(24, 768))])
        ids = [f'novel-{i:06d}' for i in range(len(matrix))]
        expected = geometry_edges(ids, matrix, reference=True)
        for block in (1, 7, 256):
            self.assertEqual(geometry_edges(ids, matrix, block), expected)


class LearnerOrderingAndFixturesTests(unittest.TestCase):
    def test_disabled_lambda_and_no_hierarchy_exact_baseline_leaf_state(self):
        rng = np.random.default_rng(7)
        known = {0: [angle(0)], 1: [angle(90)]}
        for options in (SelexOptions(mixing=0), SelexOptions(hierarchy_enabled=False)):
            base = PrototypeBank(768, .7, .9)
            bank = SelexEuclideanBank(768, .7, structure_threshold=.5, options=options)
            base.initialize_known(known)
            bank.initialize_known(known)
            with threadpool_limits(limits=1):
                for i, x in enumerate(rng.normal(size=(45, 768))):
                    self.assertEqual(bank.observe(str(i), 1, x), base.observe(str(i), 1, x))
                self.assertEqual(bank.prototypes, base.prototypes)
                probes = rng.normal(size=(21, 768))
                self.assertEqual(bank.predict_batch(probes), base.predict_batch(probes))
                bank.rebuild()
                self.assertEqual(bank.prototypes, base.prototypes)

    def test_pre_update_evidence_and_emission_before_sample_update(self):
        bank = seeded_bank([angle(0), angle(12), angle(25)])
        expected = bank._pending_evidence(normalize(angle(8), 768))
        pre = bank.predict(angle(8))
        snapshot = deepcopy(bank.edges)
        cache = bank._cache_selected
        def audit(selected):
            self.assertEqual(bank.edges, snapshot)
            self.assertEqual(selected.cluster_id, pre)
            return cache(selected)
        with patch.object(bank, '_cache_selected', side_effect=audit):
            emitted = bank.observe('current', 2, angle(8))
        self.assertEqual(emitted, pre)
        for pair, value in expected:
            self.assertEqual(bank.edges[pair], (value, 1))

    def test_evidence_neighborhood_independent_of_gate_excludes_current_birth(self):
        bank=seeded_bank([angle(0),angle(1)],threshold=.99999)
        predicted=bank.observe('far-current',2,angle(-90))
        self.assertEqual(predicted,'novel-000003')
        self.assertEqual(set(bank.edges),{('novel-000001','novel-000002')})
        self.assertEqual(bank.edges[('novel-000001','novel-000002')][1],1)

    def test_close_subclasses_related_but_not_collapsed_and_no_leaf_retirement(self):
        bank = seeded_bank([angle(0), angle(8)], structure=.1, threshold=.9999)
        for i in range(8):
            bank.observe(str(i), 1, angle(4))
        ids = set(bank._prototypes)
        counts = sum(p.observation_count for p in bank.prototypes)
        report = bank.rebuild()
        self.assertTrue(bank.nodes)
        self.assertEqual(set(bank._prototypes), ids)
        self.assertEqual(sum(p.observation_count for p in bank.prototypes), counts)
        self.assertFalse(any(hasattr(bank, name) for name in ('aliases', 'retired_nodes', 'birth_origins')))
        self.assertTrue(all(len(members) <= 8 for members in bank.nodes.values()))
        self.assertEqual(bank.predict(angle(0)),'novel-000001')
        self.assertEqual(bank.predict(angle(8)),'novel-000002')

    def test_fragmented_class_associates_without_labels(self):
        bank = seeded_bank([angle(-2), angle(2)], threshold=.99999, structure=.1)
        for i in range(9):
            bank.observe(f'fragment-{i}', 1, angle((-1 if i % 2 else 1) * 2))
        bank.rebuild()
        self.assertTrue(any('novel-000001' in group and 'novel-000002' in group for group in bank.nodes.values()))
        self.assertGreaterEqual(len([p for p in bank.prototypes if not p.originated_known]), 2)

    def test_similar_different_classes_geometry_alone_cannot_create_main_structure(self):
        bank = seeded_bank([angle(0), angle(.1)], structure=.1)
        self.assertEqual(bank.rebuild()['active_structural_node_count'], 0)
        bank.edges[('novel-000001', 'novel-000002')] = (1.9, 2)
        self.assertEqual(bank.rebuild()['active_structural_node_count'], 0)
        geometry = seeded_bank([angle(0), angle(.1)], mode='geometry', geometry=.9)
        self.assertEqual(geometry.rebuild()['active_structural_node_count'], 1)
        self.assertFalse(geometry.edges)

    def test_known_leaf_is_singleton_but_unlabelled_ema_still_allowed(self):
        near_known=np.zeros(768)
        near_known[-1],near_known[0]=1.,.01
        bank = seeded_bank([near_known,angle(0)], structure=.1)
        known_before = bank._prototypes['known-0']
        feature = np.zeros(768)
        feature[-1], feature[0] = 1., -.05
        self.assertEqual(bank.observe('known-like', 2, feature), 'known-0')
        self.assertEqual(bank._prototypes['known-0'].observation_count, known_before.observation_count + 1)
        self.assertNotEqual(bank._prototypes['known-0'].vector, known_before.vector)
        bank.rebuild()
        self.assertTrue(all('known-0' not in group for group in bank.nodes.values()))

    def test_global_rng_adversarial_order_and_multiple_sessions(self):
        stream = [angle(i) for i in (70, -40, 10, 80, -80, 25, 12, -3)]
        expected = None
        for seed in (0, 994):
            np.random.seed(seed)
            bank = seeded_bank([angle(0), angle(5)], threshold=.999)
            emitted = []
            for i, x in enumerate(stream):
                emitted.append(bank.observe(str(i), 2 + i // 4, x))
                if i % 4 == 3:
                    bank.rebuild()
            snapshot = (emitted, state_json(bank))
            if expected is None:
                expected = snapshot
            self.assertEqual(snapshot, expected)

    def test_probes_reordered_repeated_scalar_and_batch_do_not_mutate(self):
        bank = seeded_bank([angle(0), angle(8), angle(30)])
        for i in range(6):
            bank.observe(str(i), 1, angle(4))
        bank.rebuild()
        probes = [angle(i) for i in (-30, 0, 3, 9, 50)]
        before = state_json(bank)
        expected = [bank.predict(x) for x in probes]
        for _ in range(3):
            for batch in (1, 3, 256):
                self.assertEqual(bank.predict_batch(probes, batch), expected)
            self.assertEqual(bank.predict_batch(probes[::-1]), expected[::-1])
        self.assertEqual(state_json(bank), before)

    def test_reference_predictions_vectors_counts_and_evidence(self):
        bank = seeded_bank([angle(0), angle(8), angle(26), angle(70)], threshold=.97, structure=.2)
        reference = ReferenceBank(768, .97, structure_threshold=.2)
        reference.__dict__.update(deepcopy(bank.__dict__))
        rng = np.random.default_rng(29)
        for i, theta in enumerate(rng.uniform(-25, 85, 35)):
            self.assertEqual(bank.observe(str(i), 2 + i // 10, angle(theta)),
                             reference.observe(str(i), 2 + i // 10, angle(theta)))
            self.assertEqual(bank.prototypes, reference.prototypes)
            self.assertEqual(bank.edges.keys(), reference.edges.keys())
            for pair in bank.edges:
                self.assertEqual(bank.edges[pair][1], reference.edges[pair][1])
                self.assertAlmostEqual(bank.edges[pair][0], reference.edges[pair][0], delta=2e-13)
            if i % 10 == 9:
                bank.rebuild()
                reference.rebuild()
                self.assertEqual(bank.nodes, reference.nodes)
        for x in [angle(i) for i in range(-20, 90, 3)]:
            self.assertEqual(bank.predict(x), reference.predict(x))


class NodeStateTests(unittest.TestCase):
    def structured(self):
        bank = seeded_bank([angle(0), angle(10), angle(50)])
        bank.edges[('novel-000001', 'novel-000002')] = (3., 3)
        bank.rebuild()
        return bank

    def test_active_only_id_reuse_retirement_and_reappearance(self):
        bank = self.structured()
        first = dict(bank.nodes)
        self.assertEqual(bank.rebuild()['created_node_ids'], [])
        self.assertEqual(bank.nodes, first)
        bank.edges.clear()
        self.assertEqual(bank.rebuild()['retired_node_ids'], list(first))
        self.assertFalse(bank.nodes)
        bank.edges[('novel-000001', 'novel-000002')] = (3., 3)
        bank.rebuild()
        self.assertEqual(list(bank.nodes), ['node-000002'])
        self.assertEqual(bank.next_node_id, 3)
        self.assertNotIn('retired_node_ids', state_dict(bank))

    def test_save_load_mid_session_and_boundary_exact_resume(self):
        bank = self.structured()
        for i in range(3):
            bank.observe(str(i), 2, angle(8 + i))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'learner.json'
            save_state(bank, path)
            loaded = load_state(path, 768)
        self.assertEqual(state_json(bank), state_json(loaded))
        for i in range(6):
            self.assertEqual(bank.observe(str(i + 3), 2 + i // 3, angle(i * 12)),
                             loaded.observe(str(i + 3), 2 + i // 3, angle(i * 12)))
            if i % 3 == 2:
                self.assertEqual(bank.rebuild()['created_node_ids'], loaded.rebuild()['created_node_ids'])
        self.assertEqual(state_json(bank), state_json(loaded))

    def test_malformed_node_cycle_missing_leaf_and_counter_rejected(self):
        saved = state_dict(self.structured())
        variants = []
        for members in [['novel-000001', 'node-000001'], ['novel-000001', 'novel-999999'],
                        ['novel-000001', 'novel-000001'], ['known-0', 'novel-000001'],
                        ['novel-000001']]:
            bad = deepcopy(saved)
            bad['nodes'][0]['members'] = members
            variants.append(bad)
        bad = deepcopy(saved)
        bad['nodes'] *= 2
        variants.append(bad)
        for key, value in [('next_node_id', 1), ('observation_steps', 999), ('feature_dim', 3),
                           ('retired_nodes', []), ('last_rebuild_session', 99), ('config_checksum', '0'*64)]:
            bad = deepcopy(saved)
            bad[key] = value
            variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad.keys()), self.assertRaises(ValueError):
                bank_from_state(bad)
        with self.assertRaisesRegex(ValueError, 'dimension'):
            bank_from_state(saved, 32)

    def test_nonfinite_and_invalid_edge_state_rejected(self):
        saved = state_dict(self.structured())
        for key, value in [('evidence_sum', float('nan')), ('evidence_sum', float('inf')),
                           ('evidence_sum', 4.), ('opportunities', True), ('opportunities', 0)]:
            bad = deepcopy(saved)
            bad['edges'][0][key] = value
            with self.assertRaises(ValueError):
                bank_from_state(bad)
        bad = deepcopy(saved)
        bad['prototypes'][0]['vector'] = list(bad['prototypes'][0]['vector'])
        bad['prototypes'][0]['vector'][0] = float('nan')
        with self.assertRaises(ValueError):
            bank_from_state(bad)
        bad = deepcopy(saved)
        bad['edges'] *= 2
        with self.assertRaises(ValueError):
            bank_from_state(bad)

    def test_frozen_options_method_identity_and_degree_validation(self):
        for kwargs in ({'mode':'unknown'}, {'mixing':.1}, {'hierarchy_enabled':1}, {'block_size':0},
                       {'mode':'geometry','mixing':0}):
            with self.assertRaises(ValueError):
                SelexOptions(**kwargs)
        with self.assertRaises(ValueError):
            SelexEuclideanBank(768,.8)
        with self.assertRaises(ValueError):
            SelexEuclideanBank(768,.8,structure_threshold=.5,geometry_threshold=.5)
        for kwargs in ({'options':{}},{'calibration_identities':[]},
                       {'calibration_identities':{'novel_labels':[1,2]}},
                       {'calibration_identities':{'assignment':'bad'}},{'_collect_only':1}):
            with self.assertRaises(ValueError):
                SelexEuclideanBank(768,.8,structure_threshold=.5,**kwargs)
        bad=state_dict(self.structured())
        bad['options']['max_degree']=8.0
        from online_gcd.artifacts import digest
        bad['config_checksum']=digest(bad['options'])
        with self.assertRaises(ValueError):
            bank_from_state(bad)
        saved=state_dict(self.structured())
        for key,value in [('method_version','another-method'),('calibration_identities',{'labels':[1,2]}),
                          ('next_node_id',True),('geometry_threshold',.5)]:
            bad=deepcopy(saved)
            bad[key]=value
            with self.assertRaises(ValueError):
                bank_from_state(bad)
        many=seeded_bank([angle(i) for i in range(11)])
        many.observation_steps=11
        bad=state_dict(many)
        bad['edges']=[{'members':['novel-000001',f'novel-{j:06d}'],
                       'evidence_sum':1.,'opportunities':1} for j in range(2,12)]
        with self.assertRaisesRegex(ValueError,'degree'):
            bank_from_state(bad)

    def test_state_contains_no_labels_samples_or_history_and_old_schemas_work(self):
        bank = self.structured()
        text = state_json(bank)
        for forbidden in ('birth_origin', 'probe_history', 'mapping', 'sample_id', 'class_id', 'retired_registry', 'aliases'):
            self.assertNotIn(forbidden, text)
        baseline = PrototypeBank(768)
        baseline.initialize_known({0: [angle(0)]})
        self.assertEqual(state_json(bank_from_state(state_dict(baseline))), state_json(baseline))
        from online_gcd.consolidation import ConsolidatingBank
        old = ConsolidatingBank(768, .8, .9, .7)
        old.initialize_known({0: [angle(0)]})
        self.assertEqual(state_json(bank_from_state(state_dict(old))), state_json(old))


if __name__ == '__main__':
    unittest.main()
