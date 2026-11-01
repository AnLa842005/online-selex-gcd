import unittest

from online_gcd.evaluator import OnlineEvaluator
from online_gcd.metrics import clustering_report, fragmentation
from online_gcd.prototype_bank import PrototypeBank
from online_gcd.state import state_json
from online_gcd.stream import EvaluationRecord
from online_gcd.synthetic import LabelledEmbedding


class MetricTests(unittest.TestCase):
    def test_perfect_and_wrong_predictions(self):
        labels = [0, 0, 1, 1, 2, 2]
        perfect = clustering_report(labels, ['a', 'a', 'b', 'b', 'c', 'c'], {0})
        self.assertEqual(perfect['hungarian_accuracy'], {
            'all': {'value': 1.0, 'count': 6}, 'old': {'value': 1.0, 'count': 2},
            'new': {'value': 1.0, 'count': 4}})
        self.assertEqual(perfect['novel_nmi'], 1)
        self.assertEqual(perfect['novel_ari'], 1)
        wrong = clustering_report(labels, ['a'] * 6, {0})
        self.assertLess(wrong['hungarian_accuracy']['all']['value'], 1)
        self.assertLess(wrong['novel_nmi'], 1)
        self.assertLess(wrong['novel_ari'], 1)

    def test_old_new_slices_reuse_one_global_mapping(self):
        report = clustering_report([0, 0, 0, 1, 1], ['a', 'a', 'a', 'a', 'b'], {0})
        self.assertEqual(report['hungarian_accuracy']['old']['value'], 1)
        self.assertEqual(report['hungarian_accuracy']['new']['value'], 0.5)
        self.assertEqual(report['hungarian_accuracy']['all']['value'], 0.8)

    def test_fragmentation_counts_persistent_ids(self):
        report = fragmentation([0, 0, 1, 1, 1], ['a', 'a', 'b', 'c', 'd'])
        self.assertEqual(report['duplicate_cluster_count'], 2)
        self.assertEqual(report['mean_excess_clusters'], 1)
        self.assertEqual(report['fragmented_class_fraction'], 0.5)
        self.assertEqual(report['clusters_per_class'], {'0': 1, '1': 3})

    def test_empty_rejected_and_invalid_inputs(self):
        report = clustering_report([], [], {0})
        for value in report['hungarian_accuracy'].values():
            self.assertEqual(value, {'value': None, 'count': 0})
        self.assertIsNone(report['novel_nmi'])
        self.assertIsNone(report['fragmentation']['mean_excess_clusters'])
        report = clustering_report([0], [None], {0})
        self.assertEqual(report['hungarian_accuracy']['all']['value'], 0)
        self.assertEqual(report['rejection_count'], 1)
        self.assertIsNone(report['hungarian_accuracy']['new']['value'])
        for labels, predictions in (([0], []), ([0], [2]), ([-1], ['x'])):
            with self.assertRaises(ValueError):
                clustering_report(labels, predictions, {0})


class EvaluatorTests(unittest.TestCase):
    def record(self, sid, session, label):
        return EvaluationRecord(sid, session, label, None, False)

    def setup_evaluator(self, threshold=0.5, momentum=0.0, intro=2, probe=(1., 0.)):
        bank = PrototypeBank(2, threshold=threshold, momentum=momentum)
        bank.initialize_known({0: [[1, 0]]})
        evaluator = OnlineEvaluator([0], [(1, intro)], (
            LabelledEmbedding('probe-known', 0, probe), LabelledEmbedding('probe-novel', 1, (-1., 0.))))
        evaluator.evaluate_base(bank)
        return bank, evaluator

    def test_probe_is_read_only_excludes_future_and_retention_has_zero_forgetting(self):
        bank, evaluator = self.setup_evaluator()
        prediction = bank.observe('s1', 1, [1, 0])
        before = state_json(bank)
        row = evaluator.evaluate_session(1, [self.record('s1', 1, 0)], {'s1': prediction}, bank)
        self.assertEqual(state_json(bank), before)
        self.assertEqual(row['probe']['eligible_class_count'], 1)
        self.assertEqual(row['probe']['average_forgetting'], 0)
        self.assertEqual(row['probe']['identity_churn'], 0)
        self.assertEqual(row['probe']['forgetting_class_count'], 1)
        self.assertIsNone(row['probe']['fixed_mapping_accuracy']['new']['value'])

    def test_probe_drop_has_positive_forgetting(self):
        bank, evaluator = self.setup_evaluator()
        predictions = {'s1': bank.observe('s1', 1, [.8, .6]),
                       's2': bank.observe('s2', 1, [.28, .96])}
        row = evaluator.evaluate_session(1, [self.record('s1', 1, 0), self.record('s2', 1, 0)], predictions, bank)
        self.assertEqual(row['probe']['average_forgetting'], 1)
        self.assertEqual(row['probe']['fixed_mapping_accuracy']['all']['value'], 0)
        self.assertEqual(row['probe']['identity_churn'], 1)

    def test_forgetting_uses_best_previous_and_keeps_signed_improvement(self):
        bank, evaluator = self.setup_evaluator(intro=4, probe=(0., 1.))
        predictions = {'s1': bank.observe('s1', 1, [.8, .6]),
                       's2': bank.observe('s2', 1, [.28, .96])}
        row = evaluator.evaluate_session(1, [self.record('s1', 1, 0), self.record('s2', 1, 0)], predictions, bank)
        self.assertEqual(row['probe']['average_forgetting'], -1)
        self.assertEqual(row['probe']['average_forgetting_clipped'], 0)
        for t, feature in ((2, [.8, .6]), (3, [1, 0])):
            sid = f's{t + 1}'
            prediction = bank.observe(sid, t, feature)
            row = evaluator.evaluate_session(t, [self.record(sid, t, 0)], {sid: prediction}, bank)
        self.assertEqual(row['probe']['average_forgetting'], 1)

    def test_independent_matching_can_hide_identity_swaps_but_cumulative_does_not(self):
        bank, evaluator = self.setup_evaluator(intro=1, momentum=1)
        p0 = bank.observe('a', 1, [1, 0])
        p1 = bank.observe('b', 1, [-1, 0])
        first = evaluator.evaluate_session(1, [self.record('a', 1, 0), self.record('b', 1, 1)],
                                           {'a': p0, 'b': p1}, bank)
        bank.observe('c', 2, [1, 0])
        bank.observe('d', 2, [-1, 0])
        # Deliberately corrupted persistent predictions: rematching hides the swap.
        second = evaluator.evaluate_session(2, [self.record('c', 2, 0), self.record('d', 2, 1)],
                                            {'c': p1, 'd': p0}, bank)
        self.assertEqual(first['session_local']['hungarian_accuracy']['all']['value'], 1)
        self.assertEqual(second['session_local']['hungarian_accuracy']['all']['value'], 1)
        self.assertEqual(second['cumulative']['hungarian_accuracy']['all']['value'], 0.5)
        self.assertEqual(second['cumulative_fixed_mapping_accuracy']['all']['value'], 0.5)
        self.assertEqual(second['cumulative']['fragmentation']['duplicate_cluster_count'], 2)
        self.assertEqual(evaluator.mapping, {p0: 0, p1: 1})
        self.assertEqual(first['probe']['forgetting_class_count'], 1)  # Novel has no past yet.
        self.assertEqual(second['probe']['forgetting_class_count'], 2)

    def test_future_labels_duplicate_ids_and_session_order_are_rejected(self):
        bank, evaluator = self.setup_evaluator()
        prediction = bank.observe('s1', 1, [1, 0])
        for records, predictions, session in (
            ([self.record('s1', 1, 1)], {'s1': prediction}, 1),
            ([self.record('s1', 1, 0)], {'other': prediction}, 1),
            ([self.record('s1', 1, 0)] * 2, {'s1': prediction}, 1),
            ([self.record('probe-known', 1, 0)], {'probe-known': prediction}, 1),
            ([self.record('s1', 1, 0)], {'s1': 'missing-cluster'}, 1),
            ([self.record('s1', 1, 0)], {'s1': prediction}, 2),
        ):
            with self.assertRaises(ValueError):
                evaluator.evaluate_session(session, records, predictions, bank)


if __name__ == '__main__':
    unittest.main()
