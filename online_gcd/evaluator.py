"""Privileged evaluation and frozen identity calibration; labels never enter the bank."""
import numpy as np

from online_gcd.metrics import accuracy_slices, clustering_report, hungarian_mapping, validate_predictions
from online_gcd.prototype_bank import integer
from online_gcd.state import state_json


class OnlineEvaluator:
    def __init__(self, known_classes, introduction_schedule, probes):
        self.known = set(known_classes)
        self.introduction = dict(introduction_schedule)
        if (not self.known or len(self.introduction) != len(introduction_schedule)
                or self.known & set(self.introduction)):
            raise ValueError('invalid evaluator class split or introduction schedule')
        for c in self.known | set(self.introduction):
            integer(c, 'class_id')
        for session_id in self.introduction.values():
            integer(session_id, 'introduction session', 1)
        self.probes = tuple(probes)
        if (len({p.sample_id for p in self.probes}) != len(self.probes)
                or {p.class_id for p in self.probes} != self.known | set(self.introduction)):
            raise ValueError('probes need unique IDs and coverage of exactly the declared classes')
        self.mapping = {f'known-{c}': c for c in sorted(self.known)}
        self._labels, self._predictions = [], []
        self._stream_ids = set()
        self._best_probe = {}
        self._previous_probe = {}
        self._last_session = None

    def _probe(self, bank, session_id):
        eligible = self.known | {c for c, t in self.introduction.items() if t <= session_id}
        records = [p for p in self.probes if p.class_id in eligible]
        before = state_json(bank)
        predictions = [bank.predict(p.feature) for p in records]
        if state_json(bank) != before:
            raise RuntimeError('probe prediction mutated learner state')
        labels = [p.class_id for p in records]
        accuracies, forgetting, churn = {}, [], []
        for c in sorted(eligible):
            pairs = [(p, z) for p, z in zip(records, predictions) if p.class_id == c]
            value = sum(z is not None and self.mapping.get(z) == c for _, z in pairs) / len(pairs)
            accuracies[str(c)] = value
            if c in self._best_probe:
                forgetting.append(self._best_probe[c] - value)
                churn.append(sum(self._previous_probe[p.sample_id] != z for p, z in pairs) / len(pairs))
            self._best_probe[c] = max(self._best_probe.get(c, value), value)
        self._previous_probe = {p.sample_id: z for p, z in zip(records, predictions)}
        return {'fixed_mapping_accuracy': accuracy_slices(labels, predictions, self.known, self.mapping),
                'per_class_accuracy': accuracies,
                'average_forgetting': float(np.mean(forgetting)) if forgetting else None,
                'average_forgetting_clipped': float(np.mean(np.maximum(forgetting, 0))) if forgetting else None,
                'forgetting_class_count': len(forgetting),
                'identity_churn': float(np.mean(churn)) if churn else None,
                'eligible_class_count': len(eligible)}

    def evaluate_base(self, bank):
        if self._last_session is not None or bank.last_session != 0:
            raise ValueError('base evaluation must occur once before stream processing')
        if {p.cluster_id for p in bank.prototypes} != set(self.mapping):
            raise ValueError('bank base IDs do not match evaluator known classes')
        report = self._probe(bank, 0)
        self._last_session = 0
        return report

    def evaluate_session(self, session_id, records, predictions_by_id, bank):
        integer(session_id, 'session_id', 1)
        if self._last_session is None or session_id != self._last_session + 1 or bank.last_session != session_id:
            raise ValueError('evaluate sessions consecutively after processing their observations')
        records = tuple(records)
        ids = [r.sample_id for r in records]
        if (not records or len(set(ids)) != len(ids) or set(ids) != set(predictions_by_id)
                or set(ids) & self._stream_ids or set(ids) & {p.sample_id for p in self.probes}):
            raise ValueError('session prediction IDs must match unique, disjoint stream records exactly')
        eligible = self.known | {c for c, t in self.introduction.items() if t <= session_id}
        if any(r.session_id != session_id or r.class_id not in eligible for r in records):
            raise ValueError('future or invalid evaluator observation')
        labels = [r.class_id for r in records]
        predictions = [predictions_by_id[r.sample_id] for r in records]
        validate_predictions(labels, predictions)
        active_ids = {q.cluster_id for q in bank.prototypes}
        if any(p is not None and p not in active_ids for p in predictions):
            raise ValueError('stream prediction ID is absent from the persistent bank')
        # Calibrate only with this session's stream, never probes or future labels.
        unused_classes = eligible - set(self.mapping.values())
        pairs = [(y, p) for y, p in zip(labels, predictions)
                 if y in unused_classes and p is not None and p not in self.mapping]
        if pairs:
            self.mapping.update(hungarian_mapping([y for y, _ in pairs], [p for _, p in pairs]))
        self._labels.extend(labels)
        self._predictions.extend(predictions)
        self._stream_ids.update(ids)
        report = {
            'session_id': session_id,
            'session_local': clustering_report(labels, predictions, self.known),
            'cumulative': clustering_report(self._labels, self._predictions, self.known),
            'cumulative_fixed_mapping_accuracy': accuracy_slices(
                self._labels, self._predictions, self.known, self.mapping),
            'probe': self._probe(bank, session_id),
            'active_prototype_count': len(bank.prototypes),
            'known_prototype_count': sum(p.originated_known for p in bank.prototypes),
            'discovered_novel_cluster_count': sum(not p.originated_known for p in bank.prototypes),
        }
        self._last_session = session_id
        return report
