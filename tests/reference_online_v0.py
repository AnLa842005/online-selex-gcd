"""Frozen pre-Phase-3C reference (85a9bca); deliberately unoptimized test oracle."""

"""CPU cosine prototypes. This module has no evaluator or stream-label access."""
from dataclasses import dataclass
import math

import numpy as np

BASELINE_VERSION = 'euclidean-prototype-v0'


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def real(value, name, lower, upper):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not lower <= value <= upper):
        raise ValueError(f'{name} must be finite and in [{lower}, {upper}]')
    return float(value)


def vector(value, dimension):
    try:
        array = np.asarray(value)
        if array.shape != (dimension,) or array.dtype.kind not in 'fiu':
            raise ValueError('feature must be a numeric vector of the configured dimension')
        array = array.astype(np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError('feature must be a numeric vector') from exc
    if not np.isfinite(array).all():
        raise ValueError('feature must contain finite values')
    if not np.any(array):
        raise ValueError('feature must be non-zero')
    return array


def normalize(value, dimension):
    array = vector(value, dimension)
    scaled = array / np.max(np.abs(array))
    return scaled / np.linalg.norm(scaled)


@dataclass(frozen=True)
class Prototype:
    cluster_id: str
    vector: tuple
    observation_count: int
    originated_known: bool
    creation_session: int
    last_updated_session: int


class PrototypeBank:
    def __init__(self, feature_dim, threshold=0.85, momentum=0.9):
        self.feature_dim = integer(feature_dim, 'feature_dim', 1)
        self.threshold = real(threshold, 'threshold', -1, 1)
        self.momentum = real(momentum, 'momentum', 0, 1)
        self._prototypes = {}
        self._next_novel_id = 1
        self._last_session = 0

    @property
    def prototypes(self):
        # Immutable snapshots, in the same stable order used for ties.
        return tuple(self._prototypes[key] for key in sorted(self._prototypes))

    @property
    def last_session(self):
        return self._last_session

    def initialize_known(self, labelled_groups):
        """Only the caller's explicitly labelled base-known features belong here."""
        if self._prototypes or self._last_session:
            raise ValueError('known initialization is allowed only on an empty base bank')
        if not isinstance(labelled_groups, dict) or not labelled_groups:
            raise ValueError('labelled_groups must be a non-empty mapping of known labels to features')
        for class_id in labelled_groups:
            integer(class_id, 'known class ID')
        prepared = {}
        for class_id in sorted(labelled_groups):
            samples = list(labelled_groups[class_id])
            if not samples:
                raise ValueError('each known class requires labelled base samples')
            normalized = [normalize(sample, self.feature_dim) for sample in samples]
            mean = normalize(np.mean(normalized, axis=0), self.feature_dim)
            cluster_id = f'known-{class_id}'
            prepared[cluster_id] = Prototype(cluster_id, tuple(mean), len(samples), True, 0, 0)
        self._prototypes = prepared

    def _nearest(self, normalized):
        if not self._prototypes:
            raise ValueError('initialize labelled known prototypes before inference')
        ordered = self.prototypes
        similarities = np.asarray([p.vector for p in ordered]) @ normalized
        index = int(np.argmax(similarities))  # First maximum in lexicographic ID order.
        return ordered[index], float(similarities[index])

    def predict(self, feature):
        """Read-only inference. Below-threshold probes are rejected (None)."""
        nearest, similarity = self._nearest(normalize(feature, self.feature_dim))
        return nearest.cluster_id if similarity >= self.threshold else None

    def observe(self, sample_id, session_id, feature):
        """Assign against the pre-update bank, then update; no labels accepted.

        The runner enforces exactly-once sample delivery. IDs/features are not
        retained in this bank; it stores only aggregate prototype state.
        """
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError('sample_id must be a non-empty opaque string')
        integer(session_id, 'session_id', 1)
        if session_id < self._last_session:
            raise ValueError('session_id cannot move backwards')
        observation = normalize(feature, self.feature_dim)
        nearest, similarity = self._nearest(observation)
        if similarity >= self.threshold:
            updated = normalize(self.momentum * np.asarray(nearest.vector)
                                + (1 - self.momentum) * observation, self.feature_dim)
            selected = Prototype(nearest.cluster_id, tuple(updated), nearest.observation_count + 1,
                                 nearest.originated_known, nearest.creation_session, session_id)
        else:
            cluster_id = f'novel-{self._next_novel_id:06d}'
            if cluster_id in self._prototypes:
                raise ValueError('duplicate cluster ID in prototype state')
            # For a new cluster old=observation, so normalized EMA is observation.
            selected = Prototype(cluster_id, tuple(observation), 1, False, session_id, session_id)
            self._next_novel_id += 1
        self._prototypes[selected.cluster_id] = selected
        self._last_session = session_id
        return selected.cluster_id

"""Known-labelled-base-only LOO calibration. No stream/probe API or label inputs."""
import numpy as np

from online_gcd.artifacts import digest, seal
from online_gcd.prototype_bank import normalize

METHOD = 'known_only_loo_balanced_accuracy_v0'


def calibrate_known_base(labelled_groups, *, base_identity, cache_identity):
    if len(labelled_groups) < 2:
        raise ValueError('calibration requires at least two known classes')
    groups = {}
    dimension = None
    for c in sorted(labelled_groups):
        if type(c) is not int or c < 0:
            raise ValueError('invalid known class label')
        values = np.asarray(labelled_groups[c])
        if values.ndim != 2 or len(values) < 2:
            raise ValueError(f'known class {c} requires at least two base samples for leave-one-out calibration')
        dimension = values.shape[1] if dimension is None else dimension
        groups[c] = np.stack([normalize(v, dimension) for v in values])
    sums = {c: values.sum(axis=0) for c, values in groups.items()}
    means = {c: normalize(total, dimension) for c, total in sums.items()}
    positive, negative = [], []
    for c, values in groups.items():
        other_means = np.stack([means[k] for k in sorted(means) if k != c])
        for x in values:
            loo = normalize(sums[c] - x, dimension)
            positive.append(float(np.clip(x @ loo, -1, 1)))
            negative.append(float(np.clip(np.max(other_means @ x), -1, 1)))
    pos, neg = np.sort(positive), np.sort(negative)
    boundaries = np.unique(np.r_[pos, neg, -1., 1.])
    after = np.nextafter(boundaries, np.inf)
    candidates = np.unique(np.r_[boundaries, after[after <= 1]])
    accepted = len(pos) - np.searchsorted(pos, candidates, side='left')
    rejected = np.searchsorted(neg, candidates, side='left')
    scores = accepted + rejected  # Both groups have equal sample counts.
    best = np.flatnonzero(scores == scores.max())[-1]  # Highest threshold on ties.
    return seal({'schema_version': 1, 'method': METHOD, 'base_manifest_checksum': base_identity,
                 'cache_checksum': cache_identity, 'feature_dim': dimension,
                 'base_features_sha256': digest({str(c): v.tolist() for c, v in groups.items()}),
                 'positive_count': len(pos), 'negative_count': len(neg), 'known_class_count': len(groups),
                 'objective': '0.5 * (same_class_acceptance + different_class_rejection)',
                 'tie_break': 'largest_threshold', 'threshold': float(candidates[best]),
                 'balanced_accuracy': float(scores[best] / (2 * len(pos))),
                 'candidates': [{'threshold': float(t), 'positive_accepted': int(a), 'negative_rejected': int(r),
                                 'balanced_accuracy': float((a + r) / (2 * len(pos)))}
                                for t, a, r in zip(candidates, accepted, rejected)]})

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
