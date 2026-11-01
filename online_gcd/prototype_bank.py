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
