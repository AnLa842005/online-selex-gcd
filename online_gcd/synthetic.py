"""Privileged synthetic builder; only learner_session() is delivered online."""
from dataclasses import dataclass, fields
import hashlib
import json

import numpy as np

from online_gcd.prototype_bank import BASELINE_VERSION, integer, normalize, real
from online_gcd.stream import SampleRecord, build_stream


@dataclass(frozen=True)
class SyntheticConfig:
    baseline_version: str = BASELINE_VERSION
    dataset: str = 'synthetic'
    protocol_name: str = 'online-selex-cub-v0'
    seed: int = 1
    feature_dim: int = 32
    num_known_classes: int = 3
    num_novel_classes: int = 10
    num_online_sessions: int = 5
    base_samples_per_class: int = 20
    samples_per_class: int = 30
    probe_samples_per_class: int = 12
    noise_std: float = 0.025
    threshold: float = 0.85
    momentum: float = 0.9

    def __post_init__(self):
        for name in ('seed', 'feature_dim', 'num_known_classes', 'num_novel_classes',
                     'num_online_sessions', 'base_samples_per_class', 'samples_per_class',
                     'probe_samples_per_class'):
            integer(getattr(self, name), name, 0 if name == 'seed' else 1)
        if self.num_novel_classes % self.num_online_sessions:
            raise ValueError('num_novel_classes must be divisible by num_online_sessions')
        if self.samples_per_class < self.num_online_sessions:
            raise ValueError('samples_per_class must cover every known session')
        if self.feature_dim < self.num_known_classes + self.num_novel_classes:
            raise ValueError('feature_dim must cover all orthogonal synthetic class centers')
        real(self.noise_std, 'noise_std', 0, 1)
        real(self.threshold, 'threshold', -1, 1)
        real(self.momentum, 'momentum', 0, 1)
        if (self.baseline_version != BASELINE_VERSION or self.dataset != 'synthetic'
                or self.protocol_name != 'online-selex-cub-v0'):
            raise ValueError('unsupported synthetic baseline, dataset or stream protocol')

    @classmethod
    def from_dict(cls, values):
        names = {f.name for f in fields(cls)}
        if not isinstance(values, dict) or set(values) != names:
            raise ValueError('synthetic configuration must contain exactly the documented fields')
        return cls(**values)

    # Reuse Phase 1 allocation with synthetic class IDs; no SSB/CUB file access.
    @property
    def novel_classes_per_session(self):
        return self.num_novel_classes // self.num_online_sessions

    known_classes_present_each_session = True
    allow_seen_novel_reappearance = True
    expose_labelled_known_membership = False
    sample_allocation = 'round_robin'


@dataclass(frozen=True)
class LabelledEmbedding:
    sample_id: str
    class_id: int
    feature: tuple


@dataclass(frozen=True)
class FeatureObservation:
    sample_id: str
    session_id: int
    feature: tuple


@dataclass(frozen=True)
class SyntheticData:
    """Builder/evaluator-only container. Never give this object to the bank."""
    base: tuple
    stream: object
    features: dict
    probes: tuple
    known_classes: tuple

    def labelled_base_groups(self):
        groups = {c: [] for c in self.known_classes}
        for sample in self.base:
            if sample.class_id not in groups:
                raise ValueError('novel label in base initialization')
            groups[sample.class_id].append(sample.feature)
        return groups

    def learner_session(self, session_id):
        return tuple(FeatureObservation(r.sample_id, r.session_id, self.features[r.sample_id])
                     for r in self.stream.learner_session(session_id))

    @property
    def checksum(self):
        from dataclasses import asdict
        payload = {'base': [asdict(r) for r in self.base], 'probes': [asdict(r) for r in self.probes],
                   'features': self.features, 'stream_checksum': self.stream.checksum}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode('utf-8')).hexdigest()


def generate_synthetic(config):
    rng = np.random.default_rng(config.seed)
    num_classes = config.num_known_classes + config.num_novel_classes
    # QR gives well-separated random unit centers; fix column signs deterministically.
    q, r = np.linalg.qr(rng.normal(size=(config.feature_dim, num_classes)))
    centers = (q * np.where(np.diag(r) < 0, -1, 1)).T
    known = tuple(range(config.num_known_classes))
    novel = tuple(range(config.num_known_classes, num_classes))
    base, probes, metadata, features = [], [], [], {}
    used_ids = set()

    def sample(prefix, class_id):
        # Random opaque IDs avoid a class-major integer ID leaking class identity.
        sid = f'{prefix}-{rng.bytes(16).hex()}'
        if sid in used_ids:
            raise ValueError('duplicate generated sample ID')
        used_ids.add(sid)
        feature = tuple(normalize(centers[class_id] + config.noise_std * rng.normal(
            size=config.feature_dim), config.feature_dim))
        return LabelledEmbedding(sid, class_id, feature)

    for class_id in known:
        base.extend(sample('base', class_id) for _ in range(config.base_samples_per_class))
    for class_id in known + novel:
        for _ in range(config.samples_per_class):
            observation = sample('stream', class_id)
            metadata.append(SampleRecord(observation.sample_id, class_id))
            features[observation.sample_id] = observation.feature
        probes.extend(sample('probe', class_id) for _ in range(config.probe_samples_per_class))
    return SyntheticData(tuple(base), build_stream(metadata, known, novel, config),
                         features, tuple(probes), known)
