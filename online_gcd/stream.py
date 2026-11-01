"""Deterministic metadata streams. Full manifests are evaluator-only artifacts."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import random
from typing import Optional


@dataclass(frozen=True)
class ProtocolConfig:
    protocol_name: str = 'online-selex-cub-v0'
    dataset: str = 'CUB-200-2011'
    use_ssb_splits: bool = True
    seed: int = 1
    num_known_classes: int = 100
    num_novel_classes: int = 100
    num_online_sessions: int = 5
    novel_classes_per_session: int = 20
    known_classes_present_each_session: bool = True
    allow_seen_novel_reappearance: bool = True
    reveal_total_novel_class_count: bool = False
    sample_ordering: str = 'deterministic_seeded_shuffle'
    sample_allocation: str = 'round_robin'
    memory_budget: int = 0
    encoder_update: str = 'frozen'
    replay_enabled: bool = False
    expose_labelled_known_membership: bool = False

    def __post_init__(self):
        for name in ('seed', 'memory_budget', 'num_known_classes',
                     'num_novel_classes', 'num_online_sessions', 'novel_classes_per_session'):
            value = getattr(self, name)
            minimum = 0 if name in ('seed', 'memory_budget') else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name in ('use_ssb_splits', 'known_classes_present_each_session',
                     'allow_seen_novel_reappearance', 'reveal_total_novel_class_count',
                     'replay_enabled', 'expose_labelled_known_membership'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if self.num_online_sessions * self.novel_classes_per_session != self.num_novel_classes:
            raise ValueError('num_online_sessions * novel_classes_per_session must equal num_novel_classes')
        if self.protocol_name != 'online-selex-cub-v0' or self.dataset != 'CUB-200-2011':
            raise ValueError('unsupported protocol_name or dataset')
        if not self.use_ssb_splits:
            raise ValueError('use_ssb_splits must be true; no replacement class split')
        if self.sample_ordering != 'deterministic_seeded_shuffle':
            raise ValueError('sample_ordering must be deterministic_seeded_shuffle')
        if self.sample_allocation not in ('round_robin', 'contiguous_chunks'):
            raise ValueError('sample_allocation must be round_robin or contiguous_chunks')
        if self.memory_budget != 0 or self.encoder_update != 'frozen' or self.replay_enabled:
            raise ValueError('Phase 1 requires memory_budget=0, encoder_update=frozen, replay_enabled=false')

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError('configuration must be a mapping')
        names = {field.name for field in fields(cls)}
        if set(values) != names:
            raise ValueError(f'configuration fields: missing={sorted(names - set(values))}, '
                             f'unknown={sorted(set(values) - names, key=str)}')
        return cls(**values)


def load_config(path):
    """PyYAML is needed only for loading the tracked configuration."""
    import yaml
    try:
        values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ValueError(f'invalid YAML configuration: {exc}') from exc
    return ProtocolConfig.from_dict(values)


def validate_class_split(known_classes, novel_classes, config, *, expected_cub=False):
    known, novel = tuple(known_classes), tuple(novel_classes)
    for name, classes in (('known', known), ('novel', novel)):
        if not classes or any(type(c) is not int or c < 0 for c in classes):
            raise ValueError(f'{name} classes must be non-empty non-negative integer IDs')
        if len(set(classes)) != len(classes):
            raise ValueError(f'{name} classes contain duplicate IDs')
    if set(known) & set(novel):
        raise ValueError('known and novel classes must be disjoint')
    if len(known) != config.num_known_classes or len(novel) != config.num_novel_classes:
        raise ValueError('class counts do not match configuration')
    if expected_cub and (len(known) != 100 or len(novel) != 100
                         or set(known) | set(novel) != set(range(200))):
        raise ValueError('CUB SSB requires 100 known and 100 novel classes covering IDs 0..199')
    return known, novel


def introduction_schedule(known_classes, novel_classes, config, *, expected_cub=False):
    """Return evaluator-only (class_id, session_id) pairs in introduction order."""
    _, novel = validate_class_split(known_classes, novel_classes, config, expected_cub=expected_cub)
    order = list(novel)  # Preserve SSB Hard + Medium + Easy input order before shuffling.
    random.Random(config.seed).shuffle(order)
    return tuple((class_id, index // config.novel_classes_per_session + 1)
                 for index, class_id in enumerate(order))


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    class_id: int
    path: Optional[str] = None
    is_labelled_known: bool = False


@dataclass(frozen=True)
class LearnerRecord:
    sample_id: str
    session_id: int


@dataclass(frozen=True)
class LabelledMembershipRecord(LearnerRecord):
    is_labelled_known: bool


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    session_id: int
    class_id: int
    path: Optional[str]
    is_labelled_known: bool


@dataclass(frozen=True)
class StreamManifest:
    """Privileged builder/evaluator state. Never pass this object to a learner."""
    config: ProtocolConfig
    class_introduction_schedule: tuple
    evaluator_sessions: tuple
    manifest_kind: str

    def learner_session(self, session_id):
        """Project only one session; the caller must enforce chronological delivery.

        Paths are intentionally withheld: CUB paths encode true class names.
        A trusted future adapter must resolve opaque IDs to images/features.
        """
        if type(session_id) is not int or not 1 <= session_id <= self.config.num_online_sessions:
            raise ValueError('session_id is outside the online session range')
        records = self.evaluator_sessions[session_id - 1]
        if self.config.expose_labelled_known_membership:
            return tuple(LabelledMembershipRecord(r.sample_id, r.session_id, r.is_labelled_known)
                         for r in records)
        return tuple(LearnerRecord(r.sample_id, r.session_id) for r in records)

    def _payload(self):
        return {
            'audience': 'evaluator_only',
            'manifest_kind': self.manifest_kind,
            'protocol_version': self.config.protocol_name,
            'seed': self.config.seed,
            'configuration': asdict(self.config),
            'class_introduction_schedule': [
                {'class_id': c, 'session_id': s} for c, s in self.class_introduction_schedule],
            'sessions': [
                {'session_id': index, 'sample_ids': [r.sample_id for r in records],
                 'evaluator_records': [asdict(r) for r in records]}
                for index, records in enumerate(self.evaluator_sessions, 1)],
        }

    @property
    def checksum(self):
        canonical = json.dumps(self._payload(), sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def to_json(self):
        return json.dumps(dict(self._payload(), manifest_checksum=self.checksum),
                          sort_keys=True, indent=2, ensure_ascii=True) + '\n'

    def export(self, path):
        """Write a new evaluator-only file; never overwrite an existing artifact."""
        with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(self.to_json())


def build_stream(records, known_classes, novel_classes, config, *, expected_cub=False):
    """Allocate each input sample exactly once across eligible sessions.

    None means schedule-only validation (not a real-data stream). Otherwise
    require metadata for every class and enough known samples for coverage.
    Input iteration order is canonicalized by stable sample ID within each class.
    """
    known, novel = validate_class_split(known_classes, novel_classes, config, expected_cub=expected_cub)
    schedule = introduction_schedule(known, novel, config, expected_cub=expected_cub)
    sessions = [[] for _ in range(config.num_online_sessions)]
    if records is None:
        return StreamManifest(config, schedule, tuple(tuple(s) for s in sessions), 'schedule_only')
    groups = {c: [] for c in known + novel}
    identifiers = set()
    known_set = set(known)
    for record in records:
        if not isinstance(record, SampleRecord):
            raise ValueError('records must contain SampleRecord values')
        if not isinstance(record.sample_id, str) or not record.sample_id.strip():
            raise ValueError('sample_id must be a non-empty opaque string')
        if record.sample_id in identifiers:
            raise ValueError(f'duplicate sample_id: {record.sample_id}')
        identifiers.add(record.sample_id)
        if type(record.class_id) is not int or record.class_id not in groups:
            raise ValueError(f'class_id outside the declared split: {record.class_id}')
        if record.path is not None and (not isinstance(record.path, str) or not record.path.strip()):
            raise ValueError('path must be a non-empty string or null')
        if type(record.is_labelled_known) is not bool:
            raise ValueError('is_labelled_known must be a boolean')
        if record.is_labelled_known and record.class_id not in known_set:
            raise ValueError('novel samples cannot belong to the labelled known subset')
        groups[record.class_id].append(record)
    intro = dict(schedule)
    rng = random.Random(config.seed)
    for class_id in known + novel:
        samples = sorted(groups[class_id], key=lambda r: r.sample_id)
        if not samples:
            raise ValueError(f'no samples for declared class {class_id}')
        start = intro.get(class_id, 1)
        stop = config.num_online_sessions
        if class_id in intro and not config.allow_seen_novel_reappearance:
            stop = start
        eligible = list(range(start, stop + 1))
        if (class_id in known_set and config.known_classes_present_each_session
                and len(samples) < len(eligible)):
            raise ValueError(f'known class {class_id} needs at least {len(eligible)} samples for session coverage')
        rng.shuffle(samples)
        if config.sample_allocation == 'round_robin':
            assignments = [eligible[i % len(eligible)] for i in range(len(samples))]
        else:
            quotient, remainder = divmod(len(samples), len(eligible))
            assignments = [s for i, s in enumerate(eligible)
                           for _ in range(quotient + (i < remainder))]
        for record, session_id in zip(samples, assignments):
            sessions[session_id - 1].append(EvaluationRecord(
                record.sample_id, session_id, record.class_id, record.path, record.is_labelled_known))
    for session in sessions:
        rng.shuffle(session)
    return StreamManifest(config, schedule, tuple(tuple(s) for s in sessions), 'sample_stream')
