"""Metadata-only CUB partitions, preserving original SSB and labelled selection."""
from pathlib import Path, PurePosixPath
import os
import tempfile

import numpy as np

from online_gcd.artifacts import digest, read_json, seal, verify, write_json
from online_gcd.clean_protocol import PROTOCOL
from online_gcd.ssb import CUB_SPLIT_SHA256, load_cub_ssb_classes
from online_gcd.stream import ProtocolConfig, SampleRecord, build_stream

METADATA_FILES = ('images.txt', 'image_class_labels.txt', 'train_test_split.txt')
VIEW_NAMES = ('base', 'stream_learner', 'stream_evaluator', 'probe_evaluator')


def safe_relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('image path must remain relative to the CUB images directory')
    return value


def read_metadata(cub_root):
    """cub_root contains images.txt directly; image contents are never opened."""
    tables, hashes = {}, {}
    for name in METADATA_FILES:
        path = Path(cub_root) / name
        content = path.read_bytes()
        import hashlib
        hashes[name] = hashlib.sha256(content).hexdigest()
        rows = {}
        for line in content.decode('utf-8').splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f'malformed CUB metadata row in {name}')
            image_id = int(parts[0])
            if image_id <= 0 or image_id in rows:
                raise ValueError(f'duplicate/invalid image ID in {name}')
            rows[image_id] = parts[1]
        tables[name] = rows
    images, labels, splits = (tables[name] for name in METADATA_FILES)
    if not images or set(images) != set(labels) or set(images) != set(splits):
        raise ValueError('CUB metadata tables must have identical non-empty image ID sets')
    records = []
    for image_id, path in images.items():  # Preserve images.txt order, like the pandas joins.
        label, split = int(labels[image_id]), int(splits[image_id])
        if label <= 0 or split not in (0, 1):
            raise ValueError('invalid official CUB class or train/test flag')
        records.append({'image_id': image_id, 'path': safe_relative_path(path),
                        'class_id': label - 1, 'official_split': 'train' if split else 'test'})
    return records, hashes


def labelled_known_indices(count):
    # Exact data.data_utils.subsample_instances algorithm, local legacy RNG so
    # metadata building does not mutate global state or import torch/datasets.
    return np.random.RandomState(0).choice(range(count), replace=False, size=int(0.5 * count))


def build_bundle(records, metadata_hashes, known, novel, config, split_hash=CUB_SPLIT_SHA256):
    """Generic internal builder supports tiny metadata fixtures; CLI pins real SSB."""
    known, novel = tuple(known), tuple(novel)
    stream_config = ProtocolConfig(seed=config.seed, num_known_classes=len(known),
                                   num_novel_classes=len(novel), novel_classes_per_session=len(novel) // 5)
    ids, paths = set(), set()
    for r in records:
        if set(r) != {'image_id', 'path', 'class_id', 'official_split'}:
            raise ValueError('invalid source metadata fields')
        if type(r['image_id']) is not int or r['image_id'] <= 0 or r['image_id'] in ids:
            raise ValueError('duplicate/invalid source image ID')
        ids.add(r['image_id'])
        safe_relative_path(r['path'])
        if r['path'] in paths:
            raise ValueError('duplicate image path')
        paths.add(r['path'])
        if type(r['class_id']) is not int or r['class_id'] not in known + novel or r['official_split'] not in ('train', 'test'):
            raise ValueError('source class/split outside protocol')
    known_train = [r for r in records if r['official_split'] == 'train' and r['class_id'] in known]
    selected = {known_train[i]['image_id'] for i in labelled_known_indices(len(known_train))}
    # Opaque ID and class-independent row permutation; do not expose official
    # class-major image numbering or class-bearing paths via learner records.
    sample_ids = {r['image_id']: 'cub-' + digest(['CUB-200-2011', r['image_id']]) for r in records}
    rows = {sid: i for i, sid in enumerate(sorted(sample_ids.values()))}
    train = [r for r in records if r['official_split'] == 'train' and r['image_id'] not in selected]
    stream = build_stream([SampleRecord(sample_ids[r['image_id']], r['class_id']) for r in train],
                          known, novel, stream_config)
    schedule = dict(stream.class_introduction_schedule)
    base = [{'sample_id': sample_ids[r['image_id']], 'feature_row': rows[sample_ids[r['image_id']]],
             'known_class_id': r['class_id']} for r in records if r['image_id'] in selected]
    if {r['known_class_id'] for r in base} != set(known):
        raise ValueError('labelled subset lacks a known class; cannot initialize the fixed protocol')
    learner, evaluator = [], []
    for session in stream.evaluator_sessions:
        for r in session:
            safe = {'sample_id': r.sample_id, 'session_id': r.session_id, 'feature_row': rows[r.sample_id]}
            learner.append(safe)
            evaluator.append(dict(safe, class_id=r.class_id, is_known=r.class_id in known,
                                  introduction_session=schedule.get(r.class_id, 0)))
    probes = [{'sample_id': sample_ids[r['image_id']], 'feature_row': rows[sample_ids[r['image_id']]],
               'class_id': r['class_id'], 'official_split': 'test',
               'eligibility_session': schedule.get(r['class_id'], 0)} for r in records if r['official_split'] == 'test']
    if {r['class_id'] for r in probes} != set(known + novel):
        raise ValueError('test probes must cover all declared classes')
    populations = [{r['sample_id'] for r in subset} for subset in (base, learner, probes)]
    if sum(map(len, populations)) != len(records) or len(set.union(*populations)) != len(records):
        raise ValueError('base/stream/probe overlap or incomplete coverage')
    labelled_identity = digest({'algorithm': 'numpy_RandomState0_global_half',
                                'sample_ids': [r['sample_id'] for r in base]})
    partition = seal({'protocol_version': PROTOCOL, 'configuration': config.snapshot(),
                      'seed': config.seed, 'ssb_sha256': split_hash, 'metadata_hashes': metadata_hashes,
                      'metadata_sha256': digest(metadata_hashes), 'labelled_subset_sha256': labelled_identity,
                      'known_classes': list(known), 'novel_classes': list(novel),
                      'source_records': records, 'feature_sample_ids': sorted(rows),
                      'class_introduction_schedule': [list(pair) for pair in stream.class_introduction_schedule]})
    bundle = {'partition': partition}
    for name, subset in zip(VIEW_NAMES, (base, learner, evaluator, probes)):
        bundle[name] = seal({'protocol_version': PROTOCOL, 'seed': config.seed, 'view': name,
                             'partition_checksum': partition['checksum'], 'ssb_sha256': split_hash,
                             'labelled_subset_sha256': labelled_identity,
                             'sample_ordering': 'record_order', 'records': subset})
    return bundle


def build_cub_bundle(cub_root, config):
    known, novel = load_cub_ssb_classes()
    if len(known) != 100 or len(novel) != 100 or set(known) | set(novel) != set(range(200)):
        raise ValueError('expected original CUB SSB 100/100 split')
    records, hashes = read_metadata(cub_root)
    return build_bundle(records, hashes, known, novel, config)


def validate_bundle(bundle, config, *, official=True):
    if set(bundle) != {'partition', *VIEW_NAMES}:
        raise ValueError('incomplete manifest bundle')
    for value in bundle.values():
        verify(value)
    p = bundle['partition']
    if p['configuration'] != config.snapshot():
        raise ValueError('manifest configuration mismatch')
    if official:
        known, novel = load_cub_ssb_classes()
        if (p['known_classes'] != list(known) or p['novel_classes'] != list(novel)
                or p['ssb_sha256'] != CUB_SPLIT_SHA256):
            raise ValueError('manifest does not use the pinned official SSB split')
    rebuilt = build_bundle(p['source_records'], p['metadata_hashes'], p['known_classes'],
                           p['novel_classes'], config, p['ssb_sha256'])
    if bundle != rebuilt:
        raise ValueError('manifest views/partition disagree with deterministic protocol')
    return bundle


def save_bundle(bundle, output):
    output = Path(output)
    if output.exists():
        raise ValueError('manifest output directory already exists')
    with tempfile.TemporaryDirectory(prefix='.manifest-', dir=output.parent) as folder:
        staging = Path(folder) / 'complete'
        staging.mkdir()
        for name, value in bundle.items():
            write_json(staging / (name + '.json'), value)
        os.rename(staging, output)


def load_bundle(folder, config, *, official=True):
    return validate_bundle({name: read_json(Path(folder) / (name + '.json'))
                            for name in ('partition', *VIEW_NAMES)}, config, official=official)


def learner_session(bundle, session_id):
    """Only current-session records, never the privileged bundle, enter the learner."""
    if type(session_id) is not int or not 1 <= session_id <= 5:
        raise ValueError('online session must be in 1..5')
    return tuple(r.copy() for r in bundle['stream_learner']['records'] if r['session_id'] == session_id)
