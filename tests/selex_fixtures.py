"""Artificial fixtures only. Labels belong to builders/evaluators, never learners."""
from pathlib import Path

import numpy as np

from online_gcd.artifacts import digest
from online_gcd.clean_protocol import CleanConfig
from online_gcd.cub_manifest import save_bundle
from online_gcd.feature_cache import write_cache
from online_gcd.prototype_bank import Prototype, normalize
from online_gcd.selex_euclidean import SelexEuclideanBank, SelexOptions
from online_cub_fixtures import fixture_bundle


def angle(degrees, dimension=768):
    value = np.zeros(dimension)
    value[0], value[1] = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return value


def seeded_bank(vectors, *, threshold=.75, structure=.5, mode='temporal', geometry=.8,
                mixing=.25, hierarchy_enabled=True, block_size=256):
    dimension = len(vectors[0]) if len(vectors) else 768
    bank = SelexEuclideanBank(dimension, threshold, structure_threshold=structure if mode == 'temporal' else None,
                              geometry_threshold=geometry if mode == 'geometry' else None,
                              options=SelexOptions(mode, mixing, hierarchy_enabled, block_size))
    known = np.zeros(dimension)
    known[-1] = 1.
    bank.initialize_known({0: [known]})
    for i, feature in enumerate(vectors, 1):
        cid = f'novel-{i:06d}'
        bank._prototypes[cid] = Prototype(cid, tuple(normalize(feature, dimension)), 1, False, 1, 1)
    bank._next_novel_id = len(vectors) + 1
    bank._last_session = 1 if len(vectors) else 0
    bank.observation_steps = len(vectors)
    bank._rebuild_cache()
    return bank


def hard_fake_cache(root):
    """Known halves plus noisy/overlapping novel modes; deliberately nontrivial.

    Fixed artificial geometry, not a model of real CUB. Metadata/protocol builder
    is unchanged. Class labels are used by the privileged data generator only.
    """
    root = Path(root)
    bundle = fixture_bundle(root)
    manifests, cache = root / 'manifests', root / 'cache'
    save_bundle(bundle, manifests)
    rows = {sid: i for i, sid in enumerate(bundle['partition']['feature_sample_ids'])}
    values = np.zeros((len(rows), 768), dtype=np.float32)
    rng = np.random.default_rng(42)
    known = set(bundle['partition']['known_classes'])
    centers = np.eye(768)[:7].copy()
    # Similar but distinct novel classes; initial dispersion and arrival order
    # generate ambiguity as well as fragmentation. Nothing sets a target K.
    centers[2] = .94 * centers[0] + np.sqrt(1 - .94**2) * centers[2]
    centers[3] = .94 * centers[0] + np.sqrt(1 - .94**2) * centers[3]
    for row in bundle['partition']['source_records']:
        sid = 'cub-' + digest(['CUB-200-2011', row['image_id']])
        c = row['class_id']
        values[rows[sid]] = centers[c]
        if c not in known:
            values[rows[sid], 710:718] = rng.normal(0, .055, 8)
            if row['official_split'] == 'train':
                values[rows[sid], 720 + c] = (-1 if row['image_id'] % 2 else 1) * np.tan(np.deg2rad(10))
    for c in known:
        base = [r for r in bundle['base']['records'] if r['known_class_id'] == c]
        for i, row in enumerate(base):
            values[row['feature_row'], 720 + c] = (-1 if i < len(base) // 2 else 1) * np.tan(np.deg2rad(15))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    metadata = write_cache(cache, bundle, [values], {'kind': 'fixture'})
    return bundle, manifests, cache, values, metadata
