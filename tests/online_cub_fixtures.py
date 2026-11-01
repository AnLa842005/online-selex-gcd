"""Tiny fake CUB metadata and unit vectors; never real CUB assets."""
from pathlib import Path

import numpy as np

from online_gcd.clean_protocol import CleanConfig
from online_gcd.cub_manifest import build_bundle, read_metadata, save_bundle
from online_gcd.feature_cache import write_cache


def metadata_fixture(root, classes=7, train_per_class=30, test_per_class=3):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images, labels, splits = [], [], []
    image_id = 1
    for c in range(classes):
        for j in range(train_per_class + test_per_class):
            images.append(f'{image_id} {c+1:03}.class_name/img_{j}.jpg\n')
            labels.append(f'{image_id} {c+1}\n')
            splits.append(f'{image_id} {int(j < train_per_class)}\n')
            image_id += 1
    for name, values in (('images.txt', images), ('image_class_labels.txt', labels), ('train_test_split.txt', splits)):
        (root / name).write_text(''.join(values), encoding='utf-8')
    return root


def fixture_bundle(root):
    source = metadata_fixture(Path(root) / 'metadata')
    records, hashes = read_metadata(source)
    return build_bundle(records, hashes, [1, 5], [0, 2, 3, 4, 6], CleanConfig())


def fixture_cache(root):
    root = Path(root)
    bundle = fixture_bundle(root)
    manifests, cache = root / 'manifests', root / 'cache'
    save_bundle(bundle, manifests)
    features = np.zeros((len(bundle['partition']['source_records']), 768), dtype=np.float32)
    from online_gcd.artifacts import digest
    rows = {sid: i for i, sid in enumerate(bundle['partition']['feature_sample_ids'])}
    rng = np.random.default_rng(4)
    for r in bundle['partition']['source_records']:
        sid = 'cub-' + digest(['CUB-200-2011', r['image_id']])
        features[rows[sid], r['class_id']] = 1
        features[rows[sid], 7:16] = rng.normal(0, .01, 9)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    provenance = write_cache(cache, bundle, [features[:100], features[100:]], {'kind': 'fixture'})
    return bundle, manifests, cache, features, provenance
