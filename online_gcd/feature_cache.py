"""Label-free NPY cache with strict provenance/content validation. CPU only."""
from pathlib import Path
import os
import tempfile

import numpy as np

from online_gcd.artifacts import digest, file_sha256, read_json, seal, verify, write_json
from online_gcd.clean_protocol import DINO_SHA256, FIXED, PREPROCESSING, PROTOCOL


def cache_contract(bundle):
    p = bundle['partition']
    return {'protocol_version': PROTOCOL, 'manifest_checksum': p['checksum'],
            'metadata_sha256': p['metadata_sha256'], 'ssb_sha256': p['ssb_sha256'],
            'labelled_subset_sha256': p['labelled_subset_sha256'],
            'checkpoint_sha256': DINO_SHA256, 'checkpoint_role': FIXED['checkpoint_role'],
            'architecture': FIXED['backbone'], 'feature_layer': FIXED['feature_layer'],
            'feature_dim': 768, 'dtype': 'float32', 'normalization': 'L2',
            'preprocessing': PREPROCESSING, 'preprocessing_sha256': digest(PREPROCESSING),
            'sample_ids_sha256': digest(p['feature_sample_ids']), 'num_samples': len(p['feature_sample_ids'])}


def validate_features(features, count):
    if features.dtype != np.float32 or features.shape != (count, 768):
        raise ValueError('feature cache must be float32 with the expected row count and dimension 768')
    for start in range(0, count, 1024):
        block = features[start:start + 1024]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=2e-5, rtol=0):
            raise ValueError('feature cache contains nonfinite, zero or non-normalized rows')


def write_cache(output, bundle, batches, extraction):
    """Sequential batches -> staged memmap -> complete directory. No partial resume."""
    output = Path(output)
    if output.exists():
        raise ValueError('cache output already exists; use --reuse only after validation')
    contract = cache_contract(bundle)
    with tempfile.TemporaryDirectory(prefix='.features-', dir=output.parent) as folder:
        staging = Path(folder) / 'complete'
        staging.mkdir()
        path = staging / 'features.npy'
        features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                             shape=(contract['num_samples'], 768))
        try:
            offset = 0
            for batch in batches:
                batch = np.asarray(batch)
                validate_features(batch, len(batch))
                end = offset + len(batch)
                if end > len(features):
                    raise ValueError('too many extracted rows')
                features[offset:end] = batch
                offset = end
            if offset != len(features):
                raise ValueError('incomplete extraction: missing feature rows')
            features.flush()
        finally:
            del features  # Close Windows mmap before rename/temporary cleanup.
        ids = bundle['partition']['feature_sample_ids']
        write_json(staging / 'sample_ids.json', ids)
        metadata = seal(dict(contract, features_sha256=file_sha256(path),
                             extraction=extraction, schema_version=1))
        write_json(staging / 'cache.json', metadata)
        os.rename(staging, output)
    return metadata


def load_cache(folder, bundle, *, allow_fixture=False):
    folder = Path(folder)
    metadata = verify(read_json(folder / 'cache.json'))
    contract = cache_contract(bundle)
    if set(metadata) != set(contract) | {'checksum', 'schema_version', 'features_sha256', 'extraction'}:
        raise ValueError('invalid cache metadata fields')
    for name, expected in contract.items():
        if metadata.get(name) != expected:
            raise ValueError(f'incompatible feature cache: {name}')
    extraction = metadata.get('extraction')
    if not isinstance(extraction, dict):
        raise ValueError('invalid extraction provenance')
    kind = extraction.get('kind')
    if type(metadata.get('schema_version')) is not int or metadata['schema_version'] != 1 or kind not in ('frozen_dino', 'fixture'):
        raise ValueError('invalid cache provenance/schema')
    if kind == 'fixture' and not allow_fixture:
        raise ValueError('fixture cache cannot produce a real CUB result')
    if kind == 'frozen_dino':
        required = {'kind', 'checkpoint_path', 'repository_commit', 'timestamp_utc', 'device',
                    'batch_size', 'workers', 'seed', 'deterministic_algorithms', 'tf32', 'versions'}
        if (set(extraction) != required or extraction['seed'] != bundle['partition']['seed']
                or extraction['deterministic_algorithms'] is not True or extraction['tf32'] is not False
                or not all(isinstance(extraction[k], str) and extraction[k]
                           for k in ('checkpoint_path', 'repository_commit', 'timestamp_utc', 'device'))
                or not isinstance(extraction['versions'], dict)
                or set(extraction['versions']) != {'python', 'torch', 'torchvision', 'pillow', 'numpy'}):
            raise ValueError('incomplete or incompatible frozen-DINO extraction provenance')
        from online_gcd.prototype_bank import integer
        integer(extraction['batch_size'], 'batch_size', 1)
        integer(extraction['workers'], 'workers')
    ids = read_json(folder / 'sample_ids.json')
    if ids != bundle['partition']['feature_sample_ids'] or digest(ids) != metadata['sample_ids_sha256']:
        raise ValueError('cache sample IDs/order mismatch')
    path = folder / 'features.npy'
    if file_sha256(path) != metadata.get('features_sha256'):
        raise ValueError('feature data checksum mismatch')
    # CUB's full float32 cache is about 35 MiB. Load a read-only array instead
    # of retaining OS mappings through evaluator views/exception tracebacks.
    features = np.load(path, allow_pickle=False)
    features.setflags(write=False)
    validate_features(features, len(ids))
    return features, metadata
