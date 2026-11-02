"""Validate a persisted Phase 3B calibration without rerunning a CUB workload."""
from copy import deepcopy
import math
import re

from online_gcd.artifacts import verify
from online_gcd.calibration import METHOD


def import_calibration(record, artifact):
    verify(artifact)
    required = {'schema_version', 'method', 'base_manifest_checksum', 'cache_checksum', 'feature_dim',
                'base_features_sha256', 'positive_count', 'negative_count', 'known_class_count',
                'objective', 'tie_break', 'threshold', 'balanced_accuracy', 'candidates', 'checksum'}
    if set(artifact) != required or artifact['schema_version'] != 1 or artifact['method'] != METHOD:
        raise ValueError('unsupported calibration schema/method')
    if record['calibration']['status'] != 'pending_import':
        raise ValueError('calibration already imported')
    if (artifact['cache_checksum'] != record['cache_checksum'] or artifact['feature_dim'] != 768
            or artifact['positive_count'] != record['population']['base']
            or artifact['negative_count'] != record['population']['base']
            or artifact['known_class_count'] != record['population']['known_classes']
            or artifact['tie_break'] != 'largest_threshold'
            or artifact['objective'] != '0.5 * (same_class_acceptance + different_class_rejection)'):
        raise ValueError('calibration identity/count/objective mismatch')
    for key in ('base_manifest_checksum', 'base_features_sha256'):
        if not isinstance(artifact[key], str) or not re.fullmatch('[0-9a-f]{64}', artifact[key]):
            raise ValueError('invalid calibration provenance hash')
    n = artifact['positive_count']
    candidates = artifact['candidates']
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('missing candidate diagnostics')
    previous = -math.inf
    for row in candidates:
        if (set(row) != {'threshold', 'positive_accepted', 'negative_rejected', 'balanced_accuracy'}
                or type(row['threshold']) not in (int, float)
                or not math.isfinite(row['threshold']) or not -1 <= row['threshold'] <= 1
                or row['threshold'] <= previous):
            raise ValueError('invalid candidate threshold/order')
        for key in ('positive_accepted', 'negative_rejected'):
            if type(row[key]) is not int or not 0 <= row[key] <= n:
                raise ValueError('invalid calibration candidate counts')
        if row['balanced_accuracy'] != (row['positive_accepted'] + row['negative_rejected']) / (2 * n):
            raise ValueError('candidate objective mismatch')
        previous = row['threshold']
    best = max(candidates, key=lambda row: (row['balanced_accuracy'], row['threshold']))
    if artifact['threshold'] != best['threshold'] or artifact['balanced_accuracy'] != best['balanced_accuracy']:
        raise ValueError('selected threshold violates objective/tie policy')
    result = deepcopy(record)
    result['calibration'] = {
        'status': 'imported', 'threshold': artifact['threshold'], 'artifact': deepcopy(artifact),
        'validation_scope': 'Checksum, cache identity, counts and recorded objective; no feature recomputation. '
                            'Original base-manifest identity was not supplied in console record.'}
    return result
