"""Separate merge calibration from labelled base-known halves only."""
import numpy as np
from threadpoolctl import threadpool_limits

from online_gcd.artifacts import digest, seal
from online_gcd.consolidation import pairwise_cosine
from online_gcd.prototype_bank import integer, normalize

METHOD = 'known_only_deterministic_halves_balanced_accuracy_v0'


def calibrate_consolidation(labelled_groups, *, base_identity, cache_identity):
    with threadpool_limits(limits=1, user_api='blas'):
        return _calibrate_consolidation(labelled_groups, base_identity=base_identity, cache_identity=cache_identity)


def _calibrate_consolidation(labelled_groups, *, base_identity, cache_identity):
    if not isinstance(labelled_groups, dict) or len(labelled_groups) < 2:
        raise ValueError('merge calibration requires at least two known classes')
    dimension, normalized, halves = None, {}, []
    for c in sorted(labelled_groups):
        integer(c, 'known class ID')
        samples = np.asarray(labelled_groups[c])
        if samples.ndim != 2 or len(samples) < 2:
            raise ValueError(f'known class {c} requires at least two base samples for merge calibration')
        dimension = samples.shape[1] if dimension is None else dimension
        values = np.stack([normalize(x, dimension) for x in samples])
        normalized[str(c)] = values.tolist()
        middle = len(values) // 2
        halves.append([normalize(np.mean(part, axis=0), dimension)
                       for part in (values[:middle], values[middle:])])
    prototypes = np.asarray(halves).reshape(-1, dimension)
    scores = pairwise_cosine(prototypes, prototypes)
    pos = np.sort([scores[2 * i, 2 * i + 1] for i in range(len(halves))])
    neg = np.sort([scores[i, j] for i in range(len(prototypes)) for j in range(i + 1, len(prototypes))
                   if i // 2 != j // 2])
    boundaries = np.unique(np.r_[pos, neg, -1., 1.])
    after = np.nextafter(boundaries, np.inf)
    candidates = np.unique(np.r_[boundaries, after[after <= 1]])
    accepted = len(pos) - np.searchsorted(pos, candidates, side='left')
    rejected = np.searchsorted(neg, candidates, side='left')
    # Unequal positive/negative counts: compare exact rational numerators.
    objectives = [int(a) * len(neg) + int(r) * len(pos) for a, r in zip(accepted, rejected)]
    best = max(range(len(candidates)), key=lambda i: (objectives[i], candidates[i]))
    return seal({'schema_version': 1, 'method': METHOD, 'base_manifest_checksum': base_identity,
                 'cache_checksum': cache_identity, 'feature_dim': dimension,
                 'base_features_sha256': digest(normalized), 'known_class_count': len(halves),
                 'base_sample_count': sum(len(v) for v in normalized.values()),
                 'split_rule': 'input base order; first floor(n/2) versus remaining samples per class',
                 'insufficient_class_policy': 'fail; fewer than two samples or degenerate mean',
                 'positive_count': len(pos), 'negative_count': len(neg),
                 'negative_pair_rule': 'all unordered cross-class sub-prototype pairs',
                 'objective': '0.5 * (same_class_acceptance + different_class_rejection)',
                 'tie_break': 'largest_threshold', 'threshold': float(candidates[best]),
                 'balanced_accuracy': float(objectives[best] / (2 * len(pos) * len(neg))),
                 'candidates': [{'threshold': float(t), 'positive_accepted': int(a), 'negative_rejected': int(r),
                                 'balanced_accuracy': float(s / (2 * len(pos) * len(neg)))}
                                for t, a, r, s in zip(candidates, accepted, rejected, objectives)]})
