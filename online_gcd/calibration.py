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
