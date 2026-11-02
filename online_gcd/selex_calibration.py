"""External known-base calibrator; birth-origin labels never enter the learner."""
import random

import numpy as np
from threadpoolctl import threadpool_limits

from online_gcd.artifacts import digest, seal
from online_gcd.prototype_bank import Prototype, integer, normalize, real, vector
from online_gcd.selex_euclidean import SelexEuclideanBank, SelexOptions
from online_gcd.selex_math import squared_distances


def select_threshold(positive, negative, *, lower):
    if not positive or not negative:
        raise ValueError('structural calibration requires eligible positive and negative diagnostics; no fallback')
    pos, neg = np.sort(positive), np.sort(negative)
    boundaries = np.unique(np.r_[pos, neg, lower, 1.])
    after = np.nextafter(boundaries, np.inf)
    candidates = np.unique(np.r_[boundaries, after[after <= 1]])
    accepted = len(pos) - np.searchsorted(pos, candidates, side='left')
    rejected = np.searchsorted(neg, candidates, side='left')
    numerators = [int(a) * len(neg) + int(r) * len(pos) for a, r in zip(accepted, rejected)]
    best = max(range(len(candidates)), key=lambda i: (numerators[i], candidates[i]))
    return {'threshold': float(candidates[best]), 'positive_count': len(pos), 'negative_count': len(neg),
            'balanced_accuracy': numerators[best] / (2 * len(pos) * len(neg)),
            'objective': '0.5 * (positive_acceptance + negative_rejection)', 'tie_break': 'largest_threshold',
            'candidates': [{'threshold': float(t), 'positive_accepted': int(a), 'negative_rejected': int(r),
                            'balanced_accuracy': n / (2 * len(pos) * len(neg))}
                           for t, a, r, n in zip(candidates, accepted, rejected, numerators)]}


def calibrate_selex_base(labelled_base, assignment_threshold, *, base_identity, cache_identity,
                         mode='temporal', block_size=256):
    with threadpool_limits(limits=1, user_api='blas'):
        return _calibrate(labelled_base, assignment_threshold, base_identity=base_identity,
                          cache_identity=cache_identity, mode=mode, block_size=block_size)


def _calibrate(labelled_base, assignment_threshold, *, base_identity, cache_identity, mode, block_size):
    options = SelexOptions(mode=mode, block_size=block_size)
    real(assignment_threshold, 'assignment_threshold', -1, 1)
    if not isinstance(labelled_base, dict) or len(labelled_base) < 2:
        raise ValueError('at least two known base classes required')
    for c in labelled_base:
        integer(c, 'known class ID')
    prepared, used, dimension = {}, set(), None
    for c in sorted(labelled_base):
        samples = list(labelled_base[c])
        if len(samples) < 5:
            raise ValueError(f'known class {c} requires at least five labelled base samples')
        rows = []
        for item in samples:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError('base samples require (opaque sample ID, feature)')
            sid, value = item
            if not isinstance(sid, str) or not sid.strip() or sid in used:
                raise ValueError('base IDs must be unique non-empty strings')
            used.add(sid)
            array = np.asarray(value)
            if dimension is None:
                if array.ndim != 1 or not len(array):
                    raise ValueError('base features must be vectors')
                dimension = len(array)
            rows.append((sid, vector(value, dimension)))
        prepared[c] = sorted(rows, key=lambda row: row[0])
    # The pseudo learner accepts vectors and opaque IDs only, no origin labels.
    pseudo = SelexEuclideanBank(dimension, assignment_threshold, options=options, _collect_only=True)
    origins, remaining = {}, []
    for c, rows in prepared.items():
        for _, feature in rows[:2]:
            cid = f'novel-{pseudo._next_novel_id:06d}'
            pseudo._next_novel_id += 1
            pseudo._prototypes[cid] = Prototype(cid, tuple(normalize(feature, dimension)), 1, False, 0, 0)
            origins[cid] = c  # Calibrator-private, immutable birth origin.
        remaining.extend((sid, c, feature) for sid, feature in rows[2:])
    initial = len(origins)
    pseudo._rebuild_cache()
    random.Random(1).shuffle(remaining)
    for sid, known_label, feature in remaining:
        predicted = pseudo.observe(sid, 1, feature)
        if predicted not in origins:
            origins[predicted] = known_label
    assert not pseudo.nodes and pseudo.next_node_id == 1
    positive, negative = [], []
    if mode == 'temporal':
        for (a, b), (total, count) in sorted(pseudo.edges.items()):
            if count >= 3:
                (positive if origins[a] == origins[b] else negative).append(total / count)
    else:
        ids = pseudo._active_ids
        matrix = pseudo._matrix[:len(ids)]
        for i, a in enumerate(ids):
            for start in range(i + 1, len(ids), block_size):
                columns = list(range(start, min(start + block_size, len(ids))))
                scores = np.clip(1 - .5 * squared_distances(matrix[columns], matrix[i], block_size), -1., 1.)
                for j, score in zip(columns, scores):
                    (positive if origins[a] == origins[ids[j]] else negative).append(float(score))
    objective = select_threshold(positive, negative, lower=0. if mode == 'temporal' else -1.)
    return seal({'schema_version': 1, 'method': f'known_base_singleton_pseudostream_{mode}_v0',
                 'base_manifest_checksum': base_identity, 'cache_checksum': cache_identity,
                 'base_features_sha256': digest({str(c): [[sid, x.tolist()] for sid, x in rows]
                                                 for c, rows in prepared.items()}),
                 'assignment_threshold': float(assignment_threshold), 'feature_dim': dimension,
                 'known_class_count': len(prepared), 'base_sample_count': len(used),
                 'pseudo_session_count': 1, 'pseudo_hierarchy_rebuilds': 0,
                 'initial_pseudo_leaf_count': initial, 'pseudo_leaf_count': len(origins),
                 'born_pseudo_leaf_count': len(origins) - initial,
                 'diagnostic_leaf_scope': 'all_initial_and_born', 'birth_origin_labels_in_learner': False,
                 'diagnostic_opportunity_gate': 3 if mode == 'temporal' else None,
                 'split_rule': 'sorted opaque base IDs; first two per known class initialize leaves',
                 'shuffle': 'independent random.Random(1); remaining rows in sorted known class and sample ID order',
                 'settings': options.snapshot(), **objective})
