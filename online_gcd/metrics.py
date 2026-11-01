"""Evaluator-only metrics over persistent prediction IDs; never imported by the bank."""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def validate_predictions(labels, predictions):
    if len(labels) != len(predictions):
        raise ValueError('labels and predictions must have the same length')
    if any(type(c) is not int or c < 0 for c in labels):
        raise ValueError('evaluation labels must be non-negative integers')
    if any(p is not None and (not isinstance(p, str) or not p) for p in predictions):
        raise ValueError('predictions must be persistent string IDs or None for rejection')


def hungarian_mapping(labels, predictions):
    """One-to-one maximum-count mapping, sorted inputs for deterministic ties.

    None is a rejection, never a matchable cluster. Zero-support pairs are omitted.
    Tie resolution follows SciPy's deterministic solver on sorted rows/columns.
    """
    validate_predictions(labels, predictions)
    classes, clusters = sorted(set(labels)), sorted({p for p in predictions if p is not None})
    if not classes or not clusters:
        return {}
    counts = Counter(zip(labels, predictions))
    matrix = np.asarray([[counts[c, p] for p in clusters] for c in classes], dtype=np.int64)
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return {clusters[j]: classes[i] for i, j in zip(rows, columns) if matrix[i, j] > 0}


def accuracy(labels, predictions, mapping, selected=None):
    indices = [i for i, c in enumerate(labels) if selected is None or c in selected]
    correct = sum(predictions[i] is not None and mapping.get(predictions[i]) == labels[i] for i in indices)
    return {'value': correct / len(indices) if indices else None, 'count': len(indices)}


def accuracy_slices(labels, predictions, known_classes, mapping):
    validate_predictions(labels, predictions)
    known = set(known_classes)
    return {'all': accuracy(labels, predictions, mapping),
            'old': accuracy(labels, predictions, mapping, known),
            'new': accuracy(labels, predictions, mapping, set(labels) - known)}


def fragmentation(labels, predictions):
    validate_predictions(labels, predictions)
    counts = {c: len({p for y, p in zip(labels, predictions) if y == c and p is not None})
              for c in sorted(set(labels))}
    duplicates = sum(max(0, n - 1) for n in counts.values())
    return {'mean_excess_clusters': duplicates / len(counts) if counts else None,
            'duplicate_cluster_count': duplicates,
            'fragmented_class_fraction': sum(n > 1 for n in counts.values()) / len(counts) if counts else None,
            'clusters_per_class': {str(c): n for c, n in counts.items()}}


def clustering_report(labels, predictions, known_classes):
    mapping = hungarian_mapping(labels, predictions)
    known = set(known_classes)
    novel = [(c, p) for c, p in zip(labels, predictions) if c not in known]
    nmi, ari = None, None
    if len(novel) >= 2 and len({c for c, _ in novel}) >= 2:
        # Integer encoding supports None as a single rejection category without
        # colliding with any real persistent string ID.
        codes = {p: i for i, p in enumerate(dict.fromkeys(p for _, p in novel))}
        y, z = [c for c, _ in novel], [codes[p] for _, p in novel]
        nmi = float(normalized_mutual_info_score(y, z, average_method='arithmetic'))
        ari = float(adjusted_rand_score(y, z))
    return {'hungarian_accuracy': accuracy_slices(labels, predictions, known, mapping),
            'novel_nmi': nmi, 'novel_ari': ari, 'novel_count': len(novel),
            'predicted_cluster_count': len({p for p in predictions if p is not None}),
            'rejection_count': sum(p is None for p in predictions),
            'fragmentation': fragmentation(labels, predictions)}
