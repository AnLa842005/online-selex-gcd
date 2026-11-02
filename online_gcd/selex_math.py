"""Deterministic CPU Euclidean primitives for the frozen-evidence method."""
import numpy as np


def squared_distances(matrix, value, block_size=256, *, reference=False):
    if reference:
        return np.asarray([np.sum((row - value) ** 2) for row in matrix], dtype=np.float64)
    result = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), block_size):
        delta = matrix[start:start + block_size] - value
        result[start:start + len(delta)] = np.sum(delta * delta, axis=1)
    return result


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    weights = np.exp(logits - np.max(logits))
    return weights / np.sum(weights)


def mixture(ids, distances, memberships, mixing=.25):
    """Group-balanced hierarchy mass on the hard-gated candidate set only."""
    logits = -np.asarray(distances, dtype=np.float64) / .1
    p = softmax(logits)
    groups = {}
    for index, cid in enumerate(ids):
        groups.setdefault(memberships.get(cid, ('singleton', cid)), []).append(index)
    # exp(h_g) after one common shift. No temperature in geometry edge scores.
    shifted = np.exp(logits - np.max(logits))
    means = [float(np.mean(shifted[indices])) for indices in groups.values()]
    mass = np.sum(means)
    u = np.zeros(len(ids), dtype=np.float64)
    for indices, mean in zip(groups.values(), means):
        u[indices] = mean / (mass * len(indices))
    return (1 - mixing) * p + mixing * u


def geometry_score(left, right):
    # Clip only roundoff at the mathematical [-1,1] boundaries.
    return float(np.clip(1 - .5 * np.sum((np.asarray(left) - np.asarray(right)) ** 2), -1., 1.))
