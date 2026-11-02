"""Small scalar oracle. Deliberately no production mixture/evidence/heap calls."""
import math
from itertools import combinations

import numpy as np

from online_gcd.selex_euclidean import SelexEuclideanBank


def distribution(ids, distances, memberships, mixing=.25):
    logits = [-float(d) / .1 for d in distances]
    weights = [math.exp(z - max(logits)) for z in logits]
    total = sum(weights)
    p = [w / total for w in weights]
    groups = {}
    for i, cid in enumerate(ids):
        groups.setdefault(memberships.get(cid, ('singleton', cid)), []).append(i)
    means = {g: sum(weights[i] for i in rows) / len(rows) for g, rows in groups.items()}
    denominator = sum(means.values())
    u = [0.] * len(ids)
    for g, rows in groups.items():
        for i in rows:
            u[i] = means[g] / denominator / len(rows)
    return np.asarray([(1 - mixing) * a + mixing * b for a, b in zip(p, u)])


def prune(edges):
    kept, degree = {}, {}
    for pair in sorted(edges, key=lambda pair: (-edges[pair][0] / edges[pair][1], pair)):
        a, b = pair
        if degree.get(a, 0) == 8 or degree.get(b, 0) == 8:
            continue
        kept[pair] = edges[pair]
        degree[a], degree[b] = degree.get(a, 0) + 1, degree.get(b, 0) + 1
    return kept


def groups(ids, scores, creation):
    active = [tuple([cid]) for cid in sorted(ids)]
    while True:
        candidates = []
        for i, j in combinations(range(len(active)), 2):
            a, b = active[i], active[j]
            union = tuple(sorted(a + b))
            if len(union) > 8:
                continue
            cross = []
            for x in a:
                for y in b:
                    if tuple(sorted((x, y))) not in scores:
                        break
                    cross.append(scores[tuple(sorted((x, y)))])
            if len(cross) == len(a) * len(b):
                candidates.append((-min(cross), min(creation[c] for c in union), union, i, j))
        if not candidates:
            return sorted(group for group in active if len(group) > 1)
        _, _, union, i, j = min(candidates)
        active = [group for k, group in enumerate(active) if k not in (i, j)] + [union]


class ReferenceBank(SelexEuclideanBank):
    def _select(self, scores, normalized):
        if self.options.mixing == 0 or not self.options.hierarchy_enabled or not self.nodes:
            row = int(np.argmax(scores))
        else:
            accepted = [i for i, value in enumerate(scores) if value >= self.threshold]
            if not accepted:
                row = int(np.argmax(scores))
            else:
                ids = [self._active_ids[i] for i in accepted]
                distances = [np.sum((np.asarray(self._prototypes[cid].vector) - normalized) ** 2) for cid in ids]
                probabilities = distribution(ids, distances, self._memberships, self.options.mixing)
                row = accepted[int(np.argmax(probabilities))]
        return self._prototypes[self._active_ids[row]], float(scores[row])

    def _pending_evidence(self, observation):
        if self.options.mode != 'temporal':
            return []
        rows = [(float(np.sum((np.asarray(p.vector) - observation) ** 2)), p.cluster_id)
                for p in self.prototypes if not p.originated_known]
        rows = sorted(rows)[:9]
        if len(rows) < 2:
            return []
        logits = [-distance / .1 for distance, _ in rows]
        weights = [math.exp(z - max(logits)) for z in logits]
        # Common softmax denominator cancels in this ratio.
        return [(tuple(sorted((rows[0][1], cid))), 2 * min(weights[0], w) / (weights[0] + w))
                for (_, cid), w in zip(rows[1:], weights[1:])]

    def rebuild(self, *, reference=False):
        return super().rebuild(reference=True)
