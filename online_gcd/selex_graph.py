"""Bounded undirected evidence and deterministic complete-link structure."""
from itertools import combinations
import heapq

import numpy as np

from online_gcd.selex_math import squared_distances


def prune_edges(edges):
    """Greedy global score/ID order; retain at most eight incident edges."""
    degrees, retained = {}, {}
    for pair, (total, count) in sorted(edges.items(), key=lambda item: (-item[1][0] / item[1][1], item[0])):
        a, b = pair
        if degrees.get(a, 0) < 8 and degrees.get(b, 0) < 8:
            retained[pair] = (total, count)
            degrees[a] = degrees.get(a, 0) + 1
            degrees[b] = degrees.get(b, 0) + 1
    return retained


def geometry_edges(ids, matrix, block_size=256, *, reference=False):
    """Exact globally greedy degree bound without a production N-by-N matrix.

    Cache one best remaining partner per row. Rescan only changed/invalidated
    rows; low-score edges are retained too, matching full global pruning order.
    """
    if reference:
        all_edges = {}
        for i, j in combinations(range(len(ids)), 2):
            score = float(np.clip(1 - .5 * np.sum((matrix[i] - matrix[j]) ** 2), -1., 1.))
            all_edges[(ids[i], ids[j])] = (score, 1)
        return {pair: value[0] for pair, value in prune_edges(all_edges).items()}
    degrees = [0] * len(ids)
    active = set(range(len(ids)))
    selected = {}

    def best(i):
        others = [j for j in sorted(active) if j != i and tuple(sorted((ids[i], ids[j]))) not in selected]
        answer = None
        for start in range(0, len(others), block_size):
            indices = others[start:start + block_size]
            scores = np.clip(1 - .5 * squared_distances(matrix[indices], matrix[i], block_size), -1., 1.)
            for offset in np.flatnonzero(scores == scores.max()):
                j = indices[int(offset)]
                pair = tuple(sorted((ids[i], ids[j])))
                candidate = (-float(scores[offset]), pair, j)
                if answer is None or candidate < answer:
                    answer = candidate
        return answer

    rows = {i: best(i) for i in sorted(active)}
    while True:
        candidates = [(row[0], row[1], i, row[2]) for i, row in rows.items() if row is not None]
        if not candidates:
            break
        negative, pair, i, j = min(candidates)
        selected[pair] = -negative
        degrees[i] += 1
        degrees[j] += 1
        removed = {k for k in (i, j) if degrees[k] == 8}
        active -= removed
        for k in removed:
            rows.pop(k)
        for k in sorted(active):
            row = rows[k]
            if k in (i, j) or row is not None and row[2] in removed:
                rows[k] = best(k)
    return selected


def complete_link_groups(ids, scores, creation, *, reference=False):
    """Only graph-neighbor candidates; heap storage O(64N), group size <=8."""
    groups = {cid: (cid,) for cid in ids}
    owner = {cid: cid for cid in ids}
    adjacency = {cid: set() for cid in ids}
    for a, b in scores:
        adjacency[a].add(b)
        adjacency[b].add(a)

    def candidate(a, b):
        if a == b or a not in groups or b not in groups:
            return None
        left, right = groups[a], groups[b]
        members = tuple(sorted(left + right))
        if len(members) > 8:
            return None
        cross = [scores.get(tuple(sorted((i, j)))) for i in left for j in right]
        if any(value is None for value in cross):
            return None
        return (-min(cross), min(creation[i] for i in members), members, a, b, left, right)

    heap = []
    if not reference:
        for a, b in sorted(scores):
            row = candidate(a, b)
            if row is not None:
                heapq.heappush(heap, row)
    while True:
        if reference:
            candidates = [candidate(a, b) for a, b in combinations(sorted(groups), 2)]
            candidates = [row for row in candidates if row is not None]
            if not candidates:
                break
            row = min(candidates)
        else:
            if not heap:
                break
            row = heapq.heappop(heap)
            _, _, _, a, b, left, right = row
            if groups.get(a) != left or groups.get(b) != right:
                continue
        _, _, members, a, b, _, _ = row
        del groups[a], groups[b]
        key = members[0]
        groups[key] = members
        for cid in members:
            owner[cid] = key
        if not reference:
            neighbors = {owner[other] for cid in members for other in adjacency[cid]} - {key}
            for other in sorted(neighbors):
                first, second = sorted((key, other))
                proposal = candidate(first, second)
                if proposal is not None:
                    heapq.heappush(heap, proposal)
    return sorted((members for members in groups.values() if len(members) > 1))
