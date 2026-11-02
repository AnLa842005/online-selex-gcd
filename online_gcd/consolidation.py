"""Label-free novel-only greedy consolidation control; no evaluator imports."""
from dataclasses import dataclass
import time

import numpy as np
from threadpoolctl import threadpool_limits

from online_gcd.prototype_bank import Prototype, PrototypeBank, integer, normalize, real

EXPERIMENT = 'frozen-dino-prototype-consolidation-v0'


def pairwise_cosine(left, right):
    # Explicit fixed reduction order, independent of BLAS workers or block shape.
    return np.clip(np.einsum('ik,jk->ij', left, right, optimize=False), -1., 1.)


def pair_key(a, b, similarity):
    oldest = sorted(((a.creation_session, a.cluster_id), (b.creation_session, b.cluster_id)))
    # Pair age is its earliest creation session; IDs break remaining ties.
    return (-float(similarity), oldest[0][0], *sorted((a.cluster_id, b.cluster_id)))


@dataclass(frozen=True)
class ConsolidationOptions:
    enabled: bool = False
    block_size: int = 256

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('consolidation enabled must be boolean')
        integer(self.block_size, 'consolidation block size', 1)


class ConsolidatingBank(PrototypeBank):
    consolidation_enabled = True

    def __init__(self, feature_dim, threshold, momentum, consolidation_threshold, block_size=256):
        super().__init__(feature_dim, threshold, momentum)
        self.consolidation_threshold = real(consolidation_threshold, 'consolidation_threshold', -1, 1)
        self.block_size = integer(block_size, 'consolidation block size', 1)
        self.aliases = {}
        self.merge_history = []

    def resolve_alias(self, cluster_id, *, compress=False):
        path, seen = [], set()
        current = cluster_id
        while current in self.aliases:
            if current in seen:
                raise ValueError('cycle in cluster aliases')
            seen.add(current)
            path.append(current)
            current = self.aliases[current]
        if current not in self._prototypes:
            raise ValueError(f'alias resolves to missing cluster: {current}')
        if compress:
            for retired in path:
                self.aliases[retired] = current
        return current

    def max_merge_chain_depth(self):
        # Audit original retirement edges even after live aliases are compressed.
        edges = {row['retired_id']: row['survivor_id'] for row in self.merge_history}
        maximum = 0
        for retired in edges:
            seen, current = set(), retired
            while current in edges:
                if current in seen:
                    raise ValueError('cycle in merge history')
                seen.add(current)
                current = edges[current]
            maximum = max(maximum, len(seen))
        return maximum

    def _merge(self, first, second, similarity):
        a, b = self._prototypes[first], self._prototypes[second]
        if a.originated_known or b.originated_known or first == second:
            raise ValueError('only two distinct active novel prototypes may merge')
        survivor, retired = sorted((a, b), key=lambda p: (p.creation_session, p.cluster_id))
        count = survivor.observation_count + retired.observation_count
        # Count-weighted normalized means of current EMA prototypes (not replay).
        mean = normalize((survivor.observation_count / count) * np.asarray(survivor.vector)
                         + (retired.observation_count / count) * np.asarray(retired.vector), self.feature_dim)
        self._prototypes[survivor.cluster_id] = Prototype(
            survivor.cluster_id, tuple(mean), count, False,
            min(a.creation_session, b.creation_session), max(a.last_updated_session, b.last_updated_session))
        del self._prototypes[retired.cluster_id]
        self.aliases[retired.cluster_id] = survivor.cluster_id
        self.merge_history.append({'retired_id': retired.cluster_id, 'survivor_id': survivor.cluster_id,
                                   'session_id': self.last_session, 'similarity': float(similarity)})
        return survivor.cluster_id, retired.cluster_id

    def consolidate(self, *, implementation='optimized', progress=None):
        # Preserve one numerical BLAS policy for centroid norms, regardless of
        # caller thread settings. Pairwise einsum itself never dispatches BLAS.
        with threadpool_limits(limits=1, user_api='blas'):
            return self._consolidate(implementation=implementation, progress=progress)

    def _consolidate(self, *, implementation, progress):
        if implementation not in ('optimized', 'reference'):
            raise ValueError('unknown consolidation implementation')
        started = time.perf_counter()
        before = len(self._prototypes)
        ids = [cid for cid in self._active_ids if not self._prototypes[cid].originated_known]
        # Reuse the normalized Phase 3C cache. Stable slots are private to this call.
        vectors = np.ascontiguousarray(self._matrix[[self._row_by_id[cid] for cid in ids]])
        alive = set(range(len(ids)))
        similarity_seconds = 0.
        merge_seconds = 0.
        history_start = len(self.merge_history)

        def key(i, j, score):
            return pair_key(self._prototypes[ids[i]], self._prototypes[ids[j]], score)

        def row_best(i):
            best = None
            others = sorted(alive - {i})
            for start in range(0, len(others), self.block_size):
                columns = others[start:start + self.block_size]
                scores = pairwise_cosine(vectors[i:i + 1], vectors[columns])[0]
                maximum = float(scores.max())
                for offset in np.flatnonzero(scores == maximum):
                    j = columns[int(offset)]
                    candidate = (key(i, j, maximum), j, maximum)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            return best

        if progress:
            progress(f'Consolidation similarity start: novel={len(ids)} block_size={self.block_size}')
        tick = time.perf_counter()
        best_rows = {i: row_best(i) for i in sorted(alive)} if implementation == 'optimized' else {}
        similarity_seconds += time.perf_counter() - tick
        while len(alive) > 1:
            tick = time.perf_counter()
            if implementation == 'reference':
                slots = sorted(alive)
                scores = pairwise_cosine(vectors[slots], vectors[slots])
                candidates = [(key(i, j, scores[x, y]), i, j, float(scores[x, y]))
                              for x, i in enumerate(slots) for y, j in enumerate(slots) if x < y]
                _, i, j, score = min(candidates)
            else:
                candidates = [(row[0], i, row[1], row[2]) for i, row in best_rows.items() if row is not None]
                _, i, j, score = min(candidates)
            similarity_seconds += time.perf_counter() - tick
            if score < self.consolidation_threshold:
                break
            tick = time.perf_counter()
            survivor, retired = self._merge(ids[i], ids[j], score)
            keep, remove = (i, j) if ids[i] == survivor else (j, i)
            vectors[keep] = self._prototypes[survivor].vector
            alive.remove(remove)
            merge_seconds += time.perf_counter() - tick
            if implementation == 'optimized':
                tick = time.perf_counter()
                del best_rows[remove]
                # Only a changed/removed best partner requires a full row rescan.
                # All other rows compare only the affected survivor similarity.
                slots = sorted(alive - {keep})
                for start in range(0, len(slots), self.block_size):
                    rows = slots[start:start + self.block_size]
                    scores = pairwise_cosine(vectors[rows], vectors[keep:keep + 1])[:, 0]
                    for index, k in enumerate(rows):
                        old = best_rows[k]
                        if old is None or old[1] in (keep, remove):
                            best_rows[k] = row_best(k)
                        else:
                            candidate = (key(k, keep, scores[index]), keep, float(scores[index]))
                            if candidate[0] < old[0]:
                                best_rows[k] = candidate
                best_rows[keep] = row_best(keep)
                similarity_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        # Compress transitive aliases once, retaining original edges in the audit.
        for retired in self.aliases:
            self.resolve_alias(retired, compress=True)
        self._rebuild_cache()
        merge_seconds += time.perf_counter() - tick
        if progress:
            progress(f'Consolidation similarity end: elapsed={similarity_seconds:.3f}s')
            progress(f'Consolidation merge end: merges={len(self.merge_history) - history_start} elapsed={merge_seconds:.3f}s')
        return {'prototypes_before_consolidation': before,
                'prototypes_after_consolidation': len(self._prototypes),
                'novel_prototypes_before': len(ids), 'novel_prototypes_after': len(alive),
                'merge_count': len(self.merge_history) - history_start,
                'merge_history': [dict(row) for row in self.merge_history[history_start:]],
                'retired_cluster_count': len(self.aliases),
                'maximum_merge_chain_depth': self.max_merge_chain_depth(),
                'consolidation_threshold': self.consolidation_threshold,
                'consolidation_time_seconds': time.perf_counter() - started,
                'similarity_time_seconds': similarity_seconds, 'merge_time_seconds': merge_seconds,
                'consolidation_block_size': self.block_size}
