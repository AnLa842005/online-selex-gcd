"""Frozen-feature Online SelEx Euclidean v0; no label/evaluator/data imports."""
from dataclasses import dataclass
import time
import re

import numpy as np

from online_gcd.prototype_bank import PrototypeBank, integer, normalize, real
from online_gcd.selex_graph import complete_link_groups, geometry_edges, prune_edges
from online_gcd.selex_math import mixture, softmax, squared_distances

EXPERIMENT = 'online-selex-euclidean-frozen-evidence-v0'
GEOMETRY_EXPERIMENT = 'online-selex-euclidean-instantaneous-geometry-v0'
LAMBDA_ZERO_EXPERIMENT = 'online-selex-euclidean-lambda-zero-v0'
NO_HIERARCHY_EXPERIMENT = 'online-selex-euclidean-no-hierarchy-v0'


def experiment_identity(options):
    if options.mode == 'geometry':
        return GEOMETRY_EXPERIMENT
    if options.mixing == 0:
        return LAMBDA_ZERO_EXPERIMENT
    if not options.hierarchy_enabled:
        return NO_HIERARCHY_EXPERIMENT
    return EXPERIMENT


@dataclass(frozen=True)
class SelexOptions:
    mode: str = 'temporal'
    mixing: float = .25
    hierarchy_enabled: bool = True
    block_size: int = 256

    def __post_init__(self):
        if self.mode not in ('temporal', 'geometry'):
            raise ValueError('unknown Euclidean evidence mode')
        if type(self.mixing) not in (int, float) or self.mixing not in (0, .25):
            raise ValueError('frozen mixing must be 0 or 0.25')
        if type(self.hierarchy_enabled) is not bool:
            raise ValueError('hierarchy_enabled must be boolean')
        integer(self.block_size, 'block_size', 1)
        if sum((self.mode == 'geometry', self.mixing == 0, not self.hierarchy_enabled)) > 1:
            raise ValueError('mechanism ablations are separate frozen arms, not combined variants')

    def snapshot(self):
        return {'mode': self.mode, 'mixing': self.mixing, 'hierarchy_enabled': self.hierarchy_enabled,
                'block_size': self.block_size, 'temperature': .1, 'max_degree': 8,
                'max_group_size': 8, 'minimum_opportunities': 3, 'momentum': .9}


class SelexEuclideanBank(PrototypeBank):
    selex_euclidean_enabled = True

    def __init__(self, feature_dim, assignment_threshold, *, structure_threshold=None,
                 geometry_threshold=None, options=None, calibration_identities=None, _collect_only=False):
        super().__init__(feature_dim, assignment_threshold, .9)
        self.options = SelexOptions() if options is None else options
        if not isinstance(self.options, SelexOptions):
            raise ValueError('options must be SelexOptions')
        self.structure_threshold = (None if structure_threshold is None else
                                    real(structure_threshold, 'structure_threshold', 0, 1))
        self.geometry_threshold = (None if geometry_threshold is None else
                                   real(geometry_threshold, 'geometry_threshold', -1, 1))
        if self.options.mode == 'temporal' and self.geometry_threshold is not None:
            raise ValueError('temporal mode cannot borrow a geometry threshold')
        if self.options.mode == 'geometry' and self.structure_threshold is not None:
            raise ValueError('geometry mode cannot borrow a structure threshold')
        if not _collect_only:
            required = self.structure_threshold if self.options.mode == 'temporal' else self.geometry_threshold
            if required is None:
                raise ValueError('separate calibrated structural/geometry threshold required')
        if type(_collect_only) is not bool:
            raise ValueError('internal calibration role must be boolean')
        self._collect_only = _collect_only
        identities = {} if calibration_identities is None else calibration_identities
        allowed = {'assignment', 'structure'} if self.options.mode == 'temporal' else {'assignment', 'geometry'}
        if (not isinstance(identities, dict) or not set(identities) <= allowed
                or any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None
                       for value in identities.values())):
            raise ValueError('calibration identities must contain applicable SHA-256 strings only')
        self.calibration_identities = dict(identities)
        self.edges = {}  # sorted ID pair -> (evidence sum, opportunities)
        self.nodes = {}  # active composite ID -> sorted leaf membership only
        self.next_node_id = 1
        self.base_observation_count = 0
        self.observation_steps = 0
        self.pruned_edge_count = 0
        self.last_rebuild_session = 0
        self._memberships = {}
        self._novel_start = 0
        self.last_timing = {}

    def _rebuild_cache(self):
        super()._rebuild_cache()
        # Canonical known-/novel- IDs sort into contiguous regions. No leaf
        # retirement or known births occur in this method.
        self._novel_start = sum(p.originated_known for p in self._prototypes.values())

    def initialize_known(self, labelled_groups):
        super().initialize_known(labelled_groups)
        self.base_observation_count = sum(p.observation_count for p in self.prototypes)

    def _select(self, scores, normalized):
        if self.options.mixing == 0 or not self.options.hierarchy_enabled or not self.nodes:
            row = int(np.argmax(scores))
        else:
            accepted = np.flatnonzero(scores >= self.threshold)
            if not len(accepted):
                row = int(np.argmax(scores))
            else:
                ids = [self._active_ids[int(i)] for i in accepted]
                distances = squared_distances(self._matrix[accepted], normalized, self.options.block_size)
                probabilities = mixture(ids, distances, self._memberships, self.options.mixing)
                row = int(accepted[int(np.argmax(probabilities))])
        return self._prototypes[self._active_ids[row]], float(scores[row])

    def _nearest(self, normalized):
        if not self._prototypes:
            raise ValueError('initialize known prototypes before inference')
        # Explicit baseline shortcut, including its reduction/tie behavior.
        if self.options.mixing == 0 or not self.options.hierarchy_enabled or not self.nodes:
            return super()._nearest(normalized)
        scores = self._matrix[:len(self._active_ids)] @ normalized
        return self._select(scores, normalized)

    def predict_batch(self, features, batch_size=256):
        integer(batch_size, 'batch_size', 1)
        if not self._prototypes:
            raise ValueError('initialize known prototypes before inference')
        result = []
        for start in range(0, len(features), batch_size):
            values = np.stack([normalize(x, self.feature_dim) for x in features[start:start + batch_size]])
            # Broadcast GEMV retains scalar matrix-vector reduction order.
            scores = np.matmul(self._matrix[:len(self._active_ids)], values[:, :, None])[:, :, 0]
            for value, row in zip(values, scores):
                nearest, similarity = self._select(row, value)
                result.append(nearest.cluster_id if similarity >= self.threshold else None)
        return result

    def _pending_evidence(self, observation):
        if self.options.mode != 'temporal':
            return []
        start = self._novel_start
        if len(self._active_ids) - start < 2:
            return []
        distances = squared_distances(self._matrix[start:len(self._active_ids)], observation, self.options.block_size)
        # The cached contiguous region follows ID order, including tie order.
        nearest = np.argsort(distances, kind='stable')[:9]
        ids = [self._active_ids[start + int(i)] for i in nearest]
        probabilities = softmax(-distances[nearest] / .1)
        return [(tuple(sorted((ids[0], cid))),
                 float(2 * min(probabilities[0], probability) / (probabilities[0] + probability)))
                for cid, probability in zip(ids[1:], probabilities[1:])]

    def observe(self, sample_id, session_id, feature):
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError('sample_id must be non-empty')
        integer(session_id, 'session_id', 1)
        if session_id < self.last_session:
            raise ValueError('session_id cannot move backwards')
        started = time.perf_counter()
        observation = normalize(feature, self.feature_dim)
        normalized_at = time.perf_counter()
        pending = self._pending_evidence(observation)  # strictly pre-update, no mutation
        evidence_at = time.perf_counter()
        # Assignment and birth/EMA are the baseline lifecycle. _nearest above
        # supplies only a pre-update reranking within the accepted set.
        selected = super().observe(sample_id, session_id, feature)
        assigned_at = time.perf_counter()
        for pair, value in pending:
            total, count = self.edges.get(pair, (0., 0))
            self.edges[pair] = (total + value, count + 1)
        retained = prune_edges(self.edges)
        self.pruned_edge_count += len(self.edges) - len(retained)
        self.edges = retained
        self.observation_steps += 1
        self.last_timing = {'assignment_seconds': normalized_at - started + assigned_at - evidence_at,
                            'structural_evidence_seconds': evidence_at - normalized_at + time.perf_counter() - assigned_at}
        return selected

    def rebuild(self, *, reference=False):
        if self._collect_only:
            raise ValueError('calibration hierarchy must remain singleton; rebuild prohibited')
        started = time.perf_counter()
        ids = self._active_ids[self._novel_start:]
        if self.options.mode == 'geometry':
            matrix = self._matrix[self._novel_start:len(self._active_ids)]
            bounded = geometry_edges(ids, matrix, self.options.block_size, reference=reference)
            scores = {pair: score for pair, score in bounded.items() if score >= self.geometry_threshold}
            edge_count = len(bounded)
        else:
            scores = {pair: total / count for pair, (total, count) in self.edges.items()
                      if count >= 3 and total / count >= self.structure_threshold}
            edge_count = len(self.edges)
        similarity_at = time.perf_counter()
        memberships = (complete_link_groups(ids, scores, {cid: self._prototypes[cid].creation_session for cid in ids},
                                            reference=reference) if self.options.hierarchy_enabled else [])
        old = {members: nid for nid, members in self.nodes.items()}
        nodes, created = {}, []
        for members in memberships:
            nid = old.get(members)
            if nid is None:
                nid = f'node-{self.next_node_id:06d}'
                self.next_node_id += 1
                created.append(nid)
            nodes[nid] = members
        retired = sorted(set(self.nodes) - set(nodes))
        self.nodes = nodes
        self._rebuild_membership_cache()
        self.last_rebuild_session = self.last_session
        elapsed = time.perf_counter() - started
        # Retirement audit is returned to runner; never retained in learner state.
        return {'active_structural_node_count': len(nodes), 'structural_edge_count': edge_count,
                'eligible_structural_edge_count': len(scores), 'created_node_ids': created,
                'retired_node_ids': retired, 'hierarchy_update_seconds': elapsed,
                'structural_similarity_seconds': similarity_at - started,
                'pruned_edge_count': self.pruned_edge_count,
                'parent_geometry': self.parent_geometry()}

    def _rebuild_membership_cache(self):
        self._memberships = {cid: nid for nid, members in self.nodes.items() for cid in members}

    def parent_geometry(self):
        # Diagnostics only: count-weighted Euclidean center and maximum radius.
        result = {}
        for nid, members in self.nodes.items():
            vectors = np.asarray([self._prototypes[cid].vector for cid in members])
            counts = np.asarray([self._prototypes[cid].observation_count for cid in members], dtype=np.float64)
            center = np.sum(vectors * (counts / counts.sum())[:, None], axis=0)
            radius = float(np.sqrt(np.max(squared_distances(vectors, center, self.options.block_size))))
            result[nid] = {'centroid': center.tolist(), 'radius': radius}
        return result
