"""CPU-testable HSSK LEAF engine, not a complete hierarchy or training provider.

No model, DEAN predictions, classifier, discovery/VFA state or legacy EMA enters
this API. Production OnlineHSSKHierarchyProvider intentionally remains blocked.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from types import MappingProxyType

import torch
from torch.nn import functional as F

from .selex_hierarchy import integer_ids
from .selex_hierarchy_provider import canonical, digest
from .selex_labeled_replay import VerifiedLabeledReplay, fingerprint

LEAF_SCHEMA = "online-hssk-leaf-v1"
MEMORY_SCHEMA = "online-hssk-memory-v1"


@dataclass(frozen=True)
class HSSKLeafConfig:
    max_free: int = 16
    min_support: int = 2
    k_penalty: float = 0.02
    max_iterations: int = 10
    tolerance: float = 1e-4
    momentum: float = 0.9
    memory_capacity: int = 512
    memory_seed_limit: int = 128
    epsilon: float = 1e-8

    def __post_init__(self):
        for key, value in (("max_free", 16), ("min_support", 2), ("k_penalty", .02),
                           ("max_iterations", 10), ("tolerance", 1e-4), ("momentum", .9)):
            if type(getattr(self, key)) is not type(value) or getattr(self, key) != value:
                raise ValueError(key + " is locked by the v1 contract")
        for key, cap in (("memory_capacity", 512), ("memory_seed_limit", 128)):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= cap:
                raise ValueError(key + " exceeds its contract bound")
        if type(self.epsilon) is not float or not math.isfinite(self.epsilon) or not 2**-126 <= self.epsilon <= 1e-4:
            raise ValueError("Invalid FP32 epsilon")


def checked_rows(features, sample_ids):
    ids = integer_ids(sample_ids, "HSSK sample IDs").detach().cpu()
    if (ids < 0).any() or len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate/negative HSSK sample IDs")
    if (not isinstance(features, torch.Tensor) or not features.is_floating_point()
            or features.shape != (len(ids), 768) or not torch.isfinite(features).all()):
        raise ValueError("HSSK real features must be finite floating [N,768], aligned with IDs")
    rows = features.detach().to(device="cpu", dtype=torch.float32).clone()
    if not torch.isfinite(rows).all():
        raise ValueError("HSSK features overflow FP32")
    order = torch.argsort(ids)
    return rows[order], ids[order]


@dataclass(frozen=True, init=False)
class VerifiedReplayFeatures:
    """Ephemeral batch features bound to Phase-1 verified, frozen selection.

    Construct with from_selection, never by accepting free-form GT tensors.
    Features are not persisted in the replay buffer or HSSK memory.
    """
    _payload: str

    @classmethod
    def from_selection(cls, replay, features, sample_ids):
        if not isinstance(replay, VerifiedLabeledReplay) or replay.selection is None:
            raise ValueError("HSSK replay features require a frozen verified selection")
        rows, ids = checked_rows(features, sample_ids)
        selected = replay.selection
        if set(ids.tolist()) != set(selected.sample_ids):
            raise ValueError("Replay features do not cover the exact selected real references")
        entries = {e.sample_id: e for e in replay.entries}
        labels = []
        for key in ids.tolist():
            replay.eligibility.verify(entries[key])
            labels.append(entries[key].true_label)
        counts = Counter(labels)
        if any(n != 2 for n in counts.values()) or len(counts) > 16 or len(ids) > 32:
            raise ValueError("Replay anchors require exactly two distinct verified references/class")
        obj = object.__new__(cls)
        p = dict(rows=rows.tolist(), sample_ids=ids.tolist(), true_labels=labels,
                 selection_hash=selected.selection_hash, batch_id=selected.batch_id,
                 eligibility_manifest=replay.eligibility.identity,
                 dataset_fingerprint=replay.eligibility.dataset_fingerprint,
                 split_fingerprint=replay.eligibility.split_fingerprint)
        object.__setattr__(obj, "_payload", canonical(p))
        return obj

    def values(self):
        p = json.loads(self._payload)
        return (torch.tensor(p["rows"], dtype=torch.float32).reshape(-1, 768),
                torch.tensor(p["sample_ids"], dtype=torch.long),
                torch.tensor(p["true_labels"], dtype=torch.long), p)


@dataclass(frozen=True)
class HSSKMemoryTrack:
    track_id: int
    center_values: tuple
    support_count: int
    radius: float
    last_active_batch: int
    created_batch: int
    updates: int
    anchor_gt: object
    lineage: str
    provenance: str

    @property
    def center(self):
        return torch.tensor(self.center_values, dtype=torch.float32)


class HSSKPrototypeMemory:
    """Bounded HSSK-owned summaries. Reads and rejected fits never mutate state."""
    def __init__(self, config=None):
        self.config = config or HSSKLeafConfig()
        self._tracks = {}
        self._next_id = -1
        self._last_batch = -1
        self._evictions = 0

    @property
    def tracks(self):
        return MappingProxyType(dict(self._tracks))

    @property
    def state_hash(self):
        return self.state_dict()["state_hash"]

    @property
    def next_id(self):
        return self._next_id

    @torch.no_grad()
    def seed_candidates(self, current):
        if current.shape[0] == 0 or not self._tracks:
            return torch.empty(0, 768), ()
        keys = sorted(self._tracks)
        centers = torch.stack([self._tracks[k].center for k in keys])
        distances = squared_distances(centers, current).min(dim=1).values
        relevant = sorted(range(len(keys)), key=lambda i: (float(distances[i]), keys[i]))[:self.config.memory_seed_limit]
        return centers[relevant].clone(), tuple(keys[i] for i in relevant)

    @torch.no_grad()
    def updated_copy(self, centers, support_counts, radii, anchor_gt, batch_id, lineage, provenance=None):
        """Plan an atomic one-time update; caller commits only accepted leaf fit."""
        if type(batch_id) is not int or batch_id <= self._last_batch:
            raise ValueError("HSSK memory may update only once per successful incoming batch")
        keys = sorted(centers)
        if not keys or len(keys) > self.config.memory_capacity:
            raise ValueError("Memory capacity cannot exclude an active leaf")
        if set(keys) != set(support_counts) or set(keys) != set(radii) or set(keys) != set(anchor_gt):
            raise ValueError("Misaligned leaf summaries")
        for k in keys:
            if (centers[k].shape != (768,) or not torch.isfinite(centers[k]).all()
                    or type(support_counts[k]) is not int or support_counts[k] < 2
                    or not math.isfinite(float(radii[k])) or radii[k] < 0
                    or (anchor_gt[k] is not None and (type(anchor_gt[k]) is not int or anchor_gt[k] < 0))):
                raise ValueError("Invalid real-support leaf summary")
        if not fingerprint(lineage):
            raise ValueError("Leaf lineage requires a deterministic fit identity")
        provenance = provenance or dict(eligibility_manifest=None, dataset_fingerprint=None, split_fingerprint=None)
        if (set(provenance) != {"eligibility_manifest", "dataset_fingerprint", "split_fingerprint"}
                or any(v is not None and not fingerprint(v) for v in provenance.values())):
            raise ValueError("Memory provenance accepts only verified fingerprint metadata")
        planned = HSSKPrototypeMemory(self.config)
        planned.load_state_dict(self.state_dict())
        old_keys = sorted(self._tracks)
        normalized = {k: F.normalize(centers[k].detach().cpu().float(), dim=0, eps=self.config.epsilon) for k in keys}
        # Greedy global geometric matching, stable distance/track/leaf ties.
        # Conflicting verified GT identities cannot exchange anchored tracks.
        edges = []
        if old_keys:
            d = squared_distances(torch.stack([normalized[k] for k in keys]),
                                  torch.stack([self._tracks[k].center for k in old_keys]))
            for i, k in enumerate(keys):
                for j, track_id in enumerate(old_keys):
                    old_gt = self._tracks[track_id].anchor_gt
                    if anchor_gt[k] is not None and old_gt is not None and anchor_gt[k] != old_gt:
                        continue
                    edges.append((float(d[i, j]), track_id, k))
        matched, used = {}, set()
        for _, track_id, k in sorted(edges):
            if k not in matched and track_id not in used:
                matched[k] = track_id
                used.add(track_id)
        for k in keys:
            if k in matched:
                old = self._tracks[matched[k]]
                # Only stored memory vectors are normalized. Lloyd centers remain means.
                center = F.normalize(self.config.momentum*old.center + (1-self.config.momentum)*centers[k].cpu().float(),
                                     dim=0, eps=self.config.epsilon)
                count, created, updates = old.support_count + support_counts[k], old.created_batch, old.updates+1
            else:
                matched[k] = planned._next_id
                planned._next_id -= 1
                center = normalized[k]
                count, created, updates = support_counts[k], batch_id, 1
            tid = matched[k]
            origin = dict(provenance, source="hssk_leaf_summary", label_origin=("ground_truth" if
                          anchor_gt[k] is not None and provenance["eligibility_manifest"] else "hierarchy_only"))
            planned._tracks[tid] = HSSKMemoryTrack(tid, tuple(center.tolist()), count, float(radii[k]),
                                                  batch_id, created, updates, anchor_gt[k], lineage, canonical(origin))
        active = set(matched.values())
        victims = sorted(set(planned._tracks)-active, key=lambda k: (planned._tracks[k].last_active_batch, k))
        while len(planned._tracks) > self.config.memory_capacity:
            if not victims:
                raise ValueError("Cannot evict an active HSSK leaf")
            del planned._tracks[victims.pop(0)]
            planned._evictions += 1
        planned._last_batch = batch_id
        return planned, matched

    def state_dict(self):
        p = dict(schema=MEMORY_SCHEMA, provider="hssk_online", config=asdict(self.config),
                 tracks=[asdict(self._tracks[k]) for k in sorted(self._tracks)],
                 next_id=self._next_id, last_batch=self._last_batch, evictions=self._evictions)
        return json.loads(canonical(dict(payload=p, state_hash=digest(p))))

    def load_state_dict(self, state):
        fields = {"schema", "provider", "config", "tracks", "next_id", "last_batch", "evictions"}
        if not isinstance(state, dict) or set(state) != {"payload", "state_hash"}:
            raise ValueError("Invalid HSSK memory envelope")
        p = state["payload"]
        if not isinstance(p, dict) or set(p) != fields or digest(p) != state["state_hash"]:
            raise ValueError("Corrupt HSSK memory hash/fields")
        if p["schema"] != MEMORY_SCHEMA or p["provider"] != "hssk_online" or p["config"] != asdict(self.config):
            raise ValueError("HSSK memory schema/provider/config mismatch")
        if (type(p["next_id"]) is not int or p["next_id"] >= 0 or type(p["last_batch"]) is not int
                or p["last_batch"] < -1 or type(p["evictions"]) is not int or p["evictions"] < 0
                or not isinstance(p["tracks"], list) or len(p["tracks"]) > self.config.memory_capacity):
            raise ValueError("Invalid HSSK memory bounds/allocator")
        restored = {}
        for row in p["tracks"]:
            if set(row) != set(HSSKMemoryTrack.__dataclass_fields__):
                raise ValueError("No raw images or historical per-sample features in memory schema")
            item = dict(row)
            item["center_values"] = tuple(item["center_values"])
            track = HSSKMemoryTrack(**item)
            origin = json.loads(track.provenance)
            if (set(origin) != {"source", "label_origin", "eligibility_manifest", "dataset_fingerprint", "split_fingerprint"}
                    or origin["source"] != "hssk_leaf_summary" or origin["label_origin"] not in ("ground_truth", "hierarchy_only")
                    or any(origin[k] is not None and not fingerprint(origin[k]) for k in
                           ("eligibility_manifest", "dataset_fingerprint", "split_fingerprint"))
                    or (origin["label_origin"] == "ground_truth" and (track.anchor_gt is None or
                        any(origin[k] is None for k in ("eligibility_manifest", "dataset_fingerprint", "split_fingerprint"))))):
                raise ValueError("Invalid HSSK memory provenance")
            if (type(track.track_id) is not int or not p["next_id"] < track.track_id < 0
                    or track.track_id in restored or len(track.center_values) != 768
                    or not torch.isfinite(track.center).all()
                    or (float(track.center.norm()) > self.config.epsilon and abs(float(track.center.norm())-1) > 1e-5)
                    or type(track.support_count) is not int or track.support_count < 2
                    or not math.isfinite(track.radius) or track.radius < 0
                    or type(track.last_active_batch) is not int or not 0 <= track.created_batch <= track.last_active_batch <= p["last_batch"]
                    or type(track.updates) is not int or track.updates < 1 or not fingerprint(track.lineage)
                    or (track.anchor_gt is not None and (type(track.anchor_gt) is not int or track.anchor_gt < 0))):
                raise ValueError("Invalid HSSK memory track")
            restored[track.track_id] = track
        self._tracks, self._next_id, self._last_batch, self._evictions = restored, p["next_id"], p["last_batch"], p["evictions"]


class LeafFitStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"


@dataclass(frozen=True, init=False)
class LeafFitResult:
    """Immutable copy-out LEAF result, intentionally not HierarchySnapshot."""
    _payload: str
    fit_hash: str

    def __init__(self, payload):
        data = canonical(payload)
        object.__setattr__(self, "_payload", data)
        object.__setattr__(self, "fit_hash", digest(payload))

    def state_dict(self):
        return json.loads(self._payload)

    @property
    def status(self):
        return LeafFitStatus(self.state_dict()["status"])

    @property
    def sample_to_leaf(self):
        return MappingProxyType({int(k): v for k, v in self.state_dict()["sample_to_leaf"].items()})

    @property
    def centers(self):
        return {int(k): torch.tensor(v, dtype=torch.float32) for k, v in self.state_dict()["centers"].items()}

    @property
    def selected_k(self):
        return self.state_dict()["selected_k"]


def candidate_counts(n, a):
    if type(n) is not int or type(a) is not int or n < 0 or not 0 <= a <= 16 or 2*a > n:
        raise ValueError("Invalid real support N/verified anchor count A")
    return tuple(a+free for free in range(0 if a else 1, min(16, n//2-a)+1))


def squared_distances(rows, centers):
    # N x K, never dimension x dimension. Explicit direct kernel avoids the
    # shape-dependent matrix-multiplication cdist mode and clamps roundoff.
    return torch.cdist(rows, centers, compute_mode="donot_use_mm_for_euclid_dist").square()


@dataclass
class CandidateFit:
    labels: torch.Tensor
    centers: torch.Tensor
    score: float
    diagnostics: dict


class HSSKLeafFitter:
    def __init__(self, config=None):
        self.config = config or HSSKLeafConfig()

    @torch.no_grad()
    @torch.autocast(device_type="cpu", enabled=False)
    def fit(self, current_features, sample_ids, *, replay=None, memory, seed, batch_id):
        if not isinstance(memory, HSSKPrototypeMemory) or memory.config != self.config:
            raise ValueError("Fit requires a compatible HSSK-owned pre-batch memory")
        if type(seed) is not int or not 0 <= seed < 2**63 or type(batch_id) is not int or batch_id < 0:
            raise ValueError("Invalid private HSSK seed/incoming batch ID")
        if batch_id <= memory._last_batch:
            raise ValueError("A successful HSSK batch cannot update memory twice")
        current, ids = checked_rows(current_features, sample_ids)
        current = F.normalize(current, dim=-1, eps=self.config.epsilon)
        r_rows, r_ids, r_labels = torch.empty(0, 768), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        replay_meta = None
        if replay is not None:
            if not isinstance(replay, VerifiedReplayFeatures):
                raise ValueError("Replay must be features of a selected verified GT reference batch")
            r_rows, r_ids, r_labels, replay_meta = replay.values()
            if replay_meta["batch_id"] != batch_id:
                raise ValueError("Frozen replay belongs to a different incoming batch")
            r_rows = F.normalize(r_rows, dim=-1, eps=self.config.epsilon)
        if set(ids.tolist()) & set(r_ids.tolist()):
            raise ValueError("Duplicate real sample IDs across current/replay support")
        classes = sorted(set(r_labels.tolist()))
        a, n = len(classes), len(current)+len(r_rows)
        real_ids = torch.cat((ids, r_ids))
        order = torch.argsort(real_ids)
        real_ids = real_ids[order]
        rows = torch.cat((current, r_rows))[order]
        is_current = torch.tensor([True]*len(current)+[False]*len(r_rows), dtype=torch.bool)[order]
        class_lookup = {c: i for i, c in enumerate(classes)}
        fixed = torch.tensor([-1]*len(current)+[class_lookup[k] for k in r_labels.tolist()], dtype=torch.long)[order]
        anchors = torch.stack([rows[fixed == c].mean(0) for c in range(a)]) if a else torch.empty(0, 768)
        before = memory.state_hash
        seeds, seed_ids = memory.seed_candidates(current)
        identity = dict(schema=LEAF_SCHEMA, config=asdict(self.config), seed=seed, batch_id=batch_id,
                        real_ids=real_ids.tolist(), normalized_real_rows=rows.tolist(),
                        current_ids=ids.tolist(), replay=replay_meta,
                        memory_hash_before=before)
        candidates, rejected = [], {}
        for k in candidate_counts(n, a):
            generator = torch.Generator(device="cpu")
            private = int(digest(dict(domain=LEAF_SCHEMA+"/kpp", seed=seed, batch=batch_id, k=k,
                                      config=asdict(self.config)))[:16], 16) % 2**63
            generator.manual_seed(private)
            initial, seed_order = self._initialize(current, ids, anchors, k, seeds, seed_ids, generator)
            if initial is None:
                rejected[str(k)] = "insufficient_distinct_seed_locations"
                continue
            result, reason = self._fit_candidate(rows, real_ids, is_current, fixed, initial, a)
            if result is None:
                rejected[str(k)] = reason
                continue
            result.diagnostics.update(seed_order=seed_order, private_seed=private)
            candidates.append((result.score, k, tuple(seed_order), result))
        counters = {"hssk.leaf.candidate_rejected."+v: list(rejected.values()).count(v) for v in set(rejected.values())}
        common = dict(schema=LEAF_SCHEMA, batch_id=batch_id, a=a, n=n, current_ids=ids.tolist(),
                      replay_ids=r_ids.tolist(), rejected_k=rejected, counters=counters,
                      input_hash=digest(identity), memory_hash_before=before, memory_seed_ids=list(seed_ids))
        if not candidates:
            counters["hssk.leaf.insufficient_support"] = 1
            return LeafFitResult(dict(common, status=LeafFitStatus.INSUFFICIENT_SUPPORT.value,
                reason="no_supported_leaf_candidate", selected_k=None, k_free=None, score=None,
                sample_to_leaf={}, centers={}, leaves={}, diagnostics={}, memory_hash_after=before))
        _, k, _, best = min(candidates, key=lambda entry: entry[:3])
        counts = {j: int((best.labels == j).sum()) for j in range(k)}
        raw_radii = {j: float(torch.linalg.vector_norm(rows[best.labels == j]-best.centers[j],
                                                      dim=-1).mean()) for j in range(k)}
        radii = {j: max(self.config.epsilon, value) for j, value in raw_radii.items()}
        counters["hssk.leaf.zero_radius_clamped"] = sum(r < self.config.epsilon for r in raw_radii.values())
        counters["hssk.leaf.success"] = 1
        anchor_gt = {j: classes[j] if j < a else None for j in range(k)}
        lineage = digest(dict(input_hash=common["input_hash"], labels=best.labels.tolist(), centers=best.centers.tolist(), k=k))
        provenance = {key: replay_meta[key] if replay_meta else None for key in
                      ("eligibility_manifest", "dataset_fingerprint", "split_fingerprint")}
        planned, group_ids = memory.updated_copy(dict(enumerate(best.centers)), counts, radii, anchor_gt,
                                                 batch_id, lineage, provenance)
        counters["hssk.memory.new_tracks"] = memory.next_id-planned.next_id
        counters["hssk.memory.evictions"] = planned._evictions-memory._evictions
        leaves = {str(group_ids[j]): dict(support_count=counts[j], current_count=int(((best.labels == j) & is_current).sum()),
                                         anchored=j < a, anchor_gt=anchor_gt[j], radius=radii[j]) for j in range(k)}
        result = LeafFitResult(dict(common, status=LeafFitStatus.SUCCESS.value, reason=None, selected_k=k,
                    k_free=k-a, score=best.score, diagnostics=best.diagnostics,
                    sample_to_leaf={str(key): group_ids[int(label)] for key, label in zip(real_ids.tolist(), best.labels.tolist())},
                    centers={str(group_ids[j]): best.centers[j].tolist() for j in range(k)}, leaves=leaves,
                    memory_hash_after=planned.state_hash))
        # Publish only after every candidate/result/state check has succeeded.
        memory.load_state_dict(planned.state_dict())
        return result

    def _initialize(self, current, ids, anchors, k, memory_seeds, memory_ids, generator):
        centers = [row.clone() for row in anchors]
        pool = [row for row in current] + [row for row in memory_seeds]
        pool_tensor = torch.stack(pool) if pool else torch.empty(0, 768)
        tags = [(0, k) for k in ids.tolist()] + [(1, k) for k in memory_ids]
        selected = []
        while len(centers) < k:
            distances = (squared_distances(pool_tensor, torch.stack(centers)).min(dim=1).values
                         if centers else torch.ones(len(pool_tensor)))
            available = torch.where(distances > self.config.epsilon**2)[0]
            if not len(available):
                return None, selected
            chosen = int(torch.multinomial(distances[available].double(), 1, generator=generator))
            index = int(available[chosen])
            centers.append(pool[index].clone())
            selected.append(tags[index])
        return torch.stack(centers), selected

    def _assign(self, rows, centers, fixed, balanced):
        distances = squared_distances(rows, centers)
        labels = distances.argmin(dim=1)
        anchored = fixed >= 0
        labels[anchored] = fixed[anchored]
        if balanced:
            capacity = math.ceil(len(rows)/len(centers))
            limits = [max(capacity, int((fixed == j).sum())) for j in range(len(centers))]
            displaced = []
            for j in range(len(centers)):
                free_rows = torch.where((labels == j) & ~anchored)[0].tolist()
                free_rows.sort(key=lambda i: (float(distances[i, j]), i))
                keep = limits[j] - int((fixed == j).sum())
                for i in free_rows[keep:]:
                    labels[i] = -1
                    displaced.append(i)
            for i in sorted(displaced):
                available = [j for j in range(len(centers)) if int((labels == j).sum()) < limits[j]]
                if not available:
                    raise ValueError("Structural capacity mismatch in balanced HSSK")
                labels[i] = min(available, key=lambda j: (float(distances[i, j]), j))
        return labels, distances

    def _repair_empty(self, rows, ids, is_current, labels, centers, a, distances):
        reseeded = []
        for empty in range(a, len(centers)):
            if (labels == empty).any():
                continue
            eligible = []
            for j in range(len(centers)):
                members = torch.where((labels == j) & is_current)[0].tolist()
                # Transfer two distinct current observations without invalidating donor support.
                if len(members) < (2 if j < a else 4):
                    continue
                eligible.extend(members)
            if not eligible:
                return None
            seed = min(eligible, key=lambda i: (-float(distances[i, labels[i]]), int(ids[i])))
            donor = int(labels[seed])
            others = [i for i in eligible if int(labels[i]) == donor and i != seed]
            other = min(others, key=lambda i: (-float(distances[i, donor]), int(ids[i])))
            centers[empty] = rows[seed]  # Real high-residual seed, never replay/memory.
            labels[seed] = labels[other] = empty
            reseeded.append(int(ids[seed]))
        return reseeded

    def _support_reason(self, labels, is_current, k, a):
        for j in range(k):
            if int((labels == j).sum()) < 2:
                return "post_refinement_min_real_support"
            if j >= a and int(((labels == j) & is_current).sum()) < 2:
                return "post_refinement_min_current_support"
        return None

    def _fit_candidate(self, rows, ids, is_current, fixed, initial, a):
        centers = initial.clone()
        diagnostics = {}
        all_reseeds = []
        for phase in ("balanced", "refinement"):
            converged = False
            for iteration in range(self.config.max_iterations):
                previous = centers.clone()
                labels, distances = self._assign(rows, centers, fixed, balanced=phase == "balanced")
                reseeds = self._repair_empty(rows, ids, is_current, labels, centers, a, distances)
                if reseeds is None:
                    return None, "empty_free_center_without_supported_current_donor"
                all_reseeds.extend(reseeds)
                for j in range(0 if phase == "refinement" else a, len(centers)):
                    if not (labels == j).any():
                        return None, "empty_center"
                    centers[j] = rows[labels == j].mean(0)
                shift = float(torch.linalg.vector_norm(centers-previous, dim=1).sum())**2
                if shift < self.config.tolerance:
                    converged = True
                    break
            diagnostics[phase] = dict(iterations=iteration+1, converged=converged, final_shift=shift)
        reason = self._support_reason(labels, is_current, len(centers), a)
        if reason:
            return None, reason
        if not torch.equal(labels[fixed >= 0], fixed[fixed >= 0]):
            raise ValueError("Structural GT anchor violation")
        inertia = float((rows-centers[labels]).square().sum(dim=1).double().mean())
        score = inertia + self.config.k_penalty*(len(centers)-a)
        diagnostics.update(mean_real_squared_distance=inertia, empty_reseed_sample_ids=all_reseeds)
        return CandidateFit(labels, centers, score, diagnostics), None
