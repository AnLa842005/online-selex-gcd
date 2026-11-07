"""Detached HSSK snapshots with locked coarsening and transactional publication."""
from collections import Counter
from dataclasses import asdict
import torch
from torch.nn import functional as F
from .selex_config import ReferenceConfig
from .selex_hierarchy_provider import (HierarchyBuildResult, HierarchyBuildStatus, HierarchyProviderKind,
    HierarchySnapshot, HierarchySupportPool, SampleHierarchyAssignment, SCHEMA, digest, validate_state)
from .selex_hssk_online import (HSSKLeafFitter, HSSKPrototypeMemory, LeafFitStatus, squared_distances,
                               VerifiedReplayFeatures)
from .selex_hssk_hierarchy import coarsen_children, coarsening_stop_info


class HSSKProviderImplementation:
    kind = HierarchyProviderKind.HSSK_ONLINE

    def __init__(self, config=None):
        self.config = config or ReferenceConfig(hierarchy_source="hssk_online")
        if self.config.hierarchy_source != "hssk_online":
            raise ValueError("HSSK provider requires explicit hssk_online configuration")
        self.memory = HSSKPrototypeMemory()
        self.counters = Counter()
        self.previous = None
        self.identity = None
        self.last_batch = -1
        self.last_status = None

    def _identity(self, context):
        fields = ("model_lineage", "dataset", "feature_version", "seed_protocol")
        if any(not isinstance(context.get(k), str) or not context[k] for k in fields):
            raise ValueError("HSSK build needs explicit model/dataset/feature/seed-protocol identity")
        return dict(**{k: context[k] for k in fields}, seed=context.get("seed"), config=digest(asdict(self.config)), schema=SCHEMA)

    @torch.no_grad()
    def build(self, support, replay, context):
        before = self.state_dict()
        try:
            return self._build(support, replay, context)
        except Exception:
            self.load_state_dict(before)
            self.counters["hssk.error"] += 1
            raise

    @torch.no_grad()
    def _build(self, support, replay, context):
        if not isinstance(support, HierarchySupportPool) or not isinstance(context, dict):
            raise ValueError("HSSK requires a typed support pool and context")
        identity = self._identity(context)
        batch, seed = context.get("batch"), context.get("seed")
        if type(batch) is not int or batch <= self.last_batch or type(seed) is not int:
            raise ValueError("Hierarchy may be prepared only once per incoming batch")
        rows = F.normalize(support.current_features, dim=-1, eps=self.config.epsilon)
        ids = support.sample_ids
        before = self.memory.state_hash
        plan = HSSKPrototypeMemory()
        plan.load_state_dict(self.memory.state_dict())
        self.counters["hssk.batch_decisions"] += 1
        reason = "empty_current_support" if not len(ids) else None
        if self.identity is not None and identity != self.identity:
            reason = "context_identity_mismatch"
        leaf = None
        if reason is None:
            leaf = HSSKLeafFitter().fit(rows, ids, replay=replay, memory=plan, seed=seed, batch_id=batch)
            if leaf.status != LeafFitStatus.SUCCESS:
                reason = "insufficient_leaf_support"
        if reason is not None:
            result = self._reuse(rows, ids, replay, identity, batch, reason)
            self.last_batch, self.last_status = batch, result.status.value
            return result
        value = leaf.state_dict()
        mappings = [{int(key): group for key, group in value["sample_to_leaf"].items() if int(key) in set(ids.tolist())}]
        centers, radii = [leaf.centers], [{int(k): v["radius"] for k, v in value["leaves"].items()}]
        counts = {int(k): v["support_count"] for k, v in value["leaves"].items()}
        anchors = {int(k): (v["anchor_gt"],) if v["anchored"] else () for k, v in value["leaves"].items()}
        groups, child_maps, support_counts = [anchors], [], [counts]
        stop = dict(reason="projection_prefix_limit")
        # A level l consumes prefix D//2**(l+1); never build a zero-width level.
        while self.config.output_dim // 2**(len(mappings)+1) > 0:
            higher = coarsen_children(centers=centers[-1], support_counts=counts,
                                     anchor_groups=anchors, seed=seed, level=len(mappings))
            if higher is None:
                stop = coarsening_stop_info(anchors)
                break
            local = sorted(higher["centers"])
            allocated = {k: plan._next_id-i for i, k in enumerate(local)}
            plan._next_id -= len(local)  # Only on proposed state; never on a failed/reused build.
            parent_map = {child: allocated[parent] for child, parent in higher["mapping"].items()}
            child_maps.append(parent_map)
            mappings.append({key: parent_map[child] for key, child in mappings[-1].items()})
            centers.append({allocated[k]: v for k, v in higher["centers"].items()})
            radii.append({allocated[k]: v for k, v in higher["radii"].items()})
            counts = {allocated[k]: v for k, v in higher["counts"].items()}
            anchors = {allocated[k]: v for k, v in higher["anchor_groups"].items()}
            groups.append(anchors)
            support_counts.append(counts)
        stop["status"] = "LEAF_ONLY" if len(mappings) == 1 else "MULTI_LEVEL"
        diagnostics = dict(status="BUILD", higher_stop=stop, hierarchy_type=stop["status"], k=value["selected_k"], a=value["a"], k_free=value["k_free"],
            leaf_fit_hash=leaf.fit_hash, leaf_score=value["score"], levels=len(mappings),
            memory_hash_before=before, memory_hash_after=plan.state_hash,
            child_maps=child_maps, anchor_groups=groups, support_counts=support_counts,
            identity=identity, counters=value["counters"])
        snapshot = self._snapshot(ids.tolist(), mappings, centers, radii, batch, diagnostics)
        self.memory.load_state_dict(plan.state_dict())
        self.previous, self.identity = snapshot, identity
        self.last_batch, self.last_status = batch, "BUILD"
        self.counters["hssk.build"] += 1
        self.counters["hssk.build."+stop["status"].lower()] += 1
        self.counters["hssk.levels."+str(len(mappings))] += 1
        self.counters["hssk.used_higher"] += int(len(mappings) > 1)
        self.counters["hssk.batch_levels."+str(len(mappings))] += 1
        self.counters["hssk.higher.stop."+stop["reason"]] += 1
        self.counters.update(value["counters"])
        return HierarchyBuildResult(self.kind, HierarchyBuildStatus.BUILD, snapshot)

    def _snapshot(self, ids, mappings, centers, radii, batch, diagnostics):
        assignments = [SampleHierarchyAssignment(k, mappings[0][k], tuple(m[k] for m in mappings[1:])) for k in ids]
        return HierarchySnapshot(self.kind, assignments, centers, radii, source_batch=batch,
            feature_version=diagnostics["identity"]["feature_version"], config_version=digest(asdict(self.config)),
            diagnostics=diagnostics)

    def _reuse(self, rows, ids, replay, identity, batch, failed_reason):
        old = self.previous
        reason = failed_reason
        if old is not None and identity == self.identity and batch-old.source_batch == 1 and self.last_status != "REUSE" and len(ids):
            leaf_keys = sorted(old.centers[0])
            distances = squared_distances(rows, torch.stack([old.centers[0][k] for k in leaf_keys])).sqrt()
            distance, nearest = distances.min(dim=1)
            mapping = {key: leaf_keys[int(label)] for key, label in zip(ids.tolist(), nearest.tolist())}
            counts = Counter(mapping.values())
            current_counts = counts.copy()
            anchor_groups = {int(k): v for k, v in old.diagnostics["anchor_groups"][0].items()}
            reason = "distance_exceeds_0.5" if (distance > .5).any() else None
            if replay is not None:
                if not isinstance(replay, VerifiedReplayFeatures):
                    raise ValueError("Reuse accepts only verified replay features")
                _, replay_ids, gt, meta = replay.values()
                if meta["batch_id"] != batch or set(replay_ids.tolist()) & set(ids.tolist()):
                    raise ValueError("Reuse replay IDs/batch mismatch")
                for label in gt.tolist():
                    matched = [k for k in leaf_keys if anchor_groups[k] == [label]]
                    if len(matched) != 1:
                        reason = "incompatible_gt_anchor"
                        break
                    counts[matched[0]] += 1
            if any(counts[k] < 2 or (not anchor_groups[k] and current_counts[k] < 2) for k in leaf_keys):
                reason = reason or "reuse_leaf_support"
            if reason is None:
                mappings, count_levels = [mapping], [dict(counts)]
                for level, child_map in enumerate(old.diagnostics["child_maps"]):
                    child_map = {int(k): v for k, v in child_map.items()}
                    parents = Counter()
                    for child, count in counts.items():
                        parents[child_map[child]] += count
                    if any(v < 2 for v in parents.values()):
                        reason = "reuse_parent_support"
                        break
                    mappings.append({key: child_map[value] for key, value in mappings[-1].items()})
                    count_levels.append(dict(parents))
                    counts = parents
                if reason is None:
                    diag = dict(old.diagnostics, status="REUSE", reused_for_batch=batch, age=1,
                        support_counts=count_levels, memory_hash_before=self.memory.state_hash,
                        memory_hash_after=self.memory.state_hash, previous_snapshot_hash=old.snapshot_hash,
                        reuse_hash=digest(dict(previous=old.snapshot_hash, batch=batch, assignment=mapping)))
                    snapshot = self._snapshot(ids.tolist(), mappings, old.centers, old.radii, old.source_batch, diag)
                    self.counters["hssk.reuse"] += 1
                    self.counters["hssk.used_higher"] += int(len(mappings) > 1)
                    self.counters["hssk.batch_levels."+str(len(mappings))] += 1
                    return HierarchyBuildResult(self.kind, HierarchyBuildStatus.REUSE, snapshot)
        self.counters["hssk.reuse_rejected."+str(reason)] += 1
        self.counters["hssk.insufficient_support."+failed_reason] += 1
        if self.config.insufficient_support_policy == "reuse_then_error":
            raise ValueError("HSSK insufficient support; reuse rejected: " + str(reason))
        self.counters["selex.batch_skipped"] += 1
        return HierarchyBuildResult(self.kind, HierarchyBuildStatus.SKIP, reason=str(reason))

    def state_dict(self):
        p = dict(schema=SCHEMA, provider=self.kind.value, version="hssk-full-smoke-v1", config=asdict(self.config),
                 memory=self.memory.state_dict(), previous=self.previous.state_dict() if self.previous else None,
                 identity=self.identity, last_batch=self.last_batch, last_status=self.last_status, counters=dict(self.counters))
        return dict(payload=p, state_hash=digest(p))

    def load_state_dict(self, state):
        validate_state(state, self.kind)
        p = state["payload"]
        if p["config"] != asdict(self.config) or p.get("version") != "hssk-full-smoke-v1":
            raise ValueError("HSSK provider config/version mismatch")
        memory = HSSKPrototypeMemory()
        memory.load_state_dict(p["memory"])
        previous = HierarchySnapshot.from_state_dict(p["previous"]) if p["previous"] else None
        if previous is not None and previous.provider != self.kind:
            raise ValueError("HSSK cannot restore prototype snapshots")
        if (type(p["last_batch"]) is not int or p["last_batch"] < -1 or p["last_status"] not in (None, "BUILD", "REUSE", "SKIP")
                or any(type(v) is not int or v < 0 for v in p["counters"].values())):
            raise ValueError("Invalid HSSK batch/status/counters")
        if previous is not None:
            if (previous.source_batch > p["last_batch"] or memory._last_batch != previous.source_batch
                    or previous.diagnostics["identity"] != p["identity"]
                    or previous.diagnostics["memory_hash_after"] != memory.state_hash
                    or previous.config_version != digest(asdict(self.config))):
                raise ValueError("HSSK snapshot/memory/lineage state mismatch")
        elif memory.tracks or p["identity"] is not None:
            raise ValueError("HSSK memory needs its successful source snapshot")
        self.memory, self.previous, self.identity = memory, previous, p["identity"]
        self.last_batch, self.last_status, self.counters = p["last_batch"], p["last_status"], Counter(p["counters"])
