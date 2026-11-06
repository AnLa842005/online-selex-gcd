"""Provider contracts and the explicitly selected legacy prototype ablation.

HSSK is deliberately incomplete in Phase 1. No clustering, K selection,
coarsening or HSSK prototype update is implemented here.
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Optional, Tuple

import torch

from .selex_hierarchy import EMAHierarchy, integer_ids

SCHEMA = "hierarchy-provider-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class HierarchyProviderKind(str, Enum):
    HSSK_ONLINE = "hssk_online"
    DEAN_PROTOTYPES = "dean_prototypes"


class HierarchyBuildStatus(str, Enum):
    BUILD = "BUILD"
    REUSE = "REUSE"
    SKIP = "SKIP"


class HSSKImplementationIncomplete(ValueError):
    """An intentional fail-closed research gate, not a degeneracy fallback."""


@dataclass(frozen=True)
class SampleHierarchyAssignment:
    sample_id: int
    leaf_id: int
    ancestor_ids: Tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.sample_id) is not int or self.sample_id < 0 or type(self.leaf_id) is not int:
            raise ValueError("Assignments require stable nonnegative sample IDs and integer groups")
        if not isinstance(self.ancestor_ids, tuple) or any(type(k) is not int for k in self.ancestor_ids):
            raise ValueError("Ancestor IDs must be an immutable integer tuple")


@dataclass(frozen=True, init=False)
class HierarchySupportPool:
    """Detached current observations only; no DEAN targets or historical rows.

    Replay/memory arguments remain separate provider-owned inputs. Accessors
    return fresh tensors so callers cannot mutate captured observations.
    """
    _ids: tuple
    _features: tuple

    def __init__(self, sample_ids, current_features):
        ids = integer_ids(sample_ids, "support sample IDs").cpu().tolist()
        if len(set(ids)) != len(ids) or any(k < 0 for k in ids):
            raise ValueError("Support IDs must be unique nonnegative sample IDs")
        if current_features.shape != (len(ids), 768) or not torch.isfinite(current_features).all():
            raise ValueError("Support features must be finite [B,768]")
        object.__setattr__(self, "_ids", tuple(ids))
        object.__setattr__(self, "_features", tuple(tuple(r) for r in current_features.detach().cpu().float().tolist()))

    @property
    def sample_ids(self):
        return torch.tensor(self._ids, dtype=torch.long)

    @property
    def current_features(self):
        return torch.tensor(self._features, dtype=torch.float32).reshape(len(self._ids), 768)


@dataclass(frozen=True, init=False)
class HierarchySnapshot:
    """Value snapshot: immutable canonical storage, detached copy-out tensors.

    Mappings/centers/radii satisfy the existing loss lookup protocol but keys
    are SAMPLE IDs. The legacy EMA snapshot class remains untouched.
    """
    _payload: str
    snapshot_hash: str

    def __init__(self, provider, assignments, centers=(), radii=(), source_batch=0,
                 feature_version="dino768-eval-v1", config_version="v1",
                 schema_version=SCHEMA, frozen=True, diagnostics=None):
        kind = HierarchyProviderKind(provider)
        assignments = tuple(assignments)
        if not assignments or any(not isinstance(a, SampleHierarchyAssignment) for a in assignments):
            raise ValueError("A valid hierarchy requires explicit sample assignments; use SKIP instead")
        ids = [a.sample_id for a in assignments]
        depth = len(assignments[0].ancestor_ids)
        if len(set(ids)) != len(ids) or any(len(a.ancestor_ids) != depth for a in assignments):
            raise ValueError("Duplicate IDs or inconsistent hierarchy depth")
        if type(source_batch) is not int or source_batch < 0 or frozen is not True:
            raise ValueError("Snapshot must be frozen with nonnegative source batch")
        if schema_version != SCHEMA or not feature_version or not config_version:
            raise ValueError("Invalid hierarchy feature/config/schema version")
        levels = depth + 1
        if bool(centers) != bool(radii) or (centers and (len(centers) != levels or len(radii) != levels)):
            raise ValueError("Centers/radii must align with all hierarchy levels")
        mappings = [{a.sample_id: (a.leaf_id, *a.ancestor_ids)[i] for a in assignments} for i in range(levels)]
        for child, parent in zip(mappings, mappings[1:]):
            parents = {}
            for sample_id, child_id in child.items():
                if child_id in parents and parents[child_id] != parent[sample_id]:
                    raise ValueError("A child group cannot have inconsistent ancestor assignments")
                parents[child_id] = parent[sample_id]
        encoded_centers, encoded_radii, used = [], [], set()
        for i, mapping in enumerate(mappings):
            groups = set(centers[i]) if centers else set(mapping.values())
            if groups & used:
                raise ValueError("Group IDs must be globally unique across levels")
            used |= groups
            if centers:
                if set(centers[i]) != set(radii[i]) or not set(mapping.values()) <= set(centers[i]):
                    raise ValueError("Hierarchy geometry/group IDs mismatch")
                row, radius = {}, {}
                for key, value in centers[i].items():
                    if type(key) is not int or value.shape != (768,) or not torch.isfinite(value).all():
                        raise ValueError("Hierarchy centers must be finite detached 768-D vectors")
                    r = float(radii[i][key])
                    if not 0 <= r < float("inf"):
                        raise ValueError("Invalid hierarchy radius")
                    row[str(key)] = value.detach().cpu().float().tolist()
                    radius[str(key)] = r
                encoded_centers.append(row)
                encoded_radii.append(radius)
        value = dict(provider=kind.value, schema_version=schema_version,
                     feature_version=feature_version, config_version=config_version,
                     source_batch=source_batch, frozen=True,
                     assignments=[asdict(a) for a in sorted(assignments, key=lambda a: a.sample_id)],
                     centers=encoded_centers, radii=encoded_radii, diagnostics=diagnostics or {})
        # JSON object keys become strings. Normalize before hashing so nested
        # integer-keyed child/parent diagnostics round-trip canonically too.
        payload = canonical(json.loads(canonical(value)))
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "snapshot_hash", hashlib.sha256(payload.encode()).hexdigest())

    def _value(self):
        return json.loads(self._payload)

    @property
    def provider(self):
        return HierarchyProviderKind(self._value()["provider"])

    @property
    def source_batch(self):
        return self._value()["source_batch"]

    @property
    def feature_version(self):
        return self._value()["feature_version"]

    @property
    def config_version(self):
        return self._value()["config_version"]

    @property
    def schema_version(self):
        return self._value()["schema_version"]

    @property
    def frozen(self):
        return True

    @property
    def diagnostics(self):
        return self._value()["diagnostics"]

    @property
    def mappings(self):
        rows = self._value()["assignments"]
        return tuple(MappingProxyType({a["sample_id"]: [a["leaf_id"], *a["ancestor_ids"]][i] for a in rows})
                     for i in range(1 + len(rows[0]["ancestor_ids"])))

    @property
    def sample_to_leaf(self):
        return self.mappings[0]

    @property
    def sample_to_ancestors(self):
        return self.mappings[1:]

    @property
    def centers(self):
        return tuple({int(k): torch.tensor(v, dtype=torch.float32) for k, v in row.items()}
                     for row in self._value()["centers"])

    @property
    def radii(self):
        return tuple({int(k): v for k, v in row.items()} for row in self._value()["radii"])

    def assignments(self, sample_ids, device=None):
        ids = integer_ids(sample_ids, "lookup sample IDs").cpu().tolist()
        if not set(ids) <= set(self.sample_to_leaf):
            raise ValueError("Unknown sample ID in frozen hierarchy")
        return [torch.tensor([row[k] for k in ids], dtype=torch.long, device=device) for row in self.mappings]

    def state_dict(self):
        return dict(payload=self._value(), snapshot_hash=self.snapshot_hash)

    @classmethod
    def from_state_dict(cls, state):
        if set(state) != {"payload", "snapshot_hash"} or digest(state["payload"]) != state["snapshot_hash"]:
            raise ValueError("Corrupt hierarchy snapshot hash")
        value = dict(state["payload"])
        rows = value.pop("assignments")
        assignments = [SampleHierarchyAssignment(r["sample_id"], r["leaf_id"], tuple(r["ancestor_ids"])) for r in rows]
        centers = [{int(k): torch.tensor(v, dtype=torch.float32) for k, v in c.items()} for c in value.pop("centers")]
        radii = [{int(k): v for k, v in c.items()} for c in value.pop("radii")]
        result = cls(assignments=assignments, centers=centers, radii=radii, **value)
        if result.snapshot_hash != state["snapshot_hash"]:
            raise ValueError("Noncanonical hierarchy state")
        return result


@dataclass(frozen=True)
class HierarchyBuildResult:
    provider: HierarchyProviderKind
    status: HierarchyBuildStatus
    snapshot: Optional[HierarchySnapshot] = None
    reason: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.provider, HierarchyProviderKind) or not isinstance(self.status, HierarchyBuildStatus):
            raise ValueError("Typed provider/status required")
        if self.status == HierarchyBuildStatus.SKIP:
            if self.snapshot is not None or not self.reason:
                raise ValueError("SKIP requires a reason and no valid snapshot")
        elif not isinstance(self.snapshot, HierarchySnapshot) or self.snapshot.provider != self.provider:
            raise ValueError("BUILD/REUSE require a compatible valid snapshot")

    def require_snapshot(self):
        if self.status == HierarchyBuildStatus.SKIP:
            raise ValueError("Skipped SelEx batch has no valid hierarchy: " + self.reason)
        return self.snapshot


class HierarchyProvider(ABC):
    @abstractmethod
    def build(self, support, dean_leaf_ids=None, labeled_replay=None, prototype_memory=None, context=None):
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state):
        raise NotImplementedError


class DeanPrototypeHierarchyProvider(HierarchyProvider):
    """Explicit ablation. Delegation performs exactly one original EMA update."""
    kind = HierarchyProviderKind.DEAN_PROTOTYPES

    def __init__(self, config, base_classes=160):
        if config.hierarchy_source != self.kind.value:
            raise ValueError("Prototype provider is an explicit ablation, never an HSSK fallback")
        self.ema = EMAHierarchy(config, base_classes)
        self.legacy_snapshot = None

    def build_legacy(self, features, labels, sample_ids):
        # Numerical path uses original input dtype/order, not SupportPool's FP32 copy.
        self.legacy_snapshot = self.ema.update(features, labels, sample_ids)
        return self.legacy_snapshot

    def build(self, support, dean_leaf_ids=None, labeled_replay=None, prototype_memory=None, context=None):
        if prototype_memory is not None or labeled_replay is not None:
            raise ValueError("Legacy hierarchy does not accept HSSK memory/replay inputs")
        old = self.build_legacy(support.current_features, dean_leaf_ids, support.sample_ids)
        if not len(support.sample_ids):
            return HierarchyBuildResult(self.kind, HierarchyBuildStatus.SKIP, reason="empty_current_support")
        labels = integer_ids(dean_leaf_ids, "DEAN leaves").tolist()
        rows = [SampleHierarchyAssignment(key, label, tuple(m[label] for m in old.mappings[1:]))
                for key, label in zip(support.sample_ids.tolist(), labels)]
        snapshot = HierarchySnapshot(self.kind, rows, old.centers, old.radii,
                                     source_batch=(context or {}).get("batch", self.ema.tick-1),
                                     config_version=digest(asdict(self.ema.config)), diagnostics={"ablation": True})
        return HierarchyBuildResult(self.kind, HierarchyBuildStatus.BUILD, snapshot)

    def state_dict(self):
        e = self.ema
        payload = dict(schema=SCHEMA, provider=self.kind.value, config=asdict(e.config), base_classes=e.base_classes,
                       prototypes={str(k): v.tolist() for k, v in e.prototypes.items()},
                       updated={str(k): v for k, v in e.updated.items()}, counts={str(k): v for k, v in e.counts.items()},
                       tick=e.tick, next_parent=e.next_parent, counters=dict(e.counters))
        return dict(payload=payload, state_hash=digest(payload))

    def load_state_dict(self, state):
        validate_state(state, self.kind)
        p = state["payload"]
        if p["config"] != asdict(self.ema.config) or p["base_classes"] != self.ema.base_classes:
            raise ValueError("Prototype config identity mismatch")
        prototypes = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in p["prototypes"].items()}
        if any(v.shape != (768,) or not torch.isfinite(v).all() for v in prototypes.values()):
            raise ValueError("Invalid prototype state")
        if set(prototypes) != {int(k) for k in p["counts"]} or set(prototypes) != {int(k) for k in p["updated"]}:
            raise ValueError("Misaligned prototype state")
        self.ema.prototypes = prototypes
        self.ema.updated = {int(k): v for k, v in p["updated"].items()}
        self.ema.counts = {int(k): v for k, v in p["counts"].items()}
        self.ema.tick, self.ema.next_parent = p["tick"], p["next_parent"]
        self.ema.counters.clear()
        self.ema.counters.update(p["counters"])
        self.legacy_snapshot = None


def validate_state(state, kind):
    if not isinstance(state, dict) or set(state) != {"payload", "state_hash"}:
        raise ValueError("Invalid provider state envelope")
    p = state["payload"]
    if digest(p) != state["state_hash"] or p.get("schema") != SCHEMA or p.get("provider") != kind.value:
        raise ValueError("Provider state hash/schema/identity mismatch")


class OnlineHSSKHierarchyProvider(HierarchyProvider):
    kind = HierarchyProviderKind.HSSK_ONLINE
    implementation_status = "smoke_ready, full_train_blocked"

    def __init__(self, config=None):
        from .selex_hssk_provider import HSSKProviderImplementation
        self.implementation = HSSKProviderImplementation(config)

    @property
    def memory(self):
        return self.implementation.memory

    @property
    def counters(self):
        return self.implementation.counters

    def build(self, support, dean_leaf_ids=None, labeled_replay=None, prototype_memory=None, context=None):
        if prototype_memory is not None:
            raise ValueError("HSSK owns its memory; external/DEAN EMA is forbidden")
        return self.implementation.build(support, labeled_replay, context)

    def state_dict(self):
        return self.implementation.state_dict()

    def load_state_dict(self, state):
        self.implementation.load_state_dict(state)


def provider_metadata(config):
    kind = HierarchyProviderKind(config.hierarchy_source)
    return dict(hierarchy_source=kind.value, hierarchy_provider_status=("smoke_ready, full_train_blocked" if kind ==
                HierarchyProviderKind.HSSK_ONLINE else "ablation"))


class FullTrainingBlocked(ValueError):
    pass


def ensure_provider_ready(config, bounded_smoke=False):
    if config.hierarchy_source == HierarchyProviderKind.HSSK_ONLINE.value and not bounded_smoke:
        raise FullTrainingBlocked("Full training blocked: Phase 4 resume/provenance readiness required")


def create_provider(config, base_classes=160):
    kind = HierarchyProviderKind(config.hierarchy_source)
    if kind == HierarchyProviderKind.HSSK_ONLINE:
        return OnlineHSSKHierarchyProvider(config)
    return DeanPrototypeHierarchyProvider(config, base_classes)
