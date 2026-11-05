"""Bounded EMA DEAN prototypes and detached, globally identified higher groups.

No leaf discovery, raw history, labelled replay or future samples are used here.
Higher prototype-only grouping is an intentional online adaptation of HSSK.
"""
from collections import Counter
from dataclasses import dataclass
import heapq

import torch
from torch.nn import functional as F


def integer_ids(values, name):
    if isinstance(values, (tuple, list)) and not values:
        return torch.empty(0, dtype=torch.long)
    values = torch.as_tensor(values)
    if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} requires a one-dimensional integer tensor")
    return values.long()


@dataclass(frozen=True)
class HierarchySnapshot:
    leaf_ids: tuple
    mappings: tuple
    centers: tuple
    radii: tuple

    def __post_init__(self):
        if len(self.mappings) != len(self.centers) or len(self.mappings) != len(self.radii):
            raise ValueError("Hierarchy level arrays must align")
        if len(set(self.leaf_ids)) != len(self.leaf_ids) or any(k < 0 for k in self.leaf_ids):
            raise ValueError("Hierarchy leaves must preserve unique nonnegative DEAN IDs")
        used = set()
        for level, mapping in enumerate(self.mappings):
            ids = set(mapping.values())
            if set(mapping) != set(self.leaf_ids) or ids != set(self.centers[level]) or ids != set(self.radii[level]) or used & ids:
                raise ValueError("Hierarchy ID mappings/centers/radii must be globally aligned")
            used |= ids

    def assignments(self, leaves, device=None):
        leaves = integer_ids(leaves, "DEAN leaves")
        if not set(leaves.tolist()) <= set(self.leaf_ids):
            raise ValueError("Unknown DEAN leaf in fixed hierarchy snapshot")
        output = []
        for mapping in self.mappings:
            if any(int(v) not in mapping for v in leaves.tolist()):
                raise ValueError("Unknown DEAN leaf in fixed hierarchy snapshot")
            output.append(torch.tensor([mapping[int(v)] for v in leaves.tolist()], device=device, dtype=torch.long))
        return output


class EMAHierarchy:
    def __init__(self, config, base_classes=160):
        self.config, self.base_classes = config, base_classes
        self.prototypes, self.updated, self.counts = {}, {}, {}
        self.tick = 0
        self.next_parent = -1
        self.counters = Counter()

    @torch.no_grad()
    @torch.autocast(device_type="cpu", enabled=False)
    def update(self, features, labels, sample_ids):
        labels = integer_ids(labels, "prototype labels").cpu()
        sample_ids = integer_ids(sample_ids, "sample IDs").cpu()
        if features.ndim != 2 or features.shape != (len(labels), 768) or len(sample_ids) != len(labels):
            raise ValueError("Prototype features/labels/IDs must align at 768 dimensions")
        if len(set(sample_ids.tolist())) != len(sample_ids) or not torch.isfinite(features).all() or (labels < 0).any():
            raise ValueError("Invalid/duplicate prototype sample IDs, labels or nonfinite features")
        active = set(labels.tolist())
        if len(active) > self.config.prototype_capacity:
            raise ValueError("Prototype capacity cannot contain current batch class support")
        self.tick += 1
        needed = len(set(self.prototypes) | active) - self.config.prototype_capacity
        victims = sorted(set(self.prototypes) - active, key=lambda key: (self.updated[key], key))
        for key in victims[:max(0, needed)]:
            del self.prototypes[key], self.updated[key]
            del self.counts[key]
            self.counters["prototype_evictions"] += 1
        # Prototype state is always detached FP32, including under AMP.
        normalized = F.normalize(features.detach().to(device="cpu", dtype=torch.float32),
                                 dim=-1, eps=self.config.epsilon)
        for key in sorted(active):
            center = normalized[labels == key].mean(0)
            if key in self.prototypes:
                center = self.config.prototype_momentum * self.prototypes[key] + (1-self.config.prototype_momentum) * center
            self.prototypes[key] = F.normalize(center, dim=0, eps=self.config.epsilon).clone()
            self.updated[key] = self.tick
            self.counts[key] = self.counts.get(key, 0) + int((labels == key).sum())
        if not active:
            self.counters["prototype_empty_batch"] += 1
        return self.snapshot()

    @torch.no_grad()
    @torch.autocast(device_type="cpu", enabled=False)
    def snapshot(self):
        self.counters["hierarchy_rebuilds"] += 1
        leaves = tuple(sorted(self.prototypes))
        if not leaves:
            return HierarchySnapshot((), (), (), ())
        groups = [tuple([key]) for key in leaves]
        mappings, centers, radii = [], [], []
        # DEAN leaves retain their exact IDs; parents use disjoint negative IDs.
        while True:
            ids = [g[0] for g in groups] if not mappings else list(range(self.next_parent, self.next_parent-len(groups), -1))
            if mappings:
                self.next_parent -= len(groups)
            mapping = {leaf: group_id for group_id, members in zip(ids, groups) for leaf in members}
            prototypes = torch.stack([F.normalize(torch.stack([self.prototypes[k] for k in g]).mean(0), dim=0,
                                                 eps=self.config.epsilon) for g in groups])
            distances = torch.cdist(prototypes, prototypes)
            if len(groups) > 1:
                distances.fill_diagonal_(float("inf"))
                radius = distances.min(dim=1).values / 2
            else:
                radius = torch.zeros(1)
            mappings.append(mapping)
            centers.append({key: value.clone() for key, value in zip(ids, prototypes)})
            radii.append({key: float(value) for key, value in zip(ids, radius)})
            coarser = []
            for known in (True, False):
                side = [g for g in groups if (g[0] < self.base_classes) == known]
                target = max(len(side)//2, 1) if side else 0
                coarser.extend(self._coarsen(side, target))
            coarser.sort()
            if coarser == groups:
                break
            groups = coarser
        return HierarchySnapshot(leaves, tuple(mappings), tuple(centers), tuple(radii))

    def _coarsen(self, groups, target):
        if len(groups) <= target:
            return groups
        active = {g: torch.stack([self.prototypes[k] for k in g]).sum(0) for g in groups}
        ordered = sorted(active)
        means = F.normalize(torch.stack([active[g] for g in ordered]), dim=-1, eps=self.config.epsilon)
        distances = torch.cdist(means, means)
        heap = [(float(distances[i, j]), ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i+1, len(ordered))]
        heapq.heapify(heap)
        while len(active) > target:
            _, a, b = heapq.heappop(heap)
            if a not in active or b not in active:
                continue
            merged = tuple(sorted(a+b))
            summed = active.pop(a) + active.pop(b)
            center = F.normalize(summed, dim=0, eps=self.config.epsilon)
            for other in sorted(active):
                other_center = F.normalize(active[other], dim=0, eps=self.config.epsilon)
                lo, hi = sorted((merged, other))
                heapq.heappush(heap, (float(torch.linalg.vector_norm(center-other_center)), lo, hi))
            active[merged] = summed
        return sorted(active)
