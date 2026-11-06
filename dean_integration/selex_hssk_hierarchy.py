"""Detached weighted coarsening of CHILD CENTERS only, never raw stream rows."""
import math
import torch
from .selex_hierarchy_provider import digest
from .selex_hssk_online import HSSKLeafFitter, squared_distances


@torch.no_grad()
def weighted_fit(child_centers, weights, fixed, k, seed, min_free_children=2):
    """Two-phase BSSK on child rows; balance child counts, weight centroid means."""
    a = int(fixed.max())+1 if (fixed >= 0).any() else 0
    anchors = torch.stack([(child_centers[fixed == j]*weights[fixed == j, None]).sum(0)/weights[fixed == j].sum()
                           for j in range(a)]) if a else torch.empty(0, 768)
    helper = HSSKLeafFitter()
    generator = torch.Generator().manual_seed(seed)
    centers, _ = helper._initialize(child_centers[fixed < 0], torch.where(fixed < 0)[0], anchors,
                                    k, torch.empty(0, 768), (), generator)
    if centers is None:
        return None
    for balanced in (True, False):
        for _ in range(10):
            previous = centers.clone()
            labels, distances = helper._assign(child_centers, centers, fixed, balanced)
            for empty in range(a, k):
                if (labels == empty).any():
                    continue
                candidates = [i for i in range(len(labels)) if fixed[i] < 0 and
                              int(((labels == labels[i]) & (fixed < 0)).sum()) >= 2*min_free_children]
                if not candidates:
                    return None
                source = min(candidates, key=lambda i: (-float(distances[i, labels[i]]), i))
                donor = int(labels[source])
                take = sorted((i for i in candidates if labels[i] == donor),
                              key=lambda i: (-float(distances[i, donor]), i))[:min_free_children]
                labels[take] = empty
            for j in range(0 if not balanced else a, k):
                mask = labels == j
                if not mask.any():
                    return None
                centers[j] = (child_centers[mask]*weights[mask, None]).sum(0)/weights[mask].sum()
            if float(torch.linalg.vector_norm(centers-previous, dim=1).sum())**2 < 1e-4:
                break
    for j in range(k):
        mask = labels == j
        if int(mask.sum()) < (min_free_children if j >= a else 1) or float(weights[mask].sum()) < 2:
            return None
    return labels, centers


@torch.no_grad()
def coarsen_children(*, centers, support_counts, anchor_groups, seed, level):
    """Return a local child->parent map/geometry, or None when reduction is invalid.

    Signature intentionally has no current rows, replay rows, model or DEAN state.
    Weights are real image support; each distinct child counts once for balance
    and the free-parent two-child constraint.
    """
    keys = sorted(centers)
    if set(keys) != set(support_counts) or set(keys) != set(anchor_groups):
        raise ValueError("Child geometry/support/anchor identities do not align")
    if any(type(support_counts[k]) is not int or support_counts[k] < 2 for k in keys):
        raise ValueError("Child real-image support must be at least two")
    rows = torch.stack([centers[k].detach().cpu().float() for k in keys])
    if rows.shape != (len(keys), 768) or not torch.isfinite(rows).all():
        raise ValueError("Child centers require finite detached 768-D geometry")
    weights = torch.tensor([support_counts[k] for k in keys], dtype=torch.float32)
    anchored = [i for i, k in enumerate(keys) if anchor_groups[k]]
    free = len(keys)-len(anchored)
    target_a, target_f = math.ceil(len(anchored)/2), math.ceil(free/2)
    k = target_a+target_f
    if k >= len(keys) or (target_f and free < 2*target_f):
        return None
    private = int(digest(dict(domain="hssk-higher-v1", seed=seed, level=level))[:16], 16) % 2**63
    fixed = torch.full((len(keys),), -1, dtype=torch.long)
    if anchored:
        known = weighted_fit(rows[anchored], weights[anchored], torch.full((len(anchored),), -1),
                             target_a, private, min_free_children=1)
        if known is None:
            return None
        fixed[anchored] = known[0]
    fitted = weighted_fit(rows, weights, fixed, k, private)
    if fitted is None:
        return None
    labels, parents = fitted
    counts, radii, groups = {}, {}, {}
    for j in range(k):
        mask = labels == j
        counts[j] = int(weights[mask].sum())
        radii[j] = max(1e-8, float((torch.linalg.vector_norm(rows[mask]-parents[j], dim=1)*weights[mask]).sum()/weights[mask].sum()))
        groups[j] = tuple(sorted({gt for i in torch.where(mask)[0].tolist() for gt in anchor_groups[keys[i]]}))
    return dict(mapping={key: int(labels[i]) for i, key in enumerate(keys)}, centers=dict(enumerate(parents)),
                counts=counts, radii=radii, anchor_groups=groups)
