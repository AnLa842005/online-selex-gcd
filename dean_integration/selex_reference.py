"""Corrected Euclidean SelEx kernels, adapted from c3d3b42 equations in the audit.

No intentionally bug-compatible production path. Two views are view-major for
distance matrices. SSE prefixes retain reference geometry (no prefix renorm).
"""
from collections import Counter
from functools import wraps

import torch
from torch.nn import functional as F


def checked_views(features):
    if features.ndim != 3 or features.shape[1] != 2 or features.shape[2] < 1:
        raise ValueError("Contrastive features must have shape [images, 2, dimensions]")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("Contrastive features must be finite floating tensors")


def view_major(features):
    # A view when inputs originate from [2*B,D], including coordinate prefixes.
    return features.transpose(0, 1).reshape(-1, features.shape[-1])


def accumulation_tensor(features):
    return features.float() if features.dtype in (torch.float16, torch.bfloat16) else features


def full_precision(function):
    @wraps(function)
    def wrapped(features, *args, **kwargs):
        # cdist has no CPU half/bfloat16 kernel; FP32 also protects epsilon,
        # logsumexp and normalization from mixed-precision under/overflow.
        with torch.autocast(device_type=features.device.type, enabled=False):
            return function(accumulation_tensor(features), *args, **kwargs)
    return wrapped


def zero(features):
    # Avoid overflow in a full reduction before multiplying by zero, and avoid
    # allocating a projection-sized temporary solely for a skipped term.
    scalar = features[(0,)*features.ndim] if features.numel() else features.sum()
    return accumulation_tensor(scalar) * 0.0


def label_support(labels, features):
    if labels.ndim != 1 or labels.dtype != torch.long or len(labels) != len(features) or (labels < 0).any() or labels.device != features.device:
        # Negative parent IDs are valid: callers remap each level to compact labels.
        raise ValueError("Contrastive labels must be aligned nonnegative integer IDs")
    _, inverse, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
    return counts[inverse], counts


@full_precision
def supervised_contrastive(features, labels, config, counters=None, prefix="sse"):
    checked_views(features)
    counters = counters if counters is not None else Counter()
    support, counts = label_support(labels, features)
    if not len(features):
        counters[prefix + ".empty_groups"] += 1
        return zero(features)
    counters[prefix + ".singleton_groups"] += int((counts == 1).sum())
    valid = (support >= 2).repeat(2)
    if not valid.any():
        counters[prefix + ".no_positive_anchors"] += 2*len(features)
        return zero(features)
    flat = view_major(features)
    repeated = labels.repeat(2)
    diagonal = torch.eye(len(flat), dtype=torch.bool, device=flat.device)
    positive = repeated[:, None].eq(repeated[None, :]) & ~diagonal
    logits = -torch.cdist(flat, flat) / config.sse_temperature
    log_prob = logits - torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    denominator = positive.sum(1).clamp_min(config.epsilon)
    per_anchor = -(positive * log_prob).sum(1) / denominator
    counters[prefix + ".no_positive_anchors"] += int((~valid).sum())
    return (config.sse_temperature/config.base_temperature) * per_anchor[valid].mean()


@torch.no_grad()
@full_precision
def confusion_targets(geometry, leaves, snapshot, config, counters):
    checked_views(geometry)
    if geometry.shape[-1] != 768:
        raise ValueError("SelEx hierarchy geometry requires 768-D features")
    normalized = F.normalize(view_major(geometry.detach()), dim=-1, eps=config.epsilon)
    distances = torch.cdist(normalized, normalized)
    count = len(geometry)
    confusion = torch.zeros_like(distances)
    assignments = snapshot.assignments(leaves, device=geometry.device)
    eligible = torch.zeros(2*count, dtype=torch.bool, device=geometry.device)
    for level, labels in enumerate(assignments):
        _, inverse, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
        supported = (counts[inverse] >= 2).repeat(2)
        counters[f"use.level{level}.singleton_groups"] += int((counts == 1).sum())
        absent = set(snapshot.centers[level])-set(labels.tolist())
        counters[f"use.level{level}.empty_groups"] += len(absent)
        eligible |= supported
        repeated = labels.repeat(2)
        radii = torch.tensor([snapshot.radii[level][int(k)] for k in repeated.tolist()],
                             device=geometry.device, dtype=geometry.dtype)
        weights = (distances > 2*radii[None, :]).to(geometry.dtype)
        weights *= supported[:, None] & supported[None, :]
        confusion += config.hierarchy_decay**level * weights
    if not assignments:
        counters["use.empty_hierarchy"] += 1
    diagonal = torch.eye(2*count, dtype=torch.bool, device=geometry.device)
    if count:
        confusion = (confusion-confusion.min()) / (confusion.max()-confusion.min()).clamp_min(config.epsilon)
    confusion.masked_fill_(diagonal, 0)
    mass = confusion.sum(1, keepdim=True)
    valid = eligible & (mass.squeeze(1) > config.epsilon)
    counters["use.zero_target_rows"] += int((eligible & ~valid).sum())
    counters["use.no_positive_anchors"] += int((~eligible).sum())
    return confusion / mass.clamp_min(config.epsilon), valid


@full_precision
def unsupervised_expertise(projected, geometry, leaves, snapshot, config, counters=None):
    checked_views(projected)
    checked_views(geometry)
    if len(projected) != len(geometry) or len(leaves) != len(projected) or geometry.shape[-1] != 768 or projected.device != geometry.device:
        raise ValueError("USE projected/geometry/leaf arrays must align")
    if leaves.ndim != 1 or leaves.dtype != torch.long or (leaves < 0).any():
        raise ValueError("USE requires aligned nonnegative DEAN leaf IDs")
    snapshot.assignments(leaves, geometry.device)
    counters = counters if counters is not None else Counter()
    if len(projected) < 2:
        counters["use.empty_groups" if not len(projected) else "use.singleton_groups"] += 1
        return zero(projected)
    targets, valid = confusion_targets(geometry, leaves, snapshot, config, counters)
    if not valid.any():
        return zero(projected)
    flat = F.normalize(view_major(projected), dim=-1, eps=config.epsilon)
    diagonal = torch.eye(len(flat), dtype=torch.bool, device=flat.device)
    logits = -torch.cdist(flat, flat) / config.use_temperature
    log_prob = logits - torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    # Positive is the other augmented view, independent of pseudo class IDs.
    positive = torch.zeros_like(targets)
    rows = torch.arange(len(flat), device=flat.device)
    positive[rows, (rows+len(projected)) % len(flat)] = 1
    soft = (1-config.smoothing)*positive + config.smoothing*targets
    return -(soft * log_prob).sum(1)[valid].mean()


def expertise_losses(projected, geometry, leaves, snapshot, config, *, replay_projected=None,
                     replay_labels=None, counters=None):
    counters = counters if counters is not None else Counter()
    checked_views(projected)
    use = unsupervised_expertise(projected, geometry, leaves, snapshot, config, counters)
    hierarchy = zero(projected)
    if not snapshot.mappings:
        counters["sse.empty_hierarchy"] += 1
    for level, labels in enumerate(snapshot.assignments(leaves, projected.device)):
        # Coordinate resolution is fixed by the reference, independently of
        # configurable per-level loss weights. Never create zero-width slices.
        dim = projected.shape[-1] // 2**(level+1)
        name = f"sse.level{level}"
        counters[name + ".empty_groups"] += len(set(snapshot.centers[level])-set(labels.tolist()))
        if dim < 1:
            counters[name + ".empty_dimensions"] += 1
            continue
        _, compact = torch.unique(labels, sorted=True, return_inverse=True)
        hierarchy = hierarchy + config.hierarchy_decay**(level+1) * supervised_contrastive(
            projected[:, :, :dim], compact, config, counters, name)
    if replay_projected is None:
        if replay_labels is not None:
            raise ValueError("Replay labels require explicit labelled replay views")
        counters["sse.gt.empty_groups"] += 1
        gt = zero(projected)
    else:
        if replay_labels is None:
            raise ValueError("Replay GT term requires true labels")
        checked_views(replay_projected)
        if replay_projected.shape[-1] != projected.shape[-1] or replay_projected.device != projected.device:
            raise ValueError("Replay projection space must match current projection space")
        gt = supervised_contrastive(replay_projected, replay_labels, config, counters, "sse.gt")
    sse = 0.5*(gt+hierarchy)
    total = (1-config.sse_weight)*use + config.sse_weight*sse
    return dict(use=use, sse=sse, sse_gt=gt, sse_hierarchy=hierarchy, selex_total=total)
