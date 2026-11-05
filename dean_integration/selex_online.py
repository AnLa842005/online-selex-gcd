"""Synthetic SelEx interfaces, not a reproduction of paper USE/SSE objectives.

Hierarchy uses only current classifier prototypes; targets are DEAN pseudo IDs.
Replace these fixtures with audited paper objectives before scientific runs.
"""
from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SelExConfig:
    use_weight: float = 0.1
    sse_weight: float = 0.1

    def __post_init__(self):
        if any(not isinstance(w, Real) or isinstance(w, bool) or not math.isfinite(w) or w < 0
               for w in (self.use_weight, self.sse_weight)):
            raise ValueError("USE/SSE weights must be finite and nonnegative")


def build_hierarchy(prototypes):
    """Deterministic prototype-neighbor parents, rebuilt after head expansion."""
    if prototypes.ndim != 2 or len(prototypes) == 0 or not torch.isfinite(prototypes).all():
        raise ValueError("Hierarchy requires finite nonempty class prototypes")
    p = F.normalize(prototypes.detach(), dim=1)
    similarity = p @ p.T
    similarity.fill_diagonal_(-torch.inf)
    return similarity.argmax(1) if len(p) > 1 else torch.zeros(1, dtype=torch.long, device=p.device)


def use_loss(logits, pseudo_targets, parents):
    """Fixture: soften each pseudo target equally toward its hierarchy parent."""
    return 0.5 * (F.cross_entropy(logits, pseudo_targets) + F.cross_entropy(logits, parents[pseudo_targets]))


def sse_loss(features, pseudo_targets, prototypes, parents):
    """Fixture: align features with detached parent prototype directions."""
    targets = prototypes.detach()[parents[pseudo_targets]]
    return (1 - F.cosine_similarity(features, targets, dim=1)).mean()


def augment_online_loss(loss, features, logits, pseudo_targets, prototypes, *, enabled=False, config=None):
    if not enabled:
        return loss  # Exact object identity; no hierarchy, RNG, or arithmetic.
    config = config or SelExConfig()
    parents = build_hierarchy(prototypes)
    return (loss + config.use_weight * use_loss(logits, pseudo_targets, parents)
            + config.sse_weight * sse_loss(features, pseudo_targets, prototypes, parents))
