"""Reviewed v1 configuration for SelEx adapted to DEAN online discovery."""
from dataclasses import dataclass, asdict
import math
from numbers import Real
from pathlib import Path
import hashlib
import logging

import yaml

SOURCE_SHA = "c3d3b42a0aa7d7c9ea748a8305a6890a34100f73"


@dataclass(frozen=True)
class ReferenceConfig:
    gamma: float = 0.1
    output_dim: int = 65536
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    sse_weight: float = 0.35
    smoothing: float = 1.0
    use_temperature: float = 1.0
    sse_temperature: float = 0.07
    base_temperature: float = 0.07
    hierarchy_decay: float = 0.5
    prototype_momentum: float = 0.9
    prototype_capacity: int = 512
    projector_lr: float = 1e-4
    epsilon: float = 1e-8
    # Append metadata to preserve the historical numerical argument order.
    # Programmatic ReferenceConfig fixtures explicitly denote the legacy ablation;
    # the production research loader defaults to the separate HSSK YAML instead.
    hierarchy_source: str = "dean_prototypes"
    replay_capacity: int = 320
    replay_per_class: int = 2
    replay_selection_size: int = 32
    insufficient_support_policy: str = "reuse_then_skip"

    def __post_init__(self):
        if self.hierarchy_source not in ("hssk_online", "dean_prototypes"):
            raise ValueError("Unknown hierarchy_source")
        if self.hierarchy_source == "hssk_online" and (type(self.output_dim) is not int or self.output_dim < 2):
            raise ValueError("HSSK needs positive-width hierarchy prefixes (output_dim >= 2)")
        if self.insufficient_support_policy not in ("reuse_then_skip", "reuse_then_error"):
            raise ValueError("Unknown insufficient_support_policy")
        for name, expected in (("replay_capacity", 320), ("replay_per_class", 2), ("replay_selection_size", 32)):
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f"{name} is locked to {expected} by the design contract")
        for name, value in asdict(self).items():
            if name in ("hierarchy_source", "insufficient_support_policy", "replay_capacity", "replay_per_class", "replay_selection_size"):
                continue
            if name in ("output_dim", "hidden_dim", "bottleneck_dim", "prototype_capacity"):
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            elif not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
            elif abs(value) > (2-2**-23)*2**127:
                raise ValueError(f"{name} must be representable in FP32")
        for name in ("sse_weight", "smoothing", "prototype_momentum"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("use_temperature", "sse_temperature", "base_temperature", "projector_lr", "epsilon"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("use_temperature", "sse_temperature", "base_temperature"):
            if getattr(self, name) < 2**-126:
                raise ValueError(f"{name} must be a normal FP32 value")
        if self.gamma < 0 or not 0 < self.hierarchy_decay < 1:
            raise ValueError("gamma must be nonnegative; hierarchy_decay must be in (0,1)")
        if not 2**-126 <= self.epsilon <= 1:
            raise ValueError("epsilon must be representable as a normal FP32 value in (0,1]")


def load_reference_config(path=None):
    path = path or Path(__file__).resolve().parents[1] / "configs/dean/selex_online_reference_v1.yaml"
    raw = Path(path).read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    expected = set(asdict(ReferenceConfig()))
    # Only this immutable historical blob may omit provider metadata. Hash Git
    # normalized newlines to support both LF and Windows CRLF checkouts.
    legacy_sha = "23c2bc5e0b19acd2468659adb76da1f145c9cffd29063cd8288f85e9ee6ac8d5"
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == legacy_sha:
        logging.getLogger(__name__).warning("Historical config adapter: dean_prototypes ablation (not HSSK)")
        value = {**asdict(ReferenceConfig()), **value}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Reference v1 config fields must exactly match ReferenceConfig")
    return ReferenceConfig(**value)
