"""Separate experimental configuration; the locked official contract stays frozen."""
from pathlib import Path

import yaml

from .selex_online import SelExConfig


def load_selex_config(path=None, implementation="synthetic"):
    if implementation == "reference_v1":
        from .selex_config import load_reference_config
        return load_reference_config(path or Path(__file__).resolve().parents[1] / "configs/dean/selex_online_hssk_v1.yaml")
    if implementation != "synthetic":
        raise ValueError("Unknown SelEx implementation")
    if path is None:
        return SelExConfig()
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"use_weight", "sse_weight"}:
        raise ValueError("SelEx config requires only use_weight and sse_weight")
    return SelExConfig(**value)
