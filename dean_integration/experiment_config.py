"""Separate experimental configuration; the locked official contract stays frozen."""
from pathlib import Path

import yaml

from .selex_online import SelExConfig


def load_selex_config(path=None):
    if path is None:
        return SelExConfig()
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"use_weight", "sse_weight"}:
        raise ValueError("SelEx config requires only use_weight and sse_weight")
    return SelExConfig(**value)
