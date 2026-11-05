"""Strict base checkpoint loading and exact base-boundary RNG sidecars."""
import json
from pathlib import Path
import random

import numpy as np
import torch

from .preflight import sha256


def validate_checkpoint(checkpoint, model, header):
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model_state_dict", "header_state_dict"}:
        raise ValueError("Base checkpoint requires exactly model_state_dict and header_state_dict")
    for name, module in (("model_state_dict", model), ("header_state_dict", header)):
        state, expected = checkpoint[name], module.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError(f"Incompatible {name}: missing/unexpected keys")
        for key, reference in expected.items():
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f"Incompatible {name}.{key}: tensor shape/dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite tensor: {name}.{key}")
    return checkpoint


def load_base(path, model, header):
    # Tensor-only deserialization; never retry an unsafe or non-strict load.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return validate_checkpoint(checkpoint, model, header)


def audit_base(path, checkout):
    """Validate against the exact post-initialization architecture on CPU."""
    from .checkpoint_audit import build_reference_model
    model = build_reference_model(checkout)
    model.backbone.embedding = torch.nn.Linear(model.backbone.embed_dim, 768)
    model.head = torch.nn.Identity()
    header = torch.nn.Module()
    header.head = torch.nn.Linear(768, 160)
    load_base(path, model, header)
    return {"status": "PASS", "path": str(Path(path).resolve()), "sha256": sha256(path),
            "validation": "exact keys, shapes, dtypes, finite tensors; no key rewriting"}


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state().tolist(),
            "cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value, list) else value


def restore_rng(state):
    try:
        py = tuples(state["python"])
        ns = state["numpy"]
        np_state = (ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4])
        cpu = torch.tensor(state["torch"], dtype=torch.uint8)
        cuda = [torch.tensor(s, dtype=torch.uint8) for s in state["cuda"]]
        # Validate CPU states with private generators before touching global state.
        random.Random().setstate(py)
        np.random.RandomState().set_state(np_state)
        torch.Generator().set_state(cpu)
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid base-boundary RNG state: {exc}") from exc
    if len(cuda) != torch.cuda.device_count():
        raise ValueError("Boundary RNG CUDA device count mismatch")
    random.setstate(py)
    np.random.set_state(np_state)
    torch.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def save_boundary(checkpoint_path, sidecar, context):
    record = dict(version=1, checkpoint_sha256=sha256(checkpoint_path), context=context, rng=capture_rng())
    with Path(sidecar).open("x", encoding="utf-8") as handle:
        json.dump(record, handle)


def read_boundary(checkpoint_path, sidecar, context):
    try:
        record = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Exact online-only requires a base-boundary RNG sidecar; base checkpoints alone contain no RNG state") from exc
    if not isinstance(record, dict) or record.get("version") != 1 or record.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("Boundary sidecar checkpoint hash/version mismatch")
    if record.get("context") != context:
        raise ValueError("Boundary sidecar seed/protocol/assets mismatch")
    rng = record.get("rng")
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Invalid base-boundary RNG sidecar schema")
    return rng
