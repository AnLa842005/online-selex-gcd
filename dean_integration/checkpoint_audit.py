"""Read-only matching against the exact external DEAN model, without key rewriting."""
from collections import Counter
from collections.abc import Mapping
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from .external import verify_checkout
from .preflight import sha256


def build_reference_model(checkout):
    """Construct exactly the model used at the upstream teacher-load call.

    Embedding/head replacements happen AFTER that call upstream, so they must
    not be added to this reference. No pretrained=True factory is invoked.
    Use a dedicated CLI process to keep upstream's generic `net` package isolated.
    """
    checkout = verify_checkout(checkout)
    package = sys.modules.get("net")
    if package is not None:
        paths = [Path(p).resolve() for p in getattr(package, "__path__", [])]
        if checkout / "net" not in paths:
            raise ValueError("Conflicting net package; run checkpoint audit in a fresh process")
    sys.path.insert(0, str(checkout))
    try:
        module = importlib.import_module("net.vision_transformer")
        if Path(module.__file__).resolve() != checkout / "net/vision_transformer.py":
            raise ValueError("Reference model did not come from the verified external checkout")
        return module.ViT_Dino(SimpleNamespace(sz_embedding=768))
    finally:
        sys.path.pop(0)


def prefix_counts(keys):
    return dict(sorted(Counter(str(key).split(".")[0] for key in keys).items()))


def compare_teacher(checkpoint, model):
    """Report exact strict=False key/shape eligibility; never call load_state_dict.

    Conservative gate: all backbone tensors must match, no shape mismatch or
    nonfinite matching tensor. Unexpected projection-head keys are disclosed and
    permitted, matching official strict=False. Prefix alternatives are diagnostic
    counts only, never used for pass/fail or loading.
    """
    target = model.state_dict()
    parameters = dict(model.named_parameters())
    teacher = checkpoint.get("teacher") if isinstance(checkpoint, Mapping) else None
    valid_teacher = isinstance(teacher, Mapping) and bool(teacher)
    source = teacher if valid_teacher else {}
    source_keys = set(source)
    target_keys = set(target)
    matched, mismatches, nonfinite, non_tensor = [], [], [], []
    for key in sorted(source_keys & target_keys):
        value = source[key]
        if not torch.is_tensor(value):
            non_tensor.append(key)
        elif value.shape != target[key].shape:
            mismatches.append({"key": key, "checkpoint_shape": list(value.shape),
                               "model_shape": list(target[key].shape)})
        else:
            matched.append(key)
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                nonfinite.append(key)
    groups = {"patch_embed": "backbone.patch_embed.", "cls_token": "backbone.cls_token",
              "pos_embed": "backbone.pos_embed", "final_norm": "backbone.norm."}
    groups.update({f"block_{i}": f"backbone.blocks.{i}." for i in range(12)})
    critical = {}
    for name, prefix in groups.items():
        expected = sorted(k for k in target if k == prefix or k.startswith(prefix))
        critical[name] = {"expected": len(expected), "matched": sum(k in matched for k in expected),
                          "missing_or_incompatible": [k for k in expected if k not in matched],
                          "pass": bool(expected) and all(k in matched for k in expected)}
    loaded_parameters = [k for k in matched if k in parameters]
    total_numel = sum(p.numel() for p in parameters.values())
    loaded_numel = sum(parameters[k].numel() for k in loaded_parameters)
    reasons = []
    if not valid_teacher:
        reasons.append("Missing nonempty checkpoint['teacher'] mapping; bare backbone is not accepted")
    if len(loaded_parameters) != len(parameters):
        reasons.append("Not all reference backbone parameter tensors match exact original keys/shapes")
    if not all(group["pass"] for group in critical.values()):
        reasons.append("One or more mandatory backbone groups are missing or incompatible")
    if mismatches or non_tensor or nonfinite:
        reasons.append("Shape mismatch, non-tensor entry or nonfinite matched tensor")
    prefix_diagnostics = {}
    for prefix in ("teacher.", "module.", "backbone.", "module.backbone.", "teacher.backbone."):
        prefix_diagnostics[prefix] = sum(isinstance(k, str) and k.startswith(prefix) for k in source)
    # A prefix hint is evidence for a new audit, never an implicit conversion.
    alternatives = {}
    for strip in ("module.", "teacher.", "module.teacher."):
        alternatives[f"strip_{strip}"] = sum(
            isinstance(k, str) and k.startswith(strip) and k[len(strip):] in target for k in source)
    alternatives["add_backbone_to_bare_keys"] = sum(
        isinstance(k, str) and "backbone." + k in target for k in source)
    return {"status": "FAIL" if reasons else "PASS", "verified_scope": "exact key/shape/finite compatibility only",
            "checkpoint_top_level_key_count": len(checkpoint) if isinstance(checkpoint, Mapping) else None,
            "teacher_key_count": len(source), "teacher_mapping_valid": valid_teacher,
            "source_prefix_counts": prefix_counts(source), "prefix_diagnostics": prefix_diagnostics,
            "prefix_alternatives_not_applied": alternatives,
            "matched_key_count": len(matched), "matched_keys": matched,
            "missing_keys": sorted(target_keys - source_keys),
            "unexpected_keys": sorted(str(k) for k in source_keys - target_keys),
            "shape_mismatches": mismatches, "non_tensor_keys": non_tensor, "nonfinite_keys": nonfinite,
            "parameter_tensors": {"matched": len(loaded_parameters), "total": len(parameters),
                                  "ratio": len(loaded_parameters) / len(parameters) if parameters else 0.},
            "parameter_numel": {"matched": loaded_numel, "total": total_numel,
                                "ratio": loaded_numel / total_numel if total_numel else 0.},
            "critical_groups": critical, "key_transformations_applied": [],
            "actual_weights_loaded": False, "official_strict_false_unchanged": True,
            "errors": reasons}


def audit_checkpoint(path, checkout):
    path = Path(path).resolve()
    digest = sha256(path)
    # Explicit trusted local input only. Do not wrap a bare checkpoint as teacher.
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = build_reference_model(checkout)
    result = compare_teacher(state, model)
    result.update(path=str(path), sha256=digest, size_bytes=path.stat().st_size,
                  checkpoint_top_level_prefix_counts=prefix_counts(state) if isinstance(state, Mapping) else {})
    return result
