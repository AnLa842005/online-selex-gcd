"""Observe unmodified official CPU loading; prefix alternatives never reach the loader."""
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import torch

from .checkpoint_audit import build_reference_model, compare_teacher
from .config import UPSTREAM_SHA, UPSTREAM_URL
from .external import verify_checkout
from .preflight import sha256

EXPECTED_FULL_CHECKPOINT_BYTES = 871077689


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def hypothetical_matching(teacher, target, parameters, strip_prefix):
    """Compare names/shapes only; alias tensor references without mutation or loading."""
    transformed, collisions = {}, []
    for key, value in teacher.items():
        renamed = key[len(strip_prefix):] if isinstance(key, str) and key.startswith(strip_prefix) else key
        if renamed in transformed:
            collisions.append(str(renamed))
        transformed[renamed] = value
    exact = set(target) & set(transformed)
    matched = [key for key in sorted(exact) if torch.is_tensor(transformed[key])
               and transformed[key].shape == target[key].shape]
    mismatches = [{"key": key, "source_shape": list(transformed[key].shape),
                   "target_shape": list(target[key].shape)} for key in sorted(exact)
                  if torch.is_tensor(transformed[key]) and transformed[key].shape != target[key].shape]
    total_numel = sum(p.numel() for p in parameters.values())
    matched_numel = sum(parameters[k].numel() for k in matched if k in parameters)
    return {"strip_prefix": strip_prefix, "applied_to_official_load": False,
            "scope": "hypothetical names/shapes only; not a loader-fix validation",
            "exact_key_matches": len(exact), "shape_compatible_matches": len(matched),
            "target_tensor_count": len(target), "matched_tensor_ratio": len(matched) / len(target),
            "matched_parameter_numel": matched_numel, "total_parameter_numel": total_numel,
            "matched_parameter_numel_ratio": matched_numel / total_numel,
            "missing_keys": sorted(set(target) - set(transformed)),
            "unexpected_keys": sorted(str(k) for k in set(transformed) - set(target)),
            "shape_mismatches": mismatches, "key_collisions": collisions,
            "all_target_tensors_shape_compatible": len(matched) == len(target) and not collisions}


def observe_official_load(checkpoint, model):
    """Mutate ONLY the disposable reference model by the exact official operation."""
    raw = compare_teacher(checkpoint, model)
    teacher = checkpoint.get("teacher") if isinstance(checkpoint, Mapping) else None
    valid = isinstance(teacher, Mapping) and bool(teacher)
    target = model.state_dict()
    parameter_names = set(dict(model.named_parameters()))
    raw["raw_exact_key_matches"] = len(set(target) & set(teacher)) if valid else 0
    raw["raw_shape_compatible_matches"] = raw["matched_key_count"]
    raw["target_state_tensor_count"] = len(target)
    raw["matched_tensor_ratio"] = raw["matched_key_count"] / len(target)
    raw["matched_parameter_numel_ratio"] = raw["parameter_numel"]["ratio"]
    before = {key: tensor_digest(value) for key, value in target.items()}
    result, error = None, None
    if valid:
        try:
            with torch.no_grad():
                result = model.load_state_dict(teacher, strict=False)
        except (RuntimeError, ValueError, TypeError) as exc:
            # PyTorch can copy some tensors before raising on a shape mismatch.
            error = f"{type(exc).__name__}: {exc}"
    changed = [key for key, value in model.state_dict().items() if before[key] != tensor_digest(value)]
    changed_parameters = [key for key in changed if key in parameter_names and key.startswith("backbone.")]
    raw["observed_official_load"] = {
        "operation": 'model.load_state_dict(checkpoint["teacher"], strict=False)',
        "attempted": valid, "strict": False, "key_rewrites": [], "error": error,
        "returned_missing_keys": list(result.missing_keys) if result is not None else None,
        "returned_unexpected_keys": list(result.unexpected_keys) if result is not None else None,
        "changed_state_tensor_count": len(changed),
        "changed_backbone_parameter_tensor_count": len(changed_parameters),
        "changed_backbone_parameter_keys": changed_parameters}
    raw["actual_weights_loaded"] = bool(changed_parameters)
    hypothetical = {}
    if valid and raw["status"] != "PASS":
        hypothetical["A_strip_module_against_official_target"] = hypothetical_matching(
            teacher, target, dict(model.named_parameters()), "module.")
        hypothetical["B_strip_module_backbone_against_bare_backbone"] = hypothetical_matching(
            teacher, model.backbone.state_dict(), dict(model.backbone.named_parameters()), "module.backbone.")
        for prefix in ("teacher.", "module.teacher.", "teacher.module."):
            if any(isinstance(key, str) and key.startswith(prefix) for key in teacher):
                hypothetical[f"strip_{prefix}_against_official_target"] = hypothetical_matching(
                    teacher, target, dict(model.named_parameters()), prefix)
    blockers = list(raw["errors"])
    if error:
        blockers.append("Actual official load raised: " + error)
    if not changed_parameters:
        blockers.append("Official load did not change any backbone parameter tensor")
    if not blockers:
        decision = "PASS"
    elif raw["status"] != "PASS" and any(
            item["all_target_tensors_shape_compatible"] for item in hypothetical.values()):
        decision = "BLOCKED_UPSTREAM_CHECKPOINT_PREFIX_MISMATCH"
    elif raw["status"] == "PASS" and not changed_parameters:
        decision = "BLOCKED_OFFICIAL_LOAD_DID_NOT_CHANGE_BACKBONE_WEIGHTS"
    else:
        decision = "BLOCKED_UPSTREAM_CHECKPOINT_COMPATIBILITY"
    return {"raw_official_matching": raw, "hypothetical_prefix_matching": hypothetical,
            "critical_groups": raw["critical_groups"],
            "weights_changed_after_official_load": bool(changed_parameters),
            "decision": decision, "blockers": blockers,
            "registered_gate": {"matched_parameter_tensor_ratio": 1.0,
                                "matched_parameter_numel_ratio": 1.0,
                                "all_critical_groups_required": True,
                                "shape_mismatches_allowed": 0, "backbone_weights_must_change": True}}


def audit_full_checkpoint(path, checkout):
    checkout = verify_checkout(checkout)
    path = Path(path).resolve()
    size = path.stat().st_size
    if size != EXPECTED_FULL_CHECKPOINT_BYTES:
        raise ValueError(f"Expected {EXPECTED_FULL_CHECKPOINT_BYTES} bytes, found {size}")
    before = sha256(path)
    # Caller explicitly authorizes this trusted local full checkpoint (may contain argparse metadata).
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = build_reference_model(checkout)
    result = observe_official_load(checkpoint, model)
    teacher = checkpoint.get("teacher", {}) if isinstance(checkpoint, Mapping) else {}
    teacher = teacher if isinstance(teacher, Mapping) else {}
    prefixes = ("module.backbone.", "backbone.", "module.", "teacher.", "head.", "module.head.")
    result.update(schema="dean-full-checkpoint-official-load-v1",
                  created_at_utc=datetime.now(timezone.utc).isoformat(), path=str(path), size=size,
                  sha256=before, top_level_keys=[str(k) for k in checkpoint] if isinstance(checkpoint, Mapping) else [],
                  teacher_mapping_nonempty=bool(teacher), teacher_key_count=len(teacher),
                  prefix_distribution={prefix: sum(isinstance(k, str) and k.startswith(prefix) for k in teacher)
                                       for prefix in prefixes}, sample_teacher_keys=[str(k) for k in list(teacher)[:20]],
                  upstream={"url": UPSTREAM_URL, "sha": UPSTREAM_SHA, "path": str(checkout),
                            "origin_sha_clean_locked_hashes": "PASS"},
                  reference_model={"class": type(model).__module__ + "." + type(model).__name__,
                                   "load_point": "before official embedding/head replacement",
                                   "embedding_present_at_load": hasattr(model.backbone, "embedding"),
                                   "block_count": len(model.backbone.blocks),
                                   "patch_embed_weight_shape": list(getattr(model.backbone.patch_embed, "proj",
                                                                            model.backbone.patch_embed).weight.shape)},
                  initializer_seed=0, load_device="cpu", trusted_local=True, weights_only=False,
                  gpu_smoke="NOT RUN", optimizer_steps=0, full_training="NOT RUN")
    result["checkpoint_sha256_after_audit"] = sha256(path)
    result["checkpoint_unchanged"] = result["checkpoint_sha256_after_audit"] == before
    result["upstream"]["clean_after_audit"] = bool(verify_checkout(checkout))
    if not result["checkpoint_unchanged"]:
        result["decision"] = "BLOCKED_CHECKPOINT_CHANGED_DURING_AUDIT"
        result["blockers"].append("Checkpoint hash changed during audit")
    return result
