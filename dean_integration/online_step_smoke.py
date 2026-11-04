"""Bounded smoke orchestration and exact CPU snapshots; not a DEAN training runner."""
import ast
import copy
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from .external import verify_checkout
from .smoke import execute_nodes

LORA_LAYERS = (7, 8, 9, 10, 11)


def external_head_class(checkout):
    tree = ast.parse((Path(checkout) / "net/Decoder.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Head")
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<external-dean-head>", "exec"), namespace)
    return namespace["Head"]


def build_online_pair(offline, checkout, embedding_dim=768):
    verify_checkout(checkout)
    head = external_head_class(checkout)(embedding_dim, 160)
    online, online_head = copy.deepcopy(offline), copy.deepcopy(head)
    offline.requires_grad_(False)
    head.requires_grad_(False)
    offline.eval()
    head.eval()
    online.requires_grad_(False)
    online.backbone.prepare_LoRA(lora_layers=list(LORA_LAYERS))
    online_head.update(nb_classes_now=162)  # Two fixture pseudo-novel slots, not estimated/true K.
    return offline, head, online, online_head


def parameter_groups(offline, offline_head, online, online_head):
    groups = {"offline_model": dict(offline.named_parameters()),
              "offline_head": dict(offline_head.named_parameters()), "online_frozen": {},
              "online_pet": {}, "online_head": dict(online_head.named_parameters())}
    for name, parameter in online.named_parameters():
        pet = any(name.startswith(f"backbone.blocks.{layer}.attn.kv_lora.") for layer in LORA_LAYERS)
        groups["online_pet" if pet else "online_frozen"][name] = parameter
    return groups


def snapshot_parameters(groups):
    """CPU copies avoid allocating a second full model on GPU; never saved to disk."""
    return {group: {name: {"value": p.detach().cpu().clone(), "requires_grad": p.requires_grad,
                          "parameter_id": id(p)}
                    for name, p in parameters.items()} for group, parameters in groups.items()}


def gradient_report(groups):
    report = {}
    for group, parameters in groups.items():
        present, nonzero, invalid = [], [], []
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                gradient = parameter.grad.detach().cpu()
                present.append(name)
                if not torch.isfinite(gradient).all():
                    invalid.append(name)
                elif torch.count_nonzero(gradient).item():
                    nonzero.append(name)
        report[group] = {"present": present, "nonzero": nonzero, "nonfinite": invalid}
    return report


def validate_setup(groups):
    errors, identities, storage = [], set(), set()
    for group, parameters in groups.items():
        if not parameters:
            errors.append(f"Empty parameter group: {group}")
        expected_trainable = group in ("online_pet", "online_head")
        for name, parameter in parameters.items():
            pointer = (str(parameter.device), parameter.untyped_storage().data_ptr())
            if id(parameter) in identities or pointer in storage:
                errors.append(f"Parameter/storage shared between groups: {group}:{name}")
            identities.add(id(parameter))
            storage.add(pointer)
            if parameter.requires_grad != expected_trainable:
                errors.append(f"Invalid trainability: {group}:{name}")
    return errors


def validate_gradients(gradients):
    errors = []
    for group, record in gradients.items():
        if record["nonfinite"]:
            errors.append(f"Nonfinite gradient: {group}")
        if group in ("online_pet", "online_head"):
            if not record["nonzero"]:
                errors.append(f"No nonzero gradient: {group}")
        elif record["present"]:
            errors.append(f"Frozen group received gradients: {group}")
    return errors


def validate_structure(before, groups):
    errors = []
    if set(before) != set(groups):
        errors.append("Parameter groups changed")
    for group, parameters in groups.items():
        prior = before.get(group, {})
        if set(prior) != set(parameters):
            errors.append(f"Parameter names changed: {group}")
        for name, parameter in parameters.items():
            if name in prior and (id(parameter) != prior[name]["parameter_id"]
                                  or parameter.shape != prior[name]["value"].shape
                                  or parameter.requires_grad != prior[name]["requires_grad"]):
                errors.append(f"Parameter identity/shape/trainability changed: {group}:{name}")
    return errors


def audit_parameter_diff(before, groups, gradients):
    errors = validate_setup(groups) + validate_gradients(gradients) + validate_structure(before, groups)
    report = {}
    for group, parameters in groups.items():
        prior = before.get(group, {})
        changed, nonfinite, trainability_changed = [], [], []
        if set(prior) != set(parameters):
            errors.append(f"Parameter names changed: {group}")
        for name, parameter in parameters.items():
            value = parameter.detach().cpu()
            if not torch.isfinite(value).all():
                nonfinite.append(name)
            if name in prior:
                if not torch.equal(value, prior[name]["value"]):
                    changed.append(name)
                if parameter.requires_grad != prior[name]["requires_grad"]:
                    trainability_changed.append(name)
        gradient_changed = sorted(set(changed) & set(gradients[group]["nonzero"]))
        trainable = [p for p in parameters.values() if p.requires_grad]
        report[group] = {"parameter_tensors": len(parameters), "numel": sum(p.numel() for p in parameters.values()),
                         "trainable_tensors": len(trainable), "trainable_numel": sum(p.numel() for p in trainable),
                         "frozen_numel": sum(p.numel() for p in parameters.values() if not p.requires_grad),
                         "changed": changed, "gradient_nonzero_and_changed": gradient_changed,
                         "nonfinite": nonfinite, "trainability_changed": trainability_changed,
                         "gradients": gradients[group]}
        if nonfinite or trainability_changed:
            errors.append(f"Nonfinite value or trainability change: {group}")
        if group in ("online_pet", "online_head"):
            if not gradient_changed:
                errors.append(f"No gradient-driven update: {group}")
        elif changed:
            errors.append(f"Frozen group changed: {group}")
    return {"status": "FAIL" if errors else "PASS", "errors": errors, "groups": report}


def pseudo_target_fixture(offline_logits):
    """Even rows use offline argmax; odd rows use two synthetic pseudo-novel slots."""
    if offline_logits.ndim != 2 or offline_logits.shape[1] != 160 or offline_logits.shape[0] < 2:
        raise ValueError("Online fixture requires >=2 samples and 160 offline logits")
    targets = offline_logits.detach().argmax(dim=1).clone()
    row = torch.arange(targets.shape[0], device=targets.device)
    targets[1::2] = 160 + (row[1::2] // 2) % 2
    return targets


def run_online_step(models, batch, checkout):
    """Exactly one original loss execution/backward/AdamW step; no epoch loop."""
    verify_checkout(checkout)
    offline, offline_head, online, online_head = models
    groups = parameter_groups(*models)
    report = {"status": "FAIL", "execution_device": str(batch.device), "epochs": 0,
              "optimizer_steps": 0, "optimizer_step_attempts": 0, "backward_calls": 0,
              "forward_calls": {"offline": 0, "online": 0},
              "batches": 1, "full_training": "NOT RUN",
              "pseudo_targets": "offline argmax on even rows; fixture slots 160/161 on odd rows; no GT input",
              "untrained_components": ["post-load embedding", "base head", "new online head slots"],
              "discovery": "NOT RUN; pseudo target fixture replaces discovery only for this smoke",
              "forward_seconds": None, "backward_seconds": None, "step_seconds": None,
              "input_shape": list(batch.shape), "loss_finite": False, "errors": [],
              "parameter_counts": {group: {"tensors": len(parameters),
                    "trainable_tensors": sum(p.requires_grad for p in parameters.values()),
                    "frozen_tensors": sum(not p.requires_grad for p in parameters.values()),
                    "trainable_numel": sum(p.numel() for p in parameters.values() if p.requires_grad),
                    "frozen_numel": sum(p.numel() for p in parameters.values() if not p.requires_grad)}
                  for group, parameters in groups.items()}}
    def sync():
        if batch.is_cuda:
            torch.cuda.synchronize(batch.device)
    try:
        errors = validate_setup(groups)
        if errors:
            raise ValueError("; ".join(errors))
        before = snapshot_parameters(groups)
        if any(not torch.isfinite(item["value"]).all() for group in before.values() for item in group.values()):
            raise ValueError("Nonfinite parameter before forward")
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            report["forward_calls"]["offline"] += 1
            offline_features = offline(batch)
            offline_logits = offline_head(offline_features)
        sync()
        report["offline_forward_seconds"] = time.perf_counter() - start
        if not torch.isfinite(offline_features).all() or not torch.isfinite(offline_logits).all():
            raise ValueError("Nonfinite offline features/logits")
        targets = pseudo_target_fixture(offline_logits)
        report.update(offline_feature_shape=list(offline_features.shape), offline_logit_shape=list(offline_logits.shape),
                      target_shape=list(targets.shape), target_range=[int(targets.min()), int(targets.max())])
        online.train()
        online_head.train()
        # Same initial online optimizer groups/settings as upstream (including frozen params).
        embedding = list(online.backbone.embedding.parameters())
        optimizer = torch.optim.AdamW([
            {"params": list(set(online.parameters()).difference(set(embedding)))},
            {"params": embedding, "lr": 1e-4}, {"params": online_head.parameters(), "lr": 1e-4}],
            lr=1e-4, weight_decay=1e-4, betas=(.9, .999))
        optimizer.zero_grad(set_to_none=True)
        tree = ast.parse((Path(checkout) / "Ours_train_online.py").read_text(encoding="utf-8"))
        context = {"torch": torch, "F": torch.nn.functional,
                   "args": SimpleNamespace(use_EC=True, energy_hp=1.), "nb_classes": 160,
                   "model_now": online, "header_now": online_head, "on_x": batch, "on_y": targets}
        sync()
        start = time.perf_counter()
        # Includes the original pseudo masks, [0] indexing and CE + energy loss; no fixes/clamps.
        def count_online_forward(module, inputs):
            report["forward_calls"]["online"] += 1
        handle = online.register_forward_pre_hook(count_online_forward)
        try:
            execute_nodes(tree, 731, 767, context)
        finally:
            handle.remove()
        sync()
        report["forward_seconds"] = time.perf_counter() - start
        features, logits, loss = context["feats"], context["logits"], context["loss"]
        if not all(torch.isfinite(value).all() for value in (features, logits, loss)):
            raise ValueError("Nonfinite online features/logits/loss; no optimizer step")
        report.update(output_shape=list(features.shape), logit_shape=list(logits.shape), loss=float(loss.detach()),
                      ce_loss=float(context["ce_loss"].detach()), energy_loss=float(context["energy_loss"].detach()),
                      loss_finite=True)
        start = time.perf_counter()
        loss.backward()
        report["backward_calls"] = 1
        sync()
        report["backward_seconds"] = time.perf_counter() - start
        groups = parameter_groups(*models)
        gradients = gradient_report(groups)
        errors = validate_setup(groups) + validate_structure(before, groups) + validate_gradients(gradients)
        if errors:
            report["gradients"] = gradients
            raise ValueError("; ".join(errors))
        start = time.perf_counter()
        report["optimizer_step_attempts"] = 1
        optimizer.step()
        report["optimizer_steps"] = 1
        sync()
        report["step_seconds"] = time.perf_counter() - start
        boundary = audit_parameter_diff(before, parameter_groups(*models), gradients)
        report["optimizer_state_finite"] = all(
            torch.isfinite(value.detach().cpu()).all().item()
            for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))
        if not report["optimizer_state_finite"]:
            boundary["status"] = "FAIL"
            boundary["errors"].append("Nonfinite optimizer state")
        report["parameter_audit"] = boundary
        report["errors"] = boundary["errors"]
        report["status"] = boundary["status"]
    except (RuntimeError, ValueError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report
