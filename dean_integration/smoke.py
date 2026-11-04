"""Execute selected ORIGINAL AST nodes from verified external source on CPU.

No upstream source is stored or redistributed here. This is a component fixture,
not a replacement learner or proof of end-to-end training correctness.
"""
import ast
import copy
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

from .external import verify_checkout


def execute_nodes(tree, first, last, namespace):
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)
             and node.lineno >= first and node.end_lineno <= last]
    # Retain only outermost selected statements, preserving official ordering.
    selected = [node for node in nodes if not any(
        other is not node and other.lineno <= node.lineno and other.end_lineno >= node.end_lineno
        for other in nodes)]
    if not selected:
        raise ValueError("Pinned fixture range no longer contains statements")
    exec(compile(ast.Module(body=sorted(selected, key=lambda n: n.lineno), type_ignores=[]),
                 "<external-dean-fixture>", "exec"), namespace)


def run_smoke(checkout):
    checkout = verify_checkout(checkout)
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from sklearn.mixture import GaussianMixture

    torch.manual_seed(1)
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("dean_external_pet", checkout / "net/pet_modules.py")
    pet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pet)
    head_tree = ast.parse((checkout / "net/Decoder.py").read_text(encoding="utf-8"))
    head = next(n for n in head_tree.body if isinstance(n, ast.ClassDef) and n.name == "Head")
    ns = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[head], type_ignores=[]), "<external-dean-head>", "exec"), ns)
    offline = ns["Head"](8, 2)
    online = copy.deepcopy(offline)
    online.update(4)
    assert torch.equal(offline.head.weight, online.head.weight[:2])
    features = torch.randn(16, 8)
    assert online(features).shape == (16, 4)
    # LoRA module import/forward uses a tiny local tensor; no ViT/download.
    lora = pet.KVLoRA(in_features=8, out_features=8, rank=2)
    assert lora(module=nn.Linear(8, 24), input=torch.randn(2, 3, 8)).shape == (2, 3, 24)
    tree = ast.parse((checkout / "Ours_train_online.py").read_text(encoding="utf-8"))
    context = {"torch": torch, "np": np, "F": F, "GaussianMixture": GaussianMixture,
               "args": SimpleNamespace(seed=1, n_replay=5, use_VFA=True, energy_hp=1.),
               "prev_logits": torch.cat((torch.ones(8, 2) * 8, torch.zeros(8, 2))),
               "feats": features, "idx_unseen": np.arange(16)}
    execute_nodes(tree, 472, 488, context)
    assert len(context["idx_o"]) == len(context["idx_n"]) == 8
    torch.manual_seed(8)
    execute_nodes(tree, 569, 580, context)
    augmented = context["proliferated_feats"].clone()
    torch.manual_seed(8)
    execute_nodes(tree, 569, 580, context)
    assert torch.equal(augmented, context["proliferated_feats"])
    assert augmented.shape == (96, 8)
    context.update(unlabeled_feats=features, header_now=online, nb_classes=2,
                   on_y=torch.arange(16) % 4)
    # Controlled positive logits keep the unchanged upstream log-ratio finite.
    with torch.no_grad():
        online.head.weight.fill_(.01)
        online.head.bias.fill_(2.)
    execute_nodes(tree, 746, 752, context)
    context["energy_loss"] = context["unlabeled_energy_loss"]
    execute_nodes(tree, 763, 767, context)
    assert torch.isfinite(context["loss"])
    context["loss"].backward()
    saved = io.BytesIO()
    torch.save({"offline": offline.state_dict(), "online": online.state_dict()}, saved)
    saved.seek(0)
    restored = torch.load(saved, map_location="cpu", weights_only=True)
    other = ns["Head"](8, 4)
    other.load_state_dict(restored["online"])
    assert torch.equal(other(features), online(features))
    return {"status": "PASS", "kind": "external CPU component fixtures only",
            "checks": ["PET import/LoRA forward", "offline/online Head forward and expansion",
                       "EGD two-component GMM", "deterministic VFA", "finite official loss/backward",
                       "minimal state save/load"],
            "not_checked": ["full entrypoint imports", "ViT/checkpoint compatibility", "real CUB training",
                            "discovery clustering edge cases", "optimizer resume"]}
